"""Paper-informed T2/ADC adaptation of Chen et al., DOI 10.1186/s13058-024-01921-7.

Independent sequence habitats, first-order descriptors, SLIC and CH audit.
This is an adaptation, not an exact reproduction of the DWI/T1C experiment.
"""
import hashlib
import json
import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from image_preprocessing import zscore, isotropic_reference, PROTOCOL
from run_every_patient_whole_tumor import (
    read_nifti, find_t2, find_adc, find_sequence_mask, same_geometry, resample_to_t2)

METHOD = 'slic_firstorder_1mm_zscore_v3'
RESULTS_DIR = 'habitat_firstorder_zscore_results'

S1 = {
    'firstorder': ['Entropy', 'MeanAbsoluteDeviation', 'Median'],
}
S1_NAMES = [f'original_{group}_{name}' for group, names in S1.items() for name in names]


def image(a, reference):
    result = sitk.GetImageFromArray(a)
    result.CopyInformation(reference)
    return result


def save_nifti(im, path):
    affine = np.eye(4)
    affine[:3, :3] = np.array(im.GetDirection()).reshape(3, 3) @ np.diag(im.GetSpacing())
    affine[:3, 3] = im.GetOrigin()
    affine = np.diag([-1., -1., 1., 1.]) @ affine
    result = nib.Nifti1Image(sitk.GetArrayFromImage(im).transpose(2, 1, 0), affine)
    result.header.set_xyzt_units('mm')
    nib.save(result, str(path))


def isotropic(t2, adc, mask):
    """Crop, then sample a shared physical 1-mm grid; no registration is inferred."""
    ref = isotropic_reference(mask)
    ims = (t2, adc, mask)
    result = [sitk.Resample(im, ref, sitk.Transform(), sitk.sitkLinear,
                          float('nan'), sitk.sitkFloat32) for im in ims[:2]]
    result.append(sitk.Resample(ims[2], ref, sitk.Transform(), sitk.sitkNearestNeighbor,
                              0, sitk.sitkUInt8))
    return result


def extractor(supervoxels=False, whole_tumor=False):
    ex = featureextractor.RadiomicsFeatureExtractor(binWidth=.25, normalize=False,
                                                   force2D=False, label=1)
    ex.disableAllFeatures()
    if supervoxels:
        ex.enableFeaturesByName(**S1)
    else:
        groups = (('firstorder', 'glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm', 'shape')
                  if whole_tumor else ('firstorder', 'shape'))
        for group in groups:
            ex.enableFeatureClassByName(group)
    return ex


def prepare_sequence(im, mask, roi_policy='strict'):
    """Resample the explicitly assigned contour within its own acquisition."""
    if roi_policy not in ('strict', 'finite'):
        raise ValueError('Use strict or finite per-sequence ROI policy; common ROI is no longer supported')
    native = sitk.GetArrayFromImage(mask) > 0
    original = int(native.sum())
    if not original:
        raise ValueError('Empty sequence-specific tumor ROI')
    # NIfTI contours can store the same physical slices in reversed order.
    # Check physical coverage before nearest-neighbour mapping, never copy headers.
    xyz = np.argwhere(native)[:, ::-1]
    physical = (xyz * mask.GetSpacing()) @ np.array(mask.GetDirection()).reshape(3, 3).T + mask.GetOrigin()
    indices = (physical - im.GetOrigin()) @ np.array(im.GetDirection()).reshape(3, 3) / im.GetSpacing()
    inside = np.all((indices >= -.5-1e-4) & (indices <= np.array(im.GetSize())-.5+1e-4), axis=1)
    if not inside.all():
        raise ValueError('Sequence contour extends outside its image FOV; verify image/mask pairing')
    mapped = not same_geometry(im, mask)
    if mapped:
        mask = resample_to_t2(mask, im, True)
        native = sitk.GetArrayFromImage(mask) > 0
        if not native.any():
            raise ValueError('Empty contour after within-sequence grid mapping')
    valid = native & np.isfinite(sitk.GetArrayFromImage(im))
    excluded = int(np.sum(native & ~valid))
    if excluded and roi_policy == 'strict':
        raise ValueError('Nonfinite intensities in sequence-specific ROI')
    sampled, _, sampled_mask = isotropic(im, im, image(valid.astype(np.uint8), im))
    roi = sitk.GetArrayFromImage(sampled_mask) > 0
    finite = np.isfinite(sitk.GetArrayFromImage(sampled))
    missing = int(np.sum(roi & ~finite))
    if missing and roi_policy == 'strict':
        raise ValueError('Nonfinite ROI after isotropic resampling')
    roi &= finite
    if not roi.any():
        raise ValueError('Empty resampled sequence ROI')
    return sampled, roi, {'original_roi_voxels': original, 'mask_resampled_within_sequence': mapped,
        'mapped_roi_voxels': int(native.sum()),
        'native_excluded_voxels': excluded, 'resampling_excluded_voxels': missing,
        'resampled_roi_voxels': int(roi.sum()), 'spacing_mm': sampled.GetSpacing(),
        'origin_mm': sampled.GetOrigin(), 'direction': sampled.GetDirection()}


def cluster_regions(features, means, k=3):
    if not 3 <= k <= 10:
        raise ValueError('k must be between 3 and 10')
    x = StandardScaler().fit_transform(features)
    if not np.isfinite(x).all() or len(x) <= k or len(np.unique(x, axis=0)) < k:
        raise ValueError('Insufficient distinct supervoxel descriptors for clustering')
    scores = {}; solutions = {}
    with threadpool_limits(limits=1):
        for candidate in range(3, min(10, len(x)-1, len(np.unique(x, axis=0)))+1):
            labels = KMeans(n_clusters=candidate, n_init=20, random_state=42).fit_predict(x)
            if len(np.unique(labels)) != candidate:
                continue
            scores[str(candidate)] = float(calinski_harabasz_score(x, labels))
            solutions[candidate] = labels
    raw = solutions[k]
    # Sequence-specific rank, not a histologically established tissue class.
    order = sorted(range(k), key=lambda c: (float(np.mean(means[raw == c])), c))
    mapping = np.zeros(k, np.uint8)
    for rank, old in enumerate(order, 1):
        mapping[old] = rank
    return mapping[raw], scores


def has_texture_neighbors(binary):
    """GLCM distance=1 requires at least one 26-neighbor voxel pair."""
    from scipy.ndimage import convolve
    kernel=np.ones((3,3,3),dtype=np.uint8);kernel[1,1,1]=0
    return bool(np.any(binary & (convolve(binary.astype(np.uint8),kernel,mode='constant',cval=0)>0)))


def generate_habitats(im, roi, k=3, target=100):
    """ITK grid SLIC on cropped image, masked afterwards; actual count is audited."""
    values = sitk.GetArrayFromImage(im)
    if target < 4 or roi.sum() < 40:
        raise ValueError('Insufficient tumor voxels or invalid supervoxel target')
    grid = max(1, int(round((roi.sum()/target)**(1/3))))
    slic_input = image(np.where(roi, values, 0).astype(np.float32), im)
    # Mask intersections add boundary fragments. Search grid widths using only ROI
    # geometry/intensity, so the actual count stays close to the requested target.
    trials = {}; best = None
    low, high = 1, max(values.shape)
    for _ in range(8):
        if grid in trials: break
        candidate = sitk.GetArrayFromImage(sitk.SLIC(slic_input, [grid]*3, 10., 10, True, False))
        count = len(np.unique(candidate[roi])); trials[grid] = count
        error = abs(count-target)
        if best is None or error < best[0]: best = (error, grid, candidate)
        if count == target: break
        if count > target: low = grid+1
        else: high = grid-1
        if low > high: break
        grid = (low+high)//2
    _, grid, raw = best
    superlabels = np.zeros(roi.shape, np.uint32)
    for index, old in enumerate(np.unique(raw[roi]), 1):
        superlabels[(raw == old) & roi] = index
    ex = extractor(True); descriptors = []; means = []; ids = []; omitted = []
    for index in np.unique(superlabels[roi]):
        binary = superlabels == index
        extent = np.ptp(np.argwhere(binary), axis=0)
        if binary.sum() < 2 or np.count_nonzero(extent) < 2:
            omitted.append(int(index)); continue
        result = ex.execute(im, image(binary.astype(np.uint8), im))
        row = [float(result[name]) for name in S1_NAMES]
        if not np.isfinite(row).all():
            raise ValueError('Nonfinite first-order supervoxel descriptor')
        descriptors.append(row); means.append(float(values[binary].mean())); ids.append(int(index))
    if len(ids) <= k:
        raise ValueError('Too few extractable supervoxels')
    labels, scores = cluster_regions(np.asarray(descriptors), np.asarray(means), k)
    habitat = np.zeros(roi.shape, np.uint8)
    for index, label in zip(ids, labels):
        habitat[superlabels == index] = label
    # Tiny regions get the spatially nearest valid habitat, never an outcome-based label.
    if omitted:
        from scipy.ndimage import distance_transform_edt
        nearest = distance_transform_edt(habitat == 0, return_distances=False, return_indices=True)
        missing = roi & (habitat == 0)
        habitat[missing] = habitat[tuple(nearest[:, missing])]
    if not np.array_equal(habitat > 0, roi):
        raise ValueError('Habitat coverage differs from tumor ROI')
    return image(habitat, im), image(superlabels, im), {
        'requested_supervoxels': target, 'actual_supervoxels': int(superlabels.max()),
        'extractable_supervoxels': len(ids), 'tiny_supervoxels_spatially_assigned': omitted,
        'feature_policy': 'first-order supervoxel descriptors; no texture matrices',
        'slic_grid': grid, 'slic_grid_counts': trials, 'ch_scores': scores, 'ch_best_k': int(max(scores, key=scores.get)),
        'used_k': k, 'k_policy': 'prespecified; CH diagnostic only', 'descriptor_names': S1_NAMES}


def extract_patient(patient_dir, output_dir=None, k=3, roi_policy='strict', t2_mask=None, adc_mask=None):
    logging.getLogger('radiomics').setLevel(logging.ERROR)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    p = Path(patient_dir).resolve()
    out = Path(output_dir) if output_dir else p.parent/RESULTS_DIR/p.name
    if (out/'complete.json').exists():
        raise ValueError('Completed output exists; choose a new output directory')
    out.mkdir(parents=True, exist_ok=True)
    # A failed rerun must not retain a success marker or old feature table.
    for name in ('complete.json', 'features.csv', 'qc.json', 'T2_habitats.nii.gz',
                 'ADC_habitats.nii.gz', 'T2_supervoxels.nii.gz', 'ADC_supervoxels.nii.gz',
                 'T2_roi.nii.gz', 'ADC_roi.nii.gz'):
        (out/name).unlink(missing_ok=True)
    paths = dict(t2=Path(find_t2(str(p))), adc=Path(find_adc(str(p))),
                 t2_mask=Path(t2_mask or find_sequence_mask(str(p), 'T2')),
                 adc_mask=Path(adc_mask or find_sequence_mask(str(p), 'ADC')))
    if paths['t2_mask'].resolve() == paths['adc_mask'].resolve():
        raise ValueError('T2 and ADC require separate contour files')
    row = {'PatientID': p.name}; metadata = {}
    for channel in ('T2', 'ADC'):
        key = channel.lower()
        im, roi, spatial_qc = prepare_sequence(read_nifti(paths[key]),
            read_nifti(paths[key+'_mask'], True), roi_policy)
        normalized, norm = zscore(sitk.GetArrayFromImage(im), roi)
        im = image(normalized, im)
        habitats, supervoxels, qc = generate_habitats(im, roi, k)
        la = sitk.GetArrayFromImage(habitats)
        habitat_ex = extractor(); whole_ex = extractor(whole_tumor=True); unavailable = []
        for region, binary in [('Whole', roi)]+[(f'H{j}', la == j) for j in range(1, k+1)]:
            prefix = f'Paper_{channel}_{region}_'
            count = int(binary.sum())
            row[prefix+'Fraction'] = count/float(roi.sum())
            if count < 20 or np.count_nonzero(np.ptp(np.argwhere(binary), axis=0)) < 2:
                unavailable.append(region)
                for name in [n for n in row if n.startswith(f'Paper_{channel}_Whole_original_')
                             and ('_original_firstorder_' in n or '_original_shape_' in n)]:
                    row[name.replace(f'Paper_{channel}_Whole_', prefix)] = np.nan
                continue
            ex = whole_ex if region == 'Whole' else habitat_ex
            result = ex.execute(im, image(binary.astype(np.uint8), im))
            for name, value in result.items():
                if name.startswith('original_'):
                    value = float(value)
                    if not np.isfinite(value): raise ValueError(f'Nonfinite feature: {name}')
                    row[prefix+name] = value
        save_nifti(habitats, out/f'{channel}_habitats.nii.gz')
        save_nifti(supervoxels, out/f'{channel}_supervoxels.nii.gz')
        save_nifti(image(roi.astype(np.uint8), im), out/f'{channel}_roi.nii.gz')
        metadata[channel] = {**qc, **spatial_qc, 'normalization': norm, 'unavailable_regions': unavailable}
    qc = {'method': METHOD, 'k': k, 'roi_policy': roi_policy, 'preprocessing_protocol': PROTOCOL,
          'spatial_policy': 'independent sequence images and contours; patient-level feature fusion only',
          'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'software': {'SimpleITK': sitk.Version_VersionString()},
          'sequences': metadata,
          'sources': {name: {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                      for name, path in paths.items()},
          'adaptations': ['T2/ADC instead of DWI/T1C', 'patient-local tumor Z-score',
                         'first-order descriptors replace historical S1 texture descriptors',
                         'texture features only in Whole region; habitats use firstorder and shape',
                         'ITK grid SLIC with post-mask, approximate 100 regions',
                         'original-image feature classes only; no unspecified filter bank',
                         'sequence-specific ascending mean intensity labels; no tissue interpretation']}
    (out/'qc.json').write_text(json.dumps(qc, indent=2, ensure_ascii=False), encoding='utf-8')
    pd.DataFrame([row]).to_csv(out/'features.csv', index=False, encoding='utf-8-sig')
    (out/'complete.json').write_text(json.dumps({'method': qc['method'], 'feature_count': len(row)-1}), encoding='utf-8')
    return row
