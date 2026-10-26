"""Outcome-blind compact K3 features for physical 2/3-mm contour perturbations."""
import logging
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt
import radiomics
from radiomics import featureextractor
import normalization_habitat_features_20261012 as prior
import k345_habitat_features_20261013 as geometry
import habitat_stable_views_20261013 as compact
from quantile_habitat_features import original_values

NORMS = ('PercentileZ', 'RobustZ')
MODS = ('T2', 'ADC', 'IVIM')
RADII = (1., 2., 3.)


def extended_profile(entry, mod, old):
    """Resample original sources onto a padded, otherwise identical 1-mm grid.

    The old profile's valid mask was truncated to its 1-mm support. It cannot
    define a real 2/3-mm expansion. Re-read full native quality maps instead.
    Normalization and original tumor remain identical to the old profile.
    """
    import fold_locked_lvi_engine as engine
    import image_preprocessing as prep
    import paper_habitats as paper
    reference = prep.isotropic_reference
    array_profile = engine._array_profile
    captured = {}; expanded = [False]

    def padded(mask):
        r = reference(mask)
        padding = 4
        out = sitk.Image([int(s + 2*padding) for s in r.GetSize()], sitk.sitkFloat32)
        out.SetSpacing(r.GetSpacing()); out.SetDirection(r.GetDirection())
        out.SetOrigin(r.TransformIndexToPhysicalPoint((-padding,)*3))
        return out

    def capture(values, roi, boundary, spacing, local_size):
        p = {'spacing':tuple(spacing)} if expanded[0] else array_profile(values, roi, boundary, spacing, local_size)
        a = np.asarray(values)
        if a.ndim==3:a=a[...,None]
        p['full_valid'] = np.all(np.isfinite(a), axis=-1) if a.ndim == 4 else np.isfinite(a)
        p['geometric_roi'] = np.asarray(roi, bool)
        p['sampling_values'] = a
        captured['p'] = p
        return p

    try:
        engine._array_profile = capture
        source = engine._ivim_profile(entry) if mod == 'IVIM' else engine._sequence_profile(entry, {}, mod.lower())
        expanded[0] = True
        prep.isotropic_reference = padded; paper.isotropic_reference = padded
        p = engine._ivim_profile(entry) if mod == 'IVIM' else engine._sequence_profile(entry, {}, mod.lower())
    finally:
        prep.isotropic_reference = reference; paper.isotropic_reference = reference
        engine._array_profile = array_profile
    # Audit originalsource onitsORIGINALgrid,including historicalFOV edgebehavior.
    old_support = source['valid'] | source['roi'] | source['boundary'] | source['erode'] | source['dilate']
    coords = np.argwhere(old_support)
    low, high = coords.min(0), coords.max(0)+1
    box = tuple(slice(int(a), int(b)) for a, b in zip(low, high))
    if tuple(high-low) != old['roi'].shape:
        # Additional20 profiles kept the uncropped original reference grid.
        box = tuple(slice(None) for _ in range(3))
    assert source['roi'][box].shape == old['roi'].shape, (entry['AnalysisID'], mod)
    for name in ('roi', 'erode', 'dilate', 'valid'):
        np.testing.assert_array_equal(source[name][box], old[name], err_msg=entry['AnalysisID']+' '+mod+' '+name)
    np.testing.assert_array_equal(source['z'][box][old['valid']], old['z'][old['valid']])
    assert source['normalization'] == old['normalization']
    # Origin arithmeticcanperturb nearest-neighbor half-gridties. Pasteevery
    # oldcore sample exactly,usingnewnative resampling onlyoutside thatcore.
    shape=tuple(int(n)+8 for n in source['roi'].shape)
    assert p['full_valid'].shape==shape
    core=tuple(slice(4,int(n)+4) for n in source['roi'].shape)
    p['full_valid'][core]=source['full_valid']
    p['geometric_roi']=np.zeros(shape,bool);p['geometric_roi'][core]=source['geometric_roi']
    z=np.zeros(shape+(source['z'].shape[-1],),np.float32)
    for c,normal in enumerate(source['normalization']):
        a=p['sampling_values'][...,c];valid=p['full_valid']
        z[...,c][valid]=((a[valid]-normal['mean'])/normal['std']).astype(np.float32)
    z[core]=source['z'];p['z']=z;p['normalization']=source['normalization']
    for name in ('roi','erode','dilate','valid','boundary'):
        p[name]=np.zeros(shape,bool);p[name][core]=source[name]
    del p['sampling_values']
    p['old_box']=box
    return p


def masks(p, radius):
    tumor = p['geometric_roi']; valid = p['full_valid']
    inside = distance_transform_edt(tumor, sampling=p['spacing'])
    outside = distance_transform_edt(~tumor, sampling=p['spacing'])
    return {'base': tumor & valid,
            'erode': tumor & (inside > radius) & valid,
            'dilate': (tumor | (outside <= radius)) & valid}


def setup():
    radiomics.logger.setLevel(logging.ERROR)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    names = sorted('original_'+s for s in compact.METRICS)
    ex = featureextractor.RadiomicsFeatureExtractor(binWidth=.25, normalize=False,
        label=1, distances=[1], additionalInfo=False, minimumROISize=12)
    ex.disableAllFeatures()
    groups = {}
    for suffix in compact.METRICS:
        group, feature = suffix.split('_', 1)
        groups.setdefault(group, []).append(feature)
    ex.enableFeaturesByName(**groups)
    metadata = []
    for mod in MODS:
        metadata.extend(m for m in geometry.layout(mod, names, 3)
                        if not m['name'].endswith('_TumorFraction'))
    assert len(metadata) == 321
    return names, ex, metadata


def measure(p, mod, mask, normalized, raw, images, names, ex):
    meta = geometry.layout(mod, names, 3); result = []; audit = []
    source = (p['z'] if mod == 'IVIM' else raw) if normalized['name'] == 'RobustZ' else normalized['values']
    labels = prior.partition(source, mask, mod)
    assert np.array_equal(labels > 0, mask) or mask.sum() < 12
    for c, channel in enumerate(('D', 'f') if mod == 'IVIM' else (mod,)):
        for h in range(1, 4):
            region = mask & (labels == h); error = None
            try:
                if region.sum() < 12: raise ValueError('habitat fewer than12voxels')
                roi = sitk.GetImageFromArray(region.astype(np.uint8)); roi.CopyInformation(images[c])
                measured = ex.execute(images[c], roi)
                values = [float(measured[n]) for n in names]
            except (ValueError, RuntimeError) as exc:
                values = [np.nan]*len(names); error = type(exc).__name__+': '+str(exc)
            result.extend(values)
            audit.append({'Channel': channel, 'Habitat': h, 'Voxels': int(region.sum()), 'Error': error})
        v, n = geometry.descriptors(normalized['values'][..., c] if mod == 'T2' else raw[..., c], labels, mask, p['spacing'], 3)
        result.extend(v[j] for j, s in enumerate(n) if c == 0 or any(s.endswith('_'+q) for q in ('Mean','SD','P10','Median','P90','IQR')))
    v, _ = geometry.invariant_centroids(labels, mask, p['spacing'], 3); result.extend(v)
    assert len(meta) == len(result)
    keep = [j for j, m in enumerate(meta) if not m['name'].endswith('_TumorFraction')]
    return [result[j] for j in keep], audit


def extract_one(task):
    index, aid, entry, old, check_reference = task
    names, ex, metadata = setup()
    rows = {(n, r, kind): [] for n in NORMS for r in RADII for kind in ('erode','dilate')}
    references = {(n, kind): [] for n in NORMS for kind in ('base','erode','dilate')}
    base_values = {n: [] for n in NORMS}
    audit, texture, normalization = [], [], []
    for mod in MODS:
        p = extended_profile(entry, mod, old[mod]); raw = original_values(p, mod)
        rois = {r: masks(p, r) for r in RADII}
        for r in RADII:
            for kind, mask in rois[r].items():
                if kind == 'base' and r != 1.: continue
                touches = sum(int(np.take(mask, edge, axis=ax).sum()) for ax in range(3) for edge in (0,-1))
                assert touches == 0, (aid, mod, r, kind, 'extended reference too small')
                audit.append({'AnalysisID': aid, 'Modality': mod, 'RadiusMM': r, 'Variant': kind,
                    'Voxels': int(mask.sum()), 'BaseVoxels': int(p['roi'].sum()),
                    'AddedVsBase': int((mask & ~p['roi']).sum()), 'RemovedVsBase': int((p['roi'] & ~mask).sum()),
                    'ValidFractionOfGeometricRegion': float(mask.sum()/max(1, (p['geometric_roi'] & (distance_transform_edt(p['geometric_roi'],sampling=p['spacing'])>r)).sum() if kind=='erode' else (p['geometric_roi'] | (distance_transform_edt(~p['geometric_roi'],sampling=p['spacing'])<=r)).sum() if kind=='dilate' else p['geometric_roi'].sum())),
                    'OutsideOldOneMMValidSupport': int((mask & ~p['valid']).sum()), 'ReferenceEdgeVoxels': touches})
        for norm in NORMS:
            arrays = []
            for c in range(raw.shape[-1]):
                v, info = prior.normalize(raw[..., c], p['roi'], norm, p['z'][..., c]); arrays.append(v)
                normalization.append({'AnalysisID': aid,'Modality': mod,'Channel': c, **info})
            values = np.stack(arrays, axis=-1); images = []
            for c in range(values.shape[-1]):
                im = sitk.GetImageFromArray(values[..., c]); im.SetSpacing(tuple(p['spacing'])[::-1]); images.append(im)
            n = {'name': norm, 'values': values}
            v, _ = measure(p, mod, rois[1.]['base'], n, raw, images, names, ex)
            base_values[norm].extend(v)
            for r in RADII:
                for kind in ('erode','dilate'):
                    v, ta = measure(p, mod, rois[r][kind], n, raw, images, names, ex)
                    rows[(norm, r, kind)].extend(v)
                    texture.extend({'AnalysisID': aid,'Modality': mod,'RadiusMM': r,'Normalization':norm,'Variant':kind, **a} for a in ta)
            if check_reference:
                op = old[mod]; oraw = original_values(op, mod); ov = []
                for c in range(oraw.shape[-1]):
                    vv, _ = prior.normalize(oraw[...,c], op['roi'], norm, op['z'][...,c]); ov.append(vv)
                ov = np.stack(ov,axis=-1); oi = []
                for c in range(ov.shape[-1]):
                    im = sitk.GetImageFromArray(ov[...,c]); im.SetSpacing(tuple(op['spacing'])[::-1]); oi.append(im)
                for kind in ('base','erode','dilate'):
                    mask = op['roi' if kind=='base' else kind] & op['valid']
                    v, _ = measure(op, mod, mask, {'name':norm,'values':ov}, oraw, oi, names, ex)
                    references[(norm, kind)].extend(v)
    assert all(len(v)==321 for v in rows.values())
    return {'index': index, 'metadata': metadata, 'values': rows, 'base_values':base_values,'references': references if check_reference else {},
            'geometry_audit': audit,'texture_audit':texture,'normalization_audit':normalization,
            'reconstructed_old_profile_verified':True}
