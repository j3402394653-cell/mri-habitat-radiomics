"""Shared, outcome-independent 1-mm / tumor-ROI Z-score protocol."""
import numpy as np
import SimpleITK as sitk

PROTOCOL_ID = 'isotropic1mm_zscore_firstorder_v1'
PROTOCOL = {
    'id': PROTOCOL_ID,
    'spacing_mm': [1., 1., 1.],
    'image_interpolation': 'linear (valid-weighted for IVIM); each acquisition in its own physical space',
    'mask_interpolation': 'nearest neighbor',
    'normalization': 'per patient/channel, (x - valid tumor mean) / valid tumor SD, ddof=0',
    'clipping': None,
    'ivim_validity': 'native D>0, 0<=f<0.7, NRMSE<=0.15, all finite; nearest-neighbor quality mask; normalize linear interpolation by valid weight',
    'habitat_features': 'first-order intensity statistics and geometric proportions; no texture matrices',
    'whole_tumor_radiomics': 'shape, firstorder, glcm, glrlm, glszm, gldm, ngtdm',
}


def zscore(a, roi):
    """Fit on this patient's valid tumor only; never clip Z-scores."""
    a = np.asarray(a, dtype=float)
    valid = np.isfinite(a)
    reference = a[np.asarray(roi, bool) & valid]
    if reference.size < 2:
        raise ValueError('Insufficient finite tumor intensities for Z-score')
    mean, std = float(reference.mean()), float(reference.std(ddof=0))
    if not np.isfinite(std) or std <= 0:
        raise ValueError('Tumor intensity SD is zero; Z-score undefined')
    result = np.zeros(a.shape, dtype=np.float32)
    result[valid] = ((a[valid] - mean) / std).astype(np.float32)
    return result, {'method': 'zscore', 'mean': mean, 'std': std, 'ddof': 0,
                    'reference_voxels': int(reference.size), 'clipping': None}


def image_like(a, reference):
    result = sitk.GetImageFromArray(a)
    result.CopyInformation(reference)
    return result


def isotropic_reference(mask):
    """Cropped 1-mm grid preserving the source direction and physical origin."""
    coords = np.argwhere(sitk.GetArrayFromImage(mask) > 0)
    if not len(coords):
        raise ValueError('Empty ROI for resampling')
    low = np.maximum(coords.min(0) - 2, 0)[::-1]
    high = np.minimum(coords.max(0) + 3, np.array(mask.GetSize())[::-1])[::-1]
    crop = sitk.RegionOfInterest(mask, list(map(int, high-low)), list(map(int, low)))
    size = np.ceil((np.array(crop.GetSize())-1) * crop.GetSpacing()).astype(int) + 1
    ref = sitk.Image(list(map(int, size)), sitk.sitkFloat32)
    ref.SetSpacing((1., 1., 1.))
    ref.SetOrigin(crop.GetOrigin())
    ref.SetDirection(crop.GetDirection())
    return ref


def resample_valid_maps(maps, tumor, boundary, valid):
    """Interpolate only valid fits, renormalize weights, and map quality by nearest neighbor."""
    union = (sitk.GetArrayFromImage(tumor) > 0) | (sitk.GetArrayFromImage(boundary) > 0)
    ref = isotropic_reference(image_like(union.astype(np.uint8), tumor))
    support = sitk.Resample(image_like(valid.astype(np.float32), tumor), ref,
                            sitk.Transform(), sitk.sitkLinear, 0., sitk.sitkFloat32)
    weights = sitk.GetArrayFromImage(support)
    nearest_valid = sitk.Resample(image_like(valid.astype(np.uint8), tumor), ref,
                                 sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    keep = (sitk.GetArrayFromImage(nearest_valid) > 0) & (weights > 1e-6)
    sampled = []
    for im in maps:
        clean = image_like(np.where(valid, sitk.GetArrayFromImage(im), 0.).astype(np.float32), im)
        out = sitk.Resample(clean, ref, sitk.Transform(), sitk.sitkLinear, 0., sitk.sitkFloat32)
        a = sitk.GetArrayFromImage(out)
        a[keep] /= weights[keep]
        a[~keep] = np.nan
        sampled.append(a)
    masks = [sitk.GetArrayFromImage(sitk.Resample(m, ref, sitk.Transform(),
                 sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)) > 0 for m in (tumor, boundary)]
    return np.stack(sampled, axis=-1), masks[0], masks[1], ref
