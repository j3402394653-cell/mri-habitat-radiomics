"""Outcome-blind 3-D SLIC units followed by training-fold-only K-means.

The voxel engine's three first-order descriptors per channel are averaged
inside each supervoxel. Habitat features and nested validation stay unchanged.
"""
from __future__ import annotations

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import find_objects, label
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

SLIC_PROTOCOL = {
    "method": "slic_supervoxel_then_fold_locked_kmeans_v1",
    "requested_tumor_supervoxels": 100,
    "spatial_proximity_weight": 10.0,
    "iterations": 10,
    "enforce_connectivity": True,
    "initialization_perturbation": False,
    "input": "tumor-relative Z-score channels; zeros outside valid support",
    "grid_search": "up to 8 widths; closest tumor unit count, without outcomes",
    "mask_policy": "intersect SLIC labels with valid support; separate tumor and shell; split connected components",
    "descriptor": "supervoxel mean of the unchanged intensity/local-mean/local-SD voxel descriptors",
    "training_units": "tumor supervoxels only, at most 250 per training patient",
    "kmeans": "K=3/4/5 and C chosen in saved inner folds; centroids never fit on held-out patients",
    "texture_matrices": False,
}


def connected_units(raw, roi, support):
    """Split mask intersections and keep tumor units distinct from outer shell."""
    result = np.zeros(raw.shape, np.uint32)
    count = 0
    tumor_ids = []
    # Bounding boxes avoid scanning the entire volume for every component.
    boxes = find_objects(raw.astype(np.int32) + 1)
    for old_id, sl in enumerate(boxes):
        if sl is None:
            continue
        matching = raw[sl] == old_id
        for zone, is_tumor in ((roi[sl], True), (support[sl] & ~roi[sl], False)):
            components, n = label(matching & zone)
            target = result[sl]
            occupied = components > 0
            target[occupied] = components[occupied] + count
            if is_tumor:
                tumor_ids.extend(range(count + 1, count + n + 1))
            count += n
    if not np.array_equal(result > 0, support):
        raise ValueError("SLIC units do not cover exactly the valid support")
    return result, np.asarray(tumor_ids, int)


def prepare_profile(p, target=100):
    if not np.allclose(p["spacing"], (1, 1, 1)):
        raise ValueError("This comparison requires isotropic 1-mm inputs")
    roi, support = p["roi"], p["valid"]
    values = np.where(support[..., None], p["z"], 0).astype(np.float32)
    im = sitk.GetImageFromArray(values, isVector=True)
    im.SetSpacing(tuple(p["spacing"])[::-1])
    grid = max(1, int(round((roi.sum() / target) ** (1 / 3))))
    low, high = 1, max(roi.shape)
    trials, best = {}, None
    for _ in range(8):
        if grid in trials:
            break
        raw = sitk.GetArrayFromImage(sitk.SLIC(im, [grid] * 3, 10.0, 10, True, False))
        count = len(np.unique(raw[roi]))
        trials[grid] = count
        error = abs(count - target)
        if best is None or error < best[0]:
            best = (error, grid, raw)
        if count == target:
            break
        if count > target:
            low = grid + 1
        else:
            high = grid - 1
        if low > high:
            break
        grid = (low + high) // 2
    _, grid, raw = best
    units, tumor_ids = connected_units(raw, roi, support)
    flat = units[support].astype(int)
    counts = np.bincount(flat, minlength=int(units.max()) + 1)
    features = np.zeros((len(counts), p["x"].shape[-1]), np.float32)
    for c in range(features.shape[1]):
        sums = np.bincount(flat, weights=p["x"][..., c][support], minlength=len(counts))
        features[1:, c] = sums[1:] / counts[1:]
    if len(tumor_ids) < 5 or not np.isfinite(features).all():
        raise ValueError("Insufficient finite tumor supervoxel descriptors")
    result = {key: value for key, value in p.items() if key != "x"}
    result.update(superlabels=units, super_x=features, tumor_units=tumor_ids)
    tumor_sizes = counts[tumor_ids]
    audit = {
        "tumor_voxels": int(roi.sum()), "support_voxels": int(support.sum()),
        "requested_supervoxels": target, "tumor_supervoxels": len(tumor_ids),
        "all_support_supervoxels": len(counts) - 1, "grid_mm": grid,
        "grid_trials": trials, "minimum_tumor_unit_voxels": int(tumor_sizes.min()),
        "median_tumor_unit_voxels": float(np.median(tumor_sizes)),
        "maximum_tumor_unit_voxels": int(tumor_sizes.max()),
        "tumor_coverage": float((units[roi] > 0).mean()),
        "support_coverage": float((units[support] > 0).mean()),
        "descriptor_dimensions": features.shape[1],
    }
    return result, audit


def fit_partition(profiles, train, modality, k, seed):
    samples = []
    for idx in train:
        p = profiles[int(idx)][modality]
        x = p["super_x"][p["tumor_units"]]
        if len(x) < k:
            raise ValueError("Too few tumor supervoxels for requested K")
        rng = np.random.default_rng(seed + int(idx) * 1009)
        chosen = rng.choice(len(x), size=min(250, len(x)), replace=False)
        samples.append(x[chosen])
    sample = np.concatenate(samples)
    scaler = StandardScaler().fit(sample)
    with threadpool_limits(limits=1):
        cluster = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(scaler.transform(sample))
    order = np.argsort(cluster.cluster_centers_[:, 0], kind="stable")
    remap = np.empty(k, int)
    remap[order] = np.arange(1, k + 1)
    return scaler, cluster, remap


def assign(p, fitted):
    scaler, cluster, remap = fitted
    lookup = np.zeros(len(p["super_x"]), np.uint8)
    with threadpool_limits(limits=1):
        lookup[1:] = remap[cluster.predict(scaler.transform(p["super_x"][1:]))]
    return lookup[p["superlabels"]]


def install():
    import fold_locked_lvi_engine as engine
    import full_comparison_fold as fold
    engine.fit_partition = fit_partition
    engine.assign = assign
    fold.fit_partition = fit_partition
