"""Patient-local quantitative tertile habitats, independent of outcomes.

This is a new partition method. It does not import the excluded LVI transfer
experiment. T2 uses relative intensity; ADC and IVIM retain original map values.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

PROTOCOL = {
    "method": "patient_local_tertile_quantitative_habitats_v1",
    "quantiles": [1 / 3, 2 / 3],
    "partition_channels": {"T2": [0], "ADC": [0], "IVIM": [0, 1]},
    "quantitative_values": "ADC/D/f recovered from unclipped Z-score and saved patient mean/SD",
    "T2_values": "patient tumor Z-score; no cross-patient raw T2 comparison",
    "periphery": "automatic inner 2-mm band of each current mask",
    "texture": "symmetric 16-level GLCM, fixed Z-score range [-3,3], 3 axial offsets of 1 mm",
    "perturbation": "refit patient quantiles independently on original, 1-mm eroded and dilated masks",
    "pooled_patient_partition_fit": False,
    "whole_tumor_features": False,
    "clinical_features": False,
    "shape_transfer_features": False,
}


def original_values(p, mod):
    z = np.asarray(p["z"], float)
    if mod == "T2":
        return z
    means = np.asarray([v["mean"] for v in p["normalization"]])
    stds = np.asarray([v["std"] for v in p["normalization"]])
    if any(v.get("clipping") is not None for v in p["normalization"]):
        raise ValueError("Original map values cannot be recovered from clipped Z-scores")
    return z * stds + means


def neighbor_pairs(a, mask):
    for axis in range(3):
        left, right = [slice(None)] * 3, [slice(None)] * 3
        left[axis], right[axis] = slice(None, -1), slice(1, None)
        left, right = tuple(left), tuple(right)
        valid = mask[left] & mask[right]
        yield a[left][valid], a[right][valid]


def glcm_metrics(levels, region):
    counts = np.zeros(256, float)
    for a, b in neighbor_pairs(levels, region):
        counts += np.bincount(a.astype(int) * 16 + b, minlength=256)
    matrix = counts.reshape(16, 16)
    matrix += matrix.T.copy()
    if matrix.sum() < 40:
        return [np.nan] * 5
    matrix /= matrix.sum()
    i, j = np.indices((16, 16))
    nz = matrix > 0
    mi, mj = np.sum(i * matrix), np.sum(j * matrix)
    vi, vj = np.sum((i - mi) ** 2 * matrix), np.sum((j - mj) ** 2 * matrix)
    return [float(np.sum((i - j) ** 2 * matrix)),
            float(np.sum(matrix / (1 + (i - j) ** 2))),
            float(np.sum(matrix ** 2)), float(-np.sum(matrix[nz] * np.log2(matrix[nz]))),
            float(np.sum((i - mi) * (j - mj) * matrix) / np.sqrt(vi * vj)) if vi * vj > 1e-16 else 0.]


def one_profile(p, mod, mask):
    mask = np.asarray(mask, bool)
    raw, z = original_values(p, mod), np.asarray(p["z"], float)
    if mask.sum() < 20:
        raise ValueError("Insufficient valid voxels in perturbed mask")
    shell = mask & (distance_transform_edt(mask, sampling=p["spacing"]) <= 2.)
    core = mask & ~shell
    values, metadata = [], []

    def add(value, name, panel):
        values.append(float(value))
        metadata.append({"name": mod + "__" + name, "modality": mod, "panel": panel})

    channels = ["T2z"] if mod == "T2" else ["ADCraw"] if mod == "ADC" else ["D", "f"]
    levels = np.clip(np.floor((z + 3) * (16 / 6)), 0, 15).astype(np.uint8)
    for driver in ([0, 1] if mod == "IVIM" else [0]):
        cuts = np.quantile(z[..., driver][mask], [1 / 3, 2 / 3])
        labels = np.searchsorted(cuts, z[..., driver], side="right") + 1
        stem = "by" + channels[driver] + "_"
        medians = {}
        for h in range(1, 4):
            region = mask & (labels == h)
            n = region.sum()
            add(n / mask.sum(), stem + f"H{h}_Fraction", "quantitative")
            shell_fraction = (region & shell).sum() / max(shell.sum(), 1)
            core_fraction = (region & core).sum() / core.sum() if core.any() else np.nan
            add(shell_fraction, stem + f"H{h}_PeripheryFraction", "spatial_texture")
            add(shell_fraction - core_fraction, stem + f"H{h}_PeripheryCoreDifference", "spatial_texture")
            for c, channel in enumerate(channels):
                a = raw[..., c][region]
                if n >= 12:
                    p10, q25, median, q75, p90 = np.quantile(a, [.1, .25, .5, .75, .9])
                    stats = [np.mean(a), median, p10, p90, q75 - q25, np.std(a)]
                else:
                    stats = [np.nan] * 6
                medians[h, c] = stats[1]
                for value, suffix in zip(stats, ["Mean", "Median", "P10", "P90", "IQR", "SD"]):
                    add(value, stem + f"H{h}_{channel}_{suffix}", "quantitative")
                for value, suffix in zip(glcm_metrics(levels[..., c], region),
                                         ["Contrast", "Homogeneity", "Energy", "Entropy", "Correlation"]):
                    add(value, stem + f"H{h}_{channel}_GLCM{suffix}", "spatial_texture")
        for c, channel in enumerate(channels):
            add(medians[3, c] - medians[1, c], stem + channel + "_HighLowMedianDifference", "quantitative")
        contacts = np.zeros((3, 3), float)
        for a, b in neighbor_pairs(labels, mask):
            contacts += np.bincount((a.astype(int) - 1) * 3 + b - 1, minlength=9).reshape(3, 3)
        contacts += contacts.T.copy()
        if contacts.sum():
            contacts /= contacts.sum()
        add(1 - np.trace(contacts), stem + "DifferentHabitatContact", "spatial_texture")
        for a, b in [(0, 1), (0, 2), (1, 2)]:
            add(contacts[a, b] + contacts[b, a], stem + f"H{a+1}H{b+1}Contact", "spatial_texture")
    return np.asarray(values), metadata


def extract(profiles, progress=None):
    rows = {"base": [], "erode": [], "dilate": []}
    names = None
    reconstruction_error = []
    for idx, patient in enumerate(profiles):
        assembled = {key: [] for key in rows}
        current_names = []
        for mod in ["T2", "ADC", "IVIM"]:
            p = patient[mod]
            assert np.allclose(p["spacing"], [1., 1., 1.])
            for key, mask_key in [("base", "roi"), ("erode", "erode"), ("dilate", "dilate")]:
                v, meta = one_profile(p, mod, p[mask_key])
                assembled[key].extend(v)
                if key == "base":
                    current_names.extend(meta)
            if mod != "T2":
                restored = original_values(p, mod)[p["roi"]]
                for c, norm in enumerate(p["normalization"]):
                    error = abs(restored[:, c].mean() - norm["mean"]) / max(abs(norm["mean"]), 1e-10)
                    reconstruction_error.append(error)
                    assert error < 1e-5
        if names is None:
            names = current_names
        assert current_names == names
        for key in rows:
            rows[key].append(assembled[key])
        if progress and (idx + 1) % 10 == 0:
            progress(idx + 1, len(profiles))
    return {**{key: np.asarray(value, float) for key, value in rows.items()},
            "metadata": names, "protocol": PROTOCOL,
            "max_relative_reconstruction_error": float(max(reconstruction_error))}
