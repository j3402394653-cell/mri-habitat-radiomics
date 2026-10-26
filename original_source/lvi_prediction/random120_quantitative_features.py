"""Quantitative SLIC habitats with independently recomputed contour shells.

No outcomes, shape-transfer features, or fitted classifiers are used here.
Patient normalization is fixed; contour checks measure conditional sampling
repeatability, not full resegmentation/renormalization reproducibility.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt
from quantile_habitat_features import original_values, one_profile
import slic_habitat_backend as slic


def shell(mask, spacing):
    return mask & (distance_transform_edt(mask, sampling=spacing) <= 2.)


def slic_profile(p, mod, fitted, mask):
    labels = slic.assign(p, fitted)
    raw = original_values(p, mod)
    periphery = shell(mask, p["spacing"])
    core = mask & ~periphery
    channels = ["T2z"] if mod == "T2" else ["ADCraw"] if mod == "ADC" else ["Draw", "fraw"]
    values, metadata, medians = [], [], {}
    fractions = {}

    def add(value, name):
        values.append(float(value))
        metadata.append({"name": mod+"__"+name, "modality": mod, "panel": "quantitative_slic"})

    for h in range(1, 4):
        region = mask & (labels == h)
        fraction = float(region.sum()/max(mask.sum(), 1))
        pfraction = float((region & periphery).sum()/max(periphery.sum(), 1))
        cfraction = float((region & core).sum()/core.sum()) if core.any() else np.nan
        fractions[h] = (fraction, pfraction)
        add(fraction, f"H{h}_Fraction")
        add(pfraction, f"H{h}_PeripheryFraction")
        add(pfraction-cfraction, f"H{h}_PeripheryCoreDifference")
        for c, channel in enumerate(channels):
            a = raw[..., c][region]
            if len(a) >= 12:
                p10, q25, median, q75, p90 = np.quantile(a, [.1, .25, .5, .75, .9])
                stats = [np.mean(a), median, p10, p90, q75-q25, np.std(a)]
            else:
                stats = [np.nan]*6
            medians[h, c] = stats[1]
            for value, suffix in zip(stats, ["Mean", "Median", "P10", "P90", "IQR", "SD"]):
                add(value, f"H{h}_{channel}_{suffix}")
    for j, fraction in enumerate(["Fraction", "PeripheryFraction"]):
        for h in [1, 2]:
            add(np.log((fractions[h][j]+.005)/(fractions[3][j]+.005)), f"LogH{h}H3_{fraction}")
    for c, channel in enumerate(channels):
        add(medians[3, c]-medians[1, c], f"HighLowMedianDifference_{channel}")
    return np.asarray(values), metadata


def tables(profiles, partitions, indices):
    rows = {"base": [], "erode": [], "dilate": []}
    metadata = None
    for i in indices:
        current = {kind: [] for kind in rows}
        meta = []
        for mod in ["T2", "ADC", "IVIM"]:
            p = profiles[int(i)][mod]
            for kind, mask_key in [("base", "roi"), ("erode", "erode"), ("dilate", "dilate")]:
                values, names = slic_profile(p, mod, partitions[mod], p[mask_key])
                current[kind].extend(values)
                if kind == "base": meta.extend(names)
        if metadata is None: metadata = meta
        assert metadata == meta
        for kind in rows: rows[kind].append(current[kind])
    n_features = len(metadata) if metadata is not None else 0
    return {**{kind: np.asarray(row, float).reshape(len(indices), n_features) for kind, row in rows.items()}, "metadata": metadata}


def patient_quantiles(profiles, indices):
    rows = {"base": [], "erode": [], "dilate": []}
    metadata = None
    maximum_recovery_error = 0.
    for number, i in enumerate(indices, 1):
        current = {kind: [] for kind in rows}
        meta = []
        for mod in ["T2", "ADC", "IVIM"]:
            p = profiles[int(i)][mod]
            assert np.allclose(p["spacing"], (1, 1, 1))
            for kind, mask_key in [("base", "roi"), ("erode", "erode"), ("dilate", "dilate")]:
                values, names = one_profile(p, mod, p[mask_key])
                current[kind].extend(values)
                if kind == "base": meta.extend(names)
            if mod != "T2":
                restored = original_values(p, mod)[p["roi"]]
                for c, norm in enumerate(p["normalization"]):
                    err = abs(restored[:, c].mean()-norm["mean"])/max(abs(norm["mean"]), 1e-10)
                    assert err < 1e-5
                    maximum_recovery_error = max(maximum_recovery_error, float(err))
        if metadata is None: metadata = meta
        assert metadata == meta
        for kind in rows: rows[kind].append(current[kind])
        if number % 20 == 0: print(f"Patient-local quantitative training features {number}/{len(indices)}", flush=True)
    return {**{kind: np.asarray(row, float) for kind, row in rows.items()}, "metadata": metadata,
            "indices": np.asarray(indices, int), "max_relative_raw_value_recovery_error": maximum_recovery_error}
