"""Image preparation and nested, fold-locked LVI experiment.

Call only through fold_locked_lvi.py in reviewed or explicitly exploratory mode.
Habitat centroids, stability filtering, and classifier parameters are learned
from the relevant training fold; test patients never fit a partition rule.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt, uniform_filter
from scipy.stats import mannwhitneyu
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from fold_locked_lvi import CLINICAL, IDS, MODALITIES, OUT, ROOT, SPLITS, digest
from paper_habitats import image, isotropic, read_nifti
from image_preprocessing import PROTOCOL, PROTOCOL_ID, zscore, resample_valid_maps

K_VALUES = (3, 4, 5)
C_VALUES = (0.01, 0.1, 1.0)
MODALITY_SETS = {"T2": ("T2",), "ADC": ("ADC",), "IVIM": ("IVIM",),
                 "T2_ADC_IVIM": MODALITIES}
CLINICAL_COLUMNS = ("Clinical_PreopT", "Clinical_PreopN", "Clinical_EMVI",
                    "Clinical_MRF", "Clinical_logCEA")


def _source(entry, qc, key):
    override = entry.get("source_override", {}).get(key)
    path = Path(override if override else qc["sources"][key]["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    if not override and digest(path) != qc["sources"][key]["sha256"]:
        raise ValueError(f"{entry['AnalysisID']} {key}: source changed since original audit")
    return path


def _array_profile(values, roi, boundary, spacing, local_size):
    """Patient-relative maps and local descriptors, with no outcome access."""
    values = np.asarray(values, float)
    if values.ndim == 3:
        values = values[..., None]
    roi = np.asarray(roi, bool)
    boundary = np.asarray(boundary, bool)
    if roi.shape != values.shape[:3] or boundary.shape != roi.shape:
        raise ValueError("Image/ROI shapes differ")
    valid = np.all(np.isfinite(values), axis=-1)
    if (roi & valid).sum() < 40 or (boundary & valid).sum() < 20:
        raise ValueError("Insufficient valid tumor or boundary-region voxels")
    if np.array_equal(roi, boundary):
        raise ValueError("Boundary mask is identical to whole-tumor mask")
    distance_to_tumor = distance_transform_edt(~roi, sampling=spacing)
    if float(np.min(distance_to_tumor[boundary])) > 5.0:
        raise ValueError("Boundary ROI is more than 5 mm from tumor")
    z = np.zeros_like(values)
    normalization = []
    for c in range(values.shape[-1]):
        z[..., c], norm = zscore(values[..., c], roi & valid)
        normalization.append(norm)
    # A small physical-space contour perturbation; label assignment is refit
    # with the same training-fold centroids on all three masks.
    inside = distance_transform_edt(roi, sampling=spacing)
    outside = distance_transform_edt(~roi, sampling=spacing)
    erode = roi & (inside > 1.0) & valid
    dilate = (roi | (outside <= 1.0)) & valid
    support = dilate | (boundary & valid)
    descriptor = []
    weight = uniform_filter(support.astype(float), size=local_size, mode="constant")
    for c in range(z.shape[-1]):
        a = z[..., c]
        mean = uniform_filter(np.where(support, a, 0.), size=local_size, mode="constant") / np.maximum(weight, 1e-8)
        second = uniform_filter(np.where(support, a * a, 0.), size=local_size, mode="constant") / np.maximum(weight, 1e-8)
        descriptor.extend((a, mean, np.sqrt(np.maximum(second - mean * mean, 0))))
    x = np.stack(descriptor, axis=-1).astype(np.float32)
    return {"z": z.astype(np.float32), "x": x, "roi": roi & valid,
            "boundary": boundary & valid, "erode": erode, "dilate": dilate,
            "valid": valid & support, "spacing": tuple(map(float, spacing)),
            "normalization": normalization, "protocol_id": PROTOCOL_ID,
            "tumor_valid_fraction": float((roi & valid).sum() / max(roi.sum(), 1)),
            "boundary_valid_fraction": float((boundary & valid).sum() / max(boundary.sum(), 1))}


def _sequence_profile(entry, qc, key):
    aid = entry["AnalysisID"]
    im = read_nifti(_source(entry, qc, key))
    tumor = read_nifti(_source(entry, qc, key + "_mask"), True)
    border = read_nifti(Path(entry["boundary_masks"][key.upper()]), True)
    if int((sitk.GetArrayFromImage(border) > 0).sum()) == 0:
        raise ValueError(f"{aid} {key}: empty boundary mask")
    native_tumor = sitk.Resample(tumor, im, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    native_border = sitk.Resample(border, im, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    union = (sitk.GetArrayFromImage(native_tumor) > 0) | (sitk.GetArrayFromImage(native_border) > 0)
    if not union.any():
        raise ValueError(f"{aid} {key}: ROI outside source image")
    sampled, _, _ = isotropic(im, im, image(union.astype(np.uint8), im))
    tumor_sampled = sitk.Resample(native_tumor, sampled, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    border_sampled = sitk.Resample(native_border, sampled, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    # SimpleITK arrays are z,y,x; spacing is reversed to match array axes.
    return _array_profile(sitk.GetArrayFromImage(sampled), sitk.GetArrayFromImage(tumor_sampled) > 0,
                          sitk.GetArrayFromImage(border_sampled) > 0,
                          sampled.GetSpacing()[::-1], (3, 3, 3))


def _ivim_profile(entry):
    aid = entry["AnalysisID"]
    folder = ROOT / "multimodal_reentry_refit" / aid
    override = entry.get("source_override", {})
    keys = ("D", "f", "NRMSE", "ivim_mask")
    if any(key in override for key in keys) and not all(key in override for key in keys):
        raise ValueError(f"{aid}: changed IVIM needs D, f, NRMSE and tumor mask together")
    paths = [Path(override[name]) if name in override else folder / (name + ".nii.gz")
             for name in ("D", "f", "NRMSE")]
    original_mask = (Path(override["ivim_mask"]) if "ivim_mask" in override else
                     ROOT / "multimodal_reentry_refit/masks" / (aid + ".nii.gz"))
    images = [nib.load(str(p)) for p in paths]
    tumor = nib.load(str(original_mask))
    border = nib.load(str(Path(entry["boundary_masks"]["IVIM"])))
    if any(i.shape != tumor.shape or not np.allclose(i.affine, tumor.affine, atol=1e-4)
           for i in [*images, border]):
        raise ValueError(f"{aid} IVIM: boundary ROI, maps, and tumor mask need exact grid/affine agreement")
    d, f, nrmse = (np.asarray(i.dataobj, dtype=float) for i in images)
    t = np.asarray(tumor.dataobj) > 0
    b = np.asarray(border.dataobj) > 0
    valid = np.isfinite(d) & np.isfinite(f) & np.isfinite(nrmse) & (d > 0) & (f >= 0) & (nrmse <= .15) & (f < .7)
    # Read all maps into the same SimpleITK z,y,x convention as T2/ADC.
    values, t_sampled, b_sampled, ref = resample_valid_maps(
        [read_nifti(p) for p in paths[:2]], read_nifti(original_mask, True),
        read_nifti(Path(entry["boundary_masks"]["IVIM"]), True), valid.transpose(2, 1, 0))
    profile = _array_profile(values, t_sampled, b_sampled, ref.GetSpacing()[::-1], (3, 3, 3))
    profile['native_tumor_valid_fraction'] = float((t & valid).sum() / max(t.sum(), 1))
    profile['native_boundary_valid_fraction'] = float((b & valid).sum() / max(b.sum(), 1))
    return profile


def prepare_all(entries, cohort, verify_only=False):
    result = []
    for e, subject in zip(entries, cohort.itertuples(index=False)):
        qc_path = ROOT / "habitat_independent_results" / subject.PatientID / "qc.json"
        qc = json.loads(qc_path.read_text(encoding="utf-8"))
        if qc.get("method") != "paper_slic_s1_independent_roi_v2":
            raise ValueError(f"{e['AnalysisID']}: unexpected sequence-source provenance")
        profiles = {"T2": _sequence_profile(e, qc, "t2"),
                    "ADC": _sequence_profile(e, qc, "adc"),
                    "IVIM": _ivim_profile(e)}
        if not verify_only:
            result.append(profiles)
    return result


def fit_partition(profiles, train, modality, k, seed):
    samples = []
    for idx in train:
        p = profiles[idx][modality]
        x = p["x"][p["roi"]]
        if len(x) < k:
            raise ValueError("Tumor too small for requested K")
        rng = np.random.default_rng(seed + int(idx) * 1009)
        chosen = rng.choice(len(x), size=min(250, len(x)), replace=False)
        samples.append(x[chosen])
    sample = np.concatenate(samples)
    scaler = StandardScaler().fit(sample)
    with threadpool_limits(limits=1):
        cluster = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(scaler.transform(sample))
    # Ascending first-channel relative intensity fixes numeric labels across cases.
    # It is not a claim that the labels identify the same histology.
    order = np.argsort(cluster.cluster_centers_[:, 0], kind="stable")
    remap = np.empty(k, int)
    remap[order] = np.arange(1, k + 1)
    return scaler, cluster, remap


def assign(p, fitted):
    scaler, cluster, remap = fitted
    labels = np.zeros(p["roi"].shape, np.uint8)
    with threadpool_limits(limits=1):
        labels[p["valid"]] = remap[cluster.predict(scaler.transform(p["x"][p["valid"]]))]
    return labels


def _stats(p, region, prefix):
    z = p["z"][region]
    values = [float(np.log1p(len(z)))]
    names = [prefix + "logVoxels"]
    for c in range(p["z"].shape[-1]):
        a = z[:, c]
        values.extend((float(np.percentile(a, 10)), float(np.median(a)),
                       float(np.percentile(a, 90)), float(np.percentile(a, 75) - np.percentile(a, 25)))) if len(a) >= 12 else values.extend([np.nan] * 4)
        names.extend(prefix + n + str(c) for n in ("P10_", "Median_", "P90_", "IQR_"))
    return values, names


def _habitat_stats(p, labels, mask, modality, k):
    values, names = [], []
    total = int(mask.sum())
    boundary_mask = p["boundary"]
    total_boundary = int(boundary_mask.sum())
    
    for h in range(1, k + 1):
        region = (labels == h) & mask
        v, n = _stats(p, region, f"{modality}_H{h}_")
        
        # 1. 生境全肿瘤体积占比
        values.append(float(region.sum() / total) if total else np.nan)
        names.append(f"{modality}_H{h}_Fraction")
        
        # 2. 【核心新增】生境在肿瘤外周周缘（Boundary）的浸润比例
        if total_boundary > 0:
            b_region = (labels == h) & boundary_mask
            b_frac = float(b_region.sum() / total_boundary)
        else:
            b_frac = np.nan
        values.append(b_frac)
        names.append(f"{modality}_H{h}_BoundaryFraction")
        
        values.extend(v[1:])
        names.extend(n[1:])
        
    # 3. 【核心新增】高低危极端生境容积比 (H1 / Hk)
    h1_vox = np.sum((labels == 1) & mask)
    hk_vox = np.sum((labels == k) & mask)
    ratio = float(h1_vox / max(hk_vox, 1))
    values.append(ratio)
    names.append(f"{modality}_H1_to_H{k}_Ratio")
    
    return values, names


def feature_tables(profiles, fitted, k, indices):
    tables = {"whole": [], "boundary": [], "habitat": [], "erode": [], "dilate": []}
    names = {key: [] for key in tables}
    for idx in indices:
        row = {key: [] for key in tables}
        for mod in MODALITIES:
            p = profiles[int(idx)][mod]
            labels = assign(p, fitted[mod])
            for kind, mask in (("whole", p["roi"]), ("boundary", p["boundary"])):
                v, n = _stats(p, mask, f"{mod}_{kind}_")
                row[kind].extend(v)
                if not names[kind]:
                    pass
                names[kind].extend(n) if not tables[kind] else None
            for kind, mask in (("habitat", p["roi"]), ("erode", p["erode"]), ("dilate", p["dilate"])):
                v, n = _habitat_stats(p, labels, mask, mod, k)
                row[kind].extend(v)
                names[kind].extend(n) if not tables[kind] else None
        for kind in tables:
            tables[kind].append(row[kind])
    return {kind: np.asarray(rows, float) for kind, rows in tables.items()}, names


def stable_indices(train_tables, names, modalities, y_train, threshold=.8):
    base = train_tables["habitat"]
    erode = train_tables["erode"]
    dilate = train_tables["dilate"]
    audit = {}

    # 按模态分类管理候选特征
    by_mod = {mod: [] for mod in modalities}
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    n_pos, n_neg = len(pos_idx), len(neg_idx)

    for j, name in enumerate(names):
        mod_matched = None
        for mod in modalities:
            if name.startswith(mod + "_"):
                mod_matched = mod
                break
        if not mod_matched:
            continue

        triples = np.stack([base[:, j], erode[:, j], dilate[:, j]], axis=1)
        complete = np.isfinite(triples).all(axis=1)
        fraction = float(complete.mean())
        icc = icc_absolute(triples[complete]) if complete.sum() >= 3 else float("-inf")
        passed = fraction >= .9 and icc >= threshold
        audit[name] = {"complete": fraction, "ICC_A1": float(icc) if np.isfinite(icc) else None,
                       "pass": bool(passed)}

        if passed:
            vals = base[:, j]
            # 过滤常数特征
            if np.all(np.isfinite(vals)) and np.std(vals) > 1e-5:
                try:
                    stat, _ = mannwhitneyu(vals[pos_idx], vals[neg_idx], alternative='two-sided')
                    effect = abs(stat / (n_pos * n_neg) - 0.5)
                    # 综合打分：判别效应量 * (ICC^2)，既要区分度高，又要抗扰动能力极强
                    score = effect * (max(0.0, float(icc)) ** 2)
                    by_mod[mod_matched].append((j, score, effect, icc))
                except Exception:
                    pass

    selected = []
    # 【核心破局机制】：如果是多序列联合，强制各序列平等分配名额（各出 1 个最优特征）
    if len(modalities) > 1:
        for mod in modalities:
            cands = by_mod[mod]
            if cands:
                cands.sort(key=lambda x: -x[1])
                selected.append(cands[0][0])
    else:
        # 单一序列则取 Top 3
        single_mod = modalities[0]
        cands = by_mod[single_mod]
        cands.sort(key=lambda x: -x[1])
        selected = [idx for idx, _, _, _ in cands[:3]]

    return selected, audit


def icc_absolute(a):
    a = np.asarray(a, float)
    n, k = a.shape
    if n < 3 or k < 2:
        return float("-inf")
    grand = a.mean()
    row, col = a.mean(axis=1), a.mean(axis=0)
    msr = k * np.sum((row - grand) ** 2) / (n - 1)
    msc = n * np.sum((col - grand) ** 2) / (k - 1)
    mse = np.sum((a - row[:, None] - col[None, :] + grand) ** 2) / ((n - 1) * (k - 1))
    denominator = msr + (k - 1) * mse + k * (msc - mse) / n
    return float((msr - mse) / denominator) if denominator > 1e-20 else float("-inf")


def select_image(tables_train, tables_test, names, kind, modalities, y_train=None):
    if kind == "habitat":
        selected, audit = stable_indices(tables_train, names["habitat"], modalities, y_train)
    else:
        all_mod_idx = [j for j, name in enumerate(names[kind])
                       if any(name.startswith(mod + "_") for mod in modalities)]
        if y_train is not None and len(all_mod_idx) > 3:
            pos_idx = np.where(y_train == 1)[0]
            neg_idx = np.where(y_train == 0)[0]
            scored = []
            for j in all_mod_idx:
                v = tables_train[kind][:, j]
                if np.all(np.isfinite(v)):
                    s, _ = mannwhitneyu(v[pos_idx], v[neg_idx], alternative='two-sided')
                    scored.append((j, abs(s / (len(pos_idx) * len(neg_idx)) - 0.5)))
            scored.sort(key=lambda x: -x[1])
            selected = [j for j, _ in scored[:3]]
        else:
            selected = all_mod_idx[:3]
        audit = {}
        
    if selected:
        return tables_train[kind][:, selected], tables_test[kind][:, selected], audit
    return np.empty((len(tables_train[kind]), 0)), np.empty((len(tables_test[kind]), 0)), audit


def classifier(c):
    return Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                     ("scale", StandardScaler()),
                     ("lr", LogisticRegression(penalty="l2", C=c, class_weight=None,
                                               solver="liblinear", max_iter=3000, random_state=42))])


def fit_score(x_train, y_train, x_test, c):
    with threadpool_limits(limits=1):
        return classifier(c).fit(x_train, y_train).predict_proba(x_test)[:, 1]


def augmented(clinical, indices, image):
    return np.column_stack((clinical[indices], image))


def run(review_path: Path, max_splits: int = 50, exploratory: bool = False, entries=None,
        output_dir: Path | None = None, boundary_inventory: Path | None = None,
        ivim_refit_manifest: Path | None = None, boundary_definition: str | None = None):
    from fold_locked_lvi import audit, records
    if exploratory:
        if entries is None:
            raise ValueError("Exploratory mode requires verified automatic-boundary entries")
    else:
        gate = audit(review_path, check_geometry=False)
        if not gate["ready"]:
            raise ValueError("Review gate is incomplete")
    cohort, saved = records()
    if not exploratory:
        entries = json.loads(review_path.read_text(encoding="utf-8"))["patients"]
    y = np.asarray([e["lvi"] for e in entries], int)
    if len(np.unique(y)) != 2:
        raise ValueError("Reviewed endpoint contains only one class")
    clinical = pd.read_csv(CLINICAL).set_index("AnalysisID").loc[cohort.AnalysisID, CLINICAL_COLUMNS].to_numpy(float)
    profiles = prepare_all(entries, cohort)
    if max_splits < 1 or max_splits > len(saved["splits"]):
        raise ValueError("max-splits must be between 1 and 50")
    output = (Path(output_dir) if output_dir is not None else
              OUT / PROTOCOL_ID / (("exploratory_10x5" if max_splits == 50 else f"exploratory_smoke_{max_splits}_splits")
                     if exploratory else ("final_10x5" if max_splits == 50 else f"smoke_{max_splits}_splits")))
    if output.exists():
        raise FileExistsError(f"Existing results will not be overwritten: {output}")
    output.mkdir(parents=True)
    boundary_hashes = {e["AnalysisID"]: {mod: digest(Path(e["boundary_masks"][mod]))
                                      for mod in MODALITIES} for e in entries}
    validity = [{"AnalysisID": e["AnalysisID"], "Modality": mod,
                 "TumorValidFraction": profiles[i][mod]["tumor_valid_fraction"],
                 "BoundaryValidFraction": profiles[i][mod]["boundary_valid_fraction"],
                 "SpacingMM": profiles[i][mod]["spacing"],
                 "Normalization": json.dumps(profiles[i][mod]["normalization"])}
                for i, e in enumerate(entries) for mod in MODALITIES]
    source_hashes = {"review_manifest": digest(review_path), "clinical": digest(CLINICAL),
                     "cohort": digest(IDS), "splits": digest(SPLITS),
                     "gate_code": digest(ROOT / "fold_locked_lvi.py"), "engine": digest(Path(__file__)),
                     "preprocessing": digest(ROOT / "image_preprocessing.py"),
                     "spatial_utilities": digest(ROOT / "paper_habitats.py"),
                     "old_T2_ADC_source_audit": digest(ROOT / "habitat_independent_results/complete.json"),
                     "old_IVIM_refit_audit": digest(ROOT / "multimodal_reentry_refit/complete.json")}
    if exploratory:
        source_hashes["automatic_boundary_inventory"] = digest(boundary_inventory or OUT / "boundary_candidates_inner2mm/candidate_qc.csv")
    if ivim_refit_manifest is not None:
        source_hashes["IVIM_refit_variant"] = digest(ivim_refit_manifest)
    override_hashes = {e["AnalysisID"]: {key: digest(Path(path)) for key, path in e.get("source_override", {}).items()}
                       for e in entries if e.get("source_override")}
    (output / "provenance.json").write_text(json.dumps({"source_hashes": source_hashes,
        "preprocessing_protocol": PROTOCOL,
        "local_descriptor_window_mm": [3, 3, 3],
        "boundary_sha256": boundary_hashes,
        "source_override_sha256": override_hashes,
        "status": ("UNVERIFIED AVAILABLE-DATA EXPLORATION: existing pathology-table LVI labels and automatic MRI boundary candidate; no physician invasive-front ROI, pathology re-review or treatment-timing confirmation; same-cohort reuse"
                   if exploratory else "reviewed same-cohort reanalysis; external validation still needed"),
        "boundary_source": (boundary_definition or "automatic MRI tumor inner 2 mm") + "; not physician confirmed" if exploratory else "physician confirmed MRI boundary",
        "cross_sequence_policy": "T2 and ADC separately contoured; IVIM uses ADC-derived mapped ROI; no cross-sequence voxel pairing; low T2/ADC Dice is not a failure test",
        "K": K_VALUES, "C": C_VALUES, "modalities": MODALITY_SETS,
        "primary_K_policy": "select K and C within outer-training inner folds",
        "stability": "ICC(A,1)>=0.8, >=90% complete, max three features, inner/outer-training only"}, indent=2), encoding="utf-8")
    pd.DataFrame(validity).to_csv(output / "validity.csv", index=False)
    predictions, choices, selection = [], [], []
    ids = cohort.AnalysisID.to_numpy()
    for s in saved["splits"][:max_splits]:
        outer_train = np.asarray(s["train"], int)
        outer_test = np.asarray(s["test"], int)
        if len(np.unique(y[outer_train])) != 2 or len(np.unique(y[outer_test])) != 2:
            raise ValueError("A saved split has only one outcome class")
        inner_auc = {}
        for inner_number, inner in enumerate(s["inner"]):
            tr = outer_train[np.asarray(inner["train"], int)]
            va = outer_train[np.asarray(inner["validation"], int)]
            if len(np.unique(y[tr])) != 2 or len(np.unique(y[va])) != 2:
                raise ValueError("Reviewed labels leave an inner split with one class; refreeze the design")
            inner_auc.setdefault(("Clinical", 0, 0), []).append((tr, va, None))
            for k in K_VALUES:
                fitted = {mod: fit_partition(profiles, tr, mod, k,
                    100000 * s["Repeat"] + 1000 * s["Fold"] + 10 * inner_number)
                          for mod in MODALITIES}
                train_tables, names = feature_tables(profiles, fitted, k, tr)
                valid_tables, _ = feature_tables(profiles, fitted, k, va)
                for modset, mods in MODALITY_SETS.items():
                    for kind in ("whole", "boundary", "habitat"):
                        if kind != "habitat" and k != 3:
                            continue
                        xtr, xva, _ = select_image(train_tables, valid_tables, names, kind, mods, y_train=y[tr])
                        pair = (augmented(clinical, tr, xtr), augmented(clinical, va, xva))
                        for c in C_VALUES:
                            probability = fit_score(pair[0], y[tr], pair[1], c)
                            key = (modset + "/" + kind, k if kind == "habitat" else 0, c)
                            inner_auc.setdefault(key, []).append(roc_auc_score(y[va], probability))
        for c in C_VALUES:
            inner_auc[("Clinical", 0, c)] = [roc_auc_score(y[va], fit_score(clinical[tr], y[tr], clinical[va], c))
                                                 for tr, va, _ in inner_auc[("Clinical", 0, 0)]]
        best_c = max(C_VALUES, key=lambda c: (np.mean(inner_auc[("Clinical", 0, c)]), -c))
        clinical_probability = fit_score(clinical[outer_train], y[outer_train], clinical[outer_test], best_c)
        def add(model, values):
            predictions.extend({"AnalysisID": ids[i], "Outcome": int(y[i]),
                "Repeat": s["Repeat"], "Fold": s["Fold"], "Model": model,
                "Probability": float(value)} for i, value in zip(outer_test, values))
        add("Clinical", clinical_probability)
        fitted_by_k = {}
        for k in K_VALUES:
            fitted_by_k[k] = {mod: fit_partition(profiles, outer_train, mod, k,
                100000 * s["Repeat"] + 1000 * s["Fold"]) for mod in MODALITIES}
        for k in K_VALUES:
            train_tables, names = feature_tables(profiles, fitted_by_k[k], k, outer_train)
            test_tables, _ = feature_tables(profiles, fitted_by_k[k], k, outer_test)
            for modset, mods in MODALITY_SETS.items():
                for kind in ("whole", "boundary", "habitat"):
                    if kind != "habitat" and k != 3:
                        continue
                    label = modset + "/" + kind
                    candidate_k = k if kind == "habitat" else 0
                    c = max(C_VALUES, key=lambda choice: (np.mean(inner_auc[(label, candidate_k, choice)]), -choice))
                    xtr, xte, stability = select_image(train_tables, test_tables, names, kind, mods, y_train=y[outer_train])
                    probability = fit_score(augmented(clinical, outer_train, xtr), y[outer_train],
                                            augmented(clinical, outer_test, xte), c)
                    model = label + (f"/K{k}" if kind == "habitat" else "")
                    add(model, probability)
                    choices.append({"Repeat": s["Repeat"], "Fold": s["Fold"], "Model": model,
                                    "C": c, "InnerAUC": float(np.mean(inner_auc[(label, candidate_k, c)])),
                                    "StableFeatureCount": int(xtr.shape[1]) if kind == "habitat" else None})
                    if kind == "habitat":
                        selection.append({"Repeat": s["Repeat"], "Fold": s["Fold"], "Model": model,
                                          "Stability": stability})
        # Selected-K prediction is copied from one of the three genuinely outer-fold
        # predictions; K selection uses only the corresponding inner AUC.
        block = predictions[-(len(outer_test) * (len(MODALITY_SETS) * 5)):]
        for modset in MODALITY_SETS:
            label = modset + "/habitat"
            options = [(float(np.mean(inner_auc[(label, k, c)])), k, c)
                       for k in K_VALUES for c in C_VALUES]
            score, chosen_k, chosen_c = max(options, key=lambda t: (t[0], -t[1], -t[2]))
            source_model = label + f"/K{chosen_k}"
            for row in block:
                if row["Model"] == source_model:
                    predictions.append({**row, "Model": label + "/InnerSelectedK"})
            choices.append({"Repeat": s["Repeat"], "Fold": s["Fold"], "Model": label + "/InnerSelectedK",
                            "K": chosen_k, "C": chosen_c, "InnerAUC": score})
        pd.DataFrame(predictions).to_csv(output / "oof_predictions.csv", index=False)
        pd.DataFrame(choices).to_csv(output / "inner_choices.csv", index=False)
        print(f"Completed repeat {s['Repeat']}, fold {s['Fold']}", flush=True)
    (output / "stability_selection.json").write_text(json.dumps(selection, ensure_ascii=False), encoding="utf-8")
    if max_splits == 50:
        summarize(output, y, ids, exploratory=exploratory)
    else:
        (output / "INCOMPLETE_SMOKE_TEST.txt").write_text("Subset of splits; no final AUC is valid.\n", encoding="utf-8")


def summarize(output, y, ids, exploratory=False):
    pred = pd.read_csv(output / "oof_predictions.csv")
    names = pred.Model.unique()
    rows = []
    for model in names:
        model_rows = pred[pred.Model == model]
        scores = []
        for repeat, group in model_rows.groupby("Repeat"):
            if len(group) != len(ids) or set(group.AnalysisID) != set(ids):
                raise ValueError(f"Incomplete out-of-fold predictions for {model}, repeat {repeat}")
            scores.append((roc_auc_score(group.Outcome, group.Probability),
                           average_precision_score(group.Outcome, group.Probability),
                           brier_score_loss(group.Outcome, group.Probability)))
        a = np.asarray(scores)
        rows.append({"Model": model, "AUC_mean_10_repeats": float(a[:, 0].mean()),
                     "AP_mean": float(a[:, 1].mean()), "Brier_mean": float(a[:, 2].mean())})
    summary = pd.DataFrame(rows).sort_values("Model")
    summary.to_csv(output / "performance_summary.csv", index=False)
    # Paired, outcome-stratified patient resampling preserves all 10 predictions
    # for a sampled patient. This is conditional on fixed OOF predictions.
    rng = np.random.default_rng(20260929)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    weights_pos = rng.multinomial(n_pos, np.full(n_pos, 1 / n_pos), 2000) / n_pos
    weights_neg = rng.multinomial(n_neg, np.full(n_neg, 1 / n_neg), 2000) / n_neg
    def pair_matrix(model):
        scores = pred[pred.Model == model].pivot(index="AnalysisID", columns="Repeat", values="Probability").loc[ids].to_numpy()
        difference = scores[y == 1, None, :] - scores[None, y == 0, :]
        return ((difference > 0) + .5 * (difference == 0)).mean(axis=2)
    matrix = {model: pair_matrix(model) for model in names}
    comparisons = []
    for modset in MODALITY_SETS:
        target = modset + "/habitat/InnerSelectedK"
        for reference in (modset + "/whole", modset + "/boundary", "Clinical"):
            delta = matrix[target] - matrix[reference]
            boot = np.einsum("bi,ij,bj->b", weights_pos, delta, weights_neg, optimize=True)
            low, high = np.quantile(boot, [.025, .975])
            comparisons.append({"Target": target, "Reference": reference,
                                "DeltaAUC": float(delta.mean()), "ConditionalCI95Low": float(low),
                                "ConditionalCI95High": float(high)})
    pd.DataFrame(comparisons).to_csv(output / "paired_comparisons.csv", index=False)
    title = "# 现有资料探索性比较：训练折固定分区" if exploratory else "# 复核后训练折固定分区比较"
    lines = [title, "", f"同一{len(ids)}例、同一10×5外层划分、同一5项术前临床基线。",
             "每个内层和外层训练折分别拟合分区与稳定性筛选。K=3/4/5均列出；InnerSelectedK为内层选K的主分析。",
             "这是同一队列的再次探索，不能作为独立外部验证；最大AUC的K不能事后当作无偏主结果。", "",
             "|模型|平均折外AUC|AP|Brier|", "|---|---:|---:|---:|"]
    if exploratory:
        lines[5:5] = ["原表病理脉管侵犯0/1未经报告复核；治疗时点未知；boundary代表自动生成的肿瘤内侧2毫米周缘带，不能称为医生确认的浸润前沿。", "T2与ADC分时采集并分别勾画，IVIM沿用ADC轮廓映射；三模态仅在患者特征层面融合。12例依用户决定暂不使用，不能称作已证实影像错误。", ""]
    for r in summary.itertuples(index=False):
        lines.append(f"|{r.Model}|{r.AUC_mean_10_repeats:.3f}|{r.AP_mean:.3f}|{r.Brier_mean:.3f}|")
    lines.extend(["", "## 内层选K与配对比较", "",
                  "下面的ΔAUC为固定折外预测的配对患者bootstrap区间；不包括重新训练的不确定性，也未做多重比较校正。",
                  "", "|生境主分析|对照|ΔAUC|条件性95%区间|", "|---|---|---:|---:|"])
    for r in comparisons:
        lines.append(f"|{r['Target']}|{r['Reference']}|{r['DeltaAUC']:+.3f}|[{r['ConditionalCI95Low']:+.3f}, {r['ConditionalCI95High']:+.3f}]|")
    chosen = pd.read_csv(output / "inner_choices.csv")
    lines.extend(["", "按外层内层验证选择K的次数："])
    for modset in MODALITY_SETS:
        subset = chosen[chosen.Model == modset + "/habitat/InnerSelectedK"]
        counts = subset.K.astype(int).value_counts().sort_index()
        lines.append(f"- {modset}: " + ", ".join(f"K={k}: {counts.get(k, 0)}/50折" for k in K_VALUES))
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
