"""Training-only nested optimization on the unchanged random120 84/36 split."""
import argparse
import json
import time
import warnings
from contextlib import nullcontext
from pathlib import Path
import lvi_paths
import joblib
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, rankdata
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
import fold_locked_lvi_engine as engine
import slic_habitat_backend as slic
import random120_quantitative_features as features
import run_random120_retained_methods_20261010 as prior

ROOT, OUT = prior.ROOT, prior.ROOT / "lvi_prediction/results/lvi_random120_optimization_20261012"
CLINICAL_PATH = prior.POOL / "clinical_138.csv"
METHODS = ["QuantitativeSLIC", "PatientQuantiles"]
MODS = {"ADC": ("ADC",), "Fusion": ("T2", "ADC", "IVIM")}
ALGORITHMS = [("L2", .03), ("L2", .3), ("L2", 3.), ("ElasticNet", .3)]
ANALYSES = ["Image", "ImageClinical", "ADCImage", "ClinicalReference"]
TRAIN_ALLOWED, PROFILES, QUANTILES, CLINICAL = set(), None, None, None
save, digest = prior.save, prior.digest


def options():
    rows = [{"method": method, "modality": mod, "clinical": clinical, "top_n": top,
             "algorithm": algorithm, "C": c}
            for method in METHODS for mod in MODS for clinical in [False, True]
            for top in [3, 5] for algorithm, c in ALGORITHMS]
    rows += [{"method": "ClinicalOnly", "modality": "None", "clinical": True, "top_n": 0, "algorithm": "L2", "C": c}
             for c in [.03, .3, 3.]]
    for i, row in enumerate(rows): row["candidate_id"] = i
    return rows


def eligible(config, analysis):
    return ((analysis == "Image" and not config["clinical"]) or
            (analysis == "ImageClinical" and config["clinical"] and config["method"] != "ClinicalOnly") or
            (analysis == "ADCImage" and not config["clinical"] and config["modality"] == "ADC") or
            (analysis == "ClinicalReference" and config["method"] == "ClinicalOnly"))


def choose(rows, analysis):
    candidates = [row for row in rows if eligible(row, analysis)]
    return max(candidates, key=lambda row: (row["Objective"], -row["MeanImageFeatureCount"],
        row["algorithm"] == "L2", -row["C"], -row["candidate_id"]))


def config(row):
    return {k: row[k] for k in options()[0]}


def folds(pool, y, seed, n):
    cv = StratifiedKFold(n_splits=n, shuffle=True, random_state=seed)
    return [(pool[a], pool[b]) for a, b in cv.split(pool, y[pool])]


def load_inputs():
    cohort = pd.read_csv(prior.POOL / "cohort_138.csv", dtype={"SubjectID": str})
    manifest = json.loads((prior.OUT / "sample_split_manifest.json").read_text(encoding="utf-8"))
    assert manifest["analysis_id_order"] == cohort.AnalysisID.tolist()
    tr, te = np.asarray(manifest["train"], int), np.asarray(manifest["test"], int)
    assert len(tr) == 84 and len(te) == 36 and not set(tr) & set(te)
    y = np.full(138, -1, int)
    y[tr] = cohort.Outcome.to_numpy(int)[tr]
    return cohort, manifest, y, tr, te


def freeze():
    if (OUT / "protocol.json").exists():
        raise FileExistsError("The declared optimization is frozen; do not expand search or overwrite")
    OUT.mkdir(parents=True, exist_ok=True)
    prior.check_sources()
    assert json.loads((prior.OUT / "independent_verification.json").read_text())["passed"]
    cohort, manifest, y, tr, te = load_inputs()
    old = json.loads((prior.OUT / "protocol.json").read_text(encoding="utf-8"))
    source_hashes = dict(old["source_hashes"])
    for p in [Path(__file__).resolve(), ROOT / "lvi_prediction/random120_quantitative_features.py",
              ROOT / "lvi_prediction/quantile_habitat_features.py", CLINICAL_PATH,
              prior.OUT / "sample_split_manifest.json", prior.OUT / "delivery_manifest.json"]:
        source_hashes[str(p.relative_to(ROOT))] = digest(p)
    protected_names = set(json.loads((prior.OUT / "preservation_before.json").read_text(encoding="utf-8")))
    protected_names.update(str(p.relative_to(ROOT)) for p in (ROOT / "lvi_prediction/results").rglob("*")
                           if p.is_file() and OUT not in p.parents and "__pycache__" not in p.parts)
    save(OUT / "preservation_before.json", {name: digest(ROOT / name) for name in sorted(protected_names)})
    protocol = {"date": "2026-10-12", "user_target": "尽量让训练拟合AUC和测试AUC接近0.75",
        "authorization": "根据这120例再次修改模型；允许增加影像＋术前临床联合模型，另行报告",
        "fixed_membership_and_split": manifest, "source_hashes": source_hashes,
        "candidate_count": len(options()), "candidates": options(), "analyses": ANALYSES,
        "outer_folds": 5, "outer_seed": 42, "inner_folds": 3, "inner_seeds": [61201+i for i in range(5)],
        "final_training_tuning": {"folds": 5, "seed": 61212}, "K": 3,
        "feature_methods": {"QuantitativeSLIC": "current training-only Kmeans; original ADC/D/f recovered; T2 stays patient Z-score; conditional physical 2mm shells recomputed for each contour",
                            "PatientQuantiles": "patient-local quantitative tertiles and spatial/GLCM descriptors; repeat local partition for each contour; no pooled partition fit"},
        "feature_gate": "within current training fold: >=90% finite across three contours, ICC(A,1)>=.8, nonconstant",
        "selection": "training-only MannWhitney effect * ICC^2; greedy absolute Spearman <.9; top3/top5 imaging features",
        "preprocessing": "training-only medians, 1/99 percentile winsorization, StandardScaler",
        "clinical_columns": list(engine.CLINICAL_COLUMNS), "clinical_role": "fixed five columns, no postoperative pathology; source/timing confirmation pending",
        "objective": "inner mean validation AUC - .05*validation SD - .15*max(mean fit AUC-mean validation AUC-.10,0) - .02*abs(mean fit AUC-.75)",
        "primary_selection_does_not_use": ["outer validation AUC", "test predictions/AUC", "which model is closest to .75 on test"],
        "primary_nested_metric": "arithmetic mean outer fold AUC; pooled OOF also separately reported",
        "test": "one joint evaluation of four training-chosen finalists after model and threshold freezing; same test already explored historically",
        "automatic_adoption": False, "excluded_shape_transfer_used": False,
        "discarded_nested5_model_used": False, "clinical_confirmation": "unknown; MRI/treatment timing and uniform pathology standard not confirmed",
        "robustness_scope": "sampling under fixed patient normalization/SLIC/current training centroids; not full resegmentation reproducibility",
        "raw_value_units": "recover existing map numeric values; no unconfirmed physical unit conversion"}
    save(OUT / "protocol.json", protocol)
    save(OUT / "split_manifest.json", manifest)
    (OUT / "fold_features").mkdir(exist_ok=True)
    (OUT / "models").mkdir(exist_ok=True)
    print(f"Frozen {len(options())} candidates; same 120 cases and 84/36 split", flush=True)


def check_sources():
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    for name, h in protocol["source_hashes"].items(): assert digest(ROOT / name) == h, name
    return protocol


def preserved():
    before = json.loads((OUT / "preservation_before.json").read_text(encoding="utf-8"))
    for name, h in before.items(): assert digest(ROOT / name) == h, name
    return len(before)


def install():
    slic.threadpool_limits = lambda **kwargs: nullcontext()


def subset_quantiles(indices):
    lookup = {int(i): j for j, i in enumerate(QUANTILES["indices"])}
    rows = np.asarray([lookup[int(i)] for i in indices], int)
    return {**{kind: QUANTILES[kind][rows] for kind in ["base", "erode", "dilate"]}, "metadata": QUANTILES["metadata"]}


def part(train, validation, seed):
    a, b = np.asarray(train, int), np.asarray(validation, int)
    assert set(a) <= TRAIN_ALLOWED and set(b) <= TRAIN_ALLOWED and not set(a) & set(b)
    dest = OUT / "fold_features" / f"part_{seed}.joblib"
    if dest.exists():
        cached = joblib.load(dest)
        np.testing.assert_array_equal(cached["train"], a)
        np.testing.assert_array_equal(cached["validation"], b)
        return cached
    partitions = {mod: slic.fit_partition(PROFILES, a, mod, 3, seed) for mod in engine.MODALITIES}
    ta = features.tables(PROFILES, partitions, a)
    tb = features.tables(PROFILES, partitions, b) if len(b) else {**{kind: np.empty((0, len(ta["metadata"]))) for kind in ["base", "erode", "dilate"]}, "metadata": ta["metadata"]}
    result = {"train": a, "validation": b, "seed": seed, "partitions": partitions,
        "QuantitativeSLIC": {"train": ta, "validation": tb},
        "PatientQuantiles": {"train": subset_quantiles(a), "validation": subset_quantiles(b)}}
    joblib.dump(result, dest, compress=3)
    return result


def prepare(current, y_train, method, modality, clinical):
    a, b = current["train"], current["validation"]
    indices, names, audit = [], [], []
    image_xa, image_xb = np.empty((len(a), 0)), np.empty((len(b), 0))
    if method != "ClinicalOnly":
        data = current[method]
        ta, tb = data["train"], data["validation"]
        assert ta["metadata"] == tb["metadata"]
        names = [m["name"] for m in ta["metadata"]]
        complete, score, iccs = {}, {}, {}
        eligible_columns = []
        for j, meta in enumerate(ta["metadata"]):
            if meta["modality"] not in MODS[modality]: continue
            triples = np.column_stack([ta[k][:, j] for k in ["base", "erode", "dilate"]])
            good = np.isfinite(triples).all(axis=1)
            icc = engine.icc_absolute(triples[good]) if good.sum() >= 3 else -np.inf
            passed = bool(good.mean() >= .9 and np.isfinite(icc) and icc >= .8 and np.nanstd(ta["base"][:, j]) > 1e-6)
            row = {"Feature": names[j], "ICC": float(icc) if np.isfinite(icc) else None, "CompleteFraction": float(good.mean()), "Gate": passed}
            if passed:
                eligible_columns.append(j)
                col = ta["base"][:, j]
                col = np.where(np.isfinite(col), col, np.nanmedian(col))
                effect = abs(float(mannwhitneyu(col[y_train == 1], col[y_train == 0]).statistic)/
                             (int((y_train == 1).sum())*int((y_train == 0).sum()))-.5)
                score[j], iccs[j] = effect*icc**2, icc
                row["TrainingEffect"] = effect
            audit.append(row)
        priority = sorted(eligible_columns, key=lambda j: (-score[j], -iccs[j], names[j]))
        rank_columns = {}
        for j in priority:
            col = ta["base"][:, j]
            rank_columns[j] = rankdata(np.where(np.isfinite(col), col, np.nanmedian(col)))
            if all(abs(float(np.corrcoef(rank_columns[j], rank_columns[k])[0, 1])) < .9 for k in indices):
                indices.append(j)
        image_xa, image_xb = ta["base"][:, indices], tb["base"][:, indices]
    clinical_xa = CLINICAL[a] if clinical else np.empty((len(a), 0))
    clinical_xb = CLINICAL[b] if clinical else np.empty((len(b), 0))
    xa, xb = np.column_stack((image_xa, clinical_xa)), np.column_stack((image_xb, clinical_xb))
    med = np.nanmedian(np.where(np.isfinite(xa), xa, np.nan), axis=0) if xa.shape[1] else np.array([])
    med = np.where(np.isfinite(med), med, 0.)
    filled = np.where(np.isfinite(xa), xa, med)
    lo, hi = np.quantile(filled, [.01, .99], axis=0) if xa.shape[1] else (np.array([]), np.array([]))
    filled = np.clip(filled, lo, hi)
    mean, scale = filled.mean(axis=0), filled.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.)
    transformed_a = (filled-mean)/scale
    transformed_b = (np.clip(np.where(np.isfinite(xb), xb, med), lo, hi)-mean)/scale
    return {"method": method, "modality": modality, "clinical": clinical, "indices": indices, "names": names,
        "feature_audit": audit, "image_count": len(indices), "median": med, "low": lo, "high": hi,
        "mean": mean, "scale": scale, "xa": transformed_a, "xb": transformed_b, "y_train": y_train}


def fit_prepared(p, c):
    n = min(c["top_n"], p["image_count"])
    columns = list(range(n))+list(range(p["image_count"], p["xa"].shape[1]))
    if columns:
        kw = {"penalty": "elasticnet", "solver": "saga", "l1_ratio": .5} if c["algorithm"] == "ElasticNet" else {"penalty": "l2", "solver": "liblinear"}
        model = LogisticRegression(C=c["C"], max_iter=30000, tol=1e-5, random_state=61212, **kw)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(p["xa"][:, columns], p["y_train"])
        if any(issubclass(w.category, ConvergenceWarning) for w in caught): raise RuntimeError("Convergence failure "+str(c))
        pa = model.predict_proba(p["xa"][:, columns])[:, 1]
        pb = model.predict_proba(p["xb"][:, columns])[:, 1] if len(p["xb"]) else np.array([])
    else:
        model = None
        pa, pb = np.full(len(p["y_train"]), p["y_train"].mean()), np.full(len(p["xb"]), p["y_train"].mean())
    image_positions = p["indices"][:n]
    selected = [p["names"][j] for j in image_positions] + (list(engine.CLINICAL_COLUMNS) if c["clinical"] else [])
    return {"config": c, "model": model, "image_positions": image_positions, "selected_features": selected,
        "feature_names": p["names"], "feature_audit": p["feature_audit"], "image_count": n,
        "median": p["median"][columns], "low": p["low"][columns], "high": p["high"][columns],
        "mean": p["mean"][columns], "scale": p["scale"][columns], "train_probability": pa,
        "validation_probability": pb, "prevalence": float(p["y_train"].mean())}


def fit_model(current, y, c):
    fitted = fit_prepared(prepare(current, y[current["train"]], c["method"], c["modality"], c["clinical"]), c)
    fitted.update({"train": current["train"], "validation": current["validation"], "seed": current["seed"]})
    return fitted


def tune(parts, y):
    raw, predicted = {c["candidate_id"]: [] for c in options()}, {c["candidate_id"]: [] for c in options()}
    for current in parts:
        a, b = current["train"], current["validation"]
        prepared = {}
        for c in options():
            key = (c["method"], c["modality"], c["clinical"])
            if key not in prepared: prepared[key] = prepare(current, y[a], *key)
            fit = fit_prepared(prepared[key], c)
            raw[c["candidate_id"]].append((roc_auc_score(y[b], fit["validation_probability"]),
                roc_auc_score(y[a], fit["train_probability"]), fit["image_count"]))
            predicted[c["candidate_id"]].append((b, fit["validation_probability"]))
    rows = []
    for c in options():
        values = np.asarray(raw[c["candidate_id"]], float)
        mean, sd, trainfit = float(values[:, 0].mean()), float(values[:, 0].std(ddof=1)), float(values[:, 1].mean())
        rows.append({**c, "MeanValidationAUC": mean, "ValidationSD": sd, "MeanFitAUC": trainfit,
            "MeanImageFeatureCount": float(values[:, 2].mean()),
            "Objective": mean-.05*sd-.15*max(trainfit-mean-.10, 0)-.02*abs(trainfit-.75)})
    return rows, predicted


def train():
    global TRAIN_ALLOWED, PROFILES, QUANTILES, CLINICAL
    started = time.time()
    freeze()
    check_sources()
    cohort, manifest, y, tr, te = load_inputs()
    TRAIN_ALLOWED = set(tr)
    PROFILES = prior.load_profiles()
    CLINICAL = pd.read_csv(CLINICAL_PATH).set_index("AnalysisID").loc[cohort.AnalysisID, list(engine.CLINICAL_COLUMNS)].to_numpy(float)
    save(OUT / "clinical_training_missingness.json", {n: int(np.isnan(CLINICAL[tr, j]).sum()) for j, n in enumerate(engine.CLINICAL_COLUMNS)})
    QUANTILES = features.patient_quantiles(PROFILES, tr)
    joblib.dump(QUANTILES, OUT / "training_quantile_features.joblib", compress=3)
    predictions, metrics, grids = [], [], []
    for f, (a, b) in enumerate(folds(tr, y, 42, 5), 1):
        inner = [part(ia, ib, 612000+f*100+j) for j, (ia, ib) in enumerate(folds(a, y, 61200+f, 3))]
        grid, _ = tune(inner, y)
        grids.extend({"OuterFold": f, **row} for row in grid)
        outer = part(a, b, 613000+f*100)
        fitted_cache, stored = {}, {"grid": grid, "models": {}}
        for analysis in ANALYSES:
            selected = config(choose(grid, analysis))
            if selected["candidate_id"] not in fitted_cache: fitted_cache[selected["candidate_id"]] = fit_model(outer, y, selected)
            fit = fitted_cache[selected["candidate_id"]]
            stored["models"][analysis] = fit
            p = fit["validation_probability"]
            metrics.append({"Analysis": analysis, "Fold": f, **selected, "AUC": float(roc_auc_score(y[b], p)),
                "TrainFitAUC": float(roc_auc_score(y[a], fit["train_probability"])), "ImageFeatureCount": fit["image_count"]})
            predictions.extend({"Analysis": analysis, "Fold": f, "AnalysisID": cohort.AnalysisID.iloc[i],
                "Outcome": int(y[i]), "Probability": float(v)} for i, v in zip(b, p))
        joblib.dump(stored, OUT / f"models/outer_fold_{f}.joblib", compress=3)
        print(f"Nested outer {f}/5:", json.dumps({r["Analysis"]:r["AUC"] for r in metrics if r["Fold"] == f}), flush=True)
    pred, fold_metrics = pd.DataFrame(predictions), pd.DataFrame(metrics)
    summary = []
    for analysis in ANALYSES:
        scores = fold_metrics[fold_metrics.Analysis == analysis]
        rows = pred[pred.Analysis == analysis].set_index("AnalysisID").loc[cohort.AnalysisID.iloc[tr]]
        assert len(rows) == 84 and not rows.index.duplicated().any()
        summary.append({"Analysis": analysis, "OuterMeanAUC": float(scores.AUC.mean()), "OuterSD": float(scores.AUC.std(ddof=1)),
            "PooledOOFAUC": float(roc_auc_score(y[tr], rows.Probability)), "OOFBrier": float(brier_score_loss(y[tr], rows.Probability)),
            "MeanImageFeatureCount": float(scores.ImageFeatureCount.mean())})
    pred.to_csv(OUT / "nested_training_predictions.csv", index=False, encoding="utf-8-sig")
    fold_metrics.to_csv(OUT / "outer_fold_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(grids).to_csv(OUT / "inner_candidate_scores.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary).to_csv(OUT / "nested_summary.csv", index=False, encoding="utf-8-sig")
    final_parts = [part(a, b, 614000+j) for j, (a, b) in enumerate(folds(tr, y, 61212, 5))]
    final_grid, final_cv = tune(final_parts, y)
    pd.DataFrame(final_grid).to_csv(OUT / "final_training_grid.csv", index=False, encoding="utf-8-sig")
    final_part = part(tr, [], 615000)
    fitted_cache, final, rows, coefficients = {}, {"models": {}, "partitions": final_part["partitions"], "train": tr}, [], []
    selected_grid = []
    for analysis in ANALYSES:
        chosen = choose(final_grid, analysis)
        c = config(chosen)
        if c["candidate_id"] not in fitted_cache: fitted_cache[c["candidate_id"]] = fit_model(final_part, y, c)
        fit = dict(fitted_cache[c["candidate_id"]])
        oof = np.full(138, np.nan)
        for idx, scores in final_cv[c["candidate_id"]]: oof[idx] = scores
        assert np.isfinite(oof[tr]).all() and np.isnan(oof[te]).all()
        fit["threshold"] = prior.threshold(y[tr], oof[tr])
        final["models"][analysis] = fit
        selected_grid.append({"Analysis": analysis, **c, "TrainFitAUC": float(roc_auc_score(y[tr], fit["train_probability"])),
            "FinalCVMeanAUC": chosen["MeanValidationAUC"], "ImageFeatureCount": fit["image_count"],
            "ClinicalFeatureCount": 5 if c["clinical"] else 0, "Threshold": fit["threshold"]})
        rows.extend({"Analysis": analysis, "AnalysisID": cohort.AnalysisID.iloc[i], "Outcome": int(y[i]),
            "TrainFitProbability": float(pa), "FinalSelectionOOFProbability": float(pb)} for i, pa, pb in zip(tr, fit["train_probability"], oof[tr]))
        beta = fit["model"].coef_[0] if fit["model"] else []
        coefficients.extend({"Analysis": analysis, "Feature": name, "StandardizedCoefficient": float(value)}
            for name, value in zip(fit["selected_features"], beta))
    joblib.dump(final, OUT / "models/final_models.joblib", compress=3)
    pd.DataFrame(rows).to_csv(OUT / "train_fit_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(selected_grid).to_csv(OUT / "final_training_choices.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(coefficients).to_csv(OUT / "selected_feature_coefficients.csv", index=False, encoding="utf-8-sig")
    save(OUT / "training_frozen.json", {"passed": True, "protocol_sha256": digest(OUT / "protocol.json"),
        "model_sha256": digest(OUT / "models/final_models.joblib"), "test_predictions_generated": False,
        "test_AUC_computed": False, "training_csv_hashes": {p.name: digest(p) for p in OUT.glob("*.csv")},
        "fold_cache_hashes": {p.name: digest(p) for p in (OUT / "fold_features").glob("*.joblib")},
        "training_quantile_sha256": digest(OUT / "training_quantile_features.joblib"),
        "protected_file_count": preserved(), "elapsed_seconds": time.time()-started})
    save(OUT / "candidate_status.json", {"status": "exploratory_not_automatically_adopted", "formal_model_use": False,
        "clinical_confirmation_pending": True, "previous_adoption_decisions_unchanged": True})
    print(pd.DataFrame(summary).to_string(index=False), flush=True)
    print(pd.DataFrame(selected_grid).to_string(index=False), flush=True)
    print("All finalists frozen; test unscored this round", flush=True)


def predict(fit, raw, clinical):
    if fit["config"]["method"] != "ClinicalOnly":
        assert [meta["name"] for meta in raw["metadata"]] == fit["feature_names"]
    image = raw["base"][:, fit["image_positions"]] if fit["config"]["method"] != "ClinicalOnly" else np.empty((len(clinical), 0))
    values = np.column_stack((image, clinical if fit["config"]["clinical"] else np.empty((len(clinical), 0))))
    x = (np.clip(np.where(np.isfinite(values), values, fit["median"]), fit["low"], fit["high"])-fit["mean"])/fit["scale"]
    return fit["model"].predict_proba(x)[:, 1] if fit["model"] else np.full(len(clinical), fit["prevalence"])


def evaluate():
    if (OUT / "test_evaluation_started.json").exists(): raise FileExistsError("The joint test evaluation has already started")
    check_sources()
    frozen = json.loads((OUT / "training_frozen.json").read_text(encoding="utf-8"))
    assert frozen["model_sha256"] == digest(OUT / "models/final_models.joblib")
    assert frozen["protocol_sha256"] == digest(OUT / "protocol.json")
    for name, h in frozen["training_csv_hashes"].items(): assert digest(OUT / name) == h, name
    cohort, manifest, y_masked, tr, te = load_inputs()
    final = joblib.load(OUT / "models/final_models.joblib")
    np.testing.assert_array_equal(final["train"], tr)
    save(OUT / "test_evaluation_started.json", {"joint_evaluation_count": 1, "models": ANALYSES,
        "model_sha256": frozen["model_sha256"], "protocol_sha256": frozen["protocol_sha256"], "test_ids": manifest["test_ids"]})
    profiles = prior.load_profiles()
    raw_slic = features.tables(profiles, final["partitions"], te)
    raw_quantiles = features.patient_quantiles(profiles, te)
    clinical = pd.read_csv(CLINICAL_PATH).set_index("AnalysisID").loc[cohort.AnalysisID.iloc[te], list(engine.CLINICAL_COLUMNS)].to_numpy(float)
    raw = {"QuantitativeSLIC": raw_slic, "PatientQuantiles": raw_quantiles, "ClinicalOnly": {"base": np.empty((36, 0))}}
    probability = {analysis: predict(fit, raw[fit["config"]["method"]], clinical) for analysis, fit in final["models"].items()}
    joblib.dump({"raw": raw, "clinical": clinical, "indices": te}, OUT / "test_inference_inputs.joblib", compress=3)
    y = cohort.Outcome.to_numpy(int)[te]
    draws = prior.bootstrap(y, probability)
    rows, results = [], []
    for analysis, p in probability.items():
        fit = final["models"][analysis]
        low, high = np.quantile(draws[analysis], [.025, .975])
        results.append({"Analysis": analysis, **prior.score_metrics(y, p, fit["threshold"]),
            "AUC_CI95Low": float(low), "AUC_CI95High": float(high)})
        rows.extend({"Analysis": analysis, "AnalysisID": cohort.AnalysisID.iloc[i], "Outcome": int(value), "Probability": float(score),
            "PredictedPositive": int(score >= fit["threshold"])} for i, value, score in zip(te, y, p))
    pd.DataFrame(rows).to_csv(OUT / "test_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(results).to_csv(OUT / "test_metrics.csv", index=False, encoding="utf-8-sig")
    pair = []
    for a, b in [("ImageClinical", "ClinicalReference"), ("ImageClinical", "Image")]:
        low, high = np.quantile(draws[a]-draws[b], [.025, .975])
        pair.append({"Comparison": a+" minus "+b, "DeltaAUC": float(roc_auc_score(y, probability[a])-roc_auc_score(y, probability[b])),
                     "CI95Low": float(low), "CI95High": float(high)})
    pd.DataFrame(pair).to_csv(OUT / "paired_auc_comparisons.csv", index=False, encoding="utf-8-sig")
    save(OUT / "evaluation_complete.json", {"passed": True, "joint_test_evaluation_count": 1, "test_patients": 36,
        "prediction_rows": len(rows), "prediction_sha256": digest(OUT / "test_predictions.csv"),
        "metrics_sha256": digest(OUT / "test_metrics.csv"), "protected_file_count": preserved(),
        "automatic_adoption": False})
    print(pd.DataFrame(results).to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["train", "evaluate"], required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        install()
        train() if args.phase == "train" else evaluate()
