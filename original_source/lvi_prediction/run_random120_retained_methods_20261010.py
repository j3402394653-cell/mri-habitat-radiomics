"""One random 120/138 sample, one 84/36 split, two retained-method refits.

Fixed K/C and fixed feature-selection rules. Fivefold training validation,
then freeze both models and perform one joint test evaluation. No adoption.
"""
from contextlib import nullcontext
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fold_locked_lvi_engine as engine
import slic_habitat_backend as slic
import optimized_habitat_backend as fast

BASE = ROOT / "fold_locked_lvi_study/isotropic1mm_zscore_firstorder_v1"
POOL = BASE / "slic_138_holdout_70_30_seed42"
HISTORICAL = BASE / "five_models_fixed118_seed42_20261008"
PROFILE = BASE / "slic_supervoxel_100_10x5/profiles.joblib"
OUT = ROOT / "lvi_prediction/results/lvi_random120_refit_20261010"
SAMPLE_SEED, SPLIT_SEED, CV_SEED, BOOTSTRAP_SEED = 20261010, 42, 42, 20261010
MODELS = {"Fusion": {"modalities": ("T2", "ADC", "IVIM"), "K": 3, "C": 1.},
          "ADC": {"modalities": ("ADC",), "K": 3, "C": .01}}
PROFILES, TRAIN_ALLOWED = None, set()


def digest(path):
    with Path(path).open("rb") as stream:
        h = hashlib.sha256()
        for chunk in iter(lambda: stream.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sample_split(cohort):
    y, ids = cohort.Outcome.to_numpy(int), cohort.AnalysisID.to_numpy()
    sampled = np.sort(np.random.default_rng(SAMPLE_SEED).choice(len(cohort), 120, replace=False))
    tr, te = train_test_split(sampled, test_size=.3, stratify=y[sampled], random_state=SPLIT_SEED)
    tr, te = np.sort(tr), np.sort(te)
    folds = [{"Fold": f, "train": tr[a].tolist(), "validation": tr[b].tolist(),
              "partition_seed": 101000 + 10*(f-1)}
             for f, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=CV_SEED).split(tr, y[tr]), 1)]
    assert len(tr) == 84 and len(te) == 36 and not set(tr) & set(te)
    assert set(tr) | set(te) == set(sampled)
    assert sorted(i for fold in folds for i in fold["validation"]) == tr.tolist()
    counts = {group: {"n": len(idx), "positive": int(y[idx].sum()), "negative": int((1-y[idx]).sum())}
              for group, idx in [("pool", np.arange(len(cohort))), ("sample", sampled), ("train", tr), ("test", te)]}
    return {"sample_seed": SAMPLE_SEED, "split_seed": SPLIT_SEED, "training_cv_seed": CV_SEED,
        "sampling": "uniform without replacement; outcomes not used for sampling",
        "split": "one outcome-stratified 70/30 split", "analysis_id_order": ids.tolist(),
        "sampled_indices": sampled.tolist(), "omitted_indices": sorted(set(range(len(cohort)))-set(sampled)),
        "train": tr.tolist(), "test": te.tolist(), "train_ids": ids[tr].tolist(), "test_ids": ids[te].tolist(),
        "folds": folds, "counts": counts, "final_partition_seed": 101000}


def load_inputs():
    cohort = pd.read_csv(POOL / "cohort_138.csv", dtype={"SubjectID": str})
    manifest = json.loads((OUT / "sample_split_manifest.json").read_text(encoding="utf-8"))
    assert manifest["analysis_id_order"] == cohort.AnalysisID.tolist()
    tr, te = np.asarray(manifest["train"], int), np.asarray(manifest["test"], int)
    y = np.full(len(cohort), -1, int)
    y[tr] = cohort.Outcome.to_numpy(int)[tr]
    return cohort, manifest, y, tr, te


def check_sources():
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    for name, h in protocol["source_hashes"].items():
        assert digest(ROOT / name) == h, name
    return protocol


def check_preserved():
    before = json.loads((OUT / "preservation_before.json").read_text(encoding="utf-8"))
    for name, h in before.items():
        assert digest(ROOT / name) == h, name
    return len(before)


def freeze():
    if (OUT / "protocol.json").exists():
        raise FileExistsError("A frozen random120 experiment already exists; do not redraw or overwrite")
    audit = json.loads((OUT / "clinical_label_audit.json").read_text(encoding="utf-8"))
    assert audit["passed"] and audit["cohort_sha256"] == digest(POOL / "cohort_138.csv")
    assert audit["clinical_workbook_sha256"] == digest(ROOT / "clinical_data.xlsx")
    complete = json.loads((POOL / "complete.json").read_text(encoding="utf-8"))
    assert complete["passed"] and json.loads((POOL / "independent_verification.json").read_text())["passed"]
    assert digest(PROFILE) == json.loads((BASE / "slic_supervoxel_100_10x5/profile_cache_hash.json").read_text())["sha256"]
    historical = json.loads((HISTORICAL / "delivery_manifest.json").read_text(encoding="utf-8"))
    for name in ("models/primary_fusion__H.joblib", "models/supplement_adc__H.joblib"):
        assert digest(HISTORICAL / name) == historical["output_hashes"][name.replace("/", chr(92))]
    cohort = pd.read_csv(POOL / "cohort_138.csv", dtype={"SubjectID": str})
    assert len(cohort) == cohort.AnalysisID.nunique() == cohort.SubjectID.nunique() == 138
    original = pd.read_csv(ROOT / "corrected_118_cohort/cohort_118.csv")
    assert cohort.AnalysisID.iloc[:118].tolist() == original.AnalysisID.tolist()
    assert (cohort.Outcome.iloc[:118].to_numpy() == original.Outcome.to_numpy()).all()
    manifest = sample_split(cohort)
    paths = [Path(__file__).resolve(), ROOT / "lvi_prediction/audit_random120_labels_20261010.py",
        ROOT / "fold_locked_lvi_engine.py", ROOT / "optimized_habitat_backend.py",
        ROOT / "slic_habitat_backend.py", ROOT / "image_preprocessing.py", PROFILE,
        POOL / "cohort_138.csv", POOL / "source_manifest.json", POOL / "provenance.json",
        ROOT / "clinical_data.xlsx", OUT / "clinical_label_audit.json"]
    for aid in cohort.AnalysisID.iloc[118:]:
        path = POOL / "prepared" / aid
        qc = json.loads((path / "complete.json").read_text(encoding="utf-8"))
        assert digest(path / "profile.joblib") == qc["profile_sha256"]
        for item in qc["boundary_qc"]:
            assert digest(Path(item["BoundaryPath"])) == item["BoundarySHA256"]
        paths.extend([path / "profile.joblib", path / "complete.json"])
    protected = set(json.loads((ROOT / "lvi_prediction/results/lvi_nested5_supervised_20261010/preservation_before.json").read_text(encoding="utf-8")))
    protected.update(["AGENTS.md", "ACTIVE_COHORT.json", "lvi_prediction/LVI_RESULT_SELECTION.json", "lvi_prediction/MODEL_DECISION.md"])
    for p in (ROOT / "lvi_prediction/results").rglob("*"):
        if p.is_file() and OUT not in p.parents and "__pycache__" not in p.parts:
            protected.add(str(p.relative_to(ROOT)))
    save(OUT / "preservation_before.json", {name: digest(ROOT / name) for name in sorted(protected)})
    protocol = {"date": "2026-10-10", "authorization": "从138例随机抽120例后重新做7:3训练和测试，作为新实验；同时测试保留的脉管侵犯主模型和ADC模型",
        "source_hashes": {str(p.relative_to(ROOT)): digest(p) for p in paths},
        "model_specifications": MODELS, "manifest": manifest,
        "feature_selection": "retained original definitions; completeness >=.9 and original ICC>=.8; rank |U/(n_pos*n_neg)-.5|*ICC^2; fusion top1 per modality, ADC top3",
        "classifier": "median imputation + StandardScaler + L2 liblinear LR, random_state42, no class weighting",
        "cv": "fivefold training-only validation with full partition and feature selection refit; fixed K/C; no inner hyperparameter search",
        "threshold": "maximum Youden J on training fivefold OOF, ties favor higher threshold",
        "test": "one joint evaluation after both models, thresholds and training outputs are frozen",
        "boundary_ICC_limitation": "Original BoundaryFraction uses the same boundary mask for original/erode/dilate; its ICC does not demonstrate contour robustness. Original definitions retained for method comparison.",
        "clinical_confirmation": "MRI/treatment timing and uniform pathology criteria unknown",
        "sample_count": 120, "source_pool_count": 138, "one_sample_one_split": True,
        "bootstrap": {"replicates": 5000, "seed": BOOTSTRAP_SEED, "conditional_on_fitted_models": True},
        "adoption": "new exploratory experiment; original adoption decisions remain unchanged",
        "excluded_methods_used": False, "external_independent_validation": False}
    save(OUT / "protocol.json", protocol)
    save(OUT / "sample_split_manifest.json", manifest)
    active = json.loads((ROOT / "ACTIVE_COHORT.json").read_text(encoding="utf-8"))
    old_train, old_test = set(active["train_ids"]), set(active["test_ids"])
    excluded_qc = set(pd.read_csv(ROOT / "quality_review_138_v2/excluded_35.csv").AnalysisID)
    member = cohort[["AnalysisID", "Outcome", "SourceGroup"]].copy()
    member["Role"] = ["train" if i in manifest["train"] else "test" if i in manifest["test"] else "not_sampled" for i in range(138)]
    member["Previous118Role"] = ["old_train" if aid in old_train else "old_test" if aid in old_test else "outside_118" for aid in cohort.AnalysisID]
    member["PreviousQCExclusionFlag"] = [aid in excluded_qc for aid in cohort.AnalysisID]
    member.to_csv(OUT / "cohort_membership_138.csv", index=False, encoding="utf-8-sig")
    member[member.Role != "not_sampled"].to_csv(OUT / "sampled_120_membership.csv", index=False, encoding="utf-8-sig")
    (OUT / "models").mkdir(exist_ok=True)
    print("Frozen sample/split before modeling:", json.dumps(manifest["counts"]), flush=True)


def install_runtime():
    engine.assign = slic.assign
    engine._stats = fast.stats
    slic.threadpool_limits = lambda **kw: nullcontext()
    engine.threadpool_limits = lambda **kw: nullcontext()


def load_profiles():
    cohort = pd.read_csv(POOL / "cohort_138.csv")
    profiles = list(joblib.load(PROFILE, mmap_mode="r"))
    for aid in cohort.AnalysisID.iloc[118:]:
        profiles.append(joblib.load(POOL / "prepared" / aid / "profile.joblib", mmap_mode="r"))
    assert len(profiles) == 138
    for p in profiles:
        for mod in engine.MODALITIES:
            assert np.allclose(p[mod]["spacing"], (1., 1., 1.))
    return profiles


def fit_part(train, evaluation, seed, y):
    train, evaluation = np.asarray(train, int), np.asarray(evaluation, int)
    assert set(train) <= TRAIN_ALLOWED and not set(train) & set(evaluation)
    assert set(evaluation) <= TRAIN_ALLOWED  # test is forbidden during training
    assert set(np.unique(y[train])) == {0, 1}
    fitted = {mod: slic.fit_partition(PROFILES, train, mod, 3, seed) for mod in engine.MODALITIES}
    tables, names = fast.tables(PROFILES, fitted, 3, train)
    evaluation_tables = fast.tables(PROFILES, fitted, 3, evaluation)[0] if len(evaluation) else None
    result = {"train": train, "evaluation": evaluation, "partition_seed": seed, "partitions": fitted, "models": {}}
    for label, spec in MODELS.items():
        selected, audit = engine.stable_indices(tables, names["habitat"], spec["modalities"], y[train])
        xa = tables["habitat"][:, selected]
        model = engine.classifier(spec["C"]).fit(xa, y[train]) if selected else None
        pa = model.predict_proba(xa)[:, 1] if model else np.full(len(train), y[train].mean())
        xb = evaluation_tables["habitat"][:, selected] if len(evaluation) else np.empty((0, len(selected)))
        pb = model.predict_proba(xb)[:, 1] if model and len(evaluation) else np.full(len(evaluation), y[train].mean())
        result["models"][label] = {"model": model, "specification": spec, "selected_positions": selected,
            "selected_features": [names["habitat"][j] for j in selected], "feature_names": names["habitat"],
            "feature_audit": audit, "train_x": xa, "evaluation_x": xb, "train_probability": pa,
            "evaluation_probability": pb, "prevalence": float(y[train].mean()),
            "boundary_ICC_is_not_contour_robustness_evidence": True}
    return result


def predict_part(part, evaluation):
    evaluation = np.asarray(evaluation, int)
    assert not set(evaluation) & set(part["train"]) or set(evaluation) <= set(part["train"])
    tables, names = fast.tables(PROFILES, part["partitions"], 3, evaluation)
    result = {}
    for label, item in part["models"].items():
        assert names["habitat"] == item["feature_names"]
        x = tables["habitat"][:, item["selected_positions"]]
        result[label] = item["model"].predict_proba(x)[:, 1] if item["model"] else np.full(len(evaluation), item["prevalence"])
    return result


def threshold(y, probabilities):
    fpr, tpr, values = roc_curve(y, probabilities, drop_intermediate=False)
    allowed = np.flatnonzero(np.isfinite(values) & (values <= 1))
    j = max(allowed, key=lambda i: (float(tpr[i]-fpr[i]), float(values[i])))
    return float(values[j])


def train():
    global PROFILES, TRAIN_ALLOWED
    started = time.time()
    freeze()
    check_sources()
    cohort, manifest, y, tr, te = load_inputs()
    TRAIN_ALLOWED = set(tr)
    PROFILES = load_profiles()
    rows, metrics, sets = [], [], []
    for fold in manifest["folds"]:
        part = fit_part(fold["train"], fold["validation"], fold["partition_seed"], y)
        joblib.dump(part, OUT / f"models/training_fold_{fold['Fold']}.joblib", compress=3)
        idx = part["evaluation"]
        for label, item in part["models"].items():
            p = item["evaluation_probability"]
            rows.extend({"Model": label, "Fold": fold["Fold"], "AnalysisID": cohort.AnalysisID.iloc[i],
                "Outcome": int(y[i]), "Probability": float(v)} for i, v in zip(idx, p))
            metrics.append({"Model": label, "Fold": fold["Fold"], "TrainN": len(part["train"]),
                "ValidationN": len(idx), "AUC": float(roc_auc_score(y[idx], p)),
                "SelectedFeatureCount": len(item["selected_features"]),
                "SelectedFeatures": json.dumps(item["selected_features"])})
            sets.append({"Model": label, "Fold": fold["Fold"], "Features": item["selected_features"]})
        print(f"Training CV fold {fold['Fold']}/5 completed", flush=True)
    oof = pd.DataFrame(rows)
    assert len(oof) == 168 and not oof.duplicated(["Model", "AnalysisID"]).any()
    assert set(oof.AnalysisID) == set(cohort.AnalysisID.iloc[tr])
    assert not set(oof.AnalysisID) & set(cohort.AnalysisID.iloc[te])
    oof.to_csv(OUT / "training_oof_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(metrics).to_csv(OUT / "training_fold_metrics.csv", index=False, encoding="utf-8-sig")
    save(OUT / "training_feature_sets.json", sets)
    final = fit_part(tr, [], manifest["final_partition_seed"], y)
    train_rows, features, summaries = [], [], []
    for label, item in final["models"].items():
        p = item["train_probability"]
        current = oof[oof.Model == label].set_index("AnalysisID").loc[cohort.AnalysisID.iloc[tr]]
        item["threshold"] = threshold(y[tr], current.Probability.to_numpy())
        beta = item["model"]["lr"].coef_[0] if item["model"] else []
        features.extend({"Model": label, "Feature": name, "StandardizedCoefficient": float(b),
            "BoundaryICCNotContourEvidence": "BoundaryFraction" in name}
            for name, b in zip(item["selected_features"], beta))
        train_rows.extend({"Model": label, "AnalysisID": cohort.AnalysisID.iloc[i], "Outcome": int(y[i]),
            "Probability": float(v)} for i, v in zip(tr, p))
        fold_auc = [r["AUC"] for r in metrics if r["Model"] == label]
        summaries.append({"Model": label, "TrainFitAUC": float(roc_auc_score(y[tr], p)),
            "TrainingCVMeanAUC": float(np.mean(fold_auc)), "TrainingCVFoldSD": float(np.std(fold_auc, ddof=1)),
            "TrainingPooledOOFAUC": float(roc_auc_score(y[tr], current.Probability)),
            "TrainingOOFBrier": float(brier_score_loss(y[tr], current.Probability)),
            "SelectedFeatureCount": len(item["selected_features"]), "K": 3, "C": MODELS[label]["C"],
            "FrozenOOFThreshold": item["threshold"]})
    final_path = OUT / "models/final_training_models.joblib"
    joblib.dump(final, final_path, compress=3)
    pd.DataFrame(train_rows).to_csv(OUT / "train_fit_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(features).to_csv(OUT / "selected_features_and_coefficients.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summaries).to_csv(OUT / "training_summary.csv", index=False, encoding="utf-8-sig")
    save(OUT / "final_training_feature_audit.json", {k: v["feature_audit"] for k, v in final["models"].items()})
    protected = check_preserved()
    save(OUT / "training_frozen.json", {"passed": True, "model_sha256": digest(final_path),
        "protocol_sha256": digest(OUT / "protocol.json"), "training_n": 84,
        "test_patient_features_used_during_training": False, "test_labels_masked": True,
        "test_predictions_exist_at_freeze": False, "protected_file_count": protected,
        "training_output_hashes": {p.name: digest(p) for p in OUT.glob("*.csv") if p.name.startswith("train") or p.name.startswith("selected")},
        "fold_model_hashes": {p.name: digest(p) for p in (OUT / "models").glob("training_fold_*.joblib")},
        "elapsed_seconds": time.time()-started})
    save(OUT / "candidate_status.json", {"status": "new_exploratory_experiment_not_adopted", "formal_model_use": False,
        "historical_adoption_decisions_unchanged": True, "excluded_methods_used": False,
        "test_evaluation_completed": False})
    print("Both models and OOF thresholds frozen; no test prediction generated", flush=True)
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)


def score_metrics(y, p, cutoff):
    tn, fp, fn, tp = map(int, confusion_matrix(y, p >= cutoff, labels=[0, 1]).ravel())
    return {"AUC": float(roc_auc_score(y, p)), "AUPRC": float(average_precision_score(y, p)),
        "Brier": float(brier_score_loss(y, p)), "Threshold": float(cutoff),
        "TP": tp, "FN": fn, "TN": tn, "FP": fp, "Sensitivity": tp/(tp+fn),
        "Specificity": tn/(tn+fp), "Accuracy": (tp+tn)/len(y),
        "BalancedAccuracy": .5*(tp/(tp+fn)+tn/(tn+fp)),
        "PPV": tp/(tp+fp) if tp+fp else None, "NPV": tn/(tn+fn) if tn+fn else None}


def bootstrap(y, probabilities):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    pos, neg = int(y.sum()), int((1-y).sum())
    wp = rng.multinomial(pos, np.full(pos, 1/pos), 5000)/pos
    wn = rng.multinomial(neg, np.full(neg, 1/neg), 5000)/neg
    draws = {}
    for label, p in probabilities.items():
        diff = p[y == 1, None]-p[None, y == 0]
        ranks = (diff > 0).astype(float) + .5*(diff == 0)
        draws[label] = np.einsum("bi,ij,bj->b", wp, ranks, wn, optimize=True)
    return draws


def evaluate():
    global PROFILES
    if (OUT / "test_evaluation_started.json").exists():
        raise FileExistsError("The one joint test evaluation has already started; never retry or select a different split")
    check_sources()
    frozen = json.loads((OUT / "training_frozen.json").read_text(encoding="utf-8"))
    model_path = OUT / "models/final_training_models.joblib"
    assert frozen["model_sha256"] == digest(model_path) and frozen["protocol_sha256"] == digest(OUT / "protocol.json")
    for name, h in frozen["training_output_hashes"].items():
        assert digest(OUT / name) == h
    cohort, manifest, y_masked, tr, te = load_inputs()
    assert (y_masked[te] == -1).all()
    final = joblib.load(model_path)
    assert set(final["train"]) == set(tr) and not len(final["evaluation"])
    save(OUT / "test_evaluation_started.json", {"model_sha256": digest(model_path),
        "protocol_sha256": digest(OUT / "protocol.json"), "test_ids": manifest["test_ids"],
        "joint_model_evaluation_count": 1})
    PROFILES = load_profiles()
    probabilities = predict_part(final, te)
    y = cohort.Outcome.to_numpy(int)[te]
    draws = bootstrap(y, probabilities)
    rows, summaries, secondary = [], [], []
    for label, p in probabilities.items():
        item = final["models"][label]
        low, high = np.quantile(draws[label], [.025, .975])
        summaries.append({"Model": label, "TestN": 36, "TestPositive": int(y.sum()), "TestNegative": int((1-y).sum()),
            **score_metrics(y, p, item["threshold"]), "AUC_CI95Low": float(low), "AUC_CI95High": float(high)})
        secondary.append({"Model": label, **score_metrics(y, p, .5)})
        rows.extend({"Model": label, "AnalysisID": cohort.AnalysisID.iloc[i], "Outcome": int(outcome),
            "Probability": float(v), "FrozenThreshold": item["threshold"], "PredictedPositive": int(v >= item["threshold"])}
            for i, outcome, v in zip(te, y, p))
    pd.DataFrame(rows).to_csv(OUT / "test_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summaries).to_csv(OUT / "test_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(secondary).to_csv(OUT / "test_threshold_0p5_metrics.csv", index=False, encoding="utf-8-sig")
    delta = draws["Fusion"]-draws["ADC"]
    low, high = np.quantile(delta, [.025, .975])
    save(OUT / "paired_auc_comparison.json", {"comparison": "Fusion minus ADC on the same 36 test patients",
        "delta_AUC": summaries[0]["AUC"]-summaries[1]["AUC"], "delta_conditional_CI95": [float(low), float(high)],
        "replicates": 5000, "models_not_refitted": True})
    protected = check_preserved()
    save(OUT / "evaluation_complete.json", {"passed": True, "joint_evaluation_count": 1,
        "models": 2, "test_patients": 36, "test_prediction_rows": 72,
        "test_predictions_sha256": digest(OUT / "test_predictions.csv"),
        "test_metrics_sha256": digest(OUT / "test_metrics.csv"), "protected_file_count": protected})
    save(OUT / "candidate_status.json", {"status": "new_exploratory_experiment_not_adopted", "formal_model_use": False,
        "historical_adoption_decisions_unchanged": True, "excluded_methods_used": False,
        "test_evaluation_completed": True, "external_independent_validation": False})
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("train", "evaluate"), required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        install_runtime()
        train() if args.phase == "train" else evaluate()
