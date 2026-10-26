"""118-case LVI comparison with separate reviewed and available-data modes.

The reviewed run requires signed source and boundary review. The exploratory
run uses existing labels and automatic inner-2-mm tumor-periphery masks, and
must never be described as physician-confirmed invasive-front validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "fold_locked_lvi_study"
IDS = ROOT / "corrected_118_cohort/cohort_118.csv"
SPLITS = ROOT / "corrected_118_cohort/split_manifest_118.json"
CLINICAL = ROOT / "clinical_structure_validation/model_input.csv"
MODALITIES = ("T2", "ADC", "IVIM")
REVIEW = OUT / "review_manifest_118.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def records():
    import pandas as pd
    cohort = pd.read_csv(IDS)
    split = json.loads(SPLITS.read_text(encoding="utf-8"))
    if cohort.AnalysisID.tolist() != split["analysis_id_order"]:
        raise ValueError("Cohort and saved validation splits disagree")
    if len(cohort) != 118 or cohort.AnalysisID.duplicated().any():
        raise ValueError("Expected 118 unique retained research IDs")
    return cohort, split


def template():
    cohort, _ = records()
    if REVIEW.exists():
        raise FileExistsError(f"Review template already exists: {REVIEW}")
    OUT.mkdir(parents=True, exist_ok=True)
    old = json.loads((OUT / "review_manifest.json").read_text(encoding="utf-8"))
    old_by_id = {entry["AnalysisID"]: entry for entry in old["patients"]}
    amended = pd_read_csv(ROOT / "corrected_118_cohort/corrected_4_source_audit.csv")
    amended_by_id = {row.AnalysisID: row for row in amended.itertuples(index=False)}
    patients = []
    for row in cohort.itertuples(index=False):
        entry = dict(old_by_id[row.AnalysisID])
        if row.AnalysisID in amended_by_id:
            change = amended_by_id[row.AnalysisID]
            override = dict(entry.get("source_override", {}))
            if change.T2MaskChanged:
                override["t2_mask"] = change.T2MaskPath
            if change.ADCMaskChanged:
                override["adc_mask"] = change.ADCMaskPath
                refit = ROOT / "corrected_118_cohort/refit_ivim_3"
                override.update({"D": str(refit / row.AnalysisID / "D.nii.gz"),
                                 "f": str(refit / row.AnalysisID / "f.nii.gz"),
                                 "NRMSE": str(refit / row.AnalysisID / "NRMSE.nii.gz"),
                                 "ivim_mask": str(refit / "masks" / (row.AnalysisID + ".nii.gz"))})
            entry["source_override"] = override
            entry["source_change_approved"] = True
            entry["notes"] = (f"User reported corrected ROI; old Dice={change.OldDice:.3f}, "
                              f"new Dice={change.CurrentDice:.3f}. Separate clinical/image sign-off remains pending.")
        patients.append(entry)
    review = {
        "schema": 1,
        "note": "118 retained by user decision after four corrected masks; 12 low cross-sequence Dice cases remain temporarily unused, without a proven image-error diagnosis. T2 and ADC were acquired and contoured separately; IVIM uses the ADC-derived ROI. Pathology, image, and physician-boundary sign-off remain pending.",
        "patients": patients,
    }
    REVIEW.write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Created {REVIEW}")


def pd_read_csv(path):
    import pandas as pd
    return pd.read_csv(path)


def audit(review_path: Path, check_geometry: bool = False):
    import numpy as np
    import pandas as pd
    cohort, split = records()
    clinical = pd.read_csv(CLINICAL)
    ids = cohort.AnalysisID.tolist()
    if clinical.AnalysisID.duplicated().any() or not set(ids).issubset(set(clinical.AnalysisID)):
        raise ValueError("Clinical baseline is missing retained research IDs")
    required = ["Clinical_PreopT", "Clinical_PreopN", "Clinical_EMVI",
                "Clinical_MRF", "Clinical_logCEA"]
    if not set(required).issubset(clinical.columns):
        raise ValueError("The five established preoperative predictors are missing")
    if len(split["splits"]) != 50:
        raise ValueError("Expected the existing 10 by 5 outer splits")
    for s in split["splits"]:
        tr, te = set(s["train"]), set(s["test"])
        if tr & te or tr | te != set(range(len(ids))) or len(s["inner"]) != 4:
            raise ValueError("Invalid saved nested split")
    if not review_path.exists():
        return {"ready": False, "patients": len(ids),
                "missing_review_manifest": str(review_path), "pending": {"all": ids}}
    review = json.loads(review_path.read_text(encoding="utf-8"))
    entries = review.get("patients", [])
    if len(entries) != len(ids) or [e.get("AnalysisID") for e in entries] != ids:
        raise ValueError("Review manifest must contain every saved research ID exactly once, in order")
    pending = {}
    for e, original in zip(entries, cohort.itertuples(index=False)):
        aid = e["AnalysisID"]
        reasons = []
        if type(e.get("lvi")) is not int or e["lvi"] not in (0, 1):
            reasons.append("pathology LVI value")
        if not e.get("pathology_confirmed") or not e.get("pathology_reviewer") or not e.get("pathology_review_date"):
            reasons.append("signed pathology review")
        if not e.get("mri_before_treatment_confirmed"):
            reasons.append("MRI before treatment")
        if not e.get("image_and_tumor_roi_confirmed") or not e.get("image_reviewer") or not e.get("image_review_date"):
            reasons.append("signed image/tumor ROI review")
        if not e.get("boundary_confirmed") or not e.get("boundary_reviewer") or not e.get("boundary_review_date"):
            reasons.append("signed invasive-boundary review")
        if e.get("source_override") and not e.get("source_change_approved"):
            reasons.append("approval for changed image sources")
        for key, name in e.get("source_override", {}).items():
            if key not in ("t2", "t2_mask", "adc", "adc_mask", "D", "f", "NRMSE", "ivim_mask"):
                reasons.append(f"unknown source override {key}")
            if not name or not Path(name).is_absolute() or not Path(name).is_file():
                reasons.append(f"valid absolute path for source override {key}")
        for mod in MODALITIES:
            name = e.get("boundary_masks", {}).get(mod, "")
            if not name or not Path(name).is_absolute() or not Path(name).is_file():
                reasons.append(f"{mod} boundary mask")
        if reasons:
            pending[aid] = reasons
    counts = pd.Series([e.get("lvi") for e in entries]).value_counts(dropna=False).to_dict()
    result = {"ready": not pending, "patients": len(ids), "reviewed_label_counts": {str(k): int(v) for k, v in counts.items()},
              "pending_count": len(pending), "pending": pending,
              "source_hashes": {str(p.relative_to(ROOT)): digest(p) for p in (IDS, SPLITS, CLINICAL)}}
    if not pending and check_geometry:
        # Full physical-space checks happen before model fitting, not after an AUC is seen.
        from fold_locked_lvi_engine import prepare_all
        prepare_all(entries, cohort, verify_only=True)
    return result


def exploratory_entries(review_path: Path):
    """Use only the frozen 118 cases, existing LVI labels and automatic ROIs."""
    import pandas as pd
    cohort, _ = records()
    review = json.loads(review_path.read_text(encoding="utf-8"))
    entries = review.get("patients", [])
    if [e.get("AnalysisID") for e in entries] != cohort.AnalysisID.tolist():
        raise ValueError("Review manifest and fixed 118-case cohort disagree")
    qc_path = OUT / "boundary_candidates_inner2mm/candidate_qc.csv"
    qc = pd.read_csv(qc_path)
    expected = {(aid, mod) for aid in cohort.AnalysisID for mod in MODALITIES}
    if len(qc) != len(expected) or set(zip(qc.AnalysisID, qc.Modality)) != expected:
        raise ValueError("Automatic boundary inventory is not exactly 118 x 3")
    lookup = {(r.AnalysisID, r.Modality): r for r in qc.itertuples(index=False)}
    prepared = []
    for e, row in zip(entries, cohort.itertuples(index=False)):
        if int(row.Outcome) not in (0, 1):
            raise ValueError(f"{row.AnalysisID}: invalid existing LVI label")
        item = dict(e)
        item["lvi"] = int(row.Outcome)
        item["boundary_masks"] = {}
        for mod in MODALITIES:
            source = lookup[(row.AnalysisID, mod)]
            path = Path(source.CandidateMask)
            if not path.is_file() or digest(path) != source.CandidateSHA256:
                raise ValueError(f"{row.AnalysisID} {mod}: automatic boundary changed")
            item["boundary_masks"][mod] = str(path)
        prepared.append(item)
    return prepared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "audit", "run", "audit-exploratory", "run-exploratory"))
    parser.add_argument("--review", type=Path, default=REVIEW)
    parser.add_argument("--max-splits", type=int, default=50,
                        help="For a smoke test only; fewer than 50 splits never produces final results")
    args = parser.parse_args()
    if args.command == "init":
        template()
        return
    if args.command.startswith("run"):
        from whole_tumor_runtime import ensure_runtime
        ensure_runtime(("numpy", "pandas", "nibabel", "SimpleITK", "scipy", "sklearn"))
    if args.command.endswith("exploratory"):
        entries = exploratory_entries(args.review)
        if args.command == "audit-exploratory":
            print(json.dumps({"ready_for_exploratory_run": True, "patients": len(entries),
                "label_source": str(IDS), "label_status": "unverified existing 0/1",
                "boundary_status": "automatic inner 2 mm; not physician confirmed",
                "temporarily_excluded": 12, "cross_sequence_Dice_is_exclusion_test": False},
                ensure_ascii=False, indent=2))
            return
        from fold_locked_lvi_engine import run
        run(args.review, max_splits=args.max_splits, exploratory=True, entries=entries)
        return
    report = audit(args.review, check_geometry=args.command == "run")
    if args.command == "audit":
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    if not report["ready"]:
        raise SystemExit(f"Review gate is incomplete for {report['pending_count']} patients; run audit first")
    from fold_locked_lvi_engine import run
    run(args.review, max_splits=args.max_splits)


if __name__ == "__main__":
    main()
