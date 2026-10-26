"""Exact native 25-model outer fold; copied from the unchanged engine.run loop.

The parallel runner checks the first-fold predictions against native smoke outputs.
"""
import numpy as np
from sklearn.metrics import roc_auc_score
from fold_locked_lvi_engine import (K_VALUES, C_VALUES, MODALITIES, MODALITY_SETS,
    fit_partition, feature_tables, select_image, fit_score, augmented)


def fit_outer_fold(profiles, clinical, y, ids, s):
    predictions, choices, selection = [], [], []
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
    return predictions, choices, selection
