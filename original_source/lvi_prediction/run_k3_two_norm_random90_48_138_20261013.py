"""Prespecified random90/48, two fixed K3 pure habitat nested5/5 models."""
import argparse
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
import run_k23_five_models_120_20261013 as previous

ROOT = previous.ROOT
OUT = ROOT/'lvi_prediction/results/lvi_k3_two_norm_random90_48_138_20261013'
NORMALIZATIONS = ('PercentileZ', 'RobustZ')
SEED = 20261013
KINDS = ('base', 'erode', 'dilate')
origin, old = previous.origin, previous.old
save, digest = previous.save, previous.digest


def options():
    rows = []
    for norm in NORMALIZATIONS:
        for family in ('Descriptor', 'Texture', 'Combined'):
            for top in (3, 6, 9, 12, 15, 0):
                for C in (.03, .3, 1.):
                    rows.append({'normalization': norm, 'K': 3, 'analysis': 'FusionFree',
                                 'family': family, 'top': top, 'quota': 0,
                                 'algorithm': 'LR', 'C': C, 'candidate_id': len(rows)})
    return rows


def config(row):
    return {k: row[k] for k in options()[0]}


def choose(grid, norm):
    rows = [r for r in grid if r['normalization'] == norm]
    valid = [r for r in rows if r['AvailableAllFolds']]
    return max(valid or rows, key=lambda r: (r['Objective'], -r['MeanFeatureCount'], -r['C'], -r['candidate_id']))


def inputs():
    cohort, prior_manifest, _, oldtr, oldte = origin.inputs()
    truth = cohort.Outcome.to_numpy(int)
    tr, te = train_test_split(np.arange(len(cohort)), train_size=90,
                             stratify=truth, random_state=SEED)
    tr, te = np.sort(tr), np.sort(te)
    assert len(cohort) == 138 and len(tr) == 90 and len(te) == 48
    assert not set(tr)&set(te)
    y = truth.copy()
    y[te] = -1
    return cohort, y, tr, te, oldtr, oldte, prior_manifest


def protect():
    protocol = json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    for field in ('protected_hashes', 'source_hashes'):
        for name, sha in protocol[field].items():
            assert digest(ROOT/name) == sha, name
    return len(protocol['protected_hashes'])


def freeze_protocol():
    if OUT.exists():
        raise FileExistsError('Do not overwrite a frozen random-split experiment')
    previous.frozen()
    assert json.loads((previous.OUT/'inference_verification.json').read_text())['passed']
    cohort, y, tr, te, oldtr, oldte, prior_manifest = inputs()
    protected = dict(json.loads((previous.OUT/'protocol.json').read_text(encoding='utf-8'))['protected_hashes'])
    protected.update({str(p.relative_to(ROOT)): digest(p) for p in previous.OUT.rglob('*') if p.is_file()})
    delivery = json.loads((previous.OUT/'delivery_manifest.json').read_text(encoding='utf-8'))
    protected.update(delivery['source_hashes'])
    sources = [Path(__file__).resolve(), ROOT/'lvi_prediction/verify_k3_two_norm_random90_48_20261013.py']
    OUT.mkdir(parents=True)
    (OUT/'models').mkdir()
    cohort.to_csv(OUT/'cohort_138.csv', index=False, encoding='utf-8-sig')
    manifest = {'split_seed': SEED, 'stratified': True, 'train_size': 90, 'test_size': 48,
                'train_indices': tr.tolist(), 'test_indices': te.tolist(),
                'train_ids': cohort.AnalysisID.iloc[tr].tolist(), 'test_ids': cohort.AnalysisID.iloc[te].tolist(),
                'train_positive': int(y[tr].sum()), 'train_negative': int((y[tr] == 0).sum()),
                'test_positive': int(cohort.Outcome.iloc[te].sum()), 'test_negative': int((cohort.Outcome.iloc[te] == 0).sum()),
                'outer_seed': 42, 'inner_seeds': [62201, 62202, 62203, 62204, 62205], 'final_inner_seed': 62212}
    save(OUT/'split_manifest.json', manifest)
    save(OUT/'protocol.json', {
        'date': '2026-10-13', 'authorization': 'human requested new random90 train/48 test from138; added PercentileZ fixedK3 pure habitat comparator to RobustZ fixedK3 pure habitat',
        'normalizations': NORMALIZATIONS, 'K': 3, 'models': 'two pure habitat classifiers; no clinical or whole-tumor predictors',
        'split': manifest, 'split_fixed_before_any_new_model_scores': True,
        'split_scope_override': 'latest90/48 request supersedes previous120/84/36 and frozen54 workflow for this experiment only; preserve earlier policy records',
        'one_split_only': True, 'resplit_to_improve_test_AUC': False,
        'features': 'verified patient-local original K3 caches; independent per-patient normalization and clustering; no pooled fit or outcomes;1242 same-definition candidate features fromT2 ADC IVIM D/f',
        'normalization': 'PercentileZ=P1/P99 clipped ROI meanSD; RobustZ=(signal-median)/(1.482602218505602*MAD), same prespecified IQR thenSD fallback',
        'Dstar_excluded': True, 'clinical_used': False,
        'nested': 'outer5 seed42; inner5 seeds62201..62205; final selection inner5 seed62212; same folds for both norms',
        'candidates': options(), 'candidate_count': 108,
        'gate': 'train-fold ICC>=.8 complete>=.9 stratified-bootstrap sign consistency>=.8;train-ranked absSpearman<.8',
        'preprocessing': 'train-fold median/P1P99winsor/meanSD only',
        'classifier': 'L2 LR liblinear seed64212 maxiter3000 C=.03/.3/1',
        'feature_budget': 'Descriptor/Texture/Combined times3/6/9/12/15/all eligible decorrelated features selected in inner5',
        'objective': 'unchanged mean inner AUC-.10SD-.30max(mean fit-mean validation-.10,0)-.03abs(mean fit-.75)',
        'threshold': 'final selected configuration training90 innerOOF Youden, never test-selected',
        'evaluation': 'verify training and freeze both models; one48 probability batch permodel; derive metrics/bootstrap/paired differences from saved probabilities',
        'train_test_overlap_current_split': 0, 'historical_case_reuse': True, 'new_external_validation': False,
        'automatic_adoption': False, 'formal_model_use': False, 'previous_model_pointer_unchanged': True,
        'sources': ['https://scikit-learn.org/1.3/modules/generated/sklearn.model_selection.train_test_split.html',
                    'https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html'],
        'protected_hashes': protected, 'source_hashes': {str(p.relative_to(ROOT)): digest(p) for p in sources}})
    records = []
    omitted = set(prior_manifest['omitted_indices'])
    for i in range(len(cohort)):
        prior = 'PriorTrain84' if i in oldtr else 'PriorTest36' if i in oldte else 'PriorOmitted18'
        assert (i in omitted) == (prior == 'PriorOmitted18')
        records.append({'AnalysisID': cohort.AnalysisID.iloc[i], 'Outcome': int(cohort.Outcome.iloc[i]),
                        'CurrentGroup': 'Train90' if i in tr else 'Test48', 'Previous120Group': prior})
    pd.DataFrame(records).to_csv(OUT/'split_membership_and_previous_exposure.csv', index=False, encoding='utf-8-sig')
    return cohort, y, tr, te


def cached_features(indices):
    directory = previous.baseline.prior.OUT
    a = joblib.load(directory/'training_features.joblib')
    b = joblib.load(directory/'frozen54_features.joblib')
    data = {}
    for norm in NORMALIZATIONS:
        aa, bb = a[norm][3], b[norm][3]
        assert aa['metadata'] == bb['metadata']
        merged = np.r_[aa['indices'], bb['indices']]
        order = np.argsort(merged)
        np.testing.assert_array_equal(merged[order], np.arange(138))
        assert len(aa['metadata']) == len({m['name'] for m in aa['metadata']}) == 1242
        data[norm] = {'metadata': aa['metadata'], 'indices': np.asarray(indices, int),
                      **{kind: np.concatenate([aa[kind], bb[kind]])[order][indices] for kind in KINDS}}
    return data


def fit(data, a, b, y, c):
    return old.fit(data[c['normalization']], a, b, y, c)


def tune(data, folds, y, label):
    opts = options()
    values, cv, audit = {c['candidate_id']: [] for c in opts}, {c['candidate_id']: [] for c in opts}, []
    for fold, (a, b) in enumerate(folds, 1):
        ranks = {n: old.ranking(data[n], a, y) for n in NORMALIZATIONS}
        fullsets, cache = {}, {}
        for c in opts:
            identity = (c['normalization'], c['family'], c['top'])
            if identity not in cache:
                fullkey = identity[:2]
                if fullkey not in fullsets:
                    fullsets[fullkey] = old.prepare(data[c['normalization']], a, b, {**c, 'top': 0}, y, ranks[c['normalization']])
                cache[identity] = previous.baseline.core.budget_prefix(fullsets[fullkey], c['top'])
            f = old.fitted(cache[identity], y[a], c)
            v = (float(roc_auc_score(y[b], f['evaluation_probability'])), float(roc_auc_score(y[a], f['train_probability'])),
                 len(f['indices']), f['available'])
            values[c['candidate_id']].append(v)
            cv[c['candidate_id']].append((b, f['evaluation_probability']))
            audit.append({'Phase': label, 'InnerFold': fold, 'candidate_id': c['candidate_id'],
                          'ValidationAUC': v[0], 'FitAUC': v[1], 'FeatureCount': v[2], 'Available': v[3]})
        print(label+' inner'+str(fold)+'/5 complete', flush=True)
    grid = []
    for c in opts:
        v = np.asarray(values[c['candidate_id']])
        mean, sd, train, count = float(v[:, 0].mean()), float(v[:, 0].std(ddof=1)), float(v[:, 1].mean()), float(v[:, 2].mean())
        valid = bool(v[:, 3].all())
        score = mean-.10*sd-.30*max(train-mean-.10, 0)-.03*abs(train-.75)
        grid.append({**c, 'MeanValidationAUC': mean, 'SD': sd, 'MeanFitAUC': train,
                     'MeanFeatureCount': count, 'AvailableAllFolds': valid, 'Objective': score if valid else -1.})
    return grid, cv, audit


def train():
    started = time.time()
    cohort, y, tr, te = freeze_protocol()
    data = cached_features(tr)
    joblib.dump(data, OUT/'training_features.joblib', compress=3)
    directory = previous.baseline.prior.OUT
    save(OUT/'image_feature_reuse.json', {'feature_count_per_method': 1242, 'patient_local_no_fitted_cohort_parameters': True,
        'classifier_refitted_with_new90_only': True, 'old_probabilities_not_reused': True,
        'source_cache_hashes': {str((directory/name).relative_to(ROOT)): digest(directory/name)
                               for name in ('training_features.joblib', 'frozen54_features.joblib')}})
    predictions, metrics, grids, inner_audit, frequency = [], [], [], [], []
    for fold, (a, b) in enumerate(old.common.old.previous.folds(tr, y, 42, 5), 1):
        grid, _, audit = tune(data, old.common.old.previous.folds(a, y, 62200+fold, 5), y, 'outer'+str(fold))
        models = {}
        for norm in NORMALIZATIONS:
            c = config(choose(grid, norm))
            f = fit(data, a, b, y, c)
            models[norm] = f
            metrics.append({'Normalization': norm, 'Fold': fold, **c, 'FeatureCount': len(f['indices']),
                            'AUC': float(roc_auc_score(y[b], f['evaluation_probability']))})
            predictions.extend({'Normalization': norm, 'Fold': fold, 'AnalysisID': cohort.AnalysisID.iloc[i],
                                'Outcome': int(y[i]), 'Probability': float(p)} for i, p in zip(b, f['evaluation_probability']))
            frequency.extend({'Normalization': norm, 'Fold': fold, 'Feature': name} for name in f['names'])
        joblib.dump({'train': a, 'evaluation': b, 'grid': grid, 'models': models}, OUT/('models/outer_fold_'+str(fold)+'.joblib'), compress=3)
        grids.extend({'Phase': 'outer'+str(fold), **r} for r in grid)
        inner_audit.extend(audit)
        print('outer'+str(fold)+'/5 complete', flush=True)
    grid, cv, audit = tune(data, old.common.old.previous.folds(tr, y, 62212, 5), y, 'final')
    inner_audit.extend(audit)
    final = {'train': tr, 'models': {}, 'formal_model_use': False}
    choices, features, trainpred, oofpred, gates = [], [], [], [], []
    nested, foldmetrics = pd.DataFrame(predictions), pd.DataFrame(metrics)
    for norm in NORMALIZATIONS:
        row = choose(grid, norm)
        c = config(row)
        f = fit(data, tr, [], y, c)
        oof = np.full(138, np.nan)
        for ix, p in cv[c['candidate_id']]:
            oof[ix] = p
        assert np.isfinite(oof[tr]).all() and np.isnan(oof[te]).all()
        f['threshold'] = old.common.threshold(y[tr], oof[tr])
        final['models'][norm] = f
        pp = nested[nested.Normalization == norm].set_index('AnalysisID').loc[cohort.AnalysisID.iloc[tr]]
        ff = foldmetrics[foldmetrics.Normalization == norm].AUC
        meta = data[norm]['metadata']
        choices.append({**c, 'FeatureCount': len(f['indices']), 'TrainFitAUC': float(roc_auc_score(y[tr], f['train_probability'])),
                        'OuterMeanAUC': float(ff.mean()), 'OuterSD': float(ff.std(ddof=1)),
                        'PooledOOFAUC': float(roc_auc_score(y[tr], pp.Probability)),
                        'FinalSelectionCVMeanAUC': row['MeanValidationAUC'], 'SelectionObjective': row['Objective'],
                        'Threshold': f['threshold'], **{m+'Count': sum(meta[j]['modality'] == m for j in f['indices']) for m in ('T2', 'ADC', 'IVIM')}})
        features.extend({'Normalization': norm, 'Feature': name, 'Modality': meta[j]['modality'], 'Block': meta[j]['block'],
                         'LRStandardizedCoefficient': float(beta)} for j, name, beta in zip(f['indices'], f['names'], f['model'].coef_[0] if f['model'] is not None else []))
        trainpred.extend({'Normalization': norm, 'AnalysisID': cohort.AnalysisID.iloc[i], 'Outcome': int(y[i]), 'Probability': float(p)}
                         for i, p in zip(tr, f['train_probability']))
        oofpred.extend({'Normalization': norm, 'AnalysisID': cohort.AnalysisID.iloc[i], 'Outcome': int(y[i]), 'Probability': float(oof[i])} for i in tr)
        gates.extend({'Normalization': norm, **r} for r in f['audit'])
    joblib.dump(final, OUT/'models/final_models.joblib', compress=3)
    for name, rows in [('nested_predictions', nested), ('outer_fold_metrics', foldmetrics), ('inner_candidate_scores', grids),
                       ('inner_fold_audit', inner_audit), ('final_training_grid', grid), ('final_training_choices', choices),
                       ('selected_features', features), ('train_fit_predictions', trainpred), ('final_selection_oof_predictions', oofpred),
                       ('feature_gate', gates), ('outer_feature_selections', frequency)]:
        pd.DataFrame(rows).to_csv(OUT/(name+'.csv'), index=False, encoding='utf-8-sig')
    save(OUT/'training_frozen.json', {'passed': True, 'model_sha256': digest(OUT/'models/final_models.joblib'),
        'protocol_sha256': digest(OUT/'protocol.json'), 'feature_sha256': digest(OUT/'training_features.joblib'),
        'training_csv_hashes': {p.name: digest(p) for p in OUT.glob('*.csv')},
        'test_scored': False, 'elapsed_seconds': time.time()-started, 'protected_count': protect()})
    print(pd.DataFrame(choices)[['normalization', 'TrainFitAUC', 'OuterMeanAUC', 'OuterSD', 'PooledOOFAUC', 'FeatureCount']].to_string(index=False), flush=True)


def frozen():
    f = json.loads((OUT/'training_frozen.json').read_text(encoding='utf-8'))
    protect()
    for name, field in [('models/final_models.joblib', 'model_sha256'), ('protocol.json', 'protocol_sha256'), ('training_features.joblib', 'feature_sha256')]:
        assert digest(OUT/name) == f[field]
    for name, sha in f['training_csv_hashes'].items():
        assert digest(OUT/name) == sha, name
    return f


def test():
    if (OUT/'test_started.json').exists():
        raise FileExistsError('Do not repeat48 model prediction')
    fr = frozen()
    assert json.loads((OUT/'training_verification.json').read_text())['passed']
    cohort, y, tr, te, _, _, _ = inputs()
    final = joblib.load(OUT/'models/final_models.joblib')
    data = cached_features(te)
    joblib.dump(data, OUT/'test_features.joblib', compress=3)
    save(OUT/'test_started.json', {'n': 48, 'batch_count': 1, 'models': 2,
                                   'model_sha256': fr['model_sha256'], 'training_overlap': 0})
    scores = {n: old.predict(final['models'][n], data[n]) for n in NORMALIZATIONS}
    truth = cohort.Outcome.to_numpy(int)[te]
    draws = old.common.old.retained.bootstrap(truth, scores)
    rows, metrics = [], []
    for norm, p in scores.items():
        threshold = final['models'][norm]['threshold']
        rows.extend({'Normalization': norm, 'AnalysisID': cohort.AnalysisID.iloc[i], 'Outcome': int(v),
                     'Probability': float(pp), 'Threshold': threshold, 'PredictedPositive': int(pp >= threshold)}
                    for i, v, pp in zip(te, truth, p))
        lo, hi = np.quantile(draws[norm], [.025, .975])
        metrics.append({'Normalization': norm, 'N': len(te), 'Positive': int(truth.sum()), 'Negative': int((truth == 0).sum()),
                        **old.common.old.metrics(truth, p, threshold), 'AUC_CI95Low': float(lo), 'AUC_CI95High': float(hi)})
    delta = float(roc_auc_score(truth, scores['PercentileZ'])-roc_auc_score(truth, scores['RobustZ']))
    lo, hi = np.quantile(draws['PercentileZ']-draws['RobustZ'], [.025, .975])
    pd.DataFrame(rows).to_csv(OUT/'test_predictions.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(metrics).to_csv(OUT/'test_metrics.csv', index=False, encoding='utf-8-sig')
    save(OUT/'test_paired_auc_difference.json', {'comparison': 'PercentileZ minus RobustZ', 'N': 48,
        'DeltaAUC': delta, 'CI95Low': float(lo), 'CI95High': float(hi), 'bootstrap_repetitions': 5000,
        'case_paired': True, 'no_model_refitting_in_bootstrap': True})
    save(OUT/'test_complete.json', {'passed': True, 'n': 48, 'prediction_rows': 96, 'model_predict_calls': 2,
        'no_refit_after_freeze': True, 'model_sha256': fr['model_sha256'], 'protected_count': protect()})
    print(pd.DataFrame(metrics).to_string(index=False), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['train', 'test'], required=True)
    args = p.parse_args()
    with threadpool_limits(limits=1):
        {'train': train, 'test': test}[args.phase]()
