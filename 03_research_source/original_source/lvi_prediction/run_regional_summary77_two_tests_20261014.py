# -*- coding: utf-8 -*-
"""Locked77/33/28 regional summary refits: matchedPCA and training-only retuning."""
import argparse
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
import run_k3_stability_improvement90_20261013 as previous
import define_lvi138_two_test_sets_20261014 as split

ROOT = previous.ROOT
OUT = ROOT/'lvi_prediction/results/lvi_regional_summary77_two_tests_20261014'
NORMS = previous.NORMALIZATIONS
save, digest, old = previous.save, previous.digest, previous.old
BOOTSTRAP_SEEDS = {'TestA33': 2026101404, 'TestB28': 2026101405, 'Combined61': 2026101406}


def read(directory, name):
    return pd.read_csv(directory/(name+'.csv'), float_precision='round_trip')


def options():
    rows = [dict(c) for c in previous.options() if c['view'] == 'Invariant']
    for i, c in enumerate(rows):
        c['candidate_id'] = i
    assert len(rows) == 160
    return rows


def config(row):
    return {k: row[k] for k in options()[0]}


def matched():
    choices = read(previous.OUT, 'final_training_choices').set_index('ModelKey')
    result = {}
    for n in NORMS:
        row = choices.loc[n]
        wanted = {k: row[k] for k in options()[0] if k != 'candidate_id'}
        exact = [c for c in options() if all(c[k] == v for k, v in wanted.items())]
        assert len(exact) == 1 and exact[0]['method'] == 'PCA' and exact[0]['components'] == 2 and exact[0]['C'] == .03
        result['Matched__'+n] = exact[0]
    return result


def selections(grid):
    return {**matched(), **{'Retuned__'+n: config(previous.choose(grid, n)) for n in NORMS},
            'Retuned__Adaptive': config(previous.choose(grid))}


def inputs():
    cohort = split.cohort()
    manifest = json.loads((split.OUT/'split_manifest.json').read_text(encoding='utf-8'))
    parts = {g: np.asarray(manifest['groups'][g]['indices'], int) for g in split.GROUPS}
    tr, a, b = [parts[g] for g in split.GROUPS]
    te = np.sort(np.r_[a, b])
    assert len(tr) == 77 and len(a) == 33 and len(b) == 28 and not set(tr)&set(te)
    truth = cohort.Outcome.to_numpy(int)
    y = np.full(len(cohort), -1, int)
    y[tr] = truth[tr]
    return cohort, y, tr, te, parts


def folds(which, outer=None):
    saved = json.loads((split.OUT/'nested5_training_folds.json').read_text(encoding='utf-8'))
    rows = saved['outer_folds'] if which == 'outer' else saved['final_tuning_folds'] if which == 'final' else saved['outer_folds'][outer-1]['inner_folds']
    return [(np.asarray(r['train_indices'], int), np.asarray(r['validation_indices'], int)) for r in rows]


def features(indices):
    # Per-patient caches only; slice current indices before constructing invariant features.
    train = joblib.load(previous.baseline.OUT/'training_features.joblib')
    test = joblib.load(previous.baseline.OUT/'test_features.joblib')
    raw = {}
    for n in NORMS:
        a, b = train[n], test[n]
        assert a['metadata'] == b['metadata']
        pooled = np.r_[a['indices'], b['indices']]
        np.testing.assert_array_equal(np.sort(pooled), np.arange(138))
        lookup = {int(i): j for j, i in enumerate(pooled)}
        rows = np.asarray([lookup[int(i)] for i in indices])
        raw[n] = {'indices': np.asarray(indices, int), 'metadata': a['metadata'],
                  **{k: np.concatenate([a[k], b[k]], axis=0)[rows] for k in previous.views.KINDS}}
    built = previous.views.build(raw)
    return {(n, 'Invariant'): built[(n, 'Invariant')] for n in NORMS}


def protect():
    protocol = json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    for field in ('protected_hashes', 'source_hashes'):
        for name, sha in protocol[field].items():
            assert digest(ROOT/name) == sha, name
    return len(protocol['protected_hashes'])


def freeze_protocol():
    if OUT.exists():
        protect()
        assert not (OUT/'training_frozen.json').exists() and not (OUT/'test_started.json').exists()
        return inputs()
    split.frozen()
    previous.frozen()
    assert json.loads((split.OUT/'split_verification.json').read_text())['passed']
    active = json.loads(split.CURRENT.read_text(encoding='utf-8'))
    assert active['split_manifest_sha256'] == digest(split.OUT/'split_manifest.json')
    protected = dict(json.loads((split.OUT/'split_protocol.json').read_text(encoding='utf-8'))['protected_hashes'])
    protected.update({str(p.relative_to(ROOT)): digest(p) for p in split.OUT.rglob('*') if p.is_file()})
    protected.update(json.loads((split.OUT/'delivery_manifest.json').read_text(encoding='utf-8'))['source_hashes'])
    protected[str(split.CURRENT.relative_to(ROOT))] = digest(split.CURRENT)
    for name in ('run_k3_stability_improvement90_20261013.py', 'habitat_stable_views_20261013.py',
                 'habitat_stable_models_20261013.py', 'regional_habitat_latent_model_20261012.py'):
        protected['lvi_prediction/'+name] = digest(ROOT/'lvi_prediction'/name)
    sources = [Path(__file__).resolve(), ROOT/'lvi_prediction/verify_regional_summary77_two_tests_20261014.py']
    OUT.mkdir(parents=True)
    (OUT/'models').mkdir()
    save(OUT/'protocol.json', {'date': '2026-10-14', 'authorization': 'rerun yesterday regional summary improved models on new77training/33testA/28testB',
        'split_directory': str(split.OUT.relative_to(ROOT)), 'split_manifest_sha256': digest(split.OUT/'split_manifest.json'),
        'unchanged_definition': 'patient-local K3; invariant330 regional summaries; same images,ROIs,normalizations andD/f channels',
        'inputs': ['T2', 'ADC', 'IVIM_D', 'IVIM_f'], 'clinical': False, 'Dstar': False, 'whole_tumor_features': False,
        'new_common_centers': False, 'representation_search': False, 'fixed_view': 'Invariant',
        'matched_arm': matched(), 'retuned_arm': 'same previous invariant-view160 candidate subset; inner5 norm-specific andadaptive selection',
        'candidates': options(), 'candidate_count': 160,
        'quality_and_preprocessing': 'same train-fold ICC>=.8 completeness>=.9,20bootstrap Association orICC Stability,Spearman<.95,medianfill,P2.5/P97.5,median/IQRscale',
        'objective': 'same mean innerAUC-.10SD-.30max(meanfit-meanvalidation-.10,0)',
        'method_change': 'only current training group andfixed invariant representation; classifier retuning distinguished from exact yesterday PCA2/C.03 templates',
        'nested': 'saved train77 outer5seed42,inner62201..62205,final62212; all choices fit onlycurrenttraining fold',
        'thresholds': 'each final configuration train77 OOF Youden; freeze beforetesting',
        'test_stage': 'one joint61 batch perdistinct model; group33/28 andpooled61 metrics fromsamepredictions; no changebetweenAandB',
        'group_bootstrap_seeds': BOOTSTRAP_SEEDS, 'bootstrap_replicates': 5000,
        'A_minus_B': 'difference of separate group-stratified bootstrap draws, independent groupseeds',
        'retuned_minus_matched': 'samecase paired bootstrap pergroup', 'adaptive_final': 'aliasoftraining-chosen retunednorm model; no extra testcalls',
        'historical_comparison': 'saved yesterday90/48 metrics; different groups andtrainingN; descriptive only,notpaired samecases',
        'test_seen_historically': True, 'external_validation': False, 'formal_model_use': False, 'automatic_adoption': False,
        'protected_hashes': protected, 'source_hashes': {str(p.relative_to(ROOT)): digest(p) for p in sources}})
    save(OUT/'split_manifest.json', json.loads((split.OUT/'split_manifest.json').read_text(encoding='utf-8')))
    return inputs()


def fit(data, a, b, y, c):
    return previous.fit(data, a, b, y, c)


def tune(data, ff, y, phase):
    opts = options()
    values, cv, audit = {c['candidate_id']: [] for c in opts}, {c['candidate_id']: [] for c in opts}, []
    for fold, (a, b) in enumerate(ff, 1):
        ranks = {key: previous.ranking(d, a, y) for key, d in data.items()}
        full, prepared, estimates = {}, {}, {}
        for c in opts:
            identity = (c['normalization'], c['view'], c['ranking'], c['top'])
            if identity not in prepared:
                if identity[:3] not in full:
                    key = identity[:2]
                    full[identity[:3]] = previous.prepare(data[key], a, b, y, {**c, 'top': 0}, ranks[key])
                prepared[identity] = previous.prefix(full[identity[:3]], c['top'])
            modelkey = identity[:3]+(tuple(prepared[identity]['indices']), c['method'], c['C'], c['components'])
            if modelkey not in estimates:
                estimates[modelkey] = previous.fitted(prepared[identity], y[a], c, 66413+int(np.sum(a)))
            f = estimates[modelkey]
            v = (float(roc_auc_score(y[b], f['evaluation_probability'])), float(roc_auc_score(y[a], f['train_probability'])),
                 len(f['indices']), f['effective_dimensions'], f['available'])
            values[c['candidate_id']].append(v)
            cv[c['candidate_id']].append((b, f['evaluation_probability']))
            audit.append({'Phase': phase, 'InnerFold': fold, 'candidate_id': c['candidate_id'], 'ValidationAUC': v[0],
                          'FitAUC': v[1], 'FeatureCount': v[2], 'EffectiveDimensions': v[3], 'Available': v[4]})
        print(phase+' inner'+str(fold)+'/5 complete', flush=True)
    grid = []
    for c in opts:
        values_ = np.asarray(values[c['candidate_id']])
        mean, sd, train = values_[:, 0].mean(), values_[:, 0].std(ddof=1), values_[:, 1].mean()
        available = bool(values_[:, 4].all())
        objective = mean-.10*sd-.30*max(train-mean-.10, 0)
        grid.append({**c, 'MeanValidationAUC': float(mean), 'SD': float(sd), 'MeanFitAUC': float(train),
                     'MeanFeatureCount': float(values_[:, 2].mean()), 'MeanEffectiveDimensions': float(values_[:, 3].mean()),
                     'AvailableAllFolds': available, 'Objective': float(objective) if available else -1.})
    return grid, cv, audit


def train():
    started = time.time()
    cohort, y, tr, te, parts = freeze_protocol()
    data = features(tr)
    joblib.dump(data, OUT/'training_features.joblib', compress=3)
    predictions, metrics, grids, audit, frequency = [], [], [], [], []
    for fold, (a, b) in enumerate(folds('outer'), 1):
        path = OUT/('models/outer_fold_'+str(fold)+'.joblib')
        if path.exists():
            stored = joblib.load(path)
            np.testing.assert_array_equal(stored['train'], a)
            np.testing.assert_array_equal(stored['evaluation'], b)
        else:
            grid, _, inner = tune(data, folds('inner', fold), y, 'outer'+str(fold))
            models, cache = {}, {}
            for name, c in selections(grid).items():
                if c['candidate_id'] not in cache:
                    cache[c['candidate_id']] = fit(data, a, b, y, c)
                models[name] = cache[c['candidate_id']]
            stored = {'train': a, 'evaluation': b, 'grid': grid, 'audit': inner, 'models': models}
            joblib.dump(stored, path, compress=3)
        grids.extend({'Phase': 'outer'+str(fold), **r} for r in stored['grid'])
        audit.extend(stored['audit'])
        for name, f in stored['models'].items():
            c = f['config']
            metrics.append({'ModelKey': name, 'Fold': fold, **c, 'FeatureCount': len(f['indices']),
                            'EffectiveDimensions': f['effective_dimensions'], 'AUC': float(roc_auc_score(y[b], f['evaluation_probability']))})
            predictions.extend({'ModelKey': name, 'Fold': fold, 'AnalysisID': cohort.AnalysisID.iloc[i], 'Outcome': int(y[i]),
                                'Probability': float(p)} for i, p in zip(b, f['evaluation_probability']))
            frequency.extend({'ModelKey': name, 'Fold': fold, 'Feature': n} for n in f['names'])
        print('Region77 outer'+str(fold)+'/5 complete', flush=True)
    grid, cv, inner = tune(data, folds('final'), y, 'final')
    audit.extend(inner)
    nested, outer = pd.DataFrame(predictions), pd.DataFrame(metrics)
    final = {'train': tr, 'models': {}, 'formal_model_use': False}
    choices, selected, trainpred, oofpred, cache = [], [], [], [], {}
    for name, c in selections(grid).items():
        cid = c['candidate_id']
        if cid not in cache:
            f = fit(data, tr, [], y, c)
            assert f['available'], 'Final pipeline must be nonconstant andconverged'
            oof = np.full(138, np.nan)
            for ix, pp in cv[cid]:
                oof[ix] = pp
            assert np.isfinite(oof[tr]).all() and np.isnan(oof[te]).all()
            f['threshold'] = old.common.threshold(y[tr], oof[tr])
            cache[cid] = f, oof
        f, oof = cache[cid]
        final['models'][name] = f
        ff = outer[outer.ModelKey == name].AUC
        pp = nested[nested.ModelKey == name].set_index('AnalysisID').loc[cohort.AnalysisID.iloc[tr]]
        row = next(r for r in grid if r['candidate_id'] == cid)
        meta = data[(c['normalization'], 'Invariant')]['metadata']
        choices.append({'ModelKey': name, **c, 'FeatureCount': len(f['indices']), 'EffectiveDimensions': f['effective_dimensions'],
            'TrainFitAUC': float(roc_auc_score(y[tr], f['train_probability'])), 'OuterMeanAUC': float(ff.mean()), 'OuterSD': float(ff.std(ddof=1)),
            'PooledOOFAUC': float(roc_auc_score(y[tr], pp.Probability)), 'FinalSelectionCVMeanAUC': row['MeanValidationAUC'],
            'SelectionObjective': row['Objective'], 'Threshold': f['threshold'],
            **{m+'Count': sum(meta[j]['modality'] == m for j in f['indices']) for m in ('T2', 'ADC', 'IVIM')}})
        beta = f['model'].coefficients_.mean(0) if c['method'] == 'BagRidge' else f['model'].coef_[0]
        selected.extend({'ModelKey': name, 'Feature': feature, 'Modality': meta[j]['modality'], 'Block': meta[j]['block'],
                         'Coefficient': float(b)} for j, feature, b in zip(f['indices'], f['names'], beta))
        trainpred.extend({'ModelKey': name, 'AnalysisID': cohort.AnalysisID.iloc[i], 'Outcome': int(y[i]), 'Probability': float(p)} for i, p in zip(tr, f['train_probability']))
        oofpred.extend({'ModelKey': name, 'AnalysisID': cohort.AnalysisID.iloc[i], 'Outcome': int(y[i]), 'Probability': float(oof[i])} for i in tr)
    joblib.dump(final, OUT/'models/final_models.joblib', compress=3)
    gates = []
    for (n, _), d in data.items():
        _, _, quality = previous.ranking(d, tr, y)
        gates.extend({'Normalization': n, **row} for row in quality)
    for name, rows in [('nested_predictions', nested), ('outer_fold_metrics', outer), ('inner_candidate_scores', grids),
                       ('inner_fold_audit', audit), ('final_training_grid', grid), ('final_training_choices', choices),
                       ('selected_features', selected), ('train_fit_predictions', trainpred), ('final_selection_oof_predictions', oofpred),
                       ('feature_gate', gates), ('outer_feature_selections', frequency)]:
        pd.DataFrame(rows).to_csv(OUT/(name+'.csv'), index=False, encoding='utf-8-sig')
    save(OUT/'training_frozen.json', {'passed': True, 'model_sha256': digest(OUT/'models/final_models.joblib'),
        'protocol_sha256': digest(OUT/'protocol.json'), 'training_features_sha256': digest(OUT/'training_features.joblib'),
        'training_csv_hashes': {p.name: digest(p) for p in OUT.glob('*.csv')}, 'train_N': 77, 'test_scored': False,
        'protected_count': protect(), 'elapsed_seconds_this_run': time.time()-started})
    print(pd.DataFrame(choices)[['ModelKey', 'method', 'FeatureCount', 'TrainFitAUC', 'OuterMeanAUC', 'OuterSD', 'PooledOOFAUC']].to_string(index=False), flush=True)


def frozen():
    fr = json.loads((OUT/'training_frozen.json').read_text(encoding='utf-8'))
    protect()
    for path, field in [('models/final_models.joblib', 'model_sha256'), ('protocol.json', 'protocol_sha256'),
                        ('training_features.joblib', 'training_features_sha256')]:
        assert digest(OUT/path) == fr[field]
    for name, sha in fr['training_csv_hashes'].items():
        assert digest(OUT/name) == sha
    return fr


def bootstrap(y, probabilities, seed):
    rng = np.random.default_rng(seed)
    np_, nn = int(y.sum()), int((y == 0).sum())
    wp = rng.multinomial(np_, np.full(np_, 1/np_), 5000)/np_
    wn = rng.multinomial(nn, np.full(nn, 1/nn), 5000)/nn
    draws = {}
    for name, p in probabilities.items():
        difference = np.subtract.outer(p[y == 1], p[y == 0])
        matrix = (difference > 0).astype(float)+(difference == 0)*.5
        draws[name] = np.einsum('bi,ij,bj->b', wp, matrix, wn, optimize=True)
    return draws


def test():
    assert not (OUT/'test_started.json').exists(), 'Do not repeat joint61 inference'
    fr = frozen()
    assert json.loads((OUT/'training_verification.json').read_text())['passed']
    cohort, _, tr, te, parts = inputs()
    final = joblib.load(OUT/'models/final_models.joblib')
    data = features(te)
    joblib.dump(data, OUT/'test_features61.joblib', compress=3)
    save(OUT/'test_started.json', {'n': 61, 'A': 33, 'B': 28, 'joint_stage': 1, 'model_sha256': fr['model_sha256']})
    scores, cache = {}, {}
    for name, f in final['models'].items():
        cid = f['config']['candidate_id']
        if cid not in cache:
            cache[cid] = previous.predict(f, data[(f['config']['normalization'], 'Invariant')])
        scores[name] = cache[cid]
    truth = cohort.Outcome.to_numpy(int)[te]
    membership = np.where(np.isin(te, parts['TestA33']), 'TestA33', 'TestB28')
    predictions = []
    for name, p in scores.items():
        threshold = final['models'][name]['threshold']
        predictions.extend({'ModelKey': name, 'Group': group, 'AnalysisID': cohort.AnalysisID.iloc[i], 'Outcome': int(y),
            'Probability': float(pp), 'Threshold': threshold, 'PredictedPositive': int(pp >= threshold)} for i, group, y, pp in zip(te, membership, truth, p))
    metrics, paired, groupdraws = [], [], {}
    for group in BOOTSTRAP_SEEDS:
        mask = np.ones(61, bool) if group == 'Combined61' else membership == group
        y, ps = truth[mask], {name: p[mask] for name, p in scores.items()}
        draws = bootstrap(y, ps, BOOTSTRAP_SEEDS[group])
        groupdraws[group] = draws
        for name, p in ps.items():
            low, high = np.quantile(draws[name], [.025, .975])
            metrics.append({'ModelKey': name, 'Group': group, 'N': len(y), 'Positive': int(y.sum()), 'Negative': int((y == 0).sum()),
                **old.common.old.metrics(y, p, final['models'][name]['threshold']), 'BrierScore': float(np.mean((y-p)**2)),
                'AUC_CI95Low': float(low), 'AUC_CI95High': float(high)})
        for n in NORMS:
            first, second = 'Retuned__'+n, 'Matched__'+n
            low, high = np.quantile(draws[first]-draws[second], [.025, .975])
            paired.append({'Group': group, 'First': first, 'Second': second,
                'DeltaAUC': float(roc_auc_score(y, ps[first])-roc_auc_score(y, ps[second])), 'CI95Low': float(low), 'CI95High': float(high)})
    groups = []
    table = pd.DataFrame(metrics).set_index(['ModelKey', 'Group'])
    for name in scores:
        low, high = np.quantile(groupdraws['TestA33'][name]-groupdraws['TestB28'][name], [.025, .975])
        groups.append({'ModelKey': name, 'First': 'TestA33', 'Second': 'TestB28',
            'DeltaAUC': float(table.loc[(name, 'TestA33'), 'AUC']-table.loc[(name, 'TestB28'), 'AUC']),
            'CI95Low': float(low), 'CI95High': float(high), 'Method': 'separate group-stratified bootstrap, independent seeds'})
    for name, rows in [('test_predictions', predictions), ('test_metrics', metrics), ('retuned_minus_matched_auc', paired), ('testA_minus_testB_auc', groups)]:
        pd.DataFrame(rows).to_csv(OUT/(name+'.csv'), index=False, encoding='utf-8-sig')
    save(OUT/'test_complete.json', {'passed': True, 'N': 61, 'prediction_rows': 305, 'joint_scoring_stages': 1,
        'distinct_candidate_predict_calls': len(cache), 'model_refits_after_freeze': 0, 'threshold_changes': 0,
        'adaptive_alias_normalization': final['models']['Retuned__Adaptive']['config']['normalization'],
        'model_sha256_after': digest(OUT/'models/final_models.joblib'), 'protected_count': protect()})
    print(pd.DataFrame(metrics)[['ModelKey', 'Group', 'N', 'AUC', 'AUC_CI95Low', 'AUC_CI95High']].to_string(index=False), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['train', 'test'], required=True)
    args = p.parse_args()
    with threadpool_limits(limits=1):
        {'train': train, 'test': test}[args.phase]()
