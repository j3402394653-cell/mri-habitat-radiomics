"""Meaningful representation/fit-isolation and frozen-probability checks."""
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
import run_k3_stability_improvement90_20261013 as run

OUT = run.OUT


def manual(estimator, x):
    if hasattr(estimator, 'coefficients_'):
        return expit(x@estimator.coefficients_.T+estimator.intercepts_).mean(1)
    return expit(x@estimator.coef_[0]+estimator.intercept_[0])


def transformed(f, d, indices):
    x = d['base'][run.old.common.rows(d, indices)][:, f['indices']]
    return (np.clip(np.where(np.isfinite(x), x, f['median']), f['low'], f['high'])-f['center'])/f['scale']


def equal(a, b, evaluation=True):
    assert a['config'] == b['config'] and a['names'] == b['names'] and a['indices'] == b['indices']
    for field in ['median', 'low', 'high', 'center', 'scale', 'train_probability']+(['evaluation_probability'] if evaluation else []):
        np.testing.assert_allclose(a[field], b[field], atol=1e-11, rtol=0)
    if hasattr(a['model'], 'coefficients_'):
        np.testing.assert_allclose(a['model'].coefficients_, b['model'].coefficients_, atol=1e-11, rtol=0)
        np.testing.assert_allclose(a['model'].intercepts_, b['model'].intercepts_, atol=1e-11, rtol=0)
    elif a['model'] is not None:
        np.testing.assert_allclose(a['model'].coef_, b['model'].coef_, atol=1e-11, rtol=0)
        np.testing.assert_allclose(a['model'].intercept_, b['model'].intercept_, atol=1e-11, rtol=0)


def smoke():
    cohort, y, tr, te, _, _, _ = run.baseline.inputs()
    raw = joblib.load(run.baseline.OUT/'training_features.joblib')
    data = run.views.build(raw)
    assert len(run.options()) == 480
    for n in run.NORMALIZATIONS:
        compact, spec = run.views.specifications(raw[n]['metadata'])
        assert len(compact) == 321 and len(spec) == 330
        # Independently permute each three-habitat/pair group; invariant values stay fixed.
        groups = {tuple(s['source_indices']) for s in spec}
        changed = raw[n]['base'].copy()
        for ix in groups:
            changed[:, ix] = changed[:, list(ix)[::-1]]
        np.testing.assert_allclose(run.views.aggregate(changed, spec), data[(n, 'Invariant')]['base'], atol=1e-10, rtol=1e-12, equal_nan=True)
        assert not any('Dstar' in m['name'] or 'Dfast' in m['name'] or m['modality'] == 'Clinical'
                       for m in data[(n, 'Invariant')]['metadata'])
    a, b = run.old.common.old.previous.folds(tr, y, 42, 5)[0]
    for method in ('Ridge', 'ElasticNet', 'PCA', 'PLS', 'BagRidge'):
        c = next(c for c in run.options() if c['normalization'] == 'RobustZ' and c['view'] == 'Compact'
                 and c['ranking'] == 'Association' and c['method'] == method and c['C'] >= .3)
        f = run.fit(data, a, b, y, c)
        x = transformed(f, data[(c['normalization'], c['view'])], b)
        np.testing.assert_allclose(manual(f['model'], x), f['evaluation_probability'], atol=1e-12, rtol=0)
        altered = y.copy()
        altered[b] = 1-y[b]
        equal(f, run.fit(data, a, b, altered, c))
        if method == 'BagRidge':
            assert len(f['model'].members_) == 15
            assert all(np.array_equal(np.bincount(y[a][draw], minlength=2), np.bincount(y[a], minlength=2))
                       for draw in f['model'].draws_)
        print('Smoke '+method+' input'+str(len(f['indices']))+' passed', flush=True)
    print('Invariant permutation,321/330 schemas, five estimator probability formulas and held-out-label isolation passed', flush=True)


def training():
    run.frozen()
    cohort, y, tr, te, _, _, _ = run.baseline.inputs()
    data = joblib.load(OUT/'training_features.joblib')
    reference = run.views.build(joblib.load(run.baseline.OUT/'training_features.joblib'))
    for key, d in data.items():
        np.testing.assert_array_equal(d['indices'], tr)
        assert d['metadata'] == reference[key]['metadata']
        for kind in run.views.KINDS:
            np.testing.assert_array_equal(d[kind], reference[key][kind])
    audit = pd.read_csv(OUT/'inner_fold_audit.csv', float_precision='round_trip')
    grids = pd.concat([pd.read_csv(OUT/'inner_candidate_scores.csv', float_precision='round_trip'),
                      pd.read_csv(OUT/'final_training_grid.csv', float_precision='round_trip').assign(Phase='final')])
    assert len(audit) == 480*30 and len(grids) == 480*6
    assert not audit.duplicated(['Phase', 'InnerFold', 'candidate_id']).any()
    for (phase, cid), v in audit.groupby(['Phase', 'candidate_id']):
        assert sorted(v.InnerFold) == [1, 2, 3, 4, 5]
        mean, sd, fit = v.ValidationAUC.mean(), v.ValidationAUC.std(ddof=1), v.FitAUC.mean()
        score = mean-.10*sd-.30*max(fit-mean-.10, 0) if v.Available.all() else -1.
        row = grids[(grids.Phase == phase)&(grids.candidate_id == cid)].iloc[0]
        np.testing.assert_allclose([mean, sd, fit, v.FeatureCount.mean(), v.EffectiveDimensions.mean(), score],
            row[['MeanValidationAUC', 'SD', 'MeanFitAUC', 'MeanFeatureCount', 'MeanEffectiveDimensions', 'Objective']].to_numpy(float), atol=1e-12, rtol=0)
    nested = pd.read_csv(OUT/'nested_predictions.csv', float_precision='round_trip')
    assert len(nested) == 3*90 and not nested.duplicated(['ModelKey', 'AnalysisID']).any()
    refits = 0
    for fold, (a, b) in enumerate(run.old.common.old.previous.folds(tr, y, 42, 5), 1):
        stored = joblib.load(OUT/('models/outer_fold_'+str(fold)+'.joblib'))
        np.testing.assert_array_equal(stored['train'], a)
        np.testing.assert_array_equal(stored['evaluation'], b)
        chosen = run.selections(stored['grid'])
        chosen.update({n+'__'+v: run.choose(stored['grid'], n, v) for n in run.NORMALIZATIONS for v in run.views.VIEWS})
        changed = y.copy()
        changed[b] = 1-y[b]
        cache = {}
        for name, f in stored['models'].items():
            assert run.config(chosen[name]) == f['config']
            c = f['config']
            if c['candidate_id'] not in cache:
                cache[c['candidate_id']] = run.fit(data, a, b, changed, c)
                refits += 1
            equal(cache[c['candidate_id']], f)
            if name in run.NORMALIZATIONS or name == 'Adaptive':
                rows = nested[(nested.ModelKey == name)&(nested.Fold == fold)].set_index('AnalysisID').loc[cohort.AnalysisID.iloc[b]]
                np.testing.assert_array_equal(rows.Outcome, y[b])
                np.testing.assert_allclose(rows.Probability, f['evaluation_probability'], atol=1e-12, rtol=0)
                x = transformed(f, data[(c['normalization'], c['view'])], b)
                np.testing.assert_allclose(manual(f['model'], x), rows.Probability, atol=1e-12, rtol=0)
        if fold == 1:
            for n in run.NORMALIZATIONS:
                f = stored['models'][n]
                c = f['config']
                d = data[(n, c['view'])]
                altered = {**d, **{kind: d[kind].copy() for kind in run.views.KINDS}}
                for kind in run.views.KINDS:
                    altered[kind][run.old.common.rows(d, b)] = 1e12
                equal(run.fit({**data, (n, c['view']): altered}, a, b, changed, c), f, evaluation=False)
        print('Training verification outer'+str(fold)+'/5 passed', flush=True)
    final = joblib.load(OUT/'models/final_models.joblib')
    grid = grids[grids.Phase == 'final'].to_dict('records')
    choices = pd.read_csv(OUT/'final_training_choices.csv', float_precision='round_trip').set_index('ModelKey')
    inner_oof = pd.read_csv(OUT/'final_selection_oof_predictions.csv', float_precision='round_trip')
    finalfolds = run.old.common.old.previous.folds(tr, y, 62212, 5)
    for name, f in final['models'].items():
        c = f['config']
        assert run.choose(grid, name if name != 'Adaptive' else None)['candidate_id'] == c['candidate_id']
        equal(run.fit(data, tr, [], y, c), f)
        refits += 1
        oof = np.full(138, np.nan)
        for a, b in finalfolds:
            oof[b] = run.fit(data, a, b, y, c)['evaluation_probability']
            refits += 1
        assert np.isfinite(oof[tr]).all() and np.isnan(oof[te]).all()
        rows = inner_oof[inner_oof.ModelKey == name].set_index('AnalysisID').loc[cohort.AnalysisID.iloc[tr]]
        np.testing.assert_allclose(oof[tr], rows.Probability, atol=1e-11, rtol=0)
        assert run.old.common.threshold(y[tr], oof[tr]) == f['threshold']
        assert abs(roc_auc_score(y[tr], f['train_probability'])-choices.loc[name, 'TrainFitAUC']) < 1e-12
        gate = pd.DataFrame(f['audit']).set_index('Feature').loc[f['names']]
        assert gate.Eligible.all() and (gate.ICC >= .8).all() and (gate.CompleteFraction >= .9).all()
    joint = final['models']['Adaptive']
    equal(joint, final['models'][joint['config']['normalization']])
    run.save(OUT/'training_verification.json', {'passed': True, 'candidate_objectives_recomputed': len(grids),
        'inner_fold_audit_rows': len(audit), 'selected_and_final_inner_models_refitted': refits,
        'quality_gates_checked': True, 'transformed_feature_caches_exact': True,
        'heldout_labels_and_values_cannot_change_fit': True, 'finalOOF_thresholds_replayed': True,
        'representation_method_and_norm_choices_inside_innerCV': True, 'test_scored': False, 'protected_count': run.protect()})
    print('Training verification passed', flush=True)


def inference():
    run.frozen()
    cohort, _, tr, te, _, _, _ = run.baseline.inputs()
    final = joblib.load(OUT/'models/final_models.joblib')
    data = joblib.load(OUT/'test_features.joblib')
    pred = pd.read_csv(OUT/'test_predictions.csv', float_precision='round_trip')
    metric = pd.read_csv(OUT/'test_metrics.csv', float_precision='round_trip').set_index('ModelKey')
    assert len(pred) == 3*48 and not pred.duplicated(['ModelKey', 'AnalysisID']).any()
    y = cohort.Outcome.to_numpy(int)[te]
    probabilities, maximum = {}, 0.
    for name, f in final['models'].items():
        c = f['config']
        d = data[(c['normalization'], c['view'])]
        np.testing.assert_array_equal(d['indices'], te)
        p = manual(f['model'], transformed(f, d, te))
        rows = pred[pred.ModelKey == name].set_index('AnalysisID').loc[cohort.AnalysisID.iloc[te]]
        np.testing.assert_array_equal(rows.Outcome, y)
        error = float(np.max(np.abs(rows.Probability-p)))
        assert error < 1e-12
        maximum = max(maximum, error)
        assert (rows.Threshold == f['threshold']).all()
        probabilities[name] = p
    previous = pd.read_csv(run.baseline.OUT/'test_predictions.csv', float_precision='round_trip')
    for n in run.NORMALIZATIONS:
        probabilities['Baseline__'+n] = previous[previous.Normalization == n].set_index('AnalysisID').loc[cohort.AnalysisID.iloc[te]].Probability.to_numpy()
    rng = np.random.default_rng(run.old.common.old.retained.BOOTSTRAP_SEED)
    np_, nn = int(y.sum()), int((y == 0).sum())
    wp = rng.multinomial(np_, np.full(np_, 1/np_), 5000)/np_
    wn = rng.multinomial(nn, np.full(nn, 1/nn), 5000)/nn
    draws = {}
    for name, p in probabilities.items():
        diff = np.subtract.outer(p[y == 1], p[y == 0])
        matrix = np.where(diff > 0, 1., np.where(diff == 0, .5, 0.))
        draws[name] = np.sum((wp@matrix)*wn, axis=1)
        if name in final['models']:
            row = metric.loc[name]
            assert abs(row.AUC-matrix.mean()) < 1e-12
            np.testing.assert_allclose(np.quantile(draws[name], [.025, .975]), row[['AUC_CI95Low', 'AUC_CI95High']].to_numpy(float), atol=1e-12, rtol=0)
            classified = p >= final['models'][name]['threshold']
            for s, v in {'TP': ((y == 1)&classified).sum(), 'FN': ((y == 1)&~classified).sum(),
                         'TN': ((y == 0)&~classified).sum(), 'FP': ((y == 0)&classified).sum()}.items():
                assert row[s] == v
    pairs = pd.read_csv(OUT/'test_paired_auc_comparisons.csv', float_precision='round_trip')
    for _, r in pairs.iterrows():
        np.testing.assert_allclose(np.quantile(draws[r.First]-draws[r.Second], [.025, .975]),
                                  r[['CI95Low', 'CI95High']].to_numpy(float), atol=1e-12, rtol=0)
    jointnorm = final['models']['Adaptive']['config']['normalization']
    np.testing.assert_array_equal(probabilities['Adaptive'], probabilities[jointnorm])
    assert not set(tr)&set(te)
    complete = json.loads((OUT/'test_complete.json').read_text())
    assert complete['model_predict_calls'] == 2 and complete['baseline_model_predict_calls'] == 0
    run.save(OUT/'inference_verification.json', {'passed': True, 'manual_probability_max_error': maximum,
        'AUC_CI5000_confusion_counts_and_pairedCI_independently_checked': True,
        'adaptive_alias_exact': True, 'training_overlap': 0, 'model_refits_after_freeze': 0,
        'baseline_comparators_read_only': True, 'original_predict_calls': 2, 'protected_count': run.protect()})
    print('Inference verification passed', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['smoke', 'training', 'inference'], required=True)
    args = p.parse_args()
    with threadpool_limits(limits=1):
        {'smoke': smoke, 'training': training, 'inference': inference}[args.phase]()
