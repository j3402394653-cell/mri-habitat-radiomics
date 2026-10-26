# -*- coding: utf-8 -*-
"""Independent frozen probability, CV isolation and two-group uncertainty checks."""
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from verify_k3_stability_improvement90_20261013 import equal, manual, transformed
import run_regional_summary77_two_tests_20261014 as run


def reference_features(indices):
    # Independently slice old derived330 caches, rather than repeat raw1242->330 path.
    aa = joblib.load(run.previous.OUT/'training_features.joblib')
    bb = joblib.load(run.previous.OUT/'test_features.joblib')
    result = {}
    for n in run.NORMS:
        key = (n, 'Invariant')
        a, b = aa[key], bb[key]
        ix = np.r_[a['indices'], b['indices']]
        np.testing.assert_array_equal(np.sort(ix), np.arange(138))
        lookup = {int(i): j for j, i in enumerate(ix)}
        rr = np.asarray([lookup[int(i)] for i in indices])
        result[key] = {'indices': np.asarray(indices, int), 'metadata': a['metadata'],
            **{k: np.concatenate([a[k], b[k]], axis=0)[rr] for k in run.previous.views.KINDS}}
    return result


def features_equal(data, reference, indices):
    for key, d in data.items():
        np.testing.assert_array_equal(d['indices'], indices)
        assert d['metadata'] == reference[key]['metadata'] and len(d['metadata']) == 330
        assert {m['modality'] for m in d['metadata']} == {'T2', 'ADC', 'IVIM'}
        assert not any('Dstar' in m['name'] or 'Dfast' in m['name'] for m in d['metadata'])
        for kind in run.previous.views.KINDS:
            np.testing.assert_array_equal(d[kind], reference[key][kind])


def smoke():
    _, y, tr, te, _ = run.inputs()
    data = run.features(tr)
    features_equal(data, reference_features(tr), tr)
    assert len(run.options()) == 160 and (y[te] == -1).all()
    a, b = run.folds('outer')[0]
    for name, c in run.matched().items():
        f = run.fit(data, a, b, y, c)
        assert f['available']
        np.testing.assert_allclose(manual(f['model'], transformed(f, data[(c['normalization'], 'Invariant')], b)),
                                  f['evaluation_probability'], atol=1e-12, rtol=0)
        print('Region77 smoke '+name+' input'+str(len(f['indices']))+' passed', flush=True)
    print('Region330 feature provenance, fixed templates andtest-label mask passed', flush=True)


def training():
    run.frozen()
    cohort, y, tr, te, _ = run.inputs()
    data = joblib.load(run.OUT/'training_features.joblib')
    features_equal(data, reference_features(tr), tr)
    audit = run.read(run.OUT, 'inner_fold_audit')
    grid = pd.concat([run.read(run.OUT, 'inner_candidate_scores'), run.read(run.OUT, 'final_training_grid').assign(Phase='final')])
    assert len(audit) == 4800 and len(grid) == 960
    assert not audit.duplicated(['Phase', 'InnerFold', 'candidate_id']).any()
    for (phase, cid), rows in audit.groupby(['Phase', 'candidate_id']):
        assert len(rows) == 5
        mean, sd, fit = rows.ValidationAUC.mean(), rows.ValidationAUC.std(ddof=1), rows.FitAUC.mean()
        objective = mean-.10*sd-.30*max(fit-mean-.10, 0) if rows.Available.all() else -1.
        saved = grid[(grid.Phase == phase)&(grid.candidate_id == cid)].iloc[0]
        np.testing.assert_allclose([mean, sd, fit, objective], saved[['MeanValidationAUC', 'SD', 'MeanFitAUC', 'Objective']].to_numpy(float), atol=1e-12, rtol=0)
    nested = run.read(run.OUT, 'nested_predictions')
    assert len(nested) == 385 and not nested.duplicated(['ModelKey', 'AnalysisID']).any()
    refits, value_checks = 0, 0
    for fold, (a, b) in enumerate(run.folds('outer'), 1):
        stored = joblib.load(run.OUT/('models/outer_fold_'+str(fold)+'.joblib'))
        np.testing.assert_array_equal(stored['train'], a)
        np.testing.assert_array_equal(stored['evaluation'], b)
        selected = run.selections(stored['grid'])
        altered = y.copy()
        altered[b] = 1-y[b]
        cache = {}
        for name, f in stored['models'].items():
            c = selected[name]
            assert c == f['config']
            if c['candidate_id'] not in cache:
                cache[c['candidate_id']] = run.fit(data, a, b, altered, c)
                refits += 1
            equal(cache[c['candidate_id']], f)
            rows = nested[(nested.ModelKey == name)&(nested.Fold == fold)].set_index('AnalysisID').loc[cohort.AnalysisID.iloc[b]]
            np.testing.assert_array_equal(rows.Outcome, y[b])
            np.testing.assert_allclose(rows.Probability, f['evaluation_probability'], atol=1e-11, rtol=0)
            if fold == 1 and name != 'Retuned__Adaptive':
                key = (c['normalization'], 'Invariant')
                d = data[key]
                changed = {**d, **{k: d[k].copy() for k in run.previous.views.KINDS}}
                for k in run.previous.views.KINDS:
                    changed[k][run.old.common.rows(d, b)] = 1e12
                equal(run.fit({**data, key: changed}, a, b, altered, c), f, evaluation=False)
                value_checks += 1
        print('Region77 training verification outer'+str(fold)+'/5 passed', flush=True)
    final = joblib.load(run.OUT/'models/final_models.joblib')
    selected = run.selections(run.read(run.OUT, 'final_training_grid').to_dict('records'))
    savedOOF = run.read(run.OUT, 'final_selection_oof_predictions')
    cache = {}
    for name, f in final['models'].items():
        c = selected[name]
        assert c == f['config']
        if c['candidate_id'] not in cache:
            repeated = run.fit(data, tr, [], y, c)
            equal(repeated, f)
            refits += 1
            oof = np.full(138, np.nan)
            for a, b in run.folds('final'):
                oof[b] = run.fit(data, a, b, y, c)['evaluation_probability']
                refits += 1
            cache[c['candidate_id']] = oof
        oof = cache[c['candidate_id']]
        rows = savedOOF[savedOOF.ModelKey == name].set_index('AnalysisID').loc[cohort.AnalysisID.iloc[tr]]
        np.testing.assert_allclose(rows.Probability, oof[tr], atol=1e-11, rtol=0)
        assert np.isnan(oof[te]).all() and run.old.common.threshold(y[tr], oof[tr]) == f['threshold']
        quality = pd.DataFrame(f['audit']).set_index('Feature').loc[f['names']]
        assert quality.Eligible.all() and (quality.ICC >= .8).all() and (quality.CompleteFraction >= .9).all()
    adaptive = final['models']['Retuned__Adaptive']
    equal(adaptive, final['models']['Retuned__'+adaptive['config']['normalization']])
    run.save(run.OUT/'training_verification.json', {'passed': True, 'train_N': 77,
        'old_derived330_matches_new_raw1242_reconstruction': True, 'candidate_objectives_recomputed': 960,
        'inner_fold_records': 4800, 'distinct_classifier_refits': refits, 'heldout_value_checks': value_checks,
        'heldout_labels_and_values_cannot_change_fit': True, 'training_quality_gates_checked': True,
        'finalOOF_and_thresholds_replayed': True, 'test_cases_in_model_fitting': 0, 'test_scored': False,
        'protected_count': run.protect()})
    print('Region77 complete training verification passed', flush=True)


def inference():
    run.frozen()
    cohort, _, tr, te, parts = run.inputs()
    final = joblib.load(run.OUT/'models/final_models.joblib')
    data = joblib.load(run.OUT/'test_features61.joblib')
    features_equal(data, reference_features(te), te)
    predictions = run.read(run.OUT, 'test_predictions')
    metrics = run.read(run.OUT, 'test_metrics').set_index(['ModelKey', 'Group'])
    assert len(predictions) == 305 and not predictions.duplicated(['ModelKey', 'AnalysisID']).any()
    assert len(metrics) == 15
    truth = cohort.Outcome.to_numpy(int)[te]
    membership = np.where(np.isin(te, parts['TestA33']), 'TestA33', 'TestB28')
    probabilities, maximum = {}, 0.
    for name, f in final['models'].items():
        key = (f['config']['normalization'], 'Invariant')
        p = manual(f['model'], transformed(f, data[key], te))
        saved = predictions[predictions.ModelKey == name].set_index('AnalysisID').loc[cohort.AnalysisID.iloc[te]]
        np.testing.assert_array_equal(saved.Outcome, truth)
        np.testing.assert_array_equal(saved.Group, membership)
        np.testing.assert_allclose(saved.Probability, p, atol=1e-12, rtol=0)
        assert (saved.Threshold == f['threshold']).all()
        maximum = max(maximum, float(np.abs(saved.Probability-p).max()))
        probabilities[name] = p
    draws = {}
    for group, seed in run.BOOTSTRAP_SEEDS.items():
        mask = np.ones(61, bool) if group == 'Combined61' else membership == group
        y = truth[mask]
        np_, nn = int(y.sum()), int((y == 0).sum())
        rng = np.random.default_rng(seed)
        wp = rng.multinomial(np_, np.full(np_, 1/np_), 5000)/np_
        wn = rng.multinomial(nn, np.full(nn, 1/nn), 5000)/nn
        draws[group] = {}
        for name, probability in probabilities.items():
            p = probability[mask]
            matrix = np.subtract.outer(p[y == 1], p[y == 0])
            rank = np.where(matrix > 0, 1., np.where(matrix == 0, .5, 0.))
            row = metrics.loc[(name, group)]
            assert row.N == len(y) and row.Positive == np_ and row.Negative == nn
            assert abs(row.AUC-rank.mean()) < 1e-12
            samples = np.sum((wp@rank)*wn, axis=1)
            draws[group][name] = samples
            np.testing.assert_allclose(np.quantile(samples, [.025, .975]), row[['AUC_CI95Low', 'AUC_CI95High']].to_numpy(float), atol=1e-12, rtol=0)
            positive = p >= final['models'][name]['threshold']
            for field, value in {'TP': ((y == 1)&positive).sum(), 'FN': ((y == 1)&~positive).sum(),
                                 'TN': ((y == 0)&~positive).sum(), 'FP': ((y == 0)&positive).sum()}.items():
                assert row[field] == value
    for _, row in run.read(run.OUT, 'retuned_minus_matched_auc').iterrows():
        np.testing.assert_allclose(np.quantile(draws[row.Group][row.First]-draws[row.Group][row.Second], [.025, .975]),
                                  row[['CI95Low', 'CI95High']].to_numpy(float), atol=1e-12, rtol=0)
    for _, row in run.read(run.OUT, 'testA_minus_testB_auc').iterrows():
        np.testing.assert_allclose(np.quantile(draws['TestA33'][row.ModelKey]-draws['TestB28'][row.ModelKey], [.025, .975]),
                                  row[['CI95Low', 'CI95High']].to_numpy(float), atol=1e-12, rtol=0)
    n = final['models']['Retuned__Adaptive']['config']['normalization']
    np.testing.assert_array_equal(probabilities['Retuned__Adaptive'], probabilities['Retuned__'+n])
    assert not set(tr)&set(te)
    complete = json.loads((run.OUT/'test_complete.json').read_text())
    assert complete['distinct_candidate_predict_calls'] == len({f['config']['candidate_id'] for f in final['models'].values()})
    assert complete['model_sha256_after'] == run.digest(run.OUT/'models/final_models.joblib')
    run.save(run.OUT/'inference_verification.json', {'passed': True, 'train_test_overlap': 0,
        'testA_testB_overlap': 0, 'predictions_verified': 305, 'manual_sigmoid_max_error': maximum,
        'group_rank_AUC_CI5000_and_confusion_counts_verified': True,
        'paired_arm_difference_and_unpaired_group_difference_CIs_verified': True,
        'joint_scoring_stages': 1, 'model_refits_after_freeze': 0, 'threshold_changes': 0,
        'adaptive_alias_exact': True, 'protected_count': run.protect()})
    print('Region77 joint61 inference verification passed', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['smoke', 'training', 'inference'], required=True)
    args = p.parse_args()
    with threadpool_limits(limits=1):
        {'smoke': smoke, 'training': training, 'inference': inference}[args.phase]()
