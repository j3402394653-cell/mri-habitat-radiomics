# -*- coding: utf-8 -*-
"""Independent rank-AUC, train-fold isolation, adaptive selection andfrozen score audit."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import time
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
import ivim_df63_study_20261020 as study
import run_ivim_df63_20261020 as run
from verify_k3_stability_improvement90_20261013 import equal, transformed


def verify_feature_cache(stage):
    from verify_regional_summary77_two_tests_20261014 import reference_features
    indices = np.asarray(sorted(run.PATIENTS), int)
    reference = reference_features(indices)
    minimum, maximum_difference, audited = 10**12, 0., 0
    for position, index in enumerate(indices):
        row = run.PATIENTS[int(index)]
        assert row['index'] == int(index) and len(row['values']) == 126
        for record in row['partition_audit']:
            sizes = [record['H1'], record['H2'], record['H3']]
            assert sum(sizes) == record['MaskVoxels']
            if 12 in sizes:
                assert record['Exact12VoxelTextureBoundaryApplied']
            minimum = min(minimum, min(sizes))
            a = row['values'][(record['Normalization'], record['ComboID'])][record['Variant']]
            assert len(a) == 156 and np.isfinite(a).sum() == record['FiniteFeatureCount']
            if 12 in sizes:
                assert np.isnan(a[run.minroi.texture_columns()]).all()
            audited += 1
        for norm in study.NORMS:
            d = reference[(norm, 'Invariant')]
            keep = [j for j, meta in enumerate(d['metadata']) if meta['modality'] == 'IVIM']
            for kind, _ in study.feature.KINDS:
                actual = row['values'][(norm, 'C63')][kind]
                expected = d[kind][position, keep]
                np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-9, equal_nan=True)
                maximum_difference = max(maximum_difference, float(np.nanmax(np.abs(actual-expected))))
    run.save(run.OUT/('feature_'+stage+'_verification.json'), {'passed': True, 'patients': len(indices),
        'partition_records_checked': audited, 'minimum_habitat_voxels': int(minimum),
        'all156_full6_values_match_historical': True, 'maximum_difference': maximum_difference,
        'exact12_voxel_boundary_matches_full_extractor': True, 'all_feature_lengths_and_finite_counts_checked': True})


def auc(y, p):
    difference = np.subtract.outer(np.asarray(p)[np.asarray(y) == 1], np.asarray(p)[np.asarray(y) == 0])
    return float(np.mean((difference > 0)+.5*(difference == 0)))


def manual(f, d, indices):
    if f['model'] is None:
        return np.full(len(indices), f['prevalence'])
    x = transformed(f, d, indices)
    return expit(x@f['model'].coef_[0]+f['model'].intercept_[0])


def independent_best(grid, norm=None):
    rows = [r for r in grid if norm is None or r['normalization'] == norm]
    return sorted(rows, key=lambda r: (-int(r['AvailableAllFolds']), -r['Objective'], r.get('DimensionCount', 0),
        r['MeanFeatureCount'], r['C'], r['candidate_id'], int(r.get('ComboID', 'C00')[1:])))[0]


def one(group, combo):
    cohort, y, tr, te = study.inputs(group)
    stored = joblib.load(run.task_path(group, combo))
    data = run.data_for(tr, combo)
    np.testing.assert_array_equal(stored['train'], tr)
    checked, refits, perturbations = 0, 0, 0
    for phase, result in stored['phases'].items():
        ff = study.folds(group, 'final') if phase == 'final' else study.folds(group, 'inner', int(phase[-1]))
        audit = pd.DataFrame(result['audit'])
        assert len(audit) == 240 and not audit.duplicated(['normalization', 'candidate_id', 'InnerFold']).any()
        for row in result['grid']:
            norm, cid = row['normalization'], row['candidate_id']
            records = audit[(audit.normalization == norm)&(audit.candidate_id == cid)]
            assert len(records) == 5 and run.config(row) == study.options(norm)[cid]
            for fold, ((a, b), saved) in enumerate(zip(ff, result['cv'][(norm, cid)]), 1):
                np.testing.assert_array_equal(saved['train'], a)
                np.testing.assert_array_equal(saved['validation'], b)
                assert not (set(a)|set(b))&set(te)
                actual = records[records.InnerFold == fold].iloc[0]
                np.testing.assert_allclose([actual.FitAUC, actual.ValidationAUC],
                    [auc(y[a], saved['train_probability']), auc(y[b], saved['validation_probability'])], atol=1e-12, rtol=0)
            mean, sd, fit = records.ValidationAUC.mean(), records.ValidationAUC.std(ddof=1), records.FitAUC.mean()
            objective = mean-.10*sd-.30*max(fit-mean-.10, 0) if records.Available.all() else -1.
            np.testing.assert_allclose([mean, sd, fit, objective, records.FeatureCount.mean()],
                [row['MeanValidationAUC'], row['SD'], row['MeanFitAUC'], row['Objective'], row['MeanFeatureCount']], atol=1e-12, rtol=0)
            checked += 1
    assert checked == 288
    nested, metrics = pd.DataFrame(stored['nested_predictions']), pd.DataFrame(stored['outer_metrics'])
    assert len(nested) == 140 and not nested.duplicated(['Normalization', 'AnalysisID']).any()
    for fold, (a, b) in enumerate(study.folds(group, 'outer'), 1):
        saved = stored['outer'][fold-1]
        np.testing.assert_array_equal(saved['train'], a)
        np.testing.assert_array_equal(saved['validation'], b)
        altered = y.copy();altered[b] = 1-y[b]
        for norm, f in saved['models'].items():
            chosen = independent_best(stored['phases']['outer'+str(fold)]['grid'], norm)
            assert f['config'] == run.config(chosen)
            repeated = run.stable.fit(data, a, b, altered, f['config'])
            equal(repeated, f);refits += 1
            d = data[(norm, 'Invariant')]
            np.testing.assert_allclose(manual(f, d, b), f['evaluation_probability'], atol=1e-12, rtol=0)
            rows = nested[(nested.Normalization == norm)&(nested.Fold == fold)].set_index('AnalysisID').loc[cohort.AnalysisID.iloc[b]]
            np.testing.assert_array_equal(rows.Outcome, y[b])
            np.testing.assert_allclose(rows.Probability, f['evaluation_probability'], atol=1e-12, rtol=0)
            assert abs(metrics[(metrics.Normalization == norm)&(metrics.Fold == fold)].iloc[0].AUC-auc(y[b], rows.Probability)) < 1e-12
            if fold == 1 and combo in ('C01', 'C06', 'C63'):
                modified = {**d, **{kind: d[kind].copy() for kind, _ in study.feature.KINDS}}
                for kind, _ in study.feature.KINDS:
                    modified[kind][run.stable.old.common.rows(d, b)] = 1e12
                equal(run.stable.fit({**data, (norm, 'Invariant'): modified}, a, b, altered, f['config']), f, evaluation=False)
                perturbations += 1
    choices = pd.DataFrame(stored['choices']).set_index('Normalization')
    for norm, f in stored['final'].items():
        chosen = independent_best(stored['phases']['final']['grid'], norm)
        assert f['config'] == run.config(chosen)
        equal(run.stable.fit(data, tr, [], y, f['config']), f);refits += 1
        oof = np.full(138, np.nan)
        for a, b in study.folds(group, 'final'):
            oof[b] = run.stable.fit(data, a, b, y, f['config'])['evaluation_probability']
            refits += 1
        stored_oof = np.full(138, np.nan)
        for row in stored['phases']['final']['cv'][(norm, f['config']['candidate_id'])]:
            stored_oof[row['validation']] = row['validation_probability']
        np.testing.assert_allclose(oof, stored_oof, atol=1e-12, rtol=0, equal_nan=True)
        assert np.isnan(oof[te]).all() and run.stable.old.common.threshold(y[tr], oof[tr]) == f['threshold']
        row = choices.loc[norm]
        pp = nested[nested.Normalization == norm]
        ff = metrics[metrics.Normalization == norm].AUC
        np.testing.assert_allclose([row.TrainFitAUC, row.OuterMeanAUC, row.OuterSD, row.PooledOOFAUC],
            [auc(y[tr], f['train_probability']), ff.mean(), ff.std(ddof=1), auc(pp.Outcome, pp.Probability)], atol=1e-12, rtol=0)
        if f['indices']:
            quality = pd.DataFrame(f['audit']).set_index('Feature').loc[f['names']]
            assert quality.Eligible.all() and (quality.ICC >= .8).all() and (quality.CompleteFraction >= .9).all()
        else:
            assert not row.AvailableFinal and f['model'] is None
    return {'Group': group, 'ComboID': combo, 'CandidateObjectives': checked, 'Refits': refits, 'Perturbations': perturbations}


def verify_adaptive():
    selections = study.read(run.OUT, 'adaptive_inner_selections')
    predictions = study.read(run.OUT, 'adaptive_nested_predictions')
    aliases = json.loads((run.OUT/'adaptive_frozen_aliases.json').read_text(encoding='utf-8'))
    choices = study.read(run.OUT, 'adaptive_training_choices').set_index(['Group', 'ModelKey'])
    assert len(selections) == 36 and len(predictions) == 420 and len(aliases) == 6
    for group in study.GROUPS:
        cohort, y, tr, te = study.inputs(group)
        loaded = {c['ComboID']: joblib.load(run.task_path(group, c['ComboID'])) for c in study.feature.combinations63()}
        for phase in ['outer'+str(i) for i in range(1, 6)]+['final']:
            grid = [{**row, 'ComboID': c['ComboID'], 'DimensionCount': c['DimensionCount']}
                for c in study.feature.combinations63() for row in loaded[c['ComboID']]['phases'][phase]['grid']]
            for key, norm in [('Adaptive__'+n, n) for n in study.NORMS]+[('Adaptive__BothNorms', None)]:
                expected = independent_best(grid, norm)
                actual = selections[(selections.Group == group)&(selections.ModelKey == key)&(selections.Phase == phase)].iloc[0]
                assert actual.ComboID == expected['ComboID'] and actual.normalization == expected['normalization'] and actual.candidate_id == expected['candidate_id']
                if phase != 'final':
                    fold = int(phase[-1])
                    stored = loaded[actual.ComboID]['outer'][fold-1]
                    rows = predictions[(predictions.Group == group)&(predictions.ModelKey == key)&(predictions.Fold == fold)].set_index('AnalysisID').loc[cohort.AnalysisID.iloc[stored['validation']]]
                    np.testing.assert_allclose(rows.Probability, stored['models'][actual.normalization]['evaluation_probability'], atol=1e-12, rtol=0)
                else:
                    alias = next(a for a in aliases if a['Group'] == group and a['ModelKey'] == key)
                    assert alias['ComboID'] == actual.ComboID and alias['Normalization'] == actual.normalization
        for key in ('Adaptive__PercentileZ', 'Adaptive__RobustZ', 'Adaptive__BothNorms'):
            rows = predictions[(predictions.Group == group)&(predictions.ModelKey == key)]
            ff = [auc(r.Outcome, r.Probability) for _, r in rows.groupby('Fold')]
            row = choices.loc[(group, key)]
            np.testing.assert_allclose([row.OuterMeanAUC, row.OuterSD, row.PooledOOFAUC],
                [np.mean(ff), np.std(ff, ddof=1), auc(rows.Outcome, rows.Probability)], atol=1e-12, rtol=0)


def training():
    run.frozen()
    assert not (run.OUT/'test_started.json').exists()
    study.verify_definition()
    run.save(run.OUT/'minROI_boundary_smoke_verification.json', run.minroi.smoke())
    run.initialize('train')
    verify_feature_cache('train')
    run.PATIENTS = None
    results, start = [], time.time()
    with ProcessPoolExecutor(max_workers=study.WORKERS, initializer=run.initialize) as pool:
        futures = [pool.submit(one, group, c['ComboID']) for group in study.GROUPS for c in study.feature.combinations63()]
        for count, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if count % 6 == 0 or count == 126:
                print('Training verification %d/126 elapsed%.0fs'%(count, time.time()-start), flush=True)
    verify_adaptive()
    run.csv('training_verification_details', results)
    run.save(run.OUT/'training_verification.json', {'passed': True, 'fixed_combos_checked': 126, 'fixed_models_checked': 252,
        'candidate_objectives_checked': int(sum(r['CandidateObjectives'] for r in results)),
        'classifier_refits_before_testing': int(sum(r['Refits'] for r in results)),
        'heldout_feature_perturbations': int(sum(r['Perturbations'] for r in results)),
        'all_inner_fit_and_validation_AUC_independently_rank_recomputed': True,
        'all_outer_label_flip_and_finalOOF_replayed': True, 'quality_gates_checked': True,
        'all36_adaptive_inner_choices_and6_finalaliases_verified': True,
        'test_scored': False, 'protected_count': study.protect(), 'elapsed_seconds': time.time()-start})
    print('Alltraining andnested63-combination selection verified beforetest', flush=True)


def draws(y, probabilities, seed):
    np_, nn = int(y.sum()), int((y == 0).sum())
    rng = np.random.default_rng(seed)
    positive = rng.multinomial(np_, np.full(np_, 1/np_), 5000)/np_
    negative = rng.multinomial(nn, np.full(nn, 1/nn), 5000)/nn
    result = {}
    for key, p in probabilities.items():
        difference = np.subtract.outer(p[y == 1], p[y == 0])
        ranks = (difference > 0).astype(float)+.5*(difference == 0)
        result[key] = np.sum((positive@ranks)*negative, axis=1)
    return result


def inference():
    run.frozen()
    run.initialize('sample')
    verify_feature_cache('all126')
    predictions = study.read(run.OUT, 'test_predictions')
    metrics = study.read(run.OUT, 'test_metrics').set_index(['Group', 'ModelKey'])
    pairs = study.read(run.OUT, 'paired_combo_vs_all6_auc').set_index(['Group', 'Normalization', 'ComboID'])
    aliases = json.loads((run.OUT/'adaptive_frozen_aliases.json').read_text(encoding='utf-8'))
    assert len(predictions) == 7740 and len(metrics) == 258 and len(pairs) == 252
    maximum = 0.
    for offset, group in enumerate(study.GROUPS):
        cohort, _, tr, te = study.inputs(group)
        truth, scores, thresholds = cohort.Outcome.to_numpy(int)[te], {}, {}
        assert not set(tr)&set(te)
        for combo in study.feature.combinations63():
            data = run.data_for(te, combo['ComboID'])
            stored = joblib.load(run.task_path(group, combo['ComboID']))
            for norm, f in stored['final'].items():
                key = norm+'__'+combo['ComboID']
                scores[key] = manual(f, data[(norm, 'Invariant')], te)
                thresholds[key] = f['threshold']
        for a in aliases:
            if a['Group'] == group:
                source = a['Normalization']+'__'+a['ComboID']
                scores[a['ModelKey']], thresholds[a['ModelKey']] = scores[source], thresholds[source]
        samples = draws(truth, scores, 2026102031+offset)
        for key, p in scores.items():
            saved = predictions[(predictions.Group == group)&(predictions.ModelKey == key)].set_index('AnalysisID').loc[cohort.AnalysisID.iloc[te]]
            np.testing.assert_array_equal(saved.Outcome, truth)
            np.testing.assert_allclose(saved.Probability, p, atol=1e-12, rtol=0)
            maximum = max(maximum, float(np.max(np.abs(saved.Probability-p))))
            assert (saved.Threshold == thresholds[key]).all()
            positive = p >= thresholds[key]
            np.testing.assert_array_equal(saved.PredictedPositive, positive)
            row = metrics.loc[(group, key)]
            assert row.N == 30 and row.Positive == truth.sum() and row.Negative == (truth == 0).sum()
            assert abs(row.AUC-auc(truth, p)) < 1e-12
            np.testing.assert_allclose(np.quantile(samples[key], [.025, .975]), row[['AUC_CI95Low', 'AUC_CI95High']].to_numpy(float), atol=1e-12, rtol=0)
            counts = {'TP': ((truth == 1)&positive).sum(), 'FN': ((truth == 1)&~positive).sum(),
                'TN': ((truth == 0)&~positive).sum(), 'FP': ((truth == 0)&positive).sum()}
            for field, n in counts.items():
                assert row[field] == n
            np.testing.assert_allclose([row.Sensitivity, row.Specificity, row.Accuracy],
                [counts['TP']/truth.sum(), counts['TN']/(truth == 0).sum(), np.mean(truth == positive)], atol=1e-12, rtol=0)
        for norm in study.NORMS:
            reference = norm+'__C63'
            for combo in study.feature.combinations63():
                key = norm+'__'+combo['ComboID']
                row = pairs.loc[(group, norm, combo['ComboID'])]
                assert abs(row.DeltaVsAll6-(auc(truth, scores[key])-auc(truth, scores[reference]))) < 1e-12
                np.testing.assert_allclose(np.quantile(samples[key]-samples[reference], [.025, .975]),
                    row[['CI95Low', 'CI95High']].to_numpy(float), atol=1e-12, rtol=0)
        print(group+' all129 frozen scores andpaired all6comparisons verified', flush=True)
    complete = json.loads((run.OUT/'test_complete.json').read_text(encoding='utf-8'))
    assert complete['fixed_model_predict_calls'] == 252 and complete['adaptive_additional_calls'] == 0
    run.save(run.OUT/'inference_verification.json', {'passed': True, 'prediction_rows_checked': 7740, 'metric_rows_checked': 258,
        'manual_probability_max_error': maximum, 'rank_AUC_and5000_patientCIs_checked': True,
        'paired252_combo_vs_all6_CIs_checked': True, 'adaptive_exact_training_selected_aliases': True,
        'test_cases_in_current_group_fit': 0, 'model_refits_after_test_started': 0, 'threshold_changes': 0,
        'protected_count': study.protect()})
