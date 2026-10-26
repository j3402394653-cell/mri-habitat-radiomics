# -*- coding: utf-8 -*-
"""Feature parity, nested-selection isolation and independent prediction checks."""
import json
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
import run_multimodal_fixed118_20261023 as run
import ivim_df63_minroi_guard_20261020 as guard
from verify_k3_stability_improvement90_20261013 import equal
from verify_ivim_df63_20261020 import auc, draws


def smoke():
    indices = run.study.previous.union_indices('train')
    data = run.study.data(indices)
    checked = verify_features(data, indices)
    assert len(run.study.options()) == 1540
    return {'passed': True, 'training_cases': 82, 'IVIM_K3_parity_vectors': checked,
            'habitat_dimension': 330, 'whole_dimension': 372, 'T2_ADC_fixed_K': 3,
            'patient_native_feature_fusion': True, 'Dstar_used': False,
            'models_fitted': 0, 'test_predictions_computed': 0}


def independent_choose(grid, family, k=None):
    rows = [r for r in grid if r['family'] == family and (k is None or r['K'] == k)]
    return sorted(rows, key=lambda r: (-int(r['AvailableAllFolds']), -r['Objective'],
        r['K'], r['MeanFeatureCount'], r['C'], r['candidate_id']))[0]


def manual(f, data, indices):
    if f['model'] is None:
        return np.full(len(indices), f['prevalence'])
    return expit(run.transformed(f, data, indices) @ f['model'].coef_[0] + f['model'].intercept_[0])


def verify_features(data, indices):
    fresh = run.study.data(indices)
    assert set(data) == set(fresh)
    k3_checked = 0
    for key, d in data.items():
        np.testing.assert_array_equal(d['indices'], indices)
        assert d['metadata'] == fresh[key]['metadata']
        assert all(m['modality'] in ('T2', 'ADC', 'IVIM', 'Clinical') for m in d['metadata'])
        assert not any('Dstar' in m['name'] or 'Dfast' in m['name'] for m in d['metadata'])
        for kind in ('base',) if key == 'Clinical' else run.stable.views.KINDS:
            np.testing.assert_array_equal(d[kind], fresh[key][kind])
    from verify_regional_summary77_two_tests_20261014 import reference_features
    reference = reference_features(indices)
    ivim = run.study.prior.data(indices)
    for key, d in data.items():
        if key == 'Clinical':
            np.testing.assert_array_equal(d['base'], ivim['Clinical']['base'])
        elif key.endswith('__Whole'):
            # Independently reconstruct row alignment from the protected372-feature sources.
            norm = key.split('__')[0]
            sources = [joblib.load(run.study.WHOLE / (s + '_features.joblib'))[key]
                       for s in ('training', 'test120', 'omitted18')]
            ix = np.concatenate([p['indices'] for p in sources])
            rr = [np.flatnonzero(ix == i).item() for i in indices]
            assert len(d['metadata']) == 372 and d['metadata'] == sources[0]['metadata']
            for kind in run.stable.views.KINDS:
                np.testing.assert_array_equal(d[kind], np.concatenate([p[kind] for p in sources])[rr])
        else:
            norm = key.split('__')[0]
            original = reference[(norm, 'Invariant')]
            assert len(d['metadata']) == 330 and d['metadata'] == original['metadata']
            other = [j for j,m in enumerate(d['metadata']) if m['modality'] in ('T2','ADC')]
            pos = [j for j,m in enumerate(d['metadata']) if m['modality'] == 'IVIM']
            assert len(other) == 174 and len(pos) == 156
            for kind in run.stable.views.KINDS:
                np.testing.assert_array_equal(d[kind][:,other], original[kind][:,other])
                np.testing.assert_array_equal(d[kind][:,pos], ivim[key][kind])
    for i in indices:
        actual = joblib.load(run.study.patient_path(i))
        prior = guard.canonical(joblib.load(run.study.previous.patient_path(i)))
        assert actual['index'] == int(i) and len(actual['values']) == 30
        for record in actual['partition_audit']:
            assert len(record['HabitatVoxels']) == record['K']
            assert sum(record['HabitatVoxels']) == record['MaskVoxels']
            a = actual['values'][record['ImageKey']][record['Variant']]
            assert len(a) == 156 and np.isfinite(a).sum() == record['FiniteFeatures']
            if record['SmallHabitats']:
                texture_cols = [j for j, m in enumerate(run.study.feature.feature_metadata()) if m['block'] == 'Texture']
                assert np.isnan(a[texture_cols]).all()
            if record['K'] == 3:
                norm, combo, kind = record['Normalization'], record['ComboID'], record['Variant']
                np.testing.assert_allclose(a, prior['values'][(norm, combo)][kind], atol=1e-10, rtol=1e-10, equal_nan=True)
                assert actual['partition_hashes'][(record['ImageKey'], kind)] == prior['partition_hashes'][(norm, combo, kind)]
                k3_checked += 1
    return k3_checked


def training():
    run.frozen()
    assert not (run.OUT / 'test_started.json').exists()
    data = joblib.load(run.OUT / 'training_features.joblib')
    k3_count = verify_features(data, run.study.previous.union_indices('train'))
    checked, records, refits, perturbations = 0, 0, 0, 0
    for group in run.study.GROUPS:
        cohort, y, tr, te = run.study.inputs(group)
        assert len(tr) == 82 and len(te) == 36 and (y[te] == -1).all() and not set(tr) & set(te)
        for phase in ['outer' + str(i) for i in range(1, 6)] + ['final']:
            result = joblib.load(run.phase_path(group, phase))
            ff = run.study.folds(group, 'final') if phase == 'final' else run.study.folds(group, 'inner', int(phase[-1]))
            audit = pd.DataFrame(result['audit']).set_index(['candidate_id', 'InnerFold'])
            assert len(result['grid']) == 1540 and len(audit) == 7700 and audit.index.is_unique
            for row in result['grid']:
                cid = row['candidate_id']
                assert run.study.config(row) == run.study.options()[cid]
                rr = audit.loc[cid]
                for fold, ((a, b), pp) in enumerate(zip(ff, result['cv'][cid]), 1):
                    np.testing.assert_array_equal(a, pp['train']); np.testing.assert_array_equal(b, pp['validation'])
                    assert not (set(a) | set(b)) & set(te)
                    np.testing.assert_allclose([rr.loc[fold].FitAUC, rr.loc[fold].ValidationAUC],
                        [auc(y[a], pp['train_probability']), auc(y[b], pp['validation_probability'])], atol=1e-12, rtol=0)
                    records += 1
                mean, sd, fitmean = rr.ValidationAUC.mean(), rr.ValidationAUC.std(ddof=1), rr.FitAUC.mean()
                objective = mean - .10 * sd - .30 * max(fitmean - mean - .10, 0) if rr.Available.all() else -1.
                np.testing.assert_allclose([mean, sd, fitmean, objective],
                    [row['MeanValidationAUC'], row['SD'], row['MeanFitAUC'], row['Objective']], atol=1e-12, rtol=0)
                checked += 1
            print('Verified all1540 inner candidates ' + group + ' ' + phase, flush=True)
        stored = joblib.load(run.OUT / 'models' / (group + '.joblib'))
        nested = pd.DataFrame(stored['nested_predictions'])
        metrics = pd.DataFrame(stored['outer_metrics'])
        assert len(nested) == 1230 and len(metrics) == 75
        for fold, (a, b) in enumerate(run.study.folds(group, 'outer'), 1):
            outer = stored['outer'][fold - 1]
            np.testing.assert_array_equal(outer['train'], a); np.testing.assert_array_equal(outer['validation'], b)
            grid = joblib.load(run.phase_path(group, 'outer' + str(fold)))['grid']
            altered = y.copy(); altered[b] = 1 - y[b]
            seen = {}
            for name, f in outer['models'].items():
                c = f['config']
                assert c == run.study.config(independent_choose(grid, c['family'], c['K'] if '__K' in name else None))
                cid = c['candidate_id']
                if cid not in seen:
                    seen[cid] = run.fit(data, a, b, altered, c); refits += 1
                equal(seen[cid], f)
                pp = nested[(nested.ModelKey == name) & (nested.Fold == fold)].set_index('AnalysisID').loc[cohort.AnalysisID.iloc[b]]
                np.testing.assert_array_equal(pp.Outcome, y[b])
                np.testing.assert_allclose(pp.Probability, manual(f, data, b), atol=1e-12, rtol=0)
                m = metrics[(metrics.ModelKey == name) & (metrics.Fold == fold)].iloc[0]
                assert abs(m.AUC - auc(y[b], pp.Probability)) < 1e-12
                if fold == 1 and name in run.study.FAMILIES:
                    changed = {key: {**d, **{kind: d[kind].copy() for kind in
                        (('base',) if key == 'Clinical' else run.stable.views.KINDS)}} for key, d in data.items()}
                    for key, d in changed.items():
                        for kind in ('base',) if key == 'Clinical' else run.stable.views.KINDS:
                            d[kind][run.rows(d, b)] = 1e12
                    equal(run.fit(changed, a, b, altered, c), f, evaluation=False); perturbations += 1
        phase = joblib.load(run.phase_path(group, 'final'))
        choice = pd.DataFrame(stored['choices']).set_index('ModelKey')
        seen = {}
        for name, f in stored['final'].items():
            c = f['config']
            assert c == run.study.config(independent_choose(phase['grid'], c['family'], c['K'] if '__K' in name else None))
            cid = c['candidate_id']
            if cid not in seen:
                equal(run.fit(data, tr, [], y, c), f); refits += 1
                oof = np.full(138, np.nan)
                for a, b in run.study.folds(group, 'final'):
                    oof[b] = run.fit(data, a, b, y, c)['evaluation_probability']; refits += 1
                seen[cid] = oof
            np.testing.assert_allclose(seen[cid], f['training_oof'], atol=1e-12, rtol=0, equal_nan=True)
            assert np.isnan(f['training_oof'][te]).all()
            assert run.stable.old.common.threshold(y[tr], f['training_oof'][tr]) == f['threshold']
            pp, mm = nested[nested.ModelKey == name], metrics[metrics.ModelKey == name]
            actual = choice.loc[name]
            np.testing.assert_allclose([actual.TrainFitAUC, actual.OuterMeanAUC, actual.OuterSD, actual.PooledOOFAUC],
                [auc(y[tr], f['train_probability']), mm.AUC.mean(), mm.AUC.std(ddof=1), auc(pp.Outcome, pp.Probability)],
                atol=1e-12, rtol=0)
            if f['image_indices']:
                quality = pd.DataFrame(f['audit']).set_index('Feature').loc[f['names'][:len(f['image_indices'])]]
                assert quality.Eligible.all() and (quality.ICC >= .8).all() and (quality.CompleteFraction >= .9).all()
            assert actual.ImageFeatureCount == len(f['image_indices'])
            assert actual.FeatureCount == actual.ImageFeatureCount + actual.ClinicalFeatureCount
        print('Verified ' + group + ' outer isolation,final refits,OOF,quality andthresholds', flush=True)
    old = joblib.load(run.study.prior.OUT / 'models/Fixed118.joblib')
    current = joblib.load(run.OUT / 'models/Fixed118.joblib')
    equal(old['final']['Clinical'], current['final']['Clinical'])
    for old_fold, new_fold in zip(old['outer'], current['outer']):
        equal(old_fold['models']['Clinical'], new_fold['models']['Clinical'])
    assert checked == 9240 and records == 46200 and perturbations == 5
    run.save(run.OUT / 'training_verification.json', {'passed': True, 'candidate_objectives_checked': checked,
        'inner_candidate_fold_records_independently_rank_verified': records,
        'verification_classifier_refits_before_test': refits, 'heldout_feature_perturbations': perturbations,
        'K3_full_cache_parity_vectors': k3_count, 'features_only_T2_ADC_IVIM_and_preoperative_clinical': True,
        'outer_label_flip_isolation_checked': True, 'complete_inner_K_selection_checked': True,
        'final_OOF_and_thresholds_replayed': True, 'test_scored': False, 'protected_files': run.study.protect()})


def inference():
    run.frozen()
    complete = json.loads((run.OUT / 'test_complete.json').read_text(encoding='utf-8'))
    assert complete['model_predict_calls'] == 5 and complete['fixedK_alternative_test_calls'] == 0
    data = joblib.load(run.OUT / 'all118_features.joblib')
    k3 = verify_features(data, run.study.previous.union_indices('sample'))
    predictions, metrics = run.study.read('test_predictions'), run.study.read('test_metrics')
    contrasts = run.study.read('paired_family_AUC_differences')
    assert len(predictions) == 180 and len(metrics) == 5
    maximum = 0.
    for offset, group in enumerate(run.study.GROUPS):
        cohort, _, tr, te = run.study.inputs(group)
        truth = cohort.Outcome.to_numpy(int)[te]
        models = joblib.load(run.OUT / 'models' / (group + '.joblib'))['final']
        scores = {family: manual(models[family], data, te) for family in run.study.FAMILIES}
        samples = draws(truth, scores, 2026102151 + offset)
        for family, p in scores.items():
            rr = predictions[(predictions.Group == group) & (predictions.ModelKey == family)]
            assert rr.AnalysisID.nunique() == len(rr) == 36
            rr = rr.set_index('AnalysisID').loc[cohort.AnalysisID.iloc[te]]
            np.testing.assert_array_equal(rr.Outcome, truth)
            np.testing.assert_allclose(rr.Probability, p, atol=1e-12, rtol=0)
            maximum = max(maximum, float(np.max(np.abs(rr.Probability - p))))
            threshold = models[family]['threshold']
            assert (rr.Threshold == threshold).all()
            m = metrics[(metrics.Group == group) & (metrics.ModelKey == family)].iloc[0]
            np.testing.assert_allclose([m.AUC, m.AUC_CI95Low, m.AUC_CI95High],
                [auc(truth, p), *np.quantile(samples[family], [.025, .975])], atol=1e-12, rtol=0)
            positive = p >= threshold
            np.testing.assert_array_equal(m[['TP', 'FN', 'TN', 'FP']].to_numpy(int),
                [((truth == 1) & positive).sum(), ((truth == 1) & ~positive).sum(),
                 ((truth == 0) & ~positive).sum(), ((truth == 0) & positive).sum()])
        for row in contrasts[contrasts.Group == group].itertuples(index=False):
            np.testing.assert_allclose([row.DeltaAUC, row.CI95Low, row.CI95High],
                [auc(truth, scores[row.First]) - auc(truth, scores[row.Second]),
                 *np.quantile(samples[row.First] - samples[row.Second], [.025, .975])], atol=1e-12, rtol=0)
    run.save(run.OUT / 'inference_verification.json', {'passed': True, 'predictions_verified': 180,
        'manual_sigmoid_max_error': maximum, 'K3_cache_parity_vectors': k3,
        'AUC_confusion_counts_bootstrap5000_and_paired_contrasts_verified': True,
        'scoring_stages': 1, 'test_model_calls': 5, 'fixedK_alternative_test_calls': 0,
        'model_refits_after_test_start': 0, 'threshold_changes': 0, 'protected_files': run.study.protect()})
