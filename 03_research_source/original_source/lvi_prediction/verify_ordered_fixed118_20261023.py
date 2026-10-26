# -*- coding: utf-8 -*-
"""Feature parity, nested-selection isolation and independent prediction checks."""
import json
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
import run_ordered_fixed118_20261023 as run
import ivim_df63_minroi_guard_20261020 as guard

from verify_ivim_df63_20261020 import auc, draws


def equal(a,b,evaluation=True):
    assert a['config']==b['config'] and a['names']==b['names'] and a['indices']==b['indices']
    assert a['clinical_indices']==b['clinical_indices']
    for field in ['median','low','high','center','scale','train_probability']+(['evaluation_probability'] if evaluation else []):
        np.testing.assert_allclose(a[field],b[field],atol=1e-10,rtol=0)
    if a['model'] is not None:
        attrs = ('support_','support_vectors_','dual_coef_','intercept_') if a['config']['method']=='RBF' else ('coef_','intercept_')
        for attr in attrs:
            np.testing.assert_allclose(getattr(a['model'],attr),getattr(b['model'],attr),atol=1e-10,rtol=0)


def smoke():
    indices=run.study.previous.union_indices('train')
    data=run.study.data(indices)
    count=verify_features(data,indices)
    _,y,_,_=run.study.inputs('Fixed118')
    a,b=run.study.folds('Fixed118','outer')[0]
    fits=0
    for family in run.study.FAMILIES:
        for method in ('Ridge','RBF'):
            c=next(c for c in run.study.options() if c['family']==family and c['method']==method)
            f=run.fit(data,a,b,y,c)
            np.testing.assert_allclose(f['train_probability'],manual(f,data,a),atol=1e-10,rtol=0)
            np.testing.assert_allclose(f['evaluation_probability'],manual(f,data,b),atol=1e-10,rtol=0)
            final=run.fit(data,a,[],y,c)
            assert len(final['evaluation_probability'])==0
            fits+=1
    assert len(run.study.options())==1320
    return {'passed':True,'training_cases':82,'feature_case_blocks_verified':count,
            'LR_RBF_manual_kernel_checks':fits,'empty_evaluation_batches_checked':True,
            'test_scored':False,'Dstar_used':False}


def independent_choose(grid, family, k=None):
    rows = [r for r in grid if r['family'] == family and (k is None or r['K'] == k)]
    return sorted(rows, key=lambda r: (-int(r['AvailableAllFolds']), -r['Objective'],
        r['K'], r['MeanFeatureCount'], r['C'], r['candidate_id']))[0]


def manual(f,data,indices):
    if f['model'] is None:
        return np.full(len(indices),f['prevalence'])
    x=run.transformed(f,data,indices)
    model=f['model']
    if f['config']['method']=='RBF':
        sv=model.support_vectors_
        distances=np.maximum(np.sum(x*x,axis=1)[:,None]+np.sum(sv*sv,axis=1)[None,:]-2*x@sv.T,0.)
        decision=np.exp(-model._gamma*distances)@model.dual_coef_[0]+model.intercept_[0]
        assert list(model.classes_)==[0,1]
        return expit(decision)
    return expit(x@model.coef_[0]+model.intercept_[0])


def verify_features(data,indices):
    reference=run.study.data(indices)
    assert set(data)==set(reference) and len(data)==7
    for key,d in data.items():
        np.testing.assert_array_equal(d['indices'],indices)
        assert d['metadata']==reference[key]['metadata']
        assert all(m['modality'] in ('T2','ADC','IVIM','Clinical') for m in d['metadata'])
        assert not any('Dstar' in m['name'] or 'Dfast' in m['name'] for m in d['metadata'])
        expected=321 if key.endswith('OrderedCompact') else 1242 if key.endswith('OrderedFull') else 402 if key.endswith('WholePhysical') else 11
        assert len(d['metadata'])==expected
        for kind in ('base',) if key=='Clinical' else run.stable.views.KINDS:
            np.testing.assert_array_equal(d[kind],reference[key][kind])
    # Independent reconstruction of two physical-statistics cache rows from maps.
    import run_random120_retained_methods_20261010 as loader
    profiles=loader.load_profiles()
    for i in (int(indices[0]),int(indices[-1])):
        saved=joblib.load(run.study.patient_path(i))
        for kind,maskkey in [('base','roi'),('erode','erode'),('dilate','dilate')]:
            values=[]
            for mod,ch in [('ADC',0),('IVIM',0),('IVIM',1)]:
                p=profiles[i][mod]
                mask=np.asarray(p[maskkey],bool)&p['valid']
                x=run.study.original_values(p,mod)[...,ch][mask]
                q=np.percentile(x,[10,25,50,75,90]);mean=np.mean(x);sd=np.std(x)
                z=(x-mean)/sd if sd>1e-12 else np.zeros(len(x))
                values.extend([mean,sd,*q,q[3]-q[1],np.mean(z**3),np.mean(z**4)])
            np.testing.assert_allclose(saved[kind],values,atol=1e-10,rtol=1e-10)
    return len(indices)*6


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
            assert len(result['grid']) == 1320 and len(audit) == 6600 and audit.index.is_unique
            for row in result['grid']:
                cid = row['candidate_id']
                assert run.study.config(row) == run.study.options()[cid]
                rr = audit.loc[cid]
                for fold, ((a, b), pp) in enumerate(zip(ff, result['cv'][cid]), 1):
                    np.testing.assert_array_equal(a, pp['train']); np.testing.assert_array_equal(b, pp['validation'])
                    assert not (set(a) | set(b)) & set(te)
                    np.testing.assert_allclose([rr.loc[fold].FitAUC, rr.loc[fold].ValidationAUC],
                        [auc(y[a], pp['train_probability']), auc(y[b], pp['validation_probability'])], atol=1e-10, rtol=0)
                    records += 1
                mean, sd, fitmean = rr.ValidationAUC.mean(), rr.ValidationAUC.std(ddof=1), rr.FitAUC.mean()
                objective = mean - .10 * sd - .30 * max(fitmean - mean - .10, 0) if rr.Available.all() else -1.
                np.testing.assert_allclose([mean, sd, fitmean, objective],
                    [row['MeanValidationAUC'], row['SD'], row['MeanFitAUC'], row['Objective']], atol=1e-10, rtol=0)
                checked += 1
            print('Verified all1320 inner candidates ' + group + ' ' + phase, flush=True)
        stored = joblib.load(run.OUT / 'models' / (group + '.joblib'))
        nested = pd.DataFrame(stored['nested_predictions'])
        metrics = pd.DataFrame(stored['outer_metrics'])
        assert len(nested) == 410 and len(metrics) == 25
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
                np.testing.assert_allclose(pp.Probability, manual(f, data, b), atol=1e-10, rtol=0)
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
            np.testing.assert_allclose(seen[cid], f['training_oof'], atol=1e-10, rtol=0, equal_nan=True)
            assert np.isnan(f['training_oof'][te]).all()
            assert run.stable.old.common.threshold(y[tr], f['training_oof'][tr]) == f['threshold']
            pp, mm = nested[nested.ModelKey == name], metrics[metrics.ModelKey == name]
            actual = choice.loc[name]
            np.testing.assert_allclose([actual.TrainFitAUC, actual.OuterMeanAUC, actual.OuterSD, actual.PooledOOFAUC],
                [auc(y[tr], f['train_probability']), mm.AUC.mean(), mm.AUC.std(ddof=1), auc(pp.Outcome, pp.Probability)],
                atol=1e-10, rtol=0)
            if f['image_indices']:
                quality = pd.DataFrame(f['audit']).set_index('Feature').loc[f['names'][:len(f['image_indices'])]]
                assert quality.Eligible.all() and (quality.ICC >= .75).all() and (quality.CompleteFraction >= .9).all()
            assert actual.ImageFeatureCount == len(f['image_indices'])
            assert actual.FeatureCount == actual.ImageFeatureCount + actual.ClinicalFeatureCount
        print('Verified ' + group + ' outer isolation,final refits,OOF,quality andthresholds', flush=True)
    assert checked == 7920 and records == 39600 and perturbations == 5
    run.save(run.OUT / 'training_verification.json', {'passed': True, 'candidate_objectives_checked': checked,
        'inner_candidate_fold_records_independently_rank_verified': records,
        'verification_classifier_refits_before_test': refits, 'heldout_feature_perturbations': perturbations,
        'ordered_feature_case_blocks_verified': k3_count, 'features_only_T2_ADC_IVIM_and_preoperative_clinical': True,
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
            np.testing.assert_allclose(rr.Probability, p, atol=1e-10, rtol=0)
            maximum = max(maximum, float(np.max(np.abs(rr.Probability - p))))
            threshold = models[family]['threshold']
            assert (rr.Threshold == threshold).all()
            m = metrics[(metrics.Group == group) & (metrics.ModelKey == family)].iloc[0]
            np.testing.assert_allclose([m.AUC, m.AUC_CI95Low, m.AUC_CI95High],
                [auc(truth, p), *np.quantile(samples[family], [.025, .975])], atol=1e-10, rtol=0)
            positive = p >= threshold
            np.testing.assert_array_equal(m[['TP', 'FN', 'TN', 'FP']].to_numpy(int),
                [((truth == 1) & positive).sum(), ((truth == 1) & ~positive).sum(),
                 ((truth == 0) & ~positive).sum(), ((truth == 0) & positive).sum()])
        for row in contrasts[contrasts.Group == group].itertuples(index=False):
            np.testing.assert_allclose([row.DeltaAUC, row.CI95Low, row.CI95High],
                [auc(truth, scores[row.First]) - auc(truth, scores[row.Second]),
                 *np.quantile(samples[row.First] - samples[row.Second], [.025, .975])], atol=1e-10, rtol=0)
    run.save(run.OUT / 'inference_verification.json', {'passed': True, 'predictions_verified': 180,
        'manual_sigmoid_max_error': maximum, 'ordered_feature_case_blocks_verified': k3,
        'AUC_confusion_counts_bootstrap5000_and_paired_contrasts_verified': True,
        'scoring_stages': 1, 'test_model_calls': 5, 'fixedK_alternative_test_calls': 0,
        'model_refits_after_test_start': 0, 'threshold_changes': 0, 'protected_files': run.study.protect()})
