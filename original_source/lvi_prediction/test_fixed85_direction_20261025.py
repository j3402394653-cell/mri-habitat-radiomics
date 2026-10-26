# -*- coding: utf-8 -*-
"""User-requested, separately registered holdout evaluation of the frozen T85 model."""
import os
os.environ.update({'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
import argparse
import copy
from datetime import datetime, timezone
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from threadpoolctl import threadpool_limits
import run_original_pls_direction_20261025 as source

ROOT, save, digest = source.ROOT, source.save, source.digest
base = source.base
OUT = ROOT/'lvi_prediction/results/lvi_fixed85_direction_holdout_20261025'
MODEL = 'FixedT85'


def csv(name, rows):
    pd.DataFrame(rows).to_csv(OUT/(name+'.csv'), index=False, encoding='utf-8-sig')


def read(name):
    return pd.read_csv(OUT/(name+'.csv'), float_precision='round_trip')


def protect():
    protocol = json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    for group in ('protected_hashes', 'source_hashes'):
        for name, sha in protocol[group].items():
            assert digest(ROOT/name) == sha, name
    return len(protocol['protected_hashes'])


def frozen():
    protect()
    record = json.loads((OUT/'training_frozen.json').read_text(encoding='utf-8'))
    for name, sha in record['files'].items():
        assert digest(OUT/name) == sha, name
    return record


def define():
    assert not OUT.exists()
    source.frozen()
    for name in ('training_verification', 'delivery_final_verification'):
        assert json.loads((source.OUT/(name+'.json')).read_text(encoding='utf-8'))['passed']
    assert json.loads((source.OUT/'test_complete.json').read_text(encoding='utf-8'))['new_model_calls'] == 0
    cohort, y, tr, te = base.old.inputs()
    assert len(tr) == 84 and len(te) == 36 and set(tr).isdisjoint(te)
    # No re-fit: copy the exact already trained fixed-T85 meta model and its Bag15 bases.
    fit = copy.deepcopy(joblib.load(source.OUT/'phases/final.joblib')['models'][.85])
    assert fit['direction_threshold'] == .85
    np.testing.assert_array_equal(fit['train'], tr)
    nested = source.read('nested_predictions')
    oof = nested[nested.ModelKey == MODEL].set_index('Index').loc[tr]
    np.testing.assert_array_equal(oof.Outcome, y[tr])
    x, z, thresholds = roc_curve(y[tr], oof.Probability, drop_intermediate=False)
    fit['threshold'] = float(thresholds[np.argmax(z-x)])
    fit['threshold_source'] = 'FixedT85_all84_existing_outerOOF_Youden_before_new_test'
    assert np.isfinite(fit['threshold'])
    original = json.loads((source.OUT/'protocol.json').read_text(encoding='utf-8'))
    protected = {**original['protected_hashes'], **original['source_hashes']}
    protected.update({str(p.relative_to(ROOT)):digest(p) for p in source.OUT.rglob('*') if p.is_file()})
    scripts = [ROOT/'lvi_prediction'/n for n in ('test_fixed85_direction_20261025.py', 'evaluate_fixed85_direction_20261025.py')]
    OUT.mkdir(parents=True)
    (OUT/'models').mkdir()
    (OUT/'figures').mkdir()
    save(OUT/'protocol.json', {
        'date':'2026-10-25', 'created_utc':datetime.now(timezone.utc).isoformat(),
        'authorization':'User explicitly requests testing the fixed 85% direction-stability candidate before deciding further improvement.',
        'primary':'Exactly FixedT85 already fitted in the source experiment, not its Adaptive3 selection. No retraining or anchor sweep.',
        'governance':'Separate user-requested exploration supersedes the earlier adaptive fit-floor gate for this evaluation only; source records and formal adoption unchanged.',
        'cohort':'Unchanged A120: 84 training, 36 test. Same five outer folds; all folds/patients retained. No new sampling or split.',
        'model':'Pure habitat T2/ADC/IVIM D/f, k3, only 1mm contour median, two normalizations, 12 PLS bases, Bag15; source meta C.03 anchor.25 signed coefficients.',
        '85_percent_meaning':'At least 34/40 stratified bootstraps agree with feature direction in its current base-training subset; not a prediction-probability cutoff or 85% accuracy.',
        'threshold':fit['threshold'], 'threshold_rule':fit['threshold_source'],
        'training_metrics':'Exact frozen fixed-recipe full-five-fold results; candidate was prioritized after examining source outer diagnostics, not an untouched outer validation of that choice.',
        'test':'One new fixed-T85 evaluation stage on the same historical 36; baseline scores copied. Native inference plus independent coefficient replay for numerical audit only.',
        'uncertainty':'5000 paired patient-level stratified bootstrap samples, fixed scores, seed20261025173; no refit or repeated model selection included.',
        'history':'The 36 cases have been reused in past exploratory work. Not new independent or external validation. No test-driven threshold change, score reversal, or post-test fit.',
        'source_hashes':{str(p.relative_to(ROOT)):digest(p) for p in scripts},
        'protected_hashes':protected,
    })
    joblib.dump({'fit':fit, 'bases':source.basis('final', .85)}, OUT/'models/frozen_fixed85.joblib', compress=3)
    for name in ('training_summary', 'outer_metrics', 'nested_predictions'):
        df = source.read(name)
        csv(name, df[df.ModelKey.isin(['FixedT0', MODEL])])
    csv('split_membership', [{'Index':int(i), 'AnalysisID':cohort.AnalysisID.iloc[i], 'Set':name}
                             for name, indices in [('Training', tr), ('Test', te)] for i in indices])
    paths = {str(p.relative_to(OUT)):digest(p) for p in OUT.rglob('*') if p.is_file()}
    save(OUT/'training_frozen.json', {'passed':True, 'new_test_calls':0, 'retraining_calls':0, 'files':paths})
    print('Frozen exact FixedT85; training-OOF threshold %.15g; no test scores calculated' % fit['threshold'], flush=True)


def test():
    frozen()
    assert not (OUT/'test_started.json').exists()
    assert json.loads((OUT/'training_verification.json').read_text(encoding='utf-8'))['passed']
    cohort, _, tr, te = base.old.inputs()
    checkpoint = joblib.load(OUT/'models/frozen_fixed85.joblib')
    save(OUT/'test_started.json', {'created_utc':datetime.now(timezone.utc).isoformat(),
         'new_candidate_count':1, 'candidate':MODEL, 'retraining_calls':0, 'threshold_frozen':True})
    tables = joblib.load(base.OUT/'test_features.joblib')
    p = base.infer(checkpoint, tables, te)
    manual = base.formula(checkpoint, tables, te)
    np.testing.assert_allclose(p, manual, atol=1e-9, rtol=0)
    truth = cohort.Outcome.to_numpy(int)[te]
    old = base.read('test_predictions')
    old = old[old.ModelKey == 'R1_ContourMedian'].set_index('Index').loc[te]
    np.testing.assert_array_equal(truth, old.Outcome)
    from verify_ivim_df63_20261020 import auc, draws
    samples = draws(truth, {'FixedT85':p, 'Baseline':old.Probability.to_numpy()}, 20261025173)
    metrics = []
    for name, values, threshold in [(MODEL, p, checkpoint['fit']['threshold']),
                                     ('FixedT0', old.Probability.to_numpy(), float(base.read('test_metrics').set_index('ModelKey').loc['R1_ContourMedian','Threshold']))]:
        np.testing.assert_allclose(auc(truth, values), roc_auc_score(truth, values), atol=1e-12, rtol=0)
        lo, hi = np.quantile(samples['FixedT85' if name == MODEL else 'Baseline'], [.025, .975])
        tn, fp, fn, tp = confusion_matrix(truth, values >= threshold, labels=[0,1]).ravel()
        metrics.append({'ModelKey':name, 'TestN':len(te), 'TestPositive':int(truth.sum()), 'TestNegative':int((truth == 0).sum()),
            'TestAUC':roc_auc_score(truth, values), 'AUC_CI95Low':float(lo), 'AUC_CI95High':float(hi), 'Threshold':threshold,
            'Sensitivity':tp/(tp+fn), 'Specificity':tn/(tn+fp), 'Accuracy':(tp+tn)/len(te),
            'TP':int(tp), 'FN':int(fn), 'TN':int(tn), 'FP':int(fp)})
    dl, dh = np.quantile(samples['FixedT85']-samples['Baseline'], [.025, .975])
    save(OUT/'paired_difference.json', {'DifferenceVsBaseline':float(metrics[0]['TestAUC']-metrics[1]['TestAUC']),
         'CI95Low':float(dl), 'CI95High':float(dh), 'bootstrap_samples':5000, 'seed':20261025173})
    csv('test_metrics', metrics)
    csv('test_predictions', [{'Index':int(i), 'AnalysisID':cohort.AnalysisID.iloc[i], 'Outcome':int(t),
         'Fixed85Probability':float(v), 'Fixed85Prediction':int(v >= checkpoint['fit']['threshold']),
         'BaselineProbability':float(w), 'BaselinePrediction':int(c)}
         for i,t,v,w,c in zip(te, truth, p, old.Probability, old.Prediction)])
    frozen()
    save(OUT/'test_complete.json', {'passed':True, 'new_candidate_count':1, 'native_inference_calls':1,
         'independent_formula_replay':1, 'max_formula_error':float(np.max(np.abs(p-manual))),
         'other_new_candidates_tested':0, 'post_test_fits':0, 'post_test_threshold_changes':0,
         'baseline_score_source':'Historical unchanged R1_ContourMedian, no baseline retrain', 'formal_adoption_changed':False})
    print(pd.DataFrame(metrics).to_string(index=False), flush=True)
    print('Paired test-AUC difference %+.4f, 95%% CI [%+.4f, %+.4f]' % (metrics[0]['TestAUC']-metrics[1]['TestAUC'], dl, dh), flush=True)


def seal():
    frozen()
    for name in ('training_verification', 'test_complete', 'report_verification', 'visual_verification'):
        assert json.loads((OUT/(name+'.json')).read_text(encoding='utf-8'))['passed']
    files = {str(p.relative_to(OUT)):digest(p) for p in OUT.rglob('*') if p.is_file()
             and p.name not in ('delivery_manifest.json','delivery_final_verification.json')}
    save(OUT/'delivery_manifest.json', {'outputs':files})
    for name, sha in files.items():
        assert digest(OUT/name) == sha, name
    save(OUT/'delivery_final_verification.json', {'passed':True, 'outputs_checked':len(files), 'only1mm':True,
         'new_test_candidates':1, 'no_retraining':True, 'all5outer_folds_kept':True, 'formal_adoption_changed':False})
    print('FixedT85 exploratory evaluation delivered; original models and adoption unchanged', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('define','test','seal'))
    args = parser.parse_args()
    with threadpool_limits(1):
        globals()[args.action]()
