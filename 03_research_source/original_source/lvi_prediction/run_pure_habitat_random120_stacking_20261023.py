# -*- coding: utf-8 -*-
"""Fixed cross-fitted PLS stack on the same one-draw cohort, before any holdout score."""
import os
os.environ.update({'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from threadpoolctl import threadpool_limits
import run_pure_habitat_random120_20261023 as parent

ROOT, save, digest = parent.ROOT, parent.save, parent.digest
OUT = ROOT/'lvi_prediction/results/lvi_pure_habitat_random120_stacking_20261023'
GROUPS = parent.GROUPS


def csv(name, rows):
    pd.DataFrame(rows).to_csv(OUT/(name+'.csv'), index=False, encoding='utf-8-sig')


def read(name):
    return pd.read_csv(OUT/(name+'.csv'), float_precision='round_trip', keep_default_na=False, na_values=[''])


def members():
    result = [c for c in parent.method.reference_options() if c['method'] == 'PLS' and c['C'] == .03]
    assert len(result) == 12
    return result


def checkpoint(group, phase):
    return joblib.load(parent.previous.OUT/'phases'/(phase+'.joblib')) if group == 'Original118' else joblib.load(parent.OUT/group/'phases'/(phase+'.joblib'))


def crossfit_features(group, phase):
    _, _, allowed, _ = parent.inputs(group)
    if phase != 'final':
        allowed = parent.folds(group)[int(phase[-1])-1][0]
    allowed = np.asarray(allowed, int)
    cp = checkpoint(group, phase)
    x = np.full((len(allowed), 3, 12), np.nan)
    lookup = {int(i): j for j, i in enumerate(allowed)}
    for col, c in enumerate(members()):
        key = c['candidate_id'] if group == 'Original118' else c['CandidateKey']
        for row in cp['cv'][key]:
            assert set(row['validation']) <= set(allowed)
            x[[lookup[int(i)] for i in row['validation']], row['Repeat']-1, col] = row['validation_score']
    assert np.isfinite(x).all()
    return allowed, x.mean(1), x


def meta_prepare(x, xb):
    low, high = np.quantile(x, [.025, .975], axis=0)
    z = np.clip(x, low, high)
    center, scale = z.mean(0), z.std(0)
    scale = np.where(scale > 1e-12, scale, 1.)
    return {'low': low, 'high': high, 'center': center, 'scale': scale,
        'xa': (z-center)/scale, 'xb': (np.clip(xb, low, high)-center)/scale}


def fit(group, phase, tables, a, b, y):
    allowed, oof, repeated = crossfit_features(group, phase)
    np.testing.assert_array_equal(allowed, a)
    base = [parent.method.fit(tables, a, b, y, c) for c in members()]
    pa = np.column_stack([f['train_probability'] for f in base])
    pb = np.column_stack([f['evaluation_probability'] for f in base])
    pp = meta_prepare(oof, pb)
    meta = LogisticRegression(C=.03, penalty='l2', solver='liblinear', class_weight='balanced', max_iter=3000, random_state=64212).fit(pp['xa'], y[a])
    za = (np.clip(pa, pp['low'], pp['high'])-pp['center'])/pp['scale']
    return {'group': group, 'phase': phase, 'train': a, 'validation': b, 'base': base, 'meta': meta,
        **{k: pp[k] for k in ('low', 'high', 'center', 'scale')}, 'crossfit_base_scores': oof,
        'repeated_crossfit_base_scores': repeated, 'meta_training_probability': meta.predict_proba(pp['xa'])[:, 1],
        'train_probability': meta.predict_proba(za)[:, 1],
        'evaluation_probability': meta.predict_proba(pp['xb'])[:, 1] if len(b) else np.empty(0),
        'mean_train_score': pa.mean(1), 'mean_evaluation_score': pb.mean(1),
        'ScoreType': 'StackLR_probability_uncalibrated'}


def predict(f, tables, indices):
    x = np.column_stack([parent.method.predict(base, tables, indices) for base in f['base']])
    z = (np.clip(x, f['low'], f['high'])-f['center'])/f['scale']
    return f['meta'].predict_proba(z)[:, 1]


def define():
    assert not OUT.exists()
    parent.frozen()
    assert not (parent.OUT/'test_started.json').exists()
    assert json.loads((parent.OUT/'training_verification.json').read_text(encoding='utf-8'))['passed']
    p = json.loads((parent.OUT/'protocol.json').read_text(encoding='utf-8'))
    protected = {**p['protected_hashes'], **p['source_hashes']}
    protected.update({str(path.relative_to(ROOT)): digest(path) for path in parent.OUT.rglob('*') if path.is_file()})
    OUT.mkdir(parents=True)
    for group in GROUPS:
        (OUT/group).mkdir(); (OUT/group/'models').mkdir()
    (OUT/'figures').mkdir()
    for name in ('sample_manifest.json', 'nested_folds.json', 'sampling_record.json'):
        (OUT/name).write_bytes((parent.OUT/name).read_bytes())
    sources = [ROOT/'lvi_prediction'/name for name in ('run_pure_habitat_random120_stacking_20261023.py',
        'verify_pure_habitat_random120_stacking_20261023.py', 'report_pure_habitat_random120_stacking_20261023.py')]
    protocol = {'date': '2026-10-23', 'authorization': 'Continue improving pure-habitat outer5>=.7;no additional patient draw.',
        'same_one_random120': True, 'sampling_seed': 2026102307, 'split_seed': 2026102308, 'new_random_draws': 0,
        'motivation': 'Parent128 nonlinear procedure outermean.5271;unchanged66 method onnewcohort.65394;all before anynewholdout prediction.',
        'primary': 'FixedCrossfitStack12 onRandom120;Original118 training-only control.',
        'members': '2norms x3regionalviews xPLS1/2,all C.03,ICC.75;all12 fixed,notrankedbytest/outer AUC.',
        'meta': 'Average3 repeatedinner heldout base scores perpatient;fit training-only P2.5/P97.5 clipping/meanSD;balancedLR C.03 fixed;no metaparameter search.',
        'outer': 'Same5 outerfolds. Eachouter training part generatesbaseOOFs usingitsown15innerfolds,learnsmeta onlyfromthatpart,refits12bases there,andpredictsoutervalidation.',
        'final': 'Same construction withinfull84;mean12 reported as secondary training-only diagnosis,notseparately tested.',
        'threshold': 'Youden from FULLSTACK outerOOF across84 trainingpatients,thenfreeze. Thisisnotin-samplemeta OOF performance;parentcomparators keep theirinnerOOF thresholds.',
        'cache': 'Parentverified base predictions onexact samecases/folds/configs;Original118 usesimmutableprevious66 checkpoints;Random120 usesparentReference66.',
        'parent_test_deferred': 'Parenttwo-final-model experiment remainsfrozen with0testcalls;scoreitsReference andImproved alongsideStack12 inone joint36 stage.',
        'test': 'Exactly3 frozen models onRandom12036;no Original118 newtest,noMean12 test,noalternativeweights.',
        'pure_habitat': True, 'Dstar': False, 'clinical': False, 'whole_predictors': False, 'K': 3,
        'prior_case_and_outer_exposure': True, 'exploratory': True, 'formal_use': False, 'automatic_adoption': False,
        'target': 'Random120 Stack12 outermean>=.70;reportallpreviousfailures,donotresample.',
        'source_hashes': {str(path.relative_to(ROOT)): digest(path) for path in sources}, 'protected_hashes': protected}
    save(OUT/'protocol.json', protocol)
    save(OUT/'definition_frozen.json', {'passed': True, 'new_test_calls': 0, 'protocol_sha256': digest(OUT/'protocol.json')})
    csv('fixed_base_members', members())
    save(OUT/'parent_test_deferral.json', {'parent_test_calls': 0, 'parent_models_frozen': True,
        'reason': 'Training-only fixedstack followup on same cohort before joint3model test;no newpatient draw.'})
    print('FixedStack12 defined;one sameRandom120;parent0test;three frozen models intendedforjointtest', flush=True)


def protect():
    p = json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    for field in ('protected_hashes', 'source_hashes'):
        for name, sha in p[field].items():
            assert digest(ROOT/name) == sha, name
    assert digest(OUT/'protocol.json') == json.loads((OUT/'definition_frozen.json').read_text(encoding='utf-8'))['protocol_sha256']
    assert not (parent.OUT/'test_started.json').exists()
    return len(p['protected_hashes'])


def train():
    protect(); assert not (OUT/'training_frozen.json').exists() and not (OUT/'test_started.json').exists()
    metrics, predictions, choices, weights, features = [], [], [], [], []
    for group in GROUPS:
        cohort, y, tr, te = parent.inputs(group)
        tables = joblib.load(parent.OUT/group/'training_features.joblib')
        outer = []
        for k, (a, b) in enumerate(parent.folds(group), 1):
            f = fit(group, 'outer%d' % k, tables, a, b, y); outer.append(f)
            for label, pa, pb in [('Stack12', f['train_probability'], f['evaluation_probability']),
                                  ('Mean12', f['mean_train_score'], f['mean_evaluation_score'])]:
                metrics.append({'Group': group, 'ModelKey': label, 'Fold': k,
                    'FitAUC': float(roc_auc_score(y[a], pa)), 'AUC': float(roc_auc_score(y[b], pb))})
                predictions.extend({'Group': group, 'ModelKey': label, 'Fold': k, 'AnalysisID': cohort.AnalysisID.iloc[i],
                    'Index': int(i), 'Outcome': int(y[i]), 'Probability': float(p)} for i, p in zip(b, pb))
            print('Stack12 '+group+' outer%d fit12bases+meta andvalidated' % k, flush=True)
        final = fit(group, 'final', tables, tr, [], y)
        frame = pd.DataFrame(predictions); pp = frame[(frame.Group == group) & (frame.ModelKey == 'Stack12')].set_index('Index').loc[tr]
        fp, tp, th = roc_curve(y[tr], pp.Probability.to_numpy(), drop_intermediate=False)
        final['threshold'] = float(th[np.argmax(tp-fp)])
        final['threshold_source'] = 'training_fullstack_outerOOF_Youden'
        mm = pd.DataFrame(metrics); mm = mm[(mm.Group == group) & (mm.ModelKey == 'Stack12')]
        choices.append({'Group': group, 'ModelKey': 'Stack12', 'TrainFitAUC': float(roc_auc_score(y[tr], final['train_probability'])),
            'OuterMeanAUC': float(mm.AUC.mean()), 'OuterSD': float(mm.AUC.std(ddof=1)),
            'PooledOOFAUC': float(roc_auc_score(pp.Outcome, pp.Probability)),
            'MetaInSampleOOFAssociationAUC': float(roc_auc_score(y[tr], final['meta_training_probability'])),
            'BaseCount': 12, 'MetaC': .03, 'Threshold': final['threshold'], 'TargetOuter07Met': bool(mm.AUC.mean() >= .7)})
        joblib.dump({'group': group, 'outer': outer, 'final': final}, OUT/group/'models/stack12.joblib', compress=3)
        for j, (c, base) in enumerate(zip(members(), final['base'])):
            weights.append({'Group': group, 'MemberID': c['CandidateKey'], 'normalization': c['normalization'],
                'view': c['view'], 'components': c['components'], 'MetaCoefficientAfterScoreScaling': float(final['meta'].coef_[0, j])})
            features.extend({'Group': group, 'MemberID': c['CandidateKey'], 'Feature': name, 'Modality': name.split('__')[0],
                'BaseCoefficientInGaussianSpace': float(beta)} for name, beta in zip(base['names'], base['coef_gaussian']))
    csv('training_choices', choices); csv('outer_metrics', metrics); csv('nested_predictions', predictions)
    csv('meta_weights', weights); csv('base_selected_features', features)
    files = [path for path in OUT.rglob('*') if path.is_file()]
    save(OUT/'training_frozen.json', {'passed': True, 'new_test_calls': 0, 'test_models': 3,
        'protocol_sha256': digest(OUT/'protocol.json'), 'files': {str(path.relative_to(OUT)): digest(path) for path in files}})
    print(pd.DataFrame(choices).to_string(index=False), flush=True)


def frozen():
    protect()
    f = json.loads((OUT/'training_frozen.json').read_text(encoding='utf-8'))
    assert f['passed'] and digest(OUT/'protocol.json') == f['protocol_sha256']
    for name, sha in f['files'].items():
        assert digest(OUT/name) == sha, name
    return f


def test():
    frozen(); assert not (OUT/'test_started.json').exists()
    assert json.loads((OUT/'training_verification.json').read_text(encoding='utf-8'))['passed']
    cohort, _, tr, te = parent.inputs('Random120'); tables = parent.data(te)
    joblib.dump(tables, OUT/'Random120/test_features.joblib', compress=3)
    stack = joblib.load(OUT/'Random120/models/stack12.joblib')['final']
    models = joblib.load(parent.OUT/'Random120/models/models.joblib')['final']
    save(OUT/'test_started.json', {'models': ['Stack12', 'Reference66', 'Nonlinear128'], 'test_model_calls': 3,
        'test_stage': 1, 'freeze_sha256': digest(OUT/'training_frozen.json')})
    scores = {'Stack12': predict(stack, tables, te), 'Reference66': parent.method.predict(models['Reference'], tables, te),
        'Nonlinear128': parent.method.predict(models['Improved'], tables, te)}
    fitted = {'Stack12': stack, 'Reference66': models['Reference'], 'Nonlinear128': models['Improved']}
    truth = cohort.Outcome.to_numpy(int)[te]
    from verify_ivim_df63_20261020 import draws
    samples = draws(truth, scores, 2026102397)
    metrics, predictions = [], []
    for name, p in scores.items():
        f = fitted[name]; lo, hi = np.quantile(samples[name], [.025, .975])
        tn, fp, fn, tp = confusion_matrix(truth, p >= f['threshold'], labels=[0, 1]).ravel()
        metrics.append({'Group': 'Random120', 'ModelKey': name, 'N': len(te), 'Positive': int(truth.sum()), 'Negative': int((1-truth).sum()),
            'AUC': float(roc_auc_score(truth, p)), 'AUC_CI95Low': float(lo), 'AUC_CI95High': float(hi),
            'Threshold': f['threshold'], 'Sensitivity': tp/(tp+fn), 'Specificity': tn/(tn+fp), 'Accuracy': (tp+tn)/len(te),
            'TP': int(tp), 'FN': int(fn), 'TN': int(tn), 'FP': int(fp), 'ScoreType': f['ScoreType']})
        predictions.extend({'Group': 'Random120', 'ModelKey': name, 'AnalysisID': cohort.AnalysisID.iloc[i], 'Index': int(i),
            'Outcome': int(t), 'Probability': float(s), 'Prediction': int(s >= f['threshold']), 'ScoreType': f['ScoreType']} for i, t, s in zip(te, truth, p))
    csv('test_metrics', metrics); csv('test_predictions', predictions)
    paired = []
    for name in ('Reference66', 'Nonlinear128'):
        lo, hi = np.quantile(samples['Stack12']-samples[name], [.025, .975])
        paired.append({'Comparison': 'Stack12_minus_'+name, 'Difference': float(roc_auc_score(truth, scores['Stack12'])-roc_auc_score(truth, scores[name])),
            'CI95Low': float(lo), 'CI95High': float(hi), 'BootstrapN': 5000})
    csv('paired_test_differences', paired)
    save(OUT/'test_complete.json', {'passed': True, 'test_model_calls': 3, 'test_stages': 1,
        'parent_standalone_test_calls': 0, 'Original118_new_test_calls': 0, 'post_test_fits': 0,
        'post_test_threshold_changes': 0, 'metrics': metrics})
    print(pd.DataFrame(metrics).to_string(index=False), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=('define', 'train', 'test'))
    with threadpool_limits(1):
        globals()[p.parse_args().action]()
