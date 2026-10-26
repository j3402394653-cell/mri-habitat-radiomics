# -*- coding: utf-8 -*-
"""Multimodal five-family nested IVIM K sweep with a single final test stage."""
import os
os.environ.update({'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import time
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sklearn.metrics import roc_auc_score
import filtered_ordered_fixed118_study_20261023 as study
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from scipy.special import expit
import run_k3_stability_improvement90_20261013 as stable
from run_regional_summary77_two_tests_20261014 import bootstrap

OUT, ROOT = study.OUT, study.ROOT
save, digest, csv = study.save, study.digest, study.csv
TABLES, PROFILES = None, None


def extract(stage, workers):
    study.protect()
    if stage == 'test':
        assert json.loads((OUT / 'training_verification.json').read_text(encoding='utf-8'))['passed']
    indices = study.previous.union_indices('train' if stage == 'train' else 'sample')
    study.extract_physical(indices)
    save(OUT / ('extraction_' + stage + '_complete.json'), {'passed': True, 'N': len(indices),
        'hashes': {str(int(i)): digest(study.patient_path(i)) for i in indices}, 'outcome_used': False})


def initialize():
    global TABLES
    threadpool_limits(1)
    TABLES = joblib.load(OUT / 'training_features.joblib')


def rows(d, indices):
    return stable.old.common.rows(d, np.asarray(indices, int))


def transform(x, xb):
    median = np.nanmedian(np.where(np.isfinite(x), x, np.nan), axis=0)
    median = np.where(np.isfinite(median), median, 0.)
    a = np.where(np.isfinite(x), x, median)
    low, high = np.quantile(a, [.025, .975], axis=0)
    a = np.clip(a, low, high)
    center = np.median(a, axis=0)
    scale = np.subtract(*np.quantile(a, [.75, .25], axis=0)) / 1.348979500392164
    scale = np.where(scale > 1e-12, scale, a.std(0)); scale = np.where(scale > 1e-12, scale, 1.)
    return {'median': median, 'low': low, 'high': high, 'center': center, 'scale': scale,
            'xa': (a - center) / scale,
            'xb': (np.clip(np.where(np.isfinite(xb), xb, median), low, high) - center) / scale}


def clinical_prepare(data, a, b, scope):
    d = data['Clinical']
    names = [m['name'] for m in d['metadata']]
    cols = list(range(11)) if scope == 'Extended11' else [names.index(n) for n in study.BASIC] if scope == 'Basic5' else []
    return {**transform(d['base'][rows(d, a)][:, cols], d['base'][rows(d, b)][:, cols]),
            'indices': cols, 'names': [names[j] for j in cols]}


def ranking(d, a, y):
    _, ranks, audit = stable.ranking(d, a, y)
    eligible = [j for j,r in enumerate(audit) if r['CompleteFraction'] >= .9 and r['ICC'] is not None and r['ICC'] >= .75 and np.ptp(ranks[:,j]) > 0]
    for j,r in enumerate(audit):
        r['Eligible'] = j in eligible
    ordered = {'Stability': sorted(eligible, key=lambda j:(-audit[j]['ICC'],d['metadata'][j]['name'])),
               'Association': sorted(eligible, key=lambda j:(-audit[j]['AssociationScore'],-audit[j]['ICC'],d['metadata'][j]['name']))}
    return ordered,ranks,audit


def prepare(data, a, b, y, c, full=None, cp=None):
    if c['image_key'] is not None:
        image = data[c['image_key']]
        p = stable.prefix(full, c['top']) if full is not None else stable.prepare(image, a, b, y, c, ranking(image, a, y))
        p = {**p, 'image_indices': list(p['indices'])}
    else:
        p = {'indices': [], 'names': [], 'audit': [], 'image_indices': [],
             **{s: np.empty(0) for s in ('median', 'low', 'high', 'center', 'scale')},
             'xa': np.empty((len(a), 0)), 'xb': np.empty((len(b), 0))}
    if c['clinical']:
        cp = clinical_prepare(data, a, b, c['clinical_scope']) if cp is None else cp
        offset = len(data[c['image_key']]['metadata']) if c['image_key'] is not None else 0
        names = cp['names']
        p = {**p, 'indices': p['indices'] + list(range(offset, offset + len(names))),
             'names': p['names'] + names,
             **{s: np.r_[p[s], cp[s]] for s in ('median', 'low', 'high', 'center', 'scale')},
             'xa': np.column_stack([p['xa'], cp['xa']]), 'xb': np.column_stack([p['xb'], cp['xb']])}
        p['clinical_indices'] = cp['indices']
    p.setdefault('clinical_indices', [])
    return p


def classifier_score(model, x, c):
    if not len(x):
        return np.empty(0)
    return expit(model.decision_function(x)) if c['method'] == 'RBF' else model.predict_proba(x)[:, 1]


def fitted(p, y, c, a):
    model = None
    if p['indices']:
        model = SVC(C=c['C'], kernel='rbf', gamma=c['gamma'], class_weight='balanced', probability=False, random_state=64212) if c['method'] == 'RBF' else LogisticRegression(C=c['C'], penalty='l2', solver='liblinear', class_weight='balanced', max_iter=3000, random_state=64212)
        model.fit(p['xa'], y)
    pa = classifier_score(model, p['xa'], c) if model is not None else np.full(len(y), y.mean())
    pb = classifier_score(model, p['xb'], c) if model is not None else np.full(len(p['xb']), y.mean())
    return {**{s:p[s] for s in ('indices','names','audit','median','low','high','center','scale','image_indices','clinical_indices')},
            'config': c, 'model': model, 'prevalence':float(y.mean()), 'train_probability':pa,
            'evaluation_probability':pb, 'available':bool(model is not None and np.ptp(pa)>1e-12)}


def fit(data, a, b, y, c):
    return fitted(prepare(data, a, b, y, c), y[a], c, a)


def raw(f, data, indices):
    c = f['config']
    parts = []
    if c['image_key'] is not None:
        d = data[c['image_key']]
        parts.append(d['base'][rows(d, indices)][:, f['image_indices']])
    if c['clinical']:
        d = data['Clinical']; parts.append(d['base'][rows(d, indices)][:, f['clinical_indices']])
    return np.column_stack(parts) if parts else np.empty((len(indices), 0))


def transformed(f, data, indices):
    x = raw(f, data, indices)
    return (np.clip(np.where(np.isfinite(x), x, f['median']), f['low'], f['high']) - f['center']) / f['scale']


def predict(f, data, indices):
    x = transformed(f, data, indices)
    return classifier_score(f['model'], x, f['config']) if f['model'] is not None else np.full(len(indices), f['prevalence'])


def choose(grid, family, k=None):
    candidates = [r for r in grid if r['family'] == family and (k is None or r['K'] == k)]
    return max(candidates, key=lambda r: (bool(r['AvailableAllFolds']), r['Objective'],
        -r['K'], -r['MeanFeatureCount'], -r['C'], -r['candidate_id']))


def phase_path(group, phase):
    return OUT / 'phases' / (group + '_' + phase + '.joblib')


def tune_phase(group, phase):
    dest = phase_path(group, phase)
    if dest.exists():
        return group, phase, 'checkpoint'
    _, y, _, te = study.inputs(group)
    ff = study.folds(group, 'final') if phase == 'final' else study.folds(group, 'inner', int(phase[-1]))
    opts = study.options()
    cv, audit = {c['candidate_id']: [] for c in opts}, []
    start = time.time()
    for fold, (a, b) in enumerate(ff, 1):
        assert not (set(a) | set(b)) & set(te) and (y[te] == -1).all()
        cp = {scope: clinical_prepare(TABLES, a, b, scope) for scope in ('None','Basic5','Extended11')}
        full = {}
        for key, d in TABLES.items():
            if key == 'Clinical':
                continue
            rank = ranking(d, a, y)
            for ranking_name in ('Association', 'Stability'):
                c = {'image_key': key, 'ranking': ranking_name, 'top': 0}
                full[(key, ranking_name)] = stable.prepare(d, a, b, y, c, rank)
        prepared, estimates = {}, {}
        for c in opts:
            identity = (c['image_key'], c['ranking'], c['top'], c['clinical_scope'])
            if identity not in prepared:
                prepared[identity] = prepare(TABLES, a, b, y, c,
                    full=full.get((c['image_key'], c['ranking'])), cp=cp[c['clinical_scope']])
            p = prepared[identity]
            estimator_key = (c['image_key'], tuple(p['indices']), c['clinical_scope'], c['method'], c['C'], c['gamma'])
            if estimator_key not in estimates:
                estimates[estimator_key] = fitted(p, y[a], c, a)
            f = estimates[estimator_key]
            cid = c['candidate_id']
            audit.append({'InnerFold': fold, 'candidate_id': cid,
                'ValidationAUC': float(roc_auc_score(y[b], f['evaluation_probability'])),
                'FitAUC': float(roc_auc_score(y[a], f['train_probability'])),
                'FeatureCount': len(f['indices']), 'ImageFeatureCount': len(f['image_indices']),
                'Available': bool(f['available'])})
            cv[cid].append({'train': a, 'validation': b, 'train_probability': f['train_probability'],
                            'validation_probability': f['evaluation_probability']})
        print(group + ' ' + phase + ' inner' + str(fold) + '/5 complete', flush=True)
    frame = pd.DataFrame(audit).set_index(['candidate_id', 'InnerFold'])
    grid = []
    for c in opts:
        rr = frame.loc[c['candidate_id']]
        mean, sd, train = rr.ValidationAUC.mean(), rr.ValidationAUC.std(ddof=1), rr.FitAUC.mean()
        available = bool(rr.Available.all())
        grid.append({**c, 'MeanValidationAUC': float(mean), 'SD': float(sd), 'MeanFitAUC': float(train),
            'MeanFeatureCount': float(rr.FeatureCount.mean()), 'AvailableAllFolds': available,
            'Objective': float(mean - .10 * sd - .30 * max(train - mean - .10, 0)) if available else -1.})
    result = {'group': group, 'phase': phase, 'grid': grid, 'audit': audit, 'cv': cv,
              'elapsed_seconds': time.time() - start}
    temp = dest.with_suffix('.pending'); joblib.dump(result, temp, compress=3); temp.replace(dest)
    return group, phase, '%.1fs' % result['elapsed_seconds']


def selections(grid):
    selected = {f: study.config(choose(grid, f)) for f in study.FAMILIES}
    selected.update({f + '__K' + str(k): study.config(choose(grid, f, k))
                     for f in ('Habitat', 'HabitatClinical') for k in study.feature.KS})
    return selected


def assemble(group, data):
    cohort, y, tr, te = study.inputs(group)
    outer, metrics, predictions = [], [], []
    for fold, (a, b) in enumerate(study.folds(group, 'outer'), 1):
        phase = joblib.load(phase_path(group, 'outer' + str(fold)))
        cache, models = {}, {}
        for key, c in selections(phase['grid']).items():
            cid = c['candidate_id']
            if cid not in cache:
                cache[cid] = fit(data, a, b, y, c)
            f = cache[cid]; models[key] = f
            metrics.append({'Group': group, 'ModelKey': key, 'Fold': fold, **c,
                            'AUC': float(roc_auc_score(y[b], f['evaluation_probability'])),
                            'ImageFeatureCount': len(f['image_indices'])})
            predictions.extend({'Group': group, 'ModelKey': key, 'Fold': fold, 'AnalysisID': cohort.AnalysisID.iloc[i],
                'Outcome': int(y[i]), 'Probability': float(p)} for i, p in zip(b, f['evaluation_probability']))
        outer.append({'train': a, 'validation': b, 'models': models})
    phase = joblib.load(phase_path(group, 'final'))
    final, choices, cache, feature_rows = {}, [], {}, []
    nested, mm = pd.DataFrame(predictions), pd.DataFrame(metrics)
    for key, c in selections(phase['grid']).items():
        cid = c['candidate_id']
        if cid not in cache:
            f = fit(data, tr, [], y, c)
            oof = np.full(138, np.nan)
            for row in phase['cv'][cid]:
                oof[row['validation']] = row['validation_probability']
            assert np.isfinite(oof[tr]).all() and np.isnan(oof[te]).all()
            f['threshold'] = stable.old.common.threshold(y[tr], oof[tr]); f['training_oof'] = oof
            cache[cid] = f
        f = cache[cid]; final[key] = f
        vv = mm[mm.ModelKey == key].AUC
        pp = nested[nested.ModelKey == key]
        chosen = choose(phase['grid'], c['family'], c['K'] if '__K' in key else None)
        choices.append({'Group': group, 'ModelKey': key, **c,
            'TrainFitAUC': float(roc_auc_score(y[tr], f['train_probability'])),
            'OuterMeanAUC': float(vv.mean()), 'OuterSD': float(vv.std(ddof=1)),
            'PooledOOFAUC': float(roc_auc_score(pp.Outcome, pp.Probability)),
            'FinalInnerAUC': chosen['MeanValidationAUC'], 'SelectionObjective': chosen['Objective'],
            'FeatureCount': len(f['names']), 'ImageFeatureCount': len(f['image_indices']),
            'ClinicalFeatureCount': len(f['clinical_indices']),
            'Threshold': f['threshold'], 'AvailableFinal': bool(f['available'])})
        if f['model'] is not None:
            coefficients = f['model'].coef_[0] if c['method'] == 'Ridge' else np.full(len(f['names']), np.nan)
            feature_rows.extend({'Group': group, 'ModelKey': key, 'Feature': name, 'Coefficient': float(beta), 'WeightScope': 'linear_logit' if c['method'] == 'Ridge' else 'nonlinear_no_single_coefficient',
                'Block': data[c['image_key']]['metadata'][f['image_indices'][j]]['modality'] if j < len(f['image_indices']) else 'Clinical'}
                for j, (name, beta) in enumerate(zip(f['names'], coefficients)))
    result = {'group': group, 'train': tr, 'outer': outer, 'final': final,
              'choices': choices, 'outer_metrics': metrics, 'nested_predictions': predictions,
              'selected_features': feature_rows}
    joblib.dump(result, OUT / 'models' / (group + '.joblib'), compress=3)
    return result


def train(workers):
    study.protect()
    assert not (OUT / 'test_started.json').exists()
    assert json.loads((OUT / 'extraction_train_complete.json').read_text(encoding='utf-8'))['passed']
    if (OUT / 'training_frozen.json').exists():
        frozen(); return
    if not (OUT / 'training_features.joblib').exists():
        joblib.dump(study.data(study.previous.union_indices('train')), OUT / 'training_features.joblib', compress=3)
    tasks = [(group, phase) for group in study.GROUPS for phase in ['outer' + str(k) for k in range(1, 6)] + ['final']]
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize) as pool:
        futures = [pool.submit(tune_phase, group, phase) for group, phase in tasks]
        for future in as_completed(futures):
            print('Phase complete ' + ' '.join(future.result()), flush=True)
    initialize()
    results = [assemble(group, TABLES) for group in study.GROUPS]
    for name in ('choices', 'outer_metrics', 'nested_predictions', 'selected_features'):
        csv('training_choices' if name == 'choices' else name, [r for result in results for r in result[name]])
    grid = [{ 'Group': group, 'Phase': phase, **row} for group, phase in tasks
            for row in joblib.load(phase_path(group, phase))['grid']]
    csv('inner_candidate_scores', grid)
    files = list((OUT / 'models').glob('*.joblib')) + list((OUT / 'phases').glob('*.joblib'))
    files += [OUT / 'training_features.joblib'] + list(OUT.glob('*.csv'))
    save(OUT / 'training_frozen.json', {'passed': True, 'final_models_for_test': 5,
        'fixed_K_diagnostics_for_test': 0, 'protocol_sha256': digest(OUT / 'protocol.json'),
        'files': {str(p.relative_to(OUT)): digest(p) for p in files},
        'patient_feature_hashes': json.loads((OUT / 'extraction_train_complete.json').read_text(encoding='utf-8'))['hashes'],
        'test_scored': False})
    print('Training frozen:5 main models;fixedK diagnostics not to be tested', flush=True)


def frozen():
    study.protect()
    f = json.loads((OUT / 'training_frozen.json').read_text(encoding='utf-8'))
    assert f['passed'] and digest(OUT / 'protocol.json') == f['protocol_sha256']
    for name, sha in f['files'].items():
        assert digest(OUT / name) == sha, name
    for index, sha in f['patient_feature_hashes'].items():
        assert digest(study.patient_path(int(index))) == sha, index
    return f


def test():
    frozen()
    assert not (OUT / 'test_started.json').exists(), 'Do not repeat holdout prediction'
    assert json.loads((OUT / 'training_verification.json').read_text(encoding='utf-8'))['passed']
    assert json.loads((OUT / 'extraction_test_complete.json').read_text(encoding='utf-8'))['passed']
    data = study.data(study.previous.union_indices('sample'))
    joblib.dump(data, OUT / 'all118_features.joblib', compress=3)
    save(OUT / 'test_started.json', {'scoring_stage': 1, 'fixed_main_models': 5,
                                   'freeze_sha256': digest(OUT / 'training_frozen.json')})
    predictions, metrics, paired = [], [], []
    calls = 0
    for offset, group in enumerate(study.GROUPS):
        cohort, _, tr, te = study.inputs(group)
        truth = cohort.Outcome.to_numpy(int)[te]
        final = joblib.load(OUT / 'models' / (group + '.joblib'))['final']
        scores = {family: predict(final[family], data, te) for family in study.FAMILIES}
        calls += len(scores)
        draws = bootstrap(truth, scores, 2026102151 + offset)
        for key, p in scores.items():
            threshold = final[key]['threshold']
            low, high = np.quantile(draws[key], [.025, .975])
            metrics.append({'Group': group, 'ModelKey': key, 'N': len(te), 'Positive': int(truth.sum()),
                'Negative': int((truth == 0).sum()), **stable.old.common.old.metrics(truth, p, threshold),
                'AUC_CI95Low': float(low), 'AUC_CI95High': float(high)})
            predictions.extend({'Group': group, 'ModelKey': key, 'AnalysisID': cohort.AnalysisID.iloc[i],
                'Outcome': int(y), 'Probability': float(pp), 'Threshold': threshold, 'ScoreType': 'sigmoid_SVM_margin' if final[key]['config']['method'] == 'RBF' else 'LR_probability'}
                for i, y, pp in zip(te, truth, p))
        for first, second in (('Habitat', 'Whole'), ('HabitatClinical', 'Clinical'),
                              ('WholeClinical', 'Clinical'), ('HabitatClinical', 'WholeClinical')):
            low, high = np.quantile(draws[first] - draws[second], [.025, .975])
            paired.append({'Group': group, 'First': first, 'Second': second,
                'DeltaAUC': float(roc_auc_score(truth, scores[first]) - roc_auc_score(truth, scores[second])),
                'CI95Low': float(low), 'CI95High': float(high)})
        print(group + ' own36 single frozen5-model scoring complete', flush=True)
    csv('test_predictions', predictions); csv('test_metrics', metrics); csv('paired_family_AUC_differences', paired)
    frozen()
    save(OUT / 'test_complete.json', {'passed': True, 'model_predict_calls': calls, 'scoring_stages': 1,
        'fixedK_alternative_test_calls': 0, 'prediction_rows': len(predictions),
        'model_refits_after_test_start': 0, 'threshold_changes': 0})


def main(phase, workers):
    import verify_filtered_ordered_fixed118_20261023 as verify
    if phase == 'smoke':
        print(json.dumps(verify.smoke()), flush=True)
    elif phase == 'define':
        study.define(); extract('train', workers); result = verify.smoke(); save(OUT / 'feature_smoke_verification.json', result)
    elif phase in ('extract-train', 'extract-test'):
        extract(phase.split('-')[1], workers)
    elif phase == 'train':
        train(workers)
    elif phase == 'verify-training':
        verify.training()
    elif phase == 'test':
        test()
    elif phase == 'verify-test':
        verify.inference()
    else:
        import report_filtered_ordered_fixed118_20261023 as report
        report.main()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('smoke', 'define', 'extract-train', 'extract-test', 'train',
        'verify-training', 'test', 'verify-test', 'report'), required=True)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    with threadpool_limits(1):
        main(args.phase, args.workers)
