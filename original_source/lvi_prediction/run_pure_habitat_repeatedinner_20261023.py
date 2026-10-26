# -*- coding: utf-8 -*-
"""Pure habitat, repeated inner5x3, outcome-blind QC and latent joint representations."""
import os
os.environ.update({'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
import argparse
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
import run_pure_habitat_ensemble_fixed118_20261023 as previous
import pure_habitat_latent_features_20261023 as feature

ROOT, save, digest = previous.ROOT, previous.save, previous.digest
OUT = ROOT/'lvi_prediction/results/lvi_pure_habitat_repeatedinner_20261023'
PHASES = tuple('outer%d' % k for k in range(1, 6))+('final',)


def csv(name, rows):
    pd.DataFrame(rows).to_csv(OUT/(name+'.csv'), index=False, encoding='utf-8-sig')


def read(name):
    return pd.read_csv(OUT/(name+'.csv'), float_precision='round_trip', keep_default_na=False, na_values=[''])


def options():
    result = []
    for norm in ('PercentileZ', 'RobustZ'):
        for view in feature.VIEWS:
            for method, components, C in [('Ridge', 0, .003), ('Ridge', 0, .03), ('Ridge', 0, .3)]+[
                (method, n, C) for method, numbers in [('PCA', (2, 5)), ('PLS', (1, 2))] for n in numbers for C in (.03, .3)]:
                result.append({'candidate_id': len(result), 'normalization': norm, 'view': view,
                    'image_key': norm+'__'+view, 'method': method, 'components': components, 'C': C})
    assert len(result) == 66
    return result


def config(r):
    return {k: r[k] for k in options()[0]}


def data():
    ll = previous.local_data('train')
    return feature.build({key.split('__')[0]: d for key, d in ll.items() if key.endswith('__OrderedCompact')})


def define():
    assert not (OUT/'protocol.json').exists() and not (OUT/'definition_frozen.json').exists()
    assert not (OUT/'training_features.joblib').exists() and not (OUT/'training_frozen.json').exists()
    assert not list((OUT/'phases').glob('*.joblib')) and not list((OUT/'models').glob('*.joblib'))
    previous.frozen()
    assert json.loads((previous.OUT/'delivery_final_verification.json').read_text(encoding='utf-8'))['passed']
    parent = json.loads((previous.OUT/'protocol.json').read_text(encoding='utf-8'))
    protected = {**parent['protected_hashes'], **parent['source_hashes']}
    protected.update({str(p.relative_to(ROOT)): digest(p) for p in previous.OUT.rglob('*') if p.is_file()})
    OUT.mkdir(parents=True, exist_ok=True)
    for folder in ('Fixed118', 'phases', 'models', 'figures'):
        (OUT/folder).mkdir(exist_ok=True)
    for name in ('Train82_cases.csv', 'Test36_cases.csv', 'Sample118_cases.csv', 'nested5_folds.json'):
        (OUT/'Fixed118'/name).write_bytes((previous.OUT/'Fixed118'/name).read_bytes())
    (OUT/'sample_manifest.json').write_bytes((previous.OUT/'sample_manifest.json').read_bytes())
    cohort, y, tr, te = previous.common.inputs('Fixed118')
    nested = {}
    for phase in PHASES:
        old = previous.common.phase_folds(phase)
        allowed = tr if phase == 'final' else previous.local.study.folds('Fixed118', 'outer')[int(phase[-1])-1][0]
        base_seed = 62212 if phase == 'final' else 62200+int(phase[-1])
        repetitions = [old]
        for repeat in (1, 2):
            splitter = StratifiedKFold(5, shuffle=True, random_state=base_seed+repeat*10000)
            repetitions.append([(allowed[a], allowed[b]) for a, b in splitter.split(np.zeros(len(allowed)), y[allowed])])
        nested[phase] = [{'Repeat': repeat+1, 'Fold': fold+1, 'train': a.tolist(), 'validation': b.tolist()}
                         for repeat, ff in enumerate(repetitions) for fold, (a, b) in enumerate(ff)]
        assert all(not (set(r['train']) | set(r['validation'])) & set(te) for r in nested[phase])
    save(OUT/'repeated_inner_folds.json', nested)
    audit = feature.previous_selection_audit(joblib.load(previous.OUT/'models/pure_habitat_ensemble.joblib'))
    csv('previous_feature_overlap_audit', audit)
    sources = [ROOT/'lvi_prediction'/name for name in ('pure_habitat_latent_features_20261023.py',
        'run_pure_habitat_repeatedinner_20261023.py', 'verify_pure_habitat_repeatedinner_20261023.py',
        'report_pure_habitat_repeatedinner_20261023.py')]
    method = {'date': '2026-10-23', 'authorization': 'Improve pure-habitat outer5 performance;retain last .7018 test model.',
        'cohort': 'Original118/exact82train36test;original5outer folds unchanged.',
        'scope': 'Training-only development. No new36 test predictions;prior .7018 cannot be assigned to this new model.',
        'pure_habitat': True, 'Dstar': False, 'clinical': False, 'whole_predictors': False, 'K': 3,
        'partition': 'Original outcome-blind patient-localK3,T2/ADC/IVIM independent native spaces;unchanged input6D/f dimensions.',
        'normalization': 'Original two patient-level habitat caches PercentileZ/RobustZ;not common-center construction.',
        'views': {'OrderedCompact': 321, 'MeanContrast': '321:regionalmean,(H3-H1),(H2-(H1+H3)/2);existing pairwise contacts/distances kept',
                  'ContrastOnly': '220:same contrasts andpairwise relations,without regionalmean'},
        'QC': 'Training-only complete>=.9,absolute perturbationICC>=.75,nonconstant;absSpearman<.95 dedup in descendingICC/name order.',
        'screening': 'No outcome-based univariate feature ranking or top5/15 truncation;use all eligible decorrelated regional variables.',
        'feature_preprocessing': 'Current training-fold median fill;QuantileTransformer normal(n_quantiles=ntrain,no subsampling),clip[-2.5,2.5]. Not an MRI intensity-standardization method.',
        'models': 'BalancedL2LR C.003/.03/.3;PCA2/5+balancedLR C.03/.3;PLS1/2+balancedLR C.03/.3;latent scores scaled bytraining mean/SD.',
        'candidate_count': 66, 'repeated_folds_sha256': digest(OUT/'repeated_inner_folds.json'),
        'preflight': 'LR/PCA/PLS formula andempty-batch checks on training-only inputs;no AUC used for selecting definitions.',
        'innerCV': 'Five folds repeated3 times;repeat1 exact previousinner5;repeat2/3 use fixedseed+10000/+20000. Never select an advantageous outer split.',
        'selection_objective': 'mean15foldAUC-.10SD-.30max(meanFit-meanValidation-.10,0);availableallfolds;ties fewer effective dimensions,raw featurecount,C,candidateID.',
        'primary': 'Adaptive repeatedinner recipe;outer5 estimates entire66-candidate procedure.',
        'diagnostics': 'Ridge/PCA/PLS restrictions andSingleInner(repeat1only) assessed on outer5 as secondary training-only comparisons;do not promote highestouter row.',
        'threshold': 'Full82 selectedconfig:average3 repeat OOFscores percase,thenYouden;freeze forpossible future evaluation.',
        'previous_feature_overlap': {'medianJaccard': float(np.median([r['Jaccard'] for r in audit])),
            'pairs': len(audit), 'shared_features': sum(r['SharedFeatures'] for r in audit),
            'sign_flips': sum(r['SharedCoefficientSignFlips'] for r in audit), 'interpretation': 'Diagnostic only;correlated features can substitute,not proof of causation.'},
        'prior_test_exposure': True, 'formal_use': False, 'automatic_adoption': False, 'excluded_experiments_used': False,
        'source_hashes': {str(p.relative_to(ROOT)): digest(p) for p in sources}, 'protected_hashes': protected,
        'references': ['https://scikit-learn.org/1.3/modules/generated/sklearn.model_selection.RepeatedStratifiedKFold.html',
            'https://scikit-learn.org/1.3/modules/generated/sklearn.preprocessing.QuantileTransformer.html',
            'https://scikit-learn.org/1.3/modules/generated/sklearn.cross_decomposition.PLSRegression.html']}
    save(OUT/'protocol.json', method)
    save(OUT/'definition_frozen.json', {'passed': True, 'protocol_sha256': digest(OUT/'protocol.json'),
        'candidate_search_started': False, 'new_test_calls': 0})
    csv('candidate_definitions', options())
    print('Frozen66-candidate pure-habitat protocol:originalouter5,inner5x3;training-only', flush=True)


def protect():
    p = json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    for field in ('protected_hashes', 'source_hashes'):
        for name, sha in p[field].items():
            assert digest(ROOT/name) == sha, name
    assert digest(OUT/'protocol.json') == json.loads((OUT/'definition_frozen.json').read_text(encoding='utf-8'))['protocol_sha256']
    assert digest(OUT/'repeated_inner_folds.json') == p['repeated_folds_sha256']
    assert not (OUT/'test_started.json').exists()
    return len(p['protected_hashes'])


def folds(phase):
    return [(r['Repeat'], r['Fold'], np.asarray(r['train'], int), np.asarray(r['validation'], int))
            for r in json.loads((OUT/'repeated_inner_folds.json').read_text(encoding='utf-8'))[phase]]


def rows(d, indices):
    lookup = {int(i): j for j, i in enumerate(d['indices'])}
    return [lookup[int(i)] for i in indices]


def prepare(d, a, b):
    rr = rows(d, a)
    measured = np.stack([d[kind][rr] for kind in feature.KINDS], axis=1)
    complete = np.isfinite(measured).all(1).mean(0)
    raw = d['base'][rr]
    median = np.nanmedian(np.where(np.isfinite(raw), raw, np.nan), axis=0)
    median = np.where(np.isfinite(median), median, 0.)
    filled = np.where(np.isfinite(raw), raw, median)
    icc = np.full(raw.shape[1], np.nan)
    for j in np.flatnonzero((complete >= .9) & (filled.std(0) > 1e-12)):
        valid = np.isfinite(measured[:, :, j]).all(1)
        icc[j] = previous.local.stable.old.common.old.engine.icc_absolute(measured[valid, :, j])
    eligible = np.flatnonzero((complete >= .9) & np.isfinite(icc) & (icc >= .75) & (filled.std(0) > 1e-12))
    ordered = sorted(eligible, key=lambda j: (-icc[j], d['metadata'][j]['name']))
    selected = []
    if len(eligible):
        ranks = rankdata(filled[:, eligible], axis=0)
        corr = np.atleast_2d(np.corrcoef(ranks, rowvar=False))
        lookup = {int(j): n for n, j in enumerate(eligible)}
        for j in ordered:
            if all(abs(corr[lookup[j], lookup[k]]) < .95 for k in selected):
                selected.append(j)
    xa = filled[:, selected]
    xb = d['base'][rows(d, b)][:, selected]
    med = median[selected]
    xb = np.where(np.isfinite(xb), xb, med)
    qt = None
    if selected:
        qt = QuantileTransformer(n_quantiles=len(a), output_distribution='normal', subsample=len(a), random_state=64212).fit(xa)
        xa = np.clip(qt.transform(xa), -2.5, 2.5)
        xb = np.clip(qt.transform(xb), -2.5, 2.5) if len(b) else np.empty((0, len(selected)))
    audit = [{'Feature': m['name'], 'CompleteFraction': float(complete[j]),
        'ICC': float(icc[j]) if np.isfinite(icc[j]) else None, 'Eligible': bool(j in eligible), 'Selected': bool(j in selected)} for j, m in enumerate(d['metadata'])]
    return {'indices': selected, 'names': [d['metadata'][j]['name'] for j in selected], 'median': med,
        'quantile': qt, 'audit': audit, 'xa': xa, 'xb': xb}


def reduction(p, target, c):
    xa, xb = p['xa'], p['xb']
    reducer = None
    if c['method'] == 'Ridge':
        return {'reducer': None, 'za': xa, 'zb': xb, 'latent_center': None, 'latent_scale': None,
            'projection': None, 'input_center': None, 'effective': xa.shape[1]}
    n = min(c['components'], xa.shape[1], len(xa)-1)
    reducer = PCA(n_components=n, svd_solver='full').fit(xa) if c['method'] == 'PCA' else PLSRegression(n_components=n, scale=False, max_iter=1000).fit(xa, target)
    za = reducer.transform(xa)
    zb = reducer.transform(xb) if len(xb) else np.empty((0, n))
    mu, sd = za.mean(0), za.std(0)
    sd = np.where(sd > 1e-12, sd, 1.)
    projection = reducer.components_.T if c['method'] == 'PCA' else reducer.x_rotations_/reducer._x_std[:, None]
    center = reducer.mean_ if c['method'] == 'PCA' else reducer._x_mean
    return {'reducer': reducer, 'za': (za-mu)/sd, 'zb': (zb-mu)/sd,
        'latent_center': mu, 'latent_scale': sd, 'projection': projection, 'input_center': center, 'effective': n}


def fitted(p, target, c, latent=None):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        if not p['indices']:
            return {**{k: p[k] for k in ('indices', 'names', 'median', 'quantile', 'audit')}, 'config': c,
                'classifier': None, 'available': False, 'effective': 0, 'prevalence': float(target.mean()),
                'train_probability': np.full(len(target), target.mean()), 'evaluation_probability': np.full(len(p['xb']), target.mean())}
        latent = reduction(p, target, c) if latent is None else latent
        lr = LogisticRegression(C=c['C'], penalty='l2', solver='liblinear', class_weight='balanced', max_iter=3000, random_state=64212).fit(latent['za'], target)
    pa = lr.predict_proba(latent['za'])[:, 1]
    pb = lr.predict_proba(latent['zb'])[:, 1] if len(p['xb']) else np.empty(0)
    if c['method'] == 'Ridge':
        beta, intercept = lr.coef_[0].copy(), float(lr.intercept_[0])
    else:
        weight = lr.coef_[0]/latent['latent_scale']
        beta = latent['projection']@weight
        intercept = float(lr.intercept_[0]-latent['input_center']@beta-latent['latent_center']@weight)
    np.testing.assert_allclose(pa, expit(p['xa']@beta+intercept), atol=1e-10, rtol=0)
    available = bool(np.ptp(pa) > 1e-12 and not any(issubclass(w.category, ConvergenceWarning) for w in caught))
    return {**{k: p[k] for k in ('indices', 'names', 'median', 'quantile', 'audit')}, 'config': c,
        **{k: latent[k] for k in ('reducer', 'latent_center', 'latent_scale', 'effective')},
        'classifier': lr, 'coef_gaussian': beta, 'intercept_gaussian': intercept, 'available': available,
        'prevalence': float(target.mean()), 'train_probability': pa, 'evaluation_probability': pb}


def fit(tables, a, b, y, c):
    return fitted(prepare(tables[c['image_key']], a, b), y[a], c)


def transformed(f, tables, indices):
    d = tables[f['config']['image_key']]
    x = d['base'][rows(d, indices)][:, f['indices']]
    x = np.where(np.isfinite(x), x, f['median'])
    return np.clip(f['quantile'].transform(x), -2.5, 2.5) if f['indices'] and len(indices) else np.empty((len(indices), len(f['indices'])))


def predict(f, tables, indices):
    if f['classifier'] is None:
        return np.full(len(indices), f['prevalence'])
    x = transformed(f, tables, indices)
    if f['reducer'] is not None:
        x = (f['reducer'].transform(x)-f['latent_center'])/f['latent_scale']
    return f['classifier'].predict_proba(x)[:, 1]


def grid_scores(audit):
    frame = pd.DataFrame(audit)
    result = []
    for c in options():
        r = frame[frame.candidate_id == c['candidate_id']]
        mu, sd, fm = r.ValidationAUC.mean(), r.ValidationAUC.std(ddof=1), r.FitAUC.mean()
        available = bool(r.Available.all())
        result.append({**c, 'MeanValidationAUC': float(mu), 'SD': float(sd), 'MeanFitAUC': float(fm),
            'MeanFeatureCount': float(r.FeatureCount.mean()), 'MeanEffectiveDimensions': float(r.Effective.mean()),
            'AvailableAllFolds': available, 'Objective': float(mu-.1*sd-.3*max(fm-mu-.1, 0)) if available else -1.})
    return result


def choose(grid, method=None):
    candidates = [r for r in grid if method is None or r['method'] == method]
    return max(candidates, key=lambda r: (r['AvailableAllFolds'], r['Objective'], -r['MeanEffectiveDimensions'],
        -r['MeanFeatureCount'], -r['C'], -r['candidate_id']))


def tune(phase, tables, y, te):
    path = OUT/'phases'/(phase+'.joblib')
    if path.exists():
        return
    audit, cv = [], {c['candidate_id']: [] for c in options()}
    for repeat, fold, a, b in folds(phase):
        assert not (set(a) | set(b)) & set(te) and (y[te] == -1).all()
        prepared = {key: prepare(d, a, b) for key, d in tables.items()}
        latents = {}
        for c in options():
            key = (c['image_key'], c['method'], c['components'])
            p = prepared[c['image_key']]
            if p['indices'] and key not in latents:
                latents[key] = reduction(p, y[a], c)
            f = fitted(p, y[a], c, latents.get(key))
            audit.append({'Repeat': repeat, 'InnerFold': fold, 'candidate_id': c['candidate_id'],
                'FitAUC': float(roc_auc_score(y[a], f['train_probability'])),
                'ValidationAUC': float(roc_auc_score(y[b], f['evaluation_probability'])),
                'Available': f['available'], 'FeatureCount': len(f['indices']), 'Effective': f['effective']})
            cv[c['candidate_id']].append({'Repeat': repeat, 'Fold': fold, 'train': a, 'validation': b,
                'fit_score': f['train_probability'], 'validation_score': f['evaluation_probability']})
        print('%s repeat%d fold%d/5 complete:66 pure-habitat configs' % (phase, repeat, fold), flush=True)
    result = {'phase': phase, 'audit': audit, 'cv': cv, 'grid': grid_scores(audit),
        'single_grid': grid_scores([r for r in audit if r['Repeat'] == 1])}
    joblib.dump(result, path, compress=3)


def selected(cp):
    return {'Adaptive': config(choose(cp['grid'])), **{m: config(choose(cp['grid'], m)) for m in ('Ridge', 'PCA', 'PLS')},
        'SingleInner': config(choose(cp['single_grid']))}


def train():
    protect()
    assert not (OUT/'training_frozen.json').exists()
    cohort, y, tr, te = previous.common.inputs('Fixed118')
    tables = data(); assert all(set(d['indices']) == set(tr) for d in tables.values())
    joblib.dump(tables, OUT/'training_features.joblib', compress=3)
    for phase in PHASES:
        tune(phase, tables, y, te)
    outer, metric, predictions = [], [], []
    for num, (a, b) in enumerate(previous.local.study.folds('Fixed118', 'outer'), 1):
        cp = joblib.load(OUT/'phases'/('outer%d.joblib' % num))
        models, cache = {}, {}
        for label, c in selected(cp).items():
            if c['candidate_id'] not in cache:
                cache[c['candidate_id']] = fit(tables, a, b, y, c)
            f = cache[c['candidate_id']]; models[label] = f
            metric.append({'ModelKey': label, 'Fold': num, **c,
                'FitAUC': float(roc_auc_score(y[a], f['train_probability'])),
                'AUC': float(roc_auc_score(y[b], f['evaluation_probability'])), 'FeatureCount': len(f['indices']), 'Effective': f['effective']})
            predictions.extend({'ModelKey': label, 'Fold': num, 'AnalysisID': cohort.AnalysisID.iloc[i],
                'Index': int(i), 'Outcome': int(y[i]), 'Probability': float(p)} for i, p in zip(b, f['evaluation_probability']))
        outer.append({'train': a, 'validation': b, 'models': models})
    cp = joblib.load(OUT/'phases/final.joblib')
    winner = choose(cp['grid']); c = config(winner)
    final = fit(tables, tr, [], y, c)
    repeated_oof = np.full((138, 3), np.nan)
    for row in cp['cv'][c['candidate_id']]:
        repeated_oof[row['validation'], row['Repeat']-1] = row['validation_score']
    assert np.isfinite(repeated_oof[tr]).all() and np.isnan(repeated_oof[te]).all()
    average_oof = repeated_oof[tr].mean(1)
    fp, tp, thresholds = roc_curve(y[tr], average_oof, drop_intermediate=False)
    threshold = float(thresholds[np.argmax(tp-fp)])
    assert np.isfinite(threshold)
    m = pd.DataFrame(metric); m = m[m.ModelKey == 'Adaptive']
    pred = pd.DataFrame(predictions); pred = pred[pred.ModelKey == 'Adaptive']
    choice = {**c, 'TrainFitAUC': float(roc_auc_score(y[tr], final['train_probability'])),
        'OuterMeanAUC': float(m.AUC.mean()), 'OuterSD': float(m.AUC.std(ddof=1)),
        'PooledOOFAUC': float(roc_auc_score(pred.Outcome, pred.Probability)), 'FinalInnerMeanAUC': winner['MeanValidationAUC'],
        'FeatureCount': len(final['indices']), 'EffectiveDimensions': final['effective'], 'Threshold': threshold,
        'NewTestScored': False, 'InnerRepeats': 3}
    result = {'train': tr, 'outer': outer, 'final': final, 'repeated_training_oof': repeated_oof,
        'threshold': threshold, 'choice': choice}
    joblib.dump(result, OUT/'models/pure_habitat.joblib', compress=3)
    csv('outer_metrics', metric); csv('nested_predictions', predictions); csv('training_choices', [choice])
    csv('inner_candidate_scores', [{'Phase': phase, **r} for phase in PHASES for r in joblib.load(OUT/'phases'/(phase+'.joblib'))['grid']])
    csv('selected_features', [{'Feature': name, 'Modality': name.split('__')[0], 'CoefficientInGaussianSpace': float(beta),
        'TrainingICC': next(r['ICC'] for r in final['audit'] if r['Feature'] == name)}
        for name, beta in zip(final['names'], final.get('coef_gaussian', []))])
    files = [p for p in OUT.rglob('*') if p.is_file()]
    save(OUT/'training_frozen.json', {'passed': True, 'training_only': True, 'new_test_calls': 0,
        'protocol_sha256': digest(OUT/'protocol.json'), 'files': {str(p.relative_to(OUT)): digest(p) for p in files}})
    print('Training-only candidate frozen:'+json.dumps(choice), flush=True)


def frozen():
    protect()
    f = json.loads((OUT/'training_frozen.json').read_text(encoding='utf-8'))
    assert f['passed'] and digest(OUT/'protocol.json') == f['protocol_sha256']
    for path, sha in f['files'].items():
        assert digest(OUT/path) == sha, path
    return f


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=('define', 'train'))
    with threadpool_limits(1):
        globals()[parser.parse_args().action]()
