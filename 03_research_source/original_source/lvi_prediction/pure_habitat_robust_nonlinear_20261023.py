"""Robust contour summaries and restrained nonlinear classifiers, pure habitat only."""
import warnings
import numpy as np
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.preprocessing import QuantileTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
import run_pure_habitat_repeatedinner_20261023 as reference

SEED = 64212


def options():
    result = []
    learners = [('Ridge', .03), ('Ridge', .3), ('PCA', .03), ('PCA', .3),
                ('RBF', 1.), ('ExtraTrees2', 0.), ('ExtraTrees4', 0.), ('HistGB3', 0.)]
    for norm in ('PercentileZ', 'RobustZ'):
        for view in ('OrderedCompact', 'ContrastOnly'):
            for mode in ('Base', 'ContourMedian'):
                for icc in (.60, .75):
                    for method, C in learners:
                        result.append({'candidate_id': len(result), 'Family': 'Improved',
                            'CandidateKey': 'I%03d' % len(result), 'normalization': norm, 'view': view,
                            'image_key': norm+'__'+view, 'mask_mode': mode, 'icc_min': icc,
                            'method': method, 'components': 2 if method == 'PCA' else 0,
                            'C': C, 'gamma': 'scale' if method == 'RBF' else 'NA'})
    assert len(result) == 128
    return result


def reference_options():
    return [{**c, 'Family': 'Reference', 'CandidateKey': 'R%03d' % c['candidate_id']} for c in reference.options()]


def values(d, indices, mode):
    rr = reference.rows(d, indices)
    if mode == 'Base':
        return d['base'][rr]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmedian(np.stack([d[kind][rr] for kind in reference.feature.KINDS], axis=1), axis=1)


def quality(d, a):
    rr = reference.rows(d, a)
    measured = np.stack([d[kind][rr] for kind in reference.feature.KINDS], axis=1)
    complete = np.isfinite(measured).all(1).mean(0)
    icc = np.full(measured.shape[-1], np.nan)
    for j in np.flatnonzero(complete >= .9):
        valid = np.isfinite(measured[:, :, j]).all(1)
        if valid.sum() >= 3 and np.ptp(measured[valid, 0, j]) > 1e-12:
            icc[j] = reference.previous.local.stable.old.common.old.engine.icc_absolute(measured[valid, :, j])
    return complete, icc


def prepare(d, a, b, c, qc=None):
    complete, icc = quality(d, a) if qc is None else qc
    raw = values(d, a, c['mask_mode'])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        median = np.nanmedian(np.where(np.isfinite(raw), raw, np.nan), axis=0)
    median = np.where(np.isfinite(median), median, 0.)
    filled = np.where(np.isfinite(raw), raw, median)
    eligible = np.flatnonzero((complete >= .9) & np.isfinite(icc) & (icc >= c['icc_min']) & (filled.std(0) > 1e-12))
    ordered = sorted(eligible, key=lambda j: (-icc[j], d['metadata'][j]['name']))
    selected = []
    if len(eligible):
        corr = np.atleast_2d(np.corrcoef(rankdata(filled[:, eligible], axis=0), rowvar=False))
        lookup = {int(j): n for n, j in enumerate(eligible)}
        for j in ordered:
            if all(abs(corr[lookup[j], lookup[k]]) < .95 for k in selected):
                selected.append(j)
    xa, med = filled[:, selected], median[selected]
    xb = values(d, b, c['mask_mode'])[:, selected]
    xb = np.where(np.isfinite(xb), xb, med)
    qt = None
    if selected:
        qt = QuantileTransformer(n_quantiles=len(a), output_distribution='normal', subsample=len(a), random_state=SEED).fit(xa)
        xa = np.clip(qt.transform(xa), -2.5, 2.5)
        xb = np.clip(qt.transform(xb), -2.5, 2.5) if len(b) else np.empty((0, len(selected)))
    audit = [{'Feature': m['name'], 'CompleteFraction': float(complete[j]),
        'ICC': float(icc[j]) if np.isfinite(icc[j]) else None,
        'Eligible': bool(j in eligible), 'Selected': bool(j in selected)} for j, m in enumerate(d['metadata'])]
    return {'indices': selected, 'names': [d['metadata'][j]['name'] for j in selected],
        'median': med, 'quantile': qt, 'audit': audit, 'xa': xa, 'xb': xb}


def fitted(p, target, c, latent=None):
    if c['method'] in ('Ridge', 'PCA'):
        result = reference.fitted(p, target, c, latent)
        result['ScoreType'] = 'LR_probability_uncalibrated'
        return result
    if not p['indices']:
        return {**{k: p[k] for k in ('indices', 'names', 'median', 'quantile', 'audit')},
            'config': c, 'classifier': None, 'available': False, 'effective': 0,
            'ScoreType': 'constant_prevalence', 'prevalence': float(target.mean()),
            'train_probability': np.full(len(target), target.mean()),
            'evaluation_probability': np.full(len(p['xb']), target.mean())}
    leaf = max(8, int(np.ceil(.15*len(target))))
    if c['method'] == 'RBF':
        model = SVC(C=1., kernel='rbf', gamma='scale', class_weight='balanced', probability=False, random_state=SEED)
    elif c['method'].startswith('ExtraTrees'):
        model = ExtraTreesClassifier(n_estimators=100, max_depth=int(c['method'][-1]), min_samples_leaf=leaf,
            max_features='sqrt', class_weight='balanced', bootstrap=False, n_jobs=1, random_state=SEED)
    else:
        model = HistGradientBoostingClassifier(max_iter=100, learning_rate=.05, max_leaf_nodes=3, max_depth=2,
            min_samples_leaf=leaf, l2_regularization=5., early_stopping=False, class_weight='balanced', random_state=SEED)
    model.fit(p['xa'], target)
    def score(x):
        if not len(x):
            return np.empty(0)
        return expit(model.decision_function(x)) if c['method'] == 'RBF' else model.predict_proba(x)[:, 1]
    pa, pb = score(p['xa']), score(p['xb'])
    return {**{k: p[k] for k in ('indices', 'names', 'median', 'quantile', 'audit')},
        'config': c, 'classifier': model, 'available': bool(np.ptp(pa) > 1e-12), 'effective': len(p['indices']),
        'min_leaf_patients': leaf if c['method'] != 'RBF' else None, 'prevalence': float(target.mean()),
        'ScoreType': 'sigmoid_SVM_margin' if c['method'] == 'RBF' else 'tree_probability_uncalibrated',
        'train_probability': pa, 'evaluation_probability': pb}


def fit(tables, a, b, y, c):
    if c['Family'] == 'Reference':
        f = reference.fit(tables, a, b, y, c)
        f['ScoreType'] = 'LR_probability_uncalibrated'
        return f
    return fitted(prepare(tables[c['image_key']], a, b, c), y[a], c)


def transformed(f, tables, indices):
    if f['config']['Family'] == 'Reference':
        return reference.transformed(f, tables, indices)
    x = values(tables[f['config']['image_key']], indices, f['config']['mask_mode'])[:, f['indices']]
    x = np.where(np.isfinite(x), x, f['median'])
    return np.clip(f['quantile'].transform(x), -2.5, 2.5) if f['indices'] and len(indices) else np.empty((len(indices), len(f['indices'])))


def predict(f, tables, indices):
    if f['classifier'] is None:
        return np.full(len(indices), f['prevalence'])
    if f['config']['Family'] == 'Reference':
        return reference.predict(f, tables, indices)
    x = transformed(f, tables, indices)
    if f['config']['method'] == 'PCA':
        x = (f['reducer'].transform(x)-f['latent_center'])/f['latent_scale']
    return expit(f['classifier'].decision_function(x)) if f['config']['method'] == 'RBF' else f['classifier'].predict_proba(x)[:, 1]


def used_columns(f):
    model, method = f['classifier'], f['config']['method']
    if model is None:
        return set()
    if method.startswith('ExtraTrees'):
        return {int(j) for tree in model.estimators_ for j in tree.tree_.feature if j >= 0}
    if method == 'HistGB3':
        return {int(n['feature_idx']) for iteration in model._predictors for tree in iteration for n in tree.nodes if not n['is_leaf']}
    if method in ('Ridge', 'PCA', 'PLS'):
        return set(np.flatnonzero(np.abs(f['coef_gaussian']) > 1e-12).tolist())
    return set(range(len(f['indices'])))
