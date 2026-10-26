"""Fixed90/48 pure-habitat development: train-only representation/model choice."""
import argparse
import json
from pathlib import Path
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
import run_k3_two_norm_random90_48_138_20261013 as baseline
import habitat_stable_views_20261013 as views
from habitat_stable_models_20261013 import BootstrapRidge
from regional_habitat_latent_model_20261012 import LatentLogisticModel

ROOT = baseline.ROOT
OUT = ROOT/'lvi_prediction/results/lvi_k3_stability_improvement90_20261013'
NORMALIZATIONS = baseline.NORMALIZATIONS
RANKINGS = ('Stability', 'Association')
save, digest, old = baseline.save, baseline.digest, baseline.old


def options():
    rows = []
    for norm in NORMALIZATIONS:
        for view in views.VIEWS:
            for ranking in RANKINGS:
                def add(method, top, C, components=0):
                    rows.append({'normalization': norm, 'view': view, 'ranking': ranking,
                                 'method': method, 'top': top, 'C': C, 'components': components,
                                 'candidate_id': len(rows)})
                for top in (20, 50, 0):
                    for C in (.003, .03, .3):
                        add('Ridge', top, C)
                    for C in (.03, .3, 1.):
                        add('ElasticNet', top, C)
                    for C in (.03, .3):
                        add('BagRidge', top, C)
                        for components in (1, 2):
                            add('PLS', top, C, components)
                for components in (2, 5):
                    for C in (.03, .3):
                        add('PCA', 0, C, components)
    assert len(rows) == 480
    return rows


def config(row):
    return {k: row[k] for k in options()[0]}


def choose(grid, norm=None, view=None):
    rows = [r for r in grid if (norm is None or r['normalization'] == norm) and (view is None or r['view'] == view)]
    valid = [r for r in rows if r['AvailableAllFolds']]
    return max(valid or rows, key=lambda r: (r['Objective'], -r['MeanEffectiveDimensions'],
        -r['MeanFeatureCount'], -r['C'], -r['candidate_id']))


def selections(grid):
    result = {n: choose(grid, n) for n in NORMALIZATIONS}
    result['Adaptive'] = choose(grid)
    return result


def protect():
    p = json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    for field in ('protected_hashes', 'source_hashes'):
        for name, sha in p[field].items():
            assert digest(ROOT/name) == sha, name
    return len(p['protected_hashes'])


def freeze_protocol():
    if OUT.exists():
        raise FileExistsError('Do not overwrite a prespecified improvement run')
    baseline.frozen()
    assert json.loads((baseline.OUT/'inference_verification.json').read_text())['passed']
    cohort, y, tr, te, _, _, _ = baseline.inputs()
    protected = dict(json.loads((baseline.OUT/'protocol.json').read_text(encoding='utf-8'))['protected_hashes'])
    protected.update({str(p.relative_to(ROOT)): digest(p) for p in baseline.OUT.rglob('*') if p.is_file()})
    protected.update(json.loads((baseline.OUT/'delivery_manifest.json').read_text(encoding='utf-8'))['source_hashes'])
    sources = [Path(__file__).resolve(), Path(views.__file__).resolve(), ROOT/'lvi_prediction/habitat_stable_models_20261013.py',
               ROOT/'lvi_prediction/regional_habitat_latent_model_20261012.py',
               ROOT/'lvi_prediction/verify_k3_stability_improvement90_20261013.py']
    OUT.mkdir(parents=True)
    (OUT/'models').mkdir()
    save(OUT/'protocol.json', {
        'date': '2026-10-13', 'authorization': 'human asked to improve both failed pure-habitat models on latest fixed90/48',
        'same90_48_seed20261013': True, 'split_manifest_source_sha256': digest(baseline.OUT/'split_manifest.json'),
        'train_ids': cohort.AnalysisID.iloc[tr].tolist(), 'test_ids': cohort.AnalysisID.iloc[te].tolist(),
        'fixed_K': 3, 'normalizations': NORMALIZATIONS, 'MRI_channels': ['T2', 'ADC', 'IVIM_D', 'IVIM_f'],
        'Dstar': False, 'clinical_predictors': False, 'whole_tumor_predictors': False,
        'views': {'Original': '1242 original K3 features', 'Compact': '321 prespecified17 texture metrics per4channels3habitats plus117 non-fraction descriptors',
                  'Invariant': '330 features: means/SD/ranges of the same3 habitats and3 pairwise relations, plus occupancy dispersion/max/entropy; no label-specific statistics'},
        'same_KMeans_partitions_and_ROIs': True, 'new_cross_patient_cluster_fitting': False,
        'feature_definition_outcome_independent': True, 'changed_habitat_representation': True,
        'quality_gate': 'training-fold ICC>=.8 completeness>=.9 nonconstant; no hard outcome sign-stability exclusion',
        'rankings': {'Stability': 'ICC only', 'Association': '20stratified bootstraps: abs(mean signed effect)/(SD effect+.05)*ICC^2; no sign cutoff'},
        'redundancy': 'train-fold absSpearman<.95; less aggressive than previous.8, collinearity handled with shrinkage/compression',
        'table_preprocessing': 'current training-fold median fill;P2.5/P97.5 clipping;median/IQR(1.348979500392164) scaling, fallback trainSD then1',
        'methods': 'Ridge,ElasticNet l1ratio.5 saga maxiter10000 tol1e-4,PCA2/5+LR,PLS1/2+LR,15stratified-bootstrap Ridge average',
        'candidates': options(), 'candidate_count': 480,
        'selection_objective': 'mean inner AUC-.10SD-.30max(mean fit-mean validation-.10,0); remove old penalty targeting fitAUC.75',
        'constant_or_nonconverged_candidate': 'unavailable if any inner fold has constant training probabilities, missing feature set or convergence warning',
        'nested': 'same outer5 seed42; inner5 seeds62201..62205; final5 seed62212; all representation/ranking/budget/method/parameter choices inside innerCV',
        'finalists': 'one training-selected improved model pernormalization; additional adaptive reference jointly chooses norm inside innerCV; final adaptive aliases selected norm model',
        'ablations': 'six norm/view choices assessed by outerCV, training-only; not separately scored on48',
        'test_stage': 'training verification then freeze; score both norm finalists in one joint48 stage,2original model calls; adaptive reference aliases its new score',
        'test_seen_in_previous_rounds': True, 'exploratory_method_development': True, 'new_external_validation': False,
        'no_test_score_in_candidate_selection': True, 'no_automatic_adoption': True, 'formal_model_use': False,
        'protected_hashes': protected, 'source_hashes': {str(p.relative_to(ROOT)): digest(p) for p in sources},
        'sources': ['https://scikit-learn.org/1.3/modules/generated/sklearn.linear_model.LogisticRegression.html',
                    'https://scikit-learn.org/1.3/modules/generated/sklearn.cross_decomposition.PLSRegression.html',
                    'https://scikit-learn.org/1.3/modules/generated/sklearn.decomposition.PCA.html']})
    save(OUT/'split_manifest.json', json.loads((baseline.OUT/'split_manifest.json').read_text(encoding='utf-8')))
    return cohort, y, tr, te


def ranking(data, a, y):
    rr = old.common.rows(data, a)
    measured = np.stack([data[k][rr] for k in views.KINDS], axis=1)
    complete = np.isfinite(measured).all(1).mean(0)
    raw = data['base'][rr]
    median = np.nanmedian(np.where(np.isfinite(raw), raw, np.nan), axis=0)
    median = np.where(np.isfinite(median), median, 0.)
    raw = np.where(np.isfinite(raw), raw, median)
    icc = np.full(raw.shape[1], np.nan)
    for j in range(raw.shape[1]):
        valid = np.isfinite(measured[:, :, j]).all(1)
        if complete[j] >= .9 and valid.sum() >= 3 and raw[:, j].std() > 1e-12:
            icc[j] = old.common.old.engine.icc_absolute(measured[valid, :, j])
    eligible = (complete >= .9)&np.isfinite(icc)&(icc >= .8)&(raw.std(0) > 1e-12)
    for j, m in enumerate(data['metadata']):
        if m['name'].endswith('_TumorFraction'):
            eligible[j] = False
    target = y[a]
    pos, neg = np.flatnonzero(target == 1), np.flatnonzero(target == 0)
    rng = np.random.default_rng(65313+int(np.sum(a)))
    effects = []
    for _ in range(20):
        ix = np.r_[rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)]
        effects.append(old.common.signed_effect(raw[ix], target[ix]))
    effects = np.asarray(effects)
    score = np.abs(effects.mean(0))/(effects.std(0, ddof=1)+.05)*np.maximum(np.nan_to_num(icc), 0.)**2
    ordered = {'Stability': sorted(np.flatnonzero(eligible), key=lambda j: (-icc[j], data['metadata'][j]['name'])),
               'Association': sorted(np.flatnonzero(eligible), key=lambda j: (-score[j], -icc[j], data['metadata'][j]['name']))}
    audit = [{'Feature': m['name'], 'ICC': float(icc[j]) if np.isfinite(icc[j]) else None,
              'CompleteFraction': float(complete[j]), 'Eligible': bool(eligible[j]),
              'BootstrapEffect': float(effects[:, j].mean()), 'BootstrapEffectSD': float(effects[:, j].std(ddof=1)),
              'AssociationScore': float(score[j])} for j, m in enumerate(data['metadata'])]
    return ordered, rankdata(raw, axis=0), audit


def prepare(data, a, b, y, c, ranked=None):
    order, ranks, audit = ranking(data, a, y) if ranked is None else ranked
    selected = []
    for j in order[c['ranking']]:
        if all(abs(float(np.corrcoef(ranks[:, j], ranks[:, k])[0, 1])) < .95 for k in selected):
            selected.append(j)
        if c['top'] and len(selected) >= c['top']:
            break
    xa = data['base'][old.common.rows(data, a)][:, selected]
    xb = data['base'][old.common.rows(data, b)][:, selected]
    if selected:
        median = np.nanmedian(np.where(np.isfinite(xa), xa, np.nan), axis=0)
        xa = np.where(np.isfinite(xa), xa, median)
        low, high = np.quantile(xa, [.025, .975], axis=0)
        xa = np.clip(xa, low, high)
        center = np.median(xa, axis=0)
        scale = np.subtract(*np.quantile(xa, [.75, .25], axis=0))/1.348979500392164
        scale = np.where(scale > 1e-12, scale, xa.std(0))
        scale = np.where(scale > 1e-12, scale, 1.)
        xa = (xa-center)/scale
        xb = (np.clip(np.where(np.isfinite(xb), xb, median), low, high)-center)/scale
    else:
        median, low, high, center, scale = [np.empty(0) for _ in range(5)]
    return {'indices': selected, 'names': [data['metadata'][j]['name'] for j in selected], 'audit': audit,
            'median': median, 'low': low, 'high': high, 'center': center, 'scale': scale, 'xa': xa, 'xb': xb}


def prefix(full, top):
    n = len(full['indices']) if top == 0 else min(top, len(full['indices']))
    return {**full, 'indices': full['indices'][:n], 'names': full['names'][:n],
            **{s: full[s][:n] for s in ('median', 'low', 'high', 'center', 'scale')},
            'xa': full['xa'][:, :n], 'xb': full['xb'][:, :n]}


def fitted(p, target, c, seed):
    n = len(p['indices'])
    caught = []
    if n:
        if c['method'] == 'Ridge':
            estimator = LogisticRegression(C=c['C'], penalty='l2', solver='liblinear', max_iter=3000, random_state=64212)
        elif c['method'] == 'ElasticNet':
            estimator = LogisticRegression(C=c['C'], penalty='elasticnet', l1_ratio=.5, solver='saga',
                                           max_iter=10000, tol=1e-4, random_state=64212)
        elif c['method'] == 'BagRidge':
            estimator = BootstrapRidge(c['C'], seed)
        else:
            estimator = LatentLogisticModel(c['method'], c['components'], c['C'])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            estimator.fit(p['xa'], target)
        pa = estimator.predict_proba(p['xa'])[:, 1]
        pb = estimator.predict_proba(p['xb'])[:, 1] if len(p['xb']) else np.empty(0)
        effective = (min(c['components'], n) if c['method'] in ('PCA', 'PLS') else
                     int(np.count_nonzero(np.abs(estimator.coef_[0]) > 1e-10)) if c['method'] != 'BagRidge' else n)
    else:
        estimator, effective = None, 0
        pa, pb = np.full(len(target), target.mean()), np.full(len(p['xb']), target.mean())
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    available = bool(n and converged and np.ptp(pa) > 1e-12)
    return {**{s: p[s] for s in ('indices', 'names', 'audit', 'median', 'low', 'high', 'center', 'scale')},
            'config': c, 'model': estimator, 'prevalence': float(target.mean()),
            'train_probability': pa, 'evaluation_probability': pb, 'effective_dimensions': effective,
            'converged': converged, 'available': available}


def fit(data, a, b, y, c):
    return fitted(prepare(data[(c['normalization'], c['view'])], a, b, y, c), y[a], c, 66413+int(np.sum(a)))


def tune(data, folds, y, label):
    opts = options()
    values, cv, audit = {c['candidate_id']: [] for c in opts}, {c['candidate_id']: [] for c in opts}, []
    for fold, (a, b) in enumerate(folds, 1):
        ranks = {key: ranking(d, a, y) for key, d in data.items()}
        fullsets, cache, fitted_cache = {}, {}, {}
        for c in opts:
            identity = (c['normalization'], c['view'], c['ranking'], c['top'])
            if identity not in cache:
                if identity[:3] not in fullsets:
                    key = identity[:2]
                    fullsets[identity[:3]] = prepare(data[key], a, b, y, {**c, 'top': 0}, ranks[key])
                cache[identity] = prefix(fullsets[identity[:3]], c['top'])
            modelkey = identity[:3]+(tuple(cache[identity]['indices']), c['method'], c['C'], c['components'])
            if modelkey not in fitted_cache:
                fitted_cache[modelkey] = fitted(cache[identity], y[a], c, 66413+int(np.sum(a)))
            f = fitted_cache[modelkey]
            v = (float(roc_auc_score(y[b], f['evaluation_probability'])), float(roc_auc_score(y[a], f['train_probability'])),
                 len(f['indices']), f['effective_dimensions'], f['available'], f['converged'])
            values[c['candidate_id']].append(v)
            cv[c['candidate_id']].append((b, f['evaluation_probability']))
            audit.append({'Phase': label, 'InnerFold': fold, 'candidate_id': c['candidate_id'], 'ValidationAUC': v[0],
                          'FitAUC': v[1], 'FeatureCount': v[2], 'EffectiveDimensions': v[3], 'Available': v[4], 'Converged': v[5]})
        print(label+' inner'+str(fold)+'/5 complete', flush=True)
    grid = []
    for c in opts:
        v = np.asarray(values[c['candidate_id']])
        mean, sd, train = float(v[:, 0].mean()), float(v[:, 0].std(ddof=1)), float(v[:, 1].mean())
        available = bool(v[:, 4].all())
        score = mean-.10*sd-.30*max(train-mean-.10, 0)
        grid.append({**c, 'MeanValidationAUC': mean, 'SD': sd, 'MeanFitAUC': train, 'MeanFeatureCount': float(v[:, 2].mean()),
                     'MeanEffectiveDimensions': float(v[:, 3].mean()), 'AvailableAllFolds': available,
                     'Objective': score if available else -1.})
    return grid, cv, audit


def train():
    start = time.time()
    cohort, y, tr, te = freeze_protocol()
    raw = joblib.load(baseline.OUT/'training_features.joblib')
    for n in NORMALIZATIONS:
        np.testing.assert_array_equal(raw[n]['indices'], tr)
    data = views.build(raw)
    joblib.dump(data, OUT/'training_features.joblib', compress=3)
    predictions, metrics, ablations, grids, inner_audit, frequency = [], [], [], [], [], []
    for fold, (a, b) in enumerate(old.common.old.previous.folds(tr, y, 42, 5), 1):
        grid, _, audit = tune(data, old.common.old.previous.folds(a, y, 62200+fold, 5), y, 'outer'+str(fold))
        selected = selections(grid)
        selected.update({n+'__'+v: choose(grid, n, v) for n in NORMALIZATIONS for v in views.VIEWS})
        models, cache = {}, {}
        for name, row in selected.items():
            c = config(row)
            if c['candidate_id'] not in cache:
                cache[c['candidate_id']] = fit(data, a, b, y, c)
            f = cache[c['candidate_id']]
            models[name] = f
            result = {'ModelKey': name, 'Fold': fold, **c, 'FeatureCount': len(f['indices']),
                      'EffectiveDimensions': f['effective_dimensions'], 'AUC': float(roc_auc_score(y[b], f['evaluation_probability']))}
            if name in NORMALIZATIONS or name == 'Adaptive':
                metrics.append(result)
                predictions.extend({'ModelKey': name, 'Fold': fold, 'AnalysisID': cohort.AnalysisID.iloc[i],
                                    'Outcome': int(y[i]), 'Probability': float(p)} for i, p in zip(b, f['evaluation_probability']))
                frequency.extend({'ModelKey': name, 'Fold': fold, 'Feature': feature, 'View': c['view'], 'Normalization': c['normalization']}
                                 for feature in f['names'])
            else:
                ablations.append(result)
        joblib.dump({'train': a, 'evaluation': b, 'grid': grid, 'models': models}, OUT/('models/outer_fold_'+str(fold)+'.joblib'), compress=3)
        grids.extend({'Phase': 'outer'+str(fold), **r} for r in grid)
        inner_audit.extend(audit)
        print('outer'+str(fold)+'/5 complete', flush=True)
    grid, cv, audit = tune(data, old.common.old.previous.folds(tr, y, 62212, 5), y, 'final')
    inner_audit.extend(audit)
    final = {'train': tr, 'models': {}, 'formal_model_use': False}
    choices, features, trainpred, oofpred, gates, cache = [], [], [], [], [], {}
    nested, foldmetrics = pd.DataFrame(predictions), pd.DataFrame(metrics)
    for name, row in selections(grid).items():
        c = config(row)
        if c['candidate_id'] not in cache:
            f = fit(data, tr, [], y, c)
            assert f['available'], 'Training-selected final model must converge and have nonconstant probabilities'
            oof = np.full(138, np.nan)
            for ix, p in cv[c['candidate_id']]:
                oof[ix] = p
            assert np.isfinite(oof[tr]).all() and np.isnan(oof[te]).all()
            f['threshold'] = old.common.threshold(y[tr], oof[tr])
            cache[c['candidate_id']] = f, oof
        f, oof = cache[c['candidate_id']]
        final['models'][name] = f
        pp = nested[nested.ModelKey == name].set_index('AnalysisID').loc[cohort.AnalysisID.iloc[tr]]
        ff = foldmetrics[foldmetrics.ModelKey == name].AUC
        meta = data[(c['normalization'], c['view'])]['metadata']
        choices.append({'ModelKey': name, **c, 'FeatureCount': len(f['indices']), 'EffectiveDimensions': f['effective_dimensions'],
                        'TrainFitAUC': float(roc_auc_score(y[tr], f['train_probability'])),
                        'OuterMeanAUC': float(ff.mean()), 'OuterSD': float(ff.std(ddof=1)),
                        'PooledOOFAUC': float(roc_auc_score(y[tr], pp.Probability)),
                        'FinalSelectionCVMeanAUC': row['MeanValidationAUC'], 'SelectionObjective': row['Objective'],
                        'Threshold': f['threshold'], **{m+'Count': sum(meta[j]['modality'] == m for j in f['indices']) for m in ('T2', 'ADC', 'IVIM')}})
        beta = f['model'].coefficients_.mean(0) if c['method'] == 'BagRidge' else f['model'].coef_[0]
        features.extend({'ModelKey': name, 'Feature': feature, 'Modality': meta[j]['modality'], 'Block': meta[j]['block'],
                         'StandardizedCoefficient': float(b), 'CoefficientScope': 'mean_member_diagnostic' if c['method'] == 'BagRidge' else 'exact_linear_logit'}
                        for j, feature, b in zip(f['indices'], f['names'], beta))
        trainpred.extend({'ModelKey': name, 'AnalysisID': cohort.AnalysisID.iloc[i], 'Outcome': int(y[i]), 'Probability': float(p)}
                         for i, p in zip(tr, f['train_probability']))
        oofpred.extend({'ModelKey': name, 'AnalysisID': cohort.AnalysisID.iloc[i], 'Outcome': int(y[i]), 'Probability': float(oof[i])} for i in tr)
    for (n, v), d in data.items():
        _, _, audit = ranking(d, tr, y)
        gates.extend({'Normalization': n, 'View': v, **r} for r in audit)
    joint = final['models']['Adaptive']
    assert joint is final['models'][joint['config']['normalization']]
    joblib.dump(final, OUT/'models/final_models.joblib', compress=3)
    for name, rows in [('nested_predictions', nested), ('outer_fold_metrics', foldmetrics), ('view_ablations_outer_metrics', ablations),
                       ('inner_candidate_scores', grids), ('inner_fold_audit', inner_audit), ('final_training_grid', grid),
                       ('final_training_choices', choices), ('selected_features', features), ('train_fit_predictions', trainpred),
                       ('final_selection_oof_predictions', oofpred), ('feature_gate', gates), ('outer_feature_selections', frequency)]:
        pd.DataFrame(rows).to_csv(OUT/(name+'.csv'), index=False, encoding='utf-8-sig')
    save(OUT/'training_frozen.json', {'passed': True, 'model_sha256': digest(OUT/'models/final_models.joblib'),
        'protocol_sha256': digest(OUT/'protocol.json'), 'feature_sha256': digest(OUT/'training_features.joblib'),
        'training_csv_hashes': {p.name: digest(p) for p in OUT.glob('*.csv')}, 'test_scored': False,
        'protected_count': protect(), 'elapsed_seconds': time.time()-start})
    print(pd.DataFrame(choices)[['ModelKey', 'normalization', 'view', 'method', 'FeatureCount', 'EffectiveDimensions',
          'TrainFitAUC', 'OuterMeanAUC', 'OuterSD', 'PooledOOFAUC']].to_string(index=False), flush=True)


def frozen():
    fr = json.loads((OUT/'training_frozen.json').read_text(encoding='utf-8'))
    protect()
    for path, field in [('models/final_models.joblib', 'model_sha256'), ('protocol.json', 'protocol_sha256'), ('training_features.joblib', 'feature_sha256')]:
        assert digest(OUT/path) == fr[field]
    for name, sha in fr['training_csv_hashes'].items():
        assert digest(OUT/name) == sha
    return fr


def predict(f, d):
    x = d['base'][:, f['indices']]
    x = (np.clip(np.where(np.isfinite(x), x, f['median']), f['low'], f['high'])-f['center'])/f['scale']
    return f['model'].predict_proba(x)[:, 1] if f['model'] is not None else np.full(len(x), f['prevalence'])


def test():
    if (OUT/'test_started.json').exists():
        raise FileExistsError('Do not repeat48 model prediction')
    fr = frozen()
    assert json.loads((OUT/'training_verification.json').read_text())['passed']
    cohort, _, tr, te, _, _, _ = baseline.inputs()
    final = joblib.load(OUT/'models/final_models.joblib')
    raw = joblib.load(baseline.OUT/'test_features.joblib')
    for n in NORMALIZATIONS:
        np.testing.assert_array_equal(raw[n]['indices'], te)
    data = views.build(raw)
    joblib.dump(data, OUT/'test_features.joblib', compress=3)
    save(OUT/'test_started.json', {'n': 48, 'joint_stage_count': 1, 'distinct_models': 2,
                                  'adaptive_reference_alias': True, 'model_sha256': fr['model_sha256']})
    scores = {}
    for n in NORMALIZATIONS:
        f = final['models'][n]
        scores[n] = predict(f, data[(n, f['config']['view'])])
    scores['Adaptive'] = scores[final['models']['Adaptive']['config']['normalization']]
    y = cohort.Outcome.to_numpy(int)[te]
    previous_scores = pd.read_csv(baseline.OUT/'test_predictions.csv', float_precision='round_trip')
    comparators = {**scores}
    for n in NORMALIZATIONS:
        rows = previous_scores[previous_scores.Normalization == n].set_index('AnalysisID').loc[cohort.AnalysisID.iloc[te]]
        np.testing.assert_array_equal(rows.Outcome, y)
        comparators['Baseline__'+n] = rows.Probability.to_numpy()
    draws = old.common.old.retained.bootstrap(y, comparators)
    predictions, metrics, pairs = [], [], []
    for name, p in scores.items():
        f = final['models'][name]
        threshold = f['threshold']
        predictions.extend({'ModelKey': name, 'AnalysisID': cohort.AnalysisID.iloc[i], 'Outcome': int(v),
                            'Probability': float(pp), 'Threshold': threshold, 'PredictedPositive': int(pp >= threshold)}
                           for i, v, pp in zip(te, y, p))
        lo, hi = np.quantile(draws[name], [.025, .975])
        metrics.append({'ModelKey': name, 'Normalization': f['config']['normalization'], 'N': 48, 'Positive': int(y.sum()),
                        'Negative': int((y == 0).sum()), **old.common.old.metrics(y, p, threshold),
                        'AUC_CI95Low': float(lo), 'AUC_CI95High': float(hi)})
    for first, second in [('PercentileZ', 'RobustZ'), ('PercentileZ', 'Baseline__PercentileZ'),
                           ('RobustZ', 'Baseline__RobustZ'), ('Adaptive', 'Baseline__PercentileZ'), ('Adaptive', 'Baseline__RobustZ')]:
        lo, hi = np.quantile(draws[first]-draws[second], [.025, .975])
        pairs.append({'First': first, 'Second': second,
                      'DeltaAUC': float(roc_auc_score(y, comparators[first])-roc_auc_score(y, comparators[second])),
                      'CI95Low': float(lo), 'CI95High': float(hi)})
    for name, rows in [('test_predictions', predictions), ('test_metrics', metrics), ('test_paired_auc_comparisons', pairs)]:
        pd.DataFrame(rows).to_csv(OUT/(name+'.csv'), index=False, encoding='utf-8-sig')
    save(OUT/'test_complete.json', {'passed': True, 'n': 48, 'prediction_rows': 144, 'model_predict_calls': 2,
        'adaptive_alias_normalization': final['models']['Adaptive']['config']['normalization'], 'baseline_model_predict_calls': 0,
        'no_refit_after_freeze': True, 'protected_count': protect()})
    print(pd.DataFrame(metrics).to_string(index=False), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['train', 'test'], required=True)
    args = p.parse_args()
    with threadpool_limits(limits=1):
        {'train': train, 'test': test}[args.phase]()
