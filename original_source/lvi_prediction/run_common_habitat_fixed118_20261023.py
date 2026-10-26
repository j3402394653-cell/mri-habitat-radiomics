# -*- coding: utf-8 -*-
"""Pure habitat improvement: quantitative fold-common centroids and regional contrasts."""
import os
os.environ.update({'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
import argparse
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from threadpoolctl import threadpool_limits
import common_quantitative_habitats_20261023 as feature
import run_filtered_ordered_fixed118_20261023 as prior

ROOT = prior.ROOT
OUT = ROOT/'lvi_prediction/results/lvi_common_quantitative_habitat_fixed118_20261023'
save, digest = prior.save, prior.digest
inputs = prior.study.inputs


def csv(name, rows):
    pd.DataFrame(rows).to_csv(OUT/(name+'.csv'), index=False, encoding='utf-8-sig')


def read(name):
    return pd.read_csv(OUT/(name+'.csv'), float_precision='round_trip', keep_default_na=False, na_values=[''])


def options():
    result = []
    for normalization in ('PercentileZ', 'RobustZ'):
        for k in (2, 3, 4):
            for panel in ('Spatial', 'Combined'):
                for top in (5, 15, 0):
                    for method, c in [('Ridge', .003), ('Ridge', .03), ('Ridge', .3), ('ShrinkageLDA', 0.)]:
                        result.append({'candidate_id': len(result), 'normalization': normalization, 'K': k,
                            'panel': panel, 'top': top, 'method': method, 'C': c})
    assert len(result) == 144
    return result


def cfg(r):
    return {k: r[k] for k in options()[0]}


def prepare():
    assert not OUT.exists()
    prior.frozen()
    OUT.mkdir(parents=True)
    for folder in ('patient_features', 'phases', 'models', 'fold_tables', 'Fixed118', 'figures'):
        (OUT/folder).mkdir()
    import run_random120_retained_methods_20261010 as loader
    profiles = loader.load_profiles()
    _, y, tr, te = inputs('Fixed118')
    for num, i in enumerate(tr, 1):
        joblib.dump(feature.compact_patient(profiles[int(i)], int(i)), OUT/'patient_features'/('case_%03d.joblib' % i), compress=3)
        if num % 10 == 0:
            print('Compact training profiles %d/82' % num, flush=True)
    assert (y[te] == -1).all()
    for name in ('Train82_cases.csv', 'Test36_cases.csv', 'Sample118_cases.csv', 'nested5_folds.json'):
        (OUT/'Fixed118'/name).write_bytes((prior.OUT/'Fixed118'/name).read_bytes())
    (OUT/'sample_manifest.json').write_bytes((prior.OUT/'sample_manifest.json').read_bytes())
    save(OUT/'preparation.json', {'passed': True, 'training_cases': 82, 'test_feature_cases': 0,
        'outcomes_used': False, 'Dstar': False})


def define():
    assert not (OUT/'protocol.json').exists() and not list((OUT/'phases').glob('*.joblib'))
    assert json.loads((OUT/'preparation.json').read_text(encoding='utf-8'))['passed']
    prior.frozen()
    previous = json.loads((prior.OUT/'protocol.json').read_text(encoding='utf-8'))
    protected = {**previous['protected_hashes'], **previous['source_hashes']}
    protected.update({str(p.relative_to(ROOT)): digest(p) for p in prior.OUT.rglob('*') if p.is_file()})
    protected[str(ROOT.joinpath('fold_locked_lvi_study/isotropic1mm_zscore_firstorder_v1/slic_supervoxel_100_10x5/profiles.joblib').relative_to(ROOT))] = digest(ROOT/'fold_locked_lvi_study/isotropic1mm_zscore_firstorder_v1/slic_supervoxel_100_10x5/profiles.joblib')
    sources = [ROOT/'lvi_prediction'/name for name in ('common_quantitative_habitats_20261023.py',
        'run_common_habitat_fixed118_20261023.py', 'verify_common_habitat_fixed118_20261023.py',
        'report_common_habitat_fixed118_20261023.py')]
    method = {'date': '2026-10-23', 'authorization': 'Pure habitat is the primary target;continue improving it;other families auxiliary only.',
        'cohort': 'Exact original118;82training58positive24negative;36test25positive11negative;no redraw.',
        'CV': 'Same outer5 and inner5 folds as last comparison;all centroids,scalers,feature selection,classifier and K fitted inside current training fold.',
        'pure_habitat': True, 'clinical': False, 'whole_tumor_predictors': False, 'Dstar': False,
        'construction': 'Existing outcome-blind native-space SLIC units;patient-balanced common KMeans centroids fitted independently in T2,ADC,IVIM.',
        'inputs': 'T2 patient-relative z intensity,3mm mean,SD;ADC original map values of same3;IVIM original D/f map values of same6.',
        'units': 'Original quantitative numbers recovered from stored unclipped mean/SD;not a new certification of physical units.',
        'normalization_scope': 'Across current training-fold quantitative supervoxel samples,not the previous within-patient normalization comparison. T2 is patient-relative before this step.',
        'normalizations': 'Fold P1/P99 clipping+mean/SD,or fold median/IQR(1.34898) robust scaling. Parameters frozen for validation/test.',
        'sampling': 'Exactly min(64,minimum patient tumor-unit count) tumor supervoxels per training patient;fixed per-patient random order;no outcome access.',
        'K': [2, 3, 4], 'identities': 'Common labels ordered by T2/ADC intensity or IVIM D then f;never reorder independently within a held-out case.',
        'features': 'Exact voxel-weighted habitat fractions,inner2mm periphery/core composition,6-connected contact fractions,composition entropy;regional original-value mean/SD,supervoxel mean3mmSD,D-f correlation,pairwise mean differences and relative contrasts.',
        'panels': ['Spatial', 'Combined'], 'no_texture_matrices': True, 'no_global_shape': True,
        'quality': 'Current training-fold contour perturbation ICC>=.75,complete>=.9,nonconstant;20bootstrap Association ranking;absSpearman<.95.',
        'budgets': [5, 15, 'all eligible'], 'classifiers': 'Balanced ridge logistic C.003/.03/.3;LDA lsqr with automatic covariance shrinkage.',
        'objective': 'InnerMeanAUC-.10SD-.30max(meanFit-InnerMean-.10,0);ties prefer fewer features,smallerK/smallerC/candidateID.',
        'threshold': 'Youden threshold from complete inner5 OOF of full82 selected configuration;frozen before test.',
        'test': 'Only ONE adaptive pure-habitat final model scored on36 after training verification;spatial/combined/fixedK3 diagnostics training-only.',
        'prior_test_exposure': True, 'external_validation': False, 'formal_use': False, 'automatic_adoption': False,
        'excluded_experiments_used': False, 'candidate_count': 144,
        'sources': ['https://scikit-learn.org/1.3/modules/clustering.html#k-means',
                    'https://scikit-learn.org/1.3/common_pitfalls.html#data-leakage'],
        'protected_hashes': protected, 'source_hashes': {str(p.relative_to(ROOT)): digest(p) for p in sources}}
    save(OUT/'protocol.json', method)
    save(OUT/'definition_frozen.json', {'passed': True, 'protocol_sha256': digest(OUT/'protocol.json'),
        'test_scored': False, 'candidate_models_fitted': 0})
    csv('candidate_definitions', options())
    print('Definition frozen:144 pure-habitat candidates;all centroids fold-fitted;one final test call', flush=True)


def protect():
    p = json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    for field in ('protected_hashes', 'source_hashes'):
        for path, sha in p[field].items():
            assert digest(ROOT/path) == sha, path
    assert digest(OUT/'protocol.json') == json.loads((OUT/'definition_frozen.json').read_text(encoding='utf-8'))['protocol_sha256']
    return len(p['protected_hashes'])


def patients(indices):
    return {int(i): joblib.load(OUT/'patient_features'/('case_%03d.joblib' % int(i))) for i in indices}


def rows(d, a):
    lookup = {int(i): j for j, i in enumerate(d['indices'])}
    return [lookup[int(i)] for i in a]


def view(d, panel):
    ix = [j for j, m in enumerate(d['metadata']) if panel == 'Combined' or m['panel'] == 'Spatial']
    return {'indices': d['indices'], 'metadata': [d['metadata'][j] for j in ix],
        **{kind: d[kind][:, ix] for kind in feature.KINDS}}


def prepared(d, a, b, y):
    rank = prior.ranking(d, a, y)
    full = prior.stable.prepare(d, a, b, y, {'top': 0, 'ranking': 'Association'}, rank)
    return full


def fitted(p, target, c):
    model = None
    if p['indices']:
        if c['method'] == 'Ridge':
            model = LogisticRegression(C=c['C'], penalty='l2', solver='liblinear', class_weight='balanced', max_iter=3000, random_state=feature.SEED)
        else:
            model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        model.fit(p['xa'], target)
    def score(x):
        return np.empty(0) if not len(x) else model.predict_proba(x)[:, 1] if model is not None else np.full(len(x), target.mean())
    pa, pb = score(p['xa']), score(p['xb'])
    return {**{k: p[k] for k in ('indices', 'names', 'audit', 'median', 'low', 'high', 'center', 'scale')},
        'config': c, 'model': model, 'prevalence': float(target.mean()), 'train_probability': pa,
        'validation_probability': pb, 'available': bool(model is not None and np.ptp(pa) > 1e-12)}


def fit(d, a, b, y, c, full=None):
    dd = view(d, c['panel'])
    p = prior.stable.prefix(prepared(dd, a, b, y) if full is None else full, c['top'])
    return fitted(p, y[a], c)


def transformed(f, d, indices):
    dd = view(d, f['config']['panel'])
    raw = dd['base'][rows(dd, indices)][:, f['indices']]
    return (np.clip(np.where(np.isfinite(raw), raw, f['median']), f['low'], f['high'])-f['center'])/f['scale']


def predict(f, d, indices):
    xx = transformed(f, d, indices)
    return f['model'].predict_proba(xx)[:, 1] if f['model'] is not None else np.full(len(xx), f['prevalence'])


def choose(grid, panel=None, k=None):
    candidates = [r for r in grid if (panel is None or r['panel'] == panel) and (k is None or r['K'] == k)]
    return max(candidates, key=lambda r: (r['AvailableAllFolds'], r['Objective'],
        -r['MeanFeatureCount'], -r['K'], -r['C'], -r['candidate_id']))


def selections(grid):
    return {'Adaptive': choose(grid), 'Spatial': choose(grid, panel='Spatial'),
        'Combined': choose(grid, panel='Combined'), 'FixedK3': choose(grid, k=3)}


def phase_folds(phase):
    return prior.study.folds('Fixed118', 'final') if phase == 'final' else prior.study.folds('Fixed118', 'inner', int(phase[-1]))


def tune(phase, pp, y, te):
    dest = OUT/'phases'/(phase+'.joblib')
    if dest.exists():
        return
    audit, cv = [], {c['candidate_id']: [] for c in options()}
    for num, (a, b) in enumerate(phase_folds(phase), 1):
        assert not (set(a) | set(b)) & set(te) and (y[te] == -1).all()
        fold_file = OUT/'fold_tables'/('%s_inner%d.joblib' % (phase, num))
        if fold_file.exists():
            bundles = joblib.load(fold_file)
        else:
            bundles = {}
            for norm in ('PercentileZ', 'RobustZ'):
                for k in (2, 3, 4):
                    partitions = feature.fit_partitions(pp, a, norm, k)
                    d = feature.tables(pp, partitions, np.r_[a, b])
                    bundles[norm, k] = {'partitions': partitions, 'tables': d}
            joblib.dump(bundles, fold_file, compress=3)
        full, prepared_sets = {}, {}
        for c in options():
            key = (c['normalization'], c['K'], c['panel'])
            d = bundles[key[:2]]['tables']
            if key not in full:
                full[key] = prepared(view(d, c['panel']), a, b, y)
            identity = key+(c['top'],)
            if identity not in prepared_sets:
                prepared_sets[identity] = prior.stable.prefix(full[key], c['top'])
            f = fitted(prepared_sets[identity], y[a], c)
            audit.append({'InnerFold': num, 'candidate_id': c['candidate_id'],
                'FitAUC': float(roc_auc_score(y[a], f['train_probability'])),
                'ValidationAUC': float(roc_auc_score(y[b], f['validation_probability'])),
                'FeatureCount': len(f['names']), 'Available': f['available']})
            cv[c['candidate_id']].append({'train': a, 'validation': b, 'fit_score': f['train_probability'],
                'validation_score': f['validation_probability']})
        print('%s inner%d/5:all6 fold-only common partitions and144 candidates complete' % (phase, num), flush=True)
    frame = pd.DataFrame(audit).set_index(['candidate_id', 'InnerFold'])
    grid = []
    for c in options():
        rr = frame.loc[c['candidate_id']]
        mu, sd, fit_mu = rr.ValidationAUC.mean(), rr.ValidationAUC.std(ddof=1), rr.FitAUC.mean()
        available = bool(rr.Available.all())
        grid.append({**c, 'MeanValidationAUC': float(mu), 'SD': float(sd), 'MeanFitAUC': float(fit_mu),
            'MeanFeatureCount': float(rr.FeatureCount.mean()), 'AvailableAllFolds': available,
            'Objective': float(mu-.10*sd-.30*max(fit_mu-mu-.10, 0)) if available else -1.})
    joblib.dump({'phase': phase, 'grid': grid, 'audit': audit, 'cv': cv}, dest, compress=3)


def train():
    protect()
    assert not (OUT/'training_frozen.json').exists() and not (OUT/'test_started.json').exists()
    cohort, y, tr, te = inputs('Fixed118')
    pp = patients(tr)
    for phase in ['outer%d' % k for k in range(1, 6)]+['final']:
        tune(phase, pp, y, te)
    outer, metric, predictions = [], [], []
    for num, (a, b) in enumerate(prior.study.folds('Fixed118', 'outer'), 1):
        phase = joblib.load(OUT/'phases'/('outer%d.joblib' % num))
        partitions_cache, models = {}, {}
        for label, selected in selections(phase['grid']).items():
            c = cfg(selected); key = (c['normalization'], c['K'])
            if key not in partitions_cache:
                partitions = feature.fit_partitions(pp, a, *key)
                partitions_cache[key] = (partitions, feature.tables(pp, partitions, np.r_[a, b]))
            part, d = partitions_cache[key]
            f = fit(d, a, b, y, c)
            models[label] = {'partitions': part, 'tables': d, 'fit': f}
            metric.append({'ModelKey': label, 'Fold': num, **c,
                'FitAUC': float(roc_auc_score(y[a], f['train_probability'])),
                'AUC': float(roc_auc_score(y[b], f['validation_probability'])), 'FeatureCount': len(f['names'])})
            predictions.extend({'ModelKey': label, 'Fold': num, 'AnalysisID': cohort.AnalysisID.iloc[i],
                'Index': int(i), 'Outcome': int(y[i]), 'Probability': float(p)} for i, p in zip(b, f['validation_probability']))
        outer.append({'train': a, 'validation': b, 'models': models})
        print('Outer%d selected models refitted and validated' % num, flush=True)
    phase = joblib.load(OUT/'phases/final.joblib')
    selected = choose(phase['grid']); c = cfg(selected)
    part = feature.fit_partitions(pp, tr, c['normalization'], c['K'])
    d = feature.tables(pp, part, tr)
    f = fit(d, tr, [], y, c)
    oof = np.full(138, np.nan)
    for rr in phase['cv'][c['candidate_id']]:
        oof[rr['validation']] = rr['validation_score']
    assert np.isfinite(oof[tr]).all() and np.isnan(oof[te]).all()
    fp, tp, thresholds = roc_curve(y[tr], oof[tr], drop_intermediate=False)
    cutoff = float(thresholds[int(np.argmax(tp-fp))])
    assert np.isfinite(cutoff)
    f['threshold'], f['training_oof'] = cutoff, oof
    frame = pd.DataFrame(metric); mm = frame[frame.ModelKey == 'Adaptive']
    pred = pd.DataFrame(predictions); pn = pred[pred.ModelKey == 'Adaptive']
    choice = {**c, 'TrainFitAUC': float(roc_auc_score(y[tr], f['train_probability'])),
        'OuterMeanAUC': float(mm.AUC.mean()), 'OuterSD': float(mm.AUC.std(ddof=1)),
        'PooledOOFAUC': float(roc_auc_score(pn.Outcome, pn.Probability)),
        'FinalInnerAUC': selected['MeanValidationAUC'], 'SelectionObjective': selected['Objective'],
        'FeatureCount': len(f['names']), 'Threshold': cutoff}
    result = {'train': tr, 'outer': outer, 'final': {'partitions': part, 'tables': d, 'fit': f}, 'choice': choice}
    joblib.dump(result, OUT/'models/pure_habitat.joblib', compress=3)
    csv('outer_metrics', metric); csv('nested_predictions', predictions); csv('training_choices', [choice])
    dd = view(d, c['panel'])
    csv('selected_features', [{'Feature': name, 'Modality': dd['metadata'][j]['modality'],
        'Panel': dd['metadata'][j]['panel'], 'Coefficient': float(beta),
        'TrainingICC': next(r['ICC'] for r in f['audit'] if r['Feature'] == name)}
        for name, j, beta in zip(f['names'], f['indices'], f['model'].coef_[0])])
    csv('inner_candidate_scores', [{'Phase': phase, **rr} for phase in ['outer%d' % k for k in range(1, 6)]+['final']
        for rr in joblib.load(OUT/'phases'/(phase+'.joblib'))['grid']])
    files = list((OUT/'patient_features').glob('*.joblib'))+list((OUT/'fold_tables').glob('*.joblib'))
    files += list((OUT/'models').glob('*.joblib'))+list((OUT/'phases').glob('*.joblib'))+list(OUT.glob('*.csv'))
    save(OUT/'training_frozen.json', {'passed': True, 'test_scored': False, 'test_models': 1,
        'protocol_sha256': digest(OUT/'protocol.json'), 'files': {str(p.relative_to(OUT)): digest(p) for p in files}})
    print('Pure habitat frozen:'+json.dumps(choice), flush=True)


def frozen():
    protect()
    f = json.loads((OUT/'training_frozen.json').read_text(encoding='utf-8'))
    assert f['passed'] and digest(OUT/'protocol.json') == f['protocol_sha256']
    for p, sha in f['files'].items():
        assert digest(OUT/p) == sha, p
    return f


def test():
    frozen()
    assert not (OUT/'test_started.json').exists()
    assert json.loads((OUT/'training_verification.json').read_text(encoding='utf-8'))['passed']
    cohort, _, tr, te = inputs('Fixed118')
    import run_random120_retained_methods_20261010 as loader
    profiles = loader.load_profiles()
    pp = {}
    for i in te:
        pp[int(i)] = feature.compact_patient(profiles[int(i)], int(i))
        joblib.dump(pp[int(i)], OUT/'patient_features'/('case_%03d.joblib' % i), compress=3)
    result = joblib.load(OUT/'models/pure_habitat.joblib')
    final = result['final']; f = final['fit']
    # Never fit any object after the test stage is opened.
    save(OUT/'test_started.json', {'test_model_calls': 1, 'freeze_sha256': digest(OUT/'training_frozen.json')})
    d = feature.tables(pp, final['partitions'], te)
    joblib.dump(d, OUT/'test_features.joblib', compress=3)
    p = predict(f, d, te)
    yt = cohort.Outcome.to_numpy(int)[te]
    tn, fp, fn, tp = confusion_matrix(yt, p >= f['threshold'], labels=[0, 1]).ravel()
    from verify_ivim_df63_20261020 import draws
    samples = draws(yt, {'Common': p}, 2026102381)['Common']
    lo, hi = np.quantile(samples, [.025, .975])
    metrics = {'N': 36, 'Positive': int(yt.sum()), 'Negative': int((1-yt).sum()),
        'AUC': float(roc_auc_score(yt, p)), 'AUC_CI95Low': float(lo), 'AUC_CI95High': float(hi),
        'Sensitivity': tp/(tp+fn), 'Specificity': tn/(tn+fp), 'Accuracy': (tp+tn)/len(yt),
        'TP': int(tp), 'FN': int(fn), 'TN': int(tn), 'FP': int(fp), 'Threshold': f['threshold']}
    csv('test_metrics', [metrics])
    csv('test_predictions', [{'AnalysisID': cohort.AnalysisID.iloc[i], 'Index': int(i), 'Outcome': int(t),
        'Probability': float(s), 'Prediction': int(s >= f['threshold']), 'ScoreType': 'fitted_probability'} for i, t, s in zip(te, yt, p)])
    save(OUT/'test_complete.json', {'passed': True, 'test_model_calls': 1, 'test_stage_count': 1,
        'post_test_fits': 0, 'post_test_threshold_changes': 0, 'metrics': metrics})
    print('Frozen pure habitat test:'+json.dumps(metrics), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'define', 'train', 'test'))
    args = parser.parse_args()
    with threadpool_limits(1):
        globals()[args.action]()


if __name__ == '__main__':
    main()
