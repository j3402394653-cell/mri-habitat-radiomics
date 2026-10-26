"""Four habitat settings, five families,13 distinct nested5/5 L2 LR models."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import shutil
import time
import joblib
import numpy as np
import pandas as pd
import radiomics
import SimpleITK as sitk
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
import run_k2_habitat_auc_120_20261013 as baseline
import whole_multiseq_features_20261013 as whole

ROOT = baseline.ROOT
OUT = ROOT/'lvi_prediction/results/lvi_k23_five_models_120_20261013'
CLINICAL = ROOT/'lvi_prediction/results/lvi_random120_clinical_extension_20261012'
NORMALIZATIONS = whole.NORMALIZATIONS
KS = (2, 3)
FAMILIES = ('Habitat', 'Whole', 'Clinical', 'HabitatClinical', 'WholeClinical')
old, origin = baseline.old, baseline.origin
save, digest = baseline.save, baseline.digest
KINDS = ('base', 'erode', 'dilate')


def model_specs():
    result = []
    for n in NORMALIZATIONS:
        for k in KS:
            for family in ('Habitat', 'HabitatClinical'):
                result.append({'ModelKey': n+'__K'+str(k)+'__'+family,
                               'normalization': n, 'K': k, 'model_family': family,
                               'image_key': n+'__K'+str(k), 'clinical': family.endswith('Clinical')})
        for family in ('Whole', 'WholeClinical'):
            result.append({'ModelKey': n+'__'+family, 'normalization': n, 'K': 0,
                           'model_family': family, 'image_key': n+'__Whole',
                           'clinical': family.endswith('Clinical')})
    result.append({'ModelKey': 'Clinical', 'normalization': 'NA', 'K': 0,
                   'model_family': 'Clinical', 'image_key': None, 'clinical': True})
    return result


def options():
    result = []
    for spec in model_specs():
        families = ('Descriptor', 'Texture', 'Combined') if spec['model_family'].startswith('Habitat') else ('Texture',)
        budgets = (1, 2, 3, 6, 9, 12, 15, 0) if spec['clinical'] else (3, 6, 9, 12, 15, 0)
        if spec['image_key'] is None:
            families, budgets = ('Clinical',), (0,)
        for family in families:
            for top in budgets:
                for C in (.03, .3, 1.):
                    result.append({**spec, 'analysis': 'FusionFree', 'family': family,
                                   'top': top, 'quota': 0, 'algorithm': 'LR', 'C': C,
                                   'candidate_id': len(result)})
    return result


def config(row):
    return {k: row[k] for k in options()[0]}


def choose(grid, name):
    pool = [r for r in grid if r['ModelKey'] == name]
    valid = [r for r in pool if r['AvailableAllFolds']]
    return max(valid or pool, key=lambda r: (r['Objective'], -r['MeanFeatureCount'], -r['C'], -r['candidate_id']))


def aliases():
    rows = []
    for n in NORMALIZATIONS:
        for k in KS:
            for family in FAMILIES:
                name = ('Clinical' if family == 'Clinical' else n+'__'+family
                        if family.startswith('Whole') else n+'__K'+str(k)+'__'+family)
                rows.append({'Setting': n+'__K'+str(k), 'Normalization': n, 'K': k,
                             'Family': family, 'ModelKey': name})
    return rows


def protect():
    p = json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    for field in ('protected_hashes', 'source_hashes'):
        for name, sha in p[field].items():
            assert digest(ROOT/name) == sha, name
    return len(p['protected_hashes'])


def freeze_protocol():
    if OUT.exists():
        raise FileExistsError('Never overwrite the frozen experiment')
    baseline.frozen()
    cohort, manifest, y, tr, te = origin.inputs()
    protected = dict(json.loads((baseline.OUT/'protocol.json').read_text(encoding='utf-8'))['protected_hashes'])
    for directory in (baseline.OUT, baseline.prior.OUT, CLINICAL):
        protected.update({str(p.relative_to(ROOT)): digest(p) for p in directory.rglob('*') if p.is_file()})
    sources = [Path(__file__).resolve(), Path(whole.__file__).resolve(),
               ROOT/'lvi_prediction/verify_k23_five_models_20261013.py']
    OUT.mkdir(parents=True)
    (OUT/'models').mkdir()
    save(OUT/'protocol.json', {
        'date': '2026-10-13', 'authorization': 'four norm/K settings and five families; human explicitly authorized T2+ADC+IVIM D/f for new whole-tumor control',
        'normalizations': NORMALIZATIONS, 'K_values': KS, 'five_families': FAMILIES,
        'unique_models': 13, 'display_aliases': aliases(), 'candidates': options(), 'candidate_count': len(options()),
        'cohort': 'same random120;84train58positive26negative;36test25positive11negative; frozen54=36+18,36positive18negative',
        'sample_seed': manifest['sample_seed'], 'split_seed': manifest['split_seed'],
        'nested': 'outer5 seed42; inner5 seeds62201..62205; final inner5 seed62212',
        'classifier': 'all families L2 LR liblinear random_state64212 max_iter3000 C=.03/.3/1',
        'clinical': '11 audited preoperative clinical and conventional MRI fields; all retained; train-fold median/P1P99/meanSD only',
        'image_gate': 'same training-only ICC>=.8;complete>=.9;5stratified-bootstrap sign consistency>=.8;absSpearman<.8',
        'image_budgets': 'pure3/6/9/12/15/all; joint adds1/2 image options to11 clinical; chosen inside inner5, no fixed maximum',
        'objective': 'mean inner AUC-.10*SD-.30*max(mean fit-mean validation-.10,0)-.03*abs(mean fit-.75); unchanged from preceding K study',
        'habitat_features': 'verified patient-local K2/K3 caches reused; all classifiers freshly trained',
        'whole_features': 'new372 Original non-shape PyRadiomics features: T2/ADC/D/f each93; same1mm grids and valid base/erode/dilate ROI; matched patient-local normalization; binWidth.25; no K dependence',
        'Dstar_excluded': True, 'same120_84_36_split': True, 'no_test_driven_selection': True,
        'initial_test': 'training verification then one36 batch per13models; freeze byte-identical snapshot then one new54 batch per13models',
        'separate_postfreeze36_or18_model_predict_calls': 0, 'reuse_initial36_probabilities_in54': False,
        'shared_controls': 'Whole/WholeClinical shared across K within normalization; one Clinical across all4settings',
        'historical_test_reuse': True, 'new_external_validation': False, 'formal_model_use': False,
        'automatic_adoption': False, 'previous_adoption_and_pointer_unchanged': True,
        'protected_hashes': protected, 'source_hashes': {str(p.relative_to(ROOT)): digest(p) for p in sources},
        'sources': ['https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html',
                    'https://pyradiomics.readthedocs.io/en/v3.0/customization.html']})
    save(OUT/'split_manifest.json', manifest)
    pd.DataFrame(aliases()).to_csv(OUT/'display_aliases.csv', index=False, encoding='utf-8-sig')
    return cohort, manifest, y, tr, te


def clinical_data(cohort, indices):
    table = pd.read_csv(CLINICAL/'clinical_extension_138.csv').set_index('AnalysisID')
    audit = json.loads((CLINICAL/'clinical_extension_audit.json').read_text(encoding='utf-8'))
    columns = audit['clinical_columns'] if 'clinical_columns' in audit else [s for s in table if s != 'Outcome']
    assert len(columns) == 11
    ids = cohort.AnalysisID.iloc[indices]
    assert audit['passed'] and audit['exact_ID_and_label_mapping']
    assert digest(CLINICAL/'clinical_extension_138.csv') == audit['parsed_data_sha256']
    assert table.index.is_unique and set(table.index) == set(cohort.AnalysisID)
    return {'indices': np.asarray(indices, int), 'columns': columns,
            'base': table.loc[ids, columns].to_numpy(float)}


def data_stage(cohort, indices, stage):
    data = {}
    for k, directory in ((2, baseline.OUT), (3, baseline.prior.OUT)):
        loaded = joblib.load(directory/(('frozen54' if stage == 'omitted18' else stage)+'_features.joblib'))
        for n in NORMALIZATIONS:
            d = loaded[n][k]
            if stage == 'omitted18':
                positions = old.common.rows(d, indices)
                d = {**d, 'indices': np.asarray(indices, int), **{kind: d[kind][positions] for kind in KINDS}}
            np.testing.assert_array_equal(d['indices'], indices)
            data[n+'__K'+str(k)] = d
    profiles = old.common.old.retained.load_profiles()
    radiomics.logger.setLevel(40)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    results, start = {}, time.time()
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(whole.extract_one, (int(i), cohort.AnalysisID.iloc[i], profiles[int(i)])): int(i) for i in indices}
        for done, f in enumerate(as_completed(futures), 1):
            results[futures[f]] = f.result()
            if done % 4 == 0 or done == len(indices):
                print('Whole %s %d/%d %.0fs' % (stage, done, len(indices), time.time()-start), flush=True)
    meta = results[int(indices[0])]['metadata']
    assert len(meta) == len({m['name'] for m in meta}) == 372
    assert all(results[int(i)]['metadata'] == meta for i in indices)
    for n in NORMALIZATIONS:
        data[n+'__Whole'] = {'metadata': meta, 'indices': np.asarray(indices, int),
            **{kind: np.asarray([results[int(i)]['values'][n][kind] for i in indices], float) for kind in KINDS}}
    data['clinical'] = clinical_data(cohort, indices)
    for name in ('normalization_audit', 'texture_audit'):
        pd.DataFrame([r for i in indices for r in results[int(i)][name]]).to_csv(
            OUT/(stage+'_whole_'+name+'.csv'), index=False, encoding='utf-8-sig')
    joblib.dump(data, OUT/(stage+'_features.joblib'), compress=3)
    return data


def transform_fit(x, xb):
    median = np.nanmedian(np.where(np.isfinite(x), x, np.nan), axis=0)
    median = np.where(np.isfinite(median), median, 0.)
    x = np.where(np.isfinite(x), x, median)
    low, high = np.quantile(x, [.01, .99], axis=0)
    x = np.clip(x, low, high)
    mean, scale = x.mean(0), x.std(0)
    scale = np.where(scale > 1e-12, scale, 1.)
    return {'median': median, 'low': low, 'high': high, 'mean': mean, 'scale': scale,
            'xa': (x-mean)/scale, 'xb': (np.clip(np.where(np.isfinite(xb), xb, median), low, high)-mean)/scale}


def prepare(data, a, b, y, c, ranked=None, full=None):
    if c['image_key'] is not None:
        image = data[c['image_key']]
        p = baseline.core.budget_prefix(full, c['top']) if full is not None else old.prepare(image, a, b, c, y, ranked)
    else:
        p = {'indices': [], 'names': [], 'audit': [], 'available': True,
             **transform_fit(np.empty((len(a), 0)), np.empty((len(b), 0)))}
    if not c['clinical']:
        return p
    d = data['clinical']
    xa = d['base'][old.common.rows(d, a)]
    xb = d['base'][old.common.rows(d, b)]
    cp = transform_fit(xa, xb)
    return {**p, 'names': p['names']+d['columns'],
            **{s: np.concatenate([p[s], cp[s]]) for s in ('median', 'low', 'high', 'mean', 'scale')},
            'xa': np.column_stack([p['xa'], cp['xa']]), 'xb': np.column_stack([p['xb'], cp['xb']])}


def fitted(p, target, c):
    estimator = LogisticRegression(C=c['C'], penalty='l2', solver='liblinear', max_iter=3000, random_state=64212)
    if p['xa'].shape[1]:
        estimator.fit(p['xa'], target)
        pa = estimator.predict_proba(p['xa'])[:, 1]
        pb = estimator.predict_proba(p['xb'])[:, 1] if len(p['xb']) else np.empty(0)
    else:
        estimator = None
        pa, pb = np.full(len(target), target.mean()), np.full(len(p['xb']), target.mean())
    return {**{s: p[s] for s in ('indices', 'names', 'audit', 'available', 'median', 'low', 'high', 'mean', 'scale')},
            'config': c, 'model': estimator, 'prevalence': float(target.mean()),
            'train_probability': pa, 'evaluation_probability': pb}


def fit(data, a, b, y, c):
    return fitted(prepare(data, a, b, y, c), y[a], c)


def tune(data, folds, y, label):
    opts = options()
    values, cv, audit = {c['candidate_id']: [] for c in opts}, {c['candidate_id']: [] for c in opts}, []
    for fold, (a, b) in enumerate(folds, 1):
        ranks = {s: old.ranking(d, a, y) for s, d in data.items() if s != 'clinical'}
        fullsets, prepared = {}, {}
        for c in opts:
            identity = (c['image_key'], c['family'], c['top'], c['clinical'])
            if identity not in prepared:
                full = None
                if c['image_key'] is not None:
                    identity_full = (c['image_key'], c['family'])
                    if identity_full not in fullsets:
                        fullsets[identity_full] = old.prepare(data[c['image_key']], a, b, {**c, 'top': 0}, y, ranks[c['image_key']])
                    full = fullsets[identity_full]
                prepared[identity] = prepare(data, a, b, y, c, full=full)
            f = fitted(prepared[identity], y[a], c)
            v = (float(roc_auc_score(y[b], f['evaluation_probability'])),
                 float(roc_auc_score(y[a], f['train_probability'])), len(f['names']), f['available'])
            values[c['candidate_id']].append(v)
            cv[c['candidate_id']].append((b, f['evaluation_probability']))
            audit.append({'Phase': label, 'InnerFold': fold, 'candidate_id': c['candidate_id'],
                          'ValidationAUC': v[0], 'FitAUC': v[1], 'FeatureCount': v[2], 'Available': v[3]})
        print(label+' inner'+str(fold)+'/5 complete', flush=True)
    grid = []
    for c in opts:
        v = np.asarray(values[c['candidate_id']])
        mean, sd, train, count = float(v[:, 0].mean()), float(v[:, 0].std(ddof=1)), float(v[:, 1].mean()), float(v[:, 2].mean())
        valid = bool(v[:, 3].all())
        objective = mean-.10*sd-.30*max(train-mean-.10, 0)-.03*abs(train-.75)
        grid.append({**c, 'MeanValidationAUC': mean, 'SD': sd, 'MeanFitAUC': train,
                     'MeanFeatureCount': count, 'AvailableAllFolds': valid, 'Objective': objective if valid else -1.})
    return grid, cv, audit


def train():
    start = time.time()
    cohort, manifest, y, tr, te = freeze_protocol()
    data = data_stage(cohort, tr, 'training')
    predictions, metrics, grids, inner_audit, frequency = [], [], [], [], []
    for fold, (a, b) in enumerate(old.common.old.previous.folds(tr, y, 42, 5), 1):
        grid, _, audit = tune(data, old.common.old.previous.folds(a, y, 62200+fold, 5), y, 'outer'+str(fold))
        models = {}
        for s in model_specs():
            name = s['ModelKey']
            c = config(choose(grid, name))
            f = fit(data, a, b, y, c)
            models[name] = f
            metrics.append({'ModelKey': name, 'Fold': fold, **c, 'ImageFeatureCount': len(f['indices']),
                            'FeatureCount': len(f['names']), 'AUC': float(roc_auc_score(y[b], f['evaluation_probability']))})
            predictions.extend({'ModelKey': name, 'Fold': fold, 'AnalysisID': cohort.AnalysisID.iloc[i],
                                'Outcome': int(y[i]), 'Probability': float(p)} for i, p in zip(b, f['evaluation_probability']))
            frequency.extend({'ModelKey': name, 'Fold': fold, 'Feature': n} for n in f['names'])
        joblib.dump({'train': a, 'evaluation': b, 'grid': grid, 'models': models}, OUT/('models/outer_fold_'+str(fold)+'.joblib'), compress=3)
        grids.extend({'Phase': 'outer'+str(fold), **r} for r in grid)
        inner_audit.extend(audit)
        print('outer'+str(fold)+'/5 complete', flush=True)
    grid, cv, audit = tune(data, old.common.old.previous.folds(tr, y, 62212, 5), y, 'final')
    inner_audit.extend(audit)
    final = {'models': {}, 'train': tr, 'formal_model_use': False}
    choices, features, trainpred, oofpred, gates = [], [], [], [], []
    nested, foldmetrics = pd.DataFrame(predictions), pd.DataFrame(metrics)
    for s in model_specs():
        name = s['ModelKey']
        row = choose(grid, name)
        c = config(row)
        f = fit(data, tr, [], y, c)
        oof = np.full(138, np.nan)
        for ix, p in cv[c['candidate_id']]:
            oof[ix] = p
        f['threshold'] = old.common.threshold(y[tr], oof[tr])
        final['models'][name] = f
        ff = foldmetrics[foldmetrics.ModelKey == name].AUC
        pp = nested[nested.ModelKey == name].set_index('AnalysisID').loc[cohort.AnalysisID.iloc[tr]]
        choices.append({**c, 'ImageFeatureCount': len(f['indices']), 'FeatureCount': len(f['names']),
                        'TrainFitAUC': float(roc_auc_score(y[tr], f['train_probability'])),
                        'OuterMeanAUC': float(ff.mean()), 'OuterSD': float(ff.std(ddof=1)),
                        'PooledOOFAUC': float(roc_auc_score(y[tr], pp.Probability)),
                        'FinalSelectionCVMeanAUC': row['MeanValidationAUC'], 'SelectionObjective': row['Objective'],
                        'Threshold': f['threshold'], 'Available': f['available']})
        meta = data[c['image_key']]['metadata'] if c['image_key'] else []
        for j, (n, beta) in enumerate(zip(f['names'], f['model'].coef_[0] if f['model'] is not None else [])):
            m = meta[f['indices'][j]] if j < len(f['indices']) else {'modality': 'Clinical', 'block': 'Clinical'}
            features.append({'ModelKey': name, 'Feature': n, 'Modality': m['modality'], 'Block': m['block'],
                             'LRStandardizedCoefficient': float(beta)})
        trainpred.extend({'ModelKey': name, 'AnalysisID': cohort.AnalysisID.iloc[i], 'Outcome': int(y[i]),
                          'Probability': float(p)} for i, p in zip(tr, f['train_probability']))
        oofpred.extend({'ModelKey': name, 'AnalysisID': cohort.AnalysisID.iloc[i], 'Outcome': int(y[i]),
                        'Probability': float(oof[i])} for i in tr)
    for imagekey, d in data.items():
        if imagekey != 'clinical':
            _, _, audit = old.ranking(d, tr, y)
            gates.extend({'ImageKey': imagekey, **r} for r in audit)
    joblib.dump(final, OUT/'models/five_models.joblib', compress=3)
    for name, rows in [('nested_predictions', nested), ('outer_fold_metrics', foldmetrics),
                       ('inner_candidate_scores', grids), ('inner_fold_audit', inner_audit),
                       ('final_training_grid', grid), ('final_training_choices', choices), ('selected_features', features),
                       ('train_fit_predictions', trainpred), ('final_selection_oof_predictions', oofpred),
                       ('feature_gate', gates), ('outer_feature_selections', frequency)]:
        pd.DataFrame(rows).to_csv(OUT/(name+'.csv'), index=False, encoding='utf-8-sig')
    save(OUT/'training_frozen.json', {'passed': True, 'model_sha256': digest(OUT/'models/five_models.joblib'),
         'protocol_sha256': digest(OUT/'protocol.json'), 'feature_sha256': digest(OUT/'training_features.joblib'),
         'training_csv_hashes': {p.name: digest(p) for p in OUT.glob('*.csv')},
         'test_scored': False, 'protected_count': protect(), 'elapsed_seconds': time.time()-start})
    print(pd.DataFrame(choices)[['ModelKey', 'TrainFitAUC', 'OuterMeanAUC', 'PooledOOFAUC', 'FeatureCount']].to_string(index=False), flush=True)


def frozen():
    p = json.loads((OUT/'training_frozen.json').read_text(encoding='utf-8'))
    protect()
    for name, field in [('models/five_models.joblib', 'model_sha256'), ('protocol.json', 'protocol_sha256'), ('training_features.joblib', 'feature_sha256')]:
        assert digest(OUT/name) == p[field]
    for name, sha in p['training_csv_hashes'].items():
        assert digest(OUT/name) == sha
    return p


def raw_input(f, data, indices):
    c = f['config']
    x = data[c['image_key']]['base'][old.common.rows(data[c['image_key']], indices)][:, f['indices']] if c['image_key'] else np.empty((len(indices), 0))
    if c['clinical']:
        x = np.column_stack([x, data['clinical']['base'][old.common.rows(data['clinical'], indices)]])
    assert x.shape[1] == len(f['names'])
    return x


def predict(f, data, indices):
    x = raw_input(f, data, indices)
    x = (np.clip(np.where(np.isfinite(x), x, f['median']), f['low'], f['high'])-f['mean'])/f['scale']
    return f['model'].predict_proba(x)[:, 1] if f['model'] is not None else np.full(len(indices), f['prevalence'])


def evaluation(final, data, cohort, indices, stage, groups):
    scores = {name: predict(f, data, indices) for name, f in final['models'].items()}
    y = cohort.Outcome.to_numpy(int)[indices]
    rows, metrics = [], []
    for name, p in scores.items():
        threshold = final['models'][name]['threshold']
        rows.extend({'ModelKey': name, 'AnalysisID': cohort.AnalysisID.iloc[i],
                     'Group': 'PreviousTest36' if j < 36 else 'PreviouslyOmitted18',
                     'Outcome': int(v), 'Probability': float(pp), 'Threshold': threshold,
                     'PredictedPositive': int(pp >= threshold)} for j, (i, v, pp) in enumerate(zip(indices, y, p)))
    for group, pos in groups.items():
        yy, ps = y[pos], {name: p[pos] for name, p in scores.items()}
        draws = old.common.old.retained.bootstrap(yy, ps)
        for name, p in ps.items():
            lo, hi = np.quantile(draws[name], [.025, .975])
            metrics.append({'Group': group, 'ModelKey': name, 'N': len(yy), 'Positive': int(yy.sum()),
                            'Negative': int((yy == 0).sum()), **old.common.old.metrics(yy, p, final['models'][name]['threshold']),
                            'AUC_CI95Low': float(lo), 'AUC_CI95High': float(hi)})
    for name, r in [('predictions', rows), ('metrics', metrics)]:
        pd.DataFrame(r).to_csv(OUT/(stage+'_'+name+'.csv'), index=False, encoding='utf-8-sig')
    return len(scores)


def test120():
    if (OUT/'test120_started.json').exists():
        raise FileExistsError('Initial36 already scored')
    fr = frozen()
    assert json.loads((OUT/'training_verification.json').read_text())['passed']
    cohort, manifest, y, tr, te = origin.inputs()
    data = data_stage(cohort, te, 'test120')
    final = joblib.load(OUT/'models/five_models.joblib')
    save(OUT/'test120_started.json', {'n': 36, 'distinct_models': 13, 'joint_count': 1, 'model_sha256': fr['model_sha256']})
    calls = evaluation(final, data, cohort, te, 'test120', {'PreviousTest36': np.arange(36)})
    save(OUT/'test120_complete.json', {'passed': True, 'n': 36, 'distinct_predict_calls': calls,
                                      'model_sha256': fr['model_sha256'], 'protected_count': protect()})


def freeze54():
    if (OUT/'frozen54_started.json').exists():
        raise FileExistsError('Frozen54 already scored')
    fr = frozen()
    assert json.loads((OUT/'test120_complete.json').read_text())['passed']
    cohort, manifest, y, tr, te = origin.inputs()
    indices = np.r_[te, np.asarray(manifest['omitted_indices'], int)]
    assert not set(tr) & set(indices)
    snapshot = OUT/'models/post120_frozen_models.joblib'
    shutil.copyfile(OUT/'models/five_models.joblib', snapshot)
    assert digest(snapshot) == fr['model_sha256']
    final = joblib.load(snapshot)
    save(OUT/'post120_freeze_record.json', {'model_sha256': digest(snapshot), 'same_as_training_snapshot': True,
        'no_refit_after36': True, 'thresholds': {name: float(f['threshold']) for name, f in final['models'].items()},
        'training_ids': cohort.AnalysisID.iloc[tr].tolist(), 'evaluation_ids': cohort.AnalysisID.iloc[indices].tolist()})
    # Reuse the newly extracted36 per-patient whole features, not model probabilities.
    # Other18 are extracted independently; assemble one54 feature batch before prediction.
    initial = joblib.load(OUT/'test120_features.joblib')
    omitted = data_stage(cohort, indices[36:], 'omitted18')
    data = {}
    for name in initial:
        d = initial[name]
        data[name] = {**d, 'indices': indices,
                      **{kind: np.concatenate([d[kind], omitted[name][kind]]) for kind in (('base',) if name == 'clinical' else KINDS)}}
    joblib.dump(data, OUT/'frozen54_features.joblib', compress=3)
    save(OUT/'frozen54_started.json', {'n': 54, 'joint_count': 1, 'reuse_initial36_probabilities': False,
                                     'separate36_prediction_calls': 0, 'separate18_prediction_calls': 0})
    calls = evaluation(final, data, cohort, indices, 'frozen54', {
        'All54': np.arange(54), 'PreviousTest36': np.arange(36), 'PreviouslyOmitted18': np.arange(36, 54)})
    save(OUT/'frozen54_complete.json', {'passed': True, 'n': 54, 'distinct_predict_calls': calls,
                                       'models_refitted': 0, 'protected_count': protect()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['train', 'test120', 'freeze54'], required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        {'train': train, 'test120': test120, 'freeze54': freeze54}[args.phase]()
