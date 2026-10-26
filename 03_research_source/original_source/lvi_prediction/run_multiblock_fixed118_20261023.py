# -*- coding: utf-8 -*-
"""Low-dimensional sequence-balanced modelling and honest cross-fitted late fusion."""
import os
os.environ.update({'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
import multimodal_fixed118_study_20261023 as old
import run_multimodal_fixed118_20261023 as previous
from run_regional_summary77_two_tests_20261014 import bootstrap

ROOT = old.ROOT
OUT = ROOT / 'lvi_prediction/results/lvi_multiblock_fixed118_20261023'
FAMILIES = old.FAMILIES
MODALITIES = ('T2', 'ADC', 'IVIM')
BASIC = ('Clinical_PreopT', 'Clinical_PreopN', 'Clinical_EMVI', 'Clinical_MRF', 'Clinical_logCEA')
save, digest = old.save, old.digest
TABLES = None
RANK_CACHE, BLOCK_CACHE = {}, {}


def csv(name, rows):
    pd.DataFrame(rows).to_csv(OUT / (name + '.csv'), index=False, encoding='utf-8-sig')


def read(name):
    return pd.read_csv(OUT / (name + '.csv'), float_precision='round_trip', keep_default_na=False, na_values=[''])


@lru_cache(None)
def options():
    rows = []
    specs = [s for s in old.feature.specifications() if s['K'] in (2, 3)]
    for family in FAMILIES:
        images = specs if family.startswith('Habitat') else [
            {'ImageKey': n + '__Whole', 'Normalization': n, 'K': 0, 'ComboID': 'NA'} for n in old.feature.NORMS]
        if family == 'Clinical':
            images = [{'ImageKey': None, 'Normalization': 'NA', 'K': 0, 'ComboID': 'NA'}]
        scopes = ('Basic5', 'Extended11') if family in ('Clinical', 'HabitatClinical', 'WholeClinical') else ('None',)
        for spec in images:
            methods = (('Clinical', 0),) if family == 'Clinical' else (('Balanced', 3), ('Balanced', 5), ('PCA', 1), ('PCA', 2), ('Late', 3))
            for method, size in methods:
                for q in ((0.,) if family == 'Clinical' else (.75, .8)):
                    for scope in scopes:
                        for C in (.003, .03, .3):
                            rows.append({'family': family, 'image_key': spec['ImageKey'], 'normalization': spec['Normalization'],
                                'K': spec['K'], 'ComboID': spec['ComboID'], 'architecture': method, 'size': size,
                                'quality': q, 'clinical_scope': scope, 'C': C, 'candidate_id': len(rows)})
    assert len(rows) == 1266
    return rows


def config(row):
    return {k: row[k] for k in options()[0]}


def input_data(stage):
    # Use only the previous82-row feature table during development.
    path = old.OUT / ('training_features.joblib' if stage == 'train' else 'all118_features.joblib')
    data = joblib.load(path)
    indices = old.inputs('Fixed118')[2 if stage == 'train' else 3]
    keys = {c['image_key'] for c in options()} - {None}
    result = {}
    for key in list(sorted(keys)) + ['Clinical']:
        d = data[key]
        rr = previous.rows(d, indices)
        kinds = ('base',) if key == 'Clinical' else previous.stable.views.KINDS
        result[key] = {'indices': indices.copy(), 'metadata': d['metadata'], **{kind: d[kind][rr].copy() for kind in kinds}}
    return result


def protect():
    protocol = json.loads((OUT / 'protocol.json').read_text(encoding='utf-8'))
    for field in ('protected_hashes', 'source_hashes'):
        for name, sha in protocol[field].items():
            assert digest(ROOT / name) == sha, name
    definition = json.loads((OUT / 'definition_frozen.json').read_text(encoding='utf-8'))
    assert definition['protocol_sha256'] == digest(OUT / 'protocol.json')
    return len(protocol['protected_hashes'])


def define():
    assert not OUT.exists(), 'Do not replace an experiment or redraw cases'
    previous.frozen()
    p = json.loads((old.OUT / 'protocol.json').read_text(encoding='utf-8'))
    delivery = json.loads((old.OUT / 'delivery_manifest.json').read_text(encoding='utf-8'))
    assert delivery['passed'] and delivery['visual_verification_passed']
    protected = {**p['protected_hashes'], **p['source_hashes']}
    for name, sha in delivery['outputs'].items():
        path = old.OUT / name
        assert digest(path) == sha, name
        protected[str(path.relative_to(ROOT))] = sha
    protected[str((old.OUT / 'delivery_manifest.json').relative_to(ROOT))] = digest(old.OUT / 'delivery_manifest.json')
    files = [Path(__file__).resolve(), ROOT / 'lvi_prediction/verify_multiblock_fixed118_20261023.py',
             ROOT / 'lvi_prediction/report_multiblock_fixed118_20261023.py']
    OUT.mkdir(parents=True)
    for folder in ('models', 'phases', 'Fixed118'):
        (OUT / folder).mkdir()
    for name in ('Train82_cases.csv', 'Test36_cases.csv', 'Sample118_cases.csv', 'nested5_folds.json'):
        (OUT / 'Fixed118' / name).write_bytes((old.OUT / 'Fixed118' / name).read_bytes())
    (OUT / 'sample_manifest.json').write_bytes((old.OUT / 'sample_manifest.json').read_bytes())
    save(OUT / 'protocol.json', {
        'date': '2026-10-23', 'authorization': 'Human requested further improvement after multimodal fixed118 failed.',
        'cohort_and_folds': 'Original118,exact82/36 andsame nestedouter5/inner5 as previous;no AUC-based case selection.',
        'families': FAMILIES, 'T2_ADC_K': 3, 'IVIM_K_candidates': [2, 3],
        'IVIM_inputs': ['C08', 'C18', 'C38'], 'normalizations': old.feature.NORMS,
        'quality': 'train-fold completeness>=.9,nonconstant,ICC .75/.8 chosen insideinner;within-block Spearman<.95.',
        'Balanced': 'Each sequence top3/top5 Association-ranked features,not pooled global ranking;all3 blocks must be available.',
        'PCA': 'Within each sequence,allquality-eligible correlation-pruned features,unsupervised PCA1/2 components;allfit on current training subset only.',
        'Late': 'Each sequence Association top3 + ridgeLR C.03;stratified3-fold cross-fitting inside EACH current fitting subset;metaLR trained on OOF base logits;current validation/test never used in base ormeta fitting.',
        'Late_prediction': 'Refit each base on current fulltrain;apply OOF-fitted meta input clipping/scaling andmeta classifier.',
        'clinical': 'Basic5 preopT,N,EMVI,MRF,logCEA versus auditedExtended11,chosen insideinner;no pathology/postoperative predictors.',
        'regularization': 'L2liblinearLR C=.003,.03,.3;exclude previous weakly regularizedC3.',
        'selection_objective': 'unchanged meanInnerAUC-.10*SD-.30*max(meanFitAUC-meanInnerAUC-.10,0);available first,then simpler effective image dimensions.',
        'preprocessing': 'Current-fit-subset medianfill,P2.5/P97.5 clip,median/IQR scale;each sequence processed separately;PCA/late inputs also trained only within subset.',
        'test': 'Freeze andverify all5 families;one final36-patient scoring stage;no architecture alternatives separately tested.',
        'threshold': 'Full82 selected configuration complete-pipeline inner5 OOFYouden;fixed before test.',
        'uncertainty': '5000stratifiedpatientbootstrap;paired old/new test scores on same36 patients.',
        'Dstar': False, 'sequence_registration': False, 'test_prior_exposure': True,
        'historical_candidate_space_exposure': True, 'formal_model_use': False, 'automatic_adoption': False,
        'candidate_count': len(options()), 'candidate_counts': {f: sum(c['family'] == f for c in options()) for f in FAMILIES},
        'source_hashes': {str(path.relative_to(ROOT)): digest(path) for path in files}, 'protected_hashes': protected})
    csv('candidate_definitions', options())
    save(OUT / 'definition_frozen.json', {'passed': True, 'protocol_sha256': digest(OUT / 'protocol.json'),
        'models_fitted': 0, 'test_scored': False})
    joblib.dump(input_data('train'), OUT / 'training_features.joblib', compress=3)
    print('Defined multiblock118:1266 configs;exact82/36;3 architectures;all3 image sequences', flush=True)


def row(d, ix):
    return previous.rows(d, np.asarray(ix, int))


def scaler(x):
    return {k: v for k, v in previous.transform(x, np.empty((0, x.shape[1]))).items() if k not in ('xa', 'xb')}


def scale(x, s):
    return (np.clip(np.where(np.isfinite(x), x, s['median']), s['low'], s['high']) - s['center']) / s['scale']


def rank_block(d, a, y, modality, quality):
    rr = row(d, a)
    cols = [j for j, m in enumerate(d['metadata']) if m['modality'] == modality]
    small = {'indices': np.asarray(a), 'metadata': [d['metadata'][j] for j in cols],
             **{kind: d[kind][rr][:, cols] for kind in previous.stable.views.KINDS}}
    h = hashlib.blake2b(digest_size=20)
    h.update(np.asarray(a, dtype=np.int64).tobytes()); h.update(np.asarray(y[a], dtype=np.int64).tobytes())
    h.update(json.dumps([(j, d['metadata'][j]['name']) for j in cols]).encode('utf-8'))
    for kind in previous.stable.views.KINDS:
        h.update(np.ascontiguousarray(small[kind]).tobytes())
    signature = h.hexdigest()
    # Original audit computes ICC/scores without using evaluation observations.
    if signature not in RANK_CACHE:
        _, ranks, audit = previous.stable.ranking(small, a, y)
        RANK_CACHE[signature] = (ranks, audit)
    ranks, audit = RANK_CACHE[signature]
    valid = [j for j, r in enumerate(audit) if r['CompleteFraction'] >= .9 and r['ICC'] is not None and r['ICC'] >= quality]
    valid = [j for j in valid if np.nanstd(np.where(np.isfinite(small['base'][:, j]), small['base'][:, j], np.nan)) > 1e-12]
    return cols, small, ranks, audit, valid, signature


def block_fit(d, a, y, modality, quality, architecture, size):
    cols, small, ranks, audit, eligible, signature = rank_block(d, a, y, modality, quality)
    cache_key = (signature, quality, architecture, size)
    if cache_key in BLOCK_CACHE:
        return BLOCK_CACHE[cache_key]
    if architecture == 'PCA':
        order = sorted(eligible, key=lambda j: (-audit[j]['ICC'], small['metadata'][j]['name']))
    else:
        order = sorted(eligible, key=lambda j: (-audit[j]['AssociationScore'], -audit[j]['ICC'], small['metadata'][j]['name']))
    selected = []
    for j in order:
        if all(abs(float(np.corrcoef(ranks[:, j], ranks[:, k])[0, 1])) < .95 for k in selected):
            selected.append(j)
        if architecture != 'PCA' and len(selected) >= size:
            break
    indices = [cols[j] for j in selected]
    b = {'modality': modality, 'indices': indices, 'names': [d['metadata'][j]['name'] for j in indices],
         'audit': audit, 'eligible_names': [small['metadata'][j]['name'] for j in eligible],
         'available': bool(indices), 'architecture': architecture, 'projector': None, 'base_model': None}
    if not indices:
        b['dimensions'] = 0
        BLOCK_CACHE[cache_key] = b
        return b
    raw = d['base'][row(d, a)][:, indices]
    b['scaler'] = scaler(raw)
    x = scale(raw, b['scaler'])
    if architecture == 'PCA':
        b['projector'] = PCA(n_components=min(size, x.shape[1], len(x) - 1), svd_solver='full').fit(x)
        b['dimensions'] = b['projector'].n_components_
    elif architecture == 'Late':
        b['base_model'] = LogisticRegression(C=.03, penalty='l2', solver='liblinear', max_iter=3000, random_state=64212).fit(x, y[a])
        b['dimensions'] = 1
    else:
        b['dimensions'] = len(indices)
    BLOCK_CACHE[cache_key] = b
    return b


def block_values(b, d, ix):
    if not b['available']:
        return np.empty((len(ix), 0))
    if not len(ix):
        return np.empty((0, b['dimensions']))
    x = scale(d['base'][row(d, ix)][:, b['indices']], b['scaler'])
    if b['projector'] is not None:
        return b['projector'].transform(x)
    if b['base_model'] is not None:
        return b['base_model'].decision_function(x)[:, None]
    return x


def image_prepare(data, a, b, y, c):
    if c['image_key'] is None:
        return {'blocks': [], 'available': True, 'xa': np.empty((len(a), 0)),
                'xa_predict': np.empty((len(a), 0)), 'xb': np.empty((len(b), 0)), 'crossfit': [], 'meta_scaler': None}
    d = data[c['image_key']]
    blocks = [block_fit(d, a, y, m, c['quality'], c['architecture'], c['size']) for m in MODALITIES]
    available = all(s['available'] for s in blocks)
    pa = np.column_stack([block_values(s, d, a) for s in blocks])
    pb = np.column_stack([block_values(s, d, b) for s in blocks])
    crossfit, ms = [], None
    train = pa
    if c['architecture'] == 'Late':
        # Cross-fit the entire base pipeline,including feature choice and preprocessing.
        oo = np.full((len(a), 3), np.nan)
        local = StratifiedKFold(3, shuffle=True, random_state=70923 + int(np.sum(a)))
        for ai, bi in local.split(a, y[a]):
            aa, bb = np.asarray(a)[ai], np.asarray(a)[bi]
            base = [block_fit(d, aa, y, m, c['quality'], 'Late', 3) for m in MODALITIES]
            available = available and all(s['available'] for s in base)
            values = np.column_stack([block_values(s, d, bb) if s['available'] else np.zeros((len(bb), 1)) for s in base])
            oo[bi] = values
            crossfit.append({'train': aa, 'validation': bb, 'blocks': base, 'logits': values})
        assert np.isfinite(oo).all()
        # If a full-fit block is unavailable,still retain3 placeholder columns andflag unavailable.
        pa = np.column_stack([block_values(s, d, a) if s['available'] else np.zeros((len(a), 1)) for s in blocks])
        pb = np.column_stack([block_values(s, d, b) if s['available'] else np.zeros((len(b), 1)) for s in blocks])
        ms = scaler(oo)
        train, pa, pb = scale(oo, ms), scale(pa, ms), scale(pb, ms)
    return {'blocks': blocks, 'available': bool(available), 'xa': train, 'xa_predict': pa, 'xb': pb,
            'crossfit': crossfit, 'meta_scaler': ms}


def clinical_prepare(data, a, b, scope):
    if scope == 'None':
        return {'indices': [], 'names': [], 'scaler': None, 'xa': np.empty((len(a), 0)), 'xb': np.empty((len(b), 0))}
    d = data['Clinical']
    names = [m['name'] for m in d['metadata']]
    cols = list(range(11)) if scope == 'Extended11' else [names.index(n) for n in BASIC]
    xa, xb = d['base'][row(d, a)][:, cols], d['base'][row(d, b)][:, cols]
    s = scaler(xa)
    return {'indices': cols, 'names': [names[j] for j in cols], 'scaler': s, 'xa': scale(xa, s), 'xb': scale(xb, s)}


def fit_prepared(ip, cp, a, y, c):
    train = np.column_stack([ip['xa'], cp['xa']])
    fitted = np.column_stack([ip['xa_predict'], cp['xa']])
    valid = np.column_stack([ip['xb'], cp['xb']])
    available = bool(ip['available'] and train.shape[1])
    model = LogisticRegression(C=c['C'], penalty='l2', solver='liblinear', max_iter=3000, random_state=64212).fit(train, y[a]) if available else None
    pa = model.predict_proba(fitted)[:, 1] if model is not None else np.full(len(a), y[a].mean())
    pb = model.predict_proba(valid)[:, 1] if model is not None and len(valid) else np.full(len(valid), y[a].mean())
    return {'config': c, 'model': model, 'blocks': ip['blocks'], 'meta_scaler': ip['meta_scaler'],
        'crossfit': ip['crossfit'], 'clinical': {k: cp[k] for k in ('indices', 'names', 'scaler')},
        'prevalence': float(y[a].mean()), 'available': bool(available and np.ptp(pa) > 1e-12),
        'train_probability': pa, 'evaluation_probability': pb,
        'image_dimensions': ip['xa'].shape[1], 'clinical_dimensions': len(cp['indices']),
        'image_raw_features': sum(len(s['indices']) for s in ip['blocks'])}


def fit(data, a, b, y, c):
    return fit_prepared(image_prepare(data, a, b, y, c), clinical_prepare(data, a, b, c['clinical_scope']), a, y, c)


def design(f, data, ix):
    parts = []
    if f['blocks']:
        d = data[f['config']['image_key']]
        image = np.column_stack([block_values(s, d, ix) for s in f['blocks']])
        if f['meta_scaler'] is not None:
            image = scale(image, f['meta_scaler'])
        parts.append(image)
    cp = f['clinical']
    if cp['indices']:
        d = data['Clinical']
        parts.append(scale(d['base'][row(d, ix)][:, cp['indices']], cp['scaler']))
    return np.column_stack(parts)


def predict(f, data, ix):
    return f['model'].predict_proba(design(f, data, ix))[:, 1] if f['model'] is not None else np.full(len(ix), f['prevalence'])


def choose(grid, family):
    return max([r for r in grid if r['family'] == family], key=lambda r: (bool(r['AvailableAllFolds']),
        r['Objective'], -r['MeanImageDimensions'], -r['MeanClinicalDimensions'], -r['K'], -r['C'], -r['candidate_id']))


def initialize():
    global TABLES
    threadpool_limits(1)
    TABLES = joblib.load(OUT / 'training_features.joblib')


def tune(phase):
    destination = OUT / 'phases' / (phase + '.joblib')
    if destination.exists():
        return phase, 'checkpoint'
    _, y, _, te = old.inputs('Fixed118')
    folds = old.folds('Fixed118', 'final') if phase == 'final' else old.folds('Fixed118', 'inner', int(phase[-1]))
    opts = options()
    audit, cv = [], {c['candidate_id']: [] for c in opts}
    started = time.time()
    for fold, (a, b) in enumerate(folds, 1):
        RANK_CACHE.clear(); BLOCK_CACHE.clear()
        assert not (set(a) | set(b)) & set(te) and (y[te] == -1).all()
        prepared, clinical = {}, {scope: clinical_prepare(TABLES, a, b, scope) for scope in ('None', 'Basic5', 'Extended11')}
        fitted = {}
        for c in opts:
            key = (c['image_key'], c['architecture'], c['size'], c['quality'])
            if key not in prepared:
                prepared[key] = image_prepare(TABLES, a, b, y, c)
            identity = key + (c['clinical_scope'], c['C'])
            if identity not in fitted:
                fitted[identity] = fit_prepared(prepared[key], clinical[c['clinical_scope']], a, y, c)
            f = fitted[identity]
            cid = c['candidate_id']
            audit.append({'InnerFold': fold, 'candidate_id': cid, 'ValidationAUC': float(roc_auc_score(y[b], f['evaluation_probability'])),
                'FitAUC': float(roc_auc_score(y[a], f['train_probability'])), 'ImageDimensions': f['image_dimensions'],
                'ClinicalDimensions': f['clinical_dimensions'], 'Available': f['available']})
            cv[cid].append({'train': a, 'validation': b, 'train_probability': f['train_probability'], 'validation_probability': f['evaluation_probability']})
        print(phase + ' inner' + str(fold) + '/5 complete', flush=True)
    frame = pd.DataFrame(audit).set_index(['candidate_id', 'InnerFold'])
    grid = []
    for c in opts:
        rr = frame.loc[c['candidate_id']]
        mean, sd, fitted_mean = rr.ValidationAUC.mean(), rr.ValidationAUC.std(ddof=1), rr.FitAUC.mean()
        available = bool(rr.Available.all())
        grid.append({**c, 'MeanValidationAUC': float(mean), 'SD': float(sd), 'MeanFitAUC': float(fitted_mean),
            'MeanImageDimensions': float(rr.ImageDimensions.mean()), 'MeanClinicalDimensions': float(rr.ClinicalDimensions.mean()),
            'AvailableAllFolds': available, 'Objective': float(mean - .10 * sd - .30 * max(fitted_mean - mean - .10, 0)) if available else -1.})
    joblib.dump({'phase': phase, 'grid': grid, 'audit': audit, 'cv': cv}, destination, compress=3)
    return phase, round(time.time() - started, 1)


def assemble():
    data = joblib.load(OUT / 'training_features.joblib')
    cohort, y, tr, te = old.inputs('Fixed118')
    outer, metrics, predictions = [], [], []
    for fold, (a, b) in enumerate(old.folds('Fixed118', 'outer'), 1):
        phase = joblib.load(OUT / 'phases' / ('outer' + str(fold) + '.joblib'))
        models = {}
        for family in FAMILIES:
            c = config(choose(phase['grid'], family))
            f = fit(data, a, b, y, c)
            models[family] = f
            metrics.append({'ModelKey': family, 'Fold': fold, **c, 'AUC': float(roc_auc_score(y[b], f['evaluation_probability']))})
            predictions.extend({'ModelKey': family, 'Fold': fold, 'AnalysisID': cohort.AnalysisID.iloc[i], 'Outcome': int(y[i]), 'Probability': float(p)} for i, p in zip(b, f['evaluation_probability']))
        outer.append({'train': a, 'validation': b, 'models': models})
    phase = joblib.load(OUT / 'phases/final.joblib')
    final, choices, features = {}, [], []
    for family in FAMILIES:
        chosen = choose(phase['grid'], family)
        c = config(chosen)
        f = fit(data, tr, [], y, c)
        oof = np.full(138, np.nan)
        for rr in phase['cv'][c['candidate_id']]:
            oof[rr['validation']] = rr['validation_probability']
        assert np.isfinite(oof[tr]).all() and np.isnan(oof[te]).all()
        f['training_oof'] = oof
        f['threshold'] = previous.stable.old.common.threshold(y[tr], oof[tr])
        final[family] = f
        vv = [r['AUC'] for r in metrics if r['ModelKey'] == family]
        pp = [r for r in predictions if r['ModelKey'] == family]
        choices.append({'ModelKey': family, **c, 'TrainFitAUC': float(roc_auc_score(y[tr], f['train_probability'])),
            'OuterMeanAUC': float(np.mean(vv)), 'OuterSD': float(np.std(vv, ddof=1)),
            'PooledOOFAUC': float(roc_auc_score([r['Outcome'] for r in pp], [r['Probability'] for r in pp])),
            'FinalInnerAUC': chosen['MeanValidationAUC'], 'ImageDimensions': f['image_dimensions'],
            'ClinicalDimensions': f['clinical_dimensions'], 'RawImageFeatureCount': f['image_raw_features'],
            'AvailableFinal': f['available'], 'Threshold': f['threshold'],
            **{m + 'Count': sum(len(s['indices']) for s in f['blocks'] if s['modality'] == m) for m in MODALITIES}})
        for block in f['blocks']:
            features.extend({'ModelKey': family, 'Modality': block['modality'], 'Feature': name, 'Role': 'PCA source' if c['architecture'] == 'PCA' else 'Base LR input' if c['architecture'] == 'Late' else 'LR input'} for name in block['names'])
        features.extend({'ModelKey': family, 'Modality': 'Clinical', 'Feature': name, 'Role': 'Clinical input'} for name in f['clinical']['names'])
    joblib.dump({'outer': outer, 'final': final}, OUT / 'models/frozen_models.joblib', compress=3)
    csv('training_choices', choices); csv('outer_metrics', metrics); csv('nested_predictions', predictions); csv('selected_features', features)
    csv('inner_candidate_scores', [{'Phase': p, **r} for p in ['outer' + str(i) for i in range(1, 6)] + ['final'] for r in joblib.load(OUT / 'phases' / (p + '.joblib'))['grid']])
    files = [OUT / 'models/frozen_models.joblib', OUT / 'training_features.joblib'] + list((OUT / 'phases').glob('*.joblib')) + list(OUT.glob('*.csv'))
    save(OUT / 'training_frozen.json', {'passed': True, 'protocol_sha256': digest(OUT / 'protocol.json'),
        'files': {str(p.relative_to(OUT)): digest(p) for p in files}, 'test_scored': False})
    print(pd.DataFrame(choices)[['ModelKey', 'architecture', 'K', 'TrainFitAUC', 'OuterMeanAUC', 'OuterSD']].to_string(index=False), flush=True)


def frozen():
    protect()
    fr = json.loads((OUT / 'training_frozen.json').read_text(encoding='utf-8'))
    assert digest(OUT / 'protocol.json') == fr['protocol_sha256']
    for name, sha in fr['files'].items():
        assert digest(OUT / name) == sha, name
    return fr


def train(workers):
    protect()
    assert not (OUT / 'test_started.json').exists()
    if (OUT / 'training_frozen.json').exists():
        frozen(); return
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize) as pool:
        futures = [pool.submit(tune, p) for p in ['outer' + str(i) for i in range(1, 6)] + ['final']]
        for f in as_completed(futures):
            print('Phase complete ' + str(f.result()), flush=True)
    assemble()


def test():
    frozen()
    assert json.loads((OUT / 'training_verification.json').read_text(encoding='utf-8'))['passed']
    assert not (OUT / 'test_started.json').exists(), 'Do not repeat holdout scoring'
    data = input_data('test')
    joblib.dump(data, OUT / 'test_features.joblib', compress=3)
    save(OUT / 'test_started.json', {'scoring_stages': 1, 'test_model_calls': 5, 'freeze_sha256': digest(OUT / 'training_frozen.json')})
    cohort, _, _, te = old.inputs('Fixed118')
    truth = cohort.Outcome.to_numpy(int)[te]
    final = joblib.load(OUT / 'models/frozen_models.joblib')['final']
    scores = {family: predict(final[family], data, te) for family in FAMILIES}
    draws = bootstrap(truth, scores, 2026102361)
    predictions, metrics = [], []
    for family, p in scores.items():
        threshold = final[family]['threshold']
        low, high = np.quantile(draws[family], [.025, .975])
        metrics.append({'ModelKey': family, 'N': len(te), **previous.stable.old.common.old.metrics(truth, p, threshold),
            'AUC_CI95Low': low, 'AUC_CI95High': high})
        predictions.extend({'ModelKey': family, 'AnalysisID': cohort.AnalysisID.iloc[i], 'Outcome': int(y), 'Probability': float(pp), 'Threshold': threshold} for i, y, pp in zip(te, truth, p))
    csv('test_predictions', predictions); csv('test_metrics', metrics)
    frozen()
    save(OUT / 'test_complete.json', {'passed': True, 'scoring_stages': 1, 'test_model_calls': 5,
        'prediction_rows': len(predictions), 'model_refits_after_test_start': 0, 'threshold_changes': 0})
    print('Five frozen multiblock models scored once on36', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=('define', 'smoke', 'train', 'verify-training', 'test', 'verify-test', 'report'))
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    with threadpool_limits(1):
        if args.phase == 'define':
            define()
        elif args.phase == 'train':
            train(args.workers)
        elif args.phase == 'test':
            test()
        elif args.phase == 'report':
            import report_multiblock_fixed118_20261023 as report
            report.main()
        else:
            import verify_multiblock_fixed118_20261023 as verify
            {'smoke': verify.smoke, 'verify-training': verify.training, 'verify-test': verify.inference}[args.phase]()
