# -*- coding: utf-8 -*-
"""Add four model families to the immutable Random120 habitat Stack12 comparison."""
import os
os.environ.update({'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
import argparse
import json
import time
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from threadpoolctl import threadpool_limits
import run_pure_habitat_random120_stacking_20261023 as habitat
import run_pure_habitat_repeatedinner_20261023 as latent
import multimodal_fixed118_study_20261023 as source
import ivim_five_k_study_20261021 as clinical_source
from run_filtered_ordered_fixed118_20261023 import transform as clinical_transform

ROOT, save, digest = habitat.ROOT, habitat.save, habitat.digest
OUT = ROOT/'lvi_prediction/results/lvi_random120_five_models_20261023'
FAMILIES = ('Habitat', 'Whole', 'Clinical', 'HabitatClinical', 'WholeClinical')
TUNED = ('Whole', 'Clinical', 'WholeClinical')
PHASES = tuple('outer%d' % i for i in range(1, 6))+('final',)
BASIC = ('Clinical_PreopT', 'Clinical_PreopN', 'Clinical_EMVI', 'Clinical_MRF', 'Clinical_logCEA')


def csv(name, rows):
    pd.DataFrame(rows).to_csv(OUT/(name+'.csv'), index=False, encoding='utf-8-sig')


def read(name):
    return pd.read_csv(OUT/(name+'.csv'), float_precision='round_trip', keep_default_na=False, na_values=[''])


def inputs():
    return habitat.parent.inputs('Random120')


def options():
    result = []
    learners = [('Ridge', 0, .003), ('Ridge', 0, .03), ('Ridge', 0, .3)]+[
        (method, n, C) for method, numbers in [('PCA', (2, 5)), ('PLS', (1, 2))]
        for n in numbers for C in (.03, .3)]
    for family in TUNED:
        specs = [(None, 'NA', 'Ridge', 0, C) for C in (.003, .03, .3)] if family == 'Clinical' else [
            (norm+'__Whole', norm, method, n, C) for norm in ('PercentileZ', 'RobustZ') for method, n, C in learners]
        scopes = ('Basic5', 'Extended11') if family != 'Whole' else ('None',)
        for key, norm, method, n, C in specs:
            for scope in scopes:
                result.append({'candidate_id': len(result), 'family': family, 'image_key': key,
                    'normalization': norm, 'method': method, 'components': n, 'C': C, 'clinical_scope': scope})
    assert {f: sum(c['family'] == f for c in result) for f in TUNED} == {'Whole': 22, 'Clinical': 6, 'WholeClinical': 44}
    return result


def config(row):
    return {k: row[k] for k in options()[0]}


def data(indices):
    result = source.whole(indices)
    result['Clinical'] = clinical_source.clinical(indices)
    return result


def clinical_prepare(tables, a, b, scope):
    d = tables['Clinical']; names = [r['name'] for r in d['metadata']]
    cols = [] if scope == 'None' else [names.index(n) for n in BASIC] if scope == 'Basic5' else list(range(11))
    x = d['base'][latent.rows(d, a)][:, cols]; xb = d['base'][latent.rows(d, b)][:, cols]
    return {**clinical_transform(x, xb), 'indices': cols, 'names': [names[j] for j in cols]}


def fit_prepared(p, cp, target, c, a, b, reduced=None):
    if p is None:
        reduced = {'za': np.empty((len(a), 0)), 'zb': np.empty((len(b), 0)),
            'reducer': None, 'latent_center': None, 'latent_scale': None, 'effective': 0,
            'projection': None, 'input_center': None}
    else:
        reduced = latent.reduction(p, target, c) if reduced is None else reduced
    xa = np.column_stack([reduced['za'], cp['xa']]); xb = np.column_stack([reduced['zb'], cp['xb']])
    lr = LogisticRegression(C=c['C'], penalty='l2', solver='liblinear', class_weight='balanced', max_iter=3000, random_state=64212)
    assert xa.shape[1] > 0 and np.isfinite(xa).all()
    lr.fit(xa, target)
    pa = lr.predict_proba(xa)[:, 1]; pb = lr.predict_proba(xb)[:, 1] if len(b) else np.empty(0)
    effective = reduced['effective']
    if p is None:
        beta, intercept = np.empty(0), float(lr.intercept_[0])
    elif c['method'] == 'Ridge':
        beta, intercept = lr.coef_[0, :effective].copy(), float(lr.intercept_[0])
    else:
        w = lr.coef_[0, :effective]/reduced['latent_scale']
        beta = reduced['projection']@w
        intercept = float(lr.intercept_[0]-reduced['input_center']@beta-reduced['latent_center']@w)
    clinical_beta = lr.coef_[0, effective:].copy()
    image_train = p['xa']@beta if p is not None else np.zeros(len(a))
    np.testing.assert_allclose(expit(image_train+cp['xa']@clinical_beta+intercept), pa, atol=1e-10, rtol=0)
    return {'config': c, 'train': np.asarray(a, int), 'validation': np.asarray(b, int),
        'image': {s: p[s] for s in ('indices', 'names', 'median', 'quantile', 'audit')} if p is not None else None,
        'clinical': {s: cp[s] for s in ('indices', 'names', 'median', 'low', 'high', 'center', 'scale')},
        'reducer': reduced['reducer'], 'latent_center': reduced['latent_center'], 'latent_scale': reduced['latent_scale'],
        'image_effective': effective, 'effective': effective+len(cp['indices']),
        'image_coef_gaussian': beta, 'clinical_coef': clinical_beta, 'intercept': intercept,
        'classifier': lr, 'available': bool(np.ptp(pa) > 1e-12), 'train_probability': pa, 'evaluation_probability': pb,
        'ScoreType': 'L2LR_probability_uncalibrated'}


def fit(tables, a, b, y, c):
    p = latent.prepare(tables[c['image_key']], a, b) if c['image_key'] is not None else None
    cp = clinical_prepare(tables, a, b, c['clinical_scope'])
    return fit_prepared(p, cp, y[a], c, a, b)


def transformed(f, tables, indices):
    p = f['image']
    if p is None:
        zi = np.empty((len(indices), 0))
    else:
        d = tables[f['config']['image_key']]
        x = d['base'][latent.rows(d, indices)][:, p['indices']]
        x = np.where(np.isfinite(x), x, p['median'])
        zi = np.clip(p['quantile'].transform(x), -2.5, 2.5) if len(indices) else np.empty_like(x)
    cp = f['clinical']; d = tables['Clinical']
    x = d['base'][latent.rows(d, indices)][:, cp['indices']]
    zc = (np.clip(np.where(np.isfinite(x), x, cp['median']), cp['low'], cp['high'])-cp['center'])/cp['scale']
    return zi, zc


def predict(f, tables, indices):
    zi, zc = transformed(f, tables, indices)
    return expit(zi@f['image_coef_gaussian']+zc@f['clinical_coef']+f['intercept'])


def source_stack():
    return joblib.load(habitat.OUT/'Random120/models/stack12.joblib')


def base_scores(sf, stage):
    return np.column_stack([f[stage+'_probability'] for f in sf['base']])


def fit_hc(tables, sf, y):
    a, b = sf['train'], sf['validation']
    pa, pb = base_scores(sf, 'train'), base_scores(sf, 'evaluation')
    pp = habitat.meta_prepare(sf['crossfit_base_scores'], pb)
    cp = clinical_prepare(tables, a, b, 'Extended11')
    xa = np.column_stack([pp['xa'], cp['xa']]); xb = np.column_stack([pp['xb'], cp['xb']])
    meta = LogisticRegression(C=.03, penalty='l2', solver='liblinear', class_weight='balanced', max_iter=3000, random_state=64212).fit(xa, y[a])
    za = np.column_stack([(np.clip(pa, pp['low'], pp['high'])-pp['center'])/pp['scale'], cp['xa']])
    return {'family': 'HabitatClinical', 'train': np.asarray(a, int), 'validation': np.asarray(b, int),
        'base_model_source': str((habitat.OUT/'Random120/models/stack12.joblib').relative_to(ROOT)),
        'base_count': 12, 'meta': meta, **{s: pp[s] for s in ('low', 'high', 'center', 'scale')},
        'clinical': {s: cp[s] for s in ('indices', 'names', 'median', 'low', 'high', 'center', 'scale')},
        'train_probability': meta.predict_proba(za)[:, 1],
        'evaluation_probability': meta.predict_proba(xb)[:, 1] if len(b) else np.empty(0),
        'meta_in_sample_association_probability': meta.predict_proba(xa)[:, 1],
        'ScoreType': 'StackLR_plus_clinical_probability_uncalibrated'}


def predict_hc(f, sf, tables, habitat_tables, indices):
    x = np.column_stack([habitat.parent.method.predict(base, habitat_tables, indices) for base in sf['base']])
    _, zc = transformed({'image': None, 'clinical': f['clinical']}, tables, indices)
    z = np.column_stack([(np.clip(x, f['low'], f['high'])-f['center'])/f['scale'], zc])
    return f['meta'].predict_proba(z)[:, 1]


def choose(grid, family):
    rr = [r for r in grid if r['family'] == family]
    return max(rr, key=lambda r: (r['AvailableAllFolds'], r['Objective'], -r['MeanEffective'],
        -r['MeanFeatureCount'], -r['C'], -r['candidate_id']))


def threshold(y, scores):
    fp, tp, thresholds = roc_curve(y, scores, drop_intermediate=False)
    return float(thresholds[np.argmax(tp-fp)])


def protect():
    p = json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    for field in ('source_hashes', 'protected_hashes'):
        for name, sha in p[field].items():
            assert digest(ROOT/name) == sha, name
    assert digest(OUT/'protocol.json') == json.loads((OUT/'definition_frozen.json').read_text(encoding='utf-8'))['protocol_sha256']
    return len(p['protected_hashes'])


def define():
    assert not OUT.exists(), 'Never overwrite an existing experiment'
    habitat.frozen()
    assert json.loads((habitat.OUT/'delivery_final_verification.json').read_text(encoding='utf-8'))['passed']
    old = json.loads((habitat.OUT/'protocol.json').read_text(encoding='utf-8'))
    protected = {**old['source_hashes'], **old['protected_hashes']}
    protected.update({str(p.relative_to(ROOT)): digest(p) for p in habitat.OUT.rglob('*') if p.is_file()})
    extra = [source.WHOLE/(stage+'_features.joblib') for stage in ('training', 'test120', 'omitted18')]+[
        clinical_source.CLINICAL/'clinical_extension_138.csv', clinical_source.CLINICAL/'clinical_extension_audit.json',
        ROOT/'lvi_prediction/run_pure_habitat_repeatedinner_20261023.py',
        ROOT/'lvi_prediction/run_filtered_ordered_fixed118_20261023.py',
        ROOT/'lvi_prediction/multimodal_fixed118_study_20261023.py', ROOT/'lvi_prediction/ivim_five_k_study_20261021.py',
        ROOT/'lvi_prediction/CURRENT_LVI_PURE_HABITAT_RANDOM120_EXPLORATION_20261023.json']
    for path in extra:
        protected[str(path.relative_to(ROOT))] = digest(path)
    OUT.mkdir(parents=True)
    for folder in ('phases', 'models', 'Random120', 'figures'):
        (OUT/folder).mkdir()
    for name in ('sample_manifest.json', 'nested_folds.json', 'sampling_record.json'):
        (OUT/name).write_bytes((habitat.OUT/name).read_bytes())
    for label in ('Train', 'Test', 'Sample'):
        (OUT/'Random120'/(label+'_cases.csv')).write_bytes((habitat.parent.OUT/'Random120'/(label+'_cases.csv')).read_bytes())
    sources = [ROOT/'lvi_prediction'/name for name in ('run_random120_five_models_20261023.py',
        'verify_random120_five_models_20261023.py', 'report_random120_five_models_20261023.py')]
    protocol = {'date': '2026-10-23', 'authorization': 'Add the previously agreed five model families and assess their performance.',
        'families': FAMILIES, 'cohort': 'Same ONE Random120 sample;84train57positive27negative/36test25positive11negative;no new draw.',
        'seeds': {'sample': 2026102307, 'split': 2026102308}, 'same_case_and_outer_inner_folds': True,
        'Habitat': 'Reuse immutable current Stack12 final fit,outer5 andtest36 scores;no H training or H test call in this extension.',
        'Whole': '372 nonshape radiomics:T2/ADC/D/f93each,patient-local original2 image normalizations;no new rawphysical features.',
        'whole_QC': 'Current training part complete>=.9,perturbation absoluteICC>=.75,nonconstant,absSpearman<.95;all eligible features;no top3/6 cap andno outcome-univariate ranking.',
        'whole_preprocess': 'Current train medianfill,Gaussian QuantileTransformer,no subsampling,clip[-2.5,2.5];same as currentH bases.',
        'Whole_and_WholeClinical_search': '2norms x11recipes:Ridge C.003/.03/.3,PCA2/5 orPLS1/2 C.03/.3. WC each image representation appends Basic5 orExtended11 clinical after image reduction.',
        'Clinical_search': 'Basic5 orExtended11;balancedL2LR C.003/.03/.3. No postoperative predictors.',
        'clinical_preprocess': 'Fold-only medianfill,P2.5/P97.5clip,median/IQR-normal-equivalent scale;SD then1 fallback.',
        'HabitatClinical': 'Reuse exact12H bases andtheir repeatedinnerOOFs,append11clinical variables to12 standardizedOOFs;fixed balancedL2LR C.03;no new HC hyperparameter search. Refit meta separately inside eachoutertrain andfull84.',
        'HC_explanation': 'This tests augmenting currentH with clinical information;HC is a23-input score-level stack,not a new habitat feature extraction.',
        'inner': 'Same inner5 repeated3 times peroutertrain,15 heldout AUCs;each imageQC,reduction andclinicaltransform fitoncurrentinnertrain.',
        'objective': 'mean15AUC-.10SD-.30max(meanFit-mean15AUC-.10,0);ties fewer effective predictors,rawfeatures,C,candidateID.',
        'outer': 'Same5fixedfolds;independent train-only selection perouterfold;full84 final selection doesnot consultouterAUC or36test.',
        'thresholds': 'Four new families:Youden from their FULL pipelineouterOOF on84 train;freeze before36 scoring. H reuses originalsamekindthreshold.',
        'test_stage': 'Freeze andverify allfournew families beforeonejointstage;4new completepipeline predictions plusreuse36savedHscores. HC12base predictions are components ofitsonepipeline.',
        'comparison': 'Report fivefamilies allfit/outerSD/pooledOOF/testCI/classification;paired5000 bootstrap usingalignedcaseIDs.',
        'interpretation': 'Conditional exploratory comparison after prior H selection and many prior cohort/test uses;not external validation or unbiased confirmatory estimation.',
        'Dstar': False, 'automatic_adoption': False, 'formal_use': False, 'excluded_experiments_used': False,
        'candidate_counts': {f: sum(c['family'] == f for c in options()) for f in TUNED},
        'source_hashes': {str(p.relative_to(ROOT)): digest(p) for p in sources}, 'protected_hashes': protected}
    save(OUT/'protocol.json', protocol)
    save(OUT/'definition_frozen.json', {'passed': True, 'new_test_calls': 0, 'protocol_sha256': digest(OUT/'protocol.json')})
    csv('candidate_definitions', options())
    print('Defined72 nested candidates andfixedHC;H reused;same84/36,no newdraw', flush=True)


def tune_phase(tables, phase, y):
    dest = OUT/'phases'/(phase+'.joblib')
    if dest.exists():
        return joblib.load(dest)
    opts = options(); audit = []; cv = {c['candidate_id']: [] for c in opts}
    for rep, fold, a, b in habitat.parent.folds('Random120', phase):
        image = {key: latent.prepare(d, a, b) for key, d in tables.items() if key != 'Clinical'}
        cp = {scope: clinical_prepare(tables, a, b, scope) for scope in ('None', 'Basic5', 'Extended11')}
        reductions = {}
        for c in opts:
            p = image.get(c['image_key']); key = (c['image_key'], c['method'], c['components'])
            if p is not None and key not in reductions:
                reductions[key] = latent.reduction(p, y[a], c)
            f = fit_prepared(p, cp[c['clinical_scope']], y[a], c, a, b, reductions.get(key))
            cid = c['candidate_id']
            audit.append({'Repeat': rep, 'Fold': fold, 'candidate_id': cid,
                'FitAUC': float(roc_auc_score(y[a], f['train_probability'])),
                'AUC': float(roc_auc_score(y[b], f['evaluation_probability'])),
                'FeatureCount': len(p['indices']) if p is not None else 0,
                'Effective': f['effective'], 'Available': f['available']})
            cv[cid].append({'Repeat': rep, 'Fold': fold, 'train': a, 'validation': b,
                'train_probability': f['train_probability'], 'validation_probability': f['evaluation_probability']})
        if fold == 5:
            print(phase+' repeat%d/3 complete:72 configurations x5inner folds' % rep, flush=True)
    frame = pd.DataFrame(audit)
    grid = []
    for c in opts:
        rr = frame[frame.candidate_id == c['candidate_id']]
        mean, sd, train = rr.AUC.mean(), rr.AUC.std(ddof=1), rr.FitAUC.mean()
        grid.append({**c, 'MeanValidationAUC': float(mean), 'SD': float(sd), 'MeanFitAUC': float(train),
            'MeanFeatureCount': float(rr.FeatureCount.mean()), 'MeanEffective': float(rr.Effective.mean()),
            'AvailableAllFolds': bool(rr.Available.all()),
            'Objective': float(mean-.1*sd-.3*max(train-mean-.1, 0))})
    result = {'phase': phase, 'cv': cv, 'audit': audit, 'grid': grid}
    joblib.dump(result, dest, compress=3)
    return result


def train():
    protect(); assert not (OUT/'training_frozen.json').exists() and not (OUT/'test_started.json').exists()
    cohort, y, tr, te = inputs(); assert (y[te] == -1).all()
    tables = data(tr); joblib.dump(tables, OUT/'training_features.joblib', compress=3)
    stack = source_stack(); results = {}; metrics = []; predictions = []; selected = []; final_choices = []
    for phase in PHASES:
        if phase == 'final':
            a, b, sf = tr, np.empty(0, int), stack['final']
        else:
            k = int(phase[-1]); a, b = habitat.parent.folds('Random120')[k-1]; sf = stack['outer'][k-1]
        cp = tune_phase(tables, phase, y); results[phase] = {}
        for family in TUNED:
            row = choose(cp['grid'], family); c = config(row)
            f = fit(tables, a, b, y, c); results[phase][family] = f
            selected.append({'Phase': phase, **row})
        results[phase]['HabitatClinical'] = fit_hc(tables, sf, y)
        for family, f in results[phase].items():
            if phase != 'final':
                metrics.append({'ModelKey': family, 'Fold': int(phase[-1]), 'FitAUC': float(roc_auc_score(y[a], f['train_probability'])),
                    'AUC': float(roc_auc_score(y[b], f['evaluation_probability']))})
                predictions.extend({'ModelKey': family, 'Fold': int(phase[-1]), 'Index': int(i),
                    'AnalysisID': cohort.AnalysisID.iloc[i], 'Outcome': int(y[i]), 'Probability': float(p)} for i, p in zip(b, f['evaluation_probability']))
            else:
                pp = pd.DataFrame(predictions); pp = pp[pp.ModelKey == family].set_index('Index').loc[tr]
                mm = pd.DataFrame(metrics); mm = mm[mm.ModelKey == family]
                f['threshold'] = threshold(y[tr], pp.Probability.to_numpy())
                f['threshold_source'] = 'full_pipeline_outerOOF84_Youden'
                final_choices.append({'ModelKey': family, 'TrainFitAUC': float(roc_auc_score(y[tr], f['train_probability'])),
                    'OuterMeanAUC': float(mm.AUC.mean()), 'OuterSD': float(mm.AUC.std(ddof=1)),
                    'PooledOOFAUC': float(roc_auc_score(y[tr], pp.Probability.to_numpy())), 'Threshold': f['threshold'],
                    'MetaInSampleAssociationAUC': float(roc_auc_score(y[tr], f['meta_in_sample_association_probability'])) if family == 'HabitatClinical' else None})
        joblib.dump(results[phase], OUT/'models'/(phase+'.joblib'), compress=3)
        print(phase+' fournewmodel refits complete;no testscoring', flush=True)
    h = habitat.read('training_choices'); h = h[h.Group == 'Random120'].iloc[0]
    final_choices.append({'ModelKey': 'Habitat', **{s: h[s] for s in ('TrainFitAUC', 'OuterMeanAUC', 'OuterSD', 'PooledOOFAUC', 'Threshold')},
        'MetaInSampleAssociationAUC': h.MetaInSampleOOFAssociationAUC})
    hm = habitat.read('outer_metrics'); hm = hm[(hm.Group == 'Random120') & (hm.ModelKey == 'Stack12')].copy(); hm['ModelKey'] = 'Habitat'
    hp = habitat.read('nested_predictions'); hp = hp[(hp.Group == 'Random120') & (hp.ModelKey == 'Stack12')].copy(); hp['ModelKey'] = 'Habitat'
    csv('training_choices', final_choices); csv('outer_metrics', metrics+hm.drop(columns='Group').to_dict('records'))
    csv('nested_predictions', predictions+hp.drop(columns='Group').to_dict('records')); csv('selected_configurations', selected)
    features = []
    for family, f in results['final'].items():
        if family == 'HabitatClinical':
            features.extend({'ModelKey': family, 'Feature': 'HabitatBaseScore%d' % (j+1), 'Modality': 'HabitatScore', 'CoefficientSpace': 'standardized_OOFscore', 'Coefficient': float(w)} for j, w in enumerate(f['meta'].coef_[0, :12]))
            features.extend({'ModelKey': family, 'Feature': n, 'Modality': 'Clinical', 'CoefficientSpace': 'robust_scaled_clinical', 'Coefficient': float(w)} for n, w in zip(f['clinical']['names'], f['meta'].coef_[0, 12:]))
        else:
            if f['image'] is not None:
                features.extend({'ModelKey': family, 'Feature': n, 'Modality': n.split('__')[0], 'CoefficientSpace': 'Gaussian_feature', 'Coefficient': float(w)} for n, w in zip(f['image']['names'], f['image_coef_gaussian']))
            features.extend({'ModelKey': family, 'Feature': n, 'Modality': 'Clinical', 'CoefficientSpace': 'robust_scaled_clinical', 'Coefficient': float(w)} for n, w in zip(f['clinical']['names'], f['clinical_coef']))
    csv('final_model_features', features)
    files = [p for p in OUT.rglob('*') if p.is_file()]
    save(OUT/'training_frozen.json', {'passed': True, 'new_test_calls': 0, 'protocol_sha256': digest(OUT/'protocol.json'),
        'files': {str(p.relative_to(OUT)): digest(p) for p in files}, 'new_final_models': 4, 'reused_Habitat': True})
    print(pd.DataFrame(final_choices).to_string(index=False), flush=True)


def frozen():
    protect(); f = json.loads((OUT/'training_frozen.json').read_text(encoding='utf-8'))
    assert f['passed'] and digest(OUT/'protocol.json') == f['protocol_sha256']
    for name, sha in f['files'].items():
        assert digest(OUT/name) == sha, name
    return f


def test():
    frozen(); assert not (OUT/'test_started.json').exists()
    assert json.loads((OUT/'training_verification.json').read_text(encoding='utf-8'))['passed']
    cohort, _, tr, te = inputs(); tables = data(te)
    joblib.dump(tables, OUT/'test_features.joblib', compress=3)
    final = joblib.load(OUT/'models/final.joblib'); sf = source_stack()['final']
    ht = joblib.load(habitat.OUT/'Random120/test_features.joblib')
    old = habitat.read('test_predictions'); old = old[old.ModelKey == 'Stack12'].set_index('Index').loc[te]
    assert old.AnalysisID.tolist() == cohort.AnalysisID.iloc[te].tolist()
    save(OUT/'test_started.json', {'new_complete_model_calls': 4, 'reused_Habitat_scores': 36, 'new_Habitat_calls': 0,
        'test_stages': 1, 'freeze_sha256': digest(OUT/'training_frozen.json')})
    scores = {'Habitat': old.Probability.to_numpy()}
    for family in TUNED:
        scores[family] = predict(final[family], tables, te)
    scores['HabitatClinical'] = predict_hc(final['HabitatClinical'], sf, tables, ht, te)
    truth = cohort.Outcome.to_numpy(int)[te]
    from verify_ivim_df63_20261020 import draws
    samples = draws(truth, scores, 2026102399)
    metrics, predictions = [], []
    for family in FAMILIES:
        p = scores[family]; f = sf if family == 'Habitat' else final[family]
        lo, hi = np.quantile(samples[family], [.025, .975])
        tn, fp, fn, tp = confusion_matrix(truth, p >= f['threshold'], labels=[0, 1]).ravel()
        metrics.append({'ModelKey': family, 'N': len(te), 'Positive': int(truth.sum()), 'Negative': int((1-truth).sum()),
            'AUC': float(roc_auc_score(truth, p)), 'AUC_CI95Low': float(lo), 'AUC_CI95High': float(hi),
            'Threshold': f['threshold'], 'Sensitivity': tp/(tp+fn), 'Specificity': tn/(tn+fp), 'Accuracy': (tp+tn)/len(te),
            'TP': int(tp), 'FN': int(fn), 'TN': int(tn), 'FP': int(fp), 'ScoreType': f['ScoreType'],
            'ReusedSavedScores': family == 'Habitat'})
        predictions.extend({'ModelKey': family, 'Index': int(i), 'AnalysisID': cohort.AnalysisID.iloc[i], 'Outcome': int(t),
            'Probability': float(v), 'Prediction': int(v >= f['threshold'])} for i, t, v in zip(te, truth, p))
    csv('test_metrics', metrics); csv('test_predictions', predictions)
    paired = []
    for aa, bb in [('HabitatClinical', 'Habitat'), ('WholeClinical', 'Whole'), ('HabitatClinical', 'Clinical'),
                   ('WholeClinical', 'Clinical'), ('Habitat', 'Whole')]:
        lo, hi = np.quantile(samples[aa]-samples[bb], [.025, .975])
        paired.append({'Comparison': aa+'_minus_'+bb, 'AUC_Difference': float(roc_auc_score(truth, scores[aa])-roc_auc_score(truth, scores[bb])),
            'CI95Low': float(lo), 'CI95High': float(hi), 'BootstrapN': 5000})
    csv('paired_test_differences', paired)
    save(OUT/'test_complete.json', {'passed': True, 'new_complete_model_calls': 4, 'new_Habitat_calls': 0,
        'reused_Habitat_scores': 36, 'test_stages': 1, 'post_test_fits': 0, 'post_test_threshold_changes': 0})
    print(pd.DataFrame(metrics).to_string(index=False), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=('define', 'train', 'test'))
    with threadpool_limits(1):
        globals()[p.parse_args().action]()
