# -*- coding: utf-8 -*-
"""One new random cohort; original118 control; nested pure habitat model improvement."""
import os
os.environ.update({'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from threadpoolctl import threadpool_limits
import pure_habitat_robust_nonlinear_20261023 as method
import run_k3_two_norm_random90_48_138_20261013 as cache
import habitat_stable_views_20261013 as views

previous = method.reference
ROOT, save, digest = previous.ROOT, previous.save, previous.digest
OUT = ROOT/'lvi_prediction/results/lvi_pure_habitat_random120_robust_nonlinear_20261023'
GROUPS = ('Original118', 'Random120')
PHASES = previous.PHASES


def csv(name, rows):
    pd.DataFrame(rows).to_csv(OUT/(name+'.csv'), index=False, encoding='utf-8-sig')


def read(name):
    return pd.read_csv(OUT/(name+'.csv'), float_precision='round_trip', keep_default_na=False, na_values=[''])


def cohort():
    return previous.previous.common.inputs('Fixed118')[0]


def draw():
    c = cohort(); truth = c.Outcome.to_numpy(int)
    sample, omitted = train_test_split(np.arange(138), train_size=120, stratify=truth, random_state=2026102307)
    tr, te = train_test_split(sample, train_size=84, stratify=truth[sample], random_state=2026102308)
    return np.sort(sample), np.sort(tr), np.sort(te), np.sort(omitted)


def inputs(group):
    c = cohort()
    record = json.loads((OUT/'sample_manifest.json').read_text(encoding='utf-8'))[group]
    tr, te = np.asarray(record['train'], int), np.asarray(record['test'], int)
    y = np.full(138, -1, int); y[tr] = c.Outcome.to_numpy(int)[tr]
    assert not set(tr) & set(te)
    return c, y, tr, te


def data(indices):
    raw = cache.cached_features(indices)
    compact = {}
    for norm, d in raw.items():
        selected, _ = views.specifications(d['metadata'])
        compact[norm] = {'indices': d['indices'], 'metadata': [d['metadata'][j] for j in selected],
            **{kind: d[kind][:, selected] for kind in previous.feature.KINDS}}
    return previous.feature.build(compact)


def candidates(group):
    return method.options()+(method.reference_options() if group == 'Random120' else [])


def families(group):
    return ('Improved', 'Reference') if group == 'Random120' else ('Improved',)


def define():
    assert not OUT.exists()
    previous.frozen()
    assert json.loads((previous.OUT/'delivery_final_verification.json').read_text(encoding='utf-8'))['passed']
    parent = json.loads((previous.OUT/'protocol.json').read_text(encoding='utf-8'))
    protected = {**parent['protected_hashes'], **parent['source_hashes']}
    protected.update({str(p.relative_to(ROOT)): digest(p) for p in previous.OUT.rglob('*') if p.is_file()})
    OUT.mkdir(parents=True)
    c, _, original_tr, original_te = previous.previous.common.inputs('Fixed118')
    sample, newtr, newte, omitted = draw()
    manifest = {}
    folds = {}
    for group, ss, tr, te in [('Original118', np.arange(118), original_tr, original_te), ('Random120', sample, newtr, newte)]:
        (OUT/group).mkdir(); (OUT/group/'phases').mkdir(); (OUT/group/'models').mkdir()
        manifest[group] = {'sample': ss.tolist(), 'train': tr.tolist(), 'test': te.tolist(),
            'sample_ids': c.AnalysisID.iloc[ss].tolist(), 'train_ids': c.AnalysisID.iloc[tr].tolist(), 'test_ids': c.AnalysisID.iloc[te].tolist(),
            'counts': {label: {'N': len(ix), 'Positive': int(c.Outcome.iloc[ix].sum()), 'Negative': int((c.Outcome.iloc[ix] == 0).sum())}
                       for label, ix in [('Sample', ss), ('Train', tr), ('Test', te)]}}
        for label, ix in [('Sample', ss), ('Train', tr), ('Test', te)]:
            c.iloc[ix].to_csv(OUT/group/(label+'_cases.csv'), index=False, encoding='utf-8-sig')
        if group == 'Original118':
            outer = previous.previous.local.study.folds('Fixed118', 'outer')
            inner = {phase: [{'Repeat': r, 'Fold': k, 'train': a.tolist(), 'validation': b.tolist()} for r, k, a, b in previous.folds(phase)] for phase in PHASES}
        else:
            truth = c.Outcome.to_numpy(int)
            outer = [(tr[a], tr[b]) for a, b in StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(tr)), truth[tr])]
            inner = {}
            for phase in PHASES:
                allowed = tr if phase == 'final' else outer[int(phase[-1])-1][0]
                seed = 62212 if phase == 'final' else 62200+int(phase[-1])
                inner[phase] = []
                for repeat in range(3):
                    ff = StratifiedKFold(5, shuffle=True, random_state=seed+10000*repeat)
                    for k, (a, b) in enumerate(ff.split(np.zeros(len(allowed)), truth[allowed]), 1):
                        inner[phase].append({'Repeat': repeat+1, 'Fold': k, 'train': allowed[a].tolist(), 'validation': allowed[b].tolist()})
        folds[group] = {'outer': [{'train': a.tolist(), 'validation': b.tolist()} for a, b in outer], 'inner': inner}
    save(OUT/'sample_manifest.json', manifest); save(OUT/'nested_folds.json', folds)
    save(OUT/'sampling_record.json', {'sample_seed': 2026102307, 'split_seed': 2026102308, 'number_of_random_draws': 1,
        'sample': sample.tolist(), 'omitted': omitted.tolist(), 'AUC_known_at_draw': False,
        'cross_group_overlap': {a+'_'+b: len(set(x) & set(z)) for a, x in [('OriginalTrain', original_tr), ('OriginalTest', original_te)]
                               for b, z in [('NewTrain', newtr), ('NewTest', newte)]}})
    sources = [ROOT/'lvi_prediction'/name for name in ('pure_habitat_robust_nonlinear_20261023.py',
        'run_pure_habitat_random120_20261023.py', 'verify_pure_habitat_random120_20261023.py', 'report_pure_habitat_random120_20261023.py')]
    protocol = {'date': '2026-10-23', 'authorization': 'Continue pure-habitat improvement toward outer5>=.7;human explicitly allows one additional random patient cohort of any size.',
        'primary_cohort': 'Random120', 'random_draws': 1, 'sampling_seed': 2026102307, 'split_seed': 2026102308,
        'sampling': 'Stratified sample120/138,then84train/36test;one fixed draw before AUC. Original118 exact82/36 is a training-only control.',
        'target': 'Primary Improved Random120 outer5 mean>=.70;report failures aswell;do not redraw/reseed to cross threshold.',
        'CV': 'Outer5 seed42;inner5x3 seeds62201..62205/final62212 with+10000/+20000 repeats. Original118 exactpreviousfolds.',
        'pure_habitat': True, 'Dstar': False, 'clinical': False, 'whole_predictors': False, 'K': 3,
        'images': 'Existing outcome-blind patient-local T2/ADC/IVIM D,f K3 partitions;independent native sequences;original two norm caches.',
        'views': 'OrderedCompact321 orContrastOnly220;regionalfeatures only.',
        'mask_modes': 'Base or perpatient median ofbase/1mm-erode/1mm-dilate feature values;no extra independent patients created.',
        'quality': 'Train-only complete>=.9,ICC>=.60/.75 chosen insideinnerCV,nonconstant,absSpearman<.95 dedup;all eligiblevariables,notopcount.',
        'preprocess': 'Current training-only median fill andGaussian quantile mapping clipped[-2.5,2.5],unchanged fromprevious study.',
        'classifiers': 'BalancedLR C.03/.3;PCA2+balancedLR C.03/.3;balancedRBF C1 gammaScale;ExtraTrees100 depth2/4 sqrtfeatures;HistGB100 lr.05 maxleaf3 depth2 L2=5 noearlystop.',
        'tree_min_leaf': 'max(8,ceil(.15*n_current_training_patients));all trees classbalanced;fixedseed64212;n_jobs1.',
        'improved_candidate_count': 128, 'reference_candidate_count': 66,
        'reference': 'Unchanged previous66-candidate repeatedinner method on exact sameRandom120 training folds;Original118 previous result reused frozen.',
        'improved_objective': 'mean15foldValidationAUC-.05SD;fitgap reported diagnostically,notpenalized inselection. Ties availability,effective dimensions,rawfeaturecount,C,candidateID.',
        'reference_objective': 'Unchanged mean-.10SD-.30max(meanFit-meanValidation-.10,0).',
        'test': 'After alltraining verification freezeRandom120 Improved+Reference andscoreboth on36 once. No perclassifier/feature/mask alternatives;no Original118 newtest scoring.',
        'threshold': 'Each full84 selectedconfig:average3 repeatedinnerOOFs percase;Youden thenfreeze.',
        'score_types': 'SVM sigmoidmargin ranking score,notcalibrated probability;other fitted probabilities also not externallycalibrated.',
        'prior_case_exposure': True, 'new_external_validation': False, 'automatic_adoption': False, 'formal_use': False,
        'excluded_experiments_used': False, 'sample_manifest_sha256': digest(OUT/'sample_manifest.json'),
        'nested_folds_sha256': digest(OUT/'nested_folds.json'), 'sampling_record_sha256': digest(OUT/'sampling_record.json'),
        'protected_hashes': protected, 'source_hashes': {str(p.relative_to(ROOT)): digest(p) for p in sources},
        'references': ['https://scikit-learn.org/1.3/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html',
            'https://scikit-learn.org/1.3/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html',
            'https://scikit-learn.org/1.3/auto_examples/model_selection/plot_nested_cross_validation_iris.html']}
    save(OUT/'protocol.json', protocol)
    save(OUT/'definition_frozen.json', {'passed': True, 'candidate_search_started': False, 'new_test_calls': 0, 'protocol_sha256': digest(OUT/'protocol.json')})
    csv('candidate_definitions', method.options()+method.reference_options())
    print('Frozen oneRandom120:84train57positive27negative;36test25positive11negative;128new/66reference;Original118 control', flush=True)


def protect():
    p = json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    for field in ('protected_hashes', 'source_hashes'):
        for name, sha in p[field].items():
            assert digest(ROOT/name) == sha, name
    for name in ('sample_manifest', 'nested_folds', 'sampling_record'):
        assert digest(OUT/(name+'.json')) == p[name+'_sha256']
    assert digest(OUT/'protocol.json') == json.loads((OUT/'definition_frozen.json').read_text(encoding='utf-8'))['protocol_sha256']
    return len(p['protected_hashes'])


def folds(group, phase=None):
    record = json.loads((OUT/'nested_folds.json').read_text(encoding='utf-8'))[group]
    return [(np.asarray(r['train'], int), np.asarray(r['validation'], int)) for r in record['outer']] if phase is None else [
        (r['Repeat'], r['Fold'], np.asarray(r['train'], int), np.asarray(r['validation'], int)) for r in record['inner'][phase]]


def choose(grid, family):
    return max([r for r in grid if r['Family'] == family], key=lambda r: (r['AvailableAllFolds'], r['Objective'],
        -r['MeanEffectiveDimensions'], -r['MeanFeatureCount'], -r['C'], -r['candidate_id']))


def config(row, group):
    return next(c for c in candidates(group) if c['CandidateKey'] == row['CandidateKey'])


def tune(group, phase, tables, y, te):
    path = OUT/group/'phases'/(phase+'.joblib')
    if path.exists():
        return
    audit, cv = [], {c['CandidateKey']: [] for c in candidates(group)}
    for repeat, fold, a, b in folds(group, phase):
        assert not (set(a) | set(b)) & set(te) and (y[te] == -1).all()
        qc = {key: method.quality(d, a) for key, d in tables.items() if not key.endswith('__MeanContrast')}
        prepared, latents = {}, {}
        for c in candidates(group):
            identity = (c['image_key'], c.get('mask_mode', 'Base'), c.get('icc_min', .75))
            if identity not in prepared:
                prepared[identity] = method.prepare(tables[c['image_key']], a, b, {'mask_mode': identity[1], 'icc_min': identity[2]}, qc.get(c['image_key']))
            p = prepared[identity]
            key = identity+(c['method'], c['components'])
            if c['method'] in ('Ridge', 'PCA', 'PLS') and p['indices'] and key not in latents:
                latents[key] = previous.reduction(p, y[a], c)
            if c['Family'] == 'Reference':
                f = previous.fitted(p, y[a], c, latents.get(key))
            else:
                f = method.fitted(p, y[a], c, latents.get(key))
            audit.append({'Repeat': repeat, 'InnerFold': fold, 'CandidateKey': c['CandidateKey'],
                'FitAUC': float(roc_auc_score(y[a], f['train_probability'])), 'ValidationAUC': float(roc_auc_score(y[b], f['evaluation_probability'])),
                'Available': f['available'], 'FeatureCount': len(f['indices']), 'Effective': f['effective']})
            cv[c['CandidateKey']].append({'Repeat': repeat, 'Fold': fold, 'train': a, 'validation': b,
                'fit_score': f['train_probability'], 'validation_score': f['evaluation_probability']})
        print('%s %s repeat%d fold%d/5:%d candidates complete' % (group, phase, repeat, fold, len(candidates(group))), flush=True)
    frame = pd.DataFrame(audit); grid = []
    for c in candidates(group):
        r = frame[frame.CandidateKey == c['CandidateKey']]
        mu, sd, fm = r.ValidationAUC.mean(), r.ValidationAUC.std(ddof=1), r.FitAUC.mean()
        objective = mu-.05*sd if c['Family'] == 'Improved' else mu-.1*sd-.3*max(fm-mu-.1, 0)
        available = bool(r.Available.all())
        grid.append({**c, 'MeanValidationAUC': float(mu), 'SD': float(sd), 'MeanFitAUC': float(fm),
            'MeanFeatureCount': float(r.FeatureCount.mean()), 'MeanEffectiveDimensions': float(r.Effective.mean()),
            'AvailableAllFolds': available, 'Objective': float(objective) if available else -1.})
    joblib.dump({'group': group, 'phase': phase, 'audit': audit, 'cv': cv, 'grid': grid}, path, compress=3)


def train():
    protect(); assert not (OUT/'test_started.json').exists() and not (OUT/'training_frozen.json').exists()
    choices, metrics, predictions, features = [], [], [], []
    for group in GROUPS:
        c, y, tr, te = inputs(group)
        tables = data(tr); joblib.dump(tables, OUT/group/'training_features.joblib', compress=3)
        for phase in PHASES:
            tune(group, phase, tables, y, te)
        outer = []
        for k, (a, b) in enumerate(folds(group), 1):
            cp = joblib.load(OUT/group/'phases'/('outer%d.joblib' % k)); models = {}
            for family in families(group):
                cfg = config(choose(cp['grid'], family), group)
                f = method.fit(tables, a, b, y, cfg); models[family] = f
                metrics.append({'Group': group, 'Family': family, 'Fold': k, **cfg,
                    'FitAUC': float(roc_auc_score(y[a], f['train_probability'])), 'AUC': float(roc_auc_score(y[b], f['evaluation_probability'])),
                    'FeatureCount': len(f['indices']), 'Effective': f['effective']})
                predictions.extend({'Group': group, 'Family': family, 'Fold': k, 'AnalysisID': c.AnalysisID.iloc[i],
                    'Index': int(i), 'Outcome': int(y[i]), 'Probability': float(s)} for i, s in zip(b, f['evaluation_probability']))
            outer.append({'train': a, 'validation': b, 'models': models})
        cp = joblib.load(OUT/group/'phases/final.joblib'); final = {}
        for family in families(group):
            winner = choose(cp['grid'], family); cfg = config(winner, group)
            f = method.fit(tables, tr, [], y, cfg)
            oo = np.full((138, 3), np.nan)
            for r in cp['cv'][cfg['CandidateKey']]:
                oo[r['validation'], r['Repeat']-1] = r['validation_score']
            assert np.isfinite(oo[tr]).all() and np.isnan(oo[te]).all()
            fp, tp, thresholds = roc_curve(y[tr], oo[tr].mean(1), drop_intermediate=False)
            f['threshold'] = float(thresholds[np.argmax(tp-fp)]); f['repeated_oof'] = oo
            assert np.isfinite(f['threshold'])
            final[family] = f
            mm = pd.DataFrame(metrics); mm = mm[(mm.Group == group) & (mm.Family == family)]
            pp = pd.DataFrame(predictions); pp = pp[(pp.Group == group) & (pp.Family == family)]
            choices.append({'Group': group, **cfg, 'TrainFitAUC': float(roc_auc_score(y[tr], f['train_probability'])),
                'OuterMeanAUC': float(mm.AUC.mean()), 'OuterSD': float(mm.AUC.std(ddof=1)),
                'PooledOOFAUC': float(roc_auc_score(pp.Outcome, pp.Probability)), 'FinalInnerMeanAUC': winner['MeanValidationAUC'],
                'FeatureCount': len(f['indices']), 'EffectiveDimensions': f['effective'], 'Threshold': f['threshold'],
                'TargetOuter07Met': bool(mm.AUC.mean() >= .7), 'NewTestScored': False})
            used = method.used_columns(f)
            for j, name in enumerate(f['names']):
                weight = f.get('coef_gaussian', np.full(len(f['names']), np.nan))[j]
                if cfg['method'].startswith('ExtraTrees'):
                    weight = f['classifier'].feature_importances_[j]
                features.append({'Group': group, 'Family': family, 'Feature': name, 'Modality': name.split('__')[0],
                    'UsedByClassifier': j in used, 'Weight': float(weight),
                    'WeightType': 'Gaussian_space_logit' if cfg['method'] in ('Ridge', 'PCA', 'PLS') else 'training_impurity_importance' if cfg['method'].startswith('ExtraTrees') else 'no_single_linear_weight'})
        joblib.dump({'group': group, 'train': tr, 'outer': outer, 'final': final}, OUT/group/'models/models.joblib', compress=3)
        print('Group frozen candidates:'+group, flush=True)
    csv('training_choices', choices); csv('outer_metrics', metrics); csv('nested_predictions', predictions); csv('selected_features', features)
    csv('inner_candidate_scores', [{'Group': group, 'Phase': phase, **r} for group in GROUPS for phase in PHASES
        for r in joblib.load(OUT/group/'phases'/(phase+'.joblib'))['grid']])
    files = [p for p in OUT.rglob('*') if p.is_file()]
    save(OUT/'training_frozen.json', {'passed': True, 'new_test_calls': 0, 'test_group': 'Random120', 'test_models': 2,
        'Original118_new_test_calls': 0, 'protocol_sha256': digest(OUT/'protocol.json'),
        'files': {str(p.relative_to(OUT)): digest(p) for p in files}})
    print(pd.DataFrame(choices)[['Group', 'Family', 'method', 'mask_mode' if 'mask_mode' in pd.DataFrame(choices) else 'view', 'TrainFitAUC', 'OuterMeanAUC', 'OuterSD']].to_string(index=False), flush=True)


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
    c, _, tr, te = inputs('Random120'); tables = data(te)
    joblib.dump(tables, OUT/'Random120/test_features.joblib', compress=3)
    result = joblib.load(OUT/'Random120/models/models.joblib')
    save(OUT/'test_started.json', {'model_calls': 2, 'test_stage': 1, 'freeze_sha256': digest(OUT/'training_frozen.json')})
    scores = {family: method.predict(f, tables, te) for family, f in result['final'].items()}
    truth = c.Outcome.to_numpy(int)[te]
    from verify_ivim_df63_20261020 import draws
    samples = draws(truth, scores, 2026102393)
    metrics, predictions = [], []
    for family, p in scores.items():
        f = result['final'][family]; lo, hi = np.quantile(samples[family], [.025, .975])
        tn, fp, fn, tp = confusion_matrix(truth, p >= f['threshold'], labels=[0, 1]).ravel()
        metrics.append({'Group': 'Random120', 'Family': family, 'N': len(te), 'Positive': int(truth.sum()), 'Negative': int((1-truth).sum()),
            'AUC': float(roc_auc_score(truth, p)), 'AUC_CI95Low': float(lo), 'AUC_CI95High': float(hi), 'Threshold': f['threshold'],
            'Sensitivity': tp/(tp+fn), 'Specificity': tn/(tn+fp), 'Accuracy': (tp+tn)/len(te),
            'TP': int(tp), 'FN': int(fn), 'TN': int(tn), 'FP': int(fp), 'ScoreType': f['ScoreType']})
        predictions.extend({'Group': 'Random120', 'Family': family, 'AnalysisID': c.AnalysisID.iloc[i], 'Index': int(i),
            'Outcome': int(t), 'Probability': float(s), 'Prediction': int(s >= f['threshold']), 'ScoreType': f['ScoreType']} for i, t, s in zip(te, truth, p))
    csv('test_metrics', metrics); csv('test_predictions', predictions)
    delta = samples['Improved']-samples['Reference']; lo, hi = np.quantile(delta, [.025, .975])
    csv('paired_test_difference', [{'Difference': float(roc_auc_score(truth, scores['Improved'])-roc_auc_score(truth, scores['Reference'])),
        'CI95Low': float(lo), 'CI95High': float(hi), 'BootstrapN': 5000}])
    save(OUT/'test_complete.json', {'passed': True, 'model_calls': 2, 'test_stages': 1, 'post_test_fits': 0,
        'post_test_threshold_changes': 0, 'Original118_new_test_calls': 0, 'metrics': metrics})
    print(pd.DataFrame(metrics).to_string(index=False), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=('define', 'train', 'test'))
    with threadpool_limits(1):
        globals()[p.parse_args().action]()
