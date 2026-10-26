# -*- coding: utf-8 -*-
"""Small prespecified mean-ensemble search; pure habitat only, no test selection."""
import os
os.environ.update({'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from threadpoolctl import threadpool_limits
import run_common_habitat_fixed118_20261023 as common
import run_filtered_ordered_fixed118_20261023 as local

ROOT, save, digest = common.ROOT, common.save, common.digest
OUT = ROOT/'lvi_prediction/results/lvi_pure_habitat_ensemble_fixed118_20261023'
GROUPS = ('LocalMean8', 'CommonMean12', 'EqualBlend')


def csv(name, rows):
    pd.DataFrame(rows).to_csv(OUT/(name+'.csv'), index=False, encoding='utf-8-sig')


def read(name):
    return pd.read_csv(OUT/(name+'.csv'), float_precision='round_trip', keep_default_na=False, na_values=[''])


def members():
    ll = [c for c in local.study.options() if c['family'] == 'Habitat' and c['method'] == 'Ridge' and c['C'] == .03 and c['top'] in (5, 15)]
    cc = [c for c in common.options() if c['panel'] == 'Combined' and c['method'] == 'Ridge' and c['C'] == .03 and c['top'] in (5, 15)]
    assert len(ll) == 8 and len(cc) == 12
    return {**{'L%d' % c['candidate_id']: {'source': 'Local', 'config': c} for c in ll},
            **{'C%d' % c['candidate_id']: {'source': 'Common', 'config': c} for c in cc}}


def ids(group):
    return [key for key, m in members().items() if group == 'EqualBlend' or m['source'] == ('Local' if group == 'LocalMean8' else 'Common')]


def weights(group):
    keys = ids(group)
    if group == 'EqualBlend':
        return {key: .5/(8 if key.startswith('L') else 12) for key in keys}
    return {key: 1./len(keys) for key in keys}


def average(scores, group):
    ww = weights(group)
    assert abs(sum(ww.values())-1.) < 1e-12
    return sum(ww[key]*scores[key] for key in ww)


def define():
    assert not OUT.exists()
    common.frozen(); local.frozen()
    assert not (common.OUT/'test_started.json').exists()
    assert json.loads((common.OUT/'training_verification.json').read_text(encoding='utf-8'))['passed']
    prior = json.loads((common.OUT/'protocol.json').read_text(encoding='utf-8'))
    protected = {**prior['protected_hashes'], **prior['source_hashes']}
    protected.update({str(p.relative_to(ROOT)): digest(p) for p in common.OUT.rglob('*') if p.is_file()})
    OUT.mkdir(parents=True)
    for folder in ('models', 'Fixed118', 'test_patient_features', 'figures'):
        (OUT/folder).mkdir()
    for name in ('Train82_cases.csv', 'Test36_cases.csv', 'Sample118_cases.csv', 'nested5_folds.json'):
        (OUT/'Fixed118'/name).write_bytes((common.OUT/'Fixed118'/name).read_bytes())
    (OUT/'sample_manifest.json').write_bytes((common.OUT/'sample_manifest.json').read_bytes())
    sources = [ROOT/'lvi_prediction'/name for name in ('run_pure_habitat_ensemble_fixed118_20261023.py',
        'verify_pure_habitat_ensemble_fixed118_20261023.py', 'report_pure_habitat_ensemble_fixed118_20261023.py')]
    method = {'date': '2026-10-23', 'authorization': 'Focus on pure habitat improvement;other model families auxiliary.',
        'cohort': 'Exact original118 and82/36;identical nestedouter5/inner5 folds.',
        'motivation': 'The common-centroid winner-selection procedure had outermean .38515 versus previous .47462;try fixed averaging to reduce model-selection variance before any new36test call.',
        'parent_common_test_decision': 'Keep parent144-candidate standalone experiment frozen as training-only;no Test36 predictions or threshold changes;not permanently excluded.',
        'groups': {'LocalMean8': '2norms x2orderedviews x top5/15;all balanced ridge C.03;patient-local K3.',
                   'CommonMean12': '2norms x K2/3/4 x top5/15;Combined panel;all balanced ridge C.03;fold-fitted common quantitative centers.',
                   'EqualBlend': '50%LocalMean8 +50%CommonMean12;each group averages its constituent fitted probabilities.'},
        'selection': 'Only3 fixed ensemble recipes compared insideinner5;select by meanAUC-.10SD-.30max(meanFit-meanAUC-.10,0),ties fewer members then group order.',
        'member_selection': 'No top-by-test or top-by-inner model filtering. Every constituent fixed in advance;feature ranking remains fitted on each current training fold.',
        'cached_inner_predictions': 'Reuse immutable predictions of exact prespecified constituent configs on exact same innerfolds;these were produced with fold-fitted features/centers/preprocessing. Refit all selected members forouter andfinal.',
        'pure_habitat': True, 'clinical': False, 'whole_tumor_predictors': False, 'Dstar': False,
        'no_test_fits': True, 'threshold': 'Selected final recipe complete inner5 OOFYouden;freeze before test.',
        'test': 'One selected frozen ensemble on36;constituent calls counted separately;no separate recipe ormember performance ontest.',
        'prior_test_exposure': True, 'external_validation': False, 'formal_use': False, 'automatic_adoption': False,
        'source_hashes': {str(p.relative_to(ROOT)): digest(p) for p in sources}, 'protected_hashes': protected}
    save(OUT/'protocol.json', method)
    save(OUT/'definition_frozen.json', {'passed': True, 'protocol_sha256': digest(OUT/'protocol.json'), 'test_scored': False})
    csv('prespecified_members', [{'MemberID': key, 'Source': m['source'], **m['config']} for key, m in members().items()])
    save(OUT/'parent_common_training_only_decision.json', {'passed': True, 'reason': 'Poor training-only nested performance;preserve as training-only diagnosis.',
        'parent': str(common.OUT.relative_to(ROOT)), 'parent_outermean': float(common.read('training_choices').iloc[0].OuterMeanAUC),
        'parent_test_calls': 0, 'based_on_new_test_results': False})
    print('Defined3 mean-ensemble recipes;20 prespecified LR components;one final36 ensemble call', flush=True)


def protect():
    p = json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    for field in ('protected_hashes', 'source_hashes'):
        for path, sha in p[field].items():
            assert digest(ROOT/path) == sha, path
    assert digest(OUT/'protocol.json') == json.loads((OUT/'definition_frozen.json').read_text(encoding='utf-8'))['protocol_sha256']
    assert not (common.OUT/'test_started.json').exists()
    return len(p['protected_hashes'])


def checkpoint(phase, source):
    return joblib.load((local.OUT/'phases'/('Fixed118_'+phase+'.joblib')) if source == 'Local' else (common.OUT/'phases'/(phase+'.joblib')))


def tune(phase, y):
    checkpoints = {source: checkpoint(phase, source) for source in ('Local', 'Common')}
    cv = {label: [] for label in GROUPS}; audit = []
    for num, (a, b) in enumerate(common.phase_folds(phase), 1):
        scores_a, scores_b = {}, {}
        for key, m in members().items():
            cp = checkpoints[m['source']]
            row = cp['cv'][m['config']['candidate_id']][num-1]
            np.testing.assert_array_equal(row['train'], a); np.testing.assert_array_equal(row['validation'], b)
            scores_a[key] = row['train_probability' if m['source'] == 'Local' else 'fit_score']
            scores_b[key] = row['validation_probability' if m['source'] == 'Local' else 'validation_score']
            assert next(r for r in cp['grid'] if r['candidate_id'] == m['config']['candidate_id'])['AvailableAllFolds']
        for label in GROUPS:
            pa, pb = average(scores_a, label), average(scores_b, label)
            cv[label].append({'train': a, 'validation': b, 'fit_score': pa, 'validation_score': pb})
            audit.append({'Group': label, 'InnerFold': num, 'FitAUC': float(roc_auc_score(y[a], pa)),
                'ValidationAUC': float(roc_auc_score(y[b], pb))})
    frame = pd.DataFrame(audit)
    grid = []
    for label in GROUPS:
        rr = frame[frame.Group == label]; mu, sd, fm = rr.ValidationAUC.mean(), rr.ValidationAUC.std(ddof=1), rr.FitAUC.mean()
        grid.append({'Group': label, 'MeanValidationAUC': float(mu), 'SD': float(sd), 'MeanFitAUC': float(fm),
            'MemberCount': len(ids(label)), 'Objective': float(mu-.1*sd-.3*max(fm-mu-.1, 0))})
    return {'phase': phase, 'grid': grid, 'audit': audit, 'cv': cv}


def choose(grid):
    return max(grid, key=lambda r: (r['Objective'], -r['MemberCount'], -GROUPS.index(r['Group'])))


def local_data(stage):
    d = joblib.load(local.OUT/('training_features.joblib' if stage == 'train' else 'all118_features.joblib'))
    return {key: table for key, table in d.items() if key.endswith('__OrderedCompact') or key.endswith('__FilteredFull')}


def refit(pp, ll, a, b, y, required):
    fitted, partitions, tables, full_local, full_common = {}, {}, {}, {}, {}
    for key in required:
        m = members()[key]; c = m['config']
        if m['source'] == 'Local':
            image_key = c['image_key']
            if image_key not in full_local:
                full_local[image_key] = local.prepare(ll, a, b, y, {**c, 'top': 0})
            p = local.prepare(ll, a, b, y, c, full=full_local[image_key])
            f = local.fitted(p, y[a], c, a)
        else:
            identity = (c['normalization'], c['K'])
            if identity not in partitions:
                partitions[identity] = common.feature.fit_partitions(pp, a, *identity)
                tables[identity] = common.feature.tables(pp, partitions[identity], np.r_[a, b])
                full_common[identity] = common.prepared(tables[identity], a, b, y)
            p = local.stable.prefix(full_common[identity], c['top'])
            f = common.fitted(p, y[a], c)
        fitted[key] = {'source': m['source'], 'config': c, 'fit': f}
    return {'members': fitted, 'partitions': partitions, 'tables': tables}


def scored_fit(result, group, which):
    return average({key: m['fit'][('evaluation_probability' if m['source'] == 'Local' else 'validation_probability') if which == 'validation' else 'train_probability']
                    for key, m in result['members'].items()}, group)


def train():
    protect()
    assert not (OUT/'training_frozen.json').exists() and not (OUT/'test_started.json').exists()
    cohort, y, tr, te = common.inputs('Fixed118')
    pp, ll = common.patients(tr), local_data('train')
    phases = {phase: tune(phase, y) for phase in ['outer%d' % k for k in range(1, 6)]+['final']}
    joblib.dump(phases, OUT/'inner_ensemble_recipes.joblib', compress=3)
    csv('inner_ensemble_scores', [{'Phase': phase, **r} for phase, cp in phases.items() for r in cp['grid']])
    outer, metrics, predictions = [], [], []
    for num, (a, b) in enumerate(local.study.folds('Fixed118', 'outer'), 1):
        selected = choose(phases['outer%d' % num]['grid'])['Group']
        fitted = refit(pp, ll, a, b, y, list(members()))
        for label in ('Adaptive',)+GROUPS:
            group = selected if label == 'Adaptive' else label
            pa, pb = scored_fit(fitted, group, 'train'), scored_fit(fitted, group, 'validation')
            metrics.append({'ModelKey': label, 'Fold': num, 'Recipe': group, 'MemberCount': len(ids(group)),
                'FitAUC': float(roc_auc_score(y[a], pa)), 'AUC': float(roc_auc_score(y[b], pb))})
            predictions.extend({'ModelKey': label, 'Fold': num, 'Recipe': group, 'AnalysisID': cohort.AnalysisID.iloc[i],
                'Index': int(i), 'Outcome': int(y[i]), 'Probability': float(s)} for i, s in zip(b, pb))
        outer.append({'train': a, 'validation': b, 'selected_recipe': selected, 'refit': fitted})
        print('Pure ensemble outer%d refitted20 prespecified members;selected%s' % (num, selected), flush=True)
    selected = choose(phases['final']['grid'])
    group = selected['Group']
    final = refit(pp, ll, tr, [], y, ids(group))
    pa = scored_fit(final, group, 'train')
    oof = np.full(138, np.nan)
    for r in phases['final']['cv'][group]:
        oof[r['validation']] = r['validation_score']
    assert np.isfinite(oof[tr]).all() and np.isnan(oof[te]).all()
    fp, tp, th = roc_curve(y[tr], oof[tr], drop_intermediate=False)
    threshold = float(th[np.argmax(tp-fp)])
    assert np.isfinite(threshold)
    mm = pd.DataFrame(metrics); mm = mm[mm.ModelKey == 'Adaptive']
    pr = pd.DataFrame(predictions); pr = pr[pr.ModelKey == 'Adaptive']
    choice = {'Recipe': group, 'MemberCount': len(ids(group)), 'TrainFitAUC': float(roc_auc_score(y[tr], pa)),
        'OuterMeanAUC': float(mm.AUC.mean()), 'OuterSD': float(mm.AUC.std(ddof=1)),
        'PooledOOFAUC': float(roc_auc_score(pr.Outcome, pr.Probability)), 'FinalInnerAUC': selected['MeanValidationAUC'],
        'SelectionObjective': selected['Objective'], 'Threshold': threshold}
    result = {'train': tr, 'outer': outer, 'final': final, 'recipe': group, 'training_oof': oof,
        'threshold': threshold, 'train_probability': pa, 'choice': choice}
    joblib.dump(result, OUT/'models/pure_habitat_ensemble.joblib', compress=3)
    csv('training_choices', [choice]); csv('outer_metrics', metrics); csv('nested_predictions', predictions)
    features = []
    for key, member in final['members'].items():
        f = member['fit']; c = member['config']
        for j, name in enumerate(f['names']):
            features.append({'MemberID': key, 'Source': member['source'], 'Normalization': c['normalization'],
                'K': c['K'], 'Feature': name, 'Modality': name.split('__')[0], 'MemberWeight': weights(group)[key],
                'MemberCoefficient': float(f['model'].coef_[0, j])})
    csv('selected_features_by_member', features)
    files = [p for p in OUT.rglob('*') if p.is_file() and p.name not in ('training_frozen.json',)]
    save(OUT/'training_frozen.json', {'passed': True, 'test_scored': False, 'test_ensemble_models': 1,
        'test_constituent_models': len(ids(group)), 'protocol_sha256': digest(OUT/'protocol.json'),
        'files': {str(p.relative_to(OUT)): digest(p) for p in files}})
    print('Frozen pure habitat ensemble:'+json.dumps(choice), flush=True)


def frozen():
    protect()
    f = json.loads((OUT/'training_frozen.json').read_text(encoding='utf-8'))
    assert f['passed'] and digest(OUT/'protocol.json') == f['protocol_sha256']
    for path, sha in f['files'].items():
        assert digest(OUT/path) == sha, path
    return f


def predict(result, ll, cc, indices):
    scores = {}
    for key, member in result['final']['members'].items():
        f, c = member['fit'], member['config']
        scores[key] = local.predict(f, ll, indices) if member['source'] == 'Local' else common.predict(f, cc[c['normalization'], c['K']], indices)
    return average(scores, result['recipe'])


def test():
    frozen()
    assert not (OUT/'test_started.json').exists()
    assert json.loads((OUT/'training_verification.json').read_text(encoding='utf-8'))['passed']
    result = joblib.load(OUT/'models/pure_habitat_ensemble.joblib')
    cohort, _, tr, te = common.inputs('Fixed118')
    ll = local_data('test'); cc = {}
    if result['final']['partitions']:
        import run_random120_retained_methods_20261010 as loader
        profiles = loader.load_profiles(); pp = {}
        for i in te:
            pp[int(i)] = common.feature.compact_patient(profiles[int(i)], int(i))
            joblib.dump(pp[int(i)], OUT/'test_patient_features'/('case_%03d.joblib' % i), compress=3)
        cc = {key: common.feature.tables(pp, part, te) for key, part in result['final']['partitions'].items()}
    joblib.dump(cc, OUT/'test_common_tables.joblib', compress=3)
    save(OUT/'test_started.json', {'ensemble_prediction_calls': 1, 'constituent_prediction_calls': len(result['final']['members']),
        'freeze_sha256': digest(OUT/'training_frozen.json')})
    p = predict(result, ll, cc, te)
    yt = cohort.Outcome.to_numpy(int)[te]
    from verify_ivim_df63_20261020 import draws
    lo, hi = np.quantile(draws(yt, {'Ensemble': p}, 2026102385)['Ensemble'], [.025, .975])
    tn, fp, fn, tp = confusion_matrix(yt, p >= result['threshold'], labels=[0, 1]).ravel()
    metrics = {'N': len(te), 'Positive': int(yt.sum()), 'Negative': int((1-yt).sum()),
        'AUC': float(roc_auc_score(yt, p)), 'AUC_CI95Low': float(lo), 'AUC_CI95High': float(hi),
        'Sensitivity': tp/(tp+fn), 'Specificity': tn/(tn+fp), 'Accuracy': (tp+tn)/len(yt),
        'TP': int(tp), 'FN': int(fn), 'TN': int(tn), 'FP': int(fp), 'Threshold': result['threshold']}
    csv('test_metrics', [metrics])
    csv('test_predictions', [{'AnalysisID': cohort.AnalysisID.iloc[i], 'Index': int(i), 'Outcome': int(t),
        'Probability': float(s), 'Prediction': int(s >= result['threshold'])} for i, t, s in zip(te, yt, p)])
    save(OUT/'test_complete.json', {'passed': True, 'ensemble_prediction_calls': 1,
        'constituent_prediction_calls': len(result['final']['members']), 'post_test_fits': 0, 'post_test_threshold_changes': 0,
        'parent_common_standalone_test_calls': 0, 'metrics': metrics})
    print('Frozen ensemble test:'+json.dumps(metrics), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=('define', 'train', 'test'))
    with threadpool_limits(1):
        globals()[parser.parse_args().action]()
