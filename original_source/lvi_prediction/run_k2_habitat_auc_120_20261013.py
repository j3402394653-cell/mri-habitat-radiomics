"""Controlled K2 addition to the frozen flexible-feature K3/4/5 experiment."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import json
from pathlib import Path
import shutil
import time

import joblib
import numpy as np
import pandas as pd
import radiomics
import SimpleITK as sitk
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
import lvi_paths
import run_k345_flexible_features_120_20261013 as prior

ROOT = lvi_paths.PROJECT_ROOT
OUT = ROOT / 'lvi_prediction/results/lvi_k2_habitat_auc_120_20261013'
KS, NORMALIZATIONS = (2,), ('PercentileZ', 'RobustZ')
KINDS = ('base', 'erode', 'dilate')


def isolated(name, source):
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Separate module namespaces retain the verified numerical implementation,
# without changing any original source file or its live module constants.
feature = isolated('k2_feature_implementation', ROOT / 'lvi_prediction/k345_habitat_features_20261013.py')
feature.KS = KS
feature.PROTOCOL = {**feature.PROTOCOL, 'K_values': [2],
                    'scope': 'fixedK2 pure habitats; unchanged signal design and feature definitions'}
core = isolated('k2_training_implementation', ROOT / 'lvi_prediction/run_k345_flexible_features_120_20261013.py')
core.KS, core.OUT, core.feature = KS, OUT, feature
old, origin, previous = core.old, core.origin, core.previous
save, digest, key = core.save, core.digest, core.key
options, config, choose, fit, tune = core.options, core.config, core.choose, core.fit, core.tune
ANALYSES = core.ANALYSES


def selections(grid):
    result = {key(n, 2, a): choose(grid, a, n, 2) for n in NORMALIZATIONS for a in ANALYSES}
    result['JointSelection__FusionFree'] = choose(grid, 'FusionFree')
    return result


core.selections = selections


def protect():
    p = json.loads((OUT / 'protocol.json').read_text(encoding='utf-8'))
    for name, h in p['protected_hashes'].items():
        assert digest(ROOT / name) == h, name
    for name, h in p['source_hashes'].items():
        assert digest(ROOT / name) == h, name
    return len(p['protected_hashes'])


def freeze_protocol():
    if OUT.exists():
        raise FileExistsError('Do not overwrite a completed or frozen experiment')
    prior.frozen()
    assert json.loads((prior.OUT / 'inference_verification.json').read_text())['passed']
    policy_path = ROOT / 'lvi_prediction/FUTURE_HABITAT_EXPERIMENT_POLICY_20261013.json'
    policy = json.loads(policy_path.read_text(encoding='utf-8'))
    assert policy['normalization_routes'] == list(NORMALIZATIONS)
    assert policy['after_freeze_scoring']['score_all54_after_freeze']
    assert not policy['reuse36_predictions_in54_report']
    cohort, manifest, y, tr, te = origin.inputs()
    alltest = np.r_[te, np.asarray(manifest['omitted_indices'], int)]
    assert policy['train_ids'] == cohort.AnalysisID.iloc[tr].tolist()
    assert policy['evaluation54_ids'] == cohort.AnalysisID.iloc[alltest].tolist()
    protected = dict(json.loads((prior.OUT / 'protocol.json').read_text(encoding='utf-8'))['protected_hashes'])
    for directory in [prior.OUT, ROOT / 'lvi_prediction/results/lvi_bootstrap_ch_k_train84_20261013']:
        protected.update({str(p.relative_to(ROOT)): digest(p) for p in directory.rglob('*') if p.is_file()})
    for p in [policy_path, policy_path.with_suffix('.md'), ROOT / 'AGENTS.md',
              ROOT / 'lvi_prediction/CURRENT_LVI_EXPLORATORY_FREEZE_20261013.json']:
        protected[str(p.relative_to(ROOT))] = digest(p)
    sources = [Path(__file__).resolve(), ROOT / 'lvi_prediction/test_k2_habitat_auc_20261013.py',
               ROOT / 'lvi_prediction/verify_k2_habitat_auc_20261013.py',
               ROOT / 'lvi_prediction/k345_habitat_features_20261013.py',
               ROOT / 'lvi_prediction/run_k345_flexible_features_120_20261013.py']
    OUT.mkdir(parents=True)
    (OUT / 'models').mkdir()
    save(OUT / 'protocol.json', {
        'date': '2026-10-13', 'authorization': 'human requested K2 AUC experiment and comparison with existing K3/4/5',
        'normalizations': NORMALIZATIONS, 'K_values': [2], 'analyses': ANALYSES,
        'same120_84_36_split': True, 'sample_seed': manifest['sample_seed'], 'split_seed': manifest['split_seed'],
        'nested': 'outer5 seed42; inner5 seeds62201..62205; final inner5 seed62212',
        'candidates': options(), 'candidate_count': len(options()),
        'feature_protocol': feature.PROTOCOL, 'candidate_feature_count': 822,
        'model_candidates': 'Descriptor/Texture/Combined;3/6/9/12/15/all train-gated decorrelated features; unchanged L2 LR C.03/.3/1',
        'selection': 'fixedK2; features/budget/LR parameters chosen in inner5 per norm; extra Fusion reference chooses normalization inside inner5',
        'objective': 'unchanged mean inner AUC-.10*SD-.30*max(mean fit-mean validation-.10,0)-.03*abs(mean fit-.75)',
        'fold_only_transforms': 'ICC>=.8, completeness>=.9, bootstrap sign stability>=.8, Spearman<.8; train median fill/1-99winsorization/meanSD',
        'initial_test': 'train verification then score36 once; no choices changed by test scores',
        'post120': 'copy exact parameter snapshot then score all54 together anew per each of8 distinct frozen models; one normalization-selected reference aliases its current scores',
        'no_separate_postfreeze36_or18_model_predict': True,
        'comparison': 'same cases/folds/feature budget/objective/threshold rule as verified saved K3/4/5; existing comparator probabilities are read for paired statistics without refit',
        'CH_scope': 'K2 is a fixed new candidate, suggested by training-only CH study; this is not outer validation of the CH selector',
        'new_full_K2_3_4_5_adaptive_pipeline': False,
        'Dstar_excluded': True, 'IVIM_channels': ['D', 'f'],
        'formal_model_use': False, 'automatic_adoption': False,
        'current_adopted_and_frozen_models_replaced': False,
        'test_previously_reused': True, 'new_external_validation': False,
        'protected_hashes': protected,
        'source_hashes': {str(p.relative_to(ROOT)): digest(p) for p in sources},
        'sources': ['https://scikit-learn.org/1.3/modules/generated/sklearn.linear_model.LogisticRegression.html',
                    'https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html']})
    save(OUT / 'split_manifest.json', manifest)
    return cohort, manifest, y, tr, te


def extract(indices, cohort, stage):
    start = time.time()
    profiles = old.common.old.retained.load_profiles()
    radiomics.logger.setLevel(40)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    results = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(feature.extract_one, (int(i), cohort.AnalysisID.iloc[i], profiles[int(i)], KS)): int(i) for i in indices}
        for done, f in enumerate(as_completed(futures), 1):
            i = futures[f]
            results[i] = f.result()
            if done % 4 == 0 or done == len(indices):
                print('K2 %s extraction %d/%d %.0fs' % (stage, done, len(indices), time.time()-start), flush=True)
    metadata = results[int(indices[0])]['metadata'][2]
    names = [m['name'] for m in metadata]
    assert len(names) == len(set(names)) == 822
    assert all(results[int(i)]['metadata'][2] == metadata for i in indices)
    data = {n: {2: {'metadata': metadata, 'indices': np.asarray(indices, int),
            **{kind: np.asarray([[results[int(i)]['values'][n][2][kind][name] for name in names] for i in indices], float) for kind in KINDS}}} for n in NORMALIZATIONS}
    for name in ['partition_audit', 'texture_audit', 'normalization_audit', 'contour_partition_audit']:
        pd.DataFrame([r for i in indices for r in results[int(i)][name]]).to_csv(OUT / (stage+'_'+name+'.csv'), index=False, encoding='utf-8-sig')
    joblib.dump(data, OUT / (stage+'_features.joblib'), compress=3)
    return data


def train():
    start = time.time()
    cohort, manifest, y, tr, te = freeze_protocol()
    data = extract(tr, cohort, 'training')
    predictions, metrics, grids, frequency = [], [], [], []
    for fold, (a, b) in enumerate(old.common.old.previous.folds(tr, y, 42, 5), 1):
        grid, _ = tune(data, old.common.old.previous.folds(a, y, 62200+fold, 5), y, 'K2_outer'+str(fold))
        chosen = selections(grid)
        stored, cache = {'train': a, 'evaluation': b, 'grid': grid, 'models': {}}, {}
        for model_key, row in chosen.items():
            c = config(row)
            if c['candidate_id'] not in cache:
                cache[c['candidate_id']] = fit(data, a, b, y, c)
            f = cache[c['candidate_id']]
            stored['models'][model_key] = f
            metrics.append({'ModelKey': model_key, 'Fold': fold, **c, 'FeatureCount': len(f['indices']), 'AUC': float(roc_auc_score(y[b], f['evaluation_probability']))})
            predictions.extend({'ModelKey': model_key, 'Fold': fold, 'AnalysisID': cohort.AnalysisID.iloc[i], 'Outcome': int(y[i]), 'Probability': float(p)} for i, p in zip(b, f['evaluation_probability']))
            frequency.extend({'ModelKey': model_key, 'Fold': fold, 'Normalization': c['normalization'], 'K': 2, 'Feature': name} for name in f['names'])
        joblib.dump(stored, OUT / ('models/outer_fold_'+str(fold)+'.joblib'), compress=3)
        grids.extend({'OuterFold': fold, **r} for r in grid)
        print('K2 outer'+str(fold)+'/5 complete', flush=True)
    grid, cv = tune(data, old.common.old.previous.folds(tr, y, 62212, 5), y, 'K2_final')
    chosen, cache = selections(grid), {}
    final = {'models': {}, 'train': tr, 'metadata_by_K': {2: data[NORMALIZATIONS[0]][2]['metadata']}, 'formal_model_use': False}
    choices, features, trainpred, oofpred, gates = [], [], [], [], []
    predictions, folds = pd.DataFrame(predictions), pd.DataFrame(metrics)
    for model_key, row in chosen.items():
        c = config(row)
        if c['candidate_id'] not in cache:
            f = fit(data, tr, [], y, c)
            oof = np.full(138, np.nan)
            for ix, p in cv[c['candidate_id']]:
                oof[ix] = p
            assert np.isfinite(oof[tr]).all() and np.isnan(oof[te]).all()
            f['threshold'] = old.common.threshold(y[tr], oof[tr])
            f['train'] = tr
            cache[c['candidate_id']] = f, oof
        f, oof = cache[c['candidate_id']]
        final['models'][model_key] = f
        meta, ff = final['metadata_by_K'][2], folds[folds.ModelKey == model_key].AUC
        pp = predictions[predictions.ModelKey == model_key].set_index('AnalysisID').loc[cohort.AnalysisID.iloc[tr]]
        choices.append({'ModelKey': model_key, 'KChoice': 'FixedK_NormAdaptive' if model_key.startswith('JointSelection') else 'Fixed', **c,
                        'FeatureCount': len(f['indices']), 'TrainFitAUC': float(roc_auc_score(y[tr], f['train_probability'])),
                        'FinalSelectionCVMeanAUC': row['MeanValidationAUC'], 'SelectionObjective': row['Objective'],
                        'OuterMeanAUC': float(ff.mean()), 'OuterSD': float(ff.std(ddof=1)), 'PooledOOFAUC': float(roc_auc_score(y[tr], pp.Probability)),
                        'Threshold': f['threshold'], **{mod+'Count': sum(meta[j]['modality'] == mod for j in f['indices']) for mod in feature.MODALITIES}})
        features.extend({'ModelKey': model_key, 'Normalization': c['normalization'], 'K': 2, 'Analysis': c['analysis'], 'Feature': name,
                         'Modality': meta[j]['modality'], 'Block': meta[j]['block'], 'LRStandardizedCoefficient': float(beta)} for name, j, beta in zip(f['names'], f['indices'], f['model'].coef_[0] if f['model'] is not None else []))
        trainpred.extend({'ModelKey': model_key, 'AnalysisID': cohort.AnalysisID.iloc[i], 'Outcome': int(y[i]), 'Probability': float(p)} for i, p in zip(tr, f['train_probability']))
        oofpred.extend({'ModelKey': model_key, 'AnalysisID': cohort.AnalysisID.iloc[i], 'Outcome': int(y[i]), 'Probability': float(oof[i])} for i in tr)
    for n in NORMALIZATIONS:
        _, _, audit = old.ranking(data[n][2], tr, y)
        gates.extend({'Normalization': n, 'K': 2, 'Modality': m['modality'], 'Block': m['block'], **r} for m, r in zip(data[n][2]['metadata'], audit))
    joblib.dump(final, OUT / 'models/k2_models.joblib', compress=3)
    for name, rows in [('nested_predictions', predictions), ('outer_fold_metrics', folds), ('inner_candidate_scores', grids),
                       ('final_training_grid', grid), ('final_training_choices', choices), ('selected_features', features),
                       ('train_fit_predictions', trainpred), ('final_selection_oof_predictions', oofpred), ('feature_gate', gates), ('outer_feature_selections', frequency)]:
        pd.DataFrame(rows).to_csv(OUT / (name+'.csv'), index=False, encoding='utf-8-sig')
    recommendation = next(r for r in choices if r['ModelKey'] == 'JointSelection__FusionFree')
    save(OUT / 'training_only_recommendation.json', {'overall_fusion': recommendation, 'selection': 'fixedK2, normalization/features/parameters chosen inside inner5 and evaluated by outer5', 'test_used': False, 'formal_model_use': False})
    save(OUT / 'training_frozen.json', {'passed': True, 'model_sha256': digest(OUT/'models/k2_models.joblib'),
         'protocol_sha256': digest(OUT/'protocol.json'), 'feature_sha256': digest(OUT/'training_features.joblib'),
         'training_csv_hashes': {p.name: digest(p) for p in OUT.glob('*.csv')}, 'test_scored': False,
         'protected_count': protect(), 'elapsed_seconds': time.time()-start})
    print(pd.DataFrame(choices)[['ModelKey', 'TrainFitAUC', 'OuterMeanAUC', 'FeatureCount']].to_string(index=False), flush=True)


def frozen():
    p = json.loads((OUT/'training_frozen.json').read_text(encoding='utf-8'))
    protect()
    for name, field in [('models/k2_models.joblib', 'model_sha256'), ('protocol.json', 'protocol_sha256'), ('training_features.joblib', 'feature_sha256')]:
        assert digest(OUT/name) == p[field]
    for name, h in p['training_csv_hashes'].items():
        assert digest(OUT/name) == h, name
    return p


def write_evaluation(final, data, cohort, indices, stage, groups):
    scores, calls = core.probabilities(final, data)
    outcomes = cohort.Outcome.to_numpy(int)[indices]
    rows, metrics, pairs = [], [], []
    for name, p in scores.items():
        threshold = final['models'][name]['threshold']
        rows.extend({'ModelKey': name, 'AnalysisID': cohort.AnalysisID.iloc[i], 'Group': 'PreviousTest36' if j < 36 else 'PreviouslyOmitted18',
                     'Outcome': int(y), 'Probability': float(pp), 'Threshold': threshold, 'PredictedPositive': int(pp >= threshold)} for j, (i, y, pp) in enumerate(zip(indices, outcomes, p)))
    for group, pos in groups.items():
        y, ps = outcomes[pos], {name: p[pos] for name, p in scores.items()}
        draws = old.common.old.retained.bootstrap(y, ps)
        for name, p in ps.items():
            f = final['models'][name]
            lo, hi = np.quantile(draws[name], [.025, .975])
            metrics.append({'Group': group, 'ModelKey': name, 'Normalization': f['config']['normalization'], 'K': 2, 'Analysis': f['config']['analysis'],
                            'N': len(y), 'Positive': int(y.sum()), 'Negative': int((y == 0).sum()), **old.common.old.metrics(y, p, f['threshold']),
                            'AUC_CI95Low': float(lo), 'AUC_CI95High': float(hi)})
        for a in ANALYSES:
            first, second = key('PercentileZ', 2, a), key('RobustZ', 2, a)
            lo, hi = np.quantile(draws[first]-draws[second], [.025, .975])
            pairs.append({'Group': group, 'Normalization': 'PercentileZ minus RobustZ', 'Analysis': a, 'K': 2, 'ReferenceK': 2,
                          'DeltaAUC': float(roc_auc_score(y, ps[first])-roc_auc_score(y, ps[second])), 'CI95Low': float(lo), 'CI95High': float(hi)})
    for name, r in [('predictions', rows), ('metrics', metrics), ('paired_auc_comparisons', pairs)]:
        pd.DataFrame(r).to_csv(OUT / (stage+'_'+name+'.csv'), index=False, encoding='utf-8-sig')
    return scores, calls


def test120():
    if (OUT/'test120_started.json').exists():
        raise FileExistsError('Do not repeat initial36 inference')
    f = frozen()
    assert json.loads((OUT/'training_verification.json').read_text())['passed']
    cohort, manifest, y, tr, te = origin.inputs()
    data = extract(te, cohort, 'test120')
    final = joblib.load(OUT/'models/k2_models.joblib')
    save(OUT/'test120_started.json', {'joint_count': 1, 'n': 36, 'distinct_models': 8, 'reference_models': 1, 'model_sha256': f['model_sha256']})
    _, calls = write_evaluation(final, data, cohort, te, 'test120', {'PreviousTest36': np.arange(36)})
    save(OUT/'test120_complete.json', {'passed': True, 'n': 36, 'joint_count': 1, 'distinct_predict_calls': calls,
         'prediction_rows': 9*36, 'model_sha256': f['model_sha256'], 'protected_count': protect()})
    print('K2 initial36 evaluation complete', flush=True)


def freeze54():
    if (OUT/'frozen54_started.json').exists():
        raise FileExistsError('Do not repeat frozen54 inference')
    f = frozen()
    assert json.loads((OUT/'test120_complete.json').read_text())['passed']
    cohort, manifest, y, tr, te = origin.inputs()
    omitted = np.asarray(manifest['omitted_indices'], int)
    alltest = np.r_[te, omitted]
    assert not set(tr) & set(alltest)
    snapshot = OUT/'models/post120_frozen_models.joblib'
    shutil.copyfile(OUT/'models/k2_models.joblib', snapshot)
    assert digest(snapshot) == f['model_sha256']
    final = joblib.load(snapshot)
    np.testing.assert_array_equal(final['train'], tr)
    save(OUT/'post120_freeze_record.json', {'model_sha256': digest(snapshot), 'same_as_training_snapshot': True, 'no_refit_after36': True,
         'before_frozen54_scoring': True, 'training_ids': manifest['train_ids'], 'evaluation_ids': cohort.AnalysisID.iloc[alltest].tolist(),
         'thresholds': {name: float(model['threshold']) for name, model in final['models'].items()},
         'model_specifications': {name: {'config': model['config'], 'features': model['names'], 'coefficients': model['model'].coef_[0].tolist() if model['model'] is not None else [],
             'intercept': float(model['model'].intercept_[0]) if model['model'] is not None else None} for name, model in final['models'].items()}})
    new = extract(omitted, cohort, 'frozen54_new18')
    initial = joblib.load(OUT/'test120_features.joblib')
    data = {n: {2: {'indices': alltest, 'metadata': final['metadata_by_K'][2],
            **{kind: np.vstack([initial[n][2][kind], new[n][2][kind]]) for kind in KINDS}}} for n in NORMALIZATIONS}
    for n in NORMALIZATIONS:
        assert initial[n][2]['metadata'] == new[n][2]['metadata'] == final['metadata_by_K'][2]
    joblib.dump(data, OUT/'frozen54_features.joblib', compress=3)
    for name in ['partition_audit', 'texture_audit', 'normalization_audit', 'contour_partition_audit']:
        pd.concat([pd.read_csv(OUT/('test120_'+name+'.csv')), pd.read_csv(OUT/('frozen54_new18_'+name+'.csv'))], ignore_index=True).to_csv(OUT/('frozen54_'+name+'.csv'), index=False, encoding='utf-8-sig')
    for name in ['membership54', 'historical_exposure54']:
        shutil.copyfile(prior.OUT/(name+'.csv'), OUT/(name+'.csv'))
    save(OUT/'frozen54_started.json', {'joint_count': 1, 'n': 54, 'direct_all54_new_predictions': True, 'reuse_initial36_probabilities': False,
         'separate36_prediction_calls': 0, 'separate18_prediction_calls': 0, 'training_overlap': 0, 'model_sha256': digest(snapshot)})
    scores, calls = write_evaluation(final, data, cohort, alltest, 'frozen54', previous.previous54.GROUPS)
    initial_pred = pd.read_csv(OUT/'test120_predictions.csv', float_precision='round_trip')
    maximum = 0.
    for name, p in scores.items():
        ref = initial_pred[initial_pred.ModelKey == name].set_index('AnalysisID').loc[cohort.AnalysisID.iloc[te]].Probability.to_numpy()
        maximum = max(maximum, float(abs(p[:36]-ref).max()))
        np.testing.assert_allclose(p[:36], ref, atol=1e-12, rtol=0)
    assert digest(snapshot) == f['model_sha256']
    save(OUT/'frozen54_complete.json', {'passed': True, 'joint_count': 1, 'n': 54, 'direct_all54_new_predictions': True,
         'distinct_predict_calls': calls, 'reference_models_reuse_current54_scores': 1, 'old36_probabilities_not_used_to_build54': True,
         'initial36_vs_new54_subset_max_error': maximum, 'training_overlap': 0, 'models_refitted': 0, 'thresholds_changed': False,
         'prediction_rows': 9*54, 'model_sha256': f['model_sha256'], 'protected_count': protect()})
    print('K2 frozen direct54 evaluation complete', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['train', 'test120', 'freeze54'], required=True)
    args = p.parse_args()
    with threadpool_limits(limits=1):
        {'train': train, 'test120': test120, 'freeze54': freeze54}[args.phase]()
