# -*- coding: utf-8 -*-
"""Two100-cohort63-input ablation; same LR family, nested selection andone teststage."""
import os
os.environ.update({'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'LOKY_MAX_CPU_COUNT': '8'})
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
import ivim_df63_study_20261020 as study
import ivim_df63_minroi_guard_20261020 as minroi
import run_k3_stability_improvement90_20261013 as stable
from run_regional_summary77_two_tests_20261014 import bootstrap

OUT, ROOT = study.OUT, study.ROOT
save, digest, read = study.save, study.digest, study.read
PATIENTS = None


def csv(name, rows):
    pd.DataFrame(rows).to_csv(OUT/(name+'.csv'), index=False, encoding='utf-8-sig')


def initialize(which='train'):
    global PATIENTS
    threadpool_limits(1)
    indices = study.union_indices(which)
    PATIENTS = {int(i): minroi.canonical(joblib.load(study.patient_path(i))) for i in indices}


def data_for(indices, combo):
    meta = [{k: v for k, v in m.items() if k not in ('source_indices', 'statistic')} for m in study.feature.aggregation()]
    return {(norm, 'Invariant'): {'indices': np.asarray(indices, int), 'metadata': meta,
        **{kind: np.asarray([PATIENTS[int(i)]['values'][(norm, combo)][kind] for i in indices]) for kind, _ in study.feature.KINDS}} for norm in study.NORMS}


def config(row):
    return {k: row[k] for k in study.options(row['normalization'])[0]}


def best(grid, norm=None):
    choices = [r for r in grid if norm is None or r['normalization'] == norm]
    return max(choices, key=lambda r: (bool(r['AvailableAllFolds']), r['Objective'], -r.get('DimensionCount', 0),
        -r['MeanFeatureCount'], -r['C'], -r['candidate_id'], -int(r.get('ComboID', 'C00')[1:])))


def tune(data, folds, y, phase):
    opts = [c for norm in study.NORMS for c in study.options(norm)]
    records, cv = [], {(c['normalization'], c['candidate_id']): [] for c in opts}
    for fold, (a, b) in enumerate(folds, 1):
        assert set(y[a]) == set(y[b]) == {0, 1}
        ranks = {norm: stable.ranking(data[(norm, 'Invariant')], a, y) for norm in study.NORMS}
        full, preps, estimates = {}, {}, {}
        for c in opts:
            n, ranking, top = c['normalization'], c['ranking'], c['top']
            key = (n, ranking)
            if key not in full:
                full[key] = stable.prepare(data[(n, 'Invariant')], a, b, y, {**c, 'top': 0}, ranks[n])
            identity = key+(top,)
            if identity not in preps:
                preps[identity] = stable.prefix(full[key], top)
            p = preps[identity]
            estimator_key = (n, tuple(p['indices']), c['C'])
            if estimator_key not in estimates:
                estimates[estimator_key] = stable.fitted(p, y[a], c, 66413+int(np.sum(a)))
            f = estimates[estimator_key]
            records.append({'Phase': phase, 'InnerFold': fold, 'normalization': n, 'candidate_id': c['candidate_id'],
                'ValidationAUC': float(roc_auc_score(y[b], f['evaluation_probability'])),
                'FitAUC': float(roc_auc_score(y[a], f['train_probability'])), 'FeatureCount': len(f['indices']),
                'Available': bool(f['available'])})
            cv[(n, c['candidate_id'])].append({'train': a, 'validation': b,
                'train_probability': f['train_probability'], 'validation_probability': f['evaluation_probability']})
    audit = pd.DataFrame(records)
    grid = []
    for c in opts:
        rows = audit[(audit.normalization == c['normalization'])&(audit.candidate_id == c['candidate_id'])]
        mean, sd, fit = rows.ValidationAUC.mean(), rows.ValidationAUC.std(ddof=1), rows.FitAUC.mean()
        available = bool(rows.Available.all())
        grid.append({**c, 'MeanValidationAUC': float(mean), 'SD': float(sd), 'MeanFitAUC': float(fit),
            'MeanFeatureCount': float(rows.FeatureCount.mean()), 'AvailableAllFolds': available,
            'Objective': float(mean-.10*sd-.30*max(fit-mean-.10, 0)) if available else -1.})
    return {'grid': grid, 'audit': records, 'cv': cv}


def task_path(group, combo):
    return OUT/'training'/group/(combo+'.joblib')


def train_one(group, combo):
    dest = task_path(group, combo)
    if dest.exists():
        stored = joblib.load(dest)
        assert stored['group'] == group and stored['combo'] == combo
        return group, combo, 'checkpoint', digest(dest)
    start = time.time()
    cohort, y, tr, te = study.inputs(group)
    data = data_for(tr, combo)
    outer, metrics, predictions, phases = [], [], [], {}
    for fold, (a, b) in enumerate(study.folds(group, 'outer'), 1):
        phase = tune(data, study.folds(group, 'inner', fold), y, 'outer'+str(fold))
        models = {}
        for norm in study.NORMS:
            c = config(best(phase['grid'], norm))
            f = stable.fit(data, a, b, y, c)
            models[norm] = f
            metrics.append({'Normalization': norm, 'Fold': fold, **c, 'FeatureCount': len(f['indices']),
                'AUC': float(roc_auc_score(y[b], f['evaluation_probability'])), 'Available': bool(f['available'])})
            predictions.extend({'Normalization': norm, 'Fold': fold, 'AnalysisID': cohort.AnalysisID.iloc[i],
                'Outcome': int(y[i]), 'Probability': float(p)} for i, p in zip(b, f['evaluation_probability']))
        outer.append({'train': a, 'validation': b, 'models': models})
        phases['outer'+str(fold)] = phase
    phase = tune(data, study.folds(group, 'final'), y, 'final')
    phases['final'] = phase
    final, choices, selected = {}, [], []
    for norm in study.NORMS:
        row = best(phase['grid'], norm)
        c = config(row)
        f = stable.fit(data, tr, [], y, c)
        oof = np.full(138, np.nan)
        for cv in phase['cv'][(norm, c['candidate_id'])]:
            oof[cv['validation']] = cv['validation_probability']
        assert np.isfinite(oof[tr]).all() and np.isnan(oof[te]).all()
        f['threshold'] = stable.old.common.threshold(y[tr], oof[tr])
        final[norm] = f
        ff = [r['AUC'] for r in metrics if r['Normalization'] == norm]
        pp = pd.DataFrame([r for r in predictions if r['Normalization'] == norm]).set_index('AnalysisID').loc[cohort.AnalysisID.iloc[tr]]
        choices.append({'Normalization': norm, **c, 'FeatureCount': len(f['indices']), 'AvailableFinal': bool(f['available']),
            'AvailableAllInnerFolds': row['AvailableAllFolds'], 'TrainFitAUC': float(roc_auc_score(y[tr], f['train_probability'])),
            'OuterMeanAUC': float(np.mean(ff)), 'OuterSD': float(np.std(ff, ddof=1)),
            'PooledOOFAUC': float(roc_auc_score(y[tr], pp.Probability)), 'FinalInnerAUC': row['MeanValidationAUC'],
            'SelectionObjective': row['Objective'], 'Threshold': f['threshold']})
        beta = f['model'].coef_[0] if f['model'] is not None else []
        selected.extend({'Normalization': norm, 'Feature': name, 'Coefficient': float(b)} for name, b in zip(f['names'], beta))
    stored = {'group': group, 'combo': combo, 'train': tr, 'outer': outer, 'phases': phases,
        'final': final, 'choices': choices, 'outer_metrics': metrics, 'nested_predictions': predictions,
        'selected_features': selected, 'elapsed_seconds': time.time()-start}
    temp = dest.with_suffix('.pending')
    joblib.dump(stored, temp, compress=3);temp.replace(dest)
    return group, combo, '%.1fs'%stored['elapsed_seconds'], digest(dest)


def freeze_training_protocol():
    study.protect()
    assert json.loads((OUT/'extraction_train_complete.json').read_text(encoding='utf-8'))['passed']
    assert not (OUT/'test_started.json').exists()
    if (OUT/'training_protocol.json').exists():
        protect_training_sources()
        return
    for group in study.GROUPS:
        (OUT/'training'/group).mkdir(parents=True)
    sources = [Path(__file__).resolve(), ROOT/'lvi_prediction/verify_ivim_df63_20261020.py',
        ROOT/'lvi_prediction/ivim_df63_minroi_guard_20261020.py']
    save(OUT/'training_protocol.json', {'date': '2026-10-20', 'definition_protocol_sha256': digest(OUT/'protocol.json'),
        'definition_sha256': digest(OUT/'definition_frozen.json'), 'candidate_count_each_norm': 24,
        'source_hashes': {str(p.relative_to(ROOT)): digest(p) for p in sources},
        'training_feature_hashes': json.loads((OUT/'extraction_train_complete.json').read_text(encoding='utf-8'))['hashes'],
        'constant_fallback': 'if noeligible predictor:prevalence-only,AUC.5,unavailableflag;no caseorcombination silentlydropped',
        'minROI_boundary': 'canonicalreadview makesexact12-voxel habitattexture missing,matching fullPyRadiomics minimumROISize12;rawcaches preserved;descriptorsunchanged',
        'adaptive_selection_ties': 'availability,objective,lowercluster dimensioncount,fewerinputs,smallerC,candidateID,comboID',
        'test_scored': False})


def protect_training_sources():
    protocol = json.loads((OUT/'training_protocol.json').read_text(encoding='utf-8'))
    assert digest(OUT/'protocol.json') == protocol['definition_protocol_sha256']
    for path, sha in protocol['source_hashes'].items():
        assert digest(ROOT/path) == sha, path
    for index, sha in protocol['training_feature_hashes'].items():
        assert digest(study.patient_path(int(index))) == sha


def adaptive():
    selections, nested, choices, metrics, aliases = [], [], [], [], []
    combos = study.feature.combinations63()
    for group in study.GROUPS:
        cohort, y, tr, _ = study.inputs(group)
        loaded = {c['ComboID']: joblib.load(task_path(group, c['ComboID'])) for c in combos}
        for phase_name in ['outer'+str(i) for i in range(1, 6)]+['final']:
            grid = [{**row, 'ComboID': c['ComboID'], 'DimensionCount': c['DimensionCount']}
                for c in combos for row in loaded[c['ComboID']]['phases'][phase_name]['grid']]
            for key, norm in [('Adaptive__'+n, n) for n in study.NORMS]+[('Adaptive__BothNorms', None)]:
                chosen = best(grid, norm)
                combo, actual_norm = chosen['ComboID'], chosen['normalization']
                stored = loaded[combo]
                selections.append({'Group': group, 'ModelKey': key, 'Phase': phase_name, **chosen})
                if phase_name != 'final':
                    fold = int(phase_name[-1])
                    outer = stored['outer'][fold-1]
                    f = outer['models'][actual_norm]
                    assert f['config'] == config(chosen)
                    metrics.append({'Group': group, 'ModelKey': key, 'Fold': fold, 'ComboID': combo,
                        'Normalization': actual_norm, 'AUC': float(roc_auc_score(y[outer['validation']], f['evaluation_probability']))})
                    nested.extend({'Group': group, 'ModelKey': key, 'Fold': fold, 'AnalysisID': cohort.AnalysisID.iloc[i],
                        'Outcome': int(y[i]), 'Probability': float(p)} for i, p in zip(outer['validation'], f['evaluation_probability']))
                else:
                    f = stored['final'][actual_norm]
                    assert f['config'] == config(chosen)
                    aliases.append({'Group': group, 'ModelKey': key, 'ComboID': combo, 'Normalization': actual_norm})
    nestedframe, metricframe = pd.DataFrame(nested), pd.DataFrame(metrics)
    for a in aliases:
        stored = joblib.load(task_path(a['Group'], a['ComboID']))
        c = next(r for r in stored['choices'] if r['Normalization'] == a['Normalization'])
        ff = metricframe[(metricframe.Group == a['Group'])&(metricframe.ModelKey == a['ModelKey'])].AUC
        pp = nestedframe[(nestedframe.Group == a['Group'])&(nestedframe.ModelKey == a['ModelKey'])]
        choices.append({**c, **a, 'OuterMeanAUC': float(ff.mean()), 'OuterSD': float(ff.std(ddof=1)),
            'PooledOOFAUC': float(roc_auc_score(pp.Outcome, pp.Probability))})
    for name, rows in [('adaptive_inner_selections', selections), ('adaptive_nested_predictions', nested),
        ('adaptive_outer_metrics', metrics), ('adaptive_training_choices', choices)]:
        csv(name, rows)
    save(OUT/'adaptive_frozen_aliases.json', aliases)


def summarize_training():
    choices, features, outer, nested, audit = [], [], [], [], []
    for group in study.GROUPS:
        for c in study.feature.combinations63():
            stored = joblib.load(task_path(group, c['ComboID']))
            prefix = {'Group': group, 'ComboID': c['ComboID'], 'DimensionCount': c['DimensionCount'], 'Dimensions': '+'.join(c['Dimensions'])}
            choices.extend({**prefix, **r} for r in stored['choices'])
            features.extend({**prefix, **r} for r in stored['selected_features'])
            outer.extend({**prefix, **r} for r in stored['outer_metrics'])
            nested.extend({**prefix, **r} for r in stored['nested_predictions'])
            audit.extend({**prefix, 'Phase': phase, **r} for phase, p in stored['phases'].items() for r in p['grid'])
    for name, rows in [('fixed_training_choices', choices), ('selected_features', features), ('fixed_outer_metrics', outer),
        ('fixed_nested_predictions', nested), ('candidate_grids', audit)]:
        csv(name, rows)
    adaptive()


def train():
    freeze_training_protocol()
    if (OUT/'training_frozen.json').exists():
        frozen()
        return
    start = time.time()
    tasks = [(g, c['ComboID']) for g in study.GROUPS for c in study.feature.combinations63()]
    with ProcessPoolExecutor(max_workers=study.WORKERS, initializer=initialize) as pool:
        futures = [pool.submit(train_one, g, c) for g, c in tasks]
        for count, future in enumerate(as_completed(futures), 1):
            group, combo, status, _ = future.result()
            print('Training %d/126 %s %s %s elapsed%.0fs'%(count, group, combo, status, time.time()-start), flush=True)
    summarize_training()
    files = [p for p in (OUT/'training').rglob('*.joblib')]+list(OUT.glob('*.csv'))+[OUT/'adaptive_frozen_aliases.json']
    save(OUT/'training_frozen.json', {'passed': True, 'fixed_models': 252, 'adaptive_aliases': 6,
        'files': {str(p.relative_to(OUT)): digest(p) for p in files},
        'training_protocol_sha256': digest(OUT/'training_protocol.json'), 'test_scored': False,
        'elapsed_seconds': time.time()-start, 'protected_count': study.protect()})
    print('All252 fixed models and6 adaptivealiases frozen beforetests', flush=True)


def frozen():
    study.protect();protect_training_sources()
    fr = json.loads((OUT/'training_frozen.json').read_text(encoding='utf-8'))
    assert digest(OUT/'training_protocol.json') == fr['training_protocol_sha256']
    for path, sha in fr['files'].items():
        assert digest(OUT/path) == sha, path
    return fr


def test():
    frozen()
    assert json.loads((OUT/'training_verification.json').read_text(encoding='utf-8'))['passed']
    assert json.loads((OUT/'extraction_test_complete.json').read_text(encoding='utf-8'))['passed']
    assert not (OUT/'test_started.json').exists(), 'Do not repeat model holdout scoring'
    save(OUT/'test_started.json', {'all252_models_verified_andfrozen': True, 'stage': 1, 'freeze_sha256': digest(OUT/'training_frozen.json')})
    initialize('sample')
    predictions, metrics, pairs = [], [], []
    aliases = json.loads((OUT/'adaptive_frozen_aliases.json').read_text(encoding='utf-8'))
    calls = 0
    for offset, group in enumerate(study.GROUPS):
        cohort, _, tr, te = study.inputs(group)
        truth = cohort.Outcome.to_numpy(int)[te]
        scores, thresholds = {}, {}
        for combo in study.feature.combinations63():
            cid = combo['ComboID']
            data = data_for(te, cid)
            stored = joblib.load(task_path(group, cid))
            for norm, f in stored['final'].items():
                key = norm+'__'+cid
                scores[key] = stable.predict(f, data[(norm, 'Invariant')])
                thresholds[key] = f['threshold']
                calls += 1
        for a in aliases:
            if a['Group'] == group:
                original = a['Normalization']+'__'+a['ComboID']
                scores[a['ModelKey']] = scores[original]
                thresholds[a['ModelKey']] = thresholds[original]
        draws = bootstrap(truth, scores, 2026102031+offset)
        for key, p in scores.items():
            threshold = thresholds[key]
            low, high = np.quantile(draws[key], [.025, .975])
            metrics.append({'Group': group, 'ModelKey': key, 'N': 30, 'Positive': int(truth.sum()), 'Negative': int((truth == 0).sum()),
                **stable.old.common.old.metrics(truth, p, threshold), 'AUC_CI95Low': float(low), 'AUC_CI95High': float(high)})
            predictions.extend({'Group': group, 'ModelKey': key, 'AnalysisID': cohort.AnalysisID.iloc[i], 'Outcome': int(y),
                'Probability': float(pp), 'Threshold': threshold, 'PredictedPositive': int(pp >= threshold)} for i, y, pp in zip(te, truth, p))
        # Paired comparisons to allsix within each normalization; no independent-group difference claim.
        for norm in study.NORMS:
            reference = norm+'__C63'
            for combo in study.feature.combinations63():
                key = norm+'__'+combo['ComboID']
                low, high = np.quantile(draws[key]-draws[reference], [.025, .975])
                pairs.append({'Group': group, 'Normalization': norm, 'ComboID': combo['ComboID'],
                    'DeltaVsAll6': float(roc_auc_score(truth, scores[key])-roc_auc_score(truth, scores[reference])),
                    'CI95Low': float(low), 'CI95High': float(high)})
        print(group+' frozen129 scores complete', flush=True)
    for name, rows in [('test_metrics', metrics), ('test_predictions', predictions), ('paired_combo_vs_all6_auc', pairs)]:
        csv(name, rows)
    frozen()
    save(OUT/'test_complete.json', {'passed': True, 'fixed_model_predict_calls': calls, 'adaptive_additional_calls': 0,
        'prediction_rows': len(predictions), 'metric_rows': len(metrics), 'scoring_stages': 1,
        'model_refits_after_freeze': 0, 'threshold_changes': 0, 'protected_count': study.protect()})
    print('Completed one frozen teststage:252 fixed models,6 aliases,7740 predictions', flush=True)


def pipeline():
    print('Pipeline waits for all training feature checkpoints before fitting', flush=True)
    while not (OUT/'extraction_train_complete.json').exists():
        time.sleep(30)
    train()
    import verify_ivim_df63_20261020 as verification
    verification.training()
    import extract_ivim_df63_20261020 as extraction
    extraction.extract('test')
    test()
    verification.inference()
    import report_ivim_df63_20261020 as reporting
    reporting.main()
    print('Pipeline complete; rendered figures await visual inspection', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['pipeline', 'train', 'verify-training', 'test', 'verify-test'], required=True)
    args = parser.parse_args()
    with threadpool_limits(1):
        if args.phase.startswith('verify'):
            import verify_ivim_df63_20261020 as verification
            {'verify-training': verification.training, 'verify-test': verification.inference}[args.phase]()
        else:
            {'pipeline': pipeline, 'train': train, 'test': test}[args.phase]()
