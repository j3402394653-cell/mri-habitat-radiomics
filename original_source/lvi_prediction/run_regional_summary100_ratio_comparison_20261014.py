# -*- coding: utf-8 -*-
"""Fixed70/30 versus80/20 in100 selected cases controls; both trainings finish before holdout scoring."""
import argparse
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
import run_regional_summary77_two_tests_20261014 as previous
import define_lvi100_nested_ratios_20261014 as split
import run_regional_summary_ratio_comparison_20261014 as history

ROOT = previous.ROOT
OUT = ROOT/'lvi_prediction/results/lvi_regional_summary100_70_30_vs80_20_20261014'
RATIOS = ('70_30', '80_20')
NORMS = previous.NORMS
save, digest, old = previous.save, previous.digest, previous.old
read, options, matched, selections = previous.read, previous.options, previous.matched, previous.selections
features, fit, tune, bootstrap = previous.features, previous.fit, previous.tune, previous.bootstrap
BOOTSTRAP_SEEDS = {'70_30': 2026101421, '80_20': 2026101422, 'Common20': 2026101423}


def directory(ratio):
    assert ratio in RATIOS
    return OUT/ratio


def inputs(ratio):
    cohort = split.source.cohort()
    selection = json.loads((split.OUT/'sampling_manifest.json').read_text(encoding='utf-8'))
    spec = selection['schemes'][ratio]
    source = ROOT/spec['directory']
    assert digest(source/'split_manifest.json') == spec['manifest_sha256']
    manifest = json.loads((source/'split_manifest.json').read_text(encoding='utf-8'))
    names = ('Train70', 'Test30') if ratio == '70_30' else ('Train80', 'Test20')
    tr, te = [np.asarray(manifest['groups'][g]['indices'], int) for g in names]
    assert len(tr) == (70 if ratio == '70_30' else 80)
    assert len(te) == (30 if ratio == '70_30' else 20)
    assert not set(tr)&set(te) and set(tr)|set(te) == set(selection['selected_indices'])
    y = np.full(138, -1, int)
    y[tr] = cohort.Outcome.to_numpy(int)[tr]
    return cohort, y, tr, te, source

def folds(ratio, which, outer=None):
    source = inputs(ratio)[4]
    saved = json.loads((source/'nested5_training_folds.json').read_text(encoding='utf-8'))
    rows = saved['outer_folds'] if which == 'outer' else saved['final_tuning_folds'] if which == 'final' else saved['outer_folds'][outer-1]['inner_folds']
    ff = [(np.asarray(r['train_indices'], int), np.asarray(r['validation_indices'], int)) for r in rows]
    tr, te = inputs(ratio)[2:4]
    allowed = set(tr) if which != 'inner' else set(folds(ratio, 'outer')[outer-1][0])
    for a, b in ff:
        assert not set(a)&set(b) and set(a)|set(b) == allowed and not (set(a)|set(b))&set(te)
    assert len(ff) == 5
    return ff


def protect():
    protocol = json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    for field in ('protected_hashes', 'source_hashes'):
        for name, sha in protocol[field].items():
            assert digest(ROOT/name) == sha, name
    return len(protocol['protected_hashes'])


def freeze_protocol():
    if OUT.exists():
        protect()
        assert not (OUT/'test_started.json').exists()
        return
    split.frozen()
    history.frozen()
    assert json.loads((split.OUT/'split_verification.json').read_text(encoding='utf-8'))['passed']
    inherited = json.loads((split.OUT/'protocol.json').read_text(encoding='utf-8'))
    protected = {**inherited['protected_hashes'], **inherited['source_hashes']}
    protected.update({str(p.relative_to(ROOT)): digest(p) for p in split.OUT.rglob('*') if p.is_file()})
    sources = [Path(__file__).resolve(), ROOT/'lvi_prediction/verify_regional_summary100_ratio_comparison_20261014.py']
    OUT.mkdir(parents=True)
    for ratio in RATIOS:
        (directory(ratio)/'models').mkdir(parents=True)
    save(OUT/'protocol.json', {'date': '2026-10-14', 'authorization': '把总数据下降到100例试试',
        'sample_definition': json.loads((split.OUT/'sampling_manifest.json').read_text(encoding='utf-8')),
        'feature_definition': 'unchanged patient-local k3; invariant330 regional summaries; T2,ADC,IVIM D/f',
        'normalizations': list(NORMS), 'clinical': False, 'Dstar': False, 'whole_tumor_direct_features': False,
        'common_centers': False, 'representation_search': False, 'fixed_view': 'Invariant',
        'matched_arm': matched(), 'retuned_arm': 'same160 Invariant candidates; training-only inner5 selection',
        'candidates': options(), 'objective': 'inner meanAUC-.10SD-.30max(meanfit-meanAUC-.10,0)',
        'training': 'separate70 and80 masked labels; saved outer5/inner5/final5; all preprocessing andfeature choices inside currenttraining fold',
        'preprocessing': 'unchanged ICC>=.8 completeness>=.9,20bootstrapAssociation orICCStability,Spearman<.95,medianfill,P2.5/P97.5,median/IQR scale',
        'freeze': 'both trainings verified and frozen before any holdout score',
        'thresholds': 'each finalconfiguration training OOF Youden, frozen beforetest',
        'test': 'one own30/20 holdout batch per distinct configuration;common20 derived from same scores',
        'unused38_scored': False, 'extra10_not_independent_for80_20': True,
        'adaptive': 'training-chosen norm alias; no additional prediction call',
        'bootstrap': '5000 outcome-stratified patient draws; same draws for pairedmodels',
        'bootstrap_seeds': BOOTSTRAP_SEEDS,
        'historical_comparison': 'subset saved138-model ownholdout probabilities to exactsame new30/20 andcommon20; no historicalmodelcalls',
        'historical_training_nesting': '70 subset97;80 subset110;newtest cases outside historicaltraining',
        'test_seen_historically': True, 'external_validation': False, 'formal_model_use': False, 'automatic_adoption': False,
        'protected_hashes': {p.replace(chr(92), '/'): h for p, h in protected.items()},
        'source_hashes': {str(p.relative_to(ROOT)): digest(p) for p in sources}})

def train_ratio(ratio):
    dest = directory(ratio)
    if (dest/'training_frozen.json').exists():
        frozen_ratio(ratio)
        print(ratio+' training already frozen; skip', flush=True)
        return
    started = time.time()
    cohort, y, tr, te, _ = inputs(ratio)
    data = features(tr)
    joblib.dump(data, dest/'training_features.joblib', compress=3)
    predictions, metrics, grids, audit, frequency = [], [], [], [], []
    for fold, (a, b) in enumerate(folds(ratio, 'outer'), 1):
        path = dest/('models/outer_fold_'+str(fold)+'.joblib')
        if path.exists():
            stored = joblib.load(path)
            np.testing.assert_array_equal(stored['train'], a)
            np.testing.assert_array_equal(stored['evaluation'], b)
        else:
            grid, _, inner = tune(data, folds(ratio, 'inner', fold), y, ratio+'_outer'+str(fold))
            models, cache = {}, {}
            for name, c in selections(grid).items():
                if c['candidate_id'] not in cache:
                    cache[c['candidate_id']] = fit(data, a, b, y, c)
                models[name] = cache[c['candidate_id']]
            stored = {'train': a, 'evaluation': b, 'grid': grid, 'audit': inner, 'models': models}
            joblib.dump(stored, path, compress=3)
        grids.extend({'Phase': ratio+'_outer'+str(fold), **r} for r in stored['grid'])
        audit.extend(stored['audit'])
        for name, f in stored['models'].items():
            metrics.append({'ModelKey': name, 'Fold': fold, **f['config'], 'FeatureCount': len(f['indices']),
                            'EffectiveDimensions': f['effective_dimensions'], 'AUC': float(roc_auc_score(y[b], f['evaluation_probability']))})
            predictions.extend({'ModelKey': name, 'Fold': fold, 'AnalysisID': cohort.AnalysisID.iloc[i], 'Outcome': int(y[i]),
                                'Probability': float(p)} for i, p in zip(b, f['evaluation_probability']))
            frequency.extend({'ModelKey': name, 'Fold': fold, 'Feature': n} for n in f['names'])
        print(ratio+' outer'+str(fold)+'/5 complete', flush=True)
    grid, cv, inner = tune(data, folds(ratio, 'final'), y, ratio+'_final')
    audit.extend(inner)
    nested, outer = pd.DataFrame(predictions), pd.DataFrame(metrics)
    final = {'train': tr, 'models': {}, 'ratio': ratio, 'formal_model_use': False}
    choices, selected, trainpred, oofpred, cache = [], [], [], [], {}
    for name, c in selections(grid).items():
        cid = c['candidate_id']
        if cid not in cache:
            f = fit(data, tr, [], y, c)
            assert f['available'], 'Final pipeline must be nonconstant and converged'
            oof = np.full(138, np.nan)
            for ix, pp in cv[cid]:
                oof[ix] = pp
            assert np.isfinite(oof[tr]).all() and np.isnan(oof[te]).all()
            f['threshold'] = old.common.threshold(y[tr], oof[tr])
            cache[cid] = f, oof
        f, oof = cache[cid]
        final['models'][name] = f
        ff = outer[outer.ModelKey == name].AUC
        pp = nested[nested.ModelKey == name].set_index('AnalysisID').loc[cohort.AnalysisID.iloc[tr]]
        row = next(r for r in grid if r['candidate_id'] == cid)
        meta = data[(c['normalization'], 'Invariant')]['metadata']
        choices.append({'ModelKey': name, **c, 'FeatureCount': len(f['indices']), 'EffectiveDimensions': f['effective_dimensions'],
            'TrainFitAUC': float(roc_auc_score(y[tr], f['train_probability'])), 'OuterMeanAUC': float(ff.mean()), 'OuterSD': float(ff.std(ddof=1)),
            'PooledOOFAUC': float(roc_auc_score(y[tr], pp.Probability)), 'FinalSelectionCVMeanAUC': row['MeanValidationAUC'],
            'SelectionObjective': row['Objective'], 'Threshold': f['threshold'],
            **{m+'Count': sum(meta[j]['modality'] == m for j in f['indices']) for m in ('T2', 'ADC', 'IVIM')}})
        beta = f['model'].coefficients_.mean(0) if c['method'] == 'BagRidge' else f['model'].coef_[0]
        selected.extend({'ModelKey': name, 'Feature': feature, 'Modality': meta[j]['modality'], 'Block': meta[j]['block'],
                         'Coefficient': float(b)} for j, feature, b in zip(f['indices'], f['names'], beta))
        trainpred.extend({'ModelKey': name, 'AnalysisID': cohort.AnalysisID.iloc[i], 'Outcome': int(y[i]), 'Probability': float(p)} for i, p in zip(tr, f['train_probability']))
        oofpred.extend({'ModelKey': name, 'AnalysisID': cohort.AnalysisID.iloc[i], 'Outcome': int(y[i]), 'Probability': float(oof[i])} for i in tr)
    joblib.dump(final, dest/'models/final_models.joblib', compress=3)
    gates = []
    for (n, _), d in data.items():
        _, _, quality = previous.previous.ranking(d, tr, y)
        gates.extend({'Normalization': n, **row} for row in quality)
    for name, rows in [('nested_predictions', nested), ('outer_fold_metrics', outer), ('inner_candidate_scores', grids),
                       ('inner_fold_audit', audit), ('final_training_grid', grid), ('final_training_choices', choices),
                       ('selected_features', selected), ('train_fit_predictions', trainpred), ('final_selection_oof_predictions', oofpred),
                       ('feature_gate', gates), ('outer_feature_selections', frequency)]:
        pd.DataFrame(rows).to_csv(dest/(name+'.csv'), index=False, encoding='utf-8-sig')
    save(dest/'training_frozen.json', {'passed': True, 'ratio': ratio, 'model_sha256': digest(dest/'models/final_models.joblib'),
        'protocol_sha256': digest(OUT/'protocol.json'), 'training_features_sha256': digest(dest/'training_features.joblib'),
        'training_csv_hashes': {p.name: digest(p) for p in dest.glob('*.csv')}, 'train_N': len(tr), 'test_scored': False,
        'protected_count': protect(), 'elapsed_seconds_this_run': time.time()-started})
    print(ratio+' training complete', flush=True)
    print(pd.DataFrame(choices)[['ModelKey', 'method', 'FeatureCount', 'TrainFitAUC', 'OuterMeanAUC', 'OuterSD', 'PooledOOFAUC']].to_string(index=False), flush=True)


def frozen_ratio(ratio):
    dest = directory(ratio)
    fr = json.loads((dest/'training_frozen.json').read_text(encoding='utf-8'))
    for path, field in [('models/final_models.joblib', 'model_sha256'), ('training_features.joblib', 'training_features_sha256')]:
        assert digest(dest/path) == fr[field]
    assert digest(OUT/'protocol.json') == fr['protocol_sha256']
    for name, sha in fr['training_csv_hashes'].items():
        assert digest(dest/name) == sha
    return fr


def frozen():
    protect()
    fr = json.loads((OUT/'training_frozen.json').read_text(encoding='utf-8'))
    for ratio in RATIOS:
        frozen_ratio(ratio)
        assert digest(directory(ratio)/'training_frozen.json') == fr['ratio_freeze_hashes'][ratio]
    return fr


def train():
    freeze_protocol()
    for ratio in RATIOS:
        train_ratio(ratio)
    if not (OUT/'training_frozen.json').exists():
        save(OUT/'training_frozen.json', {'passed': True, 'all_trainings_finished_before_test': True,
            'ratio_freeze_hashes': {r: digest(directory(r)/'training_frozen.json') for r in RATIOS},
            'test_scored': False, 'protected_count': protect()})
    frozen()


def test():
    frozen()
    assert not (OUT/'test_started.json').exists(), 'Do not repeat holdout inference'
    assert json.loads((OUT/'training_verification.json').read_text())['passed']
    for r in RATIOS:
        assert json.loads((directory(r)/'training_verification.json').read_text())['passed']
    save(OUT/'test_started.json', {'both_ratio_trainings_frozen_and_verified': True,
        'frozen_sha256': digest(OUT/'training_frozen.json'), 'stage': 1})
    predictions, metrics, commonpred, commonps, pairedarms, calls, hashes = [], [], [], {}, [], {}, {}
    common = inputs('80_20')[3]
    for ratio in RATIOS:
        dest = directory(ratio)
        cohort, _, tr, te, _ = inputs(ratio)
        final = joblib.load(dest/'models/final_models.joblib')
        assert set(common) <= set(te)
        data = features(te)
        joblib.dump(data, dest/'test_features.joblib', compress=3)
        scores, cache = {}, {}
        for name, f in final['models'].items():
            cid = f['config']['candidate_id']
            if cid not in cache:
                cache[cid] = previous.previous.predict(f, data[(f['config']['normalization'], 'Invariant')])
            scores[name] = cache[cid]
        truth = cohort.Outcome.to_numpy(int)[te]
        mask = np.isin(te, common)
        np.testing.assert_array_equal(te[mask], common)
        for name, p in scores.items():
            threshold = final['models'][name]['threshold']
            rows = [{'Ratio': ratio, 'ModelKey': name, 'AnalysisID': cohort.AnalysisID.iloc[i], 'Outcome': int(y),
                'Probability': float(pp), 'Threshold': threshold, 'PredictedPositive': int(pp >= threshold)} for i, y, pp in zip(te, truth, p)]
            predictions.extend(rows)
            commonpred.extend(row for row, keep in zip(rows, mask) if keep)
            commonps[ratio+'::'+name] = p[mask]
        for group, selection, seed in [('OwnHoldout', np.ones(len(te), bool), BOOTSTRAP_SEEDS[ratio]), ('Common20', mask, BOOTSTRAP_SEEDS['Common20'])]:
            y, ps = truth[selection], {name: p[selection] for name, p in scores.items()}
            draws = bootstrap(y, ps, seed)
            for name, p in ps.items():
                low, high = np.quantile(draws[name], [.025, .975])
                metrics.append({'Ratio': ratio, 'ModelKey': name, 'Group': group, 'N': len(y), 'Positive': int(y.sum()), 'Negative': int((y == 0).sum()),
                    **old.common.old.metrics(y, p, final['models'][name]['threshold']), 'BrierScore': float(np.mean((y-p)**2)),
                    'AUC_CI95Low': float(low), 'AUC_CI95High': float(high)})
            for n in NORMS:
                first, second = 'Retuned__'+n, 'Matched__'+n
                low, high = np.quantile(draws[first]-draws[second], [.025, .975])
                pairedarms.append({'Ratio': ratio, 'Group': group, 'First': first, 'Second': second,
                    'DeltaAUC': float(roc_auc_score(y, ps[first])-roc_auc_score(y, ps[second])), 'CI95Low': float(low), 'CI95High': float(high)})
        calls[ratio] = len(cache)
        hashes[ratio] = digest(dest/'models/final_models.joblib')
    truth = inputs('80_20')[0].Outcome.to_numpy(int)[common]
    draws = bootstrap(truth, commonps, BOOTSTRAP_SEEDS['Common20'])
    paired = []
    for name in selections(read(directory('70_30'), 'final_training_grid').to_dict('records')):
        first, second = '80_20::'+name, '70_30::'+name
        low, high = np.quantile(draws[first]-draws[second], [.025, .975])
        paired.append({'ModelKey': name, 'Group': 'Common20', 'First': '80_20', 'Second': '70_30', 'N': 20,
            'DeltaAUC': float(roc_auc_score(truth, commonps[first])-roc_auc_score(truth, commonps[second])),
            'CI95Low': float(low), 'CI95High': float(high)})
    for name, rows in [('test_predictions', predictions), ('test_metrics', metrics), ('common20_predictions', commonpred),
                       ('paired_ratio_common20_auc', paired), ('retuned_minus_matched_auc', pairedarms)]:
        pd.DataFrame(rows).to_csv(OUT/(name+'.csv'), index=False, encoding='utf-8-sig')
    frozen()
    save(OUT/'test_complete.json', {'passed': True, 'prediction_rows': len(predictions), 'common20_rows': len(commonpred),
        'distinct_candidate_predict_calls_by_ratio': calls, 'one_holdout_batch_per_configuration': True,
        'common20_extra_predict_calls': 0, 'scoring_stages': 1, 'model_refits_after_freeze': 0, 'threshold_changes': 0,
        'model_sha256_after': hashes, 'protected_count': protect()})
    compare_with138()
    print(pd.DataFrame(metrics)[['Ratio', 'ModelKey', 'Group', 'N', 'AUC', 'AUC_CI95Low', 'AUC_CI95High']].to_string(index=False), flush=True)



def compare_with138():
    historical = read(history.OUT, 'test_predictions')
    current = read(OUT, 'test_predictions')
    selected_old, comparisons = [], []
    common = inputs('80_20')[3]
    for ratio in RATIOS:
        cohort, _, tr, te, _ = inputs(ratio)
        oldtr, oldte = history.inputs(ratio)[2:4]
        assert set(tr) <= set(oldtr) and set(te) <= set(oldte) and not set(te)&set(oldtr)
        truth = cohort.Outcome.to_numpy(int)[te]
        scores = {}
        for name in current[current.Ratio == ratio].ModelKey.unique():
            older = historical[(historical.Ratio == ratio)&(historical.ModelKey == name)].set_index('AnalysisID').loc[cohort.AnalysisID.iloc[te]]
            newer = current[(current.Ratio == ratio)&(current.ModelKey == name)].set_index('AnalysisID').loc[cohort.AnalysisID.iloc[te]]
            np.testing.assert_array_equal(older.Outcome, truth)
            scores['Old::'+name] = older.Probability.to_numpy()
            scores['New::'+name] = newer.Probability.to_numpy()
            selected_old.extend({'Ratio': ratio, 'ModelKey': name, 'AnalysisID': aid, 'Outcome': int(y), 'Probability': float(p)}
                for aid, y, p in zip(older.index, truth, older.Probability))
        for group, mask, seed in [('OwnHoldout', np.ones(len(te), bool), BOOTSTRAP_SEEDS[ratio]), ('Common20', np.isin(te, common), BOOTSTRAP_SEEDS['Common20'])]:
            y, pp = truth[mask], {name: p[mask] for name, p in scores.items()}
            draws = bootstrap(y, pp, seed)
            for name in current[current.Ratio == ratio].ModelKey.unique():
                newkey, oldkey = 'New::'+name, 'Old::'+name
                newer, older = [float(roc_auc_score(y, pp[k])) for k in (newkey, oldkey)]
                newci, oldci, diffci = [np.quantile(draws[k], [.025, .975]) for k in (newkey, oldkey,)] + [np.quantile(draws[newkey]-draws[oldkey], [.025, .975])]
                comparisons.append({'Ratio': ratio, 'ModelKey': name, 'Group': group, 'N': len(y),
                    'OldTrainingN': len(oldtr), 'NewTrainingN': len(tr), 'OldAUC': older, 'NewAUC': newer,
                    'OldCI95Low': float(oldci[0]), 'OldCI95High': float(oldci[1]),
                    'NewCI95Low': float(newci[0]), 'NewCI95High': float(newci[1]), 'DeltaNewMinus138': newer-older,
                    'DeltaCI95Low': float(diffci[0]), 'DeltaCI95High': float(diffci[1])})
    pd.DataFrame(selected_old).to_csv(OUT/'historical138_predictions_on_new_tests.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(comparisons).to_csv(OUT/'samecase100_vs138_comparison.csv', index=False, encoding='utf-8-sig')
    save(OUT/'historical_comparison_complete.json', {'passed': True, 'historical_model_predict_calls': 0,
        'comparison_rows': len(comparisons), 'historical_prediction_rows': len(selected_old),
        'no_test_case_in_either_historical_or_current_training': True})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['train', 'test'], required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        {'train': train, 'test': test}[args.phase]()
