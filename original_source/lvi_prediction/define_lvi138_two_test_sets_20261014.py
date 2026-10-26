# -*- coding: utf-8 -*-
"""Prespecified 77/33/28 internal split; no model fit, predictions or AUC selection."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'lvi_prediction/splits/lvi138_train77_testA33_testB28_20261014'
CURRENT = ROOT/'lvi_prediction/CURRENT_LVI_DEVELOPMENT_SPLIT_20261014.json'
COHORT = ROOT/'fold_locked_lvi_study/isotropic1mm_zscore_firstorder_v1/slic_138_holdout_70_30_seed42/cohort_138.csv'
HISTORY = ROOT/'lvi_prediction/results/lvi_regional_summary_frozen_random50_138_20261014'
OLD90 = ROOT/'lvi_prediction/results/lvi_k3_stability_improvement90_20261013'
SEED_B, SEED_70_30 = 20261014, 2026101403
GROUPS = ('Train77', 'TestA33', 'TestB28')


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            sha.update(chunk)
    return sha.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def cohort():
    data = pd.read_csv(COHORT, dtype={'SubjectID': str, 'PatientID': str})
    assert len(data) == 138 and data.AnalysisID.nunique() == 138
    assert data.SubjectID.notna().all() and data.SubjectID.nunique() == 138
    assert data.PatientID.notna().all() and data.PatientID.nunique() == 138
    assert set(data.Outcome) == {0, 1} and data.Outcome.sum() == 94
    return data


def partition(data):
    y = data.Outcome.to_numpy(int)
    development, b = train_test_split(np.arange(len(data)), train_size=110, test_size=28,
                                     stratify=y, random_state=SEED_B)
    a_train, a_test = train_test_split(development, train_size=77, test_size=33,
                                     stratify=y[development], random_state=SEED_70_30)
    return {name: np.sort(ix) for name, ix in zip(GROUPS, (a_train, a_test, b))}


def counts(data, indices):
    y = data.Outcome.to_numpy(int)[indices]
    return {'N': len(indices), 'Positive': int(y.sum()), 'Negative': int((y == 0).sum())}


def cv_folds(data, indices, seed):
    y, result = data.Outcome.to_numpy(int), []
    for fold, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(indices)), y[indices]), 1):
        train, validation = indices[a], indices[b]
        result.append({'Fold': fold, 'train_indices': train.tolist(), 'validation_indices': validation.tolist(),
            'train_ids': data.AnalysisID.iloc[train].tolist(), 'validation_ids': data.AnalysisID.iloc[validation].tolist(),
            'train_counts': counts(data, train), 'validation_counts': counts(data, validation)})
    return result


def nested(data, train):
    outer = cv_folds(data, train, 42)
    for row in outer:
        seed = 62200+row['Fold']
        row['inner_seed'] = seed
        row['inner_folds'] = cv_folds(data, np.asarray(row['train_indices'], int), seed)
    return {'outer_seed': 42, 'outer_folds': outer, 'final_tuning_seed': 62212,
            'final_tuning_folds': cv_folds(data, train, 62212), 'all_training_cases': train.tolist(),
            'test_case_ids_in_CV': [], 'actual_models_trained': 0}


def protect():
    p = json.loads((OUT/'split_protocol.json').read_text(encoding='utf-8'))
    for field in ('protected_hashes', 'source_hashes'):
        for path, sha in p[field].items():
            assert digest(ROOT/path) == sha, path
    return len(p['protected_hashes'])


def define():
    assert not OUT.exists(), 'Never replace this split or search another seed'
    assert not CURRENT.exists(), 'Do not replace an active split without recording its history'
    data = cohort()
    previous = json.loads((HISTORY/'protocol.json').read_text(encoding='utf-8'))
    protected = dict(previous['protected_hashes'])
    protected.update(previous['source_hashes'])
    protected.update({str(p.relative_to(ROOT)): digest(p) for p in HISTORY.rglob('*') if p.is_file()})
    protected.update(json.loads((HISTORY/'delivery_manifest.json').read_text(encoding='utf-8'))['source_hashes'])
    protected[str(COHORT.relative_to(ROOT))] = digest(COHORT)
    if (ROOT/'ACTIVE_COHORT.json').exists():
        protected['ACTIVE_COHORT.json'] = digest(ROOT/'ACTIVE_COHORT.json')
    parts = partition(data)
    labels = np.full(138, '', dtype=object)
    for group, ix in parts.items():
        labels[ix] = group
    old = json.loads((OLD90/'split_manifest.json').read_text(encoding='utf-8'))
    oldtrain = set(old['train_ids'])
    rows = data.copy()
    rows.insert(0, 'CohortIndex', np.arange(138))
    rows['Group'] = labels
    rows['Previous90Group'] = np.where(rows.AnalysisID.isin(oldtrain), 'PriorTraining90', 'PriorTest48')
    rows['PreviousPoolExploration'] = True
    OUT.mkdir(parents=True)
    rows.to_csv(OUT/'all138_group_assignment.csv', index=False, encoding='utf-8-sig')
    for group, ix in parts.items():
        rows.iloc[ix].to_csv(OUT/(group+'_cases.csv'), index=False, encoding='utf-8-sig')
    summary = pd.DataFrame([{'Group': group, **counts(data, ix)} for group, ix in parts.items()])
    summary.to_csv(OUT/'group_counts.csv', index=False, encoding='utf-8-sig')
    exposure = pd.crosstab(rows.Group, rows.Previous90Group).reindex(GROUPS)
    exposure.to_csv(OUT/'previous90_exposure_counts.csv', encoding='utf-8-sig')
    manifest = {'date': '2026-10-14', 'split_name': 'Train77_TestA33_TestB28', 'source_cohort_sha256': digest(COHORT),
        'seed_reserve_B': SEED_B, 'seed_train_A': SEED_70_30, 'outcome_stratification': True,
        'stage1': '138 ->110 development +28 TestB', 'stage2': '110 ->77 training +33 TestA, exactly70/30',
        'distinct_split_solutions_created': 1, 'seeds_searched_by_AUC': False, 'counts': summary.to_dict('records'),
        'groups': {g: {'indices': ix.tolist(), 'ids': data.AnalysisID.iloc[ix].tolist()} for g, ix in parts.items()},
        'all138_id_order': data.AnalysisID.tolist(), 'training_to_testA_ratio': '77:33 =7:3',
        'training_to_all_tests_ratio': '77:61; 7:3 refers to110 development cases, not all138',
        'same_hospital_same_cohort': True, 'external_validation': False, 'all_test_cases_previously_explored': True}
    save(OUT/'split_manifest.json', manifest)
    save(OUT/'nested5_training_folds.json', nested(data, parts['Train77']))
    save(OUT/'cohort_source_confirmation.json', {'date': '2026-10-14', 'source': 'direct human answer in this chat',
        'answer': '同一家医院、同一队列', 'scope': 'current138 pool', 'external_dataset_available': False,
        'clarified_intent': 'a second internal test set to check whether results depend on one evaluation sample'})
    save(OUT/'split_protocol.json', {'date': '2026-10-14', 'authorization': 'human requests7:3 train/test plusa second test set; clarified reliability check, samehospital samecohort',
        'status': 'dataset roles fixed; no new model fitted or evaluated', 'partition': manifest,
        'stratification_uses_outcome_only_for_balance': True, 'features_probabilities_AUC_not_used_for_partitioning': True,
        'model_development_allowed_ids': manifest['groups']['Train77']['ids'],
        'reserved_internal_testA_ids': manifest['groups']['TestA33']['ids'],
        'reserved_internal_testB_ids': manifest['groups']['TestB28']['ids'],
        'default_methods': 'continue pure regional-summary habitat comparison: PercentileZ and RobustZ,k3,T2/ADC/IVIM D/f,noDstar; method selection remains train-only',
        'validation': 'train77 outer5/inner5; fit all outcome-driven choices, quality gates, table preprocessing and learned transforms in each currenttraining fold',
        'final': 'select pipeline in train77 final5-fold, refit on77, threshold only trainOOF, freeze all parameters',
        'tests': 'both TestA33 and TestB28 scored in same joint final stage per frozen model; no adjust afterA beforeB',
        'reporting': 'reportA,B andcombined61 AUC+95%CI; combined61 is overall internal holdout assessment; A/B evaluate sample-composition variation',
        'A_minus_B': 'bootstrap difference using separate stratified patient resampling in the two nonoverlapping test groups; descriptive comparison without treating groups as matchedcases',
        'no_choose_highest_test_set': True, 'no_future_redraw_by_test_AUC': True,
        'reuse_old90_frozen_model_for_independent_new_test': False, 'new77_training_required_before_future_scoring': True,
        'historical_pool_used_repeatedly': True, 'formal_model_adoption_changed': False,
        'existing_model_adoption_decisions_preserved': True,
        'sources': ['https://www.bmj.com/content/384/bmj-2023-074819', 'https://www.bmj.com/content/384/bmj-2023-074821'],
        'protected_hashes': protected, 'source_hashes': {str(Path(__file__).resolve().relative_to(ROOT)): digest(__file__),
            'lvi_prediction/verify_lvi138_two_test_sets_20261014.py': digest(ROOT/'lvi_prediction/verify_lvi138_two_test_sets_20261014.py')},
        'versions': {'numpy': np.__version__, 'scikit_learn': sklearn.__version__}})
    save(OUT/'definition_frozen.json', {'passed': True, 'protocol_sha256': digest(OUT/'split_protocol.json'),
        'frozen_files': {p.name: digest(p) for p in OUT.iterdir() if p.is_file() and p.name != 'definition_frozen.json'},
        'models_trained': 0, 'model_predict_calls': 0, 'AUC_computed': 0, 'protected_count': protect()})
    print(summary.to_string(index=False), flush=True)
    print(exposure.to_string(), flush=True)


def frozen():
    fr = json.loads((OUT/'definition_frozen.json').read_text(encoding='utf-8'))
    protect()
    for name, sha in fr['frozen_files'].items():
        assert digest(OUT/name) == sha, name
    return fr


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['define'], required=True)
    p.parse_args()
    define()
