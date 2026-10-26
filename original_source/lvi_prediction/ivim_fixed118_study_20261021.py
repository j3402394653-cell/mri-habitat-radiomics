# -*- coding: utf-8 -*-
"""Original fixed118 cohort and82/36 split; unchanged pure-IVIM five-family method."""
import json
from pathlib import Path
from types import SimpleNamespace
import joblib
import numpy as np
import pandas as pd
import ivim_five_k_study_20261021 as base

ROOT = base.ROOT
OUT = ROOT / 'lvi_prediction/results/lvi_ivim_fixed118_five_models_20261021'
GROUPS = ('Fixed118',)
FAMILIES, feature = base.FAMILIES, base.feature
save, digest, options, config = base.save, base.digest, base.options, base.config
CLINICAL, WHOLE = base.CLINICAL, base.WHOLE
HISTORICAL = ROOT / 'fold_locked_lvi_study/isotropic1mm_zscore_firstorder_v1/five_models_fixed118_seed42_20261008'
SELECTION = ROOT / 'fixed_118_seed42_cohort/selection_manifest.json'


def read(name):
    return pd.read_csv(OUT / (name + '.csv'), float_precision='round_trip', keep_default_na=False, na_values=[''])


def csv(name, rows):
    pd.DataFrame(rows).to_csv(OUT / (name + '.csv'), index=False, encoding='utf-8-sig')


def specifications():
    cohort = base.previous.source.cohort()
    selection = json.loads(SELECTION.read_text(encoding='utf-8'))
    original = json.loads((HISTORICAL / 'split_manifest.json').read_text(encoding='utf-8'))
    corrected = pd.read_csv(ROOT / 'corrected_118_cohort/cohort_118.csv', dtype={'SubjectID': str})
    assert corrected.AnalysisID.tolist() == cohort.AnalysisID.iloc[:118].tolist() == original['analysis_id_order']
    np.testing.assert_array_equal(corrected.Outcome, cohort.Outcome.iloc[:118])
    assert selection['train_ids'] == original['train_ids'] and selection['test_ids'] == original['test_ids']
    lookup = {aid: i for i, aid in enumerate(cohort.AnalysisID)}
    tr, te = [np.sort([lookup[aid] for aid in selection[key]]) for key in ('train_ids', 'test_ids')]
    assert len(tr) == 82 and len(te) == 36 and not set(tr) & set(te)
    assert set(tr) | set(te) == set(range(118))
    assert int(cohort.Outcome.iloc[tr].sum()) == 58 and int(cohort.Outcome.iloc[te].sum()) == 25
    return cohort, tr, te


def inputs(group):
    assert group == 'Fixed118'
    cohort, tr, te = specifications()
    y = np.full(138, -1, int); y[tr] = cohort.Outcome.to_numpy(int)[tr]
    return cohort, y, tr, te


def union_indices(which):
    _, tr, te = specifications()
    return tr if which == 'train' else te if which == 'test' else np.arange(118)


def prior_path(index):
    old = base.previous.patient_path(index)
    return old if old.exists() else OUT / 'k3_reference' / ('case_%03d.joblib' % int(index))


previous = SimpleNamespace(source=base.previous.source, union_indices=union_indices, patient_path=prior_path)


def patient_path(index):
    assert 0 <= int(index) < 118
    old = base.patient_path(index)
    return old if old.exists() else OUT / 'patient_features' / ('case_%03d.joblib' % int(index))


def folds(group, which, outer=None):
    assert group == 'Fixed118'
    record = json.loads((OUT / group / 'nested5_folds.json').read_text(encoding='utf-8'))
    ff = record['outer_folds'] if which == 'outer' else record['final_tuning_folds'] if which == 'final' else record['outer_folds'][outer - 1]['inner_folds']
    result = [(np.asarray(r['train_indices'], int), np.asarray(r['validation_indices'], int)) for r in ff]
    allowed = set(specifications()[1]) if which != 'inner' else set(folds(group, 'outer')[outer - 1][0])
    for a, b in result:
        assert not set(a) & set(b) and set(a) | set(b) == allowed
    assert len(result) == 5
    return result


def define():
    assert not OUT.exists(), 'Never redraw or replace the fixed118 experiment'
    base.protect()
    old = json.loads((base.OUT / 'protocol.json').read_text(encoding='utf-8'))
    protected = {**old['protected_hashes'], **old['source_hashes']}
    delivery = json.loads((base.OUT / 'delivery_manifest.json').read_text(encoding='utf-8'))
    assert delivery['passed'] and delivery['visual_verification_passed']
    protected.update({str((base.OUT / name).relative_to(ROOT)): sha for name, sha in delivery['outputs'].items()})
    for p in [base.OUT / 'delivery_manifest.json', SELECTION, ROOT / 'corrected_118_cohort/cohort_118.csv']:
        protected[str(p.relative_to(ROOT))] = digest(p)
    protected.update({str(p.relative_to(ROOT)): digest(p) for p in HISTORICAL.rglob('*') if p.is_file()})
    for name, sha in protected.items():
        assert digest(ROOT / name) == sha, name
    cohort, tr, te = specifications()
    OUT.mkdir(parents=True)
    for name in ('patient_features', 'k3_reference', 'models', 'phases', 'Fixed118'):
        (OUT / name).mkdir()
    nested = base.previous.source.nested(cohort, tr)
    save(OUT / 'Fixed118/nested5_folds.json', nested)
    manifest = {'Fixed118': {'train': tr.tolist(), 'test': te.tolist(), 'sample': list(range(118)),
        'source_selection_sha256': digest(SELECTION), 'source_historical_split_sha256': digest(HISTORICAL / 'split_manifest.json'),
        'counts': {'train': {'N': 82, 'Positive': 58, 'Negative': 24},
                   'test': {'N': 36, 'Positive': 25, 'Negative': 11}}}}
    save(OUT / 'sample_manifest.json', manifest)
    for label, ix in [('Train82', tr), ('Test36', te), ('Sample118', np.arange(118))]:
        cohort.iloc[ix].to_csv(OUT / 'Fixed118' / (label + '_cases.csv'), index=False, encoding='utf-8-sig')
    sources = [ROOT / 'lvi_prediction' / name for name in ('ivim_fixed118_study_20261021.py',
        'run_ivim_fixed118_20261021.py', 'verify_ivim_fixed118_20261021.py', 'report_ivim_fixed118_20261021.py')]
    method = {k: v for k, v in old.items() if k not in ('protected_hashes', 'source_hashes')}
    method.update({'authorization': 'use original retained118 dataset with original82/36split;rerun current pureIVIM five-model nestedK method',
        'sampling_and_splits': 'exact original retained118 and82train/36test,seed42;no new patient selection',
        'samples': manifest, 'groups': GROUPS, 'overlapping_cases_between_samples': None,
        'CV': 'newouter5/inner5 within original82 training;seed42/62201..62205/final62212;no test in anytraining fold',
        'freeze': 'verify/freeze5 main models,then exactly5 model calls on original36 inone stage',
        'previous_method_source': str(base.OUT.relative_to(ROOT)), 'previous_method_changed': False,
        'historical_good_auc_scope': 'retained multimodal andADC habitat test AUC.7709/.7964;not pureIVIM orouter-fivefold evidence',
        'historical_results_replaced': False, 'previous_adoption_decisions_changed': False,
        'source_hashes': {str(p.relative_to(ROOT)): digest(p) for p in sources}, 'protected_hashes': protected})
    save(OUT / 'protocol.json', method)
    save(OUT / 'definition_frozen.json', {'passed': True, 'protocol_sha256': digest(OUT / 'protocol.json'),
        'train82_test36_exact_original_IDs': True, 'models_fitted': 0, 'test_scored': False})
    csv('candidate_definitions', options())
    print('Original118 verified:82train58positive24negative;36test25positive11negative', flush=True)


def protect():
    protocol = json.loads((OUT / 'protocol.json').read_text(encoding='utf-8'))
    for field in ('protected_hashes', 'source_hashes'):
        for name, sha in protocol[field].items():
            assert digest(ROOT / name) == sha, name
    definition = json.loads((OUT / 'definition_frozen.json').read_text(encoding='utf-8'))
    assert digest(OUT / 'protocol.json') == definition['protocol_sha256']
    return len(protocol['protected_hashes'])


def data(indices):
    indices = np.asarray(indices, int)
    patients = [joblib.load(patient_path(i)) for i in indices]
    assert all(r['index'] == int(i) for r, i in zip(patients, indices))
    result = {s['ImageKey']: {'indices': indices, 'metadata': feature.feature_metadata(),
        **{kind: np.asarray([r['values'][s['ImageKey']][kind] for r in patients]) for kind, _ in feature.KINDS}}
        for s in feature.specifications()}
    result.update(base.whole(indices)); result['Clinical'] = base.clinical(indices)
    return result
