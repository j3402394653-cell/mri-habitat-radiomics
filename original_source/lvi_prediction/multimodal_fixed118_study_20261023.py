"""Add native T2/ADC features to the frozen118 IVIM comparison; preserve82/36."""
import json
from functools import lru_cache
import joblib
import numpy as np
import pandas as pd
import ivim_fixed118_study_20261021 as prior
from verify_regional_summary77_two_tests_20261014 import reference_features

ROOT = prior.ROOT
OUT = ROOT / 'lvi_prediction/results/lvi_multimodal_fixed118_five_models_20261023'
GROUPS, FAMILIES, feature = prior.GROUPS, prior.FAMILIES, prior.feature
save, digest, options, config = prior.save, prior.digest, prior.options, prior.config
CLINICAL, WHOLE, HISTORICAL = prior.CLINICAL, prior.WHOLE, prior.HISTORICAL
inputs, specifications, patient_path, prior_path = prior.inputs, prior.specifications, prior.patient_path, prior.prior_path
previous = prior.previous


def read(name):
    return pd.read_csv(OUT / (name + '.csv'), float_precision='round_trip', keep_default_na=False, na_values=[''])


def csv(name, rows):
    pd.DataFrame(rows).to_csv(OUT / (name + '.csv'), index=False, encoding='utf-8-sig')


def folds(group, which, outer=None):
    # Exact same existing nested folds as the IVIM-only comparator.
    return prior.folds(group, which, outer)


@lru_cache(None)
def _whole_sources():
    return [joblib.load(WHOLE / (stage + '_features.joblib')) for stage in ('training', 'test120', 'omitted18')]


def whole(indices):
    result = {}
    for norm in feature.NORMS:
        key = norm + '__Whole'
        parts = [r[key] for r in _whole_sources()]
        all_indices = np.concatenate([r['indices'] for r in parts])
        assert len(set(all_indices)) == len(all_indices) == 138
        lookup = {int(i): j for j, i in enumerate(all_indices)}
        rr = [lookup[int(i)] for i in indices]
        meta = parts[0]['metadata']
        assert all(p['metadata'] == meta for p in parts) and len(meta) == 372
        assert {m: sum(s['modality'] == m for s in meta) for m in ('T2', 'ADC', 'IVIM')} == {'T2': 93, 'ADC': 93, 'IVIM': 186}
        result[key] = {'indices': np.asarray(indices, int), 'metadata': meta,
            **{kind: np.concatenate([r[kind] for r in parts])[rr] for kind, _ in feature.KINDS}}
    return result


def data(indices):
    indices = np.asarray(indices, int)
    image = prior.data(indices)
    original = reference_features(indices)
    result = {}
    for spec in feature.specifications():
        key = spec['ImageKey']
        source = original[(spec['Normalization'], 'Invariant')]
        meta = source['metadata']
        pos = [j for j, m in enumerate(meta) if m['modality'] == 'IVIM']
        assert len(meta) == 330 and len(pos) == 156
        assert [meta[j] for j in pos] == image[key]['metadata']
        result[key] = {'indices': indices.copy(), 'metadata': meta}
        for kind, _ in feature.KINDS:
            values = source[kind].copy()
            values[:, pos] = image[key][kind]
            result[key][kind] = values
    result.update(whole(indices))
    result['Clinical'] = image['Clinical']
    return result


def define():
    assert not OUT.exists(), 'Never overwrite an existing multimodal experiment'
    import run_ivim_fixed118_20261021 as previous_run
    previous_run.frozen()
    previous_protocol = json.loads((prior.OUT / 'protocol.json').read_text(encoding='utf-8'))
    delivery = json.loads((prior.OUT / 'delivery_manifest.json').read_text(encoding='utf-8'))
    assert delivery['passed'] and delivery['visual_verification_passed']
    protected = {**previous_protocol['protected_hashes'], **previous_protocol['source_hashes']}
    for name, sha in delivery['outputs'].items():
        p = prior.OUT / name
        assert digest(p) == sha, name
        protected[str(p.relative_to(ROOT))] = sha
    protected[str((prior.OUT / 'delivery_manifest.json').relative_to(ROOT))] = digest(prior.OUT / 'delivery_manifest.json')
    source_paths = [ROOT / 'lvi_prediction' / name for name in (
        'multimodal_fixed118_study_20261023.py', 'run_multimodal_fixed118_20261023.py',
        'verify_multimodal_fixed118_20261023.py', 'report_multimodal_fixed118_20261023.py')]
    for name, sha in protected.items():
        assert digest(ROOT / name) == sha, name
    OUT.mkdir(parents=True)
    for folder in ('models', 'phases', 'Fixed118'):
        (OUT / folder).mkdir()
    for name in ('nested5_folds.json', 'Train82_cases.csv', 'Test36_cases.csv', 'Sample118_cases.csv'):
        (OUT / 'Fixed118' / name).write_bytes((prior.OUT / 'Fixed118' / name).read_bytes())
    (OUT / 'sample_manifest.json').write_bytes((prior.OUT / 'sample_manifest.json').read_bytes())
    method = {k: v for k, v in previous_protocol.items() if k not in ('protected_hashes', 'source_hashes', 'pretraining_amendment')}
    method.update({
        'date': '2026-10-23', 'authorization': 'Human requested to stop IVIM-only analysis and add the other MRI sequences.',
        'MRI_channels': ['T2', 'ADC', 'IVIM_D', 'IVIM_f'], 'Dstar': False,
        'T2_ADC_image_features': True,
        'fusion': 'Separate native-space sequence clustering then concatenate patient-level habitat features; no cross-sequence voxel correspondence.',
        'T2_ADC_clustering_inputs': 'Each uses intensity,3mm local mean,3mm local SD; previously validated patient-local extraction.',
        'T2_ADC_K': 3, 'IVIM_K_candidates': feature.KS,
        'habitat_features': '330 invariant candidates: T2 87 + ADC 87 at fixedK3 + IVIM D/f156 at inner-selectedK2..6.',
        'whole_features': '372 original non-shape radiomics candidates: T2 93,ADC93,D93,f93.',
        'feature_quota': 'No forced sequence quota; report actual selected modality counts.',
        'threshold': 'Selected full82 configuration trainingOOFYouden;fixed before original36 test scoring.',
        'comparison': 'Same patients,exact same inner/outer folds,clinical fields,selection objective andclassifier as IVIM-only; paired5000 stratified bootstrap for saved holdout probabilities.',
        'previous_method_source': str(prior.OUT.relative_to(ROOT)),
        'previous_method_changed': 'AddT2/ADC image blocks; IVIM extraction andother model choices unchanged.',
        'prior_candidate_exploration_exposure': 'ThreeIVIM inputs selected using prior two100 exploration; nested estimate is conditional on this previously chosen search space.',
        'formal_model_use': False, 'automatic_adoption': False,
        'source_hashes': {str(p.relative_to(ROOT)): digest(p) for p in source_paths},
        'protected_hashes': protected
    })
    save(OUT / 'protocol.json', method)
    save(OUT / 'definition_frozen.json', {'passed': True, 'protocol_sha256': digest(OUT / 'protocol.json'),
        'train82_test36_exact_original_IDs': True, 'models_fitted': 0, 'test_scored': False,
        'T2_ADC_fixedK3_IVIM_nestedK2_to6': True})
    csv('candidate_definitions', options())
    print('Defined multimodal118:82/36 unchanged;330 habitat/372 whole;1540 inner configurations', flush=True)


def protect():
    p = json.loads((OUT / 'protocol.json').read_text(encoding='utf-8'))
    for field in ('protected_hashes', 'source_hashes'):
        for name, sha in p[field].items():
            assert digest(ROOT / name) == sha, name
    definition = json.loads((OUT / 'definition_frozen.json').read_text(encoding='utf-8'))
    assert digest(OUT / 'protocol.json') == definition['protocol_sha256']
    return len(p['protected_hashes'])
