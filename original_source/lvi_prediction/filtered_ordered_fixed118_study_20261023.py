"""Training-diagnosed endpoint ablation with a smaller Association-only search."""
from functools import lru_cache
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import ordered_fixed118_study_20261023 as prior
import run_ordered_fixed118_20261023 as previous_run

ROOT = prior.ROOT
OUT = ROOT / 'lvi_prediction/results/lvi_filtered_ordered_fixed118_20261023'
GROUPS, FAMILIES, feature = prior.GROUPS, prior.FAMILIES, prior.feature
save, digest, inputs, folds = prior.save, prior.digest, prior.inputs, prior.folds
previous, BASIC = prior.previous, prior.BASIC
last = prior.last  # The last scored low-dimensional control, not the unscored1320 study.
original_values, PHYSICAL_METADATA = prior.original_values, prior.PHYSICAL_METADATA


def read(name):
    return pd.read_csv(OUT / (name + '.csv'), float_precision='round_trip', keep_default_na=False, na_values=[''])


def csv(name, rows):
    pd.DataFrame(rows).to_csv(OUT / (name + '.csv'), index=False, encoding='utf-8-sig')


def removed_feature(meta):
    return any(meta['name'].endswith('original_firstorder_' + name) for name in ('Minimum', 'Maximum', 'Range'))


@lru_cache(None)
def options():
    result = []
    for old in prior.options():
        if old['ranking'] == 'Stability':
            continue
        if old['method'] == 'RBF' and (old['C'] > 1 or old['gamma'] == .1):
            continue
        r = old.copy()
        if r['image_key'] is not None:
            r['image_key'] = r['image_key'].replace('__OrderedFull', '__FilteredFull').replace('__WholePhysical', '__WholePhysicalFiltered')
        r['view'] = r['view'].replace('OrderedFull', 'FilteredFull').replace('WholePhysical', 'WholePhysicalFiltered')
        r['candidate_id'] = len(result)
        result.append(r)
    assert len(result) == 392
    return result


def config(row):
    return {k: row[k] for k in options()[0]}


def patient_path(index):
    old = prior.patient_path(index)
    return old if old.exists() else OUT / 'physical_features' / ('case_%03d.joblib' % int(index))


def extract_physical(indices):
    import run_random120_retained_methods_20261010 as loader
    todo = [int(i) for i in indices if not patient_path(i).exists()]
    if todo:
        profiles = loader.load_profiles()
        for i in todo:
            destination = patient_path(i)
            temp = destination.with_suffix('.pending')
            joblib.dump(prior.physical(i, profiles[i]), temp, compress=3); temp.replace(destination)
    print('Physical rows:needed%d new%d;previous physical cache read-only' % (len(indices), len(todo)), flush=True)


def data(indices):
    indices = np.asarray(indices, int)
    original = prior.ordinal.cached_features(indices)
    result = {}
    for norm, d in original.items():
        compact, _ = prior.views.specifications(d['metadata'])
        for view, source in [('OrderedCompact', compact), ('FilteredFull', list(range(1242)))]:
            keep = [j for j in source if not removed_feature(d['metadata'][j])]
            assert len(keep) == (321 if view == 'OrderedCompact' else 1206)
            result[norm + '__' + view] = {'indices': indices.copy(), 'metadata': [d['metadata'][j] for j in keep],
                **{kind: d[kind][:, keep] for kind in ('base', 'erode', 'dilate')}}
    physical = [joblib.load(patient_path(i)) for i in indices]
    assert all(r['index'] == int(i) and r['metadata'] == PHYSICAL_METADATA for i, r in zip(indices, physical))
    for key, d in last.old.whole(indices).items():
        keep = [j for j, m in enumerate(d['metadata']) if not removed_feature(m)]
        assert len(keep) == 360
        norm = key.split('__')[0]
        result[norm + '__WholePhysicalFiltered'] = {'indices': indices.copy(),
            'metadata': [d['metadata'][j] for j in keep] + PHYSICAL_METADATA,
            **{kind: np.column_stack([d[kind][:, keep], np.asarray([p[kind] for p in physical])]) for kind in ('base', 'erode', 'dilate')}}
    result['Clinical'] = last.old.prior.base.clinical(indices)
    return result


def protect():
    p = json.loads((OUT / 'protocol.json').read_text(encoding='utf-8'))
    for field in ('protected_hashes', 'source_hashes'):
        for name, sha in p[field].items():
            assert digest(ROOT / name) == sha, name
    f = json.loads((OUT / 'definition_frozen.json').read_text(encoding='utf-8'))
    assert digest(OUT / 'protocol.json') == f['protocol_sha256']
    return len(p['protected_hashes'])


def define():
    assert not OUT.exists()
    previous_run.frozen()
    assert not (prior.OUT / 'test_started.json').exists()
    assert json.loads((prior.OUT / 'training_verification.json').read_text(encoding='utf-8'))['passed']
    old = json.loads((prior.OUT / 'protocol.json').read_text(encoding='utf-8'))
    protected = {**old['protected_hashes'], **old['source_hashes']}
    protected.update({str(p.relative_to(ROOT)): digest(p) for p in prior.OUT.rglob('*') if p.is_file()})
    for key, sha in protected.items():
        assert digest(ROOT / key) == sha, key
    OUT.mkdir(parents=True)
    for folder in ('models', 'phases', 'physical_features', 'Fixed118'):
        (OUT / folder).mkdir()
    for name in ('Train82_cases.csv', 'Test36_cases.csv', 'Sample118_cases.csv', 'nested5_folds.json'):
        (OUT / 'Fixed118' / name).write_bytes((prior.OUT / 'Fixed118' / name).read_bytes())
    (OUT / 'sample_manifest.json').write_bytes((prior.OUT / 'sample_manifest.json').read_bytes())
    sources = [ROOT / 'lvi_prediction' / name for name in ('filtered_ordered_fixed118_study_20261023.py',
        'run_filtered_ordered_fixed118_20261023.py', 'verify_filtered_ordered_fixed118_20261023.py', 'report_filtered_ordered_fixed118_20261023.py')]
    method = {k: v for k, v in old.items() if k not in ('protected_hashes', 'source_hashes', 'pretraining_code_amendment')}
    method.update({'authorization': 'Continue improvement;focused ablation motivated only by training-selected endpoint features andoverfit gap.',
        'prior1320_experiment': 'Frozen andverified,holdout never scored;preserved as training-only diagnostic.',
        'views': {'OrderedCompact': '321 unchanged fixed17 texture/ordered descriptors', 'FilteredFull': '1206 original ordered features excluding Minimum/Maximum/Range'},
        'whole': '360 nonshape radiomics without normalized Minimum/Maximum/Range +30 unchanged raw ADC/D/f statistics.',
        'endpoint_filter': 'Prespecified name-based exclusion,independent of patient outcome andtest observations.',
        'quality': 'Training-only ICC>=.75 completeness>=.9 nonconstant;20-bootstrap Association ranking;absSpearman<.95.',
        'ranking': 'Association only;ICC remains aquality gate,not a standalone predictive ranking.',
        'classifiers': 'BalancedLR C.003/.03/.3;RBF C.1/1 gamma scale/.01;Basic5/Extended11 insideinner5.',
        'candidates': len(options()), 'test_calls_before_this_experiment_in_active_improvement': 0,
        'source_hashes': {str(p.relative_to(ROOT)): digest(p) for p in sources}, 'protected_hashes': protected})
    save(OUT / 'protocol.json', method)
    save(OUT / 'definition_frozen.json', {'passed': True, 'protocol_sha256': digest(OUT / 'protocol.json'), 'models_fitted': 0, 'test_scored': False})
    csv('candidate_definitions', options())
    prior.read('training_choices').to_csv(OUT / 'previous1320_training_only_choices.csv', index=False, encoding='utf-8-sig')
    print('Defined392-option endpoint-filtered control;prior1320 holdout calls0;exact82/36', flush=True)
