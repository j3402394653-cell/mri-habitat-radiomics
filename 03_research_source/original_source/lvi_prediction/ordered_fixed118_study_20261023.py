"""Preserve ordered habitats and unnormalized quantitative map statistics."""
from functools import lru_cache
from types import SimpleNamespace
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import run_multiblock_fixed118_20261023 as last
import run_k3_two_norm_random90_48_138_20261013 as ordinal
import habitat_stable_views_20261013 as views
from quantile_habitat_features import original_values

ROOT = last.ROOT
OUT = ROOT / 'lvi_prediction/results/lvi_ordered_quantitative_fixed118_20261023'
GROUPS = ('Fixed118',)
FAMILIES = last.FAMILIES
inputs, folds = last.old.inputs, last.old.folds
save, digest = last.save, last.digest
previous = last.old.previous
feature = SimpleNamespace(KS=(), NORMS=last.old.feature.NORMS)
BASIC = last.BASIC
STATS = ('Mean', 'SD', 'P10', 'P25', 'Median', 'P75', 'P90', 'IQR', 'Skewness', 'Kurtosis')
PHYSICAL_METADATA = [{'name': mod + '__WholePhysical_' + ch + '_' + s, 'modality': mod,
                     'channel': ch, 'block': 'PhysicalStatistics', 'habitat': 0}
    for mod, ch in (('ADC', 'ADC'), ('IVIM', 'D'), ('IVIM', 'f')) for s in STATS]


def csv(name, rows):
    pd.DataFrame(rows).to_csv(OUT / (name + '.csv'), index=False, encoding='utf-8-sig')


def read(name):
    return pd.read_csv(OUT / (name + '.csv'), float_precision='round_trip', keep_default_na=False, na_values=[''])


@lru_cache(None)
def options():
    rows = []
    algorithms = [('Ridge', C, 'NA') for C in (.003, .03, .3)]
    algorithms += [('RBF', C, gamma) for C in (.1, 1., 10.) for gamma in ('scale', .01, .1)]
    for family in FAMILIES:
        specs = [(n + '__' + view, n, view, 3) for n in feature.NORMS for view in ('OrderedCompact', 'OrderedFull')] if family.startswith('Habitat') else [
            (n + '__WholePhysical', n, 'WholePhysical', 0) for n in feature.NORMS]
        if family == 'Clinical':
            specs = [(None, 'NA', 'Clinical', 0)]
        for key, norm, view, k in specs:
            for ranking in (('Clinical',) if family == 'Clinical' else ('Association', 'Stability')):
                for top in ((0,) if family == 'Clinical' else (5, 15, 0)):
                    for scope in (('Basic5', 'Extended11') if family in ('Clinical', 'HabitatClinical', 'WholeClinical') else ('None',)):
                        for algorithm, C, gamma in algorithms:
                            rows.append({'family': family, 'image_key': key, 'normalization': norm, 'view': view,
                                'K': k, 'ComboID': 'All6' if k else 'NA', 'ranking': ranking, 'top': top,
                                'C': C, 'method': algorithm, 'gamma': gamma, 'clinical_scope': scope,
                                'clinical': scope != 'None', 'candidate_id': len(rows)})
    assert len(rows) == 1320
    return rows


def config(row):
    return {k: row[k] for k in options()[0]}


def patient_path(index):
    return OUT / 'physical_features' / ('case_%03d.joblib' % int(index))


def physical(index, profile):
    result = {'index': int(index), 'metadata': PHYSICAL_METADATA}
    for kind, mask_key in (('base', 'roi'), ('erode', 'erode'), ('dilate', 'dilate')):
        values = []
        for mod, channel in (('ADC', 0), ('IVIM', 0), ('IVIM', 1)):
            p = profile[mod]
            mask = np.asarray(p[mask_key], bool) & p['valid']
            x = original_values(p, mod)[..., channel][mask]
            assert len(x) >= 12 and np.isfinite(x).all()
            q = np.quantile(x, [.1, .25, .5, .75, .9])
            mean, sd = x.mean(), x.std()
            centered = (x - mean) / sd if sd > 1e-12 else np.zeros(len(x))
            values.extend([mean, sd, *q, q[3] - q[1], np.mean(centered ** 3), np.mean(centered ** 4)])
            if kind == 'base':
                expected = p['normalization'][channel]['mean']
                assert abs(mean - expected) / max(abs(expected), 1e-10) < 1e-5
        result[kind] = np.asarray(values, float)
        assert len(values) == 30
    return result


def extract_physical(indices):
    import run_random120_retained_methods_20261010 as loader
    todo = [int(i) for i in indices if not patient_path(i).exists()]
    if todo:
        profiles = loader.load_profiles()
        for i in todo:
            dest = patient_path(i)
            temp = dest.with_suffix('.pending')
            joblib.dump(physical(i, profiles[i]), temp, compress=3); temp.replace(dest)
    print('Physical statistics:needed%d new%d;outcome independent' % (len(indices), len(todo)), flush=True)


def data(indices):
    indices = np.asarray(indices, int)
    original = ordinal.cached_features(indices)
    result = {}
    for norm, d in original.items():
        compact, _ = views.specifications(d['metadata'])
        assert len(d['metadata']) == 1242 and len(compact) == 321
        for view, keep in [('OrderedFull', list(range(1242))), ('OrderedCompact', compact)]:
            result[norm + '__' + view] = {'indices': indices.copy(), 'metadata': [d['metadata'][j] for j in keep],
                **{kind: d[kind][:, keep] for kind in ('base', 'erode', 'dilate')}}
    physical_rows = [joblib.load(patient_path(i)) for i in indices]
    assert all(r['index'] == int(i) and r['metadata'] == PHYSICAL_METADATA for i, r in zip(indices, physical_rows))
    for key, d in last.old.whole(indices).items():
        norm = key.split('__')[0]
        result[norm + '__WholePhysical'] = {'indices': indices.copy(), 'metadata': d['metadata'] + PHYSICAL_METADATA,
            **{kind: np.column_stack([d[kind], np.asarray([p[kind] for p in physical_rows])]) for kind in ('base', 'erode', 'dilate')}}
    result['Clinical'] = last.old.prior.base.clinical(indices)
    for key, d in result.items():
        assert not any('Dstar' in m['name'] or 'Dfast' in m['name'] for m in d['metadata'])
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
    assert not OUT.exists(), 'Never replace an experiment'
    last.frozen()
    p = json.loads((last.OUT / 'protocol.json').read_text(encoding='utf-8'))
    delivery = json.loads((last.OUT / 'delivery_manifest.json').read_text(encoding='utf-8'))
    assert delivery['passed'] and delivery['visual_verification_passed']
    protected = {**p['protected_hashes'], **p['source_hashes']}
    for name, sha in delivery['outputs'].items():
        path = last.OUT / name
        assert digest(path) == sha, name
        protected[str(path.relative_to(ROOT))] = sha
    protected[str((last.OUT / 'delivery_manifest.json').relative_to(ROOT))] = digest(last.OUT / 'delivery_manifest.json')
    rawdir = ordinal.previous.baseline.prior.OUT
    for path in (rawdir / 'training_features.joblib', rawdir / 'frozen54_features.joblib', Path(ordinal.__file__).resolve()):
        protected[str(path.relative_to(ROOT))] = digest(path)
    OUT.mkdir(parents=True)
    for folder in ('physical_features', 'phases', 'models', 'Fixed118'):
        (OUT / folder).mkdir()
    source = last.OUT / 'Fixed118'
    for name in ('Train82_cases.csv', 'Test36_cases.csv', 'Sample118_cases.csv', 'nested5_folds.json'):
        (OUT / 'Fixed118' / name).write_bytes((source / name).read_bytes())
    (OUT / 'sample_manifest.json').write_bytes((last.OUT / 'sample_manifest.json').read_bytes())
    files = [ROOT / 'lvi_prediction' / name for name in ('ordered_fixed118_study_20261023.py',
        'run_ordered_fixed118_20261023.py', 'verify_ordered_fixed118_20261023.py', 'report_ordered_fixed118_20261023.py')]
    save(OUT / 'protocol.json', {'date': '2026-10-23', 'authorization': 'Human requested further improvement after testAUC below.6.',
        'sampling': 'Original118,exact82/36;same nestedouter5/inner5;no patient selection byAUC.',
        'habitats': 'Original patient-local K3 clusters;T2/ADC intensity+3mmmean/SD,IVIM D/f all6 dimensions;ordered canonical region identities retained.',
        'views': {'OrderedCompact': '321 fixed17 textures plus ordered spatial/raw descriptors', 'OrderedFull': '1242 original93 textures plus ordered spatial/raw descriptors'},
        'whole': '372 original nonshape radiomics +30 raw ADC/D/f map statistics;no shared raw T2 interpretation.',
        'physical_values': 'Recover original unclipped map numbers from saved z/mean/SD;not new physical-unit certification;verify recovered base means.',
        'quality': 'Training-only ICC>=.75 completeness>=.9 nonconstant;20-bootstrap Association orStability ranking;withintraining absSpearman<.95.',
        'budgets': '5/15/all eligible decorrelated image variables,selected insideinner5.',
        'classifiers': 'Balanced-class L2LR C.003/.03/.3;RBF-SVM C.1/1/10 gamma scale/.01/.1;allinsideinner5.',
        'clinical': 'Basic5 orExtended11 chosen insideinner5;audited preoperative fields only.',
        'score_interpretation': 'LR returns fitted probabilities;SVM returns sigmoid(decision margin) as ranking score,not calibrated disease risk.',
        'objective': 'unchanged meanInnerAUC-.10SD-.30max(meanFit-meanInnerAUC-.10,0)',
        'threshold': 'Selected full82 configuration inner5 OOFYouden;fixed before36 scoring.',
        'test': 'One frozen five-model stage only;no perclassifier orview alternatives separately scored.',
        'Dstar': False, 'formal_model_use': False, 'automatic_adoption': False, 'prior_test_exposure': True,
        'new_external_validation': False, 'excluded_experiments_used': False, 'candidates': len(options()),
        'source_hashes': {str(path.relative_to(ROOT)): digest(path) for path in files}, 'protected_hashes': protected})
    save(OUT / 'definition_frozen.json', {'passed': True, 'protocol_sha256': digest(OUT / 'protocol.json'), 'models_fitted': 0, 'test_scored': False})
    csv('candidate_definitions', options())
    print('Defined1320 configurations;ordered K3 habitats andraw quantitative whole features;exact82/36', flush=True)
