# -*- coding: utf-8 -*-
"""Frozen two100 cohorts; pure-IVIM five-family K sweep definition."""
import json
from functools import lru_cache
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import ivim_df63_study_20261020 as previous
import ivim_k_sweep_features_20261021 as feature

ROOT = previous.ROOT
OUT = ROOT / 'lvi_prediction/results/lvi_ivim_five_models_k_sweep_20261021'
GROUPS = previous.GROUPS
FAMILIES = ('Habitat', 'Whole', 'Clinical', 'HabitatClinical', 'WholeClinical')
CLINICAL = ROOT / 'lvi_prediction/results/lvi_random120_clinical_extension_20261012'
WHOLE = ROOT / 'lvi_prediction/results/lvi_k23_five_models_120_20261013'
save, digest = previous.save, previous.digest


def read(name):
    # Preserve the explicit 'NA' not-applicable sentinel used by Clinical/Whole.
    return pd.read_csv(OUT / (name + '.csv'), float_precision='round_trip',
                       keep_default_na=False, na_values=[''])


def csv(name, rows):
    pd.DataFrame(rows).to_csv(OUT / (name + '.csv'), index=False, encoding='utf-8-sig')


def patient_path(index):
    return OUT / 'patient_features' / ('case_%03d.joblib' % int(index))


@lru_cache(None)
def options():
    result = []
    for family in FAMILIES:
        specs = feature.specifications() if family.startswith('Habitat') else [
            {'ImageKey': n + '__Whole', 'Normalization': n, 'K': 0, 'ComboID': 'NA'} for n in feature.NORMS]
        if family == 'Clinical':
            specs = [{'ImageKey': None, 'Normalization': 'NA', 'K': 0, 'ComboID': 'NA'}]
        for spec in specs:
            for ranking in (('Clinical',) if family == 'Clinical' else ('Association', 'Stability')):
                for top in ((0,) if family == 'Clinical' else (10, 20, 0)):
                    for C in (.003, .03, .3, 3.):
                        result.append({'family': family, 'image_key': spec['ImageKey'],
                            'normalization': spec['Normalization'], 'K': spec['K'], 'ComboID': spec['ComboID'],
                            'ranking': ranking, 'top': top, 'C': C, 'method': 'Ridge',
                            'clinical': family.endswith('Clinical') or family == 'Clinical',
                            'candidate_id': len(result)})
    assert len(result) == 1540
    return result


def config(row):
    return {k: row[k] for k in options()[0]}


def inputs(group):
    return previous.inputs(group)


def folds(group, which, outer=None):
    return previous.folds(group, which, outer)


def define():
    assert not OUT.exists(), 'Never replace an existing experiment'
    prior = ROOT / 'lvi_prediction/results/lvi_ivim_top3_adc_t2_fusion_20261020'
    p = json.loads((prior / 'protocol.json').read_text(encoding='utf-8'))
    protected = {**p['protected_hashes'], **p['source_hashes']}
    delivery = json.loads((prior / 'delivery_manifest.json').read_text(encoding='utf-8'))
    assert delivery['passed']
    protected.update({str((prior / name).relative_to(ROOT)): sha for name, sha in delivery['outputs'].items()})
    protected[str((prior / 'delivery_manifest.json').relative_to(ROOT))] = digest(prior / 'delivery_manifest.json')
    for path in [WHOLE / (stage + '_features.joblib') for stage in ('training', 'test120', 'omitted18')] + [
            CLINICAL / 'clinical_extension_138.csv', CLINICAL / 'clinical_extension_audit.json',
            ROOT / 'lvi_prediction/COHORT_CONFIRMATION_20261012.json', ROOT / 'AGENTS.md']:
        protected[str(path.relative_to(ROOT))] = digest(path)
    for name, sha in protected.items():
        assert digest(ROOT / name) == sha, name
    sources = [ROOT / 'lvi_prediction' / name for name in (
        'ivim_k_sweep_features_20261021.py', 'ivim_five_k_study_20261021.py',
        'run_ivim_five_k_20261021.py', 'verify_ivim_five_k_20261021.py', 'report_ivim_five_k_20261021.py')]
    OUT.mkdir(parents=True)
    for folder in ('patient_features', 'phases', 'models'):
        (OUT / folder).mkdir()
    save(OUT / 'protocol.json', {
        'date': '2026-10-21', 'authorization': 'pureIVIM five families;existing two100 samples70/30;k selected inside training;no outcome-driven patient selection',
        'five_families': FAMILIES, 'K_candidates': feature.KS, 'image_specs': feature.specifications(),
        'normalizations': feature.NORMS, 'clustering_inputs': feature.DRIVERS,
        'clustering': 'patient-local KMeans n_init5 seed64212 sampledmax4000;unchanged D/f valid masks and1mm geometry;3mm local mean/SD',
        'K3_equivalence': 'generalized descriptor/contact/centroid formulas;all156 invariant values and labels verified against prior63 experiment',
        'habitat_features': '156 invariant features for everyK;17textures/channel plus original raw/spatial metrics;strict minimumROI12 texture rule',
        'whole_features': '186 original non-shape PyRadiomics IVIM D93/f93;norm-specific patient caches;no T2/ADC columns',
        'clinical_features': 'same11 audited preoperative fields;all retained;train-fold-only preprocessing;no pathology or postoperative predictors',
        'CV': 'exact same outer5 andinner5 as previous two100;everyouter fold independently selectsK,input,norm,feature subset,C in its inner5',
        'fixed_K_diagnostic': 'within eachK selectinput/norm/config insideinner;reportfit/outer;do notscore fixedK alternatives ontest',
        'training_only_objective': 'unchanged meanInnerAUC-.10*SD-.30*max(meanFit-meanInnerAUC-.10,0)',
        'image_quality': 'training-fold completeness>=.9 ICC>=.8 nonconstant;Association20 bootstrap/Stability ranking;absSpearman<.95',
        'image_budgets': '10/20/all selected insideinner5;no fixed3or6 maximum',
        'classifier': 'L2liblinearLR C=.003/.03/.3/3;max_iter3000 seed64212, allfivefamilies',
        'preprocessing': 'train-fold medianfill,P2.5/P97.5clip,median/IQR scale;clinicalall11 appended afterimage selection',
        'threshold': 'selected full70 configuration trainingOOFYouden;fixed before tests',
        'freeze': 'finish andverify10 finalmodels plusalloutermodels beforetest;exactly10 own30 model calls inone stage',
        'uncertainty': '5000stratifiedpatientbootstrap;pairedmodel contrasts within eachsample',
        'test_history': 'previously inspected andreused cohort;exploratory internal validation,not new external validation',
        'overlapping_cases_between_samples': 74, 'sampling_redrawn': False, 'test_AUC_used_for_selection': False,
        'Dstar': False, 'T2_ADC_image_features': False, 'formal_model_use': False, 'automatic_adoption': False,
        'candidate_counts': {f: sum(c['family'] == f for c in options()) for f in FAMILIES},
        'source_hashes': {str(path.relative_to(ROOT)): digest(path) for path in sources}, 'protected_hashes': protected})
    for group in GROUPS:
        (OUT / group).mkdir()
        for name in ('nested5_folds.json', 'Train70_cases.csv', 'Test30_cases.csv'):
            (OUT / group / name).write_bytes((previous.OUT / group / name).read_bytes())
    for name in ('sample_manifest.json', 'sample_counts.csv', 'between_sample_overlap.json'):
        (OUT / name).write_bytes((previous.OUT / name).read_bytes())
    save(OUT / 'definition_frozen.json', {'passed': True, 'protocol_sha256': digest(OUT / 'protocol.json'),
        'models_fitted': 0, 'cases_changed': False, 'test_scored': False})
    csv('candidate_definitions', options())
    print('Defined pureIVIM K2..6;30 habitat settings;1540 configurations;10 final models', flush=True)


def protect():
    p = json.loads((OUT / 'protocol.json').read_text(encoding='utf-8'))
    for field in ('protected_hashes', 'source_hashes'):
        for name, sha in p[field].items():
            assert digest(ROOT / name) == sha, name
    d = json.loads((OUT / 'definition_frozen.json').read_text(encoding='utf-8'))
    assert digest(OUT / 'protocol.json') == d['protocol_sha256']
    return len(p['protected_hashes'])


def clinical(indices):
    cohort = previous.source.cohort()
    audit = json.loads((CLINICAL / 'clinical_extension_audit.json').read_text(encoding='utf-8'))
    assert audit['passed'] and not audit['postoperative_pathology_predictors_used']
    assert digest(CLINICAL / 'clinical_extension_138.csv') == audit['parsed_data_sha256']
    table = pd.read_csv(CLINICAL / 'clinical_extension_138.csv').set_index('AnalysisID')
    assert table.index.is_unique and set(table.index) == set(cohort.AnalysisID)
    columns = audit['clinical_columns']
    assert len(columns) == 11 and not any('Outcome' in c or 'patholog' in c.lower() for c in columns)
    return {'indices': np.asarray(indices), 'metadata': [{'name': c, 'modality': 'Clinical', 'block': 'Clinical'} for c in columns],
            'base': table.loc[cohort.AnalysisID.iloc[indices], columns].to_numpy(float)}


def whole(indices):
    loaded = [joblib.load(WHOLE / (stage + '_features.joblib')) for stage in ('training', 'test120', 'omitted18')]
    result = {}
    for norm in feature.NORMS:
        key = norm + '__Whole'
        parts = [r[key] for r in loaded]
        ix = np.concatenate([r['indices'] for r in parts])
        assert len(ix) == len(set(ix)) == 138 and set(ix) == set(range(138))
        lookup = {int(v): j for j, v in enumerate(ix)}
        rows = [lookup[int(i)] for i in indices]
        meta = parts[0]['metadata']
        assert all(p['metadata'] == meta for p in parts)
        keep = [j for j, m in enumerate(meta) if m['modality'] == 'IVIM']
        assert len(keep) == 186 and {meta[j]['channel'] for j in keep} == {'D', 'f'}
        result[key] = {'indices': np.asarray(indices), 'metadata': [meta[j] for j in keep],
            **{kind: np.concatenate([r[kind] for r in parts])[rows][:, keep] for kind, _ in feature.KINDS}}
    return result


def data(indices):
    indices = np.asarray(indices, int)
    patients = [joblib.load(patient_path(i)) for i in indices]
    assert all(r['index'] == int(i) for r, i in zip(patients, indices))
    result = {s['ImageKey']: {'indices': indices, 'metadata': feature.feature_metadata(),
        **{kind: np.asarray([r['values'][s['ImageKey']][kind] for r in patients]) for kind, _ in feature.KINDS}}
        for s in feature.specifications()}
    result.update(whole(indices)); result['Clinical'] = clinical(indices)
    return result
