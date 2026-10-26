# -*- coding: utf-8 -*-
"""Prespecified two random100 samples and all63 IVIM clustering-input combinations."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import ivim_df63_features_20261020 as feature
import run_regional_summary100_ratio_comparison_20261014 as history
import define_lvi138_two_test_sets_20261014 as source

ROOT = history.ROOT
OUT = ROOT/'lvi_prediction/results/lvi_ivim_df63_two100_20261020'
GROUPS = ('SampleA', 'SampleB')
NORMS = feature.NORMALIZATIONS
SAMPLE_SEEDS = (2026102001, 2026102002)
SPLIT_SEEDS = (2026102011, 2026102012)
WORKERS = 6
save, digest = history.save, history.digest


def read(directory, name):
    return pd.read_csv(Path(directory)/(name+'.csv'), float_precision='round_trip')


def specifications():
    data = source.cohort()
    truth, result = data.Outcome.to_numpy(int), {}
    for group, sample_seed, split_seed in zip(GROUPS, SAMPLE_SEEDS, SPLIT_SEEDS):
        sample, unused = train_test_split(np.arange(138), train_size=100, test_size=38,
            stratify=truth, random_state=sample_seed)
        sample = np.sort(sample)
        tr, te = train_test_split(sample, train_size=70, test_size=30, stratify=truth[sample], random_state=split_seed)
        tr, te = np.sort(tr), np.sort(te)
        result[group] = {'sample': sample.tolist(), 'train': tr.tolist(), 'test': te.tolist(),
            'sample_seed': sample_seed, 'split_seed': split_seed,
            'counts': {name: source.counts(data, ix) for name, ix in [('sample', sample), ('train', tr), ('test', te)]}}
    return data, result


def options(norm):
    configs = []
    for ranking in ('Association', 'Stability'):
        for top in (10, 20, 0):
            for C in (.003, .03, .3, 3.):
                configs.append({'normalization': norm, 'view': 'Invariant', 'ranking': ranking, 'top': top,
                    'method': 'Ridge', 'C': C, 'components': 0, 'candidate_id': len(configs)})
    return configs


def inputs(group):
    data = source.cohort()
    saved = json.loads((OUT/'sample_manifest.json').read_text(encoding='utf-8'))[group]
    tr, te = [np.asarray(saved[k], int) for k in ('train', 'test')]
    y = np.full(138, -1, int)
    y[tr] = data.Outcome.to_numpy(int)[tr]
    return data, y, tr, te


def union_indices(which):
    manifest = json.loads((OUT/'sample_manifest.json').read_text(encoding='utf-8'))
    return np.asarray(sorted(set(i for group in GROUPS for i in manifest[group][which])), int)


def folds(group, which, outer=None):
    saved = json.loads((OUT/group/'nested5_folds.json').read_text(encoding='utf-8'))
    rows = saved['outer_folds'] if which == 'outer' else saved['final_tuning_folds'] if which == 'final' else saved['outer_folds'][outer-1]['inner_folds']
    ff = [(np.asarray(r['train_indices'], int), np.asarray(r['validation_indices'], int)) for r in rows]
    allowed = set(inputs(group)[2]) if which != 'inner' else set(folds(group, 'outer')[outer-1][0])
    for a, b in ff:
        assert not set(a)&set(b) and set(a)|set(b) == allowed
    assert len(ff) == 5
    return ff


def define():
    assert not OUT.exists(), 'Do not redraw samples or replace an experiment'
    history.frozen()
    old = json.loads((history.OUT/'protocol.json').read_text(encoding='utf-8'))
    protected = {**old['protected_hashes'], **old['source_hashes']}
    protected.update({str(p.relative_to(ROOT)): digest(p) for p in history.OUT.rglob('*') if p.is_file()})
    protected.update(json.loads((history.OUT/'delivery_manifest.json').read_text(encoding='utf-8'))['source_hashes'])
    pointer = ROOT/'lvi_prediction/CURRENT_LVI_REGIONAL_SUMMARY100_RESULT_20261014.json'
    protected[str(pointer.relative_to(ROOT))] = digest(pointer)
    data, specs = specifications()
    OUT.mkdir(parents=True)
    (OUT/'patient_features').mkdir()
    counts, assignments = [], data.copy()
    assignments.insert(0, 'CohortIndex', np.arange(138))
    for group in GROUPS:
        dest = OUT/group
        dest.mkdir()
        tr, te, sample = [np.asarray(specs[group][k], int) for k in ('train', 'test', 'sample')]
        assignments[group+'Role'] = np.where(np.isin(np.arange(138), tr), 'Train70', np.where(np.isin(np.arange(138), te), 'Test30', 'Unused38'))
        save(dest/'nested5_folds.json', source.nested(data, tr))
        for role, ix in [('Train70', tr), ('Test30', te), ('Sample100', sample)]:
            data.iloc[ix].to_csv(dest/(role+'_cases.csv'), index=False, encoding='utf-8-sig')
            counts.append({'Group': group, 'Role': role, **source.counts(data, ix)})
    assignments.to_csv(OUT/'all138_group_membership.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(counts).to_csv(OUT/'sample_counts.csv', index=False, encoding='utf-8-sig')
    save(OUT/'sample_manifest.json', specs)
    save(OUT/'combinations63.json', feature.combinations63())
    pd.DataFrame([{**r, 'Dimensions': '+'.join(r['Dimensions']), 'Indices': '+'.join(map(str, r['Indices']))} for r in feature.combinations63()]).to_csv(OUT/'combinations63.csv', index=False, encoding='utf-8-sig')
    overlap = {a+'__'+b: len(set(specs['SampleA'][a])&set(specs['SampleB'][b])) for a in ('sample', 'train', 'test') for b in ('sample', 'train', 'test')}
    save(OUT/'between_sample_overlap.json', {'counts': overlap, 'independent_external_validation': False,
        'overlap_does_not_allow_sharing_supervised_choices_between_groups': True})
    modules = [Path(__file__).resolve(), ROOT/'lvi_prediction/ivim_df63_features_20261020.py', ROOT/'lvi_prediction/extract_ivim_df63_20261020.py']
    save(OUT/'protocol.json', {'date': '2026-10-20', 'authorization': 'two random100 cohorts;all63 nonempty D/f intensity/local3mmmean/SD clustering combinations;70/30;IVIMonly;measureD/f aftereachpartition',
        'dimensions': feature.DIMENSIONS, 'combinations': feature.combinations63(), 'dimension_count_histogram': [6, 15, 20, 15, 6, 1],
        'samples': specs, 'overlap': overlap, 'normalizations': NORMS, 'K': 3,
        'image_normalization': 'unchanged per-patient P1/P99 clipped Zscore andmedian/MAD RobustZ; saved fallback rules',
        'clustering': 'patient-local current-mask intensity/local3mm mean/SD;selectedcolumns only;KMeans3 seed64212 n_init5 sampledmax4000;training outcome neverused',
        '3mm_definition': '3x3x3 voxel box on verified1mm isotropic grid;support-normalized within currentvalid ROI;population localSD',
        'excluded_dimensions': 'only specifiedcolumns enter clustering;canonical labelnames use includedcenters;label-invariant downstreamfeatures',
        'measurements': 'D/f both measured forall63combinations;17prespecified PyRadiomics metrics plus same raw/spatial descriptors;156 label-invariant features',
        'not_D_only_prediction': 'D-only clustering input combinations stillmeasure f, andf-only clustering combinations stillmeasureD',
        'texture': 'same exact PyRadiomics formulas/binWidth.25;direct cropped featureclass calls verified against full extractor',
        'quality': 'unchanged train-fold completeness>=.9 ICC>=.8,nonconstant,Association20bootstrap orStability rank,Spearman<.95',
        'preprocessing': 'train-fold medianfill P2.5/P97.5 clipping,median/IQR scale',
        'classifier': 'same L2 liblinear LR for allcombinations;inner5 selects ranking,top10/20/all,C.003/.03/.3/3',
        'candidate_configs_per_norm': options('RobustZ'), 'objective': 'inner meanAUC-.10SD-.30max(meanfit-meanAUC-.10,0)',
        'CV': 'fixed sharedouter5seed42,inner62201..62205,final62212 within eachsample;no supervisedsharing acrosssamples',
        'adaptive_combo': 'eachouter training inner5 additionallyselects combo;full70 traininginner5 fixes finalcombo beforetests;normspecific andjointnorm aliases',
        'freeze': 'finish andverify all252 fixed-combo trainingmodels andall adaptivechoices before any testprediction',
        'tests': 'score each fixedfrozen model onceon own30;deriveadaptivealiases without additional calls;all63 results exploratory',
        'threshold': 'configuration-specific final70 training OOFYouden frozen beforetest',
        'bootstrap': '5000 stratified patientdraws;pairedsamecaseswithin sample;two samples overlap so not independent external validations',
        'clinical': False, 'T2_ADC': False, 'Dstar': False, 'validity_and_ROI': 'unchanged original D/f validity D>0,0<=f<.7,NRMSE<=.15;existing1mmerode/dilate',
        'historical_exploration': True, 'formal_model_use': False, 'automatic_adoption': False,
        'protected_hashes': {p.replace(chr(92), '/'): h for p, h in protected.items()},
        'source_hashes': {str(p.relative_to(ROOT)): digest(p) for p in modules}})
    save(OUT/'definition_frozen.json', {'passed': True, 'files': {str(p.relative_to(OUT)): digest(p) for p in OUT.rglob('*') if p.is_file()},
        'models_fitted': 0, 'AUC_used_for_sampling': False})
    print(pd.DataFrame(counts).to_string(index=False), flush=True)
    print('Overlap '+json.dumps(overlap), flush=True)


def protect():
    p = json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    for field in ('protected_hashes', 'source_hashes'):
        for name, sha in p[field].items():
            assert digest(ROOT/name) == sha, name
    fr = json.loads((OUT/'definition_frozen.json').read_text(encoding='utf-8'))
    for name, sha in fr['files'].items():
        assert digest(OUT/name) == sha, name
    return len(p['protected_hashes'])


def patient_path(index):
    return OUT/'patient_features'/('case_%03d.joblib'%int(index))


def data(indices, combo):
    rows = [joblib.load(patient_path(i)) for i in indices]
    assert all(r['index'] == int(i) for i, r in zip(indices, rows))
    meta = [{k: v for k, v in m.items() if k not in ('source_indices', 'statistic')} for m in feature.aggregation()]
    return {(norm, 'Invariant'): {'indices': np.asarray(indices, int), 'metadata': meta,
        **{kind: np.asarray([r['values'][(norm, combo)][kind] for r in rows]) for kind, _ in feature.KINDS}} for norm in NORMS}


def verify_definition():
    protect()
    cohort, specs = specifications()
    assert json.loads((OUT/'sample_manifest.json').read_text(encoding='utf-8')) == specs
    assert len(feature.combinations63()) == 63
    checked = 0
    for group in GROUPS:
        _, y, tr, te = inputs(group)
        assert len(tr) == 70 and len(te) == 30 and not set(tr)&set(te) and (y[te] == -1).all()
        for which, outer in [('outer', None), ('final', None)]+[('inner', i) for i in range(1, 6)]:
            validation = []
            for a, b in folds(group, which, outer):
                assert not (set(a)|set(b))&set(te) and set(y[a]) == set(y[b]) == {0, 1}
                validation.extend(b)
                checked += 1
            expected = tr if which != 'inner' else folds(group, 'outer')[outer-1][0]
            np.testing.assert_array_equal(np.sort(validation), np.sort(expected))
    save(OUT/'definition_verification.json', {'passed': True, 'folds_checked': checked, 'all63_unique_nonempty_subsets': True,
        'sample_and_splits_regenerated': True, 'train_test_overlap_within_group': 0, 'sampling_feature_or_AUC_selection': False})
