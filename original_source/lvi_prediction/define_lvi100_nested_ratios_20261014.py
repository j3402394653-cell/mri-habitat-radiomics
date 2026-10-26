# -*- coding: utf-8 -*-
"""Role-preserving outcome-stratified downsample; no feature/AUC-based selection."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import run_regional_summary_ratio_comparison_20261014 as baseline
import define_lvi138_two_test_sets_20261014 as source

ROOT = baseline.ROOT
OUT = ROOT/'lvi_prediction/splits/lvi100_downsample_nested_ratios_20261014'
SEED = 2026101415
save, digest = baseline.save, baseline.digest


def sampling():
    cohort, y, tr7, te7, _ = baseline.inputs('70_30')
    tr8, te8 = baseline.inputs('80_20')[2:4]
    truth = cohort.Outcome.to_numpy(int)
    extra = np.asarray(sorted(set(tr8)-set(tr7)), int)
    assert len(tr7) == 97 and len(extra) == 13 and len(te8) == 28
    selected = {}
    for offset, (name, original, n) in enumerate([('Train70', tr7, 70), ('ExtraTraining10', extra, 10), ('CommonTest20', te8, 20)]):
        kept, excluded = train_test_split(original, train_size=n, test_size=len(original)-n,
                                           stratify=truth[original], random_state=SEED+offset)
        selected[name] = np.sort(kept)
    tr70, extra10, te20 = [selected[g] for g in ('Train70', 'ExtraTraining10', 'CommonTest20')]
    parts = {'70_30': {'Train70': tr70, 'Test30': np.sort(np.r_[extra10, te20])},
             '80_20': {'Train80': np.sort(np.r_[tr70, extra10]), 'Test20': te20}}
    sample = np.sort(np.r_[tr70, extra10, te20])
    assert len(sample) == len(set(sample)) == 100
    assert set(parts['70_30']['Train70']) <= set(tr7) and set(parts['70_30']['Test30']) <= set(te7)
    assert set(parts['80_20']['Train80']) <= set(tr8) and set(te20) <= set(te8)
    return cohort, sample, parts, selected


def protect():
    protocol = json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    for field in ('protected_hashes', 'source_hashes'):
        for path, sha in protocol[field].items():
            assert digest(ROOT/path) == sha, path
    return len(protocol['protected_hashes'])


def define():
    assert not OUT.exists(), 'Never replace sample or search seeds by AUC'
    baseline.frozen()
    assert json.loads((baseline.OUT/'delivery_manifest.json').read_text(encoding='utf-8'))['passed']
    cohort, sample, parts, selected = sampling()
    parent = json.loads((baseline.OUT/'protocol.json').read_text(encoding='utf-8'))
    protected = {**parent['protected_hashes'], **parent['source_hashes']}
    protected.update({str(p.relative_to(ROOT)): digest(p) for p in baseline.OUT.rglob('*') if p.is_file()})
    protected.update(json.loads((baseline.OUT/'delivery_manifest.json').read_text(encoding='utf-8'))['source_hashes'])
    pointer = ROOT/'lvi_prediction/CURRENT_LVI_REGIONAL_SUMMARY_RATIO_RESULT_20261014.json'
    protected[str(pointer.relative_to(ROOT))] = digest(pointer)
    OUT.mkdir(parents=True)
    rows = cohort.copy()
    rows.insert(0, 'CohortIndex', np.arange(138))
    rows['Selected100'] = np.isin(np.arange(138), sample)
    for ratio, groups in parts.items():
        labels = np.full(138, 'Unused38', dtype=object)
        for name, ix in groups.items():
            labels[ix] = name
        rows['Group'+ratio] = labels
    rows.to_csv(OUT/'all138_assignment.csv', index=False, encoding='utf-8-sig')
    rows.iloc[sample].to_csv(OUT/'selected100_cases.csv', index=False, encoding='utf-8-sig')
    unused = np.asarray(sorted(set(range(138))-set(sample)), int)
    rows.iloc[unused].to_csv(OUT/'unused38_cases.csv', index=False, encoding='utf-8-sig')
    counts = []
    specs = {}
    for ratio, groups in parts.items():
        dest = OUT/ratio
        dest.mkdir()
        tr = next(ix for name, ix in groups.items() if name.startswith('Train'))
        for name, ix in groups.items():
            counts.append({'Ratio': ratio, 'Group': name, **source.counts(cohort, ix)})
            rows.iloc[ix].to_csv(dest/(name+'_cases.csv'), index=False, encoding='utf-8-sig')
        manifest = {'date': '2026-10-14', 'source_cohort_sha256': digest(source.COHORT), 'sample_N': 100,
            'sample_indices': sample.tolist(), 'sample_ids': cohort.AnalysisID.iloc[sample].tolist(),
            'ratio': ratio, 'groups': {name: {'indices': ix.tolist(), 'ids': cohort.AnalysisID.iloc[ix].tolist()} for name, ix in groups.items()},
            'all138_id_order': cohort.AnalysisID.tolist(), 'external_validation': False,
            'role_preserving_downsample': True, 'previous_pool_exploration': True, 'AUC_used_for_sampling': False}
        save(dest/'split_manifest.json', manifest)
        save(dest/'nested5_training_folds.json', source.nested(cohort, tr))
        specs[ratio] = {'directory': str(dest.relative_to(ROOT)), 'manifest_sha256': digest(dest/'split_manifest.json')}
    pd.DataFrame(counts).to_csv(OUT/'group_counts.csv', index=False, encoding='utf-8-sig')
    save(OUT/'sampling_manifest.json', {'date': '2026-10-14', 'authorization': '把总数据下降到100例试试',
        'sampling_type': 'role-preserving outcome-stratified random downsample within existing97/13/28 roles',
        'simple_random_sample_of138': False, 'selected_count': 100, 'selected_counts': source.counts(cohort, sample),
        'seeds': {g: SEED+i for i, g in enumerate(selected)},
        'role_quotas': {'old_train97': 70, 'old_extra_training13': 10, 'old_common_test28': 20},
        'selection_uses': ['previous fixed split membership', 'Outcome for stratification only', 'prespecified seeds'],
        'selection_does_not_use': ['features', 'prediction scores', 'AUC', 'classification difficulty'],
        'selected_indices': sample.tolist(), 'unused_indices': unused.tolist(),
        'selected_roles': {g: ix.tolist() for g, ix in selected.items()}, 'schemes': specs,
        'group_counts': counts, 'common_test_indices': selected['CommonTest20'].tolist(),
        'samecase_historical_comparison': 'subset saved138-model holdout scores to new own30/20 andcommon20; no old-model calls',
        'training_nesting': 'new70 subsetold97;new80 subsetold110',
        'interpretation': 'single random role-stratified reduction; training patient composition andN both change; no universal sample-size causal inference',
        'unused38_scored': False, 'models_fitted': 0, 'model_predict_calls': 0, 'AUC_computed': 0})
    save(OUT/'protocol.json', {'date': '2026-10-14', 'sampling_manifest_sha256': digest(OUT/'sampling_manifest.json'),
        'frozen_before_any_new_training': True, 'feature_or_AUC_case_selection': False, 'formal_use': False,
        'protected_hashes': {p.replace('\\', '/'): h for p, h in protected.items()},
        'source_hashes': {str(Path(__file__).resolve().relative_to(ROOT)): digest(__file__)}})
    save(OUT/'definition_frozen.json', {'passed': True, 'files': {str(p.relative_to(OUT)): digest(p) for p in OUT.rglob('*') if p.is_file()},
        'model_predict_calls': 0, 'model_fit_calls': 0, 'protected_count': protect()})
    print(pd.DataFrame(counts).to_string(index=False))
    print('Selected100 positive%d negative%d; unused38'% (int(cohort.Outcome.iloc[sample].sum()), int((cohort.Outcome.iloc[sample] == 0).sum())))


def frozen():
    protect()
    fr = json.loads((OUT/'definition_frozen.json').read_text(encoding='utf-8'))
    for path, sha in fr['files'].items():
        assert digest(OUT/path) == sha, path
    return fr


def verify():
    frozen()
    cohort, sample, parts, selected = sampling()
    manifest = json.loads((OUT/'sampling_manifest.json').read_text(encoding='utf-8'))
    np.testing.assert_array_equal(manifest['selected_indices'], sample)
    rows = pd.read_csv(OUT/'all138_assignment.csv')
    np.testing.assert_array_equal(rows.Selected100, np.isin(np.arange(138), sample))
    checked = 0
    for ratio, groups in parts.items():
        tr, te = list(groups.values())
        assert len(tr)+len(te) == 100 and not set(tr)&set(te)
        saved = json.loads((OUT/ratio/'split_manifest.json').read_text(encoding='utf-8'))
        for name, ix in groups.items():
            np.testing.assert_array_equal(saved['groups'][name]['indices'], ix)
        cv = json.loads((OUT/ratio/'nested5_training_folds.json').read_text(encoding='utf-8'))
        for phase, foldrows, allowed in [('outer', cv['outer_folds'], tr), ('final', cv['final_tuning_folds'], tr)]+[
            ('inner', row['inner_folds'], np.asarray(row['train_indices'])) for row in cv['outer_folds']]:
            validation = []
            for row in foldrows:
                a, b = [np.asarray(row[k], int) for k in ('train_indices', 'validation_indices')]
                assert not set(a)&set(b) and set(a)|set(b) == set(allowed) and not (set(a)|set(b))&set(te)
                assert set(cohort.Outcome.iloc[a]) == set(cohort.Outcome.iloc[b]) == {0, 1}
                validation.extend(b)
                checked += 1
            np.testing.assert_array_equal(np.sort(validation), np.sort(allowed))
    assert checked == 70
    save(OUT/'split_verification.json', {'passed': True, 'sample_N': 100, 'unique_selected_ids': 100,
        'unused_N': 38, 'train_test_overlap': 0, 'all70_CV_folds_validated': True,
        'downsample_regenerated_exactly': True, 'role_preserving_training_and_test_subsets': True,
        'feature_or_AUC_case_selection': False, 'models_fitted': 0, 'protected_count': protect()})
    print('100sample andboth70/30,80/20 nestedfold definitions verified')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['define', 'verify'], required=True)
    args = parser.parse_args()
    {'define': define, 'verify': verify}[args.phase]()
