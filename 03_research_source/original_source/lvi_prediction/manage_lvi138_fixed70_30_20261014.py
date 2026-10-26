# -*- coding: utf-8 -*-
"""Authorized successor split97/41; immutable history and no AUC-based selection."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
import define_lvi138_two_test_sets_20261014 as previous_split
import run_regional_summary77_two_tests_20261014 as previous_run

ROOT = previous_split.ROOT
OUT = ROOT/'lvi_prediction/splits/lvi138_train97_test41_20261014'
ACTIVE = ROOT/'lvi_prediction/LVI_ACTIVE_DEVELOPMENT_SPLIT.json'
SEED = 20261014
digest, save = previous_split.digest, previous_split.save


def protect():
    p = json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    for field in ('protected_hashes', 'source_hashes'):
        for name, sha in p[field].items():
            assert digest(ROOT/name) == sha, name
    return len(p['protected_hashes'])


def define():
    assert not OUT.exists(), 'Do not replace this prespecified split'
    assert not ACTIVE.exists(), 'An active successor already exists; inspect before replacing'
    previous_run.frozen()
    cohort = previous_split.cohort()
    inherited = json.loads((previous_run.OUT/'protocol.json').read_text(encoding='utf-8'))
    hashes = {**inherited['protected_hashes'], **inherited['source_hashes']}
    hashes.update({str(p.relative_to(ROOT)): digest(p) for p in previous_run.OUT.rglob('*') if p.is_file()})
    hashes.update(json.loads((previous_run.OUT/'delivery_manifest.json').read_text(encoding='utf-8'))['source_hashes'])
    result_pointer = ROOT/'lvi_prediction/CURRENT_LVI_REGIONAL_SUMMARY77_RESULT_20261014.json'
    hashes[str(result_pointer.relative_to(ROOT))] = digest(result_pointer)
    protected = {}
    for name, sha in hashes.items():
        canonical = name.replace('\\', '/')
        assert canonical not in protected or protected[canonical] == sha
        protected[canonical] = sha
    truth = cohort.Outcome.to_numpy(int)
    tr, te = train_test_split(np.arange(138), train_size=97, test_size=41, stratify=truth, random_state=SEED)
    tr, te = np.sort(tr), np.sort(te)
    groups = {'Train97': tr, 'Test41': te}
    oldrows = pd.read_csv(previous_split.OUT/'all138_group_assignment.csv', dtype={'SubjectID': str, 'PatientID': str}).set_index('AnalysisID').loc[cohort.AnalysisID]
    np.testing.assert_array_equal(oldrows.Outcome, truth)
    rows = cohort.copy()
    rows.insert(0, 'CohortIndex', np.arange(138))
    rows['Group'] = np.where(np.isin(np.arange(138), tr), 'Train97', 'Test41')
    rows['Previous77Group'] = oldrows.Group.to_numpy()
    rows['Previous90Group'] = oldrows.Previous90Group.to_numpy()
    rows['PreviouslyExploredPool'] = True
    OUT.mkdir(parents=True)
    shutil.copyfile(previous_split.CURRENT, OUT/'superseded_development_pointer.json')
    shutil.copyfile(result_pointer, OUT/'previous_experiment_result_pointer.json')
    rows.to_csv(OUT/'all138_group_assignment.csv', index=False, encoding='utf-8-sig')
    for group, ix in groups.items():
        rows.iloc[ix].to_csv(OUT/(group+'_cases.csv'), index=False, encoding='utf-8-sig')
    counts = pd.DataFrame([{'Group': group, **previous_split.counts(cohort, ix)} for group, ix in groups.items()])
    counts.to_csv(OUT/'group_counts.csv', index=False, encoding='utf-8-sig')
    transition = pd.crosstab(rows.Group, rows.Previous77Group).reindex(['Train97', 'Test41'])
    transition.to_csv(OUT/'previous77_to_current_groups.csv', encoding='utf-8-sig')
    pd.crosstab(rows.Group, rows.Previous90Group).reindex(['Train97', 'Test41']).to_csv(OUT/'previous90_exposure_counts.csv', encoding='utf-8-sig')
    manifest = {'date': '2026-10-14', 'authorization': '重新规划训练集和测试集，全部138例按7：3划分',
        'split_name': 'Train97_Test41', 'source_cohort_sha256': digest(previous_split.COHORT),
        'seed': SEED, 'algorithm': 'train_test_split indices,train_size97,test_size41,stratifyOutcome,random_state20261014',
        'integer_rounding': '138*.7=96.6; nearest integer training97, remaining41',
        'actual_training_fraction': 97/138, 'actual_test_fraction': 41/138,
        'outcome_stratified': True, 'counts': counts.to_dict('records'),
        'groups': {g: {'indices': ix.tolist(), 'ids': cohort.AnalysisID.iloc[ix].tolist()} for g, ix in groups.items()},
        'all138_id_order': cohort.AnalysisID.tolist(), 'distinct_partition_solutions': 1, 'AUC_selection': False,
        'no_second_test_group': True, 'external_validation': False, 'previous_pool_exploration': True,
        'supersedes_for_new_experiments': str(previous_split.OUT.relative_to(ROOT))}
    save(OUT/'split_manifest.json', manifest)
    save(OUT/'nested5_training_folds.json', previous_split.nested(cohort, tr))
    save(OUT/'protocol.json', {'date': '2026-10-14', 'authorization': manifest['authorization'], 'split': manifest,
        'future_workflow': 'Train97 only; outer5/inner5,final5 within97; freeze chosen pipeline andOOF threshold; one final Test41 stage',
        'scope': 'all138 now inone training/test pair; previous77/33/28 no longeractive for newexperiments',
        'test_not_used_for_tuning': True, 'features_predictions_AUC_not_used_to_select_split': True,
        'new97_training_required': True, 'old_models_not_relabelled_as_independent_new_test_models': True,
        'historical_results_and_adoption_unchanged': True, 'models_trained': 0, 'models_scored': 0,
        'active_registry': str(ACTIVE.relative_to(ROOT)),
        'old_split_pointer_preserved': str(previous_split.CURRENT.relative_to(ROOT)),
        'old_split_pointer_sha256': digest(previous_split.CURRENT),
        'reason_for_successor_registry': 'oldpointer is hashed by immutable77 experiment; retain it ascreation-time history anduse newregistry for latesthuman-selected roles',
        'protected_hashes': protected, 'source_hashes': {str(Path(__file__).resolve().relative_to(ROOT)): digest(__file__)}})
    save(OUT/'definition_frozen.json', {'passed': True, 'frozen_files': {p.name: digest(p) for p in OUT.iterdir() if p.is_file()},
        'source_sha256': digest(__file__), 'models_trained': 0, 'models_scored': 0, 'protected_count': protect()})
    print(counts.to_string(index=False), flush=True)
    print(transition.to_string(), flush=True)


def frozen():
    fr = json.loads((OUT/'definition_frozen.json').read_text(encoding='utf-8'))
    protect()
    for name, sha in fr['frozen_files'].items():
        assert digest(OUT/name) == sha, name
    return fr


def verify():
    frozen()
    data = previous_split.cohort()
    y = data.Outcome.to_numpy(int)
    tr, te = train_test_split(np.arange(138), train_size=97, test_size=41, stratify=y, random_state=SEED)
    tr, te = np.sort(tr), np.sort(te)
    rows = pd.read_csv(OUT/'all138_group_assignment.csv', dtype={'SubjectID': str, 'PatientID': str})
    manifest = json.loads((OUT/'split_manifest.json').read_text(encoding='utf-8'))
    assert len(rows) == rows.AnalysisID.nunique() == rows.SubjectID.nunique() == rows.PatientID.nunique() == 138
    assert not set(tr)&set(te) and set(tr)|set(te) == set(range(138))
    for name, ix in [('Train97', tr), ('Test41', te)]:
        selected = rows[rows.Group == name]
        np.testing.assert_array_equal(selected.CohortIndex, ix)
        np.testing.assert_array_equal(selected.AnalysisID, data.AnalysisID.iloc[ix])
        np.testing.assert_array_equal(selected.Outcome, y[ix])
        np.testing.assert_array_equal(manifest['groups'][name]['indices'], ix)
        saved = pd.read_csv(OUT/(name+'_cases.csv'), dtype={'SubjectID': str, 'PatientID': str})
        pd.testing.assert_frame_equal(selected.reset_index(drop=True), saved.reset_index(drop=True))
    nested = json.loads((OUT/'nested5_training_folds.json').read_text(encoding='utf-8'))
    checked = [0]
    def check(parent, records, seed):
        parent = np.asarray(parent, int)
        validation = []
        for fold, ((a, b), record) in enumerate(zip(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(parent)), y[parent]), records), 1):
            aa, bb = parent[a], parent[b]
            assert record['Fold'] == fold
            np.testing.assert_array_equal(record['train_indices'], aa)
            np.testing.assert_array_equal(record['validation_indices'], bb)
            np.testing.assert_array_equal(record['train_ids'], data.AnalysisID.iloc[aa])
            np.testing.assert_array_equal(record['validation_ids'], data.AnalysisID.iloc[bb])
            assert not set(aa)&set(bb) and set(aa)|set(bb) == set(parent)
            assert not (set(aa)|set(bb))&set(te) and set(y[aa]) == set(y[bb]) == {0, 1}
            assert record['train_counts'] == previous_split.counts(data, aa)
            assert record['validation_counts'] == previous_split.counts(data, bb)
            validation.extend(bb.tolist())
            checked[0] += 1
        assert len(records) == 5 and sorted(validation) == parent.tolist()
    check(tr, nested['outer_folds'], 42)
    for record in nested['outer_folds']:
        check(record['train_indices'], record['inner_folds'], 62200+record['Fold'])
    check(tr, nested['final_tuning_folds'], 62212)
    assert checked[0] == 35
    protocol = json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    assert digest(previous_split.CURRENT) == protocol['old_split_pointer_sha256']
    save(OUT/'split_verification.json', {'passed': True, 'N': 138, 'train_N': 97, 'test_N': 41,
        'training_positive': int(y[tr].sum()), 'test_positive': int(y[te].sum()), 'patient_overlap': 0,
        'all138_accounted_for': True, 'seed_reproduction_exact': True, 'nested_training_folds_verified': 35,
        'test_patients_in_CV': 0, 'old_pointer_unchanged': True, 'models_trained': 0, 'models_scored': 0,
        'AUC_used_to_choose_split': False, 'protected_count': protect()})
    print('97/41 split and35 train-only CVfolds verified', flush=True)


def deliver():
    frozen()
    assert json.loads((OUT/'split_verification.json').read_text(encoding='utf-8'))['passed']
    manifest = json.loads((OUT/'split_manifest.json').read_text(encoding='utf-8'))
    count = pd.read_csv(OUT/'group_counts.csv').set_index('Group')
    transition = pd.read_csv(OUT/'previous77_to_current_groups.csv').set_index('Group')
    overlap = int(transition.loc['Test41', 'Train77'])
    pointer = {'date': '2026-10-14', 'source': 'directlatesthumaninstruction:全部138例按7：3划分',
        'status': 'split_fixed_new97_model_not_yet_trained', 'cohort_N': 138,
        'split_directory': str(OUT.relative_to(ROOT)), 'split_manifest_sha256': digest(OUT/'split_manifest.json'),
        'protocol_sha256': digest(OUT/'protocol.json'), 'verification_sha256': digest(OUT/'split_verification.json'),
        'train_ids': manifest['groups']['Train97']['ids'], 'test_ids': manifest['groups']['Test41']['ids'],
        'counts': manifest['counts'], 'single_test_group': True, 'testA_and_testB_active': False,
        'external_validation': False, 'prior_pool_exploration': True, 'new97_training_required': True,
        'training_nested5': True, 'test41_final_only': True, 'supersedes': str(previous_split.CURRENT.relative_to(ROOT)),
        'old_pointer_remains_immutable_history': True, 'model_adoption_unchanged': True,
        'models_trained': 0, 'models_scored': 0}
    if ACTIVE.exists():
        assert json.loads(ACTIVE.read_text(encoding='utf-8')) == pointer
    else:
        save(ACTIVE, pointer)
    p = lambda name: str((OUT/name).resolve()).replace('\\', '/')
    report = OUT/'全部138例固定7比3划分_97训练41测试_20261014.md'
    text = '\n'.join(['# 全部138例重新固定7∶3划分', '', '日期：2026年10月14日。依据用户最新要求，将全部138例用于一套训练/测试划分，取代此前77/33/28的后续使用计划。', '',
        '| 用途 | 例数 | 阳性 | 阴性 | 占全部病例比例 |', '|---|---:|---:|---:|---:|',
        *[f'| {"训练集" if g == "Train97" else "测试集"} | {int(count.loc[g,"N"])} | {int(count.loc[g,"Positive"])} | {int(count.loc[g,"Negative"])} | {count.loc[g,"N"]/138:.1%} |' for g in ('Train97', 'Test41')], '',
        '138×70%＝96.6，按最接近的整数取97例训练、41例测试，实际比例70.3%∶29.7%。按结局分层，固定seed＝20261014；不参考预测概率或AUC挑选病例，没有搜索种子。三项患者标识各有138个唯一值，训练/测试交集为0，完整覆盖138例。', '',
        '## 后续流程', '',
        '1. 只使用97例进行特征筛选、预处理、降维、分类器与参数选择。',
        '2. 97例内沿用外层五折、内层五折；外层seed＝42，内层62201—62205；最后97例内再五折选最终方案（62212）。已保存35份训练/验证折，均排除41例测试。',
        '3. 在97例重新训练新模型，阈值取训练OOF，模型全部冻结后，41例进行一次最终测试。',
        '4. 此前两套测试A、B不再用于这轮新实验；历史名单、模型、结果保留。该改动只重规划数据角色，不改变正式采用结果。', '',
        f'新41例中，有{overlap}例曾参加上一轮77例模型训练，不能直接用旧模型评分后称为独立测试。因此后续须在97例上重新开发模型。138例此前已有多轮探索，本轮为内部探索性划分，分组改变不消除历史开发暴露。', '',
        '## 文件', '',
        f'- [138例完整分组名单]({p("all138_group_assignment.csv")})',
        f'- [97例训练名单]({p("Train97_cases.csv")})',
        f'- [41例测试名单]({p("Test41_cases.csv")})',
        f'- [嵌套五折病例安排]({p("nested5_training_folds.json")})',
        f'- [固定随机种子及分组记录]({p("split_manifest.json")})', '',
        '后续新实验以LVI_ACTIVE_DEVELOPMENT_SPLIT.json为当前分组入口。旧CURRENT_LVI_DEVELOPMENT_SPLIT_20261014.json因被历史冻结实验引用，保留原字节作为历史记录；其快照也保存在本目录superseded_development_pointer.json。旧实验仍可按原哈希核验。', '',
        '当前状态：分组与35份嵌套折已核验，未训练新97例模型、未计算新的测试AUC。', ''])
    report.write_text(text, encoding='utf-8')
    assert '\ufffd' not in text
    outputs = {str(x.relative_to(OUT)): digest(x) for x in OUT.rglob('*') if x.is_file() and x.name != 'delivery_manifest.json'}
    protocol = json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    distinct = len({str((ROOT/x).resolve()).casefold() for x in protocol['protected_hashes']})
    protect()
    save(OUT/'delivery_manifest.json', {'passed': True, 'outputs': outputs, 'output_count': len(outputs),
        'source_sha256': digest(__file__), 'report': report.name, 'report_sha256': digest(report),
        'active_registry_sha256': digest(ACTIVE), 'historical_distinct_files_unchanged': distinct,
        'old_split_pointer_unchanged': True, 'latest_split_active': '97/41', 'model_fit_calls': 0, 'model_predict_calls': 0})
    print('97/41 delivered: %d outputs; %d historical files unchanged'%(len(outputs), distinct), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['define', 'verify', 'deliver'], required=True)
    {'define': define, 'verify': verify, 'deliver': deliver}[parser.parse_args().phase]()
