# -*- coding: utf-8 -*-
"""Locked7:3 and8:2 dataset controls, with shared28 case-matched evaluation design."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
import manage_lvi138_fixed70_30_20261014 as baseline

ROOT = baseline.ROOT
EIGHT = ROOT/'lvi_prediction/splits/lvi138_train110_test28_20261014'
PAIR = ROOT/'lvi_prediction/splits/lvi138_70_30_vs80_20_20261014'
ACTIVE = ROOT/'lvi_prediction/LVI_ACTIVE_SPLIT_COMPARISON.json'
SEED = 20261014
digest, save = baseline.digest, baseline.save


def inputs():
    data = baseline.previous_split.cohort()
    seven = json.loads((baseline.OUT/'split_manifest.json').read_text(encoding='utf-8'))
    tr7, te7 = [np.asarray(seven['groups'][g]['indices'], int) for g in ('Train97', 'Test41')]
    return data, seven, tr7, te7


def protect():
    protocol = json.loads((PAIR/'protocol.json').read_text(encoding='utf-8'))
    for field in ('protected_hashes', 'source_hashes'):
        for name, sha in protocol[field].items():
            assert digest(ROOT/name) == sha, name
    return len(protocol['protected_hashes'])


def define():
    assert not EIGHT.exists() and not PAIR.exists() and not ACTIVE.exists(), 'Do not replace fixedcomparison roles'
    baseline.frozen()
    assert json.loads((baseline.OUT/'split_verification.json').read_text())['passed']
    data, seven, tr7, te7 = inputs()
    y = data.Outcome.to_numpy(int)
    tr8, te8 = train_test_split(np.arange(138), train_size=110, test_size=28, stratify=y, random_state=SEED)
    tr8, te8 = np.sort(tr8), np.sort(te8)
    assert set(tr7) <= set(tr8) and set(te8) <= set(te7)
    extra = np.asarray(sorted(set(tr8)-set(tr7)), int)
    assert len(extra) == 13 and set(extra) == set(te7)-set(te8)
    inherited = json.loads((baseline.OUT/'protocol.json').read_text(encoding='utf-8'))
    protected = {**inherited['protected_hashes'], **inherited['source_hashes']}
    protected.update({str(p.relative_to(ROOT)): digest(p) for p in baseline.OUT.rglob('*') if p.is_file()})
    protected[str(baseline.ACTIVE.relative_to(ROOT))] = digest(baseline.ACTIVE)
    protected = {name.replace('\\', '/'): sha for name, sha in protected.items()}
    EIGHT.mkdir(parents=True)
    PAIR.mkdir(parents=True)
    rows = pd.read_csv(baseline.OUT/'all138_group_assignment.csv', dtype={'SubjectID': str, 'PatientID': str})
    np.testing.assert_array_equal(rows.AnalysisID, data.AnalysisID)
    rows = rows.rename(columns={'Group': 'Group70_30'})
    rows['Group80_20'] = np.where(np.isin(np.arange(138), tr8), 'Train110', 'Test28')
    rows['CommonTest28'] = np.isin(np.arange(138), te8)
    rows['ExtraTraining13In80_20'] = np.isin(np.arange(138), extra)
    rows.to_csv(PAIR/'all138_split_comparison.csv', index=False, encoding='utf-8-sig')
    eightrows = rows.copy()
    eightrows['Group'] = rows.Group80_20
    eightrows.to_csv(EIGHT/'all138_group_assignment.csv', index=False, encoding='utf-8-sig')
    groups = {'Train110': tr8, 'Test28': te8}
    for group, ix in groups.items():
        eightrows.iloc[ix].to_csv(EIGHT/(group+'_cases.csv'), index=False, encoding='utf-8-sig')
    rows.iloc[te8].to_csv(PAIR/'common_test28_cases.csv', index=False, encoding='utf-8-sig')
    rows.iloc[extra].to_csv(PAIR/'extra13_training_cases_in80_20.csv', index=False, encoding='utf-8-sig')
    counts = pd.DataFrame([{'Scheme': scheme, 'Group': group, **baseline.previous_split.counts(data, ix)}
        for scheme, parts in [('70_30', {'Train97': tr7, 'Test41': te7}), ('80_20', groups)] for group, ix in parts.items()])
    counts.to_csv(PAIR/'group_counts_comparison.csv', index=False, encoding='utf-8-sig')
    counts[counts.Scheme == '80_20'].drop(columns='Scheme').to_csv(EIGHT/'group_counts.csv', index=False, encoding='utf-8-sig')
    pd.crosstab(rows.Group70_30, rows.Group80_20).to_csv(PAIR/'between_split_membership.csv', encoding='utf-8-sig')
    eight = {'date': '2026-10-14', 'authorization': '再把138例按8：2划分，把7：3和8：2作为对照',
        'split_name': 'Train110_Test28', 'source_cohort_sha256': digest(baseline.previous_split.COHORT),
        'seed': SEED, 'algorithm': 'train_test_split indices,train_size110,test_size28,stratifyOutcome,random_state20261014',
        'integer_rounding': '138*.8=110.4; nearest integer training110, remaining28',
        'actual_training_fraction': 110/138, 'actual_test_fraction': 28/138,
        'outcome_stratified': True, 'counts': counts[counts.Scheme == '80_20'].drop(columns='Scheme').to_dict('records'),
        'groups': {g: {'indices': ix.tolist(), 'ids': data.AnalysisID.iloc[ix].tolist()} for g, ix in groups.items()},
        'all138_id_order': data.AnalysisID.tolist(), 'AUC_selection': False, 'external_validation': False,
        'previous_pool_exploration': True, 'seven_split_preserved_as_control': True}
    save(EIGHT/'split_manifest.json', eight)
    save(EIGHT/'nested5_training_folds.json', baseline.previous_split.nested(data, tr8))
    pair = {'date': '2026-10-14', 'authorization': eight['authorization'],
        'schemes': {'70_30': {'directory': str(baseline.OUT.relative_to(ROOT)), 'train_N': 97, 'test_N': 41,
                             'manifest_sha256': digest(baseline.OUT/'split_manifest.json')},
                    '80_20': {'directory': str(EIGHT.relative_to(ROOT)), 'train_N': 110, 'test_N': 28,
                             'manifest_sha256': digest(EIGHT/'split_manifest.json')}},
        'nested_train_sets': True, 'nested_test_sets': True, 'common_test_indices': te8.tolist(),
        'common_test_ids': data.AnalysisID.iloc[te8].tolist(), 'common_test_counts': baseline.previous_split.counts(data, te8),
        'extra13_training_indices': extra.tolist(), 'extra13_training_ids': data.AnalysisID.iloc[extra].tolist(),
        'extra13_counts': baseline.previous_split.counts(data, extra),
        'model_counts_at_split_definition': {'fitted': 0, 'scored': 0}, 'AUC_used_for_case_selection': False,
        'historically_same_as_old_testB28': True, 'external_validation': False}
    save(PAIR/'comparison_manifest.json', pair)
    save(PAIR/'protocol.json', {'date': '2026-10-14', 'comparison': pair,
        'same_pipeline': 'same feature definition/normalizations/K,quality andpreprocessing rules,modelcandidate space andselection objective forboth ratios',
        'training': 'separate97 and110 outer5/inner5/final5; fit each transformation andchoice onlyinside currentratio training fold',
        'no_shared_outcome_driven_feature_ranking_across_ratios': True,
        'matched_template_vs_retuned': 'if both recent strategies are studied, keep theirdefinitions andcandidate spaces identical acrossratios andreportseparately',
        'freeze': 'finish andverify all requestedmodels inboth ratios,freezebefore any newtestscore',
        'own_holdout_report': '70/30 model testedon41;80/20 model testedon28; descriptive differences include bothtrainingN andtestcasechanges',
        'common_holdout_report': 'both models comparedonexactsame28,excludedfromboth training sets; paired patientbootstrap forDeltaAUC',
        'seven_common28_probabilities': 'subsetofsaved41 predictions, no extra modelcall',
        'eight_evaluation_on_extra13': 'not allowed asindependent test:these13 aretraining cases under80/20',
        'no_model_or_split_adoption_by_highest_test_AUC': True,
        'each_ratio_uses_both_classes_in_35_saved_CV_folds': True,
        'previous_pool_exploration_disclosed': True, 'formal_model_use_changed': False,
        'active_comparison_registry': str(ACTIVE.relative_to(ROOT)),
        'seven_single_split_registry_preserved': str(baseline.ACTIVE.relative_to(ROOT)),
        'protected_hashes': protected, 'source_hashes': {str(Path(__file__).resolve().relative_to(ROOT)): digest(__file__)}})
    save(EIGHT/'protocol.json', {'date': '2026-10-14', 'split': eight, 'comparison_control': 'fixed70/30 unchanged',
        'protection_protocol': str((PAIR/'protocol.json').relative_to(ROOT)), 'protection_protocol_sha256': digest(PAIR/'protocol.json'),
        'model_fit_calls': 0, 'model_predict_calls': 0})
    save(EIGHT/'definition_frozen.json', {'passed': True, 'frozen_files': {p.name: digest(p) for p in EIGHT.iterdir() if p.is_file()},
        'source_sha256': digest(__file__), 'model_fit_calls': 0, 'model_predict_calls': 0})
    files = {str(p.relative_to(ROOT)): digest(p) for directory in (EIGHT, PAIR) for p in directory.iterdir() if p.is_file()}
    save(PAIR/'definition_frozen.json', {'passed': True, 'frozen_files': files, 'protected_count': protect(),
        'model_fit_calls': 0, 'model_predict_calls': 0})
    print(counts.to_string(index=False), flush=True)
    print('Sharedtest28; extra13 addedtotraining under80/20; no AUC ormodelfits', flush=True)


def frozen():
    fr = json.loads((PAIR/'definition_frozen.json').read_text(encoding='utf-8'))
    protect()
    for name, sha in fr['frozen_files'].items():
        assert digest(ROOT/name) == sha, name
    return fr


def verify():
    frozen()
    data, seven, tr7, te7 = inputs()
    y = data.Outcome.to_numpy(int)
    tr8, te8 = train_test_split(np.arange(138), train_size=110, test_size=28, stratify=y, random_state=SEED)
    tr8, te8 = np.sort(tr8), np.sort(te8)
    eight = json.loads((EIGHT/'split_manifest.json').read_text(encoding='utf-8'))
    pair = json.loads((PAIR/'comparison_manifest.json').read_text(encoding='utf-8'))
    rows = pd.read_csv(PAIR/'all138_split_comparison.csv', dtype={'SubjectID': str, 'PatientID': str})
    assert len(rows) == rows.AnalysisID.nunique() == rows.SubjectID.nunique() == rows.PatientID.nunique() == 138
    np.testing.assert_array_equal(rows.AnalysisID, data.AnalysisID)
    np.testing.assert_array_equal(rows.Outcome, y)
    for g, ix in [('Train110', tr8), ('Test28', te8)]:
        np.testing.assert_array_equal(eight['groups'][g]['indices'], ix)
        np.testing.assert_array_equal(eight['groups'][g]['ids'], data.AnalysisID.iloc[ix])
        saved = pd.read_csv(EIGHT/(g+'_cases.csv'), dtype={'SubjectID': str, 'PatientID': str})
        np.testing.assert_array_equal(saved.CohortIndex, ix)
        np.testing.assert_array_equal(saved.AnalysisID, data.AnalysisID.iloc[ix])
    assert not set(tr7)&set(te7) and not set(tr8)&set(te8)
    assert set(tr7)|set(te7) == set(tr8)|set(te8) == set(range(138))
    assert set(tr7) < set(tr8) and set(te8) < set(te7)
    assert len(set(tr8)-set(tr7)) == 13
    np.testing.assert_array_equal(pair['common_test_indices'], te8)
    np.testing.assert_array_equal(rows.CommonTest28, np.isin(np.arange(138), te8))
    np.testing.assert_array_equal(rows.ExtraTraining13In80_20, np.isin(np.arange(138), np.asarray(pair['extra13_training_indices'])))
    common = pd.read_csv(PAIR/'common_test28_cases.csv')
    np.testing.assert_array_equal(common.AnalysisID, data.AnalysisID.iloc[te8])
    oldB = json.loads((baseline.previous_split.OUT/'split_manifest.json').read_text(encoding='utf-8'))['groups']['TestB28']['indices']
    np.testing.assert_array_equal(te8, oldB)
    checked = [0]
    def check(parent, records, seed, forbidden):
        parent = np.asarray(parent, int)
        validation = []
        for (a, b), record in zip(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(parent)), y[parent]), records):
            aa, bb = parent[a], parent[b]
            np.testing.assert_array_equal(record['train_indices'], aa)
            np.testing.assert_array_equal(record['validation_indices'], bb)
            np.testing.assert_array_equal(record['train_ids'], data.AnalysisID.iloc[aa])
            np.testing.assert_array_equal(record['validation_ids'], data.AnalysisID.iloc[bb])
            assert not set(aa)&set(bb) and set(aa)|set(bb) == set(parent)
            assert not (set(aa)|set(bb))&set(forbidden) and set(y[aa]) == set(y[bb]) == {0, 1}
            assert record['train_counts'] == baseline.previous_split.counts(data, aa)
            assert record['validation_counts'] == baseline.previous_split.counts(data, bb)
            validation.extend(bb.tolist())
            checked[0] += 1
        assert len(records) == 5 and sorted(validation) == parent.tolist()
    for directory, train, test in [(baseline.OUT, tr7, te7), (EIGHT, tr8, te8)]:
        nested = json.loads((directory/'nested5_training_folds.json').read_text(encoding='utf-8'))
        check(train, nested['outer_folds'], 42, test)
        for record in nested['outer_folds']:
            check(record['train_indices'], record['inner_folds'], 62200+record['Fold'], test)
        check(train, nested['final_tuning_folds'], 62212, test)
    assert checked[0] == 70
    save(PAIR/'comparison_verification.json', {'passed': True, 'unique_patients': 138,
        'own_train_test_overlap': 0, 'train97_nested_inside110': True, 'test28_nested_inside41': True,
        'common_test_N': 28, 'common_test_positive': int(y[te8].sum()), 'common_test_negative': int((y[te8] == 0).sum()),
        'extra_training_N': 13, 'nested_CV_folds_checked': 70, 'test_patients_in_own_CV': 0,
        'same_as_historical_testB28_disclosed': True, 'model_fit_calls': 0, 'model_predict_calls': 0, 'AUC_selection': False,
        'protected_count': protect()})
    save(EIGHT/'split_verification.json', {'passed': True, 'train_N': 110, 'test_N': 28,
        'train_positive': int(y[tr8].sum()), 'test_positive': int(y[te8].sum()), 'patient_overlap': 0,
        'seed_reproduction_exact': True, 'own_CV_folds_verified': 35, 'test_in_training_CV': 0,
        'comparison_verification_sha256': digest(PAIR/'comparison_verification.json')})
    print('Both controls verified:138 unique cases,common28,extra13,70 train-only CVfolds', flush=True)


def deliver():
    frozen()
    assert json.loads((PAIR/'comparison_verification.json').read_text(encoding='utf-8'))['passed']
    counts = pd.read_csv(PAIR/'group_counts_comparison.csv')
    pair = json.loads((PAIR/'comparison_manifest.json').read_text(encoding='utf-8'))
    pointer = {'date': '2026-10-14', 'authorization': pair['authorization'], 'status': 'both_split_controls_fixed_models_not_yet_trained',
        'comparison_directory': str(PAIR.relative_to(ROOT)), 'comparison_manifest_sha256': digest(PAIR/'comparison_manifest.json'),
        'verification_sha256': digest(PAIR/'comparison_verification.json'), 'schemes': pair['schemes'],
        'counts': counts.to_dict('records'), 'common_test_ids': pair['common_test_ids'],
        'own_holdout_and_common28_results_required': True, 'both_schemes_active_as_controls': True,
        'prior77_33_28_remains_history': True, 'seven_single_split_registry_unchanged': True,
        'new97_and110_training_required': True, 'same_pipeline_across_ratios': True,
        'all_models_frozen_before_any_new_test_scoring': True, 'formal_model_adoption_unchanged': True,
        'models_trained': 0, 'models_scored': 0, 'external_validation': False, 'previous_pool_exploration': True}
    if ACTIVE.exists():
        assert json.loads(ACTIVE.read_text(encoding='utf-8')) == pointer
    else:
        save(ACTIVE, pointer)
    p = lambda directory, name: str((directory/name).resolve()).replace('\\', '/')
    report = PAIR/'138例7比3与8比2固定划分对照_20261014.md'
    text = '\n'.join(['# 138例：固定7∶3与8∶2划分对照', '', '日期：2026年10月14日。保留已固定的7∶3名单，新增8∶2名单，两套同时作为后续新实验的数据划分对照。', '',
        '| 划分 | 用途 | 例数 | 阳性 | 阴性 |', '|---|---|---:|---:|---:|',
        *[f'| {"7∶3" if r.Scheme == "70_30" else "8∶2"} | {"训练" if r.Group.startswith("Train") else "测试"} | {r.N} | {r.Positive} | {r.Negative} |' for _, r in counts.iterrows()], '',
        '7∶3为97例训练/41例测试，实际70.3%/29.7%；8∶2为110例训练/28例测试，实际79.7%/20.3%。均按最接近训练比例的整数取整。相同seed＝20261014、结局分层；不参考影像、预测概率或AUC选择病例或种子。', '',
        '## 对照病例的关系', '',
        '97例训练全部包含在110例训练中；28例测试全部包含在41例测试中。8∶2比7∶3多13例训练（阳性9、阴性4），这13例属于7∶3的测试集。每种划分自己的训练和测试交集均为0。', '',
        '共同28例为两种新模型均未参与拟合的同一批测试病例（阳性19、阴性9）。使用固定种子产生的这28例与更早77/33/28实验中的测试B名单相同，已被历史探索，不是全新外部验证。', '',
        '## 后续怎样比较模型效果', '',
        '1. 两套比例分别在各自训练97例、110例内做外层五折/内层五折与最终五折选择。同一特征定义、两种标准化、k、质量门槛、预处理、候选模型空间和选择目标保持可比。若保留昨日配置与训练内重选两种策略，它们也各自在两种比例中采用一致规则，分开报告。',
        '2. 各比例的特征筛选、缩放、降维和超参数只在当前训练折学习，不共享利用另一比例训练结局得到的筛选结果。',
        '3. 完成两套比例的所有训练选择与核验，冻结全部模型及训练OOF阈值后，再统一进行最终测试。',
        '4. 各自测试：7∶3模型评估41例，8∶2模型评估28例；报告AUC、95%区间及分类指标。这种差异同时包含训练样本数和测试病例变化。',
        '5. 同病例对照：将7∶3模型保存的41例预测取共同28例，与8∶2模型的28例预测配对比较，报告ΔAUC及患者配对bootstrap区间。7∶3取子集不新增模型调用。',
        '6. 增加的13例已属于8∶2训练集，不能把8∶2模型在这些病例的结果计入独立测试。全部结果一起报告，不选测试分数较高的划分替换另一组。', '',
        '两种比例每组各有35份训练/验证折，合计70份，均已核验并排除本组测试病例。当前只固定名单和对照设计，训练、预测及AUC计算次数均为0；不能据此判断哪个比例效果更好。', '',
        '## 名单与记录', '',
        f'- [全部138例双划分对照名单]({p(PAIR,"all138_split_comparison.csv")})',
        f'- [原7∶3训练97例]({p(baseline.OUT,"Train97_cases.csv")}) · [原7∶3测试41例]({p(baseline.OUT,"Test41_cases.csv")})',
        f'- [新8∶2训练110例]({p(EIGHT,"Train110_cases.csv")}) · [新8∶2测试28例]({p(EIGHT,"Test28_cases.csv")})',
        f'- [共同28例测试名单]({p(PAIR,"common_test28_cases.csv")})',
        f'- [8∶2增加的13例训练病例]({p(PAIR,"extra13_training_cases_in80_20.csv")})',
        f'- [110例训练嵌套五折安排]({p(EIGHT,"nested5_training_folds.json")})', '',
        '后续两比例对照以LVI_ACTIVE_SPLIT_COMPARISON.json为入口；原LVI_ACTIVE_DEVELOPMENT_SPLIT.json保留7∶3定义作为对照，原名单和哈希均不改写。此前模型和正式采用决定保持不变。', ''])
    report.write_text(text, encoding='utf-8')
    assert '\ufffd' not in text
    save(PAIR/'report_generation.json', {'passed': True, 'report': report.name, 'report_sha256': digest(report),
        'active_registry_sha256': digest(ACTIVE), 'model_fit_calls': 0, 'model_predict_calls': 0})
    outputs = {str(path.relative_to(ROOT)): digest(path) for directory in (EIGHT, PAIR) for path in directory.rglob('*')
               if path.is_file() and path.name != 'delivery_manifest.json'}
    protocol = json.loads((PAIR/'protocol.json').read_text(encoding='utf-8'))
    distinct = len({str((ROOT/path).resolve()).casefold() for path in protocol['protected_hashes']})
    protect()
    save(PAIR/'delivery_manifest.json', {'passed': True, 'outputs': outputs, 'output_count': len(outputs),
        'source_sha256': digest(__file__), 'historical_files_unchanged': distinct, 'seven_split_unchanged': True,
        'both_ratios_registered': True, 'active_registry_sha256': digest(ACTIVE), 'model_fit_calls': 0,
        'model_predict_calls': 0, 'AUC_based_selection': False, 'formal_model_adoption_changed': False})
    print('7:3 vs8:2 comparison delivered: %d outputs; %d historical files unchanged'%(len(outputs), distinct), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['define', 'verify', 'deliver'], required=True)
    {'define': define, 'verify': verify, 'deliver': deliver}[parser.parse_args().phase]()
