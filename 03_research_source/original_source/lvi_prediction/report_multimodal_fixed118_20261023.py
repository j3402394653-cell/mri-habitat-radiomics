# -*- coding: utf-8 -*-
"""Deliver the five multimodal families and the matched IVIM-only comparison."""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import run_multimodal_fixed118_20261023 as run
from verify_ivim_df63_20261020 import auc, draws

LABELS = {'Habitat': '①多序列生境', 'Whole': '②多序列全肿瘤影像组学', 'Clinical': '③术前临床',
          'HabitatClinical': '④多序列生境＋临床', 'WholeClinical': '⑤多序列全肿瘤影像组学＋临床'}
NORMS = {'PercentileZ': '截尾＋Z-score', 'RobustZ': '稳健Z-score', 'NA': '—'}


def main():
    run.frozen()
    assert json.loads((run.OUT / 'inference_verification.json').read_text(encoding='utf-8'))['passed']
    choices = run.study.read('training_choices')
    primary = choices[choices.ModelKey.isin(run.study.FAMILIES)]
    result = primary.merge(run.study.read('test_metrics'), on=['Group', 'ModelKey'], validate='one_to_one', suffixes=('', '_Test'))
    features = run.study.read('selected_features')
    for modality in ('T2', 'ADC', 'IVIM', 'Clinical'):
        result[modality + 'Count'] = [int(((features.ModelKey == m) & (features.Block == modality)).sum()) for m in result.ModelKey]
    run.csv('five_models_summary', result)
    diag = choices[~choices.ModelKey.isin(run.study.FAMILIES)]
    run.csv('fixed_IVIM_K_training_only_summary', diag)
    outer = run.study.read('outer_metrics')
    run.csv('outer_IVIM_K_selection', outer[outer.ModelKey.isin(('Habitat', 'HabitatClinical'))])
    prior = run.study.prior.read('five_models_summary').set_index('ModelKey')
    current_predictions = run.study.read('test_predictions')
    old_predictions = run.study.prior.read('test_predictions')
    comparison, scores = [], {}
    ids = run.study.inputs('Fixed118')[0].AnalysisID.iloc[run.study.inputs('Fixed118')[3]]
    for family in run.study.FAMILIES:
        new = current_predictions[current_predictions.ModelKey == family].set_index('AnalysisID').loc[ids]
        old = old_predictions[old_predictions.ModelKey == family].set_index('AnalysisID').loc[ids]
        np.testing.assert_array_equal(new.Outcome, old.Outcome)
        scores['New_' + family] = new.Probability.to_numpy()
        scores['Old_' + family] = old.Probability.to_numpy()
        truth = new.Outcome.to_numpy(int)
        if family == 'Clinical':
            np.testing.assert_allclose(new.Probability, old.Probability, atol=1e-12, rtol=0)
    boot = draws(truth, scores, 2026102351)
    for r in result.itertuples(index=False):
        old = prior.loc[r.ModelKey]
        low, high = np.quantile(boot['New_' + r.ModelKey] - boot['Old_' + r.ModelKey], [.025, .975])
        comparison.append({'ModelKey': r.ModelKey, 'IVIMOnlyTrainFitAUC': old.TrainFitAUC, 'MultiTrainFitAUC': r.TrainFitAUC,
            'IVIMOnlyOuterMeanAUC': old.OuterMeanAUC, 'MultiOuterMeanAUC': r.OuterMeanAUC,
            'OuterMeanDifference': r.OuterMeanAUC - old.OuterMeanAUC,
            'IVIMOnlyTestAUC': old.AUC, 'MultiTestAUC': r.AUC, 'PairedTestDifference': r.AUC - old.AUC,
            'PairedTestDifferenceCI95Low': low, 'PairedTestDifferenceCI95High': high})
    run.csv('comparison_with_IVIM_only', comparison)
    run.csv('paired_comparison_bootstrap5000', pd.DataFrame(boot))
    labels = list(run.study.FAMILIES)
    positions = np.arange(5)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for offset, key, color, label in ((-.24, 'TrainFitAUC', '#90b4d4', 'Training fit'),
                                    (0, 'OuterMeanAUC', '#2474a5', 'Nested outer5 mean'),
                                    (.24, 'AUC', '#e48b36', 'Frozen test36')):
        values = result.set_index('ModelKey').loc[labels, key].to_numpy()
        ax.bar(positions + offset, values, width=.23, label=label, color=color)
    ax.set(xticks=positions, xticklabels=['Habitat', 'Whole', 'Clinical', 'Habitat+Clinical', 'Whole+Clinical'],
           ylabel='AUC', ylim=(0, 1), title='Original118: T2 + ADC + IVIM (D,f), fixed82/36')
    ax.axhline(.5, color='#888888', linestyle=':', linewidth=1)
    ax.legend(fontsize=9, ncol=3); ax.grid(axis='y', alpha=.15); fig.tight_layout()
    folder = run.OUT / 'figures'; folder.mkdir(exist_ok=True)
    for suffix in ('png', 'svg'):
        fig.savefig(folder / ('multimodal118_five_models.' + suffix), dpi=180, bbox_inches='tight')
    plt.close(fig)
    lines = ['# 原118例：T2＋ADC＋IVIM五模型对照', '',
        '日期：2026年10月23日。精确沿用原118例及82例训练（58阳性、24阴性）／36例测试（25阳性、11阴性）名单，'
        '与上一轮纯IVIM对照使用完全相同的外层／内层五折。未按本轮AUC重新选病例。', '',
        'T2、ADC各在原生空间独立聚类，使用强度、3 mm局部均值及局部SD，沿用此前验证的k＝3；'
        'IVIM聚类输入仍比较既定三种D／f组合，k＝2～6在内层选择。各序列生境特征按患者拼接，'
        '不做跨序列逐体素融合，不取最高值，不加入D*。', '',
        '## 五模型结果', '',
        '所有特征质量筛选、结局相关筛选、预处理、标准化选择及C均在相应训练折内完成。'
        '最终五模型训练、冻结及核查后，各对原36例预测一次。±为五个外层折AUC的样本SD。', '',
        '| 模型 | 训练拟合AUC | 外层五折均值±SD | 测试AUC（95%区间） |', '|---|---|---|---|']
    for r in result.itertuples(index=False):
        lines.append(f'| {LABELS[r.ModelKey]} | {r.TrainFitAUC:.4f} | {r.OuterMeanAUC:.4f}±{r.OuterSD:.4f} | '
                     f'{r.AUC:.4f}（{r.AUC_CI95Low:.4f}—{r.AUC_CI95High:.4f}） |')
    lines += ['', '## 与同划分纯IVIM对照', '',
        '| 模型 | 原纯IVIM外层均值 → 多序列 | 原纯IVIM测试AUC → 多序列 | 测试AUC差值及配对95%区间 |', '|---|---|---|---|']
    for r in pd.DataFrame(comparison).itertuples(index=False):
        lines.append(f'| {LABELS[r.ModelKey]} | {r.IVIMOnlyOuterMeanAUC:.4f} → {r.MultiOuterMeanAUC:.4f} | '
            f'{r.IVIMOnlyTestAUC:.4f} → {r.MultiTestAUC:.4f} | {r.PairedTestDifference:+.4f} '
            f'[{r.PairedTestDifferenceCI95Low:+.4f}, {r.PairedTestDifferenceCI95High:+.4f}] |')
    lines += ['', '临床模型作为相同变量、相同划分的对照，外层及最终权重已核对与上一轮一致，测试概率也保持一致。'
        '配对区间以相同36例进行5000次分层患者bootstrap；不据此反复选择模型。', '',
        '## 最终配置与实际使用序列', '',
        '| 模型 | 标准化 | IVIM k | T2／ADC／IVIM／临床特征数 | C |', '|---|---|---|---|---|']
    for r in result.itertuples(index=False):
        lines.append(f'| {LABELS[r.ModelKey]} | {NORMS[r.normalization]} | {int(r.K) if r.K else "—"} | '
            f'{r.T2Count}／{r.ADCCount}／{r.IVIMCount}／{r.ClinicalCount} | {r.C:g} |')
    lines += ['', '生境候选特征330个：T2 87、ADC 87、IVIM 156；全肿瘤候选372个：T2、ADC、D、f各93个。'
        '不强制每个序列入选，表中为最终实际特征数。临床沿用11项已审计术前变量，含常规MRI人工测量，'
        '没有术后或病理预测变量。所有影像块分别使用同一候选标准化：截尾＋Z-score或稳健Z-score。', '',
        '## 解读', '']
    winner = result.sort_values('OuterMeanAUC', ascending=False).iloc[0]
    lines.append(f'按预先侧重的外层五折均值，本轮最高为{LABELS[winner.ModelKey]}：'
        f'{winner.OuterMeanAUC:.4f}±{winner.OuterSD:.4f}，其测试AUC为{winner.AUC:.4f}。'
        '模型评价需同时查看训练拟合与验证差距，以及测试区间。测试最高值不用于事后替换训练内选择。')
    for r in result.itertuples(index=False):
        if r.ModelKey != 'Clinical':
            change = r.OuterMeanAUC - prior.loc[r.ModelKey].OuterMeanAUC
            lines.append(f'- {LABELS[r.ModelKey]}：外层均值变化{change:+.4f}，'
                f'训练拟合与外层均值相差{r.TrainFitAUC-r.OuterMeanAUC:.4f}。')
    lines += ['', '本轮36例此前已被多次查看；三种IVIM输入也来自既往同队列探索。'
        '本轮嵌套验证评估已确定候选流程，不能消除此前探索造成的选择影响，仍属探索性内部复验。'
        '当前计算不能单独证明增加某序列的独立生物学价值。原版采用决定不变，不自动作为正式模型使用。', '',
        '固定IVIM k的训练内诊断见fixed_IVIM_K_training_only_summary.csv；这些诊断模型未额外测试。'
        '所有原始病例、历史结果及上一轮纯IVIM模型按哈希核对未改变。', '',
        f'![五模型AUC]({(folder / "multimodal118_five_models.png").as_posix()})', '',
        '完整结果见five_models_summary.csv；同划分对照见comparison_with_IVIM_only.csv；'
        '逐患者预测见test_predictions.csv；实际特征与系数见selected_features.csv。', '']
    report = run.OUT / '原118例多序列五模型验证报告_20261023.md'
    report.write_text('\n'.join(lines), encoding='utf-8')
    protected = run.study.protect()
    run.save(run.OUT / 'delivery_manifest.json', {'passed': True, 'date': '2026-10-23',
        'outputs': {str(p.relative_to(run.OUT)): run.digest(p) for p in run.OUT.rglob('*')
                    if p.is_file() and p.name != 'delivery_manifest.json'},
        'protected_files_unchanged': protected, 'report': report.name,
        'formal_model_use': False, 'historical_results_replaced': False,
        'final_test_models': 5, 'test_stage_count': 1})
    print(result[['ModelKey', 'K', 'TrainFitAUC', 'OuterMeanAUC', 'OuterSD', 'AUC', 'T2Count', 'ADCCount', 'IVIMCount']].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
