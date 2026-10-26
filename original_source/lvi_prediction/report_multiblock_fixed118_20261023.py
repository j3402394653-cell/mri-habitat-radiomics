# -*- coding: utf-8 -*-
"""Report training-selected low-dimensional models with matched old controls."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import run_multiblock_fixed118_20261023 as run
from verify_ivim_df63_20261020 import draws

LABELS = {'Habitat': '①生境', 'Whole': '②全肿瘤影像组学', 'Clinical': '③临床',
          'HabitatClinical': '④生境＋临床', 'WholeClinical': '⑤全肿瘤影像组学＋临床'}
METHODS = {'Balanced': '按序列均衡筛选', 'PCA': '按序列PCA压缩', 'Late': '三折交叉拟合评分融合', 'Clinical': '临床逻辑回归'}


def main():
    run.frozen()
    assert json.loads((run.OUT / 'inference_verification.json').read_text(encoding='utf-8'))['passed']
    result = run.read('training_choices').merge(run.read('test_metrics'), on='ModelKey', validate='one_to_one', suffixes=('', '_Test'))
    assert len(result) == 5
    run.csv('five_models_summary', result)
    old = run.old.read('five_models_summary').set_index('ModelKey')
    newp, oldp = run.read('test_predictions'), run.old.read('test_predictions')
    cohort, _, _, te = run.old.inputs('Fixed118')
    ids = cohort.AnalysisID.iloc[te]
    scores = {}
    for family in run.FAMILIES:
        scores['New_' + family] = newp[newp.ModelKey == family].set_index('AnalysisID').loc[ids].Probability.to_numpy()
        scores['Old_' + family] = oldp[oldp.ModelKey == family].set_index('AnalysisID').loc[ids].Probability.to_numpy()
    boot = draws(cohort.Outcome.to_numpy(int)[te], scores, 2026102362)
    comparison = []
    for r in result.itertuples(index=False):
        previous = old.loc[r.ModelKey]
        low, high = np.quantile(boot['New_' + r.ModelKey] - boot['Old_' + r.ModelKey], [.025, .975])
        comparison.append({'ModelKey': r.ModelKey, 'OldTrainFitAUC': previous.TrainFitAUC, 'NewTrainFitAUC': r.TrainFitAUC,
            'OldOuterMeanAUC': previous.OuterMeanAUC, 'NewOuterMeanAUC': r.OuterMeanAUC, 'OuterMeanDifference': r.OuterMeanAUC - previous.OuterMeanAUC,
            'OldTestAUC': previous.AUC, 'NewTestAUC': r.AUC, 'TestDifference': r.AUC - previous.AUC,
            'TestDifferenceCI95Low': low, 'TestDifferenceCI95High': high})
    run.csv('comparison_with_previous_multimodal', comparison)
    run.csv('paired_comparison_bootstrap5000', pd.DataFrame(boot))
    # Training-only architecture diagnostics: each architecture tuned inside each outer split.
    diagnostic = []
    for fold, (a, b) in enumerate(run.old.folds('Fixed118', 'outer'), 1):
        import joblib
        phase = joblib.load(run.OUT / 'phases' / ('outer' + str(fold) + '.joblib'))
        for family in run.FAMILIES:
            for architecture in sorted({c['architecture'] for c in phase['grid'] if c['family'] == family}):
                restricted = [c for c in phase['grid'] if c['architecture'] == architecture]
                chosen = run.choose(restricted, family)
                diagnostic.append({'ModelKey': family, 'Architecture': architecture, 'Fold': fold,
                    'SelectedInnerMeanAUC': chosen['MeanValidationAUC'], 'SelectedInnerObjective': chosen['Objective'],
                    'candidate_id': chosen['candidate_id'], 'Note': 'Inner selection diagnostic only; no additional outer refits orholdout calls.'})
    run.csv('architecture_inner_choices_by_outer_fold', diagnostic)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    positions = np.arange(5)
    for offset, column, color, label in ((-.24, 'TrainFitAUC', '#95bad6', 'Training fit'), (0, 'OuterMeanAUC', '#2879a6', 'Nested outer5'), (.24, 'AUC', '#dc8a32', 'Frozen test36')):
        ax.bar(positions + offset, result.set_index('ModelKey').loc[list(run.FAMILIES), column], width=.23, color=color, label=label)
    ax.set(xticks=positions, xticklabels=['Habitat', 'Whole', 'Clinical', 'Habitat+Clinical', 'Whole+Clinical'], ylim=(0, 1), ylabel='AUC', title='Original118: low-dimensional multiblock models (82/36)')
    ax.axhline(.5, color='gray', linestyle=':', linewidth=1)
    ax.legend(ncol=3, fontsize=8); ax.grid(axis='y', alpha=.15); fig.tight_layout()
    folder = run.OUT / 'figures'; folder.mkdir(exist_ok=True)
    for extension in ('png', 'svg'):
        fig.savefig(folder / ('multiblock_five_models.' + extension), dpi=180, bbox_inches='tight')
    plt.close(fig)
    lines = ['# 原118例：低维多序列融合改进实验', '', '日期：2026年10月23日。精确沿用原82例训练／36例测试及上一轮外层五折／内层五折名单。保留T2、ADC、IVIM D／f，不加入D*。', '',
        '## 改进内容', '',
        '- 不再把三个序列混在一个特征排序中竞争。比较每序列3／5个特征、每序列PCA 1／2个成分、每序列一个交叉拟合评分三种架构；三种影像序列均须有符合质量要求的输入，否则该候选标为不可用。',
        '- ICC 0.75／0.80在内层比较，完整率仍要求≥0.9；按序列去除绝对Spearman相关≥0.95的重复特征，所有步骤均只使用对应训练子集。',
        '- C限定0.003／0.03／0.3，不再使用C＝3。临床在原5项术前变量与扩展11项之间进行内层选择。',
        '- 评分融合在每个当前训练子集内部再做三折交叉拟合；基模型的特征筛选、预处理及拟合也只使用该三折中的训练部分。融合层使用这些OOF评分训练，验证病例不进入基模型或融合层训练。',
        '- T2／ADC的k固定为3；IVIM仅在2／3及既定三组输入之间选择。标准化保留截尾＋Z-score和稳健Z-score。', '',
        '## 五模型结果', '', '每一外层折独立由其内层选择完整架构、特征、质量阈值、参数和临床变量集。全部82例重新内层选择后冻结，原36例各预测一次；±为五折样本SD。', '',
        '| 模型 | 训练拟合AUC | 外层五折均值±SD | 测试AUC（95%区间） |', '|---|---|---|---|']
    for r in result.itertuples(index=False):
        lines.append(f'| {LABELS[r.ModelKey]} | {r.TrainFitAUC:.4f} | {r.OuterMeanAUC:.4f}±{r.OuterSD:.4f} | {r.AUC:.4f}（{r.AUC_CI95Low:.4f}—{r.AUC_CI95High:.4f}） |')
    lines += ['', '## 最终配置与维度', '', '| 模型 | 架构 | IVIM k | ICC | T2／ADC／IVIM原始特征数 | 影像有效维数 | 临床数 | C |', '|---|---|---|---|---|---|---|---|']
    for r in result.itertuples(index=False):
        lines.append(f'| {LABELS[r.ModelKey]} | {METHODS[r.architecture]} | {r.K if r.K else "—"} | {r.quality if r.quality else "—"} | '
            f'{r.T2Count}／{r.ADCCount}／{r.IVIMCount} | {r.ImageDimensions} | {r.ClinicalDimensions} | {r.C:g} |')
    lines += ['', 'PCA原始特征数表示进入降维的原始变量数，不能将主成分系数直接解释为单个原始特征权重。评分融合的影像有效维数为三个序列评分；每个基模型全训练拟合后的原始输入另行列出。', '',
        '## 与上一轮直接拼接比较', '', '| 模型 | 外层均值：原→本轮 | 测试AUC：原→本轮 | 测试差值及配对95%区间 |', '|---|---|---|---|']
    for r in pd.DataFrame(comparison).itertuples(index=False):
        lines.append(f'| {LABELS[r.ModelKey]} | {r.OldOuterMeanAUC:.4f} → {r.NewOuterMeanAUC:.4f} | {r.OldTestAUC:.4f} → {r.NewTestAUC:.4f} | '
            f'{r.TestDifference:+.4f} [{r.TestDifferenceCI95Low:+.4f}, {r.TestDifferenceCI95High:+.4f}] |')
    best = result.sort_values('OuterMeanAUC', ascending=False).iloc[0]
    lines += ['', f'按预先侧重的外层五折均值，本轮最高为{LABELS[best.ModelKey]}，均值{best.OuterMeanAUC:.4f}±{best.OuterSD:.4f}，'
        f'训练拟合{best.TrainFitAUC:.4f}，测试{best.AUC:.4f}。需要结合差值、折间波动与测试区间判断改进是否稳定，不能只按单个最高测试AUC采用。', '',
        '## 验证边界', '',
        '这118例与原36例测试集此前已反复参与探索；本轮嵌套验证只评估本次预先固定的候选流程，不能消除整个历史探索的选择影响。'
        '本轮依然属于探索性内部验证。原版模型采用决定不变；此前明确排除的迁移实验和20261010监督模型未重新作为采用候选。', '',
        '仅五个训练内选定模型计算本轮测试预测，没有额外逐架构、逐k或逐参数测试。验证包括独立排序AUC、'
        '外层标签及数值扰动、三折交叉拟合基模型回放、完整训练OOF阈值回放、独立PCA／基模型／融合层手工概率及5000次分层bootstrap区间。', '',
        f'![五模型对照]({(folder / "multiblock_five_models.png").as_posix()})', '',
        '完整数值见five_models_summary.csv、comparison_with_previous_multimodal.csv；逐病例预测见test_predictions.csv，原始输入见selected_features.csv。', '']
    report = run.OUT / '原118例低维多序列融合改进报告_20261023.md'
    report.write_text('\n'.join(lines), encoding='utf-8')
    run.save(run.OUT / 'delivery_manifest.json', {'passed': True, 'report': report.name, 'formal_model_use': False,
        'historical_results_replaced': False, 'protected_files_unchanged': run.protect(), 'test_model_calls': 5,
        'test_stage_count': 1, 'outputs': {str(p.relative_to(run.OUT)): run.digest(p) for p in run.OUT.rglob('*') if p.is_file() and p.name != 'delivery_manifest.json'}})
    print(result[['ModelKey', 'architecture', 'TrainFitAUC', 'OuterMeanAUC', 'OuterSD', 'AUC']].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
