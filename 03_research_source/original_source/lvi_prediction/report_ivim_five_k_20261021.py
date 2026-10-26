# -*- coding: utf-8 -*-
"""Five-family report, training-only K comparison, and immutable delivery record."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import run_ivim_five_k_20261021 as run

LABELS = {'Habitat': '①IVIM生境', 'Whole': '②IVIM全肿瘤影像组学', 'Clinical': '③术前临床',
          'HabitatClinical': '④IVIM生境＋临床', 'WholeClinical': '⑤IVIM全肿瘤影像组学＋临床'}
NORMS = {'PercentileZ': '截尾＋Z-score', 'RobustZ': '稳健Z-score', 'NA': '—'}
INPUTS = {'C08': 'D强度＋D局部SD', 'C18': 'D局部SD＋f局部SD',
          'C38': 'D局部SD＋f强度＋f局部均值', 'NA': '—'}


def plot(diag, main):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, group in zip(axes, run.study.GROUPS):
        for family, color in [('Habitat', '#2368ae'), ('HabitatClinical', '#c35d1d')]:
            d = diag[(diag.Group == group) & (diag.family == family)].sort_values('K')
            x, y, sd = d.K.to_numpy(), d.OuterMeanAUC.to_numpy(), d.OuterSD.to_numpy()
            ax.plot(x, y, marker='o', color=color, label=family)
            ax.fill_between(x, np.clip(y - sd, 0, 1), np.clip(y + sd, 0, 1), color=color, alpha=.12)
        clinical = main[(main.Group == group) & (main.ModelKey == 'Clinical')].iloc[0].OuterMeanAUC
        whole = main[(main.Group == group) & (main.ModelKey == 'Whole')].iloc[0].OuterMeanAUC
        ax.axhline(clinical, color='#7352a5', linestyle=':', label='Clinical')
        ax.axhline(whole, color='#35885c', linestyle='--', label='Whole IVIM')
        ax.set(title=group + ': nested outer-fold AUC', xlabel='Number of habitats (K)',
               xticks=list(run.study.feature.KS), ylim=(0, 1))
        ax.grid(alpha=.2)
    axes[0].set_ylabel('AUC (shade: five-fold SD)')
    axes[1].legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    folder = run.OUT / 'figures'; folder.mkdir(exist_ok=True)
    for extension in ('png', 'svg'):
        fig.savefig(folder / ('training_only_K_comparison.' + extension), dpi=180, bbox_inches='tight')
    plt.close(fig)
    return folder / 'training_only_K_comparison.png'


def main():
    run.frozen()
    assert json.loads((run.OUT / 'inference_verification.json').read_text(encoding='utf-8'))['passed']
    choices, tests = run.study.read('training_choices'), run.study.read('test_metrics')
    primary = choices[choices.ModelKey.isin(run.study.FAMILIES)].copy()
    result = primary.merge(tests, on=['Group', 'ModelKey'], suffixes=('', '_Test'), validate='one_to_one')
    assert len(result) == 10
    run.csv('five_models_summary', result)
    diag = choices[~choices.ModelKey.isin(run.study.FAMILIES)].copy()
    assert len(diag) == 20
    run.csv('fixed_K_training_only_summary', diag)
    outer = run.study.read('outer_metrics')
    selected = outer[outer.ModelKey.isin(['Habitat', 'HabitatClinical'])]
    frequency = selected.groupby(['Group', 'ModelKey', 'K']).size().reset_index(name='OuterFoldCount')
    run.csv('K_selection_frequency', frequency)
    ranking = result.groupby('ModelKey').agg(MeanOuterAUC=('OuterMeanAUC', 'mean'),
        MinimumOuterAUC=('OuterMeanAUC', 'min'), MeanTrainFitAUC=('TrainFitAUC', 'mean'),
        MeanTestAUC=('AUC', 'mean')).reset_index().sort_values(['MeanOuterAUC', 'MinimumOuterAUC'], ascending=False)
    run.csv('five_family_descriptive_comparison', ranking)
    figure = plot(diag, result)
    lines = [
        '# 纯IVIM聚类数改进：两组100例、五种模型', '',
        '日期：2026年10月21日。依据用户确认，使用既有A／B两组各100例，'
        '每组70例训练、30例测试，病例及全部内外层五折名单保持不变。'
        '影像预测特征只使用IVIM的D、f；不使用T2、ADC或D*影像特征。', '',
        '## 五模型主要结果', '',
        '下表为完整嵌套选择流程：各外层折内，k、输入组合、标准化、特征筛选及C均由该折内层选择。'
        '最终k为全部70例训练内层选择的结果，不要求与每个外层折相同。'
        '±为五折AUC的样本SD，另行保存合并OOF AUC，不相互替代。', '',
        '| 数据组 | 模型 | 最终k | 训练拟合AUC | 外层五折均值±SD | 冻结30例测试AUC（95%区间） |',
        '|---|---|---|---|---|---|']
    for group in run.study.GROUPS:
        for family in run.study.FAMILIES:
            r = result[(result.Group == group) & (result.ModelKey == family)].iloc[0]
            k = str(int(r.K)) if r.K else '—'
            lines.append(f'| {group} | {LABELS[family]} | {k} | {r.TrainFitAUC:.4f} | '
                         f'{r.OuterMeanAUC:.4f}±{r.OuterSD:.4f} | {r.AUC:.4f}（{r.AUC_CI95Low:.4f}—{r.AUC_CI95High:.4f}） |')
    lines += ['', '## 训练内选定的影像设置与特征数', '',
        '| 数据组 | 模型 | 标准化 | IVIM聚类输入 | 最终k | 影像／临床特征数 | C |',
        '|---|---|---|---|---|---|---|']
    for r in result.itertuples(index=False):
        lines.append(f'| {r.Group} | {LABELS[r.ModelKey]} | {NORMS[r.normalization]} | {INPUTS[r.ComboID]} | '
                     f'{int(r.K) if r.K else "—"} | {int(r.ImageFeatureCount)}／{int(r.ClinicalFeatureCount)} | {r.C:g} |')
    lines += ['', '## k＝2、3、4、5、6的训练内对照', '',
        '以下为固定k时，在每个外层训练部分内再选择输入、标准化、特征及参数所得的外层验证均值。'
        '这些诊断模型没有运行测试集，避免额外按测试AUC选择k。主结果使用上表完整的内层选k流程。', '',
        '| 模型 | k | A组拟合／外层均值±SD | B组拟合／外层均值±SD | 两组外层均值的平均 |',
        '|---|---|---|---|---|']
    for family in ('Habitat', 'HabitatClinical'):
        for k in run.study.feature.KS:
            aa = diag[(diag.Group == 'SampleA') & (diag.family == family) & (diag.K == k)].iloc[0]
            bb = diag[(diag.Group == 'SampleB') & (diag.family == family) & (diag.K == k)].iloc[0]
            lines.append(f'| {LABELS[family]} | {k} | {aa.TrainFitAUC:.4f}／{aa.OuterMeanAUC:.4f}±{aa.OuterSD:.4f} | '
                         f'{bb.TrainFitAUC:.4f}／{bb.OuterMeanAUC:.4f}±{bb.OuterSD:.4f} | {(aa.OuterMeanAUC + bb.OuterMeanAUC) / 2:.4f} |')
    lines += ['', f'![训练内k比较]({figure.as_posix()})', '', '## 各外层折选中的k', '',
        '| 数据组 | 模型 | k | 选中外层折数／5 |', '|---|---|---|---|']
    for r in frequency.itertuples(index=False):
        lines.append(f'| {r.Group} | {LABELS[r.ModelKey]} | {r.K} | {r.OuterFoldCount} |')
    lines += ['', '## 对改进意见的落实及评价', '',
        '- 已先回到纯IVIM影像研究，并将k扩展到2—6；保留C08、C18、C38三种聚类输入，'
        '每种输入均比较截尾＋Z-score和稳健Z-score，共30种生境设置。分区后均测量D和f。',
        '- 使用此前17种区域纹理指标及原始定量／空间指标，推广为任意k的156个标签不变汇总特征。'
        '接触与质心距离按该k的全部生境对汇总；k＝2时只有一个生境对，其SD／Range自然为0，'
        '会由训练折的非常量筛选去除。k＝3与原缓存的全部值及标签进行了核对。',
        '- 全肿瘤影像组学采用D、f各93个Original非形状特征，共186个，复用无结局依赖的逐患者提取缓存；'
        '不按k重复提取或重复建模。纯临床模型不依赖影像标准化或k。',
        '- 临床部分沿用此前五模型的11项术前变量：术前T／N、EMVI、MRF、logCEA、年龄、性别、'
        'MRI位置、厚度、长度、环周比例。后四项是常规MRI人工测量信息，不是T2／ADC影像组学特征。'
        '未使用术后病理预测变量。MRI均为新辅助治疗前、病理标准统一，采用用户既往确认；'
        '旧解析审计中的unknown描述没有被当作新的临床否定结论。',
        '- 每一内层训练折重新做完整性、ICC、关联／稳定性排序、去相关及预处理；'
        '影像预算10／20／全部及C在内层选择。临床11项均保留，分别以训练折参数填补、截尾及缩放后进入联合模型。',
        '- 所有模型均使用L2逻辑回归，内层目标保持原定义：平均验证AUC−0.10×SD−'
        '0.30×max（平均拟合AUC−平均验证AUC−0.10，0）。本轮没有同时按测试表现改换分类器或权重。',
        '- 截图提出的挑选A／B病例使训练、五折及测试AUC升高，会让划分参与结果选优。'
        '本轮按用户确认保留原病例及划分；没有按结局、特征或AUC删换病例。',
        '- 两组各100例重叠74例，属于相关的内部重复实验，不能当作两套外部验证或拼接为独立200例。'
        '每组仅使用本组70例训练标签。所有10个主模型冻结、训练验证通过后，才完成一次测试阶段。',
        '- 原队列测试集已有多轮查看，聚类输入候选也受此前探索启发。'
        '本轮结果用于方法比较，不是新的独立验证；不自动替换正式模型或恢复此前明确排除的模型。', '',
        '## 五类模型的描述性排序', '',
        '按两组外层五折均值的平均排序，病例重叠使此平均仅具描述性意义，不按10个独立折计算显著性。', '',
        '| 模型 | 两组平均外层AUC | 较低组外层AUC |', '|---|---|---|']
    for r in ranking.itertuples(index=False):
        lines.append(f'| {LABELS[r.ModelKey]} | {r.MeanOuterAUC:.4f} | {r.MinimumOuterAUC:.4f} |')
    lines += ['', '## 文件与核查', '',
        '- five_models_summary.csv：10个主模型的拟合、外层、测试及分类指标。',
        '- fixed_K_training_only_summary.csv：20个固定k训练诊断模型的拟合及外层结果，无额外测试预测。',
        '- K_selection_frequency.csv、outer_metrics.csv：逐外层折选中的k、输入、标准化和配置。',
        '- selected_features.csv：每组每模型最终特征及系数。',
        '- nested_predictions.csv、test_predictions.csv：逐患者外层OOF／冻结测试概率。',
        '- paired_family_AUC_differences.csv：同组同30例上的探索性配对bootstrap差值，未作多重比较校正。',
        '- protocol.json、feature_smoke_verification.json、training_verification.json、inference_verification.json：'
        '预设范围、特征等价、内外层隔离、独立AUC重算及预测核查。', '']
    report = run.OUT / '纯IVIM聚类数改进_五模型两组100例报告_20261021.md'
    report.write_text('\n'.join(lines), encoding='utf-8')
    protected = run.study.protect()
    outputs = {str(p.relative_to(run.OUT)): run.digest(p) for p in run.OUT.rglob('*') if p.is_file()}
    run.save(run.OUT / 'delivery_manifest.json', {'passed': True, 'outputs': outputs,
        'protected_files_unchanged': protected, 'report': report.name, 'final_test_models': 10,
        'test_stage_count': 1, 'formal_model_use': False, 'automatic_adoption': False})
    print(result[['Group', 'ModelKey', 'K', 'normalization', 'ComboID', 'TrainFitAUC', 'OuterMeanAUC', 'OuterSD', 'AUC']].to_string(index=False), flush=True)
    print('Report and training-only K figure generated;visual check required', flush=True)
