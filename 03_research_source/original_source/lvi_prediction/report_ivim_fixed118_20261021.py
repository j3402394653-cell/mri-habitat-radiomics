# -*- coding: utf-8 -*-
"""Current pure-IVIM method on the original118 cohort and82/36 split."""
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import run_ivim_fixed118_20261021 as run
from report_ivim_five_k_20261021 import LABELS, NORMS, INPUTS


def main():
    run.frozen()
    assert json.loads((run.OUT / 'inference_verification.json').read_text(encoding='utf-8'))['passed']
    choices = run.study.read('training_choices')
    primary = choices[choices.ModelKey.isin(run.study.FAMILIES)]
    result = primary.merge(run.study.read('test_metrics'), on=['Group', 'ModelKey'],
                           suffixes=('', '_Test'), validate='one_to_one')
    assert len(result) == 5
    run.csv('five_models_summary', result)
    diag = choices[~choices.ModelKey.isin(run.study.FAMILIES)]
    assert len(diag) == 10
    run.csv('fixed_K_training_only_summary', diag)
    selected = run.study.read('outer_metrics')
    selected = selected[selected.ModelKey.isin(('Habitat', 'HabitatClinical'))]
    run.csv('outer_K_selection', selected)
    historical = pd.read_csv(run.study.HISTORICAL / 'metrics.csv')
    historical.to_csv(run.OUT / 'historical118_metrics_reference.csv', index=False, encoding='utf-8-sig')
    fig, ax = plt.subplots(figsize=(7.3, 4.5))
    for family, color in [('Habitat', '#2368ae'), ('HabitatClinical', '#c35d1d')]:
        d = diag[diag.family == family].sort_values('K')
        ax.errorbar(d.K, d.OuterMeanAUC, yerr=d.OuterSD, marker='o', capsize=3, color=color, label=family)
    for family, color, style in [('Whole', '#35885c', '--'), ('Clinical', '#7352a5', ':')]:
        ax.axhline(result[result.ModelKey == family].iloc[0].OuterMeanAUC, color=color, linestyle=style, label=family)
    ax.set(title='Original118: nested outer-fold AUC', xlabel='Number of habitats (K)',
           ylabel='Mean AUC (error bars: fold SD)', xticks=list(run.study.feature.KS), ylim=(0, 1))
    ax.legend(fontsize=8); ax.grid(alpha=.2); fig.tight_layout()
    folder = run.OUT / 'figures'; folder.mkdir(exist_ok=True)
    for extension in ('png', 'svg'):
        fig.savefig(folder / ('fixed118_training_K_comparison.' + extension), dpi=180, bbox_inches='tight')
    plt.close(fig)
    lines = ['# 原固定118例：当前纯IVIM五模型验证', '',
        '日期：2026年10月21日。使用用户此前保留的原固定118例（83阳性、35阴性），'
        '精确恢复原82例训练（58阳性、24阴性）／36例测试（25阳性、11阴性）名单。'
        '没有重新抽样或按AUC挑病例。', '',
        '## 本轮五模型结果', '',
        '在原82例训练内进行外层五折／内层五折。每个外层折中，k＝2～6、三种D／f聚类输入、'
        '两种标准化、特征和C均在该折内层选择。最终在全部82例训练内层选优、重训和冻结后，'
        '五个模型各对原36例测试一次。±为五折AUC的样本SD。', '',
        '| 模型 | 最终k | 训练拟合AUC | 外层五折均值±SD | 冻结测试AUC（95%区间） |',
        '|---|---|---|---|---|']
    for family in run.study.FAMILIES:
        r = result[result.ModelKey == family].iloc[0]
        lines.append(f'| {LABELS[family]} | {int(r.K) if r.K else "—"} | {r.TrainFitAUC:.4f} | '
            f'{r.OuterMeanAUC:.4f}±{r.OuterSD:.4f} | {r.AUC:.4f}（{r.AUC_CI95Low:.4f}—{r.AUC_CI95High:.4f}） |')
    lines += ['', '## 最终设置', '',
        '| 模型 | 标准化 | 生境聚类输入 | k | 影像／临床特征数 | C |', '|---|---|---|---|---|---|']
    for r in result.itertuples(index=False):
        lines.append(f'| {LABELS[r.ModelKey]} | {NORMS[r.normalization]} | {INPUTS[r.ComboID]} | '
            f'{int(r.K) if r.K else "—"} | {int(r.ImageFeatureCount)}／{int(r.ClinicalFeatureCount)} | {r.C:g} |')
    lines += ['', '## 固定k的训练内对照', '',
        '本表固定k后，在每个外层训练部分的内层选择输入、标准化、特征及参数。'
        '这些固定k诊断模型未额外计算测试集AUC，不能与上表完整选k流程混用。', '',
        '| 模型 | k | 训练拟合AUC | 外层五折均值±SD |', '|---|---|---|---|']
    for family in ('Habitat', 'HabitatClinical'):
        for r in diag[diag.family == family].sort_values('K').itertuples(index=False):
            lines.append(f'| {LABELS[family]} | {r.K} | {r.TrainFitAUC:.4f} | {r.OuterMeanAUC:.4f}±{r.OuterSD:.4f} |')
    lines += ['', f'![原118例k对照]({(folder / "fixed118_training_K_comparison.png").as_posix()})', '',
        '## 与以前“效果不错”的118例结果的关系', '',
        '保留的历史多序列生境模型测试AUC为0.7709，训练拟合为0.6200，训练OOF为0.4763；'
        'ADC生境模型测试AUC为0.7964，训练拟合为0.5603，训练OOF为0.4274。'
        '这些历史模型包含不同序列、特征定义与验证方式，其较高测试AUC并不等于已证明外层五折泛化较好。'
        '本轮使用相同病例及原划分，运行当前纯IVIM方法，不是恢复历史模型权重。', '',
        '| 保留历史主分析 | 训练拟合AUC | 当时内层均值AUC | 训练OOF AUC | 测试AUC |',
        '|---|---|---|---|---|']
    mapping = {'H': 'Habitat', 'W': 'Whole', 'C': 'Clinical', 'HC': 'HabitatClinical', 'WC': 'WholeClinical'}
    for r in historical[historical.Panel == 'primary_fusion'].itertuples(index=False):
        lines.append(f'| {LABELS[mapping[r.Model]].replace(chr(73)+chr(86)+chr(73)+chr(77), chr(22810)+chr(24207)+chr(21015))}（历史多序列口径） | {r.TrainAUC:.4f} | '
            f'{r.InnerMeanAUC:.4f} | {r.TrainingOOFAUC:.4f} | {r.TestAUC:.4f} |')
    lines += ['', '## 对照范围与核查', '',
        '- 除数据集恢复为原118例／82∶36外，保持上一轮纯IVIM方法、1540种内层配置、'
        '质量规则、预处理、分类器及选择目标不变。五类模型分别重新训练；未把已在A／B组训练的模型'
        '直接作用于全部118例并将其当作独立测试。',
        '- 生境156个候选特征，全肿瘤为D／f各93个共186个候选特征；不使用T2、ADC或D*影像特征。'
        '临床沿用已审计11项术前变量，包含常规MRI人工测量信息。',
        '- 可复用的逐患者、无结局依赖影像缓存按索引和哈希复用；缺少的病例重新提取同定义特征。'
        'k＝3原公式参考缓存独立核对，所有有监督筛选与预处理仍在对应训练折内拟合。',
        '- 本轮36例测试集属于此前已多次查看的原队列测试，仍为探索性内部验证。'
        '恢复原病例集合不会使它重新成为未查看的独立验证。',
        '- 原版结果和正式采用决定保持原样。未重新采用此前排除的迁移实验或20261010监督特征选择模型。',
        '- 已核查内层候选AUC、外层隔离、最终训练OOF阈值、手工sigmoid概率、'
        '5000次分层患者bootstrap区间与同病例配对模型差值。', '',
        '完整数值见five_models_summary.csv；逐折与逐患者记录见outer_metrics.csv、'
        'nested_predictions.csv、test_predictions.csv；特征与系数见selected_features.csv。', '']
    report = run.OUT / '原118例纯IVIM五模型_82训练36测试报告_20261021.md'
    report.write_text('\n'.join(lines), encoding='utf-8')
    protected = run.study.protect()
    run.save(run.OUT / 'delivery_manifest.json', {'passed': True, 'outputs': {
        str(p.relative_to(run.OUT)): run.digest(p) for p in run.OUT.rglob('*') if p.is_file()},
        'protected_files_unchanged': protected, 'report': report.name, 'formal_model_use': False,
        'historical_results_replaced': False, 'final_test_models': 5, 'test_stage_count': 1})
    print(result[['ModelKey', 'K', 'TrainFitAUC', 'OuterMeanAUC', 'OuterSD', 'AUC']].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
