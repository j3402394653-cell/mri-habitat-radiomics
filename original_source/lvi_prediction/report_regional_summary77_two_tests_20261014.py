# -*- coding: utf-8 -*-
"""Verified 77/33/28 results, classifier-template control and frozen-group comparison."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.metrics import roc_curve, roc_auc_score
import run_regional_summary77_two_tests_20261014 as run

OUT = run.OUT
NORMS = {'PercentileZ': '截尾＋Z-score', 'RobustZ': '稳健Z-score'}
ARMS = {'Matched': '昨日配置重训', 'Retuned': '训练内重选'}
GROUPS = {'TestA33': '测试A：33例', 'TestB28': '测试B：28例', 'Combined61': '合并61例'}


def label(name):
    arm, norm = name.split('__')
    return '训练内联合选择参考' if norm == 'Adaptive' else NORMS[norm]+'／'+ARMS[arm]


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '|'+'|'.join(['---']*len(headers))+'|']+
                     ['| '+' | '.join(map(str, row))+' |' for row in rows])


def comparisons(choices, metrics):
    oldchoices = run.read(run.previous.OUT, 'final_training_choices').set_index('ModelKey')
    oldmetrics = run.read(run.previous.OUT, 'test_metrics').set_index('ModelKey')
    records = []
    for n in run.NORMS:
        c, m = oldchoices.loc[n], oldmetrics.loc[n]
        records.append({'Normalization': n, 'Stage': 'Previous90', 'TrainN': 90, 'FeatureCount': c.FeatureCount,
            'TrainFitAUC': c.TrainFitAUC, 'OuterMeanAUC': c.OuterMeanAUC, 'OuterSD': c.OuterSD, 'PooledOOFAUC': c.PooledOOFAUC,
            'Test48AUC': m.AUC, 'TestAAUC': np.nan, 'TestBAUC': np.nan, 'Combined61AUC': np.nan})
        for arm in ARMS:
            name = arm+'__'+n
            c = choices.loc[name]
            records.append({'Normalization': n, 'Stage': arm, 'TrainN': 77, 'FeatureCount': c.FeatureCount,
                'TrainFitAUC': c.TrainFitAUC, 'OuterMeanAUC': c.OuterMeanAUC, 'OuterSD': c.OuterSD, 'PooledOOFAUC': c.PooledOOFAUC,
                'Test48AUC': np.nan, 'TestAAUC': metrics.loc[(name, 'TestA33'), 'AUC'], 'TestBAUC': metrics.loc[(name, 'TestB28'), 'AUC'],
                'Combined61AUC': metrics.loc[(name, 'Combined61'), 'AUC']})
    comparison = pd.DataFrame(records)
    comparison.to_csv(OUT/'previous90_new77_comparison.csv', index=False, encoding='utf-8-sig')
    return comparison


def figures(metrics, predictions):
    font = FontProperties(fname='C:/Windows/Fonts/simhei.ttf')
    plt.rcParams.update({'font.family': font.get_name(), 'font.size': 11, 'axes.unicode_minus': False})
    directory = OUT/'figures'
    directory.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.3, 5.7), constrained_layout=True)
    groups = ['TestA33', 'TestB28', 'Combined61']
    for ax, n in zip(axes, run.NORMS):
        for arm, color, offset, ls in [('Matched', '#b77632', -.05, '--'), ('Retuned', '#277e78', .05, '-')]:
            rows = metrics.loc[[(arm+'__'+n, g) for g in groups]]
            value = rows.AUC.to_numpy()
            error = np.maximum(np.vstack([value-rows.AUC_CI95Low.to_numpy(), rows.AUC_CI95High.to_numpy()-value]), 0)
            ax.errorbar(np.arange(3)+offset, value, yerr=error, fmt='o', ls=ls, color=color, capsize=4, label=ARMS[arm])
        ax.axhline(.75, color='#b55d67', ls='--', lw=1)
        ax.axhline(.5, color='#999999', ls='--', lw=.8)
        ax.set(xticks=np.arange(3), xticklabels=['测试A\n33例', '测试B\n28例', '合并\n61例'], ylim=(0, 1.03),
               ylabel='AUC及95%病例bootstrap区间', title=NORMS[n]+'：77例重训后冻结测试')
        ax.grid(axis='y', alpha=.18)
        ax.legend(loc='lower left', fontsize=10)
    for ext in ('png', 'svg'):
        fig.savefig(directory/('区域汇总77例重训_双测试集AUC.'+ext), dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.4, 6), constrained_layout=True)
    for n, color in [('PercentileZ', '#b77632'), ('RobustZ', '#277e78')]:
        for arm, ls in [('Matched', '--'), ('Retuned', '-')]:
            name = arm+'__'+n
            rows = predictions[predictions.ModelKey == name]
            fpr, tpr, _ = roc_curve(rows.Outcome, rows.Probability)
            assert abs(roc_auc_score(rows.Outcome, rows.Probability)-metrics.loc[(name, 'Combined61'), 'AUC']) < 1e-12
            ax.plot(fpr, tpr, color=color, ls=ls, lw=1.8, label=NORMS[n]+('同配置 ' if arm == 'Matched' else '重选 ')+f'{metrics.loc[(name, "Combined61"), "AUC"]:.3f}')
    ax.plot([0, 1], [0, 1], '--', color='#999999', lw=.8)
    ax.set(xlim=(0, 1), ylim=(0, 1.03), xlabel='1-特异度', ylabel='敏感度', title='固定77例训练：合并61例冻结测试ROC')
    ax.grid(alpha=.18)
    ax.legend(loc='lower right', fontsize=9)
    for ext in ('png', 'svg'):
        fig.savefig(directory/('区域汇总77例重训_合并61例ROC.'+ext), dpi=180)
    plt.close(fig)


def report(choices, metrics, comparison):
    names = [arm+'__'+n for n in run.NORMS for arm in ARMS]
    groupdiff = run.read(OUT, 'testA_minus_testB_auc').set_index('ModelKey')
    paired = run.read(OUT, 'retuned_minus_matched_auc')
    joint = choices.loc['Retuned__Adaptive']
    body = ['# 昨日区域汇总生境方法：77例重训与两套测试集', '',
        '日期：2026年10月14日。按已固定的77例训练、33例测试A、28例测试B执行；新实验不替换原正式采用结果。', '',
        '## 主要结果', '', table(['标准化', '方案', '输入数', '训练拟合AUC', '外层五折均值±SD', '汇总OOF AUC', '测试A 33例AUC', '测试B 28例AUC', '合并61例AUC'],
            [[NORMS[c.normalization], ARMS[name.split('__')[0]], int(c.FeatureCount), f'{c.TrainFitAUC:.4f}',
              f'{c.OuterMeanAUC:.4f}±{c.OuterSD:.4f}', f'{c.PooledOOFAUC:.4f}',
              *[f'{metrics.loc[(name, g), "AUC"]:.4f}' for g in GROUPS]] for name in names for c in [choices.loc[name]]]), '',
        '训练拟合来自最终77例模型；外层每例OOF来自未见该病例的训练子模型，外层均值与汇总OOF是不同的统计量。测试A、B及合并61例均来自同一个冻结模型的同一次批量预测。合并AUC不是两组AUC的简单平均。', '',
        '## 两种重训方案', '',
        '昨日配置重训：保留昨日最终模板。截尾模型为Association排序、PCA2分量＋L2逻辑回归、C＝0.03；稳健模型为Stability排序、PCA2分量＋L2逻辑回归、C＝0.03；预算为全部质量合格去相关项。所有特征门槛、具体输入项、填补、缩放、PCA中心/载荷及回归权重在当前训练病例中重新估计，不复制旧90例模型参数。该方案不在本轮重新选择分类器超参数；内层逐折特征学习及OOF用来构建最终训练阈值。', '',
        '训练内重选：固定同一330项区域汇总表达，仅在训练77例的内层五折选择排序、输入预算20/50/全部、Ridge、ElasticNet、PCA、PLS或15次bootstrap Ridge集成及其参数。160个候选等于昨日480项中的区域汇总子空间，不再比较原始或紧凑表达。依据为mean AUC − 0.10×折间SD − 0.30×max(mean拟合AUC−mean验证AUC−0.10,0)，常数训练概率或未收敛候选不可选。', '',
        '两组都保留逐患者k＝3原生境、同一ROI及图像；T2＋ADC＋IVIM（D、f），不加入D*、临床或直接全肿瘤特征。未采用本日共用生境中心方案，也未重新引入被排除的迁移实验。330项为同类三生境指标的均值/SD/范围，加区域占比离散度、最大值、熵；它们来源于生境区域，不是重新提取的全肿瘤纹理。', '',
        '质量筛选仍为当前训练折ICC≥0.8、完整率≥90%、非常数；ICC或20次分层bootstrap关联排序、|Spearman|＜0.95；表格特征训练中位数填补、P2.5/P97.5截尾、中位数/IQR缩放。图像标准化与表格输入缩放是不同步骤。', '',
        '## 分组、嵌套验证与冻结', '', table(['用途', '例数', '阳性', '阴性'],
            [['训练', 77, 52, 25], ['测试A', 33, 23, 10], ['测试B', 28, 19, 9], ['合并测试', 61, 42, 19]]), '',
        '严格沿用split_manifest.json中的77/33/28名单，未另选种子或重抽病例。7∶3指训练77∶测试A33；28例测试B预留于110例开发集之外。三组互不重叠。', '',
        '外层五折seed＝42，内层62201—62205；最终77例内层选择seed＝62212。训练标签向量只包含77例结局，其余61例为−1；训练特征也仅保留77例。逐患者、无结局的原生境缓存可复用；当前训练质量门槛、关联、相关性处理、预处理、降维、模型和阈值全部重新拟合。', '',
        '先完成全部外层/内层训练及阈值选择，再独立核验、保存模型哈希并冻结；之后两套测试在一个阶段一起预测，不在测试A之后改模型再测试B。分类阈值取最终配置训练77例OOF的Youden指数。95%区间为5000次按结局分层患者bootstrap；A、B使用独立种子，模型间同病例差值使用相同抽样以保留配对关系。', '',
        '## 最终配置', '', table(['模型', '排序', '分类器', 'C', '输入数', '有效维数/非零数', 'T2/ADC/IVIM数', '阈值'],
            [[label(name), c.ranking, c.method, c.C, int(c.FeatureCount), int(c.EffectiveDimensions),
              '/'.join(str(int(c[m+'Count'])) for m in ('T2', 'ADC', 'IVIM')), f'{c.Threshold:.6f}'] for name, c in choices.iterrows()]), '',
        '有效维数对PCA/PLS指分量数，对ElasticNet指非零系数数，其余指输入数。实际输入数可小于20/50预算；输入特征、线性系数和外层选择记录见selected_features.csv、outer_feature_selections.csv。集成系数为成员均值描述，预测为成员sigmoid概率均值。', '',
        f'训练内联合选择参考也在内层选择标准化：最终选{NORMS[joint.normalization]}、{joint.method}；训练拟合{joint.TrainFitAUC:.4f}，外层{joint.OuterMeanAUC:.4f}±{joint.OuterSD:.4f}，汇总OOF{joint.PooledOOFAUC:.4f}。最终模型与相应训练内重选的标准化模型相同，测试概率共享而不新增预测调用。其外层结果评估每折完整选择流程，可能与固定标准化不同。', '',
        '## 测试AUC及不确定性', '', table(['模型', '测试集合', 'AUC（95%区间）'],
            [[label(name), GROUPS[g], f'{r.AUC:.4f}（{r.AUC_CI95Low:.4f}—{r.AUC_CI95High:.4f}）']
             for name in choices.index for g in GROUPS for r in [metrics.loc[(name, g)]]]), '',
        '这些区间条件于已冻结概率，不包含整套开发过程反复重训的不确定性。两套测试同源且小，点估计差异需结合区间判断；来源于同一队列的两组不构成外部验证。', '',
        '## 测试A与测试B变化幅度', '', table(['模型', 'A−B ΔAUC（95%区间）'],
            [[label(name), f'{r.DeltaAUC:+.4f}（{r.CI95Low:.4f}—{r.CI95High:.4f}）'] for name, r in groupdiff.iterrows()]), '',
        'A/B病例互不相同，差值由两组各自分层bootstrap后相减计算，不按同病例配对计算；区间包含0不能证明两组等效，也不能只看较高那组宣称达标。', '',
        '## 重新选参数相对昨日配置的增益', '', table(['标准化', '测试集合', '重选−昨日配置 ΔAUC（95%区间）'],
            [[NORMS[r.First.split('__')[1]], GROUPS[r.Group], f'{r.DeltaAUC:+.4f}（{r.CI95Low:.4f}—{r.CI95High:.4f}）'] for _, r in paired.iterrows()]), '',
        '同一组病例内比较两个方案，所以此处使用配对bootstrap。多项比较未作多重校正，作为探索描述。', '',
        '## 冻结阈值下的分类表现', '', table(['模型', '集合', '敏感度', '特异度', '准确率', 'TP/FN/TN/FP'],
            [[label(name), GROUPS[g], f'{r.Sensitivity:.1%}', f'{r.Specificity:.1%}', f'{r.Accuracy:.1%}',
              '/'.join(str(int(r[s])) for s in ('TP', 'FN', 'TN', 'FP'))] for name in names for g in GROUPS for r in [metrics.loc[(name, g)]]]), '',
        '## 与昨日90/48结果对照', '', table(['标准化', '实验', '训练拟合', '外层均值±SD', '汇总OOF', '原48例测试', '新A／B／合并61例'],
            [[NORMS[r.Normalization], '昨日90/48' if r.Stage == 'Previous90' else ARMS[r.Stage]+'77例', f'{r.TrainFitAUC:.4f}',
              f'{r.OuterMeanAUC:.4f}±{r.OuterSD:.4f}', f'{r.PooledOOFAUC:.4f}',
              f'{r.Test48AUC:.4f}' if np.isfinite(r.Test48AUC) else '—',
              '—' if r.Stage == 'Previous90' else '/'.join(f'{r[s]:.4f}' for s in ('TestAAUC', 'TestBAUC', 'Combined61AUC'))] for _, r in comparison.iterrows()]), '',
        '昨日最终两组同样为区域汇总＋PCA2，但昨日完整选择流程还比较其他表达；本轮预先固定区域汇总表达。训练病例数、病例组成及测试名单都变化，昨日测试仅读取已保存结果，不重新调用旧模型。上表为描述性对照，不能把差值解释为相同病例、相同训练条件下的纯方法增益。', '',
        '## 判断', '']
    for name in names:
        aa, bb, combined = [metrics.loc[(name, g)] for g in GROUPS]
        c = choices.loc[name]
        body += [f'{label(name)}：拟合{c.TrainFitAUC:.4f}，外层均值{c.OuterMeanAUC:.4f}；测试A {aa.AUC:.4f}、B {bb.AUC:.4f}、合并{combined.AUC:.4f}。'+
                 ('两套测试点估计均达到0.75，仍需结合外层验证及区间判断稳定性。' if min(aa.AUC, bb.AUC) >= .75 else
                  '两套测试未同时达到0.75，尚不能据此认定稳定满足目标。'), '']
    robust_gain = paired[(paired.Group == 'Combined61')&(paired.First == 'Retuned__RobustZ')].iloc[0]
    body += [f'稳健重选相对昨日配置重训的合并61例增益为{robust_gain.DeltaAUC:+.4f}，95%区间{robust_gain.CI95Low:.4f}—{robust_gain.CI95High:.4f}。区间包含0，当前仅支持点估计提高，尚不足以确认明确优于同配置重训。', '',
        '不按测试最高AUC自动采用或调整模型。内部开发参考以训练内选择及其外层验证为依据，A/B及61例结果全部保留。当前实验为探索记录，不替换原正式模型。', '',
        '这些138例已被多轮探索，新77例模型的实际拟合没有使用本次61例结局，但新划分不能清除历史开发暴露。因此为既有队列内部复评，而非从未接触的独立外部验证。', '',
        '## 核验与文件', '',
        '原始1242项缓存重构330项，与昨日派生缓存逐项一致；训练候选960个目标从4800条内层折记录重算；外层模型验证标签翻转与验证特征数值改变不影响训练；最终模型、OOF及阈值重训重放。305条测试概率按独立sigmoid/集成公式核验，AUC、5000次区间、混淆计数、同病例方案差异及不同病例A/B差异独立计算。模型和阈值冻结后未变；详情见训练/推理核验JSON与delivery_manifest.json。', '',
        '![77例重训双测试集AUC]('+str((OUT/'figures/区域汇总77例重训_双测试集AUC.png').resolve()).replace('\\', '/')+')', '',
        '![61例冻结测试ROC]('+str((OUT/'figures/区域汇总77例重训_合并61例ROC.png').resolve()).replace('\\', '/')+')', '',
        '方法参考：[PCA官方文档](https://scikit-learn.org/1.3/modules/generated/sklearn.decomposition.PCA.html)、[嵌套交叉验证](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)。', '']
    path = OUT/'区域汇总生境77例重训_双测试集报告_20261014.md'
    path.write_text('\n'.join(body), encoding='utf-8')
    run.save(OUT/'report_generation.json', {'passed': True, 'report': path.name, 'sha256': run.digest(path),
        'model_predict_calls': 0, 'historical_model_calls': 0, 'comparison_rows': len(comparison),
        'training_selected_reference': joint.to_dict(), 'formal_model_use': False})


def main():
    run.frozen()
    assert json.loads((OUT/'inference_verification.json').read_text())['passed']
    choices = run.read(OUT, 'final_training_choices').set_index('ModelKey')
    metrics = run.read(OUT, 'test_metrics').set_index(['ModelKey', 'Group'])
    comparison = comparisons(choices, metrics)
    figures(metrics, run.read(OUT, 'test_predictions'))
    report(choices, metrics, comparison)
    print(comparison.to_string(index=False))


if __name__ == '__main__':
    main()
