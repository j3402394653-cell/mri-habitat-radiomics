# -*- coding: utf-8 -*-
"""Ordered habitat and quantitative-value improvement versus matched118 controls."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import run_ordered_fixed118_20261023 as run
from verify_ivim_df63_20261020 import draws

LABELS={'Habitat':'①生境','Whole':'②全肿瘤影像组学','Clinical':'③临床','HabitatClinical':'④生境＋临床','WholeClinical':'⑤全肿瘤影像组学＋临床'}


def main():
    run.frozen()
    assert json.loads((run.OUT/'inference_verification.json').read_text(encoding='utf-8'))['passed']
    result=run.study.read('training_choices').merge(run.study.read('test_metrics'),on=['Group','ModelKey'],validate='one_to_one',suffixes=('','_Test'))
    selected=run.study.read('selected_features')
    for mod in ('T2','ADC','IVIM','Clinical'):
        result[mod+'Count']=[int(((selected.ModelKey==family)&(selected.Block==mod)).sum()) for family in result.ModelKey]
    run.csv('five_models_summary',result)
    previous=run.study.last.read('five_models_summary').set_index('ModelKey')
    newp,oldp=run.study.read('test_predictions'),run.study.last.read('test_predictions')
    cohort,_,_,te=run.study.inputs('Fixed118');ids=cohort.AnalysisID.iloc[te]
    scores={}
    for family in run.study.FAMILIES:
        scores['New_'+family]=newp[newp.ModelKey==family].set_index('AnalysisID').loc[ids].Probability.to_numpy()
        scores['Old_'+family]=oldp[oldp.ModelKey==family].set_index('AnalysisID').loc[ids].Probability.to_numpy()
    samples=draws(cohort.Outcome.to_numpy(int)[te],scores,2026102371)
    comparison=[]
    for r in result.itertuples(index=False):
        old=previous.loc[r.ModelKey]
        low,high=np.quantile(samples['New_'+r.ModelKey]-samples['Old_'+r.ModelKey],[.025,.975])
        comparison.append({'ModelKey':r.ModelKey,'PreviousOuterMeanAUC':old.OuterMeanAUC,'NewOuterMeanAUC':r.OuterMeanAUC,
            'OuterDifference':r.OuterMeanAUC-old.OuterMeanAUC,'PreviousTestAUC':old.AUC,'NewTestAUC':r.AUC,
            'TestDifference':r.AUC-old.AUC,'DifferenceCI95Low':low,'DifferenceCI95High':high})
    run.csv('comparison_with_lowdim_fusion',comparison)
    run.csv('paired_bootstrap5000',pd.DataFrame(samples))
    folder=run.OUT/'figures';folder.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(9,4.6));xx=np.arange(5)
    for offset,key,color,label in [(-.24,'TrainFitAUC','#93b9d7','Training fit'),(0,'OuterMeanAUC','#2675a6','Nested outer5'),(.24,'AUC','#dd8a32','Frozen test36')]:
        ax.bar(xx+offset,result.set_index('ModelKey').loc[list(run.study.FAMILIES),key],width=.23,label=label,color=color)
    ax.set(xticks=xx,xticklabels=['Habitat','Whole','Clinical','Habitat+Clinical','Whole+Clinical'],ylim=(0,1),ylabel='AUC',title='Original118: ordered habitats + raw quantitative values + LR/RBF')
    ax.axhline(.5,color='gray',linestyle=':',linewidth=1);ax.grid(axis='y',alpha=.15);ax.legend(ncol=3,fontsize=8);fig.tight_layout()
    for ext in ('png','svg'):
        fig.savefig(folder/('ordered_quantitative_five_models.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)
    lines=['# 原118例：有序生境与原始定量信息改进', '', '日期：2026年10月23日。精确保留原82例训练／36例测试及相同外层五折／内层五折。所有候选设定在本轮测试前固定；不重新抽病例。', '',
        '## 本轮改动', '',
        '- 生境保留按强度排序的H1／H2／H3区域身份；不再只用三生境的均值、SD、极差汇总。比较321个固定纹理精简特征与1242个完整区域特征，均不含整体形状预测量。',
        '- T2、ADC、IVIM各在原生空间独立聚类后融合特征，k固定3；IVIM使用D、f强度及3 mm均值、SD共六维，不加入D*。本轮不沿用此前63组合选出的三条输入作为先验筛选。',
        '- 全肿瘤候选为372个已有影像组学特征＋30个原始ADC／D／f统计量；原始数值由未截尾Z-score及保存的均值、SD恢复。恢复数值已核对，未重新认证扫描物理单位；不跨患者比较原始T2强度。',
        '- 内层比较L2逻辑回归与RBF-SVM、5／15／全部合格特征、两种标准化及5／11项临床变量。分类器使用训练类别平衡权重；没有删减病例。',
        '- 所有有监督筛选、ICC≥0.75、完整率≥0.9、相关性去重和预处理均在对应训练折内完成。选择目标沿用内层均值、折间波动及拟合差距惩罚。', '',
        '## 五模型结果', '', '每个外层训练部分独立选择完整模型；最终在全部82例内层选择后拟合、冻结、核查。五个最终模型各对原36例评分一次。±为五个外层折AUC的样本SD。', '',
        '| 模型 | 训练拟合AUC | 外层五折均值±SD | 测试AUC（95%区间） |', '|---|---|---|---|']
    for r in result.itertuples(index=False):
        lines.append(f'| {LABELS[r.ModelKey]} | {r.TrainFitAUC:.4f} | {r.OuterMeanAUC:.4f}±{r.OuterSD:.4f} | {r.AUC:.4f}（{r.AUC_CI95Low:.4f}—{r.AUC_CI95High:.4f}） |')
    lines+=['','## 最终配置与实际使用序列','','| 模型 | 表征 | 分类器 | 标准化 | T2／ADC／IVIM／临床特征数 | C／gamma |','|---|---|---|---|---|---|']
    for r in result.itertuples(index=False):
        lines.append(f'| {LABELS[r.ModelKey]} | {r.view} | {r.method} | {r.normalization} | {r.T2Count}／{r.ADCCount}／{r.IVIMCount}／{r.ClinicalCount} | {r.C:g}／{r.gamma} |')
    lines+=['','允许筛选后某个序列未进入最终模型，实际使用情况以上表为准。RBF-SVM输出的sigmoid间隔值只作为排序评分，不能解释为已校准患病风险；AUC和训练内阈值均使用相同评分定义。RBF模型没有单个原始特征的固定线性系数，selected_features.csv中不编造这类系数。', '',
        '## 与上一轮低维评分融合比较','','| 模型 | 外层均值：原→本轮 | 测试AUC：原→本轮 | 测试差值及配对95%区间 |','|---|---|---|---|']
    for r in pd.DataFrame(comparison).itertuples(index=False):
        lines.append(f'| {LABELS[r.ModelKey]} | {r.PreviousOuterMeanAUC:.4f} → {r.NewOuterMeanAUC:.4f} | {r.PreviousTestAUC:.4f} → {r.NewTestAUC:.4f} | {r.TestDifference:+.4f} [{r.DifferenceCI95Low:+.4f}, {r.DifferenceCI95High:+.4f}] |')
    best=result.sort_values('OuterMeanAUC',ascending=False).iloc[0]
    lines+=['',f'按预先侧重的外层均值，本轮最高为{LABELS[best.ModelKey]}：{best.OuterMeanAUC:.4f}±{best.OuterSD:.4f}；其训练拟合AUC为{best.TrainFitAUC:.4f}，测试AUC为{best.AUC:.4f}。模型比较还需结合折间波动、拟合差距和配对区间，不按本轮最高测试值追加筛选。','',
        '## 验证与采用边界','','原36例已经被多轮探索查看，本轮仍属探索性内部复验。嵌套五折只评估本次固定候选流程，不能消除历史探索选择影响。本轮不自动替换原版采用结果，也未重新使用此前排除的迁移实验或20261010监督模型。','',
        '已核对原始有序特征缓存、定量图数值恢复、内层所有候选的独立排序AUC、外层隔离、最终训练OOF阈值、LR／RBF核函数手工评分、5000次分层bootstrap区间及同病例模型差值。','',
        f'![五模型对照]({(folder/"ordered_quantitative_five_models.png").as_posix()})','',
        '完整数值：five_models_summary.csv；对照：comparison_with_lowdim_fusion.csv；逐病例评分：test_predictions.csv；原始输入：selected_features.csv。','']
    report=run.OUT/'原118例有序生境与定量信息改进报告_20261023.md'
    report.write_text('\n'.join(lines),encoding='utf-8')
    run.save(run.OUT/'delivery_manifest.json',{'passed':True,'report':report.name,'formal_model_use':False,'historical_results_replaced':False,
        'protected_files_unchanged':run.study.protect(),'test_model_calls':5,'test_stage_count':1,
        'outputs':{str(p.relative_to(run.OUT)):run.digest(p) for p in run.OUT.rglob('*') if p.is_file() and p.name!='delivery_manifest.json'}})
    print(result[['ModelKey','view','method','TrainFitAUC','OuterMeanAUC','OuterSD','AUC']].to_string(index=False),flush=True)


if __name__=='__main__':
    main()
