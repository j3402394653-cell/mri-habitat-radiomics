# -*- coding: utf-8 -*-
"""Verify direct-PLS outputs and report both prespecified four-stream attempts."""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,confusion_matrix
from verify_ivim_df63_20261020 import draws
from verify_random120_five_models_20261023 import manual
import run_dualroute_directpls_20261025 as run

LABELS={'HabitatOriginal':'原路线·纯生境','HabitatClinicalOriginal':'原路线·生境＋临床',
        'Habitat85':'85%路线·纯生境','HabitatClinical85':'85%路线·生境＋临床'}


def report():
    run.frozen()
    cohort,_,tr,te=run.base.old.inputs();truth=cohort.Outcome.to_numpy(int)[te]
    training=run.read('training_summary').set_index('ModelKey');test=run.read('test_metrics').set_index('ModelKey')
    predictions=run.read('test_predictions');old=run.parent.read('test_predictions')
    d=run.base.route_tables(joblib.load(run.base.OUT/'test_features.joblib')[1.],'ContourMedian')
    tables={**d,**joblib.load(run.parent.OUT/'test_clinical.joblib')}
    scores,prior,rows={}, {},[]
    for key,t,joint in run.STREAMS:
        f=joblib.load(run.OUT/'models'/(key+'.joblib'));p=predictions[predictions.ModelKey==key].set_index('Index').loc[te]
        np.testing.assert_array_equal(p.AnalysisID,cohort.AnalysisID.iloc[te]);np.testing.assert_array_equal(p.Outcome,truth)
        np.testing.assert_allclose(p.Probability,manual(f,tables,te),atol=1e-9,rtol=0)
        m=test.loc[key];assert m.Threshold==f['threshold'];np.testing.assert_allclose(m.TestAUC,roc_auc_score(truth,p.Probability),atol=1e-12)
        np.testing.assert_array_equal(p.Prediction,p.Probability>=m.Threshold)
        tn,fp,fn,tp=confusion_matrix(truth,p.Prediction,labels=[0,1]).ravel();np.testing.assert_array_equal([tn,fp,fn,tp],[m.TN,m.FP,m.FN,m.TP])
        scores[key]=p.Probability.to_numpy();prior['Prior_'+key]=old[old.ModelKey==key].set_index('Index').loc[te].Probability.to_numpy()
        rows.append({'ModelKey':key,**training.loc[key].to_dict(),**m.to_dict(),
                     'BothTrainOuter075':bool(training.loc[key,'TrainFitAUC']>=.75 and training.loc[key,'OuterMeanAUC']>=.75)})
    samples=draws(truth,{**scores,**prior},run.SEED)
    for key,t,joint in run.STREAMS:
        np.testing.assert_allclose(np.quantile(samples[key],[.025,.975]),test.loc[key,['AUC_CI95Low','AUC_CI95High']].to_numpy(float),atol=1e-12)
    paired=run.read('paired_test_differences')
    for r in paired.itertuples():
        key=r.Comparison[:-len('_minus_Prior')]
        np.testing.assert_allclose(np.quantile(samples[key]-samples['Prior_'+key],[.025,.975]),[r.CI95Low,r.CI95High],atol=1e-12)
    run.csv('primary_comparison',rows)
    # Report every attempted architecture,including the failed first selection.
    studyrows=[]
    for label,directory in [('保留基线',run.parent.OUT),('第一轮：评分融合细化',run.first.OUT),('第二轮：直接PLS',run.OUT)]:
        tt=pd.read_csv(directory/'training_summary.csv',float_precision='round_trip').set_index('ModelKey')
        mt=pd.read_csv(directory/'test_metrics.csv',float_precision='round_trip').set_index('ModelKey')
        for key,t,joint in run.STREAMS:
            studyrows.append({'Study':label,'ModelKey':key,'TrainFitAUC':tt.loc[key,'TrainFitAUC'],
                'OuterMeanAUC':tt.loc[key,'OuterMeanAUC'],'OuterSD':tt.loc[key,'OuterSD'],'TestAUC':mt.loc[key,'TestAUC'],
                'BothTrainOuter075':bool(tt.loc[key,'TrainFitAUC']>=.75 and tt.loc[key,'OuterMeanAUC']>=.75)})
    run.csv('all_attempts_vs_baseline',studyrows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12.8,4.6))
    keys=[k for k,t,joint in run.STREAMS];pos=np.arange(4)
    for i,(label,color) in enumerate([('保留基线','#8f9ba5'),('第一轮：评分融合细化','#a96f83'),('第二轮：直接PLS','#277da1')]):
        vals=[next(r['OuterMeanAUC'] for r in studyrows if r['Study']==label and r['ModelKey']==k) for k in keys]
        axes[0].bar(pos+(i-1)*.25,vals,.25,color=color,label=['Retained baseline','Stack refinement','Direct PLS'][i])
    axes[0].set(xticks=pos,xticklabels=['Original H','Original H+C','Fixed85 H','Fixed85 H+C'],ylim=(0,1.02),ylabel='All-five outer mean AUC',title='All attempts retained')
    axes[0].axhline(.75,color='gray',ls=':',lw=1);axes[0].legend(loc='lower left',fontsize=8)
    for key,t,joint in run.STREAMS:
        z=run.read('outer_metrics');z=z[z.ModelKey==key].sort_values('Fold')
        axes[1].plot(z.Fold,z.AUC,'o-',label=key+' %.4f'%training.loc[key,'OuterMeanAUC'])
    axes[1].set(xticks=range(1,6),ylim=(0,1.02),xlabel='Outer fold',ylabel='AUC',title='Direct PLS / all five folds')
    axes[1].axhline(.75,color='gray',ls=':',lw=1);axes[1].legend(loc='lower right',fontsize=8)
    for ax in axes:ax.grid(alpha=.15)
    fig.tight_layout()
    for ext in ('png','svg'):fig.savefig(run.OUT/('figures/two_attempts_all4streams.'+ext),dpi=180)
    plt.close(fig)
    lines=['# 两条研究路线：纯生境与联合模型的两轮改进','','日期：2026年10月25日。用户指定保留原路线和固定85%路线，每轮均评估纯生境与生境＋临床。研究方向已写入RESEARCH_DIRECTIONS_ORIGINAL_FIXED85_20261025.json。','',
      '## 全部尝试与保留基线','','| 轮次 | 模型 | 真实训练拟合AUC | 完整外层均值±SD | 测试AUC | 训练/外层均≥0.75 |','|---|---|---:|---:|---:|---|']
    for r in studyrows:
        lines.append('| %s | %s | %.4f | %.4f ± %.4f | %.4f | %s |'%(r['Study'],LABELS[r['ModelKey']],r['TrainFitAUC'],r['OuterMeanAUC'],r['OuterSD'],r['TestAUC'],'是' if r['BothTrainOuter075'] else '否'))
    lines+=['','## 第二轮分类表现','','| 模型 | 测试AUC（95% CI） | 阈值 | 敏感度 | 特异度 | 准确率 | TP/FN/TN/FP |','|---|---:|---:|---:|---:|---:|---|']
    for r in rows:
        lines.append('| %s | %.4f（%.4f～%.4f） | %.6f | %.2f%% | %.2f%% | %.2f%% | %d/%d/%d/%d |'%(LABELS[r['ModelKey']],r['TestAUC'],r['AUC_CI95Low'],r['AUC_CI95High'],r['Threshold'],100*r['Sensitivity'],100*r['Specificity'],100*r['Accuracy'],r['TP'],r['FN'],r['TN'],r['FP']))
    lines+=['','## 第二轮训练内选定配置','','| 模型 | 影像视图 | PLS成分 | C | 内层均值±SD |','|---|---|---:|---:|---:|']
    for r in rows:
        lines.append('| %s | %s | %d | %g | %.4f ± %.4f |'%(LABELS[r['ModelKey']],r['image_key'],r['components'],r['C'],r['MeanInnerAUC'],r['InnerSD']))
    lines+=['','## 设计和解释','',
      '第一轮保留12路PLS基础模型，比较PLS1/2/两者合用、融合C和锚定，共36配方/模型。训练内选优后实际训练拟合均达到约0.75，但外层均值下降。全部固定配方外层诊断也未达到0.75，因此没有按外层最高分挑一个固定模型作为主结果。第一轮四个冻结主流程仍完整测试并保留。','',
      '第二轮在第一轮新增测试之前锁定。它保留完全相同影像与生境特征、原/85%方向筛选，改为直接在高斯特征上拟合PLS，再接平衡L2逻辑回归；联合模型在PLS之后加入11项术前临床变量。省去二层概率评分堆叠，用于检验评分层训练/OOF分布差异是否影响结果。没有更改分区、轮廓或原始生境特征，也不是已放弃的独立区域L1拼接实验。','',
      '每模型36候选：两种标准化×三个视图×PLS1/2/3×C=0.03/0.3。每个外层训练部分独立执行5折×3次内层验证，QC、40次bootstrap方向筛选、缺失填补、高斯变换、PLS和临床预处理全部限于当前内层训练部分。直接模型没有二层评分融合，因此不需要为了拟合融合器再生成第三层基础OOF。','',
      '两轮均优先当前训练部分实际拟合≥0.75的可用候选，再依据内层均值、SD和过拟合差距选择；没有者如实降为0.70或标记未达。所有外层折保留，实际训练拟合与汇总OOF AUC分开。第二轮选择配置和全五折结果只用于预先定义主流程评价，没有根据测试AUC另换候选。','',
      '同一A120中的84训练/36测试及原内外层划分保持不变。每轮四个主流程与各自84例外层OOF分类阈值全部冻结后，进行一次共同测试阶段。没有逐一测试固定候选，没有测试后调参、换阈值、剔除低折或翻转概率。原正式模型采用状态保持不变。','',
      '这批病例、折划分和36例测试已有多轮探索与选择背景，结果不属于新的独立或外部验证。5000次配对分层bootstrap只反映固定36例评分的病例抽样不确定性，不包含重训或反复选择的偏差。联合模型改善不能归为纯生境改善。','',
      '第一轮核验记录：发现1.11e-16的概率数值舍入差被零容差拦截，在测试前存档并仅修正核验概率容差为1e-12；系数仍严格相同且AUC独立核对，模型、预测、选择均未改变。存档见第一轮amendments/baseline_probability_roundoff_before_test。','',
      '![两轮全部四流程对照]('+(run.OUT/'figures/two_attempts_all4streams.png').as_posix()+')','',
      '文件：all_attempts_vs_baseline.csv、primary_comparison.csv、outer_metrics.csv、selected_configurations.csv、test_predictions.csv、paired_test_differences.csv、training_verification.json。']
    (run.OUT/'双路线两轮改进完整报告_20261025.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    run.save(run.OUT/'report_verification.json',{'passed':True,'direct_primary_streams':4,'all_attempts_reported':12,
        'test_metrics_classification_and_bootstrap_verified':True,'all5folds_kept':True,'formal_adoption_changed':False})
    print(pd.DataFrame(studyrows).to_string(index=False),flush=True)


if __name__=='__main__':report()
