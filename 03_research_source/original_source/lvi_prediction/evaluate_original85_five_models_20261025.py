# -*- coding: utf-8 -*-
"""Independent current-fold refit, score, shared baseline and report checks."""
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from threadpoolctl import threadpool_limits
import run_original85_five_models_20261025 as run

LABELS = {'HabitatOriginal':'① 生境','Habitat85':'① 生境','Whole':'② 全肿瘤影像组学',
          'Clinical':'③ 临床','HabitatClinicalOriginal':'④ 生境＋临床','HabitatClinical85':'④ 生境＋临床',
          'WholeClinical':'⑤ 全肿瘤影像组学＋临床'}


def clinical_reference(d,a,b):
    ra = run.five.latent.rows(d,a)
    rb = run.five.latent.rows(d,b)
    x = d['base'][ra]
    z = d['base'][rb]
    median = np.nanmedian(np.where(np.isfinite(x),x,np.nan),axis=0)
    median = np.where(np.isfinite(median),median,0.)
    filled = np.where(np.isfinite(x),x,median)
    lo,hi = np.quantile(filled,[.025,.975],axis=0)
    clipped = np.minimum(np.maximum(filled,lo),hi)
    center = np.median(clipped,axis=0)
    q25,q75 = np.quantile(clipped,[.25,.75],axis=0)
    scale = (q75-q25)/1.348979500392164
    scale = np.where(scale > 1e-12,scale,np.std(clipped,axis=0))
    scale = np.where(scale > 1e-12,scale,1.)
    return {'median':median,'low':lo,'high':hi,'center':center,'scale':scale,
            'xa':(clipped-center)/scale,'xb':(np.minimum(np.maximum(np.where(np.isfinite(z),z,median),lo),hi)-center)/scale}


def independent_refit(ctx,tables,y):
    a,b = ctx['train'],ctx['validation']
    cp = clinical_reference(tables['Clinical'],a,b)
    lo,hi = np.quantile(ctx['oof'],[.025,.975],axis=0)
    clipped = np.minimum(np.maximum(ctx['oof'],lo),hi)
    center,scale = clipped.mean(0),clipped.std(0)
    scale = np.where(scale > 1e-12,scale,1.)
    zo = (clipped-center)/scale
    za = (np.minimum(np.maximum(ctx['bag_train'],lo),hi)-center)/scale
    zb = (np.minimum(np.maximum(ctx['bag_validation'],lo),hi)-center)/scale
    aoo = np.column_stack([zo,cp['xa']])
    actual = np.column_stack([za,cp['xa']])
    validation = np.column_stack([zb,cp['xb']])
    estimator = LogisticRegression(C=.03,penalty='l2',solver='liblinear',class_weight='balanced',max_iter=3000,random_state=64212)
    estimator.fit(np.concatenate([aoo,actual],axis=0),np.tile(y[a],2),
                  sample_weight=np.r_[np.full(len(a),.8),np.full(len(a),.2)])
    return {'model':estimator,'clinical':cp,'train_probability':expit(actual@estimator.coef_[0]+estimator.intercept_[0]),
            'validation_probability':expit(validation@estimator.coef_[0]+estimator.intercept_[0]),
            'low':lo,'high':hi,'center':center,'scale':scale}


def manual_base_matrix(basis,tables,indices):
    d = run.base.route_tables(tables[1.],'ContourMedian')
    bags = []
    for row in basis['bag']:
        columns = []
        for f in row['models']:
            if f['classifier'] is None:
                columns.append(np.full(len(indices),f['prevalence']))
                continue
            dd = d[f['config']['image_key']]
            x = dd['base'][run.base.old.latent.rows(dd,indices)][:,f['indices']]
            x = np.where(np.isfinite(x),x,f['median'])
            x = np.clip(f['quantile'].transform(x),-2.5,2.5)
            columns.append(expit(x@f['coef_gaussian']+f['intercept_gaussian']))
        bags.append(np.column_stack(columns))
    return np.mean(bags,axis=0)


def manual_hc(f,basis,tables,clinical,indices):
    x = manual_base_matrix(basis,tables,indices)
    z = run.hc_design(f,x,clinical,indices)
    return expit(z@f['coef']+f['intercept'])


def verify():
    run.frozen()
    assert not (run.OUT/'test_started.json').exists()
    cohort,y,tr,te = run.base.old.inputs()
    clinical = joblib.load(run.OUT/'training_clinical.joblib')
    regenerated = run.five.clinical_source.clinical(tr)
    np.testing.assert_array_equal(clinical['Clinical']['base'],regenerated['base'])
    assert clinical['Clinical']['metadata'] == regenerated['metadata']
    tables = joblib.load(run.base.OUT/'training_features.joblib')
    inner = run.read('inner_fixed_hc_scores')
    metrics = run.read('outer_metrics')
    pp = run.read('nested_predictions')
    summary = run.read('training_summary').set_index('ModelKey')
    refits,mutations,maxerr = 0,0,0.
    for ph in run.PHASES:
        saved = joblib.load(run.OUT/'models'/(ph+'.joblib'))
        a,b = (tr,np.empty(0,int)) if ph == 'final' else run.base.old.previous.parent.folds('Random120')[int(ph[-1])-1]
        for _,t in run.SCHEMES:
            key = run.hkey(t,True)
            for rep,k,aa,bb in run.base.old.previous.parent.folds('Random120',ph):
                assert set(aa).isdisjoint(bb) and (set(aa)|set(bb)) == set(a)
                assert not (set(aa)|set(bb)) & (set(b)|set(te))
                ctx = run.source.context(ph,t,rep,k)
                np.testing.assert_array_equal(ctx['train'],aa)
                np.testing.assert_array_equal(ctx['validation'],bb)
                for fold in ctx['folds']:
                    assert set(fold['train']).isdisjoint(fold['validation'])
                    assert (set(fold['train'])|set(fold['validation'])) == set(aa)
                reference = independent_refit(ctx,clinical,y)
                row = inner[(inner.ModelKey == key)&(inner.Phase == ph)&(inner.Repeat == rep)&(inner.Fold == k)].iloc[0]
                np.testing.assert_allclose([row.FitAUC,row.AUC],
                       [roc_auc_score(y[aa],reference['train_probability']),roc_auc_score(y[bb],reference['validation_probability'])],atol=1e-12,rtol=0)
                refits += 1
            f = saved[key]
            ctx = run.source.context(ph,t)
            reference = independent_refit(ctx,clinical,y)
            refits += 1
            np.testing.assert_allclose(f['coef'],reference['model'].coef_[0],atol=1e-10,rtol=0)
            np.testing.assert_allclose(f['intercept'],reference['model'].intercept_[0],atol=1e-10,rtol=0)
            for name in ('low','high','center','scale'):
                np.testing.assert_allclose(f[name],reference[name],atol=1e-12,rtol=0)
            for name in ('median','low','high','center','scale'):
                np.testing.assert_allclose(f['clinical'][name],reference['clinical'][name],atol=1e-12,rtol=0)
            np.testing.assert_array_equal(f['train'],a)
            np.testing.assert_array_equal(f['validation'],b)
            basis = run.source.basis(ph,t)
            for row in basis['bag']:
                assert set(row['train']).isdisjoint(row['validation'])
                assert (set(row['train'])|set(row['validation'])) == set(a)
            for indices,expected in [(a,f['train_probability']),(b,f['evaluation_probability'])]:
                if not len(indices):
                    continue
                p = run.predict_hc(f,basis,tables,clinical,indices)
                manual = manual_hc(f,basis,tables,clinical,indices)
                maxerr = max(maxerr,float(np.max(np.abs(p-expected))),float(np.max(np.abs(manual-p))))
                np.testing.assert_allclose(p,expected,atol=1e-9,rtol=0)
                np.testing.assert_allclose(p,manual,atol=1e-9,rtol=0)
                if ph != 'final' and np.array_equal(indices,b):
                    rows = pp[(pp.ModelKey == key)&(pp.Fold == int(ph[-1]))].set_index('Index').loc[b]
                    np.testing.assert_allclose(rows.Probability,p,atol=1e-12,rtol=0)
            # Perturb all held-out targets; fixed current-training fit must be unchanged.
            mutated = y.copy()
            mutated[np.r_[b,te]] = 1-(np.arange(len(b)+len(te)) % 2)
            check = run.fit_hc(clinical,ctx,mutated,t)
            np.testing.assert_allclose(check['coef'],f['coef'],atol=0,rtol=0)
            mutations += 1
        print('Verified '+ph+':30private-context refits,2root fits,native/formula scores',flush=True)
    # Recheck retained three baselines on training only, with both coefficient spaces.
    from verify_random120_five_models_20261023 import manual
    common = joblib.load(run.five.OUT/'training_features.joblib')
    for ph in run.PHASES:
        models = joblib.load(run.five.OUT/'models'/(ph+'.joblib'))
        for key in run.SHARED:
            f = models[key]
            for indices,expected in [(f['train'],f['train_probability']),(f['validation'],f['evaluation_probability'])]:
                if len(indices):
                    p = run.five.predict(f,common,indices)
                    np.testing.assert_allclose(p,expected,atol=1e-9,rtol=0)
                    np.testing.assert_allclose(p,manual(f,common,indices),atol=1e-9,rtol=0)
    for key in run.UNIQUE:
        z = metrics[metrics.ModelKey == key]
        p = pp[pp.ModelKey == key].set_index('Index').loc[tr]
        assert len(z) == 5 and set(z.Fold) == set(range(1,6)) and p.index.is_unique
        np.testing.assert_array_equal(p.Outcome,y[tr])
        np.testing.assert_allclose([z.AUC.mean(),z.AUC.std(ddof=1),roc_auc_score(y[tr],p.Probability)],
                  summary.loc[key,['OuterMeanAUC','OuterSD','PooledOOFAUC']].to_numpy(float),atol=1e-12,rtol=0)
        np.testing.assert_allclose(run.five.threshold(y[tr],p.Probability.to_numpy()),summary.loc[key,'Threshold'],atol=1e-12,rtol=0)
    assert len(inner) == 180
    run.frozen()
    run.save(run.OUT/'training_verification.json',{'passed':True,'new_hc_independent_refits':refits,
         'heldout_label_mutation_checks':mutations,'max_probability_error':maxerr,'shared_baseline_replays':18,
         'exact_source_habitat_and_shared_metrics':True,'all5folds_kept':True,'new_test_calls':0,
         'no_outer_or_test_targets_in_fit':True,'new_HC_thresholds_own84outerOOF':True})
    print('Both newHC pipelines verified andfrozen;only2new36-model evaluations authorized',flush=True)


def report():
    run.frozen()
    assert json.loads((run.OUT/'test_complete.json').read_text(encoding='utf-8'))['passed']
    summary = run.read('training_summary').set_index('ModelKey')
    metrics = run.read('test_metrics').set_index('ModelKey')
    predictions = run.read('test_predictions')
    outer = run.read('outer_metrics')
    cohort,_,tr,te = run.base.old.inputs()
    truth = cohort.Outcome.to_numpy(int)[te]
    scores,maxerr = {},0.
    clinical = joblib.load(run.OUT/'test_clinical.joblib')
    tables = joblib.load(run.base.OUT/'test_features.joblib')
    final = joblib.load(run.OUT/'models/final.joblib')
    for _,t in run.SCHEMES:
        key = run.hkey(t,True)
        frame = predictions[predictions.ModelKey == key].set_index('Index').loc[te]
        manual = manual_hc(final[key],run.source.basis('final',t),tables,clinical,te)
        maxerr = max(maxerr,float(np.max(np.abs(manual-frame.Probability))))
        np.testing.assert_allclose(manual,frame.Probability,atol=1e-9,rtol=0)
    # Reused scores must be exactly the recorded historic scores, not a re-fit.
    old = run.retained.read('test_predictions').set_index('Index').loc[te]
    oldfive = run.five.read('test_predictions')
    for key in run.UNIQUE:
        frame = predictions[predictions.ModelKey == key].set_index('Index').loc[te]
        assert len(frame) == 36 and frame.index.is_unique
        np.testing.assert_array_equal(frame.Outcome,truth)
        np.testing.assert_array_equal(frame.AnalysisID,cohort.AnalysisID.iloc[te])
        scores[key] = frame.Probability.to_numpy()
        if key in run.SHARED:
            expected = oldfive[oldfive.ModelKey == key].set_index('Index').loc[te].Probability
            np.testing.assert_array_equal(frame.Probability,expected)
        elif key == 'HabitatOriginal':
            np.testing.assert_array_equal(frame.Probability,old.BaselineProbability)
        elif key == 'Habitat85':
            np.testing.assert_array_equal(frame.Probability,old.Fixed85Probability)
        row = metrics.loc[key]
        np.testing.assert_allclose(row.TestAUC,roc_auc_score(truth,scores[key]),atol=1e-12,rtol=0)
        assert row.Threshold == summary.loc[key,'Threshold']
        np.testing.assert_array_equal(frame.Prediction,scores[key]>=row.Threshold)
        tn,fp,fn,tp = confusion_matrix(truth,frame.Prediction,labels=[0,1]).ravel()
        np.testing.assert_array_equal([tn,fp,fn,tp],[row.TN,row.FP,row.FN,row.TP])
    from verify_ivim_df63_20261020 import draws
    samples = draws(truth,scores,run.SEED)
    for key in run.UNIQUE:
        np.testing.assert_allclose(np.quantile(samples[key],[.025,.975]),metrics.loc[key,['AUC_CI95Low','AUC_CI95High']].to_numpy(float),atol=1e-12)
    paired = run.read('paired_test_differences')
    for row in paired.itertuples():
        a,b = row.Comparison.split('_minus_')
        np.testing.assert_allclose(np.quantile(samples[a]-samples[b],[.025,.975]),[row.CI95Low,row.CI95High],atol=1e-12)
    comparison = []
    for scheme,t in run.SCHEMES:
        for family,key in [('Habitat',run.hkey(t)),('Whole','Whole'),('Clinical','Clinical'),('HabitatClinical',run.hkey(t,True)),('WholeClinical','WholeClinical')]:
            comparison.append({'Group':scheme,'Family':family,'UniqueModel':key,
                  'SharedBaseline':key in run.SHARED,**summary.loc[key].to_dict(),**metrics.loc[key].to_dict()})
    run.csv('two_schemes_five_models',comparison)
    unique = summary.join(metrics.drop(columns='Threshold'))
    unique.to_csv(run.OUT/'seven_unique_models.csv',encoding='utf-8-sig',index_label='ModelKey')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,2,figsize=(13.3,5.5),gridspec_kw={'width_ratios':[1.2,1]})
    order = ['HabitatOriginal','Habitat85','Whole','Clinical','HabitatClinicalOriginal','HabitatClinical85','WholeClinical']
    names = ['Habitat / original','Habitat / fixed85','Whole radiomics (shared)','Clinical (shared)',
             'Habitat+clinical / original','Habitat+clinical / fixed85','Whole+clinical (shared)']
    pos = np.arange(7)
    for col,offset,color,marker,label in [('TrainFitAUC',-.15,'#919ba4','o','Actual training fit'),
             ('OuterMeanAUC',0,'#247ba0','s','All-five outer mean'),('TestAUC',.15,'#d98a3c','^','Historical test36')]:
        values = unique.loc[order,col].to_numpy(float)
        axes[0].scatter(values,pos+offset,c=color,marker=marker,s=30,label=label)
    axes[0].set(yticks=pos,yticklabels=names,xlim=(0,1.02),xlabel='AUC',title='Seven unique models / same A84-A36')
    axes[0].invert_yaxis()
    axes[0].axvline(.75,color='gray',ls=':',lw=1)
    axes[0].legend(loc='lower left',fontsize=8)
    for key,label,color in [('HabitatOriginal','Habitat original','#929ba3'),('Habitat85','Habitat fixed85','#7fb8cc'),
              ('Clinical','Clinical','#bba969'),('HabitatClinicalOriginal','Habitat+clinical original','#215b81'),
              ('HabitatClinical85','Habitat+clinical fixed85','#af4c62')]:
        x,z,_ = roc_curve(truth,scores[key])
        axes[1].plot(x,z,color=color,label='%s %.4f'%(label,metrics.loc[key,'TestAUC']))
    axes[1].plot([0,1],[0,1],'--',color='gray',lw=1)
    axes[1].set(xlabel='1 - specificity',ylabel='Sensitivity',title='Frozen scores / historical test36')
    axes[1].legend(loc='lower right',fontsize=8)
    for ax in axes:
        ax.grid(alpha=.15)
    fig.tight_layout()
    for ext in ('png','svg'):
        fig.savefig(run.OUT/('figures/two_schemes_five_models.'+ext),dpi=180)
    plt.close(fig)
    lines = ['# 原方案与固定85%方向筛选：五模型对照','','日期：2026年10月25日。用户确认85%只用于生境及生境＋临床，其他三项共用基线。本轮列出两组各五模型，共10行，实际7个不同模型。','',
        'A组120例及同一84训练（57阳性/27阴性）、36测试（25阳性/11阴性）保持不变；同一完整外层五折，不抽新病例或删除低折。只使用1 mm，T2＋ADC＋IVIM D/f，生境k=3，不含D*。','',
        '## 主要结果','','| 方案 | 模型 | 实际训练拟合AUC | 完整外层五折均值±SD | 汇总OOF AUC | 测试AUC（95% CI） |','|---|---|---:|---:|---:|---:|']
    for r in comparison:
        lines.append('| %s | %s%s | %.4f | %.4f ± %.4f | %.4f | %.4f（%.4f～%.4f） |'%(
            '原方案' if r['Group']=='Original' else '固定85%',LABELS[r['UniqueModel']],'〔共用〕' if r['SharedBaseline'] else '',
            r['TrainFitAUC'],r['OuterMeanAUC'],r['OuterSD'],r['PooledOOFAUC'],r['TestAUC'],r['AUC_CI95Low'],r['AUC_CI95High']))
    hc0,hc85,h0,h85,c = [unique.loc[k] for k in ('HabitatClinicalOriginal','HabitatClinical85','HabitatOriginal','Habitat85','Clinical')]
    lines += ['', '## 如何判断','','纯生境结果与刚才冻结评估完全一致：原方案训练/外层/测试为%.4f/%.4f/%.4f，85%%为%.4f/%.4f/%.4f。本次没有因加入对照而重新训练这两个纯生境模型。'%(h0.TrainFitAUC,h0.OuterMeanAUC,h0.TestAUC,h85.TrainFitAUC,h85.OuterMeanAUC,h85.TestAUC),'',
        '新增原方案生境＋临床为%.4f/%.4f/%.4f，85%%生境＋临床为%.4f/%.4f/%.4f（依次实际训练拟合/全部五折均值/测试）。85%%对联合模型的变化为训练%+.4f、外层%+.4f、测试%+.4f，应三者一起判断。'%(hc0.TrainFitAUC,hc0.OuterMeanAUC,hc0.TestAUC,hc85.TrainFitAUC,hc85.OuterMeanAUC,hc85.TestAUC,hc85.TrainFitAUC-hc0.TrainFitAUC,hc85.OuterMeanAUC-hc0.OuterMeanAUC,hc85.TestAUC-hc0.TestAUC),'',
        '联合模型包含11项术前临床信息，其表现不能归为纯生境的提升。临床单独训练/外层/测试为%.4f/%.4f/%.4f，务必同时对照，避免将临床信息的贡献误认为生境贡献。'%(c.TrainFitAUC,c.OuterMeanAUC,c.TestAUC),'',
        '| 条件：训练拟合及外层均值都≥0.75 | 是否满足 |','|---|---|']
    for key in ('HabitatOriginal','Habitat85','HabitatClinicalOriginal','HabitatClinical85'):
        r = unique.loc[key]
        lines.append('| %s %s | %s |'%('固定85%' if '85' in key else '原方案',LABELS[key],'是' if r.TrainFitAUC>=.75 and r.OuterMeanAUC>=.75 else '否'))
    lines += ['', '## 冻结阈值的分类表现','','| 方案 | 模型 | 阈值 | 敏感度 | 特异度 | 准确率 | TP/FN/TN/FP |','|---|---|---:|---:|---:|---:|---|']
    for r in comparison:
        lines.append('| %s | %s | %.6f | %.2f%% | %.2f%% | %.2f%% | %d/%d/%d/%d |'%(
            '原方案' if r['Group']=='Original' else '固定85%',LABELS[r['UniqueModel']],r['Threshold'],
            r['Sensitivity']*100,r['Specificity']*100,r['Accuracy']*100,r['TP'],r['FN'],r['TN'],r['FP']))
    lines += ['', '## 外层五折明细','','| 模型 | 第1折 | 第2折 | 第3折 | 第4折 | 第5折 |','|---|---:|---:|---:|---:|---:|']
    for key in order:
        z = outer[outer.ModelKey==key].sort_values('Fold')
        label = ('固定85% ' if '85' in key else '原方案 ' if 'Original' in key else '共用 ')+LABELS[key]
        lines.append('| %s | %s |'%(label,' | '.join('%.4f'%v for v in z.AUC)))
    lines += ['', '## 预先列出的测试AUC配对差值','','| 比较（前者减后者） | 差值 | 配对95%区间 |','|---|---:|---|']
    for r in paired.itertuples():
        lines.append('| %s | %+.4f | [%+.4f，%+.4f] |'%(r.Comparison,r.AUC_Difference,r.CI95Low,r.CI95High))
    lines += ['', '## 本轮新增与复用的边界','',
        '两种纯生境、全肿瘤影像组学、临床、全肿瘤＋临床这5个不同模型使用各自已冻结的原权重、折外分数、测试分数及分类阈值。三项共同基线在两组的重复行完全相同，不是额外独立验证。全肿瘤为原轮廓372项非形状影像组学（T2、ADC、D、f各93项），不是三轮廓中位数生境特征。','',
        '只新增两种生境＋临床模型：各自使用对应原/85%生境基础模型的12路Bag15输出，拼接11项术前临床变量，形成23项融合输入。沿用生境融合层C=0.03、锚定0.25、平衡L2逻辑回归，系数允许正负。OOF分数和实际训练推断分数按0.8/0.2加权用于训练，同一临床训练输入重复进入两块；该过程不构成额外验证。没有新的超参数搜索。','',
        '每个外层训练部分独立拟合融合器；内层5折×3次仅作为固定配方诊断，每个内层训练部分使用其自身已核验的私有基础OOF。临床缺失填补、截尾和稳健缩放只在当前训练部分拟合。没有用外层验证或测试结局拟合预处理、特征或权重。','',
        '临床变量：术前T/N、EMVI、MRF、logCEA、年龄、性别、MRI肿瘤位置/厚度/长度/环周比例。11项全部保留，不进行85%方向筛选。旧来源审计的治疗时间unknown为历史记录；按用户2026年10月12日确认，解释为治疗前MRI、统一病理标准。','',
        '两种新联合模型的分类阈值分别由完整84例外层OOF的Youden指数确定，在36例评分前冻结；其他五个模型阈值保持原样。只新增两种联合模型的一次共同测试阶段，共72个新预测；复用5个模型共180个历史预测。模型推断、独立系数公式、逐例标签、阈值和分类计数均已核对。','',
        '全部7个模型的95%区间在此按同一5000次病例分层bootstrap重新计算，seed=20261025211，所以复用模型的区间端点可能与旧报告略有不同，点估计与逐例评分不变。区间未包含重训和多轮模型选择的不确定性。','',
        '这批患者、外层划分及36例测试已多次用于探索；固定85%也是关注此前外层结果后提出的候选。本轮结果属于有选择背景的内部探索，不能称新的独立或外部验证。所有折保留，实际训练拟合与融合层OOF关联表现严格区分，没有测试后调阈值、翻转分数或重训。原模型及正式采用状态不变。','',
        '![七个不同模型与测试ROC]('+(run.OUT/'figures/two_schemes_five_models.png').as_posix()+')','',
        '主要文件：two_schemes_five_models.csv（10行）、seven_unique_models.csv（7行）、outer_metrics.csv、test_metrics.csv、test_predictions.csv（252行）、paired_test_differences.csv、new_hc_coefficients.csv、training_verification.json。']
    (run.OUT/'原方案与85%方向筛选五模型对照报告_20261025.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    run.save(run.OUT/'report_verification.json',{'passed':True,'rows':10,'unique_models':7,'individual_test_scores':252,
         'new_hc_formula_error':maxerr,'five_reused_models_scores_exact':True,'bootstrap5000_recomputed':True,
         'all_thresholds_unchanged':True,'shared_baselines_identical':True,'formal_adoption_changed':False})
    print(pd.DataFrame(comparison)[['Group','Family','TrainFitAUC','OuterMeanAUC','OuterSD','TestAUC','Sensitivity','Specificity','Accuracy']].to_string(index=False),flush=True)
    print('All7unique results verified;10row report generated',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=('verify','report'))
    args = parser.parse_args()
    with threadpool_limits(1):
        globals()[args.action]()
