"""Three normalization protocols with controlled comparators and nested method choice."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
import SimpleITK as sitk
import radiomics
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
import lvi_paths
import normalization_habitat_features_20261012 as feature
import run_multiseq_habitat_120_20261012 as old

ROOT=old.ROOT
OUT=ROOT/'lvi_prediction/results/lvi_normalization_habitat_120_20261012'
NORMALIZATIONS=feature.NORMALIZATIONS
GROUPS={'T2':('T2',),'ADC':('ADC',),'IVIM':('IVIM',),'FusionFree':('T2','ADC','IVIM')}
NORM_LABELS={'ZScore':'普通Z-score','RobustZ':'稳健Z-score','PercentileZ':'P1/P99截尾＋Z-score','Adaptive':'内层选择标准化'}
GROUP_LABELS={k:old.LABELS[k] for k in GROUPS}
save,digest=old.save,old.digest


def inputs():return old.common.old.previous.load_inputs()


def options():
    result=[]
    for normalization in NORMALIZATIONS:
        for analysis in GROUPS:
            for family in ['Descriptor','Texture','Combined']:
                for top in [3,6]:
                    for C in [.03,.3,1.]:
                        result.append({'normalization':normalization,'analysis':analysis,'family':family,'top':top,'quota':0,'algorithm':'LR','C':C})
    return [{**r,'candidate_id':i} for i,r in enumerate(result)]


def config(row):return {k:row[k] for k in options()[0]}


def choose(grid,analysis,normalization=None):
    pool=[r for r in grid if r['analysis']==analysis and (normalization is None or r['normalization']==normalization)]
    valid=[r for r in pool if r['AvailableAllFolds']]
    return max(valid if valid else pool,key=lambda r:(r['Objective'],-r['MeanFeatureCount'],-r['C'],-r['candidate_id']))


def key(normalization,analysis):return normalization+'__'+analysis


def all_keys():return [key(n,a) for n in list(NORMALIZATIONS)+['Adaptive'] for a in GROUPS]


def check_sources():
    p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    for n,h in p['protected_hashes'].items():assert digest(ROOT/n)==h,n
    for n,h in p['source_hashes'].items():assert digest(ROOT/n)==h,n
    return len(p['protected_hashes'])


def freeze():
    if OUT.exists():raise FileExistsError('Normalization experiment already frozen')
    old.check_sources();assert json.loads((old.OUT/'test_verification.json').read_text())['passed']
    decision=json.loads((ROOT/'lvi_prediction/DSTAR_DECISION_20261012.json').read_text(encoding='utf-8'))
    assert decision['include_Dstar_in_future_models'] is False
    cohort,manifest,y,tr,te=inputs()
    protected=dict(json.loads((old.OUT/'protocol.json').read_text(encoding='utf-8'))['protected_hashes'])
    for directory in [old.OUT,ROOT/'lvi_prediction/results/lvi_ivim_dstar_120_20261012']:
        protected.update({str(p.relative_to(ROOT)):digest(p) for p in directory.rglob('*') if p.is_file()})
    protected['lvi_prediction/DSTAR_DECISION_20261012.json']=digest(ROOT/'lvi_prediction/DSTAR_DECISION_20261012.json')
    protected['lvi_prediction/COHORT_CONFIRMATION_20261012.json']=digest(ROOT/'lvi_prediction/COHORT_CONFIRMATION_20261012.json')
    paths=[Path(__file__).resolve(),Path(feature.__file__).resolve(),ROOT/'lvi_prediction/test_normalization_habitat_120_20261012.py',
        ROOT/'lvi_prediction/localtexture_habitat_features_20261012.py',ROOT/'lvi_prediction/regional_habitat_features_20261012.py',
        ROOT/'lvi_prediction/multiseq_habitat_features_20261012.py',ROOT/'lvi_prediction/run_multiseq_habitat_120_20261012.py',
        old.common.old.retained.PROFILE,old.common.old.retained.POOL/'cohort_138.csv']
    paths.extend(p for aid in cohort.AnalysisID.iloc[118:] for p in [old.common.old.retained.POOL/'prepared'/aid/'profile.joblib'])
    OUT.mkdir(parents=True);(OUT/'models').mkdir()
    save(OUT/'protocol.json',{'date':'2026-10-12','authorization':'按推荐比较普通Z-score、稳健Z-score、P1/P99截尾＋Z-score；不加D*',
        'same120_84_36_split':True,'normalizations':NORMALIZATIONS,'groups':GROUPS,'feature_protocol':feature.PROTOCOL,
        'MRI_before_neoadjuvant_and_pathology_uniform_user_confirmed':True,
        'Dstar_excluded':True,'IVIM_validity':'original D/f/NRMSE profiles; no Dstar-specific mask',
        'Nyul_reference_tissue_and_other_changes_included':False,
        'candidates':options(),'candidate_count':216,'per_normalization_group_candidates':18,
        'controlled_models':'12 fixed normalization/group pipelines; same3/6 features,LR C.03/.3/1,feature families',
        'adaptive_models':'4 extra pipelines choose normalization+features+parameters jointly INSIDE each inner5; outer fold validates complete selection',
        'nested':'outer5 seed42,inner5 seed62201..62205; final selection5 seed62212',
        'robustness':'unchanged current training ICC>=.8,complete>=.9,5 stratified bootstrap sign consistency>=.8',
        'ranking':'abs mean signed effect*ICC^2*sign consistency; Spearman<.8',
        'feature_preprocessing':'unchanged training median fill,training1/99% winsorization,trainingmean/SD',
        'objective':'unchanged mean inner AUC-.10*SD-.30*max(mean fit-mean validation-.10,0)-.03*abs(mean fit-.75); kept to isolate normalization',
        'baseline':'verified cached original1242 features; canonical original partitions; baseline models must replay original four category results',
        'test':'freeze12 fixed finalists and4 adaptive references; one joint stage,12 distinct model scores,adaptive references reuse chosen scores',
        'test_previously_reused':True,'formal_model_use':False,'automatic_adoption':False,'prior_nonadoption_decisions_unchanged':True,
        'protected_hashes':protected,'source_hashes':{str(p.relative_to(ROOT)):digest(p) for p in paths},
        'sources':{'normalization':'https://pubmed.ncbi.nlm.nih.gov/37413967/'}})
    save(OUT/'split_manifest.json',manifest)
    return cohort,manifest,y,tr,te


def extract(indices,cohort,stage):
    start=time.time();profiles=old.common.old.retained.load_profiles()
    baseline=joblib.load(old.OUT/('training_multiseq_features.joblib' if stage=='training' else 'test_multiseq_features.joblib'))
    np.testing.assert_array_equal(baseline['indices'],indices)
    metadata=baseline['metadata'];names=[m['name'] for m in metadata]
    tasks=[(int(i),cohort.AnalysisID.iloc[i],profiles[int(i)]) for i in indices];results={}
    radiomics.logger.setLevel(40);sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(feature.extract_one,t):t[0] for t in tasks}
        for done,f in enumerate(as_completed(futures),1):
            i=futures[f];results[i]=f.result()
            assert set(m['name'] for m in results[i]['metadata'])==set(names)
            if done%4==0 or done==len(indices):print(f'Normalization {stage} extraction {done}/{len(indices)}, {time.time()-start:.0f}s',flush=True)
    data={'ZScore':{k:baseline[k] for k in ['base','erode','dilate','indices','metadata']}}
    for method in NORMALIZATIONS[1:]:
        data[method]={'indices':np.asarray(indices,int),'metadata':metadata,
                      **{kind:np.asarray([[results[int(i)]['values'][method][kind][n] for n in names] for i in indices],float)
                         for kind in ['base','erode','dilate']}}
    for name in ['normalization_audit','partition_audit','texture_audit']:
        pd.DataFrame([r for i in indices for r in results[int(i)][name]]).to_csv(OUT/(stage+'_'+name+'.csv'),index=False,encoding='utf-8-sig')
    changes=[]
    for method in NORMALIZATIONS[1:]:
        for mod in feature.MODALITIES:
            for block in ['Descriptor','Texture']:
                ix=[j for j,m in enumerate(metadata) if m['modality']==mod and m['block']==block]
                a,b=data['ZScore']['base'][:,ix],data[method]['base'][:,ix];valid=np.isfinite(a)&np.isfinite(b)
                changed=valid&~np.isclose(a,b,rtol=1e-6,atol=1e-8)
                changes.append({'Normalization':method,'Modality':mod,'Block':block,'FeatureCount':len(ix),
                    'ChangedFiniteCellsFraction':float(changed.sum()/max(valid.sum(),1)),
                    'FiniteAvailabilityDifferenceCells':int((np.isfinite(a)!=np.isfinite(b)).sum())})
                if method=='RobustZ' and mod in ['ADC','IVIM'] and block=='Descriptor':
                    np.testing.assert_allclose(a,b,atol=1e-12,rtol=0,equal_nan=True)
    pd.DataFrame(changes).to_csv(OUT/(stage+'_feature_change_audit.csv'),index=False,encoding='utf-8-sig')
    joblib.dump(data,OUT/(stage+'_features.joblib'),compress=3)
    return data


def fit(data,a,b,y,c):return old.fit(data[c['normalization']],a,b,y,c)


def tune(data,folds,y):
    values={c['candidate_id']:[] for c in options()};cv={c['candidate_id']:[] for c in options()}
    for a,b in folds:
        rank={n:old.ranking(data[n],a,y) for n in NORMALIZATIONS};cache={}
        for c in options():
            k=(c['normalization'],c['analysis'],c['family'],c['top'])
            if k not in cache:cache[k]=old.prepare(data[c['normalization']],a,b,c,y,rank[c['normalization']])
            f=old.fitted(cache[k],y[a],c)
            values[c['candidate_id']].append((roc_auc_score(y[b],f['evaluation_probability']),
                roc_auc_score(y[a],f['train_probability']),len(f['indices']),f['available']))
            cv[c['candidate_id']].append((b,f['evaluation_probability']))
    records=[]
    for c in options():
        v=np.asarray(values[c['candidate_id']]);mean,sd,train,n=float(v[:,0].mean()),float(v[:,0].std(ddof=1)),float(v[:,1].mean()),float(v[:,2].mean())
        valid=bool(v[:,3].all());obj=mean-.10*sd-.30*max(train-mean-.10,0)-.03*abs(train-.75)
        records.append({**c,'MeanValidationAUC':mean,'SD':sd,'MeanFitAUC':train,'MeanFeatureCount':n,
                        'AvailableAllFolds':valid,'Objective':obj if valid else -1.})
    return records,cv


def selected_grid(grid):
    return {key(n,a):choose(grid,a,None if n=='Adaptive' else n) for n in list(NORMALIZATIONS)+['Adaptive'] for a in GROUPS}


def baseline_equal(f,original):
    assert f['names']==original['names'] and f['indices']==original['indices']
    for n in ['median','low','high','mean','scale','train_probability','evaluation_probability']:np.testing.assert_allclose(f[n],original[n],atol=1e-12,rtol=0)
    np.testing.assert_allclose(f['model'].coef_,original['model'].coef_,atol=1e-12,rtol=0)
    np.testing.assert_allclose(f['model'].intercept_,original['model'].intercept_,atol=1e-12,rtol=0)


def train():
    start=time.time();cohort,manifest,y,tr,te=freeze();data=extract(tr,cohort,'training')
    pred=[];metrics=[];grids=[]
    for fold,(a,b) in enumerate(old.common.old.previous.folds(tr,y,42,5),1):
        grid,_=tune(data,old.common.old.previous.folds(a,y,62200+fold,5),y);chosen=selected_grid(grid)
        stored={'train':a,'evaluation':b,'grid':grid,'models':{}};cache={}
        prior=joblib.load(old.OUT/f'models/outer_fold_{fold}.joblib')['models']
        for model_key,row in chosen.items():
            c=config(row)
            if c['candidate_id'] not in cache:cache[c['candidate_id']]=fit(data,a,b,y,c)
            f=cache[c['candidate_id']];stored['models'][model_key]=f
            if model_key.startswith('ZScore__'):baseline_equal(f,prior[c['analysis']])
            metrics.append({'ModelKey':model_key,'Fold':fold,**c,'FeatureCount':len(f['indices']),
                            'AUC':float(roc_auc_score(y[b],f['evaluation_probability']))})
            pred.extend({'ModelKey':model_key,'Fold':fold,'AnalysisID':cohort.AnalysisID.iloc[i],'Outcome':int(y[i]),'Probability':float(p)} for i,p in zip(b,f['evaluation_probability']))
        grids.extend({'OuterFold':fold,**r} for r in grid);joblib.dump(stored,OUT/f'models/outer_fold_{fold}.joblib',compress=3)
        print(f'Normalization nested outer fold {fold}/5 complete',flush=True)
    predictions=pd.DataFrame(pred);folds=pd.DataFrame(metrics)
    grid,cv=tune(data,old.common.old.previous.folds(tr,y,62212,5),y);chosen=selected_grid(grid)
    final={'models':{},'train':tr,'metadata':data['ZScore']['metadata'],'formal_model_use':False};cache={};choices=[];features=[];trainpred=[];finalcv=[];gates=[]
    prior=joblib.load(old.OUT/'models/five_multiseq_models.joblib')['models']
    for model_key,row in chosen.items():
        c=config(row)
        if c['candidate_id'] not in cache:
            f=fit(data,tr,[],y,c);oof=np.full(138,np.nan)
            for ix,p in cv[c['candidate_id']]:oof[ix]=p
            assert np.isfinite(oof[tr]).all() and np.isnan(oof[te]).all()
            f['threshold']=old.common.threshold(y[tr],oof[tr]);f['train']=tr;cache[c['candidate_id']]=(f,oof)
        f,oof=cache[c['candidate_id']];final['models'][model_key]=f
        if model_key.startswith('ZScore__'):baseline_equal(f,prior[c['analysis']]);assert f['threshold']==prior[c['analysis']]['threshold']
        pp=predictions[predictions.ModelKey==model_key].set_index('AnalysisID').loc[cohort.AnalysisID.iloc[tr]];ff=folds[folds.ModelKey==model_key].AUC
        choices.append({'ModelKey':model_key,'NormalizationChoice':'Adaptive' if model_key.startswith('Adaptive__') else c['normalization'],**c,
            'FeatureCount':len(f['indices']),'TrainFitAUC':float(roc_auc_score(y[tr],f['train_probability'])),
            'FinalSelectionCVMeanAUC':row['MeanValidationAUC'],'OuterMeanAUC':float(ff.mean()),'OuterSD':float(ff.std(ddof=1)),
            'PooledOOFAUC':float(roc_auc_score(y[tr],pp.Probability)),'Threshold':f['threshold']})
        coef=f['model'].coef_[0] if f['model'] is not None else []
        features.extend({'ModelKey':model_key,'Normalization':c['normalization'],'Analysis':c['analysis'],'Feature':n,
                         'Modality':data['ZScore']['metadata'][j]['modality'],'Block':data['ZScore']['metadata'][j]['block'],
                         'LRStandardizedCoefficient':float(beta)} for n,j,beta in zip(f['names'],f['indices'],coef))
        trainpred.extend({'ModelKey':model_key,'AnalysisID':cohort.AnalysisID.iloc[i],'Outcome':int(y[i]),'Probability':float(p)} for i,p in zip(tr,f['train_probability']))
        finalcv.extend({'ModelKey':model_key,'AnalysisID':cohort.AnalysisID.iloc[i],'Outcome':int(y[i]),'Probability':float(oof[i])} for i in tr)
    for n in NORMALIZATIONS:
        _,_,audit=old.ranking(data[n],tr,y)
        gates.extend({'Normalization':n,'Modality':m['modality'],'Block':m['block'],**r} for r,m in zip(audit,data[n]['metadata']))
    joblib.dump(final,OUT/'models/normalization_models.joblib',compress=3)
    for name,obj in [('nested_predictions.csv',predictions),('outer_fold_metrics.csv',folds),('inner_candidate_scores.csv',pd.DataFrame(grids)),
        ('final_training_choices.csv',pd.DataFrame(choices)),('final_training_grid.csv',pd.DataFrame(grid)),('selected_features.csv',pd.DataFrame(features)),
        ('train_fit_predictions.csv',pd.DataFrame(trainpred)),('final_selection_oof_predictions.csv',pd.DataFrame(finalcv)),('feature_gate.csv',pd.DataFrame(gates))]:
        obj.to_csv(OUT/name,index=False,encoding='utf-8-sig')
    adaptive=[r for r in choices if r['ModelKey'].startswith('Adaptive__')]
    save(OUT/'training_only_recommendation.json',{'per_group':adaptive,'criterion':'joint inner5 selection of normalization/features/parameters; outer5 assessment',
        'test_used':False,'formal_model_use':False})
    save(OUT/'training_frozen.json',{'passed':True,'model_sha256':digest(OUT/'models/normalization_models.joblib'),
        'feature_sha256':digest(OUT/'training_features.joblib'),'protocol_sha256':digest(OUT/'protocol.json'),
        'training_csv_hashes':{p.name:digest(p) for p in OUT.glob('*.csv')},'test_scored':False,'protected_count':check_sources(),
        'elapsed_seconds':time.time()-start,'fixed_models':12,'adaptive_references':4})
    print(pd.DataFrame(choices)[['ModelKey','normalization','family','C','FeatureCount','TrainFitAUC','OuterMeanAUC','OuterSD','PooledOOFAUC']].to_string(index=False),flush=True)
    print('Normalization training frozen; test not scored',flush=True)


def frozen_training():
    check_sources();f=json.loads((OUT/'training_frozen.json').read_text())
    for p,k in [('models/normalization_models.joblib','model_sha256'),('training_features.joblib','feature_sha256'),('protocol.json','protocol_sha256')]:assert digest(OUT/p)==f[k]
    for n,h in f['training_csv_hashes'].items():assert digest(OUT/n)==h,n
    return f


def compare(a,b):
    assert a['config']==b['config'] and a['names']==b['names'] and a['indices']==b['indices'] and a['available']==b['available']
    for n in ['median','low','high','mean','scale','train_probability','evaluation_probability']:np.testing.assert_allclose(a[n],b[n],atol=1e-12,rtol=0)


def verify_training():
    frozen_training();cohort,manifest,y,tr,te=inputs();data=joblib.load(OUT/'training_features.joblib')
    assert not set(tr)&set(te)
    for n in NORMALIZATIONS:
        np.testing.assert_array_equal(data[n]['indices'],tr);assert len(data[n]['metadata'])==1242
        assert not any('Dstar' in m['name'] or 'Dfast' in m['name'] for m in data[n]['metadata'])
    original=joblib.load(old.OUT/'training_multiseq_features.joblib')
    for kind in ['base','erode','dilate']:np.testing.assert_array_equal(data['ZScore'][kind],original[kind])
    pa=pd.read_csv(OUT/'training_partition_audit.csv');assert (pa[pa.Normalization=='RobustZ'].ARIvsZScore==1.).all()
    predictions=pd.read_csv(OUT/'nested_predictions.csv',float_precision='round_trip');folds=pd.read_csv(OUT/'outer_fold_metrics.csv',float_precision='round_trip')
    assert len(predictions)==16*84 and len(folds)==80 and not predictions.duplicated(['ModelKey','AnalysisID']).any()
    refits=0;objectives=0;frequency=[]
    for fold,(a,b) in enumerate(old.common.old.previous.folds(tr,y,42,5),1):
        grid,_=tune(data,old.common.old.previous.folds(a,y,62200+fold,5),y);chosen=selected_grid(grid);stored=joblib.load(OUT/f'models/outer_fold_{fold}.joblib')
        np.testing.assert_allclose([r['Objective'] for r in grid],[r['Objective'] for r in stored['grid']],atol=1e-12,rtol=0)
        changed=y.copy();changed[b]=1-y[b]
        for k,f in stored['models'].items():
            assert config(chosen[k])==f['config'];compare(fit(data,a,b,changed,f['config']),f)
            if k.startswith('Adaptive__'):assert f['config']==stored['models'][key(f['config']['normalization'],f['config']['analysis'])]['config']
            rows=predictions[(predictions.ModelKey==k)&(predictions.Fold==fold)].set_index('AnalysisID').loc[cohort.AnalysisID.iloc[b]]
            np.testing.assert_array_equal(rows.Outcome,y[b]);np.testing.assert_allclose(rows.Probability,f['evaluation_probability'],atol=1e-12,rtol=0)
            assert abs(float(folds[(folds.ModelKey==k)&(folds.Fold==fold)].iloc[0].AUC)-roc_auc_score(y[b],rows.Probability))<1e-12
            frequency.extend({'ModelKey':k,'Fold':fold,'Normalization':f['config']['normalization'],'Feature':n} for n in f['names']);refits+=1
        objectives+=len(grid);print(f'Normalization training verification fold {fold}/5',flush=True)
    grid,cv=tune(data,old.common.old.previous.folds(tr,y,62212,5),y);chosen=selected_grid(grid);final=joblib.load(OUT/'models/normalization_models.joblib')
    choices=pd.read_csv(OUT/'final_training_choices.csv',float_precision='round_trip').set_index('ModelKey');changed=y.copy();changed[te]=1-cohort.Outcome.to_numpy(int)[te]
    for k,f in final['models'].items():
        assert config(chosen[k])==f['config'];compare(fit(data,tr,[],changed,f['config']),f)
        oof=np.full(138,np.nan)
        for ix,p in cv[f['config']['candidate_id']]:oof[ix]=p
        assert np.isnan(oof[te]).all() and old.common.threshold(y[tr],oof[tr])==f['threshold']
        selected=pd.DataFrame(f['audit']).set_index('Feature').loc[f['names']]
        assert selected.Eligible.all() and (selected.ICC>=.8).all() and (selected.CompleteFraction>=.9).all() and (selected.BootstrapSignConsistency>=.8).all()
        assert abs(choices.loc[k,'TrainFitAUC']-roc_auc_score(y[tr],f['train_probability']))<1e-12
        assert all(final['metadata'][j]['modality'] in GROUPS[f['config']['analysis']] for j in f['indices'])
        refits+=1
    objectives+=len(grid);pd.DataFrame(frequency).to_csv(OUT/'outer_feature_selections.csv',index=False,encoding='utf-8-sig')
    save(OUT/'training_verification.json',{'passed':True,'refits':refits,'candidate_objectives_recomputed':objectives,
        'adaptive_normalization_chosen_inside_inner_folds':True,'outer_and_test_labels_do_not_affect_fit':True,
        'original_baseline_replayed_exactly':True,'RobustZ_partitions_identical':True,'Dstar_excluded':True,
        'feature_gates_thresholds_and_same_split_verified':True,'protected_count':check_sources()})
    print('Normalization training verification passed',flush=True)


def evaluate():
    if (OUT/'test_evaluation_started.json').exists():raise FileExistsError('Do not repeat joint test stage')
    frozen=frozen_training();assert json.loads((OUT/'training_verification.json').read_text())['passed']
    cohort,manifest,_,tr,te=inputs();data=extract(te,cohort,'test');final=joblib.load(OUT/'models/normalization_models.joblib')
    save(OUT/'test_evaluation_started.json',{'joint_count':1,'distinct_model_count':12,'adaptive_references':4,'test_ids':manifest['test_ids'],'model_sha256':frozen['model_sha256']})
    scores={k:old.predict(f,data[f['config']['normalization']]) for k,f in final['models'].items() if not k.startswith('Adaptive__')}
    for k,f in final['models'].items():
        if k.startswith('Adaptive__'):scores[k]=scores[key(f['config']['normalization'],f['config']['analysis'])]
    original=pd.read_csv(old.OUT/'test_predictions.csv',float_precision='round_trip')
    for a in GROUPS:
        rows=original[original.Analysis==a].set_index('AnalysisID').loc[cohort.AnalysisID.iloc[te]]
        np.testing.assert_allclose(scores[key('ZScore',a)],rows.Probability,atol=1e-12,rtol=0)
    y=cohort.Outcome.to_numpy(int)[te];draws=old.common.old.retained.bootstrap(y,scores);pred=[];metrics=[];pairs=[]
    for k,p in scores.items():
        f=final['models'][k];threshold=f['threshold'];low,high=np.quantile(draws[k],[.025,.975])
        metrics.append({'ModelKey':k,'Normalization':f['config']['normalization'],'Analysis':f['config']['analysis'],
            **old.common.old.metrics(y,p,threshold),'AUC_CI95Low':float(low),'AUC_CI95High':float(high)})
        pred.extend({'ModelKey':k,'AnalysisID':cohort.AnalysisID.iloc[i],'Outcome':int(yy),'Probability':float(pp),'Threshold':threshold,'PredictedPositive':int(pp>=threshold)} for i,yy,pp in zip(te,y,p))
    for a in GROUPS:
        for n in ['RobustZ','PercentileZ','Adaptive']:
            ka,kb=key(n,a),key('ZScore',a);low,high=np.quantile(draws[ka]-draws[kb],[.025,.975])
            pairs.append({'Analysis':a,'Method':n,'Comparison':ka+' minus '+kb,'DeltaAUC':float(roc_auc_score(y,scores[ka])-roc_auc_score(y,scores[kb])),
                          'CI95Low':float(low),'CI95High':float(high)})
    for name,obj in [('test_predictions.csv',pd.DataFrame(pred)),('test_metrics.csv',pd.DataFrame(metrics)),('paired_auc_comparisons.csv',pd.DataFrame(pairs))]:
        obj.to_csv(OUT/name,index=False,encoding='utf-8-sig')
    save(OUT/'evaluation_complete.json',{'passed':True,'joint_count':1,'distinct_model_predictions':432,'adaptive_reference_predictions':144,
        'test_n':36,'formal_model_use':False,'protected_count':check_sources()})
    print(pd.DataFrame(metrics).to_string(index=False),flush=True)


def verify_test():
    frozen_training();cohort,manifest,_,tr,te=inputs();data=joblib.load(OUT/'test_features.joblib');final=joblib.load(OUT/'models/normalization_models.joblib')
    pred=pd.read_csv(OUT/'test_predictions.csv',float_precision='round_trip');metrics=pd.read_csv(OUT/'test_metrics.csv',float_precision='round_trip').set_index('ModelKey')
    assert len(pred)==16*36 and not pred.duplicated(['ModelKey','AnalysisID']).any();y=cohort.Outcome.to_numpy(int)[te];manual={};maximum=0.
    for k,f in final['models'].items():
        np.testing.assert_array_equal(data[f['config']['normalization']]['indices'],te)
        x=data[f['config']['normalization']]['base'][:,f['indices']]
        x=(np.clip(np.where(np.isfinite(x),x,f['median']),f['low'],f['high'])-f['mean'])/f['scale']
        p=expit(x@f['model'].coef_[0]+f['model'].intercept_[0]) if f['model'] is not None else np.full(len(te),f['prevalence'])
        rows=pred[pred.ModelKey==k].set_index('AnalysisID').loc[cohort.AnalysisID.iloc[te]];np.testing.assert_array_equal(rows.Outcome,y)
        error=float(np.max(np.abs(p-rows.Probability.to_numpy())));assert error<1e-12;maximum=max(maximum,error)
        difference=np.subtract.outer(p[y==1],p[y==0]);matrix=np.where(difference>0,1.,np.where(difference==0,.5,0.));assert abs(float(matrix.mean())-metrics.loc[k,'AUC'])<1e-12
        classified=p>=f['threshold']
        for n,v in {'TP':((y==1)&classified).sum(),'TN':((y==0)&~classified).sum(),'FP':((y==0)&classified).sum(),'FN':((y==1)&~classified).sum()}.items():assert metrics.loc[k,n]==v
        manual[k]=p
    for a in GROUPS:
        f=final['models'][key('Adaptive',a)];np.testing.assert_array_equal(manual[key('Adaptive',a)],manual[key(f['config']['normalization'],a)])
    rng=np.random.default_rng(old.common.old.retained.BOOTSTRAP_SEED);np_,nn=int(y.sum()),int((y==0).sum())
    wp=rng.multinomial(np_,np.full(np_,1/np_),5000)/np_;wn=rng.multinomial(nn,np.full(nn,1/nn),5000)/nn;draws={}
    for k,p in manual.items():
        d=np.subtract.outer(p[y==1],p[y==0]);matrix=np.where(d>0,1.,np.where(d==0,.5,0.));draws[k]=np.sum((wp@matrix)*wn,axis=1)
        np.testing.assert_allclose(np.quantile(draws[k],[.025,.975]),metrics.loc[k,['AUC_CI95Low','AUC_CI95High']].to_numpy(float),atol=1e-12,rtol=0)
    pairs=pd.read_csv(OUT/'paired_auc_comparisons.csv',float_precision='round_trip')
    for _,r in pairs.iterrows():
        a,b=r.Comparison.split(' minus ');np.testing.assert_allclose(np.quantile(draws[a]-draws[b],[.025,.975]),r[['CI95Low','CI95High']].to_numpy(float),atol=1e-12,rtol=0)
    save(OUT/'test_verification.json',{'passed':True,'sigmoid_max_error':maximum,'prediction_rows':576,'distinct_model_prediction_rows':432,
        'adaptive_references_exact':True,'AUC_pair_ranking_and_confusion_counts_verified':True,'bootstrap5000_and_paired_CIs_verified':True,
        'baseline_predictions_identical_to_previous':True,'joint_test_count':1,'protected_count':check_sources(),'formal_model_use':False})
    print('Normalization independent test verification passed',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',required=True,choices=['train','verify-training','evaluate','verify-test']);args=parser.parse_args()
    with threadpool_limits(limits=1):{'train':train,'verify-training':verify_training,'evaluate':evaluate,'verify-test':verify_test}[args.phase]()
