"""Fold-local T2 Nyul extension, followed by frozen54 one-stage inference."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import json
import time
import joblib
import numpy as np
import pandas as pd
import radiomics
import SimpleITK as sitk
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
import lvi_paths
import nyul_t2_features_20261013 as feature
import run_normalization_habitat_120_20261012 as origin
import evaluate_frozen54_normalization_20261013 as previous54

ROOT=origin.ROOT
OUT=ROOT/'lvi_prediction/results/lvi_nyul_t2_120_20261013'
old=origin.old
save,digest=origin.save,origin.digest
ANALYSES=('T2','FusionFree')
KINDS=('base','erode','dilate')
KEYS=[n+'__'+a for n in ['ZScore','NyulT2'] for a in ANALYSES]

def options():
    return [{**c,'candidate_id':i} for i,c in enumerate(c for c in origin.options()
        if c['normalization']=='ZScore' and c['analysis'] in ANALYSES)]

def config(row):
    result={k:row[k] for k in options()[0]};result['normalization']='NyulT2';return result

def protected():
    protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    for n,h in protocol['protected_hashes'].items():assert digest(ROOT/n)==h,n
    for n,h in protocol['source_hashes'].items():assert digest(ROOT/n)==h,n
    return len(protocol['protected_hashes'])

def freeze_protocol():
    if OUT.exists():raise FileExistsError('Do not overwrite existing Nyul experiment')
    previous54.protect();origin.frozen_training()
    assert json.loads((previous54.OUT/'verification.json').read_text(encoding='utf-8'))['passed']
    cohort,manifest,y,tr,te=origin.inputs()
    hashes=dict(json.loads((previous54.OUT/'protocol.json').read_text(encoding='utf-8'))['protected_hashes'])
    for directory in [previous54.OUT,ROOT/'lvi_prediction/results/lvi_frozen54_robustz_detail_20261013']:
        hashes.update({str(p.relative_to(ROOT)):digest(p) for p in directory.rglob('*') if p.is_file()})
    for p in [ROOT/'lvi_prediction/CURRENT_LVI_EXPLORATORY_FREEZE_20261013.json',
              ROOT/'CURRENT_LVI_EXPLORATORY_FREEZE_20261013.json']:
        if p.exists():hashes[str(p.relative_to(ROOT))]=digest(p)
    paths=[Path(__file__).resolve(),Path(feature.__file__).resolve(),ROOT/'lvi_prediction/test_nyul_t2_20261013.py']
    OUT.mkdir(parents=True);(OUT/'models').mkdir();(OUT/'maps').mkdir()
    save(OUT/'protocol.json',{'date':'2026-10-13','authorization':'T2 Nyul extension in unchanged120; freeze then evaluate old36 and omitted18 together',
        'same120_and_84_36_split':True,'feature_protocol':feature.PROTOCOL,'analyses':ANALYSES,
        'candidates':[{**c,'normalization':'NyulT2'} for c in options()],'candidate_count':36,
        'nested':'outer5 seed42; inner5 seeds62201..62205; final5 seed62212',
        'features_and_parameters':'unchanged ICC>=.8,complete>=.9,bootstrap sign consistency>=.8,Spearman<.8;3/6 features;LR C.03/.3/1',
        'objective':'unchanged mean inner AUC-.10*SD-.30*max(mean fit-mean validation-.10,0)-.03*abs(mean fit-.75)',
        'baselines':'reuse frozen matching ZScore models and their identical outer predictions; no baseline alterations',
        'test':'freeze T2 and FusionFree Nyul finalists before54 feature generation/scoring; one joint scoring; all4 prespecified models reported',
        'formal_model_use':False,'automatic_adoption':False,'current_retained_model_changed':False,
        'MRI_before_neoadjuvant_and_pathology_uniform_user_confirmed':True,'Dstar_excluded':True,
        'test_previously_reused':True,'new_external_validation':False,'historical_adoption_decisions_unchanged':True,
        'protected_hashes':hashes,'source_hashes':{str(p.relative_to(ROOT)):digest(p) for p in paths},
        'sources':['https://pubmed.ncbi.nlm.nih.gov/10784285/','https://pmc.ncbi.nlm.nih.gov/articles/PMC4636718/']})
    save(OUT/'split_manifest.json',manifest)
    return cohort,manifest,y,tr,te

class Engine:
    def __init__(self,cohort):
        self.profiles=old.common.old.retained.load_profiles();self.ids=cohort.AnalysisID.to_numpy()
        self.landmarks=np.asarray([feature.landmarks(p['T2']) for p in self.profiles])
        self.baseline=joblib.load(old.OUT/'training_multiseq_features.joblib')
        self.lookup={int(i):j for j,i in enumerate(self.baseline['indices'])}
        self.cols=[j for j,m in enumerate(self.baseline['metadata']) if m['modality']=='T2']
        self.t2names=[self.baseline['metadata'][j]['name'] for j in self.cols]
        radiomics.logger.setLevel(40);sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)

    def materialize(self,mapping,indices,label,baseline=None):
        indices=np.asarray(indices,int);path=OUT/'maps'/(label+'.joblib')
        if path.exists():
            cached=joblib.load(path)
            assert cached['mapping']['target_sha256']==mapping['target_sha256']
            np.testing.assert_array_equal(cached['data']['indices'],indices)
            np.testing.assert_array_equal(cached['mapping']['train_indices'],mapping['train_indices'])
            return cached['data']
        results={};start=time.time()
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures={pool.submit(feature.extract_one,(int(i),self.ids[i],self.profiles[i]['T2'],self.landmarks[i],mapping)):int(i) for i in indices}
            for done,f in enumerate(as_completed(futures),1):
                i=futures[f];results[i]=f.result()
                assert [m['name'] for m in results[i]['metadata']]==self.t2names
                if done%16==0 or done==len(indices):print('%s extraction %d/%d %.0fs'%(label,done,len(indices),time.time()-start),flush=True)
        if baseline is None:
            rows=[self.lookup[int(i)] for i in indices]
            data={'indices':indices,'metadata':self.baseline['metadata'],**{k:self.baseline[k][rows].copy() for k in KINDS}}
        else:
            np.testing.assert_array_equal(baseline['indices'],indices)
            assert baseline['metadata']==self.baseline['metadata']
            data={'indices':indices,'metadata':baseline['metadata'],**{k:baseline[k].copy() for k in KINDS}}
        for kind in KINDS:data[kind][:,self.cols]=np.asarray([results[int(i)]['rows'][kind] for i in indices])
        audit=[r for i in indices for r in results[int(i)]['audit']]
        errors=[{'AnalysisID':self.ids[i],**r} for i in indices for r in results[int(i)]['errors']]
        joblib.dump({'mapping':mapping,'data':data,'audit':audit,'errors':errors},path,compress=3)
        return data

    def fold(self,a,b,label):
        mapping=feature.learn(self.landmarks,a,self.ids)
        assert not set(a)&set(b)
        return self.materialize(mapping,np.r_[a,b],label),mapping

def tune(engine,folds,y,label):
    values={c['candidate_id']:[] for c in options()};cv={c['candidate_id']:[] for c in options()}
    for fold,(a,b) in enumerate(folds,1):
        data,mapping=engine.fold(a,b,label+'_inner'+str(fold));rank=old.ranking(data,a,y);cache={}
        for row in options():
            c=config(row);key=(c['analysis'],c['family'],c['top'])
            if key not in cache:cache[key]=old.prepare(data,a,b,c,y,rank)
            f=old.fitted(cache[key],y[a],c)
            values[c['candidate_id']].append((roc_auc_score(y[b],f['evaluation_probability']),roc_auc_score(y[a],f['train_probability']),len(f['indices']),f['available']))
            cv[c['candidate_id']].append((b,f['evaluation_probability']))
        print(label+' inner fold '+str(fold)+'/5 selected features and parameters',flush=True)
    records=[]
    for row in options():
        c=config(row);v=np.asarray(values[c['candidate_id']]);mean=v[:,0].mean();sd=v[:,0].std(ddof=1);fit=v[:,1].mean();n=v[:,2].mean()
        valid=bool(v[:,3].all());obj=mean-.10*sd-.30*max(fit-mean-.10,0)-.03*abs(fit-.75)
        records.append({**c,'MeanValidationAUC':float(mean),'SD':float(sd),'MeanFitAUC':float(fit),'MeanFeatureCount':float(n),
            'AvailableAllFolds':valid,'Objective':float(obj) if valid else -1.})
    return records,cv

def choose(grid,a):return origin.choose(grid,a,'NyulT2')

def train():
    start=time.time();cohort,manifest,y,tr,te=freeze_protocol();engine=Engine(cohort)
    baseline=joblib.load(origin.OUT/'models/normalization_models.joblib')
    pred=pd.read_csv(origin.OUT/'nested_predictions.csv',float_precision='round_trip')
    pred=pred[pred.ModelKey.isin(['ZScore__'+a for a in ANALYSES])].to_dict('records')
    folds=pd.read_csv(origin.OUT/'outer_fold_metrics.csv',float_precision='round_trip')
    folds=folds[folds.ModelKey.isin(['ZScore__'+a for a in ANALYSES])].to_dict('records');grids=[]
    for fold,(a,b) in enumerate(old.common.old.previous.folds(tr,y,42,5),1):
        grid,_=tune(engine,old.common.old.previous.folds(a,y,62200+fold,5),y,'outer'+str(fold))
        data,mapping=engine.fold(a,b,'outer'+str(fold)+'_fit');stored={'train':a,'evaluation':b,'mapping':mapping,'models':{},'grid':grid}
        for analysis in ANALYSES:
            c=config(choose(grid,analysis));f=old.fit(data,a,b,y,c);f['nyul_mapping']=mapping;key='NyulT2__'+analysis
            stored['models'][key]=f
            folds.append({'ModelKey':key,'Fold':fold,**c,'FeatureCount':len(f['indices']),'AUC':float(roc_auc_score(y[b],f['evaluation_probability']))})
            pred.extend({'ModelKey':key,'Fold':fold,'AnalysisID':cohort.AnalysisID.iloc[i],'Outcome':int(y[i]),'Probability':float(p)} for i,p in zip(b,f['evaluation_probability']))
        joblib.dump(stored,OUT/'models'/('outer_fold_'+str(fold)+'.joblib'),compress=3)
        grids.extend({'OuterFold':fold,**r} for r in grid)
        print('Nyul outer fold '+str(fold)+'/5 complete',flush=True)
    grid,cv=tune(engine,old.common.old.previous.folds(tr,y,62212,5),y,'final')
    data,mapping=engine.fold(tr,[],'final_fit');final={'models':{},'train':tr,'metadata':data['metadata'],'nyul_mapping':mapping,'formal_model_use':False}
    choices=[];features=[];trainpred=[];finalcv=[]
    baseline_choices=pd.read_csv(origin.OUT/'final_training_choices.csv',float_precision='round_trip').set_index('ModelKey')
    predictions=pd.DataFrame(pred);foldmetrics=pd.DataFrame(folds)
    for key in KEYS:
        analysis=key.split('__')[1]
        if key.startswith('ZScore'):
            f=baseline['models'][key];row=baseline_choices.loc[key].to_dict()
        else:
            row=choose(grid,analysis);c=config(row);f=old.fit(data,tr,[],y,c);f['nyul_mapping']=mapping;f['train']=tr
            oof=np.full(138,np.nan)
            for i,p in cv[c['candidate_id']]:oof[i]=p
            assert np.isfinite(oof[tr]).all() and np.isnan(oof[te]).all()
            f['threshold']=old.common.threshold(y[tr],oof[tr])
            finalcv.extend({'ModelKey':key,'AnalysisID':cohort.AnalysisID.iloc[i],'Outcome':int(y[i]),'Probability':float(oof[i])} for i in tr)
        final['models'][key]=f
        selected=predictions[predictions.ModelKey==key].set_index('AnalysisID').loc[cohort.AnalysisID.iloc[tr]]
        ff=foldmetrics[foldmetrics.ModelKey==key].AUC
        choices.append({'ModelKey':key,**f['config'],'FeatureCount':len(f['indices']),'TrainFitAUC':float(roc_auc_score(y[tr],f['train_probability'])),
            'OuterMeanAUC':float(ff.mean()),'OuterSD':float(ff.std(ddof=1)),'PooledOOFAUC':float(roc_auc_score(y[tr],selected.Probability)),
            'FinalSelectionCVMeanAUC':row.get('MeanValidationAUC',row.get('FinalSelectionCVMeanAUC')),'Threshold':f['threshold'],
            'T2SelectedCount':sum(data['metadata'][j]['modality']=='T2' for j in f['indices'])})
        features.extend({'ModelKey':key,'Feature':n,'Modality':data['metadata'][j]['modality'],'Block':data['metadata'][j]['block'],
            'LRStandardizedCoefficient':float(beta)} for n,j,beta in zip(f['names'],f['indices'],f['model'].coef_[0] if f['model'] is not None else []))
        trainpred.extend({'ModelKey':key,'AnalysisID':cohort.AnalysisID.iloc[i],'Outcome':int(y[i]),'Probability':float(p)} for i,p in zip(tr,f['train_probability']))
    joblib.dump(final,OUT/'models/nyul_and_baseline_models.joblib',compress=3)
    for name,rows in [('nested_predictions',predictions),('outer_fold_metrics',foldmetrics),('inner_candidate_scores',grids),('final_training_grid',grid),
        ('final_training_choices',choices),('selected_features',features),('train_fit_predictions',trainpred),('final_selection_oof_predictions',finalcv)]:
        pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')
    save(OUT/'training_frozen.json',{'passed':True,'model_sha256':digest(OUT/'models/nyul_and_baseline_models.joblib'),
        'protocol_sha256':digest(OUT/'protocol.json'),'map_and_feature_hashes':{p.name:digest(p) for p in (OUT/'maps').glob('*.joblib')},
        'csv_hashes':{p.name:digest(p) for p in OUT.glob('*.csv')},'test_scored':False,'elapsed_seconds':time.time()-start,'protected_count':protected()})
    print(pd.DataFrame(choices).to_string(index=False),flush=True)
    print('Nyul mapping, models, features and thresholds frozen BEFORE54 inference',flush=True)

def frozen():
    record=json.loads((OUT/'training_frozen.json').read_text(encoding='utf-8'));protected()
    assert digest(OUT/'models/nyul_and_baseline_models.joblib')==record['model_sha256']
    assert digest(OUT/'protocol.json')==record['protocol_sha256']
    for n,h in record['map_and_feature_hashes'].items():assert digest(OUT/'maps'/n)==h,n
    for n,h in record['csv_hashes'].items():assert digest(OUT/n)==h,n
    return record

def evaluate():
    if (OUT/'evaluation_started.json').exists():raise FileExistsError('Do not repeat54 scoring')
    frozen_record=frozen();assert json.loads((OUT/'training_verification.json').read_text())['passed']
    cohort,manifest,_,tr,te=origin.inputs();alltest=np.r_[te,np.asarray(manifest['omitted_indices'],int)]
    final=joblib.load(OUT/'models/nyul_and_baseline_models.joblib');engine=Engine(cohort)
    baseline=joblib.load(previous54.OUT/'frozen_inference_inputs54.joblib')['ZScore']
    np.testing.assert_array_equal(baseline['indices'],alltest);np.testing.assert_array_equal(final['nyul_mapping']['train_indices'],tr)
    assert not set(tr)&set(alltest)
    nyul=engine.materialize(final['nyul_mapping'],alltest,'frozen54',baseline=baseline)
    members=pd.read_csv(previous54.OUT/'membership54.csv');members.to_csv(OUT/'membership54.csv',index=False,encoding='utf-8-sig')
    history=pd.read_csv(previous54.OUT/'historical_exposure54.csv');history.to_csv(OUT/'historical_exposure54.csv',index=False,encoding='utf-8-sig')
    save(OUT/'evaluation_started.json',{'joint_count':1,'model_sha256':frozen_record['model_sha256'],'map_target_sha256':final['nyul_mapping']['target_sha256'],
        'evaluation_ids':members.AnalysisID.tolist(),'training_overlap':0,'models_refitted':0,'thresholds_changed':False})
    scores={k:old.predict(f,baseline if k.startswith('ZScore') else nyul) for k,f in final['models'].items()}
    oldpred=pd.read_csv(previous54.OUT/'predictions54.csv',float_precision='round_trip');rows=[];metrics=[];pairs=[]
    y=cohort.Outcome.to_numpy(int)[alltest]
    for k,p in scores.items():
        if k.startswith('ZScore'):
            reference=oldpred[oldpred.ModelKey==k].set_index('AnalysisID').loc[members.AnalysisID].Probability.to_numpy()
            np.testing.assert_allclose(p,reference,atol=1e-12,rtol=0)
        f=final['models'][k]
        rows.extend({'ModelKey':k,'AnalysisID':members.AnalysisID.iloc[i],'Group':members.Group.iloc[i],'Outcome':int(y[i]),
            'Probability':float(pp),'Threshold':f['threshold'],'PredictedPositive':int(pp>=f['threshold'])} for i,pp in enumerate(p))
    for group,positions in previous54.GROUPS.items():
        yy=y[positions];ps={k:p[positions] for k,p in scores.items()};draws=old.common.old.retained.bootstrap(yy,ps)
        for k,p in ps.items():
            low,high=np.quantile(draws[k],[.025,.975]);f=final['models'][k]
            metrics.append({'Group':group,'ModelKey':k,'N':len(yy),'Positive':int(yy.sum()),'Negative':int((yy==0).sum()),
                **old.common.old.metrics(yy,p,f['threshold']),'AUC_CI95Low':float(low),'AUC_CI95High':float(high)})
        for a in ANALYSES:
            ka,kb='NyulT2__'+a,'ZScore__'+a;lo,hi=np.quantile(draws[ka]-draws[kb],[.025,.975])
            pairs.append({'Group':group,'Analysis':a,'DeltaAUC':float(roc_auc_score(yy,ps[ka])-roc_auc_score(yy,ps[kb])),
                'CI95Low':float(lo),'CI95High':float(hi)})
    for name,r in [('predictions54',rows),('metrics54',metrics),('paired_auc_comparisons',pairs)]:pd.DataFrame(r).to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')
    save(OUT/'evaluation_complete.json',{'passed':True,'joint_count':1,'models_refitted':0,'thresholds_changed':False,'training_overlap':0,
        'model_sha256':frozen_record['model_sha256'],'prediction_rows':216,'protected_count':protected(),'formal_model_use':False})
    print(pd.DataFrame(metrics).to_string(index=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['train','evaluate'],required=True);args=p.parse_args()
    with threadpool_limits(limits=1):train() if args.phase=='train' else evaluate()
