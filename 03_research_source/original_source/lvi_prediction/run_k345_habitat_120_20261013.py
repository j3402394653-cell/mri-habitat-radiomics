"""Two normalization routes x K3/4/5, nested selection, then joint frozen54."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import json
import shutil
import time
import joblib
import numpy as np
import pandas as pd
import radiomics
import SimpleITK as sitk
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
import lvi_paths
import k345_habitat_features_20261013 as feature
import run_normalization_habitat_120_20261012 as origin
import run_nyul_t2_120_20261013 as previous

ROOT=origin.ROOT
OUT=ROOT/'lvi_prediction/results/lvi_k345_habitat_120_20261013'
old=origin.old
save,digest=origin.save,origin.digest
NORMALIZATIONS=feature.NORMALIZATIONS
KS=feature.KS
ANALYSES=tuple(origin.GROUPS)
KINDS=('base','erode','dilate')

def key(n,k,a):return n+'__K'+str(k)+'__'+a

def options():
    result=[]
    for n in NORMALIZATIONS:
        for k in KS:
            for a in ANALYSES:
                for family in ['Descriptor','Texture','Combined']:
                    for top in [3,6]:
                        for C in [.03,.3,1.]:
                            result.append({'normalization':n,'K':k,'analysis':a,'family':family,'top':top,'quota':0,'algorithm':'LR','C':C,'candidate_id':len(result)})
    return result

def config(row):return {k:row[k] for k in options()[0]}

def choose(grid,a,n=None,k=None):
    pool=[r for r in grid if r['analysis']==a and (n is None or r['normalization']==n) and (k is None or r['K']==k)]
    valid=[r for r in pool if r['AvailableAllFolds']]
    return max(valid if valid else pool,key=lambda r:(r['Objective'],-r['MeanFeatureCount'],-r['K'],-r['C'],-r['candidate_id']))

def selections(grid):
    result={key(n,k,a):choose(grid,a,n,k) for n in NORMALIZATIONS for k in KS for a in ANALYSES}
    result.update({key(n,'Adaptive',a):choose(grid,a,n) for n in NORMALIZATIONS for a in ANALYSES})
    result['JointSelection__FusionFree']=choose(grid,'FusionFree')
    return result

def protect():
    p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    for name,h in p['protected_hashes'].items():assert digest(ROOT/name)==h,name
    for name,h in p['source_hashes'].items():assert digest(ROOT/name)==h,name
    return len(p['protected_hashes'])

def freeze_protocol():
    if OUT.exists():raise FileExistsError('K comparison already exists; do not overwrite')
    previous.frozen();previous.previous54.protect()
    policy_path=ROOT/'lvi_prediction/FUTURE_HABITAT_EXPERIMENT_POLICY_20261013.json'
    policy=json.loads(policy_path.read_text(encoding='utf-8'))
    assert policy['normalization_routes']==list(NORMALIZATIONS)
    assert policy['after_freeze_scoring']['score_all54_after_freeze'] and not policy['reuse36_predictions_in54_report']
    cohort,manifest,y,tr,te=origin.inputs();alltest=np.r_[te,np.asarray(manifest['omitted_indices'],int)]
    assert policy['train_ids']==cohort.AnalysisID.iloc[tr].tolist() and policy['evaluation54_ids']==cohort.AnalysisID.iloc[alltest].tolist()
    protected=dict(json.loads((previous.OUT/'protocol.json').read_text(encoding='utf-8'))['protected_hashes'])
    for directory in [previous.OUT,ROOT/'lvi_prediction/results/lvi_normalization_comparison_20261013']:
        protected.update({str(p.relative_to(ROOT)):digest(p) for p in directory.rglob('*') if p.is_file()})
    for p in [policy_path,policy_path.with_suffix('.md')]:protected[str(p.relative_to(ROOT))]=digest(p)
    sources=[Path(__file__).resolve(),Path(feature.__file__).resolve(),ROOT/'lvi_prediction/test_k345_habitat_20261013.py']
    OUT.mkdir(parents=True);(OUT/'models').mkdir()
    save(OUT/'protocol.json',{'date':'2026-10-13','authorization':'test habitat K3/4/5 on both PercentileZ and RobustZ; same120 then direct joint54 after freeze',
        'normalizations':NORMALIZATIONS,'K_values':KS,'analyses':ANALYSES,'feature_protocol':feature.PROTOCOL,
        'same120_84_36_split':True,'sample_seed':manifest['sample_seed'],'split_seed':manifest['split_seed'],
        'nested':'outer5 seed42; inner5 seeds62201..62205; final inner5 seed62212',
        'feature_selection':'unchanged training ICC>=.8,complete>=.9,5 stratified bootstrap sign consistency>=.8; redundancy abs Spearman<.8',
        'model_candidates':'unchanged Descriptor/Texture/Combined; maximum3/6features; LR L2 C.03/.3/1',
        'objective':'unchanged mean inner AUC-.10*SD-.30*max(mean fit-mean validation-.10,0)-.03*abs(mean fit-.75)',
        'candidates':options(),'candidate_count':432,'fixed_K_models':24,'inner_K_selected_references':8,'joint_norm_K_fusion_reference':1,
        'selection':'K/features/parameters jointly selected inside inner loops for each normalization; extra fusion selects normalization+K inside inner loops',
        'recommendation':'final training-inner selection plus outer assessment; ties prefer fewer features then lower K; do not choose by54 highest AUC',
        'initial_test':'freeze trained parameters then score original36 once for120 report',
        'frozen54':'same models copied and confirmed frozen after120 stage; directly call each distinct model with all54 in one batch; do not reuse old36 probabilities',
        'case_feature_caches':'deterministic patient-local image features may be reused; all54 model probabilities generated anew',
        'Dstar_excluded':True,'MRI_before_neoadjuvant_and_pathology_uniform_user_confirmed':True,
        'formal_model_use':False,'automatic_adoption':False,'current_model_pointer_changed':False,'test_previously_reused':True,
        'new_external_validation':False,'prior_nonadoption_decisions_unchanged':True,
        'protected_hashes':protected,'source_hashes':{str(p.relative_to(ROOT)):digest(p) for p in sources},
        'sources':['https://scikit-learn.org/1.3/modules/generated/sklearn.cluster.KMeans.html',
            'https://pyradiomics.readthedocs.io/en/v3.0/customization.html']})
    save(OUT/'split_manifest.json',manifest)
    return cohort,manifest,y,tr,te

def extract(indices,cohort,stage,k3):
    start=time.time();profiles=old.common.old.retained.load_profiles();results={}
    radiomics.logger.setLevel(40);sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(feature.extract_one,(int(i),cohort.AnalysisID.iloc[i],profiles[int(i)],(4,5))):int(i) for i in indices}
        for done,f in enumerate(as_completed(futures),1):
            i=futures[f];results[i]=f.result()
            if done%4==0 or done==len(indices):print('K345 %s extraction %d/%d %.0fs'%(stage,done,len(indices),time.time()-start),flush=True)
    data={n:{3:{k:k3[n][k] for k in ['base','erode','dilate','indices','metadata']}} for n in NORMALIZATIONS}
    for n in NORMALIZATIONS:
        np.testing.assert_array_equal(k3[n]['indices'],indices)
        for k in [4,5]:
            metadata=results[int(indices[0])]['metadata'][k];names=[m['name'] for m in metadata]
            assert len(names)==len(set(names))==3*k*k+405*k
            assert all(results[int(i)]['metadata'][k]==metadata for i in indices)
            data[n][k]={'metadata':metadata,'indices':np.asarray(indices,int),
                **{kind:np.asarray([[results[int(i)]['values'][n][k][kind][name] for name in names] for i in indices],float) for kind in KINDS}}
    for name in ['partition_audit','texture_audit','normalization_audit','contour_partition_audit']:
        pd.DataFrame([r for i in indices for r in results[int(i)][name]]).to_csv(OUT/(stage+'_'+name+'.csv'),index=False,encoding='utf-8-sig')
    joblib.dump(data,OUT/(stage+'_features.joblib'),compress=3)
    return data

def fit(data,a,b,y,c):return old.fit(data[c['normalization']][c['K']],a,b,y,c)

def tune(data,folds,y,label):
    values={c['candidate_id']:[] for c in options()};cv={c['candidate_id']:[] for c in options()}
    for fold,(a,b) in enumerate(folds,1):
        ranks={(n,k):old.ranking(data[n][k],a,y) for n in NORMALIZATIONS for k in KS};cache={}
        for c in options():
            group=(c['normalization'],c['K']);identity=group+(c['analysis'],c['family'],c['top'])
            if identity not in cache:cache[identity]=old.prepare(data[group[0]][group[1]],a,b,c,y,ranks[group])
            f=old.fitted(cache[identity],y[a],c)
            values[c['candidate_id']].append((roc_auc_score(y[b],f['evaluation_probability']),roc_auc_score(y[a],f['train_probability']),len(f['indices']),f['available']))
            cv[c['candidate_id']].append((b,f['evaluation_probability']))
        print('K345 '+label+' inner'+str(fold)+'/5 complete',flush=True)
    records=[]
    for c in options():
        v=np.asarray(values[c['candidate_id']]);mean,sd,train,n=float(v[:,0].mean()),float(v[:,0].std(ddof=1)),float(v[:,1].mean()),float(v[:,2].mean())
        valid=bool(v[:,3].all());objective=mean-.10*sd-.30*max(train-mean-.10,0)-.03*abs(train-.75)
        records.append({**c,'MeanValidationAUC':mean,'SD':sd,'MeanFitAUC':train,'MeanFeatureCount':n,'AvailableAllFolds':valid,'Objective':objective if valid else -1.})
    return records,cv

def train():
    start=time.time();cohort,manifest,y,tr,te=freeze_protocol()
    cached=joblib.load(origin.OUT/'training_features.joblib');data=extract(tr,cohort,'training',cached)
    predictions=[];metrics=[];grids=[];frequency=[]
    for fold,(a,b) in enumerate(old.common.old.previous.folds(tr,y,42,5),1):
        grid,_=tune(data,old.common.old.previous.folds(a,y,62200+fold,5),y,'outer'+str(fold));chosen=selections(grid)
        stored={'train':a,'evaluation':b,'grid':grid,'models':{}};cache={};prior=joblib.load(origin.OUT/('models/outer_fold_'+str(fold)+'.joblib'))['models']
        for model_key,row in chosen.items():
            c=config(row)
            if c['candidate_id'] not in cache:cache[c['candidate_id']]=fit(data,a,b,y,c)
            f=cache[c['candidate_id']];stored['models'][model_key]=f
            if '__K3__' in model_key:origin.baseline_equal(f,prior[origin.key(c['normalization'],c['analysis'])])
            metrics.append({'ModelKey':model_key,'Fold':fold,**c,'FeatureCount':len(f['indices']),'AUC':float(roc_auc_score(y[b],f['evaluation_probability']))})
            predictions.extend({'ModelKey':model_key,'Fold':fold,'AnalysisID':cohort.AnalysisID.iloc[i],'Outcome':int(y[i]),'Probability':float(p)} for i,p in zip(b,f['evaluation_probability']))
            frequency.extend({'ModelKey':model_key,'Fold':fold,'Normalization':c['normalization'],'K':c['K'],'Feature':name} for name in f['names'])
        joblib.dump(stored,OUT/('models/outer_fold_'+str(fold)+'.joblib'),compress=3)
        grids.extend({'OuterFold':fold,**r} for r in grid);print('K345 outer'+str(fold)+'/5 complete',flush=True)
    grid,cv=tune(data,old.common.old.previous.folds(tr,y,62212,5),y,'final');chosen=selections(grid)
    final={'models':{},'train':tr,'metadata_by_K':{k:data[NORMALIZATIONS[0]][k]['metadata'] for k in KS},'formal_model_use':False}
    prior=joblib.load(origin.OUT/'models/normalization_models.joblib')['models'];cache={};choices=[];features=[];trainpred=[];oofpred=[];gates=[]
    predictions=pd.DataFrame(predictions);folds=pd.DataFrame(metrics)
    for model_key,row in chosen.items():
        c=config(row)
        if c['candidate_id'] not in cache:
            f=fit(data,tr,[],y,c);oof=np.full(138,np.nan)
            for ix,p in cv[c['candidate_id']]:oof[ix]=p
            assert np.isfinite(oof[tr]).all() and np.isnan(oof[te]).all()
            f['threshold']=old.common.threshold(y[tr],oof[tr]);f['train']=tr;cache[c['candidate_id']]=(f,oof)
        f,oof=cache[c['candidate_id']];final['models'][model_key]=f
        if '__K3__' in model_key:
            reference=prior[origin.key(c['normalization'],c['analysis'])];origin.baseline_equal(f,reference);assert f['threshold']==reference['threshold']
        meta=final['metadata_by_K'][c['K']];ff=folds[folds.ModelKey==model_key].AUC
        pp=predictions[predictions.ModelKey==model_key].set_index('AnalysisID').loc[cohort.AnalysisID.iloc[tr]]
        choices.append({'ModelKey':model_key,'KChoice':'Adaptive' if 'KAdaptive' in model_key or model_key.startswith('JointSelection') else 'Fixed',**c,
            'FeatureCount':len(f['indices']),'TrainFitAUC':float(roc_auc_score(y[tr],f['train_probability'])),
            'FinalSelectionCVMeanAUC':row['MeanValidationAUC'],'SelectionObjective':row['Objective'],
            'OuterMeanAUC':float(ff.mean()),'OuterSD':float(ff.std(ddof=1)),'PooledOOFAUC':float(roc_auc_score(y[tr],pp.Probability)),
            'Threshold':f['threshold'],**{mod+'Count':sum(meta[j]['modality']==mod for j in f['indices']) for mod in feature.MODALITIES}})
        features.extend({'ModelKey':model_key,'Normalization':c['normalization'],'K':c['K'],'Analysis':c['analysis'],'Feature':name,
            'Modality':meta[j]['modality'],'Block':meta[j]['block'],'LRStandardizedCoefficient':float(beta)}
            for name,j,beta in zip(f['names'],f['indices'],f['model'].coef_[0] if f['model'] is not None else []))
        trainpred.extend({'ModelKey':model_key,'AnalysisID':cohort.AnalysisID.iloc[i],'Outcome':int(y[i]),'Probability':float(p)} for i,p in zip(tr,f['train_probability']))
        oofpred.extend({'ModelKey':model_key,'AnalysisID':cohort.AnalysisID.iloc[i],'Outcome':int(y[i]),'Probability':float(oof[i])} for i in tr)
    for n in NORMALIZATIONS:
        for k in KS:
            _,_,audit=old.ranking(data[n][k],tr,y)
            gates.extend({'Normalization':n,'K':k,'Modality':m['modality'],'Block':m['block'],**r} for m,r in zip(data[n][k]['metadata'],audit))
    joblib.dump(final,OUT/'models/k345_models.joblib',compress=3)
    for name,rows in [('nested_predictions',predictions),('outer_fold_metrics',folds),('inner_candidate_scores',grids),('final_training_grid',grid),
        ('final_training_choices',choices),('selected_features',features),('train_fit_predictions',trainpred),('final_selection_oof_predictions',oofpred),
        ('feature_gate',gates),('outer_feature_selections',frequency)]:pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')
    recommendation=next(r for r in choices if r['ModelKey']=='JointSelection__FusionFree')
    save(OUT/'training_only_recommendation.json',{'overall_fusion':recommendation,'per_normalization':[r for r in choices if r['ModelKey'] in [key(n,'Adaptive','FusionFree') for n in NORMALIZATIONS]],
        'selection':'final training inner5 objective; complete selection assessed by outer5','test_used':False,'formal_model_use':False})
    save(OUT/'training_frozen.json',{'passed':True,'model_sha256':digest(OUT/'models/k345_models.joblib'),
        'protocol_sha256':digest(OUT/'protocol.json'),'feature_sha256':digest(OUT/'training_features.joblib'),
        'training_csv_hashes':{p.name:digest(p) for p in OUT.glob('*.csv')},'test_scored':False,'protected_count':protect(),'elapsed_seconds':time.time()-start})
    print(pd.DataFrame(choices)[['ModelKey','K','TrainFitAUC','OuterMeanAUC','OuterSD','FeatureCount']].to_string(index=False),flush=True)
    print('K345 training frozen; no test scores generated',flush=True)

def frozen():
    p=json.loads((OUT/'training_frozen.json').read_text(encoding='utf-8'));protect()
    for name,field in [('models/k345_models.joblib','model_sha256'),('protocol.json','protocol_sha256'),('training_features.joblib','feature_sha256')]:assert digest(OUT/name)==p[field]
    for name,h in p['training_csv_hashes'].items():assert digest(OUT/name)==h,name
    return p

def probabilities(final,data):
    scores={};calls=0
    for n in NORMALIZATIONS:
        for k in KS:
            for a in ANALYSES:
                name=key(n,k,a);scores[name]=old.predict(final['models'][name],data[n][k]);calls+=1
    for name,f in final['models'].items():
        if name not in scores:scores[name]=scores[key(f['config']['normalization'],f['config']['K'],f['config']['analysis'])]
    return scores,calls

def write_evaluation(final,data,cohort,indices,stage,groups):
    scores,calls=probabilities(final,data);outcomes=cohort.Outcome.to_numpy(int)[indices];rows=[];metrics=[];pairs=[]
    for name,p in scores.items():
        threshold=final['models'][name]['threshold']
        rows.extend({'ModelKey':name,'AnalysisID':cohort.AnalysisID.iloc[i],'Group':'PreviousTest36' if j<36 else 'PreviouslyOmitted18',
            'Outcome':int(y),'Probability':float(pp),'Threshold':threshold,'PredictedPositive':int(pp>=threshold)} for j,(i,y,pp) in enumerate(zip(indices,outcomes,p)))
    for group,pos in groups.items():
        y=outcomes[pos];ps={name:p[pos] for name,p in scores.items()};draws=old.common.old.retained.bootstrap(y,ps)
        for name,p in ps.items():
            f=final['models'][name];lo,hi=np.quantile(draws[name],[.025,.975])
            metrics.append({'Group':group,'ModelKey':name,'Normalization':f['config']['normalization'],'K':f['config']['K'],'Analysis':f['config']['analysis'],
                'N':len(y),'Positive':int(y.sum()),'Negative':int((y==0).sum()),**old.common.old.metrics(y,p,f['threshold']),
                'AUC_CI95Low':float(lo),'AUC_CI95High':float(hi)})
        for n in NORMALIZATIONS:
            for a in ANALYSES:
                for ka,kb in [(4,3),(5,3),(5,4)]:
                    first,second=key(n,ka,a),key(n,kb,a);lo,hi=np.quantile(draws[first]-draws[second],[.025,.975])
                    pairs.append({'Group':group,'Normalization':n,'Analysis':a,'K':ka,'ReferenceK':kb,
                        'DeltaAUC':float(roc_auc_score(y,ps[first])-roc_auc_score(y,ps[second])),'CI95Low':float(lo),'CI95High':float(hi)})
        for k in KS:
            for a in ANALYSES:
                first,second=key('PercentileZ',k,a),key('RobustZ',k,a);lo,hi=np.quantile(draws[first]-draws[second],[.025,.975])
                pairs.append({'Group':group,'Normalization':'PercentileZ minus RobustZ','Analysis':a,'K':k,'ReferenceK':k,
                    'DeltaAUC':float(roc_auc_score(y,ps[first])-roc_auc_score(y,ps[second])),'CI95Low':float(lo),'CI95High':float(hi)})
    for name,r in [('predictions',rows),('metrics',metrics),('paired_auc_comparisons',pairs)]:pd.DataFrame(r).to_csv(OUT/(stage+'_'+name+'.csv'),index=False,encoding='utf-8-sig')
    return scores,calls

def test120():
    if (OUT/'test120_started.json').exists():raise FileExistsError('Do not repeat initial36 test stage')
    f=frozen();assert json.loads((OUT/'training_verification.json').read_text())['passed']
    cohort,manifest,y,tr,te=origin.inputs();k3=joblib.load(origin.OUT/'test_features.joblib');data=extract(te,cohort,'test120',k3)
    final=joblib.load(OUT/'models/k345_models.joblib')
    save(OUT/'test120_started.json',{'joint_count':1,'n':36,'distinct_models':24,'reference_models':9,'model_sha256':f['model_sha256']})
    scores,calls=write_evaluation(final,data,cohort,te,'test120',{'PreviousTest36':np.arange(36)})
    prior=pd.read_csv(origin.OUT/'test_predictions.csv',float_precision='round_trip')
    for n in NORMALIZATIONS:
        for a in ANALYSES:
            ref=prior[prior.ModelKey==origin.key(n,a)].set_index('AnalysisID').loc[cohort.AnalysisID.iloc[te]].Probability
            np.testing.assert_allclose(scores[key(n,3,a)],ref,atol=1e-12,rtol=0)
    save(OUT/'test120_complete.json',{'passed':True,'joint_count':1,'n':36,'prediction_rows':33*36,'distinct_predict_calls':calls,'K3_baselines_replayed':True,
        'model_sha256':f['model_sha256'],'protected_count':protect()})
    print('Initial120 experiment complete: train84 and test36; K3 predictions exactly reproduced',flush=True)

def freeze54():
    if (OUT/'frozen54_started.json').exists():raise FileExistsError('Do not repeat frozen54 prediction stage')
    f=frozen();assert json.loads((OUT/'test120_complete.json').read_text())['passed']
    cohort,manifest,y,tr,te=origin.inputs();omitted=np.asarray(manifest['omitted_indices'],int);alltest=np.r_[te,omitted]
    snapshot=OUT/'models/post120_frozen_models.joblib'
    shutil.copyfile(OUT/'models/k345_models.joblib',snapshot);assert digest(snapshot)==f['model_sha256']
    final=joblib.load(snapshot);np.testing.assert_array_equal(final['train'],tr)
    assert not set(tr)&set(alltest)
    save(OUT/'post120_freeze_record.json',{'model_sha256':digest(snapshot),'same_as_training_snapshot':True,'no_refit_after36':True,
        'before_frozen54_scoring':True,'training_ids':manifest['train_ids'],'evaluation_ids':cohort.AnalysisID.iloc[alltest].tolist(),
        'thresholds':{name:float(model['threshold']) for name,model in final['models'].items()},'model_specifications':{
            name:{'config':model['config'],'features':model['names'],'coefficients':model['model'].coef_[0].tolist() if model['model'] is not None else [],
                'intercept':float(model['model'].intercept_[0]) if model['model'] is not None else None} for name,model in final['models'].items()}})
    allk3=joblib.load(previous.previous54.OUT/'frozen_inference_inputs54.joblib')
    k3={n:{'indices':omitted,'metadata':allk3[n]['metadata'],**{kind:allk3[n][kind][36:] for kind in KINDS}} for n in NORMALIZATIONS}
    new=extract(omitted,cohort,'frozen54_new18',k3);prior=joblib.load(OUT/'test120_features.joblib');data={}
    for n in NORMALIZATIONS:
        data[n]={3:{kind:allk3[n][kind] for kind in ['base','erode','dilate','metadata','indices']}}
        np.testing.assert_array_equal(data[n][3]['indices'],alltest)
        for k in [4,5]:
            assert prior[n][k]['metadata']==new[n][k]['metadata']==final['metadata_by_K'][k]
            data[n][k]={'indices':alltest,'metadata':final['metadata_by_K'][k],**{kind:np.vstack([prior[n][k][kind],new[n][k][kind]]) for kind in KINDS}}
    joblib.dump(data,OUT/'frozen54_features.joblib',compress=3)
    for name in ['partition_audit','texture_audit','normalization_audit','contour_partition_audit']:
        combined=pd.concat([pd.read_csv(OUT/('test120_'+name+'.csv')),pd.read_csv(OUT/('frozen54_new18_'+name+'.csv'))],ignore_index=True)
        combined.to_csv(OUT/('frozen54_'+name+'.csv'),index=False,encoding='utf-8-sig')
    pd.read_csv(previous.previous54.OUT/'membership54.csv').to_csv(OUT/'membership54.csv',index=False,encoding='utf-8-sig')
    pd.read_csv(previous.previous54.OUT/'historical_exposure54.csv').to_csv(OUT/'historical_exposure54.csv',index=False,encoding='utf-8-sig')
    save(OUT/'frozen54_started.json',{'joint_count':1,'n':54,'direct_all54_new_predictions':True,'reuse_initial36_probabilities':False,
        'separate36_prediction_calls':0,'separate18_prediction_calls':0,'model_sha256':digest(snapshot),'training_overlap':0})
    scores,calls=write_evaluation(final,data,cohort,alltest,'frozen54',previous.previous54.GROUPS)
    initial=pd.read_csv(OUT/'test120_predictions.csv',float_precision='round_trip')
    for name,p in scores.items():
        ref=initial[initial.ModelKey==name].set_index('AnalysisID').loc[cohort.AnalysisID.iloc[te]].Probability.to_numpy()
        np.testing.assert_allclose(p[:36],ref,atol=1e-12,rtol=0)
    assert digest(snapshot)==f['model_sha256']
    save(OUT/'frozen54_complete.json',{'passed':True,'joint_count':1,'n':54,'direct_all54_new_predictions':True,'distinct_predict_calls':calls,
        'reference_models_reuse_current54_scores':9,'old36_probabilities_not_used_to_build54':True,'initial36_vs_new54_subset_max_error':0.,
        'training_overlap':0,'models_refitted':0,'thresholds_changed':False,'prediction_rows':33*54,'model_sha256':f['model_sha256'],'protected_count':protect()})
    print('Frozen direct54 complete: new one-batch scores;36/18/all54 subgroup metrics saved',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['train','test120','freeze54'],required=True);args=parser.parse_args()
    with threadpool_limits(limits=1):{'train':train,'test120':test120,'freeze54':freeze54}[args.phase]()
