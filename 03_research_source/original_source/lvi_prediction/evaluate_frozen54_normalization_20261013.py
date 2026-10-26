"""Frozen normalization inference on previous36 plus previously omitted18."""
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
from scipy.special import expit
from threadpoolctl import threadpool_limits
import run_normalization_habitat_120_20261012 as origin

ROOT=origin.ROOT
OUT=ROOT/'lvi_prediction/results/lvi_frozen54_normalization_20261013'
PRIMARY='Adaptive__FusionFree'
GROUPS={'PreviousTest36':np.arange(36),'PreviouslyOmitted18':np.arange(36,54),'AllNonTraining54':np.arange(54)}
GROUP_LABELS={'PreviousTest36':'原测试36例','PreviouslyOmitted18':'此前未抽中18例','AllNonTraining54':'全部54例'}
save,digest=origin.save,origin.digest


def protect():
    p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    for n,h in p['protected_hashes'].items():assert digest(ROOT/n)==h,n
    assert digest(OUT/'models/frozen_normalization_models.joblib')==p['model_sha256']
    assert digest(OUT/'membership54.csv')==p['membership_sha256']
    return len(p['protected_hashes'])


def freeze():
    if OUT.exists():raise FileExistsError('Frozen54 evaluation already exists; do not overwrite or repeat')
    origin.frozen_training()
    for file in ['test_verification.json','report_verification.json']:assert json.loads((origin.OUT/file).read_text())['passed']
    cohort,manifest,_,tr,te=origin.inputs();omitted=np.asarray(manifest['omitted_indices'],int);all_test=np.r_[te,omitted]
    assert len(te)==36 and len(omitted)==18 and len(set(all_test))==54
    assert not set(tr)&set(all_test) and set(tr)|set(all_test)==set(range(138))
    source=origin.OUT/'models/normalization_models.joblib';final=joblib.load(source)
    np.testing.assert_array_equal(final['train'],tr)
    assert set(final['models'])==set(origin.all_keys())
    f=final['models'][PRIMARY]
    assert f['config']['normalization']=='PercentileZ' and f['config']['analysis']=='FusionFree'
    assert len(f['names'])==3 and abs(f['threshold']-.7204498116102594)<1e-12
    recommendation=json.loads((origin.OUT/'training_only_recommendation.json').read_text(encoding='utf-8'))
    assert next(r for r in recommendation['per_group'] if r['ModelKey']==PRIMARY)['candidate_id']==f['config']['candidate_id']
    protected=dict(json.loads((origin.OUT/'protocol.json').read_text(encoding='utf-8'))['protected_hashes'])
    for folder in [origin.OUT,ROOT/'lvi_prediction/results/lvi_frozen54_clinical_extension_20261012',ROOT/'lvi_prediction/results/lvi_frozen54_pure_habitat_20261012']:
        protected.update({str(p.relative_to(ROOT)):digest(p) for p in folder.rglob('*') if p.is_file()})
    extra=[Path(__file__).resolve(),ROOT/'lvi_prediction/NORMALIZATION_STUDY_STATUS_20261013.json',
           ROOT/'ACTIVE_COHORT.json',origin.old.common.old.retained.POOL/'split_manifest.json']
    protected.update({str(p.relative_to(ROOT)):digest(p) for p in extra})
    OUT.mkdir(parents=True);(OUT/'models').mkdir();shutil.copyfile(source,OUT/'models/frozen_normalization_models.joblib')
    assert digest(source)==digest(OUT/'models/frozen_normalization_models.joblib')
    members=pd.DataFrame({'AnalysisID':cohort.AnalysisID.iloc[all_test].to_numpy(),'CohortIndex':all_test,
        'Group':['PreviousTest36']*36+['PreviouslyOmitted18']*18,'Outcome':cohort.Outcome.iloc[all_test].to_numpy(int)})
    previous=pd.read_csv(ROOT/'lvi_prediction/results/lvi_frozen54_clinical_extension_20261012/membership54.csv')
    assert members.AnalysisID.tolist()==previous.AnalysisID.tolist()
    np.testing.assert_array_equal(members.Outcome,previous.Outcome)
    members.to_csv(OUT/'membership54.csv',index=False,encoding='utf-8-sig')
    history=members[['AnalysisID','Group']].copy()
    for label,path in [('Historical118',ROOT/'ACTIVE_COHORT.json'),('Historical138',origin.old.common.old.retained.POOL/'split_manifest.json')]:
        old=json.loads(path.read_text(encoding='utf-8'));train_ids,test_ids=set(old['train_ids']),set(old['test_ids']);assert not train_ids&test_ids
        history[label+'Role']=['train' if aid in train_ids else 'test' if aid in test_ids else 'not_in_split' for aid in history.AnalysisID]
    history.to_csv(OUT/'historical_exposure54.csv',index=False,encoding='utf-8-sig')
    counts={g:{'n':len(ix),'positive':int(members.Outcome.iloc[ix].sum()),'negative':int((members.Outcome.iloc[ix]==0).sum())} for g,ix in GROUPS.items()}
    assert counts['PreviouslyOmitted18']=={'n':18,'positive':11,'negative':7}
    save(OUT/'protocol.json',{'date':'2026-10-13','authorization':'冻结当前模型，评估之前全部54例并分别报告36例和18例',
        'source_result_directory':str(origin.OUT.relative_to(ROOT)),'primary_model_key':PRIMARY,
        'primary_selection':'same training-only selected normalization/fusion finalist; not selected using new18 or54 scores',
        'model_sha256':digest(source),'membership_sha256':digest(OUT/'membership54.csv'),
        'model_keys':origin.all_keys(),'distinct_models':12,'adaptive_references':4,'groups':counts,
        'training_ids':manifest['train_ids'],'evaluation_ids':members.AnalysisID.tolist(),'training_overlap':0,
        'no_refit_retuning_or_threshold_change':True,'old36':'reuse frozen features and reproduce previous probabilities exactly',
        'new18':'patient-local image preprocessing/partition under unchanged frozen rules; no cohort learning or outcome-driven adaptation',
        'Dstar_excluded':True,'MRI_before_neoadjuvant_and_pathology_uniform_user_confirmed':True,
        'feature_protocol':origin.feature.PROTOCOL,'bootstrap_replicates':5000,'bootstrap_seed':origin.old.common.old.retained.BOOTSTRAP_SEED,
        'same_pool_historical_exposure':True,'new_external_validation':False,'formal_model_use':False,
        'previous_adoption_decisions_unchanged':True,'protected_hashes':protected})
    specs={k:{'config':f['config'],'features':f['names'],'threshold':float(f['threshold']),
        'median':f['median'].tolist(),'low':f['low'].tolist(),'high':f['high'].tolist(),'mean':f['mean'].tolist(),'scale':f['scale'].tolist(),
        'coefficients':f['model'].coef_[0].tolist() if f['model'] is not None else [],
        'intercept':float(f['model'].intercept_[0]) if f['model'] is not None else None} for k,f in final['models'].items()}
    save(OUT/'model_freeze_record.json',{'model_sha256':digest(source),'before_feature_generation_and_scoring':True,
        'primary_model':PRIMARY,'model_specifications':specs})
    return cohort,tr,te,omitted,all_test,final


def extract18(cohort,omitted,metadata):
    start=time.time();profiles=origin.old.common.old.retained.load_profiles();ids=cohort.AnalysisID.to_numpy()
    radiomics.logger.setLevel(40);sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    cache=origin.old.feature.local.extract(profiles,omitted,ids)
    baseline=origin.old.feature.extract(profiles,omitted,ids,cache)
    assert baseline['metadata']==metadata
    data={'ZScore':{k:baseline[k] for k in ['base','erode','dilate','indices','metadata']}}
    for key,name in [('IVIM_quality_audit','new18_IVIM_quality_audit.csv'),('IVIM_extraction_audit','new18_IVIM_baseline_extraction_audit.csv')]:
        pd.DataFrame(baseline[key]).to_csv(OUT/name,index=False,encoding='utf-8-sig')
    results={};tasks=[(int(i),ids[int(i)],profiles[int(i)]) for i in omitted]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(origin.feature.extract_one,t):t[0] for t in tasks}
        for done,f in enumerate(as_completed(futures),1):
            i=futures[f];results[i]=f.result()
            if done%3==0 or done==len(omitted):print(f'Frozen54 new18 normalization extraction {done}/{len(omitted)}, {time.time()-start:.0f}s',flush=True)
    names=[m['name'] for m in metadata]
    for method in origin.NORMALIZATIONS[1:]:
        data[method]={'indices':omitted,'metadata':metadata,**{kind:np.asarray([[results[int(i)]['values'][method][kind][n] for n in names] for i in omitted],float)
                         for kind in ['base','erode','dilate']}}
        quantitative=[j for j,m in enumerate(metadata) if m['modality'] in ['ADC','IVIM'] and m['block']=='Descriptor']
        if method=='RobustZ':
            for kind in ['base','erode','dilate']:np.testing.assert_allclose(data[method][kind][:,quantitative],data['ZScore'][kind][:,quantitative],atol=1e-12,rtol=0,equal_nan=True)
    for name in ['normalization_audit','partition_audit','texture_audit']:
        pd.DataFrame([r for i in omitted for r in results[int(i)][name]]).to_csv(OUT/('new18_'+name+'.csv'),index=False,encoding='utf-8-sig')
    joblib.dump(data,OUT/'new18_features.joblib',compress=3)
    return data


def prepare54(cohort,te,omitted,all_test,final):
    new=extract18(cohort,omitted,final['metadata']);previous=joblib.load(origin.OUT/'test_features.joblib');data={}
    for n in origin.NORMALIZATIONS:
        np.testing.assert_array_equal(previous[n]['indices'],te);np.testing.assert_array_equal(new[n]['indices'],omitted)
        assert previous[n]['metadata']==new[n]['metadata']==final['metadata']
        data[n]={'indices':all_test,'metadata':final['metadata'],**{k:np.vstack([previous[n][k],new[n][k]]) for k in ['base','erode','dilate']}}
    joblib.dump(data,OUT/'frozen_inference_inputs54.joblib',compress=3)
    return data


def score(final,data):
    values={}
    for k,f in final['models'].items():
        if not k.startswith('Adaptive__'):values[k]=origin.old.predict(f,data[f['config']['normalization']])
    for k,f in final['models'].items():
        if k.startswith('Adaptive__'):values[k]=values[origin.key(f['config']['normalization'],f['config']['analysis'])]
    return values


def main():
    start=time.time();cohort,tr,te,omitted,all_test,final=freeze();data=prepare54(cohort,te,omitted,all_test,final)
    protect();save(OUT/'evaluation_started.json',{'joint_count':1,'new_cases':18,'reproduced_cases':36,'no_refit':True})
    probabilities=score(final,data);old=pd.read_csv(origin.OUT/'test_predictions.csv',float_precision='round_trip');rows=[];maximum=0.
    for k,p in probabilities.items():
        reference=old[old.ModelKey==k].set_index('AnalysisID').loc[cohort.AnalysisID.iloc[te]].Probability.to_numpy()
        error=float(np.max(np.abs(p[:36]-reference)));assert error<1e-12;maximum=max(maximum,error)
        f=final['models'][k];rows.extend({'ModelKey':k,'AnalysisID':cohort.AnalysisID.iloc[i],
            'Group':'PreviousTest36' if j<36 else 'PreviouslyOmitted18','Outcome':int(cohort.Outcome.iloc[i]),'Probability':float(pp),
            'Threshold':f['threshold'],'PredictedPositive':int(pp>=f['threshold'])} for j,(i,pp) in enumerate(zip(all_test,p)))
    pd.DataFrame(rows).to_csv(OUT/'predictions54.csv',index=False,encoding='utf-8-sig')
    metrics=[];pairs=[]
    for g,positions in GROUPS.items():
        y=cohort.Outcome.to_numpy(int)[all_test[positions]];ps={k:p[positions] for k,p in probabilities.items()}
        draws=origin.old.common.old.retained.bootstrap(y,ps)
        for k,p in ps.items():
            f=final['models'][k];lo,hi=np.quantile(draws[k],[.025,.975]);difference=np.subtract.outer(p[y==1],p[y==0])
            result=origin.old.common.old.metrics(y,p,f['threshold']);assert abs(result['AUC']-np.mean((difference>0)+.5*(difference==0)))<1e-12
            metrics.append({'Group':g,'ModelKey':k,'Normalization':f['config']['normalization'],'Analysis':f['config']['analysis'],
                'N':len(y),'Positive':int(y.sum()),'Negative':int((y==0).sum()),**result,'AUC_CI95Low':float(lo),'AUC_CI95High':float(hi)})
        for a in origin.GROUPS:
            for n in ['RobustZ','PercentileZ','Adaptive']:
                ka,kb=origin.key(n,a),origin.key('ZScore',a);lo,hi=np.quantile(draws[ka]-draws[kb],[.025,.975])
                pairs.append({'Group':g,'Analysis':a,'Method':n,'Comparison':ka+' minus '+kb,'DeltaAUC':float(result_auc(ps[ka],y)-result_auc(ps[kb],y)),
                              'CI95Low':float(lo),'CI95High':float(hi)})
    results=pd.DataFrame(metrics);results.to_csv(OUT/'metrics54.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(pairs).to_csv(OUT/'paired_auc_comparisons.csv',index=False,encoding='utf-8-sig')
    results[results.ModelKey==PRIMARY].to_csv(OUT/'primary_model_metrics54.csv',index=False,encoding='utf-8-sig')
    save(OUT/'evaluation_complete.json',{'passed':True,'models_refitted':0,'thresholds_changed':False,'training_overlap':0,
        'test_patients':54,'new_patients':18,'prediction_rows':864,'distinct_model_predictions':648,'adaptive_references':216,
        'old36_prediction_max_error':maximum,'joint_evaluation_count':1,'protected_count':protect(),'formal_model_use':False,
        'elapsed_seconds':time.time()-start})
    print(results[results.ModelKey==PRIMARY].to_string(index=False),flush=True)


def result_auc(p,y):
    d=np.subtract.outer(p[y==1],p[y==0]);return float(np.mean((d>0)+.5*(d==0)))


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
