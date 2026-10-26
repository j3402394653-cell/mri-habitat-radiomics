# -*- coding: utf-8 -*-
"""Prespecified 1/2/3-mm pure-habitat comparison on the same A84/A36."""
import os
os.environ.update({'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from threadpoolctl import threadpool_limits
import habitat_margin_features_20261024 as feature
import run_pure_habitat_consistent_stack_20261024 as old
import run_random120_retained_methods_20261010 as loader
import pure_habitat_robust_nonlinear_20261023 as backend

ROOT, save, digest = old.ROOT, old.save, old.digest
OUT = ROOT/'lvi_prediction/results/lvi_habitat_margin_20261024'
LATEST = ROOT/'lvi_prediction/results/lvi_A2_meta72_frozen_test_20261024'
MODES = ('Base','Erode','Dilate','ContourMedian')
ROUTES = tuple((r,m) for r in (1.,2.,3.) for m in MODES)
CONFIG = {'candidate_id':4,'base_mode':'Bag15','anchor':.25,'nonnegative':False,'C':.03}
QC = {}


def key(radius, mode): return 'R%d_%s'%(radius,mode)


def csv(name, rows):
    pd.DataFrame(rows).to_csv(OUT/(name+'.csv'), index=False, encoding='utf-8-sig')


def read(name): return pd.read_csv(OUT/(name+'.csv'),float_precision='round_trip')


def protect(images=False):
    p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    for field in ('protected_hashes','source_hashes'):
        for name,sha in p[field].items(): assert digest(ROOT/name)==sha,name
    if images:
        for name,sha in p['image_hashes'].items(): assert digest(ROOT/name)==sha,name
    return len(p['protected_hashes'])


def define():
    assert not OUT.exists()
    assert json.loads((LATEST/'delivery_final_verification.json').read_text(encoding='utf-8'))['passed']
    previous=json.loads((LATEST/'protocol.json').read_text(encoding='utf-8'))
    protected=dict(previous['protected_hashes'])
    protected.update({str(p.relative_to(ROOT)):digest(p) for p in LATEST.rglob('*') if p.is_file()})
    old.frozen()
    c,y,tr,te=old.inputs()
    entries=json.loads((loader.POOL/'source_manifest.json').read_text(encoding='utf-8'))['entries']
    assert [e['AnalysisID'] for e in entries]==c.AnalysisID.tolist()
    sources=[ROOT/'lvi_prediction'/n for n in ('run_habitat_margin_20261024.py','habitat_margin_features_20261024.py')]
    sources += [loader.POOL/'source_manifest.json',ROOT/'lvi_prediction/LVI_RESULT_SELECTION.json',ROOT/'lvi_prediction/MODEL_DECISION.md']
    images={}
    for i in np.r_[tr,te]:
        for name in entries[int(i)]['source_override'].values():
            path=ROOT/name if not os.path.isabs(name) else loader.Path(name)
            images[str(path.relative_to(ROOT))]=digest(path)
    OUT.mkdir(parents=True)
    for name in ('patients','models','contexts','figures'):(OUT/name).mkdir()
    for label,ix in (('Train',tr),('Test',te)):c.iloc[ix].to_csv(OUT/(label+'_cases.csv'),index=False,encoding='utf-8-sig')
    p={'date':'2026-10-24','authorization':'Latest human instruction:1mm control,add2mm and3mm;cancel0.5and1.5mm.',
        'cohort':'ExactA120:84train57positive27negative;36test25positive11negative;no patient draw or exclusions.',
        'pure_habitat':True,'K':3,'modalities':['T2','ADC','IVIM_D_f'],'Dstar':False,'clinical':False,'whole_predictors':False,
        'radius_mm':[1.,2.,3.],'prespecified_routes':[key(r,m) for r,m in ROUTES],
        'primary_comparisons':'R2_Base andR3_Base versusR1_Base:changeICC perturbationdistance;all shareexactnewlyreconstructed originalROI featuresonacommonpadded1mmgrid.',
        'secondary':'Erode,Dilate,ContourMedian atall3distances:actual inference inputchanges. QC alwaysoriginal/erode/dilate tripleforcurrentdistance.',
        'geometry':'Euclideanphysicaldistance on1mm grid;erosioninside>r,dilationoutside<=r;computedfromgeometrictumormask beforeIVIM qualityholes;qualityintersectedaftergeometry.',
        'expanded_support':'Re-readnativeT2/ADC/D/f/NRMSEandmasks;padexistingphysicalgrid4mmeachside;notexpandtheold1mm-truncatedvalidcache. IVIM usesnativeD>0,0<=f<.7,NRMSE<=.15,valid-weightedlinear resampling;noDstar.',
        'normalization':'Samepatientbasevalidtumorreference;fixedacrossradii;PercentileZandRobustZbothin12memberstack.',
        'features':'Same321compactregionalfeatures;17nonshapetexturesperhabitat/channelplusoriginalquantitativeandspatialdescriptors;three6views321/321/220;no fixedtop3or6cap.',
        'QC':'Currentbase-training-only completeness.90,absoluteICC.75,Spearman.95,medianfill,GaussianQTclipped[-2.5,2.5].',
        'parameters':'FreezeoriginalA2finalrecipeforallroutes:Bag15,anchor.25,balancedL2LogisticC.03;12PLS1/2memberswithC.03. Doesnotreuse newattempt1allpositive-thresholdmodel.',
        'validation':'Sameouter5andtraininginner5x3. InnerfoldsproduceOOFbase scores;meta fittedontrainingOOFplusactualtraininganchor;predictoutervalidationwith15baggedbases. No model/radius/mode hyperparameter searchinthiscontrolledcomparison.',
        'interpretation':'Prespecifiedfixedrecipecomparators;notadaptive8originalprocedure. Recomputeall1/2/3mmfeaturesoncommonpaddedgrid. Old1mmcroppedgridcanaffectEDTboundary/depthdescriptors;reportoldA2separately,notattributecommonpaddingcontrolchangetoradius. Multiplefixedrouteoutercomparisonsareexploratory;no claimofunbiasedperformanceofaposthocbest.',
        'threshold':'Eachroutefullpipeline84outerOOFYouden;frozenbeforeanynew36prediction. No testthreshold tuning.',
        'test':'Freezeandverifyall12thenonejoint36stage;all12reported,nofilteringbytestAUC;no trainingsetoverlap.',
        'prior_exposure':'SameA36historicallyreused;exploratoryinternalcomparison,notexternalvalidation.',
        'formal_adoption_changed':False,'source_hashes':{str(p.relative_to(ROOT)):digest(p) for p in sources},
        'protected_hashes':protected,'image_hashes':images}
    save(OUT/'protocol.json',p)
    save(OUT/'definition_frozen.json',{'passed':True,'protocol_sha256':digest(OUT/'protocol.json'),'new_test_calls':0})
    print('Defined12fixedrecipe1/2/3mmroutes;sameA84/36;testunscored',flush=True)


def extraction_worker(task):
    with threadpool_limits(1):
        result=feature.extract_one(task)
        joblib.dump(result,OUT/'patients'/('%03d.joblib'%result['index']),compress=3)
    return result['index']


def extraction(stage, workers=4):
    protect(images=True);c,y,tr,te=old.inputs();ix=tr if stage=='training' else te
    if stage=='test':
        frozen(); assert json.loads((OUT/'training_verification.json').read_text(encoding='utf-8'))['passed']
        assert not (OUT/'test_started.json').exists()
    profiles=loader.load_profiles()
    entries=json.loads((loader.POOL/'source_manifest.json').read_text(encoding='utf-8'))['entries']
    tasks=[(int(i),c.AnalysisID.iloc[i],entries[int(i)],profiles[int(i)],i in (ix[0],ix[-1])) for i in ix if not (OUT/'patients'/('%03d.joblib'%i)).exists()]
    start=time.time()
    failures=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        fs={pool.submit(extraction_worker,t):t[0] for t in tasks}
        for done,ff in enumerate(as_completed(fs),1):
            try:ff.result()
            except Exception as exc:
                failures.append({'index':int(fs[ff]),'error':str(exc)})
                print('Case %d extraction failed: %s'%(fs[ff],exc),flush=True)
            if done%4==0 or done==len(tasks):print('%s native2/3mm extraction%d/%d;%.0fs'%(stage,done,len(tasks),time.time()-start),flush=True)
    if failures:
        save(OUT/(stage+'_extraction_failures.json'),{'failures':failures,'patient_exclusion':False,'test_calls':0})
        raise RuntimeError('Extractionfailurespreserved;donotexcludepatients:'+str(failures))
    raw=old.previous.parent.cache.cached_features(ix);compact={}
    for n,d in raw.items():
        selected,_=old.previous.parent.views.specifications(d['metadata'])
        compact[n]={'indices':ix,'metadata':[d['metadata'][j] for j in selected],**{k:d[k][:,selected] for k in ('base','erode','dilate')}}
    allrows=[joblib.load(OUT/'patients'/('%03d.joblib'%i)) for i in ix]
    maximum=0.;reference_count=0
    for pos,row in enumerate(allrows):
        assert row['index']==ix[pos] and row['reconstructed_old_profile_verified']
        lookup={m['name']:j for j,m in enumerate(row['metadata'])}
        reorder=[lookup[m['name']] for m in compact['RobustZ']['metadata']]
        assert [row['metadata'][j] for j in reorder]==compact['RobustZ']['metadata']
        row['values']={k:np.asarray(v)[reorder] for k,v in row['values'].items()}
        row['base_values']={k:np.asarray(v)[reorder] for k,v in row['base_values'].items()}
        row['references']={k:np.asarray(v)[reorder] for k,v in row['references'].items()}
        for (n,k),v in row['references'].items():
            # Reference computed ontheoriginalcroppedarrays tokeepnumericpartitionpathunchanged.
            np.testing.assert_allclose(v,compact[n][k][pos],rtol=1e-8,atol=1e-8,equal_nan=True)
            finite=np.isfinite(v)&np.isfinite(compact[n][k][pos])
            maximum=max(maximum,float(np.max(np.abs(np.asarray(v)[finite]-compact[n][k][pos,finite]))));reference_count+=1
    result={};differences=[]
    for r in (1.,2.,3.):
        d={n:{'indices':ix,'metadata':compact[n]['metadata'],'base':np.vstack([row['base_values'][n] for row in allrows]),
            **{k:np.vstack([row['values'][(n,r,k)] for row in allrows]) for k in ('erode','dilate')}} for n in feature.NORMS}
        if r==1.:
            for n in feature.NORMS:
                for k in ('base','erode','dilate'):
                    for j,m in enumerate(compact[n]['metadata']):
                        same=np.isclose(d[n][k][:,j],compact[n][k][:,j],rtol=1e-8,atol=1e-8,equal_nan=True)
                        if not same.all():differences.append({'Normalization':n,'Variant':k,'Feature':m['name'],'ChangedPatients':int((~same).sum()),'SpatialBoundaryDepth':bool(m['name'].endswith(('_BoundaryEnrichment','_NormalizedDepth')))})
        result[r]=old.latent.feature.build(d)
    csv(stage+'_padded1mm_vs_historical_feature_changes',differences)
    joblib.dump(result,OUT/(stage+'_features.joblib'),compress=3)
    for kind in ('geometry_audit','texture_audit','normalization_audit'):csv(stage+'_'+kind,[r for row in allrows for r in row[kind]])
    save(OUT/(stage+'_feature_verification.json'),{'passed':True,'patients':len(ix),'reference_vectors':reference_count,
        'maximum_reference_difference':maximum,'all_old_profiles_verified':True,'all_radii_share_same_original_roi_feature_values':True,
        'padded_reference_no_target_on_edge':True,'source_images_unchanged':True})
    print(stage+' featurecacheverified',flush=True)


def route_tables(tables, mode):
    result={}
    for name,d in tables.items():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            x=d['base'] if mode=='Base' else d['erode'] if mode=='Erode' else d['dilate'] if mode=='Dilate' else np.nanmedian(np.stack([d[k] for k in ('base','erode','dilate')]),axis=0)
        result[name]={**d,'base':x}
    return result


def fit12(original,tables,a,b,y,r):
    prepared={};models=[]
    for c in old.previous.members():
        name=c['image_key'];ck=(r,name,tuple(a))
        if ck not in QC:QC[ck]=backend.quality(original[name],a)
        if name not in prepared:prepared[name]=backend.prepare(tables[name],a,b,{'mask_mode':'Base','icc_min':.75},QC[ck])
        f=old.latent.fitted(prepared[name],y[a],c)
        # Emptychannelsremain explicitconstants,neverpatientexclusions.
        f.pop('audit',None);models.append(f)
    return models


def phase_fit(original,a,b,y,phase,r,mode):
    path=OUT/'models'/(key(r,mode)+'_'+phase+'.joblib')
    if path.exists():return joblib.load(path)
    tables=route_tables(original,mode);batch=np.r_[a,b];lookup={int(i):j for j,i in enumerate(a)}
    repeat=np.full((len(a),3,12),np.nan);bag=np.zeros((len(batch),12));kept=[]
    for rep,k,aa,bb in old.previous.parent.folds('Random120',phase):
        assert set(aa).isdisjoint(bb) and set(aa)|set(bb)==set(a)
        models=fit12(original,tables,aa,batch,y,r)
        score=np.column_stack([f['evaluation_probability'] for f in models])
        rr=[lookup[int(i)] for i in bb];repeat[rr,rep-1]=score[rr];bag+=score/15.
        kept.append({'Repeat':rep,'Fold':k,'train':aa,'validation':bb,'models':models})
    assert np.isfinite(repeat).all()
    ctx={'train':a,'validation':b,'oof':repeat.mean(1),'bag_train':bag[:len(a)],'bag_validation':bag[len(a):]}
    f=old.meta_fit(ctx,y,CONFIG.copy());f.update(radius_mm=r,mask_mode=mode,phase=phase)
    cp={'fit':f,'bases':{'bag':kept,'full':[],'train':a,'validation':b},'context':ctx}
    joblib.dump(cp,path,compress=3)
    print(key(r,mode)+' '+phase+' trained;notest',flush=True)
    return cp


def train():
    protect();assert not (OUT/'training_frozen.json').exists() and not (OUT/'test_started.json').exists()
    assert json.loads((OUT/'training_feature_verification.json').read_text(encoding='utf-8'))['passed']
    c,y,tr,te=old.inputs();tables=joblib.load(OUT/'training_features.joblib');metrics=[];predictions=[];summary=[]
    for phase in old.PHASES:
        a,b=(tr,np.empty(0,int)) if phase=='final' else old.previous.parent.folds('Random120')[int(phase[-1])-1]
        for r,mode in ROUTES:
            cp=phase_fit(tables[r],a,b,y,phase,r,mode);f=cp['fit']
            if phase!='final':
                metrics.append({'ModelKey':key(r,mode),'RadiusMM':r,'Mode':mode,'Fold':int(phase[-1]),
                    'FitAUC':roc_auc_score(y[a],f['train_probability']),'AUC':roc_auc_score(y[b],f['evaluation_probability'])})
                predictions.extend({'ModelKey':key(r,mode),'Fold':int(phase[-1]),'Index':int(i),'AnalysisID':c.AnalysisID.iloc[i],'Outcome':int(y[i]),'Probability':float(p)} for i,p in zip(b,f['evaluation_probability']))
    mm=pd.DataFrame(metrics);pp=pd.DataFrame(predictions)
    for r,mode in ROUTES:
        name=key(r,mode);cp=joblib.load(OUT/'models'/(name+'_final.joblib'));f=cp['fit']
        pv=pp[pp.ModelKey==name].set_index('Index').loc[tr];rr=mm[mm.ModelKey==name]
        fp,tp,thresholds=roc_curve(y[tr],pv.Probability,drop_intermediate=False)
        f['threshold']=float(thresholds[np.argmax(tp-fp)]);f['threshold_source']='all84_fullpipeline_outerOOF_Youden'
        joblib.dump(cp,OUT/'models'/(name+'_final.joblib'),compress=3)
        summary.append({'ModelKey':name,'RadiusMM':r,'Mode':mode,'TrainFitAUC':roc_auc_score(y[tr],f['train_probability']),
            'OuterMeanAUC':rr.AUC.mean(),'OuterSD':rr.AUC.std(ddof=1),'PooledOOFAUC':roc_auc_score(y[tr],pv.Probability),'Threshold':f['threshold']})
    csv('training_summary',summary);csv('outer_metrics',metrics);csv('nested_predictions',predictions)
    files=[p for p in OUT.rglob('*') if p.is_file() and p.parent.name!='figures']
    save(OUT/'training_frozen.json',{'passed':True,'new_test_calls':0,'files':{str(p.relative_to(OUT)):digest(p) for p in files}})
    print(pd.DataFrame(summary).to_string(index=False),flush=True)


def frozen():
    protect();f=json.loads((OUT/'training_frozen.json').read_text(encoding='utf-8'))
    for name,sha in f['files'].items():assert digest(OUT/name)==sha,name
    return f


def infer(cp,tables,ix):
    f=cp['fit'];d=route_tables(tables[f['radius_mm']],f['mask_mode'])
    # Standardlatent.predict also handles unavailableconstantbasechannels.
    x=sum(np.column_stack([old.latent.predict(m,d,ix) for m in row['models']]) for row in cp['bases']['bag'])/15.
    return expit(((np.clip(x,f['low'],f['high'])-f['center'])/f['scale'])@f['coef']+f['intercept'])


def formula(cp,tables,ix):
    f=cp['fit'];d=route_tables(tables[f['radius_mm']],f['mask_mode']);bag=[]
    for row in cp['bases']['bag']:
        columns=[]
        for m in row['models']:
            if m['classifier'] is None:columns.append(np.full(len(ix),m['prevalence']));continue
            dd=d[m['config']['image_key']];x=dd['base'][old.latent.rows(dd,ix)][:,m['indices']]
            x=np.where(np.isfinite(x),x,m['median']);x=np.clip(m['quantile'].transform(x),-2.5,2.5)
            columns.append(expit(x@m['coef_gaussian']+m['intercept_gaussian']))
        bag.append(np.column_stack(columns))
    x=np.mean(bag,axis=0);z=(np.minimum(np.maximum(x,f['low']),f['high'])-f['center'])/f['scale']
    return 1./(1.+np.exp(-(z@f['coef']+f['intercept'])))


def verify_training():
    frozen();assert not (OUT/'test_started.json').exists()
    c,y,tr,te=old.inputs();tables=joblib.load(OUT/'training_features.joblib');err=0.;count=0
    predictions=read('nested_predictions');metrics=read('outer_metrics')
    for phase in old.PHASES:
        a,b=(tr,np.empty(0,int)) if phase=='final' else old.previous.parent.folds('Random120')[int(phase[-1])-1]
        for r,mode in ROUTES:
            cp=joblib.load(OUT/'models'/(key(r,mode)+'_'+phase+'.joblib'));f=cp['fit']
            np.testing.assert_array_equal(f['train'],a);np.testing.assert_array_equal(f['validation'],b)
            for row,(rep,k,aa,bb) in zip(cp['bases']['bag'],old.previous.parent.folds('Random120',phase)):
                np.testing.assert_array_equal(row['train'],aa);np.testing.assert_array_equal(row['validation'],bb)
                assert set(aa).isdisjoint(bb) and not (set(aa)|set(bb))&(set(b)|set(te));count+=1
            pa=infer(cp,tables,a);err=max(err,float(np.max(np.abs(pa-f['train_probability']))))
            np.testing.assert_allclose(pa,f['train_probability'],atol=1e-9,rtol=0)
            np.testing.assert_allclose(formula(cp,tables,a),pa,atol=1e-9,rtol=0)
            if len(b):
                pb=infer(cp,tables,b);np.testing.assert_allclose(pb,f['evaluation_probability'],atol=1e-9,rtol=0)
                pv=predictions[(predictions.ModelKey==key(r,mode))&(predictions.Fold==int(phase[-1]))].set_index('Index').loc[b]
                np.testing.assert_allclose(pb,pv.Probability,atol=1e-9,rtol=0)
    for r,mode in ROUTES:
        name=key(r,mode);s=read('training_summary').set_index('ModelKey').loc[name];rr=metrics[metrics.ModelKey==name]
        np.testing.assert_allclose([s.OuterMeanAUC,s.OuterSD],[rr.AUC.mean(),rr.AUC.std(ddof=1)],atol=1e-12)
    # Heldoutvaluescannotchange trainingQC,medians,transformsorPLS/LRfit.
    a,b=old.previous.parent.folds('Random120')[0];checks=0
    for r in (1.,2.,3.):
        d=tables[r]['PercentileZ__OrderedCompact'];changed={**d,**{k:d[k].copy() for k in ('base','erode','dilate')}}
        rows=old.latent.rows(d,b)
        for k in ('base','erode','dilate'):changed[k][rows]=1e9
        for mode in ('Base','Erode','Dilate','ContourMedian'):
            first=route_tables({'x':d},mode)['x'];second=route_tables({'x':changed},mode)['x']
            p=backend.prepare(first,a,b,{'mask_mode':'Base','icc_min':.75},backend.quality(d,a))
            q=backend.prepare(second,a,b,{'mask_mode':'Base','icc_min':.75},backend.quality(changed,a))
            assert p['indices']==q['indices'];np.testing.assert_array_equal(p['xa'],q['xa']);checks+=1
    save(OUT/'training_verification.json',{'passed':True,'fold_bags_audited':count,'maximum_frozen_inference_error':err,
        'all_radii_common_padded1mm_grid':True,'heldout_mutation_checks':checks,'new_test_calls':0,'all5folds_retained':True})
    print('Trainingverified;all12frozen;commonpaddedgrid1mmcontrol',flush=True)


def test():
    frozen();assert not (OUT/'test_started.json').exists()
    for n in ('training_verification','test_feature_verification'):assert json.loads((OUT/(n+'.json')).read_text(encoding='utf-8'))['passed']
    c,_,tr,te=old.inputs();truth=c.Outcome.to_numpy(int)[te];tables=joblib.load(OUT/'test_features.joblib')
    from verify_ivim_df63_20261020 import auc,draws
    save(OUT/'test_started.json',{'joint_test_stages':1,'prespecified_model_count':12,'post_test_fits':0})
    scores={};predictions=[];metrics=[];maximum_error=0.
    for r,mode in ROUTES:
        name=key(r,mode);cp=joblib.load(OUT/'models'/(name+'_final.joblib'));f=cp['fit'];p=infer(cp,tables,te)
        assert np.isfinite(p).all();scores[name]=p
        manual=formula(cp,tables,te);np.testing.assert_allclose(manual,p,atol=1e-9,rtol=0)
        maximum_error=max(maximum_error,float(np.max(np.abs(manual-p))))
        np.testing.assert_allclose(auc(truth,p),roc_auc_score(truth,p),atol=1e-12,rtol=0)
        tn,fp,fn,tp=confusion_matrix(truth,p>=f['threshold'],labels=[0,1]).ravel()
        metrics.append({'ModelKey':name,'RadiusMM':r,'Mode':mode,'AUC':roc_auc_score(truth,p),'Threshold':f['threshold'],
            'Sensitivity':tp/(tp+fn),'Specificity':tn/(tn+fp),'Accuracy':(tp+tn)/len(te),'TP':int(tp),'FN':int(fn),'TN':int(tn),'FP':int(fp)})
        predictions.extend({'ModelKey':name,'Index':int(i),'AnalysisID':c.AnalysisID.iloc[i],'Outcome':int(t),'Probability':float(v),'Prediction':int(v>=f['threshold'])} for i,t,v in zip(te,truth,p))
    oldp=old.read('test_predictions').set_index('Index').loc[te].Probability.to_numpy()
    samples=draws(truth,scores,20261024103)
    for row in metrics:
        n=row['ModelKey'];lo,hi=np.quantile(samples[n],[.025,.975]);dl,dh=np.quantile(samples[n]-samples['R1_Base'],[.025,.975])
        row.update(AUC_CI95Low=float(lo),AUC_CI95High=float(hi),DifferenceVs1mmBase=float(row['AUC']-roc_auc_score(truth,scores['R1_Base'])),
            PairedDifferenceCI95Low=float(dl),PairedDifferenceCI95High=float(dh))
    csv('test_metrics',metrics);csv('test_predictions',predictions)
    comparison=read('training_summary').merge(pd.DataFrame(metrics).rename(columns={'AUC':'TestAUC'}),on=['ModelKey','RadiusMM','Mode'],suffixes=('','_test'))
    comparison.to_csv(OUT/'comparison.csv',index=False,encoding='utf-8-sig')
    frozen()
    save(OUT/'test_inference_verification.json',{'passed':True,'independent_formula_probability_checks':432,
        'maximum_formula_error':maximum_error,'rank_AUC_verified':True,'training_hashes_unchanged':True})
    save(OUT/'test_complete.json',{'passed':True,'joint_test_stages':1,'prespecified_models':12,'predictions':432,
        'all_radii_compared_with_common_padded1mm_control':True,'post_test_fits':0,'post_test_threshold_changes':0,'formal_adoption_changed':False})
    print(comparison[['ModelKey','TrainFitAUC','OuterMeanAUC','OuterSD','TestAUC','Sensitivity','Specificity']].to_string(index=False),flush=True)


def report():
    frozen();assert json.loads((OUT/'test_complete.json').read_text(encoding='utf-8'))['passed']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d=read('comparison');mm=read('outer_metrics');p=read('test_predictions')
    for name,g in p.groupby('ModelKey'):
        row=d.set_index('ModelKey').loc[name];tn,fp,fn,tp=confusion_matrix(g.Outcome,g.Prediction,labels=[0,1]).ravel()
        np.testing.assert_allclose(row.TestAUC,roc_auc_score(g.Outcome,g.Probability),atol=1e-12)
        assert (tn,fp,fn,tp)==(row.TN,row.FP,row.FN,row.TP)
    fig,axes=plt.subplots(1,2,figsize=(11,4.6))
    colors=['#707b84','#237ca7','#d68b31']
    for r,color in zip((1,2,3),colors):
        z=mm[mm.ModelKey==key(r,'Base')];axes[0].plot(z.Fold,z.AUC,'o-',label='%d mm'%r,color=color)
        z=p[p.ModelKey==key(r,'Base')];x,y,_=roc_curve(z.Outcome,z.Probability)
        axes[1].plot(x,y,label='%d mm AUC %.4f'%(r,roc_auc_score(z.Outcome,z.Probability)),color=color)
    axes[0].set(xlabel='Outer fold',ylabel='AUC',ylim=(0,1.02),xticks=range(1,6),title='Base input / changed contour QC')
    axes[1].plot([0,1],[0,1],'--',color='gray',linewidth=1)
    axes[1].set(xlabel='1 - specificity',ylabel='Sensitivity',title='Same A36 / frozen models')
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.15)
    fig.tight_layout();fig.savefig(OUT/'figures/margin_primary_comparison.png',dpi=175);fig.savefig(OUT/'figures/margin_primary_comparison.svg');plt.close(fig)
    labels={'Base':'原轮廓输入＋该距离稳定性筛选','Erode':'内缩轮廓输入','Dilate':'外扩轮廓输入','ContourMedian':'原/内缩/外扩特征中位数'}
    lines=['# 1、2、3 mm内缩外扩距离：纯生境模型对照报告','','2026年10月24日。按用户最终决定，以1 mm为对照，新增2和3 mm；未测试0.5或1.5 mm。A组原120例及84/36划分、外层5折、内层5折×3次全部保持不变。无病例筛选，无低折剔除。', '',
        '## 对照设计','','为了把距离影响与模型参数影响分开，本轮12组均固定原A组②最终配方：15袋集成、锚定权重0.25、逻辑回归C=0.03，12个PLS1/2基础成员，T2、ADC、IVIM D/f，两种标准化按原方式融合。未沿用上一轮特异度为0的新尝试一。', '',
        '本轮是固定配方的距离对照，内层5折×3次用于产生基础模型OOF评分，并非再次执行原Adaptive8的8候选选参。历史原A组②训练拟合0.7680、Adaptive8外层均值0.5661、测试0.8218；固定其最终配方的历史外层诊断为0.5927。本轮1 mm对照也重新提取，以使1/2/3 mm共同使用完整有效域和相同网格边缘处理。旧裁剪网格会影响部分边界富集/归一化深度特征，因此本轮1 mm与历史1 mm的差异不能归因为距离改变。', '',
        '主要比较是原轮廓输入不变、只改变ICC扰动距离的1/2/3 mm三组。另报告内缩、外扩及三轮廓特征中位数作为次要对照。12组全部预先定义、冻结后一次联合测试，未按测试结果筛选或再训练。', '',
        '旧缓存的有效掩膜被限于约1 mm支持域；本轮重新读取原始影像和D/f/NRMSE，按原物理1 mm网格四周延伸4 mm，并恢复原生质量约束。内缩、外扩均按原始几何肿瘤轮廓的欧氏距离计算，再与有效质量区域相交；IVIM坏拟合孔洞不作为额外肿瘤边界。标准化始终使用同一原始有效肿瘤。新边界没有触及扩展采样网格边缘。', '',
        '所有1/2/3 mm均重新聚类与提取相同321项精简区域特征、再构建321/321/220三个视图；各距离组原轮廓特征完全相同。另在原裁剪数组上重提取核对样本，确认特征公式可复现旧缓存。ICC、完整率、去相关、填补、分位数变换和PLS均在当前基础训练折内拟合，不加入临床、全肿瘤直接预测因子或D*。', '',
        '## 全部结果','','| 距离 | 输入方式 | 真实训练拟合AUC | 完整外层五折均值±SD | 测试AUC | 敏感度 | 特异度 | 准确率 | TP/FN/TN/FP |','|---|---|---:|---:|---:|---:|---:|---:|---|']
    for row in d.itertuples():
        lines.append('| %g mm | %s | %.4f | %.4f±%.4f | %.4f | %.3f | %.3f | %.3f | %d/%d/%d/%d |'%(row.RadiusMM,labels[row.Mode],row.TrainFitAUC,row.OuterMeanAUC,row.OuterSD,row.TestAUC,row.Sensitivity,row.Specificity,row.Accuracy,row.TP,row.FN,row.TN,row.FP))
    lines += ['', '## 主要距离对照的同病例不确定性', '', '| 组别 | 测试AUC的95%区间 | 相对1 mm原轮廓AUC差值 | 配对差值95%区间 |','|---|---|---:|---|']
    for row in d[d.Mode=='Base'].itertuples():lines.append('| %g mm | %.4f～%.4f | %+.4f | [%+.4f, %+.4f] |'%(row.RadiusMM,row.AUC_CI95Low,row.AUC_CI95High,row.DifferenceVs1mmBase,row.PairedDifferenceCI95Low,row.PairedDifferenceCI95High))
    lines += ['', '区间为5000次按阳性/阴性分层的同病例bootstrap。区间不包含重新训练和多轮模型探索的不确定性，12组比较未做多重比较校正，不把单组较高值视为已证明的总体增益。', '',
        '## 解释与使用范围','','每组阈值均由84例完整流程外层OOF的Youden指数确定，在36例测试前冻结。AUC评价排序，敏感度和特异度评价固定阈值的分类；不能从测试集重新选阈值来改善分类指标。', '',
        'A组36例此前已多轮用于探索，本报告是同批病例的内部对照，并非新的独立或外部验证。所有低折完整保留。所有历史模型、排除决定和正式采用状态不变。', '',
        '![主要距离对照图]('+(OUT/'figures/margin_primary_comparison.png').as_posix()+')', '',
        '逐折结果见outer_metrics.csv；逐例预测见test_predictions.csv；实际新增/删除体素及IVIM有效覆盖率见training_geometry_audit.csv和test_geometry_audit.csv。']
    (OUT/'内缩外扩1_2_3mm纯生境对照报告_20261024.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(OUT/'report_verification.json',{'passed':True,'models':12,'all_metrics_recomputed':True,'formal_adoption_changed':False})
    print('Reportgenerated;awaitvisualinspectionandseal',flush=True)


def seal():
    frozen();protect(images=True)
    for n in ('training_verification','test_complete','report_verification','visual_verification'):assert json.loads((OUT/(n+'.json')).read_text(encoding='utf-8'))['passed']
    files={str(p.relative_to(OUT)):digest(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ('delivery_manifest.json','delivery_final_verification.json')}
    save(OUT/'delivery_manifest.json',{'outputs':files,'source_andprior_files_unchanged':True})
    for name,sha in files.items():assert digest(OUT/name)==sha,name
    save(OUT/'delivery_final_verification.json',{'passed':True,'outputs_checked':len(files),'prespecified_fixed_models':12,
        'joint_test_stages':1,'post_test_fits':0,'post_test_threshold_changes':0,'formal_adoption_changed':False})
    print('Deliveredall12routes;historicalresultsunchanged',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('define','extract_training','train','verify_training','extract_test','test','report','seal'));p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    with threadpool_limits(1):
        if args.action.startswith('extract_'):extraction('training' if args.action=='extract_training' else 'test',args.workers)
        else:globals()[args.action]()
