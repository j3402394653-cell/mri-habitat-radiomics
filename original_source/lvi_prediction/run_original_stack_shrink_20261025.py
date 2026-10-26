# -*- coding: utf-8 -*-
"""Improve the retained 1-mm median PLS stack; no region concatenation/L1."""
import os
os.environ.update({'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import time
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score,roc_curve,confusion_matrix
from threadpoolctl import threadpool_limits
import run_habitat_margin_20261024 as base
import run_habitat_region_fusion_20261024 as previous

ROOT,save,digest=base.ROOT,base.save,base.digest
OUT=ROOT/'lvi_prediction/results/lvi_original_stack_shrink_20261025'
PHASES=base.old.PHASES


def csv(name,rows):pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')
def read(name):return pd.read_csv(OUT/(name+'.csv'),float_precision='round_trip')


def options():
    result=[{'candidate_id':0,'family':'Baseline','anchor':.25,'C':.03,'mix':0.,'dimensions':12}]
    start=len(result)
    result.extend({'candidate_id':start+i,'family':'Blend','anchor':.25,'C':.03,'mix':w,'dimensions':1 if w==1 else 12} for i,w in enumerate((.25,.5,.75,1.)))
    for anchor in (.25,1.):
        for C in (.01,.03,.1):result.append({'candidate_id':len(result),'family':'NN12','anchor':anchor,'C':C,'mix':0.,'dimensions':12})
    for family in ('Norm2','View3'):
        for C in (.01,.03,.1):result.append({'candidate_id':len(result),'family':family,'anchor':.25,'C':C,'mix':0.,'dimensions':2 if family=='Norm2' else 3})
    assert len(result)==17 and [r['candidate_id'] for r in result]==list(range(17))
    return result


def protect():
    p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    for field in ('protected_hashes','source_hashes'):
        for name,sha in p[field].items():assert digest(ROOT/name)==sha,name
    return len(p['protected_hashes'])


def define():
    assert not OUT.exists();previous.frozen();base.frozen()
    assert json.loads((previous.OUT/'delivery_final_verification.json').read_text(encoding='utf-8'))['passed']
    prior=json.loads((previous.OUT/'protocol.json').read_text(encoding='utf-8'))
    protected={**prior['protected_hashes'],**prior['source_hashes']}
    protected.update({str(p.relative_to(ROOT)):digest(p) for p in previous.OUT.rglob('*') if p.is_file()})
    OUT.mkdir(parents=True)
    for n in ('contexts','phases','models','figures'):(OUT/n).mkdir()
    for label in ('Train','Test'):(OUT/(label+'_cases.csv')).write_bytes((base.OUT/(label+'_cases.csv')).read_bytes())
    sources=[ROOT/'lvi_prediction'/n for n in ('run_original_stack_shrink_20261025.py','verify_original_stack_shrink_20261025.py','report_original_stack_shrink_20261025.py')]
    p={'date':'2026-10-25','authorization':'UserabandonednewA/Cregionmethods;improveonretainedoriginalpurehabitatfoundation. Latestdistanceconstraintonly1mm.',
        'baseline':'ExactR1_ContourMedianPLSstack:fit.7680311890838207,ALL5outer.6218181818181818,test.8218181818181818;keeporiginalROIstackasreference.',
        'cohort':'ExactsameA12084train57+/27-,36test25+/11-;sameouter5andinner5x3;no patient/foldselection.',
        'unchanged':'1mmoriginal/erode/dilatefeaturemedian;T2/ADC/IVIMD/f;k3;PercentileZ/RobustZ;321/321/220views;12PLS1/2basesC.03;ICC.75,complete.9,Spearman.95,trainingmedianfillandGaussianQT;Bag15inference.',
        'excluded_paths':'No newcore/periringfeatures,noL1,no2/3mm,noDstar,no clinicalorwholepredictors. No formallyexcludedexperimentmodelsused.',
        'improvement':'Stabilize12-scorefusion,notre-extractimages:convexdecision-scoreblendtowardpositiveequal-memberone-scoremodel;nonnegative12weights;nonnegative2normalizationgroupsor3viewgroups.',
        'candidates':options(),'candidate_count':17,
        'mean_model':'Average12OOF-standardizedclippedmemberscores;trainpositive1DmetaL2C.03withanchor.25. Blendlogitswithexactbaseline atweights.25/.5/.75/1;notaveragingAUCsorselectingbytest.',
        'nonnegative':'BalancedloglosswithL2penaltyoncoefficientsandintercept,coef>=0,interceptfree;NN12anchor.25/1,C.01/.03/.1;groupNorm2/View3anchor.25,C.01/.03/.1.',
        'normalization':'Memberclip/meanSDfromcurrenttrainingOOFonly;groupmeanSDfromsameOOFonly;applyunchangedtoactualtrain/heldout. Collapsegroupcoefficientsbackto12inputspaceforfrozeninference.',
        'anchor':'TrainmetaonOOFplusactualBag15traininginputs,weights1/(1+anchor),anchor/(1+anchor),totalsampleweightN. Actualtraininginputsarenotindependentvalidation.',
        'selection':'For eachoutertrainingportion,all17 evaluatedonits15innerholdouts;EACHinnertrainingportiongeneratesprivate5x3OOFs. Requireavailableallinnerfoldsandactualfull-current-trainingfit>=exactbaselinefit;objectiveMeanInnerAUC-.10SD;within.01bestpreferfewerdimensions,lowerC,loweranchor,candidateID. Full84fitfloor.7680311890838207.',
        'outer':'All5heldoutfoldsretained;oneAdaptive17primary. All17fixedrecipestraining-onlydiagnostics,neverchoosebyhighestouterrow.',
        'source_contexts':'RootcontextsandBag15basesexactimmutablemedianstackcheckpoints. 90newprivatecontextsrefitunchangedPLSbasepipelinewitholddeepfoldseeds;noBaseROIcontextsusedasMediancontexts.',
        'threshold':'OwnAdaptive17all84outerOOFYoudenbeforetest. If finalpredictorisexactbaseline0,preservebaseline thresholdand36scores;statepredictorunchanged.',
        'test':'Oneadaptiveprimaryonlyafterfreeze/verification;17fixedrecipesnevertested. Reusehistoricalbaselinescores. No testscorefloorusedforselection,posttestfit,rethresholdingorlabeldirectionflip.',
        'history':'KnownbaselineA36historicallyreusedandAselectedfromfivegroups;allresultsconditionalexploratory,nofreshexternalvalidation. Noformaladoptionchanges.',
        'source_hashes':{str(p.relative_to(ROOT)):digest(p) for p in sources},'protected_hashes':protected}
    save(OUT/'protocol.json',p);save(OUT/'definition_frozen.json',{'passed':True,'protocol_sha256':digest(OUT/'protocol.json'),'test_calls':0});csv('candidate_definitions',options())
    print('Defined17 originalmedianstack shrink/stability candidates;sameA84/36;only1mm',flush=True)


def root_context(ph):return joblib.load(base.OUT/'models'/('R1_ContourMedian_'+ph+'.joblib'))['context']


def make_context(tables,y,ph,rep,k,a,b):
    dest=OUT/'contexts'/('%s_r%df%d.joblib'%(ph,rep,k))
    if dest.exists():return joblib.load(dest)
    original=tables[1.];d=base.route_tables(original,'ContourMedian');batch=np.r_[a,b]
    lookup={int(i):j for j,i in enumerate(a)};repeat=np.full((len(a),3,12),np.nan);bag=np.zeros((len(batch),12));audit=[]
    for rr,kk,aa,bb in base.old.deep_folds(ph,rep,k,a,y):
        assert set(aa).isdisjoint(bb) and set(aa)|set(bb)==set(a) and set(a).isdisjoint(b)
        models=base.fit12(original,d,aa,batch,y,1.)
        p=np.column_stack([m['evaluation_probability'] for m in models]);positions=[lookup[int(i)] for i in bb]
        repeat[positions,rr-1]=p[positions];bag+=p/15.
        audit.append({'Repeat':rr,'Fold':kk,'train':aa,'validation':bb})
    assert np.isfinite(repeat).all()
    ctx={'train':a,'validation':b,'oof':repeat.mean(1),'bag_train':bag[:len(a)],'bag_validation':bag[len(a):],
        'repeated_oof':repeat,'folds':audit,'private_median_context':True}
    joblib.dump(ctx,dest,compress=3);return ctx


def build_worker(ph):
    with threadpool_limits(1):
        cohort,y,tr,te=base.old.inputs();tables=joblib.load(base.OUT/'training_features.joblib');done=0
        for rep,k,a,b in base.old.previous.parent.folds('Random120',ph):
            make_context(tables,y,ph,rep,k,a,b);done+=1
            if k==5:print(ph+' privateMedianOOF repeat%d/3complete'%rep,flush=True)
        return ph,done


def contexts(workers):
    protect();assert not (OUT/'training_frozen.json').exists() and not (OUT/'test_started.json').exists()
    start=time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        fs=[pool.submit(build_worker,ph) for ph in PHASES]
        for f in as_completed(fs):print('Builtprivatecontexts '+str(f.result())+' %.0fs'%(time.time()-start),flush=True)
    save(OUT/'context_build_complete.json',{'passed':True,'private_contexts':90,'test_calls':0,'median_only':True})


def mapping(family):
    members=base.old.previous.members()
    if family=='Mean1':return np.ones((12,1))/12.
    if family=='NN12':return np.eye(12)
    values=[c['normalization'] if family=='Norm2' else c['view'] for c in members]
    groups=list(dict.fromkeys(values));B=np.zeros((12,len(groups)))
    for j,g in enumerate(groups):
        ix=[i for i,v in enumerate(values) if v==g];B[ix,j]=1./len(ix)
    assert len(groups)==(2 if family=='Norm2' else 3)
    return B


def nn_objective(w,x,target,weights,C):
    design=np.column_stack([x,np.ones(len(x))]);s=design@w
    loss=np.sum(weights*(np.logaddexp(0,s)-target*s))+.5*np.sum(w*w)/C
    grad=design.T@(weights*(expit(s)-target))+w/C
    return float(loss),grad


def fit_nn(ctx,y,family,anchor,C):
    a=ctx['train'];pp=base.old.previous.meta_prepare(ctx['oof'],ctx['bag_validation'])
    za=(np.clip(ctx['bag_train'],pp['low'],pp['high'])-pp['center'])/pp['scale'];B=mapping(family)
    g=pp['xa']@B;center=g.mean(0);scale=g.std(0);scale=np.where(scale>1e-12,scale,1.)
    ga=(za@B-center)/scale;go=(g-center)/scale
    x=np.row_stack([go,ga]);target=np.r_[y[a],y[a]]
    sw=np.r_[np.full(len(a),1/(1+anchor)),np.full(len(a),anchor/(1+anchor))]
    weights=sw*len(target)/(2*np.bincount(target)[target]);n=B.shape[1]
    solution=minimize(lambda w:nn_objective(w,x,target,weights,C),np.zeros(n+1),jac=True,method='L-BFGS-B',
        bounds=[(0.,None)]*n+[(None,None)],options={'maxiter':3000,'ftol':1e-13,'gtol':1e-8})
    beta=B@(solution.x[:n]/scale);intercept=float(solution.x[-1]-center@(solution.x[:n]/scale))
    pa=expit(za@beta+intercept);pb=expit(pp['xb']@beta+intercept)
    return {'train':a,'validation':ctx['validation'],'coef':beta,'intercept':intercept,
        **{k:pp[k] for k in ('low','high','center','scale')},'train_probability':pa,'evaluation_probability':pb,
        'meta_training_association_probability':expit(pp['xa']@beta+intercept),
        'available':bool(solution.success and np.ptp(pa)>1e-12),'optimizer_success':bool(solution.success),
        'objective':float(solution.fun),'group_mapping':B,'group_center':center,'group_scale':scale,'group_coef':solution.x[:n],
        'group_intercept':float(solution.x[-1]),'dimensions':n,'ScoreType':'regularized_positive_habitat_score_uncalibrated'}


def fit_all(ctx,y):
    reference=base.old.meta_fit(ctx,y,base.CONFIG.copy());mean=fit_nn(ctx,y,'Mean1',.25,.03);result={}
    pp=base.old.previous.meta_prepare(ctx['oof'],ctx['bag_validation']);za=(np.clip(ctx['bag_train'],pp['low'],pp['high'])-pp['center'])/pp['scale']
    for c in options():
        if c['family']=='Baseline':f=reference.copy()
        elif c['family']=='Blend':
            w=c['mix'];beta=(1-w)*reference['coef']+w*mean['coef'];intercept=float((1-w)*reference['intercept']+w*mean['intercept'])
            f={k:reference[k] for k in ('train','validation','low','high','center','scale')}
            f.update(coef=beta,intercept=intercept,train_probability=expit(za@beta+intercept),
                evaluation_probability=expit(pp['xb']@beta+intercept),meta_training_association_probability=expit(pp['xa']@beta+intercept),
                available=bool(reference['available'] and mean['available'] and np.ptp(expit(za@beta+intercept))>1e-12),dimensions=c['dimensions'])
        else:f=fit_nn(ctx,y,c['family'],c['anchor'],c['C'])
        f['config']={**c,'base_mode':'Bag15'};f['dimensions']=c['dimensions'];result[c['candidate_id']]=f
    return result


def choose(rows,models,y,a):
    floor=float(roc_auc_score(y[a],models[0]['train_probability']))
    pool=[r for r in rows if r['AvailableAllFolds'] and models[r['candidate_id']]['available'] and roc_auc_score(y[a],models[r['candidate_id']]['train_probability'])>=floor-1e-12]
    assert pool
    top=max(r['Objective'] for r in pool);near=[r for r in pool if r['Objective']>=top-.01-1e-12]
    selected=min(near,key=lambda r:(r['dimensions'],r['C'],r['anchor'],r['candidate_id']))
    return selected,{'FitFloor':floor,'Eligible':len(pool),'BestEligibleObjective':top,'NearBestTolerance':.01}


def train():
    protect();assert not (OUT/'training_frozen.json').exists() and not (OUT/'test_started.json').exists()
    assert json.loads((OUT/'context_build_complete.json').read_text(encoding='utf-8'))['passed']
    cohort,y,tr,te=base.old.inputs();metrics=[];predictions=[];grids=[];audits=[];choices=[]
    for ph in PHASES:
        dest=OUT/'phases'/(ph+'.joblib')
        if dest.exists():continue
        entries=[]
        for rep,k,a,b in base.old.previous.parent.folds('Random120',ph):
            ctx=joblib.load(OUT/'contexts'/('%s_r%df%d.joblib'%(ph,rep,k)))
            for cid,f in fit_all(ctx,y).items():entries.append({'Phase':ph,'Repeat':rep,'Fold':k,'candidate_id':cid,
                'FitAUC':roc_auc_score(y[a],f['train_probability']),'AUC':roc_auc_score(y[b],f['evaluation_probability']),'Available':f['available']})
        frame=pd.DataFrame(entries);rows=[]
        for c in options():
            z=frame[frame.candidate_id==c['candidate_id']];mean,sd=float(z.AUC.mean()),float(z.AUC.std(ddof=1))
            rows.append({**c,'Phase':ph,'MeanInnerAUC':mean,'InnerSD':sd,'MeanFitAUC':float(z.FitAUC.mean()),'AvailableAllFolds':bool(z.Available.all()),'Objective':mean-.1*sd})
        ctx=root_context(ph);models=fit_all(ctx,y);selected,guard=choose(rows,models,y,ctx['train'])
        old=joblib.load(base.OUT/'models'/('R1_ContourMedian_'+ph+'.joblib'))['fit']
        np.testing.assert_array_equal(models[0]['train_probability'],old['train_probability']);np.testing.assert_array_equal(models[0]['evaluation_probability'],old['evaluation_probability'])
        cp={'phase':ph,'entries':entries,'grid':rows,'selected':selected,'guard':guard,'models':models}
        joblib.dump(cp,dest,compress=3);print(ph+' Adaptive17 candidate%d;trainfloor%.4f;notest'%(selected['candidate_id'],guard['FitFloor']),flush=True)
    for ph in PHASES:
        cp=joblib.load(OUT/'phases'/(ph+'.joblib'));grids.extend(cp['grid']);audits.extend(cp['entries']);choices.append(cp['selected'])
        if ph=='final':continue
        k=int(ph[-1]);a,b=base.old.previous.parent.folds('Random120')[k-1]
        for name,f in [('Adaptive17',cp['models'][cp['selected']['candidate_id']])]+[('Fixed%d'%cid,f) for cid,f in cp['models'].items()]:
            metrics.append({'ModelKey':name,'Fold':k,'candidate_id':f['config']['candidate_id'],'FitAUC':roc_auc_score(y[a],f['train_probability']),'AUC':roc_auc_score(y[b],f['evaluation_probability'])})
            predictions.extend({'ModelKey':name,'Fold':k,'Index':int(i),'AnalysisID':cohort.AnalysisID.iloc[i],'Outcome':int(y[i]),'Probability':float(p)} for i,p in zip(b,f['evaluation_probability']))
    final=joblib.load(OUT/'phases/final.joblib');f=final['models'][final['selected']['candidate_id']].copy();pp=pd.DataFrame(predictions);pv=pp[pp.ModelKey=='Adaptive17'].set_index('Index').loc[tr]
    identity=f['config']['candidate_id']==0
    if identity:f['threshold']=float(joblib.load(base.OUT/'models/R1_ContourMedian_final.joblib')['fit']['threshold']);f['threshold_source']='unchanged_exactbaseline_threshold'
    else:
        x,z,thresholds=roc_curve(y[tr],pv.Probability,drop_intermediate=False);f['threshold']=float(thresholds[np.argmax(z-x)]);f['threshold_source']='Adaptive17all84outerOOF_Youden'
    f['baseline_predictor_identical']=identity;joblib.dump(f,OUT/'models/final_primary.joblib',compress=3)
    mm=pd.DataFrame(metrics);summary=[]
    for name,fit in [('Adaptive17',f)]+[('Fixed%d'%cid,x) for cid,x in final['models'].items()]:
        z=mm[mm.ModelKey==name];p=pp[pp.ModelKey==name]
        summary.append({'ModelKey':name,**fit['config'],'TrainFitAUC':roc_auc_score(y[tr],fit['train_probability']),
            'OuterMeanAUC':z.AUC.mean(),'OuterSD':z.AUC.std(ddof=1),'PooledOOFAUC':roc_auc_score(p.Outcome,p.Probability),
            'MetaInSampleOOFAssociationAUC':roc_auc_score(y[tr],fit['meta_training_association_probability'])})
    csv('training_summary',summary);csv('outer_metrics',metrics);csv('nested_predictions',predictions);csv('selected_configurations',choices);csv('inner_scores',audits);csv('inner_candidate_summary',grids)
    csv('final_member_weights',[{'Member':c['CandidateKey'],'Normalization':c['normalization'],'View':c['view'],'PLScomponents':c['components'],'StandardizedScoreCoefficient':float(beta)} for c,beta in zip(base.old.previous.members(),f['coef'])])
    files={str(p.relative_to(OUT)):digest(p) for p in OUT.rglob('*') if p.is_file()}
    save(OUT/'training_frozen.json',{'passed':True,'test_calls':0,'files':files});print(pd.DataFrame(summary).to_string(index=False),flush=True)


def frozen():
    protect();p=json.loads((OUT/'training_frozen.json').read_text(encoding='utf-8'))
    for name,sha in p['files'].items():assert digest(OUT/name)==sha,name
    return p


def test():
    frozen();assert not (OUT/'test_started.json').exists()
    assert json.loads((OUT/'training_verification.json').read_text(encoding='utf-8'))['passed']
    cohort,_,tr,te=base.old.inputs();f=joblib.load(OUT/'models/final_primary.joblib');old=base.read('test_predictions');old=old[old.ModelKey=='R1_ContourMedian'].set_index('Index').loc[te]
    calls=0 if f['baseline_predictor_identical'] else 1
    save(OUT/'test_started.json',{'new_complete_model_calls':calls,'primary_count':1,'fixed_candidate_tests':0,'post_test_fits':0})
    if calls:
        cp=joblib.load(base.OUT/'models/R1_ContourMedian_final.joblib');tables=joblib.load(base.OUT/'test_features.joblib');cp['fit']={**f,'radius_mm':1.,'mask_mode':'ContourMedian'}
        p=base.infer(cp,tables,te);manual=base.formula(cp,tables,te);np.testing.assert_allclose(p,manual,atol=1e-9,rtol=0)
    else:p=old.Probability.to_numpy()
    truth=cohort.Outcome.to_numpy(int)[te];from verify_ivim_df63_20261020 import draws,auc
    np.testing.assert_array_equal(old.Outcome.to_numpy(int),truth)
    assert old.AnalysisID.tolist()==cohort.AnalysisID.iloc[te].tolist()
    np.testing.assert_allclose(auc(truth,p),roc_auc_score(truth,p),atol=1e-12)
    samples=draws(truth,{'new':p,'baseline':old.Probability.to_numpy()},20261025128);lo,hi=np.quantile(samples['new'],[.025,.975]);dl,dh=np.quantile(samples['new']-samples['baseline'],[.025,.975])
    tn,fp,fn,tp=confusion_matrix(truth,p>=f['threshold'],labels=[0,1]).ravel()
    row={'ModelKey':'Adaptive17','TestAUC':roc_auc_score(truth,p),'AUC_CI95Low':float(lo),'AUC_CI95High':float(hi),
        'DifferenceVsBaseline':float(roc_auc_score(truth,p)-roc_auc_score(truth,old.Probability)),
        'PairedDifferenceCI95Low':float(dl),'PairedDifferenceCI95High':float(dh),'Threshold':f['threshold'],
        'Sensitivity':tp/(tp+fn),'Specificity':tn/(tn+fp),'Accuracy':(tp+tn)/36.,'TP':int(tp),'FN':int(fn),'TN':int(tn),'FP':int(fp),
        'NewModelCalls':calls,'BaselinePredictorIdentical':bool(f['baseline_predictor_identical'])}
    csv('test_metrics',[row]);csv('test_predictions',[{'Index':int(i),'AnalysisID':cohort.AnalysisID.iloc[i],'Outcome':int(y),'Probability':float(v),'Prediction':int(v>=f['threshold'])} for i,y,v in zip(te,truth,p)])
    train=read('training_summary');train[train.ModelKey=='Adaptive17'].merge(pd.DataFrame([row]).drop(columns='ModelKey'),how='cross').to_csv(OUT/'primary_comparison.csv',index=False,encoding='utf-8-sig')
    frozen();save(OUT/'test_complete.json',{'passed':True,'joint_stages':1,'new_complete_model_calls':calls,'fixed_candidate_tests':0,'post_test_fits':0,'post_test_threshold_changes':0,'formal_adoption_changed':False})
    print(pd.DataFrame([row]).to_string(index=False),flush=True)


def seal():
    frozen()
    for n in ('training_verification','test_complete','report_verification','visual_verification'):assert json.loads((OUT/(n+'.json')).read_text(encoding='utf-8'))['passed']
    files={str(p.relative_to(OUT)):digest(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ('delivery_manifest.json','delivery_final_verification.json')}
    save(OUT/'delivery_manifest.json',{'outputs':files,'protected_files_unchanged':True})
    for name,sha in files.items():assert digest(OUT/name)==sha,name
    save(OUT/'delivery_final_verification.json',{'passed':True,'outputs_checked':len(files),'adaptive_primary':1,'fixed_candidate_tests':0,'post_test_fits':0,'formal_adoption_changed':False})
    print('Originalmedianstack refinementdelivered;priorrecordsunchanged',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('define','contexts','train','test','seal'));p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    with threadpool_limits(1):contexts(a.workers) if a.action=='contexts' else globals()[a.action]()
