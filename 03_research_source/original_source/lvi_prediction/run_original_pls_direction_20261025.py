# -*- coding: utf-8 -*-
"""Original median PLS stack with fold-local bootstrap direction screening."""
import os
os.environ.update({'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import copy
import json
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import roc_auc_score,roc_curve,confusion_matrix
from threadpoolctl import threadpool_limits
import run_original_stack_shrink_20261025 as parent

ROOT,save,digest=parent.ROOT,parent.save,parent.digest
base=parent.base
OUT=ROOT/'lvi_prediction/results/lvi_original_pls_direction_20261025'
THRESHOLDS=(0.,.75,.85)
PHASES=parent.PHASES
PREPARED={}


def csv(name,rows):pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')
def read(name):return pd.read_csv(OUT/(name+'.csv'),float_precision='round_trip')
def tag(t):return 'T%d'%round(100*t)
def location(prefix,t):return OUT/'contexts'/(prefix+'_'+tag(t)+'.joblib')


def protect():
    p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    for field in ('source_hashes','protected_hashes'):
        for name,sha in p[field].items():assert digest(ROOT/name)==sha,name
    return len(p['protected_hashes'])


def define():
    assert not OUT.exists();parent.frozen()
    assert json.loads((parent.OUT/'training_verification.json').read_text(encoding='utf-8'))['passed']
    assert json.loads((parent.OUT/'test_deferral.json').read_text(encoding='utf-8'))['new_test_calls']==0
    p=json.loads((parent.OUT/'protocol.json').read_text(encoding='utf-8'));protected={**p['protected_hashes'],**p['source_hashes']}
    protected.update({str(f.relative_to(ROOT)):digest(f) for f in parent.OUT.rglob('*') if f.is_file()})
    OUT.mkdir(parents=True)
    for n in ('contexts','models','phases','figures'):(OUT/n).mkdir()
    sources=[ROOT/'lvi_prediction'/n for n in ('run_original_pls_direction_20261025.py','verify_original_pls_direction_20261025.py','report_original_pls_direction_20261025.py')]
    save(OUT/'protocol.json',{'date':'2026-10-25','authorization':'Continueimprovingonoriginalretainedpurehabitatstack,notnewregionL1fusion. Only1mm.',
        'cohort':'ExactsameA120,84train57+/27-,36test25+/11-;sameouter5,inner5x3,private5x3;nocase/foldselection.',
        'preserved':'Exact1mmcontourmedianT2/ADC/IVIMD/fk3,2normalizations,3regionalviews,12PLS1/2basesC.03,ICC.75complete.9Spearman.95GaussianQT;Bag15;signedmetaC.03anchor.25.',
        'new_step':'Afteroriginalfold-onlyQCanddecorrelation,withinCURRENTBASETRAININGONLY computeeachfeatureMannWhitneyAUCdirection and40stratifiedbootstrapdirections;retainallfeatureswithagreement>=.75or.85. Nofixedfeaturecountcap,noL1,newregionsorweightconstraints.',
        'bootstrap':'40multinomialpositive/negativeclass-preservingweights;seed2026102501+weightedcurrentbase-trainingindexsum%1000000,sameseedacrossviewsforthesamepatients. Positive/negativeAUCpaircontributionwithties.5;zerooriginaleffectdirectioncannotpass.',
        'interpretation':'Exploratorydirection-consistencyfilter,notformalMeinshausen-Buhlmannstabilityselectionandnofalse-discoveryguarantee. Thesamedatamustnotbeusedtochoosefeaturesoutsideitscurrenttrainingfold.',
        'after_selection':'Re-fitfeaturewiseGaussianQTonthechosenrawcurrenttraincolumns;PLS/Logisticrefittedinsideeachbasefold. Nofeaturesavailable=>originalconstantprevalencebasefallback;neverexcludeapatient.',
        'candidate_thresholds':THRESHOLDS,'candidate_count':3,'baseline':'ExactoriginalmedianstackrootandprivateOOFcheckpointreuse,preservedreference0.768031/0.621818/0.821818.',
        'selection':'15innerfoldsmeanAUC-.10SD;availableall15andactualfull-current-trainingfit>=baselinefloor. Within.01bestpreferbaseline,thenlessstrictdirectionthreshold. Outer5fullmeanprimary;fixedrecipestrainingdiagnosticsonly.',
        'strict_nesting':'EveryinnertrainingpartgeneratesOWN15baseOOFswithdirectionselectionfittedinsideeachprivatebasetrainingpart;noouter/globaltargetfeatureselection.',
        'test':'OneAdaptive3onlyafterfreeze/verification;ifbaseline0selectedpreservebaselinefunction/thresholdandold36scores. No3candidatetestcomparison,no test-basedselectionorposttestfit.',
        'threshold':'Adaptivecomplete84outerOOFYouden;baselineidentitykeepsoldthreshold.',
        'failed_parent':'17fusion-stabilitycandidatesverifiedandtraining-onlydeferredforlackofoutergain;no newtestcalls there.',
        'history':'SameA36historicallyreused;exploratoryinternal,nofreshexternalvalidation. Clinical/whole/Dstar/2or3mmnone. Formaladoptionunchanged.',
        'source_hashes':{str(f.relative_to(ROOT)):digest(f) for f in sources},'protected_hashes':protected})
    save(OUT/'definition_frozen.json',{'passed':True,'protocol_sha256':digest(OUT/'protocol.json'),'test_calls':0})
    print('Defined3originalPLSdirection candidates;only1mm;allfeaturecountsadaptivewithoutcap',flush=True)


def direction_stats(x,target,a):
    if x.shape[1]==0:return np.empty(0),np.empty(0)
    positive=x[target==1];negative=x[target==0];p,n=len(positive),len(negative)
    delta=positive[:,None,:]-negative[None,:,:];phi=(delta>0).astype(float)+.5*(delta==0)
    effect=phi.mean((0,1))-.5;direction=np.where(np.abs(effect)>1e-12,np.sign(effect),0.)
    seed=2026102501+sum((j+1)*(int(i)+1) for j,i in enumerate(a))%1000000;rng=np.random.default_rng(seed)
    wp=rng.multinomial(p,np.full(p,1/p),40)/p;wn=rng.multinomial(n,np.full(n,1/n),40)/n
    boot=np.einsum('bi,ijm,bj->bm',wp,phi,wn,optimize=True)-.5
    agreement=((boot*direction[None,:])>1e-12).mean(0)
    return agreement,direction


def prepared(original,tables,a,key,y):
    token=(key,tuple(a))
    if token in PREPARED:return PREPARED[token]
    qc=base.backend.quality(original[key],a)
    p=base.backend.prepare(tables[key],a,np.empty(0,int),{'mask_mode':'Base','icc_min':.75},qc)
    p['agreement'],p['direction']=direction_stats(p['xa'],y[a],a);p['audit']=[]
    PREPARED[token]=p;return p


def filtered(p,d,a,b,t):
    keep=np.flatnonzero(p['agreement']>=t);indices=[p['indices'][j] for j in keep]
    q={**p,'indices':indices,'names':[p['names'][j] for j in keep],'median':p['median'][keep],
        'agreement':p['agreement'][keep],'direction':p['direction'][keep],'before_count':len(p['indices'])}
    if not indices:q.update(quantile=None,xa=np.empty((len(a),0)),xb=np.empty((len(b),0)));return q
    x=d['base'][base.old.latent.rows(d,a)][:,indices];x=np.where(np.isfinite(x),x,q['median'])
    qt=QuantileTransformer(n_quantiles=len(a),output_distribution='normal',subsample=len(a),random_state=64212).fit(x)
    xa=np.clip(qt.transform(x),-2.5,2.5);np.testing.assert_allclose(xa,p['xa'][:,keep],atol=1e-10,rtol=0)
    z=d['base'][base.old.latent.rows(d,b)][:,indices];z=np.where(np.isfinite(z),z,q['median'])
    q.update(quantile=qt,xa=xa,xb=np.clip(qt.transform(z),-2.5,2.5) if len(b) else np.empty((0,len(indices))))
    return q


def fit_pairs(original,tables,a,b,y):
    fitted={t:[] for t in THRESHOLDS[1:]};cache={}
    for c in base.old.previous.members():
        key=c['image_key']
        if key not in cache:
            p=prepared(original,tables,a,key,y);cache[key]={t:filtered(p,tables[key],a,b,t) for t in THRESHOLDS[1:]}
        for t in THRESHOLDS[1:]:
            p=cache[key][t];f=base.old.latent.fitted(p,y[a],c);f.pop('audit',None)
            f.update(direction_threshold=t,train=np.asarray(a,int),direction_agreement=p['agreement'],direction_sign=p['direction'],before_direction_count=p['before_count'])
            fitted[t].append(f)
    return fitted


def build_pair(prefix,tables,y,a,b,folds,keep=False,replay=False):
    if not replay and all(location(prefix,t).exists() for t in THRESHOLDS[1:]):return {t:joblib.load(location(prefix,t)) for t in THRESHOLDS[1:]}
    original=tables[1.];d=base.route_tables(original,'ContourMedian');batch=np.r_[a,b];lookup={int(i):j for j,i in enumerate(a)}
    repeat={t:np.full((len(a),3,12),np.nan) for t in THRESHOLDS[1:]};bag={t:np.zeros((len(batch),12)) for t in THRESHOLDS[1:]};kept={t:[] for t in THRESHOLDS[1:]};audit=[]
    for rr,kk,aa,bb in folds:
        assert set(aa).isdisjoint(bb) and set(aa)|set(bb)==set(a) and set(a).isdisjoint(b)
        models=fit_pairs(original,d,aa,batch,y);positions=[lookup[int(i)] for i in bb]
        for t in THRESHOLDS[1:]:
            p=np.column_stack([f['evaluation_probability'] for f in models[t]]);repeat[t][positions,rr-1]=p[positions];bag[t]+=p/15.
            if keep:kept[t].append({'Repeat':rr,'Fold':kk,'train':aa,'validation':bb,'models':models[t]})
        audit.append({'Repeat':rr,'Fold':kk,'train':aa,'validation':bb})
    result={}
    for t in THRESHOLDS[1:]:
        assert np.isfinite(repeat[t]).all()
        ctx={'train':a,'validation':b,'oof':repeat[t].mean(1),'bag_train':bag[t][:len(a)],'bag_validation':bag[t][len(a):],
            'repeated_oof':repeat[t],'folds':audit,'direction_threshold':t}
        result[t]=ctx
        if not replay:
            joblib.dump(ctx,location(prefix,t),compress=3)
            if keep:joblib.dump({'train':a,'validation':b,'bag':kept[t],'full':[]},OUT/'models'/(prefix+'_'+tag(t)+'_bases.joblib'),compress=3)
    return result


def worker(ph):
    with threadpool_limits(1):
        cohort,y,tr,te=base.old.inputs();tables=joblib.load(base.OUT/'training_features.joblib')
        a,b=(tr,np.empty(0,int)) if ph=='final' else base.old.previous.parent.folds('Random120')[int(ph[-1])-1]
        build_pair(ph,tables,y,a,b,base.old.previous.parent.folds('Random120',ph),True)
        for rep,k,a,b in base.old.previous.parent.folds('Random120',ph):
            build_pair('%s_r%df%d'%(ph,rep,k),tables,y,a,b,base.old.deep_folds(ph,rep,k,a,y))
            if k==5:print(ph+' direction privateOOF repeat%d/3complete'%rep,flush=True)
        PREPARED.clear();return ph


def contexts(workers):
    protect();assert not (OUT/'training_frozen.json').exists()
    start=time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(worker,ph) for ph in PHASES]):print('Builtdirection '+f.result()+' %.0fs'%(time.time()-start),flush=True)
    save(OUT/'contexts_complete.json',{'passed':True,'new_private_contexts':180,'new_root_contexts':12,'test_calls':0})


def context(ph,t,rep=None,k=None):
    if t==0:return parent.root_context(ph) if rep is None else joblib.load(parent.OUT/'contexts'/('%s_r%df%d.joblib'%(ph,rep,k)))
    return joblib.load(location(ph if rep is None else '%s_r%df%d'%(ph,rep,k),t))


def fit(ctx,y,t):
    f=base.old.meta_fit(ctx,y,{**base.CONFIG,'direction_threshold':t});f['radius_mm']=1.;f['mask_mode']='ContourMedian';f['direction_threshold']=t
    return f


def choose(rows,models,y,a):
    floor=roc_auc_score(y[a],models[0.]['train_probability'])
    pool=[r for r in rows if r['AvailableAllFolds'] and models[r['threshold']]['available'] and roc_auc_score(y[a],models[r['threshold']]['train_probability'])>=floor-1e-12]
    assert pool;best=max(r['Objective'] for r in pool);near=[r for r in pool if r['Objective']>=best-.01-1e-12]
    return min(near,key=lambda r:r['threshold']),{'FitFloor':float(floor),'Eligible':len(pool),'BestObjective':best}


def train():
    protect();assert not (OUT/'training_frozen.json').exists() and not (OUT/'test_started.json').exists()
    assert json.loads((OUT/'contexts_complete.json').read_text(encoding='utf-8'))['passed']
    cohort,y,tr,te=base.old.inputs();metrics=[];predictions=[];choices=[];inner=[];grids=[]
    for ph in PHASES:
        path=OUT/'phases'/(ph+'.joblib')
        if path.exists():continue
        entries=[]
        for rep,k,a,b in base.old.previous.parent.folds('Random120',ph):
            for t in THRESHOLDS:
                f=fit(context(ph,t,rep,k),y,t)
                entries.append({'Phase':ph,'Repeat':rep,'Fold':k,'threshold':t,'FitAUC':roc_auc_score(y[a],f['train_probability']),'AUC':roc_auc_score(y[b],f['evaluation_probability']),'Available':f['available']})
        frame=pd.DataFrame(entries);rows=[]
        for t in THRESHOLDS:
            z=frame[frame.threshold==t];m,s=float(z.AUC.mean()),float(z.AUC.std(ddof=1))
            rows.append({'Phase':ph,'threshold':t,'MeanInnerAUC':m,'InnerSD':s,'MeanFitAUC':float(z.FitAUC.mean()),'AvailableAllFolds':bool(z.Available.all()),'Objective':m-.1*s})
        models={t:fit(context(ph,t),y,t) for t in THRESHOLDS};a=models[0.]['train'];selected,guard=choose(rows,models,y,a)
        cp={'models':models,'selected':selected,'guard':guard,'entries':entries,'grid':rows}
        joblib.dump(cp,path,compress=3);print(ph+' DirectionAdaptive threshold%g;notest'%selected['threshold'],flush=True)
    for ph in PHASES:
        cp=joblib.load(OUT/'phases'/(ph+'.joblib'));choices.append(cp['selected']);inner.extend(cp['entries']);grids.extend(cp['grid'])
        if ph=='final':continue
        k=int(ph[-1]);a,b=base.old.previous.parent.folds('Random120')[k-1]
        for name,f in [('Adaptive3',cp['models'][cp['selected']['threshold']])]+[('Fixed'+tag(t),cp['models'][t]) for t in THRESHOLDS]:
            metrics.append({'ModelKey':name,'Fold':k,'DirectionThreshold':f['direction_threshold'],'FitAUC':roc_auc_score(y[a],f['train_probability']),'AUC':roc_auc_score(y[b],f['evaluation_probability'])})
            predictions.extend({'ModelKey':name,'Fold':k,'Index':int(i),'AnalysisID':cohort.AnalysisID.iloc[i],'Outcome':int(y[i]),'Probability':float(v)} for i,v in zip(b,f['evaluation_probability']))
    final=joblib.load(OUT/'phases/final.joblib');selected=final['selected']['threshold'];f=final['models'][selected].copy();pp=pd.DataFrame(predictions);pv=pp[pp.ModelKey=='Adaptive3'].set_index('Index').loc[tr]
    if selected==0.:
        f['threshold']=float(joblib.load(base.OUT/'models/R1_ContourMedian_final.joblib')['fit']['threshold']);f['threshold_source']='unchanged_baseline'
    else:
        x,z,thr=roc_curve(y[tr],pv.Probability,drop_intermediate=False);f['threshold']=float(thr[np.argmax(z-x)]);f['threshold_source']='Adaptive3all84outerOOF_Youden'
    f['baseline_predictor_identical']=selected==0.;joblib.dump(f,OUT/'models/final_primary.joblib',compress=3)
    mm=pd.DataFrame(metrics);summary=[]
    for name,ff in [('Adaptive3',f)]+[('Fixed'+tag(t),x) for t,x in final['models'].items()]:
        z=mm[mm.ModelKey==name];p=pp[pp.ModelKey==name]
        summary.append({'ModelKey':name,'DirectionThreshold':ff['direction_threshold'],'TrainFitAUC':roc_auc_score(y[tr],ff['train_probability']),
            'OuterMeanAUC':z.AUC.mean(),'OuterSD':z.AUC.std(ddof=1),'PooledOOFAUC':roc_auc_score(p.Outcome,p.Probability)})
    for n,v in [('training_summary',summary),('outer_metrics',metrics),('nested_predictions',predictions),('selected_configurations',choices),('inner_scores',inner),('inner_candidate_summary',grids)]:csv(n,v)
    files={str(p.relative_to(OUT)):digest(p) for p in OUT.rglob('*') if p.is_file()};save(OUT/'training_frozen.json',{'passed':True,'test_calls':0,'files':files})
    print(pd.DataFrame(summary).to_string(index=False),flush=True)


def frozen():
    protect();p=json.loads((OUT/'training_frozen.json').read_text(encoding='utf-8'))
    for name,sha in p['files'].items():assert digest(OUT/name)==sha,name
    return p


def basis(ph,t):return joblib.load(base.OUT/'models'/('R1_ContourMedian_'+ph+'.joblib'))['bases'] if t==0. else joblib.load(OUT/'models'/(ph+'_'+tag(t)+'_bases.joblib'))


def test():
    frozen();assert not (OUT/'test_started.json').exists()
    assert json.loads((OUT/'training_verification.json').read_text(encoding='utf-8'))['passed']
    cohort,_,tr,te=base.old.inputs();f=joblib.load(OUT/'models/final_primary.joblib');t=f['direction_threshold'];old=base.read('test_predictions');old=old[old.ModelKey=='R1_ContourMedian'].set_index('Index').loc[te]
    calls=0 if t==0. else 1;save(OUT/'test_started.json',{'new_model_calls':calls,'fixed_candidate_tests':0,'post_test_fits':0})
    if calls:
        tables=joblib.load(base.OUT/'test_features.joblib');cp={'fit':f,'bases':basis('final',t)};p=base.infer(cp,tables,te);q=base.formula(cp,tables,te);np.testing.assert_allclose(p,q,atol=1e-9,rtol=0)
    else:p=old.Probability.to_numpy()
    truth=cohort.Outcome.to_numpy(int)[te];np.testing.assert_array_equal(truth,old.Outcome)
    from verify_ivim_df63_20261020 import draws,auc
    np.testing.assert_allclose(auc(truth,p),roc_auc_score(truth,p),atol=1e-12)
    samples=draws(truth,{'new':p,'baseline':old.Probability.to_numpy()},20261025139);lo,hi=np.quantile(samples['new'],[.025,.975]);dl,dh=np.quantile(samples['new']-samples['baseline'],[.025,.975])
    tn,fp,fn,tp=confusion_matrix(truth,p>=f['threshold'],labels=[0,1]).ravel()
    row={'ModelKey':'Adaptive3','DirectionThreshold':t,'TestAUC':roc_auc_score(truth,p),'AUC_CI95Low':float(lo),'AUC_CI95High':float(hi),
        'DifferenceVsBaseline':float(roc_auc_score(truth,p)-roc_auc_score(truth,old.Probability)),'PairedDifferenceCI95Low':float(dl),'PairedDifferenceCI95High':float(dh),
        'Threshold':f['threshold'],'Sensitivity':tp/(tp+fn),'Specificity':tn/(tn+fp),'Accuracy':(tp+tn)/36.,'TP':int(tp),'FN':int(fn),'TN':int(tn),'FP':int(fp),
        'NewModelCalls':calls,'BaselinePredictorIdentical':t==0.}
    csv('test_metrics',[row]);csv('test_predictions',[{'Index':int(i),'AnalysisID':cohort.AnalysisID.iloc[i],'Outcome':int(y),'Probability':float(v),'Prediction':int(v>=f['threshold'])} for i,y,v in zip(te,truth,p)])
    frozen();save(OUT/'test_complete.json',{'passed':True,'new_model_calls':calls,'fixed_candidate_tests':0,'post_test_fits':0,'post_test_threshold_changes':0,'formal_adoption_changed':False});print(pd.DataFrame([row]).to_string(index=False),flush=True)


def seal():
    frozen()
    for n in ('training_verification','test_complete','report_verification','visual_verification'):assert json.loads((OUT/(n+'.json')).read_text(encoding='utf-8'))['passed']
    files={str(p.relative_to(OUT)):digest(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ('delivery_manifest.json','delivery_final_verification.json')};save(OUT/'delivery_manifest.json',{'outputs':files})
    for n,sha in files.items():assert digest(OUT/n)==sha,n
    save(OUT/'delivery_final_verification.json',{'passed':True,'outputs_checked':len(files),'only1mm':True,'new_test_primary_count':1,'fixed_candidate_tests':0,'post_test_fits':0,'formal_adoption_changed':False})
    print('OriginalPLSdirection experimentdelivered;baselineandpriorrecordsunchanged',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('define','contexts','train','test','seal'));p.add_argument('--workers',type=int,default=3);args=p.parse_args()
    with threadpool_limits(1):contexts(args.workers) if args.action=='contexts' else globals()[args.action]()
