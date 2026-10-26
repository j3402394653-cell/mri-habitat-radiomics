# -*- coding: utf-8 -*-
"""Training-only nested development of consistent pure-habitat score fusion."""
import os
os.environ.update({'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from threadpoolctl import threadpool_limits
import run_pure_habitat_random120_stacking_20261023 as previous
import run_random120_five_models_20261023 as five

ROOT, save, digest = previous.ROOT, previous.save, previous.digest
OUT = ROOT/'lvi_prediction/results/lvi_pure_habitat_consistent_stack_20261024'
PHASES = tuple('outer%d'%i for i in range(1,6))+('final',)
latent = previous.parent.previous


def csv(name, rows):
    pd.DataFrame(rows).to_csv(OUT/(name+'.csv'), index=False, encoding='utf-8-sig')


def read(name):
    return pd.read_csv(OUT/(name+'.csv'), float_precision='round_trip', keep_default_na=False, na_values=[''])


def inputs():
    return previous.parent.inputs('Random120')


def options():
    return [{'candidate_id':i,'base_mode':mode,'anchor':anchor,'nonnegative':nn,'C':.03}
        for i,(mode,anchor,nn) in enumerate([(mode,a,False) for mode in ('Full','Bag15') for a in (0.,.25,1.)]+
        [(mode,0.,True) for mode in ('Full','Bag15')])]


def protect():
    p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    for field in ('source_hashes','protected_hashes'):
        for name,sha in p[field].items():
            assert digest(ROOT/name)==sha,name
    assert digest(OUT/'protocol.json')==json.loads((OUT/'definition_frozen.json').read_text(encoding='utf-8'))['protocol_sha256']
    return len(p['protected_hashes'])


def define():
    assert not OUT.exists()
    five.frozen()
    assert json.loads((five.OUT/'delivery_final_verification.json').read_text(encoding='utf-8'))['passed']
    old=json.loads((five.OUT/'protocol.json').read_text(encoding='utf-8'))
    protected={**old['protected_hashes'],**old['source_hashes']}
    protected.update({str(p.relative_to(ROOT)):digest(p) for p in five.OUT.rglob('*') if p.is_file()})
    OUT.mkdir(parents=True)
    for folder in ('contexts','models','figures','Random120'):
        (OUT/folder).mkdir()
    for name in ('sample_manifest.json','nested_folds.json','sampling_record.json'):
        (OUT/name).write_bytes((previous.OUT/name).read_bytes())
    for label in ('Train','Test','Sample'):
        (OUT/'Random120'/(label+'_cases.csv')).write_bytes((five.OUT/'Random120'/(label+'_cases.csv')).read_bytes())
    files=[ROOT/'lvi_prediction'/name for name in ('run_pure_habitat_consistent_stack_20261024.py',
        'verify_pure_habitat_consistent_stack_20261024.py','report_pure_habitat_consistent_stack_20261024.py')]
    p={'date':'2026-10-24','authorization':'Improve pure habitat trainfit andouter5mean toward.75;investigate occasional poorfold.',
       'cohort':'SameRandom120;84train57+/27-,36test25+/11-;no resampling or patient exclusion.',
       'pure_habitat':True,'K':3,'clinical':False,'whole_predictors':False,'Dstar':False,
       'base_models':'Exact previous12PLS1/2,C.03:2normalizations x3habitatviews;fold-onlyICC.75,complete.9,Spearman.95,Gaussianquantilefeatures.',
       'novelty':'Compare retaining15inner fittedbase models perchannel versus full-training refit;compare OOF-only meta toanchoredOOF+actual inference-ontraining meta,plusnonnegative meta.',
       'anchor':'Meta training concatenatesOOF andactualtraininginferencescores withweights1/(1+a),a/(1+a);same totalweightN,balancedL2LR C.03. These are training inputs,not independent validation.',
       'nonnegative':'Same balanced weightedlogloss andL2 intercept/coef penalty asliblinear;12coefficients constrained>=0;interceptfree in sign. This is a diagnostic control,not a causality assumption.',
       'meta_transform':'Learn exact previous P2.5/P97.5 clipping andmeanSD from current trainingOOFonly;applyunchanged toboth actualtraining andheldout scores.',
       'candidate_count':8,'primary':'Adaptive eight-candidate complete pipeline selected insideeachoutertraining part.',
       'inner_selection':'Exact same 5foldx3repeats. To evaluatea stackhyperparameter,each innertraining part independently producesitsOWN5foldx3repeatedOOFs;no outertraining-wideOOF reusedasinnertrainingOOF.',
       'third_level_seeds':'2026102400+phase_index*100+inner_repeat*10+inner_fold;repeatincrements10000. All fixedbefore fitting.',
       'selection':'Available all15folds;prefermean actualfit>=.75 ifany,then meanInnerAUC-.10SD-.30max(meanFit-meanInnerAUC-.10,0);ties smalleranchor,Full overBag,candidateID.',
       'outer':'Original5folds unchanged;never exclude a fold byAUC. Reportall8fixedrecipes as secondarydiagnostics,not newprimaryselectedbyoutermax.',
       'low_fold_audit':'CaseIDs,labels,classcounts,featuremissingness;no individual MRI/pathology re-review assumed. All5 kept.',
       'sensitivity':'Report eachleave-one-fold-out meanandmeanexcludingminimum as POSTHOC4fold sensitivity;not called5foldCV,not usedforselection or.75target.',
       'threshold':'Primary fullpipeline84outerOOF Youden,freeze before anynewtestscore.',
       'test':'AtmostONE primarynewmodel onexisting36 afteralltrainingverification;8secondaryrecipes never scoredon36. HistoricalStack12 scores reused forcomparison.',
       'prior_exposure':'Samecasepool andouterfolds previously explored;conditional exploratory performance,not external orconfirmatory.',
       'formal_use':False,'automatic_adoption':False,'excluded_experiments_used':False,
       'source_hashes':{str(p.relative_to(ROOT)):digest(p) for p in files},'protected_hashes':protected,
       'references':['https://scikit-learn.org/stable/common_pitfalls.html',
        'https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html',
        'https://scikit-learn.org/stable/modules/cross_validation.html']}
    save(OUT/'protocol.json',p)
    save(OUT/'definition_frozen.json',{'passed':True,'new_test_calls':0,'protocol_sha256':digest(OUT/'protocol.json')})
    csv('candidate_definitions',options())
    print('Defined8 purehabitat consistentstack candidates;all5outer retained;strict thirdlevelinnerOOFs',flush=True)


def fit12(tables,a,b,y):
    prepared={};result=[]
    for c in previous.members():
        key=c['image_key']
        if key not in prepared:
            prepared[key]=latent.prepare(tables[key],a,b)
        f=latent.fitted(prepared[key],y[a],c)
        f['ScoreType']='LR_probability_uncalibrated'
        assert f['available']
        result.append(f)
    return result


def matrix(models,tables,indices):
    return np.column_stack([latent.predict(f,tables,indices) for f in models])


def make_context(name,tables,a,b,y,folds,source_oof=None,keep_models=False):
    dest=OUT/'contexts'/(name+'.joblib')
    if dest.exists():
        return joblib.load(dest)
    a=np.asarray(a,int);b=np.asarray(b,int);batch=np.r_[a,b]
    lookup={int(i):j for j,i in enumerate(a)}
    repeat=np.full((len(a),3,12),np.nan);bag=np.zeros((len(batch),12));kept=[]
    audit=[]
    for rep,k,aa,bb in folds:
        assert set(aa).isdisjoint(bb) and set(aa)|set(bb)==set(a) and not set(a)&set(b)
        models=fit12(tables,aa,batch,y)
        score=np.column_stack([f['evaluation_probability'] for f in models])
        rr=[lookup[int(i)] for i in bb]
        repeat[rr,rep-1]=score[rr]
        bag+=score/15.
        audit.append({'Repeat':rep,'Fold':k,'train':aa,'validation':bb})
        if keep_models:
            kept.append({'Repeat':rep,'Fold':k,'train':aa,'validation':bb,'models':models})
    assert np.isfinite(repeat).all()
    oof=repeat.mean(1)
    if source_oof is not None:
        np.testing.assert_allclose(oof,source_oof,atol=1e-10,rtol=0)
    full=fit12(tables,a,b,y)
    result={'name':name,'train':a,'validation':b,'oof':oof,'repeated_oof':repeat,
        'full_train':np.column_stack([f['train_probability'] for f in full]),
        'full_validation':np.column_stack([f['evaluation_probability'] for f in full]),
        'bag_train':bag[:len(a)],'bag_validation':bag[len(a):],'folds':audit}
    if keep_models:
        joblib.dump({'full':full,'bag':kept,'train':a,'validation':b},OUT/'models'/(name+'_bases.joblib'),compress=3)
    joblib.dump(result,dest,compress=3)
    return result


def deep_folds(phase,rep,k,a,y):
    index=PHASES.index(phase)+1; seed=2026102400+index*100+rep*10+k
    result=[]
    for repeat in range(3):
        splitter=StratifiedKFold(5,shuffle=True,random_state=seed+repeat*10000)
        result.extend((repeat+1,fold+1,a[aa],a[bb]) for fold,(aa,bb) in enumerate(splitter.split(np.zeros(len(a)),y[a])))
    return result


def meta_fit(ctx,y,c):
    a,b=ctx['train'],ctx['validation']
    mode=c['base_mode'].lower().replace('15','')
    actual=ctx[mode+'_train'];evaluation=ctx[mode+'_validation']
    pp=previous.meta_prepare(ctx['oof'],evaluation)
    za=(np.clip(actual,pp['low'],pp['high'])-pp['center'])/pp['scale']
    alpha=c['anchor']; x=pp['xa'] if alpha==0 else np.row_stack([pp['xa'],za])
    target=y[a] if alpha==0 else np.r_[y[a],y[a]]
    sw=np.ones(len(a)) if alpha==0 else np.r_[np.ones(len(a))/(1+alpha),np.full(len(a),alpha/(1+alpha))]
    if not c['nonnegative']:
        model=LogisticRegression(C=c['C'],penalty='l2',solver='liblinear',class_weight='balanced',max_iter=3000,random_state=64212).fit(x,target,sample_weight=sw)
        beta=model.coef_[0].copy();intercept=float(model.intercept_[0]);success=True
    else:
        design=np.column_stack([x,np.ones(len(x))]);cw=len(target)/(2*np.bincount(target)[target]);weights=sw*cw
        def fun(w):
            score=design@w
            loss=np.sum(weights*(np.logaddexp(0,score)-target*score))+.5*np.sum(w*w)/c['C']
            grad=design.T@(weights*(expit(score)-target))+w/c['C']
            return loss,grad
        solution=minimize(fun,np.zeros(13),jac=True,method='L-BFGS-B',bounds=[(0.,None)]*12+[(None,None)],options={'ftol':1e-13,'gtol':1e-8,'maxiter':3000})
        beta,intercept=solution.x[:12],float(solution.x[12]);model=None;success=bool(solution.success)
    pa=expit(za@beta+intercept);pb=expit(pp['xb']@beta+intercept)
    return {'config':c,'train':a,'validation':b,'coef':beta,'intercept':intercept,
        **{k:pp[k] for k in ('low','high','center','scale')},'model':model,
        'train_probability':pa,'evaluation_probability':pb,
        'meta_training_association_probability':expit(pp['xa']@beta+intercept),
        'available':bool(success and np.ptp(pa)>1e-12),'ScoreType':'consistent_habitat_stack_probability_uncalibrated'}


def infer(f,bases,tables,indices):
    if f['config']['base_mode']=='Full':
        x=matrix(bases['full'],tables,indices)
    else:
        x=sum(matrix(row['models'],tables,indices) for row in bases['bag'])/15.
    z=(np.clip(x,f['low'],f['high'])-f['center'])/f['scale']
    return expit(z@f['coef']+f['intercept'])


def choose(grid):
    valid=[r for r in grid if r['AvailableAllFolds']]
    feasible=[r for r in valid if r['MeanFitAUC']>=.75]
    pool=feasible or valid
    assert pool
    return max(pool,key=lambda r:(r['Objective'],-r['anchor'],r['base_mode']=='Full',-r['candidate_id']))


def train():
    protect();assert not (OUT/'training_frozen.json').exists() and not (OUT/'test_started.json').exists()
    cohort,y,tr,te=inputs();tables=joblib.load(previous.parent.OUT/'Random120/training_features.joblib')
    oldstack=joblib.load(previous.OUT/'Random120/models/stack12.joblib')
    metrics=[];predictions=[];choices=[];grids=[];audits=[]
    for phase in PHASES:
        checkpoint=OUT/'models'/(phase+'_complete.joblib')
        if checkpoint.exists():
            print(phase+' phasecheckpoint preserved',flush=True);continue
        a,b=(tr,np.empty(0,int)) if phase=='final' else previous.parent.folds('Random120')[int(phase[-1])-1]
        entries=[]
        for rep,k,aa,bb in previous.parent.folds('Random120',phase):
            name=phase+'_r%df%d'%(rep,k)
            ctx=make_context(name,tables,aa,bb,y,deep_folds(phase,rep,k,aa,y))
            for c in options():
                f=meta_fit(ctx,y,c)
                entries.append({'Phase':phase,'Repeat':rep,'Fold':k,'candidate_id':c['candidate_id'],
                    'FitAUC':float(roc_auc_score(y[aa],f['train_probability'])),
                    'AUC':float(roc_auc_score(y[bb],f['evaluation_probability'])),'Available':f['available']})
            print(name+' strictnestedstack context complete',flush=True)
        frame=pd.DataFrame(entries);grid=[]
        for c in options():
            rr=frame[frame.candidate_id==c['candidate_id']];mean,sd,fit=rr.AUC.mean(),rr.AUC.std(ddof=1),rr.FitAUC.mean()
            grid.append({**c,'Phase':phase,'MeanInnerAUC':float(mean),'InnerSD':float(sd),'MeanFitAUC':float(fit),
                'AvailableAllFolds':bool(rr.Available.all()),'Objective':float(mean-.1*sd-.3*max(fit-mean-.1,0))})
        chosen=choose(grid);allowed,source_oof,_=previous.crossfit_features('Random120',phase)
        np.testing.assert_array_equal(a,allowed)
        ctx=make_context(phase,tables,a,b,y,previous.parent.folds('Random120',phase),source_oof,True)
        fitted={c['candidate_id']:meta_fit(ctx,y,c) for c in options()}
        sf=oldstack['final'] if phase=='final' else oldstack['outer'][int(phase[-1])-1]
        # SignedFull anchor0 must be numerically identical to the immutable previousStack12.
        np.testing.assert_allclose(fitted[0]['train_probability'],sf['train_probability'],atol=1e-9,rtol=0)
        np.testing.assert_allclose(fitted[0]['evaluation_probability'],sf['evaluation_probability'],atol=1e-9,rtol=0)
        result={'phase':phase,'inner_audit':entries,'grid':grid,'selected':chosen,'models':fitted}
        joblib.dump(result,checkpoint,compress=3)
        print(phase+' eight recipes refitted;chosen%d;no test prediction'%chosen['candidate_id'],flush=True)
    for phase in PHASES:
        result=joblib.load(OUT/'models'/(phase+'_complete.joblib'));grids.extend(result['grid']);audits.extend(result['inner_audit'])
        choices.append(result['selected'])
        if phase=='final':continue
        k=int(phase[-1]);a,b=previous.parent.folds('Random120')[k-1]
        labels=[('Adaptive8',result['models'][result['selected']['candidate_id']])]+[('Fixed%d'%cid,f) for cid,f in result['models'].items()]
        for label,f in labels:
            metrics.append({'ModelKey':label,'Fold':k,'candidate_id':f['config']['candidate_id'],
                'FitAUC':float(roc_auc_score(y[a],f['train_probability'])),'AUC':float(roc_auc_score(y[b],f['evaluation_probability']))})
            predictions.extend({'ModelKey':label,'Fold':k,'Index':int(i),'AnalysisID':cohort.AnalysisID.iloc[i],
                'Outcome':int(y[i]),'Probability':float(p)} for i,p in zip(b,f['evaluation_probability']))
    final=joblib.load(OUT/'models/final_complete.joblib');selected=final['models'][final['selected']['candidate_id']]
    pp=pd.DataFrame(predictions);pp=pp[pp.ModelKey=='Adaptive8'].set_index('Index').loc[tr]
    fp,tp,thresholds=roc_curve(y[tr],pp.Probability,drop_intermediate=False);selected['threshold']=float(thresholds[np.argmax(tp-fp)])
    selected['threshold_source']='fullpipeline_outerOOF84_Youden'
    joblib.dump(selected,OUT/'models/final_primary.joblib',compress=3)
    summaries=[];mm=pd.DataFrame(metrics)
    for label,f in [('Adaptive8',selected)]+[('Fixed%d'%cid,f) for cid,f in final['models'].items()]:
        rr=mm[mm.ModelKey==label];pv=pd.DataFrame(predictions);pv=pv[pv.ModelKey==label]
        summaries.append({'ModelKey':label,'FinalCandidateID':f['config']['candidate_id'],**f['config'],
            'TrainFitAUC':float(roc_auc_score(y[tr],f['train_probability'])),
            'OuterMeanAUC':float(rr.AUC.mean()),'OuterSD':float(rr.AUC.std(ddof=1)),
            'PooledOOFAUC':float(roc_auc_score(pv.Outcome,pv.Probability)),
            'MetaInSampleOOFAssociationAUC':float(roc_auc_score(y[tr],f['meta_training_association_probability'])),
            'TargetBoth075':bool(roc_auc_score(y[tr],f['train_probability'])>=.75 and rr.AUC.mean()>=.75)})
    csv('training_summary',summaries);csv('outer_metrics',metrics);csv('nested_predictions',predictions)
    csv('inner_scores',audits);csv('inner_candidate_summary',grids);csv('selected_configurations',choices)
    files=[p for p in OUT.rglob('*') if p.is_file()]
    save(OUT/'training_frozen.json',{'passed':True,'new_test_calls':0,'protocol_sha256':digest(OUT/'protocol.json'),
        'files':{str(p.relative_to(OUT)):digest(p) for p in files}})
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)


def frozen():
    protect();f=json.loads((OUT/'training_frozen.json').read_text(encoding='utf-8'))
    assert f['passed'] and f['protocol_sha256']==digest(OUT/'protocol.json')
    for name,sha in f['files'].items():assert digest(OUT/name)==sha,name
    return f


def test():
    frozen();assert not (OUT/'test_started.json').exists()
    assert json.loads((OUT/'training_verification.json').read_text(encoding='utf-8'))['passed']
    cohort,_,tr,te=inputs();f=joblib.load(OUT/'models/final_primary.joblib');bases=joblib.load(OUT/'models/final_bases.joblib')
    tables=joblib.load(previous.OUT/'Random120/test_features.joblib')
    save(OUT/'test_started.json',{'new_complete_model_calls':1,'test_stages':1,'primary_candidate':f['config'],
        'secondary_test_calls':0,'freeze_sha256':digest(OUT/'training_frozen.json')})
    p=infer(f,bases,tables,te);truth=cohort.Outcome.to_numpy(int)[te]
    from verify_ivim_df63_20261020 import draws
    lo,hi=np.quantile(draws(truth,{'Adaptive8':p},2026102491)['Adaptive8'],[.025,.975])
    tn,fp,fn,tp=confusion_matrix(truth,p>=f['threshold'],labels=[0,1]).ravel()
    row={'ModelKey':'Adaptive8','N':36,'AUC':float(roc_auc_score(truth,p)),'AUC_CI95Low':float(lo),'AUC_CI95High':float(hi),
        'Threshold':f['threshold'],'Sensitivity':float(tp/(tp+fn)),'Specificity':float(tn/(tn+fp)),
        'Accuracy':float((tp+tn)/36),'TP':int(tp),'FN':int(fn),'TN':int(tn),'FP':int(fp)}
    csv('test_metrics',[row]);csv('test_predictions',[{'Index':int(i),'AnalysisID':cohort.AnalysisID.iloc[i],
        'Outcome':int(t),'Probability':float(v),'Prediction':int(v>=f['threshold'])} for i,t,v in zip(te,truth,p)])
    save(OUT/'test_complete.json',{'passed':True,'new_complete_model_calls':1,'secondary_test_calls':0,'post_test_fits':0,'post_test_threshold_changes':0})
    print(pd.DataFrame([row]).to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('define','train','test'))
    with threadpool_limits(1):globals()[p.parse_args().action]()
