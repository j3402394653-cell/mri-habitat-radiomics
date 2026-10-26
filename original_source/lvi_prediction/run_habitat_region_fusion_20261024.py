# -*- coding: utf-8 -*-
"""One-mm-only A feature concatenation and C strict OOF score fusion."""
import os
os.environ.update({'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata,norm
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score,roc_curve,confusion_matrix
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
import run_habitat_margin_20261024 as previous
import habitat_region_fusion_features_20261024 as feature

ROOT,save,digest=previous.ROOT,previous.save,previous.digest
OUT=ROOT/'lvi_prediction/results/lvi_habitat_region_fusion_20261024'
ICCS=(.60,.75)
CS=(.03,.1,.3,1.,3.)
FUSION_CS=(.1,.3,1.)
SCOPES=feature.REGIONS+('Median','IntraCore','A1')
KEYS=SCOPES+('C1',)
REFERENCES={
    'nested_cv':'https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html',
    'stacking':'https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html',
    'DeLong':'https://pubmed.ncbi.nlm.nih.gov/3203132/'}


def csv(name,rows):pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')
def read(name):return pd.read_csv(OUT/(name+'.csv'),float_precision='round_trip')
def model_path(phase):return OUT/'models'/(phase+'.joblib')


def protect(images=False):
    p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    for field in ('protected_hashes','source_hashes'):
        for name,sha in p[field].items():assert digest(ROOT/name)==sha,name
    if images:
        for name,sha in p['image_hashes'].items():assert digest(ROOT/name)==sha,name
    return len(p['protected_hashes'])


def define():
    assert not OUT.exists();previous.frozen()
    assert json.loads((previous.OUT/'delivery_final_verification.json').read_text(encoding='utf-8'))['passed']
    parent=json.loads((previous.OUT/'protocol.json').read_text(encoding='utf-8'))
    protected={**parent['protected_hashes'],**parent['source_hashes']}
    protected.update({str(p.relative_to(ROOT)):digest(p) for p in previous.OUT.rglob('*') if p.is_file()})
    OUT.mkdir(parents=True)
    for name in ('patients','models','figures'):(OUT/name).mkdir()
    for label in ('Train','Test'):(OUT/(label+'_cases.csv')).write_bytes((previous.OUT/(label+'_cases.csv')).read_bytes())
    files=[ROOT/'lvi_prediction'/n for n in ('run_habitat_region_fusion_20261024.py','habitat_region_fusion_features_20261024.py','report_habitat_region_fusion_20261024.py')]
    p={'date':'2026-10-24','authorization':'Evaluate screenshot A/B/C and actually try viable strategies;retain purehabitat focus.',
        'cohort':'SameA120,84train57+/27-,36test25+/11-;no resampling,patientdrop orfoldexclusion.',
        'pure_habitat':True,'clinical':False,'whole_predictors':False,'Dstar':False,'K':3,
        'regions':'Only1mm:Intra=geometrictumor;Core1=1mm-erodedtumor(overlapsIntra);Peri1=1mmdilationminusgeometrictumor,strictlyexterior. Eachacquisitionindependentspace;featurelevelfusion,no voxelregistration.',
        'repeatability':'Fororiginaltumorcontours0/-1/+1mm,rederiveEACHregionandrefitK3. ICCcomputedseparatelyWITHINsameregionacrossthese3variants;notbetweenIntra/Core/Peri. Thisisconditionalcontoursensitivity,nottest-retest.',
        'features':'Same321compactK3 featuresperregionpernormalization;PercentileZandRobustZprefixconcatenated642. A concatenatesIntra/Core1/Peri_r(1926);IntraCore1284secondarycontrol;no shape/clinical/whole/Dstar.',
        'normalization':'Originalpatientvalidtumorreferencefixedacrossregionsandperturbations;samenativequalityrulesandverifiedcommon1mm grid aspreviousstudy.',
        'QC':'Currenttrainingonly:completeness>=.9,ICCcutoff.60/.75(innerselected),nonconstant. Spearmanabs<.90;greedyretainhighercurrent-trainingunivariateabs(AUC-.5),thenICC,name. No fullcohortpvaluefilter,no fixedtopfeaturecap.',
        'transform':'TrainingmedianfillandGaussianquantiletransformclip[-2.5,2.5];eachfeaturecomparablepenaltyscale.',
        'A':'BinaryL1LogisticRegression,balanced,liblinear,C.03/.1/.3/1/3,10000iterations,tol1e-6. Jointfeatureselectionandweights;noordinarylinearLassoforbinarytarget. OnlyA1;ICC/CselectedininnerCV.',
        'B':'Notrun onlatestexplicitusersteering:only1mm,no2/3mmmodelsor tests,noscale selection.',
        'C':'3regionalL1bases(Intra,Core1,Peri_r),sharedcandidateICCandC.1/.3/1;independentlylearnedcoefficients/features. Meta balancedL2LogisticC.1onOOFregionalprobabilities,OOF-derivedclipping/meanSD. Inferenceaverages5crossfitbasesperregion(Bag5),nometaactualtraininganchor.',
        'C_strict_nesting':'ToselectCconfiguration,eachinnertrainingportiongeneratesitsOWNprivate5foldbaseOOFswithallQC/selectiontransformsfittedprivately. NooutertrainingwideOOFreusedasinnertrainingOOF. Metafittingneverusesbasein-samplepredictions. TrueTrainFitAUC comesfromactualBag5inferenceon84,notmetatrainingOOFassociation.',
        'folds':'Exactold5outerfolds. Thisnewstudyuses5innerfolds(thefirstrepeatofoldmanifest),notold5x3. PrivateCfolds:StratifiedKFold5,seed2026102407+weightedtrainingindexsum%1000000;fitonlycurrenttraininglabels.',
        'selection':'Availableall5innerfolds preferred;objectiveMeanAUC-.10SD;tie1e-12 strongerregularization(lowerC),higherICC,smallerradius. NofitAUCfloor/test-basedguard.',
        'primary':['A1','C1'],'controls':list(SCOPES),'frozen_models':7,'only_one_mm':True,
        'threshold':'Eachcompleteprocedureall84outerOOFYouden;frozenbeforeanynewtest.',
        'test':'All7prespecified1mmmodelsfrozenandverifiedbeforeonejoint36stage;allreported;no posttestretraining/rethresholding. Old1mmmedianstackmetricsreusedonlyashistoricalreference.',
        'statistics':'5000stratifiedpairedbootstrap;pairedDeLonginfluencecovariancewithties;Holmcorrectionfor2primaryvsIntracomparisons. Notusetestsforselection;noNRI/IDIprimaryclaim.',
        'missing_regions':'Neverdroppatient;missingregionfeaturesgated/foldmedianfilled. Unavailablebaseconstant.5neutralbalancedscore. ReportnativeIVIMgeometriccoverageandregionemptiness.',
        'limitations':'A84has27negativeevents;smallcohort,manycorrelatedfeatures,CoreoverlapsIntra,oldA36historicallyreused,nogenuineexternalvalidation. Coefficientmagnitudesnotcausalcontributions.',
        'formal_adoption_changed':False,'references':REFERENCES,'protected_hashes':protected,
        'source_hashes':{str(p.relative_to(ROOT)):digest(p) for p in files},'image_hashes':parent['image_hashes']}
    assert len(KEYS)==7
    save(OUT/'protocol.json',p);save(OUT/'definition_frozen.json',{'passed':True,'protocol_sha256':digest(OUT/'protocol.json'),'test_calls':0})
    print('Defined1mmstrictregionstudy7models;same84/36;inner5;testunscored',flush=True)


def worker(task):
    with threadpool_limits(1):result=feature.extract_one(task)
    joblib.dump(result,OUT/'patients'/('%03d.joblib'%result['index']),compress=3)
    return result['index']


def extraction(stage,workers):
    protect(images=True);cohort,y,tr,te=previous.old.inputs();ix=tr if stage=='training' else te
    if stage=='test':
        frozen();assert json.loads((OUT/'training_verification.json').read_text(encoding='utf-8'))['passed']
        assert not (OUT/'test_started.json').exists()
    profiles=previous.loader.load_profiles()
    entries=json.loads((previous.loader.POOL/'source_manifest.json').read_text(encoding='utf-8'))['entries']
    tasks=[(int(i),cohort.AnalysisID.iloc[i],entries[int(i)],profiles[int(i)],joblib.load(previous.OUT/'patients'/('%03d.joblib'%i))) for i in ix if not (OUT/'patients'/('%03d.joblib'%i)).exists()]
    start=time.time();failures=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        fs={pool.submit(worker,t):t[0] for t in tasks}
        for done,ff in enumerate(as_completed(fs),1):
            try:ff.result()
            except Exception as exc:failures.append({'index':int(fs[ff]),'error':str(exc)});print('Failure'+str(failures[-1]),flush=True)
            if done%4==0 or done==len(tasks):print('%s separate-regions%d/%d %.0fs'%(stage,done,len(tasks),time.time()-start),flush=True)
    assert not failures,failures
    records=[joblib.load(OUT/'patients'/('%03d.joblib'%i)) for i in ix];metadata=records[0]['metadata'];tables={}
    for r in feature.REGIONS:
        meta=[{**m,'name':r+'__'+n+'__'+m['name'],'region':r,'normalization':n} for n in feature.NORMS for m in metadata]
        tables[r]={'indices':ix,'metadata':meta,**{k:np.vstack([np.r_[row['values'][('PercentileZ',r,k)],row['values'][('RobustZ',r,k)]] for row in records]) for k in ('base','erode','dilate')}}
    for pos,row in enumerate(records):
        assert row['metadata']==metadata and row['index']==ix[pos] and row['native_core_verified']
        # ExactexistingverifiedIntra andCore1basecachecomparisonbyfeature-name.
        cached=joblib.load(previous.OUT/'patients'/('%03d.joblib'%ix[pos]))
        for n in feature.NORMS:
            np.testing.assert_array_equal(row['values'][(n,'Intra','base')],cached['base_values'][n])
            np.testing.assert_array_equal(row['values'][(n,'Core1','base')],cached['values'][(n,1.,'erode')])
    geometry=[r for row in records for r in row['geometry_audit'] if r['Region'] in feature.REGIONS]
    for mod in feature.prior.MODS:
        for aid in cohort.AnalysisID.iloc[ix]:
            z=[r for r in geometry if r['AnalysisID']==aid and r['Modality']==mod]
            assert len(z)==len(feature.REGIONS)*3 and all(r['ReferenceEdgeVoxels']==0 for r in z)
    joblib.dump(tables,OUT/(stage+'_features.joblib'),compress=3)
    csv(stage+'_geometry_audit',geometry);csv(stage+'_texture_audit',[r for row in records for r in row['texture_audit'] if r['Region'] in feature.REGIONS])
    save(OUT/(stage+'_feature_verification.json'),{'passed':True,'patients':len(ix),'regions':3,'dimensions_per_region':642,
        'strict_rings_exclude_geometric_tumor':True,'Core_subset_Intra':True,'only_one_mm':True,
        'Intra_Corebase_match_existing_exactly':True,'within_region_only_perturbations':True,'patient_exclusions':0})
    print(stage+' separate-regionfeaturesverified',flush=True)


def row_indices(d,ix):
    lookup={int(i):j for j,i in enumerate(d['indices'])};return [lookup[int(i)] for i in ix]


class Engine:
    def __init__(self,tables,y):
        self.tables=dict(tables);self.y=y;self.qc={};self.prep={};self.fits={};self.contexts={};self.foldaudit={}
        intra=tables['Intra']
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            med=np.nanmedian(np.stack([intra[k] for k in ('base','erode','dilate')]),axis=0)
        self.tables['Median']={**intra,'base':med,'quality_source':'Intra'}
        for scope,regions in [('IntraCore',('Intra','Core1')),('A1',('Intra','Core1','Peri1'))]:
            self.tables[scope]={'indices':intra['indices'],'metadata':[m for r in regions for m in tables[r]['metadata']],
                **{k:np.concatenate([tables[r][k] for r in regions],axis=1) for k in ('base','erode','dilate')},'regions':regions}

    def folds(self,a):
        a=np.asarray(a,int);token=tuple(a)
        if token in self.foldaudit:return self.foldaudit[token]
        _,_,tr,_=previous.old.inputs();found=None
        for phase in previous.old.PHASES:
            aa=tr if phase=='final' else previous.old.previous.parent.folds('Random120')[int(phase[-1])-1][0]
            if np.array_equal(aa,a):found=[(x,z) for rep,k,x,z in previous.old.previous.parent.folds('Random120',phase) if rep==1];break
        if found is None:
            seed=2026102407+sum((j+1)*(int(i)+1) for j,i in enumerate(a))%1000000
            found=[(a[x],a[z]) for x,z in StratifiedKFold(5,shuffle=True,random_state=seed).split(np.zeros(len(a)),self.y[a])]
        for x,z in found:assert set(x).isdisjoint(z) and set(x)|set(z)==set(a)
        self.foldaudit[token]=found;return found

    def quality(self,scope,a):
        token=(scope,tuple(a))
        if token in self.qc:return self.qc[token]
        d=self.tables[scope]
        if scope=='Median':result=self.quality('Intra',a)
        elif 'regions' in d:
            q=[self.quality(r,a) for r in d['regions']];result=(np.concatenate([x[0] for x in q]),np.concatenate([x[1] for x in q]))
        else:result=previous.backend.quality(d,a)
        self.qc[token]=result;return result

    def prepare(self,scope,a,icc):
        token=(scope,tuple(a),icc)
        if token in self.prep:return self.prep[token]
        d=self.tables[scope];x=d['base'][row_indices(d,a)];complete,stability=self.quality(scope,a)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning);median=np.nanmedian(np.where(np.isfinite(x),x,np.nan),axis=0)
        median=np.where(np.isfinite(median),median,0.);filled=np.where(np.isfinite(x),x,median)
        eligible=np.flatnonzero((complete>=.9)&np.isfinite(stability)&(stability>=icc)&(filled.std(0)>1e-12))
        selected=[];association={}
        if len(eligible):
            ranks=rankdata(filled[:,eligible],axis=0);target=self.y[a];np1=int(target.sum());nn=len(a)-np1
            aucs=(ranks[target==1].sum(0)-np1*(np1+1)/2)/(np1*nn)
            association={int(j):float(abs(v-.5)) for j,v in zip(eligible,aucs)}
            corr=np.atleast_2d(np.corrcoef(ranks,rowvar=False));lookup={int(j):k for k,j in enumerate(eligible)}
            ordered=sorted(eligible,key=lambda j:(-association[int(j)],-stability[j],d['metadata'][j]['name']))
            for j in ordered:
                if all(abs(corr[lookup[int(j)],lookup[k]])<.90 for k in selected):selected.append(int(j))
        qt=None;xa=filled[:,selected]
        if selected:
            qt=QuantileTransformer(n_quantiles=len(a),output_distribution='normal',subsample=len(a),random_state=64212).fit(xa)
            xa=np.clip(qt.transform(xa),-2.5,2.5)
        p={'scope':scope,'train':np.asarray(a,int),'icc':icc,'indices':selected,'names':[d['metadata'][j]['name'] for j in selected],
            'median':median[selected],'quantile':qt,'xa':xa,'eligible_count':len(eligible)}
        self.prep[token]=p;return p

    def fixed(self,scope,a,icc,C):
        token=(scope,tuple(a),icc,C)
        if token in self.fits:return self.fits[token]
        p=self.prepare(scope,a,icc);model=None;converged=True
        if p['indices']:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always',ConvergenceWarning)
                model=LogisticRegression(C=C,penalty='l1',solver='liblinear',class_weight='balanced',max_iter=10000,tol=1e-6,random_state=64212).fit(p['xa'],self.y[a])
            converged=not any(issubclass(w.category,ConvergenceWarning) for w in caught)
        beta=np.zeros(len(p['indices'])) if model is None else model.coef_[0].copy()
        intercept=0. if model is None else float(model.intercept_[0]);pa=expit(p['xa']@beta+intercept)
        f={k:p[k] for k in ('scope','train','icc','indices','names','median','quantile','eligible_count')}
        f.update(C=C,model=model,coef=beta,intercept=intercept,available=bool(converged and np.ptp(pa)>1e-12),
            converged=converged,effective=int((np.abs(beta)>1e-12).sum()),train_probability=pa,kind='L1',ScoreType='balanced_L1_logistic_uncalibrated')
        self.fits[token]=f;return f

    def context(self,a,icc,C):
        token=(tuple(a),icc,C)
        if token in self.contexts:return self.contexts[token]
        lookup={int(i):j for j,i in enumerate(a)};oof=np.full((len(a),len(feature.REGIONS)),np.nan);actual=np.zeros((len(a),len(feature.REGIONS)));bags=[]
        for k,(aa,bb) in enumerate(self.folds(a),1):
            members=[self.fixed(r,aa,icc,C) for r in feature.REGIONS]
            s=np.column_stack([predict_l1(f,self.tables,a) for f in members]);rr=[lookup[int(i)] for i in bb]
            oof[rr]=s[rr];actual+=s/5.;bags.append({'Fold':k,'train':aa,'validation':bb,'members':members})
        assert np.isfinite(oof).all()
        result={'train':np.asarray(a,int),'oof':oof,'actual':actual,'bags':bags};self.contexts[token]=result;return result

    def fusion(self,a,icc,C,r):
        ctx=self.context(a,icc,C);columns=[0,1,1+r];x=ctx['oof'][:,columns]
        low,high=np.quantile(x,[.025,.975],axis=0);z=np.clip(x,low,high);mu=z.mean(0);sd=z.std(0);sd=np.where(sd>1e-12,sd,1.)
        z=(z-mu)/sd
        lr=LogisticRegression(C=.1,penalty='l2',solver='liblinear',class_weight='balanced',max_iter=5000,random_state=64212).fit(z,self.y[a])
        za=(np.clip(ctx['actual'][:,columns],low,high)-mu)/sd;pa=lr.predict_proba(za)[:,1]
        return {'kind':'Fusion','train':np.asarray(a,int),'icc':icc,'C':C,'radius':r,'low':low,'high':high,'center':mu,'scale':sd,
            'coef':lr.coef_[0].copy(),'intercept':float(lr.intercept_[0]),'meta':lr,'columns':columns,
            'bags':[{'Fold':b['Fold'],'train':b['train'],'validation':b['validation'],'members':[b['members'][j] for j in columns]} for b in ctx['bags']],
            'train_probability':pa,'meta_training_association':lr.predict_proba(z)[:,1],
            'available':bool(np.ptp(pa)>1e-12),'effective':3,'ScoreType':'OOF_L2_meta_Bag5_regional_L1_uncalibrated'}


def transformed_l1(f,tables,ix):
    d=tables[f['scope']];x=d['base'][row_indices(d,ix)][:,f['indices']];x=np.where(np.isfinite(x),x,f['median'])
    return np.clip(f['quantile'].transform(x),-2.5,2.5) if len(f['indices']) and len(ix) else np.empty((len(ix),len(f['indices'])))


def predict_l1(f,tables,ix):
    if f['model'] is None:return np.full(len(ix),.5)
    return f['model'].predict_proba(transformed_l1(f,tables,ix))[:,1]


def predict(f,tables,ix):
    if f['kind']=='L1':return predict_l1(f,tables,ix)
    x=sum(np.column_stack([predict_l1(m,tables,ix) for m in b['members']]) for b in f['bags'])/5.
    z=(np.clip(x,f['low'],f['high'])-f['center'])/f['scale']
    return f['meta'].predict_proba(z)[:,1]


def choose(rows):
    valid=[r for r in rows if r['AvailableAllFolds']];pool=valid or rows
    return sorted(pool,key=lambda r:(-round(r['Objective'],12),r['C'],-r['icc'],r.get('radius',0)))[0]


def phase(engine,ph,y,tr):
    path=model_path(ph)
    if path.exists():return joblib.load(path)
    a,b=(tr,np.empty(0,int)) if ph=='final' else previous.old.previous.parent.folds('Random120')[int(ph[-1])-1]
    entries=[]
    for fold,(aa,bb) in enumerate(engine.folds(a),1):
        for scope in SCOPES:
            for icc in ICCS:
                for C in CS:
                    f=engine.fixed(scope,aa,icc,C);p=predict(f,engine.tables,bb)
                    entries.append({'Family':scope,'Fold':fold,'icc':icc,'C':C,'AUC':roc_auc_score(y[bb],p),
                        'FitAUC':roc_auc_score(y[aa],f['train_probability']),'Available':f['available'],'Effective':f['effective']})
        for icc in ICCS:
            for C in FUSION_CS:
                for r in (1,):
                    f=engine.fusion(aa,icc,C,r);p=predict(f,engine.tables,bb)
                    entries.append({'Family':'C%d'%r,'Fold':fold,'icc':icc,'C':C,'radius':r,'AUC':roc_auc_score(y[bb],p),
                        'FitAUC':roc_auc_score(y[aa],f['train_probability']),'Available':f['available'],'Effective':3})
        print(ph+' inner%d/5 A/B/C privateOOFdone'%fold,flush=True)
    frame=pd.DataFrame(entries);grids={};selected={};models={}
    for family in SCOPES+('C1',):
        rows=[]
        for icc in ICCS:
            for C in (FUSION_CS if family.startswith('C') and family!='Core1' else CS):
                rr=frame[(frame.Family==family)&(frame.icc==icc)&(frame.C==C)]
                mean,sd=float(rr.AUC.mean()),float(rr.AUC.std(ddof=1))
                row={'Family':family,'icc':icc,'C':C,'MeanInnerAUC':mean,'InnerSD':sd,'MeanFitAUC':float(rr.FitAUC.mean()),
                    'AvailableAllFolds':bool(rr.Available.all()),'MeanEffective':float(rr.Effective.mean()),'Objective':mean-.10*sd}
                if family in ('C1','A1'):row['radius']=1
                rows.append(row)
        grids[family]=rows;selected[family]=choose(rows)
    for name,row in selected.items():
        family=row['Family'];models[name]=(engine.fusion(a,row['icc'],row['C'],1) if family=='C1' else engine.fixed(family,a,row['icc'],row['C'])).copy()
        models[name]['validation']=b
    cp={'phase':ph,'train':a,'validation':b,'entries':entries,'grids':grids,'selected':selected,'models':models,
        'evaluation':{name:predict(f,engine.tables,b) if len(b) else np.empty(0) for name,f in models.items()},
        'fold_audit':[{'train':list(k),'folds':[{'train':x.tolist(),'validation':z.tolist()} for x,z in v]} for k,v in engine.foldaudit.items()]}
    joblib.dump(cp,path,compress=3);print(ph+' selected7procedures;only1mm;notest',flush=True);return cp


def train():
    protect();assert not (OUT/'training_frozen.json').exists() and not (OUT/'test_started.json').exists()
    assert json.loads((OUT/'training_feature_verification.json').read_text(encoding='utf-8'))['passed']
    cohort,y,tr,te=previous.old.inputs();engine=Engine(joblib.load(OUT/'training_features.joblib'),y)
    metrics=[];predictions=[];choices=[];inner=[];features=[]
    for ph in previous.old.PHASES:
        cp=phase(engine,ph,y,tr);choices.extend({'Phase':ph,'ModelKey':n,**r} for n,r in cp['selected'].items());inner.extend({'Phase':ph,**r} for r in cp['entries'])
        if ph=='final':continue
        a,b=cp['train'],cp['validation']
        for name in KEYS:
            f=cp['models'][name];p=cp['evaluation'][name]
            metrics.append({'ModelKey':name,'Fold':int(ph[-1]),'FitAUC':roc_auc_score(y[a],f['train_probability']),'AUC':roc_auc_score(y[b],p)})
            predictions.extend({'ModelKey':name,'Fold':int(ph[-1]),'Index':int(i),'AnalysisID':cohort.AnalysisID.iloc[i],'Outcome':int(y[i]),'Probability':float(z)} for i,z in zip(b,p))
    mm=pd.DataFrame(metrics);pp=pd.DataFrame(predictions);final=joblib.load(model_path('final'));summary=[]
    for name in KEYS:
        f=final['models'][name];pv=pp[pp.ModelKey==name].set_index('Index').loc[tr];rr=mm[mm.ModelKey==name]
        x,z,thresholds=roc_curve(y[tr],pv.Probability,drop_intermediate=False);f['threshold']=float(thresholds[np.argmax(z-x)])
        f['threshold_source']='all84_fullpipeline_outerOOF_Youden'
        s={'ModelKey':name,'TrainFitAUC':roc_auc_score(y[tr],f['train_probability']),'OuterMeanAUC':rr.AUC.mean(),'OuterSD':rr.AUC.std(ddof=1),
            'PooledOOFAUC':roc_auc_score(y[tr],pv.Probability),'Threshold':f['threshold'],'Family':final['selected'][name]['Family'],
            'ICC':f['icc'],'C':f['C'],'Radius':final['selected'][name].get('radius',int(name[-1]) if name.startswith('Peri') else 0),
            'MetaInSampleOOFAssociationAUC':roc_auc_score(y[tr],f['meta_training_association']) if f['kind']=='Fusion' else np.nan}
        summary.append(s)
        if f['kind']=='L1':
            features.extend({'ModelKey':name,'Feature':n,'Region':n.split('__')[0],'CoefficientGaussian':float(beta),'Nonzero':bool(abs(beta)>1e-12)} for n,beta in zip(f['names'],f['coef']))
    joblib.dump(final,model_path('final'),compress=3)
    csv('training_summary',summary);csv('outer_metrics',metrics);csv('nested_predictions',predictions);csv('selected_configurations',choices);csv('inner_scores',inner);csv('final_L1_features',features)
    files={str(p.relative_to(OUT)):digest(p) for p in OUT.rglob('*') if p.is_file()}
    save(OUT/'training_frozen.json',{'passed':True,'test_calls':0,'files':files})
    print(pd.DataFrame(summary).to_string(index=False),flush=True)


def frozen():
    protect();p=json.loads((OUT/'training_frozen.json').read_text(encoding='utf-8'))
    for name,sha in p['files'].items():assert digest(OUT/name)==sha,name
    return p


def manual(f,tables,ix):
    if f['kind']=='L1':return expit(transformed_l1(f,tables,ix)@f['coef']+f['intercept'])
    x=np.mean([np.column_stack([manual(m,tables,ix) for m in b['members']]) for b in f['bags']],axis=0)
    z=(np.clip(x,f['low'],f['high'])-f['center'])/f['scale']
    return expit(z@f['coef']+f['intercept'])


def verify():
    frozen();assert not (OUT/'test_started.json').exists()
    cohort,y,tr,te=previous.old.inputs();engine=Engine(joblib.load(OUT/'training_features.joblib'),y);count=0;err=0.
    for ph in previous.old.PHASES:
        cp=joblib.load(model_path(ph));a,b=cp['train'],cp['validation'];assert set(a).isdisjoint(b) and not set(a)&set(te)
        frame=pd.DataFrame(cp['entries'])
        for name,f in cp['models'].items():
            row=cp['selected'][name]
            pool=cp['grids'][row['Family']]
            assert choose(pool)==row
            pa=predict(f,engine.tables,a);np.testing.assert_allclose(pa,f['train_probability'],atol=1e-10,rtol=0)
            ma=manual(f,engine.tables,a);err=max(err,float(np.max(np.abs(ma-pa))));np.testing.assert_allclose(ma,pa,atol=1e-10,rtol=0)
            if len(b):np.testing.assert_allclose(predict(f,engine.tables,b),cp['evaluation'][name],atol=1e-10,rtol=0)
            if f['kind']=='Fusion':
                for bag in f['bags']:
                    assert set(bag['train']).isdisjoint(bag['validation']) and set(bag['train'])|set(bag['validation'])==set(a)
                    for m in bag['members']:np.testing.assert_array_equal(m['train'],bag['train'])
                ctx=engine.context(a,f['icc'],f['C']);cols=f['columns'];x=ctx['oof'][:,cols]
                low,high=np.quantile(x,[.025,.975],axis=0);mu=np.clip(x,low,high).mean(0);sd=np.clip(x,low,high).std(0);sd=np.where(sd>1e-12,sd,1.)
                lr=LogisticRegression(C=.1,penalty='l2',solver='liblinear',class_weight='balanced',max_iter=5000,random_state=64212).fit((np.clip(x,low,high)-mu)/sd,y[a])
                np.testing.assert_allclose(lr.coef_[0],f['coef'],atol=1e-10,rtol=0)
                for aa,bb in engine.folds(a):
                    innerctx=engine.context(aa,f['icc'],f['C'])
                    for bag in innerctx['bags']:assert set(bag['train']).isdisjoint(bag['validation']) and not (set(bag['train'])|set(bag['validation']))&set(bb)
            count+=1
        for g in cp['grids'].values():
            for row in g:
                z=frame[(frame.Family==row['Family'])&(frame.icc==row['icc'])&(frame.C==row['C'])]
                np.testing.assert_allclose([row['MeanInnerAUC'],row['InnerSD'],row['Objective']],[z.AUC.mean(),z.AUC.std(ddof=1),z.AUC.mean()-.10*z.AUC.std(ddof=1)],atol=1e-12)
    # Currentheldoutfeaturevaluesandlabels cannotalter preprocessing or coefficients.
    a,b=previous.old.previous.parent.folds('Random120')[0];changed={}
    original=joblib.load(OUT/'training_features.joblib')
    for scope,d in original.items():
        changed[scope]={**d,**{k:d[k].copy() for k in ('base','erode','dilate')}};rr=row_indices(d,b)
        for k in ('base','erode','dilate'):changed[scope][k][rr]=1e9
    yy=y.copy();yy[b]=1-yy[b];other=Engine(changed,yy)
    for scope in SCOPES:
        f=engine.fixed(scope,a,.60,.3);g=other.fixed(scope,a,.60,.3)
        assert f['indices']==g['indices'];np.testing.assert_array_equal(f['coef'],g['coef']);np.testing.assert_array_equal(f['train_probability'],g['train_probability'])
    save(OUT/'training_verification.json',{'passed':True,'selected_models_audited':count,'maximum_formula_error':err,
        'C_private_OOF_and_meta_refits_verified':True,'all_inner_grid_recalculated':True,'heldout_mutation_checks':len(SCOPES),
        'all5outerfolds_retained':True,'test_calls':0})
    print('Trainingverified7models;only1mm;strictprivateOOF;all5foldsretained',flush=True)


def delong(y,p,q):
    rows=[];cols=[];values=[]
    for s in (p,q):
        delta=np.subtract.outer(np.asarray(s)[y==1],np.asarray(s)[y==0]);phi=(delta>0)+.5*(delta==0)
        rows.append(phi.mean(1));cols.append(phi.mean(0));values.append(phi.mean())
    cov=np.cov(rows,ddof=1)/sum(y==1)+np.cov(cols,ddof=1)/sum(y==0)
    variance=max(float(np.array([1.,-1.])@cov@np.array([1.,-1.])),0.);difference=float(values[0]-values[1])
    pvalue=float(2*norm.sf(abs(difference)/np.sqrt(variance))) if variance>1e-20 else 1. if abs(difference)<1e-12 else np.nan
    return difference,variance,pvalue


def test():
    frozen();assert not (OUT/'test_started.json').exists()
    for n in ('training_verification','test_feature_verification'):assert json.loads((OUT/(n+'.json')).read_text(encoding='utf-8'))['passed']
    c,_,tr,te=previous.old.inputs();truth=c.Outcome.to_numpy(int)[te];engine=Engine(joblib.load(OUT/'test_features.joblib'),np.full(138,-1,int));final=joblib.load(model_path('final'))
    from verify_ivim_df63_20261020 import auc,draws
    save(OUT/'test_started.json',{'joint_stages':1,'prespecified_models':7,'only_one_mm':True,'post_test_fits':0})
    scores={};metrics=[];predictions=[];error=0.
    for name in KEYS:
        f=final['models'][name];p=predict(f,engine.tables,te);q=manual(f,engine.tables,te);np.testing.assert_allclose(p,q,atol=1e-10,rtol=0);error=max(error,float(np.max(np.abs(p-q))))
        assert np.isfinite(p).all();scores[name]=p
        np.testing.assert_allclose(auc(truth,p),roc_auc_score(truth,p),atol=1e-12,rtol=0)
        tn,fp,fn,tp=confusion_matrix(truth,p>=f['threshold'],labels=[0,1]).ravel()
        metrics.append({'ModelKey':name,'TestAUC':roc_auc_score(truth,p),'Threshold':f['threshold'],'Sensitivity':tp/(tp+fn),'Specificity':tn/(tn+fp),
            'Accuracy':(tp+tn)/36.,'TP':int(tp),'FN':int(fn),'TN':int(tn),'FP':int(fp)})
        predictions.extend({'ModelKey':name,'Index':int(i),'AnalysisID':c.AnalysisID.iloc[i],'Outcome':int(t),'Probability':float(v),'Prediction':int(v>=f['threshold'])} for i,t,v in zip(te,truth,p))
    samples=draws(truth,scores,20261024117)
    for row in metrics:
        n=row['ModelKey'];lo,hi=np.quantile(samples[n],[.025,.975]);dl,dh=np.quantile(samples[n]-samples['Intra'],[.025,.975]);delta,var,p=delong(truth,scores[n],scores['Intra'])
        row.update(AUC_CI95Low=float(lo),AUC_CI95High=float(hi),DifferenceVsIntra=float(delta),PairedDifferenceCI95Low=float(dl),PairedDifferenceCI95High=float(dh),DeLongPVsIntra=p)
    csv('test_metrics',metrics);csv('test_predictions',predictions)
    read('training_summary').merge(pd.DataFrame(metrics).drop(columns='Threshold'),on='ModelKey').to_csv(OUT/'comparison.csv',index=False,encoding='utf-8-sig')
    paired=[]
    groups={'primary':[('A1','Intra'),('C1','Intra')]}
    for group,pairs in groups.items():
        items=[]
        for n,m in pairs:
            diff,var,p=delong(truth,scores[n],scores[m]);lo,hi=np.quantile(samples[n]-samples[m],[.025,.975])
            items.append({'Group':group,'Model':n,'Reference':m,'AUC_Difference':diff,'PairedCI95Low':float(lo),'PairedCI95High':float(hi),'DeLongP':p})
        ordered=sorted(range(len(items)),key=lambda i:items[i]['DeLongP'] if np.isfinite(items[i]['DeLongP']) else 1.);maximum=0.
        for rank,i in enumerate(ordered):
            pv=items[i]['DeLongP'] if np.isfinite(items[i]['DeLongP']) else 1.
            maximum=max(maximum,(len(items)-rank)*pv);items[i]['HolmP']=min(maximum,1.)
        paired.extend(items)
    csv('paired_comparisons',paired);frozen()
    save(OUT/'test_complete.json',{'passed':True,'joint_test_stages':1,'models':7,'predictions':252,'only_one_mm':True,'independent_formula_maxerror':error,
        'post_test_fits':0,'post_test_threshold_changes':0,'formal_adoption_changed':False})
    print(read('comparison')[['ModelKey','TrainFitAUC','OuterMeanAUC','TestAUC','Sensitivity','Specificity']].to_string(index=False),flush=True)


def seal():
    frozen();protect(images=True)
    for n in ('training_verification','test_complete','report_verification','visual_verification'):assert json.loads((OUT/(n+'.json')).read_text(encoding='utf-8'))['passed']
    files={str(p.relative_to(OUT)):digest(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ('delivery_manifest.json','delivery_final_verification.json')}
    save(OUT/'delivery_manifest.json',{'outputs':files,'protected_files_unchanged':True})
    for name,sha in files.items():assert digest(OUT/name)==sha,name
    save(OUT/'delivery_final_verification.json',{'passed':True,'outputs_checked':len(files),'models':7,'only_one_mm':True,'joint_test_stages':1,'post_test_fits':0,'formal_adoption_changed':False})
    print('A/B/C regionfusion delivered;allpriorrecordsunchanged',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('define','extract_training','train','verify','extract_test','test','seal'));p.add_argument('--workers',type=int,default=4);args=p.parse_args()
    with threadpool_limits(1):
        if args.action.startswith('extract_'):extraction('training' if args.action=='extract_training' else 'test',args.workers)
        else:globals()[args.action]()
