# -*- coding: utf-8 -*-
"""Same habitats, direct PLS instead of the two-level probability stack."""
import os
os.environ.update({'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
import argparse
from datetime import datetime,timezone
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,confusion_matrix,roc_curve
from threadpoolctl import threadpool_limits
import run_dualroute_refinement_20261025 as first
from evaluate_original85_five_models_20261025 import clinical_reference
from verify_original_pls_direction_20261025 import independent_direction
from verify_random120_five_models_20261023 import manual

ROOT,save,digest = first.ROOT,first.save,first.digest
parent,source,base,five = first.parent,first.source,first.base,first.five
OUT = ROOT/'lvi_prediction/results/lvi_dualroute_directpls_20261025'
STREAMS,PHASES = first.STREAMS,first.PHASES
SEED = 20261025301


def csv(name,rows):
    pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')


def read(name):
    return pd.read_csv(OUT/(name+'.csv'),float_precision='round_trip')


def options():
    keys = list(dict.fromkeys(c['image_key'] for c in base.old.previous.members()))
    return [{'candidate_id':i,'image_key':key,'normalization':key.split('__')[0],
             'method':'PLS','components':n,'C':c} for i,(key,n,c) in enumerate(
                 (key,n,c) for key in keys for n in (1,2,3) for c in (.03,.3))]


def protect():
    p = json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    for field in ('source_hashes','protected_hashes'):
        for name,sha in p[field].items():
            assert digest(ROOT/name)==sha,name
    assert digest(OUT/'protocol.json')==json.loads((OUT/'definition_frozen.json').read_text(encoding='utf-8'))['protocol_sha256']


def frozen():
    protect()
    for name,sha in json.loads((OUT/'training_frozen.json').read_text(encoding='utf-8'))['files'].items():
        assert digest(OUT/name)==sha,name


def define():
    assert not OUT.exists()
    first.frozen()
    assert not (first.OUT/'test_started.json').exists(),'Declare second experiment before seeing first new tests'
    p = json.loads((first.OUT/'protocol.json').read_text(encoding='utf-8'))
    protected = {**p['protected_hashes'],**p['source_hashes']}
    for name in ('protocol.json','definition_frozen.json','training_frozen.json','training_summary.csv','fixed_recipe_summary.csv'):
        path = first.OUT/name
        protected[str(path.relative_to(ROOT))] = digest(path)
    OUT.mkdir(parents=True)
    for folder in ('phases','models','figures'):
        (OUT/folder).mkdir()
    save(OUT/'protocol.json',{'date':'2026-10-25','created_utc':datetime.now(timezone.utc).isoformat(),
        'authorization':'Continue both original/fixed85 routes,each pureHabitat andHabitatClinical;seekall5mean.75 andactualtrainingfitnottoolow.',
        'motivation_training_only':'First36recipe stackselection raisedtrainingfits butall4outermeansdropped;fixedrecipe diagnosticmaximaalsobelow.75. Thisexperiment declaredbeforefirst newtestscoring.',
        'unchanged':'ExactA12084train/36test,all5outer andinner5x3,1mmcontourmedian,k3,T2/ADC/IVIM_D/f,two normalizationsthreeviews,samefold-onlyQC/ICC/GaussianQT and40bootstrapdirectionfilter0/.85.',
        'new_architecture':'DirectGaussianhabitatfeaturePLS followedbybalancedL2LR;no secondlevel12probabilitystack ortrainOOFanchor. HC appends11preoperativeclinical variablesafterPLS withsamefold-onlyclinicaltransform.',
        'candidates':'36perstream:2norm x3views xPLS1/2/3 xC.03/.3. Direction0/.85fixed perroute,not selectedbetweenroutes. No rawfeaturecountcap.',
        'selection':'Eachoutertrainingpartowninner15folds. Availableall15;preferrootactualfit>=.75ifany,else>=.70,elseallavailablewithfailureflag. Same meanInnerAUC-.10SD-.30max(meanFit-meanInner-.10,0);tiesfewerPLScomponents,smallerC,ID.',
        'no_stacking':'Onlytwo-level nestedCV needed;allQC/direction/QT/PLS/clinicalfitinsidecurrentinnertrainingpart. No globaloutcome-driven featureselection.',
        'threshold':'Eachselectedpipeline owncomplete84outerOOFYouden,freezeallfour beforejoint36teststage.',
        'test':'ExactlyfournewdirectPLSpipelines,onejointstage;nofixedcandidate tests orposttestfit/thresholdtuning.',
        'uncertainty':{'bootstrap':5000,'seed':SEED,'method':'Class-stratifiedpairedfixed36scores'},
        'interpretation':'Samehistoricallyreusedpatients/folds/tests;conditionalexploratory,not newexternal/independentvalidation. Noadoptionchange.',
        'source_hashes':{str((ROOT/'lvi_prediction'/name).relative_to(ROOT)):digest(ROOT/'lvi_prediction'/name)
                         for name in ('run_dualroute_directpls_20261025.py','report_dualroute_studies_20261025.py')},
        'protected_hashes':protected})
    csv('candidate_definitions',options())
    (OUT/'split_membership.csv').write_bytes((parent.OUT/'split_membership.csv').read_bytes())
    save(OUT/'definition_frozen.json',{'passed':True,'new_test_calls':0,'first_new_test_not_started':True,'protocol_sha256':digest(OUT/'protocol.json')})
    print('Defined4directPLSstreams x36candidates,BEFOREfirstnewtests',flush=True)


def prepared(original,d,a,b,y):
    result = {}
    for key in d:
        p = source.prepared(original,d,a,key,y)
        for t in (0.,.85):
            result[(t,key)] = source.filtered(p,d[key],a,b,t)
    return result


def fitset(original,d,clinical,a,b,y):
    pp = prepared(original,d,a,b,y)
    cps = {joint:five.clinical_prepare(clinical,a,b,'Extended11' if joint else 'None') for joint in (False,True)}
    models = {key:{} for key,t,joint in STREAMS}
    reductions = {}
    for key,t,joint in STREAMS:
        for c in options():
            p = pp[(t,c['image_key'])]
            if not p['indices']:
                cc = {**c,'clinical_scope':'Extended11' if joint else 'None'}
                if joint:
                    f = five.fit_prepared(None,cps[joint],y[a],cc,a,b)
                else:
                    prevalence = float(y[a].mean())
                    f = {'config':cc,'train':a,'validation':b,'image':None,
                         'clinical':{name:cps[False][name] for name in ('indices','names','median','low','high','center','scale')},
                         'reducer':None,'latent_center':None,'latent_scale':None,'effective':0,'image_effective':0,
                         'classifier':None,'image_coef_gaussian':np.empty(0),'clinical_coef':np.empty(0),
                         'intercept':float(np.log(prevalence/(1-prevalence))),
                         'train_probability':np.full(len(a),prevalence),'evaluation_probability':np.full(len(b),prevalence),
                         'prevalence':prevalence,'ScoreType':'constant_prevalence_unavailable_image'}
                f.update(available=False,direction_threshold=t,direction_agreement=np.empty(0),zero_feature_fallback=True)
                models[key][c['candidate_id']] = f
                continue
            token = (t,c['image_key'],c['components'])
            if token not in reductions:
                reductions[token] = five.latent.reduction(p,y[a],c)
            cc = {**c,'clinical_scope':'Extended11' if joint else 'None'}
            f = five.fit_prepared(p,cps[joint],y[a],cc,a,b,reductions[token])
            f['direction_threshold'] = t
            f['direction_agreement'] = p['agreement']
            models[key][c['candidate_id']] = f
    return models


def choose(grid,models):
    adapted = [{**r,'channels':'Both','anchor':0.} for r in grid]
    valid = [r for r in adapted if r['AvailableAllFolds'] and models[r['candidate_id']]['available']]
    level = None;pool=valid
    for floor in (.75,.70):
        z=[r for r in valid if r['ActualRootFitAUC']>=floor-1e-12]
        if z:
            pool,level=z,floor
            break
    assert pool
    best=max(r['Objective'] for r in pool)
    selected=min([r for r in pool if r['Objective']>=best-1e-12],key=lambda r:(r['components'],r['C'],r['candidate_id']))
    return next(r for r in grid if r['candidate_id']==selected['candidate_id']),{'FitPreference':level,'TargetFitMet':level==.75}


def train():
    protect()
    assert not (OUT/'training_frozen.json').exists() and not (OUT/'test_started.json').exists()
    cohort,y,tr,te = base.old.inputs()
    tables = joblib.load(base.OUT/'training_features.joblib')[1.]
    d = base.route_tables(tables,'ContourMedian')
    clinical = joblib.load(parent.OUT/'training_clinical.joblib')
    entries,grids,choices,outer,predictions,summary = [],[],[],[],[],[]
    for ph in PHASES:
        dest=OUT/'phases'/(ph+'.joblib')
        if dest.exists():
            continue
        records=[]
        for rep,k,a,b in base.old.previous.parent.folds('Random120',ph):
            assert set(a).isdisjoint(b) and not (set(a)|set(b))&set(te)
            fitted=fitset(tables,d,clinical,a,b,y)
            for key,t,joint in STREAMS:
                for cid,f in fitted[key].items():
                    records.append({'ModelKey':key,'Phase':ph,'Repeat':rep,'Fold':k,'candidate_id':cid,
                         'FitAUC':roc_auc_score(y[a],f['train_probability']),'AUC':roc_auc_score(y[b],f['evaluation_probability']),
                         'Available':f['available']})
            if k==5:
                print(ph+' directPLS innerrepeat%d/3done'%rep,flush=True)
        a,b=(tr,np.empty(0,int)) if ph=='final' else base.old.previous.parent.folds('Random120')[int(ph[-1])-1]
        fitted=fitset(tables,d,clinical,a,b,y)
        frame=pd.DataFrame(records);states={}
        for key,t,joint in STREAMS:
            grid=[]
            for c in options():
                z=frame[(frame.ModelKey==key)&(frame.candidate_id==c['candidate_id'])]
                mean,sd,tf=float(z.AUC.mean()),float(z.AUC.std(ddof=1)),float(z.FitAUC.mean())
                grid.append({**c,'ModelKey':key,'Phase':ph,'MeanInnerAUC':mean,'InnerSD':sd,'MeanInnerFitAUC':tf,
                     'AvailableAllFolds':bool(z.Available.all()),'ActualRootFitAUC':roc_auc_score(y[a],fitted[key][c['candidate_id']]['train_probability']),
                     'Objective':mean-.1*sd-.3*max(tf-mean-.1,0)})
            selected,guard=choose(grid,fitted[key])
            states[key]={'models':fitted[key],'grid':grid,'selected':selected,'guard':guard,'entries':frame[frame.ModelKey==key].to_dict('records')}
            print(ph+' '+key+' selected%d'%selected['candidate_id'],flush=True)
        joblib.dump(states,dest,compress=3)
        source.PREPARED.clear()
    for ph in PHASES:
        states=joblib.load(OUT/'phases'/(ph+'.joblib'))
        for key,t,joint in STREAMS:
            state=states[key];entries.extend(state['entries']);grids.extend(state['grid']);choices.append({**state['selected'],**state['guard']})
            f=state['models'][state['selected']['candidate_id']]
            a,b=f['train'],f['validation']
            if ph!='final':
                outer.append({'ModelKey':key,'Fold':int(ph[-1]),'FitAUC':roc_auc_score(y[a],f['train_probability']),
                              'AUC':roc_auc_score(y[b],f['evaluation_probability'])})
                predictions.extend({'ModelKey':key,'Fold':int(ph[-1]),'Index':int(i),'AnalysisID':cohort.AnalysisID.iloc[i],
                     'Outcome':int(y[i]),'Probability':float(v)} for i,v in zip(b,f['evaluation_probability']))
            else:
                z=pd.DataFrame(predictions);z=z[z.ModelKey==key].set_index('Index').loc[tr]
                f['threshold']=five.threshold(y[tr],z.Probability.to_numpy())
                f['threshold_source']='OwnselecteddirectPLS_all84outerOOF_Youden'
                joblib.dump(f,OUT/'models'/(key+'.joblib'),compress=3)
                mm=pd.DataFrame(outer);mm=mm[mm.ModelKey==key]
                summary.append({'ModelKey':key,**state['selected'],'TrainFitAUC':roc_auc_score(y[tr],f['train_probability']),
                     'OuterMeanAUC':mm.AUC.mean(),'OuterSD':mm.AUC.std(ddof=1),'PooledOOFAUC':roc_auc_score(y[tr],z.Probability),
                     'Threshold':f['threshold']})
    for name,rows in [('training_summary',summary),('inner_scores',entries),('inner_candidate_summary',grids),
                      ('selected_configurations',choices),('outer_metrics',outer),('nested_predictions',predictions)]:
        csv(name,rows)
    files={str(p.relative_to(OUT)):digest(p) for p in OUT.rglob('*') if p.is_file()}
    save(OUT/'training_frozen.json',{'passed':True,'new_test_calls':0,'files':files})
    print(pd.DataFrame(summary).to_string(index=False),flush=True)


def verify():
    frozen()
    assert not (OUT/'test_started.json').exists()
    cohort,y,tr,te=base.old.inputs()
    original=joblib.load(base.OUT/'training_features.joblib')[1.]
    d=base.route_tables(original,'ContourMedian')
    clinical=joblib.load(parent.OUT/'training_clinical.joblib');alltables={**d,**clinical}
    counts,maxerr,refits,mutations=0,0.,0,0
    # Refit every selected candidate insideall15inner parts, everyrootselection,
    # and check allstoredcandidate grids against their heldoutrecords.
    for ph in PHASES:
        states=joblib.load(OUT/'phases'/(ph+'.joblib'))
        a,b=(tr,np.empty(0,int)) if ph=='final' else base.old.previous.parent.folds('Random120')[int(ph[-1])-1]
        pp=prepared(original,d,a,b,y)
        for key,t,joint in STREAMS:
            st=states[key];frame=pd.DataFrame(st['entries'])
            for row in st['grid']:
                z=frame[frame.candidate_id==row['candidate_id']]
                np.testing.assert_allclose([z.AUC.mean(),z.AUC.std(ddof=1),z.FitAUC.mean()],
                     [row['MeanInnerAUC'],row['InnerSD'],row['MeanInnerFitAUC']],atol=1e-12)
                np.testing.assert_allclose(row['Objective'],row['MeanInnerAUC']-.1*row['InnerSD']-.3*max(row['MeanInnerFitAUC']-row['MeanInnerAUC']-.1,0),atol=1e-12)
            selected,guard=choose(st['grid'],st['models'])
            assert selected==st['selected'] and guard==st['guard']
            for cid,f in st['models'].items():
                np.testing.assert_array_equal(f['train'],a);np.testing.assert_array_equal(f['validation'],b)
                prep=pp[(t,f['config']['image_key'])]
                if prep['indices']:
                    assert f['image']['indices']==prep['indices']
                else:
                    assert f['image'] is None and not f['available'] and f['zero_feature_fallback']
                if t:
                    assert (f['direction_agreement']>=.85).all()
                for ix,expected in [(a,f['train_probability']),(b,f['evaluation_probability'])]:
                    if len(ix):
                        p=five.predict(f,alltables,ix);m=manual(f,alltables,ix) if f['classifier'] is not None else np.full(len(ix),f['prevalence'])
                        np.testing.assert_allclose(p,expected,atol=1e-9,rtol=0);np.testing.assert_allclose(p,m,atol=1e-9,rtol=0)
                        maxerr=max(maxerr,float(np.max(np.abs(p-m))))
                np.testing.assert_allclose(roc_auc_score(y[a],f['train_probability']),st['grid'][cid]['ActualRootFitAUC'],atol=1e-12)
                counts+=1
            c={k:selected[k] for k in options()[0]}
            for rep,k,aa,bb in base.old.previous.parent.folds('Random120',ph):
                assert set(aa).isdisjoint(bb) and not (set(aa)|set(bb))&(set(b)|set(te))
                keyimage=c['image_key'];p0=source.prepared(original,d,aa,keyimage,y);p=source.filtered(p0,d[keyimage],aa,bb,t)
                cp=five.clinical_prepare(clinical,aa,bb,'Extended11' if joint else 'None')
                reference=clinical_reference(clinical['Clinical'],aa,bb)
                if joint:
                    for name in ('median','low','high','center','scale'):
                        np.testing.assert_allclose(cp[name],reference[name],atol=1e-12)
                f=five.fit_prepared(p,cp,y[aa],{**c,'clinical_scope':'Extended11' if joint else 'None'},aa,bb)
                row=frame[(frame.Repeat==rep)&(frame.Fold==k)&(frame.candidate_id==c['candidate_id'])].iloc[0]
                np.testing.assert_allclose([roc_auc_score(y[aa],f['train_probability']),roc_auc_score(y[bb],f['evaluation_probability'])],[row.FitAUC,row.AUC],atol=1e-12)
                refits+=1
            # A newtraining-onlyselectedrootfit must ignoreallheldoutlabels.
            yy=y.copy();yy[np.r_[b,te]]=np.arange(len(b)+len(te))%2
            f=st['models'][selected['candidate_id']]
            p=pp[(t,c['image_key'])];cp=five.clinical_prepare(clinical,a,b,'Extended11' if joint else 'None')
            replay=five.fit_prepared(p,cp,yy[a],f['config'],a,b)
            np.testing.assert_array_equal(f['image_coef_gaussian'],replay['image_coef_gaussian']);mutations+=1
        source.PREPARED.clear()
        print('VerifieddirectPLS '+ph+':all144rootmodels,60innerselectedrefits;notest',flush=True)
    checked=0
    for key in d:
        p=source.prepared(original,d,tr,key,y);expected=independent_direction(p['xa'],y[tr],tr)
        np.testing.assert_allclose(expected,p['agreement'][:len(expected)],atol=1e-12);checked+=len(expected)
    summary=read('training_summary').set_index('ModelKey');pred=read('nested_predictions');mm=read('outer_metrics')
    for key,t,joint in STREAMS:
        f=joblib.load(OUT/'models'/(key+'.joblib'));z=pred[pred.ModelKey==key].set_index('Index').loc[tr];m=mm[mm.ModelKey==key]
        assert len(m)==5 and z.index.is_unique and len(z)==84
        assert f['threshold']==five.threshold(y[tr],z.Probability.to_numpy())
        np.testing.assert_allclose([roc_auc_score(y[tr],f['train_probability']),m.AUC.mean(),m.AUC.std(ddof=1)],summary.loc[key,['TrainFitAUC','OuterMeanAUC','OuterSD']].to_numpy(float),atol=1e-12)
    frozen()
    save(OUT/'training_verification.json',{'passed':True,'all_root_models_checked':counts,'selected_inner_refits':refits,
         'independent_integer_bootstrap_features':checked,'heldout_label_mutations':mutations,'max_manual_error':maxerr,
         'all24selections_verified':True,'all5folds_kept':True,'new_test_calls':0})
    print('FourdirectPLSpipelines verifiedbeforetest',flush=True)


def test():
    frozen();assert not (OUT/'test_started.json').exists()
    assert json.loads((OUT/'training_verification.json').read_text(encoding='utf-8'))['passed']
    cohort,_,tr,te=base.old.inputs()
    save(OUT/'test_started.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'new_models':4,'joint_stage':1,'thresholds_frozen':True})
    d=base.route_tables(joblib.load(base.OUT/'test_features.joblib')[1.],'ContourMedian')
    tables={**d,**joblib.load(parent.OUT/'test_clinical.joblib')}
    truth=cohort.Outcome.to_numpy(int)[te];scores={};metrics=[];predictions=[];paired=[]
    old=parent.read('test_predictions')
    from verify_ivim_df63_20261020 import draws
    for key,t,joint in STREAMS:
        f=joblib.load(OUT/'models'/(key+'.joblib'))
        scores[key]=five.predict(f,tables,te)
        np.testing.assert_allclose(scores[key],manual(f,tables,te),atol=1e-9,rtol=0)
        prev=old[old.ModelKey==key].set_index('Index').loc[te]
        np.testing.assert_array_equal(prev.AnalysisID,cohort.AnalysisID.iloc[te])
        scores['Prior_'+key]=prev.Probability.to_numpy()
    samples=draws(truth,scores,SEED)
    for key,t,joint in STREAMS:
        f=joblib.load(OUT/'models'/(key+'.joblib'));p=scores[key]
        lo,hi=np.quantile(samples[key],[.025,.975]);tn,fp,fn,tp=confusion_matrix(truth,p>=f['threshold'],labels=[0,1]).ravel()
        metrics.append({'ModelKey':key,'TestAUC':roc_auc_score(truth,p),'AUC_CI95Low':float(lo),'AUC_CI95High':float(hi),
            'Threshold':f['threshold'],'Sensitivity':tp/(tp+fn),'Specificity':tn/(tn+fp),'Accuracy':(tp+tn)/36.,
            'TP':int(tp),'FN':int(fn),'TN':int(tn),'FP':int(fp)})
        predictions.extend({'ModelKey':key,'Index':int(i),'AnalysisID':cohort.AnalysisID.iloc[i],'Outcome':int(y),
             'Probability':float(v),'Prediction':int(v>=f['threshold'])} for i,y,v in zip(te,truth,p))
        lo,hi=np.quantile(samples[key]-samples['Prior_'+key],[.025,.975])
        paired.append({'Comparison':key+'_minus_Prior','AUC_Difference':roc_auc_score(truth,p)-roc_auc_score(truth,scores['Prior_'+key]),
                       'CI95Low':float(lo),'CI95High':float(hi)})
    csv('test_metrics',metrics);csv('test_predictions',predictions);csv('paired_test_differences',paired)
    frozen();save(OUT/'test_complete.json',{'passed':True,'new_models':4,'joint_test_stages':1,'fixed_candidate_tests':0,
         'post_test_fits':0,'post_test_threshold_changes':0,'formal_adoption_changed':False})
    print(pd.DataFrame(metrics).to_string(index=False),flush=True)


def report():
    frozen();assert json.loads((OUT/'test_complete.json').read_text(encoding='utf-8'))['passed']
    from report_dualroute_studies_20261025 import report as make_report
    make_report()


def seal():
    frozen()
    for n in ('training_verification','test_complete','report_verification','visual_verification'):
        assert json.loads((OUT/(n+'.json')).read_text(encoding='utf-8'))['passed']
    paths={str(p.relative_to(OUT)):digest(p) for p in OUT.rglob('*') if p.is_file() and not p.name.startswith('delivery_')}
    save(OUT/'delivery_manifest.json',{'outputs':paths})
    for n,sha in paths.items():
        assert digest(OUT/n)==sha,n
    save(OUT/'delivery_final_verification.json',{'passed':True,'outputs_checked':len(paths),'four_required_streams':True,'all5folds_kept':True,'formal_adoption_changed':False})
    print('DirectPLSfour-streamstudy delivered',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('define','train','verify','test','report','seal'))
    with threadpool_limits(1):
        globals()[parser.parse_args().action]()
