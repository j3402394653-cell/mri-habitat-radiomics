# -*- coding: utf-8 -*-
"""Nested refinement of fixed original/85% habitat and habitat-clinical routes."""
import os
os.environ.update({'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
import argparse
from datetime import datetime, timezone
import json
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,confusion_matrix
from threadpoolctl import threadpool_limits
import run_original85_five_models_20261025 as parent

ROOT,save,digest = parent.ROOT,parent.save,parent.digest
source,base,five = parent.source,parent.base,parent.five
OUT = ROOT/'lvi_prediction/results/lvi_dualroute_refinement_20261025'
PHASES = source.PHASES
STREAMS = (('HabitatOriginal',0.,False),('HabitatClinicalOriginal',0.,True),
           ('Habitat85',.85,False),('HabitatClinical85',.85,True))
SEED = 20261025267


def csv(name,rows):
    pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')


def read(name):
    return pd.read_csv(OUT/(name+'.csv'),float_precision='round_trip')


def options():
    pairs = [('Both',.03,.25)]+[(mode,c,a) for mode in ('Both','PLS1','PLS2')
              for c in (.01,.03,.1) for a in (.25,.5,1.,2.) if (mode,c,a) != ('Both',.03,.25)]
    result = [{'candidate_id':i,'channels':mode,'C':c,'anchor':a} for i,(mode,c,a) in enumerate(pairs)]
    assert len(result) == 36
    return result


def columns(mode):
    return np.asarray([j for j,c in enumerate(base.old.previous.members())
                       if mode == 'Both' or c['components'] == int(mode[-1])],int)


def protect():
    p = json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    for field in ('protected_hashes','source_hashes'):
        for name,sha in p[field].items():
            assert digest(ROOT/name) == sha,name
    assert digest(OUT/'protocol.json') == json.loads((OUT/'definition_frozen.json').read_text(encoding='utf-8'))['protocol_sha256']
    return len(p['protected_hashes'])


def frozen():
    protect()
    p = json.loads((OUT/'training_frozen.json').read_text(encoding='utf-8'))
    for name,sha in p['files'].items():
        assert digest(OUT/name) == sha,name
    return p


def define():
    assert not OUT.exists()
    parent.frozen()
    assert json.loads((parent.OUT/'delivery_final_verification.json').read_text(encoding='utf-8'))['passed']
    original = json.loads((parent.OUT/'protocol.json').read_text(encoding='utf-8'))
    protected = {**original['protected_hashes'],**original['source_hashes']}
    protected.update({str(p.relative_to(ROOT)):digest(p) for p in parent.OUT.rglob('*') if p.is_file()})
    decision = ROOT/'lvi_prediction/RESEARCH_DIRECTIONS_ORIGINAL_FIXED85_20261025.json'
    protected[str(decision.relative_to(ROOT))] = digest(decision)
    OUT.mkdir(parents=True)
    for folder in ('phases','models','figures'):
        (OUT/folder).mkdir()
    scripts = [ROOT/'lvi_prediction'/name for name in ('run_dualroute_refinement_20261025.py','evaluate_dualroute_refinement_20261025.py')]
    save(OUT/'protocol.json',{
        'date':'2026-10-25','created_utc':datetime.now(timezone.utc).isoformat(),
        'authorization':'Keep original andfixed85 asfuture researchroutes;everyexploration evaluatespureHabitat andHabitatClinical;improveall5outermean toward.75 while actualtrainfitnottoolow.',
        'required_streams':STREAMS,'direction_choices_fixed':[0.,.85],
        'cohort':'Exact sameA120,84train57+/27-,36test25+/11-. All5outerfolds,inner5x3,eachinnertrain ownverifiedprivatebaseOOF unchanged;no caseorfoldselection.',
        'unchanged_images':'Only1mmcontourmedian,k3,T2/ADC/IVIM_D/f,2normalizations x3views,existingPLS1/2baseC.03,fold-onlyQC/transform anddirection40bootstrap. No newimageorbasefit.',
        'new_search':'Within eachstream independently compare36metarecipes:Both12/PLS1six/PLS2six existingbasechannels x C.01/.03/.1 x anchor.25/.5/1/2. Baseline recipe0exact Both,C.03,anchor.25.',
        'PLS_interpretation':'Selectwhichalreadytrained PLS1/2 basechannelsenterfusion;not a caponoriginalrawhabitatfeatures andnot a newPLSfeaturefit.',
        'clinical':'Same11preoperativevariablesforjointmodels only,fold-onlymedianfill,P2.5/P97.5clip,median/IQR scalewithSD/1fallback. No directionfilterappliedtoclinical.',
        'meta':'UnconstrainedsignedbalancedL2liblinearLR,max_iter3000seed64212;OOF-scoretrain-onlyclip/meanSD;anchor duplicatesOOFandactualBag15train inputsweighted1/(1+a),a/(1+a),totalweightN.',
        'selection':'Availableall15innerfolds androotfit. Preferactualcurrent-trainingfit>=.75 ifany,else>=.70ifany,elseallvalidwithfailureflag. Max meanInnerAUC-.10SD-.30max(meanInnerFit-meanInnerAUC-.10,0);tieswithin1e-12fewerhabitatscorechannels,lowerC,loweranchor,ID.',
        'selection_isolation':'Rootfitfloor usescurrenttrainingonly. NeitheroutervalidationAUCnor36testguidesrecipechoice. Fixedrecipeouterdiagnosticssecondaryonly;neverselectprimarybytheiroutermax.',
        'primary':'Fourseparatelynestedselectedpipelines,oneperrequiredstream;finishandverifyallfourbeforeonejointteststage eveniftrainingtargetsnotmet.',
        'threshold':'Eachadaptivepipeline owncomplete84outerOOFYouden,frozenbeforetest. Finalbaseline-identicalprobabilityfunctioncanreusepinnedoldscoresbutownadaptivethresholdstillreported.',
        'test':'Atmostfournewcomplete-modelpredictions;exactbaselinefunctionusespinnedoldscores. No testall36recipes,no posttestfit,rethreshold,scoreflip oradoptionchange.',
        'uncertainty':{'bootstrap':5000,'seed':SEED,'method':'Pairedclass-stratifiedpatientbootstrap,fixedscores,same36,notretraining/selectionuncertainty'},
        'interpretation':'Exploratory aftermanysourceouter/testuses;not external/untouchedvalidation. Desired.75notguaranteed. Clinicaljointresultsnotpurehabitatresults.',
        'source_hashes':{str(p.relative_to(ROOT)):digest(p) for p in scripts},'protected_hashes':protected})
    (OUT/'split_membership.csv').write_bytes((parent.OUT/'split_membership.csv').read_bytes())
    csv('candidate_definitions',options())
    save(OUT/'definition_frozen.json',{'passed':True,'new_test_calls':0,'protocol_sha256':digest(OUT/'protocol.json')})
    print('Defined4requiredstreams x36recipes;fixed0/.85;no newbase/imagefit',flush=True)


def prepare(ctx,clinical):
    pp = base.old.previous.meta_prepare(ctx['oof'],ctx['bag_validation'])
    return {'ctx':ctx,'score':pp,'actual':(np.clip(ctx['bag_train'],pp['low'],pp['high'])-pp['center'])/pp['scale'],
            'clinical':five.clinical_prepare(clinical,ctx['train'],ctx['validation'],'Extended11')}


def fit(p,y,c,t,joint):
    ctx,pp,cp = p['ctx'],p['score'],p['clinical']
    a,b = ctx['train'],ctx['validation']
    cols = columns(c['channels'])
    co = cp['xa'] if joint else np.empty((len(a),0))
    cb = cp['xb'] if joint else np.empty((len(b),0))
    oof = np.column_stack([pp['xa'][:,cols],co])
    actual = np.column_stack([p['actual'][:,cols],co])
    evaluation = np.column_stack([pp['xb'][:,cols],cb])
    alpha = c['anchor']
    weights = np.r_[np.full(len(a),1/(1+alpha)),np.full(len(a),alpha/(1+alpha))]
    model = LogisticRegression(C=c['C'],penalty='l2',solver='liblinear',class_weight='balanced',max_iter=3000,random_state=64212)
    model.fit(np.row_stack([oof,actual]),np.tile(y[a],2),sample_weight=weights)
    pa = model.predict_proba(actual)[:,1]
    return {'config':c,'direction_threshold':t,'joint':joint,'columns':cols,'train':a,'validation':b,
            'model':model,'coef':model.coef_[0].copy(),'intercept':float(model.intercept_[0]),
            **{name:pp[name][cols] for name in ('low','high','center','scale')},
            'clinical':{name:cp[name] for name in ('indices','names','median','low','high','center','scale')} if joint else None,
            'train_probability':pa,'evaluation_probability':model.predict_proba(evaluation)[:,1] if len(b) else np.empty(0),
            'meta_training_association_probability':model.predict_proba(oof)[:,1],
            'available':bool(np.ptp(pa)>1e-12),'ScoreType':'Refined_Bag15_habitat_probability_uncalibrated'}


def choose(grid,models):
    valid = [r for r in grid if r['AvailableAllFolds'] and models[r['candidate_id']]['available']]
    assert valid
    good = [r for r in valid if r['ActualRootFitAUC'] >= .75-1e-12]
    moderate = [r for r in valid if r['ActualRootFitAUC'] >= .70-1e-12]
    pool = good or moderate or valid
    best = max(r['Objective'] for r in pool)
    ties = [r for r in pool if r['Objective'] >= best-1e-12]
    selected = min(ties,key=lambda r:(len(columns(r['channels'])),r['C'],r['anchor'],r['candidate_id']))
    return selected,{'FitPreference':.75 if good else .70 if moderate else None,
                     'AvailableCandidates':len(valid),'CandidatesAt075':len(good),'EligibleCandidates':len(pool),'TargetFitMet':bool(good)}


def train():
    protect()
    assert not (OUT/'training_frozen.json').exists() and not (OUT/'test_started.json').exists()
    cohort,y,tr,te = base.old.inputs()
    assert (y[te] == -1).all()
    clinical = joblib.load(parent.OUT/'training_clinical.joblib')
    entries,grids,choices,outer,predictions,fixed = [],[],[],[],[],[]
    for ph in PHASES:
        dest = OUT/'phases'/(ph+'.joblib')
        if dest.exists():
            continue
        states = {}
        for _,t in parent.SCHEMES:
            inners = {False:[],True:[]}
            for rep,k,a,b in base.old.previous.parent.folds('Random120',ph):
                p = prepare(source.context(ph,t,rep,k),clinical)
                for joint in (False,True):
                    for c in options():
                        f = fit(p,y,c,t,joint)
                        inners[joint].append({'Phase':ph,'ModelKey':parent.hkey(t,joint),'Repeat':rep,'Fold':k,
                              'candidate_id':c['candidate_id'],'FitAUC':roc_auc_score(y[a],f['train_probability']),
                              'AUC':roc_auc_score(y[b],f['evaluation_probability']),'Available':f['available']})
            p = prepare(source.context(ph,t),clinical)
            for joint in (False,True):
                key = parent.hkey(t,joint)
                frame = pd.DataFrame(inners[joint])
                models = {c['candidate_id']:fit(p,y,c,t,joint) for c in options()}
                grid = []
                for c in options():
                    z = frame[frame.candidate_id == c['candidate_id']]
                    mean,sd,tf = float(z.AUC.mean()),float(z.AUC.std(ddof=1)),float(z.FitAUC.mean())
                    grid.append({**c,'Phase':ph,'ModelKey':key,'MeanInnerAUC':mean,'InnerSD':sd,'MeanInnerFitAUC':tf,
                        'AvailableAllFolds':bool(z.Available.all()),'ActualRootFitAUC':roc_auc_score(y[p['ctx']['train']],models[c['candidate_id']]['train_probability']),
                        'Objective':mean-.1*sd-.3*max(tf-mean-.1,0)})
                selected,guard = choose(grid,models)
                states[key] = {'models':models,'grid':grid,'entries':inners[joint],'selected':selected,'guard':guard}
                print('%s %s chose%d %s C%g anchor%g;notest'%(ph,key,selected['candidate_id'],selected['channels'],selected['C'],selected['anchor']),flush=True)
        joblib.dump(states,dest,compress=3)
    for ph in PHASES:
        states = joblib.load(OUT/'phases'/(ph+'.joblib'))
        for key,t,joint in STREAMS:
            cp = states[key]
            entries.extend(cp['entries']);grids.extend(cp['grid'])
            choices.append({**cp['selected'],**cp['guard']})
            if ph == 'final':
                continue
            k = int(ph[-1])
            f = cp['models'][cp['selected']['candidate_id']]
            a,b = f['train'],f['validation']
            outer.append({'ModelKey':key,'Fold':k,'candidate_id':f['config']['candidate_id'],
                          'FitAUC':roc_auc_score(y[a],f['train_probability']),'AUC':roc_auc_score(y[b],f['evaluation_probability'])})
            predictions.extend({'ModelKey':key,'Fold':k,'Index':int(i),'AnalysisID':cohort.AnalysisID.iloc[i],
                   'Outcome':int(y[i]),'Probability':float(v)} for i,v in zip(b,f['evaluation_probability']))
            for cid,ff in cp['models'].items():
                fixed.append({'ModelKey':key,'candidate_id':cid,'Fold':k,'FitAUC':roc_auc_score(y[a],ff['train_probability']),
                              'AUC':roc_auc_score(y[b],ff['evaluation_probability'])})
    finals = joblib.load(OUT/'phases/final.joblib')
    pp,mm,fx = pd.DataFrame(predictions),pd.DataFrame(outer),pd.DataFrame(fixed)
    summary,fixedsummary = [],[]
    for key,t,joint in STREAMS:
        state = finals[key]
        f = state['models'][state['selected']['candidate_id']].copy()
        z = pp[pp.ModelKey == key].set_index('Index').loc[tr]
        f['threshold'] = five.threshold(y[tr],z.Probability.to_numpy())
        f['threshold_source'] = 'Own_adaptive36_complete84_outerOOF_Youden'
        f['baseline_probability_identical'] = f['config']['candidate_id'] == 0
        joblib.dump(f,OUT/'models'/(key+'.joblib'),compress=3)
        m = mm[mm.ModelKey == key]
        summary.append({'ModelKey':key,**f['config'],'DirectionThreshold':t,'Clinical':joint,
             'TrainFitAUC':roc_auc_score(y[tr],f['train_probability']),'OuterMeanAUC':m.AUC.mean(),'OuterSD':m.AUC.std(ddof=1),
             'PooledOOFAUC':roc_auc_score(y[tr],z.Probability),'Threshold':f['threshold'],
             'BaselineProbabilityIdentical':f['baseline_probability_identical'],
             'MetaTrainingAssociationAUC':roc_auc_score(y[tr],f['meta_training_association_probability'])})
        for cid,ff in state['models'].items():
            m = fx[(fx.ModelKey == key)&(fx.candidate_id == cid)]
            fixedsummary.append({'ModelKey':key,**ff['config'],'TrainFitAUC':roc_auc_score(y[tr],ff['train_probability']),
                         'OuterMeanAUC':m.AUC.mean(),'OuterSD':m.AUC.std(ddof=1)})
    for name,rows in [('training_summary',summary),('selected_configurations',choices),('inner_scores',entries),
                      ('inner_candidate_summary',grids),('outer_metrics',outer),('nested_predictions',predictions),
                      ('fixed_recipe_outer_diagnostics',fixed),('fixed_recipe_summary',fixedsummary)]:
        csv(name,rows)
    files = {str(p.relative_to(OUT)):digest(p) for p in OUT.rglob('*') if p.is_file()}
    save(OUT/'training_frozen.json',{'passed':True,'new_test_calls':0,'four_primary_pipelines':True,'files':files})
    print(pd.DataFrame(summary).to_string(index=False),flush=True)


def design(f,x,clinical,indices):
    z = (np.clip(x[:,f['columns']],f['low'],f['high'])-f['center'])/f['scale']
    if f['joint']:
        cp = f['clinical'];d = clinical['Clinical']
        raw = d['base'][five.latent.rows(d,indices)][:,cp['indices']]
        zc = (np.clip(np.where(np.isfinite(raw),raw,cp['median']),cp['low'],cp['high'])-cp['center'])/cp['scale']
        z = np.column_stack([z,zc])
    return z


def predict(f,x,clinical,indices):
    z = design(f,x,clinical,indices)
    p = f['model'].predict_proba(z)[:,1]
    np.testing.assert_allclose(p,expit(z@f['coef']+f['intercept']),atol=1e-10,rtol=0)
    return p


def test():
    frozen()
    assert not (OUT/'test_started.json').exists()
    assert json.loads((OUT/'training_verification.json').read_text(encoding='utf-8'))['passed']
    cohort,_,tr,te = base.old.inputs()
    final = {key:joblib.load(OUT/'models'/(key+'.joblib')) for key,t,joint in STREAMS}
    calls = sum(not f['baseline_probability_identical'] for f in final.values())
    save(OUT/'test_started.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'selected_primary_pipelines':4,
         'new_probability_functions':calls,'baseline_functions_reused':4-calls,'joint_test_stages':1,'thresholds_frozen':True})
    tables = joblib.load(base.OUT/'test_features.joblib')
    clinical = joblib.load(parent.OUT/'test_clinical.joblib')
    old = parent.read('test_predictions')
    scores,previous = {},{}
    matrices = {}
    for key,t,joint in STREAMS:
        f = final[key]
        prior = old[old.ModelKey == key].set_index('Index').loc[te]
        np.testing.assert_array_equal(prior.AnalysisID,cohort.AnalysisID.iloc[te])
        previous[key] = prior.Probability.to_numpy()
        if f['baseline_probability_identical']:
            scores[key] = previous[key].copy()
        else:
            if t not in matrices:
                matrices[t] = parent.base_matrix(source.basis('final',t),tables,te)
            scores[key] = predict(f,matrices[t],clinical,te)
    truth = cohort.Outcome.to_numpy(int)[te]
    from verify_ivim_df63_20261020 import draws
    allscores = {**scores,**{'Prior_'+k:v for k,v in previous.items()}}
    samples = draws(truth,allscores,SEED)
    metrics,predictions,paired = [],[],[]
    for key,t,joint in STREAMS:
        f,p = final[key],scores[key]
        lo,hi = np.quantile(samples[key],[.025,.975])
        tn,fp,fn,tp = confusion_matrix(truth,p >= f['threshold'],labels=[0,1]).ravel()
        metrics.append({'ModelKey':key,'N':36,'TestAUC':roc_auc_score(truth,p),'AUC_CI95Low':float(lo),'AUC_CI95High':float(hi),
             'Threshold':f['threshold'],'Sensitivity':tp/(tp+fn),'Specificity':tn/(tn+fp),'Accuracy':(tp+tn)/36.,
             'TP':int(tp),'FN':int(fn),'TN':int(tn),'FP':int(fp),'BaselineProbabilityIdentical':f['baseline_probability_identical']})
        predictions.extend({'ModelKey':key,'Index':int(i),'AnalysisID':cohort.AnalysisID.iloc[i],'Outcome':int(y),
             'Probability':float(v),'Prediction':int(v >= f['threshold']),'PriorProbability':float(w)} for i,y,v,w in zip(te,truth,p,previous[key]))
        dl,dh = np.quantile(samples[key]-samples['Prior_'+key],[.025,.975])
        paired.append({'Comparison':key+'_minus_Prior','AUC_Difference':roc_auc_score(truth,p)-roc_auc_score(truth,previous[key]),
                       'CI95Low':float(dl),'CI95High':float(dh)})
    for a,b in [('Habitat85','HabitatOriginal'),('HabitatClinical85','HabitatClinicalOriginal'),
                ('HabitatClinicalOriginal','HabitatOriginal'),('HabitatClinical85','Habitat85')]:
        lo,hi = np.quantile(samples[a]-samples[b],[.025,.975])
        paired.append({'Comparison':a+'_minus_'+b,'AUC_Difference':roc_auc_score(truth,scores[a])-roc_auc_score(truth,scores[b]),
                       'CI95Low':float(lo),'CI95High':float(hi)})
    csv('test_metrics',metrics);csv('test_predictions',predictions);csv('paired_test_differences',paired)
    frozen()
    save(OUT/'test_complete.json',{'passed':True,'primary_pipelines_evaluated':4,'new_probability_functions':calls,
         'reused_baseline_probability_functions':4-calls,'joint_test_stages':1,'fixed_candidate_tests':0,
         'post_test_fits':0,'post_test_threshold_changes':0,'formal_adoption_changed':False})
    print(pd.DataFrame(metrics).to_string(index=False),flush=True)


def seal():
    frozen()
    for name in ('training_verification','test_complete','report_verification','visual_verification'):
        assert json.loads((OUT/(name+'.json')).read_text(encoding='utf-8'))['passed']
    files = {str(p.relative_to(OUT)):digest(p) for p in OUT.rglob('*') if p.is_file()
              and p.name not in ('delivery_manifest.json','delivery_final_verification.json')}
    save(OUT/'delivery_manifest.json',{'outputs':files})
    for name,sha in files.items():
        assert digest(OUT/name) == sha,name
    save(OUT/'delivery_final_verification.json',{'passed':True,'outputs_checked':len(files),'four_required_streams':True,
         'all5folds_kept':True,'only1mm':True,'no_posttest_fit':True,'formal_adoption_changed':False})
    print('Dual-route four-pipeline refinement delivered;allpriorrecords preserved',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=('define','train','test','seal'))
    with threadpool_limits(1):
        globals()[parser.parse_args().action]()
