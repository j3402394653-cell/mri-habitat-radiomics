# -*- coding: utf-8 -*-
"""Two habitat recipes, five families each; three shared immutable baselines."""
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
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from threadpoolctl import threadpool_limits
import test_fixed85_direction_20261025 as retained
import run_random120_five_models_20261023 as five

ROOT, save, digest = retained.ROOT, retained.save, retained.digest
source, base = retained.source, retained.base
OUT = ROOT/'lvi_prediction/results/lvi_original85_five_models_20261025'
SCHEMES = (('Original',0.),('Fixed85',.85))
SHARED = ('Whole','Clinical','WholeClinical')
UNIQUE = ('HabitatOriginal','Whole','Clinical','HabitatClinicalOriginal','WholeClinical','Habitat85','HabitatClinical85')
PHASES = source.PHASES
PAIRS = [('Habitat85','HabitatOriginal'),('HabitatClinical85','HabitatClinicalOriginal'),
         ('HabitatClinicalOriginal','HabitatOriginal'),('HabitatClinical85','Habitat85'),
         ('HabitatClinicalOriginal','Clinical'),('HabitatClinical85','Clinical'),
         ('WholeClinical','Whole'),('HabitatClinicalOriginal','WholeClinical'),('HabitatClinical85','WholeClinical')]
SEED = 20261025211


def csv(name, rows):
    pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')


def read(name):
    return pd.read_csv(OUT/(name+'.csv'),float_precision='round_trip')


def hkey(t, joint=False):
    return ('HabitatClinical' if joint else 'Habitat')+('85' if t else 'Original')


def protect():
    protocol = json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    for field in ('protected_hashes','source_hashes'):
        for name, sha in protocol[field].items():
            assert digest(ROOT/name) == sha,name
    assert digest(OUT/'protocol.json') == json.loads((OUT/'definition_frozen.json').read_text(encoding='utf-8'))['protocol_sha256']
    return len(protocol['protected_hashes'])


def frozen():
    protect()
    record = json.loads((OUT/'training_frozen.json').read_text(encoding='utf-8'))
    for name, sha in record['files'].items():
        assert digest(OUT/name) == sha,name
    return record


def define():
    assert not OUT.exists()
    retained.frozen()
    five.frozen()
    for directory in (retained.OUT,five.OUT):
        assert json.loads((directory/'delivery_final_verification.json').read_text(encoding='utf-8'))['passed']
    cohort,y,tr,te = base.old.inputs()
    _,y2,tr2,te2 = five.inputs()
    np.testing.assert_array_equal(tr,tr2)
    np.testing.assert_array_equal(te,te2)
    np.testing.assert_array_equal(y[tr],y2[tr])
    for (a,b),(c,d) in zip(base.old.previous.parent.folds('Random120'),five.habitat.parent.folds('Random120')):
        np.testing.assert_array_equal(a,c)
        np.testing.assert_array_equal(b,d)
    old = json.loads((retained.OUT/'protocol.json').read_text(encoding='utf-8'))
    protected = {**old['protected_hashes'],**old['source_hashes']}
    for directory in (retained.OUT,five.OUT):
        protected.update({str(p.relative_to(ROOT)):digest(p) for p in directory.rglob('*') if p.is_file()})
    for p in (ROOT/'AGENTS.md',ROOT/'lvi_prediction/COHORT_CONFIRMATION_20261012.json',
              ROOT/'lvi_prediction/LVI_RESULT_SELECTION.json',ROOT/'lvi_prediction/MODEL_DECISION.md'):
        protected[str(p.relative_to(ROOT))] = digest(p)
    OUT.mkdir(parents=True)
    for name in ('models','figures'):
        (OUT/name).mkdir()
    scripts = [ROOT/'lvi_prediction'/n for n in ('run_original85_five_models_20261025.py','evaluate_original85_five_models_20261025.py')]
    save(OUT/'protocol.json',{
        'date':'2026-10-25','created_utc':datetime.now(timezone.utc).isoformat(),
        'authorization':'User requests both original and fixed85 corresponding five families; explicitly confirms direction filter only affects habitat and habitat+clinical, three other baselines shared.',
        'cohort':'Exact same A120,84training57+/27-,36test25+/11-. Same5outer and15inner/privatefolds,allfolds/casesretained.',
        'five_families':['Habitat','Whole','Clinical','HabitatClinical','WholeClinical'],
        'groups':SCHEMES,'unique_model_count':7,'display_rows':10,
        'Habitat':'Reuse exact original 1mm median Bag15 .768031/.621818/.821818 and fixed85 .653671/.709394/.789091; no new pure-H test call or re-fit.',
        'shared_baselines':'Reuse pinned previous Whole,Clinical,WholeClinical models,training/outer/test predictions after exact case/fold check; identical repeated rows in both groups,not new independent results.',
        'shared_Whole':'372original nonshape radiomics,T2/ADC93each,IVIM_D/f93each,two original normalizations;train-only QC/transform and22candidate inner selection preserved.',
        'shared_Clinical':'Previous6candidate Basic5/Extended11,C.003/.03/.3 inner search preserved;finalExtended11,C.3.',
        'shared_WholeClinical':'Previous44candidate inner selection preserved;finalPercentileZ_PCA5+C.03+Extended11.',
        'new_HabitatClinical':'Only2new pipelines;use exact same current12channelBag15 basesandstrictprivateOOFcontexts for eachdirection choice;append11preoperativeclinicalcovariates tooriginal12OOFscorefeatures.',
        'new_HC_fit':'FixedbalancedL2liblinearLR C.03,anchor.25,seed64212,max_iter3000;same metaOOF clipping/meanSD,score signs unconstrained. No new hyperparameter search.',
        'new_HC_anchor':'OOF+actualBag15training inputs weighted1/(1+.25) and.25/(1+.25),totalweightN. Clinicaltraincolumns repeated identically in both blocks;not validation.',
        'clinical_preprocess':'Current-training-onlymedianfill,P2.5/P97.5clip,median/IQR-normal-equivalent scaling,SDthen1fallback. All11clinicalvariablesretained,not85%filtered.',
        'clinical_columns':[m['name'] for m in five.clinical_source.clinical(tr)['metadata']],
        'new_HC_inner':'Report fixedrecipe inner5x3 diagnostics for every outertrain/final. Eachinnertrain usesitsown verifiedprivatebaseOOF;not usedasadditionalcandidate search.',
        'new_HC_outer':'Fitseparatelyoneachof5outertrainingparts; actual full84 fit reported separately; not the metaOOF association AUC.',
        'threshold':'EachnewHC owncomplete84outerOOFYouden,beforetest;five reused pipelinesretain oldthresholds.',
        'test':'FinishandverifybothnewHC,beforeonejointteststage:72new completepipeline predictions;180scoresfrom5retainedunique models reused. 10displayrowsare7unique models.',
        'uncertainty':{'method':'5000pairedclass-stratifiedbootstrap of fixed36patient scores,commonseedall7','seed':SEED,'pairs':PAIRS},
        'interpretation':'Exploratory extension after source outer/test inspections;not newindependent/externalvalidation. No test-driven selection,scoreflip,thresholdchange,posttestfit orformaladoption.',
        'scope':'Only1mm,T2/ADC/IVIM_D/f,k3,noDstar. PureH reuse doesnot adopt excluded20261008migrationor20261010supervised experiments.',
        'source_hashes':{str(p.relative_to(ROOT)):digest(p) for p in scripts},'protected_hashes':protected})
    csv('split_membership',[{'Index':int(i),'AnalysisID':cohort.AnalysisID.iloc[i],'Set':label}
                           for label,indices in [('Training',tr),('Test',te)] for i in indices])
    save(OUT/'definition_frozen.json',{'passed':True,'new_test_calls':0,'protocol_sha256':digest(OUT/'protocol.json')})
    print('Defined2newHC+5retained unique models;10rows,exact84/36;directiononlyinhabitat',flush=True)


def fit_hc(tables,ctx,y,t):
    a,b = ctx['train'],ctx['validation']
    cp = five.clinical_prepare(tables,a,b,'Extended11')
    pp = base.old.previous.meta_prepare(ctx['oof'],ctx['bag_validation'])
    actual = (np.clip(ctx['bag_train'],pp['low'],pp['high'])-pp['center'])/pp['scale']
    oof = np.column_stack([pp['xa'],cp['xa']])
    za = np.column_stack([actual,cp['xa']])
    zb = np.column_stack([pp['xb'],cp['xb']])
    x = np.row_stack([oof,za])
    alpha = .25
    weights = np.r_[np.ones(len(a))/(1+alpha),np.full(len(a),alpha/(1+alpha))]
    model = LogisticRegression(C=.03,penalty='l2',solver='liblinear',class_weight='balanced',max_iter=3000,random_state=64212)
    model.fit(x,np.r_[y[a],y[a]],sample_weight=weights)
    return {'direction_threshold':t,'train':a,'validation':b,'meta':model,'coef':model.coef_[0].copy(),
            'intercept':float(model.intercept_[0]),**{k:pp[k] for k in ('low','high','center','scale')},
            'clinical':{k:cp[k] for k in ('indices','names','median','low','high','center','scale')},
            'train_probability':model.predict_proba(za)[:,1],
            'evaluation_probability':model.predict_proba(zb)[:,1] if len(b) else np.empty(0),
            'meta_training_association_probability':model.predict_proba(oof)[:,1],
            'config':{'C':.03,'anchor':.25,'clinical_scope':'Extended11','base_mode':'Bag15'},
            'ScoreType':'Anchored_Bag15_habitat_plus_11clinical_probability_uncalibrated'}


def base_matrix(basis,tables,indices):
    d = base.route_tables(tables[1.],'ContourMedian')
    return np.mean([np.column_stack([base.old.latent.predict(m,d,indices) for m in row['models']])
                    for row in basis['bag']],axis=0)


def hc_design(f,x,tables,indices):
    cp = f['clinical']
    d = tables['Clinical']
    raw = d['base'][five.latent.rows(d,indices)][:,cp['indices']]
    zc = (np.clip(np.where(np.isfinite(raw),raw,cp['median']),cp['low'],cp['high'])-cp['center'])/cp['scale']
    zh = (np.clip(x,f['low'],f['high'])-f['center'])/f['scale']
    return np.column_stack([zh,zc])


def predict_hc(f,basis,tables,clinical,indices):
    z = hc_design(f,base_matrix(basis,tables,indices),clinical,indices)
    p = f['meta'].predict_proba(z)[:,1]
    np.testing.assert_allclose(p,expit(z@f['coef']+f['intercept']),atol=1e-10,rtol=0)
    return p


def train():
    protect()
    assert not (OUT/'training_frozen.json').exists() and not (OUT/'test_started.json').exists()
    cohort,y,tr,te = base.old.inputs()
    assert (y[te] == -1).all()
    tables = {'Clinical':five.clinical_source.clinical(tr)}
    joblib.dump(tables,OUT/'training_clinical.joblib',compress=3)
    metrics,predictions,inner,summary = [],[],[],[]
    for ph in PHASES:
        models = {}
        for _,t in SCHEMES:
            key = hkey(t,True)
            for rep,k,a,b in base.old.previous.parent.folds('Random120',ph):
                f = fit_hc(tables,source.context(ph,t,rep,k),y,t)
                inner.append({'ModelKey':key,'Phase':ph,'Repeat':rep,'Fold':k,
                              'TrainN':len(a),'ValidationN':len(b),'FitAUC':roc_auc_score(y[a],f['train_probability']),
                              'AUC':roc_auc_score(y[b],f['evaluation_probability'])})
            f = fit_hc(tables,source.context(ph,t),y,t)
            models[key] = f
            a,b = f['train'],f['validation']
            if ph != 'final':
                k = int(ph[-1])
                metrics.append({'ModelKey':key,'Fold':k,'FitAUC':roc_auc_score(y[a],f['train_probability']),
                                'AUC':roc_auc_score(y[b],f['evaluation_probability'])})
                predictions.extend({'ModelKey':key,'Fold':k,'Index':int(i),'AnalysisID':cohort.AnalysisID.iloc[i],
                                    'Outcome':int(y[i]),'Probability':float(v)} for i,v in zip(b,f['evaluation_probability']))
            else:
                z = pd.DataFrame(predictions)
                z = z[z.ModelKey == key].set_index('Index').loc[tr]
                f['threshold'] = five.threshold(y[tr],z.Probability.to_numpy())
                f['threshold_source'] = 'Own_fixedrecipe_all84_outerOOF_Youden'
                mm = pd.DataFrame(metrics)
                mm = mm[mm.ModelKey == key]
                summary.append({'ModelKey':key,'TrainFitAUC':roc_auc_score(y[tr],f['train_probability']),
                    'OuterMeanAUC':mm.AUC.mean(),'OuterSD':mm.AUC.std(ddof=1),
                    'PooledOOFAUC':roc_auc_score(y[tr],z.Probability),'Threshold':f['threshold'],
                    'MetaTrainingAssociationAUC':roc_auc_score(y[tr],f['meta_training_association_probability']),
                    'Reused':False})
        joblib.dump(models,OUT/'models'/(ph+'.joblib'),compress=3)
        print(ph+' two fixedHC models and30inner diagnostics complete;notest',flush=True)
    cols = ('TrainFitAUC','OuterMeanAUC','OuterSD','PooledOOFAUC','Threshold')
    old = five.read('training_choices').set_index('ModelKey')
    for key in SHARED:
        summary.append({'ModelKey':key,**{c:old.loc[key,c] for c in cols},'Reused':True})
    sm = five.read('outer_metrics')
    sp = five.read('nested_predictions')
    metrics += sm[sm.ModelKey.isin(SHARED)].to_dict('records')
    predictions += sp[sp.ModelKey.isin(SHARED)].to_dict('records')
    old = retained.read('training_summary').set_index('ModelKey')
    for _,t in SCHEMES:
        sourcekey = 'FixedT85' if t else 'FixedT0'
        key = hkey(t)
        f = joblib.load(retained.OUT/'models/frozen_fixed85.joblib')['fit'] if t else joblib.load(base.OUT/'models/R1_ContourMedian_final.joblib')['fit']
        summary.append({'ModelKey':key,**{c:old.loc[sourcekey,c] for c in cols if c != 'Threshold'},
                        'Threshold':f['threshold'],'Reused':True})
        m = retained.read('outer_metrics')
        p = retained.read('nested_predictions')
        m = m[m.ModelKey == sourcekey].copy()
        p = p[p.ModelKey == sourcekey].copy()
        m['ModelKey'],p['ModelKey'] = key,key
        metrics += m.to_dict('records')
        predictions += p.to_dict('records')
    for name,rows in [('training_summary',summary),('outer_metrics',metrics),('nested_predictions',predictions),('inner_fixed_hc_scores',inner)]:
        csv(name,rows)
    features=[]
    final=joblib.load(OUT/'models/final.joblib')
    names=[c['CandidateKey'] for c in base.old.previous.members()]
    for key,f in final.items():
        features.extend({'ModelKey':key,'Feature':name,'Type':'HabitatBaseScore','Coefficient':float(w)} for name,w in zip(names,f['coef'][:12]))
        features.extend({'ModelKey':key,'Feature':name,'Type':'Clinical','Coefficient':float(w)} for name,w in zip(f['clinical']['names'],f['coef'][12:]))
    csv('new_hc_coefficients',features)
    paths = {str(p.relative_to(OUT)):digest(p) for p in OUT.rglob('*') if p.is_file()}
    save(OUT/'training_frozen.json',{'passed':True,'new_test_calls':0,'new_final_models':2,'reused_models':5,'files':paths})
    print(pd.DataFrame(summary).to_string(index=False),flush=True)


def test():
    frozen()
    assert not (OUT/'test_started.json').exists()
    assert json.loads((OUT/'training_verification.json').read_text(encoding='utf-8'))['passed']
    cohort,_,tr,te = base.old.inputs()
    save(OUT/'test_started.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'new_complete_models':2,
         'reused_unique_models':5,'new_pure_habitat_calls':0,'thresholds_frozen':True})
    tables = joblib.load(base.OUT/'test_features.joblib')
    clinical = {'Clinical':five.clinical_source.clinical(te)}
    joblib.dump(clinical,OUT/'test_clinical.joblib',compress=3)
    final = joblib.load(OUT/'models/final.joblib')
    scores = {hkey(t,True):predict_hc(final[hkey(t,True)],source.basis('final',t),tables,clinical,te) for _,t in SCHEMES}
    old = retained.read('test_predictions').set_index('Index').loc[te]
    np.testing.assert_array_equal(old.AnalysisID,cohort.AnalysisID.iloc[te])
    scores['HabitatOriginal'] = old.BaselineProbability.to_numpy()
    scores['Habitat85'] = old.Fixed85Probability.to_numpy()
    oldfive = five.read('test_predictions')
    for key in SHARED:
        frame = oldfive[oldfive.ModelKey == key].set_index('Index').loc[te]
        np.testing.assert_array_equal(frame.AnalysisID,cohort.AnalysisID.iloc[te])
        scores[key] = frame.Probability.to_numpy()
    truth = cohort.Outcome.to_numpy(int)[te]
    np.testing.assert_array_equal(truth,old.Outcome)
    from verify_ivim_df63_20261020 import draws
    samples = draws(truth,scores,SEED)
    summary = read('training_summary').set_index('ModelKey')
    metrics,predictions = [],[]
    for key in UNIQUE:
        p = scores[key]
        threshold = float(summary.loc[key,'Threshold'])
        lo,hi = np.quantile(samples[key],[.025,.975])
        tn,fp,fn,tp = confusion_matrix(truth,p >= threshold,labels=[0,1]).ravel()
        metrics.append({'ModelKey':key,'N':len(te),'Positive':int(truth.sum()),'Negative':int((truth == 0).sum()),
             'TestAUC':roc_auc_score(truth,p),'AUC_CI95Low':float(lo),'AUC_CI95High':float(hi),'Threshold':threshold,
             'Sensitivity':tp/(tp+fn),'Specificity':tn/(tn+fp),'Accuracy':(tp+tn)/len(te),
             'TP':int(tp),'FN':int(fn),'TN':int(tn),'FP':int(fp),'ReusedSavedScores':key not in final})
        predictions.extend({'ModelKey':key,'Index':int(i),'AnalysisID':cohort.AnalysisID.iloc[i],
             'Outcome':int(y),'Probability':float(v),'Prediction':int(v >= threshold)} for i,y,v in zip(te,truth,p))
    csv('test_metrics',metrics)
    csv('test_predictions',predictions)
    paired = []
    for a,b in PAIRS:
        lo,hi = np.quantile(samples[a]-samples[b],[.025,.975])
        paired.append({'Comparison':a+'_minus_'+b,'AUC_Difference':roc_auc_score(truth,scores[a])-roc_auc_score(truth,scores[b]),
                       'CI95Low':float(lo),'CI95High':float(hi),'BootstrapN':5000})
    csv('paired_test_differences',paired)
    frozen()
    save(OUT/'test_complete.json',{'passed':True,'new_complete_models':2,'new_predictions':72,
         'reused_unique_models':5,'reused_predictions':180,'unique_models':7,'display_rows':10,
         'joint_test_stages':1,'new_pure_habitat_calls':0,'post_test_fits':0,'post_test_threshold_changes':0,'formal_adoption_changed':False})
    print(pd.DataFrame(metrics).to_string(index=False),flush=True)


def seal():
    frozen()
    for n in ('training_verification','test_complete','report_verification','visual_verification'):
        assert json.loads((OUT/(n+'.json')).read_text(encoding='utf-8'))['passed']
    files={str(p.relative_to(OUT)):digest(p) for p in OUT.rglob('*') if p.is_file()
           and p.name not in ('delivery_manifest.json','delivery_final_verification.json')}
    save(OUT/'delivery_manifest.json',{'outputs':files})
    for n,sha in files.items():
        assert digest(OUT/n) == sha,n
    save(OUT/'delivery_final_verification.json',{'passed':True,'outputs_checked':len(files),'only1mm':True,
         'five_families_per_group':True,'unique_models':7,'new_complete_test_models':2,'all5folds_kept':True,
         'formal_adoption_changed':False})
    print('Original/85 five-family comparison delivered;oldrecords unchanged',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=('define','train','test','seal'))
    args = parser.parse_args()
    with threadpool_limits(1):
        globals()[args.action]()
