"""One declared nested experiment: bootstrap habitats, textures, LR/RBF SVM."""
import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path
import lvi_paths
import joblib
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix
from threadpoolctl import threadpool_limits
import fold_locked_lvi_engine as engine
import slic_habitat_backend as slic
import random120_bootstrap_texture_features as feature
import run_random120_retained_methods_20261010 as retained
import run_random120_optimization_20261012 as previous

ROOT = retained.ROOT
OUT = ROOT / "lvi_prediction/results/lvi_random120_texture_bootstrap_20261012"
ALGORITHMS = [("LR", .03), ("LR", .3), ("LR", 3.), ("RBF", .3), ("RBF", 3.)]
MODS = {"ADC": ("ADC",), "T2ADC": ("T2", "ADC")}
ANALYSES = ["Habitat", "HabitatWhole", "WholeControl", "HabitatClinical", "HybridClinical", "ClinicalControl"]
TRAIN_ALLOWED, PROFILES, TEXTURES, CLINICAL = set(), None, None, None
save, digest = retained.save, retained.digest


def options():
    result = []
    for family, quotas in [("Habitat", [(3,0),(5,0)]), ("Hybrid", [(2,3),(3,5)]), ("Whole", [(0,3),(0,5)])]:
        for mod in MODS:
            for clinical in [False, True]:
                for top_h, top_w in quotas:
                    for algorithm, c in ALGORITHMS:
                        result.append({"family": family, "modality": mod, "clinical": clinical,
                            "top_h": top_h, "top_w": top_w, "algorithm": algorithm, "C": c})
    result += [{"family": "Clinical", "modality": "None", "clinical": True, "top_h": 0, "top_w": 0,
                "algorithm": "LR", "C": c} for c in [.03,.3,3.]]
    for i, row in enumerate(result): row["candidate_id"] = i
    return result


def eligible(c, analysis):
    family, clinical = {"Habitat":("Habitat",False),"HabitatWhole":("Hybrid",False),
        "WholeControl":("Whole",False),"HabitatClinical":("Habitat",True),
        "HybridClinical":("Hybrid",True),"ClinicalControl":("Clinical",True)}[analysis]
    return c["family"] == family and c["clinical"] == clinical


def choose(rows, analysis):
    return max([r for r in rows if eligible(r, analysis)], key=lambda r:(r["Objective"],
        -r["MeanImageCount"], r["algorithm"]=="LR", -r["C"], -r["candidate_id"]))


def config(row): return {k:row[k] for k in options()[0]}


def freeze():
    if (OUT/"protocol.json").exists(): raise FileExistsError("Frozen experiment: no expanded search or overwriting")
    OUT.mkdir(parents=True,exist_ok=True)
    previous.check_sources()
    assert json.loads((previous.OUT/"independent_verification.json").read_text())["passed"]
    cohort, manifest, y, tr, te = previous.load_inputs()
    old=json.loads((previous.OUT/"protocol.json").read_text(encoding="utf-8"))
    sources=dict(old["source_hashes"])
    for p in [Path(__file__).resolve(),ROOT/"lvi_prediction/random120_bootstrap_texture_features.py",
        previous.OUT/"delivery_manifest.json"]: sources[str(p.relative_to(ROOT))]=digest(p)
    protected=set(json.loads((previous.OUT/"preservation_before.json").read_text(encoding="utf-8")))
    protected.update(str(p.relative_to(ROOT)) for p in (ROOT/"lvi_prediction/results").rglob("*")
        if p.is_file() and OUT not in p.parents and "tmp" not in p.parts and "__pycache__" not in p.parts)
    save(OUT/"preservation_before.json",{name:digest(ROOT/name) for name in sorted(protected)})
    protocol={"date":"2026-10-12","authorization":"继续尝试接近0.75；允许加入T2/ADC全肿瘤纹理，分开报告",
        "target_AUC":.75,"split":manifest,"candidates":options(),"candidate_count":len(options()),"analyses":ANALYSES,
        "feature_scope":"T2/ADC only; no IVIM predictors, shape-transfer experiment or discarded nested5 model",
        "habitat":"K3 SLIC; mean of 5 outcome-blind training-patient bootstrap partitions; native cached inner band descriptors and first-order intensity",
        "habitat_gate":"training bootstrap partition ICC(A,1)>=.8, >=90% complete and nonconstant; this is partition repeatability, not contour robustness",
        "texture":"non-shape PyRadiomics Original firstorder/GLCM/GLRLM/GLSZM/GLDM/NGTDM; binWidth .25; patient Z-score, 1mm; base/erode/dilate conditional texture ICC>=.8",
        "hybrid_quota":"top2 habitat plus top3 whole, or top3 habitat plus top5 whole; never silently rename a whole-only result as pure habitat",
        "selection":"current training fold only: rank effect*ICC^2, abs Spearman<.9, requested block quota",
        "preprocess":"training median + 1/99% winsorization + zscore",
        "models":"L2 LR C=.03/.3/3; RBF SVM C=.3/3 gamma=scale probability=False",
        "scores":"LR probabilities or SVM raw decision scores; AUC/thresholds use these scores. SVM scores are not calibrated risks.",
        "clinical_columns":list(engine.CLINICAL_COLUMNS),"clinical_source_timing_pending":True,
        "nested":{"outer_folds":5,"outer_seed":42,"inner_folds":3,"inner_seeds":[62201+i for i in range(5)]},
        "final_tuning":{"folds":5,"seed":62212},
        "objective":"mean inner AUC - .05*SD - .20*max(mean fit AUC-mean validation AUC-.10,0) - .03*abs(mean fit AUC-.75)",
        "test":"six training-chosen category finalists in one joint evaluation after freezing; no selecting by test-highest or test-nearest-.75",
        "test_previously_reused":True,"external_independent_validation":False,"automatic_adoption":False,
        "source_hashes":sources}
    save(OUT/"protocol.json",protocol);save(OUT/"split_manifest.json",manifest)
    (OUT/"fold_features").mkdir(exist_ok=True);(OUT/"models").mkdir(exist_ok=True)
    print(f"Frozen {len(options())} candidates; unchanged 120 cases and 84/36 split",flush=True)


def check_sources():
    p=json.loads((OUT/"protocol.json").read_text(encoding="utf-8"))
    for name,h in p["source_hashes"].items(): assert digest(ROOT/name)==h,name


def preserved():
    before=json.loads((OUT/"preservation_before.json").read_text(encoding="utf-8"))
    for name,h in before.items(): assert digest(ROOT/name)==h,name
    return len(before)


def texture_subset(indices):
    lookup={int(i):j for j,i in enumerate(TEXTURES["indices"])}
    rows=np.asarray([lookup[int(i)] for i in indices],int)
    return {**{kind:TEXTURES[kind][rows] for kind in ["base","erode","dilate"]},"metadata":TEXTURES["metadata"]}


def part(a,b,seed):
    a,b=np.asarray(a,int),np.asarray(b,int)
    assert set(a)<=TRAIN_ALLOWED and set(b)<=TRAIN_ALLOWED and not set(a)&set(b)
    path=OUT/"fold_features"/f"part_{seed}.joblib"
    if path.exists():
        p=joblib.load(path);np.testing.assert_array_equal(p["train"],a);np.testing.assert_array_equal(p["evaluation"],b)
        return p
    habitat=feature.bootstrap_tables(PROFILES,a,b,seed)
    p={"train":a,"evaluation":b,"seed":seed,"Habitat":habitat,
       "Whole":{"train":texture_subset(a),"evaluation":texture_subset(b)}}
    joblib.dump(p,path,compress=3)
    return p


def rank_block(measurements,metadata,y,modalities,reference=None):
    base=np.nanmean(measurements,axis=1) if reference is None else reference
    selected,audit,ranking,ranks=[],[],[],{}
    for j,m in enumerate(metadata):
        if m["modality"] not in modalities: continue
        triple=measurements[:,:,j]
        finite=np.isfinite(triple).all(axis=1)
        icc=engine.icc_absolute(triple[finite]) if finite.sum()>=3 else -np.inf
        passed=bool(finite.mean()>=.9 and np.isfinite(icc) and icc>=.8 and np.nanstd(base[:,j])>1e-6)
        row={"Feature":m["name"],"ICC":float(icc) if np.isfinite(icc) else None,"CompleteFraction":float(finite.mean()),"Gate":passed}
        if passed:
            x=base[:,j];x=np.where(np.isfinite(x),x,np.nanmedian(x))
            effect=abs(float(mannwhitneyu(x[y==1],x[y==0]).statistic)/(int((y==1).sum())*int((y==0).sum()))-.5)
            ranking.append((j,effect*icc**2,icc));ranks[j]=rankdata(x);row["Effect"]=effect
        audit.append(row)
    for j,score,icc in sorted(ranking,key=lambda v:(-v[1],-v[2],metadata[v[0]]["name"])):
        if all(abs(float(np.corrcoef(ranks[j],ranks[k])[0,1]))<.9 for k in selected): selected.append(j)
    return selected,audit,base


def prepare(current,y,c,rank_cache=None):
    a,b=current["train"],current["evaluation"]
    indices,names,audit,xa,xb,block_names={},[],{},[],[],{}
    rank_cache={} if rank_cache is None else rank_cache
    for block,top in [("Habitat",c["top_h"]),("Whole",c["top_w"])]:
        if not top: continue
        if block=="Habitat":
            data=current[block];measured=data["train_measurements"];meta=data["metadata"]
            va=np.nanmean(measured,axis=1);vb=np.nanmean(data["evaluation_measurements"],axis=1)
        else:
            data=current[block];meta=data["train"]["metadata"]
            measured=np.stack([data["train"][k] for k in ["base","erode","dilate"]],axis=1)
            va=data["train"]["base"];vb=data["evaluation"]["base"]
        key=(block,c["modality"])
        if key not in rank_cache: rank_cache[key]=rank_block(measured,meta,y[a],MODS[c["modality"]],va)
        selected,details,ignored=rank_cache[key]
        idx=selected[:top];indices[block]=idx;audit[block]=details
        block_names[block]=[m["name"] for m in meta]
        names.extend(meta[j]["name"] for j in idx);xa.append(va[:,idx]);xb.append(vb[:,idx])
    if c["clinical"]:
        xa.append(CLINICAL[a]);xb.append(CLINICAL[b]);names.extend(engine.CLINICAL_COLUMNS)
    xa=np.column_stack(xa) if xa else np.empty((len(a),0));xb=np.column_stack(xb) if xb else np.empty((len(b),0))
    med=np.nanmedian(np.where(np.isfinite(xa),xa,np.nan),axis=0) if xa.shape[1] else np.array([])
    med=np.where(np.isfinite(med),med,0.)
    filled=np.where(np.isfinite(xa),xa,med)
    low,high=np.quantile(filled,[.01,.99],axis=0) if xa.shape[1] else (np.array([]),np.array([]))
    filled=np.clip(filled,low,high);mean,scale=filled.mean(0),filled.std(0)
    scale=np.where(scale>1e-8,scale,1.)
    return {"indices":indices,"names":names,"audit":audit,"block_names":block_names,"image_count":sum(map(len,indices.values())),
        "median":med,"low":low,"high":high,"mean":mean,"scale":scale,
        "xa":(filled-mean)/scale,"xb":(np.clip(np.where(np.isfinite(xb),xb,med),low,high)-mean)/scale}


def estimator(c):
    return LogisticRegression(C=c["C"],penalty="l2",solver="liblinear",max_iter=3000,random_state=62212) if c["algorithm"]=="LR" else SVC(C=c["C"],kernel="rbf",gamma="scale",probability=False,random_state=62212)


def scores(model,x,c,prevalence):
    if not len(x): return np.array([])
    if model is None: return np.full(len(x),prevalence)
    return model.predict_proba(x)[:,1] if c["algorithm"]=="LR" else model.decision_function(x)


def fit_prepared(p,y,c):
    model=estimator(c).fit(p["xa"],y) if p["xa"].shape[1] else None
    prevalence=float(y.mean())
    return {"config":c,"model":model,"indices":p["indices"],"feature_names":p["names"],"feature_audit":p["audit"],"block_names":p["block_names"],
        "image_count":p["image_count"],**{k:p[k] for k in ["median","low","high","mean","scale"]},
        "train_score":scores(model,p["xa"],c,prevalence),"evaluation_score":scores(model,p["xb"],c,prevalence),"prevalence":prevalence}


def fit_model(p,y,c):
    result=fit_prepared(prepare(p,y,c),y[p["train"]],c)
    result.update({"train":p["train"],"evaluation":p["evaluation"],"seed":p["seed"]})
    return result


def tune(parts,y):
    raw={c["candidate_id"]:[] for c in options()};probability={c["candidate_id"]:[] for c in options()}
    for p in parts:
        a,b=p["train"],p["evaluation"];prepared={};rank_cache={}
        for c in options():
            key=(c["family"],c["modality"],c["clinical"],c["top_h"],c["top_w"])
            if key not in prepared: prepared[key]=prepare(p,y,c,rank_cache)
            f=fit_prepared(prepared[key],y[a],c)
            raw[c["candidate_id"]].append((roc_auc_score(y[b],f["evaluation_score"]),roc_auc_score(y[a],f["train_score"]),f["image_count"]))
            probability[c["candidate_id"]].append((b,f["evaluation_score"]))
    rows=[]
    for c in options():
        v=np.asarray(raw[c["candidate_id"]]);mean,sd,fit=float(v[:,0].mean()),float(v[:,0].std(ddof=1)),float(v[:,1].mean())
        rows.append({**c,"MeanValidationAUC":mean,"ValidationSD":sd,"MeanFitAUC":fit,"MeanImageCount":float(v[:,2].mean()),
            "Objective":mean-.05*sd-.20*max(fit-mean-.10,0)-.03*abs(fit-.75)})
    return rows,probability


def threshold(y,s):
    fpr,tpr,values=roc_curve(y,s,drop_intermediate=False)
    valid=np.flatnonzero(np.isfinite(values))
    j=max(valid,key=lambda k:(float(tpr[k]-fpr[k]),float(values[k])))
    return float(values[j])


def train():
    global TRAIN_ALLOWED,PROFILES,TEXTURES,CLINICAL
    start=time.time();freeze();check_sources()
    cohort,manifest,y,tr,te=previous.load_inputs();TRAIN_ALLOWED=set(tr);PROFILES=retained.load_profiles()
    CLINICAL=pd.read_csv(previous.CLINICAL_PATH).set_index("AnalysisID").loc[cohort.AnalysisID,list(engine.CLINICAL_COLUMNS)].to_numpy(float)
    TEXTURES=feature.whole_textures(PROFILES,tr,cohort.AnalysisID.to_numpy())
    joblib.dump(TEXTURES,OUT/"training_texture_features.joblib",compress=3)
    pd.DataFrame(TEXTURES["audit"]).to_csv(OUT/"training_texture_audit.csv",index=False,encoding="utf-8-sig")
    predictions,metrics,grids=[],[],[]
    for fold,(a,b) in enumerate(previous.folds(tr,y,42,5),1):
        inner=[part(ia,ib,622000+fold*100+j) for j,(ia,ib) in enumerate(previous.folds(a,y,62200+fold,3))]
        grid,_=tune(inner,y);grids.extend({"OuterFold":fold,**row} for row in grid)
        outer=part(a,b,623000+fold*100);stored={"grid":grid,"models":{}}
        for analysis in ANALYSES:
            c=config(choose(grid,analysis));f=fit_model(outer,y,c);stored["models"][analysis]=f
            metrics.append({"Analysis":analysis,"Fold":fold,**c,"AUC":float(roc_auc_score(y[b],f["evaluation_score"])),
                "TrainFitAUC":float(roc_auc_score(y[a],f["train_score"])),"ImageFeatureCount":f["image_count"]})
            predictions.extend({"Analysis":analysis,"Fold":fold,"AnalysisID":cohort.AnalysisID.iloc[i],"Outcome":int(y[i]),"Score":float(s)} for i,s in zip(b,f["evaluation_score"]))
        joblib.dump(stored,OUT/f"models/outer_fold_{fold}.joblib",compress=3)
        print(f"Nested outer {fold}/5:",json.dumps({row["Analysis"]:row["AUC"] for row in metrics if row["Fold"]==fold}),flush=True)
    pred,fold_metrics=pd.DataFrame(predictions),pd.DataFrame(metrics)
    summary=[]
    for analysis in ANALYSES:
        rows=pred[pred.Analysis==analysis].set_index("AnalysisID").loc[cohort.AnalysisID.iloc[tr]]
        auc=fold_metrics[fold_metrics.Analysis==analysis].AUC
        summary.append({"Analysis":analysis,"OuterMeanAUC":float(auc.mean()),"OuterSD":float(auc.std(ddof=1)),"PooledOOFAUC":float(roc_auc_score(y[tr],rows.Score))})
    pred.to_csv(OUT/"nested_predictions.csv",index=False,encoding="utf-8-sig")
    fold_metrics.to_csv(OUT/"outer_fold_metrics.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(grids).to_csv(OUT/"inner_candidate_scores.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(summary).to_csv(OUT/"nested_summary.csv",index=False,encoding="utf-8-sig")
    final_parts=[part(a,b,624000+j) for j,(a,b) in enumerate(previous.folds(tr,y,62212,5))]
    grid,cv=tune(final_parts,y);pd.DataFrame(grid).to_csv(OUT/"final_training_grid.csv",index=False,encoding="utf-8-sig")
    full=part(tr,[],625000);final={"models":{},"habitat_partitions":full["Habitat"]["partitions"],"train":tr}
    chosen,train_rows,coefficients=[],[],[]
    for analysis in ANALYSES:
        best=choose(grid,analysis);c=config(best);f=fit_model(full,y,c)
        oof=np.full(138,np.nan)
        for idx,s in cv[c["candidate_id"]]:oof[idx]=s
        assert np.isfinite(oof[tr]).all() and np.isnan(oof[te]).all()
        f["threshold"]=threshold(y[tr],oof[tr]);final["models"][analysis]=f
        chosen.append({"Analysis":analysis,**c,"TrainFitAUC":float(roc_auc_score(y[tr],f["train_score"])),
            "FinalSelectionCVMeanAUC":best["MeanValidationAUC"],"ImageFeatureCount":f["image_count"],"Threshold":f["threshold"]})
        train_rows.extend({"Analysis":analysis,"AnalysisID":cohort.AnalysisID.iloc[i],"Outcome":int(y[i]),"TrainFitScore":float(s),"FinalSelectionOOFScore":float(v)} for i,s,v in zip(tr,f["train_score"],oof[tr]))
        beta=f["model"].coef_[0] if f["model"] is not None and c["algorithm"]=="LR" else [None]*len(f["feature_names"])
        coefficients.extend({"Analysis":analysis,"Feature":name,"LRStandardizedCoefficient":b,"Algorithm":c["algorithm"]} for name,b in zip(f["feature_names"],beta))
    joblib.dump(final,OUT/"models/final_models.joblib",compress=3)
    pd.DataFrame(chosen).to_csv(OUT/"final_training_choices.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(train_rows).to_csv(OUT/"train_fit_scores.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(coefficients).to_csv(OUT/"selected_features.csv",index=False,encoding="utf-8-sig")
    save(OUT/"training_frozen.json",{"passed":True,"model_sha256":digest(OUT/"models/final_models.joblib"),
        "protocol_sha256":digest(OUT/"protocol.json"),"test_score_generated":False,
        "training_csv_hashes":{p.name:digest(p) for p in OUT.glob("*.csv")},
        "cache_hashes":{p.name:digest(p) for p in (OUT/"fold_features").glob("*.joblib")},
        "texture_sha256":digest(OUT/"training_texture_features.joblib"),"protected_file_count":preserved(),"elapsed_seconds":time.time()-start})
    save(OUT/"candidate_status.json",{"status":"exploratory_not_automatically_adopted","formal_model_use":False,"previous_decisions_unchanged":True,"test_previously_reused":True})
    print(pd.DataFrame(summary).to_string(index=False),flush=True);print(pd.DataFrame(chosen).to_string(index=False),flush=True)
    print("Six training-chosen finalists frozen; test unscored this round",flush=True)


def predict(f,raw,clinical):
    arrays=[]
    for block in ["Habitat","Whole"]:
        if block in f["indices"]:
            assert [m["name"] for m in raw[block]["metadata"]]==f["block_names"][block]
            arrays.append(raw[block]["base"][:,f["indices"][block]])
    if f["config"]["clinical"]: arrays.append(clinical)
    x=np.column_stack(arrays) if arrays else np.empty((len(clinical),0))
    x=(np.clip(np.where(np.isfinite(x),x,f["median"]),f["low"],f["high"])-f["mean"])/f["scale"]
    return scores(f["model"],x,f["config"],f["prevalence"])


def metrics(y,s,cutoff):
    tn,fp,fn,tp=map(int,confusion_matrix(y,s>=cutoff,labels=[0,1]).ravel())
    return {"AUC":float(roc_auc_score(y,s)),"AUPRC":float(average_precision_score(y,s)),"Threshold":cutoff,
        "TP":tp,"FN":fn,"TN":tn,"FP":fp,"Sensitivity":tp/(tp+fn),"Specificity":tn/(tn+fp),"Accuracy":(tp+tn)/len(y)}


def evaluate():
    if (OUT/"test_evaluation_started.json").exists():raise FileExistsError("Joint evaluation already started")
    check_sources();frozen=json.loads((OUT/"training_frozen.json").read_text(encoding="utf-8"))
    assert frozen["model_sha256"]==digest(OUT/"models/final_models.joblib")
    assert frozen["protocol_sha256"]==digest(OUT/"protocol.json")
    for name,h in frozen["training_csv_hashes"].items():assert digest(OUT/name)==h,name
    cohort,manifest,ymasked,tr,te=previous.load_inputs();final=joblib.load(OUT/"models/final_models.joblib")
    save(OUT/"test_evaluation_started.json",{"joint_evaluation_count":1,"model_sha256":frozen["model_sha256"],"test_ids":manifest["test_ids"]})
    profiles=retained.load_profiles()
    whole=feature.whole_textures(profiles,te,cohort.AnalysisID.to_numpy())
    habitat=feature.inference_habitats(profiles,te,final["habitat_partitions"])
    clinical=pd.read_csv(previous.CLINICAL_PATH).set_index("AnalysisID").loc[cohort.AnalysisID.iloc[te],list(engine.CLINICAL_COLUMNS)].to_numpy(float)
    raw={"Habitat":habitat,"Whole":whole}
    joblib.dump({"raw":raw,"clinical":clinical,"indices":te},OUT/"test_inference_inputs.joblib",compress=3)
    predictions={a:predict(f,raw,clinical) for a,f in final["models"].items()};y=cohort.Outcome.to_numpy(int)[te]
    draws=retained.bootstrap(y,predictions);rows,result=[],[]
    for analysis,s in predictions.items():
        low,high=np.quantile(draws[analysis],[.025,.975]);f=final["models"][analysis]
        result.append({"Analysis":analysis,"Algorithm":f["config"]["algorithm"],**metrics(y,s,f["threshold"]),"AUC_CI95Low":float(low),"AUC_CI95High":float(high)})
        rows.extend({"Analysis":analysis,"AnalysisID":cohort.AnalysisID.iloc[i],"Outcome":int(value),"Score":float(score),"PredictedPositive":int(score>=f["threshold"])} for i,value,score in zip(te,y,s))
    pd.DataFrame(rows).to_csv(OUT/"test_scores.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(result).to_csv(OUT/"test_metrics.csv",index=False,encoding="utf-8-sig")
    save(OUT/"evaluation_complete.json",{"passed":True,"joint_evaluation_count":1,"test_patients":36,"test_score_rows":len(rows),
        "score_sha256":digest(OUT/"test_scores.csv"),"metrics_sha256":digest(OUT/"test_metrics.csv"),"protected_file_count":preserved(),"automatic_adoption":False})
    print(pd.DataFrame(result).to_string(index=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--phase",choices=["train","evaluate"],required=True);args=parser.parse_args()
    with threadpool_limits(limits=1):
        slic.threadpool_limits=lambda **kw:nullcontext()
        train() if args.phase=="train" else evaluate()
