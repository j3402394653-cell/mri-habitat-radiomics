"""Fixed three-sequence comparison: T2, ADC, IVIM, free fusion, and sequence-quota fusion."""
import argparse
from pathlib import Path
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
import lvi_paths
import multiseq_habitat_features_20261012 as feature
import run_regional_habitat_120_20261012 as common
import run_localtexture_habitat_120_20261012 as prior

ROOT = common.ROOT
OUT = ROOT/"lvi_prediction/results/lvi_multiseq_habitat_120_20261012"
CACHE = prior.OUT
GROUPS = {"T2":("T2",),"ADC":("ADC",),"IVIM":("IVIM",),
          "FusionFree":("T2","ADC","IVIM"),"FusionQuota":("T2","ADC","IVIM")}
LABELS = {"T2":"T2生境","ADC":"ADC生境","IVIM":"IVIM生境（D＋f）",
          "FusionFree":"三序列生境：自由筛选","FusionQuota":"三序列生境：按序列配额"}
save,digest = common.save,common.digest


def options():
    result = []
    for analysis in GROUPS:
        for family in ["Descriptor","Texture","Combined"]:
            for top in [3,6]:
                for C in [.03,.3,1.]:
                    result.append({"analysis":analysis,"family":family,"top":top,
                        "quota":top//3 if analysis=="FusionQuota" else 0,"algorithm":"LR","C":C})
    for i,r in enumerate(result):r["candidate_id"]=i
    return result


def config(row):return {k:row[k] for k in options()[0]}


def choose(grid,analysis):
    rows = [r for r in grid if r["analysis"]==analysis]
    valid = [r for r in rows if r["AvailableAllFolds"]]
    pool = valid if valid else rows
    return max(pool,key=lambda r:(r["Objective"],-r["MeanFeatureCount"],-r["C"],-r["candidate_id"]))


def freeze():
    if OUT.exists():raise FileExistsError("Three-sequence comparison already frozen")
    cohort,manifest,y,tr,te = common.old.previous.load_inputs()
    old = json.loads((CACHE/"training_frozen.json").read_text(encoding="utf-8"))
    assert digest(CACHE/"training_regional_features.joblib")==old["features_sha256"]
    assert json.loads((CACHE/"test_verification.json").read_text())["passed"]
    protected = dict(json.loads((CACHE/"protocol.json").read_text(encoding="utf-8"))["protected_hashes"])
    for directory in [CACHE,ROOT/"lvi_prediction/results/lvi_latent_habitat_120_20261012",
                      ROOT/"lvi_prediction/results/lvi_habitat_improvement_120_20261012"]:
        protected.update({str(p.relative_to(ROOT)):digest(p) for p in directory.rglob("*") if p.is_file()})
    for p in [Path(__file__).resolve(),ROOT/"lvi_prediction/multiseq_habitat_features_20261012.py",
              ROOT/"lvi_prediction/localtexture_habitat_features_20261012.py",
              ROOT/"lvi_prediction/run_localtexture_habitat_120_20261012.py",common.old.retained.PROFILE,
              common.old.retained.POOL/"cohort_138.csv",ROOT/"image_preprocessing.py"]:
        protected[str(p.relative_to(ROOT))]=digest(p)
    OUT.mkdir(parents=True);(OUT/"models").mkdir()
    save(OUT/"protocol.json",{"date":"2026-10-12","authorization":"进行更完整的改进：T2、ADC、IVIM及三序列联合，对比自由与配额筛选",
        "same_random120_and_84_36_split":True,"sample_seed":20261010,"split_seed":42,
        "feature_protocol":feature.PROTOCOL,"groups":GROUPS,"group_labels":LABELS,
        "candidates":options(),"candidate_count":90,"per_group_candidates":18,
        "nested":"outer5 seed42, inner5 seed62201..62205; final training selection5 seed62212",
        "robustness":"current training contour ICC>=.8, complete>=.9, 5 class-stratified bootstrap signed effect; sign consistency>=.8",
        "ranking":"abs mean signed effect * ICC^2 * sign consistency; current training only",
        "free_selection":"total3/6, global abs Spearman<.8",
        "quota_selection":"1/2 robust features per sequence, redundancy abs Spearman<.8 within each sequence; cross-sequence correlation handled by ridge regularization",
        "quota_unavailable":"do not force unstable features; unavailable configurations excluded where valid alternatives exist; prior-only fallback recorded if none available",
        "preprocessing":"training median, 1/99% training winsorization, training standardization",
        "objective":"mean inner AUC-.10*SD-.30*max(mean fit-mean validation-.10,0)-.03*abs(mean fit-.75)",
        "test":"five training-selected category finalists frozen, one joint scoring stage; no choosing final adopted winner by test AUC",
        "test_previously_reused":True,"clinical_MRI_timing_and_pathology_definition":"unknown",
        "whole_tumor_and_clinical_predictors":False,"automatic_adoption":False,"excluded_experiments_used":False,
        "sources":{"PyRadiomics":"https://pyradiomics.readthedocs.io/en/v3.0/customization.html"},"protected_hashes":protected})
    save(OUT/"split_manifest.json",manifest)
    return cohort,manifest,y,tr,te


def check_sources():
    p = json.loads((OUT/"protocol.json").read_text(encoding="utf-8"))
    for n,h in p["protected_hashes"].items():assert digest(ROOT/n)==h,n
    return len(p["protected_hashes"])


def ranking(data,train,y):return prior.ranking(data,train,y)


def prepare(data,train,evaluation,c,y,ranked=None):
    ordered,ranks,audit = ranking(data,train,y) if ranked is None else ranked
    allowed = GROUPS[c["analysis"]]
    pool = [j for j in ordered if data["metadata"][j]["modality"] in allowed
            and (c["family"]=="Combined" or data["metadata"][j]["block"]==c["family"])]
    selected,available = [],True
    if c["quota"]:
        for mod in allowed:
            current = []
            for j in pool:
                if data["metadata"][j]["modality"]!=mod:continue
                if all(abs(float(np.corrcoef(ranks[:,j],ranks[:,k])[0,1]))<.8 for k in current):current.append(j)
                if len(current)==c["quota"]:break
            if len(current)<c["quota"]:available=False
            selected.extend(current)
    else:
        for j in pool:
            if all(abs(float(np.corrcoef(ranks[:,j],ranks[:,k])[0,1]))<.8 for k in selected):selected.append(j)
            if len(selected)==c["top"]:break
        available = bool(selected)
    if not available:selected=[]
    xa = data["base"][common.rows(data,train)][:,selected]
    xb = data["base"][common.rows(data,evaluation)][:,selected]
    if selected:
        median = np.nanmedian(np.where(np.isfinite(xa),xa,np.nan),axis=0)
        filled = np.where(np.isfinite(xa),xa,median);low,high = np.quantile(filled,[.01,.99],axis=0)
        filled = np.clip(filled,low,high);mean,scale = filled.mean(0),filled.std(0);scale = np.where(scale>1e-12,scale,1.)
        xa = (filled-mean)/scale;xb = (np.clip(np.where(np.isfinite(xb),xb,median),low,high)-mean)/scale
    else:median,low,high,mean,scale = [np.array([]) for _ in range(5)]
    return {"indices":selected,"names":[data["metadata"][j]["name"] for j in selected],"audit":audit,
            "median":median,"low":low,"high":high,"mean":mean,"scale":scale,"xa":xa,"xb":xb,"available":available}


def fitted(p,y,c):
    f = common.fitted(p,y,c);f["available"]=p["available"]
    return f


def fit(data,a,b,y,c):return fitted(prepare(data,a,b,c,y),y[a],c)


def tune(data,folds,y):
    values = {c["candidate_id"]:[] for c in options()};cv = {c["candidate_id"]:[] for c in options()}
    for a,b in folds:
        rank = ranking(data,a,y);cache = {}
        for c in options():
            key = (c["analysis"],c["family"],c["top"])
            if key not in cache:cache[key]=prepare(data,a,b,c,y,rank)
            f = fitted(cache[key],y[a],c)
            values[c["candidate_id"]].append((roc_auc_score(y[b],f["evaluation_probability"]),
                roc_auc_score(y[a],f["train_probability"]),len(f["indices"]),f["available"]))
            cv[c["candidate_id"]].append((b,f["evaluation_probability"]))
    records = []
    for c in options():
        v = np.asarray(values[c["candidate_id"]]);mean,sd,train,n = float(v[:,0].mean()),float(v[:,0].std(ddof=1)),float(v[:,1].mean()),float(v[:,2].mean())
        valid = bool(v[:,3].all())
        objective = mean-.10*sd-.30*max(train-mean-.10,0)-.03*abs(train-.75)
        records.append({**c,"MeanValidationAUC":mean,"SD":sd,"MeanFitAUC":train,"MeanFeatureCount":n,
                        "AvailableAllFolds":valid,"Objective":objective if valid else -1.})
    return records,cv


def train():
    started=time.time();cohort,manifest,y,tr,te=freeze()
    profiles = common.old.retained.load_profiles()
    cache = joblib.load(CACHE/"training_regional_features.joblib")
    data = feature.extract(profiles,tr,cohort.AnalysisID.to_numpy(),cache);del profiles
    assert len(data["metadata"])==1242
    joblib.dump(data,OUT/"training_multiseq_features.joblib",compress=3)
    for key,name in [("IVIM_extraction_audit","training_IVIM_extraction_audit.csv"),
                     ("IVIM_quality_audit","training_IVIM_quality_audit.csv"),
                     ("IVIM_partition_audit","training_IVIM_partition_audit.csv")]:
        pd.DataFrame(data[key]).to_csv(OUT/name,index=False,encoding="utf-8-sig")
    predictions,metrics,grids = [],[],[]
    for fold,(a,b) in enumerate(common.old.previous.folds(tr,y,42,5),1):
        grid,_ = tune(data,common.old.previous.folds(a,y,62200+fold,5),y)
        stored = {"grid":grid,"models":{},"train":a,"evaluation":b}
        grids.extend({"OuterFold":fold,**r} for r in grid)
        for analysis in GROUPS:
            c=config(choose(grid,analysis));f=fit(data,a,b,y,c);stored["models"][analysis]=f
            metrics.append({"Analysis":analysis,"Fold":fold,**c,"FeatureCount":len(f["indices"]),
                            "Available":f["available"],"AUC":float(roc_auc_score(y[b],f["evaluation_probability"]))})
            predictions.extend({"Analysis":analysis,"Fold":fold,"AnalysisID":cohort.AnalysisID.iloc[i],
                                "Outcome":int(y[i]),"Probability":float(p)} for i,p in zip(b,f["evaluation_probability"]))
        joblib.dump(stored,OUT/f"models/outer_fold_{fold}.joblib",compress=3)
        print(f"Three-sequence nested fold {fold}/5 completed",flush=True)
    pred,folds = pd.DataFrame(predictions),pd.DataFrame(metrics)
    grid,cv = tune(data,common.old.previous.folds(tr,y,62212,5),y)
    models,choices,features,trainpred = {},[],[],[]
    for analysis in GROUPS:
        chosen=choose(grid,analysis);c=config(chosen);f=fit(data,tr,[],y,c)
        oof=np.full(138,np.nan)
        for ix,p in cv[c["candidate_id"]]:oof[ix]=p
        assert np.isfinite(oof[tr]).all() and np.isnan(oof[te]).all()
        f["threshold"]=common.threshold(y[tr],oof[tr]);f["train"]=tr;models[analysis]=f
        ff=folds[folds.Analysis==analysis].AUC
        pp=pred[pred.Analysis==analysis].set_index("AnalysisID").loc[cohort.AnalysisID.iloc[tr]]
        counts={mod:sum(data["metadata"][j]["modality"]==mod for j in f["indices"]) for mod in feature.MODALITIES}
        choices.append({"Analysis":analysis,"Model":LABELS[analysis],**c,"Available":f["available"],"FeatureCount":len(f["indices"]),
            **{mod+"Features":n for mod,n in counts.items()},"TrainFitAUC":float(roc_auc_score(y[tr],f["train_probability"])),
            "FinalSelectionCVMeanAUC":chosen["MeanValidationAUC"],"OuterMeanAUC":float(ff.mean()),"OuterSD":float(ff.std(ddof=1)),
            "PooledOOFAUC":float(roc_auc_score(y[tr],pp.Probability)),"Threshold":f["threshold"]})
        coef=f["model"].coef_[0] if f["model"] is not None else []
        features.extend({"Analysis":analysis,"Feature":name,"Modality":data["metadata"][j]["modality"],
            "Block":data["metadata"][j]["block"],"LRStandardizedCoefficient":float(beta)} for name,j,beta in zip(f["names"],f["indices"],coef))
        trainpred.extend({"Analysis":analysis,"AnalysisID":cohort.AnalysisID.iloc[i],"Outcome":int(y[i]),"Probability":float(p)} for i,p in zip(tr,f["train_probability"]))
    final={"models":models,"train":tr,"metadata":data["metadata"],"feature_protocol":feature.PROTOCOL,"formal_model_use":False}
    joblib.dump(final,OUT/"models/five_multiseq_models.joblib",compress=3)
    for name,obj in [("nested_predictions.csv",pred),("outer_fold_metrics.csv",folds),
        ("inner_candidate_scores.csv",pd.DataFrame(grids)),("final_training_grid.csv",pd.DataFrame(grid)),
        ("final_training_choices.csv",pd.DataFrame(choices)),("selected_features.csv",pd.DataFrame(features)),
        ("train_fit_predictions.csv",pd.DataFrame(trainpred))]:obj.to_csv(OUT/name,index=False,encoding="utf-8-sig")
    ranks,_,audit = ranking(data,tr,y)
    pd.DataFrame(audit).to_csv(OUT/"final_feature_gate.csv",index=False,encoding="utf-8-sig")
    gate=pd.DataFrame(audit);gate["Modality"]=[m["modality"] for m in data["metadata"]]
    gate.groupby("Modality").agg(Candidates=("Feature","size"),Passed=("Eligible","sum")).to_csv(OUT/"feature_gate_by_sequence.csv",encoding="utf-8-sig")
    recommend=max(choices,key=lambda r:(r["OuterMeanAUC"]-.10*r["OuterSD"],-r["FeatureCount"]))
    save(OUT/"training_only_recommendation.json",{"analysis":recommend["Analysis"],
        "criterion":"outer mean-.10*SD; exploratory internal training evidence","test_used":False,"formal_model_use":False})
    save(OUT/"training_frozen.json",{"passed":True,"model_sha256":digest(OUT/"models/five_multiseq_models.joblib"),
        "protocol_sha256":digest(OUT/"protocol.json"),"feature_sha256":digest(OUT/"training_multiseq_features.joblib"),
        "training_csv_hashes":{p.name:digest(p) for p in OUT.glob("*.csv")},"test_scored":False,
        "protected_count":check_sources(),"elapsed_seconds":time.time()-started})
    print(pd.DataFrame(choices).to_string(index=False),flush=True)
    print("Five sequence-category finalists frozen; test not scored in this experiment",flush=True)


def predict(f,data):
    x=data["base"][:,f["indices"]]
    if f["model"] is None:return np.full(len(x),f["prevalence"])
    x=(np.clip(np.where(np.isfinite(x),x,f["median"]),f["low"],f["high"])-f["mean"])/f["scale"]
    return f["model"].predict_proba(x)[:,1]


def evaluate():
    if (OUT/"test_evaluation_started.json").exists():raise FileExistsError("Do not repeat joint test scoring")
    check_sources();frozen=json.loads((OUT/"training_frozen.json").read_text(encoding="utf-8"))
    assert digest(OUT/"models/five_multiseq_models.joblib")==frozen["model_sha256"]
    for n,h in frozen["training_csv_hashes"].items():assert digest(OUT/n)==h,n
    cohort,manifest,_,tr,te=common.old.previous.load_inputs()
    profiles=common.old.retained.load_profiles();cache=joblib.load(CACHE/"test_regional_features.joblib")
    data=feature.extract(profiles,te,cohort.AnalysisID.to_numpy(),cache);del profiles
    joblib.dump(data,OUT/"test_multiseq_features.joblib",compress=3)
    for key,name in [("IVIM_extraction_audit","test_IVIM_extraction_audit.csv"),
                     ("IVIM_quality_audit","test_IVIM_quality_audit.csv"),
                     ("IVIM_partition_audit","test_IVIM_partition_audit.csv")]:pd.DataFrame(data[key]).to_csv(OUT/name,index=False,encoding="utf-8-sig")
    final=joblib.load(OUT/"models/five_multiseq_models.joblib");assert final["metadata"]==data["metadata"]
    save(OUT/"test_evaluation_started.json",{"joint_count":1,"model_sha256":frozen["model_sha256"],"test_ids":manifest["test_ids"]})
    probabilities={a:predict(f,data) for a,f in final["models"].items()};y=cohort.Outcome.to_numpy(int)[te]
    draws=common.old.retained.bootstrap(y,probabilities);rows,metrics,pairs=[],[],[]
    for analysis,p in probabilities.items():
        threshold=final["models"][analysis]["threshold"];lo,hi=np.quantile(draws[analysis],[.025,.975])
        metrics.append({"Analysis":analysis,"Model":LABELS[analysis],**common.old.metrics(y,p,threshold),
                        "AUC_CI95Low":float(lo),"AUC_CI95High":float(hi)})
        rows.extend({"Analysis":analysis,"AnalysisID":cohort.AnalysisID.iloc[i],"Outcome":int(yy),
                     "Probability":float(pp),"Threshold":threshold,"PredictedPositive":int(pp>=threshold)} for i,yy,pp in zip(te,y,p))
    for a,b in [("FusionFree","ADC"),("FusionQuota","ADC"),("FusionQuota","FusionFree"),("IVIM","ADC")]:
        lo,hi=np.quantile(draws[a]-draws[b],[.025,.975]);pairs.append({"Comparison":a+" minus "+b,
            "DeltaAUC":float(roc_auc_score(y,probabilities[a])-roc_auc_score(y,probabilities[b])),"CI95Low":float(lo),"CI95High":float(hi)})
    for name,obj in [("test_predictions.csv",pd.DataFrame(rows)),("test_metrics.csv",pd.DataFrame(metrics)),
                     ("paired_auc_comparisons.csv",pd.DataFrame(pairs))]:obj.to_csv(OUT/name,index=False,encoding="utf-8-sig")
    save(OUT/"evaluation_complete.json",{"passed":True,"joint_count":1,"test_n":36,"prediction_rows":len(rows),
        "protected_count":check_sources(),"formal_model_use":False})
    print(pd.DataFrame(metrics).to_string(index=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--phase",choices=["train","evaluate"],required=True)
    args=parser.parse_args()
    with threadpool_limits(limits=1):train() if args.phase=="train" else evaluate()
