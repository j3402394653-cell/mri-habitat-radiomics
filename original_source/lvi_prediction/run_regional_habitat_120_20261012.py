"""One frozen pure-habitat improvement experiment on the existing random120 split."""
import argparse
from pathlib import Path
import json
import time
import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score,roc_curve
from threadpoolctl import threadpool_limits
import lvi_paths
import regional_habitat_features_20261012 as feature
import run_random120_texture_bootstrap_20261012 as old

ROOT = old.ROOT
OUT = ROOT/"lvi_prediction/results/lvi_regional_habitat_120_20261012"
MODS = {"ADC":("ADC",),"T2ADC":("T2","ADC")}
save,digest = old.save,old.digest


def options():
    rows = [{"family":fam,"modality":m,"top":n,"algorithm":algo,"C":c}
            for fam in ["Descriptor","Texture","Combined"] for m in MODS for n in [3,5]
            for algo,c in [("LR",.03),("LR",.3),("LR",1.),("BoostStumps",0.)]]
    for i,r in enumerate(rows):r["candidate_id"] = i
    return rows


def config(row):return {k:row[k] for k in options()[0]}


def choose(rows):
    return max(rows,key=lambda r:(r["Objective"],-r["MeanFeatureCount"],r["algorithm"]=="LR",-r["C"],-r["candidate_id"]))


def freeze():
    if OUT.exists():raise FileExistsError("Regional habitat protocol already exists")
    old.check_sources()
    cohort,manifest,y,tr,te = old.previous.load_inputs()
    previous = ROOT/"lvi_prediction/results/lvi_random120_five_models_20261012"
    assert json.loads((previous/"test_verification.json").read_text())["passed"]
    protection = dict(json.loads((previous/"protocol.json").read_text(encoding="utf-8"))["protected_hashes"])
    protection.update({str(p.relative_to(ROOT)):digest(p) for p in previous.rglob("*") if p.is_file() and "tmp" not in p.parts})
    for p in [Path(__file__).resolve(),ROOT/"lvi_prediction/regional_habitat_features_20261012.py",
              ROOT/"lvi_prediction/quantile_habitat_features.py",old.retained.PROFILE,old.retained.POOL/"cohort_138.csv"]:
        protection[str(p.relative_to(ROOT))] = digest(p)
    OUT.mkdir(parents=True);(OUT/"models").mkdir()
    save(OUT/"protocol.json",{"date":"2026-10-12","authorization":"对生境模型进行改善",
        "same_random120_and_84_36_split":True,"sample_seed":20261010,"split_seed":42,
        "scope":"pure regional habitat model; no whole-tumor or clinical predictors; original adopted results unchanged",
        "feature_protocol":feature.PROTOCOL,"candidate_count":len(options()),"candidates":options(),
        "selection":"current training only; ICC(A,1)>=.8, complete>=.9, 5 stratified bootstrap signed effects, sign consistency>=.8, abs Spearman<.8",
        "partition_fraction_features":"tertile TumorFraction retained in audit but excluded as near-fixed by construction",
        "nested":"outer5 seed42; inner3 seed62201..62205; final training5 seed62212",
        "objective":"mean inner AUC-.10*SD-.30*max(mean fit-mean validation-.10,0)-.03*abs(mean fit-.75)",
        "preprocessing":"training median fill; training 1/99% winsorize; training standardize",
        "models":"L2 LR C=.03/.3/1; GradientBoosting40 depth1, lr.03,min_leaf10",
        "test":"one overall training-selected finalist; evaluate once after freeze; never choose a test winner",
        "prior_test_exposure":True,"clinical_timing_and_pathology_definition":"unknown",
        "automatic_adoption":False,"excluded_transfer_and_discarded_nested_models_used":False,
        "sources":{"PyRadiomics":"https://pyradiomics.readthedocs.io/en/v3.0/customization.html"},
        "protected_hashes":protection})
    save(OUT/"split_manifest.json",manifest)
    return cohort,manifest,y,tr,te


def check_sources():
    p = json.loads((OUT/"protocol.json").read_text(encoding="utf-8"))
    for name,h in p["protected_hashes"].items():assert digest(ROOT/name)==h,name
    return len(p["protected_hashes"])


def rows(data,indices):
    lookup = {int(i):j for j,i in enumerate(data["indices"])}
    return np.asarray([lookup[int(i)] for i in indices],int)


def signed_effect(x,y):
    npos,nneg = int(y.sum()),int((y==0).sum())
    assert npos and nneg
    ranks = rankdata(x,axis=0)
    return (ranks[y==1].sum(0)-npos*(npos+1)/2)/(npos*nneg)-.5


def ranking(data,train,y):
    rr = rows(data,train)
    measured = np.stack([data[k][rr] for k in ["base","erode","dilate"]],axis=1)
    nfeatures = measured.shape[2]
    complete = np.isfinite(measured).all(1).mean(0)
    reference = data["base"][rr]
    median = np.nanmedian(np.where(np.isfinite(reference),reference,np.nan),axis=0)
    median = np.where(np.isfinite(median),median,0.)
    reference = np.where(np.isfinite(reference),reference,median)
    icc = np.full(nfeatures,np.nan)
    for j in range(nfeatures):
        valid = np.isfinite(measured[:,:,j]).all(1)
        if valid.sum()>=3 and complete[j]>=.9 and reference[:,j].std()>1e-12:
            icc[j] = old.engine.icc_absolute(measured[valid,:,j])
    target = y[train]; rng = np.random.default_rng(64212+int(np.sum(train)))
    pos,neg = np.flatnonzero(target==1),np.flatnonzero(target==0)
    boot = []
    for _ in range(5):
        ix = np.r_[rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)]
        boot.append(signed_effect(reference[ix],target[ix]))
    boot = np.asarray(boot); effect = boot.mean(0)
    stability = np.maximum((boot>0).mean(0),(boot<0).mean(0))
    eligible = (complete>=.9)&np.isfinite(icc)&(icc>=.8)&(stability>=.8)&(reference.std(0)>1e-12)
    for j,m in enumerate(data["metadata"]):
        if m["name"].endswith("_TumorFraction"):eligible[j]=False
    scores = np.abs(effect)*np.maximum(np.nan_to_num(icc,nan=0.),0.)**2*stability
    ordered = sorted(np.flatnonzero(eligible),key=lambda j:(-scores[j],-icc[j],data["metadata"][j]["name"]))
    ranked_values = rankdata(reference,axis=0)
    audit = [{"Feature":m["name"],"ICC":float(icc[j]) if np.isfinite(icc[j]) else None,
              "CompleteFraction":float(complete[j]),"BootstrapMeanSignedEffect":float(effect[j]),
              "BootstrapSignConsistency":float(stability[j]),"Eligible":bool(eligible[j]),"RankingScore":float(scores[j])}
             for j,m in enumerate(data["metadata"])]
    return ordered,ranked_values,audit


def prepare(data,train,evaluation,c,y,ranked=None):
    ordered,ranks,audit = ranking(data,train,y) if ranked is None else ranked
    selected = []
    for j in ordered:
        m = data["metadata"][j]
        if m["modality"] not in MODS[c["modality"]]:continue
        if c["family"]!="Combined" and m["block"]!=c["family"]:continue
        if all(abs(float(np.corrcoef(ranks[:,j],ranks[:,k])[0,1]))<.8 for k in selected):selected.append(j)
        if len(selected)==c["top"]:break
    xa,xb = data["base"][rows(data,train)][:,selected],data["base"][rows(data,evaluation)][:,selected]
    if selected:
        median = np.nanmedian(np.where(np.isfinite(xa),xa,np.nan),axis=0)
        filled = np.where(np.isfinite(xa),xa,median);low,high = np.quantile(filled,[.01,.99],axis=0)
        filled = np.clip(filled,low,high);mean,scale = filled.mean(0),filled.std(0);scale = np.where(scale>1e-12,scale,1.)
        xa = (filled-mean)/scale
        xb = (np.clip(np.where(np.isfinite(xb),xb,median),low,high)-mean)/scale
    else:median,low,high,mean,scale = [np.array([]) for _ in range(5)]
    return {"indices":selected,"names":[data["metadata"][j]["name"] for j in selected],"audit":audit,
            "median":median,"low":low,"high":high,"mean":mean,"scale":scale,"xa":xa,"xb":xb}


def fitted(prepared,target,c):
    prevalence = float(target.mean())
    if prepared["indices"]:
        model = (LogisticRegression(C=c["C"],penalty="l2",solver="liblinear",max_iter=3000,random_state=64212)
                 if c["algorithm"]=="LR" else GradientBoostingClassifier(n_estimators=40,learning_rate=.03,
                    max_depth=1,min_samples_leaf=10,random_state=64212))
        model.fit(prepared["xa"],target)
        pa = model.predict_proba(prepared["xa"])[:,1]
        pb = model.predict_proba(prepared["xb"])[:,1] if len(prepared["xb"]) else np.array([])
    else:model,pa,pb = None,np.full(len(target),prevalence),np.full(len(prepared["xb"]),prevalence)
    return {"config":c,"model":model,"prevalence":prevalence,
        **{k:prepared[k] for k in ["indices","names","median","low","high","mean","scale","audit"]},
        "train_probability":pa,"evaluation_probability":pb}


def fit(data,train,evaluation,y,c):return fitted(prepare(data,train,evaluation,c,y),y[train],c)


def tune(data,folds,y):
    values = {c["candidate_id"]:[] for c in options()};cv = {c["candidate_id"]:[] for c in options()}
    for a,b in folds:
        rank = ranking(data,a,y);cache = {}
        for c in options():
            key = (c["family"],c["modality"],c["top"])
            if key not in cache:cache[key]=prepare(data,a,b,c,y,rank)
            f = fitted(cache[key],y[a],c)
            values[c["candidate_id"]].append((roc_auc_score(y[b],f["evaluation_probability"]),
                roc_auc_score(y[a],f["train_probability"]),len(f["indices"])))
            cv[c["candidate_id"]].append((b,f["evaluation_probability"]))
    records = []
    for c in options():
        v = np.asarray(values[c["candidate_id"]]);mean,sd,train,n = float(v[:,0].mean()),float(v[:,0].std(ddof=1)),float(v[:,1].mean()),float(v[:,2].mean())
        records.append({**c,"MeanValidationAUC":mean,"SD":sd,"MeanFitAUC":train,"MeanFeatureCount":n,
                        "Objective":mean-.10*sd-.30*max(train-mean-.10,0)-.03*abs(train-.75)})
    return records,cv


def threshold(y,p):
    fpr,tpr,cut = roc_curve(y,p,drop_intermediate=False)
    valid = np.flatnonzero(np.isfinite(cut));j = max(valid,key=lambda i:(float(tpr[i]-fpr[i]),float(cut[i])))
    return float(cut[j])


def train():
    started = time.time();cohort,manifest,y,tr,te = freeze()
    profiles = old.retained.load_profiles()
    data = feature.extract(profiles,tr,cohort.AnalysisID.to_numpy());del profiles
    assert len(data["metadata"])==636 and set(data["indices"])==set(tr)
    joblib.dump(data,OUT/"training_regional_features.joblib",compress=3)
    pd.DataFrame(data["extraction_audit"]).to_csv(OUT/"training_extraction_audit.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(data["partition_audit"]).to_csv(OUT/"training_partition_audit.csv",index=False,encoding="utf-8-sig")
    predictions,metrics,grids = [],[],[]
    for fold,(a,b) in enumerate(old.previous.folds(tr,y,42,5),1):
        grid,_ = tune(data,old.previous.folds(a,y,62200+fold,3),y);c = config(choose(grid))
        f = fit(data,a,b,y,c)
        joblib.dump({"grid":grid,"fitted":f,"train":a,"evaluation":b},OUT/f"models/outer_fold_{fold}.joblib",compress=3)
        grids.extend({"OuterFold":fold,**r} for r in grid)
        metrics.append({"Fold":fold,**c,"FeatureCount":len(f["indices"]),"AUC":float(roc_auc_score(y[b],f["evaluation_probability"]))})
        predictions.extend({"Fold":fold,"AnalysisID":cohort.AnalysisID.iloc[i],"Outcome":int(y[i]),"Probability":float(p)} for i,p in zip(b,f["evaluation_probability"]))
        print(f"Regional habitat nested fold {fold}/5 completed",flush=True)
    pred,folds = pd.DataFrame(predictions),pd.DataFrame(metrics)
    nested = {"OuterMeanAUC":float(folds.AUC.mean()),"OuterSD":float(folds.AUC.std(ddof=1)),
              "PooledOOFAUC":float(roc_auc_score(pred.Outcome,pred.Probability))}
    grid,cv = tune(data,old.previous.folds(tr,y,62212,5),y);chosen = choose(grid);c = config(chosen)
    f = fit(data,tr,[],y,c);oof = np.full(138,np.nan)
    for ix,p in cv[c["candidate_id"]]:oof[ix]=p
    assert np.isfinite(oof[tr]).all() and np.isnan(oof[te]).all()
    f["threshold"] = threshold(y[tr],oof[tr]);f["train"] = tr;f["feature_protocol"] = feature.PROTOCOL
    f["metadata"] = data["metadata"];f["formal_model_use"] = False
    joblib.dump(f,OUT/"models/final_pure_habitat.joblib",compress=3)
    choice = {**c,"FeatureCount":len(f["indices"]),"TrainFitAUC":float(roc_auc_score(y[tr],f["train_probability"])),
              "FinalSelectionCVMeanAUC":chosen["MeanValidationAUC"],"Threshold":f["threshold"],**nested}
    train_pred = pd.DataFrame({"AnalysisID":cohort.AnalysisID.iloc[tr],"Outcome":y[tr],"Probability":f["train_probability"]})
    coef = f["model"].coef_[0] if c["algorithm"]=="LR" and f["model"] is not None else [None]*len(f["names"])
    selected = pd.DataFrame([{"Feature":n,"LRStandardizedCoefficient":b} for n,b in zip(f["names"],coef)])
    for name,obj in [("nested_predictions.csv",pred),("outer_fold_metrics.csv",folds),
                     ("inner_candidate_scores.csv",pd.DataFrame(grids)),("final_training_grid.csv",pd.DataFrame(grid)),
                     ("final_training_choice.csv",pd.DataFrame([choice])),("train_fit_predictions.csv",train_pred),
                     ("selected_features.csv",selected),("final_feature_gate.csv",pd.DataFrame(f["audit"]))]:
        obj.to_csv(OUT/name,index=False,encoding="utf-8-sig")
    save(OUT/"training_frozen.json",{"passed":True,"model_sha256":digest(OUT/"models/final_pure_habitat.joblib"),
        "features_sha256":digest(OUT/"training_regional_features.joblib"),"protocol_sha256":digest(OUT/"protocol.json"),
        "training_output_hashes":{p.name:digest(p) for p in OUT.glob("*.csv")},"test_scored":False,
        "protected_count":check_sources(),"elapsed_seconds":time.time()-started})
    print(pd.DataFrame([choice]).to_string(index=False),flush=True)
    print("One improved pure-habitat finalist frozen; test not scored in this experiment",flush=True)


def predict(f,data):
    assert f["metadata"]==data["metadata"]
    x = data["base"][:,f["indices"]]
    if f["model"] is None:return np.full(len(x),f["prevalence"])
    x = (np.clip(np.where(np.isfinite(x),x,f["median"]),f["low"],f["high"])-f["mean"])/f["scale"]
    return f["model"].predict_proba(x)[:,1]


def evaluate():
    if (OUT/"test_evaluation_started.json").exists():raise FileExistsError("Do not repeat evaluation")
    check_sources();frozen = json.loads((OUT/"training_frozen.json").read_text(encoding="utf-8"))
    assert digest(OUT/"models/final_pure_habitat.joblib")==frozen["model_sha256"]
    for n,h in frozen["training_output_hashes"].items():assert digest(OUT/n)==h,n
    cohort,manifest,_,tr,te = old.previous.load_inputs()
    profiles = old.retained.load_profiles()
    data = feature.extract(profiles,te,cohort.AnalysisID.to_numpy());del profiles
    joblib.dump(data,OUT/"test_regional_features.joblib",compress=3)
    pd.DataFrame(data["extraction_audit"]).to_csv(OUT/"test_extraction_audit.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(data["partition_audit"]).to_csv(OUT/"test_partition_audit.csv",index=False,encoding="utf-8-sig")
    f = joblib.load(OUT/"models/final_pure_habitat.joblib")
    save(OUT/"test_evaluation_started.json",{"count":1,"model_sha256":frozen["model_sha256"],"test_ids":manifest["test_ids"]})
    probability = predict(f,data);target = cohort.Outcome.to_numpy(int)[te]
    baseline = pd.read_csv(old.OUT/"test_scores.csv",float_precision="round_trip")
    baseline = baseline[baseline.Analysis=="Habitat"].set_index("AnalysisID").loc[cohort.AnalysisID.iloc[te],"Score"].to_numpy()
    ps = {"ImprovedHabitat":probability,"OriginalHabitat":baseline};draws = old.retained.bootstrap(target,ps)
    metrics = old.metrics(target,probability,f["threshold"])
    lo,hi = np.quantile(draws["ImprovedHabitat"],[.025,.975]);metrics.update({"AUC_CI95Low":float(lo),"AUC_CI95High":float(hi)})
    pd.DataFrame({"AnalysisID":cohort.AnalysisID.iloc[te],"Outcome":target,"Probability":probability,
        "Threshold":f["threshold"],"PredictedPositive":(probability>=f["threshold"]).astype(int)}).to_csv(OUT/"test_predictions.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame([metrics]).to_csv(OUT/"test_metrics.csv",index=False,encoding="utf-8-sig")
    lo,hi = np.quantile(draws["ImprovedHabitat"]-draws["OriginalHabitat"],[.025,.975])
    pd.DataFrame([{"Comparison":"Improved minus original pure habitat","DeltaAUC":float(metrics["AUC"]-roc_auc_score(target,baseline)),
        "CI95Low":float(lo),"CI95High":float(hi)}]).to_csv(OUT/"paired_auc_comparison.csv",index=False,encoding="utf-8-sig")
    save(OUT/"evaluation_complete.json",{"passed":True,"count":1,"test_n":36,"formal_model_use":False,
        "protected_count":check_sources(),"model_sha256":digest(OUT/"models/final_pure_habitat.joblib")})
    print(pd.DataFrame([metrics]).to_string(index=False),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser();parser.add_argument("--phase",choices=["train","evaluate"],required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):train() if args.phase=="train" else evaluate()
