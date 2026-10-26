"""Fold-isolation, bootstrap-rank and original-pipeline replay validation."""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
import run_original_pls_direction_20261025 as run


def independent_direction(x,target,a):
    p,n=int(target.sum()),int((target==0).sum());seed=2026102501+sum((j+1)*(int(i)+1) for j,i in enumerate(a))%1000000
    rng=np.random.default_rng(seed);positive=rng.multinomial(p,np.full(p,1/p),40);negative=rng.multinomial(n,np.full(n,1/n),40)
    result=[]
    for j in range(min(3,x.shape[1])):
        effect=roc_auc_score(target,x[:,j])-.5;direction=0. if abs(effect)<=1e-12 else np.sign(effect);count=0
        for cp,cn in zip(positive,negative):
            v=np.r_[np.repeat(x[target==1,j],cp),np.repeat(x[target==0,j],cn)]
            value=roc_auc_score(np.r_[np.ones(p,int),np.zeros(n,int)],v)-.5
            count+=bool(value*direction>1e-12)
        result.append(count/40.)
    return np.asarray(result)


def main():
    run.frozen();assert not (run.OUT/'test_started.json').exists()
    cohort,y,tr,te=run.base.old.inputs();tables=joblib.load(run.base.OUT/'training_features.joblib');d=run.base.route_tables(tables[1.],'ContourMedian')
    private=0;meta_refits=0;maxerr=0.;feature_models=0
    for ph in run.PHASES:
        cp=joblib.load(run.OUT/'phases'/(ph+'.joblib'));frame=pd.DataFrame(cp['entries'])
        for rep,k,a,b in run.base.old.previous.parent.folds('Random120',ph):
            for t in run.THRESHOLDS:
                ctx=run.context(ph,t,rep,k);np.testing.assert_array_equal(ctx['train'],a);np.testing.assert_array_equal(ctx['validation'],b)
                assert set(a).isdisjoint(b) and not set(a)&set(te)
                for row,(rr,kk,aa,bb) in zip(ctx['folds'],run.base.old.deep_folds(ph,rep,k,a,y)):
                    np.testing.assert_array_equal(row['train'],aa);np.testing.assert_array_equal(row['validation'],bb)
                    assert set(aa).isdisjoint(bb) and set(aa)|set(bb)==set(a) and not (set(aa)|set(bb))&(set(b)|set(te))
                f=run.fit(ctx,y,t);meta_refits+=1;row=frame[(frame.Repeat==rep)&(frame.Fold==k)&(frame.threshold==t)].iloc[0]
                np.testing.assert_allclose([roc_auc_score(y[a],f['train_probability']),roc_auc_score(y[b],f['evaluation_probability'])],[row.FitAUC,row.AUC],atol=1e-12)
                if t>0:private+=1
        a,b=run.context(ph,0.)['train'],run.context(ph,0.)['validation']
        fitted={t:run.fit(run.context(ph,t),y,t) for t in run.THRESHOLDS};chosen,guard=run.choose(cp['grid'],fitted,y,a)
        assert chosen==cp['selected'];assert roc_auc_score(y[a],cp['models'][chosen['threshold']]['train_probability'])>=guard['FitFloor']-1e-12
        for row in cp['grid']:
            z=frame[frame.threshold==row['threshold']];np.testing.assert_allclose([row['MeanInnerAUC'],row['InnerSD'],row['Objective']],[z.AUC.mean(),z.AUC.std(ddof=1),z.AUC.mean()-.1*z.AUC.std(ddof=1)],atol=1e-12)
        for t,f in cp['models'].items():
            basis=run.basis(ph,t);wrapped={'fit':f,'bases':basis};pa=run.base.formula(wrapped,tables,a)
            err=float(np.max(np.abs(pa-f['train_probability'])));maxerr=max(maxerr,err);assert err<1e-9
            if len(b):np.testing.assert_allclose(run.base.formula(wrapped,tables,b),f['evaluation_probability'],atol=1e-9,rtol=0)
            for bag in basis['bag']:
                assert set(bag['train']).isdisjoint(bag['validation']) and not set(bag['train'])&set(te)
                if t>0:
                    for m in bag['models']:
                        assert len(m['names'])==len(m['direction_agreement']) and (m['direction_agreement']>=t).all()
                        assert len(m['names'])<=m['before_direction_count'];feature_models+=1
        old=joblib.load(run.base.OUT/'models'/('R1_ContourMedian_'+ph+'.joblib'))['fit']
        np.testing.assert_array_equal(cp['models'][0.]['train_probability'],old['train_probability'])
    # Independent integer bootstrap replay onselectedtraining-foldfeatures.
    checked=0
    for key in d:
        p=run.prepared(tables[1.],d,tr,key,y);expected=independent_direction(p['xa'],y[tr],tr)
        np.testing.assert_allclose(expected,p['agreement'][:len(expected)],atol=1e-12,rtol=0);checked+=len(expected)
    # Rebuild both new root feature pipelines' first privatecontext perphase.
    replay=0.;contexts=0
    for ph in run.PHASES:
        a,b=run.base.old.previous.parent.folds('Random120',ph)[0][2:]
        outputs=run.build_pair(ph+'_audit_only',tables,y,a,b,run.base.old.deep_folds(ph,1,1,a,y),replay=True)
        for t,ctx in outputs.items():
            stored=run.context(ph,t,1,1)
            for field in ('oof','bag_train','bag_validation'):
                np.testing.assert_allclose(ctx[field],stored[field],atol=1e-10,rtol=0);replay=max(replay,float(np.max(np.abs(ctx[field]-stored[field]))))
            contexts+=1
    # Heldoutfeature/labelmutationdoesnotchange learnedfeature setsorPLS coefficients.
    a,b=run.base.old.previous.parent.folds('Random120')[0];changed={}
    for key,value in tables[1.].items():
        changed[key]={**value,**{kind:value[kind].copy() for kind in ('base','erode','dilate')}};rr=run.base.old.latent.rows(value,b)
        for kind in ('base','erode','dilate'):changed[key][kind][rr]=1e9
    yy=y.copy();yy[b]=1-yy[b];original_models=run.fit_pairs(tables[1.],d,a,b,y);run.PREPARED.clear()
    modified_models=run.fit_pairs(changed,run.base.route_tables(changed,'ContourMedian'),a,b,yy)
    for t in run.THRESHOLDS[1:]:
        for f,g in zip(original_models[t],modified_models[t]):
            assert f['indices']==g['indices'];np.testing.assert_array_equal(f['train_probability'],g['train_probability'])
    final=joblib.load(run.OUT/'models/final_primary.joblib');assert roc_auc_score(y[tr],final['train_probability'])>=.7680311890838207-1e-12
    run.save(run.OUT/'training_verification.json',{'passed':True,'new_private_contexts':private,'meta_refits':meta_refits,'base_models_features_audited':feature_models,
        'max_formula_error':maxerr,'independent_integer_bootstrap_features':checked,'private_context_replays':contexts,'max_replay_error':replay,
        'heldout_mutation_models':24,'original_baseline_exact':True,'actual_training_fit_floor_held':True,'all5folds_kept':True,'test_calls':0})
    print('Directionselectionverified;bootstrap/privateOOF/exactbaseline/heldoutisolation',flush=True)


if __name__=='__main__':
    with threadpool_limits(1):main()
