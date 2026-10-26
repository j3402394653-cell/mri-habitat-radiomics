"""Equivalent batching/caching for the current habitat-only experiment.

No feature definitions or fitting rules change. Original single-sequence
stability selection is reused; fusion concatenates each sequence's first choice
exactly as the existing engine does. A global worker thread limit replaces
repeated Windows library introspection in thousands of nested calls.
"""
from contextlib import nullcontext
import numpy as np
from threadpoolctl import threadpool_limits
import fold_locked_lvi_engine as engine

ORIGINAL_STABLE=engine.stable_indices
ORIGINAL_TABLES=engine.feature_tables
ORIGINAL_STATS=engine._stats
LIMITER=None
CACHE={}


def stats(p,region,prefix):
    z=p['z'][region]
    values=[float(np.log1p(len(z)))];names=[prefix+'logVoxels']
    for c in range(p['z'].shape[-1]):
        a=z[:,c]
        if len(a)>=12:
            q=np.percentile(a,[10,90,75,25])
            values.extend((float(q[0]),float(np.median(a)),float(q[1]),float(q[2]-q[3])))
        else:values.extend([np.nan]*4)
        names.extend(prefix+n+str(c) for n in ('P10_','Median_','P90_','IQR_'))
    return values,names


def tables(profiles,fitted,k,indices):
    keys=('habitat','erode','dilate')
    rows={key:[] for key in keys};names={key:[] for key in keys}
    for idx in indices:
        row={key:[] for key in keys}
        for mod in engine.MODALITIES:
            p=profiles[int(idx)][mod];labels=engine.assign(p,fitted[mod])
            for key,mask in (('habitat',p['roi']),('erode',p['erode']),('dilate',p['dilate'])):
                v,n=engine._habitat_stats(p,labels,mask,mod,k)
                row[key].extend(v)
                if not rows[key]:names[key].extend(n)
        for key in keys:rows[key].append(row[key])
    return {key:np.asarray(v,float) for key,v in rows.items()},names


def stable(ta,names,modalities,y_train,threshold=.8):
    key=id(ta)
    if key not in CACHE:CACHE[key]=(ta,{})
    stored=CACHE[key][1]
    for mod in modalities:
        if mod not in stored:stored[mod]=ORIGINAL_STABLE(ta,names,(mod,),y_train,threshold)
    if len(modalities)==1:return stored[modalities[0]]
    selected=[];audit={}
    for mod in modalities:
        indices,a=stored[mod]
        selected.extend(indices[:1]);audit.update(a)
    return selected,audit


def install():
    global LIMITER
    LIMITER=threadpool_limits(limits=1)
    engine.threadpool_limits=lambda **kwargs:nullcontext()
    engine._stats=stats;engine.feature_tables=tables;engine.stable_indices=stable


def clear():CACHE.clear()
