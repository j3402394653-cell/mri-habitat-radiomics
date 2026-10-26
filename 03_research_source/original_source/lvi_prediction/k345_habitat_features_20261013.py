"""Controlled extension of existing patient-local habitats from K3 to K4/K5."""
from itertools import combinations
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import uniform_filter,distance_transform_edt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score,silhouette_score
from radiomics import featureextractor
import radiomics
import normalization_habitat_features_20261012 as prior
from quantile_habitat_features import original_values

NORMALIZATIONS=('PercentileZ','RobustZ')
KS=(3,4,5)
MODALITIES=('T2','ADC','IVIM')
KINDS=(('base','roi'),('erode','erode'),('dilate','dilate'))
PROTOCOL={**prior.PROTOCOL,'methods':{n:prior.PROTOCOL['methods'][n] for n in NORMALIZATIONS},
    'K_values':KS,'partition':'unchanged patient-local intensity/local3mm mean/SD KMeans; only requested K changes',
    'small_habitats':'minimum12voxels unchanged; empty/small habitat features missing, gated and train-median filled; never drop a patient based on outcome',
    'spatial_extension':'same per-habitat descriptors and all pairwise contacts/invariant centroid distances generalized to K',
    'feature_count_formula':'T2/ADC K^2+102*K each; IVIM K^2+201*K; total3*K^2+405*K',
    'K3_comparator':'exact verified prior feature caches; generic K3 extraction and model replay checked',
    'quality_audit':'habitat size, missing texture, sample500 silhouette on base contour, base-vs-perturbed partition ARI on common valid voxels',
    'silhouette_scope':'patient-local cluster separation only; not a clinical prediction accuracy metric',
    'K_selection':'jointly with features and LR parameters inside each inner training CV; fixed-K comparators also reported'}

def pairs(k):return list(combinations(range(1,k+1),2))

def design(values,mask):
    values=np.asarray(values,float)
    if values.ndim==mask.ndim:values=values[...,None]
    support=uniform_filter(mask.astype(float),size=3,mode='constant');columns=[]
    for c in range(values.shape[-1]):
        a=values[...,c];mu,sd=float(a[mask].mean()),float(a[mask].std())
        v=np.where(mask,(a-mu)/max(sd,1e-12),0.)
        mean=uniform_filter(v,size=3,mode='constant')/np.maximum(support,1e-12)
        second=uniform_filter(v*v,size=3,mode='constant')/np.maximum(support,1e-12)
        variance=second-(mean**2 if values.shape[-1]==1 else mean*mean)
        columns.extend([v[mask],mean[mask],np.sqrt(np.maximum(variance,0.))[mask]])
    x=np.column_stack(columns);rng=np.random.default_rng(64212)
    ix=rng.choice(len(x),min(4000,len(x)),replace=False)
    return StandardScaler().fit(x[ix]).transform(x),ix

def partition(values,mask,k,mod):
    mask=np.asarray(mask,bool);labels=np.zeros(mask.shape,np.uint8)
    if mask.sum()<12:return labels,{'ActualK':0,'Silhouette':None,'SampleCount':0}
    normalized,ix=design(values,mask)
    if k==3:
        shaped=np.asarray(values);shaped=shaped[...,None] if shaped.ndim==mask.ndim else shaped
        labels=prior.partition(shaped,mask,mod)
    elif np.unique(normalized[ix],axis=0).shape[0]<k:
        labels[mask]=1
    else:
        model=KMeans(n_clusters=k,n_init=5,random_state=64212).fit(normalized[ix])
        order=np.lexsort((model.cluster_centers_[:,3],model.cluster_centers_[:,0])) if mod=='IVIM' else np.argsort(model.cluster_centers_[:,0],kind='stable')
        remap=np.empty(k,np.uint8);remap[order]=np.arange(1,k+1);labels[mask]=remap[model.predict(normalized)]
    assert np.array_equal(labels>0,mask)
    rng=np.random.default_rng(64213);sample=rng.choice(len(normalized),min(500,len(normalized)),replace=False)
    current=labels[mask][sample];silhouette=None
    if 1<len(np.unique(current))<len(current):silhouette=float(silhouette_score(normalized[sample],current))
    return labels,{'ActualK':int(len(np.unique(labels[mask]))),'Silhouette':silhouette,'SampleCount':len(sample)}

def contacts(labels,mask,k):
    counts={p:0 for p in pairs(k)};total=0
    for axis in range(3):
        left,right=[slice(None)]*3,[slice(None)]*3;left[axis],right[axis]=slice(None,-1),slice(1,None)
        left,right=tuple(left),tuple(right);valid=mask[left]&mask[right];a,b=labels[left][valid],labels[right][valid];total+=len(a)
        for h,j in counts:counts[(h,j)]+=int(((a==h)&(b==j)|(a==j)&(b==h)).sum())
    return [counts[p]/total if total else np.nan for p in pairs(k)]

def descriptors(values,labels,mask,spacing,k):
    depth=distance_transform_edt(mask,sampling=spacing);band=mask&(depth<=2.)
    maxdepth=max(float(depth.max()),1.);numbers=[];names=[]
    for h in range(1,k+1):
        region=mask&(labels==h);a=values[region];n=len(a)
        if n>=12:
            qs=np.quantile(a,[.1,.25,.5,.75,.9])
            v=[float(a.mean()),float(a.std()),float(qs[0]),float(qs[2]),float(qs[4]),float(qs[3]-qs[1]),
               n/max(int(mask.sum()),1),int((region&band).sum())/n,float(np.mean(depth[region]))/maxdepth]
        else:v=[np.nan]*9
        numbers.extend(v);names.extend('H%d_%s'%(h,s) for s in ['Mean','SD','P10','Median','P90','IQR','TumorFraction','BoundaryEnrichment','NormalizedDepth'])
    numbers.extend(contacts(labels,mask,k));names.extend('ContactH%dH%d'%p for p in pairs(k))
    return numbers,names

def invariant_centroids(labels,mask,spacing,k):
    coords=np.argwhere(mask)*np.asarray(spacing);center=coords.mean(0) if len(coords) else np.zeros(3)
    radius=max(float(np.sqrt(np.mean(np.sum((coords-center)**2,axis=1)))),1e-12) if len(coords) else 1.
    centers=[]
    for h in range(1,k+1):
        points=np.argwhere(mask&(labels==h))*np.asarray(spacing);centers.append(points.mean(0) if len(points)>=12 else np.full(3,np.nan))
    numbers=[float(np.linalg.norm(c-center)/radius) for c in centers]
    numbers.extend(float(np.linalg.norm(centers[a-1]-centers[b-1])/radius) for a,b in pairs(k))
    return numbers,['H%d_CentroidOffsetInvariant'%h for h in range(1,k+1)]+['H%dH%d_CentroidDistanceInvariant'%p for p in pairs(k)]

def layout(mod,texture_names,k):
    channels=['D','f'] if mod=='IVIM' else [mod];metadata=[]
    _,descriptor_names=descriptors(np.zeros((2,2,2)),np.ones((2,2,2),np.uint8),np.ones((2,2,2),bool),(1.,1.,1.),k)
    for ci,channel in enumerate(channels):
        for h in range(1,k+1):
            metadata.extend({'name':'%s__Regional_H%d_%s%s'%(mod,h,channel+'_' if mod=='IVIM' else '',n),
                'modality':mod,'block':'Texture','habitat':h} for n in texture_names)
        keep=[j for j,n in enumerate(descriptor_names) if ci==0 or any(n.endswith('_'+s) for s in ['Mean','SD','P10','Median','P90','IQR'])]
        prefix=mod+'__Regional_'+(channel+'_Raw_' if mod=='IVIM' else 'Z_' if mod=='T2' else 'Raw_')
        metadata.extend({'name':prefix+descriptor_names[j],'modality':mod,'block':'Descriptor','habitat':0} for j in keep)
    _,names=invariant_centroids(np.ones((2,2,2),np.uint8),np.ones((2,2,2),bool),(1.,1.,1.),k)
    metadata.extend({'name':mod+'__Regional_'+n,'modality':mod,'block':'Descriptor','habitat':0} for n in names)
    return metadata

def extract_one(task):
    index,aid,profiles,extract_ks=task
    classes=radiomics.getFeatureClasses()
    names=sorted('original_'+g+'_'+n for g in prior.regional.CLASSES for n,deprecated in classes[g].getFeatureNames().items() if not deprecated)
    assert len(names)==93
    extractor=featureextractor.RadiomicsFeatureExtractor(binWidth=.25,normalize=False,label=1,distances=[1],additionalInfo=False,minimumROISize=12)
    extractor.disableAllFeatures()
    for group in prior.regional.CLASSES:extractor.enableFeatureClassByName(group)
    rows={n:{k:{kind:{} for kind,_ in KINDS} for k in extract_ks} for n in NORMALIZATIONS}
    metadata={k:[] for k in extract_ks};partaudit=[];textures=[];normaudit=[];contour=[]
    for mod in MODALITIES:
        p=profiles[mod];raw=original_values(p,mod);channels=['D','f'] if mod=='IVIM' else [mod]
        normalized={};images={}
        for n in NORMALIZATIONS:
            arrays=[]
            for c,channel in enumerate(channels):
                value,audit=prior.normalize(raw[...,c],p['roi'],n,p['z'][...,c]);arrays.append(value)
                normaudit.append({'AnalysisID':aid,'Modality':mod,'Channel':channel,**audit})
            normalized[n]=np.stack(arrays,axis=-1);images[n]=[]
            for c in range(len(channels)):
                image=sitk.GetImageFromArray(normalized[n][...,c]);image.SetSpacing(tuple(p['spacing'])[::-1]);images[n].append(image)
        for k in KS:
            meta=layout(mod,names,k)
            if k in extract_ks:metadata[k].extend(meta)
            labelsets={n:{} for n in NORMALIZATIONS}
            for kind,maskkey in KINDS:
                mask=np.asarray(p[maskkey],bool)&p['valid']
                for n in NORMALIZATIONS:
                    values=(p['z'] if mod=='IVIM' else raw) if n=='RobustZ' else normalized[n]
                    labels,quality=partition(values,mask,k,mod);labelsets[n][kind]=(labels,mask)
                    sizes=[int((labels==h).sum()) for h in range(1,k+1)]
                    partaudit.append({'AnalysisID':aid,'Modality':mod,'Normalization':n,'K':k,'Variant':kind,'ValidVoxels':int(mask.sum()),
                        **quality,'MinimumHabitatVoxels':min(sizes),'MaximumHabitatFraction':max(sizes)/max(int(mask.sum()),1),
                        'SmallHabitatCount':sum(s<12 for s in sizes),**{'H'+str(h):s for h,s in enumerate(sizes,1)}})
                    if k not in extract_ks:continue
                    row=[]
                    for c,channel in enumerate(channels):
                        for h in range(1,k+1):
                            region=mask&(labels==h);error=None
                            try:
                                if region.sum()<12:raise ValueError('habitat fewer than12voxels')
                                roi=sitk.GetImageFromArray(region.astype(np.uint8));roi.CopyInformation(images[n][c])
                                measured=extractor.execute(images[n][c],roi);numbers=[float(measured[j]) for j in names]
                            except (ValueError,RuntimeError) as exc:numbers=[np.nan]*93;error=type(exc).__name__+': '+str(exc)
                            row.extend(numbers);textures.append({'AnalysisID':aid,'Modality':mod,'Channel':channel,'Normalization':n,'K':k,
                                'Variant':kind,'Habitat':h,'Voxels':int(region.sum()),'Error':error})
                        v,dnames=descriptors(normalized[n][...,c] if mod=='T2' else raw[...,c],labels,mask,p['spacing'],k)
                        row.extend(v[j] for j,s in enumerate(dnames) if c==0 or any(s.endswith('_'+q) for q in ['Mean','SD','P10','Median','P90','IQR']))
                    v,_=invariant_centroids(labels,mask,p['spacing'],k);row.extend(v)
                    assert len(row)==len(meta)==k*k+(201 if mod=='IVIM' else 102)*k
                    rows[n][k][kind].update({m['name']:float(v) for m,v in zip(meta,row)})
            for n in NORMALIZATIONS:
                base,bmask=labelsets[n]['base']
                for kind in ['erode','dilate']:
                    labels,mask=labelsets[n][kind];common=mask&bmask
                    contour.append({'AnalysisID':aid,'Modality':mod,'Normalization':n,'K':k,'Variant':kind,
                        'CommonVoxels':int(common.sum()),'ARI':float(adjusted_rand_score(base[common],labels[common])) if common.any() else None,
                        'LabelAgreement':float(np.mean(base[common]==labels[common])) if common.any() else None})
    return {'index':index,'values':rows,'metadata':metadata,'partition_audit':partaudit,'texture_audit':textures,
            'normalization_audit':normaudit,'contour_partition_audit':contour}
