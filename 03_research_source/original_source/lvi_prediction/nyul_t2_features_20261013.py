"""Training-landmark Nyul adaptation using the unchanged valid tumor reference."""
import hashlib
import numpy as np
import SimpleITK as sitk
import radiomics
from radiomics import featureextractor
from sklearn.metrics import adjusted_rand_score
import normalization_habitat_features_20261012 as prior

PERCENTILES=np.array([1,10,20,30,40,50,60,70,80,90,99],float)
ENDPOINTS=(-2.3263478740408408,2.3263478740408408)
PROTOCOL={
    'method':'Nyul piecewise-linear histogram landmark standardization, tumor-ROI adaptation',
    'reference':'unchanged base valid T2 tumor ROI, including only patient-local reference for each held-out image',
    'percentiles':PERCENTILES.tolist(),'fixed_target_endpoints':ENDPOINTS,
    'learning':'per training image linearly rescale P1/P99 to fixed endpoints; equally average its eleven landmarks over current training patients only',
    'transform':'map patient percentiles to frozen training target, linear between knots and linear extrapolation outside P1/P99',
    'ties':'average target knots corresponding to tied source landmarks; constant reference is rejected explicitly',
    'input':'stored T2 Z-score; positive affine-invariance makes percentile mapping identical to recovered raw T2',
    'fold_fitting':'learn target independently in every inner training fold and outer training fold; final target learned on84 only',
    'feature_materialization':'recompute T2 habitats and all315 features independently for every distinct training target',
    'no_post_Nyul_image_zscore':True,'texture_bin_width':.25,
    'scale_reason':'fixed Z-like endpoints keep the existing .25 texture binWidth interpretable; no test-driven scale choice',
    'partition':'unchanged patient-local K3 intensity/local3mm mean/SD with current-mask centering/scaling',
    'contour':'unchanged1mm grids/base/erode/dilate, same base-reference mapping for all contours',
    'ADC_IVIM':'reuse exact original ZScore features; IVIM D/f only, no Dstar or Dstar-specific mask',
    'feature_names':'legacy T2 Z_ prefix retained for schema compatibility; values in Nyul target units',
    'scope':'T2-only and three-sequence free-fusion habitat models, no clinical/whole-tumor models',
    'limitation':'tumor histogram equalization may remove between-tumor signal; this is not whole-body/normal-tissue Nyul',
}

def landmarks(p):
    reference=np.asarray(p['z'][...,0],float)[np.asarray(p['roi'],bool)&p['valid']]
    if reference.size<12 or not np.isfinite(reference).all():raise ValueError('Invalid T2 reference')
    result=np.percentile(reference,PERCENTILES)
    if result[-1]-result[0]<=1e-12:raise ValueError('Constant T2 reference')
    return result

def learn(source_landmarks,train,ids):
    train=np.asarray(train,int);rows=np.asarray(source_landmarks,float)[train]
    low,high=ENDPOINTS
    rescaled=low+(rows-rows[:,0,None])/(rows[:,-1]-rows[:,0])[:,None]*(high-low)
    target=rescaled.mean(0);target[0],target[-1]=low,high
    if np.any(np.diff(target)<=0):raise ValueError('Non-increasing target')
    return {'target':target,'train_indices':train,'train_ids':[str(ids[i]) for i in train],
            'percentiles':PERCENTILES.copy(),'endpoints':ENDPOINTS,
            'target_sha256':hashlib.sha256(target.astype('<f8').tobytes()).hexdigest()}

def transform(values,source,target):
    source=np.asarray(source,float);target=np.asarray(target,float)
    unique,inverse=np.unique(source,return_inverse=True)
    averaged=np.array([target[inverse==i].mean() for i in range(len(unique))])
    if len(unique)<2:raise ValueError('Constant source landmarks')
    values=np.asarray(values,float)
    out=np.interp(values.ravel(),unique,averaged).reshape(values.shape)
    left,right=values<unique[0],values>unique[-1]
    out[left]=averaged[0]+(values[left]-unique[0])*(averaged[1]-averaged[0])/(unique[1]-unique[0])
    out[right]=averaged[-1]+(values[right]-unique[-1])*(averaged[-1]-averaged[-2])/(unique[-1]-unique[-2])
    return out.astype(np.float32)

def extract_one(task):
    index,aid,p,source,mapping=task
    values=transform(p['z'][...,0],source,mapping['target'])
    classes=radiomics.getFeatureClasses()
    texture_names=sorted('original_'+g+'_'+n for g in prior.regional.CLASSES
        for n,deprecated in classes[g].getFeatureNames().items() if not deprecated)
    metadata=prior.layout('T2',texture_names)
    extractor=featureextractor.RadiomicsFeatureExtractor(binWidth=.25,normalize=False,label=1,
        distances=[1],additionalInfo=False,minimumROISize=12)
    extractor.disableAllFeatures()
    for g in prior.regional.CLASSES:extractor.enableFeatureClassByName(g)
    image=sitk.GetImageFromArray(values);image.SetSpacing(tuple(p['spacing'])[::-1])
    rows={};audit=[];errors=[]
    for kind,key in [('base','roi'),('erode','erode'),('dilate','dilate')]:
        mask=np.asarray(p[key],bool)&p['valid']
        labels=prior.local.partition(values,mask)[0]
        canonical=prior.local.partition(p['z'][...,0],mask)[0]
        row=[]
        for h in range(1,4):
            region=mask&(labels==h)
            try:
                if region.sum()<12:raise ValueError('Habitat fewer than12 voxels')
                roi=sitk.GetImageFromArray(region.astype(np.uint8));roi.CopyInformation(image)
                measured=extractor.execute(image,roi);numbers=[float(measured[n]) for n in texture_names]
            except (ValueError,RuntimeError) as exc:
                numbers=[np.nan]*93;errors.append({'Variant':kind,'Habitat':h,'Error':str(exc)})
            row.extend(numbers)
        v,n=prior.regional.descriptors(values,labels,mask,p['spacing'])
        row.extend(v[j] for j,s in enumerate(n) if not any(s.endswith('Centroid'+a) for a in ['X','Y','Z']))
        v,_=prior.previous.invariant_centroids(labels,mask,p['spacing']);row.extend(v)
        assert len(row)==len(metadata)==315
        rows[kind]=np.asarray(row,float)
        audit.append({'AnalysisID':aid,'Variant':kind,'ReferenceVoxels':int((p['roi']&p['valid']).sum()),
            'ValidVoxels':int(mask.sum()),'ARIvsZScore':float(adjusted_rand_score(canonical[mask],labels[mask])) if mask.any() else 1.,
            'LabelAgreementvsZScore':float(np.mean(canonical[mask]==labels[mask])) if mask.any() else 1.,
            'NormalizedMean':float(values[mask].mean()) if mask.any() else None,
            'NormalizedSD':float(values[mask].std()) if mask.any() else None,
            **{'H'+str(h):int((labels==h).sum()) for h in range(1,4)}})
    return {'index':index,'rows':rows,'metadata':metadata,'audit':audit,'errors':errors}
