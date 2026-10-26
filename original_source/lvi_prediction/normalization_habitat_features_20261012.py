"""Controlled image-normalization comparison on unchanged D/f-only profiles."""
import numpy as np
import SimpleITK as sitk
from sklearn.metrics import adjusted_rand_score
from radiomics import featureextractor
import radiomics
import regional_habitat_features_20261012 as regional
import localtexture_habitat_features_20261012 as local
import multiseq_habitat_features_20261012 as previous
from quantile_habitat_features import original_values

NORMALIZATIONS=('ZScore','RobustZ','PercentileZ')
MODALITIES=('T2','ADC','IVIM')
PROTOCOL={
    'methods':{'ZScore':'existing valid tumor mean/SD, unchanged cached baseline',
        'RobustZ':'valid tumor median / (1.482602218505602*MAD); fallback IQR/1.348979500392164 then ordinary SD',
        'PercentileZ':'base valid tumor P1/P99 clipping, then mean/SD of clipped base tumor'},
    'reference':'per patient/channel base valid tumor; fixed normalization across erosion/dilation; no pooled fitting or outcomes',
    'raw_quantitative':'ADC,D,f raw values recovered once from original unclipped profiles; never recover raw maps from modified normalized images',
    'T2_input':'stored tumor Z-score is affine-equivalent to raw T2 for these three methods; no cross-patient raw T2 interpretation',
    'IVIM':'original D/f validity and same original masks; no Dstar predictor, partition input, or Dstar-specific mask',
    'geometry':'same1mm grids and existing base/1mm erode/dilate masks; no registration/ROI changes',
    'partition':'unchanged patient-local K3 intensity/local3mm mean/SD, seed64212,n_init5,at most4000voxels; current-mask Zscore and descriptor StandardScaler retained',
    'linear_invariance':'ZScore and RobustZ use identical canonical baseline labels: positive affine normalization cancels in existing partition; prevents rounding from becoming an extra intervention',
    'percentile_partition':'partition modified clipped signals using unchanged current-mask normalization/clustering',
    'texture':'93 Original non-shape PyRadiomics per habitat/channel; binWidth.25 in Z-like units; no MinMax/raw-unit texture comparison',
    'descriptors':'T2 normalized statistics; ADC/D/f original quantitative statistics; prior spatial descriptors unchanged',
    'influence_audit':'ARI and label agreement against baseline, signal/clipping/MAD fallback audits, occupied gray levels, finite feature changes by block',
    'whole_tumor_or_clinical':False,'Dstar':False,'Nyul_included':False,
}


def normalize(values,roi,method,baseline=None):
    values=np.asarray(values,float);roi=np.asarray(roi,bool);reference=values[roi]
    if reference.size<2:raise ValueError('Insufficient normalization reference')
    low,high=None,None;fallback='none'
    if method=='ZScore':
        center,scale=float(reference.mean()),float(reference.std())
        out=np.asarray(baseline,np.float32).copy() if baseline is not None else ((values-center)/scale).astype(np.float32)
    elif method=='RobustZ':
        center=float(np.median(reference));scale=float(np.median(np.abs(reference-center))*1.482602218505602)
        if not np.isfinite(scale) or scale<=1e-12:
            scale=float(np.subtract(*np.quantile(reference,[.75,.25]))/1.348979500392164);fallback='IQR'
        if not np.isfinite(scale) or scale<=1e-12:scale=float(reference.std());fallback='SD'
        out=((values-center)/scale).astype(np.float32)
    elif method=='PercentileZ':
        low,high=map(float,np.quantile(reference,[.01,.99]));clipped=np.clip(values,low,high)
        center,scale=float(clipped[roi].mean()),float(clipped[roi].std());out=((clipped-center)/scale).astype(np.float32)
    else:raise ValueError(method)
    if not np.isfinite(scale) or scale<=0:raise ValueError('Undefined normalization scale')
    r=out[roi]
    audit={'Normalization':method,'ReferenceVoxels':int(roi.sum()),'Center':center,'Scale':scale,'Fallback':fallback,
        'P1':low,'P99':high,'ClippedFraction':float(np.mean((reference<low)|(reference>high))) if low is not None else 0.,
        'NormalizedMean':float(r.mean()),'NormalizedSD':float(r.std()),'NormalizedMedian':float(np.median(r)),
        'NormalizedMin':float(r.min()),'NormalizedMax':float(r.max())}
    return out,audit


def partition(values,mask,mod):
    if mod=='IVIM':return previous.ivim_partition(values,mask)
    return local.partition(values[...,0],mask)[0]


def layout(mod,texture_names):
    channels=['D','f'] if mod=='IVIM' else [mod]
    names=[]
    for channel_index,channel in enumerate(channels):
        for h in range(1,4):
            names.extend({'name':f'{mod}__Regional_H{h}_'+(channel+'_' if mod=='IVIM' else '')+n,
                          'modality':mod,'block':'Texture','habitat':h} for n in texture_names)
        _,descriptor_names=regional.descriptors(np.zeros((2,2,2)),np.ones((2,2,2),np.uint8),np.ones((2,2,2),bool),(1.,1.,1.))
        keep=[j for j,s in enumerate(descriptor_names) if not any(s.endswith('Centroid'+a) for a in ['X','Y','Z'])
              and (channel_index==0 or any(s.endswith('_'+q) for q in ['Mean','SD','P10','Median','P90','IQR']))]
        prefix=f'{mod}__Regional_'+(channel+'_Raw_' if mod=='IVIM' else 'Z_' if mod=='T2' else 'Raw_')
        names.extend({'name':prefix+descriptor_names[j],'modality':mod,'block':'Descriptor','habitat':0} for j in keep)
    _,inv=previous.invariant_centroids(np.ones((2,2,2),np.uint8),np.ones((2,2,2),bool),(1.,1.,1.))
    names.extend({'name':f'{mod}__Regional_'+n,'modality':mod,'block':'Descriptor','habitat':0} for n in inv)
    return names


def extract_one(task):
    index,aid,profiles=task
    classes=radiomics.getFeatureClasses()
    texture_names=sorted(f'original_{g}_{n}' for g in regional.CLASSES for n,deprecated in classes[g].getFeatureNames().items() if not deprecated)
    assert len(texture_names)==93
    extractor=featureextractor.RadiomicsFeatureExtractor(binWidth=.25,normalize=False,label=1,distances=[1],additionalInfo=False,minimumROISize=12)
    extractor.disableAllFeatures()
    for group in regional.CLASSES:extractor.enableFeatureClassByName(group)
    result={method:{k:{} for k in ['base','erode','dilate']} for method in NORMALIZATIONS[1:]}
    metadata=[];normaudit=[];partaudit=[];textureaudit=[]
    for mod in MODALITIES:
        p=profiles[mod];raw=original_values(p,mod)
        assert raw.shape[-1]==(2 if mod=='IVIM' else 1) and np.allclose(p['spacing'],(1.,1.,1.))
        meta=layout(mod,texture_names);metadata.extend(meta)
        transformed={}
        channels=['D','f'] if mod=='IVIM' else [mod]
        for method in NORMALIZATIONS:
            arrays=[]
            for channel,name in enumerate(channels):
                v,audit=normalize(raw[...,channel],p['roi'],method,p['z'][...,channel]);arrays.append(v)
                normaudit.append({'AnalysisID':aid,'Modality':mod,'Channel':name,**audit})
            transformed[method]=np.stack(arrays,axis=-1)
        images={}
        for method in NORMALIZATIONS[1:]:
            images[method]=[]
            for channel in range(len(channels)):
                im=sitk.GetImageFromArray(transformed[method][...,channel]);im.SetSpacing(tuple(p['spacing'])[::-1]);images[method].append(im)
        for kind,key in [('base','roi'),('erode','erode'),('dilate','dilate')]:
            mask=np.asarray(p[key],bool)&p['valid']
            canonical=partition(p['z'] if mod=='IVIM' else raw,mask,mod)
            for method in NORMALIZATIONS:
                labels=canonical if method in ['ZScore','RobustZ'] else partition(transformed[method],mask,mod)
                assert np.array_equal(labels>0,mask) or mask.sum()<12
                partaudit.append({'AnalysisID':aid,'Modality':mod,'Variant':kind,'Normalization':method,'ValidVoxels':int(mask.sum()),
                    'ARIvsZScore':float(adjusted_rand_score(canonical[mask],labels[mask])) if mask.any() else 1.,
                    'LabelAgreementvsZScore':float(np.mean(canonical[mask]==labels[mask])) if mask.any() else 1.,
                    **{f'H{h}':int((labels==h).sum()) for h in range(1,4)}})
                row=[]
                for channel,name in enumerate(channels):
                    for h in range(1,4):
                        region=mask&(labels==h);v=transformed[method][...,channel][region]
                        error=None
                        if method!='ZScore':
                            if region.sum()>=12:
                                roi=sitk.GetImageFromArray(region.astype(np.uint8));roi.CopyInformation(images[method][channel])
                                try:
                                    measured=extractor.execute(images[method][channel],roi);numbers=[float(measured[n]) for n in texture_names]
                                except (ValueError,RuntimeError) as exc:numbers,error=[np.nan]*93,type(exc).__name__+': '+str(exc)
                            else:numbers,error=[np.nan]*93,'habitat fewer than12voxels'
                            row.extend(numbers)
                        textureaudit.append({'AnalysisID':aid,'Modality':mod,'Channel':name,'Variant':kind,'Normalization':method,'Habitat':h,
                            'Voxels':int(region.sum()),'OccupiedGrayLevels':int(np.unique(np.floor(v/.25)).size),
                            'SignalMin':float(v.min()) if len(v) else None,'SignalMax':float(v.max()) if len(v) else None,'Error':error,
                            'ExtractedNow':method!='ZScore'})
                    if method!='ZScore':
                        descriptor_values=transformed[method][...,channel] if mod=='T2' else raw[...,channel]
                        v,n=regional.descriptors(descriptor_values,labels,mask,p['spacing'])
                        keep=[j for j,s in enumerate(n) if not any(s.endswith('Centroid'+a) for a in ['X','Y','Z'])
                              and (channel==0 or any(s.endswith('_'+q) for q in ['Mean','SD','P10','Median','P90','IQR']))]
                        row.extend(v[j] for j in keep)
                if method!='ZScore':
                    v,_=previous.invariant_centroids(labels,mask,p['spacing']);row.extend(v)
                    assert len(row)==len(meta)==(612 if mod=='IVIM' else 315)
                    result[method][kind].update({m['name']:float(v) for m,v in zip(meta,row)})
    assert len(metadata)==1242
    return {'index':index,'values':result,'metadata':metadata,'normalization_audit':normaudit,
            'partition_audit':partaudit,'texture_audit':textureaudit}
