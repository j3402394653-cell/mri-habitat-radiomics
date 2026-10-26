"""Outcome-independent patient-local regional radiomics and spatial habitat features."""
import logging
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt
import radiomics
from radiomics import featureextractor
from quantile_habitat_features import original_values

MODALITIES = ("T2", "ADC")
CLASSES = ("firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm")
QUANTILES = (1/3, 2/3)
PROTOCOL = {
    "partition": "patient-local intensity tertiles, refitted separately in each contour variant; ties stay together",
    "modalities": MODALITIES, "K": 3, "quantiles": QUANTILES,
    "texture": "93 Original non-shape PyRadiomics features per habitat per modality; binWidth=.25 on existing patient Z-score",
    "raw_ADC": "recover original unclipped map values using saved normalization mean/SD; units not independently certified",
    "spatial": "6-neighbor inter-habitat contact fractions, normalized habitat centroids, boundary enrichment, mean normalized boundary depth",
    "boundary": "recompute current-mask internal 2-mm band; no shared original boundary mask across variants",
    "perturbation": "base, existing 1-mm erode/dilate masks; refit quantiles; image normalization remains fixed",
    "repeatability_scope": "conditional contour+partition sensitivity; not test-retest or independent-reader reproducibility",
    "whole_tumor_predictors": False, "clinical_predictors": False, "excluded_transfer_method": False,
}


def partition(values, mask):
    mask = np.asarray(mask,bool)
    if int(mask.sum()) < 12:
        return np.zeros(mask.shape,np.uint8), np.array([np.nan,np.nan])
    cut = np.quantile(values[mask],QUANTILES)
    labels = np.zeros(mask.shape,np.uint8)
    labels[mask] = np.digitize(values[mask],cut,right=True)+1
    return labels,cut


def contacts(labels, mask):
    count,total = {(1,2):0,(1,3):0,(2,3):0},0
    for axis in range(3):
        left,right = [slice(None)]*3,[slice(None)]*3
        left[axis],right[axis] = slice(None,-1),slice(1,None)
        left,right = tuple(left),tuple(right)
        valid = mask[left]&mask[right]
        a,b = labels[left][valid],labels[right][valid]
        total += len(a)
        for h,j in count: count[(h,j)] += int(((a==h)&(b==j)|(a==j)&(b==h)).sum())
    return [count[p]/total if total else np.nan for p in count]


def descriptors(values,labels,mask,spacing):
    depth = distance_transform_edt(mask,sampling=spacing)
    band = mask&(depth<=2.)
    loc = np.argwhere(mask)
    center = loc.mean(0) if len(loc) else np.zeros(3)
    size = np.maximum(np.ptp(loc,axis=0),1.) if len(loc) else np.ones(3)
    max_depth = max(float(depth.max()),1.)
    v,names = [],[]
    for h in range(1,4):
        region = mask&(labels==h)
        a = values[region]; n = len(a)
        if n >= 12:
            qs = np.quantile(a,[.1,.25,.5,.75,.9])
            stats = [float(a.mean()),float(a.std()),float(qs[0]),float(qs[2]),float(qs[4]),float(qs[3]-qs[1])]
            centroid = (np.argwhere(region).mean(0)-center)/size
            spatial = [n/max(int(mask.sum()),1),int((region&band).sum())/n,
                       float(np.mean(depth[region]))/max_depth,*centroid]
        else: stats,spatial = [np.nan]*6,[np.nan]*6
        v.extend(stats+spatial)
        names.extend(f"H{h}_{s}" for s in ["Mean","SD","P10","Median","P90","IQR",
            "TumorFraction","BoundaryEnrichment","NormalizedDepth","CentroidZ","CentroidY","CentroidX"])
    v.extend(contacts(labels,mask)); names.extend(["ContactH1H2","ContactH1H3","ContactH2H3"])
    return v,names


def extract(profiles,indices,ids):
    radiomics.logger.setLevel(logging.ERROR)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    classes = radiomics.getFeatureClasses()
    texture_names = sorted(f"original_{group}_{name}" for group in CLASSES
                           for name,deprecated in classes[group].getFeatureNames().items() if not deprecated)
    assert len(texture_names) == 93
    extractor = featureextractor.RadiomicsFeatureExtractor(binWidth=.25,normalize=False,label=1,
        distances=[1],additionalInfo=False,force2D=False,minimumROISize=12)
    extractor.disableAllFeatures()
    for group in CLASSES: extractor.enableFeatureClassByName(group)
    matrices,audit,partition_audit = {k:[] for k in ["base","erode","dilate"]},[],[]
    metadata = None
    for done,i in enumerate(indices,1):
        current = {k:[] for k in matrices}; meta = []
        for mod in MODALITIES:
            p = profiles[int(i)][mod]
            z = np.asarray(p["z"][...,0],np.float32)
            values = original_values(p,mod)[...,0] if mod == "ADC" else z
            assert np.allclose(p["spacing"],(1.,1.,1.))
            image = sitk.GetImageFromArray(z); image.SetSpacing(tuple(p["spacing"])[::-1])
            for kind,maskkey in [("base","roi"),("erode","erode"),("dilate","dilate")]:
                mask = np.asarray(p[maskkey],bool)&np.asarray(p["valid"],bool)
                labels,cut = partition(values,mask)
                assert np.array_equal(labels>0,mask) or mask.sum()<12
                row,names = [],[]
                for h in range(1,4):
                    region = mask&(labels==h)
                    if int(region.sum()) >= 12:
                        roi = sitk.GetImageFromArray(region.astype(np.uint8)); roi.CopyInformation(image)
                        try:
                            measured = extractor.execute(image,roi)
                            numbers = [float(measured[n]) for n in texture_names]
                            error = None
                        except (ValueError,RuntimeError) as exc:
                            numbers,error = [np.nan]*93,type(exc).__name__+": "+str(exc)
                    else: numbers,error = [np.nan]*93,"habitat fewer than 12 voxels"
                    row.extend(numbers)
                    names.extend({"name":f"{mod}__Regional_H{h}_{n}","modality":mod,"block":"Texture","habitat":h} for n in texture_names)
                    audit.append({"AnalysisID":ids[int(i)],"Modality":mod,"Variant":kind,
                                  "Habitat":h,"Voxels":int(region.sum()),"Error":error})
                numbers,descriptor_names = descriptors(values,labels,mask,p["spacing"])
                row.extend(numbers)
                names.extend({"name":f"{mod}__Regional_{'Raw' if mod=='ADC' else 'Z'}_{n}",
                              "modality":mod,"block":"Descriptor","habitat":0} for n in descriptor_names)
                current[kind].extend(row)
                if kind == "base": meta.extend(names)
                else:
                    assert names == meta[-len(names):]
                partition_audit.append({"AnalysisID":ids[int(i)],"Modality":mod,"Variant":kind,
                    "CutLow":None if not np.isfinite(cut[0]) else float(cut[0]),
                    "CutHigh":None if not np.isfinite(cut[1]) else float(cut[1]),
                    "ValidVoxels":int(mask.sum()),"H1":int((labels==1).sum()),"H2":int((labels==2).sum()),"H3":int((labels==3).sum())})
        if metadata is None: metadata = meta
        assert metadata == meta
        for kind in matrices: matrices[kind].append(current[kind])
        if done%5 == 0 or done == len(indices): print(f"Regional habitat extraction {done}/{len(indices)}",flush=True)
    return {**{k:np.asarray(v,float) for k,v in matrices.items()},"indices":np.asarray(indices,int),
            "metadata":metadata,"extraction_audit":audit,"partition_audit":partition_audit,
            "protocol":PROTOCOL,"pyradiomics_version":radiomics.__version__}
