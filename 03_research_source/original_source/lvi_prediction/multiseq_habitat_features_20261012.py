"""Three-sequence pure habitats: cached T2/ADC, joint D/f IVIM, invariant spatial descriptors."""
import logging
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import uniform_filter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import radiomics
from radiomics import featureextractor
import regional_habitat_features_20261012 as shared
import localtexture_habitat_features_20261012 as local
from quantile_habitat_features import original_values

MODALITIES = ("T2","ADC","IVIM")
PROTOCOL = {
    "scope":"pure habitats in T2, ADC, IVIM(D/f); no whole-tumor or clinical predictors",
    "partition_T2_ADC":local.PROTOCOL["partition"],
    "partition_IVIM":"patient-local K=3 KMeans on normalized D/f intensities and 3-mm local means/SDs; order D center then f",
    "refit":"separately for base/erode/dilate contours; no patient-outcome or cohort information",
    "sampling":"up to 4000 current-mask voxels, seed64212, n_init5",
    "texture":"93 Original non-shape PyRadiomics per habitat per channel; Z-score binWidth .25",
    "IVIM_quality":"existing valid maps: native D>0, 0<=f<.7, NRMSE<=.15; quality masks and existing resampling unchanged",
    "quantitative_values":"recover original ADC/D/f with saved unclipped normalization; no new unit certification",
    "spatial":"habitat occupancy, current inner 2-mm band enrichment, normalized boundary depth, 6-neighbor contact fractions, rotation-invariant centroid offsets/distances",
    "spatial_direction":"axis-specific centroid X/Y/Z excluded; physical Euclidean distances normalized by tumor RMS radius",
    "cross_sequence":"each acquisition uses its own ROI/space; patient-feature fusion only, no voxel pairing",
    "repeatability":"conditional contour and partition refit on fixed normalized images; not scan-rescan or reader-repeatability",
}


def invariant_centroids(labels,mask,spacing):
    coords = np.argwhere(mask)*np.asarray(spacing)
    center = coords.mean(0) if len(coords) else np.zeros(3)
    radius = max(float(np.sqrt(np.mean(np.sum((coords-center)**2,axis=1)))),1e-12) if len(coords) else 1.
    centers = []
    for h in range(1,4):
        points = np.argwhere(mask&(labels==h))*np.asarray(spacing)
        centers.append(points.mean(0) if len(points)>=12 else np.full(3,np.nan))
    values = [float(np.linalg.norm(c-center)/radius) for c in centers]
    values += [float(np.linalg.norm(centers[a-1]-centers[b-1])/radius) for a,b in [(1,2),(1,3),(2,3)]]
    names = [f"H{h}_CentroidOffsetInvariant" for h in range(1,4)]+["H1H2_CentroidDistanceInvariant","H1H3_CentroidDistanceInvariant","H2H3_CentroidDistanceInvariant"]
    return values,names


def ivim_partition(z,mask):
    mask = np.asarray(mask,bool);labels = np.zeros(mask.shape,np.uint8)
    if mask.sum()<12:return labels
    support = uniform_filter(mask.astype(float),size=3,mode="constant")
    columns = []
    for c in range(2):
        a = np.asarray(z[...,c],float);mu,sd = float(a[mask].mean()),float(a[mask].std())
        v = np.where(mask,(a-mu)/max(sd,1e-12),0.)
        mean = uniform_filter(v,size=3,mode="constant")/np.maximum(support,1e-12)
        second = uniform_filter(v*v,size=3,mode="constant")/np.maximum(support,1e-12)
        columns.extend([v[mask],mean[mask],np.sqrt(np.maximum(second-mean*mean,0.))[mask]])
    x = np.column_stack(columns)
    if np.unique(x,axis=0).shape[0]<3:
        labels[mask]=1
        return labels
    rng = np.random.default_rng(64212);ix = rng.choice(len(x),min(4000,len(x)),replace=False)
    scaler = StandardScaler().fit(x[ix]);normalized = scaler.transform(x)
    cluster = KMeans(n_clusters=3,n_init=5,random_state=64212).fit(normalized[ix])
    order = np.lexsort((cluster.cluster_centers_[:,3],cluster.cluster_centers_[:,0]))
    remap = np.empty(3,np.uint8);remap[order]=np.arange(1,4)
    labels[mask]=remap[cluster.predict(normalized)]
    return labels


def extract(profiles,indices,ids,cache):
    radiomics.logger.setLevel(logging.ERROR);sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    np.testing.assert_array_equal(cache["indices"],indices)
    classes = radiomics.getFeatureClasses()
    texture_names = sorted(f"original_{g}_{n}" for g in shared.CLASSES for n,depr in classes[g].getFeatureNames().items() if not depr)
    assert len(texture_names)==93
    extractor = featureextractor.RadiomicsFeatureExtractor(binWidth=.25,normalize=False,label=1,distances=[1],additionalInfo=False,minimumROISize=12)
    extractor.disableAllFeatures()
    for group in shared.CLASSES:extractor.enableFeatureClassByName(group)
    keep = [j for j,m in enumerate(cache["metadata"]) if not any(m["name"].endswith("_Centroid"+a) for a in ["X","Y","Z"])]
    assert len(keep)==618
    matrices,audit,quality,part_audit = {k:[] for k in ["base","erode","dilate"]},[],[],[]
    metadata = None
    for done,i in enumerate(indices,1):
        current = {k:cache[k][done-1,keep].tolist() for k in matrices}
        meta = [cache["metadata"][j] for j in keep]
        for mod in ["T2","ADC"]:
            p = profiles[int(i)][mod]
            values = original_values(p,mod)[...,0]
            for kind,maskkey in [("base","roi"),("erode","erode"),("dilate","dilate")]:
                mask = np.asarray(p[maskkey],bool)&p["valid"]
                labels,_ = local.partition(values,mask)
                numbers,names = invariant_centroids(labels,mask,p["spacing"])
                current[kind].extend(numbers)
                if kind=="base":meta.extend({"name":f"{mod}__Regional_{n}","modality":mod,"block":"Descriptor","habitat":0} for n in names)
        p = profiles[int(i)]["IVIM"];raw = original_values(p,"IVIM")
        assert p["z"].shape[-1]==2 and np.allclose(p["spacing"],(1.,1.,1.))
        quality.append({"AnalysisID":ids[int(i)],"ROIValidVoxels":int(p["roi"].sum()),
            "NativeTumorValidFraction":float(p.get("native_tumor_valid_fraction",p["tumor_valid_fraction"])),
            "TumorValidFraction":float(p["tumor_valid_fraction"]),"DMean":float(raw[...,0][p["roi"]].mean()),
            "fMean":float(raw[...,1][p["roi"]].mean())})
        images = []
        for channel in range(2):
            im = sitk.GetImageFromArray(np.asarray(p["z"][...,channel],np.float32));im.SetSpacing(tuple(p["spacing"])[::-1]);images.append(im)
        for kind,maskkey in [("base","roi"),("erode","erode"),("dilate","dilate")]:
            mask = np.asarray(p[maskkey],bool)&p["valid"];labels = ivim_partition(p["z"],mask)
            assert np.array_equal(labels>0,mask) or mask.sum()<12
            part_audit.append({"AnalysisID":ids[int(i)],"Variant":kind,"ValidVoxels":int(mask.sum()),
                               **{f"H{h}":int((labels==h).sum()) for h in range(1,4)}})
            numbers,names = [],[]
            for channel,name in enumerate(["D","f"]):
                for h in range(1,4):
                    region = mask&(labels==h);error = None
                    if region.sum()>=12:
                        roi = sitk.GetImageFromArray(region.astype(np.uint8));roi.CopyInformation(images[channel])
                        try:
                            result = extractor.execute(images[channel],roi);v = [float(result[n]) for n in texture_names]
                        except (ValueError,RuntimeError) as exc:v,error=[np.nan]*93,type(exc).__name__+": "+str(exc)
                    else:v,error=[np.nan]*93,"habitat fewer than 12 voxels"
                    numbers.extend(v)
                    names.extend({"name":f"IVIM__Regional_H{h}_{name}_{n}","modality":"IVIM","block":"Texture","habitat":h} for n in texture_names)
                    audit.append({"AnalysisID":ids[int(i)],"Variant":kind,"Channel":name,"Habitat":h,"Voxels":int(region.sum()),"Error":error})
                v,n = shared.descriptors(raw[...,channel],labels,mask,p["spacing"])
                kept = [j for j,s in enumerate(n) if not any(s.endswith("Centroid"+a) for a in ["X","Y","Z"])
                        and (channel==0 or any(s.endswith("_"+q) for q in ["Mean","SD","P10","Median","P90","IQR"]))]
                numbers.extend(v[j] for j in kept)
                names.extend({"name":f"IVIM__Regional_{name}_Raw_{n[j]}","modality":"IVIM","block":"Descriptor","habitat":0} for j in kept)
            v,n = invariant_centroids(labels,mask,p["spacing"]);numbers.extend(v)
            names.extend({"name":f"IVIM__Regional_{s}","modality":"IVIM","block":"Descriptor","habitat":0} for s in n)
            assert len(numbers)==len(names)==612
            current[kind].extend(numbers)
            if kind=="base":meta.extend(names)
        assert len(meta)==1242 and all(len(v)==1242 for v in current.values())
        if metadata is None:metadata=meta
        assert metadata==meta
        for kind in matrices:matrices[kind].append(current[kind])
        if done%5==0 or done==len(indices):print(f"Three-sequence habitat extraction {done}/{len(indices)}",flush=True)
    return {**{k:np.asarray(v,float) for k,v in matrices.items()},"indices":np.asarray(indices,int),"metadata":metadata,
        "protocol":PROTOCOL,"IVIM_extraction_audit":audit,"IVIM_quality_audit":quality,"IVIM_partition_audit":part_audit,
        "pyradiomics_version":radiomics.__version__}
