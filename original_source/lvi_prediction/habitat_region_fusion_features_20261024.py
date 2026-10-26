"""Separate intra/core/peritumoral K3 habitats with within-region perturbations."""
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt as edt
import habitat_margin_features_20261024 as prior
from quantile_habitat_features import original_values

REGIONS=('Intra','Core1','Peri1')
KINDS=('erode','base','dilate')
NORMS=prior.NORMS


def region_masks(p):
    g=p['geometric_roi'];v=p['full_valid'];spacing=p['spacing']
    variants={'base':g,'erode':g&(edt(g,sampling=spacing)>1.),'dilate':g|(edt(~g,sampling=spacing)<=1.)}
    result={};audit=[]
    for kind,t in variants.items():
        core=t&(edt(t,sampling=spacing)>1.) if t.any() else t.copy()
        d=edt(~t,sampling=spacing) if t.any() else np.full(t.shape,np.inf)
        geometries={'Intra':t,'Core1':core,'Peri1':(~t)&(d<=1.)}
        for region,geo in geometries.items():
            m=geo&v;result[(region,kind)]=m
            edge=sum(int(np.take(m,i,axis=ax).sum()) for ax in range(3) for i in (0,-1))
            assert edge==0
            if region.startswith('Peri'):assert not (m&t).any()
            if region=='Core1':assert not (m&~t).any()
            audit.append({'Region':region,'Variant':kind,'GeometricVoxels':int(geo.sum()),'ValidVoxels':int(m.sum()),
                'ValidGeometricCoverage':float(m.sum()/geo.sum()) if geo.any() else 0.,'ReferenceEdgeVoxels':edge})
        assert not (result[('Core1',kind)]&~result[('Intra',kind)]).any()
    return result,audit


def extract_one(task):
    index,aid,entry,old,cached=task
    names,ex,metadata=prior.setup()
    assert metadata==cached['metadata']
    values={(n,r,k):[] for n in NORMS for r in REGIONS for k in KINDS}
    audits=[];textures=[]
    for mod in prior.MODS:
        p=prior.extended_profile(entry,mod,old[mod]);raw=original_values(p,mod)
        masks,audit=region_masks(p)
        audits.extend({'AnalysisID':aid,'Modality':mod,**a} for a in audit)
        for n in NORMS:
            arrays=[];images=[]
            for c in range(raw.shape[-1]):
                z,_=prior.prior.normalize(raw[...,c],p['roi'],n,p['z'][...,c]);arrays.append(z)
                image=sitk.GetImageFromArray(z);image.SetSpacing(tuple(p['spacing'])[::-1]);images.append(image)
            normalized={'name':n,'values':np.stack(arrays,axis=-1)}
            for region in REGIONS:
                for kind in KINDS:
                    # Intra and unperturbedCore1 already have exact verified new-grid features.
                    if region=='Intra' or (region=='Core1' and kind=='base'):continue
                    row,ta=prior.measure(p,mod,masks[(region,kind)],normalized,raw,images,names,ex)
                    values[(n,region,kind)].extend(row)
                    textures.extend({'AnalysisID':aid,'Modality':mod,'Normalization':n,'Region':region,'Variant':kind,**a} for a in ta)
    for n in NORMS:
        for kind in KINDS:
            values[(n,'Intra',kind)]=cached['base_values'][n] if kind=='base' else cached['values'][(n,1.,kind)]
        values[(n,'Core1','base')]=cached['values'][(n,1.,'erode')]
    assert all(len(v)==321 for v in values.values())
    return {'index':index,'metadata':metadata,'values':values,'geometry_audit':audits,'texture_audit':textures,
        'protocol':'Three tumor-contour variants0/-1/+1mm. Derive eachregionseparatelyinsideeachvariant;onlythen compute within-regionICC. Peri isstrictlyexterior,notwholedilatedROI.',
        'native_core_verified':True}
