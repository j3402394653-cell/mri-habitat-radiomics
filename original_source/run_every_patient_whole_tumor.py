"""Shared image reading, geometry and ROI utilities for the latest pipeline.

The obsolete whole-tumor extraction command has been removed.
"""
import os
import re
import numpy as np
import nibabel as nib
import SimpleITK as sitk

def read_nifti(path, is_mask=False):
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    nii = nib.load(path)
    data = np.asarray(nii.dataobj)
    while data.ndim > 3 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise RuntimeError(f"只支持3D NIfTI，实际维度：{data.shape}")
    arr = np.transpose(data, (2,1,0))
    if is_mask:
        if not np.all(np.isfinite(arr)):
            raise ValueError("Mask 含 NaN 或 Inf。")
        arr = (arr > 0).astype(np.uint8)
    else:
        arr = arr.astype(np.float32)
    img = sitk.GetImageFromArray(arr)
    affine = np.diag([-1.,-1.,1.,1.]) @ np.asarray(nii.affine, dtype=float)
    if not np.isfinite(affine).all():
        raise ValueError("NIfTI affine 含非有限值")
    matrix = affine[:3,:3]
    spacing = np.linalg.norm(matrix, axis=0)
    if not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("NIfTI spacing 无效")
    direction = matrix / spacing
    if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-4):
        raise ValueError("NIfTI affine 含剪切，需先校正到正交网格")
    origin = affine[:3,3]
    img.SetSpacing(tuple(map(float, spacing)))
    img.SetOrigin(tuple(map(float, origin)))
    img.SetDirection(tuple(map(float, direction.flatten())))
    return img


def is_nii(name):
    low = name.lower()
    return low.endswith(".nii") or low.endswith(".nii.gz")


def find_t2(patient_dir):
    c=[]
    for name in os.listdir(patient_dir):
        p=os.path.join(patient_dir,name)
        low=name.lower()
        if os.path.isfile(p) and is_nii(name) and "t2" in low and not any(x in low for x in ["adc","dwi","segmentation","label","mask","habitat"]):
            c.append(p)
    if not c: raise FileNotFoundError("没有找到 T2 NIfTI 文件。")
    c.sort()
    if len(c)>1: raise ValueError("发现多个 T2 候选，请使用 --t2 指定文件。")
    return c[0]


def find_adc(patient_dir):
    c=[]
    adc_series=set()
    for name in os.listdir(patient_dir):
        directory=os.path.join(patient_dir,name)
        if not os.path.isdir(directory) or "adc" not in name.lower(): continue
        for series_id in sitk.ImageSeriesReader.GetGDCMSeriesIDs(directory) or ():
            files=sitk.ImageSeriesReader.GetGDCMSeriesFileNames(directory,series_id)
            if not files: continue
            reader=sitk.ImageFileReader()
            reader.SetFileName(files[0])
            reader.ReadImageInformation()
            if reader.HasMetaDataKey("0020|0011"):
                adc_series.add(int(reader.GetMetaData("0020|0011").strip()))
    for name in os.listdir(patient_dir):
        p=os.path.join(patient_dir,name)
        low=name.lower()
        if not (os.path.isfile(p) and is_nii(name)): continue
        if any(x in low for x in ["segmentation","label","mask","habitat"]): continue
        match=re.search(r"_(\d+)\.nii(?:\.gz)?$",low)
        matched=match is not None and int(match.group(1)) in adc_series
        if "adc" not in low and not matched: continue
        shape=nib.load(p).shape
        if len(shape)<3 or any(size!=1 for size in shape[3:]): continue
        c.append(p)
    if not c: raise FileNotFoundError("没有找到 ADC NIfTI 文件。")
    c.sort()
    if len(c)>1: raise ValueError("发现多个 ADC 候选，请使用 --adc 指定文件。")
    return c[0]


def find_mask(patient_dir):
    for name in ["Segmentation-label-T2.nii.gz","Segmentation-label-T2.nii"]:
        p=os.path.join(patient_dir,name)
        if os.path.isfile(p): return p
    c=[]
    for name in os.listdir(patient_dir):
        p=os.path.join(patient_dir,name)
        low=name.lower()
        if os.path.isfile(p) and is_nii(name) and "adc" not in low and "habitat" not in low and any(x in low for x in ["segmentation","mask","label"]):
            c.append(p)
    if not c: raise FileNotFoundError("没有找到 T2 segmentation mask。")
    c.sort()
    if len(c)>1: raise ValueError("发现多个 Mask 候选，请使用 --mask 指定文件。")
    return c[0]


def same_geometry(a,b):
    return (a.GetSize()==b.GetSize()
            and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=1e-5, rtol=0)
            and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=1e-4, rtol=0)
            and np.allclose(a.GetDirection(), b.GetDirection(), atol=1e-5, rtol=0))


def find_sequence_mask(patient_dir, sequence):
    """Require a sequence-labelled contour; never borrow the other sequence ROI."""
    sequence = sequence.lower()
    if sequence not in ('t2', 'adc', 'dwi'):
        raise ValueError('Unknown mask sequence')
    candidates = []
    for name in os.listdir(patient_dir):
        low = name.lower()
        if (os.path.isfile(os.path.join(patient_dir, name)) and is_nii(name)
                and sequence in low and 'habitat' not in low
                and any(token in low for token in ('segmentation', 'mask', 'label'))
                and not any(other in low for other in ('t2', 'adc', 'dwi') if other != sequence)):
            candidates.append(os.path.join(patient_dir, name))
    if len(candidates) != 1:
        raise ValueError(f'{sequence.upper()}: expected one sequence-specific mask, found {len(candidates)}; specify its path explicitly')
    return candidates[0]


def resample_to_t2(img,t2,is_mask=False):
    return sitk.Resample(img,t2,sitk.Transform(),
                         sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear,
                         0 if is_mask else float("nan"),
                         sitk.sitkUInt8 if is_mask else sitk.sitkFloat32)


def roi_coverage(tumor,t2a,adca):
    original=int(tumor.sum())
    if not original:raise ValueError('Mask 中没有肿瘤体素。')
    valid=tumor & np.isfinite(t2a) & np.isfinite(adca)
    return valid,{'OriginalTumorVoxels':original,'CommonValidVoxels':int(valid.sum()),
        'CommonCoverageFraction':float(valid.sum()/original),
        'ADCFiniteFractionInOriginalTumor':float(np.isfinite(adca[tumor]).mean()),
        'T2FiniteFractionInOriginalTumor':float(np.isfinite(t2a[tumor]).mean()),
        'ADCZeroVoxelsInCommonROI':int((adca[valid]==0).sum()),
        'ADCNegativeVoxelsInCommonROI':int((adca[valid]<0).sum()),
        'ROIComplete':bool(valid.sum()==original),
        'ROIMeaning':'COMMON_FINITE_T2_ADC_TUMOR_REGION'}
