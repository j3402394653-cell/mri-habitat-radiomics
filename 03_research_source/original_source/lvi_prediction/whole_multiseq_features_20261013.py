"""Patient-local whole-tumor T2/ADC/D/f textures, matched to current habitats."""
import numpy as np
import radiomics
import SimpleITK as sitk
from radiomics import featureextractor
from quantile_habitat_features import original_values
from normalization_habitat_features_20261012 import normalize

NORMALIZATIONS = ('PercentileZ', 'RobustZ')
KINDS = (('base', 'roi'), ('erode', 'erode'), ('dilate', 'dilate'))
CLASSES = ('firstorder', 'glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm')


def extract_one(task):
    index, aid, profiles = task
    classes = radiomics.getFeatureClasses()
    names = sorted('original_'+g+'_'+n for g in CLASSES
                   for n, deprecated in classes[g].getFeatureNames().items() if not deprecated)
    assert len(names) == 93
    extractor = featureextractor.RadiomicsFeatureExtractor(
        binWidth=.25, normalize=False, label=1, distances=[1],
        additionalInfo=False, minimumROISize=12)
    extractor.disableAllFeatures()
    for g in CLASSES:
        extractor.enableFeatureClassByName(g)
    values = {n: {kind: [] for kind, _ in KINDS} for n in NORMALIZATIONS}
    metadata, norms, audits = [], [], []
    for mod in ('T2', 'ADC', 'IVIM'):
        p = profiles[mod]
        raw = original_values(p, mod)
        assert raw.shape[-1] == (2 if mod == 'IVIM' else 1)
        assert np.allclose(p['spacing'], (1., 1., 1.))
        for ci, channel in enumerate(('D', 'f') if mod == 'IVIM' else (mod,)):
            metadata.extend({'name': mod+'__Whole_'+(channel+'_' if mod == 'IVIM' else '')+s,
                             'modality': mod, 'channel': channel, 'block': 'Texture', 'habitat': 0}
                            for s in names)
            for norm in NORMALIZATIONS:
                normalized, audit = normalize(raw[..., ci], p['roi'], norm, p['z'][..., ci])
                norms.append({'AnalysisID': aid, 'Modality': mod, 'Channel': channel, **audit})
                im = sitk.GetImageFromArray(normalized)
                im.SetSpacing(tuple(p['spacing'])[::-1])
                for kind, maskkey in KINDS:
                    mask = np.asarray(p[maskkey], bool) & p['valid']
                    error = None
                    try:
                        if mask.sum() < 12:
                            raise ValueError('whole mask fewer than12voxels')
                        roi = sitk.GetImageFromArray(mask.astype(np.uint8))
                        roi.CopyInformation(im)
                        measured = extractor.execute(im, roi)
                        numbers = [float(measured[s]) for s in names]
                    except (ValueError, RuntimeError) as exc:
                        numbers = [np.nan]*93
                        error = type(exc).__name__+': '+str(exc)
                    values[norm][kind].extend(numbers)
                    audits.append({'AnalysisID': aid, 'Modality': mod, 'Channel': channel,
                                   'Normalization': norm, 'Variant': kind, 'Voxels': int(mask.sum()),
                                   'FiniteFeatureCount': int(np.isfinite(numbers).sum()), 'Error': error})
    assert len(metadata) == 372
    return {'index': index, 'metadata': metadata, 'values': values,
            'normalization_audit': norms, 'texture_audit': audits}
