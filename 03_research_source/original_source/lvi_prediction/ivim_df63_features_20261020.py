# -*- coding: utf-8 -*-
"""Patient-local63-way clustering ablation; fixed D/f regional measurements."""
from itertools import combinations
from functools import lru_cache
import hashlib
import logging
import time
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import uniform_filter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import radiomics
from radiomics import featureextractor
import normalization_habitat_features_20261012 as prior
import habitat_stable_views_20261013 as views
from quantile_habitat_features import original_values

NORMALIZATIONS = ('PercentileZ', 'RobustZ')
KINDS = (('base', 'roi'), ('erode', 'erode'), ('dilate', 'dilate'))
DIMENSIONS = ('D_Intensity', 'D_Mean3mm', 'D_SD3mm', 'f_Intensity', 'f_Mean3mm', 'f_SD3mm')
PARTITION_SEED = 64212


def combinations63():
    result = []
    for size in range(1, 7):
        for ix in combinations(range(6), size):
            result.append({'ComboID': 'C%02d'%(len(result)+1), 'Dimensions': [DIMENSIONS[j] for j in ix],
                           'Indices': list(ix), 'DimensionCount': len(ix), 'BitMask': sum(1 << j for j in ix)})
    assert len(result) == 63 and len({r['BitMask'] for r in result}) == 63
    return result


@lru_cache(None)
def metadata():
    names = sorted('original_'+metric for metric in views.METRICS)
    meta = prior.layout('IVIM', names)
    assert len(meta) == 156 and len({m['name'] for m in meta}) == 156
    return meta


@lru_cache(None)
def aggregation():
    meta, groups, specs = metadata(), {}, []
    for j, m in enumerate(meta):
        if not m['name'].endswith('_TumorFraction'):
            groups.setdefault(views.grouped_name(m['name']), []).append(j)
    assert len(groups) == 51 and all(len(v) == 3 for v in groups.values())
    for name, ix in groups.items():
        for statistic in ('Mean', 'SD', 'Range'):
            specs.append({'name': name.replace('Regional_', 'Invariant_')+'__'+statistic, 'modality': 'IVIM',
                'block': meta[ix[0]]['block'], 'source_indices': ix, 'statistic': statistic})
    ix = [j for j, m in enumerate(meta) if m['name'].endswith('_TumorFraction')]
    assert len(ix) == 3
    name = views.grouped_name(meta[ix[0]]['name'])
    for statistic in ('SD', 'Maximum', 'Entropy'):
        specs.append({'name': name.replace('Regional_', 'Invariant_')+'__'+statistic, 'modality': 'IVIM',
            'block': 'Descriptor', 'source_indices': ix, 'statistic': statistic})
    assert len(specs) == 156
    return specs


def invariant(raw):
    return views.aggregate(np.asarray(raw)[None], aggregation())[0]


def design(values, mask):
    support = uniform_filter(mask.astype(float), size=3, mode='constant')
    columns = []
    for channel in range(2):
        a = np.asarray(values[..., channel], float)
        v = np.where(mask, (a-float(a[mask].mean()))/max(float(a[mask].std()), 1e-12), 0.)
        mean = uniform_filter(v, size=3, mode='constant')/np.maximum(support, 1e-12)
        second = uniform_filter(v*v, size=3, mode='constant')/np.maximum(support, 1e-12)
        columns.extend([v[mask], mean[mask], np.sqrt(np.maximum(second-mean*mean, 0.))[mask]])
    x = np.column_stack(columns)
    ix = np.random.default_rng(PARTITION_SEED).choice(len(x), min(4000, len(x)), replace=False)
    scaler = StandardScaler().fit(x[ix])
    return scaler.transform(x), ix


def partition(full_design, sample, mask, indices):
    x = full_design[:, indices]
    labels = np.zeros(mask.shape, np.uint8)
    if len(x) < 12:
        return labels
    if len(np.unique(x[sample], axis=0)) < 3:
        labels[mask] = 1
        return labels
    model = KMeans(n_clusters=3, n_init=5, random_state=PARTITION_SEED).fit(x[sample])
    # Canonical names use only included dimensions; downstream features are label invariant.
    priority = [indices.index(j) for j in (0, 3) if j in indices]
    priority.extend(j for j in range(len(indices)) if j not in priority)
    order = np.lexsort(tuple(model.cluster_centers_[:, j] for j in priority[::-1]))
    remap = np.empty(3, np.uint8)
    remap[order] = np.arange(1, 4)
    labels[mask] = remap[model.predict(x)]
    return labels


@lru_cache(None)
def selected_groups():
    result = {}
    for metric in views.METRICS:
        group, name = metric.split('_', 1)
        result.setdefault(group, []).append(name)
    return result


def texture(image, region, direct=True):
    names = sorted('original_'+metric for metric in views.METRICS)
    if region.sum() < 12:
        return np.full(17, np.nan), 'habitat fewer than12voxels'
    coords = np.argwhere(region)
    lo, hi = coords.min(0), coords.max(0)+1
    if np.sum(hi-lo > 1) < 2:
        return np.full(17, np.nan), 'habitat has fewer than2 dimensions'
    try:
        if direct:
            slices = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
            a = sitk.GetImageFromArray(np.asarray(image[slices], np.float32))
            m = sitk.GetImageFromArray(region[slices].astype(np.uint8))
            a.SetSpacing((1., 1., 1.));m.CopyInformation(a)
            result = {}
            for group, selected in selected_groups().items():
                calculator = radiomics.getFeatureClasses()[group](a, m, binWidth=.25, label=1, distances=[1])
                calculator.disableAllFeatures()
                for name in selected:
                    calculator.enableFeatureByName(name)
                result.update({'original_'+group+'_'+k: float(v) for k, v in calculator.execute().items()})
        else:
            a = sitk.GetImageFromArray(np.asarray(image, np.float32))
            m = sitk.GetImageFromArray(region.astype(np.uint8));m.CopyInformation(a)
            extractor = featureextractor.RadiomicsFeatureExtractor(binWidth=.25, normalize=False, label=1,
                distances=[1], additionalInfo=False, minimumROISize=12)
            extractor.disableAllFeatures()
            extractor.enableFeaturesByName(**selected_groups())
            result = extractor.execute(a, m)
        return np.asarray([float(result[n]) for n in names]), None
    except (ValueError, RuntimeError) as error:
        return np.full(17, np.nan), type(error).__name__+': '+str(error)


def measurements(raw, images, labels, mask, direct=True):
    numbers, errors = [], []
    for channel in range(2):
        for h in (1, 2, 3):
            values, error = texture(images[..., channel], mask&(labels == h), direct)
            numbers.extend(values)
            if error:
                errors.append({'Channel': ('D', 'f')[channel], 'Habitat': h, 'Error': error})
        values, names = prior.regional.descriptors(raw[..., channel], labels, mask, (1., 1., 1.))
        keep = [j for j, name in enumerate(names) if not any(name.endswith('Centroid'+a) for a in ('X', 'Y', 'Z'))
            and (channel == 0 or any(name.endswith('_'+q) for q in ('Mean', 'SD', 'P10', 'Median', 'P90', 'IQR')))]
        numbers.extend(values[j] for j in keep)
    values, _ = prior.previous.invariant_centroids(labels, mask, (1., 1., 1.))
    numbers.extend(values)
    assert len(numbers) == 156
    return np.asarray(numbers), errors


def prepare(profile):
    assert np.allclose(profile['spacing'], (1., 1., 1.)) and profile['z'].shape[-1] == 2
    raw, images, audit = original_values(profile, 'IVIM'), {}, []
    for norm in NORMALIZATIONS:
        arrays = []
        for channel in range(2):
            a, record = prior.normalize(raw[..., channel], profile['roi'], norm, profile['z'][..., channel])
            arrays.append(a)
            audit.append({'Channel': ('D', 'f')[channel], **record})
        images[norm] = np.stack(arrays, axis=-1)
    return raw, images, audit


def extract_patient(index, aid, profile, combos=None):
    started = time.time()
    raw, images, normalization = prepare(profile)
    combos = combinations63() if combos is None else combos
    values = {(norm, c['ComboID']): {} for norm in NORMALIZATIONS for c in combos}
    partitions, errors, hashes = [], [], {}
    for kind, key in KINDS:
        mask = np.asarray(profile[key], bool)&np.asarray(profile['valid'], bool)
        assert mask.sum() >= 12
        for norm in NORMALIZATIONS:
            # Affine RobustZ cancels under current-mask zscore; preserve canonical original labels.
            signal = profile['z'] if norm == 'RobustZ' else images[norm]
            x, sample = design(signal, mask)
            for c in combos:
                labels = partition(x, sample, mask, c['Indices'])
                assert np.array_equal(labels > 0, mask)
                row, failed = measurements(raw, images[norm], labels, mask)
                values[(norm, c['ComboID'])][kind] = invariant(row)
                hashes[(norm, c['ComboID'], kind)] = hashlib.sha256(labels.tobytes()).hexdigest()
                sizes = [int((labels == h).sum()) for h in (1, 2, 3)]
                partitions.append({'AnalysisID': aid, 'Normalization': norm, 'ComboID': c['ComboID'], 'Variant': kind,
                    'MaskVoxels': int(mask.sum()), 'H1': sizes[0], 'H2': sizes[1], 'H3': sizes[2],
                    'SmallHabitatCount': sum(n < 12 for n in sizes), 'FiniteFeatureCount': int(np.isfinite(values[(norm, c['ComboID'])][kind]).sum())})
                errors.extend({'AnalysisID': aid, 'Normalization': norm, 'ComboID': c['ComboID'], 'Variant': kind, **e} for e in failed)
    return {'index': int(index), 'AnalysisID': aid, 'values': values, 'partition_hashes': hashes,
        'partition_audit': partitions, 'texture_errors': errors, 'normalization_audit': normalization,
        'elapsed_seconds': time.time()-started}


def configure():
    radiomics.logger.setLevel(logging.ERROR)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
