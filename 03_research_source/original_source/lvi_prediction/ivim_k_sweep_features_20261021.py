# -*- coding: utf-8 -*-
"""Outcome-independent IVIM K2..6 features with exactly matched K3 representation."""
from functools import lru_cache
from itertools import combinations
import hashlib
import re
import time
import numpy as np
from scipy.ndimage import distance_transform_edt
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
import ivim_df63_features_20261020 as old

KS = (2, 3, 4, 5, 6)
DRIVERS = (('C08', (0, 2)), ('C18', (2, 5)), ('C38', (2, 3, 4)))
NORMS = old.NORMALIZATIONS
KINDS = old.KINDS


def specifications():
    return [{'ImageKey': n + '__' + c + '__K' + str(k), 'Normalization': n,
             'ComboID': c, 'Indices': list(ix), 'K': k,
             'Dimensions': '+'.join(old.DIMENSIONS[i] for i in ix)}
            for n in NORMS for c, ix in DRIVERS for k in KS]


def partition(x, sample, mask, indices, k):
    a = x[:, indices]
    labels = np.zeros(mask.shape, np.uint8)
    if len(np.unique(a[sample], axis=0)) < k:
        labels[mask] = 1
        return labels
    model = KMeans(n_clusters=k, n_init=5, random_state=old.PARTITION_SEED).fit(a[sample])
    priority = [indices.index(j) for j in (0, 3) if j in indices]
    priority.extend(j for j in range(len(indices)) if j not in priority)
    order = np.lexsort(tuple(model.cluster_centers_[:, j] for j in priority[::-1]))
    remap = np.empty(k, np.uint8); remap[order] = np.arange(1, k + 1)
    labels[mask] = remap[model.predict(a)]
    return labels


def descriptors(values, labels, mask, k):
    depth = distance_transform_edt(mask)
    band = mask & (depth <= 2.)
    loc = np.argwhere(mask)
    center, size = loc.mean(0), np.maximum(np.ptp(loc, axis=0), 1.)
    max_depth = max(float(depth.max()), 1.)
    numbers, names = [], []
    for h in range(1, k + 1):
        region = mask & (labels == h)
        a = values[region]; count = len(a)
        if count >= 12:
            qs = np.quantile(a, [.1, .25, .5, .75, .9])
            stats = [float(a.mean()), float(a.std()), float(qs[0]), float(qs[2]), float(qs[4]), float(qs[3] - qs[1])]
            centroid = (np.argwhere(region).mean(0) - center) / size
            spatial = [count / int(mask.sum()), int((region & band).sum()) / count,
                       float(np.mean(depth[region])) / max_depth, *centroid]
        else:
            stats, spatial = [np.nan] * 6, [np.nan] * 6
        numbers.extend(stats + spatial)
        names.extend('H%d_%s' % (h, name) for name in ('Mean', 'SD', 'P10', 'Median', 'P90', 'IQR',
                      'TumorFraction', 'BoundaryEnrichment', 'NormalizedDepth', 'CentroidZ', 'CentroidY', 'CentroidX'))
    pairs = list(combinations(range(1, k + 1), 2))
    counts, total = {pair: 0 for pair in pairs}, 0
    for axis in range(3):
        left, right = [slice(None)] * 3, [slice(None)] * 3
        left[axis], right[axis] = slice(None, -1), slice(1, None)
        left, right = tuple(left), tuple(right)
        valid = mask[left] & mask[right]
        a, b = labels[left][valid], labels[right][valid]
        total += len(a)
        for h, j in pairs:
            counts[(h, j)] += int((((a == h) & (b == j)) | ((a == j) & (b == h))).sum())
    numbers.extend(counts[pair] / total if total else np.nan for pair in pairs)
    names.extend('ContactH%dH%d' % pair for pair in pairs)
    return numbers, names


def centroid_features(labels, mask, k):
    loc = np.argwhere(mask)
    center = loc.mean(0)
    radius = max(float(np.sqrt(np.mean(np.sum((loc - center) ** 2, axis=1)))), 1e-12)
    centers = []
    for h in range(1, k + 1):
        p = np.argwhere(mask & (labels == h))
        centers.append(p.mean(0) if len(p) >= 12 else np.full(3, np.nan))
    values = [float(np.linalg.norm(c - center) / radius) for c in centers]
    values.extend(float(np.linalg.norm(centers[a - 1] - centers[b - 1]) / radius)
                  for a, b in combinations(range(1, k + 1), 2))
    names = ['H%d_CentroidOffsetInvariant' % h for h in range(1, k + 1)]
    names.extend('H%dH%d_CentroidDistanceInvariant' % pair for pair in combinations(range(1, k + 1), 2))
    return values, names


def measurements(raw, images, labels, mask, k, direct=True):
    values, metadata, errors = [], [], []
    texture_names = sorted('original_' + m for m in old.views.METRICS)
    for channel, name in enumerate(('D', 'f')):
        for h in range(1, k + 1):
            region = mask & (labels == h)
            if int(region.sum()) <= 12:
                a, error = np.full(17, np.nan), 'minimumROISize12 requires more than12voxels'
            else:
                a, error = old.texture(images[..., channel], region, direct)
            values.extend(a)
            metadata.extend({'name': 'IVIM__Regional_H%d_%s_%s' % (h, name, s),
                             'modality': 'IVIM', 'block': 'Texture', 'habitat': h} for s in texture_names)
            if error:
                errors.append({'Channel': name, 'Habitat': h, 'Error': error})
        a, names = descriptors(raw[..., channel], labels, mask, k)
        keep = [j for j, s in enumerate(names) if not any(s.endswith('Centroid' + axis) for axis in ('X', 'Y', 'Z'))
                and (channel == 0 or any(s.endswith('_' + q) for q in ('Mean', 'SD', 'P10', 'Median', 'P90', 'IQR')))]
        values.extend(a[j] for j in keep)
        metadata.extend({'name': 'IVIM__Regional_' + name + '_Raw_' + names[j],
                         'modality': 'IVIM', 'block': 'Descriptor', 'habitat': 0} for j in keep)
    a, names = centroid_features(labels, mask, k)
    values.extend(a)
    metadata.extend({'name': 'IVIM__Regional_' + name, 'modality': 'IVIM',
                     'block': 'Descriptor', 'habitat': 0} for name in names)
    return np.asarray(values), metadata, errors


def grouped(name):
    return re.sub(r'H\d+', 'H#', name)


def invariant(values, metadata):
    groups, fractions = {}, []
    for j, m in enumerate(metadata):
        if m['name'].endswith('_TumorFraction'):
            fractions.append(j)
        else:
            groups.setdefault(grouped(m['name']), []).append(j)
    specs = []
    for name, ix in groups.items():
        for stat in ('Mean', 'SD', 'Range'):
            specs.append({'name': name.replace('Regional_', 'Invariant_') + '__' + stat,
                'modality': 'IVIM', 'block': metadata[ix[0]]['block'], 'source_indices': ix, 'statistic': stat})
    for stat in ('SD', 'Maximum', 'Entropy'):
        specs.append({'name': grouped(metadata[fractions[0]]['name']).replace('Regional_', 'Invariant_') + '__' + stat,
            'modality': 'IVIM', 'block': 'Descriptor', 'source_indices': fractions, 'statistic': stat})
    assert len(specs) == 156
    meta = [{q: v for q, v in m.items() if q not in ('source_indices', 'statistic')} for m in specs]
    assert meta == feature_metadata()
    return old.views.aggregate(values[None], specs)[0]


@lru_cache(None)
def feature_metadata():
    return [{q: v for q, v in m.items() if q not in ('source_indices', 'statistic')}
            for m in old.aggregation()]


def extract(index, aid, profile):
    started = time.time()
    raw, images, norms = old.prepare(profile)
    values = {s['ImageKey']: {} for s in specifications()}
    audits, errors, hashes = [], [], {}
    for norm in NORMS:
        signal = profile['z'] if norm == 'RobustZ' else images[norm]
        for kind, key in KINDS:
            mask = np.asarray(profile[key], bool) & profile['valid']
            x, sample = old.design(signal, mask)
            for combo, ix in DRIVERS:
                for k in KS:
                    imagekey = norm + '__' + combo + '__K' + str(k)
                    labels = partition(x, sample, mask, list(ix), k)
                    assert np.array_equal(labels > 0, mask)
                    a, meta, failed = measurements(raw, images[norm], labels, mask, k)
                    values[imagekey][kind] = invariant(a, meta)
                    sizes = [int((labels == h).sum()) for h in range(1, k + 1)]
                    ch = None
                    selected_labels = labels[mask][sample]
                    if kind == 'base' and len(np.unique(selected_labels)) == k:
                        ch = float(calinski_harabasz_score(x[sample][:, ix], selected_labels))
                    audits.append({'AnalysisID': aid, 'ImageKey': imagekey, 'Normalization': norm, 'ComboID': combo,
                        'K': k, 'Variant': kind, 'MaskVoxels': int(mask.sum()), 'HabitatVoxels': sizes,
                        'SmallHabitats': sum(n <= 12 for n in sizes), 'SampleCH': ch,
                        'FiniteFeatures': int(np.isfinite(values[imagekey][kind]).sum())})
                    errors.extend({'AnalysisID': aid, 'ImageKey': imagekey, 'Variant': kind, **e} for e in failed)
                    hashes[(imagekey, kind)] = hashlib.sha256(labels.tobytes()).hexdigest()
    return {'index': int(index), 'AnalysisID': aid, 'values': values, 'partition_audit': audits,
            'texture_errors': errors, 'normalization_audit': norms, 'partition_hashes': hashes,
            'elapsed_seconds': time.time() - started}
