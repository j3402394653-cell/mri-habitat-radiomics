"""Fold-fitted common quantitative SLIC habitats; no outcomes or clinical data."""
import numpy as np
from scipy.ndimage import distance_transform_edt
from sklearn.cluster import KMeans

MODALITIES = ('T2', 'ADC', 'IVIM')
KINDS = ('base', 'erode', 'dilate')
SEED = 64212


def compact_patient(patient, index):
    result = {'index': int(index), 'modalities': {}}
    for mod in MODALITIES:
        p = patient[mod]
        assert np.allclose(p['spacing'], (1, 1, 1))
        assert all(v.get('clipping') is None for v in p['normalization'])
        units = np.asarray(p['superlabels'], int)
        n = len(p['super_x'])
        x = np.array(p['super_x'], dtype=float, copy=True)
        z = np.asarray(p['z'], float)
        if mod != 'T2':
            for c, norm in enumerate(p['normalization']):
                x[:, 3*c:3*c+2] = x[:, 3*c:3*c+2]*norm['std'] + norm['mean']
                x[:, 3*c+2] *= norm['std']
                z[..., c] = z[..., c]*norm['std'] + norm['mean']
                error = abs(z[..., c][p['roi']].mean()-norm['mean']) / max(abs(norm['mean']), 1e-12)
                assert error < 1e-5
        d = {'x': x, 'tumor_units': np.asarray(p['tumor_units'], int), 'masks': {}}
        # Equal number of tumor supervoxels from every current training patient.
        # A case-specific fixed ordering is independent of outcomes and CV membership.
        rng = np.random.default_rng(SEED + int(index)*1009)
        d['sample_order'] = rng.permutation(d['tumor_units'])
        assert set(d['tumor_units']) == set(np.unique(units[p['roi']]))
        for kind, maskkey in zip(KINDS, ('roi', 'erode', 'dilate')):
            mask = np.asarray(p[maskkey], bool) & p['valid']
            uid = units[mask]
            assert (uid > 0).all() and mask.sum() >= 20
            count = np.bincount(uid, minlength=n).astype(float)
            shell = mask & (distance_transform_edt(mask, sampling=p['spacing']) <= 2.)
            shell_count = np.bincount(units[shell], minlength=n).astype(float)
            sums = np.column_stack([np.bincount(uid, weights=z[..., c][mask], minlength=n) for c in range(z.shape[-1])])
            squares = np.column_stack([np.bincount(uid, weights=z[..., c][mask]**2, minlength=n) for c in range(z.shape[-1])])
            cross = np.bincount(uid, weights=np.prod(z[mask], axis=1), minlength=n) if mod == 'IVIM' else None
            edges = np.zeros(n*n, np.int64)
            for axis in range(3):
                a, b = [slice(None)]*3, [slice(None)]*3
                a[axis], b[axis] = slice(None, -1), slice(1, None)
                a, b = tuple(a), tuple(b)
                valid = mask[a] & mask[b]
                edges += np.bincount(units[a][valid]*n+units[b][valid], minlength=n*n)
            edge_indices = np.flatnonzero(edges)
            d['masks'][kind] = {'count': count, 'shell_count': shell_count, 'sum': sums, 'square': squares,
                'cross': cross, 'edge_a': edge_indices//n, 'edge_b': edge_indices % n,
                'edge_weight': edges[edge_indices]}
        result['modalities'][mod] = d
    return result


def scale_fit(x, normalization):
    if normalization == 'PercentileZ':
        low, high = np.quantile(x, [.01, .99], axis=0)
        clipped = np.clip(x, low, high)
        center, scale = clipped.mean(0), clipped.std(0)
    elif normalization == 'RobustZ':
        low, high = np.full(x.shape[1], -np.inf), np.full(x.shape[1], np.inf)
        center = np.median(x, axis=0)
        scale = np.subtract(*np.quantile(x, [.75, .25], axis=0))/1.348979500392164
    else:
        raise ValueError(normalization)
    scale = np.where(scale > 1e-12, scale, x.std(0))
    scale = np.where(scale > 1e-12, scale, 1.)
    return {'low': low, 'high': high, 'center': center, 'scale': scale}


def scale_apply(f, x):
    return (np.clip(x, f['low'], f['high'])-f['center'])/f['scale']


def fit_partitions(patients, train, normalization, k):
    train = np.asarray(train, int)
    result = {'train': train.copy(), 'normalization': normalization, 'K': k, 'modalities': {}}
    for mod in MODALITIES:
        m = min(64, min(len(patients[int(i)]['modalities'][mod]['tumor_units']) for i in train))
        assert m >= k
        samples = np.concatenate([patients[int(i)]['modalities'][mod]['x'][patients[int(i)]['modalities'][mod]['sample_order'][:m]] for i in train])
        f = scale_fit(samples, normalization)
        model = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=SEED).fit(scale_apply(f, samples))
        centers = model.cluster_centers_*f['scale']+f['center']
        # Common identities use ascending T2/ADC intensity, or D followed by f.
        order = np.lexsort((centers[:, 3], centers[:, 0])) if mod == 'IVIM' else np.argsort(centers[:, 0], kind='stable')
        remap = np.empty(k, int); remap[order] = np.arange(k)
        result['modalities'][mod] = {**f, 'model': model, 'remap': remap, 'samples_per_patient': m,
            'training_sample_count': len(samples), 'centers_original': centers[order]}
    return result


def assigned(d, f):
    return f['remap'][f['model'].predict(scale_apply(f, d['x']))]


def measure(d, labels, mod, k, kind):
    m = d['masks'][kind]
    counts, shell = m['count'], m['shell_count']
    core = counts-shell
    n, ns, nc = counts.sum(), shell.sum(), core.sum()
    values, metadata, means = [], [], {}
    channels = ('T2z',) if mod == 'T2' else ('ADC',) if mod == 'ADC' else ('D', 'f')

    def add(value, name, panel):
        values.append(float(value)); metadata.append({'name': mod+'__Common_'+name, 'modality': mod, 'panel': panel})

    fractions = []
    for h in range(k):
        chosen = labels == h
        nh = counts[chosen].sum()
        frac = nh/n; fractions.append(frac)
        add(frac, 'H%d_VolumeFraction' % (h+1), 'Spatial')
        sf = shell[chosen].sum()/ns if ns else np.nan
        cf = core[chosen].sum()/nc if nc else np.nan
        add(sf, 'H%d_PeripheryFraction' % (h+1), 'Spatial')
        add(sf-cf, 'H%d_PeripheryCoreDifference' % (h+1), 'Spatial')
        for c, channel in enumerate(channels):
            mean = m['sum'][chosen, c].sum()/nh if nh >= 12 else np.nan
            var = m['square'][chosen, c].sum()/nh-mean**2 if nh >= 12 else np.nan
            sd = np.sqrt(max(var, 0.)) if np.isfinite(var) else np.nan
            local = np.dot(counts[chosen], d['x'][chosen, 3*c+2])/nh if nh >= 12 else np.nan
            means[h, c] = mean
            for value, suffix in ((mean, 'Mean'), (sd, 'SD'), (local, 'MeanLocalSD')):
                add(value, 'H%d_%s_%s' % (h+1, channel, suffix), 'Quantitative')
        if mod == 'IVIM':
            if nh >= 12:
                exy = m['cross'][chosen].sum()/nh
                variances = [max(m['square'][chosen, c].sum()/nh-means[h, c]**2, 0.) for c in (0, 1)]
                den = np.sqrt(np.prod(variances))
                corr = np.clip((exy-means[h, 0]*means[h, 1])/den, -1., 1.) if den > 1e-15 else 0.
            else:
                corr = np.nan
            add(corr, 'H%d_DfCorrelation' % (h+1), 'Quantitative')
    entropy = -sum(p*np.log(p) for p in fractions if p > 0)/np.log(k)
    add(entropy, 'CompositionEntropy', 'Spatial')
    edge_h = labels[m['edge_a']]*k+labels[m['edge_b']]
    contacts = np.bincount(edge_h, weights=m['edge_weight'], minlength=k*k).reshape(k, k)
    denominator = contacts.sum()
    add((denominator-np.trace(contacts))/denominator if denominator else np.nan, 'DifferentHabitatContact', 'Spatial')
    for a in range(k):
        for b in range(a+1, k):
            add((contacts[a, b]+contacts[b, a])/denominator if denominator else np.nan,
                'H%dH%d_ContactFraction' % (a+1, b+1), 'Spatial')
            for c, channel in enumerate(channels):
                ma, mb = means[a, c], means[b, c]
                add(mb-ma, 'H%dH%d_%s_MeanDifference' % (a+1, b+1, channel), 'Quantitative')
                den = abs(ma)+abs(mb)
                add((mb-ma)/den if den > 1e-15 else 0. if np.isfinite(den) else np.nan,
                    'H%dH%d_%s_RelativeContrast' % (a+1, b+1, channel), 'Quantitative')
    return values, metadata


def tables(patients, partitions, indices):
    rows = {kind: [] for kind in KINDS}
    metadata = None
    for i in indices:
        row, names = {kind: [] for kind in KINDS}, []
        for mod in MODALITIES:
            d = patients[int(i)]['modalities'][mod]
            labels = assigned(d, partitions['modalities'][mod])
            for kind in KINDS:
                values, mm = measure(d, labels, mod, partitions['K'], kind)
                row[kind].extend(values)
                if kind == 'base':
                    names.extend(mm)
        if metadata is None:
            metadata = names
        assert names == metadata
        for kind in KINDS:
            rows[kind].append(row[kind])
    return {'indices': np.asarray(indices, int), 'metadata': metadata,
            **{kind: np.asarray(x, float) for kind, x in rows.items()}}
