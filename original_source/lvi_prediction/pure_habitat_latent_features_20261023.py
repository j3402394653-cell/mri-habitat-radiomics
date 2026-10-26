"""Outcome-blind K3 regional contrast bases and audit of previous feature selection."""
from itertools import combinations
import re
import numpy as np

VIEWS = ('OrderedCompact', 'MeanContrast', 'ContrastOnly')
KINDS = ('base', 'erode', 'dilate')


def specifications(metadata, view):
    if view == 'OrderedCompact':
        return [{**m, 'source_indices': [j], 'weights': [1.]} for j, m in enumerate(metadata)]
    groups = {}
    for j, m in enumerate(metadata):
        key = re.sub(r'H[123]H[123]', 'H#H#', m['name'])
        key = re.sub(r'H[123](?!\d)', 'H#', key)
        groups.setdefault(key, []).append(j)
    assert len(metadata) == 321 and len(groups) == 107 and all(len(v) == 3 for v in groups.values())
    result = []
    for name, indices in groups.items():
        if 'H#H#' in name:
            result.extend({**metadata[j], 'source_indices': [j], 'weights': [1.]} for j in indices)
            continue
        indices = sorted(indices, key=lambda j: int(re.search(r'H([123])(?!\d)', metadata[j]['name']).group(1)))
        assert [int(re.search(r'H([123])(?!\d)', metadata[j]['name']).group(1)) for j in indices] == [1, 2, 3]
        stem = name.replace('Regional_', 'RegionalContrast_').replace('H#', 'H123')
        choices = [('H3minusH1', [-1., 0., 1.]), ('MiddleMinusExtremes', [-.5, 1., -.5])]
        if view == 'MeanContrast':
            choices.insert(0, ('RegionalMean', [1/3, 1/3, 1/3]))
        for suffix, weights in choices:
            result.append({**metadata[indices[0]], 'name': stem+'__'+suffix,
                'source_indices': indices, 'weights': weights})
    assert len(result) == (321 if view == 'MeanContrast' else 220)
    return result


def derive(d, view):
    ss = specifications(d['metadata'], view)
    result = {'indices': d['indices'].copy(), 'metadata': [{k: v for k, v in s.items() if k not in ('source_indices', 'weights')} for s in ss]}
    for kind in KINDS:
        columns = []
        for s in ss:
            x = d[kind][:, s['source_indices']]
            # A contrast is missing if any of its three constituent regions is missing.
            columns.append(np.where(np.isfinite(x).all(1), np.where(np.isfinite(x), x, 0.)@np.asarray(s['weights']), np.nan))
        result[kind] = np.column_stack(columns)
    return result


def build(original):
    return {norm+'__'+view: derive(d, view) for norm, d in original.items() for view in VIEWS}


def previous_selection_audit(result):
    rows = []
    local_keys = [key for key in result['outer'][0]['refit']['members'] if key.startswith('L')]
    for key in local_keys:
        fits = [outer['refit']['members'][key]['fit'] for outer in result['outer']]
        for a, b in combinations(range(5), 2):
            f, g = fits[a], fits[b]
            sf, sg = set(f['names']), set(g['names'])
            bf, bg = dict(zip(f['names'], f['model'].coef_[0])), dict(zip(g['names'], g['model'].coef_[0]))
            common = sf & sg
            rows.append({'MemberID': key, 'FoldA': a+1, 'FoldB': b+1,
                'FeatureCountA': len(sf), 'FeatureCountB': len(sg), 'SharedFeatures': len(common),
                'Jaccard': len(common)/len(sf | sg), 'SharedCoefficientSignFlips': int(sum(bf[n]*bg[n] < 0 for n in common))})
    return rows
