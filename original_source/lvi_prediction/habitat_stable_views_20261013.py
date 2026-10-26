"""Outcome-independent compact and habitat-label-invariant K3 representations."""
import re
import numpy as np

METRICS = (
    'firstorder_10Percentile', 'firstorder_90Percentile', 'firstorder_Median',
    'firstorder_InterquartileRange', 'firstorder_Entropy', 'firstorder_Uniformity',
    'firstorder_Skewness', 'firstorder_Kurtosis', 'glcm_Contrast', 'glcm_JointEntropy',
    'glcm_JointEnergy', 'glcm_Correlation', 'glcm_Idmn',
    'glrlm_RunLengthNonUniformityNormalized', 'glszm_SmallAreaEmphasis',
    'glszm_ZoneVariance', 'gldm_DependenceNonUniformityNormalized')
KINDS = ('base', 'erode', 'dilate')
VIEWS = ('Original', 'Compact', 'Invariant')


def grouped_name(name):
    name = re.sub(r'H[123]H[123]', 'H#H#', name)
    return re.sub(r'H[123](?!\d)', 'H#', name)


def specifications(metadata):
    compact = [i for i, m in enumerate(metadata) if
               (m['block'] == 'Descriptor' and not m['name'].endswith('_TumorFraction')) or
               any(m['name'].endswith('_original_'+suffix) for suffix in METRICS)]
    groups = {}
    for i in compact:
        groups.setdefault(grouped_name(metadata[i]['name']), []).append(i)
    assert len(compact) == 321 and len(groups) == 107
    assert all(len(ix) == 3 for ix in groups.values())
    aggregation = []
    for name, ix in groups.items():
        for statistic in ('Mean', 'SD', 'Range'):
            aggregation.append({'name': name.replace('Regional_', 'Invariant_')+'__'+statistic,
                                'modality': metadata[ix[0]]['modality'], 'block': metadata[ix[0]]['block'],
                                'source_indices': ix, 'statistic': statistic})
    fractions = {}
    for i, m in enumerate(metadata):
        if m['name'].endswith('_TumorFraction'):
            fractions.setdefault(grouped_name(m['name']), []).append(i)
    assert len(fractions) == 3 and all(len(ix) == 3 for ix in fractions.values())
    for name, ix in fractions.items():
        for statistic in ('SD', 'Maximum', 'Entropy'):
            aggregation.append({'name': name.replace('Regional_', 'Invariant_')+'__'+statistic,
                                'modality': metadata[ix[0]]['modality'], 'block': 'Descriptor',
                                'source_indices': ix, 'statistic': statistic})
    assert len(aggregation) == 330
    return compact, aggregation


def aggregate(x, specifications):
    columns = []
    for s in specifications:
        v = x[:, s['source_indices']]
        valid = np.isfinite(v).all(1)
        v = np.where(np.isfinite(v), v, 0.)
        stat = s['statistic']
        if stat == 'Mean':
            a = v.mean(1)
        elif stat == 'SD':
            a = v.std(1)
        elif stat == 'Range':
            a = v.max(1)-v.min(1)
        elif stat == 'Maximum':
            a = v.max(1)
        elif stat == 'Entropy':
            p = np.maximum(v, 0.)
            p = p/np.maximum(p.sum(1, keepdims=True), 1e-12)
            a = -np.sum(p*np.log(np.maximum(p, 1e-12)), axis=1)
        else:
            raise ValueError(stat)
        columns.append(np.where(valid, a, np.nan))
    return np.column_stack(columns)


def build(data):
    result = {}
    for norm, d in data.items():
        compact, aggregation = specifications(d['metadata'])
        result[(norm, 'Original')] = d
        result[(norm, 'Compact')] = {'indices': d['indices'], 'metadata': [d['metadata'][i] for i in compact],
                                    **{kind: d[kind][:, compact] for kind in KINDS}}
        result[(norm, 'Invariant')] = {'indices': d['indices'],
            'metadata': [{k: v for k, v in s.items() if k not in ('source_indices', 'statistic')} for s in aggregation],
            **{kind: aggregate(d[kind], aggregation) for kind in KINDS}}
    return result
