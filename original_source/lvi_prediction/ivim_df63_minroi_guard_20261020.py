# -*- coding: utf-8 -*-
"""Canonical read view reproducing PyRadiomics minimumROISize=12 strict boundary."""
from functools import lru_cache
import numpy as np
import ivim_df63_features_20261020 as feature


@lru_cache(None)
def texture_columns():
    ix = [i for i, s in enumerate(feature.aggregation()) if s['block'] == 'Texture']
    assert len(ix) == 102
    return ix


def canonical(row):
    affected = [r for r in row['partition_audit'] if 12 in [r['H1'], r['H2'], r['H3']]
                and not r.get('Exact12VoxelTextureBoundaryApplied', False)]
    if not affected:
        return row
    values = {key: dict(kinds) for key, kinds in row['values'].items()}
    audit = [dict(r) for r in row['partition_audit']]
    errors = list(row['texture_errors'])
    for record in audit:
        if record.get('Exact12VoxelTextureBoundaryApplied', False):
            continue
        habitats = [h for h in (1, 2, 3) if record['H'+str(h)] == 12]
        if not habitats:
            continue
        key, kind = (record['Normalization'], record['ComboID']), record['Variant']
        a = values[key][kind].copy()
        # Every invariant texture uses all three habitats; one missinghabitat makesall102 textureaggregates missing.
        a[texture_columns()] = np.nan
        values[key][kind] = a
        record['RawFiniteFeatureCount'] = record['FiniteFeatureCount']
        record['FiniteFeatureCount'] = int(np.isfinite(a).sum())
        record['Exact12VoxelTextureBoundaryApplied'] = True
        for channel in ('D', 'f'):
            errors.extend({'AnalysisID': row['AnalysisID'], 'Normalization': record['Normalization'],
                'ComboID': record['ComboID'], 'Variant': kind, 'Channel': channel, 'Habitat': h,
                'Error': 'PyRadiomics minimumROISize12 requires more than12voxels;canonicalreadview'} for h in habitats)
    return {**row, 'values': values, 'partition_audit': audit, 'texture_errors': errors,
        'exact12_boundary_records_corrected': len(affected)}


def smoke():
    rng = np.random.default_rng(64220)
    images = rng.normal(size=(5, 6, 6, 2)).astype(np.float32)
    raw = images*.1+1.
    mask = np.ones((5, 6, 6), bool)
    labels = np.full(mask.shape, 2, np.uint8)
    labels[0, :2, :] = 1
    labels[3:, 3:, :] = 3
    a, _ = feature.measurements(raw, images, labels, mask, True)
    b, _ = feature.measurements(raw, images, labels, mask, False)
    record = {'AnalysisID': 'synthetic12', 'Normalization': 'RobustZ', 'ComboID': 'C63', 'Variant': 'base',
        'MaskVoxels': int(mask.sum()), **{'H'+str(h): int((labels == h).sum()) for h in (1, 2, 3)},
        'FiniteFeatureCount': int(np.isfinite(feature.invariant(a)).sum())}
    row = {'AnalysisID': 'synthetic12', 'values': {('RobustZ', 'C63'): {'base': feature.invariant(a)}},
        'partition_audit': [record], 'texture_errors': []}
    corrected = canonical(row)
    np.testing.assert_allclose(corrected['values'][('RobustZ', 'C63')]['base'], feature.invariant(b),
        atol=1e-10, rtol=1e-10, equal_nan=True)
    assert corrected['partition_audit'][0]['Exact12VoxelTextureBoundaryApplied']
    return {'passed': True, 'exact12_voxel_synthetic_full_extractor_equivalence': True,
        'preserves_descriptors_and_patient': True, 'outcome_used': False}
