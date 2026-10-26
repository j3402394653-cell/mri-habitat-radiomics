# -*- coding: utf-8 -*-
"""Checkpointed6-process feature extraction with no outcome/model fitting."""
import os
os.environ.update({'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'LOKY_MAX_CPU_COUNT': '8'})
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import time
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
import ivim_df63_study_20261020 as study
import ivim_df63_features_20261020 as feature
import run_random120_retained_methods_20261010 as retained

PROFILES = None


def initialize():
    global PROFILES
    feature.configure()
    threadpool_limits(1)
    PROFILES = retained.load_profiles()


def one(index, aid):
    result = feature.extract_patient(index, aid, PROFILES[index]['IVIM'])
    path = study.patient_path(index)
    temporary = path.with_suffix('.pending')
    joblib.dump(result, temporary, compress=3)
    temporary.replace(path)
    return index, result['elapsed_seconds'], len(result['texture_errors']), study.digest(path)


def smoke():
    from verify_regional_summary77_two_tests_20261014 import reference_features
    profiles = retained.load_profiles()
    errors, details = [], []
    for index in (0, 7):
        p = profiles[index]['IVIM']
        result = feature.extract_patient(index, 'smoke', p, [feature.combinations63()[-1]])
        reference = reference_features([index])
        for norm in feature.NORMALIZATIONS:
            d = reference[(norm, 'Invariant')]
            keep = [j for j, m in enumerate(d['metadata']) if m['modality'] == 'IVIM']
            for kind, _ in feature.KINDS:
                a, b = result['values'][(norm, 'C63')][kind], d[kind][0, keep]
                np.testing.assert_allclose(a, b, atol=1e-9, rtol=1e-9, equal_nan=True)
                details.append(float(np.nanmax(np.abs(a-b))))
        raw, images, _ = feature.prepare(p)
        mask = np.asarray(p['roi'], bool)&p['valid']
        x, sample = feature.design(p['z'], mask)
        for indices in ([0], [3, 5], list(range(6))):
            labels = feature.partition(x, sample, mask, indices)
            for norm in feature.NORMALIZATIONS:
                direct, _ = feature.measurements(raw, images[norm], labels, mask, True)
                full, _ = feature.measurements(raw, images[norm], labels, mask, False)
                np.testing.assert_allclose(direct, full, atol=1e-10, rtol=1e-10, equal_nan=True)
            # Excluded dimensions cannot alter assignment.
            excluded = sorted(set(range(6))-set(indices))
            changed = x.copy();changed[:, excluded] = 1e6
            np.testing.assert_array_equal(feature.partition(changed, sample, mask, indices), labels)
        print('Feature smoke case%d exact6D baseline,directPyRadiomics,andexcluded-dimension isolation passed'%index, flush=True)
    return {'passed': True, 'patients': [0, 7], 'all6_matches_historical156_invariant_features': True,
        'max_baseline_difference': max(details), 'direct17_metric_calls_match_full_extractor': True,
        'excluded_dimensions_cannot_change_assignment': True, 'outcome_used': False}


def extract(stage):
    study.protect()
    if stage == 'test':
        assert json.loads((study.OUT/'training_verification.json').read_text(encoding='utf-8'))['passed']
    indices = study.union_indices('train' if stage == 'train' else 'sample')
    cohort = study.source.cohort()
    existing = [i for i in indices if study.patient_path(i).exists()]
    for i in existing:
        row = joblib.load(study.patient_path(i))
        assert row['index'] == int(i) and len(row['values']) == 126
    todo = [int(i) for i in indices if i not in existing]
    print('Extraction '+stage+': needed%d already%d remaining%d'%(len(indices), len(existing), len(todo)), flush=True)
    started, audit = time.time(), []
    with ProcessPoolExecutor(max_workers=study.WORKERS, initializer=initialize) as pool:
        futures = {pool.submit(one, i, cohort.AnalysisID.iloc[i]): i for i in todo}
        for completed, future in enumerate(as_completed(futures), 1):
            i, seconds, errors, sha = future.result()
            audit.append({'index': i, 'seconds': seconds, 'errors': errors, 'sha256': sha})
            print('Extraction '+stage+' %d/%d case%d %.1fs errors%d elapsed%.0fs'%(completed, len(todo), i, seconds, errors, time.time()-started), flush=True)
    study.save(study.OUT/('extraction_'+stage+'_complete.json'), {'passed': True, 'required_N': len(indices),
        'newly_extracted_N': len(todo), 'hashes': {str(int(i)): study.digest(study.patient_path(i)) for i in indices},
        'worker_audit': audit, 'elapsed_seconds': time.time()-started, 'outcome_used': False, 'models_fitted': 0})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['define', 'train', 'test'], required=True)
    args = parser.parse_args()
    with threadpool_limits(1):
        if args.phase == 'define':
            check = smoke()
            study.define()
            study.verify_definition()
            study.save(study.OUT/'feature_smoke_verification.json', check)
        else:
            extract(args.phase)
