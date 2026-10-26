"""Independent AUC/manual inference checks and within-training cross-fit isolation."""
import json
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
import run_multiblock_fixed118_20261023 as run
from verify_ivim_df63_20261020 import auc, draws


def standardize(x, s):
    v = np.where(np.isfinite(x), x, s['median'])
    return (np.minimum(np.maximum(v, s['low']), s['high']) - s['center']) / s['scale']


def block_manual(block, d, ix):
    rows = [np.flatnonzero(d['indices'] == i).item() for i in ix]
    x = standardize(d['base'][rows][:, block['indices']], block['scaler'])
    if block['projector'] is not None:
        x = (x - block['projector'].mean_) @ block['projector'].components_.T
    if block['base_model'] is not None:
        x = (x @ block['base_model'].coef_[0] + block['base_model'].intercept_[0])[:, None]
    return x


def manual(f, data, ix):
    if f['model'] is None:
        return np.full(len(ix), f['prevalence'])
    parts = []
    if f['blocks']:
        image = np.column_stack([block_manual(b, data[f['config']['image_key']], ix) for b in f['blocks']])
        if f['meta_scaler'] is not None:
            image = standardize(image, f['meta_scaler'])
        parts.append(image)
    cp = f['clinical']
    if cp['indices']:
        d = data['Clinical']
        rr = [np.flatnonzero(d['indices'] == i).item() for i in ix]
        parts.append(standardize(d['base'][rr][:, cp['indices']], cp['scaler']))
    x = np.column_stack(parts)
    return expit(x @ f['model'].coef_[0] + f['model'].intercept_[0])


def compare(a, b):
    assert a['config'] == b['config'] and a['available'] == b['available']
    assert a['clinical']['indices'] == b['clinical']['indices']
    assert len(a['blocks']) == len(b['blocks'])
    for x, y in zip(a['blocks'], b['blocks']):
        assert x['indices'] == y['indices'] and x['names'] == y['names'] and x['available'] == y['available']
        if x['available']:
            for key in ('median', 'low', 'high', 'center', 'scale'):
                np.testing.assert_array_equal(x['scaler'][key], y['scaler'][key])
            for key in ('projector', 'base_model'):
                if x[key] is not None:
                    attr = 'components_' if key == 'projector' else 'coef_'
                    np.testing.assert_allclose(getattr(x[key], attr), getattr(y[key], attr), atol=1e-12, rtol=0)
    if a['model'] is not None:
        np.testing.assert_allclose(a['model'].coef_, b['model'].coef_, atol=1e-12, rtol=0)
        np.testing.assert_allclose(a['model'].intercept_, b['model'].intercept_, atol=1e-12, rtol=0)
    np.testing.assert_allclose(a['train_probability'], b['train_probability'], atol=1e-12, rtol=0)


def crossfit_check(f, data, a, y):
    if f['config']['architecture'] != 'Late':
        return 0
    assert len(f['crossfit']) == 3
    seen, count = [], 0
    d = data[f['config']['image_key']]
    for entry in f['crossfit']:
        aa, bb = entry['train'], entry['validation']
        assert not set(aa) & set(bb) and set(aa) | set(bb) == set(a)
        altered = y.copy(); altered[bb] = 1 - y[bb]
        for j, block in enumerate(entry['blocks']):
            if not block['available']:
                continue
            run.RANK_CACHE.clear(); run.BLOCK_CACHE.clear()
            rebuilt = run.block_fit(d, aa, altered, block['modality'], f['config']['quality'], 'Late', 3)
            assert rebuilt['indices'] == block['indices']
            np.testing.assert_allclose(rebuilt['base_model'].coef_, block['base_model'].coef_, atol=1e-12, rtol=0)
            np.testing.assert_allclose(block_manual(block, d, bb)[:, 0], entry['logits'][:, j], atol=1e-10, rtol=0)
            count += 1
        seen.extend(bb)
    assert sorted(seen) == sorted(a)
    return count


def smoke():
    data = run.input_data('train')
    _, y, tr, te = run.old.inputs('Fixed118')
    a, b = run.old.folds('Fixed118', 'outer')[0]
    checked, crossfits = [], 0
    for method, size in [('Balanced', 3), ('PCA', 1), ('Late', 3)]:
        c = next(c for c in run.options() if c['family'] == 'Habitat' and c['image_key'] == 'PercentileZ__C08__K2'
                 and c['architecture'] == method and c['size'] == size and c['quality'] == .75 and c['C'] == .03)
        f = run.fit(data, a, b, y, c)
        np.testing.assert_allclose(manual(f, data, a), f['train_probability'], atol=1e-10, rtol=0)
        np.testing.assert_allclose(manual(f, data, b), f['evaluation_probability'], atol=1e-10, rtol=0)
        altered = y.copy(); altered[b] = 1 - y[b]
        run.RANK_CACHE.clear(); run.BLOCK_CACHE.clear()
        compare(f, run.fit(data, a, b, altered, c))
        crossfits += crossfit_check(f, data, a, y)
        checked.append({'architecture': method, 'available': f['available'], 'image_dimensions': f['image_dimensions']})
    assert len(tr) == 82 and len(te) == 36 and (y[te] == -1).all()
    result = {'passed': True, 'methods': checked, 'manual_pipeline_probabilities_verified': True,
              'heldout_label_isolation': True, 'late_crossfit_base_models_checked': crossfits, 'test_scored': False}
    if run.OUT.exists():
        run.save(run.OUT / 'smoke_verification.json', result)
    print(json.dumps(result), flush=True)


def training():
    run.frozen()
    assert not (run.OUT / 'test_started.json').exists()
    data = joblib.load(run.OUT / 'training_features.joblib')
    reference = run.input_data('train')
    cohort, y, tr, te = run.old.inputs('Fixed118')
    assert set(data) == set(reference)
    for key, d in data.items():
        np.testing.assert_array_equal(d['indices'], tr)
        assert d['metadata'] == reference[key]['metadata']
        for kind in ('base',) if key == 'Clinical' else run.previous.stable.views.KINDS:
            np.testing.assert_array_equal(d[kind], reference[key][kind])
    counts, records = 0, 0
    opts = run.options()
    for phase in ['outer' + str(i) for i in range(1, 6)] + ['final']:
        stored = joblib.load(run.OUT / 'phases' / (phase + '.joblib'))
        folds = run.old.folds('Fixed118', 'final') if phase == 'final' else run.old.folds('Fixed118', 'inner', int(phase[-1]))
        audit = pd.DataFrame(stored['audit']).set_index(['candidate_id', 'InnerFold'])
        assert len(stored['grid']) == 1266 and len(audit) == 6330 and audit.index.is_unique
        for r in stored['grid']:
            cid = r['candidate_id']; rr = audit.loc[cid]
            assert run.config(r) == opts[cid]
            for fold, ((a, b), pp) in enumerate(zip(folds, stored['cv'][cid]), 1):
                np.testing.assert_array_equal(a, pp['train']); np.testing.assert_array_equal(b, pp['validation'])
                assert not (set(a) | set(b)) & set(te)
                np.testing.assert_allclose([rr.loc[fold].FitAUC, rr.loc[fold].ValidationAUC],
                    [auc(y[a], pp['train_probability']), auc(y[b], pp['validation_probability'])], atol=1e-12, rtol=0)
                records += 1
            mean, sd, fit = rr.ValidationAUC.mean(), rr.ValidationAUC.std(ddof=1), rr.FitAUC.mean()
            objective = mean - .10 * sd - .30 * max(fit - mean - .10, 0) if rr.Available.all() else -1.
            np.testing.assert_allclose([mean, sd, fit, objective], [r['MeanValidationAUC'], r['SD'], r['MeanFitAUC'], r['Objective']], atol=1e-12, rtol=0)
            counts += 1
        print('Verified1266 candidates ' + phase, flush=True)
    model = joblib.load(run.OUT / 'models/frozen_models.joblib')
    perturbations, crossfits = 0, 0
    for fold, (a, b) in enumerate(run.old.folds('Fixed118', 'outer'), 1):
        stored = model['outer'][fold - 1]
        grid = joblib.load(run.OUT / 'phases' / ('outer' + str(fold) + '.joblib'))['grid']
        altered = y.copy(); altered[b] = 1 - y[b]
        for family, f in stored['models'].items():
            assert f['config'] == run.config(run.choose(grid, family))
            run.RANK_CACHE.clear(); run.BLOCK_CACHE.clear()
            compare(f, run.fit(data, a, b, altered, f['config']))
            np.testing.assert_allclose(manual(f, data, b), f['evaluation_probability'], atol=1e-10, rtol=0)
            crossfits += crossfit_check(f, data, a, y)
            if fold == 1:
                changed = {key: {**d, **{kind: d[kind].copy() for kind in ('base',) if key == 'Clinical'}} if key == 'Clinical' else
                    {**d, **{kind: d[kind].copy() for kind in run.previous.stable.views.KINDS}} for key, d in data.items()}
                for key, d in changed.items():
                    for kind in ('base',) if key == 'Clinical' else run.previous.stable.views.KINDS:
                        d[kind][run.row(d, b)] = 1e12
                run.RANK_CACHE.clear(); run.BLOCK_CACHE.clear()
                compare(f, run.fit(changed, a, b, altered, f['config'])); perturbations += 1
    final_grid = joblib.load(run.OUT / 'phases/final.joblib')['grid']
    for family, f in model['final'].items():
        assert f['config'] == run.config(run.choose(final_grid, family))
        run.RANK_CACHE.clear(); run.BLOCK_CACHE.clear()
        compare(f, run.fit(data, tr, [], y, f['config']))
        np.testing.assert_allclose(manual(f, data, tr), f['train_probability'], atol=1e-10, rtol=0)
        assert np.isnan(f['training_oof'][te]).all()
        oof = np.full(138, np.nan)
        for a, b in run.old.folds('Fixed118', 'final'):
            run.RANK_CACHE.clear(); run.BLOCK_CACHE.clear()
            rebuilt = run.fit(data, a, b, y, f['config'])
            oof[b] = rebuilt['evaluation_probability']
        np.testing.assert_allclose(oof, f['training_oof'], atol=1e-10, rtol=0, equal_nan=True)
        assert run.previous.stable.old.common.threshold(y[tr], oof[tr]) == f['threshold']
        crossfits += crossfit_check(f, data, tr, y)
    assert counts == 7596 and records == 37980 and perturbations == 5
    run.save(run.OUT / 'training_verification.json', {'passed': True, 'candidate_objectives_checked': counts,
        'inner_prediction_records_rank_verified': records, 'outer_heldout_value_perturbations': perturbations,
        'late_crossfit_base_models_replayed': crossfits, 'selected_models_refit_and_OOF_thresholds_replayed': True,
        'manual_PCA_base_logit_meta_probability_checks': True, 'test_scored': False, 'protected_files': run.protect()})
    print('Verified training,outer isolation,crossfitting andOOF thresholds', flush=True)


def inference():
    run.frozen()
    complete = json.loads((run.OUT / 'test_complete.json').read_text(encoding='utf-8'))
    assert complete['scoring_stages'] == 1 and complete['test_model_calls'] == 5 and complete['model_refits_after_test_start'] == 0
    data = joblib.load(run.OUT / 'test_features.joblib')
    cohort, _, _, te = run.old.inputs('Fixed118')
    truth = cohort.Outcome.to_numpy(int)[te]
    models = joblib.load(run.OUT / 'models/frozen_models.joblib')['final']
    scores = {family: manual(models[family], data, te) for family in run.FAMILIES}
    samples = draws(truth, scores, 2026102361)
    pp, metrics = run.read('test_predictions'), run.read('test_metrics').set_index('ModelKey')
    assert len(pp) == 180 and len(metrics) == 5
    maximum = 0.
    for family, p in scores.items():
        r = pp[pp.ModelKey == family].set_index('AnalysisID').loc[cohort.AnalysisID.iloc[te]]
        np.testing.assert_array_equal(r.Outcome, truth)
        np.testing.assert_allclose(r.Probability, p, atol=1e-10, rtol=0)
        maximum = max(maximum, float(np.max(np.abs(r.Probability - p))))
        m = metrics.loc[family]
        np.testing.assert_allclose([m.AUC, m.AUC_CI95Low, m.AUC_CI95High], [auc(truth, p), *np.quantile(samples[family], [.025, .975])], atol=1e-12, rtol=0)
        positive = p >= models[family]['threshold']
        np.testing.assert_array_equal(m[['TP', 'FN', 'TN', 'FP']].to_numpy(int),
            [((truth == 1) & positive).sum(), ((truth == 1) & ~positive).sum(), ((truth == 0) & ~positive).sum(), ((truth == 0) & positive).sum()])
    run.save(run.OUT / 'inference_verification.json', {'passed': True, 'predictions_verified': 180,
        'manual_probability_max_error': maximum, 'AUC_confusion_bootstrap5000_verified': True,
        'test_model_calls': 5, 'scoring_stages': 1, 'model_refits_after_test_start': 0,
        'threshold_changes': 0, 'protected_files': run.protect()})
    print('Verified180 frozen holdout probabilities andmetrics', flush=True)
