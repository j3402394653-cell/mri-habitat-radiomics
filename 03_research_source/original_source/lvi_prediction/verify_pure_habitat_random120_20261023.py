"""Independent ranking AUCs and explicit tree/kernel/linear inference checks."""
import argparse
import json
import joblib
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_curve
from threadpoolctl import threadpool_limits
import run_pure_habitat_random120_20261023 as run
from verify_ivim_df63_20261020 import auc, draws


def manual_x(f, x):
    method = f['config']['method']; model = f['classifier']
    if model is None:
        return np.full(len(x), f['prevalence'])
    if method in ('Ridge', 'PCA', 'PLS'):
        return expit(x@f['coef_gaussian']+f['intercept_gaussian'])
    if method == 'RBF':
        sv = model.support_vectors_
        dd = np.maximum(np.sum(x*x, axis=1)[:, None]+np.sum(sv*sv, axis=1)[None]-2*x@sv.T, 0.)
        return expit(np.exp(-model._gamma*dd)@model.dual_coef_[0]+model.intercept_[0])
    if method.startswith('ExtraTrees'):
        x = x.astype(np.float32)
        result = np.zeros(len(x))
        for estimator in model.estimators_:
            tree = estimator.tree_
            for i, row in enumerate(x):
                node = 0
                while tree.children_left[node] != -1:
                    node = tree.children_left[node] if row[tree.feature[node]] <= tree.threshold[node] else tree.children_right[node]
                counts = tree.value[node, 0]
                result[i] += counts[1]/counts.sum()
        return result/len(model.estimators_)
    score = np.full(len(x), float(model._baseline_prediction.ravel()[0]))
    for iteration in model._predictors:
        for predictor in iteration:
            nodes = predictor.nodes
            assert not nodes['is_categorical'].any()
            for i, row in enumerate(x):
                j = 0
                while not nodes[j]['is_leaf']:
                    node = nodes[j]
                    v = row[node['feature_idx']]
                    left = bool(node['missing_go_to_left']) if np.isnan(v) else v <= node['num_threshold']
                    j = node['left'] if left else node['right']
                score[i] += nodes[j]['value']
    return expit(score)


def manual(f, tables, indices):
    return manual_x(f, run.method.transformed(f, tables, indices))


def best(grid, family):
    return sorted([r for r in grid if r['Family'] == family], key=lambda r: (-int(r['AvailableAllFolds']), -r['Objective'],
        r['MeanEffectiveDimensions'], r['MeanFeatureCount'], r['C'], r['candidate_id']))[0]


def same(f, g, evaluation=True):
    assert f['config'] == g['config'] and f['names'] == g['names'] and f['indices'] == g['indices']
    for field in ['median', 'train_probability']+(['evaluation_probability'] if evaluation else []):
        np.testing.assert_allclose(f[field], g[field], atol=1e-10, rtol=0)
    if f['quantile'] is not None:
        np.testing.assert_array_equal(f['quantile'].quantiles_, g['quantile'].quantiles_)
    if 'coef_gaussian' in f:
        np.testing.assert_allclose(f['coef_gaussian'], g['coef_gaussian'], atol=1e-10, rtol=0)


def smoke():
    _, y, tr, te = run.previous.previous.common.inputs('Fixed118')
    tables = run.data(tr)
    reference = joblib.load(run.previous.OUT/'training_features.joblib')
    for key, d in tables.items():
        assert d['metadata'] == reference[key]['metadata']
        for kind in run.previous.feature.KINDS:
            np.testing.assert_array_equal(d[kind], reference[key][kind])
    a, b = run.previous.previous.local.study.folds('Fixed118', 'outer')[0]
    for key, d in tables.items():
        old = run.previous.prepare(d, a, b)
        new = run.method.prepare(d, a, b, {'mask_mode': 'Base', 'icc_min': .75})
        assert old['indices'] == new['indices'] and old['names'] == new['names']
        np.testing.assert_array_equal(old['xa'], new['xa']); np.testing.assert_array_equal(old['xb'], new['xb'])
        rr = run.previous.rows(d, a)
        expected = np.median(np.stack([d[kind][rr] for kind in run.previous.feature.KINDS]), axis=0)
        actual = run.method.values(d, a, 'ContourMedian')
        finite = np.isfinite(expected)
        np.testing.assert_allclose(expected[finite], actual[finite], atol=1e-12)
    methods = ('Ridge', 'PCA', 'RBF', 'ExtraTrees2', 'ExtraTrees4', 'HistGB3')
    for mode in ('Base', 'ContourMedian'):
        for method in methods:
            c = next(c for c in run.method.options() if c['method'] == method and c['mask_mode'] == mode)
            f = run.method.fit(tables, a, b, y, c)
            np.testing.assert_allclose(manual(f, tables, a), f['train_probability'], atol=1e-9, rtol=0)
            np.testing.assert_allclose(manual(f, tables, b), f['evaluation_probability'], atol=1e-9, rtol=0)
            empty = run.method.fit(tables, a, [], y, c)
            assert not len(empty['evaluation_probability'])
    print('Smoke passed:all6 feature views;Reference preprocessing parity;12 linear/kernel/tree checks;empty batches;no newcohort AUC inspected', flush=True)


def training():
    run.frozen(); assert not (run.OUT/'test_started.json').exists()
    sampling = json.loads((run.OUT/'sampling_record.json').read_text(encoding='utf-8'))
    sample, trnew, tenew, omitted = run.draw()
    np.testing.assert_array_equal(sample, sampling['sample']); np.testing.assert_array_equal(omitted, sampling['omitted'])
    assert sampling['number_of_random_draws'] == 1
    _, _, tr, te = run.inputs('Random120')
    np.testing.assert_array_equal(tr, trnew); np.testing.assert_array_equal(te, tenew)
    records, objectives, refits, outer_refits, maxerr = 0, 0, 0, 0, 0.
    for group in run.GROUPS:
        cohort, y, tr, te = run.inputs(group)
        assert (y[te] == -1).all()
        tables = joblib.load(run.OUT/group/'training_features.joblib')
        assert all(set(d['indices']) == set(tr) for d in tables.values())
        independently = run.data(tr)
        for key, d in tables.items():
            assert d['metadata'] == independently[key]['metadata']
            assert not any('Clinical' in m['name'] or 'Whole' in m['name'] or 'Dstar' in m['name'] for m in d['metadata'])
            for kind in run.previous.feature.KINDS:
                np.testing.assert_array_equal(d[kind], independently[key][kind])
        result = joblib.load(run.OUT/group/'models/models.joblib')
        for phase in run.PHASES:
            cp = joblib.load(run.OUT/group/'phases'/(phase+'.joblib'))
            ff = {(r, k): (a, b) for r, k, a, b in run.folds(group, phase)}
            allowed = tr if phase == 'final' else run.folds(group)[int(phase[-1])-1][0]
            for repeat in (1, 2, 3):
                val = []
                for fold in range(1, 6):
                    a, b = ff[repeat, fold]
                    assert not set(a) & set(b) and set(a) | set(b) == set(allowed)
                    assert not (set(a) | set(b)) & set(te)
                    val.extend(b)
                assert sorted(val) == sorted(allowed)
            audits = {(r['CandidateKey'], r['Repeat'], r['InnerFold']): r for r in cp['audit']}
            for c in run.candidates(group):
                values = []
                for row in cp['cv'][c['CandidateKey']]:
                    a, b = ff[row['Repeat'], row['Fold']]
                    np.testing.assert_array_equal(row['train'], a); np.testing.assert_array_equal(row['validation'], b)
                    fit_auc, val_auc = auc(y[a], row['fit_score']), auc(y[b], row['validation_score'])
                    audit = audits[c['CandidateKey'], row['Repeat'], row['Fold']]
                    np.testing.assert_allclose([fit_auc, val_auc], [audit['FitAUC'], audit['ValidationAUC']], atol=1e-12)
                    values.append((fit_auc, val_auc)); records += 1
                values = np.asarray(values)
                mu, sd, fm = values[:, 1].mean(), values[:, 1].std(ddof=1), values[:, 0].mean()
                grid = next(r for r in cp['grid'] if r['CandidateKey'] == c['CandidateKey'])
                objective = mu-.05*sd if c['Family'] == 'Improved' else mu-.1*sd-.3*max(fm-mu-.1, 0)
                if not grid['AvailableAllFolds']:
                    objective = -1.
                np.testing.assert_allclose([grid['MeanValidationAUC'], grid['SD'], grid['Objective']], [mu, sd, objective], atol=1e-12)
                objectives += 1
            for family in run.families(group):
                winner = best(cp['grid'], family); assert winner == run.choose(cp['grid'], family)
                c = run.config(winner, group)
                for repeat, fold, a, b in run.folds(group, phase):
                    f = run.method.fit(tables, a, b, y, c)
                    row = next(r for r in cp['cv'][c['CandidateKey']] if r['Repeat'] == repeat and r['Fold'] == fold)
                    np.testing.assert_allclose(f['train_probability'], row['fit_score'], atol=1e-10, rtol=0)
                    np.testing.assert_allclose(f['evaluation_probability'], row['validation_score'], atol=1e-10, rtol=0)
                    maxerr = max(maxerr, float(np.max(np.abs(manual(f, tables, b)-row['validation_score']))))
                    refits += 1
            print('Verified '+group+' '+phase+':all candidate AUCs/objectives andselected15fold fits', flush=True)
        for k, outer in enumerate(result['outer'], 1):
            a, b = run.folds(group)[k-1]
            for family, f in outer['models'].items():
                cp = joblib.load(run.OUT/group/'phases'/('outer%d.joblib' % k))
                assert f['config'] == run.config(best(cp['grid'], family), group)
                same(f, run.method.fit(tables, a, b, y, f['config']))
                maxerr = max(maxerr, float(np.max(np.abs(manual(f, tables, b)-f['evaluation_probability']))))
                outer_refits += 1
        for family, f in result['final'].items():
            same(f, run.method.fit(tables, tr, [], y, f['config']))
            cp = joblib.load(run.OUT/group/'phases/final.joblib')
            oo = np.full((138, 3), np.nan)
            for row in cp['cv'][f['config']['CandidateKey']]:
                oo[row['validation'], row['Repeat']-1] = row['validation_score']
            np.testing.assert_array_equal(oo, f['repeated_oof']); assert np.isnan(oo[te]).all()
            fp, tp, thresholds = roc_curve(y[tr], oo[tr].mean(1), drop_intermediate=False)
            assert f['threshold'] == thresholds[np.argmax(tp-fp)]
        a, b = run.folds(group)[0]; f = result['outer'][0]['models']['Improved']
        key = f['config']['image_key']; mutated = dict(tables); d = tables[key]
        mutated[key] = {**d, **{kind: d[kind].copy() for kind in run.previous.feature.KINDS}}
        for kind in run.previous.feature.KINDS:
            mutated[key][kind][run.previous.rows(d, b)] = 999999.
        changed_y = y.copy(); changed_y[b] = 1-changed_y[b]
        same(f, run.method.fit(mutated, a, b, changed_y, f['config']), evaluation=False)
    assert maxerr < 1e-8
    run.save(run.OUT/'training_verification.json', {'passed': True, 'new_test_calls': 0,
        'one_fixed_random120_draw_reproduced': True, 'inner_AUC_records': records, 'objective_records': objectives,
        'selected_inner_refits': refits, 'outer_refits': outer_refits, 'final_refits': 3,
        'manual_linear_kernel_forest_boosting_error': maxerr, 'training_only_feature_and_label_isolation_checked': True,
        'repeated_OOF_thresholds_replayed': True, 'Dstar_clinical_whole_used': False, 'prior_files_unchanged': run.protect()})
    print('All training checks passed;no test scores computed', flush=True)


def inference():
    run.frozen()
    done = json.loads((run.OUT/'test_complete.json').read_text(encoding='utf-8'))
    assert done['passed'] and done['model_calls'] == 2 and done['post_test_fits'] == 0
    cohort, _, tr, te = run.inputs('Random120')
    result = joblib.load(run.OUT/'Random120/models/models.joblib')
    tables = joblib.load(run.OUT/'Random120/test_features.joblib')
    ref = run.data(te)
    for key, d in tables.items():
        for kind in run.previous.feature.KINDS:
            np.testing.assert_array_equal(d[kind], ref[key][kind])
    pp = run.read('test_predictions'); mm = run.read('test_metrics').set_index('Family')
    scores = {}; truth = cohort.Outcome.to_numpy(int)[te]
    maxerr = 0.
    for family, f in result['final'].items():
        frame = pp[pp.Family == family].set_index('AnalysisID').loc[cohort.AnalysisID.iloc[te]]
        exact = frame.Probability.to_numpy()
        maxerr = max(maxerr, float(np.max(np.abs(exact-manual(f, tables, te)))))
        np.testing.assert_allclose(auc(truth, exact), mm.loc[family, 'AUC'], atol=1e-12)
        prediction = exact >= f['threshold']
        for name, value in [('TP', ((truth == 1) & prediction).sum()), ('FN', ((truth == 1) & ~prediction).sum()),
                            ('TN', ((truth == 0) & ~prediction).sum()), ('FP', ((truth == 0) & prediction).sum())]:
            assert mm.loc[family, name] == value
        scores[family] = exact
    samples = draws(truth, scores, 2026102393)
    for family in scores:
        np.testing.assert_allclose(np.quantile(samples[family], [.025, .975]), mm.loc[family, ['AUC_CI95Low', 'AUC_CI95High']].to_numpy(float), atol=1e-12)
    delta = run.read('paired_test_difference').iloc[0]
    np.testing.assert_allclose(np.quantile(samples['Improved']-samples['Reference'], [.025, .975]), [delta.CI95Low, delta.CI95High], atol=1e-12)
    assert maxerr < 1e-8
    run.save(run.OUT/'inference_verification.json', {'passed': True, 'test_scores_manually_replayed': 72,
        'model_calls': 2, 'test_stages': 1, 'post_test_fits': 0, 'Original118_new_test_calls': 0,
        'manual_score_error': maxerr, 'bootstrap5000_andpaired_difference_replayed': True, 'prior_files_unchanged': run.protect()})
    print('Two frozen Random120 test models verified;no alternatives rescored', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=('smoke', 'training', 'inference'))
    with threadpool_limits(1):
        globals()[p.parse_args().action]()
