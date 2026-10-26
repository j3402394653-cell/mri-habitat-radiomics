# -*- coding: utf-8 -*-
"""Independent score, fold isolation, selection, refit and paired-inference verification."""
import argparse
import json
import joblib
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
import run_random120_five_models_20261023 as run
from verify_ivim_df63_20261020 import auc, draws


def manual(f, tables, indices):
    zi, zc = run.transformed(f, tables, indices)
    p = f['image']
    if p is not None and f['reducer'] is not None:
        zi = (f['reducer'].transform(zi)-f['latent_center'])/f['latent_scale']
    x = np.column_stack([zi, zc])
    return expit(x@f['classifier'].coef_[0]+f['classifier'].intercept_[0])


def manual_hc(f, sf, tables, habitat_tables, indices):
    from verify_pure_habitat_random120_20261023 import manual as manual_base
    x = np.column_stack([manual_base(base, habitat_tables, indices) for base in sf['base']])
    _, zc = run.transformed({'image': None, 'clinical': f['clinical']}, tables, indices)
    z = np.column_stack([(np.clip(x, f['low'], f['high'])-f['center'])/f['scale'], zc])
    return expit(z@f['meta'].coef_[0]+f['meta'].intercept_[0])


def training():
    run.frozen(); assert not (run.OUT/'test_started.json').exists()
    cohort, y, tr, te = run.inputs()
    assert len(tr) == 84 and len(te) == 36 and y[tr].sum() == 57 and (y[te] == -1).all()
    tables = joblib.load(run.OUT/'training_features.joblib')
    ht = joblib.load(run.habitat.parent.OUT/'Random120/training_features.joblib')
    regenerated = run.data(tr)
    for key, d in tables.items():
        np.testing.assert_array_equal(d['indices'], tr)
        assert d['metadata'] == regenerated[key]['metadata']
        for kind in (('base',) if key == 'Clinical' else ('base', 'erode', 'dilate')):
            np.testing.assert_array_equal(d[kind], regenerated[key][kind])
    for key in ('PercentileZ__Whole', 'RobustZ__Whole'):
        mm = tables[key]['metadata']
        assert len(mm) == 372 and {m: sum(r['modality'] == m for r in mm) for m in ('T2', 'ADC', 'IVIM')} == {'T2': 93, 'ADC': 93, 'IVIM': 186}
        assert not any('Dstar' in r['name'] or 'Dfast' in r['name'] or 'shape' in r['name'].lower() for r in mm)
    stack = run.source_stack(); outer = run.read('outer_metrics'); pp = run.read('nested_predictions')
    selected = run.read('selected_configurations'); choices = run.read('training_choices').set_index('ModelKey')
    records, refits, maxerr = 0, 0, 0.
    for phase in run.PHASES:
        cp = joblib.load(run.OUT/'phases'/(phase+'.joblib'))
        ffs = {(r, k): (a, b) for r, k, a, b in run.habitat.parent.folds('Random120', phase)}
        a, b = (tr, np.empty(0, int)) if phase == 'final' else run.habitat.parent.folds('Random120')[int(phase[-1])-1]
        sf = stack['final'] if phase == 'final' else stack['outer'][int(phase[-1])-1]
        results = joblib.load(run.OUT/'models'/(phase+'.joblib'))
        for c in run.options():
            rr = []
            for entry in cp['cv'][c['candidate_id']]:
                aa, bb = ffs[entry['Repeat'], entry['Fold']]
                np.testing.assert_array_equal(aa, entry['train']); np.testing.assert_array_equal(bb, entry['validation'])
                assert set(aa).isdisjoint(bb) and set(aa) | set(bb) == set(a)
                assert not (set(aa) | set(bb)) & (set(te) | set(b))
                rr.append((auc(y[aa], entry['train_probability']), auc(y[bb], entry['validation_probability'])))
                records += 1
            row = next(r for r in cp['grid'] if r['candidate_id'] == c['candidate_id'])
            rr = np.asarray(rr); mean = rr[:, 1].mean(); sd = rr[:, 1].std(ddof=1); train = rr[:, 0].mean()
            np.testing.assert_allclose([mean, sd, train, mean-.1*sd-.3*max(train-mean-.1, 0)],
                [row['MeanValidationAUC'], row['SD'], row['MeanFitAUC'], row['Objective']], atol=1e-12, rtol=0)
        for family in run.TUNED:
            wanted = sorted([r for r in cp['grid'] if r['family'] == family], key=lambda r: (
                -int(r['AvailableAllFolds']), -r['Objective'], r['MeanEffective'], r['MeanFeatureCount'], r['C'], r['candidate_id']))[0]
            chosen = selected[(selected.Phase == phase) & (selected.family == family)].iloc[0]
            assert chosen.candidate_id == wanted['candidate_id']
            f = results[family]; replay = run.fit(tables, a, b, y, run.config(wanted)); refits += 1
            if f['image'] is not None:
                assert f['image']['names'] == replay['image']['names'] and f['image']['indices'] == replay['image']['indices']
                assert all(r['Selected'] is False or r['CompleteFraction'] >= .9 and r['ICC'] >= .75 for r in f['image']['audit'])
            np.testing.assert_allclose(replay['train_probability'], f['train_probability'], atol=1e-10, rtol=0)
            np.testing.assert_allclose(replay['evaluation_probability'], f['evaluation_probability'], atol=1e-10, rtol=0)
            maxerr = max(maxerr, float(np.max(np.abs(manual(f, tables, a)-f['train_probability']))))
            if len(b):
                maxerr = max(maxerr, float(np.max(np.abs(manual(f, tables, b)-f['evaluation_probability']))))
            # Replay the selected setting on each of its 15 genuine inner validation folds.
            for entry in cp['cv'][wanted['candidate_id']]:
                fitted = run.fit(tables, entry['train'], entry['validation'], y, run.config(wanted)); refits += 1
                np.testing.assert_allclose(fitted['evaluation_probability'], entry['validation_probability'], atol=1e-10, rtol=0)
                np.testing.assert_allclose(fitted['train_probability'], entry['train_probability'], atol=1e-10, rtol=0)
        hc = results['HabitatClinical']; replay = run.fit_hc(tables, sf, y); refits += 1
        np.testing.assert_array_equal(replay['meta'].coef_, hc['meta'].coef_)
        np.testing.assert_array_equal(replay['meta'].intercept_, hc['meta'].intercept_)
        maxerr = max(maxerr, float(np.max(np.abs(manual_hc(hc, sf, tables, ht, a)-hc['train_probability']))))
        for family, f in results.items():
            if phase == 'final':
                pred = pp[pp.ModelKey == family].set_index('Index').loc[tr]
                assert f['threshold'] == run.threshold(y[tr], pred.Probability.to_numpy())
                np.testing.assert_allclose(auc(y[tr], f['train_probability']), choices.loc[family, 'TrainFitAUC'], atol=1e-12)
                np.testing.assert_allclose(auc(y[tr], pred.Probability.to_numpy()), choices.loc[family, 'PooledOOFAUC'], atol=1e-12)
            else:
                pred = pp[(pp.ModelKey == family) & (pp.Fold == int(phase[-1]))].set_index('Index').loc[b]
                np.testing.assert_array_equal(pred.Probability.to_numpy(), f['evaluation_probability'])
                row = outer[(outer.ModelKey == family) & (outer.Fold == int(phase[-1]))].iloc[0]
                np.testing.assert_allclose(auc(y[b], f['evaluation_probability']), row.AUC, atol=1e-12)
        if phase == 'outer1':
            altered = y.copy(); altered[b] = 1-altered[b]
            for family in run.TUNED:
                other = run.fit(tables, a, b, altered, results[family]['config'])
                np.testing.assert_array_equal(other['train_probability'], results[family]['train_probability'])
                np.testing.assert_array_equal(other['evaluation_probability'], results[family]['evaluation_probability'])
            other = run.fit_hc(tables, sf, altered)
            np.testing.assert_array_equal(other['meta'].coef_, hc['meta'].coef_)
        print('Verified '+phase+':all innerAUCs,choices,45innerrefits,3outer/finalrefits,fixedHC andscores', flush=True)
    hchoice = run.habitat.read('training_choices'); hchoice = hchoice[hchoice.Group == 'Random120'].iloc[0]
    for field in ('TrainFitAUC', 'OuterMeanAUC', 'OuterSD', 'PooledOOFAUC', 'Threshold'):
        assert choices.loc['Habitat', field] == hchoice[field]
    for family in run.FAMILIES:
        rr = outer[outer.ModelKey == family]; assert len(rr) == 5
        np.testing.assert_allclose([rr.AUC.mean(), rr.AUC.std(ddof=1)], choices.loc[family, ['OuterMeanAUC', 'OuterSD']].to_numpy(float), atol=1e-12)
        rows = pp[pp.ModelKey == family]; assert len(rows) == 84 and rows.Index.is_unique
    assert maxerr < 1e-8
    run.save(run.OUT/'training_verification.json', {'passed': True, 'new_test_calls': 0,
        'inner_records_checked': records, 'selected_model_refits': refits, 'maximum_manual_probability_error': maxerr,
        'same_case_and_fold_ids': True, 'no_test_oroutervalidation_fit_leak': True,
        'validation_label_perturbation_passed': True, 'thresholds_fullpipeline_outerOOF_verified': True,
        'Habitat_original_metrics_unchanged': True, 'protected_files_unchanged': run.protect()})
    print('Trainingverified;four newfrozen models ready foronejoint36 stage', flush=True)


def inference():
    run.frozen(); done = json.loads((run.OUT/'test_complete.json').read_text(encoding='utf-8'))
    assert done['passed'] and done['new_complete_model_calls'] == 4 and done['new_Habitat_calls'] == 0 and done['post_test_fits'] == 0
    cohort, _, tr, te = run.inputs(); truth = cohort.Outcome.to_numpy(int)[te]
    tables = joblib.load(run.OUT/'test_features.joblib'); regenerated = run.data(te)
    for key, d in tables.items():
        assert set(d['indices']) == set(te) and set(d['indices']).isdisjoint(tr)
        for kind in (('base',) if key == 'Clinical' else ('base', 'erode', 'dilate')):
            np.testing.assert_array_equal(d[kind], regenerated[key][kind])
    final = joblib.load(run.OUT/'models/final.joblib'); sf = run.source_stack()['final']
    ht = joblib.load(run.habitat.OUT/'Random120/test_features.joblib')
    old = run.habitat.read('test_predictions'); old = old[old.ModelKey == 'Stack12'].set_index('Index').loc[te]
    formulas = {family: manual(final[family], tables, te) for family in run.TUNED}
    formulas['HabitatClinical'] = manual_hc(final['HabitatClinical'], sf, tables, ht, te)
    formulas['Habitat'] = old.Probability.to_numpy()
    metrics = run.read('test_metrics').set_index('ModelKey'); predictions = run.read('test_predictions'); scores = {}
    maxerr = 0.
    for family in run.FAMILIES:
        frame = predictions[predictions.ModelKey == family].set_index('Index').loc[te]
        assert frame.AnalysisID.tolist() == cohort.AnalysisID.iloc[te].tolist()
        np.testing.assert_array_equal(frame.Outcome.to_numpy(), truth)
        p = frame.Probability.to_numpy(); scores[family] = p
        maxerr = max(maxerr, float(np.max(np.abs(formulas[family]-p))))
        np.testing.assert_allclose(auc(truth, p), metrics.loc[family, 'AUC'], atol=1e-12)
        threshold = sf['threshold'] if family == 'Habitat' else final[family]['threshold']
        assert threshold == metrics.loc[family, 'Threshold']
        pred = p >= threshold
        for name, value in [('TP', ((truth == 1) & pred).sum()), ('FN', ((truth == 1) & ~pred).sum()),
                            ('TN', ((truth == 0) & ~pred).sum()), ('FP', ((truth == 0) & pred).sum())]:
            assert metrics.loc[family, name] == value
    samples = draws(truth, scores, 2026102399)
    for family in run.FAMILIES:
        np.testing.assert_allclose(np.quantile(samples[family], [.025, .975]), metrics.loc[family, ['AUC_CI95Low', 'AUC_CI95High']].to_numpy(float), atol=1e-12)
    for row in run.read('paired_test_differences').itertuples(index=False):
        a, b = row.Comparison.split('_minus_')
        np.testing.assert_allclose(np.quantile(samples[a]-samples[b], [.025, .975]), [row.CI95Low, row.CI95High], atol=1e-12)
    assert maxerr < 1e-8
    run.save(run.OUT/'inference_verification.json', {'passed': True, 'independent_formula_scores': 144, 'reused_Habitat_scores_verified': 36,
        'maximum_manual_probability_error': maxerr, 'new_complete_model_calls': 4, 'test_stages': 1,
        'bootstrap5000_and_paired_differences_verified': True, 'post_test_fits': 0,
        'protected_files_unchanged': run.protect()})
    print('Five-family results independently verified;oldHscoresidentical;no posttestfits', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=('training', 'inference'))
    with threadpool_limits(1):
        globals()[p.parse_args().action]()
