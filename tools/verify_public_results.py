"""Verify aggregate report tables. Standard library only; no patient data."""
from pathlib import Path
import csv,hashlib,json,math,statistics
BASE=Path(__file__).resolve().parents[1]
def rows(name):
    with (BASE/'results'/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def main():
    primary=rows('primary_comparison.csv');folds=rows('outer_metrics.csv');test=rows('test_metrics.csv')
    assert len(primary)==4 and len(folds)==20 and len(test)==4
    bytest={r['ModelKey']:r for r in test}
    lines=[]
    for r in primary:
        key=r['ModelKey'];ff=[x for x in folds if x['ModelKey']==key]
        assert sorted(int(x['Fold']) for x in ff)==[1,2,3,4,5]
        scores=[float(x['AUC']) for x in ff]
        assert math.isclose(statistics.mean(scores),float(r['OuterMeanAUC']),abs_tol=1e-12)
        assert math.isclose(statistics.stdev(scores),float(r['OuterSD']),abs_tol=1e-12)
        t=bytest[key];tp,fn,tn,fp=(int(float(t[k])) for k in ['TP','FN','TN','FP'])
        assert (tp+fn,tn+fp)==(25,11)
        for field,value in [('Sensitivity',tp/(tp+fn)),('Specificity',tn/(tn+fp)),('Accuracy',(tp+tn)/36)]:
            assert math.isclose(float(t[field]),value,abs_tol=1e-12)
        assert math.isclose(float(t['TestAUC']),float(r['TestAUC']),abs_tol=1e-12)
        lines.append(f"{key}: CV {statistics.mean(scores):.4f} +/- {statistics.stdev(scores):.4f}; test AUC {float(t['TestAUC']):.4f}; TP/FN/TN/FP={tp}/{fn}/{tn}/{fp}")
    manifest=json.loads((BASE/'docs/source_manifest.json').read_text(encoding='utf-8'))
    for r in manifest:
        assert hashlib.sha256((BASE/r['path']).read_bytes()).hexdigest()==r['sha256'],r['path']
    print('PASS: four models, all five folds, aggregate classification metrics and source hashes.')
    print('\n'.join(lines))
    print('Limits: individual ROC AUC ordering and bootstrap confidence intervals cannot be recomputed from aggregate values alone. No raw-data model retraining is performed.')
if __name__=='__main__':main()
