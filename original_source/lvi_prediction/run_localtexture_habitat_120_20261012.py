"""Second prespecified training-only attempt; same pipeline, new patient-local partition."""
import argparse
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import run_regional_habitat_120_20261012 as shared
import localtexture_habitat_features_20261012 as feature

FIRST_OUT = shared.OUT
OUT = shared.ROOT/"lvi_prediction/results/lvi_localtexture_habitat_120_20261012"
original_freeze,original_ranking = shared.freeze,shared.ranking


def freeze():
    inputs = original_freeze()
    protocol = json.loads((OUT/"protocol.json").read_text(encoding="utf-8"))
    protected = protocol["protected_hashes"]
    for p in [Path(__file__).resolve(),shared.ROOT/"lvi_prediction/localtexture_habitat_features_20261012.py"]:
        protected[str(p.relative_to(shared.ROOT))] = shared.digest(p)
    protected.update({str(p.relative_to(shared.ROOT)):shared.digest(p) for p in FIRST_OUT.rglob("*") if p.is_file()})
    protocol["partition_fraction_features"] = "variable habitat occupancy allowed when training ICC, completeness and sign-stability pass"
    protocol["prior_attempt"] = "tertile regional texture attempt: outer mean .4618; retained training-only, no test evaluated"
    protocol["prior_attempt_test_evaluated"] = False
    shared.save(OUT/"protocol.json",protocol)
    return inputs


def ranking(data,train,y):
    ordered,ranks,audit = original_ranking(data,train,y)
    reference = data["base"][shared.rows(data,train)]
    for j,m in enumerate(data["metadata"]):
        if not m["name"].endswith("_TumorFraction"):continue
        a = audit[j]
        eligible = bool(a["ICC"] is not None and a["ICC"]>=.8 and a["CompleteFraction"]>=.9
            and a["BootstrapSignConsistency"]>=.8 and np.nanstd(reference[:,j])>1e-12)
        a["Eligible"] = eligible
        if eligible:ordered.append(j)
    ordered.sort(key=lambda j:(-audit[j]["RankingScore"],-audit[j]["ICC"],data["metadata"][j]["name"]))
    return ordered,ranks,audit


def configure():
    shared.OUT = OUT;shared.feature = feature
    shared.freeze = freeze;shared.ranking = ranking


if __name__=="__main__":
    parser = argparse.ArgumentParser();parser.add_argument("--phase",choices=["train","evaluate"],required=True)
    args = parser.parse_args();configure()
    with threadpool_limits(limits=1):shared.train() if args.phase=="train" else shared.evaluate()
