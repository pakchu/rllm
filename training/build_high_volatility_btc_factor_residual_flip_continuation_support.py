"""Deterministic source support for HVBFRRFC-8."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_high_volatility_btc_factor_residual_flip_continuation as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv

PREREG_SHA = "21077ee8eee579633bc6aaad3df66fda6a5468cece4bdbb2e0da8ff2171b337f"
FEATURES = Path("data/high_volatility_btc_factor_residual_flip_continuation_features_2023_2026.csv.gz")
CLOCK = Path("data/high_volatility_btc_factor_residual_flip_continuation_clocks_2023_2026.csv.gz")
RESULT = Path("results/high_volatility_btc_factor_residual_flip_continuation_support_2026-08-18.json")
STAGES = {key: tuple(pd.Timestamp(item) for item in value) for key, value in prereg.build()["stages"].items()}
GATES = prereg.build()["source_support_gates"]

def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["decision_time", "btc_return", *prereg.ALTS, "btc_realized_variation", "variation_rank"]
    if any(column not in frame for column in required):
        raise RuntimeError("HVBFRRFC-8 source schema drift")
    output = frame.loc[:, required].copy(); output["decision_time"] = pd.to_datetime(output["decision_time"], utc=True, errors="raise")
    for column in required[1:]: output[column] = pd.to_numeric(output[column], errors="coerce")
    if output["decision_time"].duplicated().any() or not output["decision_time"].is_monotonic_increasing:
        raise RuntimeError("HVBFRRFC-8 decision order drift")
    alts = output.loc[:, prereg.ALTS].to_numpy(float)
    output["alt_factor"] = np.median(alts, axis=1)
    output["alt_mad"] = np.median(np.abs(alts - output["alt_factor"].to_numpy()[:, None]), axis=1)
    output["residual"] = output["btc_return"] - output["alt_factor"]
    output["previous_decision"] = output["decision_time"].shift(1)
    output["previous_residual"] = output["residual"].shift(1)
    output["previous_alt_mad"] = output["alt_mad"].shift(1)
    output["prior_standardized_residual"] = output["previous_residual"].abs() / output["previous_alt_mad"]
    finite = np.isfinite(output[["btc_return", *prereg.ALTS, "alt_factor", "alt_mad", "residual", "previous_residual", "previous_alt_mad", "prior_standardized_residual", "btc_realized_variation", "variation_rank"]]).all(axis=1)
    consecutive = output["decision_time"].sub(output["previous_decision"]).eq(pd.Timedelta("8h"))
    opposite_sign = np.sign(output["residual"]) == -np.sign(output["previous_residual"])
    output["eligible"] = (finite & consecutive & output["alt_mad"].gt(0) & output["previous_alt_mad"].gt(0) & output["residual"].ne(0) & output["previous_residual"].ne(0) & opposite_sign & output["prior_standardized_residual"].ge(prereg.build()["policy"]["prior_standardized_residual_min"]) & output["variation_rank"].ge(prereg.build()["policy"]["variation_rank_min"]))
    return output

def stage_for(entry: pd.Timestamp, exit_: pd.Timestamp) -> str | None:
    return next((name for name, (start, end) in STAGES.items() if start <= entry and exit_ <= end), None)

def build_clock(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in features.loc[features["eligible"]].itertuples(index=False):
        decision = pd.Timestamp(row.decision_time); entry = decision + pd.Timedelta("5m"); exit_ = entry + pd.Timedelta("8h"); split = stage_for(entry, exit_)
        if split is None: continue
        side = int(np.sign(row.residual))
        rows.append({"candidate":prereg.POLICY_ID,"control":"primary","split":split,"decision_time":decision,"feature_available_time":decision,"entry_time":entry,"exit_time":exit_,"side":side,"residual":float(row.residual),"previous_residual":float(row.previous_residual),"prior_standardized_residual":float(row.prior_standardized_residual),"alt_mad":float(row.alt_mad),"variation_rank":float(row.variation_rank)})
    columns=("candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side","residual","previous_residual","prior_standardized_residual","alt_mad","variation_rank")
    output=pd.DataFrame(rows,columns=columns).sort_values("decision_time",kind="stable").reset_index(drop=True)
    if len(output)>1 and not output["entry_time"].iloc[1:].reset_index(drop=True).ge(output["exit_time"].iloc[:-1].reset_index(drop=True)).all(): raise RuntimeError("HVBFRRFC-8 overlap")
    return output

def stats(clock: pd.DataFrame, split: str) -> dict[str, Any]:
    frame=clock.loc[clock["split"].eq(split)];n=len(frame);longs=int(frame["side"].eq(1).sum());shorts=int(frame["side"].eq(-1).sum());month=float(frame["entry_time"].dt.strftime("%Y-%m").value_counts().max()/n) if n else 0.0
    return {"events":n,"longs":longs,"shorts":shorts,"minority_side_share":min(longs,shorts)/n if n else 0.0,"max_month_share":month}

def run(features_path: Path = FEATURES, clock_path: Path = CLOCK, result_path: Path = RESULT) -> dict[str, Any]:
    if prereg.sha256(prereg.DEFAULT_OUTPUT)!=PREREG_SHA: raise RuntimeError("HVBFRRFC-8 prereg drift")
    registration=json.loads(prereg.DEFAULT_OUTPUT.read_text());prereg.validate(registration)
    features=prepare(pd.read_csv(prereg.SOURCE));_write_gzip_csv(features,features_path);clock=build_clock(features);_write_gzip_csv(clock,clock_path)
    support={split:stats(clock,split) for split in STAGES};checks={}
    for split,value in support.items():
        checks[f"{split}_minimum_events"]=value["events"]>=GATES["minimum_events"][split];checks[f"{split}_side_balance"]=value["minority_side_share"]>=GATES["minority_side_share_min"];checks[f"{split}_month_concentration"]=value["max_month_share"]<=GATES["max_month_share"]
    passed=all(checks.values());core={"protocol_version":"hvbfr rfc_8_source_support_v1".replace(" ",""),"policy_id":prereg.POLICY_ID,"preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA,"manifest_hash":registration["manifest_hash"]},"source":{"path":str(prereg.SOURCE),"sha256":prereg.SOURCE_SHA},"completed_preentry_sources_opened":True,"candidate_incidence_opened":True,"postentry_return_pnl_execution_price_opened":False,"funding_opened":False,"gross9_rows_opened":False,"features":{"path":str(features_path),"sha256":prereg.sha256(features_path),"rows":len(features)},"clock":{"path":str(clock_path),"sha256":prereg.sha256(clock_path),"rows":len(clock)},"support":support,"support_checks":checks,"support_passed":passed,"advance_to_gross9_novelty":passed,"advance_to_economic_outcomes":False,"decision":"pass_to_novelty" if passed else "terminal_source_support_reject"}
    result={**core,"manifest_hash":prereg.canonical_hash(core)};result_path.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n");return result

if __name__=="__main__":
    report=run();print(json.dumps({"passed":report["support_passed"],"support":report["support"]},indent=2))
