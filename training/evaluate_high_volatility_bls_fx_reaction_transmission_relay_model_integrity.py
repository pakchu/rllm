"""Outcome-blind calendar feasibility gate for frozen HVBLSFX-12."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
from typing import Any
import pandas as pd
from training import preregister_high_volatility_bls_fx_reaction_transmission_relay as prereg

PREREG_SHA="77bfe0a5d56e16c1fe92fffe9ddd0d9ef87d5c0083b5c4fb92c25745d89c922b"
CALENDAR=Path("data/high_volatility_bls_fx_reaction_transmission_relay_sources_2023_2026/official_bls_selected_releases.csv")
CALENDAR_SHA="7bd399c71e97070f0e904c3e9d58d7da55f06b472599a330b084a419fd796a4b"
RESULT=Path("results/high_volatility_bls_fx_reaction_transmission_relay_model_integrity_failure_2026-08-13.json")
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def chash(x:Any)->str:return prereg.canonical_hash(x)
def run()->dict[str,Any]:
 if sha(prereg.DEFAULT_OUTPUT)!=PREREG_SHA or sha(CALENDAR)!=CALENDAR_SHA:raise RuntimeError("HVBLSFX predecessor drift")
 x=pd.read_csv(CALENDAR);dates=pd.to_datetime(x.release_date,errors="raise");train_start=pd.Timestamp("2023-07-01");train_end=pd.Timestamp("2024-01-01");minimum=prereg.build()["policy"]["minimum_history_releases"]
 before_start=int((dates<train_start).sum());before_end=int((dates<train_end).sum());maximum_ranked_train=max(0,before_end-minimum);required=prereg.build()["source_support_gates"]["minimum_events"]["train"]
 checks={"minimum_history_available_before_train":before_start>=minimum,"maximum_possible_train_events_meets_gate":maximum_ranked_train>=required};passed=all(checks.values())
 core={"protocol_version":"hvblsfx_12_model_integrity_v1","policy_id":prereg.POLICY_ID,"preregistration":{"path":str(prereg.DEFAULT_OUTPUT),"sha256":PREREG_SHA},"official_calendar":{"path":str(CALENDAR),"sha256":CALENDAR_SHA,"rows":len(x),"unique_release_timestamps":int(dates.nunique()),"source_urls":sorted(x.source_url.unique().tolist()),"transport":"official BLS pages retrieved through browser because direct curl returned HTTP 403"},"frozen_requirements":{"minimum_history_releases":minimum,"train_minimum_events":required,"train_window":[train_start.isoformat(),train_end.isoformat()]},"feasibility":{"selected_releases_before_train":before_start,"selected_releases_before_train_end":before_end,"maximum_possible_ranked_train_events":maximum_ranked_train},"checks":checks,"model_integrity_passed":passed,"candidate_incidence_opened":False,"fx_or_btc_feature_values_opened":False,"postentry_outcomes_opened":False,"gross9_rows_opened":False,"advance_to_source_support":False,"decision":"terminal_preregistered_history_floor_failure","failure_action":"reject HVBLSFX-12 unchanged; no history, universe, threshold, clock, side, hold, subset, source, or control repair"};r={**core,"manifest_hash":chash(core)};RESULT.write_text(json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False)+"\n");return r
if __name__=="__main__":print(json.dumps(run()["feasibility"]))
