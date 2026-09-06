"""Preregister pre-2024 open-interest support for PPOSM residual actions."""
from __future__ import annotations
import hashlib, inspect, json
from pathlib import Path
from typing import Any
PROJECT_ROOT=Path(__file__).resolve().parents[1]
POLICY_ID="pposm_pre2024_oi_action_source_v1"
DEFAULT_OUTPUT=Path("results/pposm_pre2024_oi_action_source_preregistration_2026-09-05.json")
BUILDER_PATH=PROJECT_ROOT/"training"/"build_pposm_pre2024_oi_action_source.py"
THIS_PATH=Path(__file__).resolve()
QUERY_START="2020-01-01T00:00:00Z"; QUERY_END_EXCLUSIVE="2024-01-01T00:00:00Z"
FEATURE_WINDOWS_PERIODS=[12,48,96,288]  # 1h,4h,8h,24h at 5m cadence
VALIDATION_YEARS=[2021,2022,2023]
OI_QUERY="""
    SELECT ts AS date, sum_open_interest
    FROM open_interest_binance
    WHERE symbol = :symbol AND period = '5m' AND ts >= :start AND ts < :end
    ORDER BY ts
"""
ACCEPTANCE={"split_policy":"pre-2024 expanding-forward years only; validate 2021, 2022, 2023 when both classes are present and at least 80 earlier signals exist","candidate_actions":["SKIP","TP12"],"default_action":"TP4","minimum_training_signals_before_fold":80,"minimum_validation_signals_per_fold":10,"pooled_auc_min_each_candidate":0.60,"bootstrap_lower95_auc_min_each_candidate":0.50,"balanced_accuracy_min_each_candidate":0.55,"consistent_year_direction":"every evaluated candidate-year AUC must be >= 0.50","advance_if_all_pass":"only then open a separate preregistered historical OOS/residual-router check"}
FORBIDDEN=["2024_or_later_counterfactual_labels","fresh_forward_lifecycle_or_return_outcomes","gross9_future_rows","trained_adapter_scores","post_entry_outcomes_in_features","threshold_tuning_after_source_results"]
FEATURES={"source":"BTCUSDT open_interest_binance five-minute sum_open_interest paths ending strictly before each frozen PPOSM decision timestamp","windows_periods_5m":FEATURE_WINDOWS_PERIODS,"per_window_features":["level_mean","level_std","level_min","level_max","level_range","last_over_first_log","second_half_minus_first_half_log","abs_logdiff_sum","positive_logdiff_share","negative_logdiff_share","end_zscore"],"normalization":"fit mean/std on each expanding training fold only","model":"deterministic balanced L2 logistic regression, one binary model per candidate action","feature_selection":"none; all preregistered finite features are used"}
def canonical_json(v:Any)->str:return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)
def sha256_bytes(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def sha256_path(p:Path)->str|None:
 if not p.exists(): return None
 h=hashlib.sha256();
 with p.open('rb') as f:
  for c in iter(lambda:f.read(1048576),b''): h.update(c)
 return h.hexdigest()
def source_hash(o:Any)->str:
 try:return sha256_bytes(inspect.getsource(o).encode())
 except Exception:return "unavailable"
def build()->dict[str,Any]:
 d={"policy_id":POLICY_ID,"created_at":"2026-09-05T00:00:00Z","purpose":"test whether materially new pre-2024 OI paths can explain frozen PPOSM residual action labels before any OOS reopen","source_contract":{"oi_table":"open_interest_binance","symbol":"BTCUSDT","period":"5m","query_start":QUERY_START,"query_end_exclusive":QUERY_END_EXCLUSIVE,"required_exact_grid":True,"required_finite_positive_column":"sum_open_interest","minimum_feature_lookback_minutes":1440,"query_rows_opened_by_this_preregistration":0,"forbidden_sources":FORBIDDEN,"oi_query":OI_QUERY,"oi_query_sha256":sha256_bytes(OI_QUERY.encode())},"label_contract":{"labels":"pre-2024 frozen PPOSM counterfactual utility only; SWITCH iff candidate utility exceeds TP4 utility","candidate_actions":ACCEPTANCE["candidate_actions"],"default_action":ACCEPTANCE["default_action"],"oos_labels_opened":False,"fresh_outcomes_opened":False},"features":FEATURES,"validation":ACCEPTANCE,"code_hashes":{"preregistration_module":sha256_path(THIS_PATH),"builder_module":sha256_path(BUILDER_PATH),"build_function":source_hash(build)},"terminal_semantics":"fail closed unless every acceptance check passes; no OOS repair or model promotion from a failed source-support result"}
 d["preregistration_hash"]=sha256_bytes(canonical_json({k:v for k,v in d.items() if k!="preregistration_hash"}).encode())
 return d
def main():
 out=build(); DEFAULT_OUTPUT.parent.mkdir(parents=True,exist_ok=True); payload=json.dumps(out,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+"\n"
 if DEFAULT_OUTPUT.exists() and json.loads(DEFAULT_OUTPUT.read_text()).get("preregistration_hash")!=out["preregistration_hash"]: raise RuntimeError("refusing to overwrite different OI preregistration")
 DEFAULT_OUTPUT.write_text(payload); print(json.dumps({"path":str(DEFAULT_OUTPUT),"preregistration_hash":out["preregistration_hash"]},sort_keys=True))
if __name__=='__main__': main()
