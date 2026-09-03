"""Preregister the overlap-allowed portfolio search over 71 frozen schedules."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from training import optimize_gross9_overlap_portfolio as optimizer

POLICY_ID = optimizer.POLICY_ID
PROTOCOL_VERSION = "gross9_overlap_allowed_portfolio_preregistration_v1"
AS_OF_DATE = "2026-09-03"
DEFAULT_OUTPUT = Path("results/gross9_overlap_portfolio_preregistration_2026-09-03.json")
UNIVERSE = Path("results/gross9_overlap_portfolio_universe_2026-09-03.json")
UNIVERSE_SHA256 = "e2a631cface501a1264d736c6635e64c2931667425b6abe873123d5e6c37ac8c"
UNIVERSE_MANIFEST_HASH = "b1c88b50fc923e3b54f63347b9e828c98540db2148690fc5d130e03bb7cf12cc"


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build() -> dict[str, Any]:
    universe = json.loads(UNIVERSE.read_text(encoding="utf-8"))
    if sha256_file(UNIVERSE) != UNIVERSE_SHA256 or universe.get("manifest_hash") != UNIVERSE_MANIFEST_HASH:
        raise RuntimeError(f"{POLICY_ID} universe receipt drift")
    cfg = optimizer.OptimizerConfig()
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": POLICY_ID,
        "as_of_date": AS_OF_DATE,
        "objective": "find the best shadow portfolio inside a frozen deterministic grammar while allowing inter-sleeve position overlap",
        "research_status": "adaptive_train_search_shadow_only_not_live",
        "immutable_universe": {"path": str(UNIVERSE), "sha256": UNIVERSE_SHA256, "manifest_hash": UNIVERSE_MANIFEST_HASH, "canonical_sleeves": universe["canonical_sleeve_count"]},
        "overlap_policy": {
            "inter_sleeve_positions_allowed": True,
            "intra_sleeve_positions_allowed": False,
            "candidate_to_candidate_overlap": "disclosure_only",
            "candidate_to_gross9_near_6h_overlap": "disclosure_only",
            "gross_risk_nets_opposite_positions": False,
            "execution_and_funding_use_aggregate_net_quantity": True,
            "exact_schedule_aliases": "one canonical sleeve only",
        },
        "selection_windows": {
            "proxy_and_exact_search": list(optimizer.TRAIN_PROXY_WINDOW),
            "internal_holdout": list(optimizer.DECEMBER_HOLDOUT_WINDOW),
            "holdout_opened_by_preregistration": False,
            "oos_opened_by_preregistration": False,
        },
        "search_grammar": {
            "weight_grid": list(cfg.weight_grid),
            "minimum_gross": cfg.min_gross,
            "maximum_gross": cfg.max_gross,
            "maximum_sleeves": cfg.max_sleeves,
            "beam_width": cfg.beam_width,
            "proxy_candidate_cap": cfg.proxy_candidate_cap,
            "exact_finalists": cfg.exact_finalist_count,
            "proxy": "cost-adjusted signed entry/exit-open return aggregation for bounded search only",
            "authority": "top proxy finalists replayed with fixed-quantity aggregate-net ledger",
            "random_search": False,
            "raw_rank_one_no_substitution": True,
        },
        "train_gates": {
            "minimum_sleeve_intervals": cfg.min_trade_count,
            "minimum_active_iso_weeks": cfg.min_active_weeks,
            "absolute_return_positive": True,
            "cagr_to_strict_mdd_min": cfg.min_cagr_to_strict_mdd,
            "strict_mdd_max_pct": cfg.max_strict_mdd_pct,
            "stress_return_positive": True,
            "stress_cagr_to_strict_mdd_min": cfg.min_stress_cagr_to_strict_mdd,
            "mean_exposure_weighted_gross_edge_min_bp": 20.0,
            "both_chronological_halves_positive": True,
            "base_positive_months_min": 4,
            "stress_positive_months_min": 3,
            "worst_stress_month_min_pct": -2.5,
            "maximum_simultaneous_gross": cfg.max_gross,
            "maximum_mean_gross": cfg.max_mean_gross_exposure,
            "maximum_turnover_weight_per_day": cfg.max_turnover_weight_per_day,
            "maximum_single_sleeve_turnover_share": cfg.max_sleeve_turnover_share,
        },
        "internal_holdout": {
            "opened_only_after_rank1_freeze": True,
            "base_and_stress_return_positive": True,
            "strict_mdd_max_pct": 8.0,
            "minimum_sleeve_intervals": 8,
            "minimum_active_weeks": 3,
            "failure_terminal": True,
        },
        "oos_sequence": {
            "stages": ["test2024", "eval2025", "final2026"],
            "single_hypothesis_weekly_p_max": 0.10,
            "source_signed_episode_min": {"test2024": 12, "eval2025": 12, "final2026": 8},
            "economic_gates_same_as_existing": True,
            "stop_on_first_failure": True,
            "rerank_or_repair_authorized": False,
        },
        "reporting": ["sleeve_intervals", "long_short_intervals", "atomic_transitions", "nonzero_net_execution_events", "signed_episodes", "active_weeks", "max_and_mean_gross", "max_abs_net", "gross_and_net_turnover", "netting_savings", "fees", "funding", "monthly_returns"],
        "implementation": {
            "preregister": {"path": "training/preregister_gross9_overlap_portfolio.py", "sha256": sha256_file(__file__)},
            "optimizer": {"path": "training/optimize_gross9_overlap_portfolio.py", "sha256": sha256_file(optimizer.__file__)},
        },
        "evidence_boundary": {"market_rows_opened": 0, "funding_rows_opened": 0, "train_outcomes_opened": False, "december_holdout_outcomes_opened": False, "oos_outcomes_opened": False},
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: Mapping[str, Any]) -> None:
    core = dict(value); observed = core.pop("manifest_hash", None)
    if observed != canonical_hash(core): raise RuntimeError(f"{POLICY_ID} preregistration drift")
    if value.get("policy_id") != POLICY_ID or value.get("protocol_version") != PROTOCOL_VERSION: raise RuntimeError(f"{POLICY_ID} identity drift")
    if value.get("immutable_universe") != {"path": str(UNIVERSE), "sha256": UNIVERSE_SHA256, "manifest_hash": UNIVERSE_MANIFEST_HASH, "canonical_sleeves": 71}: raise RuntimeError(f"{POLICY_ID} universe binding drift")
    overlap = value.get("overlap_policy", {})
    if overlap.get("inter_sleeve_positions_allowed") is not True or overlap.get("gross_risk_nets_opposite_positions") is not False: raise RuntimeError(f"{POLICY_ID} overlap policy drift")
    if value.get("selection_windows", {}).get("holdout_opened_by_preregistration") is not False: raise RuntimeError(f"{POLICY_ID} holdout boundary drift")
    if value.get("oos_sequence", {}).get("rerank_or_repair_authorized") is not False: raise RuntimeError(f"{POLICY_ID} no-repair drift")
    for record in value.get("implementation", {}).values():
        if sha256_file(record["path"]) != record["sha256"]: raise RuntimeError(f"{POLICY_ID} implementation binding drift")


def main(argv: Sequence[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT);args=parser.parse_args(argv)
    result=build();validate(result);args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2,sort_keys=True,ensure_ascii=False)+"\n",encoding="utf-8");return 0

if __name__=="__main__": raise SystemExit(main())
