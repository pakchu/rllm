"""Freeze POWR-12 before inspecting any policy outcome."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = "results/perp_only_wick_rejection_preregistration_2026-07-17.json"
SOURCE_MANIFEST = "results/delta_neutral_carry_sources_pre2024_manifest_2026-07-16.json"
PERP_SOURCE = "data/binance_perp_btc_1m_2020_2023.csv.gz"
SPOT_SOURCE = "data/binance_spot_btc_1m_2020_2023.csv.gz"


@dataclass(frozen=True)
class Policy:
    policy_id: str = "POWR-12"
    baseline_bars: int = 8_640
    baseline_min_periods: int = 2_016
    wick_excess_quantile: float = 0.95
    minimum_perp_wick_bp: float = 6.0
    maximum_spot_to_perp_wick_ratio: float = 0.5
    entry_delay_bars: int = 3
    hold_bars: int = 12
    leverage: float = 0.5
    base_cost_notional_per_side: float = 0.0006
    stress_cost_notional_per_side: float = 0.0008


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_manifest() -> dict[str, Any]:
    core: dict[str, Any] = {
        "protocol_version": "perp_only_wick_rejection_v1",
        "outcomes_opened": False,
        "policy": asdict(Policy()),
        "research_history_boundary": {
            "2020_2023_market_returns_seen_by_unrelated_repo_research": True,
            "exact_powr12_outcomes_opened": False,
            "claim": (
                "exact-policy mechanical freeze, not a globally pristine market "
                "holdout; 2024+ and live forward remain final generalization tests"
            ),
        },
        "novelty_boundary": {
            "economic_axis": (
                "a derivative-venue tail excursion rejected inside the completed "
                "perp bar while the cash venue does not validate the tail"
            ),
            "not": [
                "spot-perp basis level or compression",
                "funding carry",
                "spot-perp absorption state grid",
                "phase slip or relock",
                "transfer entropy",
                "OI/funding/Kimchi/REX/Markov/aggTrade HHI",
            ],
        },
        "source_contract": {
            "source_manifest": SOURCE_MANIFEST,
            "source_manifest_sha256": (
                "6f732c2ba8a158e82cecdd954b2d14310dd564bddc4d04dbd04f8672b545844d"
            ),
            "perp": PERP_SOURCE,
            "perp_sha256": (
                "0b55bb0c3b845a90da738e746c769b19c1de4ac230ca8f1fccb6c361c4a9a41f"
            ),
            "perp_rows": 2_103_840,
            "spot": SPOT_SOURCE,
            "spot_sha256": (
                "bc6e0fd6b773ab6458a5de88fb9589161d1adf4ac1d0e7024f252515909f4a54"
            ),
            "spot_rows": 2_101_493,
            "interval": ["2020-01-01", "2024-01-01"],
            "database_snapshot_is_point_in_time": False,
            "revision_boundary": (
                "exchange timestamps define semantic availability and hashes freeze "
                "this backfilled snapshot; live promotion requires forward parity"
            ),
            "missing_spot_policy": "never impute; incomplete five-minute bucket is ineligible",
        },
        "five_minute_aggregation": {
            "bucket_label": "UTC five-minute bucket start t",
            "required_rows": "exactly five consecutive one-minute rows per venue",
            "open": "first one-minute open",
            "high": "maximum one-minute high",
            "low": "minimum one-minute low",
            "close": "last one-minute close",
            "availability": "after the fifth one-minute row closes at t+5m",
        },
        "causal_feature_contract": {
            "upper_v": "log(high_v / max(open_v, close_v))",
            "lower_v": "log(min(open_v, close_v) / low_v)",
            "perp_body": "log(perp_close / perp_open)",
            "upper_excess": "max(0, upper_perp - upper_spot)",
            "lower_excess": "max(0, lower_perp - lower_spot)",
            "thresholds": (
                "strictly-prior q95 over the last 8,640 complete joint buckets; "
                "minimum 2,016 observations"
            ),
            "short": (
                "upper_excess>=prior q95, upper_perp>=6bp, "
                "upper_spot<=0.5*upper_perp, perp_body<=0"
            ),
            "long": (
                "lower_excess>=prior q95, lower_perp>=6bp, "
                "lower_spot<=0.5*lower_perp, perp_body>=0"
            ),
            "both_branches": "reject bucket if long and short both fire",
        },
        "execution_contract": {
            "decision": "after signal bucket t closes at t+5m",
            "latency_health_gate": (
                "the immediately following joint spot/perp five-minute bucket must "
                "complete before entry; this is known at entry time"
            ),
            "entry": (
                "next tradable perp open at t+15m (signal_position + 3), after "
                "the t+5m..t+10m latency bucket is fully observed"
            ),
            "exit": "scheduled perp open 12 five-minute bars after entry",
            "hold": "60 minutes fixed",
            "nonoverlap": True,
            "stop_or_take_profit": None,
            "leverage": 0.5,
            "base_cost": "6bp/notional/side",
            "stress_cost": "8bp/notional/side",
            "funding_interval": "entry_time <= funding_time < exit_time",
            "cagr": "full wall-clock split including idle cash",
            "strict_mdd": (
                "global/pre-entry HWM, favorable-before-adverse held OHLC, funding, "
                "entry/exit/hypothetical liquidation costs"
            ),
        },
        "support_freeze_before_returns": {
            "train_2020_2022_nonoverlap_min": 500,
            "each_train_year_min": 80,
            "selection_2023_min": 40,
            "selection_2023_h1_min": 20,
            "selection_2023_h2_min": 10,
            "each_side_share_range": [0.35, 0.65],
            "each_branch_share_min": 0.20,
            "maximum_single_month_share": 0.12,
            "signal_and_latency_joint_buckets_complete": True,
            "future_hold_spot_availability_used_to_filter": False,
            "failure_action": "reject without opening post-entry outcomes",
        },
        "selection_protocol": {
            "train": ["2020-01-01", "2023-01-01"],
            "selection": ["2023-01-01", "2024-01-01"],
            "selection_halves": {
                "h1": ["2023-01-01", "2023-07-01"],
                "h2": ["2023-07-01", "2024-01-01"],
            },
            "sealed": ["2024", "2025", "2026_ytd"],
            "candidate_count": 1,
            "no_parameter_repair": True,
            "gates": {
                "train_and_2023_absolute_return_positive": True,
                "train_and_2023_cagr_to_strict_mdd_min": 3.0,
                "train_and_2023_strict_mdd_pct_max": 15.0,
                "train_and_2023_weekly_cluster_signflip_p_max": 0.10,
                "train_and_2023_mean_gross_underlying_bp_min": 12.0,
                "train_and_2023_eight_bp_stress_positive": True,
                "2023_h1_and_h2_absolute_return_positive": True,
                "one_bar_delayed_entry_train_and_2023_positive": True,
            },
        },
        "controls": {
            "direction_flip": (
                "primary clock, opposite side; diagnostic-only and never a repair, "
                "replacement, or rejection gate"
            ),
            "spot_only_wick": "identical wick/body logic on spot without perp anchor",
            "common_wick": (
                "perp tail q95 and >=6bp but spot wick >=0.5*perp wick"
            ),
            "basis_free_perp_wick": (
                "perp wick prior q95 and >=6bp with body rejection; no spot ratio"
            ),
            "one_bar_delayed_entry": (
                "primary side, entry one additional bar later at signal_position+4 "
                "(t+20m)"
            ),
            "stale_spot_1h": "compare current perp wick with spot wick 12 bars earlier",
            "stale_spot_1d": "compare current perp wick with spot wick 288 bars earlier",
            "mechanism_rejection_rule": (
                "reject POWR-12 if spot-only, common-wick, basis-free, or either "
                "stale-spot control independently passes every primary gate"
            ),
        },
        "orthogonality_after_performance": {
            "exact_entry_jaccard_max": 0.05,
            "position_time_jaccard_max": 0.15,
            "absolute_daily_pnl_pearson_max": 0.30,
            "minimum_nonzero_daily_pnl_days": 20,
            "marginal_portfolio_improvement_required": True,
            "undefined_metric": "fail_closed",
        },
        "rejection_contract": (
            "any support/performance/mechanism failure rejects POWR-12 without "
            "changing threshold, wick floor, spot ratio, latency, side, or hold"
        ),
    }
    return {
        **core,
        "manifest_hash": canonical_hash(core),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def validate_manifest(payload: dict[str, Any]) -> None:
    core = {
        key: value
        for key, value in payload.items()
        if key not in {"manifest_hash", "created_at"}
    }
    if canonical_hash(core) != payload.get("manifest_hash"):
        raise RuntimeError("POWR-12 preregistration hash mismatch")
    if payload.get("outcomes_opened") is not False:
        raise RuntimeError("POWR-12 preregistration cannot open outcomes")
    if payload.get("policy") != asdict(Policy()):
        raise RuntimeError("POWR-12 policy differs from code")
    if payload.get("selection_protocol", {}).get("candidate_count") != 1:
        raise RuntimeError("POWR-12 must remain a singleton")


def write_once(path: str | Path, payload: dict[str, Any]) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = json.loads(output.read_text())
        validate_manifest(existing)
        if existing["manifest_hash"] != payload["manifest_hash"]:
            raise RuntimeError("refusing to overwrite frozen POWR-12 preregistration")
        return "verified_existing"
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return "created"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build_manifest()
    status = write_once(args.output, payload)
    print(
        json.dumps(
            {
                "status": status,
                "outcomes_opened": False,
                "policy_id": payload["policy"]["policy_id"],
                "manifest_hash": payload["manifest_hash"],
                "output": args.output,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
