"""Preregister the tail-arrival absorption/release alpha before support inspection."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = "results/tail_arrival_absorption_preregistration_2026-07-16.json"
SELECTION_END = "2023-01-01"
HOLDOUT = ("2023-01-01", "2024-01-01")


@dataclass(frozen=True, order=True)
class Policy:
    policy_id: str
    branch: str
    hold_bars: int


def policy_grid() -> list[Policy]:
    policies: list[Policy] = []
    for branch in ("tail_absorption_fade", "tail_release_follow"):
        for hold_bars in (12, 36):
            policies.append(Policy(f"T{len(policies) + 1:02d}", branch, hold_bars))
    return sorted(policies)


def canonical_hash(payload: Any) -> str:
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def build_manifest() -> dict[str, Any]:
    core: dict[str, Any] = {
        "protocol_version": "tail_arrival_absorption_v1",
        "outcomes_opened": False,
        "research_history_boundary": {
            "2020_2022_market_returns_globally_seen": True,
            "2023_market_returns_globally_seen_by_unrelated_repo_research": True,
            "exact_tail_arrival_policy_2023_outcome_opened": False,
            "claim": (
                "mechanically sealed exact-policy holdout, not a globally pristine "
                "historical market period; forward live remains the final test"
            ),
        },
        "novelty_check": {
            "repo_search_before_preregistration": {
                "event_notional_mean_alpha_uses": 0,
                "event_notional_std_alpha_uses": 0,
                "event_notional_p50_alpha_uses": 0,
                "event_notional_p90_alpha_uses": 0,
                "event_notional_p99_alpha_uses": 0,
                "event_notional_max_alpha_uses": 0,
                "interarrival_mean_ms_alpha_uses": 0,
                "interarrival_std_ms_alpha_uses": 0,
            },
            "nearest_existing_families": [
                "metaorder fragmentation impact curvature",
                "notional-event topology fracture",
            ],
            "distinct_axis": (
                "absolute event-size tail shape and arrival-time coefficient of "
                "variation; no HHI, effective-event-count, OI, funding, or REX gate"
            ),
        },
        "source_contract": {
            "historical_features": (
                "data/binance_um_aggtrade_microstructure_btc_2020_2023/"
                "BTCUSDT_aggtrade_5m_2020-01-01_2023-12-31.csv.gz"
            ),
            "historical_feature_manifest": (
                "data/binance_um_aggtrade_microstructure_btc_2020_2023/build_manifest.json"
            ),
            "historical_market": (
                "data/binance_um_kline_reference_btc_2020_2023/"
                "BTCUSDT_5m_2020-01-01_2023-12-31.csv.gz"
            ),
            "official_archive_reference": (
                "https://data.binance.vision/?prefix=data/futures/um/daily/aggTrades/BTCUSDT/"
            ),
            "official_api_reference": (
                "https://developers.binance.com/docs/derivatives/usds-margined-futures/"
                "websocket-market-streams/Aggregate-Trade-Streams"
            ),
            "selection_prefix": "physically stop before parsing 2023 non-date values",
            "source_gap_policy": (
                "full verified source-gap day, missing feature bar, and following 24 "
                "bars quarantined; never impute"
            ),
            "live_parity": (
                "compute identical per-5m statistics from Binance UM aggTrade stream, "
                "verify aggregate/underlying trade-ID continuity, fail closed on gaps"
            ),
        },
        "claim_boundary": {
            "claim": (
                "event-size tail shape and aggressive-event arrival irregularity may "
                "separate absorbed large packets from released packets"
            ),
            "not_claimed": [
                "participant identity",
                "institutional order classification",
                "resting-book liquidity",
                "true hidden orders",
            ],
        },
        "feature_contract": {
            "tail_span": "log(event_notional_p99 / p50) + 0.5*log(max / p99)",
            "event_dispersion": "log1p(event_notional_std / event_notional_mean)",
            "arrival_cv": "log1p(interarrival_std_ms / interarrival_mean_ms)",
            "size_asymmetry": "buy_sell_event_size_log_ratio",
            "packet_direction": "sign(size_asymmetry); zero is ineligible",
            "signed_price_response": "packet_direction * micro_log_return",
            "thresholds": {
                "tail_span": "at or above strictly-prior 30d 95th percentile",
                "event_dispersion": "at or above strictly-prior 30d 75th percentile",
                "arrival_cv": "at or above strictly-prior 30d 75th percentile",
                "absolute_size_asymmetry": (
                    "at or above strictly-prior 30d 80th percentile"
                ),
                "release_response": (
                    "signed response at or above strictly-prior 30d 75th percentile "
                    "of absolute micro return"
                ),
            },
            "rolling_window_bars": 8_640,
            "rolling_minimum_bars": 2_016,
            "all_thresholds_shifted_bars": 1,
            "minimum_agg_trade_count": 64,
            "episode_start": (
                "current mechanism active and no same-branch active bar in the prior "
                "12 completed bars"
            ),
        },
        "policies": [asdict(policy) for policy in policy_grid()],
        "branch_rules": {
            "common": (
                "clean, agg_trade_count>=64, packet_direction!=0, tail/dispersion/"
                "arrival/absolute-size-asymmetry thresholds all pass"
            ),
            "tail_absorption_fade": (
                "common and signed_price_response<=0; side=-packet_direction"
            ),
            "tail_release_follow": (
                "common and signed_price_response>=prior q75(abs micro return); "
                "side=packet_direction"
            ),
        },
        "support_freeze_before_returns": {
            "nonoverlap_events_min_each_policy": 120,
            "nonoverlap_events_min_each_year": 25,
            "minimum_each_side_share": 0.20,
            "maximum_single_month_share": 0.20,
            "global_missing_or_quarantined_fraction_max": 0.02,
            "monthly_missing_or_quarantined_fraction_max": 0.05,
            "failure_action": "reject policy without computing forward trade returns",
        },
        "selection_protocol": {
            "fit_windows": {
                "fit_2020": ["2020-01-01", "2021-01-01"],
                "fit_2021": ["2021-01-01", "2022-01-01"],
            },
            "selection": ["2022-01-01", SELECTION_END],
            "sealed_holdout": list(HOLDOUT),
            "future_2024_plus_sealed": True,
            "selection_inputs_physically_end_before": SELECTION_END,
            "rank": [
                "descending minimum(fit_2020, fit_2021, selection_2022) CAGR/strict_MDD",
                "descending combined_2020_2022 CAGR/strict_MDD",
                "ascending policy_id",
            ],
            "selection_gates": {
                "every_calendar_year_absolute_return_positive": True,
                "positive_half_years_min_of_six": 5,
                "strict_mdd_pct_max_each_year": 10.0,
                "combined_cagr_to_strict_mdd_min": 2.0,
                "combined_trades_min": 120,
                "each_calendar_year_trades_min": 25,
                "eight_bp_notional_side_cost_stress_absolute_return_positive": True,
                "familywise_weekly_cluster_signflip_p_max": 0.10,
            },
            "holdout_2023_gates": {
                "absolute_return_positive": True,
                "cagr_to_strict_mdd_min": 3.0,
                "strict_mdd_pct_max": 10.0,
                "trades_min": 20,
                "h1_absolute_return_nonnegative": True,
                "h2_absolute_return_nonnegative": True,
                "eight_bp_notional_side_cost_stress_absolute_return_positive": True,
                "one_bar_delay_absolute_return_positive": True,
            },
            "multiple_testing_hypotheses": len(policy_grid()),
            "familywise_adjustment": "Bonferroni over all four policies",
        },
        "execution_contract": {
            "feature_bucket": "completed Binance UM five-minute aggTrade bucket t",
            "decision_time": "after t closes and feature computation completes",
            "entry_delay_bars": 2,
            "entry": (
                "Binance open at t+2, leaving one complete five-minute latency bar "
                "between feature completion and assumed fill"
            ),
            "nonoverlap": True,
            "leverage": 0.5,
            "base_cost_notional_per_side": 0.0006,
            "stress_cost_notional_per_side": 0.0008,
            "realized_funding": True,
            "strict_mdd": (
                "global/pre-entry high-water plus held favorable-before-adverse OHLC, "
                "entry/hypothetical liquidation costs, slippage, and funding"
            ),
            "cagr_clock": "full calendar including idle periods",
        },
        "controls": {
            "direction_flip": True,
            "one_additional_bar_delay": True,
            "twelve_additional_bar_delay": True,
            "branch_side_swap": True,
            "no_arrival_filter_ablation": True,
            "no_tail_filter_ablation": True,
            "time_of_week_matched_random_samples": 5_000,
            "weekly_cluster_signflip_samples": 5_000,
            "random_seed": 20260716,
        },
        "orthogonality_after_holdout": {
            "exact_entry_jaccard_max": 0.02,
            "candidate_entries_near_existing_6h_fraction_target": 0.10,
            "candidate_entries_near_existing_6h_fraction_max": 0.25,
            "position_jaccard_max": 0.15,
            "absolute_daily_pnl_pearson_max": 0.30,
            "minimum_nonzero_daily_pnl_days": 10,
            "undefined_metric": "fail_closed",
            "marginal_portfolio_improvement_required": True,
        },
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
        raise RuntimeError("tail-arrival preregistration hash mismatch")
    if payload.get("outcomes_opened") is not False:
        raise RuntimeError("tail-arrival preregistration cannot open outcomes")
    if payload.get("policies") != [asdict(policy) for policy in policy_grid()]:
        raise RuntimeError("tail-arrival policy family differs from code")


def write_manifest_once(path: str | Path, payload: dict[str, Any]) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = json.loads(output.read_text())
        validate_manifest(existing)
        if existing["manifest_hash"] != payload["manifest_hash"]:
            raise RuntimeError("refusing to overwrite tail-arrival preregistration")
        return "verified_existing"
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return "created"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_manifest()
    validate_manifest(payload)
    status = write_manifest_once(args.output, payload)
    print(
        json.dumps(
            {
                "output": args.output,
                "manifest_hash": payload["manifest_hash"],
                "policies": len(payload["policies"]),
                "status": status,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
