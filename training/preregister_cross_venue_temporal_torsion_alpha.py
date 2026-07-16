"""Preregister Cross-Venue Temporal Torsion before support or return inspection."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = (
    "results/cross_venue_temporal_torsion_preregistration_2026-07-16.json"
)
SELECTION_END = "2023-01-01"
HOLDOUT = ("2023-01-01", "2024-01-01")
EXPECTED_FEATURE_SHA256 = (
    "00ab6a55fc7bfeb3012584db5bc97a7d7b98dd995491acfd3f865c6bd41f92cc"
)
EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "544c2945a2b56be478a1edc4abbb93b762bda5afc32cbd0658dd6822ff6b70fa"
)
EXPECTED_SOURCE_AUDIT_SHA256 = (
    "ffe0124ac9c5c0c3f1d1c284b672618cf910dc16cae36e65c1efe79710f039af"
)


@dataclass(frozen=True, order=True)
class Policy:
    policy_id: str
    route: str
    hold_bars: int


def policy_grid() -> list[Policy]:
    policies: list[Policy] = []
    for route in ("spot_preload_um_echo", "um_preload_spot_echo"):
        for hold_bars in (6, 18):
            policies.append(Policy(f"V{len(policies) + 1:02d}", route, hold_bars))
    return sorted(policies)


def canonical_hash(payload: Any) -> str:
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def build_manifest() -> dict[str, Any]:
    core: dict[str, Any] = {
        "protocol_version": "cross_venue_temporal_torsion_v1",
        "outcomes_opened": False,
        "research_tier": "mechanically_sealed_exact_policy_not_pristine_market_history",
        "research_history_boundary": {
            "2020_2022_market_returns_globally_seen": True,
            "2023_market_returns_globally_seen_by_unrelated_repo_research": True,
            "exact_cvtt_2023_outcome_opened": False,
            "final_required_evidence": "untuned forward shadow/live observation",
        },
        "novelty_check": {
            "repo_search_before_preregistration": {
                "spot_return_centroid_minus_spot_flow_centroid_alpha_uses": 0,
                "um_return_centroid_minus_um_flow_centroid_alpha_uses": 0,
                "crossed_within_venue_flow_return_clock_alpha_uses": 0,
                "temporal_torsion_alpha_uses": 0,
            },
            "nearest_existing_families": [
                "cash late-arrival spillover propagation",
                "cash auction transfer catch-up",
                "USD-M forced-flow reversion",
            ],
            "distinct_axis": (
                "within-venue flow-to-return temporal order is crossed between Spot "
                "and USD-M; route is not selected by which venue is merely later, "
                "by basis magnitude, or by lagged flow-response magnitude"
            ),
            "forbidden_signal_dependencies": [
                "open interest",
                "funding",
                "premium index",
                "kimchi premium",
                "DXY/FX",
                "REX",
                "HHI/effective event count",
                "event-notional tail/interarrival TAAR features",
                "Coinbase venue leadership",
            ],
        },
        "economic_hypothesis": {
            "spot_preload_um_echo": (
                "Spot aggressive flow occurs before Spot price response while USD-M "
                "price response occurs before same-bar USD-M flow; the crossed clock "
                "is consistent with cash information propagating into derivatives, "
                "so follow the Spot flow direction"
            ),
            "um_preload_spot_echo": (
                "USD-M aggressive flow occurs before USD-M price response while Spot "
                "price response occurs before same-bar Spot flow; follow the USD-M "
                "flow direction"
            ),
            "not_claimed": [
                "participant identity",
                "causal proof of information transfer",
                "order-book queue state",
                "future price direction from centroid values alone",
            ],
        },
        "source_contract": {
            "feature_source": (
                "data/binance_cross_venue_minute_leadership_btc_2020_2023/"
                "BTCUSDT_cross_venue_minute_leadership_5m_2020-01_2023-12.csv.gz"
            ),
            "feature_sha256": EXPECTED_FEATURE_SHA256,
            "feature_manifest": (
                "data/binance_cross_venue_minute_leadership_btc_2020_2023/"
                "build_manifest.json"
            ),
            "feature_manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA256,
            "source_audit": (
                "results/binance_cross_venue_minute_leadership_audit_2026-07-14.json"
            ),
            "source_audit_sha256": EXPECTED_SOURCE_AUDIT_SHA256,
            "source_provenance": (
                "official Binance Spot and USD-M monthly one-minute kline archives "
                "with archive checksums verified"
            ),
            "selection_prefix": (
                "physically stop before parsing any 2023 non-date feature value"
            ),
            "source_gap_policy": (
                "invalid current feature bucket and following 24 five-minute bars "
                "are quarantined; never impute"
            ),
            "live_parity": (
                "build each completed five-minute bucket from five complete Spot and "
                "USD-M one-minute candles, then compute the identical weighted "
                "flow/absolute-return centroids"
            ),
        },
        "feature_contract": {
            "centroid_definition": (
                "normalized weighted mean minute position in [0,1] inside the five-"
                "minute bucket; flow weights are absolute taker-flow imbalance and "
                "return weights are absolute one-minute log returns"
            ),
            "spot_flow_to_return_delay": (
                "spot_return_time_centroid - spot_flow_time_centroid"
            ),
            "um_flow_to_return_delay": (
                "um_return_time_centroid - um_flow_time_centroid"
            ),
            "spot_preload": "max(spot_flow_to_return_delay, 0)",
            "spot_echo": "max(-spot_flow_to_return_delay, 0)",
            "um_preload": "max(um_flow_to_return_delay, 0)",
            "um_echo": "max(-um_flow_to_return_delay, 0)",
            "spot_to_um_score": "sqrt(spot_preload * um_echo)",
            "um_to_spot_score": "sqrt(um_preload * spot_echo)",
            "source_side": {
                "spot_preload_um_echo": "sign(spot_flow_fraction)",
                "um_preload_spot_echo": "sign(um_flow_fraction)",
            },
            "directional_confirmation": (
                "source side is nonzero; destination flow has the same sign; and "
                "both completed-bar venue returns have the source-side sign"
            ),
            "route_threshold": (
                "route score at or above its strictly-prior rolling 30-day 95th "
                "percentile over clean directionally-confirmed crossed-clock bars"
            ),
            "rolling_window_bars": 8_640,
            "rolling_minimum_bars": 2_016,
            "all_thresholds_shifted_bars": 1,
            "episode_start": (
                "current route active and no same-route active bar in the prior 12 "
                "completed buckets"
            ),
        },
        "policies": [asdict(policy) for policy in policy_grid()],
        "route_rules": {
            "spot_preload_um_echo": (
                "clean, spot delay>0, UM delay<0, directional confirmation, and "
                "spot_to_um_score>=strictly-prior q95; side=Spot flow sign"
            ),
            "um_preload_spot_echo": (
                "clean, UM delay>0, Spot delay<0, directional confirmation, and "
                "um_to_spot_score>=strictly-prior q95; side=USD-M flow sign"
            ),
        },
        "support_freeze_before_returns": {
            "nonoverlap_events_min_each_policy": 600,
            "nonoverlap_events_min_each_year": 150,
            "minimum_each_side_share": 0.35,
            "maximum_single_month_share": 0.10,
            "global_missing_or_quarantined_fraction_max": 0.01,
            "monthly_missing_or_quarantined_fraction_max": 0.03,
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
            "rank": [
                "descending minimum annual CAGR/strict_MDD over 2020, 2021, 2022",
                "descending combined 2020-2022 CAGR/strict_MDD",
                "ascending policy_id",
            ],
            "selection_gates": {
                "every_calendar_year_absolute_return_positive": True,
                "positive_half_years_min_of_six": 5,
                "strict_mdd_pct_max_each_year": 12.0,
                "combined_cagr_to_strict_mdd_min": 2.0,
                "combined_trades_min": 600,
                "each_calendar_year_trades_min": 150,
                "eight_bp_notional_side_cost_stress_absolute_return_positive": True,
                "familywise_weekly_cluster_signflip_p_max": 0.10,
            },
            "holdout_2023_gates": {
                "absolute_return_positive": True,
                "cagr_to_strict_mdd_min": 3.0,
                "strict_mdd_pct_max": 10.0,
                "trades_min": 150,
                "h1_absolute_return_nonnegative": True,
                "h2_absolute_return_nonnegative": True,
                "eight_bp_notional_side_cost_stress_absolute_return_positive": True,
                "one_bar_delay_absolute_return_positive": True,
            },
            "multiple_testing_hypotheses": len(policy_grid()),
            "familywise_adjustment": "Bonferroni over all four frozen policies",
        },
        "execution_contract": {
            "feature_bucket": "five complete Spot/USD-M one-minute candles in bucket t",
            "feature_available": "t plus five minutes",
            "entry_delay_bars_from_bucket_open": 2,
            "entry": (
                "Binance USD-M open at t+10 minutes, leaving one complete five-minute "
                "latency bucket after feature availability"
            ),
            "holds_bars": [6, 18],
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
            "route_side_swap": True,
            "one_additional_bar_delay": True,
            "twelve_additional_bar_delay": True,
            "aggregate_flow_without_crossed_clock": True,
            "same_venue_preload_only": True,
            "same_venue_echo_only": True,
            "time_of_week_matched_random_samples": 5_000,
            "weekly_cluster_signflip_samples": 5_000,
            "random_seed": 20260716,
        },
        "orthogonality_after_holdout": {
            "baseline": "full frozen portfolio, not a single representative alpha",
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
        raise RuntimeError("CVTT preregistration hash mismatch")
    if payload.get("outcomes_opened") is not False:
        raise RuntimeError("CVTT preregistration cannot open outcomes")
    if payload.get("policies") != [asdict(policy) for policy in policy_grid()]:
        raise RuntimeError("CVTT policy family differs from code")


def write_manifest_once(path: str | Path, payload: dict[str, Any]) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = json.loads(output.read_text())
        validate_manifest(existing)
        if existing["manifest_hash"] != payload["manifest_hash"]:
            raise RuntimeError("refusing to overwrite CVTT preregistration")
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
