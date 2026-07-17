"""Preregister CIHM-1 before opening any exact-policy BTC outcome."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from training import cboe_institutional_hedge_migration_clock as clock


DEFAULT_OUTPUT = (
    "results/cboe_institutional_hedge_migration_"
    "preregistration_2026-07-18.json"
)
SOURCE_PATH = clock.DEFAULT_SOURCE
SOURCE_MANIFEST = "data/cboe_option_flow_2020_2023/build_manifest.json"
MARKET_PATH = (
    "data/binance_um_kline_reference_btc_2020_2023/"
    "BTCUSDT_5m_2020-01-01_2023-12-31.csv.gz"
)
MARKET_MANIFEST = "data/binance_um_kline_reference_btc_2020_2023/build_manifest.json"
FUNDING_PATH = "data/binance_um_btcusdt_funding_marks_2020_2023.csv.gz"
FUNDING_MANIFEST = (
    "results/binance_um_btcusdt_funding_marks_2020_2023_manifest_2026-07-17.json"
)
COMPOSITE_THRESHOLD_GRID = (0.55, 0.56, 0.57, 0.575, 0.58, 0.59, 0.60)


@dataclass(frozen=True)
class Policy:
    policy_id: str = "CIHM-1"
    lookback_observations: int = 252
    minimum_history: int = 126
    composite_threshold: float = 0.575
    component_threshold: float = 0.70
    direction: str = "SHORT_ONLY"
    decision_clock: str = "next Cboe option-statistics date 09:35 America/New_York"
    hold_clock: str = "one Cboe option-statistics session"
    leverage: float = 0.5
    base_cost_notional_per_side: float = 0.0006
    stress_cost_notional_per_side: float = 0.0010


SOURCE_HASHES = {
    "cboe_panel_sha256": "35ef106ef01e3abadbcb4a6227187dd1d7cf2722191bd146bac06d08d1684a78",
    "cboe_manifest_sha256": "0a513b146ad5857d9ab7311e978152c308de64db8ef29c4d463eb07ea503089e",
    "cboe_manifest_hash": "07c2effecd8c67e7ddb81abf5e01620a667a52e2db02c78b742eb49b506e1bac",
    "market_sha256": "e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d",
    "market_manifest_sha256": "c04fbbd299cc748a6745c0ef030787da4d560833c744c81c98dd8840efc7913e",
    "funding_sha256": "3284bbb6bb67946acb673c6b67459543e217f752589e1d47b6c7c3b659f733e6",
    "funding_manifest_sha256": "a0b2d27e1aa8cf2d9ab8cb659b598ee0a6d7bd25401c9e10ae92d1a74415845b",
}


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def policy_payload() -> dict[str, Any]:
    return asdict(Policy())


WINDOWS = {
    "2021": ("2021-01-01T00:00:00+00:00", "2022-01-01T00:00:00+00:00"),
    "2022": ("2022-01-01T00:00:00+00:00", "2023-01-01T00:00:00+00:00"),
    "stage1": ("2021-01-01T00:00:00+00:00", "2023-01-01T00:00:00+00:00"),
    "2023_h1": ("2023-01-01T00:00:00+00:00", "2023-07-01T00:00:00+00:00"),
    "2023_h2": ("2023-07-01T00:00:00+00:00", "2024-01-01T00:00:00+00:00"),
    "2023": ("2023-01-01T00:00:00+00:00", "2024-01-01T00:00:00+00:00"),
}


def _counts(events: list[clock.Event], start: str, end: str) -> dict[str, Any]:
    selected = [event for event in events if start <= event.entry_time < end]
    months: dict[str, int] = {}
    for event in selected:
        month = event.entry_time[:7]
        months[month] = months.get(month, 0) + 1
    maximum = max(months.values(), default=0)
    return {
        "events": len(selected),
        "longs": sum(event.side == "LONG" for event in selected),
        "shorts": sum(event.side == "SHORT" for event in selected),
        "months": len(months),
        "max_single_month_count": maximum,
        "max_single_month_share": maximum / len(selected) if selected else 0.0,
    }


def _window_counts(events: list[clock.Event]) -> dict[str, dict[str, Any]]:
    return {window: _counts(events, *bounds) for window, bounds in WINDOWS.items()}


def source_only_disclosure() -> dict[str, Any]:
    rows = clock.read_source(SOURCE_PATH)
    policy = Policy()
    modes: dict[str, list[clock.Event]] = {
        "primary": clock.build_events(rows),
        "institutional_gap_only": clock.build_events(
            rows, mode="institutional_gap_only"
        ),
        "vix_call_pressure_only": clock.build_events(
            rows, mode="vix_call_pressure_only"
        ),
        "index_share_only": clock.build_events(rows, mode="index_share_only"),
        "level_composite": clock.build_events(rows, mode="level_composite"),
        "one_release_delay": clock.build_events(rows, release_delay=1),
        "seven_release_placebo": clock.build_events(rows, release_delay=7),
    }
    grid = {
        format(threshold, ".3f"): _window_counts(
            clock.build_events(rows, composite_threshold=threshold)
        )
        for threshold in COMPOSITE_THRESHOLD_GRID
    }
    return {
        "definition": (
            "Cboe option volumes, strict-prior ranks, timestamps, fixed short "
            "sides, and event counts only"
        ),
        "outcomes_joined": False,
        "market_rows_loaded": 0,
        "funding_rows_loaded": 0,
        "selected_thresholds": {
            "composite": policy.composite_threshold,
            "single_component": policy.component_threshold,
        },
        "selected_clocks": {
            name: _window_counts(events) for name, events in modes.items()
        },
        "composite_threshold_support_grid": grid,
    }


def build_manifest() -> dict[str, Any]:
    disclosure = source_only_disclosure()
    core: dict[str, Any] = {
        "protocol_version": "cboe_institutional_hedge_migration_v1",
        "as_of_date": "2026-07-18",
        "outcomes_opened": False,
        "policy": policy_payload(),
        "research_history_boundary": {
            "global_2021_2023_market_returns_seen_by_unrelated_research": True,
            "exact_cihm_post_entry_outcomes_opened": False,
            "source_only_composite_threshold_grid_inspected": list(
                COMPOSITE_THRESHOLD_GRID
            ),
            "threshold_choice_rule": (
                "select the highest source-only composite threshold with >=150 "
                "Stage1 events, >=70 events in each Stage1 year, >=60 sealed-2023 "
                "events, >=25 in each sealed half, and <=15% single-month "
                "concentration in Stage1 and 2023; 0.575 is the first passing "
                "threshold when inspected from high to low"
            ),
            "disclosure": disclosure,
            "forbidden_before_evaluator_freeze": [
                "post_entry_BTC_OHLC",
                "entry_to_exit_return",
                "funding_PnL",
                "win_rate",
                "absolute_return_CAGR_or_drawdown",
                "existing_alpha_return_overlap",
            ],
        },
        "novelty_boundary": {
            "distinct_axis": (
                "completed Cboe index/equity/VIX option-volume migration; no "
                "crypto state enters the signal clock"
            ),
            "economic_hypothesis_is_inference": (
                "a same-day rise in index put/call relative to equity, VIX call "
                "pressure, and index share proxies abrupt migration toward convex "
                "macro hedging; Cboe does not identify trader type or trade intent"
            ),
            "excluded_inputs": [
                "BTC_OHLC_returns_or_calendar_regime",
                "REX_extrema_taker_volume_or_OI",
                "funding_premium_basis_or_Kimchi",
                "DXY_USDKRW_EMFX_or_central_bank_liquidity",
                "CFTC_network_blockspace_stablecoin_or_existing_alpha_state",
                "SKEW_VVIX_VIX_levels_or_term_structure",
            ],
            "forbidden_repairs_after_outcomes": [
                "reverse the primary direction or replace it with a control",
                "add a long side",
                "change rank history, score weights, or either threshold",
                "change next-session entry or one-source-session hold",
                "add BTC FX crypto calendar implied-volatility or regime gates",
                "change leverage or costs",
            ],
        },
        "source_contract": {
            "cboe_panel": SOURCE_PATH,
            "cboe_panel_sha256": SOURCE_HASHES["cboe_panel_sha256"],
            "cboe_manifest": SOURCE_MANIFEST,
            "cboe_manifest_sha256": SOURCE_HASHES["cboe_manifest_sha256"],
            "cboe_manifest_hash": SOURCE_HASHES["cboe_manifest_hash"],
            "official_daily_url": (
                "https://www.cboe.com/us/options/market_statistics/daily/"
            ),
            "official_historical_information_url": (
                "https://www.cboe.com/us/options/market_statistics/historical_data/"
            ),
            "market": MARKET_PATH,
            "market_sha256": SOURCE_HASHES["market_sha256"],
            "market_manifest": MARKET_MANIFEST,
            "market_manifest_sha256": SOURCE_HASHES["market_manifest_sha256"],
            "funding": FUNDING_PATH,
            "funding_sha256": SOURCE_HASHES["funding_sha256"],
            "funding_manifest": FUNDING_MANIFEST,
            "funding_manifest_sha256": SOURCE_HASHES["funding_manifest_sha256"],
            "signal_columns_loaded": list(clock.SOURCE_COLUMNS),
            "signal_market_or_funding_rows_loaded": 0,
            "source_revision_boundary": (
                "frozen current Cboe web vintage with per-response hashes; "
                "next-source-session entry avoids same-close ambiguity; forward "
                "timestamped collection and schema parity are mandatory for live"
            ),
        },
        "causal_feature_contract": {
            "institutional_gap": (
                "log((index_put+0.5)/(index_call+0.5)) - "
                "log((equity_put+0.5)/(equity_call+0.5))"
            ),
            "vix_call_pressure": "log((VIX_call+0.5)/(VIX_put+0.5))",
            "index_share": "log((index_total+1)/(all_products_total+1))",
            "change": "current completed source session level minus prior source session level",
            "rank": (
                "strict-prior midrank of each change against at most 252 earlier "
                "changes; require 126; append current only after rank"
            ),
            "score": "mean of the three strict-prior change ranks with equal weights",
            "direction": {
                "score>=0.575": "SHORT",
                "otherwise": "ABSTAIN",
            },
            "price_or_derivative_feature_columns_loaded": [],
        },
        "execution_contract": {
            "source_observation": "completed Cboe option-statistics sessions",
            "decision_and_entry": (
                "next option-statistics source date at 09:35 America/New_York"
            ),
            "exit": (
                "following option-statistics source date at 09:35 America/New_York"
            ),
            "weekends_and_holidays": "source calendar only; no synthetic fill",
            "nonoverlap": True,
            "leverage": Policy.leverage,
            "base_cost": "6bp/notional/side",
            "stress_cost": "10bp/notional/side",
            "funding_interval": "entry_time <= funding_time < exit_time",
            "cagr": "full wall-clock split including idle cash",
            "strict_mdd": (
                "global/pre-entry HWM, favorable-before-adverse held OHLC, funding, "
                "entry/exit/hypothetical-liquidation costs"
            ),
        },
        "support_freeze_before_returns": {
            "stage1_events_min": 150,
            "each_stage1_year_min": 70,
            "sealed_2023_events_min": 60,
            "each_sealed_2023_half_min": 25,
            "maximum_single_month_share": 0.15,
            "failure_action": "reject without opening BTC outcomes",
        },
        "selection_protocol": {
            "stage1": ["2021-01-01", "2023-01-01"],
            "stage1_subperiods": {
                "2021": ["2021-01-01", "2022-01-01"],
                "2022": ["2022-01-01", "2023-01-01"],
            },
            "stage2": ["2023-01-01", "2024-01-01"],
            "stage2_subperiods": {
                "2023_h1": ["2023-01-01", "2023-07-01"],
                "2023_h2": ["2023-07-01", "2024-01-01"],
            },
            "sealed_after_stage2": ["2024", "2025", "2026_ytd"],
            "candidate_count": 1,
            "no_parameter_repair": True,
            "gates": {
                "absolute_return_positive": True,
                "cagr_to_strict_mdd_min": 3.0,
                "strict_mdd_pct_max": 15.0,
                "weekly_cluster_signflip_p_max": 0.10,
                "minimum_trades": 150,
                "minimum_short_trades": 150,
                "mean_gross_underlying_bp_min": 35.0,
                "stress_cost_absolute_return_positive": True,
                "each_subperiod_absolute_return_positive": True,
                "each_subperiod_minimum_trades": 70,
                "mechanism_margin_ratio_min": 0.25,
            },
            "stage2_gates": {
                "absolute_return_positive": True,
                "cagr_to_strict_mdd_min": 3.0,
                "strict_mdd_pct_max": 15.0,
                "weekly_cluster_signflip_p_max": 0.10,
                "minimum_trades": 60,
                "minimum_short_trades": 60,
                "mean_gross_underlying_bp_min": 35.0,
                "stress_cost_absolute_return_positive": True,
                "each_subperiod_absolute_return_positive": True,
                "each_subperiod_minimum_trades": 25,
                "mechanism_margin_ratio_min": 0.25,
            },
        },
        "controls": {
            "institutional_gap_only": (
                "same clock; strict-prior delta institutional-gap rank >=0.70"
            ),
            "vix_call_pressure_only": (
                "same clock; strict-prior delta VIX-call-pressure rank >=0.70"
            ),
            "index_share_only": (
                "same clock; strict-prior delta index-share rank >=0.70"
            ),
            "level_composite": (
                "same 0.575 threshold on strict-prior level ranks instead of changes"
            ),
            "direction_flip": "primary event clock with every short changed to long",
            "one_release_delay": "primary source state entered one Cboe release later",
            "seven_release_placebo": (
                "primary source state entered seven Cboe releases later"
            ),
            "mechanism_rejection_rule": (
                "primary must exceed the best component/level control CAGR/MDD by "
                "at least 0.25; no control may replace the primary"
            ),
        },
        "orthogonality_after_performance": {
            "comparison_set": "promoted/live/shadow sleeves frozen before CIHM outcomes",
            "exact_entry_jaccard_max": 0.05,
            "candidate_entries_near_6h_fraction_max": 0.25,
            "position_time_jaccard_max": 0.15,
            "absolute_daily_pnl_pearson_max": 0.30,
            "minimum_nonzero_daily_pnl_days": 20,
            "marginal_portfolio_improvement_required": True,
            "undefined_metric": "fail_closed",
        },
        "rejection_contract": (
            "any support, performance, mechanism, or orthogonality failure rejects "
            "CIHM-1 without changing formula, direction, threshold, clock, size, or costs"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate_manifest(payload: dict[str, Any], *, verify_sources: bool = True) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("CIHM-1 preregistration hash mismatch")
    if payload.get("outcomes_opened") is not False:
        raise RuntimeError("CIHM-1 preregistration opened outcomes")
    if payload.get("policy") != policy_payload():
        raise RuntimeError("CIHM-1 policy differs from code")
    if payload.get("selection_protocol", {}).get("candidate_count") != 1:
        raise RuntimeError("CIHM-1 must remain a singleton")
    if (
        payload.get("causal_feature_contract", {}).get(
            "price_or_derivative_feature_columns_loaded"
        )
        != []
    ):
        raise RuntimeError("CIHM-1 signal uses a forbidden market feature")
    if verify_sources:
        checks = {
            SOURCE_PATH: SOURCE_HASHES["cboe_panel_sha256"],
            SOURCE_MANIFEST: SOURCE_HASHES["cboe_manifest_sha256"],
            MARKET_PATH: SOURCE_HASHES["market_sha256"],
            MARKET_MANIFEST: SOURCE_HASHES["market_manifest_sha256"],
            FUNDING_PATH: SOURCE_HASHES["funding_sha256"],
            FUNDING_MANIFEST: SOURCE_HASHES["funding_manifest_sha256"],
        }
        for path, expected in checks.items():
            if sha256_file(path) != expected:
                raise RuntimeError(f"CIHM-1 frozen source changed: {path}")


def write_once(path: str | Path, payload: dict[str, Any]) -> str:
    validate_manifest(payload)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = json.loads(output.read_text())
        validate_manifest(existing)
        if existing["manifest_hash"] != payload["manifest_hash"]:
            raise RuntimeError("refusing to overwrite frozen CIHM-1 preregistration")
        return "verified_existing"
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2) + "\n")
    return "created"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--clock-output", default=clock.DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build_manifest()
    status = write_once(args.output, payload)
    events = clock.build_events(clock.read_source(SOURCE_PATH))
    clock.write_events(args.clock_output, events)
    print(
        json.dumps(
            {
                "status": status,
                "outcomes_opened": False,
                "policy_id": payload["policy"]["policy_id"],
                "manifest_hash": payload["manifest_hash"],
                "clock_events": len(events),
                "output": args.output,
                "clock_output": args.clock_output,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
