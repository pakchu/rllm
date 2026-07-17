"""Preregister CTHD-1 before opening any exact-policy BTC outcome."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from training import cboe_tail_hedge_disagreement_clock as clock


DEFAULT_OUTPUT = "results/cboe_tail_hedge_disagreement_preregistration_2026-07-18.json"
SOURCE_PATH = clock.DEFAULT_SOURCE
SOURCE_MANIFEST = "data/cboe_tail_risk_2018_2023/build_manifest.json"
MARKET_PATH = (
    "data/binance_um_kline_reference_btc_2020_2023/"
    "BTCUSDT_5m_2020-01-01_2023-12-31.csv.gz"
)
MARKET_MANIFEST = "data/binance_um_kline_reference_btc_2020_2023/build_manifest.json"
FUNDING_PATH = "data/binance_um_btcusdt_funding_marks_2020_2023.csv.gz"
FUNDING_MANIFEST = (
    "results/binance_um_btcusdt_funding_marks_2020_2023_manifest_2026-07-17.json"
)
TAIL_GRID = (0.10, 0.125, 0.15, 0.175, 0.20, 0.225, 0.25)


@dataclass(frozen=True)
class Policy:
    policy_id: str = "CTHD-1"
    lookback_observations: int = 252
    minimum_history: int = 126
    upper_tail_rank: float = 0.225
    direction: str = "SHORT_ONLY"
    decision_clock: str = "next Cboe observation date 09:35 America/New_York"
    hold_clock: str = "one Cboe source session"
    leverage: float = 0.5
    base_cost_notional_per_side: float = 0.0006
    stress_cost_notional_per_side: float = 0.0010


SOURCE_HASHES = {
    "cboe_panel_sha256": "cdde3f8d4bb1e23d00b192f5f9ef759aefba9087be5fd60653e9c02479dfa41a",
    "cboe_manifest_sha256": "9ef80ef3034c93d97c5b2a8160b2502527287d570d15f9d7166d631d9866c7bd",
    "cboe_manifest_hash": "091ddf3050035156814fe168e1edcac193e23cca9f39a3ef0140bcb5f8265d72",
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
    selected_tail = Policy.upper_tail_rank
    modes: dict[str, list[clock.Event]] = {
        "primary": clock.build_events(rows, upper_tail=selected_tail),
        "skew_only": clock.build_events(
            rows, mode="skew_only", upper_tail=selected_tail
        ),
        "vvix_relative_only": clock.build_events(
            rows, mode="vvix_relative_only", upper_tail=selected_tail
        ),
        "low_vix_only": clock.build_events(
            rows, mode="low_vix_only", upper_tail=selected_tail
        ),
        "tail_pair_only": clock.build_events(
            rows, mode="tail_pair_only", upper_tail=selected_tail
        ),
        "one_release_delay": clock.build_events(
            rows, upper_tail=selected_tail, release_delay=1
        ),
        "seven_release_placebo": clock.build_events(
            rows, upper_tail=selected_tail, release_delay=7
        ),
    }
    grid = {
        format(tail, ".3f"): _window_counts(
            clock.build_events(rows, upper_tail=tail)
        )
        for tail in TAIL_GRID
    }
    return {
        "definition": (
            "Cboe values, strict-prior ranks, timestamps, fixed short sides, "
            "and counts only"
        ),
        "outcomes_joined": False,
        "market_rows_loaded": 0,
        "funding_rows_loaded": 0,
        "selected_clocks": {
            name: _window_counts(events) for name, events in modes.items()
        },
        "upper_tail_support_grid": grid,
    }


def build_manifest() -> dict[str, Any]:
    disclosure = source_only_disclosure()
    core: dict[str, Any] = {
        "protocol_version": "cboe_tail_hedge_disagreement_v1",
        "as_of_date": "2026-07-18",
        "outcomes_opened": False,
        "policy": policy_payload(),
        "research_history_boundary": {
            "global_2021_2023_market_returns_seen_by_unrelated_research": True,
            "exact_cthd_post_entry_outcomes_opened": False,
            "source_only_upper_tail_grid_inspected": list(TAIL_GRID),
            "threshold_choice_rule": (
                "select the sparsest upper tail with >=150 Stage1 events, >=30 "
                "events per Stage1 year, >=140 source-only 2023 events, >=20 per "
                "2023 half, and <=16% single-month concentration in Stage1 and "
                "2023; 0.225 is the first passing tail"
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
                "SPX option-implied tail hedging and volatility-of-volatility "
                "relative to visible VIX; no crypto state enters the signal clock"
            ),
            "excluded_inputs": [
                "BTC_OHLC_returns_or_calendar_regime",
                "REX_extrema_taker_volume_or_OI",
                "funding_premium_basis_or_Kimchi",
                "DXY_USDKRW_EMFX_or_central_bank_liquidity",
                "CFTC_network_blockspace_stablecoin_or_existing_alpha_state",
            ],
            "forbidden_repairs_after_outcomes": [
                "reverse the primary direction or replace it with a control",
                "add a long side",
                "change either strict-prior rank layer or the 22.5% upper tail",
                "change next-session entry or one-source-session hold",
                "add BTC FX crypto calendar or regime gates",
                "change leverage or costs",
            ],
        },
        "source_contract": {
            "cboe_panel": SOURCE_PATH,
            "cboe_panel_sha256": SOURCE_HASHES["cboe_panel_sha256"],
            "cboe_manifest": SOURCE_MANIFEST,
            "cboe_manifest_sha256": SOURCE_HASHES["cboe_manifest_sha256"],
            "cboe_manifest_hash": SOURCE_HASHES["cboe_manifest_hash"],
            "official_history_urls": {
                symbol: (
                    "https://cdn.cboe.com/api/global/us_indices/daily_prices/"
                    f"{symbol}_History.csv"
                )
                for symbol in ("SKEW", "VVIX", "VIX")
            },
            "official_methodology": {
                "SKEW": "https://cdn.cboe.com/resources/indices/documents/SKEWwhitepaperjan2011.pdf",
                "VVIX": "https://cdn.cboe.com/resources/indices/documents/vvix-termstructure.pdf",
                "SKEW_2025_version_notice": (
                    "https://cdn.cboe.com/resources/release_notes/2025/"
                    "Consultation-Results-Regarding-Proposed-Changes-to-the-Cboe-SKEW-Index-SKEW-.pdf"
                ),
            },
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
                "frozen current Cboe history vintage; next-source-session clock "
                "avoids same-close availability ambiguity; forward-vintage and "
                "active-methodology parity are mandatory before live promotion"
            ),
        },
        "causal_feature_contract": {
            "skew_level": "log(SKEW_close / 100)",
            "vvix_relative": "log(VVIX_close / VIX_close)",
            "vix_level": "log(VIX_close)",
            "first_layer_rank": (
                "strict-prior midrank of each input against at most 252 earlier "
                "Cboe observations; require 126; append current only after rank"
            ),
            "hidden_pressure": (
                "0.5 * (skew_rank + vvix_relative_rank) - vix_level_rank"
            ),
            "second_layer_rank": (
                "strict-prior midrank of hidden_pressure against at most 252 earlier "
                "available pressure observations; require 126; append after rank"
            ),
            "direction": {
                "hidden_pressure_rank>=0.775": (
                    "SHORT: tail hedging and volatility uncertainty are elevated "
                    "relative to visible VIX"
                ),
                "otherwise": "ABSTAIN",
            },
            "price_or_derivative_feature_columns_loaded": [],
        },
        "execution_contract": {
            "source_observation": "completed Cboe trading-day closes",
            "decision_and_entry": "next source observation date at 09:35 America/New_York",
            "exit": "following source observation date at 09:35 America/New_York",
            "weekends_and_holidays": "source calendar only; no synthetic date fill",
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
            "each_stage1_year_min": 30,
            "sealed_2023_events_min": 140,
            "each_sealed_2023_half_min": 20,
            "maximum_single_month_share": 0.16,
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
                "each_subperiod_minimum_trades": 30,
                "mechanism_margin_ratio_min": 0.25,
            },
            "stage2_gates": {
                "absolute_return_positive": True,
                "cagr_to_strict_mdd_min": 3.0,
                "strict_mdd_pct_max": 15.0,
                "weekly_cluster_signflip_p_max": 0.10,
                "minimum_trades": 100,
                "minimum_short_trades": 100,
                "mean_gross_underlying_bp_min": 35.0,
                "stress_cost_absolute_return_positive": True,
                "each_subperiod_absolute_return_positive": True,
                "each_subperiod_minimum_trades": 20,
                "mechanism_margin_ratio_min": 0.25,
            },
        },
        "controls": {
            "skew_only": "same tail and clock using only strict-prior SKEW rank",
            "vvix_relative_only": "same tail and clock using only strict-prior log(VVIX/VIX) rank",
            "low_vix_only": "same tail and clock using only inverse strict-prior VIX-level rank",
            "tail_pair_only": "same tail and clock using mean SKEW and VVIX/VIX ranks without VIX subtraction",
            "direction_flip": "primary event clock with every short changed to long",
            "one_release_delay": "primary source state entered one Cboe release later",
            "seven_release_placebo": "primary source state entered seven Cboe releases later",
            "mechanism_rejection_rule": (
                "primary must exceed the best source-component control CAGR/MDD by "
                "at least 0.25; no control may replace the primary"
            ),
        },
        "orthogonality_after_performance": {
            "comparison_set": "promoted/live/shadow sleeves frozen before CTHD outcomes",
            "exact_entry_jaccard_max": 0.02,
            "candidate_entries_near_6h_fraction_max": 0.25,
            "position_time_jaccard_max": 0.15,
            "absolute_daily_pnl_pearson_max": 0.30,
            "minimum_nonzero_daily_pnl_days": 20,
            "marginal_portfolio_improvement_required": True,
            "undefined_metric": "fail_closed",
        },
        "rejection_contract": (
            "any support, performance, mechanism, or orthogonality failure rejects "
            "CTHD-1 without changing formula, direction, threshold, clock, size, or costs"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate_manifest(payload: dict[str, Any], *, verify_sources: bool = True) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("CTHD-1 preregistration hash mismatch")
    if payload.get("outcomes_opened") is not False:
        raise RuntimeError("CTHD-1 preregistration opened outcomes")
    if payload.get("policy") != policy_payload():
        raise RuntimeError("CTHD-1 policy differs from code")
    if payload.get("selection_protocol", {}).get("candidate_count") != 1:
        raise RuntimeError("CTHD-1 must remain a singleton")
    if (
        payload.get("causal_feature_contract", {}).get(
            "price_or_derivative_feature_columns_loaded"
        )
        != []
    ):
        raise RuntimeError("CTHD-1 signal uses a forbidden market feature")
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
                raise RuntimeError(f"CTHD-1 frozen source changed: {path}")


def write_once(path: str | Path, payload: dict[str, Any]) -> str:
    validate_manifest(payload)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = json.loads(output.read_text())
        validate_manifest(existing)
        if existing["manifest_hash"] != payload["manifest_hash"]:
            raise RuntimeError("refusing to overwrite frozen CTHD-1 preregistration")
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
