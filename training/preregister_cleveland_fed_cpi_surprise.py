"""Preregister CFCS-1 before loading any post-release BTC outcome."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from training import cleveland_fed_cpi_surprise_clock as clock


AS_OF_DATE = "2026-07-18"
SOURCE_COMMIT = "a6f824f"
DEFAULT_OUTPUT = "results/cleveland_fed_cpi_surprise_preregistration_2026-07-18.json"
DEFAULT_CLOCK = clock.DEFAULT_OUTPUT
DEFAULT_DOCS = "docs/cleveland-fed-cpi-surprise-preregistration-2026-07-18.md"
MARKET_PATH = (
    "data/binance_um_kline_reference_btc_2020_2023/"
    "BTCUSDT_5m_2020-01-01_2023-12-31.csv.gz"
)
MARKET_MANIFEST = "data/binance_um_kline_reference_btc_2020_2023/build_manifest.json"
MARKET_SHA256 = "e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d"
MARKET_MANIFEST_SHA256 = (
    "c04fbbd299cc748a6745c0ef030787da4d560833c744c81c98dd8840efc7913e"
)
FUNDING_PATH = "data/binance_um_btcusdt_funding_marks_2020_2023.csv.gz"
FUNDING_MANIFEST = (
    "results/binance_um_btcusdt_funding_marks_2020_2023_manifest_2026-07-17.json"
)
FUNDING_SHA256 = "3284bbb6bb67946acb673c6b67459543e217f752589e1d47b6c7c3b659f733e6"
FUNDING_MANIFEST_SHA256 = (
    "a0b2d27e1aa8cf2d9ab8cb659b598ee0a6d7bd25401c9e10ae92d1a74415845b"
)
THRESHOLD_GRID = ("0.025", "0.050", "0.075", "0.100", "0.125")


@dataclass(frozen=True)
class Policy:
    policy_id: str = "CFCS-1"
    threshold_pct: float = 0.05
    require_headline_core_sign_concordance: bool = True
    direction: str = "negative_surprise_LONG_positive_surprise_SHORT"
    decision_clock: str = "official CPI release at 08:30 America/New_York"
    entry_clock: str = "08:35 America/New_York on release day"
    exit_clock: str = "16:00 America/New_York on release day"
    leverage: float = 0.5
    base_cost_notional_per_side: float = 0.0006
    stress_cost_notional_per_side: float = 0.001


WINDOWS = {
    "2019_source_history": ("2019-01-01", "2020-01-01"),
    "2020": ("2020-01-01", "2021-01-01"),
    "2021": ("2021-01-01", "2022-01-01"),
    "2022": ("2022-01-01", "2023-01-01"),
    "stage1": ("2020-01-01", "2023-01-01"),
    "2023_h1": ("2023-01-01", "2023-07-01"),
    "2023_h2": ("2023-07-01", "2024-01-01"),
    "2023": ("2023-01-01", "2024-01-01"),
}


def canonical_json(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _gzip_bytes(payload: bytes) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", filename="", mtime=0) as handle:
        handle.write(payload)
    return output.getvalue()


def _distribution(events: list[clock.Event], start: str, end: str) -> dict[str, Any]:
    lower = pd.Timestamp(start, tz="UTC")
    upper = pd.Timestamp(end, tz="UTC")
    frame = clock.events_frame(events)
    if frame.empty:
        return {
            "events": 0,
            "longs": 0,
            "shorts": 0,
            "months": 0,
            "max_single_month_count": 0,
            "max_single_month_share": 0.0,
        }
    opened = frame.loc[frame["entry_time"].ge(lower) & frame["entry_time"].lt(upper)]
    months = opened["entry_time"].dt.strftime("%Y-%m").value_counts()
    maximum = int(months.max()) if len(months) else 0
    return {
        "events": int(len(opened)),
        "longs": int(opened["side"].eq(1).sum()),
        "shorts": int(opened["side"].eq(-1).sum()),
        "months": int(len(months)),
        "max_single_month_count": maximum,
        "max_single_month_share": maximum / len(opened) if len(opened) else 0.0,
    }


def _window_counts(events: list[clock.Event]) -> dict[str, dict[str, Any]]:
    return {
        name: _distribution(events, start, end)
        for name, (start, end) in WINDOWS.items()
    }


def _source_only_disclosure(rows: list[clock.SourceRow]) -> dict[str, Any]:
    selected: dict[str, list[clock.Event]] = {
        "primary": clock.build_events(rows),
        "headline_only": clock.build_events(rows, mode="headline_only"),
        "core_only": clock.build_events(rows, mode="core_only"),
        "composite_no_concordance": clock.build_events(
            rows, mode="composite_no_concordance"
        ),
        "one_day_delay": clock.build_events(rows, delay_days=1),
        "seven_day_placebo": clock.build_events(rows, delay_days=7),
    }
    grid = {
        threshold: _window_counts(
            clock.build_events(rows, threshold_pct=Decimal(threshold))
        )
        for threshold in THRESHOLD_GRID
    }
    return {
        "definition": (
            "Cleveland Fed nowcasts, first-release actuals, official release "
            "timestamps, surprise signs, and event counts only"
        ),
        "outcomes_joined": False,
        "outcome_sources_opened": [],
        "market_rows_loaded": 0,
        "funding_rows_loaded": 0,
        "selected_threshold_pct": Policy.threshold_pct,
        "selected_clocks": {
            name: _window_counts(events) for name, events in selected.items()
        },
        "threshold_support_grid": grid,
    }


def build_manifest(clock_output: str | Path = DEFAULT_CLOCK) -> dict[str, Any]:
    rows = clock.load_source()
    primary = clock.build_events(rows)
    frame = clock.events_frame(primary)
    write_once(
        clock_output,
        _gzip_bytes(frame.to_csv(index=False, lineterminator="\n").encode()),
    )
    disclosure = _source_only_disclosure(rows)
    core: dict[str, Any] = {
        "protocol_version": "cleveland_fed_cpi_surprise_v1",
        "policy_id": "CFCS-1",
        "as_of_date": AS_OF_DATE,
        "source_commit": SOURCE_COMMIT,
        "outcomes_opened": False,
        "opened_outcome_windows": [],
        "sealed_outcome_windows": ["stage1_2020_2022", "stage2_2023", "2024_plus"],
        "hypothesis": (
            "A concordant CPI release above the Cleveland Fed's final pre-release "
            "headline/core nowcasts is an adverse liquidity-rate shock for BTC; "
            "a concordant downside surprise is the symmetric risk-on shock."
        ),
        "research_history_boundary": {
            "global_market_history_seen_by_unrelated_research": True,
            "exact_cfcs_post_entry_outcomes_opened": False,
            "source_only_threshold_grid_inspected": list(THRESHOLD_GRID),
            "threshold_choice_rule": (
                "select the highest source-only threshold with >=24 Stage1 events, "
                ">=8 in each Stage1 year, >=8 sealed-2023 events, >=4 in each "
                "sealed half, and <=12.5% single-month concentration; 0.050 is "
                "the first passing threshold when inspected from high to low"
            ),
            "disclosure": disclosure,
            "forbidden_before_evaluator_freeze": [
                "post_release_BTC_OHLC",
                "entry_to_exit_return",
                "funding_PnL",
                "win_rate",
                "absolute_return_CAGR_or_drawdown",
                "existing_alpha_return_overlap",
            ],
        },
        "novelty_boundary": {
            "distinct_axis": (
                "monthly public inflation forecast error; no crypto or market-price "
                "state enters the signal"
            ),
            "economic_hypothesis_is_inference": (
                "actual-minus-nowcast proxies an unanticipated inflation shock; "
                "the Cleveland Fed does not claim a BTC trading implication"
            ),
            "excluded_inputs": [
                "BTC_OHLC_returns_or_regime",
                "taker_volume_open_interest_funding_premium_or_basis",
                "Kimchi_DXY_USDKRW_or_EMFX",
                "REX_Cboe_options_onchain_stablecoin_or_existing_alpha_state",
            ],
            "forbidden_repairs_after_outcomes": [
                "reverse primary direction or promote a control",
                "change threshold, concordance, release clock, entry, or exit",
                "add BTC crypto FX calendar volatility or regime gates",
                "change leverage or costs",
            ],
        },
        "source_contract": {
            "panel": str(clock.SOURCE),
            "panel_sha256": clock.SOURCE_SHA256,
            "build_manifest": str(clock.BUILD_MANIFEST),
            "build_manifest_sha256": clock.BUILD_MANIFEST_SHA256,
            "official_indicator_page": (
                "https://www.clevelandfed.org/indicators-and-data/inflation-nowcasting"
            ),
            "official_chart_json": (
                "https://www.clevelandfed.org/-/media/files/webcharts/"
                "inflationnowcasting/nowcast_month.json?sc_lang=en"
            ),
            "market": MARKET_PATH,
            "market_sha256": MARKET_SHA256,
            "market_manifest": MARKET_MANIFEST,
            "market_manifest_sha256": MARKET_MANIFEST_SHA256,
            "funding": FUNDING_PATH,
            "funding_sha256": FUNDING_SHA256,
            "funding_manifest": FUNDING_MANIFEST,
            "funding_manifest_sha256": FUNDING_MANIFEST_SHA256,
            "signal_columns_loaded": list(clock.source_builder.PANEL_COLUMNS),
            "signal_market_or_funding_rows_loaded": 0,
            "revision_boundary": (
                "frozen current Cleveland Fed chart vintage; forward timestamped "
                "collection of the final pre-release nowcast is mandatory for live"
            ),
        },
        "policy": asdict(Policy()),
        "causal_feature_contract": {
            "headline_surprise_pct": "first-release headline CPI MoM - latest pre-release headline nowcast",
            "core_surprise_pct": "first-release core CPI MoM - latest pre-release core nowcast",
            "composite_surprise_pct": "equal mean of headline and core surprises",
            "eligibility": "headline/core surprises have the same nonzero sign and abs(composite)>=0.05pp",
            "direction": {"negative": "LONG", "positive": "SHORT"},
            "price_or_derivative_feature_columns_loaded": [],
        },
        "execution_contract": {
            "signal": "official CPI release at 08:30 America/New_York",
            "entry": "08:35 America/New_York on release day",
            "exit": "16:00 America/New_York on release day",
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
            "stage1_events_min": 24,
            "each_stage1_year_min": 8,
            "stage1_long_min": 8,
            "stage1_short_min": 8,
            "sealed_2023_events_min": 8,
            "each_sealed_2023_half_min": 4,
            "maximum_single_month_share": 0.125,
            "failure_action": "reject without opening BTC outcomes",
        },
        "selection_protocol": {
            "stage1": ["2020-01-01", "2023-01-01"],
            "stage1_subperiods": {
                "2020": ["2020-01-01", "2021-01-01"],
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
            "stage1_gates": {
                "absolute_return_positive": True,
                "cagr_to_strict_mdd_at_least": 3.0,
                "strict_mdd_pct_at_most": 15.0,
                "weekly_cluster_signflip_p_at_most": 0.10,
                "minimum_trades": 24,
                "minimum_long_trades": 8,
                "minimum_short_trades": 8,
                "mean_gross_underlying_bp_at_least": 50.0,
                "stress_cost_absolute_return_positive": True,
                "each_calendar_year_absolute_return_positive": True,
                "each_calendar_year_minimum_trades": 8,
                "mechanism_control_ratio_margin_at_least": 0.25,
            },
            "stage2_gates": {
                "absolute_return_positive": True,
                "cagr_to_strict_mdd_at_least": 3.0,
                "strict_mdd_pct_at_most": 15.0,
                "weekly_cluster_signflip_p_at_most": 0.10,
                "minimum_trades": 8,
                "minimum_long_trades": 8,
                "mean_gross_underlying_bp_at_least": 50.0,
                "stress_cost_absolute_return_positive": True,
                "each_half_absolute_return_nonnegative": True,
                "each_half_minimum_trades": 4,
                "mechanism_control_ratio_margin_at_least": 0.25,
            },
        },
        "controls": {
            "mechanism": {
                "headline_only": "same threshold and clock using headline surprise only",
                "core_only": "same threshold and clock using core surprise only",
                "composite_no_concordance": "same composite threshold without sign concordance",
            },
            "falsification": {
                "direction_flip": "primary clock with every side reversed",
                "one_day_delay": "primary side at the next calendar day's 08:35-16:00 ET",
                "seven_day_placebo": "primary side seven calendar days later at 08:35-16:00 ET",
            },
            "mechanism_rejection_rule": (
                "primary CAGR/MDD must exceed every mechanism control by at least "
                "0.25; no control may replace the primary"
            ),
        },
        "orthogonality_after_performance": {
            "comparison_set": "promoted/live/shadow sleeves frozen before CFCS outcomes",
            "exact_entry_jaccard_max": 0.02,
            "candidate_entries_near_6h_fraction_max": 0.25,
            "position_time_jaccard_max": 0.15,
            "absolute_daily_pnl_pearson_max": 0.30,
            "absolute_daily_pnl_spearman_max": 0.30,
            "minimum_nonzero_daily_pnl_days": 20,
            "marginal_portfolio_improvement_required": True,
            "undefined_metric": "fail_closed",
        },
        "clock_output": str(clock_output),
        "clock_sha256": sha256_file(clock_output),
        "simulation_run": False,
    }
    return {**core, "manifest_hash": sha256_bytes(canonical_json(core))}


def validate_manifest(payload: dict[str, Any], *, verify_sources: bool = True) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != sha256_bytes(canonical_json(core)):
        raise RuntimeError("CFCS-1 preregistration hash mismatch")
    if payload.get("policy_id") != "CFCS-1":
        raise RuntimeError("CFCS-1 policy identity changed")
    if payload.get("outcomes_opened") is not False:
        raise RuntimeError("CFCS-1 preregistration opened outcomes")
    if payload.get("opened_outcome_windows") != []:
        raise RuntimeError("CFCS-1 preregistration opened a window")
    if payload.get("selection_protocol", {}).get("candidate_count") != 1:
        raise RuntimeError("CFCS-1 must remain a singleton")
    if payload.get("policy") != asdict(Policy()):
        raise RuntimeError("CFCS-1 policy differs from code")
    if (
        payload["causal_feature_contract"]["price_or_derivative_feature_columns_loaded"]
        != []
    ):
        raise RuntimeError("CFCS-1 signal uses a forbidden market feature")
    if verify_sources:
        clock.verify_source()
        if sha256_file(payload["clock_output"]) != payload["clock_sha256"]:
            raise RuntimeError("CFCS-1 preregistered clock changed")


def render_docs(report: dict[str, Any]) -> str:
    distributions = report["research_history_boundary"]["disclosure"][
        "selected_clocks"
    ]["primary"]
    rows = "\n".join(
        f"| {name} | {item['events']} | {item['longs']} | {item['shorts']} |"
        for name, item in distributions.items()
    )
    return f"""# CFCS-1 Cleveland Fed CPI-surprise preregistration

## Mechanism

CFCS-1 compares first-release headline and core CPI month-over-month values
with the last Cleveland Fed nowcasts strictly before the official release.
When both forecast errors share a sign and their equal mean is at least
0.05 percentage points in magnitude, a downside surprise goes long BTC and an
upside surprise goes short. The signal contains no crypto or market-price
feature.

## Frozen execution

- signal: official 08:30 America/New_York CPI release;
- entry: 08:35 America/New_York on the release day;
- exit: 16:00 America/New_York on the release day;
- exposure: 0.5x BTCUSDT perpetual;
- costs: 6 bp/notional/side, 10 bp stress;
- exact realized funding on `[entry, exit)`;
- full-calendar CAGR and strict intratrade MDD;
- singleton policy with no mutable parameter.

## Source-only density

| Window | Events | Long | Short |
|---|---:|---:|---:|
{rows}

The 0.05pp threshold is the highest source-only grid value retaining at least
24 Stage1 events, eight in each Stage1 year, eight in sealed 2023, and four in
each sealed half. No post-release BTC bar or funding row was loaded to choose
it. Sealed 2023 has eight long and zero short signals; that known source-side
regime imbalance cannot be repaired after outcomes.

## Controls

- headline-only, core-only, and no-concordance mechanism controls;
- exact direction flip;
- one-calendar-day delay and seven-calendar-day placebo;
- every control receives the same costs, funding, strict-MDD, subperiod, and
  significance battery.

## Sequential boundary

Stage1 may physically parse only `[2020-01-01, 2023-01-01)`. Calendar 2023 can
open only after exact replay of a passing Stage1 artifact under an immutable
evaluator. Any Stage1 failure rejects CFCS-1 unchanged. Calendar 2024+ remains
sealed. Portfolio overlap is inspected only after standalone Stage2 passes.

## Vintage limitation

The Cleveland Fed chart is a frozen current historical vintage, not a proven
immutable point-in-time archive. Forward timestamped capture of the final
pre-release nowcast is mandatory before live promotion.

## Frozen identity

- source commit: `{report["source_commit"]}`
- source panel SHA-256: `{report["source_contract"]["panel_sha256"]}`
- clock SHA-256: `{report["clock_sha256"]}`
- preregistration manifest: `{report["manifest_hash"]}`
"""


def write_once(path: str | Path, payload: bytes) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.read_bytes() != payload:
            raise RuntimeError(
                f"refusing to overwrite frozen CFCS-1 artifact: {output}"
            )
        return "verified_existing"
    output.write_bytes(payload)
    return "created"


def run(
    output: str | Path = DEFAULT_OUTPUT,
    clock_output: str | Path = DEFAULT_CLOCK,
    docs: str | Path = DEFAULT_DOCS,
) -> dict[str, Any]:
    report = build_manifest(clock_output)
    validate_manifest(report)
    write_once(output, canonical_json(report))
    write_once(docs, render_docs(report).encode())
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--clock-output", default=DEFAULT_CLOCK)
    parser.add_argument("--docs", default=DEFAULT_DOCS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run(args.output, args.clock_output, args.docs)
    print(
        json.dumps(
            {
                "policy_id": report["policy_id"],
                "outcomes_opened": False,
                "manifest_hash": report["manifest_hash"],
                "clock_sha256": report["clock_sha256"],
                "output": args.output,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
