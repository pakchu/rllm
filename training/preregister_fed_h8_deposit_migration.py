"""Preregister H8DM-1 before loading any post-release BTC outcome."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from training import fed_h8_deposit_migration_clock as clock


AS_OF_DATE = "2026-07-18"
SOURCE_COMMIT = "14ceca3"
DEFAULT_OUTPUT = "results/fed_h8_deposit_migration_preregistration_2026-07-18.json"
DEFAULT_CLOCK = clock.DEFAULT_OUTPUT
DEFAULT_DOCS = "docs/fed-h8-deposit-migration-preregistration-2026-07-18.md"
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
TAIL_QUANTILE_GRID = (0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50)


@dataclass(frozen=True)
class Policy:
    policy_id: str = "H8DM-1"
    adjustment: str = "seasonally_adjusted_archived_release_vintage"
    robust_window_releases: int = clock.ROBUST_WINDOW_RELEASES
    tail_window_releases: int = clock.TAIL_WINDOW_RELEASES
    tail_quantile: float = 0.50
    minimum_component_agreement: int = 2
    direction: str = "positive_bank_stress_SHORT_negative_bank_relief_LONG"
    decision_clock: str = "printed H.8 release date at nominal 16:15 America/New_York"
    entry_clock: str = "17:00 America/New_York on eligible release day"
    exit_clock: str = "exactly 48 hours after entry"
    eligible_release_weekdays: tuple[str, ...] = ("Thursday", "Friday")
    leverage: float = 0.5
    base_cost_notional_per_side: float = 0.0006
    stress_cost_notional_per_side: float = 0.001


WINDOWS = {
    "2017_2019_source_history": ("2017-01-01", "2020-01-01"),
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


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _gzip_bytes(payload: bytes) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", filename="", mtime=0) as handle:
        handle.write(payload)
    return output.getvalue()


def write_once(path: str | Path, payload: bytes) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.read_bytes() != payload:
            raise RuntimeError(f"refusing to overwrite frozen artifact: {output}")
        return
    output.write_bytes(payload)


def _distribution(
    events: list[clock.Event], start: str, end: str
) -> dict[str, Any]:
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


def _tail_support_passes(counts: Mapping[str, Mapping[str, Any]]) -> bool:
    stage1 = counts["stage1"]
    sealed = counts["2023"]
    return bool(
        stage1["events"] >= 72
        and min(counts[year]["events"] for year in ("2020", "2021", "2022"))
        >= 18
        and stage1["longs"] >= 24
        and stage1["shorts"] >= 24
        and stage1["max_single_month_count"] <= 4
        and sealed["events"] >= 20
        and sealed["longs"] >= 8
        and sealed["shorts"] >= 8
        and min(counts[half]["events"] for half in ("2023_h1", "2023_h2"))
        >= 8
        and sealed["max_single_month_count"] <= 3
    )


def select_tail_quantile(source: pd.DataFrame) -> tuple[float, dict[str, Any]]:
    grid: dict[str, Any] = {}
    selected: float | None = None
    for quantile in TAIL_QUANTILE_GRID:
        counts = _window_counts(clock.build_events(source, tail_quantile=quantile))
        passed = _tail_support_passes(counts)
        grid[f"{quantile:.2f}"] = {"counts": counts, "support_passed": passed}
        if selected is None and passed:
            selected = quantile
    if selected is None:
        raise ValueError("H8DM-1 source-only tail grid has no supported cell")
    if selected != Policy.tail_quantile:
        raise ValueError("H8DM-1 selected tail quantile drifted")
    return selected, grid


def _source_only_disclosure(source: pd.DataFrame) -> dict[str, Any]:
    selected, grid = select_tail_quantile(source)
    modes = {
        "primary": clock.build_events(source, tail_quantile=selected),
        "migration_only": clock.build_events(
            source, mode="migration_only", tail_quantile=selected
        ),
        "borrowings_only": clock.build_events(
            source, mode="borrowings_only", tail_quantile=selected
        ),
        "cash_only": clock.build_events(
            source, mode="cash_only", tail_quantile=selected
        ),
        "no_agreement": clock.build_events(
            source, mode="no_agreement", tail_quantile=selected
        ),
        "nsa_primary": clock.build_events(
            source, mode="nsa_primary", tail_quantile=selected
        ),
        "one_week_delay": clock.build_events(
            source, tail_quantile=selected, delay_weeks=1
        ),
        "four_week_placebo": clock.build_events(
            source, tail_quantile=selected, delay_weeks=4
        ),
    }
    return {
        "definition": (
            "archived H.8 levels, strictly-prior robust transforms, source-only "
            "event counts, release clocks, and documented source breaks only"
        ),
        "outcomes_joined": False,
        "outcome_sources_opened": [],
        "market_rows_loaded": 0,
        "funding_rows_loaded": 0,
        "selected_tail_quantile": selected,
        "tail_support_grid": grid,
        "selected_clocks": {
            name: _window_counts(events) for name, events in modes.items()
        },
    }


def build_manifest(clock_output: str | Path = DEFAULT_CLOCK) -> dict[str, Any]:
    source = clock.load_source()
    disclosure = _source_only_disclosure(source)
    selected = float(disclosure["selected_tail_quantile"])
    primary = clock.build_events(source, tail_quantile=selected)
    primary_frame = clock.events_frame(primary)
    write_once(
        clock_output,
        _gzip_bytes(primary_frame.to_csv(index=False, lineterminator="\n").encode()),
    )
    core: dict[str, Any] = {
        "protocol_version": "fed_h8_deposit_migration_v1",
        "policy_id": "H8DM-1",
        "as_of_date": AS_OF_DATE,
        "source_commit": SOURCE_COMMIT,
        "outcomes_opened": False,
        "opened_outcome_windows": [],
        "sealed_outcome_windows": ["stage1_2020_2022", "stage2_2023", "2024_plus"],
        "hypothesis": (
            "A weekly balance-sheet stress composite--large-bank other-deposit "
            "outperformance, small-bank borrowing growth, and small-bank cash "
            "contraction--is a slow risk-liquidity shock for the following 48 "
            "hours in BTC; the opposite composite is symmetric relief."
        ),
        "research_history_boundary": {
            "global_market_history_seen_by_unrelated_research": True,
            "exact_h8dm_post_entry_outcomes_opened": False,
            "source_only_tail_grid_inspected": list(TAIL_QUANTILE_GRID),
            "threshold_choice_rule": (
                "scan 0.90 down to 0.50 and select the highest strictly-source-only "
                "tail with >=72 Stage1 events, >=18/year, >=24/side, <=4/month, "
                ">=20 sealed-2023 events, >=8/side, >=8/half, and <=3/month; "
                "0.50 is the first passing cell"
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
                "weekly point-in-time commercial-bank balance-sheet composition "
                "released on a late-week public clock"
            ),
            "economic_hypothesis_is_inference": (
                "H.8 does not claim that deposit migration or bank funding predicts BTC"
            ),
            "excluded_inputs": [
                "BTC_OHLC_returns_or_regime",
                "taker_volume_open_interest_funding_premium_or_basis",
                "Kimchi_DXY_USDKRW_or_EMFX",
                "REX_Cboe_options_onchain_stablecoin_or_existing_alpha_state",
            ],
            "forbidden_repairs_after_outcomes": [
                "reverse direction or promote a control",
                "change SA vintage, component weights, robust windows, tail, or agreement",
                "change structural exclusions, release eligibility, entry, exit, leverage, or costs",
                "add BTC crypto FX calendar volatility or regime gates",
            ],
        },
        "source_contract": {
            "panel": str(clock.SOURCE),
            "panel_sha256": clock.SOURCE_SHA256,
            "build_manifest": str(clock.BUILD_MANIFEST),
            "build_manifest_sha256": clock.BUILD_MANIFEST_SHA256,
            "official_release_dates": "https://www.federalreserve.gov/releases/h8/",
            "official_about": "https://www.federalreserve.gov/releases/h8/about.htm",
            "official_technical_qa": "https://www.federalreserve.gov/releases/h8/h8_technical_qa.html",
            "official_notes": "https://www.federalreserve.gov/releases/h8/h8notes.htm",
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
                "exact dated HTML release bytes are frozen; current-vintage FRED/DDP "
                "history is forbidden"
            ),
        },
        "policy": asdict(Policy()),
        "causal_feature_contract": {
            "migration_bp": "10000*(log(large other deposits latest/prior)-log(small other deposits latest/prior))",
            "small_borrowings_bp": "10000*log(small borrowings latest/prior)",
            "small_cash_bp": "10000*log(small cash assets latest/prior)",
            "component_z": "(current-prior104_median)/(1.4826*prior104_MAD)",
            "stress_score": "equal mean(migration_z, borrowings_z, -small_cash_z)",
            "tail": "abs(score) >= prior52 absolute-score q0.50",
            "agreement": "at least two component signs agree with score sign",
            "direction": {"positive_stress": "SHORT", "negative_relief": "LONG"},
            "structural_exclusion_releases": sorted(clock.STRUCTURAL_EXCLUSION_RELEASES),
            "price_or_derivative_feature_columns_loaded": [],
        },
        "execution_contract": {
            "signal": "printed H.8 release date at nominal 16:15 America/New_York",
            "entry": "17:00 America/New_York; fixed 45-minute publication buffer",
            "exit": "exactly 48 hours after entry",
            "eligible_release_weekdays": list(Policy.eligible_release_weekdays),
            "Monday_release_action": "no_trade",
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
            "stage1_events_min": 72,
            "each_stage1_year_min": 18,
            "stage1_long_min": 24,
            "stage1_short_min": 24,
            "stage1_max_single_month_count": 4,
            "sealed_2023_events_min": 20,
            "sealed_2023_long_min": 8,
            "sealed_2023_short_min": 8,
            "each_sealed_2023_half_min": 8,
            "sealed_2023_max_single_month_count": 3,
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
                "minimum_trades": 72,
                "minimum_long_trades": 24,
                "minimum_short_trades": 24,
                "mean_gross_underlying_bp_at_least": 40.0,
                "stress_cost_absolute_return_positive": True,
                "each_calendar_year_absolute_return_positive": True,
                "each_calendar_year_minimum_trades": 18,
                "best_single_component_ratio_margin_at_least": 0.25,
            },
            "stage2_gates": {
                "absolute_return_positive": True,
                "cagr_to_strict_mdd_at_least": 3.0,
                "strict_mdd_pct_at_most": 15.0,
                "weekly_cluster_signflip_p_at_most": 0.10,
                "minimum_trades": 20,
                "minimum_long_trades": 8,
                "minimum_short_trades": 8,
                "mean_gross_underlying_bp_at_least": 40.0,
                "stress_cost_absolute_return_positive": True,
                "each_half_absolute_return_nonnegative": True,
                "each_half_minimum_trades": 8,
                "pooled_stage1_stage2_ratio_at_least": 3.0,
                "pooled_weekly_cluster_signflip_p_at_most": 0.05,
            },
            "stage2_open_condition": "all Stage1 gates pass unchanged",
            "orthogonality_open_condition": "standalone Stage2 and pooled gates pass unchanged",
        },
        "controls": {
            "mechanism": [
                "migration_only",
                "borrowings_only",
                "cash_only",
                "no_agreement",
                "nsa_primary",
            ],
            "falsification": ["direction_flip", "one_week_delay", "four_week_placebo"],
            "source_breaks_reported": sorted(clock.STRUCTURAL_EXCLUSION_RELEASES),
        },
        "orthogonality_protocol_after_standalone_pass": {
            "entry_jaccard_at_most": 0.02,
            "entry_near_existing_within_6h_share_at_most": 0.25,
            "occupied_position_bar_jaccard_at_most": 0.15,
            "absolute_daily_pnl_pearson_at_most": 0.30,
            "absolute_daily_pnl_spearman_at_most": 0.30,
        },
        "primary_clock": {
            "path": str(clock_output),
            "sha256": sha256_file(clock_output),
            "events": len(primary),
            "columns": list(clock.EVENT_COLUMNS),
        },
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate_manifest(payload: Mapping[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise ValueError("H8DM-1 preregistration manifest hash mismatch")
    if payload.get("outcomes_opened") is not False:
        raise ValueError("H8DM-1 preregistration opened outcomes")
    if payload.get("opened_outcome_windows") != []:
        raise ValueError("H8DM-1 preregistration opened an outcome window")
    disclosure = payload["research_history_boundary"]["disclosure"]
    if disclosure["market_rows_loaded"] != 0 or disclosure["funding_rows_loaded"] != 0:
        raise ValueError("H8DM-1 preregistration opened execution rows")
    if payload["policy"]["tail_quantile"] != 0.50:
        raise ValueError("H8DM-1 tail quantile changed")


def render_docs(payload: Mapping[str, Any]) -> str:
    disclosure = payload["research_history_boundary"]["disclosure"]
    counts = disclosure["selected_clocks"]["primary"]
    rows = "\n".join(
        f"| {name} | {item['events']} | {item['longs']} | {item['shorts']} |"
        for name, item in counts.items()
    )
    grid = "\n".join(
        f"| {quantile} | {item['counts']['stage1']['events']} | "
        f"{item['counts']['2023']['events']} | "
        f"{'PASS' if item['support_passed'] else 'FAIL'} |"
        for quantile, item in disclosure["tail_support_grid"].items()
    )
    return f"""# H8DM-1 preregistration — 2026-07-18

## Status

**PREREGISTERED; ALL BTC/FUNDING OUTCOMES SEALED**

This is one candidate, not a family search. No result-aware repair, direction
flip, component promotion, or new regime gate is permitted.

## Fixed mechanism

- source: point-in-time Federal Reserve H.8 dated archive pages;
- feature vintage: seasonally adjusted levels printed in each release;
- components: large-minus-small other-deposit growth, small-bank borrowings
  growth, and negative small-bank cash-asset growth;
- normalization: prior 104-release median/MAD only;
- score: equal mean of the three robust z-scores;
- event: absolute score at or above the prior-52-score q0.50, with at least two
  component signs agreeing;
- direction: positive stress SHORT, negative relief LONG;
- exclusions fixed before outcomes: {', '.join(sorted(clock.STRUCTURAL_EXCLUSION_RELEASES))};
- entry: 17:00 New York on Thursday/Friday release dates, 45 minutes after the
  nominal 16:15 publication time;
- exit: exactly 48 hours later;
- size/cost: 0.5x, 6 bp/notional/side; 10 bp stress.

## Source-only threshold selection

| Tail q | Stage1 events | Sealed 2023 events | Support |
|---:|---:|---:|:---:|
{grid}

q0.50 is the highest cell meeting the frozen density, direction, half-year,
and concentration requirements. No BTC or funding row was used.

## Frozen primary clock distribution

| Window | Events | Long | Short |
|---|---:|---:|---:|
{rows}

## Evaluation order

1. freeze source and controls;
2. freeze evaluator source hash with zero outcome rows parsed;
3. open only 2020–2022 Stage1;
4. keep 2023 sealed unless every Stage1 gate passes unchanged;
5. inspect existing-alpha overlap only after standalone Stage2 and pooled pass.

## Identity

- source commit: `{payload['source_commit']}`
- source panel SHA-256: `{payload['source_contract']['panel_sha256']}`
- primary clock SHA-256: `{payload['primary_clock']['sha256']}`
- preregistration manifest: `{payload['manifest_hash']}`
"""


def run(
    *,
    output: str | Path = DEFAULT_OUTPUT,
    clock_output: str | Path = DEFAULT_CLOCK,
    docs: str | Path = DEFAULT_DOCS,
) -> dict[str, Any]:
    payload = build_manifest(clock_output=clock_output)
    validate_manifest(payload)
    write_once(output, canonical_json(payload))
    write_once(docs, render_docs(payload).encode())
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--clock-output", default=DEFAULT_CLOCK)
    parser.add_argument("--docs", default=DEFAULT_DOCS)
    args = parser.parse_args()
    print(
        json.dumps(
            run(output=args.output, clock_output=args.clock_output, docs=args.docs),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
