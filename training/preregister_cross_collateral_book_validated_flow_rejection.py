"""Outcome-blind preregistration and support qualification for CBFR-72.

CBFR-72 fades a completed five-minute aggressive taker-flow impulse only when
that impulse failed to move the completed bar in its own direction and both
USD-M and COIN-M book-credibility panels support the opposite side.  This
module may inspect signal-time inputs and prior event clocks, but it must not
inspect any post-entry price, return, funding, PnL, equity, CAGR, or drawdown.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from training.preregister_cross_collateral_liquidity_credibility_fracture import (
    Config as CredibilityConfig,
    _quarterly_schedule as pdf_quarterly_schedule,
    _venue_features,
    build_signal as build_pdf_signal,
    load_credibility,
)
from training.preregister_metaorder_fragmentation_impact_curvature import (
    nonoverlapping_schedule,
)


POLICY_ID = "CBFR-72"
SELECTION_START = pd.Timestamp("2023-01-01 00:00:00")
SELECTION_END = pd.Timestamp("2024-01-01 00:00:00")
FIVE_MINUTES = pd.Timedelta(minutes=5)

CREDIBILITY_MANIFEST = Path(
    "results/binance_cross_collateral_book_credibility_btc_2023_manifest.json"
)
CREDIBILITY_DATA = Path(
    "data/binance_cross_collateral_book_credibility_btc_2023/"
    "BTC_cross_collateral_book_credibility_5m_2023.csv.gz"
)
MARKET_MANIFEST = Path(
    "data/binance_um_kline_reference_btc_2020_2023/build_manifest.json"
)
MARKET_DATA = Path(
    "data/binance_um_kline_reference_btc_2020_2023/"
    "BTCUSDT_5m_2020-01-01_2023-12-31.csv.gz"
)

CREDIBILITY_MANIFEST_SHA256 = (
    "f530f472765c8cb56bf564efd346c734e2404e072b87e2ff8dc3b84e303c30f7"
)
CREDIBILITY_DATA_SHA256 = (
    "45026cc02620d9a0c67f250804f2a06705bf0e824f72257d6c2414f40ab7d429"
)
MARKET_MANIFEST_SHA256 = (
    "c04fbbd299cc748a6745c0ef030787da4d560833c744c81c98dd8840efc7913e"
)
MARKET_DATA_SHA256 = (
    "e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d"
)

PRIOR_CLOCKS = {
    "crrc72": Path(
        "results/cross_venue_radial_refill_compression_event_clock_2026-07-17.json"
    ),
    "cspr": Path("results/cash_sponsored_perp_rejection_clock_2026-07-14.csv"),
    "umfr36": Path("results/um_forced_flow_reversion_clock_2026-07-14.csv"),
}


@dataclass(frozen=True)
class Config:
    output: str = (
        "results/cross_collateral_book_validated_flow_rejection_"
        "support_2026-07-18.json"
    )
    event_clock_output: str = (
        "results/cross_collateral_book_validated_flow_rejection_"
        "event_clock_2026-07-18.json"
    )
    robust_baseline_bars: int = 8_640
    robust_min_periods: int = 2_016
    flow_quantiles: tuple[float, ...] = (0.75, 0.80, 0.85)
    defense_thresholds: tuple[float, ...] = (0.00, 0.25, 0.50, 0.75)
    hold_bars: int = 72
    minimum_events: int = 120
    minimum_half_events: int = 50
    minimum_quarter_events: int = 20
    minimum_side_share: float = 0.35
    maximum_quarter_share: float = 0.40
    overlap_tolerance_bars: int = 12
    maximum_prior_jaccard: float = 0.20
    maximum_new_clock_containment: float = 0.30


def protocol() -> dict[str, Any]:
    """Return the immutable, outcome-blind CBFR-72 research contract."""
    cfg = Config()
    return {
        "policy_id": POLICY_ID,
        "hypothesis": (
            "an unusually one-sided completed-bar taker flow that fails to move "
            "price in its own direction is more likely to mean absorption when "
            "both collateral venues show credible opposite-side replenishment"
        ),
        "evidence_boundary": {
            "allowed_before_clock_freeze": [
                "completed signal-bar open and close",
                "completed signal-bar taker-buy and total quote volume",
                "completed UM and CM book-credibility summaries",
                "strictly lagged feature baselines",
                "prior alpha event timestamps and sides",
            ],
            "forbidden_before_evaluator_freeze": [
                "entry or later OHLC",
                "post-entry return",
                "funding paid during the trade",
                "PnL, equity, CAGR, drawdown, hit rate, or payoff",
            ],
            "post_entry_outcomes_opened": False,
        },
        "clock": {
            "bar_label": "UTC five-minute open timestamp t",
            "signal_inputs_complete": "t+5m",
            "entry": "next five-minute open at t+5m",
            "exit": "open 72 bars after entry (six hours)",
            "quarter_contained": True,
            "nonoverlapping_within_each_quarter": True,
        },
        "feature_formula": {
            "flow": "2 * taker_buy_quote / quote_asset_volume - 1",
            "flow_threshold": {
                "statistic": "rolling quantile(abs(flow))",
                "window_rows": cfg.robust_baseline_bars,
                "minimum_rows": cfg.robust_min_periods,
                "shift_rows": 1,
                "current_and_future_rows_excluded": True,
            },
            "flow_direction": "sign(flow)",
            "completed_bar_return": "log(close / open)",
            "rejection": "flow_direction * completed_bar_return <= 0",
            "venue_defense": "-flow_direction * venue_credibility",
            "cross_collateral_defense": "mean(UM defense, CM defense)",
            "venue_agreement": "UM defense > 0 AND CM defense > 0",
            "action": "side = -flow_direction",
        },
        "support_selection": {
            "flow_quantiles": list(cfg.flow_quantiles),
            "defense_thresholds": list(cfg.defense_thresholds),
            "outcomes_used": False,
            "selection_rule": (
                "among cells passing every incidence gate, maximize defense "
                "threshold first, then flow quantile; no return tie-break"
            ),
            "incidence_pilot_disclosed": True,
        },
        "support_gate": {
            "events_at_least": cfg.minimum_events,
            "events_each_half_at_least": cfg.minimum_half_events,
            "events_each_quarter_at_least": cfg.minimum_quarter_events,
            "each_side_share_at_least": cfg.minimum_side_share,
            "largest_quarter_share_at_most": cfg.maximum_quarter_share,
        },
        "independence_gate": {
            "prior_clocks": ["PDF-10", "CRRC-72", "CSPR", "UMFR-36"],
            "tolerance_bars": cfg.overlap_tolerance_bars,
            "jaccard_at_most": cfg.maximum_prior_jaccard,
            "new_clock_containment_at_most": cfg.maximum_new_clock_containment,
            "post_entry_return_or_pnl_forbidden": True,
        },
        "eventual_execution": {
            "instrument": "Binance BTCUSDT USD-M perpetual",
            "leverage": 0.5,
            "cost_bp_per_notional_side": 6.0,
            "funding": "realized funding cash flows on the held path",
            "strict_mdd": "global pre-entry high-water mark plus held OHLC path",
            "cagr": "full declared wall-clock, including idle periods",
        },
        "sequential_oos": {
            "first_opened_stage": "calendar 2023 only, after evaluator freeze",
            "2024_plus_sealed": True,
            "failed_stage_action": "retire this exact policy without repair",
        },
    }


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _validate_config(cfg: Config) -> None:
    if cfg != Config(output=cfg.output, event_clock_output=cfg.event_clock_output):
        raise ValueError("CBFR-72 signal/support configuration is frozen")


def lagged_flow_threshold(
    absolute_flow: pd.Series,
    *,
    quantile: float,
    window: int,
    minimum: int,
) -> pd.Series:
    """Compute a causal threshold whose baseline ends at t-1."""
    return (
        absolute_flow.rolling(window=window, min_periods=minimum)
        .quantile(quantile)
        .shift(1)
    )


def load_support_inputs(cfg: Config) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load signal-time fields only and enforce frozen source identities."""
    _validate_config(cfg)
    if _sha256(CREDIBILITY_MANIFEST) != CREDIBILITY_MANIFEST_SHA256:
        raise ValueError("credibility manifest hash mismatch")
    if _sha256(CREDIBILITY_DATA) != CREDIBILITY_DATA_SHA256:
        raise ValueError("credibility data hash mismatch")
    if _sha256(MARKET_MANIFEST) != MARKET_MANIFEST_SHA256:
        raise ValueError("market manifest hash mismatch")
    if _sha256(MARKET_DATA) != MARKET_DATA_SHA256:
        raise ValueError("market data hash mismatch")

    market_manifest = json.loads(MARKET_MANIFEST.read_text())
    source_protocol = market_manifest.get("protocol", {})
    if source_protocol.get("source") != "official Binance USD-M daily kline archives":
        raise ValueError("unexpected market source")
    if source_protocol.get("archive_checksums_verified") is not True:
        raise ValueError("market archives were not checksum verified")
    if market_manifest.get("combined_sha256") != MARKET_DATA_SHA256:
        raise ValueError("market manifest points to different bytes")

    credibility, credibility_source = load_credibility(CredibilityConfig())
    market = pd.read_csv(
        MARKET_DATA,
        compression="gzip",
        usecols=["date", "open", "close", "quote_asset_volume", "taker_buy_quote"],
        parse_dates=["date"],
    )
    market = market.loc[
        market["date"].ge(SELECTION_START) & market["date"].lt(SELECTION_END)
    ].reset_index(drop=True)
    expected_dates = pd.date_range(
        SELECTION_START, SELECTION_END - FIVE_MINUTES, freq=FIVE_MINUTES
    )
    if not market["date"].equals(pd.Series(expected_dates, name="date")):
        raise ValueError("market source is not a complete UTC five-minute grid")
    numeric = market[["open", "close", "quote_asset_volume", "taker_buy_quote"]]
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError("signal-time market fields contain non-finite values")
    if (market[["open", "close"]] <= 0.0).any().any():
        raise ValueError("signal-time prices must be positive")
    if (market[["quote_asset_volume", "taker_buy_quote"]] < 0.0).any().any():
        raise ValueError("signal-time volumes must be non-negative")
    if (
        market["quote_asset_volume"].gt(0.0)
        & market["taker_buy_quote"].gt(market["quote_asset_volume"])
    ).any():
        raise ValueError("taker buy quote volume is outside total quote volume")

    frame = credibility.merge(market, on="date", how="inner", validate="one_to_one")
    if len(frame) != len(expected_dates):
        raise ValueError("book and market panels do not share the full 2023 grid")
    frame["quarantined"] = (
        ~frame["source_complete"].astype(bool)
        | frame["quote_asset_volume"].le(0.0)
    )
    return frame, {
        **credibility_source,
        "market_manifest_sha256": _sha256(MARKET_MANIFEST),
        "market_data_sha256": _sha256(MARKET_DATA),
        "market_columns_loaded": [
            "date",
            "open",
            "close",
            "quote_asset_volume",
            "taker_buy_quote",
        ],
        "entry_or_later_ohlc_loaded": False,
        "high_or_low_loaded": False,
        "post_entry_return_funding_pnl_or_equity_loaded": False,
    }


def completed_bar_flow(frame: pd.DataFrame) -> pd.Series:
    """Return signed taker flow, leaving zero-volume bars unavailable."""
    total = frame["quote_asset_volume"].where(frame["quote_asset_volume"].gt(0.0))
    return 2.0 * frame["taker_buy_quote"] / total - 1.0


def build_feature_panel(frame: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    credibility_cfg = CredibilityConfig(
        robust_baseline_bars=cfg.robust_baseline_bars,
        robust_min_periods=cfg.robust_min_periods,
    )
    um = _venue_features(frame, "um", credibility_cfg)
    cm = _venue_features(frame, "cm", credibility_cfg)
    flow = completed_bar_flow(frame)
    direction = np.sign(flow.fillna(0.0)).astype(np.int8)
    completed_bar_return = np.log(frame["close"] / frame["open"])
    um_defense = -direction * um["credibility"]
    cm_defense = -direction * cm["credibility"]
    defense = 0.5 * (um_defense + cm_defense)
    finite = np.isfinite(
        pd.concat(
            [flow, completed_bar_return, um_defense, cm_defense, defense], axis=1
        ).to_numpy(float)
    ).all(axis=1)
    clean = frame["source_complete"].astype(bool) & finite & direction.ne(0)
    return pd.DataFrame(
        {
            "date": frame["date"],
            "flow": flow,
            "direction": direction,
            "completed_bar_return": completed_bar_return,
            "um_defense": um_defense,
            "cm_defense": cm_defense,
            "defense": defense,
            "clean": clean,
        }
    )


def build_signal(
    panel: pd.DataFrame,
    frame: pd.DataFrame,
    cfg: Config,
    *,
    flow_quantile: float,
    defense_threshold: float,
) -> pd.DataFrame:
    threshold = lagged_flow_threshold(
        panel["flow"].abs(),
        quantile=flow_quantile,
        window=cfg.robust_baseline_bars,
        minimum=cfg.robust_min_periods,
    )
    candidate = (
        panel["clean"]
        & threshold.notna()
        & panel["flow"].abs().ge(threshold)
        & (panel["direction"] * panel["completed_bar_return"]).le(0.0)
        & panel["um_defense"].gt(0.0)
        & panel["cm_defense"].gt(0.0)
        & panel["defense"].ge(defense_threshold)
    )
    side = pd.Series(0, index=panel.index, dtype=np.int8)
    side.loc[candidate] = -panel.loc[candidate, "direction"].astype(np.int8)
    branch = pd.Series("none", index=panel.index, dtype="string")
    branch.loc[side.gt(0)] = "sell_flow_rejected_by_credible_bids"
    branch.loc[side.lt(0)] = "buy_flow_rejected_by_credible_asks"
    return pd.DataFrame(
        {
            "date": panel["date"],
            "side": side,
            "branch": branch,
            "hold_bars": np.where(side.ne(0), cfg.hold_bars, 0).astype(np.int16),
            "quarantined": frame["quarantined"].astype(bool),
        }
    )


def quarterly_schedule(signal: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    schedules: list[pd.DataFrame] = []
    for quarter, start, end in (
        ("q1", "2023-01-01", "2023-04-01"),
        ("q2", "2023-04-01", "2023-07-01"),
        ("q3", "2023-07-01", "2023-10-01"),
        ("q4", "2023-10-01", "2024-01-01"),
    ):
        scheduled = nonoverlapping_schedule(signal, frame, start=start, end=end)
        if not scheduled.empty:
            scheduled.insert(0, "quarter", quarter)
            schedules.append(scheduled)
    if not schedules:
        return pd.DataFrame(
            columns=[
                "quarter",
                "signal_position",
                "entry_position",
                "exit_position",
                "signal_date",
                "entry_date",
                "exit_date",
                "side",
                "branch",
                "hold_bars",
            ]
        )
    return pd.concat(schedules, ignore_index=True)


def support_summary(schedule: pd.DataFrame, cfg: Config) -> dict[str, Any]:
    total = int(len(schedule))
    by_quarter = {
        quarter: int(schedule["quarter"].eq(quarter).sum())
        for quarter in ("q1", "q2", "q3", "q4")
    }
    longs = int(schedule["side"].gt(0).sum())
    shorts = int(schedule["side"].lt(0).sum())
    h1 = by_quarter["q1"] + by_quarter["q2"]
    h2 = by_quarter["q3"] + by_quarter["q4"]
    long_share = longs / total if total else 0.0
    short_share = shorts / total if total else 0.0
    maximum_quarter_share = max(by_quarter.values()) / total if total else 1.0
    passes = (
        total >= cfg.minimum_events
        and h1 >= cfg.minimum_half_events
        and h2 >= cfg.minimum_half_events
        and all(value >= cfg.minimum_quarter_events for value in by_quarter.values())
        and min(long_share, short_share) >= cfg.minimum_side_share
        and maximum_quarter_share <= cfg.maximum_quarter_share
    )
    return {
        "nonoverlap_total": total,
        "by_quarter": by_quarter,
        "h1": h1,
        "h2": h2,
        "longs": longs,
        "shorts": shorts,
        "long_share": long_share,
        "short_share": short_share,
        "maximum_quarter_share": maximum_quarter_share,
        "passes": bool(passes),
    }


def select_support_cell(cells: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    """Select by incidence strength only; never consume an outcome field."""
    cells = list(cells)
    forbidden = {"return", "pnl", "cagr", "mdd", "drawdown", "hit_rate"}
    for cell in cells:
        if forbidden.intersection(map(str.lower, cell)):
            raise ValueError("support selection received a forbidden outcome field")
    passing = [cell for cell in cells if cell["support"]["passes"]]
    if not passing:
        return None
    return max(
        passing,
        key=lambda cell: (cell["defense_threshold"], cell["flow_quantile"]),
    )


def fuzzy_overlap(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    tolerance_bars: int,
) -> dict[str, float | int]:
    """One-to-one timestamp overlap, independent of any trade outcome."""
    left_ns = np.sort(pd.to_datetime(left["signal_date"]).astype("int64").to_numpy())
    right_ns = np.sort(pd.to_datetime(right["signal_date"]).astype("int64").to_numpy())
    tolerance = int(FIVE_MINUTES.value * tolerance_bars)
    i = j = matches = 0
    while i < len(left_ns) and j < len(right_ns):
        delta = int(left_ns[i] - right_ns[j])
        if abs(delta) <= tolerance:
            matches += 1
            i += 1
            j += 1
        elif delta < 0:
            i += 1
        else:
            j += 1
    union = len(left_ns) + len(right_ns) - matches
    return {
        "matches": matches,
        "new_events": int(len(left_ns)),
        "prior_events": int(len(right_ns)),
        "jaccard": matches / union if union else 0.0,
        "new_clock_containment": matches / len(left_ns) if len(left_ns) else 0.0,
    }


def load_prior_clocks(credibility: pd.DataFrame) -> dict[str, pd.DataFrame]:
    pdf_signal = build_pdf_signal(credibility, CredibilityConfig())
    pdf = pdf_quarterly_schedule(pdf_signal, credibility)
    clocks: dict[str, pd.DataFrame] = {"pdf10": pdf[["signal_date", "side"]].copy()}
    for name, path in PRIOR_CLOCKS.items():
        if path.suffix == ".json":
            payload = json.loads(path.read_text())
            frame = pd.DataFrame(payload.get("events", []))
        else:
            frame = pd.read_csv(path, usecols=["signal_date", "side"])
        dates = pd.to_datetime(frame["signal_date"])
        frame = frame.loc[dates.ge(SELECTION_START) & dates.lt(SELECTION_END)].copy()
        clocks[name] = frame[["signal_date", "side"]]
    return clocks


def _event_clock_payload(schedule: pd.DataFrame) -> dict[str, Any]:
    columns = [
        "quarter",
        "signal_position",
        "entry_position",
        "exit_position",
        "signal_date",
        "entry_date",
        "exit_date",
        "side",
        "branch",
        "hold_bars",
    ]
    events = schedule[columns].to_dict("records")
    for event in events:
        for key in ("signal_position", "entry_position", "exit_position", "side", "hold_bars"):
            event[key] = int(event[key])
    return {
        "protocol": "CBFR-72 canonical outcome-blind event-clock freeze",
        "post_entry_outcomes_opened": False,
        "entry_or_later_ohlc_loaded": False,
        "selection_end_exclusive": str(SELECTION_END),
        "event_count": int(len(events)),
        "event_clock_sha256": canonical_hash(events),
        "events": events,
    }


def run_support(cfg: Config) -> dict[str, Any]:
    frame, source = load_support_inputs(cfg)
    panel = build_feature_panel(frame, cfg)
    cells: list[dict[str, Any]] = []
    schedules: dict[tuple[float, float], pd.DataFrame] = {}
    for flow_quantile in cfg.flow_quantiles:
        for defense_threshold in cfg.defense_thresholds:
            signal = build_signal(
                panel,
                frame,
                cfg,
                flow_quantile=flow_quantile,
                defense_threshold=defense_threshold,
            )
            schedule = quarterly_schedule(signal, frame)
            schedules[(flow_quantile, defense_threshold)] = schedule
            cells.append(
                {
                    "flow_quantile": flow_quantile,
                    "defense_threshold": defense_threshold,
                    "support": support_summary(schedule, cfg),
                }
            )
    selected = select_support_cell(cells)
    if selected is None:
        selected_schedule = quarterly_schedule(
            build_signal(
                panel,
                frame,
                cfg,
                flow_quantile=cfg.flow_quantiles[0],
                defense_threshold=cfg.defense_thresholds[0],
            ),
            frame,
        ).iloc[0:0]
        independence: dict[str, Any] = {}
        support_passes = False
    else:
        key = (selected["flow_quantile"], selected["defense_threshold"])
        selected_schedule = schedules[key]
        prior_clocks = load_prior_clocks(frame)
        independence = {
            name: fuzzy_overlap(
                selected_schedule,
                clock,
                tolerance_bars=cfg.overlap_tolerance_bars,
            )
            for name, clock in prior_clocks.items()
        }
        support_passes = all(
            metrics["jaccard"] <= cfg.maximum_prior_jaccard
            and metrics["new_clock_containment"]
            <= cfg.maximum_new_clock_containment
            for metrics in independence.values()
        )

    clock_payload = _event_clock_payload(selected_schedule)
    result = {
        "protocol": protocol(),
        "config": asdict(cfg),
        "frozen_sources": source,
        "support_selection": {
            "post_entry_outcomes_used": False,
            "cells": cells,
            "selected_cell": selected,
        },
        "outcome_blind_independence": independence,
        "event_clock_sha256": clock_payload["event_clock_sha256"],
        "all_support_gates_pass": bool(selected is not None and support_passes),
        "next_action": (
            "freeze evaluator before opening calendar-2023 post-entry outcomes"
            if selected is not None and support_passes
            else "retire CBFR-72 before any post-entry outcome is opened"
        ),
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    Path(cfg.event_clock_output).write_text(
        json.dumps(clock_payload, indent=2, sort_keys=True) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=Config.output)
    parser.add_argument("--event-clock-output", default=Config.event_clock_output)
    args = parser.parse_args()
    result = run_support(
        Config(output=args.output, event_clock_output=args.event_clock_output)
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
