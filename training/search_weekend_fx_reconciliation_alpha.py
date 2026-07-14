"""Search a closed-market BTC/FX reconciliation alpha.

BTC trades while spot FX is closed.  At the first fully completed six-pair FX
hour after an observed weekend closure, the experiment compares two pieces of
information that arrived on different clocks:

* BTC's displacement while FX was unavailable; and
* the cross-sectional safe-haven differential in the first completed FX hour.

Each component is normalized only by earlier closure events.  The fixed policy
trades the residual ``fx_event_z - btc_event_z`` at the next 5-minute open.  A
positive residual means BTC has not reflected the relatively risk-on FX reopen
and is bought; a negative residual is sold.

The FX differential is deliberately described as a cross-sectional proxy, not
as a pure global risk factor.  Pair orientation, closure detection, source-bar
completion, normalization, entry, hold and all controls are fixed before any
return outcome is opened.  Returned source frames are strictly pre-2024.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.search_hidden_safe_haven_cancellation_alpha import (
    CYCLICALS,
    FX_PATH,
    MARKET_PATH,
    SAFE_HAVENS,
    TICKERS,
    USD_ORIENTATION,
    load_market_before,
    read_completed_fx_hours_before,
)
from training.search_positioning_disagreement_alpha import (
    _future_extreme,
    _simulate_no_stop,
)


RESULT_PATH = Path("results/weekend_fx_reconciliation_alpha_scan_2026-07-14.json")
DESIGN_PATH = Path("docs/weekend-fx-reconciliation-alpha-design-2026-07-14.md")
CUTOFF = "2024-01-01"

MIN_CLOSURE_HOURS = 45.0
MAX_CLOSURE_HOURS = 72.0
EVENT_LOOKBACK = 52
EVENT_MIN_OBSERVATIONS = 26
HOLD_BARS = 24 * 12
SIDE_COST = 0.0006
LEVERAGE = 0.5
BTC_INSTRUMENT = "BTCUSDT"

WINDOWS = {
    "fit": ("2020-06-01", "2023-01-01"),
    "fit_2020_h2": ("2020-06-01", "2021-01-01"),
    "fit_2021_h1": ("2021-01-01", "2021-07-01"),
    "fit_2021_h2": ("2021-07-01", "2022-01-01"),
    "fit_2022_h1": ("2022-01-01", "2022-07-01"),
    "fit_2022_h2": ("2022-07-01", "2023-01-01"),
    "select_2023": ("2023-01-01", "2024-01-01"),
    "select_2023_h1": ("2023-01-01", "2023-07-01"),
    "select_2023_h2": ("2023-07-01", "2024-01-01"),
}
SEGMENTS = (
    "fit_2020_h2",
    "fit_2021_h1",
    "fit_2021_h2",
    "fit_2022_h1",
    "fit_2022_h2",
    "select_2023_h1",
    "select_2023_h2",
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prior_event_zscore(
    values: pd.Series | np.ndarray,
    *,
    lookback: int = EVENT_LOOKBACK,
    min_observations: int = EVENT_MIN_OBSERVATIONS,
) -> pd.Series:
    """Normalize an event with only strictly earlier closure events."""
    series = pd.Series(values, dtype=float).reset_index(drop=True)
    prior = series.shift(1)
    mean = prior.rolling(lookback, min_periods=min_observations).mean()
    std = prior.rolling(lookback, min_periods=min_observations).std(ddof=0).replace(0.0, np.nan)
    return (series - mean) / std


def validate_market_source(market: pd.DataFrame) -> None:
    """Reject malformed BTC bars without applying outcome-aware cleaning."""
    required = {"date", "open", "high", "low", "close"}
    missing = required.difference(market.columns)
    if missing:
        raise ValueError(f"market source is missing columns: {sorted(missing)}")
    if "tic" in market.columns:
        instruments = set(market["tic"].dropna().astype(str).unique())
        if instruments != {BTC_INSTRUMENT}:
            raise ValueError(f"expected only {BTC_INSTRUMENT}, found {sorted(instruments)}")
    values = market[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy(float)).all() or (values <= 0.0).any().any():
        raise ValueError("market OHLC must be finite and strictly positive")
    inconsistent = (
        (values["high"] < values[["open", "close"]].max(axis=1))
        | (values["low"] > values[["open", "close"]].min(axis=1))
        | (values["high"] < values["low"])
    )
    if inconsistent.any():
        raise ValueError("market source contains inconsistent OHLC bars")


def oriented_fx_gap_returns(current: pd.Series, previous: pd.Series) -> pd.Series:
    """Return exact pair-oriented log gaps; positive means USD strength."""
    output: dict[str, float] = {}
    for ticker in TICKERS:
        now = float(current[f"close_{ticker}"])
        before = float(previous[f"close_{ticker}"])
        if not np.isfinite(now) or not np.isfinite(before) or now <= 0.0 or before <= 0.0:
            output[ticker] = float("nan")
        else:
            output[ticker] = float(np.log(now / before) * USD_ORIENTATION[ticker])
    return pd.Series(output, dtype=float)


def safe_haven_gap_differential(oriented_returns: pd.Series) -> float:
    """Cross-sectional JPY/CHF differential, explicitly not a pure risk factor."""
    if not set(TICKERS).issubset(oriented_returns.index):
        raise ValueError("oriented return vector is missing required FX pairs")
    values = pd.to_numeric(oriented_returns.reindex(TICKERS), errors="coerce")
    if not np.isfinite(values.to_numpy(float)).all():
        return float("nan")
    safe = float(values.reindex(SAFE_HAVENS).mean())
    cyclical = float(values.reindex(CYCLICALS).mean())
    return safe - cyclical


def _btc_boundary_table(
    market: pd.DataFrame,
    dates: pd.Series,
) -> pd.DataFrame:
    positions = np.flatnonzero(dates.dt.minute.eq(0).to_numpy(bool))
    positions = positions[positions > 0]
    effective_time = pd.DatetimeIndex(dates.iloc[positions])
    source_time = pd.DatetimeIndex(dates.iloc[positions - 1])
    if not (source_time == effective_time - pd.Timedelta("5min")).all():
        raise RuntimeError("BTC boundary must use the completed minute-55 bar")
    close = pd.to_numeric(market["close"], errors="coerce").to_numpy(float)[positions - 1]
    return pd.DataFrame(
        {
            "effective_time": effective_time,
            "position": positions,
            "btc_source_time": source_time,
            "btc_boundary_close": close,
        }
    ).set_index("effective_time", drop=False)


def build_event_table(
    market: pd.DataFrame,
    dates: pd.Series,
    fx_hourly: pd.DataFrame,
) -> pd.DataFrame:
    """Build the frozen weekend-closure event table without future outcomes."""
    btc = _btc_boundary_table(market, dates)
    valid = fx_hourly.loc[fx_hourly["valid_hour"].astype(bool)].copy()
    valid = valid.sort_values("effective_time").reset_index().rename(columns={"index": "fx_row"})
    valid["previous_fx_row"] = valid["fx_row"].shift(1)
    valid["previous_effective_time"] = valid["effective_time"].shift(1)
    valid["closure_hours"] = (
        pd.to_datetime(valid["effective_time"])
        - pd.to_datetime(valid["previous_effective_time"])
    ).dt.total_seconds() / 3600.0
    event = valid.loc[
        valid["closure_hours"].between(MIN_CLOSURE_HOURS, MAX_CLOSURE_HOURS, inclusive="both")
        & pd.to_datetime(valid["effective_time"]).dt.dayofweek.isin([6, 0])
    ].copy()
    if event.empty:
        raise ValueError("no valid weekend FX reopen events")

    rows: list[dict[str, Any]] = []
    for _, current in event.iterrows():
        previous_row = int(current["previous_fx_row"])
        previous = fx_hourly.loc[previous_row]
        now = pd.Timestamp(current["effective_time"])
        before = pd.Timestamp(current["previous_effective_time"])
        if now not in btc.index or before not in btc.index:
            continue
        now_btc = btc.loc[now]
        before_btc = btc.loc[before]
        elapsed = float(current["closure_hours"])
        current_source = pd.Timestamp(current["source_time"])
        previous_source = pd.Timestamp(previous["source_time"])
        if current_source > now - pd.Timedelta("1min"):
            raise RuntimeError("current FX source is not complete at the decision boundary")
        if previous_source > before - pd.Timedelta("1min"):
            raise RuntimeError("previous FX source is not complete at the closure boundary")
        oriented = oriented_fx_gap_returns(current, previous)
        fx_gap = safe_haven_gap_differential(oriented)
        btc_now = float(now_btc["btc_boundary_close"])
        btc_before = float(before_btc["btc_boundary_close"])
        btc_gap = (
            float(np.log(btc_now / btc_before))
            if np.isfinite(btc_now)
            and np.isfinite(btc_before)
            and btc_now > 0.0
            and btc_before > 0.0
            else float("nan")
        )
        root_elapsed = np.sqrt(elapsed)
        rows.append(
            {
                "effective_time": now,
                "position": int(now_btc["position"]),
                "btc_source_time": pd.Timestamp(now_btc["btc_source_time"]),
                "fx_source_time": current_source,
                "previous_effective_time": before,
                "previous_btc_source_time": pd.Timestamp(before_btc["btc_source_time"]),
                "previous_fx_source_time": previous_source,
                "closure_hours": elapsed,
                "btc_gap_return": btc_gap,
                "fx_safe_haven_gap_differential": fx_gap,
                "btc_gap_scaled": btc_gap / root_elapsed,
                "fx_gap_scaled": fx_gap / root_elapsed,
            }
        )
    output = pd.DataFrame(rows).sort_values("effective_time").reset_index(drop=True)
    output["btc_event_z"] = prior_event_zscore(output["btc_gap_scaled"])
    output["fx_event_z"] = prior_event_zscore(output["fx_gap_scaled"])
    output["reconciliation_residual"] = output["fx_event_z"] - output["btc_event_z"]
    output["eligible"] = np.isfinite(output["reconciliation_residual"])
    if pd.to_datetime(output["effective_time"]).max() >= pd.Timestamp(CUTOFF):
        raise RuntimeError("post-cutoff closure event entered analysis")
    return output


def build_state(market: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    state = pd.DataFrame(
        {
            "weekend_reopen_event": np.zeros(len(market), dtype=bool),
            "eligible": np.zeros(len(market), dtype=bool),
            "btc_event_z": np.full(len(market), np.nan),
            "fx_event_z": np.full(len(market), np.nan),
            "reconciliation_residual": np.full(len(market), np.nan),
            "closure_hours": np.full(len(market), np.nan),
        }
    )
    positions = events["position"].to_numpy(np.int64)
    if len(np.unique(positions)) != len(positions):
        raise ValueError("weekend events must map one-to-one to BTC decision rows")
    state.loc[positions, "weekend_reopen_event"] = True
    for column in (
        "eligible",
        "btc_event_z",
        "fx_event_z",
        "reconciliation_residual",
        "closure_hours",
    ):
        state.loc[positions, column] = events[column].to_numpy()
    return state


def signed_masks(values: np.ndarray, eligible: np.ndarray, *, flip: bool = False) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    active = np.asarray(eligible, dtype=bool) & np.isfinite(values) & (values != 0.0)
    side = np.sign(values) * (-1.0 if flip else 1.0)
    return active & (side > 0.0), active & (side < 0.0)


def policy_masks(state: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return signed_masks(
        state["reconciliation_residual"].to_numpy(float),
        state["eligible"].to_numpy(bool),
    )


def lag_sparse_event_masks(
    long_active: np.ndarray,
    short_active: np.ndarray,
    event_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    event_positions = np.flatnonzero(np.asarray(event_mask, dtype=bool))
    output_long = np.zeros(len(event_mask), dtype=bool)
    output_short = np.zeros(len(event_mask), dtype=bool)
    if len(event_positions) > 1:
        output_long[event_positions[1:]] = np.asarray(long_active, dtype=bool)[event_positions[:-1]]
        output_short[event_positions[1:]] = np.asarray(short_active, dtype=bool)[event_positions[:-1]]
    return output_long, output_short


def shift_masks(long_active: np.ndarray, short_active: np.ndarray, bars: int) -> tuple[np.ndarray, np.ndarray]:
    if bars < 0:
        raise ValueError("bar shift cannot be negative")
    if bars == 0:
        return np.asarray(long_active, dtype=bool).copy(), np.asarray(short_active, dtype=bool).copy()
    pad = np.zeros(bars, dtype=bool)
    return (
        np.r_[pad, np.asarray(long_active, dtype=bool)[:-bars]],
        np.r_[pad, np.asarray(short_active, dtype=bool)[:-bars]],
    )


def support_counts(
    dates: pd.Series,
    long_active: np.ndarray,
    short_active: np.ndarray,
    *,
    window: str,
    hold_bars: int = HOLD_BARS,
) -> dict[str, int]:
    start, end = WINDOWS[window]
    period = ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy(bool)
    long_active = np.asarray(long_active, dtype=bool)
    short_active = np.asarray(short_active, dtype=bool)
    raw_positions = np.flatnonzero(period & (long_active | short_active))
    executable: list[int] = []
    next_position = 0
    for position in raw_positions:
        entry = int(position) + 1
        exit_position = entry + hold_bars
        if position < next_position or exit_position >= len(dates) or not period[exit_position]:
            continue
        if long_active[position] == short_active[position]:
            continue
        executable.append(int(position))
        next_position = exit_position + 1
    selected = np.asarray(executable, dtype=np.int64)
    return {
        "raw": int(len(raw_positions)),
        "raw_long": int((period & long_active & ~short_active).sum()),
        "raw_short": int((period & short_active & ~long_active).sum()),
        "strict_executable": int(len(selected)),
        "strict_executable_long": int(long_active[selected].sum()) if len(selected) else 0,
        "strict_executable_short": int(short_active[selected].sum()) if len(selected) else 0,
    }


def support_passes(support: dict[str, dict[str, int]]) -> bool:
    fit = support["fit"]
    select = support["select_2023"]
    h1 = support["select_2023_h1"]
    h2 = support["select_2023_h2"]
    return bool(
        fit["strict_executable"] >= 80
        and select["strict_executable"] >= 24
        and min(h1["strict_executable"], h2["strict_executable"]) >= 8
        and min(fit["strict_executable_long"], fit["strict_executable_short"]) >= 15
        and min(select["strict_executable_long"], select["strict_executable_short"]) >= 4
    )


def simulate(
    market: pd.DataFrame,
    dates: pd.Series,
    long_active: np.ndarray,
    short_active: np.ndarray,
    *,
    hold_bars: int = HOLD_BARS,
    side_cost: float = SIDE_COST,
    extremes: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, dict[str, Any]]:
    if extremes is None:
        extremes = (
            _future_extreme(market["low"].to_numpy(float), hold_bars, "min"),
            _future_extreme(market["high"].to_numpy(float), hold_bars, "max"),
        )
    return {
        window: _simulate_no_stop(
            market,
            dates,
            long_active,
            short_active,
            window=window,
            hold_bars=hold_bars,
            stride_bars=1,
            leverage=LEVERAGE,
            fee_rate=side_cost,
            slippage_rate=0.0,
            extremes=extremes,
            windows=WINDOWS,
        )
        for window in WINDOWS
    }


def admission(stats: dict[str, dict[str, Any]]) -> bool:
    return bool(
        stats["fit"]["trades"] >= 80
        and stats["select_2023"]["trades"] >= 24
        and min(stats["select_2023_h1"]["trades"], stats["select_2023_h2"]["trades"]) >= 8
        and min(stats["fit"]["longs"], stats["fit"]["shorts"]) >= 15
        and min(stats["select_2023"]["longs"], stats["select_2023"]["shorts"]) >= 4
        and stats["fit"]["return_pct"] > 0.0
        and stats["fit"]["ratio"] >= 3.0
        and stats["select_2023"]["return_pct"] > 0.0
        and stats["select_2023"]["ratio"] >= 3.0
        and stats["select_2023_h1"]["return_pct"] > 0.0
        and stats["select_2023_h2"]["return_pct"] > 0.0
    )


def finite_spearman(left: np.ndarray, right: np.ndarray, minimum: int = 20) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < minimum:
        return float("nan")
    return float(pd.Series(left[valid]).corr(pd.Series(right[valid]), method="spearman"))


def side_agreement(
    left: tuple[np.ndarray, np.ndarray],
    right: tuple[np.ndarray, np.ndarray],
) -> float:
    left_side = left[0].astype(np.int8) - left[1].astype(np.int8)
    right_side = right[0].astype(np.int8) - right[1].astype(np.int8)
    both = (left_side != 0) & (right_side != 0)
    return float((left_side[both] == right_side[both]).mean()) if both.any() else float("nan")


def _component_masks(state: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    eligible = state["eligible"].to_numpy(bool)
    event = state["weekend_reopen_event"].to_numpy(bool) & eligible
    btc = signed_masks(state["btc_event_z"].to_numpy(float), eligible)
    fx = signed_masks(state["fx_event_z"].to_numpy(float), eligible)
    return {
        "schedule_always_long": (event.copy(), np.zeros(len(state), dtype=bool)),
        "schedule_always_short": (np.zeros(len(state), dtype=bool), event.copy()),
        "btc_weekend_continuation": btc,
        "btc_weekend_reversal": (btc[1].copy(), btc[0].copy()),
        "fx_reopen_only": fx,
        "fx_reopen_opposite": (fx[1].copy(), fx[0].copy()),
    }


def _print_stats(title: str, stats: dict[str, dict[str, Any]]) -> None:
    print("\n" + title)
    for window in ("fit", "select_2023", *SEGMENTS):
        value = stats[window]
        print(
            window,
            f"ret={value['return_pct']:.2f}",
            f"cagr={value['cagr_pct']:.2f}",
            f"mdd={value['strict_mdd_pct']:.2f}",
            f"ratio={value['ratio']:.2f}",
            f"n={value['trades']}",
            f"L/S={value['longs']}/{value['shorts']}",
        )


def run(
    *,
    market_path: str | Path = MARKET_PATH,
    fx_path: str | Path = FX_PATH,
    support_only: bool = False,
) -> dict[str, Any]:
    market, dates = load_market_before(market_path, CUTOFF)
    validate_market_source(market)
    fx_hourly = read_completed_fx_hours_before(fx_path, CUTOFF)
    events = build_event_table(market, dates, fx_hourly)
    state = build_state(market, events)
    primary_masks = policy_masks(state)
    support = {
        window: support_counts(dates, *primary_masks, window=window)
        for window in ("fit", "select_2023", "select_2023_h1", "select_2023_h2")
    }
    preflight = {
        "support_only": True,
        "support": support,
        "support_passed": support_passes(support),
        "closure_events": int(len(events)),
        "eligible_events": int(events["eligible"].sum()),
        "event_year_counts": {
            str(year): int(count)
            for year, count in events.groupby(pd.to_datetime(events["effective_time"]).dt.year).size().items()
        },
        "source_latest": str(pd.to_datetime(events["fx_source_time"]).max()),
        "outcomes_opened": False,
    }
    if support_only:
        return preflight
    if not preflight["support_passed"]:
        raise RuntimeError("support preflight failed; outcome access remains closed")

    extremes = (
        _future_extreme(market["low"].to_numpy(float), HOLD_BARS, "min"),
        _future_extreme(market["high"].to_numpy(float), HOLD_BARS, "max"),
    )
    primary_stats = simulate(market, dates, *primary_masks, extremes=extremes)
    controls_masks = _component_masks(state)
    controls_masks["exact_direction_flip"] = (primary_masks[1].copy(), primary_masks[0].copy())
    controls_masks["prior_closure_side"] = lag_sparse_event_masks(
        *primary_masks, state["weekend_reopen_event"].to_numpy(bool)
    )
    controls = {
        name: simulate(market, dates, *masks, extremes=extremes)
        for name, masks in controls_masks.items()
    }
    cost_stress = {
        str(bp): simulate(
            market,
            dates,
            *primary_masks,
            side_cost=bp / 10_000.0,
            extremes=extremes,
        )
        for bp in (0, 1, 3, 6, 10, 15)
    }
    entry_delay = {
        str(minutes): simulate(
            market,
            dates,
            *shift_masks(*primary_masks, bars=bars),
            extremes=extremes,
        )
        for minutes, bars in ((5, 0), (10, 1), (15, 2))
    }
    hold_diagnostics: dict[str, dict[str, dict[str, Any]]] = {}
    for hours in (12, 24, 48):
        bars = hours * 12
        hold_diagnostics[str(hours)] = simulate(
            market,
            dates,
            *primary_masks,
            hold_bars=bars,
        )

    event_rows = events.loc[events["eligible"]].copy()
    feature_spearman = {
        "residual_vs_btc": finite_spearman(
            event_rows["reconciliation_residual"].to_numpy(float),
            event_rows["btc_event_z"].to_numpy(float),
        ),
        "residual_vs_fx": finite_spearman(
            event_rows["reconciliation_residual"].to_numpy(float),
            event_rows["fx_event_z"].to_numpy(float),
        ),
        "btc_weekend_vs_later_fx_reopen": finite_spearman(
            event_rows["btc_event_z"].to_numpy(float),
            event_rows["fx_event_z"].to_numpy(float),
        ),
    }
    component_agreement = {
        name: side_agreement(primary_masks, masks)
        for name, masks in controls_masks.items()
        if name in {
            "btc_weekend_continuation",
            "btc_weekend_reversal",
            "fx_reopen_only",
            "fx_reopen_opposite",
        }
    }
    control_admissions = {name: admission(stats) for name, stats in controls.items()}
    component_controls = {
        name: control_admissions[name]
        for name in (
            "btc_weekend_continuation",
            "btc_weekend_reversal",
            "fx_reopen_only",
            "fx_reopen_opposite",
        )
    }
    residual_distinct = bool(
        max(
            abs(feature_spearman["residual_vs_btc"]),
            abs(feature_spearman["residual_vs_fx"]),
        )
        < 0.85
    )
    output = {
        "protocol": {
            "source_cutoff": "returned BTC and FX frames strictly before 2024-01-01",
            "source_io_disclosure": "cutoff-crossing chunks may be physically read and immediately discarded; discarded rows never enter returned frames or computation",
            "btc_source_semantics": "fixed CSV path/hash; tic must equal BTCUSDT; project provenance treats OHLC as Binance USD-M perpetual bar fields, but the file itself does not encode venue/feed provenance",
            "fx_source_semantics": "fixed Wave Trading CSV path/hash; stored one-minute close field is used as-is and the cache does not encode bid/ask/mid provenance",
            "timezone": "all source timestamps are parsed with utc=True and represented internally as timezone-naive UTC",
            "source_cleaning": "BTC duplicate timestamps keep the last row in the shared loader, a complete 5m grid is required, and finite positive OHLC consistency is enforced; FX requires sorted timestamps, positive finite event closes, all six pairs, >=55 rows and minute59; no statistical outlier clipping",
            "historical_availability_limit": "FX source_time is the candle timestamp, not publication/ingestion time; the replay waits five minutes after the hour but cannot certify historical feed latency",
            "event": "first valid six-pair FX hour after an observed 45-72h gap, only Sunday/Monday UTC",
            "valid_fx_hour": "all six fixed pairs, >=55 one-minute rows each, minute-59 present, source timestamp strictly before next-hour effective time",
            "pairs": list(TICKERS),
            "orientation": USD_ORIENTATION,
            "safe_haven_group": list(SAFE_HAVENS),
            "comparison_group": list(CYCLICALS),
            "factor_disclaimer": "cross-sectional safe-haven differential; not claimed to be a pure global risk-on factor",
            "normalization": f"divide each closure gap by sqrt(elapsed hours), then z-score with shift(1) over prior {EVENT_LOOKBACK} closure events, minimum {EVENT_MIN_OBSERVATIONS}",
            "primary": "sign(fx_event_z - btc_event_z); no magnitude threshold or parameter grid",
            "entry": "signal row is current effective_time minute00; enter at open of the BTC 5m bar beginning effective_time+5min",
            "exit": "exit at open of the BTC 5m bar exactly 288 bars after entry (effective_time+24h+5min)",
            "hold_bars": HOLD_BARS,
            "hold_hours": HOLD_BARS / 12,
            "leverage": LEVERAGE,
            "cost": "6bp/side canonical implementation cost; historical perp funding is not included and is an explicit live-promotion blocker",
            "strict_mdd": "favorable-first adverse-second OHLC high-water",
            "selection": "fit 2020-06..2022; one-shot 2023 plus fixed H1/H2 diagnostics; no 2024+",
            "sample_assignment": "event belongs by current effective_time; replay requires entry and exit inside the same declared window, otherwise the event is skipped",
            "online_state_2023": "earlier 2023 closure feature values update the prior-event z-score state for later 2023 events; no return outcome updates the state",
            "support_only_preflight": {"performed_before_returns": True, **preflight},
            "diagnostics_not_selected": "entry 5/10/15 minutes and hold 12/24/48 hours are reported without choosing among them",
            "oos_opened": False,
            "research_only_data_caveat": "unknown FX quote basis and absent publication/ingestion timestamps block live promotion even if the historical alpha passes",
        },
        "source": {
            "market_path": str(market_path),
            "market_sha256": _sha256(market_path),
            "fx_path": str(fx_path),
            "fx_sha256": _sha256(fx_path),
        },
        "state_summary": {
            "closure_events": int(len(events)),
            "eligible_events": int(events["eligible"].sum()),
            "primary_raw_long_short": [int(primary_masks[0].sum()), int(primary_masks[1].sum())],
            "closure_hours_min_max": [
                float(events["closure_hours"].min()),
                float(events["closure_hours"].max()),
            ],
        },
        "primary": {"stats": primary_stats, "prelim_admitted": admission(primary_stats)},
        "controls": controls,
        "control_admissions": control_admissions,
        "cost_stress": cost_stress,
        "entry_delay_diagnostics": entry_delay,
        "hold_diagnostics": hold_diagnostics,
        "mechanism_audit": {
            "feature_spearman": feature_spearman,
            "primary_side_agreement": component_agreement,
            "component_control_admissions": component_controls,
            "residual_distinct": residual_distinct,
            "gate": "abs residual/component Spearman below 0.85 and no component-only policy admitted",
        },
        "prelim_admitted": admission(primary_stats),
        "final_admitted": bool(
            admission(primary_stats)
            and residual_distinct
            and not any(component_controls.values())
        ),
        "live_promotion_blocked": True,
        "live_promotion_blocker": "historical perp funding is not included in this discovery replay",
        "oos_opened": False,
    }
    _print_stats("PRIMARY weekend FX reconciliation", primary_stats)
    for name, stats in controls.items():
        _print_stats("CONTROL " + name, stats)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market-path", default=str(MARKET_PATH))
    parser.add_argument("--fx-path", default=str(FX_PATH))
    parser.add_argument("--support-only", action="store_true")
    args = parser.parse_args()
    output = run(
        market_path=args.market_path,
        fx_path=args.fx_path,
        support_only=args.support_only,
    )
    if args.support_only:
        print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
