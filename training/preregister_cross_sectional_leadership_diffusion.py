"""Outcome-blind preregistration for CLD-72 leadership diffusion.

CLD-72 treats six liquid USD-M alt perpetuals as a distributed risk sensor.
It trades BTC only when a previously concentrated six-hour alt move diffuses
across names, the prior leader loses share, aggressive flow agrees, and BTC
still lags the common alt move.  This module is restricted to signal-time
inputs and event incidence; it must not inspect any post-entry outcome.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from training.preregister_cross_collateral_book_validated_flow_rejection import (
    fuzzy_overlap,
)


POLICY_ID = "CLD-72"
PREREGISTRATION_SOURCE = Path(
    "training/preregister_cross_sectional_leadership_diffusion.py"
)
SELECTION_START = pd.Timestamp("2023-01-01 00:00:00")
SELECTION_END = pd.Timestamp("2024-01-01 00:00:00")
FIVE_MINUTES = pd.Timedelta(minutes=5)
ONE_HOUR = pd.Timedelta(hours=1)
SYMBOLS = (
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
)

ALT_DIRECTORY = Path("data/binance_um_pool_5m_2023_2026")
ALT_MANIFEST = ALT_DIRECTORY / "download_summary_5m_2023-01_2026-05.json"
BTC_MANIFEST = Path("data/binance_um_kline_reference_btc_2020_2023/build_manifest.json")
BTC_DATA = Path(
    "data/binance_um_kline_reference_btc_2020_2023/"
    "BTCUSDT_5m_2020-01-01_2023-12-31.csv.gz"
)
ALT_MANIFEST_SHA256 = "9010bf1110e2555581942d6a0400d98466b64a5bee355c05ce5b7d2c85af173a"
BTC_MANIFEST_SHA256 = "c04fbbd299cc748a6745c0ef030787da4d560833c744c81c98dd8840efc7913e"
BTC_DATA_SHA256 = "e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d"
ALT_DATA_SHA256 = {
    "ETHUSDT": "68492d340e6617cda65dcc6cb42ba9138d0866ed0ce7fc1328d66f4e01b55e82",
    "SOLUSDT": "4216e63393ad66a0e4aab6d8b07199f209188a513c680a489c6d055a604607d1",
    "BNBUSDT": "b173cb40adf6a81b23a1b1ede988e001b8f7abbf7c184c1a3f1f96b1fc5e1263",
    "XRPUSDT": "ccb993e92efa586db0d8c8c5d4d1734221b50a50db04da14f447f7905cf50593",
    "ADAUSDT": "9f63ea66c2872f7941b9fde742746ba808fd0efb2a3b35a81bf4eb8cfc6479e1",
    "DOGEUSDT": "a60e33dc12e1480d284e5fdbda18a06e5aeeb59ee3aa542521bcea211d01861a",
}

PRIOR_CLOCKS = {
    "cbfr72": Path(
        "results/cross_collateral_book_validated_flow_rejection_event_clock_2026-07-18.json"
    ),
    "crrc72": Path("results/cross_venue_radial_refill_compression_event_clock_2026-07-17.json"),
    "cspr": Path("results/cash_sponsored_perp_rejection_clock_2026-07-14.csv"),
    "umfr36": Path("results/um_forced_flow_reversion_clock_2026-07-14.csv"),
}


@dataclass(frozen=True)
class Config:
    output: str = "results/cross_sectional_leadership_diffusion_support_2026-07-18.json"
    event_clock_output: str = (
        "results/cross_sectional_leadership_diffusion_event_clock_2026-07-18.json"
    )
    docs_output: str = (
        "docs/cross-sectional-leadership-diffusion-preregistration-2026-07-18.md"
    )
    return_lookback_hours: int = 6
    state_lag_hours: int = 6
    flow_baseline_hours: int = 30 * 24
    flow_minimum_hours: int = 14 * 24
    threshold_baseline_hours: int = 90 * 24
    threshold_minimum_hours: int = 30 * 24
    move_quantiles: tuple[float, ...] = (0.60, 0.70)
    prior_hhi_quantiles: tuple[float, ...] = (0.60, 0.70, 0.80)
    maximum_hhi_ratios: tuple[float, ...] = (0.70, 0.80, 0.90)
    minimum_participations: tuple[float, ...] = (4 / 6, 5 / 6)
    minimum_flow_alignments: tuple[float, ...] = (0.50, 4 / 6)
    turnover_quantiles: tuple[float, ...] = (0.50, 0.60)
    leader_decline_quantiles: tuple[float, ...] = (0.50, 0.60)
    hold_bars: int = 72
    entry_delay_bars_after_hour_boundary: int = 1
    minimum_events: int = 100
    minimum_half_events: int = 35
    minimum_quarter_events: int = 15
    minimum_side_share: float = 0.35
    maximum_quarter_share: float = 0.40
    overlap_tolerance_bars: int = 12
    maximum_prior_jaccard: float = 0.20
    maximum_new_clock_containment: float = 0.30


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protocol() -> dict[str, Any]:
    cfg = Config()
    return {
        "policy_id": POLICY_ID,
        "hypothesis": (
            "BTC tends to catch up when a six-hour alt move changes from "
            "single-name concentration to broad participation, the old leader "
            "loses share, current taker flow agrees, and BTC still trails"
        ),
        "evidence_boundary": {
            "allowed_before_evaluator_freeze": [
                "completed alt and BTC bars at or before each feature boundary",
                "strictly prior rolling thresholds",
                "event timestamps, sides, and overlap with prior clocks",
            ],
            "forbidden_before_evaluator_freeze": [
                "entry or later OHLC",
                "post-entry return or excursion",
                "funding paid while held",
                "PnL, equity, CAGR, MDD, hit rate, or payoff",
            ],
            "post_entry_outcomes_opened": False,
        },
        "clock": {
            "feature_boundary": "right edge of a completed UTC hour",
            "entry": "one full five-minute bar after that boundary",
            "exit": "72 five-minute bars after entry (six hours)",
            "quarter_contained": True,
            "nonoverlapping_within_each_quarter": True,
        },
        "feature_formula": {
            "alt_return": "log(hourly_close_t / hourly_close_t_minus_6h)",
            "common_direction": "sign(cross-sectional median alt_return)",
            "participation": "fraction of six alt returns with common_direction",
            "flow": "2 * hourly_taker_buy_quote / hourly_quote_volume - 1",
            "flow_z": "current flow standardized by strictly prior 30-day history",
            "flow_alignment": "fraction of flow_z signs with common_direction",
            "residual": "alt_return - cross-sectional median alt_return",
            "residual_hhi": "sum((abs(residual) / sum(abs(residual))) ** 2)",
            "prior_state": "residual HHI and leader weights six hours earlier",
            "rank_turnover": "(1 - Spearman(rank_t, rank_t_minus_6h)) / 2",
            "leader_decline": "prior leader share at t-6h minus its current share",
            "btc_lag": (
                "direction * (median_alt_trailing_6h_return_t - "
                "btc_trailing_6h_return_t) > 0; both end at the completed "
                "feature boundary, never after it"
            ),
            "action": "trade BTC in common_direction",
            "tie_contract": (
                "average ranks; exact leader ties resolve by frozen SYMBOLS order"
            ),
        },
        "support_selection": {
            "outcomes_used": False,
            "grid_disclosed": True,
            "selection_rule": (
                "among support-passing cells, maximize participation, flow "
                "alignment, HHI contraction, leader decline, turnover, prior "
                "concentration, then move threshold; never use a return tie-break"
            ),
        },
        "eventual_execution": {
            "instrument": "Binance BTCUSDT USD-M perpetual",
            "leverage": 0.5,
            "cost_bp_per_notional_side": 6.0,
            "funding": "realized funding cash flows on the held path",
            "strict_mdd": "global pre-entry high-water mark plus held OHLC path",
            "cagr": "full declared wall-clock, including idle periods",
            "position_state": "one BTC position maximum; skip signals until fixed exit",
            "frozen_controls": [
                "BTC trailing-six-hour momentum only",
                "static alt breadth without leadership transition",
                "leadership transition without taker-flow alignment",
                "direction flip",
                "additional five-minute delay",
                "10 bp per-notional-side cost stress",
            ],
        },
        "live_parity": {
            "hour_requirement": "exactly twelve completed five-minute bars per symbol",
            "joins": "exact UTC timestamps only; no nearest, ffill, or bfill",
            "rolling_thresholds": "shifted one row and exclude the current observation",
        },
        "sequential_oos": {
            "first_opened_stage": "calendar 2023 only, after evaluator freeze",
            "later_stages": ["2024", "2025", "2026H1"],
            "failed_stage_action": "retire the exact frozen policy without repair",
        },
        "support_gate": {
            "events_at_least": cfg.minimum_events,
            "events_each_half_at_least": cfg.minimum_half_events,
            "events_each_quarter_at_least": cfg.minimum_quarter_events,
            "each_side_share_at_least": cfg.minimum_side_share,
            "largest_quarter_share_at_most": cfg.maximum_quarter_share,
        },
    }


def _validate_config(cfg: Config) -> None:
    if cfg != Config(
        output=cfg.output,
        event_clock_output=cfg.event_clock_output,
        docs_output=cfg.docs_output,
    ):
        raise ValueError("CLD-72 signal/support configuration is frozen")


def prior_rolling_quantile(
    series: pd.Series, *, quantile: float, window: int, minimum: int
) -> pd.Series:
    """A threshold whose newest baseline observation is t-1."""
    return series.rolling(window, min_periods=minimum).quantile(quantile).shift(1)


def prior_zscore(series: pd.Series, *, window: int, minimum: int) -> pd.Series:
    mean = series.rolling(window, min_periods=minimum).mean().shift(1)
    std = series.rolling(window, min_periods=minimum).std(ddof=1).shift(1)
    return ((series - mean) / std.replace(0.0, np.nan)).replace(
        [np.inf, -np.inf], np.nan
    )


def _alt_path(symbol: str) -> Path:
    paths = sorted(ALT_DIRECTORY.glob(f"{symbol}_5m_*.csv.gz"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one CLD source for {symbol}, found {len(paths)}")
    return paths[0]


def _validate_sources() -> dict[str, Any]:
    expected = {
        ALT_MANIFEST: ALT_MANIFEST_SHA256,
        BTC_MANIFEST: BTC_MANIFEST_SHA256,
        BTC_DATA: BTC_DATA_SHA256,
    }
    expected.update({_alt_path(symbol): digest for symbol, digest in ALT_DATA_SHA256.items()})
    for path, digest in expected.items():
        if sha256(path) != digest:
            raise RuntimeError(f"CLD frozen source hash mismatch: {path}")
    manifest = json.loads(ALT_MANIFEST.read_text())
    if tuple(manifest["config"]["symbols"].split(",")) != SYMBOLS:
        raise RuntimeError("CLD alt universe changed")
    return {
        "alt_manifest": str(ALT_MANIFEST),
        "alt_manifest_sha256": ALT_MANIFEST_SHA256,
        "alt_data_sha256": ALT_DATA_SHA256,
        "btc_manifest": str(BTC_MANIFEST),
        "btc_manifest_sha256": BTC_MANIFEST_SHA256,
        "btc_data": str(BTC_DATA),
        "btc_data_sha256": BTC_DATA_SHA256,
        "funding_rows_loaded": 0,
        "post_entry_outcome_columns_computed": [],
        "post_entry_return_funding_pnl_or_equity_loaded": False,
    }


def _load_completed_hour(path: Path, symbol: str) -> pd.DataFrame:
    columns = [
        "date",
        "open",
        "close",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_quote",
    ]
    raw = pd.read_csv(path, usecols=columns, parse_dates=["date"])
    raw = raw.loc[
        raw["date"].ge(SELECTION_START) & raw["date"].lt(SELECTION_END)
    ].copy()
    expected = pd.date_range(SELECTION_START, SELECTION_END - FIVE_MINUTES, freq="5min")
    if not pd.DatetimeIndex(raw["date"]).equals(expected):
        raise RuntimeError(f"{symbol} CLD source is not a complete UTC 5m grid")
    raw = raw.set_index("date")
    hourly = pd.DataFrame(
        {
            "open": raw["open"].resample("1h", closed="left", label="right").first(),
            "close": raw["close"].resample("1h", closed="left", label="right").last(),
            "quote_volume": raw["quote_asset_volume"]
            .resample("1h", closed="left", label="right")
            .sum(),
            "taker_buy_quote": raw["taker_buy_quote"]
            .resample("1h", closed="left", label="right")
            .sum(),
            "bar_count": raw["close"].resample("1h", closed="left", label="right").count(),
            "positive_activity_count": (
                raw["quote_asset_volume"].gt(0) & raw["number_of_trades"].gt(0)
            )
            .resample("1h", closed="left", label="right")
            .sum(),
        }
    )
    expected_hourly = pd.date_range(SELECTION_START + ONE_HOUR, SELECTION_END, freq="1h")
    if not hourly.index.equals(expected_hourly) or not hourly["bar_count"].eq(12).all():
        raise RuntimeError(f"{symbol} CLD completed-hour grid failure")
    hourly["quality"] = hourly["positive_activity_count"].eq(12)
    return hourly


def _load_btc_hourly_close() -> pd.Series:
    raw = pd.read_csv(BTC_DATA, usecols=["date", "close"], parse_dates=["date"])
    raw = raw.loc[
        raw["date"].ge(SELECTION_START) & raw["date"].lt(SELECTION_END)
    ].copy()
    expected = pd.date_range(SELECTION_START, SELECTION_END - FIVE_MINUTES, freq="5min")
    if not pd.DatetimeIndex(raw["date"]).equals(expected):
        raise RuntimeError("BTC CLD source is not a complete UTC 5m grid")
    return (
        raw.set_index("date")["close"]
        .resample("1h", closed="left", label="right")
        .last()
    )


def _rank_turnover(current: pd.DataFrame, lag: int) -> pd.Series:
    ranks = current.rank(axis=1, method="average").to_numpy(float)
    prior = np.roll(ranks, lag, axis=0)
    prior[:lag] = np.nan
    values: list[float] = []
    for left, right in zip(ranks, prior):
        if not (np.isfinite(left).all() and np.isfinite(right).all()):
            values.append(np.nan)
            continue
        values.append(float((1.0 - np.corrcoef(left, right)[0, 1]) / 2.0))
    return pd.Series(values, index=current.index, name="rank_turnover")


def build_feature_panel(cfg: Config) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build signal-time CLD features without computing any future return."""
    _validate_config(cfg)
    source = _validate_sources()
    hourly = {symbol: _load_completed_hour(_alt_path(symbol), symbol) for symbol in SYMBOLS}
    index = hourly[SYMBOLS[0]].index
    returns = pd.DataFrame(
        {
            symbol: np.log(
                hourly[symbol]["close"]
                / hourly[symbol]["close"].shift(cfg.return_lookback_hours)
            ).where(hourly[symbol]["quality"])
            for symbol in SYMBOLS
        },
        index=index,
    )
    flow_z = pd.DataFrame(
        {
            symbol: prior_zscore(
                (
                    2.0
                    * hourly[symbol]["taker_buy_quote"]
                    / hourly[symbol]["quote_volume"]
                    - 1.0
                ).where(hourly[symbol]["quality"]),
                window=cfg.flow_baseline_hours,
                minimum=cfg.flow_minimum_hours,
            )
            for symbol in SYMBOLS
        },
        index=index,
    )
    median_return = returns.median(axis=1)
    direction = np.sign(median_return)
    participation = np.sign(returns).eq(direction, axis=0).mean(axis=1)
    flow_alignment = np.sign(flow_z).eq(direction, axis=0).mean(axis=1)
    residual = returns.sub(median_return, axis=0)
    absolute_residual = residual.abs()
    weights = absolute_residual.div(
        absolute_residual.sum(axis=1).replace(0.0, np.nan), axis=0
    )
    hhi = (weights**2).sum(axis=1, min_count=len(SYMBOLS))
    prior_weights = weights.shift(cfg.state_lag_hours)
    prior_hhi = hhi.shift(cfg.state_lag_hours)
    prior_leader = np.argmax(prior_weights.fillna(-1.0).to_numpy(), axis=1)
    current_weights = weights.to_numpy(float)
    current_prior_leader_share = np.array(
        [
            current_weights[row, column]
            if np.isfinite(current_weights[row, column])
            else np.nan
            for row, column in enumerate(prior_leader)
        ]
    )
    prior_top_share = prior_weights.max(axis=1)
    leader_decline = pd.Series(
        prior_top_share.to_numpy() - current_prior_leader_share,
        index=index,
        name="leader_decline",
    )
    current_leader = np.argmax(weights.fillna(-1.0).to_numpy(), axis=1)
    leader_changed = pd.Series(current_leader != prior_leader, index=index)
    rank_turnover = _rank_turnover(returns, cfg.state_lag_hours)
    btc_close = _load_btc_hourly_close()
    btc_return = np.log(
        btc_close / btc_close.shift(cfg.return_lookback_hours)
    )
    btc_lag = direction * (median_return - btc_return)

    panel = pd.DataFrame(
        {
            "feature_boundary": index,
            "signal_date": index - FIVE_MINUTES,
            "entry_date": index
            + cfg.entry_delay_bars_after_hour_boundary * FIVE_MINUTES,
            "median_return_abs": median_return.abs(),
            "direction": direction.astype(float),
            "participation": participation,
            "flow_alignment": flow_alignment,
            "residual_hhi": hhi,
            "prior_residual_hhi": prior_hhi,
            "hhi_ratio": hhi / prior_hhi.replace(0.0, np.nan),
            "rank_turnover": rank_turnover,
            "leader_decline": leader_decline,
            "leader_changed": leader_changed,
            "btc_lag": btc_lag,
        }
    )
    threshold_inputs = {
        "move": panel["median_return_abs"],
        "prior_hhi": panel["prior_residual_hhi"],
        "turnover": panel["rank_turnover"],
        "leader_decline": panel["leader_decline"],
    }
    for name, values in threshold_inputs.items():
        quantiles: Iterable[float]
        if name == "move":
            quantiles = cfg.move_quantiles
        elif name == "prior_hhi":
            quantiles = cfg.prior_hhi_quantiles
        elif name == "turnover":
            quantiles = cfg.turnover_quantiles
        else:
            quantiles = cfg.leader_decline_quantiles
        for quantile in quantiles:
            panel[f"{name}_q{int(round(100 * quantile))}"] = prior_rolling_quantile(
                values,
                quantile=quantile,
                window=cfg.threshold_baseline_hours,
                minimum=cfg.threshold_minimum_hours,
            )
    feature_columns = [
        "median_return_abs",
        "direction",
        "participation",
        "flow_alignment",
        "residual_hhi",
        "prior_residual_hhi",
        "hhi_ratio",
        "rank_turnover",
        "leader_decline",
        "btc_lag",
    ]
    panel["clean"] = np.isfinite(panel[feature_columns].to_numpy(float)).all(axis=1)
    source["completed_alt_hour_rows"] = int(len(panel))
    source["clean_feature_rows"] = int(panel["clean"].sum())
    return panel, source


def build_signal(
    panel: pd.DataFrame,
    *,
    move_quantile: float,
    prior_hhi_quantile: float,
    maximum_hhi_ratio: float,
    minimum_participation: float,
    minimum_flow_alignment: float,
    turnover_quantile: float,
    leader_decline_quantile: float,
) -> pd.DataFrame:
    active = (
        panel["clean"]
        & panel["median_return_abs"].ge(panel[f"move_q{int(round(move_quantile * 100))}"])
        & panel["prior_residual_hhi"].ge(
            panel[f"prior_hhi_q{int(round(prior_hhi_quantile * 100))}"]
        )
        & panel["hhi_ratio"].le(maximum_hhi_ratio)
        & panel["participation"].ge(minimum_participation)
        & panel["flow_alignment"].ge(minimum_flow_alignment)
        & panel["rank_turnover"].ge(
            panel[f"turnover_q{int(round(turnover_quantile * 100))}"]
        )
        & panel["leader_decline"].ge(
            panel[f"leader_decline_q{int(round(leader_decline_quantile * 100))}"]
        )
        & panel["leader_changed"]
        & panel["btc_lag"].gt(0.0)
    )
    side = pd.Series(0, index=panel.index, dtype=np.int8)
    side.loc[active] = panel.loc[active, "direction"].astype(np.int8)
    return pd.DataFrame(
        {
            "signal_date": panel["signal_date"],
            "feature_boundary": panel["feature_boundary"],
            "entry_date": panel["entry_date"],
            "side": side,
            "branch": np.where(
                side.gt(0),
                "alt_leadership_diffusion_long",
                np.where(side.lt(0), "alt_leadership_diffusion_short", "none"),
            ),
        }
    )


def quarterly_schedule(signal: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    btc_grid = pd.date_range(SELECTION_START, SELECTION_END - FIVE_MINUTES, freq="5min")
    position = pd.Series(np.arange(len(btc_grid), dtype=int), index=btc_grid)
    active = signal.loc[signal["side"].ne(0)].copy()
    for quarter in range(1, 5):
        previous_exit: pd.Timestamp | None = None
        quarter_rows = active.loc[pd.to_datetime(active["entry_date"]).dt.quarter.eq(quarter)]
        for row in quarter_rows.itertuples(index=False):
            signal_date = pd.Timestamp(row.signal_date)
            feature_boundary = pd.Timestamp(row.feature_boundary)
            entry_date = pd.Timestamp(row.entry_date)
            exit_date = entry_date + cfg.hold_bars * FIVE_MINUTES
            if exit_date >= SELECTION_END or exit_date.quarter != quarter:
                continue
            if previous_exit is not None and entry_date < previous_exit:
                continue
            if not (signal_date < feature_boundary < entry_date < exit_date):
                raise RuntimeError("CLD causal clock ordering failed")
            rows.append(
                {
                    "quarter": f"q{quarter}",
                    "signal_position": int(position.loc[signal_date]),
                    "entry_position": int(position.loc[entry_date]),
                    "exit_position": int(position.loc[exit_date]),
                    "signal_date": signal_date,
                    "feature_boundary": feature_boundary,
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "side": int(row.side),
                    "branch": str(row.branch),
                    "hold_bars": cfg.hold_bars,
                }
            )
            previous_exit = exit_date
    return pd.DataFrame(rows).sort_values("entry_date").reset_index(drop=True)


def support_summary(schedule: pd.DataFrame, cfg: Config) -> dict[str, Any]:
    entries = pd.to_datetime(schedule.get("entry_date", pd.Series(dtype="datetime64[ns]")))
    total = int(len(schedule))
    by_quarter = {
        f"q{quarter}": int((entries.dt.quarter == quarter).sum()) for quarter in range(1, 5)
    }
    h1 = int((entries < pd.Timestamp("2023-07-01")).sum())
    h2 = total - h1
    longs = int(schedule.get("side", pd.Series(dtype=int)).eq(1).sum())
    shorts = int(schedule.get("side", pd.Series(dtype=int)).eq(-1).sum())
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


def _contains_forbidden_outcome_key(value: Any) -> bool:
    forbidden = ("return", "pnl", "cagr", "mdd", "drawdown", "hit_rate", "payoff")
    if isinstance(value, dict):
        return any(
            any(token in str(key).lower() for token in forbidden)
            or _contains_forbidden_outcome_key(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden_outcome_key(item) for item in value)
    return False


def select_support_cell(cells: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    """Select only by mechanism strictness after outcome-blind support gates."""
    cells = list(cells)
    if _contains_forbidden_outcome_key(cells):
        raise ValueError("CLD support selection received a forbidden outcome field")
    passing = [cell for cell in cells if cell["support"]["passes"]]
    if not passing:
        return None
    return max(
        passing,
        key=lambda cell: (
            cell["minimum_participation"],
            cell["minimum_flow_alignment"],
            -cell["maximum_hhi_ratio"],
            cell["leader_decline_quantile"],
            cell["turnover_quantile"],
            cell["prior_hhi_quantile"],
            cell["move_quantile"],
        ),
    )


def _load_prior_clocks() -> dict[str, pd.DataFrame]:
    clocks: dict[str, pd.DataFrame] = {}
    for name, path in PRIOR_CLOCKS.items():
        if path.suffix == ".json":
            frame = pd.DataFrame(json.loads(path.read_text()).get("events", []))
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
        "feature_boundary",
        "entry_date",
        "exit_date",
        "side",
        "branch",
        "hold_bars",
    ]
    events = schedule[columns].to_dict("records") if len(schedule) else []
    for event in events:
        for key in ("signal_position", "entry_position", "exit_position", "side", "hold_bars"):
            event[key] = int(event[key])
        for key in ("signal_date", "feature_boundary", "entry_date", "exit_date"):
            event[key] = str(pd.Timestamp(event[key]))
    return {
        "protocol": "CLD-72 canonical outcome-blind event-clock freeze",
        "preregistration_source_sha256": sha256(PREREGISTRATION_SOURCE),
        "post_entry_outcomes_opened": False,
        "entry_or_later_ohlc_loaded": False,
        "selection_end_exclusive": str(SELECTION_END),
        "event_count": len(events),
        "event_clock_sha256": canonical_hash(events),
        "events": events,
    }


def _markdown(result: dict[str, Any]) -> str:
    selected = result["support_selection"]["selected_cell"]
    support = selected["support"] if selected else {}
    lines = [
        "# CLD-72 cross-sectional leadership diffusion — preregistration",
        "",
        "## Decision boundary",
        "",
        "- This artifact used signal-time incidence only; post-entry returns and PnL remain unopened.",
        "- The mechanism is a transition from concentrated alt leadership to broad aligned participation, not a static breadth level.",
        f"- Support status: **{'PASS' if result['all_support_gates_pass'] else 'FAIL'}**.",
        "",
        "## Frozen support cell",
        "",
    ]
    if selected:
        lines.extend(
            [
                f"- Move quantile: `{selected['move_quantile']}`",
                f"- Prior residual-HHI quantile: `{selected['prior_hhi_quantile']}`",
                f"- Maximum current/prior HHI ratio: `{selected['maximum_hhi_ratio']}`",
                f"- Minimum return participation: `{selected['minimum_participation']}`",
                f"- Minimum taker-flow alignment: `{selected['minimum_flow_alignment']}`",
                f"- Rank-turnover quantile: `{selected['turnover_quantile']}`",
                f"- Prior-leader decline quantile: `{selected['leader_decline_quantile']}`",
                f"- Non-overlapping events: `{support['nonoverlap_total']}` "
                f"({support['longs']} long / {support['shorts']} short)",
                f"- Quarter counts: `{support['by_quarter']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Causal contract",
            "",
            "- Every feature uses a completed UTC hour and strictly prior rolling thresholds.",
            "- Entry is delayed one full five-minute bar after the hour boundary; hold is six hours.",
            "- Signals are non-overlapping and contained inside each calendar quarter.",
            "- No entry/later OHLC, funding cash flow, PnL, CAGR, MDD, win rate, or payoff was computed.",
            "- A failed first strict 2023 evaluation retires this exact clock without threshold or direction repair.",
            "",
        ]
    )
    return "\n".join(lines)


def run_support(cfg: Config) -> dict[str, Any]:
    panel, source = build_feature_panel(cfg)
    cells: list[dict[str, Any]] = []
    schedules: dict[tuple[float, ...], pd.DataFrame] = {}
    grid = itertools.product(
        cfg.move_quantiles,
        cfg.prior_hhi_quantiles,
        cfg.maximum_hhi_ratios,
        cfg.minimum_participations,
        cfg.minimum_flow_alignments,
        cfg.turnover_quantiles,
        cfg.leader_decline_quantiles,
    )
    for parameters in grid:
        signal = build_signal(
            panel,
            move_quantile=parameters[0],
            prior_hhi_quantile=parameters[1],
            maximum_hhi_ratio=parameters[2],
            minimum_participation=parameters[3],
            minimum_flow_alignment=parameters[4],
            turnover_quantile=parameters[5],
            leader_decline_quantile=parameters[6],
        )
        schedule = quarterly_schedule(signal, cfg)
        schedules[parameters] = schedule
        cells.append(
            {
                "move_quantile": parameters[0],
                "prior_hhi_quantile": parameters[1],
                "maximum_hhi_ratio": parameters[2],
                "minimum_participation": parameters[3],
                "minimum_flow_alignment": parameters[4],
                "turnover_quantile": parameters[5],
                "leader_decline_quantile": parameters[6],
                "support": support_summary(schedule, cfg),
            }
        )
    selected = select_support_cell(cells)
    if selected is None:
        selected_schedule = quarterly_schedule(build_signal(
            panel,
            move_quantile=cfg.move_quantiles[0],
            prior_hhi_quantile=cfg.prior_hhi_quantiles[0],
            maximum_hhi_ratio=cfg.maximum_hhi_ratios[-1],
            minimum_participation=cfg.minimum_participations[0],
            minimum_flow_alignment=cfg.minimum_flow_alignments[0],
            turnover_quantile=cfg.turnover_quantiles[0],
            leader_decline_quantile=cfg.leader_decline_quantiles[0],
        ), cfg).iloc[0:0]
        independence: dict[str, Any] = {}
        independence_passes = False
    else:
        key = tuple(
            selected[name]
            for name in (
                "move_quantile",
                "prior_hhi_quantile",
                "maximum_hhi_ratio",
                "minimum_participation",
                "minimum_flow_alignment",
                "turnover_quantile",
                "leader_decline_quantile",
            )
        )
        selected_schedule = schedules[key]
        independence = {
            name: fuzzy_overlap(
                selected_schedule, clock, tolerance_bars=cfg.overlap_tolerance_bars
            )
            for name, clock in _load_prior_clocks().items()
        }
        independence_passes = all(
            metrics["jaccard"] <= cfg.maximum_prior_jaccard
            and metrics["new_clock_containment"] <= cfg.maximum_new_clock_containment
            for metrics in independence.values()
        )
    clock_payload = _event_clock_payload(selected_schedule)
    result = {
        "protocol": protocol(),
        "preregistration_source": str(PREREGISTRATION_SOURCE),
        "preregistration_source_sha256": sha256(PREREGISTRATION_SOURCE),
        "config": asdict(cfg),
        "frozen_sources": source,
        "support_selection": {
            "post_entry_outcomes_used": False,
            "cells": cells,
            "selected_cell": selected,
        },
        "outcome_blind_independence": independence,
        "event_clock_sha256": clock_payload["event_clock_sha256"],
        "all_support_gates_pass": bool(selected is not None and independence_passes),
        "next_action": (
            "freeze evaluator before opening calendar-2023 post-entry outcomes"
            if selected is not None and independence_passes
            else "retire CLD-72 before opening any post-entry outcome"
        ),
    }
    result["manifest_hash"] = canonical_hash(result)
    for path in (Path(cfg.output), Path(cfg.event_clock_output), Path(cfg.docs_output)):
        path.parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    Path(cfg.event_clock_output).write_text(
        json.dumps(clock_payload, indent=2, sort_keys=True) + "\n"
    )
    Path(cfg.docs_output).write_text(_markdown(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=Config.output)
    parser.add_argument("--event-clock-output", default=Config.event_clock_output)
    parser.add_argument("--docs-output", default=Config.docs_output)
    args = parser.parse_args()
    result = run_support(Config(**vars(args)))
    selected = result["support_selection"]["selected_cell"]
    print(
        json.dumps(
            {
                "all_support_gates_pass": result["all_support_gates_pass"],
                "selected_cell": selected,
                "event_clock_sha256": result["event_clock_sha256"],
                "post_entry_outcomes_opened": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
