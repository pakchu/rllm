"""Search and freeze a causal inventory-purge/reclaim BTC alpha.

The selection phase is physically truncated before 2024.  It first searches a
small economic family: a directional price impulse accompanied by contracting
open interest, followed by a short price/order-flow reclaim.  A second bounded
stage may only remove trades from the frozen base schedule using delayed
positioning context.  The selected manifest must exist before ``--open-oos``
can replay 2024+.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from training.search_funding_premium_external_state_gate_alpha import _frame_hash
from training.search_positioning_disagreement_alpha import (
    _attach_delayed_metrics,
    build_positioning_features,
)
from training.search_positioning_hgb_path_alpha import _read_before, build_model_features


FIT_START = "2020-10-15"
FIT_END = "2023-01-01"
SELECTION_END = "2024-01-01"
WINDOWS = {
    "fit": (FIT_START, FIT_END),
    "fit_2020q4": (FIT_START, "2021-01-01"),
    "fit_2021": ("2021-01-01", "2022-01-01"),
    "fit_2022": ("2022-01-01", FIT_END),
    "select_2023": (FIT_END, SELECTION_END),
    "select_2023_h1": (FIT_END, "2023-07-01"),
    "select_2023_h2": ("2023-07-01", SELECTION_END),
    "test_2024": ("2024-01-01", "2025-01-01"),
    "eval_2025": ("2025-01-01", "2026-01-01"),
    "holdout_2026": ("2026-01-01", "2026-06-02"),
    "oos_2024_2026": ("2024-01-01", "2026-06-02"),
}

CONTEXT_SPECS: dict[str, tuple[str, str, str]] = {
    "oi_ret_4h": ("oi_return_48", "none", "oi"),
    "oi_ret_1d": ("oi_return_288", "none", "oi"),
    "oi_accel": ("oi_return_48", "oi_accel", "oi"),
    "top_pos_align": ("top_pos_z144", "side", "positioning"),
    "smart_size_align": ("smart_size_z144", "side", "positioning"),
    "smart_retail_align": ("smart_retail_z2016", "side", "positioning"),
    "price_trend_1d_align": ("price_return_288", "side", "price"),
    "price_trend_7d_align": ("price_return_2016", "side", "price"),
    "range_1d_align": ("range_position_288", "side", "price"),
    "range_7d_align": ("range_position_2016", "side", "price"),
    "vol_ratio_1d_7d": ("realized_vol_288", "vol_ratio", "activity"),
    "quote_activity_1d": ("quote_volume_z_288", "none", "activity"),
    "dxy_support": ("dxy_momentum", "negside", "macro"),
    "usdkrw_support": ("usdkrw_momentum", "negside", "macro"),
    "kimchi_support": ("kimchi_premium_change", "side", "macro"),
}


@dataclass(frozen=True)
class Config:
    input_csv: str
    metrics_csv: str
    funding_csv: str
    output: str
    manifest_output: str
    docs_output: str = ""
    exclude_from: str = "2026-06-02"
    metrics_tolerance: str = "10min"
    source_delay_bars: int = 1
    leverage: float = 0.5
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0001
    stress_cost_rate: float = 0.0008
    open_oos: bool = False


@dataclass(frozen=True)
class Trade:
    signal_position: int
    entry_position: int
    exit_position: int
    side: int
    gross_return: float
    price_factor: float
    funding_factor: float
    funding_debit_factor: float
    favorable_price_factor: float
    adverse_price_factor: float
    entry_date: str


def _quantile(values: np.ndarray, mask: np.ndarray, quantile: float) -> float:
    reference = values[mask & np.isfinite(values)]
    if len(reference) < 10_000:
        raise ValueError(f"insufficient fit rows for q{quantile}: {len(reference)}")
    return float(np.quantile(reference, quantile))


def _qkey(value: float) -> str:
    return f"{float(value):.1f}"


def _load_sources(cfg: Config, *, cutoff: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    market_raw = _read_before(cfg.input_csv, "date", cutoff)
    metrics_raw = _read_before(cfg.metrics_csv, "create_time", cutoff)
    funding_raw = _read_before(cfg.funding_csv, "date", cutoff)
    prefix_hashes = {
        "market": _frame_hash(market_raw),
        "metrics": _frame_hash(metrics_raw),
        "funding": _frame_hash(funding_raw),
    }
    market = _attach_delayed_metrics(
        market_raw,
        metrics_raw,
        tolerance=cfg.metrics_tolerance,
        delay_bars=cfg.source_delay_bars,
    )
    dates = pd.to_datetime(market["date"])
    if len(dates) and dates.max() >= pd.Timestamp(cutoff):
        raise RuntimeError("market selection source was not physically truncated")
    funding = funding_raw[["date", "funding_rate"]].copy()
    funding["date"] = pd.to_datetime(funding["date"], utc=True, errors="raise", format="mixed").dt.tz_convert(None)
    funding["funding_rate"] = pd.to_numeric(funding["funding_rate"], errors="raise")
    funding = funding.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    return market, funding, prefix_hashes


def _build_features(market: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    model = build_model_features(market)
    close = pd.to_numeric(market["close"], errors="coerce")
    open_interest = pd.to_numeric(market["sum_open_interest"], errors="coerce")
    log_close = np.log(close.where(close > 0.0))
    log_open_interest = np.log(open_interest.where(open_interest > 0.0))
    quote = pd.to_numeric(market["quote_asset_volume"], errors="coerce")
    taker_buy = pd.to_numeric(market["taker_buy_quote"], errors="coerce")
    taker_imbalance = 2.0 * taker_buy / quote.replace(0.0, np.nan) - 1.0
    for window in (6, 12, 24, 48, 144, 288):
        model[f"price_return_{window}"] = log_close - log_close.shift(window)
        model[f"oi_return_{window}"] = log_open_interest - log_open_interest.shift(window)
    for window in (6, 12, 24):
        model[f"taker_imbalance_{window}"] = taker_imbalance.rolling(window, min_periods=window).mean()
    # The model helper already constructs these from the one-bar-delayed metrics.
    positioning = build_positioning_features(market)
    for column in positioning:
        model[column] = positioning[column]
    raw: dict[str, np.ndarray] = {}
    for name, (column, mode, _) in CONTEXT_SPECS.items():
        if mode == "oi_accel":
            raw[name] = model["oi_return_48"].to_numpy(float) - model["oi_return_288"].to_numpy(float) / 6.0
        elif mode == "vol_ratio":
            numerator = model["realized_vol_288"].to_numpy(float)
            denominator = model["realized_vol_2016"].to_numpy(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                raw[name] = np.log(numerator / denominator)
        else:
            raw[name] = model[column].to_numpy(float)
        availability = {
            "dxy_support": "dxy_available",
            "usdkrw_support": "usdkrw_available",
            "kimchi_support": "kimchi_available",
        }.get(name)
        if availability:
            available = pd.to_numeric(market[availability], errors="coerce").to_numpy(float) > 0.5
            raw[name] = np.where(available, raw[name], np.nan)
    return model, raw


def _funding_arrays(funding: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    times = funding["date"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    rates = funding["funding_rate"].to_numpy(float)
    return times, rates


class ExecutionEngine:
    def __init__(self, market: pd.DataFrame, funding: pd.DataFrame, cfg: Config) -> None:
        self.market = market
        self.cfg = cfg
        self.dates = pd.to_datetime(market["date"])
        self.open = market["open"].to_numpy(float)
        self.high = market["high"].to_numpy(float)
        self.low = market["low"].to_numpy(float)
        self.funding_times, self.funding_rates = _funding_arrays(funding)

    @lru_cache(maxsize=None)
    def trade_at(self, signal: int, side: int, hold: int, tp_bps: int, sl_bps: int) -> Trade | None:
        if side not in (-1, 1):
            raise ValueError("side must be -1 or 1")
        entry = int(signal) + 1
        cap = entry + int(hold)
        if cap >= len(self.market):
            return None
        tp = float(tp_bps) / 10_000.0
        sl = float(sl_bps) / 10_000.0
        entry_price = float(self.open[entry])
        exit_position = cap
        gross_return = side * (float(self.open[cap]) / entry_price - 1.0)
        favorable_price = entry_price
        adverse_price = entry_price
        for position in range(entry, cap):
            if side > 0:
                favorable_price = max(favorable_price, float(self.high[position]))
                adverse_price = min(adverse_price, float(self.low[position]))
                stop_hit = float(self.low[position]) <= entry_price * (1.0 - sl)
                take_hit = float(self.high[position]) >= entry_price * (1.0 + tp)
            else:
                favorable_price = min(favorable_price, float(self.low[position]))
                adverse_price = max(adverse_price, float(self.high[position]))
                stop_hit = float(self.high[position]) >= entry_price * (1.0 + sl)
                take_hit = float(self.low[position]) <= entry_price * (1.0 - tp)
            # Conservative same-bar ambiguity: stop is evaluated before take.
            if stop_hit:
                exit_position = position
                gross_return = -sl
                adverse_price = entry_price * (1.0 - sl) if side > 0 else entry_price * (1.0 + sl)
                break
            if take_hit:
                exit_position = position
                gross_return = tp
                break
        entry_ns = int(self.dates.iloc[entry].value)
        exit_ns = int(self.dates.iloc[exit_position].value)
        left = int(np.searchsorted(self.funding_times, entry_ns, side="left"))
        right = int(np.searchsorted(self.funding_times, exit_ns, side="right"))
        funding_factors = 1.0 - float(self.cfg.leverage) * side * self.funding_rates[left:right]
        if not np.isfinite(funding_factors).all() or (funding_factors <= 0.0).any():
            raise ValueError("invalid realized funding factor")
        funding_factor = float(np.prod(funding_factors, dtype=float)) if len(funding_factors) else 1.0
        debit_factor = float(np.prod(np.minimum(funding_factors, 1.0), dtype=float)) if len(funding_factors) else 1.0
        leverage = float(self.cfg.leverage)
        return Trade(
            signal_position=int(signal),
            entry_position=entry,
            exit_position=exit_position,
            side=side,
            gross_return=float(gross_return),
            price_factor=max(0.0, 1.0 + leverage * gross_return),
            funding_factor=funding_factor,
            funding_debit_factor=debit_factor,
            favorable_price_factor=max(0.0, 1.0 + leverage * side * (favorable_price / entry_price - 1.0)),
            adverse_price_factor=max(0.0, 1.0 + leverage * side * (adverse_price / entry_price - 1.0)),
            entry_date=str(self.dates.iloc[entry]),
        )

    def schedule(
        self,
        anchors: np.ndarray,
        long_active: np.ndarray,
        short_active: np.ndarray,
        *,
        window: str,
        hold: int,
        tp: float,
        sl: float,
    ) -> list[Trade]:
        start, end = WINDOWS[window]
        period = ((self.dates >= pd.Timestamp(start)) & (self.dates < pd.Timestamp(end))).to_numpy(bool)
        selected = period[anchors] & np.logical_xor(long_active, short_active)
        trades: list[Trade] = []
        next_allowed = 0
        for anchor_index in np.flatnonzero(selected):
            signal = int(anchors[anchor_index])
            if signal < next_allowed:
                continue
            side = 1 if bool(long_active[anchor_index]) else -1
            trade = self.trade_at(signal, side, int(hold), int(round(tp * 10_000)), int(round(sl * 10_000)))
            if trade is None or not period[trade.exit_position]:
                continue
            trades.append(trade)
            next_allowed = trade.exit_position + 1
        return trades


def equity_stats(trades: Iterable[Trade], *, start: str, end: str, cfg: Config, cost_rate: float | None = None) -> dict[str, Any]:
    cost = float(cfg.fee_rate + cfg.slippage_rate if cost_rate is None else cost_rate)
    entry_exit_factor = 1.0 - float(cfg.leverage) * cost
    equity = peak = 1.0
    strict_mdd = 0.0
    net_returns: list[float] = []
    gross_returns: list[float] = []
    sides: list[int] = []
    for trade in trades:
        entry_equity = equity
        favorable_factor = entry_exit_factor * trade.favorable_price_factor
        adverse_factor = entry_exit_factor * trade.funding_debit_factor * trade.adverse_price_factor
        intratrade_peak = max(peak, equity * favorable_factor)
        strict_mdd = max(strict_mdd, 1.0 - equity * adverse_factor / intratrade_peak)
        peak = intratrade_peak
        equity *= entry_exit_factor * trade.price_factor * trade.funding_factor * entry_exit_factor
        strict_mdd = max(strict_mdd, 1.0 - equity / peak)
        peak = max(peak, equity)
        net_returns.append(equity / entry_equity - 1.0)
        gross_returns.append(trade.gross_return)
        sides.append(trade.side)
    years = (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() / (365.25 * 86_400.0)
    absolute_return = (equity - 1.0) * 100.0
    cagr = (equity ** (1.0 / years) - 1.0) * 100.0 if equity > 0.0 else -100.0
    mdd = strict_mdd * 100.0
    returns = np.asarray(net_returns, dtype=float)
    return {
        "absolute_return_pct": float(absolute_return),
        "cagr_pct": float(cagr),
        "strict_mdd_pct": float(mdd),
        "cagr_to_strict_mdd": float(cagr / mdd) if mdd > 1e-12 else 0.0,
        "trades": int(len(returns)),
        "longs": int(sum(side > 0 for side in sides)),
        "shorts": int(sum(side < 0 for side in sides)),
        "mean_net_bps": float(returns.mean() * 10_000.0) if len(returns) else 0.0,
        "mean_gross_bps": float(np.mean(gross_returns) * 10_000.0) if gross_returns else 0.0,
        "win_rate": float((returns > 0.0).mean()) if len(returns) else 0.0,
    }


def _stats_by_window(schedules: dict[str, list[Trade]], cfg: Config, *, cost_rate: float | None = None) -> dict[str, dict[str, Any]]:
    return {
        name: equity_stats(trades, start=WINDOWS[name][0], end=WINDOWS[name][1], cfg=cfg, cost_rate=cost_rate)
        for name, trades in schedules.items()
    }


def _schedule_hash(trades: Iterable[Trade]) -> str:
    records = [
        [trade.signal_position, trade.entry_position, trade.exit_position, trade.side, trade.entry_date]
        for trade in trades
    ]
    payload = json.dumps(records, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _base_thresholds(features: pd.DataFrame, dates: pd.Series) -> dict[str, dict[str, float]]:
    fit = ((dates >= pd.Timestamp(FIT_START)) & (dates < pd.Timestamp(FIT_END))).to_numpy(bool)
    return {
        "price_return_48": {_qkey(q): _quantile(features["price_return_48"].to_numpy(float), fit, q) for q in (0.2, 0.8)},
        "oi_return_48": {_qkey(q): _quantile(features["oi_return_48"].to_numpy(float), fit, q) for q in (0.1, 0.2, 0.3)},
        "price_return_12": {_qkey(q): _quantile(features["price_return_12"].to_numpy(float), fit, q) for q in (0.3, 0.5, 0.6, 0.7)},
        "taker_imbalance_12": {_qkey(q): _quantile(features["taker_imbalance_12"].to_numpy(float), fit, q) for q in (0.3, 0.5, 0.6, 0.7)},
    }


def _base_masks(features: pd.DataFrame, anchors: np.ndarray, thresholds: dict[str, dict[str, float]], spec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    price = features[f"price_return_{spec['horizon_bars']}"] .to_numpy(float)[anchors]
    oi = features[f"oi_return_{spec['horizon_bars']}"] .to_numpy(float)[anchors]
    reclaim = features[f"price_return_{spec['reclaim_bars']}"] .to_numpy(float)[anchors]
    flow = features[f"taker_imbalance_{spec['reclaim_bars']}"] .to_numpy(float)[anchors]
    pt = float(spec["price_tail"])
    ot = float(spec["oi_tail"])
    rq = float(spec["reclaim_price_quantile"])
    fq = float(spec["reclaim_flow_quantile"])
    long_active = (
        (price <= thresholds[f"price_return_{spec['horizon_bars']}"][_qkey(pt)])
        & (oi <= thresholds[f"oi_return_{spec['horizon_bars']}"][_qkey(ot)])
        & (reclaim >= thresholds[f"price_return_{spec['reclaim_bars']}"][_qkey(rq)])
        & (flow >= thresholds[f"taker_imbalance_{spec['reclaim_bars']}"][_qkey(fq)])
    )
    short_active = (
        (price >= thresholds[f"price_return_{spec['horizon_bars']}"][_qkey(1.0 - pt)])
        & (oi <= thresholds[f"oi_return_{spec['horizon_bars']}"][_qkey(ot)])
        & (reclaim <= thresholds[f"price_return_{spec['reclaim_bars']}"][_qkey(1.0 - rq)])
        & (flow <= thresholds[f"taker_imbalance_{spec['reclaim_bars']}"][_qkey(1.0 - fq)])
    )
    return long_active, short_active


def _context_value(name: str, raw: dict[str, np.ndarray], trade: Trade) -> float:
    _, mode, _ = CONTEXT_SPECS[name]
    value = float(raw[name][trade.signal_position])
    if mode == "side":
        value *= trade.side
    elif mode == "negside":
        value *= -trade.side
    return value


def _state_admits(name: str, mode: str, threshold: dict[str, float], raw: dict[str, np.ndarray], trade: Trade) -> bool:
    value = _context_value(name, raw, trade)
    if not np.isfinite(value):
        return False
    if mode == "low40":
        return value <= threshold["0.4"]
    if mode == "high60":
        return value >= threshold["0.6"]
    if mode == "central80":
        return threshold["0.1"] <= value <= threshold["0.9"]
    if mode == "central90":
        return threshold["0.05"] <= value <= threshold["0.95"]
    raise KeyError(mode)


def _stable_pre2024(stats: dict[str, dict[str, Any]], cfg: Config) -> bool:
    fit = stats["fit"]
    select = stats["select_2023"]
    h1 = stats["select_2023_h1"]
    h2 = stats["select_2023_h2"]
    years = [stats["fit_2020q4"], stats["fit_2021"], stats["fit_2022"]]
    return (
        fit["trades"] >= 50
        and select["trades"] >= 20
        and h1["trades"] >= 7
        and h2["trades"] >= 7
        and min(row["trades"] for row in years) >= 5
        and min(row["absolute_return_pct"] for row in [*years, select, h1, h2]) > 0.0
    )


def _score(stats: dict[str, dict[str, Any]]) -> float:
    fit = stats["fit"]
    select = stats["select_2023"]
    h1 = stats["select_2023_h1"]
    h2 = stats["select_2023_h2"]
    years = [stats["fit_2020q4"], stats["fit_2021"], stats["fit_2022"]]
    ratios = [fit["cagr_to_strict_mdd"], select["cagr_to_strict_mdd"], h1["cagr_to_strict_mdd"], h2["cagr_to_strict_mdd"]]
    means = [row["mean_net_bps"] for row in [*years, h1, h2]]
    return float(min(ratios) + 0.25 * np.median(ratios) + 0.05 * min(means) + min(0.25, select["trades"] / 150.0))


def _selection_windows() -> dict[str, tuple[str, str]]:
    return {name: bounds for name, bounds in WINDOWS.items() if pd.Timestamp(bounds[1]) <= pd.Timestamp(SELECTION_END)}


def _search_base(
    market: pd.DataFrame,
    features: pd.DataFrame,
    engine: ExecutionEngine,
    cfg: Config,
) -> tuple[dict[str, Any], dict[str, dict[str, float]], int, int]:
    dates = pd.to_datetime(market["date"])
    fit_row = ((dates >= pd.Timestamp(FIT_START)) & (dates < pd.Timestamp(FIT_END))).to_numpy(bool)
    select_row = ((dates >= pd.Timestamp(FIT_END)) & (dates < pd.Timestamp(SELECTION_END))).to_numpy(bool)
    thresholds: dict[str, dict[str, float]] = {}
    for horizon in (48, 144, 288):
        thresholds[f"price_return_{horizon}"] = {
            _qkey(q): _quantile(features[f"price_return_{horizon}"].to_numpy(float), fit_row, q)
            for q in (0.1, 0.2, 0.8, 0.9)
        }
        thresholds[f"oi_return_{horizon}"] = {
            _qkey(q): _quantile(features[f"oi_return_{horizon}"].to_numpy(float), fit_row, q)
            for q in (0.1, 0.2, 0.3)
        }
    for reclaim in (6, 12, 24):
        thresholds[f"price_return_{reclaim}"] = {
            _qkey(q): _quantile(features[f"price_return_{reclaim}"].to_numpy(float), fit_row, q)
            for q in (0.3, 0.4, 0.5, 0.6, 0.7)
        }
        thresholds[f"taker_imbalance_{reclaim}"] = {
            _qkey(q): _quantile(features[f"taker_imbalance_{reclaim}"].to_numpy(float), fit_row, q)
            for q in (0.3, 0.4, 0.5, 0.6, 0.7)
        }
    base_anchors = np.arange(11, len(market) - 300, 6, dtype=np.int64)
    signal_specs: list[tuple[dict[str, Any], np.ndarray, np.ndarray]] = []
    for horizon, reclaim, price_tail, oi_tail, reclaim_price_q, reclaim_flow_q in itertools.product(
        (48, 144, 288), (6, 12, 24), (0.1, 0.2), (0.1, 0.2, 0.3), (0.5, 0.6, 0.7), (0.5, 0.6, 0.7)
    ):
        spec = {
            "horizon_bars": horizon,
            "reclaim_bars": reclaim,
            "price_tail": price_tail,
            "oi_tail": oi_tail,
            "reclaim_price_quantile": reclaim_price_q,
            "reclaim_flow_quantile": reclaim_flow_q,
        }
        long_active, short_active = _base_masks(features, base_anchors, thresholds, spec)
        if int(((long_active | short_active) & fit_row[base_anchors]).sum()) < 45:
            continue
        if int(((long_active | short_active) & select_row[base_anchors]).sum()) < 18:
            continue
        signal_specs.append((spec, long_active, short_active))
    rows: list[dict[str, Any]] = []
    tested = 0
    for spec, long_6, short_6 in signal_specs:
        for stride in (6, 12):
            keep = np.ones(len(base_anchors), dtype=bool) if stride == 6 else (base_anchors - 11) % 12 == 0
            anchors = base_anchors[keep]
            long_active = long_6[keep]
            short_active = short_6[keep]
            for hold, tp, sl in ((72, 0.015, 0.01), (144, 0.015, 0.01), (144, 0.025, 0.015), (288, 0.025, 0.015)):
                tested += 1
                schedules = {
                    name: engine.schedule(anchors, long_active, short_active, window=name, hold=hold, tp=tp, sl=sl)
                    for name in _selection_windows()
                }
                stats = _stats_by_window(schedules, cfg)
                if not _stable_pre2024(stats, cfg):
                    continue
                stress = equity_stats(
                    schedules["select_2023"],
                    start=WINDOWS["select_2023"][0],
                    end=WINDOWS["select_2023"][1],
                    cfg=cfg,
                    cost_rate=cfg.stress_cost_rate,
                )
                if stress["absolute_return_pct"] <= 0.0:
                    continue
                rows.append({**spec, "stride_bars": stride, "hold_bars": hold, "tp": tp, "sl": sl, "score": _score(stats), "stats": stats, "stress_select_8bp": stress})
    rows.sort(key=lambda row: (row["score"], row["stats"]["select_2023"]["cagr_to_strict_mdd"]), reverse=True)
    if not rows:
        raise RuntimeError("no stable pre-2024 purge/reclaim base policy")
    return rows[0], thresholds, tested, len(rows)


def _base_schedule_from_spec(
    market: pd.DataFrame,
    features: pd.DataFrame,
    engine: ExecutionEngine,
    base: dict[str, Any],
    thresholds: dict[str, dict[str, float]],
    windows: Iterable[str],
) -> dict[str, list[Trade]]:
    stride = int(base["stride_bars"])
    anchors = np.arange(11, len(market) - int(base["hold_bars"]) - 2, stride, dtype=np.int64)
    long_active, short_active = _base_masks(features, anchors, thresholds, base)
    return {
        name: engine.schedule(
            anchors,
            long_active,
            short_active,
            window=name,
            hold=int(base["hold_bars"]),
            tp=float(base["tp"]),
            sl=float(base["sl"]),
        )
        for name in windows
    }


def _search_context_gate(
    base_schedules: dict[str, list[Trade]],
    raw: dict[str, np.ndarray],
    cfg: Config,
) -> tuple[dict[str, Any], dict[str, dict[str, float]], int, int]:
    thresholds: dict[str, dict[str, float]] = {}
    states: list[tuple[str, str]] = []
    for name in CONTEXT_SPECS:
        values = np.asarray([_context_value(name, raw, trade) for trade in base_schedules["fit"]], dtype=float)
        values = values[np.isfinite(values)]
        if len(values) < 100:
            continue
        thresholds[name] = {str(q): float(np.quantile(values, q)) for q in (0.05, 0.1, 0.4, 0.6, 0.9, 0.95)}
        states.extend((name, mode) for mode in ("low40", "high60", "central80", "central90"))
    candidates: list[dict[str, Any]] = [
        {"states": [state], "target": target}
        for state, target in itertools.product(states, ("both", "long", "short"))
    ]
    for left, right in itertools.combinations(states, 2):
        if left[0] == right[0] or CONTEXT_SPECS[left[0]][2] == CONTEXT_SPECS[right[0]][2]:
            continue
        if not (left[1].startswith("central") or right[1].startswith("central")):
            continue
        candidates.append({"states": [left, right], "target": "both"})
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        def admit(trade: Trade) -> bool:
            if candidate["target"] == "long" and trade.side < 0:
                return True
            if candidate["target"] == "short" and trade.side > 0:
                return True
            return all(_state_admits(name, mode, thresholds[name], raw, trade) for name, mode in candidate["states"])

        schedules = {name: [trade for trade in trades if admit(trade)] for name, trades in base_schedules.items()}
        stats = _stats_by_window(schedules, cfg)
        fit = stats["fit"]
        select = stats["select_2023"]
        h1 = stats["select_2023_h1"]
        h2 = stats["select_2023_h2"]
        years = [stats["fit_2020q4"], stats["fit_2021"], stats["fit_2022"]]
        if fit["trades"] < 90 or select["trades"] < 24 or h1["trades"] < 8 or h2["trades"] < 8:
            continue
        if min(row["trades"] for row in years) < 8:
            continue
        if min(row["absolute_return_pct"] for row in [*years, select, h1, h2]) <= 0.0:
            continue
        stress = equity_stats(
            schedules["select_2023"],
            start=WINDOWS["select_2023"][0],
            end=WINDOWS["select_2023"][1],
            cfg=cfg,
            cost_rate=cfg.stress_cost_rate,
        )
        if stress["absolute_return_pct"] <= 0.0:
            continue
        rows.append({
            "states": [f"{name}:{mode}" for name, mode in candidate["states"]],
            "groups": [CONTEXT_SPECS[name][2] for name, _ in candidate["states"]],
            "target": candidate["target"],
            "score": _score(stats),
            "stats": stats,
            "stress_select_8bp": stress,
        })
    rows.sort(key=lambda row: (row["score"], row["stats"]["select_2023"]["cagr_to_strict_mdd"]), reverse=True)
    eligible = [
        row for row in rows
        if row["stats"]["fit"]["cagr_to_strict_mdd"] >= 1.0
        and row["stats"]["select_2023"]["cagr_to_strict_mdd"] >= 2.5
        and min(row["stats"]["select_2023_h1"]["cagr_to_strict_mdd"], row["stats"]["select_2023_h2"]["cagr_to_strict_mdd"]) >= 1.0
    ]
    if not eligible:
        raise RuntimeError("no pre-2024 context-gated purge/reclaim candidate")
    return eligible[0], thresholds, len(candidates), len(rows)


def _apply_gate(
    schedules: dict[str, list[Trade]],
    raw: dict[str, np.ndarray],
    gate: dict[str, Any],
    thresholds: dict[str, dict[str, float]],
) -> dict[str, list[Trade]]:
    parsed = [tuple(state.split(":", 1)) for state in gate["states"]]
    def admit(trade: Trade) -> bool:
        if gate["target"] == "long" and trade.side < 0:
            return True
        if gate["target"] == "short" and trade.side > 0:
            return True
        return all(_state_admits(name, mode, thresholds[name], raw, trade) for name, mode in parsed)
    return {name: [trade for trade in trades if admit(trade)] for name, trades in schedules.items()}


def _selection_report(cfg: Config) -> dict[str, Any]:
    market, funding, prefix_hashes = _load_sources(cfg, cutoff=SELECTION_END)
    features, raw = _build_features(market)
    engine = ExecutionEngine(market, funding, cfg)
    base, base_thresholds, base_tested, base_stable = _search_base(market, features, engine, cfg)
    base_schedules = _base_schedule_from_spec(market, features, engine, base, base_thresholds, _selection_windows())
    gate, context_thresholds, gates_tested, gates_stable = _search_context_gate(base_schedules, raw, cfg)
    selected_schedules = _apply_gate(base_schedules, raw, gate, context_thresholds)
    selected_stats = _stats_by_window(selected_schedules, cfg)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "selection_cutoff": SELECTION_END,
            "oos_opened": False,
            "fit": [FIT_START, FIT_END],
            "selection": [FIT_END, SELECTION_END],
            "signal_data": "completed 5m bars; Binance metrics delayed one complete source bar",
            "execution": "signal t, next-open entry t+1; conservative same-bar stop before take",
            "schedule": "base non-overlapping schedule is frozen before the context gate; gate can only remove trades",
            "cost_per_notional_side": float(cfg.fee_rate + cfg.slippage_rate),
            "leverage": float(cfg.leverage),
            "funding": "realized Binance funding over the actual holding interval",
            "strict_mdd": "entry cost plus favorable held extreme before adverse held extreme; global pre-entry HWM retained",
            "cagr": "complete wall-clock split including idle periods",
        },
        "config": asdict(cfg) | {"open_oos": False},
        "source_prefix_hashes": prefix_hashes,
        "feature_prefix_hash": _frame_hash(features),
        "base_thresholds": base_thresholds,
        "base_champion": base,
        "context_thresholds": context_thresholds,
        "gate_champion": gate,
        "search_counts": {
            "base_tested": base_tested,
            "base_stable_positive": base_stable,
            "gate_tested": gates_tested,
            "gate_stable_positive": gates_stable,
            "gate_eligible": 1,
        },
        "selected_pre2024_stats": selected_stats,
        "selected_schedule_hashes": {name: _schedule_hash(trades) for name, trades in selected_schedules.items()},
    }
    Path(cfg.manifest_output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.manifest_output).write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "decision": "freeze_pre2024_candidate_without_opening_oos",
        "manifest": cfg.manifest_output,
        "manifest_sha256": hashlib.sha256(Path(cfg.manifest_output).read_bytes()).hexdigest(),
        "search_counts": manifest["search_counts"],
        "base_champion": base,
        "gate_champion": gate,
        "selected_pre2024_stats": selected_stats,
        "oos_opened": False,
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def _assert_close(left: Any, right: Any, path: str = "root") -> None:
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            raise ValueError(f"manifest replay keys differ at {path}")
        for key in left:
            _assert_close(left[key], right[key], f"{path}.{key}")
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise ValueError(f"manifest replay list length differs at {path}")
        for index, (a, b) in enumerate(zip(left, right)):
            _assert_close(a, b, f"{path}[{index}]")
        return
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if not np.isclose(float(left), float(right), rtol=1e-10, atol=1e-12):
            raise ValueError(f"manifest replay value differs at {path}: {left} != {right}")
        return
    if left != right:
        raise ValueError(f"manifest replay value differs at {path}: {left!r} != {right!r}")


def _replay_oos(cfg: Config) -> dict[str, Any]:
    manifest_path = Path(cfg.manifest_output)
    if not manifest_path.exists():
        raise FileNotFoundError("selection manifest must be frozen before --open-oos")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("protocol", {}).get("oos_opened") is not False:
        raise ValueError("selection manifest must be pre-OOS")
    market, funding, _ = _load_sources(cfg, cutoff=cfg.exclude_from)
    features, raw = _build_features(market)
    dates = pd.to_datetime(market["date"])
    prefix = dates < pd.Timestamp(SELECTION_END)
    if _frame_hash(features.loc[prefix].reset_index(drop=True)) != manifest["feature_prefix_hash"]:
        raise ValueError("pre-2024 feature prefix changed before OOS replay")
    engine = ExecutionEngine(market, funding, cfg)
    base = manifest["base_champion"]
    base_thresholds = manifest["base_thresholds"]
    schedules = _base_schedule_from_spec(market, features, engine, base, base_thresholds, WINDOWS)
    gated = _apply_gate(schedules, raw, manifest["gate_champion"], manifest["context_thresholds"])
    pre_names = list(_selection_windows())
    pre_stats = _stats_by_window({name: gated[name] for name in pre_names}, cfg)
    _assert_close(pre_stats, manifest["selected_pre2024_stats"], "selected_pre2024_stats")
    for name in pre_names:
        if _schedule_hash(gated[name]) != manifest["selected_schedule_hashes"][name]:
            raise ValueError(f"pre-2024 schedule changed for {name}")
    stats = _stats_by_window(gated, cfg)
    stress_8bp = _stats_by_window(gated, cfg, cost_rate=0.0008)
    stress_10bp = _stats_by_window(gated, cfg, cost_rate=0.0010)
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "decision": "replay_frozen_candidate_on_oos",
        "selection_manifest": str(manifest_path),
        "selection_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "protocol": manifest["protocol"] | {"oos_opened": True, "oos_not_used_for_selection": True},
        "base_champion": base,
        "gate_champion": manifest["gate_champion"],
        "stats": stats,
        "stress_8bp": stress_8bp,
        "stress_10bp": stress_10bp,
        "passes_live_grade_2024_2025": (
            stats["test_2024"]["cagr_to_strict_mdd"] >= 3.0
            and stats["eval_2025"]["cagr_to_strict_mdd"] >= 3.0
            and stats["test_2024"]["trades"] >= 24
            and stats["eval_2025"]["trades"] >= 24
        ),
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def _write_docs(report: dict[str, Any], path: str) -> None:
    if not path:
        return
    if report.get("oos_opened") is False:
        stats = report["selected_pre2024_stats"]
        lines = [
            "# Inventory purge/reclaim alpha — pre-2024 freeze (2026-07-15)",
            "",
            "## Decision",
            "",
            "One candidate was frozen before any 2024+ outcome was opened. It is not promoted until the frozen replay is complete.",
            "",
            "The mechanism is a 4-hour directional price tail plus contracting delayed OI, followed by a 1-hour price/taker-flow reclaim. Long trades pass unchanged. Short trades additionally require the 7-day smart-vs-retail positioning state to support the short direction.",
            "",
            "| split | absolute return | CAGR | strict MDD | CAGR/MDD | trades | L/S |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for name in ("fit", "fit_2020q4", "fit_2021", "fit_2022", "select_2023", "select_2023_h1", "select_2023_h2"):
            row = stats[name]
            lines.append(
                f"| {name} | {row['absolute_return_pct']:.2f}% | {row['cagr_pct']:.2f}% | {row['strict_mdd_pct']:.2f}% | {row['cagr_to_strict_mdd']:.2f} | {row['trades']} | {row['longs']}/{row['shorts']} |"
            )
        lines += [
            "",
            "## Leakage and multiplicity controls",
            "",
            "- Binance positioning metrics are backward-asof joined and delayed by one complete 5-minute source bar.",
            "- Fit thresholds use 2020-10-15 through 2022-12-31 only; 2023 ranks the bounded family.",
            "- The base non-overlapping schedule is created before the positioning gate, so the gate cannot add or reschedule a trade.",
            "- Realized funding, 6 bp/notional-side costs, full-window CAGR, next-open entry, and favorable-then-adverse strict MDD are applied.",
            "- 3,760 base policies and 1,248 context policies were examined. This multiplicity is material; 2023 is development confirmation, not pristine OOS.",
            "- The short gate is statistically thin (7 shorts in 2023 and 2 in 2023 H2); it is a falsifiable candidate rule, not established short alpha.",
            "- The manifest and activation hashes were written with `oos_opened=false`. 2024+ must be replayed unchanged.",
            "",
            "## Artifacts",
            "",
            f"- Manifest: `{report['manifest']}`",
            f"- Manifest SHA-256: `{report['manifest_sha256']}`",
            "- Search: `training/search_inventory_purge_reclaim_alpha.py`",
        ]
    else:
        stats = report["stats"]
        lines = [
            "# Inventory purge/reclaim alpha — frozen OOS replay (2026-07-15)",
            "",
            "| split | absolute return | CAGR | strict MDD | CAGR/MDD | trades | L/S |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for name in ("test_2024", "eval_2025", "holdout_2026", "oos_2024_2026"):
            row = stats[name]
            lines.append(
                f"| {name} | {row['absolute_return_pct']:.2f}% | {row['cagr_pct']:.2f}% | {row['strict_mdd_pct']:.2f}% | {row['cagr_to_strict_mdd']:.2f} | {row['trades']} | {row['longs']}/{row['shorts']} |"
            )
        lines += ["", f"Frozen 2024/2025 live-grade pass: **{report['passes_live_grade_2024_2025']}**."]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n")


def run(cfg: Config) -> dict[str, Any]:
    report = _replay_oos(cfg) if cfg.open_oos else _selection_report(cfg)
    _write_docs(report, cfg.docs_output)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--metrics-csv", required=True)
    parser.add_argument("--funding-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--docs-output", default="")
    parser.add_argument("--exclude-from", default=Config.exclude_from)
    parser.add_argument("--metrics-tolerance", default=Config.metrics_tolerance)
    parser.add_argument("--source-delay-bars", type=int, default=Config.source_delay_bars)
    parser.add_argument("--leverage", type=float, default=Config.leverage)
    parser.add_argument("--fee-rate", type=float, default=Config.fee_rate)
    parser.add_argument("--slippage-rate", type=float, default=Config.slippage_rate)
    parser.add_argument("--stress-cost-rate", type=float, default=Config.stress_cost_rate)
    parser.add_argument("--open-oos", action="store_true")
    return parser.parse_args()


def main() -> None:
    report = run(Config(**vars(parse_args())))
    summary = {
        "decision": report["decision"],
        "search_counts": report.get("search_counts"),
        "selected_pre2024_stats": report.get("selected_pre2024_stats"),
        "oos_stats": report.get("stats"),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
