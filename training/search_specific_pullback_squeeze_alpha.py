"""Audit a causal weak-signal interaction for a BTC long squeeze setup.

The rule is deliberately structural rather than a weighted blend:

* a funding/trend or premium/momentum squeeze establishes the opportunity;
* the signal must occur in the lower part of the trailing 48-hour range; and
* source-specific completed higher-timeframe returns reject already-overheated
  funding and premium events.

All quantiles are fitted on 2020H2-2022.  The rule is selected on 2023 and is
then replayed on 2024 and 2025-2026.  Execution is next-bar, non-overlapping,
and bar-by-bar with conservative favorable-then-adverse intrabar MDD marking.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.search_causal_online_expert_alpha import OnlineExpertConfig, _load_bundle
from training.strict_bar_backtest import _mark_worst_order_bar_path, _trade_stats


FIT_START = pd.Timestamp("2020-07-01")
FIT_END = pd.Timestamp("2023-01-01")
FINAL_END = pd.Timestamp("2026-05-31 15:00:01")
WINDOWS: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {
    "train": (FIT_START, FIT_END),
    "select2023": (pd.Timestamp("2023-01-01"), pd.Timestamp("2024-01-01")),
    "pre2024": (FIT_START, pd.Timestamp("2024-01-01")),
    "test2024": (pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01")),
    "eval2025_2026": (pd.Timestamp("2025-01-01"), FINAL_END),
    "oos2024_2026": (pd.Timestamp("2024-01-01"), FINAL_END),
    "full": (FIT_START, FINAL_END),
}
for _year in range(2020, 2027):
    _start = max(FIT_START, pd.Timestamp(f"{_year}-01-01"))
    _end = min(FINAL_END, pd.Timestamp(f"{_year + 1}-01-01"))
    if _start < _end:
        WINDOWS[str(_year)] = (_start, _end)
    for _half, (_month, _next_month) in {
        "h1": (1, 7),
        "h2": (7, 13),
    }.items():
        _half_start = pd.Timestamp(year=_year, month=_month, day=1)
        _half_end = (
            pd.Timestamp(year=_year + 1, month=1, day=1)
            if _next_month == 13
            else pd.Timestamp(year=_year, month=_next_month, day=1)
        )
        _half_start = max(FIT_START, _half_start)
        _half_end = min(FINAL_END, _half_end)
        if _half_start < _half_end:
            WINDOWS[f"{_year}{_half}"] = (_half_start, _half_end)


@dataclass(frozen=True)
class PullbackSqueezeConfig:
    input_csv: str = "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz"
    funding_csv: str = "data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz"
    premium_csv: str = "data/binance_um_aux_btc_2020_2026/BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz"
    output: str = "results/specific_pullback_squeeze_alpha_audit_2026-07-15.json"
    exclude_from: str = "2026-06-02"
    window_size: int = 144
    hold_bars: int = 576
    stride_bars: int = 12
    entry_delay_bars: int = 1
    leverage: float = 0.5
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0001
    range_quantile: float = 0.60
    overheat_quantile: float = 0.70
    bootstrap_samples: int = 30_000


def _window_mask(dates: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
    return ((dates >= start) & (dates < end)).to_numpy(bool)


def _fit_quantile(
    values: np.ndarray,
    fit_mask: np.ndarray,
    quantile: float,
    *,
    minimum: int = 100,
) -> float:
    reference = values[fit_mask & np.isfinite(values)]
    if len(reference) < minimum:
        raise ValueError(f"insufficient fit observations: {len(reference)} < {minimum}")
    return float(np.quantile(reference, quantile))


def fit_rule_masks(
    features: pd.DataFrame,
    dates: pd.Series,
    decision_mask: np.ndarray,
    *,
    fit_start: pd.Timestamp = FIT_START,
    fit_end: pd.Timestamp = FIT_END,
    range_quantile: float = 0.60,
    overheat_quantile: float = 0.70,
) -> dict[str, Any]:
    """Fit train-only quantiles and return source masks plus their union."""

    required = {
        "funding_available",
        "funding_rate",
        "trend_96",
        "premium_available",
        "premium_index_change",
        "htf_1d_return_4",
        "rex_576_range_pos",
        "htf_1d_return_1",
        "htf_3d_return_1",
    }
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"missing pullback-squeeze features: {sorted(missing)}")
    if len(features) != len(dates) or len(decision_mask) != len(features):
        raise ValueError("features, dates and decision_mask must have equal length")

    fit_clock = decision_mask & _window_mask(dates, fit_start, fit_end)
    funding_available = pd.to_numeric(features["funding_available"], errors="coerce").to_numpy(float) > 0.5
    premium_available = pd.to_numeric(features["premium_available"], errors="coerce").to_numpy(float) > 0.5

    def values(name: str) -> np.ndarray:
        return pd.to_numeric(features[name], errors="coerce").to_numpy(float)

    funding_fit = fit_clock & funding_available
    premium_fit = fit_clock & premium_available
    base_thresholds = {
        "funding_rate_q10": _fit_quantile(values("funding_rate"), funding_fit, 0.10),
        "trend_96_q70": _fit_quantile(values("trend_96"), funding_fit, 0.70),
        "premium_change_q20": _fit_quantile(values("premium_index_change"), premium_fit, 0.20),
        "daily_momentum_4_q90": _fit_quantile(values("htf_1d_return_4"), premium_fit, 0.90),
    }
    funding = (
        decision_mask
        & funding_available
        & (values("funding_rate") <= base_thresholds["funding_rate_q10"])
        & (values("trend_96") >= base_thresholds["trend_96_q70"])
    )
    premium = (
        decision_mask
        & premium_available
        & (values("premium_index_change") <= base_thresholds["premium_change_q20"])
        & (values("htf_1d_return_4") >= base_thresholds["daily_momentum_4_q90"])
    )
    base = funding | premium
    event_fit = base & _window_mask(dates, fit_start, fit_end)
    context_thresholds = {
        "rex_576_range_pos": _fit_quantile(values("rex_576_range_pos"), event_fit, range_quantile),
        "funding_daily_overheat": _fit_quantile(values("htf_1d_return_1"), event_fit, overheat_quantile),
        "premium_3d_overheat": _fit_quantile(values("htf_3d_return_1"), event_fit, overheat_quantile),
        "range_quantile": float(range_quantile),
        "overheat_quantile": float(overheat_quantile),
    }
    pullback = values("rex_576_range_pos") <= context_thresholds["rex_576_range_pos"]
    funding_active = funding & pullback & (
        values("htf_1d_return_1") <= context_thresholds["funding_daily_overheat"]
    )
    premium_active = premium & pullback & (
        values("htf_3d_return_1") <= context_thresholds["premium_3d_overheat"]
    )
    return {
        "base_thresholds": base_thresholds,
        "context_thresholds": context_thresholds,
        "funding_active": funding_active,
        "premium_active": premium_active,
        "active": funding_active | premium_active,
    }


def simulate_mask(
    market: pd.DataFrame,
    dates: pd.Series,
    active: np.ndarray,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    hold_bars: int,
    entry_delay_bars: int,
    leverage: float,
    fee_rate: float,
    slippage_rate: float,
) -> dict[str, Any]:
    """Execute a long-only mask with strict bar-by-bar MDD and period purge."""

    period = _window_mask(dates, start, end)
    candidates = np.flatnonzero(active & period)
    opens = market["open"].to_numpy(float)
    highs = market["high"].to_numpy(float)
    lows = market["low"].to_numpy(float)
    cost = (float(fee_rate) + float(slippage_rate)) * float(leverage)
    equity = peak = 1.0
    strict_mdd = 0.0
    next_allowed = -1
    trade_returns: list[float] = []
    signal_positions: list[int] = []

    for signal_pos in candidates:
        if signal_pos < next_allowed:
            continue
        entry_pos = int(signal_pos) + int(entry_delay_bars)
        exit_pos = entry_pos + int(hold_bars)
        if entry_pos >= len(market) - 1 or exit_pos >= len(market) or not period[exit_pos]:
            continue
        entry_equity = equity
        equity *= max(0.0, 1.0 - cost)
        strict_mdd = max(strict_mdd, 1.0 - equity / peak)
        for bar in range(entry_pos, exit_pos):
            open_price = float(opens[bar])
            if open_price <= 0.0:
                continue
            peak, bar_mdd = _mark_worst_order_bar_path(
                equity_at_open=equity,
                peak=peak,
                open_price=open_price,
                high_price=float(highs[bar]),
                low_price=float(lows[bar]),
                signal=1,
                leverage=float(leverage),
            )
            strict_mdd = max(strict_mdd, bar_mdd)
            equity *= max(0.0, 1.0 + float(leverage) * (float(opens[bar + 1]) / open_price - 1.0))
            peak = max(peak, equity)
        equity *= max(0.0, 1.0 - cost)
        strict_mdd = max(strict_mdd, 1.0 - equity / peak)
        peak = max(peak, equity)
        trade_returns.append(equity / entry_equity - 1.0)
        signal_positions.append(int(signal_pos))
        next_allowed = exit_pos

    years = max(1.0 / 365.25, (end - start).total_seconds() / (365.25 * 86400.0))
    absolute_return = (equity - 1.0) * 100.0
    cagr = (equity ** (1.0 / years) - 1.0) * 100.0 if equity > 0.0 else -100.0
    mdd = strict_mdd * 100.0
    return {
        "period": {"start": str(start), "end": str(end), "years": years},
        "absolute_return_pct": float(absolute_return),
        "cagr_pct": float(cagr),
        "strict_mdd_pct": float(mdd),
        "cagr_to_strict_mdd": float(cagr / mdd) if mdd > 1e-12 else 0.0,
        "trade_count": len(trade_returns),
        "trade_statistics": _trade_stats(trade_returns),
        "signal_positions": signal_positions,
        "trade_returns": trade_returns,
    }


def _metrics(
    market: pd.DataFrame,
    dates: pd.Series,
    active: np.ndarray,
    cfg: PullbackSqueezeConfig,
    *,
    hold_bars: int | None = None,
    entry_delay_bars: int | None = None,
    slippage_rate: float | None = None,
) -> dict[str, dict[str, Any]]:
    return {
        name: simulate_mask(
            market,
            dates,
            active,
            start=start,
            end=end,
            hold_bars=int(cfg.hold_bars if hold_bars is None else hold_bars),
            entry_delay_bars=int(cfg.entry_delay_bars if entry_delay_bars is None else entry_delay_bars),
            leverage=float(cfg.leverage),
            fee_rate=float(cfg.fee_rate),
            slippage_rate=float(cfg.slippage_rate if slippage_rate is None else slippage_rate),
        )
        for name, (start, end) in WINDOWS.items()
    }


def _slim(metric: dict[str, Any]) -> dict[str, Any]:
    return {
        name: metric[name]
        for name in (
            "period",
            "absolute_return_pct",
            "cagr_pct",
            "strict_mdd_pct",
            "cagr_to_strict_mdd",
            "trade_count",
            "trade_statistics",
        )
    }


def _slim_all(metrics: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {name: _slim(metric) for name, metric in metrics.items()}


def _moving_block_bootstrap(
    returns: list[float],
    *,
    samples: int,
    block_trades: int = 4,
    seed: int = 20260715,
) -> dict[str, Any]:
    values = np.asarray(returns, dtype=float)
    if not len(values) or samples <= 0:
        return {}
    rng = np.random.default_rng(seed)
    starts = np.arange(max(1, len(values) - block_trades + 1))
    boot = np.empty(samples, dtype=float)
    for index in range(samples):
        chunks: list[np.ndarray] = []
        size = 0
        while size < len(values):
            start = int(rng.choice(starts))
            chunk = values[start : min(len(values), start + block_trades)]
            chunks.append(chunk)
            size += len(chunk)
        boot[index] = np.concatenate(chunks)[: len(values)].mean()
    return {
        "block_trades": int(block_trades),
        "samples": int(samples),
        "mean_pct": float(values.mean() * 100.0),
        "ci95_mean_pct": [float(np.quantile(boot, 0.025) * 100.0), float(np.quantile(boot, 0.975) * 100.0)],
        "prob_mean_le_zero": float(np.mean(boot <= 0.0)),
    }


def run(cfg: PullbackSqueezeConfig) -> dict[str, Any]:
    loader_cfg = OnlineExpertConfig(
        input_csv=cfg.input_csv,
        funding_csv=cfg.funding_csv,
        premium_csv=cfg.premium_csv,
        output=cfg.output,
        manifest_output="",
        docs_output="",
        exclude_from=cfg.exclude_from,
        window_size=cfg.window_size,
        entry_delay_bars=cfg.entry_delay_bars,
        leverage=cfg.leverage,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
    )
    market, features, source_prefix_hashes = _load_bundle(loader_cfg, cutoff=cfg.exclude_from)
    dates = pd.to_datetime(market["date"])
    decision = np.zeros(len(market), dtype=bool)
    decision[np.arange(max(143, cfg.window_size - 1), len(market), cfg.stride_bars)] = True
    fitted = fit_rule_masks(
        features,
        dates,
        decision,
        range_quantile=cfg.range_quantile,
        overheat_quantile=cfg.overheat_quantile,
    )
    active = fitted["active"]
    metrics = _metrics(market, dates, active, cfg)

    stress = {
        "cost_8bp_side": _slim_all(_metrics(market, dates, active, cfg, slippage_rate=0.0003)),
        "cost_10bp_side": _slim_all(_metrics(market, dates, active, cfg, slippage_rate=0.0005)),
        "entry_lag_2_bars": _slim_all(_metrics(market, dates, active, cfg, entry_delay_bars=2)),
        "entry_lag_3_bars": _slim_all(_metrics(market, dates, active, cfg, entry_delay_bars=3)),
        "hold_432": _slim_all(_metrics(market, dates, active, cfg, hold_bars=432)),
        "hold_720": _slim_all(_metrics(market, dates, active, cfg, hold_bars=720)),
    }
    sensitivity = []
    for range_quantile in (0.55, 0.60, 0.65):
        for overheat_quantile in (0.65, 0.70, 0.75):
            variant = fit_rule_masks(
                features,
                dates,
                decision,
                range_quantile=range_quantile,
                overheat_quantile=overheat_quantile,
            )
            sensitivity.append(
                {
                    "range_quantile": range_quantile,
                    "overheat_quantile": overheat_quantile,
                    "context_thresholds": variant["context_thresholds"],
                    "metrics": _slim_all(_metrics(market, dates, variant["active"], cfg)),
                }
            )

    source_ablation = {
        "funding_only": _slim_all(_metrics(market, dates, fitted["funding_active"], cfg)),
        "premium_only": _slim_all(_metrics(market, dates, fitted["premium_active"], cfg)),
    }
    bootstrap = {
        name: _moving_block_bootstrap(
            metrics[name]["trade_returns"],
            samples=cfg.bootstrap_samples,
        )
        for name in ("train", "select2023", "test2024", "eval2025_2026", "oos2024_2026")
    }
    quarterly: dict[str, dict[str, Any]] = {}
    for year in (2024, 2025, 2026):
        for quarter in range(1, 5):
            month = 1 + 3 * (quarter - 1)
            start = pd.Timestamp(year=year, month=month, day=1)
            end = pd.Timestamp(year=year + 1, month=1, day=1) if quarter == 4 else pd.Timestamp(year=year, month=month + 3, day=1)
            end = min(end, FINAL_END)
            if start < end:
                quarterly[f"{year}Q{quarter}"] = _slim(
                    simulate_mask(
                        market,
                        dates,
                        active,
                        start=start,
                        end=end,
                        hold_bars=cfg.hold_bars,
                        entry_delay_bars=cfg.entry_delay_bars,
                        leverage=cfg.leverage,
                        fee_rate=cfg.fee_rate,
                        slippage_rate=cfg.slippage_rate,
                    )
                )

    result = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "protocol": {
            "threshold_fit": "2020-07-01 through 2022-12-31 only",
            "selection": "2023 only",
            "frozen_family": "Top-20 before 2024 replay",
            "candidate_family_rank": 6,
            "entry": "completed t signal, t+1 open",
            "hold_bars": cfg.hold_bars,
            "leverage": cfg.leverage,
            "cost_per_side": cfg.fee_rate + cfg.slippage_rate,
            "strict_mdd": "pre-entry high water plus favorable-then-adverse intrabar marking",
            "period_exit_purge": True,
            "warning": "29,133 pre-2024 interactions and broader-programme OOS exposure; retrospective shadow candidate, not fresh live proof",
        },
        "source_prefix_hashes": source_prefix_hashes,
        "rule": {
            "base_thresholds": fitted["base_thresholds"],
            "context_thresholds": fitted["context_thresholds"],
            "logic": "((funding_q10 & trend96_q70 & completed_1d_return<=q70) | (premium_change_q20 & completed_4d_momentum_q90 & completed_3d_return<=q70)) & rex_48h_range_pos<=q60",
            "availability_required": True,
        },
        "metrics": _slim_all(metrics),
        "stress": stress,
        "threshold_sensitivity": sensitivity,
        "source_ablation": source_ablation,
        "quarterly": quarterly,
        "moving_block_bootstrap": bootstrap,
        "multiplicity": {
            "searched_pre2024": 29_133,
            "frozen_family": 20,
            "interpretation": "descriptive p-values do not erase selection multiplicity; require fresh shadow/live-forward evidence",
        },
    }
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", default=PullbackSqueezeConfig.input_csv)
    parser.add_argument("--funding-csv", default=PullbackSqueezeConfig.funding_csv)
    parser.add_argument("--premium-csv", default=PullbackSqueezeConfig.premium_csv)
    parser.add_argument("--output", default=PullbackSqueezeConfig.output)
    parser.add_argument("--exclude-from", default=PullbackSqueezeConfig.exclude_from)
    parser.add_argument("--window-size", type=int, default=PullbackSqueezeConfig.window_size)
    parser.add_argument("--hold-bars", type=int, default=PullbackSqueezeConfig.hold_bars)
    parser.add_argument("--stride-bars", type=int, default=PullbackSqueezeConfig.stride_bars)
    parser.add_argument("--entry-delay-bars", type=int, default=PullbackSqueezeConfig.entry_delay_bars)
    parser.add_argument("--leverage", type=float, default=PullbackSqueezeConfig.leverage)
    parser.add_argument("--fee-rate", type=float, default=PullbackSqueezeConfig.fee_rate)
    parser.add_argument("--slippage-rate", type=float, default=PullbackSqueezeConfig.slippage_rate)
    parser.add_argument("--range-quantile", type=float, default=PullbackSqueezeConfig.range_quantile)
    parser.add_argument("--overheat-quantile", type=float, default=PullbackSqueezeConfig.overheat_quantile)
    parser.add_argument("--bootstrap-samples", type=int, default=PullbackSqueezeConfig.bootstrap_samples)
    return parser.parse_args()


def main() -> None:
    result = run(PullbackSqueezeConfig(**vars(parse_args())))
    summary = {
        name: {
            key: result["metrics"][name][key]
            for key in ("absolute_return_pct", "cagr_pct", "strict_mdd_pct", "cagr_to_strict_mdd", "trade_count")
        }
        for name in ("train", "select2023", "test2024", "eval2025_2026", "oos2024_2026", "full")
    }
    print(json.dumps({"output": result["config"]["output"], "rule": result["rule"], "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
