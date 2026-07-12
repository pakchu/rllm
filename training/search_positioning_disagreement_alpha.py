"""Search BTC alpha from delayed Binance futures positioning disagreement.

This experiment uses the public 5-minute Binance USD-M metrics archive.  Every
metrics row is delayed by one complete source bar before it can influence a
signal.  Thresholds are fitted on 2020Q4-2021, 2022 is quarantined because the
official archive has large top-trader-field gaps, and a Top-10 manifest is
selected on 2023 before 2024/2025/2026 statistics are computed.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


WINDOWS = {
    "fit": ("2020-10-15", "2022-01-01"),
    "select2023": ("2023-01-01", "2024-01-01"),
    "select2023_h1": ("2023-01-01", "2023-07-01"),
    "select2023_h2": ("2023-07-01", "2024-01-01"),
    "test2024": ("2024-01-01", "2025-01-01"),
    "eval2025": ("2025-01-01", "2026-01-01"),
    "ytd2026": ("2026-01-01", "2026-06-02"),
}


@dataclass(frozen=True)
class PositioningSearchConfig:
    input_csv: str
    metrics_csv: str
    output: str
    top_n: int = 10
    leverage: float = 0.5
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0001
    min_select_trades: int = 24
    min_half_trades: int = 8
    metrics_tolerance: str = "10min"
    source_delay_bars: int = 1


def _rolling_z(series: pd.Series, window: int) -> pd.Series:
    minimum = max(24, window // 2)
    mean = series.rolling(window, min_periods=minimum).mean()
    std = series.rolling(window, min_periods=minimum).std(ddof=0).replace(0.0, np.nan)
    return (series - mean) / std


def _attach_delayed_metrics(
    market: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    tolerance: str,
    delay_bars: int,
) -> pd.DataFrame:
    left = market.copy()
    left["date"] = pd.to_datetime(left["date"], utc=True, errors="raise").dt.tz_convert(None)
    right = metrics.copy()
    right["create_time"] = pd.to_datetime(right["create_time"], utc=True, errors="raise").dt.tz_convert(None)
    value_columns = [column for column in right.columns if column not in {"create_time", "symbol"}]
    joined = pd.merge_asof(
        left.sort_values("date"),
        right[["create_time", *value_columns]].sort_values("create_time"),
        left_on="date",
        right_on="create_time",
        direction="backward",
        tolerance=pd.Timedelta(tolerance),
    )
    joined[value_columns] = joined[value_columns].shift(max(1, int(delay_bars)))
    joined["positioning_available"] = joined[value_columns].notna().all(axis=1).astype(float)
    return joined.drop(columns=["create_time"])


def build_positioning_features(market: pd.DataFrame) -> pd.DataFrame:
    raw = pd.DataFrame(index=market.index)
    source_map = {
        "top_acct": "count_toptrader_long_short_ratio",
        "top_pos": "sum_toptrader_long_short_ratio",
        "global_acct": "count_long_short_ratio",
        "taker": "sum_taker_long_short_vol_ratio",
    }
    for name, source in source_map.items():
        values = pd.to_numeric(market[source], errors="coerce")
        raw[name] = np.log(values.where(values > 0.0))
    raw["smart_size"] = raw["top_pos"] - raw["top_acct"]
    raw["smart_retail"] = raw["top_pos"] - raw["global_acct"]
    raw["topacct_retail"] = raw["top_acct"] - raw["global_acct"]
    raw["position_flow"] = raw["top_pos"] - raw["taker"]

    columns: dict[str, pd.Series] = {}
    for name in raw.columns:
        for window in (48, 144, 288, 2016, 8640):
            columns[f"{name}_z{window}"] = _rolling_z(raw[name], window)
            columns[f"{name}_chg{window}"] = raw[name] - raw[name].shift(window)
    for window in (48, 144, 288):
        columns[f"crowding_{window}"] = (
            _rolling_z(raw["top_pos"], window)
            + _rolling_z(raw["global_acct"], window)
            + _rolling_z(raw["taker"], window)
        )
        columns[f"smart_absorb_{window}"] = _rolling_z(raw["smart_size"], window) - _rolling_z(raw["taker"], window)
    return pd.DataFrame(columns, index=market.index).replace([np.inf, -np.inf], np.nan)


def _window_mask(dates: pd.Series, name: str) -> np.ndarray:
    start, end = WINDOWS[name]
    return ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy(bool)


def _fit_quantile(features: pd.DataFrame, fit_mask: np.ndarray, name: str, quantile: float) -> float:
    values = features[name].to_numpy(float)
    reference = values[fit_mask & np.isfinite(values)]
    if len(reference) < 10_000:
        raise ValueError(f"insufficient fit observations for {name}: {len(reference)}")
    return float(np.quantile(reference, quantile))


def _signal_mask(values: np.ndarray, op: str, threshold: float) -> np.ndarray:
    finite = np.isfinite(values)
    return finite & ((values >= threshold) if op == "ge" else (values <= threshold))


def _future_extreme(values: np.ndarray, hold_bars: int, reducer: str) -> np.ndarray:
    reversed_values = pd.Series(values[::-1])
    rolling = getattr(reversed_values.rolling(hold_bars, min_periods=hold_bars), reducer)()
    return rolling.to_numpy()[::-1]


def _simulate_no_stop(
    market: pd.DataFrame,
    dates: pd.Series,
    long_active: np.ndarray,
    short_active: np.ndarray,
    *,
    window: str,
    hold_bars: int,
    stride_bars: int,
    leverage: float,
    fee_rate: float,
    slippage_rate: float,
    extremes: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    period = _window_mask(dates, window)
    open_price = market["open"].to_numpy(float)
    high = market["high"].to_numpy(float)
    low = market["low"].to_numpy(float)
    if extremes is None:
        future_low = _future_extreme(low, hold_bars, "min")
        future_high = _future_extreme(high, hold_bars, "max")
    else:
        future_low, future_high = extremes
    candidates = np.arange(0, len(market) - hold_bars - 2, stride_bars, dtype=np.int64)
    candidates = candidates[period[candidates] & (long_active[candidates] | short_active[candidates])]
    start, end = WINDOWS[window]
    cost = (fee_rate + slippage_rate) * leverage
    equity = peak = 1.0
    strict_mdd = 0.0
    next_position = 0
    trade_returns: list[float] = []
    sides: list[int] = []
    for position in candidates:
        if position < next_position:
            continue
        side = 1 if long_active[position] and not short_active[position] else (-1 if short_active[position] and not long_active[position] else 0)
        if side == 0:
            continue
        entry_position = position + 1
        exit_position = entry_position + hold_bars
        if exit_position >= len(market) or not period[exit_position]:
            continue
        entry_price = open_price[entry_position]
        entry_equity = equity
        equity *= 1.0 - cost
        strict_mdd = max(strict_mdd, 1.0 - equity / peak)
        adverse = (
            future_low[entry_position] / entry_price - 1.0
            if side > 0
            else 1.0 - future_high[entry_position] / entry_price
        )
        strict_mdd = max(strict_mdd, 1.0 - max(0.0, equity * (1.0 + leverage * adverse)) / peak)
        raw_return = side * (open_price[exit_position] / entry_price - 1.0)
        equity *= max(0.0, 1.0 + leverage * raw_return)
        equity *= 1.0 - cost
        strict_mdd = max(strict_mdd, 1.0 - equity / peak)
        peak = max(peak, equity)
        trade_returns.append(equity / entry_equity - 1.0)
        sides.append(side)
        next_position = exit_position + 1
    years = (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() / (365.25 * 86400.0)
    absolute_return = (equity - 1.0) * 100.0
    cagr = (equity ** (1.0 / years) - 1.0) * 100.0 if equity > 0.0 else -100.0
    mdd = strict_mdd * 100.0
    returns = np.asarray(trade_returns, dtype=float)
    return {
        "return_pct": float(absolute_return),
        "cagr_pct": float(cagr),
        "strict_mdd_pct": float(mdd),
        "ratio": float(cagr / mdd) if mdd > 1e-12 else 0.0,
        "trades": int(len(returns)),
        "longs": int(sum(side > 0 for side in sides)),
        "shorts": int(sum(side < 0 for side in sides)),
        "win_rate": float((returns > 0.0).mean()) if len(returns) else 0.0,
    }


def _candidate_specs(features: pd.DataFrame, fit_mask: np.ndarray) -> list[dict[str, Any]]:
    positive = (
        "smart_size_z144",
        "smart_size_z2016",
        "smart_size_z8640",
        "smart_retail_z144",
        "smart_retail_z2016",
        "smart_retail_z8640",
        "topacct_retail_z48",
        "topacct_retail_z144",
        "topacct_retail_z2016",
        "smart_retail_chg2016",
        "smart_size_chg2016",
        "smart_absorb_144",
    )
    negative = (
        "top_acct_z144",
        "top_acct_z2016",
        "top_acct_z8640",
        "global_acct_z144",
        "global_acct_z2016",
        "global_acct_z8640",
        "top_acct_chg2016",
        "global_acct_chg2016",
        "crowding_144",
    )
    specs: list[dict[str, Any]] = []
    for feature, direction in itertools.chain(((name, 1) for name in positive), ((name, -1) for name in negative)):
        for tail in (0.05, 0.10, 0.15, 0.20, 0.25):
            lower = _fit_quantile(features, fit_mask, feature, tail)
            upper = _fit_quantile(features, fit_mask, feature, 1.0 - tail)
            long_op, long_threshold = ("ge", upper) if direction > 0 else ("le", lower)
            short_op, short_threshold = ("le", lower) if direction > 0 else ("ge", upper)
            specs.append(
                {
                    "name": f"positioning_{feature}_tail{tail:.2f}",
                    "feature": feature,
                    "direction": direction,
                    "tail_quantile": tail,
                    "long": {"op": long_op, "threshold": long_threshold},
                    "short": {"op": short_op, "threshold": short_threshold},
                }
            )
    return specs


def run(cfg: PositioningSearchConfig) -> dict[str, Any]:
    market = pd.read_csv(cfg.input_csv, compression="infer")
    metrics = pd.read_csv(cfg.metrics_csv, compression="infer")
    market = _attach_delayed_metrics(
        market,
        metrics,
        tolerance=cfg.metrics_tolerance,
        delay_bars=cfg.source_delay_bars,
    )
    dates = pd.to_datetime(market["date"])
    features = build_positioning_features(market)
    fit_mask = _window_mask(dates, "fit")
    extremes_by_hold = {
        hold: (
            _future_extreme(market["low"].to_numpy(float), hold, "min"),
            _future_extreme(market["high"].to_numpy(float), hold, "max"),
        )
        for hold in (48, 96, 144, 216, 288)
    }
    rows: list[dict[str, Any]] = []
    for spec in _candidate_specs(features, fit_mask):
        values = features[spec["feature"]].to_numpy(float)
        long_active = _signal_mask(values, spec["long"]["op"], spec["long"]["threshold"])
        short_active = _signal_mask(values, spec["short"]["op"], spec["short"]["threshold"])
        for hold_bars, stride_bars in itertools.product((48, 96, 144, 216, 288), (6, 12, 24)):
            selection = _simulate_no_stop(
                market,
                dates,
                long_active,
                short_active,
                window="select2023",
                hold_bars=hold_bars,
                stride_bars=stride_bars,
                leverage=cfg.leverage,
                fee_rate=cfg.fee_rate,
                slippage_rate=cfg.slippage_rate,
                extremes=extremes_by_hold[hold_bars],
            )
            if selection["trades"] < cfg.min_select_trades or min(selection["longs"], selection["shorts"]) < 4:
                continue
            halves = [
                _simulate_no_stop(
                    market,
                    dates,
                    long_active,
                    short_active,
                    window=name,
                    hold_bars=hold_bars,
                    stride_bars=stride_bars,
                    leverage=cfg.leverage,
                    fee_rate=cfg.fee_rate,
                    slippage_rate=cfg.slippage_rate,
                    extremes=extremes_by_hold[hold_bars],
                )
                for name in ("select2023_h1", "select2023_h2")
            ]
            if min(half["trades"] for half in halves) < cfg.min_half_trades:
                continue
            rows.append(
                {
                    **spec,
                    "hold_bars": hold_bars,
                    "stride_bars": stride_bars,
                    "select2023": selection,
                    "select2023_halves": halves,
                    "selection_score": {
                        "positive_halves": sum(half["return_pct"] > 0.0 for half in halves),
                        "min_half_ratio": min(half["ratio"] for half in halves),
                        "full_ratio": selection["ratio"],
                    },
                }
            )
    rows.sort(
        key=lambda row: (
            row["selection_score"]["positive_halves"],
            row["selection_score"]["min_half_ratio"],
            row["selection_score"]["full_ratio"],
            row["select2023"]["return_pct"],
        ),
        reverse=True,
    )
    selected = rows[: cfg.top_n]
    manifest_payload = [
        {
            key: row[key]
            for key in ("name", "feature", "direction", "tail_quantile", "long", "short", "hold_bars", "stride_bars")
        }
        for row in selected
    ]
    manifest_json = json.dumps(manifest_payload, sort_keys=True, separators=(",", ":"))
    manifest_hash = hashlib.sha256(manifest_json.encode("utf-8")).hexdigest()
    for row in selected:
        values = features[row["feature"]].to_numpy(float)
        long_active = _signal_mask(values, row["long"]["op"], row["long"]["threshold"])
        short_active = _signal_mask(values, row["short"]["op"], row["short"]["threshold"])
        for name in ("fit", "test2024", "eval2025", "ytd2026"):
            row[name] = _simulate_no_stop(
                market,
                dates,
                long_active,
                short_active,
                window=name,
                hold_bars=row["hold_bars"],
                stride_bars=row["stride_bars"],
                leverage=cfg.leverage,
                fee_rate=cfg.fee_rate,
                slippage_rate=cfg.slippage_rate,
                extremes=extremes_by_hold[row["hold_bars"]],
            )
        row["passes_alpha_target"] = (
            row["test2024"]["ratio"] >= 3.0
            and row["eval2025"]["ratio"] >= 3.0
            and row["test2024"]["trades"] >= 16
            and row["eval2025"]["trades"] >= 16
        )
        row["passes_live_target"] = (
            row["passes_alpha_target"]
            and row["ytd2026"]["ratio"] >= 5.0
            and row["ytd2026"]["trades"] >= 8
            and row["ytd2026"]["return_pct"] > 0.0
        )
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "protocol": {
            "threshold_fit": WINDOWS["fit"],
            "archive_gap_quarantine": "2022 excluded because top-trader fields are mostly missing",
            "selection": "2023 full-year plus H1/H2 robustness; Top-10 frozen before future statistics",
            "future_windows": {name: WINDOWS[name] for name in ("test2024", "eval2025", "ytd2026")},
            "source_delay_bars": cfg.source_delay_bars,
            "execution": "signal at completed bar; entry next bar open; fixed hold; 6bp/side default; strict intraposition MDD",
        },
        "input": {
            "rows": int(len(market)),
            "start": str(dates.min()),
            "end": str(dates.max()),
            "positioning_available_fraction": float(market["positioning_available"].mean()),
        },
        "tested": int(len(rows)),
        "manifest_hash": manifest_hash,
        "manifest": manifest_payload,
        "selected": selected,
        "alpha_qualifiers": [row for row in selected if row["passes_alpha_target"]],
        "live_qualifiers": [row for row in selected if row["passes_live_target"]],
    }
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> PositioningSearchConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--metrics-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-n", type=int, default=PositioningSearchConfig.top_n)
    parser.add_argument("--leverage", type=float, default=PositioningSearchConfig.leverage)
    parser.add_argument("--fee-rate", type=float, default=PositioningSearchConfig.fee_rate)
    parser.add_argument("--slippage-rate", type=float, default=PositioningSearchConfig.slippage_rate)
    parser.add_argument("--min-select-trades", type=int, default=PositioningSearchConfig.min_select_trades)
    parser.add_argument("--min-half-trades", type=int, default=PositioningSearchConfig.min_half_trades)
    parser.add_argument("--metrics-tolerance", default=PositioningSearchConfig.metrics_tolerance)
    parser.add_argument("--source-delay-bars", type=int, default=PositioningSearchConfig.source_delay_bars)
    return PositioningSearchConfig(**vars(parser.parse_args()))


def main() -> None:
    report = run(parse_args())
    print(
        json.dumps(
            {
                "tested": report["tested"],
                "manifest_hash": report["manifest_hash"],
                "alpha_qualifiers": len(report["alpha_qualifiers"]),
                "live_qualifiers": len(report["live_qualifiers"]),
                "selected": report["selected"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
