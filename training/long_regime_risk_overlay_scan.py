"""Risk-overlay scan for existing-data long-regime REX rules.

The previous composite search found long entry edge but poor drawdown control.
This scanner keeps the interpretable long entry conditions fixed and searches
only execution overlays: stop loss, take profit, trailing stop and max hold.
Thresholds are fitted on train only; ranking uses train+validation only; 2025
eval remains report-only.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from preprocessing.market_features import build_market_feature_frame
from training.long_regime_combo_scan import LongComboScanConfig, _load_market, _split_mask
from training.strict_bar_backtest import _trade_stats


@dataclass(frozen=True)
class LongRiskOverlayConfig(LongComboScanConfig):
    stop_losses: str = "0,1.5,2.5,3.5,5.0"
    take_profits: str = "0,2.5,4.0,6.0,9.0"
    trail_stops: str = "0,2.0,3.5,5.0"
    hold_bars: str = "72,144,216,288"
    stride_bars: str = "6,12,24"
    quantiles: str = "0.70,0.75,0.80,0.85"
    rules: str = "pb30,pb30_funding,pb12_trend,w30_trend,range_upper"


def _parse_list(raw: str, cast: Any) -> list[Any]:
    return [cast(x.strip()) for x in str(raw).split(",") if x.strip()]


def _arr(features: pd.DataFrame, name: str) -> np.ndarray:
    return features.get(name, pd.Series(0.0, index=features.index)).to_numpy(dtype=float)


def _rule_components(features: pd.DataFrame) -> dict[str, list[tuple[str, np.ndarray, str]]]:
    """Candidate entry rules as train-quantile conditions.

    Tuple form is (label, values, operator).  Operators are only threshold
    direction; thresholds are fitted from train for each quantile.
    """
    return {
        "pb30": [("pb30", _arr(features, "rex_8640_max_to_cur_pct"), "ge")],
        "pb30_funding": [
            ("pb30", _arr(features, "rex_8640_max_to_cur_pct"), "ge"),
            ("funding_low", _arr(features, "funding_zscore"), "le"),
        ],
        "pb12_trend": [
            ("pb12", _arr(features, "rex_144_max_to_cur_pct"), "ge"),
            ("trend1d", _arr(features, "htf_1d_return_4"), "ge"),
        ],
        "w30_trend": [
            ("w30", _arr(features, "rex_8640_range_width_pct"), "ge"),
            ("trend1d", _arr(features, "htf_1d_return_4"), "ge"),
        ],
        "range_upper": [
            ("pos30", _arr(features, "rex_8640_range_pos"), "ge"),
            ("w30", _arr(features, "rex_8640_range_width_pct"), "ge"),
        ],
    }


def _fit_active(
    components: list[tuple[str, np.ndarray, str]],
    *,
    train_mask: np.ndarray,
    quantile: float,
) -> tuple[np.ndarray, list[dict[str, Any]]] | None:
    active = np.ones(len(train_mask), dtype=bool)
    spec: list[dict[str, Any]] = []
    for name, values, op in components:
        train_values = values[train_mask & np.isfinite(values)]
        if train_values.size < 200:
            return None
        threshold = float(np.quantile(train_values, float(quantile)))
        cond = values >= threshold if op == "ge" else values <= threshold
        active &= cond & np.isfinite(values)
        spec.append({"name": name, "op": op, "quantile": float(quantile), "threshold": threshold})
    return active, spec


def _overlay_long_sim(
    signal_positions: np.ndarray,
    *,
    market: pd.DataFrame,
    hold_bars: int,
    entry_delay_bars: int,
    leverage: float,
    fee_rate: float,
    slippage_rate: float,
    stop_loss_pct: float,
    take_profit_pct: float,
    trail_stop_pct: float,
) -> tuple[dict[str, Any], list[float]]:
    opens = market["open"].to_numpy(float)
    highs = market["high"].to_numpy(float)
    lows = market["low"].to_numpy(float)
    dates = pd.to_datetime(market["date"]).to_numpy()
    cost = (float(fee_rate) + float(slippage_rate)) * float(leverage)
    stop_loss = float(stop_loss_pct) / 100.0
    take_profit = float(take_profit_pct) / 100.0
    trail_stop = float(trail_stop_pct) / 100.0

    eq = peak_eq = 1.0
    max_dd = 0.0
    next_allowed = 0
    trade_returns: list[float] = []
    exit_reasons: dict[str, int] = {"time": 0, "stop": 0, "take": 0, "trail": 0}
    first_signal: int | None = None
    last_signal: int | None = None
    for pos in np.asarray(signal_positions, dtype=np.int64):
        if pos < next_allowed:
            continue
        entry = int(pos) + int(entry_delay_bars)
        max_exit = entry + int(hold_bars)
        if entry >= len(market) - 1 or max_exit >= len(market):
            continue
        entry_open = float(opens[entry])
        if entry_open <= 0.0:
            continue
        first_signal = int(pos) if first_signal is None else first_signal
        last_signal = int(pos)
        entry_eq = eq
        eq *= max(0.0, 1.0 - cost)
        max_dd = max(max_dd, 1.0 - max(0.0, eq) / peak_eq if peak_eq > 0.0 else 0.0)
        trade_peak_price = entry_open
        reason = "time"
        exit_pos = max_exit

        for j in range(entry, max_exit):
            open_j = float(opens[j])
            if open_j <= 0.0:
                continue
            trade_peak_price = max(trade_peak_price, float(highs[j]))
            barriers: list[tuple[str, float]] = []
            if stop_loss > 0.0:
                barriers.append(("stop", entry_open * (1.0 - stop_loss)))
            if trail_stop > 0.0:
                barriers.append(("trail", trade_peak_price * (1.0 - trail_stop)))
            stop_barrier = max((price for name, price in barriers), default=None)
            take_barrier = entry_open * (1.0 + take_profit) if take_profit > 0.0 else None

            # Conservative same-bar ordering for long: if low and high both hit,
            # count the downside barrier first.
            if stop_barrier is not None and float(lows[j]) <= stop_barrier:
                realized_ret = (stop_barrier - open_j) / open_j
                max_dd = max(max_dd, 1.0 - max(0.0, eq * (1.0 + float(leverage) * realized_ret)) / peak_eq)
                eq *= max(0.0, 1.0 + float(leverage) * realized_ret)
                reason = "trail" if trail_stop > 0.0 and stop_barrier > entry_open * (1.0 - stop_loss if stop_loss > 0 else 0.0) else "stop"
                exit_pos = j
                break
            if take_barrier is not None and float(highs[j]) >= take_barrier:
                adverse_ret = (float(lows[j]) - open_j) / open_j
                max_dd = max(max_dd, 1.0 - max(0.0, eq * (1.0 + float(leverage) * adverse_ret)) / peak_eq)
                realized_ret = (take_barrier - open_j) / open_j
                eq *= max(0.0, 1.0 + float(leverage) * realized_ret)
                reason = "take"
                exit_pos = j
                break

            adverse_ret = (float(lows[j]) - open_j) / open_j
            max_dd = max(max_dd, 1.0 - max(0.0, eq * (1.0 + float(leverage) * adverse_ret)) / peak_eq)
            close_ret = (float(opens[j + 1]) - open_j) / open_j
            eq *= max(0.0, 1.0 + float(leverage) * close_ret)
            peak_eq = max(peak_eq, eq)
            if eq <= 0.0:
                exit_pos = j
                break

        eq *= max(0.0, 1.0 - cost)
        max_dd = max(max_dd, 1.0 - max(0.0, eq) / peak_eq if peak_eq > 0.0 else 0.0)
        peak_eq = max(peak_eq, eq)
        trade_returns.append(eq / entry_eq - 1.0)
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        next_allowed = exit_pos + 1
        if eq <= 0.0:
            break

    if first_signal is None or last_signal is None:
        start_dt = end_dt = datetime.now()
        years = 1.0 / 365.25
    else:
        start_dt = pd.Timestamp(dates[first_signal]).to_pydatetime()
        end_dt = pd.Timestamp(dates[last_signal]).to_pydatetime()
        years = max(1.0 / 365.25, float((end_dt - start_dt).days) / 365.25)
    ret_pct = (eq - 1.0) * 100.0
    cagr_pct = ((eq ** (1.0 / years) - 1.0) * 100.0) if eq > 0.0 else -100.0
    mdd_pct = max_dd * 100.0
    return (
        {
            "period": {"start": str(start_dt), "end": str(end_dt), "years": years},
            "cagr_pct": cagr_pct,
            "strict_mdd_pct": mdd_pct,
            "cagr_to_strict_mdd": cagr_pct / mdd_pct if mdd_pct > 1e-12 else float("inf"),
            "trade_entries": len(trade_returns),
            "win_rate": sum(1 for r in trade_returns if r > 0.0) / len(trade_returns) if trade_returns else 0.0,
            "total_return_pct": ret_pct,
            "exit_reasons": exit_reasons,
            "return_application": "long_only_ohlc_stop_take_trail_strict_mdd",
        },
        trade_returns,
    )


def _score(row: dict[str, Any], cfg: LongRiskOverlayConfig) -> float:
    train = row["train"]["sim"]
    val = row["val"]["sim"]
    if int(train["trade_entries"]) < int(cfg.min_train_trades) or int(val["trade_entries"]) < int(cfg.min_val_trades):
        return -1e9
    if float(train["cagr_pct"]) <= 0.0 or float(val["cagr_pct"]) <= 0.0:
        return -1e9
    if float(train["strict_mdd_pct"]) > 30.0 or float(val["strict_mdd_pct"]) > 18.0:
        return -1e9
    train_ratio = float(train["cagr_to_strict_mdd"])
    val_ratio = float(val["cagr_to_strict_mdd"])
    val_p = float(row["val"]["trade_stats"].get("p_value_mean_ret_approx", 1.0))
    return val_ratio + 0.6 * train_ratio + min(1.0, float(val["trade_entries"]) / 150.0) - 0.3 * abs(val_ratio - train_ratio) - 0.2 * val_p


def _compact(row: dict[str, Any]) -> dict[str, Any]:
    out = {k: row[k] for k in ("rule", "spec", "hold_bars", "hold_hours", "stride_bars", "stop_loss_pct", "take_profit_pct", "trail_stop_pct", "selection_score")}
    for split in ("train", "val", "eval"):
        sim = row[split]["sim"]
        out[split] = {
            "cagr_pct": sim["cagr_pct"],
            "strict_mdd_pct": sim["strict_mdd_pct"],
            "cagr_to_strict_mdd": sim["cagr_to_strict_mdd"],
            "trade_entries": sim["trade_entries"],
            "win_rate": sim["win_rate"],
            "exit_reasons": sim["exit_reasons"],
            "p_value": row[split]["trade_stats"].get("p_value_mean_ret_approx"),
        }
    return out


def run_scan(cfg: LongRiskOverlayConfig) -> dict[str, Any]:
    market = _load_market(cfg)
    features = build_market_feature_frame(market, window_size=int(cfg.window_size))
    components_by_rule = _rule_components(features)
    enabled_rules = set(_parse_list(cfg.rules, str))
    dates = pd.to_datetime(market["date"])
    split_masks = {
        "train": _split_mask(dates, cfg.train_start, cfg.train_end),
        "val": _split_mask(dates, cfg.val_start, cfg.val_end),
        "eval": _split_mask(dates, cfg.eval_start, cfg.eval_end),
    }
    max_hold = max(_parse_list(cfg.hold_bars, int))
    positions_by_stride = {
        s: np.arange(max(0, int(cfg.window_size) - 1), max(0, len(market) - max_hold - int(cfg.entry_delay_bars) - 1), s, dtype=np.int64)
        for s in _parse_list(cfg.stride_bars, int)
    }

    rows: list[dict[str, Any]] = []
    for rule, comps in components_by_rule.items():
        if rule not in enabled_rules:
            continue
        for q in _parse_list(cfg.quantiles, float):
            fit = _fit_active(comps, train_mask=split_masks["train"], quantile=q)
            if fit is None:
                continue
            active, spec = fit
            for hold in _parse_list(cfg.hold_bars, int):
                for stride, base_positions in positions_by_stride.items():
                    for stop in _parse_list(cfg.stop_losses, float):
                        for take in _parse_list(cfg.take_profits, float):
                            for trail in _parse_list(cfg.trail_stops, float):
                                if stop == 0.0 and take == 0.0 and trail == 0.0:
                                    continue
                                row: dict[str, Any] = {
                                    "rule": rule,
                                    "spec": spec,
                                    "hold_bars": int(hold),
                                    "hold_hours": float(hold) * 5.0 / 60.0,
                                    "stride_bars": int(stride),
                                    "stop_loss_pct": float(stop),
                                    "take_profit_pct": float(take),
                                    "trail_stop_pct": float(trail),
                                }
                                for split, mask in split_masks.items():
                                    positions = base_positions[active[base_positions] & mask[base_positions]]
                                    sim, returns = _overlay_long_sim(
                                        positions,
                                        market=market,
                                        hold_bars=int(hold),
                                        entry_delay_bars=int(cfg.entry_delay_bars),
                                        leverage=float(cfg.leverage),
                                        fee_rate=float(cfg.fee_rate),
                                        slippage_rate=float(cfg.slippage_rate),
                                        stop_loss_pct=float(stop),
                                        take_profit_pct=float(take),
                                        trail_stop_pct=float(trail),
                                    )
                                    row[split] = {"sim": sim, "trade_stats": _trade_stats(returns)}
                                row["selection_score"] = _score(row, cfg)
                                rows.append(row)

    ranked = sorted(rows, key=lambda r: float(r.get("selection_score", -1e9)), reverse=True)
    robust = sorted(
        [
            r
            for r in rows
            if int(r["train"]["sim"]["trade_entries"]) >= int(cfg.min_train_trades)
            and int(r["val"]["sim"]["trade_entries"]) >= int(cfg.min_val_trades)
            and int(r["eval"]["sim"]["trade_entries"]) >= int(cfg.min_eval_trades_for_robust)
            and all(float(r[s]["sim"]["cagr_pct"]) > 0.0 for s in ("train", "val", "eval"))
        ],
        key=lambda r: min(float(r[s]["sim"]["cagr_to_strict_mdd"]) for s in ("train", "val", "eval")),
        reverse=True,
    )
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "input": {"rows": len(market), "start": str(market["date"].iloc[0]), "end": str(market["date"].iloc[-1])},
        "selection_protocol": "entry thresholds fit on train only; overlay ranked on train+val only; eval report-only",
        "top_selection": [_compact(r) for r in ranked[:50]],
        "top_robust": [_compact(r) for r in robust[:50]],
        "all_count": len(rows),
        "leakage_guard": {
            "market_rows_after_exclude_from_removed_before_feature_build": True,
            "entry_thresholds_fit_train_only": True,
            "eval_not_used_for_selection": True,
            "same_bar_stop_before_take_for_conservative_long": True,
        },
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--funding-csv", default="")
    p.add_argument("--premium-csv", default="")
    for field in (
        "train-start",
        "train-end",
        "val-start",
        "val-end",
        "eval-start",
        "eval-end",
        "exclude-from",
        "hold-bars",
        "stride-bars",
        "quantiles",
        "stop-losses",
        "take-profits",
        "trail-stops",
        "rules",
    ):
        p.add_argument(f"--{field}", default=getattr(LongRiskOverlayConfig, field.replace("-", "_")))
    p.add_argument("--window-size", type=int, default=LongRiskOverlayConfig.window_size)
    p.add_argument("--entry-delay-bars", type=int, default=LongRiskOverlayConfig.entry_delay_bars)
    p.add_argument("--leverage", type=float, default=LongRiskOverlayConfig.leverage)
    p.add_argument("--fee-rate", type=float, default=LongRiskOverlayConfig.fee_rate)
    p.add_argument("--slippage-rate", type=float, default=LongRiskOverlayConfig.slippage_rate)
    p.add_argument("--min-train-trades", type=int, default=LongRiskOverlayConfig.min_train_trades)
    p.add_argument("--min-val-trades", type=int, default=LongRiskOverlayConfig.min_val_trades)
    p.add_argument("--min-eval-trades-for-robust", type=int, default=LongRiskOverlayConfig.min_eval_trades_for_robust)
    return p.parse_args()


def main() -> None:
    report = run_scan(LongRiskOverlayConfig(**vars(parse_args())))
    print(
        json.dumps(
            {
                "output": report["config"]["output"],
                "input": report["input"],
                "all_count": report["all_count"],
                "top_selection": report["top_selection"][:10],
                "top_robust": report["top_robust"][:10],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
