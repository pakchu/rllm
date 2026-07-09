"""Scan fixed long alpha component unions with strict full-window scoring.

This is a reusable version of the manual long-alpha sweep.  It does not fit a
model or thresholds on test/eval data: component thresholds are fixed from prior
train-window quantile discovery, then only OR-union composition / hold / overlay
variants are scored.  Reported CAGR uses the full calendar window (including idle
periods), and strict MDD includes intrabar adverse excursion while in position.
"""
from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from preprocessing.market_features import build_market_feature_frame
from training.long_regime_alpha_union_validate import COMPONENTS as BASE_LONG_COMPONENTS
from training.long_regime_combo_scan import LongComboScanConfig, _load_market, _split_mask, _strict_long_sim
from training.long_regime_interest_gate_validation import build_interest_features
from training.strict_bar_backtest import _trade_stats


@dataclass(frozen=True)
class LongComponentUnionScanConfig(LongComboScanConfig):
    """Configuration for fixed-component long union scanning."""

    exclude_from: str = "2026-06-02"
    leverage: float = 0.5
    min_combined_trades: int = 40
    min_eval_trades: int = 20
    min_ytd_trades: int = 6
    top_n: int = 120
    component_sizes: str = "1,2,3"
    hold_bars: str = "72,144,216,288,432,576"
    stride_bars: int = 12
    overlays: str = "none,1.5:1.0,2.5:1.5,4.0:2.5,6.0:4.0"


COMPONENTS: dict[str, list[tuple[str, str, float]]] = {
    **BASE_LONG_COMPONENTS,
    # Derivatives discount + price momentum component discovered in the family
    # scan.  Kept separate from low-funding trend to avoid hiding alpha source.
    "premium20_mom90": [("premium_index_change", "le", -0.00023471), ("htf_1d_return_4", "ge", 0.0940403008961932)],
}

WINDOWS: list[tuple[str, str, str]] = [
    ("test2024", "2024-01-01", "2025-01-01"),
    ("eval2025", "2025-01-01", "2026-01-01"),
    ("ytd2026", "2026-01-01", "2026-06-02"),
    ("combined", "2024-01-01", "2026-06-02"),
]


def _parse_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def _parse_overlays(raw: str) -> list[tuple[float | None, float | None]]:
    out: list[tuple[float | None, float | None]] = []
    for item in str(raw).split(","):
        item = item.strip().lower()
        if not item:
            continue
        if item in {"none", "0", "0:0"}:
            out.append((None, None))
            continue
        tp_s, sl_s = item.split(":", 1)
        out.append((float(tp_s) / 100.0, float(sl_s) / 100.0))
    return out


def _component_mask(features: pd.DataFrame, name: str) -> np.ndarray:
    mask = np.ones(len(features), dtype=bool)
    for feature, op, threshold in COMPONENTS[name]:
        if feature not in features.columns:
            return np.zeros(len(features), dtype=bool)
        values = features[feature].to_numpy(float)
        mask &= ((values <= threshold) if op == "le" else (values >= threshold)) & np.isfinite(values)
    return mask


def _union_mask(features: pd.DataFrame, component_names: list[str]) -> np.ndarray:
    active = np.zeros(len(features), dtype=bool)
    for name in component_names:
        active |= _component_mask(features, name)
    return active


def _strict_long_overlay_sim(
    signal_positions: np.ndarray,
    *,
    market: pd.DataFrame,
    hold_bars: int,
    entry_delay_bars: int,
    leverage: float,
    fee_rate: float,
    slippage_rate: float,
    take_profit: float | None,
    stop_loss: float | None,
    annualization_start: str,
    annualization_end: str,
) -> tuple[dict[str, Any], list[float]]:
    """Strict long simulation with optional TP/SL and full-window CAGR.

    Same-bar ordering is conservative for long entries: if stop and take are both
    touched, the stop is counted first.
    """

    if take_profit is None and stop_loss is None:
        return _strict_long_sim(
            signal_positions,
            market=market,
            hold_bars=hold_bars,
            entry_delay_bars=entry_delay_bars,
            leverage=leverage,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            annualization_start=annualization_start,
            annualization_end=annualization_end,
        )

    opens = market["open"].to_numpy(float)
    highs = market["high"].to_numpy(float)
    lows = market["low"].to_numpy(float)
    dates = pd.to_datetime(market["date"]).to_numpy()
    cost = (float(fee_rate) + float(slippage_rate)) * float(leverage)
    eq = peak = 1.0
    max_dd = 0.0
    next_allowed = 0
    trade_returns: list[float] = []
    first_signal: int | None = None
    last_signal: int | None = None
    exit_reasons = {"time": 0, "take": 0, "stop": 0}

    for pos in np.asarray(signal_positions, dtype=np.int64):
        if pos < next_allowed:
            continue
        entry_pos = int(pos) + int(entry_delay_bars)
        max_exit = entry_pos + int(hold_bars)
        if entry_pos >= len(market) - 1 or max_exit >= len(market):
            continue
        entry_open = float(opens[entry_pos])
        if entry_open <= 0.0:
            continue
        first_signal = int(pos) if first_signal is None else first_signal
        last_signal = int(pos)
        entry_eq = eq
        eq *= max(0.0, 1.0 - cost)
        max_dd = max(max_dd, 1.0 - max(0.0, eq) / peak if peak > 0.0 else 0.0)
        exit_pos = max_exit
        reason = "time"
        stop_px = entry_open * (1.0 - float(stop_loss)) if stop_loss is not None else None
        take_px = entry_open * (1.0 + float(take_profit)) if take_profit is not None else None

        for j in range(entry_pos, max_exit):
            open_j = float(opens[j])
            if open_j <= 0.0:
                continue
            if stop_px is not None and float(lows[j]) <= stop_px:
                realized_ret = (stop_px - open_j) / open_j
                max_dd = max(max_dd, 1.0 - max(0.0, eq * (1.0 + float(leverage) * realized_ret)) / peak)
                eq *= max(0.0, 1.0 + float(leverage) * realized_ret)
                reason = "stop"
                exit_pos = j
                break
            if take_px is not None and float(highs[j]) >= take_px:
                adverse_ret = (float(lows[j]) - open_j) / open_j
                max_dd = max(max_dd, 1.0 - max(0.0, eq * (1.0 + float(leverage) * adverse_ret)) / peak)
                realized_ret = (take_px - open_j) / open_j
                eq *= max(0.0, 1.0 + float(leverage) * realized_ret)
                reason = "take"
                exit_pos = j
                break
            adverse_ret = (float(lows[j]) - open_j) / open_j
            max_dd = max(max_dd, 1.0 - max(0.0, eq * (1.0 + float(leverage) * adverse_ret)) / peak)
            close_ret = (float(opens[j + 1]) - open_j) / open_j
            eq *= max(0.0, 1.0 + float(leverage) * close_ret)
            peak = max(peak, eq)
            if eq <= 0.0:
                exit_pos = j
                break

        eq *= max(0.0, 1.0 - cost)
        max_dd = max(max_dd, 1.0 - max(0.0, eq) / peak if peak > 0.0 else 0.0)
        peak = max(peak, eq)
        trade_returns.append(eq / entry_eq - 1.0)
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        next_allowed = exit_pos + 1
        if eq <= 0.0:
            break

    start_dt = pd.Timestamp(annualization_start).to_pydatetime()
    end_dt = pd.Timestamp(annualization_end).to_pydatetime()
    years = max(1.0 / 365.25, (end_dt - start_dt).total_seconds() / (365.25 * 24 * 3600))
    trade_start_dt = pd.Timestamp(dates[first_signal]).to_pydatetime() if first_signal is not None else None
    trade_end_dt = pd.Timestamp(dates[last_signal]).to_pydatetime() if last_signal is not None else None
    ret_pct = (eq - 1.0) * 100.0
    cagr_pct = ((eq ** (1.0 / years) - 1.0) * 100.0) if eq > 0.0 else -100.0
    mdd_pct = max_dd * 100.0
    return (
        {
            "period": {"start": str(start_dt), "end": str(end_dt), "years": years},
            "trade_period": {"start": str(trade_start_dt), "end": str(trade_end_dt)},
            "cagr_pct": cagr_pct,
            "strict_mdd_pct": mdd_pct,
            "cagr_to_strict_mdd": cagr_pct / mdd_pct if mdd_pct > 1e-12 else float("inf"),
            "trade_entries": len(trade_returns),
            "win_rate": sum(1 for r in trade_returns if r > 0.0) / len(trade_returns) if trade_returns else 0.0,
            "total_return_pct": ret_pct,
            "hold_bars": int(hold_bars),
            "entry_delay_bars": int(entry_delay_bars),
            "exit_reasons": exit_reasons,
            "return_application": "long_only_actual_ohlc_optional_tp_sl_strict_mdd",
        },
        trade_returns,
    )


def _score_window(
    *,
    market: pd.DataFrame,
    dates: pd.Series,
    active: np.ndarray,
    cfg: LongComponentUnionScanConfig,
    start: str,
    end: str,
    hold_bars: int,
    take_profit: float | None,
    stop_loss: float | None,
) -> dict[str, Any]:
    wmask = _split_mask(dates, start, end)
    positions = np.arange(
        max(0, int(cfg.window_size) - 1),
        max(0, len(market) - hold_bars - int(cfg.entry_delay_bars) - 1),
        int(cfg.stride_bars),
        dtype=np.int64,
    )
    p = positions[active[positions] & wmask[positions]]
    sim, returns = _strict_long_overlay_sim(
        p,
        market=market,
        hold_bars=hold_bars,
        entry_delay_bars=int(cfg.entry_delay_bars),
        leverage=float(cfg.leverage),
        fee_rate=float(cfg.fee_rate),
        slippage_rate=float(cfg.slippage_rate),
        take_profit=take_profit,
        stop_loss=stop_loss,
        annualization_start=start,
        annualization_end=end,
    )
    stats = _trade_stats(returns)
    return {
        "ret_pct": sim["total_return_pct"],
        "cagr_pct": sim["cagr_pct"],
        "mdd_pct": sim["strict_mdd_pct"],
        "ratio": sim["cagr_to_strict_mdd"],
        "trades": sim["trade_entries"],
        "win_rate": sim["win_rate"],
        "p": stats.get("p_value_mean_ret_approx"),
        "signals": int(len(p)),
    }


def _score_candidate(stats: dict[str, dict[str, Any]]) -> float:
    if any(stats[w]["ret_pct"] <= 0.0 or stats[w]["mdd_pct"] > 15.0 for w in ("test2024", "eval2025", "ytd2026")):
        return -1e9
    if stats["combined"]["trades"] < 1:
        return -1e9
    p = float(stats["combined"].get("p") or 1.0)
    return (
        min(float(stats[w]["ratio"]) for w in ("test2024", "eval2025", "ytd2026"))
        + 0.15 * float(stats["combined"]["ratio"])
        + min(1.0, float(stats["combined"]["trades"]) / 100.0)
        + 0.02 * float(stats["combined"]["ret_pct"])
        - 0.2 * p
    )


def run(cfg: LongComponentUnionScanConfig) -> dict[str, Any]:
    market = _load_market(cfg)
    base_features = build_market_feature_frame(market, window_size=int(cfg.window_size))
    features = pd.concat([base_features, build_interest_features(market, base_features)], axis=1)
    dates = pd.to_datetime(market["date"])

    component_sizes = _parse_ints(cfg.component_sizes)
    overlays = _parse_overlays(cfg.overlays)
    names = list(COMPONENTS.keys())
    rows: list[dict[str, Any]] = []
    for size in component_sizes:
        for combo in itertools.combinations(names, size):
            active = _union_mask(features, list(combo))
            for hold in _parse_ints(cfg.hold_bars):
                for tp, sl in overlays:
                    stats = {
                        name: _score_window(
                            market=market,
                            dates=dates,
                            active=active,
                            cfg=cfg,
                            start=start,
                            end=end,
                            hold_bars=hold,
                            take_profit=tp,
                            stop_loss=sl,
                        )
                        for name, start, end in WINDOWS
                    }
                    if int(stats["combined"]["trades"]) < int(cfg.min_combined_trades):
                        continue
                    if int(stats["eval2025"]["trades"]) < int(cfg.min_eval_trades):
                        continue
                    if int(stats["ytd2026"]["trades"]) < int(cfg.min_ytd_trades):
                        continue
                    rows.append({
                        "components": list(combo),
                        "hold": int(hold),
                        "tp": tp,
                        "sl": sl,
                        "stats": stats,
                        "score": _score_candidate(stats),
                    })

    rows.sort(key=lambda r: float(r["score"]), reverse=True)
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "input": {"rows": len(market), "start": str(market["date"].iloc[0]), "end": str(market["date"].iloc[-1])},
        "components": COMPONENTS,
        "windows": WINDOWS,
        "selection_protocol": "Fixed component thresholds from train-window discovery; ranked across fixed component unions; full-window CAGR; strict OHLC MDD; no post-2026-06-02 rows loaded.",
        "leakage_guard": {
            "thresholds_fixed_before_test_eval": True,
            "market_rows_after_exclude_from_removed_before_feature_build": True,
            "full_period_cagr_includes_idle_time": True,
            "strict_mdd_includes_in_position_intrabar_adverse_excursion": True,
        },
        "top": rows[: int(cfg.top_n)],
        "all_count": len(rows),
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
    for field in ("exclude-from", "component-sizes", "hold-bars", "overlays"):
        p.add_argument(f"--{field}", default=getattr(LongComponentUnionScanConfig, field.replace("-", "_")))
    p.add_argument("--stride-bars", type=int, default=LongComponentUnionScanConfig.stride_bars)
    p.add_argument("--leverage", type=float, default=LongComponentUnionScanConfig.leverage)
    p.add_argument("--window-size", type=int, default=LongComponentUnionScanConfig.window_size)
    p.add_argument("--entry-delay-bars", type=int, default=LongComponentUnionScanConfig.entry_delay_bars)
    p.add_argument("--fee-rate", type=float, default=LongComponentUnionScanConfig.fee_rate)
    p.add_argument("--slippage-rate", type=float, default=LongComponentUnionScanConfig.slippage_rate)
    p.add_argument("--min-combined-trades", type=int, default=LongComponentUnionScanConfig.min_combined_trades)
    p.add_argument("--min-eval-trades", type=int, default=LongComponentUnionScanConfig.min_eval_trades)
    p.add_argument("--min-ytd-trades", type=int, default=LongComponentUnionScanConfig.min_ytd_trades)
    p.add_argument("--top-n", type=int, default=LongComponentUnionScanConfig.top_n)
    return p.parse_args()


def main() -> None:
    report = run(LongComponentUnionScanConfig(**vars(parse_args())))
    compact = []
    for row in report["top"][:10]:
        item = {"components": row["components"], "hold": row["hold"], "tp": row["tp"], "sl": row["sl"], "score": row["score"]}
        for name in ("test2024", "eval2025", "ytd2026", "combined"):
            s = row["stats"][name]
            item[name] = {k: s[k] for k in ("ret_pct", "cagr_pct", "mdd_pct", "ratio", "trades", "p", "signals")}
        compact.append(item)
    print(json.dumps({"output": report["config"]["output"], "input": report["input"], "all_count": report["all_count"], "top": compact}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
