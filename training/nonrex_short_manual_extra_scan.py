"""Manual extra scan for non-REX bearish short feature ideas.

This script preserves the follow-up search beyond the already recorded REX,
FX-stress, and premium-discount shorts.  It intentionally tests a small set of
interpretable ideas (kimchi unwind, DXY pressure, sell-flow liquidation, failed
bounce, funding crowding) with short TP/SL exits and reports full-window CAGR,
strict short MDD, absolute return, and trades.
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
from training.long_regime_interest_gate_validation import build_interest_features
from training.strict_bar_backtest import _trade_stats


@dataclass(frozen=True)
class ManualExtraShortConfig(LongComboScanConfig):
    exclude_from: str = "2026-06-02"
    leverage: float = 0.5


WINDOWS = {
    "test2024": ("2024-01-01", "2025-01-01"),
    "eval2025": ("2025-01-01", "2026-01-01"),
    "ytd2026": ("2026-01-01", "2026-06-02"),
    "bear_combined_2025_2026h1": ("2025-01-01", "2026-06-02"),
}

SPECS: list[tuple[str, list[tuple[str, str, float]]]] = [
    ("kimchi_unwind_weak3d", [("htf_3d_return_1", "le", 0.2), ("kimchi_premium_change", "le", 0.1)]),
    ("dxy_weak3d", [("htf_3d_return_1", "le", 0.2), ("dxy_momentum", "ge", 0.8)]),
    ("dxy_weak3d_kimchi", [("htf_3d_return_1", "le", 0.2), ("dxy_momentum", "ge", 0.8), ("kimchi_premium_change", "le", 0.2)]),
    ("sellflow_weak3d", [("htf_3d_return_1", "le", 0.2), ("taker_imbalance", "le", 0.2), ("quote_vol_z_1d", "ge", 0.7)]),
    ("funding_crowded_weak1d", [("htf_1d_return_4", "le", 0.2), ("funding_zscore", "ge", 0.8)]),
    ("mfi_bounce_fail_weak3d", [("htf_3d_return_1", "le", 0.2), ("mfi_norm", "ge", 0.8), ("upper_shadow", "ge", 0.8)]),
    ("bb_bounce_fail_weekweak", [("weekly_return_1w", "le", 0.2), ("bb_z", "ge", 0.8), ("upper_shadow", "ge", 0.7)]),
    ("dxy_premium_panic", [("dxy_momentum", "ge", 0.8), ("premium_index_zscore", "le", 0.1), ("htf_1d_return_4", "le", 0.3)]),
]


def _threshold(features: pd.DataFrame, train: np.ndarray, col: str, quantile: float) -> float:
    values = features[col].to_numpy(float)
    ref = values[train & np.isfinite(values)]
    if ref.size < 500 or np.nanstd(ref) <= 1e-12:
        raise ValueError(f"unusable feature for quantile threshold: {col}")
    return float(np.quantile(ref, quantile))


def _mask(features: pd.DataFrame, train: np.ndarray, conds: list[tuple[str, str, float]]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    active = np.ones(len(features), dtype=bool)
    metas = []
    for col, op, quantile in conds:
        thr = _threshold(features, train, col, quantile)
        values = features[col].to_numpy(float)
        active &= ((values <= thr) if op == "le" else (values >= thr)) & np.isfinite(values)
        metas.append({"feature": col, "op": op, "q": quantile, "thr": thr})
    return active, metas


def _sim_short(
    *,
    market: pd.DataFrame,
    signal_positions: np.ndarray,
    start: str,
    end: str,
    hold_bars: int,
    take_profit: float | None,
    stop_loss: float | None,
    leverage: float,
    fee_rate: float,
    slippage_rate: float,
) -> dict[str, Any]:
    opens = market["open"].to_numpy(float)
    highs = market["high"].to_numpy(float)
    lows = market["low"].to_numpy(float)
    eq = peak = 1.0
    max_dd = 0.0
    returns: list[float] = []
    next_allowed = 0
    cost = (fee_rate + slippage_rate) * leverage
    for pos in np.asarray(signal_positions, dtype=np.int64):
        if pos < next_allowed:
            continue
        entry_pos = int(pos) + 1
        exit_pos = entry_pos + int(hold_bars)
        if exit_pos >= len(market):
            continue
        entry = float(opens[entry_pos])
        if entry <= 0:
            continue
        entry_eq = eq
        eq *= max(0.0, 1.0 - cost)
        max_dd = max(max_dd, 1.0 - eq / peak)
        exit_ret = -(float(opens[exit_pos]) - entry) / entry
        actual_exit = exit_pos
        for j in range(entry_pos, exit_pos):
            adverse = (float(highs[j]) - entry) / entry
            max_dd = max(max_dd, 1.0 - max(0.0, eq * (1.0 - leverage * adverse)) / peak)
            if stop_loss is not None and float(highs[j]) >= entry * (1.0 + stop_loss):
                exit_ret = -float(stop_loss)
                actual_exit = j
                break
            if take_profit is not None and float(lows[j]) <= entry * (1.0 - take_profit):
                exit_ret = float(take_profit)
                actual_exit = j
                break
        eq *= max(0.0, 1.0 + leverage * exit_ret)
        peak = max(peak, eq)
        eq *= max(0.0, 1.0 - cost)
        max_dd = max(max_dd, 1.0 - eq / peak)
        peak = max(peak, eq)
        returns.append(eq / entry_eq - 1.0)
        next_allowed = actual_exit + 1
    years = max(1.0 / 365.25, (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() / (365.25 * 24 * 3600))
    ret_pct = (eq - 1.0) * 100.0
    cagr_pct = ((eq ** (1.0 / years) - 1.0) * 100.0) if eq > 0 else -100.0
    mdd_pct = max_dd * 100.0
    return {
        "ret_pct": ret_pct,
        "cagr_pct": cagr_pct,
        "mdd_pct": mdd_pct,
        "ratio": cagr_pct / mdd_pct if mdd_pct > 1e-12 else (float("inf") if cagr_pct > 0 else 0.0),
        "trades": len(returns),
        "win_rate": sum(r > 0 for r in returns) / len(returns) if returns else 0.0,
        "p": _trade_stats(returns).get("p_value_mean_ret_approx"),
    }


def run(cfg: ManualExtraShortConfig) -> dict[str, Any]:
    market = _load_market(cfg)
    base_features = build_market_feature_frame(market, window_size=int(cfg.window_size))
    features = pd.concat([base_features, build_interest_features(market, base_features)], axis=1)
    dates = pd.to_datetime(market["date"])
    train = _split_mask(dates, "2020-01-01", "2024-01-01")
    rows = []
    for family, conds in SPECS:
        active, metas = _mask(features, train, conds)
        for hold in [72, 144, 216, 288]:
            for take_profit, stop_loss in [(0.015, 0.01), (0.025, 0.015), (0.04, 0.025), (0.06, 0.04), (None, None)]:
                stats = {}
                for window_name, (start, end) in WINDOWS.items():
                    wmask = _split_mask(dates, start, end)
                    positions = np.arange(max(0, int(cfg.window_size) - 1), max(0, len(market) - hold - 2), 12, dtype=np.int64)
                    signals = positions[active[positions] & wmask[positions]]
                    st = _sim_short(
                        market=market,
                        signal_positions=signals,
                        start=start,
                        end=end,
                        hold_bars=hold,
                        take_profit=take_profit,
                        stop_loss=stop_loss,
                        leverage=float(cfg.leverage),
                        fee_rate=float(cfg.fee_rate),
                        slippage_rate=float(cfg.slippage_rate),
                    )
                    st["signals"] = int(len(signals))
                    stats[window_name] = st
                if stats["bear_combined_2025_2026h1"]["ratio"] >= 2.5 and stats["bear_combined_2025_2026h1"]["trades"] >= 20 and stats["eval2025"]["ret_pct"] > 0 and stats["ytd2026"]["ret_pct"] > 0:
                    rows.append({"family": family, "conditions": metas, "hold_bars": hold, "take_profit": take_profit, "stop_loss": stop_loss, "stats": stats})
    rows.sort(key=lambda r: (float(r["stats"]["bear_combined_2025_2026h1"]["ratio"]), float(r["stats"]["bear_combined_2025_2026h1"]["ret_pct"])), reverse=True)
    report = {"as_of": datetime.now(timezone.utc).isoformat(), "config": asdict(cfg), "top": rows, "all_count": len(rows)}
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--funding-csv", default="")
    p.add_argument("--premium-csv", default="")
    p.add_argument("--exclude-from", default=ManualExtraShortConfig.exclude_from)
    p.add_argument("--leverage", type=float, default=ManualExtraShortConfig.leverage)
    p.add_argument("--window-size", type=int, default=ManualExtraShortConfig.window_size)
    p.add_argument("--fee-rate", type=float, default=ManualExtraShortConfig.fee_rate)
    p.add_argument("--slippage-rate", type=float, default=ManualExtraShortConfig.slippage_rate)
    return p.parse_args()


def main() -> None:
    report = run(ManualExtraShortConfig(**vars(parse_args())))
    print(json.dumps({"output": report["config"]["output"], "all_count": report["all_count"], "top": report["top"][:10]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
