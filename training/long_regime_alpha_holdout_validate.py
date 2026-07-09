"""Validate discovered long-regime alpha families across fixed holdout windows.

The discovery scan is allowed to find hypotheses, but this script makes the
selection/evaluation boundary explicit.  Thresholds are always fitted on the
train window only; each named candidate is then scored on calendar holdouts
with CAGR annualized over the full window including cash/no-trade time.
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
from training.long_regime_alpha_family_scan import _fit_mask
from training.long_regime_combo_scan import LongComboScanConfig, _load_market, _split_mask, _strict_long_sim
from training.long_regime_interest_gate_validation import build_interest_features
from training.strict_bar_backtest import _trade_stats


@dataclass(frozen=True)
class HoldoutConfig(LongComboScanConfig):
    train_start: str = "2020-01-01"
    train_end: str = "2024-01-01"
    exclude_from: str = "2026-06-02"
    leverage: float = 0.5
    min_trades: int = 5


def _candidates() -> list[dict[str, Any]]:
    return [
        {
            "name": "range_breakout_12h_width90_z48_70_h720_s12",
            "family": "range_breakout",
            "conditions": [("rex_576_range_width_pct", "ge", 0.9), ("close_zscore_48", "ge", 0.7)],
            "hold_bars": 720,
            "stride_bars": 12,
        },
        {
            "name": "range_breakout_12h_width90_z48_80_h720_s12",
            "family": "range_breakout",
            "conditions": [("rex_576_range_width_pct", "ge", 0.9), ("close_zscore_48", "ge", 0.8)],
            "hold_bars": 720,
            "stride_bars": 12,
        },
        {
            "name": "range_breakout_12h_width90_z48_70_h432_s12",
            "family": "range_breakout",
            "conditions": [("rex_576_range_width_pct", "ge", 0.9), ("close_zscore_48", "ge", 0.7)],
            "hold_bars": 432,
            "stride_bars": 12,
        },
        {
            "name": "range_breakout_12h_width90_bb90_h432_s12",
            "family": "range_breakout",
            "conditions": [("rex_576_range_width_pct", "ge", 0.9), ("bb_z", "ge", 0.9)],
            "hold_bars": 432,
            "stride_bars": 12,
        },
        {
            "name": "deriv_squeeze_prem20_mom90_h720_s12",
            "family": "deriv_squeeze_long",
            "conditions": [("premium_index_change", "le", 0.2), ("htf_1d_return_4", "ge", 0.9)],
            "hold_bars": 720,
            "stride_bars": 12,
        },
        {
            "name": "compression_2w_width05_trend24_90_h576_s12",
            "family": "compression_breakout",
            "conditions": [("rex_2016_range_width_pct", "le", 0.05), ("trend_24", "ge", 0.9)],
            "hold_bars": 576,
            "stride_bars": 12,
        },
        {
            "name": "htf_momentum_1d4_90_pos50_h432_s24",
            "family": "htf_momentum_position",
            "conditions": [("htf_1d_return_4", "ge", 0.9), ("rex_8640_range_pos", "ge", 0.5)],
            "hold_bars": 432,
            "stride_bars": 24,
        },
    ]


def _windows() -> list[tuple[str, str, str]]:
    return [
        ("test_2024", "2024-01-01", "2025-01-01"),
        ("eval_2025", "2025-01-01", "2026-01-01"),
        ("eval_2026h1", "2026-01-01", "2026-06-02"),
        ("combined_2024_2026h1", "2024-01-01", "2026-06-02"),
    ]


def _fit_candidate(features: pd.DataFrame, train: np.ndarray, cand: dict[str, Any]) -> tuple[np.ndarray, list[dict[str, Any]]] | None:
    active = np.ones(len(features), dtype=bool)
    metas: list[dict[str, Any]] = []
    for feature, op, q in cand["conditions"]:
        fm = _fit_mask(features, train, feature, op, q)
        if fm is None:
            return None
        mask, meta = fm
        active &= mask
        metas.append(meta)
    return active, metas


def _positions(active: np.ndarray, window: np.ndarray, cfg: HoldoutConfig, hold_bars: int, stride_bars: int, n: int) -> np.ndarray:
    positions = np.arange(max(0, int(cfg.window_size) - 1), max(0, n - hold_bars - int(cfg.entry_delay_bars) - 1), stride_bars, dtype=np.int64)
    return positions[active[positions] & window[positions]]


def run(cfg: HoldoutConfig) -> dict[str, Any]:
    market = _load_market(cfg)
    base_features = build_market_feature_frame(market, window_size=int(cfg.window_size))
    features = pd.concat([base_features, build_interest_features(market, base_features)], axis=1)
    dates = pd.to_datetime(market["date"])
    train = _split_mask(dates, cfg.train_start, cfg.train_end)
    rows: list[dict[str, Any]] = []
    for cand in _candidates():
        fit = _fit_candidate(features, train, cand)
        if fit is None:
            rows.append({"candidate": cand, "error": "feature_missing_or_unusable"})
            continue
        active, thresholds = fit
        windows: dict[str, Any] = {}
        for name, start, end in _windows():
            wmask = _split_mask(dates, start, end)
            p = _positions(active, wmask, cfg, int(cand["hold_bars"]), int(cand["stride_bars"]), len(market))
            sim, returns = _strict_long_sim(
                p,
                market=market,
                hold_bars=int(cand["hold_bars"]),
                entry_delay_bars=int(cfg.entry_delay_bars),
                leverage=float(cfg.leverage),
                fee_rate=float(cfg.fee_rate),
                slippage_rate=float(cfg.slippage_rate),
                annualization_start=start,
                annualization_end=end,
            )
            windows[name] = {"signals": int(len(p)), "sim": sim, "trade_stats": _trade_stats(returns)}
        rows.append({"candidate": cand, "thresholds": thresholds, "windows": windows})
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "input": {"rows": len(market), "start": str(market["date"].iloc[0]), "end": str(market["date"].iloc[-1])},
        "protocol": "Fixed candidates discovered separately; thresholds fitted on train only; reported CAGR uses full calendar window including no-trade time.",
        "rows": rows,
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
    p.add_argument("--train-start", default=HoldoutConfig.train_start)
    p.add_argument("--train-end", default=HoldoutConfig.train_end)
    p.add_argument("--exclude-from", default=HoldoutConfig.exclude_from)
    p.add_argument("--leverage", type=float, default=HoldoutConfig.leverage)
    p.add_argument("--window-size", type=int, default=HoldoutConfig.window_size)
    p.add_argument("--entry-delay-bars", type=int, default=HoldoutConfig.entry_delay_bars)
    p.add_argument("--fee-rate", type=float, default=HoldoutConfig.fee_rate)
    p.add_argument("--slippage-rate", type=float, default=HoldoutConfig.slippage_rate)
    return p.parse_args()


def main() -> None:
    report = run(HoldoutConfig(**vars(parse_args())))
    summary = []
    for row in report["rows"]:
        if "error" in row:
            summary.append({"name": row["candidate"]["name"], "error": row["error"]})
            continue
        item = {"name": row["candidate"]["name"], "family": row["candidate"]["family"]}
        for win_name, win in row["windows"].items():
            s = win["sim"]
            item[win_name] = {
                "ret_pct": s["total_return_pct"],
                "cagr_pct": s["cagr_pct"],
                "mdd_pct": s["strict_mdd_pct"],
                "ratio": s["cagr_to_strict_mdd"],
                "trades": s["trade_entries"],
                "signals": win["signals"],
                "p": win["trade_stats"].get("p_value_mean_ret_approx"),
            }
        summary.append(item)
    print(json.dumps({"output": report["config"]["output"], "summary": summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
