"""Refine non-REX short candidates for persistent bear regimes with TP/SL exits.

The REX short sleeve is useful but already known.  This script searches a
separate idea class: macro/FX stress and derivatives panic shorts without using
REX state as the primary thesis.  It starts from the existing non-LLM short-base
scan candidates, then re-scores fixed gate recipes with short-specific TP/SL
exits over bear-regime windows.
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
class NonRexShortBearConfig(LongComboScanConfig):
    source_scan_json: str = "results/short_base_alpha_scan_fast2_2026-07-08.json"
    exclude_from: str = "2026-06-02"
    leverage: float = 0.5
    top_source_count: int = 30


WINDOWS: dict[str, tuple[str, str]] = {
    "test2024": ("2024-01-01", "2025-01-01"),
    "eval2025": ("2025-01-01", "2026-01-01"),
    "ytd2026": ("2026-01-01", "2026-06-02"),
    "bear_combined_2025_2026h1": ("2025-01-01", "2026-06-02"),
}

MANUAL_CANDIDATES: list[dict[str, Any]] = [
    {
        "kind": "manual_fx_stress_short",
        "gates": [
            {"feature": "htf_3d_return_1", "op": "le", "thr": -0.03252949727545951, "q": 0.2},
            {"feature": "usdkrw_zscore", "op": "ge", "thr": 1.3870063774765273, "q": 0.9},
        ],
    },
    {
        "kind": "manual_premium_panic_short",
        "gates": [
            {"feature": "htf_1d_return_4", "op": "le", "thr": -0.07234231335497887, "q": 0.1},
            {"feature": "premium_index_zscore", "op": "le", "thr": -1.472093119977103, "q": 0.1},
        ],
    },
    {
        "kind": "manual_dxy_riskoff_short",
        "gates": [
            {"feature": "dxy_momentum", "op": "ge", "thr": 0.0021818982809893497, "q": 0.9},
            {"feature": "htf_1d_return_4", "op": "ge", "thr": 0.016096783732847175, "q": 0.6},
        ],
    },
]


def _load_candidates(path: str, top_source_count: int) -> list[dict[str, Any]]:
    rows = []
    p = Path(path)
    if p.exists():
        data = json.loads(p.read_text())
        rows.extend(data.get("top", [])[: int(top_source_count)])
    rows.extend(MANUAL_CANDIDATES)
    return rows


def _active_from_gates(features: pd.DataFrame, gates: list[dict[str, Any]]) -> np.ndarray | None:
    active = np.ones(len(features), dtype=bool)
    for gate in gates:
        col = str(gate["feature"])
        if col not in features.columns:
            return None
        op = str(gate["op"])
        thr = float(gate["thr"])
        values = features[col].to_numpy(float)
        active &= ((values <= thr) if op in {"le", "<="} else (values >= thr)) & np.isfinite(values)
    return active


def _sim_short_tp(
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
    trade_returns: list[float] = []
    next_allowed = 0
    cost = (float(fee_rate) + float(slippage_rate)) * float(leverage)
    for pos in np.asarray(signal_positions, dtype=np.int64):
        if pos < next_allowed:
            continue
        entry_pos = int(pos) + 1
        exit_pos = entry_pos + int(hold_bars)
        if exit_pos >= len(market):
            continue
        entry = float(opens[entry_pos])
        if entry <= 0.0:
            continue
        entry_eq = eq
        eq *= max(0.0, 1.0 - cost)
        max_dd = max(max_dd, 1.0 - eq / peak)
        exit_ret = -(float(opens[exit_pos]) - entry) / entry
        actual_exit = exit_pos
        for j in range(entry_pos, exit_pos):
            adverse = (float(highs[j]) - entry) / entry
            max_dd = max(max_dd, 1.0 - max(0.0, eq * (1.0 - float(leverage) * adverse)) / peak)
            # Conservative same-bar ordering for shorts: stop before take-profit.
            if stop_loss is not None and float(highs[j]) >= entry * (1.0 + float(stop_loss)):
                exit_ret = -float(stop_loss)
                actual_exit = j
                break
            if take_profit is not None and float(lows[j]) <= entry * (1.0 - float(take_profit)):
                exit_ret = float(take_profit)
                actual_exit = j
                break
        eq *= max(0.0, 1.0 + float(leverage) * exit_ret)
        peak = max(peak, eq)
        eq *= max(0.0, 1.0 - cost)
        max_dd = max(max_dd, 1.0 - eq / peak)
        peak = max(peak, eq)
        trade_returns.append(eq / entry_eq - 1.0)
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
        "trades": len(trade_returns),
        "win_rate": sum(r > 0 for r in trade_returns) / len(trade_returns) if trade_returns else 0.0,
        "p": _trade_stats(trade_returns).get("p_value_mean_ret_approx"),
    }


def run(cfg: NonRexShortBearConfig) -> dict[str, Any]:
    market = _load_market(cfg)
    base_features = build_market_feature_frame(market, window_size=int(cfg.window_size))
    features = pd.concat([base_features, build_interest_features(market, base_features)], axis=1)
    dates = pd.to_datetime(market["date"])
    candidates = _load_candidates(cfg.source_scan_json, int(cfg.top_source_count))
    rows: list[dict[str, Any]] = []
    for idx, candidate in enumerate(candidates):
        gates = candidate.get("gates", [])
        active = _active_from_gates(features, gates)
        if active is None:
            continue
        for hold in [48, 72, 96, 144, 216, 288]:
            for take_profit, stop_loss in [(0.01, 0.008), (0.015, 0.01), (0.025, 0.015), (0.04, 0.025), (0.06, 0.04), (None, None)]:
                stats: dict[str, Any] = {}
                for window_name, (start, end) in WINDOWS.items():
                    wmask = _split_mask(dates, start, end)
                    positions = np.arange(max(0, int(cfg.window_size) - 1), max(0, len(market) - hold - 2), 12, dtype=np.int64)
                    signals = positions[active[positions] & wmask[positions]]
                    st = _sim_short_tp(
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
                if stats["eval2025"]["ret_pct"] <= 0 or stats["ytd2026"]["ret_pct"] <= 0:
                    continue
                if int(stats["bear_combined_2025_2026h1"]["trades"]) < 20:
                    continue
                score = min(float(stats["eval2025"]["ratio"]), float(stats["ytd2026"]["ratio"])) + 0.2 * float(stats["bear_combined_2025_2026h1"]["ratio"])
                rows.append({"source_index": idx, "source_kind": candidate.get("kind", candidate.get("kind", "source_top")), "gates": gates, "hold_bars": hold, "take_profit": take_profit, "stop_loss": stop_loss, "stats": stats, "score": score})
    rows.sort(key=lambda r: (float(r["score"]), float(r["stats"]["bear_combined_2025_2026h1"]["ratio"]), float(r["stats"]["bear_combined_2025_2026h1"]["ret_pct"])), reverse=True)
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "protocol": "Non-REX short candidates only; score TP/SL exits for manually declared bearish regime; full-window CAGR and strict short adverse high drawdown.",
        "top": rows[:200],
        "all_count": len(rows),
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--source-scan-json", default=NonRexShortBearConfig.source_scan_json)
    p.add_argument("--funding-csv", default="")
    p.add_argument("--premium-csv", default="")
    p.add_argument("--exclude-from", default=NonRexShortBearConfig.exclude_from)
    p.add_argument("--leverage", type=float, default=NonRexShortBearConfig.leverage)
    p.add_argument("--top-source-count", type=int, default=NonRexShortBearConfig.top_source_count)
    p.add_argument("--window-size", type=int, default=NonRexShortBearConfig.window_size)
    p.add_argument("--fee-rate", type=float, default=NonRexShortBearConfig.fee_rate)
    p.add_argument("--slippage-rate", type=float, default=NonRexShortBearConfig.slippage_rate)
    return p.parse_args()


def main() -> None:
    report = run(NonRexShortBearConfig(**vars(parse_args())))
    print(json.dumps({"output": report["config"]["output"], "all_count": report["all_count"], "top": report["top"][:10]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
