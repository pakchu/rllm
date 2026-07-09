"""Rolling/month-block validation for fixed long squeeze alpha candidates.

The candidate definitions are fixed from the long component union scan.  This
script does not search thresholds; it validates split/month stability and overlap
between candidates using full-window CAGR, absolute return, strict MDD, and
non-overlapping trade simulation.
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
from training.long_component_tp_union_scan import COMPONENTS, _union_mask
from training.long_regime_combo_scan import LongComboScanConfig, _load_market, _split_mask, _strict_long_sim
from training.long_regime_interest_gate_validation import build_interest_features
from training.strict_bar_backtest import _trade_stats


@dataclass(frozen=True)
class LongSqueezeRollingConfig(LongComboScanConfig):
    exclude_from: str = "2026-06-02"
    leverage: float = 0.5
    hold_bars: int = 576
    stride_bars: int = 12
    min_month_trades: int = 1


CANDIDATES: dict[str, list[str]] = {
    "range_funding_premium": ["range_bb90", "funding10_trend70", "premium20_mom90"],
    "funding_compression_premium": ["funding10_trend70", "compress05_trend80", "premium20_mom90"],
    "range_funding_compression": ["range_bb90", "funding10_trend70", "compress05_trend80"],
    "minimal_funding_premium": ["funding10_trend70", "premium20_mom90"],
}

WINDOWS: list[tuple[str, str, str]] = [
    ("test2024", "2024-01-01", "2025-01-01"),
    ("eval2025", "2025-01-01", "2026-01-01"),
    ("ytd2026", "2026-01-01", "2026-06-02"),
    ("combined", "2024-01-01", "2026-06-02"),
]


def _positions(market_len: int, hold_bars: int, entry_delay_bars: int, stride_bars: int, window_size: int) -> np.ndarray:
    return np.arange(
        max(0, int(window_size) - 1),
        max(0, int(market_len) - int(hold_bars) - int(entry_delay_bars) - 1),
        int(stride_bars),
        dtype=np.int64,
    )


def _score_positions(
    p: np.ndarray,
    *,
    market: pd.DataFrame,
    cfg: LongSqueezeRollingConfig,
    start: str,
    end: str,
) -> dict[str, Any]:
    sim, returns = _strict_long_sim(
        p,
        market=market,
        hold_bars=int(cfg.hold_bars),
        entry_delay_bars=int(cfg.entry_delay_bars),
        leverage=float(cfg.leverage),
        fee_rate=float(cfg.fee_rate),
        slippage_rate=float(cfg.slippage_rate),
        annualization_start=start,
        annualization_end=end,
    )
    stats = _trade_stats(returns)
    return {
        "return_pct": sim["total_return_pct"],
        "cagr_pct": sim["cagr_pct"],
        "strict_mdd_pct": sim["strict_mdd_pct"],
        "cagr_to_strict_mdd": sim["cagr_to_strict_mdd"],
        "trades": sim["trade_entries"],
        "win_rate": sim["win_rate"],
        "p_value_mean_ret_approx": stats.get("p_value_mean_ret_approx"),
    }


def _monthly_windows(start: str, end: str) -> list[tuple[str, str]]:
    starts = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="MS")
    out: list[tuple[str, str]] = []
    for s in starts:
        e = min(s + pd.offsets.MonthBegin(1), pd.Timestamp(end))
        if s < e:
            out.append((str(s), str(e)))
    return out


def _compact_month(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "return_pct": round(float(stats["return_pct"]), 4),
        "cagr_pct": round(float(stats["cagr_pct"]), 4),
        "strict_mdd_pct": round(float(stats["strict_mdd_pct"]), 4),
        "cagr_to_strict_mdd": round(float(stats["cagr_to_strict_mdd"]), 4) if np.isfinite(stats["cagr_to_strict_mdd"]) else None,
        "trades": int(stats["trades"]),
        "p_value_mean_ret_approx": None if stats.get("p_value_mean_ret_approx") is None else round(float(stats["p_value_mean_ret_approx"]), 8),
    }


def _month_summary(months: list[dict[str, Any]]) -> dict[str, Any]:
    active = [m for m in months if int(m["stats"]["trades"]) > 0]
    if not active:
        return {"months": len(months), "active_months": 0, "positive_active_months": 0, "negative_active_months": 0, "min_active_return_pct": 0.0, "median_active_return_pct": 0.0, "total_month_trades": 0}
    rets = [float(m["stats"]["return_pct"]) for m in active]
    return {
        "months": len(months),
        "active_months": len(active),
        "positive_active_months": sum(1 for r in rets if r > 0.0),
        "negative_active_months": sum(1 for r in rets if r < 0.0),
        "min_active_return_pct": float(min(rets)),
        "median_active_return_pct": float(np.median(rets)),
        "total_month_trades": int(sum(int(m["stats"]["trades"]) for m in active)),
    }


def run(cfg: LongSqueezeRollingConfig) -> dict[str, Any]:
    market = _load_market(cfg)
    base_features = build_market_feature_frame(market, window_size=int(cfg.window_size))
    features = pd.concat([base_features, build_interest_features(market, base_features)], axis=1)
    dates = pd.to_datetime(market["date"])
    base_positions = _positions(len(market), int(cfg.hold_bars), int(cfg.entry_delay_bars), int(cfg.stride_bars), int(cfg.window_size))

    active_by_name: dict[str, np.ndarray] = {name: _union_mask(features, comps) for name, comps in CANDIDATES.items()}
    pos_by_name: dict[str, np.ndarray] = {name: base_positions[active[base_positions]] for name, active in active_by_name.items()}

    rows: dict[str, Any] = {}
    for name, comps in CANDIDATES.items():
        active = active_by_name[name]
        window_stats = {}
        for wname, start, end in WINDOWS:
            mask = _split_mask(dates, start, end)
            p = base_positions[active[base_positions] & mask[base_positions]]
            window_stats[wname] = _score_positions(p, market=market, cfg=cfg, start=start, end=end)
        months = []
        for start, end in _monthly_windows("2024-01-01", "2026-06-02"):
            mask = _split_mask(dates, start, end)
            p = base_positions[active[base_positions] & mask[base_positions]]
            st = _score_positions(p, market=market, cfg=cfg, start=start, end=end)
            months.append({"start": start, "end": end, "signals": int(len(p)), "stats": _compact_month(st)})
        rows[name] = {
            "components": comps,
            "windows": {k: _compact_month(v) for k, v in window_stats.items()},
            "month_summary": _month_summary(months),
            "months": months,
        }

    overlaps: dict[str, Any] = {}
    names = list(CANDIDATES)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            pa = set(map(int, pos_by_name[a]))
            pb = set(map(int, pos_by_name[b]))
            inter = len(pa & pb)
            union = len(pa | pb)
            overlaps[f"{a}__{b}"] = {
                "signals_a": len(pa),
                "signals_b": len(pb),
                "intersection": inter,
                "jaccard": inter / union if union else 0.0,
                "a_overlap_frac": inter / len(pa) if pa else 0.0,
                "b_overlap_frac": inter / len(pb) if pb else 0.0,
            }

    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "input": {"rows": len(market), "start": str(market["date"].iloc[0]), "end": str(market["date"].iloc[-1])},
        "candidates": CANDIDATES,
        "components": COMPONENTS,
        "rows": rows,
        "overlaps": overlaps,
        "selection_protocol": "No search: fixed candidates from pool are validated by split and calendar month; overlap is measured on raw signal positions.",
        "leakage_guard": {
            "candidate_definitions_fixed_before_validation": True,
            "market_rows_after_exclude_from_removed_before_feature_build": True,
            "full_period_cagr_includes_idle_time": True,
            "strict_mdd_includes_in_position_intrabar_adverse_excursion": True,
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
    p.add_argument("--exclude-from", default=LongSqueezeRollingConfig.exclude_from)
    p.add_argument("--hold-bars", type=int, default=LongSqueezeRollingConfig.hold_bars)
    p.add_argument("--stride-bars", type=int, default=LongSqueezeRollingConfig.stride_bars)
    p.add_argument("--leverage", type=float, default=LongSqueezeRollingConfig.leverage)
    p.add_argument("--window-size", type=int, default=LongSqueezeRollingConfig.window_size)
    p.add_argument("--entry-delay-bars", type=int, default=LongSqueezeRollingConfig.entry_delay_bars)
    p.add_argument("--fee-rate", type=float, default=LongSqueezeRollingConfig.fee_rate)
    p.add_argument("--slippage-rate", type=float, default=LongSqueezeRollingConfig.slippage_rate)
    return p.parse_args()


def main() -> None:
    report = run(LongSqueezeRollingConfig(**vars(parse_args())))
    compact = {}
    for name, row in report["rows"].items():
        compact[name] = {"components": row["components"], "combined": row["windows"]["combined"], "month_summary": row["month_summary"]}
    print(json.dumps({"output": report["config"]["output"], "input": report["input"], "summary": compact, "overlaps": report["overlaps"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
