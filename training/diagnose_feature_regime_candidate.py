"""Diagnose a fixed feature-regime filter across splits, months, and overlays.

This is a validation tool, not a selector. It applies a predeclared filter to
existing prediction streams and reports robustness slices without fitting on the
final eval window.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from preprocessing.external_features import attach_wave_trading_external_features
from preprocessing.market_features import build_market_feature_frame
from training.alpha_linear_combo_scan import _load_market, _parse_list
from training.online_risk_overlay_backtest import OnlineRiskOverlayConfig, run_overlay
from training.scan_feature_regime_filters import _apply_filter, _availability_for_feature, _is_trade, _read_jsonl, _slice_rows, _write_jsonl


@dataclass(frozen=True)
class DiagnoseConfig:
    input_csv: str
    base_predictions: str
    eval_predictions: str
    output: str
    work_dir: str
    feature: str
    direction: str = "le"
    threshold: float = 0.0
    scope: str = "ALL"
    preserve_other_side: bool = True
    base_start: str = "2024-07-01"
    base_end: str = "2025-12-31 23:59:59"
    eval_start: str = "2026-01-01"
    eval_end: str = "2026-06-01 00:00:00"
    split_points: str = "2024-07-01:2024-12-31 23:59:59,2025-01-01:2025-12-31 23:59:59"
    leverages: str = "0.20,0.30,0.40"
    pause_after_losses: str = "0,4"
    pause_bars: int = 288
    monthly_loss_stop_pct: float = 0.0
    trade_take_profit_pct: float = 0.0
    window_size: int = 144
    wave_trading_root: str = ""
    external_tolerance: str = "30min"


def _sim_summary(bt: dict[str, Any]) -> dict[str, Any]:
    sim = bt["sim"]
    ts = bt["trade_stats"]
    return {
        "cagr_pct": sim.get("cagr_pct"),
        "strict_mdd_pct": sim.get("strict_mdd_pct"),
        "cagr_to_strict_mdd": sim.get("cagr_to_strict_mdd"),
        "trade_entries": sim.get("trade_entries"),
        "p_value_mean_ret_approx": ts.get("p_value_mean_ret_approx"),
        "mean_trade_ret_pct": ts.get("mean_trade_ret_pct"),
        "t_stat_like": ts.get("t_stat_like"),
    }


def _bt(path: Path, cfg: DiagnoseConfig, *, leverage: float, pause_after_losses: int) -> dict[str, Any]:
    return run_overlay(
        OnlineRiskOverlayConfig(
            predictions_jsonl=str(path),
            market_csv=cfg.input_csv,
            output=str(path.with_suffix(f".lev{str(leverage).replace('.', 'p')}.pal{pause_after_losses}.bt.json")),
            leverage=float(leverage),
            pause_after_losses=int(pause_after_losses),
            pause_bars=int(cfg.pause_bars),
            monthly_loss_stop_pct=float(cfg.monthly_loss_stop_pct),
            trade_take_profit_pct=float(cfg.trade_take_profit_pct),
        )
    )


def _month_ranges(start: str, end: str) -> list[tuple[str, str, str]]:
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    out = []
    cur = pd.Timestamp(year=s.year, month=s.month, day=1)
    while cur <= e:
        nxt = cur + pd.offsets.MonthBegin(1)
        ms = max(s, cur)
        me = min(e, nxt - pd.Timedelta(seconds=1))
        out.append((cur.strftime("%Y-%m"), str(ms), str(me)))
        cur = nxt
    return out


def _run_slice(name: str, rows: list[dict[str, Any]], values: np.ndarray, availability: np.ndarray | None, cfg: DiagnoseConfig, work: Path, *, leverage: float, pause_after_losses: int) -> dict[str, Any]:
    filtered, stats = _apply_filter(rows, values, feature=cfg.feature, direction=cfg.direction, threshold=float(cfg.threshold), scope=cfg.scope, preserve_other_side=bool(cfg.preserve_other_side), availability=availability)
    filtered = [r for r in filtered if _is_trade(r)]
    path = work / f"{name}.jsonl"
    if not filtered:
        _write_jsonl(path, [])
        return {
            "filter_stats": stats,
            "summary": {
                "cagr_pct": 0.0,
                "strict_mdd_pct": 0.0,
                "cagr_to_strict_mdd": 0.0,
                "trade_entries": 0,
                "p_value_mean_ret_approx": None,
                "mean_trade_ret_pct": None,
                "t_stat_like": None,
            },
            "period": {"start": None, "end": None, "years": 0.0},
        }
    _write_jsonl(path, filtered)
    bt = _bt(path, cfg, leverage=leverage, pause_after_losses=pause_after_losses)
    return {"filter_stats": stats, "summary": _sim_summary(bt), "period": bt["period"]}


def run_diagnostics(cfg: DiagnoseConfig) -> dict[str, Any]:
    market = _load_market(cfg.input_csv)
    if cfg.wave_trading_root:
        market = attach_wave_trading_external_features(market, wave_trading_root=cfg.wave_trading_root, tolerance=cfg.external_tolerance)
    features = build_market_feature_frame(market, window_size=int(cfg.window_size))
    if cfg.feature not in features.columns:
        raise ValueError(f"missing feature {cfg.feature}; available sample={list(features.columns)[:20]}")
    values = features[cfg.feature].to_numpy(dtype=float)
    availability = _availability_for_feature(market, cfg.feature)

    base_rows = _slice_rows(_read_jsonl(cfg.base_predictions), cfg.base_start, cfg.base_end)
    eval_rows = _slice_rows(_read_jsonl(cfg.eval_predictions), cfg.eval_start, cfg.eval_end)
    all_rows = base_rows + eval_rows
    work = Path(cfg.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    split_defs: list[tuple[str, str, str, list[dict[str, Any]]]] = []
    for raw in _parse_list(cfg.split_points, str):
        a, b = raw.split(":", 1)
        split_defs.append((a[:7] if a[:4].isdigit() else a, a, b, _slice_rows(base_rows, a, b)))
    split_defs.append(("eval", cfg.eval_start, cfg.eval_end, eval_rows))

    overlays: list[dict[str, Any]] = []
    for lev in _parse_list(cfg.leverages, float):
        for pal in _parse_list(cfg.pause_after_losses, int):
            split_results = {}
            for label, start, end, rows in split_defs:
                split_results[label] = _run_slice(f"{label}_lev{lev}_pal{pal}", rows, values, availability, cfg, work, leverage=float(lev), pause_after_losses=int(pal))
            overlays.append({"overlay": {"leverage": float(lev), "pause_after_losses": int(pal)}, "splits": split_results})

    monthly = {}
    # Use the currently preferred overlay: first matching lev=0.30/pal=4, else first grid item.
    preferred_lev = 0.30 if 0.30 in _parse_list(cfg.leverages, float) else _parse_list(cfg.leverages, float)[0]
    preferred_pal = 4 if 4 in _parse_list(cfg.pause_after_losses, int) else _parse_list(cfg.pause_after_losses, int)[0]
    for label, start, end in _month_ranges(cfg.base_start, cfg.eval_end):
        rows = _slice_rows(all_rows, start, end)
        if not rows:
            continue
        monthly[label] = _run_slice(f"month_{label}_lev{preferred_lev}_pal{preferred_pal}", rows, values, availability, cfg, work, leverage=float(preferred_lev), pause_after_losses=int(preferred_pal))

    report = {
        "config": asdict(cfg),
        "filter": {"feature": cfg.feature, "direction": cfg.direction, "threshold": cfg.threshold, "scope": cfg.scope, "preserve_other_side": cfg.preserve_other_side},
        "overlays": overlays,
        "monthly_preferred_overlay": {"leverage": preferred_lev, "pause_after_losses": preferred_pal, "months": monthly},
        "leakage_guard": {"fixed_filter_only": True, "no_threshold_fit_in_this_script": True, "external_join": "backward_asof_no_future" if cfg.wave_trading_root else "disabled"},
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose a fixed feature regime candidate")
    p.add_argument("--input-csv", required=True)
    p.add_argument("--base-predictions", required=True)
    p.add_argument("--eval-predictions", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--work-dir", required=True)
    p.add_argument("--feature", required=True)
    p.add_argument("--direction", choices=["ge", "le"], default=DiagnoseConfig.direction)
    p.add_argument("--threshold", type=float, default=DiagnoseConfig.threshold)
    p.add_argument("--scope", default=DiagnoseConfig.scope)
    p.add_argument("--block-other-side", dest="preserve_other_side", action="store_false")
    p.add_argument("--base-start", default=DiagnoseConfig.base_start)
    p.add_argument("--base-end", default=DiagnoseConfig.base_end)
    p.add_argument("--eval-start", default=DiagnoseConfig.eval_start)
    p.add_argument("--eval-end", default=DiagnoseConfig.eval_end)
    p.add_argument("--split-points", default=DiagnoseConfig.split_points)
    p.add_argument("--leverages", default=DiagnoseConfig.leverages)
    p.add_argument("--pause-after-losses", default=DiagnoseConfig.pause_after_losses)
    p.add_argument("--pause-bars", type=int, default=DiagnoseConfig.pause_bars)
    p.add_argument("--monthly-loss-stop-pct", type=float, default=DiagnoseConfig.monthly_loss_stop_pct)
    p.add_argument("--trade-take-profit-pct", type=float, default=DiagnoseConfig.trade_take_profit_pct)
    p.add_argument("--window-size", type=int, default=DiagnoseConfig.window_size)
    p.add_argument("--wave-trading-root", default="")
    p.add_argument("--external-tolerance", default=DiagnoseConfig.external_tolerance)
    return p.parse_args()


def main() -> None:
    rep = run_diagnostics(DiagnoseConfig(**vars(parse_args())))
    for item in rep["overlays"]:
        print(json.dumps({"overlay": item["overlay"], "splits": {k: v["summary"] for k, v in item["splits"].items()}}, ensure_ascii=False))
    bad = []
    for m, v in rep["monthly_preferred_overlay"]["months"].items():
        s = v["summary"]
        if s["cagr_pct"] < 0 or s["strict_mdd_pct"] > 8:
            bad.append((m, s))
    print(json.dumps({"bad_months_preferred_overlay": bad[:20]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
