"""Sweep causal micro-path opportunity filters plus simple side rules.

This is not an ML model; it is an alpha surface probe. Parameters are selected on
validation only, then applied once to eval.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from training.online_risk_overlay_backtest import OnlineRiskOverlayConfig, run_overlay

@dataclass(frozen=True)
class Cfg:
    train_candidates: str
    eval_candidates: str
    output: str
    work_dir: str = "results/event_micro_rule_sweep"
    validation_start: str = "2023-01-01"
    validation_end: str = "2024-12-31 23:59:59"
    market_csv: str = "data/2020-01-01_2026-06-01_btcusdt_futures_5m.csv.gz"
    min_val_trades: int = 50
    top_n: int = 20

OPPORTUNITY_FEATURES = [
    "mp_absret_sum_24", "mp_absret_sum_48", "mp_absret_sum_96",
    "mp_realized_vol_24", "mp_realized_vol_48", "mp_realized_vol_96",
    "mp_range_24", "mp_range_48", "mp_range_96", "range_vol", "window_drawdown",
]
SIGNED_FEATURES = [
    "mp_ret_3", "mp_ret_6", "mp_ret_12", "mp_ret_24", "mp_ret_48", "mp_ret_96", "mp_ret_288",
    "mp_ret_accel_12_48", "mp_ret_accel_24_96",
    "mp_range_pos_12", "mp_range_pos_24", "mp_range_pos_48", "mp_range_pos_96", "mp_range_pos_288",
    "mp_taker_imbalance_mean_12", "mp_taker_imbalance_mean_24", "mp_taker_imbalance_mean_48", "mp_taker_imbalance_mean_96",
    "trend_12", "trend_24", "trend_96", "htf_4h_return_4", "htf_1d_return_4", "htf_3d_return_4", "htf_1w_return_4",
]
QUANTILES = [0.5, 0.6, 0.7, 0.8, 0.9]
MODES = ["follow", "fade"]


def load(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in open(path) if line.strip()]


def group(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    d: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        d[int(row["signal_pos"])].append(row)
    return [d[k] for k in sorted(d)]


def date(g: list[dict[str, Any]]) -> str:
    return str(g[0].get("date", ""))


def feat(g: list[dict[str, Any]], name: str) -> float:
    snap = g[0].get("feature_snapshot", {}) if isinstance(g[0].get("feature_snapshot"), dict) else {}
    try:
        return float(snap.get(name, 0.0) or 0.0)
    except Exception:
        return 0.0


def thresholds(groups: list[list[dict[str, Any]]], feature: str) -> list[tuple[float, float]]:
    vals = np.asarray([feat(g, feature) for g in groups], dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 50 or np.std(vals) < 1e-12:
        return []
    return [(q, float(np.quantile(vals, q))) for q in QUANTILES]


def decide_side(v: float, mode: str) -> str:
    if mode == "follow":
        return "LONG" if v >= 0 else "SHORT"
    return "SHORT" if v >= 0 else "LONG"


def write_predictions(groups: list[list[dict[str, Any]]], path: str, *, opp_feature: str, threshold: float, side_feature: str, mode: str) -> dict[str, Any]:
    out=[]; trade=long=short=0
    for g in groups:
        opp = feat(g, opp_feature)
        side_v = feat(g, side_feature)
        if opp >= threshold and abs(side_v) > 1e-12:
            side = decide_side(side_v, mode)
            pred={"gate":"TRADE", "side":side, "hold_bars":288, "confidence":"HIGH", "family":"event_micro_rule"}
            scale=0.5; trade += 1; long += int(side=="LONG"); short += int(side=="SHORT")
        else:
            pred={"gate":"NO_TRADE", "side":"NONE", "hold_bars":0, "confidence":"LOW", "family":"event_micro_rule"}
            scale=0.0
        out.append({"date":g[0]["date"], "signal_pos":g[0]["signal_pos"], "prediction":pred, "position_scale":scale})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, sort_keys=True) for r in out)+"\n")
    return {"rows": len(out), "trade_signals": trade, "long_signals": long, "short_signals": short, "output": path}


def eval_rule(groups: list[list[dict[str, Any]]], cfg: Cfg, name: str, params: dict[str, Any]) -> dict[str, Any]:
    pp = str(Path(cfg.work_dir)/f"{name}.jsonl")
    ps = write_predictions(groups, pp, **params)
    bt = run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=pp, market_csv=cfg.market_csv, output=str(Path(cfg.work_dir)/f"{name}.bt.json"), leverage=1.0, entry_delay_bars=1))
    return {"params": params, "prediction_summary": ps, "sim": bt["sim"], "trade_stats": bt["trade_stats"]}


def run(cfg: Cfg) -> dict[str, Any]:
    allg = group(load(cfg.train_candidates))
    evg = group(load(cfg.eval_candidates))
    fit = [g for g in allg if date(g) < cfg.validation_start]
    val = [g for g in allg if cfg.validation_start <= date(g) <= cfg.validation_end]
    Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
    rows=[]
    for opp in OPPORTUNITY_FEATURES:
        for q, thr in thresholds(fit, opp):
            for side_feature in SIGNED_FEATURES:
                if side_feature == opp:
                    continue
                for mode in MODES:
                    params={"opp_feature":opp, "threshold":thr, "side_feature":side_feature, "mode":mode}
                    res = eval_rule(val, cfg, f"val_{len(rows):04d}", params)
                    score = float(res["sim"]["cagr_to_strict_mdd"])
                    if int(res["sim"]["trade_entries"]) < cfg.min_val_trades:
                        score -= 1000.0
                    rows.append({"quantile": q, "threshold": thr, "score": score, **res})
    rows.sort(key=lambda r: r["score"], reverse=True)
    selected = rows[0] if rows else None
    eval_res = None
    if selected:
        eval_res = eval_rule(evg, cfg, "selected_eval", selected["params"])
    report={
        "config": cfg.__dict__,
        "rows": {"fit": len(fit), "val": len(val), "eval": len(evg)},
        "searched": len(rows),
        "top_val": rows[:cfg.top_n],
        "selected": selected,
        "eval": eval_res,
        "leakage_guard": "Thresholds are computed on fit distribution; rule ranking/selection uses validation; eval is final holdout.",
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> Cfg:
    p=argparse.ArgumentParser()
    p.add_argument("--train-candidates", required=True)
    p.add_argument("--eval-candidates", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--work-dir", default=Cfg.work_dir)
    p.add_argument("--validation-start", default=Cfg.validation_start)
    p.add_argument("--validation-end", default=Cfg.validation_end)
    p.add_argument("--market-csv", default=Cfg.market_csv)
    p.add_argument("--min-val-trades", type=int, default=Cfg.min_val_trades)
    p.add_argument("--top-n", type=int, default=Cfg.top_n)
    return Cfg(**vars(p.parse_args()))


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
