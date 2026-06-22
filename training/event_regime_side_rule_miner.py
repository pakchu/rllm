"""Mine regime-conditioned side rules for event candidates.

This isolates the current blocker: side prediction. Candidate regimes are built
from signal-time categorical tokens and fit-only numeric quantile thresholds. A
fast validation proxy narrows the grid, then only top validation candidates are
run through the strict overlay backtester. Eval is applied once to the selected
validation rule.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from training.online_risk_overlay_backtest import OnlineRiskOverlayConfig, run_overlay

SIGNED_FEATURES = [
    "mp_ret_3", "mp_ret_6", "mp_ret_12", "mp_ret_24", "mp_ret_48", "mp_ret_96", "mp_ret_288",
    "mp_ret_accel_12_48", "mp_ret_accel_24_96",
    "mp_range_pos_12", "mp_range_pos_24", "mp_range_pos_48", "mp_range_pos_96", "mp_range_pos_288",
    "mp_taker_imbalance_mean_12", "mp_taker_imbalance_mean_24", "mp_taker_imbalance_mean_48", "mp_taker_imbalance_mean_96",
    "trend_12", "trend_24", "trend_96", "htf_4h_return_4", "htf_1d_return_4", "htf_3d_return_4", "htf_1w_return_4",
]
REGIME_NUMERIC_FEATURES = [
    "mp_absret_sum_24", "mp_absret_sum_48", "mp_absret_sum_96", "mp_absret_sum_288",
    "mp_realized_vol_24", "mp_realized_vol_48", "mp_realized_vol_96", "mp_realized_vol_288",
    "mp_range_24", "mp_range_48", "mp_range_96", "mp_range_288",
    "range_vol", "window_drawdown",
    "mp_range_pos_24", "mp_range_pos_48", "mp_range_pos_96", "mp_range_pos_288",
    "trend_96", "htf_4h_return_4", "htf_1d_return_4", "htf_3d_return_4", "htf_1w_return_4",
]
TOKEN_KEYS = [
    "event_trigger_family", "volatility", "window_drawdown", "range_location",
    "mp_short_momentum", "mp_session_momentum", "mp_volatility", "mp_location_24", "mp_flow_24",
    "four_hour_context", "daily_context", "three_day_context", "weekly_context",
    "dxy_pressure", "kimchi_pressure", "usdkrw_pressure",
]
QUANTILES = [0.2, 0.35, 0.5, 0.65, 0.8]
MODES = ["follow", "fade"]

@dataclass(frozen=True)
class Cfg:
    train_candidates: str
    eval_candidates: str
    output: str
    work_dir: str = "results/event_regime_side_rule_miner"
    validation_start: str = "2023-01-01"
    validation_end: str = "2024-12-31 23:59:59"
    market_csv: str = "data/2020-01-01_2026-06-01_btcusdt_futures_5m.csv.gz"
    min_val_signals: int = 80
    min_val_trades: int = 40
    top_proxy: int = 250
    top_backtest: int = 30


def load(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in open(path) if line.strip()]


def group(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    d: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        d[int(row["signal_pos"])].append(row)
    return [d[k] for k in sorted(d)]


def date(g: list[dict[str, Any]]) -> str:
    return str(g[0].get("date", ""))


def snap(g: list[dict[str, Any]]) -> dict[str, Any]:
    return g[0].get("feature_snapshot", {}) if isinstance(g[0].get("feature_snapshot"), dict) else {}


def tokens(g: list[dict[str, Any]]) -> dict[str, Any]:
    return g[0].get("state_tokens", {}) if isinstance(g[0].get("state_tokens"), dict) else {}


def feat(g: list[dict[str, Any]], name: str) -> float:
    try:
        return float(snap(g).get(name, 0.0) or 0.0)
    except Exception:
        return 0.0


def side_reward(g: list[dict[str, Any]], side: str) -> dict[str, float]:
    for row in g:
        if str(row.get("side")) == side:
            rw = row.get("reward", {}) if isinstance(row.get("reward"), dict) else {}
            net = float(rw.get("net_return_pct", 0.0) or 0.0)
            util = float(rw.get("utility", net) or 0.0)
            mae = float(rw.get("mae_pct", 0.0) or 0.0)
            return {"net": net, "utility": util, "mae": mae}
    return {"net": 0.0, "utility": -999.0, "mae": 0.0}


def best_side(g: list[dict[str, Any]]) -> str:
    return "LONG" if side_reward(g, "LONG")["utility"] >= side_reward(g, "SHORT")["utility"] else "SHORT"


def decide(v: float, mode: str) -> str | None:
    if abs(v) <= 1e-12:
        return None
    if mode == "follow":
        return "LONG" if v > 0 else "SHORT"
    return "SHORT" if v > 0 else "LONG"


def build_conditions(fit: list[list[dict[str, Any]]], min_fit_count: int) -> list[dict[str, Any]]:
    conds: list[dict[str, Any]] = [{"kind": "all", "name": "all"}]
    # Categorical token regimes.
    counts: Counter[tuple[str, str]] = Counter()
    for g in fit:
        t = tokens(g)
        for k in TOKEN_KEYS:
            if k in t:
                counts[(k, str(t[k]))] += 1
    for (k, v), n in counts.items():
        if n >= min_fit_count:
            conds.append({"kind": "token", "key": k, "value": v, "name": f"{k}={v}", "fit_count": n})
    # Numeric low/high regimes; thresholds from fit only.
    for f in REGIME_NUMERIC_FEATURES:
        vals = np.asarray([feat(g, f) for g in fit], dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) < min_fit_count or float(np.std(vals)) < 1e-12:
            continue
        for q in QUANTILES:
            thr = float(np.quantile(vals, q))
            conds.append({"kind": "numeric", "feature": f, "op": ">=", "threshold": thr, "quantile": q, "name": f"{f}>={q:.2f}"})
            conds.append({"kind": "numeric", "feature": f, "op": "<=", "threshold": thr, "quantile": q, "name": f"{f}<={q:.2f}"})
    return conds


def cond_match(g: list[dict[str, Any]], cond: dict[str, Any]) -> bool:
    kind = cond["kind"]
    if kind == "all":
        return True
    if kind == "token":
        return str(tokens(g).get(cond["key"], "")) == str(cond["value"])
    v = feat(g, cond["feature"])
    return v >= float(cond["threshold"]) if cond["op"] == ">=" else v <= float(cond["threshold"])


def proxy_eval(groups: list[list[dict[str, Any]]], cond: dict[str, Any], side_feature: str, mode: str) -> dict[str, Any] | None:
    nets=[]; utils=[]; correct=0; long_n=short_n=0
    for g in groups:
        if not cond_match(g, cond):
            continue
        side = decide(feat(g, side_feature), mode)
        if side is None:
            continue
        rw = side_reward(g, side)
        nets.append(rw["net"]); utils.append(rw["utility"])
        correct += int(side == best_side(g))
        long_n += int(side == "LONG"); short_n += int(side == "SHORT")
    n = len(nets)
    if n == 0:
        return None
    arr=np.asarray(nets,float); u=np.asarray(utils,float)
    mean=float(arr.mean()); std=float(arr.std())
    t=float(mean/(std/np.sqrt(n))) if n > 1 and std > 1e-12 else 0.0
    return {
        "signals": n,
        "mean_net_pct": mean,
        "mean_utility": float(u.mean()),
        "hit_rate": float((arr > 0).mean()),
        "side_acc": float(correct / n),
        "t_like": t,
        "long_signals": long_n,
        "short_signals": short_n,
    }


def write_predictions(groups: list[list[dict[str, Any]]], path: str, rule: dict[str, Any]) -> dict[str, Any]:
    cond=rule["condition"]; side_feature=rule["side_feature"]; mode=rule["mode"]
    rows=[]; counts=Counter()
    for g in groups:
        side = decide(feat(g, side_feature), mode) if cond_match(g, cond) else None
        if side:
            pred={"gate":"TRADE","side":side,"hold_bars":288,"confidence":"HIGH","family":"event_regime_side_rule"}
            scale=0.5; counts["TRADE"] += 1; counts[side] += 1
        else:
            pred={"gate":"NO_TRADE","side":"NONE","hold_bars":0,"confidence":"LOW","family":"event_regime_side_rule"}
            scale=0.0; counts["NO_TRADE"] += 1
        rows.append({"date": g[0]["date"], "signal_pos": g[0]["signal_pos"], "prediction": pred, "position_scale": scale})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows)+"\n")
    return {"rows": len(rows), "counts": dict(counts), "output": path}


def backtest(groups: list[list[dict[str, Any]]], cfg: Cfg, name: str, rule: dict[str, Any]) -> dict[str, Any]:
    pred_path=str(Path(cfg.work_dir)/f"{name}.jsonl")
    ps=write_predictions(groups, pred_path, rule)
    bt=run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=pred_path,market_csv=cfg.market_csv,output=str(Path(cfg.work_dir)/f"{name}.bt.json"),leverage=1.0,entry_delay_bars=1))
    return {"rule": rule, "prediction_summary": ps, "sim": bt["sim"], "trade_stats": bt["trade_stats"]}


def run(cfg: Cfg) -> dict[str, Any]:
    allg=group(load(cfg.train_candidates)); evg=group(load(cfg.eval_candidates))
    fit=[g for g in allg if date(g)<cfg.validation_start]
    val=[g for g in allg if cfg.validation_start<=date(g)<=cfg.validation_end]
    conds=build_conditions(fit, max(80, cfg.min_val_signals//2))
    proxy=[]
    for cond in conds:
        for sf in SIGNED_FEATURES:
            for mode in ["follow", "fade"]:
                pr=proxy_eval(val, cond, sf, mode)
                if not pr or pr["signals"] < cfg.min_val_signals:
                    continue
                score=pr["mean_net_pct"]*np.sqrt(pr["signals"]) + 0.25*pr["t_like"] + 0.5*(pr["side_acc"]-0.5)
                proxy.append({"score": float(score), "condition": cond, "side_feature": sf, "mode": mode, "proxy": pr})
    proxy.sort(key=lambda r: r["score"], reverse=True)
    Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
    bt_rows=[]
    for i, rule in enumerate(proxy[:cfg.top_proxy]):
        res=backtest(val, cfg, f"val_{i:03d}", rule)
        score=float(res["sim"]["cagr_to_strict_mdd"])
        if int(res["sim"]["trade_entries"]) < cfg.min_val_trades:
            score -= 1000.0
        bt_rows.append({"score": score, **res})
    bt_rows.sort(key=lambda r: r["score"], reverse=True)
    selected=bt_rows[0] if bt_rows else None
    eval_res=backtest(evg, cfg, "selected_eval", selected["rule"]) if selected else None
    report={
        "config": cfg.__dict__,
        "rows": {"fit": len(fit), "val": len(val), "eval": len(evg)},
        "conditions": len(conds),
        "proxy_candidates": len(proxy),
        "top_proxy": proxy[:20],
        "top_val_backtests": bt_rows[:cfg.top_backtest],
        "selected": selected,
        "eval": eval_res,
        "leakage_guard": "numeric regime thresholds from fit; validation selects rules; eval applied once",
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
    p.add_argument("--min-val-signals", type=int, default=Cfg.min_val_signals)
    p.add_argument("--min-val-trades", type=int, default=Cfg.min_val_trades)
    p.add_argument("--top-proxy", type=int, default=Cfg.top_proxy)
    p.add_argument("--top-backtest", type=int, default=Cfg.top_backtest)
    return Cfg(**vars(p.parse_args()))


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
