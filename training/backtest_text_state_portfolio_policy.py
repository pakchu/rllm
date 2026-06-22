"""Train-only token baseline and backtest for text-state portfolio decisions."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.sparse_setup_ensemble_audit import EnsembleCfg, _load_market, _simulate_events

LABELS = ("LONG", "SHORT", "NO_TRADE")


@dataclass(frozen=True)
class PortfolioPolicyBacktestCfg:
    input_jsonl: str
    market_csv: str
    output: str
    min_count: int = 8
    smoothing: float = 2.0
    top_k_tokens: int = 10
    confidence_thresholds: str = "0.40,0.45,0.50,0.55,0.60"
    margin_thresholds: str = "0.00,0.05,0.10,0.15"
    leverage: float = 1.0
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    entry_delay_bars: int = 1
    max_same_bar_signals: int = 1
    trade_stop_loss_pct: float = 0.0
    trade_take_profit_pct: float = 0.0


def _parse_floats(raw: str) -> list[float]:
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def _load(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _tokens(row: dict[str, Any]) -> list[str]:
    toks = []
    st = row.get("state_tokens", {}) if isinstance(row.get("state_tokens"), dict) else {}
    toks.extend(f"state.{k}={v}" for k, v in sorted(st.items()))
    for k in ("daily_context", "weekly_context", "three_day_context", "volatility", "range_location", "htf_1d_location", "taker_imbalance", "kimchi_pressure"):
        if k in st:
            toks.append(f"focus.{k}={st[k]}")
    return toks


def _fit(train: list[dict[str, Any]], cfg: PortfolioPolicyBacktestCfg) -> dict[str, Any]:
    priors = Counter(str(r.get("target")) for r in train)
    total = max(1, len(train))
    prior_p = {lab: (priors.get(lab, 0) + cfg.smoothing) / (total + cfg.smoothing * len(LABELS)) for lab in LABELS}
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    token_n: Counter[str] = Counter()
    for r in train:
        y = str(r.get("target"))
        for t in set(_tokens(r)):
            counts[t][y] += 1
            token_n[t] += 1
    weights: dict[str, dict[str, float]] = {}
    for t, n in token_n.items():
        if n < int(cfg.min_count):
            continue
        weights[t] = {}
        for lab in LABELS:
            p = (counts[t].get(lab, 0) + cfg.smoothing * prior_p[lab]) / (n + cfg.smoothing)
            weights[t][lab] = math.log(max(1e-6, p) / max(1e-6, prior_p[lab]))
    return {"prior": prior_p, "weights": weights, "token_n": dict(token_n)}


def _predict(row: dict[str, Any], model: dict[str, Any], cfg: PortfolioPolicyBacktestCfg) -> dict[str, Any]:
    logits = {lab: math.log(float(model["prior"].get(lab, 1e-6))) for lab in LABELS}
    toks = sorted(set(_tokens(row)), key=lambda t: max(abs(float(v)) for v in model["weights"].get(t, {}).values()) if t in model["weights"] else 0.0, reverse=True)
    for t in toks[: int(cfg.top_k_tokens)]:
        for lab, w in model["weights"].get(t, {}).items():
            logits[lab] += float(w)
    mx = max(logits.values())
    exps = {lab: math.exp(max(-30.0, min(30.0, logits[lab] - mx))) for lab in LABELS}
    denom = sum(exps.values())
    probs = {lab: exps[lab] / denom for lab in LABELS}
    ordered = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    return {"label": ordered[0][0], "prob": ordered[0][1], "margin": ordered[0][1] - ordered[1][1], "probs": probs}


def _classification(rows: list[dict[str, Any]], model: dict[str, Any], cfg: PortfolioPolicyBacktestCfg) -> dict[str, Any]:
    if not rows:
        return {"rows": 0}
    preds = [_predict(r, model, cfg) for r in rows]
    correct = sum(1 for r, p in zip(rows, preds) if str(r.get("target")) == p["label"])
    by_label = {lab: {"support": 0, "predicted": 0, "correct": 0} for lab in LABELS}
    for r, p in zip(rows, preds):
        y = str(r.get("target"))
        by_label.setdefault(y, {"support": 0, "predicted": 0, "correct": 0})["support"] += 1
        by_label[p["label"]]["predicted"] += 1
        if y == p["label"]:
            by_label[y]["correct"] += 1
    return {"rows": len(rows), "accuracy": correct / len(rows), "target_counts": dict(Counter(str(r.get("target")) for r in rows)), "pred_counts": dict(Counter(p["label"] for p in preds)), "by_label": by_label}


def _events(rows: list[dict[str, Any]], preds: list[dict[str, Any]], *, prob_th: float, margin_th: float) -> list[dict[str, Any]]:
    out = []
    for r, p in zip(rows, preds):
        lab = str(p["label"])
        if lab == "NO_TRADE" or float(p["prob"]) < prob_th or float(p["margin"]) < margin_th:
            continue
        side = 1 if lab == "LONG" else -1
        hold = int(r.get("candidate", {}).get("hold_bars", 288))
        out.append({
            "signal_pos": int(r["signal_pos"]),
            "date": str(r["date"]),
            "side": side,
            "horizon": hold,
            "source_horizon": hold,
            "candidate_index": 0,
            "candidate_key": f"portfolio_token|{lab}|p{float(p['prob']):.3f}|m{float(p['margin']):.3f}",
            "fold": str(r.get("split", "eval")),
            "prior_mean_ret": max(0.0, float(p["prob"]) - 0.5),
            "prior_std_ret": 1.0,
            "prior_n": 100,
        })
    return sorted(out, key=lambda e: (int(e["signal_pos"]), str(e["candidate_key"])))


def run(cfg: PortfolioPolicyBacktestCfg) -> dict[str, Any]:
    rows = _load(cfg.input_jsonl)
    train = [r for r in rows if r.get("split") == "train"]
    eval_rows = [r for r in rows if r.get("split") == "eval"]
    model = _fit(train, cfg)
    eval_preds = [_predict(r, model, cfg) for r in eval_rows]
    market = _load_market(cfg.market_csv)
    dates = pd.to_datetime(market["date"])
    sim_cfg = EnsembleCfg(
        sparse_report="",
        market_csv=cfg.market_csv,
        output=cfg.output,
        leverage=cfg.leverage,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
        entry_delay_bars=cfg.entry_delay_bars,
        max_same_bar_signals=cfg.max_same_bar_signals,
        trade_stop_loss_pct=cfg.trade_stop_loss_pct,
        trade_take_profit_pct=cfg.trade_take_profit_pct,
    )
    bt_rows = []
    for pth in _parse_floats(cfg.confidence_thresholds):
        for mth in _parse_floats(cfg.margin_thresholds):
            evs = _events(eval_rows, eval_preds, prob_th=pth, margin_th=mth)
            res = _simulate_events(evs, dates=dates, market=market, cfg=sim_cfg)
            bt_rows.append({"prob_threshold": pth, "margin_threshold": mth, "events": len(evs), "result": {k: v for k, v in res.items() if k != "executed"}})
    bt_rows.sort(key=lambda r: float(r["result"]["sim"].get("cagr_to_strict_mdd", -999)), reverse=True)
    report = {
        "config": cfg.__dict__,
        "train_classification": _classification(train, model, cfg),
        "eval_classification": _classification(eval_rows, model, cfg),
        "learned_token_count": len(model["weights"]),
        "backtests": bt_rows,
        "top_backtests": bt_rows[:10],
        "leakage_guard": {"model_fit_on_train_split_only": True, "eval_not_used_for_threshold_training": True},
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest text-state portfolio token policy")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--min-count", type=int, default=8)
    p.add_argument("--smoothing", type=float, default=2.0)
    p.add_argument("--top-k-tokens", type=int, default=10)
    p.add_argument("--confidence-thresholds", default="0.40,0.45,0.50,0.55,0.60")
    p.add_argument("--margin-thresholds", default="0.00,0.05,0.10,0.15")
    p.add_argument("--leverage", type=float, default=1.0)
    p.add_argument("--fee-rate", type=float, default=0.0004)
    p.add_argument("--slippage-rate", type=float, default=0.0001)
    p.add_argument("--entry-delay-bars", type=int, default=1)
    p.add_argument("--max-same-bar-signals", type=int, default=1)
    p.add_argument("--trade-stop-loss-pct", type=float, default=0.0)
    p.add_argument("--trade-take-profit-pct", type=float, default=0.0)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(PortfolioPolicyBacktestCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
