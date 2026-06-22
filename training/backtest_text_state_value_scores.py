"""Backtest text-state value baseline scores as an action selector."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.eval_text_state_value_baseline import BaselineCfg, _fit, _load, _score
from training.sparse_setup_ensemble_audit import _load_market, _simulate_events, EnsembleCfg


@dataclass(frozen=True)
class TextStateScoreBacktestCfg:
    input_jsonl: str
    market_csv: str
    output: str
    split: str = "eval"
    quantiles: str = "0.80,0.90,0.95"
    min_gap: float = 0.0
    leverage: float = 1.0
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    entry_delay_bars: int = 1
    max_same_bar_signals: int = 1
    trade_stop_loss_pct: float = 0.0
    trade_take_profit_pct: float = 0.0


def _parse_floats(raw: str) -> list[float]:
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def _dates(market: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(market["date"])


def _event_from_row(row: dict[str, Any], score: float) -> dict[str, Any]:
    cand = row["candidate"]
    side = 1 if str(cand["side"]).upper() == "LONG" else -1
    return {
        "signal_pos": int(row["signal_pos"]),
        "date": str(row["date"]),
        "side": side,
        "horizon": int(cand["hold_bars"]),
        "source_horizon": int(cand["hold_bars"]),
        "candidate_index": 0,
        "candidate_key": f"text_state_score|{cand['side']}|h{cand['hold_bars']}|s{score:.4f}",
        "fold": str(row.get("split", "eval")),
        "prior_mean_ret": max(0.0, score - 0.5),
        "prior_std_ret": 1.0,
        "prior_n": 100,
        "score": float(score),
    }


def _dedupe_events(events: list[dict[str, Any]], *, max_same_bar: int) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for e in events:
        grouped.setdefault(int(e["signal_pos"]), []).append(e)
    out: list[dict[str, Any]] = []
    for pos in sorted(grouped):
        picks = sorted(grouped[pos], key=lambda e: float(e.get("score", 0.0)), reverse=True)[: max(1, int(max_same_bar))]
        out.extend(picks)
    return sorted(out, key=lambda e: (int(e["signal_pos"]), -float(e.get("score", 0.0))))


def run(cfg: TextStateScoreBacktestCfg) -> dict[str, Any]:
    rows = _load(cfg.input_jsonl)
    train = [r for r in rows if r.get("split") == "train"]
    target = [r for r in rows if r.get("split") == cfg.split]
    model = _fit(train, BaselineCfg(input_jsonl=cfg.input_jsonl, output=cfg.output))
    scored = [{"row": r, "score": _score(r, model, BaselineCfg(input_jsonl=cfg.input_jsonl, output=cfg.output))} for r in target]
    scores = np.asarray([x["score"] for x in scored], dtype=float)
    market = _load_market(cfg.market_csv)
    dates = _dates(market)
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
    rows_out = []
    for q in _parse_floats(cfg.quantiles):
        th = float(np.quantile(scores, q)) if len(scores) else 1.0
        events = [_event_from_row(x["row"], x["score"]) for x in scored if float(x["score"]) >= th and float(x["score"]) - th >= float(cfg.min_gap)]
        events = _dedupe_events(events, max_same_bar=int(cfg.max_same_bar_signals))
        res = _simulate_events(events, dates=dates, market=market, cfg=sim_cfg)
        rows_out.append({"quantile": q, "threshold": th, "events": len(events), "result": {k: v for k, v in res.items() if k != "executed"}})
    report = {"config": cfg.__dict__, "score_summary": {"rows": len(scored), "min": float(np.min(scores)) if len(scores) else 0.0, "max": float(np.max(scores)) if len(scores) else 0.0, "mean": float(np.mean(scores)) if len(scores) else 0.0}, "rows": rows_out, "leakage_guard": {"score_model_fit_on_train_split_only": True, "backtest_split": cfg.split}}
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest text-state value scores")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--split", default="eval")
    p.add_argument("--quantiles", default="0.80,0.90,0.95")
    p.add_argument("--min-gap", type=float, default=0.0)
    p.add_argument("--leverage", type=float, default=1.0)
    p.add_argument("--fee-rate", type=float, default=0.0004)
    p.add_argument("--slippage-rate", type=float, default=0.0001)
    p.add_argument("--entry-delay-bars", type=int, default=1)
    p.add_argument("--max-same-bar-signals", type=int, default=1)
    p.add_argument("--trade-stop-loss-pct", type=float, default=0.0)
    p.add_argument("--trade-take-profit-pct", type=float, default=0.0)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(TextStateScoreBacktestCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
