"""Backtest binary-edge A/B score margins as candidate selectors."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.sparse_setup_ensemble_audit import EnsembleCfg, _load_market, _simulate_events


@dataclass(frozen=True)
class BinaryEdgeScoreBacktestCfg:
    predictions_jsonl: str
    market_csv: str
    output: str
    quantiles: str = "0.80,0.85,0.90,0.95"
    thresholds: str = ""
    leverage: float = 1.0
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    entry_delay_bars: int = 1
    max_same_bar_signals: int = 1
    trade_stop_loss_pct: float = 0.0
    trade_take_profit_pct: float = 0.0
    cooldown_bars: int = 0


def _load_predictions(path: str) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    for row in rows:
        scores = row.get("scores") or {}
        row["margin"] = float(scores.get("A", -1e9)) - float(scores.get("B", -1e9))
        source = row.get("source") if isinstance(row.get("source"), dict) else {}
        cand = row.get("candidate") if isinstance(row.get("candidate"), dict) else source.get("candidate", {})
        row["_side"] = str(row.get("side") or cand.get("side") or source.get("side") or "").upper()
        row["_hold"] = int(row.get("hold_bars") or cand.get("hold_bars") or cand.get("horizon") or 0)
        row["_signal_pos"] = int(row.get("signal_pos") or source.get("signal_pos") or -1)
    return rows


def _parse_floats(raw: str) -> list[float]:
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def _thresholds(rows: list[dict[str, Any]], cfg: BinaryEdgeScoreBacktestCfg) -> list[dict[str, float]]:
    margins = np.asarray([float(r["margin"]) for r in rows], dtype=float)
    out: list[dict[str, float]] = []
    for q in _parse_floats(cfg.quantiles):
        out.append({"kind": "quantile", "value": q, "threshold": float(np.quantile(margins, q))})
    for th in _parse_floats(cfg.thresholds):
        out.append({"kind": "threshold", "value": th, "threshold": th})
    # stable unique thresholds
    seen: set[float] = set()
    uniq: list[dict[str, float]] = []
    for row in out:
        key = round(float(row["threshold"]), 12)
        if key not in seen:
            seen.add(key); uniq.append(row)
    return uniq


def _event(row: dict[str, Any]) -> dict[str, Any]:
    side = 1 if row["_side"] == "LONG" else -1
    return {
        "signal_pos": int(row["_signal_pos"]),
        "date": str(row.get("date")),
        "side": side,
        "horizon": int(row["_hold"]),
        "source_horizon": int(row["_hold"]),
        "candidate_index": 0,
        "candidate_key": f"binary_edge|{row['_side']}|h{row['_hold']}|m{float(row['margin']):.4f}",
        "fold": "eval",
        "prior_mean_ret": max(0.0, float(row.get("margin", 0.0))),
        "prior_std_ret": 1.0,
        "prior_n": 100,
        "score": float(row["margin"]),
    }


def _dedupe(events: list[dict[str, Any]], max_same_bar: int) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for ev in events:
        grouped.setdefault(int(ev["signal_pos"]), []).append(ev)
    out: list[dict[str, Any]] = []
    for pos in sorted(grouped):
        out.extend(sorted(grouped[pos], key=lambda e: float(e.get("score", 0.0)), reverse=True)[: max(1, int(max_same_bar))])
    return out


def run(cfg: BinaryEdgeScoreBacktestCfg) -> dict[str, Any]:
    rows = _load_predictions(cfg.predictions_jsonl)
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
        cooldown_bars=cfg.cooldown_bars,
    )
    results = []
    for spec in _thresholds(rows, cfg):
        selected = [r for r in rows if float(r["margin"]) >= float(spec["threshold"]) and int(r["_signal_pos"]) >= 0 and int(r["_hold"]) > 0 and r["_side"] in {"LONG", "SHORT"}]
        events = _dedupe([_event(r) for r in selected], int(cfg.max_same_bar_signals))
        sim = _simulate_events(events, dates=dates, market=market, cfg=sim_cfg)
        utilities = [float((r.get("choice_utility") or {}).get("A", 0.0)) for r in selected]
        results.append(
            {
                **spec,
                "selected_rows": len(selected),
                "events": len(events),
                "label_edge_rate": sum(str(r.get("target")) == "A" for r in selected) / max(1, len(selected)),
                "mean_label_utility": float(np.mean(utilities)) if utilities else 0.0,
                "sim": sim.get("sim", {}),
                "trade_stats": sim.get("trade_stats", {}),
            }
        )
    report = {"config": asdict(cfg), "rows": len(rows), "results": results}
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions-jsonl", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--quantiles", default=BinaryEdgeScoreBacktestCfg.quantiles)
    p.add_argument("--thresholds", default=BinaryEdgeScoreBacktestCfg.thresholds)
    p.add_argument("--leverage", type=float, default=BinaryEdgeScoreBacktestCfg.leverage)
    p.add_argument("--fee-rate", type=float, default=BinaryEdgeScoreBacktestCfg.fee_rate)
    p.add_argument("--slippage-rate", type=float, default=BinaryEdgeScoreBacktestCfg.slippage_rate)
    p.add_argument("--entry-delay-bars", type=int, default=BinaryEdgeScoreBacktestCfg.entry_delay_bars)
    p.add_argument("--max-same-bar-signals", type=int, default=BinaryEdgeScoreBacktestCfg.max_same_bar_signals)
    p.add_argument("--trade-stop-loss-pct", type=float, default=BinaryEdgeScoreBacktestCfg.trade_stop_loss_pct)
    p.add_argument("--trade-take-profit-pct", type=float, default=BinaryEdgeScoreBacktestCfg.trade_take_profit_pct)
    p.add_argument("--cooldown-bars", type=int, default=BinaryEdgeScoreBacktestCfg.cooldown_bars)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(BinaryEdgeScoreBacktestCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
