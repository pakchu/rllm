"""Backtest selected candidates from pairwise option rows."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from training.sparse_setup_ensemble_audit import EnsembleCfg, _load_market, _simulate_events


@dataclass(frozen=True)
class PairwiseOptionOracleBacktestCfg:
    pairs_jsonl: str
    market_csv: str
    output: str
    selector: str = "target"  # target | prediction
    predictions_jsonl: str = ""
    margin_quantile: float = 0.0
    leverage: float = 1.0
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    entry_delay_bars: int = 1
    max_same_bar_signals: int = 1


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _event_from_candidate(row: dict[str, Any], cand: dict[str, Any], score: float) -> dict[str, Any] | None:
    source = cand or {}
    c = source.get("candidate") if isinstance(source.get("candidate"), dict) else {}
    side_txt = str(c.get("side", source.get("side", ""))).upper()
    side = 1 if side_txt == "LONG" else -1 if side_txt == "SHORT" else 0
    hold = int(c.get("hold_bars", c.get("horizon", 0)) or 0)
    pos = int(source.get("signal_pos", -1) or -1)
    if side == 0 or hold <= 0 or pos < 0:
        return None
    return {
        "signal_pos": pos,
        "date": str(source.get("date", row.get("date"))),
        "side": side,
        "horizon": hold,
        "source_horizon": hold,
        "candidate_index": 0,
        "candidate_key": f"pairwise_oracle|{side_txt}|h{hold}|s{score:.4f}",
        "fold": str(row.get("month", "eval")),
        "prior_mean_ret": max(0.0, score),
        "prior_std_ret": 1.0,
        "prior_n": 100,
        "score": score,
    }


def _dedupe(events: list[dict[str, Any]], max_same_bar: int) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for ev in events:
        grouped.setdefault(int(ev["signal_pos"]), []).append(ev)
    out: list[dict[str, Any]] = []
    for pos in sorted(grouped):
        out.extend(sorted(grouped[pos], key=lambda e: float(e.get("score", 0.0)), reverse=True)[: max(1, int(max_same_bar))])
    return out


def _prediction_rows(path: str) -> list[dict[str, Any]]:
    return _load_jsonl(path) if path else []


def _prediction_confidence(row: dict[str, Any]) -> tuple[str, float]:
    pred = str(row.get("prediction", "A"))
    scores = row.get("scores") or {}
    margin = float(scores.get("A", 0.0)) - float(scores.get("B", 0.0))
    return pred, abs(margin)


def run(cfg: PairwiseOptionOracleBacktestCfg) -> dict[str, Any]:
    rows = _load_jsonl(cfg.pairs_jsonl)
    pred_rows = _prediction_rows(cfg.predictions_jsonl)
    margins = sorted(_prediction_confidence(r)[1] for r in pred_rows) if pred_rows else []
    margin_th = margins[int(float(cfg.margin_quantile) * (len(margins) - 1))] if margins and float(cfg.margin_quantile) > 0 else 0.0
    selected = []
    iterable = pred_rows if cfg.selector == "prediction" else rows
    for row in iterable:
        if cfg.selector == "target":
            choice = str(row.get("target", "A"))
            confidence = float(row.get("utility_gap", 0.0))
        elif cfg.selector == "prediction":
            choice, confidence = _prediction_confidence(row)
            if not choice or confidence < margin_th:
                continue
        else:
            raise ValueError("selector must be target or prediction")
        cand = (row.get("candidates") or {}).get(choice)
        ev = _event_from_candidate(row, cand, confidence)
        if ev is not None:
            selected.append(ev)
    events = _dedupe(selected, int(cfg.max_same_bar_signals))
    market = _load_market(cfg.market_csv)
    sim_cfg = EnsembleCfg(
        sparse_report="",
        market_csv=cfg.market_csv,
        output=cfg.output,
        leverage=cfg.leverage,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
        entry_delay_bars=cfg.entry_delay_bars,
        max_same_bar_signals=cfg.max_same_bar_signals,
    )
    bt = _simulate_events(events, dates=pd.to_datetime(market["date"]), market=market, cfg=sim_cfg)
    report = {
        "config": asdict(cfg),
        "rows": len(rows),
        "margin_threshold": margin_th,
        "selected_events_raw": len(selected),
        "events": len(events),
        "sim": bt.get("sim", {}),
        "trade_stats": bt.get("trade_stats", {}),
        "leakage_guard": {"target_selector_is_oracle_only": cfg.selector == "target", "prediction_selector_uses_model_scores_only": cfg.selector == "prediction"},
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs-jsonl", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--selector", choices=["target", "prediction"], default=PairwiseOptionOracleBacktestCfg.selector)
    p.add_argument("--predictions-jsonl", default="")
    p.add_argument("--margin-quantile", type=float, default=0.0)
    p.add_argument("--leverage", type=float, default=1.0)
    p.add_argument("--fee-rate", type=float, default=0.0004)
    p.add_argument("--slippage-rate", type=float, default=0.0001)
    p.add_argument("--entry-delay-bars", type=int, default=1)
    p.add_argument("--max-same-bar-signals", type=int, default=1)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(PairwiseOptionOracleBacktestCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
