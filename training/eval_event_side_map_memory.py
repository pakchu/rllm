"""Evaluate event-level side-map reliability memory baselines and replay them."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.apply_side_map_memory_predictions import NO_TRADE, _write_jsonl
from training.nested_score_geometry_transform_selection import _invert_prediction
from training.online_risk_overlay_backtest import OnlineRiskOverlayConfig, run_overlay


@dataclass(frozen=True)
class EventSideMapMemoryEvalCfg:
    dataset_jsonl: str
    predictions_jsonl: str
    output: str
    transformed_predictions_jsonl: str = ""
    market_csv: str = ""
    backtest_output: str = ""
    train_splits: str = "train,val"
    eval_split: str = "eval"
    method: str = "token_signature"
    trade_take_profit_pct: float = 3.0
    leverage: float = 0.5
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    entry_delay_bars: int = 1


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _target_label(row: dict[str, Any]) -> str:
    try:
        return str(json.loads(row.get("target", "{}")) .get("side_map", "unreliable"))
    except Exception:
        return "unreliable"


def _state(row: dict[str, Any]) -> dict[str, str]:
    return row.get("state_tokens") if isinstance(row.get("state_tokens"), dict) else {}


def _score(row: dict[str, Any]) -> dict[str, str]:
    return row.get("score_tokens") if isinstance(row.get("score_tokens"), dict) else {}


def _signature(row: dict[str, Any], method: str) -> str:
    st = _state(row); sc = _score(row)
    side = str(row.get("generated_side", "NONE"))
    if method == "score_only":
        keys = [f"side={side}"] + [f"{k}={sc.get(k, 'unknown')}" for k in sorted(sc)]
    elif method == "core_state":
        use = ("trend_alignment", "risk_state", "pa_event_pressure", "pa_long_window_event")
        keys = [f"side={side}"] + [f"{k}={st.get(k, 'unknown')}" for k in use]
    elif method == "token_signature":
        use = ("trend_alignment", "risk_state", "pa_event_pressure", "pa_long_window_event")
        keys = [f"side={side}"] + [f"{k}={sc.get(k, 'unknown')}" for k in sorted(sc)] + [f"{k}={st.get(k, 'unknown')}" for k in use]
    else:
        raise ValueError(f"unknown method: {method}")
    return "|".join(keys)


def _majority(labels: list[str], default: str = "unreliable") -> str:
    if not labels:
        return default
    c = Counter(labels)
    # Prefer unreliable on ties for risk control, then inverse/normal lexical order is not used.
    order = {"unreliable": 2, "inverse": 1, "normal": 0}
    return sorted(c.items(), key=lambda kv: (kv[1], order.get(kv[0], -1)), reverse=True)[0][0]


def _key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row.get("date")), int(row.get("signal_pos", -1) or -1)


def _apply_prediction(row: dict[str, Any], label: str) -> dict[str, Any]:
    nr = dict(row)
    pred = dict(row.get("prediction", {})) if isinstance(row.get("prediction"), dict) else {}
    if label == "normal":
        nr["prediction"] = pred
    elif label == "inverse":
        nr["prediction"] = _invert_prediction(pred)
    else:
        nr["prediction"] = dict(NO_TRADE)
    nr["event_side_map_memory_prediction"] = label
    return nr


def run(cfg: EventSideMapMemoryEvalCfg) -> dict[str, Any]:
    rows = _read_jsonl(cfg.dataset_jsonl)
    train_splits = {s.strip() for s in cfg.train_splits.split(",") if s.strip()}
    train = [r for r in rows if str(r.get("split")) in train_splits]
    eval_rows = [r for r in rows if str(r.get("split")) == cfg.eval_split]
    global_label = _majority([_target_label(r) for r in train], "unreliable")
    memory: dict[str, list[str]] = defaultdict(list)
    for r in train:
        memory[_signature(r, cfg.method)].append(_target_label(r))
    pred_rows = []
    correct = 0
    pred_by_key: dict[tuple[str, int], str] = {}
    for r in eval_rows:
        sig = _signature(r, cfg.method)
        pred = _majority(memory.get(sig, []), global_label)
        truth = _target_label(r)
        correct += int(pred == truth)
        pred_by_key[_key(r)] = pred
        pred_rows.append({"date": r.get("date"), "signal_pos": r.get("signal_pos"), "truth": truth, "prediction": pred, "signature_seen": sig in memory})
    report: dict[str, Any] = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "train_rows": len(train),
        "eval_rows": len(eval_rows),
        "accuracy": float(correct / len(eval_rows)) if eval_rows else 0.0,
        "correct": correct,
        "predictions": pred_rows,
        "prediction_counts": dict(Counter(p["prediction"] for p in pred_rows)),
        "truth_counts": dict(Counter(p["truth"] for p in pred_rows)),
        "leakage_guard": {"memory_built_from_train_splits_only": True, "eval_truth_not_used_for_prediction": True},
    }
    if cfg.transformed_predictions_jsonl:
        base = _read_jsonl(cfg.predictions_jsonl)
        eval_keys = set(pred_by_key)
        transformed = [_apply_prediction(r, pred_by_key[_key(r)]) for r in base if _key(r) in eval_keys]
        _write_jsonl(cfg.transformed_predictions_jsonl, transformed)
        report["transformed_rows"] = len(transformed)
        if cfg.market_csv and cfg.backtest_output:
            bt = run_overlay(OnlineRiskOverlayConfig(
                predictions_jsonl=cfg.transformed_predictions_jsonl,
                market_csv=cfg.market_csv,
                output=cfg.backtest_output,
                leverage=float(cfg.leverage),
                fee_rate=float(cfg.fee_rate),
                slippage_rate=float(cfg.slippage_rate),
                entry_delay_bars=int(cfg.entry_delay_bars),
                trade_take_profit_pct=float(cfg.trade_take_profit_pct),
            ))
            report["backtest"] = {"period": bt["period"], "sim": bt["sim"], "trade_stats": bt["trade_stats"]}
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate event side-map memory baseline")
    p.add_argument("--dataset-jsonl", required=True)
    p.add_argument("--predictions-jsonl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--transformed-predictions-jsonl", default="")
    p.add_argument("--market-csv", default="")
    p.add_argument("--backtest-output", default="")
    p.add_argument("--train-splits", default=EventSideMapMemoryEvalCfg.train_splits)
    p.add_argument("--eval-split", default=EventSideMapMemoryEvalCfg.eval_split)
    p.add_argument("--method", default=EventSideMapMemoryEvalCfg.method)
    p.add_argument("--trade-take-profit-pct", type=float, default=EventSideMapMemoryEvalCfg.trade_take_profit_pct)
    return p.parse_args()


def main() -> None:
    report = run(EventSideMapMemoryEvalCfg(**vars(parse_args())))
    sim = report.get("backtest", {}).get("sim", {})
    print(json.dumps({"method": report["config"]["method"], "accuracy": report["accuracy"], "prediction_counts": report["prediction_counts"], "truth_counts": report["truth_counts"], "sim": sim}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
