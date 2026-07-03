"""Sweep month-level abstention rules from first in-month signal features.

A rule is live-usable if it only reads the first candidate row available in a
calendar month.  If the rule matches, all TRADE predictions in that month are
changed to NO_TRADE.  Rules are ranked on train only; test/eval are replayed
after selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from training.online_risk_overlay_backtest import OnlineRiskOverlayConfig, run_overlay
from training.sweep_conjunctive_event_gates import load_rows


def _month(date: Any) -> str:
    return str(date)[:7]


def month_first_features(candidate_jsonl: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for row in sorted(load_rows(candidate_jsonl), key=lambda r: (str(r.get("date")), int(r.get("signal_pos", -1) or -1))):
        m = _month(row.get("date"))
        if m in out:
            continue
        fs = row.get("_fs", {}) if isinstance(row.get("_fs"), dict) else {}
        out[m] = {k: float(v) for k, v in fs.items() if isinstance(v, (int, float)) and np.isfinite(float(v))}
    return out


def _read_preds(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def apply_month_rule(predictions_jsonl: str, month_features: dict[str, dict[str, float]], *, feature: str, op: str, threshold: float, output: str) -> dict[str, Any]:
    out = []
    blocked_months: set[str] = set()
    blocked_trades = 0
    for row in _read_preds(predictions_jsonl):
        m = _month(row.get("date"))
        fs = month_features.get(m, {})
        val = fs.get(feature)
        block = isinstance(val, (int, float)) and (float(val) >= float(threshold) if op == ">=" else float(val) <= float(threshold))
        pred = row.get("prediction", {}) if isinstance(row.get("prediction"), dict) else {}
        if block:
            blocked_months.add(m)
        if block and pred.get("gate") == "TRADE":
            row = {**row, "blocked_prediction": pred, "prediction": {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0, "family": "MONTH_FEATURE_ABSTAIN", "confidence": "HIGH"}}
            blocked_trades += 1
        out.append(row)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in out) + "\n")
    return {"output": output, "blocked_months": sorted(blocked_months), "blocked_trade_rows": blocked_trades, "rows": len(out)}


def _score(sim: dict[str, Any], stats: dict[str, Any]) -> float:
    return float(sim["cagr_to_strict_mdd"]) + 0.04 * float(sim["cagr_pct"]) + 0.001 * float(sim["trade_entries"]) - 0.2 * max(0.0, float(sim["strict_mdd_pct"]) - 12.0) - 0.25 * max(0.0, float(stats.get("p_value_mean_ret_approx", 1.0)) - 0.1)


def _bt(predictions: str, market_csv: str, output: str) -> dict[str, Any]:
    return run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=predictions, market_csv=market_csv, output=output, leverage=0.5, max_hold_bars=144))


def sweep_month_feature_gate(*, train_candidate_jsonl: str, test_candidate_jsonl: str, eval_candidate_jsonl: str, train_predictions: str, test_predictions: str, eval_predictions: str, market_csv: str, output: str, work_dir: str) -> dict[str, Any]:
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    train_mf = month_first_features(train_candidate_jsonl)
    test_mf = month_first_features(test_candidate_jsonl)
    eval_mf = month_first_features(eval_candidate_jsonl)
    features = sorted({k for fs in train_mf.values() for k, v in fs.items() if isinstance(v, (int, float)) and np.isfinite(float(v))})
    candidates = []
    for feature in features:
        vals = np.array([fs[feature] for fs in train_mf.values() if feature in fs and np.isfinite(float(fs[feature]))], dtype=float)
        if vals.size < 12 or np.nanstd(vals) <= 1e-12:
            continue
        for q in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
            thr = float(np.quantile(vals, q))
            for op in [">=", "<="]:
                tag = f"{feature}_{op}_{q}".replace("/", "_").replace(">=", "ge").replace("<=", "le")
                pred = work / f"train_{tag}.jsonl"
                meta = apply_month_rule(train_predictions, train_mf, feature=feature, op=op, threshold=thr, output=str(pred))
                bt = _bt(str(pred), market_csv, str(work / f"train_{tag}.json"))
                sim = bt["sim"]
                if sim["trade_entries"] < 60 or sim["cagr_pct"] <= 0:
                    continue
                candidates.append({"feature": feature, "op": op, "threshold": thr, "quantile": q, "train_gate": meta, "train": {"sim": sim, "trade_stats": bt["trade_stats"]}, "selection_score": _score(sim, bt["trade_stats"])})
    candidates.sort(key=lambda r: r["selection_score"], reverse=True)
    top = candidates[:30]
    for i, row in enumerate(top):
        for split, cand_jsonl, preds, mf in [("test", test_candidate_jsonl, test_predictions, test_mf), ("eval", eval_candidate_jsonl, eval_predictions, eval_mf)]:
            pred = work / f"{split}_top{i:02d}.jsonl"
            meta = apply_month_rule(preds, mf, feature=row["feature"], op=row["op"], threshold=float(row["threshold"]), output=str(pred))
            bt = _bt(str(pred), market_csv, str(work / f"{split}_top{i:02d}.json"))
            row[f"{split}_gate"] = meta
            row[split] = {"sim": bt["sim"], "trade_stats": bt["trade_stats"]}
    report = {"leakage_guard": {"rules_ranked_on_train_only": True, "month_gate_uses_first_candidate_features_only": True, "test_eval_replayed_after_selection": True}, "feature_count": len(features), "candidate_count": len(candidates), "top": top}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep month feature abstention gates")
    p.add_argument("--train-candidate-jsonl", required=True)
    p.add_argument("--test-candidate-jsonl", required=True)
    p.add_argument("--eval-candidate-jsonl", required=True)
    p.add_argument("--train-predictions", required=True)
    p.add_argument("--test-predictions", required=True)
    p.add_argument("--eval-predictions", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--work-dir", required=True)
    return p.parse_args()


def main() -> None:
    print(json.dumps(sweep_month_feature_gate(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
