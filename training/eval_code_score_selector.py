"""Evaluate candidate selection quality from neutral-code label scores."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

LABELS = ("Q1", "Q2", "Q3", "Q4")
RANK_VALUE = {"Q1": -1.0, "Q2": 0.0, "Q3": 0.5, "Q4": 1.0}


@dataclass(frozen=True)
class CodeScoreSelectorConfig:
    scored_json: str
    output: str
    score_key: str = "mean"
    center_means_json: str = ""
    selector: str = "expected_rank"


def _rows_from_report(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text())
    rows = payload.get("score_rows")
    if not isinstance(rows, list):
        raise ValueError(f"{path} does not contain score_rows list; regenerate audit output")
    return rows


def _utility(row: dict[str, Any]) -> float:
    audit = row.get("action_audit", {}) if isinstance(row.get("action_audit"), dict) else {}
    val = audit.get("rank_utility", audit.get("utility", 0.0))
    return 0.0 if val is None else float(val)


def _score(row: dict[str, Any], label: str, key: str, centers: dict[str, float]) -> float:
    raw = float(row.get("score", {}).get(label, {}).get(key, 0.0) or 0.0)
    return raw - float(centers.get(label, 0.0))


def _selector_value(row: dict[str, Any], key: str, centers: dict[str, float], selector: str) -> float:
    scores = {label: _score(row, label, key, centers) for label in LABELS}
    if selector == "q4_minus_q2":
        return scores["Q4"] - scores["Q2"]
    if selector == "q4_minus_q1":
        return scores["Q4"] - scores["Q1"]
    if selector == "expected_rank":
        # Softmax over centered label scores, then ordinal expected value.
        import math

        m = max(scores.values())
        exps = {label: math.exp(max(-60.0, min(60.0, val - m))) for label, val in scores.items()}
        z = sum(exps.values()) or 1.0
        return sum((exps[label] / z) * RANK_VALUE[label] for label in LABELS)
    raise ValueError("selector must be one of {'expected_rank','q4_minus_q2','q4_minus_q1'}")


def _load_centers(path: str) -> dict[str, float]:
    if not path:
        return {}
    payload = json.loads(Path(path).read_text())
    if "train_label_means" in payload:
        return {str(k): float(v) for k, v in payload["train_label_means"].items()}
    if "summary" in payload and "mean_score_by_label" in payload["summary"]:
        return {str(k): float(v) for k, v in payload["summary"]["mean_score_by_label"].items()}
    return {str(k): float(v) for k, v in payload.items() if str(k) in LABELS}


def evaluate(rows: list[dict[str, Any]], cfg: CodeScoreSelectorConfig, centers: dict[str, float]) -> dict[str, Any]:
    by_signal: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("date")), int(row.get("signal_pos", -1) or -1))
        by_signal.setdefault(key, []).append(row)
    selected = []
    oracle = []
    first = []
    for key, group in sorted(by_signal.items(), key=lambda kv: kv[0]):
        valid = [r for r in group if isinstance(r.get("action_audit"), dict)]
        if not valid:
            continue
        best_model = max(valid, key=lambda r: _selector_value(r, cfg.score_key, centers, cfg.selector))
        best_oracle = max(valid, key=_utility)
        selected.append(_utility(best_model))
        oracle.append(_utility(best_oracle))
        first.append(_utility(valid[0]))
    def stats(xs: list[float]) -> dict[str, float]:
        return {"n": len(xs), "mean": sum(xs) / max(1, len(xs)), "positive_frac": sum(1 for x in xs if x > 0) / max(1, len(xs)), "min": min(xs) if xs else 0.0, "max": max(xs) if xs else 0.0}
    return {
        "signals": len(selected),
        "selected_utility": stats(selected),
        "oracle_utility": stats(oracle),
        "first_candidate_utility": stats(first),
        "selected_minus_first_mean": (sum(selected) - sum(first)) / max(1, len(selected)),
        "oracle_gap_mean": (sum(oracle) - sum(selected)) / max(1, len(selected)),
    }


def run(cfg: CodeScoreSelectorConfig) -> dict[str, Any]:
    centers = _load_centers(cfg.center_means_json)
    rows = _rows_from_report(cfg.scored_json)
    report = {"config": asdict(cfg), "centers": centers, "metrics": evaluate(rows, cfg, centers), "leakage_guard": {"centers_loaded_from_external_file": bool(cfg.center_means_json), "does_not_use_future_for_selection_score": True, "oracle_used_for_diagnostic_only": True}}
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Q-code score candidate selector")
    p.add_argument("--scored-json", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--score-key", choices=["mean", "sum"], default="mean")
    p.add_argument("--center-means-json", default="")
    p.add_argument("--selector", choices=["expected_rank", "q4_minus_q2", "q4_minus_q1"], default="expected_rank")
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(CodeScoreSelectorConfig(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
