"""Sweep candidate-logprob trade margins from an existing model-policy eval report.

The input report must contain generated_preview score rows from
`eval_calibrated_policy_model --prediction-mode candidate_logprob`.  This helper
is intentionally lightweight: it summarizes the margin distribution and derives
candidate thresholds, but full strict metrics for each threshold should be run by
`eval_calibrated_policy_model` because previews are capped and do not contain all
records.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def extract_margins(report: dict[str, Any]) -> list[float]:
    margins: list[float] = []
    for item in report.get("generated_preview", []):
        scores = item.get("scores", [])
        if len(scores) < 2:
            continue
        no_score = float(scores[0]["mean_logprob"])
        trade_score = float(scores[1]["mean_logprob"])
        margins.append(trade_score - no_score)
    return margins


def summarize_margins(margins: list[float]) -> dict[str, Any]:
    if not margins:
        return {"count": 0}
    xs = sorted(margins)
    def q(p: float) -> float:
        idx = min(len(xs) - 1, max(0, round((len(xs) - 1) * p)))
        return xs[idx]
    return {
        "count": len(xs),
        "min": xs[0],
        "max": xs[-1],
        "mean": sum(xs) / len(xs),
        "p10": q(0.10),
        "p25": q(0.25),
        "p50": q(0.50),
        "p75": q(0.75),
        "p90": q(0.90),
    }


def candidate_margins(raw: str) -> list[float]:
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def run(*, input: str, output: str, margins: str = "-0.20,-0.15,-0.10,-0.05,0.0,0.05") -> dict[str, Any]:
    report = json.loads(Path(input).read_text())
    ms = extract_margins(report)
    out = {
        "input": str(Path(input).resolve()),
        "margin_summary_from_preview": summarize_margins(ms),
        "candidate_margins": candidate_margins(margins),
        "note": "Preview margins are capped; run eval_calibrated_policy_model for strict metrics at each selected margin.",
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize candidate-logprob margin candidates from eval preview")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--margins", default="-0.20,-0.15,-0.10,-0.05,0.0,0.05")
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
