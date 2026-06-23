"""Apply score-spread abstention to side-rationale eval JSON predictions."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ThresholdSideRationaleEvalCfg:
    input_json: str
    output_json: str
    min_spread: float
    abstain_label: str = "UNRELIABLE"


def _spread(scores: dict[str, Any]) -> float:
    return abs(float(scores.get("normal", 0.0)) - float(scores.get("inverse", 0.0)))


def run(cfg: ThresholdSideRationaleEvalCfg) -> dict[str, Any]:
    obj = json.loads(Path(cfg.input_json).read_text())
    out = dict(obj)
    preds = []
    counts: dict[str, int] = {}
    kept = 0
    correct = 0
    for pred in obj.get("predictions", []):
        nr = dict(pred)
        spread = _spread(dict(nr.get("scores", {})))
        nr["score_spread"] = spread
        if spread < float(cfg.min_spread):
            nr["pre_threshold_prediction"] = nr.get("prediction")
            nr["prediction"] = str(cfg.abstain_label).upper()
        else:
            kept += 1
            correct += int(str(nr.get("prediction", "")).upper() == str(nr.get("target", "")).upper())
        counts[str(nr.get("prediction", "")).upper()] = counts.get(str(nr.get("prediction", "")).upper(), 0) + 1
        preds.append(nr)
    out["predictions"] = preds
    out["threshold"] = {
        "config": asdict(cfg),
        "kept": kept,
        "abstained": len(preds) - kept,
        "kept_accuracy": correct / max(1, kept),
        "prediction_counts": dict(sorted(counts.items())),
    }
    Path(cfg.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output_json).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out["threshold"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Threshold side rationale eval predictions by score spread")
    p.add_argument("--input-json", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--min-spread", type=float, required=True)
    p.add_argument("--abstain-label", default="UNRELIABLE")
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(ThresholdSideRationaleEvalCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
