"""Sweep score-margin abstention thresholds for portfolio label predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

LABELS = ("LONG", "SHORT", "NO_TRADE")


def _load(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in open(path) if line.strip()]


def _margin(row: dict[str, Any]) -> float:
    scores = row.get("scores", {}) if isinstance(row.get("scores"), dict) else {}
    pred = str(row.get("prediction", "NO_TRADE"))
    if pred not in {"LONG", "SHORT"}:
        return 0.0
    pred_score = float(scores.get(pred, 0.0))
    alternatives = [float(scores.get(l, -1e9)) for l in LABELS if l != pred]
    return pred_score - max(alternatives)


def _write(rows: list[dict[str, Any]], path: str, threshold: float, invert: bool) -> dict[str, Any]:
    out=[]; counts={}
    for r in rows:
        rr=dict(r)
        raw=str(r.get("prediction", "NO_TRADE"))
        pred=raw
        if raw in {"LONG", "SHORT"} and _margin(r) < threshold:
            pred="NO_TRADE"
        if invert and pred == "LONG":
            pred="SHORT"
        elif invert and pred == "SHORT":
            pred="LONG"
        rr["raw_prediction"] = raw
        rr["score_margin"] = _margin(r)
        rr["prediction"] = pred
        counts[pred]=counts.get(pred,0)+1
        out.append(rr)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in out)+"\n")
    return {"path": path, "threshold": threshold, "invert": invert, "rows": len(out), "prediction_counts": dict(sorted(counts.items()))}


def run(input_jsonl: str, output_dir: str, thresholds: str, invert: bool) -> dict[str, Any]:
    rows=_load(input_jsonl)
    thrs=[float(x) for x in thresholds.split(',') if x.strip()]
    outs=[]
    for thr in thrs:
        tag=(f"m{thr:.4f}".replace('-','m').replace('.','p'))
        suffix="_invert" if invert else ""
        outs.append(_write(rows, str(Path(output_dir)/f"margin_{tag}{suffix}.jsonl"), thr, invert))
    return {"input": input_jsonl, "output_dir": output_dir, "thresholds": thrs, "invert": invert, "outputs": outs}


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description="Apply score-margin abstention sweep")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--thresholds", default="0,0.01,0.02,0.05,0.1,0.2,0.4,0.8")
    p.add_argument("--invert", action="store_true")
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
