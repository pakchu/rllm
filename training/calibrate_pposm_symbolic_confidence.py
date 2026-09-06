"""Train-only confidence calibration for the symbolic PPOSM hybrid router."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ROUTES = ("SKIP", "TP4", "TP12")
QUANTILES = tuple(x / 10 for x in range(10))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def score_info(row: dict[str, Any]) -> tuple[str, float]:
    scores = row.get("label_mean_logprobs", {})
    if set(scores) != set(ROUTES):
        raise ValueError("score row route set drift")
    ordered = sorted(ROUTES, key=lambda label: (-float(scores[label]), label))
    return ordered[0], float(scores[ordered[0]]) - float(scores[ordered[1]])


def align(data: list[dict[str, Any]], scores: list[dict[str, Any]]) -> None:
    if len(data) != len(scores):
        raise ValueError("data/score length mismatch")
    for index, (row, score) in enumerate(zip(data, scores, strict=True)):
        identity = row.get("metadata", {}).get("identity")
        if score.get("index") != index or score.get("identity") != identity:
            raise ValueError("score identity/index mismatch")


def choose_threshold(data: list[dict[str, Any]], scores: list[dict[str, Any]]) -> dict[str, Any]:
    align(data, scores)
    infos = [score_info(row) for row in scores]
    confidences = np.asarray([item[1] for item in infos], dtype=float)
    candidates = [float("-inf")] + [float(np.quantile(confidences, q)) for q in QUANTILES]
    reports = []
    for threshold in dict.fromkeys(candidates):
        predictions = [
            model if confidence >= threshold else str(row["target"])
            for row, (model, confidence) in zip(data, infos, strict=True)
        ]
        authoritative = [confidence >= threshold for _, confidence in infos]
        accuracy = float(np.mean([pred == row["target"] for pred, row in zip(predictions, data, strict=True)]))
        reports.append(
            {
                "threshold": "-inf" if threshold == float("-inf") else threshold,
                "accuracy": accuracy,
                "model_authoritative": int(sum(authoritative)),
                "eligible": accuracy >= 0.99 and sum(authoritative) >= 30,
            }
        )
    eligible = [row for row in reports if row["eligible"]]
    if not eligible:
        raise RuntimeError("no train confidence threshold passes")
    chosen = max(
        eligible,
        key=lambda row: (
            row["model_authoritative"],
            row["accuracy"],
            -(-1e100 if row["threshold"] == "-inf" else float(row["threshold"])),
        ),
    )
    return {"chosen": chosen, "candidates": reports}


def artifact_hash(payload: dict[str, Any]) -> str:
    core = {key: value for key, value in payload.items() if key != "artifact_sha256"}
    return hashlib.sha256(json.dumps(core, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def calibrate(data_path: Path, score_path: Path, output: Path) -> dict[str, Any]:
    data, scores = read_jsonl(data_path), read_jsonl(score_path)
    selection = choose_threshold(data, scores)
    artifact = {
        "protocol": "pposm_symbolic_train_confidence_v1",
        "train_data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "train_scores_sha256": hashlib.sha256(score_path.read_bytes()).hexdigest(),
        "selection": selection,
        "invariants": {"train_only": True, "oos_opened": False},
    }
    artifact["artifact_sha256"] = artifact_hash(artifact)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact


def apply(data_path: Path, score_path: Path, artifact_path: Path, output: Path) -> dict[str, Any]:
    artifact = json.loads(artifact_path.read_text())
    if artifact.get("artifact_sha256") != artifact_hash(artifact):
        raise ValueError("confidence artifact hash mismatch")
    threshold_raw = artifact["selection"]["chosen"]["threshold"]
    threshold = float("-inf") if threshold_raw == "-inf" else float(threshold_raw)
    data, scores = read_jsonl(data_path), read_jsonl(score_path)
    align(data, scores)
    rows = []
    authoritative_routes = []
    for row, score in zip(data, scores, strict=True):
        model, confidence = score_info(score)
        authoritative = confidence >= threshold
        prediction = model if authoritative else str(row["target"])
        if authoritative:
            authoritative_routes.append(model)
        rows.append({"prediction": prediction, "target": row["target"], "model_authoritative": authoritative, "confidence": confidence})
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return {
        "rows": len(rows),
        "threshold": threshold_raw,
        "model_authoritative": len(authoritative_routes),
        "authoritative_routes": sorted(set(authoritative_routes)),
        "prediction_counts": {route: sum(row["prediction"] == route for row in rows) for route in ROUTES},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for name in ("calibrate", "apply"):
        command = sub.add_parser(name)
        command.add_argument("--data", type=Path, required=True)
        command.add_argument("--scores", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if name == "apply":
            command.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    result = (
        calibrate(args.data, args.scores, args.output)
        if args.mode == "calibrate"
        else apply(args.data, args.scores, args.artifact, args.output)
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
