"""Freeze BATE-288 controls and strict accounting before outcome values."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from training import evaluate_block_arrival_throughput_elasticity_pre2024 as evaluate


DEFAULT_OUTPUT = str(evaluate.EVALUATION_FREEZE)


def canonical_hash(payload: Any) -> str:
    return evaluate.canonical_hash(payload)


def build_manifest(
    commit: str,
    *,
    control_clock_hashes: dict[str, str],
    control_clock_counts: dict[str, int],
    outcome_boundaries: dict[str, Any],
) -> dict[str, Any]:
    if len(commit) != 40:
        raise ValueError("evaluator source commit must be a full Git hash")
    if set(control_clock_hashes) != set(evaluate.POLICY_NAMES):
        raise ValueError("BATE-288 freeze lacks a control-clock hash")
    if set(control_clock_counts) != set(evaluate.POLICY_NAMES):
        raise ValueError("BATE-288 freeze lacks a control-clock count")
    for source in ("market", "funding"):
        if outcome_boundaries.get(source, {}).get("value_rows_parsed") != 0:
            raise ValueError("BATE-288 freeze boundary scan parsed outcome values")
    core: dict[str, Any] = {
        "protocol": "BATE-288 evaluator pre-outcome freeze v1",
        "outcomes_opened": False,
        "evaluation_source": str(evaluate.EVALUATION_SOURCE),
        "evaluation_source_sha256": evaluate.sha256_file(evaluate.EVALUATION_SOURCE),
        "freeze_source": str(evaluate.FREEZE_SOURCE),
        "freeze_source_sha256": evaluate.sha256_file(evaluate.FREEZE_SOURCE),
        "evaluation_source_commit": commit,
        "support_commit": evaluate.SUPPORT_COMMIT,
        "preregistration_sha256": evaluate.PREREGISTRATION_SHA256,
        "support_source_sha256": evaluate.SUPPORT_SOURCE_SHA256,
        "support_document_sha256": evaluate.SUPPORT_DOCUMENT_SHA256,
        "support_result_sha256": evaluate.SUPPORT_RESULT_SHA256,
        "primary_clock_sha256": evaluate.PRIMARY_CLOCK_SHA256,
        "block_source_sha256": evaluate.BLOCK_SOURCE_SHA256,
        "block_source_manifest_sha256": evaluate.BLOCK_SOURCE_MANIFEST_SHA256,
        "market_data_sha256": evaluate.MARKET_DATA_SHA256,
        "funding_data_sha256": evaluate.FUNDING_DATA_SHA256,
        "funding_manifest_sha256": evaluate.FUNDING_MANIFEST_SHA256,
        "opened_windows": [],
        "sealed_windows": [
            "train_2021_2022",
            "selection_2023",
            "2024",
            "2025",
            "2026_ytd",
        ],
        "mutable_parameters": [],
        "labels_constructed_during_freeze": False,
        "market_value_rows_parsed_during_freeze": 0,
        "funding_value_rows_parsed_during_freeze": 0,
        "execution_simulation_run_during_freeze": False,
        "timestamp_columns_scanned_during_freeze": [
            "market.date",
            "funding.funding_time_ms",
        ],
        "policy_names": list(evaluate.POLICY_NAMES),
        "control_semantics": evaluate.CONTROL_SEMANTICS,
        "evaluation_config": evaluate.asdict(evaluate.EvaluationConfig()),
        "control_clock_hashes": control_clock_hashes,
        "control_clock_counts": control_clock_counts,
        "outcome_boundaries": outcome_boundaries,
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate_manifest(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if canonical_hash(core) != payload.get("manifest_hash"):
        raise RuntimeError("BATE-288 evaluator freeze hash mismatch")
    if payload.get("outcomes_opened") is not False:
        raise RuntimeError("BATE-288 evaluator freeze cannot open outcomes")
    if payload.get("opened_windows") != []:
        raise RuntimeError("BATE-288 evaluator freeze opened a window")
    if payload.get("labels_constructed_during_freeze") is not False:
        raise RuntimeError("BATE-288 evaluator freeze constructed labels")
    if payload.get("market_value_rows_parsed_during_freeze") != 0:
        raise RuntimeError("BATE-288 evaluator freeze parsed market values")
    if payload.get("funding_value_rows_parsed_during_freeze") != 0:
        raise RuntimeError("BATE-288 evaluator freeze parsed funding values")
    if payload.get("execution_simulation_run_during_freeze") is not False:
        raise RuntimeError("BATE-288 evaluator freeze simulated execution")
    if payload.get("mutable_parameters") != []:
        raise RuntimeError("BATE-288 evaluator freeze permits mutable parameters")
    if payload.get("policy_names") != list(evaluate.POLICY_NAMES):
        raise RuntimeError("BATE-288 evaluator freeze control set changed")
    if payload.get("control_semantics") != evaluate.CONTROL_SEMANTICS:
        raise RuntimeError("BATE-288 evaluator freeze control semantics changed")
    if payload.get("evaluation_config") != evaluate.asdict(evaluate.EvaluationConfig()):
        raise RuntimeError("BATE-288 evaluator freeze configuration changed")
    for source in ("market", "funding"):
        if payload.get("outcome_boundaries", {}).get(source, {}).get(
            "value_rows_parsed"
        ) != 0:
            raise RuntimeError("BATE-288 freeze boundary scan parsed outcome values")


def write_once(path: str | Path, payload: dict[str, Any]) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        validate_manifest(existing)
        if existing != payload:
            raise RuntimeError("refusing to overwrite frozen BATE-288 evaluator")
        return "verified_existing"
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return "created"


def current_clean_commit() -> str:
    sources = [str(evaluate.EVALUATION_SOURCE), str(evaluate.FREEZE_SOURCE)]
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *sources], text=True
    ).strip()
    if status:
        raise RuntimeError("BATE-288 evaluator sources are not clean at HEAD")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    for source in sources:
        committed = subprocess.check_output(["git", "show", f"HEAD:{source}"])
        if hashlib.sha256(committed).hexdigest() != evaluate.sha256_file(source):
            raise RuntimeError(f"BATE-288 evaluator source is not reproducible: {source}")
    return commit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output != DEFAULT_OUTPUT:
        raise ValueError("BATE-288 evaluator freeze path is immutable")
    if evaluate.TRAIN_OUTPUT.exists() or evaluate.SELECTION_OUTPUT.exists():
        raise RuntimeError("BATE-288 outcome result already exists")
    controls, _ = evaluate.verify_support_and_control_clocks()
    clock_hashes = {
        name: evaluate._clock_hash(clock) for name, clock in controls.items()
    }
    clock_counts = {name: int(len(clock)) for name, clock in controls.items()}
    outcome_boundaries = evaluate.scan_outcome_boundaries()
    payload = build_manifest(
        current_clean_commit(),
        control_clock_hashes=clock_hashes,
        control_clock_counts=clock_counts,
        outcome_boundaries=outcome_boundaries,
    )
    validate_manifest(payload)
    status = write_once(args.output, payload)
    print(
        json.dumps(
            {
                "status": status,
                "outcomes_opened": False,
                "evaluation_source_commit": payload["evaluation_source_commit"],
                "evaluation_source_sha256": payload["evaluation_source_sha256"],
                "control_clock_counts": clock_counts,
                "outcome_boundaries": outcome_boundaries,
                "output": args.output,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
