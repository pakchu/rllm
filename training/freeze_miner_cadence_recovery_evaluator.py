"""Freeze MCR-7 controls and strict accounting before market outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from training import evaluate_miner_cadence_recovery_pre2024 as evaluate


DEFAULT_OUTPUT = str(evaluate.EVALUATION_FREEZE)


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_manifest(
    commit: str,
    *,
    control_clock_hashes: dict[str, str],
    control_clock_counts: dict[str, int],
) -> dict[str, Any]:
    if len(commit) != 40:
        raise ValueError("evaluator source commit must be a full Git hash")
    if set(control_clock_hashes) != set(evaluate.POLICY_NAMES):
        raise ValueError("MCR-7 freeze lacks a control-clock hash")
    if set(control_clock_counts) != set(evaluate.POLICY_NAMES):
        raise ValueError("MCR-7 freeze lacks a control-clock count")
    core: dict[str, Any] = {
        "protocol": "MCR-7 evaluator pre-outcome freeze v1",
        "outcomes_opened": False,
        "evaluation_source": str(evaluate.EVALUATION_SOURCE),
        "evaluation_source_sha256": _sha256(evaluate.EVALUATION_SOURCE),
        "evaluation_source_commit": commit,
        "support_commit": evaluate.SUPPORT_COMMIT,
        "preregistration_sha256": evaluate.PREREGISTRATION_SHA256,
        "support_source_sha256": evaluate.SUPPORT_SOURCE_SHA256,
        "support_document_sha256": evaluate.SUPPORT_DOCUMENT_SHA256,
        "support_result_sha256": evaluate.SUPPORT_RESULT_SHA256,
        "primary_clock_sha256": evaluate.PRIMARY_CLOCK_SHA256,
        "market_data_sha256": evaluate.MARKET_DATA_SHA256,
        "funding_data_sha256": evaluate.FUNDING_DATA_SHA256,
        "funding_manifest_sha256": evaluate.FUNDING_MANIFEST_SHA256,
        "opened_windows": [],
        "sealed_windows": [*evaluate.WINDOWS, "2024", "2025", "2026_ytd"],
        "mutable_parameters": [],
        "labels_constructed_during_freeze": False,
        "market_rows_parsed_during_freeze": 0,
        "funding_rows_loaded_during_freeze": 0,
        "execution_simulation_run_during_freeze": False,
        "policy_names": list(evaluate.POLICY_NAMES),
        "evaluation_config": evaluate.asdict(evaluate.EvaluationConfig()),
        "control_clock_hashes": control_clock_hashes,
        "control_clock_counts": control_clock_counts,
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate_manifest(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if canonical_hash(core) != payload.get("manifest_hash"):
        raise RuntimeError("MCR-7 evaluator freeze hash mismatch")
    if payload.get("outcomes_opened") is not False:
        raise RuntimeError("MCR-7 evaluator freeze cannot open outcomes")
    if payload.get("labels_constructed_during_freeze") is not False:
        raise RuntimeError("MCR-7 evaluator freeze constructed labels")
    if payload.get("market_rows_parsed_during_freeze") != 0:
        raise RuntimeError("MCR-7 evaluator freeze parsed market rows")
    if payload.get("funding_rows_loaded_during_freeze") != 0:
        raise RuntimeError("MCR-7 evaluator freeze loaded funding rows")
    if payload.get("execution_simulation_run_during_freeze") is not False:
        raise RuntimeError("MCR-7 evaluator freeze simulated execution")
    if payload.get("mutable_parameters") != []:
        raise RuntimeError("MCR-7 evaluator freeze permits mutable parameters")
    if payload.get("policy_names") != list(evaluate.POLICY_NAMES):
        raise RuntimeError("MCR-7 evaluator freeze control set changed")
    if payload.get("evaluation_config") != evaluate.asdict(evaluate.EvaluationConfig()):
        raise RuntimeError("MCR-7 evaluator freeze configuration changed")


def write_once(path: str | Path, payload: dict[str, Any]) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = json.loads(output.read_text())
        validate_manifest(existing)
        if existing != payload:
            raise RuntimeError("refusing to overwrite frozen MCR-7 evaluator")
        return "verified_existing"
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return "created"


def current_clean_commit() -> str:
    source = str(evaluate.EVALUATION_SOURCE)
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", source], text=True
    ).strip()
    if status:
        raise RuntimeError("MCR-7 evaluator source is not clean at HEAD")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    committed = subprocess.check_output(["git", "show", f"HEAD:{source}"])
    if hashlib.sha256(committed).hexdigest() != _sha256(evaluate.EVALUATION_SOURCE):
        raise RuntimeError("MCR-7 evaluator source is not reproducible from HEAD")
    return commit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output != DEFAULT_OUTPUT:
        raise ValueError("MCR-7 evaluator freeze path is immutable")
    if Path(evaluate.DEFAULT_OUTPUT).exists():
        raise RuntimeError("MCR-7 outcome result already exists")
    clocks, _, _ = evaluate.verify_support_and_control_clocks()
    clock_hashes = {name: evaluate._clock_hash(clock) for name, clock in clocks.items()}
    clock_counts = {name: int(len(clock)) for name, clock in clocks.items()}
    payload = build_manifest(
        current_clean_commit(),
        control_clock_hashes=clock_hashes,
        control_clock_counts=clock_counts,
    )
    status = write_once(args.output, payload)
    print(
        json.dumps(
            {
                "status": status,
                "outcomes_opened": False,
                "evaluation_source_commit": payload["evaluation_source_commit"],
                "evaluation_source_sha256": payload["evaluation_source_sha256"],
                "control_clock_counts": clock_counts,
                "output": args.output,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
