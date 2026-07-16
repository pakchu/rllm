"""Freeze DCRM-1's strict evaluator before opening 2023 outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import evaluate_dispersion_conditioned_residual_momentum_2023 as evaluate


DEFAULT_OUTPUT = str(evaluate.EVALUATION_FREEZE)


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_manifest(commit: str, *, clock_rows: int) -> dict[str, Any]:
    if len(commit) != 40:
        raise ValueError("evaluator source commit must be a full Git hash")
    core: dict[str, Any] = {
        "protocol": "DCRM-1 strict 2023 evaluator pre-outcome freeze v1",
        "outcomes_opened": False,
        "evaluation_source": str(evaluate.EVALUATION_SOURCE),
        "evaluation_source_sha256": _sha256(evaluate.EVALUATION_SOURCE),
        "evaluation_source_commit": commit,
        "test_path": str(evaluate.TEST_PATH),
        "test_sha256": _sha256(evaluate.TEST_PATH),
        "freeze_source": str(evaluate.FREEZE_SOURCE),
        "freeze_source_sha256": _sha256(evaluate.FREEZE_SOURCE),
        "freeze_test_path": str(evaluate.FREEZE_TEST_PATH),
        "freeze_test_sha256": _sha256(evaluate.FREEZE_TEST_PATH),
        "preregistration_sha256": evaluate.PREREGISTRATION_SHA256,
        "support_manifest_sha256": evaluate.SUPPORT_MANIFEST_SHA256,
        "support_manifest_hash": evaluate.SUPPORT_MANIFEST_HASH,
        "clock_sha256": evaluate.CLOCK_SHA256,
        "clock_rows": int(clock_rows),
        "support_source_sha256": evaluate.SUPPORT_SOURCE_SHA256,
        "preregistration_source_sha256": evaluate.PREREGISTRATION_SOURCE_SHA256,
        "source_manifest_sha256": evaluate.SOURCE_MANIFEST_SHA256,
        "source_manifest_hash": evaluate.SOURCE_MANIFEST_HASH,
        "execution_source_manifest_sha256": evaluate.EXECUTION_SOURCE_MANIFEST_SHA256,
        "execution_source_manifest_hash": evaluate.EXECUTION_SOURCE_MANIFEST_HASH,
        "opened_windows": [],
        "sealed_windows": ["2023", "2024", "2025", "2026"],
        "mutable_parameters": [],
        "labels_constructed_during_freeze": False,
        "market_rows_parsed_during_freeze": 0,
        "funding_rows_loaded_during_freeze": 0,
        "execution_simulation_run_during_freeze": False,
        "evaluation_config": evaluate.asdict(evaluate.EvaluationConfig()),
        "source_prefix_contract": {
            "market_rows_per_symbol": evaluate.MARKET_ROWS_2023,
            "funding_rows_per_symbol": evaluate.FUNDING_ROWS_2023,
            "maximum_timestamp_exclusive": str(evaluate.END),
            "2024_rows_permitted": 0,
        },
        "decision_rule": (
            "open the singleton DCRM-1 2023 clock once under every preregistered gate; "
            "a failure retires DCRM-1 before loading any 2024 outcome"
        ),
    }
    return {**core, "manifest_hash": evaluate.canonical_hash(core)}


def validate_manifest(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if evaluate.canonical_hash(core) != payload.get("manifest_hash"):
        raise RuntimeError("DCRM-1 evaluator freeze hash mismatch")
    checks = {
        "protocol": "DCRM-1 strict 2023 evaluator pre-outcome freeze v1",
        "outcomes_opened": False,
        "evaluation_source": str(evaluate.EVALUATION_SOURCE),
        "evaluation_source_sha256": _sha256(evaluate.EVALUATION_SOURCE),
        "test_path": str(evaluate.TEST_PATH),
        "test_sha256": _sha256(evaluate.TEST_PATH),
        "freeze_source": str(evaluate.FREEZE_SOURCE),
        "freeze_source_sha256": _sha256(evaluate.FREEZE_SOURCE),
        "freeze_test_path": str(evaluate.FREEZE_TEST_PATH),
        "freeze_test_sha256": _sha256(evaluate.FREEZE_TEST_PATH),
        "preregistration_sha256": evaluate.PREREGISTRATION_SHA256,
        "support_manifest_sha256": evaluate.SUPPORT_MANIFEST_SHA256,
        "support_manifest_hash": evaluate.SUPPORT_MANIFEST_HASH,
        "clock_sha256": evaluate.CLOCK_SHA256,
        "support_source_sha256": evaluate.SUPPORT_SOURCE_SHA256,
        "preregistration_source_sha256": evaluate.PREREGISTRATION_SOURCE_SHA256,
        "source_manifest_sha256": evaluate.SOURCE_MANIFEST_SHA256,
        "source_manifest_hash": evaluate.SOURCE_MANIFEST_HASH,
        "execution_source_manifest_sha256": evaluate.EXECUTION_SOURCE_MANIFEST_SHA256,
        "execution_source_manifest_hash": evaluate.EXECUTION_SOURCE_MANIFEST_HASH,
        "mutable_parameters": [],
        "labels_constructed_during_freeze": False,
        "market_rows_parsed_during_freeze": 0,
        "funding_rows_loaded_during_freeze": 0,
        "execution_simulation_run_during_freeze": False,
        "evaluation_config": evaluate.asdict(evaluate.EvaluationConfig()),
        "decision_rule": (
            "open the singleton DCRM-1 2023 clock once under every preregistered gate; "
            "a failure retires DCRM-1 before loading any 2024 outcome"
        ),
    }
    for key, expected in checks.items():
        if payload.get(key) != expected:
            raise RuntimeError(f"DCRM-1 evaluator freeze changed: {key}")
    commit = str(payload.get("evaluation_source_commit", ""))
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise RuntimeError("DCRM-1 evaluator freeze commit is invalid")
    if payload.get("clock_rows") != 92:
        raise RuntimeError("DCRM-1 frozen clock row count changed")
    if payload.get("opened_windows") != []:
        raise RuntimeError("DCRM-1 freeze opened an outcome window")
    if payload.get("sealed_windows") != ["2023", "2024", "2025", "2026"]:
        raise RuntimeError("DCRM-1 sealed-window declaration changed")
    prefix = payload.get("source_prefix_contract", {})
    if prefix != {
        "market_rows_per_symbol": evaluate.MARKET_ROWS_2023,
        "funding_rows_per_symbol": evaluate.FUNDING_ROWS_2023,
        "maximum_timestamp_exclusive": str(evaluate.END),
        "2024_rows_permitted": 0,
    }:
        raise RuntimeError("DCRM-1 source-prefix contract changed")


def write_once(path: str | Path, payload: dict[str, Any]) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = json.loads(output.read_text())
        validate_manifest(existing)
        if existing != payload:
            raise RuntimeError("refusing to overwrite frozen DCRM-1 evaluator")
        return "verified_existing"
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return "created"


def current_clean_commit() -> str:
    paths = [
        str(evaluate.EVALUATION_SOURCE),
        str(evaluate.TEST_PATH),
        str(evaluate.FREEZE_SOURCE),
        str(evaluate.FREEZE_TEST_PATH),
    ]
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *paths], text=True
    ).strip()
    if status:
        raise RuntimeError("DCRM-1 evaluator source or tests are not clean at HEAD")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    for path in paths:
        committed = subprocess.check_output(["git", "show", f"HEAD:{path}"])
        if hashlib.sha256(committed).hexdigest() != _sha256(path):
            raise RuntimeError(f"DCRM-1 evaluator dependency is not reproducible from HEAD: {path}")
    return commit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output != DEFAULT_OUTPUT:
        raise ValueError("DCRM-1 evaluator freeze path is immutable")
    if evaluate.DEFAULT_OUTPUT.exists():
        raise RuntimeError("DCRM-1 2023 outcome result already exists")
    _, clock = evaluate.verify_support_and_clock()
    payload = build_manifest(current_clean_commit(), clock_rows=len(clock))
    status = write_once(args.output, payload)
    print(
        json.dumps(
            {
                "status": status,
                "outcomes_opened": False,
                "evaluation_source_commit": payload["evaluation_source_commit"],
                "evaluation_source_sha256": payload["evaluation_source_sha256"],
                "clock_rows": payload["clock_rows"],
                "output": args.output,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
