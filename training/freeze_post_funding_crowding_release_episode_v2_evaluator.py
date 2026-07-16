"""Freeze PFCR-2's evaluator before opening 2023-2024 outcomes."""
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

from training import (
    evaluate_post_funding_crowding_release_episode_v2_2023_2024 as evaluate,
)


DEFAULT_OUTPUT = str(evaluate.EVALUATION_FREEZE)


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_manifest(commit: str, *, clock_rows: int) -> dict[str, Any]:
    if len(commit) != 40:
        raise ValueError("evaluator source commit must be a full Git hash")
    core: dict[str, Any] = {
        "protocol": "PFCR-2 2023-2024 evaluator pre-outcome freeze v1",
        "outcomes_opened": False,
        "evaluation_source": str(evaluate.EVALUATION_SOURCE),
        "evaluation_source_sha256": _sha256(evaluate.EVALUATION_SOURCE),
        "evaluation_source_commit": commit,
        "test_path": str(evaluate.TEST_PATH),
        "test_sha256": _sha256(evaluate.TEST_PATH),
        "preregistration_sha256": evaluate.PREREGISTRATION_SHA256,
        "support_manifest_sha256": evaluate.SUPPORT_MANIFEST_SHA256,
        "support_manifest_hash": evaluate.EXPECTED_SUPPORT_MANIFEST_HASH,
        "clock_sha256": evaluate.CLOCK_SHA256,
        "clock_rows": int(clock_rows),
        "strict_simulator_source_sha256": evaluate.STRICT_SIMULATOR_SOURCE_SHA256,
        "pfcr2_support_source_sha256": evaluate.PFCR2_SUPPORT_SOURCE_SHA256,
        "pfcr2_preregistration_source_sha256": (
            evaluate.PFCR2_PREREGISTRATION_SOURCE_SHA256
        ),
        "lore_source_manifest_sha256": evaluate.LORE_SOURCE_MANIFEST_SHA256,
        "lore_source_manifest_hash": evaluate.LORE_SOURCE_MANIFEST_HASH,
        "opened_windows": [],
        "sealed_windows": ["2023", "2024", "2025", "2026"],
        "mutable_parameters": [],
        "labels_constructed_during_freeze": False,
        "market_rows_parsed_during_freeze": 0,
        "funding_rows_loaded_during_freeze": 0,
        "execution_simulation_run_during_freeze": False,
        "evaluation_config": evaluate.asdict(evaluate.EvaluationConfig()),
        "decision_rule": (
            "single PFCR-2 36-hour episode clock; evaluate 2023, 2024, and the "
            "full 2023-2024 calendar under the preregistered gates without repair"
        ),
    }
    return {**core, "manifest_hash": evaluate.canonical_hash(core)}


def validate_manifest(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if evaluate.canonical_hash(core) != payload.get("manifest_hash"):
        raise RuntimeError("PFCR-2 evaluator freeze hash mismatch")
    checks = {
        "outcomes_opened": False,
        "evaluation_source": str(evaluate.EVALUATION_SOURCE),
        "evaluation_source_sha256": _sha256(evaluate.EVALUATION_SOURCE),
        "test_path": str(evaluate.TEST_PATH),
        "test_sha256": _sha256(evaluate.TEST_PATH),
        "preregistration_sha256": evaluate.PREREGISTRATION_SHA256,
        "support_manifest_sha256": evaluate.SUPPORT_MANIFEST_SHA256,
        "support_manifest_hash": evaluate.EXPECTED_SUPPORT_MANIFEST_HASH,
        "clock_sha256": evaluate.CLOCK_SHA256,
        "strict_simulator_source_sha256": evaluate.STRICT_SIMULATOR_SOURCE_SHA256,
        "pfcr2_support_source_sha256": evaluate.PFCR2_SUPPORT_SOURCE_SHA256,
        "pfcr2_preregistration_source_sha256": (
            evaluate.PFCR2_PREREGISTRATION_SOURCE_SHA256
        ),
        "lore_source_manifest_sha256": evaluate.LORE_SOURCE_MANIFEST_SHA256,
        "lore_source_manifest_hash": evaluate.LORE_SOURCE_MANIFEST_HASH,
        "mutable_parameters": [],
        "labels_constructed_during_freeze": False,
        "market_rows_parsed_during_freeze": 0,
        "funding_rows_loaded_during_freeze": 0,
        "execution_simulation_run_during_freeze": False,
        "evaluation_config": evaluate.asdict(evaluate.EvaluationConfig()),
    }
    for key, expected in checks.items():
        if payload.get(key) != expected:
            raise RuntimeError(f"PFCR-2 evaluator freeze changed: {key}")
    if payload.get("clock_rows") != 82:
        raise RuntimeError("PFCR-2 frozen clock row count changed")
    if payload.get("opened_windows") != []:
        raise RuntimeError("PFCR-2 freeze opened a window")
    if payload.get("sealed_windows") != ["2023", "2024", "2025", "2026"]:
        raise RuntimeError("PFCR-2 freeze sealed-window declaration changed")


def write_once(path: str | Path, payload: dict[str, Any]) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = json.loads(output.read_text())
        validate_manifest(existing)
        if existing != payload:
            raise RuntimeError("refusing to overwrite frozen PFCR-2 evaluator")
        return "verified_existing"
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return "created"


def current_clean_commit() -> str:
    paths = [str(evaluate.EVALUATION_SOURCE), str(evaluate.TEST_PATH)]
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *paths], text=True
    ).strip()
    if status:
        raise RuntimeError("PFCR-2 evaluator source or tests are not clean at HEAD")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    for path in paths:
        committed = subprocess.check_output(["git", "show", f"HEAD:{path}"])
        if hashlib.sha256(committed).hexdigest() != _sha256(path):
            raise RuntimeError(
                f"PFCR-2 evaluator dependency is not reproducible from HEAD: {path}"
            )
    return commit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output != DEFAULT_OUTPUT:
        raise ValueError("PFCR-2 evaluator freeze path is immutable")
    if evaluate.DEFAULT_OUTPUT.exists():
        raise RuntimeError("PFCR-2 2023-2024 outcome result already exists")
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
