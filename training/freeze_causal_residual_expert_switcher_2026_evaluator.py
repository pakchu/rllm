"""Freeze CRES-1's 2026H1 evaluator before opening any outcomes."""
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

from training import evaluate_causal_residual_expert_switcher_2026 as evaluate


DEFAULT_OUTPUT = str(evaluate.EVALUATION_FREEZE)


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_manifest(commit: str, *, base_clock_rows: int, seed_rows: int) -> dict[str, Any]:
    if len(commit) != 40:
        raise ValueError("evaluator source commit must be a full Git hash")
    core: dict[str, Any] = {
        "protocol": "CRES-1 2026H1 evaluator pre-outcome freeze v1",
        "outcomes_opened": False,
        "evaluation_source": str(evaluate.EVALUATION_SOURCE),
        "evaluation_source_sha256": _sha256(evaluate.EVALUATION_SOURCE),
        "evaluation_source_commit": commit,
        "test_path": str(evaluate.TEST_PATH),
        "test_sha256": _sha256(evaluate.TEST_PATH),
        "preregistration_sha256": evaluate.PREREGISTRATION_SHA256,
        "source_manifest_sha256": evaluate.SOURCE_MANIFEST_SHA256,
        "support_manifest_sha256": evaluate.SUPPORT_MANIFEST_SHA256,
        "support_manifest_hash": evaluate.EXPECTED_SUPPORT_MANIFEST_HASH,
        "clock_sha256": evaluate.CLOCK_SHA256,
        "clock_rows": int(base_clock_rows),
        "seed_sha256": evaluate.SEED_SHA256,
        "seed_rows": int(seed_rows),
        "development_source_sha256": evaluate.DEVELOPMENT_SOURCE_SHA256,
        "development_test_sha256": evaluate.DEVELOPMENT_TEST_SHA256,
        "opened_windows": [],
        "sealed_windows": ["2026H1", "2026Q1", "2026Q2"],
        "mutable_parameters": [],
        "labels_constructed_during_freeze": False,
        "market_rows_parsed_during_freeze": 0,
        "funding_rows_loaded_during_freeze": 0,
        "execution_simulation_run_during_freeze": False,
        "evaluation_config": evaluate.asdict(evaluate.EvaluationConfig()),
        "decision_rule": (
            "each action is materialized before its event outcome; only outcomes "
            "published by prior exit plus five minutes enter later fits"
        ),
    }
    return {**core, "manifest_hash": evaluate.canonical_hash(core)}


def validate_manifest(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if evaluate.canonical_hash(core) != payload.get("manifest_hash"):
        raise RuntimeError("CRES evaluator freeze hash mismatch")
    if payload.get("outcomes_opened") is not False:
        raise RuntimeError("CRES evaluator freeze cannot open outcomes")
    if payload.get("evaluation_source") != str(evaluate.EVALUATION_SOURCE):
        raise RuntimeError("CRES evaluator freeze source changed")
    if payload.get("evaluation_source_sha256") != _sha256(evaluate.EVALUATION_SOURCE):
        raise RuntimeError("CRES evaluator freeze source hash changed")
    if payload.get("test_path") != str(evaluate.TEST_PATH):
        raise RuntimeError("CRES evaluator freeze test path changed")
    if payload.get("test_sha256") != _sha256(evaluate.TEST_PATH):
        raise RuntimeError("CRES evaluator freeze test hash changed")
    if payload.get("support_manifest_sha256") != evaluate.SUPPORT_MANIFEST_SHA256:
        raise RuntimeError("CRES evaluator freeze support binding changed")
    if payload.get("clock_sha256") != evaluate.CLOCK_SHA256:
        raise RuntimeError("CRES evaluator freeze clock binding changed")
    if payload.get("seed_sha256") != evaluate.SEED_SHA256:
        raise RuntimeError("CRES evaluator freeze seed binding changed")
    if payload.get("mutable_parameters") != []:
        raise RuntimeError("CRES evaluator freeze permits mutable parameters")
    if payload.get("labels_constructed_during_freeze") is not False:
        raise RuntimeError("CRES evaluator freeze constructed labels")
    if payload.get("market_rows_parsed_during_freeze") != 0:
        raise RuntimeError("CRES evaluator freeze parsed market rows")
    if payload.get("funding_rows_loaded_during_freeze") != 0:
        raise RuntimeError("CRES evaluator freeze loaded funding rows")
    if payload.get("execution_simulation_run_during_freeze") is not False:
        raise RuntimeError("CRES evaluator freeze simulated execution")
    if payload.get("evaluation_config") != evaluate.asdict(evaluate.EvaluationConfig()):
        raise RuntimeError("CRES evaluator freeze configuration changed")


def write_once(path: str | Path, payload: dict[str, Any]) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = json.loads(output.read_text())
        validate_manifest(existing)
        if existing != payload:
            raise RuntimeError("refusing to overwrite frozen CRES evaluator")
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
        raise RuntimeError("CRES evaluator source or tests are not clean at HEAD")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    for path in paths:
        committed = subprocess.check_output(["git", "show", f"HEAD:{path}"])
        if hashlib.sha256(committed).hexdigest() != _sha256(path):
            raise RuntimeError(f"CRES evaluator dependency is not reproducible from HEAD: {path}")
    return commit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output != DEFAULT_OUTPUT:
        raise ValueError("CRES evaluator freeze path is immutable")
    if Path(evaluate.DEFAULT_OUTPUT).exists():
        raise RuntimeError("CRES 2026 outcome result already exists")
    _, clock, seed = evaluate.verify_support_and_clock()
    payload = build_manifest(
        current_clean_commit(), base_clock_rows=len(clock), seed_rows=len(seed)
    )
    status = write_once(args.output, payload)
    print(
        json.dumps(
            {
                "status": status,
                "outcomes_opened": False,
                "evaluation_source_commit": payload["evaluation_source_commit"],
                "evaluation_source_sha256": payload["evaluation_source_sha256"],
                "clock_rows": payload["clock_rows"],
                "seed_rows": payload["seed_rows"],
                "output": args.output,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
