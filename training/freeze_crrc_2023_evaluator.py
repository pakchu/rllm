"""Freeze CRRC-72's strict evaluator before opening 2023 outcomes."""
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

from training import evaluate_crrc_2023 as evaluate


DEFAULT_OUTPUT = str(evaluate.EVALUATION_FREEZE)


def build_manifest(commit: str, *, primary_rows: int) -> dict[str, Any]:
    if len(commit) != 40:
        raise ValueError("evaluator source commit must be a full Git hash")
    core: dict[str, Any] = {
        "protocol": "CRRC-72 strict 2023 evaluator pre-outcome freeze v1",
        "outcomes_opened": False,
        "evaluation_source": str(evaluate.EVALUATION_SOURCE),
        "evaluation_source_sha256": evaluate.sha256(evaluate.EVALUATION_SOURCE),
        "evaluation_source_commit": commit,
        "test_path": str(evaluate.TEST_PATH),
        "test_sha256": evaluate.sha256(evaluate.TEST_PATH),
        "freeze_source": str(evaluate.FREEZE_SOURCE),
        "freeze_source_sha256": evaluate.sha256(evaluate.FREEZE_SOURCE),
        "freeze_test_path": str(evaluate.FREEZE_TEST_PATH),
        "freeze_test_sha256": evaluate.sha256(evaluate.FREEZE_TEST_PATH),
        "preregistration_sha256": evaluate.PREREGISTRATION_SHA256,
        "support_sha256": evaluate.SUPPORT_SHA256,
        "primary_clock_sha256": evaluate.PRIMARY_CLOCK_SHA256,
        "primary_event_clock_hash": evaluate.PRIMARY_EVENT_CLOCK_HASH,
        "control_clocks_sha256": evaluate.CONTROL_CLOCKS_SHA256,
        "execution_source_manifest_sha256": evaluate.EXECUTION_SOURCE_MANIFEST_SHA256,
        "execution_source_manifest_hash": evaluate.EXECUTION_SOURCE_MANIFEST_HASH,
        "evaluation_config": evaluate.asdict(evaluate.CONFIG),
        "mutable_parameters": [],
        "market_rows_parsed_during_freeze": 0,
        "funding_rows_loaded_during_freeze": 0,
        "execution_simulation_run_during_freeze": False,
        "opened_windows": [],
        "sealed_windows": ["2023", "2024", "2025", "2026"],
        "primary_clock_rows": int(primary_rows),
        "source_prefix_contract": {
            "market_rows": evaluate.source_export.MARKET_ROWS,
            "funding_rows": evaluate.source_export.FUNDING_ROWS,
            "maximum_timestamp_exclusive": str(evaluate.END),
            "2024_rows_permitted": 0,
        },
        "decision_rule": (
            "open the singleton CRRC-72 2023 clock once under all frozen gates; "
            "a failure retires CRRC-72 before loading any 2024 outcome"
        ),
    }
    return {**core, "manifest_hash": evaluate.canonical_hash(core)}


def validate_manifest(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if evaluate.canonical_hash(core) != payload.get("manifest_hash"):
        raise RuntimeError("CRRC evaluator freeze hash mismatch")
    if payload.get("outcomes_opened") is not False:
        raise RuntimeError("CRRC evaluator freeze opened an outcome")
    if payload.get("primary_clock_rows") != 156:
        raise RuntimeError("CRRC evaluator freeze clock count changed")
    if payload.get("mutable_parameters") != []:
        raise RuntimeError("CRRC evaluator freeze permits mutable parameters")
    if payload.get("opened_windows") != []:
        raise RuntimeError("CRRC evaluator freeze opened a window")
    if payload.get("sealed_windows") != ["2023", "2024", "2025", "2026"]:
        raise RuntimeError("CRRC evaluator freeze sealed-window contract changed")
    if payload.get("source_prefix_contract") != {
        "market_rows": evaluate.source_export.MARKET_ROWS,
        "funding_rows": evaluate.source_export.FUNDING_ROWS,
        "maximum_timestamp_exclusive": str(evaluate.END),
        "2024_rows_permitted": 0,
    }:
        raise RuntimeError("CRRC evaluator freeze source-prefix contract changed")
    if payload.get("decision_rule") != (
        "open the singleton CRRC-72 2023 clock once under all frozen gates; "
        "a failure retires CRRC-72 before loading any 2024 outcome"
    ):
        raise RuntimeError("CRRC evaluator freeze decision rule changed")


def write_once(path: str | Path, payload: dict[str, Any]) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = json.loads(output.read_text())
        validate_manifest(existing)
        if existing != payload:
            raise RuntimeError("refusing to overwrite frozen CRRC evaluator")
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
        raise RuntimeError("CRRC evaluator source or tests are not clean at HEAD")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    for path in paths:
        committed = subprocess.check_output(["git", "show", f"HEAD:{path}"])
        if hashlib.sha256(committed).hexdigest() != evaluate.sha256(path):
            raise RuntimeError(f"CRRC evaluator dependency is not reproducible from HEAD: {path}")
    return commit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output != DEFAULT_OUTPUT:
        raise ValueError("CRRC evaluator freeze path is immutable")
    if evaluate.DEFAULT_OUTPUT.exists():
        raise RuntimeError("CRRC 2023 outcome result already exists")
    _, primary, _ = evaluate.verify_preoutcome_artifacts()
    payload = build_manifest(current_clean_commit(), primary_rows=len(primary))
    status = write_once(args.output, payload)
    print(
        json.dumps(
            {
                "status": status,
                "outcomes_opened": False,
                "evaluation_source_commit": payload["evaluation_source_commit"],
                "evaluation_source_sha256": payload["evaluation_source_sha256"],
                "primary_clock_rows": payload["primary_clock_rows"],
                "output": args.output,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
