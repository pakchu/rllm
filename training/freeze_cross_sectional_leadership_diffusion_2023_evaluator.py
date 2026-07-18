"""Freeze the CLD-72 strict evaluator before opening 2023 outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

from training import evaluate_cross_sectional_leadership_diffusion_2023 as evaluate
from training import preregister_cross_sectional_leadership_diffusion as cld


DEFAULT_OUTPUT = str(evaluate.EVALUATION_FREEZE)


def build_manifest(commit: str, *, primary_rows: int) -> dict[str, Any]:
    if len(commit) != 40:
        raise ValueError("evaluator source commit must be a full Git hash")
    core = {
        "protocol": "CLD-72 strict 2023 evaluator pre-outcome freeze v1",
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
        "support_sha256": evaluate.SUPPORT_SHA256,
        "primary_clock_sha256": evaluate.PRIMARY_CLOCK_SHA256,
        "primary_event_clock_hash": evaluate.PRIMARY_EVENT_CLOCK_HASH,
        "control_clocks_sha256": evaluate.CONTROL_CLOCKS_SHA256,
        "control_clocks_manifest_hash": evaluate.CONTROL_CLOCKS_MANIFEST_HASH,
        "market_data_sha256": evaluate.MARKET_DATA_SHA256,
        "funding_data_sha256": evaluate.FUNDING_DATA_SHA256,
        "strict_ledger_source_sha256": evaluate.STRICT_LEDGER_SOURCE_SHA256,
        "evaluation_config": asdict(evaluate.CONFIG),
        "mutable_parameters": [],
        "market_rows_parsed_during_freeze": 0,
        "funding_rows_loaded_during_freeze": 0,
        "execution_simulation_run_during_freeze": False,
        "opened_windows": [],
        "sealed_windows": ["2023", "2024", "2025", "2026"],
        "primary_clock_rows": int(primary_rows),
        "control_clock_rows": evaluate.CONTROL_COUNTS,
        "source_prefix_contract": {
            "market_rows": evaluate.source_export.MARKET_ROWS,
            "funding_rows": evaluate.source_export.FUNDING_ROWS,
            "maximum_timestamp_exclusive": str(evaluate.END),
            "2024_rows_permitted": 0,
        },
        "decision_rule": (
            "open the singleton CLD-72 2023 clock once under all frozen gates; "
            "failure retires CLD-72 without opening 2024 or repairing from controls"
        ),
    }
    return {**core, "manifest_hash": cld.canonical_hash(core)}


def validate_manifest(payload: dict[str, Any]) -> None:
    body = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if cld.canonical_hash(body) != payload.get("manifest_hash"):
        raise RuntimeError("CLD evaluator freeze hash mismatch")
    if payload.get("outcomes_opened") is not False:
        raise RuntimeError("CLD evaluator freeze opened an outcome")
    if payload.get("primary_clock_rows") != 106:
        raise RuntimeError("CLD evaluator freeze primary count changed")
    if payload.get("control_clock_rows") != evaluate.CONTROL_COUNTS:
        raise RuntimeError("CLD evaluator freeze control counts changed")
    if payload.get("mutable_parameters") != [] or payload.get("opened_windows") != []:
        raise RuntimeError("CLD evaluator freeze permits mutation or an opened window")


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
        raise RuntimeError("CLD evaluator source or tests are not clean at HEAD")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    for path in paths:
        committed = subprocess.check_output(["git", "show", f"HEAD:{path}"])
        if hashlib.sha256(committed).hexdigest() != evaluate.sha256(path):
            raise RuntimeError(f"CLD evaluator dependency is not reproducible: {path}")
    return commit


def run(output: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    path = Path(output)
    if str(path) != DEFAULT_OUTPUT:
        raise ValueError("CLD evaluator freeze path is immutable")
    if evaluate.DEFAULT_OUTPUT.exists() or evaluate.DEFAULT_DOCS.exists():
        raise RuntimeError("CLD 2023 outcome result already exists")
    _, primary, _ = evaluate.verify_preoutcome_artifacts()
    payload = build_manifest(current_clean_commit(), primary_rows=len(primary))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text())
        validate_manifest(existing)
        if existing != payload:
            raise RuntimeError("refusing to overwrite frozen CLD evaluator")
        return existing
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = run(args.output)
    print(
        json.dumps(
            {
                "outcomes_opened": False,
                "evaluation_source_commit": payload["evaluation_source_commit"],
                "primary_clock_rows": payload["primary_clock_rows"],
                "control_clock_rows": payload["control_clock_rows"],
                "output": args.output,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
