"""Seal the AFCS-144 evaluator before any execution outcome is opened."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from training import evaluate_aggregate_fill_compression_sweep as evaluate


DEFAULT_OUTPUT = str(evaluate.EVALUATION_FREEZE)


def current_clean_source_commit() -> str:
    source = str(evaluate.EVALUATION_SOURCE)
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", source], text=True
    ).strip()
    if status:
        raise RuntimeError("AFCS-144 evaluator source is not clean at HEAD")
    commit = subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", source], text=True
    ).strip()
    if len(commit) != 40:
        raise RuntimeError("AFCS-144 evaluator source has no full commit hash")
    committed = subprocess.check_output(["git", "show", f"{commit}:{source}"])
    if hashlib.sha256(committed).hexdigest() != evaluate._sha256(
        evaluate.EVALUATION_SOURCE
    ):
        raise RuntimeError("AFCS-144 evaluator source is not reproducible from Git")
    return commit


def validate_manifest(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if evaluate._canonical_hash(core) != payload.get("manifest_hash"):
        raise RuntimeError("AFCS-144 evaluator freeze manifest hash mismatch")
    if payload.get("opened_windows") != []:
        raise RuntimeError("AFCS-144 evaluator freeze opened an outcome window")
    if payload.get("mutable_parameters") != []:
        raise RuntimeError("AFCS-144 evaluator freeze permits parameter repair")
    if payload.get("execution_ohlc_rows_parsed_during_freeze") != 0:
        raise RuntimeError("AFCS-144 evaluator freeze parsed execution OHLC")
    if payload.get("funding_settlement_marks_loaded_during_freeze") != 0:
        raise RuntimeError("AFCS-144 evaluator freeze loaded funding marks")
    if payload.get("execution_simulation_run_during_freeze") is not False:
        raise RuntimeError("AFCS-144 evaluator freeze ran a simulation")


def write_once(path: str | Path, payload: dict[str, Any]) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if output.exists():
        existing = json.loads(output.read_text())
        validate_manifest(existing)
        if existing != payload:
            raise RuntimeError("refusing to overwrite AFCS-144 evaluator freeze")
        return "verified_existing"
    with output.open("x", encoding="utf-8") as handle:
        handle.write(content)
    return "created"


def freeze(output: str | Path = DEFAULT_OUTPUT) -> tuple[str, dict[str, Any]]:
    if evaluate.STAGE1_OUTPUT.exists() or evaluate.STAGE2_OUTPUT.exists():
        raise RuntimeError("AFCS-144 outcome artifact predates evaluator freeze")
    commit = current_clean_source_commit()
    payload = evaluate.build_freeze_manifest(commit)
    validate_manifest(payload)
    return write_once(output, payload), payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    status, payload = freeze(args.output)
    print(
        json.dumps(
            {
                "status": status,
                "outcomes_opened": False,
                "evaluation_source_commit": payload["evaluation_source_commit"],
                "evaluation_source_sha256": payload["evaluation_source_sha256"],
                "output": args.output,
                "manifest_hash": payload["manifest_hash"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
