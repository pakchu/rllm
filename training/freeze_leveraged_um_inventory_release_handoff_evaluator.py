"""Write the LURI-48 evaluator manifest without loading prices or funding."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training import evaluate_leveraged_um_inventory_release_handoff as evaluator


@dataclass(frozen=True)
class FreezeConfig:
    output: str = str(evaluator.EVALUATION_FREEZE)
    evaluation_source_commit: str | None = None


def _git(*args: str) -> bytes:
    process = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
    )
    return process.stdout


def _resolve_commit(value: str | None) -> str:
    target = "HEAD" if value is None else value
    commit = _git("rev-parse", "--verify", f"{target}^{{commit}}").decode().strip()
    if len(commit) != 40:
        raise ValueError("LURI evaluator freeze requires a full commit hash")
    return commit


def _committed_blob_sha256(commit: str, path: Path) -> str:
    value = _git("show", f"{commit}:{path.as_posix()}")
    return hashlib.sha256(value).hexdigest()


def build_freeze_manifest(source_commit: str) -> dict[str, Any]:
    if len(source_commit) != 40:
        raise ValueError("LURI evaluator source commit must be full length")
    current_sha = evaluator._sha256(evaluator.EVALUATION_SOURCE)
    committed_sha = _committed_blob_sha256(source_commit, evaluator.EVALUATION_SOURCE)
    if committed_sha != current_sha:
        raise ValueError("working-tree LURI evaluator differs from source commit")
    evaluator.verify_preregistration()
    return {
        "protocol": "LURI-48 evaluator pre-outcome freeze",
        "outcomes_opened_for_luri48": False,
        "evaluation_source": str(evaluator.EVALUATION_SOURCE),
        "evaluation_source_sha256": current_sha,
        "evaluation_source_commit": source_commit,
        "preregistration_commit": evaluator.PREREGISTRATION_COMMIT,
        "support_commit": evaluator.SUPPORT_COMMIT,
        "clock_commit": evaluator.CLOCK_COMMIT,
        "funding_commit": evaluator.FUNDING_COMMIT,
        "support_result_sha256": evaluator.SUPPORT_RESULT_SHA256,
        "event_clock_sha256": evaluator.EVENT_CLOCK_SHA256,
        "evaluator_document_sha256": evaluator.EVALUATOR_DOCUMENT_SHA256,
        "market_manifest_sha256": evaluator.MARKET_MANIFEST_SHA256,
        "market_data_sha256": evaluator.MARKET_DATA_SHA256,
        "funding_manifest_sha256": evaluator.FUNDING_MANIFEST_SHA256,
        "funding_data_sha256": evaluator.FUNDING_DATA_SHA256,
        "opened_windows": [],
        "sealed_windows": [
            *evaluator.WINDOWS,
            "test2024",
            "eval2025",
            "ytd2026",
        ],
        "mutable_parameters": [],
        "returns_prices_or_funding_loaded_during_freeze": False,
    }


def run_freeze(cfg: FreezeConfig) -> dict[str, Any]:
    source_commit = _resolve_commit(cfg.evaluation_source_commit)
    manifest = build_freeze_manifest(source_commit)
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=FreezeConfig.output)
    parser.add_argument("--evaluation-source-commit")
    manifest = run_freeze(FreezeConfig(**vars(parser.parse_args())))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
