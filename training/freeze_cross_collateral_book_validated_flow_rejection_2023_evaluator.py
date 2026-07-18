"""Freeze the CBFR-72 strict evaluator without parsing execution outcomes."""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from pathlib import Path

from training import evaluate_cross_collateral_book_validated_flow_rejection_2023 as ev
from training.preregister_cross_collateral_book_validated_flow_rejection import (
    canonical_hash,
)


PRIOR_FREEZE_SHA256 = (
    "72b2f91d3741fb9dd52c00332b723658575571a49fa438f994e476469f336ac3"
)


def build_payload(commit: str) -> dict[str, object]:
    return {
        "protocol": "CBFR-72 strict 2023 evaluator implementation-recovery freeze v2",
        "outcomes_opened": True,
        "prior_freeze_sha256": PRIOR_FREEZE_SHA256,
        "first_attempt": {
            "execution_sources_opened": True,
            "result_artifact_written": False,
            "markdown_artifact_written": False,
            "stdout_metric_payload_written": False,
            "failure": "TypeError adding five-minute Timedelta to string control-clock dates",
            "failure_stage": "delay-control construction after primary calculations",
            "strategy_parameters_changed": False,
            "support_clock_changed": False,
            "outcome_metrics_used_for_repair": False,
        },
        "evaluation_source": str(ev.EVALUATION_SOURCE),
        "evaluation_source_sha256": ev.sha256(ev.EVALUATION_SOURCE),
        "test_path": str(ev.TEST_PATH),
        "test_sha256": ev.sha256(ev.TEST_PATH),
        "freeze_source": str(ev.FREEZE_SOURCE),
        "freeze_source_sha256": ev.sha256(ev.FREEZE_SOURCE),
        "freeze_test_path": str(ev.FREEZE_TEST_PATH),
        "freeze_test_sha256": ev.sha256(ev.FREEZE_TEST_PATH),
        "support_sha256": ev.SUPPORT_SHA256,
        "primary_clock_sha256": ev.PRIMARY_CLOCK_SHA256,
        "primary_event_clock_hash": ev.PRIMARY_EVENT_CLOCK_HASH,
        "market_data_sha256": ev.MARKET_DATA_SHA256,
        "funding_data_sha256": ev.FUNDING_DATA_SHA256,
        "strict_ledger_source_sha256": ev.STRICT_LEDGER_SOURCE_SHA256,
        "evaluation_config": asdict(ev.CONFIG),
        "mutable_parameters": [],
        "market_rows_parsed_during_freeze": 0,
        "funding_rows_loaded_during_freeze": 0,
        "execution_simulation_run_during_freeze": False,
        "opened_windows": [],
        "sealed_windows": ["2023", "2024", "2025", "2026"],
        "evaluation_source_commit": commit,
    }


def run(output: str | Path | None = None) -> dict[str, object]:
    path = ev.EVALUATION_FREEZE if output is None else Path(output)
    if path != ev.EVALUATION_FREEZE:
        raise ValueError("CBFR evaluator freeze output is immutable")
    if not path.exists() or ev.sha256(path) != PRIOR_FREEZE_SHA256:
        raise RuntimeError("CBFR recovery requires the exact failed v1 freeze")
    status = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    if status:
        raise RuntimeError("repository must be clean before evaluator freeze")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    payload = build_payload(commit)
    payload["manifest_hash"] = canonical_hash(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
