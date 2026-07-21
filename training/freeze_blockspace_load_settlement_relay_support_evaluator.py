"""Freeze the committed BLSR-288 source-only evaluator before incidence."""

# Frozen pandas timestamps are concrete, although stubs retain NaT unions.
# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Sequence

from training import evaluate_blockspace_load_settlement_relay_support as evaluate


PROTOCOL_VERSION = "blockspace_load_settlement_relay_evaluator_freeze_v1"
DEFAULT_OUTPUT = Path(
    "results/blockspace_load_settlement_relay_evaluator_freeze_2026-07-21.json"
)
EXPECTED_EVALUATOR_COMMIT = "08568c69394fcfd49c3554d2176bddf694abf86f"
EXPECTED_EVALUATOR_SHA256 = (
    "dcbb40d7de7162f9f4e1ff765036838b6673d85ab169c5f4b951c973effcc611"
)
FROZEN_EVALUATOR_PROTOCOL_VERSION = "blockspace_load_settlement_relay_support_v1"

FREEZE_BOUNDARY = {
    "source_artifact_bytes_hashed": True,
    "comparator_artifact_bytes_hashed": True,
    "source_value_rows_read": 0,
    "source_feature_rows_derived": 0,
    "candidate_event_rows_derived": 0,
    "comparator_event_rows_read": 0,
    "support_or_novelty_verdicts_produced": 0,
    "btc_market_rows_loaded": 0,
    "funding_rows_loaded": 0,
    "return_pnl_or_equity_rows_loaded": 0,
    "economic_simulations_run": 0,
    "network_calls": 0,
    "git_subprocess_calls": 1,
}


def _clean_committed_evaluator() -> tuple[str, str]:
    source = str(evaluate.EVALUATOR_SOURCE)
    committed = subprocess.check_output(
        ["git", "show", f"{EXPECTED_EVALUATOR_COMMIT}:{source}"],
        cwd=evaluate.REPOSITORY_ROOT,
    )
    source_sha = hashlib.sha256(committed).hexdigest()
    if source_sha != EXPECTED_EVALUATOR_SHA256:
        raise RuntimeError("BLSR evaluator committed source SHA drift")
    working = evaluate._repository_path(evaluate.EVALUATOR_SOURCE)
    if working.is_symlink() or not working.is_file():
        raise RuntimeError("BLSR evaluator source is missing or symlinked")
    if hashlib.sha256(working.read_bytes()).hexdigest() != source_sha:
        raise RuntimeError("BLSR evaluator working source differs from frozen commit")
    return EXPECTED_EVALUATOR_COMMIT, source_sha


def build_manifest(commit: str, source_sha: str) -> dict[str, Any]:
    registration = evaluate.load_preregistration()
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "candidate": evaluate.POLICY_ID,
        "evaluator_source": str(evaluate.EVALUATOR_SOURCE),
        "evaluator_protocol_version": FROZEN_EVALUATOR_PROTOCOL_VERSION,
        "evaluator_source_commit": commit,
        "evaluator_source_sha256": source_sha,
        "preregistration": {
            "path": str(evaluate.PREREGISTRATION_ARTIFACT),
            "sha256": evaluate.PREREGISTRATION_ARTIFACT_SHA256,
            "manifest_hash": evaluate.PREREGISTRATION_MANIFEST_HASH,
            "policy_hash": evaluate.PREREGISTRATION_POLICY_HASH,
        },
        "ledger_builder_dependency": {
            "path": str(evaluate.LEDGER_BUILDER),
            "sha256": evaluate.LEDGER_BUILDER_SHA256,
            "authorized_role": (
                "FETD comparator feature/clock reconstruction from BLSR-local "
                "allowed-column packet aggregates only"
            ),
            "source_loader_or_packetizer_reuse": False,
        },
        "frozen_contract": {
            "allowed_source_columns": list(evaluate.BLSR_SOURCE_COLUMNS),
            "forbidden_source_value_columns": list(
                evaluate.FORBIDDEN_SOURCE_VALUE_COLUMNS
            ),
            "feature_columns": list(evaluate.FEATURE_COLUMNS),
            "clock_columns": list(evaluate.CLOCK_COLUMNS),
            "controls": list(evaluate.CONTROL_NAMES),
            "support_limits": dict(evaluate.SUPPORT_LIMITS),
            "novelty_limits": dict(evaluate.NOVELTY_LIMITS),
            "comparator_capabilities": dict(evaluate.EXPECTED_COMPARATOR_CAPABILITIES),
            "evaluation_start_utc": evaluate.EVALUATION_START.isoformat(),
            "train_end_utc": evaluate.TRAIN_END.isoformat(),
            "evaluation_end_utc": evaluate.EVALUATION_END.isoformat(),
            "hold_hours": int(evaluate.HOLD.total_seconds() // 3_600),
            "tolerant_match_hours": int(
                evaluate.TOLERANT_MATCH.total_seconds() // 3_600
            ),
            "relay_deadline_packets": 3,
            "stale_response_contract": (
                "shift only a strictly-later endpoint response by one packet; "
                "same-onset-packet endpoint state is ineligible"
            ),
            "scheduler": (
                "global (entry_time,onset_packet_id,confirmation_packet_id); "
                "entry equal prior exit allowed"
            ),
            "exposure_grid": "full [2021-01-01,2024-01-01) UTC at 5m",
            "publication": (
                "aggregate support/drop/comparator statistics and clock hashes; "
                "zero event rows and zero source feature values"
            ),
        },
        "comparator_bindings": registration["comparator_bindings"],
        "freeze_boundary": dict(FREEZE_BOUNDARY),
        "mutable_parameters": [],
        "source_support_artifact_exists_before_freeze": False,
        "next_action": "run the exact committed source-only evaluator once",
    }
    return {**core, "manifest_hash": evaluate.canonical_hash(core)}


def validate_manifest(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if evaluate.canonical_hash(core) != payload.get("manifest_hash"):
        raise RuntimeError("BLSR evaluator freeze manifest hash mismatch")
    if payload.get("candidate") != evaluate.POLICY_ID:
        raise RuntimeError("BLSR evaluator freeze identity drift")
    if payload.get("evaluator_source_commit") != EXPECTED_EVALUATOR_COMMIT:
        raise RuntimeError("BLSR evaluator freeze commit drift")
    if payload.get("evaluator_source_sha256") != EXPECTED_EVALUATOR_SHA256:
        raise RuntimeError("BLSR evaluator freeze source SHA drift")
    if payload.get("mutable_parameters") != []:
        raise RuntimeError("BLSR evaluator freeze permits parameter repair")
    if payload.get("freeze_boundary") != FREEZE_BOUNDARY:
        raise RuntimeError("BLSR evaluator freeze boundary drift")
    zero_fields = (
        "source_value_rows_read",
        "source_feature_rows_derived",
        "candidate_event_rows_derived",
        "comparator_event_rows_read",
        "support_or_novelty_verdicts_produced",
        "btc_market_rows_loaded",
        "funding_rows_loaded",
        "return_pnl_or_equity_rows_loaded",
        "economic_simulations_run",
        "network_calls",
    )
    if any(payload["freeze_boundary"].get(field) != 0 for field in zero_fields):
        raise RuntimeError("BLSR evaluator freeze opened incidence or outcomes")


def freeze(output: str | Path = DEFAULT_OUTPUT) -> tuple[str, dict[str, Any]]:
    if evaluate._repository_path(evaluate.DEFAULT_OUTPUT_REPORT).exists():
        raise RuntimeError("BLSR source-support artifact predates evaluator freeze")
    commit, source_sha = _clean_committed_evaluator()
    payload = build_manifest(commit, source_sha)
    validate_manifest(payload)
    target = evaluate._repository_path(output)
    if target in evaluate._protected_paths(evaluate.load_preregistration()):
        raise ValueError("BLSR evaluator freeze aliases a protected input")
    if not target.is_relative_to(evaluate._repository_path(evaluate.ARTIFACT_ROOT)):
        raise ValueError("BLSR evaluator freeze must stay under results")
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        validate_manifest(existing)
        if existing != payload:
            raise RuntimeError("refusing to overwrite BLSR evaluator freeze")
        return "verified_existing", payload
    with target.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
    return "created", payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    status, payload = freeze(args.output)
    print(
        json.dumps(
            {
                "status": status,
                "candidate": payload["candidate"],
                "evaluator_source_commit": payload["evaluator_source_commit"],
                "manifest_hash": payload["manifest_hash"],
                "source_value_rows_read": 0,
                "comparator_event_rows_read": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
