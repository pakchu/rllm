"""Freeze corrected CDLTR-72A evaluator v3 after the audited v2 batch failure."""

# Pandas Timestamp stubs include NaT although the imported bounds are fixed.
# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Sequence

from training import (
    evaluate_cross_domain_liquidity_transmission_relay_support as evaluate,
)


PROTOCOL_VERSION = "cross_domain_liquidity_transmission_relay_evaluator_freeze_v3"
DEFAULT_OUTPUT = Path(
    "results/cross_domain_liquidity_transmission_relay_evaluator_freeze_v3_2026-07-21.json"
)
PREDECESSOR_FREEZE = Path(
    "results/cross_domain_liquidity_transmission_relay_evaluator_freeze_v2_2026-07-21.json"
)
PREDECESSOR_FREEZE_SHA256 = (
    "f11b9c52538b370b80f5d1ee3e8d62f8ec132083a6060c2d983e9b9ed8c7965d"
)
CORRECTION_DOCUMENT = Path(
    "docs/cdltr72a-source-evaluator-v2-failure-v3-correction-2026-07-21.md"
)
CORRECTION_DOCUMENT_SHA256 = (
    "8d4af0408928cf91bf7ec85609b4646b42e85259564d6ac67344bcc242b74a1b"
)
EXPECTED_EVALUATOR_COMMIT = "b42774eee3e602ab04590107e72999c4b170f973"
EXPECTED_EVALUATOR_SHA256 = (
    "30b2c85e406fdd7ef54fb97035390ccad58b97dbb23e4347271ce7037c7e3bdc"
)
FROZEN_EVALUATOR_PROTOCOL_VERSION = (
    "cross_domain_liquidity_transmission_relay_support_v3"
)

V1_FAILED_RUN_BOUNDARY = {
    "source_value_rows_loaded": 4_468,
    "comparator_event_rows_loaded": 9_985,
    "rrp_vote_rows_derived_in_memory": 1_498,
    "cboe_vote_rows_derived_in_memory": 1_508,
    "network_vote_rows_derived_in_memory": 0,
    "candidate_event_rows_derived": 0,
    "support_or_novelty_verdicts_produced": 0,
    "btc_market_rows_loaded": 0,
    "funding_rows_loaded": 0,
    "return_pnl_or_equity_rows_loaded": 0,
    "economic_simulations_run": 0,
    "output_artifacts_written": 0,
}
V2_FAILED_RUN_BOUNDARY = {
    "source_value_rows_loaded": 4_468,
    "comparator_event_rows_loaded": 9_985,
    "rrp_vote_rows_derived_in_memory": 1_498,
    "cboe_vote_rows_derived_in_memory": 1_508,
    "raw_network_vote_rows_derived_in_memory": 1_461,
    "candidate_event_rows_derived": 0,
    "support_or_novelty_verdicts_produced": 0,
    "btc_market_rows_loaded": 0,
    "funding_rows_loaded": 0,
    "return_pnl_or_equity_rows_loaded": 0,
    "economic_simulations_run": 0,
    "output_artifacts_written": 0,
}


def _require_file(path: Path, expected_sha: str, label: str) -> Path:
    target = evaluate._repository_path(path)
    if target.is_symlink() or not target.is_file():
        raise RuntimeError(f"{label} is missing or symlinked")
    if hashlib.sha256(target.read_bytes()).hexdigest() != expected_sha:
        raise RuntimeError(f"{label} SHA drift")
    return target


def _clean_committed_evaluator() -> tuple[str, str]:
    source = str(evaluate.EVALUATOR_SOURCE)
    committed = subprocess.check_output(
        ["git", "show", f"{EXPECTED_EVALUATOR_COMMIT}:{source}"],
        cwd=evaluate.REPOSITORY_ROOT,
    )
    source_sha = hashlib.sha256(committed).hexdigest()
    if source_sha != EXPECTED_EVALUATOR_SHA256:
        raise RuntimeError("CDLTR v3 evaluator source SHA drift")
    working_source = evaluate._repository_path(evaluate.EVALUATOR_SOURCE)
    if working_source.is_symlink() or not working_source.is_file():
        raise RuntimeError("CDLTR v3 evaluator source is missing or symlinked")
    if hashlib.sha256(working_source.read_bytes()).hexdigest() != source_sha:
        raise RuntimeError(
            "CDLTR v3 evaluator working source differs from frozen commit"
        )
    return EXPECTED_EVALUATOR_COMMIT, source_sha


def build_manifest(commit: str, source_sha: str) -> dict[str, Any]:
    preregistration = evaluate._load_preregistration()
    _require_file(
        PREDECESSOR_FREEZE,
        PREDECESSOR_FREEZE_SHA256,
        "CDLTR v2 evaluator freeze",
    )
    _require_file(
        CORRECTION_DOCUMENT,
        CORRECTION_DOCUMENT_SHA256,
        "CDLTR v3 evaluator correction document",
    )
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
            "policy_hash": preregistration["policy_hash"],
        },
        "predecessor_freeze": {
            "path": str(PREDECESSOR_FREEZE),
            "sha256": PREDECESSOR_FREEZE_SHA256,
        },
        "v2_failure_and_v3_correction": {
            "path": str(CORRECTION_DOCUMENT),
            "sha256": CORRECTION_DOCUMENT_SHA256,
            "correction_scope": "simultaneous network publication batching only",
        },
        "prior_failed_run_boundaries": {
            "v1_date_representation_failure": dict(V1_FAILED_RUN_BOUNDARY),
            "v2_duplicate_network_availability_failure": dict(V2_FAILED_RUN_BOUNDARY),
        },
        "frozen_contract": {
            "clock_columns": list(evaluate.CLOCK_COLUMNS),
            "controls": list(evaluate.CONTROL_NAMES),
            "support_limits": dict(evaluate.SUPPORT_LIMITS),
            "novelty_limits": dict(evaluate.NOVELTY_LIMITS),
            "evaluation_start_utc": evaluate.EVALUATION_START.isoformat(),
            "train_end_utc": evaluate.TRAIN_END.isoformat(),
            "evaluation_end_utc": evaluate.EVALUATION_END.isoformat(),
            "macro_ttl_hours": int(evaluate.MACRO_TTL.total_seconds() // 3_600),
            "relay_deadline_hours": int(
                evaluate.RELAY_DEADLINE.total_seconds() // 3_600
            ),
            "hold_hours": int(evaluate.HOLD.total_seconds() // 3_600),
            "exposure_grid": "full [2021-01-01, 2024-01-01) UTC at 5m",
            "accepted_date_representations": [
                "YYYY-MM-DD",
                "YYYY-MM-DD 00:00:00",
            ],
            "network_publication_batch_rule": (
                "raw available_at monotonic non-decreasing; exact ties collapse "
                "to latest observation_date; final timestamps unique increasing"
            ),
        },
        "v3_freeze_boundary": {
            "source_value_rows_read_during_freeze": 0,
            "comparator_event_rows_read_during_freeze": 0,
            "source_vote_rows_derived_during_freeze": 0,
            "candidate_event_rows_derived_during_freeze": 0,
            "support_or_novelty_verdicts_produced_during_freeze": 0,
            "btc_market_rows_loaded_during_freeze": 0,
            "funding_rows_loaded_during_freeze": 0,
            "return_pnl_or_equity_rows_loaded_during_freeze": 0,
            "economic_simulations_run_during_freeze": 0,
            "network_calls": 0,
        },
        "mutable_parameters": [],
        "source_support_artifact_exists_before_v3_freeze": False,
        "source_support_clock_exists_before_v3_freeze": False,
        "next_action": "run the exact corrected source-only evaluator once",
    }
    return {**core, "manifest_hash": evaluate.canonical_hash(core)}


def validate_manifest(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if evaluate.canonical_hash(core) != payload.get("manifest_hash"):
        raise RuntimeError("CDLTR v3 evaluator freeze manifest hash mismatch")
    if payload.get("mutable_parameters") != []:
        raise RuntimeError("CDLTR v3 evaluator freeze permits parameter repair")
    boundary = payload.get("v3_freeze_boundary", {})
    if not isinstance(boundary, dict) or any(value != 0 for value in boundary.values()):
        raise RuntimeError("CDLTR v3 freeze opened incidence or outcomes")
    prior = payload.get("prior_failed_run_boundaries", {})
    if not isinstance(prior, dict):
        raise RuntimeError("CDLTR prior failure boundaries missing")
    if prior.get("v1_date_representation_failure") != V1_FAILED_RUN_BOUNDARY:
        raise RuntimeError("CDLTR v1 failure boundary drift")
    if prior.get("v2_duplicate_network_availability_failure") != V2_FAILED_RUN_BOUNDARY:
        raise RuntimeError("CDLTR v2 failure boundary drift")


def freeze(output: str | Path = DEFAULT_OUTPUT) -> tuple[str, dict[str, Any]]:
    if evaluate._repository_path(evaluate.DEFAULT_OUTPUT_REPORT).exists():
        raise RuntimeError("CDLTR source-support artifact predates v3 freeze")
    if evaluate._repository_path(evaluate.DEFAULT_OUTPUT_CLOCK).exists():
        raise RuntimeError("CDLTR source-support clock predates v3 freeze")
    commit, source_sha = _clean_committed_evaluator()
    payload = build_manifest(commit, source_sha)
    validate_manifest(payload)
    target = evaluate._repository_path(output)
    if target in evaluate._protected_paths():
        raise ValueError("CDLTR v3 freeze aliases a protected input")
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        validate_manifest(existing)
        if existing != payload:
            raise RuntimeError("refusing to overwrite CDLTR v3 evaluator freeze")
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
                "v3_freeze_source_rows_read": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
