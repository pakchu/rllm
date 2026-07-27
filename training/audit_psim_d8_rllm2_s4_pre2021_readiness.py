#!/usr/bin/env python3
"""Audit sealed PSIM S4 schedules before opening any 2021 outcome.

The S4 runner authorized a later preregistered 2021 evaluation after producing
the complete target schedules.  This audit is a stricter, outcome-blind
readiness check over those already sealed schedules.  It never reads market or
funding data and fail-closes when no primary can satisfy the fixed activity,
direction-balance, and neutral action-code invariance requirements.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

_BOOTSTRAP_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_BOOTSTRAP_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_REPO_ROOT))

from training import (  # noqa: E402
    preregister_psim_d8_rllm2_s4_semantic_fqi as prereg,
)


REPO_ROOT = _BOOTSTRAP_REPO_ROOT
PROTOCOL_VERSION = "psim_d8_rllm2_s4_pre2021_readiness_audit_v1"
STAGE_ID = "PSIM-D8-RLLM2-S4-PRE2021-READINESS-AUDIT"
AS_OF_DATE = "2026-07-27"
REJECT_ACTION = (
    "REJECT_PSIM_D8_RLLM2_S4_BEFORE_2021_OUTCOME_OPEN_"
    "HOLD_S4_2021_EVALUATION_START_NEW_PREREGISTERED_ATTEMPT"
)

DEFAULT_OUTPUT = Path(
    "results/psim_d8_rllm2_s4_pre2021_readiness_"
    "rejection_2026-07-27.json"
)
S4_PREREGISTRATION = prereg.DEFAULT_OUTPUT
S4_ATTEMPT = prereg.ATTEMPT_PATH
S4_RESULT = prereg.RESULT_PATH
S4_SCHEDULE_MANIFEST = prereg.SCHEDULE_MANIFEST_PATH
S4_BASE_SCHEDULES = prereg.SCHEDULE_PATH
S4_DELAYED_SCHEDULES = prereg.DELAYED_SCHEDULE_PATH
S4_TRANSITION_LEDGER = prereg.TRANSITION_LEDGER_PATH
S4_PCA = prereg.PCA_PATH
PRIOR_2021_TRANSFER_DOCUMENT = Path(
    "docs/bctp-cheap-policy-2021-transfer-result-2026-07-25.md"
)
PRIOR_2021_STAGE_SOURCE_MANIFEST = Path(
    "data/bctp_stage_sources/2021/source_manifest.json"
)

EXPECTED_FILE_SHA256 = {
    "preregistration": (
        "2c3be7d92f61248dd26d428a9793f1ec1e703e794e6454970c8a6c2804fc2a34"
    ),
    "attempt": (
        "717a140cec4a19ef057b2e44f7a508d25e977fd98b4447c3ea6f5dadd7e1e1fd"
    ),
    "result": (
        "ca7a9a42eda7719a7e59b9927bf9c8e754689007468ec235ac4c5e2a1c619c75"
    ),
    "schedule_manifest": (
        "cd5d06f43d98b36e4d69b5630afe5e98f8d807aa05707e571aa49ed7c6ff4f6c"
    ),
    "base_schedules": (
        "2f5b04e2514ca8b328fa6a92a45cea2828225090afc2427a58b6b778655394ba"
    ),
    "delayed_schedules": (
        "326d4e8c5e866b83b331d97ee54ea2abedce35abf85887539ee97220ed354484"
    ),
    "transition_ledger": (
        "07d465538d84648793ebbf302c54dace38ef71e88b22d7bd3a19fec500b99a7a"
    ),
    "pca": (
        "6a01d505ad2531683c9e8e9e0672456daf19a17a59b32d772fc857370210f0a2"
    ),
}
EXPECTED_DECODED_SHA256 = {
    "base_schedules": (
        "82ff2e46cafb7ed9565d01b7147bccb0999d29cf45925ca7f6447b3b4afd2f9a"
    ),
    "delayed_schedules": (
        "772ade268189e0d73a720739ccff5626085f5f91d5fdf5ced2b17a476ad6d34b"
    ),
}
EXPECTED_ATTEMPT_HASH = (
    "3f314bc9633351934b7b0c7ccc198f6ac48709e1ae7cf99ff4969900aaa553bc"
)
EXPECTED_RESULT_HASH = (
    "4253626d04e70f51dbd73df6c898c4a553b8df76b1ceb9c903c82e83ab2e3c09"
)
EXPECTED_SCHEDULE_MANIFEST_HASH = (
    "e90ce8de7c82aa5ca22365f6c50a42bd975e994202b67df751a12ef804de6f85"
)
EXPECTED_PREREGISTRATION_MANIFEST_HASH = (
    "81eaa2c6ea80e792a20d4abf6e341355d66ea3b0bed7cd359af040139ce91845"
)
EXPECTED_EXECUTION_COMMIT = "e078f7c81693e30354b05ab6190c174235a3f548"
EXPECTED_RUNNER_SHA256 = (
    "f1c25b37c0c30b6bc9e81d2cbd32a7e70f5c39ba0404eb5ff4d9db977dcbe1e4"
)
EXPECTED_COLUMNS = ("policy_id", "sequence_id", "entry_time", "target")
EXPECTED_TARGETS = ("TARGET_FLAT", "TARGET_LONG", "TARGET_SHORT")
EXPECTED_PRIOR_2021_TRANSFER_DOCUMENT_SHA256 = (
    "bd72c0839035860f289aec3ffcfe6e55721dd8a15bd65c93c3bb62475fa95d95"
)
EXPECTED_PRIOR_2021_STAGE_SOURCE_MANIFEST_SHA256 = (
    "1d12d8dad47eda810933ddce7ac2d911a7a4f85262ecc02e91e22df2488c6e2d"
)
MIN_NONFLAT_TARGET_ROWS = 80
MIN_DIRECTION_SHARE = 0.20


def repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def canonical_bytes(payload: Any, *, pretty: bool = False) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8") + (b"\n" if pretty else b"")


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with repository_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_exact_json(
    path: str | Path,
    *,
    expected_sha256: str,
    self_hash_field: str,
    expected_self_hash: str,
) -> dict[str, Any]:
    target = repository_path(path)
    if target.is_symlink() or not target.is_file():
        raise RuntimeError(f"PSIM S4 readiness evidence missing: {path}")
    if sha256_file(target) != expected_sha256:
        raise RuntimeError(f"PSIM S4 readiness evidence hash changed: {path}")
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"PSIM S4 readiness expected JSON object: {path}")
    observed = payload.get(self_hash_field)
    core = {
        key: value
        for key, value in payload.items()
        if key != self_hash_field
    }
    if observed != expected_self_hash or observed != canonical_hash(core):
        raise RuntimeError(f"PSIM S4 readiness self hash changed: {path}")
    return payload


def _verify_artifacts() -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    registration = _read_exact_json(
        S4_PREREGISTRATION,
        expected_sha256=EXPECTED_FILE_SHA256["preregistration"],
        self_hash_field="manifest_hash",
        expected_self_hash=EXPECTED_PREREGISTRATION_MANIFEST_HASH,
    )
    attempt = _read_exact_json(
        S4_ATTEMPT,
        expected_sha256=EXPECTED_FILE_SHA256["attempt"],
        self_hash_field="attempt_hash",
        expected_self_hash=EXPECTED_ATTEMPT_HASH,
    )
    result = _read_exact_json(
        S4_RESULT,
        expected_sha256=EXPECTED_FILE_SHA256["result"],
        self_hash_field="result_hash",
        expected_self_hash=EXPECTED_RESULT_HASH,
    )
    schedule_manifest = _read_exact_json(
        S4_SCHEDULE_MANIFEST,
        expected_sha256=EXPECTED_FILE_SHA256["schedule_manifest"],
        self_hash_field="manifest_hash",
        expected_self_hash=EXPECTED_SCHEDULE_MANIFEST_HASH,
    )
    expected_paths = {
        "base_schedules": S4_BASE_SCHEDULES,
        "delayed_schedules": S4_DELAYED_SCHEDULES,
        "transition_ledger": S4_TRANSITION_LEDGER,
        "pca": S4_PCA,
    }
    for name, path in expected_paths.items():
        if sha256_file(path) != EXPECTED_FILE_SHA256[name]:
            raise RuntimeError(f"PSIM S4 readiness artifact changed: {path}")
    for name, path in (
        ("base_schedules", S4_BASE_SCHEDULES),
        ("delayed_schedules", S4_DELAYED_SCHEDULES),
    ):
        decoded = gzip.decompress(repository_path(path).read_bytes())
        if hashlib.sha256(decoded).hexdigest() != EXPECTED_DECODED_SHA256[name]:
            raise RuntimeError(
                f"PSIM S4 readiness decoded schedule changed: {path}"
            )
    if (
        attempt.get("execution_commit") != EXPECTED_EXECUTION_COMMIT
        or result.get("execution_commit") != EXPECTED_EXECUTION_COMMIT
        or attempt.get("runner_sha256") != EXPECTED_RUNNER_SHA256
        or result.get("runner_sha256") != EXPECTED_RUNNER_SHA256
        or result.get("attempt_hash") != EXPECTED_ATTEMPT_HASH
        or result.get("target_schedule_seal", {}).get("manifest_hash")
        != EXPECTED_SCHEDULE_MANIFEST_HASH
        or schedule_manifest.get("attempt_hash") != EXPECTED_ATTEMPT_HASH
        or schedule_manifest.get("preregistration_manifest_hash")
        != EXPECTED_PREREGISTRATION_MANIFEST_HASH
        or schedule_manifest.get("selection_from_2020_training_metrics")
        is not False
    ):
        raise RuntimeError("PSIM S4 readiness execution binding changed")
    gate = registration["chronology_and_gates"]["2021_gate"]
    if (
        gate["minimum_nonflat_intervals"] != MIN_NONFLAT_TARGET_ROWS
        or gate["minimum_each_direction_share"] != MIN_DIRECTION_SHARE
    ):
        raise RuntimeError("PSIM S4 preregistered 2021 readiness gate changed")
    return registration, result, schedule_manifest


def _verify_global_2021_history() -> dict[str, Any]:
    transfer = repository_path(PRIOR_2021_TRANSFER_DOCUMENT)
    source_manifest_path = repository_path(
        PRIOR_2021_STAGE_SOURCE_MANIFEST
    )
    if (
        sha256_file(transfer)
        != EXPECTED_PRIOR_2021_TRANSFER_DOCUMENT_SHA256
        or sha256_file(source_manifest_path)
        != EXPECTED_PRIOR_2021_STAGE_SOURCE_MANIFEST_SHA256
    ):
        raise RuntimeError("PSIM S4 global 2021 history evidence changed")
    transfer_text = transfer.read_text(encoding="utf-8")
    source_manifest = json.loads(
        source_manifest_path.read_text(encoding="utf-8")
    )
    required_fragments = (
        "BCTP-12H cheap-policy 2021 transfer result",
        "2021 transfer report",
        "Categorical linear FQI",
        "Categorical ridge FQI",
        "ExtraTrees FQI",
    )
    if (
        not all(fragment in transfer_text for fragment in required_fragments)
        or source_manifest.get("stage") != "2021"
        or source_manifest.get("validation", {}).get("market_rows")
        != 105_120
        or source_manifest.get("validation", {}).get("funding_rows") != 1_095
    ):
        raise RuntimeError("PSIM S4 global 2021 history content changed")
    return {
        "globally_pristine_2021_claim_allowed": False,
        "prior_2021_transfer_result_exists": True,
        "prior_2021_transfer_result_document": (
            PRIOR_2021_TRANSFER_DOCUMENT.as_posix()
        ),
        "prior_2021_transfer_result_document_sha256": (
            EXPECTED_PRIOR_2021_TRANSFER_DOCUMENT_SHA256
        ),
        "prior_2021_stage_source_manifest": (
            PRIOR_2021_STAGE_SOURCE_MANIFEST.as_posix()
        ),
        "prior_2021_stage_source_manifest_sha256": (
            EXPECTED_PRIOR_2021_STAGE_SOURCE_MANIFEST_SHA256
        ),
        "prior_2021_stage_market_rows": 105_120,
        "prior_2021_stage_funding_rows": 1_095,
        "interpretation": (
            "S4 itself did not read 2021 outcomes, but the repository and "
            "research history already contain 2021 BTCUSDT transfer metrics"
        ),
    }


def _schedule_diagnostics(
    schedules: pd.DataFrame,
    primary_id: str,
) -> dict[str, Any]:
    primary = (
        schedules.loc[schedules["policy_id"] == primary_id]
        .sort_values("sequence_id")
        .reset_index(drop=True)
    )
    permutation_id = f"{primary_id}_action_code_permutation"
    permutation = (
        schedules.loc[schedules["policy_id"] == permutation_id]
        .sort_values("sequence_id")
        .reset_index(drop=True)
    )
    if len(primary) != 365 or len(permutation) != 365:
        raise RuntimeError("PSIM S4 primary schedule length changed")
    if not primary["sequence_id"].equals(permutation["sequence_id"]):
        raise RuntimeError("PSIM S4 action permutation sequence changed")
    if not primary["entry_time"].equals(permutation["entry_time"]):
        raise RuntimeError("PSIM S4 action permutation clock changed")
    counts = {
        target: int((primary["target"] == target).sum())
        for target in EXPECTED_TARGETS
    }
    nonflat = counts["TARGET_LONG"] + counts["TARGET_SHORT"]
    long_share = counts["TARGET_LONG"] / nonflat if nonflat else 0.0
    short_share = counts["TARGET_SHORT"] / nonflat if nonflat else 0.0
    mismatches = primary["target"] != permutation["target"]
    mismatch_rows = [
        {
            "sequence_id": str(primary.loc[index, "sequence_id"]),
            "entry_time": str(primary.loc[index, "entry_time"]),
            "primary_target": str(primary.loc[index, "target"]),
            "permuted_target": str(permutation.loc[index, "target"]),
        }
        for index in primary.index[mismatches]
    ]
    activity_pass = nonflat >= MIN_NONFLAT_TARGET_ROWS
    direction_pass = (
        long_share >= MIN_DIRECTION_SHARE
        and short_share >= MIN_DIRECTION_SHARE
    )
    action_code_invariance_pass = not mismatch_rows
    return {
        "target_counts": counts,
        "nonflat_target_rows": nonflat,
        "long_share_of_nonflat_targets": long_share,
        "short_share_of_nonflat_targets": short_share,
        "activity_pass": activity_pass,
        "direction_balance_pass": direction_pass,
        "action_code_permutation_policy_id": permutation_id,
        "action_code_permutation_exact_target_identity": (
            action_code_invariance_pass
        ),
        "action_code_permutation_mismatch_count": len(mismatch_rows),
        "action_code_permutation_mismatches": mismatch_rows,
        "eligible_for_2021_outcome_open": (
            activity_pass
            and direction_pass
            and action_code_invariance_pass
        ),
    }


def build_audit() -> dict[str, Any]:
    registration, result, schedule_manifest = _verify_artifacts()
    global_history = _verify_global_2021_history()
    schedules = pd.read_csv(repository_path(S4_BASE_SCHEDULES))
    delayed = pd.read_csv(repository_path(S4_DELAYED_SCHEDULES))
    if tuple(schedules.columns) != EXPECTED_COLUMNS:
        raise RuntimeError("PSIM S4 base schedule columns changed")
    if tuple(delayed.columns) != EXPECTED_COLUMNS:
        raise RuntimeError("PSIM S4 delayed schedule columns changed")
    policy_ids = tuple(schedule_manifest["policy_family_ids"])
    counts = schedules.groupby("policy_id", sort=False).size().to_dict()
    if (
        tuple(policy_ids) != prereg.POLICY_FAMILY_IDS
        or schedules.shape != (9_125, 4)
        or delayed.shape != (730, 4)
        or counts != {policy_id: 365 for policy_id in policy_ids}
        or schedules.duplicated(["policy_id", "sequence_id"]).any()
        or delayed.duplicated(["policy_id", "sequence_id"]).any()
        or set(schedules["target"]) != set(EXPECTED_TARGETS)
        or set(delayed["target"]) != set(EXPECTED_TARGETS)
        or schedules["entry_time"].min() != "2021-01-01T12:05:00Z"
        or schedules["entry_time"].max() != "2021-12-31T12:05:00Z"
        or delayed["entry_time"].min() != "2021-01-01T12:10:00Z"
        or delayed["entry_time"].max() != "2021-12-31T12:10:00Z"
    ):
        raise RuntimeError("PSIM S4 sealed schedule roster changed")
    delayed_counts = delayed.groupby("policy_id", sort=False).size().to_dict()
    if delayed_counts != {
        primary_id: 365 for primary_id in prereg.PRIMARY_POLICY_IDS
    }:
        raise RuntimeError("PSIM S4 delayed primary roster changed")
    boundary = result["access_boundary"]
    outcomes_remain_sealed = (
        boundary["2021_outcomes_opened"] is False
        and boundary["2021_market_or_funding_paths_read"] == []
        and boundary["2021_reward_rows_created"] == 0
        and boundary["2021_economic_metrics_computed"] == 0
        and boundary["2022_or_2023_outcomes_opened"] is False
        and schedule_manifest["outcome_boundary"]
        ["2021_market_or_funding_payload_opened"]
        is False
        and schedule_manifest["outcome_boundary"]
        ["2021_reward_rows_created"]
        == 0
        and schedule_manifest["outcome_boundary"]
        ["2021_economic_metrics_computed"]
        == 0
    )
    if not outcomes_remain_sealed:
        raise RuntimeError("PSIM S4 readiness outcome boundary changed")
    diagnostics = {
        primary_id: _schedule_diagnostics(schedules, primary_id)
        for primary_id in prereg.PRIMARY_POLICY_IDS
    }
    eligible = [
        primary_id
        for primary_id, values in diagnostics.items()
        if values["eligible_for_2021_outcome_open"]
    ]
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "stage_id": STAGE_ID,
        "as_of_date": AS_OF_DATE,
        "predecessor": {
            "stage_id": result["stage_id"],
            "preregistration_sha256": (
                EXPECTED_FILE_SHA256["preregistration"]
            ),
            "preregistration_manifest_hash": (
                EXPECTED_PREREGISTRATION_MANIFEST_HASH
            ),
            "execution_commit": EXPECTED_EXECUTION_COMMIT,
            "runner_sha256": EXPECTED_RUNNER_SHA256,
            "attempt_sha256": EXPECTED_FILE_SHA256["attempt"],
            "attempt_hash": EXPECTED_ATTEMPT_HASH,
            "result_sha256": EXPECTED_FILE_SHA256["result"],
            "result_hash": EXPECTED_RESULT_HASH,
            "schedule_manifest_sha256": (
                EXPECTED_FILE_SHA256["schedule_manifest"]
            ),
            "schedule_manifest_hash": EXPECTED_SCHEDULE_MANIFEST_HASH,
            "predecessor_open_2021_authorization": (
                result["open_2021_outcomes_authorized"]
            ),
        },
        "artifact_verification": {
            "all_exact_hashes_verified": True,
            "base_schedule_rows": len(schedules),
            "base_schedule_policy_count": len(policy_ids),
            "base_schedule_first_entry_time": schedules["entry_time"].min(),
            "base_schedule_last_entry_time": schedules["entry_time"].max(),
            "delayed_schedule_rows": len(delayed),
            "delayed_schedule_policy_count": len(delayed_counts),
            "delayed_schedule_first_entry_time": delayed["entry_time"].min(),
            "delayed_schedule_last_entry_time": delayed["entry_time"].max(),
            "duplicate_policy_sequence_rows": 0,
            "target_domain": list(EXPECTED_TARGETS),
        },
        "readiness_contract": {
            "minimum_nonflat_target_rows": MIN_NONFLAT_TARGET_ROWS,
            "minimum_long_share_of_nonflat_targets": MIN_DIRECTION_SHARE,
            "minimum_short_share_of_nonflat_targets": MIN_DIRECTION_SHARE,
            "source": (
                "S4 preregistration chronology_and_gates.2021_gate"
            ),
            "must_beat_strongest_nonsemantic_control": registration[
                "chronology_and_gates"
            ]["2021_gate"]["must_beat_strongest_nonsemantic_control"],
            "familywise_p_max_strictly_below": registration[
                "chronology_and_gates"
            ]["2021_gate"]["familywise_p_max_strictly_below"],
            "action_code_permutation_requires_exact_target_identity": True,
            "minimum_eligible_primary_count_before_outcome_open": 1,
            "contract_fixed_before_any_2021_outcome_open": True,
        },
        "primary_schedule_diagnostics": diagnostics,
        "eligible_primary_ids": eligible,
        "eligible_primary_count": len(eligible),
        "decision": "reject" if not eligible else "pass",
        "terminal_action": REJECT_ACTION if not eligible else "UNEXPECTED_PASS",
        "audit_authorizes_open_2021_outcomes": bool(eligible),
        "predecessor_2021_authorization_exercised": False,
        "global_2021_outcome_history": global_history,
        "access_boundary": {
            "audit_market_or_funding_paths_read": [],
            "audit_market_rows_parsed": 0,
            "audit_funding_rows_parsed": 0,
            "audit_rewards_created": 0,
            "audit_economic_metrics_computed": 0,
            "s4_protocol_2021_outcomes_remain_sealed": outcomes_remain_sealed,
            "s4_protocol_2022_or_later_outcomes_remain_sealed": True,
            "globally_pristine_2021_outcome_claim_allowed": False,
        },
        "scientific_disposition": {
            "in_sample_2020_metrics_are_alpha_evidence": False,
            "2021_oos_metrics_exist": False,
            "candidate_promoted": False,
            "s4_rerun_authorized": False,
            "separate_successor_preregistration_authorized": True,
            "2021_successor_role": (
                "protocol-isolated report-only transfer with global "
                "contamination disclosure"
            ),
            "successor_must_not_use_known_2021_metrics_for_design": True,
            "globally_pristine_evidence_requires_forward_or_live_data": True,
            "successor_required_repairs": [
                (
                    "remove bull-year direction collapse without using "
                    "2021 outcomes"
                ),
                (
                    "make neutral action-code permutation exactly invariant "
                    "for every estimator"
                ),
                (
                    "pass the same activity and direction readiness gate "
                    "before opening 2021"
                ),
            ],
        },
    }
    if core["decision"] != "reject":
        raise RuntimeError("PSIM S4 readiness unexpectedly passed")
    return {**core, "result_hash": canonical_hash(core)}


def write_once(path: str | Path, payload: dict[str, Any]) -> str:
    target = repository_path(path)
    raw = canonical_bytes(payload, pretty=True)
    if target.exists():
        if target.is_symlink() or not target.is_file():
            raise RuntimeError("PSIM S4 readiness output path changed")
        if target.read_bytes() != raw:
            raise RuntimeError("PSIM S4 readiness output is write-once")
        return "verified_existing"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    return "created"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    payload = build_audit()
    status = (
        "validated"
        if args.validate_only
        else write_once(args.output, payload)
    )
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "eligible_primary_count": payload[
                    "eligible_primary_count"
                ],
                "result_hash": payload["result_hash"],
                "status": status,
                "terminal_action": payload["terminal_action"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
