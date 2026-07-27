#!/usr/bin/env python3
"""Preregister PSIM S6 all-action-mean residual semantic Ridge FQI.

S5 removed the long-vs-short prior and passed direction/invariance checks but
sealed only 74 non-flat 2021 targets versus the fixed minimum of 80.  S6 does
not lower that gate or impose a target quota.  It removes the unconditional
mean of every action within each current position, preserving only
state-conditional reward deviations for one deterministic semantic Ridge FQI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

_BOOTSTRAP_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_BOOTSTRAP_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_REPO_ROOT))

from training import (  # noqa: E402
    preregister_psim_d8_rllm2_s4_semantic_fqi as s4,
)
from training import (  # noqa: E402
    preregister_psim_d8_rllm2_s5_direction_residual_ridge_fqi as s5,
)


REPO_ROOT = _BOOTSTRAP_REPO_ROOT
PROTOCOL_VERSION = (
    "psim_d8_rllm2_s6_action_mean_residual_ridge_fqi_preregistration_v1"
)
STAGE_ID = "PSIM-D8-RLLM2-S6-ACTION-MEAN-RESIDUAL-RIDGE-FQI"
AS_OF_DATE = "2026-07-27"

PRIMARY_POLICY_ID = "semantic_ridge_action_mean_residual_fqi"
CONTROL_POLICY_IDS = (
    f"{PRIMARY_POLICY_ID}_action_code_permutation",
    f"{PRIMARY_POLICY_ID}_direction_flip",
    f"{PRIMARY_POLICY_ID}_circular_21_reward",
    f"{PRIMARY_POLICY_ID}_within_month_shuffled_reward",
    f"{PRIMARY_POLICY_ID}_current_position_only",
    f"{PRIMARY_POLICY_ID}_masked_semantic_embedding",
    f"{PRIMARY_POLICY_ID}_metadata_frontmatter_only",
)
POLICY_FAMILY_IDS = (PRIMARY_POLICY_ID, *CONTROL_POLICY_IDS)
COMBINED_2021_FAMILY_IDS = (
    *s4.POLICY_FAMILY_IDS,
    *s5.POLICY_FAMILY_IDS,
    *POLICY_FAMILY_IDS,
)
DEGENERATE_CONTROL_IDS = (
    "always_flat",
    "always_long",
    "always_short",
    "previous_target_persistence",
    f"{PRIMARY_POLICY_ID}_current_position_only",
    f"{PRIMARY_POLICY_ID}_masked_semantic_embedding",
    f"{PRIMARY_POLICY_ID}_metadata_frontmatter_only",
)

DEFAULT_OUTPUT = Path(
    "results/psim_d8_rllm2_s6_action_mean_residual_ridge_fqi_"
    "preregistration_2026-07-27.json"
)
RUNNER_PATH = Path(
    "training/run_psim_d8_rllm2_s6_train2020_"
    "action_mean_residual_ridge_fqi.py"
)
ATTEMPT_PATH = Path(
    "results/psim_d8_rllm2_s6_train2020_action_mean_residual_ridge_fqi_"
    "attempt_2026-07-27.json"
)
RESULT_PATH = Path(
    "results/psim_d8_rllm2_s6_train2020_action_mean_residual_ridge_fqi_"
    "pre2021_gate_2026-07-27.json"
)
RESIDUAL_LEDGER_PATH = Path(
    "data/psim_d8_rllm2_s6_action_mean_residual_reward_ledger_2020.csv.gz"
)
SCHEDULE_PATH = Path(
    "data/psim_d8_rllm2_s6_action_mean_residual_ridge_fqi_"
    "schedules_2021.csv.gz"
)
DELAYED_SCHEDULE_PATH = Path(
    "data/psim_d8_rllm2_s6_action_mean_residual_ridge_fqi_"
    "delayed_schedule_2021.csv.gz"
)
SCHEDULE_MANIFEST_PATH = Path(
    "results/psim_d8_rllm2_s6_action_mean_residual_ridge_fqi_"
    "schedule_gate_2021_2026-07-27.json"
)

S5_TERMINAL_RECORD_COMMIT = "c3f8e9eb377827c311c40c5dbc2b5024e9029c45"
S5_PREREGISTRATION_SHA256 = (
    "95ebcacf6cf7eaec2d60933fe6a54852a9c3fb74e4d7e50972fe99da9500cace"
)
S5_PREREGISTRATION_MANIFEST_HASH = (
    "96fa6b761154beba09fcbdaf826a408b7a5549e067eef9ff68a11c4bed887c54"
)
S5_ATTEMPT_SHA256 = (
    "ab59c4fe359fe4a8a3b05c7a128999489ba2679bf453e79c1bbb9428baaa0d42"
)
S5_ATTEMPT_HASH = (
    "f0c4dd9ac216ba9458a4541cc01af2290ff51d1bc4caa54d2b873e67d84a7efc"
)
S5_RESULT_SHA256 = (
    "aabc479e030a3a08ddd07047ccb25ce872425655ca10d877b8e78c4687bf73a7"
)
S5_RESULT_HASH = (
    "2331b46e3847d80200c88b3f3522ef35f3f710dead5ea6dca85575f249b0541e"
)
S5_SCHEDULE_MANIFEST_SHA256 = (
    "76f66a4353d0d8e3bb8f2aa1671f6a5416dd7f344c22bdeed63a485f984a1d60"
)
S5_SCHEDULE_MANIFEST_HASH = (
    "ca0ef82f5e9886288942c2ff8b115fa9e64eeee143b358ebd90ceb32b7fb5022"
)
S5_BASE_SCHEDULES_SHA256 = (
    "ed5ed208104242f571f4a07a53a815a1db2742957f43e9fcbecd9c18221a3833"
)
S5_DELAYED_SCHEDULE_SHA256 = (
    "e0580cf4bb1a0a3a05710945abfe20edb9c05bb47a30c34aafdb43556d85e8b5"
)
S5_RESIDUAL_LEDGER_SHA256 = (
    "b395aa4cb0b0288eb7287eea297bc3b1b416e1326b379a48952a6fb8489fc19b"
)
S5_RUNNER_SHA256 = (
    "320382ccea359704a5fbe3cc2599c981dee456f26fd66c4f5a98c67274e179f2"
)
S5_CORE_SHA256 = (
    "bd3fc76b640f783270c8486128a00af8a580a3de8a989d814dbe534a8d8af0bd"
)
S5_TERMINAL_DOCUMENT = Path(
    "docs/psim-d8-rllm2-s5-direction-residual-ridge-fqi-"
    "terminal-rejection-2026-07-27.md"
)
S5_TERMINAL_DOCUMENT_SHA256 = (
    "84401ba6022bce7c6e4b0394c5e62fba1b2da10f3095d3ecd01e6726d3493c0d"
)

RIDGE_ALPHA = 100.0
BELLMAN_ITERATIONS = 25
DISCOUNT = 0.99
FIXED_SHUFFLE_SEED = 20_260_727
FIXED_CIRCULAR_REWARD_SHIFT = 21
MIN_NONFLAT_TARGET_ROWS = 80
MIN_DIRECTION_SHARE = 0.20
FAMILYWISE_P_MAX = 0.25


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
        raise RuntimeError(f"PSIM S6 predecessor evidence missing: {path}")
    if sha256_file(target) != expected_sha256:
        raise RuntimeError(f"PSIM S6 predecessor evidence changed: {path}")
    payload = json.loads(target.read_text(encoding="utf-8"))
    core = {
        key: value
        for key, value in payload.items()
        if key != self_hash_field
    }
    if (
        payload.get(self_hash_field) != expected_self_hash
        or payload.get(self_hash_field) != canonical_hash(core)
    ):
        raise RuntimeError(f"PSIM S6 predecessor self hash changed: {path}")
    return payload


def validate_s5_rejection() -> dict[str, Any]:
    registration = _read_exact_json(
        s5.DEFAULT_OUTPUT,
        expected_sha256=S5_PREREGISTRATION_SHA256,
        self_hash_field="manifest_hash",
        expected_self_hash=S5_PREREGISTRATION_MANIFEST_HASH,
    )
    attempt = _read_exact_json(
        s5.ATTEMPT_PATH,
        expected_sha256=S5_ATTEMPT_SHA256,
        self_hash_field="attempt_hash",
        expected_self_hash=S5_ATTEMPT_HASH,
    )
    result = _read_exact_json(
        s5.RESULT_PATH,
        expected_sha256=S5_RESULT_SHA256,
        self_hash_field="result_hash",
        expected_self_hash=S5_RESULT_HASH,
    )
    schedule_manifest = _read_exact_json(
        s5.SCHEDULE_MANIFEST_PATH,
        expected_sha256=S5_SCHEDULE_MANIFEST_SHA256,
        self_hash_field="manifest_hash",
        expected_self_hash=S5_SCHEDULE_MANIFEST_HASH,
    )
    for path, expected in (
        (s5.SCHEDULE_PATH, S5_BASE_SCHEDULES_SHA256),
        (s5.DELAYED_SCHEDULE_PATH, S5_DELAYED_SCHEDULE_SHA256),
        (s5.RESIDUAL_LEDGER_PATH, S5_RESIDUAL_LEDGER_SHA256),
        (S5_TERMINAL_DOCUMENT, S5_TERMINAL_DOCUMENT_SHA256),
        (s5.RUNNER_PATH, S5_RUNNER_SHA256),
        (
            Path("training/psim_direction_residual_fqi.py"),
            S5_CORE_SHA256,
        ),
        (
            s4.TRANSITION_LEDGER_PATH,
            s5.S4_TRANSITION_LEDGER_SHA256,
        ),
        (s4.PCA_PATH, s5.S4_PCA_SHA256),
    ):
        if sha256_file(path) != expected:
            raise RuntimeError(f"PSIM S6 predecessor artifact changed: {path}")
    readiness = result.get("schedule_readiness", {})
    gates = readiness.get("gates", {})
    boundary = result.get("access_boundary", {})
    if (
        result.get("decision") != "reject"
        or result.get("execution_commit")
        != "c814807c7a25b0a9ec28c2bd7db4bbd5ac6ac520"
        or result.get("runner_sha256") != S5_RUNNER_SHA256
        or result.get("attempt_hash") != S5_ATTEMPT_HASH
        or result.get("authorize_separate_2021_transfer_preregistration")
        is not False
        or result.get("authorize_2022_or_later_outcomes") is not False
        or result.get("qlora_authorized") is not False
        or readiness.get("passed") is not False
        or readiness.get("nonflat_target_rows") != 74
        or readiness.get("target_counts")
        != {
            "TARGET_FLAT": 291,
            "TARGET_LONG": 44,
            "TARGET_SHORT": 30,
        }
        or gates.get("minimum_nonflat_target_rows") is not False
        or sum(not bool(value) for value in gates.values()) != 1
        or schedule_manifest.get("decision") != "reject"
        or schedule_manifest.get("schedule_readiness") != readiness
        or boundary.get("raw_market_or_funding_paths_read") != []
        or boundary.get("2021_market_or_funding_paths_read") != []
        or boundary.get("2021_reward_rows_created") != 0
        or boundary.get("2021_economic_metrics_computed") != 0
        or boundary.get("2021_policy_specific_outcomes_opened") is not False
        or boundary.get("2022_or_later_outcomes_opened") is not False
    ):
        raise RuntimeError("PSIM S6 predecessor rejection boundary changed")
    return {
        "registration": registration,
        "attempt": attempt,
        "result": result,
        "schedule_manifest": schedule_manifest,
    }


def build_preregistration() -> dict[str, Any]:
    predecessor = validate_s5_rejection()
    s5_registration = predecessor["registration"]
    source = s5_registration["frozen_source_and_outcome_artifacts"]
    future_gate = s5_registration["future_2021_transfer_gate"]
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "candidate": {
            "id": STAGE_ID,
            "predecessor": s5.STAGE_ID,
            "stage": (
                "reuse_original_2020_ledger_fit_action_mean_residual_"
                "ridge_and_seal_2021"
            ),
            "profitability_claim": False,
            "single_promotable_primary": PRIMARY_POLICY_ID,
            "selection_from_2020_metrics": False,
            "selection_from_known_2021_metrics": False,
            "globally_pristine_2021_claim": False,
            "s5_activity_threshold_lowered": False,
            "target_quota_or_q_margin_calibration": False,
        },
        "predecessor_terminal_evidence": {
            "terminal_record_commit": S5_TERMINAL_RECORD_COMMIT,
            "s5_preregistration": {
                "path": s5.DEFAULT_OUTPUT.as_posix(),
                "sha256": S5_PREREGISTRATION_SHA256,
                "manifest_hash": S5_PREREGISTRATION_MANIFEST_HASH,
            },
            "s5_attempt": {
                "path": s5.ATTEMPT_PATH.as_posix(),
                "sha256": S5_ATTEMPT_SHA256,
                "attempt_hash": S5_ATTEMPT_HASH,
            },
            "s5_result": {
                "path": s5.RESULT_PATH.as_posix(),
                "sha256": S5_RESULT_SHA256,
                "result_hash": S5_RESULT_HASH,
                "decision": "reject",
                "failed_gate": "minimum_nonflat_target_rows",
                "observed_nonflat_target_rows": 74,
                "required_nonflat_target_rows": 80,
            },
            "s5_schedule_manifest": {
                "path": s5.SCHEDULE_MANIFEST_PATH.as_posix(),
                "sha256": S5_SCHEDULE_MANIFEST_SHA256,
                "manifest_hash": S5_SCHEDULE_MANIFEST_HASH,
            },
            "s5_terminal_document": {
                "path": S5_TERMINAL_DOCUMENT.as_posix(),
                "sha256": S5_TERMINAL_DOCUMENT_SHA256,
            },
            "s5_runner_sha256": S5_RUNNER_SHA256,
            "s5_core_sha256": S5_CORE_SHA256,
        },
        "frozen_source_and_outcome_artifacts": {
            "model": source["model"],
            "selector": source["selector"],
            "source_rows": source["source_rows"],
            "embeddings": source["embeddings"],
            "relation_logits": source["relation_logits"],
            "original_transition_ledger_2020": {
                **source["transition_ledger_2020"],
                "reuse_s5_residual_rewards": False,
            },
            "pca32_2020": source["pca32_2020"],
            "s4_base_schedules_2021": source[
                "s4_base_schedules_2021"
            ],
            "s5_base_schedules_2021": {
                "path": s5.SCHEDULE_PATH.as_posix(),
                "sha256": S5_BASE_SCHEDULES_SHA256,
                "rows": 2_920,
                "role": "frozen_nonpromotable_comparison_family",
            },
        },
        "action_mean_residual_reward_contract": {
            "input": (
                "exact original S4 strict 2020 reward tensor reconstructed "
                "from the 3,288-row transition ledger"
            ),
            "positions": [
                "POSITION_SHORT",
                "POSITION_FLAT",
                "POSITION_LONG",
            ],
            "actions": [
                "TARGET_FLAT",
                "TARGET_SHORT",
                "TARGET_LONG",
            ],
            "action_mean_formula": (
                "for each current position p and action a, mu[p,a] = "
                "mean_t R[t,p,a] over all finite reachable 2020 state rows"
            ),
            "position_grand_mean_formula": (
                "bar_mu[p] = (mu[p,flat] + mu[p,short] + mu[p,long]) / 3"
            ),
            "transform": (
                "R_star[t,p,a] = R[t,p,a] - mu[p,a] + bar_mu[p]"
            ),
            "post_transform_invariant": (
                "within each current position, mean residual rewards of "
                "flat, short, and long are equal within tolerance 1e-15"
            ),
            "fit_rows": "all 366 2020 source rows",
            "clipping": False,
            "scaling": False,
            "threshold_search": False,
            "hyperparameter_search": False,
            "target_quota": False,
            "q_margin_calibration": False,
            "s5_delta_reuse": False,
            "reward_control_order": (
                "residualize all three action means first; circular and "
                "within-month controls then permute complete residual rows"
            ),
        },
        "fitted_q_contract": {
            "primary_policy_id": PRIMARY_POLICY_ID,
            "algorithm": "ridge",
            "alpha": RIDGE_ALPHA,
            "unpenalized_intercept": True,
            "features": (
                "exact frozen S4 PCA32 semantic coordinates plus separate "
                "current-position one-hot"
            ),
            "bellman_iterations": BELLMAN_ITERATIONS,
            "discount": DISCOUNT,
            "action_order": [
                "TARGET_FLAT",
                "TARGET_SHORT",
                "TARGET_LONG",
            ],
            "target_account_gross": [0.0, -0.5, 0.5],
            "tie_break": [
                "TARGET_FLAT",
                "current_target",
                "TARGET_SHORT",
                "TARGET_LONG",
            ],
            "extra_trees_authorized": False,
            "qlora_authorized": False,
            "model_load_or_new_gemma_forward_authorized": False,
        },
        "control_family": {
            "new_policy_family_ids": list(POLICY_FAMILY_IDS),
            "new_control_policy_ids": list(CONTROL_POLICY_IDS),
            "combined_2021_family_ids": list(COMBINED_2021_FAMILY_IDS),
            "combined_2021_family_count": len(COMBINED_2021_FAMILY_IDS),
            "s4_and_s5_family_role": (
                "frozen_nonpromotable_comparison_controls"
            ),
            "action_code_permutation_requires_exact_target_identity": True,
            "fixed_circular_reward_shift": FIXED_CIRCULAR_REWARD_SHIFT,
            "fixed_within_month_shuffle_seed": FIXED_SHUFFLE_SEED,
            "single_predeclared_family_for_weekly_max_stat": True,
        },
        "pre2021_schedule_readiness_gate": {
            "must_run_before_any_2021_market_or_funding_read": True,
            "base_primary_schedule_rows": 365,
            "delayed_primary_schedule_rows": 365,
            "minimum_nonflat_target_rows": MIN_NONFLAT_TARGET_ROWS,
            "minimum_long_share_of_nonflat_targets": MIN_DIRECTION_SHARE,
            "minimum_short_share_of_nonflat_targets": MIN_DIRECTION_SHARE,
            "delayed_target_counts_must_equal_base": True,
            "action_code_permutation_exact_target_identity": True,
            "minimum_target_hamming_distance_from_each_degenerate_control": 1,
            "degenerate_control_ids": list(DEGENERATE_CONTROL_IDS),
            "minimum_hamming_distance_from_s5_primary": 1,
            "failure_action": (
                "TERMINAL_REJECT_S6_WITHOUT_2021_MARKET_FUNDING_REWARD_"
                "OR_METRIC_ACCESS"
            ),
            "pass_action": (
                "SEAL_S6_2021_SCHEDULE_AUTHORIZE_SEPARATE_REPORT_ONLY_"
                "2021_TRANSFER_PREREGISTRATION"
            ),
        },
        "future_2021_transfer_gate": {
            "absolute_return_positive": True,
            "stress_return_positive": True,
            "delay_return_positive": True,
            "both_half_returns_positive": True,
            "cagr_to_strict_mdd_minimum": 1.0,
            "minimum_nonflat_intervals": MIN_NONFLAT_TARGET_ROWS,
            "minimum_each_direction_share": MIN_DIRECTION_SHARE,
            "must_beat_strongest_nonsemantic_control": True,
            "familywise_p_max_strictly_below": FAMILYWISE_P_MAX,
            "role": future_gate["role"],
            "single_promotable_primary": PRIMARY_POLICY_ID,
            "base_cost_rate": 0.0006,
            "stress_cost_rate": 0.0010,
            "execution_delay": "one complete 5m bar",
            "weekly_familywise_family_ids": list(COMBINED_2021_FAMILY_IDS),
            "selection_or_repair_from_2021_metrics": False,
            "success_is_live_promotion": False,
            "success_requires_forward_or_live_confirmation": True,
        },
        "global_outcome_contamination_disclosure": {
            **s5_registration["global_outcome_contamination_disclosure"],
            "s6_policy_specific_2021_metrics_already_exist": False,
            "known_unrelated_2021_metrics_may_not_inform_s6_design": True,
            "known_s5_schedule_counts_are_outcome_blind": True,
            "2021_may_be_called_globally_pristine": False,
        },
        "chronology": [
            "verify exact S5 terminal rejection and artifact hashes",
            "write S6 attempt before parsing the original 2020 ledger",
            "read no raw market or funding payload",
            "reconstruct the original strict 2020 reward tensor",
            "compute the fixed 3x3 action-mean baseline without search",
            "fit the complete eight-policy Ridge family on 2020 only",
            "seal 2021 schedules before any 2021 outcome read",
            "run the unchanged activity/direction/invariance readiness gate",
            (
                "terminally reject on failure; on pass authorize only a "
                "separate report-only 2021 transfer preregistration"
            ),
        ],
        "artifact_contract": {
            "runner": RUNNER_PATH.as_posix(),
            "attempt": ATTEMPT_PATH.as_posix(),
            "result": RESULT_PATH.as_posix(),
            "residual_reward_ledger_2020": (
                RESIDUAL_LEDGER_PATH.as_posix()
            ),
            "base_schedules_2021": SCHEDULE_PATH.as_posix(),
            "delayed_primary_schedule_2021": (
                DELAYED_SCHEDULE_PATH.as_posix()
            ),
            "schedule_gate_manifest_2021": (
                SCHEDULE_MANIFEST_PATH.as_posix()
            ),
            "fixed_paths_no_output_override": True,
            "deterministic_csv_gzip_and_json": True,
            "write_once": True,
            "result_published_last": True,
        },
        "execution_contract": {
            "clean_head_equals_origin_main_required": True,
            "runner_must_be_committed_and_hash_bound_before_execution": True,
            "attempt_written_before_original_2020_ledger_parse": True,
            "all_outputs_written_before_result": True,
            "exact_existing_result_resume_validation_required": True,
            "no_raw_market_or_funding_payload_read": True,
            "no_2021_stage_source_payload_read": True,
            "no_2021_reward_or_economic_metric_creation": True,
            "no_model_load_or_new_model_forward": True,
            "no_qlora": True,
        },
        "access_boundary": {
            "source_files_read": sorted(
                {
                    s5.DEFAULT_OUTPUT.as_posix(),
                    s5.ATTEMPT_PATH.as_posix(),
                    s5.RESULT_PATH.as_posix(),
                    s5.SCHEDULE_MANIFEST_PATH.as_posix(),
                    s5.SCHEDULE_PATH.as_posix(),
                    s5.DELAYED_SCHEDULE_PATH.as_posix(),
                    s5.RESIDUAL_LEDGER_PATH.as_posix(),
                    S5_TERMINAL_DOCUMENT.as_posix(),
                    s4.TRANSITION_LEDGER_PATH.as_posix(),
                    s4.PCA_PATH.as_posix(),
                }
            ),
            "raw_market_or_funding_paths_read": [],
            "original_2020_outcome_artifact_bytes_hashed": True,
            "original_2020_transition_reward_rows_parsed": 0,
            "2021_market_or_funding_paths_read": [],
            "2021_market_rows_parsed": 0,
            "2021_funding_rows_parsed": 0,
            "2021_rewards_created": 0,
            "2021_economic_metrics_computed": 0,
            "2021_policy_specific_outcomes_opened": False,
            "2022_or_later_outcomes_opened": False,
            "model_loaded": False,
            "model_forwards_started": 0,
        },
        "next_authorized_step": (
            "IMPLEMENT_REVIEW_COMMIT_AND_PUSH_S6_RUNNER_THEN_EXECUTE_"
            "OUTCOME_BLIND_PRE2021_SCHEDULE_GATE_ONLY"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def write_preregistration(
    path: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    target = repository_path(path)
    payload = build_preregistration()
    raw = canonical_bytes(payload, pretty=True)
    if target.exists():
        if (
            target.is_symlink()
            or not target.is_file()
            or target.read_bytes() != raw
        ):
            raise RuntimeError("PSIM S6 preregistration drift")
        return payload
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = write_preregistration(args.output)
    print(
        json.dumps(
            {
                "candidate": payload["candidate"]["id"],
                "manifest_hash": payload["manifest_hash"],
                "next_authorized_step": payload["next_authorized_step"],
                "output": repository_path(args.output).as_posix(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
