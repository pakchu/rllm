#!/usr/bin/env python3
"""Preregister PSIM S5 direction-residualized semantic Ridge FQI.

S4 was rejected before its policy-specific 2021 evaluation because neither
sealed primary met the preregistered activity/direction gate and the
ExtraTrees action-code control was not exactly invariant.  This successor
reuses only the exact frozen Gemma representation, 2020-only PCA, and strict
2020 transition ledger.  It removes the unconditional 2020 long-vs-short
reward drift per current position, fits one promotable Ridge FQI, and must pass
an outcome-blind schedule gate before any later evaluator may read 2021.
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
    audit_psim_d8_rllm2_s4_pre2021_readiness as s4_audit,
)
from training import (  # noqa: E402
    preregister_psim_d8_rllm2_s4_semantic_fqi as s4,
)


REPO_ROOT = _BOOTSTRAP_REPO_ROOT
PROTOCOL_VERSION = (
    "psim_d8_rllm2_s5_direction_residual_ridge_fqi_preregistration_v1"
)
STAGE_ID = "PSIM-D8-RLLM2-S5-DIRECTION-RESIDUAL-RIDGE-FQI"
AS_OF_DATE = "2026-07-27"

PRIMARY_POLICY_ID = "semantic_ridge_direction_residual_fqi"
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
COMBINED_2021_FAMILY_IDS = (*s4.POLICY_FAMILY_IDS, *POLICY_FAMILY_IDS)

DEFAULT_OUTPUT = Path(
    "results/psim_d8_rllm2_s5_direction_residual_ridge_fqi_"
    "preregistration_2026-07-27.json"
)
RUNNER_PATH = Path(
    "training/run_psim_d8_rllm2_s5_train2020_direction_residual_ridge_fqi.py"
)
ATTEMPT_PATH = Path(
    "results/psim_d8_rllm2_s5_train2020_direction_residual_ridge_fqi_"
    "attempt_2026-07-27.json"
)
RESULT_PATH = Path(
    "results/psim_d8_rllm2_s5_train2020_direction_residual_ridge_fqi_"
    "pre2021_gate_2026-07-27.json"
)
RESIDUAL_LEDGER_PATH = Path(
    "data/psim_d8_rllm2_s5_direction_residual_reward_ledger_2020.csv.gz"
)
SCHEDULE_PATH = Path(
    "data/psim_d8_rllm2_s5_direction_residual_ridge_fqi_"
    "schedules_2021.csv.gz"
)
DELAYED_SCHEDULE_PATH = Path(
    "data/psim_d8_rllm2_s5_direction_residual_ridge_fqi_"
    "delayed_schedule_2021.csv.gz"
)
SCHEDULE_MANIFEST_PATH = Path(
    "results/psim_d8_rllm2_s5_direction_residual_ridge_fqi_"
    "schedule_gate_2021_2026-07-27.json"
)

S4_TERMINAL_RECORD_COMMIT = "2bdc009e0ca60ba103a98cdce8eee17194ee44c4"
S4_AUDIT_SCRIPT_SHA256 = (
    "d2da78e9801711dfa9fad1faaa0be3e77d62003d731caa36dfba2ed3873f9221"
)
S4_REJECTION_DOCUMENT = Path(
    "docs/psim-d8-rllm2-s4-pre2021-readiness-"
    "rejection-2026-07-27.md"
)
S4_REJECTION_DOCUMENT_SHA256 = (
    "5952ccad1d7536b82dc29f785e32f99fe89e1642aaa9de6311da05205965af0b"
)
S4_REJECTION_RESULT_SHA256 = (
    "e670c0965ba9fccccc06f472cff696e1392a2853d4d8016350191849d4548560"
)
S4_REJECTION_RESULT_HASH = (
    "36e81a7a42aa3a06b95e01c5203ec8f363a11b6771bafd28614ce4f055611ba9"
)

S4_PREREGISTRATION_SHA256 = (
    "2c3be7d92f61248dd26d428a9793f1ec1e703e794e6454970c8a6c2804fc2a34"
)
S4_PREREGISTRATION_MANIFEST_HASH = (
    "81eaa2c6ea80e792a20d4abf6e341355d66ea3b0bed7cd359af040139ce91845"
)
S4_ATTEMPT_SHA256 = (
    "717a140cec4a19ef057b2e44f7a508d25e977fd98b4447c3ea6f5dadd7e1e1fd"
)
S4_ATTEMPT_HASH = (
    "3f314bc9633351934b7b0c7ccc198f6ac48709e1ae7cf99ff4969900aaa553bc"
)
S4_RESULT_SHA256 = (
    "ca7a9a42eda7719a7e59b9927bf9c8e754689007468ec235ac4c5e2a1c619c75"
)
S4_RESULT_HASH = (
    "4253626d04e70f51dbd73df6c898c4a553b8df76b1ceb9c903c82e83ab2e3c09"
)
S4_SCHEDULE_MANIFEST_SHA256 = (
    "cd5d06f43d98b36e4d69b5630afe5e98f8d807aa05707e571aa49ed7c6ff4f6c"
)
S4_SCHEDULE_MANIFEST_HASH = (
    "e90ce8de7c82aa5ca22365f6c50a42bd975e994202b67df751a12ef804de6f85"
)
S4_TRANSITION_LEDGER_SHA256 = (
    "07d465538d84648793ebbf302c54dace38ef71e88b22d7bd3a19fec500b99a7a"
)
S4_TRANSITION_LEDGER_FRAME_HASH = (
    "25a0e03be8cdc327506205db2467c40f9434fb19851cec1bd78b7e9f4c820c06"
)
S4_PCA_SHA256 = (
    "6a01d505ad2531683c9e8e9e0672456daf19a17a59b32d772fc857370210f0a2"
)
S4_BASE_SCHEDULES_SHA256 = (
    "2f5b04e2514ca8b328fa6a92a45cea2828225090afc2427a58b6b778655394ba"
)
S4_DELAYED_SCHEDULES_SHA256 = (
    "326d4e8c5e866b83b331d97ee54ea2abedce35abf85887539ee97220ed354484"
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
        raise RuntimeError(f"PSIM S5 predecessor evidence missing: {path}")
    if sha256_file(target) != expected_sha256:
        raise RuntimeError(f"PSIM S5 predecessor evidence changed: {path}")
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"PSIM S5 expected JSON object: {path}")
    core = {
        key: value
        for key, value in payload.items()
        if key != self_hash_field
    }
    if (
        payload.get(self_hash_field) != expected_self_hash
        or payload.get(self_hash_field) != canonical_hash(core)
    ):
        raise RuntimeError(f"PSIM S5 predecessor self hash changed: {path}")
    return payload


def validate_s4_rejection() -> dict[str, Any]:
    registration = _read_exact_json(
        s4.DEFAULT_OUTPUT,
        expected_sha256=S4_PREREGISTRATION_SHA256,
        self_hash_field="manifest_hash",
        expected_self_hash=S4_PREREGISTRATION_MANIFEST_HASH,
    )
    attempt = _read_exact_json(
        s4.ATTEMPT_PATH,
        expected_sha256=S4_ATTEMPT_SHA256,
        self_hash_field="attempt_hash",
        expected_self_hash=S4_ATTEMPT_HASH,
    )
    result = _read_exact_json(
        s4.RESULT_PATH,
        expected_sha256=S4_RESULT_SHA256,
        self_hash_field="result_hash",
        expected_self_hash=S4_RESULT_HASH,
    )
    schedule_manifest = _read_exact_json(
        s4.SCHEDULE_MANIFEST_PATH,
        expected_sha256=S4_SCHEDULE_MANIFEST_SHA256,
        self_hash_field="manifest_hash",
        expected_self_hash=S4_SCHEDULE_MANIFEST_HASH,
    )
    rejection = _read_exact_json(
        s4_audit.DEFAULT_OUTPUT,
        expected_sha256=S4_REJECTION_RESULT_SHA256,
        self_hash_field="result_hash",
        expected_self_hash=S4_REJECTION_RESULT_HASH,
    )
    if s4_audit.build_audit() != rejection:
        raise RuntimeError("PSIM S5 predecessor rejection no longer replays")
    exact_files = (
        (
            s4.TRANSITION_LEDGER_PATH,
            S4_TRANSITION_LEDGER_SHA256,
        ),
        (s4.PCA_PATH, S4_PCA_SHA256),
        (s4.SCHEDULE_PATH, S4_BASE_SCHEDULES_SHA256),
        (s4.DELAYED_SCHEDULE_PATH, S4_DELAYED_SCHEDULES_SHA256),
        (S4_REJECTION_DOCUMENT, S4_REJECTION_DOCUMENT_SHA256),
        (
            Path("training") / Path(s4_audit.__file__).name,
            S4_AUDIT_SCRIPT_SHA256,
        ),
    )
    for path, expected in exact_files:
        if sha256_file(path) != expected:
            raise RuntimeError(f"PSIM S5 predecessor artifact changed: {path}")
    gate = registration["chronology_and_gates"]["2021_gate"]
    if (
        result["decision"] != "pass"
        or result["open_2021_outcomes_authorized"] is not True
        or result["access_boundary"]["2021_outcomes_opened"] is not False
        or result["access_boundary"]["2021_market_or_funding_paths_read"]
        != []
        or result["access_boundary"]["2021_reward_rows_created"] != 0
        or result["access_boundary"]["2021_economic_metrics_computed"] != 0
        or schedule_manifest["outcome_boundary"]
        ["2021_market_or_funding_payload_opened"]
        is not False
        or schedule_manifest["outcome_boundary"]
        ["2021_reward_rows_created"]
        != 0
        or schedule_manifest["outcome_boundary"]
        ["2021_economic_metrics_computed"]
        != 0
        or schedule_manifest["selection_from_2020_training_metrics"]
        is not False
        or rejection["decision"] != "reject"
        or rejection["eligible_primary_count"] != 0
        or rejection["audit_authorizes_open_2021_outcomes"] is not False
        or rejection["access_boundary"][
            "s4_protocol_2021_outcomes_remain_sealed"
        ]
        is not True
        or rejection["access_boundary"][
            "globally_pristine_2021_outcome_claim_allowed"
        ]
        is not False
        or gate["minimum_nonflat_intervals"] != MIN_NONFLAT_TARGET_ROWS
        or gate["minimum_each_direction_share"] != MIN_DIRECTION_SHARE
        or gate["familywise_p_max_strictly_below"] != FAMILYWISE_P_MAX
    ):
        raise RuntimeError("PSIM S5 predecessor scientific boundary changed")
    return {
        "registration": registration,
        "attempt": attempt,
        "result": result,
        "schedule_manifest": schedule_manifest,
        "rejection": rejection,
    }


def build_preregistration() -> dict[str, Any]:
    predecessor = validate_s4_rejection()
    model_contract = predecessor["registration"]["source_feature_binding"]
    gate_2021 = predecessor["registration"]["chronology_and_gates"][
        "2021_gate"
    ]
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "candidate": {
            "id": STAGE_ID,
            "predecessor": s4_audit.STAGE_ID,
            "stage": (
                "reuse_2020_ledger_fit_direction_residual_ridge_"
                "and_seal_2021"
            ),
            "profitability_claim": False,
            "single_promotable_primary": PRIMARY_POLICY_ID,
            "selection_from_2020_metrics": False,
            "selection_from_known_2021_metrics": False,
            "globally_pristine_2021_claim": False,
        },
        "predecessor_terminal_evidence": {
            "terminal_record_commit": S4_TERMINAL_RECORD_COMMIT,
            "s4_preregistration": {
                "path": s4.DEFAULT_OUTPUT.as_posix(),
                "sha256": S4_PREREGISTRATION_SHA256,
                "manifest_hash": S4_PREREGISTRATION_MANIFEST_HASH,
            },
            "s4_attempt": {
                "path": s4.ATTEMPT_PATH.as_posix(),
                "sha256": S4_ATTEMPT_SHA256,
                "attempt_hash": S4_ATTEMPT_HASH,
            },
            "s4_result": {
                "path": s4.RESULT_PATH.as_posix(),
                "sha256": S4_RESULT_SHA256,
                "result_hash": S4_RESULT_HASH,
            },
            "s4_schedule_manifest": {
                "path": s4.SCHEDULE_MANIFEST_PATH.as_posix(),
                "sha256": S4_SCHEDULE_MANIFEST_SHA256,
                "manifest_hash": S4_SCHEDULE_MANIFEST_HASH,
            },
            "s4_readiness_rejection": {
                "path": s4_audit.DEFAULT_OUTPUT.as_posix(),
                "sha256": S4_REJECTION_RESULT_SHA256,
                "result_hash": S4_REJECTION_RESULT_HASH,
                "terminal_action": predecessor["rejection"][
                    "terminal_action"
                ],
            },
            "s4_rejection_document": {
                "path": S4_REJECTION_DOCUMENT.as_posix(),
                "sha256": S4_REJECTION_DOCUMENT_SHA256,
            },
            "s4_audit_script_sha256": S4_AUDIT_SCRIPT_SHA256,
        },
        "frozen_source_and_outcome_artifacts": {
            "model": model_contract["model"],
            "selector": model_contract["selector"],
            "source_rows": model_contract["source_rows"],
            "embeddings": model_contract["embeddings"],
            "relation_logits": model_contract["relation_logits"],
            "transition_ledger_2020": {
                "path": s4.TRANSITION_LEDGER_PATH.as_posix(),
                "sha256": S4_TRANSITION_LEDGER_SHA256,
                "frame_hash": S4_TRANSITION_LEDGER_FRAME_HASH,
                "rows": 3_288,
                "state_rows": 366,
                "current_positions": 3,
                "actions": 3,
            },
            "pca32_2020": {
                "path": s4.PCA_PATH.as_posix(),
                "sha256": S4_PCA_SHA256,
                "fit_rows": 366,
                "components": 32,
                "input_width": 2_560,
                "refit": False,
            },
            "s4_base_schedules_2021": {
                "path": s4.SCHEDULE_PATH.as_posix(),
                "sha256": S4_BASE_SCHEDULES_SHA256,
                "rows": 9_125,
                "role": "frozen_nonpromotable_comparison_family",
            },
            "s4_delayed_schedules_2021": {
                "path": s4.DELAYED_SCHEDULE_PATH.as_posix(),
                "sha256": S4_DELAYED_SCHEDULES_SHA256,
                "rows": 730,
                "role": "frozen_nonpromotable_comparison_family",
            },
        },
        "direction_residual_reward_contract": {
            "input": (
                "exact S4 strict 2020 reward tensor reconstructed from the "
                "3,288-row transition ledger"
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
            "delta_formula": (
                "for each current position p, delta_p = 0.5 * mean_t("
                "R[t,p,TARGET_LONG] - R[t,p,TARGET_SHORT]) over all finite "
                "reachable 2020 state rows"
            ),
            "transform": {
                "TARGET_FLAT": "R_star[t,p,flat] = R[t,p,flat]",
                "TARGET_SHORT": (
                    "R_star[t,p,short] = R[t,p,short] + delta_p"
                ),
                "TARGET_LONG": (
                    "R_star[t,p,long] = R[t,p,long] - delta_p"
                ),
            },
            "post_transform_invariant": (
                "for each current position, mean residual long reward equals "
                "mean residual short reward within numeric tolerance 1e-15"
            ),
            "fit_rows": "all 366 2020 source rows",
            "clipping": False,
            "scaling": False,
            "threshold_search": False,
            "hyperparameter_search": False,
            "delta_selected_from_economic_metrics": False,
            "reward_control_order": (
                "residualize first; circular and within-month controls then "
                "permute the complete residual reward rows"
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
            "s4_family_role": "frozen_nonpromotable_comparison_controls",
            "action_code_permutation": (
                "fit with internal LONG/SHORT/FLAT codes, canonicalize back "
                "to FLAT/SHORT/LONG, and require exact schedule identity"
            ),
            "direction_flip": "swap canonical short and long Q outputs",
            "fixed_circular_reward_shift": FIXED_CIRCULAR_REWARD_SHIFT,
            "fixed_within_month_shuffle_seed": FIXED_SHUFFLE_SEED,
            "current_position_only": (
                "zero semantic-width input; separate position one-hot retained"
            ),
            "masked_semantic_embedding": (
                "all 32 semantic coordinates zero; position one-hot retained"
            ),
            "metadata_frontmatter_only": (
                "the exact S4 nonsemantic metadata/frontmatter feature family"
            ),
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
            "degenerate_control_ids": [
                "always_flat",
                "always_long",
                "always_short",
                "previous_target_persistence",
                f"{PRIMARY_POLICY_ID}_current_position_only",
                f"{PRIMARY_POLICY_ID}_masked_semantic_embedding",
                f"{PRIMARY_POLICY_ID}_metadata_frontmatter_only",
            ],
            "failure_action": (
                "TERMINAL_REJECT_S5_WITHOUT_2021_MARKET_FUNDING_REWARD_"
                "OR_METRIC_ACCESS"
            ),
            "pass_action": (
                "SEAL_S5_2021_SCHEDULE_AUTHORIZE_SEPARATE_REPORT_ONLY_"
                "2021_TRANSFER_PREREGISTRATION"
            ),
        },
        "future_2021_transfer_gate": {
            **gate_2021,
            "role": (
                "policy-specific report-only transfer under disclosed global "
                "2021 contamination"
            ),
            "single_promotable_primary": PRIMARY_POLICY_ID,
            "base_cost_rate": 0.0006,
            "stress_cost_rate": 0.0010,
            "execution_delay": "one complete 5m bar",
            "strict_economics_implementation": (
                "training/bctp_strict_economics.py"
            ),
            "weekly_familywise_family_ids": list(COMBINED_2021_FAMILY_IDS),
            "selection_or_repair_from_2021_metrics": False,
            "success_is_live_promotion": False,
            "success_requires_forward_or_live_confirmation": True,
        },
        "global_outcome_contamination_disclosure": {
            **predecessor["rejection"]["global_2021_outcome_history"],
            "s5_policy_specific_2021_metrics_already_exist": False,
            "known_unrelated_2021_metrics_may_not_inform_s5_design": True,
            "2021_may_be_called_globally_pristine": False,
            "globally_pristine_evidence_requires_forward_or_live_data": True,
        },
        "chronology": [
            "verify exact S4 terminal rejection and artifact hashes",
            "write S5 attempt before parsing the 2020 transition ledger",
            "read no raw market or funding payload",
            "reconstruct exact 2020 reward tensor from the frozen ledger",
            "compute three fixed direction-residual deltas without search",
            "fit the complete eight-policy Ridge family on 2020 only",
            "seal 2021 base and delayed schedules before any 2021 outcome read",
            "run the fixed pre-2021 schedule readiness gate",
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
            "attempt_written_before_2020_ledger_parse": True,
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
                    s4.DEFAULT_OUTPUT.as_posix(),
                    s4.ATTEMPT_PATH.as_posix(),
                    s4.RESULT_PATH.as_posix(),
                    s4.SCHEDULE_MANIFEST_PATH.as_posix(),
                    s4.SCHEDULE_PATH.as_posix(),
                    s4.DELAYED_SCHEDULE_PATH.as_posix(),
                    s4.PCA_PATH.as_posix(),
                    s4.TRANSITION_LEDGER_PATH.as_posix(),
                    s4_audit.DEFAULT_OUTPUT.as_posix(),
                    S4_REJECTION_DOCUMENT.as_posix(),
                    s4_audit.PRIOR_2021_TRANSFER_DOCUMENT.as_posix(),
                    s4_audit.PRIOR_2021_STAGE_SOURCE_MANIFEST.as_posix(),
                }
            ),
            "raw_market_or_funding_paths_read": [],
            "2020_outcome_derived_artifact_bytes_hashed": True,
            "2020_transition_reward_rows_parsed": 0,
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
            "IMPLEMENT_REVIEW_COMMIT_AND_PUSH_S5_RUNNER_THEN_EXECUTE_"
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
            raise RuntimeError("PSIM S5 preregistration drift")
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
