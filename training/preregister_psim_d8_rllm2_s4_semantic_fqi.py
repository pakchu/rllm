#!/usr/bin/env python3
"""Preregister the 2020-only PSIM-D8-RLLM2 semantic fitted-Q stage.

This module is intentionally outcome blind.  It verifies the successful S3
source-feature seal and freezes the complete semantic-policy family before the
runner is allowed to open the 2020 market or funding stage.  Outcomes from
2021 and later remain forbidden until their target schedules have been sealed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_BOOTSTRAP_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_BOOTSTRAP_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_REPO_ROOT))

from training import preregister_psim_d8_rllm1 as rllm1  # noqa: E402
from training import (  # noqa: E402
    preregister_psim_d8_rllm2_s3_chunked_native_source_feature as s3,
)


REPO_ROOT = _BOOTSTRAP_REPO_ROOT
PROTOCOL_VERSION = "psim_d8_rllm2_s4_semantic_fqi_preregistration_v1"
STAGE_ID = "PSIM-D8-RLLM2-S4-SEMANTIC-FQI"
AS_OF_DATE = "2026-07-27"

DEFAULT_OUTPUT = Path(
    "results/psim_d8_rllm2_s4_semantic_fqi_"
    "preregistration_2026-07-27.json"
)
RUNNER_PATH = Path(
    "training/run_psim_d8_rllm2_s4_train2020_semantic_fqi.py"
)
ATTEMPT_PATH = Path(
    "results/psim_d8_rllm2_s4_train2020_semantic_fqi_"
    "attempt_2026-07-27.json"
)
RESULT_PATH = Path(
    "results/psim_d8_rllm2_s4_train2020_semantic_fqi_"
    "seal_2026-07-27.json"
)
TRANSITION_LEDGER_PATH = Path(
    "data/psim_d8_rllm2_s4_semantic_fqi_transition_ledger_2020.csv.gz"
)
PCA_PATH = Path(
    "data/psim_d8_rllm2_s4_semantic_fqi_pca32_2020.npz"
)
SCHEDULE_PATH = Path(
    "data/psim_d8_rllm2_s4_semantic_fqi_schedules_2021.csv.gz"
)
DELAYED_SCHEDULE_PATH = Path(
    "data/psim_d8_rllm2_s4_semantic_fqi_delayed_schedules_2021.csv.gz"
)
SCHEDULE_MANIFEST_PATH = Path(
    "results/psim_d8_rllm2_s4_semantic_fqi_"
    "schedule_seal_2021_2026-07-27.json"
)

S3_PREREGISTRATION_SHA256 = (
    "7edb7eeef115e0579099f9aa990648f1db1cde28053867c0ae1edb6c60deb196"
)
S3_PREREGISTRATION_MANIFEST_HASH = (
    "263f0476ce887cad7b5e9c0174e02b6f714a8ce54029ef5788a6179ad1c396f3"
)
S3_ATTEMPT_SHA256 = (
    "99046f2ef3d816c279136e61e7c60394b39b5adc5cc790735d38025585fd5bcb"
)
S3_ATTEMPT_HASH = (
    "9bc78f6774246a41c88c57ebbe855cb7c4f7577e9fff814b080403400497bf4f"
)
S3_GATE_SHA256 = (
    "075f19c855d49146cb38c1bb40bd6599389736f7fac9e1a998c0eac320e6db50"
)
S3_GATE_RESULT_HASH = (
    "1fce8b64b68dff2b99d0d940a2fd2d02379d21c24e37ac79b0676561c190bceb"
)
S3_RESULT_SHA256 = (
    "0278b303005ae50510004344eb8889bc464c4b61612e0e162e091e7decd8a976"
)
S3_RESULT_HASH = (
    "9bf7121d4feec0f7000626f8493e05effc3dfff8f9857b7ac5f05f6beadc3af9"
)
S3_SUCCESS_DOCUMENT = Path(
    "docs/psim-d8-rllm2-s3-chunked-native-source-feature-"
    "seal-success-2026-07-27.md"
)
S3_SUCCESS_DOCUMENT_SHA256 = (
    "74570868b2414df4356c4d3d79079b469345991a3930a1c6ea097daff07f5669"
)
S3_EXECUTION_COMMIT = "dfef2ee175f6c055da71dbc797d611db4355e921"
S3_TERMINAL_RECORD_COMMIT = (
    "a1e0b160d46882030b46f5bde1655b93ab73acca"
)
S3_RUNNER_SHA256 = (
    "ab12b42f2f9f6cbad02ae85fd61511b333736bdae6028462af8052fe77ec72a9"
)

S3_SOURCE_ROWS_SHA256 = (
    "2897845bf55506bf877c2015a670707287d3d98c3718361e24ad31504b98939c"
)
S3_EMBEDDINGS_SHA256 = (
    "509d7922561cdec0582165e5976ac9bdcc72dfe26e68c6ce46fe7ba9fab4f2a0"
)
S3_RELATION_LOGITS_SHA256 = (
    "384725cd8f2f451a8dedca76625333d66517937b744e09ea5d71cd938c500b62"
)
S3_RELATION_ROWS_SHA256 = (
    "4ea96c6af2da8f574eed88fcbfa7fe6f20b5d6131e89eb41b734a0e498cfef1a"
)
S3_SOURCE_ROW_ROSTER_HASH = (
    "033df68d9067a88cb14eb83f92b7638f0addc2372ef08184ef75e9fe3f7ba47c"
)

PRIMARY_POLICY_IDS = (
    "semantic_ridge_fqi",
    "semantic_extra_trees_fqi",
)
NONSEMANTIC_CONTROL_IDS = (
    "always_flat",
    "always_long",
    "always_short",
    "previous_target_persistence",
    "exact_redacted_payload_memory",
    "metadata_frontmatter_only",
    "path_section_diff_size_only",
    "cadence_revision_topology_only",
    "shuffled_eip_bip_daily_relation",
    "shuffled_old_new_pairing",
    "future_status_scrub",
    "ethereum_only",
    "bitcoin_only",
    "current_position_only",
    "masked_semantic_embedding",
)
ALGORITHM_ABLATION_SUFFIXES = (
    "circular_21_reward",
    "within_month_shuffled_reward",
    "direction_flip",
    "action_code_permutation",
)
ALGORITHM_ABLATION_IDS = tuple(
    f"{primary}_{suffix}"
    for primary in PRIMARY_POLICY_IDS
    for suffix in ALGORITHM_ABLATION_SUFFIXES
)
POLICY_FAMILY_IDS = (
    *PRIMARY_POLICY_IDS,
    *NONSEMANTIC_CONTROL_IDS,
    *ALGORITHM_ABLATION_IDS,
)
EXTRA_TREES_CONTRACT = {
    "bootstrap": False,
    "max_depth": 6,
    "max_features": "sqrt",
    "min_samples_leaf": 12,
    "min_samples_split": 24,
    "n_estimators": 512,
    "n_jobs": 1,
    "random_state": 20_260_727,
}


def repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def _read_exact_json(
    path: str | Path,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    target = repository_path(path)
    if target.is_symlink() or not target.is_file():
        raise RuntimeError(f"PSIM-D8-RLLM2-S4 evidence missing: {path}")
    raw = target.read_bytes()
    if s3.sha256_bytes(raw) != expected_sha256:
        raise RuntimeError(f"PSIM-D8-RLLM2-S4 evidence hash changed: {path}")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise RuntimeError(f"PSIM-D8-RLLM2-S4 expected JSON object: {path}")
    return payload


def _verify_file(path: str | Path, expected_sha256: str) -> None:
    target = repository_path(path)
    if (
        target.is_symlink()
        or not target.is_file()
        or s3.sha256_bytes(target.read_bytes()) != expected_sha256
    ):
        raise RuntimeError(f"PSIM-D8-RLLM2-S4 artifact changed: {path}")


def _verify_self_hash(
    payload: dict[str, Any],
    *,
    field: str,
    expected: str,
) -> None:
    core = {key: value for key, value in payload.items() if key != field}
    if payload.get(field) != s3.canonical_hash(core):
        raise RuntimeError("PSIM-D8-RLLM2-S4 predecessor self-hash changed")
    if payload.get(field) != expected:
        raise RuntimeError(
            "PSIM-D8-RLLM2-S4 predecessor canonical hash changed"
        )


def validate_s3_success() -> dict[str, Any]:
    preregistration = _read_exact_json(
        s3.DEFAULT_OUTPUT,
        expected_sha256=S3_PREREGISTRATION_SHA256,
    )
    attempt = _read_exact_json(
        s3.ATTEMPT_PATH,
        expected_sha256=S3_ATTEMPT_SHA256,
    )
    gate = _read_exact_json(
        s3.REPEATABILITY_RESULT_PATH,
        expected_sha256=S3_GATE_SHA256,
    )
    result = _read_exact_json(
        s3.RESULT_PATH,
        expected_sha256=S3_RESULT_SHA256,
    )
    for payload, field, expected in (
        (
            preregistration,
            "manifest_hash",
            S3_PREREGISTRATION_MANIFEST_HASH,
        ),
        (attempt, "attempt_hash", S3_ATTEMPT_HASH),
        (gate, "result_hash", S3_GATE_RESULT_HASH),
        (result, "result_hash", S3_RESULT_HASH),
    ):
        _verify_self_hash(payload, field=field, expected=expected)
    _verify_file(S3_SUCCESS_DOCUMENT, S3_SUCCESS_DOCUMENT_SHA256)
    for path, expected in (
        (s3.SOURCE_ROWS_PATH, S3_SOURCE_ROWS_SHA256),
        (s3.EMBEDDINGS_PATH, S3_EMBEDDINGS_SHA256),
        (s3.RELATION_LOGITS_PATH, S3_RELATION_LOGITS_SHA256),
        (s3.RELATION_ROWS_PATH, S3_RELATION_ROWS_SHA256),
    ):
        _verify_file(path, expected)
    artifacts = result.get("artifacts", {})
    boundary = result.get("access_boundary", {})
    if (
        result.get("decision") != "pass"
        or result.get("execution_commit") != S3_EXECUTION_COMMIT
        or attempt.get("runner_sha256") != S3_RUNNER_SHA256
        or result.get("source_feature_seal_authorized") is not True
        or result.get("open_2020_train_outcomes_authorized") is not True
        or result.get("open_2021_or_later_outcomes_authorized") is not False
        or artifacts.get("source_rows", {}).get("sha256")
        != S3_SOURCE_ROWS_SHA256
        or artifacts.get("embeddings", {}).get("sha256")
        != S3_EMBEDDINGS_SHA256
        or artifacts.get("relation_logits", {}).get("sha256")
        != S3_RELATION_LOGITS_SHA256
        or artifacts.get("relation_rows", {}).get("sha256")
        != S3_RELATION_ROWS_SHA256
        or result.get("source_roster", {}).get("source_row_roster_hash")
        != S3_SOURCE_ROW_ROSTER_HASH
        or boundary.get("market_or_funding_paths_read") != []
        or boundary.get("market_rows_parsed") != 0
        or boundary.get("funding_rows_parsed") != 0
        or boundary.get("market_or_funding_payload_bytes_hashed") is not False
        or boundary.get("rewards_created") != 0
        or boundary.get("economic_metrics_computed") != 0
        or boundary.get("train_2020_outcomes_opened") is not False
        or boundary.get("test_outcomes_opened") is not False
        or boundary.get("eval_outcomes_opened") is not False
    ):
        raise RuntimeError("PSIM-D8-RLLM2-S3 success evidence changed")
    return {
        "preregistration": preregistration,
        "attempt": attempt,
        "gate": gate,
        "result": result,
    }


def build_preregistration() -> dict[str, Any]:
    predecessor = validate_s3_success()
    inherited = predecessor["preregistration"]["scientific_contract"]
    semantic_gate = inherited["semantic_encoder_gate"]
    if (
        semantic_gate.get("algorithms")
        != ["ridge_fitted_q", "extra_trees_fitted_q"]
        or semantic_gate.get("fitted_q", {}).get("bellman_iterations") != 25
        or semantic_gate.get("fitted_q", {}).get("discount") != 0.99
        or semantic_gate.get("fitted_q", {}).get("ridge_alpha") != 100.0
        or semantic_gate.get("fitted_q", {}).get("extra_trees")
        != EXTRA_TREES_CONTRACT
    ):
        raise RuntimeError("PSIM-D8-RLLM2 semantic gate contract changed")
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "candidate": {
            "id": STAGE_ID,
            "predecessor": s3.STAGE_ID,
            "stage": "train_2020_semantic_fqi_and_seal_2021",
            "profitability_claim": False,
            "selection_from_2020_training_metrics": False,
        },
        "predecessor_terminal_evidence": {
            "terminal_record_commit": S3_TERMINAL_RECORD_COMMIT,
            "execution_commit": S3_EXECUTION_COMMIT,
            "runner_sha256": S3_RUNNER_SHA256,
            "preregistration": {
                "path": s3.DEFAULT_OUTPUT.as_posix(),
                "sha256": S3_PREREGISTRATION_SHA256,
                "manifest_hash": S3_PREREGISTRATION_MANIFEST_HASH,
            },
            "attempt": {
                "path": s3.ATTEMPT_PATH.as_posix(),
                "sha256": S3_ATTEMPT_SHA256,
                "attempt_hash": S3_ATTEMPT_HASH,
            },
            "repeatability_gate": {
                "path": s3.REPEATABILITY_RESULT_PATH.as_posix(),
                "sha256": S3_GATE_SHA256,
                "result_hash": S3_GATE_RESULT_HASH,
            },
            "terminal_result": {
                "path": s3.RESULT_PATH.as_posix(),
                "sha256": S3_RESULT_SHA256,
                "result_hash": S3_RESULT_HASH,
                "terminal_action": predecessor["result"]["terminal_action"],
            },
            "success_document": {
                "path": S3_SUCCESS_DOCUMENT.as_posix(),
                "sha256": S3_SUCCESS_DOCUMENT_SHA256,
            },
        },
        "source_feature_binding": {
            "source_rows": {
                "path": s3.SOURCE_ROWS_PATH.as_posix(),
                "sha256": S3_SOURCE_ROWS_SHA256,
                "rows": 1_461,
                "source_row_roster_hash": S3_SOURCE_ROW_ROSTER_HASH,
            },
            "embeddings": {
                "path": s3.EMBEDDINGS_PATH.as_posix(),
                "sha256": S3_EMBEDDINGS_SHA256,
                "shape": [1_461, 2_560],
                "dtype": "float32",
            },
            "relation_logits": {
                "path": s3.RELATION_LOGITS_PATH.as_posix(),
                "sha256": S3_RELATION_LOGITS_SHA256,
                "shape": [1_461, 6],
                "dtype": "float32",
            },
            "relation_rows": {
                "path": s3.RELATION_ROWS_PATH.as_posix(),
                "sha256": S3_RELATION_ROWS_SHA256,
                "rows": 1_461,
            },
            "model": inherited["model"],
            "selector": inherited["selector"],
            "model_visible": inherited["model_visible"],
        },
        "semantic_state_contract": {
            "decision_clock": "ARCHIVE_D90 daily 12:05:00Z",
            "source_rows_per_year": {
                "2020": 366,
                "2021": 365,
                "2022": 365,
                "2023": 365,
            },
            "primary_features": (
                "S3 frozen 2560-dimensional policy embedding transformed by "
                "2020-only PCA32 plus separate current-position one-hot"
            ),
            "excluded_model_features": [
                "row_index",
                "split",
                "split_year",
                "decision_at_calendar_components",
                "hash_identifiers",
                "market_price_return_funding_reward_or_pnl",
            ],
            "pca": {
                "fit_rows": "2020 only",
                "components": 32,
                "solver": "full_svd",
                "whiten": False,
                "deterministic_sign_rule": (
                    "for each component, its largest-absolute loading is "
                    "forced positive; lowest index breaks ties"
                ),
                "refit_on_2021_source_distribution": False,
            },
            "position_input": [
                "POSITION_SHORT",
                "POSITION_FLAT",
                "POSITION_LONG",
            ],
            "relation_logits_usage": (
                "control and diagnostic only; not appended to the two primary "
                "semantic policies"
            ),
        },
        "fitted_q_contract": {
            "primary_policy_ids": list(PRIMARY_POLICY_IDS),
            "algorithms": {
                "semantic_ridge_fqi": {
                    "kind": "ridge",
                    "alpha": 100.0,
                    "unpenalized_intercept": True,
                },
                "semantic_extra_trees_fqi": {
                    "kind": "extra_trees",
                    **EXTRA_TREES_CONTRACT,
                },
            },
            "bellman_iterations": 25,
            "discount": 0.99,
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
            "fit_roster": (
                "both primaries fit on all 2020 rows; neither is selected "
                "using 2020 economic metrics"
            ),
        },
        "control_family": {
            "policy_family_ids": list(POLICY_FAMILY_IDS),
            "nonsemantic_control_ids": list(NONSEMANTIC_CONTROL_IDS),
            "algorithm_ablation_ids": list(ALGORITHM_ABLATION_IDS),
            "fixed_hash_feature_width": 32,
            "fixed_shuffle_seed": 20_260_727,
            "fixed_circular_reward_shift": 21,
            "exact_payload_memory_unknown_signature_action": "TARGET_FLAT",
            "shuffles": (
                "deterministic within-month permutations fit from source "
                "identity only, never from rewards or outcomes"
            ),
            "masked_semantic_embedding": (
                "all semantic coordinates zero; current position retained"
            ),
            "future_status_scrub": (
                "nonsemantic source-summary feature family with metadata "
                "status fields removed"
            ),
            "single_predeclared_family_for_weekly_max_stat": True,
        },
        "economic_contract": {
            "accounting_implementation": "training/bctp_strict_economics.py",
            "transition_implementation": (
                "training/psim_semantic_transition_labels.py"
            ),
            "market_and_funding_isolation": "training/bctp_stage_sources.py",
            "fit_outcome_stage": "2020",
            "target_schedule_stage": "2021",
            "base_cost_rate": 0.0006,
            "stress_cost_rate": 0.0010,
            "execution_delay_diagnostic": "one complete 5m bar",
            "reward": (
                "log(max(E_end/E_pre,1e-12)) "
                "-(1/3)*held_path_downside_fraction "
                "-0.001*abs(target_new-target_old)"
            ),
            "full_calendar_cagr": True,
            "strict_mdd": (
                "single global HWM, favorable-then-adverse OHLC path, "
                "virtual liquidation, terminal flatten"
            ),
            "2020_metrics_role": "diagnostic_only_not_candidate_selection",
        },
        "authorized_2020_outcome_sources": {
            "isolation_implementation": "training/bctp_stage_sources.py",
            "evaluator_manifest_hash": (
                "25f80efcef9e8e5b08b9138dc17cb4c18edcc0b47187ad9b43f999acee0afd87"
            ),
            "stage": "2020",
            "start_inclusive": "2020-01-01T00:00:00Z",
            "end_exclusive": "2021-01-01T00:00:00Z",
            "market": {
                "parent_path": rllm1.MARKET.as_posix(),
                "parent_expected_sha256": rllm1.MARKET_SHA256,
                "parent_manifest": rllm1.MARKET_MANIFEST.as_posix(),
                "parent_manifest_sha256": rllm1.MARKET_MANIFEST_SHA256,
                "stage_local_path": (
                    "data/bctp_stage_sources/2020/"
                    "bctp_market_2020.csv.gz"
                ),
                "expected_rows": 105_408,
                "frequency": "5min",
            },
            "funding": {
                "parent_path": rllm1.FUNDING.as_posix(),
                "parent_expected_sha256": rllm1.FUNDING_SHA256,
                "parent_manifest": rllm1.FUNDING_MANIFEST.as_posix(),
                "parent_manifest_sha256": rllm1.FUNDING_MANIFEST_SHA256,
                "stage_local_path": (
                    "data/bctp_stage_sources/2020/"
                    "bctp_funding_2020.csv.gz"
                ),
                "expected_rows": 1_098,
                "frequency": "8h",
            },
            "parent_access_rule": (
                "only bctp_stage_sources may stream the parent; copy exactly "
                "the declared 2020 row count and stop without reading the "
                "first 2021 row"
            ),
            "full_parent_payload_hash_during_s4": False,
            "stage_local_payload_hash_and_validation_required": True,
            "2021_or_later_numeric_rows_authorized": False,
            "existing_2021_or_later_stage_local_payload_open_authorized": False,
        },
        "chronology_and_gates": {
            "s4": [
                "open exact 2020 market and funding only",
                "build all 2020 counterfactual transition rewards",
                "fit PCA32 and complete frozen policy family on 2020 only",
                "seal base and delayed 2021 schedules before 2021 outcomes",
                "publish 2020 diagnostics without choosing a primary",
            ],
            "2021_gate": semantic_gate["2021_gate"],
            "2021_selection": (
                "select one of the two primaries only if it passes every "
                "frozen gate; ties use minimum base/stress/delay ratio, base "
                "ratio, absolute return, lower MDD, lexical policy id"
            ),
            "2022_gate_before_qlora": semantic_gate[
                "2022_gate_before_qlora"
            ],
            "qlora_authorized_in_s4": False,
            "2021_or_later_outcomes_authorized_in_s4": False,
            "failure_action": (
                "REJECT_S4_WITHOUT_2021_OUTCOME_ACCESS_IF_TRAIN_OR_SEAL_FAILS"
            ),
            "success_action": (
                "ACCEPT_S4_SEALED_2021_SCHEDULES_AUTHORIZE_2021_EVALUATION_ONLY"
            ),
        },
        "artifact_contract": {
            "runner": RUNNER_PATH.as_posix(),
            "attempt": ATTEMPT_PATH.as_posix(),
            "result": RESULT_PATH.as_posix(),
            "transition_ledger_2020": TRANSITION_LEDGER_PATH.as_posix(),
            "pca32_2020": PCA_PATH.as_posix(),
            "base_schedules_2021": SCHEDULE_PATH.as_posix(),
            "delayed_primary_schedules_2021": (
                DELAYED_SCHEDULE_PATH.as_posix()
            ),
            "schedule_manifest_2021": SCHEDULE_MANIFEST_PATH.as_posix(),
            "deterministic_npz_csv_gzip_and_json": True,
            "authorizing_result_published_last": True,
            "fixed_paths_no_output_override": True,
        },
        "execution_contract": {
            "clean_head_equals_origin_main_required": True,
            "runner_must_be_committed_before_outcome_open": True,
            "attempt_written_before_2020_outcome_open": True,
            "2021_schedule_manifest_written_before_authorizing_result": True,
            "all_artifact_hashes_bound_in_result": True,
            "no_existing_2021_stage_source_payload_open": True,
            "no_2021_or_later_market_funding_reward_metric_access": True,
            "no_s1_or_s2_model_output_or_checkpoint_reuse": True,
            "no_model_load_or_new_model_forward": True,
        },
        "access_boundary": {
            "source_files_read": sorted(
                {
                    s3.DEFAULT_OUTPUT.as_posix(),
                    s3.ATTEMPT_PATH.as_posix(),
                    s3.REPEATABILITY_RESULT_PATH.as_posix(),
                    s3.RESULT_PATH.as_posix(),
                    S3_SUCCESS_DOCUMENT.as_posix(),
                    s3.SOURCE_ROWS_PATH.as_posix(),
                    s3.EMBEDDINGS_PATH.as_posix(),
                    s3.RELATION_LOGITS_PATH.as_posix(),
                    s3.RELATION_ROWS_PATH.as_posix(),
                }
            ),
            "market_or_funding_paths_read": [],
            "preregistration_forbidden_bound_paths": sorted(
                rllm1.FORBIDDEN_BOUND_PATHS
            ),
            "market_rows_parsed": 0,
            "funding_rows_parsed": 0,
            "market_or_funding_payload_bytes_hashed": False,
            "rewards_created": 0,
            "economic_metrics_computed": 0,
            "model_loaded": False,
            "model_forwards_started": 0,
            "train_2020_outcomes_opened": False,
            "2021_outcomes_opened": False,
            "test_outcomes_opened": False,
            "eval_outcomes_opened": False,
        },
        "next_authorized_step": (
            "IMPLEMENT_REVIEW_COMMIT_AND_PUSH_S4_RUNNER_THEN_OPEN_2020_ONLY"
        ),
    }
    return {**core, "manifest_hash": s3.canonical_hash(core)}


def write_preregistration(
    path: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    payload = build_preregistration()
    target = repository_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = s3.canonical_json_bytes(payload, pretty=True)
    if target.exists() and target.read_bytes() != encoded:
        raise RuntimeError(
            f"PSIM-D8-RLLM2-S4 preregistration drift: {target}"
        )
    target.write_bytes(encoded)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = write_preregistration(args.output)
    print(
        json.dumps(
            {
                "candidate": payload["candidate"]["id"],
                "manifest_hash": payload["manifest_hash"],
                "primary_policy_ids": list(PRIMARY_POLICY_IDS),
                "policy_family_size": len(POLICY_FAMILY_IDS),
                "next_authorized_step": payload["next_authorized_step"],
                "access_boundary": payload["access_boundary"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
