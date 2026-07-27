#!/usr/bin/env python3
"""Preregister the PSIM-D8-RLLM2 operational-only successor."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

_BOOTSTRAP_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_BOOTSTRAP_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_REPO_ROOT))

from training import preregister_psim_d8_rllm1 as rllm1_prereg
from training import run_psim_d8_rllm1_base_memorization_gate as rllm1_gate


REPO_ROOT = _BOOTSTRAP_REPO_ROOT
PROTOCOL_VERSION = "psim_d8_rllm2_operational_successor_preregistration_v1"
POLICY_ID = "PSIM-D8-RLLM2"
DEFAULT_OUTPUT = Path(
    "results/psim_d8_rllm2_operational_successor_"
    "preregistration_2026-07-27.json"
)
RLLM1_PREREGISTRATION = rllm1_prereg.DEFAULT_OUTPUT
RLLM1_PREREGISTRATION_SHA256 = (
    "6e9b034744acc1b701a283c7ba34e2bcc533e781c33bbe95657060f78c67732e"
)
RLLM1_PREREGISTRATION_MANIFEST_HASH = (
    "31b86a25fedfe9c3ef98cfa4a3b617a8df7bddf68cc6eab6be33fa66069e4d89"
)
RLLM1_ATTEMPT = Path(
    "results/psim_d8_rllm1_base_memorization_gate_"
    "attempt_2026-07-27.json"
)
RLLM1_ATTEMPT_SHA256 = (
    "a325fb09286cf921e5b9e1d65e4655a03bde11058aa09a6fe0cd5d1fc79c3179"
)
RLLM1_ATTEMPT_HASH = (
    "db2e1d7c5ce0bc7dbb061ff6f3e1d4a674d018db16dbf04a7509e04566d3a609"
)
RLLM1_FAILURE = Path(
    "results/psim_d8_rllm1_base_memorization_gate_"
    "failure_2026-07-27.json"
)
RLLM1_FAILURE_SHA256 = (
    "02728096681f058144c12090cfa5876a973fb1cbd5146e35d59e2aa260dca812"
)
RLLM1_FAILURE_RESULT_HASH = (
    "b0a40fa9904dd9b7877b3b64c9f382999d0b24a75a4edbcc687ccfe8b424fe69"
)
RLLM1_EXECUTION_COMMIT = "ce9ba77782ff0cc34411d60dc1ba7def5bea707f"
RLLM1_TERMINAL_RECORD_COMMIT = (
    "8ec8d4711900f405a206b1980a51fdcd582a1415"
)
RLLM1_RUNNER_SHA256 = (
    "931a9d5a888e4e821023a177790915a0b632762fb23fc90c17268fcf08119b5d"
)
CASE_ROSTER_HASH = (
    "5065cd58322aee8f38f11ec2c4a186fb1a7ba8133aa2b2bb0182f67322a8bf39"
)
RLLM2_ATTEMPT = Path(
    "results/psim_d8_rllm2_base_memorization_gate_"
    "attempt_2026-07-27.json"
)
RLLM2_RESULT = Path(
    "results/psim_d8_rllm2_base_memorization_gate_2026-07-27.json"
)


def repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def canonical_json_bytes(payload: Any, *, pretty: bool = True) -> bytes:
    if pretty:
        return (
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        canonical_json_bytes(payload, pretty=False)
    ).hexdigest()


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_exact_json(
    path: str | Path,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    target = repository_path(path)
    if target.is_symlink() or not target.is_file():
        raise RuntimeError(f"unsafe RLLM2 predecessor artifact: {path}")
    raw = target.read_bytes()
    if sha256_bytes(raw) != expected_sha256:
        raise RuntimeError(f"RLLM2 predecessor hash changed: {path}")
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"malformed RLLM2 predecessor artifact: {path}")
    return payload


def validate_predecessor() -> dict[str, Any]:
    preregistration = _read_exact_json(
        RLLM1_PREREGISTRATION,
        expected_sha256=RLLM1_PREREGISTRATION_SHA256,
    )
    attempt = _read_exact_json(
        RLLM1_ATTEMPT,
        expected_sha256=RLLM1_ATTEMPT_SHA256,
    )
    failure = _read_exact_json(
        RLLM1_FAILURE,
        expected_sha256=RLLM1_FAILURE_SHA256,
    )
    preregistration_core = {
        key: value
        for key, value in preregistration.items()
        if key != "manifest_hash"
    }
    if (
        preregistration.get("manifest_hash")
        != canonical_hash(preregistration_core)
        or preregistration.get("manifest_hash")
        != RLLM1_PREREGISTRATION_MANIFEST_HASH
        or preregistration.get("candidate", {}).get("id")
        != rllm1_prereg.POLICY_ID
        or preregistration.get("access_boundary", {}).get(
            "market_or_funding_payload_bytes_hashed"
        )
        is not False
    ):
        raise RuntimeError("RLLM1 scientific preregistration changed")
    attempt_core = {
        key: value
        for key, value in attempt.items()
        if key != "attempt_hash"
    }
    if (
        attempt.get("attempt_hash") != canonical_hash(attempt_core)
        or attempt.get("attempt_hash") != RLLM1_ATTEMPT_HASH
        or attempt.get("execution_commit") != RLLM1_EXECUTION_COMMIT
        or attempt.get("runner_sha256") != RLLM1_RUNNER_SHA256
        or attempt.get("case_roster_hash") != CASE_ROSTER_HASH
    ):
        raise RuntimeError("RLLM1 attempt evidence changed")
    failure_core = {
        key: value
        for key, value in failure.items()
        if key != "result_hash"
    }
    observations = failure.get("observations", {})
    if (
        failure.get("result_hash") != canonical_hash(failure_core)
        or failure.get("result_hash") != RLLM1_FAILURE_RESULT_HASH
        or failure.get("decision") != "reject"
        or failure.get("rerun_authorized") is not False
        or observations.get("model_forwards_started") != 0
        or observations.get("challenge_predictions_created") != 0
        or observations.get("challenge_statistics_computed") is not False
        or failure.get("access_boundary", {}).get(
            "market_or_funding_payload_bytes_hashed"
        )
        is not False
    ):
        raise RuntimeError("RLLM1 terminal failure evidence changed")
    return {
        "preregistration": preregistration,
        "attempt": attempt,
        "failure": failure,
    }


def build_preregistration() -> dict[str, Any]:
    predecessor = validate_predecessor()
    inherited_scientific_contract = {
        "source_authority": predecessor["preregistration"][
            "source_authority"
        ],
        "selector_contract": predecessor["preregistration"][
            "selector_contract"
        ],
        "model_visible_contract": predecessor["preregistration"][
            "model_visible_contract"
        ],
        "memorization_contract": predecessor["preregistration"][
            "memorization_contract"
        ],
        "model_contract": predecessor["preregistration"]["model_contract"],
        "source_only_capacity": predecessor["preregistration"][
            "source_only_capacity"
        ],
        "economic_contract": predecessor["preregistration"][
            "economic_contract"
        ],
        "semantic_encoder_development_gate": predecessor[
            "preregistration"
        ]["semantic_encoder_development_gate"],
        "conditional_rllm_contract": predecessor["preregistration"][
            "conditional_rllm_contract"
        ],
        "controls_and_statistics": predecessor["preregistration"][
            "controls_and_statistics"
        ],
        "final_test_and_eval_gates": predecessor["preregistration"][
            "final_test_and_eval_gates"
        ],
        "chronology": predecessor["preregistration"]["chronology"],
    }
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "candidate": {
            "id": POLICY_ID,
            "predecessor": rllm1_prereg.POLICY_ID,
            "stage": "operational_only_successor_before_first_model_forward",
            "profitability_claim": False,
        },
        "predecessor_terminal_evidence": {
            "execution_commit": RLLM1_EXECUTION_COMMIT,
            "terminal_record_commit": RLLM1_TERMINAL_RECORD_COMMIT,
            "runner_sha256": RLLM1_RUNNER_SHA256,
            "attempt": {
                "path": RLLM1_ATTEMPT.as_posix(),
                "sha256": RLLM1_ATTEMPT_SHA256,
                "attempt_hash": RLLM1_ATTEMPT_HASH,
            },
            "failure": {
                "path": RLLM1_FAILURE.as_posix(),
                "sha256": RLLM1_FAILURE_SHA256,
                "result_hash": RLLM1_FAILURE_RESULT_HASH,
                "stage": predecessor["failure"]["failure"]["stage"],
                "exception_message": predecessor["failure"]["failure"][
                    "exception_message"
                ],
            },
            "model_forwards_started": 0,
            "challenge_predictions_created": 0,
            "market_or_economic_payload_opened": False,
            "predecessor_rerun_authorized": False,
        },
        "inherited_scientific_contract": {
            "rllm1_preregistration": {
                "path": RLLM1_PREREGISTRATION.as_posix(),
                "sha256": RLLM1_PREREGISTRATION_SHA256,
                "manifest_hash": RLLM1_PREREGISTRATION_MANIFEST_HASH,
            },
            "contract_hash": canonical_hash(
                inherited_scientific_contract
            ),
            "payload": inherited_scientific_contract,
            "unchanged": [
                "D8 source authority and selected-subcard selector",
                "model ID revision files runtime and quantization",
                "all source redaction and policy prompts",
                "128-case memorization roster and candidate mappings",
                "single-token A-H scoring and lexical tie rule",
                "binomial families chance and Bonferroni threshold",
                "prompt limits VRAM cap and no-generation contract",
                "chronological train test eval and economic gates",
            ],
            "case_roster_hash": CASE_ROSTER_HASH,
        },
        "sole_operational_delta": {
            "rllm1_defect": (
                "required nonempty hf_device_map even when explicit "
                "single-device loading populated no advisory map"
            ),
            "rllm2_rule": (
                "device_map remains {'': 0}; model.device and the first "
                "model parameter must both be cuda:0; is_quantized and exact "
                "model class remain mandatory; an empty hf_device_map is "
                "accepted, while any nonempty map must contain only CUDA "
                "device zero and may not contain CPU disk or meta targets"
            ),
            "model_or_data_change": False,
            "challenge_or_threshold_change": False,
            "source_resample": False,
            "prompt_or_candidate_change": False,
            "market_or_outcome_information_used": False,
        },
        "one_shot_execution": {
            "attempt_path": RLLM2_ATTEMPT.as_posix(),
            "result_path": RLLM2_RESULT.as_posix(),
            "clean_head_equals_origin_main_required": True,
            "attempt_created_before_weight_load": True,
            "attempt_consumed_on_any_post_sentinel_failure": True,
            "output_override_allowed": False,
            "validate_only_loads_weights": False,
        },
        "terminal_actions": {
            "operational_or_memorization_failure": (
                "REJECT_PSIM_D8_RLLM2_BEFORE_NEXT_MARKET_STAGE_NO_REPAIR_"
                "RESAMPLE_MODEL_SWAP_OR_RERUN"
            ),
            "pass": (
                "ACCEPT_PSIM_D8_RLLM2_BASE_MEMORIZATION_GATE_"
                "SOURCE_FEATURES_ONLY"
            ),
        },
        "access_boundary": {
            "files_read": [
                RLLM1_PREREGISTRATION.as_posix(),
                RLLM1_ATTEMPT.as_posix(),
                RLLM1_FAILURE.as_posix(),
            ],
            "market_or_funding_paths_read": [],
            "market_rows_parsed": 0,
            "funding_rows_parsed": 0,
            "market_or_funding_payload_bytes_hashed": False,
            "model_loaded": False,
            "model_outputs_created": 0,
            "rewards_created": 0,
            "economic_metrics_computed": 0,
            "test_outcomes_opened": False,
            "eval_outcomes_opened": False,
        },
        "next_authorized_step": (
            "IMPLEMENT_REVIEW_COMMIT_AND_PUSH_RLLM2_OPERATIONAL_ONLY_"
            "ONE_SHOT_BASE_MEMORIZATION_GATE"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def write_preregistration(path: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    payload = build_preregistration()
    target = repository_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json_bytes(payload, pretty=True)
    if target.exists() and target.read_bytes() != encoded:
        raise RuntimeError(f"PSIM-D8-RLLM2 preregistration drift: {target}")
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
                "inherited_scientific_contract_hash": payload[
                    "inherited_scientific_contract"
                ]["contract_hash"],
                "next_authorized_step": payload["next_authorized_step"],
                "access_boundary": payload["access_boundary"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
