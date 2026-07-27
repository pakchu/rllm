from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import subprocess

import pytest

from training import run_psim_d8_rllm2_base_memorization_gate as gate


ATTEMPT = gate.REPO_ROOT / gate.DEFAULT_ATTEMPT
RESULT = gate.REPO_ROOT / gate.DEFAULT_OUTPUT
LOG = (
    gate.REPO_ROOT
    / "results/psim_d8_rllm2_base_memorization_gate_2026-07-27.log"
)
EXECUTION_COMMIT = "197ba160c5231ca11e9228bc73574bb157903dad"
ATTEMPT_SHA256 = (
    "e91b4c58797bd78d5062dff2c07d4363d8d897c8c3291620486f9c02aad42ea0"
)
RESULT_SHA256 = (
    "0abf3b5babe9e398e97721ddcc3e29b6d23cc742345cd5f804e78d507982818f"
)
LOG_SHA256 = (
    "45103890f0b95561665f1afcd40487b196c4254df27d8c4871b0f5d6cc80a34f"
)
ATTEMPT_HASH = (
    "b83c227d38a959a6ae2405700b5ea7b268e13a958c7b7c8282108e8169a2c759"
)
RESULT_HASH = (
    "8debfe4b37a6be1f65b306cce5b1408bf21a01a7f316254e4b42c2529a851ce3"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_attempt_result_and_log_are_exact_and_canonical() -> None:
    assert _sha256(ATTEMPT) == ATTEMPT_SHA256
    assert _sha256(RESULT) == RESULT_SHA256
    assert _sha256(LOG) == LOG_SHA256
    attempt = json.loads(ATTEMPT.read_text(encoding="utf-8"))
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    attempt_core = {
        key: value
        for key, value in attempt.items()
        if key != "attempt_hash"
    }
    result_core = {
        key: value
        for key, value in result.items()
        if key != "result_hash"
    }
    assert attempt["attempt_hash"] == _canonical_hash(attempt_core)
    assert attempt["attempt_hash"] == ATTEMPT_HASH
    assert result["result_hash"] == _canonical_hash(result_core)
    assert result["result_hash"] == RESULT_HASH
    assert result["attempt"]["attempt_hash"] == ATTEMPT_HASH
    assert result["attempt"]["sha256"] == ATTEMPT_SHA256


def test_executed_runner_and_frozen_contract_are_bound() -> None:
    attempt = json.loads(ATTEMPT.read_text(encoding="utf-8"))
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    runner = subprocess.run(
        [
            "git",
            "show",
            (
                f"{EXECUTION_COMMIT}:training/"
                "run_psim_d8_rllm2_base_memorization_gate.py"
            ),
        ],
        cwd=gate.REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    assert hashlib.sha256(runner).hexdigest() == attempt["runner_sha256"]
    assert attempt["execution_commit"] == EXECUTION_COMMIT
    assert result["execution_commit"] == EXECUTION_COMMIT
    assert attempt["preregistration_sha256"] == gate.PREREGISTRATION_SHA256
    assert attempt["preregistration_manifest_hash"] == (
        gate.PREREGISTRATION_MANIFEST_HASH
    )
    assert attempt["scientific_contract_hash"] == (
        gate.SCIENTIFIC_CONTRACT_HASH
    )
    assert attempt["case_roster_hash"] == gate.CASE_ROSTER_HASH
    assert result["preregistration"]["scientific_contract_hash"] == (
        gate.SCIENTIFIC_CONTRACT_HASH
    )
    assert result["challenge"]["case_roster_hash"] == gate.CASE_ROSTER_HASH


def test_complete_prediction_roster_passes_memorization_gate() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    challenge = result["challenge"]
    predictions = challenge["predictions"]
    assert challenge["decision"] == "pass"
    assert challenge["terminal_action"] == gate.PASS_ACTION
    assert challenge["source_feature_construction_authorized"] is True
    assert challenge["market_access_authorized"] is False
    assert len(predictions) == 128
    assert len({row["case_hash"] for row in predictions}) == 128
    assert Counter(row["protocol"] for row in predictions) == Counter(
        {"ethereum": 64, "bitcoin": 64}
    )
    assert Counter(
        (row["protocol"], bool(row["correct"]))
        for row in predictions
    ) == Counter(
        {
            ("ethereum", False): 54,
            ("ethereum", True): 10,
            ("bitcoin", False): 59,
            ("bitcoin", True): 5,
        }
    )
    assert sum(bool(row["correct"]) for row in predictions) == 15
    assert all(
        row["predicted_code"] in "ABCDEFGH"
        and set(row["code_logits"]) == set("ABCDEFGH")
        and all(math.isfinite(value) for value in row["code_logits"].values())
        for row in predictions
    )
    assert Counter(row["predicted_code"] for row in predictions) == Counter(
        {
            "A": 1,
            "B": 8,
            "C": 7,
            "D": 12,
            "E": 15,
            "F": 5,
            "G": 12,
            "H": 68,
        }
    )


def test_exact_statistics_do_not_reject_pretrained_memory_leakage_null() -> None:
    statistics = json.loads(RESULT.read_text(encoding="utf-8"))[
        "challenge"
    ]["statistics"]
    assert statistics["ethereum"] == {
        "trials": 64,
        "successes": 10,
        "accuracy": 0.15625,
        "chance": 0.125,
        "one_sided_exact_p": pytest.approx(0.27462604008264896),
        "bonferroni_reject_below": pytest.approx(0.01 / 3.0),
        "memorization_rejected": False,
    }
    assert statistics["bitcoin"] == {
        "trials": 64,
        "successes": 5,
        "accuracy": 0.078125,
        "chance": 0.125,
        "one_sided_exact_p": pytest.approx(0.915007790571787),
        "bonferroni_reject_below": pytest.approx(0.01 / 3.0),
        "memorization_rejected": False,
    }
    assert statistics["combined"] == {
        "trials": 128,
        "successes": 15,
        "accuracy": 0.1171875,
        "chance": 0.125,
        "one_sided_exact_p": pytest.approx(0.6450678052449308),
        "bonferroni_reject_below": pytest.approx(0.01 / 3.0),
        "memorization_rejected": False,
    }


def test_actual_cuda_placement_and_resource_caps_are_valid() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    metrics = result["model_metrics"]
    placement = metrics["placement"]
    assert metrics["gpu_name"] == "NVIDIA GeForce RTX 5090"
    assert metrics["cuda_device_count"] == 1
    assert metrics["bf16_supported"] is True
    assert placement["model_class"] == "Gemma4ForConditionalGeneration"
    assert placement["is_quantized"] is True
    assert placement["model_device"] == "cuda:0"
    assert placement["first_parameter_device"] == "cuda:0"
    assert placement["hf_device_map_advisory"] == {}
    assert placement["hf_device_map_normalized"] == {}
    assert placement["empty_hf_device_map_accepted"] is True
    assert placement["cuda_memory_allocated_after_load_bytes"] == 9_323_639_296
    assert metrics["peak_allocated_bytes"] == 19_078_689_792
    assert metrics["peak_reserved_bytes"] == 25_492_979_712
    assert metrics["peak_allocated_bytes"] < (
        metrics["maximum_peak_allocated_bytes"]
    )
    assert min(
        row["input_tokens"]
        for row in result["challenge"]["predictions"]
    ) == 159
    assert max(
        row["input_tokens"]
        for row in result["challenge"]["predictions"]
    ) == 10_291


def test_gate_opened_no_market_or_economic_payload() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    boundary = result["access_boundary"]
    assert boundary["market_or_funding_paths_read"] == []
    assert boundary["market_rows_parsed"] == 0
    assert boundary["funding_rows_parsed"] == 0
    assert boundary["market_or_funding_payload_bytes_hashed"] is False
    assert boundary["rewards_created"] == 0
    assert boundary["economic_metrics_computed"] == 0
    assert boundary["test_outcomes_opened"] is False
    assert boundary["eval_outcomes_opened"] is False
    assert result["challenge"]["market_access_authorized"] is False
