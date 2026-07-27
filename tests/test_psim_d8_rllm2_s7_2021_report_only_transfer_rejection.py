from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from training import (
    preregister_psim_d8_rllm2_s7_2021_report_only_transfer as prereg,
)
from training import (
    run_psim_d8_rllm2_s7_2021_report_only_transfer as runner,
)

ATTEMPT = runner.repository_path(prereg.ATTEMPT_PATH)
RESULT = runner.repository_path(prereg.RESULT_PATH)
EXECUTION_COMMIT = "538d7f590c16859823c44451b34e1f71e2253bd8"
RUNNER_SHA256 = (
    "51aa648b15394dc1d70c0cbfa82d885672ecdddf6eac126c0b49f24dc6bfa114"
)
CORE_SHA256 = (
    "bda2354dbd846fb72990f8813630c1fbfdf6ce2d32ed26d66dbf8dbe2d08792b"
)
ATTEMPT_FILE_SHA256 = (
    "922de8d8da63cc45c9536409177af80809d9d8b17bf60e9024e68691bedd8c0f"
)
RESULT_FILE_SHA256 = (
    "c061b82438a5b207801b321b864a252564fcd754f3ce09a4ff4d427c3327480a"
)
ATTEMPT_HASH = (
    "5ad95e0af8cb202e6dbe9f1ba2c3c5d8efbf509327aa0d88d90ae2e89cd0ada8"
)
RESULT_HASH = (
    "545b58bd5346d6fa5c87195e06692432a0b0447cbc31a56a917830b62da59e71"
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _self_hashed(
    path: Path,
    *,
    field: str,
    expected: str,
) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    core = {
        key: value
        for key, value in payload.items()
        if key != field
    }
    assert payload[field] == runner._canonical_hash(core)
    assert payload[field] == expected
    return payload


def test_attempt_result_and_executed_sources_are_exact() -> None:
    assert _file_sha256(ATTEMPT) == ATTEMPT_FILE_SHA256
    assert _file_sha256(RESULT) == RESULT_FILE_SHA256
    attempt = _self_hashed(
        ATTEMPT,
        field="attempt_hash",
        expected=ATTEMPT_HASH,
    )
    result = _self_hashed(
        RESULT,
        field="result_hash",
        expected=RESULT_HASH,
    )
    for path, expected in (
        (prereg.RUNNER_PATH, RUNNER_SHA256),
        (runner.CORE_PATH, CORE_SHA256),
    ):
        committed = subprocess.run(
            ["git", "show", f"{EXECUTION_COMMIT}:{path.as_posix()}"],
            cwd=prereg.REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout
        assert hashlib.sha256(committed).hexdigest() == expected
    assert attempt["execution_commit"] == EXECUTION_COMMIT
    assert attempt["runner_sha256"] == RUNNER_SHA256
    assert attempt["evaluator_core_sha256"] == CORE_SHA256
    assert result["execution_commit"] == EXECUTION_COMMIT
    assert result["runner_sha256"] == RUNNER_SHA256
    assert result["evaluator_core_sha256"] == CORE_SHA256
    assert result["attempt_hash"] == ATTEMPT_HASH


def test_attempt_precedes_every_outcome_access() -> None:
    attempt = json.loads(ATTEMPT.read_text(encoding="utf-8"))

    assert attempt["authorization"] == {
        "open_exact_2021_market_and_funding_stage": True,
        "evaluate_fixed_41_policy_family": True,
        "compute_fixed_45_economic_metric_sets": True,
        "select_or_repair_from_2021": False,
        "open_2022_or_later_outcomes": False,
        "load_or_forward_model": False,
    }
    assert attempt["access_boundary_at_attempt"] == {
        "raw_market_or_funding_paths_opened_or_read": [],
        "market_or_funding_payload_bytes_hashed": False,
        "market_rows_parsed": 0,
        "funding_rows_parsed": 0,
        "economic_metric_sets_computed": 0,
        "2021_policy_specific_outcomes_opened": False,
        "2022_or_later_outcomes_opened": False,
        "model_loaded": False,
        "model_forwards_started": 0,
    }


def test_fixed_primary_fails_strict_economic_and_robustness_gate() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    evaluation = result["evaluation"]
    primary = evaluation["primary_metrics"]
    expected = {
        "base_6bp": (
            -0.9194501425959567,
            -0.920058191332546,
            36.21166717186114,
            -0.025407783269572635,
            153,
            163,
        ),
        "stress_10bp": (
            -6.958111853776794,
            -6.9625699046461875,
            37.88499891512034,
            -0.18378171054578926,
            153,
            163,
        ),
        "delayed_5m_6bp": (
            -3.5804566337820476,
            -3.582792298577976,
            36.71451148242969,
            -0.0975851823683553,
            153,
            163,
        ),
        "first_half_6bp": (
            23.046843185904176,
            51.96882575268384,
            23.844907160378455,
            2.1794517966938067,
            80,
            84,
        ),
        "second_half_6bp": (
            -17.881104725317286,
            -32.36548382632606,
            36.21166717186122,
            -0.8937860737733752,
            74,
            81,
        ),
    }
    for label, values in expected.items():
        metrics = primary[label]
        assert metrics["absolute_return_pct"] == pytest.approx(values[0])
        assert metrics["cagr_pct"] == pytest.approx(values[1])
        assert metrics["strict_mdd_pct"] == pytest.approx(values[2])
        assert metrics["cagr_to_strict_mdd"] == pytest.approx(values[3])
        assert metrics["directional_entries_including_flips"] == values[4]
        assert (
            metrics["all_target_changes_including_terminal_flatten"]
            == values[5]
        )
    base = primary["base_6bp"]
    assert base["nonflat_interval_count"] == 356
    assert base["long_share_of_nonflat"] == pytest.approx(141 / 356)
    assert base["short_share_of_nonflat"] == pytest.approx(215 / 356)
    assert evaluation["robustness_semantics"] == {
        "half_metrics": (
            "standalone_reset_to_flat_equity_1_at_each_half_start"
        ),
        "continuous_full_path_subperiod_attribution": False,
    }


def test_familywise_inference_comparator_and_gate_are_exact() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    evaluation = result["evaluation"]
    inference = evaluation["familywise_max_stat"]

    assert evaluation["family_ids"] == list(prereg.FAMILY_IDS)
    assert set(evaluation["base_family_metrics"]) == set(prereg.FAMILY_IDS)
    assert inference["family_ids"] == list(prereg.FAMILY_IDS)
    assert inference["draws"] == 100_000
    assert inference["seed"] == 20_260_725
    assert inference["weeks"] == 53
    assert inference["local_p"][prereg.PRIMARY_POLICY_ID] == pytest.approx(
        0.5102148978510215
    )
    assert inference["p_max"][prereg.PRIMARY_POLICY_ID] == 1.0
    strongest = evaluation["strongest_nonsemantic_control"]
    assert strongest["policy_id"] == "ethereum_only"
    assert strongest["metrics"]["absolute_return_pct"] == pytest.approx(
        2.533532414566597
    )
    assert strongest["metrics"]["cagr_to_strict_mdd"] == pytest.approx(
        1.2007884159615607
    )
    assert evaluation["gate"]["checks"] == {
        "action_code_permutation_schedule_identity": True,
        "base_absolute_return_positive": False,
        "base_cagr_to_strict_mdd_minimum": False,
        "beat_strongest_nonsemantic_absolute_return": False,
        "beat_strongest_nonsemantic_cagr_to_strict_mdd": False,
        "delayed_absolute_return_positive": False,
        "familywise_p_max_strictly_below": False,
        "first_half_absolute_return_positive": True,
        "minimum_long_share": True,
        "minimum_nonflat_intervals": True,
        "minimum_short_share": True,
        "second_half_absolute_return_positive": False,
        "stress_absolute_return_positive": False,
    }
    assert evaluation["gate"]["passed"] is False


def test_rejection_boundary_forbids_repair_promotion_and_later_outcomes() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))

    assert result["decision"] == "reject"
    assert result["terminal_action"] == (
        "RETIRE_UNCHANGED_S6R1_HYPOTHESIS_NO_2021_REPAIR"
    )
    assert result[
        "authorize_separate_forward_or_live_candidate_preregistration"
    ] is False
    assert result["authorize_live_promotion"] is False
    assert result["authorize_2022_or_later_outcomes"] is False
    assert result["selection_or_repair_from_2021"] is False
    assert result["source_bindings"] == runner._source_bindings()
    assert result["access_boundary"] == runner._expected_access_boundary()
