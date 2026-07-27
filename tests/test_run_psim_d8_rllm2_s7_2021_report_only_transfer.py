from __future__ import annotations

import json
from pathlib import Path

import pytest

from training import (
    preregister_psim_d8_rllm2_s7_2021_report_only_transfer as prereg,
)
from training import (
    run_psim_d8_rllm2_s7_2021_report_only_transfer as runner,
)

BINDINGS = {
    "execution_commit": "a" * 40,
    "runner_sha256": "b" * 64,
    "evaluator_core_sha256": "c" * 64,
}


def _metrics() -> dict[str, object]:
    return {
        "absolute_return": 0.1,
        "absolute_return_pct": 10.0,
        "cagr": 0.1,
        "cagr_pct": 10.0,
        "strict_mdd": 0.05,
        "strict_mdd_pct": 5.0,
        "cagr_to_strict_mdd": 2.0,
        "directional_entries_including_flips": 100,
        "all_target_changes_including_terminal_flatten": 101,
    }


def _evaluation(*, passed: bool = True) -> dict[str, object]:
    checks = {"fixed_gate": passed}
    return {
        "protocol_version": (
            "psim_s7_report_only_transfer_calculation_v1"
        ),
        "stage": "2021",
        "robustness_semantics": {
            "half_metrics": (
                "standalone_reset_to_flat_equity_1_at_each_half_start"
            ),
            "continuous_full_path_subperiod_attribution": False,
        },
        "primary_policy_id": prereg.PRIMARY_POLICY_ID,
        "family_ids": list(prereg.FAMILY_IDS),
        "base_family_metrics": {
            policy_id: _metrics() for policy_id in prereg.FAMILY_IDS
        },
        "familywise_max_stat": {
            "draws": prereg.STATISTICAL_DRAWS,
            "seed": prereg.STATISTICAL_SEED,
            "family_ids": list(prereg.FAMILY_IDS),
            "p_max": {
                policy_id: 0.1 for policy_id in prereg.FAMILY_IDS
            },
        },
        "primary_metrics": {
            "base_6bp": _metrics(),
            "stress_10bp": _metrics(),
            "delayed_5m_6bp": _metrics(),
            "first_half_6bp": _metrics(),
            "second_half_6bp": _metrics(),
        },
        "strongest_nonsemantic_control": {
            "policy_id": prereg.NONSEMANTIC_CONTROL_IDS[0],
            "metrics": _metrics(),
        },
        "action_code_permutation_schedule_identity": True,
        "gate": {"checks": checks, "passed": passed},
    }


def test_validate_preregistration_is_exact_and_outcome_closed() -> None:
    payload = runner.validate_preregistration()

    assert payload["manifest_hash"] == (
        runner.PREREGISTRATION_MANIFEST_HASH
    )
    assert runner._sha256_file(prereg.DEFAULT_OUTPUT) == (
        runner.PREREGISTRATION_SHA256
    )
    assert payload["frozen_schedule_family"]["family_count"] == 41
    boundary = payload["access_boundary"]
    assert boundary[
        "raw_market_or_funding_paths_opened_or_read_before_attempt"
    ] == []
    assert boundary["market_or_funding_payload_bytes_hashed"] is False
    assert boundary["market_rows_parsed"] == 0
    assert boundary["funding_rows_parsed"] == 0


def test_attempt_is_self_hashed_and_precedes_outcome_access() -> None:
    payload = runner._attempt_payload(BINDINGS)
    core = {
        key: value
        for key, value in payload.items()
        if key != "attempt_hash"
    }

    assert payload["attempt_hash"] == runner._canonical_hash(core)
    assert payload["authorization"] == {
        "open_exact_2021_market_and_funding_stage": True,
        "evaluate_fixed_41_policy_family": True,
        "compute_fixed_45_economic_metric_sets": True,
        "select_or_repair_from_2021": False,
        "open_2022_or_later_outcomes": False,
        "load_or_forward_model": False,
    }
    assert payload["access_boundary_at_attempt"][
        "raw_market_or_funding_paths_opened_or_read"
    ] == []
    assert payload["access_boundary_at_attempt"][
        "market_or_funding_payload_bytes_hashed"
    ] is False


def test_write_once_accepts_identity_and_rejects_drift(
    tmp_path: Path,
) -> None:
    output = tmp_path / "artifact"
    runner._write_once(output, b"first")
    runner._write_once(output, b"first")

    assert output.read_bytes() == b"first"
    with pytest.raises(RuntimeError, match="write-once artifact drift"):
        runner._write_once(output, b"second")


def test_write_once_rejects_concurrent_different_writer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "artifact"

    def rival_link(_source: Path, target: Path) -> None:
        target.write_bytes(b"rival")
        raise FileExistsError

    monkeypatch.setattr(runner.os, "link", rival_link)

    with pytest.raises(RuntimeError, match="concurrent write-once drift"):
        runner._write_once(output, b"ours")


def test_execute_writes_attempt_before_stage_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    attempt = {
        **BINDINGS,
        "attempt_hash": "d" * 64,
    }
    monkeypatch.setattr(runner, "validate_preregistration", dict)
    monkeypatch.setattr(
        runner,
        "assert_committed_clean_runner",
        lambda: dict(BINDINGS),
    )

    def write_attempt(_bindings: object) -> dict[str, str]:
        events.append("attempt")
        return attempt

    monkeypatch.setattr(runner, "_write_or_validate_attempt", write_attempt)
    monkeypatch.setattr(
        runner,
        "_verify_post_attempt_inputs",
        lambda _bindings: events.append("verify"),
    )

    def reject_load(*_args: object, **_kwargs: object) -> None:
        events.append("stage_load")
        raise RuntimeError("sentinel stage load")

    monkeypatch.setattr(
        runner.stage_sources,
        "load_stage_source",
        reject_load,
    )

    with pytest.raises(RuntimeError, match="sentinel stage load"):
        runner.execute()

    assert events == ["attempt", "verify", "stage_load"]


def test_full_execute_publishes_and_resumes_then_rejects_tamper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    attempt_path = tmp_path / "attempt.json"
    result_path = tmp_path / "result.json"
    monkeypatch.setattr(prereg, "ATTEMPT_PATH", attempt_path)
    monkeypatch.setattr(prereg, "RESULT_PATH", result_path)
    monkeypatch.setattr(runner, "validate_preregistration", dict)
    monkeypatch.setattr(
        runner,
        "assert_committed_clean_runner",
        lambda: dict(BINDINGS),
    )
    monkeypatch.setattr(
        runner,
        "_verify_post_attempt_inputs",
        lambda _bindings: None,
    )
    stage_calls: list[str] = []

    class Sized:
        def __init__(self, length: int) -> None:
            self.length = length

        def __len__(self) -> int:
            return self.length

    def sized_stage_load(
        *_args: object,
        **_kwargs: object,
    ) -> tuple[Sized, Sized, dict[str, object]]:
        stage_calls.append("load")
        return (
            Sized(105_120),
            Sized(1_095),
            {
                "stage": "2021",
                "manifest_hash": prereg.STAGE_SOURCE_MANIFEST_HASH,
                "market_rows": 105_120,
                "funding_rows": 1_095,
            },
        )

    monkeypatch.setattr(
        runner.stage_sources,
        "load_stage_source",
        sized_stage_load,
    )
    original_sha = runner._sha256_file

    def bound_sha(path: str | Path) -> str:
        if Path(path) == prereg.MARKET_PATH:
            return prereg.MARKET_GZIP_SHA256
        if Path(path) == prereg.FUNDING_PATH:
            return prereg.FUNDING_GZIP_SHA256
        return original_sha(path)

    monkeypatch.setattr(runner, "_sha256_file", bound_sha)
    monkeypatch.setattr(
        runner,
        "_load_schedule_family",
        lambda: (object(), object(), object(), object()),
    )
    monkeypatch.setattr(
        runner.transfer,
        "evaluate_transfer",
        lambda *_args, **_kwargs: _evaluation(),
    )

    first = runner.execute()
    second = runner.execute()

    assert first == second
    assert first["decision"] == "pass"
    assert first["access_boundary"] == runner._expected_access_boundary()
    assert stage_calls == ["load", "load"]
    assert attempt_path.is_file()
    assert result_path.is_file()

    tampered = json.loads(result_path.read_text(encoding="utf-8"))
    tampered["evaluation"]["primary_metrics"]["base_6bp"][
        "absolute_return"
    ] = -999.0
    core = {
        key: value
        for key, value in tampered.items()
        if key != "result_hash"
    }
    tampered["result_hash"] = runner._canonical_hash(core)
    result_path.write_bytes(runner._canonical_bytes(tampered, pretty=True))

    with pytest.raises(RuntimeError, match="existing result changed"):
        runner.execute()
    assert stage_calls == ["load", "load", "load"]
