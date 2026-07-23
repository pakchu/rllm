from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from training import preregister_usdc_recipient_concentration_dislocation as urcd


def _rehash(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    payload["manifest_hash"] = urcd.canonical_hash(core)


def _assert_no_float(value: Any) -> None:
    if isinstance(value, dict):
        for child in value.values():
            _assert_no_float(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_no_float(child)
    else:
        assert not isinstance(value, float)


def test_policy_freezes_recipient_hhi_transition_and_execution() -> None:
    policy = urcd.policy_payload()
    assert policy["candidate"] == "URCD-72"
    assert policy["source"]["clock"] == "available_at"
    assert policy["source"]["sealed_from"] == "2024-01-01T00:00:00Z"
    assert policy["current_state"]["anchors_utc"] == [0, 6, 12, 18]
    assert policy["current_state"]["window"] == "D-24h < available_at <= D"
    assert policy["current_state"]["minimum_events"] == 4
    assert policy["current_state"]["minimum_distinct_recipients"] == 3
    assert policy["current_state"]["binary_float_forbidden"] is True
    assert policy["prior_panel"]["daily_endpoints"] == 180
    assert policy["prior_panel"]["minimum_valid"] == 120
    assert policy["state_transition"]["long"].startswith("enter DIFFUSE")
    assert policy["state_transition"]["short"].startswith("enter CONCENTRATED")
    assert policy["execution"]["hold_bars_5m"] == 864
    assert policy["execution"]["split_reservation"] == (
        "independent per split and control"
    )
    assert policy["mutable_parameters"] == []
    _assert_no_float(policy)


def test_support_controls_novelty_and_oos_extension_are_exact() -> None:
    policy = urcd.policy_payload()
    gates = policy["support_gates"]
    assert gates["train_total_minimum"] == 80
    assert gates["selection_total_minimum"] == 30
    assert gates["minimum_each_side_share"] == "1/5"
    assert gates["permutation_maximum_exact_jaccard"] == "7/20"
    assert gates["failure_action"].startswith("retire_URCD_72")
    assert policy["controls"]["order"] == list(urcd.SOURCE_CONTROLS)
    assert policy["controls"]["routing_selectivity"] == [
        "recipient_year_permutation",
        "amount_year_permutation",
    ]
    assert policy["novelty_gates"]["maximum_exact_entry_jaccard"] == "1/10"
    assert policy["novelty_gates"]["maximum_bidirectional_containment"] == "2/5"
    assert policy["oos_extension"]["current_scope_ends_pre2024"] is True
    assert policy["oos_extension"][
        "each_stage_source_frozen_before_corresponding_outcomes"
    ] is True


def test_config_repair_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="frozen policy drift"):
        urcd._validate_config(
            replace(urcd.FROZEN_CONFIG, hold_elapsed_hours=48)
        )


def test_static_preregistration_is_ineligible_and_opens_nothing() -> None:
    payload = urcd.build_preregistration(verify_bindings=False)
    urcd.validate_preregistration(payload, verify_bindings=False)
    assert payload["verification_mode"] == "static_test_fixture"
    assert payload["artifact_eligible"] is False
    assert payload["outcome_boundary"] == urcd.STATIC_BOUNDARY
    assert payload["source_values_or_incidence_opened"] is False
    assert payload["comparator_rows_opened_during_preregistration"] is False
    assert payload["outcomes_opened"] is False
    assert payload["performance_values_opened"] is False
    assert payload["source_binding"]["value_rows_read_during_preregistration"] == 0
    assert all(
        row["value_rows_read_during_preregistration"] == 0
        for row in payload["comparator_bindings"]
    )


def test_verified_preregistration_hashes_headers_but_decodes_no_rows() -> None:
    payload = urcd.build_preregistration(verify_bindings=True)
    urcd.validate_preregistration(payload, verify_bindings=True)
    assert payload["verification_mode"] == "verified_hashes_and_headers_uncommitted"
    assert payload["artifact_eligible"] is False
    assert payload["outcome_boundary"] == urcd.VERIFIED_UNCOMMITTED_BOUNDARY
    assert len(payload["comparator_bindings"]) == 9
    assert payload["source_binding"]["manifest_rows"] == 266_362
    members = {
        (row["candidate"], control)
        for row in payload["comparator_bindings"]
        for control in row["controls"]
    }
    assert ("AMTR-48", "cross_minter") in members
    assert ("SQFD-6", "no_participation") in members
    assert ("FCCM-72", "primary") in members


def test_tampering_and_forged_eligibility_fail_closed() -> None:
    payload = urcd.build_preregistration(verify_bindings=False)
    payload["policy"]["execution"]["hold_bars_5m"] = 576
    _rehash(payload)
    with pytest.raises(RuntimeError, match="binding drift"):
        urcd.validate_preregistration(payload, verify_bindings=False)

    payload = urcd.build_preregistration(verify_bindings=False)
    payload["git_protocol_subprocess_calls"] = 999
    _rehash(payload)
    with pytest.raises(RuntimeError, match="git protocol call counter drift"):
        urcd.validate_preregistration(payload, verify_bindings=False)

    payload = urcd.build_preregistration(verify_bindings=False)
    payload["artifact_eligible"] = True
    payload["verification_mode"] = "verified_hashes_headers_and_commit_guard"
    payload["outcome_boundary"] = dict(urcd.EXPECTED_BOUNDARY)
    payload["git_protocol_subprocess_calls"] = 2
    _rehash(payload)
    with pytest.raises(RuntimeError, match="validated only by the write path"):
        urcd.validate_preregistration(payload, verify_bindings=False)


@pytest.mark.parametrize("path", ["../escape.json", "/tmp/escape.json"])
def test_repository_path_escape_fails_closed(path: str) -> None:
    with pytest.raises(RuntimeError, match="repository-relative"):
        urcd._repository_path(path)


def test_protocol_commit_guard_rejects_untracked_and_dirty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(urcd, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(urcd, "SCRIPT_PATH", Path("preregister.py"))
    monkeypatch.setattr(urcd, "TEST_PATH", Path("test_preregister.py"))
    (tmp_path / "preregister.py").write_text("script\n")
    (tmp_path / "test_preregister.py").write_text("test\n")
    calls: list[tuple[str, ...]] = []

    def untracked(*args: str) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 1, "", "untracked")

    monkeypatch.setattr(urcd, "_git_check", untracked)
    with pytest.raises(RuntimeError, match="not committed"):
        urcd._assert_protocol_committed()
    assert calls == [
        (
            "ls-files",
            "--error-unmatch",
            "--",
            "preregister.py",
            "test_preregister.py",
        )
    ]

    calls.clear()
    results = iter(
        [
            subprocess.CompletedProcess((), 0, "", ""),
            subprocess.CompletedProcess((), 1, "", "dirty"),
        ]
    )

    def dirty(*args: str) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return next(results)

    monkeypatch.setattr(urcd, "_git_check", dirty)
    with pytest.raises(RuntimeError, match="differs from HEAD"):
        urcd._assert_protocol_committed()
    assert calls[-1][:3] == ("diff", "--quiet", "HEAD")


def test_write_is_no_clobber_and_byte_immutable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_build = urcd._build_preregistration
    monkeypatch.setattr(urcd, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(urcd, "SCRIPT_PATH", Path("preregister.py"))
    monkeypatch.setattr(urcd, "TEST_PATH", Path("test_preregister.py"))
    (tmp_path / "preregister.py").write_text("frozen\n")
    (tmp_path / "test_preregister.py").write_text("frozen test\n")
    monkeypatch.setattr(urcd, "_assert_protocol_committed", lambda: None)

    def fake_build(
        *, verify_bindings: bool, artifact_eligible: bool = False, git_calls: int = 0
    ) -> dict[str, Any]:
        assert verify_bindings is True
        payload = original_build(verify_bindings=False)
        payload["artifact_eligible"] = artifact_eligible
        payload["git_protocol_subprocess_calls"] = git_calls
        payload["verification_mode"] = "verified_hashes_headers_and_commit_guard"
        payload["outcome_boundary"] = dict(urcd.EXPECTED_BOUNDARY)
        _rehash(payload)
        return payload

    monkeypatch.setattr(urcd, "_build_preregistration", fake_build)
    monkeypatch.setattr(urcd, "_validate_preregistration", lambda *_a, **_k: None)
    cfg = urcd.Config()
    first, status = urcd.write_preregistration(cfg)
    assert status == "created"
    second, status = urcd.write_preregistration(cfg)
    assert status == "verified_existing"
    assert first == second
    path = tmp_path / cfg.output
    path.write_bytes(b"{}\n")
    with pytest.raises(RuntimeError, match="immutable"):
        urcd.write_preregistration(cfg)

    with pytest.raises(RuntimeError, match="must equal DEFAULT_OUTPUT"):
        urcd.write_preregistration(urcd.Config(output="other.json"))
