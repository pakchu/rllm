from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from training import (
    preregister_funding_currency_custody_mobility_consensus as fccm,
)


def _rehash(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    payload["manifest_hash"] = fccm.canonical_hash(core)


def _assert_no_float(value: Any) -> None:
    if isinstance(value, dict):
        for child in value.values():
            _assert_no_float(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_no_float(child)
    else:
        assert not isinstance(value, float)


def test_policy_freezes_exact_pair_rank_sponsor_and_transition_contract() -> None:
    policy = fccm.policy_payload()

    assert policy["candidate"] == "FCCM-72"
    assert policy["arithmetic"]["representation"].startswith("exact rational")
    assert policy["source_alignment"]["exact_pair"] == ["fUSD", "fBTC"]
    assert policy["source_alignment"]["exact_lag"] == "H-24h"
    assert policy["source_alignment"]["first_valid_after_reset"] == "baseline only"
    assert len(policy["source_alignment"]["batch_order"]) == 4
    assert policy["bitfinex_normalization"]["strict_prior_valid_pairs"] == 720
    assert policy["bitfinex_normalization"]["current_batch_excluded"] is True
    assert policy["wbtc_sponsorship"]["window"] == "D-14d < available_at <= D"
    assert policy["wbtc_sponsorship"]["prior_daily_anchors"] == 180
    assert policy["wbtc_sponsorship"]["membership_time"].startswith("available_at")
    assert policy["state_machine"]["state_updates_when_wbtc_inactive"] is True
    assert policy["state_machine"]["inactive_suppression_queued"] is False
    assert policy["execution"]["exact_grid_signal_still_adds_5m"] is True
    assert policy["execution"]["hold_elapsed_hours"] == 72
    assert policy["execution"]["notional_exposure"] == "1/2"
    _assert_no_float(policy)


def test_policy_freezes_support_controls_identities_novelty_and_economics() -> None:
    policy = fccm.policy_payload()

    support = policy["source_support_gates"]
    assert support["train_total_minimum"] == 60
    assert support["selection_total_minimum"] == 24
    assert support["train_maximum_month_share"] == "3/20"
    assert support["selection_maximum_month_share"] == "1/5"
    wbtc_share = support["wbtc_raw_transition_active_share"]
    assert wbtc_share["denominator"] == (
        "raw_directional_bitfinex_transitions_before_nonoverlap"
    )
    assert wbtc_share["train"] == {"minimum": "1/5", "maximum": "7/10"}
    assert wbtc_share["selection"] == {"minimum": "1/5", "maximum": "7/10"}
    component_share = support["bitfinex_component_vote_with_accepted_side_share"]
    assert component_share["denominator"] == "accepted_entries_within_split"
    assert component_share["train_minimum_each_component"] == "7/20"
    assert component_share["selection_minimum_each_component"] == "7/20"
    assert support["post_2023_source_value_rows"] == 0

    controls = policy["controls"]
    assert len(controls["causal"]) == 10
    assert controls["each_causal_control_owns_nonoverlap_scheduler"] is True
    assert "FCCM-72|random-side|canonical_entry" in controls["random_side"]
    assert len(controls["noncausal_source_placebos"]) == 2
    assert controls["placebo_multiset_preserved"] is True
    assert controls["placebo_rng_or_tunable_seed"] is False
    assert controls["placebo_clock_or_economics_forbidden"] is True

    identities = policy["identities"]
    assert "FCCM-72|candidate|" in identities["primary"]
    assert "FCCM-72|control-row|" in identities["control"]
    assert "FCCM-72|comparator|" in identities["comparator_row"]

    novelty = policy["novelty"]
    assert novelty["maximum_signless_exact_entry_jaccard"] == "1/10"
    assert novelty["maximum_bidirectional_signless_containment"] == "7/20"
    assert novelty["maximum_absolute_signed_exposure_correlation"] == "2/5"
    assert novelty["matching_run_independently_both_directions"] is True
    assert novelty["duplicates_or_overlapping_group_intervals"] == "fail closed"

    economics = policy["economics"]
    assert economics["base_cagr_to_strict_mdd_minimum"] == "3"
    assert economics["stress_cagr_to_strict_mdd_minimum"] == "5/2"
    assert economics["strict_mdd_pct_maximum"] == "15"
    assert economics["weekly_sign_draws"] == 100_000
    assert economics["weekly_sign_draw_indices"] == "0..99999"
    assert economics["weekly_sign_decimal_precision"] == 50
    assert economics["selection_sealed_on_train_failure"] is True
    assert policy["mutable_parameters"] == []


def test_source_and_comparator_cohort_is_exactly_frozen() -> None:
    assert fccm.MECHANISM_DECISION_SHA256 == fccm.sha256_file(
        fccm.MECHANISM_DECISION
    )
    assert len(fccm.COMPARATOR_SPECS) == 5
    assert [spec["name"] for spec in fccm.COMPARATOR_SPECS] == [
        "bfmwd_primary_variants",
        "wcdr_primary",
        "wtsl_primary",
        "wscf_primary",
        "live_portfolio_pure_clocks",
    ]
    assert fccm.COMPARATOR_SPECS[0]["groups"] == (
        "bfmwd_w12_d3_z10_h12",
        "bfmwd_w24_d3_z10_h12",
        "bfmwd_w12_d6_z10_h12",
        "bfmwd_w24_d6_z10_h12",
    )
    assert fccm.COMPARATOR_SPECS[-1]["groups"] == (
        "live:cand_rex_veto_7",
        "live:new_long_minimal_funding_premium",
        "live:oi_upbit_ratio288_low",
    )
    assert "frr" not in fccm.BITFINEX_ALLOWED_COLUMNS
    assert "funding_below_threshold" not in fccm.BITFINEX_ALLOWED_COLUMNS
    assert "event_sign" not in fccm.WBTC_ALLOWED_COLUMNS


def test_static_preregistration_is_ineligible_and_opens_nothing() -> None:
    payload = fccm.build_preregistration(verify_bindings=False)
    fccm.validate_preregistration(payload, verify_bindings=False)

    assert payload["verification_mode"] == "static_test_fixture"
    assert payload["artifact_eligible"] is False
    assert payload["outcome_boundary"] == fccm.STATIC_BOUNDARY
    assert payload["fccm_source_values_or_incidence_opened"] is False
    assert payload["comparator_rows_opened_during_preregistration"] is False
    assert payload["outcomes_opened"] is False
    assert payload["performance_values_opened"] is False
    assert all(
        source["value_rows_read_during_preregistration"] == 0
        for source in payload["source_bindings"].values()
    )
    assert all(
        comparator["value_rows_read_during_preregistration"] == 0
        for comparator in payload["comparator_bindings"]
    )


def test_verified_preregistration_checks_hashes_headers_and_zero_boundaries() -> None:
    payload = fccm.build_preregistration(verify_bindings=True)
    fccm.validate_preregistration(payload, verify_bindings=True)

    assert payload["artifact_eligible"] is False
    assert payload["verification_mode"] == (
        "verified_hashes_and_headers_uncommitted"
    )
    assert payload["source_bindings"]["bitfinex"]["manifest_rows"] == 70_116
    assert payload["source_bindings"]["wbtc"]["manifest_rows"] == 993
    assert payload["boundary_ledger"]["manifest_hash"] == (
        fccm.BOUNDARY_MANIFEST_HASH
    )
    assert payload["outcome_boundary"] == fccm.VERIFIED_UNCOMMITTED_BOUNDARY
    assert payload["outcome_boundary"][
        "git_protocol_subprocess_calls_during_artifact_write"
    ] == 0


def test_validation_does_not_reopen_verified_source_or_comparator_containers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = fccm.build_preregistration(verify_bindings=True)
    original = fccm._build_preregistration
    calls: list[bool] = []

    def guarded_build(*, verify_bindings: bool) -> dict[str, Any]:
        calls.append(verify_bindings)
        assert verify_bindings is False
        return original(verify_bindings=False)

    monkeypatch.setattr(fccm, "_build_preregistration", guarded_build)
    fccm.validate_preregistration(payload, verify_bindings=True)
    assert calls == [False]


def test_policy_disclosure_boundary_and_schema_tampering_fail_closed() -> None:
    payload = fccm.build_preregistration(verify_bindings=False)
    payload["policy"]["execution"]["hold_elapsed_hours"] = 48
    _rehash(payload)
    with pytest.raises(RuntimeError, match="frozen policy drift"):
        fccm.validate_preregistration(payload, verify_bindings=False)

    payload = fccm.build_preregistration(verify_bindings=False)
    payload["prior_research_disclosure"]["globally_clean_room"] = True
    _rehash(payload)
    with pytest.raises(RuntimeError, match="prior-research disclosure drift"):
        fccm.validate_preregistration(payload, verify_bindings=False)

    payload = fccm.build_preregistration(verify_bindings=False)
    payload["outcomes_opened"] = True
    _rehash(payload)
    with pytest.raises(RuntimeError, match="boundary opened"):
        fccm.validate_preregistration(payload, verify_bindings=False)

    payload = fccm.build_preregistration(verify_bindings=False)
    payload["unexpected_repair"] = True
    _rehash(payload)
    with pytest.raises(RuntimeError, match="binding drift"):
        fccm.validate_preregistration(payload, verify_bindings=False)


def test_public_validation_cannot_bless_forged_eligibility_without_git_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = fccm.build_preregistration(verify_bindings=True)
    payload["verification_mode"] = "verified_hashes_headers_and_commit_guard"
    payload["artifact_eligible"] = True
    payload["outcome_boundary"] = dict(fccm.EXPECTED_BOUNDARY)
    _rehash(payload)
    monkeypatch.setattr(
        fccm,
        "_assert_protocol_committed",
        lambda: (_ for _ in ()).throw(
            AssertionError("public validator attempted to substitute a later Git guard")
        ),
    )
    with pytest.raises(RuntimeError, match="validated only by the write path"):
        fccm.validate_preregistration(payload, verify_bindings=True)


@pytest.mark.parametrize("path", ["../escape.json", "/tmp/escape.json"])
def test_repository_path_escape_fails_closed(path: str) -> None:
    with pytest.raises(RuntimeError, match="repository-relative"):
        fccm._repository_path(path)


def test_protocol_commit_guard_rejects_untracked_or_dirty_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fccm, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(fccm, "SCRIPT_PATH", Path("preregister.py"))
    monkeypatch.setattr(fccm, "TEST_PATH", Path("test_preregister.py"))
    (tmp_path / "preregister.py").write_text("script\n")
    (tmp_path / "test_preregister.py").write_text("test\n")

    calls: list[tuple[str, ...]] = []

    def untracked(*args: str) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 1, "", "untracked")

    monkeypatch.setattr(fccm, "_git_check", untracked)
    with pytest.raises(RuntimeError, match="not committed"):
        fccm._assert_protocol_committed()
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

    monkeypatch.setattr(fccm, "_git_check", dirty)
    with pytest.raises(RuntimeError, match="differs from HEAD"):
        fccm._assert_protocol_committed()
    assert calls == [
        (
            "ls-files",
            "--error-unmatch",
            "--",
            "preregister.py",
            "test_preregister.py",
        ),
        (
            "diff",
            "--quiet",
            "HEAD",
            "--",
            "preregister.py",
            "test_preregister.py",
        ),
    ]

    calls.clear()
    results = iter(
        [
            subprocess.CompletedProcess((), 0, "", ""),
            subprocess.CompletedProcess((), 0, "", ""),
        ]
    )
    assert fccm._assert_protocol_committed() is None
    assert calls == [
        (
            "ls-files",
            "--error-unmatch",
            "--",
            "preregister.py",
            "test_preregister.py",
        ),
        (
            "diff",
            "--quiet",
            "HEAD",
            "--",
            "preregister.py",
            "test_preregister.py",
        ),
    ]


def test_write_is_raw_byte_immutable_and_no_clobber(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_build = fccm._build_preregistration
    monkeypatch.setattr(fccm, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(fccm, "SCRIPT_PATH", Path("preregister.py"))
    (tmp_path / "preregister.py").write_text("frozen protocol\n")
    monkeypatch.setattr(fccm, "_assert_protocol_committed", lambda: None)

    def fake_verified_build(*, verify_bindings: bool) -> dict[str, Any]:
        assert verify_bindings is True
        payload = original_build(verify_bindings=False)
        _rehash(payload)
        return payload

    monkeypatch.setattr(fccm, "_build_preregistration", fake_verified_build)
    monkeypatch.setattr(fccm, "_validate_preregistration", lambda *_a, **_k: None)

    cfg = fccm.Config(output="out/prereg.json")
    first, status = fccm.write_preregistration(cfg)
    assert status == "created"
    second, status = fccm.write_preregistration(cfg)
    assert status == "verified_existing"
    assert first == second

    output = tmp_path / cfg.output
    original_bytes = output.read_bytes()
    output.write_bytes(b'{"forbidden_outcome":123}\n')
    monkeypatch.setattr(
        fccm.json,
        "loads",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("existing output was parsed")
        ),
    )
    with pytest.raises(RuntimeError, match="existing FCCM preregistration differs"):
        fccm.write_preregistration(cfg)
    assert output.read_bytes() == b'{"forbidden_outcome":123}\n'

    output.write_bytes(original_bytes)
    with pytest.raises(FileExistsError):
        fccm._atomic_write(output, b"replacement\n")
    assert output.read_bytes() == original_bytes
