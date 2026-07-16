from __future__ import annotations

import json

import pytest

from training import preregister_tail_arrival_absorption_alpha as taar


def test_policy_grid_is_fixed_small_and_unique() -> None:
    policies = taar.policy_grid()
    assert len(policies) == 4
    assert policies == sorted(policies)
    assert len(set(policies)) == 4
    assert [policy.policy_id for policy in policies] == ["T01", "T02", "T03", "T04"]
    assert {policy.branch for policy in policies} == {
        "tail_absorption_fade",
        "tail_release_follow",
    }
    assert {policy.hold_bars for policy in policies} == {12, 36}


def test_manifest_seals_2023_and_uses_live_feasible_latency() -> None:
    manifest = taar.build_manifest()
    taar.validate_manifest(manifest)
    assert manifest["outcomes_opened"] is False
    assert manifest["selection_protocol"]["sealed_holdout"] == [
        "2023-01-01",
        "2024-01-01",
    ]
    assert manifest["execution_contract"]["entry_delay_bars"] == 2
    assert "five-minute latency" in manifest["execution_contract"]["entry"]


def test_novel_axis_excludes_existing_live_portfolio_inputs() -> None:
    manifest = taar.build_manifest()
    distinct = manifest["novelty_check"]["distinct_axis"]
    assert "no HHI" in distinct
    assert "OI" in distinct and "funding" in distinct and "REX" in distinct
    features = manifest["feature_contract"]
    assert features["tail_span"].startswith("log(event_notional_p99")
    assert features["arrival_cv"].startswith("log1p(interarrival_std_ms")


def test_support_and_selection_are_fail_closed() -> None:
    manifest = taar.build_manifest()
    support = manifest["support_freeze_before_returns"]
    assert support["nonoverlap_events_min_each_policy"] == 120
    assert support["failure_action"].endswith("without computing forward trade returns")
    selection = manifest["selection_protocol"]
    assert selection["multiple_testing_hypotheses"] == 4
    assert selection["holdout_2023_gates"]["cagr_to_strict_mdd_min"] == 3.0


def test_preregistration_is_append_only(tmp_path) -> None:
    path = tmp_path / "prereg.json"
    payload = taar.build_manifest()
    assert taar.write_manifest_once(path, payload) == "created"
    assert taar.write_manifest_once(path, taar.build_manifest()) == "verified_existing"
    changed = json.loads(path.read_text())
    changed["manifest_hash"] = "bad"
    path.write_text(json.dumps(changed))
    with pytest.raises(RuntimeError, match="hash mismatch"):
        taar.write_manifest_once(path, payload)
