from __future__ import annotations

import json
from pathlib import Path

import pytest

from training import preregister_wrapped_collateral_dollar_liquidity_rotation as prereg


def test_preregistration_binds_two_sources_without_incidence_or_outcomes() -> None:
    payload = prereg.build_preregistration()
    prereg.validate_preregistration(payload)
    assert payload["candidate"] == "WCDR-2016"
    assert payload["policy"]["singleton"] is True
    assert payload["policy"]["multiple_testing_hypotheses"] == 1
    assert set(payload["source_bindings"]) == {"wbtc", "usdc"}
    assert payload["source_incidence_opened"] is False
    assert payload["outcomes_opened"] is False
    assert payload["outcome_boundary"] == prereg.EXPECTED_OUTCOME_BOUNDARY
    assert all(
        source["value_rows_read_during_preregistration"] == 0
        for source in payload["source_bindings"].values()
    )


def test_policy_freezes_cross_domain_opposite_sign_direction() -> None:
    policy = prereg.policy_payload()
    state = policy["causal_state"]
    assert state["only_clock_field"] == "available_at"
    assert state["block_timestamp_forbidden"] is True
    assert state["windows"] == {
        "wbtc_calendar_days": 30,
        "usdc_calendar_days": 7,
        "interval": "cutoff-lookback < available_at <= cutoff",
    }
    assert state["side"] == {
        "long": "wbtc_net_raw < 0 and usdc_net_raw > 0",
        "short": "wbtc_net_raw > 0 and usdc_net_raw < 0",
        "otherwise": "no candidate",
    }
    assert (
        policy["economic_hypothesis"]["wbtc_mint_long_or_burn_short_claimed"]
        is False
    )


def test_execution_and_sealed_sequence_are_fixed() -> None:
    policy = prereg.policy_payload()
    assert policy["execution"] == {
        "decision_time": "daily UTC anchor",
        "entry_delay_minutes": 5,
        "hold_bars_5m": 2016,
        "hold_elapsed_days": 7,
        "notional_exposure": 0.5,
        "global_nonoverlap": True,
        "accept_when_entry_at_or_after_prior_exit": True,
        "split_crossing_action": "skip",
        "stops_take_profit_or_trailing_exit": False,
    }
    assert policy["windows"]["sealed_from"] == "2024-01-01T00:00:00Z"
    assert policy["strict_economic_gates"]["cagr_to_strict_mdd_minimum"] == 3.0
    assert policy["strict_economic_gates"]["strict_mdd_pct_maximum"] == 15.0
    assert policy["rllm_boundary"][
        "authorized_before_deterministic_train_and_selection_pass"
    ] is False


def test_tampering_fails_canonical_validation() -> None:
    payload = prereg.build_preregistration()
    payload["policy"]["execution"]["hold_bars_5m"] = 288
    with pytest.raises(RuntimeError, match="canonical hash mismatch"):
        prereg.validate_preregistration(payload, verify_sources=False)


def test_write_once_verifies_identical_and_refuses_drift(tmp_path: Path) -> None:
    relative = Path(".pytest_cache") / f"{tmp_path.name}-wcdr-preregistration.json"
    output = prereg.REPOSITORY_ROOT / relative
    output.unlink(missing_ok=True)
    cfg = prereg.Config(output=str(relative))
    try:
        payload, status = prereg.write_preregistration(cfg)
        assert status == "created"
        second, status = prereg.write_preregistration(cfg)
        assert status == "verified_existing"
        assert second == payload

        stored = json.loads(output.read_text(encoding="utf-8"))
        stored["manifest_hash"] = "0" * 64
        output.write_text(json.dumps(stored), encoding="utf-8")
        with pytest.raises(RuntimeError, match="canonical hash mismatch"):
            prereg.write_preregistration(cfg)
    finally:
        output.unlink(missing_ok=True)


def test_repository_path_rejects_escape() -> None:
    with pytest.raises(RuntimeError, match="repository-relative"):
        prereg._repository_path("../outside.json")
