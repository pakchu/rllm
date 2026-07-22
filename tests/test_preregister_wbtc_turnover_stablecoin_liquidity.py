from __future__ import annotations

import json
from pathlib import Path

import pytest

from training import preregister_wbtc_turnover_stablecoin_liquidity as prereg


def test_preregistration_discloses_source_seen_but_outcome_blind_status() -> None:
    payload = prereg.build_preregistration()
    prereg.validate_preregistration(payload)
    assert payload["candidate"] == "WTSL-168-SOURCE-SEEN"
    assert payload["policy"]["research_status"] == "source-seen_outcome-blind"
    assert payload["policy"]["source_family_hypotheses"] == 2
    assert set(payload["source_bindings"]) == {"wbtc", "stablecoin"}
    assert payload["source_incidence_opened"] is True
    assert payload["source_incidence_disclosure"] == prereg.PRIOR_SOURCE_DISCLOSURE
    assert payload["outcomes_opened"] is False
    assert payload["performance_values_opened"] is False
    assert payload["outcome_boundary"] == prereg.EXPECTED_OUTCOME_BOUNDARY
    assert all(
        source["value_rows_read_during_preregistration"] == 0
        for source in payload["source_bindings"].values()
    )


def test_policy_separates_wbtc_gate_from_stablecoin_direction() -> None:
    policy = prereg.policy_payload()
    hypothesis = policy["economic_hypothesis"]
    state = policy["causal_state"]
    assert hypothesis["directional_alpha"] == (
        "combined stablecoin net issuance-redemption"
    )
    assert hypothesis["wbtc_net_sign_used_by_primary"] is False
    assert state["decision_grid_utc_hours"] == [0, 6, 12, 18]
    assert state["source_cutoff"] == "decision_time - 6 elapsed hours"
    assert state["only_clock_field"] == "available_at"
    assert state["current_window"] == {
        "elapsed_hours": 168,
        "interval": "cutoff-168h < available_at <= cutoff",
    }
    baseline = state["wbtc_activity_baseline"]
    assert baseline["prior_endpoints"] == 1460
    assert baseline["strictly_prior_first_endpoint"] == "cutoff - 6h"
    assert baseline["strictly_prior_last_endpoint"] == "cutoff - 365d"
    assert baseline["zeros_included"] is True
    assert state["stablecoin_validity"]["destroyed_black_funds_in_window"] == (
        "veto"
    )
    assert state["side"] == {
        "long": "stablecoin_net_raw > 0",
        "short": "stablecoin_net_raw < 0",
        "otherwise": "no candidate",
    }


def test_execution_controls_and_staged_gates_are_fixed() -> None:
    policy = prereg.policy_payload()
    assert policy["execution"] == {
        "decision_time": "four six-hour UTC anchors per day",
        "entry_delay_minutes": 10,
        "hold_bars_5m": 288,
        "hold_elapsed_hours": 24,
        "notional_exposure": 0.5,
        "global_nonoverlap": True,
        "accept_when_entry_at_or_after_prior_exit": True,
        "split_crossing_action": "skip",
        "stops_take_profit_or_trailing_exit": False,
    }
    assert set(policy["controls"]) == {
        "direction_flip",
        "stablecoin_only_direct",
        "wbtc_signed_placebo",
        "stale_24h",
        "stale_48h",
        "actor_cap_60",
        "no_black_funds_veto",
        "usdc_only_direct",
        "usdt_only_direct",
        "year_amount_permutation",
        "deterministic_random_side",
    }
    support = policy["source_support_gates"]
    assert support["maximum_month_share"] == 0.20
    assert support["maximum_quarter_share"] == 0.40
    assert support["maximum_consecutive_same_side"] == 20
    assert policy["windows"]["sealed_from"] == "2024-01-01T00:00:00Z"
    strict = policy["strict_economic_gates"]
    assert strict["cagr_to_strict_mdd_minimum"] == 3.0
    assert strict["strict_mdd_pct_maximum"] == 15.0
    assert strict["primary_cagr_mdd_above_stablecoin_only"] is True
    assert policy["rllm_boundary"][
        "authorized_before_deterministic_train_and_selection_pass"
    ] is False


def test_tampering_fails_canonical_validation() -> None:
    payload = prereg.build_preregistration()
    payload["policy"]["execution"]["hold_bars_5m"] = 576
    with pytest.raises(RuntimeError, match="canonical hash mismatch"):
        prereg.validate_preregistration(payload, verify_sources=False)


def test_false_clean_preregistration_claim_is_rejected() -> None:
    payload = prereg.build_preregistration()
    payload["source_incidence_opened"] = False
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    payload["manifest_hash"] = prereg.canonical_hash(core)
    with pytest.raises(RuntimeError, match="source-seen disclosure drift"):
        prereg.validate_preregistration(payload, verify_sources=False)


def test_write_once_verifies_identical_and_refuses_drift(tmp_path: Path) -> None:
    relative = Path(".pytest_cache") / f"{tmp_path.name}-wtsl-preregistration.json"
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
