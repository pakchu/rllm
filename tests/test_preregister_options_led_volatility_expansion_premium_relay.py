from __future__ import annotations

from copy import deepcopy

import pytest

from training import preregister_options_led_volatility_expansion_premium_relay as p


def test_manifest_is_deterministic_singleton_and_outcome_blind() -> None:
    first = p.build_manifest()
    assert first == p.build_manifest()
    p.validate_manifest(first)
    assert first["candidate"] == "OVEPR-24"
    assert first["singleton"] is True
    assert first["outcomes_opened"] is False
    assert first["grid_or_search"] is False
    boundary = first["research_boundary"]
    assert boundary["btc_execution_prices_rows_opened"] == 0
    assert boundary["btc_return_or_pnl_rows_opened"] == 0
    assert boundary["funding_rows_opened"] == 0
    assert boundary["gross9_rows_opened"] == 0


def test_disclosed_incidence_and_conservative_support_gates_are_exact() -> None:
    report = p.build_manifest()
    incidence = report["research_boundary"]["incidence"]
    assert incidence["accepted_events"] == {
        "train_2023H2": 109,
        "test_2024": 128,
        "eval_2025": 66,
        "final_2026H1": 52,
    }
    assert incidence["long_short"]["final_2026H1"] == {"long": 30, "short": 22}
    assert report["support_gates"] == {
        "minimum_events": {"train": 80, "test": 80, "eval": 40, "final": 30},
        "minimum_each_side_share": "3/10",
        "maximum_month_share": {
            "train": "1/4",
            "test": "1/5",
            "eval": "7/20",
            "final": "2/5",
        },
        "operator_policy": {"minimums": ">=", "maximums": "<="},
        "failure_action": "reject before opening economic outcomes",
    }


def test_outcome_blind_overlap_probe_and_novelty_limits_are_frozen() -> None:
    report = p.build_manifest()
    probe = report["research_boundary"]["outcome_blind_comparator_probe"]
    assert probe["maximum_observed_prior_family_one_to_one_6h_share"] == "43/100"
    assert probe["maximum_observed_prior_family_occupied_5m_jaccard"] == "28/100"
    limits = report["novelty"]["requirements_each_comparator_and_each_gross9_sleeve"]
    assert limits["one_to_one_6h_max_matched_share_max"] == "9/20"
    assert limits["occupied_5m_bar_jaccard_max"] == "3/10"


def test_mechanism_execution_controls_and_outcome_gates_are_frozen() -> None:
    report = p.build_manifest()
    assert report["mechanism"]["side"] == (
        "follow premium move: positive=LONG, negative=SHORT"
    )
    assert report["execution"]["hold"] == "24 elapsed hours fixed"
    assert report["execution"]["leverage"] == "1/2"
    assert report["execution"]["base_cost_bp_per_notional_side"] == 6
    assert report["execution"]["stress_cost_bp_per_notional_side"] == 10
    assert set(report["controls"]["independent_own_clock"]) == {
        "no_deribit_lead",
        "deribit_fall_mirror",
        "no_premium_efficiency",
    }
    assert set(report["controls"]["same_reserved_primary_events"]) == {
        "direction_flip",
        "extra_latency_1h",
        "deterministic_random_side",
    }
    gate = report["outcome_gate"]
    assert "stop_on_first_failure" in gate["sequential_opening"]
    assert gate["requirements_every_opened_split"] == {
        "absolute_return_positive": True,
        "cagr_to_strict_mdd_min": "3",
        "strict_mdd_max_pct": "15",
        "mean_gross_move_bp_min": "20",
        "clustered_signflip_p_max": "1/10",
        "stress_absolute_return_positive": True,
        "stress_cagr_to_strict_mdd_min": "5/2",
        "each_calendar_half_absolute_return_positive": True,
    }


def test_novelty_is_required_against_named_families_and_every_gross9_sleeve() -> None:
    novelty = p.build_manifest()["novelty"]
    assert novelty["must_complete_before_any_outcome"] is True
    assert novelty["comparators"] == ["OPDR", "CVVH", "PSR", "PCBR", "CMSR"]
    assert "every sleeve" in novelty["gross9"]
    assert novelty["all_must_pass"] is True


def test_rehashed_or_unhashed_mutation_is_rejected() -> None:
    changed = deepcopy(p.build_manifest())
    changed["execution"]["hold"] = "48 hours"
    with pytest.raises(ValueError, match="frozen singleton"):
        p.validate_manifest(changed)

    core = {key: value for key, value in changed.items() if key != "manifest_hash"}
    changed["manifest_hash"] = p.canonical_hash(core)
    with pytest.raises(ValueError, match="frozen singleton"):
        p.validate_manifest(changed)
