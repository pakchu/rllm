from __future__ import annotations

import hashlib

import pytest

from training import preregister_refined_product_divergence_shock as rpds


def test_preregistration_freezes_the_exact_mixed_sign_topology() -> None:
    report = rpds.build_preregistration()
    assert report["policy_id"] == "RPDS-576"
    assert report["signal"] == {
        "crude_sign": "sign(commercial_crude_change_mmbbl)",
        "gasoline_sign": "sign(gasoline_change_mmbbl)",
        "distillate_sign": "sign(distillate_change_mmbbl)",
        "predicate": "gasoline == distillate != 0 and crude == -gasoline",
        "side": "gasoline/distillate sign: build LONG, draw SHORT",
        "thresholds": [],
        "parameter_grid": [],
    }
    assert report["execution"]["hold_five_minute_bars"] == 576
    assert report["execution"]["entry_delay_minutes"] == 5


def test_preregistration_discloses_schema_rows_and_opens_no_outcomes() -> None:
    report = rpds.build_preregistration()
    boundary = report["outcome_boundary"]
    assert boundary["prefreeze_source_value_rows_read_for_schema"] == 1
    assert boundary["prefreeze_comparator_clock_rows_read_for_schema"] == 10
    assert boundary["rpds_predicate_evaluations"] == 0
    assert boundary["candidate_clock_rows_created"] == 0
    assert boundary["comparator_overlap_metrics_computed"] == 0
    assert boundary["btc_market_rows_read"] == 0
    assert boundary["funding_rows_read"] == 0
    assert boundary["future_return_rows_read"] == 0
    assert boundary["return_or_pnl_fields_read"] == 0
    assert boundary["post_2023_source_rows_read"] == 0
    assert report["later_outcome_contract"]["authorized"] is False
    assert report["authorization"]["post_2023_source_access"] is False
    assert report["authorization"]["threshold_or_hold_repair"] is False


def test_support_and_novelty_gates_are_conjunctively_frozen() -> None:
    report = rpds.build_preregistration()
    assert report["support_gates"]["train"] == {
        "events_min": 24,
        "events_max": 75,
        "events_per_year_min": 5,
        "side_share_min": 0.25,
        "month_share_max": 0.25,
    }
    assert report["support_gates"]["selection"] == {
        "events_min": 8,
        "events_max": 24,
        "events_per_half_min": 3,
        "both_sides_required": True,
        "month_share_max": 0.25,
    }
    assert report["novelty"]["exact_entry_jaccard_max"] == 0.10
    assert report["novelty"]["maximum_bidirectional_containment_max"] == 0.25
    assert report["novelty"]["absolute_signed_exposure_correlation_max"] == 0.35
    assert report["novelty"]["comparison_start"] == "2020-01-01T00:00:00Z"
    assert report["novelty"]["comparison_end_exclusive"] == "2024-01-01T00:00:00Z"
    assert report["novelty"]["truncate_to_observed_comparator_events"] is False
    assert report["novelty"]["epsb_primary_exact_release_overlap_required"] == 0


def test_every_bound_source_and_comparator_hash_is_verified() -> None:
    report = rpds.build_preregistration()
    for group in ("source_bindings",):
        for binding in report[group].values():
            assert rpds.sha256_file(binding["path"]) == binding["sha256"]
    for binding in report["novelty"]["bindings"].values():
        assert rpds.sha256_file(binding["path"]) == binding["sha256"]


def test_verify_binding_fails_closed_on_hash_drift(tmp_path) -> None:
    path = tmp_path / "artifact"
    path.write_bytes(b"frozen")
    with pytest.raises(RuntimeError, match="hash drift"):
        rpds.verify_binding(
            {"path": path, "sha256": hashlib.sha256(b"other").hexdigest()}
        )
