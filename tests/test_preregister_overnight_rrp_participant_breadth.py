from __future__ import annotations

import copy
import hashlib

import pytest

from training import preregister_overnight_rrp_participant_breadth as orpb


def test_preregistration_freezes_one_exact_orpb_candidate() -> None:
    report = orpb.build_registration()

    assert report["policy_id"] == "ORPB-21"
    assert report["authorization"]["candidate_count"] == 1
    assert report["feature"]["lookback_complete_operations"] == 21
    assert report["feature"]["lower_tail"] == 0.10
    assert report["feature"]["upper_tail"] == 0.90
    assert report["feature"]["direction"] == {
        "lower": "LONG",
        "upper": "SHORT",
    }
    assert report["feature"]["parameter_grid"] is False


def test_support_short_circuits_before_comparator_access() -> None:
    report = orpb.build_registration()

    assert report["support"]["train_gates"] == {
        "events_min": 50,
        "events_max": 130,
        "each_year_events_min": 20,
        "each_side_share_min": 0.25,
        "maximum_month_share": 0.20,
    }
    assert report["support"]["selection_gates"] == {
        "events_min": 25,
        "events_max": 80,
        "each_half_events_min": 8,
        "each_side_share_min": 0.20,
        "maximum_month_share": 0.25,
    }
    assert report["support"]["short_circuit_before_comparator_access_on_failure"]
    assert report["support"]["repair_authorized"] is False
    assert report["support"]["integrity_gates"] == {
        "source_binding_hashes_exact": True,
        "source_schema_exact": True,
        "source_row_and_quarantine_counts_exact": True,
        "quarantined_values_blank": True,
        "quarantine_clears_feature_window": True,
        "prior_only_ols_replay_exact": True,
        "prior_only_rank_replay_exact": True,
        "decision_clock_exact": True,
        "entry_delay_exact": True,
        "next_operation_exit_exact": True,
        "last_source_row_omitted": True,
        "split_containment_exact": True,
        "nonoverlap_exact": True,
    }


def test_novelty_thresholds_and_full_comparison_grid_are_frozen() -> None:
    report = orpb.build_registration()

    assert report["novelty"]["same_source_orfr"] == {
        "exact_entry_jaccard_max": 0.15,
        "one_rrp_operation_bidirectional_containment_max": 0.35,
        "absolute_signed_exposure_correlation_max": 0.35,
        "absolute_residual_amount_innovation_spearman_max": 0.35,
    }
    assert report["novelty"]["other_macro"] == {
        "exact_entry_jaccard_max": 0.10,
        "six_hour_bidirectional_containment_max": 0.25,
        "absolute_signed_exposure_correlation_max": 0.35,
    }
    assert report["novelty"]["comparison_start"] == "2021-01-01T00:00:00Z"
    assert report["novelty"]["comparison_end_exclusive"] == "2024-01-01T00:00:00Z"
    assert report["novelty"]["truncate_to_comparator_observed_prefix"] is False
    assert report["novelty"][
        "fail_closed_on_missing_empty_schema_time_overlap_or_outcome_field"
    ]


def test_novelty_cohort_freezes_every_candidate_and_row_count() -> None:
    bindings = orpb.build_registration()["novelty"]["bindings"]

    assert len(bindings) == 7
    assert sum(binding["expected_rows"] for binding in bindings.values()) == 7104
    assert sum(len(binding["required_groups"]) for binding in bindings.values()) == 45
    for binding in bindings.values():
        assert sum(binding["required_groups"].values()) == binding["expected_rows"]
    assert bindings["orfr_clocks"]["required_groups"] == {
        "one_day_delta_tail": 346,
        "one_release_delay": 327,
        "primary": 328,
    }


def test_preregistration_opens_no_market_or_outcome_data() -> None:
    report = orpb.build_registration()
    boundary = report["evidence_boundary"]

    assert boundary["orpb_residuals_computed"] == 0
    assert boundary["orpb_incidence_or_side_counts_computed"] == 0
    assert boundary["bound_comparator_artifacts"] == 7
    assert boundary["comparator_identifier_rows_projected_for_cohort_freeze"] == 7104
    assert boundary["comparator_entry_exit_or_side_fields_materialized"] == 0
    assert boundary["comparator_overlap_metrics_computed"] == 0
    assert boundary["btc_market_rows_read"] == 0
    assert boundary["funding_rows_read"] == 0
    assert boundary["future_return_rows_read"] == 0
    assert boundary["return_or_pnl_fields_read"] == 0
    assert boundary["post_2023_source_rows_read"] == 0
    assert boundary["economic_outcomes_opened"] is False
    assert report["later_outcome_contract"]["authorized"] is False
    assert report["authorization"]["economic_evaluator_authorized"] is False


def test_every_bound_source_and_comparator_hash_is_verified() -> None:
    report = orpb.build_registration()

    assert orpb.sha256_file(report["document"]["path"]) == report["document"]["sha256"]
    for binding in report["source"]["bindings"].values():
        assert orpb.sha256_file(binding["path"]) == binding["sha256"]
    for binding in report["novelty"]["bindings"].values():
        assert orpb.sha256_file(binding["path"]) == binding["sha256"]
    orpb.validate_registration(report)


def test_verify_binding_fails_closed_on_hash_drift(tmp_path) -> None:
    path = tmp_path / "artifact"
    path.write_bytes(b"frozen")

    with pytest.raises(RuntimeError, match="hash drift"):
        orpb.verify_binding(
            {"path": path, "sha256": hashlib.sha256(b"other").hexdigest()}
        )


def test_validation_fails_closed_on_missing_comparator_candidate() -> None:
    report = copy.deepcopy(orpb.build_registration())
    del report["novelty"]["bindings"]["bdrc"]["required_groups"]["primary"]

    with pytest.raises(RuntimeError, match="novelty cohort drift"):
        orpb.validate_registration(report)
