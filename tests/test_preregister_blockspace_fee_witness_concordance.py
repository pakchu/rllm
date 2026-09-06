from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from training import preregister_blockspace_fee_witness_concordance as p


def test_manifest_is_deterministic_self_hashing_and_closed() -> None:
    first = p.build_manifest()
    second = p.build_manifest()
    assert first == second
    core = {
        key: value for key, value in first.items() if key != "manifest_hash"
    }
    assert first["manifest_hash"] == p.canonical_hash(core)
    p.validate_manifest(first)
    assert first["source_rows_opened"] is False
    assert first["source_incidence_opened"] is False
    assert first["candidate_overlap_opened"] is False
    assert first["economic_rows_opened"] is False
    assert first["outcomes_opened"] is False
    disclosure = first["evidence_disclosure"]
    assert disclosure["csv_data_rows_decoded_by_preregistration"] == 0
    assert disclosure["source_values_previously_seen"] is True
    assert disclosure["BFRT_individual_incidence_previously_seen"] is True
    assert disclosure["WCTR_individual_incidence_previously_seen"] is True
    assert disclosure["BCRT_broad_relational_family_previously_seen"] is True


def test_frozen_dependencies_and_exact_headers_validate() -> None:
    p.validate_frozen_dependencies()
    assert set(p.EXACT_CSV_HEADERS).issubset(p.FROZEN_DEPENDENCIES)
    for path, expected in p.EXACT_CSV_HEADERS.items():
        header = p.csv_header_bytes(path)
        assert header == expected.encode("utf-8")
        assert p.sha256_csv_header(path) == hashlib.sha256(header).hexdigest()
    assert p.sha256_file(p.DOCUMENT_PATH) == p.FROZEN_DEPENDENCIES[
        str(p.DOCUMENT_PATH)
    ]


def test_source_manifest_internal_hashes_are_bound() -> None:
    payload = p.build_manifest()["source_contracts"]
    assert payload["BFRT"]["source_manifest"]["internal_manifest_hash"] == (
        p.BFRT_SOURCE_MANIFEST_HASH
    )
    assert payload["WCTR"]["source_manifest"]["internal_manifest_hash"] == (
        p.WCTR_SOURCE_MANIFEST_HASH
    )
    assert payload["BFRT"]["normalized"]["header"][:3] == [
        "bucket_start_utc",
        "bucket_end_utc",
        "available_at_utc",
    ]
    assert payload["WCTR"]["normalized"]["header"][-2:] == [
        "avg_size",
        "avg_weight",
    ]


def test_exact_feature_rank_and_fixed_polarity_contract() -> None:
    payload = p.build_manifest()
    feature = payload["join_and_feature"]
    assert feature["x"] == "x[p,t]=log1p(fee_p[t]), p={10,25,75,90}"
    assert feature["delta2"] == "Delta2 z[t]=z[t]-z[t-2]"
    assert feature["R"] == (
        "0.5*((Delta2 x[10]+Delta2 x[25])-"
        "(Delta2 x[75]+Delta2 x[90]))"
    )
    assert feature["witness_share"] == (
        "(4*avg_size-avg_weight)/(3*avg_size)"
    )
    assert feature["W"] == "Delta2 witness_share"
    assert feature["U"] == "Delta2 fullness"
    rank = payload["rank_and_signal"]
    assert rank["minimum_prior_rows"] == 120
    assert rank["midrank"] == "(L+0.5*E)/n"
    assert rank["side"] == {
        "positive_R": "LONG",
        "negative_R": "SHORT",
    }
    assert "flattening/broadening" in rank["economic_interpretation"]
    assert rank["grids"] == {
        "threshold": False,
        "sign": False,
        "horizon": False,
        "latency": False,
        "leverage": False,
    }


def test_midrank_excludes_current_and_preserves_exact_ties() -> None:
    assert p.exact_midrank(2.0, [1.0, 2.0, 2.0, 3.0]) == 0.5
    assert p.exact_midrank(4.0, [1.0, 2.0, 3.0]) == 1.0
    assert p.exact_midrank(1.0, [1.0, 1.0]) == 0.5
    with pytest.raises(ValueError, match="prior"):
        p.exact_midrank(1.0, [])
    with pytest.raises(ValueError, match="finite"):
        p.exact_midrank(float("nan"), [1.0])


def test_entry_clock_always_waits_one_complete_bar() -> None:
    assert p.ceil_5m_plus_one_bar(0) == 300
    assert p.ceil_5m_plus_one_bar(300) == 600
    assert p.ceil_5m_plus_one_bar(301) == 900
    assert p.ceil_5m_plus_one_bar(599) == 900
    with pytest.raises(TypeError, match="integer"):
        p.ceil_5m_plus_one_bar(True)


def test_execution_calendar_and_support_are_exact() -> None:
    payload = p.build_manifest()
    execution = payload["execution"]
    assert execution["aligned_availability_still_waits_seconds"] == 300
    assert execution["hold_bars_5m"] == 288
    assert execution["hold_seconds"] == 86_400
    assert execution["funding_interval"] == (
        "entry_time <= funding_time < exit_time"
    )
    assert execution["reservation"]["scope"] == "global chronological"
    calendar = payload["calendar"]
    assert calendar["full"] == [
        "2023-06-01T00:00:00Z",
        "2026-06-01T00:00:00Z",
    ]
    assert calendar["full_wall_clock_years"] == 3
    support = payload["support_gates"]
    assert support["selection"] == {
        "total_min": 45,
        "2023_NovDec_min": 6,
        "2024_H1_min": 12,
        "2024_H2_min": 12,
        "each_side_min": 14,
        "maximum_month_share": 0.20,
    }
    assert support["future_2025"]["total_min"] == 30
    assert support["future_2026"]["total_min"] == 15
    assert support["exact_join_gaps"] == 0
    assert support["future_append_invariance"]["required"] is True


def test_controls_preserve_parent_or_build_own_clock_as_frozen() -> None:
    controls = p.build_manifest()["controls"]
    assert controls["component_own_nonoverlap"] == [
        "fee_rotation_only",
        "witness_fullness_only",
        "drop_witness",
        "drop_fullness",
        "one_bucket_stale_witness_fullness",
    ]
    assert controls["same_parent_set"] == [
        "exact_direction_flip",
        "deterministic_random_side",
        "constant_long",
        "constant_short",
        "one_bar_delayed_entry",
    ]
    assert controls["component_only"] == [
        "fee_rotation_only",
        "witness_fullness_only",
    ]
    assert controls["controls_cannot_replace_primary"] is True
    assert p.deterministic_random_side("example") in {"LONG", "SHORT"}
    assert p.deterministic_random_side("example") == (
        p.deterministic_random_side("example")
    )


def test_novelty_binds_prior_clocks_and_each_gross9_sleeve() -> None:
    novelty = p.build_manifest()["novelty"]
    prior = {item["id"]: item for item in novelty["prior_primary_clocks"]}
    assert set(prior) == {"BFRT-288", "WCTR-288"}
    for comparator in prior.values():
        assert comparator["thresholds"] == {
            "exact_entry_jaccard_max": 0.20,
            "candidate_6h_containment_max": 0.50,
            "absolute_signed_exposure_pearson_max": 0.40,
        }
    gross9 = novelty["gross9"]
    assert gross9["weights"] == p.GROSS9_WEIGHTS
    assert sum(gross9["weights"].values()) == 9.0
    assert gross9["gross"] == 9.0
    assert gross9["compare_each_sleeve_separately"] is True
    assert gross9["thresholds_each_sleeve"] == {
        "exact_entry_jaccard_max": 0.10,
        "candidate_6h_containment_max": 0.35,
        "occupied_bar_jaccard_max": 0.25,
        "absolute_signed_exposure_pearson_max": 0.35,
    }
    assert gross9["all_five_sleeves_must_pass"] is True


def test_economic_and_same_gross_gates_are_two_scenario_and_future_veto() -> None:
    payload = p.build_manifest()
    complete = payload["economic_gates"]["complete_gate_each_period"]
    assert complete["standalone_base_and_stress_absolute_return"] == ">0"
    assert complete["base_and_stress_full_calendar_cagr_to_strict_mdd"] == (
        ">=3.0"
    )
    assert complete["base_and_stress_strict_mdd"] == "<=0.15"
    assert complete["mean_gross_underlying_bp"] == ">=20"
    assert complete["weekly_cluster_one_sided_signflip_p"] == "<=0.10"
    marginal = payload["same_gross_marginal"]
    assert marginal["candidate_weights"] == [0.25, 0.50, 0.75, 1.00]
    assert marginal["baseline_scale"] == (
        "(9-w)/9 applied pro rata to every Gross9 sleeve"
    )
    assert marginal["selection_requirements"] == {
        "base_and_stress_cagr_mdd_absolute_ratio_improvement_min_each": 0.05,
        "base_and_stress_strict_mdd_nonworse": True,
        "base_and_stress_absolute_return_positive": True,
        "base_and_stress_baseline_return_retention_min": 0.97,
    }
    assert marginal["top_n"] == 1
    assert marginal["future_can_rerank_repair_or_select_rank2"] is False
    assert marginal["each_future_period_must_pass"] is True
    assert marginal["stitched_exact_3y_report_required"] is True


def test_strict_sequence_requires_separately_committed_evaluator() -> None:
    payload = p.build_manifest()
    assert payload["strict_sequence"][:3] == [
        "preregistration_document_commit",
        "write_once_preregistration_commit",
        "separately_committed_evaluator_bound_to_file_and_manifest_hash",
    ]
    assert payload["sequence_rules"] == {
        "stop_at_first_failure": True,
        "no_repair_grid_fallback_or_polarity_flip": True,
        "future_cannot_rerank_or_repair": True,
        "control_cannot_replace_primary": True,
        "exact_rule_retired_on_any_failure": True,
    }


def test_write_once_artifact_matches_code_exactly() -> None:
    artifact = p.REPOSITORY_ROOT / p.DEFAULT_OUTPUT
    assert artifact.is_file()
    assert not artifact.is_symlink()
    assert artifact.read_bytes() == p.canonical_manifest_bytes(
        p.build_manifest()
    )
    assert p.write_once() == "verified_existing"


@pytest.mark.parametrize(
    "path",
    [
        "/tmp/out.json",
        "../results/out.json",
        "results/../out.json",
        "~/out.json",
    ],
)
def test_write_once_rejects_non_repository_relative_output(path: str) -> None:
    with pytest.raises(RuntimeError, match="repository-relative"):
        p._output_relative(path)


def test_owned_artifact_path_is_exact() -> None:
    assert p.DEFAULT_OUTPUT == Path(
        "results/blockspace_fee_witness_concordance_"
        "preregistration_2026-07-30.json"
    )
