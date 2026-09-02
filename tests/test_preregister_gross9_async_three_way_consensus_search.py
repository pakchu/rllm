from __future__ import annotations

import copy
from itertools import combinations

import pytest

from training import preregister_gross9_async_three_way_consensus_search as p


def _rows() -> list[dict[str, object]]:
    return [
        {
            "candidate": candidate,
            "source_pass": True,
            "gross9_pass": True,
            "train_cagr_to_strict_mdd": float(index + 1),
            "train_absolute_return": 0.01 * (index + 1),
            "train_economic_pass": True,
        }
        for index, candidate in enumerate(p.CANDIDATE_FAMILY)
    ]


def test_build_fixes_exact_9_component_all_84_three_way_family_and_clock() -> None:
    value = p.build()
    p.validate(value)
    assert value["policy_id"] == "G9ASYNC3WAY-8"
    assert value["component_order"] == list(p.COMPONENT_ORDER)
    assert value["component_count"] == 9
    assert value["candidate_family_size"] == 84
    assert len(value["candidate_family"]) == 84
    assert value["candidate_family"] == [
        f"{first}__ASYNC_SAME_SIDE_3WAY_6H__{second}__{third}"
        for first, second, third in combinations(p.COMPONENT_ORDER, 3)
    ]
    construction = value["construction"]
    assert "[t-6h,t]" in construction["entry_rule"]
    assert "at least one selected event exactly at t" in construction["entry_rule"]
    assert "opposite-side events are ignored" in construction["entry_rule"]
    assert "simultaneous two-of-three" in construction["trigger_canonicalization"]
    assert "earliest component in frozen component_order" in construction["trigger_canonicalization"]
    assert "drop both" in construction["dual_side_same_timestamp_policy"]
    assert "must be <= t" in construction["availability"]
    assert "half-open 8h reservation inside each candidate-triple" in construction["reservation"]
    assert value["clock"]["hold"] == "8 elapsed hours"
    assert value["clock"]["gross_exposure"] == 0.5


def test_build_binds_same_components_gross9_two_terminal_predecessors_and_builder() -> None:
    value = p.build()
    assert set(value["implementation"]) == {"preregister", "train_clock_builder"}
    assert p.sha256_file(value["implementation"]["preregister"]["path"]) == value["implementation"]["preregister"]["sha256"]
    builder = value["implementation"]["train_clock_builder"]
    assert builder["path"] == "training/build_gross9_async_three_way_consensus_train_clocks.py"
    assert p.sha256_file(builder["path"]) == builder["sha256"]
    assert set(value["component_artifacts"]) == set(p.COMPONENT_ORDER)
    assert value["gross9_pre2025_clock_manifest"] == p.GROSS9_PRE2025_CLOCK_MANIFEST
    predecessors = value["predecessor_family_terminal_results"]
    assert predecessors == [p.SAME_SIDE_TERMINAL_TRAIN_ECONOMICS, p.OPPOSITION_HANDOFF_TERMINAL_TRAIN_ECONOMICS]
    assert predecessors[0]["sha256"] == "0b822d77415ca70a409d2e7f3c35ebe44cbf481aa7e0d2eb02605646bdb3f874"
    assert predecessors[0]["manifest_hash"] == "bb3ed8afa1eec6cddf2344515d89736a36314157ad7eeac495c759adadc45b16"
    assert predecessors[1]["sha256"] == "a2c86ae78940a331f1e0209fa5bbb8bdb374fd2d4438030900ffd3a097b85e64"
    assert predecessors[1]["manifest_hash"] == "3ad6368a44519359cf7661b7707a9f099ae23fb679c6c2115aafab095a51aa3a"
    prior_support = value["prior_clock_source_support_artifacts"]
    assert prior_support == p.PRIOR_CLOCK_SOURCE_SUPPORT_ARTIFACTS
    assert prior_support[0]["sha256"] == "c6d3929f282ba1075c2ebc091e4bc62164b923a038bce94de32884aaf7ff0009"
    assert prior_support[0]["manifest_hash"] == "b92d3afb7a3539cdd194eddc1ab09bc65068716135d0bca575db0531ac450011"
    assert prior_support[1]["sha256"] == "a8982c1b6e155f65f76af4559ca2d01b2a7824cb5c58524a260b72beb997f754"
    assert prior_support[1]["manifest_hash"] == "92501aa4c921bba20d05378b6f658f33d6c712e8b3adb9f095940dd44ac3f3b0"


def test_source_gross9_bonferroni_and_research_boundaries_are_frozen() -> None:
    value = p.build()
    source = value["source_support_gates"]
    assert source["minimum_events"]["train"] == 10
    assert source["minority_side_share_min"] == 0.20
    assert source["max_month_share"] == 0.45
    assert source["distinct_iso_weeks_min"] == 10
    assert source["each_calendar_half_min_events"] == 1
    assert "post-reservation" in source["prior_family_exact_duplicate_gate"]
    assert "constituent same-side pre-reservation" in source["prior_family_overlap_disclosure_scope"]
    assert "all prior-72" in source["prior_family_overlap_disclosure_scope"]
    assert source["prior_family_overlap_disclosure_required"] is True
    assert "constituent same-side pre-reservation" in value["construction"]["overlap_disclosure"]
    assert "all prior-72" in value["construction"]["overlap_disclosure"]
    assert "only exact post-reservation schedule duplication" in value["construction"]["overlap_disclosure"]
    assert source["prior_family_overlap_non_exact_is_gate"] is False
    assert value["gross9_novelty_gates"]["exact_entry_jaccard_max"] == 0.10
    assert value["gross9_novelty_gates"]["candidate_near_6h_share_max"] == 0.35
    assert value["gross9_novelty_gates"]["occupied_5m_jaccard_max"] == 0.25
    assert value["gross9_novelty_gates"]["absolute_signed_exposure_pearson_max"] == 0.35
    assert value["economic_gates"]["weekly_signflip_one_sided_p_max"] == pytest.approx(0.10 / 84)
    assert value["familywise_multiplicity"]["number_of_hypotheses"] == 84
    assert "cumulative adaptive exploratory research program" in value["familywise_multiplicity"]["scope_boundary"]
    boundary = value["research_boundary"]
    assert boundary["predecessor_economic_scalars_used_to_tune_operator_or_gates"] is False
    assert boundary["three_way_combination_incidence_opened_by_preregistration"] is False
    assert boundary["three_way_combination_outcomes_opened_by_preregistration"] is False
    assert boundary["market_or_funding_rows_opened_by_preregistration"] is False


def test_select_train_winner_requires_all_84_rows_and_no_substitution() -> None:
    rows = _rows()
    rows[-1]["source_pass"] = False
    rows[-2]["gross9_pass"] = False
    rows[-3]["train_cagr_to_strict_mdd"] = 99.0
    rows[-3]["train_absolute_return"] = 0.30
    rows[-4]["train_cagr_to_strict_mdd"] = 99.0
    rows[-4]["train_absolute_return"] = 0.30
    winner = p.select_train_winner(rows)
    assert winner["candidate"] == p.CANDIDATE_FAMILY[-4]
    assert winner["frozen_before_test"] is True
    assert winner["substitution_authorized"] is False
    assert winner["rerank_authorized"] is False

    with pytest.raises(ValueError, match="exactly 84 candidate rows"):
        p.select_train_winner(rows[:-1])

    duplicate = _rows()
    duplicate[-1]["candidate"] = duplicate[0]["candidate"]
    with pytest.raises(ValueError, match="candidate IDs must be unique strings"):
        p.select_train_winner(duplicate)

    missing = _rows()
    missing[-1]["candidate"] = "NOT_A_FROZEN_TRIPLE"
    with pytest.raises(ValueError, match="exact frozen family"):
        p.select_train_winner(missing)


def test_raw_rank_one_failure_is_terminal_without_substitution() -> None:
    rows = _rows()
    rows[-1]["train_economic_pass"] = False
    with pytest.raises(RuntimeError, match="raw rank one failed train; no substitution"):
        p.select_train_winner(rows)


def test_selector_rejects_non_boolean_flags_and_nonfinite_metrics() -> None:
    rows = _rows()
    rows[-1]["source_pass"] = "yes"
    with pytest.raises(ValueError, match="source_pass must be boolean"):
        p.select_train_winner(rows)

    rows = _rows()
    rows[-1]["gross9_pass"] = 1
    with pytest.raises(ValueError, match="gross9_pass must be boolean"):
        p.select_train_winner(rows)

    rows = _rows()
    rows[-1]["train_economic_pass"] = None
    with pytest.raises(ValueError, match="train_economic_pass must be boolean"):
        p.select_train_winner(rows)

    for bad in (True, float("nan"), float("inf"), float("-inf")):
        rows = _rows()
        rows[-1]["train_cagr_to_strict_mdd"] = bad
        with pytest.raises(ValueError, match="ranking metrics"):
            p.select_train_winner(rows)


def test_validate_detects_manifest_family_predecessor_source_and_boundary_drift() -> None:
    value = p.build()
    drifted = copy.deepcopy(value)
    drifted["clock"]["hold"] = "9 elapsed hours"
    with pytest.raises(RuntimeError, match="preregistration drift"):
        p.validate(drifted)

    drifted = p.build()
    drifted["candidate_family"] = drifted["candidate_family"][:-1]
    core = dict(drifted)
    core.pop("manifest_hash")
    drifted["manifest_hash"] = p.canonical_hash(core)
    with pytest.raises(RuntimeError, match="candidate family drift"):
        p.validate(drifted)

    drifted = p.build()
    drifted["predecessor_family_terminal_results"][1]["manifest_hash"] = "0" * 64
    core = dict(drifted)
    core.pop("manifest_hash")
    drifted["manifest_hash"] = p.canonical_hash(core)
    with pytest.raises(RuntimeError, match="predecessor terminal binding drift"):
        p.validate(drifted)

    drifted = p.build()
    drifted["prior_clock_source_support_artifacts"][0]["manifest_hash"] = "0" * 64
    core = dict(drifted)
    core.pop("manifest_hash")
    drifted["manifest_hash"] = p.canonical_hash(core)
    with pytest.raises(RuntimeError, match="prior source-support binding drift"):
        p.validate(drifted)

    drifted = p.build()
    drifted["construction"]["overlap_disclosure"] = "prior post-reservation only"
    core = dict(drifted)
    core.pop("manifest_hash")
    drifted["manifest_hash"] = p.canonical_hash(core)
    with pytest.raises(RuntimeError, match="overlap disclosure scope drift"):
        p.validate(drifted)

    drifted = p.build()
    drifted["source_support_gates"]["distinct_iso_weeks_min"] = 9
    core = dict(drifted)
    core.pop("manifest_hash")
    drifted["manifest_hash"] = p.canonical_hash(core)
    with pytest.raises(RuntimeError, match="distinct-week gate drift"):
        p.validate(drifted)

    drifted = p.build()
    drifted["research_boundary"]["predecessor_economic_scalars_used_to_tune_operator_or_gates"] = True
    core = dict(drifted)
    core.pop("manifest_hash")
    drifted["manifest_hash"] = p.canonical_hash(core)
    with pytest.raises(RuntimeError, match="predecessor scalar tuning boundary drift"):
        p.validate(drifted)


def test_embedded_hashes_match_current_bound_files() -> None:
    root = p.Path(__file__).resolve().parents[1]
    value = p.build()
    assert p.sha256_file(root / value["gross9_pre2025_clock_manifest"]["path"]) == value["gross9_pre2025_clock_manifest"]["sha256"]
    assert p.sha256_file(root / value["implementation"]["train_clock_builder"]["path"]) == value["implementation"]["train_clock_builder"]["sha256"]
    for predecessor in value["predecessor_family_terminal_results"]:
        assert p.sha256_file(root / predecessor["path"]) == predecessor["sha256"]
    for source_support in value["prior_clock_source_support_artifacts"]:
        assert p.sha256_file(root / source_support["path"]) == source_support["sha256"]
    for artifacts in value["component_artifacts"].values():
        for binding in artifacts.values():
            assert p.sha256_file(root / binding["path"]) == binding["sha256"]
