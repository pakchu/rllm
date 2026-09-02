from __future__ import annotations

import copy
from itertools import combinations

import pytest

from training import preregister_gross9_async_opposition_handoff_search as p


def _rows() -> list[dict[str, object]]:
    return [
        {
            "candidate": candidate,
            "source_pass": True,
            "same_side_disjointness_pass": True,
            "gross9_pass": True,
            "train_cagr_to_strict_mdd": float(index + 1),
            "train_absolute_return": 0.01 * (index + 1),
            "train_economic_pass": True,
        }
        for index, candidate in enumerate(p.CANDIDATE_FAMILY)
    ]


def test_manifest_fixes_exact_9_component_opposition_handoff_family_and_clock() -> None:
    value = p.build()
    p.validate(value)
    assert value["policy_id"] == "G9ASYNCHANDOFF-8"
    assert value["component_order"] == list(p.COMPONENT_ORDER)
    assert value["component_count"] == 9
    assert value["candidate_family_size"] == 36
    assert value["candidate_family"] == [
        f"{left}__ASYNC_OPPOSITION_HANDOFF_6H__{right}"
        for left, right in combinations(p.COMPONENT_ORDER, 2)
    ]
    construction = value["construction"]
    assert "exactly one" in construction["entry_rule"]
    assert "opposite-side" in construction["entry_rule"]
    assert "zero same-side" in construction["entry_rule"]
    assert "[t-6h,t)" in construction["lookback_window"]
    assert "same-timestamp confirmation is forbidden" in construction["lookback_window"]
    assert construction["output_side"] == "newer trigger component side, opposite to the latest confirming component side"
    assert value["clock"]["hold"] == "8 elapsed hours"
    assert value["clock"]["gross_exposure"] == 0.5


def test_manifest_binds_same_components_gross9_and_terminal_same_side_result() -> None:
    value = p.build()
    assert set(value["implementation"]) == {"preregister", "train_clock_builder"}
    assert p.sha256_file(value["implementation"]["preregister"]["path"]) == value["implementation"]["preregister"]["sha256"]
    assert p.sha256_file(value["implementation"]["train_clock_builder"]["path"]) == value["implementation"]["train_clock_builder"]["sha256"]
    assert set(value["component_artifacts"]) == set(p.COMPONENT_ORDER)
    for component, artifacts in value["component_artifacts"].items():
        assert set(artifacts) == {"train_economics", "preregistration", "source_support", "gross9", "clock"}
        for binding in artifacts.values():
            assert len(binding["sha256"]) == 64, component
        assert len(artifacts["train_economics"]["manifest_hash"]) == 64
        assert len(artifacts["gross9"]["manifest_hash"]) == 64
        assert artifacts["clock"]["rows"] > 0
    assert value["gross9_pre2025_clock_manifest"] == p.GROSS9_PRE2025_CLOCK_MANIFEST
    predecessor = value["predecessor_family_terminal_result"]
    assert predecessor == p.SAME_SIDE_TERMINAL_TRAIN_ECONOMICS
    assert predecessor["decision"] == "terminal_train_reject_no_substitution"
    assert predecessor["raw_rank_one_candidate"] == "HVDIMIO-8__ASYNC_SAME_SIDE_6H__HVLVR-8"


def test_boundary_bonferroni_source_feasibility_and_disjointness_are_frozen() -> None:
    value = p.build()
    boundary = value["research_boundary"]
    assert boundary["llm_path_paused"] is True
    assert boundary["design_fixed_before_scratch_source_feasibility_check"] is True
    assert boundary["source_pair_incidence_and_support_counts_opened_before_persistent_preregistration_artifact"] is True
    assert boundary["same_side_terminal_train_outcomes_used_to_choose_or_tune_operator_gates"] is False
    assert boundary["handoff_pair_combination_incidence_opened_before_artifact"] is True
    assert boundary["handoff_pair_combination_incidence_opened_by_this_preregistration_script"] is False
    assert boundary["handoff_pair_combination_outcomes_opened_by_preregistration"] is False
    assert value["source_support_gates"]["distinct_iso_weeks_min"] == 9
    assert "exact zero intersection" in value["construction"]["same_side_family_disjointness_invariant"]
    assert value["selection"]["raw_rank_one_no_substitution"] is True
    assert value["familywise_multiplicity"]["rule"] == "Bonferroni"
    assert "fixed 36-hypothesis family only" in value["familywise_multiplicity"]["scope_boundary"]
    assert "cumulative adaptive research program" in value["familywise_multiplicity"]["scope_boundary"]
    assert value["familywise_multiplicity"]["number_of_hypotheses"] == 36
    assert value["economic_gates"]["weekly_signflip_one_sided_p_max"] == pytest.approx(0.10 / 36)


def test_train_selection_uses_only_source_disjoint_and_gross9_eligible_pairs_and_fixed_ties() -> None:
    rows = _rows()
    rows[-1]["source_pass"] = False
    rows[-2]["same_side_disjointness_pass"] = False
    rows[-3]["gross9_pass"] = False
    rows[-4]["train_cagr_to_strict_mdd"] = 99.0
    rows[-4]["train_absolute_return"] = 0.30
    rows[-5]["train_cagr_to_strict_mdd"] = 99.0
    rows[-5]["train_absolute_return"] = 0.30
    winner = p.select_train_winner(rows)
    assert winner["candidate"] == p.CANDIDATE_FAMILY[-5]
    assert winner["frozen_before_test"] is True
    assert winner["substitution_authorized"] is False
    assert winner["rerank_authorized"] is False


def test_raw_rank_one_failure_is_terminal_without_substitution() -> None:
    rows = _rows()
    rows[-1]["train_economic_pass"] = False
    with pytest.raises(RuntimeError, match="raw rank one failed train; no substitution"):
        p.select_train_winner(rows)


@pytest.mark.parametrize("bad", [True, float("nan"), float("inf"), float("-inf")])
def test_train_selector_rejects_nonfinite_or_boolean_metrics(bad: object) -> None:
    rows = _rows()
    rows[-1]["train_cagr_to_strict_mdd"] = bad
    with pytest.raises(ValueError, match="ranking metrics"):
        p.select_train_winner(rows)


def test_selector_rejects_non_boolean_gating_flags() -> None:
    rows = _rows()
    rows[-1]["same_side_disjointness_pass"] = "yes"
    with pytest.raises(ValueError, match="same_side_disjointness_pass must be boolean"):
        p.select_train_winner(rows)


def test_validate_detects_manifest_family_predecessor_and_disjointness_drift() -> None:
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
    drifted["predecessor_family_terminal_result"]["decision"] = "pass"
    core = dict(drifted)
    core.pop("manifest_hash")
    drifted["manifest_hash"] = p.canonical_hash(core)
    with pytest.raises(RuntimeError, match="predecessor terminal binding drift"):
        p.validate(drifted)

    drifted = p.build()
    drifted["construction"]["same_side_family_disjointness_invariant"] = "allow overlap"
    core = dict(drifted)
    core.pop("manifest_hash")
    drifted["manifest_hash"] = p.canonical_hash(core)
    with pytest.raises(RuntimeError, match="disjointness invariant drift"):
        p.validate(drifted)


def test_embedded_hashes_match_current_bound_files() -> None:
    root = p.Path(__file__).resolve().parents[1]
    value = p.build()
    predecessor = value["predecessor_family_terminal_result"]
    assert p.sha256_file(root / predecessor["path"]) == predecessor["sha256"]
    assert p.sha256_file(root / value["gross9_pre2025_clock_manifest"]["path"]) == value["gross9_pre2025_clock_manifest"]["sha256"]
    for artifacts in value["component_artifacts"].values():
        for binding in artifacts.values():
            assert p.sha256_file(root / binding["path"]) == binding["sha256"]
