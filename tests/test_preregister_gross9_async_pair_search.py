from __future__ import annotations

import copy
from itertools import combinations

import pytest

from training import preregister_gross9_async_pair_search as p


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


def test_manifest_fixes_exact_9_component_async_family_and_clock() -> None:
    value = p.build()
    p.validate(value)
    assert value["policy_id"] == "G9ASYNCPAIR-8"
    assert value["component_order"] == list(p.COMPONENT_ORDER)
    assert value["component_count"] == 9
    assert value["candidate_family_size"] == 36
    assert value["candidate_family"] == [
        f"{left}__ASYNC_SAME_SIDE_6H__{right}"
        for left, right in combinations(p.COMPONENT_ORDER, 2)
    ]
    assert value["construction"]["entry_rule"].startswith("for a pair (A,B)")
    assert "prior 6 elapsed hours" in value["construction"]["entry_rule"]
    assert "allowed once" in value["construction"]["same_timestamp_confirmation"]
    assert value["clock"]["hold"] == "8 elapsed hours"
    assert value["clock"]["entry"] == "later same-side trigger event"
    assert value["clock"]["gross_exposure"] == 0.5
    assert "inside each candidate-pair clock" in value["construction"]["reservation"]


def test_manifest_binds_exact_pass_artifacts_and_gross9_manifest_hashes() -> None:
    value = p.build()
    assert set(value["implementation"]) == {"preregister", "train_clock_builder"}
    for binding in value["implementation"].values():
        assert p.sha256_file(binding["path"]) == binding["sha256"]
    assert set(value["component_artifacts"]) == set(p.COMPONENT_ORDER)
    for component, artifacts in value["component_artifacts"].items():
        assert set(artifacts) == {
            "train_economics",
            "preregistration",
            "source_support",
            "gross9",
            "clock",
        }
        assert artifacts["train_economics"]["path"].endswith("_train_economics_2026-08-16.json") or artifacts["train_economics"]["path"].endswith("_train_economics_2026-08-17.json") or artifacts["train_economics"]["path"].endswith("_train_economics_2026-08-18.json")
        for binding in artifacts.values():
            assert len(binding["sha256"]) == 64, component
        assert len(artifacts["train_economics"]["manifest_hash"]) == 64
        assert len(artifacts["gross9"]["manifest_hash"]) == 64
        assert artifacts["clock"]["rows"] > 0
    gross9 = value["gross9_pre2025_clock_manifest"]
    assert gross9 == p.GROSS9_PRE2025_CLOCK_MANIFEST
    assert gross9["sha256"] == "5433812da786a959cda1cfcf4825bc2e4a228ea8152a4b8cce1e867f29adf073"
    assert gross9["manifest_hash"] == "c1f7c2096cea035d053dd3d7b887b13f3220b6d96ddb99893b5be26cb44ae650"


def test_boundary_and_bonferroni_are_result_blind_and_raw_rank_one() -> None:
    value = p.build()
    boundary = value["research_boundary"]
    assert boundary["llm_path_paused"] is True
    assert boundary["pair_combination_incidence_opened_by_preregistration"] is False
    assert boundary["pair_combination_outcomes_opened_by_preregistration"] is False
    assert value["selection"]["raw_rank_one_no_substitution"] is True
    assert "no rerank" in value["selection"]["later_stages"]
    assert value["stages"]["train"] == ["2023-07-01T00:00:00Z", "2024-01-01T00:00:00Z"]
    assert value["familywise_multiplicity"]["rule"] == "Bonferroni"
    assert value["familywise_multiplicity"]["number_of_hypotheses"] == 36
    assert value["economic_gates"]["weekly_signflip_one_sided_p_max"] == pytest.approx(0.10 / 36)


def test_train_selection_uses_only_source_and_gross9_eligible_pairs_and_fixed_ties() -> None:
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


def test_validate_detects_manifest_and_family_drift() -> None:
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


def test_embedded_hashes_match_current_bound_files() -> None:
    root = p.Path(__file__).resolve().parents[1]
    value = p.build()
    assert p.sha256_file(root / value["gross9_pre2025_clock_manifest"]["path"]) == value["gross9_pre2025_clock_manifest"]["sha256"]
    for artifacts in value["component_artifacts"].values():
        for binding in artifacts.values():
            assert p.sha256_file(root / binding["path"]) == binding["sha256"]
