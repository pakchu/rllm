from __future__ import annotations

import copy
import json

import pytest

from training import preregister_high_volatility_dcs_ehlers_active_veto as p
from training import preregister_high_volatility_cross_structure_action_vote as hvcav


def test_manifest_fixes_independent_singleton_and_exact_components() -> None:
    value = p.build()
    p.validate(value)
    assert value["policy_id"] == "HVDCEV-8"
    assert value["action_ids"] == ["HVDCS-8", "HVELRSI-24"]
    assert value["component_ids"] == ["HVDCS-8", "HVELRSI-24"]
    assert value["candidate_family"] == ["HVDCEV-8"]
    assert value["candidate_family_size"] == 1
    assert value["single_candidate_only"] is True
    assert value["familywise_multiplicity"] == (
        "none; independent singleton family; no Bonferroni adjustment"
    )


def test_opposite_side_veto_truth_table_is_exact() -> None:
    assert p.active_state_veto_side(0, 1) == 0
    assert p.active_state_veto_side(1, 0) == 1
    assert p.active_state_veto_side(1, 1) == 1
    assert p.active_state_veto_side(-1, -1) == -1
    assert p.active_state_veto_side(1, -1) == 0
    assert p.active_state_veto_side(-1, 1) == 0
    with pytest.raises(ValueError, match=r"exact -1, 0, or \+1"):
        p.active_state_veto_side(1, True)
    with pytest.raises(ValueError, match=r"exact -1, 0, or \+1"):
        p.active_state_veto_side(2, 1)


def test_exact_clock_and_no_tuning_or_hvcav_repair() -> None:
    value = p.build()
    construction = value["construction"]
    assert value["clock"] == {
        "decisions": "exact completed four-hour boundaries 00/04/08/12/16/20 UTC",
        "entry": "exact feature-available D+5m entry",
        "hold": "8 elapsed hours",
        "component_clocks": "untouched exact frozen primary clocks",
        "timestamp_tolerance": "none",
        "funding": "not a signal input; exact held settlements only after source and Gross9 pass",
    }
    assert construction["timestamp_tolerance"] == "none"
    assert construction["weights"] == "none"
    assert construction["priority"] == (
        "none; HVDCS-8 is the fixed primary and HVELRSI-24 is only an opposite-side veto"
    )
    assert construction["alternatives"] == "none"
    assert construction["additional_or_tuned_thresholds"] == "none"
    assert construction["independent_from_HVCAV_8"] is True
    assert construction["HVCAV_subset_or_quorum_repair"] == "none"


def test_stages_and_all_gates_are_exactly_hvcav_8() -> None:
    value = p.build()
    prior = hvcav.build()
    for key in (
        "stages",
        "source_support_gates",
        "gross9_novelty_gates",
        "economic_gates",
    ):
        assert value[key] == prior[key]
    assert "stop on first failure" in value["stopping_rule"]
    assert "HVCAV repair" in value["stopping_rule"]
    assert value["operator_activation_gate"] == {
        "minimum_opposite_vetoes_train": 1,
        "minimum_opposite_vetoes_full": 3,
        "must_pass_before_gross9": True,
    }


def test_all_component_artifacts_are_hash_pinned_and_pass_scalars_hold() -> None:
    value = p.build()
    assert set(value["component_artifacts"]) == set(p.COMPONENT_IDS)
    for component, artifacts in value["component_artifacts"].items():
        assert set(artifacts) == {"preregistration", "support", "gross9", "clock"}
        for artifact_type, artifact in artifacts.items():
            assert set(artifact) == {"path", "sha256"}
            assert len(artifact["sha256"]) == 64
            assert p.sha256_file(artifact["path"]) == artifact["sha256"]
            if artifact_type != "clock":
                artifact_value = json.loads(open(artifact["path"]).read())
                for key, expected in p.EXPECTED_COMPONENT_SCALARS[component][artifact_type].items():
                    assert artifact_value[key] == expected
            assert "economic" not in artifact["path"]


def test_known_prior_evidence_disclosed_while_combination_is_sealed() -> None:
    value = p.build()
    boundary = value["research_boundary"]
    assert boundary["all_component_prior_outcomes_known"] is True
    assert boundary["prior_HVCAV_incidence_known"] is True
    assert boundary["prior_HVCAV_outcomes_known"] is True
    assert boundary["component_selection_used_prior_train_outcomes"] is True
    assert boundary["component_test_eval_final_outcomes_used"] is False
    assert boundary["archived_component_later_stage_artifacts_exist"] is True
    assert boundary["archived_component_later_stage_outcomes_used_for_selection"] is False
    assert boundary["known_component_or_HVCAV_evidence_used_to_alter_components"] is True
    assert boundary["known_HVCAV_evidence_used_for_subset_or_quorum_repair"] is False
    assert boundary["combined_incidence_opened"] is False
    assert boundary["combined_postentry_returns_or_pnl_opened"] is False
    assert boundary["economics_artifacts_read_during_validation"] is False
    assert value["combined_incidence_opened"] is False
    assert value["combined_outcomes_opened"] is False
    assert value["exploratory_discovery"] is True
    assert value["fresh_confirmatory_evidence"] is False


def test_manifest_hash_and_component_pass_scalars_detect_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    drifted = copy.deepcopy(p.build())
    drifted["construction"]["routing_rule"] = "use fallback first"
    with pytest.raises(RuntimeError, match="preregistration drift"):
        p.validate(drifted)

    original_read = p._read_json_object

    def drifted_read(path: str) -> dict[str, object]:
        value = dict(original_read(path))
        if path.endswith("_support_2026-08-10.json"):
            value["support_passed"] = False
        return value

    monkeypatch.setattr(p, "_read_json_object", drifted_read)
    with pytest.raises(RuntimeError, match="pass scalar drift"):
        p.validate(p.build())
