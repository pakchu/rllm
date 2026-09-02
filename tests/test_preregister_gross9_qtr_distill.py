from __future__ import annotations

import copy
import json
import subprocess

import pytest

from training import preregister_gross9_async_active_veto_search as active_veto
from training import preregister_gross9_qtr_distill as p


def _rehash(value: dict[str, object]) -> dict[str, object]:
    core = dict(value)
    core.pop("manifest_hash")
    value["manifest_hash"] = p.canonical_hash(core)
    return value


def test_build_freezes_qtr_distill_identity_components_sleeves_and_weights() -> None:
    value = p.build()
    p.validate(value)
    assert value["policy_id"] == "G9QTR-DISTILL-8"
    assert value["research_status"] == "adaptive_exploratory_shadow_until_all_oos_stages_pass"
    assert value["fresh_confirmatory_evidence"] is False
    assert value["llm_path_paused"] is True
    assert value["component_order"] == list(active_veto.COMPONENT_ORDER)
    assert value["selection_rule"]["winner_veto"] == "HVCQTR-24"
    assert value["selection_rule"]["selected_bases"] == ["HVDEMWMV-24", "HVCPF17-8", "HVDIMIO-8", "HVLVR-8"]
    assert value["selection_rule"]["selected_sleeves"] == list(p.DISTILLED_SLEEVES)
    assert value["portfolio_construction"]["sleeve_weights"] == p.SLEEVE_WEIGHTS
    assert sum(value["portfolio_construction"]["sleeve_weights"].values()) == pytest.approx(0.5)
    assert value["portfolio_construction"]["no_renormalization"] is True
    assert value["portfolio_construction"]["no_vol_target"] is True
    assert value["portfolio_construction"]["no_weight_retune"] is True


def test_binds_terminal_active_veto_artifacts_and_placeholders_for_future_code() -> None:
    value = p.build()
    assert value["active_veto_terminal_artifacts"] == p.ACTIVE_VETO_TERMINAL_ARTIFACTS
    for receipt in value["active_veto_terminal_artifacts"].values():
        assert p.sha256_file(receipt["path"]) == receipt["sha256"]
    assert p.sha256_file(value["implementation"]["preregister"]["path"]) == value["implementation"]["preregister"]["sha256"]
    assert p.sha256_file(value["implementation"]["portfolio_builder"]["path"]) == value["implementation"]["portfolio_builder"]["sha256"]
    assert value["implementation"]["gross9_novelty_evaluator"]["sha256"] == "PENDING_G9QTR_DISTILL_NOVELTY_EVALUATOR_BINDING"
    assert value["implementation"]["economics_evaluator"]["sha256"] == p.sha256_file("training/evaluate_gross9_qtr_distill_economics.py")


def test_selection_proof_reproduces_qtr_max_distinct_base_choice_from_train_artifact() -> None:
    rows = json.loads(open(p.ACTIVE_VETO_TERMINAL_ARTIFACTS["train_economics"]["path"], encoding="utf-8").read())["candidates"]
    proof = p.select_distilled_sleeves(rows)
    assert proof["winner_veto"] == "HVCQTR-24"
    assert proof["eligible_base_count_by_veto"] == {"HVCQTR-24": 4, "HVDEMWMV-24": 1, "HVEIV-24": 2, "HVSAUD-8": 3}
    assert proof["selected_bases"] == ["HVDEMWMV-24", "HVCPF17-8", "HVDIMIO-8", "HVLVR-8"]
    assert proof["selected_sleeves"] == list(p.DISTILLED_SLEEVES)


def test_records_legacy_weekly_p_failure_without_relabeling_train_pass() -> None:
    value = p.build()
    legacy = value["legacy_multiplicity_disclosure"]
    assert legacy["legacy_raw_weekly_p_threshold"] == pytest.approx(0.1 / 72)
    assert legacy["all_selected_sleeves_failed_legacy_familywise_weekly_p"] is True
    assert legacy["not_relabelled_train_pass"] is True
    for sleeve, diagnostics in value["distilled_train_diagnostics"].items():
        assert sleeve in p.DISTILLED_SLEEVES
        assert diagnostics["weekly_signflip_p"] > legacy["legacy_raw_weekly_p_threshold"]
        assert diagnostics["return_pct"] > 0
        assert diagnostics["stress_return_pct"] > 0
        assert diagnostics["cagr_to_strict_mdd"] >= 3
        assert diagnostics["stress_cagr_to_strict_mdd"] >= 2.5


def test_oos_sequence_is_single_hypothesis_no_repair_with_same_shape_gates() -> None:
    value = p.build()
    gates = value["oos_gate_rule"]
    assert gates["sequence"] == ["test2024", "eval2025", "final2026"]
    assert gates["single_hypothesis_weekly_signflip_one_sided_p_max"] == 0.10
    assert gates["absolute_return_positive"] is True
    assert gates["cagr_to_strict_mdd_min"] == 3.0
    assert gates["strict_mdd_max_pct"] == 15.0
    assert gates["mean_gross_underlying_min_bp"] == 20.0
    assert gates["stress_absolute_return_positive"] is True
    assert gates["stress_cagr_to_strict_mdd_min"] == 2.5
    assert gates["each_calendar_half_positive"] is True
    assert gates["source_min_nonzero_signed_episodes"] == {"test2024": 12, "eval2025": 12, "final2026": 8}
    assert gates["stop_on_first_failure"] is True
    assert gates["repair_authorized_after_failure"] is False
    assert value["evidence_boundary"]["oos_outcomes_opened_by_this_preregistration"] is False
    assert value["evidence_boundary"]["adaptive_exploratory_until_all_oos_pass"] is True


def test_validation_catches_material_drift() -> None:
    drifted = p.build()
    drifted["selection_rule"]["winner_veto"] = "HVSAUD-8"
    with pytest.raises(RuntimeError, match="selection drift"):
        p.validate(_rehash(drifted))

    drifted = p.build()
    drifted["portfolio_construction"]["sleeve_weights"] = copy.deepcopy(p.SLEEVE_WEIGHTS)
    drifted["portfolio_construction"]["sleeve_weights"][p.DISTILLED_SLEEVES[0]] = 0.25
    with pytest.raises(RuntimeError, match="weight drift"):
        p.validate(_rehash(drifted))

    drifted = p.build()
    drifted["oos_gate_rule"]["sequence"] = ["eval2025", "test2024", "final2026"]
    with pytest.raises(RuntimeError, match="OOS sequence drift"):
        p.validate(_rehash(drifted))

    drifted = p.build()
    drifted["legacy_multiplicity_disclosure"]["not_relabelled_train_pass"] = False
    with pytest.raises(RuntimeError, match="legacy p disclosure drift"):
        p.validate(_rehash(drifted))

    drifted = p.build()
    drifted["evidence_boundary"]["oos_outcomes_opened_by_this_preregistration"] = True
    with pytest.raises(RuntimeError, match="OOS boundary drift"):
        p.validate(_rehash(drifted))


def test_cli_writes_valid_json(tmp_path) -> None:
    output = tmp_path / "prereg.json"
    subprocess.run(["python", "-m", "training.preregister_gross9_qtr_distill", "--output", str(output)], check=True)
    value = json.loads(output.read_text(encoding="utf-8"))
    p.validate(value)
    assert value["manifest_hash"] == p.canonical_hash({k: v for k, v in value.items() if k != "manifest_hash"})
