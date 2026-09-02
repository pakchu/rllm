from __future__ import annotations

import copy
import json
import subprocess

import pytest

from training import preregister_gross9_qtr_compression as p
from training import preregister_gross9_qtr_distill as distill


def _rehash(value: dict) -> dict:
    out = copy.deepcopy(value)
    core = dict(out)
    core.pop("manifest_hash", None)
    out["manifest_hash"] = p.canonical_hash(core)
    return out


def test_preregisters_replacement_compression_not_additive_alpha() -> None:
    value = p.build()
    p.validate(value)
    assert value["policy_id"] == "G9QTR-COMPRESS-8"
    assert value["source_policy_id"] == "G9QTR-DISTILL-8"
    assert value["hypothesis_count"] == 1
    assert "standalone/replacement/compression" in value["objective"]
    assert value["source_clock_reuse"]["identical_sleeves_weights_and_rule"] is True
    assert tuple(value["source_clock_reuse"]["component_sleeves"]) == distill.DISTILLED_SLEEVES
    assert value["source_clock_reuse"]["sleeve_weights"] == distill.SLEEVE_WEIGHTS
    assert value["source_clock_reuse"]["gross_exposure_sum"] == pytest.approx(0.5)


def test_near_6h_overlap_is_disclosure_but_other_gross9_checks_remain_required() -> None:
    value = p.build()
    novelty = value["terminal_additive_novelty_binding"]
    assert novelty["terminal_additive_decision"] == "terminal_gross9_novelty_reject"
    assert novelty["terminal_additive_gross9_pass"] is False
    assert novelty["additive_gross9_alpha_authorized"] is False
    assert novelty["standalone_replacement_compression_authorized_to_test"] is True
    assert novelty["near_6h_overlap_is_disclosure_not_authorization_gate"] is True
    assert novelty["all_exact_entry_occupied_and_abs_pearson_passed"] is True
    assert novelty["required_pass_checks_for_replacement"] == [
        "exact_entry_jaccard",
        "occupied_5m_bar_jaccard",
        "absolute_signed_exposure_pearson",
    ]
    assert set(novelty["near_6h_failures"]) == {"cand_rex_veto_7", "markov_transition_long"}


def test_binds_train_diagnostic_and_oos_no_repair_gates() -> None:
    value = p.build()
    prelim = value["preliminary_train_diagnostic_binding"]
    assert prelim["legacy_active_veto_familywise_p_non_authorizing"] is True
    assert prelim["preliminary_train_receipt"] == distill.PRELIMINARY_SEQUENCING_RECEIPT
    assert prelim["train_diagnostic_must_be_rerun_under_compression_policy_id"] is True
    gates = value["oos_gate_rule"]
    assert gates["sequence"] == ["test2024", "eval2025", "final2026"]
    assert gates["single_hypothesis_weekly_signflip_one_sided_p_max"] == 0.10
    assert gates["source_min_nonzero_signed_episodes"] == {"test": 12, "eval": 12, "final": 8}
    assert gates["gross9_additive_near_6h_gate_removed_by_user_objective_change"] is True
    assert gates["stop_on_first_failure"] is True
    assert gates["repair_authorized_after_failure"] is False
    assert value["outputs"]["test"] == "results/gross9_qtr_compression_test_economics_2026-09-02.json"
    assert value["evidence_boundary"]["oos_outcomes_opened_by_this_preregistration"] is False


def test_validation_catches_overlap_and_sequence_drift() -> None:
    value = p.build()
    drift = _rehash({**value, "hypothesis_count": 2})
    with pytest.raises(RuntimeError, match="hypothesis count"):
        p.validate(drift)

    drift = copy.deepcopy(value)
    drift["terminal_additive_novelty_binding"]["near_6h_overlap_is_disclosure_not_authorization_gate"] = False
    with pytest.raises(RuntimeError, match="replacement overlap"):
        p.validate(_rehash(drift))

    drift = copy.deepcopy(value)
    drift["oos_gate_rule"]["repair_authorized_after_failure"] = True
    with pytest.raises(RuntimeError, match="OOS sequence/no-repair"):
        p.validate(_rehash(drift))

    drift = copy.deepcopy(value)
    drift["source_clock_reuse"]["sleeve_weights"] = dict(distill.SLEEVE_WEIGHTS)
    drift["source_clock_reuse"]["sleeve_weights"][distill.DISTILLED_SLEEVES[0]] = 0.25
    with pytest.raises(RuntimeError, match="sleeve/weight"):
        p.validate(_rehash(drift))


def test_cli_writes_valid_json(tmp_path) -> None:
    output = tmp_path / "prereg.json"
    subprocess.run(["python", "-m", "training.preregister_gross9_qtr_compression", "--output", str(output)], check=True)
    value = json.loads(output.read_text(encoding="utf-8"))
    p.validate(value)
    assert value["manifest_hash"] == p.canonical_hash({k: v for k, v in value.items() if k != "manifest_hash"})
