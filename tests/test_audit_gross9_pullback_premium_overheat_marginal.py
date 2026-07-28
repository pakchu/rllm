from __future__ import annotations

import copy

import numpy as np
import pytest

import training.audit_gross9_pullback_premium_overheat_marginal as module


def test_preregistration_is_hash_bound_and_semantically_valid() -> None:
    payload = module.load_preregistration(module.PREREGISTRATION)
    assert payload["selection_contract"]["candidate_weight_grid"] == [
        0.25,
        0.5,
        0.75,
        1.0,
    ]
    assert payload["future_veto_contract"]["future_can_rerank"] is False
    assert payload["future_veto_contract"]["future_can_repair"] is False


def test_semantic_validator_rejects_missing_contamination_disclosure() -> None:
    payload = module.load_preregistration(module.PREREGISTRATION)
    mutated = copy.deepcopy(payload)
    mutated["candidate_disclosure"]["standalone_future_already_exposed"] = False
    with pytest.raises(RuntimeError, match="contamination disclosure"):
        module.validate_preregistration_semantics(mutated)


def test_same_gross_control_has_identical_configured_gross() -> None:
    combined, comparator = module.same_gross_weights(0.75)
    assert sum(combined.values()) == pytest.approx(9.75)
    assert sum(comparator.values()) == pytest.approx(9.75)
    assert combined[module.CANDIDATE] == pytest.approx(0.75)
    assert module.CANDIDATE not in comparator


def test_result_hash_detects_mutation() -> None:
    payload = module.finalize_payload({"phase": "selection", "value": 3})
    module.verify_result_hash(payload)
    mutated = dict(payload)
    mutated["value"] = 4
    with pytest.raises(RuntimeError, match="result hash drifted"):
        module.verify_result_hash(mutated)


def test_paired_statistics_are_deterministic_and_positive() -> None:
    effects = np.asarray([0.002] * 24, dtype=float)
    first = module.paired_statistics(effects)
    second = module.paired_statistics(effects)
    assert first == second
    assert first["active_weeks"] == 24
    assert first["sign_flip_pvalue"] < 0.01
    assert first["bootstrap_90pct_lower_mean"] > 0.0


def test_frozen_candidate_replays_exact_pre2024_schedule() -> None:
    frozen, known_oos = module.validate_candidate_freeze(module.Config())
    assert frozen["freeze_hash"] == known_oos["freeze_hash"]
    assert frozen["selection_schedule_hashes"]["pre_2024"]
