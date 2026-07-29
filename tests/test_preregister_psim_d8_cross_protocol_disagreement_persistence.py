from __future__ import annotations

import json
from pathlib import Path

import pytest

from training import (
    preregister_psim_d8_cross_protocol_disagreement_persistence as prereg,
)


def test_candidate_is_independent_of_terminal_rllm2_family() -> None:
    payload = prereg.build_preregistration()
    independence = payload["independence_contract"]

    assert payload["candidate"]["id"] == "PSIM-D8-CDP1"
    assert independence["rllm2_terminal_result"]["decision"] == "reject"
    assert independence["rllm2_terminal_result"]["2022_or_later_outcomes_opened"] is False
    assert independence["forbidden_inputs"] == [
        "gemma_or_any_other_language_model_output",
        "teacher_relation_labels_or_logits",
        "text_embeddings_or_generated_text",
        "selected_subcard_or_subcard_selector",
        "ridge_fqi_q_values_or_residual_rewards",
        "2020_or_2021_market_funding_returns_or_policy_metrics",
    ]
    assert independence["repair_or_successor_of_rllm2"] is False


def test_structural_disagreement_formula_and_family_are_frozen() -> None:
    payload = prereg.build_preregistration()
    feature = payload["feature_contract"]
    family = payload["frozen_family"]

    assert feature["primary_schedule"] == "ARCHIVE_D90"
    assert feature["unit_components"] == list(prereg.UNIT_COMPONENTS)
    assert feature["unit_score_formula"] == (
        "mean(component_disagreement_j for j in the 9 frozen components)"
    )
    assert feature["daily_aggregate"] == (
        "arithmetic_mean(unit_score for all eligible relation units)"
    )
    assert feature["fast_ewma_half_life_cards"] == 3
    assert feature["slow_ewma_half_life_cards"] == 30
    assert family["slow_floor_grid"] == [0.35, 0.50, 0.65]
    assert family["fast_slow_gap_grid"] == [0.05, 0.10, 0.15]
    assert family["candidate_count"] == 9
    assert len(family["candidate_ids"]) == 9
    assert len(set(family["candidate_ids"])) == 9


def test_clock_direction_cost_and_untouched_veto_are_fixed() -> None:
    payload = prereg.build_preregistration()
    action = payload["action_contract"]
    evaluation = payload["evaluation_contract"]

    assert action["short_rule"] == (
        "slow_ewma >= slow_floor AND fast_ewma - slow_ewma >= fast_slow_gap"
    )
    assert action["long_rule"] == (
        "slow_ewma >= slow_floor AND slow_ewma - fast_ewma >= fast_slow_gap"
    )
    assert action["holding_bars_5m"] == 288
    assert action["overlap"] == "forbidden_first_signal_wins"
    assert evaluation["selection_period"] == ["2022-01-01", "2022-12-31"]
    assert evaluation["future_veto_period"] == ["2023-01-01", "2023-12-31"]
    assert evaluation["future_veto_is_untouched"] is True
    assert evaluation["base_cost_per_side"] == 0.0006
    assert evaluation["stress_cost_per_side"] == 0.0010
    assert evaluation["funding"] == "exact_mark_funding_cashflows"
    assert evaluation["drawdown"] == "strict_intra_position_adverse_excursion"


def test_preregistration_does_not_open_source_or_outcome_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[Path] = []
    real_hash = prereg.sha256_file

    def recording_hash(path: Path) -> str:
        opened.append(path)
        return real_hash(path)

    monkeypatch.setattr(prereg, "sha256_file", recording_hash)
    payload = prereg.build_preregistration()

    assert prereg.D8_CARDS not in opened
    assert prereg.D8_EVENTS not in opened
    assert prereg.MARKET not in opened
    assert prereg.FUNDING not in opened
    assert payload["access_boundary"]["source_payload_rows_parsed"] == 0
    assert payload["access_boundary"]["market_rows_parsed"] == 0
    assert payload["access_boundary"]["funding_rows_parsed"] == 0
    assert payload["access_boundary"]["economic_metrics_computed"] == 0


def test_preregistration_is_deterministic_self_hashed_and_write_once(
    tmp_path: Path,
) -> None:
    first = prereg.build_preregistration()
    second = prereg.build_preregistration()
    core = {key: value for key, value in first.items() if key != "manifest_hash"}

    assert first == second
    assert first["manifest_hash"] == prereg.canonical_hash(core)
    output = tmp_path / "registration.json"
    assert prereg.write_preregistration(output) == first
    assert json.loads(output.read_text(encoding="utf-8")) == first
    assert prereg.write_preregistration(output) == first
    output.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="drift"):
        prereg.write_preregistration(output)
