from __future__ import annotations

from training import evaluate_cross_collateral_book_validated_flow_rejection_2023 as ev
from training import freeze_cross_collateral_book_validated_flow_rejection_2023_evaluator as freeze


def test_recovery_freeze_discloses_failed_outcome_opening() -> None:
    payload = freeze.build_payload("a" * 40)
    assert payload["outcomes_opened"] is True
    assert payload["prior_freeze_sha256"] == freeze.PRIOR_FREEZE_SHA256
    assert payload["first_attempt"]["execution_sources_opened"] is True
    assert payload["first_attempt"]["result_artifact_written"] is False
    assert payload["first_attempt"]["outcome_metrics_used_for_repair"] is False
    assert payload["first_attempt"]["strategy_parameters_changed"] is False
    assert payload["market_rows_parsed_during_freeze"] == 0
    assert payload["funding_rows_loaded_during_freeze"] == 0
    assert payload["execution_simulation_run_during_freeze"] is False
    assert payload["opened_windows"] == []
    assert payload["sealed_windows"] == ["2023", "2024", "2025", "2026"]
    assert payload["evaluation_config"]["leverage"] == 0.5
    assert payload["evaluation_source"] == str(ev.EVALUATION_SOURCE)
