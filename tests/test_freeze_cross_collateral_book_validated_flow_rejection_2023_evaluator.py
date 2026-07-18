from __future__ import annotations

from training import evaluate_cross_collateral_book_validated_flow_rejection_2023 as ev
from training import freeze_cross_collateral_book_validated_flow_rejection_2023_evaluator as freeze


def test_freeze_payload_declares_zero_outcome_access() -> None:
    payload = freeze.build_payload("a" * 40)
    assert payload["outcomes_opened"] is False
    assert payload["market_rows_parsed_during_freeze"] == 0
    assert payload["funding_rows_loaded_during_freeze"] == 0
    assert payload["execution_simulation_run_during_freeze"] is False
    assert payload["opened_windows"] == []
    assert payload["sealed_windows"] == ["2023", "2024", "2025", "2026"]
    assert payload["evaluation_config"]["leverage"] == 0.5
    assert payload["evaluation_source"] == str(ev.EVALUATION_SOURCE)
