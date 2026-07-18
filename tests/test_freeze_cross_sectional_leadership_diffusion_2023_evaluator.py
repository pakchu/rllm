from __future__ import annotations

from training import freeze_cross_sectional_leadership_diffusion_2023_evaluator as freeze


def test_build_manifest_keeps_every_outcome_sealed() -> None:
    payload = freeze.build_manifest("a" * 40, primary_rows=106)
    freeze.validate_manifest(payload)
    assert payload["outcomes_opened"] is False
    assert payload["opened_windows"] == []
    assert payload["sealed_windows"] == ["2023", "2024", "2025", "2026"]
    assert payload["mutable_parameters"] == []
    assert payload["market_rows_parsed_during_freeze"] == 0
    assert payload["funding_rows_loaded_during_freeze"] == 0
    assert payload["execution_simulation_run_during_freeze"] is False


def test_manifest_binds_primary_and_control_counts() -> None:
    payload = freeze.build_manifest("b" * 40, primary_rows=106)
    assert payload["primary_clock_rows"] == 106
    assert payload["control_clock_rows"] == freeze.evaluate.CONTROL_COUNTS
