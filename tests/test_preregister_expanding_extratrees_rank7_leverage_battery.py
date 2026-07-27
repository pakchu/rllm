from __future__ import annotations

import json
from pathlib import Path

import pytest

from training import (
    preregister_expanding_extratrees_rank7_leverage_battery as prereg,
)


def test_fixed_family_selection_boundary_and_target() -> None:
    payload = prereg.build_preregistration()

    assert payload["candidate"]["leverage_grid"] == [0.5, 0.75, 1.0, 1.25, 1.5]
    assert payload["selection"]["windows"] == ["2023", "2024", "selection"]
    assert payload["report_only_evaluation"]["windows"] == [
        "2025",
        "2026h1",
        "future",
        "all",
    ]
    assert payload["research_status"]["pristine_discovery_oos"] is False
    assert payload["report_only_evaluation"]["no_repair_after_open"] is True
    assert payload["user_target"]["required_windows"] == ["future", "all"]
    assert payload["user_target"]["cagr_pct_at_least"] == 50.0
    assert payload["user_target"]["strict_mdd_pct_at_most"] == 15.0
    assert payload["user_target"]["cagr_to_strict_mdd_at_least"] == 3.0


def test_selection_gates_are_stric_and_cost_stressed() -> None:
    payload = prereg.build_preregistration()
    selection = payload["selection"]

    assert selection["base_gates_each_window"] == {
        "absolute_return_pct_strictly_above": 0.0,
        "cagr_to_strict_mdd_at_least": 3.0,
        "strict_mdd_pct_at_most": 12.0,
    }
    assert selection["minimum_trades"] == {
        "2023": 12,
        "2024": 12,
        "selection": 24,
    }
    assert selection["stress_gates_each_window"] == {
        "absolute_return_pct_strictly_above": 0.0,
        "strict_mdd_pct_at_most": 15.0,
    }
    assert payload["immutable_policy"]["base_cost_per_side"] == 0.0006
    assert payload["immutable_policy"]["stress_cost_per_side"] == 0.001


def test_integrity_contract_freezes_schedule_and_future_exclusion() -> None:
    payload = prereg.build_preregistration()
    integrity = payload["integrity"]

    assert integrity["all_leverage_cells_must_share_exact_trade_clock_hashes"]
    assert integrity["all_grid_cells_reported"]
    assert integrity["nonfinite_metrics_fail_closed"]
    assert integrity["future_metrics_must_not_enter_selection_payload"]
    assert payload["immutable_policy"]["selected_position_hash"] == (
        "8ffbd55f07ceda0e82c270fe4b370fffba44bb3fcfc807368c4385d2ba97f531"
    )


def test_preregistration_is_deterministic_self_hashed_and_drift_protected(
    tmp_path: Path,
) -> None:
    first = prereg.build_preregistration()
    second = prereg.build_preregistration()
    core = {key: value for key, value in first.items() if key != "manifest_hash"}

    assert first == second
    assert first["manifest_hash"] == prereg.canonical_hash(core)
    output = tmp_path / "registration.json"
    summary = tmp_path / "registration.md"
    assert prereg.write_preregistration(output, summary) == first
    assert json.loads(output.read_text(encoding="utf-8")) == first
    assert prereg.write_preregistration(output, summary) == first
    output.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="drift"):
        prereg.write_preregistration(output, summary)
