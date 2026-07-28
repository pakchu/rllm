from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training.build_cash_open_cross_asset_gap_features import FEATURE_COLUMNS


PREREGISTRATION = Path(
    "results/"
    "cash_open_cross_asset_gap_relay_marginal_preregistration_2026-07-28.json"
)


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _protocol() -> dict[str, object]:
    return json.loads(PREREGISTRATION.read_text(encoding="utf-8"))


def test_cogr_preregistration_is_bound_to_frozen_inputs() -> None:
    protocol = _protocol()
    provenance = protocol["input_provenance"]
    for record in provenance.values():
        assert _sha256(record["path"]) == record["sha256"]
    assert provenance["cogr_evaluator"]["path"] == (
        "training/evaluate_cash_open_cross_asset_gap_relay_marginal.py"
    )
    assert protocol["source_contract"]["feature_count"] == 31
    assert tuple(protocol["feature_contract"]["columns"]) == FEATURE_COLUMNS


def test_cogr_has_clean_h1_h2_2024_roles_and_top1_only() -> None:
    protocol = _protocol()
    assert protocol["physical_selection_cutoff"] == "2024-01-01"
    assert (
        protocol["source_contract"]["selection_cutoff_exclusive"]
        == "2024-01-01"
    )
    assert (
        protocol["source_contract"]["source_artifact_cutoff_exclusive"]
        == "2025-01-01"
    )
    folds = protocol["expanding_folds"]
    assert [fold["name"] for fold in folds] == [
        "calibration_2023h1",
        "selection_2023h2",
        "eval_2024",
    ]
    assert folds[0]["candidate_entries_allowed"] is False
    assert folds[0]["outcomes_used_for_ranking"] is False
    assert folds[1]["outcomes_used_for_ranking"] is True
    assert folds[2]["outcomes_used_for_ranking"] is False
    assert folds[2]["exact_frozen_top1_only"] is True
    selection = protocol["selection_contract"]
    assert selection["top1_only"] is True
    assert "rank2" in selection["no_fallback"]


def test_cogr_contract_freezes_costs_cells_controls_and_no_future_repair() -> None:
    protocol = _protocol()
    learner = protocol["learner_contract"]
    assert learner["normal_cost_per_notional_per_side"] == 0.0006
    assert learner["candidate_stress_cost_per_notional_per_side"] == 0.001
    assert learner["entry_delay_bars"] == 1
    assert learner["hold_bars"] == 144
    assert learner["score_quantile"] == 0.75
    assert "signal_position-1" in learner["entry_delay_definition"]
    universe = protocol["candidate_universe"]
    assert universe["portfolio_cells"] == 12
    assert universe["candidate_weight_grid"] == [0.25, 0.5, 0.75, 1.0]
    controls = protocol["controls"]
    assert len(controls["feature_model_controls"]) == 5
    assert len(controls["fixed_schedule_controls"]) == 4
    future = protocol["future_veto_contract"]
    assert future["future_can_rerank"] is False
    assert future["future_can_repair"] is False
    assert future["future_can_select_another_cell"] is False


def test_cogr_gates_require_sample_balance_concentration_and_same_gross_gain() -> None:
    protocol = _protocol()
    h2 = protocol["selection_contract"]
    standalone = h2["selection_2023h2_standalone_requirements"]
    portfolio = h2["selection_2023h2_portfolio_requirements"]
    assert standalone["minimum_trades"] == 25
    assert standalone["minimum_long_share"] == 0.2
    assert standalone["minimum_short_share"] == 0.2
    assert standalone["maximum_month_share"] == 0.35
    assert standalone["maximum_weekday_share"] == 0.35
    assert (
        portfolio[
            "minimum_cagr_mdd_improvement_vs_same_gross_prorata_gross9"
        ]
        == 0.05
    )
    eval_contract = protocol["eval_2024_veto_contract"]
    assert eval_contract["standalone_requirements"]["minimum_trades"] == 50
    stats = eval_contract["statistical_requirement"]
    assert stats["maximum_p_value_each_test"] == 0.1
    assert stats["minimum_active_weeks_each_test"] == 26
    assert protocol["controls"]["best_control_rule"].startswith("Every one")
    gross = protocol["same_gross_formula"]
    assert gross["baseline_configured_gross_units"] == 9.0
    assert "0.5*c" in gross["candidate_weight_unit"]
