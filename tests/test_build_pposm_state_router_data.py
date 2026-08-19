import pandas as pd

from training import build_pposm_state_router_data as router


def test_route_priority_is_capitulation_then_overheat():
    assert router.route_label(capitulation=True, overheat=True) == "TP4"
    assert router.route_label(capitulation=False, overheat=True) == "SKIP"
    assert router.route_label(capitulation=False, overheat=False) == "TP12"


def test_prompt_contains_formula_but_not_resolved_state():
    manifest = {
        "spec": {
            "state_priority": ["capitulation", "overheat", "normal"],
            "capitulation": "weak_week AND stress",
        },
        "state_thresholds": {"x_q50": 1.0},
    }
    prompt = router.causal_prompt({"x": 2.0}, manifest)
    assert "signal_features" in prompt and "frozen_formula" in prompt
    assert "future" not in prompt.lower()


def test_decision_positions_are_window_partitioned():
    market = pd.DataFrame(
        {"date": pd.to_datetime(["2023-01-01", "2024-02-01", "2025-02-01", "2026-02-01"])}
    )
    positions = router.decision_positions(market, [True, True, True, True])
    assert positions["pre_2024"] == (0,)
    assert positions["test_2024"] == (1,)
    assert positions["eval_2025"] == (2,)
    assert positions["holdout_2026"] == (3,)
