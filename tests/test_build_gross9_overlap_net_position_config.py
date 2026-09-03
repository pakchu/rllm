from __future__ import annotations

import json

import pandas as pd
import pytest

from training import build_gross9_overlap_net_position_config as n


def test_net_position_risk_offsets_long_and_short() -> None:
    clock = pd.DataFrame(
        [
            {
                "sleeve": "a",
                "weight": 0.3,
                "entry_time": pd.Timestamp("2023-07-01T00:00:00Z"),
                "exit_time": pd.Timestamp("2023-07-01T08:00:00Z"),
                "side": 1,
            },
            {
                "sleeve": "b",
                "weight": 0.2,
                "entry_time": pd.Timestamp("2023-07-01T00:00:00Z"),
                "exit_time": pd.Timestamp("2023-07-01T08:00:00Z"),
                "side": -1,
            },
        ]
    )
    metrics = n.net_exposure_metrics(
        clock,
        pd.Timestamp("2023-07-01T00:00:00Z"),
        pd.Timestamp("2023-07-01T08:00:00Z"),
    )
    assert metrics["max_abs_net_position"] == pytest.approx(0.1)
    assert metrics["legacy_max_nonnet_gross"] == pytest.approx(0.5)


def test_generated_config_keeps_weights_and_explains_costs(tmp_path) -> None:
    config, audit = n.run(tmp_path / "config.json", tmp_path / "audit.json")
    assert config["position_risk"]["long_short_offset_for_risk"] is True
    assert config["net_position_risk_metrics"]["max_abs_net_position"] == pytest.approx(0.6)
    assert config["net_position_risk_metrics"]["mean_abs_net_position"] == pytest.approx(
        0.07363834422657953
    )
    costs = config["operating_cost_failure"]
    assert costs["annualized_aggregate_net_turnover_x"] > 200
    assert costs["turnover_cap_multiple"] > 2
    assert costs["failed_gates"] == ["turnover_cap", "sleeve_turnover_share_cap"]
    assert costs["base_fee_less_funding_pct_of_initial_equity"] == pytest.approx(
        5.366689816970291
    )
    invariance = config["selection_invariance_evidence"]
    assert invariance["legacy_exposure_gate_failures"] == 0
    assert invariance["turnover_gate_passes"] == 0
    expected = json.loads(n.SELECTION.read_text(encoding="utf-8"))["authoritative_rank1"][
        "sleeve_weights"
    ]
    assert config["sleeve_weights"] == expected
    assert audit["config"]["path"] == str(tmp_path / "config.json")
    assert audit["portfolio_weights_changed"] is False
    assert audit["selection_rank_changed"] is False
