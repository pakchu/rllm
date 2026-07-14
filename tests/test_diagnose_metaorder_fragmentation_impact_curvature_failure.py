from __future__ import annotations

import pandas as pd
import pytest

from training import diagnose_metaorder_fragmentation_impact_curvature_failure as diagnose
from training.evaluate_metaorder_fragmentation_impact_curvature import EvaluationConfig


def test_trade_ledger_separates_gross_cost_and_posthoc_inversion() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=5, freq="5min"),
            "open": [100.0, 100.0, 100.0, 105.0, 110.0],
        }
    )
    schedule = pd.DataFrame(
        [
            {
                "entry_position": 2,
                "exit_position": 4,
                "side": 1,
                "branch": "continuation",
            }
        ]
    )
    cfg = EvaluationConfig()
    ledger = diagnose.build_trade_ledger(frame, schedule, cfg)
    per_side_cost = 0.0006 * 0.5
    gross = 0.5 * 0.10
    expected_net = (1.0 - per_side_cost) * (1.0 + gross) * (1.0 - per_side_cost) - 1.0
    expected_inverted = (
        (1.0 - per_side_cost) * (1.0 - gross) * (1.0 - per_side_cost) - 1.0
    )
    assert ledger.loc[0, "account_gross_return"] == pytest.approx(gross)
    assert ledger.loc[0, "account_net_return"] == pytest.approx(expected_net)
    assert ledger.loc[0, "posthoc_inverted_account_net_return"] == pytest.approx(
        expected_inverted
    )


def test_summary_reports_basis_points_and_win_rate() -> None:
    ledger = pd.DataFrame(
        {
            "underlying_raw_return": [0.01, -0.005],
            "account_gross_return": [0.005, -0.0025],
            "account_net_return": [0.0044, -0.0031],
            "posthoc_inverted_account_net_return": [-0.0056, 0.0019],
        }
    )
    summary = diagnose.summarize_ledger(ledger)
    assert summary["trade_count"] == 2
    assert summary["mean_account_gross_bps"] == pytest.approx(12.5)
    assert summary["mean_account_net_bps"] == pytest.approx(6.5)
    assert summary["account_gross_win_rate"] == pytest.approx(0.5)


def test_rejected_selection_artifact_is_frozen() -> None:
    result = diagnose._verify_rejected_selection()
    assert result["selection"]["rejected"] is True
    assert result["protocol"]["sealed_windows"] == [
        "test2024",
        "eval2025",
        "ytd2026",
    ]
