from __future__ import annotations

import pandas as pd

from training import build_high_volatility_oi_large_ticket_joint_sponsorship_support as support


def _raw_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    decision = pd.to_datetime(["2023-07-01T00:00:00Z", "2023-07-01T03:00:00Z"])
    oi = pd.DataFrame({
        "decision_time": decision,
        "oi_source_valid": [True, True],
        "completed_return_3h": [0.01, 0.02],
        "oi_variation": [0.03, 0.04],
        "gross_oi_activity": [0.1, 0.2],
        "coactivity": [0.5, 0.6],
        "gross_oi_activity_rank": [0.8, 0.8],
        "coactivity_rank": [0.8, 0.8],
        "oi_variation_rank": [0.8, 0.8],
    })
    ticket = pd.DataFrame({
        "decision_time": decision,
        "ticket_source_valid": [True, True],
        "large_ticket_clustering": [0.01, 0.02],
        "clustering_rank": [0.9, 0.9],
        "ticket_variation": [0.03, 0.04],
        "ticket_variation_rank": [0.8, 0.8],
        "block_return_6h": [0.01, 0.02],
        "final_hour_return": [0.01, 0.01],
    })
    return oi, ticket


def test_joint_onset_requires_prior_exact_valid_ineligible(monkeypatch) -> None:
    oi, ticket = _raw_frames()
    monkeypatch.setattr(support.oi_price, "build_features", lambda *_: oi.rename(columns={
        "oi_source_valid": "source_valid", "completed_return_3h": "completed_return",
        "oi_variation": "realized_variation", "oi_variation_rank": "variation_rank",
    }))
    monkeypatch.setattr(support.ticket, "build_states", lambda *_: ticket.rename(columns={
        "ticket_source_valid": "source_valid", "ticket_variation": "realized_variation",
        "ticket_variation_rank": "variation_rank", "block_return_6h": "block_return",
    }))
    oi.loc[0, "coactivity_rank"] = 0.5
    panel = support.build_panel((pd.DataFrame(), pd.DataFrame(), pd.DataFrame()))
    assert panel["joint_eligible"].tolist() == [False, True]
    assert panel["joint_onset"].tolist() == [False, True]


def test_direction_disagreement_blocks_joint_state(monkeypatch) -> None:
    oi, ticket = _raw_frames()
    ticket.loc[1, "final_hour_return"] = -0.01
    monkeypatch.setattr(support.oi_price, "build_features", lambda *_: oi.rename(columns={
        "oi_source_valid": "source_valid", "completed_return_3h": "completed_return",
        "oi_variation": "realized_variation", "oi_variation_rank": "variation_rank",
    }))
    monkeypatch.setattr(support.ticket, "build_states", lambda *_: ticket.rename(columns={
        "ticket_source_valid": "source_valid", "ticket_variation": "realized_variation",
        "ticket_variation_rank": "variation_rank", "block_return_6h": "block_return",
    }))
    panel = support.build_panel((pd.DataFrame(), pd.DataFrame(), pd.DataFrame()))
    assert panel["joint_eligible"].tolist() == [True, False]


def test_clock_uses_three_hour_side_and_fixed_timing(monkeypatch) -> None:
    decision = pd.Timestamp("2023-07-01T03:00:00Z")
    values = {column: 0.1 for column in support.PANEL_COLUMNS[5:]}
    panel = pd.DataFrame([{**values, "decision_time": decision, "feature_available_time": decision, "source_valid": True, "joint_eligible": True, "joint_onset": True, "completed_return_3h": -0.1}])
    monkeypatch.setattr(support, "stage_for", lambda *_: "train")
    clock = support.build_clock(panel)
    assert clock.side.tolist() == [-1]
    assert clock.entry_time.iloc[0] == decision + pd.Timedelta(minutes=5)
    assert clock.exit_time.iloc[0] == decision + pd.Timedelta(hours=8, minutes=5)
