from __future__ import annotations

import pandas as pd

from training import build_high_volatility_large_ticket_to_oi_handoff_support as support


def _raw_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    decision = pd.to_datetime(["2023-07-01T00:00:00Z", "2023-07-01T03:00:00Z", "2023-07-01T06:00:00Z"])
    oi = pd.DataFrame({
        "decision_time": decision,
        "oi_source_valid": [True, True, True],
        "completed_return_3h": [0.01, 0.02, 0.03],
        "oi_variation": [0.03, 0.04, 0.05],
        "gross_oi_activity": [0.1, 0.2, 0.3],
        "coactivity": [0.5, 0.6, 0.7],
        "gross_oi_activity_rank": [0.8, 0.8, 0.8],
        "coactivity_rank": [0.8, 0.5, 0.8],
        "oi_variation_rank": [0.8, 0.8, 0.8],
    })
    ticket = pd.DataFrame({
        "decision_time": decision,
        "ticket_source_valid": [True, True, True],
        "large_ticket_clustering": [0.01, 0.02, 0.03],
        "clustering_rank": [0.9, 0.9, 0.9],
        "ticket_variation": [0.03, 0.04, 0.05],
        "ticket_variation_rank": [0.8, 0.8, 0.8],
        "block_return_6h": [0.01, 0.02, 0.03],
        "final_hour_return": [0.01, 0.01, 0.02],
    })
    return oi, ticket


def test_handoff_onset_requires_prior_exact_valid_ineligible(monkeypatch) -> None:
    oi, ticket = _raw_frames()
    monkeypatch.setattr(support.oi_price, "build_features", lambda *_: oi.rename(columns={
        "oi_source_valid": "source_valid", "completed_return_3h": "completed_return",
        "oi_variation": "realized_variation", "oi_variation_rank": "variation_rank",
    }))
    monkeypatch.setattr(support.ticket, "build_states", lambda *_: ticket.rename(columns={
        "ticket_source_valid": "source_valid", "ticket_variation": "realized_variation",
        "ticket_variation_rank": "variation_rank", "block_return_6h": "block_return",
    }))
    panel = support.build_panel((pd.DataFrame(), pd.DataFrame(), pd.DataFrame()))
    assert panel["handoff_eligible"].tolist() == [False, False, True]
    assert panel["handoff_onset"].tolist() == [False, False, True]


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
    assert panel["handoff_eligible"].tolist() == [False, False, False]


def test_clock_uses_three_hour_side_and_fixed_timing(monkeypatch) -> None:
    decision = pd.Timestamp("2023-07-01T03:00:00Z")
    values = {column: 0.1 for column in support.PANEL_COLUMNS[5:]}
    panel = pd.DataFrame([{**values, "decision_time": decision, "feature_available_time": decision, "source_valid": True, "handoff_eligible": True, "handoff_onset": True, "completed_return_3h": -0.1}])
    monkeypatch.setattr(support, "stage_for", lambda *_: "train")
    clock = support.build_clock(panel)
    assert clock.side.tolist() == [-1]
    assert clock.entry_time.iloc[0] == decision + pd.Timedelta(minutes=5)
    assert clock.exit_time.iloc[0] == decision + pd.Timedelta(hours=8, minutes=5)
