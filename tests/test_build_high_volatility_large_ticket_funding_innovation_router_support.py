from __future__ import annotations

import pandas as pd
import pytest

from training import build_high_volatility_large_ticket_funding_innovation_router_support as support


def test_prepare_funding_uses_actual_chronology() -> None:
    raw = pd.DataFrame({
        "funding_time": pd.to_datetime(["2023-07-01T00:00:00Z", "2023-07-01T08:00:00.008Z"], format="mixed"),
        "funding_rate": [0.0001, -0.0002], "source_rows": [1, 1],
    })
    frame = support.prepare_funding(raw)
    assert frame.funding_pair_valid.tolist() == [False, True]
    assert frame.funding_innovation.iloc[1] == pytest.approx(-0.0003)
    assert 479 <= frame.funding_gap_minutes.iloc[1] <= 481


def test_build_panel_routes_against_funding_innovation(monkeypatch) -> None:
    decisions = pd.to_datetime(["2023-07-01T08:00:00Z", "2023-07-01T09:00:00Z"])
    states = pd.DataFrame({
        "decision_time": decisions, "source_valid": [True, True],
        "large_ticket_clustering": [0.01, 0.02], "clustering_rank": [0.9, 0.9],
        "realized_variation": [0.03, 0.04], "variation_rank": [0.8, 0.8],
        "block_return": [0.01, -0.01], "final_hour_return": [0.01, -0.01],
    })
    monkeypatch.setattr(support.ticket, "build_states", lambda _: states)
    monkeypatch.setattr(support.ticket, "conditions", lambda *_: (pd.Series([True, True]), pd.Series([1, -1])))
    funding = pd.DataFrame({
        "funding_time": pd.to_datetime(["2023-07-01T00:00:00Z", "2023-07-01T08:00:00Z"]),
        "funding_rate": [-0.0001, 0.0002], "source_rows": [1, 1],
    })
    panel = support.build_panel((pd.DataFrame(), funding))
    assert panel.eligible.tolist() == [True, True]
    assert panel.routed_side.tolist() == [-1, -1]


def test_clock_preserves_fixed_timing(monkeypatch) -> None:
    decision = pd.Timestamp("2023-07-01T09:00:00Z")
    values = {column: 0.1 for column in support.PANEL_COLUMNS[5:]}
    panel = pd.DataFrame([{**values, "decision_time": decision, "feature_available_time": decision, "source_valid": True, "ticket_onset": True, "eligible": True, "routed_side": 1}])
    monkeypatch.setattr(support, "stage_for", lambda *_: "train")
    clock = support.build_clock(panel)
    assert clock.side.tolist() == [1]
    assert clock.entry_time.iloc[0] == decision + pd.Timedelta(minutes=5)
    assert clock.exit_time.iloc[0] == decision + pd.Timedelta(hours=8, minutes=5)
