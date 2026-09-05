import numpy as np

from training import search_funding_flow_event_combinations as search


def test_event_hold_replaces_active_pulse_and_expires():
    raw = np.array([0, 1, 0, -0.5, 0, 0, 0], dtype=float)
    assert search.event_hold(raw, 3).tolist() == [0, 1, 1, -0.5, -0.5, -0.5, 0]


def test_design_keeps_report_and_execution_constraints_explicit():
    assert search.DESIGN["selection"].startswith("2021--2023")
    assert search.DESIGN["costs_per_side"] == [0.0, 0.0006, 0.001]
    assert "position overlap" in search.DESIGN["gates_removed"]
