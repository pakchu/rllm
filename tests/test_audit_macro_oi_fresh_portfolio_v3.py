import numpy as np

from training import audit_macro_oi_fresh_portfolio_v3 as audit


def test_event_simulator_does_not_rebalance_unchanged_target():
    blocks = {
        "open": np.array([100.0, 101.0, 102.0]),
        "end": np.array([101.0, 102.0, 103.0]),
        "high": np.array([101.5, 102.5, 103.5]),
        "low": np.array([99.5, 100.5, 101.5]),
        "funding": np.zeros(3),
    }
    stats = audit.simulate_events(
        blocks,
        np.ones(3),
        np.array([True, False, False]),
        cost=0.0,
        start="2026-01-01",
        end="2027-01-01",
    )
    assert stats["rebalance_orders"].item() == 1
    assert stats["entry_episodes"].item() == 1
    assert np.isclose(stats["return_pct"].item(), 3.0)


def test_v3_changes_only_native_rebalance_events():
    assert audit.DESIGN["version"] == 3
    assert "native event" in audit.DESIGN["correction"]
    assert "weights" in audit.DESIGN["unchanged"]
