import numpy as np

from training import audit_macro_oi_fresh_portfolio_v2 as audit


def test_corrected_row_uses_actual_window_and_five_minute_frequency():
    stats = {
        "equity": np.array([1.01]),
        "return_pct": np.array([1.0]),
        "cagr_pct": np.array([-999.0]),
        "mdd_pct": np.array([2.0]),
        "calmar": np.array([-999.0]),
        "sharpe": np.array([-999.0]),
        "entry_episodes": np.array([1]),
        "rebalance_orders": np.array([2]),
        "turnover": np.array([2.0]),
        "fees_pct_initial": np.array([0.1]),
        "funding_pct_initial": np.array([0.0]),
        "returns": np.array([[0.01], [-0.005], [0.002]]),
    }
    row = audit.corrected_row(stats, 0, "2026-01-01", "2027-01-01")
    assert 0.99 < row["cagr_pct"] < 1.02
    assert 0.49 < row["calmar"] < 0.51
    assert row["sharpe"] != -999.0


def test_v2_changes_only_reporting_units_in_design():
    assert audit.DESIGN["version"] == 2
    assert "annualize" in audit.DESIGN["correction"]
    assert "targets" in audit.DESIGN["unchanged"]
