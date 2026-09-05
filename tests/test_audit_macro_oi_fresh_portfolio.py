import numpy as np
import pandas as pd

from training import audit_macro_oi_fresh_portfolio as audit


def test_target_updates_are_causal_and_forward_filled():
    dates = pd.Series(pd.date_range("2026-01-01", periods=5, freq="5min"))
    actual = audit.target_from_updates(dates, dates.iloc[[1, 3]], [0.5, -0.25])
    assert np.allclose(actual, [0, 0.5, 0.5, -0.25, -0.25])


def test_portfolio_grid_is_predeclared_and_not_optimized():
    assert audit.WEIGHTS == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert audit.DESIGN["selection"].startswith("none")
    assert audit.DESIGN["risk"].endswith("1x")
