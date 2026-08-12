import math

import numpy as np
import pandas as pd

from training import build_high_volatility_daily_flow_impact_capacity_reversal_support as s


def test_daily_geometry_uses_through_origin_impact_and_aggregate_flow():
    flow = np.linspace(-0.11, 0.13, 24)
    flow[flow == 0] = 0.01
    returns = 0.5 * flow
    frame = pd.DataFrame(
        {
            "hour_valid": True,
            "hour_flow": flow,
            "hour_return": returns,
            "quote_turnover": 100.0,
            "signed_taker_quote": flow * 100.0,
            "minute_squared_return": 0.01,
        }
    )
    result = s.daily_geometry(frame)
    assert result["source_valid"] is True
    assert math.isclose(result["impact_beta"], 0.5)
    assert math.isclose(result["aggregate_flow"], float(flow.mean()), abs_tol=1e-15)
    assert math.isclose(result["realized_variation"], math.sqrt(0.24))


def test_prior_rank_excludes_current_and_requires_frozen_floor():
    values = pd.Series(np.arange(121, dtype=float))
    ranked = s.prior_rank(values)
    assert ranked.iloc[:120].isna().all()
    assert ranked.iloc[120] == 1.0


def test_schema_is_outcome_blind_unique_and_controls_frozen():
    assert len(s.CLOCK_COLUMNS) == len(set(s.CLOCK_COLUMNS))
    forbidden = {"pnl", "funding", "execution_price", "gross9", "future_return"}
    assert not forbidden.intersection(column.lower() for column in (*s.PANEL_COLUMNS, *s.CLOCK_COLUMNS))
    assert s.CONTROLS == (
        "no_impact_tail", "no_variation_gate", "negative_beta_state",
        "one_day_stale_capacity", "direction_flip", "forced_long",
    )
