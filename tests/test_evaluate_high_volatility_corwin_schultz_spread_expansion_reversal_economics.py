import math
from pathlib import Path

import pandas as pd

from training import evaluate_high_volatility_corwin_schultz_spread_expansion_reversal_economics as economics


def test_exact_offset_funding_posts_to_containing_bar():
    start = pd.Timestamp("2023-07-01T00:00:00Z")
    end = start + pd.Timedelta(minutes=10)
    dates = pd.date_range(start, end, freq="5min", inclusive="both")
    market = pd.DataFrame(
        {"date": dates, "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0}
    )
    funding = pd.DataFrame(
        {
            "date": [start + pd.Timedelta(milliseconds=5), end],
            "funding_rate": [0.01, 0.5],
            "mark_price": [100.0, 100.0],
        }
    )
    clock = pd.DataFrame({"entry_time": [start], "exit_time": [end], "side": [1]})
    report = economics.engine.simulate(clock, market, funding, start, end, cost=0.0)
    assert math.isclose(report["final_equity"], 0.995, abs_tol=1e-12)


def test_frozen_accounting_and_stage_contract():
    assert economics.POLICY_ID == "HVCSER-12"
    assert economics.LEVERAGE == 0.5
    assert economics.BASE_COST == 0.0006
    assert economics.STRESS_COST == 0.001
    assert economics.STAGES["final"][2] == "2026-08-01T00:00:00Z"
    assert economics.CONTROLS == ("no_spread_expansion_gate", "no_volatility_gate", "raw_positive_expansion", "one_day_stale_expansion", "direction_flip")
    source = Path(economics.__file__).read_text()
    assert "full calendar including idle time" in source
    assert "global peak, every held favorable then adverse" in source
