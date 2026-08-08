import math
from pathlib import Path

import pandas as pd

from training import evaluate_daily_taker_flow_acceleration_relay_economics as economics


def test_exact_offset_funding_posts_to_containing_bar():
    start = pd.Timestamp("2023-07-01T00:00:00Z")
    end = start + pd.Timedelta(minutes=10)
    dates = pd.date_range(start, end, freq="5min", inclusive="both")
    market = pd.DataFrame({"date": dates, "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0})
    funding = pd.DataFrame({"date": [start + pd.Timedelta(milliseconds=5), end], "funding_rate": [0.01, 0.5], "mark_price": [100.0, 100.0]})
    clock = pd.DataFrame({"entry_time": [start], "exit_time": [end], "side": [1]})
    assert math.isclose(economics.engine.simulate(clock, market, funding, start, end, cost=0.0)["final_equity"], 0.995, abs_tol=1e-12)


def test_frozen_accounting_and_stage_contract():
    assert economics.POLICY_ID == "DTFAR-12"
    assert economics.LEVERAGE == 0.5
    assert economics.BASE_COST == 0.0006
    assert economics.STRESS_COST == 0.001
    assert economics.STAGES["final"][2] == "2026-08-01T00:00:00Z"
    assert economics.CONTROLS == ("no_volatility_gate", "no_flow_tail", "flow_level", "one_day_stale_acceleration", "direction_flip")
    source = Path(economics.__file__).read_text()
    assert "full calendar including idle time" in source
    assert "global peak, every held favorable then adverse" in source


def test_outcome_blind_evaluator_freeze_is_bound():
    import hashlib
    import json

    assert hashlib.sha256(economics.FREEZE.read_bytes()).hexdigest() == "3441364b299035f4ca074e4ac8f813cb7fd2fd3ef9ecd02f2a806cc4283248e8"
    payload = json.loads(economics.FREEZE.read_text())
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == economics.canonical_hash(core)
    assert payload["outcomes_opened"] is False
    assert payload["evaluator"]["sha256"] == economics.sha256(Path(economics.__file__))
    novelty, freeze = economics.verify("train")
    assert novelty["advance_to_economic_outcomes"] is True
    assert freeze["outcomes_opened"] is False
