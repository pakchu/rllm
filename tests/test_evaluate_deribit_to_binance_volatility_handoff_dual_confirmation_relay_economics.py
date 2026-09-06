import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from training import evaluate_deribit_to_binance_volatility_handoff_dual_confirmation_relay_economics as economics


def test_exact_offset_funding_posts_to_containing_bar():
    start = pd.Timestamp("2023-07-01T00:00:00Z")
    end = start + pd.Timedelta(minutes=10)
    dates = pd.date_range(start, end, freq="5min", inclusive="both")
    market = pd.DataFrame({"date": dates, "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0})
    funding = pd.DataFrame({"date": [start + pd.Timedelta(milliseconds=5), end], "funding_rate": [0.01, 0.5], "mark_price": [100.0, 100.0]})
    clock = pd.DataFrame({"entry_time": [start], "exit_time": [end], "side": [1]})
    result = economics.engine.simulate(clock, market, funding, start, end, cost=0.0)
    assert math.isclose(result["final_equity"], 0.995, abs_tol=1e-12)


def test_frozen_accounting_and_stage_contract():
    assert economics.LEVERAGE == 0.5 and economics.BASE_COST == 0.0006 and economics.STRESS_COST == 0.001
    assert economics.STAGES["final"][2] == "2026-08-01T00:00:00Z"
    source = Path(economics.__file__).read_text()
    assert "full calendar including idle time" in source
    assert "global peak, every held favorable then adverse" in source


def test_outcome_blind_evaluator_freeze_is_bound():
    assert hashlib.sha256(economics.FREEZE.read_bytes()).hexdigest() == "7e5382c817f3b7e94f7c2daffb213bd0ecdb6e7605d28bebe363d58f7737570c"
    data = json.loads(economics.FREEZE.read_text())
    core = {key: value for key, value in data.items() if key != "manifest_hash"}
    assert data["manifest_hash"] == economics.canonical_hash(core)
    assert data["outcomes_opened"] is False
    assert data["evaluator"]["sha256"] == economics.sha256(Path(economics.__file__))
    novelty, freeze = economics.verify("train")
    assert novelty["advance_to_economic_outcomes"] is True
    assert freeze["outcomes_opened"] is False
