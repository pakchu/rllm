from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path

import pandas as pd

from training import (
    evaluate_options_led_volatility_expansion_premium_relay_economics_v4 as v4,
)
from training import (
    evaluate_options_led_volatility_expansion_premium_relay_economics_v5 as v5,
)


SCRIPT = Path(
    "training/evaluate_options_led_volatility_expansion_premium_relay_economics_v5.py"
)


def test_millisecond_late_funding_event_posts_to_its_containing_bar() -> None:
    start = pd.Timestamp("2023-07-01T00:00:00Z")
    exit_time = start + pd.Timedelta(minutes=10)
    dates = pd.date_range(start, exit_time, freq="5min", inclusive="both")
    market = pd.DataFrame(
        {
            "date": dates,
            "open": [100.0] * 3,
            "high": [100.0] * 3,
            "low": [100.0] * 3,
            "close": [100.0] * 3,
        }
    )
    funding = pd.DataFrame(
        {
            "date": [start + pd.Timedelta(milliseconds=5), exit_time],
            "funding_rate": [0.01, 0.50],
            "mark_price": [100.0, 100.0],
        }
    )
    clock = pd.DataFrame(
        {"entry_time": [start], "exit_time": [exit_time], "side": [1]}
    )

    result = v5.simulate(clock, market, funding, start, exit_time, cost=0.0)

    assert math.isclose(result["final_equity"], 0.995, abs_tol=1e-12)
    assert math.isclose(
        result["trade_rows"][0]["funding_cash_over_pre_equity"],
        -0.005,
        abs_tol=1e-12,
    )


def test_five_millisecond_reporting_offset_is_not_a_funding_gap() -> None:
    start = pd.Timestamp("2023-07-01T00:00:00Z")
    end = start + pd.Timedelta(hours=16)
    funding = pd.DataFrame(
        {
            "date": [start, start + pd.Timedelta(hours=8, milliseconds=5)],
            "funding_rate": [0.0001, 0.0001],
            "mark_price": [100.0, 100.0],
        }
    )

    v5.validate_funding(funding, start, end)


def test_v5_preserves_frozen_candidate_constants_and_stream_fix() -> None:
    assert v5.BASE_COST == v4.BASE_COST
    assert v5.STRESS_COST == v4.STRESS_COST
    assert v5.LEVERAGE == v4.LEVERAGE
    assert v5.STAGES == v4.STAGES


def test_direct_cli_preflight_reaches_verifier_without_opening_outcomes() -> None:
    completed = subprocess.run(
        [
            str(Path(".venv/bin/python")),
            "-B",
            str(SCRIPT),
            "--stage",
            "train",
            "--verify-only",
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert json.loads(completed.stdout)["outcomes_opened"] is False
