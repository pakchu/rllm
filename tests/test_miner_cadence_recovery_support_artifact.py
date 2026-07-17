from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from training import preregister_miner_cadence_recovery as prereg


RESULT = Path("results/miner_cadence_recovery_support_2026-07-17.json")
CLOCK = Path("results/miner_cadence_recovery_clock_2026-07-17.csv")
EXPECTED_RESULT_SHA256 = (
    "817081407607a3f495c93c31c31d2ce18f8c5652f3b8a83a0811bc082ff62df5"
)
EXPECTED_CLOCK_SHA256 = (
    "2535244889b046ff00c369ee854973a91c23429dff82a6dd3c1a293a01352b0b"
)
EXPECTED_RESULT_HASH = (
    "8ab764593d35e720d279d5a5ea1449ad8b9ab3c08de4c1d6a5751d14e4ab0419"
)


def test_support_artifacts_are_frozen_before_market_outcomes() -> None:
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == EXPECTED_RESULT_SHA256
    assert hashlib.sha256(CLOCK.read_bytes()).hexdigest() == EXPECTED_CLOCK_SHA256
    payload = json.loads(RESULT.read_text())
    core = {
        key: value for key, value in payload.items() if key not in {"result_hash", "created_at"}
    }
    assert prereg.canonical_hash(core) == payload["result_hash"]
    assert payload["result_hash"] == EXPECTED_RESULT_HASH
    assert payload["outcomes_opened"] is False
    assert payload["source"]["market_or_funding_rows_loaded"] == 0
    assert payload["source"]["last_observation"].startswith("2023-12-31")
    assert payload["sealed"] == [
        "all_post_entry_market_outcomes",
        "2024",
        "2025",
        "2026_ytd",
    ]


def test_support_gate_and_clock_distribution_are_frozen() -> None:
    payload = json.loads(RESULT.read_text())
    support = payload["support_gate"]
    assert support["passed"] is True
    assert support["counts"] == {
        "train_2021_2022": 38,
        "train_2021": 14,
        "train_2022": 24,
        "selection_2023": 27,
        "selection_2023_h1": 11,
        "selection_2023_h2": 16,
    }
    assert support["maximum_single_month_share"] <= 0.15
    assert payload["feature_support"] == {
        "finite_hash_z_rows": 1698,
        "recovery_cross_rows": 351,
        "eligible_events_before_nonoverlap": 104,
        "accepted_nonoverlap_events": 65,
    }


def test_clock_is_causal_nonoverlapping_and_contains_no_outcomes() -> None:
    clock = pd.read_csv(
        CLOCK,
        parse_dates=[
            "observation_date",
            "available_at",
            "earliest_tradable_open",
            "entry_date",
            "exit_date",
        ],
    )
    assert len(clock) == 65
    assert clock["entry_date"].max() < pd.Timestamp("2024-01-01")
    assert clock["available_at"].gt(clock["observation_date"]).all()
    assert clock["earliest_tradable_open"].ge(clock["available_at"]).all()
    assert clock["entry_date"].eq(
        clock["earliest_tradable_open"] + pd.Timedelta(minutes=5)
    ).all()
    assert clock["exit_date"].eq(clock["entry_date"] + pd.Timedelta(days=7)).all()
    assert clock["entry_date"].iloc[1:].reset_index(drop=True).ge(
        clock["exit_date"].iloc[:-1].reset_index(drop=True)
    ).all()
    forbidden = {
        "open",
        "high",
        "low",
        "close",
        "return",
        "pnl",
        "funding",
        "premium",
        "open_interest",
    }
    assert forbidden.isdisjoint({column.lower() for column in clock.columns})
