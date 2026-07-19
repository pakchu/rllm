from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import pandas as pd

from training import build_premium_compression_breakout_relay_support as support


RESULT = Path("results/premium_compression_breakout_relay_support_2026-07-19.json")
EXPECTED_RESULT_SHA256 = (
    "de41852acb7987685d31a799eddf56a7e59afa756f5435ce46e054ea72f83857"
)
EXPECTED_CLOCK_SHA256 = (
    "659fc1b6b6e3a20e60031ed1d50f51c8c7d2836956f911f62ad13e4152740cda"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pcbr_support_artifact_rejects_without_opening_outcomes() -> None:
    assert _sha256(RESULT) == EXPECTED_RESULT_SHA256
    report = json.loads(RESULT.read_text(encoding="utf-8"))
    assert report["outcomes_opened"] is False
    assert report["outcome_sources_opened"] == []
    assert report["source"]["btc_execution_rows_loaded"] == 0
    assert report["source"]["funding_rows_loaded"] == 0
    assert report["support_passed"] is False
    assert report["advance_to_train_outcomes"] is False
    assert report["failed_checks"] == ["test_events", "CMSR-36_near"]
    assert report["support"]["test"]["events"] == 56
    assert report["support_checks"]["test_events"] is False
    assert report["novelty"]["CMSR-36"]["primary_events"] == 283
    assert report["novelty"]["CMSR-36"]["near_primary_events"] == 93
    assert report["novelty"]["CMSR-36"]["near_primary_share"] > 0.25
    assert report["novelty_checks"]["CMSR-36_near"] is False
    assert report["sealed_outcome_windows"] == [
        "train_2020_2022",
        "test_2023",
        "eval_2024_2026",
    ]
    implementation = Path(cast(str, support.__file__))
    assert report["implementation_sha256"] == _sha256(implementation)


def test_pcbr_clock_artifact_is_causal_nonoverlapping_and_hash_bound() -> None:
    report = json.loads(RESULT.read_text(encoding="utf-8"))
    clock = Path(report["clock"]["path"])
    assert _sha256(clock) == EXPECTED_CLOCK_SHA256
    assert report["clock"]["sha256"] == EXPECTED_CLOCK_SHA256
    frame = pd.read_csv(
        clock,
        compression="gzip",
        parse_dates=[
            "context_start_time",
            "decision_time",
            "feature_available_time",
            "entry_time",
            "exit_time",
        ],
    )
    assert len(frame) == 572
    assert set(frame["side"]) == {-1, 1}
    assert bool(frame["feature_available_time"].lt(frame["entry_time"]).all())
    assert bool(
        cast(pd.Series, frame["entry_time"] - frame["decision_time"])
        .eq(pd.Timedelta(minutes=10))
        .all()
    )
    assert bool(
        cast(pd.Series, frame["exit_time"] - frame["entry_time"])
        .eq(pd.Timedelta(hours=1))
        .all()
    )
    for _, split in frame.groupby("split", sort=False):
        ordered = split.sort_values("entry_time")
        assert bool(
            ordered["entry_time"]
            .iloc[1:]
            .reset_index(drop=True)
            .ge(ordered["exit_time"].iloc[:-1].reset_index(drop=True))
            .all()
        )
    forbidden = ("btc_open", "btc_close", "return", "pnl", "funding")
    assert not any(token in column for column in frame.columns for token in forbidden)
