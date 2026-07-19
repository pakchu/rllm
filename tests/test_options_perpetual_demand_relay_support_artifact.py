from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import pandas as pd

from training import build_options_perpetual_demand_relay_support as support


RESULT = Path("results/options_perpetual_demand_relay_support_2026-07-19.json")
EXPECTED_RESULT_SHA256 = (
    "d8a82c072c45a2e965b8e4d05383aa3cb7f39d92728aef54ccd51ad54a02b9f3"
)
EXPECTED_CLOCK_SHA256 = (
    "ceb79b206c3e1f6bf78b02cd2ace9a94f875ce930a704cc6e7a5a8b255021b99"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_opdr_support_artifact_rejects_without_opening_outcomes() -> None:
    assert _sha256(RESULT) == EXPECTED_RESULT_SHA256
    report = json.loads(RESULT.read_text(encoding="utf-8"))
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert report["manifest_hash"] == support._canonical_hash(core)
    assert report["outcomes_opened"] is False
    assert report["outcome_sources_opened"] == []
    assert report["sources"]["btc_execution_rows_loaded"] == 0
    assert report["sources"]["funding_rows_loaded"] == 0
    assert report["support_passed"] is False
    assert report["advance_to_train_outcomes"] is False
    assert report["failed_checks"] == [
        "test_month_concentration",
        "eval_events",
        "eval_month_concentration",
        "final_month_concentration",
    ]
    assert report["support"]["train"]["events"] == 35
    assert report["support"]["test"]["events"] == 56
    assert report["support"]["eval"]["events"] == 25
    assert report["support"]["final"]["events"] == 29
    assert report["support"]["eval"]["max_month_share"] == 0.8
    assert report["support"]["final"]["max_month_share"] > 0.68
    assert all(report["novelty_checks"].values())
    assert report["sealed_outcome_windows"] == [
        "train_2023_h2",
        "test_2024",
        "eval_2025",
        "final_2026_h1",
    ]
    implementation = Path(cast(str, support.__file__))
    assert report["implementation_sha256"] == _sha256(implementation)


def test_opdr_clock_is_causal_globally_nonoverlapping_and_hash_bound() -> None:
    report = json.loads(RESULT.read_text(encoding="utf-8"))
    clock = Path(report["clock"]["path"])
    assert _sha256(clock) == EXPECTED_CLOCK_SHA256
    assert report["clock"]["sha256"] == EXPECTED_CLOCK_SHA256
    frame = pd.read_csv(
        clock,
        compression="gzip",
        parse_dates=[
            "decision_time",
            "feature_available_time",
            "entry_time",
            "exit_time",
        ],
    )
    assert len(frame) == 145
    assert set(frame["side"]) == {-1, 1}
    assert bool(frame["feature_available_time"].lt(frame["entry_time"]).all())
    assert bool(
        cast(pd.Series, frame["entry_time"] - frame["decision_time"])
        .eq(pd.Timedelta(minutes=5))
        .all()
    )
    assert bool(
        cast(pd.Series, frame["exit_time"] - frame["entry_time"])
        .eq(pd.Timedelta(hours=24))
        .all()
    )
    ordered = frame.sort_values("entry_time")
    assert bool(
        ordered["entry_time"]
        .iloc[1:]
        .reset_index(drop=True)
        .ge(ordered["exit_time"].iloc[:-1].reset_index(drop=True))
        .all()
    )
    forbidden = ("btc_open", "btc_close", "return", "pnl", "funding")
    assert not any(token in column for column in frame.columns for token in forbidden)
