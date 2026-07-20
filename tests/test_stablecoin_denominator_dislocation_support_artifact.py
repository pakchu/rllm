from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


RESULT = Path("results/stablecoin_denominator_dislocation_support_2026-07-20.json")
CLOCK = Path("data/stablecoin_denominator_dislocation_clocks_2023.csv.gz")
EXPECTED_RESULT_SHA256 = (
    "1d7e8561963d903c5963bbd081c5cf0c9926dc9221f9a09d23fb565ab27f7bea"
)
EXPECTED_CLOCK_SHA256 = (
    "eaf2d6c187af9855e76474d2951fcdc12267174980a72649b73d068982ca8c69"
)
EXPECTED_MANIFEST_HASH = (
    "7292e1c1f8b1979ca95eabd2ada14cc2009f970a6e21d90a70b4c016d8de0302"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_sddr_support_rejection_is_frozen_and_outcome_blind() -> None:
    assert _sha256(RESULT) == EXPECTED_RESULT_SHA256
    assert _sha256(CLOCK) == EXPECTED_CLOCK_SHA256
    report = json.loads(RESULT.read_text(encoding="utf-8"))
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert report["manifest_hash"] == EXPECTED_MANIFEST_HASH
    assert report["manifest_hash"] == _canonical_hash(core)
    assert report["outcomes_opened"] is False
    assert report["outcome_sources_opened"] == []
    assert report["btc_execution_rows_loaded"] == 0
    assert report["funding_rows_loaded"] == 0
    assert report["post_2023_sddr_source_rows_loaded"] == 0
    assert report["support_passed"] is False
    assert report["advance_to_frozen_outcome_evaluator"] is False
    assert report["failed_checks"] == ["no_usdt_lag_near_containment"]


def test_sddr_support_statistics_match_the_frozen_clock() -> None:
    report = json.loads(RESULT.read_text(encoding="utf-8"))
    clocks = pd.read_csv(CLOCK)
    primary = clocks.loc[clocks["control"].eq("primary")].copy()
    assert len(primary) == report["support"]["events"] == 78
    assert int(primary["side"].eq(1).sum()) == report["support"]["long"] == 40
    assert int(primary["side"].eq(-1).sum()) == report["support"]["short"] == 38
    assert report["support"]["month_counts"] == {
        "2023-09": 13,
        "2023-10": 27,
        "2023-11": 25,
        "2023-12": 13,
    }
    assert report["novelty"]["no_usdt_lag"][
        "max_bidirectional_near_share"
    ] == 0.44871794871794873
    assert report["checks"]["no_usdt_lag_near_containment"] is False


def test_sddr_clock_is_nonoverlapping_and_contains_no_outcome() -> None:
    clocks = pd.read_csv(CLOCK)
    forbidden = {
        "price",
        "open",
        "high",
        "low",
        "close",
        "return",
        "label",
        "pnl",
        "funding",
        "cagr",
        "drawdown",
    }
    assert forbidden.isdisjoint(column.lower() for column in clocks.columns)
    for _, group in clocks.groupby("control"):
        ordered = group.sort_values("entry_time")
        entries = pd.to_datetime(ordered["entry_time"], utc=True).reset_index(drop=True)
        exits = pd.to_datetime(ordered["exit_time"], utc=True).reset_index(drop=True)
        assert (entries.iloc[1:] >= exits.iloc[:-1].to_numpy()).all()

    primary = clocks.loc[clocks["control"].eq("primary")].sort_values("entry_time")
    stale = clocks.loc[clocks["control"].eq("stale_1h")].sort_values("entry_time")
    assert len(primary) == len(stale) == 78
    primary_entries = pd.to_datetime(primary["entry_time"], utc=True).reset_index(
        drop=True
    )
    stale_entries = pd.to_datetime(stale["entry_time"], utc=True).reset_index(
        drop=True
    )
    assert (stale_entries == primary_entries + pd.Timedelta("1h")).all()
    assert stale["side"].reset_index(drop=True).equals(
        primary["side"].reset_index(drop=True)
    )
