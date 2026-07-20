from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


RESULT = Path("results/usdt_collateral_breadth_relay_support_2026-07-20.json")
CLOCK = Path("data/usdt_collateral_breadth_relay_clocks_2023.csv.gz")
EXPECTED_RESULT_SHA256 = (
    "fad1959d09e7261d2d03fadc5abdae5f9ee0b3a78339763e3f5b6566bc42a8e8"
)
EXPECTED_CLOCK_SHA256 = (
    "20b3ee9f82696222a3adbde0045dfde53e0e240e85162e463166aa8fe90b1a8f"
)
EXPECTED_MANIFEST_HASH = (
    "e73f3522196e6a03f78b3404b2809eb9e3e09a18bfcfeae5a543fa7f4a87e8c4"
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


def test_ucbr_support_rejection_is_frozen_and_outcome_blind() -> None:
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
    assert report["post_2023_ucbr_source_rows_loaded"] == 0
    assert report["real_event_incidence_opened"] is True
    assert report["support_passed"] is False
    assert report["advance_to_frozen_outcome_evaluator"] is False
    assert report["failed_checks"] == [
        "2023-09_minimum_events",
        "SDDR-12:primary_near_containment",
    ]


def test_ucbr_support_statistics_match_clock() -> None:
    report = json.loads(RESULT.read_text(encoding="utf-8"))
    clocks = pd.read_csv(CLOCK)
    primary = clocks.loc[clocks["control"].eq("primary")].copy()
    assert len(primary) == report["support"]["events"] == 31
    assert int(primary["side"].eq(1).sum()) == report["support"]["long"] == 16
    assert int(primary["side"].eq(-1).sum()) == report["support"]["short"] == 15
    assert report["support"]["month_counts"] == {
        "2023-08": 3,
        "2023-09": 4,
        "2023-10": 9,
        "2023-11": 7,
        "2023-12": 8,
    }
    assert report["novelty"]["SDDR-12:primary"][
        "max_bidirectional_near_share"
    ] == 0.5806451612903226
    assert report["checks"]["SDDR-12:primary_near_containment"] is False


def test_ucbr_clock_is_nonoverlapping_and_stale_is_exactly_delayed() -> None:
    clocks = pd.read_csv(CLOCK)
    for _, group in clocks.groupby("control"):
        ordered = group.sort_values("entry_time")
        entries = pd.to_datetime(ordered["entry_time"], utc=True).reset_index(drop=True)
        exits = pd.to_datetime(ordered["exit_time"], utc=True).reset_index(drop=True)
        assert (entries.iloc[1:].to_numpy() >= exits.iloc[:-1].to_numpy()).all()
    primary = clocks.loc[clocks["control"].eq("primary")].sort_values("entry_time")
    stale = clocks.loc[clocks["control"].eq("stale_1h")].sort_values("entry_time")
    assert len(primary) == len(stale) == 31
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
