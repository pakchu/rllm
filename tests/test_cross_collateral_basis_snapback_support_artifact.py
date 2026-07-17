from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from training.preregister_cross_collateral_basis_snapback import canonical_hash


SUPPORT = Path("results/cross_collateral_basis_snapback_support_2026-07-17.json")
ANCHOR = Path("results/cross_collateral_basis_snapback_live_anchor_clock_2023.json")


def test_support_artifact_freezes_outcome_blind_threshold_selection() -> None:
    report = json.loads(SUPPORT.read_text())
    body = {key: value for key, value in report.items() if key not in {"as_of", "content_hash"}}
    assert canonical_hash(body) == report["content_hash"]
    assert report["content_hash"] == (
        "29a002968e784604512e83407facb0d53a0a3c6536d1038af4f9d44adf51d4f1"
    )
    assert report["selected_threshold"] == 2.0
    rows = {row["threshold"]: row for row in report["threshold_support"]}
    assert rows[2.0]["support_passed"] is True
    assert rows[2.0]["selection_period_counts"]["events"] == 143
    assert rows[2.5]["support_passed"] is False
    assert rows[2.0]["2023_diagnostic_counts_not_used_for_selection"]["events"] == 58
    assert report["forbidden_ccbs_columns_loaded"] == []
    assert "um_open" not in report["loaded_ccbs_columns"]
    assert "um_high" not in report["loaded_ccbs_columns"]
    assert "um_low" not in report["loaded_ccbs_columns"]


def test_selected_event_clock_is_causal_reserved_and_roll_safe() -> None:
    report = json.loads(SUPPORT.read_text())
    events = pd.DataFrame(report["selected_events_2021_2023"])
    for column in (
        "open_time",
        "available_time",
        "entry_time",
        "maximum_exit_time",
        "delivery_time",
    ):
        events[column] = pd.to_datetime(events[column], utc=True)
    assert len(events) == 201
    assert events["available_time"].eq(events["open_time"] + pd.Timedelta(minutes=5)).all()
    assert events["entry_time"].eq(events["open_time"] + pd.Timedelta(minutes=10)).all()
    assert events["maximum_exit_time"].eq(events["entry_time"] + pd.Timedelta(hours=12)).all()
    assert events["maximum_exit_time"].lt(events["delivery_time"]).all()
    assert events["entry_time"].diff().dropna().ge(pd.Timedelta(hours=12)).all()
    forbidden = {"um_open", "um_high", "um_low", "cm_open", "cm_high", "cm_low", "pnl"}
    assert forbidden.isdisjoint(events.columns)


def test_anchor_clock_and_2023_overlap_are_frozen() -> None:
    anchor = json.loads(ANCHOR.read_text())
    body = {key: value for key, value in anchor.items() if key != "content_hash"}
    assert canonical_hash(body) == anchor["content_hash"]
    assert anchor["content_hash"] == (
        "0d87df3cde14c542964ee6b2881d9bffac452fd10be43f952fa677a43b6ef169"
    )
    assert anchor["counts_by_sleeve"] == {
        "cand_rex_veto_7": 69,
        "new_long_minimal_funding_premium": 33,
        "oi_upbit_ratio288_low": 34,
    }
    assert anchor["unique_entry_times"] == 135

    report = json.loads(SUPPORT.read_text())
    overlap = report["2023_entry_clock_orthogonality"]
    assert overlap["ccbs_entries"] == 58
    assert overlap["exact_5m_intersections"] == 0
    assert overlap["exact_5m_overlap_share_of_ccbs"] == 0.0
    assert overlap["entry_day_jaccard"] == 18 / 136
    assert report["entry_clock_orthogonality_passed"] is True
    assert report["disposition"] == "PASS_SUPPORT_OPEN_2023_PNL"
