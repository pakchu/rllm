from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


SUPPORT = Path(
    "results/cross_collateral_book_validated_flow_rejection_"
    "support_2026-07-18.json"
)
CLOCK = Path(
    "results/cross_collateral_book_validated_flow_rejection_"
    "event_clock_2026-07-18.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_support_artifact_is_outcome_blind_and_passing() -> None:
    support = json.loads(SUPPORT.read_text())
    assert support["all_support_gates_pass"] is True
    assert support["protocol"]["evidence_boundary"]["post_entry_outcomes_opened"] is False
    assert support["support_selection"]["post_entry_outcomes_used"] is False
    source = support["frozen_sources"]
    assert source["market_columns_loaded"] == [
        "date",
        "open",
        "close",
        "quote_asset_volume",
        "taker_buy_quote",
    ]
    assert source["high_or_low_loaded"] is False
    assert source["entry_or_later_ohlc_loaded"] is False
    assert source["post_entry_return_funding_pnl_or_equity_loaded"] is False


def test_incidence_selection_and_independence_are_frozen() -> None:
    support = json.loads(SUPPORT.read_text())
    cells = support["support_selection"]["cells"]
    assert len(cells) == 12
    selected = support["support_selection"]["selected_cell"]
    assert selected["flow_quantile"] == 0.75
    assert selected["defense_threshold"] == 0.25
    assert selected["support"] == {
        "by_quarter": {"q1": 20, "q2": 37, "q3": 46, "q4": 37},
        "h1": 57,
        "h2": 83,
        "long_share": 75 / 140,
        "longs": 75,
        "maximum_quarter_share": 46 / 140,
        "nonoverlap_total": 140,
        "passes": True,
        "short_share": 65 / 140,
        "shorts": 65,
    }
    overlap = support["outcome_blind_independence"]
    assert overlap["pdf10"]["prior_events"] == 591
    assert max(item["jaccard"] for item in overlap.values()) < 0.04
    assert max(item["new_clock_containment"] for item in overlap.values()) < 0.09


def test_event_clock_has_no_outcome_and_is_quarter_contained() -> None:
    support = json.loads(SUPPORT.read_text())
    clock = json.loads(CLOCK.read_text())
    assert _sha256(SUPPORT) == "5c2793a504b63c0b928b5a75407d0099e03a6c30f41cc0bce768837fbed3aa93"
    assert _sha256(CLOCK) == "b95e49600611c21a090efb43d9949607384c0a39188da4a5a069bd99bd152631"
    assert clock["post_entry_outcomes_opened"] is False
    assert clock["entry_or_later_ohlc_loaded"] is False
    assert clock["event_count"] == 140
    assert clock["event_clock_sha256"] == support["event_clock_sha256"]
    forbidden = {"return", "pnl", "funding", "equity", "cagr", "mdd", "high", "low"}
    for event in clock["events"]:
        assert forbidden.isdisjoint(key.lower() for key in event)
        entry = pd.Timestamp(event["entry_date"])
        exit_ = pd.Timestamp(event["exit_date"])
        assert entry.quarter == exit_.quarter
        assert exit_ - entry == pd.Timedelta(hours=6)
