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
    assert selected["defense_threshold"] == 0.50
    assert selected["support"] == {
        "by_quarter": {"q1": 22, "q2": 44, "q3": 43, "q4": 35},
        "h1": 66,
        "h2": 78,
        "long_share": 74 / 144,
        "longs": 74,
        "maximum_quarter_share": 44 / 144,
        "nonoverlap_total": 144,
        "passes": True,
        "short_share": 70 / 144,
        "shorts": 70,
    }
    overlap = support["outcome_blind_independence"]
    assert overlap["pdf10"]["prior_events"] == 591
    assert max(item["jaccard"] for item in overlap.values()) < 0.04
    assert max(item["new_clock_containment"] for item in overlap.values()) < 0.07


def test_event_clock_has_no_outcome_and_is_quarter_contained() -> None:
    support = json.loads(SUPPORT.read_text())
    clock = json.loads(CLOCK.read_text())
    assert _sha256(SUPPORT) == "048a8723494a91b082bdd07d466e1741a13a974c3c3c25c8ec81e081f27cc444"
    assert _sha256(CLOCK) == "79b4838ae634efcff705e028a0ddff8b75d28d79180e3ac89f54b9cab7e5005f"
    assert clock["post_entry_outcomes_opened"] is False
    assert clock["entry_or_later_ohlc_loaded"] is False
    assert clock["event_count"] == 144
    assert clock["event_clock_sha256"] == support["event_clock_sha256"]
    forbidden = {"return", "pnl", "funding", "equity", "cagr", "mdd", "high", "low"}
    for event in clock["events"]:
        assert forbidden.isdisjoint(key.lower() for key in event)
        entry = pd.Timestamp(event["entry_date"])
        exit_ = pd.Timestamp(event["exit_date"])
        assert entry.quarter == exit_.quarter
        assert exit_ - entry == pd.Timedelta(hours=6)
