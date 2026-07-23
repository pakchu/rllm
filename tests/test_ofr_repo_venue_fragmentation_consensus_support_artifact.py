from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

from training import build_ofr_repo_venue_fragmentation_consensus_support as support


REPORT = Path(
    "results/ofr_repo_venue_fragmentation_consensus_support_2026-07-23.json"
)
CLOCK = Path(
    "results/ofr_repo_venue_fragmentation_consensus_clocks_2026-07-23.csv.gz"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_support_artifacts_are_hash_bound_and_rejected() -> None:
    report = payload()
    assert sha256(REPORT) == "c5918606c958fc8f966e8bd1884e75a91a6cec44074e2edbe86675fa7f978402"
    assert sha256(CLOCK) == "b2d251b147feecc98def94610834c7b11f078f35e580f6e7a950fc55bfe0724e"
    assert report["manifest_hash"] == "88275871e76ac1af6c5124466a4cb63426f2bbca2001bbbb0c5aa91426593f52"
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert report["manifest_hash"] == support.canonical_hash(core)
    assert report["clock_artifact"]["sha256"] == sha256(CLOCK)
    assert report["disposition"] == "reject RVFC-72 unchanged before outcomes"
    assert report["advance_to_evaluator_freeze"] is False


def test_primary_fails_frozen_source_support_gates() -> None:
    report = payload()
    assert report["source_support_passed"] is False
    failed = {name for name, passed in report["source_checks"].items() if not passed}
    assert failed == {
        "each_selection_half",
        "each_train_half",
        "each_train_year",
        "every_quarter_active",
        "maximum_entry_gap",
        "selection_each_side",
        "selection_month_concentration",
        "selection_total",
        "train_each_side",
        "train_total",
    }
    train = report["primary_support_summaries"]["train"]
    selection = report["primary_support_summaries"]["selection"]
    assert train == {
        "active_months": 15,
        "active_quarters": 7,
        "events": 30,
        "longs": 3,
        "max_single_month_share": 4 / 30,
        "maximum_entry_gap_elapsed_days": 187.0,
        "shorts": 27,
    }
    assert selection == {
        "active_months": 4,
        "active_quarters": 3,
        "events": 10,
        "longs": 10,
        "max_single_month_share": 0.5,
        "maximum_entry_gap_elapsed_days": 147.0,
        "shorts": 0,
    }
    assert report["clock_summaries"]["primary"] == report["clock_summaries"][
        "mean_without_consensus"
    ]


def test_clock_is_source_only_and_contains_exact_frozen_incidence() -> None:
    with gzip.open(CLOCK, "rt", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1219
    primary = [row for row in rows if row["control"] == "primary"]
    assert len(primary) == 40
    assert sum(row["side"] == "1" for row in primary) == 13
    assert sum(row["side"] == "-1" for row in primary) == 27
    forbidden = {
        "open",
        "high",
        "low",
        "close",
        "return",
        "pnl",
        "cagr",
        "mdd",
        "funding",
    }
    assert not forbidden.intersection(name.lower() for name in rows[0])


def test_outcome_and_comparator_boundaries_remained_closed() -> None:
    report = payload()
    boundary = report["outcome_boundary"]
    assert boundary["source_observation_rows_read"] == 77369
    assert boundary["post_2023_source_rows_read"] == 0
    assert boundary["comparator_rows_read"] == 0
    assert boundary["btc_market_rows_read"] == 0
    assert boundary["funding_rows_read"] == 0
    assert boundary["future_return_rows_read"] == 0
    assert boundary["pnl_cagr_mdd_opened"] is False
    assert boundary["comparator_access_short_circuited_on_source_failure"] is True
    assert report["novelty"] == {
        "checks": {},
        "evaluated": False,
        "metrics": {},
        "passed": False,
        "qualifying_groups": [],
        "reason": "source support failed before comparator access",
    }
