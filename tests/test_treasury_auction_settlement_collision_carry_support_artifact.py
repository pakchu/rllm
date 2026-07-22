from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

from training import build_treasury_auction_settlement_collision_carry_support as support


REPORT = Path(
    "results/treasury_auction_settlement_collision_carry_support_2026-07-23.json"
)
CLOCK = Path(
    "data/treasury_auction_settlement_collision_carry_2020_2023/"
    "tascc72_support_clocks_2020_2023.csv.gz"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_artifacts_are_hash_bound_and_rejected() -> None:
    report = payload()
    assert sha256(REPORT) == "33a0446d87cc378ec4d13c4d0e4fd3f2ff6361b36ebcb4b74eff8865c85d7c38"
    assert sha256(CLOCK) == "0333ba7f523d86a310e76ac51c15e4d273a1f4fb3e98f5e48dad530ac3696de4"
    assert report["decision"] == "REJECT_SOURCE"
    assert report["manifest_hash"] == "9b20b3455d8e2b7615d8300354a1b03c4f8e4a48447ec65fe663548749ef9736"
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert report["manifest_hash"] == support.canonical_hash(core)
    assert report["clock"]["sha256"] == sha256(CLOCK)


def test_primary_fails_support_and_term_specificity() -> None:
    report = payload()
    gates = report["source_gates"]
    assert gates["passed"] is False
    assert gates["stats"]["train"]["total"] == 11
    assert gates["stats"]["selection"]["total"] == 3
    assert gates["stats"]["train"]["maximum_calendar_gap_days"] == 92.0
    assert gates["stats"]["selection"]["by_half"] == {"2023H1": 1, "2023H2": 2}
    assert gates["checks"]["all_results_known_by_signal"] is True
    assert gates["checks"]["both_maturity_groups"] is True
    specificity = report["mechanism_specificity"]
    assert specificity["passed"] is False
    assert specificity["checks"]["term_year_permutation"] is False
    metric = specificity["metrics"]["term_year_permutation"]
    assert metric["exact_intersection"] == 14
    assert metric["primary_to_comparator_near_containment"] == 1.0


def test_clock_is_source_only_with_expected_primary_rows() -> None:
    with gzip.open(CLOCK, "rt", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 256
    assert sum(row["control"] == "primary" for row in rows) == 14
    assert {row["side"] for row in rows} == {"-1"}
    forbidden = {"open", "high", "low", "close", "return", "pnl", "cagr", "mdd"}
    assert not forbidden.intersection(name.lower() for name in rows[0])


def test_transport_and_outcome_boundaries_are_explicit() -> None:
    report = payload()
    source = report["source_stats"]
    assert source["panel_rows_read"] == 445
    assert source["raw_transport_rows_parsed"] == 4000
    assert source["raw_post_2023_transport_rows_parsed_for_key_filter"] == 1122
    assert source["post_2023_rows_materialized_into_tascc"] == 0
    boundary = report["outcome_boundary"]
    assert boundary["btc_market_rows_read"] == 0
    assert boundary["funding_rows_read"] == 0
    assert boundary["future_return_rows_read"] == 0
    assert boundary["pnl_cagr_mdd_opened"] is False
    assert boundary["network_calls"] == 0
    assert boundary["subprocess_calls"] == 0
