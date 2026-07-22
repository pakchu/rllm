from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

from training import build_sec_bitcoin_issuer_reactivation_breadth_support as support


REPORT = Path("results/sec_bitcoin_issuer_reactivation_breadth_support_2026-07-23.json")
CLOCK = Path(
    "data/sec_bitcoin_issuer_reactivation_breadth_2020_2023/"
    "birb120_support_clocks_2020_2023.csv.gz"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_support_artifacts_are_hash_bound_and_rejected() -> None:
    report = payload()
    assert sha256(REPORT) == "752e77022e8d670084327680da4e8d60d753a344dcbef754b592493ffa9bfec6"
    assert sha256(CLOCK) == "8f0831120764793a06873dc7ed4e1b97d3deff75d89572e2b4b8f9459bdfea41"
    assert report["decision"] == "REJECT_SOURCE"
    assert report["manifest_hash"] == "ed0336f1328735973331eec99848c774e0d11fd6c672c9dd4fdbd893438f779d"
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert report["manifest_hash"] == support.canonical_hash(core)
    assert report["clock"]["sha256"] == sha256(CLOCK)


def test_primary_is_structurally_under_supported() -> None:
    report = payload()
    primary = report["source_gates"]
    assert primary["passed"] is False
    assert primary["stats"]["train"]["total"] == 2
    assert primary["stats"]["selection"]["total"] == 0
    assert primary["stats"]["train"]["distinct_breadth_issuers"] == 6
    assert report["reactivation_stats"]["reactivation"] == 38
    assert report["schedule_stats"]["threshold_two"]["total"] == 4
    assert report["schedule_stats"]["threshold_four"]["total"] == 0
    assert report["schedule_stats"]["single_reactivation"]["selection"]["total"] == 8
    assert primary["checks"]["duplicate_signal_ids"] is True
    assert primary["checks"]["duplicate_accepted_accessions"] is True


def test_clock_has_source_fields_only_and_exact_primary_rows() -> None:
    with gzip.open(CLOCK, "rt", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 132
    assert sum(row["control"] == "primary" for row in rows) == 2
    assert {row["side"] for row in rows} == {"1"}
    forbidden = {"open", "high", "low", "close", "return", "pnl", "cagr", "mdd"}
    assert not forbidden.intersection(name.lower() for name in rows[0])


def test_outcome_boundary_remained_closed() -> None:
    boundary = payload()["outcome_boundary"]
    assert boundary["sec_source_value_rows_read"] == 3496
    assert boundary["sec_filing_body_rows_read"] == 0
    assert boundary["btc_market_rows_read"] == 0
    assert boundary["funding_rows_read"] == 0
    assert boundary["future_return_rows_read"] == 0
    assert boundary["post_2023_source_value_rows_read"] == 0
    assert boundary["pnl_cagr_mdd_opened"] is False
    assert boundary["network_calls"] == 0
    assert boundary["subprocess_calls"] == 0
