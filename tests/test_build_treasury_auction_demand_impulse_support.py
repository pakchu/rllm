from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from training import build_treasury_auction_demand_impulse_support as support


def test_real_support_passes_without_opening_outcomes(tmp_path: Path) -> None:
    clocks = tmp_path / "clocks.csv.gz"
    report = support.build_report(clocks_path=clocks)
    assert report["outcomes_opened"] is False
    assert report["outcome_sources_opened"] == []
    assert report["support_passed"] is True
    assert report["advance_to_stage1_outcomes"] is True
    assert report["source_rows"] == 445
    assert report["source_complete_rows"] == 440
    assert report["source_quarantined_rows"] == 5
    assert all(report["support_checks"].values())
    assert report["clocks"]["sha256"] == hashlib.sha256(clocks.read_bytes()).hexdigest()
    with gzip.open(clocks, "rt", encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
    assert header == list(support.CLOCK_COLUMNS)


def test_clock_summaries_and_controls_are_frozen(tmp_path: Path) -> None:
    report = support.build_report(clocks_path=tmp_path / "clocks.csv.gz")
    primary = report["clock_summaries"]["primary"]
    assert primary["train"]["events"] == 28
    assert primary["train"]["longs"] == 18
    assert primary["train"]["shorts"] == 10
    assert primary["2023"]["events"] == 23
    assert primary["2023"]["longs"] == 9
    assert primary["2023"]["shorts"] == 14
    assert set(report["clock_summaries"]) == {
        "primary",
        "bid_to_cover_only",
        "indirect_only",
        "one_auction_delay",
    }
    assert set(report["source_clock_jaccard"]) == {
        "bid_to_cover_only",
        "indirect_only",
        "one_auction_delay",
    }


def test_written_support_artifact_replays() -> None:
    stored = json.loads(Path(support.DEFAULT_OUTPUT).read_text())
    replayed = support.build_report(write_clock=False)
    replayed["clocks"]["sha256"] = stored["clocks"]["sha256"]
    core = {key: value for key, value in replayed.items() if key != "manifest_hash"}
    replayed["manifest_hash"] = support._canonical_hash(core)
    assert replayed == stored
