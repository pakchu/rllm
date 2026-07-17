from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from training import build_inflation_breadth_release_drift_support as support


def test_real_support_passes_without_opening_outcomes(tmp_path: Path) -> None:
    clocks = tmp_path / "clocks.csv.gz"
    report = support.build_report(clocks_path=clocks)
    assert report["outcomes_opened"] is False
    assert report["outcome_sources_opened"] == []
    assert report["market_or_funding_rows_opened"] == 0
    assert report["support_passed"] is True
    assert report["advance_to_stage1_outcomes"] is True
    assert report["source_rows"] == 60
    assert all(report["support_checks"].values())
    assert report["clocks"]["sha256"] == hashlib.sha256(clocks.read_bytes()).hexdigest()
    with gzip.open(clocks, "rt", encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
    assert header == list(support.CLOCK_COLUMNS)


def test_clock_summaries_and_controls_are_frozen(tmp_path: Path) -> None:
    report = support.build_report(clocks_path=tmp_path / "clocks.csv.gz")
    primary = report["clock_summaries"]["primary"]
    assert primary["stage1_2020_2022"] == {
        "events": 20,
        "longs": 8,
        "shorts": 12,
        "max_single_month_count": 1,
        "max_single_month_share": 0.05,
    }
    assert primary["stage2_2023"] == {
        "events": 7,
        "longs": 7,
        "shorts": 0,
        "max_single_month_count": 1,
        "max_single_month_share": 1 / 7,
    }
    assert set(report["clock_summaries"]) == set(support.CONTROL_NAMES)
    assert set(report["source_clock_jaccard"]) == set(support.CONTROL_NAMES) - {
        "primary"
    }


def test_flip_and_random_controls_share_only_the_clock(tmp_path: Path) -> None:
    report = support.build_report(clocks_path=tmp_path / "clocks.csv.gz")
    assert report["source_clock_jaccard"]["direction_flip"] == {
        "stage1": 1.0,
        "stage2": 1.0,
    }
    assert report["source_clock_jaccard"]["deterministic_random_side"] == {
        "stage1": 1.0,
        "stage2": 1.0,
    }


def test_written_support_artifact_replays() -> None:
    stored = json.loads(Path(support.DEFAULT_OUTPUT).read_text())
    replayed = support.build_report(write_clock=False)
    replayed["clocks"]["sha256"] = stored["clocks"]["sha256"]
    core = {key: value for key, value in replayed.items() if key != "manifest_hash"}
    replayed["manifest_hash"] = support._canonical_hash(core)
    assert replayed == stored
