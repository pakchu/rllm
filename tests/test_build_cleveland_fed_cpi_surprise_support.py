from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from training import build_cleveland_fed_cpi_surprise_support as support


def test_real_support_passes_without_opening_outcomes(tmp_path: Path) -> None:
    clocks = tmp_path / "clocks.csv.gz"
    report = support.build_report(clocks_path=clocks)
    support.validate_report(report)
    assert report["outcomes_opened"] is False
    assert report["outcome_sources_opened"] == []
    assert report["market_or_funding_rows_opened"] == 0
    assert report["support_passed"] is True
    assert report["advance_to_stage1_outcomes"] is True
    assert report["source"]["source_rows"] == 60
    assert all(report["support_checks"].values())
    assert report["clocks"]["sha256"] == hashlib.sha256(clocks.read_bytes()).hexdigest()
    with gzip.open(clocks, "rt", encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
    assert header == list(support.CLOCK_COLUMNS)


def test_clock_summaries_and_controls_are_frozen(tmp_path: Path) -> None:
    report = support.build_report(clocks_path=tmp_path / "clocks.csv.gz")
    primary = report["clock_summaries"]["primary"]
    assert primary["stage1"] == {
        "events": 26,
        "longs": 10,
        "shorts": 16,
        "months": 26,
        "max_single_month_count": 1,
        "max_single_month_share": 1 / 26,
    }
    assert primary["2023"] == {
        "events": 8,
        "longs": 8,
        "shorts": 0,
        "months": 8,
        "max_single_month_count": 1,
        "max_single_month_share": 0.125,
    }
    assert set(report["clock_summaries"]) == set(support.CONTROL_NAMES)
    assert report["clocks"]["rows"] == 291


def test_falsification_clocks_have_preregistered_relationships(
    tmp_path: Path,
) -> None:
    report = support.build_report(clocks_path=tmp_path / "clocks.csv.gz")
    overlap = report["entry_clock_jaccard_vs_primary"]
    assert overlap["direction_flip"] == {"stage1": 1.0, "stage2": 1.0}
    assert overlap["one_day_delay"] == {"stage1": 0.0, "stage2": 0.0}
    assert overlap["seven_day_placebo"] == {"stage1": 0.0, "stage2": 0.0}


def test_written_support_artifact_replays_and_has_frozen_identity() -> None:
    path = Path(support.DEFAULT_OUTPUT)
    stored = json.loads(path.read_text())
    replayed = support.build_report(write_clock=False)
    assert replayed == stored
    support.validate_report(stored)
    assert support._sha256(path) == (
        "d324aac5292c19314233d35db83d6b69585b929bc4650cf2bfec55a3547c485a"
    )
    assert stored["manifest_hash"] == (
        "c95bab576f9e134df3960893adcf40a89938861f138e582439f2607871adfec0"
    )
    assert support._sha256(support.DEFAULT_CLOCKS) == (
        "d1223d848a47de64224f9b21429287cbec51052e52c4184b4f10efe79d361511"
    )
