from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd

from training import build_federal_liquidity_narrative_sponsorship_support as s


REPORT = Path(
    "results/federal_liquidity_narrative_sponsorship_relay_support_2026-07-23.json"
)
CLOCK = Path(
    "data/federal_liquidity_narrative_sponsorship_relay_clocks_2020_2023.csv.gz"
)


def test_report_hash_and_source_support_pass_are_frozen() -> None:
    report = json.loads(REPORT.read_text())
    core = {
        key: value
        for key, value in report.items()
        if key not in {"report_manifest_hash", "created_at"}
    }
    assert s.canonical_hash(core) == report["report_manifest_hash"]
    assert report["support_passed"] is True
    assert report["authorized_next_stage"] == "freeze_strict_evaluator"
    assert report["failed_support_checks"] == []
    assert report["failed_novelty_checks"] == []
    assert report["outcomes_opened"] is False
    assert report["post_entry_return_computed"] is False
    assert report["funding_loaded"] is False


def test_primary_windows_and_novelty_metrics_match_frozen_result() -> None:
    report = json.loads(REPORT.read_text())
    windows = report["windows"]
    assert windows["all"]["events"] == 89
    assert windows["train_2020_2022"]["events"] == 67
    assert [windows[str(year)]["events"] for year in (2020, 2021, 2022)] == [
        25,
        23,
        18,
    ]
    assert windows["selection_2023"]["events"] == 22
    assert windows["selection_2023_h1"]["events"] == 12
    assert windows["selection_2023_h2"]["events"] == 10
    h4q60 = report["novelty"]["candidate_metrics"]["FLCC-H4-Q60"]
    assert h4q60["matched"] == 62
    assert h4q60["flnsr_containment"] < 0.70
    assert report["novelty"]["passed"] is True


def test_clock_is_outcome_free_and_matches_frozen_hash() -> None:
    report = json.loads(REPORT.read_text())
    assert s.sha256_file(CLOCK) == report["clock"]["sha256"]
    with gzip.open(CLOCK, "rt") as handle:
        clock = pd.read_csv(handle)
    assert list(clock.columns) == s.CLOCK_COLUMNS
    assert len(clock) == report["clock"]["rows"] == 774
    assert not {
        "open",
        "high",
        "low",
        "close",
        "return",
        "pnl",
        "funding",
        "cagr",
        "mdd",
        "liquidity_impulse",
        "narrative_rotation",
    }.intersection(clock.columns)
    signal = pd.to_datetime(clock["signal_time"], utc=True)
    entry = pd.to_datetime(clock["entry_time"], utc=True)
    exit_time = pd.to_datetime(clock["exit_time"], utc=True)
    assert (entry - signal).eq(pd.Timedelta(minutes=10)).all()
    assert (exit_time - entry).eq(pd.Timedelta(days=7)).all()


def test_rebuild_is_clock_deterministic(tmp_path) -> None:
    rebuilt = s.build_support(
        output=tmp_path / "support.json",
        clock_output=tmp_path / "clock.csv.gz",
    )
    frozen = json.loads(REPORT.read_text())
    assert rebuilt["support_passed"] is True
    assert rebuilt["windows"] == frozen["windows"]
    assert rebuilt["feature_funnel"] == frozen["feature_funnel"]
    assert rebuilt["controls"] == frozen["controls"]
    assert rebuilt["novelty"] == frozen["novelty"]
    assert s.sha256_file(tmp_path / "clock.csv.gz") == frozen["clock"]["sha256"]
