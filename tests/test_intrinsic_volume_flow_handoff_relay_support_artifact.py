from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd

from training import build_intrinsic_volume_flow_handoff_relay_support as s


REPORT = Path("results/intrinsic_volume_flow_handoff_relay_support_2026-07-23.json")
CLOCK = Path("data/intrinsic_volume_flow_handoff_relay_clocks_2020_2023.csv.gz")


def test_report_hash_and_terminal_source_rejection_are_frozen() -> None:
    report = json.loads(REPORT.read_text())
    core = {
        key: value
        for key, value in report.items()
        if key not in {"report_manifest_hash", "created_at"}
    }
    assert s.canonical_hash(core) == report["report_manifest_hash"]
    assert report["support_passed"] is False
    assert report["authorized_next_stage"] is None
    assert report["windows"]["all"]["events"] == 1
    assert "train_events_min" in report["failed_checks"]
    assert report["outcomes_opened"] is False
    assert report["post_entry_return_computed"] is False
    assert report["funding_loaded"] is False


def test_clock_is_outcome_free_and_matches_frozen_hash() -> None:
    report = json.loads(REPORT.read_text())
    assert s.sha256_file(CLOCK) == report["clock"]["sha256"]
    with gzip.open(CLOCK, "rt") as handle:
        clock = pd.read_csv(handle)
    assert list(clock.columns) == s.CLOCK_COLUMNS
    assert len(clock) == report["clock"]["rows"] == 107
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
    }.intersection(clock.columns)
    decision = pd.to_datetime(clock["decision_time"], utc=True)
    entry = pd.to_datetime(clock["entry_time"], utc=True)
    exit_time = pd.to_datetime(clock["exit_time"], utc=True)
    assert decision.equals(entry)
    assert (exit_time - entry).eq(pd.Timedelta(hours=6)).all()


def test_component_control_counts_explain_sparsity_without_rescuing_primary() -> None:
    report = json.loads(REPORT.read_text())
    controls = report["controls"]
    assert controls["any_handoff"]["events"] == 66
    assert controls["no_price_lag"]["events"] == 15
    assert controls["no_flow_strength"]["events"] == 18
    assert controls["fixed_noon_handoff"]["events"] == 0
    assert report["feature_funnel"]["raw_primary"] == 1
    assert report["support_checks"]["clock_has_no_market_value_or_outcome_columns"]


def test_rebuild_is_clock_deterministic(tmp_path) -> None:
    report = s.build_support(
        output=tmp_path / "support.json",
        clock_output=tmp_path / "clock.csv.gz",
    )
    frozen = json.loads(REPORT.read_text())
    assert report["support_passed"] is False
    assert report["failed_checks"] == frozen["failed_checks"]
    assert report["windows"] == frozen["windows"]
    assert report["feature_funnel"] == frozen["feature_funnel"]
    assert s.sha256_file(tmp_path / "clock.csv.gz") == frozen["clock"]["sha256"]
