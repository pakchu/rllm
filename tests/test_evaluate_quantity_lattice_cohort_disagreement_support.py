from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from preprocessing.quantity_lattice_cohort import BAR_COLUMNS
from training import evaluate_quantity_lattice_cohort_disagreement_support as support


def _source_frame(periods: int = 40) -> pd.DataFrame:
    frame = pd.DataFrame(0.0, index=np.arange(periods), columns=list(BAR_COLUMNS[1:]))
    frame.insert(0, "date", pd.date_range("2020-01-01", periods=periods, freq="5min", tz="UTC"))
    for column in (
        "source_observed",
        "source_complete",
    ):
        frame[column] = True
    for column in (
        "source_gap_day",
        "verified_zero_volume_empty",
        "post_gap_quarantine",
    ):
        frame[column] = False
    frame["agg_trade_count"] = 100
    frame["coarse_event_count"] = 3
    frame["fine_event_count"] = 16
    frame["coarse_side"] = 1
    frame["cohort_opposition"] = 1.0
    frame["qlcd_score"] = 1.0
    return frame


def _policy() -> dict[str, object]:
    return {
        "baseline_bars": 4,
        "baseline_min_periods": 3,
        "score_quantile": 0.75,
        "minimum_bar_agg_trade_count": 64,
        "minimum_coarse_event_count": 3,
        "minimum_fine_event_count": 16,
        "execution_delay_bars": 2,
        "hold_bars": 3,
    }


def test_lagged_threshold_excludes_current_row() -> None:
    score = pd.Series([1.0, 1.0, 1.0, 100.0, 1.0])
    threshold = support.lagged_threshold(
        score,
        pd.Series([True] * len(score)),
        window=4,
        minimum=3,
        quantile=0.75,
    )
    assert threshold.iloc[3] == 1.0
    assert threshold.iloc[4] > 1.0


def test_clock_is_nonoverlapping_and_does_not_read_future_source_completeness() -> None:
    frame = _source_frame()
    frame["qlcd_score"] = 0.0
    frame.loc[5, "qlcd_score"] = 10.0
    frame.loc[6:15, "source_complete"] = False
    clock, eligible = support.build_clock(frame, _policy())
    assert eligible.iloc[5]
    assert len(clock) == 1
    assert clock.loc[0, "entry_time"] == frame.loc[7, "date"]
    assert clock.loc[0, "exit_time"] == frame.loc[10, "date"]


def test_one_to_one_matching_uses_nearest_then_earlier_tie() -> None:
    primary = pd.Series(pd.to_datetime(["2023-01-01 00:10:00Z", "2023-01-01 00:20:00Z"]))
    comparator = pd.Series(
        pd.to_datetime(["2023-01-01 00:05:00Z", "2023-01-01 00:15:00Z"])
    )
    assert support.one_to_one_matches(primary, comparator, tolerance=pd.Timedelta("5min")) == 2
    metrics = support.overlap_metrics(primary, comparator)
    assert metrics["exact"]["matches"] == 0
    assert metrics["tolerant_12_bars"]["matches"] == 2


def test_overlap_matching_deduplicates_registered_clock_entries() -> None:
    primary = pd.Series(pd.to_datetime(["2023-01-01 00:10:00Z"]))
    comparator = pd.Series(
        pd.to_datetime(["2023-01-01 00:10:00Z", "2023-01-01 00:10:00Z"])
    )
    metrics = support.overlap_metrics(primary, comparator)
    assert metrics["comparator_events"] == 1
    assert metrics["exact"]["matches"] == 1


def test_dense_bafr_read_error_is_report_only(tmp_path: Path) -> None:
    sparse_path = tmp_path / "sparse.csv"
    pd.DataFrame({"entry_date": ["2020-01-01T00:00:00Z"]}).to_csv(
        sparse_path, index=False
    )
    payload = {
        "novelty_gates": {
            "comparator_registry": [
                {
                    "family": "SPARSE",
                    "path": str(sparse_path),
                    "sha256": support.sha256_file(sparse_path),
                    "members": ["SPARSE"],
                    "member_column": None,
                    "entry_column": "entry_date",
                    "coverage": ["2020-01-01", "2024-01-01"],
                }
            ],
            "exact_entry_jaccard_max": 0.05,
            "tolerant_one_to_one_jaccard_max": 0.15,
            "primary_containment_max": 0.30,
            "dense_bafr": {
                "path": str(tmp_path / "missing.csv"),
                "sha256": "0" * 64,
                "entry_column": "entry_date",
                "coverage": ["2020-01-01", "2024-01-01"],
            },
        }
    }
    clock = pd.DataFrame({"entry_time": [pd.Timestamp("2023-01-01", tz="UTC")]})
    report = support.novelty_report(clock, payload)
    assert report["passed"] is True
    assert report["errors"] == []
    assert report["dense_bafr_report"]["gated"] is False
    assert "error" in report["dense_bafr_report"]


def test_support_summary_enforces_each_period_and_concentration() -> None:
    entries = pd.to_datetime(
        [
            "2020-01-02T00:00:00Z",
            "2021-02-02T00:00:00Z",
            "2022-03-02T00:00:00Z",
            "2023-02-02T00:00:00Z",
            "2023-08-02T00:00:00Z",
        ],
        utc=True,
    )
    clock = pd.DataFrame(
        {
            "entry_time": entries,
            "side": [1, -1, 1, -1, 1],
        }
    )
    gates = {
        "total_2020_2023_min": 5,
        "total_2020_2023_max": 10,
        "each_calendar_year_min": 1,
        "each_2023_half_min": 1,
        "each_side_share_min": 0.2,
        "each_side_share_max": 0.8,
        "maximum_single_month_share": 0.2,
    }
    summary = support.support_summary(clock, gates)
    assert summary["passed"] is True
    assert summary["2023_halves"] == {"h1": 1, "h2": 1}


def test_member_entries_interprets_naive_canonical_timestamps_as_utc() -> None:
    frame = pd.DataFrame(
        {
            "policy_id": ["T01", "T02"],
            "signal_date": ["2020-01-01 00:00:00", "2020-01-01 01:00:00"],
        }
    )
    spec = {
        "family": "TAAR",
        "members": ["T01", "T02"],
        "member_column": "policy_id",
        "entry_column": None,
        "derived_entry": "signal_date + 2 completed five-minute bars",
    }
    entries = support._member_entries(frame, spec)
    assert entries["T01"].iloc[0] == pd.Timestamp("2020-01-01 00:10:00", tz="UTC")


def test_write_once_is_byte_stable(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    assert support._write_once(path, b"stable") == "created"
    assert support._write_once(path, b"stable") == "verified_existing"
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        support._write_once(path, b"changed")


def test_clock_gzip_bytes_are_deterministic() -> None:
    clock = pd.DataFrame(
        [
            {
                "decision_time": pd.Timestamp("2023-01-01 00:05:00", tz="UTC"),
                "entry_time": pd.Timestamp("2023-01-01 00:10:00", tz="UTC"),
                "exit_time": pd.Timestamp("2023-01-01 12:10:00", tz="UTC"),
                "side": 1,
                "score": 1.25,
                "threshold": 1.0,
            }
        ]
    )
    first = support._clock_bytes(clock)
    second = support._clock_bytes(clock)
    assert first == second
    assert hashlib.sha256(first).hexdigest() == hashlib.sha256(second).hexdigest()


def test_repository_preregistration_hash_is_the_frozen_evaluator_input() -> None:
    payload = json.loads(Path(support.PREREGISTRATION_PATH).read_text())
    assert support.sha256_file(support.PREREGISTRATION_PATH) == support.PREREGISTRATION_FILE_SHA256
    assert payload["manifest_hash"] == support.PREREGISTRATION_MANIFEST_HASH
