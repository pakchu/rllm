from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from preprocessing.same_millisecond_cascade import BAR_COLUMNS
from training import evaluate_same_millisecond_cascade_support as support


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
    frame["max_ms_event_count"] = 3
    frame["max_ms_coherence"] = 1.0
    frame["max_ms_side"] = 1
    frame["max_ms_sweep_bp"] = 1.0
    frame["max_ms_score"] = 1.0
    frame["quote_notional"] = 100.0
    frame["max_ms_quote_notional"] = 100.0
    frame["max_ms_signed_quote_notional"] = 100.0
    frame["max_ms_notional_share"] = 1.0
    return frame


def _policy() -> dict[str, object]:
    return {
        "baseline_bars": 4,
        "baseline_min_periods": 3,
        "score_quantile": 0.75,
        "minimum_bar_agg_trade_count": 64,
        "minimum_group_agg_trade_count": 3,
        "minimum_group_coherence": 0.8,
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
    frame["max_ms_score"] = 0.0
    frame.loc[5, "max_ms_score"] = 10.0
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
