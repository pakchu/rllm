from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import preregister_coinm_liquidation_burst_release as prereg


def _frame(periods: int = 230) -> pd.DataFrame:
    date = pd.date_range("2023-06-25", periods=periods, freq="5min")
    frame = pd.DataFrame(
        {
            "date": date,
            "feature_available_time": date + pd.Timedelta(minutes=5, seconds=1),
            "source_valid": True,
            "event_count": 1,
            "total_liquidation_usd": 100.0,
            "liquidation_imbalance": -1.0,
            "short_liquidation_usd": 0.0,
            "long_liquidation_usd": 100.0,
            "min_snapshot_average_price": 100.0,
            "max_snapshot_average_price": 101.0,
            "snapshot_price_range_bps": 100.0,
            "snapshot_price_closing_location": 0.5,
        }
    )
    return frame


def test_threshold_is_strictly_prior_and_release_requires_decay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prereg, "ROLLING_BARS", 220)
    monkeypatch.setattr(prereg, "MIN_POSITIVE_OBSERVATIONS", 200)
    frame = _frame()
    frame.loc[220, ["event_count", "total_liquidation_usd"]] = [5, 10_000.0]
    frame.loc[221, "total_liquidation_usd"] = 1_000.0
    first = prereg.derive_release_state(frame)
    assert first.loc[220, "is_burst"]
    assert first.loc[221, "is_release"]
    threshold = first.loc[220, "prior_burst_threshold_usd"]

    changed = frame.copy()
    changed.loc[220:, "total_liquidation_usd"] *= 100.0
    second = prereg.derive_release_state(changed)
    assert second.loc[220, "prior_burst_threshold_usd"] == threshold


def test_missing_release_bar_and_counterflow_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prereg, "ROLLING_BARS", 220)
    monkeypatch.setattr(prereg, "MIN_POSITIVE_OBSERVATIONS", 200)
    frame = _frame()
    frame.loc[220, ["event_count", "total_liquidation_usd"]] = [5, 10_000.0]
    frame.loc[221, "total_liquidation_usd"] = 1_000.0
    frame.loc[221, "source_valid"] = False
    assert not prereg.derive_release_state(frame).loc[221, "is_release"]

    frame.loc[221, "source_valid"] = True
    frame.loc[221, "short_liquidation_usd"] = 2_000.0
    assert not prereg.derive_release_state(frame).loc[221, "is_release"]


def test_clock_mapping_fades_forced_flow_and_delays_entry() -> None:
    frame = _frame(3)
    frame["prior_burst_threshold_usd"] = 50.0
    frame["counterflow_usd"] = 0.0
    frame.loc[0, "total_liquidation_usd"] = 1_000.0
    row = prereg._clock_row(frame, 1, "train")
    assert row["direction"] == 1
    assert row["stop_anchor"] == 100.0
    assert row["stop_price"] == pytest.approx(99.75)
    assert row["entry_time"] == pd.Timestamp("2023-06-25 00:15:00")
    assert row["planned_exit_time"] == pd.Timestamp("2023-06-25 02:15:00")

    frame.loc[0, "liquidation_imbalance"] = 1.0
    row = prereg._clock_row(frame, 1, "train")
    assert row["direction"] == -1
    assert row["stop_anchor"] == 101.0
    assert row["stop_price"] == pytest.approx(101.2525)


def test_nonoverlap_is_enforced_per_split(monkeypatch: pytest.MonkeyPatch) -> None:
    dates = pd.date_range("2023-06-25", periods=100, freq="5min")
    state = _frame(100)
    state["is_release"] = False
    state.loc[[10, 11, 34], "is_release"] = True
    state["prior_burst_threshold_usd"] = 50.0
    state["counterflow_usd"] = 0.0
    state["date"] = dates
    state["feature_available_time"] = dates + pd.Timedelta(minutes=5, seconds=1)
    monkeypatch.setattr(prereg, "derive_release_state", lambda _: state)
    monkeypatch.setattr(
        prereg,
        "SPLITS",
        {"train": ("2023-06-25", "2023-06-26")},
    )
    clocks = prereg.build_clocks(state)
    assert clocks["release_time"].tolist() == [dates[10], dates[34]]
    assert clocks.iloc[1]["entry_time"] >= clocks.iloc[0]["planned_exit_time"]


def test_source_hashes_and_committed_manifest_are_bound() -> None:
    assert prereg.sha256_file(prereg.SOURCE) == prereg.EXPECTED_SOURCE_SHA256
    assert (
        prereg.sha256_file(prereg.SOURCE_MANIFEST)
        == prereg.EXPECTED_SOURCE_MANIFEST_SHA256
    )
    source_manifest = json.loads(Path(prereg.SOURCE_MANIFEST).read_text())
    assert source_manifest["protocol"]["outcomes_opened"] is False
    assert source_manifest["file"]["sha256"] == prereg.EXPECTED_SOURCE_SHA256


def test_canonical_hash_is_order_independent() -> None:
    assert prereg.canonical_hash({"a": 1, "b": 2}) == prereg.canonical_hash(
        {"b": 2, "a": 1}
    )
    assert prereg.canonical_hash({"a": 1}) == hashlib.sha256(b'{"a":1}').hexdigest()


def test_release_state_never_uses_future_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prereg, "ROLLING_BARS", 220)
    monkeypatch.setattr(prereg, "MIN_POSITIVE_OBSERVATIONS", 200)
    frame = _frame(240)
    baseline = prereg.derive_release_state(frame)
    changed = frame.copy()
    changed.loc[230:, "total_liquidation_usd"] = np.linspace(1e6, 1e9, 10)
    replay = prereg.derive_release_state(changed)
    pd.testing.assert_series_equal(
        baseline.loc[:229, "prior_burst_threshold_usd"],
        replay.loc[:229, "prior_burst_threshold_usd"],
    )
