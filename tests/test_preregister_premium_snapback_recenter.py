from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from training import preregister_premium_snapback_recenter as prereg


SOURCE = Path(
    "data/binance_um_premium_path_btc_2020_2026/"
    "BTCUSDT_premium_path_1m_2020-01-01_2026-06-30.csv.gz"
)
MANIFEST = Path("results/binance_um_premium_path_btc_2020_2026_manifest.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(rows: int = 240) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=rows, freq="1min")
    close = np.sin(np.arange(rows) * 0.3) / 10_000.0
    return pd.DataFrame(
        {
            "date": dates,
            "source_close_time": dates + pd.Timedelta(minutes=1) - pd.Timedelta(milliseconds=1),
            "feature_available_time": dates + pd.Timedelta(minutes=1, seconds=1),
            "source_valid": True,
            "premium_open": close,
            "premium_high": close + 0.00005,
            "premium_low": close - 0.00005,
            "premium_close": close,
        }
    )


def _state(rows: int = 36) -> pd.DataFrame:
    decision = pd.date_range("2020-02-01", periods=rows, freq="5min")
    frame = pd.DataFrame(
        {
            "date": decision - pd.Timedelta(minutes=1),
            "decision_time": decision,
            "path_start_time": decision - pd.Timedelta(minutes=30),
            "feature_available_time": decision + pd.Timedelta(seconds=1),
            "path_valid": True,
            "reference_complete": True,
            "feature_reference_complete": True,
            "prior_center": 0.0,
            "path_range": 10.0,
            "efficiency": 0.1,
            "turns": 10.0,
            "up_excursion": 8.0,
            "down_excursion": 2.0,
            "max_excursion": 8.0,
            "terminal_deviation": 1.0,
            "terminal_signed_deviation": 1.0,
            "prior_q90_path_range": 9.0,
            "prior_q35_efficiency": 0.2,
            "prior_q70_turns": 8.0,
            "prior_q85_max_excursion": 7.0,
            "prior_q40_terminal_deviation": 2.0,
            "is_candidate": False,
            "direction": 0,
        }
    )
    event_rows = [index for index in (6, 7, 12, 18) if index < rows]
    event_directions: dict[int, int] = dict(
        zip((6, 7, 12, 18), (-1, -1, 1, -1), strict=True)
    )
    frame.loc[event_rows, "is_candidate"] = True
    frame.loc[event_rows, "direction"] = [event_directions[index] for index in event_rows]
    return frame


def test_config_pins_outcome_free_source_and_single_rule() -> None:
    cfg = prereg.Config()
    assert Path(cfg.source_path) == SOURCE
    assert Path(cfg.source_manifest_path) == MANIFEST
    assert _sha256(SOURCE) == cfg.expected_source_sha256
    assert _sha256(MANIFEST) == cfg.expected_source_manifest_sha256
    assert prereg.CANDIDATE == "PSR-30/6"
    assert prereg.PATH_MINUTES == 30
    assert prereg.REFERENCE_DAYS == 30
    assert prereg.ENTRY_DELAY_MINUTES == 10
    assert prereg.HOLD_MINUTES == 30
    assert prereg.SPLITS["test"] == ("2023-01-01", "2024-01-01")
    assert prereg.SPLITS["eval"] == ("2024-01-01", "2026-07-01")


def test_load_source_physically_reads_only_premium_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    original = prereg.pd.read_csv
    observed: list[set[str]] = []

    def recording_read_csv(*args: Any, **kwargs: Any) -> pd.DataFrame:
        observed.append(set(kwargs["usecols"]))
        return cast(pd.DataFrame, original(*args, **kwargs))

    monkeypatch.setattr(prereg.pd, "read_csv", recording_read_csv)
    frame = prereg.load_source(prereg.Config())
    assert observed == [set(prereg.SOURCE_COLUMNS)]
    assert set(frame.columns).isdisjoint(
        {"btc_open", "btc_high", "btc_low", "btc_close", "return", "pnl", "funding"}
    )

    with pytest.raises(ValueError, match="source manifest disagrees|source sha256"):
        prereg.load_source(replace(prereg.Config(), expected_source_sha256="0" * 64))


def test_completed_path_features_use_strictly_prior_center(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prereg, "PATH_MINUTES", 6)
    monkeypatch.setattr(prereg, "DECISION_MINUTES", 2)
    monkeypatch.setattr(prereg, "REFERENCE_MINUTES", 20)
    monkeypatch.setattr(prereg, "MIN_REFERENCE_MINUTES", 20)
    source = _source()
    baseline = prereg._completed_path_features(source)

    current = source.copy()
    current.loc[80:85, ["premium_open", "premium_high", "premium_low", "premium_close"]] *= 1_000
    replay = prereg._completed_path_features(current)
    decision = pd.Timestamp("2020-01-01 01:26")
    left = baseline.loc[baseline["decision_time"].eq(decision)].iloc[0]
    right = replay.loc[replay["decision_time"].eq(decision)].iloc[0]
    assert right["prior_center"] == pytest.approx(left["prior_center"])
    assert right["path_range"] != pytest.approx(left["path_range"])

    future = source.copy()
    future.loc[100:, ["premium_open", "premium_high", "premium_low", "premium_close"]] = 999.0
    future_replay = prereg._completed_path_features(future)
    pd.testing.assert_frame_equal(
        baseline.loc[baseline["decision_time"].lt(pd.Timestamp("2020-01-01 01:40"))].reset_index(drop=True),
        future_replay.loc[future_replay["decision_time"].lt(pd.Timestamp("2020-01-01 01:40"))].reset_index(drop=True),
    )


def test_reference_thresholds_require_full_calendar_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(prereg, "PATH_MINUTES", 6)
    monkeypatch.setattr(prereg, "DECISION_MINUTES", 2)
    monkeypatch.setattr(prereg, "REFERENCE_MINUTES", 20)
    monkeypatch.setattr(prereg, "MIN_REFERENCE_MINUTES", 19)
    features = prereg._completed_path_features(_source(80))
    before = features.loc[
        features["decision_time"].eq(pd.Timestamp("2020-01-01 00:24")),
        "reference_complete",
    ].iloc[0]
    first_full = features.loc[
        features["decision_time"].eq(pd.Timestamp("2020-01-01 00:26")),
        "reference_complete",
    ].iloc[0]
    assert not bool(before)
    assert bool(first_full)

    monkeypatch.setattr(prereg, "PATH_DECISIONS", 3)
    monkeypatch.setattr(prereg, "REFERENCE_DECISIONS", 10)
    monkeypatch.setattr(prereg, "MIN_REFERENCE_DECISIONS", 9)
    state = _state(20)
    thresholded = prereg._add_prior_thresholds(state)
    assert not bool(thresholded.loc[11, "feature_reference_complete"])
    assert bool(thresholded.loc[12, "feature_reference_complete"])


def test_candidate_direction_requires_one_sided_excursion_and_recenter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(4)
    state["is_candidate"] = False
    state["direction"] = 0
    monkeypatch.setattr(prereg, "_completed_path_features", lambda _source: state)
    monkeypatch.setattr(prereg, "_add_prior_thresholds", lambda frame: frame)
    monkeypatch.setattr(
        prereg,
        "_psi_comparators",
        lambda _source: pd.DataFrame(
            {
                "decision_time": state["decision_time"],
                "psi_2016_active": False,
                "psi_2016_direction": 0,
                "psi_8640_active": False,
                "psi_8640_direction": 0,
            }
        ),
    )
    state.loc[1, ["up_excursion", "down_excursion"]] = [2.0, 8.0]
    state.loc[2, ["up_excursion", "down_excursion"]] = [8.0, 8.0]
    state.loc[3, "terminal_deviation"] = 3.0
    derived = prereg.derive_state(pd.DataFrame())
    assert derived["is_candidate"].tolist() == [True, True, False, False]
    assert derived["direction"].tolist() == [-1, 1, 0, 0]


def test_build_clocks_enforces_empty_latency_bucket_and_nonoverlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(prereg, "SPLITS", {"train": ("2020-02-01", "2020-02-02")})
    clocks = prereg.build_clocks(_state())
    assert clocks["decision_time"].tolist() == [
        pd.Timestamp("2020-02-01 00:30"),
        pd.Timestamp("2020-02-01 01:00"),
        pd.Timestamp("2020-02-01 01:30"),
    ]
    assert bool(
        (clocks["entry_time"] - clocks["decision_time"])
        .eq(pd.Timedelta(minutes=10))
        .all()
    )
    assert bool(
        (clocks["planned_exit_time"] - clocks["entry_time"])
        .eq(pd.Timedelta(minutes=30))
        .all()
    )
    assert bool((clocks["feature_available_time"] < clocks["entry_time"]).all())
    assert bool(
        (clocks["entry_time"].iloc[1:].reset_index(drop=True)
         >= clocks["planned_exit_time"].iloc[:-1].reset_index(drop=True)).all()
    )


def test_support_gate_requires_counts_sides_concentration_and_subperiods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(prereg, "SUPPORT_MIN_TOTAL", {"train": 4})
    monkeypatch.setattr(prereg, "SUPPORT_MIN_PER_SIDE", {"train": 1})
    monkeypatch.setattr(prereg, "SUPPORT_MAX_MONTH_SHARE", {"train": 1.0})
    monkeypatch.setattr(
        prereg,
        "SUPPORT_SUBPERIODS",
        {"train": {"all": ("2020-02-01", "2020-03-01", 4)}},
    )
    clocks = prereg.build_clocks(_state())
    extra = clocks.iloc[[0]].copy()
    extra["entry_time"] += pd.Timedelta(hours=3)
    extra["planned_exit_time"] += pd.Timedelta(hours=3)
    extra["direction"] = 1
    clocks = pd.concat([clocks, extra], ignore_index=True)
    stats = {"train": prereg._support_stats(clocks, "train")}
    assert prereg._support_passes(stats)
    clocks["direction"] = -1
    assert not prereg._support_passes({"train": prereg._support_stats(clocks, "train")})


def test_overlap_reports_exact_and_near_without_outcomes() -> None:
    primary = pd.DatetimeIndex(pd.to_datetime(["2023-01-01 00:00", "2023-01-01 02:00"]))
    other = pd.DatetimeIndex(pd.to_datetime(["2023-01-01 00:00", "2023-01-01 02:25"]))
    overlap = prereg._overlap(
        primary,
        other,
        coverage_start="2023-01-01",
        coverage_end="2023-01-02",
    )
    assert overlap["exact_intersection"] == 1
    assert overlap["exact_jaccard"] == pytest.approx(1 / 3)
    assert overlap["within_30m_primary_share"] == 1.0


def test_overlap_uses_only_shared_clock_coverage() -> None:
    primary = pd.DatetimeIndex(
        pd.to_datetime(
            [
                "2020-01-01 00:00",
                "2023-01-01 00:00",
                "2023-01-01 02:00",
                "2026-01-01 00:00",
            ]
        )
    )
    other = pd.DatetimeIndex(
        pd.to_datetime(["2023-01-01 00:00", "2023-01-01 02:25"])
    )
    overlap = prereg._overlap(
        primary,
        other,
        coverage_start="2023-01-01",
        coverage_end="2023-01-02",
    )
    assert overlap["primary_full"] == 4
    assert overlap["primary"] == 2
    assert overlap["exact_jaccard"] == pytest.approx(1 / 3)
    assert overlap["within_30m_primary_share"] == 1.0


def test_shifted_controls_cannot_cross_split_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(prereg, "SPLITS", {"train": ("2020-02-01", "2020-02-02")})
    primary = prereg.build_clocks(_state())
    early = primary.iloc[[0]].copy()
    early["entry_time"] = pd.Timestamp("2020-02-01 00:20")
    early["planned_exit_time"] = pd.Timestamp("2020-02-01 00:50")
    control = prereg._derived_primary_control(
        early, candidate="future", shift_minutes=-40
    )
    assert control.empty


def test_psi_comparator_replays_prior_immediate_entry_and_eight_hour_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(prereg, "SPLITS", {"train": ("2020-02-01", "2020-02-03")})
    state = _state(180)
    state["psi_2016_active"] = False
    state["psi_2016_direction"] = 0
    state.loc[[6, 7, 102], "psi_2016_active"] = True
    state.loc[[6, 7, 102], "psi_2016_direction"] = [-1, -1, 1]
    clocks = prereg._frozen_psi_clocks(state, window=2016)
    assert clocks["decision_time"].tolist() == [
        pd.Timestamp("2020-02-01 00:30"),
        pd.Timestamp("2020-02-01 08:30"),
    ]
    assert clocks["entry_time"].equals(clocks["decision_time"])
    assert bool(
        (clocks["planned_exit_time"] - clocks["entry_time"])
        .eq(pd.Timedelta(hours=8))
        .all()
    )
    assert bool((clocks["feature_available_time"] > clocks["entry_time"]).all())
