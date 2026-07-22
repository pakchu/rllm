from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import evaluate_address_funding_divergence_relay_support as support
from training import preregister_address_funding_divergence_relay as prereg


def _funding_frame(start: str, periods: int) -> pd.DataFrame:
    times = pd.date_range(start, periods=periods, freq="8h", tz="UTC")
    milliseconds = times.asi8 // 1_000_000
    return pd.DataFrame(
        {
            "funding_time_ms": milliseconds,
            "funding_time_utc": times,
            "symbol": "BTCUSDT",
            "funding_rate": np.arange(periods, dtype=float) / 10_000,
            "canonical_slot_ms": milliseconds,
            "funding_available_at": times + pd.Timedelta(minutes=5),
        }
    )


def _address_frame(start: str, periods: int) -> pd.DataFrame:
    observation = pd.date_range(start, periods=periods, freq="1D", tz="UTC")
    return pd.DataFrame(
        {
            "observation_date": observation,
            "available_at": observation + pd.Timedelta(days=1, hours=1),
            "AdrBalCnt": np.arange(periods, dtype=np.int64) + 1_000,
            "AdrActCnt": np.arange(periods, dtype=np.int64) * 3 + 5_000,
        }
    )


def test_preregistration_and_access_seal_payload_are_outcome_blind() -> None:
    payload = support.validate_preregistration()
    assert payload["candidate"] == prereg.CANDIDATE
    seal = support.access_seal_payload()
    assert seal["feature_values_inspected_before_seal"] is False
    assert seal["comparator_rows_inspected_before_seal"] is False
    assert seal["market_outcomes_opened_before_seal"] is False
    assert all(value == 0 for value in seal["row_counters"].values())


def test_frozen_source_access_seal_hash_and_manifest() -> None:
    assert support.sha256_file(support.SOURCE_ACCESS_SEAL) == (
        "cda44620c74eb206ce191a7d6bfb90bb41211ca0ccc08650ecead0cd513d692f"
    )
    seal = support.validate_access_seal()
    assert seal["manifest_hash"] == (
        "6d56244ececa18bc99be83bf228896a027c073be42cc7e0677821b3f54e541e9"
    )
    assert all(value == 0 for value in seal["row_counters"].values())


def test_header_validation_is_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        support.pd,
        "read_csv",
        lambda *args, **kwargs: pd.DataFrame(
            columns=pd.Index(prereg.ADDRESS_COLUMNS)
        ),
    )
    support.validate_header("ignored.csv", prereg.ADDRESS_COLUMNS)
    with pytest.raises(ValueError, match="header drifted"):
        support.validate_header("ignored.csv", (*prereg.ADDRESS_COLUMNS, "future"))


def test_funding_pressure_uses_last_nine_already_available_events() -> None:
    funding = _funding_frame("2021-01-01", 12)
    decision = support._timestamp("2021-01-04 00:04:59+00:00")
    pressure, latest, valid = support.funding_pressure_at(decision, funding)
    assert valid is True
    expected = funding.iloc[0:9]["funding_rate"].sum()
    assert pressure == pytest.approx(expected)
    assert latest == pd.Timestamp("2021-01-03 16:05", tz="UTC")

    decision_after = support._timestamp("2021-01-04 00:05+00:00")
    pressure_after, latest_after, valid_after = support.funding_pressure_at(
        decision_after, funding
    )
    assert valid_after is True
    assert pressure_after == pytest.approx(funding.iloc[1:10]["funding_rate"].sum())
    assert latest_after == decision_after


def test_funding_pressure_rejects_missing_slot_and_stale_latest() -> None:
    funding = _funding_frame("2021-01-01", 10).drop(index=5).reset_index(drop=True)
    pressure, _, valid = support.funding_pressure_at(
        support._timestamp("2021-01-04 00:05+00:00"), funding
    )
    assert valid is False
    assert math.isnan(pressure)

    complete = _funding_frame("2021-01-01", 9)
    pressure, _, valid = support.funding_pressure_at(
        support._timestamp("2021-01-04 08:05:01+00:00"), complete
    )
    assert valid is False
    assert math.isnan(pressure)


def test_strict_prior_midrank_excludes_current_future_and_late_reference() -> None:
    values = np.array([1.0, 2.0, 2.0, 4.0, 2.0])
    observation = pd.date_range("2021-01-01", periods=5, tz="UTC").to_numpy(
        dtype="datetime64[ns]"
    )
    feature_available = (
        pd.date_range("2021-01-02", periods=5, tz="UTC").to_numpy(
            dtype="datetime64[ns]"
        )
    )
    feature_available[1] = np.datetime64("2021-02-01")
    decisions = pd.date_range(
        "2021-01-02 12:00", periods=5, tz="UTC"
    ).to_numpy(dtype="datetime64[ns]")
    ranks, counts = support.strict_prior_midrank(
        values,
        observation,
        feature_available,
        decisions,
        lookback_days=365,
        minimum=3,
    )
    assert counts[-1] == 3
    assert ranks[-1] == pytest.approx((1 + 0.5 * 1) / 3)


def test_exact_seven_day_lag_is_not_nearest_filled() -> None:
    address = _address_frame("2021-01-01", 10).drop(index=1).reset_index(drop=True)
    funding = _funding_frame("2020-12-01", 120)
    features = support.build_features(address, funding, minimum_prior=1)
    day8 = features.loc[
        features["observation_date"].eq(pd.Timestamp("2021-01-08", tz="UTC"))
    ].iloc[0]
    day9 = features.loc[
        features["observation_date"].eq(pd.Timestamp("2021-01-09", tz="UTC"))
    ].iloc[0]
    assert math.isfinite(float(day8["balance_growth_7d"]))
    assert math.isnan(float(day9["balance_growth_7d"]))


def test_state_onset_requires_immediately_prior_valid_flat() -> None:
    dates = pd.date_range("2021-01-01", periods=6, tz="UTC").to_numpy()
    states = np.array([0, 1, 1, 1, 0, -1])
    valid = np.array([True, True, False, True, True, True])
    events = support._state_onsets(states, valid, dates)
    assert events.tolist() == [False, True, False, False, False, True]


def test_raw_candidates_populate_required_split_placeholder() -> None:
    features = pd.DataFrame(
        {
            "observation_date": pd.to_datetime(["2021-01-01"], utc=True),
            "available_at": pd.to_datetime(["2021-01-02 00:01"], utc=True),
            "primary_event": [True],
            "primary_state": [1],
        }
    )
    candidates = support._raw_candidates(features, "primary")
    assert tuple(candidates.columns) == support.CLOCK_COLUMNS
    assert candidates.iloc[0]["split"] == ""
    assert candidates.iloc[0]["entry_time"] == pd.Timestamp(
        "2021-01-02 00:10", tz="UTC"
    )


def _candidate_rows(entries: list[str]) -> pd.DataFrame:
    entry = pd.to_datetime(entries, utc=True, format="mixed")
    return pd.DataFrame(
        {
            "candidate": prereg.CANDIDATE,
            "control": "primary",
            "split": "",
            "side": 1,
            "observation_date": entry - pd.Timedelta(days=1),
            "decision_time": entry - pd.Timedelta(minutes=5),
            "entry_time": entry,
            "exit_time": entry + pd.Timedelta(hours=72),
        }
    )


def test_schedule_is_nonoverlapping_and_split_contained() -> None:
    candidates = _candidate_rows(
        [
            "2021-01-01 00:00",
            "2021-01-01 01:00",
            "2021-01-04 00:00",
            "2022-12-29 01:00",
        ]
    )
    scheduled = support._schedule(candidates, "primary")
    assert list(scheduled["entry_time"]) == list(
        pd.to_datetime(["2021-01-01", "2021-01-04"], utc=True)
    )


def test_random_side_is_stable() -> None:
    entry = pd.Timestamp("2021-01-01 00:00", tz="UTC")
    assert support._random_side(entry) == support._random_side(entry)
    assert support._random_side(entry) in {-1, 1}


def test_support_summary_uses_half_open_rolling_window() -> None:
    clock = _candidate_rows(
        ["2021-01-01", "2021-01-30 23:55", "2021-01-31", "2021-03-02"]
    )
    summary = support.split_support_summary(clock)
    assert summary["events"] == 4
    assert summary["maximum_rolling_30day_share"] == pytest.approx(0.5)


def test_timestamp_novelty_is_bidirectional() -> None:
    candidate = pd.DatetimeIndex(
        pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03"], utc=True)
    )
    comparator = pd.DatetimeIndex(
        pd.to_datetime(
            ["2021-01-01", "2021-01-02 06:00", "2021-01-05"],
            utc=True,
            format="mixed",
        )
    )
    metrics = support.timestamp_novelty_metrics(candidate, comparator)
    assert metrics["exact_jaccard"] == pytest.approx(1 / 5)
    assert metrics["candidate_near_share"] == pytest.approx(2 / 3)
    assert metrics["comparator_near_share"] == pytest.approx(2 / 3)


def test_signed_exposure_correlation_uses_complete_five_minute_grid() -> None:
    start = support._timestamp("2021-01-01T00:00:00Z")
    end = support._timestamp(start + pd.Timedelta(minutes=20))
    candidate = pd.DataFrame(
        {
            "entry_time": [start],
            "exit_time": [start + pd.Timedelta(minutes=10)],
            "side": [1],
        }
    )
    same = candidate.copy()
    opposite = candidate.assign(side=-1)
    assert support.signed_exposure_correlation(candidate, same, start, end) == pytest.approx(1.0)
    assert support.signed_exposure_correlation(candidate, opposite, start, end) == pytest.approx(-1.0)


def test_overlapping_directional_intervals_fail_closed() -> None:
    start = support._timestamp("2021-01-01T00:00:00Z")
    end = support._timestamp(start + pd.Timedelta(hours=1))
    intervals = pd.DataFrame(
        {
            "entry_time": [start, start + pd.Timedelta(minutes=5)],
            "exit_time": [start + pd.Timedelta(minutes=15), start + pd.Timedelta(minutes=20)],
            "side": [1, 1],
        }
    )
    with pytest.raises(ValueError, match="overlap"):
        support._exposure_vector(intervals, start, end)


def _directional_csv_spec(
    path: Path, *, group_column: str | None = "candidate_id"
) -> dict[str, object]:
    return {
        "candidate": "TEST",
        "path": str(path),
        "sha256": "unused",
        "format": "csv",
        "capability": "directional_interval",
        "filters": {"clock_mode": ["primary"]},
        "group_column": group_column,
        "comparison_start": "2021-01-01T00:00:00Z",
        "comparison_end_exclusive": "2021-02-01T00:00:00Z",
        "entry_column": "entry_time",
        "side_column": "side",
        "exit_column": "exit_time",
    }


def test_grouped_comparator_empty_after_filter_becomes_registry_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "empty.csv"
    pd.DataFrame(
        {
            "clock_mode": ["control"],
            "candidate_id": ["x"],
            "entry_time": ["2021-01-01T00:00:00Z"],
            "exit_time": ["2021-01-01T01:00:00Z"],
            "side": [1],
        }
    ).to_csv(path, index=False)
    spec = _directional_csv_spec(path)
    monkeypatch.setattr(prereg, "COMPARATORS", (spec,))
    monkeypatch.setattr(support, "sha256_file", lambda _: "unused")
    members = support.load_comparator_members()
    assert len(members) == 1
    assert members[0]["member"] == "TEST:__registry_failure__"
    assert "empty after frozen filters" in members[0]["contract_failure"]


def test_grouped_comparator_rejects_missing_group_identity(tmp_path: Path) -> None:
    path = tmp_path / "missing-group.csv"
    pd.DataFrame(
        {
            "clock_mode": ["primary"],
            "candidate_id": [None],
            "entry_time": ["2021-01-01T00:00:00Z"],
            "exit_time": ["2021-01-01T01:00:00Z"],
            "side": [1],
        }
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="group identity is missing"):
        support._parse_comparator_csv(_directional_csv_spec(path))


def test_comparator_rejects_missing_timestamp(tmp_path: Path) -> None:
    path = tmp_path / "missing-time.csv"
    pd.DataFrame(
        {
            "clock_mode": ["primary"],
            "entry_time": [None],
            "exit_time": ["2021-01-01T01:00:00Z"],
            "side": [1],
        }
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="entry timestamp is missing"):
        support._parse_comparator_csv(
            _directional_csv_spec(path, group_column=None)
        )


def test_csv_comparator_rejects_timezone_less_timestamp(tmp_path: Path) -> None:
    path = tmp_path / "timezone-less.csv"
    pd.DataFrame(
        {
            "clock_mode": ["primary"],
            "entry_time": ["2021-01-01 00:00:00"],
            "exit_time": ["2021-01-01T01:00:00Z"],
            "side": [1],
        }
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="timezone-less"):
        support._parse_comparator_csv(
            _directional_csv_spec(path, group_column=None)
        )


def test_json_comparator_rejects_timezone_less_timestamp(tmp_path: Path) -> None:
    path = tmp_path / "timezone-less.json"
    path.write_text(
        json.dumps(
            {
                "protocol": {
                    "post_entry_outcomes_computed": False,
                    "output_fields": ["signal_date", "side"],
                },
                "comparators": {
                    "member": {
                        "coverage_start_inclusive": "2021-01-01T00:00:00Z",
                        "coverage_end_exclusive": "2021-02-01T00:00:00Z",
                        "events": [
                            {"signal_date": "2021-01-02 00:00:00", "side": 1}
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    spec = {
        "candidate": "MICRO",
        "path": str(path),
        "format": "json_comparator_event_bundle",
    }
    with pytest.raises(ValueError, match="timezone-less"):
        support._parse_microstructure_bundle(spec)


def _zero_variance_member() -> dict[str, object]:
    start = support._timestamp("2021-01-01T00:00:00Z")
    end = support._timestamp("2021-01-01T01:00:00Z")
    return {
        "member": "zero-variance",
        "capability": "directional_interval",
        "start": start,
        "end": end,
        "events": pd.DataFrame(
            {
                "entry_time": [start],
                "exit_time": [end],
                "side": [1],
            }
        ),
    }


def test_novelty_contract_failure_is_serialized_instead_of_raised() -> None:
    start = support._timestamp("2021-01-01T00:05:00Z")
    primary = pd.DataFrame(
        {
            "entry_time": [start],
            "exit_time": [start + pd.Timedelta(minutes=10)],
            "side": [1],
        }
    )
    results, passed = support.evaluate_novelty(primary, [_zero_variance_member()])
    assert passed is False
    assert results[0]["checks"]["comparator_contract_valid"] is False
    assert "zero variance" in results[0]["contract_failure"]


@pytest.mark.parametrize(
    ("events", "message"),
    [
        (
            pd.DataFrame(
                {
                    "entry_time": pd.to_datetime(
                        ["2021-01-01T00:10:00Z", "2021-01-01T00:15:00Z"]
                    ),
                    "exit_time": pd.to_datetime(
                        ["2021-01-01T00:25:00Z", "2021-01-01T00:30:00Z"]
                    ),
                    "side": [1, -1],
                }
            ),
            "overlap",
        ),
        (
            pd.DataFrame(
                {
                    "entry_time": pd.to_datetime(["2021-01-01T00:02:00Z"]),
                    "exit_time": pd.to_datetime(["2021-01-01T00:17:00Z"]),
                    "side": [1],
                }
            ),
            "five-minute aligned",
        ),
        (
            pd.DataFrame(
                {
                    "entry_time": ["2021-01-01 00:10:00"],
                    "exit_time": ["2021-01-01T00:20:00Z"],
                    "side": [1],
                }
            ),
            "timezone-less",
        ),
        (
            pd.DataFrame(
                {
                    "entry_time": [pd.NaT],
                    "exit_time": [pd.Timestamp("2021-01-01T00:20:00Z")],
                    "side": [1],
                }
            ),
            "timestamp is missing",
        ),
        (
            pd.DataFrame(
                {
                    "entry_time": ["not-a-time"],
                    "exit_time": ["2021-01-01T00:20:00Z"],
                    "side": [1],
                }
            ),
            "DateParseError",
        ),
        (
            pd.DataFrame(columns=pd.Index(["entry_time", "exit_time", "side"])),
            "zero variance",
        ),
    ],
)
def test_directional_novelty_failures_are_serialized(
    events: pd.DataFrame, message: str
) -> None:
    start = support._timestamp("2021-01-01T00:05:00Z")
    primary = pd.DataFrame(
        {
            "entry_time": [start],
            "exit_time": [start + pd.Timedelta(minutes=10)],
            "side": [1],
        }
    )
    member = {
        "member": "malformed",
        "capability": "directional_interval",
        "start": support._timestamp("2021-01-01T00:00:00Z"),
        "end": support._timestamp("2021-01-01T01:00:00Z"),
        "events": events,
    }
    results, passed = support.evaluate_novelty(primary, [member])
    assert passed is False
    assert results[0]["checks"]["comparator_contract_valid"] is False
    assert message in results[0]["contract_failure"]


def test_run_support_writes_rejection_for_directional_novelty_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    start = support._timestamp("2021-01-01T00:05:00Z")
    clock = pd.DataFrame(
        {
            "candidate": [prereg.CANDIDATE],
            "control": ["primary"],
            "split": ["train"],
            "side": [1],
            "observation_date": [start - pd.Timedelta(days=1)],
            "decision_time": [start - pd.Timedelta(minutes=5)],
            "entry_time": [start],
            "exit_time": [start + pd.Timedelta(minutes=10)],
        }
    )
    address = pd.DataFrame(
        {
            "observation_date": [support._timestamp("2021-01-01")],
            "available_at": [support._timestamp("2021-01-02")],
            "AdrBalCnt": [1],
            "AdrActCnt": [1],
        }
    )
    funding = pd.DataFrame(
        {"funding_time_utc": [support._timestamp("2021-01-01")]}
    )
    monkeypatch.setattr(
        support, "validate_access_seal", lambda: {"manifest_hash": "seal-manifest"}
    )
    monkeypatch.setattr(support, "load_address_source", lambda: address)
    monkeypatch.setattr(support, "load_funding_source", lambda: funding)
    monkeypatch.setattr(support, "build_features", lambda *_: pd.DataFrame())
    monkeypatch.setattr(support, "build_clocks", lambda _: clock)
    monkeypatch.setattr(support, "support_checks", lambda *_: ({}, []))
    monkeypatch.setattr(
        support, "load_comparator_members", lambda: [_zero_variance_member()]
    )
    monkeypatch.setattr(support, "sha256_file", lambda _: "0" * 64)
    result_path = tmp_path / "result.json"
    payload = support.run_support(
        clock_output=tmp_path / "clock.csv.gz",
        result_output=result_path,
    )
    assert payload["decision"] == "REJECT_NO_REPAIR"
    assert payload["novelty"]["passed"] is False
    assert "zero variance" in payload["novelty"]["members"][0][
        "contract_failure"
    ]
    assert result_path.exists()
    first_bytes = result_path.read_bytes()
    repeated = support.run_support(
        clock_output=tmp_path / "clock.csv.gz",
        result_output=result_path,
    )
    assert repeated == payload
    assert result_path.read_bytes() == first_bytes


def test_clock_encoding_is_deterministic_and_outcome_free() -> None:
    clock = _candidate_rows(["2021-01-01"])
    assert support._clock_bytes(clock) == support._clock_bytes(clock)
    forbidden = {"price", "return", "pnl", "cagr", "drawdown", "settlement_mark_price"}
    assert forbidden.isdisjoint({column.lower() for column in support.CLOCK_COLUMNS})
    source_text = Path(support.EVALUATOR_SOURCE).read_text(encoding="utf-8")
    assert "strict_bar_backtest" not in source_text
    assert "BTCUSDT_5m" not in source_text
