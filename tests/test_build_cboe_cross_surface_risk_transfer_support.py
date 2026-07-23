from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import build_cboe_cross_surface_risk_transfer_support as s


def _valid_term_frame(rows: int = 3) -> pd.DataFrame:
    dates = pd.date_range("2020-01-02", periods=rows, freq="B", tz="UTC")
    return pd.DataFrame(
        {
            "observation_date": dates.strftime("%Y-%m-%d"),
            "VIX9D_close": np.linspace(12.0, 13.0, rows),
            "VIX_close": np.linspace(14.0, 15.0, rows),
            "VIX3M_close": np.linspace(16.0, 17.0, rows),
        }
    )


def _vote_state_records(dates: pd.DatetimeIndex) -> pd.DataFrame:
    patterns = (
        (1, 1, 1),
        (-1, -1, -1),
        (-1, 1, 1),
        (1, -1, 1),
        (1, 1, -1),
        (1, -1, -1),
        (-1, 1, -1),
        (-1, -1, 1),
        (0, 1, 1),
        (-1, 0, -1),
        (1, 1, 0),
    )
    pressure = {1: 0.20, 0: 0.50, -1: 0.80}
    records = []
    previous_votes = None
    previous_date = None
    for index, date in enumerate(dates):
        votes = patterns[index % len(patterns)]
        majority = s.majority_vote(votes)
        records.append(
            {
                "observation_date": date,
                "term_pressure": pressure[votes[0]],
                "tail_pressure": pressure[votes[1]],
                "option_pressure": pressure[votes[2]],
                "term_vote": votes[0],
                "tail_vote": votes[1],
                "option_vote": votes[2],
                "majority_vote": majority,
                "eligible": majority != 0,
                "side": "LONG" if majority == 1 else "SHORT",
                "vote_relation": s.vote_relation(votes),
                "minority_surface": s.minority_surface(votes),
                "term_bucket": s.pressure_bucket(pressure[votes[0]]),
                "tail_bucket": s.pressure_bucket(pressure[votes[1]]),
                "option_bucket": s.pressure_bucket(pressure[votes[2]]),
                "term_transition": s.vote_transition(
                    None if previous_votes is None else previous_votes[0], votes[0]
                ),
                "tail_transition": s.vote_transition(
                    None if previous_votes is None else previous_votes[1], votes[1]
                ),
                "option_transition": s.vote_transition(
                    None if previous_votes is None else previous_votes[2], votes[2]
                ),
                "prior_majority_transition": s.vote_transition(
                    None if previous_votes is None else s.majority_vote(previous_votes),
                    majority,
                ),
                "calendar_gap_bucket": s._gap_bucket(previous_date, date),
            }
        )
        previous_votes = votes
        previous_date = date
    return pd.DataFrame(records)


def _full_states() -> pd.DataFrame:
    dates = pd.bdate_range("2020-06-01", "2023-12-28", tz="UTC")
    return _vote_state_records(dates)


def test_loader_passes_exact_allowlist_to_read_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    observed = {}
    frame = _valid_term_frame()

    def fake_read_csv(path: Path, **kwargs: object) -> pd.DataFrame:
        observed["path"] = path
        observed.update(kwargs)
        return frame.copy()

    monkeypatch.setattr(s.pd, "read_csv", fake_read_csv)
    loaded = s.load_term_source()
    assert observed["usecols"] == list(s.prereg.TERM_ALLOWLIST)
    assert set(observed) == {"path", "usecols", "dtype"}
    assert list(loaded.columns) == list(s.prereg.TERM_ALLOWLIST)
    assert str(loaded["VIX_close"].dtype) == "float64"


def test_source_validation_is_strict_and_pre_2024() -> None:
    valid = s.validate_source_frame(
        _valid_term_frame(),
        allowlist=s.prereg.TERM_ALLOWLIST,
        source_name="term",
    )
    assert valid["observation_date"].dt.tz is not None

    duplicate = _valid_term_frame()
    duplicate.loc[1, "observation_date"] = duplicate.loc[0, "observation_date"]
    with pytest.raises(RuntimeError, match="duplicated"):
        s.validate_source_frame(
            duplicate,
            allowlist=s.prereg.TERM_ALLOWLIST,
            source_name="term",
        )

    zero = _valid_term_frame()
    zero.loc[1, "VIX_close"] = 0.0
    with pytest.raises(RuntimeError, match="primitive invalid"):
        s.validate_source_frame(
            zero,
            allowlist=s.prereg.TERM_ALLOWLIST,
            source_name="term",
        )

    future = _valid_term_frame()
    future.loc[len(future) - 1, "observation_date"] = "2024-01-02"
    with pytest.raises(RuntimeError, match="2024-or-later"):
        s.validate_source_frame(
            future,
            allowlist=s.prereg.TERM_ALLOWLIST,
            source_name="term",
        )


def test_strict_prior_midrank_excludes_current_and_uses_tie_midpoint() -> None:
    increasing = s.strict_prior_midranks([1.0, 1.0, 2.0], lookback=2, minimum=2)
    assert np.isnan(increasing[:2]).all()
    assert increasing[2] == 1.0

    tied = s.strict_prior_midranks([1.0, 1.0, 1.0], lookback=2, minimum=2)
    assert tied[2] == 0.5

    trailing = s.strict_prior_midranks([1.0, 2.0, 3.0, 0.0], lookback=2, minimum=2)
    assert trailing[-1] == 0.0


def test_option_algebra_ranks_one_observation_deltas(monkeypatch: pytest.MonkeyPatch) -> None:
    dates = pd.date_range("2020-01-01", periods=4, freq="D", tz="UTC")
    frame = pd.DataFrame(
        {
            "observation_date": dates,
            "total_volume": [100.0, 110.0, 120.0, 130.0],
            "index_call_volume": [20.0, 22.0, 24.0, 26.0],
            "index_put_volume": [30.0, 31.0, 35.0, 36.0],
            "index_volume": [50.0, 53.0, 59.0, 62.0],
            "equity_call_volume": [40.0, 44.0, 48.0, 52.0],
            "equity_put_volume": [50.0, 49.0, 55.0, 54.0],
            "vix_call_volume": [10.0, 12.0, 11.0, 14.0],
            "vix_put_volume": [8.0, 9.0, 10.0, 11.0],
        }
    )
    captured = []

    def fake_rank(values: object, **_: object) -> np.ndarray:
        array = np.asarray(list(values), dtype=float)
        captured.append(array)
        return np.zeros(len(array), dtype=float)

    monkeypatch.setattr(s, "strict_prior_midranks", fake_rank)
    result = s.build_option_features(frame)
    institutional = np.log((frame["index_put_volume"] + 0.5) / (frame["index_call_volume"] + 0.5)) - np.log(
        (frame["equity_put_volume"] + 0.5) / (frame["equity_call_volume"] + 0.5)
    )
    vix_pressure = np.log((frame["vix_call_volume"] + 0.5) / (frame["vix_put_volume"] + 0.5))
    index_share = np.log((frame["index_volume"] + 1.0) / (frame["total_volume"] + 1.0))
    for actual, level in zip(captured, (institutional, vix_pressure, index_share), strict=True):
        assert np.isnan(actual[0])
        np.testing.assert_allclose(actual[1:], np.diff(level.to_numpy()))
    assert result["option_pressure"].eq(0.0).all()


def _source_panels(rows: int = 150) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2020-01-02", periods=rows, tz="UTC")
    phase = np.arange(rows, dtype=float)
    vix = 20.0 + 0.5 * np.sin(phase / 7.0)
    term = pd.DataFrame(
        {
            "observation_date": dates,
            "VIX9D_close": vix * np.exp(0.05 * np.sin(phase / 3.0)),
            "VIX_close": vix,
            "VIX3M_close": vix / np.exp(0.04 * np.cos(phase / 5.0)),
        }
    )
    tail = pd.DataFrame(
        {
            "observation_date": dates,
            "SKEW_close": 120.0 + 5.0 * np.sin(phase / 4.0),
            "VVIX_close": vix * np.exp(1.5 + 0.1 * np.cos(phase / 6.0)),
            "VIX_close": vix,
        }
    )
    option = pd.DataFrame(
        {
            "observation_date": dates,
            "total_volume": 1000.0 + phase,
            "index_call_volume": 200.0 + 5.0 * np.sin(phase / 4.0),
            "index_put_volume": 230.0 + 7.0 * np.cos(phase / 5.0),
            "index_volume": 450.0 + 8.0 * np.sin(phase / 7.0),
            "equity_call_volume": 400.0 + 9.0 * np.cos(phase / 8.0),
            "equity_put_volume": 420.0 + 6.0 * np.sin(phase / 9.0),
            "vix_call_volume": 100.0 + 4.0 * np.sin(phase / 3.0),
            "vix_put_volume": 95.0 + 3.0 * np.cos(phase / 4.0),
        }
    )
    return term, tail, option


def test_common_states_use_exact_join_and_cross_panel_vix_equality() -> None:
    term, tail, option = _source_panels()
    states, funnel = s.build_common_states(term, tail, option)
    assert funnel["exact_common_dates"] == 150
    assert funnel["rank_complete_common_dates"] > 0
    assert states["observation_date"].is_monotonic_increasing
    assert states.iloc[0]["term_transition"] == "NO_PRIOR"

    tail.loc[140, "VIX_close"] += 0.01
    with pytest.raises(RuntimeError, match="VIX cross-panel mismatch"):
        s.build_common_states(term, tail, option)


def test_vote_relations_buckets_and_transitions_are_exact() -> None:
    assert s.pressure_bucket(0.0) == "RELIEF_STRONG"
    assert s.pressure_bucket(0.25) == "RELIEF_WEAK"
    assert s.pressure_bucket(0.5) == "NEUTRAL"
    assert s.pressure_bucket(0.75) == "STRESS_WEAK"
    assert s.pressure_bucket(1.0) == "STRESS_STRONG"
    assert s.majority_vote((1, -1, 0)) == 0
    assert s.vote_relation((1, 1, 1)) == "UNANIMOUS"
    assert s.vote_relation((1, 1, 0)) == "NEUTRAL_SUPPORTED"
    assert s.vote_relation((1, 1, -1)) == "SPLIT_MAJORITY"
    assert s.minority_surface((1, -1, 1)) == "TAIL"
    assert s.vote_transition(None, 1) == "NO_PRIOR"
    assert s.vote_transition(0, -1) == "FROM_NEUTRAL"
    assert s.vote_transition(1, 0) == "TO_NEUTRAL"
    assert s.vote_transition(1, -1) == "FLIP"


def test_candidate_uses_first_later_common_date_ny_clock_and_hold() -> None:
    states = _vote_state_records(
        pd.DatetimeIndex(
            [pd.Timestamp("2023-03-10", tz="UTC"), pd.Timestamp("2023-03-13", tz="UTC")]
        )
    )
    row = s._candidate_row("primary", states.iloc[0], states.iloc[1]["observation_date"], 1)
    assert s._format_time(row["signal_available_time"]) == "2023-03-13T13:30:00Z"
    assert s._format_time(row["entry_time"]) == "2023-03-13T13:35:00Z"
    assert row["exit_time"] == row["entry_time"] + pd.Timedelta(days=1)
    assert row["signal_id"] == s.signal_id(row)


def test_controls_freeze_stale_random_flip_and_delayed_semantics() -> None:
    states = _vote_state_records(pd.bdate_range("2022-01-03", periods=40, tz="UTC"))
    controls, raw_counts = s.build_controls(states)
    primary = controls["primary"]
    assert tuple(controls) == s.prereg.CONTROL_ORDER
    assert raw_counts["primary"] == len(states) - 1
    assert controls["exact_direction_flip"]["entry_time"].equals(primary["entry_time"])
    assert all(
        left != right
        for left, right in zip(
            controls["exact_direction_flip"]["side"], primary["side"], strict=True
        )
    )
    expected_random = [
        "LONG"
        if hashlib.sha256(f"CXRT-288|{s._format_time(entry)}".encode("ascii")).digest()[0] < 128
        else "SHORT"
        for entry in primary["entry_time"]
    ]
    assert controls["deterministic_random_side"]["side"].tolist() == expected_random
    stale = controls["one_common_date_stale"]
    assert set(stale["entry_time"]).issubset(set(primary["entry_time"]))
    delayed = controls["one_day_execution_delay"]
    assert delayed.iloc[0]["entry_time"] == primary.iloc[0]["entry_time"] + pd.Timedelta(days=1)
    assert delayed.iloc[0]["signal_available_time"] == primary.iloc[0]["signal_available_time"]


def test_global_reservation_accepts_entry_equal_previous_exit() -> None:
    states = _vote_state_records(pd.date_range("2022-01-03", periods=4, freq="D", tz="UTC"))
    rows = s.raw_candidates(states, "primary")
    reserved = s.reserve_nonoverlap(rows)
    assert len(reserved) == len(rows)
    assert reserved.iloc[1]["entry_time"] == reserved.iloc[0]["exit_time"]


def test_dense_balanced_synthetic_states_pass_source_and_composition() -> None:
    states = _full_states()
    controls, _ = s.build_controls(states)
    statistics, source_checks, composition, composition_checks = s.support_checks(states, controls)
    assert statistics["train"]["events"] >= 400
    assert statistics["selection"]["events"] >= 190
    assert all(source_checks.values()), {key: value for key, value in source_checks.items() if not value}
    assert all(composition_checks.values()), {
        key: value for key, value in composition_checks.items() if not value
    }
    assert 0.10 <= composition["selection"]["unanimous_share"] <= 0.80


def test_first_failure_is_stage_ordered() -> None:
    assert s.first_failure(
        {"support": False}, {"composition": False}, {}, artifact_eligible=True
    ) == ("source_support", "support")
    assert s.first_failure(
        {"support": True}, {"composition": False}, {}, artifact_eligible=True
    ) == ("relational_composition", "composition")
    assert s.first_failure(
        {"support": True}, {"composition": True}, {}, artifact_eligible=False
    ) == ("artifact_eligibility", "synthetic_or_injected_build")
    assert s.first_failure(
        {"support": True}, {"composition": True}, {"novelty": False}, artifact_eligible=True
    ) == ("comparator_novelty", "novelty")


def test_comparator_decoder_binds_exact_columns_and_separate_groups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "comparator.csv.gz"
    frame = pd.DataFrame(
        {
            "control": ["primary", "random", "primary", "random"],
            "entry_time": [
                "2021-01-04T13:35:00Z",
                "2021-01-04T13:35:00Z",
                "2021-01-06T13:35:00Z",
                "2021-01-06T13:35:00Z",
            ],
            "exit_time": [
                "2021-01-05T13:35:00Z",
                "2021-01-05T13:35:00Z",
                "2021-01-07T13:35:00Z",
                "2021-01-07T13:35:00Z",
            ],
            "side": ["LONG", "SHORT", "SHORT", "LONG"],
            "forbidden_outcome": [1.0, 2.0, 3.0, 4.0],
        }
    )
    frame.to_csv(path, index=False, compression="gzip", lineterminator="\n")
    header = s.prereg.csv_header(path)
    contract = {
        "id": "TEST",
        "path": str(path),
        "sha256": s.sha256_file(path),
        "header": header,
        "header_sha256": s.prereg.sha256_csv_header(path),
        "group_column": "control",
        "selected_groups": ["primary", "random"],
        "entry_column": "entry_time",
        "exit_column": "exit_time",
        "side_column": "side",
        "side_encoding": {"LONG": 1, "SHORT": -1},
        "declared_coverage": ["2021-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
    }
    original = pd.read_csv
    observed = []

    def spy_read_csv(*args: object, **kwargs: object) -> pd.DataFrame:
        observed.append(kwargs.get("usecols"))
        return original(*args, **kwargs)

    monkeypatch.setattr(s.pd, "read_csv", spy_read_csv)
    groups, decoded = s._read_comparator_groups(
        {"novelty_contract": {"comparators": [contract]}}
    )
    assert decoded == 4
    assert set(groups) == {"TEST:primary", "TEST:random"}
    assert observed == [["control", "entry_time", "exit_time", "side"]]
    assert "forbidden_outcome" not in observed[0]


def test_occupancy_fails_closed_for_overlap_and_zero_variance() -> None:
    start = pd.Timestamp("2021-01-01T00:00:00Z")
    end = pd.Timestamp("2021-01-03T00:00:00Z")
    overlap = pd.DataFrame(
        {
            "entry_time": [start, start + pd.Timedelta(hours=12)],
            "exit_time": [start + pd.Timedelta(days=1), end],
            "side_sign": [1, -1],
        }
    )
    with pytest.raises(RuntimeError, match="overlaps itself"):
        s._signed_occupancy(overlap, start, end)

    constant = pd.DataFrame(
        {"entry_time": [start], "exit_time": [end], "side_sign": [1]}
    )
    correlation, position = s.occupancy_metrics(constant, constant, start, end)
    assert correlation is None
    assert position == 1.0


def test_deterministic_clock_is_canonical_gzip_without_outcome_columns() -> None:
    states = _vote_state_records(pd.bdate_range("2022-01-03", periods=30, tz="UTC"))
    controls, _ = s.build_controls(states)
    first = s.deterministic_clock_bytes(controls)
    second = s.deterministic_clock_bytes(controls)
    assert first == second
    assert first[4:8] == b"\x00\x00\x00\x00"
    with gzip.GzipFile(fileobj=io.BytesIO(first), mode="rb") as handle:
        text = handle.read().decode("utf-8")
    assert text.splitlines()[0].split(",") == list(s.CLOCK_COLUMNS)
    assert not any(token in text.splitlines()[0].lower() for token in s.FORBIDDEN_CLOCK_TOKENS)


def test_synthetic_report_never_opens_comparators_or_outcomes() -> None:
    report, _ = s.build_support_from_states(_full_states())
    assert report["artifact_eligible"] is False
    assert report["comparator_rows_decoded"] == 0
    assert report["outcomes_opened"] is False
    assert report["first_failing_stage"] == "artifact_eligibility"
    assert all(
        value == 0
        for key, value in report["outcome_boundary"].items()
        if key != "network_calls"
    )


def test_write_once_accepts_identical_and_rejects_drift(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    assert s._write_once(path, b"alpha") == "created"
    assert s._write_once(path, b"alpha") == "verified_existing"
    with pytest.raises(RuntimeError, match="noncanonical"):
        s._write_once(path, b"beta")


def test_contract_hash_and_preregistration_are_bound() -> None:
    assert s.sha256_file(s.IMPLEMENTATION_CONTRACT) == s.IMPLEMENTATION_CONTRACT_SHA256
    payload = s.validate_preregistration()
    assert payload["manifest_hash"] == s.PREREGISTRATION_MANIFEST_HASH
    assert payload["outcomes_opened"] is False
