from __future__ import annotations

import gzip
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import preregister_bitmex_trollbox_attention_support as tbasr


def _small_cfg(**changes: object) -> tbasr.Config:
    values: dict[str, object] = {
        "lookback_weeks": 2,
        "minimum_prior_slots": 2,
        "message_count_quantile": 0.5,
        "participant_count_quantile": 0.5,
        "minimum_messages": 2,
        "minimum_participants": 2,
        "maximum_participant_share": 0.60,
        "cooldown_bars": 2,
        "eligibility_start": "2020-01-01",
        "minimum_total": 0,
        "minimum_train_2020h2_2021": 0,
        "minimum_train_2020h2": 0,
        "minimum_train_2021": 0,
        "minimum_test_2022": 0,
        "minimum_each_test_half": 0,
        "minimum_each_quarter": 0,
        "minimum_active_weeks": 0,
        "minimum_train_active_weeks": 0,
        "minimum_test_active_weeks": 0,
        "maximum_quarter_share": 1.0,
    }
    values.update(changes)
    return replace(tbasr.Config(), **values)


def test_attention_reader_excludes_character_count_and_private_text(
    tmp_path: Path,
) -> None:
    source = tmp_path / "attention.csv"
    source.write_text(
        "date,message_count,unique_participant_count,"
        "maximum_participant_share,character_count\n"
        "2020-01-01 00:00:00,0,0,0.0,0\n"
        "2020-01-01 00:05:00,3,2,0.6666666667,999999\n"
    )
    frame = tbasr.read_attention_aggregate(source)
    assert list(frame.columns) == [
        "date",
        "message_count",
        "unique_participant_count",
        "maximum_participant_share",
    ]
    assert frame.attrs == {
        "character_count_loaded": False,
        "message_text_rows_loaded": 0,
        "market_rows_loaded": 0,
    }


def test_attention_reader_rejects_count_and_share_inconsistency(
    tmp_path: Path,
) -> None:
    source = tmp_path / "bad.csv"
    source.write_text(
        "date,message_count,unique_participant_count,"
        "maximum_participant_share,character_count\n"
        "2020-01-01 00:00:00,1,2,0.5,3\n"
    )
    with pytest.raises(RuntimeError, match="participants exceed messages"):
        tbasr.read_attention_aggregate(source)


def test_source_loader_binds_manifest_hash_source_hash_and_complete_grid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "attention.csv.gz"
    with gzip.open(source, "wt", encoding="utf-8", newline="") as handle:
        handle.write(
            "date,message_count,unique_participant_count,"
            "maximum_participant_share,character_count\n"
            "2022-12-31 23:45:00,0,0,0.0,0\n"
            "2022-12-31 23:50:00,3,2,0.6666666667,10\n"
            "2022-12-31 23:55:00,0,0,0.0,0\n"
        )
    manifest_path = tmp_path / "manifest.json"
    monkeypatch.setattr(tbasr, "ATTENTION_SOURCE", source)
    monkeypatch.setattr(tbasr, "SOURCE_MANIFEST", manifest_path)
    core = {
        "protocol_version": "bitmex_trollbox_attention_source_v1",
        "config": {
            "page_dir": "data/bitmex_trollbox_english_2020_2022_pages",
            "aggregate_output": str(source),
            "state_output": (
                "data/bitmex_trollbox_english_2020_2022_download_state.json"
            ),
            "manifest_output": str(manifest_path),
            "start_cursor": 0,
            "end_exclusive": "2023-01-01",
            "channel_id": 1,
            "page_size": 500,
            "request_pause_sec": 0.25,
            "timeout_sec": 30.0,
            "maximum_retries": 8,
            "participant_salt_label": "TBASR-24-private-participant-v1",
        },
        "source_audit": {
            "chronological_ids": True,
            "availability_timestamps_monotonic": True,
            "end_exclusive": "2023-01-01",
        },
        "aggregate": {
            "path": str(source),
            "sha256": tbasr.sha256_file(source),
            "rows": 3,
            "start": "2022-12-31 23:45:00+00:00",
            "end": "2022-12-31 23:55:00+00:00",
        },
    }
    manifest = {**core, "manifest_hash": tbasr.canonical_hash(core)}
    manifest_path.write_text(json.dumps(manifest))

    frame, loaded = tbasr.load_attention_source()
    assert len(frame) == 3
    assert loaded["manifest_hash"] == manifest["manifest_hash"]

    original_source = source.read_bytes()
    source.write_bytes(original_source + b"tampered")
    with pytest.raises(RuntimeError, match="aggregate hash mismatch"):
        tbasr.load_attention_source()
    source.write_bytes(original_source)

    manifest["source_audit"]["chronological_ids"] = False
    manifest["manifest_hash"] = tbasr.canonical_hash(
        tbasr._manifest_core(manifest)
    )
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="IDs are not chronological"):
        tbasr.load_attention_source()

    manifest["source_audit"]["chronological_ids"] = True
    with gzip.open(source, "wt", encoding="utf-8", newline="") as handle:
        handle.write(
            "date,message_count,unique_participant_count,"
            "maximum_participant_share,character_count\n"
            "2022-12-31 23:45:00,0,0,0.0,0\n"
            "2022-12-31 23:55:00,0,0,0.0,0\n"
        )
    manifest["aggregate"]["sha256"] = tbasr.sha256_file(source)
    manifest["aggregate"]["rows"] = 2
    manifest["manifest_hash"] = tbasr.canonical_hash(
        tbasr._manifest_core(manifest)
    )
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="complete frozen 5m grid"):
        tbasr.load_attention_source()


def test_slot_quantile_is_same_slot_strictly_prior_and_prefix_invariant() -> None:
    base = pd.Timestamp("2020-01-06")
    dates = pd.Series(
        [
            timestamp
            for week in range(4)
            for timestamp in (
                base + pd.Timedelta(weeks=week),
                base + pd.Timedelta(weeks=week, minutes=5),
            )
        ]
    )
    slots = tbasr.slot_of_week(dates)
    values = pd.Series([1, 1_000_000, 3, 1_000_000, 10, 1_000_000, 1, 1_000_000])
    first = tbasr.strictly_prior_slot_quantile(
        values,
        slots,
        lookback=2,
        minimum=2,
        quantile=0.5,
    )
    main_slot = [0, 2, 4, 6]
    assert np.isnan(first.iloc[main_slot[0]])
    assert np.isnan(first.iloc[main_slot[1]])
    assert first.iloc[main_slot[2]] == 2.0
    assert first.iloc[main_slot[3]] == 6.5

    changed = values.copy()
    changed.iloc[6] = 999_999
    second = tbasr.strictly_prior_slot_quantile(
        changed,
        slots,
        lookback=2,
        minimum=2,
        quantile=0.5,
    )
    pd.testing.assert_series_equal(first, second)


def test_attention_rule_uses_both_prior_thresholds_and_no_future() -> None:
    dates = pd.date_range("2020-01-06", periods=5, freq="7D")
    frame = pd.DataFrame(
        {
            "date": dates,
            "message_count": [1, 3, 10, 1, 12],
            "unique_participant_count": [1, 2, 5, 1, 6],
            "maximum_participant_share": [1.0, 0.5, 0.2, 1.0, 0.2],
        }
    )
    cfg = _small_cfg()
    first = tbasr.build_attention_panel(frame, cfg)
    assert first["candidate"].tolist() == [False, False, True, False, True]
    assert first.loc[2, "message_threshold"] == 2.0
    assert first.loc[2, "participant_threshold"] == 1.5
    assert first.loc[2, "observation_end"] == dates[2] + pd.Timedelta(minutes=5)
    assert first.loc[2, "entry_earliest"] == dates[2] + pd.Timedelta(minutes=10)
    assert first.loc[2, "exit_time"] == dates[2] + pd.Timedelta(minutes=130)

    changed = frame.copy()
    changed.loc[4, ["message_count", "unique_participant_count"]] = [999, 999]
    second = tbasr.build_attention_panel(changed, cfg)
    pd.testing.assert_frame_equal(first.loc[:3], second.loc[:3])


def test_greedy_cooldown_uses_first_event_and_inclusive_separation() -> None:
    dates = pd.Series(pd.date_range("2020-01-01", periods=5, freq="5min"))
    selected = tbasr._greedy_cooldown(
        dates,
        pd.Series([True, True, True, True, True]),
        cooldown_bars=2,
    )
    assert selected.tolist() == [True, False, True, False, True]


def _supported_schedule() -> pd.DataFrame:
    dates: list[pd.Timestamp] = []
    for quarter in pd.period_range("2020Q3", "2022Q4", freq="Q"):
        for index in range(25):
            week = index % 13
            duplicate_day = 0 if index < 13 else 1
            dates.append(
                quarter.start_time
                + pd.Timedelta(days=7 * week + duplicate_day, hours=12)
            )
    return pd.DataFrame({"date": pd.to_datetime(dates)}).sort_values(
        "date", ignore_index=True
    )


def test_support_gate_enforces_calendar_coverage_and_concentration() -> None:
    schedule = _supported_schedule()
    summary = tbasr.support_summary(schedule, tbasr.Config())
    assert summary["counts"] == {
        "total_2020h2_2022": 250,
        "train_2020h2_2021": 150,
        "train_2020h2": 50,
        "train_2021": 100,
        "test_2022": 100,
        "test_2022_h1": 50,
        "test_2022_h2": 50,
    }
    assert summary["active_weeks"] == {"all": 130, "train": 78, "test": 52}
    assert all(summary["checks"].values())
    assert summary["passed"] is True

    missing = schedule[
        schedule["date"].dt.to_period("Q").astype(str).ne("2021Q2")
    ].reset_index(drop=True)
    rejected = tbasr.support_summary(missing, tbasr.Config())
    assert rejected["checks"]["each_quarter"] is False
    assert rejected["passed"] is False


@pytest.mark.parametrize(
    ("check", "changes"),
    [
        ("total", {"minimum_total": 251}),
        ("train_total", {"minimum_train_2020h2_2021": 151}),
        ("train_2020h2", {"minimum_train_2020h2": 51}),
        ("train_2021", {"minimum_train_2021": 101}),
        ("test_total", {"minimum_test_2022": 101}),
        ("test_h1", {"minimum_each_test_half": 51}),
        ("test_h2", {"minimum_each_test_half": 51}),
        ("each_quarter", {"minimum_each_quarter": 26}),
        ("active_weeks", {"minimum_active_weeks": 131}),
        ("train_active_weeks", {"minimum_train_active_weeks": 79}),
        ("test_active_weeks", {"minimum_test_active_weeks": 53}),
        ("quarter_concentration", {"maximum_quarter_share": 0.09}),
    ],
)
def test_each_support_gate_rejects_just_beyond_observed_boundary(
    check: str,
    changes: dict[str, object],
) -> None:
    cfg = replace(tbasr.Config(), **changes)
    summary = tbasr.support_summary(_supported_schedule(), cfg)
    assert summary["checks"][check] is False
    assert summary["passed"] is False


def test_attention_clock_hash_binds_clock_config_protocol_and_source() -> None:
    events = [
        {
            "observation_start": "2022-01-01 00:00:00",
            "observation_end": "2022-01-01 00:05:00",
            "entry_earliest": "2022-01-01 00:10:00",
            "exit_time": "2022-01-01 02:10:00",
        }
    ]
    kwargs = {
        "cfg": tbasr.Config(),
        "protocol_hash": "protocol",
        "source_manifest_hash": "manifest",
        "source_sha256": "source",
    }
    baseline = tbasr.attention_clock_hash(events, **kwargs)
    changed = [{**events[0], "observation_start": "2022-01-01 00:05:00"}]
    assert tbasr.attention_clock_hash(changed, **kwargs) != baseline
    assert (
        tbasr.attention_clock_hash(
            events,
            **{**kwargs, "cfg": replace(tbasr.Config(), cooldown_bars=13)},
        )
        != baseline
    )
    assert (
        tbasr.attention_clock_hash(
            events, **{**kwargs, "source_sha256": "other"}
        )
        != baseline
    )


def test_protocol_and_default_configuration_are_frozen() -> None:
    cfg = tbasr.Config()
    tbasr._validate_config(cfg)
    payload = tbasr.protocol(cfg)
    assert payload["outcomes_opened"] is False
    assert payload["message_semantics_opened"] is False
    assert payload["source"]["message_text_loaded"] is False
    assert payload["source"]["character_count_loaded"] is False
    assert payload["source"]["aggregate_sha256"] == (
        "pending_outcome_blind_download"
    )
    assert payload["feature"]["threshold_grid"] is False
    assert payload["clock"]["frozen_later_hold_bars"] == 24
    with pytest.raises(ValueError, match="configuration is frozen"):
        tbasr._validate_config(replace(cfg, message_count_quantile=0.97))
