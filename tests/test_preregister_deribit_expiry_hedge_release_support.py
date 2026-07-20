from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import preregister_deribit_expiry_hedge_release_support as dehr


def _small_cfg(**changes: object) -> dehr.Config:
    values: dict[str, object] = {
        "reference_lookback_days": 30,
        "minimum_prior_expiries": 3,
        "total_position_quantile": 0.50,
        "release_share_quantile": 0.50,
        "eligibility_start": "2020-01-01",
        "minimum_source_eligible_expiries": 1,
        "minimum_source_expiries_per_month": 0,
        "maximum_source_gap_days": 1000.0,
        "minimum_total": 0,
        "minimum_train_2020h2_2021": 0,
        "minimum_train_2020h2": 0,
        "minimum_train_2021": 0,
        "minimum_test_2022": 0,
        "minimum_each_test_half": 0,
        "minimum_each_eligible_quarter": 0,
        "minimum_active_months": 0,
        "minimum_side_share": 0.0,
        "maximum_month_share": 1.0,
    }
    values.update(changes)
    return replace(dehr.Config(), **values)


def _source(
    dates: pd.DatetimeIndex,
    *,
    total: np.ndarray | None = None,
    share: np.ndarray | None = None,
    side: np.ndarray | None = None,
) -> pd.DataFrame:
    size = len(dates)
    total_values = total if total is not None else np.linspace(10.0, 100.0, size)
    share_values = share if share is not None else np.linspace(0.10, 0.90, size)
    side_values = side if side is not None else np.where(np.arange(size) % 2, 1, -1)
    absolute = total_values * share_values
    return pd.DataFrame(
        {
            "expiry_time": dates,
            "source_observation_earliest": dates + pd.Timedelta(minutes=65),
            "total_position": total_values,
            "absolute_release_position": absolute,
            "net_release_position": absolute * side_values,
            "release_side": side_values,
        }
    )


def _complete_source_columns(frame: pd.DataFrame) -> pd.DataFrame:
    completed = frame.copy()
    positive = completed["net_release_position"].ge(0.0)
    completed["itm_put_position"] = completed[
        "absolute_release_position"
    ].where(positive, 0.0)
    completed["itm_call_position"] = completed[
        "absolute_release_position"
    ].where(~positive, 0.0)
    completed["put_position"] = np.maximum(
        completed["itm_put_position"], completed["total_position"] / 2.0
    )
    completed["call_position"] = (
        completed["total_position"] - completed["put_position"]
    )
    negative_too_large = completed["itm_call_position"].gt(
        completed["call_position"]
    )
    completed.loc[negative_too_large, "call_position"] = completed.loc[
        negative_too_large, "itm_call_position"
    ]
    completed["put_position"] = (
        completed["total_position"] - completed["call_position"]
    )
    completed["itm_call_count"] = (~positive).astype(int)
    completed["itm_put_count"] = positive.astype(int)
    defaults: dict[str, object] = {
        "index_price": 100.0,
        "option_count": 10,
        "call_count": 5,
        "put_count": 5,
        "otm_position": 70.0,
        "atm_position": 0.0,
        "largest_instrument_share": 0.20,
        "maximum_event_timestamp_offset_seconds": 0.1,
    }
    for column, value in defaults.items():
        completed[column] = value
    return completed[dehr.SOURCE_COLUMNS]


def test_prior_calendar_quantile_excludes_current_and_future() -> None:
    dates = pd.date_range("2020-01-01 08:00", periods=7, freq="1D", tz="UTC")
    source = _source(
        dates,
        total=np.array([10.0, 20.0, 30.0, 100.0, 40.0, 50.0, 1_000.0]),
    )
    first = dehr.prior_calendar_quantile(
        source,
        "total_position",
        lookback_days=30,
        minimum=3,
        quantile=0.5,
    )
    assert np.isnan(first.iloc[2])
    assert first.iloc[3] == 20.0
    changed = source.copy()
    changed.loc[6, "total_position"] = 1_000_000.0
    second = dehr.prior_calendar_quantile(
        changed,
        "total_position",
        lookback_days=30,
        minimum=3,
        quantile=0.5,
    )
    pd.testing.assert_series_equal(first.iloc[:6], second.iloc[:6])


def test_signal_uses_frozen_side_and_causal_clock() -> None:
    dates = pd.date_range("2020-01-01 08:00", periods=6, freq="1D", tz="UTC")
    source = _source(
        dates,
        total=np.array([10.0, 20.0, 30.0, 100.0, 15.0, 120.0]),
        share=np.array([0.10, 0.20, 0.30, 0.90, 0.05, 0.80]),
        side=np.array([-1, 1, -1, 1, -1, -1]),
    )
    panel = dehr.build_signal_panel(source, _small_cfg())
    candidates = panel.loc[panel["candidate"]]
    assert candidates.index.tolist() == [3, 5]
    assert candidates["side"].tolist() == [1, -1]
    assert (
        candidates["entry_time"]
        == candidates["expiry_time"] + pd.Timedelta(minutes=70)
    ).all()
    assert (
        candidates["exit_time"]
        == candidates["entry_time"] + pd.Timedelta(hours=6)
    ).all()


def test_signal_is_prefix_invariant() -> None:
    dates = pd.date_range("2020-01-01 08:00", periods=50, freq="1D", tz="UTC")
    source = _source(dates)
    cfg = _small_cfg()
    first = dehr.build_signal_panel(source, cfg)
    changed = source.copy()
    changed.loc[40:, "total_position"] *= 1_000.0
    changed.loc[40:, "absolute_release_position"] = changed.loc[
        40:, "total_position"
    ]
    changed.loc[40:, "net_release_position"] = changed.loc[
        40:, "absolute_release_position"
    ]
    second = dehr.build_signal_panel(changed, cfg)
    pd.testing.assert_frame_equal(first.loc[:39], second.loc[:39])


def test_validate_source_frame_rejects_clock_and_release_tampering() -> None:
    dates = pd.date_range("2020-01-01 08:00", periods=3, freq="1D", tz="UTC")
    source = _complete_source_columns(_source(dates))
    checked = dehr.validate_source_frame(source, dehr.Config())
    assert len(checked) == 3

    changed = source.copy()
    changed.loc[1, "source_observation_earliest"] += pd.Timedelta(minutes=5)
    with pytest.raises(RuntimeError, match="observation clock"):
        dehr.validate_source_frame(changed, dehr.Config())

    changed = source.copy()
    changed.loc[1, "release_side"] *= -1
    with pytest.raises(RuntimeError, match="release side"):
        dehr.validate_source_frame(changed, dehr.Config())


def _valid_source_manifest() -> dict[str, object]:
    core: dict[str, object] = {
        "protocol_version": "deribit_btc_option_delivery_source_v2",
        "config": asdict(dehr.SourceConfig()),
        "source_audit": {
            "start": "2019-01-01",
            "end_exclusive": "2023-01-01",
            "crossed_start_boundary": True,
            "expiry_events": 4,
            "first_expiry": "2019-02-01T08:00:00+00:00",
            "last_expiry": "2022-12-30T08:00:00+00:00",
            "rows_by_year": {
                "2019": 10,
                "2020": 10,
                "2021": 10,
                "2022": 10,
            },
            "expiries_by_year": {
                "2019": 1,
                "2020": 1,
                "2021": 1,
                "2022": 1,
            },
        },
        "aggregate": {
            "path": str(dehr.SOURCE_DATA),
            "sha256": "source",
            "bytes": 1,
            "rows": 4,
            "columns": dehr.SOURCE_COLUMNS,
        },
        "outcome_boundary": {
            "binance_market_rows_loaded": 0,
            "funding_rows_loaded": 0,
            "post_delivery_return_or_pnl_loaded": False,
            "raw_deribit_rows_persisted": False,
        },
        "causal_availability": {
            "deribit_publication_sla_known": False,
            "source_observation_rule": (
                "expiry_time + 65 minutes after two identical canonical "
                "delivery sets observed five minutes apart"
            ),
            "source_observation_latency_seconds": 3900,
            "earliest_next_5m_entry_latency_seconds": 4200,
        },
    }
    return {**core, "manifest_hash": dehr.canonical_hash(core)}


def test_manifest_metadata_rejects_forbidden_year_before_source_read() -> None:
    manifest = _valid_source_manifest()
    aggregate, audit = dehr.validate_source_manifest_metadata(manifest)
    assert aggregate["rows"] == 4
    assert audit["last_expiry"].startswith("2022")

    changed = _valid_source_manifest()
    changed["source_audit"]["last_expiry"] = "2023-01-01T08:00:00+00:00"
    core = {key: value for key, value in changed.items() if key != "manifest_hash"}
    changed["manifest_hash"] = dehr.canonical_hash(core)
    with pytest.raises(RuntimeError, match="forbidden year"):
        dehr.validate_source_manifest_metadata(changed)

    changed = _valid_source_manifest()
    changed["source_audit"]["expiries_by_year"]["2023"] = 1
    changed["source_audit"]["expiry_events"] = 5
    changed["aggregate"]["rows"] = 5
    core = {key: value for key, value in changed.items() if key != "manifest_hash"}
    changed["manifest_hash"] = dehr.canonical_hash(core)
    with pytest.raises(RuntimeError, match="forbidden year"):
        dehr.validate_source_manifest_metadata(changed)

def _passing_schedule_and_source() -> tuple[pd.DataFrame, pd.DataFrame]:
    entries: list[pd.Timestamp] = []
    for month in pd.period_range("2020-07", "2022-12", freq="M"):
        start = pd.Timestamp(month.start_time, tz="UTC") + pd.Timedelta(
            days=2, hours=9, minutes=10
        )
        entries.extend(start + pd.to_timedelta(np.arange(5), unit="D"))
    entry_index = pd.DatetimeIndex(entries)
    schedule = pd.DataFrame(
        {
            "entry_time": entry_index,
            "side": np.where(np.arange(len(entry_index)) % 2, 1, -1),
        }
    )
    source_dates = pd.DatetimeIndex(
        [pd.Timestamp("2019-02-01 08:00", tz="UTC")]
        + [timestamp - pd.Timedelta(minutes=70) for timestamp in entry_index]
        + [pd.Timestamp("2022-12-30 08:00", tz="UTC")]
    ).sort_values()
    source = _source(source_dates)
    return schedule, source


def test_support_summary_enforces_calendar_dispersion_and_side_balance() -> None:
    schedule, source = _passing_schedule_and_source()
    cfg = replace(
        dehr.Config(),
        minimum_source_eligible_expiries=100,
        minimum_source_expiries_per_month=1,
        maximum_source_gap_days=40.0,
        minimum_active_months=30,
    )
    summary = dehr.support_summary(schedule, source, cfg)
    assert summary["counts"] == {
        "total_2020h2_2022": 150,
        "train_2020h2_2021": 90,
        "train_2020h2": 30,
        "train_2021": 60,
        "test_2022": 60,
        "test_2022_h1": 30,
        "test_2022_h2": 30,
    }
    assert summary["active_months"] == 30
    assert summary["passed"] is True

    missing = schedule[
        schedule["entry_time"]
        .dt.tz_convert(None)
        .dt.to_period("M")
        .astype(str)
        .ne("2021-02")
    ].reset_index(drop=True)
    assert dehr.support_summary(missing, source, cfg)["passed"] is False
    one_sided = schedule.copy()
    one_sided["side"] = 1
    assert dehr.support_summary(one_sided, source, cfg)["passed"] is False


def test_event_hash_binds_side_config_preregistration_and_source() -> None:
    events = [
        {
            "expiry_time": "2022-01-01T08:00:00+00:00",
            "source_observation_earliest": "2022-01-01T09:05:00+00:00",
            "entry_time": "2022-01-01T09:10:00+00:00",
            "exit_time": "2022-01-01T15:10:00+00:00",
            "side": 1,
        }
    ]
    kwargs = {
        "cfg": dehr.Config(),
        "preregistration_hash": "prereg",
        "source_manifest_hash": "manifest",
        "source_sha256": "source",
    }
    baseline = dehr.event_clock_hash(events, **kwargs)
    assert dehr.event_clock_hash([{**events[0], "side": -1}], **kwargs) != baseline
    assert dehr.event_clock_hash(
        events,
        **{**kwargs, "cfg": replace(dehr.Config(), hold_bars=73)},
    ) != baseline
    assert dehr.event_clock_hash(
        events,
        **{**kwargs, "source_sha256": "other"},
    ) != baseline


def test_protocol_and_default_configuration_are_frozen(tmp_path: Path) -> None:
    cfg = replace(
        dehr.Config(),
        preregistration_output=str(tmp_path / "prereg.json"),
        support_output=str(tmp_path / "support.json"),
        event_clock_output=str(tmp_path / "clock.json"),
    )
    dehr._validate_config(cfg)
    payload = dehr.protocol(cfg)
    assert payload["outcomes_opened"] is False
    assert payload["source"]["aggregate_sha256"] == (
        "pending_outcome_blind_download"
    )
    assert payload["source"]["rows_at_or_after_2023_loaded"] is False
    assert payload["feature"]["threshold_grid"] is False
    assert payload["clock"]["hold_bars"] == 72
    assert payload["later_evaluation_contract"]["sealed_sequential"] == [
        "2023",
        "2024",
        "2025",
        "2026_ytd",
    ]
    with pytest.raises(ValueError, match="configuration is frozen"):
        dehr._validate_config(replace(cfg, hold_bars=73))


def test_preregistration_is_deterministic_and_tamper_evident(
    tmp_path: Path,
) -> None:
    cfg = replace(
        dehr.Config(),
        preregistration_output=str(tmp_path / "prereg.json"),
        support_output=str(tmp_path / "support.json"),
        event_clock_output=str(tmp_path / "clock.json"),
    )
    first = dehr.write_preregistration(cfg)
    first_bytes = Path(cfg.preregistration_output).read_bytes()
    second = dehr.write_preregistration(cfg)
    assert Path(cfg.preregistration_output).read_bytes() == first_bytes
    assert first["artifact_hash"] == second["artifact_hash"]
    assert dehr.load_preregistration(cfg)["artifact_hash"] == first["artifact_hash"]

    tampered = json.loads(first_bytes)
    tampered["outcomes_opened"] = True
    Path(cfg.preregistration_output).write_text(json.dumps(tampered))
    with pytest.raises(RuntimeError, match="artifact hash mismatch"):
        dehr.load_preregistration(cfg)
