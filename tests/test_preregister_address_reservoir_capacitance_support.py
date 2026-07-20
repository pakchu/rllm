from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import preregister_address_reservoir_capacitance_support as arcr


def _cfg(tmp_path: Path | None = None, **changes: object) -> arcr.Config:
    cfg = arcr.Config()
    if tmp_path is not None:
        cfg = replace(
            cfg,
            preregistration_output=str(tmp_path / "prereg.json"),
            support_output=str(tmp_path / "support.json"),
            event_clock_output=str(tmp_path / "clock.json"),
        )
    return replace(cfg, **changes)


def _source(rows: int = 500, start: str = "2020-01-01") -> pd.DataFrame:
    observation = pd.date_range(start, periods=rows, freq="1D", tz="UTC")
    phase = np.arange(rows, dtype=float)
    balance = np.rint(
        20_000_000.0
        + 2_000.0 * phase
        + 80_000.0 * np.sin(phase / 19.0)
    ).astype(np.int64)
    active = np.rint(
        700_000.0
        + 200.0 * phase
        + 70_000.0 * np.sin(phase / 7.0)
        + 20_000.0 * np.cos(phase / 31.0)
    ).astype(np.int64)
    return pd.DataFrame(
        {
            "observation_date": observation,
            "available_at": observation + pd.Timedelta(days=1, hours=2),
            "AdrBalCnt": balance,
            "AdrActCnt": active,
        }
    )


def test_protocol_and_default_configuration_are_frozen(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    arcr._validate_config(cfg)
    payload = arcr.protocol(cfg)
    assert payload["outcomes_opened"] is False
    assert payload["source"]["aggregate_sha256"] == (
        "pending_outcome_blind_download"
    )
    assert payload["source"]["rows_at_or_after_2024_loaded"] is False
    assert payload["feature"]["threshold_grid"] is False
    assert payload["feature"]["change_days"] == 7
    assert payload["clock"]["hold_bars"] == 864
    assert payload["support_gate"]["count_clock"].startswith(
        "accepted entry timestamp"
    )
    with pytest.raises(ValueError, match="configuration is frozen"):
        arcr._validate_config(replace(cfg, spread_z_threshold=1.80))


def test_preregistration_is_deterministic_and_tamper_evident(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    first = arcr.write_preregistration(cfg)
    first_bytes = Path(cfg.preregistration_output).read_bytes()
    second = arcr.write_preregistration(cfg)
    assert Path(cfg.preregistration_output).read_bytes() == first_bytes
    assert first["artifact_hash"] == second["artifact_hash"]
    assert first["outcomes_opened"] is False
    assert first["bounded_source_schema_probe_opened"] is True
    assert first["complete_source_incidence_opened"] is False
    assert first["candidate_incidence_opened"] is False
    assert arcr.load_preregistration(cfg)["artifact_hash"] == first[
        "artifact_hash"
    ]

    tampered = json.loads(first_bytes)
    tampered["candidate_incidence_opened"] = True
    Path(cfg.preregistration_output).write_text(json.dumps(tampered))
    with pytest.raises(RuntimeError, match="artifact hash mismatch"):
        arcr.load_preregistration(cfg)


def test_strict_prior_z_is_calendar_and_availability_causal() -> None:
    values = np.array([0.0, 1.0, 2.0, 50.0, 1_000.0])
    observation = np.array(
        ["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04", "2021-02-20"],
        dtype="datetime64[ns]",
    )
    available = np.array(
        ["2021-01-02", "2021-01-03", "2021-01-10", "2021-01-05", "2021-02-21"],
        dtype="datetime64[ns]",
    )
    zscore, counts = arcr.strict_prior_z(
        values,
        observation,
        available,
        lookback_days=10,
        minimum=2,
    )
    assert counts[3] == 2
    assert zscore[3] == pytest.approx(
        (50.0 - 0.5) / np.std([0.0, 1.0], ddof=1)
    )
    assert counts[4] == 0

    changed = values.copy()
    changed[4] = -1_000_000.0
    changed_z, _ = arcr.strict_prior_z(
        changed,
        observation,
        available,
        lookback_days=10,
        minimum=2,
    )
    np.testing.assert_allclose(zscore[:4], changed_z[:4], equal_nan=True)


def test_features_use_complete_input_availability_and_are_prefix_invariant() -> None:
    source = _source()
    source.loc[193, "available_at"] = source.loc[
        193, "observation_date"
    ] + pd.Timedelta(days=20)
    features = arcr.build_features(source, arcr.Config())
    assert features.loc[200, "feature_available_at"] == source.loc[
        193, "available_at"
    ]
    assert features.loc[200, "source_lag_days"] == 13.0
    assert features.loc[200, "state_side"] == 0
    np.testing.assert_allclose(
        features["turnover_shift_7d"],
        features["activity_flux_7d"] - features["reservoir_flux_7d"],
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )

    changed = source.copy()
    changed.loc[480:, "AdrBalCnt"] *= 3
    changed.loc[480:, "AdrActCnt"] *= 2
    changed_features = arcr.build_features(changed, arcr.Config())
    columns = [
        "reservoir_flux_7d",
        "activity_flux_7d",
        "turnover_shift_7d",
        "reservoir_z",
        "turnover_z",
        "activity_z",
        "spread_z",
        "state_side",
        "event",
    ]
    pd.testing.assert_frame_equal(
        features.loc[:479, columns], changed_features.loc[:479, columns]
    )


def test_state_requires_neutral_before_reentry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(rows=9)
    reservoir = np.array([np.nan, 1.0, -1.0, 0.0, -1.0, -1.0, 0.0, 1.0, 1.0])
    turnover = np.array([np.nan, -1.0, 1.0, 0.0, 1.0, 1.0, 0.0, -1.0, -1.0])
    activity = np.zeros(9)
    calls = iter((reservoir, turnover, activity))

    def fixed_z(*args, **kwargs):
        values = next(calls)
        return values, np.full(len(values), 5, dtype=np.int64)

    monkeypatch.setattr(arcr, "strict_prior_z", fixed_z)
    features = arcr.build_features(source, _cfg(change_days=1))
    assert features["state_side"].tolist() == [0, 1, -1, 0, -1, -1, 0, 1, 1]
    assert features.index[features["event"]].tolist() == [1, 4, 7]


def test_nonfinite_activity_control_neutralizes_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(rows=4)
    reservoir = np.array([np.nan, 1.0, 1.0, 1.0])
    turnover = np.array([np.nan, -1.0, -1.0, -1.0])
    activity = np.full(4, np.nan)
    calls = iter((reservoir, turnover, activity))

    def fixed_z(*args, **kwargs):
        values = next(calls)
        return values, np.full(len(values), 5, dtype=np.int64)

    monkeypatch.setattr(arcr, "strict_prior_z", fixed_z)
    features = arcr.build_features(source, _cfg(change_days=1))
    assert features["state_side"].eq(0).all()
    assert not features["event"].any()


def _feature_events() -> pd.DataFrame:
    available = pd.to_datetime(
        [
            "2021-07-02 00:02Z",
            "2021-07-03 00:02Z",
            "2021-07-05 00:05Z",
            "2022-12-28 23:55Z",
            "2022-12-30 00:02Z",
            "2023-01-02 00:02Z",
        ],
        utc=True,
    )
    frame = pd.DataFrame(
        {
            "observation_date": available.floor("1D") - pd.Timedelta(days=1),
            "feature_available_at": available,
            "event": True,
            "state_side": [1, -1, 1, 1, -1, -1],
        }
    )
    numeric = {
        "reservoir_flux_7d": 0.1,
        "activity_flux_7d": 0.0,
        "turnover_shift_7d": -0.1,
        "reservoir_z": 1.0,
        "turnover_z": -1.0,
        "activity_z": 0.0,
        "spread_z": 2.0,
        "source_lag_days": 1.0,
        "reservoir_reference_count": 180,
        "turnover_reference_count": 180,
        "activity_reference_count": 180,
    }
    for column, value in numeric.items():
        frame[column] = value
    return frame


def test_schedule_is_delayed_nonoverlapping_and_split_contained() -> None:
    clock = arcr.schedule_clock(_feature_events(), arcr.Config())
    assert clock["split"].tolist() == ["train", "train", "test"]
    assert clock.iloc[0]["earliest_observable_open"] == pd.Timestamp(
        "2021-07-02 00:05", tz="UTC"
    )
    assert clock.iloc[0]["entry_time"] == pd.Timestamp(
        "2021-07-02 00:10", tz="UTC"
    )
    assert clock.iloc[1]["entry_time"] == clock.iloc[0]["exit_time"]
    assert not clock["observation_date"].eq(
        pd.Timestamp("2022-12-29", tz="UTC")
    ).any()
    assert not clock["entry_time"].eq(
        pd.Timestamp("2022-12-29 00:00", tz="UTC")
    ).any()
    assert (
        clock.loc[clock["split"].eq("test"), "entry_time"]
        >= pd.Timestamp("2023-01-01", tz="UTC")
    ).all()


def _passing_clock() -> pd.DataFrame:
    entries: list[pd.Timestamp] = []
    for month in pd.period_range("2021-07", "2023-12", freq="M"):
        base = pd.Timestamp(month.start_time, tz="UTC")
        entries.extend(base + pd.to_timedelta([2, 10, 18], unit="D"))
    entries.append(pd.Timestamp("2022-01-27", tz="UTC"))
    entries.sort()
    entry = pd.DatetimeIndex(entries)
    split = np.where(entry < pd.Timestamp("2023-01-01", tz="UTC"), "train", "test")
    side = np.where(np.arange(len(entry)) % 2, 1, -1)
    return pd.DataFrame({"entry_time": entry, "split": split, "side": side})


def test_support_counts_only_accepted_entry_clock_and_enforces_dispersion() -> None:
    clock = _passing_clock()
    summary = arcr.support_summary(clock, arcr.Config())
    assert summary["counts"] == {
        "total_2021h2_2023": 91,
        "train_2021h2_2022": 55,
        "train_2021h2": 18,
        "train_2022": 37,
        "test_2023": 36,
        "test_2023_h1": 18,
        "test_2023_h2": 18,
    }
    assert summary["active_months"] == 30
    assert min(summary["quarter_counts"].values()) >= 9
    assert summary["passed"] is True

    one_sided = clock.copy()
    one_sided["side"] = 1
    assert arcr.support_summary(one_sided, arcr.Config())["passed"] is False
    missing_quarter = clock.loc[
        ~clock["entry_time"].between(
            pd.Timestamp("2022-04-01", tz="UTC"),
            pd.Timestamp("2022-07-01", tz="UTC"),
            inclusive="left",
        )
    ]
    rejected = arcr.support_summary(missing_quarter, arcr.Config())
    assert rejected["quarter_counts"]["2022Q2"] == 0
    assert rejected["passed"] is False


def _valid_source_manifest() -> dict[str, object]:
    hashes = ["a" * 64]
    core: dict[str, object] = {
        "protocol_version": 1,
        "candidate": arcr.POLICY_ID,
        "config": asdict(arcr.SourceConfig()),
        "official_catalog_url": arcr.OFFICIAL_CATALOG_URL,
        "source_audit": {
            "source_url": arcr.source_url(arcr.SourceConfig()),
            "response_pages": 1,
            "response_page_lengths": [1826],
            "response_page_sha256": hashes,
            "response_chain_sha256": arcr.canonical_hash(hashes),
            "expected_rows": 1826,
            "observed_rows": 1826,
            "first_observation": "2019-01-01T00:00:00Z",
            "last_observation": "2023-12-31T00:00:00Z",
            "maximum_observation_gap_days": 1,
            "duplicates": 0,
            "missing_days": 0,
            "unexpected_row_fields": 0,
        },
        "output": str(arcr.SOURCE_DATA),
        "output_columns": arcr.SOURCE_COLUMNS,
        "output_sha256": "b" * 64,
        "source_semantics": arcr.EXPECTED_SOURCE_SEMANTICS,
        "causal_availability": arcr.EXPECTED_CAUSAL_AVAILABILITY,
        "revision_boundary": arcr.EXPECTED_REVISION_BOUNDARY,
        "outcome_boundary": {
            "btc_market_rows_loaded": 0,
            "funding_rows_loaded": 0,
            "return_or_pnl_fields": 0,
            "post_2023_source_rows_loaded": 0,
            "raw_api_pages_persisted": False,
        },
    }
    return {**core, "manifest_hash": arcr.canonical_hash(core)}


def _rehash(manifest: dict[str, object]) -> None:
    core = {
        key: value for key, value in manifest.items() if key != "manifest_hash"
    }
    manifest["manifest_hash"] = arcr.canonical_hash(core)


def test_manifest_rejects_forbidden_source_metadata_before_file_read() -> None:
    manifest = _valid_source_manifest()
    metadata = arcr.validate_source_manifest_metadata(manifest, arcr.Config())
    assert metadata["output_sha256"] == "b" * 64

    changed = _valid_source_manifest()
    changed["source_audit"]["last_observation"] = "2024-01-01T00:00:00Z"
    _rehash(changed)
    with pytest.raises(RuntimeError, match="last_observation"):
        arcr.validate_source_manifest_metadata(changed, arcr.Config())

    changed = _valid_source_manifest()
    changed["outcome_boundary"]["post_2023_source_rows_loaded"] = 1
    _rehash(changed)
    with pytest.raises(RuntimeError, match="outcome boundary"):
        arcr.validate_source_manifest_metadata(changed, arcr.Config())

    changed = _valid_source_manifest()
    changed["source_audit"]["response_page_lengths"] = ["1826"]
    _rehash(changed)
    with pytest.raises(RuntimeError, match="response-page audit"):
        arcr.validate_source_manifest_metadata(changed, arcr.Config())

    changed = _valid_source_manifest()
    changed["future_return_series"] = [1.0]
    _rehash(changed)
    with pytest.raises(RuntimeError, match="manifest schema drift"):
        arcr.validate_source_manifest_metadata(changed, arcr.Config())

    changed = _valid_source_manifest()
    changed["source_audit"]["market_rows"] = 1
    _rehash(changed)
    with pytest.raises(RuntimeError, match="audit schema drift"):
        arcr.validate_source_manifest_metadata(changed, arcr.Config())

    changed = _valid_source_manifest()
    changed["causal_availability"] = "available at observation time"
    _rehash(changed)
    with pytest.raises(RuntimeError, match="availability semantics"):
        arcr.validate_source_manifest_metadata(changed, arcr.Config())

    changed = _valid_source_manifest()
    changed["source_semantics"] = {}
    _rehash(changed)
    with pytest.raises(RuntimeError, match="source semantics"):
        arcr.validate_source_manifest_metadata(changed, arcr.Config())


def test_event_hash_binds_side_policy_and_source() -> None:
    events = [
        {
            "entry_time": "2023-01-01T00:05:00+00:00",
            "side": 1,
        }
    ]
    kwargs = {
        "cfg": arcr.Config(),
        "preregistration_hash": "prereg",
        "source_manifest_hash": "manifest",
        "source_sha256": "source",
    }
    baseline = arcr.event_clock_hash(events, **kwargs)
    assert arcr.event_clock_hash(
        [{**events[0], "side": -1}], **kwargs
    ) != baseline
    assert arcr.event_clock_hash(
        events,
        **{**kwargs, "cfg": replace(arcr.Config(), hold_bars=865)},
    ) != baseline
    assert arcr.event_clock_hash(
        events, **{**kwargs, "source_sha256": "other"}
    ) != baseline
