from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from training import (
    preregister_deribit_funding_impulse_absorption_support as dfia,
)


def _cfg(tmp_path: Path | None = None, **changes: object) -> dfia.Config:
    cfg = dfia.Config()
    if tmp_path is not None:
        cfg = replace(
            cfg,
            preregistration_output=str(tmp_path / "prereg.json"),
            support_output=str(tmp_path / "support.json"),
            event_clock_output=str(tmp_path / "clock.json"),
        )
    return replace(cfg, **changes)


def _small_source(
    *,
    start: str = "2022-01-01T00:00:00Z",
    hours: int = 80,
    missing_offsets: set[int] | None = None,
    interest_1h: np.ndarray | None = None,
    index_returns: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dfia.Config]:
    missing_offsets = missing_offsets or set()
    full_timestamp = pd.date_range(start, periods=hours, freq="1h", tz="UTC")
    full_interest = (
        np.asarray(interest_1h, dtype=float)
        if interest_1h is not None
        else 0.001 + np.sin(np.arange(hours, dtype=float) / 5.0) * 0.0002
    )
    full_returns = (
        np.asarray(index_returns, dtype=float)
        if index_returns is not None
        else np.sin(np.arange(hours, dtype=float) / 7.0) * 0.001
    )
    if len(full_interest) != hours or len(full_returns) != hours:
        raise ValueError("test source vectors must match hours")
    previous = np.empty(hours, dtype=float)
    index_price = np.empty(hours, dtype=float)
    prior_price = 100.0
    for offset, value in enumerate(full_returns):
        previous[offset] = prior_price
        index_price[offset] = prior_price * np.exp(value)
        prior_price = index_price[offset]
    interest_8h = np.array(
        [
            full_interest[max(0, offset - 7) : offset + 1].sum()
            for offset in range(hours)
        ],
        dtype=float,
    )
    keep = np.array(
        [offset not in missing_offsets for offset in range(hours)], dtype=bool
    )
    frame = pd.DataFrame(
        {
            "timestamp": full_timestamp[keep],
            "available_at": full_timestamp[keep] + pd.Timedelta(minutes=5),
            "interest_1h": full_interest[keep],
            "interest_8h": interest_8h[keep],
            "index_price": index_price[keep],
            "prev_index_price": previous[keep],
        }
    )
    cfg = _cfg(
        source_first_exact=frame["timestamp"].iloc[0].isoformat(),
        source_last_exact=frame["timestamp"].iloc[-1].isoformat(),
        reference_lookback_hours=24,
        minimum_prior_observations=8,
    )
    return frame, cfg


def _valid_manifest() -> dict[str, Any]:
    hashes = ["a" * 64]
    expected_hours = 40_958
    core: dict[str, Any] = {
        "protocol_version": 1,
        "candidate": dfia.POLICY_ID,
        "config": asdict(dfia.SourceConfig()),
        "official_docs": dfia.OFFICIAL_DOCS,
        "official_usage_policy": dfia.OFFICIAL_USAGE_POLICY,
        "source_decision": str(dfia.SOURCE_DECISION),
        "source_decision_sha256": dfia.SOURCE_DECISION_SHA256,
        "source_audit": {
            "request_windows": 1,
            "request_window_hours_max": dfia.SourceConfig.chunk_hours,
            "response_result_lengths": [1],
            "response_result_sha256": hashes,
            "response_chain_sha256": dfia.canonical_hash(hashes),
            "response_environment": {
                "jsonrpc": "2.0",
                "testnet": False,
                "server_timing_validated_not_persisted": True,
            },
            "requested_hours": expected_hours,
            "observed_rows": expected_hours,
            "coverage_ratio": 1.0,
            "first_observation": "2019-04-30T10:00:00Z",
            "last_observation": "2023-12-31T23:00:00Z",
            "missing_hours": 0,
            "missing_hours_head": [],
            "maximum_observation_gap_hours": 1,
            "exact_boundary_duplicates": 0,
            "conflicting_duplicates": 0,
            "contiguous_index_price_links_checked": expected_hours - 1,
            "memory_identity": {
                "contiguous_eight_hour_windows": expected_hours - 7,
                "maximum_absolute_sum1h_minus_8h": "0.0",
                "median_absolute_sum1h_minus_8h": "0.0",
                "maximum_allowed_absolute_error": "0.00005",
                "all_windows_within_tolerance": True,
            },
            "unexpected_row_fields": 0,
        },
        "output": str(dfia.SOURCE_DATA),
        "output_columns": dfia.SOURCE_COLUMNS,
        "output_sha256": "b" * 64,
        "source_semantics": dfia.EXPECTED_SOURCE_SEMANTICS,
        "causal_availability": {
            "historical_synthetic_delay_minutes": 5,
            "live_rule": (
                "use actual first successful observation when later; never backdate"
            ),
            "gap_rule": (
                "a missing hour breaks the feature chain and is never filled"
            ),
        },
        "revision_boundary": "frozen source vintage",
        "distribution_boundary": "local source remains ignored",
        "outcome_boundary": dict(dfia.EXPECTED_OUTCOME_BOUNDARY),
    }
    return {**core, "manifest_hash": dfia.canonical_hash(core)}


def _rehash(manifest: dict[str, Any]) -> None:
    core = {
        key: value for key, value in manifest.items() if key != "manifest_hash"
    }
    manifest["manifest_hash"] = dfia.canonical_hash(core)


def test_protocol_and_default_configuration_are_frozen(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    dfia._validate_config(cfg)
    payload = dfia.protocol(cfg)
    assert payload["outcomes_opened"] is False
    assert payload["source"]["aggregate_sha256"] == (
        "pending_outcome_blind_download"
    )
    assert payload["source"]["rows_at_or_after_2024_loaded"] is False
    assert payload["feature"]["threshold_grid"] is False
    assert payload["source_gate"]["expected_hourly_timestamps"] == 40_958
    assert payload["clock"]["hold_bars"] == 72
    assert payload["later_evaluation_contract"][
        "minimum_component_or_stale_control_trades_each_split"
    ] == 30
    with pytest.raises(ValueError, match="configuration is frozen"):
        dfia._validate_config(replace(cfg, funding_impulse_z_threshold=1.20))


def test_preregistration_is_deterministic_and_tamper_evident(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    first = dfia.write_preregistration(cfg)
    first_bytes = Path(cfg.preregistration_output).read_bytes()
    second = dfia.write_preregistration(cfg)
    assert Path(cfg.preregistration_output).read_bytes() == first_bytes
    assert first["artifact_hash"] == second["artifact_hash"]
    assert first["outcomes_opened"] is False
    assert first["complete_source_incidence_opened"] is False
    assert first["candidate_incidence_opened"] is False
    assert dfia.load_preregistration(cfg)["artifact_hash"] == first[
        "artifact_hash"
    ]

    tampered = json.loads(first_bytes)
    tampered["candidate_incidence_opened"] = True
    Path(cfg.preregistration_output).write_text(json.dumps(tampered))
    with pytest.raises(RuntimeError, match="artifact hash mismatch"):
        dfia.load_preregistration(cfg)


def test_manifest_rejects_forbidden_source_metadata_before_file_read() -> None:
    manifest = _valid_manifest()
    audit = dfia.validate_source_manifest_metadata(manifest, dfia.Config())
    assert audit["requested_hours"] == 40_958

    changed = _valid_manifest()
    changed["config"]["end_exclusive"] = "2025-01-01T00:00:00Z"
    _rehash(changed)
    with pytest.raises(RuntimeError, match="request contract"):
        dfia.validate_source_manifest_metadata(changed, dfia.Config())

    changed = _valid_manifest()
    changed["outcome_boundary"]["post_2023_source_rows_loaded"] = 1
    _rehash(changed)
    with pytest.raises(RuntimeError, match="outcome boundary"):
        dfia.validate_source_manifest_metadata(changed, dfia.Config())


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda manifest: manifest["source_audit"].update(
                coverage_ratio=0.99
            ),
            "coverage audit",
        ),
        (
            lambda manifest: manifest["source_audit"].update(
                first_observation="2019-04-30T11:00:00Z"
            ),
            "boundary/schema audit",
        ),
        (
            lambda manifest: manifest["source_audit"].update(
                exact_boundary_duplicates=1
            ),
            "boundary/schema audit",
        ),
        (
            lambda manifest: manifest["source_audit"].update(
                response_chain_sha256="c" * 64
            ),
            "response-chain audit",
        ),
        (
            lambda manifest: manifest["source_audit"][
                "memory_identity"
            ].update(all_windows_within_tolerance=False),
            "memory audit",
        ),
    ],
)
def test_manifest_audit_drift_fails_closed(mutator, message: str) -> None:
    manifest = _valid_manifest()
    mutator(manifest)
    _rehash(manifest)
    with pytest.raises(RuntimeError, match=message):
        dfia.validate_source_manifest_metadata(manifest, dfia.Config())


def test_source_frame_validates_clock_chain_memory_and_availability() -> None:
    source, cfg = _small_source()
    checked = dfia.validate_source_frame(source, cfg)
    assert len(checked) == len(source)

    changed = source.copy()
    changed.loc[10, "available_at"] += pd.Timedelta(minutes=1)
    with pytest.raises(RuntimeError, match="availability clock"):
        dfia.validate_source_frame(changed, cfg)

    changed = source.copy()
    changed.loc[10, "prev_index_price"] *= 0.99
    with pytest.raises(RuntimeError, match="index-price chain"):
        dfia.validate_source_frame(changed, cfg)

    changed = source.copy()
    changed.loc[10, "interest_8h"] += 0.01
    with pytest.raises(RuntimeError, match="memory invariant"):
        dfia.validate_source_frame(changed, cfg)


def test_strict_prior_z_excludes_current_future_and_invalid_rows() -> None:
    clock = pd.Series(
        pd.date_range("2022-01-01", periods=6, freq="1h", tz="UTC")
    )
    values = np.array([0.0, 1.0, 2.0, 3.0, np.nan, 1_000.0])
    score, count = dfia.strict_prior_z(
        values, clock, lookback_hours=3, minimum=2, ddof=0
    )
    assert count.tolist() == [0, 1, 2, 3, 3, 2]
    assert score[3] == pytest.approx((3.0 - 1.0) / np.std([0.0, 1.0, 2.0]))
    assert np.isnan(score[4])
    changed = values.copy()
    changed[5] = -1_000_000.0
    changed_score, _ = dfia.strict_prior_z(
        changed, clock, lookback_hours=3, minimum=2, ddof=0
    )
    np.testing.assert_allclose(score[:5], changed_score[:5], equal_nan=True)


def test_gap_breaks_current_and_reference_memory_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, cfg = _small_source(hours=50, missing_offsets={20})
    captured: list[np.ndarray] = []

    def capture_z(values, timestamps, **kwargs):
        numeric = np.asarray(values, dtype=float)
        captured.append(numeric.copy())
        return np.zeros(len(numeric)), np.full(len(numeric), 20, dtype=np.int64)

    monkeypatch.setattr(dfia, "strict_prior_z", capture_z)
    panel = dfia.build_features(source, cfg)
    assert len(captured) == 2
    after_gap = panel.index[panel["timestamp"].eq(pd.Timestamp("2022-01-01 21:00Z"))][0]
    assert not panel.loc[after_gap, "memory_chain_ready"]
    assert np.isnan(captured[0][after_gap])
    recovered = panel.index[
        panel["timestamp"].eq(pd.Timestamp("2022-01-02 04:00Z"))
    ][0]
    assert panel.loc[recovered, "memory_chain_ready"]
    assert np.isfinite(captured[0][recovered])


def test_feature_side_requires_raw_and_standardized_absorption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    returns = np.zeros(60)
    returns[20] = -0.001
    returns[30] = 0.001
    returns[40] = 0.001
    source, cfg = _small_source(hours=60, index_returns=returns)
    impulse_z = np.zeros(60)
    index_z = np.zeros(60)
    impulse_z[[20, 40]] = 2.0
    impulse_z[30] = -2.0
    index_z[20] = -0.5
    index_z[30] = 0.5
    index_z[40] = -0.5
    values = iter((impulse_z, index_z))

    def fixed_z(*args, **kwargs):
        score = next(values)
        return score, np.full(len(score), 20, dtype=np.int64)

    monkeypatch.setattr(dfia, "strict_prior_z", fixed_z)
    panel = dfia.build_features(source, cfg)
    assert panel.loc[20, "side"] == -1
    assert panel.loc[30, "side"] == 1
    assert panel.loc[40, "side"] == 0
    assert panel.loc[20, "entry_time"] == pd.Timestamp(
        "2022-01-01 20:10Z"
    )
    assert panel.loc[20, "exit_time"] == pd.Timestamp(
        "2022-01-02 02:10Z"
    )


def _event_features() -> pd.DataFrame:
    source_timestamp = pd.to_datetime(
        [
            "2020-01-01 00:00Z",
            "2020-01-01 05:00Z",
            "2020-01-01 06:00Z",
            "2022-12-31 23:00Z",
            "2023-01-01 00:00Z",
        ],
        utc=True,
    )
    entry = source_timestamp + pd.Timedelta(minutes=10)
    frame = pd.DataFrame(
        {
            "timestamp": source_timestamp,
            "feature_available_at": source_timestamp
            + pd.Timedelta(minutes=5),
            "earliest_observable_open": source_timestamp
            + pd.Timedelta(minutes=5),
            "entry_time": entry,
            "exit_time": entry + pd.Timedelta(hours=6),
            "side": [-1, 1, 1, -1, 1],
            "candidate": True,
            "funding_impulse": 0.001,
            "index_return_1h": -0.001,
            "funding_impulse_z": 2.0,
            "index_return_z": -0.5,
            "impulse_reference_count": 360,
            "index_reference_count": 360,
            "memory_chain_ready": True,
        }
    )
    return frame


def test_schedule_contains_before_greedy_nonoverlap_and_counts_entries() -> None:
    clock = dfia.schedule_clock(_event_features(), dfia.Config())
    assert clock["split"].tolist() == ["train", "train", "test"]
    assert clock["entry_time"].tolist() == [
        pd.Timestamp("2020-01-01 00:10Z"),
        pd.Timestamp("2020-01-01 06:10Z"),
        pd.Timestamp("2023-01-01 00:10Z"),
    ]
    assert not clock["source_timestamp"].eq(
        pd.Timestamp("2022-12-31 23:00Z")
    ).any()


def _passing_clock() -> pd.DataFrame:
    entries: list[pd.Timestamp] = []
    for month in pd.period_range("2020-01", "2023-12", freq="M"):
        base = pd.Timestamp(month.start_time, tz="UTC")
        entries.extend(base + pd.to_timedelta([1, 5, 9, 13, 17, 21, 25], unit="D"))
    entry = pd.Series(pd.DatetimeIndex(entries), name="entry_time")
    split = np.where(
        entry < pd.Timestamp("2023-01-01", tz="UTC"), "train", "test"
    )
    side = np.where(np.arange(len(entry)) % 2, 1, -1)
    return pd.DataFrame({"entry_time": entry, "split": split, "side": side})


def test_support_enforces_density_dispersion_and_side_balance() -> None:
    clock = _passing_clock()
    source_quality = {"passed": True}
    summary = dfia.support_summary(clock, source_quality, dfia.Config())
    assert summary["counts"] == {
        "total_2020_2023": 336,
        "train_2020_2022": 252,
        "train_2020": 84,
        "train_2021": 84,
        "train_2022": 84,
        "test_2023": 84,
        "test_2023_h1": 42,
        "test_2023_h2": 42,
    }
    assert summary["active_months"] == 48
    assert summary["passed"] is True

    one_sided = clock.copy()
    one_sided["side"] = 1
    assert not dfia.support_summary(
        one_sided, source_quality, dfia.Config()
    )["passed"]
    missing_quarter = clock.loc[
        ~clock["entry_time"].between(
            pd.Timestamp("2022-04-01", tz="UTC"),
            pd.Timestamp("2022-07-01", tz="UTC"),
            inclusive="left",
        )
    ]
    rejected = dfia.support_summary(
        missing_quarter, source_quality, dfia.Config()
    )
    assert rejected["quarter_counts"]["2022Q2"] == 0
    assert rejected["passed"] is False


def test_source_quality_uses_exact_hour_and_month_denominators() -> None:
    timestamp = pd.date_range(
        dfia.SourceConfig.start,
        dfia.SourceConfig.end_exclusive,
        freq="1h",
        inclusive="left",
    )
    source = pd.DataFrame({"timestamp": timestamp})
    manifest = {
        "source_audit": {
            "memory_identity": {"all_windows_within_tolerance": True}
        },
        "outcome_boundary": dfia.EXPECTED_OUTCOME_BOUNDARY,
    }
    summary = dfia.source_quality_summary(source, manifest, dfia.Config())
    assert summary["expected_hours"] == 40_958
    assert summary["overall_coverage_ratio"] == 1.0
    assert min(summary["eligible_month_coverage_ratio"].values()) == 1.0
    assert summary["passed"] is True

    missing = source.loc[
        ~source["timestamp"].between(
            pd.Timestamp("2023-06-01 00:00Z"),
            pd.Timestamp("2023-06-03 00:00Z"),
            inclusive="left",
        )
    ].reset_index(drop=True)
    rejected = dfia.source_quality_summary(missing, manifest, dfia.Config())
    assert rejected["maximum_gap_hours"] == 49
    assert rejected["passed"] is False


def test_event_hash_binds_side_policy_preregistration_and_source() -> None:
    event = {
        "policy_id": dfia.POLICY_ID,
        "split": "train",
        "source_timestamp": "2022-01-01T00:00:00+00:00",
        "feature_available_at": "2022-01-01T00:05:00+00:00",
        "earliest_observable_open": "2022-01-01T00:05:00+00:00",
        "entry_time": "2022-01-01T00:10:00+00:00",
        "exit_time": "2022-01-01T06:10:00+00:00",
        "side": -1,
    }
    kwargs = {
        "cfg": dfia.Config(),
        "preregistration_hash": "prereg",
        "source_manifest_hash": "manifest",
        "source_sha256": "source",
    }
    baseline = dfia.event_clock_hash([event], **kwargs)
    assert dfia.event_clock_hash([{**event, "side": 1}], **kwargs) != baseline
    assert dfia.event_clock_hash(
        [event],
        **{**kwargs, "cfg": replace(dfia.Config(), hold_bars=73)},
    ) != baseline
    assert dfia.event_clock_hash(
        [event], **{**kwargs, "source_sha256": "other"}
    ) != baseline
