from __future__ import annotations

from dataclasses import replace
import gzip
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from training import build_block_feerate_breadth_transport_support as support


def _source_frame(
    rows: int = 180,
    *,
    start: str = "2023-07-20T12:00:00Z",
) -> pd.DataFrame:
    starts = pd.date_range(start, periods=rows, freq="12h", tz="UTC")
    records: list[dict[str, Any]] = []
    for index, bucket_start in enumerate(starts):
        level = 1_000 + int(120 * math.sin(index / 5.0)) + index // 3
        gaps = [
            10 + int(4 * (1.0 + math.sin(index / (2.0 + gap) + gap)))
            for gap in range(6)
        ]
        fees = [max(1, level)]
        for gap in gaps:
            fees.append(fees[-1] + gap)
        bucket_end = bucket_start + pd.Timedelta(hours=12)
        records.append(
            {
                "bucket_start_utc": bucket_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "bucket_end_utc": bucket_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "available_at_utc": (
                    bucket_end + pd.Timedelta(hours=48)
                ).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "avg_height": 800_000 + index,
                "avg_timestamp": int(bucket_start.timestamp()) + 1_800,
                "fee_p0": fees[0],
                "fee_p10": fees[1],
                "fee_p25": fees[2],
                "fee_p50": fees[3],
                "fee_p75": fees[4],
                "fee_p90": fees[5],
                "fee_p100": fees[6],
            }
        )
    return pd.DataFrame.from_records(records, columns=support.SOURCE_COLUMNS)


def _cfg(tmp_path: Path, **changes: Any) -> support.Config:
    cfg = support.Config(
        preregistration=str(support.DEFAULT_PREREGISTRATION),
        output=str(tmp_path / "support.json"),
        primary_clock=str(tmp_path / "primary.csv.gz"),
        control_clocks=str(tmp_path / "controls.csv.gz"),
        artifact_root=str(tmp_path),
    )
    return replace(cfg, **changes)


def _clock_record(entry: pd.Timestamp, side: int, window: str) -> dict[str, Any]:
    return {
        "policy_id": support.POLICY_ID,
        "clock": "primary",
        "window": window,
        "bucket_start_utc": entry - pd.Timedelta(hours=60, minutes=5),
        "source_available_at_utc": entry - pd.Timedelta(minutes=5),
        "entry_time_utc": entry,
        "exit_time_utc": entry + pd.Timedelta(hours=24),
        "side": side,
        "location": 0.1 * side,
        "signed_coherence": 0.8 * side,
        "coherence": 0.8,
        "tail_divergence": 0.1,
        "magnitude_rank": 0.9,
        "tail_divergence_rank": 0.2,
    }


def _passing_primary() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    side = 1
    for month in pd.period_range("2023-11", "2024-12", freq="M"):
        for day in (1, 5, 9, 13, 17, 21):
            entry = pd.Timestamp(
                year=month.year, month=month.month, day=day, hour=0, minute=5, tz="UTC"
            )
            rows.append(_clock_record(entry, side, "train"))
            side *= -1
    for month in pd.period_range("2025-01", "2025-12", freq="M"):
        for day in (2, 8, 14, 20):
            entry = pd.Timestamp(
                year=month.year, month=month.month, day=day, hour=0, minute=5, tz="UTC"
            )
            rows.append(_clock_record(entry, side, "test"))
            side *= -1
    for month in pd.period_range("2026-01", "2026-06", freq="M"):
        for day in (3, 9, 15, 21):
            entry = pd.Timestamp(
                year=month.year, month=month.month, day=day, hour=0, minute=5, tz="UTC"
            )
            rows.append(_clock_record(entry, side, "eval"))
            side *= -1
    return pd.DataFrame.from_records(rows, columns=support.CLOCK_COLUMNS)


def test_strict_prior_midrank_ties_and_empty_rejection() -> None:
    assert support.strict_prior_midrank(2.0, [1.0, 2.0, 2.0, 3.0]) == 0.5
    with pytest.raises(ValueError, match="requires prior"):
        support.strict_prior_midrank(1.0, [])


def test_features_use_two_bucket_transport_strict_prior_and_signed_direction() -> None:
    features = support.build_features(
        _source_frame(140), lookback=120, minimum_prior=120
    )
    first_ready = int(features.index[features["rank_ready"]][0])
    assert not features.loc[: first_ready - 1, "rank_ready"].any()
    valid_prior = features.loc[: first_ready - 1]
    valid_prior = valid_prior[valid_prior["feature_valid"]]
    assert len(valid_prior) >= 120
    prior = valid_prior["location"].abs().tolist()[-120:]
    expected = support.strict_prior_midrank(
        abs(float(features.loc[first_ready, "location"])), prior
    )
    assert features.loc[first_ready, "magnitude_rank"] == expected
    valid = features[features["feature_valid"]]
    assert np.array_equal(
        np.sign(valid["signed_coherence"].to_numpy()),
        np.sign(valid["location"].to_numpy()),
    )
    assert valid["coherence"].between(0.0, 1.0).all()
    assert features["entry_time_utc"].equals(
        features["available_at_utc"].dt.ceil("5min") + pd.Timedelta(minutes=5)
    )
    longer = support.build_features(
        _source_frame(150), lookback=120, minimum_prior=120
    )
    pd.testing.assert_frame_equal(
        features[support.FEATURE_COLUMNS],
        longer.loc[:139, support.FEATURE_COLUMNS].reset_index(drop=True),
    )


def test_candidate_side_boundaries_and_tail_partition() -> None:
    base = {
        "rank_ready": True,
        "magnitude_rank": 0.75,
        "coherence": 0.60,
        "tail_divergence_rank": 0.75,
        "signed_coherence": 0.8,
    }
    row = SimpleNamespace(**base)
    assert support._candidate_side(row, "primary") == 1
    assert support._candidate_side(row, "tail_only") == 0
    tail = SimpleNamespace(**{**base, "tail_divergence_rank": 0.7500001})
    assert support._candidate_side(tail, "primary") == 0
    assert support._candidate_side(tail, "tail_only") == 1
    short = SimpleNamespace(**{**base, "signed_coherence": -0.8})
    assert support._candidate_side(short, "primary") == -1


def test_feature_rejects_median_and_signed_coherence_direction_conflict() -> None:
    frame = _source_frame(3)
    for column in (
        "fee_p0",
        "fee_p10",
        "fee_p25",
        "fee_p50",
        "fee_p75",
        "fee_p90",
        "fee_p100",
    ):
        frame.loc[0, column] = 1_000
    frame.loc[2, [
        "fee_p0",
        "fee_p10",
        "fee_p25",
        "fee_p50",
        "fee_p75",
        "fee_p90",
        "fee_p100",
    ]] = [0, 1, 1, 1_001, 1_002, 1_003, 1_004]
    features = support.build_features(frame, lookback=2, minimum_prior=1)
    assert features.loc[2, "location"] > 0.0
    assert features.loc[2, "signed_coherence"] < 0.0
    assert not features.loc[2, "feature_valid"]


def test_chronological_clock_suppresses_overlapping_12h_candidates() -> None:
    features = support.build_features(_source_frame(300))
    mask = features["entry_time_utc"].between(
        pd.Timestamp("2023-11-01T00:00:00Z"),
        pd.Timestamp("2024-01-01T00:00:00Z"),
        inclusive="left",
    )
    features.loc[mask, "rank_ready"] = True
    features.loc[mask, "magnitude_rank"] = 1.0
    features.loc[mask, "coherence"] = 1.0
    features.loc[mask, "tail_divergence_rank"] = 0.0
    features.loc[mask, "signed_coherence"] = 1.0
    clock = support.build_clock(features, mode="primary", clock="primary")
    train = clock[clock["window"] == "train"]
    assert len(train) > 0
    assert (
        train["entry_time_utc"].iloc[1:].reset_index(drop=True)
        >= train["exit_time_utc"].iloc[:-1].reset_index(drop=True)
    ).all()


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda frame: frame.__setitem__(
                "available_at_utc", frame["bucket_end_utc"]
            ),
            "availability clock drift",
        ),
        (
            lambda frame: frame.loc.__setitem__(
                (3, "bucket_start_utc"), frame.loc[2, "bucket_start_utc"]
            ),
            "bucket starts must be unique",
        ),
        (
            lambda frame: frame.loc.__setitem__((4, "fee_p25"), 0),
            "percentile ordering drift",
        ),
        (
            lambda frame: frame.__setitem__("close", 1.0),
            "schema drift",
        ),
    ],
)
def test_source_contract_rejects_clock_schema_and_percentile_drift(
    mutator: Any, message: str
) -> None:
    frame = _source_frame(10)
    mutator(frame)
    with pytest.raises(RuntimeError, match=message):
        support.validate_source_frame(frame)


def test_source_contract_rejects_unordered_rows() -> None:
    frame = _source_frame(10)
    frame = frame.iloc[[0, 2, 1, *range(3, 10)]].reset_index(drop=True)
    with pytest.raises(RuntimeError, match="strictly increasing"):
        support.validate_source_frame(frame)


def test_support_gate_exact_pass_and_side_failure() -> None:
    source = support.validate_source_frame(_source_frame(20))
    primary = _passing_primary()
    summary = support.support_gate_summary(primary, source)
    assert summary["passed"] is True
    assert summary["counts"]["train"] == 84
    assert summary["counts"]["test"] == 48
    assert summary["counts"]["eval"] == 24
    assert summary["counts"]["2025Q1"] == 12

    one_sided = primary.copy()
    one_sided["side"] = 1
    failed = support.support_gate_summary(one_sided, source)
    assert failed["passed"] is False
    assert failed["checks"]["train_short_minimum"] is False
    assert failed["checks"]["test_short_minimum"] is False
    assert failed["checks"]["eval_short_minimum"] is False


def test_control_clocks_preserve_every_frozen_clock_contract() -> None:
    features = support.build_features(_source_frame(2_191))
    primary = support.build_clock(features, mode="primary", clock="primary")
    controls = support.build_control_clocks(features, primary)
    assert set(controls["clock"].unique()) == set(support.CONTROL_NAMES)

    by_clock = {
        name: controls[controls["clock"] == name].reset_index(drop=True)
        for name in support.CONTROL_NAMES
    }
    primary_keys = primary[["window", "bucket_start_utc"]].reset_index(drop=True)
    for name in (
        "direction_flip",
        "constant_long_same_clock",
        "constant_short_same_clock",
    ):
        pd.testing.assert_frame_equal(
            by_clock[name][["window", "bucket_start_utc"]], primary_keys
        )
    assert np.array_equal(
        by_clock["direction_flip"]["side"].to_numpy(),
        -primary["side"].to_numpy(),
    )
    assert by_clock["constant_long_same_clock"]["side"].eq(1).all()
    assert by_clock["constant_short_same_clock"]["side"].eq(-1).all()

    magnitude = by_clock["magnitude_only"]
    assert magnitude["magnitude_rank"].ge(0.75).all()
    tail = by_clock["tail_only"]
    assert tail["magnitude_rank"].ge(0.75).all()
    assert tail["coherence"].ge(0.60).all()
    assert tail["tail_divergence_rank"].gt(0.75).all()

    delayed = by_clock["one_bar_delayed_entry"]
    primary_keyed = primary.set_index(["window", "bucket_start_utc"])
    delayed_keyed = delayed.set_index(["window", "bucket_start_utc"])
    assert delayed_keyed.index.equals(primary_keyed.index)
    assert (
        delayed_keyed["entry_time_utc"] - primary_keyed["entry_time_utc"]
    ).eq(pd.Timedelta(minutes=5)).all()
    assert (
        delayed_keyed["exit_time_utc"] - primary_keyed["exit_time_utc"]
    ).eq(pd.Timedelta(minutes=5)).all()

    stale = by_clock["stale_7d"]
    feature_lookup = features.set_index("bucket_start_utc")
    for row in stale.itertuples(index=False):
        source_bucket = row.bucket_start_utc - pd.Timedelta(days=7)
        expected = feature_lookup.loc[source_bucket]
        for column in support.FEATURE_COLUMNS:
            assert getattr(row, column) == expected[column]

    random_clock = by_clock["month_side_stratified_random_clock"]
    primary_groups = primary.groupby(
        [primary["entry_time_utc"].dt.strftime("%Y-%m"), "side"]
    ).size()
    random_groups = random_clock.groupby(
        [random_clock["entry_time_utc"].dt.strftime("%Y-%m"), "side"]
    ).size()
    pd.testing.assert_series_equal(primary_groups, random_groups)

    independent = [
        "magnitude_only",
        "tail_only",
        "stale_7d",
        "month_side_stratified_random_clock",
        "one_bar_delayed_entry",
    ]
    for name in independent:
        clock = by_clock[name]
        for window, group in clock.groupby("window"):
            ordered = group.sort_values("entry_time_utc")
            assert (
                ordered["entry_time_utc"].iloc[1:].reset_index(drop=True)
                >= ordered["exit_time_utc"].iloc[:-1].reset_index(drop=True)
            ).all()
            assert all(
                support._window_name(row.entry_time_utc, row.exit_time_utc)
                == window
                for row in ordered.itertuples(index=False)
            )


def test_build_artifacts_is_source_only_deterministic_and_immutable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path)
    source = _source_frame(2_191)
    monkeypatch.setattr(support, "load_source_frame", lambda _: source)
    artifact = support.build_support_artifacts(cfg)
    assert artifact == json.loads(Path(cfg.output).read_text())
    core = {key: value for key, value in artifact.items() if key != "manifest_hash"}
    assert artifact["manifest_hash"] == support.canonical_hash(core)
    assert artifact["performance_values_opened"] is False
    assert artifact["outcome_boundary"] == support.OUTCOME_BOUNDARY
    assert artifact["source"]["rows_read"] == 2_191
    assert artifact["feature_audit"]["source_values_summarized"] is False
    assert set(artifact["artifacts"]["control_clocks"]["clock_counts"]) == set(
        support.CONTROL_NAMES
    )
    with gzip.open(cfg.primary_clock, "rt", encoding="utf-8") as handle:
        assert handle.readline().strip().split(",") == support.CLOCK_COLUMNS
    with gzip.open(cfg.control_clocks, "rt", encoding="utf-8") as handle:
        assert handle.readline().strip().split(",") == support.CLOCK_COLUMNS
    with pytest.raises(FileExistsError, match="immutable"):
        support.build_support_artifacts(cfg)


def test_frozen_preregistration_hash_and_output_path_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        support, "EXPECTED_PREREGISTRATION_FILE_SHA256", "0" * 64
    )
    with pytest.raises(RuntimeError, match="file SHA drift"):
        support.build_support_artifacts(_cfg(tmp_path))

    monkeypatch.setattr(
        support,
        "EXPECTED_PREREGISTRATION_FILE_SHA256",
        "73b06b94db2f844d993dcd76d6ad9e60f9b6d332d4c4d6ba8d41a3de970dae42",
    )
    aliased = _cfg(
        tmp_path / "alias",
        output=str(support.DEFAULT_PREREGISTRATION),
    )
    with pytest.raises(ValueError, match="aliases a frozen input"):
        support.build_support_artifacts(aliased)

    outside = _cfg(
        tmp_path / "outside",
        output=str(tmp_path / "escaped" / "support.json"),
    )
    outside = replace(outside, artifact_root=str(tmp_path / "outside"))
    with pytest.raises(ValueError, match="explicit artifact root"):
        support.build_support_artifacts(outside)
