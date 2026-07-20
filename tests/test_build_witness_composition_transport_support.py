from __future__ import annotations

from dataclasses import replace
import gzip
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from training import build_witness_composition_transport_support as support


def _source_frame(
    rows: int = 220,
    *,
    start: str = "2022-07-20T12:00:00Z",
) -> pd.DataFrame:
    starts = pd.date_range(start, periods=rows, freq="12h", tz="UTC")
    records: list[dict[str, Any]] = []
    for index, bucket_start in enumerate(starts):
        avg_size = 900_000 + 500 * index + int(20_000 * math.sin(index / 9.0))
        witness_bytes = 220_000 + int(60_000 * math.sin(index / 5.0))
        avg_weight = 4 * avg_size - 3 * witness_bytes
        bucket_end = bucket_start + pd.Timedelta(hours=12)
        records.append(
            {
                "bucket_start_utc": bucket_start,
                "bucket_end_utc": bucket_end,
                "available_at_utc": bucket_end + pd.Timedelta(hours=48),
                "avg_height": 745_000 + 72 * index,
                "avg_timestamp": int(bucket_start.timestamp()) + 3_600,
                "avg_size": avg_size,
                "avg_weight": avg_weight,
            }
        )
    return pd.DataFrame.from_records(records, columns=support.SOURCE_COLUMNS)


def _cfg(tmp_path: Path, **changes: Any) -> support.Config:
    root = tmp_path / "artifacts"
    cfg = support.Config(
        preregistration=str(support.DEFAULT_PREREGISTRATION),
        output=str(root / "support.json"),
        primary_clock=str(root / "primary.csv.gz"),
        control_clocks=str(root / "controls.csv.gz"),
        artifact_root=str(root),
    )
    return replace(cfg, **changes)


def _feature_row(entry: str, *, index: int, fullness_rank: float = 0.75) -> dict[str, Any]:
    entry_time = pd.Timestamp(entry)
    side = 1.0 if index % 4 < 2 else -1.0
    bucket_start = entry_time - pd.Timedelta(hours=60, minutes=5)
    values = {
        "witness_share": 0.30 + index * 0.0001,
        "fullness": 0.80,
        "transport_7d": side * (0.01 + index * 0.00001),
        "impulse_24h": side * 0.005,
        "log_size_7d": side * 0.02,
        "log_size_24h": side * 0.01,
        "log_weight_7d": side * 0.03,
        "log_weight_24h": side * 0.01,
        "magnitude_rank": 0.80,
        "fullness_rank": fullness_rank,
        "impulse_magnitude_rank": 0.80,
        "log_size_magnitude_rank": 0.80,
        "log_weight_magnitude_rank": 0.80,
    }
    return {
        "bucket_start_utc": bucket_start,
        "available_at_utc": entry_time - pd.Timedelta(minutes=5),
        "entry_time_utc": entry_time,
        "exit_time_utc": entry_time + pd.Timedelta(hours=24),
        "feature_valid": True,
        "rank_ready": True,
        **values,
    }


def _clock_frame() -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    windows = {
        "train": ("2022-11-02T00:05:00Z", "2023-12-28T00:05:00Z", 45),
        "test": ("2024-01-03T00:05:00Z", "2024-12-27T00:05:00Z", 35),
        "eval": ("2025-01-03T00:05:00Z", "2025-12-27T00:05:00Z", 35),
        "forward": ("2026-01-03T00:05:00Z", "2026-06-27T00:05:00Z", 18),
    }
    for window, (start, end, count) in windows.items():
        entries = pd.date_range(start, end, periods=count, tz="UTC")
        for index, entry in enumerate(entries):
            records.append(
                {
                    "window": window,
                    "entry_time_utc": entry,
                    "exit_time_utc": entry + pd.Timedelta(hours=24),
                    "side": 1 if index % 2 == 0 else -1,
                }
            )
    return pd.DataFrame.from_records(records)


def test_strict_prior_midrank_ties_and_empty_rejection() -> None:
    assert support.strict_prior_midrank(2.0, [1.0, 2.0, 2.0, 3.0]) == 0.5
    assert support.strict_prior_midrank(0.0, [1.0, 2.0]) == 0.0
    with pytest.raises(ValueError, match="requires prior"):
        support.strict_prior_midrank(1.0, [])


def test_features_use_exact_witness_algebra_strict_prior_and_clock() -> None:
    source = _source_frame(24)
    features = support.build_features(source, lookback=3, minimum_prior=2)
    row = features.iloc[16]
    current = source.iloc[16]
    lag2 = source.iloc[14]
    lag14 = source.iloc[2]
    witness = lambda value: (
        (4.0 * int(value.avg_size) - int(value.avg_weight))
        / (3.0 * int(value.avg_size))
    )
    expected_transport = witness(current) - witness(lag14)
    expected_impulse = witness(current) - witness(lag2)
    assert row["witness_share"] == pytest.approx(witness(current), abs=1e-15)
    assert row["transport_7d"] == pytest.approx(expected_transport, abs=1e-15)
    assert row["impulse_24h"] == pytest.approx(expected_impulse, abs=1e-15)
    assert row["fullness"] == pytest.approx(current.avg_weight / 4_000_000.0)
    assert bool(row["rank_ready"]) is True
    prior = features.loc[14:15, "transport_7d"].abs().tolist()
    assert row["magnitude_rank"] == support.strict_prior_midrank(
        abs(expected_transport), prior
    )
    expected_available = pd.Timestamp(current.available_at_utc)
    assert row["entry_time_utc"] == expected_available + pd.Timedelta(minutes=5)
    assert row["exit_time_utc"] == row["entry_time_utc"] + pd.Timedelta(hours=24)
    assert not bool(features.iloc[15]["rank_ready"])


def test_candidate_boundaries_and_component_modes() -> None:
    base = dict(_feature_row("2023-02-01T00:05:00Z", index=0))
    row = SimpleNamespace(**base)
    assert support._candidate_side(row, "primary") == 1
    base["magnitude_rank"] = 0.75
    base["fullness_rank"] = 0.50
    assert support._candidate_side(SimpleNamespace(**base), "primary") == 1
    base["impulse_24h"] = -0.01
    assert support._candidate_side(SimpleNamespace(**base), "primary") == 0
    assert support._candidate_side(SimpleNamespace(**base), "transport_only") == 1
    base["transport_7d"] = 0.0
    assert support._candidate_side(SimpleNamespace(**base), "transport_only") == 0

    low = dict(_feature_row("2023-02-01T00:05:00Z", index=3, fullness_rank=0.49))
    assert support._candidate_side(SimpleNamespace(**low), "low_fullness_complement") == -1
    assert support._candidate_side(SimpleNamespace(**low), "impulse_only") == 0
    low["fullness_rank"] = 0.50
    assert support._candidate_side(SimpleNamespace(**low), "impulse_only") == -1
    assert support._candidate_side(SimpleNamespace(**low), "serialized_size_only") == -1
    assert support._candidate_side(SimpleNamespace(**low), "block_weight_only") == -1
    with pytest.raises(ValueError, match="unknown WCTR"):
        support._candidate_side(SimpleNamespace(**low), "unknown")


def test_clock_nonoverlap_allows_entry_equal_exit_and_contains_splits() -> None:
    rows = [
        _feature_row("2023-02-01T00:05:00Z", index=0),
        _feature_row("2023-02-01T12:05:00Z", index=1),
        _feature_row("2023-02-02T00:05:00Z", index=2),
        _feature_row("2023-12-31T12:05:00Z", index=3),
    ]
    clock = support.build_clock(pd.DataFrame(rows), mode="primary", clock="primary")
    assert clock["entry_time_utc"].tolist() == [
        pd.Timestamp("2023-02-01T00:05:00Z"),
        pd.Timestamp("2023-02-02T00:05:00Z"),
    ]
    assert clock["window"].tolist() == ["train", "train"]


def test_control_clocks_are_deterministic_and_preserve_contracts() -> None:
    rows = [
        _feature_row(
            str(pd.Timestamp("2023-02-01T00:05:00Z") + pd.Timedelta(hours=12 * i)),
            index=i,
            fullness_rank=0.75 if i % 2 == 0 else 0.25,
        )
        for i in range(90)
    ]
    features = pd.DataFrame(rows)
    primary = support.build_clock(features, mode="primary", clock="primary")
    first = support.build_control_clocks(features, primary)
    second = support.build_control_clocks(features, primary)
    pd.testing.assert_frame_equal(first, second)
    assert set(first["clock"].unique()) == set(support.CONTROL_NAMES)

    flipped = first[first["clock"] == "direction_flip"].reset_index(drop=True)
    assert flipped["entry_time_utc"].tolist() == primary["entry_time_utc"].tolist()
    assert flipped["side"].tolist() == (-primary["side"]).tolist()
    constant_long = first[first["clock"] == "constant_long_same_clock"]
    constant_short = first[first["clock"] == "constant_short_same_clock"]
    assert constant_long["side"].eq(1).all()
    assert constant_short["side"].eq(-1).all()

    random_clock = first[
        first["clock"] == "month_side_stratified_random_clock"
    ].copy()
    baseline = support.build_clock(
        features, mode="rank_ready", clock="random_pool"
    ).assign(month=lambda value: value["entry_time_utc"].dt.strftime("%Y-%m"))
    for (window, month), target in primary.assign(
        month=primary["entry_time_utc"].dt.strftime("%Y-%m")
    ).groupby(["window", "month"]):
        sample = random_clock[
            (random_clock["window"] == window)
            & (random_clock["entry_time_utc"].dt.strftime("%Y-%m") == month)
        ]
        assert len(sample) == len(target)
        assert int((sample["side"] == 1).sum()) == int((target["side"] == 1).sum())
        assert int((sample["side"] == -1).sum()) == int((target["side"] == -1).sum())
        candidates = baseline[
            (baseline["window"] == window) & (baseline["month"] == month)
        ].copy()
        candidates["key"] = [
            support._random_key(window, month, value)
            for value in candidates["entry_time_utc"]
        ]
        expected = candidates.sort_values(
            ["key", "entry_time_utc"], kind="stable"
        ).head(len(target))
        assert set(sample["entry_time_utc"]) == set(expected["entry_time_utc"])
        long_count = int((target["side"] == 1).sum())
        expected_side = {
            entry: (1 if offset < long_count else -1)
            for offset, entry in enumerate(expected["entry_time_utc"])
        }
        assert {
            row.entry_time_utc: int(row.side)
            for row in sample.itertuples(index=False)
        } == expected_side

    stale = first[first["clock"] == "stale_7d"]
    assert not stale.empty
    stale_row = stale.iloc[0]
    current_index = int(
        features.index[features["bucket_start_utc"] == stale_row["bucket_start_utc"]][0]
    )
    lagged = features.iloc[current_index - 14]
    current = features.iloc[current_index]
    assert stale_row["entry_time_utc"] == current["entry_time_utc"]
    assert stale_row["source_available_at_utc"] == current["available_at_utc"]
    for column in support.FEATURE_COLUMNS:
        assert stale_row[column] == pytest.approx(lagged[column], abs=1e-15)

    normal = primary.iloc[[0]].copy()
    boundary = normal.copy()
    boundary["bucket_start_utc"] = pd.Timestamp("2023-12-28T12:00:00Z")
    boundary["source_available_at_utc"] = pd.Timestamp("2023-12-31T00:00:00Z")
    boundary["entry_time_utc"] = pd.Timestamp("2023-12-31T00:00:00Z")
    boundary["exit_time_utc"] = pd.Timestamp("2024-01-01T00:00:00Z")
    boundary["window"] = "train"
    delayed = support._delayed_clock(pd.concat([normal, boundary], ignore_index=True))
    assert len(delayed) == 1
    assert delayed.iloc[0]["entry_time_utc"] == (
        normal.iloc[0]["entry_time_utc"] + pd.Timedelta(minutes=5)
    )


def test_source_contract_rejects_schema_gap_clock_and_bip141_drift() -> None:
    source = _source_frame(30)
    assert len(support.validate_source_frame(source)) == 30

    extra = source.assign(close=1.0)
    with pytest.raises(RuntimeError, match="schema drift"):
        support.validate_source_frame(extra)

    gap = source.drop(index=10).reset_index(drop=True)
    with pytest.raises(RuntimeError, match="missing or irregular"):
        support.validate_source_frame(gap)

    clock = source.copy()
    clock.loc[5, "available_at_utc"] += pd.Timedelta(minutes=5)
    with pytest.raises(RuntimeError, match="availability clock"):
        support.validate_source_frame(clock)

    impossible = source.copy()
    impossible.loc[5, "avg_weight"] = impossible.loc[5, "avg_size"] - 1
    with pytest.raises(RuntimeError, match="BIP141"):
        support.validate_source_frame(impossible)

    fractional = source.copy()
    fractional["avg_size"] = fractional["avg_size"].astype(float)
    fractional.loc[5, "avg_size"] += 0.5
    with pytest.raises(RuntimeError, match="exact integers"):
        support.validate_source_frame(fractional)


def test_support_gate_exact_pass_and_requires_sides_for_every_control() -> None:
    clock = _clock_frame()
    source = _source_frame(30)
    summary = support.support_gate_summary(clock, source)
    assert summary["passed"] is True
    assert summary["counts"]["train"] == 45
    assert summary["counts"]["test"] == 35
    assert summary["counts"]["eval"] == 35
    assert summary["counts"]["forward"] == 18
    assert summary["checks"]["test_each_quarter_minimum"] is True
    assert summary["checks"]["eval_each_quarter_minimum"] is True

    all_long = clock.copy()
    all_long["side"] = 1
    failed = support.support_gate_summary(all_long, source)
    assert failed["passed"] is False
    assert failed["checks"]["train_short_minimum"] is False

    fixed = support.support_gate_summary(
        all_long, source, control_name="constant_long_same_clock"
    )
    assert fixed["passed"] is False
    assert fixed["waived_checks"] == []
    assert fixed["checks"]["train_short_minimum"] is False


def test_source_parser_uses_the_exact_bytes_it_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source_frame(30)
    compressed = gzip.compress(
        source.to_csv(index=False, lineterminator="\n").encode("utf-8"),
        mtime=0,
    )
    path = tmp_path / "source.csv.gz"
    path.write_bytes(compressed)
    registration = {
        "source_manifest": {
            "normalized_artifact": {
                "path": str(path),
                "bytes": len(compressed),
                "sha256": hashlib.sha256(compressed).hexdigest(),
            }
        }
    }
    original_read_bytes = Path.read_bytes

    def replace_after_read(candidate: Path) -> bytes:
        payload = original_read_bytes(candidate)
        if candidate.resolve() == path.resolve():
            candidate.write_bytes(b"replacement after exact read")
        return payload

    monkeypatch.setattr(Path, "read_bytes", replace_after_read)
    loaded = support.load_source_frame(registration)
    assert len(loaded) == len(source)
    assert path.read_text() == "replacement after exact read"


def test_build_artifacts_is_source_only_deterministic_and_immutable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source_frame(260)
    compressed = gzip.compress(
        source.to_csv(index=False, lineterminator="\n").encode("utf-8"),
        mtime=0,
    )
    source_path = tmp_path / "synthetic-source.csv.gz"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(compressed)
    source_binding = {
        "path": str(source_path),
        "bytes": len(compressed),
        "sha256": hashlib.sha256(compressed).hexdigest(),
    }
    reads: list[Path] = []
    original_read_bytes = Path.read_bytes

    def track_reads(path: Path) -> bytes:
        reads.append(path.resolve())
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", track_reads)
    monkeypatch.setattr(support, "_source_binding", lambda _: source_binding)
    cfg = _cfg(tmp_path / "first")
    artifact = support.build_support_artifacts(cfg)
    assert reads == [
        support._repository_path(support.DEFAULT_PREREGISTRATION),
        source_path.resolve(),
    ]
    assert (
        support._repository_path(support.prereg.EXPECTED_RAW_ARTIFACT["path"])
        not in reads
    )
    assert (
        support._repository_path(support.prereg.EXPECTED_NORMALIZED_ARTIFACT["path"])
        not in reads
    )
    core = {key: value for key, value in artifact.items() if key != "manifest_hash"}
    assert artifact["manifest_hash"] == support.canonical_hash(core)
    assert artifact["outcome_boundary"] == support.OUTCOME_BOUNDARY
    assert artifact["performance_values_opened"] is False
    assert artifact["feature_audit"]["source_values_summarized"] is False
    assert set(artifact["control_support_gates"]) == set(support.CONTROL_NAMES)
    assert Path(cfg.output).is_file()
    assert Path(cfg.primary_clock).is_file()
    assert Path(cfg.control_clocks).is_file()
    with gzip.open(cfg.primary_clock, "rt") as handle:
        assert handle.readline().strip().split(",") == support.CLOCK_COLUMNS

    cfg2 = _cfg(tmp_path / "second")
    artifact2 = support.build_support_artifacts(cfg2)
    assert artifact["artifacts"]["primary_clock"]["frame_hash"] == (
        artifact2["artifacts"]["primary_clock"]["frame_hash"]
    )
    assert artifact["artifacts"]["control_clocks"]["frame_hash"] == (
        artifact2["artifacts"]["control_clocks"]["frame_hash"]
    )
    with pytest.raises(FileExistsError, match="immutable"):
        support.build_support_artifacts(cfg)


def test_publish_failure_rolls_back_only_owned_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source_frame(180)
    monkeypatch.setattr(support, "load_source_frame", lambda _: source.copy())
    cfg = _cfg(tmp_path)
    original = support._publish_new
    calls = 0

    def fail_third(temporary: Path, final: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("injected publish failure")
        original(temporary, final)

    monkeypatch.setattr(support, "_publish_new", fail_third)
    with pytest.raises(OSError, match="injected"):
        support.build_support_artifacts(cfg)
    assert not Path(cfg.primary_clock).exists()
    assert not Path(cfg.control_clocks).exists()
    assert not Path(cfg.output).exists()


def test_frozen_preregistration_hash_and_output_path_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = support.validate_frozen_preregistration(
        support.DEFAULT_PREREGISTRATION
    )
    assert artifact["policy_hash"] == support.EXPECTED_POLICY_HASH

    monkeypatch.setattr(support, "EXPECTED_POLICY_HASH", "0" * 64)
    with pytest.raises(RuntimeError, match="policy hash drift"):
        support.validate_frozen_preregistration(support.DEFAULT_PREREGISTRATION)
    monkeypatch.undo()

    cfg = _cfg(tmp_path)
    registration = support.validate_frozen_preregistration(cfg.preregistration)
    with pytest.raises(ValueError, match="distinct"):
        support._validate_config(
            replace(cfg, control_clocks=cfg.primary_clock), registration
        )
    with pytest.raises(ValueError, match="artifact root"):
        support._validate_config(
            replace(cfg, output=str(tmp_path / "outside.json")), registration
        )
    with pytest.raises(ValueError, match="aliases"):
        support._validate_config(
            replace(cfg, output=str(support.DEFAULT_PREREGISTRATION)), registration
        )
