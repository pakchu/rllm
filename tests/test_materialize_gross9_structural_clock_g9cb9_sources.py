from __future__ import annotations

import gzip
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pytest

from training import materialize_gross9_structural_clock_g9cb9_sources as subject


def _write_csv(path: Path, frame: pd.DataFrame, *, compressed: bool) -> None:
    raw = frame.to_csv(index=False, lineterminator="\n").encode()
    if compressed:
        with path.open("wb") as destination:
            with gzip.GzipFile(filename="", mode="wb", fileobj=destination, compresslevel=9, mtime=0) as handle:
                handle.write(raw)
    else:
        path.write_bytes(raw)
    path.chmod(0o644)


def _fd_count() -> int:
    return len(os.listdir("/proc/self/fd"))


def _market(dates: pd.DatetimeIndex) -> pd.DataFrame:
    rows: dict[str, object] = {"date": dates}
    for index, column in enumerate(subject.MARKET_SCHEMA[1:], start=1):
        rows[column] = np.arange(len(dates), dtype=float) + index
    return pd.DataFrame(rows, columns=subject.MARKET_SCHEMA)


def _manifest() -> dict[str, object]:
    names = [
        "market_5m", "funding", "premium", "open_interest",
        "rex_taker_train", "rex_taker_test", "rex_taker_eval",
        "rex_veto_source",
    ]
    return {
        "schema_version": 1,
        "as_of": "2026-07-16",
        "sources": [
            {"name": name, "path": f"original/{name}", "sha256": f"{index:064x}", **({"rows": index} if name.startswith("rex_") else {})}
            for index, name in enumerate(names, start=1)
        ],
    }


def synthetic_config(tmp_path: Path, mutate: Callable[[dict[str, tuple[pd.DataFrame, bool]]], None] | None = None) -> subject.MaterializationConfig:
    for directory in ("inputs", "results", "data", "configs/shadow"):
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
    old_dates = pd.date_range("2026-01-01 00:00:00", periods=20, freq="5min")
    complete_dates = pd.date_range(old_dates[0], periods=23, freq="5min")
    old_market = _market(old_dates)
    replacement = _market(complete_dates)
    old_oi = pd.DataFrame({"date": old_dates, "open_interest": np.arange(20, dtype=float) + 100})
    metrics_dates = pd.date_range(old_dates[-1] - pd.Timedelta(minutes=10), complete_dates[-1], freq="5min")
    metrics_values = []
    old_map = dict(zip(old_dates, old_oi.open_interest, strict=True))
    for index, date in enumerate(metrics_dates):
        metrics_values.append(old_map.get(date, 200.0 + index))
    metrics = pd.DataFrame({
        "create_time": metrics_dates,
        "symbol": "BTCUSDT",
        "sum_open_interest": metrics_values,
        "sum_open_interest_value": 1.0,
        "count_toptrader_long_short_ratio": 1.0,
        "sum_toptrader_long_short_ratio": 1.0,
        "count_long_short_ratio": 1.0,
        "sum_taker_long_short_vol_ratio": 1.0,
    }, columns=subject.METRICS_SCHEMA)
    funding = pd.DataFrame({"date": [complete_dates[0], old_dates[-1]], "funding_rate": [0.001, 0.002]})
    premium = pd.DataFrame({"date": [complete_dates[0], old_dates[-1]], "premium_index": [0.1, 0.2]})
    rank = pd.DataFrame({
        "date": complete_dates,
        "spot_close": np.arange(23, dtype=float) + 1000,
        "spot_rows": 5.0,
        "premium_index_1m_close": np.arange(23, dtype=float) / 100,
        "premium_rows": 5.0,
        "unused": "bound-but-not-projected",
    })
    frames = {
        "old_market": (old_market, True),
        "replacement_market": (replacement, True),
        "funding": (funding, True),
        "premium": (premium, True),
        "old_open_interest": (old_oi, False),
        "binance_metrics_open_interest": (metrics, True),
        "rank7_spot_premium_5m": (rank, True),
    }
    if mutate is not None:
        mutate(frames)
    bindings = []
    for name, (frame, compressed) in frames.items():
        path = tmp_path / "inputs" / f"{name}.csv{'.gz' if compressed else ''}"
        _write_csv(path, frame, compressed=compressed)
        raw = path.read_bytes()
        bindings.append(subject.InputBinding(name, str(path), hashlib.sha256(raw).hexdigest(), len(raw), compressed))
    return subject.MaterializationConfig(
        root=tmp_path,
        inputs=tuple(bindings),
        attempt_path="results/attempt.json",
        market_output_path="data/market.csv.gz",
        oi_output_path="data/oi.csv.gz",
        manifest_output_path="configs/shadow/sources.json",
        support_output_path="results/support.json",
        inherited_manifest_path="unused.json",
        old_last=old_dates[-1],
        domain_end=complete_dates[-1] + pd.Timedelta(minutes=5),
        expected_old_rows=20,
        expected_complete_rows=23,
        expected_append_rows=3,
        splice_rows=3,
        rank7_tail_rows=5,
        inherited_manifest=_manifest(),
    )


def test_canonical_hash_and_frame_hash_vectors() -> None:
    assert subject.object_hash(
        {"identity": "G9CB-9", "version": subject.VERSION}, "support_hash"
    ) == "2d9cb35565c002c9b56f7c6236913e01870b6f5cae6835197c4d9a5ab9e21f7a"
    assert subject.canonical_json_bytes({"é": "한"}) == b'{"\xc3\xa9":"\xed\x95\x9c"}\n'
    vector = pd.DataFrame({"date": [pd.Timestamp("2026-01-01")], "value": [1]})
    _, csv_bytes, frame_hash = subject.serialize_csv_gzip(vector)
    assert csv_bytes == b"date,value\n2026-01-01 00:00:00,1\n"
    assert frame_hash == "8e362a6177525d72ecae23994c231bc666be1c61995aaddbf1afcaedc805b433"


def test_synthetic_success_exact_artifacts_and_counters(tmp_path: Path) -> None:
    config = synthetic_config(tmp_path)
    events: list[str] = []
    support = subject.materialize(config, failpoint=events.append)
    assert support["access"]["raw_source_decode_count"] == 7
    assert support["access"]["readback_decode_count"] == 2
    assert support["validation"]["appended_market_rows"] == 3
    assert support["validation"]["oi_splice_window_exact_rows"] == 3
    assert support["validation"]["rank7_spot_premium_raw_column_count"] == 6
    assert set(support) == {
        "access", "attempt_sentinel", "identity", "materialized_sources",
        "raw_sources", "source_manifest", "source_support_commit",
        "support_hash", "validation", "version",
    }
    assert set(support["access"]) == {
        "candidate_rows_opened", "comparator_clock_rows_opened",
        "decoded_generated_readbacks", "decoded_preexisting_sources",
        "economic_or_overlap_values_computed",
        "feature_signal_schedule_or_interval_values_computed",
        "model_history_or_rex_values_opened", "pre2025_anchor_value_rows_opened",
        "raw_source_decode_count", "readback_decode_count",
    }
    assert set(support["validation"]) == {
        "appended_market_rows", "appended_open_interest_rows",
        "domain_end_exclusive", "funding_attachment_tolerance",
        "funding_tail_available_rows", "latest_funding_available_after_causal_attachment",
        "latest_open_interest_positive_after_exact_join",
        "latest_premium_available_after_causal_attachment", "market_grid_seconds",
        "market_prefix_exact", "metrics_common_timestamp_values_exact",
        "oi_grid_seconds", "oi_splice_window_exact_rows", "oi_tail_exact_rows",
        "old_last_timestamp", "premium_attachment_tolerance",
        "premium_tail_available_rows", "rank7_spot_premium_exact_join_rows",
        "rank7_spot_premium_first_timestamp", "rank7_spot_premium_last_timestamp",
        "rank7_spot_premium_latest_values_valid",
        "rank7_spot_premium_projection_schema",
        "rank7_spot_premium_raw_column_count",
        "rank7_spot_premium_raw_schema_sha256",
        "rank7_spot_premium_tail_complete_rows", "required_last_timestamp",
    }
    assert all(
        set(row) == {
            "decoded_rows", "mode_octal", "name", "normalized_rows", "path",
            "path_type", "sha256", "size_bytes",
        }
        for row in support["raw_sources"]
    )
    assert [event for event in events if event.startswith("checkpoint:")] == [f"checkpoint:{index}" for index in range(1, 10)]
    assert {event for event in events if event.startswith(("before_link:", "after_link:"))} == {
        f"{edge}:{label}"
        for edge in ("before_link", "after_link")
        for label in ("attempt_sentinel", "materialized_market", "materialized_open_interest", "source_manifest", "source_support")
    }
    for relative in config.output_paths:
        info = os.lstat(tmp_path / relative)
        assert info.st_mode & 0o777 == 0o444
        assert info.st_nlink == 1
    manifest = json.loads((tmp_path / config.manifest_output_path).read_bytes())
    assert len(manifest["sources"]) == 9
    assert manifest["sources"][4]["name"] == "rank7_spot_premium_5m"
    expected_manifest = _manifest()
    expected_manifest["as_of"] = "2026-07-31"
    expected_manifest["sources"][0].update(
        path=config.market_output_path,
        sha256=support["materialized_sources"]["market_5m"]["sha256"],
    )
    expected_manifest["sources"][3].update(
        path=config.oi_output_path,
        sha256=support["materialized_sources"]["open_interest"]["sha256"],
    )
    rank_binding = config.inputs[-1]
    expected_manifest["sources"].insert(
        4,
        {"name": rank_binding.name, "path": rank_binding.path, "sha256": rank_binding.sha256},
    )
    assert manifest == expected_manifest
    attempt = json.loads((tmp_path / config.attempt_path).read_bytes())
    assert set(attempt) == {
        "attempt_hash", "branch", "expected_outputs", "identity", "one_shot",
        "raw_inputs", "repository_head", "repository_parent", "resume_allowed",
        "retry_allowed", "source_access_at_publication", "status", "topology",
        "version",
    }
    assert attempt["attempt_hash"] == subject.object_hash(attempt, "attempt_hash")
    assert attempt["expected_outputs"] == list(config.output_paths)
    assert attempt["source_access_at_publication"] == {
        "opaque_bytes_hashed": sum(row.size_bytes for row in config.inputs),
        "preexisting_sources_decoded": 0,
        "source_rows_decoded": 0,
    }
    assert all(
        set(row) == {
            "mode_octal", "name", "path", "path_type", "sha256", "size_bytes",
        }
        for row in attempt["raw_inputs"]
    )
    raw_support = (tmp_path / config.support_output_path).read_bytes()
    assert raw_support == subject.canonical_json_bytes(json.loads(raw_support))
    assert support["support_hash"] == subject.object_hash(support, "support_hash")


def test_serialization_is_deterministic() -> None:
    frame = pd.DataFrame({"date": [pd.Timestamp("2026-01-01")], "value": [1.25]})
    assert subject.serialize_csv_gzip(frame) == subject.serialize_csv_gzip(frame)


def test_generated_readback_preserves_adversarial_float_bytes(tmp_path: Path) -> None:
    config = synthetic_config(tmp_path)
    frame = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-01-01")],
            "value": [0.48842166193561359],
        }
    )
    gzip_bytes, csv_bytes, _ = subject.serialize_csv_gzip(frame)
    output = tmp_path / config.market_output_path
    output.write_bytes(gzip_bytes)
    output.chmod(0o444)

    digest, size, readback = subject._readback_generated(config, config.market_output_path)
    _, recsv, _ = subject.serialize_csv_gzip(readback)

    assert digest == hashlib.sha256(gzip_bytes).hexdigest()
    assert size == len(gzip_bytes)
    assert recsv == csv_bytes


def test_irrelevant_historical_metrics_and_rank7_gaps_are_accepted(tmp_path: Path) -> None:
    def add_irrelevant_gaps(frames: dict[str, tuple[pd.DataFrame, bool]]) -> None:
        first_market = frames["old_market"][0].date.iloc[0]
        early_dates = [
            first_market - pd.Timedelta(minutes=20),
            first_market - pd.Timedelta(minutes=10),
        ]

        metrics = frames["binance_metrics_open_interest"][0]
        early_metrics = metrics.iloc[:2].copy()
        early_metrics["create_time"] = early_dates
        early_metrics["sum_open_interest"] = [700.0, 701.0]
        _replace_frame(
            frames,
            "binance_metrics_open_interest",
            pd.concat([early_metrics, metrics], ignore_index=True),
        )

        rank = frames["rank7_spot_premium_5m"][0]
        early_rank = rank.iloc[:2].copy()
        early_rank["date"] = early_dates
        _replace_frame(
            frames,
            "rank7_spot_premium_5m",
            pd.concat([early_rank, rank], ignore_index=True),
        )

    config = synthetic_config(tmp_path, add_irrelevant_gaps)
    support = subject.materialize(config)

    normalized = {row["name"]: row["normalized_rows"] for row in support["raw_sources"]}
    assert normalized["binance_metrics_open_interest"] == 8
    assert normalized["rank7_spot_premium_5m"] == 25


def test_matching_historical_oi_missingness_outside_splice_is_preserved(tmp_path: Path) -> None:
    def add_matching_missingness(frames: dict[str, tuple[pd.DataFrame, bool]]) -> None:
        old_oi = frames["old_open_interest"][0].copy()
        old_oi.loc[0, "open_interest"] = np.nan
        _replace_frame(frames, "old_open_interest", old_oi)

        metrics = frames["binance_metrics_open_interest"][0]
        historical = metrics.iloc[[0]].copy()
        historical.loc[:, "create_time"] = old_oi.date.iloc[0]
        historical.loc[:, "sum_open_interest"] = np.nan
        _replace_frame(
            frames,
            "binance_metrics_open_interest",
            pd.concat([historical, metrics], ignore_index=True),
        )

    config = synthetic_config(tmp_path, add_matching_missingness)
    subject.materialize(config)

    materialized = pd.read_csv(tmp_path / config.oi_output_path)
    assert pd.isna(materialized.open_interest.iloc[0])


def test_malformed_oi_numeric_token_is_terminal(tmp_path: Path) -> None:
    def add_malformed_token(frames: dict[str, tuple[pd.DataFrame, bool]]) -> None:
        metrics = frames["binance_metrics_open_interest"][0].copy()
        values = metrics.sum_open_interest.astype(object)
        values.iloc[-1] = "not-a-number"
        metrics["sum_open_interest"] = values
        _replace_frame(frames, "binance_metrics_open_interest", metrics)

    config = synthetic_config(tmp_path, add_malformed_token)
    with pytest.raises(subject.SourceSupportFailure, match="malformed numeric value"):
        subject.materialize(config)
    assert (tmp_path / config.attempt_path).exists()


@pytest.mark.parametrize(
    ("label", "mutate", "match"),
    [
        ("market prefix", lambda f: f["replacement_market"][0].__setitem__("close", f["replacement_market"][0]["close"] + 1), "market logical prefix"),
        ("OI overlap", lambda f: f["binance_metrics_open_interest"][0].__setitem__("sum_open_interest", f["binance_metrics_open_interest"][0]["sum_open_interest"] + 1), "OI overlap"),
        ("tail OI", lambda f: f["binance_metrics_open_interest"][0].loc.__setitem__((f["binance_metrics_open_interest"][0].index[-1], "sum_open_interest"), 0), "tail OI"),
        ("Rank7 count", lambda f: f["rank7_spot_premium_5m"][0].loc.__setitem__((f["rank7_spot_premium_5m"][0].index[-1], "spot_rows"), 4), "Rank7 row counts"),
        ("funding null", lambda f: f["funding"][0].loc.__setitem__((0, "funding_rate"), np.nan), "funding contains null"),
        ("premium duplicate", lambda f: f["premium"][0].loc.__setitem__((1, "date"), f["premium"][0].loc[0, "date"]), "premium timestamp"),
    ],
)
def test_transform_rejections(tmp_path: Path, label: str, mutate: Callable[[dict[str, tuple[pd.DataFrame, bool]]], None], match: str) -> None:
    del label
    config = synthetic_config(tmp_path, mutate)
    with pytest.raises(subject.SourceSupportFailure, match=match):
        subject.materialize(config)
    assert (tmp_path / config.attempt_path).exists()


def _replace_frame(frames: dict[str, tuple[pd.DataFrame, bool]], name: str, frame: pd.DataFrame) -> None:
    frames[name] = (frame, frames[name][1])


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda f: _replace_frame(f, "old_market", f["old_market"][0].rename(columns={"close": "closing"})), "market schema drift"),
        (lambda f: f["replacement_market"][0].loc.__setitem__((1, "date"), f["replacement_market"][0].loc[0, "date"]), "duplicate timestamps"),
        (lambda f: f["replacement_market"][0].loc.__setitem__((slice(0, 1), "date"), list(reversed(f["replacement_market"][0].loc[0:1, "date"].tolist()))), "out-of-order timestamps"),
        (lambda f: _replace_frame(f, "replacement_market", f["replacement_market"][0].drop(index=10)), "market logical prefix|grid gap|row counts"),
        (lambda f: _replace_frame(f, "replacement_market", f["replacement_market"][0].iloc[:-1]), "row counts|terminal coverage"),
        (lambda f: _replace_frame(f, "old_open_interest", f["old_open_interest"][0].rename(columns={"open_interest": "oi"})), "open-interest schema drift"),
        (lambda f: _replace_frame(f, "binance_metrics_open_interest", f["binance_metrics_open_interest"][0].iloc[1:]), "missing OI splice anchor"),
        (lambda f: _replace_frame(f, "binance_metrics_open_interest", f["binance_metrics_open_interest"][0].drop(index=f["binance_metrics_open_interest"][0].index[-1])), "grid gap|missing tail OI"),
        (lambda f: _replace_frame(f, "rank7_spot_premium_5m", f["rank7_spot_premium_5m"][0].drop(columns="spot_close")), "projection column missing"),
        (lambda f: f["rank7_spot_premium_5m"][0].loc.__setitem__((1, "date"), f["rank7_spot_premium_5m"][0].loc[0, "date"]), "duplicate timestamps"),
        (lambda f: f["rank7_spot_premium_5m"][0].loc.__setitem__((1, "date"), f["rank7_spot_premium_5m"][0].loc[1, "date"] + pd.Timedelta(minutes=1)), "off-grid|out-of-order|grid gap"),
        (lambda f: _replace_frame(f, "rank7_spot_premium_5m", f["rank7_spot_premium_5m"][0].drop(index=10)), "incomplete Rank7 projection|grid gap"),
        (lambda f: f["rank7_spot_premium_5m"][0].loc.__setitem__((f["rank7_spot_premium_5m"][0].index[-1], "spot_close"), 0), "invalid latest Rank7"),
        (lambda f: f["funding"][0].loc.__setitem__((1, "date"), f["funding"][0].loc[0, "date"]), "funding timestamp"),
        (lambda f: f["funding"][0].__setitem__("date", f["funding"][0]["date"] - pd.Timedelta(days=2)), "funding tail"),
        (lambda f: f["premium"][0].loc.__setitem__((0, "premium_index"), np.nan), "premium contains null"),
        (lambda f: f["premium"][0].__setitem__("date", f["premium"][0]["date"] - pd.Timedelta(hours=4)), "premium tail"),
    ],
)
def test_additional_schema_grid_coverage_and_aux_rejections(
    tmp_path: Path,
    mutate: Callable[[dict[str, tuple[pd.DataFrame, bool]]], None],
    match: str,
) -> None:
    config = synthetic_config(tmp_path, mutate)
    with pytest.raises(subject.SourceSupportFailure, match=match):
        subject.materialize(config)
    assert (tmp_path / config.attempt_path).exists()


def test_raw_hash_mode_and_output_preexistence_gates_precede_sentinel(tmp_path: Path) -> None:
    config = synthetic_config(tmp_path)
    first = Path(config.inputs[0].path)
    first.write_bytes(first.read_bytes() + b"drift")
    with pytest.raises(subject.SourceSupportFailure, match="metadata drift"):
        subject.materialize(config)
    assert not (tmp_path / config.attempt_path).exists()

    config = synthetic_config(tmp_path / "second")
    output = config.root / config.market_output_path
    output.write_bytes(b"existing")
    with pytest.raises(subject.SourceSupportFailure, match="already exists"):
        subject.materialize(config)
    assert not (config.root / config.attempt_path).exists()


def test_same_size_raw_hash_drift_and_raw_type_drift_precede_sentinel(tmp_path: Path) -> None:
    config = synthetic_config(tmp_path / "hash")
    first = Path(config.inputs[0].path)
    raw = bytearray(first.read_bytes())
    raw[len(raw) // 2] ^= 1
    first.write_bytes(raw)
    with pytest.raises(subject.SourceSupportFailure, match="byte binding drift"):
        subject.materialize(config)
    assert not (config.root / config.attempt_path).exists()

    config = synthetic_config(tmp_path / "type")
    first = Path(config.inputs[0].path)
    first.unlink()
    first.mkdir()
    with pytest.raises(subject.SourceSupportFailure, match="metadata drift"):
        subject.materialize(config)
    assert not (config.root / config.attempt_path).exists()


def test_symlink_and_inode_alias_inputs_are_rejected(tmp_path: Path) -> None:
    config = synthetic_config(tmp_path / "symlink")
    original = Path(config.inputs[0].path)
    target = original.with_suffix(".target")
    original.rename(target)
    original.symlink_to(target)
    with pytest.raises(subject.SourceSupportFailure, match="cannot open bound raw input"):
        subject.materialize(config)

    config = synthetic_config(tmp_path / "hardlink")
    first, second = Path(config.inputs[0].path), Path(config.inputs[1].path)
    second.unlink()
    os.link(first, second)
    first_binding = config.inputs[0]
    aliased = replace(config.inputs[1], sha256=first_binding.sha256, size_bytes=first_binding.size_bytes)
    config = replace(config, inputs=(config.inputs[0], aliased, *config.inputs[2:]))
    with pytest.raises(subject.SourceSupportFailure, match="metadata drift|aliasing"):
        subject.materialize(config)


@pytest.mark.parametrize("mode", (0o600, 0o666))
def test_raw_mode_drift_is_rejected_before_sentinel(tmp_path: Path, mode: int) -> None:
    config = synthetic_config(tmp_path)
    Path(config.inputs[0].path).chmod(mode)
    with pytest.raises(subject.SourceSupportFailure, match="metadata drift"):
        subject.materialize(config)
    assert not (tmp_path / config.attempt_path).exists()


def test_output_symlink_and_input_parent_symlink_are_rejected(tmp_path: Path) -> None:
    config = synthetic_config(tmp_path / "output")
    output = config.root / config.market_output_path
    output.symlink_to(config.root / "elsewhere")
    with pytest.raises(subject.SourceSupportFailure, match="canonical output is a symlink"):
        subject.materialize(config)

    config = synthetic_config(tmp_path / "parent")
    inputs = config.root / "inputs"
    real_inputs = config.root / "real-inputs"
    inputs.rename(real_inputs)
    inputs.symlink_to(real_inputs, target_is_directory=True)
    with pytest.raises(subject.SourceSupportFailure, match="cannot open bound raw input"):
        subject.materialize(config)


@pytest.mark.parametrize("checkpoint", range(1, 10))
def test_every_checkpoint_failpoint_closes_descriptors_and_preserves_publications(tmp_path: Path, checkpoint: int) -> None:
    config = synthetic_config(tmp_path)
    baseline_fds = _fd_count()

    def inject(name: str) -> None:
        if name == f"checkpoint:{checkpoint}":
            raise RuntimeError(name)

    with pytest.raises(RuntimeError, match=f"checkpoint:{checkpoint}"):
        subject.materialize(config, failpoint=inject)
    assert _fd_count() == baseline_fds
    expected_count = (0, 1, 1, 2, 3, 3, 4, 5, 5)[checkpoint - 1]
    assert sum((tmp_path / relative).exists() for relative in config.output_paths) == expected_count
    for relative in config.output_paths:
        path = tmp_path / relative
        if path.exists():
            assert path.stat().st_mode & 0o777 == 0o444
            assert path.stat().st_nlink == 1


@pytest.mark.parametrize(
    ("edge", "label", "expected_count"),
    [
        (edge, label, index + (edge == "after_link"))
        for index, label in enumerate(("attempt_sentinel", "materialized_market", "materialized_open_interest", "source_manifest", "source_support"))
        for edge in ("before_link", "after_link")
    ],
)
def test_every_publication_link_failpoint_is_create_only_and_immutable(tmp_path: Path, edge: str, label: str, expected_count: int) -> None:
    config = synthetic_config(tmp_path)
    baseline_fds = _fd_count()

    def inject(name: str) -> None:
        if name == f"{edge}:{label}":
            raise RuntimeError(name)

    with pytest.raises(RuntimeError, match=f"{edge}:{label}"):
        subject.materialize(config, failpoint=inject)
    assert _fd_count() == baseline_fds
    existing = [tmp_path / relative for relative in config.output_paths if (tmp_path / relative).exists()]
    assert len(existing) == expected_count
    assert all((path.stat().st_mode & 0o777, path.stat().st_nlink) == (0o444, 1) for path in existing)


@pytest.mark.parametrize(
    ("phase", "label", "expected_count"),
    [
        (phase, label, index + (phase == "linked_canonical"))
        for index, label in enumerate(
            (
                "attempt_sentinel", "materialized_market", "materialized_open_interest",
                "source_manifest", "source_support",
            )
        )
        for phase in ("unnamed_initial", "unnamed_complete", "linked_canonical")
    ],
)
def test_every_publication_descriptor_failpoint_closes_cleanly(
    tmp_path: Path, phase: str, label: str, expected_count: int
) -> None:
    config = synthetic_config(tmp_path)
    baseline_fds = _fd_count()

    def inject(name: str) -> None:
        if name == f"publication:{label}:{phase}":
            raise RuntimeError(name)

    with pytest.raises(RuntimeError, match=f"publication:{label}:{phase}"):
        subject.materialize(config, failpoint=inject)
    assert _fd_count() == baseline_fds
    assert sum((tmp_path / relative).exists() for relative in config.output_paths) == expected_count


@pytest.mark.parametrize("relative_attribute", ("market_output_path", "oi_output_path"))
def test_generated_readback_descriptor_failpoints_close_cleanly(
    tmp_path: Path, relative_attribute: str
) -> None:
    config = synthetic_config(tmp_path)
    relative = getattr(config, relative_attribute)
    baseline_fds = _fd_count()

    def inject(name: str) -> None:
        if name == f"readback:{relative}:authenticated":
            raise RuntimeError(name)

    with pytest.raises(RuntimeError, match="readback:.*:authenticated"):
        subject.materialize(config, failpoint=inject)
    assert _fd_count() == baseline_fds
    assert sum((tmp_path / path).exists() for path in config.output_paths) == 3


@pytest.mark.parametrize(
    "relative_attribute",
    (
        "attempt_path", "market_output_path", "oi_output_path",
        "manifest_output_path", "support_output_path",
    ),
)
def test_final_output_descriptor_failpoints_close_cleanly(
    tmp_path: Path, relative_attribute: str
) -> None:
    config = synthetic_config(tmp_path)
    relative = getattr(config, relative_attribute)
    baseline_fds = _fd_count()

    def inject(name: str) -> None:
        if name == f"output:{relative}:final":
            raise RuntimeError(name)

    with pytest.raises(RuntimeError, match="output:.*:final"):
        subject.materialize(config, failpoint=inject)
    assert _fd_count() == baseline_fds
    assert all((tmp_path / path).exists() for path in config.output_paths)


@pytest.mark.parametrize(
    ("source_name", "phase", "expected_count"),
    [
        (
            source_name,
            phase,
            0 if phase in ("before_hash", "after_open_hash") else 5 if phase == "final" else 1,
        )
        for source_name in (
            "old_market", "replacement_market", "funding", "premium",
            "old_open_interest", "binance_metrics_open_interest",
            "rank7_spot_premium_5m",
        )
        for phase in (
            "before_hash", "after_open_hash", "before_decode", "after_decode",
            "after_rehash", "final",
        )
    ],
)
def test_every_descriptor_identity_failpoint_closes_cleanly(
    tmp_path: Path, source_name: str, phase: str, expected_count: int
) -> None:
    config = synthetic_config(tmp_path)
    baseline_fds = _fd_count()

    def inject(name: str) -> None:
        if name == f"identity:{source_name}:{phase}":
            raise RuntimeError(name)

    with pytest.raises(RuntimeError, match=f"identity:{source_name}:{phase}"):
        subject.materialize(config, failpoint=inject)
    assert _fd_count() == baseline_fds
    assert sum((tmp_path / relative).exists() for relative in config.output_paths) == expected_count


def test_retained_descriptor_path_edge_replacement_is_terminal(tmp_path: Path) -> None:
    config = synthetic_config(tmp_path)
    source = Path(config.inputs[0].path)
    moved = source.with_suffix(source.suffix + ".moved")
    replaced = False

    def inject(name: str) -> None:
        nonlocal replaced
        if name == "identity:old_market:before_decode" and not replaced:
            source.rename(moved)
            source.write_bytes(moved.read_bytes())
            source.chmod(0o644)
            replaced = True

    with pytest.raises(subject.SourceSupportFailure, match="pathname replacement"):
        subject.materialize(config, failpoint=inject)
    assert (tmp_path / config.attempt_path).exists()


def test_retained_descriptor_metadata_drift_is_terminal(tmp_path: Path) -> None:
    config = synthetic_config(tmp_path)
    source = Path(config.inputs[0].path)
    changed = False

    def inject(name: str) -> None:
        nonlocal changed
        if name == "identity:old_market:before_decode" and not changed:
            source.chmod(0o600)
            changed = True

    with pytest.raises(subject.SourceSupportFailure, match="retained descriptor identity drift"):
        subject.materialize(config, failpoint=inject)

    assert changed is True
    assert (tmp_path / config.attempt_path).exists()


def test_each_raw_input_leaf_is_opened_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = synthetic_config(tmp_path)
    input_paths = {Path(row.path).absolute() for row in config.inputs}
    counts = {path: 0 for path in input_paths}
    original = subject._open_nofollow_components

    def tracked(path: Path, flags: int) -> int:
        absolute = Path(path).absolute()
        if absolute in counts:
            counts[absolute] += 1
        return original(path, flags)

    monkeypatch.setattr(subject, "_open_nofollow_components", tracked)
    subject.materialize(config)

    assert set(counts.values()) == {1}


def test_final_directory_delta_rejects_unauthorized_residue(tmp_path: Path) -> None:
    config = synthetic_config(tmp_path)
    created = False

    def inject(name: str) -> None:
        nonlocal created
        if name == "identity:rank7_spot_premium_5m:final" and not created:
            (tmp_path / "results/unauthorized.tmp").write_bytes(b"residue")
            created = True

    with pytest.raises(subject.SourceSupportFailure, match="unauthorized output or temporary residue"):
        subject.materialize(config, failpoint=inject)

    assert created is True


def test_final_output_reauthentication_rejects_same_inode_byte_drift(tmp_path: Path) -> None:
    config = synthetic_config(tmp_path)
    changed = False

    def inject(name: str) -> None:
        nonlocal changed
        if name == "identity:rank7_spot_premium_5m:final" and not changed:
            support = tmp_path / config.support_output_path
            raw = bytearray(support.read_bytes())
            raw[len(raw) // 2] ^= 1
            support.chmod(0o644)
            support.write_bytes(raw)
            support.chmod(0o444)
            changed = True

    with pytest.raises(subject.SourceSupportFailure, match="final output binding differs"):
        subject.materialize(config, failpoint=inject)

    assert changed is True
    assert (tmp_path / config.support_output_path).stat().st_nlink == 1
    assert (tmp_path / config.support_output_path).stat().st_mode & 0o777 == 0o444


def test_import_has_no_source_or_output_access_and_official_entry_is_not_called() -> None:
    # Importing the module above only defines constants/functions.  In
    # particular, the frozen official paths remain strings until main() runs.
    assert subject.official_config().enforce_repository_gates is True
    assert subject.main is not subject.materialize


def test_official_configuration_is_frozen_without_opening_sources() -> None:
    config = subject.official_config()
    assert config.branch == subject.BRANCH
    assert config.repository_parent == subject.T8
    assert config.authority_commit == subject.A9
    assert [
        (row.name, row.path, row.sha256, row.size_bytes, row.compressed, row.mode)
        for row in config.inputs
    ] == [
        ("old_market", "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz", "a77cd0ae5b88b3c95e509d8d2610773d34af3afdc9170c63d88564bc3d0b990c", 66_696_659, True, 0o644),
        ("replacement_market", "/home/pakchu/rllm/data/cache_market_ext_5m_wavefull_2020-01-01_2026-07-05_dbappend.csv.gz", "0447a2c89926a1deebdfd495edde069a697d9481bc5936bc360c8c1488de2ebe", 65_420_089, True, 0o644),
        ("funding", "data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz", "4d381be086e275bacaf31df431dc31307a71a26b3947b7082efffc10bb129dd7", 89_326, True, 0o644),
        ("premium", "data/binance_um_aux_btc_2020_2026/BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz", "b45fcc5a3cf75c8e594effe61a698c4652f841b1d304107e9669524e0fc9d0d7", 1_196_481, True, 0o644),
        ("old_open_interest", "/tmp/btcusdt_open_interest_5m_2020_2026.csv", "e08f93033e56959e8e7a9c1e21f27c5f01efc8d06fa6b4fbbfe7354697122b31", 19_657_777, False, 0o644),
        ("binance_metrics_open_interest", "/home/pakchu/rllm/data/binance_um_metrics_BTCUSDT_5m_2020-09-01_2026-06-01.csv.gz", "d391022352d5b14dea7ffd207a9d1f84f603d06ddae42da55dd792f722fc0106", 21_440_132, True, 0o644),
        ("rank7_spot_premium_5m", "/home/pakchu/rllm/data/cache_spot_premium_5m_2020-01-01_2026-06-01.csv.gz", "c21ed3f52804a3f879ef5167b6b81ee0fd7dd262f2e9d87ee5c1c25dbad73617", 15_772_146, True, 0o644),
    ]
    assert sum(row.size_bytes for row in config.inputs) == 190_272_610
    assert config.output_paths == (
        "results/gross9_structural_clock_bundle_g9cb9_source_support_attempt_consumed_2026-07-31.json",
        "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_g9cb9_complete.csv.gz",
        "data/btcusdt_open_interest_5m_2020-01-01_2026-06-01_g9cb9_complete.csv.gz",
        "configs/shadow/gross9_structural_clock_bundle_g9cb9_sources_2026-07-31.json",
        "results/gross9_structural_clock_bundle_g9cb9_source_support_2026-07-31.json",
    )
    assert (config.expected_old_rows, config.expected_complete_rows, config.expected_append_rows) == (674_785, 674_892, 107)
    assert config.splice_rows == 13
    assert config.rank7_tail_rows == 3000


def test_official_command_shape_and_bytecode_preflight_without_source_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = subject.official_config()
    monkeypatch.chdir(config.root)
    monkeypatch.setattr(subject.sys, "argv", ["materialize_g9cb9"])
    monkeypatch.setattr(subject.sys, "dont_write_bytecode", True)
    monkeypatch.setenv("PYTHONPATH", str(config.root))
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")

    subject._validate_command_and_root(config)
    subject._validate_bytecode_first_gate(config.root)

    monkeypatch.setattr(subject.sys, "argv", ["materialize_g9cb9", "unexpected"])
    with pytest.raises(subject.SourceSupportFailure, match="accepts no arguments"):
        subject._validate_command_and_root(config)


def test_synthetic_official_gate_order_and_final_bytecode_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(synthetic_config(tmp_path), enforce_repository_gates=True)
    events: list[str] = []
    original_absence = subject._validate_output_absence
    original_open_inputs = subject._open_inputs

    monkeypatch.setattr(subject, "_validate_command_and_root", lambda _config: events.append("command"))
    monkeypatch.setattr(subject, "_validate_bytecode_first_gate", lambda _root: events.append("bytecode"))

    def git_gate(_config: subject.MaterializationConfig) -> str:
        events.append("git")
        return "synthetic-gated-s9"

    monkeypatch.setattr(subject, "_validate_git_gate", git_gate)
    monkeypatch.setattr(subject, "_validate_t8_gate", lambda _config: events.append("t8"))

    def output_absence(bound: subject.MaterializationConfig) -> None:
        events.append("output_absence")
        original_absence(bound)

    def open_inputs(
        bound: subject.MaterializationConfig, failpoint: Callable[[str], None]
    ) -> dict[str, subject._Retained]:
        events.append("open_inputs")
        return original_open_inputs(bound, failpoint)

    monkeypatch.setattr(subject, "_validate_output_absence", output_absence)
    monkeypatch.setattr(subject, "_open_inputs", open_inputs)

    support = subject.materialize(config, failpoint=events.append)

    assert events[:6] == ["command", "bytecode", "git", "t8", "output_absence", "open_inputs"]
    assert events[-2:] == ["bytecode", "checkpoint:9"]
    assert support["source_support_commit"] == "synthetic-gated-s9"


def test_first_bytecode_gate_failure_prevents_all_later_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(synthetic_config(tmp_path), enforce_repository_gates=True)
    events: list[str] = []
    monkeypatch.setattr(subject, "_validate_command_and_root", lambda _config: events.append("command"))

    def fail_bytecode(_root: Path) -> None:
        events.append("bytecode")
        raise subject.SourceSupportFailure("bytecode gate")

    monkeypatch.setattr(subject, "_validate_bytecode_first_gate", fail_bytecode)
    monkeypatch.setattr(subject, "_validate_git_gate", lambda _config: events.append("git"))

    with pytest.raises(subject.SourceSupportFailure, match="bytecode gate"):
        subject.materialize(config)

    assert events == ["command", "bytecode"]
    assert not any((tmp_path / relative).exists() for relative in config.output_paths)


def test_git_gate_authenticates_topology_cleanliness_and_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = synthetic_config(tmp_path)
    s9 = "9" * 40
    mode = {"value": "100644"}

    def fake_git(_root: Path, *args: str) -> str:
        if args == ("rev-parse", "HEAD"):
            return s9
        if args == ("branch", "--show-current"):
            return subject.BRANCH
        if args == ("rev-parse", "@{upstream}"):
            return s9
        if args == ("rev-list", "--parents", "-n", "1", s9):
            return f"{s9} {subject.T8}"
        if args == ("rev-parse", f"{subject.T8}^"):
            return subject.A9
        if args == ("diff", "--name-status", subject.T8, s9):
            return "A\ttests/test_materialize_gross9_structural_clock_g9cb9_sources.py\nA\ttraining/materialize_gross9_structural_clock_g9cb9_sources.py"
        if args[:3] == ("ls-files", "-s", "--"):
            relative = args[3]
            return f"{mode['value']} {'a' * 40} 0 {relative}"
        if args == ("status", "--porcelain=v1", "--untracked-files=all"):
            return ""
        raise AssertionError(f"unexpected git command: {args}")

    monkeypatch.setattr(subject, "_run_git", fake_git)
    assert subject._validate_git_gate(config) == s9

    mode["value"] = "100755"
    with pytest.raises(subject.SourceSupportFailure, match="Git mode or index binding"):
        subject._validate_git_gate(config)


@pytest.mark.parametrize(
    "residue_name",
    (
        ".g9cb8-otmpfile-probe-test",
        ".gross9-structural-clock-g9cb8-worker-other",
        ".gross9_structural_clock_bundle_g9cb8_manifest_2026-07-31.json.stage-test",
    ),
)
def test_t8_gate_rejects_every_forbidden_helper_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, residue_name: str
) -> None:
    config = synthetic_config(tmp_path)
    repository = Path(subject.__file__).resolve().parents[1]
    for relative, _size, _digest, _blob in subject._T8_EVIDENCE:
        target = tmp_path / relative
        target.write_bytes((repository / relative).read_bytes())
        target.chmod(0o444)
    retained = tmp_path / "results/.gross9-structural-clock-g9cb8-worker-b04b561d045e074567a96761"
    retained.mkdir(mode=0o700)
    retained.chmod(0o700)

    def fake_git(_root: Path, *args: str) -> str:
        relative = args[-1]
        blob = next(row[3] for row in subject._T8_EVIDENCE if row[0] == relative)
        return f"100644 {blob} 0 {relative}"

    monkeypatch.setattr(subject, "_run_git", fake_git)
    subject._validate_t8_gate(config)

    (tmp_path / "results" / residue_name).write_bytes(b"forbidden")
    with pytest.raises(subject.SourceSupportFailure, match="forbidden G9CB-8 helper"):
        subject._validate_t8_gate(config)
