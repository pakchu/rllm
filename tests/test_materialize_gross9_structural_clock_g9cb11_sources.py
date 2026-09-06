from __future__ import annotations

import ast
import builtins
import copy
import csv
import errno
import gzip
import hashlib
import io
import json
import os
import stat
from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pytest

from training import materialize_gross9_structural_clock_g9cb11_sources as subject


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

def _published_snapshot(
    config: subject.MaterializationConfig,
) -> dict[str, tuple[bytes, int, int, int, int]]:
    snapshot: dict[str, tuple[bytes, int, int, int, int]] = {}
    for relative in config.output_paths:
        path = config.root / relative
        if not path.exists():
            continue
        info = os.lstat(path)
        snapshot[relative] = (
            path.read_bytes(),
            info.st_mode & 0o777,
            info.st_nlink,
            info.st_dev,
            info.st_ino,
        )
    return snapshot

def _replace_bound_input_bytes(
    config: subject.MaterializationConfig, name: str, raw: bytes
) -> subject.MaterializationConfig:
    bindings = list(config.inputs)
    index = next(index for index, row in enumerate(bindings) if row.name == name)
    path = Path(bindings[index].path)
    path.write_bytes(raw)
    path.chmod(bindings[index].mode)
    bindings[index] = replace(
        bindings[index], sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw)
    )
    return replace(config, inputs=tuple(bindings))

def _rewrite_replacement_csv(
    config: subject.MaterializationConfig,
    mutate: Callable[[list[str]], None],
) -> subject.MaterializationConfig:
    binding = next(row for row in config.inputs if row.name == "replacement_market")
    lines = gzip.decompress(Path(binding.path).read_bytes()).decode("utf-8").splitlines()
    mutate(lines)
    csv_raw = ("\n".join(lines) + "\n").encode("utf-8")
    buffer = io.BytesIO()
    with gzip.GzipFile(
        filename="", mode="wb", fileobj=buffer, compresslevel=9, mtime=0
    ) as handle:
        handle.write(csv_raw)
    return _replace_bound_input_bytes(config, "replacement_market", buffer.getvalue())

def _encode_csv_row(row: list[str]) -> str:
    buffer = io.StringIO(newline="")
    csv.writer(buffer, lineterminator="\n", quoting=csv.QUOTE_MINIMAL).writerow(row)
    return buffer.getvalue().removesuffix("\n")

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
        old_last=old_dates[-1],
        domain_end=complete_dates[-1] + pd.Timedelta(minutes=5),
        expected_old_rows=20,
        expected_complete_rows=23,
        expected_append_rows=3,
        splice_rows=3,
        rank7_tail_rows=5,
        inherited_manifest=_manifest(),
    )

def test_serialization_is_deterministic() -> None:
    frame = pd.DataFrame({"date": [pd.Timestamp("2026-01-01")], "value": [1.25]})
    assert subject.serialize_csv_gzip(frame) == subject.serialize_csv_gzip(frame)

def test_independent_materializations_emit_identical_generated_gzip(tmp_path: Path) -> None:
    first = synthetic_config(tmp_path / "first")
    second = synthetic_config(tmp_path / "second")

    subject.materialize(first)
    subject.materialize(second)

    for attribute in ("market_output_path", "oi_output_path"):
        first_raw = (first.root / getattr(first, attribute)).read_bytes()
        second_raw = (second.root / getattr(second, attribute)).read_bytes()
        assert first_raw == second_raw
        assert hashlib.sha256(first_raw).hexdigest() == hashlib.sha256(second_raw).hexdigest()

def test_replacement_overlap_values_are_never_compared_or_selected(tmp_path: Path) -> None:
    def diverge_prefix(frames: dict[str, tuple[pd.DataFrame, bool]]) -> None:
        replacement = frames["replacement_market"][0].copy()
        replacement.loc[:19, "close"] = replacement.loc[:19, "close"] + 10_000
        _replace_frame(frames, "replacement_market", replacement)

    config = synthetic_config(tmp_path, diverge_prefix)
    support = subject.materialize(config)

    output = pd.read_csv(tmp_path / config.market_output_path)
    old = pd.read_csv(config.inputs[0].path)
    replacement = pd.read_csv(config.inputs[1].path)
    pd.testing.assert_frame_equal(
        output.iloc[:20].reset_index(drop=True), old, check_exact=True, check_dtype=False
    )
    pd.testing.assert_frame_equal(
        output.iloc[20:].reset_index(drop=True),
        replacement.iloc[20:].reset_index(drop=True),
        check_exact=True,
        check_dtype=False,
    )
    assert support["validation"]["market_partition"]["replacement_prefix_rows_selected"] == 0

def test_replacement_prefix_non_date_poison_is_not_semantically_evaluated(tmp_path: Path) -> None:
    def poison_prefix(frames: dict[str, tuple[pd.DataFrame, bool]]) -> None:
        replacement = frames["replacement_market"][0].copy()
        replacement["kimchi_premium"] = replacement["kimchi_premium"].astype(object)
        replacement.loc[7, "kimchi_premium"] = "poison-that-must-not-be-parsed"
        _replace_frame(frames, "replacement_market", replacement)

    config = synthetic_config(tmp_path, poison_prefix)
    support = subject.materialize(config)

    output = pd.read_csv(tmp_path / config.market_output_path)
    old = pd.read_csv(config.inputs[0].path)
    assert output.loc[7, "kimchi_premium"] == old.loc[7, "kimchi_premium"]
    assert support["validation"]["market_partition"]["replacement_prefix_rows_selected"] == 0

def test_pandas_decodes_only_rebuilt_replacement_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    poison = "prefix-token-that-must-never-reach-pandas"

    def poison_prefix(frames: dict[str, tuple[pd.DataFrame, bool]]) -> None:
        replacement = frames["replacement_market"][0].copy()
        replacement["kimchi_premium"] = replacement["kimchi_premium"].astype(object)
        replacement.loc[3, "kimchi_premium"] = poison
        _replace_frame(frames, "replacement_market", replacement)

    config = synthetic_config(tmp_path, poison_prefix)
    original = subject.pd.read_csv
    stringio_payloads: list[str] = []

    def audited_read_csv(source: object, *args: object, **kwargs: object) -> pd.DataFrame:
        if isinstance(source, io.StringIO):
            payload = source.getvalue()
            stringio_payloads.append(payload)
            assert poison not in payload
        return original(source, *args, **kwargs)

    monkeypatch.setattr(subject.pd, "read_csv", audited_read_csv)
    subject.materialize(config)

    assert len(stringio_payloads) == 1
    assert len(stringio_payloads[0].splitlines()) == 4

def test_replacement_uses_exact_date_scan_then_tail_decode_events(tmp_path: Path) -> None:
    config = synthetic_config(tmp_path)
    events: list[str] = []
    support = subject.materialize(config, failpoint=events.append)

    replacement_events = [event for event in events if "replacement_market" in event]
    assert replacement_events == [
        "identity:replacement_market:before_hash",
        "identity:replacement_market:after_open_hash",
        "identity:replacement_market:before_date_scan",
        "identity:replacement_market:after_date_scan",
        "identity:replacement_market:after_date_scan_rehash",
        "identity:replacement_market:before_tail_decode",
        "identity:replacement_market:after_tail_decode",
        "identity:replacement_market:after_tail_rehash",
        "identity:replacement_market:final",
    ]
    assert support["access_ledger"]["current_s11"]["access"]["decode_passes"] == [
        "old_market",
        "replacement_market_date_scan",
        "replacement_market_tail",
        "funding",
        "premium",
        "old_open_interest",
        "binance_metrics_open_interest_date_scan",
        "binance_metrics_open_interest_selected_window",
        "rank7_spot_premium_5m",
    ]

def test_missing_irrelevant_replacement_prefix_row_is_accepted(tmp_path: Path) -> None:
    def remove_prefix_row(frames: dict[str, tuple[pd.DataFrame, bool]]) -> None:
        replacement = frames["replacement_market"][0].drop(index=10).reset_index(drop=True)
        _replace_frame(frames, "replacement_market", replacement)

    config = synthetic_config(tmp_path, remove_prefix_row)
    support = subject.materialize(config)

    assert support["validation"]["market_partition"]["replacement_tail_rows"] == 3
    output = pd.read_csv(tmp_path / config.market_output_path)
    old = pd.read_csv(config.inputs[0].path)
    pd.testing.assert_frame_equal(
        output.iloc[:20].reset_index(drop=True), old, check_exact=True, check_dtype=False
    )

@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda f: _replace_frame(
                f, "replacement_market", f["replacement_market"][0].drop(index=20)
            ),
            "tail row count",
        ),
        (
            lambda f: f["replacement_market"][0].loc.__setitem__(
                (21, "date"), f["replacement_market"][0].loc[20, "date"]
            ),
            "duplicate timestamps",
        ),
        (
            lambda f: f["replacement_market"][0].loc.__setitem__(
                (20, "date"),
                f["replacement_market"][0].loc[20, "date"] + pd.Timedelta(minutes=1),
            ),
            "off-grid",
        ),
        (
            lambda f: f["replacement_market"][0].loc.__setitem__(
                (slice(20, 21), "date"),
                list(reversed(f["replacement_market"][0].loc[20:21, "date"].tolist())),
            ),
            "out-of-order",
        ),
        (
            lambda f: f["replacement_market"][0].loc.__setitem__(
                (22, "date"),
                f["replacement_market"][0].loc[22, "date"] + pd.Timedelta(minutes=5),
            ),
            "tail row count",
        ),
    ],
)
def test_replacement_tail_partition_rejections(
    tmp_path: Path,
    mutate: Callable[[dict[str, tuple[pd.DataFrame, bool]]], None],
    match: str,
) -> None:
    config = synthetic_config(tmp_path, mutate)
    with pytest.raises(subject.SourceSupportFailure, match=match):
        subject.materialize(config)
    assert (tmp_path / config.attempt_path).exists()

def test_replacement_tail_rejects_explicit_early_boundary(tmp_path: Path) -> None:
    def admit_boundary(frames: dict[str, tuple[pd.DataFrame, bool]]) -> None:
        replacement = frames["replacement_market"][0].drop(index=19).reset_index(
            drop=True
        )
        replacement.loc[19, "date"] = replacement.loc[18, "date"] + pd.Timedelta(
            minutes=5
        )
        _replace_frame(frames, "replacement_market", replacement)

    config = synthetic_config(tmp_path, admit_boundary)
    with pytest.raises(subject.SourceSupportFailure, match="tail row count"):
        subject.materialize(config)

def test_replacement_tail_rejects_independent_extra_and_wrong_count(
    tmp_path: Path,
) -> None:
    def append_extra(frames: dict[str, tuple[pd.DataFrame, bool]]) -> None:
        replacement = frames["replacement_market"][0]
        extra = replacement.iloc[[-1]].copy()
        extra.loc[:, "date"] = replacement.date.iloc[-1] + pd.Timedelta(minutes=5)
        _replace_frame(
            frames,
            "replacement_market",
            pd.concat([replacement, extra], ignore_index=True),
        )

    config = synthetic_config(tmp_path, append_extra)
    config = replace(config, domain_end=config.domain_end + pd.Timedelta(minutes=5))
    with pytest.raises(subject.SourceSupportFailure, match="tail row count"):
        subject.materialize(config)

@pytest.mark.parametrize("token", ("", "not-a-timestamp"))
def test_replacement_tail_rejects_null_or_invalid_date_token(
    tmp_path: Path, token: str
) -> None:
    config = synthetic_config(tmp_path)

    def alter(lines: list[str]) -> None:
        row = next(csv.reader([lines[-1]], strict=True))
        row[0] = token
        lines[-1] = _encode_csv_row(row)

    config = _rewrite_replacement_csv(config, alter)
    with pytest.raises(subject.SourceSupportFailure, match="timestamps"):
        subject.materialize(config)

@pytest.mark.parametrize("width_delta", (-1, 1))
def test_replacement_tail_rejects_short_or_long_selected_row(
    tmp_path: Path, width_delta: int
) -> None:
    config = synthetic_config(tmp_path)

    def alter(lines: list[str]) -> None:
        row = next(csv.reader([lines[-1]], strict=True))
        if width_delta < 0:
            row.pop()
        else:
            row.append("unexpected")
        lines[-1] = _encode_csv_row(row)

    config = _rewrite_replacement_csv(config, alter)
    with pytest.raises(subject.SourceSupportFailure, match="selected tail row width"):
        subject.materialize(config)

def test_replacement_tail_rejects_malformed_selected_csv(tmp_path: Path) -> None:
    config = synthetic_config(tmp_path)

    def alter(lines: list[str]) -> None:
        date = next(csv.reader([lines[-1]], strict=True))[0]
        lines[-1] = f'{date},"unterminated'

    config = _rewrite_replacement_csv(config, alter)
    with pytest.raises(subject.SourceSupportFailure, match="date scan failed"):
        subject.materialize(config)

def test_market_partition_source_uses_strict_predicates_and_no_overlap_comparison() -> None:
    tree = ast.parse(Path(subject.__file__).read_text())
    functions = {
        node.name: ast.get_source_segment(Path(subject.__file__).read_text(), node)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    scan = functions["_scan_replacement_dates"] or ""
    transform = functions["_transform"] or ""
    assert "(parsed > config.old_last) & (parsed < config.domain_end)" in scan
    assert "parsed >= config.old_last" not in scan
    assert "parsed <= config.domain_end" not in scan
    assert "market logical prefix" not in transform
    assert "pd.concat([old_market, replacement_tail], ignore_index=True)" in transform
    for forbidden in ("fillna", "interpolate", "resample"):
        assert forbidden not in scan
        assert forbidden not in transform

def test_market_partition_is_exact_disjoint_union(tmp_path: Path) -> None:
    config = synthetic_config(tmp_path)
    support = subject.materialize(config)
    output = pd.read_csv(tmp_path / config.market_output_path, parse_dates=["date"])
    old = pd.read_csv(config.inputs[0].path, parse_dates=["date"])
    replacement = pd.read_csv(config.inputs[1].path, parse_dates=["date"])
    tail = replacement.loc[
        (replacement.date > config.old_last) & (replacement.date < config.domain_end)
    ].reset_index(drop=True)

    old_dates = set(old.date)
    tail_dates = set(tail.date)
    assert old_dates.isdisjoint(tail_dates)
    assert set(output.date) == old_dates | tail_dates
    assert len(output) == len(old) + len(tail) == config.expected_complete_rows
    pd.testing.assert_frame_equal(
        output.iloc[: len(old)].reset_index(drop=True), old, check_exact=True, check_dtype=False
    )
    pd.testing.assert_frame_equal(
        output.iloc[len(old) :].reset_index(drop=True), tail, check_exact=True, check_dtype=False
    )
    assert support["validation"]["market_partition"]["timestamp_intersection_rows"] == 0

@pytest.mark.parametrize(
    "mutation",
    ("inclusive_old_predicate", "inclusive_domain_end_predicate", "prefix_admission"),
)
def test_market_partition_predicate_and_prefix_mutants_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    def maybe_append_domain_end(
        frames: dict[str, tuple[pd.DataFrame, bool]],
    ) -> None:
        if mutation != "inclusive_domain_end_predicate":
            return
        replacement = frames["replacement_market"][0]
        extra = replacement.iloc[[-1]].copy()
        extra.loc[:, "date"] = replacement.date.iloc[-1] + pd.Timedelta(minutes=5)
        _replace_frame(
            frames,
            "replacement_market",
            pd.concat([replacement, extra], ignore_index=True),
        )

    config = synthetic_config(tmp_path, maybe_append_domain_end)
    original_scan = subject._scan_replacement_dates

    def mutated_scan(
        item: subject._Retained,
        bound: subject.MaterializationConfig,
        failpoint: Callable[[str], None],
    ) -> subject.ReplacementDateScan:
        scan = original_scan(item, bound, failpoint)
        if mutation == "inclusive_old_predicate":
            positions = (19, 20, 21, 22)
        elif mutation == "inclusive_domain_end_predicate":
            positions = (20, 21, 22, 23)
        else:
            positions = (19, 20, 21)
        dates = pd.Series(
            [
                pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=5 * position)
                for position in positions
            ]
        )
        return subject.ReplacementDateScan(
            scan.schema, scan.row_count, positions, dates
        )

    monkeypatch.setattr(subject, "_scan_replacement_dates", mutated_scan)
    with pytest.raises(subject.SourceSupportFailure):
        subject.materialize(config)

def test_market_partition_non_disjoint_mutant_reaches_intersection_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = synthetic_config(tmp_path)
    original_scan = subject._scan_replacement_dates
    original_grid = subject._require_grid

    def overlapping_scan(
        item: subject._Retained,
        bound: subject.MaterializationConfig,
        failpoint: Callable[[str], None],
    ) -> subject.ReplacementDateScan:
        scan = original_scan(item, bound, failpoint)
        positions = (19, 20, 21)
        dates = pd.Series(
            pd.date_range(bound.old_last, periods=3, freq="5min")
        )
        return subject.ReplacementDateScan(
            scan.schema, scan.row_count, positions, dates
        )

    def skip_only_mutated_tail_bounds(
        dates: pd.Series,
        *,
        first: pd.Timestamp | None = None,
        last: pd.Timestamp | None = None,
        label: str,
    ) -> None:
        if label == "replacement_market_tail":
            return
        original_grid(dates, first=first, last=last, label=label)

    monkeypatch.setattr(subject, "_scan_replacement_dates", overlapping_scan)
    monkeypatch.setattr(subject, "_require_grid", skip_only_mutated_tail_bounds)
    with pytest.raises(subject.SourceSupportFailure, match="timestamp intersection"):
        subject.materialize(config)

def test_market_partition_reordered_union_mutant_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = synthetic_config(tmp_path)
    original_concat = subject.pd.concat

    def reordered_concat(objects: object, *args: object, **kwargs: object) -> pd.DataFrame:
        if (
            isinstance(objects, list)
            and len(objects) == 2
            and all(isinstance(frame, pd.DataFrame) for frame in objects)
            and all(tuple(frame.columns) == subject.MARKET_SCHEMA for frame in objects)
        ):
            return original_concat(list(reversed(objects)), *args, **kwargs)
        return original_concat(objects, *args, **kwargs)

    monkeypatch.setattr(subject.pd, "concat", reordered_concat)
    with pytest.raises(
        subject.SourceSupportFailure,
        match="materialized old market partition|materialized_market",
    ):
        subject.materialize(config)

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

def test_selected_oi_overlap_matching_missingness_is_exact(tmp_path: Path) -> None:
    def add_matching_overlap_missingness(
        frames: dict[str, tuple[pd.DataFrame, bool]],
    ) -> None:
        old_oi = frames["old_open_interest"][0].copy()
        overlap_date = old_oi.date.iloc[-3]
        old_oi.loc[old_oi.date.eq(overlap_date), "open_interest"] = np.nan
        _replace_frame(frames, "old_open_interest", old_oi)

        metrics = frames["binance_metrics_open_interest"][0].copy()
        metrics.loc[
            metrics.create_time.eq(overlap_date), "sum_open_interest"
        ] = np.nan
        _replace_frame(frames, "binance_metrics_open_interest", metrics)

    config = synthetic_config(tmp_path, add_matching_overlap_missingness)
    support = subject.materialize(config)

    materialized = pd.read_csv(tmp_path / config.oi_output_path)
    overlap_start = config.expected_old_rows - config.splice_rows
    assert pd.isna(materialized.open_interest.iloc[overlap_start])
    assert support["validation"]["oi_splice_window_exact_rows"] == 3

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
        (lambda f: _replace_frame(f, "replacement_market", f["replacement_market"][0].iloc[:-1]), "tail row count|terminal coverage"),
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
    retained_evidence: dict[str, tuple[bytes, int, int, int, int]] = {}

    def inject(name: str) -> None:
        if name == f"checkpoint:{checkpoint}":
            retained_evidence.update(_published_snapshot(config))
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
    assert _published_snapshot(config) == retained_evidence

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
    retained_evidence: dict[str, tuple[bytes, int, int, int, int]] = {}

    def inject(name: str) -> None:
        if name == f"{edge}:{label}":
            retained_evidence.update(_published_snapshot(config))
            raise RuntimeError(name)

    with pytest.raises(RuntimeError, match=f"{edge}:{label}"):
        subject.materialize(config, failpoint=inject)
    assert _fd_count() == baseline_fds
    existing = [tmp_path / relative for relative in config.output_paths if (tmp_path / relative).exists()]
    assert len(existing) == expected_count
    assert all((path.stat().st_mode & 0o777, path.stat().st_nlink) == (0o444, 1) for path in existing)
    assert _published_snapshot(config) == retained_evidence


@pytest.mark.parametrize(
    ("label", "expected_count"),
    (
        ("attempt_sentinel", 0),
        ("materialized_market", 1),
        ("materialized_open_interest", 2),
        ("source_manifest", 3),
        ("source_support", 4),
    ),
)
def test_every_publish_before_linkat_failpoint_closes_and_preserves_exact_prefix(
    tmp_path: Path, label: str, expected_count: int
) -> None:
    config = synthetic_config(tmp_path)
    baseline_fds = _fd_count()
    retained_evidence: dict[str, tuple[bytes, int, int, int, int]] = {}
    event = f"publish:{label}:before_linkat"

    def inject(name: str) -> None:
        if name == event:
            retained_evidence.update(_published_snapshot(config))
            raise RuntimeError(name)

    with pytest.raises(RuntimeError, match=event):
        subject.materialize(config, failpoint=inject)
    assert _fd_count() == baseline_fds
    present = tuple(
        relative for relative in config.output_paths if (tmp_path / relative).exists()
    )
    assert present == config.output_paths[:expected_count]
    for relative in present:
        info = os.lstat(tmp_path / relative)
        assert stat.S_IMODE(info.st_mode) == 0o444
        assert info.st_nlink == 1
    assert _published_snapshot(config) == retained_evidence


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
    retained_evidence: dict[str, tuple[bytes, int, int, int, int]] = {}

    def inject(name: str) -> None:
        if name == f"publication:{label}:{phase}":
            retained_evidence.update(_published_snapshot(config))
            raise RuntimeError(name)

    with pytest.raises(RuntimeError, match=f"publication:{label}:{phase}"):
        subject.materialize(config, failpoint=inject)
    assert _fd_count() == baseline_fds
    assert sum((tmp_path / relative).exists() for relative in config.output_paths) == expected_count
    assert _published_snapshot(config) == retained_evidence

@pytest.mark.parametrize("relative_attribute", ("market_output_path", "oi_output_path"))
def test_generated_readback_descriptor_failpoints_close_cleanly(
    tmp_path: Path, relative_attribute: str
) -> None:
    config = synthetic_config(tmp_path)
    relative = getattr(config, relative_attribute)
    baseline_fds = _fd_count()
    retained_evidence: dict[str, tuple[bytes, int, int, int, int]] = {}

    def inject(name: str) -> None:
        if name == f"readback:{relative}:authenticated":
            retained_evidence.update(_published_snapshot(config))
            raise RuntimeError(name)

    with pytest.raises(RuntimeError, match="readback:.*:authenticated"):
        subject.materialize(config, failpoint=inject)
    assert _fd_count() == baseline_fds
    assert sum((tmp_path / path).exists() for path in config.output_paths) == 3
    assert _published_snapshot(config) == retained_evidence

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
    retained_evidence: dict[str, tuple[bytes, int, int, int, int]] = {}

    def inject(name: str) -> None:
        if name == f"output:{relative}:final":
            retained_evidence.update(_published_snapshot(config))
            raise RuntimeError(name)

    with pytest.raises(RuntimeError, match="output:.*:final"):
        subject.materialize(config, failpoint=inject)
    assert _fd_count() == baseline_fds
    assert all((tmp_path / path).exists() for path in config.output_paths)
    assert _published_snapshot(config) == retained_evidence

@pytest.mark.parametrize(
    ("source_name", "phase", "expected_count"),
    [
        (
            source_name,
            phase,
            0 if phase in ("before_hash", "after_open_hash") else 5 if phase == "final" else 1,
        )
        for source_name in (
            "old_market", "funding", "premium",
            "old_open_interest", "binance_metrics_open_interest",
            "rank7_spot_premium_5m",
        )
        for phase in (
            "before_hash", "after_open_hash", "before_decode", "after_decode",
            "after_rehash", "final",
        )
    ] + [
        (
            "replacement_market",
            phase,
            0 if phase in ("before_hash", "after_open_hash") else 5 if phase == "final" else 1,
        )
        for phase in (
            "before_hash", "after_open_hash", "before_date_scan",
            "after_date_scan", "after_date_scan_rehash", "before_tail_decode",
            "after_tail_decode", "after_tail_rehash", "final",
        )
    ],
)
def test_every_descriptor_identity_failpoint_closes_cleanly(
    tmp_path: Path, source_name: str, phase: str, expected_count: int
) -> None:
    config = synthetic_config(tmp_path)
    baseline_fds = _fd_count()
    retained_evidence: dict[str, tuple[bytes, int, int, int, int]] = {}

    def inject(name: str) -> None:
        if name == f"identity:{source_name}:{phase}":
            retained_evidence.update(_published_snapshot(config))
            raise RuntimeError(name)

    with pytest.raises(RuntimeError, match=f"identity:{source_name}:{phase}"):
        subject.materialize(config, failpoint=inject)
    assert _fd_count() == baseline_fds
    assert sum((tmp_path / relative).exists() for relative in config.output_paths) == expected_count
    assert _published_snapshot(config) == retained_evidence

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

def test_s11_has_no_candidate_or_economics_execution_or_open_surface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = Path(subject.__file__).read_text()
    tree = ast.parse(source)
    forbidden_modules = {
        "training.build_gross9_structural_clock_bundle",
        "training.evaluate_gross9_structural_clock_bundle",
    }
    forbidden_calls = {
        "pct_change",
        "cumprod",
        "rolling",
        "expanding",
        "ewm",
        "cagr",
        "max_drawdown",
        "drawdown",
        "pnl",
        "returns",
        "evaluate_candidate",
        "build_candidate",
        "build_schedule",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module not in forbidden_modules
        elif isinstance(node, ast.Import):
            assert all(alias.name not in forbidden_modules for alias in node.names)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                call_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                call_name = node.func.attr
            else:
                continue
            assert call_name.lower() not in forbidden_calls

    config = synthetic_config(tmp_path)
    opened: list[Path] = []
    original_open = subject._open_nofollow_components

    def audited_open(path: Path, flags: int) -> int:
        opened.append(Path(path).absolute())
        return original_open(path, flags)

    monkeypatch.setattr(subject, "_open_nofollow_components", audited_open)
    support = subject.materialize(config)
    allowed = {
        *(Path(binding.path).absolute() for binding in config.inputs),
        *(config.root / relative for relative in config.output_paths),
        *((config.root / relative).parent for relative in config.output_paths),
    }
    assert set(opened) <= allowed
    process = support["access_ledger"]["process_local"]
    for key in (
        "candidate_value_rows_opened", "comparator_value_rows_opened",
        "feature_value_rows_opened", "schedule_value_rows_opened",
        "signal_value_rows_opened", "return_value_rows_opened",
        "pnl_value_rows_opened", "cagr_evaluation_count",
        "mdd_evaluation_count", "drawdown_evaluation_count",
        "economic_value_rows_opened", "economic_evaluation_count",
    ):
        assert process[key] == 0


def _rewrite_metrics_rows(
    config: subject.MaterializationConfig,
    mutate: Callable[[list[list[str]]], None],
) -> subject.MaterializationConfig:
    binding = next(row for row in config.inputs if row.name == "binance_metrics_open_interest")
    with gzip.open(binding.path, "rt", newline="") as handle:
        rows = list(csv.reader(handle, strict=True))
    mutate(rows)
    buffer = io.StringIO(newline="")
    csv.writer(buffer, lineterminator="\n").writerows(rows)
    raw_csv = buffer.getvalue().encode()
    compressed = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=compressed, compresslevel=9, mtime=0) as handle:
        handle.write(raw_csv)
    return _replace_bound_input_bytes(
        config, "binance_metrics_open_interest", compressed.getvalue()
    )


def _support_json(config: subject.MaterializationConfig) -> dict[str, object]:
    return json.loads((config.root / config.support_output_path).read_text())


def test_synthetic_success_has_exact_metrics_grid_overlap_tail_and_nine_passes(
    tmp_path: Path,
) -> None:
    config = synthetic_config(tmp_path)
    support = subject.materialize(config)
    access = support["access_ledger"]
    current = access["current_s11"]["access"]
    process = access["process_local"]
    assert current["decode_pass_count"] == 9
    assert current["decode_passes"] == [
        "old_market",
        "replacement_market_date_scan",
        "replacement_market_tail",
        "funding",
        "premium",
        "old_open_interest",
        "binance_metrics_open_interest_date_scan",
        "binance_metrics_open_interest_selected_window",
        "rank7_spot_premium_5m",
    ]
    assert current["metrics_selected_row_count"] == 6
    assert current["metrics_overlap_row_count"] == 3
    assert current["metrics_tail_row_count"] == 3
    assert current["non_selected_metrics_non_date_semantic_evaluation_count"] == 0
    assert current["global_metrics_alignment_comparison_count"] == 0
    assert current["off_grid_detail_disclosure_count"] == 0
    assert process["stage"] == "S11"
    expected_addends = [20, 23, 3, 2, 2, 20, 6, 6, 23]
    assert process["source_value_rows_opened"] == sum(expected_addends)
    assert access["access_ledger_hash"] == subject.object_hash(
        access, "access_ledger_hash"
    )
    replay = access["replay_guard"]
    assert replay["replay_guard_hash"] == subject.object_hash(
        replay, "replay_guard_hash"
    )
    assert support["validation"]["oi_splice_window_exact_rows"] == 3
    assert support["validation"]["appended_oi_rows"] == 3


def test_metrics_global_off_grid_rows_and_poisoned_non_date_fields_are_ignored(
    tmp_path: Path,
) -> None:
    config = synthetic_config(tmp_path)

    def mutate(rows: list[list[str]]) -> None:
        header, body = rows[0], rows[1:]
        poison = ["POISON"] * (len(header) - 1)
        additions = [
            ["2026-01-01 01:22:00", *poison],
            ["2026-01-01 01:27:00", *poison],
            ["2026-01-01 01:32:00", *poison],
            ["2026-01-01 01:52:00", *poison],
        ]
        rows[:] = [header, *sorted([*body, *additions], key=lambda row: row[0])]

    config = _rewrite_metrics_rows(config, mutate)
    support = subject.materialize(config)
    access = support["access_ledger"]
    current = access["current_s11"]["access"]
    process = access["process_local"]
    assert process["binance_metrics_open_interest_date_rows_scanned"] == 10
    assert current["metrics_selected_row_count"] == 6
    assert current["non_selected_metrics_non_date_semantic_evaluation_count"] == 0
    assert current["global_metrics_alignment_comparison_count"] == 0


def test_metrics_interleaved_off_grid_rows_may_have_short_width_and_poison(
    tmp_path: Path,
) -> None:
    config = synthetic_config(tmp_path)

    def mutate(rows: list[list[str]]) -> None:
        rows.insert(3, ["2026-01-01 01:32:00", "POISON"])
        rows.insert(5, ["2026-01-01 01:37:00"])

    config = _rewrite_metrics_rows(config, mutate)
    support = subject.materialize(config)
    assert support["access_ledger"]["process_local"]["binance_metrics_open_interest_date_rows_scanned"] == 8


@pytest.mark.parametrize(
    ("label", "mutate", "match"),
    [
        ("missing", lambda rows: rows.pop(3), "required metrics timestamp selection differs|missing OI splice anchor window"),
        ("shifted", lambda rows: rows[3].__setitem__(0, "2026-01-01 01:31:00"), "required metrics timestamp selection differs|missing OI splice anchor window"),
        ("duplicate", lambda rows: rows[3].__setitem__(0, rows[2][0]), "duplicate timestamps"),
        ("out_of_order", lambda rows: rows.__setitem__(slice(2, 4), [rows[3], rows[2]]), "out-of-order timestamps"),
        ("null", lambda rows: rows[3].__setitem__(0, ""), "null timestamps|invalid timestamps"),
        ("malformed", lambda rows: rows[3].__setitem__(0, "not-a-date"), "invalid timestamps"),
    ],
)
def test_metrics_date_scan_rejects_missing_shifted_duplicate_out_of_order_or_bad_dates(
    tmp_path: Path,
    label: str,
    mutate: Callable[[list[list[str]]], None],
    match: str,
) -> None:
    del label
    config = _rewrite_metrics_rows(synthetic_config(tmp_path), mutate)
    with pytest.raises(subject.SourceSupportFailure, match=match):
        subject.materialize(config)
    assert (tmp_path / config.attempt_path).exists()
    assert not (tmp_path / config.market_output_path).exists()


@pytest.mark.parametrize(
    ("label", "mutate", "match"),
    [
        ("short", lambda rows: rows[2].pop(), "selected row width differs"),
        ("long", lambda rows: rows[2].append("extra"), "selected row width differs"),
        ("symbol", lambda rows: rows[2].__setitem__(1, "ETHUSDT"), "metrics symbol differs"),
        ("oi", lambda rows: rows[2].__setitem__(2, "POISON"), "malformed numeric value"),
        ("overlap", lambda rows: rows[2].__setitem__(2, "999"), "OI overlap conflict"),
        ("tail_zero", lambda rows: rows[-1].__setitem__(2, "0"), "non-positive or non-finite tail OI"),
        ("tail_negative", lambda rows: rows[-1].__setitem__(2, "-1"), "non-positive or non-finite tail OI"),
        ("tail_inf", lambda rows: rows[-1].__setitem__(2, "inf"), "non-positive or non-finite tail OI"),
    ],
)
def test_metrics_selected_only_decode_rejects_width_symbol_overlap_and_tail_values(
    tmp_path: Path,
    label: str,
    mutate: Callable[[list[list[str]]], None],
    match: str,
) -> None:
    del label
    config = _rewrite_metrics_rows(synthetic_config(tmp_path), mutate)
    with pytest.raises(subject.SourceSupportFailure, match=match):
        subject.materialize(config)


def test_metrics_selected_decode_rejects_descriptor_rehash_drift_between_passes(
    tmp_path: Path,
) -> None:
    config = synthetic_config(tmp_path)
    metrics = Path(next(row.path for row in config.inputs if row.name == "binance_metrics_open_interest"))
    changed = False

    def inject(name: str) -> None:
        nonlocal changed
        if name == "identity:binance_metrics_open_interest:before_selected_window_decode" and not changed:
            raw = bytearray(metrics.read_bytes())
            raw[len(raw) // 2] ^= 1
            metrics.write_bytes(raw)
            changed = True

    with pytest.raises(subject.SourceSupportFailure, match="identity drift|rehash differs"):
        subject.materialize(config, failpoint=inject)
    assert changed


def _terminal_class(config: subject.MaterializationConfig) -> tuple[str, tuple[str, ...]]:
    present = tuple(
        relative for relative in config.output_paths if (config.root / relative).exists()
    )
    if not present:
        return "pre_sentinel_failure", present
    assert present[0] == config.attempt_path
    if len(present) == 1:
        return "post_sentinel_pre_other_output_failure", present
    return "partial_publication_failure", present


@pytest.mark.parametrize(
    ("event", "expected_class", "expected_count"),
    [
        ("publish:attempt_sentinel:before_linkat", "pre_sentinel_failure", 0),
        ("checkpoint:2", "post_sentinel_pre_other_output_failure", 1),
        ("checkpoint:4", "partial_publication_failure", 2),
        ("checkpoint:5", "partial_publication_failure", 3),
        ("checkpoint:7", "partial_publication_failure", 4),
        ("checkpoint:8", "partial_publication_failure", 5),
    ],
)
def test_all_three_synthetic_terminal_failure_classes_preserve_exact_prefix(
    tmp_path: Path, event: str, expected_class: str, expected_count: int
) -> None:
    config = synthetic_config(tmp_path)

    def inject(name: str) -> None:
        if name == event:
            raise RuntimeError(event)

    with pytest.raises(RuntimeError, match=event):
        subject.materialize(config, failpoint=inject)
    failure_class, present = _terminal_class(config)
    assert failure_class == expected_class
    assert present == config.output_paths[:expected_count]
    for relative in present:
        info = os.lstat(config.root / relative)
        assert info.st_mode & 0o777 == 0o444
        assert info.st_nlink == 1
    snapshot = _published_snapshot(config)
    with pytest.raises(subject.SourceSupportFailure, match="already exists"):
        subject.materialize(config)
    assert _published_snapshot(config) == snapshot

def _rehash_mutated_access(ledger: dict[str, object]) -> None:
    replay = ledger["replay_guard"]
    assert isinstance(replay, dict)
    replay["replay_guard_hash"] = subject.object_hash(replay, "replay_guard_hash")
    ledger["access_ledger_hash"] = subject.object_hash(ledger, "access_ledger_hash")


@pytest.mark.parametrize(
    ("label", "mutate"),
    [
        ("missing", lambda value: value["current_s11"]["access"].pop("metrics_date_scan_count")),
        ("additional", lambda value: value["process_local"].__setitem__("extra", 0)),
        ("wrong_type", lambda value: value["process_local"].__setitem__("slot", False)),
        ("wrong_stage", lambda value: value["replay_guard"]["s11"].__setitem__("commit", value["replay_guard"]["a11"]["commit"])),
        ("path_length", lambda value: value["replay_guard"]["s11"].__setitem__("implementation_paths", value["replay_guard"]["s11"]["implementation_paths"][:1])),
        ("path_order", lambda value: value["replay_guard"]["s11"].__setitem__("implementation_paths", list(reversed(value["replay_guard"]["s11"]["implementation_paths"])))),
        ("pass_order", lambda value: value["current_s11"]["access"]["decode_passes"].__setitem__(slice(6, 8), list(reversed(value["current_s11"]["access"]["decode_passes"][6:8])))),
        ("output_length", lambda value: value["replay_guard"]["expected_output_paths"]["current_s11"].pop()),
        ("nonempty_intersection", lambda value: value["replay_guard"]["pairwise_output_intersections"]["historical_s10_current_s11"].append("collision")),
        ("identity_substitution", lambda value: value["replay_guard"]["identities"].__setitem__("current_s11", value["replay_guard"]["identities"]["historical_s10"])),
        ("attempt_hash_substitution", lambda value: value["replay_guard"]["attempt_hashes"].__setitem__("current_s11", value["replay_guard"]["attempt_hashes"]["historical_s10"])),
        ("self_hash", lambda value: value.__setitem__("access_ledger_hash", "0" * 64)),
    ],
)
def test_recursive_access_replay_type_length_order_and_wrong_stage_mutations_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    mutate: Callable[[dict[str, object]], object],
) -> None:
    del label
    original = subject._build_access_ledger

    def mutant(*args: object, **kwargs: object) -> dict[str, object]:
        ledger = original(*args, **kwargs)
        mutate(ledger)
        if ledger.get("access_ledger_hash") != "0" * 64:
            _rehash_mutated_access(ledger)
        return ledger

    monkeypatch.setattr(subject, "_build_access_ledger", mutant)
    with pytest.raises(subject.SourceSupportFailure, match="access ledger exact schema"):
        subject.materialize(synthetic_config(tmp_path))


def test_nine_addend_source_count_has_exact_order_and_eighteen_stage_bindings(
    tmp_path: Path,
) -> None:
    support = subject.materialize(synthetic_config(tmp_path))
    ledger = support["access_ledger"]
    current = ledger["current_s11"]["access"]
    process = ledger["process_local"]
    ordered_process_keys = [
        "old_market_rows_opened",
        "replacement_market_date_rows_scanned",
        "replacement_market_tail_rows_opened",
        "funding_rows_opened",
        "premium_rows_opened",
        "old_open_interest_rows_opened",
        "binance_metrics_open_interest_date_rows_scanned",
        "binance_metrics_open_interest_selected_window_rows_opened",
        "rank7_spot_premium_5m_rows_opened",
    ]
    process_values = [process[key] for key in ordered_process_keys]
    assert process_values == [20, 23, 3, 2, 2, 20, 6, 6, 23]
    assert process["source_value_rows_opened"] == sum(process_values)
    ordered_current_keys = [
        "raw_file_count", "raw_file_open_count", "decode_pass_count",
        "replacement_market_date_scan_count", "replacement_market_tail_decode_count",
        "replacement_market_tail_selected_row_count", "metrics_date_scan_count",
        "metrics_selected_decode_count", "metrics_selected_row_count",
    ]
    assert [current[key] for key in ordered_current_keys] == [7, 7, 9, 1, 1, 3, 1, 1, 6]
    assert len(ordered_process_keys) + len(ordered_current_keys) == 18


def test_official_config_is_exact_and_does_not_open_sources(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("official source opened")

    monkeypatch.setattr(Path, "open", forbidden)
    config = subject.official_config(tmp_path)
    assert config.repository_parent == subject.T10
    assert config.authority_commit == subject.A11
    assert config.old_last == pd.Timestamp("2026-05-31 15:00:00")
    assert config.domain_end == pd.Timestamp("2026-06-01 00:00:00")
    assert (config.expected_old_rows, config.expected_append_rows, config.expected_complete_rows) == (674785, 107, 674892)
    assert config.splice_rows == 13
    assert len(config.inputs) == 7
    assert sum(row.size_bytes for row in config.inputs) == 190272610
    assert len(config.output_paths) == 5
    assert all("g9cb11" in path for path in config.output_paths)
    assert not any("g9cb9_" in row.path or "g9cb10_" in row.path for row in config.inputs)


def test_synthetic_repository_gate_order_precedes_source_open_and_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(synthetic_config(tmp_path), enforce_repository_gates=True)
    events: list[str] = []
    monkeypatch.setattr(subject, "_validate_command_and_root", lambda _config: events.append("command"))
    monkeypatch.setattr(subject, "_validate_bytecode_first_gate", lambda _root: events.append("bytecode"))
    monkeypatch.setattr(subject, "_validate_git_gate", lambda _config: events.append("git") or "synthetic-s11")
    monkeypatch.setattr(subject, "_validate_history_gate", lambda _config: events.append("history"))
    original_absence = subject._validate_output_absence
    original_open = subject._open_inputs
    original_publish = subject._publish
    monkeypatch.setattr(subject, "_validate_output_absence", lambda cfg: (events.append("absence"), original_absence(cfg))[1])
    monkeypatch.setattr(subject, "_open_inputs", lambda cfg, inject: (events.append("source_open"), original_open(cfg, inject))[1])
    monkeypatch.setattr(subject, "_publish", lambda *args, **kwargs: (events.append("publication"), original_publish(*args, **kwargs))[1])
    subject.materialize(config)
    assert events[:6] == ["command", "bytecode", "git", "history", "absence", "source_open"]
    assert events.count("bytecode") == 2
    assert events.index("publication") > events.index("source_open")
    assert events[-1] == "bytecode"


def test_no_forbidden_time_repair_or_economics_ast_surface() -> None:
    source = Path(subject.__file__).read_text()
    tree = ast.parse(source)
    forbidden_attributes = {
        "floor", "ceil", "round", "asof", "merge_asof", "fillna", "ffill",
        "bfill", "interpolate", "resample", "pct_change", "cumprod",
    }
    forbidden_tokens = {"candidate", "comparator", "pnl", "cagr", "mdd", "drawdown"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ""
            assert name.lower() not in forbidden_attributes
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            text = ast.unparse(node).lower()
            assert not any(token in text for token in forbidden_tokens)

def test_git_gate_enforces_exact_a11_t10_s11_topology_modes_and_two_file_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(synthetic_config(tmp_path), branch=subject.BRANCH)
    head = "1" * 40
    answers = {
        ("rev-parse", "HEAD"): head,
        ("branch", "--show-current"): subject.BRANCH,
        ("rev-parse", "@{upstream}"): head,
        ("rev-list", "--parents", "-n", "1", head): f"{head} {subject.T10}",
        ("rev-parse", f"{subject.T10}^"): subject.A11,
        ("rev-parse", f"{subject.A11}^"): subject.S10,
        ("rev-parse", f"{subject.S10}^"): subject.T9,
        ("diff", "--name-status", subject.S10, subject.A11): "A\tdocs/gross9-structural-clock-bundle-g9cb11-successor-authority-decision-2026-07-31.md",
        ("diff", "--name-status", subject.A11, subject.T10): "A\tresults/gross9_structural_clock_bundle_g9cb10_source_support_terminal_failure_2026-07-31.json\nA\tresults/gross9_structural_clock_bundle_g9cb10_source_support_attempt_consumed_2026-07-31.json",
        ("diff", "--name-status", subject.T10, head): "A\ttraining/materialize_gross9_structural_clock_g9cb11_sources.py\nA\ttests/test_materialize_gross9_structural_clock_g9cb11_sources.py",
        ("ls-files", "-s", "--", "training/materialize_gross9_structural_clock_g9cb11_sources.py"): "100644 blob 0 training/materialize_gross9_structural_clock_g9cb11_sources.py",
        ("ls-files", "-s", "--", "tests/test_materialize_gross9_structural_clock_g9cb11_sources.py"): "100644 blob 0 tests/test_materialize_gross9_structural_clock_g9cb11_sources.py",
        ("status", "--porcelain=v1", "--untracked-files=all"): "",
    }
    monkeypatch.setattr(subject, "_run_git", lambda _root, *args: answers[args])
    assert subject._validate_git_gate(config) == head
    answers[("diff", "--name-status", subject.T10, head)] += "\nA\tunexpected.py"
    with pytest.raises(subject.SourceSupportFailure, match="exact S11 two-file diff differs"):
        subject._validate_git_gate(config)


def test_t10_historical_current_process_and_replay_schemas_are_exact(
    tmp_path: Path,
) -> None:
    ledger = subject.materialize(synthetic_config(tmp_path))["access_ledger"]
    assert list(ledger) == [
        "schema_version", "ledger_kind", "historical_s9", "historical_s10",
        "current_s11", "process_local", "replay_guard", "access_ledger_hash",
    ]
    historical = ledger["historical_s10"]
    terminal = subject._T10_TERMINAL_LITERAL
    assert set(terminal) == {
        "schema_version", "ledger_kind", "identity", "status", "authority",
        "seal_authority", "implementation", "attempt_sentinel", "execution",
        "failure", "access", "output_state", "terminal_failure_hash",
    }
    assert set(historical) == {
        "identity", "status", "authority", "seal_authority", "implementation",
        "attempt_sentinel", "terminal_ledger", "execution", "failure", "access",
        "output_state",
    }
    for key in (
        "identity", "status", "authority", "seal_authority", "implementation",
        "attempt_sentinel", "execution", "failure", "access", "output_state",
    ):
        assert historical[key] == terminal[key]
    current = ledger["current_s11"]
    assert list(current) == ["identity", "authority", "implementation", "attempt_sentinel", "execution", "access"]
    assert set(current["implementation"]) == {"commit", "parent_commit", "paths"}
    assert len(current["implementation"]["paths"]) == 2
    assert set(current["attempt_sentinel"]) == {"path", "size_bytes", "sha256", "attempt_hash"}
    assert set(current["execution"]) == {"command", "invocation_count", "one_shot", "retry_allowed", "resume_allowed"}
    process = ledger["process_local"]
    assert len(process) == 32
    assert list(process)[:12] == [
        "stage", "slot", "invocation_count", "raw_file_count", "raw_file_open_count",
        "decode_pass_count", "old_market_rows_opened",
        "replacement_market_date_rows_scanned", "replacement_market_tail_rows_opened",
        "funding_rows_opened", "premium_rows_opened", "old_open_interest_rows_opened",
    ]
    replay = ledger["replay_guard"]
    assert list(replay) == [
        "a11", "t10", "s11", "identities", "attempt_hashes",
        "terminal_failure_hashes", "expected_output_paths",
        "pairwise_output_intersections", "identities_pairwise_distinct",
        "attempt_hashes_pairwise_distinct", "replay_guard_hash",
    ]
    assert all(len(paths) == 5 for paths in replay["expected_output_paths"].values())
    assert all(not values for values in replay["pairwise_output_intersections"].values())
    assert replay["s11"]["implementation_paths"] == current["implementation"]["paths"]


def production_shape_config(
    tmp_path: Path,
    mutate: Callable[[dict[str, tuple[pd.DataFrame, bool]]], None] | None = None,
) -> subject.MaterializationConfig:
    """Create a compact fixture with the exact production 13+107 OI window."""
    for directory in ("inputs", "results", "data", "configs/shadow"):
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
    old_dates = pd.date_range("2026-05-31 12:35:00", periods=30, freq="5min")
    tail_dates = pd.date_range("2026-05-31 15:05:00", periods=107, freq="5min")
    complete_dates = old_dates.append(tail_dates)
    old_market = _market(old_dates)
    replacement = _market(complete_dates)
    old_oi = pd.DataFrame(
        {"date": old_dates, "open_interest": np.arange(len(old_dates), dtype=float) + 100}
    )
    selected_dates = pd.date_range("2026-05-31 14:00:00", periods=120, freq="5min")
    old_map = dict(zip(old_dates, old_oi.open_interest, strict=True))
    metrics = pd.DataFrame(
        {
            "create_time": selected_dates,
            "symbol": "BTCUSDT",
            "sum_open_interest": [
                old_map.get(date, 1_000.0 + index)
                for index, date in enumerate(selected_dates)
            ],
            "sum_open_interest_value": 1.0,
            "count_toptrader_long_short_ratio": 1.0,
            "sum_toptrader_long_short_ratio": 1.0,
            "count_long_short_ratio": 1.0,
            "sum_taker_long_short_vol_ratio": 1.0,
        },
        columns=subject.METRICS_SCHEMA,
    )
    aux_dates = pd.date_range(complete_dates[0], complete_dates[-1], freq="1h")
    funding = pd.DataFrame({"date": aux_dates, "funding_rate": 0.001})
    premium = pd.DataFrame({"date": aux_dates, "premium_index": 0.1})
    rank = pd.DataFrame(
        {
            "date": complete_dates,
            "spot_close": np.arange(len(complete_dates), dtype=float) + 1_000,
            "spot_rows": 5.0,
            "premium_index_1m_close": np.arange(len(complete_dates), dtype=float) / 100,
            "premium_rows": 5.0,
            "unused": "bound-but-not-projected",
        }
    )
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
    bindings: list[subject.InputBinding] = []
    for name, (frame, compressed) in frames.items():
        path = tmp_path / "inputs" / f"{name}.csv{'.gz' if compressed else ''}"
        _write_csv(path, frame, compressed=compressed)
        raw = path.read_bytes()
        bindings.append(
            subject.InputBinding(
                name, str(path), hashlib.sha256(raw).hexdigest(), len(raw), compressed
            )
        )
    return subject.MaterializationConfig(
        root=tmp_path,
        inputs=tuple(bindings),
        attempt_path="results/attempt.json",
        market_output_path="data/market.csv.gz",
        oi_output_path="data/oi.csv.gz",
        manifest_output_path="configs/shadow/sources.json",
        support_output_path="results/support.json",
        old_last=pd.Timestamp("2026-05-31 15:00:00"),
        domain_end=pd.Timestamp("2026-06-01 00:00:00"),
        expected_old_rows=30,
        expected_complete_rows=137,
        expected_append_rows=107,
        splice_rows=13,
        rank7_tail_rows=137,
        inherited_manifest=_manifest(),
    )


@pytest.mark.parametrize("overlap_index", [0, 6, 12], ids=["first", "interior", "last"])
def test_production_shape_accepts_matching_missingness_across_exact_13_row_overlap(
    tmp_path: Path, overlap_index: int
) -> None:
    def mutate(frames: dict[str, tuple[pd.DataFrame, bool]]) -> None:
        old_oi = frames["old_open_interest"][0].copy()
        metrics = frames["binance_metrics_open_interest"][0].copy()
        overlap_date = metrics.create_time.iloc[overlap_index]
        old_oi.loc[old_oi.date.eq(overlap_date), "open_interest"] = np.nan
        metrics.loc[metrics.create_time.eq(overlap_date), "sum_open_interest"] = np.nan
        _replace_frame(frames, "old_open_interest", old_oi)
        _replace_frame(frames, "binance_metrics_open_interest", metrics)

    config = production_shape_config(tmp_path, mutate)
    support = subject.materialize(config)
    output = pd.read_csv(tmp_path / config.oi_output_path)
    output_index = config.expected_old_rows - config.splice_rows + overlap_index
    assert pd.isna(output.open_interest.iloc[output_index])
    assert support["validation"]["oi_splice_window_exact_rows"] == 13
    assert support["validation"]["appended_oi_rows"] == 107
    access = support["access_ledger"]["current_s11"]["access"]
    assert access["metrics_selected_row_count"] == 120
    assert access["metrics_overlap_row_count"] == 13
    assert access["metrics_tail_row_count"] == 107


@pytest.mark.parametrize("overlap_index", [0, 6, 12], ids=["first", "interior", "last"])
@pytest.mark.parametrize("missing_side", ["inherited", "selected"])
def test_production_shape_rejects_one_sided_overlap_missingness_in_both_directions(
    tmp_path: Path, overlap_index: int, missing_side: str
) -> None:
    def mutate(frames: dict[str, tuple[pd.DataFrame, bool]]) -> None:
        old_oi = frames["old_open_interest"][0].copy()
        metrics = frames["binance_metrics_open_interest"][0].copy()
        overlap_date = metrics.create_time.iloc[overlap_index]
        if missing_side == "inherited":
            old_oi.loc[old_oi.date.eq(overlap_date), "open_interest"] = np.nan
        else:
            metrics.loc[metrics.create_time.eq(overlap_date), "sum_open_interest"] = np.nan
        _replace_frame(frames, "old_open_interest", old_oi)
        _replace_frame(frames, "binance_metrics_open_interest", metrics)

    with pytest.raises(subject.SourceSupportFailure, match="OI overlap conflict"):
        subject.materialize(production_shape_config(tmp_path, mutate))


@pytest.mark.parametrize("tail_index", [0, 53, 106], ids=["first", "interior", "last"])
@pytest.mark.parametrize("token", ["", "null"], ids=["blank", "null"])
def test_production_shape_rejects_blank_or_null_at_each_tail_position(
    tmp_path: Path, tail_index: int, token: str
) -> None:
    config = production_shape_config(tmp_path)

    def mutate(rows: list[list[str]]) -> None:
        rows[1 + 13 + tail_index][2] = token

    config = _rewrite_metrics_rows(config, mutate)
    with pytest.raises(
        subject.SourceSupportFailure,
        match="missing tail OI|malformed numeric value|non-positive or non-finite tail OI",
    ):
        subject.materialize(config)


def test_all_120_selected_rows_ignore_poison_in_all_five_unused_metrics_columns(
    tmp_path: Path,
) -> None:
    config = production_shape_config(tmp_path)

    def mutate(rows: list[list[str]]) -> None:
        for row_index, row in enumerate(rows[1:], start=1):
            row[3:] = [f"POISON-{row_index}-{column}" for column in range(5)]

    config = _rewrite_metrics_rows(config, mutate)
    support = subject.materialize(config)
    assert support["access_ledger"]["current_s11"]["access"][
        "non_selected_metrics_non_date_semantic_evaluation_count"
    ] == 0
    assert support["validation"]["oi_splice_window_exact_rows"] == 13


def _replace_metrics_raw_csv(
    config: subject.MaterializationConfig, raw_csv: bytes
) -> subject.MaterializationConfig:
    compressed = io.BytesIO()
    with gzip.GzipFile(
        filename="", mode="wb", fileobj=compressed, compresslevel=9, mtime=0
    ) as handle:
        handle.write(raw_csv)
    return _replace_bound_input_bytes(
        config, "binance_metrics_open_interest", compressed.getvalue()
    )


def test_selected_metrics_malformed_csv_is_rejected_by_selected_decode(
    tmp_path: Path,
) -> None:
    config = synthetic_config(tmp_path)
    header = _encode_csv_row(list(subject.METRICS_SCHEMA)).encode() + b"\n"
    config = _replace_metrics_raw_csv(config, header + b'"unterminated\n')
    retained = subject._open_inputs(config, lambda _event: None)
    scan = subject.MetricsDateScan(
        schema=subject.METRICS_SCHEMA,
        row_count=1,
        selected_positions=(0,),
        selected_dates=pd.Series([pd.Timestamp("2026-01-01 01:25:00")]),
    )
    try:
        with pytest.raises(
            subject.SourceSupportFailure, match="metrics selected-window decode failed"
        ):
            subject._decode_metrics_selected_window(
                retained["binance_metrics_open_interest"], scan, lambda _event: None
            )
    finally:
        for item in retained.values():
            os.close(item.fd)


@pytest.mark.parametrize("drift", ["selected_position", "physical_row_count"])
def test_selected_metrics_position_or_second_pass_row_count_drift_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drift: str
) -> None:
    config = synthetic_config(tmp_path)
    original = subject._scan_metrics_dates

    def mutate_scan(
        item: subject._Retained,
        cfg: subject.MaterializationConfig,
        failpoint: Callable[[str], None],
    ) -> subject.MetricsDateScan:
        scan = original(item, cfg, failpoint)
        if drift == "selected_position":
            positions = (*scan.selected_positions[:-1], scan.selected_positions[-1] + 1)
            return replace(scan, selected_positions=positions)
        return replace(scan, row_count=scan.row_count + 1)

    monkeypatch.setattr(subject, "_scan_metrics_dates", mutate_scan)
    with pytest.raises(
        subject.SourceSupportFailure,
        match="metrics selected physical positions drift|metrics selected dates differ",
    ):
        subject.materialize(config)


@pytest.mark.parametrize(
    ("drift", "event"),
    [
        ("metadata", "identity:binance_metrics_open_interest:after_date_scan_rehash"),
        ("path", "identity:binance_metrics_open_interest:after_date_scan_rehash"),
        ("inode", "identity:binance_metrics_open_interest:after_date_scan_rehash"),
        ("bytes", "identity:binance_metrics_open_interest:after_selected_window_decode"),
    ],
)
def test_metrics_metadata_path_inode_or_byte_drift_between_passes_is_rejected(
    tmp_path: Path, drift: str, event: str
) -> None:
    config = synthetic_config(tmp_path)
    metrics = Path(
        next(
            row.path
            for row in config.inputs
            if row.name == "binance_metrics_open_interest"
        )
    )
    original = metrics.read_bytes()
    changed = False

    def inject(name: str) -> None:
        nonlocal changed
        if name != event or changed:
            return
        if drift == "metadata":
            metrics.chmod(0o600)
        elif drift == "path":
            moved = metrics.with_name(metrics.name + ".moved")
            metrics.rename(moved)
            metrics.write_bytes(original)
            metrics.chmod(0o644)
        elif drift == "inode":
            replacement = metrics.with_name(metrics.name + ".replacement")
            replacement.write_bytes(original)
            replacement.chmod(0o644)
            os.replace(replacement, metrics)
        else:
            raw = bytearray(original)
            raw[len(raw) // 2] ^= 1
            metrics.write_bytes(raw)
        changed = True

    with pytest.raises(
        subject.SourceSupportFailure,
        match="identity drift|pathname replacement|rehash differs|byte binding drift",
    ):
        subject.materialize(config, failpoint=inject)
    assert changed


def test_same_size_final_retained_input_byte_drift_is_rejected(tmp_path: Path) -> None:
    config = synthetic_config(tmp_path)
    old_oi = Path(next(row.path for row in config.inputs if row.name == "old_open_interest"))
    original = old_oi.read_bytes()
    changed = False

    def inject(name: str) -> None:
        nonlocal changed
        if name == "checkpoint:8" and not changed:
            raw = bytearray(original)
            index = next(i for i, value in enumerate(raw) if value in b"123456789")
            raw[index] = ord("8") if raw[index] != ord("8") else ord("7")
            assert len(raw) == len(original)
            old_oi.write_bytes(raw)
            changed = True

    with pytest.raises(
        subject.SourceSupportFailure,
        match="final.*input.*drift|byte binding drift|rehash differs",
    ):
        subject.materialize(config, failpoint=inject)
    assert changed


def _install_terminal_publication_mask(
    config: subject.MaterializationConfig, mask: tuple[bool, bool, bool, bool, bool]
) -> None:
    for present, relative in zip(mask, config.output_paths, strict=True):
        if not present:
            continue
        path = config.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
        path.chmod(0o444)


@pytest.mark.parametrize(
    ("prefix_length", "expected"),
    [
        (0, subject.PRE_SENTINEL_FAILURE),
        (1, subject.POST_SENTINEL_PRE_OTHER_OUTPUT_FAILURE),
        (2, subject.PARTIAL_PUBLICATION_FAILURE),
        (3, subject.PARTIAL_PUBLICATION_FAILURE),
        (4, subject.PARTIAL_PUBLICATION_FAILURE),
        (5, subject.PARTIAL_PUBLICATION_FAILURE),
    ],
)
def test_production_terminal_classifier_accepts_zero_sentinel_and_every_prefix_state(
    tmp_path: Path, prefix_length: int, expected: str
) -> None:
    config = synthetic_config(tmp_path)
    _install_terminal_publication_mask(
        config, tuple(index < prefix_length for index in range(5))
    )
    assert subject.classify_terminal_publication_state(config) == expected
    assert subject._classify_terminal_state(config) == expected


@pytest.mark.parametrize(
    "mask",
    [
        (False, True, False, False, False),
        (True, False, True, False, False),
        (True, True, False, True, False),
        (True, True, True, False, True),
    ],
)
def test_production_terminal_classifier_rejects_non_prefix_sets(
    tmp_path: Path, mask: tuple[bool, bool, bool, bool, bool]
) -> None:
    config = synthetic_config(tmp_path)
    _install_terminal_publication_mask(config, mask)
    with pytest.raises(
        subject.SourceSupportFailure,
        match="terminal publication set is not an ordered prefix",
    ):
        subject.classify_terminal_publication_state(config)


@pytest.mark.parametrize("invalid", ["type", "mode", "link"])
def test_production_terminal_classifier_rejects_invalid_type_mode_or_link(
    tmp_path: Path, invalid: str
) -> None:
    config = synthetic_config(tmp_path)
    sentinel = tmp_path / config.attempt_path
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    if invalid == "type":
        sentinel.mkdir()
    else:
        sentinel.write_bytes(b"sentinel")
        sentinel.chmod(0o644 if invalid == "mode" else 0o444)
        if invalid == "link":
            os.link(sentinel, sentinel.with_name("sentinel-hardlink"))
    with pytest.raises(
        subject.SourceSupportFailure, match="terminal publication metadata differs"
    ):
        subject.classify_terminal_publication_state(config)


def test_same_process_pre_sentinel_retry_is_rejected_without_cross_process_guard_file(
    tmp_path: Path,
) -> None:
    """A fresh second process is unauthorized by policy and is intentionally not run.

    The supervisor owns official invocation cardinality because a pre-sentinel
    failure leaves no visible publication.  This unit test proves only the
    implementation's same-process backstop; it must not invent a guard file.
    """
    config = synthetic_config(tmp_path)

    def fail_before_sentinel(name: str) -> None:
        if name == "publish:attempt_sentinel:before_linkat":
            raise RuntimeError("pre-sentinel")

    with pytest.raises(RuntimeError, match="pre-sentinel"):
        subject.materialize(config, failpoint=fail_before_sentinel)
    assert subject.classify_terminal_publication_state(config) == subject.PRE_SENTINEL_FAILURE
    assert not any(path.name.endswith((".lock", ".guard")) for path in tmp_path.rglob("*"))
    with pytest.raises(
        subject.SourceSupportFailure, match="one-shot invocation already exists"
    ):
        subject.materialize(config)


def test_architect_binding_uses_only_process_local_guard_and_existing_output_rejection(
    tmp_path: Path,
) -> None:
    source = Path(subject.__file__).read_text()
    assert "_PROCESS_LOCAL_CONSUMED_INVOCATIONS" in source
    assert "flock" not in source
    assert "guard_file" not in source
    assert "attempt.lock" not in source
    config = synthetic_config(tmp_path)
    existing = tmp_path / config.attempt_path
    existing.write_bytes(b"existing")
    with pytest.raises(subject.SourceSupportFailure, match="already exists"):
        subject.materialize(config)


def test_validate_history_gate_succeeds_read_only_without_opening_any_raw_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(subject.__file__).resolve().parents[1]
    config = subject.official_config(root)
    raw_paths = {
        (Path(row.path) if Path(row.path).is_absolute() else root / row.path).absolute()
        for row in config.inputs
    }
    opened: list[Path] = []
    original = subject._open_nofollow_components

    def audited(path: Path, flags: int) -> int:
        absolute = Path(path).absolute()
        assert absolute not in raw_paths
        opened.append(absolute)
        return original(path, flags)

    monkeypatch.setattr(subject, "_open_nofollow_components", audited)
    subject._validate_history_gate(config)
    expected = {
        root / subject._A11_AUTHORITY[0],
        root / subject._A10_AUTHORITY[0],
        *(root / row[0] for row in subject._T9_EVIDENCE),
        *(root / row[0] for row in subject._T10_EVIDENCE),
    }
    assert set(opened) == expected
    assert raw_paths.isdisjoint(opened)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("a11_digest", "A11 authority binding differs"),
        ("a11_git", "A11 authority binding differs"),
        ("t10_attempt", "G9CB-10 attempt sentinel constants differ"),
        ("t10_terminal", "T10 terminal ledger constants differ"),
    ],
)
def test_validate_history_gate_rejects_practical_authority_git_and_t10_mutations(
    monkeypatch: pytest.MonkeyPatch, mutation: str, match: str
) -> None:
    root = Path(subject.__file__).resolve().parents[1]
    config = subject.official_config(root)
    if mutation == "a11_digest":
        monkeypatch.setattr(
            subject,
            "_A11_AUTHORITY",
            (*subject._A11_AUTHORITY[:2], "0" * 64, subject._A11_AUTHORITY[3]),
        )
    elif mutation == "a11_git":
        original_git = subject._run_git

        def mutated_git(repo: Path, *args: str) -> str:
            if args == ("ls-files", "-s", "--", subject._A11_AUTHORITY[0]):
                return "100755 wrong 0 " + subject._A11_AUTHORITY[0]
            return original_git(repo, *args)

        monkeypatch.setattr(subject, "_run_git", mutated_git)
    else:
        original_read = subject._read_bound_json

        def mutated_read(*args: object, **kwargs: object) -> dict[str, object]:
            payload = original_read(*args, **kwargs)
            relative = args[1]
            if mutation == "t10_attempt" and relative == subject._T10_EVIDENCE[0][0]:
                payload = json.loads(json.dumps(payload))
                payload["identity"] = "G9CB-10-SOURCE-SUPPORT-MUTATED"
            if mutation == "t10_terminal" and relative == subject._T10_EVIDENCE[1][0]:
                payload = json.loads(json.dumps(payload))
                payload["terminal_failure_hash"] = "0" * 64
            return payload

        monkeypatch.setattr(subject, "_read_bound_json", mutated_read)
    with pytest.raises(subject.SourceSupportFailure, match=match):
        subject._validate_history_gate(config)


def _valid_git_gate_answers(config: subject.MaterializationConfig) -> tuple[str, dict[tuple[str, ...], str]]:
    head = "1" * 40
    return head, {
        ("rev-parse", "HEAD"): head,
        ("branch", "--show-current"): config.branch,
        ("rev-parse", "@{upstream}"): head,
        ("rev-list", "--parents", "-n", "1", head): f"{head} {subject.T10}",
        ("rev-parse", f"{subject.T10}^"): subject.A11,
        ("rev-parse", f"{subject.A11}^"): subject.S10,
        ("rev-parse", f"{subject.S10}^"): subject.T9,
        ("diff", "--name-status", subject.S10, subject.A11): "A\tdocs/gross9-structural-clock-bundle-g9cb11-successor-authority-decision-2026-07-31.md",
        ("diff", "--name-status", subject.A11, subject.T10): "A\tresults/gross9_structural_clock_bundle_g9cb10_source_support_attempt_consumed_2026-07-31.json\nA\tresults/gross9_structural_clock_bundle_g9cb10_source_support_terminal_failure_2026-07-31.json",
        ("diff", "--name-status", subject.T10, head): "A\ttests/test_materialize_gross9_structural_clock_g9cb11_sources.py\nA\ttraining/materialize_gross9_structural_clock_g9cb11_sources.py",
        ("ls-files", "-s", "--", "tests/test_materialize_gross9_structural_clock_g9cb11_sources.py"): "100644 blob 0 tests/test_materialize_gross9_structural_clock_g9cb11_sources.py",
        ("ls-files", "-s", "--", "training/materialize_gross9_structural_clock_g9cb11_sources.py"): "100644 blob 0 training/materialize_gross9_structural_clock_g9cb11_sources.py",
        ("status", "--porcelain=v1", "--untracked-files=all"): "",
    }


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("branch", "official branch differs"),
        ("upstream", "HEAD and upstream differ"),
        ("parent", "S11 direct parent differs"),
        ("a11_parent", "A11/S10 topology differs"),
        ("t10_diff", "exact T10 two-file diff differs"),
        ("mode", "S11 Git mode or index binding differs"),
        ("dirty", "index or worktree is not clean"),
    ],
)
def test_git_gate_rejects_branch_topology_diff_mode_and_cleanliness_mutations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    match: str,
) -> None:
    config = replace(synthetic_config(tmp_path), branch=subject.BRANCH)
    head, answers = _valid_git_gate_answers(config)
    if mutation == "branch":
        answers[("branch", "--show-current")] = "wrong"
    elif mutation == "upstream":
        answers[("rev-parse", "@{upstream}")] = "2" * 40
    elif mutation == "parent":
        answers[("rev-list", "--parents", "-n", "1", head)] = f"{head} {'2' * 40}"
    elif mutation == "a11_parent":
        answers[("rev-parse", f"{subject.A11}^")] = "2" * 40
    elif mutation == "t10_diff":
        answers[("diff", "--name-status", subject.A11, subject.T10)] = "A\tonly-one.json"
    elif mutation == "mode":
        key = ("ls-files", "-s", "--", "tests/test_materialize_gross9_structural_clock_g9cb11_sources.py")
        answers[key] = "100755 blob 0 tests/test_materialize_gross9_structural_clock_g9cb11_sources.py"
    else:
        answers[("status", "--porcelain=v1", "--untracked-files=all")] = "?? residue"
    monkeypatch.setattr(subject, "_run_git", lambda _root, *args: answers[args])
    with pytest.raises(subject.SourceSupportFailure, match=match):
        subject._validate_git_gate(config)


def test_canonical_json_hash_and_final_lf_fixed_vectors() -> None:
    assert subject.canonical_json_bytes({"é": "한"}) == bytes.fromhex(
        "7b22c3a9223a22ed959c227d0a"
    )
    assert subject.canonical_json_bytes(
        {"b": 1, "a": 2}, trailing_lf=False
    ) == b'{"a":2,"b":1}'
    assert subject.canonical_json_bytes({"b": 1, "a": 2}) == b'{"a":2,"b":1}\n'
    assert subject.object_hash(
        {"identity": "G9CB-11", "version": subject.VERSION}, "support_hash"
    ) == "c35212e5facd9bb62a67af8517ae73f7d20905cc224c31162ed0ab11221832b4"
    with pytest.raises(ValueError):
        subject.canonical_json_bytes({"invalid": float("nan")})
    with pytest.raises(subject.SourceSupportFailure, match="duplicate JSON key"):
        subject._unique_json([("same", 1), ("same", 2)])


def test_publication_uses_otmpfile_proc_fd_symlink_follow_and_never_at_empty_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = Path(subject.__file__).read_text()
    assert "os.O_TMPFILE" in source
    assert 'f"/proc/self/fd/{fd}"' in source
    assert subject._AT_SYMLINK_FOLLOW == 0x400
    assert "AT_EMPTY_PATH" not in source
    calls: list[tuple[object, ...]] = []

    class FakeLibc:
        def linkat(self, *args: object) -> int:
            calls.append(args)
            return 0

    monkeypatch.setattr(subject, "_LIBC", FakeLibc())
    subject._linkat(17, 23, "leaf")
    assert calls == [
        (subject._AT_FDCWD, b"/proc/self/fd/17", 23, b"leaf", subject._AT_SYMLINK_FOLLOW)
    ]


_PROCESS_ROW_KEYS = [
    "old_market_rows_opened",
    "replacement_market_date_rows_scanned",
    "replacement_market_tail_rows_opened",
    "funding_rows_opened",
    "premium_rows_opened",
    "old_open_interest_rows_opened",
    "binance_metrics_open_interest_date_rows_scanned",
    "binance_metrics_open_interest_selected_window_rows_opened",
    "rank7_spot_premium_5m_rows_opened",
]


def _materialize_with_access_mutation(
    config: subject.MaterializationConfig,
    monkeypatch: pytest.MonkeyPatch,
    mutate: Callable[[dict[str, object]], None],
) -> None:
    original = subject._build_access_ledger

    def mutant(*args: object, **kwargs: object) -> dict[str, object]:
        ledger = original(*args, **kwargs)
        mutate(ledger)
        _rehash_mutated_access(ledger)
        return ledger

    monkeypatch.setattr(subject, "_build_access_ledger", mutant)
    with pytest.raises(
        subject.SourceSupportFailure, match="access ledger exact schema"
    ):
        subject.materialize(config)


@pytest.mark.parametrize("key", [*_PROCESS_ROW_KEYS, "source_value_rows_opened"])
def test_each_nine_addend_placeholder_and_total_rejects_omission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str
) -> None:
    _materialize_with_access_mutation(
        synthetic_config(tmp_path),
        monkeypatch,
        lambda ledger: ledger["process_local"].pop(key),
    )


@pytest.mark.parametrize("key", _PROCESS_ROW_KEYS)
def test_each_nine_addend_rejects_plus_one_stage_value_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str
) -> None:
    def mutate(ledger: dict[str, object]) -> None:
        process = ledger["process_local"]
        process[key] += 1
        process["source_value_rows_opened"] += 1

    _materialize_with_access_mutation(
        synthetic_config(tmp_path), monkeypatch, mutate
    )


@pytest.mark.parametrize(
    "invalid",
    [None, "UNRESOLVED", True, 1.0, "1", -1, 2**64],
    ids=["null", "unresolved", "boolean", "float", "string", "negative", "overflow"],
)
def test_source_value_total_rejects_every_invalid_placeholder_type_or_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid: object
) -> None:
    _materialize_with_access_mutation(
        synthetic_config(tmp_path),
        monkeypatch,
        lambda ledger: ledger["process_local"].__setitem__(
            "source_value_rows_opened", invalid
        ),
    )


@pytest.mark.parametrize(
    ("field_path", "invalid"),
    [
        (("replay_guard", "a11", "commit"), "a" * 39),
        (("replay_guard", "t10", "terminal_ledger_sha256"), "A" * 64),
        (("replay_guard", "t10", "terminal_failure_hash"), "g" * 64),
        (("replay_guard", "s11", "attempt_hash"), "0" * 63),
        (("replay_guard", "attempt_hashes", "current_s11"), "UNRESOLVED"),
        (("current_s11", "attempt_sentinel", "size_bytes"), True),
        (("historical_s10", "execution", "invocation_count"), "1"),
    ],
)
def test_recursive_wrong_length_case_charset_stage_and_uint_mutants_fail_even_rehashed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field_path: tuple[str, ...],
    invalid: object,
) -> None:
    def mutate(ledger: dict[str, object]) -> None:
        target: dict[str, object] = ledger
        for key in field_path[:-1]:
            target = target[key]
        target[field_path[-1]] = invalid

    _materialize_with_access_mutation(
        synthetic_config(tmp_path), monkeypatch, mutate
    )


def test_nine_addend_ast_binding_is_exact_ordered_and_has_no_early_total() -> None:
    source = Path(subject.__file__).read_text()
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_expected_access_ledger"
    )
    addends_assignment = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "addends" for target in node.targets)
    )
    assert isinstance(addends_assignment.value, ast.List)
    expressions = [ast.unparse(element) for element in addends_assignment.value.elts]
    assert expressions == [
        "decoded_rows['old_market']",
        "replacement_scan.row_count",
        "len(replacement_scan.tail_positions)",
        "decoded_rows['funding']",
        "decoded_rows['premium']",
        "decoded_rows['old_open_interest']",
        "metrics_scan.row_count",
        "len(metrics_scan.selected_positions)",
        "decoded_rows['rank7_spot_premium_5m']",
    ]
    total_assignments = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.keyword)
        and node.arg == "source_value_rows_opened"
    ]
    assert total_assignments == []
    assert source.count('"source_value_rows_opened": sum(addends)') == 1

    process_assignment = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "process_local"
            for target in node.targets
        )
    )
    assert isinstance(process_assignment.value, ast.Dict)
    process_expressions = {
        key.value: ast.unparse(value)
        for key, value in zip(
            process_assignment.value.keys, process_assignment.value.values, strict=True
        )
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    physical_provenance = {
        "replacement_market_date_rows_scanned": "addends[1]",
        "funding_rows_opened": "addends[3]",
        "premium_rows_opened": "addends[4]",
        "binance_metrics_open_interest_date_rows_scanned": "addends[6]",
        "rank7_spot_premium_5m_rows_opened": "addends[8]",
    }
    assert {
        key: process_expressions[key] for key in physical_provenance
    } == physical_provenance
    baseline_hash = subject.object_hash(
        {"bindings": physical_provenance}, "provenance_hash"
    )
    same_serialized_uints = dict.fromkeys(physical_provenance, 7)
    for destination in physical_provenance:
        for source_key in physical_provenance:
            if source_key == destination:
                continue
            assert same_serialized_uints[destination] == same_serialized_uints[source_key]
            mutant = copy.deepcopy(physical_provenance)
            mutant[destination] = physical_provenance[source_key]
            assert subject.object_hash(
                {"bindings": mutant}, "provenance_hash"
            ) != baseline_hash


def test_no_economics_runtime_open_surface_and_all_zero_counters_are_independent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = synthetic_config(tmp_path)
    opened: list[Path] = []
    original_open = subject._open_nofollow_components

    def audited(path: Path, flags: int) -> int:
        opened.append(Path(path).absolute())
        return original_open(path, flags)

    def forbidden_subprocess(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("synthetic materialization launched a subprocess")

    monkeypatch.setattr(subject, "_open_nofollow_components", audited)
    monkeypatch.setattr(subject.subprocess, "run", forbidden_subprocess)
    support = subject.materialize(config)
    raw = {Path(row.path).absolute() for row in config.inputs}
    outputs = {config.root / relative for relative in config.output_paths}
    output_parents = {path.parent for path in outputs}
    assert set(opened) <= raw | outputs | output_parents
    forbidden_path_tokens = (
        "candidate", "comparator", "schedule", "signal", "return", "pnl",
        "cagr", "mdd", "drawdown", "economics",
    )
    relative_opened = [path.relative_to(config.root) for path in opened]
    assert not any(
        token in str(path).lower()
        for path in relative_opened
        for token in forbidden_path_tokens
    )
    zero_keys = {
        "candidate_value_rows_opened", "comparator_value_rows_opened",
        "feature_value_rows_opened", "schedule_value_rows_opened",
        "signal_value_rows_opened", "return_value_rows_opened",
        "pnl_value_rows_opened", "cagr_evaluation_count", "mdd_evaluation_count",
        "drawdown_evaluation_count", "economic_value_rows_opened",
        "economic_evaluation_count",
    }
    current = support["access_ledger"]["current_s11"]["access"]
    process = support["access_ledger"]["process_local"]
    assert {key: current[key] for key in zero_keys} == {key: 0 for key in zero_keys}
    assert {key: process[key] for key in zero_keys} == {key: 0 for key in zero_keys}


_SEVEN_INPUT_NAMES = tuple(row.name for row in subject.OFFICIAL_INPUTS)


@pytest.mark.parametrize("source_name", _SEVEN_INPUT_NAMES)
@pytest.mark.parametrize("phase", ("initial", "final"))
@pytest.mark.parametrize(
    "drift", ("metadata", "path", "inode", "size", "same_size_bytes")
)
def test_every_input_slot_rejects_each_initial_and_final_binding_drift(
    tmp_path: Path, source_name: str, phase: str, drift: str
) -> None:
    config = synthetic_config(tmp_path)
    source = Path(next(row.path for row in config.inputs if row.name == source_name))
    original = source.read_bytes()
    event = (
        f"identity:{source_name}:before_hash" if phase == "initial" else "checkpoint:8"
    )
    changed = False

    def inject(name: str) -> None:
        nonlocal changed
        if changed or name != event:
            return
        if drift == "metadata":
            source.chmod(0o600)
        elif drift == "path":
            source.rename(source.with_name(source.name + ".moved"))
        elif drift == "inode":
            replacement = source.with_name(source.name + ".replacement")
            replacement.write_bytes(original)
            replacement.chmod(0o644)
            os.replace(replacement, source)
        elif drift == "size":
            source.write_bytes(original + b"x")
        else:
            raw = bytearray(original)
            raw[len(raw) // 2] ^= 1
            assert len(raw) == len(original)
            source.write_bytes(raw)
        changed = True

    with pytest.raises(
        subject.SourceSupportFailure,
        match="metadata drift|path-edge drift|pathname replacement|identity drift|byte binding drift|rehash differs",
    ):
        subject.materialize(config, failpoint=inject)
    assert changed
    expected_publications = 0 if phase == "initial" else 5
    assert sum((tmp_path / relative).exists() for relative in config.output_paths) == expected_publications


def test_final_input_reauthentication_slot_events_are_exactly_ordered_zero_through_six(
    tmp_path: Path,
) -> None:
    events: list[str] = []

    def record(name: str) -> None:
        if name.startswith("final_input_reauthentication:"):
            events.append(name)

    subject.materialize(synthetic_config(tmp_path), failpoint=record)
    assert events == [
        event
        for slot in range(7)
        for event in (
            f"final_input_reauthentication:{slot}:before_hash",
            f"final_input_reauthentication:{slot}:complete",
        )
    ]


@pytest.mark.parametrize("slot", range(7))
@pytest.mark.parametrize("phase", ("before_hash", "complete"))
def test_every_final_input_reauthentication_failpoint_closes_and_preserves_outputs(
    tmp_path: Path, slot: int, phase: str
) -> None:
    config = synthetic_config(tmp_path)
    baseline_fds = _fd_count()
    retained_evidence: dict[str, tuple[bytes, int, int, int, int]] = {}
    event = f"final_input_reauthentication:{slot}:{phase}"

    def inject(name: str) -> None:
        if name == event:
            retained_evidence.update(_published_snapshot(config))
            raise RuntimeError(name)

    with pytest.raises(RuntimeError, match=event):
        subject.materialize(config, failpoint=inject)
    assert _fd_count() == baseline_fds
    present = tuple(
        relative for relative in config.output_paths if (tmp_path / relative).exists()
    )
    assert present == config.output_paths
    for relative in present:
        info = os.lstat(tmp_path / relative)
        assert stat.S_IMODE(info.st_mode) == 0o444
        assert info.st_nlink == 1
    assert _published_snapshot(config) == retained_evidence


@pytest.mark.parametrize(
    "failed_gate",
    (
        "command", "first_bytecode", "git", "history", "output_absence",
        "source_opening", "publication", "final_bytecode",
    ),
)
def test_each_gate_failure_has_exact_prefix_and_no_unauthorized_next_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_gate: str
) -> None:
    config = replace(synthetic_config(tmp_path), enforce_repository_gates=True)
    events: list[str] = []
    gate_order = (
        "command", "first_bytecode", "git", "history", "output_absence",
        "source_opening", "publication", "final_bytecode",
    )
    original_absence = subject._validate_output_absence
    original_open_inputs = subject._open_inputs
    original_publish = subject._publish
    bytecode_calls = 0
    publication_seen = False

    def gate(name: str, result: object = None) -> Callable[..., object]:
        def invoke(*_args: object, **_kwargs: object) -> object:
            events.append(name)
            if name == failed_gate:
                raise RuntimeError(f"failed:{name}")
            return result

        return invoke

    def bytecode(_root: Path) -> None:
        nonlocal bytecode_calls
        bytecode_calls += 1
        name = "first_bytecode" if bytecode_calls == 1 else "final_bytecode"
        events.append(name)
        if name == failed_gate:
            raise RuntimeError(f"failed:{name}")

    def output_absence(current: subject.MaterializationConfig) -> None:
        events.append("output_absence")
        if failed_gate == "output_absence":
            raise RuntimeError("failed:output_absence")
        original_absence(current)

    def source_opening(
        current: subject.MaterializationConfig, inject: Callable[[str], None]
    ) -> dict[str, subject._Retained]:
        events.append("source_opening")
        if failed_gate == "source_opening":
            raise RuntimeError("failed:source_opening")
        return original_open_inputs(current, inject)

    def publication(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal publication_seen
        if not publication_seen:
            publication_seen = True
            events.append("publication")
            if failed_gate == "publication":
                raise RuntimeError("failed:publication")
        return original_publish(*args, **kwargs)

    monkeypatch.setattr(subject, "_validate_command_and_root", gate("command"))
    monkeypatch.setattr(subject, "_validate_bytecode_first_gate", bytecode)
    monkeypatch.setattr(subject, "_validate_git_gate", gate("git", "synthetic-s11"))
    monkeypatch.setattr(subject, "_validate_history_gate", gate("history"))
    monkeypatch.setattr(subject, "_validate_output_absence", output_absence)
    monkeypatch.setattr(subject, "_open_inputs", source_opening)
    monkeypatch.setattr(subject, "_publish", publication)

    with pytest.raises(RuntimeError, match=f"failed:{failed_gate}"):
        subject.materialize(config)
    failed_index = gate_order.index(failed_gate)
    assert events == list(gate_order[: failed_index + 1])
    expected_count = 5 if failed_gate == "final_bytecode" else 0
    present = tuple(
        relative for relative in config.output_paths if (tmp_path / relative).exists()
    )
    assert present == config.output_paths[:expected_count]
    for relative in present:
        info = os.lstat(tmp_path / relative)
        assert stat.S_IMODE(info.st_mode) == 0o444
        assert info.st_nlink == 1


@pytest.mark.parametrize("output_slot", range(5))
def test_each_expected_output_presence_stops_before_source_open_or_sentinel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, output_slot: int
) -> None:
    config = synthetic_config(tmp_path)
    existing = tmp_path / config.output_paths[output_slot]
    existing.write_bytes(b"preexisting")
    monkeypatch.setattr(
        subject,
        "_open_inputs",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("source opened")),
    )
    with pytest.raises(subject.SourceSupportFailure, match="already exists"):
        subject.materialize(config)
    assert not (tmp_path / config.attempt_path).exists() or output_slot == 0


@pytest.mark.parametrize("constant_name", ("_A11_AUTHORITY", "_A10_AUTHORITY"))
@pytest.mark.parametrize("field", ("path", "size", "sha256", "git_blob", "mode"))
def test_a11_a10_history_bindings_reject_every_file_identity_mutation(
    monkeypatch: pytest.MonkeyPatch, constant_name: str, field: str
) -> None:
    root = Path(subject.__file__).resolve().parents[1]
    config = subject.official_config(root)
    binding = list(getattr(subject, constant_name))
    target = (root / binding[0]).absolute()
    if field == "path":
        binding[0] += ".missing"
        monkeypatch.setattr(subject, constant_name, tuple(binding))
    elif field == "size":
        binding[1] += 1
        monkeypatch.setattr(subject, constant_name, tuple(binding))
    elif field == "sha256":
        binding[2] = "0" * 64
        monkeypatch.setattr(subject, constant_name, tuple(binding))
    elif field == "git_blob":
        binding[3] = "0" * 40
        monkeypatch.setattr(subject, constant_name, tuple(binding))
    else:
        opened: set[int] = set()
        original_open = subject._open_nofollow_components
        original_fstat = subject.os.fstat

        def track_open(path: Path, flags: int) -> int:
            fd = original_open(path, flags)
            if Path(path).absolute() == target:
                opened.add(fd)
            return fd

        def wrong_mode(fd: int) -> os.stat_result:
            info = original_fstat(fd)
            if fd in opened:
                values = list(info)
                values[0] = stat.S_IFREG | 0o600
                return os.stat_result(values)
            return info

        monkeypatch.setattr(subject, "_open_nofollow_components", track_open)
        monkeypatch.setattr(subject.os, "fstat", wrong_mode)

    with pytest.raises((subject.SourceSupportFailure, OSError)):
        subject._validate_history_gate(config)


@pytest.mark.parametrize("constant_name", ("_T9_EVIDENCE", "_T10_EVIDENCE"))
@pytest.mark.parametrize("row_index", (0, 1), ids=("attempt", "terminal"))
@pytest.mark.parametrize("field", ("path", "size", "sha256", "git_blob", "mode"))
def test_t9_t10_history_evidence_rejects_every_binding_mutation(
    monkeypatch: pytest.MonkeyPatch, constant_name: str, row_index: int, field: str
) -> None:
    root = Path(subject.__file__).resolve().parents[1]
    config = subject.official_config(root)
    evidence = [list(row) for row in getattr(subject, constant_name)]
    target_relative = evidence[row_index][0]
    if field == "path":
        evidence[row_index][0] += ".missing"
    elif field == "size":
        evidence[row_index][1] += 1
    elif field == "sha256":
        evidence[row_index][2] = "0" * 64
    elif field == "git_blob":
        evidence[row_index][3] = "0" * 40
    else:
        original = subject._read_bound_json

        def wrong_mode(*args: object, **kwargs: object) -> dict[str, object]:
            if args[1] == target_relative:
                kwargs["mode"] = 0o644
            return original(*args, **kwargs)

        monkeypatch.setattr(subject, "_read_bound_json", wrong_mode)
    if field != "mode":
        monkeypatch.setattr(subject, constant_name, tuple(tuple(row) for row in evidence))

    with pytest.raises((subject.SourceSupportFailure, OSError)):
        subject._validate_history_gate(config)


@pytest.mark.parametrize("constant_name", ("_T9_EVIDENCE", "_T10_EVIDENCE"))
@pytest.mark.parametrize("row_index", (0, 1), ids=("attempt", "terminal"))
def test_each_t9_t10_evidence_payload_is_independently_literal_bound(
    monkeypatch: pytest.MonkeyPatch, constant_name: str, row_index: int
) -> None:
    root = Path(subject.__file__).resolve().parents[1]
    config = subject.official_config(root)
    target = getattr(subject, constant_name)[row_index][0]
    original = subject._read_bound_json

    def mutate(*args: object, **kwargs: object) -> dict[str, object]:
        payload = original(*args, **kwargs)
        if args[1] == target:
            payload = copy.deepcopy(payload)
            first_key = next(iter(payload))
            payload[first_key] = "mutated"
        return payload

    monkeypatch.setattr(subject, "_read_bound_json", mutate)
    with pytest.raises(subject.SourceSupportFailure):
        subject._validate_history_gate(config)


@pytest.mark.parametrize(
    "relative",
    (*subject._G9CB9_PERMANENT_ABSENCES, *subject._G9CB10_PERMANENT_ABSENCES),
)
def test_each_historical_permanent_absence_is_an_independent_gate(
    monkeypatch: pytest.MonkeyPatch, relative: str
) -> None:
    root = Path(subject.__file__).resolve().parents[1]
    target = (root / relative).absolute()
    original_exists = Path.exists

    def injected_exists(path: Path) -> bool:
        return True if path.absolute() == target else original_exists(path)

    monkeypatch.setattr(Path, "exists", injected_exists)
    with pytest.raises(subject.SourceSupportFailure, match="permanent.*absence differs"):
        subject._validate_history_gate(subject.official_config(root))


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("t10_parent", "T10/A11 topology differs"),
        ("a11_parent", "A11/S10 topology differs"),
        ("s10_parent", "S10/T9 topology differs"),
        ("a11_diff", "exact A11 one-file diff differs"),
        ("t10_diff", "exact T10 two-file diff differs"),
        ("s11_diff", "exact S11 two-file diff differs"),
    ],
)
def test_git_gate_rejects_every_missing_topology_and_diff_edge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str, match: str
) -> None:
    config = replace(synthetic_config(tmp_path), branch=subject.BRANCH)
    head, answers = _valid_git_gate_answers(config)
    keys = {
        "t10_parent": ("rev-parse", f"{subject.T10}^"),
        "a11_parent": ("rev-parse", f"{subject.A11}^"),
        "s10_parent": ("rev-parse", f"{subject.S10}^"),
        "a11_diff": ("diff", "--name-status", subject.S10, subject.A11),
        "t10_diff": ("diff", "--name-status", subject.A11, subject.T10),
        "s11_diff": ("diff", "--name-status", subject.T10, head),
    }
    answers[keys[mutation]] = ""
    monkeypatch.setattr(subject, "_run_git", lambda _root, *args: answers[args])
    with pytest.raises(subject.SourceSupportFailure, match=match):
        subject._validate_git_gate(config)


def _walk_json(value: object, path: tuple[object, ...] = ()) -> list[tuple[tuple[object, ...], object]]:
    rows = [(path, value)]
    if isinstance(value, dict):
        for key, member in value.items():
            rows.extend(_walk_json(member, (*path, key)))
    elif isinstance(value, list):
        for index, member in enumerate(value):
            rows.extend(_walk_json(member, (*path, index)))
    return rows


def _json_at(value: object, path: tuple[object, ...]) -> object:
    for part in path:
        value = value[part]  # type: ignore[index]
    return value


def _set_json_at(value: object, path: tuple[object, ...], replacement: object) -> None:
    parent = _json_at(value, path[:-1])
    parent[path[-1]] = replacement  # type: ignore[index]


def _rehash_access_mutant(value: dict[str, object], changed: tuple[object, ...]) -> None:
    replay = value.get("replay_guard")
    replay_hash_path = ("replay_guard", "replay_guard_hash")
    if isinstance(replay, dict) and changed != replay_hash_path:
        replay["replay_guard_hash"] = subject.object_hash(replay, "replay_guard_hash")
    if changed != ("access_ledger_hash",):
        value["access_ledger_hash"] = subject.object_hash(value, "access_ledger_hash")


def test_recursive_normative_schema_types_literals_arrays_placeholders_and_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}
    original_validate = subject._validate_access_ledger

    def capture(
        ledger: dict[str, object], config: subject.MaterializationConfig, **kwargs: object
    ) -> None:
        captured.update(config=config, kwargs=kwargs)
        original_validate(ledger, config, **kwargs)

    monkeypatch.setattr(subject, "_validate_access_ledger", capture)
    ledger = subject.materialize(production_shape_config(tmp_path))["access_ledger"]
    config = captured["config"]
    kwargs = captured["kwargs"]
    assert isinstance(config, subject.MaterializationConfig)
    assert isinstance(kwargs, dict)

    # Independent PRD literals and ordered schemas, not implementation-owned expected objects.
    assert list(ledger) == [
        "schema_version", "ledger_kind", "historical_s9", "historical_s10",
        "current_s11", "process_local", "replay_guard", "access_ledger_hash",
    ]
    assert ledger["schema_version"] == 1
    assert ledger["ledger_kind"] == "gross9_structural_clock_bundle_g9cb11_access_v1"
    assert ledger["current_s11"]["identity"] == "G9CB-11-SOURCE-SUPPORT"
    assert ledger["current_s11"]["access"]["decode_passes"] == [
        "old_market", "replacement_market_date_scan", "replacement_market_tail",
        "funding", "premium", "old_open_interest",
        "binance_metrics_open_interest_date_scan",
        "binance_metrics_open_interest_selected_window", "rank7_spot_premium_5m",
    ]
    assert ledger["historical_s10"]["access"]["decode_passes"] == [
        "old_market", "replacement_market_date_scan", "replacement_market_tail",
        "funding", "premium", "old_open_interest",
        "binance_metrics_open_interest", "rank7_spot_premium_5m",
    ]
    assert list(ledger["historical_s10"]) == [
        "identity", "status", "authority", "seal_authority", "implementation",
        "attempt_sentinel", "terminal_ledger", "execution", "failure", "access",
        "output_state",
    ]
    assert list(ledger["current_s11"]) == [
        "identity", "authority", "implementation", "attempt_sentinel", "execution", "access"
    ]
    assert list(ledger["replay_guard"]) == [
        "a11", "t10", "s11", "identities", "attempt_hashes",
        "terminal_failure_hashes", "expected_output_paths",
        "pairwise_output_intersections", "identities_pairwise_distinct",
        "attempt_hashes_pairwise_distinct", "replay_guard_hash",
    ]
    assert ledger["process_local"]["stage"] == "S11"
    assert ledger["process_local"]["slot"] == 0
    assert ledger["current_s11"]["access"]["metrics_selected_row_count"] == 120
    assert ledger["current_s11"]["access"]["metrics_overlap_row_count"] == 13
    assert ledger["current_s11"]["access"]["metrics_tail_row_count"] == 107

    def rejected(mutant: dict[str, object], changed: tuple[object, ...]) -> None:
        _rehash_access_mutant(mutant, changed)
        with pytest.raises(subject.SourceSupportFailure, match="access ledger exact schema"):
            original_validate(mutant, config, **kwargs)

    nodes = _walk_json(ledger)
    for path, node in nodes:
        if isinstance(node, dict):
            for key in tuple(node):
                mutant = copy.deepcopy(ledger)
                del _json_at(mutant, path)[key]
                rejected(mutant, (*path, key))
            mutant = copy.deepcopy(ledger)
            _json_at(mutant, path)["__additional__"] = 0
            rejected(mutant, (*path, "__additional__"))
        elif isinstance(node, list):
            if node:
                for index in range(len(node)):
                    mutant = copy.deepcopy(ledger)
                    del _json_at(mutant, path)[index]
                    rejected(mutant, (*path, index))
                mutant = copy.deepcopy(ledger)
                changed_list = _json_at(mutant, path)
                changed_list.append(copy.deepcopy(changed_list[0]))
                rejected(mutant, (*path, len(node)))
                if len(node) > 1:
                    mutant = copy.deepcopy(ledger)
                    _set_json_at(mutant, path, list(reversed(_json_at(mutant, path))))
                    rejected(mutant, path)
            else:
                mutant = copy.deepcopy(ledger)
                _json_at(mutant, path).append("nonempty")
                rejected(mutant, (*path, 0))
        elif path and path[-1] not in ("replay_guard_hash", "access_ledger_hash"):
            replacement = (
                0 if isinstance(node, str) else "wrong" if isinstance(node, int) else 0
            )
            mutant = copy.deepcopy(ledger)
            _set_json_at(mutant, path, replacement)
            rejected(mutant, path)

    # Explicit recursive rename/merge/swap/collapse/substitution mutants.
    for path, node in nodes:
        if isinstance(node, dict):
            for key in tuple(node):
                mutant = copy.deepcopy(ledger)
                target = _json_at(mutant, path)
                target[f"renamed_{key}"] = target.pop(key)
                rejected(mutant, (*path, f"renamed_{key}"))
            keys = list(node)
            if len(keys) >= 2:
                mutant = copy.deepcopy(ledger)
                target = _json_at(mutant, path)
                first, second = keys[:2]
                target[first] = [target[first], target.pop(second)]
                rejected(mutant, (*path, first))
                swappable = next(
                    (
                        (left, right)
                        for index, left in enumerate(keys)
                        for right in keys[index + 1 :]
                        if type(node[left]) is type(node[right])
                        and node[left] != node[right]
                    ),
                    None,
                )
                if swappable is not None:
                    mutant = copy.deepcopy(ledger)
                    target = _json_at(mutant, path)
                    left, right = swappable
                    target[left], target[right] = target[right], target[left]
                    rejected(mutant, (*path, left))
        if path and isinstance(node, (dict, list)) and node:
            mutant = copy.deepcopy(ledger)
            collapsed = next(iter(node.values())) if isinstance(node, dict) else node[0]
            _set_json_at(mutant, path, copy.deepcopy(collapsed))
            rejected(mutant, path)

    leaves = [
        (path, node)
        for path, node in nodes
        if path and not isinstance(node, (dict, list))
    ]
    for path, node in leaves:
        donor = next(
            (
                candidate
                for other_path, candidate in leaves
                if other_path != path
                and type(candidate) is type(node)
                and candidate != node
            ),
            None,
        )
        if donor is not None:
            mutant = copy.deepcopy(ledger)
            _set_json_at(mutant, path, donor)
            rejected(mutant, path)

    implementation_arrays = (
        ("current_s11", "implementation", "paths"),
        ("replay_guard", "s11", "implementation_paths"),
    )
    for path in implementation_arrays:
        original = list(_json_at(ledger, path))
        variants = (
            [], original[:1], [*original, "extra/path.py"],
            [original[1], original[0]], [original[0], original[0]],
            ["wrong/generation.py", original[1]], [0, original[1]],
        )
        for variant in variants:
            mutant = copy.deepcopy(ledger)
            _set_json_at(mutant, path, variant)
            rejected(mutant, path)

    output_arrays = [
        ("replay_guard", "expected_output_paths", generation)
        for generation in ("historical_s9", "historical_s10", "current_s11")
    ]
    for index, path in enumerate(output_arrays):
        original = list(_json_at(ledger, path))
        cross_generation = list(_json_at(ledger, output_arrays[(index + 1) % 3]))
        variants = (
            original[:4], [*original, "results/extra.json"],
            [original[1], original[0], *original[2:]],
            [original[0], original[0], *original[2:]], cross_generation,
            [0, *original[1:]],
        )
        for variant in variants:
            mutant = copy.deepcopy(ledger)
            _set_json_at(mutant, path, variant)
            rejected(mutant, path)

    for path in (
        ("historical_s10", "access", "decode_passes"),
        ("current_s11", "access", "decode_passes"),
    ):
        original = list(_json_at(ledger, path))
        for variant in (
            original[:-1], [*original, "extra_pass"],
            [original[1], original[0], *original[2:]],
            [original[0], original[0], *original[2:]], original[:1],
        ):
            mutant = copy.deepcopy(ledger)
            _set_json_at(mutant, path, variant)
            rejected(mutant, path)

    git_oids = [
        (path, node) for path, node in leaves
        if isinstance(node, str) and len(node) == 40 and all(c in "0123456789abcdef" for c in node)
    ]
    sha256s = [
        (path, node) for path, node in leaves
        if isinstance(node, str) and len(node) == 64 and all(c in "0123456789abcdef" for c in node)
    ]
    uints = [(path, node) for path, node in leaves if isinstance(node, int) and not isinstance(node, bool)]
    for path, _node in git_oids:
        for invalid in ("a" * 39, "a" * 41, "A" * 40, "g" * 40, "", None, "UNRESOLVED"):
            mutant = copy.deepcopy(ledger)
            _set_json_at(mutant, path, invalid)
            rejected(mutant, path)
    for path, _node in sha256s:
        for invalid in ("a" * 63, "a" * 65, "A" * 64, "g" * 64, "", None, "UNRESOLVED"):
            mutant = copy.deepcopy(ledger)
            _set_json_at(mutant, path, invalid)
            rejected(mutant, path)
    for path, _node in uints:
        for invalid in (-1, 1.0, True, "1", None, 2**64, "UNRESOLVED"):
            mutant = copy.deepcopy(ledger)
            _set_json_at(mutant, path, invalid)
            rejected(mutant, path)

    wrong_stage_pairs = [
        (("current_s11", "authority", "commit"), ("current_s11", "implementation", "commit")),
        (("current_s11", "implementation", "commit"), ("current_s11", "implementation", "parent_commit")),
        (("replay_guard", "s11", "parent_commit"), ("replay_guard", "a11", "commit")),
        (("replay_guard", "attempt_hashes", "current_s11"), ("replay_guard", "attempt_hashes", "historical_s10")),
        (("replay_guard", "terminal_failure_hashes", "historical_s10"), ("replay_guard", "attempt_hashes", "historical_s10")),
        (("historical_s10", "terminal_ledger", "sha256"), ("historical_s10", "terminal_ledger", "terminal_failure_hash")),
        (("historical_s10", "terminal_ledger", "size_bytes"), ("historical_s10", "attempt_sentinel", "size_bytes")),
    ]
    for destination, source_path in wrong_stage_pairs:
        mutant = copy.deepcopy(ledger)
        _set_json_at(mutant, destination, _json_at(mutant, source_path))
        rejected(mutant, destination)

    commit_stages = [
        [
            ("historical_s10", "seal_authority", "commit"),
            ("current_s11", "authority", "commit"),
            ("replay_guard", "a11", "commit"),
        ],
        [
            ("historical_s10", "terminal_ledger", "commit"),
            ("current_s11", "implementation", "parent_commit"),
            ("replay_guard", "t10", "commit"),
            ("replay_guard", "s11", "parent_commit"),
        ],
        [
            ("current_s11", "implementation", "commit"),
            ("replay_guard", "s11", "commit"),
        ],
    ]
    hash_stages = [
        [
            ("historical_s10", "attempt_sentinel", "attempt_hash"),
            ("replay_guard", "t10", "attempt_hash"),
            ("replay_guard", "attempt_hashes", "historical_s10"),
        ],
        [
            ("current_s11", "attempt_sentinel", "attempt_hash"),
            ("replay_guard", "s11", "attempt_hash"),
            ("replay_guard", "attempt_hashes", "current_s11"),
        ],
        [
            ("historical_s10", "terminal_ledger", "sha256"),
            ("replay_guard", "t10", "terminal_ledger_sha256"),
        ],
        [
            ("historical_s10", "terminal_ledger", "terminal_failure_hash"),
            ("replay_guard", "t10", "terminal_failure_hash"),
            ("replay_guard", "terminal_failure_hashes", "historical_s10"),
        ],
        [("replay_guard", "replay_guard_hash")],
        [("access_ledger_hash",)],
    ]
    for groups in (commit_stages, hash_stages):
        for destination_group, destination_paths in enumerate(groups):
            for destination in destination_paths:
                for source_group, source_paths in enumerate(groups):
                    if source_group == destination_group:
                        continue
                    mutant = copy.deepcopy(ledger)
                    _set_json_at(mutant, destination, _json_at(mutant, source_paths[0]))
                    rejected(mutant, destination)


def test_t10_tracked_terminal_schema_hash_and_final_lf_use_independent_literals() -> None:
    root = Path(subject.__file__).resolve().parents[1]
    path = root / "results/gross9_structural_clock_bundle_g9cb10_source_support_terminal_failure_2026-07-31.json"
    raw = path.read_bytes()
    ledger = json.loads(raw)
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert list(ledger) == sorted(
        (
            "schema_version", "ledger_kind", "identity", "status", "authority",
            "seal_authority", "implementation", "attempt_sentinel", "execution",
            "failure", "access", "output_state", "terminal_failure_hash",
        )
    )
    assert set(ledger["authority"]) == {"commit", "document_path"}
    assert set(ledger["seal_authority"]) == {"commit", "document_path"}
    assert set(ledger["implementation"]) == {"commit", "parent_commit", "files"}
    assert all(
        set(row) == {"path", "size_bytes", "sha256", "git_blob", "git_mode", "worktree_mode"}
        for row in ledger["implementation"]["files"]
    )
    assert set(ledger["attempt_sentinel"]) == {
        "path", "filesystem_type", "worktree_mode", "git_mode", "link_count",
        "device", "inode", "size_bytes", "sha256", "attempt_hash",
        "repository_head", "repository_parent", "one_shot", "retry_allowed",
        "resume_allowed", "raw_input_count", "opaque_bytes_hashed_before_publication",
    }
    assert set(ledger["execution"]) == {
        "command", "invocation_count", "exit_status", "one_shot", "retry_allowed", "resume_allowed"
    }
    assert set(ledger["failure"]) == {
        "phase", "exception_class", "exception_message",
        "traceback_source_value_excerpt_emitted", "off_grid_location_disclosed",
        "off_grid_count_disclosed", "off_grid_timestamp_disclosed",
        "off_grid_value_disclosed", "off_grid_detail_disclosure_count",
    }
    assert set(ledger["access"]) == {
        "raw_file_count", "raw_file_open_count", "decode_pass_count", "decode_passes",
        "replacement_market_date_scan_count", "replacement_market_tail_decode_count",
        "replacement_market_tail_selected_row_count",
        "binance_metrics_open_interest_decode_count",
        "global_metrics_alignment_check_count", "attempt_sentinel_publication_count",
        "generated_output_publication_count", "generated_output_readback_count",
        "off_grid_detail_disclosure_count", "candidate_value_rows_opened",
        "comparator_value_rows_opened", "feature_value_rows_opened",
        "schedule_value_rows_opened", "signal_value_rows_opened",
        "return_value_rows_opened", "pnl_value_rows_opened", "cagr_evaluation_count",
        "mdd_evaluation_count", "drawdown_evaluation_count",
        "economic_value_rows_opened", "economic_evaluation_count",
    }
    assert set(ledger["output_state"]) == {
        "terminal_evidence_paths", "permanently_absent_output_paths", "forbidden_stages",
        "source_authoritative", "downstream_consumable",
    }
    assert hashlib.sha256(raw).hexdigest() == "01f99da7b06cb9195159bfedec646a22765368140d174567cb9052d24aa58ea4"
    assert subject.object_hash(ledger, "terminal_failure_hash") == ledger["terminal_failure_hash"]


def test_publish_exact_syscall_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = synthetic_config(tmp_path)
    relative = "results/publish-order.bin"
    events: list[str] = []
    unnamed_fds: set[int] = set()
    canonical_fds: set[int] = set()
    original_open_components = subject._open_nofollow_components
    original_open = subject.os.open
    original_fstat = subject.os.fstat
    original_write = subject._write_all
    original_fchmod = subject.os.fchmod
    original_fsync = subject.os.fsync
    original_pread = subject._pread_complete
    original_linkat = subject._linkat

    def open_components(path: Path, flags: int) -> int:
        events.append("open_parent")
        return original_open_components(path, flags)

    def open_fd(path: object, flags: int, *args: object, **kwargs: object) -> int:
        fd = original_open(path, flags, *args, **kwargs)
        if flags & os.O_TMPFILE == os.O_TMPFILE:
            events.append("open_otmpfile")
            unnamed_fds.add(fd)
        elif path == "publish-order.bin":
            events.append("reopen_canonical")
            canonical_fds.add(fd)
        return fd

    def fstat_fd(fd: int) -> os.stat_result:
        if fd in unnamed_fds:
            events.append("fstat_unnamed")
        elif fd in canonical_fds:
            events.append("fstat_canonical")
        return original_fstat(fd)

    def write(fd: int, raw: bytes) -> None:
        events.append("write_complete")
        original_write(fd, raw)

    def fchmod(fd: int, mode: int) -> None:
        events.append("fchmod_0444")
        original_fchmod(fd, mode)

    def fsync(fd: int) -> None:
        events.append("file_fsync" if fd in unnamed_fds else "directory_fsync")
        original_fsync(fd)

    def pread(fd: int, size: int) -> bytes:
        events.append("pread_unnamed" if fd in unnamed_fds else "pread_canonical")
        return original_pread(fd, size)

    def linkat(fd: int, directory_fd: int, leaf: str) -> None:
        events.append("linkat_create_only")
        original_linkat(fd, directory_fd, leaf)

    monkeypatch.setattr(subject, "_open_nofollow_components", open_components)
    monkeypatch.setattr(subject.os, "open", open_fd)
    monkeypatch.setattr(subject.os, "fstat", fstat_fd)
    monkeypatch.setattr(subject, "_write_all", write)
    monkeypatch.setattr(subject.os, "fchmod", fchmod)
    monkeypatch.setattr(subject.os, "fsync", fsync)
    monkeypatch.setattr(subject, "_pread_complete", pread)
    monkeypatch.setattr(subject, "_linkat", linkat)
    subject._publish(config, relative, b"fixed-publication", "probe", lambda _name: None)
    assert events == [
        "open_parent", "open_otmpfile", "fstat_unnamed", "write_complete",
        "fchmod_0444", "file_fsync", "fstat_unnamed", "pread_unnamed",
        "linkat_create_only", "directory_fsync", "reopen_canonical",
        "fstat_canonical", "pread_canonical",
    ]


@pytest.mark.parametrize(
    ("error_number", "exception"),
    [(errno.EEXIST, FileExistsError), (errno.EIO, OSError)],
)
def test_linkat_errno_mapping_is_exact(
    monkeypatch: pytest.MonkeyPatch, error_number: int, exception: type[OSError]
) -> None:
    class FailingLibc:
        @staticmethod
        def linkat(*_args: object) -> int:
            subject.ctypes.set_errno(error_number)
            return -1

    monkeypatch.setattr(subject, "_LIBC", FailingLibc())
    with pytest.raises(exception):
        subject._linkat(17, 23, "leaf")


@pytest.mark.parametrize(
    ("branch", "linked"),
    [
        ("parent_open", False), ("otmp_unavailable", False),
        ("otmp_open", False), ("unnamed_fstat_initial", False),
        ("write", False), ("fchmod", False), ("file_fsync", False),
        ("unnamed_fstat_complete", False), ("unnamed_readback", False),
        ("linkat", False), ("linkat_eexist", False),
        ("directory_fsync", True), ("canonical_reopen", True),
        ("canonical_fstat", True), ("canonical_verification", True),
    ],
)
def test_publish_every_syscall_and_reopen_failure_closes_fds_without_temp_residue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, branch: str, linked: bool
) -> None:
    config = synthetic_config(tmp_path)
    relative = f"results/publish-{branch}.bin"
    path = tmp_path / relative
    raw = b"immutable-publication-vector"
    baseline_fds = _fd_count()
    original_tmp_flag = os.O_TMPFILE
    original_open_components = subject._open_nofollow_components
    original_open = subject.os.open
    original_fstat = subject.os.fstat
    original_fsync = subject.os.fsync
    original_pread = subject._pread_complete
    parent_fds: set[int] = set()
    unnamed_fds: set[int] = set()
    canonical_fds: set[int] = set()
    unnamed_fstat_count = 0
    pread_count = 0

    def open_components(target: Path, flags: int) -> int:
        if branch == "parent_open":
            raise OSError(errno.EIO, "parent directory open failed")
        fd = original_open_components(target, flags)
        parent_fds.add(fd)
        return fd

    def open_fd(target: object, flags: int, *args: object, **kwargs: object) -> int:
        if branch == "otmp_open" and flags & original_tmp_flag == original_tmp_flag:
            raise OSError(errno.EOPNOTSUPP, "O_TMPFILE failed")
        if branch == "canonical_reopen" and target == path.name:
            raise OSError(errno.EIO, "canonical reopen failed")
        fd = original_open(target, flags, *args, **kwargs)
        if flags & original_tmp_flag == original_tmp_flag:
            unnamed_fds.add(fd)
        elif target == path.name:
            canonical_fds.add(fd)
        return fd

    def fstat(fd: int) -> os.stat_result:
        nonlocal unnamed_fstat_count
        if fd in unnamed_fds:
            unnamed_fstat_count += 1
            if branch == f"unnamed_fstat_{'initial' if unnamed_fstat_count == 1 else 'complete'}":
                raise OSError(errno.EIO, "unnamed fstat failed")
        if branch == "canonical_fstat" and fd in canonical_fds:
            raise OSError(errno.EIO, "canonical fstat failed")
        return original_fstat(fd)

    def fail_write(fd: int, data: bytes) -> None:
        raise OSError(errno.EIO, "write failed")

    def fail_fchmod(fd: int, mode: int) -> None:
        raise OSError(errno.EIO, "fchmod failed")

    def fsync(fd: int) -> None:
        if branch == "file_fsync" and fd in unnamed_fds:
            raise OSError(errno.EIO, "file fsync failed")
        if branch == "directory_fsync" and fd in parent_fds:
            raise OSError(errno.EIO, "directory fsync failed")
        original_fsync(fd)

    def linkat(_fd: int, _directory_fd: int, _leaf: str) -> None:
        if branch == "linkat_eexist":
            raise FileExistsError(errno.EEXIST, "create-only collision")
        raise OSError(errno.EIO, "linkat failed")

    def pread(fd: int, size: int) -> bytes:
        nonlocal pread_count
        pread_count += 1
        if branch == "unnamed_readback" and fd in unnamed_fds:
            raise OSError(errno.EIO, "unnamed readback failed")
        value = original_pread(fd, size)
        return b"corrupt" if branch == "canonical_verification" and pread_count == 2 else value

    monkeypatch.setattr(subject, "_open_nofollow_components", open_components)
    monkeypatch.setattr(subject.os, "open", open_fd)
    monkeypatch.setattr(subject.os, "fstat", fstat)
    monkeypatch.setattr(subject.os, "fsync", fsync)
    monkeypatch.setattr(subject, "_pread_complete", pread)
    if branch == "otmp_unavailable":
        monkeypatch.setattr(subject.os, "O_TMPFILE", 0)
    elif branch == "write":
        monkeypatch.setattr(subject, "_write_all", fail_write)
    elif branch == "fchmod":
        monkeypatch.setattr(subject.os, "fchmod", fail_fchmod)
    elif branch in ("linkat", "linkat_eexist"):
        monkeypatch.setattr(subject, "_linkat", linkat)

    with pytest.raises((subject.SourceSupportFailure, OSError, FileExistsError)):
        subject._publish(config, relative, raw, "probe", lambda _name: None)
    assert _fd_count() == baseline_fds
    assert path.exists() is linked
    if linked:
        assert path.read_bytes() == raw
        assert stat.S_IMODE(path.stat().st_mode) == 0o444
        assert path.stat().st_nlink == 1
    assert sorted(item.name for item in path.parent.iterdir() if item.name.startswith("publish-")) == (
        [path.name] if linked else []
    )


def test_csv_gzip_sha_and_frame_hash_fixed_independent_vectors() -> None:
    frame = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-01-02 03:04:05"), pd.Timestamp("2026-01-02 03:09:05")],
            "value": [1.25, -0.0],
            "note": ["alpha", "comma,value"],
        }
    )
    expected_csv = bytes.fromhex(
        "646174652c76616c75652c6e6f74650a323032362d30312d30322030333a30343a30352c312e32352c616c7068610a"
        "323032362d30312d30322030333a30393a30352c2d302c22636f6d6d612c76616c7565220a"
    )
    expected_gzip = bytes.fromhex(
        "1f8b08000000000002ff4b492c49d5294bcc294dd5c9cb2f49e532323032d33530d43530523030b63230b13230d531"
        "d43332d549cc29c8484497b60449eb1ae82825e7e7e62642cc51e20200a9bf07b254000000"
    )
    gzip_bytes, csv_bytes, frame_hash = subject.serialize_csv_gzip(frame)
    assert csv_bytes == expected_csv
    assert frame_hash == "489f40e2b115ea10be8ed986f2b76ffd4be1dbdef2b1fbc40020e49a79bab694"
    assert hashlib.sha256(csv_bytes).hexdigest() == frame_hash
    assert gzip_bytes == expected_gzip
    assert hashlib.sha256(gzip_bytes).hexdigest() == "65652e7a3106320251ef7743acba4ef90d4afaf65077a01c7b95418397b4b265"
    assert gzip.decompress(gzip_bytes) == expected_csv


def test_embedded_inherited_manifest_matches_tracked_config_and_production_path_reads_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(subject.__file__).resolve().parents[1]
    tracked_path = root / "configs/shadow/portfolio_added_alpha_signal_parity_sources_2026-07-16.json"
    authenticated = json.loads(tracked_path.read_text())
    config = subject.official_config(root)
    assert config.inherited_manifest is None
    assert subject._INHERITED_MANIFEST_LITERAL == authenticated

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("production inherited manifest path performed a filesystem read")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    loaded = subject._load_inherited_manifest(config)
    built = subject._build_manifest(config, "1" * 64, "2" * 64)
    assert loaded == authenticated
    assert loaded is not subject._INHERITED_MANIFEST_LITERAL
    assert [row["name"] for row in built["sources"]] == [
        "market_5m", "funding", "premium", "open_interest",
        "rank7_spot_premium_5m", "rex_taker_train", "rex_taker_test",
        "rex_taker_eval", "rex_veto_source",
    ]
    assert built["sources"][0]["sha256"] == "1" * 64
    assert built["sources"][3]["sha256"] == "2" * 64
