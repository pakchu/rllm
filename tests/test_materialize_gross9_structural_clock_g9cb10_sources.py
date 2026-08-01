from __future__ import annotations

import ast
import csv
import gzip
import hashlib
import io
import json
import os
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pytest

from training import materialize_gross9_structural_clock_g9cb10_sources as subject


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
        {"identity": "G9CB-10", "version": subject.VERSION}, "support_hash"
    ) == "5b326b1a18c389f41224987d80ed7ebf88121715aebbd769fc30ace5e2b5fd63"
    assert subject.canonical_json_bytes({"é": "한"}) == b'{"\xc3\xa9":"\xed\x95\x9c"}\n'
    vector = pd.DataFrame({"date": [pd.Timestamp("2026-01-01")], "value": [1]})
    gzip_bytes, csv_bytes, frame_hash = subject.serialize_csv_gzip(vector)
    assert csv_bytes == b"date,value\n2026-01-01 00:00:00,1\n"
    assert frame_hash == "8e362a6177525d72ecae23994c231bc666be1c61995aaddbf1afcaedc805b433"
    assert gzip_bytes.hex() == (
        "1f8b08000000000002ff4b492c49d5294bcc294de532323032d3353004220503"
        "032b30d231e40200aefd761a21000000"
    )
    assert hashlib.sha256(gzip_bytes).hexdigest() == (
        "57d246a01b7693e244e228076057eeba6ac845b2e6aa79c5d9c70396f81d769c"
    )


def test_synthetic_success_exact_artifacts_and_counters(tmp_path: Path) -> None:
    config = synthetic_config(tmp_path)
    events: list[str] = []
    support = subject.materialize(config, failpoint=events.append)
    access = support["access_ledger"]
    assert access["current_s10"]["raw_file_count"] == 7
    assert access["current_s10"]["decode_pass_count"] == 8
    assert access["current_s10"]["generated_readback_decode_count"] == 2
    assert access["current_s10"]["replacement_tail_rows_selected"] == 3
    assert access["current_s10"]["overlap_value_comparison_count"] == 0
    assert access["process_local"] == {
        "stage": "S10",
        "slot": 0,
        "invocation_count": 1,
        "source_files_opened": 7,
        "source_value_rows_opened": 99,
        "candidate_rows_opened": 0,
        "comparator_clock_rows_opened": 0,
        "model_files_opened": 0,
        "runtime_modules_imported": 0,
        "pre2025_anchor_value_rows_opened": 0,
        "feature_signal_schedule_or_interval_values_computed": 0,
        "portfolio_economic_values_computed": 0,
        "economic_or_overlap_values_computed": 0,
    }
    assert access["access_ledger_hash"] == subject.object_hash(
        access, "access_ledger_hash"
    )
    assert access["replay_guard"]["replay_guard_hash"] == subject.object_hash(
        access["replay_guard"], "replay_guard_hash"
    )
    assert support["validation"]["appended_market_rows"] == 3
    assert support["validation"]["oi_splice_window_exact_rows"] == 3
    assert support["validation"]["rank7_spot_premium_raw_column_count"] == 6
    assert support["validation"]["market_partition"] == {
        "boundary_inclusive_old": "2026-01-01T01:35:00Z",
        "disjoint_union": True,
        "domain_end_exclusive": "2026-01-01T01:55:00Z",
        "old_prefix_rows": 20,
        "output_rows": 23,
        "overlap_value_comparison_count": 0,
        "replacement_prefix_non_date_values_semantically_evaluated": 0,
        "replacement_prefix_rows_selected": 0,
        "replacement_tail_rows": 3,
        "timestamp_intersection_rows": 0,
    }
    assert set(support) == {
        "access_ledger", "attempt_sentinel", "identity", "materialized_sources",
        "raw_sources", "source_manifest", "source_support_commit",
        "support_hash", "validation", "version",
    }
    assert set(access) == {
        "schema_version", "ledger_kind", "historical_s9", "current_s10",
        "process_local", "replay_guard", "access_ledger_hash",
    }
    assert set(access["historical_s9"]) == {
        "identity", "attempt_sentinel", "terminal_failure_ledger",
        "official_invocation_count", "exit_status", "raw_file_count",
        "decode_pass_count", "market_prefix_comparison_count",
        "mismatch_fraction_percent", "traceback_value_excerpt_emitted",
        "traceback_value_excerpt_restated_in_ledger",
        "generated_readback_decode_count", "candidate_rows_opened",
        "comparator_clock_rows_opened",
        "feature_signal_schedule_or_interval_values_computed",
        "economic_or_overlap_values_computed",
    }
    assert set(access["current_s10"]) == {
        "identity", "attempt_sentinel", "official_invocation_count",
        "raw_file_count", "decode_pass_count", "decode_passes",
        "replacement_date_scan_count", "replacement_tail_decode_count",
        "generated_readback_decode_count", "replacement_prefix_rows_selected",
        "replacement_prefix_non_date_values_semantically_evaluated",
        "replacement_tail_rows_selected", "overlap_value_comparison_count",
        "candidate_rows_opened", "comparator_clock_rows_opened",
        "feature_signal_schedule_or_interval_values_computed",
        "economic_or_overlap_values_computed",
    }
    assert set(access["replay_guard"]) == {
        "prior_identity", "current_identity", "prior_attempt_hash",
        "current_attempt_hash", "prior_terminal_failure_hash",
        "authority_commit", "authority_path", "terminal_evidence_commit",
        "terminal_failure_ledger_path", "implementation_commit",
        "implementation_paths", "prior_expected_output_paths",
        "current_expected_output_paths", "replay_guard_hash",
    }
    assert set(support["validation"]) == {
        "appended_market_rows", "appended_open_interest_rows",
        "domain_end_exclusive", "funding_attachment_tolerance",
        "funding_tail_available_rows", "latest_funding_available_after_causal_attachment",
        "latest_open_interest_positive_after_exact_join",
        "latest_premium_available_after_causal_attachment", "market_grid_seconds",
        "market_partition", "metrics_common_timestamp_values_exact",
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
    raw_by_name = {row["name"]: row for row in support["raw_sources"]}
    assert raw_by_name["replacement_market"]["decoded_rows"] == 23
    assert raw_by_name["replacement_market"]["normalized_rows"] == 3
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
    materialized_market = pd.read_csv(tmp_path / config.market_output_path)
    old_market = pd.read_csv(config.inputs[0].path)
    replacement = pd.read_csv(config.inputs[1].path)
    pd.testing.assert_frame_equal(
        materialized_market.iloc[:20].reset_index(drop=True),
        old_market.reset_index(drop=True),
        check_exact=True,
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        materialized_market.iloc[20:].reset_index(drop=True),
        replacement.iloc[20:23].reset_index(drop=True),
        check_exact=True,
        check_dtype=False,
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


def _access_validation_inputs(
    config: subject.MaterializationConfig,
) -> tuple[
    dict[str, object],
    dict[str, object],
    subject.ReplacementDateScan,
    dict[str, int],
]:
    attempt = subject._attempt(config, config.repository_head)
    attempt_raw = subject.canonical_json_bytes(attempt)
    attempt_binding: dict[str, object] = {
        "sha256": hashlib.sha256(attempt_raw).hexdigest(),
        "size_bytes": len(attempt_raw),
    }
    tail_dates = pd.Series(
        pd.date_range(
            config.old_last + pd.Timedelta(minutes=5),
            periods=config.expected_append_rows,
            freq="5min",
        )
    )
    replacement_scan = subject.ReplacementDateScan(
        subject.MARKET_SCHEMA,
        config.expected_complete_rows,
        tuple(
            range(
                config.expected_old_rows,
                config.expected_old_rows + config.expected_append_rows,
            )
        ),
        tail_dates,
    )
    decoded_rows = {
        "old_market": 20,
        "replacement_market": 23,
        "funding": 2,
        "premium": 2,
        "old_open_interest": 20,
        "binance_metrics_open_interest": 6,
        "rank7_spot_premium_5m": 23,
    }
    return attempt, attempt_binding, replacement_scan, decoded_rows


def _json_mapping_paths(value: object, prefix: tuple[object, ...] = ()) -> list[tuple[object, ...]]:
    paths: list[tuple[object, ...]] = []
    if isinstance(value, dict):
        paths.append(prefix)
        for key, nested in value.items():
            paths.extend(_json_mapping_paths(nested, (*prefix, key)))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            paths.extend(_json_mapping_paths(nested, (*prefix, index)))
    return paths


def _json_list_paths(value: object, prefix: tuple[object, ...] = ()) -> list[tuple[object, ...]]:
    paths: list[tuple[object, ...]] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            paths.extend(_json_list_paths(nested, (*prefix, key)))
    elif isinstance(value, list):
        paths.append(prefix)
        for index, nested in enumerate(value):
            paths.extend(_json_list_paths(nested, (*prefix, index)))
    return paths


def _json_at_path(value: object, path: tuple[object, ...]) -> object:
    current = value
    for component in path:
        assert isinstance(current, (dict, list))
        current = current[component]  # type: ignore[index]
    return current


def _rehash_access_mutant(ledger: dict[str, object]) -> None:
    replay = ledger.get("replay_guard")
    if isinstance(replay, dict) and isinstance(replay.get("replay_guard_hash"), str):
        replay["replay_guard_hash"] = subject.object_hash(replay, "replay_guard_hash")
    if isinstance(ledger.get("access_ledger_hash"), str):
        ledger["access_ledger_hash"] = subject.object_hash(
            ledger, "access_ledger_hash"
        )


def test_access_ledger_rejects_every_recursive_schema_type_and_order_mutation(
    tmp_path: Path,
) -> None:
    config = synthetic_config(tmp_path)
    attempt, attempt_binding, replacement_scan, decoded_rows = _access_validation_inputs(
        config
    )
    expected = subject._expected_access_ledger(
        config,
        config.repository_head,
        attempt,
        attempt_binding,
        replacement_scan,
        decoded_rows,
    )

    def rejected(mutant: dict[str, object], label: str) -> None:
        _rehash_access_mutant(mutant)
        try:
            subject._validate_access_ledger(
                mutant,
                config,
                head=config.repository_head,
                attempt=attempt,
                attempt_binding=attempt_binding,
                replacement_scan=replacement_scan,
                decoded_rows=decoded_rows,
            )
        except subject.SourceSupportFailure:
            return
        pytest.fail(f"accepted access-ledger mutation: {label}")

    for mapping_path in _json_mapping_paths(expected):
        mapping = _json_at_path(expected, mapping_path)
        assert isinstance(mapping, dict)
        additional = deepcopy(expected)
        additional_mapping = _json_at_path(additional, mapping_path)
        assert isinstance(additional_mapping, dict)
        additional_mapping["__unexpected_member__"] = None
        rejected(additional, f"additional member at {mapping_path}")
        for key in mapping:
            missing = deepcopy(expected)
            missing_mapping = _json_at_path(missing, mapping_path)
            assert isinstance(missing_mapping, dict)
            missing_mapping.pop(key)
            rejected(missing, f"missing {(*mapping_path, key)}")

            renamed = deepcopy(expected)
            renamed_mapping = _json_at_path(renamed, mapping_path)
            assert isinstance(renamed_mapping, dict)
            renamed_mapping[f"{key}__renamed"] = renamed_mapping.pop(key)
            rejected(renamed, f"renamed {(*mapping_path, key)}")

            retyped = deepcopy(expected)
            retyped_mapping = _json_at_path(retyped, mapping_path)
            assert isinstance(retyped_mapping, dict)
            retyped_mapping[key] = None
            rejected(retyped, f"retyped {(*mapping_path, key)}")

    for list_path in _json_list_paths(expected):
        original = _json_at_path(expected, list_path)
        assert isinstance(original, list)
        retyped = deepcopy(expected)
        parent = _json_at_path(retyped, list_path[:-1])
        assert isinstance(parent, (dict, list))
        parent[list_path[-1]] = None  # type: ignore[index]
        rejected(retyped, f"retyped list {list_path}")
        if len(original) > 1:
            reordered = deepcopy(expected)
            reordered_list = _json_at_path(reordered, list_path)
            assert isinstance(reordered_list, list)
            reordered_list[0], reordered_list[1] = reordered_list[1], reordered_list[0]
            rejected(reordered, f"reordered list {list_path}")


def _delete_nested(ledger: dict[str, object], block: str, key: str) -> None:
    nested = ledger[block]
    assert isinstance(nested, dict)
    nested.pop(key)


def _add_nested(ledger: dict[str, object], block: str, key: str) -> None:
    nested = ledger[block]
    assert isinstance(nested, dict)
    nested[key] = 0


def _set_nested(ledger: dict[str, object], block: str, key: str, value: object) -> None:
    nested = ledger[block]
    assert isinstance(nested, dict)
    nested[key] = value


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda ledger: ledger.__setitem__("unexpected", 0), "access ledger exact"),
        (lambda ledger: _delete_nested(ledger, "historical_s9", "exit_status"), "access ledger exact"),
        (lambda ledger: _add_nested(ledger, "current_s10", "unexpected"), "access ledger exact"),
        (lambda ledger: _delete_nested(ledger, "process_local", "slot"), "access ledger exact"),
        (lambda ledger: _delete_nested(ledger, "replay_guard", "authority_commit"), "access ledger exact"),
        (lambda ledger: _set_nested(ledger, "process_local", "slot", "0"), "access ledger exact"),
        (lambda ledger: _set_nested(ledger, "process_local", "candidate_rows_opened", 1), "access ledger exact"),
        (lambda ledger: _set_nested(ledger, "current_s10", "identity", "G9CB-9-SOURCE-SUPPORT"), "access ledger exact"),
    ],
)
def test_access_ledger_exact_keys_types_and_separation_are_enforced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate: Callable[[dict[str, object]], None],
    match: str,
) -> None:
    config = synthetic_config(tmp_path)
    original = subject._build_access_ledger

    def altered(*args: object, **kwargs: object) -> dict[str, object]:
        ledger = original(*args, **kwargs)
        mutate(ledger)
        replay = ledger.get("replay_guard")
        if isinstance(replay, dict) and "replay_guard_hash" in replay:
            replay["replay_guard_hash"] = subject.object_hash(replay, "replay_guard_hash")
        ledger["access_ledger_hash"] = subject.object_hash(ledger, "access_ledger_hash")
        return ledger

    monkeypatch.setattr(subject, "_build_access_ledger", altered)
    with pytest.raises(subject.SourceSupportFailure, match=match):
        subject.materialize(config)
    assert not (tmp_path / config.support_output_path).exists()


def test_replay_guard_rejects_attempt_and_output_set_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = synthetic_config(tmp_path)
    original = subject._build_access_ledger

    def reused(*args: object, **kwargs: object) -> dict[str, object]:
        ledger = original(*args, **kwargs)
        historical = ledger["historical_s9"]
        current = ledger["current_s10"]
        replay = ledger["replay_guard"]
        assert isinstance(historical, dict) and isinstance(current, dict) and isinstance(replay, dict)
        prior_attempt = historical["attempt_sentinel"]
        current_attempt = current["attempt_sentinel"]
        assert isinstance(prior_attempt, dict) and isinstance(current_attempt, dict)
        current_attempt["attempt_hash"] = prior_attempt["attempt_hash"]
        replay["current_attempt_hash"] = prior_attempt["attempt_hash"]
        replay["current_expected_output_paths"] = list(replay["prior_expected_output_paths"])
        replay["replay_guard_hash"] = subject.object_hash(replay, "replay_guard_hash")
        ledger["access_ledger_hash"] = subject.object_hash(ledger, "access_ledger_hash")
        return ledger

    monkeypatch.setattr(subject, "_build_access_ledger", reused)
    with pytest.raises(subject.SourceSupportFailure, match="access ledger exact"):
        subject.materialize(config)


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
    assert support["access_ledger"]["current_s10"]["overlap_value_comparison_count"] == 0
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
    current = support["access_ledger"]["current_s10"]
    assert current["replacement_prefix_non_date_values_semantically_evaluated"] == 0
    assert current["replacement_prefix_rows_selected"] == 0


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
    assert support["access_ledger"]["current_s10"]["decode_passes"] == [
        "old_market",
        "replacement_market_date_scan",
        "replacement_market_tail",
        "funding",
        "premium",
        "old_open_interest",
        "binance_metrics_open_interest",
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


def test_s10_has_no_candidate_or_economics_execution_or_open_surface(
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
        "candidate_rows_opened",
        "comparator_clock_rows_opened",
        "model_files_opened",
        "runtime_modules_imported",
        "pre2025_anchor_value_rows_opened",
        "feature_signal_schedule_or_interval_values_computed",
        "portfolio_economic_values_computed",
        "economic_or_overlap_values_computed",
    ):
        assert process[key] == 0


def test_official_configuration_is_frozen_without_opening_sources() -> None:
    config = subject.official_config()
    assert config.branch == subject.BRANCH
    assert config.repository_parent == subject.T9
    assert config.authority_commit == subject.A10
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
        "results/gross9_structural_clock_bundle_g9cb10_source_support_attempt_consumed_2026-07-31.json",
        "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_g9cb10_complete.csv.gz",
        "data/btcusdt_open_interest_5m_2020-01-01_2026-06-01_g9cb10_complete.csv.gz",
        "configs/shadow/gross9_structural_clock_bundle_g9cb10_sources_2026-07-31.json",
        "results/gross9_structural_clock_bundle_g9cb10_source_support_2026-07-31.json",
    )
    assert (config.expected_old_rows, config.expected_complete_rows, config.expected_append_rows) == (674_785, 674_892, 107)
    assert config.splice_rows == 13
    assert config.rank7_tail_rows == 3000


def test_current_repository_t9_gate_authenticates_without_source_access() -> None:
    subject._validate_t9_gate(subject.official_config())


def test_official_command_shape_and_bytecode_preflight_without_source_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = subject.official_config()
    monkeypatch.chdir(config.root)
    monkeypatch.setattr(subject.sys, "argv", ["materialize_g9cb10"])
    monkeypatch.setattr(subject.sys, "dont_write_bytecode", True)
    monkeypatch.setenv("PYTHONPATH", str(config.root))
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")

    subject._validate_command_and_root(config)
    subject._validate_bytecode_first_gate(config.root)

    monkeypatch.setattr(subject.sys, "argv", ["materialize_g9cb10", "unexpected"])
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
        return "synthetic-gated-s10"

    monkeypatch.setattr(subject, "_validate_git_gate", git_gate)
    monkeypatch.setattr(subject, "_validate_t9_gate", lambda _config: events.append("t9"))

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

    assert events[:6] == ["command", "bytecode", "git", "t9", "output_absence", "open_inputs"]
    assert events[-2:] == ["bytecode", "checkpoint:9"]
    assert support["source_support_commit"] == "synthetic-gated-s10"


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


@pytest.mark.parametrize(
    "failing_gate", ("command", "bytecode", "git", "t9", "output_absence")
)
def test_each_official_preflight_gate_failure_short_circuits_all_later_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failing_gate: str,
) -> None:
    config = replace(synthetic_config(tmp_path), enforce_repository_gates=True)
    events: list[str] = []

    def visit(name: str, result: object = None) -> object:
        events.append(name)
        if name == failing_gate:
            raise subject.SourceSupportFailure(f"synthetic {name} gate failure")
        return result

    monkeypatch.setattr(
        subject, "_validate_command_and_root", lambda _config: visit("command")
    )
    monkeypatch.setattr(
        subject, "_validate_bytecode_first_gate", lambda _root: visit("bytecode")
    )
    monkeypatch.setattr(
        subject,
        "_validate_git_gate",
        lambda _config: visit("git", config.repository_head),
    )
    monkeypatch.setattr(subject, "_validate_t9_gate", lambda _config: visit("t9"))
    monkeypatch.setattr(
        subject, "_validate_output_absence", lambda _config: visit("output_absence")
    )

    def forbidden_open_inputs(
        _config: subject.MaterializationConfig, _failpoint: Callable[[str], None]
    ) -> dict[str, subject._Retained]:
        events.append("open_inputs")
        raise AssertionError("source opening followed a failed preflight gate")

    monkeypatch.setattr(subject, "_open_inputs", forbidden_open_inputs)
    order = ["command", "bytecode", "git", "t9", "output_absence"]
    with pytest.raises(
        subject.SourceSupportFailure,
        match=f"synthetic {failing_gate} gate failure",
    ):
        subject.materialize(config)

    assert events == order[: order.index(failing_gate) + 1]
    assert not any((tmp_path / relative).exists() for relative in config.output_paths)


def test_git_gate_authenticates_topology_cleanliness_and_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = synthetic_config(tmp_path)
    s10 = "a" * 40
    mode = {"value": "100644"}

    def fake_git(_root: Path, *args: str) -> str:
        if args == ("rev-parse", "HEAD"):
            return s10
        if args == ("branch", "--show-current"):
            return subject.BRANCH
        if args == ("rev-parse", "@{upstream}"):
            return s10
        if args == ("rev-list", "--parents", "-n", "1", s10):
            return f"{s10} {subject.T9}"
        if args == ("rev-parse", f"{subject.T9}^"):
            return subject.A10
        if args == ("rev-parse", f"{subject.A10}^"):
            return subject.S9
        if args == ("rev-parse", f"{subject.S9}^"):
            return subject.T8
        if args == ("rev-parse", f"{subject.T8}^"):
            return subject.A9
        if args == ("diff", "--name-status", subject.S9, subject.A10):
            return "A\tdocs/gross9-structural-clock-bundle-g9cb10-successor-authority-decision-2026-07-31.md"
        if args == ("diff", "--name-status", subject.A10, subject.T9):
            return "A\tresults/gross9_structural_clock_bundle_g9cb9_source_support_attempt_consumed_2026-07-31.json\nA\tresults/gross9_structural_clock_bundle_g9cb9_source_support_terminal_failure_2026-07-31.json"
        if args == ("diff", "--name-status", subject.T9, s10):
            return "A\ttests/test_materialize_gross9_structural_clock_g9cb10_sources.py\nA\ttraining/materialize_gross9_structural_clock_g9cb10_sources.py"
        if args[:3] == ("ls-files", "-s", "--"):
            relative = args[3]
            return f"{mode['value']} {'a' * 40} 0 {relative}"
        if args == ("status", "--porcelain=v1", "--untracked-files=all"):
            return ""
        raise AssertionError(f"unexpected git command: {args}")

    monkeypatch.setattr(subject, "_run_git", fake_git)
    assert subject._validate_git_gate(config) == s10

    mode["value"] = "100755"
    with pytest.raises(subject.SourceSupportFailure, match="Git mode or index binding"):
        subject._validate_git_gate(config)


def _install_t9_evidence(tmp_path: Path) -> subject.MaterializationConfig:
    config = synthetic_config(tmp_path)
    repository = Path(subject.__file__).resolve().parents[1]
    authority = tmp_path / subject._A10_AUTHORITY[0]
    authority.parent.mkdir(parents=True, exist_ok=True)
    authority.write_bytes((repository / subject._A10_AUTHORITY[0]).read_bytes())
    authority.chmod(0o644)
    for relative, _size, _digest, _blob in subject._T9_EVIDENCE:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((repository / relative).read_bytes())
        target.chmod(0o444)
    return config


def test_t9_gate_authenticates_authority_terminal_evidence_and_absences(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _install_t9_evidence(tmp_path)

    def fake_git(_root: Path, *args: str) -> str:
        relative = args[-1]
        if relative == subject._A10_AUTHORITY[0]:
            blob = subject._A10_AUTHORITY[3]
        else:
            blob = next(row[3] for row in subject._T9_EVIDENCE if row[0] == relative)
        return f"100644 {blob} 0 {relative}"

    monkeypatch.setattr(subject, "_run_git", fake_git)
    subject._validate_t9_gate(config)

    terminal_path = tmp_path / subject._T9_EVIDENCE[1][0]
    terminal_path.chmod(0o644)
    with pytest.raises(subject.SourceSupportFailure, match="bound evidence differs"):
        subject._validate_t9_gate(config)


def test_t9_gate_rejects_every_recursive_schema_type_and_order_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _install_t9_evidence(tmp_path)
    repository = Path(subject.__file__).resolve().parents[1]
    expected = {
        relative: json.loads((repository / relative).read_bytes())
        for relative, _size, _digest, _blob in subject._T9_EVIDENCE
    }
    active = {path: deepcopy(payload) for path, payload in expected.items()}

    def fake_read_bound_json(
        _config: subject.MaterializationConfig,
        relative: str,
        _size: int,
        _digest: str,
        _blob: str,
        *,
        mode: int,
    ) -> dict[str, object]:
        assert mode == 0o444
        return deepcopy(active[relative])

    def fake_git(_root: Path, *args: str) -> str:
        assert args == ("ls-files", "-s", "--", subject._A10_AUTHORITY[0])
        return (
            f"100644 {subject._A10_AUTHORITY[3]} 0 "
            f"{subject._A10_AUTHORITY[0]}"
        )

    monkeypatch.setattr(subject, "_read_bound_json", fake_read_bound_json)
    monkeypatch.setattr(subject, "_run_git", fake_git)

    def rejected(path: str, mutant: dict[str, object], label: str) -> None:
        active.clear()
        active.update({key: deepcopy(value) for key, value in expected.items()})
        active[path] = mutant
        try:
            subject._validate_t9_gate(config)
        except subject.SourceSupportFailure:
            return
        pytest.fail(f"accepted T9 mutation: {label}")

    for evidence_path, payload in expected.items():
        for mapping_path in _json_mapping_paths(payload):
            mapping = _json_at_path(payload, mapping_path)
            assert isinstance(mapping, dict)
            additional = deepcopy(payload)
            additional_mapping = _json_at_path(additional, mapping_path)
            assert isinstance(additional_mapping, dict)
            additional_mapping["__unexpected_member__"] = None
            rejected(evidence_path, additional, f"additional member at {mapping_path}")
            for key in mapping:
                missing = deepcopy(payload)
                missing_mapping = _json_at_path(missing, mapping_path)
                assert isinstance(missing_mapping, dict)
                missing_mapping.pop(key)
                rejected(evidence_path, missing, f"missing {(*mapping_path, key)}")

                renamed = deepcopy(payload)
                renamed_mapping = _json_at_path(renamed, mapping_path)
                assert isinstance(renamed_mapping, dict)
                renamed_mapping[f"{key}__renamed"] = renamed_mapping.pop(key)
                rejected(evidence_path, renamed, f"renamed {(*mapping_path, key)}")

                retyped = deepcopy(payload)
                retyped_mapping = _json_at_path(retyped, mapping_path)
                assert isinstance(retyped_mapping, dict)
                retyped_mapping[key] = None
                rejected(evidence_path, retyped, f"retyped {(*mapping_path, key)}")

        for list_path in _json_list_paths(payload):
            original = _json_at_path(payload, list_path)
            assert isinstance(original, list)
            retyped = deepcopy(payload)
            parent = _json_at_path(retyped, list_path[:-1])
            assert isinstance(parent, (dict, list))
            parent[list_path[-1]] = None  # type: ignore[index]
            rejected(evidence_path, retyped, f"retyped list {list_path}")
            if len(original) > 1:
                reordered = deepcopy(payload)
                reordered_list = _json_at_path(reordered, list_path)
                assert isinstance(reordered_list, list)
                reordered_list[0], reordered_list[1] = (
                    reordered_list[1],
                    reordered_list[0],
                )
                rejected(evidence_path, reordered, f"reordered list {list_path}")


@pytest.mark.parametrize("relative", subject._G9CB9_PERMANENT_ABSENCES)
def test_t9_gate_rejects_each_permanent_g9cb9_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative: str
) -> None:
    config = _install_t9_evidence(tmp_path)

    def fake_git(_root: Path, *args: str) -> str:
        path = args[-1]
        if path == subject._A10_AUTHORITY[0]:
            blob = subject._A10_AUTHORITY[3]
        else:
            blob = next(row[3] for row in subject._T9_EVIDENCE if row[0] == path)
        return f"100644 {blob} 0 {path}"

    monkeypatch.setattr(subject, "_run_git", fake_git)
    forbidden = tmp_path / relative
    forbidden.parent.mkdir(parents=True, exist_ok=True)
    forbidden.write_bytes(b"forbidden")

    with pytest.raises(subject.SourceSupportFailure, match="permanent G9CB-9 absence"):
        subject._validate_t9_gate(config)
