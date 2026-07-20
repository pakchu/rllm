from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
import gzip
import json
from pathlib import Path
from typing import Any, Iterable

import pytest

from training import download_mempool_witness_composition_history as source


def _size_row(index: int, *, bucket_index: int | None = None) -> dict[str, int]:
    bucket = 3_400_000 + (index if bucket_index is None else bucket_index)
    return {
        "avgHeight": 800_000 + index,
        "timestamp": bucket * source.BUCKET_SECONDS + 1_800,
        "avgSize": 1_500_000 + index,
    }


def _weight_row(index: int, *, bucket_index: int | None = None) -> dict[str, int]:
    bucket = 3_400_000 + (index if bucket_index is None else bucket_index)
    return {
        "avgHeight": 800_000 + index,
        "timestamp": bucket * source.BUCKET_SECONDS + 1_800,
        "avgWeight": 3_500_000 + index,
    }


def _payload(indices: Iterable[int]) -> dict[str, list[dict[str, int]]]:
    values = list(indices)
    return {
        "sizes": [_size_row(index) for index in values],
        "weights": [_weight_row(index) for index in values],
    }


def _raw(payload: Any) -> bytes:
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def _fetch(raw: bytes, **changes: Any) -> source.FetchResult:
    fields: dict[str, Any] = {
        "raw": raw,
        "status": 200,
        "final_url": source.ENDPOINT,
        "headers": {
            "Date": "Mon, 20 Jul 2026 00:00:00 GMT",
            "ETag": '"fixture"',
            "Last-Modified": "Sun, 19 Jul 2026 23:00:00 GMT",
            "Content-Type": "application/json; charset=utf-8",
            "X-Ignored": "not persisted",
        },
        "retrieved_at_utc": "2026-07-20T00:00:01.000000Z",
    }
    fields.update(changes)
    return source.FetchResult(**fields)


def _cfg(tmp_path: Path, **changes: Any) -> source.Config:
    cfg = source.Config(
        raw_output=str(tmp_path / "composition.raw.json.gz"),
        output_csv=str(tmp_path / "composition.csv.gz"),
        manifest_output=str(tmp_path / "composition-manifest.json"),
        request_pause_sec=0.0,
        minimum_response_rows=5,
    )
    return replace(cfg, **changes)


def test_run_archives_exact_response_and_writes_deterministic_source_only_table(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    raw = _raw(_payload(range(5)))
    manifest = source.run(cfg, fetch=lambda: _fetch(raw))

    with gzip.open(cfg.raw_output, "rb") as handle:
        assert handle.read() == raw
    with gzip.open(cfg.output_csv, "rt", encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    assert lines[0].split(",") == list(source.OUTPUT_COLUMNS)
    assert len(lines) == 4
    assert "avg_size" in lines[0] and "avg_weight" in lines[0]
    assert "witness_share" not in lines[0]
    assert not any(name in lines[0] for name in ("price", "return", "pnl"))

    first = lines[1].split(",")
    expected_start = _size_row(1)["timestamp"] // source.BUCKET_SECONDS
    assert first[0] == source._utc_text(expected_start * source.BUCKET_SECONDS)
    assert first[2] == source._utc_text(
        (expected_start + 1) * source.BUCKET_SECONDS
        + source.SOURCE_AVAILABILITY_LAG_SECONDS
    )
    assert first[-2:] == ["1500001", "3500001"]
    for line in lines[1:]:
        available = datetime.fromisoformat(
            line.split(",")[2].replace("Z", "+00:00")
        )
        available_epoch = int(available.timestamp())
        assert available.tzinfo == timezone.utc
        assert available_epoch % source.EXECUTION_LATENCY_BAR_SECONDS == 0
        earliest_entry = (
            available_epoch + source.EXECUTION_LATENCY_BAR_SECONDS
        )
        assert earliest_entry % source.EXECUTION_LATENCY_BAR_SECONDS == 0
    assert manifest["source_audit"]["response_rows"] == 5
    assert manifest["source_audit"]["retained_rows"] == 3
    assert manifest["source_audit"]["edge_rows_dropped"] == 2
    assert manifest["source_audit"]["paired_size_weight_rows"] == 5
    assert manifest["source_audit"]["rounding_tolerance_rows"] == 0
    assert manifest["source_audit"]["response_headers"] == {
        "date": "Mon, 20 Jul 2026 00:00:00 GMT",
        "etag": '"fixture"',
        "last-modified": "Sun, 19 Jul 2026 23:00:00 GMT",
        "content-type": "application/json; charset=utf-8",
    }
    assert manifest["raw_artifact"]["decompressed_sha256"] == source.sha256_bytes(raw)
    assert manifest["source_builder"] == {
        "path": source.SOURCE_BUILDER,
        "sha256": source.sha256_file(
            source._repository_path(source.SOURCE_BUILDER)
        ),
    }
    core = dict(manifest)
    manifest_hash = core.pop("manifest_hash")
    assert manifest_hash == source.canonical_hash(core)
    assert manifest["causal_availability"][
        "earliest_entry_additional_seconds"
    ] == 300
    assert manifest["outcome_boundary"] == {
        "btc_market_rows_loaded": 0,
        "funding_rows_loaded": 0,
        "premium_or_oi_rows_loaded": 0,
        "return_or_pnl_fields": 0,
        "wctr_features_derived": 0,
        "signal_incidence_rows_derived": 0,
    }
    assert json.loads(Path(cfg.manifest_output).read_text()) == manifest

    second_cfg = _cfg(tmp_path / "second")
    second_manifest = source.run(second_cfg, fetch=lambda: _fetch(raw))
    assert Path(second_cfg.raw_output).read_bytes() == Path(cfg.raw_output).read_bytes()
    assert Path(second_cfg.output_csv).read_bytes() == Path(cfg.output_csv).read_bytes()
    assert second_manifest["source_audit"] == manifest["source_audit"]


def test_gap_is_rejected_without_forward_fill(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    raw = _raw(_payload([0, 1, 2, 4, 5]))
    with pytest.raises(ValueError, match="complete 12-hour grid"):
        source.run(cfg, fetch=lambda: _fetch(raw))
    assert not Path(cfg.raw_output).exists()
    assert not Path(cfg.output_csv).exists()
    assert not Path(cfg.manifest_output).exists()


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (b"[]", "exact sizes and weights keys"),
        (b'{"sizes":[],"weights":[]}', "shorter than minimum_response_rows"),
        (b'{"sizes":[NaN],"weights":[]}', "non-standard JSON"),
        (b'{"sizes":[Inf],"weights":[]}', "Expecting value"),
        (b"\xff", "not valid UTF-8"),
        (
            b'{"sizes":[],"sizes":[],"weights":[]}',
            "duplicate JSON key",
        ),
        (
            b'{"sizes":[{"avgHeight":1,"avgHeight":2}],"weights":[]}',
            "duplicate JSON key",
        ),
    ],
)
def test_payload_envelope_rejects_invalid_json_without_artifacts(
    tmp_path: Path, raw: bytes, message: str
) -> None:
    cfg = _cfg(tmp_path)
    with pytest.raises((ValueError, json.JSONDecodeError), match=message):
        source.run(cfg, fetch=lambda: _fetch(raw))
    assert not Path(cfg.raw_output).exists()
    assert not Path(cfg.output_csv).exists()
    assert not Path(cfg.manifest_output).exists()


def test_payload_rejects_non_lists_and_unequal_pair_counts() -> None:
    with pytest.raises(ValueError, match="must both be JSON lists"):
        source._normalise_rows({"sizes": {}, "weights": []}, minimum_rows=1)
    payload = _payload(range(5))
    payload["weights"].pop()
    with pytest.raises(ValueError, match="equal length"):
        source._normalise_rows(payload, minimum_rows=5)


@pytest.mark.parametrize(
    ("side", "mutator", "message"),
    [
        ("sizes", lambda row: row.pop("avgSize"), "size row.*schema drift"),
        ("weights", lambda row: row.update(extra=1), "weight row.*schema drift"),
        ("sizes", lambda row: row.update(avgHeight=True), "JSON integer"),
        ("weights", lambda row: row.update(avgHeight=0), "must be positive"),
        ("sizes", lambda row: row.update(timestamp=1.5), "JSON integer"),
        ("sizes", lambda row: row.update(avgSize=-1), "JSON integer"),
        (
            "weights",
            lambda row: row.update(avgWeight=Decimal("1.5")),
            "JSON integer",
        ),
    ],
)
def test_row_contract_rejects_schema_type_and_range_drift(
    side: str, mutator: Any, message: str
) -> None:
    payload = _payload(range(5))
    mutator(payload[side][2])
    with pytest.raises(ValueError, match=message):
        source._normalise_rows(payload, minimum_rows=5)


@pytest.mark.parametrize(
    ("side", "field"),
    [("sizes", "avgHeight"), ("weights", "timestamp")],
)
def test_size_weight_pair_clock_mismatch_is_rejected(side: str, field: str) -> None:
    payload = _payload(range(5))
    payload[side][2][field] += 1
    with pytest.raises(ValueError, match="mismatched clock or height"):
        source._normalise_rows(payload, minimum_rows=5)


@pytest.mark.parametrize(
    ("size", "weight", "message"),
    [
        (4_000_001, 4_000_000, "serialized-size bound"),
        (1_500_000, 4_000_001, "block-weight limit"),
        (900_000, 3_600_005, "invalid witness-byte share"),
        (1_500_000, 1_499_995, "invalid witness-byte share"),
    ],
)
def test_consensus_and_witness_share_bounds_fail_closed(
    size: int, weight: int, message: str
) -> None:
    payload = _payload(range(5))
    payload["sizes"][2]["avgSize"] = size
    payload["weights"][2]["avgWeight"] = weight
    with pytest.raises(ValueError, match=message):
        source._normalise_rows(payload, minimum_rows=5)


def test_integer_average_rounding_tolerance_is_audited_without_clipping() -> None:
    payload = _payload(range(5))
    payload["sizes"][2]["avgSize"] = 900_000
    payload["weights"][2]["avgWeight"] = 3_600_001
    rows, audit = source._normalise_rows(payload, minimum_rows=5)
    assert audit["rounding_tolerance_rows"] == 1
    assert rows[1]["avg_weight"] == 3_600_001


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda payload: [
                side[3].update(avgHeight=side[2]["avgHeight"])
                for side in (payload["sizes"], payload["weights"])
            ],
            "avgHeight",
        ),
        (
            lambda payload: [
                side[3].update(timestamp=side[2]["timestamp"])
                for side in (payload["sizes"], payload["weights"])
            ],
            "timestamp",
        ),
        (
            lambda payload: [
                side[3].update(
                    timestamp=side[2]["timestamp"] + source.BUCKET_SECONDS // 2
                )
                for side in (payload["sizes"], payload["weights"])
            ],
            "bucket IDs",
        ),
        (
            lambda payload: (
                payload["sizes"].reverse(), payload["weights"].reverse()
            ),
            "avgHeight",
        ),
    ],
)
def test_monotonic_and_unique_clock_contracts_fail_closed(
    mutator: Any, message: str
) -> None:
    payload = _payload(range(5))
    mutator(payload)
    with pytest.raises(ValueError, match=message):
        source._normalise_rows(payload, minimum_rows=5)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"status": 500}, "status must be 200"),
        ({"final_url": "https://example.invalid/redirect"}, "frozen endpoint"),
        ({"raw": b""}, "must be non-empty"),
        ({"retrieved_at_utc": "2026-07-20"}, "ending in Z"),
        ({"headers": {"Content-Type": "text/html"}}, "application/json"),
        (
            {"headers": {"Content-Type": "application/jsonp"}},
            "application/json",
        ),
        (
            {"headers": {"Content-Type": "text/plain; note=application/json"}},
            "application/json",
        ),
    ],
)
def test_http_transport_contract_fails_before_writing(
    tmp_path: Path, changes: dict[str, Any], message: str
) -> None:
    cfg = _cfg(tmp_path)
    overrides = dict(changes)
    response_raw = overrides.pop(
        "raw", _raw(_payload(range(5)))
    )
    result = _fetch(response_raw, **overrides)
    with pytest.raises(ValueError, match=message):
        source.run(cfg, fetch=lambda: result)
    assert not Path(cfg.raw_output).exists()


def test_invalid_config_aliases_suffixes_and_limits_fail_before_fetch(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    invalid = [
        replace(cfg, timeout_sec=float("inf")),
        replace(cfg, request_pause_sec=-1.0),
        replace(cfg, maximum_retries=True),
        replace(cfg, maximum_response_bytes=0),
        replace(cfg, minimum_response_rows=2),
        replace(cfg, raw_output=str(tmp_path / "raw.json")),
        replace(cfg, output_csv=str(tmp_path / "normalized.json.gz")),
        replace(cfg, manifest_output=str(tmp_path / "manifest.txt")),
        replace(
            cfg,
            raw_output=str(tmp_path / "same.csv.gz"),
            output_csv=str(tmp_path / "same.csv.gz"),
        ),
    ]
    for bad in invalid:
        called = False

        def fetch() -> source.FetchResult:
            nonlocal called
            called = True
            return _fetch(b"[]")

        with pytest.raises(ValueError):
            source.run(bad, fetch=fetch)
        assert called is False


def test_symlink_alias_to_protected_input_fails_before_fetch(tmp_path: Path) -> None:
    protected_alias = tmp_path / "decision.raw.json.gz"
    protected_alias.symlink_to(Path(source.SOURCE_DECISION).resolve())
    cfg = _cfg(tmp_path, raw_output=str(protected_alias))
    with pytest.raises(ValueError, match="must not overwrite source inputs"):
        source.run(cfg, fetch=lambda: pytest.fail("path alias must fail before fetch"))


def test_source_decision_hash_mismatch_fails_before_fetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(source, "SOURCE_DECISION_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="source decision SHA mismatch"):
        source.run(cfg, fetch=lambda: pytest.fail("hash mismatch must not fetch"))


def test_repository_inputs_resolve_independently_of_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path / "artifacts")
    raw = _raw(_payload(range(5)))
    monkeypatch.chdir(tmp_path)
    manifest = source.run(cfg, fetch=lambda: _fetch(raw))
    assert manifest["source_decision"]["sha256"] == source.SOURCE_DECISION_SHA256
    assert Path(cfg.manifest_output).is_file()


def test_frozen_outputs_are_immutable_and_fail_before_fetch(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    raw = _raw(_payload(range(5)))
    source.run(cfg, fetch=lambda: _fetch(raw))
    called = False

    def fetch() -> source.FetchResult:
        nonlocal called
        called = True
        return _fetch(raw)

    with pytest.raises(FileExistsError, match="immutable"):
        source.run(cfg, fetch=fetch)
    assert called is False


def test_write_failure_leaves_no_final_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path)
    raw = _raw(_payload(range(5)))

    def fail_csv(path: Path, rows: list[source.NormalizedRow]) -> None:
        raise OSError("simulated CSV write failure")

    monkeypatch.setattr(source, "_write_csv_gzip", fail_csv)
    with pytest.raises(OSError, match="simulated CSV write failure"):
        source.run(cfg, fetch=lambda: _fetch(raw))
    assert not Path(cfg.raw_output).exists()
    assert not Path(cfg.output_csv).exists()
    assert not Path(cfg.manifest_output).exists()
    assert list(tmp_path.glob(".*.tmp")) == []


def test_post_link_failure_removes_untracked_final_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path)
    raw = _raw(_payload(range(5)))
    original_link = source.os.link

    def link_then_fail(temporary: Path, final: Path) -> None:
        original_link(temporary, final)
        raise OSError("simulated post-link failure")

    monkeypatch.setattr(source.os, "link", link_then_fail)
    with pytest.raises(OSError, match="simulated post-link failure"):
        source.run(cfg, fetch=lambda: _fetch(raw))
    assert not Path(cfg.raw_output).exists()
    assert not Path(cfg.output_csv).exists()
    assert not Path(cfg.manifest_output).exists()
    assert list(tmp_path.glob(".*.tmp")) == []


def test_publish_collision_never_deletes_another_immutable_artifact(
    tmp_path: Path,
) -> None:
    temporary = tmp_path / "ours.tmp"
    final = tmp_path / "frozen.csv.gz"
    temporary.write_bytes(b"ours")
    final.write_bytes(b"another publisher")

    with pytest.raises(FileExistsError):
        source._publish_new(temporary, final)

    assert temporary.read_bytes() == b"ours"
    assert final.read_bytes() == b"another publisher"


@pytest.mark.parametrize("failed_call", [2, 3])
def test_publish_failure_rolls_back_new_final_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_call: int,
) -> None:
    cfg = _cfg(tmp_path)
    raw = _raw(_payload(range(5)))
    original = source._publish_new
    calls = 0

    def fail_publish(temporary: Path, final: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == failed_call:
            raise OSError("simulated publish failure")
        original(temporary, final)

    monkeypatch.setattr(source, "_publish_new", fail_publish)
    with pytest.raises(OSError, match="simulated publish failure"):
        source.run(cfg, fetch=lambda: _fetch(raw))
    assert not Path(cfg.raw_output).exists()
    assert not Path(cfg.output_csv).exists()
    assert not Path(cfg.manifest_output).exists()


def test_oversized_injected_response_fails_before_decode(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, maximum_response_bytes=10)
    with pytest.raises(ValueError, match="maximum_response_bytes"):
        source.run(cfg, fetch=lambda: _fetch(b"[12345678901234567890]"))


def test_retry_delay_is_bounded_and_honours_pause() -> None:
    assert source._retry_delay(attempt=0, pause=2.0, retry_after=None) == 2.0
    assert source._retry_delay(attempt=2, pause=0.1, retry_after="7") == 7.0
    assert source._retry_delay(attempt=8, pause=0.1, retry_after="999") == 60.0
    assert source._retry_delay(attempt=1, pause=0.1, retry_after="bad") == 2.0
