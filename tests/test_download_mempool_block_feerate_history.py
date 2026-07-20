from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import gzip
import json
from pathlib import Path
from typing import Any

import pytest

from training import download_mempool_block_feerate_history as source


def _row(index: int, *, bucket_step: int = 1) -> dict[str, int]:
    bucket = 3_400_000 + index * bucket_step
    base = 10 + index
    return {
        "avgHeight": 800_000 + index,
        "timestamp": bucket * source.BUCKET_SECONDS + 1_800,
        "avgFee_0": base,
        "avgFee_10": base + 1,
        "avgFee_25": base + 2,
        "avgFee_50": base + 3,
        "avgFee_75": base + 4,
        "avgFee_90": base + 5,
        "avgFee_100": base + 6,
    }


def _raw(rows: list[dict[str, Any]]) -> bytes:
    return json.dumps(rows, separators=(",", ":")).encode("utf-8")


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
        raw_output=str(tmp_path / "fee-rates.raw.json.gz"),
        output_csv=str(tmp_path / "fee-rates.csv.gz"),
        manifest_output=str(tmp_path / "fee-rates-manifest.json"),
        request_pause_sec=0.0,
        minimum_response_rows=5,
    )
    return replace(cfg, **changes)


def test_run_archives_exact_response_and_writes_deterministic_source_only_table(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    raw = _raw([_row(index) for index in range(5)])
    manifest = source.run(cfg, fetch=lambda: _fetch(raw))

    with gzip.open(cfg.raw_output, "rb") as handle:
        assert handle.read() == raw
    with gzip.open(cfg.output_csv, "rt", encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    assert lines[0].split(",") == list(source.OUTPUT_COLUMNS)
    assert len(lines) == 4
    assert "fee_p10" in lines[0]
    assert not any(name in lines[0] for name in ("price", "return", "pnl"))

    first = lines[1].split(",")
    expected_start = _row(1)["timestamp"] // source.BUCKET_SECONDS
    assert first[0] == source._utc_text(expected_start * source.BUCKET_SECONDS)
    assert first[2] == source._utc_text(
        (expected_start + 1) * source.BUCKET_SECONDS
        + source.SOURCE_AVAILABILITY_LAG_SECONDS
    )
    assert manifest["source_audit"]["response_rows"] == 5
    assert manifest["source_audit"]["retained_rows"] == 3
    assert manifest["source_audit"]["edge_rows_dropped"] == 2
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
        "bfrt_features_derived": 0,
        "signal_incidence_rows_derived": 0,
    }
    assert json.loads(Path(cfg.manifest_output).read_text()) == manifest

    second_cfg = _cfg(tmp_path / "second")
    second_manifest = source.run(second_cfg, fetch=lambda: _fetch(raw))
    assert Path(second_cfg.raw_output).read_bytes() == Path(cfg.raw_output).read_bytes()
    assert Path(second_cfg.output_csv).read_bytes() == Path(cfg.output_csv).read_bytes()
    assert second_manifest["source_audit"] == manifest["source_audit"]


def test_gap_is_retained_and_audited_without_forward_fill(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    rows = [_row(0), _row(1), _row(2), _row(4), _row(5)]
    manifest = source.run(cfg, fetch=lambda: _fetch(_raw(rows)))
    assert manifest["source_audit"]["missing_12h_buckets"] == 1
    assert manifest["source_audit"]["maximum_bucket_gap_seconds"] == 86_400
    assert manifest["normalized_artifact"]["rows"] == 3


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (b"{}", "must be a JSON list"),
        (b"[]", "shorter than minimum_response_rows"),
        (b"[NaN,1,2,3,4]", "non-standard JSON"),
        (b"[Inf,1,2,3,4]", "Expecting value"),
        (b"\xff", "not valid UTF-8"),
        (
            b'[{"avgHeight":1,"avgHeight":2}, {}, {}, {}, {}]',
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


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda row: row.pop("avgFee_50"), "schema drift"),
        (lambda row: row.update(extra=1), "schema drift"),
        (lambda row: row.update(avgHeight=True), "non-negative JSON integer"),
        (lambda row: row.update(avgHeight=0), "must be positive"),
        (lambda row: row.update(timestamp=1.5), "non-negative JSON integer"),
        (lambda row: row.update(avgFee_10=-1), "non-negative JSON integer"),
        (
            lambda row: row.update(avgFee_50=Decimal("1.5")),
            "non-negative JSON integer",
        ),
        (lambda row: row.update(avgFee_25=100), "percentile ordering"),
    ],
)
def test_row_contract_rejects_schema_type_range_and_ordering_drift(
    mutator: Any, message: str
) -> None:
    rows = [_row(index) for index in range(5)]
    mutator(rows[2])
    with pytest.raises(ValueError, match=message):
        source._normalise_rows(rows, minimum_rows=5)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda rows: rows[3].update(avgHeight=rows[2]["avgHeight"]), "avgHeight"),
        (lambda rows: rows[3].update(timestamp=rows[2]["timestamp"]), "timestamp"),
        (
            lambda rows: rows[3].update(
                timestamp=rows[2]["timestamp"] + source.BUCKET_SECONDS // 2
            ),
            "bucket IDs",
        ),
        (lambda rows: rows.reverse(), "avgHeight"),
    ],
)
def test_monotonic_and_unique_clock_contracts_fail_closed(
    mutator: Any, message: str
) -> None:
    rows = [_row(index) for index in range(5)]
    mutator(rows)
    with pytest.raises(ValueError, match=message):
        source._normalise_rows(rows, minimum_rows=5)


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
        "raw", _raw([_row(index) for index in range(5)])
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
    raw = _raw([_row(index) for index in range(5)])
    monkeypatch.chdir(tmp_path)
    manifest = source.run(cfg, fetch=lambda: _fetch(raw))
    assert manifest["source_decision"]["sha256"] == source.SOURCE_DECISION_SHA256
    assert Path(cfg.manifest_output).is_file()


def test_frozen_outputs_are_immutable_and_fail_before_fetch(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    raw = _raw([_row(index) for index in range(5)])
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
    raw = _raw([_row(index) for index in range(5)])

    def fail_csv(path: Path, rows: list[source.NormalizedRow]) -> None:
        raise OSError("simulated CSV write failure")

    monkeypatch.setattr(source, "_write_csv_gzip", fail_csv)
    with pytest.raises(OSError, match="simulated CSV write failure"):
        source.run(cfg, fetch=lambda: _fetch(raw))
    assert not Path(cfg.raw_output).exists()
    assert not Path(cfg.output_csv).exists()
    assert not Path(cfg.manifest_output).exists()
    assert list(tmp_path.glob(".*.tmp")) == []


@pytest.mark.parametrize("failed_call", [2, 3])
def test_publish_failure_rolls_back_new_final_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_call: int,
) -> None:
    cfg = _cfg(tmp_path)
    raw = _raw([_row(index) for index in range(5)])
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
