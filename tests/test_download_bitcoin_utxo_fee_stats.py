from __future__ import annotations

import csv
from dataclasses import replace
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any
import urllib.parse

import pytest

from training import download_bitcoin_utxo_fee_stats as source


TEST_START = 823_700
TEST_END = 823_726


def _hash(height: int, *, salt: int = 0) -> str:
    return f"{height + salt:064x}"


def _row(
    height: int,
    *,
    salt: int = 0,
    timestamp: int | None = None,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    total_inputs = 200 + (height % 17)
    total_outputs = 240 + (height % 19)
    base_extras = {
        "totalFees": 10_000 + height,
        "totalInputs": total_inputs,
        "totalOutputs": total_outputs,
        "utxoSetChange": total_outputs - total_inputs,
        "feeRange": [1, 2, 3],
        "pool": {"name": "ignored"},
    }
    if extras:
        base_extras.update(extras)
    return {
        "id": _hash(height, salt=salt),
        "height": height,
        "timestamp": timestamp if timestamp is not None else 1_700_000_000 + height,
        "tx_count": 100 + height,
        "size": 1_000,
        "weight": 3_000,
        "previousblockhash": _hash(height - 1, salt=salt),
        "mediantime": 1_699_999_000 + height,
        "stale": 0,
        "extras": base_extras,
        "version": 1,
        "difficulty": "ignored unrelated metadata",
    }


def _page(cursor: int, *, salt: int = 0) -> list[dict[str, Any]]:
    return [_row(height, salt=salt) for height in range(cursor, cursor - 15, -1)]


def _cursor(url: str) -> int:
    return int(urllib.parse.urlparse(url).path.rsplit("/", 1)[-1])


def _write_reference(path: Path, *, start: int = TEST_START, end: int = TEST_END, salt: int = 0) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            import io

            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                writer = csv.DictWriter(text, fieldnames=source.BASIC_COLUMNS, lineterminator="\n")
                writer.writeheader()
                for height in range(start, end + 1):
                    row = _row(height, salt=salt)
                    writer.writerow({column: row[column] for column in source.BASIC_COLUMNS})
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cfg(tmp_path: Path, **changes: object) -> source.Config:
    reference = tmp_path / "reference.csv.gz"
    reference_sha = _write_reference(reference)
    cfg = source.Config(
        output_csv=str(tmp_path / "utxo.csv.gz"),
        manifest_output=str(tmp_path / "manifest.json"),
        checkpoint_db=str(tmp_path / "checkpoint.sqlite3"),
        reference_block_summaries=str(reference),
        reference_block_summaries_sha256=reference_sha,
        start_height=TEST_START,
        end_height=TEST_END,
        end_timestamp_exclusive=1_704_067_200,
        request_pause_sec=0.0,
        request_workers=1,
        base_url="https://example.invalid/api",
    )
    return replace(cfg, **changes)


def test_run_fake_pages_resume_manifest_and_deterministic_output(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    requested: list[int] = []

    def fetch(url: str) -> list[dict[str, Any]]:
        cursor = _cursor(url)
        requested.append(cursor)
        return _page(cursor)

    manifest = source.run(cfg, fetch=fetch, sleep=lambda _: None)
    assert requested == [TEST_END, TEST_END - 15]
    assert manifest["source_audit"]["endpoint"] == "https://example.invalid/api/v1/blocks/:start_height"
    assert manifest["source_audit"]["expected_rows"] == 27
    assert manifest["source_audit"]["observed_rows"] == 27
    assert manifest["source_audit"]["utxo_identity_checked"] is True
    assert manifest["source_builder"] == {
        "path": source.SOURCE_BUILDER,
        "sha256": source.sha256_file(source.SOURCE_BUILDER),
    }
    assert manifest["reference_audit"] == {
        "reference_path": cfg.reference_block_summaries,
        "reference_sha256": cfg.reference_block_summaries_sha256,
        "rows_cross_checked": 27,
        "columns_cross_checked": list(source.BASIC_COLUMNS),
        "all_basic_fields_match_reference": True,
    }
    assert manifest["outcome_boundary"] == {
        "market_rows_loaded": 0,
        "funding_rows_loaded": 0,
        "outcome_rows_loaded": 0,
        "return_or_pnl_fields": 0,
        "post_2023_source_rows_loaded": 0,
        "raw_mempool_responses_persisted": False,
        "unrelated_mempool_metadata_persisted": False,
    }
    with gzip.open(cfg.output_csv, "rt", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert list(written[0]) == list(source.OUTPUT_COLUMNS)
    assert [int(row["height"]) for row in written] == list(range(TEST_START, TEST_END + 1))
    assert int(written[0]["utxo_set_change"]) == int(written[0]["total_outputs"]) - int(written[0]["total_inputs"])
    assert json.loads(Path(cfg.manifest_output).read_text()) == manifest

    gzip_before = Path(cfg.output_csv).read_bytes()
    manifest_before = Path(cfg.manifest_output).read_bytes()
    rerun = source.run(
        cfg,
        fetch=lambda _: pytest.fail("complete checkpoint must avoid network"),
        sleep=lambda _: None,
    )
    assert rerun == manifest
    assert Path(cfg.output_csv).read_bytes() == gzip_before
    assert Path(cfg.manifest_output).read_bytes() == manifest_before


def test_interrupted_download_resumes_exactly_after_atomic_page(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    first_requests: list[int] = []

    def fail_second(url: str) -> list[dict[str, Any]]:
        cursor = _cursor(url)
        first_requests.append(cursor)
        if cursor == TEST_END - 15:
            raise TimeoutError("simulated interruption")
        return _page(cursor)

    with pytest.raises(TimeoutError, match="simulated interruption"):
        source.run(cfg, fetch=fail_second, sleep=lambda _: None)
    assert first_requests == [TEST_END, TEST_END - 15]
    assert not Path(cfg.output_csv).exists()
    assert not Path(cfg.manifest_output).exists()
    with sqlite3.connect(cfg.checkpoint_db) as connection:
        assert connection.execute("SELECT count(*) FROM blocks").fetchone()[0] == 15

    resumed: list[int] = []

    def resume(url: str) -> list[dict[str, Any]]:
        cursor = _cursor(url)
        resumed.append(cursor)
        return _page(cursor)

    manifest = source.run(cfg, fetch=resume, sleep=lambda _: None)
    assert resumed == [TEST_END - 15]
    assert manifest["source_audit"]["observed_rows"] == 27


def test_parallel_pages_are_validated_before_ordered_checkpoint_insert(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path, request_workers=2)
    requested: list[int] = []

    def fetch(url: str) -> list[dict[str, Any]]:
        cursor = _cursor(url)
        requested.append(cursor)
        return _page(cursor)

    rows = source.download(cfg, fetch=fetch, sleep=lambda _: None)
    assert set(requested) == {TEST_END, TEST_END - 15}
    assert [row["height"] for row in rows] == list(range(TEST_START, TEST_END + 1))
    with sqlite3.connect(cfg.checkpoint_db) as connection:
        stored = connection.execute(
            "SELECT height FROM blocks ORDER BY height"
        ).fetchall()
    assert [row[0] for row in stored] == list(range(TEST_START, TEST_END + 1))


def test_identity_mismatch_reference_sha_and_reference_field_mismatches_fail(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, reference_block_summaries_sha256="0" * 64)
    with pytest.raises(RuntimeError, match="reference SHA mismatch"):
        source.run(cfg, fetch=lambda url: _page(_cursor(url)), sleep=lambda _: None)

    bad_reference = tmp_path / "bad_reference.csv.gz"
    bad_sha = _write_reference(bad_reference, salt=99)
    cfg = _cfg(
        tmp_path,
        reference_block_summaries=str(bad_reference),
        reference_block_summaries_sha256=bad_sha,
        checkpoint_db=str(tmp_path / "bad-reference-checkpoint.sqlite3"),
        output_csv=str(tmp_path / "bad-reference-output.csv.gz"),
        manifest_output=str(tmp_path / "bad-reference-manifest.json"),
    )
    with pytest.raises(RuntimeError, match="basic field mismatch"):
        source.run(cfg, fetch=lambda url: _page(_cursor(url)), sleep=lambda _: None)


def test_resume_contract_and_checkpoint_integrity_fail_closed(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)

    def fail_second(url: str) -> list[dict[str, Any]]:
        cursor = _cursor(url)
        if cursor == TEST_END - 15:
            raise TimeoutError
        return _page(cursor)

    with pytest.raises(TimeoutError):
        source.download(cfg, fetch=fail_second, sleep=lambda _: None)
    with pytest.raises(RuntimeError, match="resume-state contract mismatch"):
        source.download(replace(cfg, base_url="https://other.invalid/api"), fetch=lambda _: [], sleep=lambda _: None)

    with sqlite3.connect(cfg.checkpoint_db) as connection:
        connection.execute("DELETE FROM blocks WHERE height = ?", (TEST_END - 4,))
        connection.commit()
    with pytest.raises(RuntimeError, match="contiguous height suffix"):
        source.download(cfg, fetch=lambda _: [], sleep=lambda _: None)


def test_cross_page_hash_boundary_break_does_not_commit_failed_page(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)

    def fetch(url: str) -> list[dict[str, Any]]:
        cursor = _cursor(url)
        return _page(cursor, salt=0 if cursor == TEST_END else 1_000_000)

    with pytest.raises(RuntimeError, match="checkpoint boundary linkage"):
        source.download(cfg, fetch=fetch, sleep=lambda _: None)
    with sqlite3.connect(cfg.checkpoint_db) as connection:
        heights = [
            row[0]
            for row in connection.execute("SELECT height FROM blocks ORDER BY height").fetchall()
        ]
    assert heights == list(range(TEST_END - 14, TEST_END + 1))


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda row: row.pop("weight"), "schema drift"),
        (lambda row: row.update(id="ABC"), "64-character hex hash"),
        (lambda row: row.update(height=True), "height must be"),
        (lambda row: row.update(timestamp=0), "timestamp must be"),
        (lambda row: row.update(tx_count=0), "tx_count must be"),
        (lambda row: row.update(size=0), "size must be"),
        (lambda row: row.update(weight=4_000_001), "BIP 141"),
        (lambda row: row.update(weight=999), "serialized block invariant"),
        (lambda row: row.update(extras=None), "extras must be an object"),
        (lambda row: row["extras"].pop("totalFees"), "extras schema drift"),
        (lambda row: row.update(stale=1), "marked stale"),
        (lambda row: row["extras"].update(totalFees=-1), "totalFees"),
        (lambda row: row["extras"].update(totalInputs=True), "totalInputs"),
        (lambda row: row["extras"].update(utxoSetChange=123456), "utxo_set_change"),
    ],
)
def test_row_contract_rejects_schema_type_stale_and_identity_drift(mutator, message: str) -> None:
    row = _row(100)
    mutator(row)
    with pytest.raises(ValueError, match=message):
        source._normalise_row(row, cutoff=1_704_067_200)


def test_cutoff_and_page_shape_failures() -> None:
    accepted = _row(823_785, timestamp=source.FIRST_2024_TIMESTAMP - 1)
    assert source._normalise_row(accepted, cutoff=source.FIRST_2024_TIMESTAMP)["height"] == 823_785
    rejected = _row(823_786, timestamp=source.FIRST_2024_TIMESTAMP)
    with pytest.raises(ValueError, match="outside the frozen interval"):
        source._normalise_row(rejected, cutoff=source.FIRST_2024_TIMESTAMP)

    with pytest.raises(RuntimeError, match="not a list"):
        source._normalise_page({}, cursor=109, start_height=100, cutoff=1_704_067_200)
    with pytest.raises(RuntimeError, match="empty page"):
        source._normalise_page([], cursor=109, start_height=100, cutoff=1_704_067_200)
    with pytest.raises(RuntimeError, match="15-block page cap"):
        source._normalise_page([_row(height) for height in range(115, 99, -1)], cursor=115, start_height=100, cutoff=1_704_067_200)
    with pytest.raises(RuntimeError, match="requested height"):
        source._normalise_page(_page(108), cursor=109, start_height=100, cutoff=1_704_067_200)

    skipped = _page(114)
    skipped.pop(3)
    with pytest.raises(RuntimeError, match="contiguous descending"):
        source._normalise_page(skipped, cursor=114, start_height=100, cutoff=1_704_067_200)

    truncated = _page(114)[:14]
    with pytest.raises(RuntimeError, match="exact requested height suffix"):
        source._normalise_page(truncated, cursor=114, start_height=100, cutoff=1_704_067_200)

    unlinked = _page(114)
    unlinked[0]["previousblockhash"] = _hash(1)
    with pytest.raises(RuntimeError, match="hash-chain linkage"):
        source._normalise_page(unlinked, cursor=114, start_height=100, cutoff=1_704_067_200)


def test_invalid_config_and_artifact_aliases_fail_before_network(tmp_path: Path) -> None:
    for changes in [
        {"start_height": TEST_END + 1},
        {"start_height": source.FROZEN_START_HEIGHT - 1},
        {"end_height": source.FROZEN_END_HEIGHT + 1},
        {"end_timestamp_exclusive": source.FIRST_2024_TIMESTAMP + 1},
        {"request_pause_sec": -1.0},
        {"request_workers": 0},
        {"request_workers": 9},
        {"request_workers": True},
        {"timeout_sec": float("inf")},
        {"maximum_retries": True},
        {"base_url": "file:///tmp/mempool"},
        {"base_url": "https://example.invalid/api?q=scope-drift"},
    ]:
        cfg = _cfg(tmp_path, **changes)
        with pytest.raises(ValueError):
            source.download(cfg, fetch=lambda _: pytest.fail("invalid config must not fetch"), sleep=lambda _: None)

    cfg = _cfg(tmp_path)
    for left, right in [
        ("output_csv", "manifest_output"),
        ("output_csv", "checkpoint_db"),
        ("manifest_output", "checkpoint_db"),
    ]:
        shared = str(tmp_path / f"shared-{left}-{right}")
        bad = replace(cfg, **{left: shared, right: shared})
        with pytest.raises(ValueError, match="paths must be distinct"):
            source.download(bad, fetch=lambda _: pytest.fail("aliased paths must not fetch"), sleep=lambda _: None)
