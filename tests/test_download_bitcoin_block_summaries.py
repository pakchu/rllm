from __future__ import annotations

import csv
from dataclasses import replace
from decimal import Decimal
import gzip
import json
from pathlib import Path
import sqlite3
from typing import Any
import urllib.parse

import pytest

from training import download_bitcoin_block_summaries as source


TEST_START = 823_700
TEST_END = 823_716


def _hash(height: int, *, salt: int = 0) -> str:
    return f"{height + salt:064x}"


def _row(
    height: int,
    *,
    salt: int = 0,
    timestamp: int | None = None,
) -> dict[str, Any]:
    return {
        "id": _hash(height, salt=salt),
        "height": height,
        "version": 1,
        "timestamp": timestamp if timestamp is not None else 1_700_000_000 + height,
        "tx_count": 100 + height,
        "size": 1_000,
        "weight": 3_000,
        "merkle_root": _hash(height, salt=2_000_000),
        "previousblockhash": _hash(height - 1, salt=salt),
        "mediantime": 1_699_999_000 + height,
        "nonce": height,
        "bits": 486_604_799,
        "difficulty": Decimal("1.25"),
    }


def _page(cursor: int, *, salt: int = 0) -> list[dict[str, Any]]:
    return [_row(height, salt=salt) for height in range(cursor, cursor - 10, -1)]


def _cfg(tmp_path: Path, **changes: object) -> source.Config:
    cfg = source.Config(
        output_csv=str(tmp_path / "blocks.csv.gz"),
        manifest_output=str(tmp_path / "manifest.json"),
        checkpoint_db=str(tmp_path / "checkpoint.sqlite3"),
        start_height=TEST_START,
        end_height=TEST_END,
        end_timestamp_exclusive=1_704_067_200,
        request_pause_sec=0.0,
        base_url="https://example.invalid/api",
    )
    return replace(cfg, **changes)


def _cursor(url: str) -> int:
    return int(urllib.parse.urlparse(url).path.rsplit("/", 1)[-1])


def test_run_paginates_descending_and_writes_exact_deterministic_range(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    requested: list[int] = []

    def fetch(url: str) -> list[dict[str, Any]]:
        cursor = _cursor(url)
        requested.append(cursor)
        return _page(cursor)

    manifest = source.run(cfg, fetch=fetch, sleep=lambda _: None)
    assert requested == [TEST_END, TEST_END - 10]
    assert manifest["source_audit"] == {
        "endpoint": "https://example.invalid/api/blocks/:start_height",
        "official_api": source.OFFICIAL_API,
        "expected_rows": 17,
        "observed_rows": 17,
        "start_height": TEST_START,
        "end_height": TEST_END,
        "latest_eligible_packet_end": TEST_END - 6,
        "minimum_header_timestamp": 1_700_000_000 + TEST_START,
        "maximum_header_timestamp": 1_700_000_000 + TEST_END,
        "end_timestamp_exclusive": 1_704_067_200,
        "height_links_checked": 16,
        "complete_inclusive_height_range": True,
        "unique_block_hashes": True,
        "all_rows_pre_cutoff": True,
    }
    assert manifest["outcome_boundary"] == {
        "market_rows_loaded": 0,
        "funding_rows_loaded": 0,
        "return_or_pnl_fields": 0,
        "post_2023_source_rows_loaded": 0,
        "raw_esplora_responses_persisted": False,
    }
    with gzip.open(cfg.output_csv, "rt", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert [int(row["height"]) for row in written] == list(
        range(TEST_START, TEST_END + 1)
    )
    assert list(written[0]) == list(source.OUTPUT_COLUMNS)
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


def test_interrupted_download_resumes_after_last_atomic_page(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    first_requests: list[int] = []

    def fail_second(url: str) -> list[dict[str, Any]]:
        cursor = _cursor(url)
        first_requests.append(cursor)
        if cursor == TEST_END - 10:
            raise TimeoutError("simulated interruption")
        return _page(cursor)

    with pytest.raises(TimeoutError, match="simulated interruption"):
        source.run(cfg, fetch=fail_second, sleep=lambda _: None)
    assert first_requests == [TEST_END, TEST_END - 10]
    assert not Path(cfg.output_csv).exists()
    assert not Path(cfg.manifest_output).exists()
    with sqlite3.connect(cfg.checkpoint_db) as connection:
        assert connection.execute("SELECT count(*) FROM blocks").fetchone()[0] == 10

    resumed: list[int] = []

    def resume(url: str) -> list[dict[str, Any]]:
        cursor = _cursor(url)
        resumed.append(cursor)
        return _page(cursor)

    manifest = source.run(cfg, fetch=resume, sleep=lambda _: None)
    assert resumed == [TEST_END - 10]
    assert manifest["source_audit"]["observed_rows"] == 17


def test_resume_contract_and_checkpoint_continuity_fail_closed(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)

    def fail_second(url: str) -> list[dict[str, Any]]:
        cursor = _cursor(url)
        if cursor == TEST_END - 10:
            raise TimeoutError
        return _page(cursor)

    with pytest.raises(TimeoutError):
        source.download(cfg, fetch=fail_second, sleep=lambda _: None)
    with pytest.raises(RuntimeError, match="resume-state contract mismatch"):
        source.download(
            replace(cfg, base_url="https://other.invalid/api"),
            fetch=lambda _: [],
            sleep=lambda _: None,
        )

    with sqlite3.connect(cfg.checkpoint_db) as connection:
        connection.execute("DELETE FROM blocks WHERE height = ?", (TEST_END - 4,))
        connection.commit()
    with pytest.raises(RuntimeError, match="contiguous height suffix"):
        source.download(cfg, fetch=lambda _: [], sleep=lambda _: None)


def test_resume_revalidates_canonical_checkpoint_fields(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    source.download(cfg, fetch=lambda url: _page(_cursor(url)), sleep=lambda _: None)
    with sqlite3.connect(cfg.checkpoint_db) as connection:
        connection.execute(
            "UPDATE blocks SET tx_count = 0 WHERE height = ?", (TEST_END - 6,)
        )
        connection.commit()
    with pytest.raises(ValueError, match="checkpoint tx_count"):
        source.download(
            cfg,
            fetch=lambda _: pytest.fail("corrupt checkpoint must not fetch"),
            sleep=lambda _: None,
        )


def test_cross_page_link_break_does_not_commit_failed_page(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)

    def fetch(url: str) -> list[dict[str, Any]]:
        cursor = _cursor(url)
        return _page(cursor, salt=0 if cursor == TEST_END else 1_000_000)

    with pytest.raises(RuntimeError, match="checkpoint boundary linkage"):
        source.download(cfg, fetch=fetch, sleep=lambda _: None)
    with sqlite3.connect(cfg.checkpoint_db) as connection:
        heights = [
            row[0]
            for row in connection.execute(
                "SELECT height FROM blocks ORDER BY height"
            ).fetchall()
        ]
    assert heights == list(range(TEST_END - 9, TEST_END + 1))


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda row: row.pop("weight"), "schema drift"),
        (lambda row: row.update(future_return=1), "schema drift"),
        (lambda row: row.update(id="ABC"), "64-character hex hash"),
        (lambda row: row.update(height=True), "height must be"),
        (lambda row: row.update(timestamp=0), "timestamp must be"),
        (lambda row: row.update(tx_count=0), "tx_count must be"),
        (lambda row: row.update(size=0), "size must be"),
        (lambda row: row.update(weight=4_000_001), "BIP 141"),
        (lambda row: row.update(weight=999), "serialized block invariant"),
        (lambda row: row.update(difficulty=1.5), "exact JSON number"),
    ],
)
def test_row_contract_rejects_schema_and_type_drift(mutator, message: str) -> None:
    row = _row(100)
    mutator(row)
    with pytest.raises(ValueError, match=message):
        source._normalise_row(row, cutoff=1_704_067_200)


def test_strict_pre_2024_timestamp_boundary() -> None:
    accepted = _row(823_785, timestamp=1_704_067_199)
    assert source._normalise_row(
        accepted, cutoff=source.FIRST_2024_TIMESTAMP
    )["height"] == 823_785
    rejected = _row(823_786, timestamp=source.FIRST_2024_TIMESTAMP)
    with pytest.raises(ValueError, match="outside the frozen interval"):
        source._normalise_row(rejected, cutoff=source.FIRST_2024_TIMESTAMP)


def test_page_contract_rejects_malformed_truncated_and_unlinked_payloads() -> None:
    with pytest.raises(RuntimeError, match="not a list"):
        source._normalise_page(
            {}, cursor=109, start_height=100, cutoff=1_704_067_200
        )
    with pytest.raises(RuntimeError, match="empty page"):
        source._normalise_page(
            [], cursor=109, start_height=100, cutoff=1_704_067_200
        )
    with pytest.raises(RuntimeError, match="ten-block page cap"):
        source._normalise_page(
            [_row(height) for height in range(110, 99, -1)],
            cursor=110,
            start_height=100,
            cutoff=1_704_067_200,
        )
    with pytest.raises(ValueError, match="item must be an object"):
        source._normalise_page(
            [None], cursor=100, start_height=100, cutoff=1_704_067_200
        )

    skipped = _page(109)
    skipped.pop(3)
    with pytest.raises(RuntimeError, match="contiguous descending"):
        source._normalise_page(
            skipped, cursor=109, start_height=100, cutoff=1_704_067_200
        )

    truncated = _page(109)[:9]
    with pytest.raises(RuntimeError, match="exact requested height suffix"):
        source._normalise_page(
            truncated, cursor=109, start_height=100, cutoff=1_704_067_200
        )

    unlinked = _page(109)
    unlinked[0]["previousblockhash"] = _hash(1)
    with pytest.raises(RuntimeError, match="hash-chain linkage"):
        source._normalise_page(
            unlinked, cursor=109, start_height=100, cutoff=1_704_067_200
        )


def test_final_page_excludes_rows_below_inclusive_start() -> None:
    page = source._normalise_page(
        _page(106),
        cursor=106,
        start_height=100,
        cutoff=1_704_067_200,
    )
    assert [row["height"] for row in page] == list(range(106, 99, -1))
    assert all(row["height"] != 99 for row in page)


@pytest.mark.parametrize(
    "changes",
    [
        {"start_height": TEST_END + 1},
        {"start_height": TEST_END - 5},
        {"start_height": source.FROZEN_START_HEIGHT - 1},
        {"end_height": source.FROZEN_END_HEIGHT + 1},
        {"end_timestamp_exclusive": source.FIRST_2024_TIMESTAMP + 1},
        {"request_pause_sec": -1.0},
        {"timeout_sec": float("inf")},
        {"maximum_retries": True},
        {"base_url": "file:///tmp/esplora"},
        {"base_url": "https://example.invalid/api?q=scope-drift"},
    ],
)
def test_invalid_config_fails_before_network(tmp_path: Path, changes: dict[str, Any]) -> None:
    cfg = _cfg(tmp_path, **changes)
    with pytest.raises(ValueError):
        source.download(
            cfg,
            fetch=lambda _: pytest.fail("invalid config must not fetch"),
            sleep=lambda _: None,
        )


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("output_csv", "manifest_output"),
        ("output_csv", "checkpoint_db"),
        ("manifest_output", "checkpoint_db"),
    ],
)
def test_artifact_paths_must_be_pairwise_distinct(
    tmp_path: Path, left: str, right: str
) -> None:
    cfg = _cfg(tmp_path)
    shared = str(tmp_path / "shared.artifact")
    cfg = replace(cfg, **{left: shared, right: shared})
    with pytest.raises(ValueError, match="paths must be distinct"):
        source.download(
            cfg,
            fetch=lambda _: pytest.fail("aliased paths must not fetch"),
            sleep=lambda _: None,
        )
