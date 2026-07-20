from __future__ import annotations

import csv
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import gzip
import json
from pathlib import Path
from typing import cast
import urllib.parse

import pytest

from training import download_deribit_btc_perpetual_funding_hourly as source


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
        timezone.utc
    )


def _ms(value: str | datetime) -> int:
    timestamp = _dt(value) if isinstance(value, str) else value
    return int(timestamp.timestamp() * 1000)


def _row(
    timestamp: datetime,
    *,
    interest_1h: Decimal = Decimal("0.001"),
    interest_8h: Decimal = Decimal("0.008"),
    index_price: Decimal = Decimal("100"),
    prev_index_price: Decimal = Decimal("99"),
) -> dict[str, object]:
    return {
        "timestamp": _ms(timestamp),
        "interest_1h": interest_1h,
        "interest_8h": interest_8h,
        "index_price": index_price,
        "prev_index_price": prev_index_price,
    }


def _rows(start: str, hours: int) -> list[dict[str, object]]:
    first = _dt(start)
    rows: list[dict[str, object]] = []
    previous = Decimal("99")
    for offset in range(hours):
        price = Decimal(100 + offset)
        rows.append(
            _row(
                first + timedelta(hours=offset),
                index_price=price,
                prev_index_price=previous,
            )
        )
        previous = price
    return rows


def _payload(rows: list[dict[str, object]], **changes: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "jsonrpc": "2.0",
        "result": rows,
        "testnet": False,
        "usIn": 1_000_000,
        "usOut": 1_000_100,
        "usDiff": 100,
    }
    payload.update(changes)
    return payload


def _cfg(tmp_path: Path, **changes: object) -> source.Config:
    cfg = source.Config(
        output_csv=str(tmp_path / "source.csv.gz"),
        manifest_output=str(tmp_path / "manifest.json"),
        start="2023-01-01T00:00:00Z",
        end_exclusive="2023-01-02T00:00:00Z",
        chunk_hours=8,
        request_pause_sec=0.0,
    )
    return replace(cfg, **changes)


def _fetch_rows(
    rows: list[dict[str, object]], *, inclusive_start: bool = False
):
    def fetch(url: str) -> dict[str, object]:
        query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        start = int(query["start_timestamp"][0])
        end = int(query["end_timestamp"][0])
        selected = [
            row
            for row in rows
            if (
                (
                    cast(int, row["timestamp"]) >= start
                    if inclusive_start
                    else cast(int, row["timestamp"]) > start
                )
                and cast(int, row["timestamp"]) <= end
            )
        ]
        return _payload(selected)

    return fetch


def test_request_windows_preserve_exclusive_output_and_overlap_boundary(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path, end_exclusive="2023-01-01T10:00:00Z", chunk_hours=4)
    windows = source.request_windows(cfg)
    assert [(start.hour, end.hour) for start, end, _ in windows] == [
        (0, 4),
        (4, 8),
        (8, 10),
    ]
    first = urllib.parse.parse_qs(urllib.parse.urlparse(windows[0][2]).query)
    second = urllib.parse.parse_qs(urllib.parse.urlparse(windows[1][2]).query)
    assert first == {
        "instrument_name": ["BTC-PERPETUAL"],
        "start_timestamp": [str(_ms("2022-12-31T23:00:00Z"))],
        "end_timestamp": [str(_ms("2023-01-01T03:00:00Z"))],
    }
    assert second["start_timestamp"] == [str(_ms("2023-01-01T03:00:00Z"))]
    assert second["end_timestamp"] == [str(_ms("2023-01-01T07:00:00Z"))]


def test_run_writes_complete_deterministic_source_and_manifest(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    rows = _rows(cfg.start, 24)
    manifest = source.run(
        cfg, fetch=_fetch_rows(rows), sleep=lambda _: None
    )
    audit = manifest["source_audit"]
    assert audit["request_windows"] == 3
    assert audit["response_result_lengths"] == [8, 8, 8]
    assert audit["requested_hours"] == audit["observed_rows"] == 24
    assert audit["coverage_ratio"] == 1.0
    assert audit["missing_hours"] == 0
    assert audit["maximum_observation_gap_hours"] == 1
    assert audit["contiguous_index_price_links_checked"] == 23
    assert audit["memory_identity"]["contiguous_eight_hour_windows"] == 17
    assert audit["memory_identity"]["maximum_absolute_sum1h_minus_8h"] == "0.000"
    assert manifest["outcome_boundary"] == {
        "binance_market_rows_loaded": 0,
        "binance_funding_rows_loaded": 0,
        "return_or_pnl_fields": 0,
        "post_2023_source_rows_loaded": 0,
        "raw_deribit_responses_persisted": False,
    }
    with gzip.open(cfg.output_csv, "rt", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == 24
    assert written[0]["timestamp"] == cfg.start
    assert written[0]["available_at"] == "2023-01-01T00:05:00Z"
    assert written[0]["interest_1h"] == "0.001"
    assert written[-1]["timestamp"] == "2023-01-01T23:00:00Z"
    assert json.loads(Path(cfg.manifest_output).read_text()) == manifest


def test_http_decode_preserves_exact_decimal_lexemes() -> None:
    raw = (
        b'{"jsonrpc":"2.0","result":[{"timestamp":1672531200000,'
        b'"interest_1h":0.123456789123456789,"interest_8h":-1e-18,'
        b'"index_price":16500.123456789,"prev_index_price":16499.9}],'
        b'"testnet":false,"usIn":100,"usOut":109,"usDiff":9}'
    )
    payload = source._decode_payload(raw)
    row = payload["result"][0]
    assert row["interest_1h"] == Decimal("0.123456789123456789")
    assert row["interest_8h"] == Decimal("-1E-18")
    normalised = source._normalise_row(
        row,
        allowed_start=_dt("2023-01-01T00:00:00Z"),
        allowed_end=_dt("2023-01-01T01:00:00Z"),
        availability_delay_minutes=5,
    )
    assert normalised["interest_1h"] == "0.123456789123456789"
    assert normalised["interest_8h"] == "-0.000000000000000001"


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda row: row.update(timestamp=_ms("2023-01-01T00:00:01Z")), "whole UTC hour"),
        (lambda row: row.update(index_price=Decimal("0")), "index_price must be positive"),
        (lambda row: row.update(prev_index_price=Decimal("NaN")), "must be finite"),
        (lambda row: row.update(interest_1h=True), "exact JSON number"),
        (lambda row: row.update(interest_8h=0.1), "exact JSON number"),
        (lambda row: row.update(extra_future_return=Decimal("1")), "schema drift"),
    ],
)
def test_row_contract_fails_closed(mutator, message: str) -> None:
    row = _row(_dt("2023-01-01T00:00:00Z"))
    mutator(row)
    with pytest.raises(ValueError, match=message):
        source._normalise_row(
            row,
            allowed_start=_dt("2023-01-01T00:00:00Z"),
            allowed_end=_dt("2023-01-01T01:00:00Z"),
            availability_delay_minutes=5,
        )


def test_payload_environment_schema_timing_and_cap_fail_closed() -> None:
    valid = _payload([])
    assert source._validate_payload(valid, maximum_rows=1) == []
    for changed, message in [
        ({**valid, "future": []}, "schema drift"),
        ({**valid, "testnet": True}, "environment changed"),
        ({**valid, "jsonrpc": "1.0"}, "environment changed"),
        ({**valid, "usDiff": 99}, "timing metadata is inconsistent"),
        ({**valid, "result": {}}, "must be a list"),
        ({**valid, "error": {"message": "bad"}}, "API error"),
    ]:
        with pytest.raises((RuntimeError, ValueError), match=message):
            source._validate_payload(changed, maximum_rows=1)
    with pytest.raises(RuntimeError, match="truncation cap"):
        source._validate_payload(_payload([{}] * 744), maximum_rows=800)
    with pytest.raises(RuntimeError, match="truncation cap"):
        source._validate_payload(_payload([{}] * 2), maximum_rows=1)


def test_exact_overlapped_boundaries_are_deduplicated(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, end_exclusive="2023-01-01T12:00:00Z", chunk_hours=4)
    rows = _rows("2022-12-31T23:00:00Z", 13)
    downloaded, audit = source.download_rows(
        cfg,
        fetch=_fetch_rows(rows, inclusive_start=True),
        sleep=lambda _: None,
    )
    assert len(downloaded) == 12
    assert audit["exact_boundary_duplicates"] == 2
    assert downloaded[0]["timestamp"] == cfg.start


def test_conflicting_or_nonboundary_duplicates_are_rejected(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, end_exclusive="2023-01-01T08:00:00Z", chunk_hours=4)
    rows = _rows("2022-12-31T23:00:00Z", 9)
    calls = 0

    def conflicting_fetch(url: str) -> dict[str, object]:
        nonlocal calls
        payload = _fetch_rows(rows, inclusive_start=True)(url)
        calls += 1
        if calls == 2:
            result = cast(list[dict[str, object]], payload["result"])
            changed = dict(result[0])
            changed["interest_1h"] = Decimal("0.002")
            result[0] = changed
        return payload

    with pytest.raises(RuntimeError, match="boundary duplicate conflicts"):
        source.download_rows(cfg, fetch=conflicting_fetch, sleep=lambda _: None)

    duplicate_cfg = replace(cfg, chunk_hours=8)
    duplicate = _rows(cfg.start, 8)
    duplicate.insert(2, dict(duplicate[1]))
    with pytest.raises(RuntimeError, match="strictly time-ascending"):
        source.download_rows(
            duplicate_cfg,
            fetch=lambda _: _payload(duplicate),
            sleep=lambda _: None,
        )


def test_missing_hours_are_reported_without_forward_fill(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, chunk_hours=24)
    rows = _rows(cfg.start, 24)
    del rows[10]
    downloaded, audit = source.download_rows(
        cfg, fetch=_fetch_rows(rows), sleep=lambda _: None
    )
    assert len(downloaded) == 23
    assert audit["missing_hours"] == 1
    assert audit["missing_hours_head"] == ["2023-01-01T10:00:00Z"]
    assert audit["maximum_observation_gap_hours"] == 2
    assert audit["contiguous_index_price_links_checked"] == 21


def test_contiguous_index_price_chain_conflict_is_rejected(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, chunk_hours=24)
    rows = _rows(cfg.start, 24)
    rows[7]["prev_index_price"] = Decimal("1")
    with pytest.raises(RuntimeError, match="index-price chain changed"):
        source.download_rows(
            cfg, fetch=_fetch_rows(rows), sleep=lambda _: None
        )


def test_memory_semantic_drift_is_rejected(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, chunk_hours=24)
    rows = _rows(cfg.start, 24)
    rows[7]["interest_8h"] = Decimal("0.1")
    with pytest.raises(RuntimeError, match="memory invariant drifted"):
        source.download_rows(
            cfg, fetch=_fetch_rows(rows), sleep=lambda _: None
        )


def test_response_order_and_window_scope_are_rejected(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, end_exclusive="2023-01-01T04:00:00Z", chunk_hours=4)
    reversed_rows = list(reversed(_rows(cfg.start, 4)))
    with pytest.raises(RuntimeError, match="strictly time-ascending"):
        source.download_rows(
            cfg, fetch=lambda _: _payload(reversed_rows), sleep=lambda _: None
        )
    outside = _rows("2023-01-01T04:00:00Z", 1)
    with pytest.raises(ValueError, match="outside its frozen request window"):
        source.download_rows(
            cfg, fetch=lambda _: _payload(outside), sleep=lambda _: None
        )


def test_output_and_manifest_are_byte_deterministic(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    rows = _rows(cfg.start, 24)
    first = source.run(cfg, fetch=_fetch_rows(rows), sleep=lambda _: None)
    source_bytes = Path(cfg.output_csv).read_bytes()
    manifest_bytes = Path(cfg.manifest_output).read_bytes()
    second = source.run(cfg, fetch=_fetch_rows(rows), sleep=lambda _: None)
    assert Path(cfg.output_csv).read_bytes() == source_bytes
    assert Path(cfg.manifest_output).read_bytes() == manifest_bytes
    assert first["manifest_hash"] == second["manifest_hash"]


def test_configuration_contracts_fail_closed(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    bad = [
        replace(cfg, start="2023-01-01"),
        replace(cfg, start="2023-01-01T00:01:00Z"),
        replace(cfg, end_exclusive=cfg.start),
        replace(cfg, chunk_hours=0),
        replace(cfg, chunk_hours=721),
        replace(cfg, chunk_hours=1.5),
        replace(cfg, synthetic_availability_delay_minutes=4),
        replace(cfg, maximum_memory_abs_error="0"),
        replace(cfg, maximum_memory_abs_error="NaN"),
        replace(cfg, timeout_sec=0.0),
        replace(cfg, request_pause_sec=-1.0),
        replace(cfg, maximum_retries=-1),
        replace(cfg, base_url="https://example.com/api"),
    ]
    for changed in bad:
        with pytest.raises(ValueError):
            source.request_windows(changed)
