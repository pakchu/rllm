from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from training import download_bitfinex_margin_funding_stats as source


def _row(timestamp_ms: int, *, used: float = 80.0) -> list[object]:
    return [
        timestamp_ms,
        None,
        None,
        0.000001,
        7.0,
        None,
        None,
        100.0,
        used,
        None,
        None,
        5.0,
    ]


def _ms(value: str) -> int:
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1_000)


def test_parse_row_keeps_official_fields_and_adds_conservative_clock() -> None:
    timestamp = _ms("2020-01-01T00:05:01Z")
    parsed = source.parse_row("fUSD", _row(timestamp))
    assert parsed["observation_time"] == pd.Timestamp("2020-01-01T00:05:01Z")
    assert parsed["available_at"] == pd.Timestamp("2020-01-01T00:15:00Z")
    assert parsed["funding_amount"] == 100.0
    assert parsed["funding_amount_used"] == 80.0


def test_parse_row_rejects_short_or_nonfinite_rows() -> None:
    with pytest.raises(ValueError, match="at least 12"):
        source.parse_row("fUSD", [1, 2, 3])
    row = _row(1)
    row[7] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        source.parse_row("fUSD", row)


def test_validate_page_requires_reverse_chronology_and_bounds() -> None:
    first = _ms("2020-01-01T00:05:00Z")
    second = _ms("2020-01-01T01:05:00Z")
    parsed = source.validate_page(
        "fBTC",
        [_row(second), _row(first)],
        requested_start_ms=first,
        requested_end_ms=second,
        page_limit=250,
    )
    assert [item["timestamp_ms"] for item in parsed] == [second, first]
    with pytest.raises(ValueError, match="reverse chronological"):
        source.validate_page(
            "fBTC",
            [_row(first), _row(second)],
            requested_start_ms=first,
            requested_end_ms=second,
            page_limit=250,
        )


def test_fetch_symbol_uses_cache_and_moves_cursor(monkeypatch, tmp_path) -> None:
    cfg = replace(
        source.Config(),
        start="2020-01-01T00:00:00Z",
        end_exclusive="2020-01-01T03:00:00Z",
        cache_dir=str(tmp_path / "cache"),
        request_pause_seconds=4.0,
    )
    stamps = [
        _ms(value)
        for value in [
            "2020-01-01T02:05:00Z",
            "2020-01-01T01:05:00Z",
            "2020-01-01T00:05:00Z",
        ]
    ]
    calls: list[str] = []

    def fake_request(url: str, _cfg: source.Config) -> list[list[object]]:
        calls.append(url)
        return [_row(value) for value in stamps] if len(calls) % 2 == 1 else []

    monkeypatch.setattr(source, "_request_json", fake_request)
    first, diagnostics = source.fetch_symbol("fUSD", cfg)
    assert len(first) == 3
    assert diagnostics == {"network_requests": 2, "cached_pages": 0}
    second, diagnostics = source.fetch_symbol("fUSD", cfg)
    assert second == first
    assert diagnostics == {"network_requests": 0, "cached_pages": 2}
    assert len(calls) == 2


def test_validate_frame_rejects_post_boundary_and_used_above_total() -> None:
    cfg = replace(
        source.Config(),
        start="2020-01-01T00:00:00Z",
        end_exclusive="2020-01-01T02:00:00Z",
    )
    rows = []
    for symbol in source.SYMBOLS:
        for value in ["2020-01-01T00:05:00Z", "2020-01-01T01:05:00Z"]:
            rows.append(source.parse_row(symbol, _row(_ms(value))))
    frame, diagnostics = source.validate_frame(pd.DataFrame(rows), cfg)
    assert len(frame) == 4
    assert diagnostics["fUSD"]["missing_utc_hours"] == 0
    broken = frame.copy()
    broken.loc[0, "funding_amount_used"] = 101.0
    with pytest.raises(ValueError, match="exceeds total"):
        source.validate_frame(broken, cfg)


def test_deterministic_gzip_has_stable_hash(tmp_path) -> None:
    payload = b"a,b\n1,2\n"
    first = tmp_path / "first.csv.gz"
    second = tmp_path / "second.csv.gz"
    first_hash = source._write_deterministic_gzip(payload, first)
    second_hash = source._write_deterministic_gzip(payload, second)
    assert first.read_bytes() == second.read_bytes()
    assert first_hash == second_hash
    assert first_hash == hashlib.sha256(first.read_bytes()).hexdigest()


def test_source_builder_is_physically_pre2024() -> None:
    text = Path(source.SOURCE_BUILDER).read_text()
    assert 'end_exclusive: str = "2024-01-01T00:00:00Z"' in text
    assert '"outcomes_opened": False' in text
    assert '"post_2023_rows_requested": False' in text
