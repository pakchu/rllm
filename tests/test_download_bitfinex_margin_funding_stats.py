from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from training import download_bitfinex_margin_funding_stats as source
from training import download_bitfinex_margin_funding_stats_v2 as source_v2


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


def test_v2_availability_never_precedes_late_observation() -> None:
    regular = source_v2.parse_row(
        "fUSD", _row(_ms("2020-01-01T00:05:28Z"))
    )
    late = source_v2.parse_row("fBTC", _row(_ms("2020-01-01T00:34:20Z")))
    assert regular["available_at"] == pd.Timestamp("2020-01-01T00:15:00Z")
    assert late["available_at"] == pd.Timestamp("2020-01-01T00:35:00Z")
    assert late["available_at"] >= late["observation_time"]


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


def test_transport_v2_amendment_is_hash_bound_and_outcome_blind() -> None:
    path = Path(
        "results/bitfinex_margin_funding_stats_transport_v2_amendment_2026-07-20.json"
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "1fc2d1b35242e7a1bd8232b3b0dfe65d479d0f8e2c4240c523efea1937dd00e9"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest_hash = payload.pop("manifest_hash")
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    assert manifest_hash == hashlib.sha256(encoded).hexdigest()
    assert payload["bindings"]["v2_builder"]["sha256"] == (
        "b3bb9434dec618c8724ad584caa2fb66cd705d210dd66889b32ab80fd8f480ca"
    )
    assert payload["outcome_boundary"]["feature_values_inspected"] is False
    assert payload["outcome_boundary"]["outcomes_opened"] is False


def test_frozen_v2_source_artifacts_match_manifest_without_opening_features() -> None:
    manifest_path = Path(
        "results/bitfinex_margin_funding_stats_source_manifest_2026-07-20.json"
    )
    canonical = Path("data/bitfinex_margin_funding_stats_2020_2023.csv.gz")
    raw = Path("data/bitfinex_margin_funding_stats_raw_2020_2023.jsonl.gz")
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == (
        "9d7c13d56983d7d33fec1c17e24f1794baca64fcfc666599b798d5d5b49cf9b9"
    )
    assert hashlib.sha256(canonical.read_bytes()).hexdigest() == (
        "71635b9f3a38efa7422a6fcf616859e6a41636bbb79ff0f85e160ef395b0d53c"
    )
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == (
        "2f5ca2b344806be5bbfa63090fb79a86259d722e03c4f136cd316eb5787f8adb"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["protocol_version"] == source_v2.PROTOCOL_VERSION
    assert manifest["files"]["canonical"]["rows"] == 70_116
    assert manifest["files"]["raw"]["rows"] == 70_116
    assert manifest["source_contract"]["late_observation_fallback_rows"] == 100
    assert manifest["source_contract"]["outcomes_opened"] is False
    assert manifest["source_contract"]["post_2023_rows_requested"] is False
