from __future__ import annotations

import json
from pathlib import Path

import pytest

from training import seal_gdelt_gnrc_premarket_access as seal


PREMARKET_SEAL_SHA256 = (
    "eef502ab306a074f790593e10f3e7bf52642d7605433a4e5e3cf2b0e07a98478"
)


def _manifest_pair() -> tuple[dict[str, object], dict[str, object]]:
    market: dict[str, object] = {
        "config": {
            "symbol": "BTCUSDT",
            "interval": "5m",
            "start": "2020-01-01",
            "end": "2024-01-01",
        },
        "protocol": {
            "source": "official Binance USD-M daily kline archives",
            "archive_checksums_verified": True,
            "end_is_exclusive": True,
            "outcomes_opened": False,
        },
        "combined_output": str(seal.MARKET_DATA),
        "combined_sha256": seal.MARKET_DATA_SHA256,
        "rows": 1_461 * 24 * 12,
        "first_date": "2020-01-01 00:00:00",
        "last_date": "2023-12-31 23:55:00",
        "columns": list(seal.EXPECTED_MARKET_COLUMNS),
    }
    funding_core: dict[str, object] = {
        "protocol_version": "btc_um_funding_settlement_marks_2020_2023_v1",
        "outcomes_opened": False,
        "strategy_outcomes_calculated": [],
        "data": {
            "path": str(seal.FUNDING_DATA),
            "sha256": seal.FUNDING_DATA_SHA256,
            "rows": 1_461 * 3,
            "columns": list(seal.EXPECTED_FUNDING_COLUMNS),
        },
        "mapping": {
            "funding_time": "exact returned fundingTime retained",
            "mark": "open of floor(fundingTime, 8h) official mark-price kline",
            "maximum_allowed_timestamp_offset_ms": 60_000,
        },
    }
    funding = {
        **funding_core,
        "manifest_hash": seal.canonical_hash(funding_core),
        "created_at": "2026-07-17T00:00:00Z",
    }
    return market, funding


def test_manifest_metadata_rejects_outcome_opening() -> None:
    market, funding = _manifest_pair()
    seal.validate_manifest_metadata(market, funding)
    protocol = market["protocol"]
    assert isinstance(protocol, dict)
    protocol["outcomes_opened"] = True
    with pytest.raises(ValueError, match="market-manifest"):
        seal.validate_manifest_metadata(market, funding)


def test_build_seal_binds_every_frozen_input_without_value_parsers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    market, funding = _manifest_pair()
    hashes = {
        seal.SOURCE_SUPPORT_REPORT: seal.SOURCE_SUPPORT_REPORT_SHA256,
        seal.EVALUATOR_SOURCE: seal.EVALUATOR_SOURCE_SHA256,
        seal.PROTOCOL_DOCUMENT: seal.PROTOCOL_DOCUMENT_SHA256,
        seal.TEST_SOURCE: seal.TEST_SOURCE_SHA256,
        seal.MARKET_DATA: seal.MARKET_DATA_SHA256,
        seal.MARKET_MANIFEST: seal.MARKET_MANIFEST_SHA256,
        seal.FUNDING_DATA: seal.FUNDING_DATA_SHA256,
        seal.FUNDING_MANIFEST: seal.FUNDING_MANIFEST_SHA256,
    }
    monkeypatch.setattr(seal, "sha256_file", lambda path: hashes[Path(path)])
    monkeypatch.setattr(
        seal,
        "_load_json",
        lambda path: market if Path(path) == seal.MARKET_MANIFEST else funding,
    )
    payload = seal.build_seal()
    assert payload["evaluator_source_sha256"] == seal.EVALUATOR_SOURCE_SHA256
    assert payload["test_source_sha256"] == seal.TEST_SOURCE_SHA256
    assert payload["market_values_inspected_before_seal"] is False
    assert payload["funding_values_inspected_before_seal"] is False
    source = seal.repository_path("training/seal_gdelt_gnrc_premarket_access.py")
    text = source.read_text(encoding="utf-8")
    assert "import gzip" not in text
    assert "import csv" not in text
    assert "pandas" not in text


def test_premarket_seal_is_write_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = {"protocol_version": seal.PROTOCOL_VERSION}
    monkeypatch.setattr(seal, "build_seal", lambda: payload)
    output = tmp_path / "seal.json"
    assert seal.write_once(output) == payload
    assert json.loads(output.read_text()) == payload
    with pytest.raises(FileExistsError, match="write-once"):
        seal.write_once(output)


def test_committed_premarket_seal_matches_frozen_builder_exactly() -> None:
    assert seal.sha256_file(seal.DEFAULT_OUTPUT) == PREMARKET_SEAL_SHA256
    with seal.repository_path(seal.DEFAULT_OUTPUT).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload == seal.build_seal()
