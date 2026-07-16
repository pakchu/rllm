from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from training import export_crrc_2023_execution_sources as export
from training.preregister_cross_venue_radial_refill_compression import canonical_hash


MANIFEST = Path("results/crrc_2023_execution_source_manifest_2026-07-17.json")


def test_physical_execution_source_manifest_is_outcome_blind_and_self_consistent() -> None:
    payload = json.loads(MANIFEST.read_text())
    body = {
        key: value
        for key, value in payload.items()
        if key not in {"manifest_hash", "created_at"}
    }
    assert canonical_hash(body) == payload["manifest_hash"]
    assert payload["outcomes_calculated"] is False
    assert payload["signals_labels_or_equity_constructed"] is False
    assert payload["2024_rows_parsed"] == 0
    assert payload["2024_rows_written"] == 0
    assert payload["selection_prefix"] == [
        "2023-01-01 00:00:00",
        "2024-01-01 00:00:00",
    ]


def test_physical_market_and_funding_files_match_manifest() -> None:
    payload = json.loads(MANIFEST.read_text())
    market_path = Path(payload["market_output"])
    funding_path = Path(payload["funding_output"])
    assert export.sha256_file(market_path) == payload["market_output_sha256"]
    assert export.sha256_file(funding_path) == payload["funding_output_sha256"]
    market = export.validate_market_2023(pd.read_csv(market_path))
    funding = export.validate_funding_2023(pd.read_csv(funding_path))
    assert len(market) == payload["market_rows"] == 105_120
    assert len(funding) == payload["funding_rows"] == 1_095
    assert int((funding["event_time"].dt.microsecond != 0).sum()) == (
        payload["funding_nonzero_millisecond_offsets"]
    ) == 405


def test_upstream_pre2024_source_hashes_are_frozen() -> None:
    payload = json.loads(MANIFEST.read_text())
    source = payload["source"]
    assert source["market_source_sha256"] == export.MARKET_SOURCE_SHA256
    assert source["funding_source_sha256"] == export.FUNDING_SOURCE_SHA256
    assert source["market_manifest_sha256"] == export.MARKET_MANIFEST_SHA256
    assert source["funding_manifest_sha256"] == export.FUNDING_MANIFEST_SHA256
