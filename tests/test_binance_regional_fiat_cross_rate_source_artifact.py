from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from training import build_binance_regional_fiat_cross_rate as builder


ROOT = Path("data/binance_regional_fiat_cross_rate_btc_2020-11_2023")
PANEL = ROOT / "BTC_regional_fiat_cross_rate_1d_2020-11-01_2023-12-31.csv.gz"
MANIFEST = ROOT / "build_manifest.json"
REBUILD_LOG = Path("results/rfxs2_source_rebuild_2026-07-20.log")
REBUILD_ATTESTATION = Path(
    "results/rfxs2_source_rebuild_attestation_2026-07-20.json"
)
PANEL_SHA256 = "5dbc697c8299ac892295a01302e9f2d883a6e252c8d3d85a8f60f3a369b533d3"
MANIFEST_SHA256 = "627fdd8298312ea61c2bfaa14d93d623e61d562d64abea0a3769d79c3a68673c"
REBUILD_LOG_SHA256 = "98862eec4d7156946ffba2e5e33c9ab91948f4a7a31e126ecbd5acbd6e87440a"
REBUILD_ATTESTATION_SHA256 = (
    "c1505a952487b9c4725f3a04fd8192312fa9011fe2610a6e0703fd61130c70a1"
)
BUILDER_COMMIT = "8259a5121959ea92735a4340d74af19dfd1786a0"
BUILDER_SHA256 = "2ce3e8f1a0d5c134d120cc1720cd14a81e9c417f79516568c33ebb038a035a87"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_rfxs2_source_artifact() -> None:
    assert _sha256(PANEL) == PANEL_SHA256
    assert _sha256(MANIFEST) == MANIFEST_SHA256
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["combined_output"] == str(PANEL)
    assert manifest["combined_sha256"] == PANEL_SHA256
    assert manifest["builder"] == "training/build_binance_regional_fiat_cross_rate.py"
    assert manifest["builder_commit"] == BUILDER_COMMIT
    assert manifest["builder_sha256"] == BUILDER_SHA256
    assert _sha256(Path(manifest["builder"])) == BUILDER_SHA256
    assert manifest["mechanism_bindings"] == [
        builder.ORIGINAL_MECHANISM,
        builder.SOURCE_REJECTION,
        builder.MECHANISM,
    ]
    for binding in manifest["mechanism_bindings"]:
        assert _sha256(Path(binding["path"])) == binding["sha256"]

    assert manifest["config"] == {
        "end": "2024-01-01",
        "output_dir": str(ROOT),
        "retries": 5,
        "start": "2020-11-01",
        "symbols": list(builder.DEFAULT_SYMBOLS),
        "timeout_seconds": 60,
        "workers": 8,
    }
    protocol = manifest["protocol"]
    assert protocol["candidate"] == "RFXS2-576"
    assert protocol["fixture_only"] is False
    assert protocol["source_values_opened"] is True
    assert protocol["execution_ohlc_opened"] is False
    assert protocol["funding_opened"] is False
    assert protocol["outcomes_opened"] is False
    assert protocol["signal_fields_retained"] == ["close"]

    assert manifest["rows"] == manifest["complete_rows"] == 1_156
    assert manifest["expected_rows"] == 1_156
    assert manifest["first_date"] == "2020-11-01"
    assert manifest["last_date"] == "2023-12-31"
    assert manifest["symbols"] == list(builder.DEFAULT_SYMBOLS)
    assert manifest["columns"] == list(builder.OUTPUT_COLUMNS)

    expected_months = pd.period_range("2020-11", "2023-12", freq="M").astype(
        str
    ).tolist()
    assert len(manifest["archives"]) == 152
    assert sum(int(item["rows"]) for item in manifest["archives"]) == 4_624
    for symbol in builder.DEFAULT_SYMBOLS:
        records = [
            item for item in manifest["archives"] if item["symbol"] == symbol
        ]
        assert [item["month"] for item in records] == expected_months
        assert all(item["timestamp_unit"] == "ms" for item in records)
        assert all(
            item["published_archive_sha256"] == item["archive_sha256"]
            for item in records
        )
        assert all(len(item["checksum_response_sha256"]) == 64 for item in records)

    frame = pd.read_csv(PANEL)
    assert tuple(frame.columns) == builder.OUTPUT_COLUMNS
    expected_dates = pd.date_range(
        "2020-11-01", "2024-01-01", freq="1D", inclusive="left"
    )
    dates = pd.to_datetime(frame["date"])
    available = pd.to_datetime(
        frame["source_available_not_before"], utc=True
    ).dt.tz_localize(None)
    assert dates.equals(pd.Series(expected_dates, name="date"))
    assert available.equals(
        pd.Series(
            expected_dates + pd.Timedelta(days=1),
            name="source_available_not_before",
        )
    )
    assert bool(frame["source_complete"].all())
    close_columns = [f"{symbol}_close" for symbol in builder.DEFAULT_SYMBOLS]
    closes = frame[close_columns].to_numpy(float)
    assert np.isfinite(closes).all()
    assert (closes > 0.0).all()
    forbidden = {
        "open",
        "high",
        "low",
        "volume",
        "trade_count",
        "return",
        "residual",
        "zscore",
        "side",
        "pnl",
        "cagr",
        "mdd",
    }
    assert not forbidden.intersection(frame.columns)


def test_rfxs2_source_network_rebuild_attestation() -> None:
    assert _sha256(REBUILD_LOG) == REBUILD_LOG_SHA256
    assert _sha256(REBUILD_ATTESTATION) == REBUILD_ATTESTATION_SHA256
    attestation = json.loads(REBUILD_ATTESTATION.read_text())
    assert attestation["candidate"] == "RFXS2-576"
    assert attestation["builder_commit"] == BUILDER_COMMIT
    assert attestation["builder_sha256"] == BUILDER_SHA256
    assert attestation["first_build"] == {
        "manifest_sha256": MANIFEST_SHA256,
        "panel_sha256": PANEL_SHA256,
    }
    assert attestation["second_build"]["manifest_sha256"] == MANIFEST_SHA256
    assert attestation["second_build"]["panel_sha256"] == PANEL_SHA256
    assert attestation["second_build"]["exit_code"] == 0
    assert attestation["byte_identical"] is True
    assert attestation["second_build_log"] == {
        "lines": 166,
        "path": str(REBUILD_LOG),
        "sha256": REBUILD_LOG_SHA256,
    }
    assert attestation["raw_archives_persisted"] is False
    assert attestation["execution_ohlc_opened"] is False
    assert attestation["funding_opened"] is False
    assert attestation["outcomes_opened"] is False
