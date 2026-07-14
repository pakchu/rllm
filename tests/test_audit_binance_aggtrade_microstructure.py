from __future__ import annotations

import json
import shutil
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from training import audit_binance_aggtrade_microstructure as audit
from training import build_binance_aggtrade_microstructure as builder


def _fixture(tmp_path: Path) -> audit.AuditConfig:
    start = pd.Timestamp("2021-01-01")
    timestamps = pd.date_range(start, start + pd.Timedelta("2d"), inclusive="left", freq="5min")
    prices = 100.0 + np.arange(len(timestamps)) * 0.01
    raw = pd.DataFrame(
        {
            "agg_trade_id": np.arange(1, len(timestamps) + 1),
            "price": prices,
            "quantity": np.ones(len(timestamps)),
            "first_trade_id": np.arange(1_001, 1_001 + len(timestamps)),
            "last_trade_id": np.arange(1_001, 1_001 + len(timestamps)),
            "transact_time": timestamps.astype("int64") // 1_000_000 + 1_000,
            "is_buyer_maker": np.zeros(len(timestamps), dtype=bool),
        }
    )
    features = builder.aggregate_five_minute(raw)
    features_path = tmp_path / "features.csv.gz"
    builder._write_gzip_csv(features, features_path)

    monthly_dir = tmp_path / "monthly"
    monthly_dir.mkdir()
    monthly_path = monthly_dir / "BTCUSDT_aggtrade_5m_2021-01.csv.gz"
    shutil.copyfile(features_path, monthly_path)
    monthly_hash = audit._sha256(monthly_path)
    archives = [
        {
            "date": "2021-01-01",
            "archive_sha256": "0" * 64,
            "agg_trade_rows": 288,
            "five_minute_rows": 288,
            "first_agg_trade_id": 1,
            "last_agg_trade_id": 288,
            "first_underlying_trade_id": 1_001,
            "last_underlying_trade_id": 1_288,
        },
        {
            "date": "2021-01-02",
            "archive_sha256": "1" * 64,
            "agg_trade_rows": 288,
            "five_minute_rows": 288,
            "first_agg_trade_id": 289,
            "last_agg_trade_id": 576,
            "first_underlying_trade_id": 1_289,
            "last_underlying_trade_id": 1_576,
        },
    ]
    manifest = {
        "config": {"symbol": "BTCUSDT", "start": "2021-01-01", "end": "2021-01-03"},
        "protocol": {"outcomes_opened": False},
        "combined_output": str(features_path),
        "combined_sha256": audit._sha256(features_path),
        "rows": len(features),
        "columns": list(features.columns),
        "months": [
            {
                "schema_version": 1,
                "month": "2021-01",
                "requested_dates": ["2021-01-01", "2021-01-02"],
                "output": str(monthly_path),
                "output_sha256": monthly_hash,
                "archives": archives,
            }
        ],
    }
    manifest_path = tmp_path / "build_manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    market = pd.DataFrame(
        {
            "date": timestamps,
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": np.ones(len(timestamps)),
            "quote_asset_volume": prices,
            "number_of_trades": np.ones(len(timestamps)),
            "taker_buy_quote": prices,
        }
    )
    market_path = tmp_path / "market.csv.gz"
    market.to_csv(market_path, index=False, compression="gzip")
    return audit.AuditConfig(
        features=str(features_path),
        manifest=str(manifest_path),
        market=str(market_path),
        start="2021-01-01",
        end="2021-01-03",
        output=str(tmp_path / "audit.json"),
    )


def test_exact_fixture_passes_all_structural_and_reconciliation_checks(tmp_path: Path) -> None:
    result = audit.run_audit(_fixture(tmp_path))
    assert result["passed"] is True
    assert result["failed_checks"] == []
    assert result["feature_diagnostics"]["rows"] == 576
    assert result["reconciliation"]["daily"]["base_volume"]["max_relative_error"] == 0.0


def test_missing_feature_bin_fails_closed(tmp_path: Path) -> None:
    cfg = _fixture(tmp_path)
    features_path = Path(cfg.features)
    frame = pd.read_csv(features_path, compression="gzip", parse_dates=["date"]).iloc[:-1]
    builder._write_gzip_csv(frame, features_path)
    manifest_path = Path(cfg.manifest)
    manifest = json.loads(manifest_path.read_text())
    manifest["combined_sha256"] = audit._sha256(features_path)
    manifest_path.write_text(json.dumps(manifest))

    result = audit.run_audit(cfg)
    assert result["passed"] is False
    assert result["checks"]["coverage.feature_index_is_expected_subset"] is True
    assert result["checks"]["coverage.feature_missing_bins_documented"] is False


def test_signed_partition_corruption_is_detected(tmp_path: Path) -> None:
    cfg = _fixture(tmp_path)
    features_path = Path(cfg.features)
    frame = pd.read_csv(features_path, compression="gzip", parse_dates=["date"])
    frame.loc[0, "signed_quote_notional"] += 1.0
    builder._write_gzip_csv(frame, features_path)
    manifest_path = Path(cfg.manifest)
    manifest = json.loads(manifest_path.read_text())
    manifest["combined_sha256"] = audit._sha256(features_path)
    manifest_path.write_text(json.dumps(manifest))

    result = audit.run_audit(cfg)
    assert result["passed"] is False
    assert result["checks"]["feature.signed_partition"] is False
    assert result["checks"]["feature.flow_coherence_identity"] is False


def test_documented_source_gap_day_is_quarantined(tmp_path: Path) -> None:
    cfg = replace(_fixture(tmp_path), maximum_missing_bin_fraction=0.01)
    features_path = Path(cfg.features)
    frame = pd.read_csv(features_path, compression="gzip", parse_dates=["date"])
    frame = frame.drop(index=10).reset_index(drop=True)
    builder._write_gzip_csv(frame, features_path)

    manifest_path = Path(cfg.manifest)
    manifest = json.loads(manifest_path.read_text())
    monthly_path = Path(manifest["months"][0]["output"])
    builder._write_gzip_csv(frame, monthly_path)
    manifest["combined_sha256"] = audit._sha256(features_path)
    manifest["rows"] = len(frame)
    manifest["months"][0]["output_sha256"] = audit._sha256(monthly_path)
    manifest["months"][0]["archives"][0]["agg_trade_rows"] = 287
    manifest["months"][0]["archives"][0]["five_minute_rows"] = 287
    manifest_path.write_text(json.dumps(manifest))

    result = audit.run_audit(cfg)
    assert result["passed"] is True
    assert result["quarantine"]["source_gap_days"] == ["2021-01-01"]
    assert result["quarantine"]["missing_bin_count"] == 1
    assert result["quarantine"]["source_gap_missing_bin_count"] == 1
