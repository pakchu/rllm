from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from training import audit_binance_cross_venue_minute_leadership as audit
from training import build_binance_cross_venue_minute_leadership as builder


SPOT_HEADER_ALIASES = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "ignore",
]
UM_HEADER_ALIASES = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_gzip_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(
        path,
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
        float_format="%.12g",
    )


def _kline_row(
    timestamp: str,
    *,
    open_price: float = 100.0,
    close_price: float | None = None,
    quote_notional: float = 1_000.0,
    flow_frac: float = 0.2,
    trade_count: int = 10,
) -> list[object]:
    open_time = int(pd.Timestamp(timestamp, tz="UTC").timestamp() * 1_000)
    close = open_price if close_price is None else close_price
    high = max(open_price, close) * 1.001
    low = min(open_price, close) * 0.999
    base_volume = quote_notional / ((open_price + close) / 2.0)
    taker_buy_quote = quote_notional * (1.0 + flow_frac) / 2.0
    taker_buy_base = base_volume * (1.0 + flow_frac) / 2.0
    return [
        open_time,
        open_price,
        high,
        low,
        close,
        base_volume,
        open_time + 59_999,
        quote_notional,
        trade_count,
        taker_buy_base,
        taker_buy_quote,
        0.0,
    ]


def _archive(
    rows: list[list[object]],
    *,
    header: bool = True,
    columns: Iterable[str] | None = None,
    member: str = "BTCUSDT-1m-test.csv",
) -> bytes:
    text = io.StringIO()
    pd.DataFrame(rows, columns=list(columns or builder.RAW_COLUMNS)).to_csv(
        text,
        index=False,
        header=header,
    )
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, text.getvalue())
    return output.getvalue()


def _venue_rows(
    *,
    start: str = "2023-01-01 00:00:00",
    base_price: float = 100.0,
    quote_scale: float = 1.0,
    flows: tuple[float, ...] = (0.9, 0.7, 0.8, 0.6, 0.4, 0.6, 0.8, 0.7, 0.9, 0.5),
    returns: tuple[float, ...] = (
        0.0, 0.001, 0.001, 0.001, 0.001, -0.001, -0.001, -0.001, -0.001, -0.001
    ),
) -> list[list[object]]:
    rows: list[list[object]] = []
    open_price = base_price
    for minute, (flow, ret) in enumerate(zip(flows, returns, strict=True)):
        close = open_price * float(np.exp(ret))
        timestamp = pd.Timestamp(start) + pd.Timedelta(minutes=minute)
        rows.append(
            _kline_row(
                str(timestamp),
                open_price=open_price,
                close_price=close,
                quote_notional=(1_000.0 + 10.0 * minute) * quote_scale,
                flow_frac=flow,
            )
        )
        open_price = close
    return rows


def _two_bar_features() -> pd.DataFrame:
    spot = builder.read_archive(
        _archive(_venue_rows(), columns=SPOT_HEADER_ALIASES),
        venue="spot",
    )
    um = builder.read_archive(
        _archive(
            _venue_rows(
                base_price=101.0,
                quote_scale=1.25,
                flows=(0.1, 0.2, 0.3, 0.2, 0.1, -0.2, -0.3, -0.2, -0.1, -0.2),
                returns=(
                    0.0, 0.004, 0.003, 0.002, 0.001,
                    -0.004, -0.003, -0.002, -0.001, -0.001,
                ),
            ),
            columns=UM_HEADER_ALIASES,
        ),
        venue="um",
    )
    features = builder.aggregate_cross_venue_five_minute(
        spot,
        um,
        expected_minutes=pd.date_range("2023-01-01 00:00:00", periods=10, freq="1min"),
    )
    assert len(features) == 2
    assert features["source_complete"].all()
    assert features["cross_venue_feature_valid"].all()
    return features


def _fixture(tmp_path: Path) -> audit.AuditConfig:
    features = _two_bar_features()
    features_path = (
        tmp_path / "BTCUSDT_cross_venue_minute_leadership_5m_2023-01_2023-01.csv.gz"
    )
    monthly_dir = tmp_path / "monthly"
    monthly_dir.mkdir()
    monthly_path = (
        monthly_dir / "BTCUSDT_cross_venue_minute_leadership_5m_2023-01.csv.gz"
    )
    _write_gzip_csv(features, features_path)
    _write_gzip_csv(features, monthly_path)

    manifest = {
        "as_of": "2026-07-14T00:00:00+00:00",
        "config": {
            "symbol": "BTCUSDT",
            "start": "2023-01-01",
            "end": "2023-01-01 00:10:00",
            "output_dir": str(tmp_path),
            "workers": 1,
            "retries": 5,
            "timeout_seconds": 60,
            "overwrite": False,
        },
        "protocol": {
            "archive_checksums_verified": True,
            "end_is_exclusive": True,
            "join_key": "exact UTC one-minute open_time",
            "feature_available_time": "five-minute bar open time plus five minutes",
            "raw_archives_persisted": False,
            "outcomes_opened": False,
            "sealed_end_exclusive": builder.SEALED_END_EXCLUSIVE.isoformat(),
        },
        "combined_output": str(features_path),
        "combined_sha256": _sha256(features_path),
        "rows": 2,
        "source_complete_rows": 2,
        "feature_valid_rows": 2,
        "quarantined_rows": 0,
        "coverage_by_year": {
            "2023": {
                "expected_rows": 2,
                "source_complete_rows": 2,
                "feature_valid_rows": 2,
                "quarantined_rows": 0,
            }
        },
        "first_date": "2023-01-01 00:00:00",
        "last_date": "2023-01-01 00:05:00",
        "columns": list(features.columns),
        "months": [
            {
                "schema_version": builder.SCHEMA_VERSION,
                "month": "2023-01",
                "symbol": "BTCUSDT",
                "spot_archive_sha256": "a" * 64,
                "um_archive_sha256": "b" * 64,
                "spot_raw_rows": 10,
                "um_raw_rows": 10,
                "rows": 2,
                "source_complete_rows": 2,
                "feature_valid_rows": 2,
                "first_date": "2023-01-01 00:00:00",
                "last_date": "2023-01-01 00:05:00",
                "output": str(monthly_path),
                "output_sha256": _sha256(monthly_path),
            }
        ],
    }
    manifest_path = tmp_path / "build_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    return audit.AuditConfig(
        features=str(features_path),
        manifest=str(manifest_path),
        start="2023-01-01 00:00:00",
        end="2023-01-01 00:10:00",
        output=str(tmp_path / "audit.json"),
        minimum_source_complete_fraction=1.0,
        minimum_feature_valid_fraction=1.0,
    )


def _rewrite_features_and_refresh_manifest(cfg: audit.AuditConfig, frame: pd.DataFrame) -> None:
    features_path = Path(cfg.features)
    manifest_path = Path(cfg.manifest)
    _write_gzip_csv(frame, features_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["combined_sha256"] = _sha256(features_path)
    manifest["rows"] = len(frame)
    manifest["source_complete_rows"] = int(frame["source_complete"].astype(bool).sum())
    manifest["feature_valid_rows"] = int(
        frame["cross_venue_feature_valid"].astype(bool).sum()
    )
    manifest["quarantined_rows"] = int(
        (~frame["cross_venue_feature_valid"].astype(bool)).sum()
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def _assert_failed_check_contains(result: dict[str, object], fragment: str) -> None:
    failed = [str(check) for check in result["failed_checks"]]
    assert any(fragment in check for check in failed), failed
    checks = result["checks"]
    assert any(fragment in str(name) and passed is False for name, passed in checks.items())


def test_exact_fixture_passes_all_cross_venue_integrity_checks(tmp_path: Path) -> None:
    result = audit.run_audit(_fixture(tmp_path))

    assert set(result) >= {"passed", "failed_checks", "checks", "diagnostics"}
    assert result["passed"] is True
    assert result["failed_checks"] == []
    assert result["diagnostics"]["rows"] == 2
    assert result["diagnostics"]["source_complete_fraction"] == 1.0
    assert result["diagnostics"]["feature_valid_fraction"] == 1.0
    assert all(result["checks"].values())


def test_combined_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    cfg = _fixture(tmp_path)
    manifest_path = Path(cfg.manifest)
    manifest = json.loads(manifest_path.read_text())
    manifest["combined_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    result = audit.run_audit(cfg)

    assert result["passed"] is False
    _assert_failed_check_contains(result, "combined_hash")


def test_feature_availability_shifted_early_fails_causal_timing_check(tmp_path: Path) -> None:
    cfg = _fixture(tmp_path)
    frame = pd.read_csv(
        cfg.features,
        compression="gzip",
        parse_dates=["date", "feature_available_time_utc", "trade_earliest_time_utc"],
    )
    frame.loc[0, "feature_available_time_utc"] = frame.loc[0, "date"] + pd.Timedelta(minutes=4)
    frame.loc[0, "trade_earliest_time_utc"] = frame.loc[0, "feature_available_time_utc"]
    _rewrite_features_and_refresh_manifest(cfg, frame)

    result = audit.run_audit(cfg)

    assert result["passed"] is False
    _assert_failed_check_contains(result, "feature_avail")


def test_valid_feature_corruption_to_nonfinite_fails(tmp_path: Path) -> None:
    cfg = _fixture(tmp_path)
    frame = pd.read_csv(
        cfg.features,
        compression="gzip",
        parse_dates=["date", "feature_available_time_utc", "trade_earliest_time_utc"],
    )
    frame.loc[0, "flow_transfer_asymmetry"] = np.inf
    _rewrite_features_and_refresh_manifest(cfg, frame)

    result = audit.run_audit(cfg)

    assert result["passed"] is False
    _assert_failed_check_contains(result, "finite")


def test_invalid_reason_and_validity_inconsistency_fails_even_when_fraction_allowed(
    tmp_path: Path,
) -> None:
    cfg = replace(_fixture(tmp_path), minimum_feature_valid_fraction=0.0)
    frame = pd.read_csv(
        cfg.features,
        compression="gzip",
        parse_dates=["date", "feature_available_time_utc", "trade_earliest_time_utc"],
    )
    frame.loc[0, "cross_venue_feature_valid"] = False
    frame.loc[0, "feature_invalid_reason"] = "ok"
    _rewrite_features_and_refresh_manifest(cfg, frame)

    result = audit.run_audit(cfg)

    assert result["passed"] is False
    _assert_failed_check_contains(result, "reason")
