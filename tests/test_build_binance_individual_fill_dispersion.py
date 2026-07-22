from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import date
from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from training import build_binance_individual_fill_dispersion as builder


def _archive(
    rows: list[list[object]],
    *,
    columns: tuple[str, ...],
    member: str,
    header: bool,
) -> bytes:
    text = io.StringIO()
    pd.DataFrame(rows, columns=pd.Index(columns)).to_csv(text, index=False, header=header)
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, text.getvalue())
    return output.getvalue()


def _trade_archive(day: date, first_id: int, *, price: float = 100.0) -> bytes:
    timestamp = int(cast(pd.Timestamp, pd.Timestamp(day, tz="UTC")).timestamp() * 1000)
    rows = [
        [first_id, price, 1.0, round(price, 2), timestamp + 1_000, "false"],
        [first_id + 1, price, 2.0, round(price * 2.0, 2), timestamp + 2_000, "true"],
    ]
    return _archive(
        rows,
        columns=builder.RAW_TRADE_COLUMNS,
        member=f"BTCUSDT-trades-{day}.csv",
        header=False,
    )


def _aggtrade_archive(day: date, first_id: int, *, price: float = 100.0) -> bytes:
    timestamp = int(cast(pd.Timestamp, pd.Timestamp(day, tz="UTC")).timestamp() * 1000)
    rows = [
        [first_id, price, 1.0, first_id, first_id, timestamp + 1_000, "false"],
        [first_id + 1, price, 2.0, first_id + 1, first_id + 1, timestamp + 2_000, "true"],
    ]
    return _archive(
        rows,
        columns=builder.RAW_AGGTRADE_COLUMNS,
        member=f"BTCUSDT-aggTrades-{day}.csv",
        header=False,
    )


class _FakeFetcher:
    def __init__(self, payloads: dict[tuple[date, str], bytes]) -> None:
        self.payloads = payloads

    def __call__(self, url: str, *, retries: int, timeout: int) -> bytes:
        del retries, timeout
        for (day, kind), payload in self.payloads.items():
            if day.isoformat() not in url or f"/{kind}/" not in url:
                continue
            if url.endswith(".CHECKSUM"):
                digest = hashlib.sha256(payload).hexdigest()
                return f"{digest}  archive.zip\n".encode()
            return payload
        raise AssertionError(f"unexpected URL: {url}")


@pytest.mark.parametrize("header", [False, True])
def test_read_trades_archive_supports_header_transition(header: bool) -> None:
    columns = builder.RAW_TRADE_COLUMNS
    payload = _archive(
        [[1, 7_189.43, 0.03, 215.68, 1_577_836_801_481, "true"]],
        columns=columns,
        member="BTCUSDT-trades-test.csv",
        header=header,
    )
    frame = builder.read_trades_archive(payload)
    assert tuple(frame.columns) == columns
    assert int(frame.loc[0, "trade_id"]) == 1
    assert bool(frame.loc[0, "is_buyer_maker"]) is True


@pytest.mark.parametrize("header", [False, True])
def test_read_aggtrades_archive_supports_header_transition(header: bool) -> None:
    columns = builder.RAW_AGGTRADE_COLUMNS
    payload = _archive(
        [[1, 7_189.43, 0.03, 10, 10, 1_577_836_801_481, "false"]],
        columns=columns,
        member="BTCUSDT-aggTrades-test.csv",
        header=header,
    )
    frame = builder.read_aggtrades_archive(payload)
    assert tuple(frame.columns) == columns
    assert int(frame.loc[0, "first_trade_id"]) == 10
    assert bool(frame.loc[0, "is_buyer_maker"]) is False


def test_archive_urls_and_checksum_are_official_and_verified() -> None:
    day = date(2023, 1, 2)
    assert builder.archive_url("BTCUSDT", day, "trades") == (
        "https://data.binance.vision/data/futures/um/daily/trades/"
        "BTCUSDT/BTCUSDT-trades-2023-01-02.zip"
    )
    payload = b"payload"
    digest = hashlib.sha256(payload).hexdigest()
    assert builder.expected_sha256(f"{digest}  file.zip\n".encode()) == digest
    assert builder.verify_sha256(payload, digest) == digest
    with pytest.raises(ValueError, match="checksum mismatch"):
        builder.verify_sha256(payload, "0" * 64)


def test_disk_guard_fails_closed_at_frozen_limit_and_uses_requested_filesystem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert builder.ensure_disk_budget(used_bytes=299 * builder.GIB) == 299 * builder.GIB
    with pytest.raises(RuntimeError, match="disk guard"):
        builder.ensure_disk_budget(used_bytes=300 * builder.GIB)

    observed: list[Path] = []
    real_disk_usage = builder.shutil.disk_usage

    def disk_usage(path: str | Path) -> object:
        observed.append(Path(path))
        return real_disk_usage(path)

    monkeypatch.setattr(builder.shutil, "disk_usage", disk_usage)
    builder.ensure_disk_budget(path=tmp_path, limit_gib=10_000)
    assert observed == [tmp_path]


def test_cross_source_requires_exact_underlying_coverage() -> None:
    day = date(2021, 1, 1)
    trades = builder.read_trades_archive(_trade_archive(day, 10))
    agg = builder.read_aggtrades_archive(_aggtrade_archive(day, 10))
    builder.validate_daily_cross_source(trades, agg)
    agg.loc[1, "first_trade_id"] = 12
    agg.loc[1, "last_trade_id"] = 12
    with pytest.raises(ValueError, match="boundaries disagree|not exactly contiguous"):
        builder.validate_daily_cross_source(trades, agg)


def test_process_month_streams_both_sources_and_writes_complete_grid(
    tmp_path: Path,
) -> None:
    day = date(2021, 1, 1)
    fetcher = _FakeFetcher(
        {
            (day, "trades"): _trade_archive(day, 10),
            (day, "aggTrades"): _aggtrade_archive(day, 10),
        }
    )
    calls: list[tuple[Path, int]] = []

    def guard(*, path: str | Path, limit_gib: int) -> int:
        calls.append((Path(path), limit_gib))
        return 1

    cfg = builder.BuildConfig(
        start="2021-01-01",
        end="2021-01-02",
        output_dir=str(tmp_path),
    )
    metadata = builder._process_month(day, cfg, fetcher=fetcher, disk_guard=guard)
    output = pd.read_csv(metadata["output"], compression="gzip")
    assert len(output) == 288
    assert tuple(output.columns) == builder.OUTPUT_COLUMNS
    assert int(output["fill_count"].sum()) == 2
    assert int(output["agg_event_count"].sum()) == 2
    assert bool(output["source_complete"].astype(bool).all())
    assert calls == [(tmp_path, 300)] * 4
    assert metadata["archives"][0]["first_trade_id"] == 10
    assert metadata["source_protocol_sha256"] == builder.SOURCE_PROTOCOL_SHA256
    assert metadata["implementation_sha256"] == builder.implementation_sha256()


def test_build_compares_reloaded_months_to_utc_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    day = date(2021, 1, 1)
    fetcher = _FakeFetcher(
        {
            (day, "trades"): _trade_archive(day, 10),
            (day, "aggTrades"): _aggtrade_archive(day, 10),
        }
    )
    cfg = builder.BuildConfig(
        start="2021-01-01",
        end="2021-01-02",
        output_dir=str(tmp_path),
    )
    original_process_month = builder._process_month

    def process_month(month: date, config: builder.BuildConfig) -> dict[str, object]:
        return original_process_month(
            month, config, fetcher=fetcher, disk_guard=lambda **_: 1
        )

    monkeypatch.setattr(builder, "_process_month", process_month)
    monkeypatch.setattr(builder, "ensure_disk_budget", lambda **_: 1)
    manifest = builder.build(cfg, allow_partial_source_for_tests=True)

    assert manifest["rows"] == 288
    output = pd.read_csv(
        manifest["combined_output"], compression="gzip", parse_dates=["date"]
    )
    assert str(output["date"].dt.tz) == "UTC"

    manifest_path = tmp_path / "build_manifest.json"
    first_manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    second = builder.build(cfg, allow_partial_source_for_tests=True)
    second_manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert second == manifest
    assert second_manifest_sha == first_manifest_sha
    assert manifest["source_protocol"]["aggregate_control_source"]["kind"] == "aggTrades"


def test_production_build_rejects_nonfrozen_source_range(tmp_path: Path) -> None:
    cfg = builder.BuildConfig(
        start="2021-01-01", end="2021-01-02", output_dir=str(tmp_path)
    )
    with pytest.raises(ValueError, match="frozen to BTCUSDT"):
        builder.build(cfg)


def test_resume_rejects_upstream_checksum_revision(tmp_path: Path) -> None:
    day = date(2021, 1, 1)
    fetcher = _FakeFetcher(
        {
            (day, "trades"): _trade_archive(day, 10, price=100.0),
            (day, "aggTrades"): _aggtrade_archive(day, 10, price=100.0),
        }
    )
    cfg = builder.BuildConfig(
        start="2021-01-01",
        end="2021-01-02",
        output_dir=str(tmp_path),
    )
    builder._process_month(day, cfg, fetcher=fetcher, disk_guard=lambda **_: 1)
    fetcher.payloads[(day, "trades")] = _trade_archive(day, 10, price=101.0)
    fetcher.payloads[(day, "aggTrades")] = _aggtrade_archive(day, 10, price=101.0)
    with pytest.raises(ValueError, match="checksum revision rejects"):
        builder._process_month(day, cfg, fetcher=fetcher, disk_guard=lambda **_: 1)


def test_resume_rejects_semantic_or_implementation_drift(tmp_path: Path) -> None:
    day = date(2021, 1, 1)
    fetcher = _FakeFetcher(
        {
            (day, "trades"): _trade_archive(day, 10),
            (day, "aggTrades"): _aggtrade_archive(day, 10),
        }
    )
    cfg = builder.BuildConfig(
        start="2021-01-01", end="2021-01-02", output_dir=str(tmp_path)
    )
    builder._process_month(day, cfg, fetcher=fetcher, disk_guard=lambda **_: 1)
    metadata_path = next((tmp_path / "monthly").glob("*.json"))
    metadata = json.loads(metadata_path.read_text())
    metadata["source_protocol_sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="source protocol differs"):
        builder._process_month(day, cfg, fetcher=fetcher, disk_guard=lambda **_: 1)


def test_archive_continuity_rejects_cross_day_id_gap() -> None:
    metadata = [
        {
            "archives": [
                {
                    "date": "2021-01-01",
                    "first_trade_id": 10,
                    "last_trade_id": 11,
                    "first_agg_trade_id": 20,
                    "last_agg_trade_id": 21,
                },
                {
                    "date": "2021-01-02",
                    "first_trade_id": 13,
                    "last_trade_id": 14,
                    "first_agg_trade_id": 22,
                    "last_agg_trade_id": 23,
                },
            ]
        }
    ]
    with pytest.raises(ValueError, match="raw trade IDs"):
        builder._validate_archive_continuity(metadata)


def test_resume_metadata_is_json_and_gzip_is_deterministic(tmp_path: Path) -> None:
    day = date(2021, 1, 1)
    fetcher = _FakeFetcher(
        {
            (day, "trades"): _trade_archive(day, 10),
            (day, "aggTrades"): _aggtrade_archive(day, 10),
        }
    )
    cfg = builder.BuildConfig(
        start="2021-01-01",
        end="2021-01-02",
        output_dir=str(tmp_path),
        overwrite=True,
    )
    first = builder._process_month(day, cfg, fetcher=fetcher, disk_guard=lambda **_: 1)
    second = builder._process_month(day, cfg, fetcher=fetcher, disk_guard=lambda **_: 1)
    assert first["output_sha256"] == second["output_sha256"]
    metadata_path = next((tmp_path / "monthly").glob("*.json"))
    assert json.loads(metadata_path.read_text())["schema_version"] == builder.SCHEMA_VERSION
