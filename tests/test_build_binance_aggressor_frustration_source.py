from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd

from training import build_binance_aggtrade_microstructure as base
from training import build_binance_aggressor_frustration_source as builder


def _archive(rows: list[list[object]]) -> bytes:
    text = io.StringIO()
    pd.DataFrame(rows, columns=base.RAW_COLUMNS).to_csv(text, index=False)
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("BTCUSDT-aggTrades-test.csv", text.getvalue())
    return output.getvalue()


def _row(identifier: int, timestamp: str, price: float, maker: bool) -> list[object]:
    timestamp_ms = int(pd.Timestamp(timestamp, tz="UTC").timestamp() * 1_000)
    return [identifier, price, 1.0, identifier, identifier, timestamp_ms, str(maker).lower()]


class _FakeFetcher:
    def __init__(self, payloads: dict[date, bytes]) -> None:
        self.payloads = payloads

    def __call__(self, url: str, *, retries: int, timeout: int) -> bytes:
        del retries, timeout
        for day, payload in self.payloads.items():
            if day.isoformat() not in url:
                continue
            if url.endswith(".CHECKSUM"):
                digest = hashlib.sha256(payload).hexdigest()
                return f"{digest}  archive.zip\n".encode()
            return payload
        raise FileNotFoundError(url)


def test_month_build_carries_verified_warmup_state(tmp_path: Path) -> None:
    warmup_day = date(2021, 1, 31)
    source_day = date(2021, 2, 1)
    fetcher = _FakeFetcher(
        {
            warmup_day: _archive(
                [
                    _row(10, "2021-01-31 23:59:58", 100.0, False),
                    _row(11, "2021-01-31 23:59:59", 101.0, False),
                ]
            ),
            source_day: _archive(
                [_row(12, "2021-02-01 00:00:00", 101.0, True)]
            ),
        }
    )
    cfg = builder.BuildConfig(
        start="2021-02-01",
        end="2021-02-02",
        output_dir=str(tmp_path),
        workers=1,
    )
    metadata = builder._process_month(date(2021, 2, 1), cfg, fetcher=fetcher)
    output = pd.read_csv(metadata["output"], compression="gzip")

    assert metadata["warmup"]["status"] == "verified"
    assert metadata["warmup"]["state_out"]["last_nonzero_tick"] == 1
    assert metadata["archives"][0]["state_in"]["previous_agg_trade_id"] == 11
    assert output.loc[0, "state_reset_count"] == 0
    assert output.loc[0, "carried_sell_frustrated_notional"] == 101.0
    assert output.loc[0, "frustration_score"] == 1.0


def test_missing_warmup_fails_closed_to_unavailable_tick(tmp_path: Path) -> None:
    source_day = date(2021, 2, 1)
    fetcher = _FakeFetcher(
        {source_day: _archive([_row(12, "2021-02-01 00:00:00", 101.0, True)])}
    )
    cfg = builder.BuildConfig(
        start="2021-02-01",
        end="2021-02-02",
        output_dir=str(tmp_path),
        workers=1,
    )
    metadata = builder._process_month(date(2021, 2, 1), cfg, fetcher=fetcher)
    output = pd.read_csv(metadata["output"], compression="gzip")
    assert metadata["warmup"]["status"] == "unavailable"
    assert output.loc[0, "state_reset_count"] == 1
    assert output.loc[0, "unavailable_tick_count"] == 1
    assert output.loc[0, "frustration_score"] == 0.0


def test_resume_rechecks_warmup_checksum_and_rebuilds(tmp_path: Path) -> None:
    warmup_day = date(2021, 1, 31)
    source_day = date(2021, 2, 1)
    fetcher = _FakeFetcher(
        {
            warmup_day: _archive(
                [
                    _row(10, "2021-01-31 23:59:58", 100.0, False),
                    _row(11, "2021-01-31 23:59:59", 101.0, False),
                ]
            ),
            source_day: _archive(
                [_row(12, "2021-02-01 00:00:00", 101.0, True)]
            ),
        }
    )
    cfg = builder.BuildConfig(
        start="2021-02-01",
        end="2021-02-02",
        output_dir=str(tmp_path),
        workers=1,
    )
    first = builder._process_month(date(2021, 2, 1), cfg, fetcher=fetcher)
    first_hash = first["output_sha256"]

    fetcher.payloads[warmup_day] = _archive(
        [
            _row(10, "2021-01-31 23:59:58", 103.0, False),
            _row(11, "2021-01-31 23:59:59", 102.0, False),
        ]
    )
    second = builder._process_month(date(2021, 2, 1), cfg, fetcher=fetcher)
    output = pd.read_csv(second["output"], compression="gzip")
    assert second["output_sha256"] != first_hash
    assert output.loc[0, "frustration_score"] == 0.0


def test_build_manifest_is_outcome_blind_and_deterministic(tmp_path: Path) -> None:
    warmup_day = date(2021, 1, 31)
    source_day = date(2021, 2, 1)
    fetcher = _FakeFetcher(
        {
            warmup_day: _archive(
                [
                    _row(10, "2021-01-31 23:59:58", 100.0, False),
                    _row(11, "2021-01-31 23:59:59", 101.0, False),
                ]
            ),
            source_day: _archive(
                [_row(12, "2021-02-01 00:00:00", 101.0, True)]
            ),
        }
    )
    cfg = builder.BuildConfig(
        start="2021-02-01",
        end="2021-02-02",
        output_dir=str(tmp_path),
        workers=1,
    )

    first = builder.build(cfg, fetcher=fetcher)
    second = builder.build(cfg, fetcher=fetcher)

    assert first["protocol"]["outcomes_opened"] is False
    assert first["combined_sha256"] == second["combined_sha256"]
    assert first["columns"] == list(builder.BAR_COLUMNS)
    manifest = json.loads((tmp_path / "build_manifest.json").read_text())
    assert manifest["protocol"]["raw_archives_persisted"] is False
