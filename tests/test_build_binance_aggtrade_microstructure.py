from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import build_binance_aggtrade_microstructure as builder


def _archive(rows: list[list[object]], *, header: bool) -> bytes:
    text = io.StringIO()
    pd.DataFrame(rows, columns=builder.RAW_COLUMNS).to_csv(text, index=False, header=header)
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("BTCUSDT-aggTrades-test.csv", text.getvalue())
    return output.getvalue()


def _daily_archive(day: date, price: float) -> bytes:
    timestamp_ms = int(pd.Timestamp(day, tz="UTC").timestamp() * 1000)
    identifier = day.toordinal()
    return _archive(
        [[identifier, price, 1.0, identifier, identifier, timestamp_ms, "false"]],
        header=True,
    )


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
        raise AssertionError(f"unexpected URL: {url}")


@pytest.mark.parametrize("header", [False, True])
def test_read_archive_supports_historical_header_transition(header: bool) -> None:
    payload = _archive(
        [[1, 100.0, 2.0, 10, 11, 1_609_459_200_000, "false"]],
        header=header,
    )
    frame = builder.read_archive(payload)
    assert frame.loc[0, "agg_trade_id"] == 1
    assert frame.loc[0, "is_buyer_maker"] == np.bool_(False)


def test_checksum_parser_and_verifier() -> None:
    payload = b"payload"
    digest = hashlib.sha256(payload).hexdigest()
    assert builder.expected_sha256(f"{digest}  file.zip\n".encode()) == digest
    assert builder.verify_sha256(payload, digest) == digest
    with pytest.raises(ValueError, match="checksum mismatch"):
        builder.verify_sha256(payload, "0" * 64)


def test_read_archive_rejects_duplicate_ids_and_non_finite_values() -> None:
    duplicate = _archive(
        [
            [1, 100.0, 1.0, 10, 10, 1_609_459_200_000, "false"],
            [1, 101.0, 1.0, 11, 11, 1_609_459_201_000, "true"],
        ],
        header=True,
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        builder.read_archive(duplicate)

    non_finite = _archive(
        [[1, np.inf, 1.0, 10, 10, 1_609_459_200_000, "false"]],
        header=True,
    )
    with pytest.raises(ValueError, match="non-finite"):
        builder.read_archive(non_finite)


def test_archive_url_is_official_usdm_daily_path() -> None:
    assert builder.archive_url("BTCUSDT", date(2023, 1, 2)) == (
        "https://data.binance.vision/data/futures/um/daily/aggTrades/"
        "BTCUSDT/BTCUSDT-aggTrades-2023-01-02.zip"
    )


def test_five_minute_aggregation_preserves_sequence_and_trade_counts() -> None:
    frame = pd.DataFrame(
        [
            [1, 100.0, 1.0, 10, 11, 1_609_459_200_000, False],
            [2, 101.0, 2.0, 12, 12, 1_609_459_201_000, False],
            [3, 99.0, 1.0, 13, 15, 1_609_459_203_000, True],
            [4, 102.0, 1.0, 16, 16, 1_609_459_500_000, True],
        ],
        columns=builder.RAW_COLUMNS,
    )
    output = builder.aggregate_five_minute(frame)
    assert len(output) == 2
    first = output.iloc[0]
    assert first["date"] == pd.Timestamp("2021-01-01 00:00:00")
    assert first["agg_trade_count"] == 3
    assert first["underlying_trade_count"] == 6
    assert np.isclose(first["base_volume"], 4.0)
    assert np.isclose(first["quote_notional"], 401.0)
    assert np.isclose(first["buy_quote_notional"], 302.0)
    assert np.isclose(first["sell_quote_notional"], 99.0)
    assert np.isclose(first["signed_quote_notional"], 203.0)
    assert np.isclose(first["sign_flip_rate"], 0.5)
    assert np.isclose(first["mean_same_sign_run_length"], 1.5)
    assert np.isclose(first["max_same_sign_run_share"], 2.0 / 3.0)
    assert np.isclose(first["interarrival_mean_ms"], 1500.0)
    assert np.isclose(first["micro_log_return"], np.log(99.0 / 100.0))
    assert first["signed_price_response"] < 0.0
    second = output.iloc[1]
    assert second["interarrival_mean_ms"] == 0.0
    assert second["interarrival_std_ms"] == 0.0
    assert second["interarrival_burstiness"] == 0.0
    assert np.isfinite(second["buy_sell_event_size_log_ratio"])
    assert tuple(output.columns) == builder.OUTPUT_COLUMNS


def test_month_boundaries_are_exclusive() -> None:
    days = builder._month_days(date(2023, 1, 1), date(2023, 1, 15), date(2023, 2, 1))
    assert days[0] == date(2023, 1, 15)
    assert days[-1] == date(2023, 1, 31)
    assert len(days) == 17


def test_partial_month_resume_is_invalidated(tmp_path: Path) -> None:
    first_day = date(2021, 1, 1)
    second_day = date(2021, 1, 2)
    fetcher = _FakeFetcher(
        {
            first_day: _daily_archive(first_day, 100.0),
            second_day: _daily_archive(second_day, 200.0),
        }
    )
    first_cfg = builder.BuildConfig(
        start="2021-01-01",
        end="2021-01-02",
        output_dir=str(tmp_path),
        workers=1,
    )
    builder._process_month(first_day, first_cfg, fetcher=fetcher)

    second_cfg = builder.BuildConfig(
        start="2021-01-02",
        end="2021-01-03",
        output_dir=str(tmp_path),
        workers=1,
    )
    metadata = builder._process_month(first_day, second_cfg, fetcher=fetcher)
    output = pd.read_csv(metadata["output"], compression="gzip")
    assert metadata["requested_dates"] == ["2021-01-02"]
    assert output.loc[0, "date"] == "2021-01-02"
    assert output.loc[0, "last_price"] == 200.0


def test_resume_rechecks_upstream_checksum_and_rebuilds(tmp_path: Path) -> None:
    day = date(2021, 1, 1)
    fetcher = _FakeFetcher({day: _daily_archive(day, 100.0)})
    cfg = builder.BuildConfig(
        start="2021-01-01",
        end="2021-01-02",
        output_dir=str(tmp_path),
        workers=1,
    )
    first = builder._process_month(day, cfg, fetcher=fetcher)
    first_hash = first["output_sha256"]

    fetcher.payloads[day] = _daily_archive(day, 101.0)
    second = builder._process_month(day, cfg, fetcher=fetcher)
    output = pd.read_csv(second["output"], compression="gzip")
    assert second["output_sha256"] != first_hash
    assert output.loc[0, "last_price"] == 101.0


def test_overwrite_produces_deterministic_gzip_hash(tmp_path: Path) -> None:
    day = date(2021, 1, 1)
    fetcher = _FakeFetcher({day: _daily_archive(day, 100.0)})
    cfg = builder.BuildConfig(
        start="2021-01-01",
        end="2021-01-02",
        output_dir=str(tmp_path),
        workers=1,
        overwrite=True,
    )
    first_hash = builder._process_month(day, cfg, fetcher=fetcher)["output_sha256"]
    second_hash = builder._process_month(day, cfg, fetcher=fetcher)["output_sha256"]
    assert first_hash == second_hash

    metadata_path = next((tmp_path / "monthly").glob("*.json"))
    assert json.loads(metadata_path.read_text())["schema_version"] == builder.SCHEMA_VERSION
