from __future__ import annotations

import hashlib
import io
import zipfile

import pandas as pd
import pytest

from training import build_binance_coinm_quarterly_strip as strip


def _archive(symbol: str, rows: list[list[object]], *, header: bool = True) -> bytes:
    frame = pd.DataFrame(rows, columns=strip.ARCHIVE_COLUMNS)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            f"{symbol}-5m-2023-01.csv",
            frame.to_csv(index=False, header=header),
        )
    return buffer.getvalue()


def _row(timestamp: str, close: float = 100.0, volume: float = 10.0):
    opened = int(pd.Timestamp(timestamp, tz="UTC").timestamp() * 1_000)
    return [
        opened,
        close,
        close + 1.0,
        close - 1.0,
        close,
        volume,
        opened + 299_999,
        volume / close,
        5,
        volume / 2.0,
        volume / close / 2.0,
        0,
    ]


def test_contract_delivery_is_utc_0800_and_rejects_perp() -> None:
    assert strip.contract_delivery("BTCUSD_231229") == pd.Timestamp("2023-12-29 08:00")
    with pytest.raises(ValueError):
        strip.contract_delivery("BTCUSD_PERP")


def test_monthly_archive_url_is_contract_specific() -> None:
    assert strip.monthly_archive_url("BTCUSD_240329", "2023-12") == (
        "https://data.binance.vision/data/futures/cm/monthly/klines/"
        "BTCUSD_240329/5m/BTCUSD_240329-5m-2023-12.zip"
    )
    with pytest.raises(ValueError, match="archive month"):
        strip.monthly_archive_url("BTCUSD_240329", "2023-12-01")


def test_archive_parser_accepts_header_and_headerless_rows() -> None:
    symbol = "BTCUSD_231229"
    rows = [_row("2023-01-01 00:00"), _row("2023-01-01 00:05", close=101.0)]
    for header in (True, False):
        parsed = strip.read_archive(
            _archive(symbol, rows, header=header),
            symbol=symbol,
            start=pd.Timestamp("2023-01-01"),
            end=pd.Timestamp("2023-01-02"),
        )
        assert len(parsed) == 2
        assert parsed["row_valid"].all()
        assert parsed.loc[0, "feature_available_time_utc"] == pd.Timestamp("2023-01-01 00:05")


def test_archive_parser_quarantines_invalid_ohlc() -> None:
    symbol = "BTCUSD_231229"
    row = _row("2023-01-01 00:00")
    row[2] = 99.0
    parsed = strip.read_archive(
        _archive(symbol, [row]),
        symbol=symbol,
        start=pd.Timestamp("2023-01-01"),
        end=pd.Timestamp("2023-01-02"),
    )
    assert not parsed.loc[0, "row_valid"]


@pytest.mark.parametrize(
    ("mutate", "column"),
    [
        (lambda row: row.__setitem__(6, row[0] + 300_000), "close_time"),
        (lambda row: row.__setitem__(5, -1.0), "negative_volume"),
        (lambda row: row.__setitem__(9, row[5] + 1.0), "taker_exceeds_total"),
    ],
)
def test_archive_parser_quarantines_timing_and_flow_errors(mutate, column) -> None:
    symbol = "BTCUSD_231229"
    row = _row("2023-01-01 00:00")
    mutate(row)
    parsed = strip.read_archive(
        _archive(symbol, [row]),
        symbol=symbol,
        start=pd.Timestamp("2023-01-01"),
        end=pd.Timestamp("2023-01-02"),
    )
    assert not parsed.loc[0, "row_valid"], column


def test_fixed_calendar_does_not_promote_next_when_front_is_missing(monkeypatch) -> None:
    contracts = ("BTCUSD_230331", "BTCUSD_230630", "BTCUSD_230929")
    start = pd.Timestamp("2023-01-01")
    end = start + pd.Timedelta("10min")
    rows = []
    for symbol in contracts[1:]:
        for timestamp in pd.date_range(start, end, freq="5min", inclusive="left"):
            rows.append(
                {
                    "date": timestamp,
                    "feature_available_time_utc": timestamp + pd.Timedelta("5min"),
                    "symbol": symbol,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": 10.0,
                    "base_asset_volume": 0.1,
                    "count": 5,
                    "taker_buy_volume": 5.0,
                    "taker_buy_base_asset_volume": 0.05,
                    "row_valid": True,
                }
            )
    panel = strip.build_fixed_strip(
        pd.DataFrame(rows), start=start, end=end, contracts=contracts
    )
    assert panel["front_symbol"].eq("BTCUSD_230331").all()
    assert panel["next_symbol"].eq("BTCUSD_230630").all()
    assert not panel["feature_valid"].any()
    assert panel["feature_invalid_reason"].eq("front_row_missing").all()


def test_fixed_calendar_uses_next_five_minute_availability() -> None:
    contracts = ("BTCUSD_230331", "BTCUSD_230630", "BTCUSD_230929")
    start = pd.Timestamp("2023-01-01")
    rows = []
    for symbol in contracts[:2]:
        rows.append(
            {
                "date": start,
                "feature_available_time_utc": start + pd.Timedelta("5min"),
                "symbol": symbol,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 10.0,
                "base_asset_volume": 0.1,
                "count": 5,
                "taker_buy_volume": 5.0,
                "taker_buy_base_asset_volume": 0.05,
                "row_valid": True,
            }
        )
    panel = strip.build_fixed_strip(
        pd.DataFrame(rows),
        start=start,
        end=start + pd.Timedelta("5min"),
        contracts=contracts,
    )
    assert panel.loc[0, "feature_valid"]
    assert panel.loc[0, "trade_earliest_time_utc"] == start + pd.Timedelta("5min")


def test_delivery_rollover_uses_availability_not_bar_open() -> None:
    contracts = ("BTCUSD_230331", "BTCUSD_230630", "BTCUSD_230929")
    start = pd.Timestamp("2023-03-31 07:50")
    raw_rows = []
    for timestamp in pd.date_range(start, periods=3, freq="5min"):
        for symbol in contracts:
            raw_rows.append(
                {
                    "date": timestamp,
                    "feature_available_time_utc": timestamp + pd.Timedelta("5min"),
                    "symbol": symbol,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": 10.0,
                    "base_asset_volume": 0.1,
                    "count": 5,
                    "taker_buy_volume": 5.0,
                    "taker_buy_base_asset_volume": 0.05,
                    "row_valid": True,
                }
            )
    panel = strip.build_fixed_strip(
        pd.DataFrame(raw_rows),
        start=start,
        end=start + pd.Timedelta("15min"),
        contracts=contracts,
    )
    assert panel["front_symbol"].tolist() == [
        "BTCUSD_230331",
        "BTCUSD_230630",
        "BTCUSD_230630",
    ]
    assert panel["feature_available_time_utc"].tolist() == list(
        pd.date_range("2023-03-31 07:55", periods=3, freq="5min")
    )


def test_default_causal_end_keeps_availability_before_2024() -> None:
    contracts = ("BTCUSD_240329", "BTCUSD_240628", "BTCUSD_240927")
    start = pd.Timestamp("2023-12-31 23:45")
    end = pd.Timestamp(strip.BuildConfig.end)
    rows = []
    for timestamp in pd.date_range(start, end, freq="5min", inclusive="left"):
        for symbol in contracts[:2]:
            rows.append(
                {
                    "date": timestamp,
                    "feature_available_time_utc": timestamp + pd.Timedelta("5min"),
                    "symbol": symbol,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": 10.0,
                    "base_asset_volume": 0.1,
                    "count": 5,
                    "taker_buy_volume": 5.0,
                    "taker_buy_base_asset_volume": 0.05,
                    "row_valid": True,
                }
            )
    panel = strip.build_fixed_strip(
        pd.DataFrame(rows), start=start, end=end, contracts=contracts
    )
    assert panel["trade_earliest_time_utc"].max() < pd.Timestamp("2024-01-01")


def test_listing_parser_extracts_only_exact_symbol_days() -> None:
    payload = b'''<?xml version="1.0" encoding="UTF-8"?>
    <ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
      <IsTruncated>false</IsTruncated>
      <Contents><Key>data/futures/cm/daily/klines/BTCUSD_231229/5m/BTCUSD_231229-5m-2023-06-30.zip</Key></Contents>
      <Contents><Key>data/futures/cm/daily/klines/BTCUSD_231229/5m/BTCUSD_231229-5m-2023-06-30.zip.CHECKSUM</Key></Contents>
      <Contents><Key>data/futures/cm/daily/klines/BTCUSD_231229/1m/BTCUSD_231229-1m-2023-06-30.zip</Key></Contents>
    </ListBucketResult>'''
    assert strip.parse_listing(payload, "BTCUSD_231229") == ["2023-06-30"]


def test_cached_fetcher_is_stable(tmp_path) -> None:
    calls = []

    def fetcher(url, *, retries, timeout):
        calls.append((url, retries, timeout))
        return b"payload"

    first = strip.fetch_cached(
        "https://example.test/file.zip",
        cache_dir=tmp_path,
        retries=2,
        timeout=3,
        fetcher=fetcher,
    )
    second = strip.fetch_cached(
        "https://example.test/file.zip",
        cache_dir=tmp_path,
        retries=9,
        timeout=9,
        fetcher=fetcher,
    )
    assert first == second == b"payload"
    assert len(calls) == 1
    assert hashlib.sha256(first).hexdigest() == hashlib.sha256(second).hexdigest()


def test_monthly_fallback_adds_only_missing_daily_keys() -> None:
    symbol = "BTCUSD_231229"
    daily = strip.read_archive(
        _archive(symbol, [_row("2023-01-01 00:00")]),
        symbol=symbol,
        start=pd.Timestamp("2023-01-01"),
        end=pd.Timestamp("2023-01-02"),
    )
    monthly = strip.read_archive(
        _archive(
            symbol,
            [_row("2023-01-01 00:00"), _row("2023-01-01 00:05", close=101.0)],
        ),
        symbol=symbol,
        start=pd.Timestamp("2023-01-01"),
        end=pd.Timestamp("2023-01-02"),
    )
    combined, added, diagnostics = strip.merge_daily_with_monthly_fallback(
        daily, monthly
    )
    assert added == 1
    assert diagnostics["conflict_rows"] == 0
    assert combined["date"].tolist() == list(
        pd.date_range("2023-01-01", periods=2, freq="5min")
    )


def test_monthly_fallback_records_overlap_disagreement_and_keeps_daily() -> None:
    symbol = "BTCUSD_231229"
    daily = strip.read_archive(
        _archive(symbol, [_row("2023-01-01 00:00")]),
        symbol=symbol,
        start=pd.Timestamp("2023-01-01"),
        end=pd.Timestamp("2023-01-02"),
    )
    monthly = daily.copy()
    monthly.loc[0, "close"] += 1.0
    combined, added, diagnostics = strip.merge_daily_with_monthly_fallback(
        daily, monthly
    )
    assert added == 0
    assert diagnostics["conflict_rows"] == 1
    assert diagnostics["conflict_fraction"] == 1.0
    assert len(diagnostics["conflict_sha256"]) == 64
    assert combined.loc[0, "close"] == daily.loc[0, "close"]


def test_download_archive_rejects_checksum_mismatch(tmp_path, monkeypatch) -> None:
    symbol = "BTCUSD_231229"
    payload = _archive(symbol, [_row("2023-01-01 00:00")])

    def fake_fetch(url, **kwargs):
        if url.endswith(".CHECKSUM"):
            return ("0" * 64 + "  file.zip\n").encode()
        return payload

    monkeypatch.setattr(strip, "fetch_cached", fake_fetch)
    cfg = strip.BuildConfig(output_dir=str(tmp_path))
    with pytest.raises(ValueError, match="checksum mismatch"):
        strip._download_archive(symbol, "2023-01-01", cfg, tmp_path)


def test_download_monthly_archive_rejects_checksum_mismatch(
    tmp_path, monkeypatch
) -> None:
    symbol = "BTCUSD_231229"
    payload = _archive(symbol, [_row("2023-12-01 00:00")])

    def fake_fetch(url, **kwargs):
        if url.endswith(".CHECKSUM"):
            return ("0" * 64 + "  file.zip\n").encode()
        return payload

    monkeypatch.setattr(strip, "fetch_cached", fake_fetch)
    cfg = strip.BuildConfig(output_dir=str(tmp_path))
    with pytest.raises(ValueError, match="checksum mismatch"):
        strip._download_monthly_archive(
            symbol, "2023-12", cfg, tmp_path
        )


def test_build_monthly_fallback_covers_both_legs_and_records_provenance(
    tmp_path, monkeypatch
) -> None:
    contracts = ("BTCUSD_230331", "BTCUSD_230630", "BTCUSD_230929")
    monkeypatch.setattr(strip, "CONTRACTS", contracts)
    day = "2023-01-01"
    front_invalid = _row(f"{day} 00:00")
    front_invalid[2] = front_invalid[1] - 1.0
    daily_payloads = {
        contracts[0]: _archive(contracts[0], [front_invalid]),
        contracts[1]: _archive(contracts[1], [_row(f"{day} 00:00")]),
    }
    monthly_payloads = {
        contracts[0]: _archive(
            contracts[0],
            [front_invalid, _row(f"{day} 00:05", close=101.0)],
        ),
        contracts[1]: _archive(
            contracts[1],
            [_row(f"{day} 00:00"), _row(f"{day} 00:05", close=101.0)],
        ),
    }

    def listing(symbol: str, include_day: bool) -> bytes:
        contents = (
            "<Contents><Key>data/futures/cm/daily/klines/"
            f"{symbol}/5m/{symbol}-5m-{day}.zip</Key></Contents>"
            if include_day
            else ""
        )
        return (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
            f"<IsTruncated>false</IsTruncated>{contents}</ListBucketResult>"
        ).encode()

    def fake_fetch(url, **kwargs):
        if url.startswith(strip.S3_ROOT):
            symbol = next(symbol for symbol in contracts if symbol in url)
            return listing(symbol, symbol in daily_payloads)
        checksum = url.endswith(".CHECKSUM")
        archive_url = url.removesuffix(".CHECKSUM")
        symbol = next(symbol for symbol in contracts if symbol in archive_url)
        payload = (
            monthly_payloads[symbol]
            if "/monthly/" in archive_url
            else daily_payloads[symbol]
        )
        if checksum:
            filename = archive_url.rsplit("/", 1)[-1]
            return f"{hashlib.sha256(payload).hexdigest()}  {filename}\n".encode()
        return payload

    monkeypatch.setattr(strip, "fetch_cached", fake_fetch)
    configs = [
        strip.BuildConfig(
            start=f"{day} 00:00",
            end=f"{day} 00:10",
            output_dir=str(tmp_path / name),
            workers=1,
        )
        for name in ("first", "second")
    ]
    reports = [strip.build(cfg) for cfg in configs]
    first = reports[0]
    assert first["monthly_rows_added"] == 2
    assert first["monthly_fallback_requests"] == [
        {"symbol": contracts[0], "month": "2023-01"},
        {"symbol": contracts[1], "month": "2023-01"},
    ]
    assert len(first["monthly_fallback_archives"]) == 2
    assert first["monthly_overlap_diagnostics"]["conflict_rows"] == 0
    assert all(
        len(row["archive_sha256"]) == 64
        for row in first["monthly_fallback_archives"]
    )
    panel = pd.read_csv(first["output"])
    assert panel["feature_valid"].tolist() == [False, True]
    assert panel.loc[0, "feature_invalid_reason"] == "front_row_invalid"
    assert first["output_sha256"] == reports[1]["output_sha256"]


def test_build_rejects_post2023_without_explicit_open(tmp_path) -> None:
    cfg = strip.BuildConfig(
        start="2023-12-01",
        end="2024-01-02",
        output_dir=str(tmp_path),
    )
    with pytest.raises(ValueError, match=r"2024\+"):
        strip.build(cfg)


def test_build_rejects_availability_at_2024_boundary(tmp_path) -> None:
    cfg = strip.BuildConfig(
        start="2023-12-01",
        end="2024-01-01",
        output_dir=str(tmp_path),
    )
    with pytest.raises(ValueError, match=r"2024\+"):
        strip.build(cfg)


def test_build_rejects_subbar_timestamp_alignment(tmp_path) -> None:
    cfg = strip.BuildConfig(
        start="2023-01-01 00:00:01",
        end="2023-01-02",
        output_dir=str(tmp_path),
    )
    with pytest.raises(ValueError, match="five-minute"):
        strip.build(cfg)
