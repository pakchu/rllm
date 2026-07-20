from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from training import build_binance_stablecoin_denominator_source as builder
from training import build_binance_stablecoin_quote_flow as flow


def _rows(
    start: str,
    *,
    hours: int,
    price_scale: float,
) -> list[list[Any]]:
    first = cast(pd.Timestamp, pd.Timestamp(start, tz="UTC"))
    step = 3_600_000
    rows: list[list[Any]] = []
    for index in range(hours):
        current = cast(pd.Timestamp, first + pd.Timedelta(hours=index))
        open_time = int(current.timestamp() * 1_000)
        close = (30_000.0 + index) * float(price_scale)
        rows.append(
            [
                open_time,
                close - 5.0,
                close + 10.0,
                close - 10.0,
                close,
                10.0,
                open_time + step - 1,
                300_000.0,
                100,
                4.0,
                120_000.0,
                0.0,
            ]
        )
    return rows


def _archive(rows: list[list[Any]], *, symbol: str) -> bytes:
    frame = pd.DataFrame(
        {
            column: [row[index] for row in rows]
            for index, column in enumerate(flow.RAW_COLUMNS)
        }
    )
    text = io.StringIO()
    frame.to_csv(text, index=False)
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{symbol}-1h-test.csv", text.getvalue())
    return output.getvalue()


class _FakeFetcher:
    def __init__(self, payloads: dict[tuple[str, str], bytes]) -> None:
        self.payloads = payloads

    def __call__(self, url: str, *, retries: int, timeout: int) -> bytes:
        del retries, timeout
        filename = url.removesuffix(".CHECKSUM").rsplit("/", 1)[-1]
        symbol, _, month_zip = filename.partition("-1h-")
        month = month_zip.removesuffix(".zip")
        payload = self.payloads[(symbol, month)]
        if url.endswith(".CHECKSUM"):
            return f"{hashlib.sha256(payload).hexdigest()}  {filename}\n".encode()
        return payload


def _reference(tmp_path: Path, payloads: dict[tuple[str, str], bytes]) -> Path:
    path = tmp_path / "reference.json"
    records = []
    for (symbol, month), payload in sorted(payloads.items()):
        start = "2023-08-04 08:00" if symbol == "BTCFDUSD" else "2023-08-01"
        hours = 664 if symbol == "BTCFDUSD" else 744
        records.append(
            {
                "symbol": symbol,
                "month": month,
                "archive_sha256": hashlib.sha256(payload).hexdigest(),
                "rows": hours,
                "first_date": start,
            }
        )
    path.write_text(
        json.dumps(
            {
                "protocol": {
                    "outcomes_opened": False,
                    "price_fields_retained": False,
                },
                "archives": records,
            },
            sort_keys=True,
        )
    )
    return path


def _august_payloads() -> dict[tuple[str, str], bytes]:
    return {
        ("BTCUSDT", "2023-08"): _archive(
            _rows("2023-08-01", hours=744, price_scale=1.0), symbol="BTCUSDT"
        ),
        ("BTCUSDC", "2023-08"): _archive(
            _rows("2023-08-01", hours=744, price_scale=1.001), symbol="BTCUSDC"
        ),
        ("BTCFDUSD", "2023-08"): _archive(
            _rows("2023-08-04 08:00", hours=664, price_scale=0.999),
            symbol="BTCFDUSD",
        ),
    }


def test_price_panel_is_ephemeral_and_timestamp_exact() -> None:
    raw, unit = flow.read_archive(
        _archive(
            _rows("2023-08-01", hours=2, price_scale=1.001),
            symbol="BTCUSDC",
        )
    )
    panel = builder.price_panel(raw, symbol="BTCUSDC", unit=unit)
    assert panel.columns.tolist() == ["date", "symbol", "close_time_us", "close"]
    assert panel.loc[0, "date"] == pd.Timestamp("2023-08-01 00:00:00")
    assert panel.loc[0, "close"] == pytest.approx(30_030.0)


def test_cross_quote_panel_cancels_common_btc_price() -> None:
    frames = []
    for symbol, scale in (
        ("BTCUSDT", 1.0),
        ("BTCUSDC", 1.001),
        ("BTCFDUSD", 0.999),
    ):
        raw, unit = flow.read_archive(
            _archive(
                _rows("2023-08-04 08:00", hours=3, price_scale=scale),
                symbol=symbol,
            )
        )
        frames.append(builder.price_panel(raw, symbol=symbol, unit=unit))
    output = builder.cross_quote_panel(
        frames,
        start=pd.Timestamp("2023-08-04 08:00"),
        end=pd.Timestamp("2023-08-04 11:00"),
    )
    assert tuple(output.columns) == builder.OUTPUT_COLUMNS
    assert len(output) == 3
    assert np.allclose(output["usdc_vs_usdt"], np.log(1.001))
    assert np.allclose(output["fdusd_vs_usdt"], np.log(0.999))
    assert np.allclose(
        output["alt_consensus"], (np.log(1.001) + np.log(0.999)) / 2.0
    )
    assert np.allclose(
        output["alt_disagreement"], abs(np.log(1.001) - np.log(0.999))
    )
    assert output["source_available_at"].iloc[0] == pd.Timestamp(
        "2023-08-04 09:00"
    )
    assert not {"close", "open", "high", "low", "volume"}.intersection(
        output.columns
    )


def test_cross_quote_panel_fails_closed_on_missing_or_misaligned_book() -> None:
    frames = []
    for symbol in builder.DEFAULT_SYMBOLS:
        raw, unit = flow.read_archive(
            _archive(
                _rows("2023-08-04 08:00", hours=3, price_scale=1.0),
                symbol=symbol,
            )
        )
        frames.append(builder.price_panel(raw, symbol=symbol, unit=unit))
    frames[-1] = frames[-1].iloc[:-1].copy()
    with pytest.raises(ValueError, match="common source grid is incomplete"):
        builder.cross_quote_panel(
            frames,
            start=pd.Timestamp("2023-08-04 08:00"),
            end=pd.Timestamp("2023-08-04 11:00"),
        )

    frames = []
    for symbol in builder.DEFAULT_SYMBOLS:
        raw, unit = flow.read_archive(
            _archive(
                _rows("2023-08-04 08:00", hours=3, price_scale=1.0),
                symbol=symbol,
            )
        )
        frames.append(builder.price_panel(raw, symbol=symbol, unit=unit))
    frames[-1].loc[1, "close_time_us"] += 1
    with pytest.raises(ValueError, match="close timestamps are misaligned"):
        builder.cross_quote_panel(
            frames,
            start=pd.Timestamp("2023-08-04 08:00"),
            end=pd.Timestamp("2023-08-04 11:00"),
        )


def test_build_is_deterministic_and_retains_no_raw_price(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payloads = _august_payloads()
    reference = _reference(tmp_path, payloads)
    monkeypatch.setattr(
        builder, "REFERENCE_MANIFEST_SHA256", hashlib.sha256(reference.read_bytes()).hexdigest()
    )
    cfg = builder.BuildConfig(
        start="2023-08-01",
        end="2023-09-01",
        output_dir=str(tmp_path / "output"),
        reference_manifest=str(reference),
        workers=3,
    )
    first = builder.build(cfg, fetcher=_FakeFetcher(payloads))
    output = Path(first["combined_output"])
    output_bytes = output.read_bytes()
    manifest_bytes = (Path(cfg.output_dir) / "build_manifest.json").read_bytes()
    second = builder.build(cfg, fetcher=_FakeFetcher(payloads))
    assert Path(second["combined_output"]).read_bytes() == output_bytes
    assert (Path(cfg.output_dir) / "build_manifest.json").read_bytes() == manifest_bytes
    assert first["rows"] == first["complete_rows"] == 664
    assert first["first_date"] == "2023-08-04T08:00:00"
    assert first["last_date"] == "2023-08-31T23:00:00"
    assert first["protocol"]["outcomes_opened"] is False
    assert first["protocol"]["raw_btc_prices_retained"] is False
    assert first["protocol"]["future_returns_labels_or_pnl_opened"] is False
    stored = pd.read_csv(output)
    assert tuple(stored.columns) == builder.OUTPUT_COLUMNS
    assert not {"open", "high", "low", "close", "volume"}.intersection(stored.columns)


def test_reference_and_config_fail_closed(tmp_path: Path) -> None:
    reference = tmp_path / "bad-reference.json"
    reference.write_text(
        json.dumps(
            {
                "protocol": {
                    "outcomes_opened": True,
                    "price_fields_retained": False,
                },
                "archives": [],
            }
        )
    )
    with pytest.raises(ValueError, match="reference manifest hash mismatch"):
        builder._load_reference(reference)
    with pytest.raises(ValueError, match="pre-2024 prefix"):
        builder._validate_config(builder.BuildConfig(end="2024-02-01"))
    with pytest.raises(ValueError, match="frozen three-book basket"):
        builder._validate_config(
            builder.BuildConfig(symbols=("BTCUSDT", "BTCUSDC", "BTCFDUSD", "BTCBUSD"))
        )
