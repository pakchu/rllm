from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from training import build_binance_stablecoin_quote_flow as flow
from training import build_binance_usdt_collateral_breadth_source as builder


def _rows(
    start: str,
    *,
    hours: int,
    price: float,
    invalid_at: int | None = None,
) -> list[list[Any]]:
    first = cast(pd.Timestamp, pd.Timestamp(start, tz="UTC"))
    step = 3_600_000
    rows: list[list[Any]] = []
    for index in range(hours):
        current = cast(pd.Timestamp, first + pd.Timedelta(hours=index))
        open_time = int(current.timestamp() * 1_000)
        active = index != invalid_at
        rows.append(
            [
                open_time,
                price,
                price,
                price,
                price,
                10.0 if active else 0.0,
                open_time + step - 1,
                10.0 if active else 0.0,
                100 if active else 0,
                4.0 if active else 0.0,
                4.0 if active else 0.0,
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


def _august_payloads() -> dict[tuple[str, str], bytes]:
    payloads: dict[tuple[str, str], bytes] = {}
    for index, symbol in enumerate(builder.DEFAULT_SYMBOLS):
        payloads[(symbol, "2023-08")] = _archive(
            _rows(
                "2023-08-01",
                hours=744,
                price=1.0 + index * 0.001,
                invalid_at=10 if symbol == "USDPUSDT" else None,
            ),
            symbol=symbol,
        )
    return payloads


def test_pair_panel_retains_log_close_and_current_validity_only() -> None:
    raw, unit = flow.read_archive(
        _archive(
            _rows("2023-08-01", hours=2, price=1.001, invalid_at=1),
            symbol="USDCUSDT",
        )
    )
    panel = builder.pair_panel(raw, symbol="USDCUSDT", unit=unit)
    assert panel.columns.tolist() == [
        "date",
        "symbol",
        "close_time_us",
        "log_close",
        "valid",
    ]
    assert panel.loc[0, "log_close"] == pytest.approx(np.log(1.001))
    assert bool(panel.loc[0, "valid"])
    assert not bool(panel.loc[1, "valid"])


def test_breadth_panel_allows_one_stale_member_but_not_two() -> None:
    frames: list[pd.DataFrame] = []
    for index, symbol in enumerate(builder.DEFAULT_SYMBOLS):
        raw, unit = flow.read_archive(
            _archive(
                _rows(
                    "2023-08-01",
                    hours=2,
                    price=1.0 + index * 0.001,
                    invalid_at=1 if symbol == "USDPUSDT" else None,
                ),
                symbol=symbol,
            )
        )
        frames.append(builder.pair_panel(raw, symbol=symbol, unit=unit))
    panel = builder.breadth_panel(
        frames,
        start=pd.Timestamp("2023-08-01"),
        end=pd.Timestamp("2023-08-01 02:00"),
    )
    assert tuple(panel.columns) == builder.OUTPUT_COLUMNS
    assert panel["valid_breadth"].tolist() == [4, 3]
    assert panel["source_complete"].tolist() == [True, True]

    frames[0].loc[1, "valid"] = False
    panel = builder.breadth_panel(
        frames,
        start=pd.Timestamp("2023-08-01"),
        end=pd.Timestamp("2023-08-01 02:00"),
    )
    assert panel["valid_breadth"].tolist() == [4, 2]
    assert panel["source_complete"].tolist() == [True, False]


def test_breadth_panel_rejects_missing_or_misaligned_source() -> None:
    frames: list[pd.DataFrame] = []
    for symbol in builder.DEFAULT_SYMBOLS:
        raw, unit = flow.read_archive(
            _archive(
                _rows("2023-08-01", hours=2, price=1.0),
                symbol=symbol,
            )
        )
        frames.append(builder.pair_panel(raw, symbol=symbol, unit=unit))
    missing = [frame.copy() for frame in frames]
    missing[-1] = missing[-1].iloc[:-1]
    with pytest.raises(ValueError, match="common grid is incomplete"):
        builder.breadth_panel(
            missing,
            start=pd.Timestamp("2023-08-01"),
            end=pd.Timestamp("2023-08-01 02:00"),
        )
    frames[-1].loc[1, "close_time_us"] += 1
    with pytest.raises(ValueError, match="misaligned"):
        builder.breadth_panel(
            frames,
            start=pd.Timestamp("2023-08-01"),
            end=pd.Timestamp("2023-08-01 02:00"),
        )


def test_build_is_deterministic_and_retains_no_raw_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payloads = _august_payloads()
    monkeypatch.setattr(
        builder,
        "EXPECTED_ARCHIVE_SHA256",
        {key: hashlib.sha256(value).hexdigest() for key, value in payloads.items()},
    )
    cfg = builder.BuildConfig(
        start="2023-08-01",
        end="2023-09-01",
        output_dir=str(tmp_path / "output"),
        workers=4,
    )
    monkeypatch.setattr(
        builder,
        "_validate_config",
        lambda _: (
            pd.Timestamp("2023-08-01").date(),
            pd.Timestamp("2023-09-01").date(),
            builder.DEFAULT_SYMBOLS,
        ),
    )
    first = builder.build(cfg, fetcher=_FakeFetcher(payloads))
    output = Path(first["combined_output"])
    first_bytes = output.read_bytes()
    manifest_bytes = (Path(cfg.output_dir) / "build_manifest.json").read_bytes()
    second = builder.build(cfg, fetcher=_FakeFetcher(payloads))
    assert output.read_bytes() == first_bytes
    assert (Path(cfg.output_dir) / "build_manifest.json").read_bytes() == manifest_bytes
    assert first["rows"] == first["complete_rows"] == 744
    assert first["minimum_observed_breadth"] == 3
    assert first["protocol"]["outcomes_opened"] is False
    assert first["protocol"]["raw_ohlc_retained"] is False
    stored = pd.read_csv(output)
    assert tuple(stored.columns) == builder.OUTPUT_COLUMNS
    assert not {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trade_count",
        "taker_buy",
    }.intersection(stored.columns)


def test_real_config_and_archive_set_are_frozen() -> None:
    with pytest.raises(ValueError, match="frozen to"):
        builder._validate_config(builder.BuildConfig(end="2024-02-01"))
    expected = {
        (symbol, month)
        for symbol in builder.DEFAULT_SYMBOLS
        for month in ("2023-08", "2023-09", "2023-10", "2023-11", "2023-12")
    }
    assert set(builder.EXPECTED_ARCHIVE_SHA256) == expected
