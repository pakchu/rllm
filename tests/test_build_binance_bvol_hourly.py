from __future__ import annotations

import io
import zipfile
from datetime import date

import numpy as np
import pandas as pd
import pytest

from training import build_binance_bvol_hourly as builder


def _archive(frame: pd.DataFrame) -> bytes:
    text = io.StringIO()
    frame.to_csv(text, index=False)
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("BTCBVOLUSDT-BVOLIndex-test.csv", text.getvalue())
    return output.getvalue()


def _raw(start: str, periods: int) -> pd.DataFrame:
    timestamps = pd.date_range(start, periods=periods, freq="1s", tz="UTC")
    values = np.linspace(40.0, 50.0, periods)
    return pd.DataFrame(
        {
            "calc_time": timestamps.astype("int64") // 1_000_000,
            "symbol": builder.SYMBOL,
            "base_asset": "BTCBVOL",
            "quote_asset": "USDT",
            "index_value": values,
        }
    )


def test_read_and_aggregate_complete_hour() -> None:
    raw = _raw("2023-06-20 00:00:00", 3_600)
    parsed = builder.read_archive(_archive(raw))
    output = builder.aggregate_day(parsed, date(2023, 6, 20))
    first = output.iloc[0]
    second = output.iloc[1]

    assert first["source_rows"] == 3_600
    assert first["source_complete"] == np.bool_(True)
    assert first["feature_valid"] == np.bool_(True)
    assert first["open"] == pytest.approx(40.0)
    assert first["close"] == pytest.approx(50.0)
    assert first["feature_available_time_utc"] == pd.Timestamp("2023-06-20 01:00:00")
    assert second["source_complete"] == np.bool_(False)
    assert second[["open", "high", "low", "close"]].isna().all()


def test_subsecond_jitter_is_floored_but_duplicate_seconds_fail_closed() -> None:
    raw = _raw("2023-06-20 00:00:00", 2)
    raw.loc[1, "calc_time"] += 1
    parsed = builder.read_archive(_archive(raw))
    assert parsed["date"].tolist() == [
        pd.Timestamp("2023-06-20 00:00:00"),
        pd.Timestamp("2023-06-20 00:00:01"),
    ]
    raw.loc[1, "calc_time"] = raw.loc[0, "calc_time"] + 999
    with pytest.raises(ValueError, match="duplicate or unordered UTC seconds"):
        builder.read_archive(_archive(raw))


def test_symbol_fails_closed() -> None:
    raw = _raw("2023-06-20 00:00:00", 2)
    raw.loc[0, "symbol"] = "ETHBVOLUSDT"
    with pytest.raises(ValueError, match="unexpected symbol"):
        builder.read_archive(_archive(raw))


def test_post2023_build_is_sealed_without_explicit_flag(tmp_path) -> None:
    cfg = builder.BuildConfig(
        start="2023-12-31",
        end="2024-01-02",
        output_dir=str(tmp_path),
    )
    with pytest.raises(ValueError, match="sealed"):
        builder.build(cfg)


def test_archive_urls_are_official_binance_vision_paths() -> None:
    day = date(2023, 6, 20)
    assert builder.archive_url(day) == (
        "https://data.binance.vision/data/option/daily/BVOLIndex/"
        "BTCBVOLUSDT/BTCBVOLUSDT-BVOLIndex-2023-06-20.zip"
    )
    assert builder.checksum_url(day).endswith(".zip.CHECKSUM")


@pytest.mark.parametrize("reason", ["archive_missing", "checksum_missing"])
def test_invalid_day_is_not_imputed(reason: str) -> None:
    output = builder.invalid_day(date(2023, 9, 25), reason)

    assert len(output) == 24
    assert not output["feature_valid"].any()
    assert not output["source_complete"].any()
    assert output[["open", "high", "low", "close"]].isna().all().all()
    assert output["feature_invalid_reason"].eq(reason).all()
    assert output["date"].iloc[0] == pd.Timestamp("2023-09-25 00:00:00")
    assert output["date"].iloc[-1] == pd.Timestamp("2023-09-25 23:00:00")


def test_missing_archive_is_sealed_invalid_in_month_output(tmp_path) -> None:
    cfg = builder.BuildConfig(
        start="2023-09-25",
        end="2023-09-26",
        output_dir=str(tmp_path),
    )

    def missing_fetcher(url: str, **_: object) -> bytes:
        raise FileNotFoundError(url)

    metadata = builder._process_month(
        date(2023, 9, 1), cfg, fetcher=missing_fetcher
    )
    output = pd.read_csv(metadata["output"], compression="gzip")

    assert metadata["feature_valid_rows"] == 0
    assert metadata["archives"] == [
        {"day": "2023-09-25", "status": "archive_missing", "raw_rows": 0}
    ]
    assert output["feature_invalid_reason"].eq("archive_missing").all()
