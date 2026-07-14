from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pytest

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


def _kline_row(
    timestamp: str,
    *,
    open_price: float = 100.0,
    close_price: float | None = None,
    quote_notional: float = 1_000.0,
    flow_frac: float = 0.0,
    trade_count: int = 10,
) -> list[object]:
    """Return one valid one-minute Binance kline row in canonical raw order."""
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
    price_scale: float = 1.0,
    flows: tuple[float, ...] = (0.8, 0.8, 0.8, 0.8, 0.0),
    returns: tuple[float, ...] = (0.0, 0.001, 0.001, 0.001, 0.001),
) -> list[list[object]]:
    rows: list[list[object]] = []
    open_price = base_price * price_scale
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


def _spot_leads_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    spot = builder.read_archive(
        _archive(
            _venue_rows(
                flows=(0.9, 0.7, 0.8, 0.6, 0.0),
                returns=(0.0, 0.0001, 0.0001, 0.0001, 0.0001),
            ),
            header=True,
            columns=SPOT_HEADER_ALIASES,
        ),
        venue="spot",
    )
    perp = builder.read_archive(
        _archive(
            _venue_rows(
                base_price=101.0,
                flows=(0.1, 0.1, 0.1, 0.1, 0.1),
                returns=(0.0, 0.004, 0.003, 0.002, 0.001),
            ),
            header=True,
            columns=UM_HEADER_ALIASES,
        ),
        venue="um",
    )
    return spot, perp


class _CrossVenueFetcher:
    def __init__(self, spot_payload: bytes, um_payload: bytes) -> None:
        self.spot_payload = spot_payload
        self.um_payload = um_payload
        self.calls: list[str] = []

    def __call__(self, url: str, *, retries: int, timeout: int) -> bytes:
        del retries, timeout
        self.calls.append(url)
        payload = self.spot_payload if "/spot/" in url else self.um_payload
        if url.endswith(".CHECKSUM"):
            digest = hashlib.sha256(payload).hexdigest()
            return f"{digest}  archive.zip\n".encode()
        return payload


@pytest.mark.parametrize(
    ("venue", "columns"),
    [("spot", SPOT_HEADER_ALIASES), ("um", UM_HEADER_ALIASES)],
)
def test_read_archive_accepts_header_aliases_and_no_header_schema(
    venue: str,
    columns: list[str],
) -> None:
    rows = _venue_rows()
    with_header = builder.read_archive(_archive(rows, header=True, columns=columns), venue=venue)
    without_header = builder.read_archive(_archive(rows, header=False), venue=venue)

    assert tuple(without_header.columns) == tuple(with_header.columns)
    assert with_header.loc[0, "open_time"] == rows[0][0]
    assert without_header.loc[0, "quote_notional"] == pytest.approx(1_000.0)
    assert with_header.loc[0, "trade_count"] == 10
    assert with_header["source_row_valid"].all()


def test_archive_urls_use_official_monthly_spot_and_um_roots() -> None:
    month = date(2023, 1, 1)
    assert builder.SPOT_BASE_URL == "https://data.binance.vision/data/spot/monthly/klines"
    assert builder.UM_BASE_URL == "https://data.binance.vision/data/futures/um/monthly/klines"
    assert builder.spot_archive_url("BTCUSDT", month) == (
        "https://data.binance.vision/data/spot/monthly/klines/"
        "BTCUSDT/1m/BTCUSDT-1m-2023-01.zip"
    )
    assert builder.um_archive_url("BTCUSDT", month) == (
        "https://data.binance.vision/data/futures/um/monthly/klines/"
        "BTCUSDT/1m/BTCUSDT-1m-2023-01.zip"
    )


def test_spot_leading_pattern_has_positive_flow_transfer_and_swap_antisymmetry() -> None:
    spot, perp = _spot_leads_rows()
    original = builder.aggregate_cross_venue_five_minute(spot, perp).iloc[0]
    swapped = builder.aggregate_cross_venue_five_minute(perp, spot).iloc[0]

    assert original["source_complete"] == np.bool_(True)
    assert original["cross_venue_feature_valid"] == np.bool_(True)
    assert original["flow_transfer_asymmetry"] > 0.0
    assert original["spot_to_um_lagged_flow_response_bp"] > 0.0
    assert swapped["flow_transfer_asymmetry"] == pytest.approx(
        -original["flow_transfer_asymmetry"]
    )
    assert swapped["spot_to_um_lagged_flow_response_bp"] == pytest.approx(
        original["um_to_spot_lagged_flow_response_bp"]
    )
    assert swapped["um_to_spot_lagged_flow_response_bp"] == pytest.approx(
        original["spot_to_um_lagged_flow_response_bp"]
    )
    assert swapped["simultaneous_flow_sign_agreement"] == pytest.approx(original["simultaneous_flow_sign_agreement"])
    assert swapped["simultaneous_return_sign_agreement"] == pytest.approx(original["simultaneous_return_sign_agreement"])


def test_positive_price_and_volume_scaling_preserves_normalized_ordering_fields() -> None:
    spot, perp = _spot_leads_rows()
    scaled_spot = builder.read_archive(
        _archive(
            _venue_rows(
                price_scale=7.0,
                quote_scale=13.0,
                flows=(0.9, 0.7, 0.8, 0.6, 0.0),
                returns=(0.0, 0.0001, 0.0001, 0.0001, 0.0001),
            ),
            header=True,
            columns=SPOT_HEADER_ALIASES,
        ),
        venue="spot",
    )
    scaled_perp = builder.read_archive(
        _archive(
            _venue_rows(
                base_price=101.0,
                price_scale=3.0,
                quote_scale=17.0,
                flows=(0.1, 0.1, 0.1, 0.1, 0.1),
                returns=(0.0, 0.004, 0.003, 0.002, 0.001),
            ),
            header=True,
            columns=UM_HEADER_ALIASES,
        ),
        venue="um",
    )
    baseline = builder.aggregate_cross_venue_five_minute(spot, perp).iloc[0]
    scaled = builder.aggregate_cross_venue_five_minute(scaled_spot, scaled_perp).iloc[0]

    invariant_columns = [
        "spot_to_um_lagged_flow_response_bp",
        "um_to_spot_lagged_flow_response_bp",
        "flow_transfer_asymmetry",
        "spot_to_um_lagged_directional_alignment",
        "um_to_spot_lagged_directional_alignment",
        "return_leadership_asymmetry",
        "simultaneous_flow_sign_agreement",
        "simultaneous_return_sign_agreement",
        "spot_activity_time_centroid",
        "um_activity_time_centroid",
        "um_minus_spot_activity_time_centroid",
        "spot_flow_time_centroid",
        "um_flow_time_centroid",
        "um_minus_spot_flow_time_centroid",
        "spot_return_time_centroid",
        "um_return_time_centroid",
        "um_minus_spot_return_time_centroid",
    ]
    for column in invariant_columns:
        assert scaled[column] == pytest.approx(baseline[column]), column


def test_next_five_minute_bar_does_not_change_current_row() -> None:
    spot, perp = _spot_leads_rows()
    extended_spot = pd.concat(
        [
            spot,
            builder.read_archive(
                _archive(
                    _venue_rows(
                        start="2023-01-01 00:05:00",
                        flows=(-0.9, -0.9, -0.9, -0.9, -0.9),
                        returns=(0.0, -0.02, -0.02, -0.02, -0.02),
                    ),
                    header=False,
                ),
                venue="spot",
            ),
        ],
        ignore_index=True,
    )
    extended_perp = pd.concat(
        [
            perp,
            builder.read_archive(
                _archive(
                    _venue_rows(
                        start="2023-01-01 00:05:00",
                        base_price=101.0,
                        flows=(0.9, 0.9, 0.9, 0.9, 0.9),
                        returns=(0.0, 0.03, 0.03, 0.03, 0.03),
                    ),
                    header=False,
                ),
                venue="um",
            ),
        ],
        ignore_index=True,
    )

    current_only = builder.aggregate_cross_venue_five_minute(spot, perp).iloc[0]
    with_next_bar = builder.aggregate_cross_venue_five_minute(extended_spot, extended_perp).iloc[0]
    pd.testing.assert_series_equal(
        current_only.loc[list(builder.OUTPUT_COLUMNS)],
        with_next_bar.loc[list(builder.OUTPUT_COLUMNS)],
        check_names=False,
    )


def test_missing_minute_fails_closed_without_accepted_nonfinite_features() -> None:
    spot, perp = _spot_leads_rows()
    missing_spot = spot.drop(index=spot.index[2]).reset_index(drop=True)
    output = builder.aggregate_cross_venue_five_minute(
        missing_spot,
        perp,
        expected_minutes=pd.date_range("2023-01-01", periods=5, freq="1min"),
    )
    row = output.iloc[0]

    assert row["source_complete"] == np.bool_(False)
    assert row["cross_venue_feature_valid"] == np.bool_(False)
    accepted = output.loc[output["cross_venue_feature_valid"].astype(bool), builder.OUTPUT_COLUMNS[1:]]
    assert np.isfinite(accepted.select_dtypes(include=[np.number]).to_numpy(float)).all()


def test_process_month_resume_rechecks_spot_and_um_checksums_and_is_deterministic(
    tmp_path: Path,
) -> None:
    spot_payload = _archive(_venue_rows(), header=True, columns=SPOT_HEADER_ALIASES)
    um_payload = _archive(
        _venue_rows(base_price=101.0),
        header=True,
        columns=UM_HEADER_ALIASES,
    )
    fetcher = _CrossVenueFetcher(spot_payload, um_payload)
    cfg = builder.BuildConfig(
        start="2023-01-01",
        end="2023-02-01",
        output_dir=str(tmp_path),
        workers=1,
    )

    first = builder._process_month(date(2023, 1, 1), cfg, fetcher=fetcher)
    second = builder._process_month(date(2023, 1, 1), cfg, fetcher=fetcher)

    assert first["output_sha256"] == second["output_sha256"]
    assert first["spot_archive_sha256"] == hashlib.sha256(spot_payload).hexdigest()
    assert first["um_archive_sha256"] == hashlib.sha256(um_payload).hexdigest()
    assert any(url.endswith(".CHECKSUM") and "/spot/" in url for url in fetcher.calls)
    assert any(url.endswith(".CHECKSUM") and "/futures/um/" in url for url in fetcher.calls)
    metadata_path = next((tmp_path / "monthly").glob("*.json"))
    assert json.loads(metadata_path.read_text())["schema_version"] == builder.SCHEMA_VERSION


def test_build_rejects_end_after_exclusive_2024_boundary() -> None:
    with pytest.raises(ValueError, match="2024"):
        builder.build(
            builder.BuildConfig(
                start="2023-12-01",
                end="2024-02-01",
                workers=1,
            )
        )
