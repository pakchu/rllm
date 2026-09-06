import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from training import build_high_volatility_inverse_linear_funding_transfer_relay_support as support


def archive_bytes(body: str) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("funding.csv", body)
    return output.getvalue()


def test_parse_archive_enforces_exact_schema_and_interval() -> None:
    body = "calc_time,funding_interval_hours,last_funding_rate\n1672531200000,8,0.0001\n"
    rows = support.parse_archive(archive_bytes(body), "BTCUSDT")
    assert rows == [{
        "instrument": "BTCUSDT",
        "calc_time": pd.Timestamp("2023-01-01T00:00:00Z"),
        "funding_rate": 0.0001,
    }]
    bad = body.replace(",8,", ",4,")
    with pytest.raises(ValueError, match="eight hours"):
        support.parse_archive(archive_bytes(bad), "BTCUSDT")


def test_parse_rest_row_requires_identity_and_finite_rate() -> None:
    row = support.parse_rest_row({
        "symbol": "BTCUSD_PERP", "fundingTime": 1672531200000,
        "fundingRate": "-0.0002", "markPrice": "16500",
    }, "BTCUSD_PERP")
    assert row["funding_rate"] == -0.0002
    with pytest.raises(ValueError, match="identity"):
        support.parse_rest_row({
            "symbol": "ETHUSD_PERP", "fundingTime": 1672531200000, "fundingRate": "0",
        }, "BTCUSD_PERP")


def test_strict_prior_midrank_excludes_current() -> None:
    ranked = support.strict_prior_midrank(pd.Series([1.0, 2.0, 3.0, 4.0]), lookback=3, minimum=2)
    assert np.isnan(ranked.iloc[0])
    assert np.isnan(ranked.iloc[1])
    assert ranked.iloc[2] == 1.0
    assert ranked.iloc[3] == 1.0


def test_candidate_clock_fades_transfer_and_allows_equal_exit_entry() -> None:
    times = pd.date_range("2024-01-01T00:00:00Z", periods=5, freq="8h")
    panel = pd.DataFrame({
        "calc_time": times, "source_valid": True,
        "coinm_funding_rate": 0.001, "usdm_funding_rate": 0.0,
        "funding_differential": 0.001, "funding_transfer": [1.0, 0.0, -1.0, 0.0, 1.0],
        "coinm_funding_change": [1.0, 0.0, -1.0, 0.0, 1.0],
        "absolute_transfer_rank": [0.9, 0.0, 0.9, 0.0, 0.9],
        "realized_variation": 1.0, "realized_variation_rank": 0.9,
    })
    clock = support.candidate_clock(panel)
    assert list(clock.observation_time) == [times[0], times[2], times[4]]
    assert list(clock.side) == [-1, 1, -1]
    assert clock.entry_time.iloc[1] == clock.exit_time.iloc[0] + pd.Timedelta(hours=8)
