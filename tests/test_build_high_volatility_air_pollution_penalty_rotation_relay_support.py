import numpy as np
import pandas as pd
import pytest

from training import build_high_volatility_air_pollution_penalty_rotation_relay_support as support


def test_pm25_aqi_breakpoints_and_truncation():
    assert support.pm25_aqi(0) == 0
    assert support.pm25_aqi(9.09) == 50
    assert support.pm25_aqi(35.49) == 100
    assert support.pm25_aqi(55.49) == 150
    assert support.pm25_aqi(500) == 500


def test_range_parser_selects_only_nyc_pm25():
    body = b"partial\n07/01/23|23:00|360050080|Morrisania|-5|PM2.5|UG/M3|31.1|Agency\r\n07/01/23|23:00|360050081|Missing|-5|PM2.5|UG/M3|-999|Agency\r\n07/01/23|23:00|360050080|Morrisania|-5|OZONE|PPB|60|Agency\r\n07/01/23|23:00|360010005|Albany|-5|PM2.5|UG/M3|10|Agency\r\npartial"
    assert support.parse_range_body(body) == [("07/01/23", "360050080", 31.1)]


def test_range_parser_rejects_missing_new_york_rows():
    with pytest.raises(RuntimeError, match="New York"):
        support.parse_range_body(b"partial\n07/01/23|23:00|340010001|X|-5|PM2.5|UG/M3|10|Agency\npartial")


def test_strict_prior_midrank_excludes_current():
    values = pd.Series(list(range(60)) + [100.0])
    ranks = support.strict_prior_midrank(values)
    assert np.isnan(ranks.iloc[59])
    assert ranks.iloc[60] == 1.0


def test_primary_clock_uses_both_frozen_gates():
    features = pd.DataFrame({
        "source_day": pd.to_datetime(["2023-07-01", "2023-07-03"], utc=True),
        "decision_time": pd.to_datetime(["2023-07-02T01:00Z", "2023-07-04T01:00Z"]),
        "city_pm25_aqi": [60, 40], "aqi_change": [10.0, -20.0], "aqi_change_rank": [0.7, 0.8],
        "pollution_side": [-1, 1], "btc_realized_variation": [0.1, 0.2], "btc_variation_rank": [0.7, 0.8],
    })
    clock = support.build_clock(features)
    assert clock.side.tolist() == [-1, 1]
    assert (pd.to_datetime(clock.entry_time, utc=True) - pd.to_datetime(clock.decision_time, utc=True)).eq(pd.Timedelta(minutes=5)).all()
