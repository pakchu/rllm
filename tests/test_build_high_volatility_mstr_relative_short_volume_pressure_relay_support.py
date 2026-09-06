import numpy as np
import pandas as pd
import pytest

from training import build_high_volatility_mstr_relative_short_volume_pressure_relay_support as support


def test_parse_target_rows_requires_exact_pair() -> None:
    raw = (
        "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n"
        "20230103|MSTR|40|1|100|Q,N\n20230103|QQQ|50|2|100|B,Q,N\n"
    ).encode()
    rows = support.parse_target_rows(raw, pd.Timestamp("2023-01-03T00:00:00Z"))
    assert [row["symbol"] for row in rows] == ["MSTR", "QQQ"]
    with pytest.raises(RuntimeError, match="exact MSTR/QQQ pair"):
        support.parse_target_rows(raw.split(b"20230103|QQQ")[0], pd.Timestamp("2023-01-03T00:00:00Z"))


def test_strict_prior_midrank_excludes_current() -> None:
    ranked = support.strict_prior_midrank(pd.Series([1.0, 2.0, 3.0]), lookback=2, minimum=2)
    assert np.isnan(ranked.iloc[0]) and np.isnan(ranked.iloc[1]) and ranked.iloc[2] == 1.0


def test_candidate_clock_uses_pressure_follow_through_side() -> None:
    dates = pd.date_range("2024-01-01T00:00:00Z", periods=3, freq="2d")
    panel = pd.DataFrame({
        "source_date": dates, "feature_available_time": dates + pd.Timedelta(days=1), "source_valid": True,
        "mstr_short_volume": 1.0, "mstr_total_volume": 2.0, "qqq_short_volume": 1.0, "qqq_total_volume": 2.0,
        "mstr_short_share": 0.5, "qqq_short_share": 0.5, "relative_pressure": 0.0,
        "pressure_change": [1.0, 0.0, -1.0], "mstr_share_change": [1.0, 0.0, -1.0],
        "absolute_pressure_change_rank": [0.9, 0.0, 0.9], "realized_variation": 1.0, "realized_variation_rank": 0.9,
    })
    clock = support.candidate_clock(panel)
    assert list(clock.side) == [-1, 1]
