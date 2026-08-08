from __future__ import annotations

import pandas as pd

from training import build_options_crowding_deleveraging_relay_support as s


def feature_rows() -> pd.DataFrame:
    times = pd.date_range("2023-08-01T00:00:00Z", periods=4, freq="1h")
    return pd.DataFrame(
        {
            "decision_time": times,
            "base_valid": [True] * 4,
            "bvol_body": [0.01] * 4,
            "dvol_body": [0.02] * 4,
            "oi_change": [0.00, 0.02, 0.03, 0.00],
            "oi_tail": [0.01] * 4,
            "funding_rate": [0.001] * 4,
            "funding_tail": [0.0005] * 4,
        }
    )


def test_primary_uses_false_to_true_onset_and_fades_funding_crowd() -> None:
    clock = s.build_clock(feature_rows())
    assert len(clock) == 1
    assert clock.iloc[0]["decision_time"] == pd.Timestamp("2023-08-01T01:00:00Z")
    assert clock.iloc[0]["entry_time"] == pd.Timestamp("2023-08-01T01:05:00Z")
    assert clock.iloc[0]["exit_time"] == pd.Timestamp("2023-08-01T13:05:00Z")
    assert clock.iloc[0]["side"] == -1


def test_direction_flip_is_diagnostic_only_on_its_own_clock_build() -> None:
    primary = s.build_clock(feature_rows())
    flipped = s.build_clock(feature_rows(), "direction_flip")
    assert primary["entry_time"].tolist() == flipped["entry_time"].tolist()
    assert primary["side"].tolist() == [-value for value in flipped["side"].tolist()]


def test_source_support_never_authorizes_economics_directly() -> None:
    source = open(s.__file__).read()
    assert '"advance_to_economic_outcomes": False' in source
    assert "bars_binance" not in source
