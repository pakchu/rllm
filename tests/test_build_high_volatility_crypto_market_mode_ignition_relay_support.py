import numpy as np
import pandas as pd

from training import build_high_volatility_crypto_market_mode_ignition_relay_support as support


def test_market_mode_share_and_orientation_are_deterministic():
    rng = np.random.default_rng(7)
    common = rng.normal(size=(480, 1))
    matrix = common + 0.1 * rng.normal(size=(480, 6))
    share, loading = support.market_mode(matrix)
    assert share > 0.95
    assert loading.sum() > 0
    assert np.all(loading > 0)


def test_source_valid_onset_uses_immediately_previous_valid_block():
    dominant = pd.Series([False, True, False, True, True])
    valid = pd.Series([True, True, False, True, True])
    assert support.source_valid_onset(dominant, valid).tolist() == [False, True, False, False, False]


def _panel() -> pd.DataFrame:
    rows = []
    for index, decision in enumerate(pd.date_range("2023-07-01T00:00Z", periods=4, freq="8h")):
        rows.append({
            "decision_time": decision, "feature_available_time": decision, "source_valid": True,
            "minute_count": 3360, "pc1_variance_share": .5, "mode_rank": .8,
            "mode_dominant": True, "mode_onset": index in (0, 1, 2), "direction_score": 1.,
            "equal_weight_direction_score": 1., "direction_side": 1,
            "equal_weight_direction_side": 1, "btc_variation": .01, "btc_variation_rank": .8,
            **{f"pc1_loading_{symbol}": 1 / np.sqrt(6) for symbol in support.ALTS},
        })
    return pd.DataFrame(rows, columns=support.BLOCK_COLUMNS)


def test_clock_is_delayed_eight_hours_and_half_open():
    clock = support.build_clock(_panel())
    assert len(clock) == 3
    assert (clock.entry_time - clock.decision_time).eq(pd.Timedelta(minutes=5)).all()
    assert (clock.exit_time - clock.entry_time).eq(pd.Timedelta(hours=8)).all()
    assert clock.entry_time.iloc[1] == clock.exit_time.iloc[0]


def test_source_artifact_schema_is_outcome_blind():
    forbidden = {"return", "pnl", "funding", "execution_price", "gross9"}
    names = {name.lower() for name in (*support.BLOCK_COLUMNS, *support.CLOCK_COLUMNS)}
    assert not names.intersection(forbidden)
    assert "btc_return" not in names
