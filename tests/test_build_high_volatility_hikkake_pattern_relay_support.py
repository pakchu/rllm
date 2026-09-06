import numpy as np
import pandas as pd

from training import build_high_volatility_hikkake_pattern_relay_support as support


def bars(values):
    return pd.DataFrame(
        {
            "bar_open": [(high + low) / 2 for high, low, _ in values],
            "bar_high": [high for high, _, _ in values],
            "bar_low": [low for _, low, _ in values],
            "bar_close": [close for _, _, close in values],
        }
    )


def test_bullish_setup_and_next_bar_confirmation() -> None:
    values = [(10 + i, 0 - i, 5) for i in range(6)]
    values += [(10, 0, 5), (9, 1, 4), (8, 0, 4), (10, 1, 9.5)]
    output = support.hikkake_outputs(bars(values), pd.Series([True] * len(values)))
    assert output.hikkake_output.iloc[-3:].tolist() == [0, 1, 2]
    assert output.entry_side.iloc[-1] == 1


def test_bearish_setup_and_third_bar_confirmation() -> None:
    values = [(10 + i, 0 - i, 5) for i in range(6)]
    values += [(10, 0, 5), (9, 1, 5), (10, 2, 6), (9, 1, 5), (9, 1, 5), (8, 0, 0.5)]
    output = support.hikkake_outputs(bars(values), pd.Series([True] * len(values)))
    assert output.hikkake_output.iloc[-4:].tolist() == [-1, 0, 0, -2]
    assert output.entry_side.iloc[-1] == -1


def test_new_setup_overwrites_same_bar_confirmation() -> None:
    values = [(10 + i, 0 - i, 5) for i in range(6)]
    values += [(10, 0, 5), (9, 1, 4), (8, 0, 4), (7, 1, 4), (8, 2, 9.5)]
    output = support.hikkake_outputs(bars(values), pd.Series([True] * len(values)))
    assert output.hikkake_output.iloc[-4:].tolist() == [0, 1, 0, -1]
    assert output.entry_side.iloc[-1] == 0


def test_invalid_hour_resets_readiness_and_pattern_state() -> None:
    values = [(10 + i, 0 - i, 5) for i in range(14)]
    valid = pd.Series([True] * 7 + [False] + [True] * 6)
    output = support.hikkake_outputs(bars(values), valid)
    assert not output.hikkake_ready.iloc[8:13].any()
    assert output.hikkake_ready.iloc[13]


def test_prior_rank_excludes_current_and_resets_on_invalid() -> None:
    values = pd.Series(np.arange(725, dtype=float))
    valid = pd.Series([True] * len(values))
    rank = support.prior_rank(values, valid)
    assert np.isnan(rank.iloc[719])
    assert rank.iloc[720] == 1.0
    valid.iloc[721] = False
    reset = support.prior_rank(values, valid)
    assert np.isnan(reset.iloc[722])


def panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_valid": [True] * 6,
            "entry_side": [0, 0, 1, 0, -1, 0],
            "initial_setup_side": [0, -1, 0, 1, 0, 0],
            "all_nonzero_side": [0, -1, 1, 1, -1, 0],
            "variation_rank": [0.8, 0.8, 0.8, 0.8, 0.4, 0.8],
        }
    )


def test_controls_are_diagnostic_only_transformations() -> None:
    activity, side, _ = support.active(panel())
    assert activity.tolist() == [False, False, True, False, False, False]
    assert side[activity].tolist() == [1]
    assert support.active(panel(), "no_variation_gate")[0].iloc[4]
    initial, initial_side, _ = support.active(panel(), "initial_setup_only")
    assert initial.iloc[1] and initial_side.iloc[1] == -1
    all_nonzero, all_side, _ = support.active(panel(), "all_nonzero_hikkake_outputs")
    assert all_nonzero.iloc[3] and all_side.iloc[3] == 1
    stale, stale_side, _ = support.active(panel(), "one_hour_stale_confirmation")
    assert stale.iloc[3] and stale_side.iloc[3] == 1
    flipped, flipped_side, _ = support.active(panel(), "direction_flip")
    assert flipped.iloc[2] and flipped_side.iloc[2] == -1
    forced, forced_side, _ = support.active(panel(), "forced_long")
    assert forced.iloc[2] and forced_side.iloc[2] == 1
