import numpy as np
import pandas as pd

from training import build_high_volatility_three_inside_reversal_relay_support as support


def bars(open_close):
    return pd.DataFrame(
        {
            "bar_open": [open_ for open_, _ in open_close],
            "bar_close": [close for _, close in open_close],
            "bar_high": [max(open_, close) + 0.5 for open_, close in open_close],
            "bar_low": [min(open_, close) - 0.5 for open_, close in open_close],
        }
    )


def prefix():
    return [(10.0, 11.0) if index % 2 == 0 else (11.0, 10.0) for index in range(10)]


def test_three_inside_up_uses_lagged_body_averages() -> None:
    values = prefix() + [(10.0, 6.0), (8.0, 7.0), (7.0, 11.0)]
    output = support.three_inside_outputs(bars(values), pd.Series([True] * len(values)))
    assert not output.three_inside_ready.iloc[11]
    assert output.three_inside_ready.iloc[12]
    assert output.three_inside_output.iloc[12] == 1
    assert output.entry_side.iloc[12] == 1


def test_three_inside_down_uses_opposite_first_color() -> None:
    values = prefix() + [(6.0, 10.0), (8.0, 9.0), (9.0, 5.0)]
    output = support.three_inside_outputs(bars(values), pd.Series([True] * len(values)))
    assert output.three_inside_output.iloc[12] == -1


def test_first_body_must_be_strictly_longer_than_average() -> None:
    values = prefix() + [(10.0, 9.0), (9.8, 9.2), (9.2, 10.5)]
    output = support.three_inside_outputs(bars(values), pd.Series([True] * len(values)))
    assert output.three_inside_output.iloc[12] == 0
    assert output.no_body_size_side.iloc[12] == 1


def test_body_containment_is_strict_at_both_endpoints() -> None:
    values = prefix() + [(10.0, 6.0), (10.0, 7.0), (7.0, 11.0)]
    output = support.three_inside_outputs(bars(values), pd.Series([True] * len(values)))
    assert output.three_inside_output.iloc[12] == 0
    assert output.no_body_size_side.iloc[12] == 0


def test_invalid_hour_resets_thirteen_bar_readiness() -> None:
    values = prefix() + [(10.0, 6.0), (8.0, 7.0), (7.0, 11.0)] + prefix() + [(10.0, 6.0), (8.0, 7.0), (7.0, 11.0)]
    valid = pd.Series([True] * len(values)); valid.iloc[12] = False
    output = support.three_inside_outputs(bars(values), valid)
    assert not output.three_inside_ready.iloc[13:25].any()
    assert output.three_inside_ready.iloc[25]


def test_prior_rank_excludes_current_and_resets() -> None:
    values = pd.Series(np.arange(725, dtype=float)); valid = pd.Series([True] * len(values))
    rank = support.prior_rank(values, valid)
    assert np.isnan(rank.iloc[719]) and rank.iloc[720] == 1.0
    valid.iloc[721] = False
    assert np.isnan(support.prior_rank(values, valid).iloc[722])


def panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_valid": [True] * 6,
            "entry_side": [0, 0, 1, 0, -1, 0],
            "no_body_size_side": [0, -1, 0, 1, 0, 0],
            "containment_side": [0, -1, 1, 1, -1, 0],
            "variation_rank": [0.8, 0.8, 0.8, 0.8, 0.4, 0.8],
        }
    )


def test_controls_are_fixed_diagnostic_transformations() -> None:
    activity, side, _ = support.active(panel())
    assert activity.iloc[2] and side.iloc[2] == 1 and not activity.iloc[4]
    assert support.active(panel(), "no_variation_gate")[0].iloc[4]
    no_size, no_size_side, _ = support.active(panel(), "no_body_size_requirements")
    assert no_size.iloc[1] and no_size_side.iloc[1] == -1
    contained, contained_side, _ = support.active(panel(), "containment_without_third_confirmation")
    assert contained.iloc[3] and contained_side.iloc[3] == 1
    stale, stale_side, _ = support.active(panel(), "one_hour_stale_pattern")
    assert stale.iloc[3] and stale_side.iloc[3] == 1
    flipped, flipped_side, _ = support.active(panel(), "direction_flip")
    assert flipped.iloc[2] and flipped_side.iloc[2] == -1
    forced, forced_side, _ = support.active(panel(), "forced_long")
    assert forced.iloc[2] and forced_side.iloc[2] == 1
