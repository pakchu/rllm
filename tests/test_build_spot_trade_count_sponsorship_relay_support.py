import numpy as np
import pandas as pd

from training import build_spot_trade_count_sponsorship_relay_support as support


def signal_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal_valid": [True] * 5,
            "spot_return": [0.02, -0.02, 0.02, 0.02, -0.02],
            "perp_return": [0.01, -0.01, -0.01, 0.01, -0.01],
            "spot_count_share_rank": [0.80, 0.90, 0.90, 0.70, 0.90],
            "perpetual_count_share_rank": [0.10, 0.80, 0.90, 0.90, 0.90],
            "variation_rank": [0.90, 0.80, 0.90, 0.90, 0.50],
        }
    )


def bars(start: str, count: float = 1.0) -> pd.DataFrame:
    timestamps = pd.date_range(start, periods=1440, freq="1min")
    prices = np.linspace(100.0, 101.0, len(timestamps))
    return pd.DataFrame(
        {
            "ts": timestamps,
            "open": prices,
            "high": prices + 1.0,
            "low": prices - 1.0,
            "close": prices + 0.1,
            "number_of_trades": count,
        }
    )


def test_rank_is_strict_prior_with_frozen_history_shape():
    values = pd.Series(np.arange(127, dtype=float))
    ranks = support.strict_prior_midrank(values)
    assert ranks.iloc[:126].isna().all()
    assert ranks.iloc[126] == 1.0


def test_daily_panel_requires_exact_aligned_integer_count_paths():
    start = "2023-01-01T08:00:00Z"
    panel = support.daily_panel(bars(start, 2), bars(start, 1))
    first = panel.iloc[0]
    assert first.source_valid
    assert first.spot_source_rows == first.perp_source_rows == 1440
    assert first.spot_count_share == 1 / 3

    invalid_spot = bars(start, 1)
    invalid_spot["number_of_trades"] = invalid_spot.number_of_trades.astype(float)
    invalid_spot.loc[10, "number_of_trades"] = 1.5
    invalid = support.daily_panel(bars(start, 2), invalid_spot).iloc[0]
    assert not invalid.source_valid


def test_primary_uses_share_variation_and_common_strict_return_sign():
    active, side = support.conditions(signal_frame(), "primary")
    assert active.tolist() == [True, True, False, False, False]
    assert side[active].tolist() == [1, -1]


def test_exact_controls_are_diagnostic_variants():
    frame = signal_frame()
    assert support.CONTROLS == (
        "no_volatility_gate",
        "no_count_share_tail",
        "perpetual_count_share",
        "one_day_stale_share",
        "direction_flip",
    )
    assert support.conditions(frame, "no_volatility_gate")[0].tolist() == [
        True,
        True,
        False,
        False,
        True,
    ]
    assert support.conditions(frame, "no_count_share_tail")[0].tolist() == [
        True,
        True,
        False,
        True,
        False,
    ]
    assert support.conditions(frame, "perpetual_count_share")[0].tolist() == [
        False,
        True,
        False,
        True,
        False,
    ]
    assert support.conditions(frame, "one_day_stale_share")[0].tolist() == [
        False,
        True,
        False,
        True,
        False,
    ]
    active, side = support.conditions(frame, "direction_flip")
    assert side[active].tolist() == [-1, 1]


def test_clock_enters_at_0805_and_holds_twelve_hours():
    decision = pd.Timestamp("2024-07-01T08:00:00Z")
    frame = pd.DataFrame(
        {
            "decision_time": [decision],
            "signal_valid": [True],
            "spot_return": [0.02],
            "perp_return": [0.01],
            "spot_count_share": [0.6],
            "spot_count_share_rank": [0.8],
            "perpetual_count_share": [0.4],
            "perpetual_count_share_rank": [0.2],
            "perp_realized_variation": [0.1],
            "variation_rank": [0.9],
        }
    )
    clock = support.clock(frame)
    assert len(clock) == 1
    assert pd.Timestamp(clock.iloc[0].entry_time) == decision + pd.Timedelta(minutes=5)
    assert pd.Timestamp(clock.iloc[0].exit_time) == decision + pd.Timedelta(
        hours=12, minutes=5
    )


def test_builder_is_frozen_and_outcome_blind():
    source = support.BUILDER.read_text()
    assert support.PREREG_SHA == (
        "f8125ea3a08d2f184f7c952ebc70b02626c424fc07673405a907ddccf4b115e7"
    )
    assert "evaluate_spot_trade_count" not in source
    assert '"candidate_incidence_opened": False' in source
    assert '"promotion_authorized": False' in source
