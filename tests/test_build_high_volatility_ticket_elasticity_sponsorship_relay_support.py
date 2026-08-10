import numpy as np
import pandas as pd

from training import build_high_volatility_ticket_elasticity_sponsorship_relay_support as s


def test_elasticity_statistics_measures_count_to_turnover_slope():
    counts = np.arange(1.0, 481.0)
    quote = np.square(1.0 + counts) - 1.0
    closes = 100.0 * np.exp(np.linspace(0.0, 0.04, 480))
    block = pd.DataFrame(
        {
            "open": np.repeat(100.0, 480),
            "high": np.maximum(100.0, closes),
            "low": np.minimum(100.0, closes),
            "close": closes,
            "quote_asset_volume": quote,
            "number_of_trades": counts,
        }
    )
    elasticity, average_ticket, variation, completed, final_return = s.elasticity_statistics(block)
    assert np.isclose(elasticity, 2.0)
    assert average_ticket > 0
    assert variation > 0
    assert completed > 0
    assert final_return > 0


def panel():
    return pd.DataFrame(
        {
            "source_valid": [True] * 6,
            "ticket_elasticity": [1.1, 1.3, 1.4, 1.2, 1.5, 1.4],
            "elasticity_rank": [0.5, 0.8, 0.9, 0.4, 0.8, 0.9],
            "aggregate_average_ticket": [2, 3, 4, 2, 5, 4],
            "average_ticket_rank": [0.5, 0.8, 0.9, 0.4, 0.8, 0.9],
            "variation_rank": [0.8] * 6,
            "completed_return": [0.01, 0.02, 0.03, -0.01, -0.02, 0.01],
            "final_two_hour_return": [0.01, 0.01, 0.02, -0.01, -0.01, 0.01],
            "feature_available_time": pd.date_range(
                "2024-01-01", periods=6, freq="8h", tz="UTC"
            ),
        }
    )


def test_primary_onset_and_side():
    active, side, _ = s.active(panel(), "primary")
    assert active.tolist() == [False, True, False, False, True, False]
    assert side[active].tolist() == [1, -1]


def test_controls_are_diagnostic():
    frame = panel()
    frame.loc[1, "variation_rank"] = 0.4
    assert s.active(frame, "no_variation_gate")[0].iloc[1]
    frame = panel()
    frame.loc[1, "elasticity_rank"] = 0.4
    assert s.active(frame, "aggregate_average_ticket_tail")[0].iloc[1]


def test_directional_agreement_is_required():
    frame = panel()
    frame.loc[1, "final_two_hour_return"] = -0.01
    assert not s.active(frame, "primary")[0].iloc[1]


def test_prior_rank_excludes_current():
    ranks = s.prior_rank(pd.Series(np.arange(181, dtype=float)))
    assert np.isnan(ranks.iloc[179])
    assert ranks.iloc[180] == 1.0
