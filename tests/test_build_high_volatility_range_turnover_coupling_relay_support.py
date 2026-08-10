import numpy as np
import pandas as pd

from training import build_high_volatility_range_turnover_coupling_relay_support as s


def test_coupling_statistics_correlates_range_energy_and_turnover():
    quote = np.linspace(1.0, 96.0, 96)
    energy = np.log1p(quote)
    log_range = np.sqrt(4 * np.log(2) * energy)
    high = 100 * np.exp(log_range / 2)
    low = 100 * np.exp(-log_range / 2)
    returns = np.linspace(-0.002, 0.003, 96)
    block = pd.DataFrame(
        {
            "open": np.repeat(100.0, 480),
            "high": np.repeat(high, 5),
            "low": np.repeat(low, 5),
            "close": np.repeat(100 * np.exp(returns), 5),
            "quote_asset_volume": np.repeat(quote / 5, 5),
        }
    )
    coupling, range_direction, variation, completed = s.coupling_statistics(block)
    assert coupling > 0.99
    assert np.isfinite(range_direction)
    assert variation > 0
    assert np.isfinite(completed)


def panel():
    return pd.DataFrame(
        {
            "source_valid": [True] * 6,
            "range_turnover_coupling": [0.1, 0.3, 0.4, 0.2, 0.5, 0.4],
            "coupling_rank": [0.5, 0.8, 0.9, 0.4, 0.8, 0.9],
            "range_only_direction": [0.2, -0.1, 0.2, -0.3, 0.4, 0.2],
            "variation_rank": [0.8] * 6,
            "completed_return": [0.01, 0.02, 0.03, -0.01, -0.02, 0.01],
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
    active, side, _ = s.active(panel(), "range_only_direction")
    assert side[active].tolist() == [-1, 1]


def test_prior_rank_excludes_current():
    ranks = s.prior_rank(pd.Series(np.arange(181, dtype=float)))
    assert np.isnan(ranks.iloc[179])
    assert ranks.iloc[180] == 1.0
