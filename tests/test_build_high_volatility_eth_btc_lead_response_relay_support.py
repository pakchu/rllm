import numpy as np
import pandas as pd

from training import build_high_volatility_eth_btc_lead_response_relay_support as s


def test_lead_statistics_pairs_eth_with_next_btc_return():
    eth_returns = np.linspace(-0.01, 0.01, 96)
    btc_returns = np.r_[0.0, eth_returns[:-1]]
    btc = pd.DataFrame(
        {
            "open": np.repeat(100.0, 480),
            "close": np.repeat(100 * np.exp(btc_returns), 5),
        }
    )
    eth = pd.DataFrame(
        {
            "open": np.repeat(50.0, 480),
            "close": np.repeat(50 * np.exp(eth_returns), 5),
        }
    )
    lead, contemporaneous, variation, completed = s.lead_statistics(btc, eth)
    assert lead > 0.99
    assert np.isfinite(contemporaneous)
    assert variation > 0
    assert np.isfinite(completed)


def panel():
    return pd.DataFrame(
        {
            "source_valid": [True] * 6,
            "lead_response": [0.1, 0.3, 0.4, -0.2, -0.5, 0.4],
            "response_strength_rank": [0.5, 0.8, 0.9, 0.4, 0.8, 0.9],
            "contemporaneous_response": [0.2, 0.1, 0.2, -0.3, -0.4, 0.2],
            "contemporaneous_strength_rank": [0.5, 0.4, 0.6, 0.8, 0.9, 0.6],
            "variation_rank": [0.8] * 6,
            "eth_completed_return": [0.01, 0.02, 0.03, -0.01, -0.02, 0.01],
            "feature_available_time": pd.date_range(
                "2024-01-01", periods=6, freq="8h", tz="UTC"
            ),
        }
    )


def test_primary_onset_and_mapped_side():
    active, side, _ = s.active(panel(), "primary")
    assert active.tolist() == [False, True, False, False, True, False]
    assert side[active].tolist() == [1, 1]


def test_controls_are_diagnostic():
    frame = panel()
    frame.loc[1, "variation_rank"] = 0.4
    assert s.active(frame, "no_variation_gate")[0].iloc[1]
    active, side, _ = s.active(panel(), "direction_flip")
    assert side[active].tolist() == [-1, -1]


def test_prior_rank_excludes_current():
    ranks = s.prior_rank(pd.Series(np.arange(181, dtype=float)))
    assert np.isnan(ranks.iloc[179])
    assert ranks.iloc[180] == 1.0
