import pandas as pd

from training.build_high_volatility_adverse_funding_contraction_relay_support import conditions


def test_primary_requires_adverse_move_and_oi_contraction() -> None:
    frame = pd.DataFrame(
        {
            "source_valid": [True, True, True],
            "post_settlement_return": [-0.01, 0.01, -0.01],
            "funding_rate": [0.0001, 0.0001, 0.0001],
            "absolute_funding_rank": [0.8, 0.8, 0.8],
            "oi_return": [-0.02, -0.02, 0.02],
            "variation_rank": [0.8, 0.8, 0.8],
        }
    )
    active, side, _ = conditions(frame)
    assert active.tolist() == [True, False, False]
    assert side.tolist() == [-1, 1, -1]
