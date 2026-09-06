import numpy as np
import pandas as pd

from training import build_high_volatility_higuchi_roughness_reversal_support as support


def test_prior_rank_excludes_current_value():
    values = pd.Series(list(range(181)), dtype=float)
    ranks = support.prior_rank(values)
    assert np.isnan(ranks.iloc[179])
    assert ranks.iloc[180] == 1.0


def test_higuchi_dimension_is_finite_and_rougher_for_noise():
    rng = np.random.default_rng(7)
    smooth = np.linspace(0.0, 1.0, 480)
    rough = np.cumsum(rng.normal(size=480))
    assert np.isfinite(support.higuchi_dimension(smooth))
    assert support.higuchi_dimension(rough) > support.higuchi_dimension(smooth)


def test_contract_is_frozen():
    assert support.PREREG_SHA == "508e50e7c9331b39fe8486f98340ac0816651b3be6080dfc421b5e8a751847b3"
    assert "FROM bars_binance" in support.QUERY
    assert support.CONTROLS == (
        "no_roughness_tail",
        "no_variation_gate",
        "raw_dimension_above_one_half",
        "one_block_stale_geometry",
        "direction_flip",
        "forced_long",
    )
