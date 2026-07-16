from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import training.build_alt_funding_carry_harvest_support as support


def synthetic_inputs(high_sum: float = 0.008, low_sum: float = -0.004):
    symbols = list(sorted(support.SYMBOLS))
    index = pd.date_range("2023-01-01 01:00", "2025-12-31 00:00", freq="1h")
    beta = pd.DataFrame(1.0, index=index, columns=symbols)
    quality = pd.DataFrame(True, index=index, columns=symbols)
    funding = {}
    event_index = pd.date_range("2023-01-01", "2025-12-31", freq="8h")
    for symbol in symbols:
        value = 0.0
        if symbol == "ETHUSDT":
            value = high_sum / 84.0
        elif symbol == "ADAUSDT":
            value = low_sum / 84.0
        funding[symbol] = pd.Series(value, index=event_index, name=symbol)
    return beta, quality, funding


def test_clock_longs_low_funding_and_shorts_high_funding() -> None:
    beta, quality, funding = synthetic_inputs()
    clock = support.build_clock(beta, quality, funding)
    assert not clock.empty
    row = clock.iloc[0]
    assert row["long_symbol"] == "ADAUSDT"
    assert row["short_symbol"] == "ETHUSDT"
    assert row["long_weight_norm"] == pytest.approx(0.5)
    assert row["short_weight_norm"] == pytest.approx(0.5)
    assert row["projected_28d_carry"] == pytest.approx(0.006)
    support.assert_clock_contract(clock)


def test_clock_rejects_projected_carry_below_hurdle() -> None:
    beta, quality, funding = synthetic_inputs(high_sum=0.001, low_sum=0.0)
    assert support.build_clock(beta, quality, funding).empty


def test_beta_neutral_weights_use_opposite_leg_beta() -> None:
    beta, quality, funding = synthetic_inputs()
    beta.loc[:, "ETHUSDT"] = 1.5
    beta.loc[:, "ADAUSDT"] = 1.0
    clock = support.build_clock(beta, quality, funding)
    row = clock.iloc[0]
    assert row["long_weight_norm"] == pytest.approx(0.6)
    assert row["short_weight_norm"] == pytest.approx(0.4)
    exposure = row["long_weight_norm"] * row["long_beta"] - row["short_weight_norm"] * row["short_beta"]
    assert exposure == pytest.approx(0.0)


def test_four_week_vintages_never_exceed_four_active_sleeves() -> None:
    beta, quality, funding = synthetic_inputs()
    clock = support.build_clock(beta, quality, funding)
    assert support.max_active_sleeves(clock) == 4


def test_clock_contract_rejects_outcome_column() -> None:
    beta, quality, funding = synthetic_inputs()
    clock = support.build_clock(beta, quality, funding)
    clock["future_return"] = np.nan
    with pytest.raises(RuntimeError, match="outcome-like"):
        support.assert_clock_contract(clock)
