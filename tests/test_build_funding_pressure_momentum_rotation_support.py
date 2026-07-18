from __future__ import annotations

import pandas as pd
import pytest

from training import build_funding_pressure_momentum_rotation_support as support


def test_cross_sectional_z_is_centered_and_scaled() -> None:
    values = pd.Series([1.0, 2.0, 3.0], index=["A", "B", "C"])
    result = support._cross_sectional_z(values)
    assert result.mean() == pytest.approx(0.0)
    assert result.std(ddof=0) == pytest.approx(1.0)


def test_pair_is_lexically_deterministic_and_beta_neutral() -> None:
    values = pd.Series({"B": 1.0, "A": 1.0, "C": -1.0})
    beta = pd.Series({"A": 0.5, "B": 1.0, "C": 2.0})
    pair = support._pair(values, beta)
    assert pair["long_symbol"] == "A"
    assert pair["short_symbol"] == "C"
    assert pair["long_weight"] + pair["short_weight_abs"] == pytest.approx(1.0)
    assert pair["long_weight"] * beta["A"] == pytest.approx(
        pair["short_weight_abs"] * beta["C"]
    )


def test_clock_contract_rejects_outcome_column() -> None:
    frame = pd.DataFrame(columns=[*support.CLOCK_COLUMNS, "pnl"])
    with pytest.raises(RuntimeError, match="schema changed"):
        support.assert_clock_contract(frame)


def test_frozen_support_artifact_passes() -> None:
    payload = support.run()
    assert payload["outcomes_opened"] is False
    assert payload["post_entry_returns_or_pnl_calculated"] is False
    assert payload["support"]["passes_support"] is True
    assert payload["support"]["events"] >= 90
    assert payload["support"]["unique_ordered_pairs"] >= 15
