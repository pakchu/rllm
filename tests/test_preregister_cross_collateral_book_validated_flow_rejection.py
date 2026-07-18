from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training import preregister_cross_collateral_book_validated_flow_rejection as cbfr


def test_protocol_is_deterministic_and_keeps_outcomes_sealed() -> None:
    first = cbfr.protocol()
    assert first == cbfr.protocol()
    assert cbfr.canonical_hash(first) == cbfr.canonical_hash(cbfr.protocol())
    assert first["evidence_boundary"]["post_entry_outcomes_opened"] is False
    assert first["support_selection"]["outcomes_used"] is False
    assert first["sequential_oos"]["2024_plus_sealed"] is True


def test_flow_threshold_excludes_current_and_future_values() -> None:
    original = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    changed = original.copy()
    changed.iloc[3:] = [4_000.0, 5_000.0]
    first = cbfr.lagged_flow_threshold(original, quantile=0.5, window=3, minimum=2)
    second = cbfr.lagged_flow_threshold(changed, quantile=0.5, window=3, minimum=2)
    assert first.iloc[3] == second.iloc[3] == 2.0
    assert np.isnan(first.iloc[1])


def test_zero_volume_bar_is_unavailable_instead_of_dividing_by_zero() -> None:
    frame = pd.DataFrame(
        {
            "quote_asset_volume": [100.0, 0.0],
            "taker_buy_quote": [75.0, 0.0],
        }
    )
    flow = cbfr.completed_bar_flow(frame)
    assert flow.iloc[0] == pytest.approx(0.5)
    assert np.isnan(flow.iloc[1])


def test_signal_fades_rejected_flow_and_requires_both_venues() -> None:
    cfg = cbfr.Config(robust_baseline_bars=2, robust_min_periods=2)
    panel = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=5, freq="5min"),
            "flow": [0.4, -0.4, 0.9, -0.9, 0.9],
            "direction": [1, -1, 1, -1, 1],
            "completed_bar_return": [0.0, 0.0, -0.01, 0.01, -0.01],
            "um_defense": [1.0, 1.0, 1.0, 1.0, -1.0],
            "cm_defense": [1.0, 1.0, 1.0, 1.0, 1.0],
            "defense": [1.0, 1.0, 1.0, 1.0, 0.0],
            "clean": [True] * 5,
        }
    )
    frame = pd.DataFrame({"quarantined": [False] * 5})
    signal = cbfr.build_signal(
        panel,
        frame,
        cfg,
        flow_quantile=0.5,
        defense_threshold=0.5,
    )
    assert signal["side"].tolist() == [0, 0, -1, 1, 0]
    assert signal.loc[2, "branch"] == "buy_flow_rejected_by_credible_asks"
    assert signal.loc[3, "branch"] == "sell_flow_rejected_by_credible_bids"


def test_support_selection_uses_no_outcome_tiebreak() -> None:
    cells = [
        {"flow_quantile": 0.85, "defense_threshold": 0.25, "support": {"passes": True}},
        {"flow_quantile": 0.75, "defense_threshold": 0.50, "support": {"passes": True}},
        {"flow_quantile": 0.80, "defense_threshold": 0.50, "support": {"passes": True}},
    ]
    selected = cbfr.select_support_cell(cells)
    assert selected is cells[2]
    with pytest.raises(ValueError, match="forbidden outcome"):
        cbfr.select_support_cell([{**cells[0], "cagr": 1.0}])


def test_fuzzy_overlap_is_one_to_one_and_outcome_free() -> None:
    left = pd.DataFrame(
        {"signal_date": ["2023-01-01 00:00", "2023-01-01 01:00"]}
    )
    right = pd.DataFrame(
        {
            "signal_date": [
                "2023-01-01 00:05",
                "2023-01-01 00:10",
                "2023-01-01 03:00",
            ]
        }
    )
    overlap = cbfr.fuzzy_overlap(left, right, tolerance_bars=2)
    assert overlap["matches"] == 1
    assert overlap["jaccard"] == pytest.approx(0.25)
    assert overlap["new_clock_containment"] == pytest.approx(0.5)
