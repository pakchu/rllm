from __future__ import annotations

import numpy as np
import pandas as pd

from training import build_high_volatility_spot_led_adverse_funding_continuation_support as support


def test_causal_midrank_excludes_current_and_requires_history() -> None:
    values = pd.Series(np.arange(181, dtype=float))
    ranked = support.causal_midrank(values)
    assert ranked.iloc[:180].isna().all()
    assert ranked.iloc[180] == 1.0


def _panel(**updates) -> pd.DataFrame:
    row = {
        "decision_time": pd.Timestamp("2023-07-01T00:00:00Z"),
        "feature_available_time": pd.Timestamp("2023-07-01T00:00:00Z"),
        "source_valid": True,
        "funding_rate": -0.001,
        "perpetual_return": 0.02,
        "spot_return": 0.03,
        "spot_flow_share": 0.1,
        "realized_variation": 0.04,
        "variation_rank": 0.8,
        "eligible": True,
    }
    row.update(updates)
    return pd.DataFrame([row])


def test_clock_uses_spot_side_and_fixed_timing() -> None:
    clock = support.build_clock(_panel())
    assert clock.side.tolist() == [1]
    assert clock.entry_time.iloc[0] == pd.Timestamp("2023-07-01T00:05:00Z")
    assert clock.exit_time.iloc[0] == pd.Timestamp("2023-07-01T08:05:00Z")


def test_build_panel_requires_all_frozen_conditions(monkeypatch) -> None:
    funding = pd.DataFrame({"decision_time": [pd.Timestamp("2023-07-01T00:00:00Z")], "funding_rate": [-0.001]})
    perpetual = pd.DataFrame({"perpetual_return": [0.02], "realized_variation": [0.04], "perpetual_valid": [True]}, index=pd.DatetimeIndex([pd.Timestamp("2023-07-01T00:00:00Z")], name="decision_time"))
    spot = pd.DataFrame({"spot_return": [0.03], "spot_flow_share": [0.1], "spot_valid": [True]}, index=pd.DatetimeIndex([pd.Timestamp("2023-07-01T00:00:00Z")], name="decision_time"))
    monkeypatch.setattr(support, "prepare", lambda raw: (funding, perpetual, spot))
    monkeypatch.setattr(support, "causal_midrank", lambda series: pd.Series([0.8], index=series.index))
    assert support.build_panel((funding, perpetual, spot)).eligible.tolist() == [True]
    funding["funding_rate"] = 0.001
    assert support.build_panel((funding, perpetual, spot)).eligible.tolist() == [False]
    funding["funding_rate"] = -0.001
    spot["spot_flow_share"] = -0.1
    assert support.build_panel((funding, perpetual, spot)).eligible.tolist() == [False]
