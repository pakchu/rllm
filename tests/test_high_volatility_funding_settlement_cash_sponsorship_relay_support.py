import numpy as np
import pandas as pd

from training import build_high_volatility_funding_settlement_cash_sponsorship_relay_support as support


def test_rank_excludes_current_and_requires_252_prior():
    ranked = support.rank(pd.Series(np.arange(253, dtype=float)))
    assert ranked.iloc[:252].isna().all()
    assert ranked.iloc[252] == 1.0


def test_features_accepts_mixed_iso_timestamp_precision(tmp_path, monkeypatch):
    panel = pd.DataFrame(
        {
            "settlement_time": ["2022-12-01 00:00:00.000000+00:00", "2022-12-01 08:00:00+00:00"],
            "decision_time": ["2022-12-01 01:00:00.000000+00:00", "2022-12-01 09:00:00+00:00"],
            "source_valid": [True, True], "funding_rate": [0.1, -0.1],
            "funding_rank": [0.8, 0.8], "pre_settlement_return": [0.1, -0.1],
            "pre_settlement_variation": [0.2, 0.2], "variation_rank": [0.8, 0.8],
            "spot_return": [0.1, -0.1], "spot_aggressive_quote_flow": [1.0, -1.0],
        }
    )
    path = tmp_path / "panel.csv.gz"
    panel.to_csv(path, index=False, compression="gzip")
    monkeypatch.setattr(support, "PANEL", path)
    loaded = support.features()
    assert str(loaded.settlement_time.dtype) == "datetime64[ns, UTC]"
    assert loaded.decision_time.notna().all()


def _frame(**updates):
    row = {
        "settlement_time": pd.Timestamp("2023-07-01T00:00:00Z"),
        "decision_time": pd.Timestamp("2023-07-01T01:00:00Z"),
        "source_valid": True, "funding_rate": 0.001, "funding_rank": 0.8,
        "pre_settlement_return": 0.02, "pre_settlement_variation": 0.03,
        "variation_rank": 0.8, "spot_return": 0.003, "spot_aggressive_quote_flow": 100.0,
    }
    row.update(updates)
    return pd.DataFrame([row])


def test_primary_requires_same_sign_cash_confirmation():
    primary = support.build_clock(_frame())
    assert primary.side.tolist() == [1]
    assert primary.entry_time.iloc[0] == pd.Timestamp("2023-07-01T01:05:00Z")
    assert support.build_clock(_frame(spot_aggressive_quote_flow=-100.0)).empty


def test_direction_and_forced_long_controls_are_diagnostic():
    short = _frame(funding_rate=-0.001, pre_settlement_return=-0.02, spot_return=-0.003, spot_aggressive_quote_flow=-100.0)
    assert support.build_clock(short).side.tolist() == [-1]
    assert support.build_clock(short, "direction_flip").side.tolist() == [1]
    assert support.build_clock(short, "same_clock_forced_long").side.tolist() == [1]


def test_named_gate_controls_do_not_modify_primary():
    low_funding = _frame(funding_rank=0.2)
    assert support.build_clock(low_funding).empty
    assert support.build_clock(low_funding, "no_funding_tail").side.tolist() == [1]
    low_variation = _frame(variation_rank=0.2)
    assert support.build_clock(low_variation).empty
    assert support.build_clock(low_variation, "no_variation_gate").side.tolist() == [1]
