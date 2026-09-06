from __future__ import annotations
import numpy as np
import pandas as pd
from training import build_high_volatility_causal_extreme_displacement_open_interest_reversal_support as s


def test_open_interest_requires_exact_positive_expanding_endpoints() -> None:
    times = pd.to_datetime(["2023-07-01T00:00:00Z", "2023-07-01T04:00:00Z"])
    got = s.prepare_open_interest(pd.DataFrame({"ts": times, "sum_open_interest": [100.0, 110.0]}))
    assert got.loc[times[0]] == 100.0 and got.loc[times[1]] == 110.0
    bad = pd.DataFrame({"ts": times, "sum_open_interest": [100.0, -1.0]})
    try:
        s.prepare_open_interest(bad)
    except RuntimeError:
        pass
    else:
        raise AssertionError("negative source OI accepted")


def test_panel_adds_only_exact_current_auction_oi_gate(monkeypatch) -> None:
    decisions = pd.to_datetime(["2023-07-01T04:00:00Z", "2023-07-01T08:00:00Z", "2023-07-01T12:00:00Z"])
    base_panel = pd.DataFrame({column: [np.nan] * 3 for column in s.base.PANEL_COLUMNS})
    base_panel["decision_time"] = decisions; base_panel["feature_available_time"] = decisions
    base_panel["source_valid"] = True; base_panel["eligible"] = True
    base_panel["reversal_side"] = [1, -1, 1]
    monkeypatch.setattr(s.base, "build_panel", lambda _: base_panel.copy())
    oi = pd.DataFrame({
        "ts": pd.to_datetime(["2023-07-01T00:00:00Z", "2023-07-01T04:00:00Z", "2023-07-01T08:00:00Z", "2023-07-01T12:00:00Z"]),
        "sum_open_interest": [100.0, 110.0, 105.0, 0.0],
    })
    panel = s.build_panel((pd.DataFrame(), oi))
    assert panel["eligible"].tolist() == [True, False, False]
    assert np.isclose(panel.loc[0, "oi_change"], np.log(1.1))
    assert panel.loc[1, "oi_change"] < 0 and np.isnan(panel.loc[2, "oi_change"])


def test_preregistration_is_hash_bound() -> None:
    assert s.sha256_file(s.prereg.DEFAULT_OUTPUT) == s.PREREG_SHA
    s.prereg.validate(s.REGISTRATION)
