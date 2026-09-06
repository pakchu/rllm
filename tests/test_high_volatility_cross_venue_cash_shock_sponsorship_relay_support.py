import pandas as pd
from training import build_high_volatility_cross_venue_cash_shock_sponsorship_relay_support as s
def test_causal_rank_excludes_current(monkeypatch):
 monkeypatch.setitem(s.P,'variation_history_decisions',3);monkeypatch.setitem(s.P,'minimum_variation_history_decisions',2);x=s.causal(pd.Series([1.,3.,2.]));assert pd.isna(x.iloc[0]) and pd.isna(x.iloc[1]) and x.iloc[2]==.5
def test_pinned_registration():assert s.PREREG_SHA=="da962421c683267355689aebca4b9da12fffa861a0ac6c8cbe8162f906f0ea66"
