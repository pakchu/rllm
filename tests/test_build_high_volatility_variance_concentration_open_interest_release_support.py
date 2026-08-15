from __future__ import annotations
import numpy as np
import pandas as pd
from training import build_high_volatility_variance_concentration_open_interest_release_support as s

def test_exact_four_hour_oi_expansion_only() -> None:
    decision=pd.Timestamp('2023-07-01T04:10:00Z')
    oi=pd.Series([100.,110.],index=pd.to_datetime(['2023-07-01T00:10:00Z',decision]))
    assert np.isclose(s.oi_expansion_at(oi,decision),np.log(1.1))
    assert s.oi_expansion_at(oi.drop(decision),decision) is None
    falling=pd.Series([110.,100.],index=oi.index)
    assert s.oi_expansion_at(falling,decision) is None

def test_release_requires_oi_and_fades_frozen_shock() -> None:
    times=pd.to_datetime(['2023-07-01T04:00:00Z','2023-07-01T04:05:00Z','2023-07-01T04:10:00Z'])
    scores=pd.DataFrame({'decision_bar_time':times,'concentration':[.2,.95,.5],'high_volatility':[1.,1.,1.],'hour_return':[.01,.02,.01]})
    oi=pd.DataFrame({'ts':pd.to_datetime(['2023-07-01T00:10:00Z','2023-07-01T04:10:00Z']),'sum_open_interest':[100.,110.]})
    thresholds={'high_volatility_q60':.5,'concentration_onset_q90':.9,'concentration_release_q60':.6}
    clock=s.build_clock(scores,thresholds,oi)
    assert len(clock)==1 and clock.side.tolist()==[-1]
    assert clock.entry_time.iloc[0]==pd.Timestamp('2023-07-01T04:15:00Z')
    assert clock.oi_change.iloc[0]>0
    empty=s.build_clock(scores,thresholds,oi.assign(sum_open_interest=[110.,100.]))
    assert empty.empty

def test_preregistration_is_hash_bound() -> None:
    assert s.sha256(s.prereg.DEFAULT_OUTPUT)==s.PREREG_SHA
    s.prereg.validate(s.prereg.build())
