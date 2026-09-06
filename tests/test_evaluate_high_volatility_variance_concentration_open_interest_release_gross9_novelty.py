from __future__ import annotations
import pandas as pd
from training import evaluate_high_volatility_variance_concentration_open_interest_release_gross9_novelty as n

def _clock(t,s):
    e=pd.to_datetime(t,utc=True);return pd.DataFrame({'entry_time':e,'exit_time':e+pd.Timedelta(hours=12),'side':s})

def test_metric_limits_and_collision() -> None:
    assert n.LIMITS=={'exact_entry_jaccard':.1,'one_to_one_6h_max_matched_share':.35,'occupied_5m_bar_jaccard':.25,'absolute_signed_exposure_pearson':.35}
    c=_clock(['2023-07-01T00:05:00Z'],[1]);assert n.pair(c,c.copy())['passed'] is False
    assert n.pair(c,_clock(['2023-07-03T00:05:00Z'],[-1]))['passed'] is True

def test_frozen_hashes_and_determinism(tmp_path) -> None:
    assert n.sha(n.PREREG)==n.PREREG_SHA and n.sha(n.SUPPORT)==n.SUPPORT_SHA and n.sha(n.CLOCK)==n.CLOCK_SHA
    out=tmp_path/'n.json';a=n.run(out);raw=out.read_bytes();b=n.run(out)
    assert out.read_bytes()==raw and a==b
    assert a['evidence_boundary']['outcomes_opened'] is False
    assert a['advance_to_economic_outcomes']==all(x['passed'] for x in a['gross9_sleeves'].values())
