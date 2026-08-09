import numpy as np
import pandas as pd
from training import build_high_volatility_small_ticket_exhaustion_reversal_support as support

TH={"execution_count_q75":100.,"average_ticket_q35":20.,"absolute_block_return_q60":.01,"range_vol_q65":.03}
def scores():
 return pd.DataFrame({"execution_count":[120.,130.,140.,150.],"average_ticket":[10.,15.,30.,10.],"block_return":[.02,-.03,.04,.02],"range_vol":[.04,.05,.06,.02]})
def test_primary_fades_eligible_small_ticket_move_and_onsets():
 a,s=support.conditions(scores(),TH);assert a.tolist()==[True,True,False,False];assert s[a].tolist()==[-1,1]
def test_controls_do_not_promote_and_flip_only_side():
 assert support.CONTROLS==("no_volatility_gate","count_tail_only","ticket_tail_only","one_block_stale_participation","direction_flip")
 a,s=support.conditions(scores(),TH,"direction_flip");assert a.tolist()==[True,True,False,False];assert s[a].tolist()==[1,-1]
def test_score_anchor_uses_count_and_quote_notional():
 idx=pd.date_range('2023-01-01',periods=144,freq='5min',tz='UTC');w=pd.DataFrame({'date':idx,'open':100.,'high':101.,'low':99.,'close':100.,'quote_asset_volume':1000.,'number_of_trades':100.});r=support._score_anchor(w);assert r['execution_count']==7200.;assert r['average_ticket']==10.;assert r['block_return']==0.
def test_source_binding_and_sealed_fields():
 assert support.sha256(support.prereg.DEFAULT_OUTPUT)==support.PREREG_SHA;s=support.Path(support.__file__).read_text();assert '"postentry_return_pnl_execution_price_opened": False' in s;assert '"gross9_rows_opened": False' in s
