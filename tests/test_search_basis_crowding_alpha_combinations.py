import numpy as np,pandas as pd
from training import search_basis_crowding_alpha_combinations as s

def test_completed_premium_close_is_causal():
 index=pd.date_range('2023-02-01',periods=100,freq='1h');x=pd.DataFrame(index=index);x['funding']=0.
 # The actual loader uses causal close_time through normalise_premium_index_frame.
 raw=pd.DataFrame({'date':index-pd.Timedelta('1h'),'close_time':((index-pd.Timedelta('1ms')).view('int64')//1_000_000),'close':np.arange(100.)})
 normal=s.normalise_premium_index_frame(raw)
 joined=pd.merge_asof(pd.DataFrame({'date':index}),normal.rename(columns={'date':'available'}),left_on='date',right_on='available',direction='backward')
 assert (joined.available<=joined.date).all()
 assert joined.premium_index.iloc[-1]==99

def test_design_has_distinct_crowding_mechanisms():
 assert 'crowded-long unwind' in s.DESIGN['mechanisms']
 assert 'discount recovery' in s.DESIGN['mechanisms']
 assert s.DESIGN['no_frequency_or_fee_ratio_gate']
