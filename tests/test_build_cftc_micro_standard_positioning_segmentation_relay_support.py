import numpy as np,pandas as pd
from training import build_cftc_micro_standard_positioning_segmentation_relay_support as s
def test_cross_contract_segmentation_and_clock():
 d=pd.date_range("2023-07-04",periods=3,freq="7D",tz="UTC");c=pd.DataFrame({"report_date":d,"standard_share":[.1,.15,.14],"micro_share":[-.1,-.08,-.12]});v=pd.DataFrame({"feature_available_time":d+pd.Timedelta(days=7,minutes=5),"btc_variation":[1.]*3,"btc_variation_rank":[.7]*3});x=s.score_states(c,v);assert np.isnan(x.segmentation.iloc[0]);assert np.allclose(x.segmentation.iloc[1:],[.03,.03]);clock=s.build_clock(x);assert clock.side.tolist()==[1,1];assert (clock.exit_time-clock.entry_time).eq(pd.Timedelta(hours=168)).all()
def test_outcome_blind_columns():assert not {"pnl","funding","execution_price","gross9"}.intersection({x.lower() for x in s.COLUMNS})
