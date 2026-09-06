import numpy as np,pandas as pd
from training import build_high_volatility_directional_ticket_size_asymmetry_relay_support as s
def test_ticket_asymmetry():
 r=np.r_[np.ones(240)*.001,np.ones(240)*-.001];q=np.r_[np.ones(240)*200,np.ones(240)*50];c=np.ones(480)*10;up,dn,mu,md,a,m,v=s.ticket_statistics(r,q,c);assert up==dn==240 and mu>md and a>0 and v>0
def test_prepare():
 f=pd.DataFrame({"ts":["2023-01-01T00:00:00Z"],"open":[100.],"high":[101.],"low":[99.],"close":[100.5],"quote_asset_volume":[10.],"number_of_trades":[3.]});assert bool(s.prepare(f).row_valid.iloc[0]);f.loc[0,"number_of_trades"]=0;assert not bool(s.prepare(f).row_valid.iloc[0])
def test_blind_bound():assert not {x.lower() for x in (*s.PANEL_COLUMNS,*s.CLOCK_COLUMNS)}.intersection({"pnl","funding","execution_price","gross9"}) and s.sha(s.prereg.DEFAULT_OUTPUT)==s.PREREG_SHA
