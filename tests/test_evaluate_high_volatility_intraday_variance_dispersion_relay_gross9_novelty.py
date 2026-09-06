import pandas as pd
from pathlib import Path
from training import evaluate_high_volatility_intraday_variance_dispersion_relay_gross9_novelty as n

def clock(times,sides):
 e=pd.to_datetime(times,utc=True);return pd.DataFrame({'entry_time':e,'exit_time':e+pd.Timedelta('8h'),'side':sides})
def test_pair_contract():
 a=clock(['2024-01-01','2024-02-01'],[1,-1]);b=clock(['2024-01-15','2024-02-15'],[-1,1])
 assert n.evaluate_pair(a,b)['passed'];assert not n.evaluate_pair(a,a.copy())['passed']
def test_blind_and_bound():
 assert n.POLICY=='HVIVDR-8' and n.sha(n.PREREG)==n.PREREG_SHA and n.sha(n.SUPPORT)==n.SUPPORT_SHA and n.sha(n.CLOCK)==n.CLOCK_SHA
 source=Path(n.__file__).read_text();assert 'bars_binance' not in source and 'funding_rates_binance' not in source and '"outcomes_opened": False' in source
