import pandas as pd
from pathlib import Path
from training import evaluate_high_volatility_execution_count_elasticity_relay_gross9_novelty as n
def c(t,s):
 e=pd.to_datetime(t,utc=True);return pd.DataFrame({'entry_time':e,'exit_time':e+pd.Timedelta('8h'),'side':s})
def test_metrics_and_binding():
 a=c(['2024-01-01','2024-02-01'],[1,-1]);b=c(['2024-01-15','2024-02-15'],[-1,1]);assert n.evaluate_pair(a,b)['passed'];assert not n.evaluate_pair(a,a)['passed']
 assert n.POLICY=='HVECE-8' and n.sha(n.PREREG)==n.PREREG_SHA and n.sha(n.SUPPORT)==n.SUPPORT_SHA and n.sha(n.CLOCK)==n.CLOCK_SHA
 assert 'bars_binance' not in Path(n.__file__).read_text()
