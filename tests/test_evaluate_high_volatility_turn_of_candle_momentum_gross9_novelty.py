import pandas as pd
from training import evaluate_high_volatility_turn_of_candle_momentum_gross9_novelty as novelty
def frame(entries,sides,minutes=30):
 e=pd.to_datetime(entries,utc=True);return pd.DataFrame({'entry_time':e,'exit_time':e+pd.Timedelta(minutes=minutes),'side':sides})
def test_disjoint_pair_passes():
 a=frame(['2024-01-01 23:30','2024-01-03 23:30'],[1,-1]);b=frame(['2024-01-02 10:00','2024-01-04 10:00'],[-1,1]);assert novelty.pair(a,b)['passed']
def test_identical_pair_rejects():
 a=frame(['2024-01-01 23:30','2024-01-03 23:30'],[1,-1]);r=novelty.pair(a,a.copy());assert not r['passed'];assert not r['checks']['exact_entry_jaccard']
