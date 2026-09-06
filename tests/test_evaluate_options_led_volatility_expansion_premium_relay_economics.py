from __future__ import annotations
import hashlib,json
import numpy as np,pandas as pd
from training import evaluate_options_led_volatility_expansion_premium_relay_economics as e

def synthetic():
 start=pd.Timestamp('2023-07-01T00:00:00Z');end=start+pd.Timedelta(hours=48);dates=pd.date_range(start,end,freq='5min',inclusive='both');price=np.linspace(100,104,len(dates));m=pd.DataFrame({'date':dates,'open':price,'high':price*1.001,'low':price*.999,'close':price});f=pd.DataFrame({'date':[start+pd.Timedelta(hours=8),start+pd.Timedelta(hours=16)],'funding_rate':[.0001,.0001]});clock=pd.DataFrame({'entry_time':[start+pd.Timedelta(minutes=5)],'exit_time':[start+pd.Timedelta(hours=24,minutes=5)],'side':[1]});return start,end,m,f,clock

def test_synthetic_exact_trade_and_strict_metrics_are_positive()->None:
 start,end,m,f,clock=synthetic();e.validate_market(m,start,end);trades=e.build_trades(clock,m,f);assert len(trades)==1 and trades[0].side==1;result=e.metrics(trades,m,f,start,end,e.BASE_COST);assert result['absolute_return_pct']>0;assert result['strict_mdd_pct']>=0;assert result['trades']==1

def test_cluster_signflip_is_seed_deterministic()->None:
 _,_,m,f,clock=synthetic();t=e.build_trades(clock,m,f);assert e.cluster_p(t,e.BASE_COST)==e.cluster_p(t,e.BASE_COST)

def test_frozen_predecessors_authorize_only_train_without_opening_prices()->None:
 n=e.verify_controls('train');assert n['advance_to_economic_outcomes'] is True
