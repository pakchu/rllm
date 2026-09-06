import json
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest
from execution import approved_portfolio_dry_run as r
from execution import approved_net_planner as planner


def test_prepare_builds_net_plan_without_transport(monkeypatch):
    approval=json.loads(r.DEFAULT_APPROVAL.read_text());now=pd.Timestamp('2026-06-17T19:05:00Z')
    market=pd.DataFrame({'date':pd.date_range(end=now-pd.Timedelta('5min'),periods=18000,freq='5min'),
                         'open':100.,'high':101.,'low':99.,'close':100.})
    signal={'name':'fresh_kimchi_fx','kind':'entry','active':True,'ready':True,'side':'LONG','hold_bars':144,'signal_id':'f1','execution_time':str(now)}
    monkeypatch.setattr(r,'score_g9',lambda *args:([signal],[]))
    monkeypatch.setattr(r.adapters,'build_macro_targets',lambda *args,**kw:pd.Series([.25],index=[now]))
    monkeypatch.setattr(r.adapters,'score_dollar_short',lambda *args,**kw:{'active':False,'execution_time':str(now)})
    state=planner.empty_state(planner.digest(approval))
    snapshot={'symbol':'BTCUSDT','position_mode':'one_way','asof':str(now),'equity':1000.,'mark_price':100.,'net_units':0.}
    out=r.prepare(approval,state,snapshot,market,execution_time=str(now))
    assert out['orders_enabled'] is False and out['production_ready'] is False
    assert out['plan']['target_net_units']==12.5
    assert len(out['plan']['order_plan'])==1


def test_live_flag_is_unconditionally_rejected(tmp_path):
    output=tmp_path/'not-created.json'
    result=subprocess.run([sys.executable,'-m','execution.approved_portfolio_dry_run','--market','missing.csv','--snapshot','missing.json','--output',str(output),'--live'],cwd=r.ROOT,capture_output=True,text=True)
    assert result.returncode!=0 and 'intentionally unavailable' in result.stderr
    assert not output.exists()


def test_outputs_are_exclusive(tmp_path):
    path=tmp_path/'paper.json';r.write_new(path,{'mode':'paper'})
    with pytest.raises(FileExistsError):r.write_new(path,{'mode':'live'})
    assert json.loads(path.read_text())=={'mode':'paper'}


def test_real_snapshot_macro_matches_research_after_warmup():
    from training import search_meaningful_alpha_combinations as base
    from training import search_macro_flow_alpha_combinations as macro
    from training import evaluate_macro_flow_fixed_fresh as frozen
    path=r.ROOT/'research/g9_september_inputs/raw_enriched_cache.pkl'
    if path.exists():
        market=pd.read_pickle(path).tail(20000).reset_index(drop=True)
    else:
        path=r.ROOT/'research/approved_runtime_preparation/macro_active_market.csv.gz'
        if not path.exists():pytest.skip('Source snapshot fixture unavailable')
        market=pd.read_csv(path).tail(20000).reset_index(drop=True)
    dates=pd.to_datetime(market.date,utc=True);asof=dates.iloc[-1]+pd.Timedelta('5min')
    actual=r.adapters.build_macro_targets(market,asof=asof)
    calc=market.copy();calc.date=dates.dt.tz_convert(None)
    funding=pd.DataFrame({'date':[calc.date.iloc[0]],'funding_rate':[0.]})
    x=base.features(calc,funding)
    cols=['date','dxy','usdkrw','kimchi_premium','dxy_available','usdkrw_available','kimchi_available']
    x=pd.concat([x,macro.macro_features(calc[cols],x.index)],axis=1)
    expected,_=frozen.fixed_positions(x)
    expected=pd.Series(expected['dollar_flow_plus_regime_switch'],index=(x.index+pd.Timedelta('5min')).tz_localize('UTC'))
    common=actual.index.intersection(expected.index)
    common=common[common>=dates.iloc[0]+pd.Timedelta(days=33)]
    assert len(common)>24
    np.testing.assert_allclose(actual.loc[common],expected.loc[common],rtol=0,atol=1e-12)


def test_altered_weight_record_rejected():
    approval=json.loads(r.DEFAULT_APPROVAL.read_text());approval['weights_notional']['macro_flow']=2
    with pytest.raises(ValueError):r.validate_approval(approval)


def test_approval_cannot_disable_evidence_checks():
    approval=json.loads(r.DEFAULT_APPROVAL.read_text());approval['evidence']={}
    with pytest.raises(ValueError):r.validate_approval(approval)
