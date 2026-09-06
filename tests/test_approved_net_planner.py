import copy
import pytest
from execution import approved_net_planner as p


def approval():
    return {'portfolio_selection_approved':True,'weights_notional':{'long':1.,'dollar_rally_short':.5,'macro_flow':1.},
            'risk_contract':{'net_cap_at_open_rebalance_after_fees':4.5}}


def snapshot(net=0,price=100):
    return {'symbol':'BTCUSDT','position_mode':'one_way','asof':'2026-09-04T00:00:00Z','equity':1000.,'mark_price':price,'net_units':net}


def signal(name,side):
    return {'name':name,'kind':'entry','active':True,'side':side,'hold_bars':144,'signal_id':name+':1','execution_time':'2026-09-04T00:00:00Z'}


def test_opposing_entries_net_before_orders_and_costs():
    a=approval();state=p.empty_state(p.digest(a));out=p.plan(a,state,snapshot(),[signal('long','LONG'),signal('dollar_rally_short','SHORT')],execution_time=snapshot()['asof'])
    assert out['target_net_units']==5
    assert len(out['order_plan'])==1 and out['order_plan'][0]['quantity']==5
    assert out['estimated_fee']==pytest.approx(.3)
    assert state['revision']==0
    after=p.apply_paper_fill(state,out)
    assert after['revision']==1
    with pytest.raises(ValueError):p.apply_paper_fill(after,out)


def test_hedge_mode_and_position_mismatch_fail_closed():
    a=approval();state=p.empty_state(p.digest(a));s=snapshot();s['position_mode']='hedge'
    with pytest.raises(ValueError):p.plan(a,state,s,[],execution_time=s['asof'])
    with pytest.raises(ValueError):p.plan(a,state,snapshot(1),[],execution_time=s['asof'])


def test_stale_signals_and_modified_approval_fail():
    a=approval();state=p.empty_state(p.digest(a));s=snapshot()
    with pytest.raises(ValueError):p.plan(a,state,s,[],execution_time='2026-09-04T00:01:00Z')
    a['weights_notional']['long']=2
    with pytest.raises(ValueError):p.plan(a,state,s,[],execution_time=s['asof'])


def test_flip_closes_before_opening_opposite_side():
    a=approval();state=p.empty_state(p.digest(a))
    state['sleeves']={'macro_flow':{'units':2.,'entry_price':100.,'expires_at':None}}
    sig={'name':'macro_flow','kind':'target','target_fraction':-.5,'signal_id':'m','execution_time':snapshot()['asof']}
    out=p.plan(a,state,snapshot(2),[sig],execution_time=snapshot()['asof'])
    assert [x['reduce_only'] for x in out['order_plan']]==[True,False]
    assert [x['quantity'] for x in out['order_plan']]==[2,5]


def test_target_update_preserves_other_sleeve_units():
    a=approval();state=p.empty_state(p.digest(a));state['sleeves']={'long':{'units':1.,'entry_price':100.,'expires_at':None}}
    sig={'name':'macro_flow','kind':'target','target_fraction':.2,'signal_id':'m','execution_time':snapshot()['asof']}
    out=p.plan(a,state,snapshot(1),[sig],execution_time=snapshot()['asof'])
    assert out['proposed_sleeves']['long']['units']==1


def test_time_exit_and_observed_price_barrier():
    a=approval();state=p.empty_state(p.digest(a));state['sleeves']={'long':{'units':1.,'entry_price':100.,'expires_at':'2026-09-05','barrier_exit':{'stop_bps':100}}}
    out=p.plan(a,state,snapshot(1,98),[],execution_time=snapshot()['asof'])
    assert out['target_net_units']==0 and out['order_plan'][0]['reduce_only']
    assert out['actions'][0]['observed_price']==98


def test_plan_tampering_rejected():
    a=approval();state=p.empty_state(p.digest(a));out=p.plan(a,state,snapshot(),[],execution_time=snapshot()['asof'])
    changed=copy.deepcopy(out);changed['target_net_units']=100
    with pytest.raises(ValueError):p.apply_paper_fill(state,changed)


def test_same_macro_slot_id_is_not_rebalanced_twice():
    a=approval();state=p.empty_state(p.digest(a));s=snapshot()
    sig={'name':'macro_flow','kind':'target','target_fraction':.2,'signal_id':'slot1','execution_time':s['asof']}
    out=p.plan(a,state,s,[sig],execution_time=s['asof']);after=p.apply_paper_fill(state,out)
    s2=snapshot(out['target_net_units']);s2['asof']='2026-09-04T00:00:01Z'
    sig['execution_time']=s2['asof'];second=p.plan(a,after,s2,[sig],execution_time=s2['asof'])
    assert second['order_plan']==[]


def test_net_cap_uses_post_fee_equity():
    a=approval();a['weights_notional']['long']=5.
    state=p.empty_state(p.digest(a));s=snapshot()
    out=p.plan(a,state,s,[signal('long','LONG')],execution_time=s['asof'])
    assert abs(out['target_net_units']*s['mark_price'])/(s['equity']-out['estimated_fee'])<=4.5+1e-9


def test_ambiguous_signal_booleans_rejected():
    a=approval();state=p.empty_state(p.digest(a));sig=signal('long','LONG');sig['active']='false'
    with pytest.raises(ValueError):p.plan(a,state,snapshot(),[sig],execution_time=snapshot()['asof'])


def test_macro_a_b_a_replay_is_ignored():
    a=approval();state=p.empty_state(p.digest(a));net=0
    for i,ident,fraction in [(0,'A',.2),(1,'B',.3),(2,'A',.2)]:
        s=snapshot(net);s['asof']=f'2026-09-04T00:00:0{i}Z'
        sig={'name':'macro_flow','kind':'target','target_fraction':fraction,'signal_id':ident,'execution_time':s['asof']}
        out=p.plan(a,state,s,[sig],execution_time=s['asof'])
        if i==2:assert out['order_plan']==[]
        state=p.apply_paper_fill(state,out);net=out['target_net_units']
    assert state['processed_signal_ids']['macro_flow']==['A','B']


def test_entry_a_b_a_replay_after_exits_is_ignored():
    a=approval();state=p.empty_state(p.digest(a));net=0
    for minute,ident in [(0,'A'),(5,None),(10,'B'),(15,None),(20,'A')]:
        s=snapshot(net);s['asof']=f'2026-09-04T00:{minute:02}:00Z'
        signals=[]
        if ident:
            sig=signal('long','LONG');sig.update(signal_id=ident,hold_bars=1,execution_time=s['asof']);signals=[sig]
        out=p.plan(a,state,s,signals,execution_time=s['asof'])
        if minute==20:assert out['order_plan']==[] and out['target_net_units']==0
        state=p.apply_paper_fill(state,out);net=out['target_net_units']
    assert state['processed_signal_ids']['long']==['A','B']


@pytest.mark.parametrize('inactive_field,inactive_value',[('active',False),('ready',False)])
def test_no_trade_decision_cannot_be_replayed_as_active(inactive_field,inactive_value):
    a=approval();state=p.empty_state(p.digest(a));s=snapshot();sig=signal('long','LONG');sig['signal_id']='A';sig[inactive_field]=inactive_value
    out=p.plan(a,state,s,[sig],execution_time=s['asof']);state=p.apply_paper_fill(state,out)
    assert out['order_plan']==[] and state['processed_signal_ids']['long']==['A']
    s['asof']='2026-09-04T00:05:00Z';sig.update(active=True,ready=True,execution_time=s['asof'])
    replay=p.plan(a,state,s,[sig],execution_time=s['asof'])
    assert replay['order_plan']==[]
