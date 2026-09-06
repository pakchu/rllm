"""Pure approved-portfolio net order planning. No exchange/network operations.

Plans are not fills. Only the explicit paper-fill helper can advance simulated
state; a production integration must reconcile authoritative actual fills.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
from decimal import Decimal, ROUND_DOWN
from typing import Any
import pandas as pd


def utc(value: Any) -> pd.Timestamp:
    t=pd.Timestamp(value)
    if pd.isna(t):raise ValueError('Invalid timestamp')
    return t.tz_localize('UTC') if t.tzinfo is None else t.tz_convert('UTC')


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def empty_state(approval_hash: str) -> dict:
    return {'version':3,'mode':'paper','revision':0,'approval_hash':approval_hash,'sleeves':{},'dust_units':0.,'last_plan_id':None,'last_execution_time':None,'processed_signals':{},'processed_signal_ids':{}}


def finite(value: Any,label: str,positive: bool=False) -> float:
    x=float(value)
    if not math.isfinite(x) or (positive and x<=0):raise ValueError(f'Invalid {label}')
    return x


def rounded_units(value: float,step: float) -> float:
    sign=1 if value>=0 else -1
    return sign*float((Decimal(str(abs(value)))/Decimal(str(step))).to_integral_value(rounding=ROUND_DOWN)*Decimal(str(step)))


def plan(approval: dict,state: dict,snapshot: dict,signals: list[dict],*,execution_time: str,fee_rate: float=.0006,quantity_step: float=.00001,max_snapshot_age_seconds: float=30.) -> dict:
    """Plan one-way aggregate deltas from signed virtual sleeve quantities.

    Signal contract: name, execution_time, signal_id, kind='entry'|'target',
    active, side for entries, target_fraction for macro, hold_bars and optional
    barrier_exit {take_bps,stop_bps}. Inactive/absent entries never resize holds.
    """
    if not approval.get('portfolio_selection_approved'):raise ValueError('Portfolio choice not approved')
    if 'processed_signal_ids' not in state:raise ValueError('Signal replay history required')
    if state.get('version')!=3 or state.get('mode')!='paper':raise ValueError('Unsupported state; production fills not integrated')
    approval_hash=digest(approval)
    if state.get('approval_hash')!=approval_hash:raise ValueError('Approval/state mismatch')
    now=utc(execution_time)
    if state.get('last_execution_time') and now<=utc(state['last_execution_time']):raise ValueError('Repeated/out-of-order execution time')
    if snapshot.get('symbol')!='BTCUSDT' or snapshot.get('position_mode')!='one_way':raise ValueError('One-way BTCUSDT snapshot required; hedge-mode migration is not automatic')
    age=(now-utc(snapshot['asof'])).total_seconds()
    if age<0 or age>max_snapshot_age_seconds:raise ValueError('Stale/future account snapshot')
    equity=finite(snapshot['equity'],'equity',True);price=finite(snapshot['mark_price'],'mark price',True)
    current=finite(snapshot['net_units'],'net position');step=finite(quantity_step,'quantity step',True)
    fee=finite(fee_rate,'fee rate')
    if fee<0:raise ValueError('Negative fee')
    cap=finite(approval['risk_contract']['net_cap_at_open_rebalance_after_fees'],'net cap',True)
    if cap*fee>=1:raise ValueError('Invalid fee/cap combination')
    weights={k:finite(v,'weight') for k,v in approval['weights_notional'].items() if v!=0}
    if any(v<0 for v in weights.values()):raise ValueError('Negative allocation coefficient')
    existing=copy.deepcopy(state['sleeves'])
    if set(existing)-set(weights):raise ValueError('Unapproved virtual sleeve; explicit reconciliation needed')
    before=sum(finite(v['units'],'virtual units') for v in existing.values())+finite(state.get('dust_units',0),'dust')
    if abs(before-current)>max(step/10,1e-10):raise ValueError('Exchange/virtual position mismatch; refusing automatic attribution')
    if abs(current-rounded_units(current,step))>step/10:raise ValueError('Exchange position not aligned with configured lot step')
    proposed=copy.deepcopy(existing);actions=[];closed=set();processed=copy.deepcopy(state.get('processed_signals',{}));history=copy.deepcopy(state.get('processed_signal_ids',{}))
    if not isinstance(history,dict) or any(not isinstance(v,list) or any(not isinstance(i,str) or not i for i in v) for v in history.values()):raise ValueError('Invalid processed signal history')
    for name,v in list(proposed.items()):
        q=finite(v['units'],'held units');entry=finite(v['entry_price'],'entry price',True)
        barrier=v.get('barrier_exit') or {};reason=None
        side=1 if q>0 else -1
        stop=barrier.get('stop_bps');take=barrier.get('take_bps')
        ret=side*(price/entry-1)
        if stop is not None and ret<=-finite(stop,'stop bps',True)/10000:reason='stop_trigger_at_observed_price'
        elif take is not None and ret>=finite(take,'take bps',True)/10000:reason='take_trigger_at_observed_price'
        elif v.get('expires_at') and now>=utc(v['expires_at']):reason='time_exit'
        if reason:
            del proposed[name];closed.add(name);actions.append({'name':name,'action':'exit','reason':reason,'observed_price':price})
    seen=set()
    for signal in signals:
        name=signal['name']
        if name not in weights or name in seen:raise ValueError('Unknown/duplicate signal sleeve')
        seen.add(name)
        if utc(signal['execution_time'])!=now:raise ValueError('Signal execution clock mismatch')
        if not isinstance(signal.get('signal_id'),str) or not signal['signal_id']:raise ValueError('Signal identity required')
        if not isinstance(signal.get('ready',True),bool):raise ValueError('Signal readiness must be boolean')
        kind=signal['kind']
        if kind not in ['entry','target']:raise ValueError('Unsupported signal kind')
        if kind=='entry' and not isinstance(signal.get('active'),bool):raise ValueError('Entry active must be boolean')
        if signal['signal_id'] in history.get(name,[]):
            actions.append({'name':name,'action':'duplicate_signal_ignored','signal_id':signal['signal_id']})
            continue
        processed[name]=signal['signal_id'];history.setdefault(name,[]).append(signal['signal_id'])
        if not signal.get('ready',True):continue
        if kind=='target':
            if name!='macro_flow':raise ValueError('Only macro supports target maintenance')
            target=finite(signal['target_fraction'],'macro target')
            if abs(target)>1+1e-12:raise ValueError('Macro target exceeds source limit')
            q=weights[name]*target*equity/price
            if abs(q)<1e-15:proposed.pop(name,None)
            else:proposed[name]={'units':q,'entry_price':price,'entry_time':str(now),'expires_at':None,'signal_id':signal['signal_id'],'barrier_exit':None}
            actions.append({'name':name,'action':'target','fraction':target})
        elif signal.get('active') and name not in existing and name not in closed:
            if signal.get('side') not in ['LONG','SHORT']:raise ValueError('Invalid entry side')
            if name=='dollar_rally_short' and signal['side']!='SHORT':raise ValueError('Dollar sleeve must be SHORT')
            hold=int(signal['hold_bars'])
            if hold<=0 or float(signal['hold_bars'])!=hold:raise ValueError('Invalid hold duration')
            side=1 if signal['side']=='LONG' else -1
            barrier=copy.deepcopy(signal.get('barrier_exit'))
            if barrier:
                for key in ['take_bps','stop_bps']:
                    if barrier.get(key) is not None:finite(barrier[key],key,True)
            proposed[name]={'units':side*weights[name]*equity/price,'entry_price':price,'entry_time':str(now),
                            'expires_at':str(now+pd.Timedelta(minutes=5*hold)),'signal_id':signal['signal_id'],'barrier_exit':barrier}
            actions.append({'name':name,'action':'entry','side':signal['side']})
    desired=sum(v['units'] for v in proposed.values());aligned=(1 if desired>=0 else -1)*current*price
    budget=cap*equity;fc=cap*fee
    allowed=(budget+fc*aligned)/(1+fc) if budget>=aligned else (budget-fc*aligned)/(1-fc)
    scale=min(1.,max(allowed,0)/max(abs(desired*price),1e-15))
    if scale<1:
        for v in proposed.values():v['units']*=scale
        actions.append({'action':'net_risk_resize_all_sleeves','scale':scale})
    unrounded=sum(v['units'] for v in proposed.values());target=rounded_units(unrounded,step)
    delta=target-current;notional=abs(delta)*price;estimated_fee=notional*fee
    if equity<=estimated_fee or abs(target*price)>cap*(equity-estimated_fee)+1e-8:raise ValueError('Post-fee risk check failed')
    legs=[]
    def leg(q,reduce):
        if abs(q)<step/10:return
        quantity=float(Decimal(str(step))*round(abs(q)/step))
        legs.append({'side':'BUY' if q>0 else 'SELL','quantity':quantity,'position_side':'BOTH','reduce_only':reduce})
    if current*target<0:
        leg(-current,True);leg(target,False)
    else:leg(delta,abs(target)<abs(current))
    payload={'mode':'dry_run_only','orders_enabled':False,'approval_hash':approval_hash,'base_state_hash':digest(state),'base_revision':state['revision'],
             'execution_time':str(now),'symbol':'BTCUSDT','reference_price':price,'equity_before':equity,'current_net_units':current,
             'target_net_units':target,'net_delta_units':delta,'estimated_fee':estimated_fee,'quantity_step':step,
             'actions':actions,'order_plan':legs,'proposed_sleeves':proposed,'proposed_dust_units':target-unrounded,'proposed_processed_signals':processed,'proposed_processed_signal_ids':history,
             'limits':['No network or broker submission.','Observed-price barrier decisions do not claim fills at historical trigger prices.',
                       'Partial fills, cancellation/recovery and hedge-mode migration require production integration.']}
    payload['plan_id']=digest(payload)
    return payload


def apply_paper_fill(state: dict,proposal: dict) -> dict:
    """Apply an ideal complete paper fill, never an actual broker acknowledgement."""
    if proposal.get('mode')!='dry_run_only' or proposal.get('orders_enabled') is not False:raise ValueError('Not a paper plan')
    expected=digest({k:v for k,v in proposal.items() if k!='plan_id'})
    if proposal.get('plan_id')!=expected:raise ValueError('Plan integrity failure')
    if proposal['base_state_hash']!=digest(state):raise ValueError('State changed or plan already applied')
    result=copy.deepcopy(state);result['revision']+=1
    result['sleeves']=copy.deepcopy(proposal['proposed_sleeves']);result['dust_units']=proposal['proposed_dust_units']
    result['processed_signals']=copy.deepcopy(proposal['proposed_processed_signals'])
    result['processed_signal_ids']=copy.deepcopy(proposal['proposed_processed_signal_ids'])
    result['last_plan_id']=proposal['plan_id'];result['last_execution_time']=proposal['execution_time']
    result['paper_equity_after_estimated_fee']=proposal['equity_before']-proposal['estimated_fee']
    return result
