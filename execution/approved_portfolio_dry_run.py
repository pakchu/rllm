"""Executable offline preparation for the approved G9+macro1+dollar-short0.5.

Consumes closed enriched5m CSV and explicit account snapshot JSON. No broker
client, credentials, account-mode mutation, service management, or order sender.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from execution import approved_net_planner as planner
from execution import approved_portfolio_signals as adapters

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_APPROVAL=ROOT/'configs/approved/g9_macro1_dollar_short05_2026-09-07.json'
G9_CONFIG=ROOT/'configs/live/portfolio_added_alpha_mainnet_live_2026-07-18.json'
EXPECTED_EVIDENCE={'configs/shadow/ten_alpha_portfolio_research_2026-09-07.json': '012c44e5d76bc0d86cd89195d5aec7be90cff6e60df4b16cc93b53c9cabc3350', 'research/ten_alpha_optimization/report.json': '7b933d652f91e55afc5fcff9f877c279bf0129208512112a9d6d300110e4933e', 'configs/shadow/macro_flow_regime_switch_candidate_2026-09-06.json': '5c398879eb090a97423b85e7d8f268d6ba2f199b709bb25d728d8b77a8e2fcdc', 'configs/shadow/legacy_dollar_rally_short_2026-09-07.json': '62fecd410deeca32b1ca8cea7e48c384ddeda06594a275d5c490a673d089c20d'}


def clean(value):
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float) and not math.isfinite(value):return None
    if isinstance(value,(pd.Timestamp,Path)):return str(value)
    return value


def validate_approval(approval):
    if approval.get('id')!='g9_macro1_dollar_short05_2026_09_07' or not approval.get('portfolio_selection_approved'):raise ValueError('Not the approved portfolio')
    reference=json.loads((ROOT/'configs/shadow/ten_alpha_portfolio_research_2026-09-07.json').read_text())
    label='g9_macro1_d0.5_r0.0'
    if approval.get('selected_label')!=label or approval.get('weights_notional')!=reference['descriptive_anchored_candidates'][label]:
        raise ValueError('Approved weights changed')
    risk=approval.get('risk_contract',{})
    if risk.get('net_cap_at_open_rebalance_after_fees')!=4.5 or risk.get('long_short_offset_before_cost_and_funding') is not True:
        raise ValueError('Approved risk contract changed')
    if approval.get('evidence')!=EXPECTED_EVIDENCE:raise ValueError('Approval evidence map missing or changed')
    for path,expected in EXPECTED_EVIDENCE.items():
        import hashlib
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=expected:raise ValueError(f'Approved evidence drift: {path}')


def closed_market(market,now):
    out=market.copy();out['date']=pd.to_datetime(out.date,utc=True,errors='raise')
    slot=planner.utc(now).floor('5min')
    out=out[out.date+pd.Timedelta('5min')<=slot].sort_values('date').reset_index(drop=True)
    if out.empty or out.date.iloc[-1]!=slot-pd.Timedelta('5min'):raise ValueError('Latest closed5m bar missing')
    if out.date.duplicated().any() or not out.date.diff().dropna().eq(pd.Timedelta('5min')).all():raise ValueError('Market grid invalid')
    if len(out)<18000:raise ValueError('At least18000 genuine historical bars required for approved G9 sources')
    for c in ['open','high','low','close']:
        v=pd.to_numeric(out[c],errors='coerce')
        if not np.isfinite(v).all() or (v<=0).any():raise ValueError(f'Invalid market {c}')
    if ((out.high<out[['open','close']].max(axis=1))|(out.low>out[['open','close']].min(axis=1))).any():raise ValueError('OHLC bounds invalid')
    return out,slot


def score_g9(market,approval,now):
    # Reuse only the documented pure scoring surface, never its order runner.
    from execution.portfolio_live import _build_portfolio_feature_frame, _score_sleeve
    from execution.wave_execution import WaveExecutionConfig
    from preprocessing.live_db_features import LiveDbFeatureConfig
    feature=_build_portfolio_feature_frame(market.assign(date=market.date.dt.tz_convert(None)),LiveDbFeatureConfig(include_spot_source=True),include_activity_flow=False)
    raw=market.copy();raw.date=raw.date.dt.tz_convert(None)
    definitions=json.loads(G9_CONFIG.read_text())['base_sleeves'];signals=[];diagnostics=[]
    for definition in definitions:
        sleeve=dict(definition);sleeve['source']=str(ROOT/sleeve['source'])
        sleeve['weight']=approval['weights_notional'][sleeve['name']]
        r=_score_sleeve(sleeve=sleeve,enriched=raw,features=feature,exec_cfg=WaveExecutionConfig(dry_run=True,allow_live_orders=False),asof=planner.utc(now))
        diagnostics.append(clean(r))
        ready=not any('error:' in x or 'missing:' in x for x in r['reasons'])
        signals.append({'name':r['name'],'kind':'entry','ready':ready,'active':bool(r['active']),
                        'side':r['side'],'hold_bars':r['hold_bars'],'signal_id':r['signal_id'],
                        'execution_time':str(planner.utc(now)),'barrier_exit':r['barrier_exit']})
    return signals,diagnostics


def prepare(approval,state,snapshot,market,*,execution_time,quantity_step=.00001):
    validate_approval(approval)
    now=planner.utc(execution_time);slot=now.floor('5min')
    if (now-slot).total_seconds()>30:raise ValueError('Entry-planning deadline exceeded; no stale-bar catchup entries')
    # Reject wrong-account attribution before expensive feature work.
    if snapshot.get('position_mode')!='one_way':raise ValueError('Hedge-mode account requires explicit reconciliation/migration; no orders planned')
    m,slot=closed_market(market,now)
    signals,diagnostics=score_g9(m,approval,now)
    if slot.minute==5:
        targets=adapters.build_macro_targets(m,asof=now)
        if slot not in targets.index:raise ValueError('Macro hourly execution target unavailable')
        signals.append({'name':'macro_flow','kind':'target','target_fraction':float(targets.loc[slot]),
                        'signal_id':f'macro_flow:{slot.isoformat()}','execution_time':str(now),'ready':True})
    dollar=adapters.score_dollar_short(m,slot-pd.Timedelta('5min'))
    diagnostics.append(clean(dollar))
    if 'execution_time' in dollar:
        signals.append({'name':'dollar_rally_short','kind':'entry','ready':True,'active':bool(dollar['active']),
                        'side':'SHORT','hold_bars':144,'barrier_exit':None,
                        'signal_id':f'dollar_rally_short:{slot.isoformat()}','execution_time':str(now)})
    result=planner.plan(approval,state,snapshot,signals,execution_time=str(now),quantity_step=quantity_step)
    return {'mode':'historical_or_offline_dry_run','orders_enabled':False,'broker_transport_connected':False,
            'production_ready':False,'closed_bar_date':str(m.date.iloc[-1]),'scheduled_slot':str(slot),
            'signals':clean(signals),'diagnostics':diagnostics,'plan':result,
            'remaining_live_requirements':['Authoritative fill/cancel/partial-fill reconciliation.',
                                           'Resolve current hedge-mode versus approved net-position ownership.',
                                           'Connect fresh market/account snapshots and validate exchange filters and latency.',
                                           'Production barrier monitoring and operational recovery/soak.']}


def write_new(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(clean(payload),f,indent=2,allow_nan=False);f.write('\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--market',required=True,help='Enriched5m CSV/CSV.gz; no pickle loading')
    p.add_argument('--snapshot',required=True,help='Explicit account/paper snapshot JSON')
    p.add_argument('--approval',default=str(DEFAULT_APPROVAL));p.add_argument('--state')
    p.add_argument('--execution-time');p.add_argument('--quantity-step',type=float,default=.00001)
    p.add_argument('--output',required=True);p.add_argument('--paper-fill-state-output')
    p.add_argument('--live',action='store_true',help='Unsupported: this runner never submits orders')
    a=p.parse_args()
    if a.live:p.error('Live submission is intentionally unavailable in this preparation runner')
    if Path.cwd().resolve()!=ROOT:p.error(f'Run from repository root: {ROOT}')
    # Refuse overwrites before any work; outputs cannot clobber live state.
    outputs=[Path(a.output)]+([Path(a.paper_fill_state_output)] if a.paper_fill_state_output else [])
    if len(set(x.resolve() for x in outputs))!=len(outputs) or any(x.exists() for x in outputs):p.error('Use distinct new output paths; existing files will not be overwritten')
    approval=json.loads(Path(a.approval).read_text());snapshot=json.loads(Path(a.snapshot).read_text())
    state=json.loads(Path(a.state).read_text()) if a.state else planner.empty_state(planner.digest(approval))
    market=pd.read_csv(a.market,compression='infer')
    result=prepare(approval,state,snapshot,market,execution_time=a.execution_time or snapshot['asof'],quantity_step=a.quantity_step)
    write_new(a.output,result)
    if a.paper_fill_state_output:write_new(a.paper_fill_state_output,planner.apply_paper_fill(state,result['plan']))
    print(json.dumps({'output':a.output,'orders_enabled':False,'production_ready':False,'target_net_units':result['plan']['target_net_units'],'net_order_count':len(result['plan']['order_plan'])},indent=2))

if __name__=='__main__':main()
