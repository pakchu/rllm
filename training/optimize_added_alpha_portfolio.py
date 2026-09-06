"""Exposed-period allocation diagnostics with sleeve-local fixed-unit accounting."""
import argparse
import json
import hashlib
import numpy as np
import pandas as pd
from training import search_meaningful_alpha_combinations as base
from training import audit_macro_oi_fresh_portfolio as old
from training import evaluate_oi_divergence_fresh as oi
from training import evaluate_macro_flow_fixed_fresh as macro
from training import search_macro_flow_alpha_combinations as features_macro
from training import evaluate_regional_trend_fresh as regional

OUT = base.ROOT / 'research/added_alpha_portfolio_optimization_v2'
NAMES = ['macro_flow', 'oi_pullback', 'regional_trend']
DESIGN = {'version': 2, 'sleeves': NAMES, 'grid': 'all nonnegative fully invested weights in 5% steps (231)',
          'selection': ['2026-06-01', '2026-07-01'], 'report': ['2026-07-01', oi.EVAL_END],
          'rank': 'June 10bp CAGR / conservative MDD; no frequency, overlap or fee-ratio gate',
          'accounting': 'update only the originating sleeve units; net actual units before fees and funding; 1x net cap at execution events',
          'status': 'all periods already exposed; chronological diagnostic, not clean OOS; no live orders'}


def grid():
    return np.array([(i/20, j/20, (20-i-j)/20) for i in range(21) for j in range(21-i)])


def simulate(d, targets, events, weights, cost):
    n, k = len(d['open']), len(weights)
    eq = np.ones(k); peak = eq.copy(); mdd = np.zeros(k)
    units = np.zeros((k, targets.shape[1])); fees = np.zeros(k); paid = np.zeros(k)
    orders = np.zeros(k, int); entries = np.zeros(k, int); turnover = np.zeros(k)
    cap_hits = np.zeros(k, int)
    for t in range(n):
        op = d['open'][t]; prior = eq.copy(); previous = units.sum(axis=1)
        # Other sleeves retain fixed units when one sleeve updates.
        proposed = np.where(events[t][None, :], weights * targets[t] * eq[:, None] / op, units)
        net = proposed.sum(axis=1)
        hit = np.abs(net * op) > np.maximum(eq, 0) + 1e-12
        scale = np.minimum(1, np.maximum(eq, 0) / np.maximum(np.abs(net*op), 1e-15))
        active_event = np.any(events[t][None, :] & (weights != 0), axis=1)
        proposed *= np.where(active_event, scale, 1.)[:, None]
        cap_hits += hit & active_event
        units = proposed; net = units.sum(axis=1)
        notional = (net-previous)*op
        charge = np.abs(notional)*cost; fees += charge; eq -= charge
        orders += np.abs(notional)>1e-10
        entries += (previous*net <= 0) & (np.abs(net)>1e-15)
        turnover += np.abs(notional)/np.maximum(prior, 1e-12)
        transfer = net*d['funding'][t]; paid += transfer
        hi = eq + net*(np.where(net>=0,d['high'][t],d['low'][t])-op) + np.maximum(-transfer,0)
        lo = eq + net*(np.where(net>=0,d['low'][t],d['high'][t])-op) - np.maximum(transfer,0)
        peak = np.maximum(peak,hi); mdd = np.maximum(mdd,1-lo/np.maximum(peak,1e-12))
        eq += net*(d['end'][t]-op)-transfer
        if t == n-1:
            liquidation = np.abs(net)*d['end'][t]
            fees += liquidation*cost; eq -= liquidation*cost
            orders += liquidation>1e-10
            turnover += liquidation/np.maximum(eq,1e-12)
        if np.any(eq<=0):
            raise RuntimeError('Insolvency: this study does not model liquidation')
        peak = np.maximum(peak,eq); mdd = np.maximum(mdd,1-eq/peak)
    years = (pd.Timestamp(d['end_date'][-1])-pd.Timestamp(d['date'][0])).total_seconds()/(365.25*86400)
    cagr = (eq**(1/years)-1)*100
    return [{'weights':dict(zip(NAMES,map(float,w))), 'return_pct':float((eq[i]-1)*100),
             'cagr_pct':float(cagr[i]), 'mdd_pct':float(mdd[i]*100),
             'calmar':float(cagr[i]/(mdd[i]*100)) if mdd[i]>1e-12 else 0.,
             'entry_episodes':int(entries[i]), 'orders_including_liquidation':int(orders[i]),
             'fees_pct_initial':float(fees[i]*100), 'funding_pct_initial':float(paid[i]*100),
             'turnover':float(turnover[i]), 'net_cap_events':int(cap_hits[i])} for i,w in enumerate(weights)]


def context():
    market,fund,source,receipt = oi.load_context()
    signal = json.loads(oi.CONFIG.read_text())['signal']
    candidate = {**signal,'hold_bars':signal['hold_bars_5m'],'stride_bars':signal['stride_bars_5m']}
    trades,_,_,_ = oi.schedule(market,fund,candidate)
    x = base.features(market,fund); x,hourly,_ = base.execution_blocks(market,fund,x)
    cols = ['date','dxy','usdkrw','kimchi_premium','dxy_available','usdkrw_available','kimchi_available']
    x = pd.concat([x,features_macro.macro_features(market[cols],x.index)],axis=1)
    p,_ = macro.fixed_positions(x); r,_ = regional.position(x)
    indices,d = old.five_minute_blocks(market,fund)
    dates = pd.Series(d['date'])
    # Reindex with union so an update immediately before window start is retained.
    def expand(values):
        s = pd.Series(values,index=pd.DatetimeIndex(hourly['date']))
        return s.reindex(s.index.union(pd.DatetimeIndex(dates))).ffill().reindex(dates).fillna(0).to_numpy()
    op = np.zeros(len(market))
    for trade in trades: op[trade.entry_position:trade.exit_position] = trade.side
    op = op[indices]
    targets = np.column_stack([expand(p['dollar_flow_plus_regime_switch']),op,expand(r)])
    he = dates.isin(hourly['date']).to_numpy()
    events = np.column_stack([he,np.r_[op[0]!=0,op[1:]!=op[:-1]],he])
    return d,targets,events,{'db':receipt,'oi_rows':len(source),'oi_last':str(source.date.max()),'oi_trades':len(trades)}


def register():
    paths = [__file__,base.__file__,old.__file__,oi.__file__,macro.__file__,regional.__file__,features_macro.__file__,oi.CONFIG,base.MARKET,base.FUNDING,oi.OLD]
    payload = {'design':DESIGN,'sources_and_code':{str(p):base.sha(p) for p in paths}}
    path = OUT/'design.json'
    if path.exists() and json.loads(path.read_text()) != payload: raise RuntimeError('Frozen design drift')
    base.write_json(path,payload)
    return payload


def run():
    reg = register(); d,p,e,receipt = context(); weights = grid()
    windows = {'june_selection':('2026-06-01','2026-07-01'),
               'july_aug_report':('2026-07-01',oi.EVAL_END),'common':(oi.START,oi.EVAL_END)}
    reports = {}
    for name,(start,end) in windows.items():
        mask = (d['date']>=pd.Timestamp(start).tz_localize(None).to_datetime64()) & (d['end_date']<=pd.Timestamp(end).tz_localize(None).to_datetime64())
        dd = base.subset(d,mask); pp=p[mask]; ee=e[mask].copy(); ee[0]=True
        reports[name] = {str(c):simulate(dd,pp,ee,weights,c) for c in [0.,.0006,.001]}
        if name=='june_selection':
            ranked=sorted(range(len(weights)),key=lambda i:reports[name]['0.001'][i]['calmar'],reverse=True)
            selected=ranked[0]
            base.write_json(OUT/'selection_freeze.json',{'index':selected,'row':reports[name]['0.001'][selected], 'uses_july_aug':False})
    best_common=max(range(len(weights)),key=lambda i:reports['common']['0.001'][i]['calmar'])
    result={'registration':reg,'receipt':receipt,'count':len(weights),'selection_index':selected,
            'retrospective_best_common_index':best_common,'reports':reports,'live_enabled':False,
            'target_hash':hashlib.sha256(p.tobytes()+e.tobytes()).hexdigest(),
            'caveats':['Exposed data, not pristine OOS.', 'Starts each window in cash and initializes known targets; OI position may be carried-in signal.',
                       'Legacy Gross9 not reconstructed here; no frozen recent REX source.',
                       'Net cap enforced at execution events, not continuous intrabar liquidation.',
                       'Regional trend retained despite weak standalone stress; no overlap rejection.']}
    base.write_json(OUT/'report.json',result)
    config={'enabled':False,'live_authorized':False,'research_only':True,'weights':dict(zip(NAMES,weights[selected])),
            'net_cap':1.,'long_short_offset':True,'overlap_allowed':True,'fee_ratio_gate':False,'trade_frequency_gate':False,
            'report_sha256':base.sha(OUT/'report.json'),'selection':'June stress Calmar, no July/Aug rerank'}
    base.write_json(OUT/'shadow_config.json',config)
    for label,i in [('june_selected',selected),('retrospective_common_best',best_common)]:
        print(label, i)
        for window in reports:
            for c in ['0.0006','0.001']: print(window,c,json.dumps(reports[window][c][i]))

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--freeze',action='store_true'); args=parser.parse_args()
    register() if args.freeze else run()
