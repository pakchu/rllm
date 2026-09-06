"""Fixed new G9/macro combinations on past years; no weight optimization."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from training import search_meaningful_alpha_combinations as base
from training import evaluate_macro_flow_fixed_fresh as macro
from training import search_macro_flow_alpha_combinations as macro_features
from training import export_gross9_structural_clocks as exporter
from training import build_g9_september_clock_inputs as builder
from training import optimize_g9_plus_added_alphas as joint
from training import g9_joint_net_ledger as ledger
from execution.portfolio_live import _build_portfolio_feature_frame
from preprocessing.live_db_features import LiveDbFeatureConfig
from preprocessing.binance_aux_features import attach_binance_um_aux_frames

OUT=base.ROOT/'research/g9_macro_historical_v2'
BARRIERS=base.ROOT/'research/g9_historical_barriers/report.json'
RAW=base.ROOT/'research/g9_september_inputs/raw_enriched_cache.pkl'
FUND=base.ROOT/'research/g9_september_inputs/raw_funding_cache.csv.gz'
SPOT=base.DATA/'cache_spot_premium_5m_2020-01-01_2026-06-01.csv.gz'
MARKET=base.DATA/'cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_oi.csv.gz'
NAMES=joint.G9+['macro_flow']
G9=joint.CONTROL_G9[:6]
WEIGHTS=np.array([G9,np.r_[G9[:5]*.5,1.],np.r_[G9[:5],.5],np.r_[G9[:5],1.]])
LABELS=['g9','g9_half_macro1','g9_macro0.5','g9_macro1']
WINDOWS={'2024':('2024-01-01','2025-01-01'),'2025':('2025-01-01','2026-01-01'),'2026H1':('2026-01-01','2026-07-01')}
DESIGN={'version':2,'windows':WINDOWS,'weights':dict(zip(LABELS,WEIGHTS.tolist())),
        'selection':'none; combinations fixed before opening these historical reports',
        'rank7':'causal annual refits for2024/2025; frozen2026 runtime model only in2026',
        'execution':'shared fixed-unit net ledger, post-fee netcap4.5, conservative barrier MDD and ruin',
        'costs':[.0006,.001],'live_enabled':False,'caveat':'historically exposed data and retrospective portfolio choice, not pristine OOS'}


def register():
    exporter.validate_frozen_inputs()
    paths=[__file__,ledger.__file__,joint.__file__,builder.__file__,macro.__file__,macro_features.__file__,RAW,FUND,SPOT,MARKET,BARRIERS]
    paths += [exporter.resolve_frozen_input(value[0]) for value in exporter.INPUT_BINDINGS.values()]
    r={'design':DESIGN,'hashes':{str(p):base.sha(p) for p in paths}}
    r=json.loads(json.dumps(r))
    path=OUT/'design.json'
    if path.exists() and json.loads(path.read_text())!=r:raise RuntimeError('Historical design drift')
    base.write_json(path,r);return r


def merged_market():
    m=pd.read_csv(MARKET);m.date=pd.to_datetime(m.date,utc=True).dt.tz_convert(None)
    hf=pd.read_csv(base.FUNDING);hf['date']=pd.to_datetime(hf.funding_time,unit='ms',utc=True).dt.tz_convert(None)
    premium=pd.read_csv(base.DATA/'binance_um_aux_btc_2020_2026/BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz')
    m=attach_binance_um_aux_frames(m,funding_frame=hf,premium_frame=premium)
    spot=pd.read_csv(SPOT);spot.date=pd.to_datetime(spot.date,utc=True).dt.tz_convert(None)
    m=m.drop(columns=[c for c in spot if c!='date' and c in m],errors='ignore').merge(spot,on='date',how='left')
    recent=pd.read_pickle(RAW);recent.date=pd.to_datetime(recent.date,utc=True).dt.tz_convert(None)
    archive=builder.archive_oi(pd.read_csv(builder.DEFAULT_OI_ARCHIVE))
    recent=builder.overlay_official_oi(recent,archive,asof=pd.Timestamp('2026-09-05T00:05:00Z'))
    seam=recent.date.min()
    m=pd.concat([m[m.date<seam],recent],ignore_index=True).sort_values('date').reset_index(drop=True)
    m=m[m.date<pd.Timestamp('2026-07-01 00:05')].reset_index(drop=True)
    if m.date.duplicated().any() or not m.date.diff().dropna().eq(pd.Timedelta('5min')).all():raise ValueError('Historical grid gap')
    funding=pd.read_csv(FUND);funding.date=pd.to_datetime(funding.date,utc=True,format='mixed').dt.tz_convert(None)
    funding=funding[funding.date>=seam]
    funding,diag=builder.canonicalize_funding_aliases(funding)
    if not diag['passed']:raise ValueError('Funding alias failure')
    funding=pd.concat([hf.loc[hf.date<seam,['date','funding_rate','mark_price']],funding],ignore_index=True).sort_values('date').reset_index(drop=True)
    base.validate_funding(funding,m.date.iloc[-1])
    return m,funding,{'seam':str(seam),'funding_aliases':diag,'market_end':str(m.date.iloc[-1])}


def native_2026(m,f):
    # Warmup and source metadata are genuine, not fabricated all-available flags.
    mm=m[m.date>=pd.Timestamp('2025-09-01')].reset_index(drop=True)
    features=_build_portfolio_feature_frame(mm,LiveDbFeatureConfig(include_spot_source=True),include_activity_flow=False)
    portfolio=json.loads(Path(builder.DEFAULT_PORTFOLIO).read_text())
    cfg={r['name']:json.loads(Path(r['source']).read_text()) for r in portfolio['base_sleeves']}
    month=builder.month;start=pd.Timestamp('2026-01-01');end=pd.Timestamp('2026-07-01')
    signals={'fresh_kimchi_fx':month._fresh_signal(mm,features,cfg['fresh_kimchi_fx']),
             'markov_transition_long':month._markov_signal(mm,features,cfg['markov_transition_long'])}
    for name in ['rex_taker_low_range_position','cand_rex_veto_7']:signals[name]=month._rex_signal(mm,features,cfg[name])
    signals['frozen_annual_rank7'],lifecycle,diag=month._rank7_signal(mm,cfg['frozen_annual_rank7'])
    out={}
    for name in joint.G9:
        if name=='fresh_kimchi_fx':
            c=cfg[name];fn=lambda p:{'hold_bars':int(c['hold_bars']),'take_bps':float(c['take_bps']),'stop_bps':float(c['stop_bps'])}
        elif name=='frozen_annual_rank7':fn=lambda p:lifecycle[p]
        else:
            out[name]=builder._fixed_hold_arrays_with_trades(mm,signals[name],name=name,hold_bars=int(cfg[name]['hold_bars']),start=start,end=end)['trades'];continue
        out[name]=builder._barrier_arrays_with_trades(mm,f,signals[name],name=name,lifecycle=fn,start=start,end=end)['trades']
    return out,diag


def fixed_historical_clocks():
    # Only export the REX helpers over extended dates; they read frozen JSONL sources.
    previous=exporter._split_bounds
    exporter._split_bounds=lambda:tuple((k,pd.Timestamp(a,tz='UTC'),pd.Timestamp(z,tz='UTC')) for k,(a,z) in WINDOWS.items() if k!='2026H1')
    try:
        return {n:exporter.SLEEVE_BUILDERS[n]() for n in ['rex_taker_low_range_position','cand_rex_veto_7']}
    finally:exporter._split_bounds=previous


def blocks(m,f,start,end):
    dates=pd.DatetimeIndex(m.date);ids=np.flatnonzero((dates>=pd.Timestamp(start))&(dates<pd.Timestamp(end)))
    if len(ids)!=int((pd.Timestamp(end)-pd.Timestamp(start))/pd.Timedelta('5min')) or ids[-1]+1>=len(m):raise ValueError('Incomplete period')
    op=m.open.to_numpy(float);transfers=np.zeros(len(m));pos=np.searchsorted(dates.to_numpy(),f.date.to_numpy(),side='right')-1
    use=(pos>=0)&(pos<len(m));marks=pd.to_numeric(f.loc[use,'mark_price'],errors='coerce').to_numpy(float)
    fallback=~np.isfinite(marks)|(marks<=0);marks[fallback]=op[pos[use][fallback]]
    np.add.at(transfers,pos[use],f.loc[use,'funding_rate'].to_numpy(float)*marks)
    return ids,{'date':dates.to_numpy()[ids],'end_date':dates.to_numpy()[ids+1],'open':op[ids],'end':op[ids+1],
                'high':m.high.to_numpy(float)[ids],'low':m.low.to_numpy(float)[ids],'funding':transfers[ids]}


def run():
    registration=register();m,f,receipt=merged_market()
    x=base.features(m,f);x,hourly,_=base.execution_blocks(m,f,x)
    cols=['date','dxy','usdkrw','kimchi_premium','dxy_available','usdkrw_available','kimchi_available']
    x=pd.concat([x,macro_features.macro_features(m[cols],x.index)],axis=1)
    mp,_=macro.fixed_positions(x);ms=pd.Series(mp['dollar_flow_plus_regime_switch'],index=pd.DatetimeIndex(hourly['date']))
    historical=json.loads(BARRIERS.read_text());rex=fixed_historical_clocks()
    native,diag=native_2026(m,f)
    # Same frozen Markov rule/stride as historical G9 reconstruction, no retraining.
    from training import portfolio_opt_added_alpha_update as portfolio
    old=m[m.date<pd.Timestamp('2026-01-01')].reset_index(drop=True)
    features=portfolio.feature_frame(old);markov=portfolio.markov_active(old,features)
    reports={};trades_receipts={}
    for window,(a,z) in WINDOWS.items():
        ids,d=blocks(m,f,a,z);dates=pd.DatetimeIndex(d['date'])
        if window=='2026H1':trades=native
        else:
            trades={n:[] for n in joint.G9}
            for name in ['fresh_kimchi_fx','frozen_annual_rank7']:
                trades[name]=historical['windows'][window]['sleeves'][name]['trades']
            for name,clock in rex.items():
                for row in clock[clock['split']==window].itertuples():
                    exit_=pd.Timestamp(row.exit_time).tz_convert(None)
                    trades[name].append({'entry_date':str(row.entry_time),'exit_date':str(exit_),'side':'LONG' if row.side>0 else 'SHORT',
                                         'exit_kind':'open','exit_price':float(m.loc[m.date==exit_,'open'].iloc[0])})
            slots=np.arange(143,len(old)-578,12);nxt=0
            for s in slots:
                if s<nxt or not markov[s]:continue
                endpos=s+577
                if not (pd.Timestamp(a)<=old.date.iloc[s] and old.date.iloc[endpos]<pd.Timestamp(z)):continue
                trades['markov_transition_long'].append({'entry_date':str(old.date.iloc[s+1]),'exit_date':str(old.date.iloc[endpos]),'side':'LONG',
                                                        'exit_kind':'open','exit_price':float(old.open.iloc[endpos])});nxt=endpos+1
        if window=='2024':
            for name in joint.G9:
                expected=exporter.EXPECTED_COUNTS[name]['test2024']
                if len(trades[name])!=expected:raise RuntimeError(f'Historical2024 clock mismatch {name}: {len(trades[name])} != {expected}')
        p,e,b=joint.clock_arrays({'sleeves':{n:{'trades':trades[n]} for n in joint.G9}},dates)
        macro_target=ms.reindex(ms.index.union(dates)).ffill().reindex(dates).fillna(0).to_numpy()
        p=np.column_stack([p,macro_target]);e=np.column_stack([e,dates.isin(ms.index)]);e[0]=True
        b=np.column_stack([b,np.full(len(dates),np.nan)])
        reports[window]={}
        for cost in DESIGN['costs']:
            rows,_=ledger.simulate(d,p,e,b,WEIGHTS,NAMES,cost=cost)
            reports[window][str(cost)]=dict(zip(LABELS,rows))
        trades_receipts[window]={n:len(trades[n]) for n in trades}
        print(window,json.dumps(reports[window]['0.0006']),flush=True)
    result={'registration':registration,'receipt':receipt,'rank7_2026_diagnostics':diag,'trade_counts':trades_receipts,'reports':reports,
            'live_enabled':False,'limitations':['No weight optimization on these reports.','2024/2025 use archived historical G9 clocks/annual refits;2026 uses fixed native runtime policies and2026 bundle.',
                                            'Entire H1 includes June30; DB/cache seam receipt is explicit.','Different historical source contracts mean this is not exact live execution parity.']}
    base.write_json(OUT/'report.json',result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');a=p.parse_args()
    register() if a.freeze else run()
