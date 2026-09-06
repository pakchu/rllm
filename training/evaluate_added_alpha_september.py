"""Fixed-weight September extension using publication-delayed official OI archives."""
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from training import optimize_added_alpha_portfolio as opt
from training import evaluate_macro_flow_fixed_fresh as mf
from training import evaluate_oi_divergence_fresh as oi
from training import search_meaningful_alpha_combinations as base
from training import search_macro_flow_alpha_combinations as macro_features
from training import evaluate_regional_trend_fresh as regional

OUT=base.ROOT/'research/added_alpha_september'
ARCHIVE=base.DATA/'binance_um_metrics_BTCUSDT_2026-05-01_2026-09-04_extension.csv.gz'
START=oi.START
END='2026-09-05T00:00:00Z'
WEIGHTS=np.array([[.8,.2,0],[.6,.4,0],[.5,.5,0],[1,0,0],[0,1,0],[0,0,1]])
LABELS=['june_selected_80_20','retrospective_60_40','fixed_50_50','macro_only','oi_only','regional_only']
DESIGN={'version':1,'window':[START,END],'asof':mf.ASOF,'weights':dict(zip(LABELS,WEIGHTS.tolist())),
        'selection':'none; prior weights frozen, report all','oi_source':'official Binance daily metrics, observation timestamp +5min; historical DB/cache not overwritten',
        'execution':'same frozen OI gates, next5m entry,8h hold; common sleeve-local net ledger with event risk override',
        'costs':[0.,.0006,.001],'caveat':'source/timing revision: replays earlier windows as well, not identical to prior DB-OI run; exposed historical diagnostics, no live promotion'}


def archive_oi(raw):
    dates=pd.to_datetime(raw.create_time,utc=True).dt.tz_convert(None)
    values=pd.to_numeric(raw.sum_open_interest,errors='raise')
    if dates.duplicated().any() or not np.isfinite(values).all() or (values<=0).any():
        raise ValueError('Invalid archive OI')
    if not ((dates.astype('int64') % pd.Timedelta('5min').value)==0).all():
        raise ValueError('Off-grid archive timestamps')
    return pd.DataFrame({'date':dates+pd.Timedelta('5min'),'open_interest':values}).sort_values('date')


def register():
    files=[__file__,opt.__file__,oi.__file__,oi.oi_eval.__file__,oi.execmod.__file__,mf.__file__,macro_features.__file__,regional.__file__,base.__file__,ARCHIVE,oi.CONFIG,oi.OLD,base.FUNDING]
    reg={'design':DESIGN,'hashes':{str(p):base.sha(p) for p in files}}
    path=OUT/'design.json'
    if path.exists() and json.loads(path.read_text())!=reg: raise RuntimeError('Frozen extension drift')
    base.write_json(path,reg); return reg


def context():
    db,funding,receipt=mf.load_extension()
    archive=archive_oi(pd.read_csv(ARCHIVE))
    # Cut physically at the frozen as-of: no future archive observations enter features.
    archive=archive[archive.date<=pd.Timestamp(mf.ASOF).tz_localize(None)]
    db=db.drop(columns=['open_interest','open_interest_available'],errors='ignore')
    db=pd.merge_asof(db.sort_values('date'),archive,on='date',direction='backward',tolerance=pd.Timedelta('10min'))
    db['open_interest_available']=db.open_interest.notna().astype(float)
    db['open_interest']=db.open_interest.ffill()
    old=pd.read_csv(oi.OLD); old.date=pd.to_datetime(old.date,utc=True).dt.tz_convert(None)
    market=oi.pposm_db.merge_cache_db_markets(old,db,cutoff=mf.SEAM)
    hist=pd.read_csv(base.FUNDING); hist['date']=pd.to_datetime(hist.funding_time,unit='ms',utc=True).dt.tz_convert(None)
    seam=pd.Timestamp(mf.SEAM).tz_localize(None)
    fund=pd.concat([hist.loc[hist.date<seam,['date','funding_rate','mark_price']],funding[funding.date>=seam]],ignore_index=True).sort_values('date')
    base.validate_funding(fund,market.date.iloc[-1])
    dates=pd.DatetimeIndex(market.date)
    if dates.has_duplicates or not np.all(np.diff(dates.asi8)==pd.Timedelta('5min').value): raise ValueError('Market grid gap')
    signal=json.loads(oi.CONFIG.read_text())['signal']
    candidate={**signal,'hold_bars':signal['hold_bars_5m'],'stride_bars':signal['stride_bars_5m']}
    feat=oi.oi_eval._feature_frame(market,window_size=144)
    active=oi.oi_eval._candidate_active(feat,candidate)&(market.open_interest_available.fillna(0).to_numpy()>.5)
    cfg=oi.execmod.Config(input_csv='',metrics_csv='',funding_csv='',output='',manifest_output='',leverage=1.,fee_rate=.0006,slippage_rate=0)
    engine=oi.execmod.ExecutionEngine(market,fund,cfg)
    slots=np.arange(143,len(market)-candidate['hold_bars']-2,candidate['stride_bars'])
    slots=slots[active[slots]&(dates[slots]>=pd.Timestamp(START).tz_localize(None))]
    trades=[]; nxt=0
    for s in slots:
        if s<nxt: continue
        t=engine.trade_at(int(s),1,candidate['hold_bars'],1000000,1000000)
        if t is None or dates[t.exit_position]>=pd.Timestamp(END).tz_localize(None):continue
        trades.append(t); nxt=t.exit_position+1
    x=base.features(market,fund);x,hourly,engine_receipt=base.execution_blocks(market,fund,x)
    cols=['date','dxy','usdkrw','kimchi_premium','dxy_available','usdkrw_available','kimchi_available']
    x=pd.concat([x,macro_features.macro_features(market[cols],x.index)],axis=1)
    macro,_=mf.fixed_positions(x); reg,_=regional.position(x)
    indices=np.flatnonzero((dates>=pd.Timestamp(START).tz_localize(None))&(dates<pd.Timestamp(END).tz_localize(None)))
    indices=indices[indices+1<len(market)]
    transfers=np.zeros(len(market)); fp=np.searchsorted(dates.to_numpy(),fund.date.to_numpy(),side='right')-1
    valid=(fp>=0)&(fp<len(market)); marks=pd.to_numeric(fund.loc[valid,'mark_price'],errors='coerce').to_numpy(float)
    fallback=~np.isfinite(marks)|(marks<=0); opens=market.open.to_numpy(float)
    marks[fallback]=opens[fp[valid][fallback]]
    np.add.at(transfers,fp[valid],marks*fund.loc[valid,'funding_rate'].to_numpy(float))
    d={'date':dates.to_numpy()[indices],'end_date':dates.to_numpy()[indices+1],'open':opens[indices],'end':opens[indices+1],
       'high':market.high.to_numpy(float)[indices],'low':market.low.to_numpy(float)[indices],'funding':transfers[indices]}
    def expand(values):
        s=pd.Series(values,index=pd.DatetimeIndex(hourly['date']))
        return s.reindex(s.index.union(pd.DatetimeIndex(d['date']))).ffill().reindex(d['date']).fillna(0).to_numpy()
    op=np.zeros(len(market))
    for t in trades:op[t.entry_position:t.exit_position]=1
    op=op[indices];targets=np.column_stack([expand(macro['dollar_flow_plus_regime_switch']),op,expand(reg)])
    he=pd.DatetimeIndex(d['date']).isin(hourly['date'])
    events=np.column_stack([he,np.r_[op[0]!=0,op[1:]!=op[:-1]],he])
    available=market.open_interest_available.to_numpy()[indices]
    receipt={'db':receipt,'engine':engine_receipt,'archive_rows':len(archive),'archive_last_available':str(archive.date.max()),
             'oi_availability':float(available.mean()),'market_end':str(d['end_date'][-1]),
             'oi_trades':[{'entry':str(dates[t.entry_position]),'exit':str(dates[t.exit_position])} for t in trades],
             'market_hash':hashlib.sha256(market.to_csv(index=False).encode()).hexdigest(),
             'funding_hash':hashlib.sha256(fund.to_csv(index=False).encode()).hexdigest()}
    if available.mean()<.99:raise ValueError('Archive coverage below99%; refusing cash substitution')
    return d,targets,events,receipt


def run():
    reg=register();d,p,e,receipt=context();reports={}
    windows={'common_to_september':(START,END),'july_to_september':('2026-07-01',END),
             'extension_since_aug4':('2026-08-04',END),'september_only':('2026-09-01',END)}
    for name,(start,end) in windows.items():
        mask=(d['date']>=pd.Timestamp(start).tz_localize(None).to_datetime64())&(d['end_date']<=pd.Timestamp(end).tz_localize(None).to_datetime64())
        pp=p[mask];ee=e[mask].copy();ee[0]=True
        reports[name]={str(c):dict(zip(LABELS,opt.simulate(base.subset(d,mask),pp,ee,WEIGHTS,c))) for c in DESIGN['costs']}
    result={'registration':reg,'receipt':receipt,'reports':reports,'live_enabled':False,
            'notes':['September only covers Sep1-Sep5 00:00UTC, not the whole month.',
                     'Independent window resets initialize known positions from cash; not chained returns.',
                     'OI archive has five-minute publication proxy delay; not live-arrival parity.',
                     'Long/short net cap overrides sleeve holds only at active execution events.']}
    base.write_json(OUT/'report.json',result)
    for name in reports:
        print(name,json.dumps(reports[name]['0.0006']))
    print('receipt',json.dumps({k:v for k,v in receipt.items() if k not in ['db','engine']}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');args=parser.parse_args()
    register() if args.freeze else run()
