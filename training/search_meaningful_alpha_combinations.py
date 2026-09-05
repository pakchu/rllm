"""Bounded economic, ML, and net-position portfolio discovery (research only).

Features at UTC hour T use completed data before T; orders execute at T+5m.
Candidate ranking uses 2023 ONLY. 2024+ is exposed historical report data and
cannot rerank the frozen winner. Cost ratios and trade-frequency are reports,
not exclusion gates. Same-symbol sleeves offset before turnover/cost calculation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/home/pakchu/rllm/data')
MARKET = DATA / 'cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz'
FUNDING = DATA / 'binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz'
OUT = ROOT / 'research/meaningful_combinations'
SEED = 20260905
DESIGN = {
    'version': 1, 'seed': SEED,
    'fit': ['2020-03-01', '2023-01-01'],
    'selection': ['2023-01-01', '2024-01-01'],
    'report_only': ['2024-01-01', 'source_end'],
    'data_contamination': 'All historical report windows were exposed in prior research; not pristine OOS or live authorization.',
    'execution': 'completed hourly features; exact next 5m open at T+5m; rebalance net units; charge every actual notional change; liquidate window end',
    'risk': 'same-symbol long/short offset; absolute net exposure cap 1.0; no position-overlap exclusion',
    'costs_per_side': [0.0, 0.0006, 0.001],
    'funding': 'actual timestamps/rates; mark_price if available, else settlement 5m open proxy (explicit approximation)',
    'rebalance_hours': [1, 6, 24],
    'volatility_sizing': 'raw 1x and 20% annualized target using past 24h volatility, clipped [0.10,1.0]',
    'models': {'ridge': {'alpha': 100.0}, 'hgb': {'max_iter': 150, 'max_leaf_nodes': 7, 'min_samples_leaf': 200, 'l2_regularization': 10.0, 'learning_rate': 0.05}, 'extra': {'n_estimators': 100, 'max_depth': 5, 'min_samples_leaf': 100, 'max_features': 0.7}},
    'forecast_horizons_hours': [6, 24],
    'forecast_thresholds': [0.0, 0.0012],
    'portfolio': 'top 12 economic/model candidates by 2023 rank; all pairs weights .25/.5/.75; equal and training-inverse-vol mixtures; no correlation rejection',
    'selection_score': 'min(H1,H2 return/std hourly annualized) + .25*full2023 return/MDD - 2*number of negative half-years',
    'freeze': 'same ranking rule for all candidates; no 2024+ substitution; report five fixed finalists and controls',
    'llm_rl': 'deferred until a viable economic teacher exists; no PPOSM action/AUC gate',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''): h.update(block)
    return h.hexdigest()


def write_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False, default=str) + '\n')


def freeze_design():
    payload = {'design': DESIGN, 'source_hashes': {'market': sha(MARKET), 'funding': sha(FUNDING)}, 'code_sha256': sha(__file__)}
    path = OUT / 'design.json'
    if path.exists() and json.loads(path.read_text()) != payload:
        raise RuntimeError('Refusing to replace frozen study design')
    write_json(path, payload)
    return payload


def load_sources():
    cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 'taker_buy_quote', 'number_of_trades']
    m = pd.read_csv(MARKET, usecols=cols)
    m['date'] = pd.to_datetime(m.date, utc=True).dt.tz_convert(None)
    m = m.sort_values('date').reset_index(drop=True)
    if m.date.duplicated().any() or not m.date.diff().dropna().eq(pd.Timedelta('5min')).all():
        raise ValueError('Market must have unique continuous 5m grid')
    if not np.isfinite(m[cols[1:]].to_numpy()).all() or (m[['open','high','low','close']] <= 0).any().any():
        raise ValueError('Invalid market values')
    f = pd.read_csv(FUNDING)
    f['date'] = pd.to_datetime(f['funding_time'], unit='ms', utc=True).dt.tz_convert(None)
    if f.date.duplicated().any(): raise ValueError('Duplicate funding timestamp')
    f = f.sort_values('date').reset_index(drop=True)
    validate_funding(f, m.date.iloc[-1])
    return m, f


def validate_funding(funding, market_end):
    rates = pd.to_numeric(funding.funding_rate, errors='coerce').to_numpy(float)
    dates = pd.DatetimeIndex(funding.date)
    if len(dates)<2 or dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing or not np.isfinite(rates).all():
        raise ValueError('Invalid funding event identity/rates')
    if np.max(np.diff(dates.asi8))>pd.Timedelta('12h').value:
        raise ValueError('Incomplete funding: gap exceeds 12h freshness contract')
    eligible=dates[dates<=pd.Timestamp(market_end)]
    if dates[0]>pd.Timestamp('2020-03-01') or not len(eligible) or pd.Timestamp(market_end)-eligible[-1]>pd.Timedelta('12h'):
        raise ValueError('Incomplete funding start/end coverage')


def window_mask(data, start, end):
    return (data['date']>=np.datetime64(start)) & (data['end_date']<=np.datetime64(end))


def features(m, funding):
    """Every row labelled T depends on bars ending at or before T."""
    agg = {'open':'first','high':'max','low':'min','close':'last','volume':'sum','quote_asset_volume':'sum','taker_buy_quote':'sum','number_of_trades':'sum'}
    h = m.set_index('date').resample('1h', label='right', closed='left').agg(agg)
    counts = m.set_index('date')['close'].resample('1h', label='right', closed='left').count()
    h = h.loc[counts.eq(12)]
    c = h.close; log = np.log(c); r = log.diff()
    x = pd.DataFrame(index=h.index)
    for w in [6,24,72,168,720]:
        vol = r.rolling(w, min_periods=w).std(ddof=0)
        x[f'mom{w}'] = log.diff(w) / (vol * np.sqrt(w)).replace(0,np.nan)
        x[f'flow{w}'] = (2*h.taker_buy_quote-h.quote_asset_volume).rolling(w).sum() / h.quote_asset_volume.rolling(w).sum().replace(0,np.nan)
        mean = c.rolling(w).mean(); std = c.rolling(w).std(ddof=0)
        x[f'z{w}'] = (c-mean)/std.replace(0,np.nan)
    x['vol24'] = r.rolling(24).std(ddof=0)
    x['volratio'] = x.vol24 / r.rolling(168).std(ddof=0).replace(0,np.nan)
    x['range_pos'] = (c-h.low.rolling(24).min())/(h.high.rolling(24).max()-h.low.rolling(24).min()).replace(0,np.nan)
    x['volume_ratio'] = h.quote_asset_volume.rolling(6).mean()/h.quote_asset_volume.rolling(168).mean().replace(0,np.nan)
    x['breakout'] = np.where(c > h.high.shift(1).rolling(24).max(),1,np.where(c < h.low.shift(1).rolling(24).min(),-1,0))
    fs = funding[['date','funding_rate']].rename(columns={'date':'funding_date'})
    join = pd.merge_asof(pd.DataFrame({'date':x.index}), fs, left_on='date',right_on='funding_date', direction='backward', tolerance=pd.Timedelta('12h'))
    x['funding'] = join.funding_rate.to_numpy()
    x['funding_available'] = join.funding_date.notna().to_numpy(float)
    x['trend_flow'] = x.mom168*x.flow24
    x['trend_pullback'] = x.mom168*x.z24
    x['carry_trend'] = x.funding*x.mom168
    return x.replace([np.inf,-np.inf],np.nan)


def execution_blocks(m, f, x):
    """Exact :05 execution blocks of twelve 5m bars; final close liquidation."""
    indexed = m.set_index('date')
    start = x.index + pd.Timedelta('5min')
    positions = indexed.index.get_indexer(start)
    valid = (positions >= 0) & (positions+12 < len(indexed))
    positions = positions[valid]; x = x.loc[valid].copy()
    assert np.all(np.diff(positions)==12)
    opens = m.open.to_numpy(float); closes = m.close.to_numpy(float)
    high = m.high.to_numpy(float); low = m.low.to_numpy(float)
    # Funding transfer per unit: funding_rate times mark, allocated to event bar.
    transfers = np.zeros(len(m))
    funding_positions = np.searchsorted(m.date.to_numpy(),f.date.to_numpy(),side='right')-1
    use = (funding_positions>=0)&(funding_positions<len(m))&(f.date.to_numpy()<=m.date.to_numpy()[-1])
    pos = funding_positions[use]; ff = f.loc[use]
    marks = pd.to_numeric(ff.mark_price,errors='coerce').to_numpy(float)
    fallback = ~np.isfinite(marks)|(marks<=0)
    marks[fallback] = opens[pos[fallback]]
    np.add.at(transfers,pos,ff.funding_rate.to_numpy(float)*marks)
    rows = positions[:,None]+np.arange(12)[None,:]
    # Maximum adverse/favorable bar path kept for finalist replay.
    data = {'date':start[valid].to_numpy(), 'end_date':m.date.to_numpy()[positions+12], 'open':opens[positions],
            'end':opens[positions+12], 'high':high[rows].max(axis=1), 'low':low[rows].min(axis=1),
            'funding':transfers[rows].sum(axis=1),
            'debit':np.maximum(transfers[rows],0).sum(axis=1),
            'credit':np.minimum(transfers[rows],0).sum(axis=1),
            'hi5':high[rows], 'lo5':low[rows], 'op5':opens[rows], 'cl5':closes[rows], 'fund5':transfers[rows]}
    return x, data, {'missing_mark_proxy_events':int(fallback.sum()), 'funding_events':int(use.sum()), 'market_end':str(m.date.iloc[-1])}


def simulate(data, positions, cost=0.0006, fine=False):
    """Vectorized independent portfolios, cash-margined linear contract.

    Target notional is fraction of pre-fee equity. Long/short sleeves must be
    netted BEFORE calling. A conservative high-before-low envelope measures
    MDD; fine=True resolves five-minute bars instead of hourly envelopes.
    """
    p = np.asarray(positions,float)
    if p.ndim==1: p=p[:,None]
    if p.shape[0]!=len(data['open']) or not np.isfinite(p).all() or np.max(np.abs(p),initial=0)>1.0000001:
        raise ValueError('Positions invalid or net cap exceeded')
    n,k=p.shape; eq=np.ones(k); peak=eq.copy(); mdd=np.zeros(k); q=np.zeros(k)
    fees=np.zeros(k); funding_paid=np.zeros(k); turnover=np.zeros(k); entries=np.zeros(k,int); changes=np.zeros(k,int)
    returns=np.zeros((n,k)); prev_close=float(data['open'][0]); dead=np.zeros(k,bool)
    for i in range(n):
        op=data['open'][i]; eq += q*(op-prev_close)
        prior=eq.copy(); target=np.where(dead,0,p[i]*eq/op)
        trade=(target-q)*op; charge=np.abs(trade)*cost
        entries += ((target*q<=0)&(np.abs(target)>1e-15)).astype(int)
        changes += (np.abs(trade)>1e-10).astype(int)
        turnover += np.abs(trade)/np.maximum(prior,1e-12)
        fees += charge; eq-=charge; q=target
        if fine:
            for j in range(12):
                mid=data['op5'][i,j]; eq+=q*(mid-op); op=mid
                transfer=q*data['fund5'][i,j]; funding_paid+=transfer
                high=eq+q*(np.where(q>=0,data['hi5'][i,j],data['lo5'][i,j])-op)+np.maximum(-transfer,0)
                low=eq+q*(np.where(q>=0,data['lo5'][i,j],data['hi5'][i,j])-op)-np.maximum(transfer,0)
                peak=np.maximum(peak,high); mdd=np.maximum(mdd,1-low/np.maximum(peak,1e-12))
                eq-=transfer
        else:
            debit=np.where(q>=0,q*data['debit'][i],q*data['credit'][i])
            credit=np.where(q>=0,-q*data['credit'][i],-q*data['debit'][i])
            high=eq+q*(np.where(q>=0,data['high'][i],data['low'][i])-op)+credit
            low=eq+q*(np.where(q>=0,data['low'][i],data['high'][i])-op)-debit
            peak=np.maximum(peak,high);mdd=np.maximum(mdd,1-low/np.maximum(peak,1e-12))
            transfer=q*data['funding'][i];funding_paid+=transfer;eq-=transfer
        end=data['end'][i];eq+=q*(end-op);prev_close=end
        if i==n-1:
            charge=np.abs(q)*end*cost;fees+=charge;eq-=charge;q=np.zeros(k)
        dead |= eq<=0
        eq=np.maximum(eq,0); q[dead]=0; peak=np.maximum(peak,eq);mdd=np.maximum(mdd,1-eq/np.maximum(peak,1e-12))
        returns[i]=eq/np.maximum(prior,1e-12)-1
    years=max(n/(365.25*24),1e-9)
    cagr=np.maximum(eq,0)**(1/years)-1
    sd=returns.std(axis=0);sharpe=np.divide(returns.mean(axis=0)*np.sqrt(365.25*24),sd,out=np.zeros(k),where=sd>1e-12)
    return {'equity':eq,'return_pct':(eq-1)*100,'cagr_pct':cagr*100,'mdd_pct':mdd*100,'calmar':np.divide(cagr,mdd,out=np.zeros(k),where=mdd>1e-12),'sharpe':sharpe,'entry_episodes':entries,'rebalance_orders':changes,'turnover':turnover,'fees_pct_initial':fees*100,'funding_pct_initial':funding_paid*100,'returns':returns}


def subset(data, mask):
    return {key:value[mask] for key,value in data.items()}


def hold_signal(raw, hours, dates):
    s=pd.Series(np.asarray(raw,float), index=pd.DatetimeIndex(dates))
    mask=(s.index.hour%hours==0)
    return s.where(mask).ffill().fillna(0).to_numpy()


def make_candidates(x, data):
    candidates={}; specs={}
    def add(name,raw,desc):
        raw=np.nan_to_num(np.asarray(raw,float));raw=np.clip(raw,-1,1)
        for hold in DESIGN['rebalance_hours']:
            key=f'{name}__reb{hold}h';candidates[key]=hold_signal(raw,hold,x.index);specs[key]={'family':name.split('_')[0],'rationale':desc,'rebalance_hours':hold}
            size=np.clip(.20/(x.vol24.to_numpy()*np.sqrt(365.25*24)),.10,1.0)
            scaled=f'{key}__vol20';candidates[scaled]=hold_signal(np.nan_to_num(raw*size),hold,x.index)
            specs[scaled]={**specs[key], 'annual_vol_target':.20}
    for w in [24,168,720]:
        trend=np.sign(x[f'mom{w}'].to_numpy());strong=np.abs(x[f'mom{w}'].to_numpy())>0.75
        add(f'trend_{w}',np.where(strong,trend,0),'time-series trend persistence above noise threshold')
        add(f'pullback_{w}',np.where(strong & (trend*x.z24.to_numpy()<0),trend,0),'enter against short-term displacement within established trend')
        add(f'flowtrend_{w}',np.where(strong & (trend*x.flow6.to_numpy()>0.02),trend,0),'aggressive order flow confirms medium-term trend')
    add('exhaustion_flow',np.where((np.abs(x.z24)>1.5)&(np.sign(x.z24)*x.flow6<0),-np.sign(x.z24),0),'price displacement opposed by aggressive flow: exhaustion reversal')
    add('range_reversion',np.where((np.abs(x.mom168)<0.75)&(np.abs(x.z24)>1.5),-np.sign(x.z24),0),'mean reversion only in non-trending regimes')
    add('breakout_compression',np.where((x.volratio<1)&(x.volume_ratio>1),x.breakout,0),'breakout during volatility compression with participation')
    add('carry_contrarian',np.where((np.abs(x.funding)>0.00005)&(np.abs(x.mom168)<1.5),-np.sign(x.funding),0),'fade crowded funding only outside strong trends')
    add('carry_trend',np.where((np.abs(x.mom168)>0.75)&(np.sign(x.mom168)*x.funding<=0),np.sign(x.mom168),0),'trend continuation with favorable funding')
    add('flow_divergence',np.where((np.abs(x.mom24)>0.75)&(np.sign(x.mom24)*x.flow24<-0.02),np.sign(x.flow24),0),'persistent aggressive flow opposes recent price movement')
    fit=(x.index>=pd.Timestamp('2020-03-01'))&(x.index<pd.Timestamp('2023-01-01'))
    xx=x.to_numpy(float)
    model_notes={}
    for horizon in DESIGN['forecast_horizons_hours']:
        y=pd.Series(data['open']).shift(-horizon).to_numpy()/data['open']-1
        mature=pd.Series(x.index+pd.Timedelta(hours=horizon,minutes=5)).to_numpy()<np.datetime64('2023-01-01')
        use=fit&mature&np.isfinite(y)
        for kind in ['ridge','hgb','extra']:
            kw=DESIGN['models'][kind]
            if kind=='ridge': model=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(**kw))
            elif kind=='hgb':model=make_pipeline(SimpleImputer(strategy='median'),HistGradientBoostingRegressor(**kw,early_stopping=False,random_state=SEED))
            else:model=make_pipeline(SimpleImputer(strategy='median'),ExtraTreesRegressor(**kw,random_state=SEED,n_jobs=2))
            model.fit(xx[use],y[use]);score=model.predict(xx)
            model_notes[f'{kind}_{horizon}h']={'train_rows':int(use.sum()),'train_last':str(x.index[use][-1]),'target_horizon':horizon,'prediction_sha256':hashlib.sha256(score.tobytes()).hexdigest()}
            for threshold in DESIGN['forecast_thresholds']:
                name=f'ml{kind}_{horizon}h_t{threshold}'
                add(name,np.where(np.abs(score)>threshold,np.sign(score),0),'fixed-model causal expected return from joint trend/flow/funding state')
    return candidates,specs,model_notes


def stats_row(stats,i=0):
    return {key:(int(value[i]) if key in ['entry_episodes','rebalance_orders'] else float(value[i])) for key,value in stats.items() if key not in ['returns','equity']}


def selection_rank(data, positions):
    use=window_mask(data,'2023-01-01','2024-01-01')
    d=subset(data,use);p=positions[use]
    full=simulate(d,p)
    first=window_mask(d,'2023-01-01','2023-07-01');second=window_mask(d,'2023-07-01','2024-01-01')
    h1=simulate(subset(d,first),p[first]);h2=simulate(subset(d,second),p[second])
    score=np.minimum(h1['sharpe'],h2['sharpe'])+.25*full['calmar']-2*((h1['return_pct']<0).astype(int)+(h2['return_pct']<0).astype(int))
    return score,full,h1,h2


def run():
    design=json.loads((OUT/'design.json').read_text())
    if design['design']!=DESIGN or design['code_sha256']!=sha(__file__) or design['source_hashes']!={'market':sha(MARKET),'funding':sha(FUNDING)}:raise RuntimeError('Study preregistration drift')
    m,f=load_sources();x=features(m,f);x,data,receipt=execution_blocks(m,f,x)
    candidates,specs,model_notes=make_candidates(x,data)
    names=list(candidates);positions=np.column_stack([candidates[n] for n in names]);ranks,ss,h1,h2=selection_rank(data,positions)
    # Exact duplicates are retained in inventory but not allowed to dominate mixture construction.
    seen=set(); top=[]
    selection_mask=window_mask(data,'2023-01-01','2024-01-01')
    for idx in np.argsort(-ranks):
        digest=hashlib.sha256(positions[selection_mask,idx].tobytes()).hexdigest()
        if digest in seen:continue
        seen.add(digest);top.append(int(idx))
        if len(top)==12:break
    for ai,i in enumerate(top):
        for j in top[ai+1:]:
            for weight in [.25,.5,.75]:
                name=f'mix_{i}_{j}_{weight}';candidates[name]=weight*positions[:,i]+(1-weight)*positions[:,j]
                specs[name]={'family':'portfolio','components':{names[i]:weight,names[j]:1-weight},'rationale':'netted complementary sleeves; no overlap rejection'}
    fit=window_mask(data,'2020-03-01','2023-01-01')
    training=simulate(subset(data,fit),positions[fit][:,top]);vol=training['returns'].std(axis=0)
    for mode in ['equal','inversevol']:
        weights=np.ones(len(top)) if mode=='equal' else 1/np.maximum(vol,1e-6);weights/=weights.sum()
        name=f'portfolio_{mode}';candidates[name]=positions[:,top]@weights
        specs[name]={'family':'portfolio','components':{names[i]:float(w) for i,w in zip(top,weights)},'rationale':'diversified aggregate with net direction and notional risk'}
    names=list(candidates);positions=np.column_stack([candidates[n] for n in names]);ranks,ss,h1,h2=selection_rank(data,positions)
    order=np.argsort(-ranks,kind='stable');selected=[int(i) for i in order[:5]]
    frozen={'selection_window':'2023 only','historical_report_reranking':False,'candidates':len(names),'top':[{ 'name':names[i],'rank_score':float(ranks[i]),'spec':specs[names[i]],'selection':stats_row(ss,i),'h1':stats_row(h1,i),'h2':stats_row(h2,i)} for i in selected], 'positions_sha256':hashlib.sha256(positions[:,selected].tobytes()).hexdigest()}
    write_json(OUT/'selection_freeze.json',frozen)
    # No candidate is selected or replaced using these historical report windows.
    finalist=positions[:,selected]; report_names=[names[i] for i in selected]+['control_long','control_cash']
    finalist=np.column_stack([finalist,np.ones(len(x)),np.zeros(len(x))])
    windows={'selection2023':['2023-01-01','2024-01-01'],'report2024':['2024-01-01','2025-01-01'],'report2025':['2025-01-01','2026-01-01'],'report2026':['2026-01-01','2026-06-01'],'report_combined':['2024-01-01','2026-06-01']}
    reports={}
    for window,(start,end) in windows.items():
        mask=window_mask(data,start,end);reports[window]={}
        for cost in DESIGN['costs_per_side']:
            st=simulate(subset(data,mask),finalist[mask],cost=cost,fine=True)
            reports[window][str(cost)]={name:stats_row(st,i) for i,name in enumerate(report_names)}
    inventory=[{'name':names[i],'spec':specs[names[i]],'rank_score':float(ranks[i]),'selection':stats_row(ss,i)} for i in order]
    result={'design':design,'data_receipt':receipt,'rows':len(x),'features':list(x.columns),'models':model_notes,'search_candidates':len(names),'selection_freeze':frozen,'reports':reports,'inventory':inventory,'live_authorized':False,'cautions':['Historical test/EVAL previously exposed; results are research diagnostics.','Intrabar high-before-low MDD is conservative, not observed tick ordering.','Missing funding mark prices use settlement 5m open proxy.','Parameter grid bounded, not all mathematical combinations; portfolio search adds selection multiplicity.','No liquidation, market impact, latency, or exchange capacity model; net exposure capped at 1x.']}
    write_json(OUT/'report.json',result)
    export={'research_only':True,'live_enabled':False,'winner':frozen['top'][0],'net_exposure_cap':1.0,'allow_sleeve_overlap':True,'long_short_offset':True,'cost_ratio_gate':False,'trade_frequency_gate':False,'signal_time':'completed-hour T','execution':'T+5m','design_sha256':sha(OUT/'design.json')}
    write_json(OUT/'research_config.json',export)
    for window in reports:
        print(window, json.dumps(reports[window]['0.0006'][report_names[0]]))
    print('winner',report_names[0], 'candidates',len(names),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');args=parser.parse_args()
    if args.freeze:freeze_design()
    else:run()
