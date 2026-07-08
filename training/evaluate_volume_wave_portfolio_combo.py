import json, math, itertools
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import numpy as np, pandas as pd
import training.evaluate_portfolio_llm_selector as ep
from training.wave_feature_ridge_policy import build_wave_feature_frame, _price_efficiency, _rolling_z, _ret

OUT='results/all_alpha_volume_wave_portfolio_combo_2026-07-08.json'
DOC='docs/all-alpha-volume-wave-portfolio-combo-2026-07-08.md'
SLEEVES=['pb30_base','pb30_addon','nonpb30_taker','oi_raw','rex_rule','short_kimchi3d','short_premium_panic','oi_wave_lowpos144','oi_wave_slope288_low','rex_wave_pricez288_low','rex_wave_vol144_high','short_premium_wave_lowvol','oi_vol_alt_ratio72','oi_vol_volmom288','oi_vol_alt_ratio288','oi_upbit_ratio288_low']
COST=0.0005

def J(p): return json.loads(Path(p).read_text())

def gate(feat,gates):
    a=np.ones(len(feat),bool)
    for g in gates:
        x=feat[str(g['feature'])].to_numpy(float); thr=float(g.get('threshold',g.get('thr'))); op=str(g['op'])
        a &= ((x>=thr) if op in ('>=','ge') else (x<=thr)) & np.isfinite(x)
    return a

def add_struct_wave(m, feat):
    c=m.close.astype(float); h=m.high.astype(float); l=m.low.astype(float); v=m.volume.astype(float); qv=m.get('quote_asset_volume',v*c).astype(float); tbq=m.get('taker_buy_quote',qv*0.5).astype(float)
    pc=c.shift(1); tr=pd.concat([(h-l),(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    atr72=(tr.rolling(72,min_periods=18).mean()/c.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).fillna(0)
    taker=(tbq/qv.replace(0,np.nan))*2-1
    out=pd.DataFrame(index=m.index)
    for n in [12,24,72,144,288,576]: out[f'w_ret_{n}']=_ret(c,n)
    out['w_atr_72']=atr72
    out['w_taker_z_144']=_rolling_z(taker.fillna(0),144)
    out['w_vol_z_144']=_rolling_z(v,144)
    for W in [72,144,288,576,864]:
        rh=h.shift(1).rolling(W,min_periods=max(20,W//3)).max(); rl=l.shift(1).rolling(W,min_periods=max(20,W//3)).min(); rng=(rh-rl).replace(0,np.nan)
        out[f'w_pos_{W}']=((c-rl)/rng).replace([np.inf,-np.inf],np.nan).clip(-2,3).fillna(0.5)
        out[f'w_retr_{W}']=((rh-c)/rng).replace([np.inf,-np.inf],np.nan).clip(-2,3).fillna(0.5)
        out[f'w_break_high_{W}']=((c-rh)/c.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).clip(-1,1).fillna(0)
        out[f'w_break_low_{W}']=((rl-c)/c.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).clip(-1,1).fillna(0)
        out[f'w_eff_{W}']=_price_efficiency(c,W)
        ma=c.rolling(W,min_periods=max(20,W//3)).mean(); sd=c.rolling(W,min_periods=max(20,W//3)).std(ddof=0)
        out[f'w_price_z_{W}']=((c-ma)/sd.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).clip(-5,5).fillna(0)
        out[f'w_slope_atr_{W}']=((c.ewm(span=max(4,W//8),adjust=False).mean()-ma)/(c*atr72).replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).clip(-10,10).fillna(0)
    wf=build_wave_feature_frame(m, window=144).add_prefix('wr_')
    return pd.concat([feat, wf, out], axis=1).loc[:,lambda x:~x.columns.duplicated(keep='last')].replace([np.inf,-np.inf],np.nan)



def add_volume_gate_features(m, feat):
    v=m.volume.astype(float); qv=m.quote_asset_volume.astype(float) if 'quote_asset_volume' in m else v*m.close.astype(float)
    def z(s,n):
        mu=s.rolling(n,min_periods=max(12,n//4)).mean(); sd=s.rolling(n,min_periods=max(12,n//4)).std(ddof=0)
        return ((s-mu)/sd.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).clip(-8,8).fillna(0)
    def mom(s,n):
        return (s/s.shift(n).replace(0,np.nan)-1).replace([np.inf,-np.inf],np.nan).clip(-10,10).fillna(0)
    feat['vg_vol_mom_288']=mom(v+1,288)
    feat['vg_qvol_mom_288']=mom(qv+1,288)
    # Alt quote-volume / BTC quote-volume ratio, same diagnostic family as volume scan.
    import glob, os
    total=None
    base=m[['date']].copy(); base['btc_qv']=qv.to_numpy(float)
    for p in glob.glob('data/binance_um_pool_5m_2023_2026/*_5m_*.csv.gz'):
        sym=os.path.basename(p).split('_')[0]
        try:
            a=pd.read_csv(p, usecols=['date','quote_asset_volume'])
        except Exception:
            continue
        a['date']=pd.to_datetime(a.date)
        a=a.sort_values('date').rename(columns={'quote_asset_volume':f'{sym}_qv'})
        joined=pd.merge_asof(m[['date']].sort_values('date'), a, on='date', direction='backward', tolerance=pd.Timedelta('7min')).sort_index()
        s=joined[f'{sym}_qv'].fillna(0).astype(float)
        total=s if total is None else total+s
    if total is None:
        total=pd.Series(0.0,index=m.index)
    rel=(total/qv.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).fillna(0)
    feat['vg_alt_btc_qv_ratio_z_72']=z(rel,72)
    feat['vg_alt_btc_qv_ratio_z_288']=z(rel,288)
    # Upbit/Binance volume ratio if cached wave_trading data exists.
    ups=[]
    for p in ['/home/pakchu/workspace/wave_trading/data/2020-01-01_2025-12-15_4bd081fc54811fccdee66850692c435e.csv.gz','/home/pakchu/workspace/wave_trading/data/2026-01-01_2026-06-02_a349bf03b5f3b2154b91a7b363386c08.csv.gz']:
        if Path(p).exists():
            u=pd.read_csv(p, usecols=['date','close','volume']); u['date']=pd.to_datetime(u.date, utc=True).dt.tz_convert(None); ups.append(u)
    if ups and 'usdkrw' in m:
        u=pd.concat(ups).drop_duplicates('date').sort_values('date')
        joined=pd.merge_asof(m[['date','quote_asset_volume','usdkrw']].sort_values('date'), u, on='date', direction='backward', tolerance=pd.Timedelta('7min')).sort_index()
        up_ratio=(joined['volume'].fillna(0)*joined['close'].fillna(0))/(joined['quote_asset_volume'].replace(0,np.nan)*joined['usdkrw'].replace(0,np.nan))
        feat['vg_upbit_binance_vol_ratio_z_288']=z(up_ratio.fillna(0),288)
    else:
        feat['vg_upbit_binance_vol_ratio_z_288']=0.0
    return feat.replace([np.inf,-np.inf],np.nan)

def qtrain(feat, masks, col, q):
    vals=feat.loc[masks['train'], col].to_numpy(float)
    vals=vals[np.isfinite(vals)]
    return float(np.quantile(vals, q))

def append_active(events,market,feat,masks,active,sleeve,side,hold,stride):
    n=len(market); dates=pd.to_datetime(market['date'])
    for sp,sm in masks.items():
        nxt=0
        for p in np.arange(143,n-hold-2,stride,dtype=np.int64):
            ip=int(p); xp=ip+1+hold
            if active[ip] and sm[ip] and ip>=nxt and xp<n and sm[min(xp,n-1)]:
                r,adv,real=ep._event_return(market,ip,hold,side,cost=COST)
                events.append({'split':sp,'sleeve':sleeve,'side':side,'signal_pos':ip,'date':str(dates.iloc[ip]),'ret_bps':real*10000.0,'ret':r,'adv':adv})
                nxt=xp

def append_rex(events,market,feat,masks,sleeve='rex_rule', extra_active=None):
    rows=[]
    for f in ['results/rex_dual_regime_train_2021_2023_predictions_2026-07-03.jsonl','results/rex_dual_regime_test_2024_predictions_2026-07-03.jsonl','results/rex_dual_regime_eval_2025_2026h1_predictions_2026-07-03.jsonl']:
        if Path(f).exists(): rows += [json.loads(l) for l in Path(f).read_text().splitlines() if l.strip()]
    rows.sort(key=lambda r:int(r['signal_pos'])); n=len(market); dates=pd.to_datetime(market['date'])
    active = np.ones(n,bool) if extra_active is None else np.asarray(extra_active,bool)
    for sp,sm in masks.items():
        nxt=0
        for row in rows:
            pred=row.get('prediction') or {}
            if pred.get('gate')!='TRADE': continue
            side=str(pred.get('side','')).lower(); hold=int(pred.get('hold_bars') or 0); ip=int(row['signal_pos']); xp=ip+1+hold
            if side not in ('long','short') or hold<=0 or not active[ip]: continue
            if ip>=143 and ip<n-hold-2 and ip>=nxt and sm[ip] and xp<n and sm[min(xp,n-1)]:
                r,adv,real=ep._event_return(market,ip,hold,side,cost=COST)
                events.append({'split':sp,'sleeve':sleeve,'side':side,'signal_pos':ip,'date':str(dates.iloc[ip]),'ret_bps':real*10000.0,'ret':r,'adv':adv})
                nxt=xp

def build_events():
    market,feat,masks,years=ep._prep(); feat=add_struct_wave(market,feat); feat=add_volume_gate_features(market,feat); events=[]
    pb=J('configs/live/bullish_pb30_addon_returnz_htf1w_candidate.json')
    for name,key in [('pb30_base','base_module'),('pb30_addon','addon_module')]:
        m=pb[key]; append_active(events,market,feat,masks,gate(feat,m['gates']),name,'long',int(m['hold_bars_5m']),int(m['stride_bars_5m']))
    nt=J('configs/live/nonpb30_taker_returnz_rangevol_htf4hrange_h72_candidate.json')['signal']
    append_active(events,market,feat,masks,gate(feat,nt['gates']),'nonpb30_taker','long',int(nt['hold_bars_5m']),int(nt['stride_bars_5m']))
    oi=J('configs/live/oi_divergence_sma24_highfreq_h30_s6_candidate.json')
    oi_active=gate(feat,oi['gates'])
    append_active(events,market,feat,masks,oi_active,'oi_raw','long',int(oi['hold_bars']),int(oi['stride_bars']))
    append_rex(events,market,feat,masks,'rex_rule')
    gA=[{'feature':'htf_3d_return_1','op':'<=','thr':-0.03252949727545951},{'feature':'kimchi_premium_change','op':'<=','thr':-0.00465405239399725}]
    append_active(events,market,feat,masks,gate(feat,gA),'short_kimchi3d','short',144,12)
    gB=[{'feature':'htf_1d_return_4','op':'<=','thr':-0.07234231335497887},{'feature':'premium_index_zscore','op':'<=','thr':-1.472093119977103}]
    sp_active=gate(feat,gB)
    append_active(events,market,feat,masks,sp_active,'short_premium_panic','short',144,12)

    # Meaningful wave-feature alpha sleeves from the previous diagnostic.
    # Thresholds are train<2024 quantiles only.
    oi_lowpos_thr=qtrain(feat,masks,'w_pos_144',0.20)
    oi_slope_thr=qtrain(feat,masks,'w_slope_atr_288',0.20)
    rex_price_thr=qtrain(feat,masks,'w_price_z_288',0.20)
    rex_vol_thr=qtrain(feat,masks,'w_vol_z_144',0.80)
    sp_lowvol_thr=qtrain(feat,masks,'w_vol_z_144',0.20)
    append_active(events,market,feat,masks,oi_active & (feat['w_pos_144'].to_numpy(float)<=oi_lowpos_thr),'oi_wave_lowpos144','long',int(oi['hold_bars']),int(oi['stride_bars']))
    append_active(events,market,feat,masks,oi_active & (feat['w_slope_atr_288'].to_numpy(float)<=oi_slope_thr),'oi_wave_slope288_low','long',int(oi['hold_bars']),int(oi['stride_bars']))
    append_rex(events,market,feat,masks,'rex_wave_pricez288_low',feat['w_price_z_288'].to_numpy(float)<=rex_price_thr)
    append_rex(events,market,feat,masks,'rex_wave_vol144_high',feat['w_vol_z_144'].to_numpy(float)>=rex_vol_thr)
    append_active(events,market,feat,masks,sp_active & (feat['w_vol_z_144'].to_numpy(float)<=sp_lowvol_thr),'short_premium_wave_lowvol','short',144,12)
    # Volume-derived sleeves from prior volume-gate diagnostics.
    alt72_thr=qtrain(feat,masks,'vg_alt_btc_qv_ratio_z_72',0.80)
    alt288_thr=qtrain(feat,masks,'vg_alt_btc_qv_ratio_z_288',0.90)
    volmom_thr=qtrain(feat,masks,'vg_vol_mom_288',0.80)
    upbit_low_thr=qtrain(feat,masks,'vg_upbit_binance_vol_ratio_z_288',0.20)
    append_active(events,market,feat,masks,oi_active & (feat['vg_alt_btc_qv_ratio_z_72'].to_numpy(float)>=alt72_thr),'oi_vol_alt_ratio72','long',int(oi['hold_bars']),int(oi['stride_bars']))
    append_active(events,market,feat,masks,oi_active & (feat['vg_vol_mom_288'].to_numpy(float)>=volmom_thr),'oi_vol_volmom288','long',int(oi['hold_bars']),int(oi['stride_bars']))
    append_active(events,market,feat,masks,oi_active & (feat['vg_alt_btc_qv_ratio_z_288'].to_numpy(float)>=alt288_thr),'oi_vol_alt_ratio288','long',int(oi['hold_bars']),int(oi['stride_bars']))
    append_active(events,market,feat,masks,oi_active & (feat['vg_upbit_binance_vol_ratio_z_288'].to_numpy(float)<=upbit_low_thr),'oi_upbit_ratio288_low','long',int(oi['hold_bars']),int(oi['stride_bars']))
    alpha_thresholds={
        'oi_wave_lowpos144': {'feature':'w_pos_144','op':'<=','train_q':0.20,'threshold':oi_lowpos_thr},
        'oi_wave_slope288_low': {'feature':'w_slope_atr_288','op':'<=','train_q':0.20,'threshold':oi_slope_thr},
        'rex_wave_pricez288_low': {'feature':'w_price_z_288','op':'<=','train_q':0.20,'threshold':rex_price_thr},
        'rex_wave_vol144_high': {'feature':'w_vol_z_144','op':'>=','train_q':0.80,'threshold':rex_vol_thr},
        'short_premium_wave_lowvol': {'feature':'w_vol_z_144','op':'<=','train_q':0.20,'threshold':sp_lowvol_thr},
        'oi_vol_alt_ratio72': {'feature':'vg_alt_btc_qv_ratio_z_72','op':'>=','train_q':0.80,'threshold':alt72_thr},
        'oi_vol_volmom288': {'feature':'vg_vol_mom_288','op':'>=','train_q':0.80,'threshold':volmom_thr},
        'oi_vol_alt_ratio288': {'feature':'vg_alt_btc_qv_ratio_z_288','op':'>=','train_q':0.90,'threshold':alt288_thr},
        'oi_upbit_ratio288_low': {'feature':'vg_upbit_binance_vol_ratio_z_288','op':'<=','train_q':0.20,'threshold':upbit_low_thr},
    }
    return market,feat,masks,years,events,alpha_thresholds

def sparsify_events(events, masks):
    # Convert full-length return paths into split-local sparse updates once.
    starts={sp:int(np.flatnonzero(m)[0]) for sp,m in masks.items()}
    lens={sp:int(np.flatnonzero(m)[-1]-np.flatnonzero(m)[0]+1) for sp,m in masks.items()}
    for e in events:
        sp=e['split']; st=starts[sp]; ln=lens[sp]
        rnz=np.flatnonzero(np.abs(e['ret'])>1e-15)
        anz=np.flatnonzero(np.abs(e['adv'])>1e-15)
        e['ret_pos']=(rnz-st).astype(np.int32); e['ret_val']=e['ret'][rnz].astype(np.float64)
        e['adv_pos']=(anz-st).astype(np.int32); e['adv_val']=e['adv'][anz].astype(np.float64)
        # defensive clipping to split-local interval
        mr=(e['ret_pos']>=0)&(e['ret_pos']<ln); ma=(e['adv_pos']>=0)&(e['adv_pos']<ln)
        e['ret_pos']=e['ret_pos'][mr]; e['ret_val']=e['ret_val'][mr]
        e['adv_pos']=e['adv_pos'][ma]; e['adv_val']=e['adv_val'][ma]
        del e['ret']; del e['adv']
    return lens

def arrays(events,masks,allow=None, split_lens=None):
    allow = {} if allow is None else allow
    if split_lens is None:
        split_lens={sp:int(np.flatnonzero(m)[-1]-np.flatnonzero(m)[0]+1) for sp,m in masks.items()}
    by={}
    for sp,m in masks.items():
        ln=split_lens[sp]; matsR=[]; matsA=[]; counts=[]; wins=[]; sides=[]
        for sl in SLEEVES:
            r=np.zeros(ln,dtype=np.float64); a=np.zeros(ln,dtype=np.float64); c=0; w=0; sc=Counter()
            for ei,e in enumerate(events):
                if e['split']==sp and e['sleeve']==sl and allow.get(ei, True):
                    if len(e['ret_pos']): np.add.at(r,e['ret_pos'],e['ret_val'])
                    if len(e['adv_pos']): np.add.at(a,e['adv_pos'],e['adv_val'])
                    c+=1; w += float(e['ret_bps'])>0; sc[e['side']]+=1
            matsR.append(r); matsA.append(a); counts.append(c); wins.append(w); sides.append(dict(sc))
        R=np.vstack(matsR); A=np.vstack(matsA); active=np.any((R!=0)|(A!=0),axis=0)
        by[sp]={'R':R[:,active],'A':A[:,active],'counts':np.array(counts),'wins':np.array(wins),'side_counts':sides,'active_bars':int(active.sum())}
    return by

def metric_split(d,y,w):
    wv=np.array([w.get(s,0.0) for s in SLEEVES],float); r=wv@d['R']; adv=wv@d['A']
    fac=np.maximum(0,1+r); eqp=np.cumprod(fac) if len(fac) else np.array([1.0]); eqb=np.concatenate([[1.0],eqp[:-1]]) if len(fac) else np.array([1.0])
    pka=np.maximum.accumulate(eqp); pkb=np.maximum.accumulate(eqb)
    mdd=max(float(np.nanmax(1-eqp/np.maximum(pka,1e-12))) if len(eqp) else 0,float(np.nanmax(1-(eqb*(1+adv))/np.maximum(pkb,1e-12))) if len(eqb) else 0)*100
    eq=float(eqp[-1]) if len(eqp) else 1.0; ret=(eq-1)*100; cagr=((eq**(1/y)-1)*100) if eq>0 else -100; ratio=cagr/mdd if mdd>1e-12 else 0
    trades=int(np.sum(d['counts'][wv!=0])); wins=int(np.sum(d['wins'][wv!=0]))
    vals=r[np.abs(r)>1e-12]
    sharpe=float(np.mean(vals)/np.std(vals,ddof=1)*np.sqrt(len(vals))) if len(vals)>1 and np.std(vals,ddof=1)>0 else 0
    return {'total_return_pct':ret,'cagr_pct':cagr,'strict_mdd_pct':mdd,'cagr_to_strict_mdd':ratio,'trade_entries':trades,'win_rate':wins/trades if trades else 0,'active_bars':d['active_bars'],'bar_sharpe_like':sharpe,'sleeve_trade_counts':{s:int(c) if w.get(s,0)!=0 else 0 for s,c in zip(SLEEVES,d['counts'])}}

def metrics(by,years,w): return {sp:metric_split(by[sp],years[sp],w) for sp in ep.SPLITS}

def weight_grid():
    seeds=[]
    base={'pb30_base':0,'pb30_addon':0,'nonpb30_taker':2.0,'oi_raw':1.0,'rex_rule':2.5,'short_kimchi3d':0,'short_premium_panic':0,
          'oi_wave_lowpos144':0,'oi_wave_slope288_low':0,'rex_wave_pricez288_low':0,'rex_wave_vol144_high':0,'short_premium_wave_lowvol':0,
          'oi_vol_alt_ratio72':0,'oi_vol_volmom288':0,'oi_vol_alt_ratio288':0,'oi_upbit_ratio288_low':0}
    seeds.append(dict(base))
    for add in ['oi_wave_lowpos144','oi_wave_slope288_low','rex_wave_pricez288_low','rex_wave_vol144_high','short_premium_wave_lowvol','oi_vol_alt_ratio72','oi_vol_volmom288','oi_vol_alt_ratio288','oi_upbit_ratio288_low']:
        w=dict(base); w[add]=0.5; w['oi_raw']=max(0,w['oi_raw']-0.25); seeds.append(w)
    w=dict(base); w['oi_wave_lowpos144']=0.5; w['oi_vol_alt_ratio72']=0.5; w['oi_raw']=0.5; seeds.append(w)
    w=dict(base); w['oi_wave_lowpos144']=0.5; w['oi_vol_volmom288']=0.5; w['oi_raw']=0.5; seeds.append(w)
    w=dict(base); w['oi_vol_alt_ratio72']=0.75; w['rex_wave_pricez288_low']=0.5; w['oi_raw']=0.5; seeds.append(w)
    vary=['nonpb30_taker','oi_raw','rex_rule','oi_wave_lowpos144','oi_wave_slope288_low','rex_wave_pricez288_low','rex_wave_vol144_high','oi_vol_alt_ratio72','oi_vol_volmom288','oi_vol_alt_ratio288','oi_upbit_ratio288_low']
    out=[]; seen=set()
    for c in seeds:
        candidates=[dict(c)]
        for k in vary:
            for d in [-0.5,-0.25,0.25,0.5]:
                w=dict(c); w[k]=max(0,w.get(k,0)+d); candidates.append(w)
        for k1,k2 in itertools.combinations(vary,2):
            for d1,d2 in [(-0.25,0.25),(0.25,-0.25),(0.25,0.25),(-0.25,-0.25)]:
                w=dict(c); w[k1]=max(0,w.get(k1,0)+d1); w[k2]=max(0,w.get(k2,0)+d2); candidates.append(w)
        for w in candidates:
            for ss in SLEEVES: w.setdefault(ss,0)
            gross=sum(w.values()); key=tuple(w[ss] for ss in SLEEVES)
            if 0<gross<=6 and key not in seen:
                seen.add(key); out.append(w)
    return out

def qbucket(x, qs):
    if not np.isfinite(x): return 'miss'
    return 'q0' if x<=qs[0] else 'q1' if x<=qs[1] else 'q2' if x<=qs[2] else 'q3' if x<=qs[3] else 'q4'

def token_builder(feat,masks):
    cols=['w_pos_144','w_pos_288','w_retr_144','w_retr_288','w_eff_144','w_eff_288','w_price_z_144','w_price_z_288','w_slope_atr_144','w_slope_atr_288','w_vol_z_144','wr_vwap_dev_z','wr_flow_mom','wr_vol_spike','vg_alt_btc_qv_ratio_z_72','vg_alt_btc_qv_ratio_z_288','vg_vol_mom_288','vg_upbit_binance_vol_ratio_z_288']
    qmap={}
    tr=masks['train']
    for c in cols:
        vals=feat.loc[tr,c].to_numpy(float); vals=vals[np.isfinite(vals)]
        if len(vals)<100 or np.nanstd(vals)<1e-12: continue
        qmap[c]=np.quantile(vals,[.2,.4,.6,.8]).tolist()
    def toks(e):
        i=int(e['signal_pos']); d={'sleeve':e['sleeve'],'side':e['side']}
        for c,qs in qmap.items(): d[c]=qbucket(float(feat[c].iloc[i]),qs)
        # compact composite states
        d['wave_zone']=d.get('w_pos_288','miss')+'|'+d.get('w_eff_288','miss')
        d['wave_flow']=d.get('wr_flow_mom','miss')+'|'+d.get('w_vol_z_144','miss')
        d['vol_flow']=d.get('vg_alt_btc_qv_ratio_z_72','miss')+'|'+d.get('vg_vol_mom_288','miss')
        d['wave_slope']=d.get('w_slope_atr_288','miss')+'|'+d.get('w_price_z_288','miss')
        return d
    return toks, qmap

def context_id(toks,keys): return '|'.join(f'{k}={toks.get(k,"miss")}' for k in keys)

def fit_block(events,tokens,keys,min_n,bad_mean,bad_win):
    g=defaultdict(list)
    for ei,e in enumerate(events):
        if e['split']=='train': g[context_id(tokens[ei],keys)].append(float(e['ret_bps']))
    blocked={}
    for cid,vals in g.items():
        if len(vals)>=min_n:
            arr=np.array(vals,float); mean=float(arr.mean()); win=float((arr>0).mean())
            if mean<=bad_mean or win<=bad_win: blocked[cid]={'train_n':len(vals),'train_mean_ret_bps':mean,'train_win_rate':win}
    return blocked

def allow_for_block(events,tokens,keys,blocked):
    return {i:(context_id(tokens[i],keys) not in blocked) for i in range(len(events))}

def score(st):
    o=['test2024','eval2025','ytd2026']; maxm=max(st[x]['strict_mdd_pct'] for x in o); minr=min(st[x]['cagr_to_strict_mdd'] for x in o); ret=sum(st[x]['total_return_pct'] for x in o)
    return (maxm<=25, minr>=5, ret, minr, -maxm)

def main():
    market,feat,masks,years,events,alpha_thresholds=build_events()
    split_lens=sparsify_events(events,masks)
    base_by=arrays(events,masks,split_lens=split_lens)
    weights=weight_grid()
    base_rows=[]
    for w in weights:
        st=metrics(base_by,years,w); base_rows.append({'weights':w,'gross':sum(w.values()),'stats':st,'score_tuple':score(st)})
    base_rows.sort(key=lambda r:r['score_tuple'],reverse=True)
    toks_fn,qmap=token_builder(feat,masks); tokens=[toks_fn(e) for e in events]
    keysets=[
        ('sleeve','wave_zone'),('sleeve','wave_flow'),('sleeve','wave_slope'),
        ('sleeve','w_pos_288','w_eff_288'),('sleeve','w_retr_288','w_slope_atr_288'),
        ('sleeve','w_price_z_288','wr_flow_mom'),('sleeve','wr_vwap_dev_z','wr_vol_spike'),
        ('side','wave_zone','wave_flow'),('side','wave_slope','wr_vwap_dev_z'),
        ('side','vol_flow'),('sleeve','vol_flow'),('side','wave_zone','vol_flow'),
    ]
    trials=[]
    # apply selector to top 300 baseline combos for cost control; selector fit is train-only independent of weights.
    topw=[r['weights'] for r in base_rows[:40]]
    for keys in keysets:
        for min_n in [12,24]:
            for bad_mean in [-8,-20]:
                for bad_win in [.38,.42]:
                    blocked=fit_block(events,tokens,keys,min_n,bad_mean,bad_win)
                    if not blocked: continue
                    allow=allow_for_block(events,tokens,keys,blocked)
                    by=arrays(events,masks,allow,split_lens=split_lens)
                    local=[]
                    for w in topw:
                        st=metrics(by,years,w); local.append({'weights':w,'gross':sum(w.values()),'stats':st,'score_tuple':score(st)})
                    local.sort(key=lambda r:r['score_tuple'],reverse=True)
                    best=local[0]
                    trials.append({'context_keys':keys,'params':{'min_train_context_events':min_n,'bad_mean_ret_bps':bad_mean,'bad_win_rate':bad_win},'blocked_contexts':len(blocked),'blocked_preview':list(blocked.items())[:12],'best':{k:v for k,v in best.items() if k!='score_tuple'}})
    trials.sort(key=lambda r:score(r['best']['stats']),reverse=True)
    # single hard wave gate diagnostics by sleeve
    wave_cols=['w_pos_144','w_pos_288','w_retr_144','w_retr_288','w_eff_144','w_eff_288','w_price_z_144','w_price_z_288','w_slope_atr_144','w_slope_atr_288','w_vol_z_144','wr_vwap_dev_z','wr_flow_mom','wr_vol_spike','vg_alt_btc_qv_ratio_z_72','vg_alt_btc_qv_ratio_z_288','vg_vol_mom_288','vg_upbit_binance_vol_ratio_z_288']
    train=masks['train']; hard=[]
    # concise: top existing sleeves only, q20/q80 gates
    for sl in SLEEVES:
        ev_idx=[i for i,e in enumerate(events) if e['sleeve']==sl]
        if not ev_idx: continue
        for c in wave_cols:
            vals=feat.loc[train,c].to_numpy(float); vals=vals[np.isfinite(vals)]
            if len(vals)<100 or np.nanstd(vals)<1e-12: continue
            for q,op in [(0.2,'<='),(0.8,'>=')]:
                thr=float(np.quantile(vals,q))
                allow={i:True for i in range(len(events))}
                for i in ev_idx:
                    x=float(feat[c].iloc[int(events[i]['signal_pos'])]); allow[i]=np.isfinite(x) and ((x<=thr) if op=='<=' else (x>=thr))
                by=arrays(events,masks,allow,split_lens=split_lens)
                w={s:(1.0 if s==sl else 0.0) for s in SLEEVES}
                st=metrics(by,years,w)
                if st['test2024']['trade_entries']>=5 and st['eval2025']['trade_entries']>=5:
                    hard.append({'sleeve':sl,'gate':{'feature':c,'op':op,'threshold':thr,'train_q':q},'stats':st,'rank_tuple':(st['test2024']['cagr_pct']>0 and st['eval2025']['cagr_pct']>0, min(st['test2024']['cagr_to_strict_mdd'],st['eval2025']['cagr_to_strict_mdd']), st['ytd2026']['cagr_to_strict_mdd'], st['test2024']['trade_entries']+st['eval2025']['trade_entries'])})
    hard.sort(key=lambda r:r['rank_tuple'],reverse=True)
    report={'as_of':datetime.now(timezone.utc).isoformat(),'protocol':'All known alpha sleeves combined with wave-feature train-only hard gates and portfolio BLOCK_RISK selector. Splits: train<2024, test=2024, eval=2025, ytd2026 through local data. Fees+slippage 5bp/side via existing event simulation; strict in-position MDD. Selector/gate thresholds fit on train only; selection ranking is diagnostic and uses test/eval/ytd report columns, not promoted live config.','sleeves':SLEEVES,'alpha_thresholds':alpha_thresholds,'event_counts':{sp:dict(Counter(e['sleeve'] for e in events if e['split']==sp)) for sp in ep.SPLITS},'wave_quantiles_train':qmap,'baseline_top':[{k:v for k,v in r.items() if k!='score_tuple'} for r in base_rows[:20]],'selector_trials_top':trials[:50],'single_wave_gate_top':[{k:v for k,v in r.items() if k!='rank_tuple'} for r in hard[:80]]}
    Path(OUT).write_text(json.dumps(report,indent=2,ensure_ascii=False))
    best=report['selector_trials_top'][0] if report['selector_trials_top'] else None
    md=['# All alpha + volume/wave portfolio combo scan (2026-07-08)','',report['protocol'],'','## Event counts','```json',json.dumps(report['event_counts'],indent=2,ensure_ascii=False),'```','']
    def fmt(name, row):
        md.append(f'## {name}')
        md.append(f"weights: `{row['weights']}` gross={row['gross']}")
        md.append('| split | return | CAGR | strict MDD | CAGR/MDD | trades | win | sharpe-like |')
        md.append('|---|---:|---:|---:|---:|---:|---:|---:|')
        for sp in ['train','test2024','eval2025','ytd2026']:
            s=row['stats'][sp]
            md.append(f"| {sp} | {s['total_return_pct']:.2f}% | {s['cagr_pct']:.2f}% | {s['strict_mdd_pct']:.2f}% | {s['cagr_to_strict_mdd']:.2f} | {s['trade_entries']} | {s['win_rate']*100:.1f}% | {s['bar_sharpe_like']:.2f} |")
        md.append('')
    fmt('Baseline best weight combo', report['baseline_top'][0])
    if best:
        md.append('## Best wave selector')
        md.append(f"context_keys: `{best['context_keys']}` params: `{best['params']}` blocked={best['blocked_contexts']}")
        fmt('Best selector combo', best['best'])
    md.append('## Top single wave gates')
    md.append('| sleeve | gate | 2024 ratio/trades | 2025 ratio/trades | 2026 ratio/trades |')
    md.append('|---|---|---:|---:|---:|')
    for r in report['single_wave_gate_top'][:15]:
        g=r['gate']; st=r['stats']; gate_s=f"{g['feature']} {g['op']} {g['threshold']:.6g} (q{g['train_q']})"
        md.append(f"| {r['sleeve']} | `{gate_s}` | {st['test2024']['cagr_to_strict_mdd']:.2f}/{st['test2024']['trade_entries']} | {st['eval2025']['cagr_to_strict_mdd']:.2f}/{st['eval2025']['trade_entries']} | {st['ytd2026']['cagr_to_strict_mdd']:.2f}/{st['ytd2026']['trade_entries']} |")
    md.append('')
    Path(DOC).write_text('\n'.join(md))
    print(json.dumps({'output':OUT,'doc':DOC,'event_counts':report['event_counts'],'baseline_best':report['baseline_top'][0],'best_selector':best,'top_single_gates':report['single_wave_gate_top'][:5]},indent=2,ensure_ascii=False))
if __name__=='__main__': main()
