import json, itertools, math
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd
import training.evaluate_volume_wave_portfolio_combo as vw
from training.evaluate_volume_wave_portfolio_combo import SLEEVES as BASE_SLEEVES

OUT='results/portfolio_with_dynamic_exit_sleeves_2026-07-08.json'
DOC='docs/portfolio-with-dynamic-exit-sleeves-2026-07-08.md'
BASE_RESULT='results/all_alpha_with_wave_alpha_sleeves_combo_2026-07-08.json'
SLEEVES=list(BASE_SLEEVES)+['rex_dyn_short_exit','oi_alt_ratio72_dyn_exit']
COST=.0005

def q(feat,mask,col,qq):
    vals=feat.loc[mask,col].to_numpy(float); vals=vals[np.isfinite(vals)]
    if len(vals)<100 or np.nanstd(vals)<1e-12: return 0.0
    return float(np.quantile(vals,qq))

def sim_dyn_events(m, masks, years, entries, exit_active, min_bars, side):
    op=m.open.to_numpy(float); hi=m.high.to_numpy(float); lo=m.low.to_numpy(float); events=[]
    for sp,sm in masks.items():
        idx=np.flatnonzero(sm); start=idx[0]; end=idx[-1]; next_allowed=0
        for p in sorted([int(x) for x in entries.get(sp,[])]):
            if p<next_allowed: continue
            ep=p+1
            if ep>=end: continue
            r=np.zeros(len(m)); adv=np.zeros(len(m)); fac=1-COST; minfac=fac; xp=end
            for j in range(ep,end):
                oj=op[j]
                if not np.isfinite(oj) or oj<=0: continue
                adverse=(lo[j]-oj)/oj if side=='long' else (oj-hi[j])/oj
                rr=(op[j+1]-oj)/oj if side=='long' else (oj-op[j+1])/oj
                adv[j]+=adverse; r[j+1]+=rr; fac*=max(0,1+rr); minfac=min(minfac,fac*max(0,1+adverse))
                if j-ep+1>=min_bars and exit_active[j]: xp=j+1; break
                if fac<=0: xp=j+1; break
            r[ep]-=COST; r[xp]-=COST
            events.append({'split':sp,'sleeve':'','side':side,'signal_pos':p,'ret_bps':(fac*(1-COST)-1)*10000,'ret':r,'adv':adv})
            next_allowed=xp+1
    return events

def build_dynamic_sleeves(m, feat, masks, years):
    train=masks['train']; out=[]
    # 1 rex short + wave_lower_or_cvd_bull exit, min 48
    # Need dynamic-exit features. evaluate_volume_wave has wave/volume features but not dx_cvd_bull; add minimal if missing.
    c=m.close.astype(float); qv=m.quote_asset_volume.astype(float); tbq=m.taker_buy_quote.astype(float)
    def z(s,n):
        mu=s.rolling(n,min_periods=max(20,n//4)).mean(); sd=s.rolling(n,min_periods=max(20,n//4)).std(ddof=0)
        return ((s-mu)/sd.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).clip(-8,8).fillna(0)
    signed=(2*tbq-qv).fillna(0); cvd=signed.cumsum()
    taker_cvd_z72=z((cvd.abs()+1)/(cvd.abs()+1).shift(72).replace(0,np.nan)-1,288).fillna(0)*np.sign(cvd.diff(72).fillna(0))
    ret72=(c/c.shift(72).replace(0,np.nan)-1).replace([np.inf,-np.inf],np.nan).fillna(0)
    cvd_bull=taker_cvd_z72.clip(lower=0)+(-z(ret72,288)).clip(lower=0)
    wave_lower=feat['w_pos_144'].to_numpy(float)<=q(feat,train,'w_pos_144',.2) if 'w_pos_144' in feat else feat['dx_wave_pos_144'].to_numpy(float)<=q(feat,train,'dx_wave_pos_144',.2)
    exit_rex_short=wave_lower | (cvd_bull.to_numpy(float)>=float(np.quantile(cvd_bull[train],.85)))
    rows=[]
    for fp in ['results/rex_dual_regime_train_2021_2023_predictions_2026-07-03.jsonl','results/rex_dual_regime_test_2024_predictions_2026-07-03.jsonl','results/rex_dual_regime_eval_2025_2026h1_predictions_2026-07-03.jsonl']:
        if Path(fp).exists(): rows += [json.loads(l) for l in Path(fp).read_text().splitlines() if l.strip()]
    entries={sp:[] for sp in masks}
    for row in rows:
        pred=row.get('prediction') or {}
        if pred.get('gate')!='TRADE' or pred.get('side')!='SHORT': continue
        p=int(row['signal_pos'])
        for sp,sm in masks.items():
            if 0<=p<len(sm) and sm[p]: entries[sp].append(p)
    ev=sim_dyn_events(m,masks,years,entries,exit_rex_short,48,'short')
    for e in ev: e['sleeve']='rex_dyn_short_exit'
    out.extend(ev)
    # 2 oi_alt_ratio72 long + vwap_overheat exit, min 48
    oi=json.load(open('configs/live/oi_divergence_sma24_highfreq_h30_s6_candidate.json'))
    oi_active=vw.gate(feat,oi['gates'])
    alt_col='vg_alt_btc_qv_ratio_z_72' if 'vg_alt_btc_qv_ratio_z_72' in feat else 'dx_alt_ratio_z72'
    alt_thr=q(feat,train,alt_col,.8)
    active=oi_active & (feat[alt_col].to_numpy(float)>=alt_thr)
    vwap_col='wr_vwap_dev_z'
    exit_oi=feat[vwap_col].to_numpy(float)>=q(feat,train,vwap_col,.85)
    entries={sp:[] for sp in masks}
    stride=int(oi['stride_bars'])
    n=len(m)
    for sp,sm in masks.items():
        idx=np.flatnonzero(active & sm)
        idx=idx[(idx>=143)&(idx<n-3)&((idx%stride)==0)]
        entries[sp]=idx.tolist()
    ev=sim_dyn_events(m,masks,years,entries,exit_oi,48,'long')
    for e in ev: e['sleeve']='oi_alt_ratio72_dyn_exit'
    out.extend(ev)
    return out

def arrays(events,masks):
    n=len(next(iter(masks.values()))); by={}
    starts={sp:int(np.flatnonzero(m)[0]) for sp,m in masks.items()}; ends={sp:int(np.flatnonzero(m)[-1])+1 for sp,m in masks.items()}
    for sp in masks:
        ln=ends[sp]-starts[sp]; matsR=[]; matsA=[]; counts=[]; wins=[]
        for sl in SLEEVES:
            r=np.zeros(ln); a=np.zeros(ln); c=0; w=0
            for e in events:
                if e['split']==sp and e['sleeve']==sl:
                    st=starts[sp]; en=ends[sp]
                    r += e['ret'][st:en]; a += e['adv'][st:en]; c+=1; w += e['ret_bps']>0
            matsR.append(r); matsA.append(a); counts.append(c); wins.append(w)
        R=np.vstack(matsR); A=np.vstack(matsA); active=np.any((R!=0)|(A!=0),axis=0)
        by[sp]={'R':R[:,active],'A':A[:,active],'counts':np.array(counts),'wins':np.array(wins),'active_bars':int(active.sum())}
    return by

def metric(d,years,w):
    wv=np.array([w.get(s,0.0) for s in SLEEVES]); r=wv@d['R']; adv=wv@d['A']
    fac=np.maximum(0,1+r); eqp=np.cumprod(fac) if len(fac) else np.array([1.0]); eqb=np.r_[1.0,eqp[:-1]] if len(fac) else np.array([1.0])
    pka=np.maximum.accumulate(eqp); pkb=np.maximum.accumulate(eqb)
    mdd=max(float(np.nanmax(1-eqp/np.maximum(pka,1e-12))) if len(eqp) else 0,float(np.nanmax(1-(eqb*(1+adv))/np.maximum(pkb,1e-12))) if len(eqb) else 0)*100
    eq=float(eqp[-1]) if len(eqp) else 1.0; cagr=(eq**(1/years)-1)*100 if eq>0 else -100
    trades=int(np.sum(d['counts'][wv!=0])); wins=int(np.sum(d['wins'][wv!=0])); vals=r[np.abs(r)>1e-12]
    sh=float(vals.mean()/vals.std(ddof=1)*np.sqrt(len(vals))) if len(vals)>1 and vals.std(ddof=1)>0 else 0
    return {'total_return_pct':(eq-1)*100,'cagr_pct':cagr,'strict_mdd_pct':mdd,'cagr_to_strict_mdd':cagr/mdd if mdd>1e-12 else 0,'trade_entries':trades,'win_rate':wins/trades if trades else 0,'active_bars':d['active_bars'],'bar_sharpe_like':sh,'sleeve_trade_counts':{s:int(c) if w.get(s,0)!=0 else 0 for s,c in zip(SLEEVES,d['counts'])}}

def metrics(by,years,w): return {sp:metric(by[sp],years[sp],w) for sp in vw.ep.SPLITS}

def score(st):
    o=['test2024','eval2025','ytd2026']; maxm=max(st[x]['strict_mdd_pct'] for x in o); minr=min(st[x]['cagr_to_strict_mdd'] for x in o); ret=sum(st[x]['total_return_pct'] for x in o)
    return (maxm<=25, minr>=5, minr, ret, -maxm)

def grid():
    seeds=[]
    base={'nonpb30_taker':2.0,'oi_raw':1.0,'rex_rule':2.5,'oi_wave_lowpos144':0.5}
    vol={'nonpb30_taker':2.0,'oi_raw':0.75,'rex_rule':2.5,'oi_upbit_ratio288_low':0.75}
    seeds += [base, vol]
    for add in ['rex_dyn_short_exit','oi_alt_ratio72_dyn_exit']:
        w=dict(base); w[add]=0.5; w['rex_rule']=max(0,w.get('rex_rule',0)-0.25); seeds.append(w)
        w=dict(base); w[add]=1.0; w['oi_raw']=max(0,w.get('oi_raw',0)-0.5); seeds.append(w)
    w=dict(base); w['rex_dyn_short_exit']=0.75; w['oi_alt_ratio72_dyn_exit']=0.5; w['rex_rule']=2.0; w['oi_raw']=0.75; seeds.append(w)
    vary=['nonpb30_taker','oi_raw','rex_rule','oi_wave_lowpos144','oi_upbit_ratio288_low','rex_dyn_short_exit','oi_alt_ratio72_dyn_exit']
    out=[]; seen=set()
    for c in seeds:
        cc={s:0 for s in SLEEVES}; cc.update(c)
        cand=[cc]
        for k in vary:
            for d in [-0.5,-0.25,0.25,0.5]:
                w=dict(cc); w[k]=max(0,w.get(k,0)+d); cand.append(w)
        for k1,k2 in itertools.combinations(vary,2):
            for d1,d2 in [(-.25,.25),(.25,-.25),(.25,.25),(-.25,-.25)]:
                w=dict(cc); w[k1]=max(0,w.get(k1,0)+d1); w[k2]=max(0,w.get(k2,0)+d2); cand.append(w)
        for w in cand:
            gross=sum(w.values()); key=tuple(w[s] for s in SLEEVES)
            if 0<gross<=6 and key not in seen:
                seen.add(key); out.append(w)
    return out

def main():
    m,feat,masks,years,events,thr=vw.build_events(); dyn=build_dynamic_sleeves(m,feat,masks,years); all_events=events+dyn; by=arrays(all_events,masks)
    rows=[]
    for w in grid():
        st=metrics(by,years,w); rows.append({'weights':w,'gross':sum(w.values()),'stats':st,'score_tuple':score(st)})
    rows.sort(key=lambda r:r['score_tuple'],reverse=True)
    # standalone dyn
    standalone={sl:metrics(by,years,{s:(1.0 if s==sl else 0.0) for s in SLEEVES}) for sl in ['rex_dyn_short_exit','oi_alt_ratio72_dyn_exit']}
    out={'protocol':'Portfolio weight scan adding dynamic-exit sleeves to previous volume/wave alpha set. Gross<=6; splits train/test2024/eval2025/ytd2026; strict in-position MDD; 5bp per side inherited from event simulations.','dynamic_sleeves':['rex_dyn_short_exit','oi_alt_ratio72_dyn_exit'],'event_counts':{sp:dict(Counter(e['sleeve'] for e in all_events if e['split']==sp)) for sp in vw.ep.SPLITS},'standalone_dynamic':standalone,'top':[{k:v for k,v in r.items() if k!='score_tuple'} for r in rows[:80]]}
    Path(OUT).write_text(json.dumps(out,indent=2,ensure_ascii=False))
    md=['# Portfolio with dynamic-exit sleeves (2026-07-08)','',out['protocol'],'','## Dynamic standalone','']
    for sl,st in standalone.items():
        md.append(f'### {sl}'); md.append('| split | ret | CAGR | MDD | ratio | trades | win |'); md.append('|---|---:|---:|---:|---:|---:|---:|')
        for sp in ['train','test2024','eval2025','ytd2026']:
            s=st[sp]; md.append(f"| {sp} | {s['total_return_pct']:.2f}% | {s['cagr_pct']:.2f}% | {s['strict_mdd_pct']:.2f}% | {s['cagr_to_strict_mdd']:.2f} | {s['trade_entries']} | {s['win_rate']*100:.1f}% |")
        md.append('')
    md.append('## Top portfolios'); md.append('| rank | gross | weights | 2024 ret/CAGR/MDD/ratio | 2025 ret/CAGR/MDD/ratio | 2026 ret/CAGR/MDD/ratio |'); md.append('|---:|---:|---|---:|---:|---:|')
    for i,r in enumerate(out['top'][:20],1):
        def fmt(s): return f"{s['total_return_pct']:.2f}/{s['cagr_pct']:.2f}/{s['strict_mdd_pct']:.2f}/{s['cagr_to_strict_mdd']:.2f}"
        w={k:v for k,v in r['weights'].items() if v}
        st=r['stats']; md.append(f"| {i} | {r['gross']:.2f} | `{w}` | {fmt(st['test2024'])} | {fmt(st['eval2025'])} | {fmt(st['ytd2026'])} |")
    Path(DOC).write_text('\n'.join(md)+'\n')
    print(json.dumps({'output':OUT,'doc':DOC,'standalone':standalone,'top':out['top'][:10]},indent=2,ensure_ascii=False))
if __name__=='__main__': main()
