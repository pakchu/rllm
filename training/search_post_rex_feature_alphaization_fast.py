"""Fast post-REX feature alphaization coarse scan.

Uses successful post-REX process in a bounded way:
- existing setup entries only
- VPIN/Alpha101/wave/volume as train-fitted allow gates
- fixed 72/144-bar exits plus a small dynamic-exit subset
- BTCUSDT target, 6bp/side, strict in-position MDD, split-end forced close
"""
from __future__ import annotations
import json, math
from collections import Counter
from pathlib import Path
from typing import Any
import numpy as np, pandas as pd
from training.evaluate_portfolio_llm_selector import _prep, SPLITS
from training.search_dynamic_exit_all_candidates import add_features as add_context_features, build_entries, q
from training.search_vpin_formulaic_alpha import add_vpin_formulaic_features, trade_arrays, stats
from training.search_alpha101_derivative_alphas import add_features as add_alpha101_features

OUT='results/post_rex_feature_alphaization_fast_2026-07-10.json'
DOC='docs/post-rex-feature-alphaization-fast-2026-07-10.md'
COST=0.0006

def arr(f,name): return f.get(name,pd.Series(0.0,index=f.index)).to_numpy(float)
def qv(f,train,col,qq): return q(f,train,col,qq)
def active_terms(f,terms):
    a=np.ones(len(f),bool)
    for col,op,thr in terms:
        x=arr(f,col); a &= np.isfinite(x)&((x>=thr) if op=='>=' else (x<=thr))
    return a

def add_all(m,base):
    f=add_context_features(m,base)
    f=add_vpin_formulaic_features(m,f)
    a=add_alpha101_features(m).add_prefix('a101ctx__')
    f=pd.concat([f,a],axis=1)
    f['ctx_vpin_sell_toxic']=f.vp_vpin_z_144.clip(lower=0)+(-f.vp_imb_z_72).clip(lower=0)
    f['ctx_vpin_buy_toxic']=f.vp_vpin_z_144.clip(lower=0)+f.vp_imb_z_72.clip(lower=0)
    f['ctx_a101_pullback_quality']=(-f['a101ctx__a_ret_z_72']).clip(lower=0)+(1-f['a101ctx__a_pos_288']).clip(lower=0)+f['a101ctx__a_range_compress_288'].clip(lower=0)
    f['ctx_a101_overheat_quality']=f['a101ctx__a_ret_z_72'].clip(lower=0)+f['a101ctx__a_pos_288'].clip(lower=0)+f['a101ctx__a_range_expand_72'].clip(lower=0)
    f['ctx_a101_vwap_washout']=(-f['a101ctx__a_vwap_gap_z']).clip(lower=0)+(-f['a101ctx__a_clv_rank_288']+0.5).clip(lower=0)
    f['ctx_a101_vwap_overheat']=f['a101ctx__a_vwap_gap_z'].clip(lower=0)+(f['a101ctx__a_clv_rank_288']-0.5).clip(lower=0)
    f['ctx_volume_participation']=f.dx_alt_ratio_z72.clip(lower=0)+f.dx_alt_breadth72.clip(lower=0)+f.dx_rvol_z_144.clip(lower=0)
    f['ctx_flow_absorption_long']=f.dx_taker_div_72.clip(lower=0)+(-f.dx_ret_72).clip(lower=0)
    f['ctx_flow_exhaust_short']=(-f.dx_taker_div_72).clip(lower=0)+f.dx_ret_72.clip(lower=0)
    return f.loc[:,~f.columns.duplicated(keep='last')].replace([np.inf,-np.inf],np.nan).fillna(0)

def gates(f,train,side):
    specs=[('none',[])]
    if side=='long':
        specs += [
          ('a101_pullback_quality',[('ctx_a101_pullback_quality','>=',.70)]),
          ('a101_vwap_washout',[('ctx_a101_vwap_washout','>=',.70)]),
          ('vpin_lowtox',[('vp_vpin_z_144','<=',.60)]),
          ('vpin_sell_toxic_reversal',[('ctx_vpin_sell_toxic','>=',.80)]),
          ('volume_participation',[('ctx_volume_participation','>=',.70)]),
          ('alt_ratio_high',[('dx_alt_ratio_z72','>=',.80)]),
          ('flow_absorption',[('ctx_flow_absorption_long','>=',.70)]),
          ('wave_lowpos',[('dx_wave_pos_144','<=',.35)]),
          ('a101_pullback_x_vpin_lowtox',[('ctx_a101_pullback_quality','>=',.70),('vp_vpin_z_144','<=',.60)]),
          ('volume_x_flow',[('ctx_volume_participation','>=',.70),('ctx_flow_absorption_long','>=',.70)]),
        ]
    else:
        specs += [
          ('a101_overheat_quality',[('ctx_a101_overheat_quality','>=',.70)]),
          ('a101_vwap_overheat',[('ctx_a101_vwap_overheat','>=',.70)]),
          ('vpin_lowtox',[('vp_vpin_z_144','<=',.60)]),
          ('vpin_buy_toxic_reversal',[('ctx_vpin_buy_toxic','>=',.80)]),
          ('volume_participation',[('ctx_volume_participation','>=',.70)]),
          ('alt_ratio_low',[('dx_alt_ratio_z72','<=',.20)]),
          ('flow_exhaust',[('ctx_flow_exhaust_short','>=',.70)]),
          ('wave_highpos',[('dx_wave_pos_144','>=',.65)]),
        ]
    out=[]
    for name,raw in specs:
        terms=[]
        ok=True
        for c,o,qq in raw:
            if c not in f: ok=False; break
            terms.append((c,o,qv(f,train,c,qq),qq))
        if ok: out.append((name,terms))
    return out

def eval_fixed(m,masks,years,entries,side,hold):
    fac,mn,rr=trade_arrays(m,hold,side)
    out={}
    for sp,mask in masks.items():
        loc=[]; nxt=0
        for e in sorted([x for x in entries if x['split']==sp], key=lambda x:x['pos']):
            p=int(e['pos']); xp=p+1+hold
            if p<nxt or xp>=len(m) or not mask[xp] or not np.isfinite(fac[p]): continue
            loc.append((float(fac[p]),float(mn[p]),float(rr[p]))); nxt=xp
        out[sp]=stats(loc,years[sp])
    return out

def eval_dyn(m,masks,years,entries,side,exit_active,min_bars=12,max_bars=288):
    op=m.open.to_numpy(float); hi=m.high.to_numpy(float); lo=m.low.to_numpy(float); out={}
    for sp,sm in masks.items():
        idx=np.flatnonzero(sm); end=int(idx[-1]); eq=peak=1.0; mdd=0.0; rets=[]; wins=0; nxt=0
        for e in sorted([x for x in entries if x['split']==sp], key=lambda x:x['pos']):
            p=int(e['pos'])
            if p<nxt: continue
            ep=p+1
            if ep>=end: continue
            entry_eq=eq; eq*=1-COST; mdd=max(mdd,1-eq/max(peak,1e-12)); limit=min(end,ep+max_bars); xp=limit
            for j in range(ep,limit):
                oj=op[j]
                if not np.isfinite(oj) or oj<=0: continue
                adverse=(lo[j]-oj)/oj if side=='long' else (oj-hi[j])/oj
                rr=(op[j+1]-oj)/oj if side=='long' else (oj-op[j+1])/oj
                mdd=max(mdd,1-(eq*max(0,1+adverse))/max(peak,1e-12)); eq*=max(0,1+rr); peak=max(peak,eq)
                if j-ep+1>=min_bars and exit_active[j]: xp=j+1; break
                if eq<=0: xp=j+1; break
            eq*=1-COST; peak=max(peak,eq); mdd=max(mdd,1-eq/max(peak,1e-12)); r=eq/entry_eq-1; rets.append(r); wins+=r>0; nxt=xp+1
            if eq<=0: break
        cagr=(eq**(1/years[sp])-1)*100 if eq>0 else -100; md=mdd*100; a=np.array(rets,float)
        sh=float(a.mean()/a.std(ddof=1)*math.sqrt(len(a)/years[sp])) if len(a)>1 and a.std(ddof=1)>0 else 0
        out[sp]=dict(total_return_pct=(eq-1)*100,cagr_pct=cagr,strict_mdd_pct=md,cagr_to_strict_mdd=cagr/md if md>1e-12 else 0,trade_entries=len(rets),win_rate=wins/len(rets) if rets else 0,bar_sharpe_like=sh,mean_trade_ret_pct=float(a.mean()*100) if len(a) else 0)
    return out

def exits(f,train,side):
    if side=='long':
        return [('vwap_overheat', arr(f,'wr_vwap_dev_z')>=qv(f,train,'wr_vwap_dev_z',.85)),('cvd_bear',arr(f,'dx_cvd_bear')>=qv(f,train,'dx_cvd_bear',.80)),('vpin_sell_toxic_exit',arr(f,'ctx_vpin_sell_toxic')>=qv(f,train,'ctx_vpin_sell_toxic',.85)),('a101_overheat_exit',arr(f,'ctx_a101_vwap_overheat')>=qv(f,train,'ctx_a101_vwap_overheat',.85))]
    return [('vwap_washout', arr(f,'wr_vwap_dev_z')<=qv(f,train,'wr_vwap_dev_z',.15)),('cvd_bull',arr(f,'dx_cvd_bull')>=qv(f,train,'dx_cvd_bull',.80)),('vpin_buy_toxic_exit',arr(f,'ctx_vpin_buy_toxic')>=qv(f,train,'ctx_vpin_buy_toxic',.85)),('a101_washout_exit',arr(f,'ctx_a101_vwap_washout')>=qv(f,train,'ctx_a101_vwap_washout',.85))]

def score(st):
    t,e,y,tr=st['test2024'],st['eval2025'],st['ytd2026'],st['train']
    return (t['cagr_pct']>0 and e['cagr_pct']>0 and y['cagr_pct']>0 and t['trade_entries']>=15 and e['trade_entries']>=10 and tr['trade_entries']>=30, min(t['cagr_to_strict_mdd'],e['cagr_to_strict_mdd']), y['cagr_to_strict_mdd'], t['total_return_pct']+e['total_return_pct']+.5*y['total_return_pct'], t['trade_entries']+e['trade_entries'])

def main():
    m,base,masks,years=_prep(); f=add_all(m,base); train=masks['train']; entries=build_entries(m,f,masks)
    keep={'nonpb30_taker','oi_raw','oi_alt_ratio72','oi_wave_lowpos144','oi_upbit_like_altlow','rex_rule','cvd_flow_cont_long','cvd_bear_div_long','cvd_bull_div_long','alt_rotation_long','alt_rotation_short','kimchi_dxy_short'}
    rows=[]
    for name in sorted(keep):
        for side in ['long','short']:
            raw=[e for e in entries if e['name']==name and e['side']==side]
            if len(raw)<20: continue
            for gname,gterms in gates(f,train,side):
                ga=active_terms(f,[(c,o,thr) for c,o,thr,_ in gterms]) if gterms else np.ones(len(f),bool)
                gated=[e for e in raw if ga[int(e['pos'])]]; cnt=Counter(e['split'] for e in gated)
                if gname!='none' and (cnt.get('test2024',0)<10 or cnt.get('eval2025',0)<8): continue
                for hold in [72,144]:
                    st=eval_fixed(m,masks,years,gated,side,hold)
                    rows.append(dict(entry=name,side=side,gate=gname,exit=f'fixed_{hold}',hold=hold,min_bars=None,max_bars=hold,gate_terms=[dict(feature=c,op=o,threshold=thr,train_q=qq) for c,o,thr,qq in gterms],entry_counts=dict(cnt),stats=st,score_tuple=score(st)))
                # Dynamic only for strongest post-REX entry families or non-none gates.
                if name in {'nonpb30_taker','oi_alt_ratio72','oi_raw','rex_rule'} or gname!='none':
                    for exn,exa in exits(f,train,side):
                        st=eval_dyn(m,masks,years,gated,side,exa,12,288)
                        rows.append(dict(entry=name,side=side,gate=gname,exit=exn,hold=None,min_bars=12,max_bars=288,gate_terms=[dict(feature=c,op=o,threshold=thr,train_q=qq) for c,o,thr,qq in gterms],entry_counts=dict(cnt),stats=st,score_tuple=score(st)))
    rows.sort(key=lambda r:r['score_tuple'], reverse=True)
    top=[{k:v for k,v in r.items() if k!='score_tuple'} for r in rows[:240]]
    out={'protocol':'Fast post-REX alphaization: existing setup entries + train-fitted VPIN/Alpha101/wave/volume gates + fixed/dynamic exits. BTC target, 6bp/side, strict MDD, forced split-end close. Discovery not clean OOS proof.','entry_counts_base':{sp:dict(Counter(e['name'] for e in entries if e['split']==sp)) for sp in SPLITS},'all_count':len(rows),'top':top}
    Path(OUT).write_text(json.dumps(out,indent=2,ensure_ascii=False))
    def fmt(s): return f"{s['total_return_pct']:.2f}/{s['cagr_pct']:.2f}/{s['strict_mdd_pct']:.2f}/{s['cagr_to_strict_mdd']:.2f}/{s['trade_entries']}/{s['win_rate']:.2f}/{s['bar_sharpe_like']:.2f}"
    md=['# Fast post-REX feature alphaization scan (2026-07-10)','',out['protocol'],'','VPIN/Alpha101/wave/volume are used as gates/exits on setup entries, not standalone signals.','','| rank | entry | side | gate | exit | train | 2024 | 2025 | 2026 | terms |','|---:|---|---|---|---|---:|---:|---:|---:|---|']
    for i,r in enumerate(top[:100],1):
        st=r['stats']; terms='; '.join(f"{t['feature']} {t['op']} q{t['train_q']:.2f}({t['threshold']:.4g})" for t in r['gate_terms']) or 'none'
        md.append(f"| {i} | {r['entry']} | {r['side']} | {r['gate']} | {r['exit']} | {fmt(st['train'])} | {fmt(st['test2024'])} | {fmt(st['eval2025'])} | {fmt(st['ytd2026'])} | `{terms}` |")
    Path(DOC).write_text('\n'.join(md)+'\n')
    print(json.dumps({'output':OUT,'doc':DOC,'all_count':len(rows),'top':top[:20]},indent=2,ensure_ascii=False))
if __name__=='__main__': main()
