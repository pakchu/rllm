"""Strict symbolic scan for live-style event candidate binary-edge rows."""
from __future__ import annotations

import argparse, json, math, re, tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from training.online_risk_overlay_backtest import OnlineRiskOverlayConfig, run_overlay

@dataclass(frozen=True)
class BinaryEdgeStrictScanConfig:
    inputs: str
    market_csv: str
    output: str
    train_start: str = "2022-01-01"
    train_end: str = "2024-12-31 23:59:59"
    test_start: str = "2025-01-01"
    test_end: str = "2025-12-31 23:59:59"
    eval_start: str = "2026-01-01"
    eval_end: str = "2026-06-01 00:00:00"
    min_support_train: int = 80
    min_support_test: int = 40
    min_test_trades: int = 20
    max_rules: int = 40
    top_k: int = 30
    max_rule_terms: int = 3
    max_state_features: int = 48
    leverage: float = 0.5
    max_hold_bars: int = 576
    entry_delay_bars: int = 1
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001

def _read_jsonl(path: str|Path) -> list[dict[str,Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

def _load(inputs: str) -> list[dict[str,Any]]:
    rows=[]
    for raw in str(inputs).split(','):
        p=raw.strip()
        if p: rows.extend(_read_jsonl(p))
    return sorted(rows,key=lambda r:(str(r.get('date','')),int(r.get('signal_pos',0) or 0)))

def _split(date: str,cfg: BinaryEdgeStrictScanConfig)->str:
    if cfg.train_start <= date <= cfg.train_end: return 'train'
    if cfg.test_start <= date <= cfg.test_end: return 'test'
    if cfg.eval_start <= date <= cfg.eval_end: return 'eval'
    return 'ignore'

def _clean(x: Any)->str:
    return re.sub(r'[^A-Za-z0-9_:+.-]+','_',str(x).strip())[:80]

def _num_bucket(v: float)->str:
    if not math.isfinite(v): return 'nan'
    av=abs(v)
    if av < 0.01: mag='flat'
    elif av < 0.25: mag='small'
    elif av < 1.0: mag='mid'
    else: mag='large'
    if v > 0: return 'pos_'+mag
    if v < 0: return 'neg_'+mag
    return 'zero'

def _strength_bucket(v: Any)->str:
    try: x=float(v)
    except Exception: return 'nan'
    if x < .25: return 'low'
    if x > .6: return 'high'
    return 'mid'

def _prompt_features(prompt: str)->set[str]:
    feats=set()
    for raw in str(prompt).splitlines():
        line=raw.strip()
        if not line.startswith('- ') or ':' not in line: continue
        k,v=line[2:].split(':',1)
        k=_clean(k); v=v.strip()
        if not k: continue
        try:
            x=float(v)
            feats.add(f'{k}={_num_bucket(x)}')
        except Exception:
            feats.add(f'{k}={_clean(v)}')
    return feats

def _examples(rows: list[dict[str,Any]], cfg: BinaryEdgeStrictScanConfig)->dict[str,list[dict[str,Any]]]:
    out=defaultdict(list); seen=set()
    for r in rows:
        date=str(r.get('date')); sp=int(r.get('signal_pos',-1) or -1); split=_split(date,cfg)
        if split=='ignore' or sp<0: continue
        cand=r.get('candidate') if isinstance(r.get('candidate'),dict) else {}
        side=str(r.get('side') or cand.get('side') or 'NONE').upper(); hold=int(r.get('hold_bars') or cand.get('hold_bars') or 0)
        fam=str(cand.get('family','unknown'))
        if side not in {'LONG','SHORT'} or hold<=0: continue
        key=(sp,side,hold,fam)
        if key in seen: continue
        seen.add(key)
        reward=r.get('reward_audit') if isinstance(r.get('reward_audit'),dict) else {}
        utility=float(reward.get('utility',reward.get('net_return_pct',0.0)) or 0.0)*100.0
        feats={f'id={fam}|h{hold}',f'family={fam}',f'side={side}',f'id_side={fam}|h{hold}|{side}',f'hold={hold}',f'strength={_strength_bucket(cand.get("strength",0.0))}'}
        feats |= _prompt_features(str(r.get('prompt','')))
        ex={'date':date,'signal_pos':sp,'id':f'{fam}|h{hold}','side':side,'hold_bars':hold,'ret_pct':utility,'features':feats,'prediction':{'gate':'TRADE','side':side,'hold_bars':min(hold,int(cfg.max_hold_bars))}}
        out[split].append(ex)
    return out

def _rules(train:list[dict[str,Any]],cfg:BinaryEdgeStrictScanConfig)->list[tuple[str,...]]:
    counts=Counter(); [counts.update(ex['features']) for ex in train]
    base=[f for f,n in counts.items() if n>=cfg.min_support_train]
    rules={(f,) for f in base}
    anchors=[f for f in base if f.startswith(('id=','family=','side=','id_side=','hold='))]
    states=[f for f in base if not f.startswith(('id=','family=','side=','id_side=','hold='))]
    states=sorted(states,key=lambda f:(-counts[f],f))[:cfg.max_state_features]
    for a in anchors:
        for b in states: rules.add(tuple(sorted((a,b))))
    if cfg.max_rule_terms>=2:
        for pair in combinations(states,2): rules.add(tuple(sorted(pair)))
    if cfg.max_rule_terms>=3:
        pairs=list(combinations(states,2))
        for a in anchors:
            for b,c in pairs: rules.add(tuple(sorted((a,b,c))))
    return sorted(rules)

def _match_preds(examples:list[dict[str,Any]],rule:tuple[str,...],action:str)->list[dict[str,Any]]:
    rs=set(rule); out=[]; seen=set()
    for ex in examples:
        if ex['signal_pos'] in seen or not rs.issubset(ex['features']): continue
        pred=dict(ex['prediction'])
        if action=='invert': pred['side']='SHORT' if pred.get('side')=='LONG' else 'LONG'
        pred['family']=f'binary_edge_symbolic_{action}'
        out.append({'date':ex['date'],'signal_pos':ex['signal_pos'],'prediction':pred,'position_scale':1.0,'score':1.0,'rule':list(rule),'action':action,'source_candidate_id':ex['id']})
        seen.add(ex['signal_pos'])
    out.sort(key=lambda r:(r['date'],r['signal_pos']))
    return out

def _prefilter(examples:list[dict[str,Any]],rule:tuple[str,...],action:str)->float:
    rs=set(rule); vals=[(1 if action=='follow' else -1)*float(ex.get('ret_pct',0.0) or 0.0) for ex in examples if rs.issubset(ex['features'])]
    if not vals: return -1e9
    arr=np.asarray(vals,float); mean=float(np.mean(arr)); std=float(np.std(arr,ddof=1)) if arr.size>1 else 0.0
    t=mean/(std/math.sqrt(float(arr.size))) if std>1e-12 else 0.0
    return mean*min(3.0,math.sqrt(float(arr.size))/12.0)+0.02*t

def _bt(rows:list[dict[str,Any]],cfg:BinaryEdgeStrictScanConfig,tmp:Path,name:str)->dict[str,Any]:
    if not rows: return {'sim':{'trade_entries':0,'cagr_pct':0.0,'strict_mdd_pct':0.0,'cagr_to_strict_mdd':0.0},'trade_stats':{'n_trades':0}}
    pred=tmp/f'{name}.jsonl'; pred.write_text('\n'.join(json.dumps(r,ensure_ascii=False,sort_keys=True) for r in rows)+'\n')
    return run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=str(pred),market_csv=cfg.market_csv,output=str(tmp/f'{name}_bt.json'),leverage=cfg.leverage,fee_rate=cfg.fee_rate,slippage_rate=cfg.slippage_rate,entry_delay_bars=cfg.entry_delay_bars,max_hold_bars=cfg.max_hold_bars))

def _score(bt:dict[str,Any],min_trades:int)->float:
    s=bt.get('sim',{}); t=bt.get('trade_stats',{}); trades=int(s.get('trade_entries',0) or 0)
    if trades<min_trades: return -1e9
    return float(s.get('cagr_to_strict_mdd',0) or 0)+0.03*float(s.get('cagr_pct',0) or 0)-0.02*max(0.0,float(s.get('strict_mdd_pct',0) or 0)-15)-0.25*float(t.get('p_value_mean_ret_approx',1) or 1)+min(1.0,trades/100.0)

def run(cfg:BinaryEdgeStrictScanConfig)->dict[str,Any]:
    splits=_examples(_load(cfg.inputs),cfg); rules=_rules(splits['train'],cfg)
    candidates=[]
    for rule in rules:
        rs=set(rule); tr=sum(1 for ex in splits['train'] if rs.issubset(ex['features'])); te=sum(1 for ex in splits['test'] if rs.issubset(ex['features']))
        if tr<cfg.min_support_train or te<cfg.min_support_test: continue
        for action in ('follow','invert'): candidates.append((rule,action,tr,te,_prefilter(splits['test'],rule,action)))
    candidates.sort(key=lambda x:(x[4],x[3],x[2]),reverse=True)
    prefiltered=len(candidates); unique=[]; seen=set()
    for cand in candidates:
        rule,action,*_=cand; sig=(action,tuple(r['signal_pos'] for r in _match_preds(splits['test'],rule,action)))
        if sig in seen: continue
        seen.add(sig); unique.append(cand)
        if len(unique)>=cfg.max_rules: break
    scanned=[]
    with tempfile.TemporaryDirectory(prefix='binary_edge_strict_scan_') as td:
        tmp=Path(td)
        for i,(rule,action,tr,te,pf) in enumerate(unique):
            train_bt=_bt(_match_preds(splits['train'],rule,action),cfg,tmp,f'r{i}_train')
            test_bt=_bt(_match_preds(splits['test'],rule,action),cfg,tmp,f'r{i}_test')
            eval_preds=_match_preds(splits['eval'],rule,action)
            eval_bt=_bt(eval_preds,cfg,tmp,f'r{i}_eval')
            scanned.append({'rule':list(rule),'action':action,'support':{'train_candidates':tr,'test_candidates':te,'eval_candidates':len(eval_preds)},'train':{'sim':train_bt['sim'],'trade_stats':train_bt['trade_stats']},'test':{'sim':test_bt['sim'],'trade_stats':test_bt['trade_stats']},'eval':{'sim':eval_bt['sim'],'trade_stats':eval_bt['trade_stats']},'prefilter_score':pf,'test_score':_score(test_bt,cfg.min_test_trades)})
    scanned.sort(key=lambda r:float(r['test_score']),reverse=True)
    report={'as_of':datetime.now(timezone.utc).isoformat(),'config':asdict(cfg),'split_candidate_examples':{k:len(v) for k,v in splits.items()},'generated_rule_count':len(rules),'prefiltered_candidate_count':prefiltered,'unique_strict_candidate_count':len(unique),'strict_scanned_count':len(scanned),'selection_protocol':'binary-edge live-style candidates; train support; test-only prefilter; strict test rank; eval untouched','top_by_test':scanned[:cfg.top_k]}
    Path(cfg.output).parent.mkdir(parents=True,exist_ok=True); Path(cfg.output).write_text(json.dumps(report,indent=2,ensure_ascii=False)); return report

def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    for f in ['inputs','market_csv','output']: p.add_argument('--'+f.replace('_','-'),required=True)
    for f in ['train_start','train_end','test_start','test_end','eval_start','eval_end']: p.add_argument('--'+f.replace('_','-'),default=getattr(BinaryEdgeStrictScanConfig,f))
    for f in ['min_support_train','min_support_test','min_test_trades','max_rules','top_k','max_rule_terms','max_state_features','max_hold_bars','entry_delay_bars']: p.add_argument('--'+f.replace('_','-'),type=int,default=getattr(BinaryEdgeStrictScanConfig,f))
    for f in ['leverage','fee_rate','slippage_rate']: p.add_argument('--'+f.replace('_','-'),type=float,default=getattr(BinaryEdgeStrictScanConfig,f))
    return p.parse_args()

def main():
    print(json.dumps({'top_by_test':run(BinaryEdgeStrictScanConfig(**vars(parse_args())))['top_by_test'][:10]},indent=2,ensure_ascii=False))
if __name__=='__main__': main()
