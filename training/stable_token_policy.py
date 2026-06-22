"""Build and evaluate period-stability-aware token policies.

Rules are accepted only if the same token combo has positive mean utility in
multiple historical periods.  This is a conservative relabeling probe: if stable
historical rules do not survive OOS, the target surface is not alpha-bearing.
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class StableTokenCfg:
    input_jsonl: str
    output: str
    predictions_output: str
    fit_end: str = "2024-12-31 23:59:59"
    eval_start: str = "2025-01-01"
    max_combo_size: int = 2
    min_count_per_period: int = 5
    min_positive_periods: int = 3
    min_total_count: int = 30
    min_mean_utility: float = 0.0
    min_side_gap: float = 0.0
    periods: str = "2020,2021,2022,2023,2024"
    max_rules_per_row: int = 1


def _load(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in open(path) if line.strip()]


def _period(row: dict[str, Any]) -> str:
    return str(row.get('date',''))[:4]


def _features(row: dict[str, Any]) -> list[tuple[str,str]]:
    toks=row.get('state_tokens',{}) if isinstance(row.get('state_tokens'),dict) else {}
    return sorted((str(k),str(v)) for k,v in toks.items())


def _combos(feats: list[tuple[str,str]], max_k: int) -> Iterable[tuple[tuple[str,str],...]]:
    for k in range(1,int(max_k)+1):
        yield from itertools.combinations(feats,k)


def _utils(row: dict[str, Any]) -> tuple[float,float]:
    a=row.get('reward_audit',{}) if isinstance(row.get('reward_audit'),dict) else {}
    return float(a.get('LONG',{}).get('utility',0.0)), float(a.get('SHORT',{}).get('utility',0.0))


def _rule_str(c: tuple[tuple[str,str],...])->str:
    return ' & '.join(f'{k}={v}' for k,v in c)


def _build_rules(rows: list[dict[str, Any]], cfg: StableTokenCfg) -> dict[tuple[tuple[str,str],...],dict[str,Any]]:
    allowed=set(x.strip() for x in cfg.periods.split(',') if x.strip())
    stats: dict[tuple[tuple[str,str],...], dict[str, list[float]]] = defaultdict(lambda: defaultdict(lambda:[0.0,0.0,0.0]))
    for r in rows:
        if str(r.get('date','')) > cfg.fit_end: continue
        per=_period(r)
        if allowed and per not in allowed: continue
        lu,su=_utils(r)
        for c in _combos(_features(r), cfg.max_combo_size):
            s=stats[c][per]; s[0]+=1; s[1]+=lu; s[2]+=su
    rules={}
    for c, byp in stats.items():
        total_n=sum(v[0] for v in byp.values())
        if total_n < cfg.min_total_count: continue
        period_votes={'LONG':0,'SHORT':0}; period_details=[]; total_long=0.0; total_short=0.0
        for per,v in byp.items():
            n,ls,ss=v
            if n < cfg.min_count_per_period: continue
            lm=ls/n; sm=ss/n; side='LONG' if lm>=sm else 'SHORT'; best=max(lm,sm); gap=abs(lm-sm)
            if best >= cfg.min_mean_utility and gap >= cfg.min_side_gap:
                period_votes[side]+=1
            period_details.append({'period':per,'n':int(n),'long_mean':lm,'short_mean':sm,'side':side,'best':best,'gap':gap})
            total_long+=ls; total_short+=ss
        side='LONG' if period_votes['LONG']>=period_votes['SHORT'] else 'SHORT'
        pos=period_votes[side]
        if pos < cfg.min_positive_periods: continue
        lm=total_long/total_n; sm=total_short/total_n; best=max(lm,sm); gap=abs(lm-sm)
        if (side=='LONG' and lm < sm) or (side=='SHORT' and sm < lm): continue
        rules[c]={'rule':_rule_str(c),'side':side,'positive_periods':int(pos),'period_votes':period_votes,'n_total':int(total_n),'long_mean':lm,'short_mean':sm,'mean_utility':best,'side_gap':gap,'periods':period_details}
    return rules


def _predict(rows: list[dict[str,Any]], rules: dict[tuple[tuple[str,str],...],dict[str,Any]], cfg: StableTokenCfg) -> list[dict[str,Any]]:
    out=[]
    for r in rows:
        if str(r.get('date','')) < cfg.eval_start: continue
        matches=[rules[c] for c in _combos(_features(r), cfg.max_combo_size) if c in rules]
        matches.sort(key=lambda x:(x['positive_periods'],x['mean_utility'],x['n_total']), reverse=True)
        matches=matches[:max(1,cfg.max_rules_per_row)]
        if not matches: pred='NO_TRADE'
        else:
            score=defaultdict(float)
            for m in matches: score[m['side']]+=float(m['positive_periods'])*float(m['mean_utility'])
            pred=max(score,key=score.get)
        out.append({'date':r.get('date'),'signal_pos':r.get('signal_pos'),'prediction':pred,'target':r.get('target'),'candidate':r.get('candidate',{}),'matched_rules':matches})
    return out


def run(cfg: StableTokenCfg)->dict[str,Any]:
    rows=_load(cfg.input_jsonl)
    rules=_build_rules(rows,cfg)
    preds=_predict(rows,rules,cfg)
    counts={}
    for p in preds: counts[p['prediction']]=counts.get(p['prediction'],0)+1
    report={'config':cfg.__dict__,'rules':len(rules),'prediction_rows':len(preds),'prediction_counts':dict(sorted(counts.items())),'top_rules':sorted(rules.values(),key=lambda x:(x['positive_periods'],x['mean_utility'],x['n_total']),reverse=True)[:30], 'leakage_guard':'Rules use rows <= fit_end only; predictions emit rows >= eval_start only.'}
    Path(cfg.output).parent.mkdir(parents=True,exist_ok=True); Path(cfg.output).write_text(json.dumps(report,indent=2,ensure_ascii=False))
    if cfg.predictions_output:
        Path(cfg.predictions_output).parent.mkdir(parents=True,exist_ok=True); Path(cfg.predictions_output).write_text('\n'.join(json.dumps(r,ensure_ascii=False,sort_keys=True) for r in preds)+'\n')
    return report


def parse_args()->argparse.Namespace:
    p=argparse.ArgumentParser(description='Period-stable token policy')
    p.add_argument('--input-jsonl',required=True); p.add_argument('--output',required=True); p.add_argument('--predictions-output',required=True)
    p.add_argument('--fit-end',default=StableTokenCfg.fit_end); p.add_argument('--eval-start',default=StableTokenCfg.eval_start)
    p.add_argument('--max-combo-size',type=int,default=2); p.add_argument('--min-count-per-period',type=int,default=5); p.add_argument('--min-positive-periods',type=int,default=3); p.add_argument('--min-total-count',type=int,default=30); p.add_argument('--min-mean-utility',type=float,default=0.0); p.add_argument('--min-side-gap',type=float,default=0.0); p.add_argument('--periods',default=StableTokenCfg.periods); p.add_argument('--max-rules-per-row',type=int,default=1)
    return p.parse_args()


def main()->None:
    print(json.dumps(run(StableTokenCfg(**vars(parse_args()))),indent=2,ensure_ascii=False))

if __name__=='__main__': main()
