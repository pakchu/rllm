"""Standalone no-LLM validation for frozen REX prediction artifacts.

This intentionally avoids model inference: it compares raw REX source candidates
against the fixed dual-regime rule gate exported by
``training/export_dual_regime_predictions.py``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import training.evaluate_portfolio_llm_selector as ep

PRED_FILES = [
 'results/rex_dual_regime_train_2021_2023_predictions_2026-07-03.jsonl',
 'results/rex_dual_regime_test_2024_predictions_2026-07-03.jsonl',
 'results/rex_dual_regime_eval_2025_2026h1_predictions_2026-07-03.jsonl',
]
DEFAULT_OUT='results/rex_no_llm_standalone_check_2026-07-10.json'
DEFAULT_DOC='docs/rex-no-llm-standalone-check-2026-07-10.md'
COST=0.0005

def load_rows(paths=PRED_FILES):
    rows=[]
    for f in paths:
        if not Path(f).exists(): continue
        for line in Path(f).read_text().splitlines():
            if line.strip():
                r=json.loads(line); r['_file']=f; rows.append(r)
    rows.sort(key=lambda r:int(r['signal_pos']))
    return rows

def metric(events, mask, years):
    idx=np.flatnonzero(mask)
    if len(idx)==0: return {}
    start, end = int(idx[0]), int(idx[-1])+1
    r=np.zeros(end-start); adv=np.zeros(end-start)
    side=Counter(); fam=Counter(); wins=0
    for e in events:
        rr=e['ret'][start:end]; aa=e['adv'][start:end]
        if np.any(np.abs(rr)>1e-15) or np.any(np.abs(aa)>1e-15):
            r += rr; adv += aa; side[e['side'].upper()]+=1; fam[e['family']]+=1; wins += e['real']>0
    fac=np.maximum(0,1+r)
    eqp=np.cumprod(fac) if len(fac) else np.array([1.0])
    eqb=np.concatenate([[1.0], eqp[:-1]]) if len(fac) else np.array([1.0])
    pka=np.maximum.accumulate(eqp); pkb=np.maximum.accumulate(eqb)
    mdd1=float(np.nanmax(1-eqp/np.maximum(pka,1e-12))) if len(eqp) else 0.0
    mdd2=float(np.nanmax(1-(eqb*(1+adv))/np.maximum(pkb,1e-12))) if len(eqb) else 0.0
    mdd=max(mdd1,mdd2)*100
    eq=float(eqp[-1]) if len(eqp) else 1.0
    ret=(eq-1)*100
    cagr=(eq**(1/years)-1)*100 if eq>0 else -100.0
    trades=sum(side.values())
    return {
        'total_return_pct':ret,
        'cagr_pct':cagr,
        'strict_mdd_pct':mdd,
        'cagr_to_strict_mdd':cagr/mdd if mdd>1e-12 else 0.0,
        'trade_entries':int(trades),
        'win_rate':wins/trades if trades else 0.0,
        'side_counts':dict(side),
        'family_counts':dict(fam),
        'active_bars':int(np.count_nonzero(np.abs(r)>1e-15))
    }

def build_events(rows, market, masks, mode):
    events=[]
    dates=pd.to_datetime(market['date'])
    for sp, sm in masks.items():
        nxt=0
        for row in rows:
            p=int(row['signal_pos'])
            if not (0<=p<len(sm) and sm[p]): continue
            if mode=='source_all':
                act=row.get('source_action') or {}
                side=str(act.get('side','')).lower(); hold=int(act.get('hold_bars') or 0); fam=str(act.get('family','unknown'))
            elif mode=='rule_gate':
                pred=row.get('prediction') or {}
                if pred.get('gate')!='TRADE': continue
                side=str(pred.get('side','')).lower(); hold=int(pred.get('hold_bars') or 0); fam=str(pred.get('family') or row.get('source_action',{}).get('family','unknown'))
            elif mode=='rule_gate_short_only':
                pred=row.get('prediction') or {}
                if pred.get('gate')!='TRADE' or str(pred.get('side')).upper()!='SHORT': continue
                side='short'; hold=int(pred.get('hold_bars') or 0); fam=str(pred.get('family') or row.get('source_action',{}).get('family','unknown'))
            else:
                raise ValueError(mode)
            xp=p+1+hold
            if side not in ('long','short') or hold<=0 or p<143 or p<nxt or xp>=len(market) or not sm[min(xp,len(sm)-1)]:
                continue
            r, adv, real = ep._event_return(market,p,hold,side,cost=COST)
            events.append({'split':sp,'signal_pos':p,'date':str(dates.iloc[p]),'side':side,'family':fam,'hold_bars':hold,'ret':r,'adv':adv,'real':real})
            nxt=xp
    return events

def run(*, output: str = DEFAULT_OUT, doc: str = DEFAULT_DOC, prediction_files: list[str] | None = None) -> dict:
    market, feat, masks, years = ep._prep()
    paths = prediction_files or list(PRED_FILES)
    rows=load_rows(paths)
    report={'protocol':{
        'llm_used':False,
        'modes':{
            'source_all':'Execute every source_action REX candidate; this is raw REX without any selector/filter.',
            'rule_gate':'Execute fixed dual-regime hardcoded gates from export_dual_regime_predictions.py; no model inference.',
            'rule_gate_short_only':'rule_gate restricted to SHORT predictions, matching bearish-regime deployment idea.'
        },
        'entry':'signal_pos+1 open', 'cost_bps_round_trip':10, 'cagr_window':'full split calendar including idle time', 'strict_mdd':'equity close-to-close plus intraposition adverse excursion'
    }, 'input_files':paths, 'row_count':len(rows), 'splits':ep.SPLITS, 'results':{}}
    for mode in ['source_all','rule_gate','rule_gate_short_only']:
        ev=build_events(rows,market,masks,mode)
        report['results'][mode]={'event_count':len(ev),'windows':{sp:metric(ev,masks[sp],years[sp]) for sp in ep.SPLITS}}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report,indent=2,ensure_ascii=False))
    def fmt(m):
        return f"ret {m['total_return_pct']:.2f}% | CAGR {m['cagr_pct']:.2f}% | strict MDD {m['strict_mdd_pct']:.2f}% | ratio {m['cagr_to_strict_mdd']:.2f} | trades {m['trade_entries']} | win {m['win_rate']:.1%} | sides {m['side_counts']}"
    lines=['# REX no-LLM standalone check (2026-07-10)','', 'This report disables LLM inference and evaluates REX in two non-LLM forms: raw source candidates and the fixed dual-regime rule gate.', '', f'- Result JSON: `{output}`', '- No model adapter / LoRA / LLM selector used.', '- CAGR denominator is the full split window including idle time.', '- Strict MDD includes intraposition adverse excursion.', '']
    for mode, payload in report['results'].items():
        lines += [f'## {mode}', '', f"events: {payload['event_count']}", '']
        for sp in ep.SPLITS:
            lines.append(f"- {sp}: {fmt(payload['windows'][sp])}")
        lines.append('')
    Path(doc).parent.mkdir(parents=True, exist_ok=True)
    Path(doc).write_text('\n'.join(lines))
    return {'output':output,'doc':doc,'summary':{mode:{sp:report['results'][mode]['windows'][sp] for sp in ['test2024','eval2025','ytd2026']} for mode in report['results']}}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default=DEFAULT_OUT)
    p.add_argument('--doc', default=DEFAULT_DOC)
    p.add_argument('--prediction-files', nargs='*', default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(run(output=args.output, doc=args.doc, prediction_files=args.prediction_files), indent=2, ensure_ascii=False))


if __name__=='__main__': main()
