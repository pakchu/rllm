"""Evaluate Kimchi-flow activation JSON policy rows."""
from __future__ import annotations

import argparse, json, re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

VALID = {
    "regime": {"KIMCHI_FLOW"},
    "decision": {"ACTIVATE", "ABSTAIN"},
    "side": {"LONG", "SHORT", "NONE"},
    "quality": {"GOOD", "MARGINAL", "BAD"},
    "confidence": {"LOW", "MID", "HIGH"},
}
DEFAULT = {"regime":"KIMCHI_FLOW","decision":"ABSTAIN","side":"NONE","quality":"MARGINAL","confidence":"LOW"}


def load_jsonl(path: str) -> list[dict[str, Any]]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def parse_activation_json(text: str) -> dict[str, str]:
    raw=str(text).strip(); obj: Any={}
    try: obj=json.loads(raw)
    except Exception:
        for m in re.finditer(r"\{[^{}]*\}", raw, flags=re.DOTALL):
            try: cand=json.loads(m.group(0))
            except Exception: continue
            if isinstance(cand, dict): obj=cand; break
    if not isinstance(obj, dict): obj={}
    out=dict(DEFAULT)
    for k, allowed in VALID.items():
        val=str(obj.get(k,out[k])).upper()
        out[k]=val if val in allowed else out[k]
    if out['decision']=='ABSTAIN': out['side']='NONE'
    elif out['side']=='NONE': out['side']='LONG'
    return out


def _key(o: dict[str,str]) -> str:
    return f"decision={o['decision']},side={o['side']},quality={o['quality']},confidence={o['confidence']}"


def metrics(rows: list[dict[str,Any]], preds: list[dict[str,str]]) -> dict[str,Any]:
    exact=0; field_ok=Counter(); field_n=Counter(); conf=Counter(); pc=Counter(); tc=Counter()
    pnl_pred=0.0; pnl_oracle=0.0; pred_trades=0; oracle_trades=0
    for r,p in zip(rows,preds):
        t=parse_activation_json(r.get('target','{}'))
        if p==t: exact+=1
        pc[_key(p)]+=1; tc[_key(t)]+=1; conf[f"target={t['decision']}/{t['side']}|pred={p['decision']}/{p['side']}"]+=1
        for k in VALID:
            field_n[k]+=1; field_ok[k]+= int(p.get(k)==t.get(k))
        ret=float(r.get('trade_ret_pct',0.0))
        if t['decision']=='ACTIVATE': pnl_oracle+=ret; oracle_trades+=1
        if p['decision']=='ACTIVATE': pnl_pred+=ret; pred_trades+=1
    n=len(rows)
    return {"rows":n,"exact_accuracy":exact/max(1,n),"field_accuracy":{k:field_ok[k]/max(1,field_n[k]) for k in sorted(VALID)},"confusion":dict(conf),"prediction_counts":dict(pc),"target_counts":dict(tc),"activation_pnl_proxy":{"pred_sum_ret_pct":pnl_pred,"oracle_sum_ret_pct":pnl_oracle,"pred_activations":pred_trades,"oracle_activations":oracle_trades}}


def evaluate(eval_jsonl: str, output: str, prediction_mode: str='target_echo', predictions_jsonl: str='') -> dict[str,Any]:
    rows=load_jsonl(eval_jsonl)
    if prediction_mode=='target_echo': preds=[parse_activation_json(r['target']) for r in rows]
    elif prediction_mode=='all_abstain': preds=[dict(DEFAULT) for _ in rows]
    elif prediction_mode=='all_activate_long': preds=[{"regime":"KIMCHI_FLOW","decision":"ACTIVATE","side":"LONG","quality":"GOOD","confidence":"HIGH"} for _ in rows]
    else: raise ValueError('unsupported prediction_mode')
    report={"as_of":datetime.now(timezone.utc).isoformat(),"eval_jsonl":eval_jsonl,"prediction_mode":prediction_mode,"metrics":metrics(rows,preds),"leakage_guard":{"target_echo_for_pipeline_only":prediction_mode=='target_echo'}}
    Path(output).parent.mkdir(parents=True, exist_ok=True); Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    if predictions_jsonl:
        Path(predictions_jsonl).write_text('\n'.join(json.dumps({"date":r['date'],"target":parse_activation_json(r['target']),"prediction":p,"trade_ret_pct":r.get('trade_ret_pct')}, ensure_ascii=False, sort_keys=True) for r,p in zip(rows,preds))+'\n')
    return report


def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--eval-jsonl',required=True); p.add_argument('--output',required=True); p.add_argument('--prediction-mode',choices=['target_echo','all_abstain','all_activate_long'],default='target_echo'); p.add_argument('--predictions-jsonl',default=''); return p.parse_args()

if __name__=='__main__': print(json.dumps(evaluate(**vars(parse_args())), indent=2, ensure_ascii=False))
