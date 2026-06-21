"""Summarize realized rewards by predicted decision for SFT/ranker eval rows."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def _read(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]

def summarize(eval_jsonl: str, prediction_report: str, output: str) -> dict:
    rows=_read(eval_jsonl)
    rep=json.loads(Path(prediction_report).read_text())
    preds=rep.get('predictions') or []
    buckets={}
    selected=[]
    non_abstain_full=[]
    take_full_only=[]
    for p in preds:
        i=int(p['index']); pred=str(p['prediction']); row=rows[i]
        reward=float(((row.get('source') or {}).get('reward') or {}).get('trade_ret_pct',0.0))/100.0
        buckets.setdefault(pred,[]).append(reward)
        if pred in {'TAKE_FULL','TAKE_SMALL'}:
            selected.append(reward if pred=='TAKE_FULL' else reward*0.5)
            non_abstain_full.append(reward)
        if pred == 'TAKE_FULL':
            take_full_only.append(reward)
    def stats(xs):
        arr=np.asarray(xs,dtype=float)
        if len(arr)==0: return {'n':0,'mean_pct':0.0,'compound_ret_pct':0.0,'positive_rate':0.0}
        return {'n':int(len(arr)),'mean_pct':float(arr.mean()*100),'compound_ret_pct':float((np.prod(1+arr)-1)*100),'positive_rate':float(np.mean(arr>0))}
    out={'eval_jsonl':eval_jsonl,'prediction_report':prediction_report,'by_prediction':{k:stats(v) for k,v in sorted(buckets.items())},'selected_policy_take_full_take_small_half':stats(selected),'selected_policy_non_abstain_full':stats(non_abstain_full),'selected_policy_take_full_only':stats(take_full_only),'all_rows':stats([float(((r.get('source') or {}).get('reward') or {}).get('trade_ret_pct',0.0))/100.0 for r in rows]),'leakage_guard':{'uses_predictions_and_realized_eval_rewards_for_reporting_only':True}}
    Path(output).parent.mkdir(parents=True,exist_ok=True); Path(output).write_text(json.dumps(out,indent=2,ensure_ascii=False)); return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--eval-jsonl',required=True); ap.add_argument('--prediction-report',required=True); ap.add_argument('--output',required=True); print(json.dumps(summarize(**vars(ap.parse_args())),indent=2,ensure_ascii=False))
if __name__=='__main__': main()
