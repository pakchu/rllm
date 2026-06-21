"""Backtest train-only token-rule selected wave state rows."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from training.eval_wave_text_state_rules import _fit,_read,_score
from training.backtest_decision_predictions import DecisionBacktestConfig, run as run_decision_bt

def run(train_jsonl:str, eval_jsonl:str, market_csv:str, output:str, predictions_output:str, quantile:float=0.5, aggregate_duplicates:bool=True, leverage:float=1.0):
    train=_read(train_jsonl); ev=_read(eval_jsonl); weights=_fit(train); scores=[_score(r,weights) for r in train]; scores=sorted(scores); th=scores[min(len(scores)-1,max(0,int(float(quantile)*(len(scores)-1))))]
    pred_report={'predictions':[]}
    bt_rows=[]
    for i,row in enumerate(ev):
        pred='TAKE_FULL' if _score(row,weights)>=th else 'ABSTAIN'
        target=(json.loads(str(row.get('target','{}'))).get('decision','') if isinstance(row.get('target'),str) else (row.get('target') or {}).get('decision',''))
        pred_report['predictions'].append({'index':i,'prediction':pred,'target':target})
        src=row.get('source') or {'date':row.get('date'),'signal_pos':row.get('signal_pos'),'side':row.get('side'),'reward':row.get('reward'),'state_tokens':row.get('state_tokens')}
        bt_rows.append({'prompt':row.get('prompt',''), 'target': json.dumps({'decision': target}), 'source': src})
    report_path=str(Path(output).with_suffix('.prediction_report.json'))
    bt_eval_path=str(Path(output).with_suffix('.bt_eval_rows.jsonl'))
    Path(report_path).write_text(json.dumps(pred_report,indent=2,ensure_ascii=False))
    Path(bt_eval_path).write_text('\n'.join(json.dumps(r,ensure_ascii=False,sort_keys=True) for r in bt_rows)+'\n')
    bt=run_decision_bt(DecisionBacktestConfig(eval_jsonl=bt_eval_path,prediction_report=report_path,market_csv=market_csv,output=output,predictions_output=predictions_output,policy='non_abstain_full',aggregate_duplicates=aggregate_duplicates,leverage=leverage))
    bt['text_rule_config']={'train_jsonl':train_jsonl,'eval_jsonl':eval_jsonl,'quantile':quantile,'threshold':th,'aggregate_duplicates':aggregate_duplicates,'leverage':leverage,'prediction_report':report_path}
    Path(output).write_text(json.dumps(bt,indent=2,ensure_ascii=False)); return bt

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--train-jsonl',required=True); ap.add_argument('--eval-jsonl',required=True); ap.add_argument('--market-csv',required=True); ap.add_argument('--output',required=True); ap.add_argument('--predictions-output',required=True); ap.add_argument('--quantile',type=float,default=0.5); ap.add_argument('--leverage',type=float,default=1.0); ap.add_argument('--no-aggregate-duplicates',dest='aggregate_duplicates',action='store_false'); ap.set_defaults(aggregate_duplicates=True)
    r=run(**vars(ap.parse_args())); print(json.dumps({'sim':r['sim'],'trade_stats':r['trade_stats'],'text_rule_config':r['text_rule_config']},indent=2,ensure_ascii=False))
if __name__=='__main__': main()
