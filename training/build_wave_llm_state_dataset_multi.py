"""Build multi-policy bucketed LLM state rows from fold-consistent wave policies."""
from __future__ import annotations
import argparse,json,tempfile
from dataclasses import dataclass,asdict
from pathlib import Path
from typing import Any
import numpy as np, pandas as pd
from preprocessing.external_features import attach_wave_trading_external_features
from preprocessing.market_features import EXTENDED_MARKET_FEATURE_COLUMNS, build_market_feature_frame
from training.build_wave_llm_state_dataset import _load_market,_state_tokens,_target_from_reward,_write_jsonl
from training.online_risk_overlay_backtest import OnlineRiskOverlayConfig, run_overlay
from training.sweep_wave_fold_consistency import _load_closes,_parse_folds,_write_policy_predictions
from training.sweep_wave_teacher_rllm_thresholds import _rolling_prob_rows
from training.validate_wave_trading_best import _build_best_features,_load_wave_module

@dataclass(frozen=True)
class MultiCfg:
    wave_root:str; market_5m_csv:str; fold_consistency_report:str; train_output:str; eval_output:str; summary_output:str
    top_k:int=5; start_date:str='2020-01-01'; end_date:str='2026-06-02'
    selection_folds:str='2021-01-01|2021-06-30 23:59:59,2021-07-01|2021-12-31 23:59:59,2022-01-01|2022-06-30 23:59:59,2022-07-01|2022-12-31 23:59:59,2023-01-01|2023-06-30 23:59:59,2023-07-01|2023-12-31 23:59:59,2024-01-01|2024-06-30 23:59:59'
    eval_start:str='2024-07-01'; eval_end:str='2026-06-01 00:00:00'; lr_c:float=0.05; lr_penalty:str='l1'
    leverage:float=1.0; entry_delay_bars:int=3; atr_trailing_stop_mult:float=3.75; atr_period:int=45; external_tolerance:str='30min'

def _prompt(row:dict[str,Any], tokens:dict[str,str], policy:dict[str,Any], rank:int)->str:
    side=str(row['side']); prob=float(row.get('teacher_probability_long',0.5)); margin=prob-float(policy['long_th']) if side=='LONG' else float(policy['short_th'])-prob
    mb='thin' if margin<0.02 else 'normal' if margin<0.06 else 'wide'
    rule=f"rank={rank}; long_th={policy['long_th']}; short_th={policy['short_th']}; trend_window={policy['trend_window']}; long_mode={policy['long_mode']}; short_mode={policy['short_mode']}"
    lines=['Task: decide whether to take, reduce, or reject this pre-generated BTCUSDT futures candidate.','Use only the bucketed state below. Do not infer from future outcome metadata.',f'Candidate: side={side}; source=15m_wave_teacher; confidence_margin={mb}; execution=next_15m_open; exit=atr_trailing_or_time.',f'Expert policy: {rule}','State buckets:']
    lines += [f'- {k}: {tokens[k]}' for k in sorted(tokens)]
    lines.append('Return JSON with decision in {TAKE_FULL, TAKE_SMALL, ABSTAIN} and a short risk reason.')
    return '\n'.join(lines)

def _rows(rows, *, market_csv, features, closes, policy, rank, tmp, tag, cfg:MultiCfg, hold_bars, split):
    pred=tmp/f'{tag}.jsonl'; _write_policy_predictions(rows,pred,closes=closes,policy=policy,hold_bars=hold_bars)
    bt=run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=str(pred),market_csv=market_csv,output=str(tmp/f'{tag}.bt.json'),leverage=cfg.leverage,entry_delay_bars=cfg.entry_delay_bars,atr_trailing_stop_mult=cfg.atr_trailing_stop_mult,atr_period=cfg.atr_period))
    src_by_pos={int(r['signal_pos']):r for r in rows}; out=[]
    for ex in bt['executed']:
        pos=int(ex['signal_pos']); src=src_by_pos.get(pos)
        if not src: continue
        side=str(ex['side']); toks=_state_tokens(features,pos,side); reward=float(ex['trade_ret_pct'])
        out.append({'split':split,'date':ex.get('date'),'signal_pos':pos,'side':side,'policy_rank':rank,'prompt':_prompt({**src,'side':side},toks,policy,rank),'target':_target_from_reward(reward),'reward':{'trade_ret_pct':reward,'equity_after_trade':float(ex.get('equity',0.0)),'exit_reason':ex.get('exit_reason')},'candidate':{'teacher_probability_long':float(src.get('teacher_probability_long',0.5)),'policy':policy,'hold_bars':hold_bars,'policy_rank':rank},'state_tokens':toks,'leakage_guard':{'prompt_uses_future_reward':False,'reward_is_label_only':True,'features_signal_time_or_prior':True}})
    return out

def _summary(rows):
    rewards=np.asarray([float(r['reward']['trade_ret_pct']) for r in rows],dtype=float) if rows else np.asarray([]); dec={}; ranks={}
    for r in rows: dec[r['target']['decision']]=dec.get(r['target']['decision'],0)+1; ranks[str(r['policy_rank'])]=ranks.get(str(r['policy_rank']),0)+1
    return {'rows':len(rows),'mean_reward_pct':float(rewards.mean()) if len(rewards) else 0.0,'positive_rate':float(np.mean(rewards>0)) if len(rewards) else 0.0,'decisions':dec,'policy_ranks':ranks}

def run(cfg:MultiCfg):
    fc=json.load(open(cfg.fold_consistency_report)); policies=[x['policy'] for x in fc['top10'][:int(cfg.top_k)]]
    psr=_load_wave_module(cfg.wave_root); data=_build_best_features(psr,start_date=cfg.start_date,end_date=cfg.end_date,time_interval='15m'); hold=int(data['params']['holding_period'])*3
    market=_load_market(cfg.market_5m_csv); enriched=attach_wave_trading_external_features(market,wave_trading_root=cfg.wave_root,tolerance=cfg.external_tolerance); features=build_market_feature_frame(enriched)
    for c in EXTENDED_MARKET_FEATURE_COLUMNS:
        if c not in features.columns: features[c]=0.0
    closes=_load_closes(cfg.market_5m_csv); folds=_parse_folds(cfg.selection_folds)
    train_prob=[_rolling_prob_rows(psr,data,market_5m_csv=cfg.market_5m_csv,eval_start=a,eval_end=b,lr_c=cfg.lr_c,lr_penalty=cfg.lr_penalty) for a,b in folds]
    eval_prob=_rolling_prob_rows(psr,data,market_5m_csv=cfg.market_5m_csv,eval_start=cfg.eval_start,eval_end=cfg.eval_end,lr_c=cfg.lr_c,lr_penalty=cfg.lr_penalty)
    train=[]; ev=[]
    with tempfile.TemporaryDirectory(prefix='rllm_wave_llm_multi_') as td:
        tmp=Path(td)
        for rank,pol in enumerate(policies,1):
            for i,rs in enumerate(train_prob): train += _rows(rs,market_csv=cfg.market_5m_csv,features=features,closes=closes,policy=pol,rank=rank,tmp=tmp,tag=f'train_p{rank}_f{i}',cfg=cfg,hold_bars=hold,split='train')
            ev += _rows(eval_prob,market_csv=cfg.market_5m_csv,features=features,closes=closes,policy=pol,rank=rank,tmp=tmp,tag=f'eval_p{rank}',cfg=cfg,hold_bars=hold,split='eval')
    _write_jsonl(cfg.train_output,train); _write_jsonl(cfg.eval_output,ev)
    summ={'config':asdict(cfg),'policies':policies,'outputs':{'train':cfg.train_output,'eval':cfg.eval_output},'train':_summary(train),'eval':_summary(ev),'leakage_guard':{'policies_selected_before_eval':True,'features_signal_time_or_prior':True,'eval_rows_not_used_for_training':True}}
    Path(cfg.summary_output).parent.mkdir(parents=True,exist_ok=True); Path(cfg.summary_output).write_text(json.dumps(summ,indent=2,ensure_ascii=False)); return summ

def parse_args():
    ap=argparse.ArgumentParser();
    for a in ['wave-root','market-5m-csv','fold-consistency-report','train-output','eval-output','summary-output']: ap.add_argument('--'+a,required=True)
    ap.add_argument('--top-k',type=int,default=5); return ap.parse_args()
def main(): print(json.dumps(run(MultiCfg(**vars(parse_args()))),indent=2,ensure_ascii=False))
if __name__=='__main__': main()
