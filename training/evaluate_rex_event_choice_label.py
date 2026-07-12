"""Evaluate equal-form REX event choice labels with candidate logprob."""
from __future__ import annotations

import argparse, json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.build_rex_event_choice_label_data import ACTIONS, LABELS
from training.event_candidate_pool_probe import EventPoolConfig, _load_market, _simulate_rows
from training.train_text_sft import resolve_text_causal_lm_alias, load_jsonl
from utils import disable_transformers_allocator_warmup

CANDIDATES = ["CHOICE_A_LONG", "CHOICE_B_SHORT", "CHOICE_C_SKIP"]

@dataclass(frozen=True)
class Cfg:
    eval_jsonl: str
    output_json: str
    market_csv: str
    model_name: str = "gemma2-2b-it"
    adapter_dir: str = ""
    max_samples: int = 0
    sample_mode: str = "sequential"
    batch_size: int = 8
    score_normalization: str = "mean"
    hold_bars: int = 144
    entry_delay_bars: int = 1
    leverage: float = 0.5
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001


def _chat(tokenizer: Any, prompt: str) -> str:
    messages=[{"role":"user","content":prompt}]
    if getattr(tokenizer,"chat_template",None):
        return str(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    return f"<|user|>\n{prompt}\n<|assistant|>\n"


def _load(model_name: str, adapter_dir: str):
    disable_transformers_allocator_warmup()
    resolved=resolve_text_causal_lm_alias(model_name, prefer_latest=True)
    tok=AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    tok.padding_side="right"
    base=AutoModelForCausalLM.from_pretrained(resolved, trust_remote_code=True, device_map="auto")
    model=PeftModel.from_pretrained(base, adapter_dir) if adapter_dir else base
    model.eval()
    return tok, model, resolved


def _score_batch(model: Any, input_ids: Any, attention_mask: Any, spans: list[tuple[int,int]], normalize: str) -> list[float]:
    import torch
    with torch.no_grad(): logits=model(input_ids=input_ids, attention_mask=attention_mask).logits
    scores=[]
    for i,(start,end) in enumerate(spans):
        pos=torch.arange(start-1,end-1,device=logits.device)
        labels=input_ids[i,start:end]
        selected=logits[i,pos,:].float()
        label_logits=selected.gather(1,labels.reshape(-1,1)).squeeze(1)
        ts=label_logits-torch.logsumexp(selected,dim=-1)
        scores.append(float((ts.sum() if normalize=="sum" else ts.mean()).detach().cpu()))
    return scores


def predict(rows: list[dict[str,Any]], cfg: Cfg):
    tok, model, resolved = _load(cfg.model_name, cfg.adapter_dir)
    cand_ids=[]
    for c in CANDIDATES:
        ids=tok(c, add_special_tokens=False)["input_ids"]
        if tok.eos_token_id is not None: ids=ids+[int(tok.eos_token_id)]
        cand_ids.append(ids)
    preds=[]; score_rows=[]
    bs=max(1,int(cfg.batch_size)); norm=str(cfg.score_normalization)
    for off in range(0,len(rows),bs):
        batch=rows[off:off+bs]
        seqs=[]; spans=[]; row_counts=[]
        for r in batch:
            pids=tok(_chat(tok,str(r["prompt"])), add_special_tokens=False)["input_ids"]
            st=len(pids); row_counts.append(len(cand_ids))
            for ids in cand_ids:
                seqs.append(pids+ids); spans.append((st,st+len(ids)))
        enc=tok.pad({"input_ids":seqs}, return_tensors="pt")
        scores=_score_batch(model, enc["input_ids"].to(model.device), enc["attention_mask"].to(model.device), spans, norm)
        k=0
        for r in batch:
            ss=scores[k:k+len(CANDIDATES)]; k+=len(CANDIDATES)
            best=max(range(len(ss)), key=lambda i:ss[i])
            pred=CANDIDATES[best]
            preds.append(pred)
            score_rows.append({"date":r.get("date"),"signal_pos":r.get("signal_pos"),"target":r.get("target"),"prediction":pred,"scores":dict(zip(CANDIDATES,ss))})
    return preds, score_rows, resolved


def backtest(rows: list[dict[str,Any]], preds: list[str], cfg: Cfg) -> dict[str,Any]:
    market=_load_market(cfg.market_csv)
    trade_rows=[]
    for r,p in zip(rows,preds):
        action=ACTIONS.get(p,"NO_TRADE")
        if action in {"LONG","SHORT"}:
            trade_rows.append({"date":r["date"],"signal_date":r["date"],"side":action,"family":"rex_choice_label_llm","strength":1.0,"score_mean":1.0})
    ecfg=EventPoolConfig(input_csv=cfg.market_csv, output="", hold_bars=cfg.hold_bars, entry_delay_bars=cfg.entry_delay_bars, leverage=cfg.leverage, fee_rate=cfg.fee_rate, slippage_rate=cfg.slippage_rate)
    res=_simulate_rows(trade_rows, market, ecfg)
    return {"predicted_trade_rows":len(trade_rows),"sim":res.get("sim",{}),"trade_stats":res.get("trade_stats",{})}


def run(cfg: Cfg) -> dict[str,Any]:
    rows=load_jsonl(cfg.eval_jsonl, max_samples=cfg.max_samples, sample_mode=cfg.sample_mode, seed=42)
    preds, score_rows, resolved = predict(rows,cfg)
    correct=0; confusion=Counter(); pc=Counter(); tc=Counter()
    for r,p in zip(rows,preds):
        t=str(r["target"]); correct += int(t==p); confusion[f"target={t}|pred={p}"]+=1; pc[p]+=1; tc[t]+=1
    report={
        "config":asdict(cfg),"model_name_resolved":resolved,
        "metrics":{"num_samples":len(rows),"accuracy":correct/max(1,len(rows)),"target_counts":dict(tc),"prediction_counts":dict(pc),"confusion":dict(confusion)},
        "backtest":backtest(rows,preds,cfg),"score_rows":score_rows,
        "leakage_guard":{"model_sees_prompt_only":True,"targets_used_for_metrics_only":True},
    }
    Path(cfg.output_json).parent.mkdir(parents=True,exist_ok=True)
    Path(cfg.output_json).write_text(json.dumps(report,indent=2,ensure_ascii=False))
    return {k:v for k,v in report.items() if k!="score_rows"}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--eval-jsonl",required=True); p.add_argument("--output-json",required=True); p.add_argument("--market-csv",required=True)
    p.add_argument("--model-name",default=Cfg.model_name); p.add_argument("--adapter-dir",default="")
    p.add_argument("--max-samples",type=int,default=0); p.add_argument("--sample-mode",default="sequential")
    p.add_argument("--batch-size",type=int,default=8); p.add_argument("--score-normalization",choices=["mean","sum"],default="mean")
    print(json.dumps(run(Cfg(**vars(p.parse_args()))),indent=2,ensure_ascii=False))
if __name__=="__main__": main()
