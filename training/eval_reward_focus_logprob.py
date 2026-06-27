"""Evaluate focused reward SFT adapters for utility_bucket and path_shape."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from models.option_b_vlm import RECOMMENDED_VLM_MODEL, resolve_vlm_model_alias
from utils import disable_transformers_allocator_warmup

OPTIONS = {
    "path_shape": ["CLEAN_WIN_PATH", "HIGH_ADVERSE_PATH", "FAILED_FOLLOW_THROUGH", "LOW_EDGE_PATH", "MIXED_PATH"],
    "utility_bucket": ["UTILITY_LOW", "UTILITY_MID", "UTILITY_HIGH"],
}
ORDER = ("path_shape", "utility_bucket")  # json sort_keys order


@dataclass(frozen=True)
class RewardFocusEvalCfg:
    eval_jsonl: str
    output: str
    predictions_jsonl: str = ""
    train_jsonl: str = ""
    mode: str = "model_logprob"
    model_name: str = RECOMMENDED_VLM_MODEL
    adapter_dir: str = ""
    max_samples: int = 0
    batch_size: int = 16


def _load(path: str, max_samples: int = 0) -> list[dict[str, Any]]:
    rows=[]
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if max_samples and len(rows) >= int(max_samples):
                    break
    return rows


def _target(row: dict[str, Any]) -> dict[str, str]:
    obj=json.loads(str(row["target"]))
    return {k: str(obj.get(k,"")) for k in ORDER}


def _majority(train_rows: list[dict[str, Any]]) -> dict[str, str]:
    counts={k: Counter() for k in ORDER}
    for r in train_rows:
        t=_target(r)
        for k in ORDER:
            counts[k][t[k]]+=1
    return {k: counts[k].most_common(1)[0][0] for k in ORDER}


def _chat(tok: Any, prompt: str) -> str:
    msgs=[{"role":"user","content":prompt}]
    if getattr(tok,"chat_template",None):
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return f"<|user|>\n{prompt}\n<|assistant|>\n"


def _load_model(model_name: str, adapter_dir: str):
    disable_transformers_allocator_warmup()
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    resolved=resolve_vlm_model_alias(model_name, prefer_latest=True)
    tok=AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token=tok.eos_token
    model=PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(resolved, trust_remote_code=True, device_map="auto"), adapter_dir)
    model.eval()
    return tok,model


def _prefix(tok: Any, prompt: str, gold: dict[str, str], key: str) -> str:
    s=_chat(tok,prompt)+"{"
    for prev in ORDER:
        if prev == key:
            break
        s += json.dumps(prev)+":"+json.dumps(gold[prev])+","
    s += json.dumps(key)+":\""
    return s


def _score(tok: Any, model: Any, prefixes: list[str], opts: list[str], batch_size: int) -> list[float]:
    import torch
    texts=[]; prefix_lens=[]
    for p,o in zip(prefixes,opts):
        texts.append(p+o)
        prefix_lens.append(tok(p, return_tensors="pt")["input_ids"].shape[-1])
    scores=[]
    for start in range(0,len(texts),int(batch_size)):
        enc=tok(texts[start:start+int(batch_size)], return_tensors="pt", padding=True).to(model.device)
        pls=prefix_lens[start:start+int(batch_size)]
        lengths=enc["attention_mask"].sum(dim=1).detach().cpu().tolist()
        with torch.no_grad():
            logits=model(**enc).logits[:, :-1, :]
        target_ids=enc["input_ids"][:, 1:]
        logp=torch.log_softmax(logits, dim=-1)
        token_logp=logp.gather(-1,target_ids.unsqueeze(-1)).squeeze(-1)
        for i,(pl,l) in enumerate(zip(pls,lengths)):
            a=max(0,int(pl)-1); b=max(a,int(l)-1)
            scores.append(float(token_logp[i,a:b].sum().detach().cpu()))
    return scores


def _model(rows: list[dict[str, Any]], cfg: RewardFocusEvalCfg) -> tuple[list[dict[str,str]], list[dict[str,dict[str,float]]]]:
    tok,model=_load_model(cfg.model_name,cfg.adapter_dir)
    targets=[_target(r) for r in rows]
    preds=[{} for _ in rows]
    raw=[{} for _ in rows]
    for key in ORDER:
        prefixes=[]; opts=[]; idx=[]
        for i,r in enumerate(rows):
            p=_prefix(tok,str(r["prompt"]),targets[i],key)
            for opt in OPTIONS[key]:
                prefixes.append(p); opts.append(opt); idx.append(i)
        scores=_score(tok,model,prefixes,opts,int(cfg.batch_size))
        grouped={i:[] for i in range(len(rows))}
        for i,opt,score in zip(idx,opts,scores):
            grouped[i].append((opt,score))
        for i,items in grouped.items():
            best=max(items,key=lambda x:x[1])
            preds[i][key]=best[0]
            raw[i][key]={opt:score for opt,score in items}
    return preds,raw


def _metrics(rows: list[dict[str, Any]], preds: list[dict[str,str]]) -> dict[str, Any]:
    targets=[_target(r) for r in rows]
    per={}
    for k in ORDER:
        correct=sum(preds[i].get(k)==targets[i].get(k) for i in range(len(rows)))
        per[k]={"accuracy":correct/max(1,len(rows)),"correct":correct}
    exact=sum(all(preds[i].get(k)==targets[i].get(k) for k in ORDER) for i in range(len(rows)))
    return {"rows":len(rows),"exact_match":exact/max(1,len(rows)),"per_key":per}


def run(cfg: RewardFocusEvalCfg) -> dict[str, Any]:
    rows=_load(cfg.eval_jsonl,int(cfg.max_samples))
    if cfg.mode == "majority":
        if not cfg.train_jsonl:
            raise ValueError("--train-jsonl required for majority")
        maj=_majority(_load(cfg.train_jsonl,0))
        preds=[dict(maj) for _ in rows]
        raw=[{} for _ in rows]
    elif cfg.mode == "model_logprob":
        preds,raw=_model(rows,cfg)
    else:
        raise ValueError(cfg.mode)
    report={"config":asdict(cfg),**_metrics(rows,preds)}
    Path(cfg.output).parent.mkdir(parents=True,exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report,indent=2,ensure_ascii=False))
    if cfg.predictions_jsonl:
        Path(cfg.predictions_jsonl).write_text("\n".join(json.dumps({"target":_target(r),"prediction":preds[i],"scores":raw[i]},ensure_ascii=False) for i,r in enumerate(rows))+"\n")
    return report


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-jsonl",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--predictions-jsonl",default="")
    p.add_argument("--train-jsonl",default="")
    p.add_argument("--mode",choices=["majority","model_logprob"],default="model_logprob")
    p.add_argument("--model-name",default=RECOMMENDED_VLM_MODEL)
    p.add_argument("--adapter-dir",default="")
    p.add_argument("--max-samples",type=int,default=0)
    p.add_argument("--batch-size",type=int,default=16)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(RewardFocusEvalCfg(**vars(parse_args()))),indent=2,ensure_ascii=False))


if __name__ == "__main__":
    main()
