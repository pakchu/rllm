"""Evaluate reward-component SFT adapters by scoring component-label options."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from models.option_b_vlm import RECOMMENDED_VLM_MODEL, resolve_vlm_model_alias
from utils import disable_transformers_allocator_warmup

OPTIONS = {
    "mae_bucket": ["ADVERSE_LOW", "ADVERSE_MID", "ADVERSE_HIGH"],
    "mfe_bucket": ["FAVORABLE_LOW", "FAVORABLE_MID", "FAVORABLE_HIGH"],
    "mfe_to_mae_bucket": ["PAYOFF_POOR", "PAYOFF_MID", "PAYOFF_GOOD"],
    "net_bucket": ["NET_WEAK", "NET_MID", "NET_STRONG"],
    "path_shape": ["CLEAN_WIN_PATH", "HIGH_ADVERSE_PATH", "FAILED_FOLLOW_THROUGH", "LOW_EDGE_PATH", "MIXED_PATH"],
    "utility_bucket": ["UTILITY_LOW", "UTILITY_MID", "UTILITY_HIGH"],
}
ORDER = tuple(OPTIONS.keys())  # sorted target JSON order from exporter.


@dataclass(frozen=True)
class RewardComponentLogprobCfg:
    eval_jsonl: str
    output: str
    predictions_jsonl: str = ""
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
    return tok, model


def _prefix_for_key(tok: Any, prompt: str, gold: dict[str, str], key: str) -> str:
    prefix = _chat(tok, prompt) + "{"
    for prev in ORDER:
        if prev == key:
            break
        prefix += json.dumps(prev) + ":" + json.dumps(gold[prev]) + ","
    prefix += json.dumps(key) + ":\""
    return prefix


def _score_options(tok: Any, model: Any, prefixes: list[str], options: list[str], batch_size: int) -> list[float]:
    import torch

    texts=[]; prefix_lens=[]
    for prefix, opt in zip(prefixes, options):
        texts.append(prefix + opt)
        prefix_lens.append(tok(prefix, return_tensors="pt")["input_ids"].shape[-1])
    scores=[]
    for start in range(0, len(texts), int(batch_size)):
        batch_texts=texts[start:start+int(batch_size)]
        batch_prefix_lens=prefix_lens[start:start+int(batch_size)]
        enc=tok(batch_texts, return_tensors="pt", padding=True).to(model.device)
        lengths=enc["attention_mask"].sum(dim=1).detach().cpu().tolist()
        with torch.no_grad():
            logits=model(**enc).logits[:, :-1, :]
        target_ids=enc["input_ids"][:, 1:]
        logp=torch.log_softmax(logits, dim=-1)
        token_logp=logp.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        for row_idx,(prefix_len,length) in enumerate(zip(batch_prefix_lens,lengths)):
            opt_start=max(0,int(prefix_len)-1)
            opt_end=max(opt_start,int(length)-1)
            scores.append(float(token_logp[row_idx,opt_start:opt_end].sum().detach().cpu()))
    return scores


def run(cfg: RewardComponentLogprobCfg) -> dict[str, Any]:
    rows=_load(cfg.eval_jsonl, int(cfg.max_samples))
    tok,model=_load_model(cfg.model_name,cfg.adapter_dir)
    targets=[_target(r) for r in rows]
    preds=[{} for _ in rows]
    raw=[{} for _ in rows]
    for key in ORDER:
        prefixes=[]; opts=[]; index=[]
        for i,r in enumerate(rows):
            prefix=_prefix_for_key(tok, str(r["prompt"]), targets[i], key)
            for opt in OPTIONS[key]:
                prefixes.append(prefix); opts.append(opt); index.append(i)
        scores=_score_options(tok,model,prefixes,opts,int(cfg.batch_size))
        grouped={i:[] for i in range(len(rows))}
        for idx,opt,score in zip(index,opts,scores):
            grouped[idx].append((opt,score))
        for i,items in grouped.items():
            best=max(items, key=lambda x:x[1])
            preds[i][key]=best[0]
            raw[i][key]={opt:score for opt,score in items}
    per_key={}
    for key in ORDER:
        correct=sum(preds[i].get(key)==targets[i].get(key) for i in range(len(rows)))
        per_key[key]={"accuracy":correct/max(1,len(rows)),"correct":correct}
    exact=sum(all(preds[i].get(k)==targets[i].get(k) for k in ORDER) for i in range(len(rows)))
    report={"config":asdict(cfg),"rows":len(rows),"exact_match":exact/max(1,len(rows)),"per_key":per_key}
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    if cfg.predictions_jsonl:
        Path(cfg.predictions_jsonl).write_text("\n".join(json.dumps({"target":targets[i],"prediction":preds[i],"scores":raw[i]},ensure_ascii=False) for i in range(len(rows)))+"\n")
    return report


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-jsonl",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--predictions-jsonl",default="")
    p.add_argument("--model-name",default=RECOMMENDED_VLM_MODEL)
    p.add_argument("--adapter-dir",default="")
    p.add_argument("--max-samples",type=int,default=0)
    p.add_argument("--batch-size",type=int,default=16)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(RewardComponentLogprobCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
