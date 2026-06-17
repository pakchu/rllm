"""Evaluate pairwise choice rows with baselines or Gemma LoRA generation."""
from __future__ import annotations

import argparse, json, re
from pathlib import Path
from typing import Any

from models.option_b_vlm import RECOMMENDED_VLM_MODEL, resolve_vlm_model_alias
from utils import disable_transformers_allocator_warmup


def load_jsonl(path: str) -> list[dict[str, Any]]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def parse_choice(text: str) -> str:
    try:
        obj=json.loads(str(text).strip())
        val=str(obj.get('choice','')).upper()
        if val in {'A','B'}: return val
    except Exception:
        pass
    m=re.search(r'"choice"\s*:\s*"?([AB])"?', str(text), re.I)
    if m: return m.group(1).upper()
    m=re.search(r'\b([AB])\b', str(text).upper())
    return m.group(1) if m else 'A'


def _load_model(model_name: str, adapter_dir: str):
    disable_transformers_allocator_warmup()
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    resolved=resolve_vlm_model_alias(model_name, prefer_latest=True)
    tok=AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    model=PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(resolved, trust_remote_code=True, device_map='auto'), adapter_dir)
    model.eval(); return tok,model


def chat(tok: Any, prompt: str) -> str:
    msgs=[{'role':'user','content':prompt}]
    if getattr(tok,'chat_template',None): return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return f"<|user|>\n{prompt}\n<|assistant|>\n"


def eval_rows(rows, mode, model_name, adapter_dir, max_new_tokens):
    raw=[]; preds=[]
    if mode in {'always_a','always_b'}:
        preds=['A' if mode=='always_a' else 'B' for _ in rows]; raw=['' for _ in rows]
    elif mode=='model':
        import torch
        tok,model=_load_model(model_name,adapter_dir)
        for r in rows:
            inp=tok(chat(tok,r['prompt']), return_tensors='pt').to(model.device)
            with torch.no_grad(): out=model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
            gen=tok.decode(out[0][inp['input_ids'].shape[-1]:], skip_special_tokens=True)
            raw.append(gen); preds.append(parse_choice(gen))
    else: raise ValueError(mode)
    targets=[parse_choice(r['target']) for r in rows]
    ok=[p==t for p,t in zip(preds,targets)]
    report={'rows':len(rows),'mode':mode,'accuracy':sum(ok)/max(1,len(ok)),'correct':sum(ok),'prediction_counts':{x:preds.count(x) for x in ['A','B']}}
    return report,preds,raw


def main():
    p=argparse.ArgumentParser(); p.add_argument('--eval-jsonl',required=True); p.add_argument('--output',required=True); p.add_argument('--predictions-jsonl',default='')
    p.add_argument('--mode',choices=['always_a','always_b','model'],default='always_a'); p.add_argument('--model-name',default=RECOMMENDED_VLM_MODEL); p.add_argument('--adapter-dir',default=''); p.add_argument('--max-new-tokens',type=int,default=32)
    args=p.parse_args(); rows=load_jsonl(args.eval_jsonl); rep,preds,raw=eval_rows(rows,args.mode,args.model_name,args.adapter_dir,args.max_new_tokens)
    Path(args.output).write_text(json.dumps(rep,indent=2,ensure_ascii=False))
    if args.predictions_jsonl: Path(args.predictions_jsonl).write_text('\n'.join(json.dumps({'target':parse_choice(r['target']),'prediction':p,'raw':raw[i]},ensure_ascii=False) for i,(r,p) in enumerate(zip(rows,preds)))+'\n')
    print(json.dumps(rep,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
