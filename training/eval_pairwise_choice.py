"""Evaluate pairwise choice rows with baselines or Gemma LoRA scoring/generation."""
from __future__ import annotations

import argparse, json, random, re
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


def _sequence_logprob_batch(
    tok: Any,
    model: Any,
    prompts: list[str],
    completions: list[str],
    batch_size: int,
) -> list[float]:
    """Return total log P(completion | prompt) using teacher forcing.

    The pairwise ranker has fixed valid JSON completions. Scoring them in
    batches avoids thousands of single-row forward passes during PoC eval.
    """
    import torch

    scores: list[float] = []
    prefixes = [chat(tok, prompt) for prompt in prompts]
    prefix_lens = [
        tok(prefix, return_tensors="pt", add_special_tokens=True)["input_ids"].shape[-1]
        for prefix in prefixes
    ]
    full_texts = [prefix + completion for prefix, completion in zip(prefixes, completions)]
    for start in range(0, len(full_texts), batch_size):
        batch_texts = full_texts[start : start + batch_size]
        batch_prefix_lens = prefix_lens[start : start + batch_size]
        enc = tok(batch_texts, return_tensors="pt", padding=True).to(model.device)
        lengths = enc["attention_mask"].sum(dim=1).detach().cpu().tolist()
        with torch.no_grad():
            logits = model(**enc).logits[:, :-1, :]
        target_ids = enc["input_ids"][:, 1:]
        logp = torch.log_softmax(logits, dim=-1)
        token_logp = logp.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        for row_idx, (prefix_len, length) in enumerate(zip(batch_prefix_lens, lengths)):
            completion_start = max(0, int(prefix_len) - 1)
            completion_end = max(completion_start, int(length) - 1)
            scores.append(float(token_logp[row_idx, completion_start:completion_end].sum().detach().cpu()))
    return scores


def _choice_token_scores(
    tok: Any,
    model: Any,
    prompts: list[str],
    batch_size: int,
) -> tuple[list[float], list[float]]:
    """Score only the A/B token after the JSON choice prefix.

    This is the fastest deterministic pairwise-choice probe. The model was
    trained to emit JSON; for ranking quality we only need whether it assigns
    higher probability to A or B at the decision token.
    """
    import torch

    a_ids = tok("A", add_special_tokens=False)["input_ids"]
    b_ids = tok("B", add_special_tokens=False)["input_ids"]
    if len(a_ids) != 1 or len(b_ids) != 1:
        raise ValueError(f"A/B must tokenize to one token, got A={a_ids}, B={b_ids}")
    prefixes = [chat(tok, prompt) + '{"choice":"' for prompt in prompts]
    score_a: list[float] = []
    score_b: list[float] = []
    for start in range(0, len(prefixes), batch_size):
        enc = tok(prefixes[start : start + batch_size], return_tensors="pt", padding=True).to(model.device)
        lengths = enc["attention_mask"].sum(dim=1) - 1
        with torch.no_grad():
            logits = model(**enc).logits
        logp = torch.log_softmax(logits, dim=-1)
        for row_idx, pos in enumerate(lengths.detach().cpu().tolist()):
            row_logp = logp[row_idx, int(pos), :]
            score_a.append(float(row_logp[a_ids[0]].detach().cpu()))
            score_b.append(float(row_logp[b_ids[0]].detach().cpu()))
    return score_a, score_b


def eval_rows(rows, mode, model_name, adapter_dir, max_new_tokens, batch_size):
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
    elif mode=='model_logprob':
        tok,model=_load_model(model_name,adapter_dir)
        cand_a='{"choice":"A","confidence":"HIGH","reason":"higher_future_path_utility"}'
        cand_b='{"choice":"B","confidence":"HIGH","reason":"higher_future_path_utility"}'
        prompts=[]; completions=[]
        for r in rows:
            prompts.extend([r['prompt'], r['prompt']])
            completions.extend([cand_a, cand_b])
        scores=_sequence_logprob_batch(tok,model,prompts,completions,batch_size)
        for i in range(0, len(scores), 2):
            score_a=scores[i]
            score_b=scores[i+1]
            pred='A' if score_a>=score_b else 'B'
            raw.append(json.dumps({'score_a':score_a,'score_b':score_b},ensure_ascii=False))
            preds.append(pred)
    elif mode=='model_choice_token':
        tok,model=_load_model(model_name,adapter_dir)
        scores_a,scores_b=_choice_token_scores(tok,model,[r['prompt'] for r in rows],batch_size)
        for score_a,score_b in zip(scores_a,scores_b):
            pred='A' if score_a>=score_b else 'B'
            raw.append(json.dumps({'score_a':score_a,'score_b':score_b},ensure_ascii=False))
            preds.append(pred)
    else: raise ValueError(mode)
    targets=[parse_choice(r['target']) for r in rows]
    ok=[p==t for p,t in zip(preds,targets)]
    report={'rows':len(rows),'mode':mode,'accuracy':sum(ok)/max(1,len(ok)),'correct':sum(ok),'prediction_counts':{x:preds.count(x) for x in ['A','B']}}
    return report,preds,raw


def main():
    p=argparse.ArgumentParser(); p.add_argument('--eval-jsonl',required=True); p.add_argument('--output',required=True); p.add_argument('--predictions-jsonl',default='')
    p.add_argument('--mode',choices=['always_a','always_b','model','model_logprob','model_choice_token'],default='always_a'); p.add_argument('--model-name',default=RECOMMENDED_VLM_MODEL); p.add_argument('--adapter-dir',default=''); p.add_argument('--max-new-tokens',type=int,default=32)
    p.add_argument('--max-samples',type=int,default=0,help='Evaluate only N rows; 0 means all rows.')
    p.add_argument('--sample-mode',choices=['sequential','random','balanced'],default='sequential')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--batch-size',type=int,default=8,help='Batch size for model_logprob candidate scoring.')
    args=p.parse_args(); rows=load_jsonl(args.eval_jsonl)
    if args.max_samples and args.max_samples > 0 and args.max_samples < len(rows):
        rng=random.Random(int(args.seed))
        if args.sample_mode == 'sequential':
            rows=rows[:args.max_samples]
        elif args.sample_mode == 'random':
            idx=sorted(rng.sample(range(len(rows)), int(args.max_samples)))
            rows=[rows[i] for i in idx]
        else:
            buckets={'A':[],'B':[]}
            for i,r in enumerate(rows): buckets.setdefault(parse_choice(r.get('target','')),[]).append(i)
            selected=[]; per=max(1,int(args.max_samples)//2)
            for key in ['A','B']:
                vals=list(buckets.get(key,[])); rng.shuffle(vals); selected.extend(vals[:per])
            if len(selected)<int(args.max_samples):
                used=set(selected); rest=[i for i in range(len(rows)) if i not in used]; rng.shuffle(rest); selected.extend(rest[:int(args.max_samples)-len(selected)])
            rows=[rows[i] for i in sorted(selected[:int(args.max_samples)])]
    rep,preds,raw=eval_rows(rows,args.mode,args.model_name,args.adapter_dir,args.max_new_tokens,args.batch_size)
    Path(args.output).write_text(json.dumps(rep,indent=2,ensure_ascii=False))
    if args.predictions_jsonl: Path(args.predictions_jsonl).write_text('\n'.join(json.dumps({'target':parse_choice(r['target']),'prediction':p,'raw':raw[i]},ensure_ascii=False) for i,(r,p) in enumerate(zip(rows,preds)))+'\n')
    print(json.dumps(rep,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
