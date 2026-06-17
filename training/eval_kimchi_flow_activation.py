"""Evaluate Kimchi-flow activation JSON policy rows."""
from __future__ import annotations

import argparse, json, re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from models.option_b_vlm import RECOMMENDED_VLM_MODEL, resolve_vlm_model_alias
from utils import disable_transformers_allocator_warmup

VALID = {
    "regime": {"KIMCHI_FLOW"},
    "decision": {"ACTIVATE", "ABSTAIN"},
    "side": {"LONG", "SHORT", "NONE"},
    "quality": {"GOOD", "MARGINAL", "BAD"},
    "confidence": {"LOW", "MID", "HIGH"},
}
DEFAULT = {"regime":"KIMCHI_FLOW","decision":"ABSTAIN","side":"NONE","quality":"MARGINAL","confidence":"LOW"}

CANDIDATE_JSON = [
    '{"confidence":"HIGH","decision":"ACTIVATE","quality":"GOOD","regime":"KIMCHI_FLOW","side":"LONG"}',
    '{"confidence":"HIGH","decision":"ACTIVATE","quality":"GOOD","regime":"KIMCHI_FLOW","side":"SHORT"}',
    '{"confidence":"HIGH","decision":"ABSTAIN","quality":"BAD","regime":"KIMCHI_FLOW","side":"NONE"}',
    '{"confidence":"LOW","decision":"ABSTAIN","quality":"MARGINAL","regime":"KIMCHI_FLOW","side":"NONE"}',
]


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


def _load_text_model(model_name: str, adapter_dir: str):
    disable_transformers_allocator_warmup()
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved = resolve_vlm_model_alias(model_name, prefer_latest=True)
    tokenizer = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    base = AutoModelForCausalLM.from_pretrained(resolved, trust_remote_code=True, device_map="auto")
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return tokenizer, model


def _chat_prompt_text(tokenizer: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"<|user|>\n{prompt}\n<|assistant|>\n"


def _generate_predictions(rows: list[dict[str, Any]], *, model_name: str, adapter_dir: str, max_new_tokens: int) -> tuple[list[dict[str, str]], list[str]]:
    import torch

    if not adapter_dir:
        raise ValueError("--adapter-dir is required for prediction_mode=model")
    tokenizer, model = _load_text_model(model_name, adapter_dir)
    preds: list[dict[str, str]] = []
    raw_outputs: list[str] = []
    for row in rows:
        prompt_text = _chat_prompt_text(tokenizer, str(row["prompt"]))
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        raw_outputs.append(generated)
        preds.append(parse_activation_json(generated))
    return preds, raw_outputs



def _score_candidate_json(rows: list[dict[str, Any]], *, model_name: str, adapter_dir: str) -> tuple[list[dict[str, str]], list[str]]:
    import torch

    if not adapter_dir:
        raise ValueError("--adapter-dir is required for prediction_mode=candidate_score")
    tokenizer, model = _load_text_model(model_name, adapter_dir)
    preds: list[dict[str, str]] = []
    raw_outputs: list[str] = []
    for row in rows:
        prompt_text = _chat_prompt_text(tokenizer, str(row["prompt"]))
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        best_text = CANDIDATE_JSON[0]
        best_score = float("-inf")
        scored: list[tuple[str, float]] = []
        for cand in CANDIDATE_JSON:
            cand_ids = tokenizer(cand, add_special_tokens=False)["input_ids"]
            input_ids = torch.tensor([prompt_ids + cand_ids], device=model.device)
            attention_mask = torch.ones_like(input_ids)
            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            start = len(prompt_ids)
            end = start + len(cand_ids)
            token_positions = torch.arange(start - 1, end - 1, device=logits.device)
            labels = input_ids[0, start:end]
            selected = logits[0, token_positions, :].float()
            label_logits = selected.gather(1, labels.reshape(-1, 1)).squeeze(1)
            score = float((label_logits - torch.logsumexp(selected, dim=-1)).mean().detach().cpu())
            scored.append((cand, score))
            if score > best_score:
                best_score = score
                best_text = cand
        preds.append(parse_activation_json(best_text))
        raw_outputs.append(json.dumps({"choice": best_text, "scores": scored}, ensure_ascii=False))
    return preds, raw_outputs

def evaluate(eval_jsonl: str, output: str, prediction_mode: str='target_echo', predictions_jsonl: str='', model_name: str = RECOMMENDED_VLM_MODEL, adapter_dir: str = '', max_new_tokens: int = 80) -> dict[str,Any]:
    rows=load_jsonl(eval_jsonl)
    raw_outputs: list[str] = []
    if prediction_mode=='target_echo': preds=[parse_activation_json(r['target']) for r in rows]
    elif prediction_mode=='all_abstain': preds=[dict(DEFAULT) for _ in rows]
    elif prediction_mode=='all_activate_long': preds=[{"regime":"KIMCHI_FLOW","decision":"ACTIVATE","side":"LONG","quality":"GOOD","confidence":"HIGH"} for _ in rows]
    elif prediction_mode=='model': preds, raw_outputs = _generate_predictions(rows, model_name=model_name, adapter_dir=adapter_dir, max_new_tokens=max_new_tokens)
    elif prediction_mode=='candidate_score': preds, raw_outputs = _score_candidate_json(rows, model_name=model_name, adapter_dir=adapter_dir)
    else: raise ValueError('unsupported prediction_mode')
    report={"as_of":datetime.now(timezone.utc).isoformat(),"eval_jsonl":eval_jsonl,"prediction_mode":prediction_mode,"model_name":resolve_vlm_model_alias(model_name, prefer_latest=True) if prediction_mode in {'model','candidate_score'} else '',"adapter_dir":adapter_dir if prediction_mode in {'model','candidate_score'} else '',"metrics":metrics(rows,preds),"leakage_guard":{"target_echo_for_pipeline_only":prediction_mode=='target_echo',"model_uses_targets":False if prediction_mode in {'model','candidate_score'} else None}}
    Path(output).parent.mkdir(parents=True, exist_ok=True); Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    if predictions_jsonl:
        Path(predictions_jsonl).write_text('\n'.join(json.dumps({"date":r['date'],"target":parse_activation_json(r['target']),"prediction":p,"raw_output":raw_outputs[i] if i < len(raw_outputs) else '',"trade_ret_pct":r.get('trade_ret_pct')}, ensure_ascii=False, sort_keys=True) for i,(r,p) in enumerate(zip(rows,preds)))+'\n')
    return report


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--eval-jsonl',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--prediction-mode',choices=['target_echo','all_abstain','all_activate_long','model','candidate_score'],default='target_echo')
    p.add_argument('--predictions-jsonl',default='')
    p.add_argument('--model-name', default=RECOMMENDED_VLM_MODEL)
    p.add_argument('--adapter-dir', default='')
    p.add_argument('--max-new-tokens', type=int, default=80)
    return p.parse_args()

if __name__=='__main__': print(json.dumps(evaluate(**vars(parse_args())), indent=2, ensure_ascii=False))
