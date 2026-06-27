"""Evaluate reward-component SFT adapters with JSON generation."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from models.option_b_vlm import RECOMMENDED_VLM_MODEL, resolve_vlm_model_alias
from utils import disable_transformers_allocator_warmup

KEYS = ("net_bucket", "mae_bucket", "mfe_bucket", "mfe_to_mae_bucket", "utility_bucket", "path_shape")


@dataclass(frozen=True)
class EvalRewardComponentCfg:
    eval_jsonl: str
    output: str
    predictions_jsonl: str = ""
    train_jsonl: str = ""
    mode: str = "model"
    model_name: str = RECOMMENDED_VLM_MODEL
    adapter_dir: str = ""
    max_samples: int = 0
    max_new_tokens: int = 96


def _load(path: str, max_samples: int = 0) -> list[dict[str, Any]]:
    rows=[]
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if max_samples and len(rows) >= int(max_samples):
                    break
    return rows


def _parse_json(text: str) -> dict[str, Any]:
    s = str(text).strip()
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    m = re.search(r"\{.*\}", s, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


def _target(row: dict[str, Any]) -> dict[str, Any]:
    return _parse_json(str(row.get("target", "")))


def _majority(train_rows: list[dict[str, Any]]) -> dict[str, str]:
    counts = {k: Counter() for k in KEYS}
    for r in train_rows:
        t = _target(r)
        for k in KEYS:
            counts[k][str(t.get(k, "UNKNOWN"))] += 1
    return {k: (counts[k].most_common(1)[0][0] if counts[k] else "UNKNOWN") for k in KEYS}


def _chat(tok: Any, prompt: str) -> str:
    msgs=[{"role":"user","content":prompt}]
    if getattr(tok,"chat_template",None):
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return f"<|user|>\n{prompt}\n<|assistant|>\n"


def _load_model(model_name: str, adapter_dir: str):
    disable_transformers_allocator_warmup()
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    resolved = resolve_vlm_model_alias(model_name, prefer_latest=True)
    tok = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(resolved, trust_remote_code=True, device_map="auto"), adapter_dir)
    model.eval()
    return tok, model


def _model_preds(rows: list[dict[str, Any]], cfg: EvalRewardComponentCfg) -> tuple[list[dict[str, Any]], list[str]]:
    import torch
    tok, model = _load_model(cfg.model_name, cfg.adapter_dir)
    preds=[]; raw=[]
    for r in rows:
        inp = tok(_chat(tok, str(r["prompt"])), return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inp,
                max_new_tokens=int(cfg.max_new_tokens),
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        gen = tok.decode(out[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True)
        raw.append(gen)
        preds.append(_parse_json(gen))
    return preds, raw


def _score(rows: list[dict[str, Any]], preds: list[dict[str, Any]]) -> dict[str, Any]:
    targets = [_target(r) for r in rows]
    per_key = {}
    for k in KEYS:
        correct = sum(str(p.get(k, "")) == str(t.get(k, "")) for p, t in zip(preds, targets))
        per_key[k] = {"accuracy": correct / max(1, len(rows)), "correct": correct}
    exact = sum(all(str(p.get(k, "")) == str(t.get(k, "")) for k in KEYS) for p, t in zip(preds, targets))
    parsed = sum(1 for p in preds if any(k in p for k in KEYS))
    return {"rows": len(rows), "json_parse_rate": parsed / max(1, len(rows)), "exact_match": exact / max(1, len(rows)), "per_key": per_key}


def run(cfg: EvalRewardComponentCfg) -> dict[str, Any]:
    rows = _load(cfg.eval_jsonl, int(cfg.max_samples))
    raw=[]
    if cfg.mode == "majority":
        if not cfg.train_jsonl:
            raise ValueError("--train-jsonl is required for majority mode")
        maj = _majority(_load(cfg.train_jsonl, 0))
        preds = [dict(maj) for _ in rows]
        raw = [json.dumps(maj, sort_keys=True) for _ in rows]
    elif cfg.mode == "model":
        preds, raw = _model_preds(rows, cfg)
    else:
        raise ValueError(cfg.mode)
    report = {"config": asdict(cfg), **_score(rows, preds)}
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    if cfg.predictions_jsonl:
        Path(cfg.predictions_jsonl).write_text("\n".join(json.dumps({"target": _target(r), "prediction": p, "raw": raw[i]}, ensure_ascii=False) for i, (r, p) in enumerate(zip(rows, preds))) + "\n")
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--predictions-jsonl", default="")
    p.add_argument("--train-jsonl", default="")
    p.add_argument("--mode", choices=["majority", "model"], default="model")
    p.add_argument("--model-name", default=RECOMMENDED_VLM_MODEL)
    p.add_argument("--adapter-dir", default="")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=96)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(EvalRewardComponentCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
