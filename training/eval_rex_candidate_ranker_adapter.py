"""Evaluate a TAKE/SKIP REX candidate ranker adapter with no eval selection.

Scores candidate rows by label logprob, selects a margin threshold on validation
rows only, then reports a strict backtest on eval rows.  This keeps the Gemma
adapter comparable to the ridge sanity floor.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from training.online_risk_overlay_backtest import OnlineRiskOverlayConfig, run_overlay
from training.train_text_sft import load_jsonl, resolve_vlm_model_alias
from utils import disable_transformers_allocator_warmup


@dataclass(frozen=True)
class RexAdapterEvalCfg:
    validation_jsonl: str
    eval_jsonl: str
    market_csv: str
    adapter_dir: str
    output: str
    predictions_dir: str
    model_name: str = "gemma4-e4b"
    validation_start: str = "2025-01-01"
    validation_end: str = "2026-01-01"
    margin_grid: str = "-1.0,-0.5,0.0,0.25,0.5,0.75,1.0"
    max_validation_rows: int = 0
    max_eval_rows: int = 0
    max_seq_length: int = 1024
    eval_batch_size: int = 8
    leverage: float = 0.5
    entry_delay_bars: int = 1
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001


def _in_period(row: dict[str, Any], start: str, end: str) -> bool:
    d = str(row.get("date", ""))
    return str(start) <= d < str(end)


def _load_rows(path: str, *, max_rows: int = 0) -> list[dict[str, Any]]:
    rows = load_jsonl(path, max_samples=0)
    return rows[: int(max_rows)] if int(max_rows) > 0 else rows


def _chat_prompt_text(tokenizer: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) if getattr(tokenizer, "chat_template", None) else f"<|user|>\n{prompt}\n<|assistant|>\n"


def _load_model(cfg: RexAdapterEvalCfg):
    disable_transformers_allocator_warmup()
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved = resolve_vlm_model_alias(cfg.model_name, prefer_latest=True)
    tokenizer = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(resolved, trust_remote_code=True, device_map="auto")
    model = PeftModel.from_pretrained(base, cfg.adapter_dir)
    if hasattr(model, "config"):
        model.config.use_cache = False
    model.eval()
    return resolved, tokenizer, model


def _label_logprobs(rows: list[dict[str, Any]], tokenizer: Any, model: Any, max_seq_length: int, batch_size: int) -> list[dict[str, Any]]:
    labels = ["SKIP", "TAKE"]
    out: list[dict[str, Any]] = []
    bs = max(1, int(batch_size))
    for offset in range(0, len(rows), bs):
        chunk = rows[offset : offset + bs]
        seqs: list[list[int]] = []
        spans: list[tuple[int, int]] = []
        owners: list[int] = []
        for owner, row in enumerate(chunk):
            prompt_ids = tokenizer(
                _chat_prompt_text(tokenizer, str(row["prompt"])),
                add_special_tokens=False,
                truncation=True,
                max_length=int(max_seq_length),
            )["input_ids"]
            for label in labels:
                label_ids = tokenizer(label, add_special_tokens=False)["input_ids"]
                if tokenizer.eos_token_id is not None:
                    label_ids = label_ids + [int(tokenizer.eos_token_id)]
                start = len(prompt_ids)
                end = start + len(label_ids)
                seqs.append(prompt_ids + label_ids)
                spans.append((start, end))
                owners.append(owner)
        encoded = tokenizer.pad({"input_ids": seqs}, return_tensors="pt")
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
            log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        chunk_scores = [{"SKIP": 0.0, "TAKE": 0.0} for _ in chunk]
        for i, (start, end) in enumerate(spans):
            positions = torch.arange(start - 1, end - 1, device=log_probs.device)
            label_tensor = input_ids[i, start:end]
            token_scores = log_probs[i, positions, label_tensor]
            chunk_scores[owners[i]][labels[i % 2]] = float(token_scores.mean().detach().cpu())
        for row, scores in zip(chunk, chunk_scores):
            margin = scores["TAKE"] - scores["SKIP"]
            out.append({"row": row, "scores": scores, "margin": margin, "prediction": "TAKE" if margin >= 0.0 else "SKIP"})
        print(json.dumps({"scored": min(offset + len(chunk), len(rows)), "total": len(rows)}, ensure_ascii=False), flush=True)
    return out

def _accuracy(scored: list[dict[str, Any]]) -> dict[str, Any]:
    confusion: dict[str, int] = {}
    correct = 0
    for item in scored:
        target = str(item["row"].get("target", "SKIP")).upper()
        pred = str(item["prediction"])
        correct += int(target == pred)
        confusion[f"target={target}|pred={pred}"] = confusion.get(f"target={target}|pred={pred}", 0) + 1
    return {"rows": len(scored), "accuracy": correct / max(1, len(scored)), "confusion": dict(sorted(confusion.items()))}


def _predictions(scored: list[dict[str, Any]], margin_threshold: float) -> list[dict[str, Any]]:
    best: dict[int, dict[str, Any]] = {}
    for item in scored:
        row = item["row"]
        pos = int(row.get("signal_pos", -1) or -1)
        if pos < 0:
            continue
        old = best.get(pos)
        if old is None or float(item["margin"]) > float(old["margin"]):
            best[pos] = item
    preds: list[dict[str, Any]] = []
    for pos, item in sorted(best.items()):
        row = item["row"]
        cand = row.get("candidate") if isinstance(row.get("candidate"), dict) else {}
        if float(item["margin"]) >= float(margin_threshold):
            pred = {"gate": "TRADE", "side": str(row.get("side", cand.get("side", "NONE"))).upper(), "hold_bars": int(cand.get("hold_bars", 288) or 288), "family": "rex_gemma_ranker"}
            scale = 1.0
        else:
            pred = {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0, "family": "rex_gemma_ranker"}
            scale = 0.0
        preds.append({"date": row.get("date"), "signal_pos": pos, "prediction": pred, "position_scale": scale, "score": float(item["margin"]), "side_candidate": row.get("side"), "target": row.get("target")})
    return preds


def _backtest(preds: list[dict[str, Any]], cfg: RexAdapterEvalCfg, name: str) -> dict[str, Any]:
    path = Path(cfg.predictions_dir) / f"{name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in preds) + ("\n" if preds else ""))
    with tempfile.TemporaryDirectory(prefix="rllm_rex_adapter_eval_") as tmp:
        bt = run_overlay(
            OnlineRiskOverlayConfig(
                predictions_jsonl=str(path),
                market_csv=cfg.market_csv,
                output=str(Path(tmp) / f"{name}_bt.json"),
                leverage=cfg.leverage,
                entry_delay_bars=cfg.entry_delay_bars,
                fee_rate=cfg.fee_rate,
                slippage_rate=cfg.slippage_rate,
            )
        )
    bt.pop("executed", None)
    return {"predictions": str(path), "sim": bt["sim"], "trade_stats": bt["trade_stats"]}


def _rank(bt: dict[str, Any]) -> float:
    sim = bt["sim"]
    trades = int(sim.get("trade_entries", 0) or 0)
    if trades < 10:
        return -1e9
    return float(sim.get("cagr_to_strict_mdd", 0.0) or 0.0) + 0.02 * float(sim.get("cagr_pct", 0.0) or 0.0)


def run(cfg: RexAdapterEvalCfg) -> dict[str, Any]:
    val_rows = [r for r in _load_rows(cfg.validation_jsonl, max_rows=cfg.max_validation_rows) if _in_period(r, cfg.validation_start, cfg.validation_end)]
    eval_rows = _load_rows(cfg.eval_jsonl, max_rows=cfg.max_eval_rows)
    resolved, tokenizer, model = _load_model(cfg)
    val_scored = _label_logprobs(val_rows, tokenizer, model, cfg.max_seq_length, cfg.eval_batch_size)
    eval_scored = _label_logprobs(eval_rows, tokenizer, model, cfg.max_seq_length, cfg.eval_batch_size)
    margins = [float(x) for x in str(cfg.margin_grid).split(",") if x.strip()]
    candidates: list[dict[str, Any]] = []
    for margin in margins:
        bt = _backtest(_predictions(val_scored, margin), cfg, f"validation_margin_{margin:g}")
        candidates.append({"margin": margin, "backtest": bt, "score": _rank(bt)})
    candidates.sort(key=lambda r: float(r["score"]), reverse=True)
    selected = candidates[0] if candidates else {"margin": 999.0, "score": -1e9}
    eval_bt = _backtest(_predictions(eval_scored, float(selected["margin"])), cfg, "selected_eval")
    report = {
        "config": asdict(cfg),
        "resolved_model": resolved,
        "rows": {"validation": len(val_rows), "eval": len(eval_rows)},
        "validation_accuracy": _accuracy(val_scored),
        "eval_accuracy": _accuracy(eval_scored),
        "selection_rule": "choose margin on validation rows only; eval is report-only",
        "top_validation_margins": candidates[:10],
        "selected": selected,
        "eval_backtest": eval_bt,
        "leakage_guard": {"margin_selected_on_validation_only": True, "eval_not_used_for_selection": True, "prompts_past_only": True},
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--validation-jsonl", required=True)
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--adapter-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--predictions-dir", required=True)
    p.add_argument("--model-name", default=RexAdapterEvalCfg.model_name)
    p.add_argument("--validation-start", default=RexAdapterEvalCfg.validation_start)
    p.add_argument("--validation-end", default=RexAdapterEvalCfg.validation_end)
    p.add_argument("--margin-grid", default=RexAdapterEvalCfg.margin_grid)
    p.add_argument("--max-validation-rows", type=int, default=RexAdapterEvalCfg.max_validation_rows)
    p.add_argument("--max-eval-rows", type=int, default=RexAdapterEvalCfg.max_eval_rows)
    p.add_argument("--max-seq-length", type=int, default=RexAdapterEvalCfg.max_seq_length)
    p.add_argument("--eval-batch-size", type=int, default=RexAdapterEvalCfg.eval_batch_size)
    p.add_argument("--leverage", type=float, default=RexAdapterEvalCfg.leverage)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(RexAdapterEvalCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
