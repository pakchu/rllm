"""Evaluate a Gemma/Gemma4 listwise REX choice adapter.

Scores every candidate id in each listwise prompt by label logprob, chooses the
highest-scoring id, converts it to a strict trade/no-trade prediction, and reports
validation-selected abstention margins plus untouched eval backtest.
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
class RexListwiseAdapterEvalCfg:
    validation_jsonl: str
    eval_jsonl: str
    market_csv: str
    adapter_dir: str
    output: str
    predictions_dir: str
    model_name: str = "gemma4-e4b"
    validation_start: str = "2025-01-01"
    validation_end: str = "2026-01-01"
    confidence_margin_grid: str = "-999,0,0.05,0.1,0.2,0.3,0.5"
    max_validation_rows: int = 0
    max_eval_rows: int = 0
    max_seq_length: int = 1024
    eval_batch_size: int = 4
    min_selection_trades: int = 10
    torch_dtype: str = "bfloat16"
    load_in_4bit: bool = False
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


def _load_model(cfg: RexListwiseAdapterEvalCfg):
    disable_transformers_allocator_warmup()
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    resolved = resolve_vlm_model_alias(cfg.model_name, prefer_latest=True)
    tokenizer = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization_config = None
    if cfg.load_in_4bit:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    dtype_name = str(cfg.torch_dtype or "").lower()
    torch_dtype = None if dtype_name in {"", "auto", "none"} else getattr(torch, dtype_name)
    base = AutoModelForCausalLM.from_pretrained(
        resolved,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
    )
    model = PeftModel.from_pretrained(base, cfg.adapter_dir)
    if hasattr(model, "config"):
        model.config.use_cache = False
    model.eval()
    return resolved, tokenizer, model


def _choice_logprobs(rows: list[dict[str, Any]], tokenizer: Any, model: Any, max_seq_length: int, batch_size: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    bs = max(1, int(batch_size))
    for offset in range(0, len(rows), bs):
        chunk = rows[offset : offset + bs]
        seqs: list[list[int]] = []
        spans: list[tuple[int, int]] = []
        owners: list[int] = []
        labels_flat: list[str] = []
        for owner, row in enumerate(chunk):
            prompt_ids = tokenizer(
                _chat_prompt_text(tokenizer, str(row["prompt"])),
                add_special_tokens=False,
                truncation=True,
                max_length=int(max_seq_length),
            )["input_ids"]
            choices = [str(x) for x in row.get("choices", [])]
            if not choices:
                choices = [str(row.get("target", "NO_TRADE"))]
            for label in choices:
                label_ids = tokenizer(label, add_special_tokens=False)["input_ids"]
                if tokenizer.eos_token_id is not None:
                    label_ids = label_ids + [int(tokenizer.eos_token_id)]
                start = len(prompt_ids)
                end = start + len(label_ids)
                seqs.append(prompt_ids + label_ids)
                spans.append((start, end))
                owners.append(owner)
                labels_flat.append(label)
        old_padding_side = getattr(tokenizer, "padding_side", "right")
        tokenizer.padding_side = "left"
        try:
            encoded = tokenizer.pad({"input_ids": seqs}, return_tensors="pt")
        finally:
            tokenizer.padding_side = old_padding_side
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device)
        label_lens = [end - start for start, end in spans]
        keep = max(label_lens) + 1
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, logits_to_keep=keep).logits
            log_probs = torch.log_softmax(logits, dim=-1)
        chunk_scores: list[dict[str, float]] = [dict() for _ in chunk]
        seq_len = int(input_ids.shape[1])
        for i, label_len in enumerate(label_lens):
            # With left padding, every label is right-aligned at the end of the
            # sequence.  For next-token scoring we need logits at positions
            # seq_len-label_len-1 .. seq_len-2, which are inside the kept tail.
            rel_start = keep - label_len - 1
            rel_end = rel_start + label_len
            label_tensor = input_ids[i, seq_len - label_len : seq_len]
            token_scores = log_probs[i, rel_start:rel_end, :].gather(1, label_tensor.unsqueeze(1)).squeeze(1)
            chunk_scores[owners[i]][labels_flat[i]] = float(token_scores.mean().detach().cpu())
        for row, scores in zip(chunk, chunk_scores):
            ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            pred = ranked[0][0] if ranked else "NO_TRADE"
            margin = (ranked[0][1] - ranked[1][1]) if len(ranked) > 1 else 999.0
            out.append({"row": row, "scores": scores, "prediction": pred, "margin": float(margin)})
        print(json.dumps({"scored": min(offset + len(chunk), len(rows)), "total": len(rows)}, ensure_ascii=False), flush=True)
    return out


def _accuracy(scored: list[dict[str, Any]]) -> dict[str, Any]:
    confusion: dict[str, int] = {}
    correct = 0
    for item in scored:
        target = str(item["row"].get("target", "NO_TRADE"))
        pred = str(item["prediction"])
        correct += int(target == pred)
        confusion[f"target={target}|pred={pred}"] = confusion.get(f"target={target}|pred={pred}", 0) + 1
    return {"rows": len(scored), "accuracy": correct / max(1, len(scored)), "confusion": dict(sorted(confusion.items()))}


def _resolve_choice_id(row: dict[str, Any], choice: str) -> str:
    cmap = row.get("choice_map") if isinstance(row.get("choice_map"), dict) else {}
    return str(cmap.get(str(choice), choice))


def _id_to_prediction(choice_id: str) -> tuple[dict[str, Any], float, str]:
    cid = str(choice_id).upper()
    if cid == "NO_TRADE" or cid.endswith("_NONE"):
        return {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0, "family": "rex_gemma_listwise"}, 0.0, "NONE"
    side = "SHORT" if cid.endswith("_SHORT") else "LONG" if cid.endswith("_LONG") else "NONE"
    if side not in {"LONG", "SHORT"}:
        return {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0, "family": "rex_gemma_listwise", "reason": "invalid_choice_id"}, 0.0, "NONE"
    return {"gate": "TRADE", "side": side, "hold_bars": 288, "family": "rex_gemma_listwise", "choice_id": cid}, 1.0, side


def _predictions(scored: list[dict[str, Any]], min_confidence_margin: float) -> list[dict[str, Any]]:
    preds: list[dict[str, Any]] = []
    for item in scored:
        row = item["row"]
        choice = str(item["prediction"])
        resolved_choice = _resolve_choice_id(row, choice)
        margin = float(item.get("margin", 0.0) or 0.0)
        pred, scale, side = _id_to_prediction(resolved_choice)
        if pred.get("gate") == "TRADE" and margin < float(min_confidence_margin):
            pred, scale, side = {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0, "family": "rex_gemma_listwise", "reason": "confidence_margin"}, 0.0, "NONE"
        preds.append({"date": row.get("date"), "signal_pos": int(row.get("signal_pos", -1) or -1), "prediction": pred, "position_scale": scale, "score": margin, "side_candidate": side, "target": row.get("target"), "choice_prediction": choice, "resolved_choice_id": resolved_choice})
    return preds


def _backtest(preds: list[dict[str, Any]], cfg: RexListwiseAdapterEvalCfg, name: str) -> dict[str, Any]:
    path = Path(cfg.predictions_dir) / f"{name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in preds) + ("\n" if preds else ""))
    with tempfile.TemporaryDirectory(prefix="rllm_rex_listwise_eval_") as tmp:
        bt = run_overlay(OnlineRiskOverlayConfig(
            predictions_jsonl=str(path),
            market_csv=cfg.market_csv,
            output=str(Path(tmp) / f"{name}_bt.json"),
            leverage=cfg.leverage,
            entry_delay_bars=cfg.entry_delay_bars,
            fee_rate=cfg.fee_rate,
            slippage_rate=cfg.slippage_rate,
        ))
    bt.pop("executed", None)
    return {"predictions": str(path), "sim": bt["sim"], "trade_stats": bt["trade_stats"]}


def _rank(bt: dict[str, Any], *, min_trades: int) -> float:
    sim = bt["sim"]
    trades = int(sim.get("trade_entries", 0) or 0)
    if trades < int(min_trades):
        return -1e9
    return float(sim.get("cagr_to_strict_mdd", 0.0) or 0.0) + 0.02 * float(sim.get("cagr_pct", 0.0) or 0.0)


def run(cfg: RexListwiseAdapterEvalCfg) -> dict[str, Any]:
    val_rows = [r for r in _load_rows(cfg.validation_jsonl, max_rows=0) if _in_period(r, cfg.validation_start, cfg.validation_end)]
    if int(cfg.max_validation_rows) > 0:
        val_rows = val_rows[: int(cfg.max_validation_rows)]
    eval_rows = _load_rows(cfg.eval_jsonl, max_rows=cfg.max_eval_rows)
    resolved, tokenizer, model = _load_model(cfg)
    val_scored = _choice_logprobs(val_rows, tokenizer, model, cfg.max_seq_length, cfg.eval_batch_size)
    eval_scored = _choice_logprobs(eval_rows, tokenizer, model, cfg.max_seq_length, cfg.eval_batch_size)
    margins = [float(x) for x in str(cfg.confidence_margin_grid).split(",") if x.strip()]
    candidates: list[dict[str, Any]] = []
    for margin in margins:
        bt = _backtest(_predictions(val_scored, margin), cfg, f"validation_conf_margin_{margin:g}")
        candidates.append({"confidence_margin": margin, "backtest": bt, "score": _rank(bt, min_trades=cfg.min_selection_trades)})
    candidates.sort(key=lambda r: float(r["score"]), reverse=True)
    selected = candidates[0] if candidates else {"confidence_margin": 999.0, "score": -1e9}
    no_valid_selection = (not candidates) or float(selected.get("score", -1e9)) <= -1e8
    if no_valid_selection:
        selected = {"confidence_margin": 999.0, "score": -1e9, "selection_failed": True, "reason": "no_validation_margin_met_min_selection_trades", "min_selection_trades": int(cfg.min_selection_trades)}
    eval_bt = _backtest(_predictions(eval_scored, float(selected["confidence_margin"])), cfg, "selected_eval")
    report = {
        "config": asdict(cfg),
        "resolved_model": resolved,
        "rows": {"validation": len(val_rows), "eval": len(eval_rows)},
        "validation_accuracy": _accuracy(val_scored),
        "eval_accuracy": _accuracy(eval_scored),
        "selection_rule": "choose confidence margin on validation rows only; if no margin satisfies min_selection_trades, fall back to no-trade",
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
    p.add_argument("--model-name", default=RexListwiseAdapterEvalCfg.model_name)
    p.add_argument("--validation-start", default=RexListwiseAdapterEvalCfg.validation_start)
    p.add_argument("--validation-end", default=RexListwiseAdapterEvalCfg.validation_end)
    p.add_argument("--confidence-margin-grid", default=RexListwiseAdapterEvalCfg.confidence_margin_grid)
    p.add_argument("--max-validation-rows", type=int, default=RexListwiseAdapterEvalCfg.max_validation_rows)
    p.add_argument("--max-eval-rows", type=int, default=RexListwiseAdapterEvalCfg.max_eval_rows)
    p.add_argument("--max-seq-length", type=int, default=RexListwiseAdapterEvalCfg.max_seq_length)
    p.add_argument("--eval-batch-size", type=int, default=RexListwiseAdapterEvalCfg.eval_batch_size)
    p.add_argument("--min-selection-trades", type=int, default=RexListwiseAdapterEvalCfg.min_selection_trades)
    p.add_argument("--torch-dtype", default=RexListwiseAdapterEvalCfg.torch_dtype)
    p.add_argument("--load-in-4bit", action="store_true", default=RexListwiseAdapterEvalCfg.load_in_4bit)
    p.add_argument("--leverage", type=float, default=RexListwiseAdapterEvalCfg.leverage)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(RexListwiseAdapterEvalCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
