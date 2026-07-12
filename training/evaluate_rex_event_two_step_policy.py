"""Evaluate single-LLM two-step REX event policy with candidate logprobs."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.build_rex_event_two_step_sft_data import _gate_prompt, _side_prompt, _utils, Cfg as BuildCfg
from training.event_candidate_pool_probe import EventPoolConfig, _load_market, _simulate_rows
from training.train_text_sft import load_jsonl, resolve_text_causal_lm_alias
from utils import disable_transformers_allocator_warmup

GATE_CANDS = ["TRADE", "NO_TRADE"]
SIDE_CANDS = ["LONG", "SHORT"]


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
    gate_calibration_split: str = "train"
    gate_bias: float = 0.0
    calibrate_gate_prior: bool = False
    hold_bars: int = 144
    entry_delay_bars: int = 1
    leverage: float = 0.5
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001


def _chat(tokenizer: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        return str(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    return f"<|user|>\n{prompt}\n<|assistant|>\n"


def _load(model_name: str, adapter_dir: str):
    disable_transformers_allocator_warmup()
    resolved = resolve_text_causal_lm_alias(model_name, prefer_latest=True)
    tok = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    base = AutoModelForCausalLM.from_pretrained(resolved, trust_remote_code=True, device_map="auto")
    model = PeftModel.from_pretrained(base, adapter_dir) if adapter_dir else base
    model.eval()
    return tok, model, resolved


def _score_batch(model: Any, input_ids: Any, attention_mask: Any, spans: list[tuple[int, int]], normalize: str) -> list[float]:
    import torch
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    scores: list[float] = []
    for i, (start, end) in enumerate(spans):
        pos = torch.arange(start - 1, end - 1, device=logits.device)
        labels = input_ids[i, start:end]
        selected = logits[i, pos, :].float()
        label_logits = selected.gather(1, labels.reshape(-1, 1)).squeeze(1)
        ts = label_logits - torch.logsumexp(selected, dim=-1)
        scores.append(float((ts.sum() if normalize == "sum" else ts.mean()).detach().cpu()))
    return scores


def _candidate_scores(rows: list[dict[str, Any]], cfg: Cfg, prompts: list[str], cands: list[str], tok: Any, model: Any) -> list[dict[str, float]]:
    cand_ids = []
    for c in cands:
        ids = tok(c, add_special_tokens=False)["input_ids"]
        if tok.eos_token_id is not None:
            ids = ids + [int(tok.eos_token_id)]
        cand_ids.append(ids)
    out: list[dict[str, float]] = []
    bs = max(1, int(cfg.batch_size))
    for off in range(0, len(rows), bs):
        batch_prompts = prompts[off:off + bs]
        seqs = []
        spans = []
        for prompt in batch_prompts:
            pids = tok(_chat(tok, prompt), add_special_tokens=False)["input_ids"]
            st = len(pids)
            for ids in cand_ids:
                seqs.append(pids + ids)
                spans.append((st, st + len(ids)))
        enc = tok.pad({"input_ids": seqs}, return_tensors="pt")
        scores = _score_batch(model, enc["input_ids"].to(model.device), enc["attention_mask"].to(model.device), spans, cfg.score_normalization)
        k = 0
        for _ in batch_prompts:
            ss = scores[k:k + len(cands)]
            k += len(cands)
            out.append(dict(zip(cands, ss)))
    return out


def _target_labels(row: dict[str, Any]) -> tuple[str, str]:
    u = _utils(row, BuildCfg(input_jsonl="", output_jsonl=""))
    best_side = "LONG" if u["LONG"] >= u["SHORT"] else "SHORT"
    gate = "TRADE" if max(u["LONG"], u["SHORT"]) - u["NO_TRADE"] >= 0.004 else "NO_TRADE"
    return gate, best_side


def _split(date: str) -> str:
    import pandas as pd
    ts = pd.Timestamp(str(date))
    if ts < pd.Timestamp("2025-01-01"):
        return "train"
    if ts < pd.Timestamp("2026-01-01"):
        return "test"
    return "eval"


def _metrics(rows: list[dict[str, Any]], pred_gate: list[str], pred_side: list[str]) -> dict[str, Any]:
    gate_ok = side_ok = 0
    gc = Counter(); sc = Counter(); gtc = Counter(); stc = Counter(); conf = Counter()
    for r, pg, ps in zip(rows, pred_gate, pred_side):
        tg, ts = _target_labels(r)
        gate_ok += int(pg == tg)
        side_ok += int(ps == ts)
        gc[pg] += 1; sc[ps] += 1; gtc[tg] += 1; stc[ts] += 1
        conf[f"target_gate={tg}|pred_gate={pg}|target_side={ts}|pred_side={ps}"] += 1
    return {
        "rows": len(rows),
        "gate_accuracy": gate_ok / max(1, len(rows)),
        "side_accuracy": side_ok / max(1, len(rows)),
        "target_gate_counts": dict(gtc),
        "pred_gate_counts": dict(gc),
        "target_side_counts": dict(stc),
        "pred_side_counts": dict(sc),
        "confusion": dict(conf),
    }


def _backtest(rows: list[dict[str, Any]], pred_gate: list[str], pred_side: list[str], cfg: Cfg) -> dict[str, Any]:
    market = _load_market(cfg.market_csv)
    trades = []
    for r, g, s in zip(rows, pred_gate, pred_side):
        if g == "TRADE" and s in {"LONG", "SHORT"}:
            trades.append({"date": r["date"], "signal_date": r["date"], "side": s, "family": "rex_two_step_llm", "strength": 1.0, "score_mean": 1.0})
    ecfg = EventPoolConfig(input_csv=cfg.market_csv, output="", hold_bars=cfg.hold_bars, entry_delay_bars=cfg.entry_delay_bars, leverage=cfg.leverage, fee_rate=cfg.fee_rate, slippage_rate=cfg.slippage_rate)
    res = _simulate_rows(trades, market, ecfg)
    return {"predicted_trade_rows": len(trades), "sim": res.get("sim", {}), "trade_stats": res.get("trade_stats", {})}


def _prior(scores: list[dict[str, float]], rows: list[dict[str, Any]], split: str, cands: list[str]) -> dict[str, float]:
    part = [(r, s) for r, s in zip(rows, scores) if _split(str(r.get("date"))) == split]
    return {c: sum(float(s[c]) for _, s in part) / max(1, len(part)) for c in cands}


def run(cfg: Cfg) -> dict[str, Any]:
    rows = load_jsonl(cfg.eval_jsonl, max_samples=cfg.max_samples, sample_mode=cfg.sample_mode, seed=42)
    tok, model, resolved = _load(cfg.model_name, cfg.adapter_dir)
    gate_scores = _candidate_scores(rows, cfg, [_gate_prompt(r) for r in rows], GATE_CANDS, tok, model)
    side_scores = _candidate_scores(rows, cfg, [_side_prompt(r) for r in rows], SIDE_CANDS, tok, model)
    gate_prior = _prior(gate_scores, rows, cfg.gate_calibration_split, GATE_CANDS) if cfg.calibrate_gate_prior else {c: 0.0 for c in GATE_CANDS}
    pred_gate = []
    pred_side = []
    for gs, ss in zip(gate_scores, side_scores):
        pred_gate.append(max(GATE_CANDS, key=lambda c: float(gs[c]) - float(gate_prior.get(c, 0.0)) + (float(cfg.gate_bias) if c == "TRADE" else 0.0)))
        pred_side.append(max(SIDE_CANDS, key=lambda c: float(ss[c])))
    report = {"config": asdict(cfg), "model_name_resolved": resolved, "splits": {}, "gate_prior": gate_prior, "leakage_guard": {"model_sees_prompt_only": True, "calibration_uses_train_scores_only": bool(cfg.calibrate_gate_prior), "targets_used_for_metrics_only": True}, "score_rows": []}
    for r, gs, ss, pg, ps in zip(rows, gate_scores, side_scores, pred_gate, pred_side):
        tg, ts = _target_labels(r)
        report["score_rows"].append({"date": r.get("date"), "split": _split(str(r.get("date"))), "signal_pos": r.get("signal_pos"), "target_gate": tg, "target_side": ts, "pred_gate": pg, "pred_side": ps, "gate_scores": gs, "side_scores": ss})
    for sp in ["train", "test", "eval"]:
        idx = [i for i, r in enumerate(rows) if _split(str(r.get("date"))) == sp]
        part = [rows[i] for i in idx]
        pg = [pred_gate[i] for i in idx]
        ps = [pred_side[i] for i in idx]
        report["splits"][sp] = {"metrics": _metrics(part, pg, ps), "backtest": _backtest(part, pg, ps, cfg)}
    Path(cfg.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output_json).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return {k: v for k, v in report.items() if k != "score_rows"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--model-name", default=Cfg.model_name)
    p.add_argument("--adapter-dir", default="")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--sample-mode", default="sequential")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--score-normalization", choices=["mean", "sum"], default="mean")
    p.add_argument("--gate-calibration-split", default="train")
    p.add_argument("--gate-bias", type=float, default=0.0)
    p.add_argument("--calibrate-gate-prior", action="store_true")
    print(json.dumps(run(Cfg(**vars(p.parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
