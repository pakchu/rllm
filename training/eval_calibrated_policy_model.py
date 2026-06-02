"""Evaluate a text trader adapter as an executable calibrated policy.

The adapter receives analyzer summaries and emits JSON actions.  This evaluator
runs those generated actions over chronological eval records with the same strict
non-overlap/intra-trade-MAE accounting used for calibrated rule baselines.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from models.option_b_vlm import RECOMMENDED_VLM_MODEL, resolve_vlm_model_alias
from training.calibrated_regime_policy import CalibratedPolicyConfig, _metrics_from_trades, build_calibration_records, fit_rules
from training.export_calibrated_policy_labels import _policy_target_for_record, build_policy_trader_input, format_policy_book
from training.text_analyzer_trader_data import analyzer_summary_to_text, load_market_frame
from training.text_step_analyzer_data import parse_hold_candidates
from utils import disable_transformers_allocator_warmup


def parse_policy_json(text: str, *, allowed_holds: tuple[int, ...]) -> dict[str, Any]:
    raw = str(text).strip()
    obj: dict[str, Any]
    try:
        parsed = json.loads(raw)
        obj = parsed if isinstance(parsed, dict) else {}
    except Exception:
        match = re.search(r"\{.*?\}", raw, flags=re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                obj = parsed if isinstance(parsed, dict) else {}
            except Exception:
                obj = {}
        else:
            obj = {}
    gate = str(obj.get("gate", "NO_TRADE")).upper()
    side = str(obj.get("side", "NONE")).upper()
    try:
        hold_bars = int(obj.get("hold_bars", 0))
    except Exception:
        hold_bars = 0
    if gate != "TRADE":
        return {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0, "policy_key": str(obj.get("policy_key", "")), "reason": str(obj.get("reason", "PARSED_NO_TRADE"))}
    if side not in {"LONG", "SHORT"} or hold_bars not in set(int(x) for x in allowed_holds):
        return {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0, "policy_key": str(obj.get("policy_key", "")), "reason": "INVALID_TRADE_JSON"}
    return {
        "gate": "TRADE",
        "side": side,
        "hold_bars": hold_bars,
        "policy_key": str(obj.get("policy_key", "")),
        "reason": str(obj.get("reason", "MODEL_TRADE")),
    }


def _load_model(model_name: str, adapter_dir: str):
    disable_transformers_allocator_warmup()
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved = resolve_vlm_model_alias(model_name, prefer_latest=True)
    tokenizer = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(resolved, trust_remote_code=True, device_map="auto")
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return resolved, tokenizer, model


def _chat_text(tokenizer: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        return str(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    return f"<|user|>\n{prompt}\n<|assistant|>\n"


def _generate_actions_batched(
    tokenizer: Any,
    model: Any,
    prompts: list[str],
    *,
    max_new_tokens: int,
    allowed_holds: tuple[int, ...],
    batch_size: int,
) -> list[tuple[dict[str, Any], str]]:
    results: list[tuple[dict[str, Any], str]] = []
    bs = max(1, int(batch_size))
    tokenizer.padding_side = "left"
    for start in range(0, len(prompts), bs):
        chunk = prompts[start : start + bs]
        texts = [_chat_text(tokenizer, prompt) for prompt in chunk]
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        prompt_length = int(inputs["input_ids"].shape[1])
        out = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        for i in range(len(chunk)):
            generated = tokenizer.decode(out[i][prompt_length:], skip_special_tokens=True)
            results.append((parse_policy_json(generated, allowed_holds=allowed_holds), generated))
    return results


def _generate_action(tokenizer: Any, model: Any, prompt: str, *, max_new_tokens: int, allowed_holds: tuple[int, ...]) -> tuple[dict[str, Any], str]:
    text = _chat_text(tokenizer, prompt)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated = tokenizer.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
    return parse_policy_json(generated, allowed_holds=allowed_holds), generated


def _apply_rule_guard(row: dict[str, Any], action: dict[str, Any], rules: dict[str, dict[str, Any]], mode: str) -> dict[str, Any]:
    guard = str(mode).strip().lower()
    if guard in {"", "none"}:
        return action
    if str(action.get("gate", "NO_TRADE")).upper() != "TRADE":
        return action
    rule = rules.get(str(row["key"]))
    if not rule:
        rejected = dict(action)
        rejected.update({"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0, "reason": "RULE_GUARD_NO_CURRENT_KEY"})
        return rejected
    expected = rule["action"]
    if guard == "current_key_any":
        return action
    if guard != "current_key_action":
        raise ValueError("rule_guard must be one of {'none','current_key_any','current_key_action'}")
    if str(action.get("side", "")).upper() == str(expected["side"]).upper() and int(action.get("hold_bars", 0) or 0) == int(expected["hold_bars"]):
        return action
    rejected = dict(action)
    rejected.update({"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0, "reason": "RULE_GUARD_ACTION_MISMATCH"})
    return rejected


def _candidate_targets_for_row(row: dict[str, Any], rules: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    no_trade = {
        "gate": "NO_TRADE",
        "hold_bars": 0,
        "policy_key": str(row["key"]),
        "reason": "NO_CALIBRATED_EDGE",
        "side": "NONE",
    }
    rule = rules.get(str(row["key"]))
    if not rule:
        return [no_trade]
    action = rule["action"]
    trade = {
        "gate": "TRADE",
        "hold_bars": int(action["hold_bars"]),
        "policy_key": str(row["key"]),
        "reason": "CALIBRATED_EDGE",
        "side": str(action["side"]),
    }
    return [no_trade, trade]


def _score_candidate_texts(tokenizer: Any, model: Any, prompt: str, candidates: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch

    prompt_ids = tokenizer(_chat_text(tokenizer, prompt), add_special_tokens=False)["input_ids"]
    sequences: list[list[int]] = []
    spans: list[tuple[int, int]] = []
    target_texts: list[str] = []
    for candidate in candidates:
        text = json.dumps(candidate, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if tokenizer.eos_token_id is not None:
            ids = ids + [int(tokenizer.eos_token_id)]
        start = len(prompt_ids)
        end = start + len(ids)
        sequences.append(prompt_ids + ids)
        spans.append((start, end))
        target_texts.append(text)
    encoded = tokenizer.pad({"input_ids": sequences}, return_tensors="pt")
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    scored: list[dict[str, Any]] = []
    for i, (start, end) in enumerate(spans):
        positions = torch.arange(start - 1, end - 1, device=log_probs.device)
        label_tensor = input_ids[i, start:end]
        token_scores = log_probs[i, positions, label_tensor]
        score = float(token_scores.mean().detach().cpu())
        scored.append({"candidate": candidates[i], "text": target_texts[i], "mean_logprob": score})
    best = max(scored, key=lambda x: float(x["mean_logprob"]))["candidate"]
    return dict(best), scored


def _candidate_logprob_actions(
    tokenizer: Any,
    model: Any,
    records: list[dict[str, Any]],
    rules: dict[str, dict[str, Any]],
    cfg: CalibratedPolicyConfig,
    *,
    candidate_trade_margin: float = 0.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    actions: list[dict[str, Any]] = []
    previews: list[dict[str, Any]] = []
    policy_book = format_policy_book(rules)
    for row in records:
        candidates = _candidate_targets_for_row(row, rules)
        if len(candidates) == 1:
            action = candidates[0]
            scored = [{"candidate": action, "mean_logprob": None}]
        else:
            prompt = build_policy_trader_input(
                analyzer_summary_to_text(row["summary"]),
                hold_candidates=cfg.hold_candidates,
                entry_delay_bars=cfg.entry_delay_bars,
                current_policy_key=str(row["key"]),
                policy_book=policy_book,
            )
            action, scored = _score_candidate_texts(tokenizer, model, prompt, candidates)
            if len(scored) == 2:
                no_score = float(scored[0]["mean_logprob"])
                trade_score = float(scored[1]["mean_logprob"])
                margin = trade_score - no_score
                action = candidates[1] if margin >= float(candidate_trade_margin) else candidates[0]
                action = {**action, "logprob_margin": margin}
        actions.append(action)
        if len(previews) < 50 and len(candidates) > 1:
            previews.append({"date": row["date"], "signal_pos": row["signal_pos"], "key": row["key"], "parsed": action, "scores": scored})
    return actions, previews


def _policy_oracle_actions(records: list[dict[str, Any]], rules: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    next_available_pos = -1
    for row in records:
        target, next_available_pos = _policy_target_for_record(row, rules, next_available_pos=next_available_pos)
        actions.append(target)
    return actions


def _metrics_from_actions(records: list[dict[str, Any]], actions: list[dict[str, Any]]) -> dict[str, Any]:
    trades: list[dict[str, Any]] = []
    next_available_pos = -1
    invalid_or_missing = 0
    overlap_skips = 0
    for row, action in zip(records, actions):
        signal_pos = int(row["signal_pos"])
        if signal_pos <= next_available_pos:
            overlap_skips += 1
            continue
        if str(action.get("gate", "NO_TRADE")).upper() != "TRADE":
            continue
        side = str(action.get("side", "")).upper()
        hold_bars = int(action.get("hold_bars", 0) or 0)
        outcome = row["actions"].get(f"{side}_{hold_bars}")
        if outcome is None:
            invalid_or_missing += 1
            continue
        trades.append({"date": row["date"], "signal_pos": signal_pos, "key": row["key"], **outcome})
        next_available_pos = signal_pos + hold_bars
    metrics = _metrics_from_trades(trades, records_count=len(records), include_intratrade_mdd=True)
    metrics["non_overlapping"] = True
    metrics["model_invalid_or_missing_action"] = invalid_or_missing
    metrics["model_overlap_skips"] = overlap_skips
    return metrics


def _agreement(records: list[dict[str, Any]], model_actions: list[dict[str, Any]], oracle_actions: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    exact = 0
    gate_correct = 0
    for row, pred, target in zip(records, model_actions, oracle_actions):
        pg, tg = str(pred.get("gate", "NO_TRADE")), str(target.get("gate", "NO_TRADE"))
        ps, ts = str(pred.get("side", "NONE")), str(target.get("side", "NONE"))
        ph, th = int(pred.get("hold_bars", 0) or 0), int(target.get("hold_bars", 0) or 0)
        gate_correct += int(pg == tg)
        exact += int((pg, ps, ph) == (tg, ts, th))
        key = f"target={tg}/{ts}/{th}|pred={pg}/{ps}/{ph}"
        counts[key] = counts.get(key, 0) + 1
    n = max(1, len(records))
    return {"records": len(records), "gate_accuracy": gate_correct / n, "exact_action_accuracy": exact / n, "confusion": dict(sorted(counts.items()))}


def run_model_policy_eval(
    *,
    market_csv: str,
    output: str,
    adapter_dir: str,
    model_name: str = RECOMMENDED_VLM_MODEL,
    wave_trading_root: str = "",
    train_start: str,
    train_end: str,
    eval_start: str,
    eval_end: str,
    stride_bars: int = 12,
    hold_candidates: str = "48,96,144,288",
    min_train_samples: int = 12,
    min_train_mean_net: float = 0.0,
    min_train_mean_utility: float = -0.002,
    min_train_win_rate: float = 0.55,
    max_train_mean_mae: float = 0.015,
    key_fields: str = "regime,trend_alignment,location,risk_state",
    max_eval_records: int = 0,
    max_new_tokens: int = 80,
    generation_batch_size: int = 4,
    prediction_mode: str = "model",
    rule_guard: str = "none",
    candidate_trade_margin: float = 0.0,
) -> dict[str, Any]:
    cfg = CalibratedPolicyConfig(
        hold_candidates=parse_hold_candidates(hold_candidates),
        min_train_samples=int(min_train_samples),
        min_train_mean_net=float(min_train_mean_net),
        min_train_mean_utility=float(min_train_mean_utility),
        min_train_win_rate=float(min_train_win_rate),
        max_train_mean_mae=float(max_train_mean_mae),
        key_fields=tuple(x.strip() for x in str(key_fields).split(",") if x.strip()),
    )
    market = load_market_frame(market_csv, wave_trading_root=wave_trading_root or None)
    train_records = build_calibration_records(market, cfg, start_date=train_start, end_date=train_end, stride_bars=stride_bars)
    eval_records = build_calibration_records(market, cfg, start_date=eval_start, end_date=eval_end, stride_bars=stride_bars)
    if max_eval_records and int(max_eval_records) > 0:
        eval_records = eval_records[: int(max_eval_records)]
    rules = fit_rules(train_records, cfg)
    oracle_actions = _policy_oracle_actions(eval_records, rules)
    policy_book = format_policy_book(rules)
    if prediction_mode == "oracle_echo":
        resolved_model = resolve_vlm_model_alias(model_name, prefer_latest=True)
        generated_rows: list[dict[str, Any]] = []
        model_actions = oracle_actions
    elif prediction_mode == "candidate_logprob":
        if not adapter_dir:
            raise ValueError("adapter_dir is required for prediction_mode=candidate_logprob")
        resolved_model, tokenizer, model = _load_model(model_name, adapter_dir)
        model_actions, generated_rows = _candidate_logprob_actions(
            tokenizer,
            model,
            eval_records,
            rules,
            cfg,
            candidate_trade_margin=float(candidate_trade_margin),
        )
    elif prediction_mode == "model":
        if not adapter_dir:
            raise ValueError("adapter_dir is required for prediction_mode=model")
        resolved_model, tokenizer, model = _load_model(model_name, adapter_dir)
        prompts = [
            build_policy_trader_input(
                analyzer_summary_to_text(row["summary"]),
                hold_candidates=cfg.hold_candidates,
                entry_delay_bars=cfg.entry_delay_bars,
                current_policy_key=str(row["key"]),
                policy_book=policy_book,
            )
            for row in eval_records
        ]
        generated = _generate_actions_batched(
            tokenizer,
            model,
            prompts,
            max_new_tokens=max_new_tokens,
            allowed_holds=cfg.hold_candidates,
            batch_size=int(generation_batch_size),
        )
        raw_model_actions = [action for action, _raw in generated]
        model_actions = [_apply_rule_guard(row, action, rules, rule_guard) for row, action in zip(eval_records, raw_model_actions)]
        generated_rows = [
            {
                "date": row["date"],
                "signal_pos": row["signal_pos"],
                "key": row["key"],
                "raw": raw,
                "parsed": action,
                "guarded": guarded,
            }
            for row, (action, raw), guarded in list(zip(eval_records, generated, model_actions))[:50]
        ]
    else:
        raise ValueError("prediction_mode must be one of {'model','candidate_logprob','oracle_echo'}")
    report = {
        "market_csv": str(Path(market_csv).resolve()),
        "model_name": resolved_model,
        "adapter_dir": adapter_dir,
        "prediction_mode": prediction_mode,
        "generation_batch_size": int(generation_batch_size),
        "rule_guard": str(rule_guard),
        "candidate_trade_margin": float(candidate_trade_margin),
        "config": asdict(cfg),
        "periods": {"train": [train_start, train_end], "eval": [eval_start, eval_end]},
        "records": {"train": len(train_records), "eval": len(eval_records)},
        "rules_count": len(rules),
        "model_metrics": _metrics_from_actions(eval_records, model_actions),
        "oracle_metrics": _metrics_from_actions(eval_records, oracle_actions),
        "agreement": _agreement(eval_records, model_actions, oracle_actions),
        "generated_preview": generated_rows,
        "leakage_guard": {"rules_fit_on_train_period_only": True, "model_inputs_use_past_only_analyzer_summary": True},
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate generated text trader actions with strict policy backtest")
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--adapter-dir", default="")
    p.add_argument("--model-name", default=RECOMMENDED_VLM_MODEL)
    p.add_argument("--wave-trading-root", default="")
    p.add_argument("--train-start", required=True)
    p.add_argument("--train-end", required=True)
    p.add_argument("--eval-start", required=True)
    p.add_argument("--eval-end", required=True)
    p.add_argument("--stride-bars", type=int, default=12)
    p.add_argument("--hold-candidates", default="48,96,144,288")
    p.add_argument("--min-train-samples", type=int, default=12)
    p.add_argument("--min-train-mean-net", type=float, default=0.0)
    p.add_argument("--min-train-mean-utility", type=float, default=-0.002)
    p.add_argument("--min-train-win-rate", type=float, default=0.55)
    p.add_argument("--max-train-mean-mae", type=float, default=0.015)
    p.add_argument("--key-fields", default="regime,trend_alignment,location,risk_state")
    p.add_argument("--max-eval-records", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--generation-batch-size", type=int, default=4)
    p.add_argument("--prediction-mode", choices=["model", "candidate_logprob", "oracle_echo"], default="model")
    p.add_argument("--rule-guard", choices=["none", "current_key_any", "current_key_action"], default="none")
    p.add_argument("--candidate-trade-margin", type=float, default=0.0)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run_model_policy_eval(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
