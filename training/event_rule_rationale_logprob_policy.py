"""Score event candidate rule-rationale decisions with a LoRA text model.

For each side-specific event candidate, this script scores three possible
assistant completions using log probability:
- same feature-derived analyzer JSON + ABSTAIN
- same feature-derived analyzer JSON + TAKE_SMALL
- same feature-derived analyzer JSON + TAKE_FULL

Then it converts the best side/decision per signal into the OnlineRiskOverlay
prediction JSONL format. The analyzer JSON is recomputed from signal-time
features only; reward labels are not read for prediction.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from training.build_event_rule_rationale_sft import _analyzer, _prompt
from training.train_text_sft import resolve_vlm_model_alias
from utils import disable_transformers_allocator_warmup

DECISIONS = ("ABSTAIN", "TAKE_SMALL", "TAKE_FULL")


@dataclass(frozen=True)
class RuleLogprobPolicyCfg:
    input_jsonl: str
    output_predictions: str
    report_output: str
    model_name: str = "gemma4-e4b"
    adapter_dir: str = "checkpoints/event_rule_rationale_gemma4_smoke_s512_step16"
    max_candidates: int = 0
    batch_size: int = 4
    score_normalization: str = "mean"
    small_scale: float = 0.5
    full_scale: float = 1.0
    decision_only: bool = False


def _load(path: str) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in open(path) if line.strip()]
    return rows


def _completion(row: dict[str, Any], decision: str) -> str:
    return json.dumps({"analyzer": _analyzer(row), "decision": decision}, ensure_ascii=False, sort_keys=True)


def _target_decision(row: dict[str, Any]) -> str:
    return str(row.get("target", {}).get("decision", "ABSTAIN"))


def _load_model(model_name: str, adapter_dir: str):
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
    return tokenizer, model, resolved


def _chat_prompt(tokenizer: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        return str(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    return f"<|user|>\n{prompt}\n<|assistant|>\n"


def _score_batch(model: Any, input_ids: Any, attention_mask: Any, spans: list[tuple[int, int]], normalize: str) -> list[float]:
    import torch

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    out: list[float] = []
    for i, (start, end) in enumerate(spans):
        positions = torch.arange(start - 1, end - 1, device=logits.device)
        labels = input_ids[i, start:end]
        selected = logits[i, positions, :].float()
        label_logits = selected.gather(1, labels.reshape(-1, 1)).squeeze(1)
        token_scores = label_logits - torch.logsumexp(selected, dim=-1)
        score = token_scores.mean() if normalize == "mean" else token_scores.sum()
        out.append(float(score.detach().cpu()))
    return out


def _candidate_scores(rows: list[dict[str, Any]], *, tokenizer: Any, model: Any, batch_size: int, normalize: str, decision_only: bool = False) -> list[dict[str, Any]]:
    import torch

    scored: list[dict[str, Any]] = []
    batch_size = max(1, int(batch_size))
    flat: list[tuple[dict[str, Any], str, str, list[int], int]] = []
    for row in rows:
        analyzer_json = json.dumps(_analyzer(row), ensure_ascii=False, sort_keys=True)
        if decision_only:
            prefix = _chat_prompt(tokenizer, _prompt(row)) + '{"analyzer": ' + analyzer_json + ', "decision": '
            prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
            for decision in DECISIONS:
                suffix = json.dumps(decision, ensure_ascii=False) + "}"
                cand_ids = tokenizer(suffix, add_special_tokens=False)["input_ids"]
                if tokenizer.eos_token_id is not None:
                    cand_ids = cand_ids + [int(tokenizer.eos_token_id)]
                flat.append((row, decision, analyzer_json, prefix_ids + cand_ids, len(prefix_ids)))
        else:
            prompt_ids = tokenizer(_chat_prompt(tokenizer, _prompt(row)), add_special_tokens=False)["input_ids"]
            for decision in DECISIONS:
                comp_ids = tokenizer(_completion(row, decision), add_special_tokens=False)["input_ids"]
                if tokenizer.eos_token_id is not None:
                    comp_ids = comp_ids + [int(tokenizer.eos_token_id)]
                flat.append((row, decision, analyzer_json, prompt_ids + comp_ids, len(prompt_ids)))
    for offset in range(0, len(flat), batch_size):
        chunk = flat[offset : offset + batch_size]
        sequences = [x[3] for x in chunk]
        spans = [(x[4], len(x[3])) for x in chunk]
        enc = tokenizer.pad({"input_ids": sequences}, return_tensors="pt")
        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc["attention_mask"].to(model.device)
        scores = _score_batch(model, input_ids, attention_mask, spans, normalize)
        for (row, decision, analyzer_json, _, _), score in zip(chunk, scores):
            scored.append({"row": row, "decision": decision, "score": score, "analyzer_json": analyzer_json})
        del input_ids, attention_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    grouped: list[dict[str, Any]] = []
    for i in range(0, len(scored), len(DECISIONS)):
        items = scored[i : i + len(DECISIONS)]
        by = {x["decision"]: float(x["score"]) for x in items}
        best = max(DECISIONS, key=lambda d: by[d])
        row = items[0]["row"]
        grouped.append({
            "row": row,
            "scores": by,
            "decision": best,
            "trade_edge": max(by["TAKE_SMALL"], by["TAKE_FULL"]) - by["ABSTAIN"],
            "analyzer_json": items[0]["analyzer_json"],
            "target_decision": _target_decision(row),
        })
    return grouped


def _to_overlay(scored: list[dict[str, Any]], cfg: RuleLogprobPolicyCfg) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    best_by_signal: dict[int, dict[str, Any]] = {}
    for item in scored:
        row = item["row"]
        pos = int(row.get("signal_pos"))
        cur = best_by_signal.get(pos)
        if cur is None or float(item["trade_edge"]) > float(cur["trade_edge"]):
            best_by_signal[pos] = item
    out: list[dict[str, Any]] = []
    counts = {"TRADE": 0, "NO_TRADE": 0, "LONG": 0, "SHORT": 0, "FULL": 0, "SMALL": 0}
    correct = 0
    for pos in sorted(best_by_signal):
        item = best_by_signal[pos]
        row = item["row"]
        decision = str(item["decision"])
        side = str(row.get("side"))
        hold = int(row.get("candidate", {}).get("hold_bars", 288) or 288)
        if decision == "TAKE_FULL":
            scale = float(cfg.full_scale)
            pred = {"gate": "TRADE", "side": side, "hold_bars": hold, "confidence": "HIGH", "family": "event_rule_rationale_logprob"}
            counts["TRADE"] += 1; counts[side] += 1; counts["FULL"] += 1
        elif decision == "TAKE_SMALL":
            scale = float(cfg.small_scale)
            pred = {"gate": "TRADE", "side": side, "hold_bars": hold, "confidence": "MEDIUM", "family": "event_rule_rationale_logprob"}
            counts["TRADE"] += 1; counts[side] += 1; counts["SMALL"] += 1
        else:
            scale = 0.0
            pred = {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0, "confidence": "LOW", "family": "event_rule_rationale_logprob"}
            counts["NO_TRADE"] += 1
        if decision == item["target_decision"]:
            correct += 1
        out.append({
            "date": row.get("date"),
            "signal_pos": row.get("signal_pos"),
            "prediction": pred,
            "position_scale": scale,
            "decision": decision,
            "side_candidate": side,
            "scores": item["scores"],
            "trade_edge": item["trade_edge"],
        })
    report = {"signals": len(out), "counts": counts, "best_candidate_decision_accuracy": correct / max(1, len(out))}
    return out, report


def run(cfg: RuleLogprobPolicyCfg) -> dict[str, Any]:
    rows = _load(cfg.input_jsonl)
    if cfg.max_candidates and int(cfg.max_candidates) < len(rows):
        rows = rows[: int(cfg.max_candidates)]
    normalize = str(cfg.score_normalization).lower().strip()
    if normalize not in {"mean", "sum"}:
        raise ValueError("score_normalization must be mean or sum")
    tokenizer, model, resolved = _load_model(cfg.model_name, cfg.adapter_dir)
    scored = _candidate_scores(rows, tokenizer=tokenizer, model=model, batch_size=cfg.batch_size, normalize=normalize, decision_only=cfg.decision_only)
    predictions, summary = _to_overlay(scored, cfg)
    Path(cfg.output_predictions).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output_predictions).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in predictions) + ("\n" if predictions else ""))
    candidate_acc = sum(1 for x in scored if x["decision"] == x["target_decision"]) / max(1, len(scored))
    report = {
        "config": asdict(cfg),
        "model_name_resolved": resolved,
        "candidate_rows": len(rows),
        "candidate_decision_accuracy": candidate_acc,
        "overlay_summary": summary,
        "leakage_guard": {"analyzer_recomputed_from_signal_features": True, "reward_target_not_used_for_prediction": True},
    }
    Path(cfg.report_output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.report_output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Event rule-rationale logprob policy")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--output-predictions", required=True)
    p.add_argument("--report-output", required=True)
    p.add_argument("--model-name", default=RuleLogprobPolicyCfg.model_name)
    p.add_argument("--adapter-dir", default=RuleLogprobPolicyCfg.adapter_dir)
    p.add_argument("--max-candidates", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--score-normalization", choices=["mean", "sum"], default=RuleLogprobPolicyCfg.score_normalization)
    p.add_argument("--small-scale", type=float, default=RuleLogprobPolicyCfg.small_scale)
    p.add_argument("--full-scale", type=float, default=RuleLogprobPolicyCfg.full_scale)
    p.add_argument("--decision-only", action="store_true")
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(RuleLogprobPolicyCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
