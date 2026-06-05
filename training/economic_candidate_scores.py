"""Export and rerank candidate action scores for economic trader policies.

Candidate logprob selection can be dominated by unconditional JSON/action priors
(e.g. always preferring hold=72).  This module records every candidate score per
prompt and supports calibration by subtracting train-only action priors before
choosing an action.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from models.option_b_vlm import RECOMMENDED_VLM_MODEL, resolve_vlm_model_alias
from training.eval_text_trader import _action_json, _candidate_actions, _chat_prompt_text, _load_text_model, parse_trader_json
from training.train_text_dpo import load_preference_jsonl


def _action_key(action: dict[str, Any]) -> str:
    return f"{action['gate']}/{action['side']}/{int(action.get('hold_bars', 0) or 0)}"


def export_candidate_scores(
    *,
    eval_jsonl: str,
    output: str,
    model_name: str = RECOMMENDED_VLM_MODEL,
    adapter_dir: str,
    max_samples: int = 0,
    sample_mode: str = "sequential",
    seed: int = 42,
    hold_candidates: str = "36,72,144,288,432",
    score_normalization: str = "mean",
) -> dict[str, Any]:
    import torch

    if not adapter_dir:
        raise ValueError("adapter_dir is required")
    normalize = str(score_normalization).strip().lower()
    if normalize not in {"sum", "mean"}:
        raise ValueError("score_normalization must be one of {'sum','mean'}")
    rows = load_preference_jsonl(eval_jsonl, max_samples=max_samples, sample_mode=sample_mode, seed=seed)
    holds = [int(x) for x in str(hold_candidates).split(",") if str(x).strip()]
    actions = _candidate_actions(holds)
    action_texts = [_action_json(a) for a in actions]
    tokenizer, model = _load_text_model(model_name, adapter_dir)

    out_rows: list[dict[str, Any]] = []
    for row in rows:
        prompt_text = _chat_prompt_text(tokenizer, str(row["prompt"]))
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        sequences: list[list[int]] = []
        spans: list[tuple[int, int]] = []
        for text in action_texts:
            candidate_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if tokenizer.eos_token_id is not None:
                candidate_ids = candidate_ids + [int(tokenizer.eos_token_id)]
            start = len(prompt_ids)
            end = start + len(candidate_ids)
            sequences.append(prompt_ids + candidate_ids)
            spans.append((start, end))
        encoded = tokenizer.pad({"input_ids": sequences}, return_tensors="pt")
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        candidates: list[dict[str, Any]] = []
        for i, (start, end) in enumerate(spans):
            token_positions = torch.arange(start - 1, end - 1, device=log_probs.device)
            labels = input_ids[i, start:end]
            token_scores = log_probs[i, token_positions, labels]
            score = token_scores.sum() if normalize == "sum" else token_scores.mean()
            action = dict(actions[i])
            candidates.append({"action": action, "action_key": _action_key(action), "score": float(score.detach().cpu())})
        chosen = parse_trader_json(str(row.get("chosen", "{}")))
        out_rows.append(
            {
                "date": row.get("date"),
                "signal_pos": row.get("signal_pos"),
                "chosen": chosen,
                "rejected": parse_trader_json(str(row.get("rejected", "{}"))),
                "chosen_action": row.get("chosen_action"),
                "utility_gap": row.get("utility_gap"),
                "candidates": candidates,
            }
        )
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in out_rows) + "\n")
    best_counts: dict[str, int] = defaultdict(int)
    for row in out_rows:
        best = max(row["candidates"], key=lambda c: float(c["score"]))
        best_counts[str(best["action_key"])] += 1
    summary = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "eval_jsonl": str(Path(eval_jsonl).resolve()),
        "output": output,
        "model_name": resolve_vlm_model_alias(model_name, prefer_latest=True),
        "adapter_dir": adapter_dir,
        "rows": len(out_rows),
        "score_normalization": normalize,
        "hold_candidates": hold_candidates,
        "raw_best_counts": dict(sorted(best_counts.items())),
    }
    return summary


def load_score_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"no score rows loaded from {path}")
    return rows


def fit_action_score_prior(rows: list[dict[str, Any]]) -> dict[str, float]:
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        for cand in row["candidates"]:
            key = str(cand["action_key"])
            sums[key] += float(cand["score"])
            counts[key] += 1
    return {k: sums[k] / counts[k] for k in sorted(sums)}


def rerank_score_rows(rows: list[dict[str, Any]], prior: dict[str, float], *, prior_scale: float = 1.0) -> list[dict[str, Any]]:
    preds: list[dict[str, Any]] = []
    for row in rows:
        best = None
        best_score = float("-inf")
        for cand in row["candidates"]:
            key = str(cand["action_key"])
            adjusted = float(cand["score"]) - float(prior_scale) * float(prior.get(key, 0.0))
            if adjusted > best_score:
                best_score = adjusted
                best = cand
        assert best is not None
        preds.append(
            {
                "date": row.get("date"),
                "signal_pos": row.get("signal_pos"),
                "prediction": best["action"],
                "chosen": row.get("chosen"),
                "chosen_action": row.get("chosen_action"),
                "utility_gap": row.get("utility_gap"),
                "rerank_score": best_score,
            }
        )
    return preds


def _prediction_summary(preds: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = defaultdict(int)
    exact = 0
    gate_ok = 0
    for row in preds:
        pred = parse_trader_json(json.dumps(row["prediction"]))
        chosen = parse_trader_json(json.dumps(row.get("chosen", {})))
        counts[_action_key(pred)] += 1
        if pred == chosen:
            exact += 1
        if pred["gate"] == chosen["gate"]:
            gate_ok += 1
    return {"rows": len(preds), "prediction_counts": dict(sorted(counts.items())), "exact_action_accuracy": exact / max(1, len(preds)), "gate_accuracy": gate_ok / max(1, len(preds))}


def rerank_candidate_scores(*, score_jsonl: str, output: str, prior_jsonl: str = "", prior_scale: float = 1.0, prior_output: str = "") -> dict[str, Any]:
    rows = load_score_rows(score_jsonl)
    prior_rows = load_score_rows(prior_jsonl) if prior_jsonl else rows
    prior = fit_action_score_prior(prior_rows)
    preds = rerank_score_rows(rows, prior, prior_scale=prior_scale)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in preds) + "\n")
    if prior_output:
        Path(prior_output).parent.mkdir(parents=True, exist_ok=True)
        Path(prior_output).write_text(json.dumps(prior, indent=2, ensure_ascii=False, sort_keys=True))
    return {"score_jsonl": score_jsonl, "prior_jsonl": prior_jsonl or score_jsonl, "output": output, "prior_scale": prior_scale, "prediction_summary": _prediction_summary(preds)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export or rerank economic trader candidate scores")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--eval-jsonl", required=True)
    e.add_argument("--output", required=True)
    e.add_argument("--model-name", default=RECOMMENDED_VLM_MODEL)
    e.add_argument("--adapter-dir", required=True)
    e.add_argument("--max-samples", type=int, default=0)
    e.add_argument("--sample-mode", choices=["sequential", "random", "balanced", "gate_balanced"], default="sequential")
    e.add_argument("--seed", type=int, default=42)
    e.add_argument("--hold-candidates", default="36,72,144,288,432")
    e.add_argument("--score-normalization", choices=["sum", "mean"], default="mean")
    r = sub.add_parser("rerank")
    r.add_argument("--score-jsonl", required=True)
    r.add_argument("--output", required=True)
    r.add_argument("--prior-jsonl", default="")
    r.add_argument("--prior-scale", type=float, default=1.0)
    r.add_argument("--prior-output", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "export":
        payload = export_candidate_scores(
            eval_jsonl=args.eval_jsonl,
            output=args.output,
            model_name=args.model_name,
            adapter_dir=args.adapter_dir,
            max_samples=args.max_samples,
            sample_mode=args.sample_mode,
            seed=args.seed,
            hold_candidates=args.hold_candidates,
            score_normalization=args.score_normalization,
        )
    else:
        payload = rerank_candidate_scores(score_jsonl=args.score_jsonl, output=args.output, prior_jsonl=args.prior_jsonl, prior_scale=args.prior_scale, prior_output=args.prior_output)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
