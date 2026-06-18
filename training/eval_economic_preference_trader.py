"""Evaluate economic-preference DPO trader actions and export predictions.

The preference dataset stores ``chosen``/``rejected`` responses instead of a
supervised ``target``.  This evaluator treats ``chosen`` as the economic oracle
for classification diagnostics, while preserving every prediction for later
strict OHLC backtesting.  Prompts remain past-only; chosen/rejected are never
fed to the model at evaluation time.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from models.option_b_vlm import RECOMMENDED_VLM_MODEL, resolve_vlm_model_alias
from training.eval_text_trader import (
    _candidate_logprob_predictions,
    _chat_prompt_text,
    _generate_predictions,
    _metrics,
    _score_candidate_batch,
    _load_text_model,
    parse_trader_json,
)
from training.train_text_dpo import load_preference_jsonl


def dedupe_signal_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, int]] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = (str(row.get("date")), int(row.get("signal_pos", -1) or -1))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _target_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**r, "target": str(r["chosen"])} for r in rows]


def _action_key(action: dict[str, Any]) -> str:
    return f"{action.get('gate')}/{action.get('side')}/{int(action.get('hold_bars', 0) or 0)}"


def _extract_candidate_book(prompt: str) -> list[dict[str, Any]]:
    marker = "Candidate action book: "
    for line in str(prompt).splitlines():
        if line.startswith(marker):
            book = json.loads(line[len(marker) :])
            if not isinstance(book, list):
                raise ValueError("candidate action book is not a list")
            return [b for b in book if isinstance(b, dict)]
    raise ValueError("prompt lacks Candidate action book")


def _full_action_json(action: dict[str, Any]) -> str:
    if str(action.get("gate", "NO_TRADE")).upper() == "NO_TRADE":
        payload = {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0, "family": "NONE", "confidence": "HIGH"}
    else:
        payload = {
            "gate": "TRADE",
            "side": str(action["side"]).upper(),
            "hold_bars": int(action["hold_bars"]),
            "family": str(action["family"]),
            "confidence": str(action.get("confidence", "HIGH")).upper(),
        }
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def _candidate_book_actions(row: dict[str, Any], hold_candidates: list[int]) -> list[dict[str, Any]]:
    holds = [int(h) for h in hold_candidates if int(h) > 0]
    actions: list[dict[str, Any]] = [{"gate": "NO_TRADE", "family": "NONE", "side": "NONE", "hold_bars": 0, "confidence": "HIGH"}]
    seen = {("NO_TRADE", "NONE", "NONE", 0)}
    for cand in _extract_candidate_book(str(row["prompt"])):
        family = str(cand.get("family", "")).strip()
        side = str(cand.get("side", "")).upper().strip()
        allowed = cand.get("allowed_holds", holds)
        allowed_holds = [int(h) for h in allowed if int(h) in holds]
        if not family or side not in {"LONG", "SHORT"}:
            continue
        for hold in allowed_holds:
            key = ("TRADE", family, side, int(hold))
            if key in seen:
                continue
            seen.add(key)
            actions.append({"gate": "TRADE", "family": family, "side": side, "hold_bars": int(hold), "confidence": "HIGH"})
    return actions


def _candidate_book_logprob_predictions(
    rows: list[dict[str, Any]],
    *,
    model_name: str,
    adapter_dir: str,
    hold_candidates: list[int],
    score_normalization: str = "mean",
    batch_size: int = 1,
) -> list[dict[str, Any]]:
    tokenizer, model = _load_text_model(model_name, adapter_dir)
    normalize = str(score_normalization).strip().lower()
    if normalize not in {"sum", "mean"}:
        raise ValueError("score_normalization must be one of {'sum','mean'}")
    batch_size = max(1, int(batch_size))

    preds: list[dict[str, Any]] = []
    for offset in range(0, len(rows), batch_size):
        row_batch = rows[offset : offset + batch_size]
        sequences: list[list[int]] = []
        candidate_spans: list[tuple[int, int]] = []
        candidate_actions_by_row: list[list[dict[str, Any]]] = []
        for row in row_batch:
            prompt_ids = tokenizer(_chat_prompt_text(tokenizer, str(row["prompt"])), add_special_tokens=False)["input_ids"]
            start = len(prompt_ids)
            actions = _candidate_book_actions(row, hold_candidates)
            candidate_actions_by_row.append(actions)
            for action in actions:
                action_ids = tokenizer(_full_action_json(action), add_special_tokens=False)["input_ids"]
                if tokenizer.eos_token_id is not None:
                    action_ids = action_ids + [int(tokenizer.eos_token_id)]
                sequences.append(prompt_ids + action_ids)
                candidate_spans.append((start, start + len(action_ids)))
        encoded = tokenizer.pad({"input_ids": sequences}, return_tensors="pt")
        flat_scores = _score_candidate_batch(
            model=model,
            input_ids=encoded["input_ids"].to(model.device),
            attention_mask=encoded["attention_mask"].to(model.device),
            spans=candidate_spans,
            score_normalization=normalize,
        )
        score_offset = 0
        for actions in candidate_actions_by_row:
            scores = flat_scores[score_offset : score_offset + len(actions)]
            best_idx = max(range(len(scores)), key=lambda i: scores[i])
            preds.append(parse_trader_json(_full_action_json(actions[best_idx])))
            score_offset += len(actions)
    return preds


def _prediction_rows(rows: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row, pred in zip(rows, predictions):
        chosen = parse_trader_json(str(row.get("chosen", "{}")))
        rejected = parse_trader_json(str(row.get("rejected", "{}")))
        out.append(
            {
                "date": row.get("date"),
                "signal_pos": row.get("signal_pos"),
                "prediction": pred,
                "chosen": chosen,
                "rejected": rejected,
                "chosen_action": row.get("chosen_action"),
                "rejected_action": row.get("rejected_action"),
                "utility_gap": row.get("utility_gap"),
            }
        )
    return out


def _summarize_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pred_counts = Counter(_action_key(r["prediction"]) for r in rows)
    chosen_counts = Counter(_action_key(r["chosen"]) for r in rows)
    duplicate_keys = Counter((str(r.get("date")), int(r.get("signal_pos", -1) or -1)) for r in rows)
    return {
        "prediction_counts": dict(sorted(pred_counts.items())),
        "chosen_counts": dict(sorted(chosen_counts.items())),
        "rows": len(rows),
        "unique_signal_rows": len(duplicate_keys),
        "duplicate_preference_pairs": sum(max(0, n - 1) for n in duplicate_keys.values()),
    }


def evaluate_economic_preference_trader(
    *,
    eval_jsonl: str,
    output: str,
    predictions_output: str = "",
    model_name: str = RECOMMENDED_VLM_MODEL,
    adapter_dir: str = "",
    max_samples: int = 0,
    sample_mode: str = "sequential",
    seed: int = 42,
    prediction_mode: str = "target_echo",
    max_new_tokens: int = 48,
    hold_candidates: str = "36,72,144,288,432",
    score_normalization: str = "mean",
    batch_size: int = 1,
    dedupe_signals: bool = False,
) -> dict[str, Any]:
    loaded_rows = load_preference_jsonl(eval_jsonl, max_samples=max_samples, sample_mode=sample_mode, seed=seed)
    rows = dedupe_signal_rows(loaded_rows) if dedupe_signals else loaded_rows
    target_rows = _target_rows(rows)
    mode = str(prediction_mode).strip().lower()
    if mode == "target_echo":
        predictions = [parse_trader_json(str(r["chosen"])) for r in rows]
    elif mode == "model":
        if not adapter_dir:
            raise ValueError("adapter_dir is required for prediction_mode=model")
        predictions = _generate_predictions(target_rows, model_name=model_name, adapter_dir=adapter_dir, max_new_tokens=max_new_tokens)
    elif mode == "candidate_logprob":
        if not adapter_dir:
            raise ValueError("adapter_dir is required for prediction_mode=candidate_logprob")
        holds = [int(x) for x in str(hold_candidates).split(",") if str(x).strip()]
        predictions = _candidate_logprob_predictions(
            target_rows,
            model_name=model_name,
            adapter_dir=adapter_dir,
            hold_candidates=holds,
            score_normalization=score_normalization,
            batch_size=batch_size,
        )
    elif mode == "candidate_book_logprob":
        if not adapter_dir:
            raise ValueError("adapter_dir is required for prediction_mode=candidate_book_logprob")
        holds = [int(x) for x in str(hold_candidates).split(",") if str(x).strip()]
        predictions = _candidate_book_logprob_predictions(
            rows,
            model_name=model_name,
            adapter_dir=adapter_dir,
            hold_candidates=holds,
            score_normalization=score_normalization,
            batch_size=batch_size,
        )
    else:
        raise ValueError("prediction_mode must be one of {'target_echo','model','candidate_logprob','candidate_book_logprob'}")

    pred_rows = _prediction_rows(rows, predictions)
    if predictions_output:
        outp = Path(predictions_output)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in pred_rows) + "\n")
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "eval_jsonl": str(Path(eval_jsonl).resolve()),
        "model_name": resolve_vlm_model_alias(model_name, prefer_latest=True),
        "adapter_dir": adapter_dir,
        "prediction_mode": mode,
        "predictions_output": predictions_output,
        "row_selection": {
            "loaded_rows": len(loaded_rows),
            "evaluated_rows": len(rows),
            "dedupe_signals": bool(dedupe_signals),
        },
        "metrics_vs_chosen": _metrics(target_rows, predictions),
        "prediction_summary": _summarize_predictions(pred_rows),
        "candidate_logprob": {
            "hold_candidates": hold_candidates,
            "score_normalization": score_normalization,
            "batch_size": batch_size,
        }
        if mode == "candidate_logprob"
        else None,
        "leakage_guard": {
            "prompt_uses_future_path": False,
            "chosen_rejected_used_for_metrics_only": True,
            "model_input_excludes_chosen_rejected": True,
            "candidate_book_logprob_uses_prompt_visible_actions_only": mode == "candidate_book_logprob",
        },
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate economic-preference DPO trader actions")
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--predictions-output", default="")
    p.add_argument("--model-name", default=RECOMMENDED_VLM_MODEL)
    p.add_argument("--adapter-dir", default="")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--sample-mode", choices=["sequential", "random", "balanced", "gate_balanced"], default="sequential")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prediction-mode", choices=["target_echo", "model", "candidate_logprob", "candidate_book_logprob"], default="target_echo")
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--hold-candidates", default="36,72,144,288,432")
    p.add_argument("--score-normalization", choices=["sum", "mean"], default="mean")
    p.add_argument("--batch-size", type=int, default=1, help="Rows per candidate-logprob scoring batch; economic action candidate sets are memory-heavy")
    p.add_argument("--dedupe-signals", action="store_true", help="Evaluate one preference row per signal for backtest-oriented scoring")
    return p.parse_args()


def main() -> None:
    print(json.dumps(evaluate_economic_preference_trader(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
