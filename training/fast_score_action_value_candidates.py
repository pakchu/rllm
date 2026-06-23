"""Fast TAKE/SKIP scorer for candidate-level event-action value rows."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from models.option_b_vlm import RECOMMENDED_VLM_MODEL, resolve_vlm_model_alias
from training.eval_text_label import _chat_prompt_text, _load_text_model
from training.score_action_value_candidates import LABELS, NO_TRADE, _first_n_signals, _key, _read


@dataclass(frozen=True)
class FastScoreActionValueCfg:
    value_jsonl: str
    adapter_dir: str
    predictions_output: str
    scores_output: str
    report_output: str
    model_name: str = RECOMMENDED_VLM_MODEL
    max_signals: int = 0
    score_key: str = "mean"
    threshold: float = 0.0
    progress_every: int = 500
    batch_size: int = 8


def _label_score_from_cache(model: Any, prompt_out: Any, device: Any, label_ids: list[int], normalize: str) -> float:
    import torch

    first_logits = prompt_out.logits[:, -1, :].float()[0]
    labels = torch.tensor(label_ids, dtype=torch.long, device=device)
    scores = [first_logits[int(labels[0])] - torch.logsumexp(first_logits, dim=-1)]
    if len(label_ids) > 1:
        out = model(input_ids=labels[:-1].reshape(1, -1), past_key_values=prompt_out.past_key_values, use_cache=False)
        logits = out.logits[0].float()
        for pos, label in enumerate(labels[1:]):
            row = logits[pos]
            scores.append(row[int(label)] - torch.logsumexp(row, dim=-1))
    stacked = torch.stack(scores)
    score = stacked.mean() if normalize == "mean" else stacked.sum()
    return float(score.detach().cpu())


def _label_scores_from_prompt_cache(model: Any, input_ids: Any, label_ids_by_label: dict[str, list[int]], normalize: str) -> dict[str, float]:
    import torch

    with torch.no_grad():
        prompt_out = model(input_ids=input_ids, use_cache=True)
        return {label: _label_score_from_cache(model, prompt_out, input_ids.device, ids, normalize) for label, ids in label_ids_by_label.items()}


def _label_score_from_prompt_cache(model: Any, input_ids: Any, label_ids: list[int], normalize: str) -> float:
    return _label_scores_from_prompt_cache(model, input_ids, {"label": label_ids}, normalize)["label"]


def _batched_label_scores(model: Any, tokenizer: Any, prompt_texts: list[str], label_ids_by_label: dict[str, list[int]], normalize: str) -> list[dict[str, dict[str, float]]]:
    """Score TAKE/SKIP labels for multiple prompts in one full-sequence forward pass.

    This intentionally recomputes each prompt once per label but keeps the GPU fed by
    batching rows. It is substantially faster than row-wise KV-cache scoring on
    Gemma-class models where prompt lengths are short enough for batched padding.
    """
    import torch

    old_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    try:
        sequences: list[list[int]] = []
        spans: list[tuple[int, int, str, int]] = []
        for row_idx, text in enumerate(prompt_texts):
            prompt_ids = tokenizer(_chat_prompt_text(tokenizer, text), add_special_tokens=False)["input_ids"]
            for label, label_ids in label_ids_by_label.items():
                start = len(prompt_ids)
                end = start + len(label_ids)
                sequences.append(prompt_ids + label_ids)
                spans.append((start, end, label, row_idx))
        encoded = tokenizer.pad({"input_ids": sequences}, return_tensors="pt")
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :].float()
            log_denoms = torch.logsumexp(logits, dim=-1)
        out: list[dict[str, dict[str, float]]] = [dict() for _ in prompt_texts]
        width = int(input_ids.shape[1])
        for seq_idx, (start, end, label, row_idx) in enumerate(spans):
            unpadded_len = end
            pad_len = width - unpadded_len
            positions = torch.arange(pad_len + start - 1, pad_len + end - 1, device=logits.device)
            label_tensor = input_ids[seq_idx, pad_len + start : pad_len + end]
            token_logits = logits[seq_idx, positions, label_tensor]
            token_scores = token_logits - log_denoms[seq_idx, positions]
            sum_score = float(token_scores.sum().detach().cpu())
            mean_score = sum_score / max(1, len(label_ids_by_label[label]))
            out[row_idx][label] = {"mean": mean_score, "sum": sum_score}
        return out
    finally:
        tokenizer.padding_side = old_padding_side


def _prediction_rows(scored: list[dict[str, Any]], *, margin_field: str, threshold: float) -> list[dict[str, Any]]:
    by_signal: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in scored:
        by_signal.setdefault(_key(row), []).append(row)
    out = []
    for key, group in sorted(by_signal.items(), key=lambda kv: kv[0]):
        best = max(group, key=lambda r: float(r[margin_field]))
        margin = float(best[margin_field])
        action = best.get("action", {}) if isinstance(best.get("action"), dict) else {}
        if margin < float(threshold):
            pred = dict(NO_TRADE)
        else:
            pred = {
                "gate": "TRADE",
                "family": action.get("family", "UNKNOWN"),
                "side": str(action.get("side", "NONE")).upper(),
                "hold_bars": int(action.get("hold_bars", 0) or 0),
                "confidence": "HIGH",
            }
        out.append({"date": key[0], "signal_pos": key[1], "prediction": pred, "value_margin": margin, "selected_action_audit": best.get("action_audit")})
    return out


def run(cfg: FastScoreActionValueCfg) -> dict[str, Any]:
    normalize = str(cfg.score_key).lower().strip()
    if normalize not in {"mean", "sum"}:
        raise ValueError("score_key must be mean or sum")
    rows = _first_n_signals(_read(cfg.value_jsonl), int(cfg.max_signals))
    tokenizer, model = _load_text_model(cfg.model_name, cfg.adapter_dir)
    label_ids: dict[str, list[int]] = {}
    for label in LABELS:
        ids = tokenizer(label, add_special_tokens=False)["input_ids"]
        if tokenizer.eos_token_id is not None:
            ids = ids + [int(tokenizer.eos_token_id)]
        label_ids[label] = ids
    scored: list[dict[str, Any]] = []
    t0 = time.time()
    batch_size = max(1, int(cfg.batch_size))
    progress_every = max(0, int(cfg.progress_every))
    next_progress = progress_every if progress_every > 0 else 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        batch_scores = _batched_label_scores(model, tokenizer, [str(row["prompt"]) for row in batch], label_ids, normalize)
        for row, scores in zip(batch, batch_scores):
            nr = dict(row)
            nr["score"] = scores
            nr["margin_mean_take_minus_skip"] = scores["TAKE"]["mean"] - scores["SKIP"]["mean"]
            nr["margin_sum_take_minus_skip"] = scores["TAKE"]["sum"] - scores["SKIP"]["sum"]
            scored.append(nr)
        done = len(scored)
        should_report = False
        if progress_every > 0 and done >= next_progress:
            should_report = True
            while next_progress <= done:
                next_progress += progress_every
        if done == len(rows):
            should_report = True
        if should_report:
            print(json.dumps({"scored_rows": done, "rows": len(rows), "elapsed_sec": round(time.time() - t0, 2)}, ensure_ascii=False), flush=True)
    margin_field = "margin_mean_take_minus_skip" if normalize == "mean" else "margin_sum_take_minus_skip"
    preds = _prediction_rows(scored, margin_field=margin_field, threshold=float(cfg.threshold))
    Path(cfg.scores_output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.scores_output).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in scored) + ("\n" if scored else ""))
    Path(cfg.predictions_output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.predictions_output).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in preds) + ("\n" if preds else ""))
    counts: dict[str, int] = {}
    for pred in preds:
        p = pred["prediction"]
        key = f"{p.get('gate')}/{p.get('side')}/{p.get('hold_bars')}"
        counts[key] = counts.get(key, 0) + 1
    report = {"config": asdict(cfg), "model_name": resolve_vlm_model_alias(cfg.model_name, prefer_latest=True), "rows_scored": len(scored), "signals": len(preds), "prediction_counts": dict(sorted(counts.items())), "elapsed_sec": time.time() - t0}
    Path(cfg.report_output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.report_output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fast KV-cache score action value candidates")
    p.add_argument("--value-jsonl", required=True)
    p.add_argument("--adapter-dir", required=True)
    p.add_argument("--predictions-output", required=True)
    p.add_argument("--scores-output", required=True)
    p.add_argument("--report-output", required=True)
    p.add_argument("--model-name", default=FastScoreActionValueCfg.model_name)
    p.add_argument("--max-signals", type=int, default=0)
    p.add_argument("--score-key", choices=["mean", "sum"], default="mean")
    p.add_argument("--threshold", type=float, default=0.0)
    p.add_argument("--progress-every", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=8)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(FastScoreActionValueCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
