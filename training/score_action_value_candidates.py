"""Score TAKE/SKIP margins for candidate-level action value rows and emit actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from models.option_b_vlm import RECOMMENDED_VLM_MODEL, resolve_vlm_model_alias
from training.eval_text_label import _chat_prompt_text, _load_text_model
from training.economic_action_backtest import EconomicActionBacktestConfig, strict_backtest_actions
from training.strict_bar_backtest import load_market_bars

LABELS = ["SKIP", "TAKE"]
NO_TRADE = {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0, "family": "NONE", "confidence": "HIGH"}


def _read(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _key(row: dict[str, Any]) -> tuple[str, int]:
    return (str(row.get("date")), int(row.get("signal_pos", -1) or -1))


def _first_n_signals(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    if not n or n <= 0:
        return rows
    seen: set[tuple[str, int]] = set()
    keep: set[tuple[str, int]] = set()
    for row in rows:
        key = _key(row)
        if key not in seen:
            seen.add(key)
            if len(keep) < n:
                keep.add(key)
            else:
                break
    return [r for r in rows if _key(r) in keep]


def _score_batch(model: Any, input_ids: Any, attention_mask: Any, spans: list[tuple[int, int]]) -> tuple[list[float], list[float]]:
    import torch

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    sums, means = [], []
    for i, (start, end) in enumerate(spans):
        positions = torch.arange(start - 1, end - 1, device=logits.device)
        labels = input_ids[i, start:end]
        selected_logits = logits[i, positions, :].float()
        label_logits = selected_logits.gather(1, labels.reshape(-1, 1)).squeeze(1)
        token_scores = label_logits - torch.logsumexp(selected_logits, dim=-1)
        sums.append(float(token_scores.sum().detach().cpu()))
        means.append(float(token_scores.mean().detach().cpu()))
    return sums, means


def score_value_rows(*, value_jsonl: str, predictions_output: str, scores_output: str, model_name: str = RECOMMENDED_VLM_MODEL, adapter_dir: str, batch_size: int = 8, max_signals: int = 0, score_key: str = "mean", threshold: float = 0.0) -> dict[str, Any]:
    rows = _first_n_signals(_read(value_jsonl), int(max_signals))
    tokenizer, model = _load_text_model(model_name, adapter_dir)
    label_ids = {}
    for label in LABELS:
        ids = tokenizer(label, add_special_tokens=False)["input_ids"]
        if tokenizer.eos_token_id is not None:
            ids = ids + [int(tokenizer.eos_token_id)]
        label_ids[label] = ids
    scored: list[dict[str, Any]] = []
    bs = max(1, int(batch_size))
    for offset in range(0, len(rows), bs):
        batch = rows[offset : offset + bs]
        sequences: list[list[int]] = []
        spans: list[tuple[int, int]] = []
        for row in batch:
            prompt_ids = tokenizer(_chat_prompt_text(tokenizer, str(row["prompt"])), add_special_tokens=False)["input_ids"]
            start = len(prompt_ids)
            for label in LABELS:
                ids = label_ids[label]
                sequences.append(prompt_ids + ids)
                spans.append((start, start + len(ids)))
        encoded = tokenizer.pad({"input_ids": sequences}, return_tensors="pt")
        sums, means = _score_batch(model, encoded["input_ids"].to(model.device), encoded["attention_mask"].to(model.device), spans)
        p = 0
        for row in batch:
            score = {"SKIP": {"sum": sums[p], "mean": means[p]}, "TAKE": {"sum": sums[p + 1], "mean": means[p + 1]}}
            scored.append({**row, "score": score, "margin_sum_take_minus_skip": score["TAKE"]["sum"] - score["SKIP"]["sum"], "margin_mean_take_minus_skip": score["TAKE"]["mean"] - score["SKIP"]["mean"]})
            p += 2
    by_signal: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in scored:
        by_signal.setdefault(_key(row), []).append(row)
    pred_rows = []
    margin_field = "margin_mean_take_minus_skip" if score_key == "mean" else "margin_sum_take_minus_skip"
    for key, group in sorted(by_signal.items(), key=lambda kv: kv[0]):
        best = max(group, key=lambda r: float(r[margin_field]))
        margin = float(best[margin_field])
        action = best.get("action", {}) if isinstance(best.get("action"), dict) else {}
        if margin < float(threshold):
            pred = dict(NO_TRADE)
        else:
            pred = {"gate": "TRADE", "family": action.get("family", "UNKNOWN"), "side": str(action.get("side", "NONE")).upper(), "hold_bars": int(action.get("hold_bars", 0) or 0), "confidence": "HIGH"}
        pred_rows.append({"date": key[0], "signal_pos": key[1], "prediction": pred, "value_margin": margin, "selected_action_audit": best.get("action_audit")})
    Path(scores_output).parent.mkdir(parents=True, exist_ok=True)
    Path(scores_output).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in scored) + "\n")
    Path(predictions_output).parent.mkdir(parents=True, exist_ok=True)
    Path(predictions_output).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in pred_rows) + "\n")
    return {"value_jsonl": str(Path(value_jsonl).resolve()), "rows_scored": len(scored), "signals": len(pred_rows), "predictions_output": predictions_output, "scores_output": scores_output, "model_name": resolve_vlm_model_alias(model_name, prefer_latest=True), "adapter_dir": adapter_dir, "score_key": score_key, "threshold": threshold}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score action value candidates")
    p.add_argument("--value-jsonl", required=True)
    p.add_argument("--predictions-output", required=True)
    p.add_argument("--scores-output", required=True)
    p.add_argument("--model-name", default=RECOMMENDED_VLM_MODEL)
    p.add_argument("--adapter-dir", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-signals", type=int, default=0)
    p.add_argument("--score-key", choices=["mean", "sum"], default="mean")
    p.add_argument("--threshold", type=float, default=0.0)
    return p.parse_args()


def main() -> None:
    print(json.dumps(score_value_rows(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
