"""Fast candidate-logprob evaluation for text trader rows using prompt KV cache."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from models.option_b_vlm import RECOMMENDED_VLM_MODEL, resolve_vlm_model_alias
from training.eval_text_trader import _action_json, _candidate_actions, _chat_prompt_text, _load_text_model, _metrics
from training.train_text_sft import load_jsonl


@dataclass(frozen=True)
class FastEvalTextTraderCfg:
    eval_jsonl: str
    adapter_dir: str
    predictions_output: str
    report_output: str
    model_name: str = RECOMMENDED_VLM_MODEL
    max_samples: int = 0
    sample_mode: str = "sequential"
    seed: int = 42
    hold_candidates: str = "72,144,288,432"
    score_normalization: str = "mean"
    progress_every: int = 25


def _candidate_logprob_from_prompt_cache(model: Any, input_ids: Any, candidate_ids: list[int], normalize: str) -> float:
    import torch

    with torch.no_grad():
        prompt_out = model(input_ids=input_ids, use_cache=True)
        first_logits = prompt_out.logits[:, -1, :].float()
        labels = torch.tensor(candidate_ids, dtype=torch.long, device=input_ids.device)
        token_scores = []
        first = first_logits[0]
        token_scores.append(first[int(labels[0])] - torch.logsumexp(first, dim=-1))
        if len(candidate_ids) > 1:
            cont = labels[:-1].reshape(1, -1)
            out = model(input_ids=cont, past_key_values=prompt_out.past_key_values, use_cache=False)
            logits = out.logits[0].float()
            for pos, label in enumerate(labels[1:]):
                row = logits[pos]
                token_scores.append(row[int(label)] - torch.logsumexp(row, dim=-1))
        stacked = torch.stack(token_scores)
        score = stacked.mean() if normalize == "mean" else stacked.sum()
        return float(score.detach().cpu())


def run(cfg: FastEvalTextTraderCfg) -> dict[str, Any]:
    normalize = str(cfg.score_normalization).lower().strip()
    if normalize not in {"mean", "sum"}:
        raise ValueError("score_normalization must be mean or sum")
    rows = load_jsonl(cfg.eval_jsonl, max_samples=int(cfg.max_samples), sample_mode=cfg.sample_mode, seed=int(cfg.seed))
    holds = [int(x) for x in str(cfg.hold_candidates).split(",") if str(x).strip()]
    actions = _candidate_actions(holds)
    tokenizer, model = _load_text_model(cfg.model_name, cfg.adapter_dir)
    action_token_ids: list[list[int]] = []
    for action in actions:
        ids = tokenizer(_action_json(action), add_special_tokens=False)["input_ids"]
        if tokenizer.eos_token_id is not None:
            ids = ids + [int(tokenizer.eos_token_id)]
        action_token_ids.append(ids)
    preds: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    t0 = time.time()
    for i, row in enumerate(rows, start=1):
        prompt_text = _chat_prompt_text(tokenizer, str(row["prompt"]))
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
        scores = []
        for action, ids in zip(actions, action_token_ids):
            scores.append({"action": dict(action), "score": _candidate_logprob_from_prompt_cache(model, prompt_ids, ids, normalize)})
        best = max(scores, key=lambda x: float(x["score"]))
        preds.append(dict(best["action"]))
        score_rows.append({"date": row.get("date"), "signal_pos": row.get("signal_pos"), "scores": scores, "prediction": dict(best["action"]), "target": json.loads(str(row.get("target", "{}")))})
        if int(cfg.progress_every) > 0 and i % int(cfg.progress_every) == 0:
            print(json.dumps({"scored": i, "rows": len(rows), "elapsed_sec": round(time.time() - t0, 2)}, ensure_ascii=False), flush=True)
    pred_path = Path(cfg.predictions_output)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred_path.write_text("\n".join(json.dumps({"date": r.get("date"), "signal_pos": r.get("signal_pos"), "prediction": p, "target": json.loads(str(r.get("target", "{}")))}, ensure_ascii=False, sort_keys=True) for r, p in zip(rows, preds)) + "\n")
    report = {
        "config": asdict(cfg),
        "model_name": resolve_vlm_model_alias(cfg.model_name, prefer_latest=True),
        "rows": len(rows),
        "elapsed_sec": time.time() - t0,
        "metrics": _metrics(rows, preds),
        "score_rows": score_rows,
    }
    Path(cfg.report_output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.report_output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return {k: v for k, v in report.items() if k != "score_rows"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fast KV-cache candidate-logprob text trader eval")
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--adapter-dir", required=True)
    p.add_argument("--predictions-output", required=True)
    p.add_argument("--report-output", required=True)
    p.add_argument("--model-name", default=FastEvalTextTraderCfg.model_name)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--sample-mode", choices=["sequential", "random", "balanced", "gate_balanced"], default="sequential")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hold-candidates", default=FastEvalTextTraderCfg.hold_candidates)
    p.add_argument("--score-normalization", choices=["mean", "sum"], default="mean")
    p.add_argument("--progress-every", type=int, default=25)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(FastEvalTextTraderCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
