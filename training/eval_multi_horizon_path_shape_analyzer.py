"""Evaluate/export multi-horizon path-shape analyzer JSON predictions."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from models.option_b_vlm import RECOMMENDED_VLM_MODEL, resolve_vlm_model_alias
from training.multi_horizon_edge_report import parse_horizons
from training.train_text_sft import load_jsonl
from utils import disable_transformers_allocator_warmup

TOP_VALID = {
    "trend_side": {"LONG", "SHORT", "NONE"},
    "direction_stability": {"TREND_STABLE", "FADE_STABLE", "HORIZON_CONFLICT", "MIXED_WEAK", "NO_STABLE_EDGE"},
    "reversal_pressure": {"HIGH", "MEDIUM", "LOW"},
    "risk_profile": {"LOW_PATH_RISK", "MIXED_PATH_RISK", "HIGH_PATH_RISK", "EXTREME_PATH_RISK"},
}
TOP_DEFAULTS = {
    "trend_side": "NONE",
    "direction_stability": "NO_STABLE_EDGE",
    "reversal_pressure": "LOW",
    "risk_profile": "MIXED_PATH_RISK",
}
HORIZON_VALID = {
    "trend_return_bucket": {"STRONG_POSITIVE", "POSITIVE", "WEAK_POSITIVE", "FLAT_NEGATIVE", "NEGATIVE", "STRONG_NEGATIVE", "UNAVAILABLE"},
    "fade_return_bucket": {"STRONG_POSITIVE", "POSITIVE", "WEAK_POSITIVE", "FLAT_NEGATIVE", "NEGATIVE", "STRONG_NEGATIVE", "UNAVAILABLE"},
    "trend_mae_bucket": {"LOW", "MEDIUM", "HIGH", "EXTREME", "UNAVAILABLE"},
    "fade_mae_bucket": {"LOW", "MEDIUM", "HIGH", "EXTREME", "UNAVAILABLE"},
    "relative_edge": {"TREND_STRONGER", "TREND_SLIGHTLY_STRONGER", "FADE_STRONGER", "FADE_SLIGHTLY_STRONGER", "NO_CLEAR_EDGE", "NO_TREND_SIDE"},
    "best_path": {"TREND", "FADE", "MIXED", "NONE"},
}
HORIZON_DEFAULTS = {
    "trend_return_bucket": "UNAVAILABLE",
    "fade_return_bucket": "UNAVAILABLE",
    "trend_mae_bucket": "UNAVAILABLE",
    "fade_mae_bucket": "UNAVAILABLE",
    "relative_edge": "NO_TREND_SIDE",
    "best_path": "NONE",
}


def _extract_json_object(text: str) -> dict[str, Any]:
    raw = str(text).strip()
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            return {}
        try:
            obj = json.loads(match.group(0))
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}


def _norm_choice(value: Any, valid: set[str], default: str) -> str:
    val = str(value).upper()
    return val if val in valid else default


def parse_path_shape_json(text: str, *, horizons: tuple[int, ...] = (36, 72, 144, 288, 432)) -> dict[str, Any]:
    obj = _extract_json_object(text)
    out: dict[str, Any] = {}
    for key, valid in TOP_VALID.items():
        out[key] = _norm_choice(obj.get(key, TOP_DEFAULTS[key]), valid, TOP_DEFAULTS[key])
    raw_horizons = obj.get("horizons") if isinstance(obj.get("horizons"), dict) else {}
    horizons_out: dict[str, dict[str, Any]] = {}
    for h in horizons:
        hkey = str(int(h))
        hobj = raw_horizons.get(hkey) if isinstance(raw_horizons, dict) else {}
        if not isinstance(hobj, dict):
            hobj = {}
        parsed_h: dict[str, Any] = {}
        for key, valid in HORIZON_VALID.items():
            parsed_h[key] = _norm_choice(hobj.get(key, HORIZON_DEFAULTS[key]), valid, HORIZON_DEFAULTS[key])
        try:
            count = int(hobj.get("tradable_path_count", 0))
        except Exception:
            count = 0
        parsed_h["tradable_path_count"] = min(2, max(0, count))
        horizons_out[hkey] = parsed_h
    out["horizons"] = horizons_out
    raw_counts = obj.get("summary_counts") if isinstance(obj.get("summary_counts"), dict) else {}
    out["summary_counts"] = {
        key: max(0, int(raw_counts.get(key, 0) or 0)) if str(raw_counts.get(key, 0) or 0).lstrip("-").isdigit() else 0
        for key in ("trend_wins", "fade_wins", "mixed", "none")
    }
    return out


def _confusion_add(confusion: dict[str, int], target: str, pred: str) -> None:
    key = f"target={target}|pred={pred}"
    confusion[key] = confusion.get(key, 0) + 1


def _choice_metric(rows: list[dict[str, Any]], preds: list[dict[str, Any]], key: str, horizons: tuple[int, ...]) -> dict[str, Any]:
    correct = 0
    confusion: dict[str, int] = {}
    for row, pred in zip(rows, preds):
        target = parse_path_shape_json(str(row["target"]), horizons=horizons)[key]
        pval = pred[key]
        correct += int(target == pval)
        _confusion_add(confusion, target, pval)
    return {"accuracy": correct / max(1, len(rows)), "confusion": dict(sorted(confusion.items()))}


def _horizon_metric(rows: list[dict[str, Any]], preds: list[dict[str, Any]], hkey: str, key: str, horizons: tuple[int, ...]) -> dict[str, Any]:
    correct = 0
    confusion: dict[str, int] = {}
    for row, pred in zip(rows, preds):
        target = parse_path_shape_json(str(row["target"]), horizons=horizons)["horizons"][hkey][key]
        pval = pred["horizons"][hkey][key]
        correct += int(target == pval)
        _confusion_add(confusion, str(target), str(pval))
    return {"accuracy": correct / max(1, len(rows)), "confusion": dict(sorted(confusion.items()))}


def _metrics(rows: list[dict[str, Any]], preds: list[dict[str, Any]], horizons: tuple[int, ...]) -> dict[str, Any]:
    top = {key: _choice_metric(rows, preds, key, horizons) for key in TOP_VALID}
    exact_top = 0
    exact_all = 0
    horizon_key_totals = {key: 0 for key in HORIZON_VALID}
    horizon_key_counts = {key: 0 for key in HORIZON_VALID}
    horizon_metrics: dict[str, Any] = {}
    for row, pred in zip(rows, preds):
        target = parse_path_shape_json(str(row["target"]), horizons=horizons)
        exact_top += int(all(target[k] == pred[k] for k in TOP_VALID))
        exact_all += int(target == pred)
        for h in horizons:
            hkey = str(int(h))
            for key in HORIZON_VALID:
                horizon_key_totals[key] += int(target["horizons"][hkey][key] == pred["horizons"][hkey][key])
                horizon_key_counts[key] += 1
    for h in horizons:
        hkey = str(int(h))
        horizon_metrics[hkey] = {key: _horizon_metric(rows, preds, hkey, key, horizons) for key in HORIZON_VALID}
    return {
        "num_samples": len(rows),
        "exact_top_level_accuracy": exact_top / max(1, len(rows)),
        "exact_all_keys_accuracy": exact_all / max(1, len(rows)),
        "top_level": top,
        "horizon_key_micro_accuracy": {key: horizon_key_totals[key] / max(1, horizon_key_counts[key]) for key in HORIZON_VALID},
        "horizons": horizon_metrics,
    }


def _format_prompt(row: dict[str, Any], tokenizer: Any) -> str:
    messages = [{"role": "user", "content": str(row["prompt"])}]
    if getattr(tokenizer, "chat_template", None):
        return str(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    return f"<|user|>\n{row['prompt']}\n<|assistant|>\n"


def _generate_predictions(
    rows: list[dict[str, Any]],
    *,
    horizons: tuple[int, ...],
    model_name: str,
    adapter_dir: str,
    max_new_tokens: int,
    batch_size: int,
    progress_every: int,
    prediction_output: str = "",
) -> list[dict[str, Any]]:
    disable_transformers_allocator_warmup()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved = resolve_vlm_model_alias(model_name, prefer_latest=True)
    tokenizer = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(resolved, trust_remote_code=True, device_map="auto")
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    device = next(model.parameters()).device
    preds: list[dict[str, Any]] = []
    stream = None
    if prediction_output:
        out_path = Path(prediction_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        stream = out_path.open("w")
    try:
        for start in range(0, len(rows), max(1, int(batch_size))):
            batch = rows[start : start + max(1, int(batch_size))]
            texts = [_format_prompt(row, tokenizer) for row in batch]
            inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)
            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    max_new_tokens=int(max_new_tokens),
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            prompt_width = inputs["input_ids"].shape[-1]
            for idx, generated_ids in enumerate(out[:, prompt_width:]):
                generated = tokenizer.decode(generated_ids, skip_special_tokens=True)
                pred = parse_path_shape_json(generated, horizons=horizons)
                preds.append(pred)
                if stream is not None:
                    merged = dict(batch[idx])
                    merged["prediction"] = json.dumps(pred, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                    stream.write(json.dumps(merged, ensure_ascii=False) + "\n")
            if stream is not None:
                stream.flush()
            done = min(start + len(batch), len(rows))
            if progress_every > 0 and (done == len(rows) or done % progress_every == 0):
                print(f"generated {done}/{len(rows)} path-shape predictions", flush=True)
    finally:
        if stream is not None:
            stream.close()
    return preds


def write_prediction_jsonl(path: str | Path, rows: list[dict[str, Any]], preds: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row, pred in zip(rows, preds):
            merged = dict(row)
            merged["prediction"] = json.dumps(pred, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            f.write(json.dumps(merged, ensure_ascii=False) + "\n")


def evaluate_path_shape_analyzer(
    *,
    eval_jsonl: str,
    output: str,
    prediction_output: str = "",
    model_name: str = RECOMMENDED_VLM_MODEL,
    adapter_dir: str = "",
    max_samples: int = 0,
    sample_mode: str = "sequential",
    seed: int = 42,
    prediction_mode: str = "target_echo",
    max_new_tokens: int = 1536,
    batch_size: int = 2,
    progress_every: int = 64,
    hold_bars_list: str = "36,72,144,288,432",
) -> dict[str, Any]:
    horizons = parse_horizons(hold_bars_list)
    rows = load_jsonl(eval_jsonl, max_samples=max_samples, sample_mode=sample_mode, seed=seed)
    if prediction_mode == "target_echo":
        preds = [parse_path_shape_json(str(r["target"]), horizons=horizons) for r in rows]
    elif prediction_mode == "model":
        if not adapter_dir:
            raise ValueError("adapter_dir is required for prediction_mode=model")
        preds = _generate_predictions(
            rows,
            horizons=horizons,
            model_name=model_name,
            adapter_dir=adapter_dir,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            progress_every=progress_every,
            prediction_output=prediction_output,
        )
    else:
        raise ValueError("prediction_mode must be one of {'target_echo','model'}")
    if prediction_output and prediction_mode != "model":
        write_prediction_jsonl(prediction_output, rows, preds)
    report = {
        "eval_jsonl": str(Path(eval_jsonl).resolve()),
        "prediction_output": prediction_output,
        "model_name": resolve_vlm_model_alias(model_name, prefer_latest=True),
        "adapter_dir": adapter_dir,
        "prediction_mode": prediction_mode,
        "hold_bars_list": list(horizons),
        "metrics": _metrics(rows, preds, horizons),
        "leakage_guard": {
            "target_echo_is_oracle_only": prediction_mode == "target_echo",
            "model_mode_uses_prompt_only": prediction_mode == "model",
            "path_shape_targets_use_future_ohlc": True,
            "target_is_path_shape_not_final_action": True,
        },
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate/export multi-horizon path-shape analyzer predictions")
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--prediction-output", default="")
    p.add_argument("--model-name", default=RECOMMENDED_VLM_MODEL)
    p.add_argument("--adapter-dir", default="")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--sample-mode", choices=["sequential", "random", "balanced"], default="sequential")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prediction-mode", choices=["target_echo", "model"], default="target_echo")
    p.add_argument("--max-new-tokens", type=int, default=1536)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--progress-every", type=int, default=64)
    p.add_argument("--hold-bars-list", default="36,72,144,288,432")
    return p.parse_args()


def main() -> None:
    print(json.dumps(evaluate_path_shape_analyzer(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
