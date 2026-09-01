"""Score pairwise PPOSM residual rows and assemble TP4-default routes."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from training.build_pposm_residual_action_data import (
    CANDIDATE_ACTIONS,
    DEFAULT_ACTION,
    LABELS,
)
from training.eval_text_label import _assert_adapter_matches_model, _chat_prompt_text
from training.train_text_sft import RECOMMENDED_TEXT_CAUSAL_LM_MODEL

DEFAULT_TRAIN_SCORES = Path(
    "results/pposm_conditional_residual_train_scores_2026-09-02.jsonl"
)
DEFAULT_OOS_SCORES = Path(
    "results/pposm_conditional_residual_oos_scores_2026-09-02.jsonl"
)
DEFAULT_THRESHOLD = Path(
    "results/pposm_conditional_residual_train_threshold_2026-09-02.json"
)
DEFAULT_PREDICTIONS = Path(
    "results/pposm_conditional_residual_oos_predictions_2026-09-02.jsonl"
)
DEFAULT_REPORT = Path(
    "results/pposm_conditional_residual_router_report_2026-09-02.json"
)


@dataclass(frozen=True)
class Config:
    train_scores: Path = DEFAULT_TRAIN_SCORES
    oos_scores: Path = DEFAULT_OOS_SCORES
    threshold_output: Path = DEFAULT_THRESHOLD
    predictions_output: Path = DEFAULT_PREDICTIONS
    report_output: Path = DEFAULT_REPORT


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"line {number} in {path} is not an object")
        rows.append(value)
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def _write_jsonl(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def validate_score_rows(rows: Sequence[dict[str, Any]]) -> None:
    identities = [row.get("identity") for row in rows]
    if not all(isinstance(identity, str) and identity for identity in identities):
        raise ValueError("score row must include a non-empty residual identity")
    if len(identities) != len(set(identities)):
        raise ValueError("score rows contain duplicate residual identities")
    by_base: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        base = row.get("base_identity")
        candidate = row.get("candidate_action")
        if not isinstance(base, str) or candidate not in CANDIDATE_ACTIONS:
            raise ValueError("score row must include base_identity and candidate_action")
        margin = float(row.get("switch_margin"))
        if not math.isfinite(margin):
            raise ValueError("switch_margin must be finite")
        by_base[base].append(str(candidate))
    for base, candidates in by_base.items():
        if tuple(sorted(candidates)) != tuple(sorted(CANDIDATE_ACTIONS)):
            raise ValueError(f"base identity {base} does not have exactly two candidate score rows")


def _threshold_candidates(margins: Sequence[float]) -> list[float]:
    finite = sorted({float(value) for value in margins if math.isfinite(float(value))})
    if not finite:
        return [0.0]
    return [math.nextafter(finite[0], -math.inf), *finite, math.nextafter(finite[-1], math.inf)]


def _route_train_scores(score_rows: Sequence[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    predictions, _ = assemble_routes(score_rows, {"threshold": threshold})
    advantage_by_pair = {
        (str(row["base_identity"]), str(row["candidate_action"])): float(row.get("residual_advantage", 0.0))
        for row in score_rows
    }
    routed: list[dict[str, Any]] = []
    for pred in predictions:
        route = str(pred["prediction"])
        utility = 0.0 if route == DEFAULT_ACTION else advantage_by_pair[(str(pred["identity"]), route)]
        routed.append({**pred, "selected_residual_utility": utility})
    return routed


def _materiality(predictions: Sequence[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(row["prediction"]) for row in predictions)
    total = len(predictions)
    non_default_counts = {action: counts.get(action, 0) for action in CANDIDATE_ACTIONS}
    non_default_total = sum(non_default_counts.values())
    return {
        "num_signals": total,
        "route_counts": dict(sorted(counts.items())),
        "non_default_count": non_default_total,
        "difference_rate_vs_always_tp4": non_default_total / total if total else 0.0,
        "non_default_counts": non_default_counts,
        "max_action_share": max(counts.values(), default=0) / total if total else 0.0,
        "passes": bool(
            total > 0
            and non_default_total >= 1
            and non_default_total / total >= 0.10
            and all(count >= 10 for count in non_default_counts.values() if count > 0)
            and max(counts.values(), default=0) / total <= 0.90
        ),
    }


def freeze_threshold(train_scores: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Freeze one global threshold using pre-2024 score margins only.

    The threshold is selected from all distinct train margins plus below-min and
    above-max sentinels.  It maximizes selected train residual utility subject to
    materiality; ties prefer the higher threshold, then deterministic serialized
    route stream.  If no threshold satisfies materiality, the above-max sentinel
    is frozen to produce the TP4 default stream.
    """
    validate_score_rows(train_scores)
    for row in train_scores:
        signal_time = datetime.fromisoformat(
            str(row.get("signal_time", row.get("date"))).replace("Z", "+00:00")
        )
        if (
            row.get("split") != "train"
            or row.get("window") != "pre_2024"
            or signal_time.year >= 2024
            or "|pre_2024|" not in str(row.get("base_identity"))
        ):
            raise ValueError("threshold selection rows must be pre-2024 train rows")
        advantage = float(row.get("residual_advantage"))
        if not math.isfinite(advantage):
            raise ValueError("train residual_advantage must be finite")
    margins = [float(row["switch_margin"]) for row in train_scores]
    candidates = _threshold_candidates(margins)
    evaluations: list[dict[str, Any]] = []
    feasible: list[dict[str, Any]] = []
    for threshold in candidates:
        routed = _route_train_scores(train_scores, threshold)
        utility_sum = sum(float(row["selected_residual_utility"]) for row in routed)
        materiality = _materiality(routed)
        route_stream = "|".join(str(row["prediction"]) for row in routed)
        item = {
            "threshold": float(threshold),
            "selected_residual_utility_sum": float(utility_sum),
            "materiality": materiality,
            "serialized_route_stream": route_stream,
        }
        evaluations.append(item)
        if materiality["passes"]:
            feasible.append(item)
    if feasible:
        best_utility = max(
            float(item["selected_residual_utility_sum"]) for item in feasible
        )
        utility_ties = [
            item
            for item in feasible
            if float(item["selected_residual_utility_sum"]) == best_utility
        ]
        best_threshold = max(float(item["threshold"]) for item in utility_ties)
        threshold_ties = [
            item
            for item in utility_ties
            if float(item["threshold"]) == best_threshold
        ]
        selected = min(
            threshold_ties, key=lambda item: str(item["serialized_route_stream"])
        )
        status = "feasible_train_materiality"
    else:
        selected = max(evaluations, key=lambda item: float(item["threshold"]))
        status = "no_feasible_train_materiality_default_tp4"
    selected_public = {k: v for k, v in selected.items() if k != "serialized_route_stream"}
    return {
        "protocol": "pposm_residual_train_only_threshold_v2",
        "default_action": DEFAULT_ACTION,
        "candidate_actions": list(CANDIDATE_ACTIONS),
        "selection_rule": "strict margin>threshold; dual switch selects larger raw margin; exact tie SKIP before TP12",
        "threshold_source": "all_distinct_pre2024_margins_plus_below_min_above_max_sentinels",
        "optimization": "maximize_train_selected_residual_utility_subject_to_materiality; ties higher_threshold_then_serialized_route_stream; no_feasible_above_max_tp4",
        "threshold": float(selected["threshold"]),
        "status": status,
        "train_rows": len(train_scores),
        "train_base_signals": len({row["base_identity"] for row in train_scores}),
        "train_pair_identity_sha256": hashlib.sha256(
            "\n".join(str(row["identity"]) for row in train_scores).encode("utf-8")
        ).hexdigest(),
        "candidate_thresholds": len(candidates),
        "selected_train_evaluation": selected_public,
    }


def assemble_routes(
    score_rows: Sequence[dict[str, Any]], threshold_spec: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validate_score_rows(score_rows)
    threshold = float(threshold_spec["threshold"])
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    first_order: list[str] = []
    for row in score_rows:
        base = str(row["base_identity"])
        if base not in grouped:
            first_order.append(base)
        grouped[base].append(dict(row))
    predictions: list[dict[str, Any]] = []
    candidate_tie_rank = {"SKIP": 0, "TP12": 1}
    for base in first_order:
        candidates = grouped[base]
        eligible = [row for row in candidates if float(row["switch_margin"]) > threshold]
        if eligible:
            chosen = max(
                eligible,
                key=lambda row: (
                    float(row["switch_margin"]),
                    -candidate_tie_rank[str(row["candidate_action"])],
                ),
            )
            route = str(chosen["candidate_action"])
            decision = "SWITCH"
            selected_margin = float(chosen["switch_margin"])
        else:
            route = DEFAULT_ACTION
            decision = "KEEP"
            selected_margin = max(float(row["switch_margin"]) for row in candidates)
        sample = candidates[0]
        predictions.append({
            "identity": base,
            "base_identity": base,
            "date": sample.get("date"),
            "signal_time": sample.get("signal_time", sample.get("date")),
            "signal_position": sample.get("signal_pos", sample.get("signal_position")),
            "signal_pos": sample.get("signal_pos", sample.get("signal_position")),
            "prediction": route,
            "residual_decision": decision,
            "selected_switch_margin": selected_margin,
            "threshold": threshold,
            "candidate_margins": dict(
                sorted(
                    (
                        str(row["candidate_action"]),
                        float(row["switch_margin"]),
                    )
                    for row in candidates
                )
            ),
        })
    counts = Counter(row["prediction"] for row in predictions)
    non_default = sum(row["prediction"] != DEFAULT_ACTION for row in predictions)
    report = {
        "threshold_spec": threshold_spec,
        "num_signals": len(predictions),
        "route_counts": dict(sorted(counts.items())),
        "non_default_count": non_default,
        "difference_rate_vs_always_tp4": non_default / len(predictions) if predictions else 0.0,
        "max_action_share": max(counts.values(), default=0) / len(predictions) if predictions else 0.0,
    }
    return predictions, report


def score_rows_with_adapter(
    rows: Sequence[dict[str, Any]],
    *,
    model_name: str,
    adapter_dir: str,
    score_normalization: str = "mean",
    load_in_4bit: bool = False,
) -> list[dict[str, Any]]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from utils import disable_transformers_allocator_warmup

    resolved = _assert_adapter_matches_model(model_name, adapter_dir)
    disable_transformers_allocator_warmup()
    tokenizer = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16",
        )
        if load_in_4bit
        else None
    )
    base = AutoModelForCausalLM.from_pretrained(
        resolved,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=quantization_config,
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    normalize = str(score_normalization).strip().lower()
    if normalize not in {"sum", "mean", "first_token"}:
        raise ValueError("score_normalization must be one of {'sum','mean','first_token'}")
    scored: list[dict[str, Any]] = []
    labels = list(LABELS)
    for row in rows:
        prompt_ids = tokenizer(_chat_prompt_text(tokenizer, str(row["prompt"])), add_special_tokens=False)["input_ids"]
        sequences: list[list[int]] = []
        spans: list[tuple[int, int]] = []
        for label in labels:
            label_ids = tokenizer(label, add_special_tokens=False)["input_ids"]
            if tokenizer.eos_token_id is not None:
                label_ids = label_ids + [int(tokenizer.eos_token_id)]
            start = len(prompt_ids)
            end = start + len(label_ids)
            sequences.append(prompt_ids + label_ids)
            spans.append((start, end))
        encoded = tokenizer.pad({"input_ids": sequences}, return_tensors="pt")
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        scores: dict[str, float] = {}
        for i, (start, end) in enumerate(spans):
            positions = torch.arange(start - 1, end - 1, device=log_probs.device)
            label_tensor = input_ids[i, start:end]
            token_scores = log_probs[i, positions, label_tensor]
            if normalize == "first_token":
                score = token_scores[0]
            else:
                score = token_scores.sum() if normalize == "sum" else token_scores.mean()
            scores[labels[i]] = float(score.detach().cpu())
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        scored.append({
            "identity": metadata.get("identity"),
            "base_identity": metadata.get("base_identity"),
            "candidate_action": metadata.get("candidate_action"),
            "split": row.get("split"),
            "window": metadata.get("window"),
            "target": row.get("target"),
            "date": metadata.get("signal_time"),
            "signal_time": metadata.get("signal_time"),
            "signal_pos": metadata.get("signal_position"),
            "scores": scores,
            "switch_margin": float(scores["SWITCH"] - scores["KEEP"]),
            "residual_advantage": metadata.get("residual_advantage"),
        })
    validate_score_rows(scored)
    return scored


def write_train_threshold(
    train_scores_path: Path, threshold_output: Path
) -> dict[str, Any]:
    train_scores = _load_jsonl(train_scores_path)
    threshold_spec = freeze_threshold(train_scores)
    threshold_spec["selection_inputs"] = {
        "train_scores": {
            "path": str(train_scores_path),
            "sha256": hashlib.sha256(train_scores_path.read_bytes()).hexdigest(),
        }
    }
    threshold_spec["future_can_rank_repair_or_reselect"] = False
    threshold_output.parent.mkdir(parents=True, exist_ok=True)
    threshold_output.write_text(
        json.dumps(
            threshold_spec,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return threshold_spec


def route_oos_scores(
    *,
    oos_scores_path: Path,
    threshold_path: Path,
    predictions_output: Path,
    report_output: Path,
) -> dict[str, Any]:
    threshold_spec = _load_json(threshold_path)
    if "train_only" not in str(threshold_spec.get("protocol", "")):
        raise ValueError("threshold artifact is not train-only")
    oos_scores = _load_jsonl(oos_scores_path)
    predictions, route_report = assemble_routes(oos_scores, threshold_spec)
    _write_jsonl(predictions_output, predictions)
    report = {
        "protocol": "pposm_residual_router_report_v1",
        "config": {
            "oos_scores": str(oos_scores_path),
            "threshold": str(threshold_path),
            "predictions_output": str(predictions_output),
            "report_output": str(report_output),
        },
        **route_report,
    }
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def run(cfg: Config) -> dict[str, Any]:
    write_train_threshold(cfg.train_scores, cfg.threshold_output)
    return route_oos_scores(
        oos_scores_path=cfg.oos_scores,
        threshold_path=cfg.threshold_output,
        predictions_output=cfg.predictions_output,
        report_output=cfg.report_output,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    score = sub.add_parser("score")
    score.add_argument("--input-jsonl", required=True)
    score.add_argument("--scores-output", required=True)
    score.add_argument("--model-name", default=RECOMMENDED_TEXT_CAUSAL_LM_MODEL)
    score.add_argument("--adapter-dir", required=True)
    score.add_argument("--score-normalization", choices=["sum", "mean", "first_token"], default="mean")
    score.add_argument("--load-in-4bit", action="store_true")
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--train-scores", type=Path, default=DEFAULT_TRAIN_SCORES)
    freeze.add_argument("--threshold-output", type=Path, default=DEFAULT_THRESHOLD)
    route = sub.add_parser("route")
    route.add_argument("--oos-scores", type=Path, default=DEFAULT_OOS_SCORES)
    route.add_argument("--threshold", type=Path, default=DEFAULT_THRESHOLD)
    route.add_argument("--predictions-output", type=Path, default=DEFAULT_PREDICTIONS)
    route.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    assemble = sub.add_parser("assemble")
    assemble.add_argument("--train-scores", type=Path, default=DEFAULT_TRAIN_SCORES)
    assemble.add_argument("--oos-scores", type=Path, default=DEFAULT_OOS_SCORES)
    assemble.add_argument("--threshold-output", type=Path, default=DEFAULT_THRESHOLD)
    assemble.add_argument("--predictions-output", type=Path, default=DEFAULT_PREDICTIONS)
    assemble.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "score":
        rows = _load_jsonl(args.input_jsonl)
        scored = score_rows_with_adapter(
            rows,
            model_name=args.model_name,
            adapter_dir=args.adapter_dir,
            score_normalization=args.score_normalization,
            load_in_4bit=args.load_in_4bit,
        )
        _write_jsonl(args.scores_output, scored)
        print(json.dumps({"scores_output": args.scores_output, "rows": len(scored)}, indent=2))
        return
    if args.command == "freeze":
        print(
            json.dumps(
                write_train_threshold(args.train_scores, args.threshold_output),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "route":
        print(
            json.dumps(
                route_oos_scores(
                    oos_scores_path=args.oos_scores,
                    threshold_path=args.threshold,
                    predictions_output=args.predictions_output,
                    report_output=args.report_output,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return
    print(json.dumps(run(Config(**{k: v for k, v in vars(args).items() if k != "command"})), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
