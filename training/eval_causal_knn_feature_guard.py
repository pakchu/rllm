"""Evaluate fixed prompt-feature guards on causal KNN path-value candidates."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from training.eval_causal_knn_path_value_scorer import CausalKNNPathValueConfig, score_eval
from training.eval_pairwise_candidate_backtest import CandidateBacktestConfig, load_jsonl, load_market, simulate_candidates
from training.kimchi_flow_pairwise_dataset import parse_prompt


def _parse_guard(text: str) -> tuple[str, str, float]:
    for op in ("<=", ">="):
        if op in text:
            key, value = text.split(op, 1)
            return key.strip(), op, float(value.strip())
    raise ValueError(f"guard must contain <= or >=: {text}")


def _attach_features(candidates: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for cand, row in zip(candidates, eval_rows):
        _, nums = parse_prompt(row["prompt"])
        out.append({**cand, "prompt_features": {str(k): float(v) for k, v in nums.items()}})
    return out


def _filter(candidates: list[dict[str, Any]], guards: list[str]) -> list[dict[str, Any]]:
    parsed = [_parse_guard(g) for g in guards]
    kept = []
    for cand in candidates:
        nums = cand.get("prompt_features", {}) or {}
        ok = True
        for key, op, value in parsed:
            x = float(nums.get(key, 0.0))
            if op == "<=":
                ok = ok and x <= value
            else:
                ok = ok and x >= value
        if ok:
            kept.append(cand)
    return kept


def run(args: argparse.Namespace) -> dict[str, Any]:
    market = load_market(args.market_csv)
    ref_rows = load_jsonl(args.reference_jsonl)
    eval_rows = load_jsonl(args.eval_jsonl)
    scorer_cfg = CausalKNNPathValueConfig(
        reference_jsonl=args.reference_jsonl,
        eval_jsonl=args.eval_jsonl,
        market_csv=args.market_csv,
        output=args.output,
        k=args.k,
        score_threshold=args.score_threshold,
        target_metric=args.target_metric,
        hold_bars=args.hold_bars,
        entry_delay_bars=args.entry_delay_bars,
        leverage=args.leverage,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        mae_penalty=args.mae_penalty,
        mfe_bonus=args.mfe_bonus,
        same_bucket=not args.no_same_bucket,
        min_bucket_refs=args.min_bucket_refs,
    )
    candidates = _attach_features(score_eval(ref_rows, eval_rows, market, scorer_cfg), eval_rows)
    guarded = _filter(candidates, args.guard)
    sim_cfg = CandidateBacktestConfig(
        market_csv=args.market_csv,
        pairwise_jsonl="",
        predictions_jsonl="",
        output="",
        score_threshold=args.score_threshold,
        hold_bars=args.hold_bars,
        entry_delay_bars=args.entry_delay_bars,
        leverage=args.leverage,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
    )
    result = simulate_candidates(guarded, market, sim_cfg)
    result["config"] = {**asdict(scorer_cfg), "guards": list(args.guard)}
    result["guard_summary"] = {
        "pre_guard_candidates": len(candidates),
        "post_guard_candidates": len(guarded),
        "guards": list(args.guard),
    }
    result["guarded_candidates"] = guarded
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate causal KNN path-value candidates with fixed prompt-feature guards")
    p.add_argument("--reference-jsonl", required=True)
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--guard", action="append", default=[], help="Feature guard such as drawdown_acceleration_6h<=0.003")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--score-threshold", type=float, default=-0.5)
    p.add_argument("--target-metric", default="path_net_pct")
    p.add_argument("--hold-bars", type=int, default=288)
    p.add_argument("--entry-delay-bars", type=int, default=1)
    p.add_argument("--leverage", type=float, default=0.5)
    p.add_argument("--fee-rate", type=float, default=0.0004)
    p.add_argument("--slippage-rate", type=float, default=0.0001)
    p.add_argument("--mae-penalty", type=float, default=1.0)
    p.add_argument("--mfe-bonus", type=float, default=0.0)
    p.add_argument("--no-same-bucket", action="store_true")
    p.add_argument("--min-bucket-refs", type=int, default=20)
    return p.parse_args()


def main() -> None:
    out = run(parse_args())
    print(json.dumps({"guard_summary": out["guard_summary"], "sim": out["sim"], "trade_stats": out["trade_stats"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
