"""Walk-forward selection/evaluation for causal KNN path-value guards.

Each fold enforces chronological separation:
1. validation candidates are scored against rows before validation start;
2. scorer/guard parameters are selected on validation only;
3. evaluation candidates are scored against rows before evaluation start;
4. the selected parameters are applied unchanged to evaluation.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.eval_causal_knn_path_value_scorer import CausalKNNPathValueConfig, score_eval
from training.eval_pairwise_candidate_backtest import CandidateBacktestConfig, load_jsonl, load_market, simulate_candidates
from training.kimchi_flow_pairwise_dataset import parse_prompt


def load_rows(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(load_jsonl(path))
    rows.sort(key=lambda r: pd.to_datetime(r["date"]))
    return rows


def rows_between(rows: list[dict[str, Any]], start: str | None, end: str | None) -> list[dict[str, Any]]:
    start_ts = pd.to_datetime(start) if start else None
    end_ts = pd.to_datetime(end) if end else None
    out = []
    for row in rows:
        ts = pd.to_datetime(row["date"])
        if start_ts is not None and ts < start_ts:
            continue
        if end_ts is not None and ts >= end_ts:
            continue
        out.append(row)
    return out


def attach_features(candidates: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for cand, row in zip(candidates, eval_rows):
        _, nums = parse_prompt(row["prompt"])
        out.append({**cand, "prompt_features": {str(k): float(v) for k, v in nums.items()}})
    return out


def score_period(
    *,
    reference_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    market: pd.DataFrame,
    market_csv: str,
    k: int,
    score_threshold: float,
    target_metric: str,
    hold_bars: int,
    entry_delay_bars: int,
    leverage: float,
    fee_rate: float,
    slippage_rate: float,
    same_bucket: bool,
    min_bucket_refs: int,
) -> list[dict[str, Any]]:
    cfg = CausalKNNPathValueConfig(
        reference_jsonl="<in-memory>",
        eval_jsonl="<in-memory>",
        market_csv=market_csv,
        output="",
        k=int(k),
        score_threshold=float(score_threshold),
        target_metric=target_metric,  # type: ignore[arg-type]
        hold_bars=int(hold_bars),
        entry_delay_bars=int(entry_delay_bars),
        leverage=float(leverage),
        fee_rate=float(fee_rate),
        slippage_rate=float(slippage_rate),
        same_bucket=bool(same_bucket),
        min_bucket_refs=int(min_bucket_refs),
    )
    return attach_features(score_eval(reference_rows, eval_rows, market, cfg), eval_rows)


def parse_guard(text: str) -> tuple[str, str, float]:
    for op in ("<=", ">="):
        if op in text:
            a, b = text.split(op, 1)
            return a.strip(), op, float(b.strip())
    raise ValueError(f"invalid guard: {text}")


def apply_guard(candidates: list[dict[str, Any]], guard: str) -> list[dict[str, Any]]:
    if guard == "NONE":
        return list(candidates)
    key, op, val = parse_guard(guard)
    out = []
    for cand in candidates:
        x = float((cand.get("prompt_features", {}) or {}).get(key, 0.0))
        if (op == "<=" and x <= val) or (op == ">=" and x >= val):
            out.append(cand)
    return out


def simulate(candidates: list[dict[str, Any]], args: argparse.Namespace, score_threshold: float) -> dict[str, Any]:
    cfg = CandidateBacktestConfig(
        market_csv=args.market_csv,
        pairwise_jsonl="",
        predictions_jsonl="",
        output="",
        score_threshold=float(score_threshold),
        hold_bars=int(args.hold_bars),
        entry_delay_bars=int(args.entry_delay_bars),
        leverage=float(args.leverage),
        fee_rate=float(args.fee_rate),
        slippage_rate=float(args.slippage_rate),
    )
    return simulate_candidates(candidates, args._market, cfg)


def guard_candidates(scored: list[dict[str, Any]], specs: list[str]) -> list[str]:
    guards = ["NONE"]
    for spec in specs:
        key, op, qs = spec.split(":", 2)
        vals = np.asarray([float((c.get("prompt_features", {}) or {}).get(key, 0.0)) for c in scored], dtype=float)
        if vals.size == 0:
            continue
        for qtxt in qs.split(","):
            q = float(qtxt)
            val = float(np.quantile(vals, q))
            guards.append(f"{key}{op}{val:.10g}")
    return guards


def choose_trial(trials: list[dict[str, Any]], min_trades: int) -> dict[str, Any]:
    eligible = [t for t in trials if int(t["sim"]["trade_entries"]) >= int(min_trades)]
    pool = eligible or trials
    return max(
        pool,
        key=lambda t: (
            float(t["sim"].get("cagr_to_strict_mdd", -1e9)),
            -float(t["trade_stats"].get("p_value_mean_ret_approx", 1.0)),
            int(t["sim"].get("trade_entries", 0)),
        ),
    )


def passes_validation_gate(trial: dict[str, Any], args: argparse.Namespace) -> bool:
    sim = trial.get("sim", {})
    stats = trial.get("trade_stats", {})
    return (
        int(sim.get("trade_entries", 0)) >= int(args.min_val_trades)
        and float(sim.get("cagr_to_strict_mdd", -1e9)) >= float(args.min_val_ratio)
        and float(sim.get("strict_mdd_pct", 1e9)) <= float(args.max_val_mdd)
        and float(stats.get("p_value_mean_ret_approx", 1.0)) <= float(args.max_val_p)
    )


def no_trade_eval(args: argparse.Namespace, eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if eval_rows:
        start = str(pd.to_datetime(eval_rows[0]["date"]))
        end = str(pd.to_datetime(eval_rows[-1]["date"]))
        years = max(1.0 / 365.25, (pd.to_datetime(end) - pd.to_datetime(start)).days / 365.25)
    else:
        start = end = ""
        years = 1.0 / 365.25
    return {
        "sim": {
            "ret_pct": 0.0,
            "cagr_pct": 0.0,
            "strict_mdd_pct": 0.0,
            "cagr_to_strict_mdd": 0.0,
            "trade_entries": 0,
            "side_counts": {"LONG": 0, "SHORT": 0},
            "skipped_missing_bars": 0,
            "hold_bars": int(args.hold_bars),
            "entry_delay_bars": int(args.entry_delay_bars),
            "return_application": "validation_gate_no_trade",
            "period": {"start": start, "end": end, "years": years},
        },
        "trade_stats": {
            "n_trades": 0,
            "mean_trade_ret_pct": 0.0,
            "std_trade_ret_pct": 0.0,
            "t_stat_like": 0.0,
            "p_value_mean_ret_approx": 1.0,
            "ci95_mean_trade_ret_pct": [0.0, 0.0],
            "effect_size_d": 0.0,
            "n_required_for_80pct_power_alpha5pct": None,
            "n_gap_to_power_rule": None,
        },
    }


def run_fold(rows: list[dict[str, Any]], fold: dict[str, str], args: argparse.Namespace) -> dict[str, Any]:
    ref_for_val = rows_between(rows, None, fold["val_start"])
    val_rows = rows_between(rows, fold["val_start"], fold["eval_start"])
    ref_for_eval = rows_between(rows, None, fold["eval_start"])
    eval_rows = rows_between(rows, fold["eval_start"], fold["eval_end"])
    if not ref_for_val or not val_rows or not ref_for_eval or not eval_rows:
        return {"fold": fold, "skipped": True, "counts": {"ref_val": len(ref_for_val), "val": len(val_rows), "ref_eval": len(ref_for_eval), "eval": len(eval_rows)}}

    trials = []
    scored_cache: dict[tuple[int, float], list[dict[str, Any]]] = {}
    for k in args.k:
        for th in args.score_threshold:
            scored = score_period(
                reference_rows=ref_for_val,
                eval_rows=val_rows,
                market=args._market,
                market_csv=args.market_csv,
                k=k,
                score_threshold=th,
                target_metric=args.target_metric,
                hold_bars=args.hold_bars,
                entry_delay_bars=args.entry_delay_bars,
                leverage=args.leverage,
                fee_rate=args.fee_rate,
                slippage_rate=args.slippage_rate,
                same_bucket=not args.no_same_bucket,
                min_bucket_refs=args.min_bucket_refs,
            )
            scored_cache[(k, th)] = scored
            for guard in guard_candidates(scored, args.guard_spec):
                guarded = apply_guard(scored, guard)
                result = simulate(guarded, args, th)
                trials.append({"k": k, "score_threshold": th, "guard": guard, "sim": result["sim"], "trade_stats": result["trade_stats"], "candidate_count": len(scored), "guarded_count": len(guarded)})
    chosen = choose_trial(trials, args.min_val_trades)
    validation_gate_passed = passes_validation_gate(chosen, args)
    if validation_gate_passed:
        eval_scored = score_period(
            reference_rows=ref_for_eval,
            eval_rows=eval_rows,
            market=args._market,
            market_csv=args.market_csv,
            k=int(chosen["k"]),
            score_threshold=float(chosen["score_threshold"]),
            target_metric=args.target_metric,
            hold_bars=args.hold_bars,
            entry_delay_bars=args.entry_delay_bars,
            leverage=args.leverage,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
            same_bucket=not args.no_same_bucket,
            min_bucket_refs=args.min_bucket_refs,
        )
        eval_guarded = apply_guard(eval_scored, str(chosen["guard"]))
        eval_result = simulate(eval_guarded, args, float(chosen["score_threshold"]))
        eval_candidate_count = len(eval_scored)
        eval_guarded_count = len(eval_guarded)
    else:
        eval_result = no_trade_eval(args, eval_rows)
        eval_candidate_count = len(eval_rows)
        eval_guarded_count = 0
    return {
        "fold": fold,
        "skipped": False,
        "counts": {"ref_val": len(ref_for_val), "val": len(val_rows), "ref_eval": len(ref_for_eval), "eval": len(eval_rows)},
        "selected_on_val": chosen,
        "validation_gate_passed": validation_gate_passed,
        "validation_gate": {"min_val_ratio": args.min_val_ratio, "max_val_mdd": args.max_val_mdd, "max_val_p": args.max_val_p, "min_val_trades": args.min_val_trades},
        "eval_fixed": {
            "sim": eval_result["sim"],
            "trade_stats": eval_result["trade_stats"],
            "candidate_count": eval_candidate_count,
            "guarded_count": eval_guarded_count,
            **({"executed": eval_result.get("executed", [])} if getattr(args, "include_executed", False) else {}),
        },
        "top_val_trials": sorted(trials, key=lambda t: float(t["sim"].get("cagr_to_strict_mdd", -1e9)), reverse=True)[:10],
        "leakage_guard": {
            "val_scored_against_rows_before_val_start": True,
            "eval_scored_against_rows_before_eval_start": True,
            "eval_parameters_fixed_from_val": True,
        },
    }


def default_folds() -> list[dict[str, str]]:
    return [
        {"name": "2023_to_2024", "val_start": "2023-01-01", "eval_start": "2024-01-01", "eval_end": "2025-01-01"},
        {"name": "2024h1_to_2024h2", "val_start": "2024-01-01", "eval_start": "2024-07-01", "eval_end": "2025-01-01"},
        {"name": "2024_to_2025", "val_start": "2024-01-01", "eval_start": "2025-01-01", "eval_end": "2026-01-01"},
        {"name": "2025h1_to_2025h2", "val_start": "2025-01-01", "eval_start": "2025-07-01", "eval_end": "2026-01-01"},
    ]


def load_folds(path: str | None) -> list[dict[str, str]]:
    if not path:
        return default_folds()
    obj = json.loads(Path(path).read_text())
    folds = obj.get("folds", obj)
    if not isinstance(folds, list):
        raise ValueError("folds JSON must be a list or an object with a folds list")
    required = {"name", "val_start", "eval_start", "eval_end"}
    out = []
    for fold in folds:
        missing = required - set(fold)
        if missing:
            raise ValueError(f"fold missing keys {sorted(missing)}: {fold}")
        out.append({k: str(fold[k]) for k in sorted(required)})
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walk-forward causal KNN guard evaluator")
    p.add_argument("--rows-jsonl", action="append", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--folds-json", default="", help="Optional JSON list/object defining non-overlapping or custom folds.")
    p.add_argument("--target-metric", default="path_net_pct")
    p.add_argument("--k", type=int, action="append", default=[])
    p.add_argument("--score-threshold", type=float, action="append", default=[])
    p.add_argument("--guard-spec", action="append", default=[])
    p.add_argument("--min-val-trades", type=int, default=50)
    p.add_argument("--min-val-ratio", type=float, default=-1e9)
    p.add_argument("--max-val-mdd", type=float, default=1e9)
    p.add_argument("--max-val-p", type=float, default=1.0)
    p.add_argument("--hold-bars", type=int, default=288)
    p.add_argument("--entry-delay-bars", type=int, default=1)
    p.add_argument("--leverage", type=float, default=0.5)
    p.add_argument("--fee-rate", type=float, default=0.0004)
    p.add_argument("--slippage-rate", type=float, default=0.0001)
    p.add_argument("--no-same-bucket", action="store_true")
    p.add_argument("--min-bucket-refs", type=int, default=20)
    p.add_argument("--include-executed", action="store_true", help="Persist executed eval trades for aggregate audits.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.k:
        args.k = [10]
    if not args.score_threshold:
        args.score_threshold = [-0.5]
    if not args.guard_spec:
        args.guard_spec = [
            "drawdown_acceleration_6h:<=:0.5,0.6,0.75,0.9",
            "strict_path_risk_score:>=:0.6,0.75,0.9",
            "strict_path_risk_score:<=:0.6,0.75,0.9",
        ]
    args._market = load_market(args.market_csv)
    rows = load_rows(args.rows_jsonl)
    folds = load_folds(args.folds_json or None)
    results = [run_fold(rows, fold, args) for fold in folds]
    out = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "rows_jsonl": args.rows_jsonl,
        "market_csv": args.market_csv,
        "config": {k: v for k, v in vars(args).items() if not k.startswith("_")},
        "folds": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    compact = []
    for r in results:
        if r.get("skipped"):
            compact.append({"name": r["fold"]["name"], "skipped": True, "counts": r["counts"]})
        else:
            compact.append({"name": r["fold"]["name"], "gate_passed": r.get("validation_gate_passed"), "selected": {k: r["selected_on_val"][k] for k in ("k", "score_threshold", "guard")}, "val": r["selected_on_val"]["sim"], "eval": r["eval_fixed"]["sim"], "eval_stats": r["eval_fixed"]["trade_stats"]})
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
