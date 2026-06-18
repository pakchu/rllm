"""Sweep causal KNN path-value scorer without reloading/relabeling every trial."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from training.eval_causal_knn_path_value_scorer import CausalKNNPathValueConfig, _prepare
from training.eval_pairwise_candidate_backtest import CandidateBacktestConfig, load_jsonl, load_market, simulate_candidates


def _parse_csv_floats(text: str) -> list[float]:
    return [float(x) for x in str(text).split(",") if str(x).strip()]


def _parse_csv_ints(text: str) -> list[int]:
    return [int(x) for x in str(text).split(",") if str(x).strip()]


def _score_from_prepared(
    ref_x: np.ndarray,
    ref_y: np.ndarray,
    ref_buckets: list[str],
    ev_x: np.ndarray,
    ev_y: np.ndarray,
    ev_buckets: list[str],
    ev_kept: list[dict[str, Any]],
    cfg: CausalKNNPathValueConfig,
) -> list[dict[str, Any]]:
    if ref_x.size == 0:
        raise ValueError("empty usable reference rows")
    mean = ref_x.mean(axis=0)
    std = ref_x.std(axis=0)
    std[std < 1e-9] = 1.0
    ref_z = (ref_x - mean) / std
    ev_z = (ev_x - mean) / std
    if cfg.standardize_targets:
        target_mean = float(ref_y.mean())
        target_std = float(ref_y.std()) or 1.0
        score_targets = (ref_y - target_mean) / target_std
    else:
        score_targets = ref_y
    all_idx = np.arange(len(ref_buckets))
    out: list[dict[str, Any]] = []
    for i, row in enumerate(ev_kept):
        b = ev_buckets[i]
        idx = np.asarray([j for j, rb in enumerate(ref_buckets) if rb == b], dtype=int)
        bucket_matched = True
        if (not cfg.same_bucket) or len(idx) < int(cfg.min_bucket_refs):
            idx = all_idx
            bucket_matched = False
        d = np.linalg.norm(ref_z[idx] - ev_z[i], axis=1)
        order = idx[np.argsort(d)[: max(1, min(int(cfg.k), len(idx)))]]
        vals = score_targets[order]
        raw_vals = ref_y[order]
        out.append({
            **row["_candidate_meta"],
            "score_mean": float(vals.mean()) if len(vals) else 0.0,
            "score_threshold": float(cfg.score_threshold),
            "knn_k": int(cfg.k),
            "knn_refs": int(len(order)),
            "knn_bucket": b,
            "knn_bucket_matched": bool(bucket_matched),
            "knn_ref_target_mean": float(raw_vals.mean()) if len(raw_vals) else 0.0,
            "knn_ref_target_median": float(np.median(raw_vals)) if len(raw_vals) else 0.0,
            "eval_target_value": float(ev_y[i]),
            **row["_path_audit"],
        })
    return out


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    reference_rows = load_jsonl(args.reference_jsonl)
    eval_rows = load_jsonl(args.eval_jsonl)
    market = load_market(args.market_csv)
    trials: list[dict[str, Any]] = []
    for metric in [x for x in args.target_metrics.split(",") if x.strip()]:
        prep_cfg = CausalKNNPathValueConfig(
            reference_jsonl=args.reference_jsonl,
            eval_jsonl=args.eval_jsonl,
            market_csv=args.market_csv,
            output=args.output,
            target_metric=metric,  # type: ignore[arg-type]
            hold_bars=args.hold_bars,
            entry_delay_bars=args.entry_delay_bars,
            leverage=args.leverage,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
            mae_penalty=args.mae_penalty,
            mfe_bonus=args.mfe_bonus,
            max_mae_pct=args.max_mae_pct,
            standardize_targets=args.standardize_targets,
            same_bucket=not args.no_same_bucket,
            min_bucket_refs=args.min_bucket_refs,
        )
        ref_x, ref_y, ref_buckets, _ = _prepare(reference_rows, market, prep_cfg)
        ev_x, ev_y, ev_buckets, ev_kept = _prepare(eval_rows, market, prep_cfg)
        for k in _parse_csv_ints(args.ks):
            score_cfg = CausalKNNPathValueConfig(**{**asdict(prep_cfg), "k": k})
            scored = _score_from_prepared(ref_x, ref_y, ref_buckets, ev_x, ev_y, ev_buckets, ev_kept, score_cfg)
            scores = [float(c["score_mean"]) for c in scored]
            targets = [float(c["eval_target_value"]) for c in scored]
            corr = float(np.corrcoef(scores, targets)[0, 1]) if len(scores) >= 2 and np.std(scores) > 1e-12 and np.std(targets) > 1e-12 else 0.0
            for th in _parse_csv_floats(args.thresholds):
                sim_cfg = CandidateBacktestConfig(
                    market_csv=args.market_csv,
                    pairwise_jsonl="",
                    predictions_jsonl="",
                    output="",
                    score_threshold=float(th),
                    hold_bars=args.hold_bars,
                    entry_delay_bars=args.entry_delay_bars,
                    leverage=args.leverage,
                    fee_rate=args.fee_rate,
                    slippage_rate=args.slippage_rate,
                )
                result = simulate_candidates(scored, market, sim_cfg)
                trials.append({
                    "config": {**asdict(score_cfg), "score_threshold": float(th)},
                    "sim": result["sim"],
                    "trade_stats": result["trade_stats"],
                    "candidate_count": result["candidate_count"],
                    "selected_count": result["selected_count"],
                    "score_summary": {
                        "min": min(scores, default=0.0),
                        "max": max(scores, default=0.0),
                        "mean": sum(scores) / max(1, len(scores)),
                        "eval_target_mean": sum(targets) / max(1, len(targets)),
                        "score_target_corr": corr,
                    },
                })
    trials.sort(key=lambda t: (float(t["sim"]["cagr_to_strict_mdd"]), int(t["sim"]["trade_entries"])), reverse=True)
    out = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "reference_jsonl": args.reference_jsonl,
        "eval_jsonl": args.eval_jsonl,
        "market_csv": args.market_csv,
        "trial_count": len(trials),
        "trials": trials,
        "leakage_guard": {
            "eval_scores_use_only_reference_rows": True,
            "threshold_selection_is_external_to_this_sweep": True,
            "eval_future_path_used_only_for_reported_audit": True,
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep causal KNN path-value scorer")
    p.add_argument("--reference-jsonl", required=True)
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--target-metrics", default="path_utility_pct,risk_adjusted_pct,path_net_pct,source_ret_pct")
    p.add_argument("--ks", default="5,10,15,25,40")
    p.add_argument("--thresholds", default="-0.5,0,0.1,0.25,0.5")
    p.add_argument("--no-same-bucket", action="store_true")
    p.add_argument("--min-bucket-refs", type=int, default=20)
    p.add_argument("--hold-bars", type=int, default=288)
    p.add_argument("--entry-delay-bars", type=int, default=1)
    p.add_argument("--leverage", type=float, default=0.5)
    p.add_argument("--fee-rate", type=float, default=0.0004)
    p.add_argument("--slippage-rate", type=float, default=0.0001)
    p.add_argument("--mae-penalty", type=float, default=1.0)
    p.add_argument("--mfe-bonus", type=float, default=0.0)
    p.add_argument("--max-mae-pct", type=float, default=100.0)
    p.add_argument("--standardize-targets", action="store_true")
    return p.parse_args()


def main() -> None:
    out = run_sweep(parse_args())
    print(json.dumps({"trial_count": out["trial_count"], "top": out["trials"][:10]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
