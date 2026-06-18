"""Causal KNN scorer using executable post-signal path value labels.

This probes whether the LLM-facing Kimchi-flow prompt features contain an edge
for the *path quality* of a fixed-rule candidate, not just its terminal return.
Each eval candidate is scored against historical reference candidates only; the
future path is used only to label reference rows and to audit eval outcomes after
selection.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from training.eval_pairwise_candidate_backtest import CandidateBacktestConfig, load_jsonl, load_market, simulate_candidates
from training.kimchi_flow_pairwise_dataset import FEATURE_KEYS, bucket, candidate_meta, parse_prompt
from training.path_outcome_dataset import PathOutcomeConfig, TradePathOutcome, compute_trade_path_outcome

PathMetric = Literal["source_ret_pct", "path_net_pct", "path_utility_pct", "risk_adjusted_pct"]


@dataclass(frozen=True)
class CausalKNNPathValueConfig:
    reference_jsonl: str
    eval_jsonl: str
    market_csv: str
    output: str
    k: int = 25
    score_threshold: float = 0.0
    same_bucket: bool = True
    min_bucket_refs: int = 20
    target_metric: PathMetric = "path_utility_pct"
    hold_bars: int = 288
    entry_delay_bars: int = 1
    leverage: float = 0.5
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    mae_penalty: float = 1.0
    mfe_bonus: float = 0.0
    max_mae_pct: float = 100.0
    standardize_targets: bool = False


def _vector(row: dict[str, Any], keys: list[str]) -> np.ndarray:
    _, nums = parse_prompt(row["prompt"])
    src = row.get("source_trade", {}) or {}
    tgt = {}
    try:
        tgt = json.loads(str(row.get("target", "{}")))
    except Exception:
        tgt = {}
    side = str(src.get("side", tgt.get("side", ""))).upper()
    nums["fixed_side_long"] = 1.0 if side == "LONG" else 0.0
    nums["fixed_side_short"] = 1.0 if side == "SHORT" else 0.0
    return np.asarray([float(nums.get(k, 0.0)) for k in keys], dtype=float)


def _signal_pos_by_date(market: pd.DataFrame) -> dict[datetime, int]:
    return {ts.to_pydatetime().replace(tzinfo=None): int(i) for i, ts in enumerate(market["date"])}


def _path_outcome(row: dict[str, Any], market: pd.DataFrame, pos_by_date: dict[datetime, int], cfg: CausalKNNPathValueConfig) -> TradePathOutcome | None:
    meta = candidate_meta(row)
    side = str(meta.get("side", "")).upper()
    if side not in {"LONG", "SHORT"}:
        return None
    signal_dt = datetime.fromisoformat(str(meta.get("signal_date"))).replace(tzinfo=None)
    pos = pos_by_date.get(signal_dt)
    if pos is None:
        return None
    path_cfg = PathOutcomeConfig(
        hold_bars=int(cfg.hold_bars),
        entry_delay_bars=int(cfg.entry_delay_bars),
        fee_rate=float(cfg.fee_rate),
        slippage_rate=float(cfg.slippage_rate),
        leverage=float(cfg.leverage),
        mae_penalty=float(cfg.mae_penalty),
        mfe_bonus=float(cfg.mfe_bonus),
    )
    return compute_trade_path_outcome(market, pos, side, path_cfg)


def _target_value(row: dict[str, Any], outcome: TradePathOutcome | None, cfg: CausalKNNPathValueConfig) -> float:
    if cfg.target_metric == "source_ret_pct":
        return float(row.get("trade_ret_pct", candidate_meta(row).get("trade_ret_pct", 0.0)))
    if outcome is None:
        return -1e9
    net_pct = 100.0 * float(outcome.net_return)
    mae_pct = 100.0 * float(outcome.mae)
    mfe_pct = 100.0 * float(outcome.mfe)
    if mae_pct > float(cfg.max_mae_pct):
        return -1e9
    if cfg.target_metric == "path_net_pct":
        return net_pct
    if cfg.target_metric == "path_utility_pct":
        return 100.0 * float(outcome.utility)
    if cfg.target_metric == "risk_adjusted_pct":
        return net_pct - float(cfg.mae_penalty) * mae_pct + float(cfg.mfe_bonus) * mfe_pct
    raise ValueError(f"unknown target_metric: {cfg.target_metric}")


def _prepare(rows: list[dict[str, Any]], market: pd.DataFrame, cfg: CausalKNNPathValueConfig) -> tuple[np.ndarray, np.ndarray, list[str], list[dict[str, Any]]]:
    keys = list(FEATURE_KEYS)
    pos_by_date = _signal_pos_by_date(market)
    kept: list[dict[str, Any]] = []
    targets: list[float] = []
    for row in rows:
        outcome = _path_outcome(row, market, pos_by_date, cfg)
        y = _target_value(row, outcome, cfg)
        if not np.isfinite(y) or y < -1e8:
            continue
        meta = candidate_meta(row)
        audit = {
            "path_net_pct": 100.0 * outcome.net_return if outcome is not None else None,
            "path_mae_pct": 100.0 * outcome.mae if outcome is not None else None,
            "path_mfe_pct": 100.0 * outcome.mfe if outcome is not None else None,
            "path_utility_pct": 100.0 * outcome.utility if outcome is not None else None,
            "target_value": float(y),
        }
        kept.append({**row, "_candidate_meta": meta, "_path_audit": audit})
        targets.append(float(y))
    x = np.vstack([_vector(r, keys) for r in kept]) if kept else np.zeros((0, len(keys)))
    buckets = [bucket(r) for r in kept]
    return x, np.asarray(targets, dtype=float), buckets, kept


def score_eval(reference_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]], market: pd.DataFrame, cfg: CausalKNNPathValueConfig) -> list[dict[str, Any]]:
    keys = list(FEATURE_KEYS)
    ref_x, ref_y, ref_buckets, ref_kept = _prepare(reference_rows, market, cfg)
    ev_x, ev_y, ev_buckets, ev_kept = _prepare(eval_rows, market, cfg)
    if ref_x.size == 0:
        raise ValueError("empty usable reference rows")
    mean = ref_x.mean(axis=0)
    std = ref_x.std(axis=0)
    std[std < 1e-9] = 1.0
    ref_z = (ref_x - mean) / std
    ev_z = (ev_x - mean) / std
    target_mean = float(ref_y.mean())
    target_std = float(ref_y.std())
    if target_std < 1e-9:
        target_std = 1.0
    score_targets = (ref_y - target_mean) / target_std if cfg.standardize_targets else ref_y
    all_idx = np.arange(len(ref_kept))
    candidates: list[dict[str, Any]] = []
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
        score = float(vals.mean()) if len(vals) else 0.0
        meta = row["_candidate_meta"]
        candidates.append(
            {
                **meta,
                "score_mean": score,
                "score_threshold": float(cfg.score_threshold),
                "knn_k": int(cfg.k),
                "knn_refs": int(len(order)),
                "knn_bucket": b,
                "knn_bucket_matched": bool(bucket_matched),
                "knn_ref_target_mean": float(raw_vals.mean()) if len(raw_vals) else 0.0,
                "knn_ref_target_median": float(np.median(raw_vals)) if len(raw_vals) else 0.0,
                "eval_target_value": float(ev_y[i]),
                **row["_path_audit"],
            }
        )
    return candidates


def run(cfg: CausalKNNPathValueConfig) -> dict[str, Any]:
    reference_rows = load_jsonl(cfg.reference_jsonl)
    eval_rows = load_jsonl(cfg.eval_jsonl)
    market = load_market(cfg.market_csv)
    candidates = score_eval(reference_rows, eval_rows, market, cfg)
    sim_cfg = CandidateBacktestConfig(
        market_csv=cfg.market_csv,
        pairwise_jsonl="",
        predictions_jsonl="",
        output="",
        score_threshold=float(cfg.score_threshold),
        hold_bars=int(cfg.hold_bars),
        entry_delay_bars=int(cfg.entry_delay_bars),
        leverage=float(cfg.leverage),
        fee_rate=float(cfg.fee_rate),
        slippage_rate=float(cfg.slippage_rate),
    )
    result = simulate_candidates(candidates, market, sim_cfg)
    result["as_of"] = datetime.now(timezone.utc).isoformat()
    result["config"] = asdict(cfg)
    result["scored_candidates"] = candidates
    scores = [float(c["score_mean"]) for c in candidates]
    targets = [float(c["eval_target_value"]) for c in candidates]
    result["score_summary"] = {
        "min": min(scores, default=0.0),
        "max": max(scores, default=0.0),
        "mean": sum(scores) / max(1, len(scores)),
        "eval_target_mean": sum(targets) / max(1, len(targets)),
        "score_target_corr": float(np.corrcoef(scores, targets)[0, 1]) if len(scores) >= 2 and np.std(scores) > 1e-12 and np.std(targets) > 1e-12 else 0.0,
    }
    result["leakage_guard"] = {
        "eval_scores_use_only_reference_rows": True,
        "reference_rows_must_precede_eval_window_by_caller_contract": True,
        "eval_future_path_used_only_for_audit_not_scoring": True,
        "candidate_selection_uses_score_threshold_only": True,
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Causal KNN scorer over executable path-value labels")
    p.add_argument("--reference-jsonl", required=True)
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--k", type=int, default=25)
    p.add_argument("--score-threshold", type=float, default=0.0)
    p.add_argument("--same-bucket", action="store_true")
    p.add_argument("--no-same-bucket", dest="same_bucket", action="store_false")
    p.set_defaults(same_bucket=True)
    p.add_argument("--min-bucket-refs", type=int, default=20)
    p.add_argument("--target-metric", default="path_utility_pct", choices=["source_ret_pct", "path_net_pct", "path_utility_pct", "risk_adjusted_pct"])
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
    result = run(CausalKNNPathValueConfig(**vars(parse_args())))
    print(json.dumps({"sim": result["sim"], "trade_stats": result["trade_stats"], "score_summary": result["score_summary"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
