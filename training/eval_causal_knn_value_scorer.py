"""Causal KNN value scorer over LLM/regime prompt features.

This is a non-LLM execution scorer that uses the LLM-exposed feature/state text
as input.  Each eval candidate is scored only against historical reference
candidates available before the eval window.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from training.eval_pairwise_candidate_backtest import CandidateBacktestConfig, load_market, simulate_candidates
from training.kimchi_flow_pairwise_dataset import FEATURE_KEYS, bucket, candidate_meta, load_jsonl, parse_prompt


@dataclass(frozen=True)
class KNNValueConfig:
    reference_jsonl: str
    eval_jsonl: str
    market_csv: str
    output: str
    k: int = 25
    score_threshold: float = 0.0
    same_bucket: bool = True
    min_bucket_refs: int = 20
    hold_bars: int = 288
    entry_delay_bars: int = 1
    leverage: float = 0.5
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001


def _vector(row: dict[str, Any], keys: list[str]) -> np.ndarray:
    _, nums = parse_prompt(row["prompt"])
    return np.asarray([float(nums.get(k, 0.0)) for k in keys], dtype=float)


def _prepare(rows: list[dict[str, Any]], keys: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    x = np.vstack([_vector(r, keys) for r in rows]) if rows else np.zeros((0, len(keys)))
    y = np.asarray([float(r.get("trade_ret_pct", 0.0)) for r in rows], dtype=float)
    b = [bucket(r) for r in rows]
    return x, y, b


def score_eval(reference_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]], cfg: KNNValueConfig) -> list[dict[str, Any]]:
    keys = list(FEATURE_KEYS)
    ref_x, ref_y, ref_buckets = _prepare(reference_rows, keys)
    ev_x, _, _ = _prepare(eval_rows, keys)
    if ref_x.size == 0:
        raise ValueError("empty reference rows")
    mean = ref_x.mean(axis=0)
    std = ref_x.std(axis=0)
    std[std < 1e-9] = 1.0
    ref_z = (ref_x - mean) / std
    ev_z = (ev_x - mean) / std
    all_idx = np.arange(len(reference_rows))
    candidates: list[dict[str, Any]] = []
    for i, row in enumerate(eval_rows):
        b = bucket(row)
        idx = np.asarray([j for j, rb in enumerate(ref_buckets) if rb == b], dtype=int)
        if (not cfg.same_bucket) or len(idx) < int(cfg.min_bucket_refs):
            idx = all_idx
        d = np.linalg.norm(ref_z[idx] - ev_z[i], axis=1)
        order = idx[np.argsort(d)[: max(1, min(int(cfg.k), len(idx)))]]
        vals = ref_y[order]
        score = float(vals.mean()) if len(vals) else 0.0
        meta = candidate_meta(row)
        candidates.append(
            {
                **meta,
                "score_mean": score,
                "score_threshold": float(cfg.score_threshold),
                "knn_k": int(cfg.k),
                "knn_refs": int(len(order)),
                "knn_bucket": b,
                "knn_bucket_matched": bool(len(idx) != len(all_idx)),
                "knn_ref_mean_ret_pct": score,
                "knn_ref_median_ret_pct": float(np.median(vals)) if len(vals) else 0.0,
            }
        )
    return candidates


def run(cfg: KNNValueConfig) -> dict[str, Any]:
    ref = load_jsonl(cfg.reference_jsonl)
    ev = load_jsonl(cfg.eval_jsonl)
    candidates = score_eval(ref, ev, cfg)
    market = load_market(cfg.market_csv)
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
    result["config"] = cfg.__dict__
    result["scored_candidates"] = candidates
    result["score_summary"] = {
        "min": min((c["score_mean"] for c in candidates), default=0.0),
        "max": max((c["score_mean"] for c in candidates), default=0.0),
        "mean": sum(c["score_mean"] for c in candidates) / max(1, len(candidates)),
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
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
    p.add_argument("--hold-bars", type=int, default=288)
    p.add_argument("--entry-delay-bars", type=int, default=1)
    p.add_argument("--leverage", type=float, default=0.5)
    p.add_argument("--fee-rate", type=float, default=0.0004)
    p.add_argument("--slippage-rate", type=float, default=0.0001)
    return p.parse_args()


def main() -> None:
    out = run(KNNValueConfig(**vars(parse_args())))
    print(json.dumps({"sim": out["sim"], "trade_stats": out["trade_stats"], "score_summary": out["score_summary"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
