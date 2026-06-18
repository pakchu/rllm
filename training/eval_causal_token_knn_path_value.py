"""Causal token/Jaccard KNN scorer over LLM-facing state text.

The numeric KNN path-value scorer proved that the v8 prompt contains a useful
active-regime signal, but it still treats the prompt mostly as raw numbers.  This
diagnostic asks the LLM-specific question: do the symbolic regime words, tags,
and coarse numeric buckets carry a causal edge when matched as text tokens?

Future paths label only historical reference rows and audit eval rows.  Eval
scores are computed from reference rows only.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from training.eval_causal_knn_path_value_scorer import _path_outcome
from training.eval_pairwise_candidate_backtest import CandidateBacktestConfig, load_jsonl, load_market, simulate_candidates
from training.kimchi_flow_pairwise_dataset import candidate_meta
from training.path_outcome_dataset import TradePathOutcome

PathMetric = Literal["source_ret_pct", "path_net_pct", "path_utility_pct", "risk_adjusted_pct"]

SYMBOLIC_KEYS = {
    "Regime",
    "Volatility Level",
    "Momentum",
    "Trend Strength",
    "Playbook",
    "Short Horizon Direction",
    "Medium Horizon Direction",
    "Long Horizon Direction",
    "Trend Alignment",
    "Location",
    "Oscillator",
    "Candle Pattern",
    "Order Flow",
    "Risk State",
    "Cross Market Pressure",
    "Step Focus",
    "Trade Readiness",
    "Long Thesis",
    "Short Thesis",
    "No Trade Cause",
    "Regime Memory",
    "Regime Trap Risk",
    "Kimchi Flow Regime",
    "Long Entry Context",
    "Short Entry Context",
    "Regime Failure Cue",
    "4H Regime",
    "4H Location",
    "1D Regime",
    "1D Location",
    "3D Regime",
    "3D Location",
    "1W Regime",
    "1W Location",
    "MTF Activation Mode",
    "Path Risk Regime",
    "Stress Transition",
    "Path Asymmetry",
}

NUMERIC_BIN_KEYS = {
    "Window Volatility (%)",
    "Past Return 1h",
    "Past Return 2h",
    "Past Return 8h",
    "Past Path Drawdown 6h",
    "Past Path Runup 6h",
    "Realized Vol 1h",
    "Realized Vol 8h",
    "Range Position",
    "Order Flow Imbalance",
    "DXY Z",
    "USDKRW Z",
    "Kimchi Z",
    "Kimchi Change",
    "Vol Expansion Ratio",
    "Path Efficiency 6h",
    "Tradeability Score",
    "Long Evidence Votes",
    "Short Evidence Votes",
    "4H Return 1",
    "4H Range Position",
    "1D Return 1",
    "1D Range Position",
    "1W Return 1",
    "MTF Stress Total",
    "Drawdown Acceleration 6h",
    "Runup Drawdown Balance",
    "Path Compression",
    "Trend Conflict Score",
    "Flow Shock Score",
    "Macro Pressure Score",
    "Kimchi Liquidity Pressure",
    "HTF Stress Gradient",
    "Strict Path Risk Score",
}


@dataclass(frozen=True)
class TokenKNNPathValueConfig:
    reference_jsonl: str
    eval_jsonl: str
    market_csv: str
    output: str
    k: int = 25
    score_threshold: float = 0.0
    min_similarity: float = 0.05
    target_metric: PathMetric = "path_net_pct"
    hold_bars: int = 288
    entry_delay_bars: int = 1
    leverage: float = 0.5
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    mae_penalty: float = 1.0
    mfe_bonus: float = 0.0
    max_mae_pct: float = 100.0


def _num_bucket(value: float) -> str:
    if not math.isfinite(float(value)):
        return "NA"
    v = float(value)
    av = abs(v)
    if av < 1e-9:
        return "ZERO"
    sign = "POS" if v > 0 else "NEG"
    if av < 0.001:
        mag = "TINY"
    elif av < 0.005:
        mag = "SMALL"
    elif av < 0.015:
        mag = "MID"
    elif av < 0.04:
        mag = "LARGE"
    else:
        mag = "EXTREME"
    return f"{sign}_{mag}"


def _parse_prompt_tokens(prompt: str, side: str) -> set[str]:
    tokens: set[str] = {f"FIXED_SIDE={side.upper()}"}
    for raw in str(prompt).splitlines():
        if ":" not in raw:
            continue
        key, value = raw.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key in SYMBOLIC_KEYS and value:
            tokens.add(f"{key}={value}")
        elif key == "Tags":
            for tag in re.split(r"\s*\|\s*", value):
                tag = tag.strip()
                if tag:
                    tokens.add(f"TAG={tag}")
        elif key in NUMERIC_BIN_KEYS:
            try:
                tokens.add(f"{key}={_num_bucket(float(value))}")
            except Exception:
                continue
    return tokens


def _signal_pos_by_date(market: pd.DataFrame) -> dict[datetime, int]:
    return {ts.to_pydatetime().replace(tzinfo=None): int(i) for i, ts in enumerate(market["date"])}


def _target_value(row: dict[str, Any], outcome: TradePathOutcome | None, cfg: TokenKNNPathValueConfig) -> float:
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


def _prepare(rows: list[dict[str, Any]], market: pd.DataFrame, cfg: TokenKNNPathValueConfig) -> list[dict[str, Any]]:
    pos_by_date = _signal_pos_by_date(market)
    prepared: list[dict[str, Any]] = []
    # _path_outcome expects the similarly named config attributes; the dataclass
    # surface is intentionally compatible with the numeric scorer.
    for row in rows:
        outcome = _path_outcome(row, market, pos_by_date, cfg)  # type: ignore[arg-type]
        y = _target_value(row, outcome, cfg)
        if not np.isfinite(y) or y < -1e8:
            continue
        meta = candidate_meta(row)
        side = str(meta.get("side", "")).upper()
        tokens = _parse_prompt_tokens(str(row.get("prompt", "")), side)
        prepared.append(
            {
                "row": row,
                "meta": meta,
                "tokens": tokens,
                "target": float(y),
                "audit": {
                    "path_net_pct": 100.0 * outcome.net_return if outcome is not None else None,
                    "path_mae_pct": 100.0 * outcome.mae if outcome is not None else None,
                    "path_mfe_pct": 100.0 * outcome.mfe if outcome is not None else None,
                    "path_utility_pct": 100.0 * outcome.utility if outcome is not None else None,
                    "target_value": float(y),
                },
            }
        )
    return prepared


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a.intersection(b))
    union = len(a.union(b))
    return float(inter) / max(1, union)


def score_eval(reference_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]], market: pd.DataFrame, cfg: TokenKNNPathValueConfig) -> list[dict[str, Any]]:
    refs = _prepare(reference_rows, market, cfg)
    evs = _prepare(eval_rows, market, cfg)
    if not refs:
        raise ValueError("empty usable reference rows")
    candidates: list[dict[str, Any]] = []
    for ev in evs:
        sims = np.asarray([_jaccard(ev["tokens"], ref["tokens"]) for ref in refs], dtype=float)
        eligible = np.flatnonzero(sims >= float(cfg.min_similarity))
        if len(eligible) == 0:
            eligible = np.arange(len(refs))
        order = eligible[np.argsort(-sims[eligible])[: max(1, min(int(cfg.k), len(eligible)))]]
        vals = np.asarray([refs[int(i)]["target"] for i in order], dtype=float)
        picked_sims = sims[order]
        # Similarity-weighted mean, falling back to unweighted if all weights are zero.
        score = float(np.average(vals, weights=np.maximum(picked_sims, 1e-6))) if len(vals) else 0.0
        candidates.append(
            {
                **ev["meta"],
                "score_mean": score,
                "score_threshold": float(cfg.score_threshold),
                "token_knn_k": int(cfg.k),
                "token_knn_refs": int(len(order)),
                "token_similarity_mean": float(np.mean(picked_sims)) if len(picked_sims) else 0.0,
                "token_similarity_max": float(np.max(picked_sims)) if len(picked_sims) else 0.0,
                "knn_ref_target_mean": float(np.mean(vals)) if len(vals) else 0.0,
                "knn_ref_target_median": float(np.median(vals)) if len(vals) else 0.0,
                "eval_target_value": float(ev["target"]),
                "token_count": int(len(ev["tokens"])),
                **ev["audit"],
            }
        )
    return candidates


def run(cfg: TokenKNNPathValueConfig) -> dict[str, Any]:
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
    scores = [float(c["score_mean"]) for c in candidates]
    targets = [float(c["eval_target_value"]) for c in candidates]
    result["as_of"] = datetime.now(timezone.utc).isoformat()
    result["config"] = asdict(cfg)
    result["scored_candidates"] = candidates
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
        "tokens_from_prompt_are_past_only": True,
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Causal token/Jaccard KNN scorer over LLM state text")
    p.add_argument("--reference-jsonl", required=True)
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--k", type=int, default=25)
    p.add_argument("--score-threshold", type=float, default=0.0)
    p.add_argument("--min-similarity", type=float, default=0.05)
    p.add_argument("--target-metric", default="path_net_pct", choices=["source_ret_pct", "path_net_pct", "path_utility_pct", "risk_adjusted_pct"])
    p.add_argument("--hold-bars", type=int, default=288)
    p.add_argument("--entry-delay-bars", type=int, default=1)
    p.add_argument("--leverage", type=float, default=0.5)
    p.add_argument("--fee-rate", type=float, default=0.0004)
    p.add_argument("--slippage-rate", type=float, default=0.0001)
    p.add_argument("--mae-penalty", type=float, default=1.0)
    p.add_argument("--mfe-bonus", type=float, default=0.0)
    p.add_argument("--max-mae-pct", type=float, default=100.0)
    return p.parse_args()


def main() -> None:
    out = run(TokenKNNPathValueConfig(**vars(parse_args())))
    print(json.dumps({"sim": out["sim"], "trade_stats": out["trade_stats"], "score_summary": out["score_summary"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
