"""Aggregate pairwise LLM choices into candidate-level scores and strict backtest.

Pair rows are comparison data, not executable trades.  This script converts
pairwise predictions into one score per fixed-rule candidate, selects candidates
with a frozen threshold, and simulates non-overlapping 24h holds on OHLC bars.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from training.strict_bar_backtest import _drawdown_from_trough, _trade_stats


@dataclass(frozen=True)
class CandidateBacktestConfig:
    market_csv: str
    pairwise_jsonl: str
    predictions_jsonl: str
    output: str
    score_threshold: float = 0.0
    hold_bars: int = 288
    entry_delay_bars: int = 1
    leverage: float = 0.5
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    candidate_role: str = ""


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def load_market(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"], compression="infer")
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="raise").dt.tz_convert(None)
    keep = ["date", "open", "high", "low", "close"]
    return df[keep].sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def _candidate_key(meta: dict[str, Any]) -> str:
    return "|".join(
        str(meta.get(k, ""))
        for k in ("signal_date", "entry_date", "exit_date", "side")
    )


def _raw_scores(pred: dict[str, Any]) -> tuple[float, float]:
    raw = pred.get("raw", "")
    try:
        obj = json.loads(raw)
        return float(obj["score_a"]), float(obj["score_b"])
    except Exception:
        # Fall back to unit votes for non-logprob prediction files.
        p = str(pred.get("prediction", "A")).upper()
        return (1.0, 0.0) if p == "A" else (0.0, 1.0)


def aggregate_candidates(pair_rows: list[dict[str, Any]], pred_rows: list[dict[str, Any]], *, candidate_role: str = "") -> list[dict[str, Any]]:
    acc: dict[str, dict[str, Any]] = {}
    for pair, pred in zip(pair_rows, pred_rows):
        if "candidate_a" not in pair or "candidate_b" not in pair:
            raise ValueError("pair rows must include candidate_a/candidate_b metadata; rebuild pairwise dataset")
        score_a, score_b = _raw_scores(pred)
        margin = score_a - score_b
        for label, meta, delta in (
            ("A", pair["candidate_a"], margin),
            ("B", pair["candidate_b"], -margin),
        ):
            if candidate_role and str(meta.get("role", "")) != str(candidate_role):
                continue
            key = _candidate_key(meta)
            row = acc.setdefault(
                key,
                {
                    **meta,
                    "candidate_key": key,
                    "score_sum": 0.0,
                    "score_votes": 0,
                    "positive_votes": 0,
                    "pair_labels": defaultdict(int),
                },
            )
            row["score_sum"] += float(delta)
            row["score_votes"] += 1
            row["positive_votes"] += int(delta > 0)
            row["pair_labels"][label] += 1
    out = []
    for row in acc.values():
        votes = max(1, int(row["score_votes"]))
        row["score_mean"] = float(row["score_sum"]) / votes
        row["positive_vote_rate"] = float(row["positive_votes"]) / votes
        row["pair_labels"] = dict(row["pair_labels"])
        out.append(row)
    return sorted(out, key=lambda r: str(r["signal_date"]))


def simulate_candidates(candidates: list[dict[str, Any]], market: pd.DataFrame, cfg: CandidateBacktestConfig) -> dict[str, Any]:
    selected = [
        c for c in candidates
        if float(c.get("score_mean", 0.0)) > float(cfg.score_threshold)
        and str(c.get("side", "")).upper() in {"LONG", "SHORT"}
    ]
    date_to_pos = {ts.to_pydatetime().replace(tzinfo=None): int(i) for i, ts in enumerate(market["date"])}
    opens = market["open"].to_numpy(float)
    highs = market["high"].to_numpy(float)
    lows = market["low"].to_numpy(float)
    eq = peak = 1.0
    max_dd = 0.0
    next_allowed = 0
    skipped = 0
    trade_returns: list[float] = []
    side_counts = {"LONG": 0, "SHORT": 0}
    executed: list[dict[str, Any]] = []
    cost = (float(cfg.fee_rate) + float(cfg.slippage_rate)) * float(cfg.leverage)

    for cand in selected:
        dt = datetime.fromisoformat(str(cand["signal_date"]))
        pos = date_to_pos.get(dt.replace(tzinfo=None))
        if pos is None:
            skipped += 1
            continue
        if pos < next_allowed:
            continue
        signal = 1 if str(cand["side"]).upper() == "LONG" else -1
        entry_pos = pos + int(cfg.entry_delay_bars)
        exit_pos = entry_pos + int(cfg.hold_bars)
        if entry_pos >= len(market) - 1 or exit_pos >= len(market):
            skipped += 1
            continue
        entry_eq = eq
        side_counts["LONG" if signal > 0 else "SHORT"] += 1
        eq *= max(0.0, 1.0 - cost)
        max_dd = max(max_dd, _drawdown_from_trough(peak, eq))
        for j in range(entry_pos, exit_pos):
            open_j = float(opens[j])
            if open_j <= 0.0:
                continue
            if signal > 0:
                adverse_ret = (float(lows[j]) - open_j) / open_j
                close_ret = (float(opens[j + 1]) - open_j) / open_j
            else:
                adverse_ret = (open_j - float(highs[j])) / open_j
                close_ret = (open_j - float(opens[j + 1])) / open_j
            max_dd = max(max_dd, _drawdown_from_trough(peak, eq * (1.0 + float(cfg.leverage) * adverse_ret)))
            eq *= max(0.0, 1.0 + float(cfg.leverage) * close_ret)
            peak = max(peak, eq)
            if eq <= 0.0:
                break
        eq *= max(0.0, 1.0 - cost)
        max_dd = max(max_dd, _drawdown_from_trough(peak, eq))
        peak = max(peak, eq)
        trade_ret = eq / entry_eq - 1.0
        trade_returns.append(trade_ret)
        executed.append({**cand, "executed_ret_pct": trade_ret * 100.0})
        next_allowed = exit_pos
        if eq <= 0.0:
            break

    if candidates:
        start_dt = datetime.fromisoformat(str(candidates[0]["signal_date"]))
        end_dt = datetime.fromisoformat(str(candidates[-1]["signal_date"]))
    else:
        start_dt = end_dt = datetime.now()
    years = max(1.0 / 365.25, float((end_dt - start_dt).days) / 365.25)
    ret_pct = (eq - 1.0) * 100.0
    gross = 1.0 + ret_pct / 100.0
    cagr_pct = ((gross ** (1.0 / years) - 1.0) * 100.0) if gross > 0.0 else -100.0
    mdd_pct = max_dd * 100.0
    return {
        "period": {"start": str(start_dt), "end": str(end_dt), "years": years},
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "executed": executed,
        "sim": {
            "ret_pct": ret_pct,
            "cagr_pct": cagr_pct,
            "strict_mdd_pct": mdd_pct,
            "cagr_to_strict_mdd": cagr_pct / mdd_pct if mdd_pct > 1e-12 else float("inf"),
            "trade_entries": len(trade_returns),
            "side_counts": side_counts,
            "skipped_missing_bars": skipped,
            "hold_bars": int(cfg.hold_bars),
            "entry_delay_bars": int(cfg.entry_delay_bars),
            "return_application": "candidate_score_selected_actual_ohlc_strict_mdd",
        },
        "trade_stats": _trade_stats(trade_returns),
    }


def run(cfg: CandidateBacktestConfig) -> dict[str, Any]:
    pair_rows = load_jsonl(cfg.pairwise_jsonl)
    pred_rows = load_jsonl(cfg.predictions_jsonl)
    if len(pair_rows) != len(pred_rows):
        raise ValueError(f"row mismatch: pairwise={len(pair_rows)} predictions={len(pred_rows)}")
    candidates = aggregate_candidates(pair_rows, pred_rows, candidate_role=cfg.candidate_role)
    market = load_market(cfg.market_csv)
    result = simulate_candidates(candidates, market, cfg)
    result["as_of"] = datetime.now(timezone.utc).isoformat()
    result["config"] = cfg.__dict__
    result["candidate_score_summary"] = {
        "min": min((float(c["score_mean"]) for c in candidates), default=0.0),
        "max": max((float(c["score_mean"]) for c in candidates), default=0.0),
        "mean": sum(float(c["score_mean"]) for c in candidates) / max(1, len(candidates)),
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--market-csv", required=True)
    p.add_argument("--pairwise-jsonl", required=True)
    p.add_argument("--predictions-jsonl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--score-threshold", type=float, default=0.0)
    p.add_argument("--hold-bars", type=int, default=288)
    p.add_argument("--entry-delay-bars", type=int, default=1)
    p.add_argument("--leverage", type=float, default=0.5)
    p.add_argument("--fee-rate", type=float, default=0.0004)
    p.add_argument("--slippage-rate", type=float, default=0.0001)
    p.add_argument("--candidate-role", default="")
    return p.parse_args()


def main() -> None:
    result = run(CandidateBacktestConfig(**vars(parse_args())))
    print(json.dumps({k: result[k] for k in ("candidate_count", "selected_count", "sim", "trade_stats")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
