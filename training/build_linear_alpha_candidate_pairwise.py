"""Build pairwise candidate-choice rows from multiple frozen linear-alpha rules.

Unlike the veto setup, this creates a true ranking task: at the same timestamp,
compare two candidate trades proposed by different frozen alpha families and
choose the one with better future path utility.  Prompts contain only signal-time
candidate descriptors; future path stats are used only for labels/metadata.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from training.build_linear_alpha_meta_sft import _load_market, _trade_path_stats


@dataclass(frozen=True)
class CandidatePairwiseConfig:
    predictions: str
    market_csv: str
    output_jsonl: str
    summary_output: str = ""
    max_pairs_per_timestamp: int = 3
    min_utility_gap_pct: float = 0.15
    leverage: float = 0.5
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    entry_delay_bars: int = 1
    seed: int = 17


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + ("\n" if rows else ""))


def _pred(row: dict[str, Any]) -> dict[str, Any]:
    pred = row.get("prediction")
    return pred if isinstance(pred, dict) else {}


def _candidate_id(row: dict[str, Any]) -> str:
    pred = _pred(row)
    return f"{row.get('group')}|h{pred.get('hold_bars', 0)}|{row.get('variant', 'original')}"


def _path_utility(path: dict[str, float]) -> float:
    # Reward final return and favorable excursion, punish adverse path risk.
    return float(path["realized_return_pct"]) + 0.25 * float(path["max_favorable_pct"]) - 0.75 * float(path["max_adverse_pct"])


def _candidate_line(label: str, row: dict[str, Any]) -> str:
    pred = _pred(row)
    return (
        f"{label}: source={_candidate_id(row)} side={pred.get('side', 'NONE')} "
        f"hold_bars={pred.get('hold_bars', 0)} alpha_score={float(row.get('score', 0.0) or 0.0):+.8f}"
    )


def _prompt(row_a: dict[str, Any], row_b: dict[str, Any]) -> str:
    return "\n".join(
        [
            "You are a BTCUSDT futures candidate ranker.",
            "Two frozen no-leak alpha rules proposed trades at the same timestamp.",
            "Use only these signal-time descriptors. Choose the candidate with better expected path utility after costs.",
            f"date: {row_a.get('date')}",
            _candidate_line("A", row_a),
            _candidate_line("B", row_b),
            'Return compact JSON with exactly keys: choice, confidence, reason.',
        ]
    )


def _target(choice: str, confidence: str, reason: str) -> str:
    return json.dumps({"choice": choice, "confidence": confidence, "reason": reason}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build(cfg: CandidatePairwiseConfig) -> dict[str, Any]:
    market = _load_market(cfg.market_csv)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for path in [p.strip() for p in cfg.predictions.split(",") if p.strip()]:
        for row in _read_jsonl(path):
            pred = _pred(row)
            if str(pred.get("gate", "")).upper() != "TRADE":
                continue
            grouped[int(row.get("signal_pos", -1) or -1)].append(row)
    rng = np.random.default_rng(int(cfg.seed))
    rows_out: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    choice_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    for pos in sorted(grouped):
        cands = grouped[pos]
        if len(cands) < 2:
            skipped["less_than_two_candidates"] += 1
            continue
        enriched = []
        for row in cands:
            pred = _pred(row)
            path = _trade_path_stats(
                market=market,
                signal_pos=int(row.get("signal_pos", -1) or -1),
                side=str(pred.get("side", "NONE")),
                hold_bars=int(pred.get("hold_bars", 0) or 0),
                leverage=float(cfg.leverage),
                fee_rate=float(cfg.fee_rate),
                slippage_rate=float(cfg.slippage_rate),
                entry_delay_bars=int(cfg.entry_delay_bars),
            )
            if path is None:
                skipped["missing_path"] += 1
                continue
            enriched.append((row, path, _path_utility(path)))
        if len(enriched) < 2:
            skipped["less_than_two_valid_paths"] += 1
            continue
        pairs = list(itertools.combinations(enriched, 2))
        rng.shuffle(pairs)
        for (row_a, path_a, util_a), (row_b, path_b, util_b) in pairs[: max(1, int(cfg.max_pairs_per_timestamp))]:
            gap = float(util_a - util_b)
            if abs(gap) < float(cfg.min_utility_gap_pct):
                skipped["utility_gap_too_small"] += 1
                continue
            choice = "A" if gap > 0 else "B"
            winner_path = path_a if choice == "A" else path_b
            loser_path = path_b if choice == "A" else path_a
            confidence = "HIGH" if abs(gap) >= 0.75 else "MEDIUM"
            if winner_path["max_adverse_pct"] < loser_path["max_adverse_pct"] and winner_path["realized_return_pct"] >= loser_path["realized_return_pct"]:
                reason = "better_return_with_lower_adverse"
            elif winner_path["realized_return_pct"] > loser_path["realized_return_pct"]:
                reason = "better_realized_return"
            else:
                reason = "better_path_risk_adjusted_utility"
            target = _target(choice, confidence, reason)
            rows_out.append(
                {
                    "task": "linear_alpha_candidate_pairwise_choice",
                    "prompt": _prompt(row_a, row_b),
                    "target": target,
                    "date": row_a.get("date"),
                    "signal_pos": int(pos),
                    "choice": choice,
                    "chosen": target,
                    "rejected": _target("B" if choice == "A" else "A", "LOW", "worse_path_utility"),
                    "metadata": {
                        "candidate_a": {"id": _candidate_id(row_a), "prediction": _pred(row_a), "score": row_a.get("score"), "path": path_a, "utility": util_a},
                        "candidate_b": {"id": _candidate_id(row_b), "prediction": _pred(row_b), "score": row_b.get("score"), "path": path_b, "utility": util_b},
                        "utility_gap_a_minus_b_pct": gap,
                        "leakage_guard": "prompt has signal-time candidate descriptors only; future path stats used only for target and metadata",
                    },
                }
            )
            choice_counts[choice] += 1
            reason_counts[reason] += 1
    _write_jsonl(cfg.output_jsonl, rows_out)
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "timestamps_with_candidates": len(grouped),
        "rows": len(rows_out),
        "choice_counts": dict(sorted(choice_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "skipped_counts": dict(sorted(skipped.items())),
        "prompt_chars": {
            "min": min((len(r["prompt"]) for r in rows_out), default=0),
            "max": max((len(r["prompt"]) for r in rows_out), default=0),
            "mean": sum(len(r["prompt"]) for r in rows_out) / max(1, len(rows_out)),
        },
    }
    if cfg.summary_output:
        Path(cfg.summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.summary_output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build pairwise candidate-choice rows from frozen linear alpha predictions")
    p.add_argument("--predictions", required=True, help="comma-separated prediction JSONL files")
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--summary-output", default="")
    p.add_argument("--max-pairs-per-timestamp", type=int, default=CandidatePairwiseConfig.max_pairs_per_timestamp)
    p.add_argument("--min-utility-gap-pct", type=float, default=CandidatePairwiseConfig.min_utility_gap_pct)
    p.add_argument("--leverage", type=float, default=CandidatePairwiseConfig.leverage)
    p.add_argument("--fee-rate", type=float, default=CandidatePairwiseConfig.fee_rate)
    p.add_argument("--slippage-rate", type=float, default=CandidatePairwiseConfig.slippage_rate)
    p.add_argument("--entry-delay-bars", type=int, default=CandidatePairwiseConfig.entry_delay_bars)
    p.add_argument("--seed", type=int, default=CandidatePairwiseConfig.seed)
    return p.parse_args()


def main() -> None:
    print(json.dumps(build(CandidatePairwiseConfig(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
