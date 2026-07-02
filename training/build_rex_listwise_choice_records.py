"""Build listwise REX signal-choice records for LLM/RLLM training.

Instead of independent TAKE/SKIP labels, each record asks the model to choose one
action among candidates available at the same signal: NO_TRADE, resume, reclaim,
with long/short side as applicable.  Features in the prompt are signal-time or
prior only; the target is computed from future rewards for training/evaluation.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RexListwiseChoiceCfg:
    input_jsonl: str
    train_output: str
    eval_output: str
    summary_output: str
    train_start: str = "2020-01-01"
    train_end: str = "2026-01-01"
    eval_start: str = "2026-01-01"
    eval_end: str = "2026-06-01"
    min_best_utility_gap_pct: float = 0.0
    target_metric: str = "utility_pct"


def _load(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _date(row: dict[str, Any]) -> str:
    return str(row.get("date", ""))


def _split(date: str, cfg: RexListwiseChoiceCfg) -> str | None:
    if cfg.train_start <= date < cfg.train_end:
        return "train"
    if cfg.eval_start <= date < cfg.eval_end:
        return "eval"
    return None


def _choice_id(row: dict[str, Any]) -> str:
    fam = str(row.get("family", "")).upper()
    side = str(row.get("side", "")).upper()
    if fam == "NO_TRADE" or side == "NONE":
        return "NO_TRADE"
    return f"{fam}_{side}"


def _choice_line(i: int, row: dict[str, Any]) -> str:
    fs = row.get("feature_snapshot") or {}
    reward = row.get("reward") or {}
    fam = str(row.get("family", ""))
    side = str(row.get("side", ""))
    hold = int((row.get("candidate") or {}).get("hold_bars", 0) or 0)
    strength = float(fs.get("rex_candidate_strength", 0.0) or 0.0)
    excess = float(fs.get("rex_threshold_excess", 0.0) or 0.0)
    ratio = float(fs.get("rex_threshold_ratio", 0.0) or 0.0)
    return (
        f"{i}. id={_choice_id(row)} family={fam} side={side} hold={hold} "
        f"strength={strength:.4f} threshold_excess={excess:.4f} threshold_ratio={ratio:.3f}"
    )


def _context(tokens: dict[str, Any], fs: dict[str, Any]) -> str:
    token_keys = [
        "weekly_context", "weekly_location", "daily_context", "three_day_context", "four_hour_context",
        "location", "recent_drawdown", "dollar_pressure", "kimchi_context", "orderflow",
        "funding_context", "open_interest_level", "open_interest_change", "volume",
    ]
    numeric_keys = [
        "rex_144_range_pos", "rex_144_range_width_pct", "rex_2016_range_width_pct", "rex_8640_range_pos",
        "rex_8640_range_width_pct", "htf_1d_return_1", "htf_1d_return_4", "htf_3d_return_4",
        "dxy_zscore", "dxy_momentum", "kimchi_premium_zscore", "usdkrw_zscore", "rsi_norm", "bb_z",
        "taker_imbalance", "window_drawdown",
    ]
    token_text = "; ".join(f"{k}={tokens.get(k, 'unknown')}" for k in token_keys)
    num_text = "; ".join(f"{k}={float(fs.get(k, 0.0) or 0.0):.4f}" for k in numeric_keys)
    return f"Regime tokens: {token_text}\nNumeric context: {num_text}"


def _prompt(date: str, rows: list[dict[str, Any]]) -> str:
    exemplar = next((r for r in rows if str(r.get("family")) != "NO_TRADE"), rows[0])
    tokens = exemplar.get("state_tokens") or {}
    fs = exemplar.get("feature_snapshot") or {}
    choices = "\n".join(_choice_line(i + 1, r) for i, r in enumerate(rows))
    return (
        "You are a BTC futures risk-aware action selector.\n"
        "Choose exactly one candidate id for the next action. Prefer NO_TRADE when the setup is low quality or drawdown risk dominates.\n"
        "Use only the provided current/prior regime context and candidate descriptions. Do not assume future prices.\n\n"
        f"Signal time: {date}\n"
        f"{_context(tokens, fs)}\n\n"
        f"Candidates:\n{choices}\n\n"
        "Answer with only the chosen id."
    )


def _make_record(rows: list[dict[str, Any]], cfg: RexListwiseChoiceCfg, split: str) -> dict[str, Any] | None:
    rows = sorted(rows, key=lambda r: (_choice_id(r), str(r.get("family")), str(r.get("side"))))
    metric = cfg.target_metric
    scored = []
    for r in rows:
        reward = r.get("reward") or {}
        scored.append((float(reward.get(metric, reward.get("utility_pct", 0.0)) or 0.0), r))
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_row = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else -999.0
    if float(cfg.min_best_utility_gap_pct) > 0 and (best_score - second_score) < float(cfg.min_best_utility_gap_pct):
        return None
    date = _date(rows[0])
    return {
        "task": "rex_listwise_choice",
        "split": split,
        "date": date,
        "signal_pos": int(rows[0].get("signal_pos", -1) or -1),
        "prompt": _prompt(date, rows),
        "target": _choice_id(best_row),
        "choices": [_choice_id(r) for r in rows],
        "choice_rewards": { _choice_id(r): float((r.get("reward") or {}).get(metric, 0.0) or 0.0) for r in rows },
        "best_margin_pct": best_score - second_score,
        "target_metric": metric,
        "leakage_guard": {
            "prompt_uses_future_path": False,
            "target_uses_future_rewards_for_training_only": True,
            "choices_share_same_signal_time": True,
            "no_trade_counterfactual_reward_is_zero": any(_choice_id(r) == "NO_TRADE" for r in rows),
        },
    }


def _write(path: str, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows))


def run(cfg: RexListwiseChoiceCfg) -> dict[str, Any]:
    raw = _load(cfg.input_jsonl)
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for r in raw:
        sp = _split(_date(r), cfg)
        if sp is None:
            continue
        groups[(_date(r), int(r.get("signal_pos", -1) or -1))].append(r)
    out = {"train": [], "eval": []}
    skipped_gap = 0
    for (_, _), rows in sorted(groups.items()):
        split = _split(_date(rows[0]), cfg)
        if split is None:
            continue
        rec = _make_record(rows, cfg, split)
        if rec is None:
            skipped_gap += 1
            continue
        out[split].append(rec)
    _write(cfg.train_output, out["train"])
    _write(cfg.eval_output, out["eval"])
    summary = {
        "config": asdict(cfg),
        "rows": {k: len(v) for k, v in out.items()},
        "target_counts": {k: dict(Counter(r["target"] for r in v)) for k, v in out.items()},
        "choice_count_distribution": {k: dict(Counter(len(r["choices"]) for r in v)) for k, v in out.items()},
        "skipped_gap": skipped_gap,
        "prompt_chars": {k: {"min": min([len(r["prompt"]) for r in v] or [0]), "max": max([len(r["prompt"]) for r in v] or [0]), "mean": (sum(len(r["prompt"]) for r in v) / len(v) if v else 0)} for k, v in out.items()},
    }
    Path(cfg.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.summary_output).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build REX listwise candidate-choice records")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--train-output", required=True)
    p.add_argument("--eval-output", required=True)
    p.add_argument("--summary-output", required=True)
    p.add_argument("--train-start", default=RexListwiseChoiceCfg.train_start)
    p.add_argument("--train-end", default=RexListwiseChoiceCfg.train_end)
    p.add_argument("--eval-start", default=RexListwiseChoiceCfg.eval_start)
    p.add_argument("--eval-end", default=RexListwiseChoiceCfg.eval_end)
    p.add_argument("--min-best-utility-gap-pct", type=float, default=RexListwiseChoiceCfg.min_best_utility_gap_pct)
    p.add_argument("--target-metric", choices=["utility_pct", "net_return_pct"], default=RexListwiseChoiceCfg.target_metric)
    return p.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(RexListwiseChoiceCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))
