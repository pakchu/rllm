"""Past-only teacher for linear-alpha candidate pairwise rows.

The pairwise Gemma smoke failed because raw future path labels were noisy and the
prompt had no calibrated notion of which candidate family works in similar
recent regimes.  This module evaluates a lightweight walk-forward teacher that
uses only previous pairwise rows to estimate per-candidate and per-context win
rates, then predicts A/B for the next chronological period.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from training.eval_pairwise_choice import parse_choice
from training.linear_alpha_meta_stability_diagnostic import _date, _period_key


@dataclass(frozen=True)
class PairwiseTeacherConfig:
    inputs: str
    output: str
    period: str = "halfyear"
    min_train_rows: int = 1000
    min_eval_rows: int = 500
    smoothing: float = 8.0
    train_window_periods: int = 0


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _load(inputs: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in inputs.split(","):
        path = raw.strip()
        if path:
            rows.extend(_read_jsonl(path))
    return sorted(rows, key=lambda r: (str(r.get("date", "")), int(r.get("signal_pos", 0) or 0)))


def _candidate_id_from_prompt(prompt: str, label: str) -> str:
    m = re.search(rf"^{label}:\s+source=([^\s]+)", str(prompt), flags=re.MULTILINE)
    return m.group(1) if m else "unknown"


def _candidate_side_from_prompt(prompt: str, label: str) -> str:
    m = re.search(rf"^{label}:.*?\sside=([^\s]+)", str(prompt), flags=re.MULTILINE)
    return m.group(1) if m else "NONE"


def _numeric_state(prompt: str) -> dict[str, float]:
    out: dict[str, float] = {}
    in_state = False
    for raw in str(prompt).splitlines():
        line = raw.strip()
        if line == "state_context:":
            in_state = True
            continue
        if in_state and not line.startswith("- "):
            continue
        if in_state and line.startswith("- ") and ":" in line:
            key, val = line[2:].split(":", 1)
            try:
                out[key.strip()] = float(val.strip())
            except Exception:
                out[key.strip()] = 0.0
    return out


def _bucket(v: float, cuts: tuple[float, float]) -> str:
    if not math.isfinite(float(v)):
        return "nan"
    if v <= cuts[0]:
        return "low"
    if v >= cuts[1]:
        return "high"
    return "mid"


def _context_keys(row: dict[str, Any], candidate_label: str) -> list[str]:
    prompt = str(row.get("prompt", ""))
    state = _numeric_state(prompt)
    cid = _candidate_id_from_prompt(prompt, candidate_label)
    side = _candidate_side_from_prompt(prompt, candidate_label)
    return [
        f"id={cid}",
        f"id_side={cid}|{side}",
        f"side={side}",
        f"range={_bucket(state.get('range_pos', 0.0), (-0.5, 0.5))}",
        f"vol={_bucket(state.get('range_vol', 0.0), (0.02, 0.08))}",
        f"trend96={_bucket(state.get('trend_96', 0.0), (-0.02, 0.02))}",
        f"dxy={_bucket(state.get('dxy_zscore', 0.0), (-1.0, 1.0))}",
        f"kimchi={_bucket(state.get('kimchi_premium_zscore', 0.0), (-1.0, 1.0))}",
        f"rex2016={_bucket(state.get('rex_2016_range_pos', 0.0), (-0.5, 0.5))}",
    ]


def _periods(rows: list[dict[str, Any]], period: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[_period_key(_date(row), period)].append(row)
    return dict(sorted(out.items()))


def _build_stats(rows: list[dict[str, Any]]) -> dict[str, tuple[int, int]]:
    wins: Counter[str] = Counter()
    total: Counter[str] = Counter()
    for row in rows:
        target = parse_choice(row.get("target", ""))
        for label in ("A", "B"):
            y = 1 if target == label else 0
            for key in _context_keys(row, label):
                total[key] += 1
                wins[key] += y
    return {k: (wins[k], total[k]) for k in total}


def _score_candidate(row: dict[str, Any], label: str, stats: dict[str, tuple[int, int]], smoothing: float) -> float:
    priors = []
    weights = []
    for key in _context_keys(row, label):
        w, n = stats.get(key, (0, 0))
        if n <= 0:
            continue
        strength = min(1.0, n / 200.0)
        rate = (w + float(smoothing) * 0.5) / (n + float(smoothing))
        priors.append(rate)
        weights.append(strength)
    if not priors:
        return 0.5
    return float(np.average(np.asarray(priors, dtype=float), weights=np.asarray(weights, dtype=float)))


def _metrics(rows: list[dict[str, Any]], preds: list[str]) -> dict[str, Any]:
    targets = [parse_choice(r.get("target", "")) for r in rows]
    correct = sum(p == t for p, t in zip(preds, targets))
    counts = Counter(preds)
    target_counts = Counter(targets)
    return {
        "rows": len(rows),
        "accuracy": correct / max(1, len(rows)),
        "correct": correct,
        "prediction_counts": dict(sorted(counts.items())),
        "target_counts": dict(sorted(target_counts.items())),
        "always_a_accuracy": target_counts.get("A", 0) / max(1, len(rows)),
        "always_b_accuracy": target_counts.get("B", 0) / max(1, len(rows)),
    }


def run(cfg: PairwiseTeacherConfig) -> dict[str, Any]:
    rows = _load(cfg.inputs)
    by_period = _periods(rows, cfg.period)
    periods = list(by_period)
    evaluated: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    all_preds: list[str] = []
    for idx, period in enumerate(periods):
        eval_rows = by_period[period]
        if len(eval_rows) < int(cfg.min_eval_rows):
            continue
        train_periods = periods[:idx]
        if cfg.train_window_periods > 0:
            train_periods = train_periods[-int(cfg.train_window_periods):]
        train_rows = [r for p in train_periods for r in by_period[p]]
        if len(train_rows) < int(cfg.min_train_rows):
            continue
        stats = _build_stats(train_rows)
        preds: list[str] = []
        margins: list[float] = []
        for row in eval_rows:
            sa = _score_candidate(row, "A", stats, float(cfg.smoothing))
            sb = _score_candidate(row, "B", stats, float(cfg.smoothing))
            preds.append("A" if sa >= sb else "B")
            margins.append(abs(sa - sb))
        m = _metrics(eval_rows, preds)
        evaluated.append({
            "period": period,
            "train_periods": train_periods,
            "train_rows": len(train_rows),
            "stats_keys": len(stats),
            "metrics": m,
            "mean_abs_margin": float(np.mean(margins)) if margins else 0.0,
        })
        all_rows.extend(eval_rows)
        all_preds.extend(preds)
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "rows": len(rows),
        "periods": periods,
        "evaluated_periods": evaluated,
        "aggregate": _metrics(all_rows, all_preds) if all_rows else {},
        "leakage_guard": {
            "each_period_uses_only_previous_pairwise_rows": True,
            "future_path_utility_used_only_as_historical_labels": True,
        },
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Past-only pairwise candidate teacher diagnostic")
    p.add_argument("--inputs", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--period", choices=["year", "halfyear", "quarter"], default=PairwiseTeacherConfig.period)
    p.add_argument("--min-train-rows", type=int, default=PairwiseTeacherConfig.min_train_rows)
    p.add_argument("--min-eval-rows", type=int, default=PairwiseTeacherConfig.min_eval_rows)
    p.add_argument("--smoothing", type=float, default=PairwiseTeacherConfig.smoothing)
    p.add_argument("--train-window-periods", type=int, default=PairwiseTeacherConfig.train_window_periods)
    return p.parse_args()


def main() -> None:
    report = run(PairwiseTeacherConfig(**vars(parse_args())))
    print(json.dumps({"evaluated_periods": report["evaluated_periods"], "aggregate": report["aggregate"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
