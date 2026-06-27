"""Export pairwise preference rows from episode survival candidates.

Each row compares two candidates at the same signal timestamp.  The prompt uses
only causal candidate/setup/macro/history descriptors; the chosen answer is the
candidate with higher future path utility for offline preference training.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.alpha_linear_combo_scan import _load_market


@dataclass(frozen=True)
class EpisodeSurvivalPairwiseCfg:
    train_jsonl: str
    test_jsonl: str
    eval_jsonl: str
    market_csv: str
    output_dir: str
    min_utility_gap_pct: float = 0.35
    max_pairs_per_signal: int = 3
    max_rows_per_split: int = 50000
    seed: int = 42
    gzip_output: bool = True


def _open(path: str, mode: str = "rt"):
    return gzip.open(path, mode, encoding="utf-8") if str(path).endswith(".gz") else open(path, mode, encoding="utf-8")


def _load(path: str) -> list[dict[str, Any]]:
    with _open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _write(path: Path, rows: list[dict[str, Any]], gzip_output: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if gzip_output else open
    with opener(path, "wt", encoding="utf-8") as f:  # type: ignore[arg-type]
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def _prompt_parts(prompt: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for line in str(prompt).splitlines():
        if line.startswith("setup_quality: "):
            out["setup_quality"] = json.loads(line.split(": ", 1)[1])
        elif line.startswith("macro_context: "):
            out["macro_context"] = json.loads(line.split(": ", 1)[1])
    return out


def _history_context(market: pd.DataFrame, pos: int) -> dict[str, Any]:
    close = market["close"].to_numpy(dtype=float)
    high = market["high"].to_numpy(dtype=float)
    low = market["low"].to_numpy(dtype=float)
    pos = int(pos)
    out: dict[str, Any] = {}
    for w in (12, 48, 144, 576):
        start = max(0, pos - int(w) + 1)
        c0 = float(close[start]) if close[start] > 0 else float(close[pos])
        ret = float(close[pos] / c0 - 1.0) if c0 > 0 else 0.0
        path = close[start : pos + 1]
        rets = np.diff(np.log(np.maximum(path, 1e-12))) if len(path) > 1 else np.asarray([0.0])
        hi = float(np.max(high[start : pos + 1]))
        lo = float(np.min(low[start : pos + 1]))
        rng = max(1e-12, hi - lo)
        out[f"ret_{w}"] = round(ret, 5)
        out[f"vol_{w}"] = round(float(np.std(rets)), 6)
        out[f"range_pos_{w}"] = round(float((close[pos] - lo) / rng), 4)
        out[f"drawdown_{w}"] = round(float(close[pos] / max(1e-12, hi) - 1.0), 5)
    return out


def _candidate_view(row: dict[str, Any]) -> dict[str, Any]:
    parts = _prompt_parts(str(row.get("prompt", "")))
    return {
        "candidate": row.get("candidate") or {},
        "setup_quality": parts.get("setup_quality") or {},
        "macro_context": parts.get("macro_context") or {},
    }


def _utility(row: dict[str, Any]) -> float:
    return float((row.get("target_audit") or {}).get("utility_pct", 0.0) or 0.0)


def _target_audit(row: dict[str, Any]) -> dict[str, Any]:
    a = dict(row.get("target_audit") or {})
    return {k: a.get(k) for k in ("net_pct", "mae_pct", "mfe_pct", "mfe_to_mae", "utility_pct", "decision", "reason")}


def _pair_prompt(date: str, history: dict[str, Any], a: dict[str, Any], b: dict[str, Any]) -> str:
    return "\n".join(
        [
            "You are a BTCUSDT futures candidate ranker.",
            "Use only causal history/setup/context. Choose the candidate more likely to produce higher path-risk-adjusted utility.",
            "Return JSON with keys: choice, confidence, reason. choice must be A or B.",
            f"date: {date}",
            f"causal_history: {json.dumps(history, sort_keys=True, separators=(',', ':'))}",
            f"candidate_A: {json.dumps(a, sort_keys=True, separators=(',', ':'))}",
            f"candidate_B: {json.dumps(b, sort_keys=True, separators=(',', ':'))}",
        ]
    )


def _group(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    g: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        g[int(r.get("signal_pos", -1))].append(r)
    return g


def _build_split(rows: list[dict[str, Any]], market: pd.DataFrame, cfg: EpisodeSurvivalPairwiseCfg, rng: random.Random) -> list[dict[str, Any]]:
    out = []
    for pos, group in _group(rows).items():
        if len(group) < 2 or pos < 0 or pos >= len(market):
            continue
        ordered = sorted(group, key=_utility, reverse=True)
        best = ordered[0]
        best_u = _utility(best)
        pairs = 0
        for loser in ordered[1:]:
            gap = best_u - _utility(loser)
            if gap < float(cfg.min_utility_gap_pct):
                continue
            best_view = _candidate_view(best)
            loser_view = _candidate_view(loser)
            flip = rng.random() < 0.5
            a, b = (loser_view, best_view) if flip else (best_view, loser_view)
            choice = "B" if flip else "A"
            date = str(best.get("date") or market.iloc[pos]["date"])
            out.append(
                {
                    "task": "episode_survival_pairwise_preference",
                    "date": date,
                    "signal_pos": pos,
                    "prompt": _pair_prompt(date, _history_context(market, pos), a, b),
                    "target": json.dumps({"choice": choice, "confidence": "HIGH", "reason": "higher_future_path_utility"}, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
                    "chosen_candidate": best.get("candidate"),
                    "rejected_candidate": loser.get("candidate"),
                    "chosen_audit": _target_audit(best),
                    "rejected_audit": _target_audit(loser),
                    "utility_gap_pct": round(gap, 6),
                    "leakage_guard": {
                        "prompt_uses_future_path": False,
                        "chosen_rejected_use_future_path_for_training_only": True,
                        "candidates_share_same_signal_timestamp": True,
                    },
                }
            )
            pairs += 1
            if pairs >= int(cfg.max_pairs_per_signal):
                break
    rng.shuffle(out)
    out = out[: int(cfg.max_rows_per_split)]
    out.sort(key=lambda r: (str(r["date"]), int(r["signal_pos"]), str(r["target"])))
    return out


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    choices = Counter(json.loads(r["target"])["choice"] for r in rows)
    gaps = [float(r.get("utility_gap_pct", 0.0) or 0.0) for r in rows]
    chosen_sides = Counter(str((r.get("chosen_candidate") or {}).get("side")) for r in rows)
    return {
        "rows": len(rows),
        "choice_counts": dict(choices),
        "chosen_sides": dict(chosen_sides),
        "mean_utility_gap_pct": float(np.mean(gaps)) if gaps else 0.0,
        "median_utility_gap_pct": float(np.median(gaps)) if gaps else 0.0,
    }


def run(cfg: EpisodeSurvivalPairwiseCfg) -> dict[str, Any]:
    rng = random.Random(int(cfg.seed))
    market = _load_market(cfg.market_csv)
    loaded = {"train": _load(cfg.train_jsonl), "test": _load(cfg.test_jsonl), "eval": _load(cfg.eval_jsonl)}
    out_dir = Path(cfg.output_dir)
    suffix = ".jsonl.gz" if cfg.gzip_output else ".jsonl"
    report = {"config": asdict(cfg), "splits": {}}
    for split, rows in loaded.items():
        pairs = _build_split(rows, market, cfg, rng)
        path = out_dir / f"episode_survival_pairwise_{split}{suffix}"
        _write(path, pairs, bool(cfg.gzip_output))
        report["splits"][split] = {**_summary(pairs), "source_rows": len(rows), "output": str(path)}
    sp = out_dir / "episode_survival_pairwise_summary.json"
    sp.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--test-jsonl", required=True)
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--min-utility-gap-pct", type=float, default=EpisodeSurvivalPairwiseCfg.min_utility_gap_pct)
    p.add_argument("--max-pairs-per-signal", type=int, default=EpisodeSurvivalPairwiseCfg.max_pairs_per_signal)
    p.add_argument("--max-rows-per-split", type=int, default=EpisodeSurvivalPairwiseCfg.max_rows_per_split)
    p.add_argument("--seed", type=int, default=EpisodeSurvivalPairwiseCfg.seed)
    p.add_argument("--no-gzip-output", dest="gzip_output", action="store_false")
    p.set_defaults(gzip_output=EpisodeSurvivalPairwiseCfg.gzip_output)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(EpisodeSurvivalPairwiseCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
