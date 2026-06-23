"""Convert row-level context labels into stable context-level policy labels.

The context miner deliberately emits row-level future labels for supervised data.
The first Gemma smoke showed those labels are too noisy.  This module smooths
labels by selecting token contexts whose action utility is stable on train/test,
then emits a single-policy SFT dataset where unselected contexts abstain.

Leakage rule: train/test may define the selected context set; eval is transformed
with the frozen context map only and is never used for selection.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

DEFAULT_CONTEXT_KEYS = (
    "trend_alignment",
    "risk_state",
    "dxy_zscore",
    "kimchi_premium_zscore",
    "funding_zscore",
    "premium_index_zscore",
    "range_pos",
    "window_drawdown",
    "taker_imbalance",
)
ACTIONS = ("LONG", "SHORT")


@dataclass(frozen=True)
class StableContextPolicyCfg:
    input_jsonl: str
    output: str
    summary_output: str = ""
    sample_output: str = ""
    context_keys: str = ",".join(DEFAULT_CONTEXT_KEYS)
    min_train_rows: int = 12
    min_test_rows: int = 4
    min_train_mean_pct: float = 0.08
    min_test_mean_pct: float = 0.02
    min_train_gap_pct: float = 0.08
    min_test_gap_pct: float = 0.00
    max_rows: int = 0


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def _keys(cfg: StableContextPolicyCfg) -> tuple[str, ...]:
    return tuple(k.strip() for k in str(cfg.context_keys).split(",") if k.strip())


def context_id(row: dict[str, Any], keys: Iterable[str]) -> str:
    tokens = row.get("state_tokens") or {}
    return "|".join(f"{k}={tokens.get(k, 'missing')}" for k in keys)


def _action_returns(row: dict[str, Any]) -> dict[str, float]:
    audit = row.get("reward_audit") or {}
    out: dict[str, float] = {}
    for action in ACTIONS:
        val = audit.get(action) or {}
        try:
            out[action] = float(val.get("net_return_pct", np.nan))
        except Exception:
            out[action] = float("nan")
    return out


def _aggregate(rows: list[dict[str, Any]], keys: tuple[str, ...], split: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"LONG": [], "SHORT": []})
    for row in rows:
        if str(row.get("split")) != split:
            continue
        cid = context_id(row, keys)
        returns = _action_returns(row)
        for action, value in returns.items():
            if np.isfinite(value):
                grouped[cid][action].append(float(value))
    summary: dict[str, dict[str, Any]] = {}
    for cid, vals in grouped.items():
        means = {a: float(np.mean(v)) if v else float("nan") for a, v in vals.items()}
        counts = {a: len(v) for a, v in vals.items()}
        if all(not np.isfinite(means[a]) for a in ACTIONS):
            continue
        best = max(ACTIONS, key=lambda a: means[a] if np.isfinite(means[a]) else -1e9)
        other = "SHORT" if best == "LONG" else "LONG"
        gap = float(means[best] - means[other]) if np.isfinite(means[best]) and np.isfinite(means[other]) else float("nan")
        summary[cid] = {"rows": max(counts.values()), "means": means, "best_action": best, "gap_pct": gap}
    return summary


def select_contexts(rows: list[dict[str, Any]], cfg: StableContextPolicyCfg) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    keys = _keys(cfg)
    train = _aggregate(rows, keys, "train")
    test = _aggregate(rows, keys, "test")
    selected: dict[str, dict[str, Any]] = {}
    rejected = Counter()
    for cid, tr in train.items():
        te = test.get(cid)
        if not te:
            rejected["missing_test"] += 1; continue
        action = str(tr["best_action"])
        if str(te["best_action"]) != action:
            rejected["test_side_mismatch"] += 1; continue
        if int(tr["rows"]) < int(cfg.min_train_rows):
            rejected["low_train_support"] += 1; continue
        if int(te["rows"]) < int(cfg.min_test_rows):
            rejected["low_test_support"] += 1; continue
        if float(tr["means"][action]) < float(cfg.min_train_mean_pct):
            rejected["low_train_mean"] += 1; continue
        if float(te["means"][action]) < float(cfg.min_test_mean_pct):
            rejected["low_test_mean"] += 1; continue
        if float(tr["gap_pct"]) < float(cfg.min_train_gap_pct):
            rejected["low_train_gap"] += 1; continue
        if float(te["gap_pct"]) < float(cfg.min_test_gap_pct):
            rejected["low_test_gap"] += 1; continue
        selected[cid] = {"action": action, "train": tr, "test": te}
    diagnostics = {"context_keys": keys, "train_contexts": len(train), "test_contexts": len(test), "selected_contexts": len(selected), "rejected": dict(sorted(rejected.items()))}
    return selected, diagnostics


def _target(action: str, selected: bool) -> dict[str, Any]:
    if selected and action in ACTIONS:
        return {"action": action, "confidence": "MEDIUM", "reason_code": "stable_context_expected_utility", "hold_bars": 288}
    return {"action": "NO_TRADE", "confidence": "LOW", "reason_code": "context_not_stable", "hold_bars": 0}


def transform_rows(rows: list[dict[str, Any]], selected: dict[str, dict[str, Any]], cfg: StableContextPolicyCfg) -> list[dict[str, Any]]:
    keys = _keys(cfg)
    out = []
    for row in rows:
        cid = context_id(row, keys)
        info = selected.get(cid)
        tgt = _target(str(info["action"]), True) if info else _target("NO_TRADE", False)
        nr = dict(row)
        nr["task"] = "stable_context_policy_sft"
        nr["context_id"] = cid
        nr["context_selection"] = {
            "selected": bool(info),
            "action": info.get("action") if info else "NO_TRADE",
            "selected_using_eval": False,
        }
        nr["target"] = json.dumps(tgt, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        nr["leakage_guard"] = {
            **dict(row.get("leakage_guard") or {}),
            "stable_context_selected_with_eval": False,
            "stable_context_selected_with_train_test_only": True,
        }
        out.append(nr)
        if int(cfg.max_rows) > 0 and len(out) >= int(cfg.max_rows):
            break
    return out


def _write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + ("\n" if rows else ""))


def summarize(rows: list[dict[str, Any]], selected: dict[str, dict[str, Any]], diagnostics: dict[str, Any], cfg: StableContextPolicyCfg) -> dict[str, Any]:
    split_counts = Counter(str(r.get("split")) for r in rows)
    action_counts: dict[str, Counter[str]] = defaultdict(Counter)
    selected_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        split = str(row.get("split"))
        action_counts[split][json.loads(str(row["target"])).get("action", "NO_TRADE")] += 1
        selected_counts[split][str(bool((row.get("context_selection") or {}).get("selected")))] += 1
    return {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "rows": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
        "action_counts_by_split": {k: dict(sorted(v.items())) for k, v in sorted(action_counts.items())},
        "selected_counts_by_split": {k: dict(sorted(v.items())) for k, v in sorted(selected_counts.items())},
        "selection": diagnostics,
        "selected_preview": list(selected.items())[:20],
        "config": asdict(cfg),
        "leakage_guard": {"context_selection_uses_eval": False, "context_selection_uses_train_test": True, "not_a_backtest_result": True},
    }


def run(cfg: StableContextPolicyCfg) -> dict[str, Any]:
    src = load_jsonl(cfg.input_jsonl)
    selected, diagnostics = select_contexts(src, cfg)
    rows = transform_rows(src, selected, cfg)
    _write_jsonl(cfg.output, rows)
    if cfg.sample_output:
        _write_jsonl(cfg.sample_output, rows[: min(20, len(rows))])
    summary = summarize(rows, selected, diagnostics, cfg)
    if cfg.summary_output:
        Path(cfg.summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.summary_output).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build stable context-level policy SFT labels")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--summary-output", default="")
    p.add_argument("--sample-output", default="")
    p.add_argument("--context-keys", default=StableContextPolicyCfg.context_keys)
    p.add_argument("--min-train-rows", type=int, default=StableContextPolicyCfg.min_train_rows)
    p.add_argument("--min-test-rows", type=int, default=StableContextPolicyCfg.min_test_rows)
    p.add_argument("--min-train-mean-pct", type=float, default=StableContextPolicyCfg.min_train_mean_pct)
    p.add_argument("--min-test-mean-pct", type=float, default=StableContextPolicyCfg.min_test_mean_pct)
    p.add_argument("--min-train-gap-pct", type=float, default=StableContextPolicyCfg.min_train_gap_pct)
    p.add_argument("--min-test-gap-pct", type=float, default=StableContextPolicyCfg.min_test_gap_pct)
    p.add_argument("--max-rows", type=int, default=StableContextPolicyCfg.max_rows)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(StableContextPolicyCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
