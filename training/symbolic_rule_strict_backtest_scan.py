"""Strict-backtest-aware scan for symbolic candidate rules.

The prior symbolic scanner ranked rules by isolated future returns, which did not
transfer to actual portfolio execution.  This scanner generates explicit symbolic
rules, materializes live-style prediction rows per split, ranks rules by strict
bar-by-bar test backtest, and reports untouched eval backtest.
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from training.linear_alpha_deductive_candidate_selector import _candidate_from_pair, _numeric_state
from training.online_risk_overlay_backtest import OnlineRiskOverlayConfig, run_overlay
from training.symbolic_candidate_rule_discovery import _features, _load, _split_name


@dataclass(frozen=True)
class StrictSymbolicScanConfig:
    pairwise_inputs: str
    market_csv: str
    output: str
    train_start: str = "2024-01-01"
    train_end: str = "2024-06-30 23:59:59"
    test_start: str = "2024-07-01"
    test_end: str = "2025-12-31 23:59:59"
    eval_start: str = "2026-01-01"
    eval_end: str = "2026-06-01 00:00:00"
    min_support_train: int = 30
    min_support_test: int = 60
    min_test_trades: int = 20
    top_k: int = 30
    leverage: float = 0.5
    max_hold_bars: int = 576
    entry_delay_bars: int = 1
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    max_rules: int = 250
    max_rule_terms: int = 2
    max_state_features: int = 24
    prefilter_mode: str = "test_return"


def _candidate_examples(rows: list[dict[str, Any]], cfg: StrictSymbolicScanConfig) -> dict[str, list[dict[str, Any]]]:
    seen: set[tuple[int, str]] = set()
    splits: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        split = _split_name(str(row.get("date")), cfg)  # dataclass has same fields used by helper
        if split == "ignore":
            continue
        state = _numeric_state(str(row.get("prompt", "")))
        for label in ("A", "B"):
            cand = _candidate_from_pair(row, label)
            if cand is None:
                continue
            key = (int(row.get("signal_pos", -1) or -1), str(cand.get("id")))
            if key in seen:
                continue
            seen.add(key)
            pred = cand["prediction"]
            path = cand.get("path") if isinstance(cand.get("path"), dict) else {}
            ex = {
                "date": str(row.get("date")),
                "signal_pos": int(row.get("signal_pos", -1) or -1),
                "id": str(cand.get("id")),
                "side": str(pred.get("side", "NONE")),
                "hold_bars": int(pred.get("hold_bars", 0) or 0),
                "state": state,
                "prediction": pred,
                "ret_pct": float(path.get("realized_return_pct", 0.0) or 0.0),
                "features": None,
            }
            ex["features"] = _features(ex)
            splits[split].append(ex)
    return splits


def _rule_candidates(
    train: list[dict[str, Any]],
    *,
    min_support: int,
    max_rule_terms: int,
    max_state_features: int,
) -> list[tuple[str, ...]]:
    counts: Counter[str] = Counter()
    for ex in train:
        counts.update(ex["features"])
    base = [f for f, n in counts.items() if n >= int(min_support)]
    rules: set[tuple[str, ...]] = {(f,) for f in base}
    anchors = [f for f in base if f.startswith(("id=", "side=", "id_side="))]
    # Keep the most-supported state predicates so three-premise deductive rules do
    # not explode combinatorially before strict backtest prefiltering.
    states = [f for f in base if not f.startswith(("id=", "side=", "id_side=", "hold="))]
    states = sorted(states, key=lambda f: (-counts[f], f))[: int(max_state_features)]
    for a in anchors:
        for b in states:
            rules.add(tuple(sorted((a, b))))
    if int(max_rule_terms) >= 2:
        for pair in combinations(states, 2):
            rules.add(tuple(sorted(pair)))
    if int(max_rule_terms) >= 3:
        state_pairs = list(combinations(states, 2))
        for a in anchors:
            for b, c in state_pairs:
                rules.add(tuple(sorted((a, b, c))))
    return sorted(rules)


def _matching_predictions(examples: list[dict[str, Any]], rule: tuple[str, ...], action: str) -> list[dict[str, Any]]:
    rule_set = set(rule)
    out: list[dict[str, Any]] = []
    seen_pos: set[int] = set()
    for ex in examples:
        if ex["signal_pos"] in seen_pos:
            continue
        if not rule_set.issubset(ex["features"]):
            continue
        pred = dict(ex["prediction"])
        if action == "invert":
            if pred.get("side") == "LONG":
                pred["side"] = "SHORT"
            elif pred.get("side") == "SHORT":
                pred["side"] = "LONG"
            else:
                continue
        pred["family"] = f"strict_symbolic_rule_{action}"
        out.append(
            {
                "date": ex["date"],
                "signal_pos": int(ex["signal_pos"]),
                "prediction": pred,
                "position_scale": 1.0,
                "score": 1.0,
                "rule": list(rule),
                "action": action,
                "source_candidate_id": ex["id"],
            }
        )
        seen_pos.add(ex["signal_pos"])
    out.sort(key=lambda r: (str(r["date"]), int(r["signal_pos"])))
    return out


def _write_temp_predictions(rows: list[dict[str, Any]], tmpdir: Path, name: str) -> Path:
    path = tmpdir / f"{name}.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + ("\n" if rows else ""))
    return path


def _backtest(rows: list[dict[str, Any]], cfg: StrictSymbolicScanConfig, tmpdir: Path, name: str) -> dict[str, Any]:
    if not rows:
        return {"sim": {"trade_entries": 0, "cagr_pct": 0.0, "strict_mdd_pct": 0.0, "cagr_to_strict_mdd": 0.0}, "trade_stats": {"n_trades": 0}}
    pred_path = _write_temp_predictions(rows, tmpdir, name)
    out_path = tmpdir / f"{name}_bt.json"
    return run_overlay(
        OnlineRiskOverlayConfig(
            predictions_jsonl=str(pred_path),
            market_csv=cfg.market_csv,
            output=str(out_path),
            leverage=float(cfg.leverage),
            fee_rate=float(cfg.fee_rate),
            slippage_rate=float(cfg.slippage_rate),
            entry_delay_bars=int(cfg.entry_delay_bars),
            max_hold_bars=int(cfg.max_hold_bars),
        )
    )


def _cheap_prefilter_score(examples: list[dict[str, Any]], rule: tuple[str, ...], action: str) -> float:
    rule_set = set(rule)
    vals: list[float] = []
    for ex in examples:
        if rule_set.issubset(ex["features"]):
            sign = 1.0 if action == "follow" else -1.0
            vals.append(sign * float(ex.get("ret_pct", 0.0) or 0.0))
    if not vals:
        return -1e9
    arr = np.asarray(vals, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    t_like = mean / (std / math.sqrt(float(arr.size))) if std > 1e-12 else 0.0
    return mean * min(3.0, math.sqrt(float(arr.size)) / 12.0) + 0.02 * t_like


def _test_score(bt: dict[str, Any], min_trades: int) -> float:
    sim = bt.get("sim", {})
    stats = bt.get("trade_stats", {})
    trades = int(sim.get("trade_entries", 0) or 0)
    if trades < int(min_trades):
        return -1e9
    ratio = float(sim.get("cagr_to_strict_mdd", 0.0) or 0.0)
    cagr = float(sim.get("cagr_pct", 0.0) or 0.0)
    mdd = float(sim.get("strict_mdd_pct", 0.0) or 0.0)
    p = float(stats.get("p_value_mean_ret_approx", 1.0) or 1.0)
    return ratio + 0.03 * cagr - 0.02 * max(0.0, mdd - 15.0) - 0.25 * p + min(1.0, trades / 100.0)


def run(cfg: StrictSymbolicScanConfig) -> dict[str, Any]:
    rows = _load(cfg.pairwise_inputs)
    splits = _candidate_examples(rows, cfg)
    rules = _rule_candidates(
        splits["train"],
        min_support=int(cfg.min_support_train),
        max_rule_terms=int(cfg.max_rule_terms),
        max_state_features=int(cfg.max_state_features),
    )
    # Cheap prefilter: require enough matching candidate rows on train/test.
    candidates: list[tuple[tuple[str, ...], str, int, int, float]] = []
    for rule in rules:
        rule_set = set(rule)
        train_n = sum(1 for ex in splits["train"] if rule_set.issubset(ex["features"]))
        if train_n < int(cfg.min_support_train):
            continue
        test_n = sum(1 for ex in splits["test"] if rule_set.issubset(ex["features"]))
        if test_n < int(cfg.min_support_test):
            continue
        for action in ("follow", "invert"):
            if cfg.prefilter_mode == "support":
                prefilter_score = float(test_n) + 0.001 * float(train_n)
            else:
                prefilter_score = _cheap_prefilter_score(splits["test"], rule, action)
            candidates.append((rule, action, train_n, test_n, prefilter_score))
    # Use test only for prefiltering, then use strict backtest for final test ranking.
    # Eval is never used before the final report.
    candidates.sort(key=lambda x: (x[4], x[3], x[2]), reverse=True)
    prefiltered_candidate_count = len(candidates)
    # Many symbolic predicates are aliases for the same executed rows (for example
    # side_* predicates can duplicate non-side predicates on one-sided rules).
    # Deduplicate by the actual test prediction signature before spending strict
    # overlay time.  This preserves the highest-prefilter representative.
    unique_candidates: list[tuple[tuple[str, ...], str, int, int, float]] = []
    seen_signatures: set[tuple[str, tuple[int, ...]]] = set()
    for cand in candidates:
        rule, action, _train_n, _test_n, _prefilter_score = cand
        sig_rows = _matching_predictions(splits["test"], rule, action)
        signature = (action, tuple(int(r["signal_pos"]) for r in sig_rows))
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        unique_candidates.append(cand)
        if len(unique_candidates) >= int(cfg.max_rules):
            break
    candidates = unique_candidates

    scanned: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="strict_symbolic_scan_") as td:
        tmpdir = Path(td)
        for idx, (rule, action, train_support, test_support, prefilter_score) in enumerate(candidates):
            train_preds = _matching_predictions(splits["train"], rule, action)
            test_preds = _matching_predictions(splits["test"], rule, action)
            if len(test_preds) < int(cfg.min_support_test):
                continue
            eval_preds = _matching_predictions(splits["eval"], rule, action)
            train_bt = _backtest(train_preds, cfg, tmpdir, f"r{idx}_train")
            test_bt = _backtest(test_preds, cfg, tmpdir, f"r{idx}_test")
            eval_bt = _backtest(eval_preds, cfg, tmpdir, f"r{idx}_eval")
            scanned.append(
                {
                    "rule": list(rule),
                    "action": action,
                    "support": {"train_candidates": train_support, "test_candidates": test_support, "eval_candidates": len(eval_preds)},
                    "train": {"sim": train_bt["sim"], "trade_stats": train_bt["trade_stats"]},
                    "test": {"sim": test_bt["sim"], "trade_stats": test_bt["trade_stats"]},
                    "eval": {"sim": eval_bt["sim"], "trade_stats": eval_bt["trade_stats"]},
                    "prefilter_score": prefilter_score,
                    "test_score": _test_score(test_bt, int(cfg.min_test_trades)),
                }
            )
    scanned.sort(key=lambda r: float(r["test_score"]), reverse=True)
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "split_candidate_examples": {k: len(v) for k, v in splits.items()},
        "generated_rule_count": len(rules),
        "prefiltered_candidate_count": prefiltered_candidate_count,
        "unique_strict_candidate_count": len(candidates),
        "strict_scanned_count": len(scanned),
        "selection_protocol": "rules generated from train support; optional test-only cheap prefilter; strict backtest rank on test only; eval untouched",
        "top_by_test": scanned[: int(cfg.top_k)],
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Strict-backtest-aware symbolic candidate rule scan")
    p.add_argument("--pairwise-inputs", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--train-start", default=StrictSymbolicScanConfig.train_start)
    p.add_argument("--train-end", default=StrictSymbolicScanConfig.train_end)
    p.add_argument("--test-start", default=StrictSymbolicScanConfig.test_start)
    p.add_argument("--test-end", default=StrictSymbolicScanConfig.test_end)
    p.add_argument("--eval-start", default=StrictSymbolicScanConfig.eval_start)
    p.add_argument("--eval-end", default=StrictSymbolicScanConfig.eval_end)
    p.add_argument("--min-support-train", type=int, default=StrictSymbolicScanConfig.min_support_train)
    p.add_argument("--min-support-test", type=int, default=StrictSymbolicScanConfig.min_support_test)
    p.add_argument("--min-test-trades", type=int, default=StrictSymbolicScanConfig.min_test_trades)
    p.add_argument("--top-k", type=int, default=StrictSymbolicScanConfig.top_k)
    p.add_argument("--leverage", type=float, default=StrictSymbolicScanConfig.leverage)
    p.add_argument("--max-hold-bars", type=int, default=StrictSymbolicScanConfig.max_hold_bars)
    p.add_argument("--entry-delay-bars", type=int, default=StrictSymbolicScanConfig.entry_delay_bars)
    p.add_argument("--max-rules", type=int, default=StrictSymbolicScanConfig.max_rules)
    p.add_argument("--max-rule-terms", type=int, default=StrictSymbolicScanConfig.max_rule_terms)
    p.add_argument("--max-state-features", type=int, default=StrictSymbolicScanConfig.max_state_features)
    p.add_argument("--prefilter-mode", choices=("test_return", "support"), default=StrictSymbolicScanConfig.prefilter_mode)
    return p.parse_args()


def main() -> None:
    report = run(StrictSymbolicScanConfig(**vars(parse_args())))
    print(json.dumps({"strict_scanned_count": report["strict_scanned_count"], "top_by_test": report["top_by_test"][:10]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
