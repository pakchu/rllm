"""Base strict diagnostic for binary-edge live candidate rows.

Evaluate singleton family/hold/side/id_side rules under strict overlay before larger
symbolic conjunction search.  This identifies whether the candidate pool contains
any base tradable slice at all.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.binary_edge_symbolic_rule_strict_backtest_scan import (
    BinaryEdgeStrictScanConfig,
    _bt,
    _examples,
    _load,
    _match_preds,
    _split,
)


@dataclass(frozen=True)
class BinaryEdgeBaseDiagConfig:
    inputs: str
    market_csv: str
    output: str
    train_start: str = "2022-01-01"
    train_end: str = "2024-12-31 23:59:59"
    test_start: str = "2025-01-01"
    test_end: str = "2025-12-31 23:59:59"
    eval_start: str = "2026-01-01"
    eval_end: str = "2026-06-01 00:00:00"
    min_support_train: int = 80
    min_support_test: int = 40
    top_k: int = 50
    leverage: float = 0.5
    max_hold_bars: int = 576
    entry_delay_bars: int = 1
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    feature_prefixes: str = "id=,family=,hold=,side=,id_side="
    actions: str = "follow,invert"


def _scan_cfg(cfg: BinaryEdgeBaseDiagConfig) -> BinaryEdgeStrictScanConfig:
    return BinaryEdgeStrictScanConfig(
        inputs=cfg.inputs,
        market_csv=cfg.market_csv,
        output=cfg.output,
        train_start=cfg.train_start,
        train_end=cfg.train_end,
        test_start=cfg.test_start,
        test_end=cfg.test_end,
        eval_start=cfg.eval_start,
        eval_end=cfg.eval_end,
        leverage=cfg.leverage,
        max_hold_bars=cfg.max_hold_bars,
        entry_delay_bars=cfg.entry_delay_bars,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
    )


def _score(split: dict[str, Any]) -> float:
    sim = split.get("sim", {})
    stats = split.get("trade_stats", {})
    trades = int(sim.get("trade_entries", 0) or 0)
    if trades <= 0:
        return -1e9
    return (
        float(sim.get("cagr_to_strict_mdd", 0.0) or 0.0)
        + 0.03 * float(sim.get("cagr_pct", 0.0) or 0.0)
        - 0.02 * max(0.0, float(sim.get("strict_mdd_pct", 0.0) or 0.0) - 15.0)
        - 0.25 * float(stats.get("p_value_mean_ret_approx", 1.0) or 1.0)
        + min(1.0, trades / 100.0)
    )


def run(cfg: BinaryEdgeBaseDiagConfig) -> dict[str, Any]:
    scan_cfg = _scan_cfg(cfg)
    splits = _examples(_load(cfg.inputs), scan_cfg)
    prefixes = tuple(x.strip() for x in str(cfg.feature_prefixes).split(",") if x.strip())
    actions = [a.strip() for a in str(cfg.actions).split(",") if a.strip() in {"follow", "invert"}]
    counts: dict[str, Counter[str]] = {name: Counter() for name in ("train", "test", "eval")}
    for name, rows in splits.items():
        for ex in rows:
            counts[name].update(f for f in ex["features"] if f.startswith(prefixes))
    features = sorted(
        f for f, n in counts["train"].items()
        if n >= int(cfg.min_support_train) and counts["test"].get(f, 0) >= int(cfg.min_support_test)
    )
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="binary_edge_base_diag_") as td:
        tmp = Path(td)
        idx = 0
        for feat in features:
            for action in actions:
                rule = (feat,)
                train_bt = _bt(_match_preds(splits["train"], rule, action), scan_cfg, tmp, f"r{idx}_train")
                test_bt = _bt(_match_preds(splits["test"], rule, action), scan_cfg, tmp, f"r{idx}_test")
                eval_preds = _match_preds(splits["eval"], rule, action)
                eval_bt = _bt(eval_preds, scan_cfg, tmp, f"r{idx}_eval")
                rows.append({
                    "rule": [feat],
                    "action": action,
                    "support": {
                        "train_candidates": counts["train"].get(feat, 0),
                        "test_candidates": counts["test"].get(feat, 0),
                        "eval_candidates": counts["eval"].get(feat, 0),
                    },
                    "train": {"sim": train_bt["sim"], "trade_stats": train_bt["trade_stats"]},
                    "test": {"sim": test_bt["sim"], "trade_stats": test_bt["trade_stats"]},
                    "eval": {"sim": eval_bt["sim"], "trade_stats": eval_bt["trade_stats"]},
                    "test_score": _score({"sim": test_bt["sim"], "trade_stats": test_bt["trade_stats"]}),
                })
                idx += 1
    for row in rows:
        row["train_score"] = _score(row["train"])
        row["eval_score"] = _score(row["eval"])
        row["stable_score"] = min(float(row["train_score"]), float(row["test_score"]), float(row["eval_score"]))
    rows.sort(key=lambda r: float(r["test_score"]), reverse=True)
    top_by_train = sorted(rows, key=lambda r: float(r["train_score"]), reverse=True)[: int(cfg.top_k)]
    top_stable = sorted(rows, key=lambda r: float(r["stable_score"]), reverse=True)[: int(cfg.top_k)]
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "split_candidate_examples": {k: len(v) for k, v in splits.items()},
        "feature_count": len(features),
        "evaluated_count": len(rows),
        "selection_protocol": "base singleton features; rank by strict test; eval reported untouched",
        "top_by_test": rows[: int(cfg.top_k)],
        "top_by_train": top_by_train,
        "top_stable": top_stable,
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--train-start", default=BinaryEdgeBaseDiagConfig.train_start)
    p.add_argument("--train-end", default=BinaryEdgeBaseDiagConfig.train_end)
    p.add_argument("--test-start", default=BinaryEdgeBaseDiagConfig.test_start)
    p.add_argument("--test-end", default=BinaryEdgeBaseDiagConfig.test_end)
    p.add_argument("--eval-start", default=BinaryEdgeBaseDiagConfig.eval_start)
    p.add_argument("--eval-end", default=BinaryEdgeBaseDiagConfig.eval_end)
    p.add_argument("--min-support-train", type=int, default=BinaryEdgeBaseDiagConfig.min_support_train)
    p.add_argument("--min-support-test", type=int, default=BinaryEdgeBaseDiagConfig.min_support_test)
    p.add_argument("--top-k", type=int, default=BinaryEdgeBaseDiagConfig.top_k)
    p.add_argument("--leverage", type=float, default=BinaryEdgeBaseDiagConfig.leverage)
    p.add_argument("--max-hold-bars", type=int, default=BinaryEdgeBaseDiagConfig.max_hold_bars)
    p.add_argument("--entry-delay-bars", type=int, default=BinaryEdgeBaseDiagConfig.entry_delay_bars)
    p.add_argument("--fee-rate", type=float, default=BinaryEdgeBaseDiagConfig.fee_rate)
    p.add_argument("--slippage-rate", type=float, default=BinaryEdgeBaseDiagConfig.slippage_rate)
    p.add_argument("--feature-prefixes", default=BinaryEdgeBaseDiagConfig.feature_prefixes)
    p.add_argument("--actions", default=BinaryEdgeBaseDiagConfig.actions)
    return p.parse_args()


def main() -> None:
    report = run(BinaryEdgeBaseDiagConfig(**vars(parse_args())))
    print(json.dumps({"feature_count": report["feature_count"], "evaluated_count": report["evaluated_count"], "top_by_test": report["top_by_test"][:5], "top_by_train": report["top_by_train"][:5], "top_stable": report["top_stable"][:5]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
