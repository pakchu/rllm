"""Strict portfolio backtest for fixed setup-quality episode rules.

Rules are predeclared as ``event@horizon:feature=bucket``.  Bucket thresholds
are fitted from train triggers only, then applied unchanged to train/test/eval.
The script does no ranking and does not use eval for rule selection.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.alpha_linear_combo_scan import _load_market, _parse_list
from training.audit_setup_quality_filters import _bucket_masks, _policy_cfg, _quality_frame, _rows_to_triggers
from training.fixed_episode_template_backtest import _parse_specs
from training.price_action_episode_policy import add_sequence_context_features, build_episode_event_features, simulate_triggers


@dataclass(frozen=True)
class SetupQualityPortfolioCfg:
    input_csv: str
    output: str
    rules: str
    train_start: str = "2024-01-01"
    train_end: str = "2024-12-31 23:59:59"
    test_start: str = "2025-01-01"
    test_end: str = "2025-12-31 23:59:59"
    eval_start: str = "2026-01-01"
    eval_end: str = "2026-06-01"
    windows: str = "36,72,144,288,576,2016,4032,8640"
    include_sequence_context: bool = True
    entry_delay_bars: int = 1
    leverage: float = 0.5
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    max_trigger_overlap: float = 0.80


def _parse_rules(raw: str) -> list[dict[str, str]]:
    rules = []
    for chunk in [x.strip() for x in raw.split(",") if x.strip()]:
        spec_part, sep, filter_part = chunk.partition(":")
        if not sep or "=" not in filter_part:
            raise ValueError(f"rule must be event@horizon:feature=bucket, got {chunk!r}")
        feature, bucket = filter_part.split("=", 1)
        spec = _parse_specs(spec_part)[0]
        rules.append({"spec": spec, "feature": feature.strip(), "bucket": bucket.strip(), "raw": chunk})
    if not rules:
        raise ValueError("at least one rule is required")
    return rules


def _date_mask(dates: pd.Series, start: str, end: str) -> np.ndarray:
    return np.asarray((dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end)), dtype=bool)


def _overlap(a: set[int], b: set[int]) -> float:
    return len(a & b) / max(1, len(a | b))


def _select_rule_rows(qf: pd.DataFrame, train_mask_all: np.ndarray, rule: dict[str, Any]) -> pd.DataFrame:
    spec = rule["spec"]
    feature = str(rule["feature"])
    bucket = str(rule["bucket"])
    sub = qf[(qf["event"] == spec["event"]) & (qf["horizon"] == spec["horizon"]) & (qf["side"] == spec["side"])].copy()
    if sub.empty:
        return sub
    if feature not in sub.columns:
        raise ValueError(f"unknown quality feature {feature!r}")
    train_vals = sub.loc[train_mask_all[sub.index.to_numpy()], feature]
    masks = _bucket_masks(train_vals, sub[feature])
    if bucket not in masks:
        raise ValueError(f"unknown bucket {bucket!r}; expected one of {sorted(masks)}")
    return sub.loc[masks[bucket]].copy()


def run(cfg: SetupQualityPortfolioCfg) -> dict[str, Any]:
    market = _load_market(cfg.input_csv)
    dates = pd.to_datetime(market["date"])
    windows = _parse_list(cfg.windows, int)
    features = build_episode_event_features(market, windows)
    if cfg.include_sequence_context:
        features = add_sequence_context_features(market, features, windows)
    rules = _parse_rules(cfg.rules)
    specs = [r["spec"] for r in rules]
    qf = _quality_frame(market, specs, features, dates, cfg)  # type: ignore[arg-type]
    q_dates = pd.to_datetime(qf["date"])
    train_mask_all = _date_mask(q_dates, cfg.train_start, cfg.train_end)

    selected = []
    rejected = []
    all_triggers: list[dict[str, Any]] = []
    position_sets: list[set[int]] = []
    policy_cfg = _policy_cfg(cfg)  # type: ignore[arg-type]
    for rule in rules:
        rows = _select_rule_rows(qf, train_mask_all, rule)
        positions = set(int(x) for x in rows["pos"].tolist()) if len(rows) else set()
        overlaps = [
            {"selected_rule": selected[i]["raw"], "jaccard": _overlap(positions, prev)}
            for i, prev in enumerate(position_sets)
            if positions and _overlap(positions, prev) > float(cfg.max_trigger_overlap)
        ]
        if overlaps:
            rejected.append({"rule": rule["raw"], "reason": "trigger_overlap_above_max", "overlaps": overlaps})
            continue
        triggers = _rows_to_triggers(rows)
        selected.append({"raw": rule["raw"], "spec": rule["spec"], "feature": rule["feature"], "bucket": rule["bucket"], "trigger_rows": len(rows)})
        position_sets.append(positions)
        all_triggers.extend(triggers)

    portfolio = {
        "train": simulate_triggers(market, dates, all_triggers, start=cfg.train_start, end=cfg.train_end, cfg=policy_cfg),
        "test": simulate_triggers(market, dates, all_triggers, start=cfg.test_start, end=cfg.test_end, cfg=policy_cfg),
        "eval": simulate_triggers(market, dates, all_triggers, start=cfg.eval_start, end=cfg.eval_end, cfg=policy_cfg),
    }
    per_rule = []
    for rule in rules:
        rows = _select_rule_rows(qf, train_mask_all, rule)
        triggers = _rows_to_triggers(rows)
        per_rule.append({
            "rule": rule["raw"],
            "train": {"sim": simulate_triggers(market, dates, triggers, start=cfg.train_start, end=cfg.train_end, cfg=policy_cfg)["sim"]},
            "test": {"sim": simulate_triggers(market, dates, triggers, start=cfg.test_start, end=cfg.test_end, cfg=policy_cfg)["sim"]},
            "eval_diagnostic": {"sim": simulate_triggers(market, dates, triggers, start=cfg.eval_start, end=cfg.eval_end, cfg=policy_cfg)["sim"]},
        })
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "selected_rules": selected,
        "rejected_rules": rejected,
        "per_rule": per_rule,
        "portfolio": {k: {"period": v["period"], "sim": v["sim"], "trade_stats": v["trade_stats"], "executed_sample": v["executed"][:20]} for k, v in portfolio.items()},
        "protocol": "fixed predeclared setup-quality rules; train-only bucket thresholds; no ranking or eval selection",
        "leakage_guard": {"bucket_thresholds_fit_on_eval": False, "rule_selection_uses_eval": False, "entry_uses_next_open": int(cfg.entry_delay_bars) >= 1},
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--rules", required=True)
    for name in ("train-start", "train-end", "test-start", "test-end", "eval-start", "eval-end"):
        p.add_argument(f"--{name}", default=getattr(SetupQualityPortfolioCfg, name.replace("-", "_")))
    p.add_argument("--windows", default=SetupQualityPortfolioCfg.windows)
    p.add_argument("--no-sequence-context", dest="include_sequence_context", action="store_false")
    p.set_defaults(include_sequence_context=SetupQualityPortfolioCfg.include_sequence_context)
    p.add_argument("--entry-delay-bars", type=int, default=SetupQualityPortfolioCfg.entry_delay_bars)
    p.add_argument("--leverage", type=float, default=SetupQualityPortfolioCfg.leverage)
    p.add_argument("--fee-rate", type=float, default=SetupQualityPortfolioCfg.fee_rate)
    p.add_argument("--slippage-rate", type=float, default=SetupQualityPortfolioCfg.slippage_rate)
    p.add_argument("--max-trigger-overlap", type=float, default=SetupQualityPortfolioCfg.max_trigger_overlap)
    return p.parse_args()


def main() -> None:
    r = run(SetupQualityPortfolioCfg(**vars(parse_args())))
    print(json.dumps({
        "output": r["config"]["output"],
        "selected_rules": r["selected_rules"],
        "rejected_rules": r["rejected_rules"],
        "portfolio": {k: v["sim"] | {"p": v["trade_stats"].get("p_value_mean_ret_approx")} for k, v in r["portfolio"].items()},
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
