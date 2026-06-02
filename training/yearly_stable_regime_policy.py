"""Fit calibrated symbolic policies that must be stable across train years."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from training.calibrated_regime_policy import CalibratedPolicyConfig, _aggregate_action, build_calibration_records, evaluate_rules
from training.text_analyzer_trader_data import load_market_frame
from training.text_step_analyzer_data import parse_hold_candidates


@dataclass(frozen=True)
class YearlyStableConfig:
    min_year_samples: int = 6
    min_year_mean_net: float = 0.0
    min_year_win_rate: float = 0.50
    max_year_mean_mae: float = 0.02


def _year(row: dict[str, Any]) -> int:
    return int(pd.to_datetime(row["date"]).year)


def _aggregate_action_for_year(rows: list[dict[str, Any]], action_key: str, year: int) -> dict[str, Any]:
    return _aggregate_action([r for r in rows if _year(r) == int(year)], action_key)


def fit_yearly_stable_rules(
    train_records: list[dict[str, Any]],
    cfg: CalibratedPolicyConfig,
    stable_cfg: YearlyStableConfig,
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in train_records:
        groups.setdefault(str(row["key"]), []).append(row)
    train_years = sorted({_year(r) for r in train_records})
    rules: dict[str, dict[str, Any]] = {}
    for key, rows in groups.items():
        action_keys = sorted({a for r in rows for a in r["actions"]})
        qualified: list[dict[str, Any]] = []
        for action_key in action_keys:
            overall = _aggregate_action(rows, action_key)
            if int(overall.get("samples", 0)) < int(cfg.min_train_samples):
                continue
            if not (
                float(overall["mean_net_return"]) >= float(cfg.min_train_mean_net)
                and float(overall["mean_utility"]) >= float(cfg.min_train_mean_utility)
                and float(overall["win_rate"]) >= float(cfg.min_train_win_rate)
                and float(overall["mean_mae"]) <= float(cfg.max_train_mean_mae)
            ):
                continue
            yearly = [_aggregate_action_for_year(rows, action_key, y) for y in train_years]
            if all(
                int(y.get("samples", 0)) >= int(stable_cfg.min_year_samples)
                and float(y["mean_net_return"]) >= float(stable_cfg.min_year_mean_net)
                and float(y["win_rate"]) >= float(stable_cfg.min_year_win_rate)
                and float(y["mean_mae"]) <= float(stable_cfg.max_year_mean_mae)
                for y in yearly
            ):
                qualified.append({**overall, "yearly": dict(zip([str(y) for y in train_years], yearly))})
        if not qualified:
            continue
        best = max(qualified, key=lambda c: (float(c["mean_utility"]), float(c["mean_net_return"]), float(c["win_rate"])))
        rules[key] = {"key": key, "train_samples_in_group": len(rows), "action": best, "qualified_actions": qualified[:5]}
    return rules


def run_yearly_stable_policy(
    *,
    market_csv: str,
    output: str,
    wave_trading_root: str = "",
    train_start: str,
    train_end: str,
    eval_start: str,
    eval_end: str,
    stride_bars: int = 12,
    hold_candidates: str = "48,96,144,288",
    min_train_samples: int = 12,
    min_train_mean_net: float = 0.0,
    min_train_mean_utility: float = -0.002,
    min_train_win_rate: float = 0.50,
    max_train_mean_mae: float = 0.02,
    key_fields: str = "regime,trend_alignment,location,risk_state",
    min_year_samples: int = 6,
    min_year_mean_net: float = 0.0,
    min_year_win_rate: float = 0.50,
    max_year_mean_mae: float = 0.02,
) -> dict[str, Any]:
    cfg = CalibratedPolicyConfig(
        hold_candidates=parse_hold_candidates(hold_candidates),
        min_train_samples=int(min_train_samples),
        min_train_mean_net=float(min_train_mean_net),
        min_train_mean_utility=float(min_train_mean_utility),
        min_train_win_rate=float(min_train_win_rate),
        max_train_mean_mae=float(max_train_mean_mae),
        key_fields=tuple(x.strip() for x in str(key_fields).split(",") if x.strip()),
    )
    stable_cfg = YearlyStableConfig(
        min_year_samples=int(min_year_samples),
        min_year_mean_net=float(min_year_mean_net),
        min_year_win_rate=float(min_year_win_rate),
        max_year_mean_mae=float(max_year_mean_mae),
    )
    market = load_market_frame(market_csv, wave_trading_root=wave_trading_root or None)
    train_records = build_calibration_records(market, cfg, start_date=train_start, end_date=train_end, stride_bars=stride_bars)
    eval_records = build_calibration_records(market, cfg, start_date=eval_start, end_date=eval_end, stride_bars=stride_bars)
    rules = fit_yearly_stable_rules(train_records, cfg, stable_cfg)
    report = {
        "market_csv": str(Path(market_csv).resolve()),
        "config": asdict(cfg),
        "stable_config": asdict(stable_cfg),
        "periods": {"train": [train_start, train_end], "eval": [eval_start, eval_end]},
        "records": {"train": len(train_records), "eval": len(eval_records)},
        "rules": {"count": len(rules), "items": list(rules.values())[:50]},
        "train_strict_metrics": evaluate_rules(train_records, rules, non_overlapping=True, include_intratrade_mdd=True),
        "eval_strict_metrics": evaluate_rules(eval_records, rules, non_overlapping=True, include_intratrade_mdd=True),
        "leakage_guard": {"rules_fit_on_train_period_only": True, "yearly_stability_uses_train_years_only": True},
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit/evaluate yearly-stable calibrated symbolic policy")
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--wave-trading-root", default="")
    p.add_argument("--train-start", required=True)
    p.add_argument("--train-end", required=True)
    p.add_argument("--eval-start", required=True)
    p.add_argument("--eval-end", required=True)
    p.add_argument("--stride-bars", type=int, default=12)
    p.add_argument("--hold-candidates", default="48,96,144,288")
    p.add_argument("--min-train-samples", type=int, default=12)
    p.add_argument("--min-train-mean-net", type=float, default=0.0)
    p.add_argument("--min-train-mean-utility", type=float, default=-0.002)
    p.add_argument("--min-train-win-rate", type=float, default=0.50)
    p.add_argument("--max-train-mean-mae", type=float, default=0.02)
    p.add_argument("--key-fields", default="regime,trend_alignment,location,risk_state")
    p.add_argument("--min-year-samples", type=int, default=6)
    p.add_argument("--min-year-mean-net", type=float, default=0.0)
    p.add_argument("--min-year-win-rate", type=float, default=0.50)
    p.add_argument("--max-year-mean-mae", type=float, default=0.02)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run_yearly_stable_policy(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
