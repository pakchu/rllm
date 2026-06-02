"""Fit leakage-safe regime-calibrated actions from train-period path outcomes.

This stage stops asking the LLM to predict each sample's future-optimal label.
Instead, it finds symbolic analyzer-summary groups whose historical train-period
realized action outcomes are positive, then applies those fixed group actions to
a later eval period.  If no such group policy has edge, there is no useful label
for the LLM to imitate.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from preprocessing.market_features import build_market_feature_frame
from training.path_outcome_dataset import PathOutcomeConfig, compute_trade_path_outcome
from training.text_analyzer_trader_data import build_analyzer_summary, load_market_frame
from training.text_step_analyzer_data import StepAnalyzerConfig, _iter_positions, parse_hold_candidates


@dataclass(frozen=True)
class CalibratedPolicyConfig:
    window_size: int = 96
    hold_candidates: tuple[int, ...] = (48, 96, 144, 288)
    entry_delay_bars: int = 1
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    leverage: float = 1.0
    mae_penalty: float = 1.0
    min_train_samples: int = 24
    min_train_mean_net: float = 0.001
    min_train_mean_utility: float = 0.0
    min_train_win_rate: float = 0.52
    max_train_mean_mae: float = 0.01
    key_fields: tuple[str, ...] = ("regime", "trend_alignment", "location", "oscillator", "risk_state")


def _path_cfg(cfg: CalibratedPolicyConfig, hold_bars: int) -> PathOutcomeConfig:
    return PathOutcomeConfig(
        hold_bars=int(hold_bars),
        entry_delay_bars=cfg.entry_delay_bars,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
        leverage=cfg.leverage,
        mae_penalty=cfg.mae_penalty,
    )


def _summary_key(summary: dict[str, Any], key_fields: tuple[str, ...]) -> str:
    parts: list[str] = []
    symbolic = summary.get("symbolic_features", {}) if isinstance(summary.get("symbolic_features", {}), dict) else {}
    for field in key_fields:
        value = summary.get(field, symbolic.get(field, ""))
        parts.append(f"{field}={value}")
    return "|".join(parts)


def _record_actions(market: pd.DataFrame, pos: int, cfg: CalibratedPolicyConfig) -> dict[str, dict[str, float | str | int]]:
    actions: dict[str, dict[str, float | str | int]] = {}
    for hold in cfg.hold_candidates:
        pcfg = _path_cfg(cfg, hold)
        for side in ("LONG", "SHORT"):
            outcome = compute_trade_path_outcome(market, pos, side, pcfg)
            if outcome is None:
                continue
            key = f"{side}_{int(hold)}"
            actions[key] = {
                "side": side,
                "hold_bars": int(hold),
                "net_return": float(outcome.net_return),
                "mae": float(outcome.mae),
                "utility": float(outcome.utility),
            }
    return actions


def build_calibration_records(
    market: pd.DataFrame,
    cfg: CalibratedPolicyConfig,
    *,
    start_date: str,
    end_date: str,
    stride_bars: int,
) -> list[dict[str, Any]]:
    step_cfg = StepAnalyzerConfig(
        window_size=cfg.window_size,
        hold_bars=max(cfg.hold_candidates),
        hold_candidates=cfg.hold_candidates,
        entry_delay_bars=cfg.entry_delay_bars,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
        leverage=cfg.leverage,
        mae_penalty=cfg.mae_penalty,
        stride_bars=int(stride_bars),
    )
    feature_frame = build_market_feature_frame(market, window_size=cfg.window_size)
    records: list[dict[str, Any]] = []
    for pos in _iter_positions(market, step_cfg, start_date, end_date):
        summary = build_analyzer_summary(market, pos, window_size=cfg.window_size, feature_frame=feature_frame)
        actions = _record_actions(market, pos, cfg)
        if not actions:
            continue
        records.append(
            {
                "date": str(pd.to_datetime(market.iloc[int(pos)]["date"])),
                "signal_pos": int(pos),
                "key": _summary_key(summary, cfg.key_fields),
                "summary": summary,
                "actions": actions,
            }
        )
    return records


def _aggregate_action(rows: list[dict[str, Any]], action_key: str) -> dict[str, Any]:
    vals = [r["actions"][action_key] for r in rows if action_key in r["actions"]]
    n = len(vals)
    if not n:
        return {"samples": 0}
    nets = [float(v["net_return"]) for v in vals]
    maes = [float(v["mae"]) for v in vals]
    utils = [float(v["utility"]) for v in vals]
    return {
        "samples": n,
        "side": str(vals[0]["side"]),
        "hold_bars": int(vals[0]["hold_bars"]),
        "mean_net_return": sum(nets) / n,
        "mean_mae": sum(maes) / n,
        "mean_utility": sum(utils) / n,
        "win_rate": sum(1 for x in nets if x > 0.0) / n,
    }


def fit_rules(train_records: list[dict[str, Any]], cfg: CalibratedPolicyConfig) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in train_records:
        groups.setdefault(str(row["key"]), []).append(row)
    rules: dict[str, dict[str, Any]] = {}
    for key, rows in groups.items():
        action_keys = sorted({a for r in rows for a in r["actions"]})
        candidates = [_aggregate_action(rows, a) for a in action_keys]
        candidates = [c for c in candidates if int(c.get("samples", 0)) >= int(cfg.min_train_samples)]
        qualified = [
            c
            for c in candidates
            if float(c["mean_net_return"]) >= float(cfg.min_train_mean_net)
            and float(c["mean_utility"]) >= float(cfg.min_train_mean_utility)
            and float(c["win_rate"]) >= float(cfg.min_train_win_rate)
            and float(c["mean_mae"]) <= float(cfg.max_train_mean_mae)
        ]
        if not qualified:
            continue
        best = max(qualified, key=lambda c: (float(c["mean_utility"]), float(c["mean_net_return"]), float(c["win_rate"])))
        rules[key] = {"key": key, "train_samples_in_group": len(rows), "action": best, "qualified_actions": qualified[:5]}
    return rules


def evaluate_rules(records: list[dict[str, Any]], rules: dict[str, dict[str, Any]]) -> dict[str, Any]:
    trades: list[dict[str, Any]] = []
    for row in records:
        rule = rules.get(str(row["key"]))
        if not rule:
            continue
        action = rule["action"]
        action_key = f"{action['side']}_{int(action['hold_bars'])}"
        outcome = row["actions"].get(action_key)
        if outcome is None:
            continue
        trades.append({"date": row["date"], "key": row["key"], **outcome})
    n = len(records)
    t = len(trades)
    if not trades:
        return {"records": n, "trades": 0, "coverage": 0.0}
    nets = [float(x["net_return"]) for x in trades]
    maes = [float(x["mae"]) for x in trades]
    utils = [float(x["utility"]) for x in trades]
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in nets:
        equity *= 1.0 + r
        peak = max(peak, equity)
        max_dd = max(max_dd, 1.0 - equity / peak)
    return {
        "records": n,
        "trades": t,
        "coverage": t / max(1, n),
        "mean_net_return": sum(nets) / t,
        "mean_mae": sum(maes) / t,
        "mean_utility": sum(utils) / t,
        "win_rate": sum(1 for x in nets if x > 0.0) / t,
        "sum_net_return": sum(nets),
        "compounded_return": equity - 1.0,
        "strict_mdd_proxy": max_dd,
    }


def run_calibrated_policy(
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
    min_train_samples: int = 24,
    min_train_mean_net: float = 0.001,
    min_train_mean_utility: float = 0.0,
    min_train_win_rate: float = 0.52,
    max_train_mean_mae: float = 0.01,
    key_fields: str = "regime,trend_alignment,location,oscillator,risk_state",
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
    market = load_market_frame(market_csv, wave_trading_root=wave_trading_root or None)
    train_records = build_calibration_records(market, cfg, start_date=train_start, end_date=train_end, stride_bars=stride_bars)
    eval_records = build_calibration_records(market, cfg, start_date=eval_start, end_date=eval_end, stride_bars=stride_bars)
    rules = fit_rules(train_records, cfg)
    report = {
        "market_csv": str(Path(market_csv).resolve()),
        "config": asdict(cfg),
        "periods": {"train": [train_start, train_end], "eval": [eval_start, eval_end]},
        "records": {"train": len(train_records), "eval": len(eval_records)},
        "rules": {"count": len(rules), "items": list(rules.values())[:50]},
        "train_metrics": evaluate_rules(train_records, rules),
        "eval_metrics": evaluate_rules(eval_records, rules),
        "leakage_guard": {"rules_fit_on_train_period_only": True, "eval_uses_fixed_train_rules": True},
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit/evaluate train-calibrated symbolic regime actions")
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--wave-trading-root", default="")
    p.add_argument("--train-start", required=True)
    p.add_argument("--train-end", required=True)
    p.add_argument("--eval-start", required=True)
    p.add_argument("--eval-end", required=True)
    p.add_argument("--stride-bars", type=int, default=12)
    p.add_argument("--hold-candidates", default="48,96,144,288")
    p.add_argument("--min-train-samples", type=int, default=24)
    p.add_argument("--min-train-mean-net", type=float, default=0.001)
    p.add_argument("--min-train-mean-utility", type=float, default=0.0)
    p.add_argument("--min-train-win-rate", type=float, default=0.52)
    p.add_argument("--max-train-mean-mae", type=float, default=0.01)
    p.add_argument("--key-fields", default="regime,trend_alignment,location,oscillator,risk_state")
    return p.parse_args()


def main() -> None:
    print(json.dumps(run_calibrated_policy(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
