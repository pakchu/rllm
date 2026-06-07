"""Build out-of-fold stable trader SFT/RL data from the fold-stable baseline policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from training.economic_fold_stability_sweep import default_folds, load_all_rows, split_rows
from training.economic_pressure_value_calibration import choose_action, context_values, fit_tables, simulate_trade_return
from training.strict_bar_backtest import load_market_bars

ACTION_TO_PRESSURE = {"LONG": "LONG_FAVORED", "SHORT": "SHORT_FAVORED", "NO_TRADE": "NO_TRADE_FAVORED", "NONE": "NO_TRADE_FAVORED"}


def reward_bucket(reward: float | None) -> str:
    if reward is None:
        return "NO_TRADE"
    pct = reward * 100.0
    if pct >= 0.4:
        return "HIGH_WIN"
    if pct > 0.05:
        return "SMALL_WIN"
    if pct >= -0.05:
        return "FLAT"
    if pct > -0.4:
        return "SMALL_LOSS"
    return "LARGE_LOSS"


def compact_prompt(row: dict[str, Any], *, policy_name: str) -> str:
    vals = context_values(row)
    compact = {
        "teacher_pressure": vals.get("teacher_pressure"),
        "teacher_conf_bucket": vals.get("teacher_conf_bucket"),
        "teacher_margin_bucket": vals.get("teacher_margin_bucket"),
        "regime": vals.get("regime"),
        "trend_alignment": vals.get("trend_alignment"),
        "momentum": vals.get("momentum"),
        "oscillator": vals.get("oscillator"),
        "location": vals.get("location"),
        "volatility_level": vals.get("volatility_level"),
        "risk_state": vals.get("risk_state"),
        "order_flow": vals.get("order_flow"),
        "kimchi": vals.get("kimchi"),
        "macro_dollar": vals.get("macro_dollar"),
        "range_pos_bucket": vals.get("range_pos_bucket"),
        "drawdown_bucket": vals.get("drawdown_bucket"),
        "seq_bias": vals.get("seq_bias"),
        "tags": vals.get("tags"),
    }
    return (
        "You are the trader stage after a compact analyzer. Choose one BTCUSDT futures action for the configured horizon.\n"
        "Return exactly one JSON object: {\"action\": <LONG|SHORT|NO_TRADE>, \"risk\": <LOW|MEDIUM|HIGH>}\n"
        "Prefer NO_TRADE unless expected net value after costs is positive and fold-stable.\n"
        f"Policy anchor: {policy_name}.\n"
        f"Past-only analyzer context: {json.dumps(compact, ensure_ascii=False, sort_keys=True)}"
    )


def risk_from_score(info: dict[str, Any]) -> str:
    score = info.get("score")
    if score is None:
        return "HIGH"
    try:
        s = float(score)
    except Exception:
        return "HIGH"
    if s >= 0.002:
        return "LOW"
    if s >= 0.0005:
        return "MEDIUM"
    return "HIGH"


def make_target(action: str, info: dict[str, Any]) -> dict[str, Any]:
    public_action = "NO_TRADE" if action in {"NONE", "NO_TRADE"} else action
    return {"action": public_action, "risk": risk_from_score(info)}


def realized_reward_for_action(row: dict[str, Any], market, *, action: str, horizon_bars: int, target_pct: float, stop_pct: float, leverage: float, fee_rate: float, slippage_rate: float, entry_delay_bars: int) -> float | None:
    if action in {"NONE", "NO_TRADE"}:
        return None
    return simulate_trade_return(row, market, side=action, horizon_bars=horizon_bars, target_pct=target_pct, stop_pct=stop_pct, leverage=leverage, fee_rate=fee_rate, slippage_rate=slippage_rate, entry_delay_bars=entry_delay_bars)


def split_name_for_fold(fold_name: str) -> str:
    if fold_name in {"2024_h1", "2024_h2_to_2025_feb"}:
        return "train"
    if fold_name == "2025_h1_val":
        return "val"
    return "eval"


def build_stable_trader_rows(
    *,
    jsonl_paths: list[str],
    market_csv: str,
    horizon_bars: int = 144,
    target_pct: float = 1.8,
    stop_pct: float = 1.5,
    level: str = "teacher_only",
    min_n: int = 20,
    min_score: float = 0.0005,
    score_mode: str = "mean",
    side_gate: str = "free",
    leverage: float = 0.5,
    fee_rate: float = 0.0004,
    slippage_rate: float = 0.0001,
    entry_delay_bars: int = 1,
) -> list[dict[str, Any]]:
    rows = load_all_rows(jsonl_paths)
    market = load_market_bars(market_csv)
    out: list[dict[str, Any]] = []
    policy_name = f"stable_h{horizon_bars}_t{target_pct}_s{stop_pct}_{level}_n{min_n}_score{min_score}_{score_mode}_{side_gate}"
    for fold in default_folds():
        train_rows, test_rows = split_rows(rows, train_end=fold["train_end"], test_start=fold["test_start"], test_end=fold["test_end"])
        tables = fit_tables(train_rows, market, horizon_bars=horizon_bars, target_pct=target_pct, stop_pct=stop_pct, leverage=leverage, fee_rate=fee_rate, slippage_rate=slippage_rate, entry_delay_bars=entry_delay_bars)
        for row in test_rows:
            action, info = choose_action(row, tables, level=level, min_n=min_n, min_score=min_score, score_mode=score_mode, side_gate=side_gate)
            target = make_target(action, info)
            reward = realized_reward_for_action(row, market, action=target["action"], horizon_bars=horizon_bars, target_pct=target_pct, stop_pct=stop_pct, leverage=leverage, fee_rate=fee_rate, slippage_rate=slippage_rate, entry_delay_bars=entry_delay_bars)
            out.append({
                "date": row["date"],
                "signal_pos": row.get("signal_pos"),
                "fold": fold["name"],
                "split": split_name_for_fold(fold["name"]),
                "task": "stable_trader_policy_sft_rl",
                "prompt": compact_prompt(row, policy_name=policy_name),
                "target": json.dumps(target, ensure_ascii=False, sort_keys=True),
                "action": target["action"],
                "policy_score": info.get("score"),
                "policy_level": info.get("used_level"),
                "bucket_n": info.get("bucket_n"),
                "bucket_mean": info.get("bucket_mean"),
                "bucket_lower95": info.get("bucket_lower95"),
                "realized_reward": reward,
                "reward_bucket": reward_bucket(reward),
                "leakage_guard": {
                    "prompt_uses_future_path": False,
                    "action_label_fit_uses_only_rows_at_or_before_fold_train_end": True,
                    "reward_uses_future_path_for_training_or_analysis_only": True,
                    "fold_train_end": fold["train_end"],
                },
            })
    return out


def write_split_files(rows: list[dict[str, Any]], *, output_prefix: str) -> dict[str, str]:
    paths = {}
    for split in ["train", "val", "eval", "all"]:
        subset = rows if split == "all" else [r for r in rows if r["split"] == split]
        path = f"{output_prefix}_{split}.jsonl"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in subset) + ("\n" if subset else ""))
        paths[split] = path
    return paths


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"total": len(rows), "by_split": {}, "by_action": {}, "by_reward_bucket": {}}
    for r in rows:
        summary["by_split"][r["split"]] = summary["by_split"].get(r["split"], 0) + 1
        summary["by_action"][r["action"]] = summary["by_action"].get(r["action"], 0) + 1
        summary["by_reward_bucket"][r["reward_bucket"]] = summary["by_reward_bucket"].get(r["reward_bucket"], 0) + 1
    for split in ["train", "val", "eval"]:
        subset = [r for r in rows if r["split"] == split and r["realized_reward"] is not None]
        if subset:
            rewards = [float(r["realized_reward"]) for r in subset]
            summary["by_split"][f"{split}_trades"] = len(rewards)
            summary["by_split"][f"{split}_mean_reward_pct"] = sum(rewards) / len(rewards) * 100.0
    return summary


def run_export(*, jsonl_paths: list[str], market_csv: str, output_prefix: str, summary_output: str) -> dict[str, Any]:
    rows = build_stable_trader_rows(jsonl_paths=jsonl_paths, market_csv=market_csv)
    paths = write_split_files(rows, output_prefix=output_prefix)
    report = {"paths": paths, "summary": summarize(rows), "policy": {"horizon_bars": 144, "target_pct": 1.8, "stop_pct": 1.5, "level": "teacher_only", "min_n": 20, "min_score": 0.0005, "score_mode": "mean", "side_gate": "free"}}
    Path(summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl-paths", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output-prefix", required=True)
    p.add_argument("--summary-output", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = run_export(jsonl_paths=[x for x in args.jsonl_paths.split(",") if x], market_csv=args.market_csv, output_prefix=args.output_prefix, summary_output=args.summary_output)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
