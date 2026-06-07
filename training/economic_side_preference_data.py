"""Build side preference pairs from realized LONG vs SHORT net returns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from training.economic_pressure_value_calibration import simulate_trade_return
from training.economic_stable_trader_sft_data import compact_prompt
from training.economic_fold_stability_sweep import load_all_rows
from training.strict_bar_backtest import load_market_bars


def side_target(side: str, risk: str = "MEDIUM") -> str:
    return json.dumps({"side": side, "risk": risk}, ensure_ascii=False, sort_keys=True)


def side_pref_prompt(row: dict[str, Any]) -> str:
    base = compact_prompt(row, policy_name="side_preference_h144_t1.8_s1.5")
    return base.replace(
        'Return exactly one JSON object: {"action": <LONG|SHORT|NO_TRADE>, "risk": <LOW|MEDIUM|HIGH>}',
        'Return exactly one JSON object: {"side": <LONG|SHORT>, "risk": <LOW|MEDIUM|HIGH>}. A trade has already been approved; choose the better side.',
    )


def split_for_date(date: str) -> str:
    if date < "2025-03-01":
        return "train"
    if date < "2025-09-01":
        return "val"
    return "eval"


def build_side_preferences(
    *,
    jsonl_paths: list[str],
    market_csv: str,
    min_abs_diff: float = 0.0005,
    horizon_bars: int = 144,
    target_pct: float = 1.8,
    stop_pct: float = 1.5,
    leverage: float = 0.5,
    fee_rate: float = 0.0004,
    slippage_rate: float = 0.0001,
    entry_delay_bars: int = 1,
) -> list[dict[str, Any]]:
    rows = load_all_rows(jsonl_paths)
    market = load_market_bars(market_csv)
    out: list[dict[str, Any]] = []
    for row in rows:
        long_ret = simulate_trade_return(row, market, side="LONG", horizon_bars=horizon_bars, target_pct=target_pct, stop_pct=stop_pct, leverage=leverage, fee_rate=fee_rate, slippage_rate=slippage_rate, entry_delay_bars=entry_delay_bars)
        short_ret = simulate_trade_return(row, market, side="SHORT", horizon_bars=horizon_bars, target_pct=target_pct, stop_pct=stop_pct, leverage=leverage, fee_rate=fee_rate, slippage_rate=slippage_rate, entry_delay_bars=entry_delay_bars)
        if long_ret is None or short_ret is None:
            continue
        diff = float(long_ret) - float(short_ret)
        if abs(diff) < float(min_abs_diff):
            continue
        chosen_side = "LONG" if diff > 0 else "SHORT"
        rejected_side = "SHORT" if chosen_side == "LONG" else "LONG"
        out.append({
            "date": row["date"],
            "signal_pos": row.get("signal_pos"),
            "split": split_for_date(str(row["date"])),
            "task": "side_preference_dpo",
            "prompt": side_pref_prompt(row),
            "chosen": side_target(chosen_side),
            "rejected": side_target(rejected_side),
            "chosen_side": chosen_side,
            "rejected_side": rejected_side,
            "long_reward": long_ret,
            "short_reward": short_ret,
            "reward_diff": abs(diff),
            "leakage_guard": {
                "prompt_uses_future_path": False,
                "preference_uses_future_path_for_training_only": True,
                "both_sides_simulated_with_same_entry_and_costs": True,
            },
        })
    return out


def write_splits(rows: list[dict[str, Any]], *, output_prefix: str) -> dict[str, str]:
    paths = {}
    for split in ["train", "val", "eval", "all"]:
        subset = rows if split == "all" else [r for r in rows if r["split"] == split]
        path = f"{output_prefix}_{split}.jsonl"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in subset) + ("\n" if subset else ""))
        paths[split] = path
    return paths


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"rows": len(rows), "by_split": {}, "by_chosen_side": {}}
    diffs = []
    for r in rows:
        out["by_split"][r["split"]] = out["by_split"].get(r["split"], 0) + 1
        out["by_chosen_side"][r["chosen_side"]] = out["by_chosen_side"].get(r["chosen_side"], 0) + 1
        diffs.append(float(r["reward_diff"]))
    if diffs:
        out["reward_diff_pct"] = {"min": min(diffs) * 100.0, "max": max(diffs) * 100.0, "mean": sum(diffs) / len(diffs) * 100.0}
    return out


def run_export(*, jsonl_paths: list[str], market_csv: str, output_prefix: str, summary_output: str, min_abs_diff: float) -> dict[str, Any]:
    rows = build_side_preferences(jsonl_paths=jsonl_paths, market_csv=market_csv, min_abs_diff=min_abs_diff)
    paths = write_splits(rows, output_prefix=output_prefix)
    report = {"paths": paths, "summary": summarize(rows), "config": {"min_abs_diff": min_abs_diff, "horizon_bars": 144, "target_pct": 1.8, "stop_pct": 1.5}}
    Path(summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl-paths", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output-prefix", required=True)
    p.add_argument("--summary-output", required=True)
    p.add_argument("--min-abs-diff", type=float, default=0.0005)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(run_export(jsonl_paths=[x for x in args.jsonl_paths.split(",") if x], market_csv=args.market_csv, output_prefix=args.output_prefix, summary_output=args.summary_output, min_abs_diff=args.min_abs_diff), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
