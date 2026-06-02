"""Export LLM SFT rows that imitate a train-calibrated symbolic policy.

Unlike future-optimal labels, these targets are produced by rules fit only on the
calibration train period.  The trader stage learns to reproduce a stable policy
from the analyzer summary rather than predict each sample's future path.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from training.calibrated_regime_policy import CalibratedPolicyConfig, build_calibration_records, fit_rules
from training.text_analyzer_trader_data import analyzer_summary_to_text, build_analyzer_input, load_market_frame, write_jsonl
from training.text_step_analyzer_data import parse_hold_candidates


def build_policy_trader_input(summary_text: str, *, hold_candidates: tuple[int, ...], entry_delay_bars: int) -> str:
    return "\n".join(
        [
            "You are the trader stage for a BTCUSDT futures trading bot.",
            "You receive only the analyzer's past-only symbolic market summary.",
            "Imitate the train-calibrated symbolic policy. Do not invent trades outside the policy.",
            f"Execution: enter after {int(entry_delay_bars)} bar(s). Allowed hold_bars: {','.join(str(x) for x in hold_candidates)}.",
            "Output exactly one JSON object with keys gate, side, hold_bars, policy_key, reason.",
            "gate must be TRADE or NO_TRADE. side must be LONG, SHORT, or NONE. hold_bars is 0 for NO_TRADE.",
            "",
            f"Analyzer summary: {summary_text}",
        ]
    )


def _policy_target_for_record(
    row: dict[str, Any],
    rules: dict[str, dict[str, Any]],
    *,
    next_available_pos: int,
) -> tuple[dict[str, Any], int]:
    signal_pos = int(row["signal_pos"])
    if signal_pos <= next_available_pos:
        return {
            "gate": "NO_TRADE",
            "side": "NONE",
            "hold_bars": 0,
            "policy_key": str(row["key"]),
            "reason": "POSITION_OPEN_SKIP",
        }, next_available_pos
    rule = rules.get(str(row["key"]))
    if not rule:
        return {
            "gate": "NO_TRADE",
            "side": "NONE",
            "hold_bars": 0,
            "policy_key": str(row["key"]),
            "reason": "NO_CALIBRATED_EDGE",
        }, next_available_pos
    action = rule["action"]
    hold_bars = int(action["hold_bars"])
    return {
        "gate": "TRADE",
        "side": str(action["side"]),
        "hold_bars": hold_bars,
        "policy_key": str(row["key"]),
        "reason": "CALIBRATED_EDGE",
    }, signal_pos + hold_bars


def build_calibrated_policy_sft_rows(
    market,
    cfg: CalibratedPolicyConfig,
    *,
    train_start: str,
    train_end: str,
    label_start: str,
    label_end: str,
    stride_bars: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train_records = build_calibration_records(market, cfg, start_date=train_start, end_date=train_end, stride_bars=stride_bars)
    label_records = build_calibration_records(market, cfg, start_date=label_start, end_date=label_end, stride_bars=stride_bars)
    rules = fit_rules(train_records, cfg)
    analyzer_rows: list[dict[str, Any]] = []
    trader_rows: list[dict[str, Any]] = []
    next_available_pos = -1
    target_counts: dict[str, int] = {}
    for row in label_records:
        summary_text = analyzer_summary_to_text(row["summary"])
        target, next_available_pos = _policy_target_for_record(row, rules, next_available_pos=next_available_pos)
        target_counts[str(target["reason"])] = target_counts.get(str(target["reason"]), 0) + 1
        analyzer_rows.append(
            {
                "task": "calibrated_policy_analyzer",
                "date": row["date"],
                "signal_pos": int(row["signal_pos"]),
                "prompt": build_analyzer_input(market, int(row["signal_pos"]), window_size=cfg.window_size),
                "target": summary_text,
                "policy_key": str(row["key"]),
                "leakage_guard": {"target_uses_future_path": False, "input_bars_end_at_signal_pos": int(row["signal_pos"])},
            }
        )
        trader_rows.append(
            {
                "task": "calibrated_policy_trader",
                "date": row["date"],
                "signal_pos": int(row["signal_pos"]),
                "prompt": build_policy_trader_input(
                    summary_text, hold_candidates=cfg.hold_candidates, entry_delay_bars=cfg.entry_delay_bars
                ),
                "target": json.dumps(target, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
                "trade_gate_label": target["gate"],
                "trade_side_label": target["side"],
                "hold_bars_label": int(target["hold_bars"]),
                "policy_key": str(row["key"]),
                "leakage_guard": {
                    "rules_fit_on_train_period_only": True,
                    "prompt_uses_analyzer_summary_only": True,
                    "label_is_train_calibrated_policy_not_sample_future_optimum": True,
                },
            }
        )
    summary = {
        "config": asdict(cfg),
        "periods": {"train": [train_start, train_end], "label": [label_start, label_end]},
        "records": {"train": len(train_records), "label": len(label_records)},
        "rules_count": len(rules),
        "target_counts": target_counts,
        "rules_preview": list(rules.values())[:20],
    }
    return analyzer_rows, trader_rows, summary


def run_export(
    *,
    market_csv: str,
    analyzer_output: str,
    trader_output: str,
    summary_output: str = "",
    wave_trading_root: str = "",
    train_start: str,
    train_end: str,
    label_start: str,
    label_end: str,
    stride_bars: int = 12,
    hold_candidates: str = "48,96,144,288",
    min_train_samples: int = 12,
    min_train_mean_net: float = 0.0,
    min_train_mean_utility: float = -0.002,
    min_train_win_rate: float = 0.55,
    max_train_mean_mae: float = 0.015,
    key_fields: str = "regime,trend_alignment,location,risk_state",
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
    analyzer_rows, trader_rows, summary = build_calibrated_policy_sft_rows(
        market,
        cfg,
        train_start=train_start,
        train_end=train_end,
        label_start=label_start,
        label_end=label_end,
        stride_bars=int(stride_bars),
    )
    write_jsonl(analyzer_output, analyzer_rows)
    write_jsonl(trader_output, trader_rows)
    if summary_output:
        Path(summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(summary_output).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export calibrated-policy analyzer/trader SFT data")
    p.add_argument("--market-csv", required=True)
    p.add_argument("--analyzer-output", required=True)
    p.add_argument("--trader-output", required=True)
    p.add_argument("--summary-output", default="")
    p.add_argument("--wave-trading-root", default="")
    p.add_argument("--train-start", required=True)
    p.add_argument("--train-end", required=True)
    p.add_argument("--label-start", required=True)
    p.add_argument("--label-end", required=True)
    p.add_argument("--stride-bars", type=int, default=12)
    p.add_argument("--hold-candidates", default="48,96,144,288")
    p.add_argument("--min-train-samples", type=int, default=12)
    p.add_argument("--min-train-mean-net", type=float, default=0.0)
    p.add_argument("--min-train-mean-utility", type=float, default=-0.002)
    p.add_argument("--min-train-win-rate", type=float, default=0.55)
    p.add_argument("--max-train-mean-mae", type=float, default=0.015)
    p.add_argument("--key-fields", default="regime,trend_alignment,location,risk_state")
    return p.parse_args()


def main() -> None:
    print(json.dumps(run_export(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
