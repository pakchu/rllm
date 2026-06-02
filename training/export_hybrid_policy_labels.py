"""Export trader SFT rows for a fixed hybrid stable policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from training.calibrated_regime_policy import CalibratedPolicyConfig, _summary_key
from training.export_calibrated_policy_labels import build_policy_trader_input, format_policy_book
from training.sweep_calibrated_regime_policy import _copy_with_keys
from training.sweep_hybrid_stable_policy import _filter_uncovered_records, _load_top_rules
from training.sweep_yearly_stable_policy import _fit_from_stats, _precompute_group_year_stats, _records_cache_path
from training.text_analyzer_trader_data import analyzer_summary_to_text, write_jsonl
from training.text_step_analyzer_data import parse_hold_candidates
from training.yearly_stable_regime_policy import YearlyStableConfig


def _load_records_from_cache(
    cache_dir: str,
    *,
    split: str,
    start_date: str,
    end_date: str,
    stride_bars: int,
    cfg: CalibratedPolicyConfig,
) -> list[dict[str, Any]]:
    path = _records_cache_path(cache_dir, split=split, start_date=start_date, end_date=end_date, stride_bars=stride_bars, cfg=cfg)
    return json.loads(path.read_text())


def _fit_addon_rules(
    train_records: list[dict[str, Any]],
    *,
    base_rules: dict[str, dict[str, Any]],
    base_key_fields: tuple[str, ...],
    addon_config: dict[str, Any],
    addon_stable_config: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], tuple[str, ...]]:
    addon_key_fields = tuple(addon_config["key_fields"])
    cfg = CalibratedPolicyConfig(
        hold_candidates=tuple(int(x) for x in addon_config.get("hold_candidates", (48, 96, 144, 288))),
        min_train_samples=int(addon_config["min_train_samples"]),
        min_train_mean_net=float(addon_config["min_train_mean_net"]),
        min_train_mean_utility=float(addon_config["min_train_mean_utility"]),
        min_train_win_rate=float(addon_config["min_train_win_rate"]),
        max_train_mean_mae=float(addon_config["max_train_mean_mae"]),
        key_fields=addon_key_fields,
    )
    stable = YearlyStableConfig(**addon_stable_config)
    uncovered = _filter_uncovered_records(train_records, base_rules, base_key_fields)
    addon_records = _copy_with_keys(uncovered, addon_key_fields)
    return _fit_from_stats(_precompute_group_year_stats(addon_records), cfg, stable), addon_key_fields


def format_hybrid_policy_book(base_rules: dict[str, dict[str, Any]], addon_rules: dict[str, dict[str, Any]]) -> str:
    return "\n".join(
        [
            "Base policy has priority:",
            format_policy_book(base_rules),
            "",
            "Add-on policy applies only when the base current_policy_key is not listed:",
            format_policy_book(addon_rules),
        ]
    )


def _target_for_hybrid_record(
    row: dict[str, Any],
    *,
    base_rules: dict[str, dict[str, Any]],
    addon_rules: dict[str, dict[str, Any]],
    base_key_fields: tuple[str, ...],
    addon_key_fields: tuple[str, ...],
    next_available_pos: int,
) -> tuple[dict[str, Any], int]:
    signal_pos = int(row["signal_pos"])
    base_key = _summary_key(row["summary"], base_key_fields)
    addon_key = _summary_key(row["summary"], addon_key_fields)
    if signal_pos <= next_available_pos:
        return {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0, "policy_key": base_key, "reason": "POSITION_OPEN_SKIP"}, next_available_pos
    rule = base_rules.get(base_key)
    policy_key = base_key
    reason = "BASE_CALIBRATED_EDGE"
    if rule is None:
        rule = addon_rules.get(addon_key)
        policy_key = addon_key
        reason = "ADDON_CALIBRATED_EDGE"
    if rule is None:
        return {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0, "policy_key": base_key, "reason": "NO_HYBRID_EDGE"}, next_available_pos
    action = rule["action"]
    hold_bars = int(action["hold_bars"])
    return {"gate": "TRADE", "side": str(action["side"]), "hold_bars": hold_bars, "policy_key": policy_key, "reason": reason}, signal_pos + hold_bars


def run_export(
    *,
    output: str,
    summary_output: str,
    records_cache_dir: str,
    base_report: str,
    addon_report: str,
    train_start: str,
    train_end: str,
    label_start: str,
    label_end: str,
    stride_bars: int = 12,
    hold_candidates: str = "48,96,144,288",
    base_top_index: int = 0,
    include_base_rule_indices: str = "",
    addon_top_index: int = 0,
) -> dict[str, Any]:
    cfg = CalibratedPolicyConfig(hold_candidates=parse_hold_candidates(hold_candidates))
    train_records = _load_records_from_cache(records_cache_dir, split="train", start_date=train_start, end_date=train_end, stride_bars=stride_bars, cfg=cfg)
    label_records = _load_records_from_cache(records_cache_dir, split="eval", start_date=label_start, end_date=label_end, stride_bars=stride_bars, cfg=cfg)
    base_rules, base_key_fields = _load_top_rules(base_report, top_index=base_top_index, include_indices=include_base_rule_indices)
    addon_item = json.loads(Path(addon_report).read_text())["top"][int(addon_top_index)]
    addon_rules, addon_key_fields = _fit_addon_rules(
        train_records,
        base_rules=base_rules,
        base_key_fields=base_key_fields,
        addon_config=addon_item["addon_config"],
        addon_stable_config=addon_item["addon_stable_config"],
    )
    policy_book = format_hybrid_policy_book(base_rules, addon_rules)
    rows: list[dict[str, Any]] = []
    next_available_pos = -1
    target_counts: dict[str, int] = {}
    for row in label_records:
        summary_text = analyzer_summary_to_text(row["summary"])
        target, next_available_pos = _target_for_hybrid_record(
            row,
            base_rules=base_rules,
            addon_rules=addon_rules,
            base_key_fields=base_key_fields,
            addon_key_fields=addon_key_fields,
            next_available_pos=next_available_pos,
        )
        target_counts[str(target["reason"])] = target_counts.get(str(target["reason"]), 0) + 1
        base_key = _summary_key(row["summary"], base_key_fields)
        addon_key = _summary_key(row["summary"], addon_key_fields)
        rows.append(
            {
                "task": "hybrid_stable_policy_trader",
                "date": row["date"],
                "signal_pos": int(row["signal_pos"]),
                "prompt": build_policy_trader_input(
                    summary_text,
                    hold_candidates=cfg.hold_candidates,
                    entry_delay_bars=cfg.entry_delay_bars,
                    current_policy_key=f"base={base_key}; addon={addon_key}",
                    policy_book=policy_book,
                ),
                "target": json.dumps(target, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
                "trade_gate_label": target["gate"],
                "trade_side_label": target["side"],
                "hold_bars_label": int(target["hold_bars"]),
                "base_policy_key": base_key,
                "addon_policy_key": addon_key,
                "policy_key": target["policy_key"],
                "leakage_guard": {
                    "base_and_addon_rules_fit_on_train_period_only": True,
                    "base_priority_applied_before_addon": True,
                    "label_is_train_calibrated_hybrid_policy_not_sample_future_optimum": True,
                },
            }
        )
    write_jsonl(output, rows)
    summary = {
        "records": {"train": len(train_records), "label": len(label_records)},
        "periods": {"train": [train_start, train_end], "label": [label_start, label_end]},
        "base_rules_count": len(base_rules),
        "addon_rules_count": len(addon_rules),
        "base_key_fields": list(base_key_fields),
        "addon_key_fields": list(addon_key_fields),
        "target_counts": target_counts,
        "policy_book": policy_book,
        "addon_rules_preview": list(addon_rules.values())[:20],
    }
    if summary_output:
        Path(summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(summary_output).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export hybrid stable policy trader SFT rows")
    for arg in ["output", "summary-output", "records-cache-dir", "base-report", "addon-report", "train-start", "train-end", "label-start", "label-end"]:
        parser.add_argument("--" + arg, required=True)
    parser.add_argument("--stride-bars", type=int, default=12)
    parser.add_argument("--hold-candidates", default="48,96,144,288")
    parser.add_argument("--base-top-index", type=int, default=0)
    parser.add_argument("--include-base-rule-indices", default="")
    parser.add_argument("--addon-top-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run_export(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
