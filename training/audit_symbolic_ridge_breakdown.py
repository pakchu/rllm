"""Audit selected symbolic-ridge trades by action/regime breakdown.

The sweep can report a strong validation segment and a weak chronological holdout.
This utility joins selected prediction rows with the online backtest executions and
prints small, leakage-safe diagnostics by side, hold, family, and month so failure
modes can be inspected without rerunning model training.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _trade_stats(rows: list[dict[str, Any]]) -> dict[str, float | int | list[float]]:
    if not rows:
        return {"n": 0, "mean_trade_ret_pct": 0.0, "sum_trade_ret_pct": 0.0, "win_rate": 0.0, "min_trade_ret_pct": 0.0, "max_trade_ret_pct": 0.0}
    vals = [float(r["trade_ret_pct"]) for r in rows]
    return {
        "n": len(vals),
        "mean_trade_ret_pct": mean(vals),
        "sum_trade_ret_pct": sum(vals),
        "win_rate": sum(v > 0 for v in vals) / len(vals),
        "min_trade_ret_pct": min(vals),
        "max_trade_ret_pct": max(vals),
    }


def join_executions(backtest_path: str | Path, predictions_path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    backtest = json.loads(Path(backtest_path).read_text())
    predictions = {(str(r["date"]), int(r["signal_pos"])): r for r in load_jsonl(predictions_path)}
    rows: list[dict[str, Any]] = []
    missing = 0
    for execution in backtest.get("executed", []):
        key = (str(execution["date"]), int(execution["signal_pos"]))
        prediction = predictions.get(key)
        if prediction is None:
            missing += 1
            continue
        action = prediction.get("selected_action") or prediction.get("prediction") or {}
        rows.append({
            **execution,
            "month": str(execution["date"])[:7],
            "family": action.get("family"),
            "predicted_utility": prediction.get("predicted_utility"),
            "runner_up_gap": prediction.get("runner_up_gap"),
            "actual_utility": prediction.get("actual_utility"),
        })
    return backtest, rows, missing


def grouped(rows: list[dict[str, Any]], keys: Iterable[str]) -> list[dict[str, Any]]:
    key_list = list(keys)
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row.get(key) for key in key_list)].append(row)
    out = []
    for key, bucket_rows in buckets.items():
        out.append({"key": dict(zip(key_list, key)), **_trade_stats(bucket_rows)})
    return sorted(out, key=lambda r: (-abs(float(r["sum_trade_ret_pct"])), -int(r["n"])))


def build_audit(label: str, backtest_path: str | Path, predictions_path: str | Path, group_keys: list[list[str]], top_n: int) -> dict[str, Any]:
    backtest, rows, missing = join_executions(backtest_path, predictions_path)
    return {
        "label": label,
        "backtest_path": str(backtest_path),
        "predictions_path": str(predictions_path),
        "sim": backtest.get("sim", {}),
        "trade_stats": backtest.get("trade_stats", {}),
        "joined_executions": len(rows),
        "missing_predictions": missing,
        "all_executions": _trade_stats(rows),
        "groups": {"|".join(keys): grouped(rows, keys)[:top_n] for keys in group_keys},
    }


def _format_pct(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def render_markdown(audits: list[dict[str, Any]]) -> str:
    lines = ["# Symbolic Ridge Execution Breakdown Audit", ""]
    for audit in audits:
        sim = audit["sim"]
        lines.extend([
            f"## {audit['label']}",
            "",
            f"- CAGR: {_format_pct(sim.get('cagr_pct'))}%",
            f"- strict MDD: {_format_pct(sim.get('strict_mdd_pct'))}%",
            f"- CAGR/strict MDD: {_format_pct(sim.get('cagr_to_strict_mdd'))}",
            f"- executed trades: {sim.get('trade_entries')} / joined: {audit['joined_executions']} / missing predictions: {audit['missing_predictions']}",
            f"- mean trade return: {_format_pct(audit['all_executions']['mean_trade_ret_pct'])}%",
            "",
        ])
        for group_name, rows in audit["groups"].items():
            lines.extend([f"### by {group_name}", "", "| key | n | mean% | sum% | win% | min% | max% |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
            for row in rows:
                key = ", ".join(f"{k}={v}" for k, v in row["key"].items())
                lines.append(
                    f"| {key} | {row['n']} | {_format_pct(row['mean_trade_ret_pct'])} | {_format_pct(row['sum_trade_ret_pct'])} | {100*float(row['win_rate']):.1f} | {_format_pct(row['min_trade_ret_pct'])} | {_format_pct(row['max_trade_ret_pct'])} |"
                )
            lines.append("")
    return "\n".join(lines)


def parse_group(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", nargs=3, metavar=("LABEL", "BACKTEST_JSON", "PREDICTIONS_JSONL"), required=True, help="Audit case. Can be supplied multiple times.")
    parser.add_argument("--group", action="append", default=[], help="Comma-separated group keys. Defaults to side/hold/family/month combinations.")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()

    default_groups = [
        ["side"], ["hold_bars"], ["family"], ["month"],
        ["side", "family"], ["side", "hold_bars"], ["family", "hold_bars"],
        ["month", "side"], ["month", "family"],
    ]
    group_keys = [parse_group(v) for v in args.group] if args.group else default_groups
    audits = [build_audit(label, bt, preds, group_keys, args.top_n) for label, bt, preds in args.case]
    payload = {"audits": audits}

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(render_markdown(audits) + "\n")
    if not args.output_json and not args.output_md:
        print(render_markdown(audits))


if __name__ == "__main__":
    main()
