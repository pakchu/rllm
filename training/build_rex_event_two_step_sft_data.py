"""Build single-LLM two-step REX event SFT data.

One model is trained with two explicit tasks instead of one 3-class output:
- GATE: decide TRADE vs NO_TRADE.
- SIDE: decide LONG vs SHORT only for trade-worthy examples.

Prompts contain only signal-time symbolic facts.  Future path utility is used
only to create offline labels.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Cfg:
    input_jsonl: str
    output_jsonl: str
    summary_output: str = ""
    train_output_jsonl: str = ""
    gate_min_margin: float = 0.004
    side_min_margin: float = 0.004
    no_trade_utility: float = 0.001
    min_trade_net_return: float = 0.001
    max_trade_mae: float = 0.035
    include_side_for_all_margin_rows: bool = True


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + ("\n" if rows else ""))


def _utils(row: dict[str, Any], cfg: Cfg) -> dict[str, float]:
    audit = row.get("target_action_audit") or {}
    long = audit.get("long") or {}
    short = audit.get("short") or {}
    out = {
        "LONG": float(long.get("utility", -1e9)),
        "SHORT": float(short.get("utility", -1e9)),
        "NO_TRADE": float(cfg.no_trade_utility),
    }
    for action, side_obj in (("LONG", long), ("SHORT", short)):
        net = float(side_obj.get("net_return", -1e9))
        mae = float(side_obj.get("mae", 1e9))
        if net <= float(cfg.min_trade_net_return) or mae > float(cfg.max_trade_mae):
            out[action] = min(out[action], float(cfg.no_trade_utility) - abs(float(cfg.gate_min_margin)))
    return out


def _base_prompt(row: dict[str, Any]) -> str:
    return str(row.get("prompt", ""))


def _gate_prompt(row: dict[str, Any]) -> str:
    return "\n".join([
        _base_prompt(row),
        "",
        "Two-step REX policy task 1/2: GATE.",
        "Decide whether this event has enough executable edge after risk/cost.",
        "Return exactly one token: TRADE or NO_TRADE.",
    ])


def _side_prompt(row: dict[str, Any]) -> str:
    return "\n".join([
        _base_prompt(row),
        "",
        "Two-step REX policy task 2/2: SIDE.",
        "Assume the gate already allowed a trade. Choose the better side.",
        "Return exactly one token: LONG or SHORT.",
    ])


def _row_common(row: dict[str, Any], task: str, target: str, utils: dict[str, float], margin: float) -> dict[str, Any]:
    return {
        "task": task,
        "prompt": _gate_prompt(row) if task.endswith("gate_sft") else _side_prompt(row),
        "target": target,
        "date": row.get("date"),
        "split": row.get("split"),
        "signal_pos": row.get("signal_pos"),
        "base_event": row.get("base_event"),
        "state_tokens": row.get("state_tokens"),
        "utilities": utils,
        "label_margin": margin,
        "target_action_audit": row.get("target_action_audit"),
        "leakage_guard": dict(row.get("leakage_guard", {})) | {
            "prompt_uses_future_path": False,
            "target_uses_future_path_for_offline_training_only": True,
            "two_step_task_separates_gate_and_side": True,
        },
    }


def build(cfg: Cfg) -> dict[str, Any]:
    src = read_jsonl(cfg.input_jsonl)
    rows: list[dict[str, Any]] = []
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in src:
        sp = str(row.get("split", "unknown"))
        u = _utils(row, cfg)
        best_side = "LONG" if u["LONG"] >= u["SHORT"] else "SHORT"
        best_trade_u = max(u["LONG"], u["SHORT"])
        gate_margin = best_trade_u - u["NO_TRADE"]
        if gate_margin >= float(cfg.gate_min_margin):
            gate_label = "TRADE"
            gate_label_margin = gate_margin
        elif -gate_margin >= float(cfg.gate_min_margin):
            gate_label = "NO_TRADE"
            gate_label_margin = -gate_margin
        else:
            gate_label = "NO_TRADE"
            gate_label_margin = abs(gate_margin)
        # Keep all gate rows; low-margin rows teach abstention conservatism.
        gate_row = _row_common(row, "rex_event_two_step_gate_sft", gate_label, u, gate_label_margin)
        rows.append(gate_row)
        counts[f"{sp}:gate"][gate_label] += 1
        counts["all:gate"][gate_label] += 1

        side_margin = abs(u["LONG"] - u["SHORT"])
        side_allowed = side_margin >= float(cfg.side_min_margin)
        if side_allowed and (cfg.include_side_for_all_margin_rows or gate_label == "TRADE"):
            side_row = _row_common(row, "rex_event_two_step_side_sft", best_side, u, side_margin)
            rows.append(side_row)
            counts[f"{sp}:side"][best_side] += 1
            counts["all:side"][best_side] += 1

    write_jsonl(cfg.output_jsonl, rows)
    if cfg.train_output_jsonl:
        write_jsonl(cfg.train_output_jsonl, [r for r in rows if r.get("split") == "train"])
    summary = {
        "config": asdict(cfg),
        "source_rows": len(src),
        "rows": len(rows),
        "train_rows": sum(1 for r in rows if r.get("split") == "train"),
        "counts": {k: dict(v) for k, v in sorted(counts.items())},
        "prompt_chars": {
            "min": min((len(r["prompt"]) for r in rows), default=0),
            "max": max((len(r["prompt"]) for r in rows), default=0),
            "mean": sum(len(r["prompt"]) for r in rows) / max(1, len(rows)),
        },
        "leakage_guard": {
            "prompt_contains_only_signal_time_symbolic_context": True,
            "future_path_used_only_to_create_offline_targets": True,
            "train_output_contains_train_split_only": bool(cfg.train_output_jsonl),
        },
    }
    if cfg.summary_output:
        Path(cfg.summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.summary_output).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--summary-output", default="")
    p.add_argument("--train-output-jsonl", default="")
    p.add_argument("--gate-min-margin", type=float, default=0.004)
    p.add_argument("--side-min-margin", type=float, default=0.004)
    p.add_argument("--no-trade-utility", type=float, default=0.001)
    p.add_argument("--min-trade-net-return", type=float, default=0.001)
    p.add_argument("--max-trade-mae", type=float, default=0.035)
    p.add_argument("--include-side-for-all-margin-rows", action=argparse.BooleanOptionalAction, default=True)
    print(json.dumps(build(Cfg(**vars(p.parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
