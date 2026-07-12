"""Build margin-filtered preference pairs for REX event choice-label DPO.

The earlier SFT targets forced a single hard class even when utilities were close.
This builder converts each REX event into pairwise preferences only when the
chosen action's offline path utility exceeds the rejected action by a minimum
margin.  Prompts remain signal-time/past-only; future path utilities are used
only to create offline training preferences.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

LABELS = {
    "LONG": "CHOICE_A_LONG",
    "SHORT": "CHOICE_B_SHORT",
    "NO_TRADE": "CHOICE_C_SKIP",
}
ACTIONS = tuple(LABELS)


@dataclass(frozen=True)
class Cfg:
    input_jsonl: str
    output_jsonl: str
    summary_output: str = ""
    min_margin: float = 0.004
    no_trade_utility: float = 0.001
    min_trade_net_return: float = 0.001
    max_trade_mae: float = 0.035
    include_side_pairs: bool = True
    include_gate_pairs: bool = True
    train_only_output_jsonl: str = ""


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + ("\n" if rows else ""))


def convert_prompt(prompt: str) -> str:
    return "\n".join([
        str(prompt),
        "",
        "Preference output contract:",
        "- CHOICE_A_LONG means take LONG.",
        "- CHOICE_B_SHORT means take SHORT.",
        "- CHOICE_C_SKIP means no trade.",
        "Return exactly one label and nothing else.",
    ])


def action_utilities(row: dict[str, Any], cfg: Cfg) -> dict[str, float]:
    audit = row.get("target_action_audit") or {}
    long = audit.get("long") or {}
    short = audit.get("short") or {}
    out = {
        "LONG": float(long.get("utility", -1e9)),
        "SHORT": float(short.get("utility", -1e9)),
        "NO_TRADE": float(cfg.no_trade_utility),
    }
    # Make NO_TRADE preferable to structurally invalid trades, matching the SFT
    # oracle constraints.  This prevents DPO from learning to trade when the
    # path net/MAE violates executable risk rules even if raw utility is close.
    for action, side_obj in (("LONG", long), ("SHORT", short)):
        net = float(side_obj.get("net_return", -1e9))
        mae = float(side_obj.get("mae", 1e9))
        if net <= float(cfg.min_trade_net_return) or mae > float(cfg.max_trade_mae):
            out[action] = min(out[action], float(cfg.no_trade_utility) - abs(float(cfg.min_margin)))
    return out


def pair_kind(chosen: str, rejected: str) -> str:
    if "NO_TRADE" in {chosen, rejected}:
        return "gate"
    return "side"


def build(cfg: Cfg) -> dict[str, Any]:
    src = read_jsonl(cfg.input_jsonl)
    rows: list[dict[str, Any]] = []
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    margins: list[float] = []
    for row in src:
        split = str(row.get("split", "unknown"))
        utils = action_utilities(row, cfg)
        prompt = convert_prompt(str(row.get("prompt", "")))
        for chosen in ACTIONS:
            for rejected in ACTIONS:
                if chosen == rejected:
                    continue
                kind = pair_kind(chosen, rejected)
                if kind == "gate" and not cfg.include_gate_pairs:
                    continue
                if kind == "side" and not cfg.include_side_pairs:
                    continue
                margin = float(utils[chosen] - utils[rejected])
                if margin < float(cfg.min_margin):
                    continue
                out = {
                    "task": "rex_event_choice_preference_dpo",
                    "prompt": prompt,
                    "chosen": LABELS[chosen],
                    "rejected": LABELS[rejected],
                    "chosen_action": chosen,
                    "rejected_action": rejected,
                    "preference_kind": kind,
                    "utility_margin": margin,
                    "utilities": utils,
                    "date": row.get("date"),
                    "signal_pos": row.get("signal_pos"),
                    "split": split,
                    "base_event": row.get("base_event"),
                    "state_tokens": row.get("state_tokens"),
                    "target_action_audit": row.get("target_action_audit"),
                    "leakage_guard": dict(row.get("leakage_guard", {})) | {
                        "prompt_uses_future_path": False,
                        "preference_uses_future_path_for_offline_training_only": True,
                        "dpo_margin_filter_reduces_noisy_ties": True,
                    },
                }
                rows.append(out)
                key = f"{chosen}>{rejected}"
                counters["all"][key] += 1
                counters[split][key] += 1
                counters[f"{split}:{kind}"][key] += 1
                margins.append(margin)
    write_jsonl(cfg.output_jsonl, rows)
    if cfg.train_only_output_jsonl:
        write_jsonl(cfg.train_only_output_jsonl, [r for r in rows if r.get("split") == "train"])
    summary = {
        "config": asdict(cfg),
        "source_rows": len(src),
        "preference_rows": len(rows),
        "train_rows": sum(1 for r in rows if r.get("split") == "train"),
        "counts": {k: dict(v) for k, v in sorted(counters.items())},
        "margin": {
            "min": min(margins) if margins else 0.0,
            "max": max(margins) if margins else 0.0,
            "mean": sum(margins) / max(1, len(margins)),
        },
        "label_contract": LABELS,
        "leakage_guard": {
            "prompt_contains_only_signal_time_symbolic_context": True,
            "future_path_used_only_to_rank_training_preferences": True,
            "test_eval_rows_are_written_for_audit_only_not_training": True,
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
    p.add_argument("--min-margin", type=float, default=0.004)
    p.add_argument("--no-trade-utility", type=float, default=0.001)
    p.add_argument("--min-trade-net-return", type=float, default=0.001)
    p.add_argument("--max-trade-mae", type=float, default=0.035)
    p.add_argument("--include-side-pairs", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-gate-pairs", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--train-only-output-jsonl", default="")
    print(json.dumps(build(Cfg(**vars(p.parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
