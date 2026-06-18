"""Build utility-threshold gate labels from event-action policy rows.

This changes the gate target from "best candidate is a trade" to "best trade is
worth taking after utility and path-risk filters".  Prompts remain past-only;
the future path audit is used only to create supervised labels.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GateUtilityConfig:
    input_jsonl: str
    output_jsonl: str
    summary_output: str = ""
    min_rank_utility: float = 0.01
    min_net_return: float = 0.004
    max_mae: float = 0.02
    min_mfe_to_mae: float = 0.0


def _read(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _prompt_core(prompt: str) -> str:
    keep = []
    for line in str(prompt).splitlines():
        if line.startswith("Date: ") or line.startswith("Past-only state: ") or line.startswith("Candidate action book: "):
            keep.append(line)
    return "\n".join(keep) if keep else str(prompt)


def _trade_label(row: dict[str, Any], cfg: GateUtilityConfig) -> str:
    audit = row.get("target_action_audit", {})
    if not isinstance(audit, dict) or str(audit.get("gate", "NO_TRADE")).upper() != "TRADE":
        return "NO_TRADE"
    rank_utility = float(audit.get("rank_utility", -1e9) or -1e9)
    net_return = float(audit.get("net_return", -1e9) or -1e9)
    mae = float(audit.get("mae", 1e9) or 1e9)
    mfe = float(audit.get("mfe", 0.0) or 0.0)
    mfe_to_mae = float(audit.get("mfe_to_mae", mfe / max(mae, 1e-9)) or 0.0)
    if rank_utility >= cfg.min_rank_utility and net_return >= cfg.min_net_return and mae <= cfg.max_mae and mfe_to_mae >= cfg.min_mfe_to_mae:
        return "TRADE"
    return "NO_TRADE"


def build_rows(rows: list[dict[str, Any]], cfg: GateUtilityConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out = []
    counts: Counter[str] = Counter()
    for row in rows:
        label = _trade_label(row, cfg)
        counts[label] += 1
        out.append(
            {
                "task": "event_action_utility_gate",
                "date": row.get("date"),
                "signal_pos": row.get("signal_pos"),
                "prompt": "\n".join(
                    [
                        "You are the first-stage gate for a BTCUSDT futures bot.",
                        "Use only the past-only state and candidate action book.",
                        "Output exactly one label: TRADE or NO_TRADE.",
                        "TRADE only when expected edge is strong enough after path-risk; otherwise NO_TRADE.",
                        "Do not output JSON or extra words.",
                        "",
                        _prompt_core(str(row.get("prompt", ""))),
                    ]
                ),
                "target": label,
                "target_action_audit": row.get("target_action_audit"),
                "leakage_guard": {
                    "prompt_uses_future_path": False,
                    "target_uses_future_audit_for_training_only": True,
                    "utility_threshold_label": True,
                },
            }
        )
    summary = {"config": asdict(cfg), "rows": len(out), "target_counts": dict(sorted(counts.items()))}
    return out, summary


def run(cfg: GateUtilityConfig) -> dict[str, Any]:
    out, summary = build_rows(_read(cfg.input_jsonl), cfg)
    Path(cfg.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output_jsonl).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in out) + "\n")
    if cfg.summary_output:
        Path(cfg.summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.summary_output).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build utility-threshold event-action gate labels")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--summary-output", default="")
    p.add_argument("--min-rank-utility", type=float, default=0.01)
    p.add_argument("--min-net-return", type=float, default=0.004)
    p.add_argument("--max-mae", type=float, default=0.02)
    p.add_argument("--min-mfe-to-mae", type=float, default=0.0)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(GateUtilityConfig(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
