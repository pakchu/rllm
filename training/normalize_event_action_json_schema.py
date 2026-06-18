"""Rewrite event-action SFT/preference rows to a gate-first JSON schema.

The original rows used ``json.dumps(..., sort_keys=True)``, which puts
``confidence``/``family`` before ``gate``.  Candidate-logprob inference then
scores many non-gate tokens before the actual abstention decision, encouraging
schema/side priors instead of a clear TRADE vs NO_TRADE boundary.

This transformer keeps prompts past-only and rewrites only supervised targets or
preference responses into an insertion-ordered compact JSON schema:
``gate, side, hold_bars, family, confidence``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GateFirstRewriteConfig:
    input_jsonl: str
    output_jsonl: str
    mode: str = "auto"


def _parse_action(text: str) -> dict[str, Any]:
    obj = json.loads(str(text))
    if not isinstance(obj, dict):
        raise ValueError("action is not a JSON object")
    gate = str(obj.get("gate", "NO_TRADE")).upper()
    if gate != "TRADE":
        return {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0, "family": "NONE", "confidence": str(obj.get("confidence", "HIGH")).upper()}
    return {
        "gate": "TRADE",
        "side": str(obj.get("side", "NONE")).upper(),
        "hold_bars": int(obj.get("hold_bars", 0) or 0),
        "family": str(obj.get("family", "UNKNOWN")),
        "confidence": str(obj.get("confidence", "HIGH")).upper(),
    }


def _dump_gate_first(action_text: str) -> str:
    return json.dumps(_parse_action(action_text), ensure_ascii=False, separators=(",", ":"))


def _detect_mode(row: dict[str, Any], requested: str) -> str:
    mode = str(requested).strip().lower()
    if mode != "auto":
        return mode
    if "target" in row:
        return "sft"
    if "chosen" in row and "rejected" in row:
        return "preference"
    raise ValueError("could not auto-detect row mode")


def rewrite_rows(rows: list[dict[str, Any]], cfg: GateFirstRewriteConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    for row in rows:
        mode = _detect_mode(row, cfg.mode)
        modes[mode] += 1
        new = dict(row)
        if mode == "sft":
            new["target"] = _dump_gate_first(str(row["target"]))
            action = _parse_action(new["target"])
            counts[f"target_gate={action['gate']}"] += 1
        elif mode == "preference":
            new["chosen"] = _dump_gate_first(str(row["chosen"]))
            new["rejected"] = _dump_gate_first(str(row["rejected"]))
            chosen = _parse_action(new["chosen"])
            rejected = _parse_action(new["rejected"])
            counts[f"chosen_gate={chosen['gate']}"] += 1
            counts[f"rejected_gate={rejected['gate']}"] += 1
        else:
            raise ValueError("mode must be one of {'auto','sft','preference'}")
        guard = dict(new.get("leakage_guard", {})) if isinstance(new.get("leakage_guard"), dict) else {}
        guard["rewrites_only_label_json_key_order"] = True
        guard["prompt_unchanged"] = True
        new["leakage_guard"] = guard
        out.append(new)
    summary = {"config": asdict(cfg), "rows": len(out), "modes": dict(modes), "counts": dict(sorted(counts.items()))}
    return out, summary


def run_rewrite(cfg: GateFirstRewriteConfig) -> dict[str, Any]:
    rows = [json.loads(line) for line in Path(cfg.input_jsonl).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"no rows loaded from {cfg.input_jsonl}")
    out, summary = rewrite_rows(rows, cfg)
    path = Path(cfg.output_jsonl)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in out) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rewrite event-action rows to gate-first target/chosen/rejected JSON")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--mode", choices=["auto", "sft", "preference"], default="auto")
    return p.parse_args()


def main() -> None:
    print(json.dumps(run_rewrite(GateFirstRewriteConfig(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
