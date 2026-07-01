"""Build DPO preference pairs for linear-alpha meta-controller rows.

The binary SFT setup asks Gemma to imitate one TAKE/SKIP label.  This builder
turns the same no-leak prompt surface into chosen/rejected completions so the
model is optimized to rank the desired veto decision above the opposite decision.
Future outcomes remain training-only labels; prompts are copied from signal-time
SFT rows.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LinearAlphaMetaPreferenceConfig:
    input_jsonl: str
    output_jsonl: str
    summary_output: str = ""
    trade_only: bool = True
    target_schema: str = "decision"


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + ("\n" if rows else ""))


def _target(row: dict[str, Any]) -> dict[str, str]:
    raw = row.get("target", {})
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        parsed = {}
    decision = str(parsed.get("decision", row.get("metadata", {}).get("target_decision", "SKIP"))).upper()
    decision = "TAKE" if decision == "TAKE" else "SKIP"
    size = str(parsed.get("size_bucket", row.get("metadata", {}).get("target_size_bucket", "SMALL" if decision == "TAKE" else "NONE"))).upper()
    if decision == "SKIP":
        size = "NONE"
    elif size not in {"SMALL", "FULL"}:
        size = "SMALL"
    return {"decision": decision, "size_bucket": size}


def _completion(decision: str, size_bucket: str, schema: str) -> str:
    decision = "TAKE" if str(decision).upper() == "TAKE" else "SKIP"
    size_bucket = str(size_bucket).upper()
    if decision == "SKIP":
        size_bucket = "NONE"
    elif size_bucket not in {"SMALL", "FULL"}:
        size_bucket = "SMALL"
    if schema == "decision":
        obj = {"decision": decision}
    elif schema == "decision_size":
        obj = {"decision": decision, "size_bucket": size_bucket}
    else:
        raise ValueError("target_schema must be decision|decision_size")
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _opposite(target: dict[str, str]) -> dict[str, str]:
    if target["decision"] == "TAKE":
        return {"decision": "SKIP", "size_bucket": "NONE"}
    return {"decision": "TAKE", "size_bucket": "SMALL"}


def build(cfg: LinearAlphaMetaPreferenceConfig) -> dict[str, Any]:
    rows = _read_jsonl(cfg.input_jsonl)
    out: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    chosen_counts: Counter[str] = Counter()
    rejected_counts: Counter[str] = Counter()
    for row in rows:
        meta = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
        if cfg.trade_only and str(meta.get("candidate_gate", "")).upper() != "TRADE":
            skipped["non_trade_candidate"] += 1
            continue
        prompt = str(row.get("prompt", ""))
        if not prompt:
            skipped["missing_prompt"] += 1
            continue
        chosen = _target(row)
        rejected = _opposite(chosen)
        chosen_text = _completion(chosen["decision"], chosen["size_bucket"], cfg.target_schema)
        rejected_text = _completion(rejected["decision"], rejected["size_bucket"], cfg.target_schema)
        if chosen_text == rejected_text:
            skipped["same_completion"] += 1
            continue
        out.append(
            {
                "task": "linear_alpha_meta_controller_preference",
                "prompt": prompt,
                "chosen": chosen_text,
                "rejected": rejected_text,
                "date": meta.get("date"),
                "signal_pos": meta.get("signal_pos"),
                "candidate_side": meta.get("candidate_side"),
                "target_decision": chosen["decision"],
                "target_size_bucket": chosen["size_bucket"],
                "metadata": {
                    "source_task": row.get("task"),
                    "candidate_gate": meta.get("candidate_gate"),
                    "candidate_side": meta.get("candidate_side"),
                    "realized_trade_ret_pct": meta.get("realized_trade_ret_pct"),
                    "trade_path_stats": meta.get("trade_path_stats"),
                    "leakage_guard": "prompt copied from signal-time SFT row; chosen/rejected use future label for offline DPO only",
                },
            }
        )
        chosen_counts[chosen_text] += 1
        rejected_counts[rejected_text] += 1
    _write_jsonl(cfg.output_jsonl, out)
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "rows_in": len(rows),
        "pairs_out": len(out),
        "chosen_counts": dict(sorted(chosen_counts.items())),
        "rejected_counts": dict(sorted(rejected_counts.items())),
        "skipped_counts": dict(sorted(skipped.items())),
        "prompt_chars": {
            "min": min((len(r["prompt"]) for r in out), default=0),
            "max": max((len(r["prompt"]) for r in out), default=0),
            "mean": sum(len(r["prompt"]) for r in out) / max(1, len(out)),
        },
        "leakage_guard": {
            "prompt_uses_signal_time_features_only": True,
            "future_path_label_used_only_for_chosen_rejected": True,
            "trade_only_default_avoids_trivial_no_signal_pairs": bool(cfg.trade_only),
        },
    }
    if cfg.summary_output:
        Path(cfg.summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.summary_output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build linear-alpha meta-controller DPO preference pairs")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--summary-output", default="")
    p.add_argument("--target-schema", choices=["decision", "decision_size"], default=LinearAlphaMetaPreferenceConfig.target_schema)
    p.add_argument("--include-no-trade", action="store_true", help="Include trivial frozen-alpha NO_TRADE rows")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = LinearAlphaMetaPreferenceConfig(
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        summary_output=args.summary_output,
        trade_only=not bool(args.include_no_trade),
        target_schema=args.target_schema,
    )
    print(json.dumps(build(cfg), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
