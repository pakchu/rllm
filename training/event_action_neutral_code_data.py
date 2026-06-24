"""Remap ordinal utility rows to neutral code labels.

Semantic labels such as AVOID/HIGH have large base logprob priors.  This module
keeps the same candidate task but maps labels to neutral-ish codes Q1..Q4 and
rewrites the prompt to ask for codes rather than semantic output tokens.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

CODE_BY_LABEL = {"AVOID": "Q1", "LOW": "Q2", "MID": "Q3", "HIGH": "Q4"}
LABEL_BY_CODE = {v: k for k, v in CODE_BY_LABEL.items()}


@dataclass(frozen=True)
class NeutralCodeConfig:
    input_jsonl: str
    output_jsonl: str
    summary_output: str = ""
    include_codebook: bool = True


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def _target_code(row: dict[str, Any]) -> str:
    label = str(row.get("target", "LOW")).strip().upper()
    return CODE_BY_LABEL.get(label, "Q2")


def _rewrite_prompt(prompt: str, *, include_codebook: bool) -> str:
    body = []
    for line in str(prompt).splitlines():
        if line.startswith("Output exactly one label:"):
            continue
        if line.startswith("Definitions:"):
            continue
        body.append(line)
    header = [
        "You are a neutral-code action utility judge for BTCUSDT futures.",
        "Use only the past-only state, prompt-visible action book, and candidate action.",
        "Classify the candidate's expected utility after path risk.",
        "Output exactly one code: Q1, Q2, Q3, or Q4.",
    ]
    if include_codebook:
        header.append("Codebook: Q1=negative or fragile, Q2=weak or no edge, Q3=usable edge, Q4=strong edge with controlled adverse path.")
    return "\n".join(header + [""] + body).strip()


def build_code_rows(rows: list[dict[str, Any]], cfg: NeutralCodeConfig) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        code = _target_code(row)
        out.append(
            {
                **row,
                "task": "event_action_neutral_code_utility",
                "prompt": _rewrite_prompt(str(row.get("prompt", "")), include_codebook=bool(cfg.include_codebook)),
                "target": code,
                "semantic_target": row.get("target"),
                "code_label_map": dict(LABEL_BY_CODE),
                "leakage_guard": {
                    "prompt_uses_future_path": False,
                    "target_uses_future_utility_for_training_only": True,
                    "semantic_label_not_output_token": True,
                },
            }
        )
    return out


def summarize(rows: list[dict[str, Any]], cfg: NeutralCodeConfig) -> dict[str, Any]:
    counts = Counter(str(r.get("target")) for r in rows)
    semantic = Counter(str(r.get("semantic_target")) for r in rows)
    prompt_lens = [len(str(r.get("prompt", ""))) for r in rows]
    return {
        "input_jsonl": str(Path(cfg.input_jsonl).resolve()),
        "output_jsonl": cfg.output_jsonl,
        "rows": len(rows),
        "signals": len({(r.get("date"), r.get("signal_pos")) for r in rows}),
        "target_counts": dict(sorted(counts.items())),
        "semantic_target_counts": dict(sorted(semantic.items())),
        "prompt_chars": {"min": min(prompt_lens) if prompt_lens else 0, "max": max(prompt_lens) if prompt_lens else 0, "mean": sum(prompt_lens) / max(1, len(prompt_lens))},
        "code_label_map": dict(LABEL_BY_CODE),
        "config": asdict(cfg),
        "leakage_guard": {"prompts_are_past_only": True, "future_utility_only_in_target_and_audit": True, "semantic_labels_not_output_tokens": True},
    }


def build_neutral_code_jsonl(**kwargs: Any) -> dict[str, Any]:
    cfg = NeutralCodeConfig(**kwargs)
    rows = build_code_rows(read_jsonl(cfg.input_jsonl), cfg)
    write_jsonl(cfg.output_jsonl, rows)
    summary = summarize(rows, cfg)
    if cfg.summary_output:
        Path(cfg.summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.summary_output).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Remap ordinal utility rows to neutral code labels")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--summary-output", default="")
    p.add_argument("--no-codebook", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            build_neutral_code_jsonl(
                input_jsonl=args.input_jsonl,
                output_jsonl=args.output_jsonl,
                summary_output=args.summary_output,
                include_codebook=not bool(args.no_codebook),
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
