"""Apply month-level abstention gates from first in-month candidate tokens.

The gate is live-usable when candidate rows are built from past/current-bar text:
for each calendar month, take the first available candidate prompt tokens and
skip all trades in that month if all required tokens are present.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in [x for x in str(path).split(",") if x.strip()]:
        rows.extend(json.loads(line) for line in Path(p).read_text().splitlines() if line.strip())
    return rows


def _month(date: Any) -> str:
    return str(date)[:7]


def _prompt_tokens(prompt: str) -> set[str]:
    out: set[str] = set()
    for line in str(prompt).splitlines():
        if line.startswith("Regime tokens:"):
            out.update(p.strip() for p in line.split(":", 1)[1].split(";") if p.strip())
        elif line.startswith("Candidate book tokens:"):
            for part in [p.strip() for p in line.split(":", 1)[1].split(";") if p.strip()]:
                out.add("book_" + part.split(":", 1)[0])
    return out


def month_tokens(candidate_jsonl: str) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for row in sorted(_read(candidate_jsonl), key=lambda r: (str(r.get("date")), int(r.get("signal_pos", -1) or -1))):
        m = _month(row.get("date"))
        if m not in out:
            out[m] = _prompt_tokens(str(row.get("prompt", "")))
    return out


def run(*, candidate_jsonl: str, predictions_jsonl: str, output: str, require_tokens: str) -> dict[str, Any]:
    required = {x.strip() for x in str(require_tokens).split(",") if x.strip()}
    if not required:
        raise ValueError("at least one required token is needed")
    tokens_by_month = month_tokens(candidate_jsonl)
    out_rows: list[dict[str, Any]] = []
    blocked_rows = 0
    blocked_months: list[str] = []
    for row in _read(predictions_jsonl):
        m = _month(row.get("date"))
        is_blocked_month = required.issubset(tokens_by_month.get(m, set()))
        pred = row.get("prediction", {}) if isinstance(row.get("prediction"), dict) else {}
        if is_blocked_month and pred.get("gate") == "TRADE":
            row = {
                **row,
                "prediction": {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0, "family": "MONTH_REGIME_ABSTAIN", "confidence": "HIGH"},
                "blocked_prediction": pred,
            }
            blocked_rows += 1
        if is_blocked_month and m not in blocked_months:
            blocked_months.append(m)
        out_rows.append(row)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in out_rows) + "\n")
    return {
        "candidate_jsonl": candidate_jsonl,
        "predictions_jsonl": predictions_jsonl,
        "output": output,
        "required_tokens": sorted(required),
        "rows": len(out_rows),
        "blocked_trade_rows": blocked_rows,
        "blocked_months": blocked_months,
        "leakage_guard": {"month_gate_uses_first_candidate_prompt_tokens_only": True},
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Apply month-regime abstention gate")
    p.add_argument("--candidate-jsonl", required=True, help="Comma-separated candidate jsonl files")
    p.add_argument("--predictions-jsonl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--require-tokens", required=True, help="Comma-separated tokens; all must be present")
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
