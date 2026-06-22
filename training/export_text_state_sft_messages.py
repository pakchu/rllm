"""Export text-state decision rows to chat-style SFT JSONL."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExportCfg:
    input_jsonl: str
    train_output: str
    eval_output: str
    summary_output: str
    task: str = "text_state_portfolio_decision"
    system_prompt: str = "You are a disciplined BTCUSDT futures portfolio policy. Follow the requested output format exactly."


def _load(path: str) -> list[dict[str, Any]]:
    rows=[]
    with open(path) as f:
        for line in f:
            line=line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _message_row(row: dict[str, Any], cfg: ExportCfg) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": cfg.system_prompt},
            {"role": "user", "content": str(row["prompt"])},
            {"role": "assistant", "content": str(row["target"])},
        ],
        "metadata": {
            "task": row.get("task"),
            "split": row.get("split"),
            "date": row.get("date"),
            "signal_pos": row.get("signal_pos"),
            "target": row.get("target"),
            "candidate": row.get("candidate", {}),
            "leakage_guard": row.get("leakage_guard", {}),
        },
    }


def _write(path: str, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + ("\n" if rows else ""))


def _summ(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts=Counter(r["metadata"]["target"] for r in rows)
    lens=[sum(len(m["content"]) for m in r["messages"]) for r in rows]
    return {"rows":len(rows),"target_counts":dict(sorted(counts.items())),"chars":{"min":min(lens) if lens else 0,"max":max(lens) if lens else 0,"mean":sum(lens)/max(1,len(lens))}}


def run(cfg: ExportCfg) -> dict[str, Any]:
    rows=[r for r in _load(cfg.input_jsonl) if str(r.get("task")) == cfg.task]
    train=[_message_row(r,cfg) for r in rows if r.get("split") == "train"]
    eval_rows=[_message_row(r,cfg) for r in rows if r.get("split") == "eval"]
    _write(cfg.train_output, train)
    _write(cfg.eval_output, eval_rows)
    report={"config":cfg.__dict__,"train":_summ(train),"eval":_summ(eval_rows),"outputs":{"train":cfg.train_output,"eval":cfg.eval_output},"contract":"assistant content is exactly one label"}
    Path(cfg.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.summary_output).write_text(json.dumps(report,indent=2,ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description="Export text-state rows to messages SFT format")
    p.add_argument("--input-jsonl",required=True)
    p.add_argument("--train-output",required=True)
    p.add_argument("--eval-output",required=True)
    p.add_argument("--summary-output",required=True)
    p.add_argument("--task",default="text_state_portfolio_decision")
    p.add_argument("--system-prompt",default=ExportCfg.system_prompt)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(ExportCfg(**vars(parse_args()))),indent=2,ensure_ascii=False))


if __name__ == "__main__":
    main()
