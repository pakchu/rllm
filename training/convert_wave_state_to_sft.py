"""Convert wave state ranker rows to train_text_sft/eval_text_json_key format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def convert(input_jsonl: str, output_jsonl: str) -> dict[str, Any]:
    rows = _read(input_jsonl)
    out = []
    counts: dict[str, int] = {}
    for row in rows:
        target = dict(row.get("target") or {})
        decision = str(target.get("decision", "ABSTAIN"))
        compact_target = {"decision": decision}
        counts[decision] = counts.get(decision, 0) + 1
        out.append({
            "prompt": str(row["prompt"]),
            "target": json.dumps(compact_target, ensure_ascii=False, sort_keys=True),
            "source": {"date": row.get("date"), "signal_pos": row.get("signal_pos"), "side": row.get("side"), "reward": row.get("reward"), "state_tokens": row.get("state_tokens")},
            "leakage_guard": row.get("leakage_guard", {}),
        })
    Path(output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(output_jsonl).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in out) + ("\n" if out else ""))
    return {"input": input_jsonl, "output": output_jsonl, "rows": len(out), "target_counts": counts}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert wave state rows to compact decision SFT rows")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--output-jsonl", required=True)
    return p.parse_args()


def main() -> None:
    print(json.dumps(convert(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
