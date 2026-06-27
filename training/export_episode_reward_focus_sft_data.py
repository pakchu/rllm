"""Export focused reward-component SFT rows with only utility/path-shape targets."""
from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RewardFocusCfg:
    train_jsonl: str
    test_jsonl: str
    eval_jsonl: str
    output_dir: str
    gzip_output: bool = True


def _open(path: str, mode: str = "rt"):
    return gzip.open(path, mode, encoding="utf-8") if str(path).endswith(".gz") else open(path, mode, encoding="utf-8")


def _load(path: str) -> list[dict[str, Any]]:
    with _open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _write(path: Path, rows: list[dict[str, Any]], gzip_output: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if gzip_output else open
    with opener(path, "wt", encoding="utf-8") as f:  # type: ignore[arg-type]
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def _convert(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out=[]
    for row in rows:
        src=json.loads(str(row["target"]))
        target={"path_shape": str(src.get("path_shape", "UNKNOWN")), "utility_bucket": str(src.get("utility_bucket", "UNKNOWN"))}
        r=dict(row)
        r["task"]="episode_reward_focus_sft"
        r["target"]=json.dumps(target, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        guard=dict(r.get("leakage_guard") or {})
        guard["focused_from_reward_components"] = True
        r["leakage_guard"] = guard
        out.append(r)
    return out


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    utility=Counter(); shape=Counter(); lens=[]; target_lens=[]
    for r in rows:
        t=json.loads(str(r["target"]))
        utility[str(t.get("utility_bucket"))]+=1
        shape[str(t.get("path_shape"))]+=1
        lens.append(len(str(r.get("prompt", ""))))
        target_lens.append(len(str(r.get("target", ""))))
    return {
        "rows": len(rows),
        "utility_bucket": dict(utility),
        "path_shape": dict(shape),
        "prompt_chars": {"min": int(min(lens)) if lens else 0, "mean": float(np.mean(lens)) if lens else 0.0, "max": int(max(lens)) if lens else 0},
        "target_chars": {"min": int(min(target_lens)) if target_lens else 0, "mean": float(np.mean(target_lens)) if target_lens else 0.0, "max": int(max(target_lens)) if target_lens else 0},
    }


def run(cfg: RewardFocusCfg) -> dict[str, Any]:
    loaded={"train": _load(cfg.train_jsonl), "test": _load(cfg.test_jsonl), "eval": _load(cfg.eval_jsonl)}
    out_dir=Path(cfg.output_dir)
    suffix=".jsonl.gz" if cfg.gzip_output else ".jsonl"
    report={"config": asdict(cfg), "splits": {}}
    for split, rows in loaded.items():
        converted=_convert(rows)
        path=out_dir / f"episode_reward_focus_{split}{suffix}"
        _write(path, converted, bool(cfg.gzip_output))
        report["splits"][split]={**_summary(converted), "source_rows": len(rows), "output": str(path)}
    sp=out_dir / "episode_reward_focus_summary.json"
    sp.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--test-jsonl", required=True)
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--no-gzip-output", dest="gzip_output", action="store_false")
    p.set_defaults(gzip_output=RewardFocusCfg.gzip_output)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(RewardFocusCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
