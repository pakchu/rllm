"""Build causal reference-comparison pair rows for candidate scoring.

Unlike split-internal pairwise ranking, this compares each evaluation candidate
only against historical reference candidates.  That avoids future-pool leakage
when converting pairwise LLM outputs into live-style candidate scores.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from training.kimchi_flow_pairwise_dataset import bucket, candidate_meta, candidate_text, load_jsonl, make_prompt


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + ("\n" if rows else ""))


def _decision(row: dict[str, Any]) -> str:
    try:
        return str(json.loads(row["target"]).get("decision", ""))
    except Exception:
        return ""


def _ret(row: dict[str, Any]) -> float:
    return float(row.get("trade_ret_pct", 0.0))


def build_rows(eval_rows: list[dict[str, Any]], ref_rows: list[dict[str, Any]], *, refs_per_side: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_bucket: dict[str, list[dict[str, Any]]] = {}
    for row in ref_rows:
        by_bucket.setdefault(bucket(row), []).append(row)
    all_refs = list(ref_rows)
    out: list[dict[str, Any]] = []
    for cand in eval_rows:
        cand_bucket = bucket(cand)
        refs = by_bucket.get(cand_bucket) or all_refs
        good = [r for r in refs if _decision(r) == "ACTIVATE"]
        bad = [r for r in refs if _decision(r) != "ACTIVATE"]
        chosen: list[dict[str, Any]] = []
        for pool in (good, bad):
            if not pool:
                continue
            k = min(int(refs_per_side), len(pool))
            chosen.extend(rng.sample(pool, k))
        if not chosen:
            continue
        for ref in chosen:
            choice = "A" if _ret(cand) >= _ret(ref) else "B"
            out.append(
                {
                    "task": "kimchi_flow_reference_candidate_score",
                    "date": cand.get("date"),
                    "bucket": cand_bucket,
                    "prompt": make_prompt(cand, ref),
                    "target": json.dumps({"choice": choice, "confidence": "HIGH"}, sort_keys=True, separators=(",", ":")),
                    "candidate_a": {**candidate_meta(cand), "role": "eval_candidate"},
                    "candidate_b": {**candidate_meta(ref), "role": "historical_reference"},
                    "winner_ret_pct": max(_ret(cand), _ret(ref)),
                    "loser_ret_pct": min(_ret(cand), _ret(ref)),
                    "leakage_guard": {
                        "prompt_uses_future_path": False,
                        "reference_pool_is_historical_only": True,
                        "target_uses_realized_pair_order_for_evaluation_only": True,
                    },
                }
            )
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--reference-jsonl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--summary-output", default="")
    p.add_argument("--refs-per-side", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    rows = build_rows(load_jsonl(args.eval_jsonl), load_jsonl(args.reference_jsonl), refs_per_side=args.refs_per_side, seed=args.seed)
    write_jsonl(args.output, rows)
    summary = {
        "eval_jsonl": args.eval_jsonl,
        "reference_jsonl": args.reference_jsonl,
        "rows": len(rows),
        "eval_candidates": len(load_jsonl(args.eval_jsonl)),
        "refs_per_side": args.refs_per_side,
        "historical_reference_only": True,
    }
    if args.summary_output:
        Path(args.summary_output).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
