"""Convert focused reward rows into oracle single-policy predictions.

This is an explicit upper-bound diagnostic: it uses future-derived focus labels
from the row target.  It must not be used as deployable validation.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from training.single_policy_sft_data import exit_profile_for_hold


def _load(path: str) -> list[dict[str, Any]]:
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _policy(row: dict[str, Any], *, allow_mixed_high: bool = False) -> dict[str, str]:
    target=json.loads(str(row.get("target", "{}")))
    cand=dict(row.get("candidate") or {})
    utility=str(target.get("utility_bucket", ""))
    shape=str(target.get("path_shape", ""))
    side=str(cand.get("side", "NO_TRADE")).upper()
    horizon=int(cand.get("horizon", 288) or 288)
    trade = utility == "UTILITY_HIGH" and (shape == "CLEAN_WIN_PATH" or (allow_mixed_high and shape == "MIXED_PATH")) and side in {"LONG", "SHORT"}
    if not trade:
        return {"regime":"RANGE","edge_quality":"NONE","risk":"LOW","action":"NO_TRADE","exit_profile":"AVOID","confidence":"LOW"}
    risk = "LOW" if shape == "CLEAN_WIN_PATH" else "MID"
    return {
        "regime": "TREND_UP" if side == "LONG" else "TREND_DOWN",
        "edge_quality": "STRONG" if shape == "CLEAN_WIN_PATH" else "MODERATE",
        "risk": risk,
        "action": side,
        "exit_profile": exit_profile_for_hold(horizon),
        "confidence": "HIGH",
    }


def convert(rows: list[dict[str, Any]], *, allow_mixed_high: bool = False) -> list[dict[str, Any]]:
    out=[]
    for row in rows:
        pred=_policy(row, allow_mixed_high=allow_mixed_high)
        out.append({
            "date": row.get("date"),
            "signal_pos": row.get("signal_pos"),
            "candidate": row.get("candidate") or {},
            "policy_prediction": pred,
            "policy_target": json.loads(str(row.get("target", "{}"))),
            "target_audit": row.get("target_audit") or {},
            "oracle_mode": True,
        })
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows=_load(args.input_jsonl)
    out=convert(rows, allow_mixed_high=bool(args.allow_mixed_high))
    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_jsonl).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in out)+"\n")
    actions=Counter(r["policy_prediction"]["action"] for r in out)
    shapes=Counter(str(r["policy_target"].get("path_shape")) for r in out)
    utilities=Counter(str(r["policy_target"].get("utility_bucket")) for r in out)
    report={
        "input_jsonl": args.input_jsonl,
        "output_jsonl": args.output_jsonl,
        "rows": len(out),
        "actions": dict(actions),
        "path_shapes": dict(shapes),
        "utility_buckets": dict(utilities),
        "allow_mixed_high": bool(args.allow_mixed_high),
        "leakage_guard": {"oracle_uses_future_derived_target": True, "deployable": False},
    }
    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--summary", default="")
    p.add_argument("--allow-mixed-high", action="store_true")
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
