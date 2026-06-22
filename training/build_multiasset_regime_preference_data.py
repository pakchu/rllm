"""Build preference pairs for monthly regime policy selection.

Chosen response is the online trailing-evidence bandit decision. Rejected responses
are alternative policy choices for the same prompt. This is not an oracle reward
preference dataset; it teaches consistency with the no-lookahead selector.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

POLICIES = ["excess_spread", "utility1_pos", "utility1_inv", "utility3_pos", "utility3_inv", "cash"]


def _response(policy: str, chosen_scores: dict[str, float]) -> str:
    sorted_scores = sorted(chosen_scores.items(), key=lambda kv: kv[1], reverse=True)
    score = float(chosen_scores.get(policy, 0.0))
    best = sorted_scores[0][0] if sorted_scores else policy
    margin = float(sorted_scores[0][1] - sorted_scores[1][1]) if len(sorted_scores) > 1 else 0.0
    analyzer = {
        "regime_actionability": "avoid" if policy == "cash" else "trade",
        "selected_family": policy,
        "evidence_strength": "high" if score > 3 and policy == best else "medium" if score > 0 else "low",
        "score_margin": round(margin if policy == best else -abs(float(chosen_scores.get(best, 0.0)) - score), 4),
        "risk_note": "preference_candidate_response",
    }
    trader = {"policy": policy, "allow_trade": policy != "cash", "reason_code": "monthly_policy_preference_candidate"}
    return json.dumps({"analyzer": analyzer, "trader": trader}, ensure_ascii=False, sort_keys=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = [json.loads(line) for line in open(args.sft_jsonl) if line.strip()]
    out=[]
    for r in rows:
        month = r["metadata"]["month"]
        selected = r["metadata"]["selected_policy"]
        # Recover trailing scores from prompt text conservatively; for rejected responses score values are only explanatory.
        # Use target-selected score margin semantics, not future outcome.
        scores = {p: 0.0 for p in POLICIES}
        for line in r["prompt"].splitlines():
            if "trailing_scores[" in line:
                chunk = line.split("trailing_scores[",1)[1].split("]",1)[0]
                for part in chunk.split(","):
                    if "=" in part:
                        k,v=part.strip().split("=",1)
                        try: scores[k]=float(v)
                        except ValueError: pass
        chosen = r["target"]
        for policy in POLICIES:
            if policy == selected:
                continue
            rejected = _response(policy, scores)
            out.append({
                "task":"multiasset_monthly_regime_policy_dpo",
                "prompt":r["prompt"],
                "chosen":chosen,
                "rejected":rejected,
                "messages":[m for m in r["messages"][:2]],
                "metadata":{
                    "month":month,
                    "chosen_policy":selected,
                    "rejected_policy":policy,
                    "leakage_guard":"chosen is no-lookahead bandit decision; rejected is alternative policy text; future outcome not used",
                }
            })
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text("\n".join(json.dumps(x,ensure_ascii=False,sort_keys=True) for x in out)+("\n" if out else ""))
    counts=Counter((x["metadata"]["chosen_policy"],x["metadata"]["rejected_policy"]) for x in out)
    summary={"pairs":len(out),"source_rows":len(rows),"chosen_counts":dict(Counter(x["metadata"]["chosen_policy"] for x in out)),"rejected_counts":dict(Counter(x["metadata"]["rejected_policy"] for x in out)),"leakage_guard":"preference labels mirror trailing bandit, not future oracle"}
    if args.summary_output:
        Path(args.summary_output).parent.mkdir(parents=True,exist_ok=True); Path(args.summary_output).write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    return summary


def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--sft-jsonl',required=True); p.add_argument('--output',required=True); p.add_argument('--summary-output',default=''); return p.parse_args()


def main(): print(json.dumps(run(parse_args()),indent=2,ensure_ascii=False))
if __name__=='__main__': main()
