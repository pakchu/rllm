"""Build/evaluate hierarchical MTF mode targets for Kimchi-flow activation.

The target is not per-trade GOOD/BAD. It asks for a higher-level mode:
- BROAD_ON: activate all Kimchi-flow rule signals in this higher-timeframe regime.
- SELECTIVE: require lower-timeframe confirmation.
- AVOID: abstain from this regime.

Rows are still individual signal rows, but labels are assigned from past-only-ish
regime bucket evidence computed on a fit split and then applied to val/test.
This is a diagnostic bridge before training Gemma on regime-mode selection.
"""
from __future__ import annotations

import argparse, json, re
from collections import defaultdict
from pathlib import Path
from typing import Any

SYMBOL_RE = re.compile(r"^(4H|1D|3D|1W) (Regime|Location):\s*(\S+)\s*$")
NUM_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9 /()_%.-]*?):\s*(-?\d+(?:\.\d+)?)\s*$")


def load_jsonl(path: str) -> list[dict[str, Any]]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + ("\n" if rows else ""))


def parse_prompt(prompt: str) -> tuple[dict[str, str], dict[str, float]]:
    sym: dict[str, str] = {}
    nums: dict[str, float] = {}
    for line in str(prompt).splitlines():
        line = line.strip()
        m = SYMBOL_RE.match(line)
        if m:
            sym[f"{m.group(1)}_{m.group(2).lower()}"] = m.group(3)
        elif line.startswith("MTF Activation Mode:"):
            sym["mtf_mode"] = line.split(":", 1)[1].strip()
        m2 = NUM_RE.match(line)
        if m2:
            key = re.sub(r"[^A-Za-z0-9]+", "_", m2.group(1).strip()).strip("_").lower()
            nums[key] = float(m2.group(2))
    return sym, nums


def target_obj(row: dict[str, Any]) -> dict[str, Any]:
    return json.loads(row["target"])


def bucket_key(row: dict[str, Any], keys: list[str]) -> str:
    sym, _ = parse_prompt(row.get("prompt", ""))
    return "|".join(f"{k}={sym.get(k, 'MISSING')}" for k in keys)


def fit_bucket_modes(rows: list[dict[str, Any]], keys: list[str], *, broad_min_ret: float, selective_min_gap: float, min_rows: int) -> dict[str, dict[str, Any]]:
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by[bucket_key(row, keys)].append(row)
    modes: dict[str, dict[str, Any]] = {}
    for key, rs in by.items():
        rets = [float(r.get("trade_ret_pct", 0.0)) for r in rs]
        targets = [target_obj(r)["decision"] == "ACTIVATE" for r in rs]
        all_ret = sum(rets)
        oracle_ret = sum(r for r, ok in zip(rets, targets) if ok)
        gap = oracle_ret - all_ret
        if len(rs) < min_rows:
            mode = "SELECTIVE"
        elif all_ret >= broad_min_ret and gap <= selective_min_gap:
            mode = "BROAD_ON"
        elif all_ret < 0.0 and oracle_ret <= 0.0:
            mode = "AVOID"
        else:
            mode = "SELECTIVE"
        modes[key] = {"mode": mode, "rows": len(rs), "all_ret_pct": all_ret, "oracle_ret_pct": oracle_ret, "gap_pct": gap, "oracle_activations": sum(targets)}
    return modes


def mode_for(row: dict[str, Any], modes: dict[str, dict[str, Any]], keys: list[str]) -> str:
    return modes.get(bucket_key(row, keys), {"mode": "SELECTIVE"})["mode"]


def lower_selective_rule(row: dict[str, Any]) -> bool:
    """Simple lower-timeframe confirmation copied from stable-v1 intuition."""
    _, nums = parse_prompt(row.get("prompt", ""))
    score = 0
    if nums.get("llm_long_context_score", 0.0) >= 2.0 or nums.get("llm_short_context_score", 0.0) >= 3.0:
        score += 1
    if nums.get("side_pressure_score", 0.0) >= 0.0:
        score += 1
    if nums.get("past_return_2h", 0.0) >= 0.0:
        score += 1
    if nums.get("tradeability_score", 0.0) >= 0.5:
        score += 1
    if nums.get("llm_failure_cue_score", 0.0) >= 2.0:
        score -= 1
    return score >= 3


def evaluate_policy(rows: list[dict[str, Any]], modes: dict[str, dict[str, Any]], keys: list[str], *, selective_rule: str) -> dict[str, Any]:
    pred_activate=[]
    labels=[]
    rets=[]
    mode_counts=defaultdict(int)
    for row in rows:
        mode=mode_for(row,modes,keys)
        mode_counts[mode]+=1
        if mode == "BROAD_ON":
            pred=True
        elif mode == "AVOID":
            pred=False
        else:
            pred = lower_selective_rule(row) if selective_rule == "simple" else (target_obj(row)["decision"] == "ACTIVATE")
        pred_activate.append(pred)
        labels.append(target_obj(row)["decision"] == "ACTIVATE")
        rets.append(float(row.get("trade_ret_pct", 0.0)))
    pred_ret=sum(r for r,p in zip(rets,pred_activate) if p)
    oracle=sum(r for r,y in zip(rets,labels) if y)
    all_ret=sum(rets)
    return {
        "rows": len(rows), "mode_counts": dict(mode_counts),
        "pred_sum_ret_pct": pred_ret, "oracle_sum_ret_pct": oracle, "all_activate_ret_pct": all_ret,
        "pred_activations": sum(pred_activate), "oracle_activations": sum(labels),
        "tp": sum(p and y for p,y in zip(pred_activate,labels)),
        "fp": sum(p and not y for p,y in zip(pred_activate,labels)),
        "fn": sum((not p) and y for p,y in zip(pred_activate,labels)),
        "tn": sum((not p) and (not y) for p,y in zip(pred_activate,labels)),
    }


def mode_prompt(row: dict[str, Any], keys: list[str]) -> str:
    sym, nums = parse_prompt(row.get("prompt", ""))
    lines = [
        "You are a BTCUSDT higher-timeframe Kimchi-flow mode selector.",
        "Use only completed higher-timeframe regime context and past-only micro context.",
        "Choose whether the Kimchi-flow rule should be broad-on, selective, or avoided now.",
        "Return compact JSON with keys: mtf_mode, activation_policy, confidence.",
        "Allowed activation_policy: BROAD_ON, SELECTIVE, AVOID.",
        "",
        "Higher-timeframe symbolic context:",
    ]
    for k in keys:
        lines.append(f"{k}: {sym.get(k, 'MISSING')}")
    lines += ["", "Higher-timeframe numeric context:"]
    for k in ["3d_return_4","3d_range_1","3d_drawdown_4","1w_return_4","1w_range_1","1w_drawdown_4","mtf_stress_total"]:
        if k in nums:
            lines.append(f"{k}: {nums[k]:.6g}")
    lines += ["", "Lower-timeframe summary:"]
    for k in ["kimchi_flow_change","trades_participation","llm_long_context_score","llm_short_context_score","llm_failure_cue_score","side_pressure_score","past_return_2h","tradeability_score"]:
        if k in nums:
            lines.append(f"{k}: {nums[k]:.6g}")
    return "\n".join(lines)


def build_mode_rows(rows: list[dict[str, Any]], modes: dict[str, dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    out=[]
    for row in rows:
        mode=mode_for(row,modes,keys)
        target={"mtf_mode":"KIMCHI_FLOW_MTF", "activation_policy": mode, "confidence":"HIGH"}
        out.append({
            "task":"kimchi_flow_mtf_mode_sft",
            "date":row["date"],
            "prompt":mode_prompt(row,keys),
            "target":json.dumps(target, sort_keys=True, separators=(",", ":")),
            "trade_ret_pct":float(row.get("trade_ret_pct",0.0)),
            "source_target":row["target"],
            "source_trade":row.get("source_trade",{}),
            "leakage_guard":{"mode_fit_uses_fit_split_only": True, "prompt_uses_future_path": False},
        })
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    train=load_jsonl(args.train_jsonl); val=load_jsonl(args.val_jsonl); test=load_jsonl(args.test_jsonl)
    keys=[x.strip() for x in args.bucket_keys.split(',') if x.strip()]
    fit_rows = train + (val if args.fit_on_trainval else [])
    modes=fit_bucket_modes(fit_rows,keys,broad_min_ret=args.broad_min_ret,selective_min_gap=args.selective_min_gap,min_rows=args.min_bucket_rows)
    report={
        "bucket_keys":keys,"fit_on_trainval":args.fit_on_trainval,"modes":modes,
        "train":evaluate_policy(train,modes,keys,selective_rule=args.selective_rule),
        "val":evaluate_policy(val,modes,keys,selective_rule=args.selective_rule),
        "test":evaluate_policy(test,modes,keys,selective_rule=args.selective_rule),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report,indent=2,ensure_ascii=False))
    if args.output_prefix:
        write_jsonl(args.output_prefix+"_train.jsonl", build_mode_rows(train,modes,keys))
        write_jsonl(args.output_prefix+"_val.jsonl", build_mode_rows(val,modes,keys))
        write_jsonl(args.output_prefix+"_test.jsonl", build_mode_rows(test,modes,keys))
    return report


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument('--train-jsonl',required=True); p.add_argument('--val-jsonl',required=True); p.add_argument('--test-jsonl',required=True)
    p.add_argument('--bucket-keys',default='3D_regime,1W_regime')
    p.add_argument('--fit-on-trainval', action='store_true')
    p.add_argument('--broad-min-ret',type=float,default=2.0)
    p.add_argument('--selective-min-gap',type=float,default=4.0)
    p.add_argument('--min-bucket-rows',type=int,default=5)
    p.add_argument('--selective-rule',choices=['simple','oracle'],default='simple')
    p.add_argument('--output',required=True); p.add_argument('--output-prefix',default='')
    args=p.parse_args()
    print(json.dumps(run(args),indent=2,ensure_ascii=False))

if __name__=='__main__': main()
