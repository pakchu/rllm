"""Build/evaluate pairwise ranking rows for Kimchi-flow selective mode."""
from __future__ import annotations

import argparse, json, random, re
from pathlib import Path
from typing import Any

NUM_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9 /()_%.-]*?):\s*(-?\d+(?:\.\d+)?)\s*$")
SYM_RE = re.compile(r"^(4H|1D|3D|1W) (Regime|Location):\s*(\S+)\s*$")

FEATURE_KEYS = [
    "kimchi_flow_change","trades_participation","taker_imbalance",
    "llm_long_context_score","llm_short_context_score","llm_failure_cue_score",
    "side_pressure_score","past_return_2h","past_return_8h","tradeability_score",
    "range_position","past_path_return_6h","past_path_drawdown_12h",
    "3d_return_4","3d_range_1","3d_drawdown_4","1w_return_4","1w_range_1","1w_drawdown_4","mtf_stress_total",
]
SYM_KEYS = ["3D_regime","3D_location","1W_regime","1W_location","mtf_mode"]


def load_jsonl(path: str) -> list[dict[str, Any]]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + ("\n" if rows else ""))


def parse_prompt(prompt: str) -> tuple[dict[str, str], dict[str, float]]:
    sym={}; nums={}
    for line in str(prompt).splitlines():
        line=line.strip()
        m=SYM_RE.match(line)
        if m: sym[f"{m.group(1)}_{m.group(2).lower()}"]=m.group(3)
        elif line.startswith('MTF Activation Mode:'): sym['mtf_mode']=line.split(':',1)[1].strip()
        m2=NUM_RE.match(line)
        if m2:
            key=re.sub(r"[^A-Za-z0-9]+","_",m2.group(1).strip()).strip('_').lower()
            nums[key]=float(m2.group(2))
    return sym, nums


def target(row: dict[str, Any]) -> dict[str, Any]:
    return json.loads(row['target'])


def bucket(row: dict[str, Any]) -> str:
    sym,_=parse_prompt(row['prompt'])
    return '|'.join(sym.get(k,'MISSING') for k in ['3D_regime','1W_regime'])


def candidate_text(row: dict[str, Any]) -> str:
    sym, nums = parse_prompt(row['prompt'])
    lines=[]
    for k in SYM_KEYS:
        if k in sym: lines.append(f"{k}: {sym[k]}")
    for k in FEATURE_KEYS:
        if k in nums: lines.append(f"{k}: {nums[k]:.6g}")
    src=target(row)
    lines.append(f"fixed_rule_side: {row.get('source_trade',{}).get('side', src.get('side','UNKNOWN'))}")
    return "\n".join(lines)


def make_prompt(a: dict[str, Any], b: dict[str, Any]) -> str:
    return "\n".join([
        "You are a BTCUSDT Kimchi-flow selective-mode ranker.",
        "Both candidates are past-only fixed-rule signals from comparable higher-timeframe context.",
        "Choose which candidate should be activated. Prefer the candidate with better expected 24h fixed-rule return after costs.",
        "Return compact JSON with keys: choice, confidence. Allowed choice: A or B.",
        "",
        "Candidate A:", candidate_text(a),
        "",
        "Candidate B:", candidate_text(b),
    ])


def build_pairs(rows: list[dict[str, Any]], *, max_pairs: int, seed: int) -> list[dict[str, Any]]:
    rng=random.Random(seed)
    by={}
    for r in rows: by.setdefault(bucket(r), []).append(r)
    pairs=[]
    for key, rs in by.items():
        good=[r for r in rs if target(r)['decision']=='ACTIVATE']
        bad=[r for r in rs if target(r)['decision']!='ACTIVATE']
        for g in good:
            for b in bad:
                if abs(float(g['trade_ret_pct'])-float(b['trade_ret_pct'])) < 0.25:
                    continue
                if rng.random() < 0.5:
                    a,brow,choice=g,b,'A'
                else:
                    a,brow,choice=b,g,'B'
                pairs.append({
                    'task':'kimchi_flow_pairwise_rank_sft',
                    'date':g['date'], 'bucket':key,
                    'prompt':make_prompt(a,brow),
                    'target':json.dumps({'choice':choice,'confidence':'HIGH'}, sort_keys=True, separators=(',',':')),
                    'winner_ret_pct':float(g['trade_ret_pct']), 'loser_ret_pct':float(b['trade_ret_pct']),
                    'leakage_guard': {'prompt_uses_future_path': False, 'target_uses_realized_pair_order_for_training_only': True},
                })
    rng.shuffle(pairs)
    if max_pairs and len(pairs)>max_pairs: pairs=pairs[:max_pairs]
    return pairs


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument('--input-jsonl', required=True); p.add_argument('--output', required=True); p.add_argument('--summary-output', default='')
    p.add_argument('--max-pairs', type=int, default=1000); p.add_argument('--seed', type=int, default=42)
    args=p.parse_args()
    rows=load_jsonl(args.input_jsonl)
    pairs=build_pairs(rows,max_pairs=args.max_pairs,seed=args.seed)
    write_jsonl(args.output,pairs)
    summary={'input':args.input_jsonl,'output':args.output,'rows':len(rows),'pairs':len(pairs),'prompt_chars':{'min':min([len(x['prompt']) for x in pairs], default=0),'max':max([len(x['prompt']) for x in pairs], default=0),'mean':sum(len(x['prompt']) for x in pairs)/max(1,len(pairs))}}
    if args.summary_output: Path(args.summary_output).write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print(json.dumps(summary,indent=2,ensure_ascii=False))

if __name__=='__main__': main()
