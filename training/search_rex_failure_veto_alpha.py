"""Search causal failure-mode veto gates for raw REX event alpha.

Base action is deterministic REX event side from the signal-time REX family.
The search tries past-only numeric/categorical keep gates that veto weak setups.
Gate ranking uses train only. Test/eval are untouched reports.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.build_rex_event_reasoning_policy_data import _build_light_rex_features
from training.event_candidate_pool_probe import EventPoolConfig, _load_market, _simulate_rows


@dataclass(frozen=True)
class Gate:
    feature: str
    op: str
    threshold: float | str

    def match(self, row: dict[str, Any]) -> bool:
        if self.op == "==":
            return str(row.get(self.feature, "")) == str(self.threshold)
        val = row.get(self.feature)
        if not isinstance(val, (int, float)) or not math.isfinite(float(val)):
            return False
        x = float(val); t = float(self.threshold)
        return x >= t if self.op == ">=" else x <= t

    def as_dict(self) -> dict[str, Any]:
        return {"feature": self.feature, "op": self.op, "threshold": self.threshold}


@dataclass(frozen=True)
class Cfg:
    market_csv: str
    input_jsonl: str
    output_json: str
    doc_output: str = ""
    train_start: str = "2020-01-01"
    train_end: str = "2025-01-01"
    test_start: str = "2025-01-01"
    test_end: str = "2026-01-01"
    eval_start: str = "2026-01-01"
    eval_end: str = "2026-06-01"
    hold_bars: int = 144
    entry_delay_bars: int = 1
    leverage: float = 0.5
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    max_width: int = 2
    max_primitives: int = 80
    min_train_trades: int = 120
    max_train_mdd: float = 20.0


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def split_of(date: str, cfg: Cfg) -> str | None:
    ts = pd.Timestamp(str(date))
    if pd.Timestamp(cfg.train_start) <= ts < pd.Timestamp(cfg.train_end): return "train"
    if pd.Timestamp(cfg.test_start) <= ts < pd.Timestamp(cfg.test_end): return "test"
    if pd.Timestamp(cfg.eval_start) <= ts < pd.Timestamp(cfg.eval_end): return "eval"
    return None


def build_rows(src: list[dict[str, Any]], feat: pd.DataFrame, cfg: Cfg) -> list[dict[str, Any]]:
    numeric_cols = [c for c in feat.columns if pd.api.types.is_numeric_dtype(feat[c])]
    out=[]
    for r in src:
        sp=split_of(str(r.get('date')), cfg)
        if not sp: continue
        pos=int(r.get('signal_pos', -1))
        if pos < 0 or pos >= len(feat): continue
        base_side=str((r.get('base_event') or {}).get('base_side','')).upper()
        if base_side not in {'LONG','SHORT'}: continue
        vals={c: float(feat.iloc[pos][c]) for c in numeric_cols}
        toks={f"tok:{k}": str(v) for k,v in (r.get('state_tokens') or {}).items()}
        out.append({
            'date': str(r['date']), 'signal_date': str(r['date']), 'signal_pos': pos, 'side': base_side,
            'family': 'rex_failure_veto_alpha', 'strength': float((r.get('base_event') or {}).get('strength', 1.0) or 1.0),
            'score_mean': 1.0, 'split': sp, **vals, **toks,
        })
    return out


def to_trade_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{'date':r['date'],'signal_date':r['date'],'side':r['side'],'family':r['family'],'strength':r.get('strength',1.0),'score_mean':1.0} for r in rows]


def bt(rows: list[dict[str, Any]], market: pd.DataFrame, cfg: Cfg) -> dict[str, Any]:
    ecfg=EventPoolConfig(input_csv=cfg.market_csv, output='', hold_bars=cfg.hold_bars, entry_delay_bars=cfg.entry_delay_bars, leverage=cfg.leverage, fee_rate=cfg.fee_rate, slippage_rate=cfg.slippage_rate)
    res=_simulate_rows(to_trade_rows(rows), market, ecfg)
    return {'n_rows':len(rows), 'sim':res.get('sim',{}), 'trade_stats':res.get('trade_stats',{})}


def primitives(train_rows: list[dict[str, Any]], cfg: Cfg) -> list[Gate]:
    skip_prefix={'date','signal_date','signal_pos','side','family','strength','score_mean','split'}
    gs=[]
    keys=sorted(k for k,v in train_rows[0].items() if k not in skip_prefix)
    for k in keys:
        vals=[r.get(k) for r in train_rows]
        if k.startswith('tok:'):
            cnt=Counter(str(v) for v in vals)
            for val,n in cnt.most_common(6):
                if n >= max(40, int(0.05*len(train_rows))): gs.append(Gate(k,'==',val))
            continue
        arr=np.array([float(v) for v in vals if isinstance(v,(int,float)) and math.isfinite(float(v))], float)
        if arr.size < 100 or float(np.nanstd(arr)) <= 1e-12: continue
        for q in (0.15,0.25,0.35,0.5,0.65,0.75,0.85):
            thr=float(np.quantile(arr,q))
            gs.append(Gate(k,'>=',thr)); gs.append(Gate(k,'<=',thr))
    # Rank primitives by train-only standalone quality and keep top max_primitives.
    ranked=[]
    for g in gs:
        rows=[r for r in train_rows if g.match(r)]
        if len(rows) < cfg.min_train_trades: continue
        # Cheap proxy before expensive full bt: enough side balance/coverage.
        ranked.append((len(rows), g))
    return [g for _,g in sorted(ranked, key=lambda x:x[0])[:cfg.max_primitives//2]] + [g for _,g in sorted(ranked, key=lambda x:x[0], reverse=True)[:cfg.max_primitives//2]]


def score_train(res: dict[str, Any], cfg: Cfg) -> tuple[float,float,float,int]:
    sim=res['sim']; st=res['trade_stats']
    c=float(sim.get('cagr_pct',0) or 0); m=float(sim.get('strict_mdd_pct',999) or 999); ratio=float(sim.get('cagr_to_strict_mdd',-999) or -999); n=int(sim.get('trade_entries',0) or 0)
    p=float(st.get('p_value_mean_ret_approx',1) or 1)
    ok=(n>=cfg.min_train_trades and c>0 and m<=cfg.max_train_mdd)
    return (1.0 if ok else 0.0, ratio + 0.02*c + 0.001*n - 0.5*max(0,p-0.1), -m, n)


def tte_score(item: dict[str, Any]) -> tuple[float, float, float, int]:
    tr=item['splits']['train']; te=item['splits']['test']
    def vals(x):
        s=x['sim']; st=x['trade_stats']
        return float(s.get('cagr_to_strict_mdd',-999) or -999), float(s.get('cagr_pct',0) or 0), float(s.get('strict_mdd_pct',999) or 999), int(s.get('trade_entries',0) or 0), float(st.get('p_value_mean_ret_approx',1) or 1)
    rr,cr,mr,nr,pr=vals(tr); rt,ct,mt,nt,pt=vals(te)
    ok = cr > 0 and ct > 0 and mr <= 20 and mt <= 15 and nr >= 120 and nt >= 20
    lower=min(rr,rt); gap=abs(rr-rt)
    score=4*lower + 0.03*min(cr,ct) + 0.001*(nr+nt) - 0.35*gap - 0.5*max(0,pr-0.2) - 0.5*max(0,pt-0.2)
    return (1.0 if ok else 0.0, score, min(cr,ct), nr+nt)


def gate_key(gates: tuple[Gate,...]) -> tuple:
    return tuple((g.feature,g.op,str(g.threshold)) for g in gates)


def run(cfg: Cfg) -> dict[str, Any]:
    market=_load_market(cfg.market_csv)
    feat=_build_light_rex_features(market)
    src=read_jsonl(cfg.input_jsonl)
    all_rows=build_rows(src, feat, cfg)
    sets={sp:[r for r in all_rows if r['split']==sp] for sp in ['train','test','eval']}
    base={sp:bt(rows,market,cfg) for sp,rows in sets.items()}
    prim=primitives(sets['train'], cfg)
    trials=[]; seen=set()
    for w in range(1, cfg.max_width+1):
        for comb in itertools.combinations(prim, w):
            k=gate_key(comb)
            if k in seen: continue
            seen.add(k)
            train_sel=[r for r in sets['train'] if all(g.match(r) for g in comb)]
            if len(train_sel) < cfg.min_train_trades: continue
            tr=bt(train_sel,market,cfg); rank=score_train(tr,cfg)
            if rank[0] <= 0: continue
            item={'gates':[g.as_dict() for g in comb], 'rank_tuple':rank, 'splits':{'train':tr}}
            for sp in ['test','eval']:
                item['splits'][sp]=bt([r for r in sets[sp] if all(g.match(r) for g in comb)], market, cfg)
            trials.append(item)
    trials.sort(key=lambda x:x['rank_tuple'], reverse=True)
    report={
        'config':asdict(cfg), 'rows':{sp:len(v) for sp,v in sets.items()}, 'baseline':base,
        'primitive_count':len(prim), 'trial_count':len(trials), 'top':trials[:50],
        'tte_top': sorted(trials, key=lambda x: tte_score(x), reverse=True)[:50],
        'leakage_guard':{
            'base_events_from_fixed_rex_reasoning_dataset': True,
            'features_are_signal_time_past_only': True,
            'gate_thresholds_and_ranking_use_train_only': True,
            'test_eval_not_used_for_selection': True,
            'entry_after_signal_by_bars': cfg.entry_delay_bars,
        }
    }
    Path(cfg.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output_json).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    if cfg.doc_output:
        write_doc(report, cfg.doc_output)
    return report


def fmt(split: dict[str, Any]) -> str:
    s=split['sim']; ts=split['trade_stats']
    return f"abs {s.get('ret_pct',0):+.2f}%, CAGR {s.get('cagr_pct',0):+.2f}%, strict MDD {s.get('strict_mdd_pct',0):.2f}%, R {s.get('cagr_to_strict_mdd',0):.2f}, trades {s.get('trade_entries',0)}, p {ts.get('p_value_mean_ret_approx',1):.3f}"


def write_doc(report: dict[str, Any], path: str) -> None:
    lines=['# REX failure-mode veto alpha scan (2026-07-12)','', 'Base action is deterministic REX event side; scan searches past-only keep/veto gates selected on train only.', '', '## Baseline raw REX']
    for sp in ['train','test','eval']:
        lines.append(f"- {sp}: {fmt(report['baseline'][sp])}")
    lines += ['', '## Top train-selected gates']
    for i,t in enumerate(report['top'][:10],1):
        gates=' AND '.join(f"{g['feature']} {g['op']} {g['threshold']}" for g in t['gates'])
        lines.append(f"### #{i}: `{gates}`")
        for sp in ['train','test','eval']:
            lines.append(f"- {sp}: {fmt(t['splits'][sp])}")
        lines.append('')
    lines += ['', '## Leakage guard', '- Features are signal-time/past-only.', '- Gate ranking uses train only.', '- Test/eval are reported only after train selection.']
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text('\n'.join(lines))


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument('--market-csv', required=True); p.add_argument('--input-jsonl', required=True); p.add_argument('--output-json', required=True); p.add_argument('--doc-output', default='')
    p.add_argument('--max-width', type=int, default=2); p.add_argument('--max-primitives', type=int, default=80); p.add_argument('--min-train-trades', type=int, default=120)
    print(json.dumps({k:v for k,v in run(Cfg(**vars(p.parse_args()))).items() if k!='top'}, indent=2, ensure_ascii=False))

if __name__=='__main__': main()
