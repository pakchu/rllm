"""Search VPIN/formulaic features as dual gates on existing signal-pool events.

Unlike additive sleeve tests, this creates gated variants of existing alphas:
    original_signal AND/NOT vpin/formulaic_gate
Then it compares standalone gated variants and portfolio combinations.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

import training.search_portfolio_gross20_step005_mdd20_with_dynamic as base
from training.search_vpin_formulaic_alpha import add_vpin_formulaic_features, active_from_terms

OUT = "results/vpin_dual_gate_signal_pool_2026-07-09.json"
DOC = "docs/vpin-dual-gate-signal-pool-2026-07-09.md"
CORE_SLEEVES = [
    "nonpb30_taker", "oi_raw", "rex_rule", "oi_upbit_ratio288_low",
    "bear_rex_short", "oi_alt_ratio72_dyn_exit", "oi_low", "oi_high_sel",
]
BASE_TOP = {"nonpb30_taker": 0.65, "oi_raw": 0.55, "rex_rule": 2.00, "oi_upbit_ratio288_low": 2.25, "bear_rex_short": 0.30, "oi_alt_ratio72_dyn_exit": 0.35}
LIVE_575 = {"nonpb30_taker": 0.95, "oi_low": 0.95, "oi_high_sel": 0.95, "bear_rex_short": 2.90}


def build_gate_masks(market, feat) -> dict[str, np.ndarray]:
    f = add_vpin_formulaic_features(market, feat)
    gates: dict[str, np.ndarray] = {}
    high_sell = active_from_terms(f, [("vp_vpin_z_144", ">=", 1.1373129661931034), ("vp_imb_z_72", "<=", -1.0336282282038078)])
    high_buy = active_from_terms(f, [("vp_vpin_z_144", ">=", 1.1373129661931034), ("vp_imb_z_72", ">=", 1.0295042386067654)])
    low_vpin = active_from_terms(f, [("vp_vpin_z_144", "<=", 0.7063)])
    vwap_breakout = active_from_terms(f, [("fq_vwap_revert_z", "<=", -0.8658), ("vp_vpin_z_144", "<=", 0.7063)])
    vol_pressure_pos = active_from_terms(f, [("fq_volume_delta_rank", ">=", 0.608900569986259), ("fq_signed_vol_pressure_rank", ">=", 0.5270347802624189)])
    vol_pressure_neg = active_from_terms(f, [("fq_volume_delta_rank", ">=", 0.608900569986259), ("fq_signed_vol_pressure_rank", "<=", 0.4757247204932479)])
    toxic_rally = active_from_terms(f, [("vx_toxic_rally", ">=", 2.9253758632146103)])
    gates.update({
        "g_vpin_high_sell": high_sell,
        "g_not_vpin_high_sell": ~high_sell,
        "g_vpin_high_buy": high_buy,
        "g_not_vpin_high_buy": ~high_buy,
        "g_low_vpin": low_vpin,
        "g_not_low_vpin": ~low_vpin,
        "g_vwap_breakout": vwap_breakout,
        "g_not_vwap_breakout": ~vwap_breakout,
        "g_vol_pressure_pos": vol_pressure_pos,
        "g_vol_pressure_neg": vol_pressure_neg,
        "g_not_toxic_rally": ~toxic_rally,
    })
    return {k: np.asarray(v, bool) for k, v in gates.items()}


def clone_gated_events(events: list[dict[str, Any]], masks: dict[str, np.ndarray], gate_masks: dict[str, np.ndarray]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    new_events = list(events)
    meta: dict[str, Any] = {}
    for sl in CORE_SLEEVES:
        source = [e for e in events if e["sleeve"] == sl]
        if not source:
            continue
        for gname, gmask in gate_masks.items():
            name = f"{sl}__{gname}"
            kept = []
            for e in source:
                ip = int(e["signal_pos"])
                if 0 <= ip < len(gmask) and bool(gmask[ip]):
                    ee = dict(e)
                    ee["sleeve"] = name
                    kept.append(ee)
            if kept:
                new_events.extend(kept)
                meta[name] = {"source_sleeve": sl, "gate": gname, "kept": len(kept), "source": len(source), "keep_rate": len(kept) / max(1, len(source))}
    return new_events, meta


def clean(w: dict[str, float]) -> dict[str, float]:
    return {k: round(float(v), 6) for k, v in w.items() if v > 1e-12}


def metric_rows_for_single(by, years, sleeves):
    rows = []
    for sl in sleeves:
        w = {s: (1.0 if s == sl else 0.0) for s in sleeves}
        st = base.metrics(by, years, w)
        rows.append({"sleeve": sl, "stats": st, "score_tuple": base.score(st)})
    rows.sort(key=lambda r: r["score_tuple"], reverse=True)
    return rows


def candidate_weights(by, years, sleeves, gated_names):
    out=[]; seen=set()
    def add(w):
        ww = base.quantize_portfolio({s:max(0.0,float(w.get(s,0.0))) for s in sleeves})
        if sum(ww.values()) <= 0 or not base.weight_unit_ok(ww): return
        key = tuple(ww[s] for s in sleeves)
        if key not in seen:
            seen.add(key); out.append(ww)
    seeds=[BASE_TOP, LIVE_575]
    # original baselines
    for s in seeds:
        add(s); add(base.scale_to_mdd(by, years, s, 19.7)); add(base.scale_to_mdd(by, years, s, 19.95))
    # replace each source sleeve by one gated version with same/nearby weight
    by_source={}
    for gn in gated_names:
        src=gn.split('__',1)[0]
        by_source.setdefault(src,[]).append(gn)
    for seed in seeds:
        for src, variants in by_source.items():
            if seed.get(src,0) <= 0: continue
            for gn in variants:
                w=dict(seed); val=w.pop(src); w[gn]=val
                add(w); add(base.scale_to_mdd(by, years, w, 19.7)); add(base.scale_to_mdd(by, years, w, 19.95))
                for mult in [0.5, 0.75, 1.25, 1.5]:
                    ww=dict(seed); val=ww.pop(src); ww[gn]=val*mult
                    add(ww); add(base.scale_to_mdd(by, years, ww, 19.8))
    # two gated replacements / mixed source variants
    rng=random.Random(90991)
    interesting=[g for g in gated_names if any(x in g for x in ['not_vpin_high_sell','not_vpin_high_buy','low_vpin','vpin_high_buy','vpin_high_sell','not_toxic_rally'])]
    for _ in range(1800):
        seed=dict(rng.choice(seeds))
        ww=dict(seed)
        for gn in rng.sample(interesting, rng.randint(1, min(3,len(interesting)))):
            src=gn.split('__',1)[0]
            if src in ww and ww[src] > 0:
                val=ww.pop(src)
                ww[gn]=ww.get(gn,0)+val*rng.choice([0.5,0.75,1.0,1.25])
        # perturb remaining weights
        for k in list(ww):
            if rng.random()<0.25:
                ww[k]=max(0, ww[k]+rng.choice([-0.25,-0.15,0.15,0.25]))
        add(ww); add(base.scale_to_mdd(by, years, ww, 19.8))
    return out


def main() -> None:
    market, feat, masks, years, events, _ = base.vw.build_events()
    base.add_old_live_events(events, market, feat, masks)
    events.extend(base.dx.build_dynamic_sleeves(market, feat, masks, years))
    gate_masks = build_gate_masks(market, feat)
    all_events, gated_meta = clone_gated_events(events, masks, gate_masks)
    sleeves = list(base.SLEEVES)
    for name in gated_meta:
        if name not in sleeves:
            sleeves.append(name)
    base.SLEEVES = sleeves
    by = base.arrays(all_events, masks)
    singles = metric_rows_for_single(by, years, sleeves)
    gated_names = list(gated_meta)

    rows=[]
    for w in candidate_weights(by, years, sleeves, gated_names):
        st=base.metrics(by, years, w)
        gated_gross=sum(v for k,v in w.items() if '__g_' in k)
        rows.append({"weights": clean(w), "gross": round(sum(w.values()),6), "gated_gross": round(gated_gross,6), "stats": st, "passes_mdd20_ratio5": base.passes(st), "score_tuple": base.score(st)})
    rows.sort(key=lambda r:r['score_tuple'], reverse=True)
    qualified=[r for r in rows if r['passes_mdd20_ratio5']]
    qualified_gated=[r for r in qualified if r['gated_gross'] >= 0.10]
    report={
        "protocol": {"cost_each_side": base.COST, "mode": "dual gate: original signal event filtered by VPIN/formulaic gate at signal_pos", "condition": "test2024/eval2025/ytd2026 strict_mdd<=20 and CAGR/MDD>=5; weights >=0.10 step 0.05"},
        "gated_meta": gated_meta,
        "event_counts": {sp: dict(Counter(e['sleeve'] for e in all_events if e['split']==sp)) for sp in masks},
        "evaluated_unique": len(rows), "qualified_count": len(qualified), "qualified_gated_count": len(qualified_gated),
        "top_single_gated": [{k:v for k,v in r.items() if k!='score_tuple'} for r in singles if '__g_' in r['sleeve']][:40],
        "top_qualified": [{k:v for k,v in r.items() if k!='score_tuple'} for r in qualified[:30]],
        "top_qualified_gated": [{k:v for k,v in r.items() if k!='score_tuple'} for r in qualified_gated[:30]],
        "top_diagnostic": [{k:v for k,v in r.items() if k!='score_tuple'} for r in rows[:30]],
    }
    Path(OUT).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    def fmt(s): return f"{s['total_return_pct']:.2f}/{s['cagr_pct']:.2f}/{s['strict_mdd_pct']:.2f}/{s['cagr_to_strict_mdd']:.2f}/{s['trade_entries']}"
    md=["# VPIN dual-gate signal-pool scan (2026-07-09)", "", json.dumps(report['protocol'], ensure_ascii=False), "", f"evaluated_unique={len(rows)}, qualified_count={len(qualified)}, qualified_gated_count={len(qualified_gated)}", "", "## Top qualified gated portfolios", "", "| rank | gross | gated gross | weights | 2024 | 2025 | 2026 |", "|---:|---:|---:|---|---:|---:|---:|"]
    for i,r in enumerate(report['top_qualified_gated'][:20],1):
        st=r['stats']; md.append(f"| {i} | {r['gross']:.2f} | {r['gated_gross']:.2f} | `{r['weights']}` | {fmt(st['test2024'])} | {fmt(st['eval2025'])} | {fmt(st['ytd2026'])} |")
    md += ["", "## Top standalone gated sleeves", "", "| rank | sleeve | 2024 | 2025 | 2026 |", "|---:|---|---:|---:|---:|"]
    for i,r in enumerate(report['top_single_gated'][:25],1):
        st=r['stats']; md.append(f"| {i} | `{r['sleeve']}` | {fmt(st['test2024'])} | {fmt(st['eval2025'])} | {fmt(st['ytd2026'])} |")
    md += ["", "## Top qualified overall", "", "| rank | gross | gated gross | weights | 2024 | 2025 | 2026 |", "|---:|---:|---:|---|---:|---:|---:|"]
    for i,r in enumerate(report['top_qualified'][:15],1):
        st=r['stats']; md.append(f"| {i} | {r['gross']:.2f} | {r['gated_gross']:.2f} | `{r['weights']}` | {fmt(st['test2024'])} | {fmt(st['eval2025'])} | {fmt(st['ytd2026'])} |")
    Path(DOC).write_text('\n'.join(md)+'\n')
    print(json.dumps({"output":OUT,"doc":DOC,"evaluated":len(rows),"qualified":len(qualified),"qualified_gated":len(qualified_gated),"top_qualified_gated":report['top_qualified_gated'][:5],"top_single_gated":report['top_single_gated'][:5]}, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
