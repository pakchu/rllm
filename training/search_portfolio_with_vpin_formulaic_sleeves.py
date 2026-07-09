"""Add VPIN/formulaic-alpha sleeves to the existing portfolio signal pool.

Diagnostic only:
- Reuses the existing gross<=20, 6bp/side, strict-MDD portfolio engine.
- VPIN/formulaic thresholds are from train<2024 quantiles only.
- Weight constraints: nonzero >=0.10, step 0.05.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

import training.search_portfolio_gross20_step005_mdd20_with_dynamic as base
from training.search_vpin_formulaic_alpha import add_vpin_formulaic_features, active_from_terms

OUT = "results/portfolio_with_vpin_formulaic_sleeves_2026-07-09.json"
DOC = "docs/portfolio-with-vpin-formulaic-sleeves-2026-07-09.md"

# Distinct candidates from the standalone VPIN/formulaic scan. Include a few weak
# but differently timed variants so the optimizer can test diversification value.
VPIN_CANDIDATES = [
    ("vpin_sell_reversal_h144_s24", "long", 144, 24, [("vp_vpin_z_144", ">=", 1.1373129661931034), ("vp_imb_z_72", "<=", -1.0336282282038078)]),
    ("vpin_sell_reversal_h72_s24", "long", 72, 24, [("vp_vpin_z_144", ">=", 1.1373129661931034), ("vp_imb_z_72", "<=", -1.0336282282038078)]),
    ("vpin_sell_reversal_h72_s12", "long", 72, 12, [("vp_vpin_z_144", ">=", 1.1373129661931034), ("vp_imb_z_72", "<=", -1.0336282282038078)]),
    ("vpin_buy_pressure_short_h96_s12", "short", 96, 12, [("vp_vpin_z_144", ">=", 1.1373129661931034), ("vp_imb_z_72", ">=", 1.0295042386067654)]),
    ("vpin_buy_pressure_short_h144_s12", "short", 144, 12, [("vp_vpin_z_144", ">=", 1.1373129661931034), ("vp_imb_z_72", ">=", 1.0295042386067654)]),
    ("formulaic_vwap_breakout_h144_s24", "long", 144, 24, [("fq_vwap_revert_z", "<=", -0.8658), ("vp_vpin_z_144", "<=", 0.7063)]),
    ("formulaic_volume_delta_h72_s24", "long", 72, 24, [("fq_volume_delta_rank", ">=", 0.608900569986259), ("fq_signed_vol_pressure_rank", ">=", 0.5270347802624189)]),
]


def _extend_sleeves() -> list[str]:
    sleeves = list(base.SLEEVES)
    for name, *_ in VPIN_CANDIDATES:
        if name not in sleeves:
            sleeves.append(name)
    base.SLEEVES = sleeves
    return sleeves


def _append_vpin_events(events, market, feat, masks) -> dict[str, dict]:
    feat2 = add_vpin_formulaic_features(market, feat)
    out = {}
    for name, side, hold, stride, terms in VPIN_CANDIDATES:
        active = active_from_terms(feat2, terms)
        before = len(events)
        base.vw.append_active(events, market, feat2, masks, active, name, side, hold, stride)
        out[name] = {"side": side, "hold": hold, "stride": stride, "terms": terms, "events_added": len(events) - before}
    return out


def _clean(w: dict[str, float]) -> dict[str, float]:
    return {k: round(float(v), 6) for k, v in w.items() if v > 1e-12}


def _base_seed_weights() -> list[dict[str, float]]:
    seeds = []
    # Existing live gross-5.75 reference.
    seeds.append({"nonpb30_taker": 0.95, "oi_low": 0.95, "oi_high_sel": 0.95, "bear_rex_short": 2.90})
    # Previous gross<=20 scan top 2.
    seeds.append({"nonpb30_taker": 0.65, "oi_raw": 0.55, "rex_rule": 2.00, "oi_upbit_ratio288_low": 2.25, "bear_rex_short": 0.30, "oi_alt_ratio72_dyn_exit": 0.35})
    seeds.append({"nonpb30_taker": 0.80, "rex_rule": 1.90, "oi_upbit_ratio288_low": 2.90})
    # Keep this bounded: we only need to test whether VPIN improves the known
    # live/top portfolios, not rerun the entire portfolio optimizer.
    return seeds


def _candidate_weights(by, years):
    global _GLOBAL_BY, _GLOBAL_YEARS
    _GLOBAL_BY, _GLOBAL_YEARS = by, years
    sleeves = base.SLEEVES
    vpin_names = [x[0] for x in VPIN_CANDIDATES]
    out = []
    seen = set()

    def add(w):
        ww = base.quantize_portfolio({s: max(0.0, float(w.get(s, 0.0))) for s in sleeves})
        if sum(ww.values()) <= 0 or not base.weight_unit_ok(ww):
            return
        key = tuple(ww[s] for s in sleeves)
        if key not in seen:
            seen.add(key); out.append(ww)

    base_only = _base_seed_weights()
    # Baselines and MDD-boundary rescaled versions.
    for w in base_only:
        add(w); add(base.scale_to_mdd(by, years, w, 19.7)); add(base.scale_to_mdd(by, years, w, 19.95))

    # Inject VPIN sleeves at small and moderate weights into each known good base.
    for w in base_only:
        for vn in vpin_names:
            for vwgt in [0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00, 1.50, 2.00]:
                ww = dict(w); ww[vn] = ww.get(vn, 0.0) + vwgt
                add(ww); add(base.scale_to_mdd(by, years, ww, 19.7)); add(base.scale_to_mdd(by, years, ww, 19.95))

    # Random mixes: force at least one VPIN/formulaic sleeve, let other pool sleeves compete.
    rng = random.Random(160100991)
    pool = [s for s in sleeves if not s.startswith('vpin_') and not s.startswith('formulaic_')]
    for _ in range(500):
        chosen = rng.sample(pool, rng.randint(2, 5)) + rng.sample(vpin_names, rng.randint(1, min(3, len(vpin_names))))
        raw = np.array([rng.random() ** 1.5 for _ in chosen], float); raw /= raw.sum()
        gross = rng.choice([3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0])
        w = {c: float(a * gross) for c, a in zip(chosen, raw)}
        add(w); add(base.scale_to_mdd(by, years, w, 19.8))
    return out


def _vpin_weight(w: dict[str, float]) -> float:
    return sum(float(w.get(x[0], 0.0)) for x in VPIN_CANDIDATES)


def main() -> None:
    _extend_sleeves()
    market, feat, masks, years, events, _ = base.vw.build_events()
    base.add_old_live_events(events, market, feat, masks)
    events.extend(base.dx.build_dynamic_sleeves(market, feat, masks, years))
    vpin_meta = _append_vpin_events(events, market, feat, masks)
    by = base.arrays(events, masks)

    rows = []
    for w in _candidate_weights(by, years):
        st = base.metrics(by, years, w)
        rows.append({
            "weights": _clean(w),
            "gross": round(sum(w.values()), 6),
            "vpin_gross": round(_vpin_weight(w), 6),
            "stats": st,
            "passes_mdd20_ratio5": base.passes(st),
            "score_tuple": base.score(st),
        })
    rows.sort(key=lambda r: r["score_tuple"], reverse=True)
    qualified = [r for r in rows if r["passes_mdd20_ratio5"]]
    qualified_with_vpin = [r for r in qualified if r["vpin_gross"] >= 0.10]
    diagnostic_with_vpin = [r for r in rows if r["vpin_gross"] >= 0.10]

    report = {
        "protocol": {
            "cost_each_side": base.COST,
            "condition": "test2024/eval2025/ytd2026 strict_mdd_pct<=20 and CAGR/MDD>=5; nonzero weights >=0.10 and multiples of 0.05",
            "selection_scope": "existing signal pool + dynamic sleeves + VPIN/formulaic candidates; VPIN thresholds from train<2024 only",
        },
        "sleeves": base.SLEEVES,
        "vpin_candidates": vpin_meta,
        "event_counts": {sp: dict(Counter(e["sleeve"] for e in events if e["split"] == sp)) for sp in masks},
        "evaluated_unique": len(rows),
        "qualified_count": len(qualified),
        "qualified_with_vpin_count": len(qualified_with_vpin),
        "top_qualified": [{k: v for k, v in r.items() if k != "score_tuple"} for r in qualified[:20]],
        "top_qualified_with_vpin": [{k: v for k, v in r.items() if k != "score_tuple"} for r in qualified_with_vpin[:20]],
        "top_diagnostic_with_vpin": [{k: v for k, v in r.items() if k != "score_tuple"} for r in diagnostic_with_vpin[:20]],
        "top_diagnostic_all": [{k: v for k, v in r.items() if k != "score_tuple"} for r in rows[:20]],
    }
    Path(OUT).write_text(json.dumps(report, indent=2, ensure_ascii=False))

    def fmt(s):
        return f"{s['total_return_pct']:.2f}/{s['cagr_pct']:.2f}/{s['strict_mdd_pct']:.2f}/{s['cagr_to_strict_mdd']:.2f}/{s['trade_entries']}"
    md = [
        "# Portfolio scan with VPIN/formulaic sleeves (2026-07-09)", "",
        json.dumps(report["protocol"], ensure_ascii=False), "",
        f"evaluated_unique={len(rows)}, qualified_count={len(qualified)}, qualified_with_vpin_count={len(qualified_with_vpin)}", "",
        "## Top qualified with VPIN", "",
        "| rank | gross | vpin gross | weights | 2024 ret/CAGR/MDD/ratio/trades | 2025 ret/CAGR/MDD/ratio/trades | 2026 ret/CAGR/MDD/ratio/trades |",
        "|---:|---:|---:|---|---:|---:|---:|",
    ]
    for i, r in enumerate(report["top_qualified_with_vpin"][:15], 1):
        st = r["stats"]
        md.append(f"| {i} | {r['gross']:.2f} | {r['vpin_gross']:.2f} | `{r['weights']}` | {fmt(st['test2024'])} | {fmt(st['eval2025'])} | {fmt(st['ytd2026'])} |")
    md += ["", "## Top qualified overall", "", "| rank | gross | vpin gross | weights | 2024 | 2025 | 2026 |", "|---:|---:|---:|---|---:|---:|---:|"]
    for i, r in enumerate(report["top_qualified"][:10], 1):
        st = r["stats"]
        md.append(f"| {i} | {r['gross']:.2f} | {r['vpin_gross']:.2f} | `{r['weights']}` | {fmt(st['test2024'])} | {fmt(st['eval2025'])} | {fmt(st['ytd2026'])} |")
    md += ["", "## Top diagnostic with VPIN even if failed", "", "| rank | pass | gross | vpin gross | weights | 2024 | 2025 | 2026 |", "|---:|---:|---:|---:|---|---:|---:|---:|"]
    for i, r in enumerate(report["top_diagnostic_with_vpin"][:15], 1):
        st = r["stats"]
        md.append(f"| {i} | {r['passes_mdd20_ratio5']} | {r['gross']:.2f} | {r['vpin_gross']:.2f} | `{r['weights']}` | {fmt(st['test2024'])} | {fmt(st['eval2025'])} | {fmt(st['ytd2026'])} |")
    Path(DOC).write_text("\n".join(md) + "\n")
    print(json.dumps({
        "output": OUT, "doc": DOC, "evaluated": len(rows), "qualified": len(qualified),
        "qualified_with_vpin": len(qualified_with_vpin),
        "top_qualified": report["top_qualified"][:3],
        "top_qualified_with_vpin": report["top_qualified_with_vpin"][:3],
        "top_diagnostic_with_vpin": report["top_diagnostic_with_vpin"][:3],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
