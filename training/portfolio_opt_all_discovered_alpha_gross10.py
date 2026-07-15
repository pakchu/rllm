"""Portfolio optimization over legacy sleeves plus every discovered alpha candidate.

This is the broad follow-up scan requested after adding new alpha candidates:
- legacy REX/OI/dynamic sleeves from the established portfolio search
- fixed new alpha pool from 2026-07-10
- all 2026-07-12 alpha_pool_qualifiers/candidates that can be causally replayed
- all Calendar/OI/funding top candidates and REX-veto top/tte candidates

Selection protocol remains conservative: fit/rank weights on 2024 test only, and
report 2025/2026 as diagnostics.  The alpha universe itself is research-contaminated
because candidates were discovered during later-window exploration.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import training.portfolio_opt_combined_rex_new_alpha as base
import training.portfolio_opt_new_alpha_pool as na
from preprocessing.market_features import build_market_feature_frame
from training.build_rex_event_reasoning_policy_data import _build_light_rex_features
from training.long_regime_interest_gate_validation import build_interest_features
from training.search_bidirectional_state_alpha import extra as state_extra, mk
from training.search_calendar_oi_funding_alpha import add_calendar_features
from training.evaluate_portfolio_llm_selector import _prep as selector_prep
from training.search_kimchi_leadlag_bidirectional_alpha import features as kimchi_features
from training.search_jump_variation_bidirectional_alpha import features as jump_features
from training.search_liquidity_recovery_bidirectional_alpha import features as liquidity_features
from training.search_volume_clock_bidirectional_alpha import features as vc_features
from training.search_lowcorr_macro_alpha import mask as macro_mask
from training.search_path_memory_bidirectional_alpha import features as path_features

OUT = "results/portfolio_all_discovered_alpha_gross10_min025_step005_2026-07-12.json"
DOC = "docs/portfolio-all-discovered-alpha-gross10-min025-step005-2026-07-12.md"

SCAN_FILES = {
    "state": "results/bidirectional_state_alpha_scan_2026-07-12.json",
    "path": "results/bidirectional_path_gate_alpha_scan_2026-07-12.json",
    "kimchi": "results/bidirectional_kimchi_gate_alpha_scan_2026-07-12.json",
    "macro_long": "results/lowcorr_macro_alpha_scan_2026-07-12.json",
    "macro_short": "results/lowcorr_macro_short_alpha_scan_2026-07-12.json",
    "jump": "results/jump_variation_bidirectional_alpha_scan_2026-07-12.json",
    "jump_volume": "results/jump_volume_clock_gate_alpha_scan_2026-07-12.json",
    "jump_same": "results/jump_volume_clock_same_direction_candidate_2026-07-12.json",
    "liquidity": "results/liquidity_recovery_bidirectional_alpha_scan_2026-07-12.json",
    "calendar": "results/calendar_oi_funding_alpha_scan_2026-07-10.json",
    "rex_veto": "results/rex_failure_veto_alpha_scan_2026-07-12.json",
}

EXTRA_SLEEVES: list[str] = []


def _add_sleeve(name: str) -> str:
    safe = "cand_" + "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)[:80]
    if safe not in base.SLEEVES:
        base.SLEEVES.append(safe)
    if safe not in EXTRA_SLEEVES:
        EXTRA_SLEEVES.append(safe)
    return safe


@dataclass(frozen=True)
class Config(base.CombinedOptConfig):
    output: str = OUT
    docs_output: str = DOC
    gross_cap: float = 10.0
    random_samples: int = 18000
    seed: int = 71210
    min_nonzero_weight: float = 0.25
    weight_step: float = 0.05
    min_test_trades: int = 80
    selection_mdd_cap: float = 20.0
    train_mdd_cap: float = 20.0
    oos_mdd_cap: float = 20.0
    candidate_calendar_top_n: int = 250
    candidate_rex_top_n: int = 50
    family_gross_cap: float = 2.0


def sleeve_family(name: str) -> str:
    if name.startswith("cand_calendar_"):
        return "calendar"
    if name.startswith("cand_rex_veto_"):
        return "rex_veto"
    if name.startswith("cand_jump_") or name.startswith("cand_jump_volume_") or name.startswith("cand_jump_same_"):
        return "jump"
    if name.startswith("cand_kimchi_"):
        return "kimchi"
    if name.startswith("cand_path_") or name.startswith("cand_state_"):
        return "state_path"
    if name.startswith("cand_macro_"):
        return "macro"
    if name.startswith("new_"):
        return "new"
    if name.startswith("cand_"):
        return "other_candidate"
    return "legacy"


def exact_duplicate_map(by: dict[str, Any]) -> tuple[dict[str, str], list[list[str]]]:
    """Map exact return/adverse paths to one canonical sleeve."""
    import hashlib
    buckets: dict[str, list[str]] = {}
    for i, sleeve in enumerate(base.SLEEVES):
        h = hashlib.sha256()
        for split in ("train", "test2024", "eval2025", "ytd2026"):
            for key in ("R", "A"):
                values = np.round(np.nan_to_num(by[split][key][i], nan=0.0, posinf=0.0, neginf=0.0), 12)
                h.update(values.tobytes())
        buckets.setdefault(h.hexdigest(), []).append(sleeve)
    groups = [names for names in buckets.values() if len(names) > 1]
    canonical = {name: names[0] for names in groups for name in names}
    return canonical, groups


def load_json(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def conds(row: dict[str, Any], key: str) -> list[tuple[str, str, float]]:
    out = []
    for x in row.get(key, []) or []:
        op = str(x["op"]).replace(">=", "ge").replace("<=", "le")
        out.append((str(x["feature"]), op, float(x["threshold"])))
    return out


def append_policy(events: list[dict[str, Any]], market: pd.DataFrame, masks: dict[str, np.ndarray], name: str,
                  long_a: np.ndarray, short_a: np.ndarray, hold: int, stride: int,
                  tp: float | None, sl: float | None, cost: float, lev: float = 0.5) -> int:
    """Append one aggregated event per split/sleeve instead of one full array per trade."""
    dates = pd.to_datetime(market.date)
    positions = np.arange(143, max(0, len(market) - int(hold) - 2), int(stride), dtype=np.int64)
    added_total = 0
    n = len(market)
    for split, sm in masks.items():
        ret_sum = np.zeros(n, dtype=np.float64)
        adv_sum = np.zeros(n, dtype=np.float64)
        added = wins = 0
        first_pos = None
        first_date = None
        nxt = 0
        active = sm[positions] & (long_a[positions] | short_a[positions])
        for ip0 in positions[active]:
            ip = int(ip0)
            if ip < nxt:
                continue
            if long_a[ip] and not short_a[ip]:
                side = "long"
            elif short_a[ip] and not long_a[ip]:
                side = "short"
            else:
                continue
            ep = na._event_path(market, ip, side=side, hold=int(hold), cost_rate=float(cost), tp=tp, sl=sl, entry_delay=1, leverage=lev)
            if ep is None:
                continue
            ret, adv, realized = ep
            nz = np.flatnonzero(np.abs(ret) > 0)
            xp = int(nz[-1]) if len(nz) else ip + 1 + int(hold)
            if xp >= len(sm) or not sm[xp]:
                continue
            ret_sum += ret
            adv_sum += adv
            added += 1
            wins += int(float(realized) > 0.0)
            added_total += 1
            if first_pos is None:
                first_pos = ip; first_date = str(dates.iloc[ip])
            nxt = xp + 1
        if added:
            events.append({"split": split, "sleeve": name, "side": "mixed", "signal_pos": int(first_pos or 0), "date": first_date or "", "ret_bps": 0.0, "ret": ret_sum, "adv": adv_sum, "trade_count": added, "win_count": wins})
    return added_total

def _gate_mask(f: pd.DataFrame, gate: dict[str, Any], mode: str, base_long: np.ndarray, base_short: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c = gate["feature"]
    if gate.get("mode") == "same_direction":
        la = base_long & mk(f, [(c, "ge", float(gate["long_threshold"]))])
        sa = base_short & mk(f, [(c, "le", float(gate["short_threshold"]))])
        return la, sa
    op = str(gate.get("op", "ge"))
    thr = float(gate["threshold"])
    g = mk(f, [(c, op, thr)])
    gm = str(gate.get("mode", mode))
    la = base_long & (g if gm in ("long", "both") else True)
    sa = base_short & (g if gm in ("short", "both") else True)
    return la, sa


def add_qualifier_candidates(events: list[dict[str, Any]], market: pd.DataFrame, masks: dict[str, np.ndarray], cfg: Config) -> dict[str, int]:
    counts: dict[str, int] = {}
    base_feat = build_market_feature_frame(market, window_size=144)
    common = pd.concat([base_feat, build_interest_features(market, base_feat)], axis=1).loc[:, lambda x: ~x.columns.duplicated(keep="last")]
    f0 = state_extra(market, common.copy())
    state = load_json(SCAN_FILES["state"]).get("alpha_pool_qualifiers", [])[0]
    bl = mk(f0, conds(state, "long_conditions")); bs = mk(f0, conds(state, "short_conditions"))
    # Prior latest sleeves are covered below by replaying all qualifier files with the low-memory append_policy.

    def add(name: str, la: np.ndarray, sa: np.ndarray, row: dict[str, Any], hold_key="hold_bars", stride_key="stride_bars") -> None:
        slv = _add_sleeve(name)
        counts[slv] = append_policy(events, market, masks, slv, la, sa, int(row[hold_key]), int(row[stride_key]), row.get("tp"), row.get("sl"), cfg.cost_rate)

    add("state_funding_relief_vs_fx_stress", bl, bs, state)

    pf = path_features(market, f0.copy())
    for i, row in enumerate(load_json(SCAN_FILES["path"]).get("alpha_pool_qualifiers", [])):
        la, sa = _gate_mask(pf, row["gate"], "both", bl, bs)
        add(f"path_gate_{i}", la, sa, state)

    kf = kimchi_features(market, f0.copy())
    for i, row in enumerate(load_json(SCAN_FILES["kimchi"]).get("alpha_pool_qualifiers", [])):
        la, sa = _gate_mask(kf, row["gate"], "both", bl, bs)
        add(f"kimchi_gate_{i}", la, sa, state)

    for i, row in enumerate(load_json(SCAN_FILES["macro_long"]).get("alpha_pool_qualifiers", [])):
        a = macro_mask(common, [(x["feature"], x["op"], float(x["threshold"])) for x in row["conditions"]])
        add(f"macro_long_{i}_{row.get('name','')}", a, np.zeros(len(market), bool), row)
    for i, row in enumerate(load_json(SCAN_FILES["macro_short"]).get("alpha_pool_qualifiers", [])):
        a = mk(f0, conds(row, "conditions"))
        add(f"macro_short_{i}_{row.get('name','')}", np.zeros(len(market), bool), a, row)

    jf = jump_features(market, common.copy())
    for i, row in enumerate(load_json(SCAN_FILES["jump"]).get("alpha_pool_qualifiers", [])):
        add(f"jump_{i}_{row.get('name','')}", mk(jf, conds(row, "long_conditions")), mk(jf, conds(row, "short_conditions")), row)
    vf = vc_features(market, jf.copy())
    jump0 = load_json(SCAN_FILES["jump"]).get("alpha_pool_qualifiers", [])[0]
    jbl = mk(vf, conds(jump0, "long_conditions")); jbs = mk(vf, conds(jump0, "short_conditions"))
    for i, row in enumerate(load_json(SCAN_FILES["jump_volume"]).get("alpha_pool_qualifiers", [])):
        la, sa = _gate_mask(vf, row["gate"], "both", jbl, jbs)
        add(f"jump_volume_gate_{i}", la, sa, jump0)
    for i, row in enumerate(load_json(SCAN_FILES["jump_same"]).get("alpha_pool_qualifiers", [])):
        add(f"jump_same_{i}_{row.get('name','')}", mk(vf, conds(row, "long_conditions")), mk(vf, conds(row, "short_conditions")), {**row, "hold_bars": 96, "stride_bars": 6, "tp": .015, "sl": .01})

    lf = liquidity_features(market, f0.copy())
    for i, row in enumerate(load_json(SCAN_FILES["liquidity"]).get("alpha_pool_qualifiers", [])):
        add(f"liquidity_{i}_{row.get('name','')}", mk(lf, conds(row, "long_conditions")), mk(lf, conds(row, "short_conditions")), row)
    return counts


def add_calendar_candidates(events: list[dict[str, Any]], market: pd.DataFrame, masks: dict[str, np.ndarray], cfg: Config) -> dict[str, int]:
    report = load_json(SCAN_FILES["calendar"])
    rows = report.get("top", [])[: int(cfg.candidate_calendar_top_n)]
    # Calendar scan was built on evaluate_portfolio_llm_selector._prep()/oi_feature_frame; use the same base to avoid feature drift.
    prep_market, prep_feat, _, _ = selector_prep()
    if len(prep_market) != len(market):
        raise RuntimeError(f"calendar prep market length mismatch: {len(prep_market)} != {len(market)}")
    f = add_calendar_features(market, prep_feat)
    counts: dict[str, int] = {}
    for i, row in enumerate(rows):
        active = np.ones(len(f), bool)
        for t in row["terms"]:
            x = f[t["feature"]].to_numpy(float)
            active &= np.isfinite(x) & ((x >= float(t["threshold"])) if t["op"] == ">=" else (x <= float(t["threshold"])))
        side = str(row["side"])
        slv = _add_sleeve(f"calendar_{i}_{row.get('name','')}_{side}_h{row['hold']}_s{row['stride']}")
        la = active if side == "long" else np.zeros(len(market), bool)
        sa = active if side == "short" else np.zeros(len(market), bool)
        counts[slv] = append_policy(events, market, masks, slv, la, sa, int(row["hold"]), int(row["stride"]), None, None, cfg.cost_rate, lev=0.5)
    return counts


def _rex_row_matches(gates: list[dict[str, Any]], feat: pd.DataFrame, src: dict[str, Any]) -> bool:
    pos = int(src.get("signal_pos", -1))
    if pos < 0 or pos >= len(feat):
        return False
    toks = {f"tok:{k}": str(v) for k, v in (src.get("state_tokens") or {}).items()}
    for g in gates:
        name = str(g["feature"])
        if name.startswith("tok:"):
            if toks.get(name, "") != str(g["threshold"]):
                return False
            continue
        if name not in feat.columns:
            return False
        x = float(feat.iloc[pos][name])
        if not np.isfinite(x):
            return False
        thr = float(g["threshold"])
        if str(g["op"]) == ">=" and not (x >= thr):
            return False
        if str(g["op"]) == "<=" and not (x <= thr):
            return False
    return True


def add_rex_veto_candidates(events: list[dict[str, Any]], market: pd.DataFrame, masks: dict[str, np.ndarray], cfg: Config) -> dict[str, int]:
    report = load_json(SCAN_FILES["rex_veto"])
    rows: list[dict[str, Any]] = []
    seen = set()
    for bucket in ["top", "tte_top"]:
        for r in report.get(bucket, [])[: int(cfg.candidate_rex_top_n)]:
            key = json.dumps(r.get("gates", []), sort_keys=True)
            if key not in seen:
                seen.add(key); rows.append(r)
    src_path = Path("data/rex_event_reasoning_policy_sft_20260712.jsonl")
    src = [json.loads(line) for line in src_path.read_text().splitlines() if line.strip()]
    feat = _build_light_rex_features(market)
    dates = pd.to_datetime(market.date)
    counts: dict[str, int] = {}
    n = len(market)
    for i, row in enumerate(rows):
        slv = _add_sleeve(f"rex_veto_{i}")
        total = 0
        for split, sm in masks.items():
            ret_sum = np.zeros(n, dtype=np.float64)
            adv_sum = np.zeros(n, dtype=np.float64)
            added = wins = 0
            first_pos = None; first_date = None
            nxt = 0
            for r in src:
                ip = int(r.get("signal_pos", -1))
                if ip < 0 or ip >= len(market) or not sm[ip] or ip < nxt:
                    continue
                base_side = str((r.get("base_event") or {}).get("base_side", "")).lower()
                if base_side not in ("long", "short") or not _rex_row_matches(row.get("gates", []), feat, r):
                    continue
                ep = na._event_path(market, ip, side=base_side, hold=144, cost_rate=cfg.cost_rate, tp=None, sl=None, entry_delay=1, leverage=0.5)
                if ep is None:
                    continue
                ret, adv, realized = ep
                xp = min(len(market) - 1, ip + 1 + 144)
                if not sm[xp]:
                    continue
                ret_sum += ret; adv_sum += adv
                added += 1; wins += int(float(realized) > 0.0); total += 1
                if first_pos is None:
                    first_pos = ip; first_date = str(dates.iloc[ip])
                nxt = xp + 1
            if added:
                events.append({"split": split, "sleeve": slv, "side": "mixed", "signal_pos": int(first_pos or 0), "date": first_date or "", "ret_bps": 0.0, "ret": ret_sum, "adv": adv_sum, "trade_count": added, "win_count": wins})
        counts[slv] = total
    return counts


def arrays_agg(events: list[dict[str, Any]], masks: dict[str, np.ndarray]) -> dict[str, Any]:
    starts, ends = base._split_starts_ends(masks)
    by: dict[str, Any] = {}
    for sp in masks:
        ln = ends[sp] - starts[sp]
        mats_r: list[np.ndarray] = []
        mats_a: list[np.ndarray] = []
        counts: list[int] = []
        wins: list[int] = []
        side_counts: list[dict[str, int]] = []
        ev_sp = [e for e in events if e["split"] == sp]
        for sl in base.SLEEVES:
            r = np.zeros(ln, dtype=np.float64)
            a = np.zeros(ln, dtype=np.float64)
            c = 0
            w = 0
            sc: Counter[str] = Counter()
            for e in ev_sp:
                if e["sleeve"] != sl:
                    continue
                st, en = starts[sp], ends[sp]
                r += e["ret"][st:en]
                a += e["adv"][st:en]
                c += int(e.get("trade_count", 1))
                w += int(e.get("win_count", 1 if float(e.get("ret_bps", 0.0)) > 0.0 else 0))
                sc[str(e.get("side", ""))] += int(e.get("trade_count", 1))
            mats_r.append(r); mats_a.append(a); counts.append(c); wins.append(w); side_counts.append(dict(sc))
        R = np.vstack(mats_r)
        A = np.vstack(mats_a)
        active = np.any((R != 0.0) | (A != 0.0), axis=0)
        by[sp] = {"R": R[:, active], "A": A[:, active], "counts": np.array(counts), "wins": np.array(wins), "side_counts": side_counts, "active_bars": int(active.sum())}
    return by


def metric_fast(d: dict[str, Any], years: float, weights: dict[str, float]) -> dict[str, Any]:
    idx = [i for i, name in enumerate(base.SLEEVES) if weights.get(name, 0.0) > 0]
    if idx:
        r = np.zeros(d["R"].shape[1], dtype=np.float64)
        adv = np.zeros(d["A"].shape[1], dtype=np.float64)
        for i in idx:
            w = float(weights.get(base.SLEEVES[i], 0.0))
            r += w * d["R"][i]
            adv += w * d["A"][i]
    else:
        r = np.zeros(0, dtype=np.float64); adv = np.zeros(0, dtype=np.float64)
    if len(r):
        fac = np.maximum(0.0, 1.0 + r)
        eqp = np.cumprod(fac)
        eqb = np.r_[1.0, eqp[:-1]]
        pka = np.maximum.accumulate(eqp)
        pkb = np.maximum.accumulate(eqb)
        dd_after = float(np.nanmax(1.0 - eqp / np.maximum(pka, 1e-12)))
        dd_adv = float(np.nanmax(1.0 - (eqb * np.maximum(0.0, 1.0 + adv)) / np.maximum(pkb, 1e-12)))
        eq = float(eqp[-1])
        mdd = max(dd_after, dd_adv) * 100.0
    else:
        eq = 1.0; mdd = 0.0
    vals = r[np.abs(r) > 1e-12]
    sharpe = float(vals.mean() / vals.std(ddof=1) * np.sqrt(len(vals))) if len(vals) > 1 and vals.std(ddof=1) > 0 else 0.0
    ret_pct = (eq - 1.0) * 100.0
    cagr = ((eq ** (1.0 / years) - 1.0) * 100.0) if eq > 0 else -100.0
    counts = d["counts"]
    wins_arr = d["wins"]
    trades = int(np.sum(counts[idx])) if idx else 0
    wins = int(np.sum(wins_arr[idx])) if idx else 0
    return {
        "total_return_pct": ret_pct,
        "cagr_pct": cagr,
        "strict_mdd_pct": mdd,
        "cagr_to_strict_mdd": cagr / mdd if mdd > 1e-12 else 0.0,
        "trade_entries": trades,
        "win_rate": wins / trades if trades else 0.0,
        "active_bars": int(d["active_bars"]),
        "bar_sharpe_like": sharpe,
        "sleeve_trade_counts": {base.SLEEVES[i]: int(counts[i]) for i in idx},
    }


def metrics_fast(by: dict[str, Any], years: dict[str, float], weights: dict[str, float]) -> dict[str, Any]:
    return {sp: metric_fast(by[sp], years[sp], weights) for sp in ["train", "test2024", "eval2025", "ytd2026"]}

def candidates(by: dict[str, Any], years: dict[str, float], cfg: Config, canonical_map: dict[str, str] | None = None) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    seen: set[tuple[float, ...]] = set()

    def add(w: dict[str, float]) -> None:
        merged: dict[str, float] = {}
        for sleeve, weight in w.items():
            canonical = (canonical_map or {}).get(sleeve, sleeve)
            merged[canonical] = merged.get(canonical, 0.0) + max(0.0, float(weight))
        q = base._discretize_weights(merged, min_nonzero=cfg.min_nonzero_weight, step=cfg.weight_step)
        gross = sum(q.values())
        if not q or gross > cfg.gross_cap + 1e-9:
            return
        family_gross: dict[str, float] = {}
        for sleeve, weight in q.items():
            family = sleeve_family(sleeve)
            family_gross[family] = family_gross.get(family, 0.0) + float(weight)
        if any(weight > cfg.family_gross_cap + 1e-9 for weight in family_gross.values()):
            return
        key = tuple(round(q.get(s, 0.0), 4) for s in base.SLEEVES)
        if key not in seen:
            seen.add(key); out.append(q)

    # Low-cost prior seeds without expensive MDD rescaling.
    for path in [
        "configs/live/portfolio_gross575_no_llm_7sleeve_minratio_2026-07-08.json",
        "configs/live/portfolio_gross610_dynamic_top1_2026-07-08.json",
        "configs/live/portfolio_gross4_no_llm_ratio5_mdd20_2026-07-08.json",
        "configs/live/portfolio_gross6_mdd20_ratio5_return_best_candidate.json",
    ]:
        w = base._load_weights(path)
        if w:
            add(w)
            for k in (0.5, 0.75, 1.0, 1.25, 1.5):
                add(base._scale_to_cap(base._scale(w, k), cfg.gross_cap))
    for i in range(20):
        w = base._load_prior_top("results/portfolio_gross6_cost6bp_mdd20_with_dynamic_2026-07-08.json", idx=i)
        if w:
            add(w)
    for i in range(20):
        w = base._new_selected_seed("top_selected_test2024", i)
        if w:
            add(w)

    rng = random.Random(cfg.seed)
    legacy_anchor = {"nonpb30_taker": 1.0, "oi_high_sel": 1.0, "bear_rex_short": 1.5, "rex_rule": 1.0}
    useful = [s for s in base.SLEEVES if by["test2024"]["counts"][base.SLEEVES.index(s)] > 0]
    for s in EXTRA_SLEEVES:
        for w in (0.25, 0.5, 1.0, 2.0, 3.0):
            add({**legacy_anchor, s: w})
            add({s: w})
    curated = [s for s in useful if s in base.LEGACY_SLEEVES or s.startswith('new_') or s in EXTRA_SLEEVES]
    for _ in range(int(cfg.random_samples)):
        pool = curated if len(curated) <= 90 else rng.sample(curated, 90)
        k = rng.randint(3, min(10, len(pool)))
        chosen = rng.sample(pool, k)
        raw = np.array([rng.random() ** 1.35 for _ in chosen], dtype=float)
        raw /= raw.sum()
        gross = rng.choice([3, 4, 5, 6, 7, 8, 9, 10])
        add({s: float(v * gross) for s, v in zip(chosen, raw)})
    return out



def score_train_sane(st: dict[str, Any], cfg: Config) -> tuple[Any, ...]:
    oos_splits = ["test2024", "eval2025", "ytd2026"]
    ok = (
        st["train"]["total_return_pct"] > 0
        and st["train"]["strict_mdd_pct"] <= cfg.train_mdd_cap
        and all(st[x]["total_return_pct"] > 0 and st[x]["strict_mdd_pct"] <= cfg.oos_mdd_cap for x in oos_splits)
    )
    oos_ok = all(st[x]["cagr_to_strict_mdd"] >= 3.0 for x in ["test2024", "eval2025", "ytd2026"])
    train_ratio = st["train"]["cagr_to_strict_mdd"]
    min_oos = min(st[x]["cagr_to_strict_mdd"] for x in ["test2024", "eval2025", "ytd2026"])
    max_mdd = max(st[x]["strict_mdd_pct"] for x in ["train", *oos_splits])
    ret_sum = sum(st[x]["total_return_pct"] for x in ["train", *oos_splits])
    trades = min(st[x]["trade_entries"] for x in ["test2024", "eval2025", "ytd2026"])
    return (ok and oos_ok, train_ratio, min_oos, ret_sum, -max_mdd, trades)

def run(cfg: Config) -> dict[str, Any]:
    market, feat, masks, years, events, meta = base.build_combined_events(cfg)
    meta["prior_latest_counts"] = add_qualifier_candidates(events, market, masks, cfg)
    meta["calendar_candidate_counts"] = add_calendar_candidates(events, market, masks, cfg)
    meta["rex_veto_candidate_counts"] = add_rex_veto_candidates(events, market, masks, cfg)
    events.sort(key=lambda e: (str(e["split"]), int(e["signal_pos"]), str(e["sleeve"])))
    by = arrays_agg(events, masks)
    canonical_map, duplicate_groups = exact_duplicate_map(by)
    rows = []
    for w in candidates(by, years, cfg, canonical_map):
        st = metrics_fast(by, years, w)
        rows.append({"weights": base._clean(w), "gross": round(sum(w.values()), 6), "stats": st, "score_test_only": base._score_test_only(st, cfg), "score_robust_diag": base._score_robust_diag(st, cfg), "score_train_sane": score_train_sane(st, cfg)})
    selected = sorted(rows, key=lambda r: r["score_test_only"], reverse=True)
    robust = sorted(rows, key=lambda r: r["score_robust_diag"], reverse=True)
    train_sane = sorted(rows, key=lambda r: r["score_train_sane"], reverse=True)
    strip = lambda r: {k: v for k, v in r.items() if not k.startswith("score_")}
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "input": {"rows": len(market), "start": str(market.date.iloc[0]), "end": str(market.date.iloc[-1])},
        "selection_protocol": f"Weights ranked on test2024 only; eval2025/ytd2026 report-only. Train-sane diagnostic uses train strict MDD<={cfg.train_mdd_cap:g}% and each OOS strict MDD<={cfg.oos_mdd_cap:g}%. Gross<=10, nonzero weight>=0.25, step=0.05, cost=6bp/side, strict MDD includes adverse excursion.",
        "contamination_caveat": "This is a broad research portfolio: weight selection is test2024-only, but the alpha universe includes candidates discovered while examining later windows. Treat 2025/2026 as diagnostics, not pristine final eval.",
        "sleeves": base.SLEEVES,
        "extra_sleeves": EXTRA_SLEEVES,
        "event_counts": {sp: dict(Counter(e["sleeve"] for e in events if e["split"] == sp)) for sp in masks},
        "evaluated": len(rows),
        "build_meta": meta,
        "exact_duplicate_groups": duplicate_groups,
        "duplicate_aliases_removed": sum(len(group) - 1 for group in duplicate_groups),
        "top_selected_test2024": [strip(r) for r in selected[:100]],
        "top_robust_diagnostic": [strip(r) for r in robust[:100]],
        "top_train_sane": [strip(r) for r in train_sane[:100]],
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    write_doc(cfg, report)
    return report


def fmt(x: dict[str, Any]) -> str:
    return f"{x['total_return_pct']:.2f}/{x['cagr_pct']:.2f}/{x['strict_mdd_pct']:.2f}/{x['cagr_to_strict_mdd']:.2f}/{x['trade_entries']}"


def write_doc(cfg: Config, o: dict[str, Any]) -> None:
    lines = [
        "# All discovered alpha portfolio — gross 10 / min 0.25 / step 0.05 (2026-07-12)",
        "",
        o["selection_protocol"],
        "",
        o["contamination_caveat"],
        "",
        f"Evaluated: {o['evaluated']}; sleeves={len(o['sleeves'])}; extra_candidate_sleeves={len(o['extra_sleeves'])}.",
        "Metric: `absolute return / full-window CAGR / strict MDD / CAGR-MDD / trades`.",
        "",
        "## Top selected by 2024 test only",
        "|#|gross|weights|train|test2024|eval2025|2026YTD|",
        "|---:|---:|---|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(o["top_selected_test2024"][:25], 1):
        s = r["stats"]
        lines.append(f"|{i}|{r['gross']:.2f}|`{r['weights']}`|{fmt(s['train'])}|{fmt(s['test2024'])}|{fmt(s['eval2025'])}|{fmt(s['ytd2026'])}|")
    lines += ["", "## Robust diagnostic only (eval-influenced)", "|#|gross|weights|test2024|eval2025|2026YTD|", "|---:|---:|---|---:|---:|---:|"]
    for i, r in enumerate(o["top_robust_diagnostic"][:25], 1):
        s = r["stats"]
        lines.append(f"|{i}|{r['gross']:.2f}|`{r['weights']}`|{fmt(s['test2024'])}|{fmt(s['eval2025'])}|{fmt(s['ytd2026'])}|")
    lines += ["", "## Train-sane diagnostic", f"Requires train strict MDD<={cfg.train_mdd_cap:g}% and each OOS strict MDD<={cfg.oos_mdd_cap:g}%; ranks passing rows by train CAGR/MDD first. Cell format is unchanged.", "|#|gross|weights|train|test2024|eval2025|2026YTD|", "|---:|---:|---|---:|---:|---:|---:|"]
    for i, r in enumerate(o.get("top_train_sane", [])[:25], 1):
        s = r["stats"]
        lines.append(f"|{i}|{r['gross']:.2f}|`{r['weights']}`|{fmt(s['train'])}|{fmt(s['test2024'])}|{fmt(s['eval2025'])}|{fmt(s['ytd2026'])}|")
    lines += ["", "## Candidate coverage", "```json", json.dumps({"extra_sleeves": len(o["extra_sleeves"]), "event_counts_summary": {k: len(v) for k, v in o["event_counts"].items()}}, indent=2, ensure_ascii=False), "```"]
    Path(cfg.docs_output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.docs_output).write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default=Config.output)
    p.add_argument("--docs-output", default=Config.docs_output)
    p.add_argument("--gross-cap", type=float, default=10.0)
    p.add_argument("--random-samples", type=int, default=18000)
    p.add_argument("--seed", type=int, default=71210)
    p.add_argument("--min-nonzero-weight", type=float, default=.25)
    p.add_argument("--weight-step", type=float, default=.05)
    p.add_argument("--candidate-calendar-top-n", type=int, default=250)
    p.add_argument("--candidate-rex-top-n", type=int, default=50)
    p.add_argument("--train-mdd-cap", type=float, default=20.0)
    p.add_argument("--oos-mdd-cap", type=float, default=20.0)
    p.add_argument("--family-gross-cap", type=float, default=2.0)
    a = p.parse_args()
    o = run(Config(**vars(a)))
    print(json.dumps({"output": o["config"]["output"], "docs_output": o["config"]["docs_output"], "evaluated": o["evaluated"], "sleeves": len(o["sleeves"]), "extra_sleeves": len(o["extra_sleeves"]), "top_selected_test2024": o["top_selected_test2024"][:5], "top_robust_diagnostic": o["top_robust_diagnostic"][:5]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
