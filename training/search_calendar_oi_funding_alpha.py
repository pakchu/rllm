"""Calendar x OI/funding/taker alpha search.

Purpose: search a new-ish BTCUSDT standalone alpha family that is not another
Alpha101/VPIN/wave standalone replay.  The hypotheses are schedule-aware
funding/premium stress, OI unwind/squeeze, and session/day-of-week interaction.

Protocol:
- BTCUSDT target only; external assets are not traded.
- Thresholds are fit on train < 2024 only.
- Splits: train < 2024, test 2024, eval 2025, ytd2026.
- 6bp/side, strict in-position MDD, split-contained/forced exits.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.evaluate_portfolio_llm_selector import _prep
from training.search_vpin_formulaic_alpha import stats, trade_arrays

OUT = "results/calendar_oi_funding_alpha_scan_2026-07-10.json"
DOC = "docs/calendar-oi-funding-alpha-scan-2026-07-10.md"
HOLDS = [24, 48, 72, 96, 144]
STRIDES = [6, 12, 24]


def z(s: pd.Series, n: int) -> pd.Series:
    mu = s.rolling(n, min_periods=max(20, n // 4)).mean()
    sd = s.rolling(n, min_periods=max(20, n // 4)).std(ddof=0)
    return ((s - mu) / sd.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).clip(-8, 8).fillna(0)


def ret(s: pd.Series, n: int) -> pd.Series:
    return (s / s.shift(n).replace(0, np.nan) - 1).replace([np.inf, -np.inf], np.nan).clip(-10, 10).fillna(0)


def add_calendar_features(m: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    f = base.copy()
    d = pd.to_datetime(m["date"])
    c = m["close"].astype(float)
    qv = (m["quote_asset_volume"] if "quote_asset_volume" in m else m["volume"] * m["close"]).astype(float)
    tbq = (m["taker_buy_quote"] if "taker_buy_quote" in m else qv * 0.5).astype(float)
    taker = ((2 * tbq - qv) / qv.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0)
    signed = (2 * tbq - qv).fillna(0)
    cvd = signed.cumsum()
    oi = m.get("open_interest", pd.Series(0.0, index=m.index)).astype(float).replace(0, np.nan).ffill().fillna(0)
    funding = f.get("funding_rate", m.get("funding_rate", pd.Series(0.0, index=m.index))).astype(float).fillna(0)
    prem = f.get("premium_index", m.get("premium_index", pd.Series(0.0, index=m.index))).astype(float).fillna(0)

    hour = d.dt.hour
    dow = d.dt.dayofweek
    minute = d.dt.minute
    bars_since_funding = ((hour % 8) * 12 + (minute // 5)).astype(int)
    bars_to_funding = (96 - bars_since_funding) % 96
    f["cal_pre_funding_1h"] = (bars_to_funding <= 12).astype(float)
    f["cal_post_funding_1h"] = (bars_since_funding <= 12).astype(float)
    f["cal_funding_window_2h"] = ((bars_to_funding <= 24) | (bars_since_funding <= 24)).astype(float)
    f["cal_asia"] = ((hour >= 0) & (hour < 8)).astype(float)
    f["cal_europe"] = ((hour >= 8) & (hour < 16)).astype(float)
    f["cal_us"] = ((hour >= 16) & (hour < 24)).astype(float)
    f["cal_weekend"] = (dow >= 5).astype(float)
    f["cal_monday_utc"] = (dow == 0).astype(float)
    f["cal_friday_utc"] = (dow == 4).astype(float)

    for n in [12, 24, 48, 72, 96, 144, 288]:
        f[f"cf_px_ret_z_{n}"] = z(ret(c, n), max(288, n * 4))
        f[f"cf_qv_z_{n}"] = z(qv, max(288, n * 4))
        f[f"cf_taker_mean_z_{n}"] = z(taker.rolling(n, min_periods=max(6, n // 4)).mean().fillna(0), max(288, n * 4))
        f[f"cf_cvd_ret_z_{n}"] = z(ret(cvd.abs() + 1, n) * np.sign(cvd.diff(n).fillna(0)), max(288, n * 4))
        f[f"cf_oi_ret_z_{n}"] = z(ret(oi, n), max(288, n * 4))
        f[f"cf_oi_minus_px_z_{n}"] = f[f"cf_oi_ret_z_{n}"] - f[f"cf_px_ret_z_{n}"]
        f[f"cf_px_minus_oi_z_{n}"] = f[f"cf_px_ret_z_{n}"] - f[f"cf_oi_ret_z_{n}"]
    f["cf_funding_z"] = z(funding, 288)
    f["cf_premium_z"] = z(prem, 288)
    f["cf_basis_z"] = z(prem - funding, 288)
    f["cf_funding_delta_z"] = z(funding.diff().fillna(0), 288)
    f["cf_premium_delta_z"] = z(prem.diff().fillna(0), 288)
    f["cf_carry_stress_long"] = (-f["cf_px_ret_z_72"]).clip(lower=0) + f["cf_oi_minus_px_z_72"].clip(lower=0) + (-f["cf_funding_z"]).clip(lower=0) + (-f["cf_premium_z"]).clip(lower=0)
    f["cf_carry_stress_short"] = f["cf_px_ret_z_72"].clip(lower=0) + f["cf_oi_minus_px_z_72"].clip(lower=0) + f["cf_funding_z"].clip(lower=0) + f["cf_premium_z"].clip(lower=0)
    f["cf_oi_unwind_long"] = (-f["cf_px_ret_z_48"]).clip(lower=0) + (-f["cf_oi_ret_z_48"]).clip(lower=0) + (-f["cf_taker_mean_z_48"]).clip(lower=0)
    f["cf_oi_squeeze_short"] = f["cf_px_ret_z_48"].clip(lower=0) + f["cf_oi_ret_z_48"].clip(lower=0) + f["cf_taker_mean_z_48"].clip(lower=0)
    f["cf_funding_flip_long"] = (-f["cf_funding_z"]).clip(lower=0) + f["cf_funding_delta_z"].clip(lower=0) + (-f["cf_px_ret_z_24"]).clip(lower=0)
    f["cf_funding_flip_short"] = f["cf_funding_z"].clip(lower=0) + (-f["cf_funding_delta_z"]).clip(lower=0) + f["cf_px_ret_z_24"].clip(lower=0)
    return f.replace([np.inf, -np.inf], np.nan).fillna(0)


def qthr(f: pd.DataFrame, train: np.ndarray, col: str, q: float) -> float | None:
    vals = f.loc[train, col].to_numpy(float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 100 or np.nanstd(vals) < 1e-12:
        return None
    return float(np.quantile(vals, q))


def materialize(f: pd.DataFrame, train: np.ndarray, spec: list[tuple[str, str, float]]) -> list[tuple[str, str, float, float]] | None:
    out = []
    for col, op, qq in spec:
        if col.startswith("cal_"):
            thr = 0.5
        else:
            thr = qthr(f, train, col, qq)
            if thr is None:
                return None
        out.append((col, op, float(thr), float(qq)))
    return out


def active(f: pd.DataFrame, terms: list[tuple[str, str, float, float]]) -> np.ndarray:
    a = np.ones(len(f), dtype=bool)
    for col, op, thr, _ in terms:
        x = f[col].to_numpy(float)
        a &= np.isfinite(x) & ((x >= thr) if op == ">=" else (x <= thr))
    return a


def eval_rule(m: pd.DataFrame, f: pd.DataFrame, masks: dict[str, np.ndarray], years: dict[str, float], terms: list[tuple[str, str, float, float]], side: str, hold: int, stride: int) -> dict[str, Any]:
    act = active(f, terms)
    fac, mn, rr = trade_arrays(m, hold, side)
    smod = (np.arange(len(m)) % int(stride)) == 0
    out = {}
    for sp, mask in masks.items():
        local = []
        nxt = 0
        idx = np.flatnonzero(act & mask & smod)
        idx = idx[(idx >= 300) & (idx < len(m) - hold - 2)]
        for p in idx:
            p = int(p)
            xp = p + 1 + hold
            if p < nxt or xp >= len(m) or not mask[xp] or not np.isfinite(fac[p]):
                continue
            local.append((float(fac[p]), float(mn[p]), float(rr[p])))
            nxt = xp
        out[sp] = stats(local, years[sp])
    return out


def score(st: dict[str, Any]) -> tuple[Any, ...]:
    tr, t, e, y = st["train"], st["test2024"], st["eval2025"], st["ytd2026"]
    alpha_candidate = t["cagr_to_strict_mdd"] >= 2.5 and e["cagr_to_strict_mdd"] >= 2.5 and t["cagr_pct"] > 0 and e["cagr_pct"] > 0
    enough = t["trade_entries"] >= 20 and e["trade_entries"] >= 15
    y_ok = y["cagr_pct"] > 0 and y["trade_entries"] >= 6
    train_not_broken = tr["trade_entries"] >= 40 and tr["strict_mdd_pct"] < 70
    min_oos = min(t["cagr_to_strict_mdd"], e["cagr_to_strict_mdd"])
    ret_score = t["total_return_pct"] + e["total_return_pct"] + 0.4 * y["total_return_pct"]
    return (alpha_candidate and enough and train_not_broken, y_ok, min_oos, y["cagr_to_strict_mdd"], ret_score, t["trade_entries"] + e["trade_entries"])


def name_for(side: str, spec: list[tuple[str, str, float]]) -> str:
    raw = side + "|" + ";".join(f"{c}{o}{q}" for c, o, q in spec)
    return "caloi_" + hashlib.md5(raw.encode()).hexdigest()[:10]


def generate_specs(seed: int = 710, n_random: int = 1200):
    rng = random.Random(seed)
    long_blocks = [
        [("cf_carry_stress_long", ">=", 0.75), ("cal_pre_funding_1h", ">=", 0.5)],
        [("cf_funding_flip_long", ">=", 0.75), ("cal_post_funding_1h", ">=", 0.5)],
        [("cf_oi_unwind_long", ">=", 0.75), ("cf_qv_z_48", ">=", 0.55)],
        [("cf_px_ret_z_72", "<=", 0.25), ("cf_oi_minus_px_z_72", ">=", 0.65), ("cf_premium_z", "<=", 0.45)],
        [("return_zscore_48", "<=", 0.25), ("oi_minus_px_4h_z", ">=", 0.65), ("funding_zscore", "<=", 0.45)],
    ]
    short_blocks = [
        [("cf_carry_stress_short", ">=", 0.75), ("cal_pre_funding_1h", ">=", 0.5)],
        [("cf_funding_flip_short", ">=", 0.75), ("cal_post_funding_1h", ">=", 0.5)],
        [("cf_oi_squeeze_short", ">=", 0.75), ("cf_qv_z_48", ">=", 0.55)],
        [("cf_px_ret_z_72", ">=", 0.75), ("cf_oi_minus_px_z_72", ">=", 0.65), ("cf_premium_z", ">=", 0.55)],
        [("return_zscore_48", ">=", 0.75), ("oi_minus_px_4h_z", ">=", 0.65), ("funding_zscore", ">=", 0.55)],
    ]
    extras = [
        ("cal_asia", ">=", 0.5), ("cal_europe", ">=", 0.5), ("cal_us", ">=", 0.5),
        ("cal_weekend", ">=", 0.5), ("cal_monday_utc", ">=", 0.5), ("cal_friday_utc", ">=", 0.5),
        ("cal_funding_window_2h", ">=", 0.5),
        ("cf_qv_z_72", ">=", 0.55), ("cf_qv_z_72", "<=", 0.60),
        ("cf_cvd_ret_z_72", ">=", 0.60), ("cf_cvd_ret_z_72", "<=", 0.40),
        ("cf_basis_z", ">=", 0.55), ("cf_basis_z", "<=", 0.45),
        ("premium_index_zscore", ">=", 0.55), ("premium_index_zscore", "<=", 0.45),
    ]
    seen = set()

    def jitter(t: tuple[str, str, float]) -> tuple[str, str, float]:
        c, o, q = t
        if c.startswith("cal_"):
            return t
        return (c, o, max(0.05, min(0.95, q + rng.choice([-0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15]))))

    for side, blocks in [("long", long_blocks), ("short", short_blocks)]:
        for block in blocks:
            for _ in range(60):
                spec = [jitter(x) for x in block] + [jitter(x) for x in rng.sample(extras, rng.randint(0, 2))]
                out, used = [], set()
                for x in spec:
                    if x[0] not in used:
                        out.append(x); used.add(x[0])
                key = (side, tuple(out))
                if key not in seen:
                    seen.add(key); yield side, out
    for _ in range(n_random):
        side = rng.choice(["long", "short"])
        blocks = long_blocks if side == "long" else short_blocks
        spec = []
        for block in rng.sample(blocks, rng.randint(1, 2)):
            spec.extend(jitter(x) for x in block)
        spec.extend(jitter(x) for x in rng.sample(extras, rng.randint(0, 2)))
        out, used = [], set()
        for x in spec:
            if x[0] not in used:
                out.append(x); used.add(x[0])
        if 2 <= len(out) <= 6:
            key = (side, tuple(out))
            if key not in seen:
                seen.add(key); yield side, out


def main() -> None:
    m, base, masks, years = _prep()
    f = add_calendar_features(m, base)
    rows = []
    tested = 0
    train = masks["train"]
    for side, spec in generate_specs():
        terms = materialize(f, train, spec)
        if terms is None:
            continue
        act = active(f, terms)
        ar = float(act[train].mean())
        if ar < 0.001 or ar > 0.25:
            continue
        tested += 1
        for hold in HOLDS:
            for stride in STRIDES:
                st = eval_rule(m, f, masks, years, terms, side, hold, stride)
                rows.append({
                    "name": name_for(side, spec),
                    "side": side,
                    "active_rate_train": ar,
                    "terms": [{"feature": c, "op": op, "threshold": thr, "train_q": q} for c, op, thr, q in terms],
                    "hold": hold,
                    "stride": stride,
                    "stats": st,
                    "score_tuple": score(st),
                })
    rows.sort(key=lambda r: r["score_tuple"], reverse=True)
    top = [{k: v for k, v in r.items() if k != "score_tuple"} for r in rows[:250]]
    report = {
        "protocol": __doc__,
        "tested_specs": tested,
        "all_count": len(rows),
        "top": top,
    }
    Path(OUT).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    def fmt(s: dict[str, Any]) -> str:
        return f"{s['total_return_pct']:.2f}/{s['cagr_pct']:.2f}/{s['strict_mdd_pct']:.2f}/{s['cagr_to_strict_mdd']:.2f}/{s['trade_entries']}/{s['win_rate']:.2f}/{s['bar_sharpe_like']:.2f}"

    md = [
        "# Calendar OI/funding alpha scan (2026-07-10)",
        "",
        "BTCUSDT standalone scan over schedule-aware funding/premium stress, OI unwind/squeeze, and session/day interactions.",
        "Thresholds are train<2024 quantiles; cost 6bp/side; strict MDD includes in-position adverse excursion; split-end forced close.",
        "",
        f"tested_specs={tested}, all_count={len(rows)}",
        "",
        "| rank | name | side | active | hold/stride | train ret/CAGR/MDD/ratio/trades/win/sharpe | 2024 | 2025 | 2026 | terms |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for i, r in enumerate(top[:100], 1):
        st = r["stats"]
        terms = "; ".join(f"{t['feature']} {t['op']} q{t['train_q']:.2f}({t['threshold']:.4g})" for t in r["terms"])
        md.append(f"| {i} | {r['name']} | {r['side']} | {r['active_rate_train']:.3f} | {r['hold']}/{r['stride']} | {fmt(st['train'])} | {fmt(st['test2024'])} | {fmt(st['eval2025'])} | {fmt(st['ytd2026'])} | `{terms}` |")
    Path(DOC).write_text("\n".join(md) + "\n")
    print(json.dumps({"output": OUT, "doc": DOC, "tested_specs": tested, "all_count": len(rows), "top": top[:20]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
