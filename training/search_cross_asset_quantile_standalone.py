"""Cross-asset standalone quantile-combo alpha search.

Independent alpha surface: BTC entries are driven only by causal cross-market
features (ETH/SOL/BNB/XRP/ADA/DOGE relative returns, dispersion, volume shocks,
and BTC funding/premium where available).  Thresholds are fitted only on the
train window (<2024).  Evaluation uses 6bp/side and strict in-position MDD from
``training.search_vpin_formulaic_alpha``.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.search_vpin_formulaic_alpha import load_market_and_splits, stats, trade_arrays

OUT = "results/cross_asset_quantile_standalone_2026-07-09.json"
DOC = "docs/cross-asset-quantile-standalone-2026-07-09.md"
SYMS = ["ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT"]
HOLDS = [96]
STRIDES = [24]


def z(s: pd.Series, n: int) -> pd.Series:
    mu = s.rolling(n, min_periods=min(n, max(20, n // 4))).mean()
    sd = s.rolling(n, min_periods=min(n, max(20, n // 4))).std(ddof=0)
    return ((s - mu) / sd.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).clip(-8, 8).fillna(0)


def ret(s: pd.Series, n: int) -> pd.Series:
    return (s / s.shift(n).replace(0, np.nan) - 1.0).replace([np.inf, -np.inf], np.nan).clip(-5, 5).fillna(0)


def load_alt(sym: str, dates: pd.Series) -> pd.DataFrame:
    p = Path(f"data/binance_um_pool_5m_2023_2026/{sym}_5m_2023-01_2026-05.csv.gz")
    if not p.exists():
        raise FileNotFoundError(p)
    raw = pd.read_csv(p, usecols=lambda c: c in {"date", "open", "high", "low", "close", "volume", "quote_asset_volume", "taker_buy_quote"})
    raw["date"] = pd.to_datetime(raw["date"], utc=True, errors="coerce").dt.tz_convert(None)
    raw = raw.dropna(subset=["date", "close"]).drop_duplicates("date").sort_values("date")
    base = pd.DataFrame({"date": dates})
    x = pd.merge_asof(base.sort_values("date"), raw.sort_values("date"), on="date", direction="backward", tolerance=pd.Timedelta("7min"))
    prefix = sym.replace("USDT", "").lower()
    return x.rename(columns={c: f"{prefix}_{c}" for c in x.columns if c != "date"})


def add_aux_1h(feat: pd.DataFrame, dates: pd.Series) -> pd.DataFrame:
    base = pd.DataFrame({"date": dates})
    for kind in ["funding", "premium_1h"]:
        p = Path(f"data/binance_um_aux_btc_2020_2026/BTCUSDT_{kind}_2020-01-01_2026-06-01.csv.gz")
        if not p.exists():
            continue
        raw = pd.read_csv(p)
        tcol = "date" if "date" in raw.columns else raw.columns[0]
        raw[tcol] = pd.to_datetime(raw[tcol], utc=True, errors="coerce").dt.tz_convert(None)
        valcols = [c for c in raw.columns if c != tcol]
        # Prefer the most numeric-looking value column.
        best = None
        for c in valcols:
            if pd.api.types.is_numeric_dtype(raw[c]):
                best = c
                break
        if best is None:
            continue
        r = raw[[tcol, best]].dropna().sort_values(tcol).rename(columns={tcol: "date", best: f"btc_{kind}"})
        x = pd.merge_asof(base.sort_values("date"), r, on="date", direction="backward", tolerance=pd.Timedelta("9h"))
        feat[f"btc_{kind}"] = x[f"btc_{kind}"].astype(float).fillna(0).to_numpy()
        feat[f"btc_{kind}_z"] = z(pd.Series(feat[f"btc_{kind}"], index=feat.index), 288)
    return feat


def build_features(m: pd.DataFrame) -> pd.DataFrame:
    dates = m["date"]
    c = m.close.astype(float).reset_index(drop=True)
    qv = (m.quote_asset_volume if "quote_asset_volume" in m else m.volume * m.close).astype(float).reset_index(drop=True)
    btc_ret = {n: ret(c, n) for n in [6, 12, 24, 48, 72, 144, 288]}
    feat = pd.DataFrame(index=m.index)
    feat["btc_ret_z_24"] = z(btc_ret[24], 288)
    feat["btc_ret_z_72"] = z(btc_ret[72], 288)
    feat["btc_qv_z"] = z(qv, 288)
    feat["btc_absret_qv_z"] = z(btc_ret[6].abs() * qv, 288)
    alt_rets: dict[int, list[pd.Series]] = {n: [] for n in [6, 12, 24, 48, 72, 144, 288]}
    alt_volzs: list[pd.Series] = []
    for sym in SYMS:
        x = load_alt(sym, dates)
        px = x[f"{sym.replace('USDT','').lower()}_close"].astype(float).ffill().bfill().reset_index(drop=True)
        aqv_col = f"{sym.replace('USDT','').lower()}_quote_asset_volume"
        av = (x[aqv_col].astype(float) if aqv_col in x else x[f"{sym.replace('USDT','').lower()}_volume"].astype(float) * px).ffill().fillna(0).reset_index(drop=True)
        pfx = sym.replace("USDT", "").lower()
        for n in [6, 12, 24, 48, 72, 144, 288]:
            r = ret(px, n)
            alt_rets[n].append(r)
            feat[f"{pfx}_relret_z_{n}"] = z(r - btc_ret[n], max(288, n * 4))
            feat[f"{pfx}_ret_z_{n}"] = z(r, max(288, n * 4))
        feat[f"{pfx}_qv_z"] = z(av, 288)
        alt_volzs.append(z(av, 288))
    for n, arrs in alt_rets.items():
        mat = pd.concat(arrs, axis=1)
        mean = mat.mean(axis=1)
        mx = mat.max(axis=1)
        mn = mat.min(axis=1)
        disp = mat.std(axis=1)
        feat[f"alt_mean_rel_z_{n}"] = z(mean - btc_ret[n], max(288, n * 4))
        feat[f"alt_max_rel_z_{n}"] = z(mx - btc_ret[n], max(288, n * 4))
        feat[f"alt_min_rel_z_{n}"] = z(mn - btc_ret[n], max(288, n * 4))
        feat[f"alt_disp_z_{n}"] = z(disp, max(288, n * 4))
        feat[f"btc_vs_altmean_z_{n}"] = z(btc_ret[n] - mean, max(288, n * 4))
    feat["alt_volume_breadth_z"] = pd.concat(alt_volzs, axis=1).mean(axis=1).fillna(0)
    feat = add_aux_1h(feat, dates)
    feat["cross_available"] = (dates >= pd.Timestamp("2023-01-01")).astype(float).to_numpy()
    # Causal session flags may capture stable cross-market liquidity timing; still fit only by train gates.
    hour = dates.dt.hour.to_numpy()
    feat["asia_session"] = ((hour >= 0) & (hour < 8)).astype(float)
    feat["europe_session"] = ((hour >= 8) & (hour < 16)).astype(float)
    feat["us_session"] = ((hour >= 16) & (hour < 24)).astype(float)
    return feat.replace([np.inf, -np.inf], np.nan).fillna(0)


def qthr(f: pd.DataFrame, train: np.ndarray, col: str, q: float) -> float | None:
    use_mask = train & (f["cross_available"].to_numpy(float) > 0.5) if "cross_available" in f else train
    vals = f.loc[use_mask, col].to_numpy(float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 100 or np.nanstd(vals) < 1e-12:
        return None
    return float(np.quantile(vals, q))


def materialize(f: pd.DataFrame, train: np.ndarray, spec: list[tuple[str, str, float]]) -> list[tuple[str, str, float, float]] | None:
    out = []
    for col, op, q in spec:
        if col.endswith("_session"):
            thr = 0.5
        else:
            thr = qthr(f, train, col, q)
            if thr is None:
                return None
        out.append((col, op, float(thr), float(q)))
    return out


def active(f: pd.DataFrame, terms: list[tuple[str, str, float, float]]) -> np.ndarray:
    a = np.ones(len(f), dtype=bool)
    if "cross_available" in f:
        a &= f["cross_available"].to_numpy(float) > 0.5
    for col, op, thr, _ in terms:
        x = f[col].to_numpy(float)
        a &= np.isfinite(x) & ((x >= thr) if op == ">=" else (x <= thr))
    return a


def eval_rule(m: pd.DataFrame, f: pd.DataFrame, masks: dict[str, np.ndarray], years: dict[str, float], terms, side: str, hold: int, stride: int) -> dict[str, Any]:
    act = active(f, terms)
    smod = (np.arange(len(m)) % stride) == 0
    fac, mn, rr = trade_arrays(m, hold, side)
    out = {}
    for sp, mask in masks.items():
        loc = []
        nxt = 0
        idx = np.flatnonzero(act & mask & smod)
        idx = idx[(idx >= 300) & (idx < len(m) - hold - 2)]
        for p in idx:
            p = int(p)
            xp = p + 1 + hold
            if p < nxt or xp >= len(m) or not mask[xp] or not np.isfinite(fac[p]):
                continue
            loc.append((float(fac[p]), float(mn[p]), float(rr[p])))
            nxt = xp
        out[sp] = stats(loc, years[sp])
    return out


def score(st: dict[str, Any]) -> tuple[Any, ...]:
    t, e, y, tr = st["test2024"], st["eval2025"], st["ytd2026"], st["train"]
    enough = t["trade_entries"] >= 20 and e["trade_entries"] >= 15 and y["trade_entries"] >= 5
    pos = t["cagr_pct"] > 0 and e["cagr_pct"] > 0 and y["cagr_pct"] > 0
    robust = tr["cagr_pct"] > -5 and min(t["cagr_to_strict_mdd"], e["cagr_to_strict_mdd"]) > 1.0
    min_oos = min(t["cagr_to_strict_mdd"], e["cagr_to_strict_mdd"], y["cagr_to_strict_mdd"])
    ret_sum = t["total_return_pct"] + e["total_return_pct"] + 0.5 * y["total_return_pct"]
    return (enough and pos and robust, min_oos, min(t["cagr_to_strict_mdd"], e["cagr_to_strict_mdd"]), ret_sum, t["trade_entries"] + e["trade_entries"])


def name_for(side: str, spec: list[tuple[str, str, float]]) -> str:
    raw = side + "|" + ";".join(f"{c}{o}{q:.2f}" for c, o, q in spec)
    return "crossq_" + hashlib.md5(raw.encode()).hexdigest()[:10]


def gen_specs(seed: int = 709, n: int = 350):
    rng = random.Random(seed)
    long_blocks = [
        [("btc_ret_z_24", "<=", 0.35), ("eth_relret_z_24", ">=", 0.55)],
        [("btc_ret_z_72", "<=", 0.35), ("sol_relret_z_72", ">=", 0.55)],
        [("btc_vs_altmean_z_48", "<=", 0.35), ("alt_disp_z_48", ">=", 0.55)],
        [("alt_mean_rel_z_24", ">=", 0.60), ("btc_absret_qv_z", "<=", 0.60)],
        [("alt_min_rel_z_72", ">=", 0.50), ("btc_ret_z_24", "<=", 0.50)],
        [("eth_ret_z_48", ">=", 0.55), ("btc_ret_z_24", "<=", 0.45)],
        [("sol_qv_z", ">=", 0.60), ("sol_relret_z_24", ">=", 0.55)],
        [("btc_premium_1h_z", "<=", 0.40), ("alt_mean_rel_z_72", ">=", 0.55)],
    ]
    short_blocks = [
        [("btc_ret_z_24", ">=", 0.65), ("eth_relret_z_24", "<=", 0.45)],
        [("btc_ret_z_72", ">=", 0.65), ("sol_relret_z_72", "<=", 0.45)],
        [("btc_vs_altmean_z_48", ">=", 0.65), ("alt_disp_z_48", ">=", 0.55)],
        [("alt_mean_rel_z_24", "<=", 0.40), ("btc_absret_qv_z", ">=", 0.55)],
        [("alt_max_rel_z_72", "<=", 0.50), ("btc_ret_z_24", ">=", 0.50)],
        [("eth_ret_z_48", "<=", 0.45), ("btc_ret_z_24", ">=", 0.55)],
        [("doge_qv_z", ">=", 0.60), ("doge_relret_z_24", "<=", 0.45)],
        [("btc_funding_z", ">=", 0.60), ("alt_mean_rel_z_72", "<=", 0.45)],
    ]
    opts_long = [
        ("alt_volume_breadth_z", ">=", 0.55), ("alt_volume_breadth_z", "<=", 0.55),
        ("btc_qv_z", "<=", 0.60), ("btc_qv_z", ">=", 0.55),
        ("alt_disp_z_144", "<=", 0.55), ("alt_disp_z_144", ">=", 0.55),
        ("xrp_relret_z_48", ">=", 0.50), ("bnb_relret_z_48", ">=", 0.50),
        ("asia_session", ">=", 0.50), ("us_session", ">=", 0.50),
    ]
    opts_short = [
        ("alt_volume_breadth_z", ">=", 0.55), ("alt_volume_breadth_z", "<=", 0.55),
        ("btc_qv_z", "<=", 0.60), ("btc_qv_z", ">=", 0.55),
        ("alt_disp_z_144", "<=", 0.55), ("alt_disp_z_144", ">=", 0.55),
        ("xrp_relret_z_48", "<=", 0.50), ("bnb_relret_z_48", "<=", 0.50),
        ("asia_session", ">=", 0.50), ("us_session", ">=", 0.50),
    ]
    seen = set()

    def jitter(t):
        c, o, q = t
        if c.endswith("_session"):
            return t
        return (c, o, max(0.05, min(0.95, q + rng.choice([-0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15]))))

    for side, blocks, opts in [("long", long_blocks, opts_long), ("short", short_blocks, opts_short)]:
        for b in blocks:
            for _ in range(60):
                spec = [jitter(x) for x in b]
                spec += [jitter(x) for x in rng.sample(opts, rng.randint(0, 2))]
                out = []
                used = set()
                for t in spec:
                    if t[0] not in used:
                        used.add(t[0]); out.append(t)
                key = (side, tuple(out))
                if key not in seen:
                    seen.add(key); yield side, out
    for _ in range(n):
        side = rng.choice(["long", "short"])
        blocks = long_blocks if side == "long" else short_blocks
        opts = opts_long if side == "long" else opts_short
        spec = []
        for b in rng.sample(blocks, rng.randint(1, 2)):
            spec.extend(jitter(x) for x in b)
        spec.extend(jitter(x) for x in rng.sample(opts, rng.randint(0, 2)))
        out = []
        used = set()
        for t in spec:
            if t[0] not in used:
                used.add(t[0]); out.append(t)
        if 2 <= len(out) <= 5:
            key = (side, tuple(out))
            if key not in seen:
                seen.add(key); yield side, out


def main() -> None:
    m, _, masks, years = load_market_and_splits()
    f = build_features(m)
    train = masks["train"]
    rows = []
    tested = 0
    for side, spec in gen_specs():
        terms = materialize(f, train, spec)
        if terms is None:
            continue
        a = active(f, terms)
        ar = float(a[train].mean())
        # Cross-asset files start in 2023; enforce enough train activity post-warmup.
        if ar < 0.002 or ar > 0.35:
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
    top = [{k: v for k, v in r.items() if k != "score_tuple"} for r in rows[:200]]
    report = {
        "protocol": "Standalone cross-asset quantile-combo search. Thresholds train<2024 only; no existing signal/portfolio context; 6bp/side; strict MDD includes in-position adverse excursion and forced period-contained exits.",
        "tested_specs": tested,
        "all_count": len(rows),
        "top": top,
    }
    Path(OUT).write_text(json.dumps(report, indent=2, ensure_ascii=False))

    def fmt(s: dict[str, Any]) -> str:
        return f"{s['total_return_pct']:.2f}/{s['cagr_pct']:.2f}/{s['strict_mdd_pct']:.2f}/{s['cagr_to_strict_mdd']:.2f}/{s['trade_entries']}/{s['win_rate']:.2f}/{s['bar_sharpe_like']:.2f}"

    md = [
        "# Cross-asset standalone quantile search (2026-07-09)", "", report["protocol"], "",
        f"tested_specs={tested}, all_count={len(rows)}", "",
        "| rank | name | side | active | hold/stride | train ret/CAGR/MDD/ratio/trades/win/sh | 2024 | 2025 | 2026 | terms |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for i, r in enumerate(top[:80], 1):
        st = r["stats"]
        terms = "; ".join(f"{t['feature']} {t['op']} q{t['train_q']:.2f}({t['threshold']:.4g})" for t in r["terms"])
        md.append(f"| {i} | {r['name']} | {r['side']} | {r['active_rate_train']:.3f} | {r['hold']}/{r['stride']} | {fmt(st['train'])} | {fmt(st['test2024'])} | {fmt(st['eval2025'])} | {fmt(st['ytd2026'])} | `{terms}` |")
    Path(DOC).write_text("\n".join(md) + "\n")
    print(json.dumps({"output": OUT, "doc": DOC, "tested_specs": tested, "all_count": len(rows), "top": top[:12]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
