"""VPIN/order-flow-toxicity and 101-formulaic-alpha inspired BTC sleeve scan.

Research-only diagnostic:
- Features are causal/past-only on 5m bars.
- Thresholds are fit from train (<2024) quantiles only.
- Splits are train/test2024/eval2025/ytd2026 from the repo's canonical prep.
- Strict MDD includes intra-position adverse excursion and period-contained exits.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

OUT = "results/vpin_formulaic_alpha_scan_2026-07-09.json"
DOC = "docs/vpin-formulaic-alpha-scan-2026-07-09.md"
COST = 0.0006



SPLIT_BOUNDS = {
    "train": ("2020-01-01", "2024-01-01"),
    "test2024": ("2024-01-01", "2025-01-01"),
    "eval2025": ("2025-01-01", "2026-01-01"),
    "ytd2026": ("2026-01-01", None),
}


def _years(start: pd.Timestamp, end: pd.Timestamp) -> float:
    return max((end - start).total_seconds() / (365.25 * 24 * 3600), 1e-9)


def load_market_and_splits() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], dict[str, float]]:
    """Load the newest BTCUSDT 5m OHLCV+taker cache without OI dependency."""
    candidates = [
        Path("data/cache_market_ext_5m_wavefull_2020-01-01_2026-07-05_dbappend.csv.gz"),
        Path("data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz"),
        Path("data/2020-01-01_2026-06-01_btcusdt_futures_5m.csv.gz"),
    ]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        raise FileNotFoundError("No BTCUSDT 5m market cache found")
    cols = [
        "date", "open", "high", "low", "close", "volume", "quote_asset_volume",
        "number_of_trades", "taker_buy_base", "taker_buy_quote",
    ]
    m = pd.read_csv(src, usecols=lambda c: c in cols)
    m["date"] = pd.to_datetime(m["date"], utc=True, errors="coerce").dt.tz_convert(None)
    m = m.dropna(subset=["date", "open", "high", "low", "close"]).drop_duplicates("date").sort_values("date").reset_index(drop=True)
    if "quote_asset_volume" not in m:
        m["quote_asset_volume"] = m["volume"].astype(float) * m["close"].astype(float)
    if "taker_buy_quote" not in m:
        m["taker_buy_quote"] = m["quote_asset_volume"].astype(float) * 0.5
    feat = pd.DataFrame(index=m.index)
    dates = m["date"]
    data_end = dates.max() + pd.Timedelta(minutes=5)
    masks: dict[str, np.ndarray] = {}
    years: dict[str, float] = {}
    for name, (start_s, end_s) in SPLIT_BOUNDS.items():
        start = pd.Timestamp(start_s)
        end = pd.Timestamp(end_s) if end_s is not None else data_end
        masks[name] = ((dates >= start) & (dates < end)).to_numpy(bool)
        years[name] = _years(start, min(end, data_end))
    return m, feat, masks, years

def _z(s: pd.Series, n: int) -> pd.Series:
    mu = s.rolling(n, min_periods=max(20, n // 4)).mean()
    sd = s.rolling(n, min_periods=max(20, n // 4)).std(ddof=0)
    return ((s - mu) / sd.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).clip(-8, 8).fillna(0)


def _ret(s: pd.Series, n: int) -> pd.Series:
    return (s / s.shift(n).replace(0, np.nan) - 1).replace([np.inf, -np.inf], np.nan).clip(-10, 10).fillna(0)


def _ts_rank(s: pd.Series, n: int) -> pd.Series:
    # Fast causal rank proxy: rolling z-score mapped to (0, 1). Exact rolling rank on
    # 5m multi-year data is too slow for exploratory sweeps and does not affect
    # train-only quantile calibration semantics.
    z = _z(s.astype(float), n).clip(-8, 8)
    return (1.0 / (1.0 + np.exp(-z))).fillna(0.5)


def _corr(a: pd.Series, b: pd.Series, n: int) -> pd.Series:
    return a.rolling(n, min_periods=max(20, n // 4)).corr(b).replace([np.inf, -np.inf], np.nan).fillna(0)


def add_vpin_formulaic_features(m: pd.DataFrame, f: pd.DataFrame) -> pd.DataFrame:
    o = m.open.astype(float)
    h = m.high.astype(float)
    l = m.low.astype(float)
    c = m.close.astype(float)
    v = m.volume.astype(float)
    qv = m.quote_asset_volume.astype(float) if "quote_asset_volume" in m else v * c
    tbq = m.taker_buy_quote.astype(float) if "taker_buy_quote" in m else qv * 0.5
    sellq = (qv - tbq).clip(lower=0)
    signedq = (2 * tbq - qv).fillna(0)
    imb = (signedq / qv.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0)
    vwap = (qv / v.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(c)
    spread = ((h - l) / c.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0)
    lr = np.log(c / c.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0)

    out = pd.DataFrame(index=m.index)
    # VPIN proxy: rolling volume imbalance over rolling total volume. True volume-bucket VPIN
    # would require trade prints; this is a bar-level volume-time approximation using taker quote flow.
    for n in [24, 48, 72, 144, 288, 576]:
        denom = qv.rolling(n, min_periods=max(20, n // 4)).sum().replace(0, np.nan)
        out[f"vp_vpin_{n}"] = ((tbq - sellq).abs().rolling(n, min_periods=max(20, n // 4)).sum() / denom).replace([np.inf, -np.inf], np.nan).fillna(0)
        out[f"vp_vpin_z_{n}"] = _z(out[f"vp_vpin_{n}"], max(288, n * 4))
        out[f"vp_imb_mean_{n}"] = imb.rolling(n, min_periods=max(20, n // 4)).mean().fillna(0)
        out[f"vp_imb_z_{n}"] = _z(out[f"vp_imb_mean_{n}"], max(288, n * 4))
        out[f"vp_qv_z_{n}"] = _z(qv, max(288, n * 4))
        out[f"vp_rvol_{n}"] = lr.rolling(n, min_periods=max(20, n // 4)).std(ddof=0).fillna(0) * math.sqrt(n)
        out[f"vp_rvol_z_{n}"] = _z(out[f"vp_rvol_{n}"], max(288, n * 4))
        out[f"vp_ret_{n}"] = _ret(c, n)
        out[f"vp_ret_rank_{n}"] = _ts_rank(out[f"vp_ret_{n}"], max(144, n * 2))

    # 101-formulaic-inspired single-asset time-series transforms.
    # alpha101-like: intrabar close-open normalized by range, traded with delay 1.
    out["fq_intrabar_strength"] = ((c - o) / (h - l).replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).clip(-5, 5).fillna(0)
    out["fq_intrabar_strength_rank"] = _ts_rank(out["fq_intrabar_strength"], 288)
    # alpha42-like: close vs VWAP location. Positive means close below vwap (reversion long candidate).
    out["fq_vwap_revert"] = ((vwap - c) / c.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).clip(-1, 1).fillna(0)
    out["fq_vwap_revert_z"] = _z(out["fq_vwap_revert"], 288)
    # Price/volume interaction and volume-rank variants inspired by multiple 101 alphas.
    out["fq_ret_vol_corr_72"] = _corr(_ret(c, 1), _z(qv, 288), 72)
    out["fq_ret_vol_corr_288"] = _corr(_ret(c, 1), _z(qv, 288), 288)
    out["fq_volume_delta_rank"] = _ts_rank(qv.diff(1).fillna(0), 288)
    out["fq_abs_move_vol_rank"] = _ts_rank(_ret(c, 1).abs() * qv, 288)
    out["fq_spread_rank"] = _ts_rank(spread, 288)
    out["fq_signed_vol_pressure"] = _z(signedq, 288) * _z(qv, 288)
    out["fq_signed_vol_pressure_rank"] = _ts_rank(out["fq_signed_vol_pressure"], 288)
    # Toxicity + formulaic composites.
    out["vx_toxic_selloff"] = out["vp_vpin_z_144"].clip(lower=0) + (-out["vp_imb_z_72"]).clip(lower=0) + (-_z(out["vp_ret_72"], 288)).clip(lower=0)
    out["vx_toxic_rally"] = out["vp_vpin_z_144"].clip(lower=0) + out["vp_imb_z_72"].clip(lower=0) + _z(out["vp_ret_72"], 288).clip(lower=0)
    out["vx_lowtox_momo_long"] = (-out["vp_vpin_z_144"]).clip(lower=0) + out["fq_intrabar_strength"].clip(lower=0) + _z(out["vp_ret_24"], 288).clip(lower=0)
    out["vx_lowtox_momo_short"] = (-out["vp_vpin_z_144"]).clip(lower=0) + (-out["fq_intrabar_strength"]).clip(lower=0) + (-_z(out["vp_ret_24"], 288)).clip(lower=0)
    return pd.concat([f, out], axis=1).loc[:, lambda x: ~x.columns.duplicated(keep="last")].replace([np.inf, -np.inf], np.nan).fillna(0)


def q(feat: pd.DataFrame, mask: np.ndarray, col: str, qq: float) -> float | None:
    vals = feat.loc[mask, col].to_numpy(float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 100 or np.nanstd(vals) < 1e-12:
        return None
    return float(np.quantile(vals, qq))


def active_from_terms(feat: pd.DataFrame, terms: list[tuple[str, str, float]]) -> np.ndarray:
    a = np.ones(len(feat), dtype=bool)
    for col, op, thr in terms:
        x = feat[col].to_numpy(float)
        a &= np.isfinite(x) & ((x >= thr) if op == ">=" else (x <= thr))
    return a


TRADE_CACHE: dict[tuple[int, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def trade_arrays(m: pd.DataFrame, hold: int, side: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    key = (int(hold), side)
    if key in TRADE_CACHE:
        return TRADE_CACHE[key]
    op = m.open.to_numpy(float)
    hi = m.high.to_numpy(float)
    lo = m.low.to_numpy(float)
    n = len(m)
    factor = np.full(n, np.nan, float)
    minrel = np.full(n, np.nan, float)
    cum = np.ones(n, float) * (1.0 - COST)
    mn = cum.copy()
    valid = np.ones(n, bool)
    for k in range(int(hold)):
        j = np.arange(n) + 1 + k
        ok = (j + 1 < n)
        jj = np.minimum(j, n - 1)
        oj = op[jj]
        if side == "long":
            adverse = (lo[jj] - oj) / oj
            rr = (op[np.minimum(jj + 1, n - 1)] - oj) / oj
        else:
            adverse = (oj - hi[jj]) / oj
            rr = (oj - op[np.minimum(jj + 1, n - 1)]) / oj
        step_ok = ok & np.isfinite(oj) & (oj > 0) & np.isfinite(adverse) & np.isfinite(rr)
        valid &= step_ok
        mn = np.minimum(mn, cum * np.maximum(0.0, 1.0 + np.where(step_ok, adverse, 0.0)))
        cum *= np.maximum(0.0, 1.0 + np.where(step_ok, rr, 0.0))
    fac = cum * (1.0 - COST)
    mn = np.minimum(mn, fac)
    factor[valid] = fac[valid]
    minrel[valid] = mn[valid]
    ret = factor - 1.0
    TRADE_CACHE[key] = (factor, minrel, ret)
    return TRADE_CACHE[key]


def stats(local: list[tuple[float, float, float]], years: float) -> dict[str, Any]:
    if not local:
        return dict(total_return_pct=0, cagr_pct=0, strict_mdd_pct=0, cagr_to_strict_mdd=0, trade_entries=0, win_rate=0, bar_sharpe_like=0, mean_trade_ret_pct=0)
    eq = 1.0
    peak = 1.0
    mdd = 0.0
    rets = []
    for fac, min_rel, r in local:
        mdd = max(mdd, 1.0 - (eq * min_rel) / max(peak, 1e-12))
        eq *= fac
        peak = max(peak, eq)
        mdd = max(mdd, 1.0 - eq / max(peak, 1e-12))
        rets.append(r)
    cagr = (eq ** (1.0 / years) - 1.0) * 100 if eq > 0 else -100.0
    md = mdd * 100
    arr = np.array(rets, float)
    sh = float(arr.mean() / arr.std(ddof=1) * math.sqrt(len(arr) / max(years, 1e-9))) if len(arr) > 1 and arr.std(ddof=1) > 0 else 0.0
    return dict(total_return_pct=(eq - 1.0) * 100, cagr_pct=cagr, strict_mdd_pct=md, cagr_to_strict_mdd=cagr / md if md > 1e-12 else 0.0, trade_entries=len(local), win_rate=float((arr > 0).mean()), bar_sharpe_like=sh, mean_trade_ret_pct=float(arr.mean() * 100))


def eval_rule(m: pd.DataFrame, feat: pd.DataFrame, masks: dict[str, np.ndarray], years: dict[str, float], terms: list[tuple[str, str, float]], side: str, hold: int, stride: int) -> dict[str, Any]:
    act = active_from_terms(feat, terms)
    n = len(m)
    ar = np.arange(n)
    stride_mask = (ar % int(stride)) == 0
    out = {}
    for sp, mask in masks.items():
        local = []
        nxt = 0
        fac, mn, ret = trade_arrays(m, hold, side)
        idx = np.flatnonzero(act & mask & stride_mask)
        idx = idx[(idx >= 143) & (idx < n - hold - 2)]
        for p0 in idx:
            p0 = int(p0)
            xp = p0 + 1 + int(hold)
            if p0 < nxt or xp >= n or not mask[xp] or not np.isfinite(fac[p0]):
                continue
            local.append((float(fac[p0]), float(mn[p0]), float(ret[p0])))
            nxt = xp
        out[sp] = stats(local, years[sp])
    return out


def score(res: dict[str, Any]) -> tuple[Any, ...]:
    t, e, y = res["test2024"], res["eval2025"], res["ytd2026"]
    ok = t["trade_entries"] >= 15 and e["trade_entries"] >= 10 and t["cagr_pct"] > 0 and e["cagr_pct"] > 0
    clean = t["cagr_to_strict_mdd"] >= 2 and e["cagr_to_strict_mdd"] >= 2
    return (ok, clean, min(t["cagr_to_strict_mdd"], e["cagr_to_strict_mdd"]), y["cagr_to_strict_mdd"], t["total_return_pct"] + e["total_return_pct"] + 0.5 * y["total_return_pct"], t["trade_entries"] + e["trade_entries"])


def build_terms(feat: pd.DataFrame, train: np.ndarray, qs: list[tuple[str, str, float]]) -> list[tuple[str, str, float]] | None:
    terms = []
    for col, op, qq in qs:
        thr = q(feat, train, col, qq)
        if thr is None:
            return None
        terms.append((col, op, thr))
    return terms


def main() -> None:
    m, feat, masks, years = load_market_and_splits()
    feat = add_vpin_formulaic_features(m, feat)
    train = masks["train"]
    families = [
        # VPIN toxicity: continuation and contrarian variants are both tested via flip.
        ("vpin_toxic_sell_cont", "short", [("vx_toxic_selloff", ">=", 0.85)]),
        ("vpin_toxic_sell_revert", "long", [("vx_toxic_selloff", ">=", 0.85)]),
        ("vpin_toxic_rally_cont", "long", [("vx_toxic_rally", ">=", 0.85)]),
        ("vpin_toxic_rally_revert", "short", [("vx_toxic_rally", ">=", 0.85)]),
        ("vpin_high_sell_pressure", "short", [("vp_vpin_z_144", ">=", 0.80), ("vp_imb_z_72", "<=", 0.20)]),
        ("vpin_high_buy_pressure", "long", [("vp_vpin_z_144", ">=", 0.80), ("vp_imb_z_72", ">=", 0.80)]),
        ("lowtox_formulaic_momo_long", "long", [("vx_lowtox_momo_long", ">=", 0.85)]),
        ("lowtox_formulaic_momo_short", "short", [("vx_lowtox_momo_short", ">=", 0.85)]),
        # 101 formulaic-inspired VWAP/range/volume effects.
        ("vwap_revert_long", "long", [("fq_vwap_revert_z", ">=", 0.85), ("vp_vpin_z_144", "<=", 0.70)]),
        ("vwap_revert_short", "short", [("fq_vwap_revert_z", "<=", 0.15), ("vp_vpin_z_144", "<=", 0.70)]),
        ("intrabar_strength_delay1_long", "long", [("fq_intrabar_strength_rank", ">=", 0.85), ("fq_abs_move_vol_rank", ">=", 0.60)]),
        ("intrabar_strength_delay1_short", "short", [("fq_intrabar_strength_rank", "<=", 0.15), ("fq_abs_move_vol_rank", ">=", 0.60)]),
        ("ret_vol_corr_reversal_long", "long", [("fq_ret_vol_corr_72", "<=", 0.15), ("vp_ret_rank_72", "<=", 0.30)]),
        ("ret_vol_corr_reversal_short", "short", [("fq_ret_vol_corr_72", ">=", 0.85), ("vp_ret_rank_72", ">=", 0.70)]),
        ("volume_delta_pressure_long", "long", [("fq_volume_delta_rank", ">=", 0.85), ("fq_signed_vol_pressure_rank", ">=", 0.80)]),
        ("volume_delta_pressure_short", "short", [("fq_volume_delta_rank", ">=", 0.85), ("fq_signed_vol_pressure_rank", "<=", 0.20)]),
    ]
    # For exploratory alpha discovery, also test side-inverted sleeves. Several
    # microstructure signals are regime-dependent and can behave as crowding/
    # exhaustion rather than continuation.
    expanded = []
    for name, side, qs in families:
        expanded.append((name, side, qs))
        expanded.append((f"{name}_flip", "short" if side == "long" else "long", qs))

    rows = []
    for name, side, qs in expanded:
        terms = build_terms(feat, train, qs)
        if terms is None:
            continue
        for hold in [12, 24, 36, 48, 72, 96, 144]:
            for stride in [6, 12, 24]:
                res = eval_rule(m, feat, masks, years, terms, side, hold, stride)
                rows.append({"name": name, "side": side, "terms": [{"feature": c, "op": op, "threshold": thr} for c, op, thr in terms], "hold": hold, "stride": stride, "stats": res, "score_tuple": score(res)})
    rows.sort(key=lambda r: r["score_tuple"], reverse=True)
    report = {
        "protocol": "VPIN/order-flow-toxicity and 101 Formulaic Alphas inspired scan. Thresholds fit train<2024 only. Cost 6bp/side. Strict MDD includes intra-position adverse excursion. Diagnostic, not live-promoted.",
        "sources": {
            "formulaic_alphas": "Kakushadze 2015/2016, 101 Formulaic Alphas, arXiv:1601.00991",
            "vpin": "Easley, Lopez de Prado, O'Hara VPIN/order-flow toxicity literature; bar-level taker quote flow approximation used here",
        },
        "top": [{k: v for k, v in r.items() if k != "score_tuple"} for r in rows[:100]],
        "all_count": len(rows),
    }
    Path(OUT).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    md = ["# VPIN + formulaic alpha scan (2026-07-09)", "", report["protocol"], "", "## Source ideas", "- 101 Formulaic Alphas: price/volume/VWAP/range transforms and short holding periods.", "- VPIN/order-flow toxicity: rolling volume imbalance over volume, approximated from Binance taker buy quote volume.", "", "| rank | name | side | hold/stride | train ratio/trades | 2024 ret/CAGR/MDD/ratio/trades | 2025 ret/CAGR/MDD/ratio/trades | 2026 ret/CAGR/MDD/ratio/trades | terms |", "|---:|---|---|---:|---:|---:|---:|---:|---|"]
    for i, r in enumerate(report["top"][:30], 1):
        st = r["stats"]
        terms_txt = "; ".join(f"{t['feature']} {t['op']} {t['threshold']:.4g}" for t in r["terms"])
        def fmt(sp: str) -> str:
            s = st[sp]
            return f"{s['total_return_pct']:.2f}/{s['cagr_pct']:.2f}/{s['strict_mdd_pct']:.2f}/{s['cagr_to_strict_mdd']:.2f}/{s['trade_entries']}"
        md.append(f"| {i} | {r['name']} | {r['side']} | {r['hold']}/{r['stride']} | {st['train']['cagr_to_strict_mdd']:.2f}/{st['train']['trade_entries']} | {fmt('test2024')} | {fmt('eval2025')} | {fmt('ytd2026')} | `{terms_txt}` |")
    Path(DOC).write_text("\n".join(md) + "\n")
    print(json.dumps({"output": OUT, "doc": DOC, "top": report["top"][:12]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
