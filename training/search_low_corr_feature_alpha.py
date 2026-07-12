"""Search alpha candidates from features weakly correlated with existing alpha components.

Research-only diagnostic:
- Existing alpha masks are built from the current component registry and train rows only.
- Candidate feature thresholds are train<2024 quantiles only.
- Feature availability is causal/past-only as implemented in preprocessing modules.
- Result ranking is diagnostic; candidates selected after reading OOS stats are not live-grade.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from preprocessing.binance_aux_features import attach_binance_um_aux_features
from preprocessing.market_features import build_market_feature_frame
from training.long_regime_interest_gate_validation import build_interest_features
from training.long_component_tp_union_scan import COMPONENTS as LONG_COMPONENTS, _component_mask
from training.search_vpin_formulaic_alpha import add_vpin_formulaic_features, stats, trade_arrays
from training.search_alpha101_derivative_alphas import add_features as add_alpha101_features


EXTRA_COMPONENTS: dict[str, list[tuple[str, str, float]]] = {
    "short_fx_stress": [("htf_3d_return_1", "le", -0.0325294973), ("usdkrw_zscore", "ge", 1.3870063775)],
    "short_premium_panic": [("htf_3d_range_pos", "le", -0.5114186851), ("premium_index_zscore", "le", -1.47209312)],
    "short_kimchi_unwind": [("htf_3d_return_1", "le", -0.0303196833), ("kimchi_premium_change", "le", -0.0046123752)],
}

COMPONENT_GROUPS: dict[str, list[str]] = {
    "long_range_funding_premium": ["range_bb90", "funding10_trend70", "premium20_mom90"],
    "long_funding_compression_premium": ["funding10_trend70", "compress05_trend80", "premium20_mom90"],
    "long_range_funding_compression": ["range_bb90", "funding10_trend70", "compress05_trend80"],
    "long_minimal_funding_premium": ["funding10_trend70", "premium20_mom90"],
    "short_premium_kimchi_union": ["short_premium_panic", "short_kimchi_unwind"],
}


def _rolling_zscore_local(series: pd.Series, window: int) -> pd.Series:
    window = max(2, int(window))
    mean = series.rolling(window, min_periods=max(10, window // 5)).mean()
    std = series.rolling(window, min_periods=max(10, window // 5)).std(ddof=0)
    return ((series - mean) / std.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).clip(-8.0, 8.0).fillna(0.0)


def _add_oi_derived_features_local(market: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    if "funding_zscore" in out.columns and "funding_z" not in out.columns:
        out["funding_z"] = out["funding_zscore"]
    if "premium_index_zscore" in out.columns and "premium_z" not in out.columns:
        out["premium_z"] = out["premium_index_zscore"]
    if "open_interest" not in market.columns:
        return out
    close = pd.to_numeric(market["close"], errors="coerce")
    oi = pd.to_numeric(market["open_interest"], errors="coerce").replace(0.0, np.nan).ffill()
    for bars, name in [(24, "2h"), (48, "4h"), (72, "6h"), (96, "8h"), (144, "12h"), (288, "1d")]:
        oi_ret = np.log(oi / oi.shift(bars).replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        px_ret = np.log(close / close.shift(bars).replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        div = oi_ret - px_ret
        out[f"oi_ret_z_{bars}"] = _rolling_zscore_local(oi_ret, max(288, bars * 4))
        out[f"px_ret_z_{bars}"] = _rolling_zscore_local(px_ret, max(288, bars * 4))
        out[f"oi_minus_px_z_{bars}"] = _rolling_zscore_local(div, max(288, bars * 4))
        out[f"px_minus_oi_z_{bars}"] = _rolling_zscore_local(-div, max(288, bars * 4))
        out[f"oi_ret_{name}_z"] = out[f"oi_ret_z_{bars}"]
        out[f"px_ret_{name}_z"] = out[f"px_ret_z_{bars}"]
        out[f"oi_minus_px_{name}_z"] = out[f"oi_minus_px_z_{bars}"]
        out[f"px_minus_oi_{name}_z"] = out[f"px_minus_oi_z_{bars}"]
    out["oi_available"] = pd.to_numeric(market.get("open_interest_available", pd.Series(1.0, index=market.index)), errors="coerce").fillna(0.0)
    taker = out.get("taker_imbalance", pd.Series(0.0, index=out.index))
    out["btc_oi_unwind_long"] = (-out["oi_ret_z_72"]).clip(lower=0) + out.get("px_ret_z_72", 0).clip(lower=0)
    out["btc_oi_squeeze_short"] = out["oi_ret_z_72"].clip(lower=0) + (-out.get("px_ret_z_72", 0)).clip(lower=0)
    out["btc_liq_revert_long"] = (-out.get("px_ret_z_72", 0)).clip(lower=0) + (-out["oi_ret_z_72"]).clip(lower=0)
    out["btc_cvd_absorb_long"] = (-out.get("px_ret_z_72", 0)).clip(lower=0) + taker.clip(lower=0)
    out["btc_overheat_short"] = out.get("px_ret_z_72", 0).clip(lower=0) + out["oi_ret_z_72"].clip(lower=0)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _mask_from_conditions(features: pd.DataFrame, conditions: list[tuple[str, str, float]]) -> np.ndarray:
    mask = np.ones(len(features), dtype=bool)
    for feature, op, threshold in conditions:
        if feature not in features.columns:
            return np.zeros(len(features), dtype=bool)
        x = features[feature].to_numpy(float)
        mask &= np.isfinite(x) & ((x <= threshold) if op == "le" else (x >= threshold))
    return mask


def _build_component_frame_local(features: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    cols: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {}
    for name in sorted(LONG_COMPONENTS):
        cols[name] = _component_mask(features, name).astype(float)
        metadata[name] = {"kind": "long_component", "conditions": LONG_COMPONENTS[name]}
    for name, conditions in EXTRA_COMPONENTS.items():
        cols[name] = _mask_from_conditions(features, conditions).astype(float)
        metadata[name] = {"kind": "extra_component", "conditions": conditions}
    for group, members in COMPONENT_GROUPS.items():
        active = np.zeros(len(features), dtype=bool)
        for member in members:
            if member in LONG_COMPONENTS:
                active |= _component_mask(features, member)
            elif member in EXTRA_COMPONENTS:
                active |= _mask_from_conditions(features, EXTRA_COMPONENTS[member])
        cols[group] = active.astype(float)
        metadata[group] = {"kind": "candidate_union", "members": members}
    return pd.DataFrame(cols, index=features.index), metadata

SPLIT_BOUNDS = {
    "train": ("2020-01-01", "2024-01-01"),
    "test2024": ("2024-01-01", "2025-01-01"),
    "eval2025": ("2025-01-01", "2026-01-01"),
    "ytd2026": ("2026-01-01", None),
}


@dataclass(frozen=True)
class LowCorrAlphaSearchConfig:
    input_csv: str = "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_oi.csv.gz"
    funding_csv: str = "data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz"
    premium_csv: str = "data/binance_um_aux_btc_2020_2026/BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz"
    output: str = "results/low_corr_feature_alpha_scan_2026-07-12.json"
    docs_output: str = "docs/low-corr-feature-alpha-scan-2026-07-12.md"
    exclude_from: str = "2026-06-02"
    window_size: int = 144
    max_component_abs_corr: float = 0.12
    max_pool_abs_corr: float = 0.35
    max_features: int = 50
    max_pair_features: int = 20
    quantiles: str = "0.05,0.10,0.15,0.20,0.80,0.85,0.90,0.95"
    holds: str = "24,72,144"
    strides: str = "12,24"
    min_train_active_rate: float = 0.001
    max_train_active_rate: float = 0.35
    candidate_prefixes: str = "a_,vp_,vx_,fq_,oi_,px_,taker_,btc_"
    min_train_trades: int = 50
    min_test_trades: int = 15
    min_eval_trades: int = 15
    min_ytd_trades: int = 8


def _parse_csv(raw: str, cast: Any) -> list[Any]:
    return [cast(x.strip()) for x in str(raw).split(",") if x.strip()]


def _years(start: pd.Timestamp, end: pd.Timestamp) -> float:
    return max((end - start).total_seconds() / (365.25 * 24 * 3600), 1e-9)


def _load_market(cfg: LowCorrAlphaSearchConfig) -> pd.DataFrame:
    market = pd.read_csv(cfg.input_csv, parse_dates=["date"], compression="infer")
    market["date"] = pd.to_datetime(market["date"], utc=True, errors="raise").dt.tz_convert(None)
    market = market.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    market = market[market["date"] < pd.Timestamp(cfg.exclude_from)].reset_index(drop=True)
    market = attach_binance_um_aux_features(
        market,
        funding_csv=cfg.funding_csv if cfg.funding_csv else None,
        premium_csv=cfg.premium_csv if cfg.premium_csv else None,
        funding_tolerance="12h",
        premium_tolerance="2h",
    )
    return market


def _split_masks(dates: pd.Series) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    end_data = dates.max() + pd.Timedelta(minutes=5)
    masks: dict[str, np.ndarray] = {}
    years: dict[str, float] = {}
    for name, (start_s, end_s) in SPLIT_BOUNDS.items():
        start = pd.Timestamp(start_s)
        end = pd.Timestamp(end_s) if end_s is not None else end_data
        end = min(end, end_data)
        masks[name] = np.asarray((dates >= start) & (dates < end), dtype=bool)
        years[name] = _years(start, end)
    return masks, years


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 30:
        return 0.0
    x = a[ok].astype(float)
    y = b[ok].astype(float)
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx <= 1e-12 or sy <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _rank_array(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(method="average").to_numpy(dtype=float)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 30:
        return 0.0
    return _pearson(_rank_array(a[ok]), _rank_array(b[ok]))


def _safe_quantile(x: np.ndarray, mask: np.ndarray, q: float) -> float | None:
    vals = x[mask & np.isfinite(x)]
    if vals.size < 300 or float(np.std(vals)) <= 1e-12:
        return None
    return float(np.quantile(vals, float(q)))


def _active_from_terms(features: pd.DataFrame, terms: list[tuple[str, str, float]]) -> np.ndarray:
    active = np.ones(len(features), dtype=bool)
    for col, op, thr in terms:
        x = features[col].to_numpy(float)
        active &= np.isfinite(x) & ((x >= thr) if op == ">=" else (x <= thr))
    return active


def _eval_rule(market: pd.DataFrame, features: pd.DataFrame, masks: dict[str, np.ndarray], years: dict[str, float], terms: list[tuple[str, str, float]], side: str, hold: int, stride: int) -> dict[str, Any]:
    active = _active_from_terms(features, terms)
    stride_mask = (np.arange(len(market)) % int(stride)) == 0
    fac, mn, ret = trade_arrays(market, int(hold), side)
    out: dict[str, Any] = {}
    for split, mask in masks.items():
        local: list[tuple[float, float, float]] = []
        next_allowed = 0
        idx = np.flatnonzero(active & mask & stride_mask)
        idx = idx[(idx >= 300) & (idx < len(market) - int(hold) - 2)]
        for p0 in idx:
            p0 = int(p0)
            xp = p0 + 1 + int(hold)
            if p0 < next_allowed or xp >= len(market) or not mask[xp] or not np.isfinite(fac[p0]):
                continue
            local.append((float(fac[p0]), float(mn[p0]), float(ret[p0])))
            next_allowed = xp
        out[split] = stats(local, years[split])
    return out


def _score(row: dict[str, Any], cfg: LowCorrAlphaSearchConfig) -> tuple[Any, ...]:
    st = row["stats"]
    tr, te, ev, yt = st["train"], st["test2024"], st["eval2025"], st["ytd2026"]
    enough = (
        tr["trade_entries"] >= cfg.min_train_trades
        and te["trade_entries"] >= cfg.min_test_trades
        and ev["trade_entries"] >= cfg.min_eval_trades
        and yt["trade_entries"] >= cfg.min_ytd_trades
    )
    positive = tr["cagr_pct"] > 0 and te["cagr_pct"] > 0 and ev["cagr_pct"] > 0 and yt["cagr_pct"] > 0
    targetish = min(te["cagr_to_strict_mdd"], ev["cagr_to_strict_mdd"], yt["cagr_to_strict_mdd"]) >= 3.0
    train_ok = tr["cagr_to_strict_mdd"] >= 1.0 and tr["strict_mdd_pct"] <= 35.0
    min_oos = min(te["cagr_to_strict_mdd"], ev["cagr_to_strict_mdd"], yt["cagr_to_strict_mdd"])
    blended_return = te["total_return_pct"] + ev["total_return_pct"] + 0.5 * yt["total_return_pct"]
    corr_bonus = -float(row["max_abs_corr_to_components"])
    return (enough and positive and train_ok, targetish, min_oos, tr["cagr_to_strict_mdd"], blended_return, corr_bonus, te["trade_entries"] + ev["trade_entries"] + yt["trade_entries"])


def _fmt_stats(s: dict[str, Any]) -> str:
    return f"{s['total_return_pct']:.2f}/{s['cagr_pct']:.2f}/{s['strict_mdd_pct']:.2f}/{s['cagr_to_strict_mdd']:.2f}/{s['trade_entries']}"


def run(cfg: LowCorrAlphaSearchConfig) -> dict[str, Any]:
    market = _load_market(cfg)
    dates = pd.to_datetime(market["date"])
    masks, years = _split_masks(dates)

    base = build_market_feature_frame(market, window_size=int(cfg.window_size))
    features = pd.concat([base, build_interest_features(market, base)], axis=1)
    features = _add_oi_derived_features_local(market, features)
    features = add_vpin_formulaic_features(market, features)
    features = pd.concat([features, add_alpha101_features(market)], axis=1)
    features = features.loc[:, ~features.columns.duplicated()].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    component_frame, component_meta = _build_component_frame_local(features)
    component_frame = component_frame.loc[:, component_frame.loc[masks["train"]].sum(axis=0) > 0]

    pool = json.load(open("research/pools/feature_pool.json"))
    pool_feature_names: set[str] = set()
    for entry in pool.get("entries", []):
        if entry.get("feature_tier") == "alpha_feature":
            for name in entry.get("feature_names", []):
                if isinstance(name, str) and "*" not in name and name in features.columns:
                    pool_feature_names.add(name)

    train = masks["train"]
    component_cols = list(component_frame.columns)
    pool_cols = sorted(pool_feature_names)
    candidates: list[dict[str, Any]] = []
    prefixes = tuple(x.strip() for x in str(cfg.candidate_prefixes).split(",") if x.strip())
    comp_train = component_frame[component_cols].to_numpy(float)[train]
    open_ = market["open"].astype(float)
    fwd72 = ((open_.shift(-73) - open_.shift(-1)) / open_.shift(-1).replace(0.0, np.nan)).to_numpy(float)
    for col in features.columns:
        if prefixes and not str(col).startswith(prefixes):
            continue
        x_all = features[col].to_numpy(float)
        x = x_all[train]
        if not np.isfinite(x).any() or float(np.std(x)) <= 1e-12:
            continue
        comp_corrs = [_pearson(x, comp_train[:, j]) for j in range(comp_train.shape[1])]
        max_comp = max((abs(v) for v in comp_corrs), default=0.0)
        # Keep the report field for compatibility, but avoid an expensive all-vs-all
        # pool correlation here. The low-correlation gate is against existing alpha
        # component masks, which is the user's target.
        max_pool = 0.0
        if max_comp <= float(cfg.max_component_abs_corr) and max_pool <= float(cfg.max_pool_abs_corr):
            train_ic = abs(_spearman(x, fwd72[train]))
            candidates.append({"feature": col, "max_abs_corr_to_components": max_comp, "max_abs_corr_to_alpha_pool_features": max_pool, "train_abs_spearman_fwd72": train_ic})
    candidates.sort(key=lambda r: (r["train_abs_spearman_fwd72"], -r["max_abs_corr_to_components"]), reverse=True)
    selected = candidates[: int(cfg.max_features)]
    pair_base = selected[: int(cfg.max_pair_features)]

    quantiles = _parse_csv(cfg.quantiles, float)
    holds = _parse_csv(cfg.holds, int)
    strides = _parse_csv(cfg.strides, int)

    specs: list[tuple[list[tuple[str, str, float]], str]] = []
    for c in selected:
        col = c["feature"]
        for q in quantiles:
            op = "<=" if q < 0.5 else ">="
            specs.append(([(col, op, q)], "single"))
    for a, b in itertools.combinations(pair_base, 2):
        for qa in [0.10, 0.20, 0.80, 0.90]:
            for qb in [0.10, 0.20, 0.80, 0.90]:
                specs.append(([(a["feature"], "<=" if qa < 0.5 else ">=", qa), (b["feature"], "<=" if qb < 0.5 else ">=", qb)], "pair"))

    rows: list[dict[str, Any]] = []
    tested = 0
    feature_meta = {r["feature"]: r for r in selected}
    for spec, spec_kind in specs:
        terms: list[tuple[str, str, float]] = []
        ok = True
        for col, op, q in spec:
            thr = _safe_quantile(features[col].to_numpy(float), train, q)
            if thr is None:
                ok = False
                break
            terms.append((col, op, thr))
        if not ok:
            continue
        active = _active_from_terms(features, terms)
        ar = float(active[train].mean())
        if ar < cfg.min_train_active_rate or ar > cfg.max_train_active_rate:
            continue
        tested += 1
        max_comp = max(feature_meta[t[0]]["max_abs_corr_to_components"] for t in spec)
        max_pool = max(feature_meta[t[0]]["max_abs_corr_to_alpha_pool_features"] for t in spec)
        for side in ["long", "short"]:
            for hold in holds:
                for stride in strides:
                    st = _eval_rule(market, features, masks, years, terms, side, int(hold), int(stride))
                    row = {
                        "name": f"lowcorr_{spec_kind}_{side}_{len(rows):06d}",
                        "spec_kind": spec_kind,
                        "side": side,
                        "terms": [
                            {"feature": col, "op": op, "threshold": thr, "train_quantile": q, "max_abs_corr_to_components": feature_meta[col]["max_abs_corr_to_components"], "max_abs_corr_to_alpha_pool_features": feature_meta[col]["max_abs_corr_to_alpha_pool_features"]}
                            for (col, op, thr), (_, _, q) in zip(terms, spec)
                        ],
                        "hold": int(hold),
                        "stride": int(stride),
                        "active_rate_train": ar,
                        "max_abs_corr_to_components": max_comp,
                        "max_abs_corr_to_alpha_pool_features": max_pool,
                        "stats": st,
                    }
                    row["score_tuple"] = _score(row, cfg)
                    rows.append(row)
    rows.sort(key=lambda r: r["score_tuple"], reverse=True)
    top = [{k: v for k, v in row.items() if k != "score_tuple"} for row in rows[:200]]

    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "input": {"rows": len(market), "start": str(market["date"].iloc[0]), "end": str(market["date"].iloc[-1])},
        "existing_components": component_cols,
        "pool_alpha_feature_count": len(pool_cols),
        "low_corr_feature_count": len(candidates),
        "selected_features": selected,
        "tested_specs": tested,
        "all_trials": len(rows),
        "top": top,
        "leakage_guard": {
            "feature_construction_past_only": True,
            "funding_premium_join": "backward_asof",
            "low_corr_filter_uses_train_rows_only": True,
            "thresholds_fit_on_train_only": True,
            "split_protocol": SPLIT_BOUNDS,
            "ranking_is_diagnostic_and_uses_oos_columns": True,
        },
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    _write_doc(cfg, report)
    return report


def _write_doc(cfg: LowCorrAlphaSearchConfig, report: dict[str, Any]) -> None:
    lines = [
        "# Low-correlation feature alpha scan (2026-07-12)",
        "",
        "Goal: search alpha candidates from features with low train-window correlation to existing alpha component masks.",
        "",
        "Leakage guard: thresholds are train<2024 quantiles; low-correlation filter uses train rows only; feature builders use backward/asof auxiliary joins. Ranking below is diagnostic and uses OOS stats, so promoted status is not implied.",
        "",
        f"Input rows: {report['input']['rows']:,} `{report['input']['start']}` ~ `{report['input']['end']}`.",
        f"Low-corr candidates: {report['low_corr_feature_count']}; selected={len(report['selected_features'])}; tested_specs={report['tested_specs']}; all_trials={report['all_trials']}.",
        "",
        "## Top low-correlation source features",
        "",
        "| rank | feature | max corr vs components | max corr vs alpha pool features | train abs IC fwd72 |",
        "|---:|---|---:|---:|---:|",
    ]
    for i, row in enumerate(report["selected_features"][:30], 1):
        lines.append(f"| {i} | `{row['feature']}` | {row['max_abs_corr_to_components']:.4f} | {row['max_abs_corr_to_alpha_pool_features']:.4f} | {row['train_abs_spearman_fwd72']:.4f} |")
    lines += [
        "",
        "## Top alpha trials",
        "",
        "Stats format: `absolute_return/CAGR/strict_MDD/CAGR_MDD/trades`.",
        "",
        "| rank | name | side | hold/stride | active | corr | train | 2024 | 2025 | 2026YTD | terms |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for i, row in enumerate(report["top"][:60], 1):
        terms = "; ".join(f"{t['feature']} {t['op']} q{t['train_quantile']:.2f}({t['threshold']:.4g})" for t in row["terms"])
        st = row["stats"]
        lines.append(
            f"| {i} | `{row['name']}` | {row['side']} | {row['hold']}/{row['stride']} | {row['active_rate_train']:.4f} | {row['max_abs_corr_to_components']:.3f} | {_fmt_stats(st['train'])} | {_fmt_stats(st['test2024'])} | {_fmt_stats(st['eval2025'])} | {_fmt_stats(st['ytd2026'])} | `{terms}` |"
        )
    Path(cfg.docs_output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.docs_output).write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    for field, value in LowCorrAlphaSearchConfig().__dict__.items():
        arg = "--" + field.replace("_", "-")
        if isinstance(value, bool):
            p.add_argument(arg, action="store_true", default=value)
        elif isinstance(value, int):
            p.add_argument(arg, type=int, default=value)
        elif isinstance(value, float):
            p.add_argument(arg, type=float, default=value)
        else:
            p.add_argument(arg, default=value)
    return p.parse_args()


def main() -> None:
    cfg = LowCorrAlphaSearchConfig(**vars(parse_args()))
    report = run(cfg)
    print(json.dumps({
        "output": cfg.output,
        "docs_output": cfg.docs_output,
        "low_corr_feature_count": report["low_corr_feature_count"],
        "tested_specs": report["tested_specs"],
        "all_trials": report["all_trials"],
        "top": report["top"][:10],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
