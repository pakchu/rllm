"""Gate the fixed funding/premium squeeze with six-alt derivative crowding.

This is the clean-room long-only version of the alt derivatives search: BTC
market/funding/premium and all six alt funding/premium sources are physically
truncated before the Top-10 manifest is selected.  Alt feature thresholds are
fit on 2023-02-15..2023-07-01, policies are selected only on 2023H2, and the
frozen manifest is replayed unchanged on 2024+ windows.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from preprocessing.binance_aux_features import (
    attach_binance_um_aux_frames,
    normalise_funding_history_frame,
    normalise_premium_index_frame,
)
from training.search_alt_derivatives_crowding_alpha import (
    BASE_FEATURES,
    QUARTER_WINDOWS,
    SELECTION_END,
    SYMBOLS,
    AltCrowdingConfig,
    _latest_source,
    _merge_source,
    _naive_utc,
    _parse_csv,
    build_alt_features,
    build_base_features,
    feature_admission,
)
from training.search_funding_premium_independent_gate_alpha import (
    HOLD_BARS,
    STRIDE_BARS,
    _apply_gate,
    _build_base_components,
    _gate_mask,
)
from training.search_positioning_disagreement_alpha import _future_extreme, _simulate_no_stop
from training.search_positioning_hgb_path_alpha import _feature_hash, _read_before


FIT_WINDOW = ("2023-02-15", "2023-07-01")
WINDOWS = {
    "fit_2023_h1": FIT_WINDOW,
    "select_2023_h2": ("2023-07-01", "2024-01-01"),
    "test_2024": ("2024-01-01", "2025-01-01"),
    "eval_2025": ("2025-01-01", "2026-01-01"),
    "holdout_2026": ("2026-01-01", "2026-06-02"),
    "oos_2024_2026": ("2024-01-01", "2026-06-02"),
}
ALT_FEATURE_AVAILABILITY = {
    "alt_funding_median_z2016": "alt_funding_available",
    "alt_funding_median_z8640": "alt_funding_available",
    "alt_funding_dispersion_z2016": "alt_funding_available",
    "alt_funding_dispersion_z8640": "alt_funding_available",
    "alt_funding_positive_breadth_z2016": "alt_funding_available",
    "alt_funding_negative_breadth_z2016": "alt_funding_available",
    "alt_funding_median_change288_z2016": "alt_funding_available",
    "alt_premium_median_z2016": "alt_premium_available",
    "alt_premium_median_z8640": "alt_premium_available",
    "alt_premium_dispersion_z2016": "alt_premium_available",
    "alt_premium_dispersion_z8640": "alt_premium_available",
    "alt_premium_positive_breadth_z2016": "alt_premium_available",
    "alt_premium_negative_breadth_z2016": "alt_premium_available",
    "alt_premium_median_change288_z2016": "alt_premium_available",
    "alt_crowding_concordance": "alt_derivatives_available",
    "alt_crowding_disagreement": "alt_derivatives_available",
}


@dataclass(frozen=True)
class AltCrowdingGateConfig(AltCrowdingConfig):
    top_per_feature: int = 2
    min_fit_trades: int = 12
    min_select_trades: int = 10


def _window_mask(dates: pd.Series, name: str) -> np.ndarray:
    start, end = WINDOWS[name]
    return ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy(bool)


def _frame_hash(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update("\n".join(map(str, frame.columns)).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype="<u8").tobytes())
    return digest.hexdigest()


def _read_premium_before(path: str | Path, cutoff: str) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    boundary = pd.Timestamp(cutoff)
    for chunk in pd.read_csv(path, compression="infer", chunksize=100_000):
        close_time = _naive_utc(chunk["close_time"], milliseconds=True)
        keep = close_time < boundary
        if keep.any():
            chunks.append(chunk.loc[keep].copy())
        if (~keep).any():
            break
    if not chunks:
        raise ValueError(f"no premium rows before {cutoff} in {path}")
    return pd.concat(chunks, ignore_index=True)


def _read_source_frames(
    cfg: AltCrowdingGateConfig,
    cutoff: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    frames = {
        "market": _read_before(cfg.input_csv, "date", cutoff),
        "btc_funding": _read_before(cfg.btc_funding_csv, "date", cutoff),
        "btc_premium": _read_premium_before(cfg.btc_premium_csv, cutoff),
    }
    for symbol, kind in itertools.product(SYMBOLS, ("funding", "premium_1h")):
        path = _latest_source(cfg.aux_dir, symbol, kind)
        key = f"{symbol.lower()}_{kind}"
        frames[key] = (
            _read_before(path, "date", cutoff)
            if kind == "funding"
            else _read_premium_before(path, cutoff)
        )
    return frames, {key: _frame_hash(frame) for key, frame in frames.items()}


def _source_prefix_hashes(cfg: AltCrowdingGateConfig, cutoff: str = SELECTION_END) -> dict[str, str]:
    return _read_source_frames(cfg, cutoff)[1]


def _attach_alt_frames(
    market: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    cfg: AltCrowdingGateConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(market["date"]).reset_index(drop=True)
    values: dict[str, pd.Series] = {}
    source_times: dict[str, pd.Series] = {}
    for symbol in SYMBOLS:
        prefix = symbol.removesuffix("USDT").lower()
        funding_value, funding_source = _merge_source(
            dates,
            frames[f"{symbol.lower()}_funding"],
            source_time="date",
            value_column="funding_rate",
            tolerance=cfg.funding_tolerance,
        )
        premium_value, premium_source = _merge_source(
            dates,
            frames[f"{symbol.lower()}_premium_1h"],
            source_time="close_time",
            value_column="close",
            tolerance=cfg.premium_tolerance,
            milliseconds=True,
        )
        values[f"{prefix}_funding"] = funding_value
        values[f"{prefix}_premium"] = premium_value
        source_times[f"{prefix}_funding_source_time"] = funding_source
        source_times[f"{prefix}_premium_source_time"] = premium_source
    attached = pd.DataFrame(values, index=market.index)
    attached["alt_funding_available"] = attached.filter(like="_funding").notna().all(axis=1).astype(float)
    attached["alt_premium_available"] = attached.filter(like="_premium").notna().all(axis=1).astype(float)
    attached["alt_derivatives_available"] = (
        (attached["alt_funding_available"] > 0.5)
        & (attached["alt_premium_available"] > 0.5)
    ).astype(float)
    return attached, pd.DataFrame(source_times, index=market.index)


def _availability_frame(attached: pd.DataFrame) -> pd.DataFrame:
    return attached[["alt_funding_available", "alt_premium_available", "alt_derivatives_available"]].copy()


def _load_bundle(
    cfg: AltCrowdingGateConfig,
    *,
    cutoff: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    frames, source_prefix_hashes = _read_source_frames(cfg, cutoff)
    market = frames["market"].copy()
    market["date"] = _naive_utc(market["date"])
    market = market.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    market = attach_binance_um_aux_frames(
        market,
        funding_frame=normalise_funding_history_frame(frames["btc_funding"]),
        premium_frame=normalise_premium_index_frame(frames["btc_premium"]),
        funding_tolerance=cfg.funding_tolerance,
        premium_tolerance=cfg.premium_tolerance,
    )
    dates = pd.to_datetime(market["date"])
    if len(dates) and dates.max() >= pd.Timestamp(cutoff):
        raise RuntimeError("market rows were not physically truncated before cutoff")
    attached, source_times = _attach_alt_frames(market, frames, cfg)
    for column in source_times:
        valid = source_times[column].notna()
        if (source_times.loc[valid, column] >= pd.Timestamp(cutoff)).any():
            raise RuntimeError(f"alt source rows were not physically truncated before cutoff: {column}")
    features = build_alt_features(attached)
    base_features = build_base_features(market)
    return market, dates, features, base_features, _availability_frame(attached), source_prefix_hashes


def _fit_threshold(values: np.ndarray, fit_mask: np.ndarray, quantile: float, *, min_observations: int) -> float:
    reference = values[fit_mask & np.isfinite(values)]
    if len(reference) < min_observations:
        raise ValueError(f"insufficient observations for threshold: {len(reference)}")
    return float(np.quantile(reference, quantile))


def _gate_specs(features: pd.DataFrame, admitted: Iterable[str], fit_mask: np.ndarray, quantiles: list[float], *, min_observations: int) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for feature, tail, gate_mode, target_component in itertools.product(
        admitted,
        quantiles,
        ("lower", "upper", "central", "outer"),
        ("all", "funding", "premium"),
    ):
        values = features[feature].to_numpy(float)
        specs.append(
            {
                "feature": feature,
                "tail": float(tail),
                "lower": _fit_threshold(values, fit_mask, tail, min_observations=min_observations),
                "upper": _fit_threshold(values, fit_mask, 1.0 - tail, min_observations=min_observations),
                "gate_mode": gate_mode,
                "target_component": target_component,
            }
        )
    return specs


def alt_gate_mask(features: pd.DataFrame, spec: dict[str, Any], availability_frame: pd.DataFrame, availability: dict[str, str] = ALT_FEATURE_AVAILABILITY) -> np.ndarray:
    feature = spec["feature"]
    if feature not in availability:
        raise RuntimeError(f"missing availability mapping for feature: {feature}")
    gate = _gate_mask(features[feature].to_numpy(float), spec)
    available = pd.to_numeric(availability_frame[availability[feature]], errors="coerce").fillna(0.0).to_numpy(float) > 0.5
    return gate & available


def _activation_hash(active: np.ndarray) -> str:
    return hashlib.sha256(np.packbits(np.asarray(active, dtype=bool)).tobytes()).hexdigest()


def _simulate(market: pd.DataFrame, dates: pd.Series, active: np.ndarray, cfg: AltCrowdingGateConfig, *, window: str, extremes: tuple[np.ndarray, np.ndarray], windows: dict[str, tuple[str, str]] = WINDOWS) -> dict[str, Any]:
    return _simulate_no_stop(
        market,
        dates,
        active,
        np.zeros(len(active), dtype=bool),
        window=window,
        hold_bars=HOLD_BARS,
        stride_bars=STRIDE_BARS,
        leverage=cfg.leverage,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
        extremes=extremes,
        windows=windows,
    )


def _selection_score(stats: dict[str, dict[str, Any]], cfg: AltCrowdingGateConfig) -> float:
    fit = stats["fit_2023_h1"]
    select = stats["select_2023_h2"]
    if fit["trades"] < cfg.min_fit_trades or select["trades"] < cfg.min_select_trades:
        return -1e12
    if min(fit["cagr_pct"], select["cagr_pct"]) <= 0.0:
        return -1e12
    if fit["strict_mdd_pct"] > 25.0 or select["strict_mdd_pct"] > 20.0:
        return -1e12
    ratios = np.asarray([fit["ratio"], select["ratio"]], dtype=float)
    return float(np.min(ratios) + 0.35 * np.median(ratios) + min(0.25, select["trades"] / 100.0))


def _select_top(rows: list[dict[str, Any]], *, top_n: int, top_per_feature: int) -> list[dict[str, Any]]:
    rows = sorted(
        rows,
        key=lambda row: (
            row["selection_score"],
            row["selection_stats"]["select_2023_h2"]["ratio"],
            row["selection_stats"]["select_2023_h2"]["return_pct"],
            row["feature"],
            row["gate_mode"],
            row["target_component"],
        ),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for row in rows:
        feature = row["feature"]
        if counts.get(feature, 0) >= top_per_feature:
            continue
        selected.append(row)
        counts[feature] = counts.get(feature, 0) + 1
        if len(selected) >= top_n:
            break
    return selected


def _manifest_core_hash(core: dict[str, Any]) -> str:
    canonical = json.dumps(core, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _validate_manifest(manifest: dict[str, Any]) -> None:
    core = {key: value for key, value in manifest.items() if key not in {"as_of", "sha256"}}
    if manifest.get("sha256") != _manifest_core_hash(core):
        raise RuntimeError("manifest content does not match its frozen SHA-256")


def _select_manifest(cfg: AltCrowdingGateConfig) -> dict[str, Any]:
    market, dates, features, base_features, availability_frame, source_prefix_hashes = _load_bundle(cfg, cutoff=SELECTION_END)
    fit_mask = _window_mask(dates, "fit_2023_h1")
    admitted, audit = feature_admission(
        features,
        base_features,
        fit_mask,
        max_abs_spearman=cfg.max_abs_spearman,
        min_observations=cfg.min_fit_observations,
    )
    admitted = [feature for feature in admitted if feature in ALT_FEATURE_AVAILABILITY]
    specs = _gate_specs(features, admitted, fit_mask, _parse_csv(cfg.quantiles, float), min_observations=cfg.min_fit_observations)
    funding_component, premium_component = _build_base_components(market)
    extremes = (_future_extreme(market["low"].to_numpy(float), HOLD_BARS, "min"), _future_extreme(market["high"].to_numpy(float), HOLD_BARS, "max"))
    base_active = funding_component | premium_component
    baseline = {window: _simulate(market, dates, base_active, cfg, window=window, extremes=extremes) for window in ("fit_2023_h1", "select_2023_h2")}
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for spec in specs:
        gate = alt_gate_mask(features, spec, availability_frame)
        active = _apply_gate(funding_component, premium_component, gate, spec["target_component"])
        activation_hash = _activation_hash(active)
        if activation_hash in seen:
            continue
        seen.add(activation_hash)
        stats = {window: _simulate(market, dates, active, cfg, window=window, extremes=extremes) for window in ("fit_2023_h1", "select_2023_h2")}
        score = _selection_score(stats, cfg)
        if score <= -1e11:
            continue
        rows.append({**spec, "activation_hash": activation_hash, "selection_score": score, "selection_stats": stats})
    selected = _select_top(rows, top_n=cfg.top_n, top_per_feature=cfg.top_per_feature)
    core = {
        "protocol": {
            "base_alpha": "fixed funding10_trend70 OR premium20_mom90 long",
            "feature_threshold_fit": FIT_WINDOW,
            "policy_selection": WINDOWS["select_2023_h2"],
            "all_future_market_btc_and_six_alt_source_rows_physically_excluded_before_manifest": True,
            "gate_sources": "six-alt Binance USD-M funding and premium-index crowding features only",
            "feature_admission": f"fit Spearman max |rho| < {cfg.max_abs_spearman} versus {BASE_FEATURES}",
            "entry": "next 5m open",
            "exit": f"fixed {HOLD_BARS} bars; stride {STRIDE_BARS}; no TP/SL",
            "cost": "5bp fee + 1bp slippage per side at 0.5x",
            "mdd": "strict favorable-high-water then adverse OHLC extreme",
            "status_ceiling": "shadow research: 2024+ has been inspected by the broader program",
        },
        "source_prefix_hashes": source_prefix_hashes,
        "external_feature_hash": _feature_hash(features, dates),
        "base_feature_hash": _feature_hash(base_features, dates),
        "availability_hash": _feature_hash(availability_frame, dates),
        "feature_availability": ALT_FEATURE_AVAILABILITY,
        "feature_admission_audit": audit,
        "search_space": {"admitted_features": admitted, "raw_specs": len(specs), "effective_unique_masks": len(seen), "eligible_variants": len(rows), "top_n": cfg.top_n, "top_per_feature": cfg.top_per_feature},
        "baseline_selection_stats": baseline,
        "selected": selected,
    }
    manifest = {"as_of": datetime.now(timezone.utc).isoformat(), "sha256": _manifest_core_hash(core), **core}
    path = Path(cfg.manifest_output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=True) + "\n")
    return manifest


def _replay(cfg: AltCrowdingGateConfig, manifest: dict[str, Any]) -> dict[str, Any]:
    _validate_manifest(manifest)
    if _source_prefix_hashes(cfg, SELECTION_END) != manifest["source_prefix_hashes"]:
        raise RuntimeError("pre-2024 source prefixes changed after manifest freeze")
    market, dates, features, base_features, availability_frame, _ = _load_bundle(cfg, cutoff=cfg.exclude_from)
    prefix = dates < pd.Timestamp(SELECTION_END)
    prefix_dates = dates.loc[prefix].reset_index(drop=True)
    if _feature_hash(features.loc[prefix].reset_index(drop=True), prefix_dates) != manifest["external_feature_hash"]:
        raise RuntimeError("pre-2024 external feature prefix changed during full replay")
    if _feature_hash(base_features.loc[prefix].reset_index(drop=True), prefix_dates) != manifest["base_feature_hash"]:
        raise RuntimeError("pre-2024 BTC admission-feature prefix changed during full replay")
    if _feature_hash(availability_frame.loc[prefix].reset_index(drop=True), prefix_dates) != manifest["availability_hash"]:
        raise RuntimeError("pre-2024 external availability prefix changed during full replay")
    if manifest.get("feature_availability") != ALT_FEATURE_AVAILABILITY:
        raise RuntimeError("feature availability mapping changed during replay")

    funding_component, premium_component = _build_base_components(market)
    extremes = (_future_extreme(market["low"].to_numpy(float), HOLD_BARS, "min"), _future_extreme(market["high"].to_numpy(float), HOLD_BARS, "max"))
    base_active = funding_component | premium_component
    baseline = {window: _simulate(market, dates, base_active, cfg, window=window, extremes=extremes) for window in WINDOWS}
    selected: list[dict[str, Any]] = []
    for rank, frozen in enumerate(manifest["selected"], start=1):
        spec = {key: frozen[key] for key in ("feature", "tail", "lower", "upper", "gate_mode", "target_component")}
        gate = alt_gate_mask(features, spec, availability_frame)
        active = _apply_gate(funding_component, premium_component, gate, spec["target_component"])
        if _activation_hash(active[prefix.to_numpy(bool)]) != frozen["activation_hash"]:
            raise RuntimeError(f"pre-2024 activation drift at rank {rank}")
        stats = {window: _simulate(market, dates, active, cfg, window=window, extremes=extremes) for window in WINDOWS}
        for window in ("fit_2023_h1", "select_2023_h2"):
            if stats[window] != frozen["selection_stats"][window]:
                raise RuntimeError(f"selection replay drift rank={rank} window={window}")
        quarterly = {window: _simulate(market, dates, active, cfg, window=window, extremes=extremes, windows=QUARTER_WINDOWS) for window in QUARTER_WINDOWS}
        test, evaluation, holdout, combined = stats["test_2024"], stats["eval_2025"], stats["holdout_2026"], stats["oos_2024_2026"]
        enough = test["trades"] >= 20 and evaluation["trades"] >= 20 and holdout["trades"] >= 10
        passes_alpha_pool = enough and test["ratio"] >= 2.5 and evaluation["ratio"] >= 2.5
        passes_live_grade = passes_alpha_pool and holdout["ratio"] >= 3.0 and combined["ratio"] >= 3.0 and combined["p_value_mean_return_approx"] <= 0.05
        selected.append({"manifest_rank": rank, **frozen, "stats": stats, "quarterly_stats": quarterly, "quarterly_summary": {"positive_return_quarters": sum(row["return_pct"] > 0.0 for row in quarterly.values()), "flat_quarters": sum(row["trades"] == 0 for row in quarterly.values()), "negative_return_quarters": sum(row["return_pct"] < 0.0 for row in quarterly.values()), "total_quarters": len(quarterly)}, "passes_alpha_pool": bool(passes_alpha_pool), "passes_live_grade": bool(passes_live_grade)})
    return {"as_of": datetime.now(timezone.utc).isoformat(), "config": asdict(cfg), "manifest": cfg.manifest_output, "manifest_sha256": manifest["sha256"], "protocol": manifest["protocol"], "feature_admission_audit": manifest["feature_admission_audit"], "baseline": baseline, "selected": selected, "alpha_pool_qualifiers": [row for row in selected if row["passes_alpha_pool"]], "live_grade": [row for row in selected if row["passes_live_grade"]]}


def run(cfg: AltCrowdingGateConfig) -> dict[str, Any]:
    manifest_path = Path(cfg.manifest_output)
    if manifest_path.exists() and not cfg.refresh_manifest:
        manifest = json.loads(manifest_path.read_text())
        _validate_manifest(manifest)
    else:
        manifest = _select_manifest(cfg)
    report = _replay(cfg, manifest)
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True) + "\n")
    return report


def parse_args() -> AltCrowdingGateConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--aux-dir", required=True)
    parser.add_argument("--btc-funding-csv", required=True)
    parser.add_argument("--btc-premium-csv", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--exclude-from", default=AltCrowdingGateConfig.exclude_from)
    parser.add_argument("--quantiles", default=AltCrowdingGateConfig.quantiles)
    parser.add_argument("--top-n", type=int, default=AltCrowdingGateConfig.top_n)
    parser.add_argument("--top-per-feature", type=int, default=AltCrowdingGateConfig.top_per_feature)
    parser.add_argument("--max-abs-spearman", type=float, default=AltCrowdingGateConfig.max_abs_spearman)
    parser.add_argument("--refresh-manifest", action="store_true")
    return AltCrowdingGateConfig(**vars(parser.parse_args()))


def main() -> None:
    report = run(parse_args())
    print(json.dumps({"manifest_sha256": report["manifest_sha256"], "selected": len(report["selected"]), "alpha_pool_qualifiers": len(report["alpha_pool_qualifiers"]), "live_grade": len(report["live_grade"]), "top": report["selected"][:3]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
