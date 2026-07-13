"""Search a standalone BTC alpha from alt-perpetual crowding states.

The signal surface uses only causally available funding and premium-index data
for ETH/SOL/BNB/XRP/ADA/DOGE perpetuals.  Feature thresholds are fitted on
2023H1, candidate policies are selected on 2023H2, and a Top-10 manifest is
written before 2024+ performance is evaluated.  Candidate features whose fit
Spearman correlation with the registered BTC base features reaches 0.30 are
rejected before the search.
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
    load_funding_history,
    load_premium_index_klines,
)
from preprocessing.market_features import _completed_timeframe_features
from training.search_liquidity_recovery_bidirectional_alpha import features as build_liquidity_features
from training.search_positioning_disagreement_alpha import _future_extreme, _simulate_no_stop
from training.search_positioning_hgb_path_alpha import _feature_hash, _read_before


SYMBOLS = ("ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT")
SELECTION_END = "2024-01-01"
HOLD_BARS = (72, 144, 288, 576)
STRIDE_BARS = (12, 24)

WINDOWS = {
    "fit_2023_h1": ("2023-02-15", "2023-07-01"),
    "select_2023_h2": ("2023-07-01", "2024-01-01"),
    "test_2024": ("2024-01-01", "2025-01-01"),
    "eval_2025": ("2025-01-01", "2026-01-01"),
    "holdout_2026": ("2026-01-01", "2026-06-02"),
    "oos_2024_2026": ("2024-01-01", "2026-06-02"),
}

QUARTER_WINDOWS = {
    "2024Q1": ("2024-01-01", "2024-04-01"),
    "2024Q2": ("2024-04-01", "2024-07-01"),
    "2024Q3": ("2024-07-01", "2024-10-01"),
    "2024Q4": ("2024-10-01", "2025-01-01"),
    "2025Q1": ("2025-01-01", "2025-04-01"),
    "2025Q2": ("2025-04-01", "2025-07-01"),
    "2025Q3": ("2025-07-01", "2025-10-01"),
    "2025Q4": ("2025-10-01", "2026-01-01"),
    "2026Q1": ("2026-01-01", "2026-04-01"),
    "2026Q2_to_Jun02": ("2026-04-01", "2026-06-02"),
}

BASE_FEATURES = (
    "btc_funding_rate",
    "btc_premium_change",
    "btc_trend_96",
    "btc_daily_momentum_4",
    "btc_lr_impact_72",
)


@dataclass(frozen=True)
class AltCrowdingConfig:
    input_csv: str
    aux_dir: str
    btc_funding_csv: str
    btc_premium_csv: str
    output: str
    manifest_output: str
    exclude_from: str = "2026-06-02"
    leverage: float = 0.5
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0001
    quantiles: str = "0.10,0.20,0.30"
    top_n: int = 10
    top_per_rule: int = 2
    min_fit_observations: int = 20_000
    min_fit_trades: int = 12
    min_select_trades: int = 16
    max_abs_spearman: float = 0.30
    funding_tolerance: str = "12h"
    premium_tolerance: str = "65min"
    refresh_manifest: bool = False


def _parse_csv(raw: str, cast: Any) -> list[Any]:
    return [cast(part.strip()) for part in str(raw).split(",") if part.strip()]


def _naive_utc(values: pd.Series, *, milliseconds: bool = False) -> pd.Series:
    if milliseconds:
        return pd.to_datetime(pd.to_numeric(values, errors="raise"), unit="ms", utc=True).dt.tz_convert(None)
    return pd.to_datetime(values, utc=True, errors="raise", format="mixed").dt.tz_convert(None)


def _merge_source(
    dates: pd.Series,
    source: pd.DataFrame,
    *,
    source_time: str,
    value_column: str,
    tolerance: str,
    milliseconds: bool = False,
) -> tuple[pd.Series, pd.Series]:
    """Backward-as-of one external source and retain its availability time."""
    left = pd.DataFrame(
        {
            "date": pd.to_datetime(dates).reset_index(drop=True),
            "_source_row": np.arange(len(dates), dtype=np.int64),
        }
    )
    right = source[[source_time, value_column]].copy()
    right["source_time"] = _naive_utc(right[source_time], milliseconds=milliseconds)
    right["source_value"] = pd.to_numeric(right[value_column], errors="coerce")
    right = right[["source_time", "source_value"]].dropna().sort_values("source_time")
    joined = pd.merge_asof(
        left.sort_values("date"),
        right,
        left_on="date",
        right_on="source_time",
        direction="backward",
        tolerance=pd.Timedelta(tolerance),
    )
    available = joined["source_time"].notna()
    if (joined.loc[available, "source_time"] > joined.loc[available, "date"]).any():
        raise RuntimeError("external derivative value was visible before source_time")
    joined = joined.sort_values("_source_row").reset_index(drop=True)
    return joined["source_value"], joined["source_time"]


def _latest_source(aux_dir: str, symbol: str, kind: str) -> Path:
    paths = sorted(Path(aux_dir).glob(f"{symbol}_{kind}_*.csv.gz"))
    if not paths:
        raise FileNotFoundError(f"missing {kind} file for {symbol} in {aux_dir}")
    return paths[-1]


def _source_hashes(cfg: AltCrowdingConfig) -> dict[str, str]:
    hashes: dict[str, str] = {}
    paths = [
        _latest_source(cfg.aux_dir, symbol, kind)
        for symbol, kind in itertools.product(SYMBOLS, ("funding", "premium_1h"))
    ]
    paths.extend((Path(cfg.btc_funding_csv), Path(cfg.btc_premium_csv)))
    paths.append(Path(cfg.input_csv))
    for path in paths:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        hashes[str(path)] = digest.hexdigest()
    return hashes


def _load_market(cfg: AltCrowdingConfig, cutoff: str) -> pd.DataFrame:
    market = _read_before(cfg.input_csv, "date", cutoff)
    market["date"] = _naive_utc(market["date"])
    market = market.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    boundary = pd.Timestamp(cutoff)
    funding = load_funding_history(cfg.btc_funding_csv)
    funding = funding.loc[funding["date"] < boundary].copy()
    premium = load_premium_index_klines(cfg.btc_premium_csv)
    premium = premium.loc[premium["date"] < boundary].copy()
    return attach_binance_um_aux_frames(
        market,
        funding_frame=funding,
        premium_frame=premium,
        funding_tolerance=cfg.funding_tolerance,
        premium_tolerance=cfg.premium_tolerance,
    )


def attach_alt_derivatives(
    market: pd.DataFrame,
    cfg: AltCrowdingConfig,
    *,
    source_cutoff: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach all six alt funding/premium streams with bounded staleness."""
    dates = pd.to_datetime(market["date"]).reset_index(drop=True)
    values: dict[str, pd.Series] = {}
    source_times: dict[str, pd.Series] = {}
    boundary = pd.Timestamp(source_cutoff)
    for symbol in SYMBOLS:
        prefix = symbol.removesuffix("USDT").lower()
        funding_path = _latest_source(cfg.aux_dir, symbol, "funding")
        funding = pd.read_csv(funding_path, compression="infer")
        funding_dates = _naive_utc(funding["date"])
        funding = funding.loc[funding_dates < boundary].copy()
        funding_value, funding_source = _merge_source(
            dates,
            funding,
            source_time="date",
            value_column="funding_rate",
            tolerance=cfg.funding_tolerance,
        )
        values[f"{prefix}_funding"] = funding_value
        source_times[f"{prefix}_funding_source_time"] = funding_source

        premium_path = _latest_source(cfg.aux_dir, symbol, "premium_1h")
        premium = pd.read_csv(premium_path, compression="infer")
        premium_dates = _naive_utc(premium["close_time"], milliseconds=True)
        premium = premium.loc[premium_dates < boundary].copy()
        premium_value, premium_source = _merge_source(
            dates,
            premium,
            source_time="close_time",
            value_column="close",
            tolerance=cfg.premium_tolerance,
            milliseconds=True,
        )
        values[f"{prefix}_premium"] = premium_value
        source_times[f"{prefix}_premium_source_time"] = premium_source
    frame = pd.DataFrame(values, index=market.index)
    sources = pd.DataFrame(source_times, index=market.index)
    frame["alt_funding_available"] = frame.filter(like="_funding").notna().all(axis=1).astype(float)
    frame["alt_premium_available"] = frame.filter(like="_premium").notna().all(axis=1).astype(float)
    frame["alt_derivatives_available"] = (
        (frame["alt_funding_available"] > 0.5) & (frame["alt_premium_available"] > 0.5)
    ).astype(float)
    return frame, sources


def _rolling_z(values: pd.Series, window: int) -> pd.Series:
    minimum = max(288, window // 2)
    mean = values.rolling(window, min_periods=minimum).mean()
    std = values.rolling(window, min_periods=minimum).std(ddof=0).replace(0.0, np.nan)
    return ((values - mean) / std).replace([np.inf, -np.inf], np.nan)


def build_alt_features(attached: pd.DataFrame) -> pd.DataFrame:
    funding = attached[[f"{symbol.removesuffix('USDT').lower()}_funding" for symbol in SYMBOLS]]
    premium = attached[[f"{symbol.removesuffix('USDT').lower()}_premium" for symbol in SYMBOLS]]
    funding_ok = attached["alt_funding_available"] > 0.5
    premium_ok = attached["alt_premium_available"] > 0.5
    raw = {
        "funding_median": funding.median(axis=1).where(funding_ok),
        "funding_dispersion": funding.std(axis=1, ddof=0).where(funding_ok),
        "funding_positive_breadth": (funding > 0.0).mean(axis=1).where(funding_ok),
        "funding_negative_breadth": (funding < 0.0).mean(axis=1).where(funding_ok),
        "premium_median": premium.median(axis=1).where(premium_ok),
        "premium_dispersion": premium.std(axis=1, ddof=0).where(premium_ok),
        "premium_positive_breadth": (premium > 0.0).mean(axis=1).where(premium_ok),
        "premium_negative_breadth": (premium < 0.0).mean(axis=1).where(premium_ok),
    }
    features: dict[str, pd.Series] = {}
    for family in ("funding", "premium"):
        for name in ("median", "dispersion"):
            source = raw[f"{family}_{name}"]
            for window in (2016, 8640):
                features[f"alt_{family}_{name}_z{window}"] = _rolling_z(source, window)
        for name in ("positive_breadth", "negative_breadth"):
            features[f"alt_{family}_{name}_z2016"] = _rolling_z(raw[f"{family}_{name}"], 2016)
        change = raw[f"{family}_median"] - raw[f"{family}_median"].shift(288)
        features[f"alt_{family}_median_change288_z2016"] = _rolling_z(change, 2016)

    funding_state = features["alt_funding_median_z2016"]
    premium_state = features["alt_premium_median_z2016"]
    features["alt_crowding_concordance"] = funding_state + premium_state
    features["alt_crowding_disagreement"] = funding_state - premium_state
    features["alt_derivatives_available"] = attached["alt_derivatives_available"].astype(float)
    return pd.DataFrame(features, index=attached.index).replace([np.inf, -np.inf], np.nan).astype(np.float32)


def build_base_features(market: pd.DataFrame) -> pd.DataFrame:
    close = pd.to_numeric(market["close"], errors="coerce")
    daily = _completed_timeframe_features(
        market,
        prefix="htf_1d",
        resample_rule="1D",
        min_source_rows=24 * 60 * 4,
    )
    liquidity = build_liquidity_features(market, pd.DataFrame(index=market.index))
    return pd.DataFrame(
        {
            "btc_funding_rate": pd.to_numeric(market.get("funding_rate"), errors="coerce"),
            "btc_premium_change": pd.to_numeric(market.get("premium_index_change"), errors="coerce"),
            "btc_trend_96": close / close.shift(95).replace(0.0, np.nan) - 1.0,
            "btc_daily_momentum_4": daily["htf_1d_return_4"],
            "btc_lr_impact_72": liquidity["lr_impact_72"],
        },
        index=market.index,
    ).replace([np.inf, -np.inf], np.nan)


def _window_mask(dates: pd.Series, name: str) -> np.ndarray:
    start, end = WINDOWS[name]
    return ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy(bool)


def feature_admission(
    features: pd.DataFrame,
    base_features: pd.DataFrame,
    fit_mask: np.ndarray,
    *,
    max_abs_spearman: float,
    min_observations: int,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    admitted: list[str] = []
    audit: dict[str, dict[str, Any]] = {}
    for feature in features.columns:
        if feature.endswith("_available"):
            continue
        values = pd.to_numeric(features[feature], errors="coerce")
        finite_fit = fit_mask & np.isfinite(values.to_numpy(float))
        correlations: dict[str, float] = {}
        pair_counts: dict[str, int] = {}
        for base in BASE_FEATURES:
            paired = finite_fit & np.isfinite(base_features[base].to_numpy(float))
            pair_counts[base] = int(paired.sum())
            correlations[base] = (
                float(values.loc[paired].corr(base_features.loc[paired, base], method="spearman"))
                if pair_counts[base] >= 100
                else float("nan")
            )
        finite_correlations = [abs(value) for value in correlations.values() if np.isfinite(value)]
        max_abs = max(finite_correlations, default=float("inf"))
        observations = int(finite_fit.sum())
        passes = observations >= min_observations and max_abs < max_abs_spearman
        audit[feature] = {
            "fit_observations": observations,
            "pair_counts": pair_counts,
            "spearman": correlations,
            "max_abs_spearman": float(max_abs),
            "passes": bool(passes),
        }
        if passes:
            admitted.append(feature)
    return admitted, audit


def _fit_threshold(values: np.ndarray, fit_mask: np.ndarray, quantile: float) -> float:
    reference = values[fit_mask & np.isfinite(values)]
    if len(reference) < 1_000:
        raise ValueError(f"insufficient observations for threshold: {len(reference)}")
    return float(np.quantile(reference, quantile))


def _term(feature: str, op: str, quantile: float, threshold: float) -> dict[str, Any]:
    return {"feature": feature, "op": op, "quantile": float(quantile), "threshold": float(threshold)}


def _single_rule_specs(
    features: pd.DataFrame,
    admitted: Iterable[str],
    fit_mask: np.ndarray,
    quantiles: list[float],
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for feature, tail, side, op in itertools.product(admitted, quantiles, ("long", "short"), ("le", "ge")):
        quantile = tail if op == "le" else 1.0 - tail
        threshold = _fit_threshold(features[feature].to_numpy(float), fit_mask, quantile)
        specs.append(
            {
                "rule_name": f"single_{feature}_{op}_{tail:.2f}_{side}",
                "side": side,
                "terms": [_term(feature, op, quantile, threshold)],
            }
        )
    return specs


def _pair_templates() -> list[tuple[str, str, str, str, str]]:
    templates: list[tuple[str, str, str, str, str]] = []
    for first_op, second_op, side in itertools.product(("le", "ge"), ("le", "ge"), ("long", "short")):
        templates.append(("alt_funding_median_z2016", first_op, "alt_premium_median_z2016", second_op, side))
    for family in ("funding", "premium"):
        median = f"alt_{family}_median_z2016"
        dispersion = f"alt_{family}_dispersion_z2016"
        for median_op, side in itertools.product(("le", "ge"), ("long", "short")):
            templates.append((median, median_op, dispersion, "ge", side))
    templates.extend(
        [
            ("alt_funding_dispersion_z2016", "ge", "alt_premium_dispersion_z2016", "ge", "long"),
            ("alt_funding_dispersion_z2016", "ge", "alt_premium_dispersion_z2016", "ge", "short"),
            ("alt_funding_negative_breadth_z2016", "ge", "alt_premium_negative_breadth_z2016", "ge", "long"),
            ("alt_funding_positive_breadth_z2016", "ge", "alt_premium_positive_breadth_z2016", "ge", "short"),
        ]
    )
    return templates


def _pair_rule_specs(
    features: pd.DataFrame,
    admitted: set[str],
    fit_mask: np.ndarray,
    quantiles: list[float],
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for first, first_op, second, second_op, side in _pair_templates():
        if first not in admitted or second not in admitted:
            continue
        for tail in quantiles:
            terms = []
            for feature, op in ((first, first_op), (second, second_op)):
                quantile = tail if op == "le" else 1.0 - tail
                threshold = _fit_threshold(features[feature].to_numpy(float), fit_mask, quantile)
                terms.append(_term(feature, op, quantile, threshold))
            specs.append(
                {
                    "rule_name": f"pair_{first}_{first_op}_{second}_{second_op}_{tail:.2f}_{side}",
                    "side": side,
                    "terms": terms,
                }
            )
    return specs


def rule_mask(features: pd.DataFrame, terms: list[dict[str, Any]]) -> np.ndarray:
    active = features["alt_derivatives_available"].to_numpy(float) > 0.5
    for term in terms:
        values = features[term["feature"]].to_numpy(float)
        finite = np.isfinite(values)
        if term["op"] == "le":
            active &= finite & (values <= float(term["threshold"]))
        elif term["op"] == "ge":
            active &= finite & (values >= float(term["threshold"]))
        else:
            raise ValueError(f"unknown operator: {term['op']}")
    return active


def _activation_hash(active: np.ndarray, side: str) -> str:
    digest = hashlib.sha256()
    digest.update(side.encode())
    digest.update(np.packbits(active).tobytes())
    return digest.hexdigest()


def _simulate(
    market: pd.DataFrame,
    dates: pd.Series,
    active: np.ndarray,
    side: str,
    cfg: AltCrowdingConfig,
    *,
    window: str,
    hold_bars: int,
    stride_bars: int,
    extremes: tuple[np.ndarray, np.ndarray],
    windows: dict[str, tuple[str, str]] = WINDOWS,
) -> dict[str, Any]:
    long_active = active if side == "long" else np.zeros(len(active), dtype=bool)
    short_active = active if side == "short" else np.zeros(len(active), dtype=bool)
    return _simulate_no_stop(
        market,
        dates,
        long_active,
        short_active,
        window=window,
        hold_bars=hold_bars,
        stride_bars=stride_bars,
        leverage=cfg.leverage,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
        extremes=extremes,
        windows=windows,
    )


def _selection_score(stats: dict[str, dict[str, Any]], cfg: AltCrowdingConfig) -> float:
    fit = stats["fit_2023_h1"]
    selection = stats["select_2023_h2"]
    if fit["trades"] < cfg.min_fit_trades or selection["trades"] < cfg.min_select_trades:
        return -1e12
    if min(fit["cagr_pct"], selection["cagr_pct"]) <= 0.0:
        return -1e12
    if fit["strict_mdd_pct"] > 25.0 or selection["strict_mdd_pct"] > 20.0:
        return -1e12
    ratios = np.asarray([fit["ratio"], selection["ratio"]], dtype=float)
    return float(np.min(ratios) + 0.35 * np.median(ratios) + min(0.25, selection["trades"] / 100.0))


def _load_feature_bundle(
    cfg: AltCrowdingConfig,
    *,
    cutoff: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    market = _load_market(cfg, cutoff)
    attached, source_times = attach_alt_derivatives(market, cfg, source_cutoff=cutoff)
    features = build_alt_features(attached)
    base_features = build_base_features(market)
    return market, pd.to_datetime(market["date"]), features, base_features, source_times


def _select_manifest(cfg: AltCrowdingConfig) -> dict[str, Any]:
    market, dates, features, base_features, source_times = _load_feature_bundle(cfg, cutoff=SELECTION_END)
    if not market.empty and dates.max() >= pd.Timestamp(SELECTION_END):
        raise RuntimeError("selection bundle contains post-2023 market rows")
    for column in source_times:
        valid = source_times[column].notna()
        if (source_times.loc[valid, column] >= pd.Timestamp(SELECTION_END)).any():
            raise RuntimeError(f"selection bundle contains post-2023 source row: {column}")
    fit_mask = _window_mask(dates, "fit_2023_h1")
    admitted, correlation_audit = feature_admission(
        features,
        base_features,
        fit_mask,
        max_abs_spearman=cfg.max_abs_spearman,
        min_observations=cfg.min_fit_observations,
    )
    quantiles = _parse_csv(cfg.quantiles, float)
    specs = _single_rule_specs(features, admitted, fit_mask, quantiles)
    specs += _pair_rule_specs(features, set(admitted), fit_mask, quantiles)
    extremes = {
        hold: (
            _future_extreme(market["low"].to_numpy(float), hold, "min"),
            _future_extreme(market["high"].to_numpy(float), hold, "max"),
        )
        for hold in HOLD_BARS
    }
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for spec in specs:
        active = rule_mask(features, spec["terms"])
        activation_hash = _activation_hash(active, spec["side"])
        rule_key = (activation_hash, spec["side"])
        if rule_key in seen:
            continue
        seen.add(rule_key)
        for hold_bars, stride_bars in itertools.product(HOLD_BARS, STRIDE_BARS):
            stats = {
                window: _simulate(
                    market,
                    dates,
                    active,
                    spec["side"],
                    cfg,
                    window=window,
                    hold_bars=hold_bars,
                    stride_bars=stride_bars,
                    extremes=extremes[hold_bars],
                )
                for window in ("fit_2023_h1", "select_2023_h2")
            }
            score = _selection_score(stats, cfg)
            if score <= -1e11:
                continue
            rows.append(
                {
                    **spec,
                    "activation_hash": activation_hash,
                    "hold_bars": hold_bars,
                    "stride_bars": stride_bars,
                    "selection_score": score,
                    "selection_stats": stats,
                }
            )
    rows.sort(
        key=lambda row: (
            row["selection_score"],
            row["selection_stats"]["select_2023_h2"]["ratio"],
            row["selection_stats"]["select_2023_h2"]["return_pct"],
        ),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for row in rows:
        key = row["rule_name"]
        if counts.get(key, 0) >= cfg.top_per_rule:
            continue
        selected.append(row)
        counts[key] = counts.get(key, 0) + 1
        if len(selected) >= cfg.top_n:
            break
    core = {
        "protocol": {
            "feature_fit": WINDOWS["fit_2023_h1"],
            "policy_selection": WINDOWS["select_2023_h2"],
            "future_market_and_source_rows_truncated_before_feature_construction": True,
            "feature_family": "six-alt Binance USD-M funding and premium-index crowding only",
            "feature_admission": f"fit Spearman max |rho| < {cfg.max_abs_spearman} versus {BASE_FEATURES}",
            "funding_availability": f"backward-asof, max staleness {cfg.funding_tolerance}",
            "premium_availability": f"hourly close_time backward-asof, max staleness {cfg.premium_tolerance}",
            "entry": "next 5m open",
            "exit": "fixed hold, non-overlapping, no TP/SL",
            "cost": "5bp fee + 1bp slippage per side at 0.5x",
            "mdd": "strict favorable-high-water then adverse OHLC extreme",
            "status_ceiling": "shadow research: broader program has inspected later years",
        },
        "source_hashes": _source_hashes(cfg),
        "feature_hash": _feature_hash(features, dates),
        "base_feature_hash": _feature_hash(base_features, dates),
        "feature_admission_audit": correlation_audit,
        "search_space": {
            "admitted_features": admitted,
            "raw_specs": len(specs),
            "effective_unique_masks": len(seen),
            "eligible_policy_variants": len(rows),
            "top_n": cfg.top_n,
        },
        "selected": selected,
    }
    canonical = json.dumps(core, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=True)
    manifest = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        **core,
    }
    path = Path(cfg.manifest_output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=True) + "\n")
    return manifest


def _validate_manifest(manifest: dict[str, Any]) -> None:
    core = {key: value for key, value in manifest.items() if key not in {"as_of", "sha256"}}
    canonical = json.dumps(core, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=True)
    expected = hashlib.sha256(canonical.encode()).hexdigest()
    if manifest.get("sha256") != expected:
        raise RuntimeError("manifest content does not match its frozen SHA-256")


def _replay(cfg: AltCrowdingConfig, manifest: dict[str, Any]) -> dict[str, Any]:
    _validate_manifest(manifest)
    current_source_hashes = _source_hashes(cfg)
    if current_source_hashes != manifest["source_hashes"]:
        raise RuntimeError("source files changed after manifest freeze")
    market, dates, features, base_features, _ = _load_feature_bundle(cfg, cutoff=cfg.exclude_from)
    prefix = dates < pd.Timestamp(SELECTION_END)
    prefix_dates = dates.loc[prefix].reset_index(drop=True)
    prefix_features = features.loc[prefix].reset_index(drop=True)
    if _feature_hash(prefix_features, prefix_dates) != manifest["feature_hash"]:
        raise RuntimeError("pre-2024 external feature prefix changed during full replay")
    prefix_base = base_features.loc[prefix].reset_index(drop=True)
    if _feature_hash(prefix_base, prefix_dates) != manifest["base_feature_hash"]:
        raise RuntimeError("pre-2024 BTC admission-feature prefix changed during full replay")
    extremes = {
        hold: (
            _future_extreme(market["low"].to_numpy(float), hold, "min"),
            _future_extreme(market["high"].to_numpy(float), hold, "max"),
        )
        for hold in HOLD_BARS
    }
    selected: list[dict[str, Any]] = []
    for rank, frozen in enumerate(manifest["selected"], start=1):
        active = rule_mask(features, frozen["terms"])
        prefix_hash = _activation_hash(active[prefix.to_numpy(bool)], frozen["side"])
        if prefix_hash != frozen["activation_hash"]:
            raise RuntimeError(f"pre-2024 activation drift at rank {rank}")
        stats = {
            window: _simulate(
                market,
                dates,
                active,
                frozen["side"],
                cfg,
                window=window,
                hold_bars=frozen["hold_bars"],
                stride_bars=frozen["stride_bars"],
                extremes=extremes[frozen["hold_bars"]],
            )
            for window in WINDOWS
        }
        for window in ("fit_2023_h1", "select_2023_h2"):
            if stats[window] != frozen["selection_stats"][window]:
                raise RuntimeError(f"selection replay drift rank={rank} window={window}")
        quarterly = {
            window: _simulate(
                market,
                dates,
                active,
                frozen["side"],
                cfg,
                window=window,
                hold_bars=frozen["hold_bars"],
                stride_bars=frozen["stride_bars"],
                extremes=extremes[frozen["hold_bars"]],
                windows=QUARTER_WINDOWS,
            )
            for window in QUARTER_WINDOWS
        }
        test = stats["test_2024"]
        evaluation = stats["eval_2025"]
        holdout = stats["holdout_2026"]
        combined = stats["oos_2024_2026"]
        enough = test["trades"] >= 20 and evaluation["trades"] >= 20 and holdout["trades"] >= 10
        passes_alpha_pool = enough and test["ratio"] >= 2.5 and evaluation["ratio"] >= 2.5
        passes_live_grade = (
            passes_alpha_pool
            and holdout["ratio"] >= 3.0
            and combined["ratio"] >= 3.0
            and combined["p_value_mean_return_approx"] <= 0.05
        )
        selected.append(
            {
                "manifest_rank": rank,
                **frozen,
                "stats": stats,
                "quarterly_stats": quarterly,
                "quarterly_summary": {
                    "positive_return_quarters": sum(row["return_pct"] > 0.0 for row in quarterly.values()),
                    "flat_quarters": sum(row["trades"] == 0 for row in quarterly.values()),
                    "negative_return_quarters": sum(row["return_pct"] < 0.0 for row in quarterly.values()),
                    "total_quarters": len(quarterly),
                },
                "passes_alpha_pool": bool(passes_alpha_pool),
                "passes_live_grade": bool(passes_live_grade),
            }
        )
    return {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "manifest": cfg.manifest_output,
        "manifest_sha256": manifest["sha256"],
        "protocol": manifest["protocol"],
        "feature_admission_audit": manifest["feature_admission_audit"],
        "selected": selected,
        "alpha_pool_qualifiers": [row for row in selected if row["passes_alpha_pool"]],
        "live_grade": [row for row in selected if row["passes_live_grade"]],
    }


def run(cfg: AltCrowdingConfig) -> dict[str, Any]:
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


def parse_args() -> AltCrowdingConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--aux-dir", required=True)
    parser.add_argument("--btc-funding-csv", required=True)
    parser.add_argument("--btc-premium-csv", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--exclude-from", default=AltCrowdingConfig.exclude_from)
    parser.add_argument("--quantiles", default=AltCrowdingConfig.quantiles)
    parser.add_argument("--top-n", type=int, default=AltCrowdingConfig.top_n)
    parser.add_argument("--top-per-rule", type=int, default=AltCrowdingConfig.top_per_rule)
    parser.add_argument("--max-abs-spearman", type=float, default=AltCrowdingConfig.max_abs_spearman)
    parser.add_argument("--refresh-manifest", action="store_true")
    return AltCrowdingConfig(**vars(parser.parse_args()))


def main() -> None:
    report = run(parse_args())
    print(
        json.dumps(
            {
                "manifest_sha256": report["manifest_sha256"],
                "selected": len(report["selected"]),
                "alpha_pool_qualifiers": len(report["alpha_pool_qualifiers"]),
                "live_grade": len(report["live_grade"]),
                "top": report["selected"][:3],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
