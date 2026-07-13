"""Search causal ordinal-pattern complexity as a standalone BTC alpha family.

Hourly price permutations, permutation entropy, pattern surprise and empirical
transition surprise are computed only from completed source hours.  Thresholds
are fitted before 2023, policies are selected on 2023/H1/H2 with 2024+ market
rows physically absent, and a frozen Top-10 is replayed on 2024-2026.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.search_funding_premium_external_state_gate_alpha import (
    _file_sha256,
    _frame_hash,
    _manifest_core_hash,
    _validate_manifest,
)
from training.search_positioning_disagreement_alpha import _future_extreme, _simulate_no_stop
from training.search_positioning_hgb_path_alpha import _feature_hash, _read_before


FIT_WINDOW = ("2020-06-01", "2023-01-01")
SELECTION_END = "2024-01-01"
HOLD_BARS = (144, 288, 576)
STRIDE_BARS = 12
ORDERS = (3, 4)
ENTROPY_WINDOWS = (168, 720)
TAILS = (0.20, 0.30)

WINDOWS = {
    "fit": FIT_WINDOW,
    "select_2023": ("2023-01-01", "2024-01-01"),
    "select_2023_h1": ("2023-01-01", "2023-07-01"),
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


@dataclass(frozen=True)
class OrdinalComplexityConfig:
    input_csv: str
    output: str
    manifest_output: str
    exclude_from: str = "2026-06-02"
    leverage: float = 0.5
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0001
    stress_fee_rate: float = 0.0009
    top_n: int = 10
    top_per_rule: int = 2
    min_fit_observations: int = 10_000
    min_fit_trades: int = 80
    min_select_trades: int = 20
    min_half_trades: int = 8
    min_fit_each_side: int = 15
    min_select_each_side: int = 4
    max_abs_spearman: float = 0.40
    refresh_manifest: bool = False


def _window_mask(dates: pd.Series, name: str) -> np.ndarray:
    start, end = WINDOWS[name]
    return ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy(bool)


def _load_market(cfg: OrdinalComplexityConfig, *, cutoff: str) -> tuple[pd.DataFrame, pd.Series, str]:
    market = _read_before(cfg.input_csv, "date", cutoff)
    source_prefix_hash = _frame_hash(market)
    market["date"] = pd.to_datetime(market["date"], utc=True, errors="raise").dt.tz_convert(None)
    market = market.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    dates = pd.to_datetime(market["date"])
    if len(dates) and dates.max() >= pd.Timestamp(cutoff):
        raise RuntimeError("market rows were not physically truncated before cutoff")
    intervals = dates.diff().dropna()
    if len(intervals) and not intervals.eq(pd.Timedelta("5min")).all():
        raise ValueError("ordinal search requires a complete 5-minute market grid")
    return market, dates, source_prefix_hash


def _completed_hourly_close(market: pd.DataFrame) -> pd.DataFrame:
    raw = pd.DataFrame(
        {
            "date": pd.to_datetime(market["date"]),
            "close": pd.to_numeric(market["close"], errors="coerce"),
        }
    )
    raw["source_hour"] = raw["date"].dt.floor("h")
    hourly = raw.groupby("source_hour", sort=True).agg(close=("close", "last"), source_rows=("date", "size"))
    hourly["close"] = hourly["close"].where(hourly["source_rows"] == 12)
    hourly["effective_time"] = hourly.index + pd.Timedelta("1h")
    return hourly.reset_index()


def _ordinal_pattern_codes(values: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    codes = np.full(len(values), -1, dtype=np.int16)
    direction = np.full(len(values), np.nan, dtype=float)
    if order < 2 or len(values) < order:
        return codes, direction
    permutations = {perm: index for index, perm in enumerate(itertools.permutations(range(order)))}
    windows = np.lib.stride_tricks.sliding_window_view(values, order)
    for offset, window in enumerate(windows, start=order - 1):
        if not np.isfinite(window).all():
            continue
        permutation = tuple(np.argsort(window, kind="stable").tolist())
        codes[offset] = permutations[permutation]
        ranks = np.empty(order, dtype=float)
        ranks[np.asarray(permutation, dtype=int)] = np.arange(order, dtype=float)
        direction[offset] = (ranks[-1] - ranks[0]) / float(order - 1)
    return codes, direction


def _historical_count(codes: np.ndarray, state: int, window: int) -> np.ndarray:
    indicator = pd.Series(codes == state, dtype=float)
    return indicator.shift(1).rolling(window, min_periods=window).sum().to_numpy(float)


def _ordinal_statistics(codes: np.ndarray, states: int, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use only patterns/transitions strictly before each current pattern."""
    codes = np.asarray(codes, dtype=int)
    valid = codes >= 0
    denominator = pd.Series(valid.astype(float)).shift(1).rolling(window, min_periods=window).sum().to_numpy(float)
    entropy = np.zeros(len(codes), dtype=float)
    pattern_surprise = np.full(len(codes), np.nan, dtype=float)
    log_states = np.log(float(states))
    for state in range(states):
        count = _historical_count(codes, state, window)
        probability = np.divide(count, denominator, out=np.zeros_like(count), where=denominator > 0.0)
        positive = probability > 0.0
        entropy[positive] -= probability[positive] * np.log(probability[positive])
        current = codes == state
        smoothed = np.divide(count + 1.0, denominator + states, out=np.full_like(count, np.nan), where=denominator > 0.0)
        pattern_surprise[current] = -np.log(smoothed[current]) / log_states
    entropy = np.divide(entropy, log_states, out=np.full_like(entropy, np.nan), where=denominator > 0.0)

    previous = np.r_[-1, codes[:-1]]
    transitions = np.where(valid & (previous >= 0), previous * states + codes, -1)
    transition_surprise = np.full(len(codes), np.nan, dtype=float)
    previous_counts: dict[int, np.ndarray] = {
        state: _historical_count(previous, state, window) for state in range(states)
    }
    for transition in range(states * states):
        previous_state = transition // states
        numerator = _historical_count(transitions, transition, window)
        denominator_previous = previous_counts[previous_state]
        probability = np.divide(
            numerator + 1.0,
            denominator_previous + states,
            out=np.full_like(numerator, np.nan),
            where=denominator_previous > 0.0,
        )
        current = transitions == transition
        transition_surprise[current] = -np.log(probability[current]) / log_states
    available = valid & np.isfinite(denominator) & (denominator >= window)
    entropy[~available] = np.nan
    pattern_surprise[~available] = np.nan
    transition_surprise[~available] = np.nan
    return entropy, pattern_surprise, transition_surprise


def _build_hourly_features(hourly: pd.DataFrame) -> pd.DataFrame:
    close = pd.to_numeric(hourly["close"], errors="coerce").to_numpy(float)
    out = pd.DataFrame({"effective_time": pd.to_datetime(hourly["effective_time"])})
    for order in ORDERS:
        codes, direction = _ordinal_pattern_codes(np.log(close), order)
        out[f"oc_pattern_{order}"] = np.where(codes >= 0, codes, np.nan)
        out[f"oc_direction_{order}"] = direction
        states = math.factorial(order)
        for window in ENTROPY_WINDOWS:
            entropy, pattern_surprise, transition_surprise = _ordinal_statistics(codes, states, window)
            prefix = f"oc_o{order}_w{window}"
            out[f"{prefix}_entropy"] = entropy
            out[f"{prefix}_pattern_surprise"] = pattern_surprise
            out[f"{prefix}_transition_surprise"] = transition_surprise
    return out


def _build_features(market: pd.DataFrame) -> pd.DataFrame:
    hourly = _build_hourly_features(_completed_hourly_close(market))
    left = pd.DataFrame({"date": pd.to_datetime(market["date"]), "_row": np.arange(len(market))})
    merged = pd.merge_asof(
        left.sort_values("date"),
        hourly.sort_values("effective_time"),
        left_on="date",
        right_on="effective_time",
        direction="backward",
        tolerance=pd.Timedelta("65min"),
    ).sort_values("_row")
    return merged.drop(columns=["date", "_row", "effective_time"]).reset_index(drop=True)


def _reference_features(market: pd.DataFrame) -> pd.DataFrame:
    close = pd.to_numeric(market["close"], errors="coerce")
    ret = np.log(close).diff()
    sign = (ret > 0.0).astype(float)
    probability = sign.rolling(144, min_periods=144).mean().clip(1e-6, 1.0 - 1e-6)
    sign_entropy = -(probability * np.log(probability) + (1.0 - probability) * np.log(1.0 - probability)) / np.log(2.0)
    variance_1 = ret.rolling(144, min_periods=144).var()
    variance_12 = np.log(close / close.shift(12)).rolling(144, min_periods=144).var() / 12.0
    path = ret.abs().rolling(72, min_periods=72).sum()
    rv24 = ret.pow(2).rolling(24, min_periods=24).sum().pow(0.5)
    rv288 = ret.pow(2).rolling(288, min_periods=288).sum().pow(0.5)
    return pd.DataFrame(
        {
            "pm_sign_entropy": sign_entropy,
            "pm_sign_autocorr": ret.rolling(144, min_periods=144).corr(ret.shift(1)),
            "pm_variance_ratio_12": variance_12 / variance_1.replace(0.0, np.nan),
            "ev_eff_72": np.log(close / close.shift(72)).abs() / path.replace(0.0, np.nan),
            "jv_vov": rv24 / rv288.replace(0.0, np.nan),
        }
    ).replace([np.inf, -np.inf], np.nan)


def _correlation_audit(
    features: pd.DataFrame,
    references: pd.DataFrame,
    fit_mask: np.ndarray,
    *,
    max_abs_spearman: float,
) -> dict[str, Any]:
    audit: dict[str, Any] = {}
    feature_names = [name for name in features if name.endswith(("_entropy", "_pattern_surprise", "_transition_surprise"))]
    for feature in feature_names:
        values = pd.to_numeric(features[feature], errors="coerce")
        correlations: dict[str, float] = {}
        counts: dict[str, int] = {}
        for reference in references:
            paired = fit_mask & values.notna().to_numpy(bool) & references[reference].notna().to_numpy(bool)
            counts[reference] = int(paired.sum())
            correlations[reference] = float(values.loc[paired].corr(references.loc[paired, reference], method="spearman")) if counts[reference] >= 100 else float("nan")
        max_abs = max((abs(value) for value in correlations.values() if np.isfinite(value)), default=float("inf"))
        audit[feature] = {
            "pair_counts": counts,
            "spearman": correlations,
            "max_abs_spearman": float(max_abs),
            "passes_independence": bool(max_abs < max_abs_spearman),
        }
    return audit


def _fit_threshold(values: np.ndarray, fit_mask: np.ndarray, quantile: float, *, min_observations: int) -> float:
    reference = values[fit_mask & np.isfinite(values)]
    if len(reference) < min_observations:
        raise ValueError(f"insufficient ordinal fit observations: {len(reference)}")
    return float(np.quantile(reference, quantile))


def _signal_specs(features: pd.DataFrame, fit_mask: np.ndarray, cfg: OrdinalComplexityConfig) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for order, window, tail in itertools.product(ORDERS, ENTROPY_WINDOWS, TAILS):
        prefix = f"oc_o{order}_w{window}"
        feature_modes = (
            ("low_entropy", f"{prefix}_entropy", "le", tail),
            ("high_entropy", f"{prefix}_entropy", "ge", 1.0 - tail),
            ("high_pattern_surprise", f"{prefix}_pattern_surprise", "ge", 1.0 - tail),
            ("high_transition_surprise", f"{prefix}_transition_surprise", "ge", 1.0 - tail),
        )
        for state_name, feature, op, quantile in feature_modes:
            threshold = _fit_threshold(
                pd.to_numeric(features[feature], errors="coerce").to_numpy(float),
                fit_mask,
                quantile,
                min_observations=cfg.min_fit_observations,
            )
            for direction_mode in ("continuation", "reversal"):
                specs.append(
                    {
                        "rule": f"{state_name}_{direction_mode}",
                        "order": order,
                        "entropy_window_hours": window,
                        "tail": tail,
                        "feature": feature,
                        "op": op,
                        "threshold": threshold,
                        "direction_mode": direction_mode,
                        "direction_threshold": 0.5,
                    }
                )
    if len(specs) != 64:
        raise RuntimeError(f"ordinal signal family changed unexpectedly: {len(specs)}")
    return specs


def _active_masks(features: pd.DataFrame, spec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    values = pd.to_numeric(features[spec["feature"]], errors="coerce").to_numpy(float)
    direction = pd.to_numeric(features[f"oc_direction_{spec['order']}"], errors="coerce").to_numpy(float)
    finite = np.isfinite(values) & np.isfinite(direction)
    gate = finite & ((values <= spec["threshold"]) if spec["op"] == "le" else (values >= spec["threshold"]))
    up = direction >= spec["direction_threshold"]
    down = direction <= -spec["direction_threshold"]
    if spec["direction_mode"] == "continuation":
        return gate & up, gate & down
    if spec["direction_mode"] == "reversal":
        return gate & down, gate & up
    raise ValueError(f"unknown direction mode: {spec['direction_mode']}")


def _activation_hash(long_active: np.ndarray, short_active: np.ndarray) -> str:
    payload = np.r_[np.asarray(long_active, dtype=bool), np.asarray(short_active, dtype=bool)]
    return hashlib.sha256(np.packbits(payload).tobytes()).hexdigest()


def _simulate(
    market: pd.DataFrame,
    dates: pd.Series,
    long_active: np.ndarray,
    short_active: np.ndarray,
    cfg: OrdinalComplexityConfig,
    *,
    hold_bars: int,
    window: str,
    extremes: tuple[np.ndarray, np.ndarray],
    windows: dict[str, tuple[str, str]] = WINDOWS,
) -> dict[str, Any]:
    return _simulate_no_stop(
        market,
        dates,
        long_active,
        short_active,
        window=window,
        hold_bars=hold_bars,
        stride_bars=STRIDE_BARS,
        leverage=cfg.leverage,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
        extremes=extremes,
        windows=windows,
    )


def _selection_score(stats: dict[str, dict[str, Any]], cfg: OrdinalComplexityConfig) -> float:
    fit, select = stats["fit"], stats["select_2023"]
    h1, h2 = stats["select_2023_h1"], stats["select_2023_h2"]
    if fit["trades"] < cfg.min_fit_trades or select["trades"] < cfg.min_select_trades:
        return -1e12
    if h1["trades"] < cfg.min_half_trades or h2["trades"] < cfg.min_half_trades:
        return -1e12
    if min(fit["longs"], fit["shorts"]) < cfg.min_fit_each_side:
        return -1e12
    if min(select["longs"], select["shorts"]) < cfg.min_select_each_side:
        return -1e12
    if min(fit["cagr_pct"], select["cagr_pct"], h1["cagr_pct"], h2["cagr_pct"]) <= 0.0:
        return -1e12
    if fit["strict_mdd_pct"] > 30.0 or select["strict_mdd_pct"] > 20.0:
        return -1e12
    ratios = np.asarray([fit["ratio"], select["ratio"], h1["ratio"], h2["ratio"]], dtype=float)
    return float(np.min(ratios) + 0.30 * np.median(ratios) + min(0.25, select["trades"] / 200.0))


def _select_top(rows: list[dict[str, Any]], *, top_n: int, top_per_rule: int) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            row["selection_score"],
            row["selection_stats"]["select_2023"]["ratio"],
            row["selection_stats"]["select_2023"]["return_pct"],
            row["rule"],
            -row["hold_bars"],
            -row["order"],
            -row["entropy_window_hours"],
        ),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for row in ordered:
        if counts.get(row["rule"], 0) >= top_per_rule:
            continue
        selected.append(row)
        counts[row["rule"]] = counts.get(row["rule"], 0) + 1
        if len(selected) >= top_n:
            break
    return selected


def _select_manifest(cfg: OrdinalComplexityConfig) -> dict[str, Any]:
    market, dates, source_prefix_hash = _load_market(cfg, cutoff=SELECTION_END)
    features = _build_features(market)
    references = _reference_features(market)
    fit_mask = _window_mask(dates, "fit")
    correlation_audit = _correlation_audit(
        features,
        references,
        fit_mask,
        max_abs_spearman=cfg.max_abs_spearman,
    )
    specs = _signal_specs(features, fit_mask, cfg)
    extremes_by_hold = {
        hold: (
            _future_extreme(market["low"].to_numpy(float), hold, "min"),
            _future_extreme(market["high"].to_numpy(float), hold, "max"),
        )
        for hold in HOLD_BARS
    }
    rows: list[dict[str, Any]] = []
    effective_masks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for spec in specs:
        long_active, short_active = _active_masks(features, spec)
        activation_hash = _activation_hash(long_active, short_active)
        effective_masks.setdefault(activation_hash, (long_active, short_active))
        for hold in HOLD_BARS:
            stats = {
                window: _simulate(
                    market,
                    dates,
                    long_active,
                    short_active,
                    cfg,
                    hold_bars=hold,
                    window=window,
                    extremes=extremes_by_hold[hold],
                )
                for window in ("fit", "select_2023", "select_2023_h1", "select_2023_h2")
            }
            score = _selection_score(stats, cfg)
            if score <= -1e11:
                continue
            rows.append(
                {
                    **spec,
                    "hold_bars": hold,
                    "stride_bars": STRIDE_BARS,
                    "activation_hash": activation_hash,
                    "selection_score": score,
                    "selection_stats": stats,
                }
            )
    if len(specs) * len(HOLD_BARS) != 192:
        raise RuntimeError("ordinal policy search exceeded or changed its 192-policy cap")
    selected = _select_top(rows, top_n=cfg.top_n, top_per_rule=cfg.top_per_rule)
    core = {
        "protocol": {
            "family": "completed-hour ordinal price patterns, permutation entropy, pattern surprise and empirical transition surprise",
            "threshold_fit": FIT_WINDOW,
            "selection": {name: WINDOWS[name] for name in ("select_2023", "select_2023_h1", "select_2023_h2")},
            "all_future_market_rows_physically_excluded_before_manifest": True,
            "hourly_timing": "12 completed 5m rows required; source hour H exposed no earlier than H+1h",
            "search_cap": "64 masks x fixed holds {144,288,576} = 192 policies",
            "entry": "next 5m open",
            "exit": "fixed selected hold; stride 12; no TP/SL",
            "cost": "6bp/side base and 10bp/side stress at 0.5x",
            "mdd": "strict favorable-high-water then adverse OHLC extreme",
            "status_ceiling": "retrospective research; no direct live promotion",
        },
        "source_prefix_hash": source_prefix_hash,
        "ordinal_feature_hash": _feature_hash(features, dates),
        "reference_feature_hash": _feature_hash(references, dates),
        "correlation_audit": correlation_audit,
        "search_space": {
            "raw_signal_specs": len(specs),
            "effective_unique_masks": len(effective_masks),
            "raw_policies": len(specs) * len(HOLD_BARS),
            "eligible_variants": len(rows),
            "top_n": cfg.top_n,
            "top_per_rule": cfg.top_per_rule,
        },
        "selected": selected,
    }
    manifest = {"as_of": datetime.now(timezone.utc).isoformat(), "sha256": _manifest_core_hash(core), **core}
    path = Path(cfg.manifest_output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=True) + "\n")
    return manifest


def _replay(cfg: OrdinalComplexityConfig, manifest: dict[str, Any]) -> dict[str, Any]:
    _validate_manifest(manifest)
    _, _, source_prefix_hash = _load_market(cfg, cutoff=SELECTION_END)
    if source_prefix_hash != manifest["source_prefix_hash"]:
        raise RuntimeError("pre-2024 market source prefix changed after manifest freeze")
    market, dates, _ = _load_market(cfg, cutoff=cfg.exclude_from)
    features = _build_features(market)
    references = _reference_features(market)
    prefix = dates < pd.Timestamp(SELECTION_END)
    prefix_dates = dates.loc[prefix].reset_index(drop=True)
    if _feature_hash(features.loc[prefix].reset_index(drop=True), prefix_dates) != manifest["ordinal_feature_hash"]:
        raise RuntimeError("pre-2024 ordinal feature prefix changed during replay")
    if _feature_hash(references.loc[prefix].reset_index(drop=True), prefix_dates) != manifest["reference_feature_hash"]:
        raise RuntimeError("pre-2024 reference feature prefix changed during replay")
    holds = {int(row["hold_bars"]) for row in manifest["selected"]}
    extremes_by_hold = {
        hold: (
            _future_extreme(market["low"].to_numpy(float), hold, "min"),
            _future_extreme(market["high"].to_numpy(float), hold, "max"),
        )
        for hold in holds
    }
    stress_cfg = replace(cfg, fee_rate=cfg.stress_fee_rate)
    selected: list[dict[str, Any]] = []
    spec_keys = (
        "rule", "order", "entropy_window_hours", "tail", "feature", "op",
        "threshold", "direction_mode", "direction_threshold",
    )
    for rank, frozen in enumerate(manifest["selected"], start=1):
        spec = {key: frozen[key] for key in spec_keys}
        hold = int(frozen["hold_bars"])
        long_active, short_active = _active_masks(features, spec)
        prefix_array = prefix.to_numpy(bool)
        if _activation_hash(long_active[prefix_array], short_active[prefix_array]) != frozen["activation_hash"]:
            raise RuntimeError(f"pre-2024 activation drift at rank {rank}")
        stats = {
            window: _simulate(
                market,
                dates,
                long_active,
                short_active,
                cfg,
                hold_bars=hold,
                window=window,
                extremes=extremes_by_hold[hold],
            )
            for window in WINDOWS
        }
        for window in ("fit", "select_2023", "select_2023_h1", "select_2023_h2"):
            if stats[window] != frozen["selection_stats"][window]:
                raise RuntimeError(f"selection replay drift rank={rank} window={window}")
        stress = {
            window: _simulate(
                market,
                dates,
                long_active,
                short_active,
                stress_cfg,
                hold_bars=hold,
                window=window,
                extremes=extremes_by_hold[hold],
            )
            for window in ("test_2024", "eval_2025", "holdout_2026", "oos_2024_2026")
        }
        quarterly = {
            window: _simulate(
                market,
                dates,
                long_active,
                short_active,
                cfg,
                hold_bars=hold,
                window=window,
                extremes=extremes_by_hold[hold],
                windows=QUARTER_WINDOWS,
            )
            for window in QUARTER_WINDOWS
        }
        summary = {
            "positive_return_quarters": sum(row["return_pct"] > 0.0 for row in quarterly.values()),
            "flat_quarters": sum(row["trades"] == 0 for row in quarterly.values()),
            "negative_return_quarters": sum(row["return_pct"] < 0.0 for row in quarterly.values()),
            "total_quarters": len(quarterly),
        }
        test, evaluation = stats["test_2024"], stats["eval_2025"]
        holdout, combined = stats["holdout_2026"], stats["oos_2024_2026"]
        enough = (
            test["trades"] >= 30 and evaluation["trades"] >= 30 and holdout["trades"] >= 15
            and min(test["longs"], test["shorts"], evaluation["longs"], evaluation["shorts"]) >= 8
            and min(holdout["longs"], holdout["shorts"]) >= 4
        )
        passes_alpha_pool = enough and test["ratio"] >= 2.5 and evaluation["ratio"] >= 2.5 and holdout["return_pct"] > 0.0
        bonferroni = min(1.0, combined["p_value_mean_return_approx"] * max(1, len(manifest["selected"])))
        strong_shadow = (
            passes_alpha_pool
            and min(test["ratio"], evaluation["ratio"], holdout["ratio"], combined["ratio"]) >= 3.0
            and summary["positive_return_quarters"] >= 7
            and summary["negative_return_quarters"] <= 2
            and bonferroni <= 0.05
            and min(stress[name]["ratio"] for name in ("test_2024", "eval_2025", "holdout_2026")) >= 2.5
        )
        selected.append(
            {
                "manifest_rank": rank,
                **frozen,
                "stats": stats,
                "stress_10bp_each_side": stress,
                "quarterly_stats": quarterly,
                "quarterly_summary": summary,
                "top_n_bonferroni_p_value": float(bonferroni),
                "passes_alpha_pool": bool(passes_alpha_pool),
                "passes_strong_shadow": bool(strong_shadow),
                "passes_live_grade": False,
            }
        )
    return {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "manifest": cfg.manifest_output,
        "manifest_sha256": manifest["sha256"],
        "protocol": manifest["protocol"],
        "source_file_sha256_after_freeze": _file_sha256(cfg.input_csv),
        "correlation_audit": manifest["correlation_audit"],
        "selected": selected,
        "alpha_pool_qualifiers": [row for row in selected if row["passes_alpha_pool"]],
        "strong_shadow": [row for row in selected if row["passes_strong_shadow"]],
        "live_grade": [],
    }


def run(cfg: OrdinalComplexityConfig) -> dict[str, Any]:
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


def parse_args() -> OrdinalComplexityConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--exclude-from", default=OrdinalComplexityConfig.exclude_from)
    parser.add_argument("--top-n", type=int, default=OrdinalComplexityConfig.top_n)
    parser.add_argument("--top-per-rule", type=int, default=OrdinalComplexityConfig.top_per_rule)
    parser.add_argument("--refresh-manifest", action="store_true")
    return OrdinalComplexityConfig(**vars(parser.parse_args()))


def main() -> None:
    report = run(parse_args())
    print(
        json.dumps(
            {
                "manifest_sha256": report["manifest_sha256"],
                "selected": len(report["selected"]),
                "alpha_pool_qualifiers": len(report["alpha_pool_qualifiers"]),
                "strong_shadow": len(report["strong_shadow"]),
                "top": report["selected"][:3],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
