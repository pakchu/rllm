"""Observable explicit-duration semi-Markov gate for the funding/premium setup.

The model augments a completed-hour market state with its causal run age.  All
state thresholds, duration-conditioned trade quality, and empirical exit
hazards are fit on 2020-2022. Candidates are ranked on the worst risk ratio
across the three fit years and the 2023 internal holdout. Test 2024 is the first
untouched overlay test; Eval 2025 and 2026 remain later diagnostics.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import training.search_bidirectional_state_alpha as state_sim
from preprocessing.market_features import build_market_feature_frame
from training.long_regime_combo_scan import _load_market, _split_mask
from training.long_regime_interest_gate_validation import build_interest_features
from training.portfolio_opt_new_alpha_pool import _alpha_active, _event_path
from training.search_bidirectional_state_alpha import Config, sim
from training.search_gaussian_hmm_regime_alpha import hourly_features


WINDOWS = {
    "fit2020_2022": ("2020-01-01", "2023-01-01"),
    "year2020": ("2020-01-01", "2021-01-01"),
    "year2021": ("2021-01-01", "2022-01-01"),
    "year2022": ("2022-01-01", "2023-01-01"),
    "holdout2023": ("2023-01-01", "2024-01-01"),
    "test2024": ("2024-01-01", "2025-01-01"),
    "eval2025": ("2025-01-01", "2026-01-01"),
    "ytd2026": ("2026-01-01", "2026-06-02"),
}
SELECTION_WINDOWS = (
    "year2020",
    "year2021",
    "year2022",
    "holdout2023",
)


def causal_run_age(
    state: np.ndarray,
    timestamps: pd.DatetimeIndex | np.ndarray | None = None,
) -> np.ndarray:
    """Return one-based age of the current observable state, using its prefix only."""
    values = np.asarray(state, dtype=int)
    times = pd.DatetimeIndex(timestamps) if timestamps is not None else None
    if times is not None and len(times) != len(values):
        raise ValueError("timestamps and state must have the same length")
    age = np.zeros(len(values), dtype=int)
    for idx, current in enumerate(values):
        if current < 0:
            continue
        continuous = (
            times is None
            or idx == 0
            or times[idx] - times[idx - 1] <= pd.Timedelta("90min")
        )
        if (
            idx > 0
            and continuous
            and current == values[idx - 1]
            and age[idx - 1] > 0
        ):
            age[idx] = age[idx - 1] + 1
        else:
            age[idx] = 1
    return age


def observable_state(
    feature_frame: pd.DataFrame,
    fit_mask: np.ndarray,
    low_quantile: float,
    high_quantile: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Build 12 train-frozen trend x volatility x flow states."""
    fit = feature_frame.loc[np.asarray(fit_mask, dtype=bool)]
    thresholds = {
        "trend_low": float(fit["trend24"].quantile(low_quantile)),
        "trend_high": float(fit["trend24"].quantile(high_quantile)),
        "vol_median": float(fit["vol24"].quantile(0.5)),
        "flow_median": float(fit["flow24"].quantile(0.5)),
    }
    trend = feature_frame["trend24"].to_numpy(float)
    volatility = feature_frame["vol24"].to_numpy(float)
    flow = feature_frame["flow24"].to_numpy(float)
    trend_bucket = np.where(
        trend <= thresholds["trend_low"],
        0,
        np.where(trend >= thresholds["trend_high"], 2, 1),
    )
    volatility_bucket = (volatility >= thresholds["vol_median"]).astype(int)
    flow_bucket = (flow >= thresholds["flow_median"]).astype(int)
    state = trend_bucket * 4 + volatility_bucket * 2 + flow_bucket
    valid = np.isfinite(trend) & np.isfinite(volatility) & np.isfinite(flow)
    return np.where(valid, state, -1), thresholds


def duration_key(
    state: np.ndarray,
    cutpoints: tuple[int, ...],
    timestamps: pd.DatetimeIndex | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    age = causal_run_age(state, timestamps=timestamps)
    bucket = np.digitize(age, cutpoints, right=True)
    bucket_count = len(cutpoints) + 1
    key = state * bucket_count + bucket
    return np.where(state >= 0, key, -1), age


def empirical_exit_hazard(
    state: np.ndarray,
    key: np.ndarray,
    fit_mask: np.ndarray,
) -> dict[int, dict[str, float | int]]:
    exposures: dict[int, int] = {}
    exits: dict[int, int] = {}
    fit = np.asarray(fit_mask, dtype=bool)
    for idx in range(len(state) - 1):
        if not (fit[idx] and fit[idx + 1]) or key[idx] < 0 or state[idx + 1] < 0:
            continue
        current_key = int(key[idx])
        exposures[current_key] = exposures.get(current_key, 0) + 1
        if state[idx + 1] != state[idx]:
            exits[current_key] = exits.get(current_key, 0) + 1
    return {
        current_key: {
            "exposures": count,
            "exits": exits.get(current_key, 0),
            "exit_hazard": (exits.get(current_key, 0) + 1.0) / (count + 2.0),
        }
        for current_key, count in exposures.items()
    }


def map_hourly_key(dates: pd.Series, hourly_index: pd.DatetimeIndex, key: np.ndarray) -> np.ndarray:
    hourly = pd.DataFrame({"date": hourly_index.to_numpy(), "key": key})
    mapped = pd.merge_asof(
        pd.DataFrame({"date": pd.to_datetime(dates), "pos": np.arange(len(dates))}),
        hourly.sort_values("date"),
        on="date",
        direction="backward",
        tolerance=pd.Timedelta("2h"),
    ).sort_values("pos")
    return mapped["key"].fillna(-1).to_numpy(int)


def _signal_hash(signal: np.ndarray) -> str:
    return hashlib.sha256(np.packbits(signal).tobytes()).hexdigest()[:16]


def _duration_label(bucket: int, cutpoints: tuple[int, ...]) -> str:
    if bucket == 0:
        return f"1-{cutpoints[0]}h"
    if bucket < len(cutpoints):
        return f"{cutpoints[bucket - 1] + 1}-{cutpoints[bucket]}h"
    return f">{cutpoints[-1]}h"


def _describe_key(key: int, cutpoints: tuple[int, ...]) -> str:
    bucket_count = len(cutpoints) + 1
    state, duration_bucket = divmod(key, bucket_count)
    trend = ("low", "mid", "high")[state // 4]
    volatility = "high" if (state % 4) // 2 else "low"
    flow = "high" if state % 2 else "low"
    return (
        f"trend={trend}, volatility={volatility}, flow={flow}, "
        f"state_age={_duration_label(duration_bucket, cutpoints)}"
    )


def _rank_key(row: dict) -> tuple[float, float, float, float]:
    ratios = [max(0.0, float(row[name]["ratio"])) for name in SELECTION_WINDOWS]
    geometric = float(np.prod(np.maximum(ratios, 1e-12)) ** (1.0 / len(ratios)))
    return (
        min(ratios),
        geometric,
        min(float(row[name]["return_pct"]) for name in SELECTION_WINDOWS),
        min(float(row[name]["cagr_pct"]) for name in SELECTION_WINDOWS),
    )


def _trade_returns(
    market: pd.DataFrame,
    dates: pd.Series,
    long_active: np.ndarray,
    cfg: Config,
    window: str,
) -> np.ndarray:
    """Reproduce ``sim`` trade returns for a long-only signal."""
    start, end = WINDOWS[window]
    window_mask = _split_mask(dates, start, end)
    opens = market["open"].to_numpy(float)
    highs = market["high"].to_numpy(float)
    lows = market["low"].to_numpy(float)
    cost = (cfg.fee_rate + cfg.slippage_rate) * cfg.leverage
    positions = np.arange(143, len(market) - 578, 12, dtype=np.int64)
    positions = positions[window_mask[positions] & long_active[positions]]
    next_allowed = 0
    returns: list[float] = []
    for pos in positions:
        if pos < next_allowed:
            continue
        entry_pos = int(pos) + 1
        cap = entry_pos + 576
        if cap >= len(market) or not window_mask[cap]:
            continue
        entry = opens[entry_pos]
        exit_pos = cap
        exit_return = opens[cap] / entry - 1.0
        for bar in range(entry_pos, cap):
            if lows[bar] <= entry * (1.0 - 10.0):
                exit_return = -10.0
                exit_pos = bar
                break
            if highs[bar] >= entry * (1.0 + 10.0):
                exit_return = 10.0
                exit_pos = bar
                break
        trade_multiplier = (1.0 - cost) * max(0.0, 1.0 + cfg.leverage * exit_return) * (
            1.0 - cost
        )
        returns.append(trade_multiplier - 1.0)
        next_allowed = exit_pos + 1
    return np.asarray(returns, dtype=float)


def _bootstrap_mean_trade_return(returns: np.ndarray, seed: int = 713) -> dict[str, float | int]:
    values = np.asarray(returns, dtype=float)
    if len(values) == 0:
        return {
            "trades": 0,
            "mean_trade_return_pct": 0.0,
            "ci95_low_pct": 0.0,
            "ci95_high_pct": 0.0,
            "probability_mean_positive": 0.0,
        }
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(20_000, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(samples, (0.025, 0.975))
    return {
        "trades": len(values),
        "mean_trade_return_pct": float(values.mean() * 100.0),
        "ci95_low_pct": float(low * 100.0),
        "ci95_high_pct": float(high * 100.0),
        "probability_mean_positive": float(np.mean(samples > 0.0)),
    }


def run(cfg: Config) -> dict:
    for name, bounds in WINDOWS.items():
        state_sim.W[name] = bounds

    market = _load_market(cfg)
    dates = pd.to_datetime(market["date"])
    base = build_market_feature_frame(market, window_size=144)
    features = pd.concat([base, build_interest_features(market, base)], axis=1)
    features = features.loc[:, ~features.columns.duplicated(keep="last")]
    setup = _alpha_active(features, "long_minimal_funding_premium")
    _, hourly_feature = hourly_features(market)
    fit_hour = np.asarray(
        (hourly_feature.index >= WINDOWS["fit2020_2022"][0])
        & (hourly_feature.index < WINDOWS["fit2020_2022"][1]),
        dtype=bool,
    )
    fit_market = _split_mask(dates, *WINDOWS["fit2020_2022"])
    positions = np.arange(143, len(market) - 578, 12)
    trend_quantiles = (0.33, 0.67)
    duration_cutpoints = (1, 6, 24, 72)
    raw_rows: list[dict] = []

    low, high = trend_quantiles
    cutpoints = duration_cutpoints
    state, thresholds = observable_state(hourly_feature, fit_hour, low, high)
    hourly_key, _ = duration_key(state, cutpoints, timestamps=hourly_feature.index)
    hazards = empirical_exit_hazard(state, hourly_key, fit_hour)
    mapped_key = map_hourly_key(dates, hourly_feature.index, hourly_key)
    quality: dict[int, list[float]] = {}
    next_allowed = 0
    for pos in positions[setup[positions] & fit_market[positions]]:
        exit_pos = int(pos) + 577
        if (
            pos < next_allowed
            or mapped_key[pos] < 0
            or exit_pos >= len(fit_market)
            or not fit_market[exit_pos]
        ):
            continue
        event = _event_path(
            market,
            int(pos),
            side="long",
            hold=576,
            cost_rate=0.0006,
            entry_delay=1,
            leverage=0.5,
        )
        if event is None:
            continue
        quality.setdefault(int(mapped_key[pos]), []).append(float(event[2]))
        next_allowed = exit_pos

    for min_count, min_edge, max_hazard in itertools.product(
        (3, 5, 8, 12),
        (0.0, 0.002, 0.005, 0.01),
        (0.25, 0.5, 1.0),
    ):
        allowed = sorted(
            current_key
            for current_key, returns in quality.items()
            if len(returns) >= min_count
            and float(np.mean(returns)) >= min_edge
            and current_key in hazards
            and float(hazards[current_key]["exit_hazard"]) <= max_hazard
        )
        if not allowed:
            continue
        long_active = setup & np.isin(mapped_key, allowed)
        short_active = np.zeros(len(market), dtype=bool)
        fit_stats = sim(
            market,
            dates,
            long_active,
            short_active,
            cfg,
            576,
            12,
            10.0,
            10.0,
            "fit2020_2022",
        )
        holdout_stats = sim(
            market,
            dates,
            long_active,
            short_active,
            cfg,
            576,
            12,
            10.0,
            10.0,
            "holdout2023",
        )
        if fit_stats["trades"] < 60 or holdout_stats["trades"] < 8:
            continue
        row = {
            "trend_quantiles": [low, high],
            "state_thresholds": thresholds,
            "duration_cutpoints_hours": list(cutpoints),
            "min_fit_key_trades": min_count,
            "min_fit_mean_trade_return": min_edge,
            "max_fit_exit_hazard": max_hazard,
            "allowed_keys": allowed,
            "allowed_key_descriptions": {
                str(current_key): _describe_key(current_key, cutpoints)
                for current_key in allowed
            },
            "key_quality": {
                str(current_key): {
                    "n": len(quality[current_key]),
                    "mean_trade_return": float(np.mean(quality[current_key])),
                    **hazards[current_key],
                }
                for current_key in allowed
            },
            "fit2020_2022": fit_stats,
            "holdout2023": holdout_stats,
            "signal_hash": _signal_hash(long_active),
            "_packed_signal": np.packbits(long_active),
            "_mapped_key": mapped_key.astype(np.int16),
        }
        for year_window in ("year2020", "year2021", "year2022"):
            row[year_window] = sim(
                market,
                dates,
                long_active,
                short_active,
                cfg,
                576,
                12,
                10.0,
                10.0,
                year_window,
            )
        if min(row[name]["trades"] for name in SELECTION_WINDOWS) < 8:
            continue
        raw_rows.append(row)

    raw_rows.sort(key=_rank_key, reverse=True)
    selected: list[dict] = []
    seen_signals: set[str] = set()
    for row in raw_rows:
        if row["signal_hash"] in seen_signals:
            continue
        seen_signals.add(row["signal_hash"])
        if len(selected) < 100:
            selected.append(row)

    selected_signals: dict[str, np.ndarray] = {}
    selected_keys: dict[str, np.ndarray] = {}
    for row in selected:
        selected_signals[row["signal_hash"]] = np.unpackbits(
            row.pop("_packed_signal"), count=len(market)
        ).astype(bool)
        selected_keys[row["signal_hash"]] = row.pop("_mapped_key")
        short_active = np.zeros(len(market), dtype=bool)
        for window in ("test2024", "eval2025", "ytd2026"):
            row[window] = sim(
                market,
                dates,
                selected_signals[row["signal_hash"]],
                short_active,
                cfg,
                576,
                12,
                10.0,
                10.0,
                window,
            )
        row["worst_selection_ratio"] = _rank_key(row)[0]
        row["passes_alpha_pool"] = bool(
            row["worst_selection_ratio"] >= 1.0
            and row["test2024"]["ratio"] >= 3.0
            and row["eval2025"]["ratio"] >= 3.0
            and row["eval2025"]["trades"] >= 8
        )
        row["passes_live_grade"] = bool(
            row["passes_alpha_pool"]
            and row["ytd2026"]["ratio"] >= 5.0
            and row["ytd2026"]["trades"] >= 6
        )

    baseline = {
        window: sim(
            market,
            dates,
            setup,
            np.zeros(len(market), dtype=bool),
            cfg,
            576,
            12,
            10.0,
            10.0,
            window,
        )
        for window in WINDOWS
    }
    yearly: dict[str, dict] = {}
    stress: dict[str, dict] = {}
    leave_one: dict[str, dict] = {}
    bootstrap: dict[str, dict] = {}
    if selected:
        top = selected[0]
        top_active = selected_signals[top["signal_hash"]]
        top_key = selected_keys[top["signal_hash"]]
        short_active = np.zeros(len(market), dtype=bool)
        for year, window in (
            (2020, "year2020"),
            (2021, "year2021"),
            (2022, "year2022"),
            (2023, "holdout2023"),
            (2024, "test2024"),
            (2025, "eval2025"),
            (2026, "ytd2026"),
        ):
            yearly[str(year)] = sim(
                market, dates, top_active, short_active, cfg, 576, 12, 10.0, 10.0, window
            )
        for bps in (6, 8, 10, 15):
            stressed_cfg = replace(cfg, fee_rate=max(0.0, bps / 10000 - cfg.slippage_rate))
            stress[str(bps)] = {
                window: sim(
                    market,
                    dates,
                    top_active,
                    short_active,
                    stressed_cfg,
                    576,
                    12,
                    10.0,
                    10.0,
                    window,
                )
                for window in ("fit2020_2022", "holdout2023", "test2024", "eval2025", "ytd2026")
            }
        for dropped in top["allowed_keys"]:
            active = setup & np.isin(
                top_key, [key for key in top["allowed_keys"] if key != dropped]
            )
            leave_one[str(dropped)] = {
                window: sim(
                    market, dates, active, short_active, cfg, 576, 12, 10.0, 10.0, window
                )
                for window in ("fit2020_2022", "holdout2023", "test2024", "eval2025", "ytd2026")
            }
        for window in ("holdout2023", "test2024", "eval2025", "ytd2026"):
            bootstrap[window] = _bootstrap_mean_trade_return(
                _trade_returns(market, dates, top_active, cfg, window)
            )
        top["passes_statistical_screen"] = bool(
            bootstrap["test2024"]["trades"] >= 8
            and bootstrap["eval2025"]["trades"] >= 8
            and bootstrap["test2024"]["probability_mean_positive"] >= 0.95
            and bootstrap["eval2025"]["probability_mean_positive"] >= 0.90
        )
        top["passes_alpha_pool"] = bool(
            top["passes_alpha_pool"] and top["passes_statistical_screen"]
        )
        top["passes_live_grade"] = bool(
            top["passes_live_grade"] and top["passes_statistical_screen"]
        )

    output = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "base_setup": "long_minimal_funding_premium",
        "protocol": (
            "observable explicit-duration semi-Markov entry gate; one predeclared state binning and "
            "duration scheme; bins, duration-key quality, and exit hazards fit 2020-2022; rank by "
            "worst CAGR/strict-MDD across each fit year and 2023 holdout; Test2024 first untouched "
            "overlay test; Eval2025/2026 report-only; hold576; 6bp/side; strict MDD"
        ),
        "source": "https://www.cs.ubc.ca/~murphyk/papers/segment.pdf",
        "selection_caveat": (
            "The semi-Markov overlay is frozen before Eval2025/2026, but the underlying base setup "
            "has prior research-history exposure. A passing composite remains research-forward shadow."
        ),
        "tested_variants": len(raw_rows),
        "distinct_signal_variants": len(seen_signals),
        "baseline": baseline,
        "yearly_top_candidate": yearly,
        "cost_stress_bps_per_side": stress,
        "leave_one_key_out": leave_one,
        "top_trade_return_bootstrap": bootstrap,
        "selected": selected,
        "alpha_pool_qualifiers": (
            [selected[0]] if selected and selected[0]["passes_alpha_pool"] else []
        ),
        "live_grade": [selected[0]] if selected and selected[0]["passes_live_grade"] else [],
    }
    Path(cfg.output).write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
            default=lambda value: value.item() if isinstance(value, np.generic) else str(value),
        )
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--funding-csv", default="")
    parser.add_argument("--premium-csv", default="")
    parser.add_argument("--exclude-from", default="2026-06-02")
    args = parser.parse_args()
    output = run(Config(**vars(args)))
    print(
        json.dumps(
            {
                "tested_variants": output["tested_variants"],
                "distinct_signal_variants": output["distinct_signal_variants"],
                "qualifiers": len(output["alpha_pool_qualifiers"]),
                "live": len(output["live_grade"]),
                "top": output["selected"][:5],
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
