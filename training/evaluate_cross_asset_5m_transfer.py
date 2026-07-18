"""Evaluate the preregistered QQQ/KODEX200/GLD five-minute transfer battery.

The evaluator deliberately has no policy-selection surface.  Every threshold is
fit on the frozen train split, then the same trade clock is reported on test and
eval.  Entries use the next available regular-session bar and strict drawdown is
marked through every held bar's high/low.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from preprocessing.market_features import build_market_feature_frame
from training import preregister_cross_asset_5m_transfer as prereg
from training.cross_asset_5m_data import run_source_audit
from training.event_candidate_pool_probe import _feature_candidates
from training.search_persistent_barrier_annihilation_alpha import persistent_barrier_features


OUTPUT = "results/cross_asset_5m_transfer_evaluation_2026-07-19.json"
DOCS_OUTPUT = "docs/cross-asset-5m-transfer-evaluation-2026-07-19.md"
SOURCE_AUDIT_OUTPUT = "results/cross_asset_5m_source_audit_2026-07-19.json"

SPLITS = {
    "train": ("2024-09-01", "2025-07-01"),
    "test": ("2025-07-01", "2026-01-01"),
    "eval": ("2026-01-01", "2026-07-19"),
}
REX_CONTEXT_BARS = 8_640
BASE_COST_BPS = 5.0
STRESS_COST_BPS = 10.0
EXACT_SIGNFLIP_CLUSTERS = 16
MONTE_CARLO_SIGNFLIP_SAMPLES = 65_536
PROVENANCE_FILES = (
    "training/evaluate_cross_asset_5m_transfer.py",
    "training/cross_asset_5m_data.py",
    "training/preregister_cross_asset_5m_transfer.py",
    "preprocessing/market_features.py",
    "training/event_candidate_pool_probe.py",
    "training/search_persistent_barrier_annihilation_alpha.py",
)


@dataclass(frozen=True)
class PolicySpec:
    name: str
    family: str
    kind: str
    quantile: float
    hold_bars: int


@dataclass(frozen=True)
class Trade:
    """Minimal immutable trade clock used by unit tests and adapters."""

    signal_index: int
    entry_index: int
    exit_index: int
    side: int


POLICIES = (
    PolicySpec(
        name="rex_htf_pullback_reclaim_5m",
        family="rex_htf_pullback_reclaim",
        kind="rex",
        quantile=0.75,
        hold_bars=144,
    ),
    PolicySpec(
        name="rex_multiscale_extreme_fade_5m",
        family="rex_multiscale_extreme_fade",
        kind="rex",
        quantile=0.75,
        hold_bars=144,
    ),
    PolicySpec(
        name="persistent_barrier_mass_density_fade_5m",
        family="mass_density",
        kind="persistent_barrier",
        quantile=0.975,
        hold_bars=288,
    ),
)


def canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def self_hash(payload: dict[str, Any], field: str = "result_hash") -> str:
    clean = dict(payload)
    clean.pop(field, None)
    return sha256_bytes(canonical_json(clean).encode())


def canonical_result_hash(payload: dict[str, Any]) -> str:
    return self_hash(payload)


def code_provenance() -> dict[str, str]:
    return {path: sha256_bytes(Path(path).read_bytes()) for path in PROVENANCE_FILES}


def _float(value: float, *, default: float = 0.0) -> float:
    result = float(value)
    return result if math.isfinite(result) else float(default)


def split_mask(dates: pd.Series, start: str, end: str) -> np.ndarray:
    values = pd.to_datetime(dates)
    return np.asarray((values >= pd.Timestamp(start)) & (values < pd.Timestamp(end)), dtype=bool)


def _mask_length_gated_htf_features(features: pd.DataFrame) -> pd.DataFrame:
    """Remove a dataset-length dependency in the shared HTF builder.

    The shared builder skips an entire HTF family when the *final* frame is
    shorter than a coarse minimum.  Without this row-wise mask, appending future
    rows can turn old features from zero to non-zero.  Masking each family until
    its own source-row support makes the five-minute projection prefix-invariant.
    """
    out = features.copy()
    support = {
        "htf_4h_": 4 * 60 * 4,
        "htf_1d_": 24 * 60 * 4,
        "htf_3d_": 3 * 24 * 60 * 4,
        "htf_1w_": 7 * 24 * 60,
    }
    positions = np.arange(len(out))
    for prefix, minimum in support.items():
        columns = [column for column in out.columns if str(column).startswith(prefix)]
        if columns:
            out.loc[positions < minimum, columns] = 0.0
    return out


def build_rex_arrays(frame: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    features = build_market_feature_frame(frame, window_size=144)
    features = _mask_length_gated_htf_features(features)
    families = _feature_candidates(features)
    context = np.arange(len(frame)) >= REX_CONTEXT_BARS - 1
    decision = np.arange(len(frame)) % 12 == 0
    active_clock = context & decision
    result: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for family in ("rex_htf_pullback_reclaim", "rex_multiscale_extreme_fade"):
        strength, direction = families[family]
        result[family] = (
            np.asarray(strength, dtype=float),
            np.asarray(direction, dtype=float),
            active_clock.copy(),
        )
    return result


def build_persistent_arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dates = pd.to_datetime(frame["date"])
    persistence = persistent_barrier_features(
        np.log(pd.to_numeric(frame["close"], errors="raise").to_numpy(float)),
        dates,
        horizon=2_016,
        minimum_prominence_z=0.5,
    )
    strength = pd.to_numeric(persistence["mass_density"], errors="coerce").to_numpy(float)
    # The preregistered mechanism fades the completed traversal.
    direction = -pd.to_numeric(persistence["side"], errors="coerce").to_numpy(float)
    decision = (dates.dt.minute == 0).to_numpy(bool)
    return strength, direction, decision


def build_policy_arrays(
    frame: pd.DataFrame,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    rex = build_rex_arrays(frame)
    persistent = build_persistent_arrays(frame)
    return {
        "rex_htf_pullback_reclaim_5m": rex["rex_htf_pullback_reclaim"],
        "rex_multiscale_extreme_fade_5m": rex["rex_multiscale_extreme_fade"],
        "persistent_barrier_mass_density_fade_5m": persistent,
    }


def fit_train_threshold(
    strength: np.ndarray,
    train_mask: np.ndarray,
    quantile: float,
    *,
    minimum_support: int = 100,
) -> tuple[float, int]:
    values = np.asarray(strength, dtype=float)
    reference = values[np.asarray(train_mask, dtype=bool) & np.isfinite(values) & (values > 0.0)]
    if len(reference) < minimum_support:
        raise ValueError(f"insufficient positive train strength rows: {len(reference)}")
    return float(np.quantile(reference, float(quantile))), int(len(reference))


def fit_threshold(
    strength: np.ndarray,
    dates: pd.Series,
    *,
    quantile: float,
    minimum_support: int = 100,
) -> tuple[float, int]:
    """Compatibility surface that explicitly derives the fit mask from train."""
    return fit_train_threshold(
        strength,
        split_mask(pd.Series(dates), *SPLITS["train"]),
        quantile,
        minimum_support=minimum_support,
    )


def build_trades(
    active: np.ndarray,
    side: np.ndarray,
    dates: pd.Series,
    *,
    split: str,
    hold_bars: int,
) -> list[Trade]:
    """Build a minimal next-bar trade clock from already-frozen signals."""
    if split not in SPLITS:
        raise KeyError(split)
    timestamps = pd.to_datetime(pd.Series(dates))
    if len(active) != len(side) or len(active) != len(timestamps):
        raise ValueError("signal arrays and dates must align")
    in_split = split_mask(timestamps, *SPLITS[split])
    rows: list[Trade] = []
    next_entry_allowed = 0
    for signal_index in np.flatnonzero(np.asarray(active, dtype=bool) & in_split):
        entry_index = int(signal_index) + 1
        exit_index = entry_index + int(hold_bars)
        if exit_index >= len(timestamps) or entry_index < next_entry_allowed:
            continue
        if not (in_split[entry_index] and in_split[exit_index]):
            continue
        direction = int(np.sign(float(side[signal_index])))
        if direction == 0:
            continue
        rows.append(Trade(int(signal_index), entry_index, exit_index, direction))
        next_entry_allowed = exit_index
    return rows


def build_trade_clock(
    frame: pd.DataFrame,
    strength: np.ndarray,
    direction: np.ndarray,
    decision_mask: np.ndarray,
    *,
    threshold: float,
    hold_bars: int,
    split_start: str,
    split_end: str,
) -> list[dict[str, Any]]:
    """Create split-contained, non-overlapping next-bar-open trades."""
    count = len(frame)
    if not all(len(value) == count for value in (strength, direction, decision_mask)):
        raise ValueError("policy arrays must align with the market frame")
    dates = pd.to_datetime(frame["date"])
    in_split = split_mask(dates, split_start, split_end)
    eligible = (
        in_split
        & np.asarray(decision_mask, dtype=bool)
        & np.isfinite(strength)
        & (np.asarray(strength, dtype=float) > float(threshold))
        & np.isfinite(direction)
        & (np.asarray(direction, dtype=float) != 0.0)
    )
    rows: list[dict[str, Any]] = []
    next_entry_allowed = 0
    for signal_index in np.flatnonzero(eligible):
        entry_index = int(signal_index) + 1
        exit_index = entry_index + int(hold_bars)
        if exit_index >= count or entry_index < next_entry_allowed:
            continue
        if not (in_split[entry_index] and in_split[exit_index]):
            continue
        side = 1 if float(direction[signal_index]) > 0.0 else -1
        entry_price = float(frame.iloc[entry_index]["open"])
        exit_price = float(frame.iloc[exit_index]["open"])
        if min(entry_price, exit_price) <= 0.0:
            raise ValueError("non-positive execution price")
        rows.append(
            {
                "signal_index": int(signal_index),
                "entry_index": entry_index,
                "exit_index": exit_index,
                "signal_time": pd.Timestamp(frame.iloc[signal_index]["timestamp_utc"]).isoformat(),
                "entry_time": pd.Timestamp(frame.iloc[entry_index]["timestamp_utc"]).isoformat(),
                "exit_time": pd.Timestamp(frame.iloc[exit_index]["timestamp_utc"]).isoformat(),
                "exit_local_date": pd.Timestamp(frame.iloc[exit_index]["date"]).isoformat(),
                "side": side,
                "strength": float(strength[signal_index]),
                "entry_price": entry_price,
                "exit_price": exit_price,
            }
        )
        next_entry_allowed = exit_index
    return rows


def _calendar_months(start: str, end: str) -> list[str]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) - pd.Timedelta(nanoseconds=1)
    if start_ts.tzinfo is not None:
        start_ts = start_ts.tz_localize(None)
    if end_ts.tzinfo is not None:
        end_ts = end_ts.tz_localize(None)
    first = start_ts.to_period("M")
    last_inclusive = end_ts.to_period("M")
    return [str(value) for value in pd.period_range(first, last_inclusive, freq="M")]


def weekly_cluster_signflip(
    trades: list[dict[str, Any]] | list[float],
    event_times: list[str] | None = None,
    *,
    permutations: int = MONTE_CARLO_SIGNFLIP_SAMPLES,
    seed: int | None = None,
) -> dict[str, Any]:
    """One-sided weekly Rademacher sign-flip test on net log returns."""
    by_week: dict[str, float] = {}
    clock_parts: list[str] = []
    records: list[dict[str, Any]]
    if event_times is not None:
        if len(trades) != len(event_times):
            raise ValueError("returns and event times must align")
        records = [
            {
                "net_factor": 1.0 + float(value),
                "entry_time": str(timestamp),
                "exit_time": str(timestamp),
            }
            for value, timestamp in zip(trades, event_times, strict=True)
        ]
    else:
        records = list(trades)  # type: ignore[arg-type]
    for trade in records:
        factor = float(trade["net_factor"])
        if factor <= 0.0:
            return {
                "cluster_count": 0,
                "p_value_one_sided": 1.0,
                "method": "invalid_nonpositive_factor",
                "samples": 0,
                "clock_hash": "",
            }
        exit_time = pd.Timestamp(trade["exit_time"])
        iso = exit_time.isocalendar()
        key = f"{int(iso.year):04d}-W{int(iso.week):02d}"
        by_week[key] = by_week.get(key, 0.0) + math.log(factor)
        clock_parts.append(f"{trade['entry_time']}|{trade['exit_time']}")
    values = np.asarray([by_week[key] for key in sorted(by_week)], dtype=float)
    clock_hash = sha256_bytes("\n".join(clock_parts).encode())
    clusters = len(values)
    if clusters == 0:
        return {
            "cluster_count": 0,
            "p_value_one_sided": 1.0,
            "method": "no_clusters",
            "samples": 0,
            "clock_hash": clock_hash,
        }
    observed = float(values.sum())
    if clusters <= EXACT_SIGNFLIP_CLUSTERS:
        totals = np.fromiter(
            (float(np.dot(values, signs)) for signs in itertools.product((-1.0, 1.0), repeat=clusters)),
            dtype=float,
            count=2**clusters,
        )
        method = "exact"
    else:
        effective_seed = int(clock_hash[:16], 16) % (2**32) if seed is None else int(seed)
        rng = np.random.default_rng(effective_seed)
        totals = np.empty(int(permutations), dtype=float)
        batch = 4_096
        for start in range(0, len(totals), batch):
            stop = min(start + batch, len(totals))
            signs = rng.integers(0, 2, size=(stop - start, clusters), dtype=np.int8) * 2 - 1
            totals[start:stop] = signs @ values
        method = "deterministic_monte_carlo"
    tolerance = max(1e-15, abs(observed) * 1e-12)
    p_value = float(np.mean(totals >= observed - tolerance))
    return {
        "cluster_count": clusters,
        "p_value_one_sided": p_value,
        "method": method,
        "samples": int(len(totals)),
        "clock_hash": clock_hash,
        "observed_log_return": observed,
    }


def simulate_trade_clock(
    frame: pd.DataFrame,
    trade_clock: list[dict[str, Any]],
    *,
    split_start: str,
    split_end: str,
    cost_bps_per_side: float,
    flip_direction: bool = False,
) -> dict[str, Any]:
    """Simulate fixed-notional 1x trades and conservative intratrade MDD."""
    cost = float(cost_bps_per_side) / 10_000.0
    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    realized: list[dict[str, Any]] = []

    def mark(value: float) -> None:
        nonlocal peak, max_drawdown
        if not math.isfinite(value):
            raise ValueError("non-finite marked equity")
        peak = max(peak, value)
        if peak > 0.0:
            max_drawdown = max(max_drawdown, (peak - value) / peak)

    for original in trade_clock:
        trade = dict(original)
        side = int(trade["side"]) * (-1 if flip_direction else 1)
        entry = float(trade["entry_price"])
        exit_price = float(trade["exit_price"])
        pre_trade_equity = equity
        entry_equity = pre_trade_equity * (1.0 - cost)
        mark(entry_equity)
        for position in range(int(trade["entry_index"]), int(trade["exit_index"])):
            high = float(frame.iloc[position]["high"])
            low = float(frame.iloc[position]["low"])
            favorable, adverse = (high, low) if side > 0 else (low, high)
            favorable_equity = entry_equity * (1.0 + side * (favorable / entry - 1.0))
            adverse_equity = entry_equity * (1.0 + side * (adverse / entry - 1.0))
            mark(favorable_equity)
            mark(adverse_equity)
        gross_return = side * (exit_price / entry - 1.0)
        mark(entry_equity * (1.0 + gross_return))
        net_factor = (1.0 - cost) * (1.0 + gross_return) * (1.0 - cost)
        equity = pre_trade_equity * net_factor
        mark(equity)
        trade.update(
            {
                "side": side,
                "gross_return": gross_return,
                "net_return": net_factor - 1.0,
                "net_factor": net_factor,
                "equity_after": equity,
            }
        )
        realized.append(trade)

    duration_days = (pd.Timestamp(split_end) - pd.Timestamp(split_start)).total_seconds() / 86_400.0
    if equity <= 0.0:
        cagr = -1.0
    else:
        annual_log_return = math.log(equity) * (365.2425 / duration_days)
        # Tiny synthetic test windows can imply an astronomically annualized
        # number; real frozen splits are months long.  Keep JSON finite.
        cagr = math.exp(min(annual_log_return, math.log(1e12))) - 1.0
    mdd_pct = max_drawdown * 100.0
    cagr_pct = cagr * 100.0
    ratio = cagr_pct / mdd_pct if mdd_pct > 0.0 else (0.0 if cagr_pct == 0.0 else 1e9)

    monthly_factors = {month: 1.0 for month in _calendar_months(split_start, split_end)}
    monthly_trade_counts = {month: 0 for month in monthly_factors}
    for trade in realized:
        month = str(pd.Timestamp(trade["exit_local_date"]).to_period("M"))
        monthly_factors[month] *= float(trade["net_factor"])
        monthly_trade_counts[month] += 1
    monthly_returns = {month: (factor - 1.0) * 100.0 for month, factor in monthly_factors.items()}
    months_with_trade = sum(value > 0 for value in monthly_trade_counts.values())
    positive_months = sum(value > 0.0 for value in monthly_returns.values())
    total_months = len(monthly_returns)

    signflip = weekly_cluster_signflip(realized)
    net_returns = np.asarray([trade["net_return"] for trade in realized], dtype=float)
    return {
        "absolute_return_pct": _float((equity - 1.0) * 100.0),
        "cagr_pct": _float(cagr_pct, default=-100.0),
        "strict_mdd_pct": _float(mdd_pct),
        "cagr_to_strict_mdd": _float(ratio),
        "trades": int(len(realized)),
        "win_rate_pct": _float(float(np.mean(net_returns > 0.0) * 100.0) if len(net_returns) else 0.0),
        "positive_calendar_month_share": _float(positive_months / total_months if total_months else 0.0),
        "positive_calendar_months": int(positive_months),
        "calendar_months": int(total_months),
        "months_with_trade": int(months_with_trade),
        "monthly_return_pct": monthly_returns,
        "weekly_cluster_signflip": signflip,
        "final_equity": _float(equity),
        "cost_bps_per_side": float(cost_bps_per_side),
        "trade_clock_hash": signflip["clock_hash"],
        "trade_records": realized,
    }


def simulate(
    frame: pd.DataFrame,
    trades: list[Trade],
    *,
    split: str,
    cost_bps_per_side: float,
) -> dict[str, Any]:
    """Adapter for deterministic metric tests using a minimal ``Trade`` list."""
    if split not in SPLITS:
        raise KeyError(split)
    market = frame.copy()
    if "timestamp_utc" not in market:
        raise ValueError("timestamp_utc is required")
    if "date" not in market:
        market["date"] = pd.to_datetime(market["timestamp_utc"]).dt.tz_convert(None)
    records: list[dict[str, Any]] = []
    for trade in trades:
        entry = float(market.iloc[trade.entry_index]["open"])
        exit_price = float(market.iloc[trade.exit_index]["open"])
        records.append(
            {
                "signal_index": trade.signal_index,
                "entry_index": trade.entry_index,
                "exit_index": trade.exit_index,
                "signal_time": pd.Timestamp(market.iloc[trade.signal_index]["timestamp_utc"]).isoformat(),
                "entry_time": pd.Timestamp(market.iloc[trade.entry_index]["timestamp_utc"]).isoformat(),
                "exit_time": pd.Timestamp(market.iloc[trade.exit_index]["timestamp_utc"]).isoformat(),
                "exit_local_date": pd.Timestamp(market.iloc[trade.exit_index]["date"]).isoformat(),
                "side": trade.side,
                "strength": 1.0,
                "entry_price": entry,
                "exit_price": exit_price,
            }
        )
    return simulate_trade_clock(
        market,
        records,
        split_start=SPLITS[split][0],
        split_end=SPLITS[split][1],
        cost_bps_per_side=cost_bps_per_side,
    )


def _metric_without_records(metric: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metric.items() if key != "trade_records"}


def evaluate_asset_policy(
    frame: pd.DataFrame,
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    policy: PolicySpec,
) -> dict[str, Any]:
    strength, direction, decision = arrays
    train_mask = split_mask(frame["date"], *SPLITS["train"])
    threshold, support = fit_train_threshold(strength, train_mask, policy.quantile)
    splits: dict[str, Any] = {}
    for split_name, (start, end) in SPLITS.items():
        trades = build_trade_clock(
            frame,
            strength,
            direction,
            decision,
            threshold=threshold,
            hold_bars=policy.hold_bars,
            split_start=start,
            split_end=end,
        )
        base = simulate_trade_clock(
            frame,
            trades,
            split_start=start,
            split_end=end,
            cost_bps_per_side=BASE_COST_BPS,
        )
        stress = simulate_trade_clock(
            frame,
            trades,
            split_start=start,
            split_end=end,
            cost_bps_per_side=STRESS_COST_BPS,
        )
        direction_flip = simulate_trade_clock(
            frame,
            trades,
            split_start=start,
            split_end=end,
            cost_bps_per_side=BASE_COST_BPS,
            flip_direction=True,
        )
        if not (
            base["trade_clock_hash"]
            == stress["trade_clock_hash"]
            == direction_flip["trade_clock_hash"]
        ):
            raise RuntimeError("stress/direction controls changed the frozen trade clock")
        splits[split_name] = {
            "base": _metric_without_records(base),
            "stress_10bp": _metric_without_records(stress),
            "direction_flip": _metric_without_records(direction_flip),
        }
    evaluation = splits["eval"]
    base = evaluation["base"]
    stress = evaluation["stress_10bp"]
    flip = evaluation["direction_flip"]
    gates = {
        "absolute_return_positive": base["absolute_return_pct"] > 0.0,
        "cagr_to_strict_mdd_at_least_3": base["cagr_to_strict_mdd"] >= 3.0,
        "strict_mdd_at_most_15pct": base["strict_mdd_pct"] <= 15.0,
        "at_least_20_trades": base["trades"] >= 20,
        "at_least_4_months_with_trade": base["months_with_trade"] >= 4,
        "stress_absolute_return_positive": stress["absolute_return_pct"] > 0.0,
        "positive_calendar_month_share_at_least_60pct": base["positive_calendar_month_share"] >= 0.60,
        "weekly_cluster_signflip_p_at_most_10pct": base["weekly_cluster_signflip"]["p_value_one_sided"] <= 0.10,
        "direction_flip_ratio_lower": flip["cagr_to_strict_mdd"] < base["cagr_to_strict_mdd"],
    }
    return {
        "threshold": threshold,
        "positive_train_strength_rows": support,
        "splits": splits,
        "eval_gates": gates,
        "eval_pass": bool(all(gates.values())),
    }


def prefix_invariance_check(
    frame: pd.DataFrame,
    full_arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    dates = pd.to_datetime(frame["date"])
    cutoffs: list[int] = []
    for boundary in (SPLITS["train"][1], SPLITS["test"][1], SPLITS["eval"][1]):
        candidates = np.flatnonzero(dates < pd.Timestamp(boundary))
        if len(candidates):
            cutoffs.append(int(candidates[-1]))
    cutoffs = sorted(set(cutoffs))
    checks: list[dict[str, Any]] = []
    for cutoff in cutoffs:
        prefix = frame.iloc[: cutoff + 1].copy().reset_index(drop=True)
        prefix_arrays = build_policy_arrays(prefix)
        for policy in POLICIES:
            full_strength, full_direction, full_decision = full_arrays[policy.name]
            prefix_strength, prefix_direction, prefix_decision = prefix_arrays[policy.name]
            positions = [cutoff]
            if policy.kind == "persistent_barrier":
                hourly = np.flatnonzero((dates.iloc[: cutoff + 1].dt.minute == 0).to_numpy(bool))
                positions = [int(hourly[-1])] if len(hourly) else []
            for position in positions:
                a = float(full_strength[position])
                b = float(prefix_strength[position])
                strength_equal = (not math.isfinite(a) and not math.isfinite(b)) or math.isclose(
                    a, b, rel_tol=1e-11, abs_tol=1e-12
                )
                row = {
                    "policy": policy.name,
                    "position": position,
                    "timestamp": pd.Timestamp(frame.iloc[position]["timestamp_utc"]).isoformat(),
                    "strength_equal": strength_equal,
                    "direction_equal": float(full_direction[position]) == float(prefix_direction[position]),
                    "decision_equal": bool(full_decision[position]) == bool(prefix_decision[position]),
                }
                row["pass"] = bool(row["strength_equal"] and row["direction_equal"] and row["decision_equal"])
                checks.append(row)
    return {
        "scope": "sampled split-boundary prefixes; causal builders are additionally regression-tested",
        "pass": bool(checks and all(row["pass"] for row in checks)),
        "checks": checks,
    }


def _verify_source_report(report: dict[str, Any]) -> None:
    if report.get("result_hash") != self_hash(report):
        raise RuntimeError("source-audit self hash mismatch")
    frozen = prereg.manifest()
    if report.get("preregistration_manifest_hash") != frozen["manifest_hash"]:
        raise RuntimeError("source audit does not match the current preregistration")


def render_docs(report: dict[str, Any]) -> str:
    lines = [
        "# Cross-asset five-minute transfer evaluation — 2026-07-19",
        "",
        f"Preregistration: `{report['preregistration_manifest_hash']}`",
        f"Source audit: `{report['source_audit_result_hash']}`",
        f"Result: `{report['result_hash']}`",
        "",
        "This is a five-minute regular-session evaluation. It is separate from the earlier daily-bar transfer check.",
        "Thresholds are fit on train only; test/eval never select, rerank, or repair a policy.",
        "REX thresholds use all positive finite train strength rows before applying the frozen 12-bar decision clock, matching the source scanner.",
        "",
        "## Eval headline (5 bp per side)",
        "",
        "| Policy | Asset | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades | Positive months | Weekly p | Pass |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for policy_name, policy in report["policies"].items():
        for symbol, asset in policy["assets"].items():
            metric = asset["splits"]["eval"]["base"]
            lines.append(
                f"| {policy_name} | {symbol} | {metric['absolute_return_pct']:.2f}% | "
                f"{metric['cagr_pct']:.2f}% | {metric['strict_mdd_pct']:.2f}% | "
                f"{metric['cagr_to_strict_mdd']:.2f} | {metric['trades']} | "
                f"{metric['positive_calendar_months']}/{metric['calendar_months']} | "
                f"{metric['weekly_cluster_signflip']['p_value_one_sided']:.4f} | "
                f"{'PASS' if asset['eval_pass'] else 'REJECT'} |"
            )
    lines += [
        "",
        "## Train / test / eval",
        "",
        "| Policy | Asset | Split | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for policy_name, policy in report["policies"].items():
        for symbol, asset in policy["assets"].items():
            for split_name in ("train", "test", "eval"):
                metric = asset["splits"][split_name]["base"]
                lines.append(
                    f"| {policy_name} | {symbol} | {split_name} | "
                    f"{metric['absolute_return_pct']:.2f}% | {metric['cagr_pct']:.2f}% | "
                    f"{metric['strict_mdd_pct']:.2f}% | {metric['cagr_to_strict_mdd']:.2f} | "
                    f"{metric['trades']} |"
                )
    lines += [
        "",
        "## Decision",
        "",
        report["decision"]["summary"],
        "",
        "## Execution and limitations",
        "",
        "- Signal: completed 5-minute bar; entry: next available regular-session 5-minute open.",
        "- Exit: fixed count of tradable 5-minute bars; missing KRX halt/no-trade rows are never synthesized.",
        "- strict MDD includes entry cost, every held high/low in favorable-then-adverse order, and exit cost.",
        "- CAGR uses the full wall-clock split, including idle periods. Absolute return is always shown.",
        "- Prefix invariance is recomputed at frozen split boundaries and backed by causal-builder regression tests; it is not claimed as an exhaustive proof over every row.",
        "- Investing.com TVC is an unofficial research source. Production replication requires an entitled feed and renewed parity checks.",
        "- Official production candidates: IBKR historical bars and KIS Open API.",
        "",
        "Official references:",
        "- https://www.interactivebrokers.com/campus/ibkr-api-page/twsapi-doc/",
        "- https://www.interactivebrokers.com/campus/ibkr-api-page/market-data-subscriptions/",
        "- https://www.interactivebrokers.com/en/trading/krx-exchange.php",
        "- https://apiportal.koreainvestment.com/apiservice",
    ]
    return "\n".join(lines) + "\n"


def run_evaluation(
    *,
    output: str = OUTPUT,
    docs_output: str = DOCS_OUTPUT,
) -> dict[str, Any]:
    frozen = prereg.manifest()
    frames, source_report = run_source_audit()
    _verify_source_report(source_report)
    source_path = Path(SOURCE_AUDIT_OUTPUT)
    source_file_hash = sha256_bytes(source_path.read_bytes())

    policy_results: dict[str, Any] = {
        policy.name: {"spec": policy.__dict__, "assets": {}} for policy in POLICIES
    }
    prefix_checks: dict[str, Any] = {}
    for symbol, frame in frames.items():
        arrays = build_policy_arrays(frame)
        prefix = prefix_invariance_check(frame, arrays)
        if not prefix["pass"]:
            raise RuntimeError(f"prefix invariance failed for {symbol}")
        prefix_checks[symbol] = prefix
        for policy in POLICIES:
            policy_results[policy.name]["assets"][symbol] = evaluate_asset_policy(
                frame,
                arrays[policy.name],
                policy,
            )

    for policy in POLICIES:
        assets = policy_results[policy.name]["assets"]
        policy_results[policy.name]["all_assets_pass"] = bool(
            set(assets) == set(frozen["instruments"])
            and all(asset["eval_pass"] for asset in assets.values())
        )
    transferred = [name for name, row in policy_results.items() if row["all_assets_pass"]]
    decision = {
        "transferred_policies": transferred,
        "pass": bool(transferred),
        "summary": (
            "At least one preregistered policy passed every five-minute gate on all three assets."
            if transferred
            else "No preregistered policy passed every five-minute gate on QQQ, KODEX 200, and GLD; cross-asset transfer is rejected."
        ),
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "research_id": "CAT-XA-5M-EVAL-1",
        "preregistration_manifest_hash": frozen["manifest_hash"],
        "source_audit_result_hash": source_report["result_hash"],
        "source_audit_file_sha256": source_file_hash,
        "execution": {
            "entry": "next available regular-session 5m open",
            "base_cost_bps_per_side": BASE_COST_BPS,
            "stress_cost_bps_per_side": STRESS_COST_BPS,
            "leverage": 1.0,
            "cagr_denominator": "full wall-clock split",
            "strict_mdd": "entry fee + every held high/low favorable-then-adverse + exit fee",
            "threshold_fit_population": (
                "all positive finite train strength rows before the decision clock, matching the source scanners"
            ),
        },
        "code_sha256": code_provenance(),
        "prefix_invariance": prefix_checks,
        "policies": policy_results,
        "decision": decision,
    }
    report["result_hash"] = self_hash(report)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(docs_output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    Path(docs_output).write_text(render_docs(report))
    return report


def main() -> None:
    report = run_evaluation()
    summary = {
        "output": OUTPUT,
        "result_hash": report["result_hash"],
        "decision": report["decision"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
