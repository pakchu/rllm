"""Pre-2025 evaluator for the preregistered RITT-1 alpha.

RITT-1 learns a small empirical-Bayes table over transitions between nine
causal residual-inventory states.  Selection sources are physically truncated
before 2025; 2025/2026 and Gross9 marginal outcomes remain unopened unless a
standalone 2023/2024 cell passes the frozen contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.search_annual_oi_spot_participation_path_alpha import (
    executable_utilities,
    fold_fit_mask,
)
from training.search_positioning_hgb_path_alpha import _read_before
from training.search_spot_perp_absorption_alpha import (
    _prior_z,
    _rolling_residual,
)
from training.strict_bar_backtest import _trade_stats


PREREGISTRATION = Path(
    "results/residual_inventory_transfer_transition_preregistration_2026-07-28.json"
)
OUTPUT = Path(
    "results/residual_inventory_transfer_transition_pre2025_selection_2026-07-28.json"
)
DOCS = Path(
    "docs/residual-inventory-transfer-transition-pre2025-selection-2026-07-28.md"
)
SELECTION_CUTOFF = "2025-01-01"
FIT_START = pd.Timestamp("2020-10-15")
FOLDS = {
    "2023": ("2023-01-01", "2024-01-01"),
    "2024": ("2024-01-01", "2025-01-01"),
}
WINDOWS = {
    "2023": FOLDS["2023"],
    "2023_h1": ("2023-01-01", "2023-07-01"),
    "2023_h2": ("2023-07-01", "2024-01-01"),
    "2024": FOLDS["2024"],
    "2024_h1": ("2024-01-01", "2024-07-01"),
    "2024_h2": ("2024-07-01", "2025-01-01"),
}
FEATURE_COLUMNS = (
    "basis_long_residual",
    "basis_residual_acceleration",
    "inventory_residual",
    "carry_pressure",
)


@dataclass(frozen=True)
class PathSpec:
    name: str
    hold_bars: int
    risk_lambda: float


@dataclass(frozen=True)
class Config:
    market_csv: str = (
        "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_oi.csv.gz"
    )
    spot_csv: str = "data/cache_spot_premium_5m_2020-01-01_2026-06-01.csv.gz"
    funding_csv: str = (
        "data/binance_um_aux_btc_2020_2026/"
        "BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz"
    )
    preregistration: str = str(PREREGISTRATION)
    output: str = str(OUTPUT)
    docs_output: str = str(DOCS)
    leverage: float = 0.5
    cost_rate: float = 0.0006
    funding_tolerance: str = "12h"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _array_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _frame_hash(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update("\n".join(map(str, frame.columns)).encode())
    digest.update(
        pd.util.hash_pandas_object(frame, index=False)
        .to_numpy(dtype="<u8")
        .tobytes()
    )
    return digest.hexdigest()


def load_preregistration(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("name") != "residual_inventory_transfer_transition":
        raise RuntimeError("unexpected preregistration name")
    if payload.get("status") != "preregistered_before_outcome_scan":
        raise RuntimeError("preregistration is not outcome-blind")
    grid = payload["frozen_grid"]
    cells = len(grid["hold_bars"]) * len(grid["lcb_z"])
    if cells != grid.get("model_policy_cells") or cells != 4:
        raise RuntimeError("frozen four-cell grid drifted")
    future = payload["selection_contract"]["future_veto"]
    if future.get("future_can_rerank") is not False:
        raise RuntimeError("future reranking is not disabled")
    if future.get("future_can_repair") is not False:
        raise RuntimeError("future repair is not disabled")
    return payload


def validate_input_provenance(
    cfg: Config, preregistration: dict[str, Any]
) -> dict[str, Any]:
    expected = preregistration["input_provenance"]
    configured = {
        "market_with_oi": cfg.market_csv,
        "spot_and_completed_premium": cfg.spot_csv,
        "funding": cfg.funding_csv,
    }
    records: dict[str, Any] = {}
    for name, path in configured.items():
        actual = _sha256(path)
        wanted = str(expected[name]["sha256"])
        if actual != wanted:
            raise RuntimeError(
                f"input provenance mismatch for {name}: {actual} != {wanted}"
            )
        records[name] = {
            "path": str(path),
            "sha256": actual,
            "validated_against_preregistration": True,
        }
    return records


def _parse_utc_naive(values: pd.Series) -> pd.Series:
    return pd.to_datetime(
        values, utc=True, errors="raise", format="mixed"
    ).dt.tz_convert(None)


def load_sources(
    cfg: Config,
    *,
    cutoff: str = SELECTION_CUTOFF,
    return_audit: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    """Physically truncate and causally align market, spot, premium, and funding."""
    market = _read_before(cfg.market_csv, "date", cutoff)
    spot = _read_before(cfg.spot_csv, "date", cutoff)
    funding = _read_before(cfg.funding_csv, "date", cutoff)
    prefix_audit = {
        "cutoff": cutoff,
        "raw_prefix_hashes": {
            "market": _frame_hash(market),
            "spot": _frame_hash(spot),
            "funding": _frame_hash(funding),
        },
    }
    market["date"] = _parse_utc_naive(market["date"])
    spot["date"] = _parse_utc_naive(spot["date"])
    funding["date"] = _parse_utc_naive(funding["date"])
    market = (
        market.sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )
    spot = (
        spot.sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )
    funding = (
        funding.sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )
    spot_columns = [
        "date",
        "spot_close",
        "spot_rows",
        "premium_index_1m_close",
        "premium_rows",
    ]
    merged = market.merge(
        spot[spot_columns],
        on="date",
        how="left",
        validate="one_to_one",
    )
    merged = pd.merge_asof(
        merged.sort_values("date"),
        funding[["date", "funding_rate"]].rename(
            columns={"date": "funding_source_time"}
        ),
        left_on="date",
        right_on="funding_source_time",
        direction="backward",
        tolerance=pd.Timedelta(cfg.funding_tolerance),
    ).reset_index(drop=True)
    dates = pd.to_datetime(merged["date"])
    if len(dates) and dates.max() >= pd.Timestamp(cutoff):
        raise RuntimeError("selection source was not physically truncated")
    intervals = dates.diff().dropna()
    if len(intervals) and not intervals.eq(pd.Timedelta("5min")).all():
        raise RuntimeError("market source is not a complete five-minute grid")
    known = merged["funding_source_time"].notna()
    if known.any() and (
        pd.to_datetime(merged.loc[known, "funding_source_time"])
        > pd.to_datetime(merged.loc[known, "date"])
    ).any():
        raise RuntimeError("funding join references a future publication")
    prefix_audit["source_max_timestamp_before_cutoff"] = {
        "market": str(pd.to_datetime(market["date"]).max()),
        "spot": str(pd.to_datetime(spot["date"]).max()),
        "funding": str(pd.to_datetime(funding["date"]).max()),
    }
    if return_audit:
        return merged, prefix_audit
    return merged


def build_features(market: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Build the four frozen prior-only residual/ownership inputs."""
    perp_close = pd.to_numeric(market["close"], errors="coerce")
    spot_close = pd.to_numeric(market["spot_close"], errors="coerce")
    premium = pd.to_numeric(
        market["premium_index_1m_close"], errors="coerce"
    )
    funding = pd.to_numeric(market["funding_rate"], errors="coerce")
    log_perp = np.log(perp_close.where(perp_close > 0.0))
    log_spot = np.log(spot_close.where(spot_close > 0.0))
    direct_basis = log_perp - log_spot
    residual_z: dict[int, pd.Series] = {}
    for window in (2016, 8640):
        residual, _ = _rolling_residual(direct_basis, premium, window)
        residual_z[window] = _prior_z(residual, window)

    delayed_log_oi = np.log(
        pd.to_numeric(market["open_interest"], errors="coerce")
        .shift(1)
        .where(lambda values: values > 0.0)
    )
    oi_change = delayed_log_oi - delayed_log_oi.shift(288)
    price_return = log_perp - log_perp.shift(288)
    inventory_residual, _ = _rolling_residual(
        oi_change, price_return.abs(), 2016
    )
    inventory_residual, _ = _rolling_residual(
        inventory_residual, price_return, 2016
    )
    inventory_residual, _ = _rolling_residual(
        inventory_residual, premium, 2016
    )
    inventory_residual, _ = _rolling_residual(
        inventory_residual, funding, 2016
    )
    inventory_residual_z = _prior_z(inventory_residual, 2016)
    carry_pressure = 0.5 * _prior_z(premium, 2016) + 0.5 * _prior_z(
        funding, 2016
    )
    features = pd.DataFrame(
        {
            "basis_long_residual": residual_z[8640],
            "basis_residual_acceleration": (
                residual_z[2016] - residual_z[8640]
            ),
            "inventory_residual": inventory_residual_z,
            "carry_pressure": carry_pressure,
        },
        index=market.index,
    ).replace([np.inf, -np.inf], np.nan)
    features = features.loc[:, FEATURE_COLUMNS].astype(np.float32)
    source_valid = (
        pd.to_numeric(market["spot_rows"], errors="coerce").eq(5)
        & pd.to_numeric(market["premium_rows"], errors="coerce").eq(5)
        & spot_close.gt(0.0)
        & premium.notna()
        & funding.notna()
        & pd.to_numeric(market["open_interest"], errors="coerce")
        .shift(1)
        .gt(0.0)
    ).to_numpy(bool)
    source_valid &= np.isfinite(features.to_numpy(float)).all(axis=1)
    return features, source_valid


def decision_mask(market: pd.DataFrame, source_valid: np.ndarray) -> np.ndarray:
    dates = pd.to_datetime(market["date"])
    return (
        (dates.dt.minute.to_numpy(int) == 0)
        & np.asarray(source_valid, dtype=bool)
    )


def fit_thresholds(
    features: pd.DataFrame, fit_positions: np.ndarray
) -> dict[str, float]:
    positions = np.asarray(fit_positions, dtype=np.int64)
    if len(positions) < 2_000:
        raise RuntimeError("insufficient purged fit decisions for state bins")
    basis = features["basis_long_residual"].to_numpy(float)[positions]
    inventory = features["inventory_residual"].to_numpy(float)[positions]
    finite = np.isfinite(basis) & np.isfinite(inventory)
    if int(finite.sum()) < 2_000:
        raise RuntimeError("insufficient finite fit decisions for state bins")
    return {
        "basis_q20": float(np.quantile(basis[finite], 0.20)),
        "basis_q80": float(np.quantile(basis[finite], 0.80)),
        "inventory_q80": float(np.quantile(inventory[finite], 0.80)),
    }


def encode_states(
    features: pd.DataFrame, thresholds: dict[str, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    basis = features["basis_long_residual"].to_numpy(float)
    acceleration = features["basis_residual_acceleration"].to_numpy(float)
    inventory = features["inventory_residual"].to_numpy(float)
    carry = features["carry_pressure"].to_numpy(float)
    finite = (
        np.isfinite(basis)
        & np.isfinite(acceleration)
        & np.isfinite(inventory)
        & np.isfinite(carry)
    )
    basis_state = np.zeros(len(features), dtype=np.int8)
    basis_state[
        finite
        & (basis <= float(thresholds["basis_q20"]))
        & (acceleration < 0.0)
    ] = -1
    basis_state[
        finite
        & (basis >= float(thresholds["basis_q80"]))
        & (acceleration > 0.0)
    ] = 1
    owner_state = np.zeros(len(features), dtype=np.int8)
    excess = finite & (inventory >= float(thresholds["inventory_q80"]))
    owner_state[excess & (carry < 0.0)] = -1
    owner_state[excess & (carry > 0.0)] = 1
    state = (basis_state.astype(np.int16) + 1) * 3 + (
        owner_state.astype(np.int16) + 1
    )
    state = state.astype(np.int8)
    state[~finite] = -1
    return state, basis_state, owner_state


def transition_keys(states: np.ndarray, lag_bars: int = 144) -> np.ndarray:
    states = np.asarray(states, dtype=np.int16)
    previous = np.full(len(states), -1, dtype=np.int16)
    previous[lag_bars:] = states[:-lag_bars]
    valid = (states >= 0) & (previous >= 0)
    keys = np.full(len(states), -1, dtype=np.int16)
    keys[valid] = previous[valid] * 9 + states[valid]
    return keys


def _fit_positions_and_utilities(
    market: pd.DataFrame,
    valid_decisions: np.ndarray,
    spec: PathSpec,
    *,
    fit_end: str,
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray]:
    dates = pd.to_datetime(market["date"])
    positions = np.flatnonzero(valid_decisions)
    cutoff_position = int(
        np.searchsorted(
            dates.to_numpy(dtype="datetime64[ns]"),
            np.datetime64(pd.Timestamp(fit_end)),
            side="left",
        )
    )
    utilities, exits = executable_utilities(
        market,
        positions,
        spec=spec,
        leverage=cfg.leverage,
        cost_rate=cfg.cost_rate,
        available_before_position=cutoff_position,
    )
    fit = fold_fit_mask(
        dates,
        positions,
        exits,
        utilities,
        fit_end=fit_end,
    )
    fit_positions = positions[fit]
    fit_utilities = utilities[fit].astype(np.float64)
    if len(fit_positions) < 2_000:
        raise RuntimeError(
            f"insufficient fit decisions for {spec.name}/{fit_end}: "
            f"{len(fit_positions)}"
        )
    lower = np.quantile(fit_utilities, 0.005, axis=0)
    upper = np.quantile(fit_utilities, 0.995, axis=0)
    return fit_positions, np.clip(fit_utilities, lower, upper)


def fit_transition_table(
    transition: np.ndarray,
    fit_positions: np.ndarray,
    fit_utilities: np.ndarray,
    *,
    prior_strength: int,
) -> dict[str, Any]:
    positions = np.asarray(fit_positions, dtype=np.int64)
    utilities = np.asarray(fit_utilities, dtype=np.float64)
    keys = np.asarray(transition, dtype=np.int16)[positions]
    valid = (keys >= 0) & np.isfinite(utilities).all(axis=1)
    keys = keys[valid].astype(np.int64)
    utilities = utilities[valid]
    if len(keys) < 2_000:
        raise RuntimeError("insufficient valid transition targets")
    global_mean = np.mean(utilities, axis=0)
    pooled_variance = np.var(utilities, axis=0, ddof=1)
    counts = np.bincount(keys, minlength=81).astype(np.int64)
    sums = np.zeros((81, 2), dtype=np.float64)
    np.add.at(sums[:, 0], keys, utilities[:, 0])
    np.add.at(sums[:, 1], keys, utilities[:, 1])
    posterior_mean = (
        sums + float(prior_strength) * global_mean[None, :]
    ) / (counts[:, None] + float(prior_strength))
    posterior_mean_variance = pooled_variance[None, :] / (
        counts[:, None] + float(prior_strength)
    )
    return {
        "global_mean": global_mean,
        "pooled_variance": pooled_variance,
        "counts": counts,
        "posterior_mean": posterior_mean,
        "posterior_mean_variance": posterior_mean_variance,
        "fit_transition_rows": int(len(keys)),
    }


def transition_table_diagnostics(
    table: dict[str, Any], *, min_support: int
) -> dict[str, Any]:
    counts = np.asarray(table["counts"], dtype=np.int64)
    means = np.asarray(table["posterior_mean"], dtype=float)
    variance = np.asarray(table["posterior_mean_variance"], dtype=float)
    keys = np.arange(81, dtype=np.int64)
    supported_changed = (
        (counts >= int(min_support)) & ((keys // 9) != (keys % 9))
    )
    if not supported_changed.any():
        return {
            "global_mean": np.asarray(
                table["global_mean"], dtype=float
            ).tolist(),
            "supported_changed_transitions": 0,
            "max_supported_posterior_mean": [None, None],
            "max_supported_lcb_0_5": None,
            "positive_supported_side_cells": 0,
            "max_supported_side_advantage": None,
        }
    supported_means = means[supported_changed]
    supported_variance = variance[supported_changed]
    changed_counts = counts[(keys // 9) != (keys % 9)]
    preferred = np.argmax(supported_means, axis=1)
    return {
        "global_mean": np.asarray(
            table["global_mean"], dtype=float
        ).tolist(),
        "supported_changed_transitions": int(np.sum(supported_changed)),
        "max_supported_posterior_mean": np.max(
            supported_means, axis=0
        ).tolist(),
        "max_supported_lcb_0_5": float(
            np.max(
                supported_means
                - 0.5 * np.sqrt(np.maximum(0.0, supported_variance))
            )
        ),
        "positive_supported_side_cells": int(
            np.sum(supported_means > 0.0)
        ),
        "max_supported_side_advantage": float(
            np.max(np.abs(supported_means[:, 0] - supported_means[:, 1]))
        ),
        "changed_transition_support_quantiles": {
            str(int(quantile * 100)): float(
                np.quantile(changed_counts, quantile)
            )
            for quantile in (0.0, 0.25, 0.5, 0.75, 1.0)
        },
        "supported_preferred_side_counts": {
            "long": int(np.sum(preferred == 0)),
            "short": int(np.sum(preferred == 1)),
        },
    }


def activation_masks(
    length: int,
    positions: np.ndarray,
    transition: np.ndarray,
    table: dict[str, Any],
    *,
    lcb_z: float,
    min_support: int,
    min_side_advantage: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    positions = np.asarray(positions, dtype=np.int64)
    keys = np.asarray(transition, dtype=np.int16)[positions]
    valid_key = keys >= 0
    safe_keys = np.where(valid_key, keys, 0).astype(np.int64)
    counts = np.asarray(table["counts"], dtype=np.int64)[safe_keys]
    means = np.asarray(table["posterior_mean"], dtype=float)[safe_keys]
    variance = np.asarray(
        table["posterior_mean_variance"], dtype=float
    )[safe_keys]
    lcb = means - float(lcb_z) * np.sqrt(np.maximum(0.0, variance))
    previous = safe_keys // 9
    current = safe_keys % 9
    support = valid_key & (counts >= int(min_support)) & (previous != current)
    best_side = np.argmax(lcb, axis=1)
    confidence = np.max(lcb, axis=1)
    advantage = np.abs(means[:, 0] - means[:, 1])
    eligible = (
        support
        & np.isfinite(confidence)
        & (confidence > 0.0)
        & (advantage >= float(min_side_advantage))
    )
    local_long = eligible & (best_side == 0)
    local_short = eligible & (best_side == 1)
    long_active = np.zeros(length, dtype=bool)
    short_active = np.zeros(length, dtype=bool)
    long_active[positions] = local_long
    short_active[positions] = local_short
    activation = np.column_stack(
        [
            positions[eligible],
            np.where(local_long[eligible], 1, -1),
            keys[eligible],
        ]
    ).astype("<i8", copy=False)
    return long_active, short_active, {
        "eligible": int(np.sum(eligible)),
        "long_signals": int(np.sum(local_long)),
        "short_signals": int(np.sum(local_short)),
        "supported_transition_count": int(
            np.sum(np.asarray(table["counts"]) >= int(min_support))
        ),
        "activation_hash": _array_hash(activation),
    }


def fit_fold(
    market: pd.DataFrame,
    features: pd.DataFrame,
    valid_decisions: np.ndarray,
    spec: PathSpec,
    *,
    fit_end: str,
    predict_start: str,
    predict_end: str,
    preregistration: dict[str, Any],
    cfg: Config,
) -> dict[str, Any]:
    dates = pd.to_datetime(market["date"])
    fit_positions, fit_utilities = _fit_positions_and_utilities(
        market,
        valid_decisions,
        spec,
        fit_end=fit_end,
        cfg=cfg,
    )
    thresholds = fit_thresholds(features, fit_positions)
    states, basis_state, owner_state = encode_states(features, thresholds)
    transition = transition_keys(
        states,
        lag_bars=int(
            preregistration["source_and_feature_contract"][
                "transition_lag_bars"
            ]
        ),
    )
    table = fit_transition_table(
        transition,
        fit_positions,
        fit_utilities,
        prior_strength=int(
            preregistration["frozen_grid"]["prior_strength"]
        ),
    )
    predict_mask = (
        (dates >= pd.Timestamp(predict_start))
        & (dates < pd.Timestamp(predict_end))
    ).to_numpy(bool)
    predict_positions = np.flatnonzero(valid_decisions & predict_mask)
    table_hash_payload = np.column_stack(
        [
            np.asarray(table["counts"], dtype="<i8"),
            np.asarray(table["posterior_mean"], dtype="<f8"),
            np.asarray(table["posterior_mean_variance"], dtype="<f8"),
        ]
    )
    return {
        "fit_positions": fit_positions,
        "predict_positions": predict_positions,
        "transition": transition,
        "table": table,
        "meta": {
            "fit_start": str(FIT_START),
            "fit_end_exclusive": fit_end,
            "predict_start": predict_start,
            "predict_end_exclusive": predict_end,
            "fit_rows": int(len(fit_positions)),
            "predict_rows": int(len(predict_positions)),
            "thresholds": thresholds,
            "threshold_hash": _json_hash(thresholds),
            "fit_position_hash": _array_hash(
                np.asarray(fit_positions, dtype="<i8")
            ),
            "predict_position_hash": _array_hash(
                np.asarray(predict_positions, dtype="<i8")
            ),
            "table_hash": _array_hash(table_hash_payload),
            "fit_transition_rows": int(table["fit_transition_rows"]),
            "supported_transitions_50": int(
                np.sum(np.asarray(table["counts"]) >= 50)
            ),
            "posterior_diagnostics": transition_table_diagnostics(
                table,
                min_support=int(
                    preregistration["frozen_grid"][
                        "min_transition_support"
                    ]
                ),
            ),
            "predict_basis_state_counts": {
                str(value): int(np.sum(basis_state[predict_positions] == value))
                for value in (-1, 0, 1)
            },
            "predict_owner_state_counts": {
                str(value): int(np.sum(owner_state[predict_positions] == value))
                for value in (-1, 0, 1)
            },
        },
    }


def _cell_passes(stats: dict[str, dict[str, Any]]) -> bool:
    return bool(
        stats["2023"]["return_pct"] > 0.0
        and stats["2024"]["return_pct"] > 0.0
        and stats["2023"]["trades"] >= 12
        and stats["2024"]["trades"] >= 12
        and all(
            stats[name]["trades"] >= 4
            for name in ("2023_h1", "2023_h2", "2024_h1", "2024_h2")
        )
        and stats["2023"]["strict_mdd_pct"] <= 15.0
        and stats["2024"]["strict_mdd_pct"] <= 15.0
        and min(stats["2023"]["ratio"], stats["2024"]["ratio"]) >= 1.5
    )


def _cell_key(row: dict[str, Any]) -> tuple[Any, ...]:
    stats = row["stats"]
    ratios = [stats["2023"]["ratio"], stats["2024"]["ratio"]]
    halves = [
        stats[name]["return_pct"]
        for name in ("2023_h1", "2023_h2", "2024_h1", "2024_h2")
    ]
    return (
        min(ratios),
        math.sqrt(max(0.0, ratios[0]) * max(0.0, ratios[1])),
        min(halves),
        stats["2023"]["return_pct"] + stats["2024"]["return_pct"],
        -int(row["hold_bars"]),
        -float(row["lcb_z"]),
        str(row["name"]),
    )


def simulate_upper_then_lower(
    market: pd.DataFrame,
    dates: pd.Series,
    long_active: np.ndarray,
    short_active: np.ndarray,
    *,
    window: str,
    hold_bars: int,
    stride_bars: int,
    leverage: float,
    fee_rate: float,
    windows: dict[str, tuple[str, str]] = WINDOWS,
) -> dict[str, Any]:
    """Strict completed-bar path: every held bar marks high before low."""
    start, end = windows[window]
    period = ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy(
        bool
    )
    opens = pd.to_numeric(market["open"], errors="coerce").to_numpy(float)
    highs = pd.to_numeric(market["high"], errors="coerce").to_numpy(float)
    lows = pd.to_numeric(market["low"], errors="coerce").to_numpy(float)
    candidates = np.arange(
        0,
        len(market) - int(hold_bars) - 2,
        int(stride_bars),
        dtype=np.int64,
    )
    candidates = candidates[
        period[candidates] & (long_active[candidates] | short_active[candidates])
    ]
    side_cost = float(fee_rate) * float(leverage)
    equity = 1.0
    peak = 1.0
    strict_mdd = 0.0
    next_position = 0
    trade_returns: list[float] = []
    sides: list[int] = []
    for position in candidates:
        if int(position) < next_position:
            continue
        side = (
            1
            if long_active[position] and not short_active[position]
            else (
                -1
                if short_active[position] and not long_active[position]
                else 0
            )
        )
        if side == 0:
            continue
        entry = int(position) + 1
        exit_position = entry + int(hold_bars)
        if exit_position >= len(market) or not period[exit_position]:
            continue
        entry_price = float(opens[entry])
        if not np.isfinite(entry_price) or entry_price <= 0.0:
            continue
        entry_equity = equity
        after_entry_fee = equity * (1.0 - side_cost)
        strict_mdd = max(strict_mdd, 1.0 - after_entry_fee / peak)
        for bar in range(entry, exit_position):
            for mark_price in (float(highs[bar]), float(lows[bar])):
                marked = max(
                    0.0,
                    after_entry_fee
                    * (
                        1.0
                        + float(leverage)
                        * side
                        * (mark_price / entry_price - 1.0)
                    ),
                )
                strict_mdd = max(strict_mdd, 1.0 - marked / peak)
                peak = max(peak, marked)
        exit_mark = max(
            0.0,
            after_entry_fee
            * (
                1.0
                + float(leverage)
                * side
                * (float(opens[exit_position]) / entry_price - 1.0)
            ),
        )
        equity = exit_mark * (1.0 - side_cost)
        strict_mdd = max(strict_mdd, 1.0 - equity / peak)
        peak = max(peak, equity)
        trade_returns.append(equity / entry_equity - 1.0)
        sides.append(side)
        next_position = exit_position + 1
    years = (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() / (
        365.25 * 86400.0
    )
    absolute_return = (equity - 1.0) * 100.0
    cagr = (
        (equity ** (1.0 / years) - 1.0) * 100.0
        if equity > 0.0
        else -100.0
    )
    mdd = strict_mdd * 100.0
    returns = np.asarray(trade_returns, dtype=float)
    evidence = _trade_stats(returns)
    return {
        "return_pct": float(absolute_return),
        "cagr_pct": float(cagr),
        "strict_mdd_pct": float(mdd),
        "ratio": float(cagr / mdd) if mdd > 1e-12 else 0.0,
        "trades": int(len(returns)),
        "longs": int(sum(side > 0 for side in sides)),
        "shorts": int(sum(side < 0 for side in sides)),
        "win_rate": float((returns > 0.0).mean()) if len(returns) else 0.0,
        "mean_trade_return_pct": float(evidence["mean_trade_ret_pct"]),
        "p_value_mean_return_approx": float(
            evidence["p_value_mean_ret_approx"]
        ),
        "effect_size_d": float(evidence["effect_size_d"]),
    }


def _feature_hash(features: pd.DataFrame, dates: pd.Series) -> str:
    values = features.to_numpy(dtype="<f4", copy=True)
    values = np.nan_to_num(
        values,
        nan=3.4028233e38,
        posinf=3.4028231e38,
        neginf=-3.4028231e38,
    )
    digest = hashlib.sha256()
    digest.update("\n".join(features.columns).encode())
    digest.update(
        pd.to_datetime(dates)
        .to_numpy(dtype="datetime64[ns]")
        .astype("<i8", copy=False)
        .tobytes()
    )
    digest.update(values.tobytes())
    return digest.hexdigest()


def _coverage(
    market: pd.DataFrame,
    features: pd.DataFrame,
    source_valid: np.ndarray,
) -> dict[str, Any]:
    dates = pd.to_datetime(market["date"])
    finite = np.isfinite(features.to_numpy(float)).all(axis=1)
    result: dict[str, Any] = {}
    for year in range(2020, 2025):
        mask = np.asarray(
            (dates >= pd.Timestamp(f"{year}-01-01"))
            & (dates < pd.Timestamp(f"{year + 1}-01-01")),
            dtype=bool,
        )
        result[str(year)] = {
            "rows": int(np.sum(mask)),
            "spot_complete_fraction": float(
                pd.to_numeric(market.loc[mask, "spot_rows"], errors="coerce")
                .eq(5)
                .mean()
            ),
            "premium_complete_fraction": float(
                pd.to_numeric(
                    market.loc[mask, "premium_rows"], errors="coerce"
                )
                .eq(5)
                .mean()
            ),
            "delayed_oi_available_fraction": float(
                pd.to_numeric(market["open_interest"], errors="coerce")
                .shift(1)
                .loc[mask]
                .gt(0.0)
                .mean()
            ),
            "full_feature_fraction": float(np.mean(finite[mask])),
            "source_valid_fraction": float(np.mean(source_valid[mask])),
        }
    return result


def _metric_cell(metric: dict[str, Any]) -> str:
    return (
        f"{metric['return_pct']:.2f}/{metric['cagr_pct']:.2f}/"
        f"{metric['strict_mdd_pct']:.2f}/{metric['ratio']:.2f}/"
        f"{metric['trades']} ({metric['longs']}/{metric['shorts']})"
    )


def _render(payload: dict[str, Any]) -> str:
    displayed = payload["shortlist"] or payload["diagnostic_top"]
    label = (
        "frozen shortlist"
        if payload["shortlist"]
        else "top failed cells (diagnostic only)"
    )
    lines = [
        "# RITT-1 pre-2025 annual selection",
        "",
        "Metric: `absolute return / full-calendar CAGR / strict MDD / CAGR-MDD / trades (long/short)`.",
        "",
        f"- policy cells: {payload['tested']}",
        f"- standalone passers: {payload['passed']}",
        f"- decision: **{payload['decision']}**",
        f"- table: {label}",
        "",
        "| # | policy | pass | 2023 | 2023 H1 | 2023 H2 | 2024 | 2024 H1 | 2024 H2 |",
        "|---:|---|:---:|---:|---:|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(displayed, start=1):
        cells = [
            _metric_cell(row["stats"][name])
            for name in (
                "2023",
                "2023_h1",
                "2023_h2",
                "2024",
                "2024_h1",
                "2024_h2",
            )
        ]
        lines.append(
            f"| {index} | `{row['name']}` | "
            f"{'Y' if row['passes'] else 'N'} | "
            + " | ".join(cells)
            + " |"
        )
    lines += [
        "",
        "## Rejection diagnosis",
        "",
    ]
    for hold, folds in payload["fold_meta"].items():
        for fold, meta in folds.items():
            diagnostic = meta["posterior_diagnostics"]
            max_means = diagnostic["max_supported_posterior_mean"]
            lines.append(
                f"- hold `{hold}` / {fold}: global utility "
                f"`{diagnostic['global_mean'][0]:+.6f}/"
                f"{diagnostic['global_mean'][1]:+.6f}` (long/short), "
                f"best supported posterior "
                f"`{max_means[0]:+.6f}/{max_means[1]:+.6f}`, "
                f"positive supported side cells "
                f"`{diagnostic['positive_supported_side_cells']}`."
            )
    lines += [
        "",
        "All adequately supported transition-side posteriors remained below zero after frozen costs and adverse-excursion penalty. The zero-trade result is therefore a fail-closed model decision, not missing data or an execution error.",
        "",
        "## Boundary",
        "",
        "- Market, spot/premium, and funding sources were physically truncated before 2025.",
        "- OI is delayed one complete 5m bar; all rolling regressions and z-scores use statistics ending at t-1.",
        "- 2023 and 2024 use separate expanding fits with exits purged before each cutoff.",
        "- Entry is the next 5m open; costs, idle calendar time, non-overlap, and split-contained exits are included.",
        "- Gross9 marginal and 2025/2026 outcomes remain unopened unless a standalone cell passes.",
    ]
    return "\n".join(lines) + "\n"


def _atomic_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def run_pre2025(cfg: Config) -> dict[str, Any]:
    preregistration = load_preregistration(cfg.preregistration)
    input_identity = validate_input_provenance(cfg, preregistration)
    loaded = load_sources(cfg, return_audit=True)
    if not isinstance(loaded, tuple):
        raise AssertionError("source audit was not returned")
    market, source_prefix_audit = loaded
    dates = pd.to_datetime(market["date"])
    features, source_valid = build_features(market)
    valid_decisions = decision_mask(market, source_valid)
    grid = preregistration["frozen_grid"]
    fold_bank: dict[tuple[int, str], dict[str, Any]] = {}
    fold_meta: dict[str, dict[str, Any]] = {}
    for hold_bars in grid["hold_bars"]:
        hold = int(hold_bars)
        spec = PathSpec(
            name=f"h{hold}_lam050",
            hold_bars=hold,
            risk_lambda=float(grid["risk_lambda"]),
        )
        fold_meta[str(hold)] = {}
        for fold_name, (start, end) in FOLDS.items():
            fitted = fit_fold(
                market,
                features,
                valid_decisions,
                spec,
                fit_end=start,
                predict_start=start,
                predict_end=end,
                preregistration=preregistration,
                cfg=cfg,
            )
            fold_bank[(hold, fold_name)] = fitted
            fold_meta[str(hold)][fold_name] = fitted["meta"]

    rows: list[dict[str, Any]] = []
    for hold_bars in grid["hold_bars"]:
        hold = int(hold_bars)
        for lcb_z in grid["lcb_z"]:
            stats: dict[str, dict[str, Any]] = {}
            policy_meta: dict[str, Any] = {}
            for fold_name in FOLDS:
                fitted = fold_bank[(hold, fold_name)]
                long_active, short_active, meta = activation_masks(
                    len(market),
                    fitted["predict_positions"],
                    fitted["transition"],
                    fitted["table"],
                    lcb_z=float(lcb_z),
                    min_support=int(grid["min_transition_support"]),
                    min_side_advantage=float(grid["min_side_advantage"]),
                )
                policy_meta[fold_name] = meta
                for metric_name in (
                    fold_name,
                    f"{fold_name}_h1",
                    f"{fold_name}_h2",
                ):
                    stats[metric_name] = simulate_upper_then_lower(
                        market,
                        dates,
                        long_active,
                        short_active,
                        window=metric_name,
                        hold_bars=hold,
                        stride_bars=int(grid["decision_stride_bars"]),
                        leverage=cfg.leverage,
                        fee_rate=cfg.cost_rate,
                        windows=WINDOWS,
                    )
            row = {
                "name": f"ritt_h{hold}_lcb{str(lcb_z).replace('.', '')}",
                "hold_bars": hold,
                "lcb_z": float(lcb_z),
                "policy_meta": policy_meta,
                "stats": stats,
            }
            row["passes"] = _cell_passes(stats)
            rows.append(row)
    if len(rows) != int(grid["model_policy_cells"]):
        raise RuntimeError("evaluated policy count drifted")
    passed = sorted(
        (row for row in rows if row["passes"]),
        key=_cell_key,
        reverse=True,
    )
    shortlist = passed[
        : int(
            preregistration["selection_contract"]["standalone_pre2025"][
                "shortlist_max"
            ]
        )
    ]
    diagnostic_top = sorted(rows, key=_cell_key, reverse=True)
    freeze = {
        "preregistration_sha256": _sha256(cfg.preregistration),
        "source_prefix": {
            "cutoff": SELECTION_CUTOFF,
            "rows": int(len(market)),
            "first": str(dates.iloc[0]),
            "last": str(dates.iloc[-1]),
            "raw_prefix_hashes": source_prefix_audit["raw_prefix_hashes"],
            "source_max_timestamp_before_cutoff": source_prefix_audit[
                "source_max_timestamp_before_cutoff"
            ],
        },
        "feature_hash": _feature_hash(features, dates),
        "fold_meta": fold_meta,
        "shortlist": shortlist,
    }
    payload = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "phase": "pre2025_annual_selection",
        "config": asdict(cfg),
        "input_identity": input_identity,
        "preregistration": cfg.preregistration,
        "preregistration_sha256": freeze["preregistration_sha256"],
        "source_prefix": freeze["source_prefix"],
        "feature_columns": list(FEATURE_COLUMNS),
        "feature_count": len(FEATURE_COLUMNS),
        "feature_hash": freeze["feature_hash"],
        "coverage": _coverage(market, features, source_valid),
        "fold_meta": fold_meta,
        "tested": len(rows),
        "passed": len(passed),
        "shortlist": shortlist,
        "diagnostic_top": diagnostic_top,
        "decision": "open_gross9_marginal" if shortlist else "reject_pre2025",
        "gross9_marginal_opened": False,
        "future_opened": False,
        "freeze_hash": _json_hash(freeze),
        "all_pre2025_rows": rows,
    }
    _atomic_json(cfg.output, payload)
    docs = Path(cfg.docs_output)
    docs.parent.mkdir(parents=True, exist_ok=True)
    docs.write_text(_render(payload), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("pre2025",))
    parser.add_argument("--market-csv", default=Config.market_csv)
    parser.add_argument("--spot-csv", default=Config.spot_csv)
    parser.add_argument("--funding-csv", default=Config.funding_csv)
    parser.add_argument("--preregistration", default=Config.preregistration)
    parser.add_argument("--output", default=Config.output)
    parser.add_argument("--docs-output", default=Config.docs_output)
    args = parser.parse_args()
    payload = run_pre2025(
        Config(
            market_csv=args.market_csv,
            spot_csv=args.spot_csv,
            funding_csv=args.funding_csv,
            preregistration=args.preregistration,
            output=args.output,
            docs_output=args.docs_output,
        )
    )
    print(
        json.dumps(
            {
                "phase": payload["phase"],
                "tested": payload["tested"],
                "passed": payload["passed"],
                "decision": payload["decision"],
                "output": args.output,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
