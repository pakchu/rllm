"""Rebuild leak-safe strict-majority sleeves from pre-evaluation Top-10 families.

The persisted state-model scans contain later-window diagnostics, but their
``selected`` arrays are ordered only by the pre-evaluation protocol documented
in each scan.  This module deliberately reads only the first ten parameter
rows, reconstructs every causal signal, verifies its frozen signal hash, and
then emits a predeclared six-of-ten majority signal.  It never consults the
future pass/fail or promotion fields.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.portfolio_opt_new_alpha_pool import _alpha_active
from training.search_gaussian_hmm_regime_alpha import SPLITS, hourly_features
import training.search_bocpd_state_gated_alpha as bocpd
import training.search_kalman_state_gated_alpha as kalman
import training.search_semimarkov_duration_alpha as semimarkov


TOP_N = 10
STRICT_MAJORITY_COUNT = 6
SCAN_PATHS = {
    "kalman": Path("results/kalman_state_gated_alpha_scan_2026-07-13.json"),
    "bocpd": Path("results/bocpd_state_gated_alpha_scan_2026-07-13.json"),
    "semimarkov": Path("results/semimarkov_duration_alpha_scan_2026-07-13.json"),
}
SLEEVE_NAMES = {
    "kalman": "kalman_top10_strict_majority_long",
    "bocpd": "bocpd_top10_strict_majority_long",
    "semimarkov": "semimarkov_top10_strict_majority_long",
}
PRE_EVALUATION_RANK_KEYS = {
    "kalman": kalman._rank_key,
    "bocpd": bocpd._rank_key,
    "semimarkov": semimarkov._rank_key,
}


def signal_hash(signal: np.ndarray) -> str:
    return hashlib.sha256(np.packbits(np.asarray(signal, dtype=bool)).tobytes()).hexdigest()[:16]


def strict_majority_mask(member_masks: list[np.ndarray]) -> np.ndarray:
    """Return a fixed strict majority (six of ten) without tuning a threshold."""
    if len(member_masks) != TOP_N:
        raise ValueError(f"expected exactly {TOP_N} member masks")
    normalized = [np.asarray(mask, dtype=bool) for mask in member_masks]
    lengths = {len(mask) for mask in normalized}
    if len(lengths) != 1:
        raise ValueError("member masks must share one market grid")
    votes = np.sum(np.stack(normalized, axis=0), axis=0)
    return votes >= STRICT_MAJORITY_COUNT


def load_pre_evaluation_top10(
    path: str | Path,
    family: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load only the frozen pre-evaluation ordering from a state-model scan."""
    source = Path(path)
    payload = json.loads(source.read_text())
    rows = payload.get("selected", [])
    if len(rows) < TOP_N:
        raise RuntimeError(f"{source} contains fewer than {TOP_N} selected rows")
    protocol = str(payload.get("protocol", ""))
    if "report-only" not in protocol and "report-only diagnostics" not in protocol:
        raise RuntimeError(f"{source} does not declare later windows report-only")
    try:
        rank_key = PRE_EVALUATION_RANK_KEYS[family]
    except KeyError as exc:
        raise ValueError(f"unknown state-model family: {family}") from exc
    pre_evaluation_keys = [rank_key(row) for row in rows]
    if any(
        pre_evaluation_keys[index] < pre_evaluation_keys[index + 1]
        for index in range(len(pre_evaluation_keys) - 1)
    ):
        raise RuntimeError(f"{source} is not ordered by its pre-evaluation rank key")
    selected = rows[:TOP_N]
    if any(not row.get("signal_hash") for row in selected):
        raise RuntimeError(f"{source} has an unfrozen Top-10 signal")
    return selected, {
        "path": str(source),
        "protocol": protocol,
        "rows_read": TOP_N,
        "fields_used": "causal parameters, train-frozen thresholds, allowed states/keys, signal_hash",
        "future_fields_used": False,
        "pre_evaluation_order_verified": True,
    }


def _verify_member(
    *,
    family: str,
    rank: int,
    row: dict[str, Any],
    active: np.ndarray,
) -> dict[str, Any]:
    actual = signal_hash(active)
    expected = str(row["signal_hash"])
    if actual != expected:
        raise RuntimeError(
            f"{family} pre-evaluation rank {rank} signal hash drifted: {actual} != {expected}"
        )
    return {
        "pre_evaluation_rank": rank,
        "signal_hash": actual,
        "active_bars": int(np.asarray(active, dtype=bool).sum()),
    }


def _kalman_masks(
    dates: pd.Series,
    setup: np.ndarray,
    hourly_market: pd.DataFrame,
    rows: list[dict[str, Any]],
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    train_hour = np.asarray(
        (hourly_market.index >= SPLITS["train"][0])
        & (hourly_market.index < SPLITS["train"][1]),
        dtype=bool,
    )
    log_price = np.log(hourly_market["close"].to_numpy(float))
    train_var = float(np.nanvar(np.diff(log_price)[train_hour[1:]]))
    cache: dict[tuple[float, ...], np.ndarray] = {}
    masks: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for rank, row in enumerate(rows, start=1):
        low, high = map(float, row["state_quantiles"])
        key = (
            float(row["q_level"]),
            float(row["q_slope"]),
            float(row["r_obs"]),
            low,
            high,
        )
        if key not in cache:
            hourly_state, _ = kalman.kalman_hourly_state(
                hourly_market,
                train_hour,
                q_level=key[0],
                q_slope=key[1],
                r_obs=key[2],
                low_quantile=key[3],
                high_quantile=key[4],
                train_var=train_var,
            )
            cache[key] = kalman.map_hourly_state(dates, hourly_state)
        active = setup & np.isin(cache[key], np.asarray(row["allowed_states"], dtype=int))
        masks.append(active)
        metadata.append(_verify_member(family="kalman", rank=rank, row=row, active=active))
    return masks, metadata


def _bocpd_masks(
    dates: pd.Series,
    setup: np.ndarray,
    hourly_feature: pd.DataFrame,
    rows: list[dict[str, Any]],
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    train_hour = np.asarray(
        (hourly_feature.index >= SPLITS["train"][0])
        & (hourly_feature.index < SPLITS["train"][1]),
        dtype=bool,
    )
    specifications: dict[str, tuple[tuple[str, ...], int | None]] = {
        "return": (("ret1",), None),
        "return_flow": (("ret1", "flow24"), 1),
        "trend_volterm": (("trend24", "volterm"), 1),
    }
    cache: dict[tuple[str, int], pd.DataFrame] = {}
    masks: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for rank, row in enumerate(rows, start=1):
        model_name = str(row["model_name"])
        hazard = int(row["model"]["hazard_lambda_hours"])
        key = (model_name, hazard)
        if key not in cache:
            columns, secondary_index = specifications[model_name]
            hourly_output, _ = bocpd._model_output(
                hourly_feature,
                train_hour,
                columns=columns,
                secondary_index=secondary_index,
                hazard_lambda=hazard,
            )
            cache[key] = bocpd._map_output(dates, hourly_output)
        state = bocpd._state_from_mapped(cache[key], row["state_thresholds"])
        active = setup & np.isin(state, np.asarray(row["allowed_states"], dtype=int))
        masks.append(active)
        metadata.append(_verify_member(family="bocpd", rank=rank, row=row, active=active))
    return masks, metadata


def _semimarkov_masks(
    dates: pd.Series,
    setup: np.ndarray,
    hourly_feature: pd.DataFrame,
    rows: list[dict[str, Any]],
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    fit_hour = np.asarray(
        (hourly_feature.index >= semimarkov.WINDOWS["fit2020_2022"][0])
        & (hourly_feature.index < semimarkov.WINDOWS["fit2020_2022"][1]),
        dtype=bool,
    )
    first = rows[0]
    low, high = map(float, first["trend_quantiles"])
    cutpoints = tuple(map(int, first["duration_cutpoints_hours"]))
    if any(
        tuple(map(float, row["trend_quantiles"])) != (low, high)
        or tuple(map(int, row["duration_cutpoints_hours"])) != cutpoints
        for row in rows
    ):
        raise RuntimeError("semi-Markov Top-10 no longer shares one frozen state surface")
    state, _ = semimarkov.observable_state(hourly_feature, fit_hour, low, high)
    hourly_key, _ = semimarkov.duration_key(
        state, cutpoints, timestamps=hourly_feature.index
    )
    mapped_key = semimarkov.map_hourly_key(dates, hourly_feature.index, hourly_key)
    masks: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for rank, row in enumerate(rows, start=1):
        active = setup & np.isin(mapped_key, np.asarray(row["allowed_keys"], dtype=int))
        masks.append(active)
        metadata.append(
            _verify_member(family="semimarkov", rank=rank, row=row, active=active)
        )
    return masks, metadata


def build_state_model_top10_ensembles(
    market: pd.DataFrame,
    features: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Return three causal strict-majority signals and their hash audit."""
    dates = pd.to_datetime(market["date"])
    setup = _alpha_active(features, "long_minimal_funding_premium")
    hourly_market, hourly_feature = hourly_features(market)
    loaded = {
        family: load_pre_evaluation_top10(path, family)
        for family, path in SCAN_PATHS.items()
    }
    family_builders = {
        "kalman": lambda rows: _kalman_masks(dates, setup, hourly_market, rows),
        "bocpd": lambda rows: _bocpd_masks(dates, setup, hourly_feature, rows),
        "semimarkov": lambda rows: _semimarkov_masks(
            dates, setup, hourly_feature, rows
        ),
    }
    signals: dict[str, np.ndarray] = {}
    audit: dict[str, Any] = {
        "aggregation": f"strict majority >= {STRICT_MAJORITY_COUNT} of {TOP_N}",
        "threshold_tuned": False,
        "future_fields_used": False,
        "families": {},
    }
    for family, builder in family_builders.items():
        rows, source_meta = loaded[family]
        member_masks, member_meta = builder(rows)
        ensemble = strict_majority_mask(member_masks)
        sleeve = SLEEVE_NAMES[family]
        signals[sleeve] = ensemble
        audit["families"][family] = {
            "sleeve": sleeve,
            "source": source_meta,
            "members": member_meta,
            "ensemble_signal_hash": signal_hash(ensemble),
            "ensemble_active_bars": int(ensemble.sum()),
        }
    return signals, audit
