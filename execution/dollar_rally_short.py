"""Frozen dollar-rally short signal and live runtime contract.

The strategy is an independent virtual SHORT sleeve whose signed quantity is
aggregated with the containing portfolio before broker execution. It evaluates
at the globally anchored hourly 5-minute slot, submits on the following bar,
and uses a scheduled 12-hour time exit without price barriers. This module has
no exchange, database, network, or order side effects.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DOLLAR_SHORT_DXY_MOMENTUM_THRESHOLD = 0.0021818982809893497
DOLLAR_SHORT_HTF_1D_RETURN_4_THRESHOLD = 0.016096783732847175
DOLLAR_SHORT_WINDOW_SIZE = 144
DOLLAR_SHORT_HOLD_BARS = 144
DOLLAR_SHORT_STRIDE_BARS = 12
DOLLAR_SHORT_PHASE_OFFSET_BARS = DOLLAR_SHORT_WINDOW_SIZE - 1
DOLLAR_SHORT_STRIDE_OFFSET_BARS = (
    DOLLAR_SHORT_PHASE_OFFSET_BARS % DOLLAR_SHORT_STRIDE_BARS
)
DOLLAR_SHORT_ENTRY_WINDOW_BARS = 1
DOLLAR_SHORT_MIN_HISTORY_BARS = 24 * 60 * 4
DOLLAR_SHORT_HOLD = pd.Timedelta(hours=12)
DOLLAR_SHORT_PHASE_ANCHOR = pd.Timestamp("2019-12-31 15:00:00", tz="UTC")
DOLLAR_SHORT_SOURCE_PATH = Path(
    "configs/shadow/legacy_dollar_rally_short_2026-09-07.json"
)
DOLLAR_SHORT_SOURCE_SHA256 = (
    "62fecd410deeca32b1ca8cea7e48c384ddeda06594a275d5c490a673d089c20d"
)

_REQUIRED_MARKET_COLUMNS = (
    "date",
    "open",
    "high",
    "low",
    "close",
    "dxy_momentum",
    "dxy_available",
)


class DollarShortContractError(ValueError):
    """Raised when live configuration drifts from the frozen strategy."""


def _require_equal(field: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise DollarShortContractError(
            f"dollar-rally short {field} must be {expected!r}, got {actual!r}"
        )


def _require_int(config: Mapping[str, Any], field: str, expected: int) -> None:
    actual = config.get(field)
    try:
        matches = int(actual) == expected
    except (TypeError, ValueError):
        matches = False
    if not matches:
        _require_equal(field, actual, expected)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dollar_short_runtime_config(
    config: Mapping[str, Any], *, configured_side: str | None = None
) -> None:
    """Validate every source-owned entry and lifecycle field.

    Portfolio weight is deliberately excluded because allocation owns it. The
    direction, gates, clock, overlap rules, source artifact, and exit contract
    are frozen. This validator authorizes new entries; recovery code must not
    depend on it because a damaged config must never block risk-reducing exits.
    """

    for field, expected in (
        ("runtime_ready", True),
        ("enabled", True),
        ("live_authorized", True),
        ("research_only", False),
        ("parent_long_required", False),
        ("same_sleeve_overlap_allowed", False),
        ("other_sleeve_overlap_allowed", True),
    ):
        actual = config.get(field)
        if actual is not expected:
            _require_equal(field, actual, expected)

    _require_equal(
        "policy_type",
        str(config.get("policy_type", "")).lower(),
        "dollar_rally_short",
    )
    _require_equal("symbol", str(config.get("symbol", "")).upper(), "BTCUSDT")
    _require_equal("side", str(config.get("side", "")).upper(), "SHORT")
    if configured_side is not None:
        _require_equal("configured sleeve side", configured_side.upper(), "SHORT")

    for field, expected in (
        ("minimum_feature_history_bars", DOLLAR_SHORT_MIN_HISTORY_BARS),
        ("hold_bars_5m", DOLLAR_SHORT_HOLD_BARS),
        ("stride_bars_5m", DOLLAR_SHORT_STRIDE_BARS),
        ("stride_offset_bars", DOLLAR_SHORT_STRIDE_OFFSET_BARS),
        ("entry_delay_bars", 1),
        ("entry_window_bars", DOLLAR_SHORT_ENTRY_WINDOW_BARS),
        ("phase_offset_bars", DOLLAR_SHORT_PHASE_OFFSET_BARS),
    ):
        _require_int(config, field, expected)

    _require_equal(
        "phase_anchor_utc",
        str(config.get("phase_anchor_utc", "")),
        DOLLAR_SHORT_PHASE_ANCHOR.isoformat(),
    )
    for field in ("take_profit", "stop_loss", "dynamic_exit", "barrier_exit"):
        actual = config.get(field)
        if actual not in (None, {}):
            _require_equal(field, actual, None)

    availability = config.get("source_availability_contract")
    if not isinstance(availability, Mapping):
        _require_equal(
            "source_availability_contract",
            availability,
            "the frozen availability contract",
        )
    for field, expected in (
        ("dxy_available_at_signal", True),
        ("freshness_wait_required", False),
        ("closed_market_missing_value_blocks_signal", True),
        ("missing_values_fail_closed", True),
    ):
        actual = availability.get(field)
        if actual is not expected:
            _require_equal(f"source_availability_contract.{field}", actual, expected)

    gates = config.get("gates")
    if not isinstance(gates, list) or len(gates) != 2:
        _require_equal("gates", gates, "the two frozen gates")
    by_feature = {
        str(gate.get("feature")): gate
        for gate in gates
        if isinstance(gate, Mapping) and gate.get("feature")
    }
    expected_gates = {
        "dxy_momentum": DOLLAR_SHORT_DXY_MOMENTUM_THRESHOLD,
        "htf_1d_return_4": DOLLAR_SHORT_HTF_1D_RETURN_4_THRESHOLD,
    }
    _require_equal("gate features", set(by_feature), set(expected_gates))
    for feature, threshold in expected_gates.items():
        gate = by_feature[feature]
        _require_equal(f"{feature} gate op", gate.get("op"), ">=")
        try:
            actual_threshold = float(gate.get("threshold"))
        except (TypeError, ValueError):
            actual_threshold = None
        _require_equal(f"{feature} threshold", actual_threshold, threshold)

    provenance = config.get("research_provenance")
    if not isinstance(provenance, Mapping):
        _require_equal(
            "research_provenance", provenance, "the pinned research provenance"
        )
    _require_equal(
        "research_provenance.source_path",
        str(provenance.get("source_path", "")),
        str(DOLLAR_SHORT_SOURCE_PATH),
    )
    _require_equal(
        "research_provenance.source_sha256",
        str(provenance.get("source_sha256", "")),
        DOLLAR_SHORT_SOURCE_SHA256,
    )
    if not DOLLAR_SHORT_SOURCE_PATH.is_file():
        raise DollarShortContractError(
            f"frozen dollar-short source is missing: {DOLLAR_SHORT_SOURCE_PATH}"
        )
    _require_equal(
        "frozen source sha256",
        _sha256(DOLLAR_SHORT_SOURCE_PATH),
        DOLLAR_SHORT_SOURCE_SHA256,
    )


def _utc_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _normalize_market(market: pd.DataFrame, *, decision_bar_date: Any) -> pd.DataFrame:
    """Validate and truncate the exact contiguous 5-minute decision history."""

    missing = [column for column in _REQUIRED_MARKET_COLUMNS if column not in market]
    if missing:
        raise ValueError(f"market missing required columns: {missing}")
    out = market.loc[:, list(_REQUIRED_MARKET_COLUMNS)].copy()
    out["date"] = pd.to_datetime(out["date"], utc=True, errors="coerce")
    if out["date"].isna().any():
        raise ValueError("market.date contains non-timestamp values")
    if out["date"].duplicated().any():
        raise ValueError("Duplicate market timestamps")
    out = out.sort_values("date").reset_index(drop=True)
    if out.empty:
        raise ValueError("Empty market")
    if (out["date"].astype("int64") % pd.Timedelta("5min").value).any():
        raise ValueError("Off-grid market timestamps")

    decision = _utc_timestamp(decision_bar_date)
    out = out.loc[out["date"] <= decision].reset_index(drop=True)
    if out.empty:
        raise ValueError("market has no rows at or before decision")
    if not out["date"].diff().dropna().eq(pd.Timedelta("5min")).all():
        raise ValueError("Market grid gap")
    for column in _REQUIRED_MARKET_COLUMNS[1:]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    prices = out[["open", "high", "low", "close"]].to_numpy(float)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Invalid market prices")
    return out


def _original_144_features(market: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the two exact frozen features used by the legacy candidate."""

    source = market.copy()
    source["date"] = pd.to_datetime(source["date"], utc=True, errors="coerce")
    out = pd.DataFrame(index=source.index)
    out["dxy_momentum"] = pd.to_numeric(source["dxy_momentum"], errors="coerce")

    if len(source) < DOLLAR_SHORT_MIN_HISTORY_BARS:
        out["htf_1d_return_4"] = 0.0
        return out.replace([np.inf, -np.inf], np.nan)

    daily = source[["date", "open", "high", "low", "close"]].set_index("date")
    higher_timeframe = pd.DataFrame(
        {
            "open": daily["open"].resample("1D", label="right", closed="right").first(),
            "high": daily["high"].resample("1D", label="right", closed="right").max(),
            "low": daily["low"].resample("1D", label="right", closed="right").min(),
            "close": daily["close"]
            .resample("1D", label="right", closed="right")
            .last(),
        }
    ).dropna()
    previous = higher_timeframe.shift(1)
    feature_rows = pd.DataFrame(
        {
            "date": higher_timeframe.index,
            "htf_1d_return_4": (
                previous["close"] / previous["close"].shift(4).replace(0.0, np.nan)
                - 1.0
            ).to_numpy(),
        }
    )
    aligned = pd.merge_asof(
        pd.DataFrame(
            {"date": source["date"], "_row": np.arange(len(source))}
        ).sort_values("date"),
        feature_rows.sort_values("date"),
        on="date",
        direction="backward",
    ).sort_values("_row")
    out["htf_1d_return_4"] = aligned["htf_1d_return_4"].to_numpy()
    return out.replace([np.inf, -np.inf], np.nan)


def is_dollar_short_signal_phase(timestamp: Any) -> bool:
    """Return whether ``timestamp`` is on the frozen global decision clock."""

    decision = _utc_timestamp(timestamp)
    delta = decision - DOLLAR_SHORT_PHASE_ANCHOR
    if delta < pd.Timedelta(0):
        return False
    bars = delta / pd.Timedelta(minutes=5)
    return (
        float(bars).is_integer()
        and int(bars) % DOLLAR_SHORT_STRIDE_BARS == DOLLAR_SHORT_STRIDE_OFFSET_BARS
    )


def score_dollar_short(market: pd.DataFrame, decision_bar_date: Any) -> dict[str, Any]:
    """Score the frozen strategy at one completed decision bar, without ordering."""

    decision = _utc_timestamp(decision_bar_date)
    normalized = _normalize_market(market, decision_bar_date=decision)
    positions = normalized.index[normalized["date"].eq(decision)]
    if len(positions) != 1:
        return {
            "name": "dollar_rally_short",
            "active": False,
            "position": 0.0,
            "decision_time": decision.isoformat(),
            "reason": "decision_bar_not_available_at_cutoff",
        }

    position = int(positions[0])
    execution = decision + pd.Timedelta(minutes=5)
    entry_expires = execution + pd.Timedelta(minutes=5 * DOLLAR_SHORT_ENTRY_WINDOW_BARS)
    exit_time = execution + DOLLAR_SHORT_HOLD
    reasons: list[str] = []
    phase_matched = is_dollar_short_signal_phase(decision)
    if not phase_matched:
        reasons.append("off_global_phase")
    if position < DOLLAR_SHORT_WINDOW_SIZE - 1 or position < 288 * 4:
        reasons.append("insufficient_warmup")

    features = _original_144_features(normalized)
    dxy_momentum = float(features.at[position, "dxy_momentum"])
    daily_return = float(features.at[position, "htf_1d_return_4"])
    if not np.isfinite(dxy_momentum) or not np.isfinite(daily_return):
        reasons.append("nonfinite_required_inputs")
    dxy_available = float(normalized.at[position, "dxy_available"])
    if not np.isfinite(dxy_available) or dxy_available <= 0.5:
        reasons.append("dxy_unavailable_at_signal")

    gates = {
        "dxy_momentum": {
            "value": dxy_momentum,
            "op": ">=",
            "threshold": DOLLAR_SHORT_DXY_MOMENTUM_THRESHOLD,
            "passed": bool(
                np.isfinite(dxy_momentum)
                and dxy_momentum >= DOLLAR_SHORT_DXY_MOMENTUM_THRESHOLD
            ),
        },
        "htf_1d_return_4": {
            "value": daily_return,
            "op": ">=",
            "threshold": DOLLAR_SHORT_HTF_1D_RETURN_4_THRESHOLD,
            "passed": bool(
                np.isfinite(daily_return)
                and daily_return >= DOLLAR_SHORT_HTF_1D_RETURN_4_THRESHOLD
            ),
        },
    }
    if not gates["dxy_momentum"]["passed"]:
        reasons.append("dxy_momentum_below_threshold")
    if not gates["htf_1d_return_4"]["passed"]:
        reasons.append("htf_1d_return_4_below_threshold")

    unique_reasons = list(dict.fromkeys(reasons))
    active = not unique_reasons
    return {
        "name": "dollar_rally_short",
        "active": active,
        "position": -1.0 if active else 0.0,
        "side": "SHORT" if active else "FLAT",
        "decision_time": decision.isoformat(),
        "execution_time": execution.isoformat(),
        "entry_window": {
            "opens_at": execution.isoformat(),
            "expires_at": entry_expires.isoformat(),
            "bars_5m": DOLLAR_SHORT_ENTRY_WINDOW_BARS,
        },
        "lifecycle": {
            "entry_time": execution.isoformat(),
            "exit_time": exit_time.isoformat(),
            "hold_bars_5m": DOLLAR_SHORT_HOLD_BARS,
            "hold_hours": 12,
            "take_profit": None,
            "stop_loss": None,
        },
        "global_phase": {
            "anchor": DOLLAR_SHORT_PHASE_ANCHOR.isoformat(),
            "signal_minute": 55,
            "entry_minute": 0,
            "matched": phase_matched,
        },
        "gates": gates,
        "reason": "active" if active else ";".join(unique_reasons),
    }
