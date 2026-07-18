"""Build DACC-48 signal/control clocks without post-entry return calculations."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from training import preregister_delayed_aftershock_compression_continuation as prereg


PREREGISTRATION_COMMIT = "1ecda026e2fd1e568d53da16c2e153e676d17a5d"
PREREGISTRATION_SOURCE_SHA256 = (
    "c93906ccf7ec3924736f022cd228277bf7dcca6c8bc1baa636f683b94b30d24e"
)
PREREGISTRATION_DOCUMENT = Path(
    "docs/delayed-aftershock-compression-continuation-preregistration-2026-07-18.md"
)
PREREGISTRATION_DOCUMENT_SHA256 = (
    "614dee461a07fd3889b7957ed8fb29d83da5a8f03443c11753f4c5f098dacaae"
)
PREREGISTRATION_RESULT = Path(prereg.DEFAULT_OUTPUT)
PREREGISTRATION_RESULT_SHA256 = (
    "126cda295359820fa360ef92488221a6c6f6c7dfdac5930e08f6c607d937835f"
)
PREREGISTRATION_MANIFEST_HASH = (
    "4fc7b7e56cf1d691e050f2fd20d7f18afb8c863302b1856dcc199e790160419d"
)

DEFAULT_OUTPUT = (
    "results/delayed_aftershock_compression_continuation_support_2026-07-18.json"
)
DEFAULT_CLOCK = (
    "results/delayed_aftershock_compression_continuation_clock_2026-07-18.csv"
)
SELECTION_END = pd.Timestamp("2024-01-01")
CLOCK_COLUMNS = (
    "policy",
    "anchor_position",
    "signal_position",
    "entry_position",
    "exit_position",
    "anchor_date",
    "signal_date",
    "entry_date",
    "exit_date",
    "side",
    "hold_bars",
)
POLICY_NAMES = (
    "primary",
    "direction_flip",
    "immediate_shock",
    "without_compression",
    "compression_without_flow",
    "without_range",
    "one_bar_delayed_entry",
    "shock_time_shift_one_day",
)


BASELINES: dict[str, dict[str, Any]] = {
    "jump_continuation_72_bidirectional_20260712": {
        "hold_bars": 96,
        "stride_bars": 6,
        "long": {
            "jv_jump_ratio_72": ("ge", 0.31734477856221055),
            "jv_signed_jump_72": ("ge", 0.15098978078538647),
            "jv_flow_recovery": ("ge", 0.025127509061786887),
        },
        "short": {
            "jv_jump_ratio_72": ("ge", 0.31734477856221055),
            "jv_signed_jump_72": ("le", -0.1495008467852352),
            "jv_flow_recovery": ("le", -0.025286664913192248),
        },
    },
    "jump_continuation_volume_clock_gate_20260712": {
        "hold_bars": 96,
        "stride_bars": 6,
        "long": {
            "jv_jump_ratio_72": ("ge", 0.31734477856221055),
            "jv_signed_jump_72": ("ge", 0.15098978078538647),
            "jv_flow_recovery": ("ge", 0.025127509061786887),
            "vc_flow_speed_0p25": ("ge", 0.00035365732808944646),
        },
        "short": {
            "jv_jump_ratio_72": ("ge", 0.31734477856221055),
            "jv_signed_jump_72": ("le", -0.1495008467852352),
            "jv_flow_recovery": ("le", -0.025286664913192248),
            "vc_flow_speed_0p25": ("le", -0.000487587473415171),
        },
    },
    "efficient_recovery_continuation_72_20260712": {
        "hold_bars": 144,
        "stride_bars": 12,
        "long": {
            "lr_signed_eff_72": ("ge", 0.1824548212170588),
            "lr_flow_72": ("ge", 0.04869572464172154),
            "lr_flow_recovery": ("ge", 0.06794383030855547),
        },
        "short": {
            "lr_signed_eff_72": ("le", -0.16331195830141687),
            "lr_flow_72": ("le", -0.055475434124467156),
            "lr_flow_recovery": ("le", -0.06783534014527763),
        },
    },
}


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_preregistration() -> dict[str, Any]:
    for path, expected in (
        (Path(prereg.__file__), PREREGISTRATION_SOURCE_SHA256),
        (PREREGISTRATION_DOCUMENT, PREREGISTRATION_DOCUMENT_SHA256),
        (PREREGISTRATION_RESULT, PREREGISTRATION_RESULT_SHA256),
    ):
        if _sha256(path) != expected:
            raise RuntimeError(f"frozen DACC-48 preregistration changed: {path}")
    payload = json.loads(PREREGISTRATION_RESULT.read_text())
    prereg.validate_manifest(payload)
    if payload["manifest_hash"] != PREREGISTRATION_MANIFEST_HASH:
        raise RuntimeError("DACC-48 preregistration manifest identity changed")
    if payload["outcomes_opened"] is not False:
        raise RuntimeError("DACC-48 outcomes opened before support freeze")
    if payload["policy"] != asdict(prereg.Policy()):
        raise RuntimeError("DACC-48 support policy differs from preregistration")
    return payload


def prior_quantile(
    values: pd.Series,
    *,
    quantile: float,
    window: int,
    min_periods: int,
) -> pd.Series:
    """Rolling quantile whose current observation is always excluded."""

    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return numeric.shift(1).rolling(window, min_periods=min_periods).quantile(quantile)


def load_support_frame() -> tuple[pd.DataFrame, dict[str, Any]]:
    payload = verify_preregistration()
    source = payload["source_contract"]
    for path, expected in (
        (source["market_manifest"], source["market_manifest_sha256"]),
        (source["market"], source["market_sha256"]),
    ):
        if _sha256(path) != expected:
            raise RuntimeError(f"DACC-48 source hash changed: {path}")
    manifest = json.loads(Path(source["market_manifest"]).read_text())
    if manifest.get("combined_sha256") != source["market_sha256"]:
        raise RuntimeError("DACC-48 source manifest does not bind the market file")
    usecols = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "quote_asset_volume",
        "taker_buy_quote",
    ]
    frame = pd.read_csv(
        source["market"],
        compression="gzip",
        usecols=usecols,
        parse_dates=["date"],
    )
    if len(frame) != source["market_rows"]:
        raise RuntimeError("DACC-48 market row count changed")
    expected_grid = pd.date_range("2020-01-01", "2023-12-31 23:55", freq="5min")
    if not frame["date"].equals(pd.Series(expected_grid, name="date")):
        raise RuntimeError("DACC-48 market grid is incomplete or reordered")
    if frame["date"].max() >= SELECTION_END:
        raise RuntimeError("DACC-48 support source contains 2024+ rows")
    values = frame[["open", "high", "low", "close"]].to_numpy(float)
    qv = frame["quote_asset_volume"].to_numpy(float)
    buy = frame["taker_buy_quote"].to_numpy(float)
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise RuntimeError("DACC-48 source has invalid OHLC")
    open_, high, low, close = values.T
    if (
        (high < np.maximum(open_, close)).any()
        or (low > np.minimum(open_, close)).any()
        or (high < low).any()
    ):
        raise RuntimeError("DACC-48 source violates OHLC invariants")
    if (
        not np.isfinite(qv).all()
        or not np.isfinite(buy).all()
        or (qv < 0.0).any()
        or (buy < 0.0).any()
        or (buy > qv).any()
    ):
        raise RuntimeError("DACC-48 source has invalid quote/taker volume")
    metadata = {
        "market_manifest_sha256": _sha256(source["market_manifest"]),
        "market_sha256": _sha256(source["market"]),
        "market_rows": int(len(frame)),
        "zero_quote_volume_bars": int(np.sum(qv == 0.0)),
        "first_date": str(frame["date"].iloc[0]),
        "last_date": str(frame["date"].iloc[-1]),
        "loaded_columns": usecols,
        "post_entry_returns_loaded": False,
        "future_trade_returns_computed": False,
        "funding_loaded": False,
    }
    return frame, metadata


def feature_frame(frame: pd.DataFrame, policy: prereg.Policy) -> pd.DataFrame:
    close = frame["close"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    qv = frame["quote_asset_volume"].astype(float)
    buy = frame["taker_buy_quote"].astype(float)
    returns = np.log(close / close.shift(1))
    signed_quote = 2.0 * buy - qv
    imbalance = signed_quote / qv
    bipower = returns.abs() * returns.shift(1).abs()
    sigma72_pre = np.sqrt(
        (np.pi / 2.0)
        * bipower.rolling(policy.shock_scale_bars, min_periods=policy.shock_scale_bars)
        .mean()
        .shift(1)
    )
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            np.log(high / low),
            np.log(high / previous_close).abs(),
            np.log(low / previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    flow6 = signed_quote.rolling(policy.compression_bars).sum() / qv.rolling(
        policy.compression_bars
    ).sum()
    flow3 = signed_quote.rolling(policy.reacceleration_bars).sum() / qv.rolling(
        policy.reacceleration_bars
    ).sum()
    return pd.DataFrame(
        {
            "return": returns,
            "shock_abs_q": prior_quantile(
                returns.abs(),
                quantile=policy.shock_abs_quantile,
                window=policy.reference_bars,
                min_periods=policy.reference_min_periods,
            ),
            "sigma72_pre": sigma72_pre,
            "quote_median_pre": prior_quantile(
                qv,
                quantile=0.5,
                window=policy.reference_bars,
                min_periods=policy.reference_min_periods,
            ),
            "signed_quote": signed_quote,
            "imbalance": imbalance,
            "true_range": true_range,
            "flow6_abs_median_pre": prior_quantile(
                flow6.abs(),
                quantile=0.5,
                window=policy.reference_bars,
                min_periods=policy.reference_min_periods,
            ),
            "flow3_abs_q70_pre": prior_quantile(
                flow3.abs(),
                quantile=policy.reacceleration_flow_quantile,
                window=policy.reference_bars,
                min_periods=policy.reference_min_periods,
            ),
        },
        index=frame.index,
    ).replace([np.inf, -np.inf], np.nan)


def _row(
    frame: pd.DataFrame,
    *,
    policy_name: str,
    anchor: int,
    signal: int,
    entry: int,
    side: int,
    hold: int,
) -> dict[str, Any] | None:
    exit_position = entry + hold
    if anchor < 0 or signal < 0 or entry < 0 or exit_position >= len(frame):
        return None
    return {
        "policy": policy_name,
        "anchor_position": int(anchor),
        "signal_position": int(signal),
        "entry_position": int(entry),
        "exit_position": int(exit_position),
        "anchor_date": frame.at[anchor, "date"],
        "signal_date": frame.at[signal, "date"],
        "entry_date": frame.at[entry, "date"],
        "exit_date": frame.at[exit_position, "date"],
        "side": int(side),
        "hold_bars": int(hold),
    }


def _nonoverlap(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    last_exit = -1
    for row in sorted(rows, key=lambda value: (value["entry_position"], value["anchor_position"])):
        if row["entry_position"] <= last_exit:
            continue
        selected.append(row)
        last_exit = int(row["exit_position"])
    return selected


def candidate_clocks(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    policy: prereg.Policy,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    price = frame[["high", "low", "close", "quote_asset_volume"]].to_numpy(float)
    high, low, close, qv = price.T
    returns = features["return"].to_numpy(float)
    imbalance = features["imbalance"].to_numpy(float)
    signed = features["signed_quote"].to_numpy(float)
    tr = features["true_range"].to_numpy(float)
    shock_q = features["shock_abs_q"].to_numpy(float)
    sigma = features["sigma72_pre"].to_numpy(float)
    qv_median = features["quote_median_pre"].to_numpy(float)
    flow6_median = features["flow6_abs_median_pre"].to_numpy(float)
    flow3_q70 = features["flow3_abs_q70_pre"].to_numpy(float)
    floor = policy.minimum_shock_return_bp / 10_000.0
    break_floor = policy.minimum_break_margin_bp / 10_000.0
    accel_floor = policy.minimum_reacceleration_return_bp / 10_000.0
    raw: dict[str, list[dict[str, Any]]] = {name: [] for name in POLICY_NAMES}
    diagnostics: list[dict[str, float | int]] = []
    stop = len(frame) - (policy.entry_offset_bars + policy.hold_bars)
    for anchor in range(policy.reference_min_periods + policy.shock_scale_bars, stop):
        shock = returns[anchor]
        if not np.isfinite(
            [shock, shock_q[anchor], sigma[anchor], qv_median[anchor]]
        ).all():
            continue
        side = int(np.sign(shock))
        if side == 0:
            continue
        shock_price = (
            abs(shock) >= max(floor, shock_q[anchor])
            and sigma[anchor] > 0.0
            and abs(shock) / sigma[anchor] >= policy.minimum_shock_scale_multiple
            and qv[anchor] >= qv_median[anchor]
        )
        shock_flow = (
            np.isfinite(imbalance[anchor])
            and side * imbalance[anchor]
            >= policy.minimum_shock_directional_taker_imbalance
        )
        if not shock_price:
            continue
        immediate = _row(
            frame,
            policy_name="immediate_shock",
            anchor=anchor,
            signal=anchor,
            entry=anchor + 1,
            side=side,
            hold=policy.hold_bars,
        )
        if shock_flow and immediate is not None:
            raw["immediate_shock"].append(immediate)

        comp_slice = slice(anchor + 1, anchor + 1 + policy.compression_bars)
        accel_start = anchor + 1 + policy.compression_bars
        accel_slice = slice(accel_start, accel_start + policy.reacceleration_bars)
        trigger = anchor + policy.trigger_offset_bars
        required = np.r_[
            high[comp_slice],
            low[comp_slice],
            close[comp_slice],
            qv[comp_slice],
            signed[comp_slice],
            tr[comp_slice],
            close[accel_slice],
            qv[accel_slice],
            signed[accel_slice],
            tr[accel_slice],
            flow6_median[anchor],
            flow3_q70[anchor],
        ]
        if not np.isfinite(required).all():
            continue
        if (
            qv[anchor] <= 0.0
            or (qv[comp_slice] <= 0.0).any()
            or (qv[accel_slice] <= 0.0).any()
        ):
            continue
        box_high = float(np.max(high[comp_slice]))
        box_low = float(np.min(low[comp_slice]))
        box_width = float(np.log(box_high / box_low))
        comp_net = float(abs(np.log(close[anchor + policy.compression_bars] / close[anchor])))
        adverse_price = box_low if side > 0 else box_high
        adverse_box = float(max(0.0, -side * np.log(adverse_price / close[anchor])))
        comp_flow = float(np.sum(signed[comp_slice]) / np.sum(qv[comp_slice]))
        acc_flow = float(np.sum(signed[accel_slice]) / np.sum(qv[accel_slice]))
        acc_ret = float(
            side
            * np.log(
                close[trigger] / close[anchor + policy.compression_bars]
            )
        )
        edge = box_high if side > 0 else box_low
        break_margin = float(side * np.log(close[trigger] / edge))
        range_acceleration = float(np.mean(tr[accel_slice]) / np.mean(tr[comp_slice]))
        compression_price_ok = (
            box_width <= policy.maximum_compression_to_shock_ratio * abs(shock)
            and comp_net <= policy.maximum_compression_net_fraction * abs(shock)
            and adverse_box <= policy.maximum_retrace_fraction * abs(shock)
        )
        compression_flow_ok = abs(comp_flow) <= flow6_median[anchor]
        compression_ok = compression_price_ok and compression_flow_ok
        flow_ok = (
            side * acc_flow >= max(policy.minimum_directional_flow, flow3_q70[anchor])
            and side * (acc_flow - comp_flow) >= policy.minimum_flow_acceleration
        )
        range_ok = (
            break_margin >= break_floor
            and acc_ret
            >= max(accel_floor, policy.minimum_reacceleration_box_fraction * box_width)
            and range_acceleration >= policy.minimum_range_acceleration
        )
        base = {
            "frame": frame,
            "anchor": anchor,
            "signal": trigger,
            "entry": anchor + policy.entry_offset_bars,
            "side": side,
            "hold": policy.hold_bars,
        }
        if shock_flow and compression_ok and flow_ok and range_ok:
            row = _row(policy_name="primary", **base)
            if row is not None:
                raw["primary"].append(row)
                diagnostics.append(
                    {
                        "anchor_position": anchor,
                        "side": side,
                        "shock_abs": abs(shock),
                        "box_width": box_width,
                        "compression_flow": comp_flow,
                        "reacceleration_flow": acc_flow,
                        "break_margin": break_margin,
                        "range_acceleration": range_acceleration,
                    }
                )
        if shock_flow and flow_ok and range_ok:
            row = _row(policy_name="without_compression", **base)
            if row is not None:
                raw["without_compression"].append(row)
        if compression_price_ok and range_ok:
            row = _row(policy_name="compression_without_flow", **base)
            if row is not None:
                raw["compression_without_flow"].append(row)
        if shock_flow and compression_ok and flow_ok:
            row = _row(policy_name="without_range", **base)
            if row is not None:
                raw["without_range"].append(row)

    selected = {name: _nonoverlap(rows) for name, rows in raw.items()}
    selected["direction_flip"] = [
        {**row, "policy": "direction_flip", "side": -int(row["side"])}
        for row in selected["primary"]
    ]
    delayed = []
    for row in selected["primary"]:
        shifted = _row(
            frame,
            policy_name="one_bar_delayed_entry",
            anchor=int(row["anchor_position"]),
            signal=int(row["signal_position"]),
            entry=int(row["entry_position"]) + 1,
            side=int(row["side"]),
            hold=policy.hold_bars,
        )
        if shifted is not None:
            delayed.append(shifted)
    selected["one_bar_delayed_entry"] = _nonoverlap(delayed)
    placebo = []
    for row in selected["primary"]:
        shifted = _row(
            frame,
            policy_name="shock_time_shift_one_day",
            anchor=int(row["anchor_position"]) - 288,
            signal=int(row["signal_position"]) - 288,
            entry=int(row["entry_position"]) - 288,
            side=int(row["side"]),
            hold=policy.hold_bars,
        )
        if shifted is not None:
            placebo.append(shifted)
    selected["shock_time_shift_one_day"] = _nonoverlap(placebo)
    rows = [row for name in POLICY_NAMES for row in selected[name]]
    clock = pd.DataFrame(rows, columns=CLOCK_COLUMNS)
    assert_clock_contract(clock)
    diagnostic_frame = pd.DataFrame(diagnostics)
    metadata = {
        "raw_event_counts": {name: len(raw[name]) for name in raw},
        "nonoverlap_event_counts": {name: len(selected[name]) for name in POLICY_NAMES},
        "primary_feature_summary": {
            column: {
                "min": float(diagnostic_frame[column].min()),
                "median": float(diagnostic_frame[column].median()),
                "max": float(diagnostic_frame[column].max()),
            }
            for column in diagnostic_frame.columns
            if column not in {"anchor_position", "side"} and not diagnostic_frame.empty
        },
    }
    return clock, metadata


def _condition_mask(features: pd.DataFrame, conditions: dict[str, tuple[str, float]]) -> np.ndarray:
    mask = np.ones(len(features), dtype=bool)
    for column, (operator, threshold) in conditions.items():
        values = features[column].to_numpy(float)
        mask &= np.isfinite(values)
        mask &= values >= threshold if operator == "ge" else values <= threshold
    return mask


def baseline_features(frame: pd.DataFrame) -> pd.DataFrame:
    close = frame["close"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    qv = frame["quote_asset_volume"].astype(float)
    buy = frame["taker_buy_quote"].astype(float)
    returns = np.log(close).diff()
    signed = 2.0 * buy - qv
    imbalance = signed / qv
    rv = returns.pow(2).rolling(72, min_periods=72).sum()
    bv = (np.pi / 2.0) * (
        returns.abs() * returns.shift(1).abs()
    ).rolling(72, min_periods=72).sum()
    jump = (rv - bv).clip(lower=0.0)
    path = returns.abs().rolling(72, min_periods=72).sum()
    net = np.log(close / close.shift(72))
    flow72 = signed.rolling(72, min_periods=72).sum() / qv.rolling(
        72, min_periods=72
    ).sum()
    output = pd.DataFrame(
        {
            "jv_jump_ratio_72": jump / rv.replace(0.0, np.nan),
            "jv_signed_jump_72": returns.pow(3).rolling(72, min_periods=72).sum()
            / rv.replace(0.0, np.nan).pow(1.5),
            "jv_flow_recovery": imbalance.rolling(12, min_periods=12).mean()
            - imbalance.rolling(48, min_periods=48).mean(),
            "lr_signed_eff_72": net / path.replace(0.0, np.nan),
            "lr_flow_72": flow72,
            "lr_flow_recovery": imbalance.rolling(12, min_periods=12).mean()
            - imbalance.rolling(72, min_periods=72).mean(),
        },
        index=frame.index,
    )
    cumulative_volume = np.r_[0.0, np.cumsum(qv.to_numpy(float))]
    cumulative_signed = np.r_[0.0, np.cumsum(signed.to_numpy(float))]
    target = qv.rolling(288, min_periods=288).sum().shift(1).to_numpy(float) * 0.25
    index = np.arange(len(frame))
    level = cumulative_volume[1:] - np.nan_to_num(target, nan=np.inf)
    start = np.searchsorted(cumulative_volume, level, side="left").clip(0, len(frame) - 1)
    valid = np.isfinite(target) & (start < index)
    duration = (index - start).astype(float)
    volume = cumulative_volume[index + 1] - cumulative_volume[start]
    directional = cumulative_signed[index + 1] - cumulative_signed[start]
    speed = (directional / np.where(volume == 0.0, np.nan, volume)) / np.where(
        duration == 0.0, np.nan, duration
    )
    speed[~valid] = np.nan
    output["vc_flow_speed_0p25"] = speed
    return output.replace([np.inf, -np.inf], np.nan)


def baseline_activation_clocks(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    features = baseline_features(frame)
    output: dict[str, dict[str, Any]] = {}
    for name, spec in BASELINES.items():
        long_mask = _condition_mask(features, spec["long"])
        short_mask = _condition_mask(features, spec["short"])
        positions = np.arange(
            143,
            len(frame) - int(spec["hold_bars"]) - 2,
            int(spec["stride_bars"]),
            dtype=np.int64,
        )
        active = positions[long_mask[positions] ^ short_mask[positions]]
        entries = active + 1
        sides = np.where(long_mask[active], 1, -1)
        output[name] = {
            "signal_positions": active,
            "entry_positions": entries,
            "sides": sides,
            "hold_bars": int(spec["hold_bars"]),
            "clock_definition": (
                "all stride-aligned fixed-rule activations, before outcome-dependent "
                "TP/SL de-overlap; max-hold occupancy is conservative"
            ),
        }
    return output


def _jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else 0.0


def _position_mask(length: int, entries: np.ndarray, hold: int) -> np.ndarray:
    difference = np.zeros(length + 1, dtype=np.int64)
    for entry in entries.astype(int):
        end = min(length, entry + hold)
        difference[entry] += 1
        difference[end] -= 1
    return np.cumsum(difference[:-1]) > 0


def orthogonality_metrics(
    clock: pd.DataFrame,
    baselines: dict[str, dict[str, Any]],
    *,
    length: int,
) -> dict[str, Any]:
    primary = clock.loc[clock["policy"].eq("primary")]
    candidate_entries = primary["entry_position"].to_numpy(np.int64)
    candidate_set = set(candidate_entries.tolist())
    candidate_position = _position_mask(length, candidate_entries, prereg.Policy().hold_bars)
    rows: dict[str, Any] = {}
    union_entries: set[int] = set()
    for name, baseline in baselines.items():
        entries = baseline["entry_positions"].astype(np.int64)
        entry_set = set(entries.tolist())
        union_entries |= entry_set
        base_position = _position_mask(length, entries, int(baseline["hold_bars"]))
        rows[name] = {
            "baseline_activation_entries": int(len(entries)),
            "exact_entry_jaccard": _jaccard(candidate_set, entry_set),
            "position_time_jaccard": _jaccard(
                set(np.flatnonzero(candidate_position).tolist()),
                set(np.flatnonzero(base_position).tolist()),
            ),
        }
    sorted_union = np.array(sorted(union_entries), dtype=np.int64)
    if len(candidate_entries) and len(sorted_union):
        insertion = np.searchsorted(sorted_union, candidate_entries)
        left = sorted_union[np.clip(insertion - 1, 0, len(sorted_union) - 1)]
        right = sorted_union[np.clip(insertion, 0, len(sorted_union) - 1)]
        distance = np.minimum(np.abs(candidate_entries - left), np.abs(candidate_entries - right))
        within = float(np.mean(distance <= 72))
    else:
        within = 0.0
    return {
        "admission_eligible": False,
        "reason": (
            "exact committed baseline trade clocks do not exist; reconstructed "
            "stride-aligned activation clocks are diagnostic proxies only"
        ),
        "by_baseline": rows,
        "maximum_exact_entry_jaccard": max(
            (row["exact_entry_jaccard"] for row in rows.values()), default=0.0
        ),
        "maximum_position_time_jaccard": max(
            (row["position_time_jaccard"] for row in rows.values()), default=0.0
        ),
        "candidate_entries_within_six_hours_of_any_baseline_share": within,
        "baseline_union_entries": int(len(union_entries)),
    }


def assert_clock_contract(clock: pd.DataFrame) -> None:
    if list(clock.columns) != list(CLOCK_COLUMNS):
        raise RuntimeError("DACC-48 clock schema changed")
    if not set(clock["policy"]).issubset(POLICY_NAMES):
        raise RuntimeError("DACC-48 clock contains an unknown policy")
    if clock.empty:
        return
    for column in ("anchor_position", "signal_position", "entry_position", "exit_position"):
        values = clock[column].to_numpy()
        if not np.issubdtype(values.dtype, np.integer):
            raise RuntimeError(f"DACC-48 clock position is not integer: {column}")
    for column in ("anchor_date", "signal_date", "entry_date", "exit_date"):
        if pd.to_datetime(clock[column], errors="coerce").isna().any():
            raise RuntimeError(f"DACC-48 clock date is invalid: {column}")
    if not (clock["anchor_position"] <= clock["signal_position"]).all():
        raise RuntimeError("DACC-48 signal precedes its anchor")
    if not (clock["signal_position"] < clock["entry_position"]).all():
        raise RuntimeError("DACC-48 entry is not after its signal")
    if not (clock["entry_position"] < clock["exit_position"]).all():
        raise RuntimeError("DACC-48 exit is not after its entry")
    if not clock["side"].isin([-1, 1]).all():
        raise RuntimeError("DACC-48 clock side is invalid")
    forbidden = {"open", "high", "low", "close", "return", "pnl", "funding"}
    if forbidden & set(clock.columns):
        raise RuntimeError("DACC-48 clock leaked an outcome field")


def _window_count(clock: pd.DataFrame, start: str, end: str) -> int:
    entry = pd.to_datetime(clock["entry_date"])
    exit_ = pd.to_datetime(clock["exit_date"])
    return int(((entry >= pd.Timestamp(start)) & (exit_ < pd.Timestamp(end))).sum())


def support_summary(clock: pd.DataFrame, orthogonality: dict[str, Any]) -> dict[str, Any]:
    primary = clock.loc[clock["policy"].eq("primary")].copy()
    counts = {
        "train_2020_2022": _window_count(primary, "2020-01-01", "2023-01-01"),
        "2020": _window_count(primary, "2020-01-01", "2021-01-01"),
        "2021": _window_count(primary, "2021-01-01", "2022-01-01"),
        "2022": _window_count(primary, "2022-01-01", "2023-01-01"),
        "2023": _window_count(primary, "2023-01-01", "2024-01-01"),
        "2023_h1": _window_count(primary, "2023-01-01", "2023-07-01"),
        "2023_h2": _window_count(primary, "2023-07-01", "2024-01-01"),
    }
    total = len(primary)
    long_share = float(primary["side"].eq(1).mean()) if total else 0.0
    entry = pd.to_datetime(primary["entry_date"])
    monthly = entry.dt.to_period("M").value_counts()
    weekly = entry.dt.to_period("W-SUN").value_counts()
    max_month = float(monthly.max() / total) if total else 1.0
    max_week = float(weekly.max() / total) if total else 1.0
    policy = prereg.Policy()
    checks = {
        "train_count": counts["train_2020_2022"] >= 150,
        "each_train_year": min(counts["2020"], counts["2021"], counts["2022"]) >= 30,
        "selection_count": counts["2023"] >= 40,
        "selection_halves": min(counts["2023_h1"], counts["2023_h2"]) >= 15,
        "direction_balance": 0.30 <= long_share <= 0.70,
        "monthly_concentration": max_month <= 0.15,
        "weekly_concentration": max_week <= 0.08,
        "exact_entry_jaccard": orthogonality["maximum_exact_entry_jaccard"] <= 0.02,
        "six_hour_proximity": orthogonality[
            "candidate_entries_within_six_hours_of_any_baseline_share"
        ]
        <= 0.25,
        "position_time_jaccard": orthogonality["maximum_position_time_jaccard"] <= 0.15,
        "exact_baseline_clock_binding": orthogonality["admission_eligible"],
        "fixed_hold": primary["hold_bars"].eq(policy.hold_bars).all() if total else False,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    return {
        "counts": counts,
        "primary_total": int(total),
        "long_share": long_share,
        "short_share": 1.0 - long_share if total else 0.0,
        "maximum_single_month_share": max_month,
        "maximum_single_utc_week_share": max_week,
        "checks": checks,
        "passes_support": all(checks.values()),
    }


def _write_once(path: str | Path, content: bytes) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.read_bytes() != content:
            raise RuntimeError(f"refusing to overwrite frozen DACC-48 artifact: {output}")
        return "verified_existing"
    with output.open("xb") as handle:
        handle.write(content)
    return "created"


def run(output: str | Path = DEFAULT_OUTPUT, clock_path: str | Path = DEFAULT_CLOCK) -> dict[str, Any]:
    preregistration = verify_preregistration()
    frame, source_metadata = load_support_frame()
    features = feature_frame(frame, prereg.Policy())
    clock, clock_metadata = candidate_clocks(frame, features, prereg.Policy())
    baselines = baseline_activation_clocks(frame)
    orthogonality = orthogonality_metrics(clock, baselines, length=len(frame))
    support = support_summary(clock, orthogonality)
    clock_bytes = clock.to_csv(index=False, date_format="%Y-%m-%d %H:%M:%S").encode()
    clock_status = _write_once(clock_path, clock_bytes)
    manifest: dict[str, Any] = {
        "protocol": "DACC-48 outcome-blind support and orthogonality freeze v1",
        "outcomes_opened": False,
        "post_entry_returns_or_pnl_calculated": False,
        "funding_loaded": False,
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "preregistration_sha256": PREREGISTRATION_RESULT_SHA256,
        "preregistration_manifest_hash": preregistration["manifest_hash"],
        "policy": asdict(prereg.Policy()),
        "implementation": {
            "path": str(Path(__file__).resolve().relative_to(Path.cwd().resolve())),
            "sha256": _sha256(__file__),
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "source": source_metadata,
        "clock": {
            "path": str(clock_path),
            "sha256": hashlib.sha256(clock_bytes).hexdigest(),
            "rows": int(len(clock)),
            "columns": list(CLOCK_COLUMNS),
            "write_contract": "immutable bytes; repeated runs must reproduce the clock",
            **clock_metadata,
        },
        "baseline_activation_contracts": {
            name: {
                "entry_count": int(len(value["entry_positions"])),
                "hold_bars": int(value["hold_bars"]),
                "clock_definition": value["clock_definition"],
                "exact_committed_trade_clock_bound": False,
                "admission_use": "diagnostic_only_fail_closed",
                "thresholds": BASELINES[name],
            }
            for name, value in baselines.items()
        },
        "orthogonality": orthogonality,
        "support": support,
        "sealed_windows": ["all_post_entry_outcomes", "2024", "2025", "2026_ytd"],
    }
    body = {
        key: value
        for key, value in manifest.items()
        if key != "manifest_hash"
    }
    manifest["manifest_hash"] = prereg.canonical_hash(body)
    content = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode()
    manifest["runtime_write_status"] = {
        "clock": clock_status,
        "manifest": _write_once(output, content),
    }
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--clock", default=DEFAULT_CLOCK)
    args = parser.parse_args()
    payload = run(args.output, args.clock)
    print(
        json.dumps(
            {
                "outcomes_opened": False,
                "passes_support": payload["support"]["passes_support"],
                "support": payload["support"],
                "orthogonality": payload["orthogonality"],
                "clock_rows": payload["clock"]["rows"],
                "clock_sha256": payload["clock"]["sha256"],
                "output": args.output,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
