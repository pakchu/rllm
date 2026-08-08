"""Build source-only support clocks for frozen DCVPR-12."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import backtest_all_alpha_month as month
from training import preregister_daily_cross_venue_participation_rotation_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


SOURCE = Path("data/daily_cross_venue_participation_rotation_relay_sources_2022_2026/signal_features.csv.gz")
CLOCK = Path("data/daily_cross_venue_participation_rotation_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/daily_cross_venue_participation_rotation_relay_controls_2023_2026")
RESULT = Path("results/daily_cross_venue_participation_rotation_relay_support_2026-08-08.json")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM_EVENTS = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_volatility_gate",
    "no_rotation_tail",
    "absolute_ratio_level",
    "one_day_stale_rotation",
    "direction_flip",
)
ECONOMIC_OUTCOMES_AUTHORIZED = False
COLUMNS = (
    "candidate", "control", "split", "decision_time", "feature_available_time",
    "entry_time", "exit_time", "side", "participation_z", "rotation",
    "absolute_rotation_rank", "realized_variation", "realized_variation_rank",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def strict_prior_midrank(
    values: pd.Series, lookback: int = 180, minimum: int = 126
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    result = pd.Series(np.nan, index=numeric.index, dtype=float)
    history: list[float] = []
    for index, current in numeric.items():
        prior = history[-lookback:]
        if math.isfinite(current) and len(prior) >= minimum:
            array = np.asarray(prior)
            result.at[index] = (
                np.sum(array < current) + 0.5 * np.sum(array == current)
            ) / len(array)
        if math.isfinite(current):
            history.append(current)
    return result


def query_snapshot() -> tuple[pd.DataFrame, dict[str, Any]]:
    cfg = month.Config(
        start="2022-12-01T00:00:00Z",
        end="2026-08-01T00:00:00Z",
        asof="2026-08-01T00:02:00Z",
        lookback_minutes=3_000_000,
    )
    market, feature_frame, _funding, engine = asyncio.run(month._query_frames(cfg))
    if engine is not None:
        engine.dispose()
    required = {"vg_upbit_binance_vol_ratio_z_288", "upbit_volume_available"}
    missing = sorted(required - set(feature_frame.columns))
    if missing:
        raise RuntimeError(f"DCVPR source columns missing: {missing}")
    snapshot = pd.DataFrame(
        {
            "date": pd.to_datetime(market.date, utc=True),
            "close": pd.to_numeric(market.close, errors="coerce").to_numpy(),
            "participation_z": pd.to_numeric(
                feature_frame.vg_upbit_binance_vol_ratio_z_288, errors="coerce"
            ).to_numpy(),
            "upbit_volume_available": pd.to_numeric(
                feature_frame.upbit_volume_available, errors="coerce"
            ).to_numpy(),
        }
    )
    snapshot = snapshot[
        snapshot.date.between(
            pd.Timestamp("2022-12-01T00:00:00Z"),
            pd.Timestamp("2026-08-01T00:00:00Z"),
            inclusive="left",
        )
    ].reset_index(drop=True)
    if snapshot.date.duplicated().any() or not snapshot.date.is_monotonic_increasing:
        raise RuntimeError("DCVPR source snapshot time drift")
    return snapshot, {
        "mode": "postgres_live_feature_builder_completed_bar",
        "rows": len(snapshot),
        "first": str(snapshot.date.iloc[0]),
        "last": str(snapshot.date.iloc[-1]),
        "signal_dependent_sources": ["bars_binance", "bars_upbit", "causal_usdkrw"],
    }


def daily_features(snapshot: pd.DataFrame) -> pd.DataFrame:
    frame = snapshot.sort_values("date").reset_index(drop=True).copy()
    consecutive = frame.date.diff().eq(pd.Timedelta(minutes=5))
    price_valid = np.isfinite(frame.close) & frame.close.gt(0)
    frame["log_return"] = np.log(frame.close / frame.close.shift(1))
    participation_valid = (
        frame.upbit_volume_available.ge(0.5)
        & np.isfinite(frame.participation_z)
        & consecutive
    )
    frame["rotation"] = frame.participation_z - frame.participation_z.shift(288)
    frame["realized_variation"] = np.sqrt(
        frame.log_return.pow(2).rolling(288, min_periods=288).sum()
    )
    frame["source_valid"] = (
        price_valid.rolling(289, min_periods=289).sum().eq(289)
        & consecutive.rolling(288, min_periods=288).sum().eq(288)
        & participation_valid.rolling(289, min_periods=289).sum().eq(289)
        & np.isfinite(frame[["rotation", "realized_variation"]]).all(axis=1)
        & frame.rotation.ne(0)
    )
    daily = frame[
        frame.date.dt.hour.eq(23) & frame.date.dt.minute.eq(55)
    ][["date", "participation_z", "rotation", "realized_variation", "source_valid"]].copy()
    daily["decision_time"] = daily.date + pd.Timedelta(minutes=5)
    daily = daily.reset_index(drop=True)
    daily["absolute_rotation_rank"] = strict_prior_midrank(
        daily.rotation.abs().where(daily.source_valid)
    )
    daily["realized_variation_rank"] = strict_prior_midrank(
        daily.realized_variation.where(daily.source_valid)
    )
    daily["absolute_level_rank"] = strict_prior_midrank(
        daily.participation_z.abs().where(daily.source_valid)
    )
    return daily


def conditions(frame: pd.DataFrame, control: str) -> tuple[pd.Series, pd.Series]:
    rotation = frame.rotation
    rotation_rank = frame.absolute_rotation_rank
    if control == "one_day_stale_rotation":
        rotation = rotation.shift(1)
        rotation_rank = rotation_rank.shift(1)
    if control == "absolute_ratio_level":
        rotation = frame.participation_z
        rotation_rank = frame.absolute_level_rank
    rotation_gate = (
        pd.Series(True, index=frame.index)
        if control == "no_rotation_tail"
        else rotation_rank.ge(0.65)
    )
    volatility_gate = (
        pd.Series(True, index=frame.index)
        if control == "no_volatility_gate"
        else frame.realized_variation_rank.ge(0.65)
    )
    active = (
        frame.source_valid
        & np.isfinite(rotation)
        & rotation.ne(0)
        & rotation_rank.notna()
        & frame.realized_variation_rank.notna()
        & rotation_gate
        & volatility_gate
    )
    return active, np.sign(rotation)


def clock(frame: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    active, side = conditions(frame, control)
    rows: list[dict[str, Any]] = []
    next_allowed: pd.Timestamp | None = None
    for index in frame.index[active]:
        decision = pd.Timestamp(frame.at[index, "decision_time"])
        entry = decision + pd.Timedelta(minutes=5)
        exit_time = entry + pd.Timedelta(hours=12)
        if next_allowed is not None and entry < next_allowed:
            continue
        split = next(
            (name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end),
            None,
        )
        if split is None:
            continue
        signal_side = int(side.at[index])
        if control == "direction_flip":
            signal_side = -signal_side
        next_allowed = exit_time
        rows.append(
            {
                "candidate": "DCVPR-12",
                "control": control,
                "split": split,
                "decision_time": decision,
                "feature_available_time": decision,
                "entry_time": entry,
                "exit_time": exit_time,
                "side": signal_side,
                "participation_z": float(frame.at[index, "participation_z"]),
                "rotation": float(frame.at[index, "rotation"]),
                "absolute_rotation_rank": float(frame.at[index, "absolute_rotation_rank"]),
                "realized_variation": float(frame.at[index, "realized_variation"]),
                "realized_variation_rank": float(frame.at[index, "realized_variation_rank"]),
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def split_stats(clocks: pd.DataFrame, split: str) -> dict[str, int | float]:
    selected = clocks[clocks.split.eq(split)]
    if selected.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs = int(selected.side.eq(1).sum())
    shorts = int(selected.side.eq(-1).sum())
    monthly = selected.entry_time.dt.strftime("%Y-%m").value_counts()
    return {
        "events": len(selected),
        "longs": longs,
        "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(selected),
        "max_month_share": int(monthly.max()) / len(selected),
    }


def run() -> dict[str, Any]:
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    snapshot, source_info = query_snapshot()
    SOURCE.parent.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(snapshot, SOURCE)
    feature_frame = daily_features(snapshot)
    primary = clock(feature_frame)
    controls = {name: clock(feature_frame, name) for name in CONTROLS}
    CLOCK.parent.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(primary, CLOCK)
    for name, control_clocks in controls.items():
        _write_gzip_csv(control_clocks, CONTROL_DIR / f"{name}.csv.gz")
    support = {name: split_stats(primary, name) for name in SPLITS}
    checks: dict[str, bool] = {}
    for name, stats in support.items():
        checks[f"{name}_minimum_events"] = stats["events"] >= MINIMUM_EVENTS[name]
        checks[f"{name}_side_balance"] = stats["minority_side_share"] >= 0.20
        checks[f"{name}_month_concentration"] = stats["max_month_share"] <= 0.45
    passed = all(checks.values())
    core = {
        "protocol_version": "dcvpr_12_source_support_v1",
        "policy_id": "DCVPR-12",
        "preregistration": {
            "path": str(prereg.DEFAULT_OUTPUT),
            "sha256": sha(prereg.DEFAULT_OUTPUT),
            "manifest_hash": registration["manifest_hash"],
        },
        "source": source_info,
        "source_snapshot": {"path": str(SOURCE), "sha256": sha(SOURCE), "rows": len(snapshot)},
        "completed_preentry_sources_opened": True,
        "postentry_return_pnl_execution_price_opened": False,
        "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "controls": {
            name: {
                "path": str(CONTROL_DIR / f"{name}.csv.gz"),
                "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"),
                "rows": len(control_clocks),
                "promotion_authorized": False,
            }
            for name, control_clocks in controls.items()
        },
        "support": support,
        "support_checks": checks,
        "support_passed": passed,
        "advance_to_gross9_novelty": passed,
        "advance_to_economic_outcomes": ECONOMIC_OUTCOMES_AUTHORIZED,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    report = run()
    print(json.dumps({"passed": report["support_passed"], "support": report["support"]}, indent=2))
