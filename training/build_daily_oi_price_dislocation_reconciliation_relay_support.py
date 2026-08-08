"""Build source-only support clocks for frozen DOPDR-12."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_daily_oi_price_dislocation_reconciliation_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


OI_DIR = Path("data/options_crowding_deleveraging_relay_sources_v4_2023_2026")
PRICE_DIR = Path("data/options_oi_chase_exhaustion_sources_2023_2026")
CLOCK = Path("data/daily_oi_price_dislocation_reconciliation_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/daily_oi_price_dislocation_reconciliation_relay_controls_2023_2026")
RESULT = Path("results/daily_oi_price_dislocation_reconciliation_relay_support_2026-08-08.json")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM_EVENTS = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_realized_variation_gate",
    "no_displacement_gate",
    "same_direction_only",
    "one_day_stale_ranks",
    "direction_flip",
)
ECONOMIC_OUTCOMES_AUTHORIZED = False
COLUMNS = (
    "candidate",
    "control",
    "split",
    "decision_time",
    "feature_available_time",
    "entry_time",
    "exit_time",
    "side",
    "price_return",
    "oi_return",
    "realized_variation",
    "realized_variation_rank",
    "displacement",
    "displacement_rank",
    "oi_current_time",
    "oi_prior_time",
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


def _asof_oi(decisions: pd.DataFrame, oi: pd.DataFrame, target: str, prefix: str) -> pd.DataFrame:
    left = decisions[[target]].sort_values(target).rename(columns={target: "target_time"})
    right = oi[["ts", "sum_open_interest"]].sort_values("ts")
    merged = pd.merge_asof(
        left,
        right,
        left_on="target_time",
        right_on="ts",
        direction="backward",
        allow_exact_matches=True,
    )
    return merged.rename(
        columns={"ts": f"{prefix}_time", "sum_open_interest": f"{prefix}_oi"}
    )


def features() -> pd.DataFrame:
    price = pd.read_csv(PRICE_DIR / "btc_completed_hour.csv.gz", compression="gzip")
    price["decision_time"] = pd.to_datetime(price.decision_time, utc=True, format="mixed")
    for column in ("open", "close"):
        price[column] = pd.to_numeric(price[column], errors="coerce")
    price["source_valid"] = price.source_valid.astype(str).str.lower().eq("true")
    price = price.sort_values("decision_time").reset_index(drop=True)
    price["hour_log_return"] = np.log(price.close / price.open)
    valid_hour = (
        price.source_valid
        & np.isfinite(price[["open", "close", "hour_log_return"]]).all(axis=1)
        & price[["open", "close"]].gt(0).all(axis=1)
    )
    consecutive = price.decision_time.diff().eq(pd.Timedelta(hours=1))
    price["day_price_return"] = np.log(price.close / price.open.shift(23))
    price["day_realized_variation"] = np.sqrt(
        price.hour_log_return.pow(2).rolling(24, min_periods=24).sum()
    )
    price["day_price_valid"] = (
        valid_hour.rolling(24, min_periods=24).sum().eq(24)
        & consecutive.rolling(23, min_periods=23).sum().eq(23)
    )
    daily = price[
        price.decision_time.dt.hour.eq(0)
        & price.decision_time.dt.minute.eq(0)
    ][
        [
            "decision_time",
            "day_price_return",
            "day_realized_variation",
            "day_price_valid",
        ]
    ].copy()
    daily["prior_target"] = daily.decision_time - pd.Timedelta(hours=24)

    oi = pd.read_csv(OI_DIR / "open_interest_5m.csv.gz", compression="gzip")
    oi["ts"] = pd.to_datetime(oi.ts, utc=True, format="mixed")
    oi["sum_open_interest"] = pd.to_numeric(oi.sum_open_interest, errors="coerce")
    oi = oi[
        np.isfinite(oi.sum_open_interest) & oi.sum_open_interest.gt(0)
    ].sort_values("ts")
    current = _asof_oi(daily, oi, "decision_time", "oi_current")
    prior = _asof_oi(daily, oi, "prior_target", "oi_prior")
    daily = daily.reset_index(drop=True)
    for column in ("oi_current_time", "oi_current_oi"):
        daily[column] = current[column].to_numpy()
    for column in ("oi_prior_time", "oi_prior_oi"):
        daily[column] = prior[column].to_numpy()
    current_age = daily.decision_time - daily.oi_current_time
    prior_age = daily.prior_target - daily.oi_prior_time
    daily["oi_return"] = np.log(daily.oi_current_oi / daily.oi_prior_oi)
    daily["price_return"] = daily.day_price_return
    daily["realized_variation"] = daily.day_realized_variation
    daily["displacement"] = (daily.oi_return - daily.price_return).abs()
    daily["source_valid"] = (
        daily.day_price_valid
        & np.isfinite(
            daily[["price_return", "oi_return", "realized_variation", "displacement"]]
        ).all(axis=1)
        & daily.price_return.ne(0)
        & daily.oi_return.ne(0)
        & current_age.between(pd.Timedelta(0), pd.Timedelta(minutes=5))
        & prior_age.between(pd.Timedelta(0), pd.Timedelta(minutes=5))
    )
    daily["realized_variation_rank"] = strict_prior_midrank(
        daily.realized_variation.where(daily.source_valid)
    )
    daily["displacement_rank"] = strict_prior_midrank(
        daily.displacement.where(daily.source_valid)
    )
    return daily


def conditions(frame: pd.DataFrame, control: str) -> tuple[pd.Series, pd.Series]:
    rv_rank = (
        frame.realized_variation_rank.shift(1)
        if control == "one_day_stale_ranks"
        else frame.realized_variation_rank
    )
    displacement_rank = (
        frame.displacement_rank.shift(1)
        if control == "one_day_stale_ranks"
        else frame.displacement_rank
    )
    rv_gate = (
        pd.Series(True, index=frame.index)
        if control == "no_realized_variation_gate"
        else rv_rank.ge(0.65)
    )
    displacement_gate = (
        pd.Series(True, index=frame.index)
        if control == "no_displacement_gate"
        else displacement_rank.ge(0.65)
    )
    opposition = frame.price_return.mul(frame.oi_return).lt(0)
    if control == "same_direction_only":
        opposition = frame.price_return.mul(frame.oi_return).gt(0)
    valid = frame.source_valid & rv_rank.notna() & displacement_rank.notna()
    return valid & rv_gate & displacement_gate & opposition, np.sign(frame.oi_return)


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
                "candidate": "DOPDR-12",
                "control": control,
                "split": split,
                "decision_time": decision,
                "feature_available_time": decision,
                "entry_time": entry,
                "exit_time": exit_time,
                "side": signal_side,
                "price_return": float(frame.at[index, "price_return"]),
                "oi_return": float(frame.at[index, "oi_return"]),
                "realized_variation": float(frame.at[index, "realized_variation"]),
                "realized_variation_rank": float(frame.at[index, "realized_variation_rank"]),
                "displacement": float(frame.at[index, "displacement"]),
                "displacement_rank": float(frame.at[index, "displacement_rank"]),
                "oi_current_time": frame.at[index, "oi_current_time"],
                "oi_prior_time": frame.at[index, "oi_prior_time"],
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def split_stats(clocks: pd.DataFrame, split: str) -> dict[str, int | float]:
    selected = clocks[clocks.split.eq(split)]
    if selected.empty:
        return {
            "events": 0,
            "longs": 0,
            "shorts": 0,
            "minority_side_share": 0.0,
            "max_month_share": 0.0,
        }
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
    frame = features()
    primary = clock(frame)
    controls = {name: clock(frame, name) for name in CONTROLS}
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
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    oi_manifest = OI_DIR / "manifest.json"
    price_manifest = PRICE_DIR / "manifest.json"
    passed = all(checks.values())
    core = {
        "protocol_version": "dopdr_12_source_support_v1",
        "policy_id": "DOPDR-12",
        "preregistration": {
            "path": str(prereg.DEFAULT_OUTPUT),
            "sha256": sha(prereg.DEFAULT_OUTPUT),
            "manifest_hash": registration["manifest_hash"],
        },
        "source_manifests": {
            "oi": {"path": str(oi_manifest), "sha256": sha(oi_manifest)},
            "completed_price": {"path": str(price_manifest), "sha256": sha(price_manifest)},
        },
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
    output = run()
    print(json.dumps({"passed": output["support_passed"], "support": output["support"]}, indent=2))
