"""Build outcome-blind FADC-21 support clocks from frozen pre-2024 sources."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.preregister_funding_adjusted_delivery_carry import (
    FROZEN_SOURCE_SHA256,
    FUNDING_MARKS,
    PERPETUAL_1M,
    QUARTERLY_PANEL,
    canonical_hash,
    file_sha256,
)


PREREGISTRATION = "results/funding_adjusted_delivery_carry_preregistration_2026-07-17.json"
PREREGISTRATION_FILE_SHA256 = (
    "6f9175d84730f0d493343bc98f33bbbe546d0ff49bf0eb7ab6511b4f8d95da95"
)
PREREGISTRATION_HASH = "39dc96a719c4fe001eb547f1a3c8be8134a391934997efae049cf06bb487b2f0"
OUTPUT = "results/funding_adjusted_delivery_carry_support_2026-07-17.json"
DOCS_OUTPUT = "docs/funding-adjusted-delivery-carry-support-2026-07-17.md"

QUARTERLY_SIGNAL_COLUMNS = (
    "open_time",
    "available_time",
    "um_close",
    "um_ohlc_valid",
    "source_complete",
    "delivery_time",
    "contract_segment",
)
PERPETUAL_SIGNAL_COLUMNS = ("date", "close")
FUNDING_SIGNAL_COLUMNS = ("funding_time_utc", "funding_rate")


@dataclass(frozen=True)
class Config:
    preregistration: str = PREREGISTRATION
    quarterly_panel: str = QUARTERLY_PANEL
    perpetual_1m: str = PERPETUAL_1M
    funding_marks: str = FUNDING_MARKS
    output: str = OUTPUT
    docs_output: str = DOCS_OUTPUT
    funding_lookback_events: int = 21
    minimum_dte_days: float = 14.0
    maximum_dte_days: float = 80.0
    minimum_edge_fraction: float = 0.003
    normalization_edge_fraction: float = 0.0005
    minimum_hold_hours: int = 24
    mandatory_exit_before_delivery_hours: int = 24
    cooldown_hours: int = 24
    physical_end: str = "2024-01-01"


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def verify_inputs(cfg: Config) -> dict[str, Any]:
    prereg = _load_json(cfg.preregistration)
    if file_sha256(cfg.preregistration) != PREREGISTRATION_FILE_SHA256:
        raise ValueError("FADC preregistration file hash drifted")
    if prereg.get("protocol_hash") != PREREGISTRATION_HASH:
        raise ValueError("FADC preregistration protocol hash drifted")
    if canonical_hash(prereg.get("protocol")) != PREREGISTRATION_HASH:
        raise ValueError("FADC preregistration body drifted")
    if prereg.get("pnl_opened") is not False:
        raise ValueError("FADC preregistration already opened PnL")
    for path, expected in FROZEN_SOURCE_SHA256.items():
        if file_sha256(path) != expected:
            raise ValueError(f"frozen source hash drifted: {path}")
    return prereg


def load_quarterly_signal(path: str) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        usecols=list(QUARTERLY_SIGNAL_COLUMNS),
        parse_dates=["open_time", "available_time", "delivery_time"],
    )
    if tuple(frame.columns) != QUARTERLY_SIGNAL_COLUMNS:
        raise ValueError("quarterly signal columns drifted")
    if frame["open_time"].duplicated().any() or not frame["open_time"].is_monotonic_increasing:
        raise ValueError("quarterly signal clock is duplicated or unsorted")
    if not frame["available_time"].eq(frame["open_time"] + pd.Timedelta(minutes=5)).all():
        raise ValueError("quarterly signal availability is not close+1ms/next five-minute boundary")
    return frame


def aggregate_perpetual_signal(path: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    minute = pd.read_csv(
        path,
        usecols=list(PERPETUAL_SIGNAL_COLUMNS),
        parse_dates=["date"],
    )
    minute["date"] = pd.to_datetime(minute["date"], utc=True)
    if minute["date"].duplicated().any() or not minute["date"].is_monotonic_increasing:
        raise ValueError("perpetual one-minute clock is duplicated or unsorted")
    close = pd.to_numeric(minute["close"], errors="coerce")
    minute = minute.assign(close=close, minute=minute["date"].dt.floor("min"))
    grouped = minute.set_index("minute").resample("5min", label="left", closed="left")
    five = grouped.agg(perp_close=("close", "last"), minute_rows=("close", "count")).reset_index()
    five = five.rename(columns={"minute": "open_time"})
    complete = five["minute_rows"].eq(5) & five["perp_close"].gt(0.0)
    five["perp_signal_complete"] = complete
    diagnostics = {
        "minute_rows": int(len(minute)),
        "five_minute_rows": int(len(five)),
        "complete_five_minute_rows": int(complete.sum()),
        "incomplete_five_minute_rows": int((~complete).sum()),
        "first_minute": minute["date"].min().isoformat(),
        "last_minute": minute["date"].max().isoformat(),
    }
    return five, diagnostics


def load_funding_signal(path: str, *, lookback_events: int) -> pd.DataFrame:
    if lookback_events < 1:
        raise ValueError("funding lookback must be positive")
    funding = pd.read_csv(path, usecols=list(FUNDING_SIGNAL_COLUMNS))
    funding["open_time"] = pd.to_datetime(funding["funding_time_utc"], utc=True)
    funding["funding_rate"] = pd.to_numeric(funding["funding_rate"], errors="coerce")
    funding = funding.sort_values("open_time").reset_index(drop=True)
    if funding["open_time"].duplicated().any():
        raise ValueError("funding signal clock is duplicated")
    funding["funding_mean"] = funding["funding_rate"].rolling(
        lookback_events,
        min_periods=lookback_events,
    ).mean()
    funding["annualized_funding"] = funding["funding_mean"] * 3.0 * 365.0
    return funding[["open_time", "funding_rate", "funding_mean", "annualized_funding"]]


def build_signal_features(cfg: Config) -> tuple[pd.DataFrame, dict[str, Any]]:
    quarterly = load_quarterly_signal(cfg.quarterly_panel)
    perpetual, perpetual_diagnostics = aggregate_perpetual_signal(cfg.perpetual_1m)
    funding = load_funding_signal(
        cfg.funding_marks,
        lookback_events=cfg.funding_lookback_events,
    )
    frame = funding.merge(quarterly, on="open_time", how="inner", validate="one_to_one")
    frame = frame.merge(perpetual, on="open_time", how="left", validate="one_to_one")
    physical_end = pd.Timestamp(cfg.physical_end, tz="UTC")
    if frame["open_time"].ge(physical_end).any():
        raise ValueError("support feature frame opened 2024+")
    frame["signal_time"] = frame["available_time"]
    frame["entry_time"] = frame["open_time"] + pd.Timedelta(minutes=10)
    frame["mandatory_exit_time"] = frame["delivery_time"] - pd.Timedelta(
        hours=cfg.mandatory_exit_before_delivery_hours
    )
    frame["dte_days"] = (
        (frame["delivery_time"] - frame["signal_time"]).dt.total_seconds() / 86_400.0
    )
    horizon_days = (
        (frame["mandatory_exit_time"] - frame["signal_time"]).dt.total_seconds()
        / 86_400.0
    )
    source_clean = (
        frame["source_complete"].astype(bool)
        & frame["um_ohlc_valid"].astype(bool)
        & frame["perp_signal_complete"].fillna(False).astype(bool)
        & frame["um_close"].gt(0.0)
        & frame["perp_close"].gt(0.0)
        & frame["annualized_funding"].notna()
    )
    frame["annualized_basis"] = np.log(frame["um_close"] / frame["perp_close"]) * (
        365.0 / frame["dte_days"]
    )
    frame["carry_gap"] = frame["annualized_basis"] - frame["annualized_funding"]
    frame["edge_to_scheduled_exit"] = frame["carry_gap"].abs() * horizon_days / 365.0
    frame["eligible"] = (
        source_clean
        & frame["dte_days"].between(
            cfg.minimum_dte_days,
            cfg.maximum_dte_days,
            inclusive="both",
        )
        & horizon_days.gt(0.0)
        & frame["edge_to_scheduled_exit"].ge(cfg.minimum_edge_fraction)
        & frame["mandatory_exit_time"].lt(physical_end)
    )
    diagnostics = {
        "quarterly_rows_loaded": int(len(quarterly)),
        "funding_rows_loaded": int(len(funding)),
        "joint_funding_decisions_pre2024": int(len(frame)),
        "clean_decisions": int(source_clean.sum()),
        "eligible_decisions": int(frame["eligible"].sum()),
        "first_decision": frame["open_time"].min().isoformat(),
        "last_decision": frame["open_time"].max().isoformat(),
        "perpetual": perpetual_diagnostics,
    }
    return frame, diagnostics


def build_events(frame: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    events: list[dict[str, Any]] = []
    position: dict[str, Any] | None = None
    cooldown_until: pd.Timestamp | None = None
    minimum_hold = pd.Timedelta(hours=cfg.minimum_hold_hours)
    cooldown = pd.Timedelta(hours=cfg.cooldown_hours)

    def close_position(exit_time: pd.Timestamp, reason: str, row: pd.Series) -> None:
        nonlocal position, cooldown_until
        if position is None:
            raise AssertionError("cannot close a flat support state")
        if exit_time <= position["entry_time"]:
            raise ValueError("support exit is not after entry")
        if exit_time > position["mandatory_exit_time"]:
            raise ValueError("support exit crossed mandatory delivery boundary")
        events.append(
            {
                **position,
                "exit_time": exit_time,
                "exit_reason": reason,
                "exit_signal_time": (
                    row["signal_time"] if reason == "normalization" else pd.NaT
                ),
            }
        )
        cooldown_until = exit_time + cooldown
        position = None

    for _, row in frame.sort_values("open_time").iterrows():
        execution_time = pd.Timestamp(row["entry_time"])
        if position is not None:
            mandatory = pd.Timestamp(position["mandatory_exit_time"])
            if execution_time >= mandatory:
                close_position(mandatory, "mandatory_pre_delivery", row)
            else:
                age = pd.Timestamp(row["signal_time"]) - pd.Timestamp(position["entry_time"])
                current_sign = int(np.sign(float(row["carry_gap"])))
                normalized = float(row["edge_to_scheduled_exit"]) <= cfg.normalization_edge_fraction
                sign_changed = current_sign != int(position["carry_gap_sign"])
                if age >= minimum_hold and (normalized or sign_changed):
                    close_position(execution_time, "normalization", row)

        if position is not None:
            continue
        if cooldown_until is not None and execution_time < cooldown_until:
            continue
        if not bool(row["eligible"]):
            continue
        gap_sign = int(np.sign(float(row["carry_gap"])))
        if gap_sign == 0:
            continue
        mandatory_exit = pd.Timestamp(row["mandatory_exit_time"])
        if execution_time >= mandatory_exit:
            continue
        position = {
            "funding_time": pd.Timestamp(row["open_time"]),
            "signal_time": pd.Timestamp(row["signal_time"]),
            "entry_time": execution_time,
            "mandatory_exit_time": mandatory_exit,
            "delivery_time": pd.Timestamp(row["delivery_time"]),
            "contract_segment": str(row["contract_segment"]),
            "carry_gap_sign": gap_sign,
            "perpetual_side": gap_sign,
            "quarterly_side": -gap_sign,
            "entry_dte_days": float(row["dte_days"]),
            "entry_annualized_basis": float(row["annualized_basis"]),
            "entry_annualized_funding": float(row["annualized_funding"]),
            "entry_carry_gap": float(row["carry_gap"]),
            "entry_edge_to_scheduled_exit": float(row["edge_to_scheduled_exit"]),
        }

    if position is not None:
        close_position(pd.Timestamp(position["mandatory_exit_time"]), "mandatory_pre_delivery", frame.iloc[-1])

    events_frame = pd.DataFrame(events)
    if events_frame.empty:
        return events_frame
    for column in (
        "funding_time",
        "signal_time",
        "entry_time",
        "mandatory_exit_time",
        "delivery_time",
        "exit_time",
        "exit_signal_time",
    ):
        events_frame[column] = pd.to_datetime(events_frame[column], utc=True)
    events_frame["holding_days"] = (
        (events_frame["exit_time"] - events_frame["entry_time"]).dt.total_seconds()
        / 86_400.0
    )
    if not events_frame["entry_time"].lt(events_frame["exit_time"]).all():
        raise ValueError("support emitted non-positive hold")
    if not events_frame["exit_time"].le(events_frame["mandatory_exit_time"]).all():
        raise ValueError("support emitted an exit after the delivery guard")
    if not events_frame["exit_time"].lt(events_frame["delivery_time"]).all():
        raise ValueError("support emitted a delivery-touching exit")
    if (events_frame["entry_time"].iloc[1:].reset_index(drop=True) < events_frame["exit_time"].iloc[:-1].reset_index(drop=True)).any():
        raise ValueError("support events overlap")
    return events_frame


def event_counts(events: pd.DataFrame) -> dict[str, Any]:
    if events.empty:
        return {
            "events": 0,
            "by_year": {},
            "by_half": {},
            "by_month": {},
            "by_direction": {},
            "maximum_month_share": 0.0,
            "active_days": 0.0,
        }
    entry = events["entry_time"]
    half = entry.dt.year.astype(str) + "H" + np.where(entry.dt.month <= 6, "1", "2")
    month = entry.dt.strftime("%Y-%m")
    direction = np.where(
        events["perpetual_side"].gt(0),
        "perp_long_quarter_short",
        "perp_short_quarter_long",
    )
    month_counts = month.value_counts().sort_index()
    direction_counts = pd.Series(direction).value_counts().sort_index()
    return {
        "events": int(len(events)),
        "by_year": {str(k): int(v) for k, v in entry.dt.year.value_counts().sort_index().items()},
        "by_half": {str(k): int(v) for k, v in half.value_counts().sort_index().items()},
        "by_month": {str(k): int(v) for k, v in month_counts.items()},
        "by_direction": {str(k): int(v) for k, v in direction_counts.items()},
        "maximum_month_share": float(month_counts.max() / len(events)),
        "active_days": float(events["holding_days"].sum()),
        "median_holding_days": float(events["holding_days"].median()),
    }


def support_passes(
    train: dict[str, Any],
    holdout_diagnostic: dict[str, Any],
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if train["events"] < 24:
        failures.append("pre2023_entries_below_24")
    for year in ("2021", "2022"):
        if train["by_year"].get(year, 0) < 10:
            failures.append(f"{year}_entries_below_10")
    for half in ("2021H1", "2021H2", "2022H1", "2022H2"):
        if train["by_half"].get(half, 0) < 5:
            failures.append(f"{half}_entries_below_5")
    for name, count in train["by_direction"].items():
        if count / max(train["events"], 1) < 0.25:
            failures.append(f"pre2023_{name}_share_below_25pct")
    if len(train["by_direction"]) != 2:
        failures.append("pre2023_missing_carry_direction")
    if train["maximum_month_share"] > 0.15:
        failures.append("pre2023_month_share_above_15pct")

    if holdout_diagnostic["events"] < 8:
        failures.append("2023_diagnostic_entries_below_8")
    for half in ("2023H1", "2023H2"):
        if holdout_diagnostic["by_half"].get(half, 0) < 3:
            failures.append(f"{half}_diagnostic_entries_below_3")
    for name, count in holdout_diagnostic["by_direction"].items():
        if count / max(holdout_diagnostic["events"], 1) < 0.25:
            failures.append(f"2023_{name}_diagnostic_share_below_25pct")
    if len(holdout_diagnostic["by_direction"]) != 2:
        failures.append("2023_diagnostic_missing_carry_direction")
    if holdout_diagnostic["maximum_month_share"] > 0.25:
        failures.append("2023_diagnostic_month_share_above_25pct")
    return not failures, failures


def _json_events(events: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in events.to_dict("records"):
        converted: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, pd.Timestamp):
                converted[key] = value.isoformat() if not pd.isna(value) else None
            elif isinstance(value, (np.integer, np.floating)):
                converted[key] = value.item()
            else:
                converted[key] = value
        rows.append(converted)
    return rows


def render_doc(report: dict[str, Any]) -> str:
    train = report["support"]["pre2023"]
    holdout = report["support"]["holdout_2023_diagnostic"]
    return f"""# FADC-21 outcome-blind support — 2026-07-17

## Boundary

This unit loaded only current/past completed closes, settled funding rates and
delivery clocks. It did not load quarterly/perpetual future entry or exit
prices, held high/low paths, funding settlement marks, returns, PnL, CAGR or
MDD. No row at or after 2024-01-01 was opened.

## Support result

| Window | Entries | Direction split | Max month share | Active days | Median hold |
|---|---:|---|---:|---:|---:|
| 2021–2022 | {train['events']} | {train['by_direction']} | {train['maximum_month_share']:.2%} | {train['active_days']:.1f} | {train['median_holding_days']:.2f}d |
| 2023 diagnostic | {holdout['events']} | {holdout['by_direction']} | {holdout['maximum_month_share']:.2%} | {holdout['active_days']:.1f} | {holdout['median_holding_days']:.2f}d |

Year/half counts:

- pre-2023 years: `{train['by_year']}`
- pre-2023 halves: `{train['by_half']}`
- 2023 halves: `{holdout['by_half']}`

Disposition: **{report['disposition']}**.

Passing this unit permits only the frozen 2021–2022 stage-1 evaluator. It does
not establish alpha, profitability or orthogonality. 2023 PnL stays sealed
unless stage 1 passes; 2024 remains source-and-outcome sealed.

Content hash: `{report['content_hash']}`
"""


def run(cfg: Config = Config()) -> dict[str, Any]:
    verify_inputs(cfg)
    features, diagnostics = build_signal_features(cfg)
    events = build_events(features, cfg)
    cutoff = pd.Timestamp("2023-01-01", tz="UTC")
    end = pd.Timestamp(cfg.physical_end, tz="UTC")
    pre2023 = events.loc[events["entry_time"].lt(cutoff)].copy()
    holdout = events.loc[events["entry_time"].between(cutoff, end, inclusive="left")].copy()
    pre_counts = event_counts(pre2023)
    holdout_counts = event_counts(holdout)
    passed, failures = support_passes(pre_counts, holdout_counts)
    body = {
        "candidate_id": "FADC-21",
        "protocol_hash": PREREGISTRATION_HASH,
        "config": asdict(cfg),
        "source_sha256": FROZEN_SOURCE_SHA256,
        "support_builder_sha256": file_sha256(__file__),
        "loaded_columns": {
            "quarterly": list(QUARTERLY_SIGNAL_COLUMNS),
            "perpetual": list(PERPETUAL_SIGNAL_COLUMNS),
            "funding": list(FUNDING_SIGNAL_COLUMNS),
        },
        "forbidden_outcome_columns_loaded": [],
        "diagnostics": diagnostics,
        "support": {
            "pre2023": pre_counts,
            "holdout_2023_diagnostic": holdout_counts,
            "passed": passed,
            "failures": failures,
        },
        "events_2021_2023": _json_events(events),
        "pnl_opened": False,
        "oos_2024_plus_opened": False,
        "disposition": (
            "PASS_SUPPORT_OPEN_2021_2022_PNL"
            if passed
            else "REJECT_SUPPORT_KEEP_ALL_PNL_SEALED"
        ),
    }
    content_hash = canonical_hash(body)
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        **body,
        "content_hash": content_hash,
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    Path(cfg.docs_output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.docs_output).write_text(render_doc(report))
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False))
