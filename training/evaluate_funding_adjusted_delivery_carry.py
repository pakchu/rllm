"""Frozen, sequential strict evaluator for FADC-21.

The freeze path may build signal/control clocks but cannot parse execution OHLC
or funding settlement marks. Stage 1 physically stops before the first 2023
execution row; 2023 can be opened only by a later explicit stage-2 command after
stage 1 passes.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import build_funding_adjusted_delivery_carry_support as support_builder
from training import preregister_funding_adjusted_delivery_carry as prereg


SUPPORT_COMMIT = "d1748eed44d3380e2159f256cd03ce7c9a0d0201"
PREREG_SOURCE_SHA256 = "0ef133cd42dde944508daab4b8ec96d53ebe54eba69acc695259906672de9f33"
SUPPORT_SOURCE_SHA256 = "b0282b5f4f1efb98834c875be9046573cc2589f40c139a460560cf15c48bae74"
PREREG_DOC_SHA256 = "b7809b747aeaad39e18d192e6cde9b2d004bb38b66f18ec752068d2a61ff04f7"
SUPPORT_DOC_SHA256 = "ce9f05ffcda1ae4bba8b4900b00c64ecb6d0cd6acd423932f20a24e627ef5cfe"
PREREG_RESULT_SHA256 = "6f9175d84730f0d493343bc98f33bbbe546d0ff49bf0eb7ab6511b4f8d95da95"
SUPPORT_RESULT_SHA256 = "06f753bfbd42199ccd016d16c94e4903ce30bb164b25a871873a8102b74f5688"
SUPPORT_CONTENT_HASH = "dab72f4495f54b91ccccf052e1a6c9d0dccadd8a90130336516d7dd3b8276785"

PREREG_SOURCE = Path("training/preregister_funding_adjusted_delivery_carry.py")
SUPPORT_SOURCE = Path("training/build_funding_adjusted_delivery_carry_support.py")
PREREG_DOC = Path("docs/funding-adjusted-delivery-carry-preregistration-2026-07-17.md")
SUPPORT_DOC = Path("docs/funding-adjusted-delivery-carry-support-2026-07-17.md")
PREREG_RESULT = Path(support_builder.PREREGISTRATION)
SUPPORT_RESULT = Path(support_builder.OUTPUT)
QUARTERLY_DATA = Path(prereg.QUARTERLY_PANEL)
PERPETUAL_DATA = Path(prereg.PERPETUAL_1M)
FUNDING_DATA = Path(prereg.FUNDING_MARKS)

EVALUATOR_SOURCE = Path("training/evaluate_funding_adjusted_delivery_carry.py")
EVALUATOR_FREEZE = Path("results/funding_adjusted_delivery_carry_evaluator_freeze_2026-07-17.json")
STAGE1_OUTPUT = Path("results/funding_adjusted_delivery_carry_stage1_2021_2022_2026-07-17.json")
STAGE1_DOC = Path("docs/funding-adjusted-delivery-carry-stage1-2021-2022-2026-07-17.md")
STAGE2_OUTPUT = Path("results/funding_adjusted_delivery_carry_stage2_2023_2026-07-17.json")
STAGE2_DOC = Path("docs/funding-adjusted-delivery-carry-stage2-2023-2026-07-17.md")

STAGE1_START = pd.Timestamp("2021-02-03 08:20:00", tz="UTC")
STAGE1_END = pd.Timestamp("2023-01-01", tz="UTC")
STAGE2_START = pd.Timestamp("2023-01-01", tz="UTC")
STAGE2_END = pd.Timestamp("2024-01-01", tz="UTC")
CONTROL_NAMES = (
    "primary",
    "direction_flip",
    "basis_only",
    "constant_long_perp_short_quarter",
    "one_funding_event_delay",
)


@dataclass(frozen=True)
class EvaluationConfig:
    base_cost_notional_per_side: float = 0.0006
    stress_cost_notional_per_side: float = 0.0010
    delay_hours: int = 8
    cluster_permutations: int = 100_000
    cluster_seed: int = 20_260_717


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _seal(core: dict[str, Any]) -> dict[str, Any]:
    return {**core, "manifest_hash": _canonical_hash(core)}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _verify_static_inputs() -> dict[str, Any]:
    expected = (
        (PREREG_SOURCE, PREREG_SOURCE_SHA256),
        (SUPPORT_SOURCE, SUPPORT_SOURCE_SHA256),
        (PREREG_DOC, PREREG_DOC_SHA256),
        (SUPPORT_DOC, SUPPORT_DOC_SHA256),
        (PREREG_RESULT, PREREG_RESULT_SHA256),
        (SUPPORT_RESULT, SUPPORT_RESULT_SHA256),
    )
    for path, digest in expected:
        if _sha256(path) != digest:
            raise ValueError(f"FADC frozen input drifted: {path}")
    for path, digest in prereg.FROZEN_SOURCE_SHA256.items():
        if _sha256(path) != digest:
            raise ValueError(f"FADC market source drifted: {path}")
    support = _load_json(SUPPORT_RESULT)
    body = {
        key: value
        for key, value in support.items()
        if key not in {"as_of", "content_hash"}
    }
    if prereg.canonical_hash(body) != SUPPORT_CONTENT_HASH:
        raise ValueError("FADC support body hash drifted")
    if support.get("content_hash") != SUPPORT_CONTENT_HASH:
        raise ValueError("FADC support content hash drifted")
    if support.get("disposition") != "PASS_SUPPORT_OPEN_2021_2022_PNL":
        raise ValueError("FADC support did not authorize stage 1")
    if support.get("pnl_opened") is not False:
        raise ValueError("FADC support artifact unexpectedly opened PnL")
    return support


def _schedule_frame(events: list[dict[str, Any]], *, control: str) -> pd.DataFrame:
    frame = pd.DataFrame(events)
    if frame.empty:
        raise ValueError(f"FADC {control} schedule is empty")
    for column in (
        "entry_time",
        "exit_time",
        "mandatory_exit_time",
        "delivery_time",
    ):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    frame["perpetual_side"] = pd.to_numeric(frame["perpetual_side"]).astype(int)
    frame["quarterly_side"] = pd.to_numeric(frame["quarterly_side"]).astype(int)
    frame["control"] = control
    return frame


def _signal_source_clean(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["source_complete"].astype(bool)
        & frame["um_ohlc_valid"].astype(bool)
        & frame["perp_signal_complete"].fillna(False).astype(bool)
        & frame["um_close"].gt(0.0)
        & frame["perp_close"].gt(0.0)
        & frame["annualized_basis"].notna()
    )


def _basis_only_schedule(features: pd.DataFrame, cfg: support_builder.Config) -> pd.DataFrame:
    altered = features.copy()
    horizon_days = (
        (altered["mandatory_exit_time"] - altered["signal_time"]).dt.total_seconds()
        / 86_400.0
    )
    altered["annualized_funding"] = 0.0
    altered["carry_gap"] = altered["annualized_basis"]
    altered["edge_to_scheduled_exit"] = altered["carry_gap"].abs() * horizon_days / 365.0
    altered["eligible"] = (
        _signal_source_clean(altered)
        & altered["dte_days"].between(
            cfg.minimum_dte_days,
            cfg.maximum_dte_days,
            inclusive="both",
        )
        & horizon_days.gt(0.0)
        & altered["edge_to_scheduled_exit"].ge(cfg.minimum_edge_fraction)
        & altered["mandatory_exit_time"].lt(pd.Timestamp(cfg.physical_end, tz="UTC"))
    )
    events = support_builder.build_events(altered, cfg)
    return _schedule_frame(support_builder._json_events(events), control="basis_only")


def _constant_schedule(features: pd.DataFrame, cfg: support_builder.Config) -> pd.DataFrame:
    altered = features.copy()
    horizon_days = (
        (altered["mandatory_exit_time"] - altered["signal_time"]).dt.total_seconds()
        / 86_400.0
    )
    altered["annualized_funding"] = altered["annualized_basis"] - 0.01
    altered["carry_gap"] = 0.01
    altered["edge_to_scheduled_exit"] = 0.01
    altered["eligible"] = (
        _signal_source_clean(altered)
        & altered["dte_days"].between(
            cfg.minimum_dte_days,
            cfg.maximum_dte_days,
            inclusive="both",
        )
        & horizon_days.gt(0.0)
        & altered["mandatory_exit_time"].lt(pd.Timestamp(cfg.physical_end, tz="UTC"))
    )
    events = support_builder.build_events(altered, cfg)
    return _schedule_frame(
        support_builder._json_events(events),
        control="constant_long_perp_short_quarter",
    )


def _delay_schedule(primary: pd.DataFrame, *, hours: int) -> pd.DataFrame:
    delayed = primary.copy()
    shift = pd.Timedelta(hours=hours)
    delayed["entry_time"] = delayed["entry_time"] + shift
    delayed["exit_time"] = pd.concat(
        [delayed["exit_time"] + shift, delayed["mandatory_exit_time"]],
        axis=1,
    ).min(axis=1)
    delayed = delayed.loc[delayed["entry_time"].lt(delayed["exit_time"])].copy()
    delayed["control"] = "one_funding_event_delay"
    return delayed.reset_index(drop=True)


def _schedule_hash(frame: pd.DataFrame) -> str:
    rows = [
        {
            "control": str(row.control),
            "entry_time": pd.Timestamp(row.entry_time).isoformat(),
            "exit_time": pd.Timestamp(row.exit_time).isoformat(),
            "mandatory_exit_time": pd.Timestamp(row.mandatory_exit_time).isoformat(),
            "contract_segment": str(row.contract_segment),
            "perpetual_side": int(row.perpetual_side),
            "quarterly_side": int(row.quarterly_side),
        }
        for row in frame.itertuples(index=False)
    ]
    return _canonical_hash(rows)


def _validate_schedule(frame: pd.DataFrame, *, name: str) -> None:
    if frame.empty:
        raise ValueError(f"FADC {name} schedule is empty")
    if not frame["entry_time"].is_monotonic_increasing:
        raise ValueError(f"FADC {name} schedule is unsorted")
    if frame["entry_time"].duplicated().any():
        raise ValueError(f"FADC {name} schedule has duplicate entries")
    if not frame["perpetual_side"].isin([-1, 1]).all():
        raise ValueError(f"FADC {name} has invalid perpetual side")
    if not frame["quarterly_side"].eq(-frame["perpetual_side"]).all():
        raise ValueError(f"FADC {name} is not delta matched")
    if not frame["entry_time"].lt(frame["exit_time"]).all():
        raise ValueError(f"FADC {name} has non-positive hold")
    if not frame["exit_time"].le(frame["mandatory_exit_time"]).all():
        raise ValueError(f"FADC {name} crosses pre-delivery exit")
    if not frame["mandatory_exit_time"].lt(frame["delivery_time"]).all():
        raise ValueError(f"FADC {name} touches delivery")
    if len(frame) > 1:
        current = frame["entry_time"].iloc[1:].reset_index(drop=True)
        previous = frame["exit_time"].iloc[:-1].reset_index(drop=True)
        if not current.ge(previous).all():
            raise ValueError(f"FADC {name} schedule overlaps")


def build_control_schedules(
    cfg: EvaluationConfig = EvaluationConfig(),
) -> dict[str, pd.DataFrame]:
    if cfg != EvaluationConfig():
        raise ValueError("FADC evaluation configuration is frozen")
    support = _verify_static_inputs()
    primary = _schedule_frame(support["events_2021_2023"], control="primary")
    direction_flip = primary.copy()
    direction_flip[["perpetual_side", "quarterly_side"]] *= -1
    direction_flip["control"] = "direction_flip"
    support_cfg = support_builder.Config()
    features, _ = support_builder.build_signal_features(support_cfg)
    schedules = {
        "primary": primary,
        "direction_flip": direction_flip,
        "basis_only": _basis_only_schedule(features, support_cfg),
        "constant_long_perp_short_quarter": _constant_schedule(features, support_cfg),
        "one_funding_event_delay": _delay_schedule(primary, hours=cfg.delay_hours),
    }
    for name in CONTROL_NAMES:
        _validate_schedule(schedules[name], name=name)
    return schedules


def freeze_evaluator() -> dict[str, Any]:
    schedules = build_control_schedules()
    stage_counts = {
        name: {
            "stage1": int(frame["entry_time"].lt(STAGE1_END).sum()),
            "stage2": int(frame["entry_time"].between(STAGE2_START, STAGE2_END, inclusive="left").sum()),
            "all_pre2024": int(len(frame)),
            "schedule_hash": _schedule_hash(frame),
        }
        for name, frame in schedules.items()
    }
    core = {
        "candidate_id": "FADC-21",
        "support_commit": SUPPORT_COMMIT,
        "evaluator_source": str(EVALUATOR_SOURCE),
        "evaluator_source_sha256": _sha256(EVALUATOR_SOURCE),
        "evaluation_config": asdict(EvaluationConfig()),
        "control_names": list(CONTROL_NAMES),
        "control_schedules": stage_counts,
        "opened_windows": [],
        "sealed_windows": ["stage1_2021_2022", "stage2_2023", "2024", "2025", "2026_ytd"],
        "mutable_parameters": [],
        "execution_ohlc_rows_parsed_during_freeze": 0,
        "funding_settlement_marks_loaded_during_freeze": 0,
        "execution_simulation_run_during_freeze": False,
    }
    report = _seal(core)
    EVALUATOR_FREEZE.parent.mkdir(parents=True, exist_ok=True)
    EVALUATOR_FREEZE.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def verify_evaluator_freeze() -> dict[str, Any]:
    if not EVALUATOR_FREEZE.exists():
        raise ValueError("FADC evaluator was not frozen before outcomes")
    payload = _load_json(EVALUATOR_FREEZE)
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if _canonical_hash(core) != payload.get("manifest_hash"):
        raise ValueError("FADC evaluator freeze hash mismatch")
    if payload.get("evaluator_source_sha256") != _sha256(EVALUATOR_SOURCE):
        raise ValueError("FADC evaluator source differs from freeze")
    if payload.get("support_commit") != SUPPORT_COMMIT:
        raise ValueError("FADC support commit differs from freeze")
    if payload.get("opened_windows") != []:
        raise ValueError("FADC evaluator freeze already opened a window")
    if payload.get("mutable_parameters") != []:
        raise ValueError("FADC evaluator freeze permits mutable parameters")
    if payload.get("execution_ohlc_rows_parsed_during_freeze") != 0:
        raise ValueError("FADC freeze parsed execution OHLC")
    if payload.get("funding_settlement_marks_loaded_during_freeze") != 0:
        raise ValueError("FADC freeze loaded funding settlement marks")
    if payload.get("execution_simulation_run_during_freeze") is not False:
        raise ValueError("FADC freeze simulated execution")
    if payload.get("evaluation_config") != asdict(EvaluationConfig()):
        raise ValueError("FADC evaluation configuration drifted")
    schedules = build_control_schedules()
    for name, frame in schedules.items():
        expected = payload["control_schedules"][name]
        if expected["schedule_hash"] != _schedule_hash(frame):
            raise ValueError(f"FADC {name} schedule hash drifted")
    return payload


def _parse_quarterly_before(path: Path, cutoff: pd.Timestamp) -> pd.DataFrame:
    rows: list[tuple[Any, ...]] = []
    boundary_seen = False
    wanted = (
        "open_time",
        "um_open",
        "um_high",
        "um_low",
        "um_close",
        "um_ohlc_valid",
        "source_complete",
        "delivery_time",
        "contract_segment",
    )
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        header = handle.readline().rstrip("\r\n").split(",")
        positions = {column: header.index(column) for column in wanted}
        if positions["open_time"] != 0:
            raise ValueError("FADC quarterly open_time must be first physical column")
        cutoff_date = cutoff.strftime("%Y-%m-%d")
        for line in handle:
            date_text = line.split(",", 1)[0]
            if date_text[:10] >= cutoff_date:
                boundary_seen = True
                break
            fields = line.rstrip("\r\n").split(",")
            rows.append(
                (
                    date_text,
                    float(fields[positions["um_open"]]),
                    float(fields[positions["um_high"]]),
                    float(fields[positions["um_low"]]),
                    float(fields[positions["um_close"]]),
                    fields[positions["um_ohlc_valid"]] == "True",
                    fields[positions["source_complete"]] == "True",
                    fields[positions["delivery_time"]],
                    fields[positions["contract_segment"]],
                )
            )
    if not boundary_seen:
        raise ValueError("FADC quarterly source did not reach sealed boundary")
    frame = pd.DataFrame(rows, columns=wanted)
    frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True)
    frame["delivery_time"] = pd.to_datetime(frame["delivery_time"], utc=True)
    return frame.rename(
        columns={
            "um_open": "quarter_open",
            "um_high": "quarter_high",
            "um_low": "quarter_low",
            "um_close": "quarter_close",
            "um_ohlc_valid": "quarter_ohlc_valid",
            "source_complete": "quarter_source_complete",
        }
    )


def _parse_perpetual_before(path: Path, cutoff: pd.Timestamp) -> pd.DataFrame:
    rows: list[tuple[Any, ...]] = []
    boundary_seen = False
    wanted = ("date", "open", "high", "low", "close")
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        header = handle.readline().rstrip("\r\n").split(",")
        positions = {column: header.index(column) for column in wanted}
        if positions["date"] != 0:
            raise ValueError("FADC perpetual date must be first physical column")
        cutoff_date = cutoff.strftime("%Y-%m-%d")
        for line in handle:
            date_text = line.split(",", 1)[0]
            if date_text[:10] >= cutoff_date:
                boundary_seen = True
                break
            fields = line.rstrip("\r\n").split(",")
            rows.append(
                (
                    date_text,
                    float(fields[positions["open"]]),
                    float(fields[positions["high"]]),
                    float(fields[positions["low"]]),
                    float(fields[positions["close"]]),
                )
            )
    if not boundary_seen:
        raise ValueError("FADC perpetual source did not reach sealed boundary")
    minute = pd.DataFrame(rows, columns=wanted)
    minute["date"] = pd.to_datetime(minute["date"], utc=True)
    grouped = minute.set_index("date").resample("5min", label="left", closed="left")
    five = grouped.agg(
        perp_open=("open", "first"),
        perp_high=("high", "max"),
        perp_low=("low", "min"),
        perp_close=("close", "last"),
        minute_rows=("close", "count"),
    ).reset_index().rename(columns={"date": "open_time"})
    five["perp_source_complete"] = five["minute_rows"].eq(5)
    return five


def _parse_funding_before(path: Path, cutoff: pd.Timestamp) -> pd.DataFrame:
    rows: list[tuple[Any, ...]] = []
    boundary_seen = False
    wanted = (
        "funding_time_ms",
        "funding_time_utc",
        "funding_rate",
        "settlement_mark_price",
    )
    cutoff_ms = int(cutoff.timestamp() * 1_000)
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        header = handle.readline().rstrip("\r\n").split(",")
        positions = {column: header.index(column) for column in wanted}
        if positions["funding_time_ms"] != 0:
            raise ValueError("FADC funding milliseconds must be first physical column")
        for line in handle:
            first = line.split(",", 1)[0]
            if int(first) >= cutoff_ms:
                boundary_seen = True
                break
            fields = line.rstrip("\r\n").split(",")
            rows.append(
                (
                    int(first),
                    fields[positions["funding_time_utc"]],
                    float(fields[positions["funding_rate"]]),
                    float(fields[positions["settlement_mark_price"]]),
                )
            )
    if not boundary_seen:
        raise ValueError("FADC funding source did not reach sealed boundary")
    frame = pd.DataFrame(rows, columns=wanted)
    frame["funding_time"] = pd.to_datetime(frame["funding_time_utc"], utc=True)
    return frame


def load_execution_inputs(cutoff: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    verify_evaluator_freeze()
    quarterly = _parse_quarterly_before(QUARTERLY_DATA, cutoff)
    perpetual = _parse_perpetual_before(PERPETUAL_DATA, cutoff)
    funding = _parse_funding_before(FUNDING_DATA, cutoff)
    market = quarterly.merge(perpetual, on="open_time", how="inner", validate="one_to_one")
    if market.empty or market["open_time"].max() >= cutoff:
        raise ValueError("FADC execution market violated cutoff")
    if market["open_time"].duplicated().any() or not market["open_time"].is_monotonic_increasing:
        raise ValueError("FADC execution market clock is invalid")
    for prefix in ("quarter", "perp"):
        columns = [f"{prefix}_{field}" for field in ("open", "high", "low", "close")]
        clean = (
            market["quarter_source_complete"].astype(bool)
            & market["quarter_ohlc_valid"].astype(bool)
            if prefix == "quarter"
            else market["perp_source_complete"].astype(bool)
        )
        values = market.loc[clean, columns].to_numpy(float)
        if not np.isfinite(values).all() or (values <= 0.0).any():
            raise ValueError(f"FADC {prefix} market has invalid prices")
        opening, high, low, close = values.T
        if (
            (high < np.maximum(opening, close)).any()
            or (low > np.minimum(opening, close)).any()
            or (high < low).any()
        ):
            raise ValueError(f"FADC {prefix} market violates OHLC invariants")
    if funding["funding_time"].max() >= cutoff:
        raise ValueError("FADC funding violated cutoff")
    diagnostics = {
        "cutoff": cutoff.isoformat(),
        "quarterly_rows_parsed": int(len(quarterly)),
        "perpetual_minute_rows_parsed": int(perpetual["minute_rows"].sum()),
        "joint_five_minute_rows": int(len(market)),
        "funding_rows_parsed": int(len(funding)),
        "physical_parse_boundary": f"stop before parsing execution values at {cutoff.isoformat()}",
        "first_market_time": market["open_time"].min().isoformat(),
        "last_market_time": market["open_time"].max().isoformat(),
    }
    return market, funding, diagnostics


def linear_pnl(*, side: int, quantity: float, entry: float, mark: float) -> float:
    if side not in {-1, 1} or min(quantity, entry, mark) <= 0:
        raise ValueError("invalid linear ledger input")
    return float(side) * quantity * (mark - entry)


def _mark_equity(
    *,
    pre_equity: float,
    entry_fee: float,
    funding_cash: float,
    quantity: float,
    quarter_side: int,
    perp_side: int,
    quarter_entry: float,
    perp_entry: float,
    quarter_mark: float,
    perp_mark: float,
) -> float:
    return (
        pre_equity
        - entry_fee
        + funding_cash
        + linear_pnl(side=quarter_side, quantity=quantity, entry=quarter_entry, mark=quarter_mark)
        + linear_pnl(side=perp_side, quantity=quantity, entry=perp_entry, mark=perp_mark)
    )


def weekly_cluster_signflip(
    daily_equity: pd.Series,
    *,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    if daily_equity.empty:
        return {"p_value_one_sided": 1.0, "cluster_count": 0, "method": "empty"}
    daily_pnl = daily_equity.diff().fillna(0.0)
    iso = daily_pnl.index.isocalendar()
    keys = iso.year.astype(str) + "-W" + iso.week.astype(str).str.zfill(2)
    weekly = daily_pnl.groupby(keys).sum()
    weekly = weekly.loc[weekly.abs().gt(1e-15)]
    values = weekly.to_numpy(float)
    if not len(values):
        return {"p_value_one_sided": 1.0, "cluster_count": 0, "method": "empty"}
    observed = float(values.sum())
    if len(values) <= 20:
        exceed = 0
        total = 2 ** len(values)
        for signs in product((-1.0, 1.0), repeat=len(values)):
            if float(np.dot(np.asarray(signs), values)) >= observed - 1e-15:
                exceed += 1
        p_value = exceed / total
        method = "exact"
    else:
        rng = np.random.default_rng(seed)
        exceed = 0
        remaining = permutations
        while remaining:
            batch = min(10_000, remaining)
            signs = rng.choice((-1.0, 1.0), size=(batch, len(values)))
            exceed += int((signs @ values >= observed - 1e-15).sum())
            remaining -= batch
        p_value = (exceed + 1) / (permutations + 1)
        method = "monte_carlo"
    return {
        "p_value_one_sided": float(p_value),
        "cluster_count": int(len(values)),
        "method": method,
        "observed_pnl": observed,
        "weekly_pnl": {str(k): float(v) for k, v in weekly.items()},
    }


def simulate_schedule(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    schedule: pd.DataFrame,
    *,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    cost_rate: float,
    include_funding: bool = True,
    cluster_permutations: int = 100_000,
    cluster_seed: int = 20_260_717,
) -> dict[str, Any]:
    if not 0 <= cost_rate < 0.1 or period_end <= period_start:
        raise ValueError("invalid FADC simulation cost or period")
    indexed = market.set_index("open_time", drop=False)
    selected = schedule.loc[
        schedule["entry_time"].ge(period_start)
        & schedule["entry_time"].lt(period_end)
        & schedule["exit_time"].le(period_end)
    ].copy()
    equity = 1.0
    high_water_mark = 1.0
    strict_mdd = 0.0
    total_cost = 0.0
    total_funding = 0.0
    total_price_pnl = 0.0
    marks: dict[pd.Timestamp, float] = {period_start: 1.0}
    trades: list[dict[str, Any]] = []
    previous_exit: pd.Timestamp | None = None

    for event in selected.itertuples(index=False):
        entry_time = pd.Timestamp(event.entry_time)
        exit_time = pd.Timestamp(event.exit_time)
        if previous_exit is not None and entry_time < previous_exit:
            raise ValueError("FADC simulation schedule overlaps")
        previous_exit = exit_time
        if entry_time not in indexed.index or exit_time not in indexed.index:
            raise ValueError("FADC entry/exit open is unavailable")
        entry_row = indexed.loc[entry_time]
        exit_row = indexed.loc[exit_time]
        held = market.loc[market["open_time"].between(entry_time, exit_time, inclusive="left")]
        if held.empty:
            raise ValueError("FADC held path is empty")
        clean = (
            held["quarter_source_complete"].astype(bool)
            & held["quarter_ohlc_valid"].astype(bool)
            & held["perp_source_complete"].astype(bool)
        )
        if not clean.all():
            raise ValueError("FADC held path contains incomplete OHLC")
        if not held["contract_segment"].astype(str).eq(str(event.contract_segment)).all():
            raise ValueError("FADC held path crosses delivery segment")
        if exit_time > pd.Timestamp(event.mandatory_exit_time):
            raise ValueError("FADC exit crosses mandatory delivery guard")

        quarter_side = int(event.quarterly_side)
        perp_side = int(event.perpetual_side)
        if quarter_side != -perp_side:
            raise ValueError("FADC schedule lost delta matching")
        quarter_entry = float(entry_row["quarter_open"])
        perp_entry = float(entry_row["perp_open"])
        pre_equity = equity
        quantity = pre_equity / (quarter_entry + perp_entry)
        entry_fee = cost_rate * quantity * (quarter_entry + perp_entry)
        total_cost += entry_fee
        equity_after_entry = pre_equity - entry_fee
        marks[entry_time] = equity_after_entry
        strict_mdd = max(
            strict_mdd,
            1.0 - equity_after_entry / max(high_water_mark, 1e-15),
        )

        trade_funding = funding.loc[
            funding["funding_time"].ge(entry_time)
            & funding["funding_time"].lt(exit_time)
        ].copy()
        funding_at = {
            pd.Timestamp(row.funding_time): (
                float(row.funding_rate),
                float(row.settlement_mark_price),
            )
            for row in trade_funding.itertuples(index=False)
        }
        cumulative_funding = 0.0
        for bar in held.itertuples(index=False):
            timestamp = pd.Timestamp(bar.open_time)
            if include_funding and timestamp in funding_at:
                rate, mark = funding_at[timestamp]
                cumulative_funding += -perp_side * quantity * rate * mark

            quarter_favorable = float(bar.quarter_high if quarter_side > 0 else bar.quarter_low)
            perp_favorable = float(bar.perp_high if perp_side > 0 else bar.perp_low)
            favorable_equity = _mark_equity(
                pre_equity=pre_equity,
                entry_fee=entry_fee,
                funding_cash=cumulative_funding,
                quantity=quantity,
                quarter_side=quarter_side,
                perp_side=perp_side,
                quarter_entry=quarter_entry,
                perp_entry=perp_entry,
                quarter_mark=quarter_favorable,
                perp_mark=perp_favorable,
            )
            high_water_mark = max(high_water_mark, favorable_equity)

            quarter_adverse = float(bar.quarter_low if quarter_side > 0 else bar.quarter_high)
            perp_adverse = float(bar.perp_low if perp_side > 0 else bar.perp_high)
            hypothetical_exit_fee = cost_rate * quantity * (quarter_adverse + perp_adverse)
            adverse_equity = _mark_equity(
                pre_equity=pre_equity,
                entry_fee=entry_fee,
                funding_cash=cumulative_funding,
                quantity=quantity,
                quarter_side=quarter_side,
                perp_side=perp_side,
                quarter_entry=quarter_entry,
                perp_entry=perp_entry,
                quarter_mark=quarter_adverse,
                perp_mark=perp_adverse,
            ) - hypothetical_exit_fee
            strict_mdd = max(
                strict_mdd,
                1.0 - adverse_equity / max(high_water_mark, 1e-15),
            )
            close_equity = _mark_equity(
                pre_equity=pre_equity,
                entry_fee=entry_fee,
                funding_cash=cumulative_funding,
                quantity=quantity,
                quarter_side=quarter_side,
                perp_side=perp_side,
                quarter_entry=quarter_entry,
                perp_entry=perp_entry,
                quarter_mark=float(bar.quarter_close),
                perp_mark=float(bar.perp_close),
            )
            marks[timestamp + pd.Timedelta(minutes=5)] = close_equity

        quarter_exit = float(exit_row["quarter_open"])
        perp_exit = float(exit_row["perp_open"])
        price_pnl = (
            linear_pnl(
                side=quarter_side,
                quantity=quantity,
                entry=quarter_entry,
                mark=quarter_exit,
            )
            + linear_pnl(
                side=perp_side,
                quantity=quantity,
                entry=perp_entry,
                mark=perp_exit,
            )
        )
        exit_fee = cost_rate * quantity * (quarter_exit + perp_exit)
        total_cost += exit_fee
        total_price_pnl += price_pnl
        total_funding += cumulative_funding
        equity = pre_equity - entry_fee + price_pnl + cumulative_funding - exit_fee
        marks[exit_time] = equity
        strict_mdd = max(
            strict_mdd,
            1.0 - equity / max(high_water_mark, 1e-15),
        )
        high_water_mark = max(high_water_mark, equity)
        trades.append(
            {
                "control": str(event.control),
                "entry_time": entry_time.isoformat(),
                "exit_time": exit_time.isoformat(),
                "contract_segment": str(event.contract_segment),
                "perpetual_side": perp_side,
                "quarterly_side": quarter_side,
                "quantity_btc": quantity,
                "entry_gross": quantity * (quarter_entry + perp_entry),
                "quarter_entry": quarter_entry,
                "perp_entry": perp_entry,
                "quarter_exit": quarter_exit,
                "perp_exit": perp_exit,
                "price_pnl": price_pnl,
                "funding_cash": cumulative_funding,
                "entry_fee": entry_fee,
                "exit_fee": exit_fee,
                "net_pnl": equity - pre_equity,
                "net_return": equity / pre_equity - 1.0,
                "funding_events": int(len(trade_funding)) if include_funding else 0,
            }
        )

    marks[period_end] = equity
    marked = pd.Series(marks, dtype=float).sort_index()
    marked = marked[~marked.index.duplicated(keep="last")]
    daily_boundaries = pd.date_range(
        period_start.normalize() + pd.Timedelta(days=1),
        period_end.normalize(),
        freq="1D",
        inclusive="both",
    )
    daily_index = pd.DatetimeIndex([period_start]).append(daily_boundaries)
    daily_equity = marked.reindex(marked.index.union(daily_index)).sort_index().ffill()
    daily_equity = daily_equity.reindex(daily_index)
    if daily_equity.empty:
        daily_equity = pd.Series([equity], index=pd.DatetimeIndex([period_end]))
    years = (period_end - period_start).total_seconds() / (365.25 * 86_400.0)
    absolute_return = equity - 1.0
    cagr = equity ** (1.0 / years) - 1.0 if equity > 0 and years > 0 else -1.0
    ratio = cagr / strict_mdd if strict_mdd > 1e-15 else (math.inf if cagr > 0 else 0.0)
    trade_frame = pd.DataFrame(trades)
    branch = {}
    if not trade_frame.empty:
        for side, name in ((1, "perp_long_quarter_short"), (-1, "perp_short_quarter_long")):
            subset = trade_frame.loc[trade_frame["perpetual_side"].eq(side)]
            branch[name] = {
                "trades": int(len(subset)),
                "net_pnl": float(subset["net_pnl"].sum()),
                "compounded_return_pct": float((np.prod(1.0 + subset["net_return"]) - 1.0) * 100.0),
            }
    significance = weekly_cluster_signflip(
        daily_equity,
        permutations=cluster_permutations,
        seed=cluster_seed,
    )
    return {
        "period_start": period_start.isoformat(),
        "period_end": period_end.isoformat(),
        "calendar_years": years,
        "absolute_return_pct": absolute_return * 100.0,
        "cagr_pct": cagr * 100.0,
        "strict_mdd_pct": strict_mdd * 100.0,
        "cagr_to_strict_mdd": ratio,
        "trades": int(len(trades)),
        "ending_equity": equity,
        "transaction_cost_pct_initial": total_cost * 100.0,
        "price_pnl_pct_initial": total_price_pnl * 100.0,
        "funding_cash_pct_initial": total_funding * 100.0,
        "branch": branch,
        "weekly_cluster_signflip": significance,
        "nonzero_daily_pnl_days": int(daily_equity.diff().fillna(0.0).abs().gt(1e-15).sum()),
        "daily_equity": {timestamp.isoformat(): float(value) for timestamp, value in daily_equity.items()},
        "trade_details": trades,
    }


def _window_schedule(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return frame.loc[
        frame["entry_time"].ge(start)
        & frame["entry_time"].lt(end)
        & frame["exit_time"].le(end)
    ].copy()


def _entry_cohort_returns(trades: list[dict[str, Any]], *, frequency: str) -> dict[str, float]:
    if not trades:
        return {}
    frame = pd.DataFrame(trades)
    entry = pd.to_datetime(frame["entry_time"], utc=True)
    if frequency == "half":
        key = entry.dt.year.astype(str) + "H" + np.where(entry.dt.month <= 6, "1", "2")
    else:
        raise ValueError("unsupported cohort frequency")
    frame["cohort"] = key
    return {
        str(name): float((np.prod(1.0 + group["net_return"].to_numpy(float)) - 1.0) * 100.0)
        for name, group in frame.groupby("cohort")
    }


def evaluate_stage1() -> dict[str, Any]:
    freeze = verify_evaluator_freeze()
    schedules = build_control_schedules()
    market, funding, diagnostics = load_execution_inputs(STAGE1_END)
    stage_schedules = {
        name: _window_schedule(frame, STAGE1_START, STAGE1_END)
        for name, frame in schedules.items()
    }
    cfg = EvaluationConfig()
    primary = simulate_schedule(
        market,
        funding,
        stage_schedules["primary"],
        period_start=STAGE1_START,
        period_end=STAGE1_END,
        cost_rate=cfg.base_cost_notional_per_side,
        cluster_permutations=cfg.cluster_permutations,
        cluster_seed=cfg.cluster_seed,
    )
    stress = simulate_schedule(
        market,
        funding,
        stage_schedules["primary"],
        period_start=STAGE1_START,
        period_end=STAGE1_END,
        cost_rate=cfg.stress_cost_notional_per_side,
        cluster_permutations=cfg.cluster_permutations,
        cluster_seed=cfg.cluster_seed,
    )
    controls = {
        name: simulate_schedule(
            market,
            funding,
            stage_schedules[name],
            period_start=STAGE1_START,
            period_end=STAGE1_END,
            cost_rate=cfg.base_cost_notional_per_side,
            include_funding=True,
            cluster_permutations=cfg.cluster_permutations,
            cluster_seed=cfg.cluster_seed,
        )
        for name in CONTROL_NAMES
        if name != "primary"
    }
    zero_funding = simulate_schedule(
        market,
        funding,
        stage_schedules["primary"],
        period_start=STAGE1_START,
        period_end=STAGE1_END,
        cost_rate=cfg.base_cost_notional_per_side,
        include_funding=False,
        cluster_permutations=cfg.cluster_permutations,
        cluster_seed=cfg.cluster_seed,
    )
    year_windows = {
        "2021_partial": (STAGE1_START, pd.Timestamp("2022-01-01", tz="UTC")),
        "2022": (pd.Timestamp("2022-01-01", tz="UTC"), STAGE1_END),
    }
    yearly = {
        name: simulate_schedule(
            market,
            funding,
            _window_schedule(stage_schedules["primary"], start, end),
            period_start=start,
            period_end=end,
            cost_rate=cfg.base_cost_notional_per_side,
            cluster_permutations=cfg.cluster_permutations,
            cluster_seed=cfg.cluster_seed,
        )
        for name, (start, end) in year_windows.items()
    }
    half_returns = _entry_cohort_returns(primary["trade_details"], frequency="half")
    positive_halves = sum(value > 0.0 for value in half_returns.values())
    branches = primary["branch"]
    gates = {
        "combined_absolute_return_positive": primary["absolute_return_pct"] > 0.0,
        "each_calendar_year_absolute_return_positive": all(
            row["absolute_return_pct"] > 0.0 for row in yearly.values()
        ),
        "at_least_three_of_four_halves_positive": (
            len(half_returns) == 4 and positive_halves >= 3
        ),
        "combined_cagr_to_strict_mdd_at_least_2": primary["cagr_to_strict_mdd"] >= 2.0,
        "strict_mdd_at_most_15pct": primary["strict_mdd_pct"] <= 15.0,
        "stress_cost_absolute_return_positive": stress["absolute_return_pct"] > 0.0,
        "both_carry_gap_directions_positive": (
            len(branches) == 2
            and all(row["trades"] > 0 and row["net_pnl"] > 0.0 for row in branches.values())
        ),
        "weekly_cluster_signflip_pvalue_at_most_10pct": (
            primary["weekly_cluster_signflip"]["p_value_one_sided"] <= 0.10
        ),
    }
    passed = all(gates.values())
    core = {
        "candidate_id": "FADC-21",
        "stage": "stage1_2021_2022",
        "evaluator_freeze_manifest_hash": freeze["manifest_hash"],
        "evaluation_source_sha256": _sha256(EVALUATOR_SOURCE),
        "config": asdict(cfg),
        "execution_diagnostics": diagnostics,
        "base_cost": primary,
        "stress_cost": stress,
        "yearly": yearly,
        "entry_cohort_half_returns_pct": half_returns,
        "controls": {**controls, "zero_funding": zero_funding},
        "gates": gates,
        "gate_passed": passed,
        "opened_windows": ["stage1_2021_2022"],
        "sealed_windows": ["stage2_2023", "2024", "2025", "2026_ytd"],
        "disposition": (
            "PASS_STAGE1_OPEN_2023_ONCE"
            if passed
            else "REJECT_STAGE1_KEEP_2023_AND_2024_PLUS_SEALED"
        ),
    }
    report = _seal(core)
    STAGE1_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    STAGE1_OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    STAGE1_DOC.parent.mkdir(parents=True, exist_ok=True)
    STAGE1_DOC.write_text(render_stage_doc(report))
    return report


def _verified_passing_stage1() -> dict[str, Any]:
    if not STAGE1_OUTPUT.exists():
        raise ValueError("FADC stage 1 has not been run")
    report = _load_json(STAGE1_OUTPUT)
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    if _canonical_hash(core) != report.get("manifest_hash"):
        raise ValueError("FADC stage-1 result hash mismatch")
    if report.get("stage") != "stage1_2021_2022":
        raise ValueError("FADC stage-1 result identity drifted")
    if report.get("evaluation_source_sha256") != _sha256(EVALUATOR_SOURCE):
        raise ValueError("FADC stage-1 result used another evaluator")
    if report.get("gate_passed") is not True:
        raise ValueError("FADC stage 1 did not authorize opening 2023")
    if report.get("disposition") != "PASS_STAGE1_OPEN_2023_ONCE":
        raise ValueError("FADC stage-1 disposition did not authorize 2023")
    return report


def evaluate_stage2() -> dict[str, Any]:
    freeze = verify_evaluator_freeze()
    stage1 = _verified_passing_stage1()
    schedules = build_control_schedules()
    market, funding, diagnostics = load_execution_inputs(STAGE2_END)
    stage_schedules = {
        name: _window_schedule(frame, STAGE2_START, STAGE2_END)
        for name, frame in schedules.items()
    }
    cfg = EvaluationConfig()
    primary = simulate_schedule(
        market,
        funding,
        stage_schedules["primary"],
        period_start=STAGE2_START,
        period_end=STAGE2_END,
        cost_rate=cfg.base_cost_notional_per_side,
        cluster_permutations=cfg.cluster_permutations,
        cluster_seed=cfg.cluster_seed,
    )
    stress = simulate_schedule(
        market,
        funding,
        stage_schedules["primary"],
        period_start=STAGE2_START,
        period_end=STAGE2_END,
        cost_rate=cfg.stress_cost_notional_per_side,
        cluster_permutations=cfg.cluster_permutations,
        cluster_seed=cfg.cluster_seed,
    )
    controls = {
        name: simulate_schedule(
            market,
            funding,
            stage_schedules[name],
            period_start=STAGE2_START,
            period_end=STAGE2_END,
            cost_rate=cfg.base_cost_notional_per_side,
            include_funding=True,
            cluster_permutations=cfg.cluster_permutations,
            cluster_seed=cfg.cluster_seed,
        )
        for name in CONTROL_NAMES
        if name != "primary"
    }
    zero_funding = simulate_schedule(
        market,
        funding,
        stage_schedules["primary"],
        period_start=STAGE2_START,
        period_end=STAGE2_END,
        cost_rate=cfg.base_cost_notional_per_side,
        include_funding=False,
        cluster_permutations=cfg.cluster_permutations,
        cluster_seed=cfg.cluster_seed,
    )
    half_returns = _entry_cohort_returns(primary["trade_details"], frequency="half")
    branches = primary["branch"]
    gates = {
        "absolute_return_positive": primary["absolute_return_pct"] > 0.0,
        "cagr_to_strict_mdd_at_least_3": primary["cagr_to_strict_mdd"] >= 3.0,
        "strict_mdd_at_most_15pct": primary["strict_mdd_pct"] <= 15.0,
        "at_least_8_trades": primary["trades"] >= 8,
        "h1_and_h2_absolute_return_positive": (
            set(half_returns) == {"2023H1", "2023H2"}
            and all(value > 0.0 for value in half_returns.values())
        ),
        "stress_cost_absolute_return_positive": stress["absolute_return_pct"] > 0.0,
        "both_carry_gap_directions_positive": (
            len(branches) == 2
            and all(row["trades"] > 0 and row["net_pnl"] > 0.0 for row in branches.values())
        ),
        "weekly_cluster_signflip_pvalue_at_most_10pct": (
            primary["weekly_cluster_signflip"]["p_value_one_sided"] <= 0.10
        ),
    }
    passed = all(gates.values())
    core = {
        "candidate_id": "FADC-21",
        "stage": "stage2_2023",
        "evaluator_freeze_manifest_hash": freeze["manifest_hash"],
        "stage1_manifest_hash": stage1["manifest_hash"],
        "evaluation_source_sha256": _sha256(EVALUATOR_SOURCE),
        "config": asdict(cfg),
        "execution_diagnostics": diagnostics,
        "base_cost": primary,
        "stress_cost": stress,
        "entry_cohort_half_returns_pct": half_returns,
        "controls": {**controls, "zero_funding": zero_funding},
        "gates": gates,
        "gate_passed": passed,
        "opened_windows": ["stage1_2021_2022", "stage2_2023"],
        "sealed_windows": ["2024", "2025", "2026_ytd"],
        "disposition": (
            "PASS_STAGE2_OPEN_ORTHOGONALITY_KEEP_2024_SEALED"
            if passed
            else "REJECT_STAGE2_KEEP_2024_PLUS_SEALED"
        ),
    }
    report = _seal(core)
    STAGE2_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    STAGE2_OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    STAGE2_DOC.parent.mkdir(parents=True, exist_ok=True)
    STAGE2_DOC.write_text(render_stage_doc(report))
    return report


def render_stage_doc(report: dict[str, Any]) -> str:
    base = report["base_cost"]
    stress = report["stress_cost"]
    return f"""# FADC-21 {report['stage']} evaluation — 2026-07-17

| Cost | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| 6 bp/notional-side | {base['absolute_return_pct']:.4f}% | {base['cagr_pct']:.4f}% | {base['strict_mdd_pct']:.4f}% | {base['cagr_to_strict_mdd']:.4f} | {base['trades']} |
| 10 bp/notional-side | {stress['absolute_return_pct']:.4f}% | {stress['cagr_pct']:.4f}% | {stress['strict_mdd_pct']:.4f}% | {stress['cagr_to_strict_mdd']:.4f} | {stress['trades']} |

Disposition: **{report['disposition']}**.

Gates: `{report['gates']}`

The parser stopped before the first 2023 execution value. Therefore 2023 PnL
and all 2024+ outcomes remain sealed unless this stage passes. Absolute return
uses the full wall-clock period, including warm-up and idle time.

Manifest hash: `{report['manifest_hash']}`
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--stage1", action="store_true")
    group.add_argument("--stage2", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        result = freeze_evaluator()
    elif args.stage1:
        result = evaluate_stage1()
    else:
        result = evaluate_stage2()
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
