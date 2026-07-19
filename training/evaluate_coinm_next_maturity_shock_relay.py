"""Sequential strict evaluator for the frozen CMSR-36 alpha.

The evaluator freeze reads only the already-frozen signal ledger.  It opens no
BTCUSDT OHLC or funding row.  Train evaluation physically stops before 2023;
the 2023 test can be opened only after an exact replay of a passing train
report.  No threshold, side, latency, hold, sizing, or cost may be repaired
after an outcome window is opened.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from training import preregister_coinm_next_maturity_shock_relay as prereg


SUPPORT_COMMIT = "d0636dffc57afe972a0893592cf4c317828c6a2b"
STATIC_INPUT_SHA256 = {
    "training/preregister_coinm_next_maturity_shock_relay.py": (
        "91924413f49e189de34884a511948d7fefeea1823797b25d9bd27763b9e40426"
    ),
    "docs/coinm-next-maturity-shock-relay-preregistration-2026-07-19.md": (
        "7d0c2e471fcc3eb877aef40c0b5f6fcf57caab38ede2c410dbd177d6755ff173"
    ),
    "results/coinm_next_maturity_shock_relay_preregistration_2026-07-19.json": (
        "44a1178110509708d7e391eb766fd02e9cd4b32bd1c790f03172e26a48eb98ea"
    ),
    "training/build_coinm_next_maturity_shock_relay_support.py": (
        "e9d69664c93383cf8d7dbb60ef270fae6d252ee771230e4f1e9b62fb69d0ba04"
    ),
    "docs/coinm-next-maturity-shock-relay-support-2026-07-19.md": (
        "cff08b398e676242d2246d63c64b37794584afa84211ef3a4eb1fb3dbfc5dc80"
    ),
    "results/coinm_next_maturity_shock_relay_support_2026-07-19.json": (
        "ca055f162eaee47efc6500ba4f178b0cfe2a0e701f337c6ce306cdc3e5ae368a"
    ),
    "data/coinm_next_maturity_shock_relay_clocks_2020_2023.csv.gz": (
        "e81450d4e76ffd0ce2ae96edf97106f2f4c473da233be0db18dc2530c8da8e87"
    ),
    "tests/test_preregister_coinm_next_maturity_shock_relay.py": (
        "4647fda633f3e1badd0b692f8e764dcd713abb159e7355c6cbb80be44456ca5f"
    ),
    "tests/test_build_coinm_next_maturity_shock_relay_support.py": (
        "f27f5599d4aeea74d03ce89757e80346129670e969a06b481c0f222dd855e548"
    ),
    "tests/test_coinm_next_maturity_shock_relay_support_artifact.py": (
        "4bc80b16cb3fba5a5524feace90a15c543578ff894c256e7548f5190c9ed6aba"
    ),
}
PREREGISTRATION_SHA256 = STATIC_INPUT_SHA256[
    "results/coinm_next_maturity_shock_relay_preregistration_2026-07-19.json"
]
SUPPORT_SHA256 = STATIC_INPUT_SHA256[
    "results/coinm_next_maturity_shock_relay_support_2026-07-19.json"
]
CLOCK_SHA256 = STATIC_INPUT_SHA256[
    "data/coinm_next_maturity_shock_relay_clocks_2020_2023.csv.gz"
]
SUPPORT_MANIFEST_HASH = (
    "8257ccccf25a528c3da411321c9eeb8e5a0fd3b960502e3ded31d1efec3723cb"
)

PREREGISTRATION = Path(prereg.DEFAULT_OUTPUT)
SUPPORT_RESULT = Path(
    "results/coinm_next_maturity_shock_relay_support_2026-07-19.json"
)
CLOCKS = Path("data/coinm_next_maturity_shock_relay_clocks_2020_2023.csv.gz")
MARKET = Path(prereg.MARKET_PATH)
FUNDING = Path(prereg.FUNDING_PATH)
MARKET_MANIFEST = Path(prereg.MARKET_MANIFEST_PATH)
FUNDING_MANIFEST = Path(prereg.FUNDING_MANIFEST_PATH)
EVALUATOR_SOURCE = Path("training/evaluate_coinm_next_maturity_shock_relay.py")
EVALUATOR_FREEZE = Path(
    "results/coinm_next_maturity_shock_relay_evaluator_freeze_2026-07-19.json"
)
TRAIN_OUTPUT = Path(
    "results/coinm_next_maturity_shock_relay_train_2020_2022_2026-07-19.json"
)
TRAIN_DOC = Path(
    "docs/coinm-next-maturity-shock-relay-train-2020-2022-2026-07-19.md"
)
TEST_OUTPUT = Path(
    "results/coinm_next_maturity_shock_relay_test_2023_2026-07-19.json"
)
TEST_DOC = Path("docs/coinm-next-maturity-shock-relay-test-2023-2026-07-19.md")

TimeWindow = tuple[pd.Timestamp, pd.Timestamp]
BAR = cast(pd.Timedelta, pd.Timedelta(minutes=5))
EIGHT_HOURS = cast(pd.Timedelta, pd.Timedelta(hours=8))
YEAR_SECONDS = 365.2425 * 86_400.0
CANDIDATE = "CMSR-36"
HOLD_BARS = 36
MECHANISM_CONTROLS = (
    "no_share_transition",
    "no_lead_shock",
    "front_led_mirror",
)
FALSIFICATION_CONTROLS = (
    "direction_flip",
    "extra_latency_1h",
    "deterministic_random_side",
)
ALL_CLOCK_NAMES = ("primary", *MECHANISM_CONTROLS, *FALSIFICATION_CONTROLS)
SOURCE_CLOCK_NAMES = ("primary", *MECHANISM_CONTROLS)


def _utc(value: str) -> pd.Timestamp:
    return cast(pd.Timestamp, pd.Timestamp(value, tz="UTC"))


TRAIN: TimeWindow = (_utc("2020-08-01"), _utc("2023-01-01"))
TRAIN_SUBPERIODS: dict[str, TimeWindow] = {
    "2020_h2": (_utc("2020-08-01"), _utc("2021-01-01")),
    "2021_h1": (_utc("2021-01-01"), _utc("2021-07-01")),
    "2021_h2": (_utc("2021-07-01"), _utc("2022-01-01")),
    "2022_h1": (_utc("2022-01-01"), _utc("2022-07-01")),
    "2022_h2": (_utc("2022-07-01"), _utc("2023-01-01")),
}
TEST: TimeWindow = (_utc("2023-01-01"), _utc("2024-01-01"))
TEST_SUBPERIODS: dict[str, TimeWindow] = {
    "2023_h1": (_utc("2023-01-01"), _utc("2023-07-01")),
    "2023_h2": (_utc("2023-07-01"), _utc("2024-01-01")),
}


@dataclass(frozen=True)
class EvaluationConfig:
    leverage: float = 0.5
    base_cost_notional_per_side: float = 0.0006
    stress_cost_notional_per_side: float = 0.0010
    cluster_draws: int = 20_000
    cluster_seed: int = 20_260_719
    mdd_denominator_floor: float = 1e-9


def _timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if bool(pd.isna(timestamp)):
        raise ValueError("CMSR-36 timestamp is NaT")
    return cast(pd.Timestamp, timestamp)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _seal(core: dict[str, Any]) -> dict[str, Any]:
    return {**core, "manifest_hash": _canonical_hash(core)}


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"CMSR-36 expected JSON object: {path}")
    return payload


def _write_json_exclusive(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def _verify_evaluation_contract(registration: dict[str, Any]) -> EvaluationConfig:
    policy = registration["policy"]
    statistical = registration["selection_protocol"]["statistical_test"]
    cfg = EvaluationConfig()
    expected = {
        "leverage": policy["leverage"],
        "base_cost_notional_per_side": policy["base_cost_notional_per_side"],
        "stress_cost_notional_per_side": policy[
            "stress_cost_notional_per_side"
        ],
        "cluster_draws": statistical["draws"],
        "cluster_seed": statistical["seed"],
        "mdd_denominator_floor": 1e-9,
    }
    if asdict(cfg) != expected:
        raise ValueError(f"CMSR-36 evaluation contract changed: {asdict(cfg)}")
    if statistical["name"] != "two-sided weekly-cluster sign flip":
        raise ValueError("CMSR-36 statistical test changed")
    if int(policy["hold_bars"]) != HOLD_BARS:
        raise ValueError("CMSR-36 hold changed")
    return cfg


def _verify_static_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    for path, expected in STATIC_INPUT_SHA256.items():
        if _sha256(path) != expected:
            raise ValueError(f"CMSR-36 frozen input changed: {path}")
    registration = _load_json(PREREGISTRATION)
    prereg.validate_manifest(registration, verify_sources=False)
    source = registration["source_contract"]
    if _sha256(MARKET_MANIFEST) != source["market_manifest_sha256"]:
        raise ValueError("CMSR-36 market manifest bytes changed")
    if _sha256(FUNDING_MANIFEST) != source["funding_manifest_sha256"]:
        raise ValueError("CMSR-36 funding manifest bytes changed")
    support = _load_json(SUPPORT_RESULT)
    if support.get("policy_id") != CANDIDATE:
        raise ValueError("CMSR-36 support identity changed")
    if support.get("manifest_hash") != SUPPORT_MANIFEST_HASH:
        raise ValueError("CMSR-36 support manifest changed")
    if support.get("outcomes_opened") is not False:
        raise ValueError("CMSR-36 support opened outcomes")
    if support.get("outcome_sources_opened") != []:
        raise ValueError("CMSR-36 support opened an outcome source")
    if support.get("support_passed") is not True:
        raise ValueError("CMSR-36 support did not pass")
    if support.get("advance_to_train_outcomes") is not True:
        raise ValueError("CMSR-36 support did not authorize train")
    if support.get("clock", {}).get("sha256") != CLOCK_SHA256:
        raise ValueError("CMSR-36 support no longer binds its clock ledger")
    if support.get("source", {}).get("execution_btcusdt_rows_loaded") != 0:
        raise ValueError("CMSR-36 support loaded BTCUSDT execution rows")
    if support.get("source", {}).get("funding_rows_loaded") != 0:
        raise ValueError("CMSR-36 support loaded funding rows")
    _verify_evaluation_contract(registration)
    return registration, support


def _deterministic_side(signal_time: pd.Timestamp) -> int:
    digest = hashlib.sha256(
        f"{CANDIDATE}|{signal_time.isoformat()}".encode("utf-8")
    ).digest()
    return 1 if digest[0] & 1 else -1


def _reserve_schedule(source: pd.DataFrame, clock_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    previous_exit: pd.Timestamp | None = None
    for row in source.sort_values("entry_time").to_dict(orient="records"):
        signal_time = _timestamp(row["signal_time"])
        available = _timestamp(row["feature_available_time"])
        entry_time = _timestamp(row["entry_time"])
        exit_time = _timestamp(row["exit_time"])
        side = int(row["side"])
        if clock_name == "extra_latency_1h":
            entry_time = _timestamp(entry_time + pd.Timedelta(hours=1))
            exit_time = _timestamp(exit_time + pd.Timedelta(hours=1))
        if exit_time >= TEST[1]:
            continue
        if previous_exit is not None and entry_time < previous_exit:
            continue
        previous_exit = exit_time
        if clock_name == "direction_flip":
            side *= -1
        elif clock_name == "deterministic_random_side":
            side = _deterministic_side(signal_time)
        rows.append(
            {
                "candidate_id": CANDIDATE,
                "clock_name": clock_name,
                "signal_time": signal_time,
                "feature_available_time": available,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "side": side,
                "pair": str(row["pair"]),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"CMSR-36 empty schedule: {clock_name}")
    for column in (
        "signal_time",
        "feature_available_time",
        "entry_time",
        "exit_time",
    ):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    frame["side"] = frame["side"].astype("int8")
    return frame.sort_values("entry_time").reset_index(drop=True)


def load_schedules() -> dict[str, pd.DataFrame]:
    _verify_static_inputs()
    frame = pd.read_csv(CLOCKS)
    for column in (
        "signal_time",
        "feature_available_time",
        "entry_time",
        "exit_time",
    ):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    if set(frame["control"]) != set(SOURCE_CLOCK_NAMES):
        raise ValueError("CMSR-36 source clock family changed")
    if not bool(cast(pd.Series, frame["side"]).isin((-1, 1)).all()):
        raise ValueError("CMSR-36 source clock side changed")
    if not bool(
        cast(pd.Series, frame["feature_available_time"])
        .eq(frame["signal_time"] + BAR)
        .all()
    ):
        raise ValueError("CMSR-36 feature availability changed")
    if not bool(
        cast(pd.Series, frame["entry_time"])
        .eq(frame["signal_time"] + 2 * BAR)
        .all()
    ):
        raise ValueError("CMSR-36 entry delay changed")
    if not bool(
        cast(pd.Series, frame["exit_time"])
        .eq(frame["entry_time"] + HOLD_BARS * BAR)
        .all()
    ):
        raise ValueError("CMSR-36 hold changed")
    schedules = {
        name: _reserve_schedule(
            frame.loc[frame["control"].eq(name if name in SOURCE_CLOCK_NAMES else "primary")],
            name,
        )
        for name in ALL_CLOCK_NAMES
    }
    return schedules


def _schedule_hash(frame: pd.DataFrame) -> str:
    rows = [
        {
            "candidate_id": str(row["candidate_id"]),
            "clock_name": str(row["clock_name"]),
            "signal_time": _timestamp(row["signal_time"]).isoformat(),
            "feature_available_time": _timestamp(
                row["feature_available_time"]
            ).isoformat(),
            "entry_time": _timestamp(row["entry_time"]).isoformat(),
            "exit_time": _timestamp(row["exit_time"]).isoformat(),
            "side": int(row["side"]),
            "pair": str(row["pair"]),
        }
        for row in frame.to_dict(orient="records")
    ]
    return _canonical_hash(rows)


def _window_schedule(frame: pd.DataFrame, window: TimeWindow) -> pd.DataFrame:
    start, end = window
    return frame.loc[
        frame["signal_time"].ge(start)
        & frame["feature_available_time"].ge(start)
        & frame["entry_time"].ge(start)
        & frame["exit_time"].lt(end)
    ].copy()


def _schedule_freeze_record(frame: pd.DataFrame, clock_name: str) -> dict[str, Any]:
    entries = [_timestamp(value) for value in frame["entry_time"]]
    exits = [_timestamp(value) for value in frame["exit_time"]]
    signals = [_timestamp(value) for value in frame["signal_time"]]
    available = [_timestamp(value) for value in frame["feature_available_time"]]
    expected_delay = pd.Timedelta(minutes=70 if clock_name == "extra_latency_1h" else 10)
    return {
        "events": int(len(frame)),
        "train_events": int(len(_window_schedule(frame, TRAIN))),
        "test_events": int(len(_window_schedule(frame, TEST))),
        "schedule_hash": _schedule_hash(frame),
        "first_signal": min(signals).isoformat(),
        "first_entry": min(entries).isoformat(),
        "last_exit": max(exits).isoformat(),
        "feature_available_after_signal_5m": all(
            value - signal == BAR for value, signal in zip(available, signals, strict=True)
        ),
        "entry_delay_exact": all(
            entry - signal == expected_delay
            for entry, signal in zip(entries, signals, strict=True)
        ),
        "hold_exact": all(
            exit_time - entry == HOLD_BARS * BAR
            for entry, exit_time in zip(entries, exits, strict=True)
        ),
        "globally_nonoverlapping": all(
            entry >= prior_exit
            for entry, prior_exit in zip(entries[1:], exits[:-1], strict=True)
        ),
        "pre_2024_only": all(exit_time < TEST[1] for exit_time in exits),
        "valid_sides": all(int(side) in {-1, 1} for side in frame["side"]),
    }


def _schedule_records(schedules: dict[str, pd.DataFrame]) -> dict[str, Any]:
    return {
        name: _schedule_freeze_record(schedule, name)
        for name, schedule in schedules.items()
    }


def _market_stage_records(registration: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    manifest = _load_json(MARKET_MANIFEST)
    source = registration["source_contract"]
    if manifest.get("combined_output") != str(MARKET):
        raise ValueError("CMSR-36 market manifest path changed")
    if manifest.get("combined_sha256") != source["market_sha256"]:
        raise ValueError("CMSR-36 market manifest no longer binds the source")
    if manifest.get("protocol", {}).get("outcomes_opened") is not False:
        raise ValueError("CMSR-36 market source manifest opened outcomes")
    months = manifest.get("months")
    if not isinstance(months, list):
        raise ValueError("CMSR-36 market month ledger is missing")

    records: dict[str, list[dict[str, Any]]] = {"train": [], "test": []}
    for row in months:
        if not isinstance(row, dict):
            raise ValueError("CMSR-36 market month record is invalid")
        month = str(row["month"])
        stage = "train" if "2020-08" <= month <= "2022-12" else None
        if "2023-01" <= month <= "2023-12":
            stage = "test"
        if stage is None:
            continue
        records[stage].append(
            {
                "month": month,
                "path": str(row["output"]),
                "sha256": str(row["output_sha256"]),
                "rows": int(row["rows"]),
                "first_date": str(row["first_date"]),
                "last_date": str(row["last_date"]),
            }
        )
    expected_months = {
        "train": list(pd.period_range("2020-08", "2022-12", freq="M").astype(str)),
        "test": list(pd.period_range("2023-01", "2023-12", freq="M").astype(str)),
    }
    for stage, expected in expected_months.items():
        if [row["month"] for row in records[stage]] != expected:
            raise ValueError(f"CMSR-36 {stage} physical market months changed")
    return records


def _funding_stage_records(registration: dict[str, Any]) -> dict[str, dict[str, Any]]:
    manifest = _load_json(FUNDING_MANIFEST)
    source = registration["source_contract"]
    data = manifest.get("data", {})
    if data.get("path") != str(FUNDING):
        raise ValueError("CMSR-36 funding manifest path changed")
    if data.get("sha256") != source["funding_sha256"]:
        raise ValueError("CMSR-36 funding manifest no longer binds the source")
    if manifest.get("outcomes_opened") is not False:
        raise ValueError("CMSR-36 funding source manifest opened outcomes")
    source_start = _utc(str(manifest["config"]["start"]))
    source_end = _utc(str(manifest["config"]["end"]))
    expected_total = int((source_end - source_start) / EIGHT_HOURS)
    if int(data.get("rows", -1)) != expected_total:
        raise ValueError("CMSR-36 funding source row count changed")
    records: dict[str, dict[str, Any]] = {}
    for stage, (start, end) in {"train": TRAIN, "test": TEST}.items():
        start_row = int((start - source_start) / EIGHT_HOURS)
        end_row = int((end - source_start) / EIGHT_HOURS)
        records[stage] = {
            "path": str(FUNDING),
            "source_sha256": str(data["sha256"]),
            "source_start": source_start.isoformat(),
            "source_end": source_end.isoformat(),
            "source_rows": expected_total,
            "start_row_inclusive": start_row,
            "end_row_exclusive": end_row,
            "rows": end_row - start_row,
            "window_start": start.isoformat(),
            "window_end_exclusive": end.isoformat(),
            "boundary_row_read": False,
        }
    return records


def freeze_evaluator(
    output_path: str | Path = EVALUATOR_FREEZE,
) -> dict[str, Any]:
    registration, support = _verify_static_inputs()
    schedules = load_schedules()
    records = _schedule_records(schedules)
    market_records = _market_stage_records(registration)
    funding_records = _funding_stage_records(registration)
    checks = {
        f"{name}/{key}": bool(value)
        for name, record in records.items()
        for key, value in record.items()
        if isinstance(value, bool)
    }
    if not all(checks.values()):
        raise ValueError("CMSR-36 evaluator schedule invariant failed")
    core: dict[str, Any] = {
        "protocol_version": "coinm_next_maturity_shock_relay_evaluator_v1",
        "policy_id": CANDIDATE,
        "as_of_date": "2026-07-19",
        "support_commit": SUPPORT_COMMIT,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "preregistration_manifest_hash": registration["manifest_hash"],
        "support_sha256": SUPPORT_SHA256,
        "support_manifest_hash": support["manifest_hash"],
        "clock_sha256": CLOCK_SHA256,
        "evaluator_source": str(EVALUATOR_SOURCE),
        "evaluator_source_sha256": _sha256(EVALUATOR_SOURCE),
        "evaluation_config": asdict(_verify_evaluation_contract(registration)),
        "static_inputs": STATIC_INPUT_SHA256,
        "schedule_records": records,
        "physical_market_files": market_records,
        "physical_funding_row_windows": funding_records,
        "schedule_invariant_checks": checks,
        "strict_accounting": {
            "global_pre_entry_high_water_mark": True,
            "entry_cost_marked": True,
            "funding_interior_symmetric": True,
            "exact_entry_exit_funding_credits_dropped": True,
            "exact_entry_exit_funding_debits_retained": True,
            "funding_settlement_mark_always_marked": True,
            "held_5m_order": "favorable_then_adverse",
            "virtual_adverse_mark_exit_cost": True,
            "actual_exit_cost_marked": True,
            "full_calendar_cagr": True,
        },
        "opened_windows": [],
        "sealed_windows": ["train_2020_2022", "test_2023", "2024_plus"],
        "execution_ohlc_rows_parsed_during_freeze": 0,
        "funding_rows_parsed_during_freeze": 0,
        "simulation_run_during_freeze": 0,
        "mutable_parameters": [],
    }
    report = _seal(core)
    _write_json_exclusive(output_path, report)
    return report


def verify_evaluator_freeze(
    path: str | Path = EVALUATOR_FREEZE,
) -> dict[str, Any]:
    report = _load_json(path)
    stored_hash = report.get("manifest_hash")
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    if stored_hash != _canonical_hash(core):
        raise ValueError("CMSR-36 evaluator freeze manifest mismatch")
    registration, support = _verify_static_inputs()
    if report.get("preregistration_manifest_hash") != registration["manifest_hash"]:
        raise ValueError("CMSR-36 evaluator freeze preregistration changed")
    if report.get("support_manifest_hash") != support["manifest_hash"]:
        raise ValueError("CMSR-36 evaluator freeze support changed")
    if report.get("evaluator_source_sha256") != _sha256(EVALUATOR_SOURCE):
        raise ValueError("CMSR-36 evaluator source changed after freeze")
    expected_records = _schedule_records(load_schedules())
    if report.get("schedule_records") != expected_records:
        raise ValueError("CMSR-36 schedules changed after freeze")
    if report.get("physical_market_files") != _market_stage_records(registration):
        raise ValueError("CMSR-36 physical market files changed after freeze")
    if report.get("physical_funding_row_windows") != _funding_stage_records(
        registration
    ):
        raise ValueError("CMSR-36 physical funding windows changed after freeze")
    if report.get("opened_windows") != []:
        raise ValueError("CMSR-36 evaluator freeze opened a window")
    if report.get("execution_ohlc_rows_parsed_during_freeze") != 0:
        raise ValueError("CMSR-36 evaluator freeze parsed OHLC")
    if report.get("funding_rows_parsed_during_freeze") != 0:
        raise ValueError("CMSR-36 evaluator freeze parsed funding")
    if report.get("simulation_run_during_freeze") != 0:
        raise ValueError("CMSR-36 evaluator freeze ran a simulation")
    return report


def _parse_market_month_window(
    records: list[dict[str, Any]], start: pd.Timestamp, end: pd.Timestamp
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    opened: list[dict[str, Any]] = []
    for record in records:
        path = Path(record["path"])
        actual_hash = _sha256(path)
        if actual_hash != record["sha256"]:
            raise ValueError(f"CMSR-36 market month changed: {path}")
        frame = pd.read_csv(path, compression="gzip")
        frame = frame.loc[:, ["date", "open", "high", "low", "close"]].copy()
        if len(frame) != int(record["rows"]):
            raise ValueError(f"CMSR-36 market month row count changed: {path}")
        frames.append(frame)
        opened.append(
            {
                "month": record["month"],
                "path": str(path),
                "sha256": actual_hash,
                "rows": int(len(frame)),
            }
        )
    market = pd.concat(frames, ignore_index=True)
    market["date"] = pd.to_datetime(market["date"], utc=True, errors="raise")
    expected = pd.date_range(start, end, freq="5min", inclusive="left")
    if not pd.DatetimeIndex(market["date"]).equals(expected):
        raise ValueError("CMSR-36 market window is not the exact five-minute grid")
    values = market[["open", "high", "low", "close"]].to_numpy(float)
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise ValueError("CMSR-36 market contains invalid prices")
    opening, high, low, close = values.T
    if (
        (high < np.maximum(opening, close)).any()
        or (low > np.minimum(opening, close)).any()
        or (high < low).any()
    ):
        raise ValueError("CMSR-36 market violates OHLC invariants")
    return market, {
        "rows": int(len(market)),
        "first_timestamp": _timestamp(market["date"].iloc[0]).isoformat(),
        "last_timestamp": _timestamp(market["date"].iloc[-1]).isoformat(),
        "physical_month_files": opened,
        "other_stage_files_opened": 0,
    }


def _parse_funding_index_window(
    path: str | Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    source_start: pd.Timestamp,
    start_row_inclusive: int,
    end_row_exclusive: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    wanted = ("funding_time_utc", "symbol", "funding_rate", "settlement_mark_price")
    rows: list[tuple[Any, ...]] = []
    window_hash = hashlib.sha256()
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        header = handle.readline().rstrip("\r\n").split(",")
        positions = {column: header.index(column) for column in wanted}
        selected_lines = itertools.islice(
            handle, start_row_inclusive, end_row_exclusive
        )
        for line in selected_lines:
            fields = line.rstrip("\r\n").split(",")
            rows.append(
                (
                    fields[positions["funding_time_utc"]],
                    fields[positions["symbol"]],
                    float(fields[positions["funding_rate"]]),
                    float(fields[positions["settlement_mark_price"]]),
                )
            )
            window_hash.update(line.encode("utf-8"))
    frame = pd.DataFrame(rows, columns=cast(Any, list(wanted))).rename(
        columns={"funding_time_utc": "funding_time"}
    )
    frame["funding_time"] = pd.to_datetime(
        frame["funding_time"], utc=True, errors="raise"
    )
    expected = pd.date_range(start, end, freq="8h", inclusive="left")
    actual = pd.DatetimeIndex(frame["funding_time"])
    if len(actual) != len(expected) or not actual.is_monotonic_increasing:
        raise ValueError("CMSR-36 funding window does not match the eight-hour grid")
    expected_start_row = int((start - source_start) / EIGHT_HOURS)
    expected_end_row = int((end - source_start) / EIGHT_HOURS)
    if (start_row_inclusive, end_row_exclusive) != (
        expected_start_row,
        expected_end_row,
    ):
        raise ValueError("CMSR-36 funding physical row window changed")
    offsets_ms = (actual - expected).total_seconds().to_numpy(float) * 1_000.0
    if np.abs(offsets_ms).max(initial=0.0) > 60_000.0:
        raise ValueError("CMSR-36 funding timestamp exceeds the frozen offset bound")
    if not bool(cast(pd.Series, frame["symbol"]).eq("BTCUSDT").all()):
        raise ValueError("CMSR-36 funding source changed symbol")
    values = frame[["funding_rate", "settlement_mark_price"]].to_numpy(float)
    if not np.isfinite(values).all() or (frame["settlement_mark_price"] <= 0.0).any():
        raise ValueError("CMSR-36 funding contains invalid values")
    return frame, {
        "rows": int(len(frame)),
        "first_timestamp": _timestamp(frame["funding_time"].iloc[0]).isoformat(),
        "last_timestamp": _timestamp(frame["funding_time"].iloc[-1]).isoformat(),
        "window_line_sha256": window_hash.hexdigest(),
        "start_row_inclusive": start_row_inclusive,
        "end_row_exclusive": end_row_exclusive,
        "source_rows_consumed": end_row_exclusive,
        "boundary_row_read": False,
        "maximum_absolute_grid_offset_ms": float(
            np.abs(offsets_ms).max(initial=0.0)
        ),
    }


def load_execution_window(
    stage: str, window: TimeWindow, freeze: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if stage not in {"train", "test"}:
        raise ValueError(f"CMSR-36 unknown physical stage: {stage}")
    start, end = window
    market_records = freeze["physical_market_files"][stage]
    funding_record = freeze["physical_funding_row_windows"][stage]
    market, market_diagnostics = _parse_market_month_window(
        market_records, start, end
    )
    funding, funding_diagnostics = _parse_funding_index_window(
        funding_record["path"],
        start,
        end,
        source_start=_timestamp(funding_record["source_start"]),
        start_row_inclusive=int(funding_record["start_row_inclusive"]),
        end_row_exclusive=int(funding_record["end_row_exclusive"]),
    )
    return market, funding, {
        "physical_window": [start.isoformat(), end.isoformat()],
        "market": market_diagnostics,
        "funding": funding_diagnostics,
        "combined_market_file_opened": False,
        "full_funding_file_sha256_not_recomputed": True,
        "reason": (
            "market uses hash-bound stage-month files; funding uses an exact "
            "calendar-derived row slice that never requests the boundary row"
        ),
    }


def weekly_cluster_signflip_two_sided(
    trades: pd.DataFrame, *, draws: int, seed: int
) -> dict[str, Any]:
    if trades.empty:
        return {
            "p_value_two_sided": 1.0,
            "cluster_count": 0,
            "draws": draws,
            "seed": seed,
            "observed_mean_net_return": 0.0,
            "weekly_net_return_sums": {},
        }
    entry = pd.to_datetime(trades["entry_time"], utc=True)
    iso = entry.dt.isocalendar()
    keys = iso.year.astype(str) + "-W" + iso.week.astype(str).str.zfill(2)
    weekly = trades["net_return"].groupby(keys).sum()
    values = weekly.to_numpy(float)
    observed = abs(float(trades["net_return"].mean()))
    generator = np.random.default_rng(seed)
    exceed = 0
    remaining = draws
    while remaining:
        batch = min(10_000, remaining)
        signs = generator.choice((-1.0, 1.0), size=(batch, len(values)))
        null = np.abs((signs @ values) / float(len(trades)))
        exceed += int(np.count_nonzero(null >= observed - 1e-15))
        remaining -= batch
    return {
        "p_value_two_sided": float((1 + exceed) / (draws + 1)),
        "cluster_count": int(len(values)),
        "draws": int(draws),
        "seed": int(seed),
        "observed_mean_net_return": float(trades["net_return"].mean()),
        "weekly_net_return_sums": {
            str(label): float(value) for label, value in weekly.items()
        },
    }


def simulate_strict(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    schedule: pd.DataFrame,
    *,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    leverage: float,
    cost_rate_per_side: float,
    mdd_denominator_floor: float = 1e-9,
) -> dict[str, Any]:
    """Run the fixed schedule with conservative path and funding accounting."""

    if leverage <= 0.0 or not 0.0 <= cost_rate_per_side < 0.1:
        raise ValueError("invalid CMSR-36 leverage or cost")
    if period_end <= period_start:
        raise ValueError("invalid CMSR-36 evaluation window")
    selected = _window_schedule(schedule, (period_start, period_end))
    positions = {
        _timestamp(value): index
        for index, value in enumerate(cast(pd.Series, market["date"]))
    }
    funding_times = cast(pd.Series, funding["funding_time"])
    if funding_times.duplicated().any() or not funding_times.is_monotonic_increasing:
        raise ValueError("CMSR-36 funding times are invalid")
    equity = 1.0
    high_water = 1.0
    strict_mdd = 0.0
    records: list[dict[str, Any]] = []
    total_funding_cash = 0.0

    def update_path(value: float) -> None:
        nonlocal high_water, strict_mdd
        if not np.isfinite(value):
            raise ValueError("CMSR-36 strict equity path is non-finite")
        high_water = max(high_water, value)
        strict_mdd = max(
            strict_mdd,
            (high_water - value) / max(high_water, 1e-15),
        )

    previous_exit: pd.Timestamp | None = None
    for event in selected.to_dict(orient="records"):
        entry_time = _timestamp(event["entry_time"])
        exit_time = _timestamp(event["exit_time"])
        if previous_exit is not None and entry_time < previous_exit:
            raise ValueError("CMSR-36 simulation schedule overlaps")
        previous_exit = exit_time
        entry_position = positions.get(entry_time)
        exit_position = positions.get(exit_time)
        if entry_position is None or exit_position is None:
            raise ValueError("CMSR-36 entry or exit open is unavailable")
        if exit_position - entry_position != HOLD_BARS:
            raise ValueError("CMSR-36 hold is not exactly 36 bars")
        side = int(event["side"])
        if side not in {-1, 1}:
            raise ValueError("CMSR-36 side is invalid")
        if equity <= 0.0:
            raise ValueError("CMSR-36 account is insolvent before entry")

        entry_price = float(market.iloc[entry_position]["open"])
        exit_price = float(market.iloc[exit_position]["open"])
        pre_equity = equity
        quantity = leverage * pre_equity / entry_price
        entry_fee = quantity * entry_price * cost_rate_per_side
        cash = pre_equity - entry_fee
        update_path(cash)

        included = funding.loc[
            funding_times.ge(entry_time) & funding_times.le(exit_time)
        ].copy()
        next_funding = 0
        funding_cash = 0.0
        applied_funding_events = 0
        dropped_boundary_credits = 0

        def apply_funding_through(upper: pd.Timestamp) -> None:
            nonlocal cash
            nonlocal funding_cash
            nonlocal next_funding
            nonlocal applied_funding_events
            nonlocal dropped_boundary_credits
            while next_funding < len(included):
                row = included.iloc[next_funding]
                event_time = _timestamp(row["funding_time"])
                if event_time > upper:
                    break
                settlement_mark = float(row["settlement_mark_price"])
                cash_flow = (
                    -side
                    * quantity
                    * settlement_mark
                    * float(row["funding_rate"])
                )
                boundary = event_time in (entry_time, exit_time)
                if boundary and cash_flow > 0.0:
                    dropped_boundary_credits += 1
                else:
                    cash += cash_flow
                    funding_cash += cash_flow
                    applied_funding_events += 1
                marked = cash + side * quantity * (settlement_mark - entry_price)
                update_path(marked)
                virtual_exit_fee = quantity * settlement_mark * cost_rate_per_side
                update_path(marked - virtual_exit_fee)
                next_funding += 1

        for position in range(entry_position, exit_position):
            bar = market.iloc[position]
            bar_time = _timestamp(bar["date"])
            accounting_bar_end = _timestamp(bar_time + BAR - pd.Timedelta(1, unit="ns"))
            apply_funding_through(accounting_bar_end)
            favorable_price = float(bar["high"] if side > 0 else bar["low"])
            favorable_equity = cash + side * quantity * (
                favorable_price - entry_price
            )
            update_path(favorable_equity)
            adverse_price = float(bar["low"] if side > 0 else bar["high"])
            adverse_equity = cash + side * quantity * (adverse_price - entry_price)
            virtual_exit_fee = quantity * adverse_price * cost_rate_per_side
            update_path(adverse_equity - virtual_exit_fee)

        apply_funding_through(exit_time)
        if next_funding != len(included):
            raise ValueError("CMSR-36 funding event remains unapplied at exit")
        gross_underlying_return = side * (exit_price / entry_price - 1.0)
        gross_pnl = side * quantity * (exit_price - entry_price)
        exit_fee = quantity * exit_price * cost_rate_per_side
        equity = cash + gross_pnl - exit_fee
        update_path(equity)
        net_return = equity / pre_equity - 1.0
        total_funding_cash += funding_cash
        records.append(
            {
                "clock_name": str(event["clock_name"]),
                "entry_time": entry_time.isoformat(),
                "exit_time": exit_time.isoformat(),
                "side": side,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "bars_held": HOLD_BARS,
                "pre_entry_equity": pre_equity,
                "fixed_quantity_btc": quantity,
                "gross_underlying_return": gross_underlying_return,
                "gross_trade_return": gross_pnl / pre_equity,
                "entry_fee": entry_fee,
                "exit_fee": exit_fee,
                "funding_cash": funding_cash,
                "funding_events_seen": int(len(included)),
                "funding_events_applied": applied_funding_events,
                "dropped_boundary_funding_credits": dropped_boundary_credits,
                "net_return": net_return,
                "post_exit_equity": equity,
            }
        )

    years = (period_end - period_start).total_seconds() / YEAR_SECONDS
    absolute_return = equity - 1.0
    cagr = equity ** (1.0 / years) - 1.0 if equity > 0.0 else -1.0
    ratio = cagr / max(strict_mdd, mdd_denominator_floor)
    trades = pd.DataFrame(records)
    directions = np.asarray([int(row["side"]) for row in records], dtype=int)
    net_returns = np.asarray(
        [float(row["net_return"]) for row in records], dtype=float
    )
    gross_trade_returns = np.asarray(
        [float(row["gross_trade_return"]) for row in records], dtype=float
    )
    gross_underlying_returns = np.asarray(
        [float(row["gross_underlying_return"]) for row in records], dtype=float
    )
    exposure_seconds = sum(
        (_timestamp(row["exit_time"]) - _timestamp(row["entry_time"])).total_seconds()
        for row in records
    )
    return {
        "period_start": period_start.isoformat(),
        "period_end": period_end.isoformat(),
        "calendar_years": float(years),
        "absolute_return_pct": float(absolute_return * 100.0),
        "cagr_pct": float(cagr * 100.0),
        "strict_mdd_pct": float(strict_mdd * 100.0),
        "cagr_to_strict_mdd": float(ratio),
        "trades": int(len(records)),
        "long_trades": int((directions > 0).sum()),
        "short_trades": int((directions < 0).sum()),
        "win_rate_pct": (
            float((net_returns > 0.0).mean() * 100.0) if len(net_returns) else 0.0
        ),
        "mean_gross_underlying_bp": (
            float(gross_underlying_returns.mean() * 10_000.0)
            if len(gross_underlying_returns)
            else 0.0
        ),
        "mean_gross_trade_bp": (
            float(gross_trade_returns.mean() * 10_000.0)
            if len(gross_trade_returns)
            else 0.0
        ),
        "mean_net_trade_bp": (
            float(net_returns.mean() * 10_000.0) if len(net_returns) else 0.0
        ),
        "funding_cash_pct_initial": float(total_funding_cash * 100.0),
        "exposure_pct": float(
            exposure_seconds / (period_end - period_start).total_seconds() * 100.0
        ),
        "ending_equity": float(equity),
        "cost_rate_per_side": float(cost_rate_per_side),
        "leverage": float(leverage),
        "trade_details": records,
        "weekly_cluster_signflip": weekly_cluster_signflip_two_sided(
            trades,
            draws=20_000,
            seed=20_260_719,
        ),
    }


def _headline(metrics: dict[str, Any]) -> dict[str, Any]:
    significance = metrics["weekly_cluster_signflip"]
    return {
        "absolute_return_pct": metrics["absolute_return_pct"],
        "cagr_pct": metrics["cagr_pct"],
        "strict_mdd_pct": metrics["strict_mdd_pct"],
        "cagr_to_strict_mdd": metrics["cagr_to_strict_mdd"],
        "trades": metrics["trades"],
        "long_trades": metrics["long_trades"],
        "short_trades": metrics["short_trades"],
        "mean_gross_underlying_bp": metrics["mean_gross_underlying_bp"],
        "mean_gross_trade_bp": metrics["mean_gross_trade_bp"],
        "mean_net_trade_bp": metrics["mean_net_trade_bp"],
        "weekly_cluster_signflip_p": significance["p_value_two_sided"],
        "weekly_clusters": significance["cluster_count"],
    }


def _simulate(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    schedule: pd.DataFrame,
    *,
    window: TimeWindow,
    cost_rate: float,
    cfg: EvaluationConfig,
) -> dict[str, Any]:
    start, end = window
    return simulate_strict(
        market,
        funding,
        schedule,
        period_start=start,
        period_end=end,
        leverage=cfg.leverage,
        cost_rate_per_side=cost_rate,
        mdd_denominator_floor=cfg.mdd_denominator_floor,
    )


def _stage_gates(
    stage: str,
    primary: dict[str, Any],
    stress: dict[str, Any],
    subperiod_metrics: dict[str, dict[str, Any]],
    mechanism_margins: dict[str, float],
) -> dict[str, bool]:
    minimum_trades = 90 if stage == "train" else 60
    gates = {
        "absolute_return_positive": primary["absolute_return_pct"] > 0.0,
        "cagr_to_strict_mdd_at_least_3": primary["cagr_to_strict_mdd"] >= 3.0,
        "strict_mdd_at_most_15pct": primary["strict_mdd_pct"] <= 15.0,
        "minimum_trades": primary["trades"] >= minimum_trades,
        "weekly_cluster_signflip_p_at_most_10pct": primary[
            "weekly_cluster_signflip"
        ]["p_value_two_sided"]
        <= 0.10,
        "each_subperiod_absolute_return_positive": all(
            row["absolute_return_pct"] > 0.0 for row in subperiod_metrics.values()
        ),
        "stress_absolute_return_positive": stress["absolute_return_pct"] > 0.0,
        "stress_cagr_to_strict_mdd_at_least_2_5": stress[
            "cagr_to_strict_mdd"
        ]
        >= 2.5,
    }
    if stage == "train":
        gates["mechanism_control_margin_at_least_0_25"] = (
            min(mechanism_margins.values()) >= 0.25
        )
    elif stage != "test":
        raise ValueError(f"CMSR-36 unknown evaluation stage: {stage}")
    return gates


def _evaluate_stage(
    stage: str,
    window: TimeWindow,
    subperiods: dict[str, TimeWindow],
    market: pd.DataFrame,
    funding: pd.DataFrame,
    schedules: dict[str, pd.DataFrame],
    cfg: EvaluationConfig,
) -> dict[str, Any]:
    base_by_clock = {
        name: _simulate(
            market,
            funding,
            schedule,
            window=window,
            cost_rate=cfg.base_cost_notional_per_side,
            cfg=cfg,
        )
        for name, schedule in schedules.items()
    }
    primary = base_by_clock["primary"]
    stress = _simulate(
        market,
        funding,
        schedules["primary"],
        window=window,
        cost_rate=cfg.stress_cost_notional_per_side,
        cfg=cfg,
    )
    subperiod_metrics = {
        name: _simulate(
            market,
            funding,
            schedules["primary"],
            window=subperiod,
            cost_rate=cfg.base_cost_notional_per_side,
            cfg=cfg,
        )
        for name, subperiod in subperiods.items()
    }
    mechanism_ratios = {
        name: base_by_clock[name]["cagr_to_strict_mdd"]
        for name in MECHANISM_CONTROLS
    }
    mechanism_margins = {
        name: primary["cagr_to_strict_mdd"] - ratio
        for name, ratio in mechanism_ratios.items()
    }
    gates = _stage_gates(
        stage, primary, stress, subperiod_metrics, mechanism_margins
    )
    return {
        "primary_metrics": primary,
        "primary_headline": _headline(primary),
        "stress_metrics": stress,
        "stress_headline": _headline(stress),
        "subperiod_metrics": subperiod_metrics,
        "subperiod_headlines": {
            name: _headline(row) for name, row in subperiod_metrics.items()
        },
        "control_metrics": {
            name: row for name, row in base_by_clock.items() if name != "primary"
        },
        "control_headlines": {
            name: _headline(row)
            for name, row in base_by_clock.items()
            if name != "primary"
        },
        "mechanism_control_ratios": mechanism_ratios,
        "mechanism_control_margins": mechanism_margins,
        "minimum_mechanism_control_margin": min(mechanism_margins.values()),
        "gates": gates,
        "qualified": bool(all(gates.values())),
    }


def _build_train_report() -> dict[str, Any]:
    freeze = verify_evaluator_freeze()
    registration, _ = _verify_static_inputs()
    schedules = load_schedules()
    market, funding, diagnostics = load_execution_window("train", TRAIN, freeze)
    result = _evaluate_stage(
        "train",
        TRAIN,
        TRAIN_SUBPERIODS,
        market,
        funding,
        schedules,
        _verify_evaluation_contract(registration),
    )
    core = {
        "protocol_version": "coinm_next_maturity_shock_relay_train_v1",
        "policy_id": CANDIDATE,
        "stage": "train_2020_2022",
        "evaluator_freeze_manifest_hash": freeze["manifest_hash"],
        "evaluator_source_sha256": freeze["evaluator_source_sha256"],
        "execution_diagnostics": diagnostics,
        "result": result,
        "train_passed": result["qualified"],
        "opened_windows": ["train_2020_2022"],
        "sealed_windows": ["test_2023", "2024_plus"],
        "disposition": (
            "ADVANCE_TO_SEALED_2023"
            if result["qualified"]
            else "REJECT_KEEP_2023_SEALED"
        ),
    }
    return _seal(core)


def evaluate_train(
    output_path: str | Path = TRAIN_OUTPUT,
    doc_path: str | Path = TRAIN_DOC,
) -> dict[str, Any]:
    report = _build_train_report()
    _write_json_exclusive(output_path, report)
    output = Path(doc_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(render_stage_doc(report))
    return report


def _verified_passing_train(expected_freeze_hash: str) -> dict[str, Any]:
    stored = _load_json(TRAIN_OUTPUT)
    stored_hash = stored.get("manifest_hash")
    core = {key: value for key, value in stored.items() if key != "manifest_hash"}
    if stored_hash != _canonical_hash(core):
        raise ValueError("CMSR-36 stored train manifest changed")
    if stored.get("evaluator_freeze_manifest_hash") != expected_freeze_hash:
        raise ValueError("CMSR-36 stored train freeze identity changed")
    if stored.get("evaluator_source_sha256") != _sha256(EVALUATOR_SOURCE):
        raise ValueError("CMSR-36 stored train evaluator changed")
    if stored.get("train_passed") is not True:
        raise ValueError("CMSR-36 train failed; 2023 remains sealed")
    replayed = _build_train_report()
    if replayed != stored:
        raise ValueError("CMSR-36 stored train does not exactly reproduce")
    return stored


def evaluate_test(
    output_path: str | Path = TEST_OUTPUT,
    doc_path: str | Path = TEST_DOC,
) -> dict[str, Any]:
    freeze = verify_evaluator_freeze()
    train = _verified_passing_train(freeze["manifest_hash"])
    registration, _ = _verify_static_inputs()
    schedules = load_schedules()
    market, funding, diagnostics = load_execution_window("test", TEST, freeze)
    result = _evaluate_stage(
        "test",
        TEST,
        TEST_SUBPERIODS,
        market,
        funding,
        schedules,
        _verify_evaluation_contract(registration),
    )
    core = {
        "protocol_version": "coinm_next_maturity_shock_relay_test_v1",
        "policy_id": CANDIDATE,
        "stage": "test_2023",
        "evaluator_freeze_manifest_hash": freeze["manifest_hash"],
        "evaluator_source_sha256": freeze["evaluator_source_sha256"],
        "verified_train_manifest_hash": train["manifest_hash"],
        "execution_diagnostics": diagnostics,
        "result": result,
        "test_passed": result["qualified"],
        "opened_windows": ["train_2020_2022", "test_2023"],
        "sealed_windows": ["2024_plus"],
        "disposition": "ADVANCE" if result["qualified"] else "REJECT_NO_REPAIR",
    }
    report = _seal(core)
    _write_json_exclusive(output_path, report)
    output = Path(doc_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(render_stage_doc(report))
    return report


def _metric_row(label: str, headline: dict[str, Any]) -> str:
    return (
        f"| {label} | {headline['absolute_return_pct']:.2f}% | "
        f"{headline['cagr_pct']:.2f}% | {headline['strict_mdd_pct']:.2f}% | "
        f"{headline['cagr_to_strict_mdd']:.2f} | {headline['trades']} | "
        f"{headline['long_trades']}/{headline['short_trades']} | "
        f"{headline['mean_gross_trade_bp']:.2f} | "
        f"{headline['mean_net_trade_bp']:.2f} | "
        f"{headline['weekly_cluster_signflip_p']:.4f} |"
    )


def render_stage_doc(report: dict[str, Any]) -> str:
    result = report["result"]
    lines = [
        f"# CMSR-36 {report['stage']} result — 2026-07-19",
        "",
        "All returns use the full declared calendar, exact funding, 6 bp per side "
        "at base cost, and intratrade strict MDD. Absolute return is always shown.",
        "",
        "| Clock | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades | L/S | Mean gross bp | Mean net bp | p(two-sided) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        _metric_row("primary", result["primary_headline"]),
        _metric_row("primary @ 10bp/side", result["stress_headline"]),
    ]
    for name, headline in result["control_headlines"].items():
        lines.append(_metric_row(name, headline))
    lines.extend(
        [
            "",
            "## Subperiods",
            "",
            "| Period | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades | L/S | Mean gross bp | Mean net bp | p(two-sided) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, headline in result["subperiod_headlines"].items():
        lines.append(_metric_row(name, headline))
    failed = [name for name, passed in result["gates"].items() if not passed]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Qualified: **{result['qualified']}**",
            f"- Failed gates: `{failed}`",
            f"- Minimum mechanism-control margin: `{result['minimum_mechanism_control_margin']:.4f}`",
            f"- Disposition: `{report['disposition']}`",
            "",
            "## Integrity",
            "",
            f"- Evaluator source SHA-256: `{report['evaluator_source_sha256']}`",
            f"- Freeze manifest: `{report['evaluator_freeze_manifest_hash']}`",
            f"- Report manifest: `{report['manifest_hash']}`",
            f"- Physical execution window: `{report['execution_diagnostics']['physical_window']}`",
            "",
        ]
    )
    return "\n".join(lines)


def _print_stage(report: dict[str, Any]) -> None:
    result = report["result"]
    print(
        json.dumps(
            {
                "stage": report["stage"],
                "passed": result["qualified"],
                "headline": result["primary_headline"],
                "stress": result["stress_headline"],
                "failed_gates": [
                    name for name, passed in result["gates"].items() if not passed
                ],
                "disposition": report["disposition"],
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--train", action="store_true")
    group.add_argument("--test", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        print(json.dumps(freeze_evaluator(), indent=2))
    elif args.train:
        _print_stage(evaluate_train())
    else:
        _print_stage(evaluate_test())


if __name__ == "__main__":
    main()
