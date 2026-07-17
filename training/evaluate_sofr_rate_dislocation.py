"""Sequential strict evaluator for frozen SFRD-1.

Evaluator freeze opens no outcome source. Stage 1 physically parses only the
2021-2022 execution/funding window. The 2023 window can be opened only by a
hash-bound Stage-1 result passing every frozen gate. 2024+ always remains
sealed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence, cast

import pandas as pd

from training import evaluate_fiat_quote_participation_rotation as strict_engine
from training import preregister_sofr_rate_dislocation as prereg
from training import sofr_rate_dislocation_clock as source_clock


SUPPORT_COMMIT = "216f3e95b8e7c9b10441560cf93deac54b374b03"
STATIC_INPUT_SHA256 = {
    "training/preregister_sofr_rate_dislocation.py": (
        "1d315d9d9d4f62a3c6441068dd3bdd03577adc6f1619955ebe9e6bf985e367e1"
    ),
    "training/sofr_rate_dislocation_clock.py": (
        "df44bc26f3d6e523df8ecd8619db97b6deb6078d4af14b6c89baabaef171f36d"
    ),
    "training/build_sofr_rate_dislocation_support.py": (
        "99adc74e13bc6173e633906e15998cec831f3d8d022d1ce8dfaba455ae46e712"
    ),
    "training/evaluate_fiat_quote_participation_rotation.py": (
        "e309f5217f033d57d2eadfec936843e736ce287f5c47f957c0ac6f0c71879c23"
    ),
    "docs/sofr-rate-dislocation-preregistration-2026-07-17.md": (
        "81cb715a58100c1c4090bab3a06806787294bed52971ff0857dc7976136f7f85"
    ),
    "docs/sofr-rate-dislocation-support-2026-07-17.md": (
        "5e31f2709fd6323c87d1a2dfc763bbc535b42f30e4bf51bd5ba2417c648e12d3"
    ),
    "results/sofr_rate_dislocation_preregistration_2026-07-17.json": (
        "cbb80c25e4b4c627b95d1992ce4ad00043acfff586e0fe3fcd086af5b4e80b06"
    ),
    "results/sofr_rate_dislocation_preregistered_clock_2026-07-17.csv.gz": (
        "391c42dd2b0d5b87ffcd73058dd9fa0c4d18fd2f535597effff5a4c8edea2e69"
    ),
    "results/sofr_rate_dislocation_support_2026-07-17.json": (
        "477286e74987c2371d92bc91eeef6546ba0d3796e6f03239b75e880ec05a29e8"
    ),
    "data/new_york_fed_sofr_distribution_2018_2023/new_york_fed_sofr_distribution_2018-04-02_2023-12-28.csv.gz": (
        "4993eda2b659e346b4d7b6e3aa0e2ff31cacf868f0e1fe2e1a5a76a03d1b5852"
    ),
    "data/new_york_fed_sofr_distribution_2018_2023/build_manifest.json": (
        "873afb5234fd013e3bc454a83713abf34d9f4a4bffc9895683add7891c636598"
    ),
    "data/binance_um_kline_reference_btc_2020_2023/build_manifest.json": (
        "c04fbbd299cc748a6745c0ef030787da4d560833c744c81c98dd8840efc7913e"
    ),
    "results/binance_um_btcusdt_funding_marks_2020_2023_manifest_2026-07-17.json": (
        "a0b2d27e1aa8cf2d9ab8cb659b598ee0a6d7bd25401c9e10ae92d1a74415845b"
    ),
}

PREREGISTRATION = Path(prereg.DEFAULT_OUTPUT)
SUPPORT_RESULT = Path("results/sofr_rate_dislocation_support_2026-07-17.json")
CLOCK_LEDGER = Path(source_clock.DEFAULT_OUTPUT)
MARKET = Path(prereg.MARKET_PATH)
FUNDING = Path(prereg.FUNDING_PATH)
MARKET_MANIFEST = Path(prereg.MARKET_MANIFEST_PATH)
FUNDING_MANIFEST = Path(prereg.FUNDING_MANIFEST_PATH)
EVALUATOR_SOURCE = Path("training/evaluate_sofr_rate_dislocation.py")
EVALUATOR_FREEZE = Path(
    "results/sofr_rate_dislocation_evaluator_freeze_2026-07-17.json"
)
STAGE1_OUTPUT = Path(
    "results/sofr_rate_dislocation_stage1_2021_2022_2026-07-17.json"
)
STAGE1_DOC = Path("docs/sofr-rate-dislocation-stage1-2021-2022-2026-07-17.md")
STAGE2_OUTPUT = Path("results/sofr_rate_dislocation_stage2_2023_2026-07-17.json")
STAGE2_DOC = Path("docs/sofr-rate-dislocation-stage2-2023-2026-07-17.md")

TimeWindow = tuple[pd.Timestamp, pd.Timestamp]


def _utc_timestamp(value: str) -> pd.Timestamp:
    return cast(pd.Timestamp, pd.Timestamp(value, tz="UTC"))


def _timestamp(value: Any) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result is pd.NaT:
        raise ValueError("SFRD-1 timestamp is NaT")
    return cast(pd.Timestamp, result)


STAGE1: TimeWindow = (_utc_timestamp("2021-01-01"), _utc_timestamp("2023-01-01"))
STAGE1_SUBPERIODS: dict[str, TimeWindow] = {
    "2021": (
        _utc_timestamp("2021-01-01"),
        _utc_timestamp("2022-01-01"),
    ),
    "2022": (
        _utc_timestamp("2022-01-01"),
        _utc_timestamp("2023-01-01"),
    ),
}
STAGE2: TimeWindow = (_utc_timestamp("2023-01-01"), _utc_timestamp("2024-01-01"))
STAGE2_SUBPERIODS: dict[str, TimeWindow] = {
    "2023_h1": (
        _utc_timestamp("2023-01-01"),
        _utc_timestamp("2023-07-01"),
    ),
    "2023_h2": (
        _utc_timestamp("2023-07-01"),
        _utc_timestamp("2024-01-01"),
    ),
}
ALL_CLOCK_NAMES = (
    "primary",
    "direction_flip",
    "level_tail",
    "five_observation_change_tail",
    "month_turn",
    "one_observation_delay",
    "random_side",
)
STAGE1_GATE_NAMES = frozenset(
    {
        "absolute_return_positive",
        "cagr_to_strict_mdd_at_least_3",
        "strict_mdd_at_most_15pct",
        "weekly_cluster_signflip_p_at_most_10pct",
        "minimum_trades",
        "mean_gross_underlying_at_least_35bp",
        "stress_cost_absolute_return_positive",
        "each_subperiod_absolute_return_positive",
        "each_subperiod_minimum_trades",
        "minimum_each_side_trades",
        "single_entry_month_share_at_most_15pct",
        "primary_ratio_strictly_beats_each_mechanism_control",
        "one_observation_delay_and_random_do_not_fully_qualify",
    }
)
BASE_QUALIFICATION_GATE_NAMES = frozenset(
    STAGE1_GATE_NAMES
    - {
        "primary_ratio_strictly_beats_each_mechanism_control",
        "one_observation_delay_and_random_do_not_fully_qualify",
    }
)
STAGE2_GATE_NAMES = frozenset(
    BASE_QUALIFICATION_GATE_NAMES
    | {
        "minimum_train_2023_ratio_strictly_beats_each_mechanism_control",
        "one_observation_delay_and_random_do_not_fully_qualify_both_splits",
    }
)

EvaluationConfig = strict_engine.EvaluationConfig
simulate_schedule = strict_engine.simulate_schedule
weekly_cluster_signflip = strict_engine.weekly_cluster_signflip
_parse_market_window = strict_engine._parse_market_window
_parse_funding_window = strict_engine._parse_funding_window


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _verify_evaluation_contract(registration: dict[str, Any]) -> None:
    policy = registration["policy"]
    cfg = asdict(EvaluationConfig())
    for key in (
        "leverage",
        "base_cost_notional_per_side",
        "stress_cost_notional_per_side",
    ):
        if cfg[key] != policy[key]:
            raise ValueError(f"SFRD-1 evaluator {key} differs from preregistration")
    statistical = registration["selection_protocol"]["statistical_test_contract"]
    if cfg["cluster_draws"] != statistical["draws"]:
        raise ValueError("SFRD-1 sign-flip draw count differs from preregistration")
    if cfg["cluster_seed"] != statistical["seed"]:
        raise ValueError("SFRD-1 sign-flip seed differs from preregistration")
    if cfg["mdd_denominator_floor"] != 1e-9:
        raise ValueError("SFRD-1 MDD denominator floor changed")
    gates = registration["selection_protocol"]["gates"]
    exact_gate_values = {
        "train_and_2023_absolute_return_positive": True,
        "train_and_2023_cagr_to_strict_mdd_min": 3.0,
        "train_and_2023_strict_mdd_pct_max": 15.0,
        "train_and_2023_weekly_cluster_signflip_p_max": 0.10,
        "train_and_2023_mean_gross_underlying_bp_min": 35.0,
        "train_and_2023_ten_bp_stress_absolute_return_positive": True,
        "2021_2022_and_2023_halves_absolute_return_positive": True,
        "stage1_primary_ratio_strictly_beats_mechanism_controls": True,
        "minimum_train_2023_primary_ratio_strictly_beats_controls": True,
        "one_observation_delay_or_random_full_qualification_rejects": True,
    }
    for key, expected in exact_gate_values.items():
        if gates.get(key) != expected:
            raise ValueError(f"SFRD-1 evaluator gate contract changed: {key}")


def _verify_static_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    for path, expected in STATIC_INPUT_SHA256.items():
        if _sha256(path) != expected:
            raise ValueError(f"SFRD-1 frozen input changed: {path}")
    registration = _load_json(PREREGISTRATION)
    prereg.validate_manifest(registration, verify_sources=False)
    _verify_evaluation_contract(registration)
    support = _load_json(SUPPORT_RESULT)
    if support.get("policy_id") != "SFRD-1":
        raise ValueError("SFRD-1 support identity changed")
    if support.get("outcomes_opened") is not False:
        raise ValueError("SFRD-1 support opened an outcome")
    if support.get("outcome_sources_opened") != []:
        raise ValueError("SFRD-1 support opened an outcome source")
    if support.get("support_replay_passed") is not True:
        raise ValueError("SFRD-1 support replay did not pass")
    if support.get("advance_to_stage1_outcomes") is not True:
        raise ValueError("SFRD-1 support did not authorize Stage 1")
    if support.get("preregistration_manifest_hash") != registration["manifest_hash"]:
        raise ValueError("SFRD-1 support is not bound to the preregistration")
    if support.get("clock_ledger_sha256") != STATIC_INPUT_SHA256[str(CLOCK_LEDGER)]:
        raise ValueError("SFRD-1 support is not bound to the clock ledger")
    return registration, support


def _schedule_frame(
    clock_name: str,
    rows: list[dict[str, Any]],
) -> pd.DataFrame:
    frame = pd.DataFrame(
        rows,
        columns=pd.Index(
            [
                "clock_name",
                "effective_date",
                "signal_day",
                "entry_time",
                "exit_time",
                "side",
            ]
        ),
    )
    if frame.empty:
        return frame
    frame["clock_name"] = clock_name
    frame["effective_date"] = pd.to_datetime(frame["effective_date"], utc=True)
    for column in ("signal_day", "entry_time", "exit_time"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    frame["side"] = frame["side"].astype("int8")
    return frame.sort_values("entry_time").reset_index(drop=True)


def _primary_schedule() -> pd.DataFrame:
    events = source_clock.read_event_ledger(CLOCK_LEDGER)
    rows = [
        {
            "clock_name": "primary",
            "effective_date": event.effective_date,
            "signal_day": event.sofr_available_at_utc,
            "entry_time": event.entry_time,
            "exit_time": event.exit_time,
            "side": event.side,
        }
        for event in events
    ]
    return _schedule_frame("primary", rows)


def _rank_states(values: Sequence[int | None], lookback: int = 120) -> list[int]:
    states = [0] * len(values)
    for index, current in enumerate(values):
        if current is None or index < lookback:
            continue
        prior = values[index - lookback : index]
        if len(prior) != lookback or any(value is None for value in prior):
            continue
        exact_prior = [int(value) for value in prior if value is not None]
        numerator = 2 * sum(value < current for value in exact_prior) + sum(
            value == current for value in exact_prior
        )
        if numerator >= 204:
            states[index] = 1
        elif numerator <= 36:
            states[index] = -1
    return states


def _reserve_state_transitions(
    name: str,
    rows: list[source_clock.SourceRow],
    states: list[int],
) -> pd.DataFrame:
    candidates: list[tuple[int, int]] = []
    for index, state in enumerate(states):
        previous = states[index - 1] if index else 0
        if state != 0 and state != previous:
            candidates.append((index, -state))
    return _reserve_candidates(name, rows, candidates)


def _reserve_candidates(
    name: str,
    rows: list[source_clock.SourceRow],
    candidates: list[tuple[int, int]],
) -> pd.DataFrame:
    reserved_until = datetime.min.replace(tzinfo=timezone.utc)
    output: list[dict[str, Any]] = []
    for index, side in candidates:
        entry = rows[index].available_at + timedelta(minutes=5)
        exit_time = entry + timedelta(days=5)
        if entry < reserved_until:
            continue
        output.append(
            {
                "clock_name": name,
                "effective_date": rows[index].effective_date.isoformat(),
                "signal_day": rows[index].available_at.isoformat(),
                "entry_time": entry.isoformat(),
                "exit_time": exit_time.isoformat(),
                "side": side,
            }
        )
        reserved_until = exit_time
    return _schedule_frame(name, output)


def _control_schedules(primary: pd.DataFrame) -> dict[str, pd.DataFrame]:
    rows = source_clock.read_source(prereg.SOFR_PATH)
    levels = [row.rate_bp for row in rows]
    level_tail = _reserve_state_transitions("level_tail", rows, _rank_states(levels))

    five_change: list[int | None] = [None] * len(rows)
    for index in range(5, len(rows)):
        five_change[index] = rows[index].rate_bp - rows[index - 5].rate_bp
    five_tail = _reserve_state_transitions(
        "five_observation_change_tail",
        rows,
        _rank_states(five_change),
    )

    month_members: dict[tuple[int, int], list[int]] = {}
    for index, row in enumerate(rows):
        key = (row.effective_date.year, row.effective_date.month)
        month_members.setdefault(key, []).append(index)
    month_indices = {
        index
        for members in month_members.values()
        for index in (members[0], members[-1])
    }
    month_candidates = []
    for index in sorted(month_indices):
        if index == 0:
            continue
        change = rows[index].rate_bp - rows[index - 1].rate_bp
        if change != 0:
            month_candidates.append((index, -1 if change > 0 else 1))
    month_turn = _reserve_candidates("month_turn", rows, month_candidates)

    direction = primary.copy()
    direction["clock_name"] = "direction_flip"
    direction["side"] = -direction["side"]

    random_side = primary.copy()
    random_side["clock_name"] = "random_side"
    random_side["side"] = [
        1
        if hashlib.sha256(
            ("SFRD-1-random-side-20260717|" + timestamp.isoformat()).encode()
        ).digest()[0]
        < 128
        else -1
        for timestamp in random_side["entry_time"]
    ]

    by_effective = {row.effective_date.isoformat(): index for index, row in enumerate(rows)}
    delayed_candidates: list[tuple[int, int]] = []
    for event in source_clock.read_event_ledger(CLOCK_LEDGER):
        index = by_effective[event.effective_date] + 1
        if index < len(rows):
            delayed_candidates.append((index, event.side))
    delayed = _reserve_candidates("one_observation_delay", rows, delayed_candidates)
    return {
        "direction_flip": direction,
        "level_tail": level_tail,
        "five_observation_change_tail": five_tail,
        "month_turn": month_turn,
        "one_observation_delay": delayed,
        "random_side": random_side,
    }


def _schedule_hash(frame: pd.DataFrame) -> str:
    rows = [
        {
            "clock_name": str(row["clock_name"]),
            "effective_date": _timestamp(row["effective_date"]).isoformat(),
            "signal_day": _timestamp(row["signal_day"]).isoformat(),
            "entry_time": _timestamp(row["entry_time"]).isoformat(),
            "exit_time": _timestamp(row["exit_time"]).isoformat(),
            "side": int(row["side"]),
        }
        for row in frame.to_dict(orient="records")
    ]
    return _canonical_hash(rows)


def _entry_distribution(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "trades": 0,
            "long_trades": 0,
            "short_trades": 0,
            "entry_month_counts": {},
            "max_single_entry_month_count": 0,
            "max_single_entry_month_share": 0.0,
        }
    sides = frame["side"].astype(int)
    months = (
        pd.to_datetime(frame["entry_time"], utc=True, errors="raise")
        .dt.strftime("%Y-%m")
        .value_counts()
        .sort_index()
    )
    maximum = int(months.max())
    return {
        "trades": int(len(frame)),
        "long_trades": int(sides.eq(1).sum()),
        "short_trades": int(sides.eq(-1).sum()),
        "entry_month_counts": {
            str(month): int(count) for month, count in months.items()
        },
        "max_single_entry_month_count": maximum,
        "max_single_entry_month_share": float(maximum / len(frame)),
    }


def _verify_primary_preflight_counts(
    registration: dict[str, Any], primary: pd.DataFrame
) -> None:
    expected = registration["support_verification"]["expected_preflight_counts"]
    train = _entry_distribution(_window_schedule(primary, STAGE1))
    stage2 = _entry_distribution(_window_schedule(primary, STAGE2))
    actual = {
        "train": train["trades"],
        "2021": len(_window_schedule(primary, STAGE1_SUBPERIODS["2021"])),
        "2022": len(_window_schedule(primary, STAGE1_SUBPERIODS["2022"])),
        "2023": stage2["trades"],
        "2023_h1": len(_window_schedule(primary, STAGE2_SUBPERIODS["2023_h1"])),
        "2023_h2": len(_window_schedule(primary, STAGE2_SUBPERIODS["2023_h2"])),
        "train_long": train["long_trades"],
        "train_short": train["short_trades"],
        "2023_long": stage2["long_trades"],
        "2023_short": stage2["short_trades"],
        "train_max_single_month_count": train["max_single_entry_month_count"],
        "train_max_single_month_share": train["max_single_entry_month_share"],
        "2023_max_single_month_count": stage2["max_single_entry_month_count"],
        "2023_max_single_month_share": stage2["max_single_entry_month_share"],
    }
    if actual != expected:
        raise ValueError("SFRD-1 primary preflight counts changed")


def load_schedules() -> dict[str, pd.DataFrame]:
    registration, _ = _verify_static_inputs()
    primary = _primary_schedule()
    schedules = {"primary": primary, **_control_schedules(primary)}
    if set(schedules) != set(ALL_CLOCK_NAMES):
        raise ValueError("SFRD-1 clock family differs from preregistration")
    if set(registration["falsification_controls"]) != set(ALL_CLOCK_NAMES) - {
        "primary"
    }:
        raise ValueError("SFRD-1 preregistered controls differ from evaluator")

    cutoff = STAGE2[1]
    for name, schedule in schedules.items():
        schedule = schedule.loc[schedule["exit_time"].le(cutoff)].reset_index(drop=True)
        if not bool(schedule["side"].isin((-1, 1)).all()):
            raise ValueError(f"SFRD-1 {name} has an invalid side")
        if not schedule["entry_time"].eq(
            schedule["signal_day"] + pd.Timedelta(minutes=5)
        ).all():
            raise ValueError(f"SFRD-1 {name} entry delay changed")
        if not schedule["exit_time"].eq(
            schedule["entry_time"] + pd.Timedelta(days=5)
        ).all():
            raise ValueError(f"SFRD-1 {name} hold changed")
        if len(schedule) > 1 and not schedule["entry_time"].iloc[1:].reset_index(
            drop=True
        ).ge(schedule["exit_time"].iloc[:-1].reset_index(drop=True)).all():
            raise ValueError(f"SFRD-1 {name} schedule overlaps")
        schedules[name] = schedule

    for name in ("direction_flip", "random_side"):
        if not schedules[name][["entry_time", "exit_time"]].equals(
            schedules["primary"][["entry_time", "exit_time"]]
        ):
            raise ValueError(f"SFRD-1 {name} does not preserve the primary clock")
    if not bool(
        schedules["direction_flip"]["side"]
        .eq(-schedules["primary"]["side"])
        .all()
    ):
        raise ValueError("SFRD-1 direction flip did not reverse every side")
    _verify_primary_preflight_counts(registration, schedules["primary"])
    return schedules


def _window_schedule(
    frame: pd.DataFrame, window: TimeWindow
) -> pd.DataFrame:
    start, end = window
    return frame.loc[
        frame["signal_day"].ge(start)
        & frame["entry_time"].ge(start)
        & frame["exit_time"].le(end)
    ].copy()


def _schedule_freeze_record(schedule: pd.DataFrame) -> dict[str, Any]:
    return {
        "all_pre2024": int(len(schedule)),
        "stage1_2021_2022": int(len(_window_schedule(schedule, STAGE1))),
        "stage2_2023": int(len(_window_schedule(schedule, STAGE2))),
        "stage1_entry_distribution": _entry_distribution(
            _window_schedule(schedule, STAGE1)
        ),
        "stage2_entry_distribution": _entry_distribution(
            _window_schedule(schedule, STAGE2)
        ),
        "schedule_hash": _schedule_hash(schedule),
    }


def freeze_evaluator(output_path: str | Path = EVALUATOR_FREEZE) -> dict[str, Any]:
    registration, support = _verify_static_inputs()
    schedules = load_schedules()
    counts = {
        name: _schedule_freeze_record(schedule)
        for name, schedule in schedules.items()
    }
    core = {
        "protocol_version": "sofr_rate_dislocation_evaluator_v1",
        "policy_id": "SFRD-1",
        "candidate_class": registration["research_history_boundary"]["candidate_class"],
        "as_of_date": "2026-07-17",
        "support_commit": SUPPORT_COMMIT,
        "support_preregistration_manifest_hash": support[
            "preregistration_manifest_hash"
        ],
        "evaluator_source": str(EVALUATOR_SOURCE),
        "evaluator_source_sha256": _sha256(EVALUATOR_SOURCE),
        "strict_engine_source": "training/evaluate_fiat_quote_participation_rotation.py",
        "strict_engine_source_sha256": STATIC_INPUT_SHA256[
            "training/evaluate_fiat_quote_participation_rotation.py"
        ],
        "evaluation_config": asdict(EvaluationConfig()),
        "selection_protocol": registration["selection_protocol"],
        "static_inputs": dict(STATIC_INPUT_SHA256),
        "execution_source_contract": {
            "market": str(MARKET),
            "market_expected_sha256_copied_without_read": registration[
                "source_contract"
            ]["market_sha256"],
            "funding": str(FUNDING),
            "funding_expected_sha256_copied_without_read": registration[
                "source_contract"
            ]["funding_sha256"],
            "stage1_physical_window": [STAGE1[0].isoformat(), STAGE1[1].isoformat()],
            "stage2_physical_window": [STAGE2[0].isoformat(), STAGE2[1].isoformat()],
        },
        "control_schedules": counts,
        "opened_windows": [],
        "sealed_windows": [
            "stage1_2021_2022",
            "stage2_2023",
            "2024",
            "2025",
            "2026_ytd",
        ],
        "mutable_parameters": [],
        "execution_ohlc_rows_parsed_during_freeze": 0,
        "funding_rows_parsed_during_freeze": 0,
        "simulation_run_during_freeze": False,
    }
    report = _seal(core)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def verify_evaluator_freeze(path: str | Path = EVALUATOR_FREEZE) -> dict[str, Any]:
    payload = _load_json(path)
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != _canonical_hash(core):
        raise ValueError("SFRD-1 evaluator freeze hash changed")
    if payload.get("evaluator_source_sha256") != _sha256(EVALUATOR_SOURCE):
        raise ValueError("SFRD-1 evaluator source changed after freeze")
    if payload.get("support_commit") != SUPPORT_COMMIT:
        raise ValueError("SFRD-1 evaluator froze another support commit")
    if payload.get("evaluation_config") != asdict(EvaluationConfig()):
        raise ValueError("SFRD-1 evaluation configuration changed")
    registration, _ = _verify_static_inputs()
    if payload.get("selection_protocol") != registration["selection_protocol"]:
        raise ValueError("SFRD-1 frozen selection protocol changed")
    if payload.get("static_inputs") != STATIC_INPUT_SHA256:
        raise ValueError("SFRD-1 frozen static-input ledger changed")
    if payload.get("strict_engine_source_sha256") != STATIC_INPUT_SHA256[
        "training/evaluate_fiat_quote_participation_rotation.py"
    ]:
        raise ValueError("SFRD-1 frozen strict engine changed")
    if payload.get("opened_windows") != [] or payload.get("mutable_parameters") != []:
        raise ValueError("SFRD-1 evaluator freeze is not sealed")
    if payload.get("execution_ohlc_rows_parsed_during_freeze") != 0:
        raise ValueError("SFRD-1 evaluator freeze parsed OHLC")
    if payload.get("funding_rows_parsed_during_freeze") != 0:
        raise ValueError("SFRD-1 evaluator freeze parsed funding")
    if payload.get("simulation_run_during_freeze") is not False:
        raise ValueError("SFRD-1 evaluator freeze simulated an outcome")
    schedules = load_schedules()
    if set(payload.get("control_schedules", {})) != set(ALL_CLOCK_NAMES):
        raise ValueError("SFRD-1 frozen schedule family changed")
    for name, schedule in schedules.items():
        if payload["control_schedules"][name] != _schedule_freeze_record(schedule):
            raise ValueError(f"SFRD-1 {name} schedule changed after freeze")
    return payload


def _verify_execution_manifests() -> None:
    registration = _load_json(PREREGISTRATION)
    source = registration["source_contract"]
    market_manifest = _load_json(MARKET_MANIFEST)
    if market_manifest.get("combined_sha256") != source["market_sha256"]:
        raise ValueError("SFRD-1 market manifest no longer binds the source")
    funding_manifest = _load_json(FUNDING_MANIFEST)
    if funding_manifest.get("data", {}).get("sha256") != source["funding_sha256"]:
        raise ValueError("SFRD-1 funding manifest no longer binds the source")
    if funding_manifest.get("outcomes_opened") is not False:
        raise ValueError("SFRD-1 funding source manifest opened an outcome")


def load_execution_window(
    window: TimeWindow,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    _verify_execution_manifests()
    start, end = window
    market, market_diagnostics = _parse_market_window(MARKET, start, end)
    funding, funding_diagnostics = _parse_funding_window(FUNDING, start, end)
    return market, funding, {
        "physical_window": [start.isoformat(), end.isoformat()],
        "market": market_diagnostics,
        "funding": funding_diagnostics,
        "full_market_file_sha256_not_recomputed": True,
        "full_funding_file_sha256_not_recomputed": True,
    }


def _simulate_subperiods(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    schedule: pd.DataFrame,
    subperiods: dict[str, TimeWindow],
    *,
    cost_rate: float,
    cfg: EvaluationConfig,
) -> dict[str, Any]:
    return {
        name: simulate_schedule(
            market,
            funding,
            schedule,
            period_start=start,
            period_end=end,
            cost_rate=cost_rate,
            cfg=cfg,
        )
        for name, (start, end) in subperiods.items()
    }


def _base_qualification(
    metrics: dict[str, Any],
    stress: dict[str, Any],
    subperiods: dict[str, Any],
    *,
    total_trades_min: int,
    subperiod_trade_mins: dict[str, int],
) -> dict[str, bool]:
    return strict_engine._base_qualification(
        metrics,
        stress,
        subperiods,
        total_trades_min=total_trades_min,
        subperiod_trade_mins=subperiod_trade_mins,
    )


def _metric_entry_distribution(metrics: dict[str, Any]) -> dict[str, Any]:
    details = metrics.get("trade_details")
    if not isinstance(details, list):
        raise ValueError("SFRD-1 simulation omitted trade details")
    return _entry_distribution(
        pd.DataFrame(details, columns=pd.Index(["entry_time", "side"]))
    )


def _full_qualification(
    metrics: dict[str, Any],
    stress: dict[str, Any],
    subperiods: dict[str, Any],
    *,
    total_trades_min: int,
    subperiod_trade_mins: dict[str, int],
    each_side_trades_min: int,
    max_single_entry_month_share: float,
) -> dict[str, bool]:
    distribution = _metric_entry_distribution(metrics)
    gates = {
        **_base_qualification(
            metrics,
            stress,
            subperiods,
            total_trades_min=total_trades_min,
            subperiod_trade_mins=subperiod_trade_mins,
        ),
        "minimum_each_side_trades": min(
            distribution["long_trades"], distribution["short_trades"]
        )
        >= each_side_trades_min,
        "single_entry_month_share_at_most_15pct": distribution[
            "max_single_entry_month_share"
        ]
        <= max_single_entry_month_share,
    }
    if set(gates) != BASE_QUALIFICATION_GATE_NAMES:
        raise ValueError("SFRD-1 base qualification gate set changed")
    return gates


def _headline(metrics: dict[str, Any]) -> dict[str, Any]:
    return strict_engine._headline(metrics)


def evaluate_stage1() -> dict[str, Any]:
    freeze = verify_evaluator_freeze()
    registration = _load_json(PREREGISTRATION)
    gate_contract = registration["selection_protocol"]["gates"]
    schedules = load_schedules()
    market, funding, diagnostics = load_execution_window(STAGE1)
    cfg = EvaluationConfig()
    base = {
        name: simulate_schedule(
            market,
            funding,
            schedule,
            period_start=STAGE1[0],
            period_end=STAGE1[1],
            cost_rate=cfg.base_cost_notional_per_side,
            cfg=cfg,
        )
        for name, schedule in schedules.items()
    }
    primary_stress = simulate_schedule(
        market,
        funding,
        schedules["primary"],
        period_start=STAGE1[0],
        period_end=STAGE1[1],
        cost_rate=cfg.stress_cost_notional_per_side,
        cfg=cfg,
    )
    primary_subperiods = _simulate_subperiods(
        market,
        funding,
        schedules["primary"],
        STAGE1_SUBPERIODS,
        cost_rate=cfg.base_cost_notional_per_side,
        cfg=cfg,
    )
    primary_gates = _full_qualification(
        base["primary"],
        primary_stress,
        primary_subperiods,
        total_trades_min=int(gate_contract["train_trades_min"]),
        subperiod_trade_mins={
            "2021": int(gate_contract["2021_trades_min"]),
            "2022": int(gate_contract["2022_trades_min"]),
        },
        each_side_trades_min=int(gate_contract["train_each_side_trades_min"]),
        max_single_entry_month_share=float(
            gate_contract["train_single_month_share_max"]
        ),
    )
    full_control_qualification: dict[str, Any] = {}
    for name in ("one_observation_delay", "random_side"):
        stress = simulate_schedule(
            market,
            funding,
            schedules[name],
            period_start=STAGE1[0],
            period_end=STAGE1[1],
            cost_rate=cfg.stress_cost_notional_per_side,
            cfg=cfg,
        )
        subperiods = _simulate_subperiods(
            market,
            funding,
            schedules[name],
            STAGE1_SUBPERIODS,
            cost_rate=cfg.base_cost_notional_per_side,
            cfg=cfg,
        )
        control_gates = _full_qualification(
            base[name],
            stress,
            subperiods,
            total_trades_min=int(gate_contract["train_trades_min"]),
            subperiod_trade_mins={
                "2021": int(gate_contract["2021_trades_min"]),
                "2022": int(gate_contract["2022_trades_min"]),
            },
            each_side_trades_min=int(gate_contract["train_each_side_trades_min"]),
            max_single_entry_month_share=float(
                gate_contract["train_single_month_share_max"]
            ),
        )
        full_control_qualification[name] = {
            "stress_cost": stress,
            "subperiods": subperiods,
            "gates": control_gates,
            "passed": bool(all(control_gates.values())),
        }
    mechanism_controls = registration["selection_protocol"][
        "control_comparison_contract"
    ]["mechanism_controls"]
    primary_ratio = base["primary"]["cagr_to_strict_mdd"]
    ratio_comparison = {
        name: {
            "primary_ratio": primary_ratio,
            "control_ratio": base[name]["cagr_to_strict_mdd"],
            "passed": primary_ratio > base[name]["cagr_to_strict_mdd"],
        }
        for name in mechanism_controls
    }
    gates = {
        **primary_gates,
        "primary_ratio_strictly_beats_each_mechanism_control": all(
            row["passed"] for row in ratio_comparison.values()
        ),
        "one_observation_delay_and_random_do_not_fully_qualify": not any(
            row["passed"] for row in full_control_qualification.values()
        ),
    }
    if set(gates) != STAGE1_GATE_NAMES:
        raise ValueError("SFRD-1 Stage-1 gate set changed")
    passed = bool(all(gates.values()))
    core = {
        "protocol_version": "sofr_rate_dislocation_stage1_v1",
        "policy_id": "SFRD-1",
        "candidate_class": registration["research_history_boundary"]["candidate_class"],
        "stage": "stage1_2021_2022",
        "evaluator_freeze_manifest_hash": freeze["manifest_hash"],
        "evaluator_source_sha256": _sha256(EVALUATOR_SOURCE),
        "config": asdict(cfg),
        "execution_diagnostics": diagnostics,
        "base_cost_by_clock": base,
        "primary_stress_cost": primary_stress,
        "primary_subperiods": primary_subperiods,
        "entry_distribution_by_clock": {
            name: _metric_entry_distribution(metrics)
            for name, metrics in base.items()
        },
        "full_qualification_by_control": full_control_qualification,
        "mechanism_ratio_comparison": ratio_comparison,
        "gates": gates,
        "gate_passed": passed,
        "opened_windows": ["stage1_2021_2022"],
        "sealed_windows": ["stage2_2023", "2024", "2025", "2026_ytd"],
        "disposition": (
            "PASS_STAGE1_OPEN_2023_ONCE"
            if passed
            else "REJECT_STAGE1_KEEP_2023_AND_LATER_SEALED"
        ),
    }
    report = _seal(core)
    STAGE1_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    STAGE1_OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    STAGE1_DOC.parent.mkdir(parents=True, exist_ok=True)
    STAGE1_DOC.write_text(render_stage_doc(report))
    return report


def _verified_passing_stage1(*, expected_freeze_manifest_hash: str) -> dict[str, Any]:
    if not STAGE1_OUTPUT.exists():
        raise ValueError("SFRD-1 Stage 1 has not been run")
    report = _load_json(STAGE1_OUTPUT)
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    if report.get("manifest_hash") != _canonical_hash(core):
        raise ValueError("SFRD-1 Stage-1 result hash changed")
    if report.get("policy_id") != "SFRD-1" or report.get("stage") != (
        "stage1_2021_2022"
    ):
        raise ValueError("SFRD-1 Stage-1 result identity changed")
    if report.get("evaluator_freeze_manifest_hash") != expected_freeze_manifest_hash:
        raise ValueError("SFRD-1 Stage 1 is not bound to the current evaluator freeze")
    if report.get("evaluator_source_sha256") != _sha256(EVALUATOR_SOURCE):
        raise ValueError("SFRD-1 Stage 1 used another evaluator")
    if report.get("config") != asdict(EvaluationConfig()):
        raise ValueError("SFRD-1 Stage-1 evaluation configuration changed")
    if report.get("opened_windows") != ["stage1_2021_2022"]:
        raise ValueError("SFRD-1 Stage-1 opened-window contract changed")
    if report.get("sealed_windows") != [
        "stage2_2023",
        "2024",
        "2025",
        "2026_ytd",
    ]:
        raise ValueError("SFRD-1 Stage-1 sealed-window contract changed")
    diagnostics = report.get("execution_diagnostics")
    if not isinstance(diagnostics, dict) or diagnostics.get("physical_window") != [
        STAGE1[0].isoformat(),
        STAGE1[1].isoformat(),
    ]:
        raise ValueError("SFRD-1 Stage-1 physical window changed")
    gates = report.get("gates")
    if (
        not isinstance(gates, dict)
        or set(gates) != STAGE1_GATE_NAMES
        or not all(value is True for value in gates.values())
    ):
        raise ValueError("SFRD-1 Stage-1 gate evidence is incomplete")
    if report.get("gate_passed") is not True:
        raise ValueError("SFRD-1 Stage 1 did not authorize opening 2023")
    if report.get("disposition") != "PASS_STAGE1_OPEN_2023_ONCE":
        raise ValueError("SFRD-1 Stage-1 disposition does not authorize 2023")
    return report


def evaluate_stage2() -> dict[str, Any]:
    freeze = verify_evaluator_freeze()
    stage1 = _verified_passing_stage1(
        expected_freeze_manifest_hash=freeze["manifest_hash"]
    )
    registration = _load_json(PREREGISTRATION)
    gate_contract = registration["selection_protocol"]["gates"]
    schedules = load_schedules()
    market, funding, diagnostics = load_execution_window(STAGE2)
    cfg = EvaluationConfig()
    base = {
        name: simulate_schedule(
            market,
            funding,
            schedule,
            period_start=STAGE2[0],
            period_end=STAGE2[1],
            cost_rate=cfg.base_cost_notional_per_side,
            cfg=cfg,
        )
        for name, schedule in schedules.items()
    }
    primary_stress = simulate_schedule(
        market,
        funding,
        schedules["primary"],
        period_start=STAGE2[0],
        period_end=STAGE2[1],
        cost_rate=cfg.stress_cost_notional_per_side,
        cfg=cfg,
    )
    primary_subperiods = _simulate_subperiods(
        market,
        funding,
        schedules["primary"],
        STAGE2_SUBPERIODS,
        cost_rate=cfg.base_cost_notional_per_side,
        cfg=cfg,
    )
    primary_gates = _full_qualification(
        base["primary"],
        primary_stress,
        primary_subperiods,
        total_trades_min=int(gate_contract["2023_trades_min"]),
        subperiod_trade_mins={
            "2023_h1": int(gate_contract["2023_h1_trades_min"]),
            "2023_h2": int(gate_contract["2023_h2_trades_min"]),
        },
        each_side_trades_min=int(gate_contract["2023_each_side_trades_min"]),
        max_single_entry_month_share=float(
            gate_contract["2023_single_month_share_max"]
        ),
    )
    full_control_qualification: dict[str, Any] = {}
    for name in ("one_observation_delay", "random_side"):
        stress = simulate_schedule(
            market,
            funding,
            schedules[name],
            period_start=STAGE2[0],
            period_end=STAGE2[1],
            cost_rate=cfg.stress_cost_notional_per_side,
            cfg=cfg,
        )
        subperiods = _simulate_subperiods(
            market,
            funding,
            schedules[name],
            STAGE2_SUBPERIODS,
            cost_rate=cfg.base_cost_notional_per_side,
            cfg=cfg,
        )
        control_gates = _full_qualification(
            base[name],
            stress,
            subperiods,
            total_trades_min=int(gate_contract["2023_trades_min"]),
            subperiod_trade_mins={
                "2023_h1": int(gate_contract["2023_h1_trades_min"]),
                "2023_h2": int(gate_contract["2023_h2_trades_min"]),
            },
            each_side_trades_min=int(gate_contract["2023_each_side_trades_min"]),
            max_single_entry_month_share=float(
                gate_contract["2023_single_month_share_max"]
            ),
        )
        full_control_qualification[name] = {
            "stress_cost": stress,
            "subperiods": subperiods,
            "gates": control_gates,
            "passed": bool(all(control_gates.values())),
        }
    mechanism_controls = registration["selection_protocol"][
        "control_comparison_contract"
    ]["mechanism_controls"]
    primary_min_ratio = min(
        stage1["base_cost_by_clock"]["primary"]["cagr_to_strict_mdd"],
        base["primary"]["cagr_to_strict_mdd"],
    )
    ratio_comparison: dict[str, Any] = {}
    for name in mechanism_controls:
        control_min = min(
            stage1["base_cost_by_clock"][name]["cagr_to_strict_mdd"],
            base[name]["cagr_to_strict_mdd"],
        )
        ratio_comparison[name] = {
            "primary_min_train_2023_ratio": primary_min_ratio,
            "control_min_train_2023_ratio": control_min,
            "passed": primary_min_ratio > control_min,
        }
    stale_or_random_full = {
        name: bool(
            stage1["full_qualification_by_control"][name]["passed"]
            and full_control_qualification[name]["passed"]
        )
        for name in ("one_observation_delay", "random_side")
    }
    gates = {
        **primary_gates,
        "minimum_train_2023_ratio_strictly_beats_each_mechanism_control": all(
            row["passed"] for row in ratio_comparison.values()
        ),
        "one_observation_delay_and_random_do_not_fully_qualify_both_splits": not any(
            stale_or_random_full.values()
        ),
    }
    if set(gates) != STAGE2_GATE_NAMES:
        raise ValueError("SFRD-1 Stage-2 gate set changed")
    passed = bool(all(gates.values()))
    core = {
        "protocol_version": "sofr_rate_dislocation_stage2_v1",
        "policy_id": "SFRD-1",
        "candidate_class": registration["research_history_boundary"]["candidate_class"],
        "stage": "stage2_2023",
        "evaluator_freeze_manifest_hash": freeze["manifest_hash"],
        "stage1_manifest_hash": stage1["manifest_hash"],
        "evaluator_source_sha256": _sha256(EVALUATOR_SOURCE),
        "config": asdict(cfg),
        "execution_diagnostics": diagnostics,
        "base_cost_by_clock": base,
        "primary_stress_cost": primary_stress,
        "primary_subperiods": primary_subperiods,
        "entry_distribution_by_clock": {
            name: _metric_entry_distribution(metrics)
            for name, metrics in base.items()
        },
        "full_qualification_by_control": full_control_qualification,
        "mechanism_ratio_comparison": ratio_comparison,
        "stale_or_random_full_qualification_both_splits": stale_or_random_full,
        "gates": gates,
        "gate_passed": passed,
        "opened_windows": ["stage1_2021_2022", "stage2_2023"],
        "sealed_windows": ["2024", "2025", "2026_ytd"],
        "disposition": (
            "PASS_STAGE2_OPEN_ORTHOGONALITY_KEEP_2024_SEALED"
            if passed
            else "REJECT_STAGE2_KEEP_2024_AND_LATER_SEALED"
        ),
    }
    report = _seal(core)
    STAGE2_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    STAGE2_OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    STAGE2_DOC.parent.mkdir(parents=True, exist_ok=True)
    STAGE2_DOC.write_text(render_stage_doc(report))
    return report


def render_stage_doc(report: dict[str, Any]) -> str:
    stage = report["stage"]
    primary = report["base_cost_by_clock"]["primary"]
    stress = report["primary_stress_cost"]
    subperiod_rows = "\n".join(
        f"| {name} | {row['absolute_return_pct']:.4f}% | {row['cagr_pct']:.4f}% | "
        f"{row['strict_mdd_pct']:.4f}% | {row['cagr_to_strict_mdd']:.4f} | "
        f"{row['trades']} |"
        for name, row in report["primary_subperiods"].items()
    )
    gate_rows = "\n".join(
        f"| `{name}` | {'PASS' if passed else 'FAIL'} |"
        for name, passed in report["gates"].items()
    )
    sealed = ", ".join(report["sealed_windows"])
    return f"""# SFRD-1 {stage} result

## Decision

**{report['disposition']}**

| Cost | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Mean gross bp |
|---|---:|---:|---:|---:|---:|---:|
| 6 bp/notional/side | {primary['absolute_return_pct']:.4f}% | {primary['cagr_pct']:.4f}% | {primary['strict_mdd_pct']:.4f}% | {primary['cagr_to_strict_mdd']:.4f} | {primary['trades']} | {primary['mean_gross_underlying_bp']:.4f} |
| 10 bp/notional/side | {stress['absolute_return_pct']:.4f}% | {stress['cagr_pct']:.4f}% | {stress['strict_mdd_pct']:.4f}% | {stress['cagr_to_strict_mdd']:.4f} | {stress['trades']} | {stress['mean_gross_underlying_bp']:.4f} |

## Contained subperiods

| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
{subperiod_rows}

## Gates

| Gate | Result |
|---|:---:|
{gate_rows}

## Isolation

- Candidate class: `{report['candidate_class']}`
- Evaluator freeze: `{report['evaluator_freeze_manifest_hash']}`
- Physically opened window: `{report['execution_diagnostics']['physical_window']}`
- Still sealed: {sealed}
- Full-clock CAGR includes warm-up and idle cash.
- Strict MDD uses favorable-before-adverse held OHLC, global high-water,
  entry/hypothetical-exit/realized-exit costs, and exact realized funding.
"""


def _print_stage(report: dict[str, Any]) -> None:
    print(
        json.dumps(
            {
                "stage": report["stage"],
                "gate_passed": report["gate_passed"],
                "disposition": report["disposition"],
                "primary": _headline(report["base_cost_by_clock"]["primary"]),
                "stress": _headline(report["primary_stress_cost"]),
                "gates": report["gates"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--freeze", action="store_true")
    action.add_argument("--stage1", action="store_true")
    action.add_argument("--stage2", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        report = freeze_evaluator()
        print(
            json.dumps(
                {
                    "output": str(EVALUATOR_FREEZE),
                    "manifest_hash": report["manifest_hash"],
                    "opened_windows": report["opened_windows"],
                    "execution_ohlc_rows_parsed_during_freeze": report[
                        "execution_ohlc_rows_parsed_during_freeze"
                    ],
                    "funding_rows_parsed_during_freeze": report[
                        "funding_rows_parsed_during_freeze"
                    ],
                },
                indent=2,
            )
        )
    elif args.stage1:
        _print_stage(evaluate_stage1())
    else:
        _print_stage(evaluate_stage2())


if __name__ == "__main__":
    main()
