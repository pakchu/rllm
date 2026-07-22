"""Sequential strict evaluator for frozen FLNSR-2016.

The evaluator freeze reads only source-clock artifacts and metadata manifests.
Stage 1 physically parses only 2020-2022 market/funding rows.  The 2023
selection window remains unreachable until a hash-bound Stage-1 report is
replayed from the frozen sources and passes every preregistered gate.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from training import evaluate_fiat_quote_participation_rotation as strict_engine
from training import preregister_federal_liquidity_narrative_sponsorship_relay as prereg


__all__ = ["freeze_evaluator", "evaluate_stage1", "evaluate_stage2", "main"]


SUPPORT_COMMIT = "cd731fe3de60f3eea4461f41ad4cb8c6175c606f"
STATIC_INPUT_SHA256 = {
    "training/preregister_federal_liquidity_narrative_sponsorship_relay.py": (
        "f616ef1270b0fc8a6400ae14a3108e9f360b102107468f1732f5c0aa94c990ed"
    ),
    "docs/federal-liquidity-narrative-sponsorship-relay-preregistration-2026-07-23.md": (
        "ba8439043fa20fd8b4717b5f0fae63e0ea1fdd08fca45fe7935fd98fc6443a04"
    ),
    "results/federal_liquidity_narrative_sponsorship_relay_preregistration_2026-07-23.json": (
        "252952438eb2a87dc5f85fbe887a4f99a5f3a7a8a7e764feac414fac2929fd6d"
    ),
    "training/build_federal_liquidity_narrative_sponsorship_support.py": (
        "f0247b3b4767fb9da892f2409116e88f32093253384a2b22fc6a41ed780a29c0"
    ),
    "docs/federal-liquidity-narrative-sponsorship-relay-support-pass-2026-07-23.md": (
        "cd74437720c5a96c820c70f1f215752faeaa012393cec0edda7ed630ad18f75a"
    ),
    "results/federal_liquidity_narrative_sponsorship_relay_support_2026-07-23.json": (
        "25f2a4f4b22e51b137b739048b06b9b015da2c6b604c88a68e4082b6f7de6f3f"
    ),
    "data/federal_liquidity_narrative_sponsorship_relay_clocks_2020_2023.csv.gz": (
        "3096143d397fc6d8dac639841c96538979772734dcf2fd8157df580f5b297c6c"
    ),
    "training/evaluate_fiat_quote_participation_rotation.py": (
        "e309f5217f033d57d2eadfec936843e736ce287f5c47f957c0ac6f0c71879c23"
    ),
    "data/binance_um_kline_reference_btc_2020_2023/build_manifest.json": (
        "c04fbbd299cc748a6745c0ef030787da4d560833c744c81c98dd8840efc7913e"
    ),
    "results/binance_um_btcusdt_funding_marks_2020_2023_manifest_2026-07-17.json": (
        "a0b2d27e1aa8cf2d9ab8cb659b598ee0a6d7bd25401c9e10ae92d1a74415845b"
    ),
}

PREREGISTRATION = Path(prereg.DEFAULT_OUTPUT)
SUPPORT = Path(
    "results/federal_liquidity_narrative_sponsorship_relay_support_2026-07-23.json"
)
CLOCK_LEDGER = Path(
    "data/federal_liquidity_narrative_sponsorship_relay_clocks_2020_2023.csv.gz"
)
MARKET = Path(
    "data/binance_um_kline_reference_btc_2020_2023/"
    "BTCUSDT_5m_2020-01-01_2023-12-31.csv.gz"
)
MARKET_SHA256 = "e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d"
MARKET_MANIFEST = Path(
    "data/binance_um_kline_reference_btc_2020_2023/build_manifest.json"
)
FUNDING = Path("data/binance_um_btcusdt_funding_marks_2020_2023.csv.gz")
FUNDING_SHA256 = "3284bbb6bb67946acb673c6b67459543e217f752589e1d47b6c7c3b659f733e6"
FUNDING_MANIFEST = Path(
    "results/binance_um_btcusdt_funding_marks_2020_2023_manifest_2026-07-17.json"
)
STAGE1_MARKET_WINDOW_LINE_SHA256 = (
    "744ac1ad59e53c088e1b6697ecaa073b2cd12cec5823957ac6ffaf2feab896bd"
)
STAGE1_FUNDING_WINDOW_LINE_SHA256 = (
    "9a211053a26eb6b3dd0f00a32cb43f2706cea2ca876ed42a936a669039ddff0b"
)
STAGE1_PREFIX_PROVENANCE = (
    "results/federal_liquidity_component_concordance_"
    "stage1_2020_2022_2026-07-17.json"
)
STAGE1_PREFIX_PROVENANCE_SHA256 = (
    "10dc911ad06c7e523d612ff34675421388fefb94fa93e157bfac7e93bd1d82a6"
)
EVALUATOR_SOURCE = Path(
    "training/evaluate_federal_liquidity_narrative_sponsorship_relay.py"
)
EVALUATOR_FREEZE = Path(
    "results/federal_liquidity_narrative_sponsorship_relay_"
    "evaluator_freeze_2026-07-23.json"
)
STAGE1_OUTPUT = Path(
    "results/federal_liquidity_narrative_sponsorship_relay_"
    "stage1_2020_2022_2026-07-23.json"
)
STAGE1_DOC = Path(
    "docs/federal-liquidity-narrative-sponsorship-relay-"
    "stage1-2020-2022-2026-07-23.md"
)
STAGE2_OUTPUT = Path(
    "results/federal_liquidity_narrative_sponsorship_relay_"
    "stage2_2023_2026-07-23.json"
)
STAGE2_DOC = Path(
    "docs/federal-liquidity-narrative-sponsorship-relay-stage2-2023-2026-07-23.md"
)

POLICY_ID = "FLNSR-2016"
TIME_COLUMNS = (
    "release_date",
    "narrative_source_date",
    "signal_time",
    "entry_time",
    "exit_time",
)
SOURCE_CLOCK_COLUMNS = (
    "clock_name",
    "release_date",
    "narrative_source_date",
    "signal_time",
    "entry_time",
    "exit_time",
    "side",
)
MECHANISM_CONTROLS = ("liquidity_only", "narrative_only", "disagreement")
FALSIFICATION_CONTROLS = (
    "exact_side_flip",
    "one_release_stale_narrative",
    "deterministic_random_side",
)
ALL_CLOCK_NAMES = ("primary", *MECHANISM_CONTROLS, *FALSIFICATION_CONTROLS)
DELAY_CLOCK_NAME = "one_extra_bar_delay"
BAR = pd.Timedelta(minutes=5)
HOLD = pd.Timedelta(days=7)
MONTHLY_CLUSTER_DRAWS = 20_000
MONTHLY_CLUSTER_SEED = 20_260_723
MONTHLY_CLUSTER_P_MAX = 0.05
MINIMUM_MEAN_GROSS_BP = 30.0
MINIMUM_COMPONENT_MARGIN_BP = 5.0

TimeWindow = tuple[pd.Timestamp, pd.Timestamp]


def _utc_timestamp(value: str) -> pd.Timestamp:
    return cast(pd.Timestamp, pd.Timestamp(value, tz="UTC"))


def _timestamp(value: Any) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result is pd.NaT:
        raise ValueError("FLNSR-2016 timestamp is NaT")
    if result.tzinfo is None:
        raise ValueError("FLNSR-2016 timestamp is not timezone-aware")
    return cast(pd.Timestamp, result)


STAGE1: TimeWindow = (_utc_timestamp("2020-01-01"), _utc_timestamp("2023-01-01"))
STAGE1_SUBPERIODS: dict[str, TimeWindow] = {
    "2020": (_utc_timestamp("2020-01-01"), _utc_timestamp("2021-01-01")),
    "2021": (_utc_timestamp("2021-01-01"), _utc_timestamp("2022-01-01")),
    "2022": (_utc_timestamp("2022-01-01"), _utc_timestamp("2023-01-01")),
}
STAGE2: TimeWindow = (_utc_timestamp("2023-01-01"), _utc_timestamp("2024-01-01"))
STAGE2_SUBPERIODS: dict[str, TimeWindow] = {
    "2023_h1": (_utc_timestamp("2023-01-01"), _utc_timestamp("2023-07-01")),
    "2023_h2": (_utc_timestamp("2023-07-01"), _utc_timestamp("2024-01-01")),
}
STAGE1_PROTOCOL = "federal_liquidity_narrative_sponsorship_stage1_v1"
STAGE2_PROTOCOL = "federal_liquidity_narrative_sponsorship_stage2_v1"
STAGE1_ID = "stage1_2020_2022"
STAGE2_ID = "stage2_2023"
STAGE1_OPENED_WINDOWS = [STAGE1_ID]
STAGE1_SEALED_WINDOWS = [STAGE2_ID, "2024", "2025", "2026_ytd"]

EvaluationConfig = strict_engine.EvaluationConfig
simulate_schedule = strict_engine.simulate_schedule
_parse_market_window = strict_engine._parse_market_window


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


def _write_once_text(path: str | Path, content: str, *, label: str) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.read_text() != content:
            raise ValueError(f"refusing to overwrite frozen FLNSR-2016 {label}")
        return "verified_existing"
    with output.open("x", encoding="utf-8") as handle:
        handle.write(content)
    return "created"


def _verify_evaluation_contract(registration: dict[str, Any]) -> None:
    policy = registration["policy"]
    config = asdict(EvaluationConfig())
    expected = {
        "leverage": policy["leverage"],
        "base_cost_notional_per_side": policy["base_cost_notional_per_side"],
        "stress_cost_notional_per_side": policy["stress_cost_notional_per_side"],
    }
    observed = {key: config[key] for key in expected}
    if observed != expected:
        raise ValueError(
            f"FLNSR-2016 strict-engine economic contract changed: {observed} != {expected}"
        )
    if config["cluster_draws"] != 20_000 or config["mdd_denominator_floor"] != 1e-9:
        raise ValueError("FLNSR-2016 strict-engine statistical contract changed")


def _validate_support_manifest(payload: dict[str, Any]) -> None:
    core = {
        key: value
        for key, value in payload.items()
        if key not in {"report_manifest_hash", "created_at"}
    }
    if payload.get("report_manifest_hash") != _canonical_hash(core):
        raise ValueError("FLNSR-2016 support manifest hash mismatch")
    if payload.get("support_passed") is not True:
        raise ValueError("FLNSR-2016 source-only support did not pass")
    if payload.get("authorized_next_stage") != "freeze_strict_evaluator":
        raise ValueError("FLNSR-2016 support did not authorize evaluator freeze")
    if payload.get("outcomes_opened") is not False:
        raise ValueError("FLNSR-2016 support opened outcomes")
    if payload.get("post_entry_return_computed") is not False:
        raise ValueError("FLNSR-2016 support computed a post-entry return")
    if payload.get("funding_loaded") is not False:
        raise ValueError("FLNSR-2016 support loaded funding")


def _verify_static_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    for path, expected in STATIC_INPUT_SHA256.items():
        if _sha256(path) != expected:
            raise ValueError(f"FLNSR-2016 frozen input changed: {path}")
    registration = _load_json(PREREGISTRATION)
    prereg.validate_manifest(registration)
    _verify_evaluation_contract(registration)
    support = _load_json(SUPPORT)
    _validate_support_manifest(support)
    if support.get("policy_id") != POLICY_ID:
        raise ValueError("FLNSR-2016 support policy identity changed")
    prereg_binding = support.get("preregistration", {})
    if prereg_binding.get("sha256") != STATIC_INPUT_SHA256[str(PREREGISTRATION)]:
        raise ValueError("FLNSR-2016 support no longer binds preregistration")
    if prereg_binding.get("manifest_hash") != registration.get("manifest_hash"):
        raise ValueError("FLNSR-2016 support preregistration manifest changed")
    clock = support.get("clock", {})
    if clock.get("sha256") != STATIC_INPUT_SHA256[str(CLOCK_LEDGER)]:
        raise ValueError("FLNSR-2016 support no longer binds clock ledger")
    if tuple(clock.get("columns", ())) != SOURCE_CLOCK_COLUMNS:
        raise ValueError("FLNSR-2016 support clock schema changed")
    return registration, support


def _window_schedule(frame: pd.DataFrame, window: TimeWindow) -> pd.DataFrame:
    start, end = window
    return frame.loc[
        frame["signal_day"].ge(start)
        & frame["entry_time"].ge(start)
        & frame["exit_time"].le(end)
    ].copy()


def _schedule_hash(frame: pd.DataFrame) -> str:
    rows: list[dict[str, Any]] = []
    for row in frame.sort_values("entry_time", kind="mergesort").itertuples(index=False):
        rows.append(
            {
                "clock_name": str(row.clock_name),
                "release_date": _timestamp(row.release_date).isoformat(),
                "narrative_source_date": _timestamp(row.narrative_source_date).isoformat(),
                "signal_time": _timestamp(row.signal_time).isoformat(),
                "entry_time": _timestamp(row.entry_time).isoformat(),
                "exit_time": _timestamp(row.exit_time).isoformat(),
                "side": int(row.side),
            }
        )
    return _canonical_hash(rows)


def _belongs_to_exactly_one_stage(row: Any) -> bool:
    return sum(
        bool(
            _timestamp(row.signal_day) >= start
            and _timestamp(row.entry_time) >= start
            and _timestamp(row.exit_time) <= end
        )
        for start, end in (STAGE1, STAGE2)
    ) == 1


def load_schedules() -> dict[str, pd.DataFrame]:
    _, support = _verify_static_inputs()
    frame = pd.read_csv(CLOCK_LEDGER)
    if tuple(frame.columns) != SOURCE_CLOCK_COLUMNS:
        raise ValueError("FLNSR-2016 clock ledger schema changed")
    for column in TIME_COLUMNS:
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    frame["side"] = frame["side"].map({"LONG": 1, "SHORT": -1})
    if frame["side"].isna().any():
        raise ValueError("FLNSR-2016 clock has an invalid side")
    frame["side"] = frame["side"].astype("int8")
    frame["signal_day"] = frame["signal_time"]
    if set(frame["clock_name"]) != set(ALL_CLOCK_NAMES):
        raise ValueError("FLNSR-2016 clock family changed")
    if len(frame) != support["clock"]["rows"]:
        raise ValueError("FLNSR-2016 clock row count changed")
    if not frame["entry_time"].eq(frame["signal_time"] + pd.Timedelta(minutes=10)).all():
        raise ValueError("FLNSR-2016 entry delay changed")
    if not frame["exit_time"].eq(frame["entry_time"] + HOLD).all():
        raise ValueError("FLNSR-2016 hold changed")
    if not all(_belongs_to_exactly_one_stage(row) for row in frame.itertuples(index=False)):
        raise ValueError("FLNSR-2016 clock is not physically split-contained")

    schedules: dict[str, pd.DataFrame] = {}
    for name in ALL_CLOCK_NAMES:
        schedule = frame.loc[frame["clock_name"].eq(name)].copy()
        schedule = schedule.sort_values("entry_time", kind="mergesort").reset_index(drop=True)
        if schedule.empty:
            raise ValueError(f"FLNSR-2016 empty clock: {name}")
        for window in (STAGE1, STAGE2):
            selected = _window_schedule(schedule, window)
            if len(selected) > 1 and not (
                selected["entry_time"].iloc[1:].reset_index(drop=True)
                >= selected["exit_time"].iloc[:-1].reset_index(drop=True)
            ).all():
                raise ValueError(f"FLNSR-2016 overlapping clock: {name}")
        expected = support["controls"][name]["events"]
        if len(schedule) != expected:
            raise ValueError(f"FLNSR-2016 support count changed: {name}")
        schedules[name] = schedule

    expected_windows = {
        "train_2020_2022": (schedules["primary"], STAGE1),
        "2020": (schedules["primary"], STAGE1_SUBPERIODS["2020"]),
        "2021": (schedules["primary"], STAGE1_SUBPERIODS["2021"]),
        "2022": (schedules["primary"], STAGE1_SUBPERIODS["2022"]),
        "selection_2023": (schedules["primary"], STAGE2),
        "selection_2023_h1": (schedules["primary"], STAGE2_SUBPERIODS["2023_h1"]),
        "selection_2023_h2": (schedules["primary"], STAGE2_SUBPERIODS["2023_h2"]),
    }
    for support_name, (schedule, window) in expected_windows.items():
        observed = len(_window_schedule(schedule, window))
        if observed != support["windows"][support_name]["events"]:
            raise ValueError(f"FLNSR-2016 primary window count changed: {support_name}")
    return schedules


def one_extra_bar_delay_schedule(primary: pd.DataFrame) -> pd.DataFrame:
    delayed = primary.copy()
    delayed["clock_name"] = DELAY_CLOCK_NAME
    delayed["entry_time"] = delayed["entry_time"] + BAR
    delayed["exit_time"] = delayed["exit_time"] + BAR
    if not delayed["exit_time"].eq(delayed["entry_time"] + HOLD).all():
        raise ValueError("FLNSR-2016 delayed hold changed")
    return delayed


def _entry_distribution(frame: pd.DataFrame, window: TimeWindow) -> dict[str, Any]:
    selected = _window_schedule(frame, window)
    months = selected["entry_time"].dt.strftime("%Y-%m").value_counts()
    maximum = int(months.max()) if len(months) else 0
    return {
        "trades": int(len(selected)),
        "longs": int(selected["side"].eq(1).sum()),
        "shorts": int(selected["side"].eq(-1).sum()),
        "max_single_month_count": maximum,
        "max_single_month_share": maximum / len(selected) if len(selected) else 0.0,
    }


def _schedule_freeze_record(
    frame: pd.DataFrame, *, expected_entry_delay: pd.Timedelta
) -> dict[str, Any]:
    entries = [_timestamp(value) for value in frame["entry_time"]]
    exits = [_timestamp(value) for value in frame["exit_time"]]
    signals = [_timestamp(value) for value in frame["signal_time"]]
    return {
        "events": int(len(frame)),
        "stage1_events": int(len(_window_schedule(frame, STAGE1))),
        "stage2_events": int(len(_window_schedule(frame, STAGE2))),
        "schedule_hash": _schedule_hash(frame),
        "first_entry": min(entries).isoformat(),
        "last_exit": max(exits).isoformat(),
        "entry_delay_exact": all(
            entry - signal == expected_entry_delay
            for entry, signal in zip(entries, signals)
        ),
        "hold_exactly_seven_days": all(
            exit_time - entry == HOLD for entry, exit_time in zip(entries, exits)
        ),
        "split_contained": all(
            _belongs_to_exactly_one_stage(row) for row in frame.itertuples(index=False)
        ),
        "nonoverlapping_within_each_stage": all(
            len(selected) < 2
            or (
                selected["entry_time"].iloc[1:].reset_index(drop=True)
                >= selected["exit_time"].iloc[:-1].reset_index(drop=True)
            ).all()
            for selected in (
                _window_schedule(frame, STAGE1),
                _window_schedule(frame, STAGE2),
            )
        ),
        "pre_2024_only": all(exit_time <= STAGE2[1] for exit_time in exits),
    }


def freeze_evaluator(output_path: str | Path = EVALUATOR_FREEZE) -> dict[str, Any]:
    registration, support = _verify_static_inputs()
    schedules = load_schedules()
    delayed = one_extra_bar_delay_schedule(schedules["primary"])
    records = {
        name: _schedule_freeze_record(
            schedule, expected_entry_delay=pd.Timedelta(minutes=10)
        )
        for name, schedule in schedules.items()
    }
    records[DELAY_CLOCK_NAME] = _schedule_freeze_record(
        delayed, expected_entry_delay=pd.Timedelta(minutes=15)
    )
    invariant_checks = {
        f"{name}/{check}": bool(value)
        for name, record in records.items()
        for check, value in record.items()
        if isinstance(value, bool)
    }
    invariant_checks["delay/stage1_count_unchanged"] = (
        records[DELAY_CLOCK_NAME]["stage1_events"] == records["primary"]["stage1_events"]
    )
    invariant_checks["delay/stage2_count_unchanged"] = (
        records[DELAY_CLOCK_NAME]["stage2_events"] == records["primary"]["stage2_events"]
    )
    if not all(invariant_checks.values()):
        failed = [name for name, value in invariant_checks.items() if not value]
        raise ValueError(f"FLNSR-2016 evaluator schedule invariant failed: {failed}")
    core = {
        "protocol_version": "federal_liquidity_narrative_sponsorship_evaluator_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-07-23",
        "support_commit": SUPPORT_COMMIT,
        "preregistration_manifest_hash": registration["manifest_hash"],
        "support_manifest_hash": support["report_manifest_hash"],
        "evaluator_source": str(EVALUATOR_SOURCE),
        "evaluator_source_sha256": _sha256(EVALUATOR_SOURCE),
        "strict_engine_sha256": STATIC_INPUT_SHA256[
            "training/evaluate_fiat_quote_participation_rotation.py"
        ],
        "evaluation_config": asdict(EvaluationConfig()),
        "monthly_cluster_signflip": {
            "cluster": "UTC calendar month of entry_time",
            "statistic": "sum net return within month; Rademacher sign flip by month",
            "alternative": "one-sided null >= observed mean trade net return",
            "draws": MONTHLY_CLUSTER_DRAWS,
            "seed": MONTHLY_CLUSTER_SEED,
            "p_value_max": MONTHLY_CLUSTER_P_MAX,
        },
        "economic_gates": registration["economic_gates"],
        "evaluator_gate_mapping": {
            "base_stage": [
                "absolute_return_positive",
                "cagr_to_strict_mdd_at_least_3",
                "strict_mdd_at_most_15pct",
                "stress_cost_absolute_return_positive",
                "one_extra_bar_delay_absolute_return_positive",
                "mean_gross_underlying_at_least_30bp",
                "monthly_cluster_signflip_p_within_limit",
                "primary_mean_gross_margin_over_each_component_at_least_5bp",
            ],
            "train_each_calendar_year_positive": (
                "each_subperiod_absolute_return_positive over 2020, 2021, 2022"
            ),
            "selection_h1_and_h2_positive": (
                "each_subperiod_absolute_return_positive over 2023_h1, 2023_h2"
            ),
            "falsification_controls_report_only": list(FALSIFICATION_CONTROLS),
        },
        "static_inputs": STATIC_INPUT_SHA256,
        "execution_source_contract": {
            "market": str(MARKET),
            "market_expected_sha256_copied_without_read": MARKET_SHA256,
            "market_manifest": str(MARKET_MANIFEST),
            "funding": str(FUNDING),
            "funding_expected_sha256_copied_without_read": FUNDING_SHA256,
            "funding_manifest": str(FUNDING_MANIFEST),
            "stage1_prefix_hash_provenance": {
                "path_copied_without_read": STAGE1_PREFIX_PROVENANCE,
                "file_sha256_copied_without_read": STAGE1_PREFIX_PROVENANCE_SHA256,
                "market_window_line_sha256": STAGE1_MARKET_WINDOW_LINE_SHA256,
                "funding_window_line_sha256": STAGE1_FUNDING_WINDOW_LINE_SHA256,
                "history_boundary": "ancestor FLCC train outcome already disclosed as seen",
            },
            "stage1_physical_window": [value.isoformat() for value in STAGE1],
            "stage2_physical_window": [value.isoformat() for value in STAGE2],
        },
        "schedules": records,
        "invariant_checks": invariant_checks,
        "execution_ohlc_rows_parsed_during_freeze": 0,
        "funding_rows_parsed_during_freeze": 0,
        "simulation_run_during_freeze": False,
        "outcomes_opened": False,
        "opened_windows": [],
        "sealed_windows": [STAGE1_ID, STAGE2_ID, "2024", "2025", "2026_ytd"],
        "mutable_parameters": [],
        "stage2_gate": (
            "requires a hash-valid Stage1 report that is replayed from the frozen "
            "2020-2022 physical prefix and passes every frozen gate"
        ),
        "api_trust_boundary": (
            "the supported evaluator API is __all__; underscore-prefixed parser helpers "
            "are trusted implementation internals, not authorization endpoints. Direct "
            "filesystem or private-helper access is outside an in-process Python audit barrier"
        ),
    }
    payload = _seal(core)
    _write_once_text(
        output_path,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        label="evaluator freeze",
    )
    return payload


def verify_evaluator_freeze(path: str | Path = EVALUATOR_FREEZE) -> dict[str, Any]:
    payload = _load_json(path)
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != _canonical_hash(core):
        raise ValueError("FLNSR-2016 evaluator freeze hash mismatch")
    if payload.get("evaluator_source_sha256") != _sha256(EVALUATOR_SOURCE):
        raise ValueError("FLNSR-2016 evaluator changed after freeze")
    if payload.get("support_commit") != SUPPORT_COMMIT:
        raise ValueError("FLNSR-2016 evaluator froze another support commit")
    if payload.get("static_inputs") != STATIC_INPUT_SHA256:
        raise ValueError("FLNSR-2016 evaluator static-input set changed")
    if payload.get("evaluation_config") != asdict(EvaluationConfig()):
        raise ValueError("FLNSR-2016 evaluator configuration changed")
    if payload.get("outcomes_opened") is not False:
        raise ValueError("FLNSR-2016 evaluator freeze opened outcomes")
    if payload.get("opened_windows") != [] or payload.get("mutable_parameters") != []:
        raise ValueError("FLNSR-2016 evaluator freeze is not sealed")
    if payload.get("execution_ohlc_rows_parsed_during_freeze") != 0:
        raise ValueError("FLNSR-2016 evaluator freeze parsed OHLC")
    if payload.get("funding_rows_parsed_during_freeze") != 0:
        raise ValueError("FLNSR-2016 evaluator freeze parsed funding")
    if payload.get("simulation_run_during_freeze") is not False:
        raise ValueError("FLNSR-2016 evaluator freeze simulated an outcome")
    schedules = load_schedules()
    schedules[DELAY_CLOCK_NAME] = one_extra_bar_delay_schedule(schedules["primary"])
    for name, schedule in schedules.items():
        expected = payload["schedules"][name]
        if expected["schedule_hash"] != _schedule_hash(schedule):
            raise ValueError(f"FLNSR-2016 {name} schedule changed after freeze")
    return payload


def _verify_execution_manifests() -> None:
    market_manifest = _load_json(MARKET_MANIFEST)
    if market_manifest.get("combined_sha256") != MARKET_SHA256:
        raise ValueError("FLNSR-2016 market manifest no longer binds frozen source")
    funding_manifest = _load_json(FUNDING_MANIFEST)
    if funding_manifest.get("data", {}).get("sha256") != FUNDING_SHA256:
        raise ValueError("FLNSR-2016 funding manifest no longer binds frozen source")
    if funding_manifest.get("outcomes_opened") is not False:
        raise ValueError("FLNSR-2016 funding source manifest opened an outcome")


def _parse_funding_window_causal(
    path: str | Path, start: pd.Timestamp, end: pd.Timestamp
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Parse a funding prefix without splitting any sealed-boundary value row."""
    wanted = ("funding_time_utc", "symbol", "funding_rate", "settlement_mark_price")
    rows: list[tuple[Any, ...]] = []
    window_hash = hashlib.sha256()
    boundary_seen = False
    start_text = start.strftime("%Y-%m-%dT%H:%M:%S")
    end_text = end.strftime("%Y-%m-%dT%H:%M:%S")
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        header_line = handle.readline()
        header = header_line.rstrip("\r\n").split(",")
        positions = {column: header.index(column) for column in wanted}
        if positions["funding_time_utc"] != 1:
            raise ValueError("FLNSR-2016 funding timestamp column changed")
        for line in handle:
            # Only the timestamp prefix is inspected until the row is known to
            # belong to the opened window.  No funding value from the first
            # sealed-boundary row is split or converted.
            first_comma = line.find(",")
            second_comma = line.find(",", first_comma + 1)
            if first_comma < 0 or second_comma < 0:
                raise ValueError("FLNSR-2016 malformed funding row prefix")
            timestamp_text = line[first_comma + 1 : second_comma]
            if timestamp_text < start_text:
                continue
            if timestamp_text >= end_text:
                boundary_seen = True
                break
            fields = line.rstrip("\r\n").split(",")
            rows.append(
                (
                    timestamp_text,
                    fields[positions["symbol"]],
                    float(fields[positions["funding_rate"]]),
                    float(fields[positions["settlement_mark_price"]]),
                )
            )
            window_hash.update(line.encode("utf-8"))
    frame = pd.DataFrame(rows, columns=wanted)
    frame = frame.rename(columns={"funding_time_utc": "funding_time"})
    frame["funding_time"] = pd.to_datetime(
        frame["funding_time"], utc=True, errors="raise"
    )
    expected = pd.date_range(start, end, freq="8h", inclusive="left")
    actual = pd.DatetimeIndex(frame["funding_time"])
    if len(actual) != len(expected) or not actual.is_monotonic_increasing:
        raise ValueError("FLNSR-2016 funding window does not match the eight-hour grid")
    offsets_ms = (actual - expected).total_seconds().to_numpy(float) * 1_000.0
    if np.abs(offsets_ms).max(initial=0.0) > 60_000.0:
        raise ValueError("FLNSR-2016 funding timestamp exceeds frozen offset bound")
    if not frame["symbol"].eq("BTCUSDT").all():
        raise ValueError("FLNSR-2016 funding source changed symbol")
    values = frame[["funding_rate", "settlement_mark_price"]].to_numpy(float)
    if not np.isfinite(values).all() or (frame["settlement_mark_price"] <= 0.0).any():
        raise ValueError("FLNSR-2016 funding contains invalid values")
    return frame, {
        "rows": int(len(frame)),
        "first_timestamp": frame["funding_time"].min().isoformat(),
        "last_timestamp": frame["funding_time"].max().isoformat(),
        "window_line_sha256": window_hash.hexdigest(),
        "stopped_before_parsing_end_boundary": boundary_seen,
        "maximum_absolute_grid_offset_ms": float(
            np.abs(offsets_ms).max(initial=0.0)
        ),
        "nonzero_grid_offset_events": int(np.count_nonzero(offsets_ms)),
        "sealed_boundary_values_parsed": False,
    }


_parse_funding_window = _parse_funding_window_causal


def _execution_diagnostics(
    window: TimeWindow,
    market_diagnostics: dict[str, Any],
    funding_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    start, end = window
    return {
        "physical_window": [start.isoformat(), end.isoformat()],
        "market": market_diagnostics,
        "funding": funding_diagnostics,
        "full_market_file_sha256_not_recomputed": window == STAGE1,
        "full_funding_file_sha256_not_recomputed": window == STAGE1,
        "reason": (
            "stop before parsing the sealed stage boundary"
            if window == STAGE1
            else "Stage1 replay passed; complete 2020-2023 files are now authorized"
        ),
        "opened_prefix_hashes_verified": True,
    }


def load_stage1_execution_window() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    _verify_execution_manifests()
    market, market_diagnostics = _parse_market_window(MARKET, *STAGE1)
    funding, funding_diagnostics = _parse_funding_window(FUNDING, *STAGE1)
    if (
        market_diagnostics.get("window_line_sha256")
        != STAGE1_MARKET_WINDOW_LINE_SHA256
    ):
        raise ValueError("FLNSR-2016 Stage1 market prefix hash changed")
    if (
        funding_diagnostics.get("window_line_sha256")
        != STAGE1_FUNDING_WINDOW_LINE_SHA256
    ):
        raise ValueError("FLNSR-2016 Stage1 funding prefix hash changed")
    return (
        market,
        funding,
        _execution_diagnostics(STAGE1, market_diagnostics, funding_diagnostics),
    )


def load_stage2_execution_window() -> tuple[
    dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, Any]
]:
    freeze = verify_evaluator_freeze()
    stage1 = _verified_passing_stage1(expected_freeze_hash=freeze["manifest_hash"])
    # This is the only supported evaluator entry point that parses 2023.  It has
    # no externally supplied authorization flag: the proof is obtained above by
    # replaying Stage1 inside this call.  Low-level underscore-prefixed parsers
    # are trusted internals, as frozen explicitly in api_trust_boundary.
    _verify_execution_manifests()
    if _sha256(MARKET) != MARKET_SHA256:
        raise ValueError("FLNSR-2016 full market source hash changed")
    if _sha256(FUNDING) != FUNDING_SHA256:
        raise ValueError("FLNSR-2016 full funding source hash changed")
    market, market_diagnostics = _parse_market_window(MARKET, *STAGE2)
    funding, funding_diagnostics = _parse_funding_window(FUNDING, *STAGE2)
    diagnostics = _execution_diagnostics(
        STAGE2, market_diagnostics, funding_diagnostics
    )
    return stage1, market, funding, diagnostics


def monthly_cluster_signflip(
    trade_details: list[dict[str, Any]] | pd.DataFrame,
    *,
    draws: int = MONTHLY_CLUSTER_DRAWS,
    seed: int = MONTHLY_CLUSTER_SEED,
) -> dict[str, Any]:
    trades = (
        trade_details.copy()
        if isinstance(trade_details, pd.DataFrame)
        else pd.DataFrame(trade_details)
    )
    if trades.empty:
        return {
            "p_value_one_sided": 1.0,
            "cluster_count": 0,
            "draws": int(draws),
            "seed": int(seed),
            "observed_mean_net_return": 0.0,
            "monthly_net_return_sums": {},
        }
    if draws <= 0:
        raise ValueError("FLNSR-2016 monthly sign-flip draws must be positive")
    entry = pd.to_datetime(trades["entry_time"], utc=True, errors="raise")
    net_return = pd.to_numeric(trades["net_return"], errors="raise")
    if not np.isfinite(net_return.to_numpy(float)).all():
        raise ValueError("FLNSR-2016 monthly sign-flip input is nonfinite")
    keys = entry.dt.strftime("%Y-%m")
    monthly = net_return.groupby(keys).sum()
    values = monthly.to_numpy(float)
    observed = float(net_return.mean())
    generator = np.random.default_rng(seed)
    exceed = 0
    remaining = draws
    while remaining:
        batch = min(10_000, remaining)
        signs = generator.choice((-1.0, 1.0), size=(batch, len(values)))
        null = (signs @ values) / float(len(trades))
        exceed += int(np.count_nonzero(null >= observed - 1e-15))
        remaining -= batch
    return {
        "p_value_one_sided": float((1 + exceed) / (draws + 1)),
        "cluster_count": int(len(values)),
        "draws": int(draws),
        "seed": int(seed),
        "observed_mean_net_return": observed,
        "monthly_net_return_sums": {
            str(label): float(value) for label, value in monthly.items()
        },
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
    metrics = simulate_schedule(
        market,
        funding,
        schedule,
        period_start=window[0],
        period_end=window[1],
        cost_rate=cost_rate,
        cfg=cfg,
    )
    metrics["monthly_cluster_signflip"] = monthly_cluster_signflip(
        metrics["trade_details"]
    )
    return metrics


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
        name: _simulate(
            market,
            funding,
            schedule,
            window=window,
            cost_rate=cost_rate,
            cfg=cfg,
        )
        for name, window in subperiods.items()
    }


def _headline(metrics: dict[str, Any]) -> dict[str, Any]:
    monthly = metrics["monthly_cluster_signflip"]
    return {
        "absolute_return_pct": metrics["absolute_return_pct"],
        "cagr_pct": metrics["cagr_pct"],
        "strict_mdd_pct": metrics["strict_mdd_pct"],
        "cagr_to_strict_mdd": metrics["cagr_to_strict_mdd"],
        "trades": metrics["trades"],
        "mean_gross_underlying_bp": metrics["mean_gross_underlying_bp"],
        "monthly_cluster_signflip_p": monthly["p_value_one_sided"],
        "monthly_clusters": monthly["cluster_count"],
    }


def _performance_gates(
    primary: dict[str, Any],
    stress: dict[str, Any],
    delayed: dict[str, Any],
    subperiods: dict[str, dict[str, Any]],
    controls: dict[str, dict[str, Any]],
) -> tuple[dict[str, bool], dict[str, float]]:
    margins = {
        name: float(
            primary["mean_gross_underlying_bp"]
            - controls[name]["mean_gross_underlying_bp"]
        )
        for name in MECHANISM_CONTROLS
    }
    gates = {
        "absolute_return_positive": primary["absolute_return_pct"] > 0.0,
        "cagr_to_strict_mdd_at_least_3": primary["cagr_to_strict_mdd"] >= 3.0,
        "strict_mdd_at_most_15pct": primary["strict_mdd_pct"] <= 15.0,
        "stress_cost_absolute_return_positive": stress["absolute_return_pct"] > 0.0,
        "one_extra_bar_delay_absolute_return_positive": (
            delayed["absolute_return_pct"] > 0.0
        ),
        "mean_gross_underlying_at_least_30bp": (
            primary["mean_gross_underlying_bp"] >= MINIMUM_MEAN_GROSS_BP
        ),
        "monthly_cluster_signflip_p_within_limit": (
            primary["monthly_cluster_signflip_p"] <= MONTHLY_CLUSTER_P_MAX
        ),
        "each_subperiod_absolute_return_positive": all(
            metrics["absolute_return_pct"] > 0.0 for metrics in subperiods.values()
        ),
        "primary_mean_gross_margin_over_each_component_at_least_5bp": all(
            margin >= MINIMUM_COMPONENT_MARGIN_BP for margin in margins.values()
        ),
    }
    return gates, margins


def _evaluate_candidate(
    schedules: dict[str, pd.DataFrame],
    market: pd.DataFrame,
    funding: pd.DataFrame,
    *,
    window: TimeWindow,
    subperiods: dict[str, TimeWindow],
) -> dict[str, Any]:
    cfg = EvaluationConfig()
    base_raw = {
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
    stress_raw = _simulate(
        market,
        funding,
        schedules["primary"],
        window=window,
        cost_rate=cfg.stress_cost_notional_per_side,
        cfg=cfg,
    )
    delayed_schedule = one_extra_bar_delay_schedule(schedules["primary"])
    delayed_raw = _simulate(
        market,
        funding,
        delayed_schedule,
        window=window,
        cost_rate=cfg.base_cost_notional_per_side,
        cfg=cfg,
    )
    subperiod_raw = _simulate_subperiods(
        market,
        funding,
        schedules["primary"],
        subperiods,
        cost_rate=cfg.base_cost_notional_per_side,
        cfg=cfg,
    )
    primary = _headline(base_raw["primary"])
    stress = _headline(stress_raw)
    delayed = _headline(delayed_raw)
    subperiod_headlines = {
        name: _headline(metrics) for name, metrics in subperiod_raw.items()
    }
    controls = {
        name: _headline(base_raw[name]) for name in ALL_CLOCK_NAMES if name != "primary"
    }
    gates, margins = _performance_gates(
        primary, stress, delayed, subperiod_headlines, controls
    )
    return {
        "policy_id": POLICY_ID,
        "primary": primary,
        "stress_10bp_per_side": stress,
        "one_extra_bar_delay": delayed,
        "subperiods": subperiod_headlines,
        "controls": controls,
        "component_mean_gross_margins_bp": margins,
        "entry_distribution": {
            name: _entry_distribution(schedule, window)
            for name, schedule in schedules.items()
        },
        "falsification_controls_report_only": True,
        "gates": gates,
        "qualified": all(gates.values()),
    }


def _build_stage1_core(
    *,
    freeze_manifest_hash: str,
    schedules: dict[str, pd.DataFrame],
    market: pd.DataFrame,
    funding: pd.DataFrame,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    candidate = _evaluate_candidate(
        schedules,
        market,
        funding,
        window=STAGE1,
        subperiods=STAGE1_SUBPERIODS,
    )
    passed = bool(candidate["qualified"])
    return {
        "protocol_version": STAGE1_PROTOCOL,
        "policy_id": POLICY_ID,
        "stage": STAGE1_ID,
        "as_of_date": "2026-07-23",
        "evaluator_freeze_manifest_hash": freeze_manifest_hash,
        "evaluator_source_sha256": _sha256(EVALUATOR_SOURCE),
        "config": asdict(EvaluationConfig()),
        "monthly_cluster_draws": MONTHLY_CLUSTER_DRAWS,
        "monthly_cluster_seed": MONTHLY_CLUSTER_SEED,
        "physical_source_diagnostics": diagnostics,
        "stage1_window": [value.isoformat() for value in STAGE1],
        "candidate": candidate,
        "stage1_passed": passed,
        "advance_to_stage2": passed,
        "2023_outcomes_opened": False,
        "2023_execution_rows_parsed": 0,
        "2023_funding_rows_parsed": 0,
        "opened_windows": STAGE1_OPENED_WINDOWS,
        "sealed_windows": STAGE1_SEALED_WINDOWS,
        "disposition": (
            "PASS_STAGE1_OPEN_2023_ONCE"
            if passed
            else "REJECT_STAGE1_KEEP_2023_AND_LATER_SEALED"
        ),
    }


def evaluate_stage1() -> dict[str, Any]:
    freeze = verify_evaluator_freeze()
    schedules = load_schedules()
    market, funding, diagnostics = load_stage1_execution_window()
    payload = _seal(
        _build_stage1_core(
            freeze_manifest_hash=freeze["manifest_hash"],
            schedules=schedules,
            market=market,
            funding=funding,
            diagnostics=diagnostics,
        )
    )
    _write_once_text(
        STAGE1_OUTPUT,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        label="Stage1 report",
    )
    _write_once_text(STAGE1_DOC, render_stage_doc(payload), label="Stage1 document")
    return payload


def _validate_stage1_identity(
    payload: dict[str, Any], *, expected_freeze_hash: str
) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != _canonical_hash(core):
        raise ValueError("FLNSR-2016 Stage1 result hash mismatch")
    if (
        payload.get("protocol_version") != STAGE1_PROTOCOL
        or payload.get("policy_id") != POLICY_ID
        or payload.get("stage") != STAGE1_ID
    ):
        raise ValueError("FLNSR-2016 Stage1 result identity changed")
    if payload.get("evaluator_freeze_manifest_hash") != expected_freeze_hash:
        raise ValueError("FLNSR-2016 Stage1 is not bound to this evaluator freeze")
    if payload.get("evaluator_source_sha256") != _sha256(EVALUATOR_SOURCE):
        raise ValueError("FLNSR-2016 Stage1 used another evaluator")
    if payload.get("config") != asdict(EvaluationConfig()):
        raise ValueError("FLNSR-2016 Stage1 evaluator configuration changed")
    if payload.get("monthly_cluster_draws") != MONTHLY_CLUSTER_DRAWS:
        raise ValueError("FLNSR-2016 Stage1 monthly draws changed")
    if payload.get("monthly_cluster_seed") != MONTHLY_CLUSTER_SEED:
        raise ValueError("FLNSR-2016 Stage1 monthly seed changed")
    if payload.get("stage1_window") != [value.isoformat() for value in STAGE1]:
        raise ValueError("FLNSR-2016 Stage1 logical window changed")
    if payload.get("opened_windows") != STAGE1_OPENED_WINDOWS:
        raise ValueError("FLNSR-2016 Stage1 opened-window contract changed")
    if payload.get("sealed_windows") != STAGE1_SEALED_WINDOWS:
        raise ValueError("FLNSR-2016 Stage1 sealed-window contract changed")
    if payload.get("2023_outcomes_opened") is not False:
        raise ValueError("FLNSR-2016 Stage1 already opened 2023 outcomes")
    if payload.get("2023_execution_rows_parsed") != 0:
        raise ValueError("FLNSR-2016 Stage1 parsed 2023 execution rows")
    if payload.get("2023_funding_rows_parsed") != 0:
        raise ValueError("FLNSR-2016 Stage1 parsed 2023 funding rows")
    diagnostics = payload.get("physical_source_diagnostics")
    if not isinstance(diagnostics, dict) or diagnostics.get("physical_window") != [
        value.isoformat() for value in STAGE1
    ]:
        raise ValueError("FLNSR-2016 Stage1 physical window changed")
    if not isinstance(payload.get("candidate"), dict):
        raise ValueError("FLNSR-2016 Stage1 candidate is malformed")
    passed = payload.get("candidate", {}).get("qualified")
    if not isinstance(passed, bool):
        raise ValueError("FLNSR-2016 Stage1 qualification is malformed")
    if payload.get("stage1_passed") is not passed:
        raise ValueError("FLNSR-2016 Stage1 pass flag is not reproducible")
    if payload.get("advance_to_stage2") is not passed:
        raise ValueError("FLNSR-2016 Stage1 advance flag is not reproducible")
    expected_disposition = (
        "PASS_STAGE1_OPEN_2023_ONCE"
        if passed
        else "REJECT_STAGE1_KEEP_2023_AND_LATER_SEALED"
    )
    if payload.get("disposition") != expected_disposition:
        raise ValueError("FLNSR-2016 Stage1 disposition changed")


def _verified_passing_stage1(*, expected_freeze_hash: str) -> dict[str, Any]:
    if not STAGE1_OUTPUT.exists():
        raise ValueError("FLNSR-2016 Stage1 report is absent; 2023 remains sealed")
    payload = _load_json(STAGE1_OUTPUT)
    _validate_stage1_identity(payload, expected_freeze_hash=expected_freeze_hash)

    # A self-sealed JSON flag is not trusted.  Reopen only the already-observed
    # train prefix and require exact report replay before parsing any 2023 row.
    schedules = load_schedules()
    market, funding, diagnostics = load_stage1_execution_window()
    expected = _seal(
        _build_stage1_core(
            freeze_manifest_hash=expected_freeze_hash,
            schedules=schedules,
            market=market,
            funding=funding,
            diagnostics=diagnostics,
        )
    )
    if payload != expected:
        raise ValueError("FLNSR-2016 Stage1 is not reproducible from frozen sources")
    if payload["stage1_passed"] is not True:
        raise ValueError("FLNSR-2016 Stage1 failed; 2023 remains sealed")
    return payload


def evaluate_stage2() -> dict[str, Any]:
    freeze = verify_evaluator_freeze()
    stage1, market, funding, diagnostics = load_stage2_execution_window()
    schedules = load_schedules()
    candidate = _evaluate_candidate(
        schedules,
        market,
        funding,
        window=STAGE2,
        subperiods=STAGE2_SUBPERIODS,
    )
    passed = bool(candidate["qualified"])
    core = {
        "protocol_version": STAGE2_PROTOCOL,
        "policy_id": POLICY_ID,
        "stage": STAGE2_ID,
        "as_of_date": "2026-07-23",
        "evaluator_freeze_manifest_hash": freeze["manifest_hash"],
        "stage1_manifest_hash": stage1["manifest_hash"],
        "evaluator_source_sha256": _sha256(EVALUATOR_SOURCE),
        "physical_source_diagnostics": diagnostics,
        "stage2_window": [value.isoformat() for value in STAGE2],
        "candidate": candidate,
        "stage2_passed": passed,
        "advance_to_2024": passed,
        "sealed_windows": ["2024", "2025", "2026_ytd"],
        "disposition": (
            "PASS_SELECTION_FREEZE_2024_EVALUATOR"
            if passed
            else "REJECT_SELECTION_KEEP_2024_AND_LATER_SEALED"
        ),
    }
    payload = _seal(core)
    _write_once_text(
        STAGE2_OUTPUT,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        label="Stage2 report",
    )
    _write_once_text(STAGE2_DOC, render_stage_doc(payload), label="Stage2 document")
    return payload


def _validate_finite_headline(metrics: dict[str, Any]) -> None:
    for key, value in metrics.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"FLNSR-2016 nonnumeric headline field: {key}")
        if not math.isfinite(float(value)):
            raise ValueError(f"FLNSR-2016 nonfinite headline field: {key}")


def render_stage_doc(report: dict[str, Any]) -> str:
    candidate = report["candidate"]
    metric = candidate["primary"]
    _validate_finite_headline(metric)
    failed = [name for name, passed in candidate["gates"].items() if not passed]
    stage = str(report["stage"])
    title = "2020–2022 train" if stage == STAGE1_ID else "sealed 2023 selection"
    passed = report["stage1_passed"] if stage == STAGE1_ID else report["stage2_passed"]
    lines = [
        f"# FLNSR-2016 — {title}",
        "",
        f"- Absolute return: **{metric['absolute_return_pct']:.2f}%**",
        f"- CAGR: **{metric['cagr_pct']:.2f}%**",
        f"- Strict MDD: **{metric['strict_mdd_pct']:.2f}%**",
        f"- CAGR / strict MDD: **{metric['cagr_to_strict_mdd']:.2f}**",
        f"- Trades: **{metric['trades']}**",
        f"- Mean gross underlying: **{metric['mean_gross_underlying_bp']:.2f} bp**",
        f"- Monthly cluster sign-flip p: **{metric['monthly_cluster_signflip_p']:.4f}**",
        f"- Stage passed: **{passed}**",
        f"- Failed gates: **{failed or 'none'}**",
        "",
        "## Subperiod absolute returns",
        "",
    ]
    for name, subperiod in candidate["subperiods"].items():
        lines.append(f"- {name}: **{subperiod['absolute_return_pct']:.2f}%**")
    lines.extend(
        [
            "",
            "Falsification controls are diagnostics only and cannot rescue or reject the primary.",
            "",
        ]
    )
    return "\n".join(lines)


def _print_stage(report: dict[str, Any]) -> None:
    print(json.dumps(report["candidate"]["primary"], indent=2, sort_keys=True))
    stage = report["stage"]
    passed = report["stage1_passed"] if stage == STAGE1_ID else report["stage2_passed"]
    print(f"{stage}_passed={passed}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--freeze", action="store_true")
    action.add_argument("--stage1", action="store_true")
    action.add_argument("--stage2", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        print(json.dumps(freeze_evaluator(), indent=2, sort_keys=True))
    elif args.stage1:
        _print_stage(evaluate_stage1())
    else:
        _print_stage(evaluate_stage2())


if __name__ == "__main__":
    main()
