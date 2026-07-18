"""Sequential strict evaluator for frozen H8DM-1.

Freezing this evaluator opens no execution outcome. Stage 1 physically parses
only 2020-2022. The 2023 execution window can be opened only after an exact,
hash-bound replay of a passing Stage-1 result. The 2024+ window stays sealed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from training import evaluate_fiat_quote_participation_rotation as strict_engine
from training import fed_h8_deposit_migration_clock as clock
from training import preregister_fed_h8_deposit_migration as prereg


SUPPORT_COMMIT = "6048a45"
STATIC_INPUT_SHA256 = {
    "training/preregister_fed_h8_deposit_migration.py": "4284964daec7f75711a6207404dc2787c004f6b26b9d2a9806dd80bd292a718b",
    "training/fed_h8_deposit_migration_clock.py": "b4d92efc4091e3c6720912dfbd4050702391ad7228c93a53c0e7194e2e7ce793",
    "docs/fed-h8-deposit-migration-preregistration-2026-07-18.md": "bb2a0d0b47557617e296d3c49d92d56244aa7ef8f88f5bed4e2e4a0571d1a0ae",
    "results/fed_h8_deposit_migration_preregistration_2026-07-18.json": "0705042e6fceb5e183e5967be846bc106ea860f642fd44cba72dfa214eb09432",
    "results/fed_h8_deposit_migration_preregistered_clock_2026-07-18.csv.gz": "20405f79b86861adcc784c81223baae1c40fdf3c73edda339578471a6a6d1b40",
    "training/build_fed_h8_deposit_migration_support.py": "55a6aad1c7996c6f45dbdd2c79ceaefa4f59e308f7de5359310abb6de23d81e4",
    "docs/fed-h8-deposit-migration-support-2026-07-18.md": "52bdb6385fc8bc09f2b6f72dfd5bf876196787bfc9f8727025542fff7d012211",
    "results/fed_h8_deposit_migration_support_2026-07-18.json": "79c2042f00f54f525275dd5e1536adb8de790e0271a1c0765e41bd2b7570f057",
    "results/fed_h8_deposit_migration_clocks_2026-07-18.csv.gz": "1d1774123480868e36b8f76e28ee11c918f18d6694ce8f16056f338710f040ba",
    "training/evaluate_fiat_quote_participation_rotation.py": "e309f5217f033d57d2eadfec936843e736ce287f5c47f957c0ac6f0c71879c23",
    "data/binance_um_kline_reference_btc_2020_2023/build_manifest.json": "85d77ef584f73461e9b5a43fd63543b5728fa76354081983e7980aaa4c125af4",
    "results/binance_um_btcusdt_funding_marks_2020_2023_manifest_2026-07-17.json": "a0b2d27e1aa8cf2d9ab8cb659b598ee0a6d7bd25401c9e10ae92d1a74415845b",
}
MARKET_DATA_SHA256 = "e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d"
FUNDING_DATA_SHA256 = "3284bbb6bb67946acb673c6b67459543e217f752589e1d47b6c7c3b659f733e6"

PREREGISTRATION = Path(prereg.DEFAULT_OUTPUT)
SUPPORT_RESULT = Path("results/fed_h8_deposit_migration_support_2026-07-18.json")
CLOCKS = Path("results/fed_h8_deposit_migration_clocks_2026-07-18.csv.gz")
MARKET = Path(prereg.MARKET_PATH)
FUNDING = Path("data/binance_um_btcusdt_funding_marks_2020_2023.csv.gz")
MARKET_MANIFEST = Path(prereg.MARKET_MANIFEST)
FUNDING_MANIFEST = Path(
    "results/binance_um_btcusdt_funding_marks_2020_2023_manifest_2026-07-17.json"
)
EVALUATOR_SOURCE = Path("training/evaluate_fed_h8_deposit_migration.py")
EVALUATOR_FREEZE = Path(
    "results/fed_h8_deposit_migration_evaluator_freeze_2026-07-18.json"
)
STAGE1_OUTPUT = Path(
    "results/fed_h8_deposit_migration_stage1_2020_2022_2026-07-18.json"
)
STAGE1_DOC = Path("docs/fed-h8-deposit-migration-stage1-2020-2022-2026-07-18.md")
STAGE2_OUTPUT = Path("results/fed_h8_deposit_migration_stage2_2023_2026-07-18.json")
STAGE2_DOC = Path("docs/fed-h8-deposit-migration-stage2-2023-2026-07-18.md")

TimeWindow = tuple[pd.Timestamp, pd.Timestamp]


def _utc(value: str) -> pd.Timestamp:
    return cast(pd.Timestamp, pd.Timestamp(value, tz="UTC"))


def _timestamp(value: Any) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result is pd.NaT:
        raise ValueError("H8DM-1 timestamp is NaT")
    return cast(pd.Timestamp, result)


STAGE1: TimeWindow = (_utc("2020-01-01"), _utc("2023-01-01"))
STAGE1_SUBPERIODS: dict[str, TimeWindow] = {
    "2020": (_utc("2020-01-01"), _utc("2021-01-01")),
    "2021": (_utc("2021-01-01"), _utc("2022-01-01")),
    "2022": (_utc("2022-01-01"), _utc("2023-01-01")),
}
STAGE2: TimeWindow = (_utc("2023-01-01"), _utc("2024-01-01"))
POOLED: TimeWindow = (_utc("2020-01-01"), _utc("2024-01-01"))
STAGE2_SUBPERIODS: dict[str, TimeWindow] = {
    "2023_h1": (_utc("2023-01-01"), _utc("2023-07-01")),
    "2023_h2": (_utc("2023-07-01"), _utc("2024-01-01")),
}
SINGLE_COMPONENT_CONTROLS = (
    "migration_only",
    "borrowings_only",
    "cash_only",
)
MECHANISM_CONTROLS = (*SINGLE_COMPONENT_CONTROLS, "no_agreement", "nsa_primary")
FALSIFICATION_CONTROLS = (
    "direction_flip",
    "one_week_delay",
    "four_week_placebo",
)
ALL_CLOCK_NAMES = (
    "primary",
    *MECHANISM_CONTROLS,
    *FALSIFICATION_CONTROLS,
)

EvaluationConfig = strict_engine.EvaluationConfig
_parse_market_window = strict_engine._parse_market_window
_parse_funding_window = strict_engine._parse_funding_window


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _seal(core: dict[str, Any]) -> dict[str, Any]:
    return {**core, "manifest_hash": _canonical_hash(core)}


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"H8DM-1 expected a JSON object: {path}")
    return payload


def _verify_registration(registration: dict[str, Any]) -> None:
    core = {key: value for key, value in registration.items() if key != "manifest_hash"}
    if registration.get("manifest_hash") != prereg.canonical_hash(core):
        raise ValueError("H8DM-1 preregistration manifest mismatch")
    if registration.get("policy_id") != "H8DM-1":
        raise ValueError("H8DM-1 preregistration identity changed")
    if registration.get("opened_outcome_windows") != []:
        raise ValueError("H8DM-1 preregistration opened outcomes")


def _verify_evaluation_contract(registration: dict[str, Any]) -> EvaluationConfig:
    cfg = EvaluationConfig()
    policy = registration["policy"]
    for key in (
        "leverage",
        "base_cost_notional_per_side",
        "stress_cost_notional_per_side",
    ):
        if asdict(cfg)[key] != policy[key]:
            raise ValueError(f"H8DM-1 strict engine differs on {key}")
    if cfg.cluster_draws != 20_000 or cfg.cluster_seed != 20_260_717:
        raise ValueError("H8DM-1 sign-flip configuration changed")
    if cfg.mdd_denominator_floor != 1e-9:
        raise ValueError("H8DM-1 MDD denominator floor changed")
    stage = registration["selection_protocol"]
    expected_stage1 = {
        "absolute_return_positive": True,
        "best_single_component_ratio_margin_at_least": 0.25,
        "cagr_to_strict_mdd_at_least": 3.0,
        "each_calendar_year_absolute_return_positive": True,
        "each_calendar_year_minimum_trades": 18,
        "mean_gross_underlying_bp_at_least": 40.0,
        "minimum_long_trades": 24,
        "minimum_short_trades": 24,
        "minimum_trades": 72,
        "stress_cost_absolute_return_positive": True,
        "strict_mdd_pct_at_most": 15.0,
        "weekly_cluster_signflip_p_at_most": 0.10,
    }
    expected_stage2 = {
        "absolute_return_positive": True,
        "cagr_to_strict_mdd_at_least": 3.0,
        "each_half_absolute_return_nonnegative": True,
        "each_half_minimum_trades": 8,
        "mean_gross_underlying_bp_at_least": 40.0,
        "minimum_long_trades": 8,
        "minimum_short_trades": 8,
        "minimum_trades": 20,
        "pooled_stage1_stage2_ratio_at_least": 3.0,
        "pooled_weekly_cluster_signflip_p_at_most": 0.05,
        "stress_cost_absolute_return_positive": True,
        "strict_mdd_pct_at_most": 15.0,
        "weekly_cluster_signflip_p_at_most": 0.10,
    }
    if stage["stage1_gates"] != expected_stage1:
        raise ValueError("H8DM-1 Stage1 gate contract changed")
    if stage["stage2_gates"] != expected_stage2:
        raise ValueError("H8DM-1 Stage2 gate contract changed")
    controls = registration["controls"]
    if tuple(controls["mechanism"]) != MECHANISM_CONTROLS:
        raise ValueError("H8DM-1 mechanism-control contract changed")
    if tuple(controls["falsification"]) != FALSIFICATION_CONTROLS:
        raise ValueError("H8DM-1 falsification-control contract changed")
    return cfg


def _verify_static_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    for path, expected in STATIC_INPUT_SHA256.items():
        if _sha256(path) != expected:
            raise ValueError(f"H8DM-1 frozen input changed: {path}")
    registration = _load_json(PREREGISTRATION)
    _verify_registration(registration)
    _verify_evaluation_contract(registration)
    support = _load_json(SUPPORT_RESULT)
    support_core = {
        key: value for key, value in support.items() if key != "manifest_hash"
    }
    if support.get("manifest_hash") != _canonical_hash(support_core):
        raise ValueError("H8DM-1 support manifest changed")
    if support.get("policy_id") != "H8DM-1":
        raise ValueError("H8DM-1 support identity changed")
    if support.get("outcomes_opened") is not False:
        raise ValueError("H8DM-1 support opened outcomes")
    if support.get("outcome_sources_opened") != []:
        raise ValueError("H8DM-1 support opened outcome sources")
    if support.get("market_or_funding_rows_opened") != 0:
        raise ValueError("H8DM-1 support opened execution rows")
    if support.get("support_passed") is not True:
        raise ValueError("H8DM-1 support failed")
    if support.get("advance_to_stage1_outcomes") is not True:
        raise ValueError("H8DM-1 support did not authorize Stage1")
    if support.get("preregistration_manifest_hash") != registration["manifest_hash"]:
        raise ValueError("H8DM-1 support is not bound to preregistration")
    if support.get("clocks", {}).get("sha256") != STATIC_INPUT_SHA256[str(CLOCKS)]:
        raise ValueError("H8DM-1 support is not bound to the clock ledger")
    return registration, support


def _schedule_frame(name: str, rows: pd.DataFrame) -> pd.DataFrame:
    frame = rows.copy()
    frame["clock_name"] = name
    frame["signal_day"] = pd.to_datetime(frame["signal_time"], utc=True, errors="raise")
    frame["entry_time"] = pd.to_datetime(frame["entry_time"], utc=True, errors="raise")
    frame["exit_time"] = pd.to_datetime(frame["exit_time"], utc=True, errors="raise")
    frame["side"] = frame["side"].astype("int8")
    schedule = frame[
        [
            "clock_name",
            "release_date",
            "signal_day",
            "entry_time",
            "exit_time",
            "side",
        ]
    ]
    order = np.argsort(np.asarray(schedule["entry_time"]))
    return schedule.iloc[order].reset_index(drop=True)


def load_schedules() -> dict[str, pd.DataFrame]:
    _, support = _verify_static_inputs()
    ledger = pd.read_csv(CLOCKS)
    if set(ledger["control"]) != set(ALL_CLOCK_NAMES):
        raise ValueError("H8DM-1 support clock family changed")
    schedules = {
        name: _schedule_frame(name, ledger.loc[ledger["control"].eq(name)])
        for name in ALL_CLOCK_NAMES
    }
    expected_counts = support["clocks"]["counts"]
    for name, schedule in schedules.items():
        if len(schedule) != int(expected_counts[name]):
            raise ValueError(f"H8DM-1 {name} source count changed")
        if not bool(schedule["side"].isin((-1, 1)).all()):
            raise ValueError(f"H8DM-1 {name} side changed")
        delay_weeks = (
            1 if name == "one_week_delay" else 4 if name == "four_week_placebo" else 0
        )
        for row in schedule.to_dict(orient="records"):
            signal = _timestamp(row["signal_day"])
            expected_entry, expected_exit = clock._execution_times(
                signal.to_pydatetime(), delay_weeks=delay_weeks
            )
            if _timestamp(row["entry_time"]) != pd.Timestamp(expected_entry):
                raise ValueError(f"H8DM-1 {name} entry clock changed")
            if _timestamp(row["exit_time"]) != pd.Timestamp(expected_exit):
                raise ValueError(f"H8DM-1 {name} exit clock changed")
        if len(schedule) > 1:
            entries = schedule["entry_time"].iloc[1:].reset_index(drop=True)
            exits = schedule["exit_time"].iloc[:-1].reset_index(drop=True)
            if not bool(entries.ge(exits).all()):
                raise ValueError(f"H8DM-1 {name} schedule overlaps")
    return schedules


def _schedule_hash(frame: pd.DataFrame) -> str:
    rows = [
        {
            "clock_name": str(row["clock_name"]),
            "release_date": str(row["release_date"]),
            "signal_day": _timestamp(row["signal_day"]).isoformat(),
            "entry_time": _timestamp(row["entry_time"]).isoformat(),
            "exit_time": _timestamp(row["exit_time"]).isoformat(),
            "side": int(row["side"]),
        }
        for row in frame.to_dict(orient="records")
    ]
    return _canonical_hash(rows)


def _window_schedule(frame: pd.DataFrame, window: TimeWindow) -> pd.DataFrame:
    start, end = window
    return frame.loc[
        frame["signal_day"].ge(start)
        & frame["entry_time"].ge(start)
        & frame["exit_time"].le(end)
    ].copy()


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


def _source_signal_distribution(
    frame: pd.DataFrame, window: TimeWindow
) -> dict[str, Any]:
    start, end = window
    selected = frame.loc[frame["signal_day"].ge(start) & frame["signal_day"].lt(end)]
    months = selected["entry_time"].dt.strftime("%Y-%m").value_counts()
    maximum = int(months.max()) if len(months) else 0
    return {
        "trades": int(len(selected)),
        "longs": int(selected["side"].eq(1).sum()),
        "shorts": int(selected["side"].eq(-1).sum()),
        "max_single_month_count": maximum,
        "max_single_month_share": maximum / len(selected) if len(selected) else 0.0,
    }


def _schedule_record(frame: pd.DataFrame) -> dict[str, Any]:
    name = str(frame["clock_name"].iloc[0])
    delay_weeks = (
        1 if name == "one_week_delay" else 4 if name == "four_week_placebo" else 0
    )
    execution_exact = True
    for row in frame.to_dict(orient="records"):
        expected_entry, expected_exit = clock._execution_times(
            _timestamp(row["signal_day"]).to_pydatetime(),
            delay_weeks=delay_weeks,
        )
        execution_exact &= _timestamp(row["entry_time"]) == pd.Timestamp(expected_entry)
        execution_exact &= _timestamp(row["exit_time"]) == pd.Timestamp(expected_exit)
    stage1_source = _source_signal_distribution(frame, STAGE1)
    stage1_execution = _entry_distribution(frame, STAGE1)
    stage2_source = _source_signal_distribution(frame, STAGE2)
    stage2_execution = _entry_distribution(frame, STAGE2)
    return {
        "events": int(len(frame)),
        "schedule_hash": _schedule_hash(frame),
        "first_entry": _timestamp(frame["entry_time"].min()).isoformat(),
        "last_exit": _timestamp(frame["exit_time"].max()).isoformat(),
        "execution_clock_exact": bool(execution_exact),
        "globally_nonoverlapping": bool(
            frame["entry_time"]
            .iloc[1:]
            .reset_index(drop=True)
            .ge(frame["exit_time"].iloc[:-1].reset_index(drop=True))
            .all()
        ),
        "stage1_source_signal_distribution": stage1_source,
        "stage1_executable_distribution": stage1_execution,
        "stage1_boundary_excluded_trades": (
            stage1_source["trades"] - stage1_execution["trades"]
        ),
        "sealed_2023_source_signal_distribution": stage2_source,
        "sealed_2023_executable_distribution": stage2_execution,
        "sealed_2023_boundary_excluded_trades": (
            stage2_source["trades"] - stage2_execution["trades"]
        ),
    }


def _evaluator_freeze_core() -> dict[str, Any]:
    registration, support = _verify_static_inputs()
    schedules = load_schedules()
    records = {name: _schedule_record(schedule) for name, schedule in schedules.items()}
    if not all(
        record[check]
        for record in records.values()
        for check in ("execution_clock_exact", "globally_nonoverlapping")
    ):
        raise ValueError("H8DM-1 evaluator schedule invariant failed")
    return {
        "protocol_version": "fed_h8_deposit_migration_evaluator_v1",
        "policy_id": "H8DM-1",
        "as_of_date": prereg.AS_OF_DATE,
        "support_commit": SUPPORT_COMMIT,
        "preregistration_manifest_hash": registration["manifest_hash"],
        "support_manifest_hash": support["manifest_hash"],
        "evaluator_source": str(EVALUATOR_SOURCE),
        "evaluator_source_sha256": _sha256(EVALUATOR_SOURCE),
        "strict_engine_sha256": STATIC_INPUT_SHA256[
            "training/evaluate_fiat_quote_participation_rotation.py"
        ],
        "evaluation_config": asdict(_verify_evaluation_contract(registration)),
        "execution_source_contract": {
            "market_path": str(MARKET),
            "market_manifest": str(MARKET_MANIFEST),
            "market_data_sha256": MARKET_DATA_SHA256,
            "funding_path": str(FUNDING),
            "funding_manifest": str(FUNDING_MANIFEST),
            "funding_data_sha256": FUNDING_DATA_SHA256,
        },
        "static_inputs": STATIC_INPUT_SHA256,
        "control_schedules": records,
        "opened_windows": [],
        "sealed_windows": ["stage1_2020_2022", "stage2_2023", "2024_plus"],
        "execution_ohlc_rows_parsed_during_freeze": 0,
        "funding_rows_parsed_during_freeze": 0,
        "simulation_run_during_freeze": False,
        "mutable_parameters": [],
    }


def freeze_evaluator(output_path: str | Path = EVALUATOR_FREEZE) -> dict[str, Any]:
    report = _seal(_evaluator_freeze_core())
    prereg.write_once(output_path, (json.dumps(report, indent=2) + "\n").encode())
    return report


def verify_evaluator_freeze(path: str | Path = EVALUATOR_FREEZE) -> dict[str, Any]:
    report = _load_json(path)
    expected = _seal(_evaluator_freeze_core())
    if report != expected:
        raise ValueError("H8DM-1 evaluator freeze contract changed")
    return report


def _verify_execution_manifests() -> None:
    market_manifest = _load_json(MARKET_MANIFEST)
    if market_manifest.get("combined_output") != str(MARKET):
        raise ValueError("H8DM-1 market manifest path changed")
    if market_manifest.get("combined_sha256") != MARKET_DATA_SHA256:
        raise ValueError("H8DM-1 market manifest data hash changed")
    if market_manifest.get("rows") != 420_768:
        raise ValueError("H8DM-1 market manifest row count changed")
    if market_manifest.get("protocol", {}).get("outcomes_opened") is not False:
        raise ValueError("H8DM-1 market manifest opened an outcome")
    funding_manifest = _load_json(FUNDING_MANIFEST)
    funding_data = funding_manifest.get("data", {})
    if funding_data.get("path") != str(FUNDING):
        raise ValueError("H8DM-1 funding manifest path changed")
    if funding_data.get("sha256") != FUNDING_DATA_SHA256:
        raise ValueError("H8DM-1 funding manifest data hash changed")
    if funding_data.get("rows") != 4_383:
        raise ValueError("H8DM-1 funding manifest row count changed")
    if funding_manifest.get("outcomes_opened") is not False:
        raise ValueError("H8DM-1 funding manifest opened an outcome")


def load_execution_window(
    window: TimeWindow,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    _verify_execution_manifests()
    start, end = window
    market, market_diagnostics = _parse_market_window(MARKET, start, end)
    funding, funding_diagnostics = _parse_funding_window(FUNDING, start, end)
    return (
        market,
        funding,
        {
            "physical_window": [start.isoformat(), end.isoformat()],
            "market": market_diagnostics,
            "funding": funding_diagnostics,
            "full_market_file_sha256_not_recomputed": True,
            "full_funding_file_sha256_not_recomputed": True,
            "reason": "avoid reading sealed rows beyond the physical stage window",
        },
    )


def weekly_cluster_signflip_two_sided(
    trades: pd.DataFrame, *, draws: int, seed: int
) -> dict[str, Any]:
    if trades.empty:
        return {
            "p_value_two_sided": 1.0,
            "cluster_count": 0,
            "method": "empty",
            "draws": 0,
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
    denominator = float(len(trades))
    if len(values) <= 20:
        total = 1 << len(values)
        exceed = 0
        bit_positions = np.arange(len(values), dtype=np.uint64)
        for start in range(0, total, 65_536):
            stop = min(start + 65_536, total)
            masks = np.arange(start, stop, dtype=np.uint64)[:, None]
            signs = np.where((masks >> bit_positions) & 1, 1.0, -1.0)
            null = np.abs((signs @ values) / denominator)
            exceed += int(np.count_nonzero(null >= observed - 1e-15))
        p_value = exceed / total
        method = "exact"
        used_draws = total
    else:
        generator = np.random.default_rng(seed)
        exceed = 0
        remaining = draws
        while remaining:
            batch = min(10_000, remaining)
            signs = generator.choice((-1.0, 1.0), size=(batch, len(values)))
            null = np.abs((signs @ values) / denominator)
            exceed += int(np.count_nonzero(null >= observed - 1e-15))
            remaining -= batch
        p_value = (1 + exceed) / (draws + 1)
        method = "deterministic_monte_carlo"
        used_draws = draws
    return {
        "p_value_two_sided": float(p_value),
        "cluster_count": int(len(values)),
        "method": method,
        "draws": int(used_draws),
        "seed": int(seed),
        "observed_mean_net_return": float(trades["net_return"].mean()),
        "weekly_net_return_sums": {
            str(label): float(value) for label, value in weekly.items()
        },
    }


def simulate_schedule(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    schedule: pd.DataFrame,
    *,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    cost_rate: float,
    cfg: EvaluationConfig,
) -> dict[str, Any]:
    metrics = strict_engine.simulate_schedule(
        market,
        funding,
        schedule,
        period_start=period_start,
        period_end=period_end,
        cost_rate=cost_rate,
        cfg=cfg,
    )
    metrics["weekly_cluster_signflip"] = weekly_cluster_signflip_two_sided(
        pd.DataFrame(metrics["trade_details"]),
        draws=cfg.cluster_draws,
        seed=cfg.cluster_seed,
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


def _headline(metrics: dict[str, Any]) -> dict[str, Any]:
    details = metrics["trade_details"]
    return {
        "absolute_return_pct": metrics["absolute_return_pct"],
        "cagr_pct": metrics["cagr_pct"],
        "strict_mdd_pct": metrics["strict_mdd_pct"],
        "cagr_to_strict_mdd": metrics["cagr_to_strict_mdd"],
        "trades": metrics["trades"],
        "long_trades": sum(int(row["side"]) == 1 for row in details),
        "short_trades": sum(int(row["side"]) == -1 for row in details),
        "mean_gross_underlying_bp": metrics["mean_gross_underlying_bp"],
        "weekly_cluster_signflip_p": metrics["weekly_cluster_signflip"][
            "p_value_two_sided"
        ],
        "weekly_clusters": metrics["weekly_cluster_signflip"]["cluster_count"],
    }


def _evaluate_clock(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    schedule: pd.DataFrame,
    *,
    window: TimeWindow,
    subperiods: dict[str, TimeWindow],
    stage: str,
    cfg: EvaluationConfig,
) -> dict[str, Any]:
    start, end = window
    base = simulate_schedule(
        market,
        funding,
        schedule,
        period_start=start,
        period_end=end,
        cost_rate=cfg.base_cost_notional_per_side,
        cfg=cfg,
    )
    stress = simulate_schedule(
        market,
        funding,
        schedule,
        period_start=start,
        period_end=end,
        cost_rate=cfg.stress_cost_notional_per_side,
        cfg=cfg,
    )
    split = _simulate_subperiods(
        market,
        funding,
        schedule,
        subperiods,
        cost_rate=cfg.base_cost_notional_per_side,
        cfg=cfg,
    )
    headline = _headline(base)
    if stage == "stage1_2020_2022":
        gates = {
            "absolute_return_positive": base["absolute_return_pct"] > 0.0,
            "cagr_to_strict_mdd_at_least": base["cagr_to_strict_mdd"] >= 3.0,
            "strict_mdd_pct_at_most": base["strict_mdd_pct"] <= 15.0,
            "weekly_cluster_signflip_p_at_most": base["weekly_cluster_signflip"][
                "p_value_two_sided"
            ]
            <= 0.10,
            "minimum_trades": base["trades"] >= 72,
            "minimum_long_trades": headline["long_trades"] >= 24,
            "minimum_short_trades": headline["short_trades"] >= 24,
            "mean_gross_underlying_bp_at_least": base["mean_gross_underlying_bp"]
            >= 40.0,
            "stress_cost_absolute_return_positive": stress["absolute_return_pct"] > 0.0,
            "each_calendar_year_absolute_return_positive": all(
                value["absolute_return_pct"] > 0.0 for value in split.values()
            ),
            "each_calendar_year_minimum_trades": all(
                value["trades"] >= 18 for value in split.values()
            ),
        }
    elif stage == "stage2_2023":
        gates = {
            "absolute_return_positive": base["absolute_return_pct"] > 0.0,
            "cagr_to_strict_mdd_at_least": base["cagr_to_strict_mdd"] >= 3.0,
            "strict_mdd_pct_at_most": base["strict_mdd_pct"] <= 15.0,
            "weekly_cluster_signflip_p_at_most": base["weekly_cluster_signflip"][
                "p_value_two_sided"
            ]
            <= 0.10,
            "minimum_trades": base["trades"] >= 20,
            "minimum_long_trades": headline["long_trades"] >= 8,
            "minimum_short_trades": headline["short_trades"] >= 8,
            "mean_gross_underlying_bp_at_least": base["mean_gross_underlying_bp"]
            >= 40.0,
            "each_half_absolute_return_nonnegative": all(
                value["absolute_return_pct"] >= 0.0 for value in split.values()
            ),
            "each_half_minimum_trades": all(
                value["trades"] >= 8 for value in split.values()
            ),
            "stress_cost_absolute_return_positive": stress["absolute_return_pct"] > 0.0,
        }
    else:
        raise ValueError(f"unknown H8DM-1 evaluation stage: {stage}")
    return {
        "metrics": base,
        "headline": headline,
        "stress_metrics": stress,
        "stress_headline": _headline(stress),
        "subperiod_metrics": split,
        "subperiod_headlines": {
            name: _headline(value) for name, value in split.items()
        },
        "gates": gates,
        "fully_qualified": all(gates.values()),
    }


def _build_stage_report(
    *,
    stage: str,
    window: TimeWindow,
    subperiods: dict[str, TimeWindow],
    diagnostics: dict[str, Any],
    market: pd.DataFrame,
    funding: pd.DataFrame,
    freeze: dict[str, Any],
) -> dict[str, Any]:
    registration, _ = _verify_static_inputs()
    cfg = _verify_evaluation_contract(registration)
    schedules = load_schedules()
    clocks = {
        name: _evaluate_clock(
            market,
            funding,
            schedule,
            window=window,
            subperiods=subperiods,
            stage=stage,
            cfg=cfg,
        )
        for name, schedule in schedules.items()
    }
    primary_ratio = clocks["primary"]["headline"]["cagr_to_strict_mdd"]
    comparisons = {
        name: {
            "primary_ratio": primary_ratio,
            "control_ratio": clocks[name]["headline"]["cagr_to_strict_mdd"],
            "margin": primary_ratio - clocks[name]["headline"]["cagr_to_strict_mdd"],
            "passed": primary_ratio - clocks[name]["headline"]["cagr_to_strict_mdd"]
            >= 0.25,
        }
        for name in MECHANISM_CONTROLS
    }
    gates = dict(clocks["primary"]["gates"])
    if stage == "stage1_2020_2022":
        gates["best_single_component_ratio_margin_at_least"] = all(
            comparisons[name]["passed"] for name in SINGLE_COMPONENT_CONTROLS
        )
    passed = all(gates.values())
    core = {
        "protocol_version": f"fed_h8_deposit_migration_{stage}_v1",
        "policy_id": "H8DM-1",
        "stage": stage,
        "evaluator_freeze_manifest_hash": freeze["manifest_hash"],
        "evaluator_source_sha256": freeze["evaluator_source_sha256"],
        "config": asdict(cfg),
        "execution_diagnostics": diagnostics,
        "base_cost_by_clock": {
            name: value["metrics"] for name, value in clocks.items()
        },
        "headline_by_clock": {
            name: value["headline"] for name, value in clocks.items()
        },
        "primary_stress_cost": clocks["primary"]["stress_metrics"],
        "primary_stress_headline": clocks["primary"]["stress_headline"],
        "primary_subperiods": clocks["primary"]["subperiod_metrics"],
        "primary_subperiod_headlines": clocks["primary"]["subperiod_headlines"],
        "control_full_battery": {
            name: value for name, value in clocks.items() if name != "primary"
        },
        "mechanism_ratio_comparison": comparisons,
        "gates": gates,
        "gate_passed": passed,
        "opened_windows": [stage],
        "sealed_windows": (
            ["stage2_2023", "2024_plus"]
            if stage == "stage1_2020_2022"
            else ["2024_plus"]
        ),
        "disposition": (
            "PASS_STAGE1_OPEN_2023_ONCE"
            if passed and stage == "stage1_2020_2022"
            else "PASS_STAGE2_OPEN_ORTHOGONALITY"
            if passed
            else "REJECT_KEEP_2023_SEALED"
            if stage == "stage1_2020_2022"
            else "REJECT_NO_REPAIR"
        ),
    }
    return _seal(core)


def evaluate_stage1() -> dict[str, Any]:
    freeze = verify_evaluator_freeze()
    market, funding, diagnostics = load_execution_window(STAGE1)
    report = _build_stage_report(
        stage="stage1_2020_2022",
        window=STAGE1,
        subperiods=STAGE1_SUBPERIODS,
        diagnostics=diagnostics,
        market=market,
        funding=funding,
        freeze=freeze,
    )
    prereg.write_once(STAGE1_OUTPUT, (json.dumps(report, indent=2) + "\n").encode())
    prereg.write_once(STAGE1_DOC, render_stage_doc(report).encode())
    return report


def _verified_passing_stage1(expected_freeze_hash: str) -> dict[str, Any]:
    if not STAGE1_OUTPUT.exists():
        raise ValueError("H8DM-1 Stage1 has not been run")
    stored = _load_json(STAGE1_OUTPUT)
    core = {key: value for key, value in stored.items() if key != "manifest_hash"}
    if stored.get("manifest_hash") != _canonical_hash(core):
        raise ValueError("H8DM-1 stored Stage1 manifest changed")
    if stored.get("evaluator_freeze_manifest_hash") != expected_freeze_hash:
        raise ValueError("H8DM-1 stored Stage1 freeze identity changed")
    if stored.get("evaluator_source_sha256") != _sha256(EVALUATOR_SOURCE):
        raise ValueError("H8DM-1 stored Stage1 evaluator changed")
    if stored.get("gate_passed") is not True:
        raise ValueError("H8DM-1 Stage1 failed; 2023 remains sealed")
    market, funding, diagnostics = load_execution_window(STAGE1)
    replayed = _build_stage_report(
        stage="stage1_2020_2022",
        window=STAGE1,
        subperiods=STAGE1_SUBPERIODS,
        diagnostics=diagnostics,
        market=market,
        funding=funding,
        freeze=verify_evaluator_freeze(),
    )
    if replayed != stored:
        raise ValueError("H8DM-1 Stage1 does not exactly replay")
    return stored


def _evaluate_pooled_primary(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    schedule: pd.DataFrame,
    *,
    diagnostics: dict[str, Any],
    cfg: EvaluationConfig,
) -> dict[str, Any]:
    start, end = POOLED
    base = simulate_schedule(
        market,
        funding,
        schedule,
        period_start=start,
        period_end=end,
        cost_rate=cfg.base_cost_notional_per_side,
        cfg=cfg,
    )
    stress = simulate_schedule(
        market,
        funding,
        schedule,
        period_start=start,
        period_end=end,
        cost_rate=cfg.stress_cost_notional_per_side,
        cfg=cfg,
    )
    headline = _headline(base)
    gates = {
        "pooled_stage1_stage2_ratio_at_least": (base["cagr_to_strict_mdd"] >= 3.0),
        "pooled_weekly_cluster_signflip_p_at_most": (
            base["weekly_cluster_signflip"]["p_value_two_sided"] <= 0.05
        ),
    }
    return {
        "physical_source_diagnostics": diagnostics,
        "metrics": base,
        "headline": headline,
        "stress_metrics": stress,
        "stress_headline": _headline(stress),
        "gates": gates,
        "fully_qualified": all(gates.values()),
    }


def _finalize_stage2_report(
    standalone: dict[str, Any],
    pooled: dict[str, Any],
    *,
    verified_stage1_manifest_hash: str,
) -> dict[str, Any]:
    core = {key: value for key, value in standalone.items() if key != "manifest_hash"}
    core["pooled_2020_2023"] = pooled
    core["gates"] = {**core["gates"], **pooled["gates"]}
    core["gate_passed"] = all(core["gates"].values())
    core["opened_windows"] = ["stage1_2020_2022", "stage2_2023"]
    core["sealed_windows"] = ["2024_plus"]
    core["disposition"] = (
        "PASS_STAGE2_OPEN_ORTHOGONALITY" if core["gate_passed"] else "REJECT_NO_REPAIR"
    )
    core["verified_stage1_manifest_hash"] = verified_stage1_manifest_hash
    return _seal(core)


def evaluate_stage2() -> dict[str, Any]:
    freeze = verify_evaluator_freeze()
    stage1 = _verified_passing_stage1(freeze["manifest_hash"])
    market, funding, diagnostics = load_execution_window(STAGE2)
    report = _build_stage_report(
        stage="stage2_2023",
        window=STAGE2,
        subperiods=STAGE2_SUBPERIODS,
        diagnostics=diagnostics,
        market=market,
        funding=funding,
        freeze=freeze,
    )
    pooled_market, pooled_funding, pooled_diagnostics = load_execution_window(POOLED)
    registration, _ = _verify_static_inputs()
    cfg = _verify_evaluation_contract(registration)
    pooled = _evaluate_pooled_primary(
        pooled_market,
        pooled_funding,
        load_schedules()["primary"],
        diagnostics=pooled_diagnostics,
        cfg=cfg,
    )
    report = _finalize_stage2_report(
        report,
        pooled,
        verified_stage1_manifest_hash=stage1["manifest_hash"],
    )
    prereg.write_once(STAGE2_OUTPUT, (json.dumps(report, indent=2) + "\n").encode())
    prereg.write_once(STAGE2_DOC, render_stage_doc(report).encode())
    return report


def render_stage_doc(report: dict[str, Any]) -> str:
    primary = report["headline_by_clock"]["primary"]
    stress = report["primary_stress_headline"]
    subperiod_rows = "\n".join(
        f"| {name} | {row['absolute_return_pct']:.4f}% | {row['cagr_pct']:.4f}% | "
        f"{row['strict_mdd_pct']:.4f}% | {row['cagr_to_strict_mdd']:.4f} | "
        f"{row['trades']} |"
        for name, row in report["primary_subperiod_headlines"].items()
    )
    gate_rows = "\n".join(
        f"| `{name}` | {'PASS' if value else 'FAIL'} |"
        for name, value in report["gates"].items()
    )
    pooled_section = ""
    if "pooled_2020_2023" in report:
        pooled = report["pooled_2020_2023"]["headline"]
        pooled_section = f"""

## Pooled replay (2020-2023)

This replay was run only after Stage1 passed an exact frozen-source replay and
the standalone 2023 window was opened. It introduces no 2024+ data.

| Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Long | Short | Mean gross bp | p(two-sided) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| {pooled["absolute_return_pct"]:.4f}% | {pooled["cagr_pct"]:.4f}% | {pooled["strict_mdd_pct"]:.4f}% | {pooled["cagr_to_strict_mdd"]:.4f} | {pooled["trades"]} | {pooled["long_trades"]} | {pooled["short_trades"]} | {pooled["mean_gross_underlying_bp"]:.4f} | {pooled["weekly_cluster_signflip_p"]:.4f} |
"""
    return f"""# H8DM-1 {report["stage"]} result — {prereg.AS_OF_DATE}

## Decision

**{report["disposition"]}**

| Cost | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Long | Short | Mean gross bp | p(two-sided) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 bp/notional/side | {primary["absolute_return_pct"]:.4f}% | {primary["cagr_pct"]:.4f}% | {primary["strict_mdd_pct"]:.4f}% | {primary["cagr_to_strict_mdd"]:.4f} | {primary["trades"]} | {primary["long_trades"]} | {primary["short_trades"]} | {primary["mean_gross_underlying_bp"]:.4f} | {primary["weekly_cluster_signflip_p"]:.4f} |
| 10 bp/notional/side | {stress["absolute_return_pct"]:.4f}% | {stress["cagr_pct"]:.4f}% | {stress["strict_mdd_pct"]:.4f}% | {stress["cagr_to_strict_mdd"]:.4f} | {stress["trades"]} | {stress["long_trades"]} | {stress["short_trades"]} | {stress["mean_gross_underlying_bp"]:.4f} | {stress["weekly_cluster_signflip_p"]:.4f} |

## Contained subperiods

| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
{subperiod_rows}
{pooled_section}

## Gates

| Gate | Result |
|---|:---:|
{gate_rows}

## Integrity

- Physically opened window: `{report["execution_diagnostics"]["physical_window"]}`
- Evaluator SHA-256: `{report["evaluator_source_sha256"]}`
- Result manifest: `{report["manifest_hash"]}`
- Full-clock CAGR includes idle cash; strict MDD includes intratrade adverse OHLC.
"""


def _print_summary(report: dict[str, Any]) -> None:
    print(
        json.dumps(
            {
                "stage": report["stage"],
                "passed": report["gate_passed"],
                "disposition": report["disposition"],
                "primary": report["headline_by_clock"]["primary"],
                "failed_gates": [
                    name for name, value in report["gates"].items() if not value
                ],
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--stage1", action="store_true")
    group.add_argument("--stage2", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        print(json.dumps(freeze_evaluator(), indent=2))
    elif args.stage1:
        _print_summary(evaluate_stage1())
    else:
        _print_summary(evaluate_stage2())


if __name__ == "__main__":
    main()
