"""Sequential strict evaluator for frozen CIHM-1.

Evaluator freeze opens no outcome. Stage 1 physically parses only 2021-2022.
Calendar 2023 is unreachable unless the unchanged Stage-1 artifact passes and
exactly replays under this evaluator identity.
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
from training import preregister_cboe_institutional_hedge_migration as prereg


SUPPORT_COMMIT = "3a4cfe5"
STATIC_INPUT_SHA256 = {
    "training/preregister_cboe_institutional_hedge_migration.py": "e25ee31535318a021e89086cc90f19736c701c3543c262a07d82ef0d4e9b314e",
    "training/cboe_institutional_hedge_migration_clock.py": "2911992773fd7a454d4fd7e988a3353ac176e65aedaf7208676451da206c6bf8",
    "docs/cboe-institutional-hedge-migration-preregistration-2026-07-18.md": "402cda95abc5bdcd58933dcdee8a646cdc968c98260fb0897276b9b5bfa5b555",
    "results/cboe_institutional_hedge_migration_preregistration_2026-07-18.json": "0709c7aff57dc1e1e7079979ec44ceb0e154c47898ea593f2bfe50d1ab4052d5",
    "training/build_cboe_institutional_hedge_migration_support.py": "55c1bb6c924c065e7e3495a57d76c373316b569c677c4f2ca538b18e1207f5d9",
    "docs/cboe-institutional-hedge-migration-support-2026-07-18.md": "34de48df18db176ccf1441e46d5f1197d53b5d19fe1621dffa9390572962cc07",
    "results/cboe_institutional_hedge_migration_support_2026-07-18.json": "df7ce11f92d50c52ffd61afd7cab6a2a3da5696e08ba45b886e3c37bec7f0a93",
    "results/cboe_institutional_hedge_migration_clocks_2026-07-18.csv.gz": "5e04cffacb1754c3111fcc32b09d72f06b546a4803b40c77d655a9787b015c0b",
    "training/evaluate_fiat_quote_participation_rotation.py": "e309f5217f033d57d2eadfec936843e736ce287f5c47f957c0ac6f0c71879c23",
}

PREREGISTRATION = Path(prereg.DEFAULT_OUTPUT)
SUPPORT_RESULT = Path("results/cboe_institutional_hedge_migration_support_2026-07-18.json")
CLOCKS = Path("results/cboe_institutional_hedge_migration_clocks_2026-07-18.csv.gz")
MARKET = Path(prereg.MARKET_PATH)
FUNDING = Path(prereg.FUNDING_PATH)
MARKET_MANIFEST = Path(prereg.MARKET_MANIFEST)
FUNDING_MANIFEST = Path(prereg.FUNDING_MANIFEST)
EVALUATOR_SOURCE = Path("training/evaluate_cboe_institutional_hedge_migration.py")
EVALUATOR_FREEZE = Path(
    "results/cboe_institutional_hedge_migration_evaluator_freeze_2026-07-18.json"
)
STAGE1_OUTPUT = Path(
    "results/cboe_institutional_hedge_migration_stage1_2021_2022_2026-07-18.json"
)
STAGE1_DOC = Path(
    "docs/cboe-institutional-hedge-migration-stage1-2021-2022-2026-07-18.md"
)
STAGE2_OUTPUT = Path(
    "results/cboe_institutional_hedge_migration_stage2_2023_2026-07-18.json"
)
STAGE2_DOC = Path(
    "docs/cboe-institutional-hedge-migration-stage2-2023-2026-07-18.md"
)

TimeWindow = tuple[pd.Timestamp, pd.Timestamp]
EvaluationConfig = strict_engine.EvaluationConfig
_parse_market_window = strict_engine._parse_market_window
_parse_funding_window = strict_engine._parse_funding_window


def _utc(value: str) -> pd.Timestamp:
    return cast(pd.Timestamp, pd.Timestamp(value, tz="UTC"))


def _timestamp(value: Any) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result is pd.NaT:
        raise ValueError("CIHM-1 timestamp is NaT")
    return cast(pd.Timestamp, result)


STAGE1: TimeWindow = (_utc("2021-01-01"), _utc("2023-01-01"))
STAGE1_SUBPERIODS: dict[str, TimeWindow] = {
    "2021": (_utc("2021-01-01"), _utc("2022-01-01")),
    "2022": (_utc("2022-01-01"), _utc("2023-01-01")),
}
STAGE2: TimeWindow = (_utc("2023-01-01"), _utc("2024-01-01"))
STAGE2_SUBPERIODS: dict[str, TimeWindow] = {
    "2023_h1": (_utc("2023-01-01"), _utc("2023-07-01")),
    "2023_h2": (_utc("2023-07-01"), _utc("2024-01-01")),
}
MECHANISM_CONTROLS = (
    "institutional_gap_only",
    "vix_call_pressure_only",
    "index_share_only",
    "level_composite",
)
FALSIFICATION_CONTROLS = (
    "direction_flip",
    "one_release_delay",
    "seven_release_placebo",
)
ALL_CLOCK_NAMES = (
    "primary",
    *MECHANISM_CONTROLS,
    *FALSIFICATION_CONTROLS,
)


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
        raise ValueError(f"CIHM-1 expected a JSON object: {path}")
    return payload


def _verify_evaluation_contract(
    registration: dict[str, Any], *, stage2: bool = False
) -> tuple[EvaluationConfig, dict[str, Any]]:
    cfg = EvaluationConfig()
    policy = registration["policy"]
    for key in (
        "leverage",
        "base_cost_notional_per_side",
        "stress_cost_notional_per_side",
    ):
        if asdict(cfg)[key] != policy[key]:
            raise ValueError(f"CIHM-1 strict engine differs on {key}")
    if cfg.cluster_draws != 20_000 or cfg.cluster_seed != 20_260_717:
        raise ValueError("CIHM-1 sign-flip configuration changed")
    key = "stage2_gates" if stage2 else "gates"
    gates = registration["selection_protocol"][key]
    expected = {
        "absolute_return_positive": True,
        "cagr_to_strict_mdd_min": 3.0,
        "strict_mdd_pct_max": 15.0,
        "weekly_cluster_signflip_p_max": 0.10,
        "minimum_trades": 60 if stage2 else 150,
        "minimum_short_trades": 60 if stage2 else 150,
        "mean_gross_underlying_bp_min": 35.0,
        "stress_cost_absolute_return_positive": True,
        "each_subperiod_absolute_return_positive": True,
        "each_subperiod_minimum_trades": 25 if stage2 else 70,
        "mechanism_margin_ratio_min": 0.25,
    }
    if gates != expected:
        raise ValueError(f"CIHM-1 {key} contract changed")
    return cfg, gates


def _verify_static_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    for path, expected in STATIC_INPUT_SHA256.items():
        if _sha256(path) != expected:
            raise ValueError(f"CIHM-1 frozen input changed: {path}")
    registration = _load_json(PREREGISTRATION)
    prereg.validate_manifest(registration, verify_sources=False)
    _verify_evaluation_contract(registration)
    _verify_evaluation_contract(registration, stage2=True)
    support = _load_json(SUPPORT_RESULT)
    support_core = {key: value for key, value in support.items() if key != "manifest_hash"}
    if support.get("manifest_hash") != _canonical_hash(support_core):
        raise ValueError("CIHM-1 support manifest changed")
    if support.get("policy_id") != "CIHM-1":
        raise ValueError("CIHM-1 support identity changed")
    if support.get("outcomes_opened") is not False or support.get("outcome_sources_opened") != []:
        raise ValueError("CIHM-1 support opened outcomes")
    if support.get("support_passed") is not True or support.get("advance_to_stage1_outcomes") is not True:
        raise ValueError("CIHM-1 source support failed")
    if support.get("preregistration_manifest_hash") != registration["manifest_hash"]:
        raise ValueError("CIHM-1 support is not bound to preregistration")
    if support.get("clocks", {}).get("sha256") != STATIC_INPUT_SHA256[str(CLOCKS)]:
        raise ValueError("CIHM-1 support is not bound to its clock ledger")
    return registration, support


def _schedule_frame(name: str, rows: pd.DataFrame) -> pd.DataFrame:
    frame = rows.copy()
    frame["clock_name"] = name
    frame["signal_day"] = pd.to_datetime(frame["signal_time"], utc=True, errors="raise")
    frame["entry_time"] = pd.to_datetime(frame["entry_time"], utc=True, errors="raise")
    frame["exit_time"] = pd.to_datetime(frame["exit_time"], utc=True, errors="raise")
    def side_value(value: Any) -> int:
        if value == "LONG":
            return 1
        if value == "SHORT":
            return -1
        raise ValueError(f"CIHM-1 {name} ledger side changed")

    frame["side"] = frame["side"].apply(side_value).astype("int8")
    schedule = frame[
        [
            "clock_name",
            "signal_day",
            "entry_time",
            "exit_time",
            "side",
            "observation_date",
        ]
    ]
    order = np.argsort(np.asarray(schedule["entry_time"]))
    return schedule.iloc[order].reset_index(drop=True)


def load_schedules() -> dict[str, pd.DataFrame]:
    _, support = _verify_static_inputs()
    ledger = pd.read_csv(CLOCKS)
    base_names = {
        "primary",
        "institutional_gap_only",
        "vix_call_pressure_only",
        "index_share_only",
        "level_composite",
        "one_release_delay",
        "seven_release_placebo",
    }
    if set(ledger["control"]) != base_names:
        raise ValueError("CIHM-1 support clock family changed")
    schedules = {
        name: _schedule_frame(name, ledger.loc[ledger["control"].eq(name)])
        for name in base_names
    }
    expected_counts = support["clocks"]["counts"]
    for name in base_names:
        if len(schedules[name]) != int(expected_counts[name]):
            raise ValueError(f"CIHM-1 {name} source count changed")
    primary = schedules["primary"]
    direction = primary.copy()
    direction["clock_name"] = "direction_flip"
    direction["side"] = -direction["side"]
    schedules["direction_flip"] = direction
    schedules = {name: schedules[name] for name in ALL_CLOCK_NAMES}
    for name, schedule in schedules.items():
        if not bool(schedule["side"].isin((-1, 1)).all()):
            raise ValueError(f"CIHM-1 {name} side changed")
        if not bool(schedule["entry_time"].eq(schedule["signal_day"]).all()):
            raise ValueError(f"CIHM-1 {name} decision/entry clock changed")
        if not bool(schedule["exit_time"].gt(schedule["entry_time"]).all()):
            raise ValueError(f"CIHM-1 {name} exit no longer follows entry")
        if len(schedule) > 1:
            entries = schedule["entry_time"].iloc[1:].reset_index(drop=True)
            exits = schedule["exit_time"].iloc[:-1].reset_index(drop=True)
            if not bool(entries.ge(exits).all()):
                raise ValueError(f"CIHM-1 {name} schedule overlaps")
        if not bool(schedule["exit_time"].lt(STAGE2[1]).all()):
            raise ValueError(f"CIHM-1 {name} clock crosses source horizon")
    return schedules


def _schedule_hash(frame: pd.DataFrame) -> str:
    rows = [
        {
            "clock_name": str(row["clock_name"]),
            "signal_day": _timestamp(row["signal_day"]).isoformat(),
            "entry_time": _timestamp(row["entry_time"]).isoformat(),
            "exit_time": _timestamp(row["exit_time"]).isoformat(),
            "side": int(row["side"]),
            "observation_date": str(row["observation_date"]),
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


def _schedule_record(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "events": int(len(frame)),
        "schedule_hash": _schedule_hash(frame),
        "first_entry": _timestamp(frame["entry_time"].min()).isoformat(),
        "last_exit": _timestamp(frame["exit_time"].max()).isoformat(),
        "decision_equals_entry": bool(frame["entry_time"].eq(frame["signal_day"]).all()),
        "positive_holding_intervals": bool(frame["exit_time"].gt(frame["entry_time"]).all()),
        "globally_nonoverlapping": bool(
            frame["entry_time"].iloc[1:].reset_index(drop=True).ge(
                frame["exit_time"].iloc[:-1].reset_index(drop=True)
            ).all()
        ),
        "stage1_distribution": _entry_distribution(frame, STAGE1),
        "sealed_2023_distribution": _entry_distribution(frame, STAGE2),
    }


def freeze_evaluator(output_path: str | Path = EVALUATOR_FREEZE) -> dict[str, Any]:
    registration, support = _verify_static_inputs()
    schedules = load_schedules()
    records = {name: _schedule_record(schedule) for name, schedule in schedules.items()}
    if not all(
        record[check]
        for record in records.values()
        for check in ("decision_equals_entry", "positive_holding_intervals", "globally_nonoverlapping")
    ):
        raise ValueError("CIHM-1 evaluator schedule invariant failed")
    core = {
        "protocol_version": "cboe_institutional_hedge_migration_evaluator_v1",
        "policy_id": "CIHM-1",
        "as_of_date": "2026-07-18",
        "support_commit": SUPPORT_COMMIT,
        "preregistration_manifest_hash": registration["manifest_hash"],
        "support_manifest_hash": support["manifest_hash"],
        "evaluator_source": str(EVALUATOR_SOURCE),
        "evaluator_source_sha256": _sha256(EVALUATOR_SOURCE),
        "strict_engine_sha256": STATIC_INPUT_SHA256[
            "training/evaluate_fiat_quote_participation_rotation.py"
        ],
        "evaluation_config": asdict(_verify_evaluation_contract(registration)[0]),
        "static_inputs": STATIC_INPUT_SHA256,
        "control_schedules": records,
        "opened_windows": [],
        "sealed_windows": ["stage1_2021_2022", "stage2_2023", "2024_plus"],
        "execution_ohlc_rows_parsed_during_freeze": 0,
        "funding_rows_parsed_during_freeze": 0,
        "simulation_run_during_freeze": False,
        "mutable_parameters": [],
    }
    report = _seal(core)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def verify_evaluator_freeze(path: str | Path = EVALUATOR_FREEZE) -> dict[str, Any]:
    report = _load_json(path)
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    if report.get("manifest_hash") != _canonical_hash(core):
        raise ValueError("CIHM-1 evaluator freeze manifest changed")
    if report.get("evaluator_source_sha256") != _sha256(EVALUATOR_SOURCE):
        raise ValueError("CIHM-1 evaluator source changed after freeze")
    if report.get("support_commit") != SUPPORT_COMMIT:
        raise ValueError("CIHM-1 evaluator froze another support commit")
    if report.get("evaluation_config") != asdict(EvaluationConfig()):
        raise ValueError("CIHM-1 evaluation configuration changed")
    if report.get("opened_windows") != [] or report.get("mutable_parameters") != []:
        raise ValueError("CIHM-1 evaluator freeze is not sealed")
    if report.get("execution_ohlc_rows_parsed_during_freeze") != 0:
        raise ValueError("CIHM-1 evaluator freeze parsed OHLC")
    if report.get("funding_rows_parsed_during_freeze") != 0:
        raise ValueError("CIHM-1 evaluator freeze parsed funding")
    if report.get("simulation_run_during_freeze") is not False:
        raise ValueError("CIHM-1 evaluator freeze simulated an outcome")
    schedules = load_schedules()
    for name, schedule in schedules.items():
        if report["control_schedules"][name]["schedule_hash"] != _schedule_hash(schedule):
            raise ValueError(f"CIHM-1 {name} schedule changed after freeze")
    return report


def _verify_execution_manifests() -> None:
    registration = _load_json(PREREGISTRATION)
    source = registration["source_contract"]
    market_manifest = _load_json(MARKET_MANIFEST)
    if market_manifest.get("combined_sha256") != source["market_sha256"]:
        raise ValueError("CIHM-1 market manifest no longer binds the frozen source")
    funding_manifest = _load_json(FUNDING_MANIFEST)
    if funding_manifest.get("data", {}).get("sha256") != source["funding_sha256"]:
        raise ValueError("CIHM-1 funding manifest no longer binds the frozen source")
    if funding_manifest.get("outcomes_opened") is not False:
        raise ValueError("CIHM-1 funding source manifest opened an outcome")


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
        for mask in range(total):
            signed = sum(
                value if mask & (1 << index) else -value
                for index, value in enumerate(values)
            )
            exceed += abs(signed / denominator) >= observed - 1e-15
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


def _headline(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "absolute_return_pct": metrics["absolute_return_pct"],
        "cagr_pct": metrics["cagr_pct"],
        "strict_mdd_pct": metrics["strict_mdd_pct"],
        "cagr_to_strict_mdd": metrics["cagr_to_strict_mdd"],
        "trades": metrics["trades"],
        "mean_gross_underlying_bp": metrics["mean_gross_underlying_bp"],
        "weekly_cluster_signflip_p": metrics["weekly_cluster_signflip"]["p_value_two_sided"],
        "weekly_clusters": metrics["weekly_cluster_signflip"]["cluster_count"],
    }


def _evaluate_clock(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    schedule: pd.DataFrame,
    *,
    window: TimeWindow,
    subperiods: dict[str, TimeWindow],
    cfg: EvaluationConfig,
    gate_contract: dict[str, Any],
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
    split = {
        name: simulate_schedule(
            market,
            funding,
            schedule,
            period_start=period_start,
            period_end=period_end,
            cost_rate=cfg.base_cost_notional_per_side,
            cfg=cfg,
        )
        for name, (period_start, period_end) in subperiods.items()
    }
    selected = _window_schedule(schedule, window)
    short_trades = int(selected["side"].eq(-1).sum())
    gates = {
        "absolute_return_positive": base["absolute_return_pct"] > 0.0,
        "cagr_to_strict_mdd_at_least_3": base["cagr_to_strict_mdd"] >= gate_contract["cagr_to_strict_mdd_min"],
        "strict_mdd_at_most_15pct": base["strict_mdd_pct"] <= gate_contract["strict_mdd_pct_max"],
        "weekly_cluster_signflip_p_at_most_10pct": base["weekly_cluster_signflip"]["p_value_two_sided"] <= gate_contract["weekly_cluster_signflip_p_max"],
        "minimum_trades": base["trades"] >= gate_contract["minimum_trades"],
        "mean_gross_underlying_at_least_35bp": base["mean_gross_underlying_bp"] >= gate_contract["mean_gross_underlying_bp_min"],
        "stress_cost_absolute_return_positive": stress["absolute_return_pct"] > 0.0,
        "each_subperiod_absolute_return_positive": all(value["absolute_return_pct"] > 0.0 for value in split.values()),
        "each_subperiod_minimum_trades": all(value["trades"] >= gate_contract["each_subperiod_minimum_trades"] for value in split.values()),
        "minimum_short_trades": short_trades >= gate_contract["minimum_short_trades"],
    }
    performance_gates = {
        name: value for name, value in gates.items() if name != "minimum_short_trades"
    }
    return {
        "metrics": base,
        "headline": _headline(base),
        "stress_metrics": stress,
        "stress_headline": _headline(stress),
        "subperiod_metrics": split,
        "subperiod_headlines": {name: _headline(value) for name, value in split.items()},
        "side_counts": {
            "long": int(selected["side"].eq(1).sum()),
            "short": int(selected["side"].eq(-1).sum()),
        },
        "gates": gates,
        "fully_qualified": all(gates.values()),
        "performance_qualified_ignoring_frozen_direction_support": all(
            performance_gates.values()
        ),
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
    is_stage2 = stage == "stage2_2023"
    cfg, gate_contract = _verify_evaluation_contract(registration, stage2=is_stage2)
    schedules = load_schedules()
    clocks = {
        name: _evaluate_clock(
            market,
            funding,
            schedule,
            window=window,
            subperiods=subperiods,
            cfg=cfg,
            gate_contract=gate_contract,
        )
        for name, schedule in schedules.items()
    }
    primary_ratio = clocks["primary"]["headline"]["cagr_to_strict_mdd"]
    comparisons = {
        name: {
            "primary_ratio": primary_ratio,
            "control_ratio": clocks[name]["headline"]["cagr_to_strict_mdd"],
            "margin": primary_ratio - clocks[name]["headline"]["cagr_to_strict_mdd"],
            "passed": primary_ratio - clocks[name]["headline"]["cagr_to_strict_mdd"] >= gate_contract["mechanism_margin_ratio_min"],
        }
        for name in MECHANISM_CONTROLS
    }
    falsification = {
        name: clocks[name][
            "performance_qualified_ignoring_frozen_direction_support"
        ]
        for name in FALSIFICATION_CONTROLS
    }
    gates = {
        **clocks["primary"]["gates"],
        "mechanism_control_margin_at_least_0_25": all(value["passed"] for value in comparisons.values()),
        "falsification_controls_do_not_performance_qualify": not any(
            falsification.values()
        ),
    }
    passed = all(gates.values())
    core = {
        "protocol_version": f"cboe_institutional_hedge_migration_{stage}_v1",
        "policy_id": "CIHM-1",
        "stage": stage,
        "evaluator_freeze_manifest_hash": freeze["manifest_hash"],
        "evaluator_source_sha256": freeze["evaluator_source_sha256"],
        "config": asdict(cfg),
        "gate_contract": gate_contract,
        "execution_diagnostics": diagnostics,
        "base_cost_by_clock": {name: value["metrics"] for name, value in clocks.items()},
        "headline_by_clock": {name: value["headline"] for name, value in clocks.items()},
        "primary_stress_cost": clocks["primary"]["stress_metrics"],
        "primary_stress_headline": clocks["primary"]["stress_headline"],
        "primary_subperiods": clocks["primary"]["subperiod_metrics"],
        "primary_subperiod_headlines": clocks["primary"]["subperiod_headlines"],
        "control_full_battery": {name: value for name, value in clocks.items() if name != "primary"},
        "mechanism_ratio_comparison": comparisons,
        "falsification_fully_qualified": falsification,
        "gates": gates,
        "gate_passed": passed,
        "opened_windows": [stage],
        "sealed_windows": ["stage2_2023", "2024_plus"] if not is_stage2 else ["2024_plus"],
        "disposition": (
            "PASS_STAGE1_OPEN_2023_ONCE"
            if passed and not is_stage2
            else "PASS_STAGE2_OPEN_ORTHOGONALITY"
            if passed
            else "REJECT_KEEP_2023_SEALED"
            if not is_stage2
            else "REJECT_NO_REPAIR"
        ),
    }
    return _seal(core)


def evaluate_stage1() -> dict[str, Any]:
    freeze = verify_evaluator_freeze()
    market, funding, diagnostics = load_execution_window(STAGE1)
    report = _build_stage_report(
        stage="stage1_2021_2022",
        window=STAGE1,
        subperiods=STAGE1_SUBPERIODS,
        diagnostics=diagnostics,
        market=market,
        funding=funding,
        freeze=freeze,
    )
    STAGE1_OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    STAGE1_DOC.write_text(render_stage_doc(report))
    return report


def _verified_passing_stage1(expected_freeze_hash: str) -> dict[str, Any]:
    if not STAGE1_OUTPUT.exists():
        raise ValueError("CIHM-1 Stage1 has not been run")
    stored = _load_json(STAGE1_OUTPUT)
    core = {key: value for key, value in stored.items() if key != "manifest_hash"}
    if stored.get("manifest_hash") != _canonical_hash(core):
        raise ValueError("CIHM-1 stored Stage1 manifest changed")
    if stored.get("evaluator_freeze_manifest_hash") != expected_freeze_hash:
        raise ValueError("CIHM-1 stored Stage1 freeze identity changed")
    if stored.get("evaluator_source_sha256") != _sha256(EVALUATOR_SOURCE):
        raise ValueError("CIHM-1 stored Stage1 evaluator changed")
    if stored.get("gate_passed") is not True:
        raise ValueError("CIHM-1 Stage1 failed; 2023 remains sealed")
    market, funding, diagnostics = load_execution_window(STAGE1)
    replayed = _build_stage_report(
        stage="stage1_2021_2022",
        window=STAGE1,
        subperiods=STAGE1_SUBPERIODS,
        diagnostics=diagnostics,
        market=market,
        funding=funding,
        freeze=verify_evaluator_freeze(),
    )
    if replayed != stored:
        raise ValueError("CIHM-1 Stage1 does not exactly replay")
    return stored


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
    report["verified_stage1_manifest_hash"] = stage1["manifest_hash"]
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    report["manifest_hash"] = _canonical_hash(core)
    STAGE2_OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    STAGE2_DOC.write_text(render_stage_doc(report))
    return report


def render_stage_doc(report: dict[str, Any]) -> str:
    primary = report["headline_by_clock"]["primary"]
    stress = report["primary_stress_headline"]
    subperiod_rows = "\n".join(
        f"| {name} | {row['absolute_return_pct']:.4f}% | {row['cagr_pct']:.4f}% | "
        f"{row['strict_mdd_pct']:.4f}% | {row['cagr_to_strict_mdd']:.4f} | {row['trades']} |"
        for name, row in report["primary_subperiod_headlines"].items()
    )
    gate_rows = "\n".join(
        f"| `{name}` | {'PASS' if value else 'FAIL'} |"
        for name, value in report["gates"].items()
    )
    return f"""# CIHM-1 {report['stage']} result — 2026-07-18

## Decision

**{report['disposition']}**

| Cost | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Mean gross bp | p(two-sided) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6 bp/notional/side | {primary['absolute_return_pct']:.4f}% | {primary['cagr_pct']:.4f}% | {primary['strict_mdd_pct']:.4f}% | {primary['cagr_to_strict_mdd']:.4f} | {primary['trades']} | {primary['mean_gross_underlying_bp']:.4f} | {primary['weekly_cluster_signflip_p']:.4f} |
| 10 bp/notional/side | {stress['absolute_return_pct']:.4f}% | {stress['cagr_pct']:.4f}% | {stress['strict_mdd_pct']:.4f}% | {stress['cagr_to_strict_mdd']:.4f} | {stress['trades']} | {stress['mean_gross_underlying_bp']:.4f} | {stress['weekly_cluster_signflip_p']:.4f} |

## Contained subperiods

| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
{subperiod_rows}

## Gates

| Gate | Result |
|---|:---:|
{gate_rows}

## Integrity

- Physically opened window: `{report['execution_diagnostics']['physical_window']}`
- Evaluator SHA-256: `{report['evaluator_source_sha256']}`
- Result manifest: `{report['manifest_hash']}`
- Full-calendar CAGR includes idle cash; strict MDD includes intratrade adverse OHLC.
"""


def _print_summary(report: dict[str, Any]) -> None:
    print(
        json.dumps(
            {
                "stage": report["stage"],
                "passed": report["gate_passed"],
                "disposition": report["disposition"],
                "primary": report["headline_by_clock"]["primary"],
                "failed_gates": [name for name, value in report["gates"].items() if not value],
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
