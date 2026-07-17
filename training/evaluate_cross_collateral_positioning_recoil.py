"""Sequential strict evaluator for frozen CCPR-1.

Evaluator freeze opens no execution outcome. Stage 1 physically parses only
2021-07-08 through 2022-12-31. The 2023 window can be opened only after an exact
replay of a hash-bound passing Stage-1 report. No threshold, direction, hold,
or signal gate may be repaired after outcomes are opened.
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
from training import preregister_cross_collateral_positioning_recoil as prereg


SUPPORT_COMMIT = "fcaa5fc328677303e86a02d1b5cc6808f41dba3b"
STATIC_INPUT_SHA256 = {
    "training/preregister_cross_collateral_positioning_recoil.py": (
        "6d1224c3a0d24686bf3b997f424b10f65d004c211269916f6007de8b8464a0a5"
    ),
    "docs/cross-collateral-positioning-recoil-preregistration-2026-07-17.md": (
        "722ec0fb44e17545a6bce3d572ce08cdd4422eaee18ce287a2557258bc735e43"
    ),
    "results/cross_collateral_positioning_recoil_preregistration_2026-07-17.json": (
        "a2ae7b4ee86cd98c409966d4c35227c62989046758d15620eec2ad8cb05dc7fd"
    ),
    "training/build_cross_collateral_positioning_recoil_support.py": (
        "19f22a19802d15600adc542bcd31dabc8d3d44e4625a47df1a72e418394f489e"
    ),
    "docs/cross-collateral-positioning-recoil-support-2026-07-17.md": (
        "894649df40196d3c0e373c75f2602d3b22472ebfaab9c2f2f548ab3ec70899c6"
    ),
    "results/cross_collateral_positioning_recoil_support_2026-07-17.json": (
        "0d6c12fe7d0fdead0749a42dca4cfefc40a3c49fba597458577a1102b6232c7d"
    ),
    "results/cross_collateral_positioning_recoil_clocks_2026-07-17.csv": (
        "2a864ec2b616a3118bf9ffa44f99f96fbe19e79d82870f21a0d7d9010d5c993a"
    ),
    "training/evaluate_fiat_quote_participation_rotation.py": (
        "e309f5217f033d57d2eadfec936843e736ce287f5c47f957c0ac6f0c71879c23"
    ),
}

PREREGISTRATION = Path(prereg.DEFAULT_OUTPUT)
SUPPORT_RESULT = Path(
    "results/cross_collateral_positioning_recoil_support_2026-07-17.json"
)
CLOCKS = Path("results/cross_collateral_positioning_recoil_clocks_2026-07-17.csv")
MARKET = Path(prereg.MARKET_PATH)
FUNDING = Path(prereg.FUNDING_PATH)
MARKET_MANIFEST = Path(prereg.MARKET_MANIFEST_PATH)
FUNDING_MANIFEST = Path(prereg.FUNDING_MANIFEST_PATH)
EVALUATOR_SOURCE = Path("training/evaluate_cross_collateral_positioning_recoil.py")
EVALUATOR_FREEZE = Path(
    "results/cross_collateral_positioning_recoil_evaluator_freeze_2026-07-17.json"
)
STAGE1_OUTPUT = Path(
    "results/cross_collateral_positioning_recoil_stage1_2021_2022_2026-07-17.json"
)
STAGE1_DOC = Path(
    "docs/cross-collateral-positioning-recoil-stage1-2021-2022-2026-07-17.md"
)
STAGE2_OUTPUT = Path(
    "results/cross_collateral_positioning_recoil_stage2_2023_2026-07-17.json"
)
STAGE2_DOC = Path("docs/cross-collateral-positioning-recoil-stage2-2023-2026-07-17.md")

TimeWindow = tuple[pd.Timestamp, pd.Timestamp]


def _utc(value: str) -> pd.Timestamp:
    return cast(pd.Timestamp, pd.Timestamp(value, tz="UTC"))


def _timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp is pd.NaT:
        raise ValueError("CCPR-1 timestamp is NaT")
    return cast(pd.Timestamp, timestamp)


STAGE1: TimeWindow = (_utc("2021-07-08"), _utc("2023-01-01"))
STAGE1_SUBPERIODS: dict[str, TimeWindow] = {
    "2021_partial": (_utc("2021-07-08"), _utc("2022-01-01")),
    "2022_h1": (_utc("2022-01-01"), _utc("2022-07-01")),
    "2022_h2": (_utc("2022-07-01"), _utc("2023-01-01")),
}
STAGE2: TimeWindow = (_utc("2023-01-01"), _utc("2024-01-01"))
STAGE2_SUBPERIODS: dict[str, TimeWindow] = {
    "2023_h1": (_utc("2023-01-01"), _utc("2023-07-01")),
    "2023_h2": (_utc("2023-07-01"), _utc("2024-01-01")),
}

MECHANISM_CONTROLS = ("oi_only", "taker_only", "um_only", "cm_only")
FALSIFICATION_CONTROLS = (
    "direction_flip",
    "entry_shift_plus_1h",
    "deterministic_random_side",
)
ALL_CLOCK_NAMES = ("primary", *MECHANISM_CONTROLS, *FALSIFICATION_CONTROLS)
CANDIDATE_HOLDS = {"CCPR-H4": 48, "CCPR-H8": 96}
CANDIDATE_IDS = tuple(CANDIDATE_HOLDS)
STAGE1_TOP_LEVEL_KEYS = frozenset(
    {
        "protocol_version",
        "policy_id",
        "stage",
        "evaluator_freeze_manifest_hash",
        "evaluator_source_sha256",
        "config",
        "execution_diagnostics",
        "candidate_results",
        "selected_candidate_id",
        "stage1_passed",
        "opened_windows",
        "sealed_windows",
        "disposition",
        "manifest_hash",
    }
)

EvaluationConfig = strict_engine.EvaluationConfig
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
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"CCPR-1 expected JSON object: {path}")
    return payload


def _verify_evaluation_contract(registration: dict[str, Any]) -> EvaluationConfig:
    policy = registration["policy"]
    statistical = registration["selection_protocol"]["statistical_test_contract"]
    cfg = EvaluationConfig()
    expected = {
        "leverage": policy["leverage"],
        "base_cost_notional_per_side": policy["base_cost_notional_per_side"],
        "stress_cost_notional_per_side": policy["stress_cost_notional_per_side"],
        "cluster_draws": statistical["draws"],
        "cluster_seed": statistical["seed"],
        "mdd_denominator_floor": 1e-9,
    }
    if asdict(cfg) != expected:
        raise ValueError(f"CCPR-1 strict-engine contract changed: {asdict(cfg)}")
    if statistical["test"] != "two-sided weekly-cluster sign-flip on net trade PnL":
        raise ValueError("CCPR-1 statistical-test contract changed")
    return cfg


def _verify_static_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    for path, expected in STATIC_INPUT_SHA256.items():
        if _sha256(path) != expected:
            raise ValueError(f"CCPR-1 frozen input changed: {path}")
    registration = _load_json(PREREGISTRATION)
    prereg.validate_manifest(registration, verify_sources=False)
    support = _load_json(SUPPORT_RESULT)
    if support.get("policy_id") != "CCPR-1":
        raise ValueError("CCPR-1 support identity changed")
    if support.get("outcomes_opened") is not False:
        raise ValueError("CCPR-1 support opened outcomes")
    if support.get("outcome_sources_opened") != []:
        raise ValueError("CCPR-1 support opened an outcome source")
    if support.get("support_passed") is not True:
        raise ValueError("CCPR-1 support did not pass")
    if support.get("advance_to_stage1_outcomes") is not True:
        raise ValueError("CCPR-1 support did not authorize Stage1")
    if support.get("selected_q") != 0.85:
        raise ValueError("CCPR-1 selected Q changed")
    if support.get("clocks", {}).get("sha256") != STATIC_INPUT_SHA256[str(CLOCKS)]:
        raise ValueError("CCPR-1 support no longer binds the clock ledger")
    _verify_evaluation_contract(registration)
    return registration, support


def _random_side(signal_time: pd.Timestamp) -> int:
    digest = hashlib.sha256(
        f"CCPR-1|{signal_time.isoformat()}".encode("utf-8")
    ).digest()
    return 1 if digest[0] & 1 else -1


def _reserve_schedule(
    frame: pd.DataFrame,
    *,
    candidate_id: str,
    clock_name: str,
    hold_bars: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    previous_exit: pd.Timestamp | None = None
    for _, row in frame.sort_values("entry_time").iterrows():
        signal_time = _timestamp(row["signal_time"])
        entry_time = _timestamp(row["entry_time"])
        side = int(row["side"])
        if clock_name == "entry_shift_plus_1h":
            entry_time = _timestamp(entry_time + pd.Timedelta(hours=1))
        exit_time = _timestamp(entry_time + pd.Timedelta(minutes=5 * hold_bars))
        if exit_time >= STAGE2[1]:
            continue
        if previous_exit is not None and entry_time < previous_exit:
            continue
        previous_exit = exit_time
        if clock_name == "direction_flip":
            side *= -1
        elif clock_name == "deterministic_random_side":
            side = _random_side(signal_time)
        rows.append(
            {
                "candidate_id": candidate_id,
                "clock_name": clock_name,
                "signal_time": signal_time,
                "signal_day": signal_time,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "side": side,
            }
        )
    schedule = pd.DataFrame(rows)
    if schedule.empty:
        raise ValueError(f"CCPR-1 empty schedule: {candidate_id}/{clock_name}")
    for column in ("signal_time", "signal_day", "entry_time", "exit_time"):
        schedule[column] = pd.to_datetime(schedule[column], utc=True, errors="raise")
    schedule["side"] = schedule["side"].astype("int8")
    return schedule.sort_values("entry_time").reset_index(drop=True)


def load_schedules() -> dict[str, dict[str, pd.DataFrame]]:
    _, support = _verify_static_inputs()
    ledger = pd.read_csv(CLOCKS)
    ledger["signal_time"] = pd.to_datetime(
        ledger["signal_time"], utc=True, errors="raise"
    )
    ledger["entry_time"] = pd.to_datetime(
        ledger["entry_time"], utc=True, errors="raise"
    )
    selected = ledger.loc[np.isclose(ledger["q"], support["selected_q"])].copy()
    if set(selected["control"]) != {"primary", *MECHANISM_CONTROLS}:
        raise ValueError("CCPR-1 source clock family changed")
    if (
        not selected["entry_time"]
        .eq(selected["signal_time"] + pd.Timedelta(minutes=10))
        .all()
    ):
        raise ValueError("CCPR-1 source clock entry delay changed")
    if not selected["side"].isin((-1, 1)).all():
        raise ValueError("CCPR-1 source clock side changed")

    schedules: dict[str, dict[str, pd.DataFrame]] = {}
    for candidate_id, hold_bars in CANDIDATE_HOLDS.items():
        schedules[candidate_id] = {}
        primary_source = selected.loc[selected["control"].eq("primary")]
        for clock_name in ("primary", *MECHANISM_CONTROLS):
            source = selected.loc[selected["control"].eq(clock_name)]
            schedules[candidate_id][clock_name] = _reserve_schedule(
                source,
                candidate_id=candidate_id,
                clock_name=clock_name,
                hold_bars=hold_bars,
            )
        for clock_name in FALSIFICATION_CONTROLS:
            schedules[candidate_id][clock_name] = _reserve_schedule(
                primary_source,
                candidate_id=candidate_id,
                clock_name=clock_name,
                hold_bars=hold_bars,
            )
    return schedules


def _schedule_hash(frame: pd.DataFrame) -> str:
    rows = [
        {
            "candidate_id": str(row["candidate_id"]),
            "clock_name": str(row["clock_name"]),
            "signal_time": _timestamp(row["signal_time"]).isoformat(),
            "signal_day": _timestamp(row["signal_day"]).isoformat(),
            "entry_time": _timestamp(row["entry_time"]).isoformat(),
            "exit_time": _timestamp(row["exit_time"]).isoformat(),
            "side": int(row["side"]),
        }
        for _, row in frame.iterrows()
    ]
    return _canonical_hash(rows)


def _window_schedule(frame: pd.DataFrame, window: TimeWindow) -> pd.DataFrame:
    start, end = window
    return frame.loc[
        frame["signal_day"].ge(start)
        & frame["entry_time"].ge(start)
        & frame["exit_time"].lt(end)
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


def _schedule_freeze_record(
    frame: pd.DataFrame, *, hold_bars: int, shifted: bool
) -> dict[str, Any]:
    entries = [_timestamp(value) for value in frame["entry_time"]]
    exits = [_timestamp(value) for value in frame["exit_time"]]
    signals = [_timestamp(value) for value in frame["signal_time"]]
    expected_delay = pd.Timedelta(minutes=70 if shifted else 10)
    expected_hold = pd.Timedelta(minutes=5 * hold_bars)
    return {
        "events": int(len(frame)),
        "schedule_hash": _schedule_hash(frame),
        "first_entry": min(entries).isoformat(),
        "last_exit": max(exits).isoformat(),
        "entry_delay_exact": all(
            entry - signal == expected_delay
            for entry, signal in zip(entries, signals, strict=True)
        ),
        "hold_exact": all(
            exit_time - entry == expected_hold
            for entry, exit_time in zip(entries, exits, strict=True)
        ),
        "globally_nonoverlapping": all(
            entry >= prior_exit
            for entry, prior_exit in zip(entries[1:], exits[:-1], strict=True)
        ),
        "pre_2024_only": all(exit_time < STAGE2[1] for exit_time in exits),
        "valid_sides": all(side in {-1, 1} for side in frame["side"]),
    }


def freeze_evaluator(output_path: str | Path = EVALUATOR_FREEZE) -> dict[str, Any]:
    registration, support = _verify_static_inputs()
    schedules = load_schedules()
    records = {
        candidate_id: {
            clock_name: _schedule_freeze_record(
                schedule,
                hold_bars=CANDIDATE_HOLDS[candidate_id],
                shifted=clock_name == "entry_shift_plus_1h",
            )
            for clock_name, schedule in candidate_schedules.items()
        }
        for candidate_id, candidate_schedules in schedules.items()
    }
    checks = {
        f"{candidate}/{clock}/{name}": bool(value)
        for candidate, candidate_records in records.items()
        for clock, record in candidate_records.items()
        for name, value in record.items()
        if isinstance(value, bool)
    }
    if not all(checks.values()):
        raise ValueError("CCPR-1 evaluator schedule invariant failed")
    core: dict[str, Any] = {
        "protocol_version": "cross_collateral_positioning_recoil_evaluator_v1",
        "policy_id": "CCPR-1",
        "as_of_date": "2026-07-17",
        "support_commit": SUPPORT_COMMIT,
        "preregistration_manifest_hash": registration["manifest_hash"],
        "support_manifest_hash": support["manifest_hash"],
        "selected_q": support["selected_q"],
        "evaluator_source": str(EVALUATOR_SOURCE),
        "evaluator_source_sha256": _sha256(EVALUATOR_SOURCE),
        "strict_engine_sha256": STATIC_INPUT_SHA256[
            "training/evaluate_fiat_quote_participation_rotation.py"
        ],
        "evaluation_config": asdict(_verify_evaluation_contract(registration)),
        "static_inputs": STATIC_INPUT_SHA256,
        "schedule_records": records,
        "schedule_invariant_checks": checks,
        "opened_windows": [],
        "sealed_windows": ["stage1_2021_2022", "stage2_2023", "2024_plus"],
        "execution_ohlc_rows_parsed_during_freeze": 0,
        "funding_rows_parsed_during_freeze": 0,
        "simulation_run_during_freeze": 0,
        "mutable_parameters": [],
    }
    report = _seal(core)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def verify_evaluator_freeze(
    path: str | Path = EVALUATOR_FREEZE,
) -> dict[str, Any]:
    report = _load_json(path)
    manifest_hash = report.pop("manifest_hash", None)
    if manifest_hash != _canonical_hash(report):
        raise ValueError("CCPR-1 evaluator freeze manifest mismatch")
    report["manifest_hash"] = manifest_hash
    if report.get("evaluator_source_sha256") != _sha256(EVALUATOR_SOURCE):
        raise ValueError("CCPR-1 evaluator source changed after freeze")
    if report.get("opened_windows") != []:
        raise ValueError("CCPR-1 evaluator freeze opened a window")
    if report.get("execution_ohlc_rows_parsed_during_freeze") != 0:
        raise ValueError("CCPR-1 evaluator freeze parsed OHLC")
    if report.get("funding_rows_parsed_during_freeze") != 0:
        raise ValueError("CCPR-1 evaluator freeze parsed funding")
    if report.get("simulation_run_during_freeze") != 0:
        raise ValueError("CCPR-1 evaluator freeze ran a simulation")
    return report


def _verify_execution_manifests() -> None:
    registration = _load_json(PREREGISTRATION)
    source = registration["source_contract"]
    market_manifest = _load_json(MARKET_MANIFEST)
    if market_manifest.get("combined_sha256") != source["market_sha256"]:
        raise ValueError("CCPR-1 market manifest no longer binds the source")
    funding_manifest = _load_json(FUNDING_MANIFEST)
    if funding_manifest.get("data", {}).get("sha256") != source["funding_sha256"]:
        raise ValueError("CCPR-1 funding manifest no longer binds the source")
    if funding_manifest.get("outcomes_opened") is not False:
        raise ValueError("CCPR-1 funding manifest opened an outcome")


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
            "reason": "stop before parsing the sealed stage boundary",
        },
    )


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
    selected = _window_schedule(schedule, (period_start, period_end))
    metrics = strict_engine.simulate_schedule(
        market,
        funding,
        selected,
        period_start=period_start,
        period_end=period_end,
        cost_rate=cost_rate,
        cfg=cfg,
    )
    trades = pd.DataFrame(metrics["trade_details"])
    metrics["weekly_cluster_signflip"] = weekly_cluster_signflip_two_sided(
        trades, draws=cfg.cluster_draws, seed=cfg.cluster_seed
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
    significance = metrics["weekly_cluster_signflip"]
    return {
        "absolute_return_pct": metrics["absolute_return_pct"],
        "cagr_pct": metrics["cagr_pct"],
        "strict_mdd_pct": metrics["strict_mdd_pct"],
        "cagr_to_strict_mdd": metrics["cagr_to_strict_mdd"],
        "trades": metrics["trades"],
        "mean_gross_underlying_bp": metrics["mean_gross_underlying_bp"],
        "weekly_cluster_signflip_p": significance["p_value_two_sided"],
        "weekly_clusters": significance["cluster_count"],
    }


def _base_gates(
    metrics: dict[str, Any],
    stress: dict[str, Any],
    subperiods: dict[str, dict[str, Any]],
    *,
    stage: str,
) -> dict[str, bool]:
    if stage == "stage1":
        minimum_trades = 80
        p_limit = 0.025
    elif stage == "stage2":
        minimum_trades = 25
        p_limit = 0.05
    else:
        raise ValueError(f"CCPR-1 unknown stage: {stage}")
    return {
        "absolute_return_positive": metrics["absolute_return_pct"] > 0.0,
        "cagr_to_strict_mdd_at_least_3": metrics["cagr_to_strict_mdd"] >= 3.0,
        "strict_mdd_at_most_15pct": metrics["strict_mdd_pct"] <= 15.0,
        "minimum_trades": metrics["trades"] >= minimum_trades,
        "weekly_cluster_signflip_p_within_limit": metrics["weekly_cluster_signflip"][
            "p_value_two_sided"
        ]
        <= p_limit,
        "each_subperiod_absolute_return_positive": all(
            item["absolute_return_pct"] > 0.0 for item in subperiods.values()
        ),
        "stress_absolute_return_positive": stress["absolute_return_pct"] > 0.0,
        "stress_cagr_to_strict_mdd_at_least_2_5": stress["cagr_to_strict_mdd"] >= 2.5,
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
    gates = _base_gates(base, stress, split, stage=stage)
    return {
        "metrics": base,
        "headline": _headline(base),
        "stress_metrics": stress,
        "stress_headline": _headline(stress),
        "subperiod_metrics": split,
        "subperiod_headlines": {name: _headline(item) for name, item in split.items()},
        "entry_distribution": _entry_distribution(schedule, window),
        "gates": gates,
        "fully_qualified": bool(all(gates.values())),
    }


def _evaluate_candidate(
    candidate_id: str,
    schedules: dict[str, pd.DataFrame],
    market: pd.DataFrame,
    funding: pd.DataFrame,
    *,
    window: TimeWindow,
    subperiods: dict[str, TimeWindow],
    stage: str,
    cfg: EvaluationConfig,
) -> dict[str, Any]:
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
    primary = clocks["primary"]
    control_ratios = {
        name: clocks[name]["headline"]["cagr_to_strict_mdd"]
        for name in MECHANISM_CONTROLS
    }
    margin = min(
        primary["headline"]["cagr_to_strict_mdd"] - ratio
        for ratio in control_ratios.values()
    )
    candidate_gates = dict(primary["gates"])
    if stage == "stage1":
        candidate_gates["mechanism_control_margin_at_least_0_25"] = margin >= 0.25
    return {
        "candidate_id": candidate_id,
        "hold_bars": CANDIDATE_HOLDS[candidate_id],
        "primary": primary,
        "controls": {name: clocks[name] for name in ALL_CLOCK_NAMES[1:]},
        "mechanism_control_ratios": control_ratios,
        "minimum_mechanism_control_margin": margin,
        "gates": candidate_gates,
        "qualified": bool(all(candidate_gates.values())),
    }


def _selection_key(row: dict[str, Any]) -> tuple[Any, ...]:
    headline = row["primary"]["headline"]
    return (
        -headline["cagr_to_strict_mdd"],
        headline["strict_mdd_pct"],
        row["hold_bars"],
        row["candidate_id"],
    )


def _build_stage1_report() -> dict[str, Any]:
    freeze = verify_evaluator_freeze()
    registration, _ = _verify_static_inputs()
    schedules = load_schedules()
    market, funding, diagnostics = load_execution_window(STAGE1)
    cfg = _verify_evaluation_contract(registration)
    candidates = [
        _evaluate_candidate(
            candidate_id,
            schedules[candidate_id],
            market,
            funding,
            window=STAGE1,
            subperiods=STAGE1_SUBPERIODS,
            stage="stage1",
            cfg=cfg,
        )
        for candidate_id in CANDIDATE_IDS
    ]
    passing = sorted(
        (row for row in candidates if row["qualified"]), key=_selection_key
    )
    selected = passing[0]["candidate_id"] if passing else None
    core = {
        "protocol_version": "cross_collateral_positioning_recoil_stage1_v1",
        "policy_id": "CCPR-1",
        "stage": "stage1_2021_2022",
        "evaluator_freeze_manifest_hash": freeze["manifest_hash"],
        "evaluator_source_sha256": freeze["evaluator_source_sha256"],
        "config": asdict(cfg),
        "execution_diagnostics": diagnostics,
        "candidate_results": candidates,
        "selected_candidate_id": selected,
        "stage1_passed": selected is not None,
        "opened_windows": ["stage1_2021_2022"],
        "sealed_windows": ["stage2_2023", "2024_plus"],
        "disposition": (
            "ADVANCE_TO_SEALED_2023" if selected else "REJECT_KEEP_2023_SEALED"
        ),
    }
    return _seal(core)


def evaluate_stage1(
    output_path: str | Path = STAGE1_OUTPUT,
    doc_path: str | Path = STAGE1_DOC,
) -> dict[str, Any]:
    report = _build_stage1_report()
    Path(output_path).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    Path(doc_path).write_text(render_stage_doc(report), encoding="utf-8")
    return report


def _validate_stored_stage1_structure(report: dict[str, Any]) -> None:
    if set(report) != STAGE1_TOP_LEVEL_KEYS:
        raise ValueError("CCPR-1 stored Stage1 top-level structure changed")
    stored_hash = report["manifest_hash"]
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    if stored_hash != _canonical_hash(core):
        raise ValueError("CCPR-1 stored Stage1 manifest mismatch")
    if report["protocol_version"] != "cross_collateral_positioning_recoil_stage1_v1":
        raise ValueError("CCPR-1 stored Stage1 protocol changed")
    if report["stage"] != "stage1_2021_2022":
        raise ValueError("CCPR-1 stored Stage1 stage changed")
    if report["opened_windows"] != ["stage1_2021_2022"]:
        raise ValueError("CCPR-1 stored Stage1 opened windows changed")
    if report["sealed_windows"] != ["stage2_2023", "2024_plus"]:
        raise ValueError("CCPR-1 stored Stage1 sealed windows changed")
    candidates = report.get("candidate_results")
    if not isinstance(candidates, list) or [
        row.get("candidate_id") for row in candidates if isinstance(row, dict)
    ] != list(CANDIDATE_IDS):
        raise ValueError("CCPR-1 stored Stage1 candidate family changed")
    passing = [row for row in candidates if row.get("qualified") is True]
    expected = (
        sorted(passing, key=_selection_key)[0]["candidate_id"] if passing else None
    )
    if report["selected_candidate_id"] != expected:
        raise ValueError("CCPR-1 stored Stage1 selection changed")
    if report["stage1_passed"] is not (expected is not None):
        raise ValueError("CCPR-1 stored Stage1 pass flag changed")


def _verified_passing_stage1(*, expected_freeze_hash: str) -> dict[str, Any]:
    stored = _load_json(STAGE1_OUTPUT)
    _validate_stored_stage1_structure(stored)
    if stored["evaluator_freeze_manifest_hash"] != expected_freeze_hash:
        raise ValueError("CCPR-1 stored Stage1 freeze identity changed")
    if stored["evaluator_source_sha256"] != _sha256(EVALUATOR_SOURCE):
        raise ValueError("CCPR-1 stored Stage1 evaluator source changed")
    if stored["stage1_passed"] is not True:
        raise ValueError("CCPR-1 Stage1 failed; 2023 remains sealed")
    replayed = _build_stage1_report()
    if replayed != stored:
        raise ValueError("CCPR-1 stored Stage1 does not exactly reproduce")
    return stored


def evaluate_stage2(
    output_path: str | Path = STAGE2_OUTPUT,
    doc_path: str | Path = STAGE2_DOC,
) -> dict[str, Any]:
    freeze = verify_evaluator_freeze()
    stage1 = _verified_passing_stage1(expected_freeze_hash=freeze["manifest_hash"])
    registration, _ = _verify_static_inputs()
    candidate_id = stage1["selected_candidate_id"]
    if candidate_id not in CANDIDATE_IDS:
        raise ValueError("CCPR-1 Stage1 selected an unknown candidate")
    schedules = load_schedules()
    market, funding, diagnostics = load_execution_window(STAGE2)
    cfg = _verify_evaluation_contract(registration)
    result = _evaluate_candidate(
        candidate_id,
        schedules[candidate_id],
        market,
        funding,
        window=STAGE2,
        subperiods=STAGE2_SUBPERIODS,
        stage="stage2",
        cfg=cfg,
    )
    core = {
        "protocol_version": "cross_collateral_positioning_recoil_stage2_v1",
        "policy_id": "CCPR-1",
        "stage": "stage2_2023",
        "evaluator_freeze_manifest_hash": freeze["manifest_hash"],
        "evaluator_source_sha256": freeze["evaluator_source_sha256"],
        "verified_stage1_manifest_hash": stage1["manifest_hash"],
        "selected_candidate_id": candidate_id,
        "config": asdict(cfg),
        "execution_diagnostics": diagnostics,
        "result": result,
        "stage2_passed": result["qualified"],
        "advance_to_orthogonality_audit": result["qualified"],
        "opened_windows": ["stage1_2021_2022", "stage2_2023"],
        "sealed_windows": ["2024_plus"],
        "disposition": (
            "ADVANCE_TO_ORTHOGONALITY_AUDIT"
            if result["qualified"]
            else "REJECT_NO_REPAIR"
        ),
    }
    report = _seal(core)
    Path(output_path).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    Path(doc_path).write_text(render_stage_doc(report), encoding="utf-8")
    return report


def _metric_row(label: str, headline: dict[str, Any]) -> str:
    return (
        f"| {label} | {headline['absolute_return_pct']:.2f}% | "
        f"{headline['cagr_pct']:.2f}% | {headline['strict_mdd_pct']:.2f}% | "
        f"{headline['cagr_to_strict_mdd']:.2f} | {headline['trades']} | "
        f"{headline['weekly_cluster_signflip_p']:.4f} |"
    )


def render_stage_doc(report: dict[str, Any]) -> str:
    lines = [
        f"# CCPR-1 {report['stage']} result — 2026-07-17",
        "",
        "Metrics use full-calendar time, realized funding, next-open fills, and "
        "intratrade strict MDD. Absolute return is always shown with CAGR.",
        "",
        "| Candidate | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades | p(two-sided) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    if report["stage"] == "stage1_2021_2022":
        for row in report["candidate_results"]:
            lines.append(_metric_row(row["candidate_id"], row["primary"]["headline"]))
        lines.extend(
            [
                "",
                f"- Stage1 pass: **{report['stage1_passed']}**",
                f"- Selected: `{report['selected_candidate_id']}`",
                f"- Disposition: `{report['disposition']}`",
                "- 2023 execution outcomes remain sealed unless Stage1 passed.",
            ]
        )
        for row in report["candidate_results"]:
            lines.extend(["", f"## {row['candidate_id']} subperiods"])
            lines.extend(
                [
                    "",
                    "| Period | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades | p(two-sided) |",
                    "|---|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for name, headline in row["primary"]["subperiod_headlines"].items():
                lines.append(_metric_row(name, headline))
            failed = [name for name, value in row["gates"].items() if not value]
            lines.extend(["", f"Failed gates: `{failed}`"])
    else:
        result = report["result"]
        lines.append(
            _metric_row(report["selected_candidate_id"], result["primary"]["headline"])
        )
        lines.extend(
            [
                "",
                f"- Stage2 pass: **{report['stage2_passed']}**",
                f"- Disposition: `{report['disposition']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Integrity",
            "",
            f"- Evaluator source SHA-256: `{report['evaluator_source_sha256']}`",
            f"- Report manifest: `{report['manifest_hash']}`",
            f"- Physical execution window: `{report['execution_diagnostics']['physical_window']}`",
            "",
        ]
    )
    return "\n".join(lines)


def _print_stage(report: dict[str, Any]) -> None:
    if report["stage"] == "stage1_2021_2022":
        print(
            json.dumps(
                {
                    "stage": report["stage"],
                    "passed": report["stage1_passed"],
                    "selected": report["selected_candidate_id"],
                    "candidates": {
                        row["candidate_id"]: row["primary"]["headline"]
                        for row in report["candidate_results"]
                    },
                },
                indent=2,
            )
        )
    else:
        print(
            json.dumps(
                {
                    "stage": report["stage"],
                    "passed": report["stage2_passed"],
                    "selected": report["selected_candidate_id"],
                    "headline": report["result"]["primary"]["headline"],
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
        _print_stage(evaluate_stage1())
    else:
        _print_stage(evaluate_stage2())


if __name__ == "__main__":
    main()
