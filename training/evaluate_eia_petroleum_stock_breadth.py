"""Sequential strict evaluator for frozen EPSB-1.

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
from training import preregister_eia_petroleum_stock_breadth as prereg


SUPPORT_COMMIT = "353f6ea"
STATIC_INPUT_SHA256 = {
    "training/preregister_eia_petroleum_stock_breadth.py": "cf18fb3153cedef0c7672d6501f7507a76bfa8197670acf65378aea3762c176e",
    "training/eia_petroleum_stock_breadth_clock.py": "5adad78d86ed054caa1071dc659a8169b5170f494597fb0a056e0bcf27988de7",
    "docs/eia-petroleum-stock-breadth-preregistration-2026-07-17.md": "2dc7fea0e205a584e010b3dd95b31a4fa6fb2a40ed62714a5bda6c34113f1d25",
    "results/eia_petroleum_stock_breadth_preregistration_2026-07-17.json": "a6ebf316d4a030406448b5574eae4a14a7fd467561b2d6b304738efe4059414f",
    "results/eia_petroleum_stock_breadth_preregistered_clock_2026-07-17.csv.gz": "4f9735a3de7506e85f657217c8358608000d823927b6875c6cc236b94c39cedd",
    "training/build_eia_petroleum_stock_breadth_support.py": "0fc6772d10bacb697bf2e6c0cf37c2469eac9066e5dda918a2cdb5411154848b",
    "results/eia_petroleum_stock_breadth_support_2026-07-17.json": "806d9f91c7b52235725b884e1b240477a648ab214320f10b3970bc36a5c37f51",
    "results/eia_petroleum_stock_breadth_clocks_2026-07-17.csv.gz": "6c6470ba90e8bd826bb566e5952755dd8a872b29c1ba0643d29e08ab23e44400",
    "training/evaluate_fiat_quote_participation_rotation.py": "e309f5217f033d57d2eadfec936843e736ce287f5c47f957c0ac6f0c71879c23",
    "data/binance_um_kline_reference_btc_2020_2023/build_manifest.json": "c04fbbd299cc748a6745c0ef030787da4d560833c744c81c98dd8840efc7913e",
    "results/binance_um_btcusdt_funding_marks_2020_2023_manifest_2026-07-17.json": "a0b2d27e1aa8cf2d9ab8cb659b598ee0a6d7bd25401c9e10ae92d1a74415845b",
    "data/eia_petroleum_stock_breadth_2019_2023/eia_petroleum_stock_breadth_2019_2023.csv.gz": "26cbe6a91079a64fd9bbcb1cb5e1f81e15df25e45ed2171f7c464d048b34757b",
    "data/eia_petroleum_stock_breadth_2019_2023/build_manifest.json": "d6813b1a5677c9222a1197343900d6b03381f35ff9db8688892b77e4cd9c0661",
    "data/eia_petroleum_stock_breadth_2019_2023/source_manifest.json": "3969288900528d103016cdb0870a11269c1b352b9077faffdc61427f7fce29fb",
}
MARKET_DATA_SHA256 = "e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d"
FUNDING_DATA_SHA256 = "3284bbb6bb67946acb673c6b67459543e217f752589e1d47b6c7c3b659f733e6"

PREREGISTRATION = Path(prereg.DEFAULT_OUTPUT)
SUPPORT_RESULT = Path("results/eia_petroleum_stock_breadth_support_2026-07-17.json")
CLOCKS = Path("results/eia_petroleum_stock_breadth_clocks_2026-07-17.csv.gz")
MARKET = Path(prereg.MARKET_PATH)
FUNDING = Path("data/binance_um_btcusdt_funding_marks_2020_2023.csv.gz")
MARKET_MANIFEST = Path(prereg.MARKET_MANIFEST)
FUNDING_MANIFEST = Path(
    "results/binance_um_btcusdt_funding_marks_2020_2023_manifest_2026-07-17.json"
)
EVALUATOR_SOURCE = Path("training/evaluate_eia_petroleum_stock_breadth.py")
EVALUATOR_FREEZE = Path(
    "results/eia_petroleum_stock_breadth_evaluator_freeze_2026-07-17.json"
)
STAGE1_OUTPUT = Path(
    "results/eia_petroleum_stock_breadth_stage1_2020_2022_2026-07-17.json"
)
STAGE1_DOC = Path(
    "docs/eia-petroleum-stock-breadth-stage1-2020-2022-2026-07-17.md"
)
STAGE2_OUTPUT = Path(
    "results/eia_petroleum_stock_breadth_stage2_2023_2026-07-17.json"
)
STAGE2_DOC = Path(
    "docs/eia-petroleum-stock-breadth-stage2-2023-2026-07-17.md"
)

TimeWindow = tuple[pd.Timestamp, pd.Timestamp]


def _utc(value: str) -> pd.Timestamp:
    return cast(pd.Timestamp, pd.Timestamp(value, tz="UTC"))


def _timestamp(value: Any) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result is pd.NaT:
        raise ValueError("EPSB-1 timestamp is NaT")
    return cast(pd.Timestamp, result)


STAGE1: TimeWindow = (_utc("2020-01-01"), _utc("2023-01-01"))
STAGE1_SUBPERIODS: dict[str, TimeWindow] = {
    "2020": (_utc("2020-01-01"), _utc("2021-01-01")),
    "2021": (_utc("2021-01-01"), _utc("2022-01-01")),
    "2022": (_utc("2022-01-01"), _utc("2023-01-01")),
}
STAGE2: TimeWindow = (_utc("2023-01-01"), _utc("2024-01-01"))
STAGE2_SUBPERIODS: dict[str, TimeWindow] = {
    "2023_h1": (_utc("2023-01-01"), _utc("2023-07-01")),
    "2023_h2": (_utc("2023-07-01"), _utc("2024-01-01")),
}
MECHANISM_CONTROLS = ("crude_only", "refined_products_only")
FALSIFICATION_CONTROLS = (
    "direction_flip",
    "one_release_delay",
    "deterministic_random_side",
)
ALL_CLOCK_NAMES = ("primary", *MECHANISM_CONTROLS, *FALSIFICATION_CONTROLS)

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
        raise ValueError(f"EPSB-1 expected a JSON object: {path}")
    return payload


def _verify_registration(registration: dict[str, Any]) -> None:
    core = {
        key: value for key, value in registration.items() if key != "manifest_hash"
    }
    if registration.get("manifest_hash") != prereg.sha256_bytes(
        prereg.canonical_json(core)
    ):
        raise ValueError("EPSB-1 preregistration manifest mismatch")
    if registration.get("policy_id") != "EPSB-1":
        raise ValueError("EPSB-1 preregistration identity changed")
    if registration.get("opened_outcome_windows") != []:
        raise ValueError("EPSB-1 preregistration opened outcomes")


def _verify_evaluation_contract(registration: dict[str, Any]) -> EvaluationConfig:
    cfg = EvaluationConfig()
    policy = registration["policy"]
    for key in (
        "leverage",
        "base_cost_notional_per_side",
        "stress_cost_notional_per_side",
    ):
        if asdict(cfg)[key] != policy[key]:
            raise ValueError(f"EPSB-1 strict engine differs on {key}")
    if cfg.cluster_draws != 20_000 or cfg.cluster_seed != 20_260_717:
        raise ValueError("EPSB-1 sign-flip configuration changed")
    if cfg.mdd_denominator_floor != 1e-9:
        raise ValueError("EPSB-1 MDD denominator floor changed")
    stage = registration["stage_contract"]
    expected_stage1 = {
        "absolute_return_positive": True,
        "cagr_to_strict_mdd_at_least": 3.0,
        "strict_mdd_pct_at_most": 15.0,
        "weekly_cluster_signflip_p_at_most": 0.10,
        "minimum_trades": 30,
        "mean_gross_underlying_bp_at_least": 35.0,
        "stress_cost_absolute_return_positive": True,
        "each_calendar_year_absolute_return_positive": True,
        "each_calendar_year_minimum_trades": 8,
        "mechanism_control_ratio_margin_at_least": 0.25,
    }
    expected_stage2 = {
        "absolute_return_positive": True,
        "cagr_to_strict_mdd_at_least": 3.0,
        "strict_mdd_pct_at_most": 15.0,
        "weekly_cluster_signflip_p_at_most": 0.20,
        "minimum_trades": 10,
        "h1_and_h2_absolute_return_nonnegative": True,
        "stress_cost_absolute_return_positive": True,
    }
    if stage["stage1_gates"] != expected_stage1:
        raise ValueError("EPSB-1 Stage1 gate contract changed")
    if stage["stage2_gates"] != expected_stage2:
        raise ValueError("EPSB-1 Stage2 gate contract changed")
    return cfg


def _verify_static_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    for path, expected in STATIC_INPUT_SHA256.items():
        if _sha256(path) != expected:
            raise ValueError(f"EPSB-1 frozen input changed: {path}")
    registration = _load_json(PREREGISTRATION)
    _verify_registration(registration)
    _verify_evaluation_contract(registration)
    support = _load_json(SUPPORT_RESULT)
    support_core = {
        key: value for key, value in support.items() if key != "manifest_hash"
    }
    if support.get("manifest_hash") != _canonical_hash(support_core):
        raise ValueError("EPSB-1 support manifest changed")
    if support.get("policy_id") != "EPSB-1":
        raise ValueError("EPSB-1 support identity changed")
    if support.get("outcomes_opened") is not False:
        raise ValueError("EPSB-1 support opened outcomes")
    if support.get("outcome_sources_opened") != []:
        raise ValueError("EPSB-1 support opened outcome sources")
    if support.get("market_or_funding_rows_opened") != 0:
        raise ValueError("EPSB-1 support opened execution rows")
    if support.get("support_passed") is not True:
        raise ValueError("EPSB-1 support failed")
    if support.get("advance_to_stage1_outcomes") is not True:
        raise ValueError("EPSB-1 support did not authorize Stage1")
    if support.get("preregistration_manifest_hash") != registration["manifest_hash"]:
        raise ValueError("EPSB-1 support is not bound to preregistration")
    if support.get("clocks", {}).get("sha256") != STATIC_INPUT_SHA256[str(CLOCKS)]:
        raise ValueError("EPSB-1 support is not bound to the clock ledger")
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
    _verify_static_inputs()
    ledger = pd.read_csv(CLOCKS)
    if set(ledger["control"]) != set(ALL_CLOCK_NAMES):
        raise ValueError("EPSB-1 support clock family changed")
    schedules = {
        name: _schedule_frame(name, ledger.loc[ledger["control"].eq(name)])
        for name in ALL_CLOCK_NAMES
    }
    for name, schedule in schedules.items():
        if not bool(schedule["side"].isin((-1, 1)).all()):
            raise ValueError(f"EPSB-1 {name} side changed")
        if not bool(
            schedule["entry_time"]
            .eq(schedule["signal_day"] + pd.Timedelta(minutes=5))
            .all()
        ):
            raise ValueError(f"EPSB-1 {name} source delay changed")
        if not bool(
            schedule["exit_time"]
            .eq(schedule["entry_time"] + pd.Timedelta(hours=72))
            .all()
        ):
            raise ValueError(f"EPSB-1 {name} hold changed")
        if len(schedule) > 1:
            entries = schedule["entry_time"].iloc[1:].reset_index(drop=True)
            exits = schedule["exit_time"].iloc[:-1].reset_index(drop=True)
            if not bool(entries.ge(exits).all()):
                raise ValueError(f"EPSB-1 {name} schedule overlaps")
        if not bool(schedule["exit_time"].lt(STAGE2[1]).all()):
            raise ValueError(f"EPSB-1 {name} clock crosses source horizon")
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


def _schedule_record(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "events": int(len(frame)),
        "schedule_hash": _schedule_hash(frame),
        "first_entry": _timestamp(frame["entry_time"].min()).isoformat(),
        "last_exit": _timestamp(frame["exit_time"].max()).isoformat(),
        "entry_delay_exact": bool(
            frame["entry_time"].eq(frame["signal_day"] + pd.Timedelta(minutes=5)).all()
        ),
        "hold_exact": bool(
            frame["exit_time"].eq(frame["entry_time"] + pd.Timedelta(hours=72)).all()
        ),
        "globally_nonoverlapping": bool(
            frame["entry_time"]
            .iloc[1:]
            .reset_index(drop=True)
            .ge(frame["exit_time"].iloc[:-1].reset_index(drop=True))
            .all()
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
        for check in ("entry_delay_exact", "hold_exact", "globally_nonoverlapping")
    ):
        raise ValueError("EPSB-1 evaluator schedule invariant failed")
    core = {
        "protocol_version": "eia_petroleum_stock_breadth_evaluator_v1",
        "policy_id": "EPSB-1",
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
    report = _seal(core)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def verify_evaluator_freeze(path: str | Path = EVALUATOR_FREEZE) -> dict[str, Any]:
    report = _load_json(path)
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    if report.get("manifest_hash") != _canonical_hash(core):
        raise ValueError("EPSB-1 evaluator freeze manifest mismatch")
    if report.get("evaluator_source_sha256") != _sha256(EVALUATOR_SOURCE):
        raise ValueError("EPSB-1 evaluator source changed after freeze")
    if report.get("opened_windows") != []:
        raise ValueError("EPSB-1 evaluator freeze opened a window")
    if report.get("execution_ohlc_rows_parsed_during_freeze") != 0:
        raise ValueError("EPSB-1 evaluator freeze parsed OHLC")
    if report.get("funding_rows_parsed_during_freeze") != 0:
        raise ValueError("EPSB-1 evaluator freeze parsed funding")
    if report.get("simulation_run_during_freeze") is not False:
        raise ValueError("EPSB-1 evaluator freeze simulated outcomes")
    schedules = load_schedules()
    for name, schedule in schedules.items():
        if report["control_schedules"][name] != _schedule_record(schedule):
            raise ValueError(f"EPSB-1 {name} schedule changed after freeze")
    return report


def _verify_execution_manifests() -> None:
    market_manifest = _load_json(MARKET_MANIFEST)
    if market_manifest.get("combined_output") != str(MARKET):
        raise ValueError("EPSB-1 market manifest path changed")
    if market_manifest.get("combined_sha256") != MARKET_DATA_SHA256:
        raise ValueError("EPSB-1 market manifest data hash changed")
    if market_manifest.get("rows") != 420_768:
        raise ValueError("EPSB-1 market manifest row count changed")
    if market_manifest.get("protocol", {}).get("outcomes_opened") is not False:
        raise ValueError("EPSB-1 market manifest opened an outcome")
    funding_manifest = _load_json(FUNDING_MANIFEST)
    funding_data = funding_manifest.get("data", {})
    if funding_data.get("path") != str(FUNDING):
        raise ValueError("EPSB-1 funding manifest path changed")
    if funding_data.get("sha256") != FUNDING_DATA_SHA256:
        raise ValueError("EPSB-1 funding manifest data hash changed")
    if funding_data.get("rows") != 4_383:
        raise ValueError("EPSB-1 funding manifest row count changed")
    if funding_manifest.get("outcomes_opened") is not False:
        raise ValueError("EPSB-1 funding manifest opened an outcome")


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
        "reason": "avoid reading sealed rows beyond the physical stage window",
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
    return {
        "absolute_return_pct": metrics["absolute_return_pct"],
        "cagr_pct": metrics["cagr_pct"],
        "strict_mdd_pct": metrics["strict_mdd_pct"],
        "cagr_to_strict_mdd": metrics["cagr_to_strict_mdd"],
        "trades": metrics["trades"],
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
    if stage == "stage1_2020_2022":
        gates = {
            "absolute_return_positive": base["absolute_return_pct"] > 0.0,
            "cagr_to_strict_mdd_at_least_3": base["cagr_to_strict_mdd"] >= 3.0,
            "strict_mdd_at_most_15pct": base["strict_mdd_pct"] <= 15.0,
            "weekly_cluster_signflip_p_at_most_10pct": base[
                "weekly_cluster_signflip"
            ]["p_value_two_sided"]
            <= 0.10,
            "minimum_30_trades": base["trades"] >= 30,
            "mean_gross_underlying_at_least_35bp": base[
                "mean_gross_underlying_bp"
            ]
            >= 35.0,
            "stress_cost_absolute_return_positive": stress["absolute_return_pct"]
            > 0.0,
            "each_calendar_year_absolute_return_positive": all(
                value["absolute_return_pct"] > 0.0 for value in split.values()
            ),
            "each_calendar_year_minimum_8_trades": all(
                value["trades"] >= 8 for value in split.values()
            ),
        }
    else:
        gates = {
            "absolute_return_positive": base["absolute_return_pct"] > 0.0,
            "cagr_to_strict_mdd_at_least_3": base["cagr_to_strict_mdd"] >= 3.0,
            "strict_mdd_at_most_15pct": base["strict_mdd_pct"] <= 15.0,
            "weekly_cluster_signflip_p_at_most_20pct": base[
                "weekly_cluster_signflip"
            ]["p_value_two_sided"]
            <= 0.20,
            "minimum_10_trades": base["trades"] >= 10,
            "h1_and_h2_absolute_return_nonnegative": all(
                value["absolute_return_pct"] >= 0.0 for value in split.values()
            ),
            "stress_cost_absolute_return_positive": stress["absolute_return_pct"]
            > 0.0,
        }
    return {
        "metrics": base,
        "headline": _headline(base),
        "stress_metrics": stress,
        "stress_headline": _headline(stress),
        "subperiod_metrics": split,
        "subperiod_headlines": {name: _headline(value) for name, value in split.items()},
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
            "passed": primary_ratio
            - clocks[name]["headline"]["cagr_to_strict_mdd"]
            >= 0.25,
        }
        for name in MECHANISM_CONTROLS
    }
    gates = dict(clocks["primary"]["gates"])
    if stage == "stage1_2020_2022":
        gates["mechanism_control_margin_at_least_0_25"] = all(
            value["passed"] for value in comparisons.values()
        )
    passed = all(gates.values())
    core = {
        "protocol_version": f"eia_petroleum_stock_breadth_{stage}_v1",
        "policy_id": "EPSB-1",
        "stage": stage,
        "evaluator_freeze_manifest_hash": freeze["manifest_hash"],
        "evaluator_source_sha256": freeze["evaluator_source_sha256"],
        "config": asdict(cfg),
        "execution_diagnostics": diagnostics,
        "base_cost_by_clock": {name: value["metrics"] for name, value in clocks.items()},
        "headline_by_clock": {name: value["headline"] for name, value in clocks.items()},
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
    STAGE1_OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    STAGE1_DOC.write_text(render_stage_doc(report))
    return report


def _verified_passing_stage1(expected_freeze_hash: str) -> dict[str, Any]:
    if not STAGE1_OUTPUT.exists():
        raise ValueError("EPSB-1 Stage1 has not been run")
    stored = _load_json(STAGE1_OUTPUT)
    core = {key: value for key, value in stored.items() if key != "manifest_hash"}
    if stored.get("manifest_hash") != _canonical_hash(core):
        raise ValueError("EPSB-1 stored Stage1 manifest changed")
    if stored.get("evaluator_freeze_manifest_hash") != expected_freeze_hash:
        raise ValueError("EPSB-1 stored Stage1 freeze identity changed")
    if stored.get("evaluator_source_sha256") != _sha256(EVALUATOR_SOURCE):
        raise ValueError("EPSB-1 stored Stage1 evaluator changed")
    if stored.get("gate_passed") is not True:
        raise ValueError("EPSB-1 Stage1 failed; 2023 remains sealed")
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
        raise ValueError("EPSB-1 Stage1 does not exactly replay")
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
        f"{row['strict_mdd_pct']:.4f}% | {row['cagr_to_strict_mdd']:.4f} | "
        f"{row['trades']} |"
        for name, row in report["primary_subperiod_headlines"].items()
    )
    gate_rows = "\n".join(
        f"| `{name}` | {'PASS' if value else 'FAIL'} |"
        for name, value in report["gates"].items()
    )
    return f"""# EPSB-1 {report['stage']} result — {prereg.AS_OF_DATE}

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

