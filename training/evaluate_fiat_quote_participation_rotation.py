"""Sequential strict evaluator for the frozen FQPR-3 protocol.

Evaluator freeze verifies source-only schedules and opens no outcome source.
Stage 1 physically parses only the 2021-2022 market/funding window.  The 2023
window can be parsed only after a hash-bound Stage-1 result passes every frozen
gate.  2024 and later remain sealed in all paths.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_fiat_quote_participation_rotation as prereg


SUPPORT_COMMIT = "4c0a3166ee576ee45981f28e3ab11be4af7ccd19"
STATIC_INPUT_SHA256 = {
    "training/preregister_fiat_quote_participation_rotation.py": (
        "63bc5bcfbf258d9d845b70904b34e47923b9afd468f977080cc7bb17eff45555"
    ),
    "training/build_fiat_quote_participation_rotation_support.py": (
        "f3976d936b8ca538aaecf21bdf1fd26639452b016679ff9785a019cd083ad3af"
    ),
    "docs/fiat-quote-participation-rotation-preregistration-2026-07-17.md": (
        "d02371e103de7ca380d44277bec1b92d3ea078f5d12a36d9308abde08736cb00"
    ),
    "docs/fiat-quote-participation-rotation-support-2026-07-17.md": (
        "2dc51dda5207c6c686133bb2a2d5b739bda085235e1e609c7af0783798bc987b"
    ),
    "results/fiat_quote_participation_rotation_preregistration_2026-07-17.json": (
        "c9e7a3fde7ca421b2efb341da80a1f173c7e15d6b6d920978af5966e63ead21c"
    ),
    "results/fiat_quote_participation_rotation_support_2026-07-17.json": (
        "635bfa69b0d10d6766bf65f8673e573e0b9614c8585723f90b32193fd7343d4b"
    ),
    "results/fiat_quote_participation_rotation_clocks_2026-07-17.csv": (
        "54a70cce565d4f1727d095707471235f01345b94179a6c37df9f4c37d1a458a2"
    ),
    "data/binance_um_kline_reference_btc_2020_2023/build_manifest.json": (
        "c04fbbd299cc748a6745c0ef030787da4d560833c744c81c98dd8840efc7913e"
    ),
    "results/binance_um_btcusdt_funding_marks_2020_2023_manifest_2026-07-17.json": (
        "a0b2d27e1aa8cf2d9ab8cb659b598ee0a6d7bd25401c9e10ae92d1a74415845b"
    ),
}
PREREGISTRATION = Path(prereg.DEFAULT_OUTPUT)
SUPPORT_RESULT = Path(
    "results/fiat_quote_participation_rotation_support_2026-07-17.json"
)
CLOCKS = Path("results/fiat_quote_participation_rotation_clocks_2026-07-17.csv")
MARKET = Path(prereg.MARKET_PATH)
FUNDING = Path(prereg.FUNDING_PATH)
MARKET_MANIFEST = Path(prereg.MARKET_MANIFEST_PATH)
FUNDING_MANIFEST = Path(prereg.FUNDING_MANIFEST_PATH)
EVALUATOR_SOURCE = Path("training/evaluate_fiat_quote_participation_rotation.py")
EVALUATOR_FREEZE = Path(
    "results/fiat_quote_participation_rotation_evaluator_freeze_2026-07-17.json"
)
STAGE1_OUTPUT = Path(
    "results/fiat_quote_participation_rotation_stage1_2021_2022_2026-07-17.json"
)
STAGE1_DOC = Path(
    "docs/fiat-quote-participation-rotation-stage1-2021-2022-2026-07-17.md"
)
STAGE2_OUTPUT = Path(
    "results/fiat_quote_participation_rotation_stage2_2023_2026-07-17.json"
)
STAGE2_DOC = Path(
    "docs/fiat-quote-participation-rotation-stage2-2023-2026-07-17.md"
)

STAGE1 = (pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2023-01-01", tz="UTC"))
STAGE1_SUBPERIODS = {
    "2021_after_warmup": (
        pd.Timestamp("2021-06-30", tz="UTC"),
        pd.Timestamp("2022-01-01", tz="UTC"),
    ),
    "2022": (
        pd.Timestamp("2022-01-01", tz="UTC"),
        pd.Timestamp("2023-01-01", tz="UTC"),
    ),
}
STAGE2 = (pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2024-01-01", tz="UTC"))
STAGE2_SUBPERIODS = {
    "2023_h1": (
        pd.Timestamp("2023-01-01", tz="UTC"),
        pd.Timestamp("2023-07-01", tz="UTC"),
    ),
    "2023_h2": (
        pd.Timestamp("2023-07-01", tz="UTC"),
        pd.Timestamp("2024-01-01", tz="UTC"),
    ),
}
ALL_CLOCK_NAMES = (
    "primary",
    "direction_flip",
    "no_ticket",
    "no_taker",
    "volume_only",
    "flow_only",
    "single_book_eur",
    "single_book_try",
    "single_book_brl",
    "usdt_only",
    "reference_suppression",
    "absolute_book_participation",
    "one_day_signal_delay",
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
        "primary_ratio_strictly_beats_each_mechanism_control",
        "one_day_delay_and_random_do_not_fully_qualify",
    }
)


@dataclass(frozen=True)
class EvaluationConfig:
    leverage: float = 0.5
    base_cost_notional_per_side: float = 0.0006
    stress_cost_notional_per_side: float = 0.0010
    cluster_draws: int = 20_000
    cluster_seed: int = 20_260_717
    mdd_denominator_floor: float = 1e-9


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


def _verify_static_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    for path, expected in STATIC_INPUT_SHA256.items():
        if _sha256(path) != expected:
            raise ValueError(f"FQPR-3 frozen input changed: {path}")
    registration = _load_json(PREREGISTRATION)
    prereg.validate_manifest(registration, verify_sources=False)
    support = _load_json(SUPPORT_RESULT)
    if support.get("policy_id") != "FQPR-3":
        raise ValueError("FQPR-3 support identity changed")
    if support.get("outcomes_opened") is not False:
        raise ValueError("FQPR-3 support opened an outcome")
    if support.get("outcome_sources_opened") != []:
        raise ValueError("FQPR-3 support opened an outcome source")
    if support.get("support_passed") is not True:
        raise ValueError("FQPR-3 support did not pass")
    if support.get("advance_to_stage1_outcomes") is not True:
        raise ValueError("FQPR-3 support did not authorize Stage 1")
    if support.get("selected_q") != 0.65:
        raise ValueError("FQPR-3 selected Q changed")
    if support.get("clocks", {}).get("sha256") != STATIC_INPUT_SHA256[str(CLOCKS)]:
        raise ValueError("FQPR-3 support does not bind the frozen clocks")
    return registration, support


def _schedule_hash(frame: pd.DataFrame) -> str:
    rows = [
        {
            "clock_name": str(row.clock_name),
            "source_signal_day": pd.Timestamp(row.source_signal_day).isoformat(),
            "signal_day": pd.Timestamp(row.signal_day).isoformat(),
            "decision_time": pd.Timestamp(row.decision_time).isoformat(),
            "entry_time": pd.Timestamp(row.entry_time).isoformat(),
            "exit_time": pd.Timestamp(row.exit_time).isoformat(),
            "side": int(row.side),
            "q": float(row.q),
        }
        for row in frame.itertuples(index=False)
    ]
    return _canonical_hash(rows)


def load_schedules() -> dict[str, pd.DataFrame]:
    _, support = _verify_static_inputs()
    frame = pd.read_csv(CLOCKS)
    timestamp_columns = (
        "source_signal_day",
        "signal_day",
        "decision_time",
        "entry_time",
        "exit_time",
    )
    for column in timestamp_columns:
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    frame["side"] = frame["side"].map({"LONG": 1, "SHORT": -1})
    if frame["side"].isna().any():
        raise ValueError("FQPR-3 schedule has an invalid side")
    frame["side"] = frame["side"].astype("int8")
    if not frame["q"].eq(0.65).all():
        raise ValueError("FQPR-3 clock Q differs from support selection")
    if set(frame["clock_name"]) != set(ALL_CLOCK_NAMES):
        raise ValueError("FQPR-3 clock family differs from the frozen family")
    policy = _load_json(PREREGISTRATION)["policy"]
    expected_hold = pd.Timedelta(minutes=5 * int(policy["hold_bars"]))
    expected_delay = pd.Timedelta(minutes=5 * int(policy["execution_delay_bars"]))
    if not frame["decision_time"].eq(frame["signal_day"] + pd.Timedelta(days=1)).all():
        raise ValueError("FQPR-3 decision clock drifted")
    if not frame["entry_time"].eq(frame["decision_time"] + expected_delay).all():
        raise ValueError("FQPR-3 entry delay drifted")
    if not frame["exit_time"].eq(frame["entry_time"] + expected_hold).all():
        raise ValueError("FQPR-3 hold drifted")
    delayed = frame["clock_name"].eq("one_day_signal_delay")
    if not frame.loc[delayed, "signal_day"].eq(
        frame.loc[delayed, "source_signal_day"] + pd.Timedelta(days=1)
    ).all():
        raise ValueError("FQPR-3 one-day control did not shift exactly one day")
    if not frame.loc[~delayed, "signal_day"].eq(
        frame.loc[~delayed, "source_signal_day"]
    ).all():
        raise ValueError("FQPR-3 non-delay clock shifted its signal")
    if frame["entry_time"].lt(STAGE1[0]).any() or frame["exit_time"].gt(STAGE2[1]).any():
        raise ValueError("FQPR-3 schedule crosses the frozen pre-2024 source window")

    schedules: dict[str, pd.DataFrame] = {}
    expected_counts = support["clocks"]["clock_counts"]
    for name in ALL_CLOCK_NAMES:
        schedule = (
            frame.loc[frame["clock_name"].eq(name)]
            .sort_values("entry_time")
            .reset_index(drop=True)
        )
        if len(schedule) != int(expected_counts[name]):
            raise ValueError(f"FQPR-3 {name} clock count changed")
        if len(schedule) > 1 and not schedule["entry_time"].iloc[1:].reset_index(
            drop=True
        ).ge(schedule["exit_time"].iloc[:-1].reset_index(drop=True)).all():
            raise ValueError(f"FQPR-3 {name} schedule overlaps")
        schedules[name] = schedule
    primary = schedules["primary"]
    for name in ("direction_flip", "random_side"):
        if not schedules[name][["entry_time", "exit_time"]].equals(
            primary[["entry_time", "exit_time"]]
        ):
            raise ValueError(f"FQPR-3 {name} does not use the primary clock")
    if not schedules["direction_flip"]["side"].eq(-1).all():
        raise ValueError("FQPR-3 direction flip is not short")
    for name in ALL_CLOCK_NAMES:
        if name not in {"direction_flip", "random_side"} and not schedules[name][
            "side"
        ].eq(1).all():
            raise ValueError(f"FQPR-3 {name} is not fixed long")
    return schedules


def _window_schedule(
    frame: pd.DataFrame, window: tuple[pd.Timestamp, pd.Timestamp]
) -> pd.DataFrame:
    start, end = window
    return frame.loc[
        frame["signal_day"].ge(start)
        & frame["entry_time"].ge(start)
        & frame["exit_time"].le(end)
    ].copy()


def freeze_evaluator(output_path: str | Path = EVALUATOR_FREEZE) -> dict[str, Any]:
    registration, support = _verify_static_inputs()
    schedules = load_schedules()
    counts = {
        name: {
            "all_pre2024": int(len(schedule)),
            "stage1_2021_2022": int(len(_window_schedule(schedule, STAGE1))),
            "stage2_2023": int(len(_window_schedule(schedule, STAGE2))),
            "schedule_hash": _schedule_hash(schedule),
        }
        for name, schedule in schedules.items()
    }
    core = {
        "protocol_version": "fiat_quote_participation_rotation_evaluator_v1",
        "policy_id": "FQPR-3",
        "as_of_date": "2026-07-17",
        "support_commit": SUPPORT_COMMIT,
        "support_manifest_hash": support["preregistration"]["manifest_hash"],
        "selected_q": support["selected_q"],
        "evaluator_source": str(EVALUATOR_SOURCE),
        "evaluator_source_sha256": _sha256(EVALUATOR_SOURCE),
        "evaluation_config": asdict(EvaluationConfig()),
        "static_inputs": dict(STATIC_INPUT_SHA256),
        "execution_source_contract": {
            "market": str(MARKET),
            "market_expected_sha256_copied_without_read": registration["source_contract"][
                "market_sha256"
            ],
            "funding": str(FUNDING),
            "funding_expected_sha256_copied_without_read": registration[
                "source_contract"
            ]["funding_sha256"],
            "stage1_physical_window": [STAGE1[0].isoformat(), STAGE1[1].isoformat()],
            "stage2_physical_window": [STAGE2[0].isoformat(), STAGE2[1].isoformat()],
        },
        "control_schedules": counts,
        "opened_windows": [],
        "sealed_windows": ["stage1_2021_2022", "stage2_2023", "2024", "2025", "2026_ytd"],
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


def verify_evaluator_freeze(
    path: str | Path = EVALUATOR_FREEZE,
) -> dict[str, Any]:
    payload = _load_json(path)
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != _canonical_hash(core):
        raise ValueError("FQPR-3 evaluator freeze hash changed")
    if payload.get("evaluator_source_sha256") != _sha256(EVALUATOR_SOURCE):
        raise ValueError("FQPR-3 evaluator source changed after freeze")
    if payload.get("support_commit") != SUPPORT_COMMIT:
        raise ValueError("FQPR-3 evaluator froze another support commit")
    if payload.get("evaluation_config") != asdict(EvaluationConfig()):
        raise ValueError("FQPR-3 evaluation configuration changed")
    if payload.get("opened_windows") != [] or payload.get("mutable_parameters") != []:
        raise ValueError("FQPR-3 evaluator freeze is not sealed")
    if payload.get("execution_ohlc_rows_parsed_during_freeze") != 0:
        raise ValueError("FQPR-3 evaluator freeze parsed OHLC")
    if payload.get("funding_rows_parsed_during_freeze") != 0:
        raise ValueError("FQPR-3 evaluator freeze parsed funding")
    if payload.get("simulation_run_during_freeze") is not False:
        raise ValueError("FQPR-3 evaluator freeze simulated an outcome")
    schedules = load_schedules()
    for name, schedule in schedules.items():
        expected = payload["control_schedules"][name]
        if expected["schedule_hash"] != _schedule_hash(schedule):
            raise ValueError(f"FQPR-3 {name} schedule changed after freeze")
    return payload


def _verify_execution_manifests() -> None:
    registration = _load_json(PREREGISTRATION)
    source = registration["source_contract"]
    market_manifest = _load_json(MARKET_MANIFEST)
    if market_manifest.get("combined_sha256") != source["market_sha256"]:
        raise ValueError("FQPR-3 market manifest no longer binds the frozen source")
    funding_manifest = _load_json(FUNDING_MANIFEST)
    if funding_manifest.get("data", {}).get("sha256") != source["funding_sha256"]:
        raise ValueError("FQPR-3 funding manifest no longer binds the frozen source")
    if funding_manifest.get("outcomes_opened") is not False:
        raise ValueError("FQPR-3 funding source manifest opened an outcome")


def _parse_market_window(
    path: str | Path, start: pd.Timestamp, end: pd.Timestamp
) -> tuple[pd.DataFrame, dict[str, Any]]:
    wanted = ("date", "open", "high", "low", "close")
    rows: list[tuple[Any, ...]] = []
    window_hash = hashlib.sha256()
    boundary_seen = False
    start_text = start.tz_convert(None).strftime("%Y-%m-%d %H:%M:%S")
    end_text = end.tz_convert(None).strftime("%Y-%m-%d %H:%M:%S")
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        header_line = handle.readline()
        header = header_line.rstrip("\r\n").split(",")
        positions = {column: header.index(column) for column in wanted}
        if positions["date"] != 0:
            raise ValueError("FQPR-3 market timestamp must be the first column")
        for line in handle:
            timestamp_text = line.split(",", 1)[0]
            if timestamp_text < start_text:
                continue
            if timestamp_text >= end_text:
                boundary_seen = True
                break
            fields = line.rstrip("\r\n").split(",")
            rows.append(
                (
                    timestamp_text,
                    float(fields[positions["open"]]),
                    float(fields[positions["high"]]),
                    float(fields[positions["low"]]),
                    float(fields[positions["close"]]),
                )
            )
            window_hash.update(line.encode("utf-8"))
    frame = pd.DataFrame(rows, columns=wanted)
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="raise")
    expected = pd.date_range(start, end, freq="5min", inclusive="left")
    if not pd.DatetimeIndex(frame["date"]).equals(expected):
        raise ValueError("FQPR-3 market window does not match the exact five-minute grid")
    values = frame[["open", "high", "low", "close"]].to_numpy(float)
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise ValueError("FQPR-3 market contains invalid prices")
    opening, high, low, close = values.T
    if (
        (high < np.maximum(opening, close)).any()
        or (low > np.minimum(opening, close)).any()
        or (high < low).any()
    ):
        raise ValueError("FQPR-3 market violates OHLC invariants")
    return frame, {
        "rows": int(len(frame)),
        "first_timestamp": frame["date"].min().isoformat(),
        "last_timestamp": frame["date"].max().isoformat(),
        "window_line_sha256": window_hash.hexdigest(),
        "stopped_before_parsing_end_boundary": boundary_seen,
    }


def _parse_funding_window(
    path: str | Path, start: pd.Timestamp, end: pd.Timestamp
) -> tuple[pd.DataFrame, dict[str, Any]]:
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
        for line in handle:
            fields = line.rstrip("\r\n").split(",")
            timestamp_text = fields[positions["funding_time_utc"]]
            if timestamp_text < start_text:
                continue
            if timestamp_text >= end_text:
                boundary_seen = True
                break
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
    frame["funding_time"] = pd.to_datetime(frame["funding_time"], utc=True, errors="raise")
    expected = pd.date_range(start, end, freq="8h", inclusive="left")
    if not pd.DatetimeIndex(frame["funding_time"]).equals(expected):
        raise ValueError("FQPR-3 funding window does not match the exact eight-hour grid")
    if not frame["symbol"].eq("BTCUSDT").all():
        raise ValueError("FQPR-3 funding source changed symbol")
    values = frame[["funding_rate", "settlement_mark_price"]].to_numpy(float)
    if not np.isfinite(values).all() or (frame["settlement_mark_price"] <= 0.0).any():
        raise ValueError("FQPR-3 funding contains invalid values")
    return frame, {
        "rows": int(len(frame)),
        "first_timestamp": frame["funding_time"].min().isoformat(),
        "last_timestamp": frame["funding_time"].max().isoformat(),
        "window_line_sha256": window_hash.hexdigest(),
        "stopped_before_parsing_end_boundary": boundary_seen,
    }


def load_execution_window(
    window: tuple[pd.Timestamp, pd.Timestamp]
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


def weekly_cluster_signflip(
    trades: pd.DataFrame, *, draws: int, seed: int
) -> dict[str, Any]:
    if trades.empty:
        return {
            "p_value_one_sided": 1.0,
            "cluster_count": 0,
            "draws": draws,
            "seed": seed,
            "observed_mean_net_return": 0.0,
        }
    entry = pd.to_datetime(trades["entry_time"], utc=True)
    iso = entry.dt.isocalendar()
    keys = iso.year.astype(str) + "-W" + iso.week.astype(str).str.zfill(2)
    weekly = trades["net_return"].groupby(keys).sum()
    values = weekly.to_numpy(float)
    observed = float(trades["net_return"].mean())
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
    cfg: EvaluationConfig = EvaluationConfig(),
) -> dict[str, Any]:
    if cfg != EvaluationConfig():
        raise ValueError("FQPR-3 evaluation configuration is frozen")
    if not 0.0 <= cost_rate < 0.1 or period_end <= period_start:
        raise ValueError("invalid FQPR-3 simulation window or cost")
    selected = _window_schedule(schedule, (period_start, period_end))
    indexed = market.set_index("date", drop=False)
    if not indexed.index.is_unique:
        raise ValueError("FQPR-3 market timestamps are not unique")
    funding_at = {
        pd.Timestamp(row.funding_time): (
            float(row.funding_rate),
            float(row.settlement_mark_price),
        )
        for row in funding.itertuples(index=False)
    }
    equity = 1.0
    high_water = 1.0
    strict_mdd = 0.0
    marks: dict[pd.Timestamp, float] = {period_start: equity}
    total_cost = 0.0
    total_funding = 0.0
    total_price_pnl = 0.0
    trades: list[dict[str, Any]] = []
    previous_exit: pd.Timestamp | None = None

    for event in selected.itertuples(index=False):
        entry_time = pd.Timestamp(event.entry_time)
        exit_time = pd.Timestamp(event.exit_time)
        if previous_exit is not None and entry_time < previous_exit:
            raise ValueError("FQPR-3 simulation schedule overlaps")
        previous_exit = exit_time
        if entry_time not in indexed.index or exit_time not in indexed.index:
            raise ValueError("FQPR-3 entry or exit open is unavailable")
        held = market.loc[
            market["date"].ge(entry_time) & market["date"].lt(exit_time)
        ]
        expected_bars = int((exit_time - entry_time) / pd.Timedelta(minutes=5))
        if len(held) != expected_bars:
            raise ValueError("FQPR-3 held path is incomplete")
        side = int(event.side)
        if side not in {-1, 1}:
            raise ValueError("FQPR-3 schedule side is invalid")
        entry_price = float(indexed.loc[entry_time, "open"])
        exit_price = float(indexed.loc[exit_time, "open"])
        pre_equity = equity
        quantity = cfg.leverage * pre_equity / entry_price
        entry_fee = cost_rate * quantity * entry_price
        total_cost += entry_fee
        equity_after_entry = pre_equity - entry_fee
        marks[entry_time] = equity_after_entry
        strict_mdd = max(strict_mdd, 1.0 - equity_after_entry / max(high_water, 1e-15))
        cumulative_funding = 0.0

        for bar in held.itertuples(index=False):
            timestamp = pd.Timestamp(bar.date)
            if timestamp in funding_at:
                rate, settlement_mark = funding_at[timestamp]
                cumulative_funding += -side * quantity * rate * settlement_mark
            favorable = float(bar.high if side > 0 else bar.low)
            favorable_equity = (
                pre_equity
                - entry_fee
                + side * quantity * (favorable - entry_price)
                + cumulative_funding
            )
            high_water = max(high_water, favorable_equity)
            adverse = float(bar.low if side > 0 else bar.high)
            hypothetical_exit_fee = cost_rate * quantity * adverse
            adverse_equity = (
                pre_equity
                - entry_fee
                + side * quantity * (adverse - entry_price)
                + cumulative_funding
                - hypothetical_exit_fee
            )
            strict_mdd = max(
                strict_mdd,
                1.0 - adverse_equity / max(high_water, 1e-15),
            )
            close_equity = (
                pre_equity
                - entry_fee
                + side * quantity * (float(bar.close) - entry_price)
                + cumulative_funding
            )
            marks[timestamp + pd.Timedelta(minutes=5)] = close_equity

        price_pnl = side * quantity * (exit_price - entry_price)
        exit_fee = cost_rate * quantity * exit_price
        total_cost += exit_fee
        total_price_pnl += price_pnl
        total_funding += cumulative_funding
        equity = pre_equity - entry_fee + price_pnl + cumulative_funding - exit_fee
        marks[exit_time] = equity
        strict_mdd = max(strict_mdd, 1.0 - equity / max(high_water, 1e-15))
        high_water = max(high_water, equity)
        trades.append(
            {
                "clock_name": str(event.clock_name),
                "entry_time": entry_time.isoformat(),
                "exit_time": exit_time.isoformat(),
                "side": side,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "quantity_btc": quantity,
                "gross_underlying_bp": side
                * (exit_price / entry_price - 1.0)
                * 10_000.0,
                "price_pnl": price_pnl,
                "funding_cash": cumulative_funding,
                "entry_fee": entry_fee,
                "exit_fee": exit_fee,
                "net_pnl": equity - pre_equity,
                "net_return": equity / pre_equity - 1.0,
                "funding_events": int(
                    sum(entry_time <= time < exit_time for time in funding_at)
                ),
            }
        )

    marks[period_end] = equity
    marked = pd.Series(marks, dtype=float).sort_index()
    marked = marked[~marked.index.duplicated(keep="last")]
    daily_index = pd.date_range(period_start, period_end, freq="1D", inclusive="both")
    daily_equity = marked.reindex(marked.index.union(daily_index)).sort_index().ffill()
    daily_equity = daily_equity.reindex(daily_index)
    years = (period_end - period_start).total_seconds() / (365.25 * 86_400.0)
    absolute_return = equity - 1.0
    cagr = equity ** (1.0 / years) - 1.0 if equity > 0.0 else -1.0
    ratio = cagr / max(strict_mdd, cfg.mdd_denominator_floor)
    trade_frame = pd.DataFrame(trades)
    significance = weekly_cluster_signflip(
        trade_frame,
        draws=cfg.cluster_draws,
        seed=cfg.cluster_seed,
    )
    mean_gross = float(trade_frame["gross_underlying_bp"].mean()) if len(trade_frame) else 0.0
    return {
        "period_start": period_start.isoformat(),
        "period_end": period_end.isoformat(),
        "calendar_years": float(years),
        "absolute_return_pct": float(absolute_return * 100.0),
        "cagr_pct": float(cagr * 100.0),
        "strict_mdd_pct": float(strict_mdd * 100.0),
        "cagr_to_strict_mdd": float(ratio),
        "trades": int(len(trades)),
        "ending_equity": float(equity),
        "mean_gross_underlying_bp": mean_gross,
        "transaction_cost_pct_initial": float(total_cost * 100.0),
        "price_pnl_pct_initial": float(total_price_pnl * 100.0),
        "funding_cash_pct_initial": float(total_funding * 100.0),
        "weekly_cluster_signflip": significance,
        "nonzero_daily_pnl_days": int(
            daily_equity.diff().fillna(0.0).abs().gt(1e-15).sum()
        ),
        "daily_equity": {
            timestamp.isoformat(): float(value)
            for timestamp, value in daily_equity.items()
        },
        "trade_details": trades,
    }


def _simulate_subperiods(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    schedule: pd.DataFrame,
    subperiods: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
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
    return {
        "absolute_return_positive": metrics["absolute_return_pct"] > 0.0,
        "cagr_to_strict_mdd_at_least_3": metrics["cagr_to_strict_mdd"] >= 3.0,
        "strict_mdd_at_most_15pct": metrics["strict_mdd_pct"] <= 15.0,
        "weekly_cluster_signflip_p_at_most_10pct": metrics[
            "weekly_cluster_signflip"
        ]["p_value_one_sided"]
        <= 0.10,
        "minimum_trades": metrics["trades"] >= total_trades_min,
        "mean_gross_underlying_at_least_35bp": metrics[
            "mean_gross_underlying_bp"
        ]
        >= 35.0,
        "stress_cost_absolute_return_positive": stress["absolute_return_pct"] > 0.0,
        "each_subperiod_absolute_return_positive": all(
            row["absolute_return_pct"] > 0.0 for row in subperiods.values()
        ),
        "each_subperiod_minimum_trades": all(
            subperiods[name]["trades"] >= minimum
            for name, minimum in subperiod_trade_mins.items()
        ),
    }


def _headline(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: metrics[key]
        for key in (
            "absolute_return_pct",
            "cagr_pct",
            "strict_mdd_pct",
            "cagr_to_strict_mdd",
            "trades",
            "mean_gross_underlying_bp",
        )
    }


def evaluate_stage1() -> dict[str, Any]:
    freeze = verify_evaluator_freeze()
    registration = _load_json(PREREGISTRATION)
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
    primary_gates = _base_qualification(
        base["primary"],
        primary_stress,
        primary_subperiods,
        total_trades_min=40,
        subperiod_trade_mins={"2021_after_warmup": 20, "2022": 18},
    )
    full_control_qualification: dict[str, Any] = {}
    for name in ("one_day_signal_delay", "random_side"):
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
        gates = _base_qualification(
            base[name],
            stress,
            subperiods,
            total_trades_min=40,
            subperiod_trade_mins={"2021_after_warmup": 20, "2022": 18},
        )
        full_control_qualification[name] = {
            "stress_cost": stress,
            "subperiods": subperiods,
            "gates": gates,
            "passed": bool(all(gates.values())),
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
        "one_day_delay_and_random_do_not_fully_qualify": not any(
            row["passed"] for row in full_control_qualification.values()
        ),
    }
    passed = bool(all(gates.values()))
    core = {
        "protocol_version": "fiat_quote_participation_rotation_stage1_v1",
        "policy_id": "FQPR-3",
        "stage": "stage1_2021_2022",
        "evaluator_freeze_manifest_hash": freeze["manifest_hash"],
        "evaluator_source_sha256": _sha256(EVALUATOR_SOURCE),
        "config": asdict(cfg),
        "execution_diagnostics": diagnostics,
        "base_cost_by_clock": base,
        "primary_stress_cost": primary_stress,
        "primary_subperiods": primary_subperiods,
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


def _verified_passing_stage1(
    *, expected_freeze_manifest_hash: str
) -> dict[str, Any]:
    if not STAGE1_OUTPUT.exists():
        raise ValueError("FQPR-3 Stage 1 has not been run")
    report = _load_json(STAGE1_OUTPUT)
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    if report.get("manifest_hash") != _canonical_hash(core):
        raise ValueError("FQPR-3 Stage-1 result hash changed")
    if report.get("policy_id") != "FQPR-3" or report.get("stage") != (
        "stage1_2021_2022"
    ):
        raise ValueError("FQPR-3 Stage-1 result identity changed")
    if report.get("evaluator_freeze_manifest_hash") != expected_freeze_manifest_hash:
        raise ValueError("FQPR-3 Stage 1 is not bound to the current evaluator freeze")
    if report.get("evaluator_source_sha256") != _sha256(EVALUATOR_SOURCE):
        raise ValueError("FQPR-3 Stage 1 used another evaluator")
    if report.get("config") != asdict(EvaluationConfig()):
        raise ValueError("FQPR-3 Stage-1 evaluation configuration changed")
    if report.get("opened_windows") != ["stage1_2021_2022"]:
        raise ValueError("FQPR-3 Stage-1 opened-window contract changed")
    if report.get("sealed_windows") != [
        "stage2_2023",
        "2024",
        "2025",
        "2026_ytd",
    ]:
        raise ValueError("FQPR-3 Stage-1 sealed-window contract changed")
    diagnostics = report.get("execution_diagnostics")
    if not isinstance(diagnostics, dict) or diagnostics.get("physical_window") != [
        STAGE1[0].isoformat(),
        STAGE1[1].isoformat(),
    ]:
        raise ValueError("FQPR-3 Stage-1 physical window changed")
    gates = report.get("gates")
    if (
        not isinstance(gates, dict)
        or set(gates) != STAGE1_GATE_NAMES
        or not all(value is True for value in gates.values())
    ):
        raise ValueError("FQPR-3 Stage-1 gate evidence is incomplete")
    if report.get("gate_passed") is not True:
        raise ValueError("FQPR-3 Stage 1 did not authorize opening 2023")
    if report.get("disposition") != "PASS_STAGE1_OPEN_2023_ONCE":
        raise ValueError("FQPR-3 Stage-1 disposition does not authorize 2023")
    return report


def evaluate_stage2() -> dict[str, Any]:
    freeze = verify_evaluator_freeze()
    stage1 = _verified_passing_stage1(
        expected_freeze_manifest_hash=freeze["manifest_hash"]
    )
    registration = _load_json(PREREGISTRATION)
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
    primary_gates = _base_qualification(
        base["primary"],
        primary_stress,
        primary_subperiods,
        total_trades_min=20,
        subperiod_trade_mins={"2023_h1": 8, "2023_h2": 8},
    )
    full_control_qualification: dict[str, Any] = {}
    for name in ("one_day_signal_delay", "random_side"):
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
        gates = _base_qualification(
            base[name],
            stress,
            subperiods,
            total_trades_min=20,
            subperiod_trade_mins={"2023_h1": 8, "2023_h2": 8},
        )
        full_control_qualification[name] = {
            "stress_cost": stress,
            "subperiods": subperiods,
            "gates": gates,
            "passed": bool(all(gates.values())),
        }
    mechanism_controls = registration["selection_protocol"][
        "control_comparison_contract"
    ]["mechanism_controls"]
    primary_min_ratio = min(
        stage1["base_cost_by_clock"]["primary"]["cagr_to_strict_mdd"],
        base["primary"]["cagr_to_strict_mdd"],
    )
    ratio_comparison = {}
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
        for name in ("one_day_signal_delay", "random_side")
    }
    gates = {
        **primary_gates,
        "minimum_train_2023_ratio_strictly_beats_each_mechanism_control": all(
            row["passed"] for row in ratio_comparison.values()
        ),
        "one_day_delay_and_random_do_not_fully_qualify_both_splits": not any(
            stale_or_random_full.values()
        ),
    }
    passed = bool(all(gates.values()))
    core = {
        "protocol_version": "fiat_quote_participation_rotation_stage2_v1",
        "policy_id": "FQPR-3",
        "stage": "stage2_2023",
        "evaluator_freeze_manifest_hash": freeze["manifest_hash"],
        "stage1_manifest_hash": stage1["manifest_hash"],
        "evaluator_source_sha256": _sha256(EVALUATOR_SOURCE),
        "config": asdict(cfg),
        "execution_diagnostics": diagnostics,
        "base_cost_by_clock": base,
        "primary_stress_cost": primary_stress,
        "primary_subperiods": primary_subperiods,
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
    return f"""# FQPR-3 {stage} result

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

- Evaluator freeze: `{report['evaluator_freeze_manifest_hash']}`
- Physically opened window: `{report['execution_diagnostics']['physical_window']}`
- Still sealed: {sealed}
- Full-clock CAGR includes idle cash.
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
    parser = argparse.ArgumentParser()
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
