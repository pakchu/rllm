"""Strict sequential pre-2024 evaluator for Packet Churn Persistence (PCP-6).

The evaluator has three irreversible stages: freeze, train, and selection.  The
train stage only parses outcome rows before 2023.  The selection stage refuses
to parse 2023 outcomes unless the write-once train artifact passed every frozen
gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, cast

import numpy as np
import pandas as pd

from training.evaluate_metaorder_fragmentation_impact_curvature import (
    weekly_cluster_sign_flip,
)
from training.preregister_packet_churn_persistence import canonical_hash
from training.search_inventory_purge_reclaim_alpha import (
    Config as EngineConfig,
    ExecutionEngine,
    Trade,
)


PREREGISTRATION_COMMIT = "47cfd8ab86a4e86898a1be64b6791ba0cb26a727"
PREREGISTRATION_SOURCE = Path("training/preregister_packet_churn_persistence.py")
PREREGISTRATION_SOURCE_SHA256 = (
    "51ae73d77c1af6dafea79c8abd140646ec5c2536b7771bdddb8d810cb01827ca"
)
PREREGISTRATION_DOCUMENT = Path(
    "docs/packet-churn-persistence-preregistration-2026-07-19.md"
)
PREREGISTRATION_DOCUMENT_SHA256 = (
    "c80e5597f9f4e0f73af163f0e49053a97f7371f7ed77d3bc7e2c4918249d3e76"
)
SUPPORT_RESULT = Path("results/packet_churn_persistence_support_2026-07-19.json")
SUPPORT_RESULT_SHA256 = (
    "c8bdbf3d16dac7623f8f291c98a11d9e40cc14defdadb4d6206364bdbc64dc4f"
)
SUPPORT_CLOCK = Path("results/packet_churn_persistence_clock_2026-07-19.csv")
SUPPORT_CLOCK_SHA256 = (
    "e50be3f0744617e3797581ea762546e7f21c32dd0707b9bdcbd264686d4f9acb"
)
EXECUTION_SOURCE = Path("training/search_inventory_purge_reclaim_alpha.py")
EXECUTION_SOURCE_SHA256 = (
    "5d8d4df7ea79790afb919bbb481d11de33ecba5768f6e26feb1f7667cd947d65"
)
CLUSTER_SOURCE = Path("training/evaluate_metaorder_fragmentation_impact_curvature.py")
CLUSTER_SOURCE_SHA256 = (
    "1589a52605386570485a7e6be3b8f3aa9439a498abb60eaa42272ac62d4cbed3"
)
EVALUATION_SOURCE = Path("training/evaluate_packet_churn_persistence_pre2024.py")
EVALUATION_DOCUMENT = Path(
    "docs/packet-churn-persistence-evaluator-contract-2026-07-19.md"
)

SELECTED_NAME = "pcp_cross_venue_churn_breakout_p70_s35_h96_confirm6"
PHYSICAL_END = "2024-01-01"
TRAIN_END = "2023-01-01"
HOLD_BARS = 96

WINDOWS: dict[str, tuple[str, str]] = {
    "train": ("2020-01-01", "2023-01-01"),
    "train_2020": ("2020-01-01", "2021-01-01"),
    "train_2021": ("2021-01-01", "2022-01-01"),
    "train_2022": ("2022-01-01", "2023-01-01"),
    "selection_2023": ("2023-01-01", "2024-01-01"),
    "selection_2023_h1": ("2023-01-01", "2023-07-01"),
    "selection_2023_h2": ("2023-07-01", "2024-01-01"),
}


@dataclass(frozen=True)
class EvaluationConfig:
    feature_csv: str = (
        "/home/pakchu/rllm/data/binance_cross_venue_minute_dispersion_btc/"
        "BTCUSDT_cross_venue_minute_dispersion_5m_2020-01_2023-12.csv.gz"
    )
    market_csv: str = (
        "/home/pakchu/rllm/data/binance_um_kline_reference_btc_2020_2023/"
        "BTCUSDT_5m_2020-01-01_2023-12-31.csv.gz"
    )
    funding_csv: str = (
        "/home/pakchu/rllm/results/binance_um_btcusdt_realized_funding_2020_2023.csv"
    )
    freeze_output: str = (
        "results/packet_churn_persistence_evaluator_freeze_2026-07-19.json"
    )
    train_output: str = (
        "results/packet_churn_persistence_train_2020_2022_2026-07-19.json"
    )
    selection_output: str = (
        "results/packet_churn_persistence_selection_2023_2026-07-19.json"
    )
    leverage: float = 0.5
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0001
    stress_cost_rate: float = 0.0010
    cluster_permutations: int = 100_000
    cluster_seed: int = 20_260_719


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _engine_config(cfg: EvaluationConfig) -> EngineConfig:
    return EngineConfig(
        input_csv=cfg.market_csv,
        metrics_csv="unused",
        funding_csv=cfg.funding_csv,
        output="unused",
        manifest_output="unused",
        leverage=cfg.leverage,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
        stress_cost_rate=cfg.stress_cost_rate,
    )


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _result_hash(payload: dict[str, Any]) -> str:
    return canonical_hash(
        {
            key: value
            for key, value in payload.items()
            if key not in {"created_at", "result_hash"}
        }
    )


def _write_json_once(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload["result_hash"] = _result_hash(payload)
    with output.open("x") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _verify_result_hash(payload: dict[str, Any], *, label: str) -> None:
    if payload.get("result_hash") != _result_hash(payload):
        raise ValueError(f"{label} result hash does not replay")


def _verify_static_dependencies() -> dict[str, Any]:
    frozen = (
        (PREREGISTRATION_SOURCE, PREREGISTRATION_SOURCE_SHA256),
        (PREREGISTRATION_DOCUMENT, PREREGISTRATION_DOCUMENT_SHA256),
        (SUPPORT_RESULT, SUPPORT_RESULT_SHA256),
        (SUPPORT_CLOCK, SUPPORT_CLOCK_SHA256),
        (EXECUTION_SOURCE, EXECUTION_SOURCE_SHA256),
        (CLUSTER_SOURCE, CLUSTER_SOURCE_SHA256),
    )
    for path, expected in frozen:
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(f"frozen dependency changed: {path}: {observed}")
    support = _read_json(SUPPORT_RESULT)
    protocol = support.get("protocol", {})
    if protocol.get("outcomes_opened") is not False:
        raise ValueError("PCP support artifact already opened outcomes")
    if protocol.get("successor_train_outcomes_opened") is not False:
        raise ValueError("PCP train was opened before evaluator freeze")
    if support.get("support_stopping_rule", {}).get("selected_name") != SELECTED_NAME:
        raise ValueError("PCP support winner changed")
    if support.get("clock", {}).get("sha256") != SUPPORT_CLOCK_SHA256:
        raise ValueError("PCP clock hash differs from support report")
    if support.get("source", {}).get("sha256") != sha256_file(
        support["source"]["path"]
    ):
        raise ValueError("PCP feature source changed after support freeze")
    return support


def _load_clock() -> pd.DataFrame:
    clock = pd.read_csv(SUPPORT_CLOCK)
    required = {
        "setup_position",
        "confirmation_end_position",
        "entry_position",
        "exit_position",
        "side",
        "setup_bar_date",
        "confirmation_end_bar_date",
        "signal_available_at",
        "entry_date",
        "exit_date",
    }
    if not required.issubset(clock.columns):
        raise ValueError("PCP clock is missing execution fields")
    if len(clock) != 192:
        raise ValueError("PCP support clock row count changed")
    positions = clock[
        [
            "setup_position",
            "confirmation_end_position",
            "entry_position",
            "exit_position",
        ]
    ].to_numpy(int)
    if not np.all(positions[:, 1] - positions[:, 0] == 6):
        raise ValueError("PCP confirmation length changed")
    if not np.all(positions[:, 2] - positions[:, 1] == 2):
        raise ValueError("PCP entry latency changed")
    if not np.all(positions[:, 3] - positions[:, 2] == HOLD_BARS):
        raise ValueError("PCP hold changed")
    confirmation_end = pd.to_datetime(clock["confirmation_end_bar_date"])
    available = pd.to_datetime(clock["signal_available_at"])
    entries = pd.to_datetime(clock["entry_date"])
    exits = pd.to_datetime(clock["exit_date"])
    if not (available == confirmation_end + pd.Timedelta("5min")).all():
        raise ValueError("PCP signal availability changed")
    if not (entries == available + pd.Timedelta("5min")).all():
        raise ValueError("PCP entry timestamp changed")
    if not (exits == entries + pd.Timedelta(minutes=5 * HOLD_BARS)).all():
        raise ValueError("PCP exit timestamp changed")
    sides = clock["side"].to_numpy(int)
    if not np.isin(sides, (-1, 1)).all():
        raise ValueError("PCP clock contains an invalid side")
    if len(clock) > 1 and not np.all(positions[1:, 2] >= positions[:-1, 3]):
        raise ValueError("PCP clock contains overlapping positions")
    return clock


def execution_clock_hash(clock: pd.DataFrame) -> str:
    columns = [
        "setup_position",
        "confirmation_end_position",
        "entry_position",
        "exit_position",
        "side",
        "setup_bar_date",
        "confirmation_end_bar_date",
        "signal_available_at",
        "entry_date",
        "exit_date",
    ]
    execution = cast(pd.DataFrame, clock.loc[:, columns])
    records = cast(list[dict[str, Any]], execution.to_dict(orient="records"))
    return canonical_hash(records)


def shift_execution_clock(clock: pd.DataFrame, bars: int) -> pd.DataFrame:
    """Shift entry and exit together while preserving the frozen signal."""

    shifted = clock.copy()
    shifted["entry_position"] = shifted["entry_position"].astype(int) + int(bars)
    shifted["exit_position"] = shifted["exit_position"].astype(int) + int(bars)
    delta = pd.Timedelta(minutes=5 * int(bars))
    shifted["entry_date"] = (pd.to_datetime(shifted["entry_date"]) + delta).astype(str)
    shifted["exit_date"] = (pd.to_datetime(shifted["exit_date"]) + delta).astype(str)
    positions = shifted[["confirmation_end_position", "entry_position"]].to_numpy(int)
    if not np.all(positions[:, 1] > positions[:, 0]):
        raise ValueError("control enters before confirmation is available")
    if len(shifted) > 1 and not np.all(
        shifted["entry_position"].to_numpy(int)[1:]
        >= shifted["exit_position"].to_numpy(int)[:-1]
    ):
        raise ValueError("latency control creates overlapping trades")
    return shifted


def _timestamp_column(path: str | Path, column: str, *, utc: bool) -> pd.Series:
    read_csv = cast(Any, pd.read_csv)
    raw = read_csv(path, compression="infer", usecols=[column])
    timestamps = pd.to_datetime(raw[column], utc=utc, errors="raise", format="mixed")
    if utc:
        timestamps = timestamps.dt.tz_convert(None)
    timestamps = cast(pd.Series, timestamps.reset_index(drop=True))
    if timestamps.empty:
        raise ValueError(f"empty timestamp source: {path}")
    if timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
        raise ValueError(f"unordered or duplicate timestamp source: {path}")
    return timestamps


def _outcome_boundaries(cfg: EvaluationConfig) -> dict[str, Any]:
    feature_dates = _timestamp_column(cfg.feature_csv, "date", utc=False)
    market_dates = _timestamp_column(cfg.market_csv, "date", utc=False)
    funding_dates = _timestamp_column(cfg.funding_csv, "funding_time_utc", utc=True)
    if not feature_dates.equals(market_dates):
        raise ValueError(
            "PCP features and official market outcomes use different grids"
        )
    if market_dates.iloc[-1] >= pd.Timestamp(PHYSICAL_END):
        raise ValueError("market outcome source is not physically pre-2024")
    if funding_dates.iloc[-1] >= pd.Timestamp(PHYSICAL_END):
        raise ValueError("funding outcome source is not physically pre-2024")
    split = np.datetime64(TRAIN_END)
    market_train_rows = int(
        np.searchsorted(market_dates.to_numpy(dtype="datetime64[ns]"), split)
    )
    funding_train_rows = int(
        np.searchsorted(funding_dates.to_numpy(dtype="datetime64[ns]"), split)
    )
    if market_train_rows <= 0 or market_train_rows >= len(market_dates):
        raise ValueError("market source cannot be split at 2023")
    if funding_train_rows <= 0 or funding_train_rows >= len(funding_dates):
        raise ValueError("funding source cannot be split at 2023")
    return {
        "market": {
            "rows_pre2023": market_train_rows,
            "rows_pre2024": int(len(market_dates)),
            "first_date": str(market_dates.iloc[0]),
            "last_date": str(market_dates.iloc[-1]),
        },
        "funding": {
            "rows_pre2023": funding_train_rows,
            "rows_pre2024": int(len(funding_dates)),
            "first_date": str(funding_dates.iloc[0]),
            "last_date": str(funding_dates.iloc[-1]),
        },
    }


def _verify_clock_against_feature_grid(
    cfg: EvaluationConfig,
    clock: pd.DataFrame,
) -> None:
    dates = _timestamp_column(cfg.feature_csv, "date", utc=False)
    fields = (
        ("setup_position", "setup_bar_date"),
        ("confirmation_end_position", "confirmation_end_bar_date"),
        ("entry_position", "entry_date"),
        ("exit_position", "exit_date"),
    )
    for position_column, date_column in fields:
        positions = clock[position_column].to_numpy(int)
        if positions.min() < 0 or positions.max() >= len(dates):
            raise ValueError(f"PCP {position_column} exceeds the feature grid")
        expected = dates.iloc[positions].reset_index(drop=True)
        observed = pd.to_datetime(clock[date_column]).reset_index(drop=True)
        if not observed.equals(expected):
            raise ValueError(
                f"PCP {date_column} differs from its feature-grid position"
            )


def freeze_evaluator(cfg: EvaluationConfig) -> dict[str, Any]:
    support = _verify_static_dependencies()
    if sha256_file(cfg.feature_csv) != support["source"]["sha256"]:
        raise ValueError("configured feature source differs from the support freeze")
    for result in (cfg.train_output, cfg.selection_output):
        if Path(result).exists():
            raise ValueError("an outcome result exists before evaluator freeze")
    if Path(cfg.freeze_output).exists():
        raise FileExistsError("PCP evaluator freeze is write-once")
    clock = _load_clock()
    _verify_clock_against_feature_grid(cfg, clock)
    immediate = shift_execution_clock(clock, -1)
    extra_latency = shift_execution_clock(clock, 1)
    boundaries = _outcome_boundaries(cfg)
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "outcomes_opened": False,
            "outcome_value_columns_parsed": False,
            "timestamp_columns_parsed_for_physical_boundaries": True,
            "opened_windows": [],
            "sealed_windows": [
                "train_2020_2022",
                "selection_2023",
                "test_2024",
                "eval_2025",
                "holdout_2026",
            ],
            "selection_loader_requires_passing_train_artifact": True,
            "selection_loader_requires_exact_train_replay": True,
            "controls_cannot_replace_primary": True,
            "strict_mdd_contract": (
                "global/pre-entry HWM plus entry cost, favorable-before-adverse "
                "held 5m OHLC, realized funding debit, virtual adverse-mark "
                "liquidation cost, and actual exit cost"
            ),
            "funding_boundary_policy": (
                "interior settlements symmetric; exact entry/exit timestamp "
                "debits included and credits excluded"
            ),
        },
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "evaluation_source": str(EVALUATION_SOURCE),
        "evaluation_source_sha256": sha256_file(EVALUATION_SOURCE),
        "evaluation_document": str(EVALUATION_DOCUMENT),
        "evaluation_document_sha256": sha256_file(EVALUATION_DOCUMENT),
        "config": asdict(cfg),
        "static_dependencies": {
            "preregistration_source_sha256": PREREGISTRATION_SOURCE_SHA256,
            "preregistration_document_sha256": PREREGISTRATION_DOCUMENT_SHA256,
            "support_result_sha256": SUPPORT_RESULT_SHA256,
            "support_clock_sha256": SUPPORT_CLOCK_SHA256,
            "execution_source_sha256": EXECUTION_SOURCE_SHA256,
            "cluster_source_sha256": CLUSTER_SOURCE_SHA256,
            "feature_source_sha256": support["source"]["sha256"],
            "market_source_sha256": sha256_file(cfg.market_csv),
            "funding_source_sha256": sha256_file(cfg.funding_csv),
        },
        "outcome_boundaries": boundaries,
        "frozen_clocks": {
            "primary": {
                "rows": int(len(clock)),
                "hash": execution_clock_hash(clock),
                "entry_delay_bars_from_confirmation_end": 2,
            },
            "same_clock_side_flip_control": {
                "rows": int(len(clock)),
                "hash": execution_clock_hash(clock),
                "entry_delay_bars_from_confirmation_end": 2,
            },
            "immediate_entry_control": {
                "rows": int(len(immediate)),
                "hash": execution_clock_hash(immediate),
                "entry_delay_bars_from_confirmation_end": 1,
            },
            "extra_latency_control": {
                "rows": int(len(extra_latency)),
                "hash": execution_clock_hash(extra_latency),
                "entry_delay_bars_from_confirmation_end": 3,
            },
        },
        "gates": {
            "train": {
                "absolute_return_pct": ">0",
                "cagr_to_strict_mdd": ">=1.5",
                "strict_mdd_pct": "<=15",
                "trades": ">=100",
                "ten_bp_per_side_stress_absolute_return_pct": ">0",
                "weekly_cluster_sign_flip_p_value_one_sided": "<0.10",
            },
            "selection_2023": {
                "absolute_return_pct": ">0",
                "cagr_to_strict_mdd": ">=3",
                "strict_mdd_pct": "<=15",
                "h1_and_h2_absolute_return_pct": ">0",
                "ten_bp_per_side_stress_absolute_return_pct": ">0",
                "weekly_cluster_sign_flip_p_value_one_sided": "<0.10",
            },
        },
    }
    _write_json_once(cfg.freeze_output, report)
    return report


def verify_evaluator_freeze(cfg: EvaluationConfig) -> dict[str, Any]:
    _verify_static_dependencies()
    freeze = _read_json(cfg.freeze_output)
    _verify_result_hash(freeze, label="evaluator freeze")
    protocol = freeze.get("protocol", {})
    if protocol.get("outcomes_opened") is not False:
        raise ValueError("evaluator freeze opened outcomes")
    if protocol.get("outcome_value_columns_parsed") is not False:
        raise ValueError("evaluator freeze parsed outcome values")
    if freeze.get("evaluation_source_sha256") != sha256_file(EVALUATION_SOURCE):
        raise ValueError("evaluator source changed after freeze")
    if freeze.get("evaluation_document_sha256") != sha256_file(EVALUATION_DOCUMENT):
        raise ValueError("evaluator contract changed after freeze")
    if freeze.get("config") != asdict(cfg):
        raise ValueError("evaluation config changed after freeze")
    dependencies = freeze["static_dependencies"]
    if dependencies["feature_source_sha256"] != sha256_file(cfg.feature_csv):
        raise ValueError("feature source changed after evaluator freeze")
    if dependencies["market_source_sha256"] != sha256_file(cfg.market_csv):
        raise ValueError("market outcome source changed after freeze")
    if dependencies["funding_source_sha256"] != sha256_file(cfg.funding_csv):
        raise ValueError("funding outcome source changed after freeze")
    clock = _load_clock()
    _verify_clock_against_feature_grid(cfg, clock)
    controls = freeze["frozen_clocks"]
    if controls["primary"]["hash"] != execution_clock_hash(clock):
        raise ValueError("primary execution clock changed after freeze")
    if controls["immediate_entry_control"]["hash"] != execution_clock_hash(
        shift_execution_clock(clock, -1)
    ):
        raise ValueError("immediate-entry control changed after freeze")
    if controls["extra_latency_control"]["hash"] != execution_clock_hash(
        shift_execution_clock(clock, 1)
    ):
        raise ValueError("extra-latency control changed after freeze")
    return freeze


def _load_market_prefix(
    cfg: EvaluationConfig,
    *,
    rows: int,
) -> pd.DataFrame:
    read_csv = cast(Any, pd.read_csv)
    market = read_csv(
        cfg.market_csv,
        compression="infer",
        nrows=int(rows),
        usecols=["date", "open", "high", "low", "close"],
        parse_dates=["date"],
    )
    if len(market) != int(rows):
        raise ValueError("market prefix row count changed")
    dates = cast(pd.Series, market["date"])
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("market prefix timestamps are invalid")
    numeric = market[["open", "high", "low", "close"]].apply(
        pd.to_numeric, errors="raise"
    )
    values = numeric.to_numpy(float)
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise ValueError("market prefix contains invalid OHLC")
    return market


def _load_funding_prefix(
    cfg: EvaluationConfig,
    *,
    rows: int,
) -> pd.DataFrame:
    read_csv = cast(Any, pd.read_csv)
    raw = read_csv(
        cfg.funding_csv,
        nrows=int(rows),
        usecols=["funding_time_utc", "funding_rate"],
    )
    if len(raw) != int(rows):
        raise ValueError("funding prefix row count changed")
    funding = pd.DataFrame(
        {
            "date": pd.to_datetime(
                raw["funding_time_utc"], utc=True, errors="raise", format="mixed"
            ).dt.tz_convert(None),
            "funding_rate": pd.to_numeric(raw["funding_rate"], errors="raise"),
        }
    )
    if (
        funding["date"].duplicated().any()
        or not funding["date"].is_monotonic_increasing
    ):
        raise ValueError("funding prefix timestamps are invalid")
    if not np.isfinite(funding["funding_rate"].to_numpy(float)).all():
        raise ValueError("funding prefix contains invalid rates")
    return funding


def _window_clock(clock: pd.DataFrame, window: str) -> pd.DataFrame:
    start, end = WINDOWS[window]
    lower = pd.Timestamp(start)
    upper = pd.Timestamp(end)
    setup = pd.to_datetime(clock["setup_bar_date"])
    available = pd.to_datetime(clock["signal_available_at"])
    entries = pd.to_datetime(clock["entry_date"])
    exits = pd.to_datetime(clock["exit_date"])
    mask = (
        (setup >= lower) & (available >= lower) & (entries >= lower) & (exits < upper)
    )
    return cast(pd.DataFrame, clock.loc[mask].copy())


def _build_trades(
    engine: ExecutionEngine,
    schedule: pd.DataFrame,
    *,
    flip: bool = False,
) -> list[Trade]:
    trades: list[Trade] = []
    previous_exit = -1
    entries = schedule["entry_position"].to_numpy(int)
    exits = schedule["exit_position"].to_numpy(int)
    sides = schedule["side"].to_numpy(int)
    for entry, exit_position, raw_side in zip(entries, exits, sides, strict=True):
        entry = int(entry)
        exit_position = int(exit_position)
        if entry < previous_exit:
            raise ValueError("frozen schedule overlaps")
        side = -int(raw_side) if flip else int(raw_side)
        trade = engine.trade_at(
            entry - 1,
            side,
            exit_position - entry,
            1_000_000,
            1_000_000,
        )
        if trade is None:
            raise ValueError("frozen schedule exceeds the loaded market prefix")
        if trade.entry_position != entry or trade.exit_position != exit_position:
            raise ValueError("execution engine changed the frozen entry or exit")
        trade = _apply_conservative_boundary_funding(trade, engine)
        trades.append(trade)
        previous_exit = exit_position
    return trades


def _apply_conservative_boundary_funding(
    trade: Trade,
    engine: ExecutionEngine,
) -> Trade:
    """Exclude boundary credits while retaining boundary debits.

    A position opened or closed exactly at a funding timestamp has ambiguous
    settlement ordering.  Interior settlements are applied symmetrically.  At
    either boundary, only an adverse payment is retained so the backtest cannot
    benefit from optimistic timestamp ordering.
    """

    entry_ns = int(engine.dates.iloc[trade.entry_position].value)
    exit_ns = int(engine.dates.iloc[trade.exit_position].value)
    interior_left = int(np.searchsorted(engine.funding_times, entry_ns, side="right"))
    interior_right = int(np.searchsorted(engine.funding_times, exit_ns, side="left"))
    interior_rates = engine.funding_rates[interior_left:interior_right]
    boundary_mask = (engine.funding_times == entry_ns) | (
        engine.funding_times == exit_ns
    )
    boundary_rates = engine.funding_rates[boundary_mask]
    leverage = float(engine.cfg.leverage)
    interior_factors = 1.0 - leverage * trade.side * interior_rates
    boundary_factors = np.minimum(
        1.0 - leverage * trade.side * boundary_rates,
        1.0,
    )
    factors = np.concatenate([interior_factors, boundary_factors])
    if not np.isfinite(factors).all() or (factors <= 0.0).any():
        raise ValueError("invalid conservative funding factor")
    funding_factor = float(np.prod(factors, dtype=float)) if len(factors) else 1.0
    debit_factors = np.minimum(factors, 1.0)
    debit_factor = (
        float(np.prod(debit_factors, dtype=float)) if len(debit_factors) else 1.0
    )
    return replace(
        trade,
        funding_factor=funding_factor,
        funding_debit_factor=debit_factor,
    )


def _net_trade_returns(
    trades: Iterable[Trade],
    cfg: EvaluationConfig,
    *,
    cost_rate: float | None = None,
) -> list[float]:
    cost = cfg.fee_rate + cfg.slippage_rate if cost_rate is None else float(cost_rate)
    execution = 1.0 - cfg.leverage * cost
    return [
        float(execution * trade.price_factor * trade.funding_factor * execution - 1.0)
        for trade in trades
    ]


def strict_equity_stats(
    trades: Iterable[Trade],
    *,
    start: str,
    end: str,
    cfg: EvaluationConfig,
    cost_rate: float | None = None,
) -> dict[str, Any]:
    """Compute full-calendar return and conservative strict drawdown.

    The held-path adverse mark pays both the actual entry cost and a virtual
    liquidation cost.  The favorable mark omits virtual liquidation cost,
    intentionally preserving the higher (more conservative) intratrade HWM.
    """

    cost = float(cfg.fee_rate + cfg.slippage_rate if cost_rate is None else cost_rate)
    execution = 1.0 - cfg.leverage * cost
    if not 0.0 < execution <= 1.0:
        raise ValueError("invalid execution factor")
    equity = 1.0
    peak = 1.0
    strict_mdd = 0.0
    net_returns: list[float] = []
    gross_returns: list[float] = []
    sides: list[int] = []
    for trade in trades:
        entry_equity = equity
        favorable_factor = execution * trade.favorable_price_factor
        adverse_liquidation_factor = (
            execution
            * trade.funding_debit_factor
            * trade.adverse_price_factor
            * execution
        )
        intratrade_peak = max(peak, equity * favorable_factor)
        strict_mdd = max(
            strict_mdd,
            1.0 - equity * adverse_liquidation_factor / intratrade_peak,
        )
        peak = intratrade_peak
        equity *= execution * trade.price_factor * trade.funding_factor * execution
        strict_mdd = max(strict_mdd, 1.0 - equity / peak)
        peak = max(peak, equity)
        net_returns.append(equity / entry_equity - 1.0)
        gross_returns.append(trade.gross_return)
        sides.append(trade.side)
    years = (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() / (
        365.25 * 86_400.0
    )
    absolute_return = (equity - 1.0) * 100.0
    cagr = (equity ** (1.0 / years) - 1.0) * 100.0 if equity > 0.0 else -100.0
    mdd = strict_mdd * 100.0
    returns = np.asarray(net_returns, dtype=float)
    return {
        "absolute_return_pct": float(absolute_return),
        "cagr_pct": float(cagr),
        "strict_mdd_pct": float(mdd),
        "cagr_to_strict_mdd": float(cagr / mdd) if mdd > 1e-12 else 0.0,
        "trades": int(len(returns)),
        "longs": int(sum(side > 0 for side in sides)),
        "shorts": int(sum(side < 0 for side in sides)),
        "mean_net_bps": float(returns.mean() * 10_000.0) if len(returns) else 0.0,
        "mean_gross_bps": (
            float(np.mean(gross_returns) * 10_000.0) if gross_returns else 0.0
        ),
        "win_rate": float((returns > 0.0).mean()) if len(returns) else 0.0,
    }


def _stats(
    trades: list[Trade],
    window: str,
    cfg: EvaluationConfig,
    *,
    include_cluster: bool,
    cost_rate: float | None = None,
) -> dict[str, Any]:
    start, end = WINDOWS[window]
    output = strict_equity_stats(
        trades,
        start=start,
        end=end,
        cfg=cfg,
        cost_rate=cost_rate,
    )
    if include_cluster:
        output["weekly_cluster_sign_flip"] = weekly_cluster_sign_flip(
            _net_trade_returns(trades, cfg, cost_rate=cost_rate),
            [trade.entry_date for trade in trades],
            permutations=cfg.cluster_permutations,
            seed=cfg.cluster_seed,
        )
    return output


def train_gates(
    base: dict[str, Any],
    stress: dict[str, Any],
) -> dict[str, bool]:
    return {
        "absolute_return_positive": base["absolute_return_pct"] > 0.0,
        "cagr_to_strict_mdd_at_least_1_5": base["cagr_to_strict_mdd"] >= 1.5,
        "strict_mdd_at_most_15": base["strict_mdd_pct"] <= 15.0,
        "trades_at_least_100": base["trades"] >= 100,
        "ten_bp_per_side_stress_positive": stress["absolute_return_pct"] > 0.0,
        "weekly_cluster_p_below_0_10": base["weekly_cluster_sign_flip"][
            "p_value_one_sided"
        ]
        < 0.10,
    }


def selection_gates(
    full: dict[str, Any],
    h1: dict[str, Any],
    h2: dict[str, Any],
    stress: dict[str, Any],
) -> dict[str, bool]:
    return {
        "absolute_return_positive": full["absolute_return_pct"] > 0.0,
        "cagr_to_strict_mdd_at_least_3": full["cagr_to_strict_mdd"] >= 3.0,
        "strict_mdd_at_most_15": full["strict_mdd_pct"] <= 15.0,
        "each_half_absolute_return_positive": min(
            h1["absolute_return_pct"], h2["absolute_return_pct"]
        )
        > 0.0,
        "ten_bp_per_side_stress_positive": stress["absolute_return_pct"] > 0.0,
        "weekly_cluster_p_below_0_10": full["weekly_cluster_sign_flip"][
            "p_value_one_sided"
        ]
        < 0.10,
    }


def _control_stats(
    engine: ExecutionEngine,
    clock: pd.DataFrame,
    window: str,
    cfg: EvaluationConfig,
) -> dict[str, dict[str, Any]]:
    primary_window = _window_clock(clock, window)
    immediate_window = _window_clock(shift_execution_clock(clock, -1), window)
    extra_window = _window_clock(shift_execution_clock(clock, 1), window)
    return {
        "same_clock_side_flip": _stats(
            _build_trades(engine, primary_window, flip=True),
            window,
            cfg,
            include_cluster=False,
        ),
        "immediate_entry": _stats(
            _build_trades(engine, immediate_window),
            window,
            cfg,
            include_cluster=False,
        ),
        "extra_latency": _stats(
            _build_trades(engine, extra_window),
            window,
            cfg,
            include_cluster=False,
        ),
    }


def _compute_train_report(
    cfg: EvaluationConfig,
    freeze: dict[str, Any],
    *,
    created_at: str,
) -> dict[str, Any]:
    """Replay train from frozen pre-2023 prefixes without writing an artifact."""

    boundaries = freeze["outcome_boundaries"]
    market = _load_market_prefix(cfg, rows=int(boundaries["market"]["rows_pre2023"]))
    funding = _load_funding_prefix(cfg, rows=int(boundaries["funding"]["rows_pre2023"]))
    if cast(pd.Series, market["date"]).iloc[-1] >= pd.Timestamp(TRAIN_END):
        raise ValueError("train loader parsed a 2023 market outcome")
    if cast(pd.Series, funding["date"]).iloc[-1] >= pd.Timestamp(TRAIN_END):
        raise ValueError("train loader parsed a 2023 funding outcome")
    engine = ExecutionEngine(market, funding, _engine_config(cfg))
    clock = _load_clock()
    train_clock = _window_clock(clock, "train")
    trades = _build_trades(engine, train_clock)
    base = _stats(trades, "train", cfg, include_cluster=True)
    stress = _stats(
        trades,
        "train",
        cfg,
        include_cluster=False,
        cost_rate=cfg.stress_cost_rate,
    )
    gates = train_gates(base, stress)
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at": created_at,
        "protocol": {
            "opened_windows": ["train_2020_2022"],
            "selection_2023_opened": False,
            "forward_windows_opened": False,
            "market_value_rows_parsed": int(len(market)),
            "funding_value_rows_parsed": int(len(funding)),
            "full_calendar_cagr": True,
            "strict_mdd": (
                "global/pre-entry HWM plus entry cost, favorable-before-adverse "
                "held 5m OHLC, realized funding debit, virtual adverse-mark "
                "liquidation cost, and actual exit cost"
            ),
            "controls_cannot_replace_primary": True,
            "funding_boundary_policy": (
                "interior settlements symmetric; exact entry/exit timestamp "
                "debits included and credits excluded"
            ),
        },
        "config": asdict(cfg),
        "freeze_sha256": sha256_file(cfg.freeze_output),
        "primary": {
            "name": SELECTED_NAME,
            "clock_hash": execution_clock_hash(clock),
            "train": base,
            "train_stress_10bp_per_side": stress,
            "train_years": {
                window: _stats(
                    _build_trades(engine, _window_clock(clock, window)),
                    window,
                    cfg,
                    include_cluster=False,
                )
                for window in ("train_2020", "train_2021", "train_2022")
            },
            "gates": gates,
            "passes": bool(all(gates.values())),
        },
        "train_controls": _control_stats(engine, clock, "train", cfg),
        "decision": (
            "open_selection_2023" if all(gates.values()) else "reject_before_selection"
        ),
    }
    return report


def evaluate_train(cfg: EvaluationConfig) -> dict[str, Any]:
    freeze = verify_evaluator_freeze(cfg)
    if Path(cfg.train_output).exists():
        raise FileExistsError("PCP train result is write-once")
    if Path(cfg.selection_output).exists():
        raise ValueError("selection result exists before train evaluation")
    report = _compute_train_report(
        cfg,
        freeze,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    _write_json_once(cfg.train_output, report)
    return report


def _verify_passing_train_result(
    cfg: EvaluationConfig,
    freeze: dict[str, Any],
) -> dict[str, Any]:
    train = _read_json(cfg.train_output)
    _verify_result_hash(train, label="train")
    if train.get("schema_version") != 1:
        raise ValueError("train result schema changed")
    if train.get("config") != asdict(cfg):
        raise ValueError("train result config differs from evaluator freeze")
    if train.get("freeze_sha256") != sha256_file(cfg.freeze_output):
        raise ValueError("train result belongs to a different evaluator freeze")
    protocol = train.get("protocol", {})
    if protocol.get("opened_windows") != ["train_2020_2022"]:
        raise ValueError("train result opened an unexpected window")
    if protocol.get("selection_2023_opened") is not False:
        raise ValueError("train artifact already opened selection outcomes")
    boundaries = freeze["outcome_boundaries"]
    if protocol.get("market_value_rows_parsed") != int(
        boundaries["market"]["rows_pre2023"]
    ):
        raise ValueError("train market prefix row count changed")
    if protocol.get("funding_value_rows_parsed") != int(
        boundaries["funding"]["rows_pre2023"]
    ):
        raise ValueError("train funding prefix row count changed")
    primary = train.get("primary", {})
    if primary.get("name") != SELECTED_NAME:
        raise ValueError("train result belongs to a different policy")
    if primary.get("clock_hash") != freeze["frozen_clocks"]["primary"]["hash"]:
        raise ValueError("train result uses a different execution clock")
    gates = primary.get("gates", {})
    if primary.get("passes") is not True or not gates or not all(gates.values()):
        raise PermissionError("2023 selection remains sealed because train failed")
    if train.get("decision") != "open_selection_2023":
        raise PermissionError("train artifact did not authorize 2023 selection")
    created_at = train.get("created_at")
    if not isinstance(created_at, str) or not created_at:
        raise ValueError("train result lacks a creation timestamp")
    replay = _compute_train_report(cfg, freeze, created_at=created_at)
    replay["result_hash"] = _result_hash(replay)
    if replay != train:
        raise ValueError("train artifact does not exactly replay from frozen prefixes")
    return train


def evaluate_selection(cfg: EvaluationConfig) -> dict[str, Any]:
    freeze = verify_evaluator_freeze(cfg)
    if Path(cfg.selection_output).exists():
        raise FileExistsError("PCP selection result is write-once")
    train = _verify_passing_train_result(cfg, freeze)
    boundaries = freeze["outcome_boundaries"]
    market = _load_market_prefix(cfg, rows=int(boundaries["market"]["rows_pre2024"]))
    funding = _load_funding_prefix(cfg, rows=int(boundaries["funding"]["rows_pre2024"]))
    engine = ExecutionEngine(market, funding, _engine_config(cfg))
    clock = _load_clock()
    windowed = {
        window: _build_trades(engine, _window_clock(clock, window))
        for window in (
            "selection_2023",
            "selection_2023_h1",
            "selection_2023_h2",
        )
    }
    full = _stats(
        windowed["selection_2023"],
        "selection_2023",
        cfg,
        include_cluster=True,
    )
    h1 = _stats(
        windowed["selection_2023_h1"],
        "selection_2023_h1",
        cfg,
        include_cluster=False,
    )
    h2 = _stats(
        windowed["selection_2023_h2"],
        "selection_2023_h2",
        cfg,
        include_cluster=False,
    )
    stress = _stats(
        windowed["selection_2023"],
        "selection_2023",
        cfg,
        include_cluster=False,
        cost_rate=cfg.stress_cost_rate,
    )
    gates = selection_gates(full, h1, h2, stress)
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "opened_windows": ["train_2020_2022", "selection_2023"],
            "train_replayed_before_selection": True,
            "forward_windows_opened": False,
            "full_calendar_cagr": True,
            "strict_mdd": (
                "global/pre-entry HWM plus entry cost, favorable-before-adverse "
                "held 5m OHLC, realized funding debit, virtual adverse-mark "
                "liquidation cost, and actual exit cost"
            ),
            "controls_cannot_replace_primary": True,
            "funding_boundary_policy": (
                "interior settlements symmetric; exact entry/exit timestamp "
                "debits included and credits excluded"
            ),
        },
        "config": asdict(cfg),
        "freeze_sha256": sha256_file(cfg.freeze_output),
        "train_result_sha256": sha256_file(cfg.train_output),
        "train_result_hash": train["result_hash"],
        "primary": {
            "name": SELECTED_NAME,
            "selection_2023": full,
            "selection_2023_h1": h1,
            "selection_2023_h2": h2,
            "selection_2023_stress_10bp_per_side": stress,
            "gates": gates,
            "passes": bool(all(gates.values())),
        },
        "selection_controls": _control_stats(engine, clock, "selection_2023", cfg),
        "decision": (
            "build_frozen_forward_feature_sources"
            if all(gates.values())
            else "reject_before_forward"
        ),
    }
    _write_json_once(cfg.selection_output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-csv", default=EvaluationConfig.feature_csv)
    parser.add_argument("--market-csv", default=EvaluationConfig.market_csv)
    parser.add_argument("--funding-csv", default=EvaluationConfig.funding_csv)
    parser.add_argument("--freeze-output", default=EvaluationConfig.freeze_output)
    parser.add_argument("--train-output", default=EvaluationConfig.train_output)
    parser.add_argument("--selection-output", default=EvaluationConfig.selection_output)
    parser.add_argument(
        "--stage", choices=("freeze", "train", "selection"), required=True
    )
    args = vars(parser.parse_args())
    stage = str(args.pop("stage"))
    cfg = EvaluationConfig(**args)
    if stage == "freeze":
        result = freeze_evaluator(cfg)
        summary = {
            "stage": stage,
            "output": cfg.freeze_output,
            "outcomes_opened": result["protocol"]["outcomes_opened"],
        }
    elif stage == "train":
        result = evaluate_train(cfg)
        summary = {
            "stage": stage,
            "output": cfg.train_output,
            "decision": result["decision"],
            "passes": result["primary"]["passes"],
        }
    else:
        result = evaluate_selection(cfg)
        summary = {
            "stage": stage,
            "output": cfg.selection_output,
            "decision": result["decision"],
            "passes": result["primary"]["passes"],
        }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
