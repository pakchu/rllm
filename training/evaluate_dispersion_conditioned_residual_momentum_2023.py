"""Open only DCRM-1's frozen 2023 outcomes under a strict simulator.

The evaluator, tests, configuration, source identities, support clock, and
freeze artifact must be committed before this module is allowed to load OHLC
or funding outcomes. Calendar 2024 and later remain sealed on a 2023 failure.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.build_dispersion_conditioned_residual_momentum_support import (
    CLOCK_COLUMNS,
    SYMBOLS,
    assert_clock_contract,
)
from training.preregister_dispersion_conditioned_residual_momentum import canonical_hash


START = pd.Timestamp("2023-01-01 00:00:00")
MID = pd.Timestamp("2023-07-01 00:00:00")
END = pd.Timestamp("2024-01-01 00:00:00")
MARKET_ROWS_2023 = 365 * 24 * 12
FUNDING_ROWS_2023 = 365 * 3

EVALUATION_SOURCE = Path("training/evaluate_dispersion_conditioned_residual_momentum_2023.py")
TEST_PATH = Path("tests/test_evaluate_dispersion_conditioned_residual_momentum_2023.py")
FREEZE_SOURCE = Path("training/freeze_dispersion_conditioned_residual_momentum_2023_evaluator.py")
FREEZE_TEST_PATH = Path(
    "tests/test_freeze_dispersion_conditioned_residual_momentum_2023_evaluator.py"
)
EVALUATION_FREEZE = Path(
    "results/dispersion_conditioned_residual_momentum_2023_evaluator_freeze_2026-07-17.json"
)
PREREGISTRATION = Path(
    "results/dispersion_conditioned_residual_momentum_preregistration_2026-07-17.json"
)
SUPPORT_MANIFEST = Path(
    "results/dispersion_conditioned_residual_momentum_support_2026-07-17.json"
)
CLOCK_PATH = Path(
    "data/dispersion_conditioned_residual_momentum_support_clock_2023_2024.csv.gz"
)
SOURCE_MANIFEST = Path(
    "results/leave_one_out_residual_exhaustion_v1_source_manifest_2026-07-17.json"
)
EXECUTION_SOURCE_MANIFEST = Path("results/dcrm_2023_execution_source_manifest_2026-07-17.json")
EXECUTION_SOURCE_DIR = Path("data/dcrm_2023_execution")
DEFAULT_OUTPUT = Path(
    "results/dispersion_conditioned_residual_momentum_selection_2023_2026-07-17.json"
)
DEFAULT_DOCS = Path(
    "docs/dispersion-conditioned-residual-momentum-selection-2023-2026-07-17.md"
)

PREREGISTRATION_SHA256 = "bfbdea9e288c2553e33ebdef1a6b4ccd6bdaf7d3c46b5b7fbc2d9d42413ca145"
SUPPORT_MANIFEST_SHA256 = "7093fc93a05bc509c55a0480033fe6fd82262d6207d6992e6c14486cc90dfd37"
SUPPORT_MANIFEST_HASH = "7a755efeb7c3e0b19a37b83571abd49d710397abe4560448b4425ab7d06dad8c"
CLOCK_SHA256 = "d7cb7b5066692b8dccc6dbc2051d01c9522acc1e0769e63b7a6135bbffeae992"
SUPPORT_SOURCE_SHA256 = "b13beb1d879773d702e372e5c61ed85a0fd8f0ad4cf42ce0bca50e8bdf7ad55f"
PREREGISTRATION_SOURCE_SHA256 = "02bc2df6da0ed295dae9b2a6f00c8a5dd4012a1c2b0572c9d8766e36c8154a5d"
SOURCE_MANIFEST_SHA256 = "b3f5841f10b3e44ee47fb5d69c7acc6a2df0975596cdc0fd4019925f49b6eb66"
SOURCE_MANIFEST_HASH = "1c54fddc45fcc516d8ce42741904e018e8d00e646eff40be514273cf10eee7ed"
EXECUTION_SOURCE_MANIFEST_SHA256 = (
    "733bffb0ba4a58350855ce2b16a35bab759ffcf896bca6bbaaeac6f3c921a1f3"
)
EXECUTION_SOURCE_MANIFEST_HASH = (
    "95d875bc1a4c56027fb75b3cff1abe3f496a08680c472844277a2df23f3b2d15"
)


@dataclass(frozen=True)
class EvaluationConfig:
    base_cost_bp_per_notional_side: float = 6.0
    stress_cost_bp_per_notional_side: float = 10.0
    delay_control_minutes: int = 5
    cluster_signflip_samples: int = 20_000
    cluster_signflip_seed: int = 20_260_717


CONFIG = EvaluationConfig()


@dataclass(frozen=True)
class MarketBundle:
    dates: pd.DatetimeIndex
    market: dict[str, dict[str, np.ndarray]]
    funding: dict[str, pd.DataFrame]
    source_hashes: dict[str, dict[str, str]]


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _body_hash(payload: dict[str, Any]) -> str:
    return canonical_hash(
        {key: value for key, value in payload.items() if key not in {"manifest_hash", "created_at"}}
    )


def verify_support_and_clock() -> tuple[dict[str, Any], pd.DataFrame]:
    dependencies = (
        (PREREGISTRATION, PREREGISTRATION_SHA256),
        (SUPPORT_MANIFEST, SUPPORT_MANIFEST_SHA256),
        (CLOCK_PATH, CLOCK_SHA256),
        (Path("training/build_dispersion_conditioned_residual_momentum_support.py"), SUPPORT_SOURCE_SHA256),
        (Path("training/preregister_dispersion_conditioned_residual_momentum.py"), PREREGISTRATION_SOURCE_SHA256),
        (SOURCE_MANIFEST, SOURCE_MANIFEST_SHA256),
        (EXECUTION_SOURCE_MANIFEST, EXECUTION_SOURCE_MANIFEST_SHA256),
    )
    for path, expected in dependencies:
        if _sha256(path) != expected:
            raise RuntimeError(f"frozen DCRM-1 dependency changed: {path}")
    support = json.loads(SUPPORT_MANIFEST.read_text())
    if support.get("manifest_hash") != SUPPORT_MANIFEST_HASH or _body_hash(support) != SUPPORT_MANIFEST_HASH:
        raise RuntimeError("DCRM-1 support manifest identity changed")
    if support.get("post_entry_returns_or_pnl_calculated") is not False:
        raise RuntimeError("DCRM-1 support artifact already opened outcomes")
    if support.get("support", {}).get("passes_support") is not True:
        raise RuntimeError("DCRM-1 support gate did not pass")
    if support.get("clock_sha256") != CLOCK_SHA256:
        raise RuntimeError("DCRM-1 support-to-clock binding changed")
    clock = pd.read_csv(CLOCK_PATH)
    assert_clock_contract(clock)
    for column in ("decision_time", "last_feature_time", "entry_time", "exit_time"):
        clock[column] = pd.to_datetime(clock[column], errors="raise")
    if len(clock) != 92:
        raise RuntimeError("DCRM-1 frozen clock row count changed")
    return support, clock


def execution_clock(clock: pd.DataFrame) -> pd.DataFrame:
    out = clock.copy()
    assert_clock_contract(out)
    for column in ("decision_time", "last_feature_time", "entry_time", "exit_time"):
        out[column] = pd.to_datetime(out[column], errors="raise")
    if not (out["last_feature_time"] < out["decision_time"]).all():
        raise RuntimeError("DCRM-1 feature cutoff crossed decision")
    if not (out["decision_time"] < out["entry_time"]).all():
        raise RuntimeError("DCRM-1 decision crossed entry")
    gross = out["long_weight"] + out["short_weight_abs"]
    if not np.allclose(gross, out["gross_scale"], atol=1e-12):
        raise RuntimeError("DCRM-1 execution gross changed")
    beta_exposure = out["long_weight"] * out["long_beta"] - out[
        "short_weight_abs"
    ] * out["short_beta"]
    if not np.allclose(beta_exposure, 0.0, atol=1e-12):
        raise RuntimeError("DCRM-1 execution clock lost beta neutrality")
    return out.sort_values("entry_time").reset_index(drop=True)


def transform_clock(clock: pd.DataFrame, kind: str) -> pd.DataFrame:
    out = clock.copy()
    if kind == "direction_flip":
        out[["long_symbol", "short_symbol"]] = out[["short_symbol", "long_symbol"]].to_numpy()
        out[["long_weight", "short_weight_abs"]] = out[
            ["short_weight_abs", "long_weight"]
        ].to_numpy()
        out[["base_long_weight", "base_short_weight_abs"]] = out[
            ["base_short_weight_abs", "base_long_weight"]
        ].to_numpy()
        out[["long_beta", "short_beta"]] = out[["short_beta", "long_beta"]].to_numpy()
    elif kind == "delay_five_minutes":
        delta = pd.Timedelta(minutes=CONFIG.delay_control_minutes)
        out["entry_time"] = pd.to_datetime(out["entry_time"]) + delta
        out["exit_time"] = pd.to_datetime(out["exit_time"]) + delta
    elif kind == "full_gross":
        out["gross_scale"] = 1.0
        out["long_weight"] = out["base_long_weight"]
        out["short_weight_abs"] = out["base_short_weight_abs"]
    elif kind == "inverted_dispersion_scale":
        out["gross_scale"] = np.where(out["gross_scale"] == 1.0, 0.25, 1.0)
        out["long_weight"] = out["base_long_weight"] * out["gross_scale"]
        out["short_weight_abs"] = out["base_short_weight_abs"] * out["gross_scale"]
    else:
        raise ValueError(kind)
    return out


def verify_evaluation_freeze() -> dict[str, Any]:
    if not EVALUATION_FREEZE.exists():
        raise RuntimeError("DCRM-1 2023 evaluator freeze is missing")
    payload = json.loads(EVALUATION_FREEZE.read_text())
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if canonical_hash(core) != payload.get("manifest_hash"):
        raise RuntimeError("DCRM-1 evaluator freeze hash mismatch")
    checks = {
        "outcomes_opened": False,
        "evaluation_source": str(EVALUATION_SOURCE),
        "evaluation_source_sha256": _sha256(EVALUATION_SOURCE),
        "test_path": str(TEST_PATH),
        "test_sha256": _sha256(TEST_PATH),
        "freeze_source": str(FREEZE_SOURCE),
        "freeze_source_sha256": _sha256(FREEZE_SOURCE),
        "freeze_test_path": str(FREEZE_TEST_PATH),
        "freeze_test_sha256": _sha256(FREEZE_TEST_PATH),
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "support_manifest_sha256": SUPPORT_MANIFEST_SHA256,
        "support_manifest_hash": SUPPORT_MANIFEST_HASH,
        "clock_sha256": CLOCK_SHA256,
        "support_source_sha256": SUPPORT_SOURCE_SHA256,
        "preregistration_source_sha256": PREREGISTRATION_SOURCE_SHA256,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "source_manifest_hash": SOURCE_MANIFEST_HASH,
        "execution_source_manifest_sha256": EXECUTION_SOURCE_MANIFEST_SHA256,
        "execution_source_manifest_hash": EXECUTION_SOURCE_MANIFEST_HASH,
        "evaluation_config": asdict(CONFIG),
        "mutable_parameters": [],
        "labels_constructed_during_freeze": False,
        "market_rows_parsed_during_freeze": 0,
        "funding_rows_loaded_during_freeze": 0,
        "execution_simulation_run_during_freeze": False,
        "clock_rows": 92,
        "opened_windows": [],
        "sealed_windows": ["2023", "2024", "2025", "2026"],
        "protocol": "DCRM-1 strict 2023 evaluator pre-outcome freeze v1",
        "source_prefix_contract": {
            "market_rows_per_symbol": MARKET_ROWS_2023,
            "funding_rows_per_symbol": FUNDING_ROWS_2023,
            "maximum_timestamp_exclusive": str(END),
            "2024_rows_permitted": 0,
        },
        "decision_rule": (
            "open the singleton DCRM-1 2023 clock once under every preregistered gate; "
            "a failure retires DCRM-1 before loading any 2024 outcome"
        ),
    }
    for key, expected in checks.items():
        if payload.get(key) != expected:
            raise RuntimeError(f"DCRM-1 evaluator freeze changed: {key}")
    commit = str(payload.get("evaluation_source_commit", ""))
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise RuntimeError("DCRM-1 evaluator freeze commit is invalid")
    for path in (EVALUATION_SOURCE, TEST_PATH, FREEZE_SOURCE, FREEZE_TEST_PATH):
        committed = subprocess.check_output(["git", "show", f"{commit}:{path}"])
        if hashlib.sha256(committed).hexdigest() != _sha256(path):
            raise RuntimeError(f"DCRM-1 evaluator freeze commit does not bind {path}")
    return payload


def _source_records() -> dict[str, dict[str, Any]]:
    payload = json.loads(EXECUTION_SOURCE_MANIFEST.read_text())
    if (
        payload.get("manifest_hash") != EXECUTION_SOURCE_MANIFEST_HASH
        or _body_hash(payload) != EXECUTION_SOURCE_MANIFEST_HASH
    ):
        raise RuntimeError("DCRM-1 physical 2023 source manifest identity changed")
    if payload.get("2024_rows_parsed") != 0 or payload.get("2024_rows_written") != 0:
        raise RuntimeError("DCRM-1 physical source opened a future row")
    return {str(row["symbol"]): row for row in payload["records"]}


def load_bundle_2023(source_dir: str | Path = EXECUTION_SOURCE_DIR) -> MarketBundle:
    """Load only physically separate files ending before 2024."""

    verify_evaluation_freeze()
    records = _source_records()
    expected_grid = pd.date_range(START, END - pd.Timedelta(minutes=5), freq="5min")
    expected_funding = pd.date_range(START, END - pd.Timedelta(hours=8), freq="8h")
    market: dict[str, dict[str, np.ndarray]] = {}
    funding: dict[str, pd.DataFrame] = {}
    source_hashes: dict[str, dict[str, str]] = {}
    dates: pd.DatetimeIndex | None = None
    for symbol in sorted(SYMBOLS):
        market_path = Path(source_dir) / f"{symbol}_5m_2023.csv.gz"
        funding_path = Path(source_dir) / f"{symbol}_funding_2023.csv.gz"
        market_hash = _sha256(market_path)
        funding_hash = _sha256(funding_path)
        record = records[symbol]
        if market_hash != record["market_output_sha256"]:
            raise RuntimeError(f"{symbol} market source changed")
        if funding_hash != record["funding_output_sha256"]:
            raise RuntimeError(f"{symbol} funding source changed")
        frame = pd.read_csv(
            market_path,
            usecols=["date", "open", "high", "low", "close", "tic"],
            parse_dates=["date"],
        )
        current_dates = pd.DatetimeIndex(frame["date"])
        if not current_dates.equals(expected_grid):
            raise RuntimeError(f"{symbol} 2023 market prefix/grid changed")
        if not frame["tic"].astype(str).eq(symbol).all():
            raise RuntimeError(f"{symbol} market identity changed")
        if dates is None:
            dates = current_dates
        elif not dates.equals(current_dates):
            raise RuntimeError("DCRM-1 symbol grids differ")
        arrays = {
            column: pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
            for column in ("open", "high", "low", "close")
        }
        if not np.isfinite(np.column_stack(list(arrays.values()))).all():
            raise RuntimeError(f"{symbol} non-finite 2023 market value")
        market[symbol] = arrays
        fund = pd.read_csv(
            funding_path,
            usecols=["event_time", "funding_rate"],
            parse_dates=["event_time"],
        )
        fund["funding_rate"] = pd.to_numeric(fund["funding_rate"], errors="raise")
        funding_times = pd.DatetimeIndex(fund["event_time"])
        if (
            len(funding_times) != len(expected_funding)
            or funding_times.duplicated().any()
            or not funding_times.is_monotonic_increasing
            or (
                (funding_times - expected_funding)
                .to_series(index=expected_funding)
                .abs()
                > pd.Timedelta(seconds=1)
            ).any()
        ):
            raise RuntimeError(f"{symbol} missing or shifted 2023 funding settlement")
        if not np.isfinite(fund["funding_rate"]).all():
            raise RuntimeError(f"{symbol} non-finite 2023 funding rate")
        funding[symbol] = fund[["event_time", "funding_rate"]].copy()
        source_hashes[symbol] = {"market": market_hash, "funding": funding_hash}
    assert dates is not None
    return MarketBundle(dates=dates, market=market, funding=funding, source_hashes=source_hashes)


def _funding_events_by_bar(
    bundle: MarketBundle,
) -> dict[str, dict[int, list[tuple[pd.Timestamp, float, float]]]]:
    out: dict[str, dict[int, list[tuple[pd.Timestamp, float, float]]]] = {}
    dates_ns = bundle.dates.to_numpy(dtype="datetime64[ns]")
    completed_ns = (bundle.dates + pd.Timedelta(minutes=5)).to_numpy(dtype="datetime64[ns]")
    for symbol, frame in bundle.funding.items():
        mapped: dict[int, list[tuple[pd.Timestamp, float, float]]] = {}
        for row in frame.itertuples(index=False):
            event = pd.Timestamp(row.event_time)
            event64 = event.to_datetime64()
            bar_index = int(np.searchsorted(dates_ns, event64, side="right") - 1)
            mark_index = int(np.searchsorted(completed_ns, event64, side="right") - 1)
            if bar_index < 0 or mark_index < 0 or bar_index >= len(bundle.dates):
                continue
            mark = float(bundle.market[symbol]["close"][mark_index])
            mapped.setdefault(bar_index, []).append((event, float(row.funding_rate), mark))
        out[symbol] = mapped
    return out


def _event_indices(
    bundle: MarketBundle, entry: pd.Timestamp, exit_time: pd.Timestamp
) -> tuple[int, int]:
    entry_index = int(bundle.dates.searchsorted(entry))
    exit_index = int(bundle.dates.searchsorted(exit_time))
    if entry_index >= len(bundle.dates) or bundle.dates[entry_index] != entry:
        raise RuntimeError(f"missing exact DCRM-1 entry open: {entry}")
    if exit_index >= len(bundle.dates) or bundle.dates[exit_index] != exit_time:
        raise RuntimeError(f"missing exact DCRM-1 exit open: {exit_time}")
    if exit_index <= entry_index:
        raise RuntimeError("non-positive DCRM-1 hold")
    return entry_index, exit_index


def simulate(
    bundle: MarketBundle,
    clock: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cost_bp: float,
) -> dict[str, Any]:
    """Strict two-leg simulation allowing cash through frozen gross scaling."""

    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    if not bundle.dates[0] <= start_ts < end_ts <= bundle.dates[-1] + pd.Timedelta(minutes=5):
        raise ValueError("DCRM-1 simulation escaped loaded outcome prefix")
    if cost_bp < 0:
        raise ValueError("negative execution cost")
    rate = cost_bp / 10_000.0
    selected = clock.loc[
        (pd.to_datetime(clock["entry_time"]) >= start_ts)
        & (pd.to_datetime(clock["exit_time"]) < end_ts)
    ].sort_values("entry_time")
    funding_by_bar = _funding_events_by_bar(bundle)
    equity = 1.0
    peak = 1.0
    close_peak = 1.0
    strict_mdd = 0.0
    close_mdd = 0.0
    total_funding = 0.0
    total_cost = 0.0
    trade_rows: list[dict[str, Any]] = []
    previous_exit: pd.Timestamp | None = None
    for row in selected.itertuples(index=False):
        entry_time, exit_time = pd.Timestamp(row.entry_time), pd.Timestamp(row.exit_time)
        if previous_exit is not None and entry_time < previous_exit:
            raise RuntimeError("DCRM-1 execution clock overlaps")
        previous_exit = exit_time
        entry_index, exit_index = _event_indices(bundle, entry_time, exit_time)
        long_symbol, short_symbol = str(row.long_symbol), str(row.short_symbol)
        long_weight, short_weight = float(row.long_weight), float(row.short_weight_abs)
        gross = long_weight + short_weight
        if not 0.0 < gross <= 1.0 + 1e-12:
            raise RuntimeError("DCRM-1 gross escaped (0, 1]")
        if not math.isclose(gross, float(row.gross_scale), rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError("DCRM-1 gross no longer matches frozen scale")
        if min(long_weight, short_weight) <= 0.0:
            raise RuntimeError("DCRM-1 non-positive leg")
        start_equity = equity
        long_entry = float(bundle.market[long_symbol]["open"][entry_index])
        short_entry = float(bundle.market[short_symbol]["open"][entry_index])
        long_qty = long_weight * start_equity / long_entry
        short_qty = short_weight * start_equity / short_entry
        entry_cost = rate * (long_qty * long_entry + short_qty * short_entry)
        total_cost += entry_cost
        cumulative_funding = 0.0
        cumulative_funding_debits_for_peak = 0.0
        equity_after_entry = start_equity - entry_cost
        strict_mdd = max(strict_mdd, 1.0 - equity_after_entry / peak)

        def settle_funding(bar_index: int) -> None:
            nonlocal cumulative_funding, cumulative_funding_debits_for_peak, total_funding
            for symbol, signed_qty in ((long_symbol, long_qty), (short_symbol, -short_qty)):
                for event_time, funding_rate, mark in funding_by_bar[symbol].get(bar_index, []):
                    if entry_time < event_time <= exit_time:
                        cash = -signed_qty * mark * funding_rate
                        cumulative_funding += cash
                        cumulative_funding_debits_for_peak += min(cash, 0.0)
                        total_funding += cash

        for bar_index in range(entry_index, exit_index):
            settle_funding(bar_index)
            long_high = float(bundle.market[long_symbol]["high"][bar_index])
            long_low = float(bundle.market[long_symbol]["low"][bar_index])
            short_high = float(bundle.market[short_symbol]["high"][bar_index])
            short_low = float(bundle.market[short_symbol]["low"][bar_index])
            favorable_pnl = long_qty * (long_high - long_entry) + short_qty * (short_entry - short_low)
            adverse_pnl = long_qty * (long_low - long_entry) + short_qty * (short_entry - short_high)
            favorable_liquidation = rate * (long_qty * long_high + short_qty * short_low)
            adverse_liquidation = rate * (long_qty * long_low + short_qty * short_high)
            favorable_for_peak = (
                start_equity
                - entry_cost
                + cumulative_funding_debits_for_peak
                + favorable_pnl
                - favorable_liquidation
            )
            peak = max(peak, favorable_for_peak)
            adverse_equity = (
                start_equity - entry_cost + cumulative_funding + adverse_pnl - adverse_liquidation
            )
            strict_mdd = max(strict_mdd, 1.0 - adverse_equity / peak)
            long_close = float(bundle.market[long_symbol]["close"][bar_index])
            short_close = float(bundle.market[short_symbol]["close"][bar_index])
            close_pnl = long_qty * (long_close - long_entry) + short_qty * (
                short_entry - short_close
            )
            close_liquidation = rate * (long_qty * long_close + short_qty * short_close)
            close_equity = (
                start_equity - entry_cost + cumulative_funding + close_pnl - close_liquidation
            )
            close_peak = max(close_peak, close_equity)
            close_mdd = max(close_mdd, 1.0 - close_equity / close_peak)
        settle_funding(exit_index)
        long_exit = float(bundle.market[long_symbol]["open"][exit_index])
        short_exit = float(bundle.market[short_symbol]["open"][exit_index])
        exit_cost = rate * (long_qty * long_exit + short_qty * short_exit)
        total_cost += exit_cost
        pnl = long_qty * (long_exit - long_entry) + short_qty * (short_entry - short_exit)
        equity = start_equity - entry_cost + cumulative_funding + pnl - exit_cost
        strict_mdd = max(strict_mdd, 1.0 - equity / peak)
        close_peak = max(close_peak, equity)
        close_mdd = max(close_mdd, 1.0 - equity / close_peak)
        peak = max(peak, equity)
        trade_return = equity / start_equity - 1.0
        trade_rows.append(
            {
                "signal_time": str(row.decision_time),
                "entry_time": str(entry_time),
                "exit_time": str(exit_time),
                "long_symbol": long_symbol,
                "short_symbol": short_symbol,
                "gross_scale": float(row.gross_scale),
                "net_return": float(trade_return),
                "net_log_return": float(
                    math.log(max(equity, 1e-15) / max(start_equity, 1e-15))
                ),
                "funding_cash": float(cumulative_funding),
            }
        )
        if equity <= 0.0:
            strict_mdd = 1.0
            break
    years = (end_ts - start_ts).total_seconds() / (365.25 * 86_400.0)
    absolute = (equity - 1.0) * 100.0
    cagr = (equity ** (1.0 / years) - 1.0) * 100.0 if equity > 0 else -100.0
    strict_pct = min(max(strict_mdd * 100.0, 0.0), 100.0)
    trade_returns = np.asarray([row["net_return"] for row in trade_rows], dtype=float)
    return {
        "absolute_return_pct": float(absolute),
        "cagr_pct": float(cagr),
        "strict_mdd_pct": float(strict_pct),
        "close_mdd_pct": float(min(max(close_mdd * 100.0, 0.0), 100.0)),
        "cagr_to_strict_mdd": float(cagr / strict_pct) if strict_pct > 1e-12 else 0.0,
        "trades": len(trade_rows),
        "mean_net_bps": float(trade_returns.mean() * 10_000.0) if len(trade_returns) else 0.0,
        "win_rate": float(np.mean(trade_returns > 0.0)) if len(trade_returns) else 0.0,
        "funding_cash_pct_initial": float(total_funding * 100.0),
        "transaction_cost_pct_initial": float(total_cost * 100.0),
        "calendar_start": str(start_ts.date()),
        "calendar_end_exclusive": str(end_ts.date()),
        "trade_rows": trade_rows,
    }


def weekly_cluster_signflip(
    trade_rows: list[dict[str, Any]], *, seed: int, samples: int
) -> dict[str, Any]:
    if not trade_rows:
        return {
            "samples": samples,
            "weekly_clusters": 0,
            "observed_net_log_return": 0.0,
            "raw_p_value": 1.0,
        }
    frame = pd.DataFrame(trade_rows)
    frame["week"] = pd.to_datetime(frame["signal_time"]).dt.to_period("W-SUN").astype(str)
    clusters = frame.groupby("week")["net_log_return"].sum().to_numpy(dtype=float)
    observed = float(clusters.sum())
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=float)
    for left in range(0, samples, 500):
        right = min(left + 500, samples)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(right - left, len(clusters)))
        draws[left:right] = signs @ clusters
    return {
        "samples": samples,
        "weekly_clusters": int(len(clusters)),
        "observed_net_log_return": observed,
        "raw_p_value": float((1 + np.count_nonzero(draws >= observed - 1e-15)) / (samples + 1)),
        "random_positive_fraction": float(np.mean(draws > 0.0)),
        "random_q95_net_log_return": float(np.quantile(draws, 0.95)),
    }


def _slim(stats: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in stats.items() if key != "trade_rows"}


def selection_checks(
    primary: dict[str, dict[str, Any]],
    stress: dict[str, Any],
    delay: dict[str, Any],
    opposite: dict[str, Any],
    signflip: dict[str, Any],
) -> dict[str, bool]:
    annual = primary["2023"]
    return {
        "2023_absolute_return_positive": annual["absolute_return_pct"] > 0.0,
        "2023_cagr_to_strict_mdd_at_least_2": annual["cagr_to_strict_mdd"] >= 2.0,
        "2023_strict_mdd_at_most_15": annual["strict_mdd_pct"] <= 15.0,
        "2023_trades_at_least_35": annual["trades"] >= 35,
        "2023_h1_absolute_return_positive": primary["2023_h1"]["absolute_return_pct"] > 0.0,
        "2023_h2_absolute_return_positive": primary["2023_h2"]["absolute_return_pct"] > 0.0,
        "ten_bp_stress_absolute_return_positive": stress["absolute_return_pct"] > 0.0,
        "entry_and_exit_delay_plus_5m_absolute_return_positive": delay["absolute_return_pct"] > 0.0,
        "direction_flip_cagr_lower": opposite["cagr_pct"] < annual["cagr_pct"],
        "weekly_cluster_signflip_p_at_most_0_10": signflip["raw_p_value"] <= 0.10,
    }


def evaluate(bundle: MarketBundle, clock: pd.DataFrame) -> dict[str, Any]:
    windows = {
        "2023": (START, END),
        "2023_h1": (START, MID),
        "2023_h2": (MID, END),
    }
    primary_raw = {
        name: simulate(
            bundle,
            clock,
            start=start,
            end=end,
            cost_bp=CONFIG.base_cost_bp_per_notional_side,
        )
        for name, (start, end) in windows.items()
    }
    stress_raw = simulate(
        bundle,
        clock,
        start=START,
        end=END,
        cost_bp=CONFIG.stress_cost_bp_per_notional_side,
    )
    delay_raw = simulate(
        bundle,
        transform_clock(clock, "delay_five_minutes"),
        start=START,
        end=END,
        cost_bp=CONFIG.base_cost_bp_per_notional_side,
    )
    opposite_raw = simulate(
        bundle,
        transform_clock(clock, "direction_flip"),
        start=START,
        end=END,
        cost_bp=CONFIG.base_cost_bp_per_notional_side,
    )
    full_gross_raw = simulate(
        bundle,
        transform_clock(clock, "full_gross"),
        start=START,
        end=END,
        cost_bp=CONFIG.base_cost_bp_per_notional_side,
    )
    inverted_raw = simulate(
        bundle,
        transform_clock(clock, "inverted_dispersion_scale"),
        start=START,
        end=END,
        cost_bp=CONFIG.base_cost_bp_per_notional_side,
    )
    signflip = weekly_cluster_signflip(
        primary_raw["2023"]["trade_rows"],
        seed=CONFIG.cluster_signflip_seed,
        samples=CONFIG.cluster_signflip_samples,
    )
    primary = {name: _slim(stats) for name, stats in primary_raw.items()}
    stress, delay, opposite, full_gross, inverted = map(
        _slim, (stress_raw, delay_raw, opposite_raw, full_gross_raw, inverted_raw)
    )
    checks = selection_checks(primary, stress, delay, opposite, signflip)
    return {
        "primary": primary,
        "ten_bp_notional_side_cost_stress": stress,
        "entry_and_exit_delay_plus_5m": delay,
        "direction_flip": opposite,
        "dispersion_scaling_disabled_full_gross": full_gross,
        "dispersion_scaling_inverted": inverted,
        "weekly_cluster_signflip": signflip,
        "selection_gates": checks,
        "passes_2023_selection": all(checks.values()),
        "executed_trade_rows": primary_raw["2023"]["trade_rows"],
    }


def _markdown(result: dict[str, Any]) -> str:
    evaluation = result["evaluation"]
    lines = [
        "# DCRM-1 strict 2023 one-shot selection — 2026-07-17",
        "",
        f"- Decision: **{result['decision']}**",
        "- 2024, 2025, and 2026 outcomes remain sealed.",
        "",
        "| Window | Absolute return | Full-calendar CAGR | Strict MDD | CAGR/MDD | Trades |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, label in (("2023", "2023"), ("2023_h1", "2023 H1"), ("2023_h2", "2023 H2")):
        stats = evaluation["primary"][key]
        lines.append(
            f"| {label} | {stats['absolute_return_pct']:+.3f}% | {stats['cagr_pct']:+.3f}% | "
            f"{stats['strict_mdd_pct']:.3f}% | {stats['cagr_to_strict_mdd']:.3f} | {stats['trades']} |"
        )
    lines.extend(
        [
            "",
            "| Control | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for key, label in (
        ("ten_bp_notional_side_cost_stress", "10 bp/side"),
        ("entry_and_exit_delay_plus_5m", "Entry/exit +5m"),
        ("direction_flip", "Direction flip"),
        ("dispersion_scaling_disabled_full_gross", "Full gross"),
        ("dispersion_scaling_inverted", "Inverted dispersion scale"),
    ):
        stats = evaluation[key]
        lines.append(
            f"| {label} | {stats['absolute_return_pct']:+.3f}% | {stats['cagr_pct']:+.3f}% | "
            f"{stats['strict_mdd_pct']:.3f}% | {stats['cagr_to_strict_mdd']:.3f} | {stats['trades']} |"
        )
    failed = [name for name, passed in evaluation["selection_gates"].items() if not passed]
    lines.extend(
        [
            "",
            f"- Weekly-cluster sign-flip p: `{evaluation['weekly_cluster_signflip']['raw_p_value']:.6f}`",
            f"- Failed gates: `{failed}`",
            "- CAGR spans the complete declared calendar, including warm-up and reduced-gross weeks.",
            "- Strict MDD uses the global/pre-entry HWM, favorable-before-adverse two-leg OHLC, exact held funding, and entry/hypothetical-liquidation/exit costs.",
            "- Funding credits cannot create an intratrade peak. No sign, lookback, hold, scale, pair, or beta repair is permitted.",
            "",
        ]
    )
    return "\n".join(lines)


def _clean_repository_head() -> str:
    status = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    if status:
        raise RuntimeError("repository must be clean before DCRM-1 2023 outcomes open")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def run(
    output: str | Path = DEFAULT_OUTPUT,
    docs_output: str | Path = DEFAULT_DOCS,
) -> dict[str, Any]:
    if Path(output).exists():
        raise RuntimeError("DCRM-1 2023 outcome result already exists")
    support, frozen_clock = verify_support_and_clock()
    freeze = verify_evaluation_freeze()
    outcome_opening_head = _clean_repository_head()
    bundle = load_bundle_2023()
    clock = execution_clock(frozen_clock)
    evaluated = evaluate(bundle, clock)
    passed = bool(evaluated["passes_2023_selection"])
    result: dict[str, Any] = {
        "protocol_version": "dcrm_v1_2023_selection_2026-07-17",
        "outcomes_opened": True,
        "opened_window": ["2023-01-01", "2024-01-01"],
        "2024_test_opened": False,
        "2025_eval_opened": False,
        "2026_holdout_opened": False,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "support_manifest_hash": SUPPORT_MANIFEST_HASH,
        "clock_sha256": CLOCK_SHA256,
        "source_manifest_hash": SOURCE_MANIFEST_HASH,
        "execution_source_manifest_hash": EXECUTION_SOURCE_MANIFEST_HASH,
        "market_rows_loaded_per_symbol": MARKET_ROWS_2023,
        "funding_rows_loaded_per_symbol": FUNDING_ROWS_2023,
        "maximum_loaded_timestamp_exclusive": str(END),
        "evaluation_config": asdict(CONFIG),
        "support_snapshot": support["support"],
        "evaluation_freeze_hash": freeze["manifest_hash"],
        "outcome_opening_head": outcome_opening_head,
        "evaluation": evaluated,
        "decision": "2023_pass_open_2024_next" if passed else "rejected_before_2024",
        "anti_repair": (
            "A rejection seals 2024+ for DCRM-1; no sign, lookback, hold, scale, pair, or beta repair."
        ),
    }
    result["manifest_hash"] = canonical_hash(result)
    result["created_at"] = datetime.now(timezone.utc).isoformat()
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    docs_path = Path(docs_output)
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text(_markdown(result))
    return result


def main() -> None:
    result = run()
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "2024_test_opened": result["2024_test_opened"],
                "primary": result["evaluation"]["primary"],
                "controls": {
                    key: result["evaluation"][key]
                    for key in (
                        "ten_bp_notional_side_cost_stress",
                        "entry_and_exit_delay_plus_5m",
                        "direction_flip",
                        "dispersion_scaling_disabled_full_gross",
                        "dispersion_scaling_inverted",
                    )
                },
                "signflip": result["evaluation"]["weekly_cluster_signflip"],
                "failed_gates": [
                    key
                    for key, passed in result["evaluation"]["selection_gates"].items()
                    if not passed
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
