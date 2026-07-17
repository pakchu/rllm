"""Open only CRRC-72's frozen calendar-2023 outcomes under a strict ledger.

The evaluator, tests, configuration, physical source identities, primary and
control clocks, and evaluator-freeze artifact must all be committed before
this module may load OHLC or funding.  A failed 2023 gate keeps 2024+ sealed.
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

from training import export_crrc_2023_execution_sources as source_export
from training import qualify_cross_venue_radial_refill_compression as qualify
from training.preregister_cross_venue_radial_refill_compression import canonical_hash


START = pd.Timestamp("2023-01-01 00:00:00")
END = pd.Timestamp("2024-01-01 00:00:00")
WINDOWS: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {
    "2023": (START, END),
    "q1": (START, pd.Timestamp("2023-04-01")),
    "q2": (pd.Timestamp("2023-04-01"), pd.Timestamp("2023-07-01")),
    "q3": (pd.Timestamp("2023-07-01"), pd.Timestamp("2023-10-01")),
    "q4": (pd.Timestamp("2023-10-01"), END),
}

EVALUATION_SOURCE = Path("training/evaluate_crrc_2023.py")
TEST_PATH = Path("tests/test_evaluate_crrc_2023.py")
FREEZE_SOURCE = Path("training/freeze_crrc_2023_evaluator.py")
FREEZE_TEST_PATH = Path("tests/test_freeze_crrc_2023_evaluator.py")
EVALUATION_FREEZE = Path("results/crrc_2023_evaluator_freeze_2026-07-17.json")
PREREGISTRATION = Path(
    "results/cross_venue_radial_refill_compression_preregistration_2026-07-17.json"
)
SUPPORT = Path(
    "results/cross_venue_radial_refill_compression_support_2026-07-17.json"
)
PRIMARY_CLOCK = Path(
    "results/cross_venue_radial_refill_compression_event_clock_2026-07-17.json"
)
CONTROL_CLOCKS = Path(
    "results/cross_venue_radial_refill_compression_control_clocks_2026-07-17.json"
)
EXECUTION_SOURCE_MANIFEST = Path(
    "results/crrc_2023_execution_source_manifest_2026-07-17.json"
)
EXECUTION_SOURCE_DIR = Path("data/crrc_2023_execution")
DEFAULT_OUTPUT = Path("results/cross_venue_radial_refill_compression_selection_2023_2026-07-17.json")
DEFAULT_DOCS = Path("docs/cross-venue-radial-refill-compression-selection-2023-2026-07-17.md")

PREREGISTRATION_SHA256 = "c5f7904b2c56f4c8e1a80514357b47954c4441377924bcb8c918044cd3b18f49"
SUPPORT_SHA256 = "b5fc746bbc2e82e1ceaf57e61efa709ad084e1db74d4769da21c9ee345996e95"
PRIMARY_CLOCK_SHA256 = "09d2ca954c5c4d06b981575c6b0f0e4dc6b49d8a693da418f3f26e5cc454c835"
PRIMARY_EVENT_CLOCK_HASH = "81e09e3d1d5592f12ce1994077efa279ebf1de4c29a6f5a144060d16ee6b2e9f"
CONTROL_CLOCKS_SHA256 = "98330c3be52a959484a4353b775a0e42aca12043418e7482613ac0b9f4c657ad"
EXECUTION_SOURCE_MANIFEST_SHA256 = (
    "2824dc802dcd045a24b2f59c5ac56943498481eb42c1ada00a7fd3486a4f0f4a"
)
EXECUTION_SOURCE_MANIFEST_HASH = (
    "8ac689f0ca7024b3b3a748981647a14947fa2b4bab75768b88f5e1b1c730b3bc"
)
PREREGISTRATION_SOURCE_SHA256 = (
    "df83e1eb587aebaa4fe11c31ca03d000bd716b085673ea2fb282a92d3c49e276"
)
QUALIFIER_SOURCE_SHA256 = (
    "96372733a597ca486b52292480ceacde631056054b2d914aa9180024218fa0e7"
)
CONTROL_FREEZER_SOURCE_SHA256 = (
    "38886c63b3aca590f2177bb3afcb2a6613c8bef28d08c2275ae1a61e84b7f644"
)
SOURCE_EXPORTER_SHA256 = (
    "feef4d10af07771a5b79bc2c3d6e2a34d16f278fd952d18064aa186532eee32b"
)


@dataclass(frozen=True)
class EvaluationConfig:
    leverage: float = 0.5
    base_cost_bp_per_notional_side: float = 6.0
    stress_cost_bp_per_notional_side: float = 10.0
    delay_control_minutes: int = 5
    monthly_signflip_samples: int = 20_000
    monthly_signflip_seed: int = 20_260_717


CONFIG = EvaluationConfig()


@dataclass(frozen=True)
class MarketBundle:
    dates: pd.DatetimeIndex
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    funding: pd.DataFrame
    source_hashes: dict[str, str]


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _body_hash(payload: dict[str, Any]) -> str:
    return canonical_hash(
        {
            key: value
            for key, value in payload.items()
            if key not in {"manifest_hash", "created_at"}
        }
    )


def _clock_frame(events: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(events)
    required = {
        "quarter",
        "signal_position",
        "entry_position",
        "exit_position",
        "signal_date",
        "entry_date",
        "exit_date",
        "side",
        "hold_bars",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"CRRC event clock misses columns: {sorted(missing)}")
    for column in ("signal_date", "entry_date", "exit_date"):
        frame[column] = pd.to_datetime(frame[column], errors="raise")
    for column in (
        "signal_position",
        "entry_position",
        "exit_position",
        "side",
        "hold_bars",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
    return frame.sort_values("entry_position").reset_index(drop=True)


def validate_clock(frame: pd.DataFrame, *, primary: bool) -> pd.DataFrame:
    if frame.empty:
        raise RuntimeError("CRRC event clock is empty")
    if not frame["side"].isin([-1, 1]).all():
        raise RuntimeError("CRRC event clock contains a non-directional side")
    if not frame["hold_bars"].eq(72).all():
        raise RuntimeError("CRRC hold changed")
    if not frame["exit_position"].eq(frame["entry_position"] + 72).all():
        raise RuntimeError("CRRC exit clock changed")
    if primary and not frame["entry_position"].eq(frame["signal_position"] + 2).all():
        raise RuntimeError("CRRC primary entry delay changed")
    if not (frame["signal_date"] < frame["entry_date"]).all():
        raise RuntimeError("CRRC signal crossed entry")
    if not (frame["entry_date"] < frame["exit_date"]).all():
        raise RuntimeError("CRRC entry crossed exit")
    if not frame["entry_date"].dt.quarter.eq(frame["exit_date"].dt.quarter).all():
        raise RuntimeError("CRRC clock crossed a quarter boundary")
    entries = frame["entry_position"].to_numpy(int)
    exits = frame["exit_position"].to_numpy(int)
    if len(frame) > 1 and np.any(entries[1:] < exits[:-1]):
        raise RuntimeError("CRRC event clock overlaps")
    return frame


def verify_preoutcome_artifacts() -> tuple[dict[str, Any], pd.DataFrame, dict[str, pd.DataFrame]]:
    dependencies = (
        (PREREGISTRATION, PREREGISTRATION_SHA256),
        (SUPPORT, SUPPORT_SHA256),
        (PRIMARY_CLOCK, PRIMARY_CLOCK_SHA256),
        (CONTROL_CLOCKS, CONTROL_CLOCKS_SHA256),
        (EXECUTION_SOURCE_MANIFEST, EXECUTION_SOURCE_MANIFEST_SHA256),
        (
            Path("training/preregister_cross_venue_radial_refill_compression.py"),
            PREREGISTRATION_SOURCE_SHA256,
        ),
        (
            Path("training/qualify_cross_venue_radial_refill_compression.py"),
            QUALIFIER_SOURCE_SHA256,
        ),
        (Path("training/freeze_crrc_control_clocks.py"), CONTROL_FREEZER_SOURCE_SHA256),
        (Path("training/export_crrc_2023_execution_sources.py"), SOURCE_EXPORTER_SHA256),
    )
    for path, expected in dependencies:
        if sha256(path) != expected:
            raise RuntimeError(f"frozen CRRC dependency changed: {path}")

    preregistration = json.loads(PREREGISTRATION.read_text())
    if preregistration.get("protocol", {}).get("evidence_boundary", {}).get(
        "post_entry_price_return_or_equity_opened"
    ) is not False:
        raise RuntimeError("CRRC preregistration opened an outcome")
    support = json.loads(SUPPORT.read_text())
    if support.get("all_support_gates_pass") is not True:
        raise RuntimeError("CRRC support gate did not pass")
    if support.get("protocol", {}).get("outcomes_opened_for_crrc72") is not False:
        raise RuntimeError("CRRC support artifact opened an outcome")
    if support.get("event_clock_sha256") != PRIMARY_EVENT_CLOCK_HASH:
        raise RuntimeError("CRRC support event hash changed")

    primary_payload = json.loads(PRIMARY_CLOCK.read_text())
    if primary_payload.get("outcomes_opened") is not False:
        raise RuntimeError("CRRC primary clock opened an outcome")
    if primary_payload.get("event_clock_sha256") != PRIMARY_EVENT_CLOCK_HASH:
        raise RuntimeError("CRRC primary event hash changed")
    if primary_payload.get("event_count") != 156:
        raise RuntimeError("CRRC primary event count changed")
    primary = validate_clock(_clock_frame(primary_payload["events"]), primary=True)
    if qualify.event_clock_hash(primary) != PRIMARY_EVENT_CLOCK_HASH:
        raise RuntimeError("CRRC primary records no longer hash to frozen clock")

    control_payload = json.loads(CONTROL_CLOCKS.read_text())
    if control_payload.get("outcomes_opened") is not False:
        raise RuntimeError("CRRC control clock opened an outcome")
    if control_payload.get("controls_are_diagnostics_not_repair_candidates") is not True:
        raise RuntimeError("CRRC control anti-repair contract changed")
    controls: dict[str, pd.DataFrame] = {}
    for name, item in control_payload["controls"].items():
        frame = validate_clock(_clock_frame(item["events"]), primary=True)
        if len(frame) != item["event_count"]:
            raise RuntimeError(f"CRRC control event count changed: {name}")
        if qualify.event_clock_hash(frame) != item["event_clock_sha256"]:
            raise RuntimeError(f"CRRC control clock changed: {name}")
        controls[name] = frame

    source_manifest = json.loads(EXECUTION_SOURCE_MANIFEST.read_text())
    if (
        source_manifest.get("manifest_hash") != EXECUTION_SOURCE_MANIFEST_HASH
        or _body_hash(source_manifest) != EXECUTION_SOURCE_MANIFEST_HASH
    ):
        raise RuntimeError("CRRC physical execution source manifest changed")
    if source_manifest.get("outcomes_calculated") is not False:
        raise RuntimeError("CRRC source freeze calculated an outcome")
    if source_manifest.get("2024_rows_parsed") != 0:
        raise RuntimeError("CRRC source freeze parsed 2024")
    return support, primary, controls


def verify_evaluation_freeze() -> dict[str, Any]:
    if not EVALUATION_FREEZE.is_file():
        raise RuntimeError("CRRC 2023 evaluator freeze is missing")
    payload = json.loads(EVALUATION_FREEZE.read_text())
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if canonical_hash(core) != payload.get("manifest_hash"):
        raise RuntimeError("CRRC evaluator freeze hash mismatch")
    checks = {
        "protocol": "CRRC-72 strict 2023 evaluator pre-outcome freeze v1",
        "outcomes_opened": False,
        "evaluation_source": str(EVALUATION_SOURCE),
        "evaluation_source_sha256": sha256(EVALUATION_SOURCE),
        "test_path": str(TEST_PATH),
        "test_sha256": sha256(TEST_PATH),
        "freeze_source": str(FREEZE_SOURCE),
        "freeze_source_sha256": sha256(FREEZE_SOURCE),
        "freeze_test_path": str(FREEZE_TEST_PATH),
        "freeze_test_sha256": sha256(FREEZE_TEST_PATH),
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "support_sha256": SUPPORT_SHA256,
        "primary_clock_sha256": PRIMARY_CLOCK_SHA256,
        "primary_event_clock_hash": PRIMARY_EVENT_CLOCK_HASH,
        "control_clocks_sha256": CONTROL_CLOCKS_SHA256,
        "execution_source_manifest_sha256": EXECUTION_SOURCE_MANIFEST_SHA256,
        "execution_source_manifest_hash": EXECUTION_SOURCE_MANIFEST_HASH,
        "evaluation_config": asdict(CONFIG),
        "mutable_parameters": [],
        "market_rows_parsed_during_freeze": 0,
        "funding_rows_loaded_during_freeze": 0,
        "execution_simulation_run_during_freeze": False,
        "opened_windows": [],
        "sealed_windows": ["2023", "2024", "2025", "2026"],
        "primary_clock_rows": 156,
        "source_prefix_contract": {
            "market_rows": source_export.MARKET_ROWS,
            "funding_rows": source_export.FUNDING_ROWS,
            "maximum_timestamp_exclusive": str(END),
            "2024_rows_permitted": 0,
        },
        "decision_rule": (
            "open the singleton CRRC-72 2023 clock once under all frozen gates; "
            "a failure retires CRRC-72 before loading any 2024 outcome"
        ),
    }
    for key, expected in checks.items():
        if payload.get(key) != expected:
            raise RuntimeError(f"CRRC evaluator freeze changed: {key}")
    commit = str(payload.get("evaluation_source_commit", ""))
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise RuntimeError("CRRC evaluator freeze commit is invalid")
    for path in (EVALUATION_SOURCE, TEST_PATH, FREEZE_SOURCE, FREEZE_TEST_PATH):
        committed = subprocess.check_output(["git", "show", f"{commit}:{path}"])
        if hashlib.sha256(committed).hexdigest() != sha256(path):
            raise RuntimeError(f"CRRC evaluator freeze commit does not bind {path}")
    return payload


def load_bundle_2023(source_dir: str | Path = EXECUTION_SOURCE_DIR) -> MarketBundle:
    """Load only dedicated files whose maximum timestamp is before 2024."""

    verify_evaluation_freeze()
    manifest = json.loads(EXECUTION_SOURCE_MANIFEST.read_text())
    market_path = Path(source_dir) / "BTCUSDT_5m_2023.csv.gz"
    funding_path = Path(source_dir) / "BTCUSDT_funding_2023.csv.gz"
    market_hash = sha256(market_path)
    funding_hash = sha256(funding_path)
    if market_hash != manifest["market_output_sha256"]:
        raise RuntimeError("CRRC physical market source changed")
    if funding_hash != manifest["funding_output_sha256"]:
        raise RuntimeError("CRRC physical funding source changed")
    market = source_export.validate_market_2023(
        pd.read_csv(market_path, usecols=source_export.MARKET_COLUMNS)
    )
    funding = source_export.validate_funding_2023(
        pd.read_csv(funding_path, usecols=source_export.FUNDING_OUTPUT_COLUMNS)
    )
    arrays = {
        column: market[column].to_numpy(float)
        for column in ("open", "high", "low", "close")
    }
    return MarketBundle(
        dates=pd.DatetimeIndex(market["date"]),
        open=arrays["open"],
        high=arrays["high"],
        low=arrays["low"],
        close=arrays["close"],
        funding=funding,
        source_hashes={"market": market_hash, "funding": funding_hash},
    )


def transform_clock(clock: pd.DataFrame, kind: str) -> pd.DataFrame:
    output = clock.copy()
    if kind == "direction_flip":
        output["side"] *= -1
    elif kind == "delay_five_minutes":
        output["entry_position"] += 1
        output["exit_position"] += 1
        output["entry_date"] += pd.Timedelta(minutes=CONFIG.delay_control_minutes)
        output["exit_date"] += pd.Timedelta(minutes=CONFIG.delay_control_minutes)
    elif kind == "long_only":
        output = output.loc[output["side"].gt(0)].copy()
    elif kind == "short_only":
        output = output.loc[output["side"].lt(0)].copy()
    else:
        raise ValueError(kind)
    return output.sort_values("entry_position").reset_index(drop=True)


def _event_indices(
    bundle: MarketBundle, entry_time: pd.Timestamp, exit_time: pd.Timestamp
) -> tuple[int, int]:
    entry = int(bundle.dates.searchsorted(entry_time))
    exit_position = int(bundle.dates.searchsorted(exit_time))
    if entry >= len(bundle.dates) or bundle.dates[entry] != entry_time:
        raise RuntimeError(f"missing exact CRRC entry open: {entry_time}")
    if exit_position >= len(bundle.dates) or bundle.dates[exit_position] != exit_time:
        raise RuntimeError(f"missing exact CRRC exit open: {exit_time}")
    if exit_position <= entry:
        raise RuntimeError("CRRC hold is non-positive")
    return entry, exit_position


def _funding_by_bar(
    bundle: MarketBundle,
) -> dict[int, list[tuple[pd.Timestamp, float, float]]]:
    dates = bundle.dates.to_numpy(dtype="datetime64[ns]")
    completed = (bundle.dates + pd.Timedelta(minutes=5)).to_numpy(dtype="datetime64[ns]")
    output: dict[int, list[tuple[pd.Timestamp, float, float]]] = {}
    for row in bundle.funding.itertuples(index=False):
        event = pd.Timestamp(row.event_time)
        event64 = event.to_datetime64()
        bar = int(np.searchsorted(dates, event64, side="right") - 1)
        mark_bar = int(np.searchsorted(completed, event64, side="right") - 1)
        if bar < 0 or mark_bar < 0 or bar >= len(bundle.dates):
            continue
        output.setdefault(bar, []).append(
            (event, float(row.funding_rate), float(bundle.close[mark_bar]))
        )
    return output


def simulate(
    bundle: MarketBundle,
    clock: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cost_bp: float,
) -> dict[str, Any]:
    """Strict one-leg simulation with global HWM and held OHLC path risk."""

    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    if not bundle.dates[0] <= start_ts < end_ts <= bundle.dates[-1] + pd.Timedelta(minutes=5):
        raise ValueError("CRRC simulation escaped the loaded 2023 prefix")
    if cost_bp < 0.0:
        raise ValueError("execution cost cannot be negative")
    rate = float(cost_bp) / 10_000.0
    leverage = float(CONFIG.leverage)
    selected = clock.loc[
        (pd.to_datetime(clock["entry_date"]) >= start_ts)
        & (pd.to_datetime(clock["exit_date"]) < end_ts)
    ].sort_values("entry_date")
    funding = _funding_by_bar(bundle)
    equity = peak = close_peak = 1.0
    strict_mdd = close_mdd = 0.0
    total_funding = total_cost = 0.0
    trade_rows: list[dict[str, Any]] = []
    previous_exit: pd.Timestamp | None = None
    for row in selected.itertuples(index=False):
        entry_time, exit_time = pd.Timestamp(row.entry_date), pd.Timestamp(row.exit_date)
        if previous_exit is not None and entry_time < previous_exit:
            raise RuntimeError("CRRC simulation clock overlaps")
        previous_exit = exit_time
        entry_index, exit_index = _event_indices(bundle, entry_time, exit_time)
        side = int(row.side)
        if side not in (-1, 1):
            raise RuntimeError("CRRC simulation side is invalid")
        start_equity = equity
        entry_price = float(bundle.open[entry_index])
        signed_quantity = side * leverage * start_equity / entry_price
        entry_cost = rate * abs(signed_quantity) * entry_price
        total_cost += entry_cost
        cumulative_funding = 0.0
        cumulative_funding_debits_for_peak = 0.0
        equity_after_entry = start_equity - entry_cost
        strict_mdd = max(strict_mdd, 1.0 - equity_after_entry / peak)

        def settle_funding(bar_index: int) -> None:
            nonlocal cumulative_funding, cumulative_funding_debits_for_peak, total_funding
            for event_time, funding_rate, mark_price in funding.get(bar_index, []):
                if entry_time < event_time <= exit_time:
                    cash = -signed_quantity * mark_price * funding_rate
                    cumulative_funding += cash
                    cumulative_funding_debits_for_peak += min(cash, 0.0)
                    total_funding += cash

        for bar_index in range(entry_index, exit_index):
            settle_funding(bar_index)
            favorable_price = (
                float(bundle.high[bar_index]) if side > 0 else float(bundle.low[bar_index])
            )
            adverse_price = (
                float(bundle.low[bar_index]) if side > 0 else float(bundle.high[bar_index])
            )
            favorable_pnl = signed_quantity * (favorable_price - entry_price)
            adverse_pnl = signed_quantity * (adverse_price - entry_price)
            favorable_liquidation = rate * abs(signed_quantity) * favorable_price
            adverse_liquidation = rate * abs(signed_quantity) * adverse_price
            favorable_for_peak = (
                start_equity
                - entry_cost
                + cumulative_funding_debits_for_peak
                + favorable_pnl
                - favorable_liquidation
            )
            peak = max(peak, favorable_for_peak)
            adverse_equity = (
                start_equity
                - entry_cost
                + cumulative_funding
                + adverse_pnl
                - adverse_liquidation
            )
            strict_mdd = max(strict_mdd, 1.0 - adverse_equity / peak)
            close_price = float(bundle.close[bar_index])
            close_equity = (
                start_equity
                - entry_cost
                + cumulative_funding
                + signed_quantity * (close_price - entry_price)
                - rate * abs(signed_quantity) * close_price
            )
            close_peak = max(close_peak, close_equity)
            close_mdd = max(close_mdd, 1.0 - close_equity / close_peak)

        settle_funding(exit_index)
        exit_price = float(bundle.open[exit_index])
        exit_cost = rate * abs(signed_quantity) * exit_price
        total_cost += exit_cost
        realized_pnl = signed_quantity * (exit_price - entry_price)
        equity = start_equity - entry_cost + cumulative_funding + realized_pnl - exit_cost
        strict_mdd = max(strict_mdd, 1.0 - equity / peak)
        close_peak = max(close_peak, equity)
        close_mdd = max(close_mdd, 1.0 - equity / close_peak)
        peak = max(peak, equity)
        trade_return = equity / start_equity - 1.0
        trade_rows.append(
            {
                "signal_time": str(pd.Timestamp(row.signal_date)),
                "entry_time": str(entry_time),
                "exit_time": str(exit_time),
                "side": side,
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

    years = calendar_year_fraction(start_ts, end_ts)
    absolute = (equity - 1.0) * 100.0
    cagr = (equity ** (1.0 / years) - 1.0) * 100.0 if equity > 0.0 else -100.0
    strict_pct = min(max(strict_mdd * 100.0, 0.0), 100.0)
    trade_returns = np.asarray([row["net_return"] for row in trade_rows], dtype=float)
    return {
        "absolute_return_pct": float(absolute),
        "cagr_pct": float(cagr),
        "strict_mdd_pct": float(strict_pct),
        "close_mdd_pct": float(min(max(close_mdd * 100.0, 0.0), 100.0)),
        "cagr_to_strict_mdd": float(cagr / strict_pct) if strict_pct > 1e-12 else 0.0,
        "trades": int(len(trade_rows)),
        "longs": int(sum(row["side"] > 0 for row in trade_rows)),
        "shorts": int(sum(row["side"] < 0 for row in trade_rows)),
        "mean_net_bps": float(trade_returns.mean() * 10_000.0) if len(trade_returns) else 0.0,
        "win_rate": float(np.mean(trade_returns > 0.0)) if len(trade_returns) else 0.0,
        "funding_cash_pct_initial": float(total_funding * 100.0),
        "transaction_cost_pct_initial": float(total_cost * 100.0),
        "calendar_start": str(start_ts.date()),
        "calendar_end_exclusive": str(end_ts.date()),
        "trade_rows": trade_rows,
    }


def calendar_year_fraction(start: pd.Timestamp, end: pd.Timestamp) -> float:
    """Express a 2023 subwindow in exact calendar-2023 years."""

    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    if not START <= start_ts < end_ts <= END:
        raise ValueError("CRRC calendar fraction must remain inside 2023")
    calendar_seconds = (END - START).total_seconds()
    return float((end_ts - start_ts).total_seconds() / calendar_seconds)


def monthly_cluster_signflip(
    trade_rows: list[dict[str, Any]], *, seed: int, samples: int
) -> dict[str, Any]:
    if not trade_rows:
        return {
            "samples": samples,
            "monthly_clusters": 0,
            "observed_net_log_return": 0.0,
            "raw_p_value": 1.0,
        }
    frame = pd.DataFrame(trade_rows)
    frame["month"] = pd.to_datetime(frame["signal_time"]).dt.to_period("M").astype(str)
    clusters = frame.groupby("month")["net_log_return"].sum().to_numpy(float)
    observed = float(clusters.sum())
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=float)
    for left in range(0, samples, 500):
        right = min(left + 500, samples)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(right - left, len(clusters)))
        draws[left:right] = signs @ clusters
    return {
        "samples": int(samples),
        "monthly_clusters": int(len(clusters)),
        "observed_net_log_return": observed,
        "raw_p_value": float((1 + np.count_nonzero(draws >= observed - 1e-15)) / (samples + 1)),
        "random_positive_fraction": float(np.mean(draws > 0.0)),
        "random_q95_net_log_return": float(np.quantile(draws, 0.95)),
    }


def _slim(stats: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in stats.items() if key != "trade_rows"}


def selection_checks(
    primary: dict[str, dict[str, Any]],
    long_only: dict[str, Any],
    short_only: dict[str, Any],
    stress: dict[str, Any],
    delayed: dict[str, Any],
    flipped: dict[str, Any],
    signflip: dict[str, Any],
) -> dict[str, bool]:
    annual = primary["2023"]
    return {
        "annual_absolute_return_positive": annual["absolute_return_pct"] > 0.0,
        "annual_cagr_to_strict_mdd_at_least_3": annual["cagr_to_strict_mdd"] >= 3.0,
        "annual_strict_mdd_at_most_15_pct": annual["strict_mdd_pct"] <= 15.0,
        "annual_trades_at_least_150": annual["trades"] >= 150,
        "every_quarter_absolute_return_positive": all(
            primary[name]["absolute_return_pct"] > 0.0 for name in ("q1", "q2", "q3", "q4")
        ),
        "long_only_absolute_return_positive": long_only["absolute_return_pct"] > 0.0,
        "short_only_absolute_return_positive": short_only["absolute_return_pct"] > 0.0,
        "ten_bp_stress_absolute_return_positive": stress["absolute_return_pct"] > 0.0,
        "delay_plus_5m_absolute_return_positive": delayed["absolute_return_pct"] > 0.0,
        "direction_flip_cagr_lower": flipped["cagr_pct"] < annual["cagr_pct"],
        "monthly_cluster_signflip_p_at_most_0_10": signflip["raw_p_value"] <= 0.10,
    }


def evaluate(
    bundle: MarketBundle,
    primary_clock: pd.DataFrame,
    control_clocks: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    primary_full: dict[str, dict[str, Any]] = {}
    for name, (start, end) in WINDOWS.items():
        primary_full[name] = simulate(
            bundle,
            primary_clock,
            start=start,
            end=end,
            cost_bp=CONFIG.base_cost_bp_per_notional_side,
        )
    long_only = simulate(
        bundle,
        transform_clock(primary_clock, "long_only"),
        start=START,
        end=END,
        cost_bp=CONFIG.base_cost_bp_per_notional_side,
    )
    short_only = simulate(
        bundle,
        transform_clock(primary_clock, "short_only"),
        start=START,
        end=END,
        cost_bp=CONFIG.base_cost_bp_per_notional_side,
    )
    stress = simulate(
        bundle,
        primary_clock,
        start=START,
        end=END,
        cost_bp=CONFIG.stress_cost_bp_per_notional_side,
    )
    delayed = simulate(
        bundle,
        transform_clock(primary_clock, "delay_five_minutes"),
        start=START,
        end=END,
        cost_bp=CONFIG.base_cost_bp_per_notional_side,
    )
    flipped = simulate(
        bundle,
        transform_clock(primary_clock, "direction_flip"),
        start=START,
        end=END,
        cost_bp=CONFIG.base_cost_bp_per_notional_side,
    )
    controls = {
        name: simulate(
            bundle,
            clock,
            start=START,
            end=END,
            cost_bp=CONFIG.base_cost_bp_per_notional_side,
        )
        for name, clock in control_clocks.items()
    }
    signflip = monthly_cluster_signflip(
        primary_full["2023"]["trade_rows"],
        seed=CONFIG.monthly_signflip_seed,
        samples=CONFIG.monthly_signflip_samples,
    )
    checks = selection_checks(
        primary_full,
        long_only,
        short_only,
        stress,
        delayed,
        flipped,
        signflip,
    )
    return {
        "primary": {name: _slim(stats) for name, stats in primary_full.items()},
        "long_only": _slim(long_only),
        "short_only": _slim(short_only),
        "ten_bp_notional_side_cost_stress": _slim(stress),
        "entry_and_exit_delay_plus_5m": _slim(delayed),
        "direction_flip": _slim(flipped),
        "mechanism_controls": {name: _slim(stats) for name, stats in controls.items()},
        "monthly_cluster_signflip": signflip,
        "selection_gates": checks,
        "passes_2023_selection": bool(all(checks.values())),
    }


def _markdown(result: dict[str, Any]) -> str:
    evaluation = result["evaluation"]
    lines = [
        "# CRRC-72 frozen 2023 selection outcome — 2026-07-17",
        "",
        f"Decision: **{result['decision']}**",
        "",
        "| Window / control | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    rows = [(name.upper(), evaluation["primary"][name]) for name in ("2023", "q1", "q2", "q3", "q4")]
    rows.extend(
        [
            ("Long only", evaluation["long_only"]),
            ("Short only", evaluation["short_only"]),
            ("10bp stress", evaluation["ten_bp_notional_side_cost_stress"]),
            ("+5m delay", evaluation["entry_and_exit_delay_plus_5m"]),
            ("Direction flip", evaluation["direction_flip"]),
        ]
    )
    rows.extend(
        (f"Control: {name}", stats)
        for name, stats in evaluation["mechanism_controls"].items()
    )
    for label, stats in rows:
        lines.append(
            f"| {label} | {stats['absolute_return_pct']:+.3f}% | {stats['cagr_pct']:+.3f}% | "
            f"{stats['strict_mdd_pct']:.3f}% | {stats['cagr_to_strict_mdd']:.3f} | {stats['trades']} |"
        )
    failed = [name for name, passed in evaluation["selection_gates"].items() if not passed]
    lines.extend(
        [
            "",
            f"- Monthly-cluster sign-flip p: `{evaluation['monthly_cluster_signflip']['raw_p_value']:.6f}`",
            f"- Failed gates: `{failed}`",
            "- CAGR spans the full declared calendar, including warm-up and idle cash.",
            "- Strict MDD uses the global/pre-entry HWM, favorable-before-adverse held OHLC, exact held funding, and entry/hypothetical-liquidation/exit costs.",
            "- Mechanism controls are diagnostics and cannot rerank or replace the frozen primary singleton.",
            "",
        ]
    )
    return "\n".join(lines)


def _clean_repository_head() -> str:
    status = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    if status:
        raise RuntimeError("repository must be clean before CRRC 2023 outcomes open")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def run(
    output: str | Path | None = None,
    docs_output: str | Path | None = None,
) -> dict[str, Any]:
    output_path = DEFAULT_OUTPUT if output is None else Path(output)
    docs_path = DEFAULT_DOCS if docs_output is None else Path(docs_output)
    if output_path != DEFAULT_OUTPUT or docs_path != DEFAULT_DOCS:
        raise ValueError("CRRC outcome output paths are immutable")
    if output_path.exists() or docs_path.exists():
        raise RuntimeError("CRRC 2023 outcome result already exists")
    support, primary, controls = verify_preoutcome_artifacts()
    freeze = verify_evaluation_freeze()
    outcome_opening_head = _clean_repository_head()
    bundle = load_bundle_2023()
    evaluated = evaluate(bundle, primary, controls)
    passed = bool(evaluated["passes_2023_selection"])
    result: dict[str, Any] = {
        "protocol_version": "crrc72_v1_2023_selection_2026-07-17",
        "outcomes_opened": True,
        "opened_window": ["2023-01-01", "2024-01-01"],
        "2024_test_opened": False,
        "2025_eval_opened": False,
        "2026_holdout_opened": False,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "support_sha256": SUPPORT_SHA256,
        "primary_clock_sha256": PRIMARY_CLOCK_SHA256,
        "control_clocks_sha256": CONTROL_CLOCKS_SHA256,
        "execution_source_manifest_hash": EXECUTION_SOURCE_MANIFEST_HASH,
        "market_rows_loaded": source_export.MARKET_ROWS,
        "funding_rows_loaded": source_export.FUNDING_ROWS,
        "maximum_loaded_timestamp_exclusive": str(END),
        "evaluation_config": asdict(CONFIG),
        "support_snapshot": support["support"],
        "evaluation_freeze_hash": freeze["manifest_hash"],
        "outcome_opening_head": outcome_opening_head,
        "evaluation": evaluated,
        "decision": "2023_pass_open_2024_next" if passed else "rejected_before_2024",
        "anti_repair": (
            "A rejection seals 2024+ for CRRC-72; no sign, threshold, hold, scale, "
            "feature, or mechanism-control substitution is permitted."
        ),
    }
    result["manifest_hash"] = canonical_hash(result)
    result["created_at"] = datetime.now(timezone.utc).isoformat()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    with docs_path.open("x", encoding="utf-8") as handle:
        handle.write(_markdown(result))
    return result


def main() -> None:
    result = run()
    evaluation = result["evaluation"]
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "2024_test_opened": result["2024_test_opened"],
                "primary": evaluation["primary"],
                "long_only": evaluation["long_only"],
                "short_only": evaluation["short_only"],
                "controls": {
                    "10bp": evaluation["ten_bp_notional_side_cost_stress"],
                    "+5m": evaluation["entry_and_exit_delay_plus_5m"],
                    "flip": evaluation["direction_flip"],
                    "mechanisms": evaluation["mechanism_controls"],
                },
                "signflip": evaluation["monthly_cluster_signflip"],
                "failed_gates": [
                    key for key, passed in evaluation["selection_gates"].items() if not passed
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
