"""Open AFCH01 2023-2024 selection outcomes while keeping 2025 sealed."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.build_alt_funding_carry_harvest_support import (
    DEFAULT_CLOCK,
    assert_clock_contract,
)
from training.compose_alt_funding_carry_sources import HANDOFF, START, SYMBOLS
from training.export_leave_one_out_residual_exhaustion_sources import sha256_file
from training.preregister_alt_funding_carry_harvest import canonical_hash, protocol


END = pd.Timestamp("2025-01-01 00:00:00")
EXPECTED_PROTOCOL_HASH = "15a7d0adbace0255e1ea4359e4869154dfb34ad891a2125239340ff70c4e2a09"
SOURCE_COMPOSITION_MANIFEST = "results/alt_funding_carry_harvest_v1_source_composition_2026-07-17.json"
EXPECTED_SOURCE_COMPOSITION_MANIFEST_HASH = "18f5455ea248b6ab5e451d734c0d6849ed25a4e67aa19970e19617f95f67fb0d"
SUPPORT_MANIFEST = "results/alt_funding_carry_harvest_v1_support_manifest_2026-07-17.json"
EXPECTED_SUPPORT_MANIFEST_HASH = "c92bfb7b0b3dc424758e6edca7614cdc4701f9d04e798bfaff9572ff003a3cd8"
EXPECTED_CLOCK_HASH = "d0cc451e74a45d5ab9f08a8a7c3d8c3df756d35d2e020d1f70dc0198ebaf0ef9"
FUNDING_MARK_MANIFEST = "results/alt_funding_carry_harvest_v1_funding_marks_2023_2026-07-17.json"
EXPECTED_FUNDING_MARK_MANIFEST_HASH = "344eebba84e747313f062fcda9fa7fbf9960d234e75a2ea66bd507bbe5ad0667"
FUNDING_MARK_DIR = Path("data/binance_um_afch_funding_marks_2023")
DEFAULT_OUTPUT = "results/alt_funding_carry_harvest_v1_selection_2023_2024_2026-07-17.json"
DEFAULT_DOCS = "docs/alt-funding-carry-harvest-v1-selection-2023-2024-2026-07-17.md"
SELECTOR_PATH = "training/select_alt_funding_carry_harvest_pre2025.py"
TEST_PATH = "tests/test_select_alt_funding_carry_harvest_pre2025.py"
BASE_COST_BP = 6.0
STRESS_COST_BP = 10.0
SIGNFLIP_SAMPLES = 20_000
SIGNFLIP_SEED = 171_117


@dataclass(frozen=True)
class MarketBundle:
    dates: pd.DatetimeIndex
    market: dict[str, dict[str, np.ndarray]]
    funding: dict[str, pd.DataFrame]


@dataclass
class ActiveSleeve:
    sleeve_id: int
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    long_symbol: str
    short_symbol: str
    long_qty: float
    short_qty: float
    long_entry: float
    short_entry: float
    entry_equity: float
    entry_cost: float
    funding_cash: float = 0.0
    positive_funding_credit: float = 0.0


def _file_hash(path: str | Path) -> str:
    return sha256_file(Path(path))


def _git_attestation() -> dict[str, str]:
    status = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    if status.strip():
        raise RuntimeError("repository must be clean before opening AFCH outcomes")
    for path in (SELECTOR_PATH, TEST_PATH):
        subprocess.check_call(["git", "ls-files", "--error-unmatch", path], stdout=subprocess.DEVNULL)
    return {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "selector_sha256": _file_hash(SELECTOR_PATH),
        "test_sha256": _file_hash(TEST_PATH),
    }


def _manifest(path: str, expected: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if payload.get("manifest_hash") != expected:
        raise RuntimeError(f"manifest hash changed: {path}")
    body = {key: value for key, value in payload.items() if key not in {"manifest_hash", "created_at"}}
    if canonical_hash(body) != expected:
        raise RuntimeError(f"manifest body changed: {path}")
    return payload


def _validate_funding_mark_manifest(payload: dict[str, Any]) -> None:
    if payload.get("outcomes_opened") is not False:
        raise RuntimeError("AFCH funding mark freeze opened outcomes")
    limit = float(payload.get("maximum_proxy_funding_cash_error_bp_notional", float("nan")))
    records = {str(row["symbol"]): row for row in payload.get("records", [])}
    if not math.isfinite(limit) or limit != 0.1 or set(records) != set(SYMBOLS):
        raise RuntimeError("AFCH funding mark freeze contract changed")
    if any(float(row["maximum_proxy_funding_cash_error_bp_notional"]) > limit for row in records.values()):
        raise RuntimeError("AFCH causal funding mark uncertainty exceeds freeze")
    if any(int(row["events"]) != 1095 for row in records.values()):
        raise RuntimeError("AFCH causal funding mark event support changed")


def compose_pre2025_market(old: pd.DataFrame, recent: pd.DataFrame, symbol: str) -> pd.DataFrame:
    frames = []
    for frame, start, end in ((old.copy(), START, HANDOFF), (recent.copy(), HANDOFF, END)):
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        if not frame["tic"].astype(str).eq(symbol).all():
            raise RuntimeError(f"{symbol} AFCH market identity failure")
        frames.append(frame.loc[(frame["date"] >= start) & (frame["date"] < end)])
    combined = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    if combined["date"].duplicated().any() or not combined["date"].is_monotonic_increasing:
        raise RuntimeError(f"{symbol} AFCH composite market order failure")
    return combined


def compose_pre2025_funding(old: pd.DataFrame, recent: pd.DataFrame, symbol: str) -> pd.DataFrame:
    frames = []
    for frame, start, end in ((old.copy(), START, HANDOFF), (recent.copy(), HANDOFF, END)):
        if not frame["symbol"].astype(str).eq(symbol).all():
            raise RuntimeError(f"{symbol} AFCH funding identity failure")
        frame["funding_time"] = pd.to_numeric(frame["funding_time"], errors="raise").astype("int64")
        frame["event_time"] = pd.to_datetime(frame["funding_time"], unit="ms")
        frame["funding_rate"] = pd.to_numeric(frame["funding_rate"], errors="raise")
        frame["mark_price"] = pd.to_numeric(frame["mark_price"], errors="coerce")
        frames.append(frame.loc[(frame["event_time"] >= start) & (frame["event_time"] < end)])
    combined = pd.concat(frames, ignore_index=True).sort_values("event_time").reset_index(drop=True)
    combined["mark_is_recorded"] = combined["mark_price"].notna()
    if combined["event_time"].duplicated().any() or not combined["event_time"].is_monotonic_increasing:
        raise RuntimeError(f"{symbol} AFCH composite funding order failure")
    return combined


def load_bundle() -> MarketBundle:
    source = _manifest(SOURCE_COMPOSITION_MANIFEST, EXPECTED_SOURCE_COMPOSITION_MANIFEST_HASH)
    mark_source = _manifest(FUNDING_MARK_MANIFEST, EXPECTED_FUNDING_MARK_MANIFEST_HASH)
    records = {str(row["symbol"]): row for row in source["records"]}
    mark_records = {str(row["symbol"]): row for row in mark_source["records"]}
    market: dict[str, dict[str, np.ndarray]] = {}
    funding: dict[str, pd.DataFrame] = {}
    dates: pd.DatetimeIndex | None = None
    for symbol in sorted(SYMBOLS):
        record = records[symbol]
        paths = {
            "old_market": Path(record["old_market_path"]),
            "recent_market": Path(record["recent_market_path"]),
            "old_funding": Path(record["old_funding_path"]),
            "recent_funding": Path(record["recent_funding_path"]),
        }
        if any(_file_hash(path) != record[f"{key}_sha256"] for key, path in paths.items()):
            raise RuntimeError(f"{symbol} AFCH composite source changed")
        old_market = pd.read_csv(
            paths["old_market"],
            usecols=["date", "open", "high", "low", "close", "tic"],
        )
        recent_market = pd.read_csv(
            paths["recent_market"],
            usecols=["date", "open", "high", "low", "close", "tic"],
        )
        frame = compose_pre2025_market(old_market, recent_market, symbol)
        current_dates = pd.DatetimeIndex(frame["date"])
        if dates is None:
            dates = current_dates
        elif not dates.equals(current_dates):
            raise RuntimeError("AFCH symbol market grids differ")
        arrays = {
            column: pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
            for column in ("open", "high", "low", "close")
        }
        if not np.isfinite(np.column_stack(list(arrays.values()))).all():
            raise RuntimeError(f"{symbol} AFCH non-finite market source")
        market[symbol] = arrays
        funding_columns = ["symbol", "funding_time", "funding_rate", "mark_price"]
        old_funding = pd.read_csv(paths["old_funding"], usecols=funding_columns)
        recent_funding = pd.read_csv(paths["recent_funding"], usecols=funding_columns)
        fund = compose_pre2025_funding(old_funding, recent_funding, symbol)
        mark_path = FUNDING_MARK_DIR / f"{symbol}_funding_marks_2023.csv.gz"
        if _file_hash(mark_path) != mark_records[symbol]["output_sha256"]:
            raise RuntimeError(f"{symbol} AFCH causal funding marks changed")
        marks = pd.read_csv(mark_path, parse_dates=["event_time"])
        marks["funding_time"] = pd.to_numeric(marks["funding_time"], errors="raise").astype("int64")
        marks["causal_mark_price"] = pd.to_numeric(marks["causal_mark_price"], errors="raise")
        if marks["funding_time"].duplicated().any() or len(marks) != 1095:
            raise RuntimeError(f"{symbol} AFCH causal funding mark identity failure")
        expected_event = pd.to_datetime(marks["funding_time"], unit="ms")
        if not pd.DatetimeIndex(marks["event_time"]).equals(pd.DatetimeIndex(expected_event)):
            raise RuntimeError(f"{symbol} AFCH causal funding mark time mismatch")
        if not set(marks["mark_source"]).issubset({"funding_record", "prior_completed_mark_close"}):
            raise RuntimeError(f"{symbol} AFCH causal funding mark source mismatch")
        mark_by_time = marks.set_index("funding_time")
        year_2023 = fund["event_time"] < pd.Timestamp("2024-01-01")
        frozen_marks = fund.loc[year_2023, "funding_time"].map(mark_by_time["causal_mark_price"])
        frozen_sources = fund.loc[year_2023, "funding_time"].map(mark_by_time["mark_source"])
        if frozen_marks.isna().any() or frozen_sources.isna().any():
            raise RuntimeError(f"{symbol} AFCH causal funding mark coverage failure")
        fund.loc[year_2023, "mark_price"] = frozen_marks.to_numpy(dtype=float)
        fund.loc[year_2023, "mark_is_recorded"] = frozen_sources.eq("funding_record").to_numpy()
        if fund["mark_price"].isna().any():
            raise RuntimeError(f"{symbol} AFCH official selector would require an unfrozen mark proxy")
        finite_mark = np.isfinite(fund["mark_price"].to_numpy(dtype=float))
        if (fund.loc[finite_mark, "mark_price"] <= 0.0).any():
            raise RuntimeError(f"{symbol} AFCH non-positive exact funding mark")
        funding[symbol] = fund[["event_time", "funding_rate", "mark_price", "mark_is_recorded"]].copy()
    assert dates is not None
    expected = pd.date_range(START, END - pd.Timedelta(minutes=5), freq="5min")
    if not dates.equals(expected):
        raise RuntimeError("AFCH selector source is not exact 2023-2024")
    return MarketBundle(dates, market, funding)


def load_clock(path: str = DEFAULT_CLOCK) -> pd.DataFrame:
    if _file_hash(path) != EXPECTED_CLOCK_HASH:
        raise RuntimeError("AFCH frozen support clock changed")
    clock = pd.read_csv(path)
    assert_clock_contract(clock)
    for column in ("signal_time", "feature_available_time", "entry_time", "exit_time"):
        clock[column] = pd.to_datetime(clock[column], errors="raise")
    return clock


def _funding_events_by_bar(
    bundle: MarketBundle,
) -> dict[int, list[tuple[pd.Timestamp, str, float, float, bool]]]:
    out: dict[int, list[tuple[pd.Timestamp, str, float, float, bool]]] = {}
    dates_ns = bundle.dates.to_numpy(dtype="datetime64[ns]")
    completed_ns = (bundle.dates + pd.Timedelta(minutes=5)).to_numpy(dtype="datetime64[ns]")
    for symbol, frame in bundle.funding.items():
        for row in frame.itertuples(index=False):
            event = pd.Timestamp(row.event_time)
            event64 = event.to_datetime64()
            bar_index = int(np.searchsorted(dates_ns, event64, side="right") - 1)
            mark_index = int(np.searchsorted(completed_ns, event64, side="right") - 1)
            if bar_index < 0 or bar_index >= len(bundle.dates):
                continue
            recorded_mark = float(getattr(row, "mark_price", float("nan")))
            mark_is_recorded = getattr(row, "mark_is_recorded", None)
            mark_available = math.isfinite(recorded_mark) and recorded_mark > 0.0
            exact_mark = mark_available if mark_is_recorded is None else bool(mark_is_recorded)
            if mark_available:
                mark = recorded_mark
            elif mark_index >= 0:
                mark = float(bundle.market[symbol]["close"][mark_index])
            else:
                continue
            out.setdefault(bar_index, []).append(
                (event, symbol, float(row.funding_rate), mark, exact_mark)
            )
    for events in out.values():
        events.sort(key=lambda item: (item[0], item[1]))
    return out


def _index(bundle: MarketBundle, timestamp: pd.Timestamp) -> int:
    index = int(bundle.dates.searchsorted(timestamp))
    if index >= len(bundle.dates) or bundle.dates[index] != timestamp:
        raise RuntimeError(f"missing exact AFCH 5m open: {timestamp}")
    return index


def _active_unrealized(active: dict[int, ActiveSleeve], bundle: MarketBundle, bar_index: int, field: str) -> float:
    total = 0.0
    for sleeve in active.values():
        long_price = float(bundle.market[sleeve.long_symbol][field][bar_index])
        short_price = float(bundle.market[sleeve.short_symbol][field][bar_index])
        total += sleeve.long_qty * (long_price - sleeve.long_entry)
        total += sleeve.short_qty * (sleeve.short_entry - short_price)
    return total


def _net_quantities(active: dict[int, ActiveSleeve]) -> dict[str, float]:
    net = {symbol: 0.0 for symbol in SYMBOLS}
    for sleeve in active.values():
        net[sleeve.long_symbol] += sleeve.long_qty
        net[sleeve.short_symbol] -= sleeve.short_qty
    return net


def _path_mark(
    active: dict[int, ActiveSleeve],
    bundle: MarketBundle,
    bar_index: int,
    *,
    favorable: bool,
    cost_rate: float,
) -> tuple[float, float]:
    net = _net_quantities(active)
    chosen: dict[str, float] = {}
    liquidation = 0.0
    for symbol, quantity in net.items():
        if quantity > 0:
            field = "high" if favorable else "low"
        elif quantity < 0:
            field = "low" if favorable else "high"
        else:
            field = "close"
        price = float(bundle.market[symbol][field][bar_index])
        chosen[symbol] = price
        liquidation += abs(quantity) * price * cost_rate
    pnl = 0.0
    for sleeve in active.values():
        pnl += sleeve.long_qty * (chosen[sleeve.long_symbol] - sleeve.long_entry)
        pnl += sleeve.short_qty * (sleeve.short_entry - chosen[sleeve.short_symbol])
    return pnl, liquidation


def weekly_cluster_signflip(sleeve_rows: list[dict[str, Any]], seed: int, samples: int = SIGNFLIP_SAMPLES) -> dict[str, Any]:
    if not sleeve_rows:
        return {"samples": samples, "weekly_clusters": 0, "observed_net_cash": 0.0, "raw_p_value": 1.0}
    frame = pd.DataFrame(sleeve_rows)
    frame["week"] = pd.to_datetime(frame["signal_time"]).dt.to_period("W-SUN").astype(str)
    clusters = frame.groupby("week")["net_cash_initial"].sum().to_numpy(dtype=float)
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
        "observed_net_cash": observed,
        "raw_p_value": float((1 + np.count_nonzero(draws >= observed - 1e-15)) / (samples + 1)),
        "random_positive_fraction": float(np.mean(draws > 0.0)),
        "random_q95_net_cash": float(np.quantile(draws, 0.95)),
    }


def simulate(
    bundle: MarketBundle,
    clock: pd.DataFrame,
    *,
    start: str,
    end: str,
    cost_bp: float = BASE_COST_BP,
    include_funding: bool = True,
) -> dict[str, Any]:
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    if not START <= start_ts < end_ts <= END:
        raise ValueError("AFCH selector escaped 2023-2024")
    if cost_bp < 0:
        raise ValueError("negative execution cost")
    cost_rate = cost_bp / 10_000.0
    selected = clock.loc[(clock["entry_time"] >= start_ts) & (clock["exit_time"] < end_ts)].copy()
    selected = selected.sort_values(["entry_time", "signal_time"]).reset_index(drop=True)
    if selected.empty:
        years = (end_ts - start_ts).total_seconds() / (365.25 * 86_400.0)
        return {
            "absolute_return_pct": 0.0, "cagr_pct": 0.0, "strict_mdd_pct": 0.0,
            "close_mdd_pct": 0.0, "cagr_to_strict_mdd": 0.0, "sleeves": 0,
            "mean_sleeve_net_bps": 0.0, "win_rate": 0.0, "funding_cash_pct_initial": 0.0,
            "price_pnl_pct_initial": 0.0, "transaction_cost_pct_initial": 0.0,
            "funding_to_cost_ratio": 0.0, "exact_mark_funding_applications": 0,
            "proxy_mark_funding_applications": 0, "exact_mark_funding_cash_pct_initial": 0.0,
            "proxy_mark_funding_cash_pct_initial": 0.0, "calendar_start": str(start_ts.date()),
            "calendar_end_exclusive": str(end_ts.date()), "sleeve_rows": [], "calendar_years": years,
        }
    entry_map: dict[int, list[tuple[int, Any]]] = {}
    exit_map: dict[int, list[int]] = {}
    for sleeve_id, row in enumerate(selected.itertuples(index=False)):
        entry_index, exit_index = _index(bundle, pd.Timestamp(row.entry_time)), _index(bundle, pd.Timestamp(row.exit_time))
        entry_map.setdefault(entry_index, []).append((sleeve_id, row))
        exit_map.setdefault(exit_index, []).append(sleeve_id)
    funding_by_bar = _funding_events_by_bar(bundle)
    active: dict[int, ActiveSleeve] = {}
    cash = 1.0
    peak = close_peak = 1.0
    strict_mdd = close_mdd = 0.0
    total_funding = total_price_pnl = total_cost = 0.0
    exact_mark_funding = proxy_mark_funding = 0.0
    exact_mark_applications = proxy_mark_applications = 0
    sleeve_rows: list[dict[str, Any]] = []

    def actual_equity_at_open(bar_index: int) -> float:
        return cash + _active_unrealized(active, bundle, bar_index, "open")

    def strict_equity_at_open(bar_index: int, *, for_peak: bool) -> float:
        net = _net_quantities(active)
        liquidation = cost_rate * sum(
            abs(quantity) * float(bundle.market[symbol]["open"][bar_index])
            for symbol, quantity in net.items()
        )
        equity = actual_equity_at_open(bar_index) - liquidation
        if for_peak:
            equity -= sum(sleeve.positive_funding_credit for sleeve in active.values())
        return equity

    def apply_funding(event: tuple[pd.Timestamp, str, float, float, bool]) -> None:
        nonlocal cash, total_funding, exact_mark_funding, proxy_mark_funding
        nonlocal exact_mark_applications, proxy_mark_applications
        event_time, symbol, funding_rate, mark, exact_mark = event
        if not include_funding:
            return
        for sleeve in active.values():
            if not sleeve.entry_time < event_time <= sleeve.exit_time:
                continue
            signed_qty = 0.0
            if sleeve.long_symbol == symbol:
                signed_qty += sleeve.long_qty
            if sleeve.short_symbol == symbol:
                signed_qty -= sleeve.short_qty
            if signed_qty == 0.0:
                continue
            funding_cash = -signed_qty * mark * funding_rate
            sleeve.funding_cash += funding_cash
            sleeve.positive_funding_credit += max(funding_cash, 0.0)
            cash += funding_cash
            total_funding += funding_cash
            if exact_mark:
                exact_mark_funding += funding_cash
                exact_mark_applications += 1
            else:
                proxy_mark_funding += funding_cash
                proxy_mark_applications += 1

    first_index = min(entry_map)
    last_index = max(exit_map)
    for bar_index in range(first_index, last_index + 1):
        timestamp = bundle.dates[bar_index]
        events = funding_by_bar.get(bar_index, [])
        for event in events:
            if event[0] <= timestamp:
                apply_funding(event)
        for sleeve_id in sorted(exit_map.get(bar_index, [])):
            sleeve = active.pop(sleeve_id)
            long_exit = float(bundle.market[sleeve.long_symbol]["open"][bar_index])
            short_exit = float(bundle.market[sleeve.short_symbol]["open"][bar_index])
            price_pnl = sleeve.long_qty * (long_exit - sleeve.long_entry)
            price_pnl += sleeve.short_qty * (sleeve.short_entry - short_exit)
            exit_cost = cost_rate * (sleeve.long_qty * long_exit + sleeve.short_qty * short_exit)
            cash += price_pnl - exit_cost
            total_price_pnl += price_pnl
            total_cost += exit_cost
            net_cash = price_pnl + sleeve.funding_cash - sleeve.entry_cost - exit_cost
            sleeve_rows.append({
                "sleeve_id": sleeve_id,
                "signal_time": str(sleeve.signal_time),
                "entry_time": str(sleeve.entry_time),
                "exit_time": str(sleeve.exit_time),
                "long_symbol": sleeve.long_symbol,
                "short_symbol": sleeve.short_symbol,
                "price_pnl_cash": float(price_pnl),
                "funding_cash": float(sleeve.funding_cash),
                "entry_cost_cash": float(sleeve.entry_cost),
                "exit_cost_cash": float(exit_cost),
                "net_cash_initial": float(net_cash),
                "net_return_on_entry_equity": float(net_cash / sleeve.entry_equity),
            })
            strict_equity_after_exit = strict_equity_at_open(bar_index, for_peak=False)
            strict_mdd = max(strict_mdd, 1.0 - strict_equity_after_exit / peak)
            strict_peak_after_exit = strict_equity_at_open(bar_index, for_peak=True)
            peak = max(peak, strict_peak_after_exit)
            equity_after_exit = actual_equity_at_open(bar_index)
            close_peak = max(close_peak, equity_after_exit)
            close_mdd = max(close_mdd, 1.0 - equity_after_exit / close_peak)
        for sleeve_id, row in entry_map.get(bar_index, []):
            peak = max(peak, strict_equity_at_open(bar_index, for_peak=True))
            entry_equity = actual_equity_at_open(bar_index)
            gross_notional = max(entry_equity, 0.0) * float(row.sleeve_gross)
            long_entry = float(bundle.market[str(row.long_symbol)]["open"][bar_index])
            short_entry = float(bundle.market[str(row.short_symbol)]["open"][bar_index])
            long_qty = gross_notional * float(row.long_weight_norm) / long_entry
            short_qty = gross_notional * float(row.short_weight_norm) / short_entry
            entry_cost = cost_rate * gross_notional
            cash -= entry_cost
            total_cost += entry_cost
            active[sleeve_id] = ActiveSleeve(
                sleeve_id=sleeve_id,
                signal_time=pd.Timestamp(row.signal_time),
                entry_time=pd.Timestamp(row.entry_time),
                exit_time=pd.Timestamp(row.exit_time),
                long_symbol=str(row.long_symbol),
                short_symbol=str(row.short_symbol),
                long_qty=long_qty,
                short_qty=short_qty,
                long_entry=long_entry,
                short_entry=short_entry,
                entry_equity=entry_equity,
                entry_cost=entry_cost,
            )
            strict_equity_after_entry = strict_equity_at_open(bar_index, for_peak=False)
            strict_mdd = max(strict_mdd, 1.0 - strict_equity_after_entry / peak)
            equity_after_entry = actual_equity_at_open(bar_index)
            close_peak = max(close_peak, equity_after_entry)
            close_mdd = max(close_mdd, 1.0 - equity_after_entry / close_peak)
        for event in events:
            if event[0] > timestamp:
                apply_funding(event)
        if not active:
            peak = max(peak, cash)
            close_peak = max(close_peak, cash)
            continue
        favorable_pnl, favorable_liquidation = _path_mark(
            active, bundle, bar_index, favorable=True, cost_rate=cost_rate
        )
        uncrystallized_positive_funding = sum(sleeve.positive_funding_credit for sleeve in active.values())
        favorable_equity_for_peak = cash - uncrystallized_positive_funding + favorable_pnl - favorable_liquidation
        peak = max(peak, favorable_equity_for_peak)
        adverse_pnl, adverse_liquidation = _path_mark(
            active, bundle, bar_index, favorable=False, cost_rate=cost_rate
        )
        adverse_equity = cash + adverse_pnl - adverse_liquidation
        strict_mdd = max(strict_mdd, 1.0 - adverse_equity / peak)
        close_pnl = _active_unrealized(active, bundle, bar_index, "close")
        net = _net_quantities(active)
        close_liquidation = cost_rate * sum(
            abs(quantity) * float(bundle.market[symbol]["close"][bar_index])
            for symbol, quantity in net.items()
        )
        close_equity = cash + close_pnl - close_liquidation
        close_peak = max(close_peak, close_equity)
        close_mdd = max(close_mdd, 1.0 - close_equity / close_peak)
        if adverse_equity <= 0.0:
            strict_mdd = 1.0
    if active:
        raise RuntimeError("AFCH split ended with active contained sleeve")
    final_equity = cash
    strict_mdd = max(strict_mdd, 1.0 - final_equity / peak)
    close_peak = max(close_peak, final_equity)
    close_mdd = max(close_mdd, 1.0 - final_equity / close_peak)
    years = (end_ts - start_ts).total_seconds() / (365.25 * 86_400.0)
    absolute = (final_equity - 1.0) * 100.0
    cagr = (final_equity ** (1.0 / years) - 1.0) * 100.0 if final_equity > 0 else -100.0
    strict_pct = min(max(strict_mdd * 100.0, 0.0), 100.0)
    sleeve_returns = np.asarray([row["net_return_on_entry_equity"] for row in sleeve_rows], dtype=float)
    return {
        "absolute_return_pct": float(absolute),
        "cagr_pct": float(cagr),
        "strict_mdd_pct": float(strict_pct),
        "close_mdd_pct": float(min(max(close_mdd * 100.0, 0.0), 100.0)),
        "cagr_to_strict_mdd": float(cagr / strict_pct) if strict_pct > 1e-12 else 0.0,
        "sleeves": len(sleeve_rows),
        "mean_sleeve_net_bps": float(sleeve_returns.mean() * 10_000.0) if len(sleeve_returns) else 0.0,
        "win_rate": float(np.mean(sleeve_returns > 0.0)) if len(sleeve_returns) else 0.0,
        "funding_cash_pct_initial": float(total_funding * 100.0),
        "price_pnl_pct_initial": float(total_price_pnl * 100.0),
        "transaction_cost_pct_initial": float(total_cost * 100.0),
        "funding_to_cost_ratio": float(total_funding / total_cost) if total_cost > 1e-15 else 0.0,
        "exact_mark_funding_applications": exact_mark_applications,
        "proxy_mark_funding_applications": proxy_mark_applications,
        "exact_mark_funding_cash_pct_initial": float(exact_mark_funding * 100.0),
        "proxy_mark_funding_cash_pct_initial": float(proxy_mark_funding * 100.0),
        "calendar_start": str(start_ts.date()),
        "calendar_end_exclusive": str(end_ts.date()),
        "calendar_years": years,
        "sleeve_rows": sleeve_rows,
    }


def _transform_clock(clock: pd.DataFrame, kind: str, seed: int = SIGNFLIP_SEED) -> pd.DataFrame:
    out = clock.copy()
    if kind == "direction_flip":
        out[["long_symbol", "short_symbol"]] = out[["short_symbol", "long_symbol"]].to_numpy()
        out[["long_weight_norm", "short_weight_norm"]] = out[["short_weight_norm", "long_weight_norm"]].to_numpy()
        out[["long_beta", "short_beta"]] = out[["short_beta", "long_beta"]].to_numpy()
    elif kind == "equal_weight":
        out["long_weight_norm"] = out["short_weight_norm"] = 0.5
    elif kind == "shift_three_days":
        for column in ("signal_time", "feature_available_time", "entry_time", "exit_time"):
            out[column] = pd.to_datetime(out[column]) + pd.Timedelta(days=3)
    elif kind == "quarterly_pair_permutation":
        rng = np.random.default_rng(seed)
        out["signal_time"] = pd.to_datetime(out["signal_time"])
        columns = [
            "long_symbol", "short_symbol", "long_weight_norm", "short_weight_norm",
            "long_beta", "short_beta",
        ]
        for _, indices in out.groupby(out["signal_time"].dt.to_period("Q")).groups.items():
            labels = np.asarray(list(indices), dtype=int)
            shuffled = labels.copy()
            rng.shuffle(shuffled)
            out.loc[labels, columns] = out.loc[shuffled, columns].to_numpy()
    else:
        raise ValueError(kind)
    return out


WINDOWS = {
    "fit_2023": ("2023-01-01", "2024-01-01"),
    "test_2024": ("2024-01-01", "2025-01-01"),
    "combined_2023_2024": ("2023-01-01", "2025-01-01"),
    "2023_h1": ("2023-01-01", "2023-07-01"),
    "2023_h2": ("2023-07-01", "2024-01-01"),
    "2024_h1": ("2024-01-01", "2024-07-01"),
    "2024_h2": ("2024-07-01", "2025-01-01"),
}


def _slim(stats: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in stats.items() if key != "sleeve_rows"}


def evaluate(bundle: MarketBundle, clock: pd.DataFrame) -> dict[str, Any]:
    raw = {name: simulate(bundle, clock, start=start, end=end) for name, (start, end) in WINDOWS.items()}
    combined_raw = raw["combined_2023_2024"]
    stress = simulate(
        bundle, clock, start="2023-01-01", end="2025-01-01", cost_bp=STRESS_COST_BP
    )
    signflip = weekly_cluster_signflip(combined_raw["sleeve_rows"], seed=SIGNFLIP_SEED)
    diagnostics = {}
    for kind in ("direction_flip", "equal_weight", "shift_three_days", "quarterly_pair_permutation"):
        diagnostics[kind] = _slim(simulate(
            bundle,
            _transform_clock(clock, kind),
            start="2023-01-01",
            end="2025-01-01",
        ))
    diagnostics["funding_removed"] = _slim(simulate(
        bundle, clock, start="2023-01-01", end="2025-01-01", include_funding=False
    ))
    annual = [raw["fit_2023"], raw["test_2024"]]
    halves = [raw[name] for name in ("2023_h1", "2023_h2", "2024_h1", "2024_h2")]
    combined = raw["combined_2023_2024"]
    gates = {
        "each_year_absolute_return_positive": all(stats["absolute_return_pct"] > 0.0 for stats in annual),
        "each_year_cagr_to_strict_mdd_at_least_1_5": all(stats["cagr_to_strict_mdd"] >= 1.5 for stats in annual),
        "positive_half_years_at_least_3": sum(stats["absolute_return_pct"] > 0.0 for stats in halves) >= 3,
        "combined_cagr_to_strict_mdd_at_least_3": combined["cagr_to_strict_mdd"] >= 3.0,
        "combined_strict_mdd_at_most_15": combined["strict_mdd_pct"] <= 15.0,
        "ten_bp_cost_stress_absolute_return_positive": stress["absolute_return_pct"] > 0.0,
        "weekly_cluster_signflip_p_at_most_0_10": signflip["raw_p_value"] <= 0.10,
        "realized_funding_cash_positive": combined["funding_cash_pct_initial"] > 0.0,
        "realized_funding_cash_at_least_transaction_cost": combined["funding_to_cost_ratio"] >= 1.0,
    }
    return {
        "stats": {name: _slim(stats) for name, stats in raw.items()},
        "ten_bp_notional_side_cost_stress": _slim(stress),
        "weekly_cluster_signflip": signflip,
        "diagnostic_controls": diagnostics,
        "selection_gates": gates,
        "passes_2023_2024_selection": all(gates.values()),
    }


def _markdown(result: dict[str, Any]) -> str:
    evaluation = result["evaluation"]
    rows = []
    labels = {
        "fit_2023": "2023 fit", "test_2024": "2024 test",
        "combined_2023_2024": "2023–2024", "2023_h1": "2023 H1",
        "2023_h2": "2023 H2", "2024_h1": "2024 H1", "2024_h2": "2024 H2",
    }
    for name in ("fit_2023", "test_2024", "combined_2023_2024", "2023_h1", "2023_h2", "2024_h1", "2024_h2"):
        stats = evaluation["stats"][name]
        rows.append(
            f"| {labels[name]} | {stats['absolute_return_pct']:+.3f}% | {stats['cagr_pct']:+.3f}% | "
            f"{stats['strict_mdd_pct']:.3f}% | {stats['cagr_to_strict_mdd']:.3f} | "
            f"{stats['sleeves']} | {stats['funding_cash_pct_initial']:+.3f}% | "
            f"{stats['transaction_cost_pct_initial']:.3f}% |"
        )
    failed = [name for name, passed in evaluation["selection_gates"].items() if not passed]
    return "\n".join([
        "# AFCH v1 2023–2024 selection — 2026-07-17", "",
        "> Only 2023–2024 execution outcomes were opened. Calendar 2025 and 2026 remain sealed.", "",
        "| Window | Absolute return | CAGR | strict MDD | CAGR/MDD | Sleeves | Funding cash | Costs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|", *rows, "", "## Decision", "",
        f"- status: **{result['decision']}**",
        f"- failed gates: `{failed}`",
        f"- weekly cluster sign-flip p: `{evaluation['weekly_cluster_signflip']['raw_p_value']:.6f}`",
        "- Funding cash and transaction cost are reported as cash percentages of initial equity; price PnL is separated in the JSON artifact.",
        "- Recorded settlement `mark_price` is used when present; missing 2023 marks use the separately frozen last-completed 5m mark-price close, with exact/proxy application counts and cash split in JSON.",
        "- CAGR includes every idle day in the declared calendar.",
        "- strict MDD uses aggregate net-symbol favorable-before-adverse OHLC, global HWM, active funding-credit exclusion, and entry/exit/hypothetical liquidation costs.",
        "- Diagnostics cannot repair a failed AFCH01 policy.", "",
    ])


def run(output: str = DEFAULT_OUTPUT, docs_output: str = DEFAULT_DOCS) -> dict[str, Any]:
    if canonical_hash(protocol()) != EXPECTED_PROTOCOL_HASH:
        raise RuntimeError("AFCH preregistration drifted")
    support = _manifest(SUPPORT_MANIFEST, EXPECTED_SUPPORT_MANIFEST_HASH)
    if support.get("clock_sha256") != EXPECTED_CLOCK_HASH or not support["support"].get("passes_support"):
        raise RuntimeError("AFCH support freeze is not approved")
    funding_marks = _manifest(FUNDING_MARK_MANIFEST, EXPECTED_FUNDING_MARK_MANIFEST_HASH)
    _validate_funding_mark_manifest(funding_marks)
    attestation = _git_attestation()
    bundle, clock = load_bundle(), load_clock()
    evaluation = evaluate(bundle, clock)
    passed = evaluation["passes_2023_2024_selection"]
    result: dict[str, Any] = {
        "protocol_version": "afch_v1_pre2025_selector_2026-07-17",
        "outcomes_opened": True,
        "opened_window": ["2023-01-01", "2025-01-01"],
        "eval_2025_opened": False,
        "final_2026_opened": False,
        "preregistration_protocol_hash": EXPECTED_PROTOCOL_HASH,
        "source_composition_manifest_hash": EXPECTED_SOURCE_COMPOSITION_MANIFEST_HASH,
        "support_manifest_hash": EXPECTED_SUPPORT_MANIFEST_HASH,
        "funding_mark_manifest_hash": EXPECTED_FUNDING_MARK_MANIFEST_HASH,
        "clock_hash": EXPECTED_CLOCK_HASH,
        "single_policy_id": "AFCH01",
        "evaluation": evaluation,
        "decision": "passed_2023_2024_pending_2025" if passed else "rejected_before_2025",
        "pre_outcome_selector_attestation": attestation,
    }
    result["manifest_hash"] = canonical_hash(result)
    result["created_at"] = datetime.now(timezone.utc).isoformat()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    Path(docs_output).parent.mkdir(parents=True, exist_ok=True)
    Path(docs_output).write_text(_markdown(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--docs-output", default=DEFAULT_DOCS)
    args = parser.parse_args()
    result = run(args.output, args.docs_output)
    print(json.dumps({
        "decision": result["decision"],
        "stats": result["evaluation"]["stats"],
        "stress": result["evaluation"]["ten_bp_notional_side_cost_stress"],
        "signflip": result["evaluation"]["weekly_cluster_signflip"],
        "gates": result["evaluation"]["selection_gates"],
    }, indent=2))


if __name__ == "__main__":
    main()
