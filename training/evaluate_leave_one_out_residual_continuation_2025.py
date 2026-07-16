"""Open the single frozen LORC01 calendar-2025 holdout, and nothing later."""
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

from training.build_leave_one_out_residual_continuation_2025_support import (
    DEFAULT_CLOCKS,
    DEFAULT_SOURCE_DIR,
    EXPECTED_SOURCE_MANIFEST_HASH,
    SOURCE_MANIFEST,
    assert_clock_contract,
)
from training.export_leave_one_out_residual_continuation_2025_sources import (
    END,
    EXPECTED_PROTOCOL_HASH,
    HOLDOUT_START,
    START,
    SYMBOLS,
)
from training.export_leave_one_out_residual_exhaustion_sources import sha256_file
from training.preregister_leave_one_out_residual_continuation import canonical_hash, protocol
from training.select_leave_one_out_residual_exhaustion_pre2025 import weekly_cluster_signflip


SUPPORT_MANIFEST = "results/leave_one_out_residual_continuation_v1_support_manifest_2026-07-17.json"
EXPECTED_SUPPORT_MANIFEST_HASH = "081f4d67e99e0a8e8d6059eb4018b0a98e595433cb15151d417ace42df99c230"
EXPECTED_CLOCK_HASH = "b7bdb75831ad597cb23212740e571760de9210a87f709eaddd35cedba5612956"
DEFAULT_OUTPUT = "results/leave_one_out_residual_continuation_v1_holdout_2025_2026-07-17.json"
DEFAULT_DOCS = "docs/leave-one-out-residual-continuation-v1-holdout-2025-2026-07-17.md"
EVALUATOR_PATH = "training/evaluate_leave_one_out_residual_continuation_2025.py"
TEST_PATH = "tests/test_evaluate_leave_one_out_residual_continuation_2025.py"
BASE_COST_BP = 6.0
STRESS_COST_BP = 10.0
SIGNFLIP_SAMPLES = 20_000
SIGNFLIP_SEED = 171_017


@dataclass(frozen=True)
class MarketBundle:
    dates: pd.DatetimeIndex
    market: dict[str, dict[str, np.ndarray]]
    funding: dict[str, pd.DataFrame]
    source_hashes: dict[str, dict[str, str]]


def _file_hash(path: str | Path) -> str:
    return sha256_file(Path(path))


def _git_attestation() -> dict[str, str]:
    status = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    if status.strip():
        raise RuntimeError("repository must be clean before opening the LORC 2025 holdout")
    for path in (EVALUATOR_PATH, TEST_PATH):
        subprocess.check_call(["git", "ls-files", "--error-unmatch", path], stdout=subprocess.DEVNULL)
    return {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "evaluator_sha256": _file_hash(EVALUATOR_PATH),
        "test_sha256": _file_hash(TEST_PATH),
    }


def _load_manifest(path: str, expected: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if payload.get("manifest_hash") != expected:
        raise RuntimeError(f"manifest hash mismatch: {path}")
    body = {k: v for k, v in payload.items() if k not in {"manifest_hash", "created_at"}}
    if canonical_hash(body) != expected:
        raise RuntimeError(f"manifest body mismatch: {path}")
    return payload


def load_bundle(
    source_dir: str = DEFAULT_SOURCE_DIR,
    source_manifest_path: str = SOURCE_MANIFEST,
) -> MarketBundle:
    source = _load_manifest(source_manifest_path, EXPECTED_SOURCE_MANIFEST_HASH)
    records = {str(row["symbol"]): row for row in source["records"]}
    market: dict[str, dict[str, np.ndarray]] = {}
    funding: dict[str, pd.DataFrame] = {}
    source_hashes: dict[str, dict[str, str]] = {}
    dates: pd.DatetimeIndex | None = None
    for symbol in sorted(SYMBOLS):
        market_path = Path(source_dir) / f"{symbol}_5m_2024_2025.csv.gz"
        funding_path = Path(source_dir) / f"{symbol}_funding_2024_2025.csv.gz"
        market_hash, funding_hash = _file_hash(market_path), _file_hash(funding_path)
        if market_hash != records[symbol]["output_market_sha256"]:
            raise RuntimeError(f"{symbol} market source changed")
        if funding_hash != records[symbol]["output_funding_sha256"]:
            raise RuntimeError(f"{symbol} funding source changed")
        frame = pd.read_csv(
            market_path,
            usecols=["date", "open", "high", "low", "close", "tic"],
            parse_dates=["date"],
        ).sort_values("date")
        if frame["date"].duplicated().any() or not frame["tic"].astype(str).eq(symbol).all():
            raise RuntimeError(f"{symbol} market identity/grid failure")
        current_dates = pd.DatetimeIndex(frame["date"])
        if dates is None:
            dates = current_dates
        elif not dates.equals(current_dates):
            raise RuntimeError("LORC symbol market grids differ")
        arrays = {
            col: pd.to_numeric(frame[col], errors="raise").to_numpy(dtype=float)
            for col in ("open", "high", "low", "close")
        }
        if not np.isfinite(np.column_stack(list(arrays.values()))).all():
            raise RuntimeError(f"{symbol} non-finite market source")
        market[symbol] = arrays
        fund = pd.read_csv(funding_path)
        fund["event_time"] = pd.to_datetime(pd.to_numeric(fund["funding_time"], errors="raise"), unit="ms")
        fund["funding_rate"] = pd.to_numeric(fund["funding_rate"], errors="raise")
        if fund["event_time"].duplicated().any() or not fund["event_time"].is_monotonic_increasing:
            raise RuntimeError(f"{symbol} funding order failure")
        funding[symbol] = fund[["event_time", "funding_rate"]].copy()
        source_hashes[symbol] = {"market": market_hash, "funding": funding_hash}
    assert dates is not None
    expected = pd.date_range(START, END - pd.Timedelta(minutes=5), freq="5min")
    if not dates.equals(expected):
        raise RuntimeError("LORC execution grid is not exact 2024-2025 prefix")
    return MarketBundle(dates, market, funding, source_hashes)


def load_clock(path: str = DEFAULT_CLOCKS) -> pd.DataFrame:
    if _file_hash(path) != EXPECTED_CLOCK_HASH:
        raise RuntimeError("LORC frozen clock hash changed")
    clock = pd.read_csv(path)
    assert_clock_contract(clock)
    for col in ("signal_time", "feature_available_time", "entry_time", "exit_time"):
        clock[col] = pd.to_datetime(clock[col], errors="raise")
    return clock


def _funding_events_by_bar(bundle: MarketBundle) -> dict[str, dict[int, list[tuple[pd.Timestamp, float, float]]]]:
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


def _event_indices(bundle: MarketBundle, entry: pd.Timestamp, exit_time: pd.Timestamp) -> tuple[int, int]:
    entry_index = int(bundle.dates.searchsorted(entry))
    exit_index = int(bundle.dates.searchsorted(exit_time))
    if entry_index >= len(bundle.dates) or bundle.dates[entry_index] != entry:
        raise RuntimeError(f"missing exact LORC entry open: {entry}")
    if exit_index >= len(bundle.dates) or bundle.dates[exit_index] != exit_time:
        raise RuntimeError(f"missing exact LORC exit open: {exit_time}")
    if exit_index <= entry_index:
        raise RuntimeError("non-positive LORC hold")
    return entry_index, exit_index


def simulate(
    bundle: MarketBundle,
    clock: pd.DataFrame,
    *,
    start: str,
    end: str,
    cost_bp: float = BASE_COST_BP,
) -> dict[str, Any]:
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    if not HOLDOUT_START <= start_ts < end_ts <= END:
        raise ValueError("LORC evaluation escaped calendar 2025")
    if cost_bp < 0:
        raise ValueError("negative execution cost")
    rate = cost_bp / 10_000.0
    selected = clock.loc[(clock["entry_time"] >= start_ts) & (clock["exit_time"] < end_ts)].sort_values("entry_time")
    funding_by_bar = _funding_events_by_bar(bundle)
    equity = peak = close_peak = 1.0
    strict_mdd = close_mdd = 0.0
    total_funding = total_cost = 0.0
    trade_rows: list[dict[str, Any]] = []
    previous_exit: pd.Timestamp | None = None
    for row in selected.itertuples(index=False):
        entry_time, exit_time = pd.Timestamp(row.entry_time), pd.Timestamp(row.exit_time)
        if previous_exit is not None and entry_time < previous_exit:
            raise RuntimeError("LORC execution clock overlaps")
        previous_exit = exit_time
        entry_index, exit_index = _event_indices(bundle, entry_time, exit_time)
        long_symbol, short_symbol = str(row.long_symbol), str(row.short_symbol)
        long_weight, short_weight = float(row.long_weight), float(row.short_weight_abs)
        if not math.isclose(long_weight + short_weight, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError("LORC event gross drift")
        start_equity = equity
        long_entry = float(bundle.market[long_symbol]["open"][entry_index])
        short_entry = float(bundle.market[short_symbol]["open"][entry_index])
        long_qty = long_weight * start_equity / long_entry
        short_qty = short_weight * start_equity / short_entry
        entry_cost = rate * (long_qty * long_entry + short_qty * short_entry)
        total_cost += entry_cost
        cumulative_funding = cumulative_funding_debits_for_peak = 0.0
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
                start_equity - entry_cost + cumulative_funding_debits_for_peak
                + favorable_pnl - favorable_liquidation
            )
            peak = max(peak, favorable_for_peak)
            adverse_equity = start_equity - entry_cost + cumulative_funding + adverse_pnl - adverse_liquidation
            strict_mdd = max(strict_mdd, 1.0 - adverse_equity / peak)
            long_close = float(bundle.market[long_symbol]["close"][bar_index])
            short_close = float(bundle.market[short_symbol]["close"][bar_index])
            close_pnl = long_qty * (long_close - long_entry) + short_qty * (short_entry - short_close)
            close_liquidation = rate * (long_qty * long_close + short_qty * short_close)
            close_equity = start_equity - entry_cost + cumulative_funding + close_pnl - close_liquidation
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
        trade_rows.append({
            "signal_time": str(row.signal_time),
            "entry_time": str(entry_time),
            "exit_time": str(exit_time),
            "long_symbol": long_symbol,
            "short_symbol": short_symbol,
            "net_return": float(trade_return),
            "net_log_return": float(math.log(max(equity, 1e-15) / max(start_equity, 1e-15))),
            "funding_cash": float(cumulative_funding),
        })
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


def _transform_clock(clock: pd.DataFrame, kind: str, seed: int = SIGNFLIP_SEED) -> pd.DataFrame:
    out = clock.copy()
    if kind == "direction_flip":
        out[["long_symbol", "short_symbol"]] = out[["short_symbol", "long_symbol"]].to_numpy()
        out[["long_weight", "short_weight_abs"]] = out[["short_weight_abs", "long_weight"]].to_numpy()
        out[["long_beta", "short_beta"]] = out[["short_beta", "long_beta"]].to_numpy()
    elif kind == "equal_weight":
        out["long_weight"] = out["short_weight_abs"] = 0.5
    elif kind in {"delay_five_minutes", "delay_one_hour"}:
        delta = pd.Timedelta(minutes=5) if kind == "delay_five_minutes" else pd.Timedelta(hours=1)
        for col in ("entry_time", "exit_time"):
            out[col] = pd.to_datetime(out[col]) + delta
    elif kind == "shift_seven_days":
        for col in ("signal_time", "feature_available_time", "entry_time", "exit_time"):
            out[col] = pd.to_datetime(out[col]) + pd.Timedelta(days=7)
    elif kind == "monthly_pair_permutation":
        rng = np.random.default_rng(seed)
        out["signal_time"] = pd.to_datetime(out["signal_time"])
        cols = ["long_symbol", "short_symbol", "long_weight", "short_weight_abs", "long_beta", "short_beta"]
        for _, idx in out.groupby(out["signal_time"].dt.to_period("M")).groups.items():
            labels = np.asarray(list(idx), dtype=int)
            shuffled = labels.copy()
            rng.shuffle(shuffled)
            out.loc[labels, cols] = out.loc[shuffled, cols].to_numpy()
    else:
        raise ValueError(kind)
    return out


def _slim(stats: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in stats.items() if key != "trade_rows"}


def evaluate(bundle: MarketBundle, clock: pd.DataFrame) -> dict[str, Any]:
    base = simulate(bundle, clock, start="2025-01-01", end="2026-01-01")
    h1 = simulate(bundle, clock, start="2025-01-01", end="2025-07-01")
    h2 = simulate(bundle, clock, start="2025-07-01", end="2026-01-01")
    stress = simulate(bundle, clock, start="2025-01-01", end="2026-01-01", cost_bp=STRESS_COST_BP)
    delay = simulate(
        bundle,
        _transform_clock(clock, "delay_five_minutes"),
        start="2025-01-01",
        end="2026-01-01",
    )
    signflip = weekly_cluster_signflip(base["trade_rows"], seed=SIGNFLIP_SEED, samples=SIGNFLIP_SAMPLES)
    diagnostics = {}
    for kind in ("direction_flip", "equal_weight", "delay_one_hour", "shift_seven_days", "monthly_pair_permutation"):
        diagnostics[kind] = _slim(simulate(
            bundle,
            _transform_clock(clock, kind),
            start="2025-01-01",
            end="2026-01-01",
        ))
    gates = {
        "absolute_return_positive": base["absolute_return_pct"] > 0.0,
        "cagr_to_strict_mdd_at_least_3": base["cagr_to_strict_mdd"] >= 3.0,
        "strict_mdd_at_most_15": base["strict_mdd_pct"] <= 15.0,
        "h1_absolute_return_positive": h1["absolute_return_pct"] > 0.0,
        "h2_absolute_return_positive": h2["absolute_return_pct"] > 0.0,
        "trades_at_least_60": base["trades"] >= 60,
        "ten_bp_cost_stress_absolute_return_positive": stress["absolute_return_pct"] > 0.0,
        "weekly_cluster_signflip_p_at_most_0_10": signflip["raw_p_value"] <= 0.10,
        "entry_delay_plus_5m_absolute_return_positive": delay["absolute_return_pct"] > 0.0,
    }
    return {
        "base": _slim(base),
        "h1": _slim(h1),
        "h2": _slim(h2),
        "ten_bp_notional_side_cost_stress": _slim(stress),
        "entry_delay_plus_5m": _slim(delay),
        "weekly_cluster_signflip": signflip,
        "diagnostic_controls": diagnostics,
        "strategy_gates": gates,
        "passes_2025_strategy_gates": all(gates.values()),
    }


def _markdown(result: dict[str, Any]) -> str:
    e = result["evaluation"]
    rows = []
    for name, label in (("base", "2025"), ("h1", "2025 H1"), ("h2", "2025 H2"), ("ten_bp_notional_side_cost_stress", "2025, 10 bp"), ("entry_delay_plus_5m", "2025, +5m entry")):
        s = e[name]
        rows.append(
            f"| {label} | {s['absolute_return_pct']:+.3f}% | {s['cagr_pct']:+.3f}% | "
            f"{s['strict_mdd_pct']:.3f}% | {s['cagr_to_strict_mdd']:.3f} | {s['trades']} |"
        )
    failed = [name for name, passed in e["strategy_gates"].items() if not passed]
    return "\n".join([
        "# LORC v1 calendar-2025 holdout — 2026-07-17",
        "",
        "> This execution opened calendar 2025 only. Calendar 2026 remains sealed.",
        "",
        "| Window | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades |",
        "|---|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "## Decision",
        "",
        f"- status: **{result['decision']}**",
        f"- weekly cluster sign-flip p: `{e['weekly_cluster_signflip']['raw_p_value']:.6f}`",
        f"- failed gates: `{failed}`",
        "- CAGR uses the full declared calendar, including idle periods.",
        "- strict MDD includes global/pre-entry HWM, two-leg favorable-before-adverse OHLC, entry/hypothetical liquidation/exit costs, and exact funding ordering.",
        "- Diagnostics cannot repair a failed single-policy holdout.",
        "",
    ])


def run(output: str = DEFAULT_OUTPUT, docs_output: str = DEFAULT_DOCS) -> dict[str, Any]:
    if canonical_hash(protocol()) != EXPECTED_PROTOCOL_HASH:
        raise RuntimeError("LORC preregistration drifted")
    support = _load_manifest(SUPPORT_MANIFEST, EXPECTED_SUPPORT_MANIFEST_HASH)
    if support.get("clock_sha256") != EXPECTED_CLOCK_HASH or not support["support"].get("passes_support"):
        raise RuntimeError("LORC support freeze is not approved")
    attestation = _git_attestation()
    bundle, clock = load_bundle(), load_clock()
    evaluation = evaluate(bundle, clock)
    passed = evaluation["passes_2025_strategy_gates"]
    result: dict[str, Any] = {
        "protocol_version": "lorc_v1_2025_holdout_2026-07-17",
        "outcomes_opened": True,
        "opened_window": ["2025-01-01", "2026-01-01"],
        "final_2026_opened": False,
        "preregistration_protocol_hash": EXPECTED_PROTOCOL_HASH,
        "source_manifest_hash": EXPECTED_SOURCE_MANIFEST_HASH,
        "support_manifest_hash": EXPECTED_SUPPORT_MANIFEST_HASH,
        "clock_hash": EXPECTED_CLOCK_HASH,
        "single_policy_id": "LORC01",
        "evaluation": evaluation,
        "decision": "passed_2025_strategy_pending_orthogonality" if passed else "rejected_before_2026",
        "pre_outcome_evaluator_attestation": attestation,
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
    e = result["evaluation"]
    print(json.dumps({
        "decision": result["decision"],
        "base": e["base"],
        "h1": e["h1"],
        "h2": e["h2"],
        "stress": e["ten_bp_notional_side_cost_stress"],
        "entry_delay_plus_5m": e["entry_delay_plus_5m"],
        "weekly_cluster_signflip": e["weekly_cluster_signflip"],
        "strategy_gates": e["strategy_gates"],
    }, indent=2))


if __name__ == "__main__":
    main()
