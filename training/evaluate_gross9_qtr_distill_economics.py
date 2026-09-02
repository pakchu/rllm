"""Staged fixed-quantity economics evaluator for G9QTR-DISTILL-8.

This module intentionally provides only reusable infrastructure.  It does not
open production/OOS outcomes unless ``run(stage=...)`` is explicitly called and
predecessor gates authorize that stage.  The simulator is a portfolio ledger:
cash plus fixed sleeve quantities, netted aggregate execution costs, exact
funding ownership, strict aggregate OHLC drawdown, and mandatory final exit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from training import evaluate_options_led_volatility_expansion_premium_relay_economics_v5 as staged_sources
from training import evaluate_gross9_async_active_veto_train_economics as train_sources

POLICY_ID = "G9QTR-DISTILL-8"
PROTOCOL_VERSION = "gross9_qtr_distill_fixed_quantity_portfolio_economics_v1"
BAR = pd.Timedelta(minutes=5)
INITIAL_EQUITY = 100_000.0
BASE_COST = 0.0006
STRESS_COST = 0.0010
CLUSTER_DRAWS = 100_000
CLUSTER_SEED = 20260902
TRAIN_LEGACY_BONFERRONI_P_MAX = 0.1 / 72.0
OOS_CLUSTER_P_MAX = 0.1

STAGES: dict[str, tuple[str, str, str]] = {
    "train": ("train", "2023-07-01T00:00:00Z", "2024-01-01T00:00:00Z"),
    "test": ("test", "2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"),
    "eval": ("eval", "2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
    "final": ("final", "2026-01-01T00:00:00Z", "2026-08-01T00:00:00Z"),
}
PREDECESSOR = {"test": "train", "eval": "test", "final": "eval"}
OUTPUTS = {
    stage: Path(f"results/gross9_qtr_distill_{stage}_economics_2026-09-02.json")
    for stage in STAGES
}

CLOCK_PACKAGE = Path("results/gross9_qtr_distill_split_clock_source_support_2026-09-02.json")
MIN_NONZERO_SIGNED_EPISODES = {"test": 12, "eval": 12, "final": 8}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _utc(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _iso_z(timestamp: pd.Timestamp) -> str:
    return _utc(timestamp).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class SleeveSpec:
    name: str
    weight: float
    clock_path: Path | None = None
    clock_sha256: str | None = None


def default_sleeves(clock_package: Path = CLOCK_PACKAGE) -> list[SleeveSpec]:
    if not clock_package.is_file():
        raise RuntimeError(f"{POLICY_ID} missing clock package: {clock_package}")
    package = json.loads(clock_package.read_text(encoding="utf-8"))
    if not isinstance(package, dict):
        raise RuntimeError(f"{POLICY_ID} clock package must be a JSON object")
    core = {key: value for key, value in package.items() if key != "manifest_hash"}
    if package.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError(f"{POLICY_ID} clock package manifest hash drift")
    if package.get("policy_id") != POLICY_ID or package.get("decision") != "materialized_shadow_distilled_clock_package":
        raise RuntimeError(f"{POLICY_ID} clock package identity drift")
    sleeves = []
    for base in package.get("components", {}).get("base_order", []):
        record = package.get("sleeves", {}).get(base)
        if not isinstance(record, Mapping):
            raise RuntimeError(f"{POLICY_ID} missing sleeve package record: {base}")
        clock = record.get("clock", {})
        sleeves.append(
            SleeveSpec(
                name=str(record["sleeve_id"]),
                weight=float(record["weight"]),
                clock_path=Path(str(clock["path"])),
                clock_sha256=str(clock["sha256"]),
            )
        )
    return sleeves


def validate_sleeves(sleeves: Sequence[SleeveSpec]) -> None:
    if len(sleeves) != 4:
        raise RuntimeError(f"{POLICY_ID} requires exactly four fixed sleeves")
    names = [s.name for s in sleeves]
    if len(set(names)) != 4:
        raise RuntimeError(f"{POLICY_ID} sleeve names must be unique")
    if any(not math.isfinite(s.weight) or s.weight <= 0 for s in sleeves):
        raise RuntimeError(f"{POLICY_ID} sleeve weights must be positive finite numbers")


def load_clock(path: str | Path, split: str, start: pd.Timestamp, end: pd.Timestamp, expected_sha256: str | None = None) -> pd.DataFrame:
    clock_path = Path(path)
    if expected_sha256 is not None and sha256_file(clock_path) != expected_sha256:
        raise RuntimeError(f"{POLICY_ID} clock hash drift: {clock_path}")
    frame = staged_sources.load_clock(clock_path, split, start, end).copy()
    frame["entry_time"] = pd.to_datetime(frame["entry_time"], utc=True, errors="raise")
    frame["exit_time"] = pd.to_datetime(frame["exit_time"], utc=True, errors="raise")
    frame["side"] = pd.to_numeric(frame["side"], errors="raise").astype(int)
    if not frame["side"].isin([-1, 1]).all():
        raise RuntimeError(f"{POLICY_ID} clock side must be +/-1: {clock_path}")
    if not ((frame.entry_time >= start) & (frame.exit_time <= end) & (frame.entry_time < frame.exit_time)).all():
        raise RuntimeError(f"{POLICY_ID} clock outside stage window: {clock_path}")
    if frame.sort_values("entry_time").entry_time.tolist() != frame.entry_time.tolist():
        frame = frame.sort_values("entry_time").reset_index(drop=True)
    if (frame.entry_time.diff().dropna() < pd.Timedelta(0)).any():
        raise RuntimeError(f"{POLICY_ID} clock order invalid: {clock_path}")
    return frame[["entry_time", "exit_time", "side"]]


def load_portfolio_clock(sleeves: Sequence[SleeveSpec], split: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    validate_sleeves(sleeves)
    rows: list[dict[str, Any]] = []
    for sleeve in sleeves:
        if sleeve.clock_path is None:
            raise RuntimeError(f"{POLICY_ID} missing clock path for sleeve {sleeve.name}")
        clock = load_clock(sleeve.clock_path, split, start, end, sleeve.clock_sha256)
        for row in clock.itertuples(index=False):
            rows.append(
                {
                    "sleeve": sleeve.name,
                    "weight": sleeve.weight,
                    "entry_time": row.entry_time,
                    "exit_time": row.exit_time,
                    "side": int(row.side),
                }
            )
    return normalize_portfolio_clock(pd.DataFrame(rows))


def normalize_portfolio_clock(clock: pd.DataFrame, require_four_sleeves: bool = True) -> pd.DataFrame:
    required = ["sleeve", "weight", "entry_time", "exit_time", "side"]
    if not set(required).issubset(clock.columns):
        raise RuntimeError(f"{POLICY_ID} portfolio clock schema drift")
    out = clock[required].copy()
    out["sleeve"] = out["sleeve"].astype(str)
    out["weight"] = pd.to_numeric(out["weight"], errors="raise")
    out["entry_time"] = pd.to_datetime(out["entry_time"], utc=True, errors="raise")
    out["exit_time"] = pd.to_datetime(out["exit_time"], utc=True, errors="raise")
    out["side"] = pd.to_numeric(out["side"], errors="raise").astype(int)
    if out.empty:
        if require_four_sleeves:
            raise RuntimeError(f"{POLICY_ID} portfolio clock is empty")
        return out.sort_values(["entry_time", "exit_time", "sleeve"]).reset_index(drop=True)
    if require_four_sleeves and out["sleeve"].nunique() != 4:
        raise RuntimeError(f"{POLICY_ID} portfolio clock must contain exactly four sleeves")
    if not out["side"].isin([-1, 1]).all():
        raise RuntimeError(f"{POLICY_ID} side must be +/-1")
    if (out["weight"] <= 0).any() or not np.isfinite(out["weight"].to_numpy(float)).all():
        raise RuntimeError(f"{POLICY_ID} weights invalid")
    if not (out.entry_time < out.exit_time).all():
        raise RuntimeError(f"{POLICY_ID} non-positive interval")
    for sleeve, group in out.sort_values(["sleeve", "entry_time"]).groupby("sleeve", sort=False):
        if len(group) > 1 and (group.entry_time.iloc[1:].to_numpy() < group.exit_time.iloc[:-1].to_numpy()).any():
            raise RuntimeError(f"{POLICY_ID} overlapping intervals in sleeve {sleeve}")
    return out.sort_values(["entry_time", "exit_time", "sleeve"]).reset_index(drop=True)


def validate_market(market: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> None:
    staged_sources.validate_market(market, start, end)
    if _utc(market.date.iloc[-1]) < end:
        raise RuntimeError(f"{POLICY_ID} market must include final exit open")


def validate_funding(funding: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> None:
    staged_sources.validate_funding(funding, start, end)


def _market_maps(market: pd.DataFrame) -> tuple[pd.DatetimeIndex, dict[pd.Timestamp, int], np.ndarray, np.ndarray, np.ndarray]:
    dates = pd.DatetimeIndex(pd.to_datetime(market.date, utc=True, errors="raise"))
    return dates, {t: i for i, t in enumerate(dates)}, market.open.to_numpy(float), market.high.to_numpy(float), market.low.to_numpy(float)


def _funding_by_bucket(funding: pd.DataFrame) -> dict[pd.Timestamp, list[tuple[float, float]]]:
    buckets: dict[pd.Timestamp, list[tuple[float, float]]] = {}
    for t, rate, mark in zip(funding.date, funding.funding_rate, funding.mark_price):
        bucket = _utc(t).floor("5min")
        buckets.setdefault(bucket, []).append((float(rate), float(mark)))
    return buckets


def simulate_portfolio(
    portfolio_clock: pd.DataFrame,
    market: pd.DataFrame,
    funding: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cost: float,
    initial_equity: float = INITIAL_EQUITY,
) -> dict[str, Any]:
    """Simulate fixed-quantity sleeves with netted aggregate execution costs."""
    start = _utc(start); end = _utc(end)
    clock = normalize_portfolio_clock(portfolio_clock, require_four_sleeves=False)
    dates, positions, opens, highs, lows = _market_maps(market)
    if start not in positions or end not in positions:
        raise RuntimeError(f"{POLICY_ID} start/end absent from exact market opens")
    missing = sorted(set(clock.entry_time).union(set(clock.exit_time)).difference(positions))
    if missing:
        raise RuntimeError(f"{POLICY_ID} clock absent from exact market opens: {missing[0]}")

    entries: dict[pd.Timestamp, list[tuple[int, pd.Series]]] = {}
    exits: dict[pd.Timestamp, list[int]] = {}
    for idx, row in clock.iterrows():
        entries.setdefault(row.entry_time, []).append((int(idx), row))
        exits.setdefault(row.exit_time, []).append(int(idx))

    funding_buckets = _funding_by_bucket(funding)
    cash = float(initial_equity)
    active: dict[int, dict[str, Any]] = {}
    peak = float(initial_equity)
    strict_mdd = 0.0
    total_fees = 0.0
    total_funding = 0.0
    transition_rows: list[dict[str, Any]] = []
    equity_effect_rows: list[dict[str, Any]] = []

    def aggregate_q() -> float:
        return float(sum(item["quantity"] for item in active.values()))

    def mark_equity(q: float, price: float) -> float:
        return float(cash + q * price)

    for i in range(positions[start], positions[end] + 1):
        timestamp = dates[i]
        open_price = float(opens[i])
        q_before = aggregate_q()
        equity_pre = mark_equity(q_before, open_price)
        if equity_pre <= 0:
            raise RuntimeError(f"{POLICY_ID} nonpositive mark equity before transition")

        removed: list[dict[str, Any]] = []
        for idx in exits.get(timestamp, []):
            position = active.pop(idx, None)
            if position is not None:
                removed.append(position)
        added: list[dict[str, Any]] = []
        for idx, row in entries.get(timestamp, []):
            quantity = int(row.side) * float(row.weight) * equity_pre / open_price
            position = {
                "sleeve": str(row.sleeve),
                "side": int(row.side),
                "weight": float(row.weight),
                "quantity": float(quantity),
                "entry_time": row.entry_time,
                "entry_price": open_price,
                "entry_equity": equity_pre,
            }
            active[idx] = position
            added.append(position)

        q_after_trade = aggregate_q()
        delta_q = q_after_trade - q_before
        fee = abs(delta_q) * open_price * cost
        if delta_q != 0.0:
            cash -= delta_q * open_price
        if fee:
            cash -= fee
            total_fees += fee
        equity_after_trade = cash + q_after_trade * open_price
        if removed or added or fee:
            transition_rows.append(
                {
                    "time": timestamp,
                    "open": open_price,
                    "q_before": q_before,
                    "q_after": q_after_trade,
                    "delta_q": delta_q,
                    "fee": fee,
                    "equity_pre": equity_pre,
                    "equity_after_trade": equity_after_trade,
                    "exits": [p["sleeve"] for p in removed],
                    "entries": [p["sleeve"] for p in added],
                }
            )
        if equity_after_trade <= 0:
            raise RuntimeError(f"{POLICY_ID} nonpositive equity after transition")

        funding_cash = 0.0
        for rate, mark in funding_buckets.get(timestamp, []):
            funding_cash += -q_after_trade * mark * rate
        if funding_cash:
            cash += funding_cash
            total_funding += funding_cash
        q_after_funding = aggregate_q()
        base_equity = cash + q_after_funding * open_price
        if base_equity <= 0:
            raise RuntimeError(f"{POLICY_ID} nonpositive equity after funding")

        if i < positions[end]:
            if q_after_funding > 0:
                favorable = float(highs[i]); adverse = float(lows[i])
            elif q_after_funding < 0:
                favorable = float(lows[i]); adverse = float(highs[i])
            else:
                favorable = adverse = open_price
            favorable_equity = cash + q_after_funding * favorable
            adverse_equity = cash + q_after_funding * adverse - abs(q_after_funding) * adverse * cost
            if min(favorable_equity, adverse_equity) <= 0:
                raise RuntimeError(f"{POLICY_ID} nonpositive intrabar equity")
            peak = max(peak, favorable_equity)
            strict_mdd = max(strict_mdd, 1.0 - adverse_equity / peak)

        next_price = float(opens[i + 1]) if i < positions[end] else open_price
        equity_next_open = cash + q_after_funding * next_price
        effect = math.log(equity_next_open / equity_pre) if equity_pre > 0 and equity_next_open > 0 else 0.0
        if effect != 0.0 or funding_cash != 0.0 or fee != 0.0 or delta_q != 0.0:
            equity_effect_rows.append(
                {
                    "time": timestamp,
                    "log_effect": effect,
                    "equity_pre": equity_pre,
                    "equity_next_open": equity_next_open,
                    "funding_cash": funding_cash,
                    "fee": fee,
                }
            )
        realized_equity = cash + q_after_funding * open_price
        strict_mdd = max(strict_mdd, 1.0 - realized_equity / peak)
        peak = max(peak, realized_equity)

    # Mandatory final liquidation at the end open.  This is a safety net for
    # malformed clocks that keep exposure open through the stage boundary.
    final_open = float(opens[positions[end]])
    final_q = aggregate_q()
    final_fee = abs(final_q) * final_open * cost
    if final_q or final_fee:
        cash += final_q * final_open - final_fee
        total_fees += final_fee
        transition_rows.append(
            {
                "time": end,
                "open": final_open,
                "q_before": final_q,
                "q_after": 0.0,
                "delta_q": -final_q,
                "fee": final_fee,
                "equity_pre": cash + final_fee,
                "equity_after_trade": cash,
                "exits": [p["sleeve"] for p in active.values()],
                "entries": [],
                "forced_final_exit": True,
            }
        )
        active.clear()
    final_equity = float(cash)
    if final_equity <= 0:
        raise RuntimeError(f"{POLICY_ID} nonpositive final equity")

    years = (end - start).total_seconds() / (365.25 * 86400.0)
    absolute_return_pct = (final_equity / initial_equity - 1.0) * 100.0
    if years > 0:
        annual_log = math.log(final_equity / initial_equity) / years
        cagr_pct = (math.exp(min(annual_log, 700.0)) - 1.0) * 100.0
    else:
        cagr_pct = 0.0
    mdd_pct = strict_mdd * 100.0
    exposure_weighted_edge_numerator = 0.0
    exposure_weighted_edge_denominator = 0.0
    gross_underlying_edges: list[float] = []
    for row in clock.itertuples(index=False):
        a = positions[row.entry_time]; b = positions[row.exit_time]
        entry = float(opens[a]); exit_price = float(opens[b])
        weight = abs(float(row.weight))
        gross_edge = int(row.side) * (exit_price / entry - 1.0)
        gross_underlying_edges.append(gross_edge)
        exposure_weighted_edge_numerator += weight * gross_edge
        exposure_weighted_edge_denominator += weight

    return {
        "initial_equity": initial_equity,
        "final_equity": final_equity,
        "absolute_return_pct": absolute_return_pct,
        "cagr_pct": cagr_pct,
        "strict_mdd_pct": mdd_pct,
        "cagr_to_strict_mdd": cagr_pct / mdd_pct if mdd_pct > 1e-12 else 0.0,
        "transitions": len(transition_rows),
        "intervals": len(clock),
        "long_intervals": int((clock.side > 0).sum()),
        "short_intervals": int((clock.side < 0).sum()),
        "total_fees": total_fees,
        "total_funding": total_funding,
        "mean_gross_underlying_bp": float(np.mean(gross_underlying_edges) * 1e4) if gross_underlying_edges else 0.0,
        "mean_exposure_weighted_gross_edge_bp": float((exposure_weighted_edge_numerator / exposure_weighted_edge_denominator) * 1e4) if exposure_weighted_edge_denominator else 0.0,
        "transition_rows": transition_rows,
        "equity_effect_rows": equity_effect_rows,
    }


def public_metric(report: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in report.items() if key not in {"transition_rows", "equity_effect_rows"}}


def cluster_signflip(effect_rows: Sequence[Mapping[str, Any]], draws: int = CLUSTER_DRAWS, seed: int = CLUSTER_SEED) -> dict[str, Any]:
    clusters: dict[tuple[int, int], float] = {}
    for row in effect_rows:
        effect = float(row.get("log_effect", 0.0))
        if effect == 0.0:
            continue
        iso = _utc(row["time"]).isocalendar()
        key = (int(iso.year), int(iso.week))
        clusters[key] = clusters.get(key, 0.0) + effect
    vals = np.array(list(clusters.values()), dtype=float)
    if vals.size == 0:
        return {
            "method": "one_sided_UTC_week_cluster_signflip_monte_carlo",
            "clusters": 0,
            "draws": draws,
            "seed": seed,
            "observed_log_effect": 0.0,
            "pvalue": 1.0,
        }
    observed = float(vals.sum())
    rng = np.random.default_rng(seed)
    null = rng.choice(np.array([-1.0, 1.0]), size=(draws, vals.size)) @ vals
    pvalue = (1 + int((null >= observed).sum())) / (draws + 1)
    return {
        "method": "one_sided_UTC_week_cluster_signflip_monte_carlo",
        "clusters": int(vals.size),
        "draws": draws,
        "seed": seed,
        "observed_log_effect": observed,
        "pvalue": float(pvalue),
    }


def evaluate_primary(clock: pd.DataFrame, market: pd.DataFrame, funding: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
    base = simulate_portfolio(clock, market, funding, start, end, BASE_COST)
    stress = simulate_portfolio(clock, market, funding, start, end, STRESS_COST)
    midpoint = start + (end - start) / 2
    halves = {
        name: public_metric(simulate_portfolio(clock[(clock.entry_time >= a) & (clock.exit_time <= b)], market, funding, a, b, BASE_COST))
        for name, a, b in (("first", start, midpoint), ("second", midpoint, end))
    }
    return {
        "base": public_metric(base),
        "stress": public_metric(stress),
        "cluster_signflip": cluster_signflip(base["equity_effect_rows"]),
        "calendar_halves": halves,
    }


def stage_checks(stage: str, primary: Mapping[str, Any]) -> dict[str, bool]:
    base = primary["base"]; stress = primary["stress"]
    checks = {
        "absolute_return_positive": float(base["absolute_return_pct"]) > 0.0,
        "cagr_to_strict_mdd_min_3": float(base["cagr_to_strict_mdd"]) >= 3.0,
        "strict_mdd_max_15": float(base["strict_mdd_pct"]) <= 15.0,
        "mean_exposure_weighted_gross_edge_min_20bp": float(base.get("mean_exposure_weighted_gross_edge_bp", base.get("mean_gross_underlying_bp", 0.0))) >= 20.0,
        "stress_absolute_return_positive": float(stress["absolute_return_pct"]) > 0.0,
        "stress_cagr_to_strict_mdd_min_2_5": float(stress["cagr_to_strict_mdd"]) >= 2.5,
        "each_calendar_half_positive": all(float(item["absolute_return_pct"]) > 0.0 for item in primary["calendar_halves"].values()),
    }
    if stage != "train":
        checks["oos_cluster_signflip_p_max_0_1"] = float(primary["cluster_signflip"]["pvalue"]) <= OOS_CLUSTER_P_MAX
        nonzero_signed_episodes = int(base.get("intervals", primary.get("nonzero_signed_episodes", 0)))
        checks["source_min_nonzero_signed_episodes"] = nonzero_signed_episodes >= MIN_NONZERO_SIGNED_EPISODES[stage]
    return checks


def verify_predecessor(stage: str, outputs: Mapping[str, Path] = OUTPUTS) -> dict[str, Any] | None:
    if stage == "train":
        return None
    predecessor_stage = PREDECESSOR[stage]
    predecessor_path = Path(outputs[predecessor_stage])
    if not predecessor_path.is_file():
        raise RuntimeError(f"{POLICY_ID} missing predecessor {predecessor_stage}: {predecessor_path}")
    report = json.loads(predecessor_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise RuntimeError(f"{POLICY_ID} predecessor is not a JSON object")
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    if report.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError(f"{POLICY_ID} predecessor manifest hash drift")
    if report.get("policy_id") != POLICY_ID or report.get("stage") != predecessor_stage:
        raise RuntimeError(f"{POLICY_ID} predecessor identity drift")
    if report.get("passed") is not True or report.get("advance_to_next_stage") is not True:
        raise RuntimeError(f"{POLICY_ID} predecessor did not pass")
    return {"stage": predecessor_stage, "path": str(predecessor_path), "sha256": sha256_file(predecessor_path), "manifest_hash": report["manifest_hash"]}


def load_sources(stage: str, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if stage in {"train", "test", "eval"}:
        market = train_sources.load_market_hash_bound(start, end)
        if stage == "train":
            funding = train_sources.load_train_funding_hash_bound(start, end)
            source = {
                "mode": "hash_bound_gzip_physical_prefix",
                "market_sha256": staged_sources.v1.MARKET_SHA,
                "funding_marks_sha256": staged_sources.TRAIN_FUNDING_SHA,
            }
        else:
            funding = staged_sources.load_postgres_funding(start, end)
            source = {
                "mode": "hash_bound_gzip_market_plus_postgres_exact_funding",
                "market_sha256": staged_sources.v1.MARKET_SHA,
                "funding_table": "funding_rates_binance",
                "symbol": "BTCUSDT",
            }
        return market, funding, source
    return staged_sources.load_sources(stage, start, end)


def run(stage: str, output: str | Path | None = None, sleeves: Sequence[SleeveSpec] | None = None, outputs: Mapping[str, Path] = OUTPUTS) -> dict[str, Any]:
    if stage not in STAGES:
        raise RuntimeError(f"{POLICY_ID} unknown stage: {stage}")
    predecessor = verify_predecessor(stage, outputs)
    split, start_s, end_s = STAGES[stage]
    start = _utc(start_s); end = _utc(end_s)
    portfolio_clock = load_portfolio_clock(sleeves or default_sleeves(), split, start, end)
    market, funding, source = load_sources(stage, start, end)
    validate_market(market, start, end)
    validate_funding(funding, start, end)
    primary = evaluate_primary(portfolio_clock, market, funding, start, end)
    checks = stage_checks(stage, primary)
    passed = all(checks.values())
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": POLICY_ID,
        "stage": stage,
        "window": [_iso_z(start), _iso_z(end)],
        "predecessor": predecessor,
        "source": source,
        "accounting": {
            "ledger": "cash plus fixed sleeve quantities; aggregate q delta netted for execution cost",
            "entry_quantity": "q=side*weight*pre_transition_portfolio_equity/open for simultaneous entries",
            "transition_order": "mark at open, remove exits, add simultaneous entries from same pre-transition equity, charge abs(net_delta_q)*open*cost",
            "funding": "post-transition aggregate q receives funding cash=-aggregate_q*settlement_mark*rate for entry<=funding<exit",
            "strict_mdd": "global HWM; every held 5m favorable then adverse OHLC on aggregate net q with virtual adverse liquidation cost",
            "final_exit": "mandatory liquidation at stage end open",
        },
        "costs": {"base_each_notional_side_bp": 6, "stress_each_notional_side_bp": 10},
        "fixed_sleeves": [{"name": s.name, "weight": s.weight, "clock_path": str(s.clock_path) if s.clock_path else None, "clock_sha256": s.clock_sha256} for s in (sleeves or default_sleeves())],
        "physical_rows_opened": {"market": len(market), "funding": len(funding), "portfolio_clock": len(portfolio_clock)},
        "later_stage_outcomes_opened": False,
        "primary": primary,
        "checks": checks,
        "train_legacy_cluster_diagnostic": {
            "reported_not_pass_authorizing": stage == "train",
            "legacy_p_max_0_1_over_72": TRAIN_LEGACY_BONFERRONI_P_MAX,
            "observed_pvalue": primary["cluster_signflip"]["pvalue"],
            "would_pass_legacy_gate": primary["cluster_signflip"]["pvalue"] <= TRAIN_LEGACY_BONFERRONI_P_MAX,
        } if stage == "train" else None,
        "passed": passed,
        "advance_to_next_stage": passed and stage != "final",
        "decision": "pass" if passed else "terminal_reject_no_repair",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    destination = Path(output) if output is not None else Path(outputs[stage])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(STAGES), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        predecessor = verify_predecessor(args.stage)
        print(json.dumps({"stage": args.stage, "verified": True, "predecessor": predecessor, "outcomes_opened": False}, ensure_ascii=False))
        return
    result = run(args.stage, args.output)
    print(json.dumps({"stage": args.stage, "passed": result["passed"], "output": str(args.output or OUTPUTS[args.stage])}, ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover
    main()
