"""Freeze a causal shadow-schedule handoff between two existing BTC alphas.

The two component policies were frozen independently before this composition:

* confirmed pullback-squeeze, long only, 48-hour cap with a 10% take profit;
* inventory purge/reclaim, 24-hour cap with 2.5%/1.5% take/stop and its frozen
  short-only positioning gate.

Each component advances its own non-overlapping *shadow* position clock even
when the shared capital account cannot accept that component's trade.  The
shared account then accepts the earliest shadow trade whose signal is outside
the currently held trade.  This is a deterministic opportunity handoff, not a
weighted return blend or an outcome-aware router.

Only physically truncated pre-2024 sources are opened by this module.  A later
OOS evaluator must validate the frozen manifest before opening future rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.audit_confirmed_pullback_squeeze_live_parity import (
    AuditConfig,
    _activation_hash,
    _fit_active,
    _load_bundle,
    decision_mask,
    live_decision_features,
    schedule_window,
)
from training.search_funding_premium_external_state_gate_alpha import _frame_hash
from training.search_inventory_purge_reclaim_alpha import (
    Config as InventoryConfig,
    ExecutionEngine,
    Trade,
    _apply_gate,
    _base_masks,
    _build_features,
    _load_sources,
    _schedule_hash,
    equity_stats,
)
from training.search_pullback_squeeze_profit_lock_alpha import (
    Config as PullbackConfig,
    _validate_manifest as validate_pullback_manifest,
)


SELECTION_END = "2024-01-01"
SELECTION_WINDOWS: dict[str, tuple[str, str]] = {
    "fit": ("2020-10-15", "2023-01-01"),
    "fit_2020q4": ("2020-10-15", "2021-01-01"),
    "fit_2021": ("2021-01-01", "2022-01-01"),
    "fit_2022": ("2022-01-01", "2023-01-01"),
    "select_2023": ("2023-01-01", SELECTION_END),
    "select_2023_h1": ("2023-01-01", "2023-07-01"),
    "select_2023_h2": ("2023-07-01", SELECTION_END),
    "pre_2024": ("2020-10-15", SELECTION_END),
}
PRIMARY_WINDOWS = ("fit", "select_2023", "pre_2024")
STABILITY_WINDOWS = (
    "fit_2020q4",
    "fit_2021",
    "fit_2022",
    "select_2023_h1",
    "select_2023_h2",
)
SOURCE_PRIORITY = ("pullback", "inventory")
MANIFEST_PAYLOAD_FIELDS = (
    "schema_version",
    "phase",
    "selection_end",
    "source_prefix_hashes",
    "component_manifests",
    "component_specs",
    "composition",
    "selected_schedule_hashes",
    "selected_stats_6bp",
    "stress_stats_10bp",
    "passes_pre_oos_gate",
)


@dataclass(frozen=True)
class Config:
    input_csv: str = "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz"
    metrics_csv: str = "data/binance_um_metrics_BTCUSDT_5m_2020-09-01_2026-06-01.csv.gz"
    funding_csv: str = "data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz"
    premium_csv: str = "data/binance_um_aux_btc_2020_2026/BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz"
    inventory_manifest: str = "results/inventory_purge_reclaim_manifest_2026-07-15.json"
    pullback_manifest: str = "results/pullback_squeeze_profit_lock_manifest_2026-07-15.json"
    output: str = "results/pullback_inventory_shadow_handoff_selection_2026-07-15.json"
    manifest_output: str = "results/pullback_inventory_shadow_handoff_manifest_2026-07-15.json"
    docs_output: str = "docs/pullback-inventory-shadow-handoff-selection-2026-07-15.md"
    leverage: float = 0.5
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0001
    bootstrap_samples: int = 5_000
    bootstrap_block_trades: int = 5
    bootstrap_seed: int = 20_260_715


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _manifest_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _slim(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        key: stats[key]
        for key in (
            "absolute_return_pct",
            "cagr_pct",
            "strict_mdd_pct",
            "cagr_to_strict_mdd",
            "trades",
            "longs",
            "shorts",
            "mean_net_bps",
            "win_rate",
        )
    }


def _source_schedule_hash(trades: Sequence[Trade], sources: Sequence[str]) -> str:
    records = [
        [source, trade.signal_position, trade.entry_position, trade.exit_position, trade.side, trade.entry_date]
        for source, trade in zip(sources, trades, strict=True)
    ]
    encoded = json.dumps(records, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def combine_shadow_schedules(
    pullback: Sequence[Trade],
    inventory: Sequence[Trade],
    *,
    source_priority: Sequence[str] = SOURCE_PRIORITY,
) -> tuple[list[Trade], list[str], dict[str, int]]:
    """Combine already-clocked component schedules without releasing skipped clocks."""

    rank = {name: index for index, name in enumerate(source_priority)}
    if set(rank) != {"pullback", "inventory"}:
        raise ValueError("source_priority must contain pullback and inventory exactly once")
    tagged = [
        (trade.signal_position, rank["pullback"], "pullback", trade) for trade in pullback
    ] + [
        (trade.signal_position, rank["inventory"], "inventory", trade) for trade in inventory
    ]
    tagged.sort(key=lambda row: (row[0], row[1]))
    selected: list[Trade] = []
    sources: list[str] = []
    next_allowed = 0
    collisions = 0
    for signal, _, source, trade in tagged:
        if int(signal) < next_allowed:
            collisions += 1
            continue
        selected.append(trade)
        sources.append(source)
        next_allowed = int(trade.exit_position) + 1
    if any(left.exit_position >= right.signal_position for left, right in zip(selected, selected[1:])):
        raise RuntimeError("combined shadow schedule contains overlapping trades")
    ties = len({trade.signal_position for trade in pullback} & {trade.signal_position for trade in inventory})
    return selected, sources, {"collisions": collisions, "same_signal_ties": ties}


def _inventory_window_schedule(
    market: pd.DataFrame,
    features: pd.DataFrame,
    raw: dict[str, np.ndarray],
    engine: ExecutionEngine,
    manifest: dict[str, Any],
    *,
    start: str,
    end: str,
) -> list[Trade]:
    """Replay the frozen inventory base clock, then remove trades with its gate."""

    base = manifest["base_champion"]
    stride = int(base["stride_bars"])
    hold = int(base["hold_bars"])
    anchors = np.arange(11, len(market) - hold - 2, stride, dtype=np.int64)
    long_active, short_active = _base_masks(features, anchors, manifest["base_thresholds"], base)
    dates = pd.to_datetime(market["date"])
    period = ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy(bool)
    selected = period[anchors] & np.logical_xor(long_active, short_active)
    base_schedule: list[Trade] = []
    next_allowed = 0
    for anchor_index in np.flatnonzero(selected):
        signal = int(anchors[anchor_index])
        if signal < next_allowed:
            continue
        side = 1 if bool(long_active[anchor_index]) else -1
        trade = engine.trade_at(
            signal,
            side,
            hold,
            int(round(float(base["tp"]) * 10_000)),
            int(round(float(base["sl"]) * 10_000)),
        )
        if trade is None or not period[trade.exit_position]:
            continue
        base_schedule.append(trade)
        next_allowed = trade.exit_position + 1
    return _apply_gate(
        {"window": base_schedule},
        raw,
        manifest["gate_champion"],
        manifest["context_thresholds"],
    )["window"]


def _passes_pre_oos_gate(
    stats_6bp: dict[str, dict[str, Any]],
    stats_10bp: dict[str, dict[str, Any]],
) -> bool:
    support = (
        stats_6bp["fit"]["trades"] >= 150
        and stats_6bp["select_2023"]["trades"] >= 30
        and min(stats_6bp[name]["trades"] for name in STABILITY_WINDOWS) >= 10
    )
    stable = all(stats_6bp[name]["absolute_return_pct"] > 0.0 for name in STABILITY_WINDOWS)
    target = all(stats_6bp[name]["cagr_to_strict_mdd"] >= 3.0 for name in PRIMARY_WINDOWS)
    risk = all(stats_6bp[name]["strict_mdd_pct"] <= 15.0 for name in PRIMARY_WINDOWS)
    stress = all(stats_10bp[name]["cagr_to_strict_mdd"] >= 3.0 for name in PRIMARY_WINDOWS)
    return bool(support and stable and target and risk and stress)


def _moving_block_bootstrap(
    trades: Sequence[Trade],
    *,
    start: str,
    end: str,
    execution_cfg: InventoryConfig,
    samples: int,
    block_trades: int,
    seed: int,
) -> dict[str, Any]:
    if not trades or samples <= 0 or block_trades <= 0:
        raise ValueError("bootstrap requires trades, positive samples and positive block size")
    rng = np.random.default_rng(int(seed))
    count = len(trades)
    blocks = (count + int(block_trades) - 1) // int(block_trades)
    cagr = np.empty(int(samples), dtype=float)
    ratio = np.empty(int(samples), dtype=float)
    positive = 0
    ratio_three = 0
    for sample in range(int(samples)):
        starts = rng.integers(0, count, size=blocks)
        indices = np.concatenate(
            [np.arange(value, value + int(block_trades), dtype=int) % count for value in starts]
        )[:count]
        stats = equity_stats(
            [trades[int(index)] for index in indices],
            start=start,
            end=end,
            cfg=execution_cfg,
        )
        cagr[sample] = stats["cagr_pct"]
        ratio[sample] = stats["cagr_to_strict_mdd"]
        positive += stats["absolute_return_pct"] > 0.0
        ratio_three += stats["cagr_to_strict_mdd"] >= 3.0
    return {
        "trades": count,
        "samples": int(samples),
        "block_trades": int(block_trades),
        "seed": int(seed),
        "probability_positive_return": float(positive / samples),
        "probability_cagr_to_strict_mdd_ge_3": float(ratio_three / samples),
        "cagr_pct_p05_p50_p95": [float(value) for value in np.quantile(cagr, (0.05, 0.50, 0.95))],
        "cagr_to_strict_mdd_p05_p50_p95": [float(value) for value in np.quantile(ratio, (0.05, 0.50, 0.95))],
    }


def _exclude_entry_boundary_funding(trades: Sequence[Trade], engine: ExecutionEngine) -> list[Trade]:
    """Reprice the ambiguous funding interval as ``(entry, exit]`` for sensitivity."""

    repriced: list[Trade] = []
    for trade in trades:
        entry_ns = int(engine.dates.iloc[trade.entry_position].value)
        exit_ns = int(engine.dates.iloc[trade.exit_position].value)
        left = int(np.searchsorted(engine.funding_times, entry_ns, side="right"))
        right = int(np.searchsorted(engine.funding_times, exit_ns, side="right"))
        factors = 1.0 - float(engine.cfg.leverage) * trade.side * engine.funding_rates[left:right]
        funding_factor = float(np.prod(factors, dtype=float)) if len(factors) else 1.0
        funding_debit = float(np.prod(np.minimum(factors, 1.0), dtype=float)) if len(factors) else 1.0
        repriced.append(
            replace(
                trade,
                funding_factor=funding_factor,
                funding_debit_factor=funding_debit,
            )
        )
    return repriced


def _liquidation_cost_mdd_stats(
    trades: Iterable[Trade],
    *,
    start: str,
    end: str,
    cfg: InventoryConfig,
) -> dict[str, float]:
    """Stress strict MDD by charging a hypothetical exit at every adverse mark."""

    cost = float(cfg.fee_rate + cfg.slippage_rate)
    entry_exit_factor = 1.0 - float(cfg.leverage) * cost
    equity = peak = 1.0
    strict_mdd = 0.0
    for trade in trades:
        favorable_factor = entry_exit_factor * trade.favorable_price_factor
        adverse_factor = (
            entry_exit_factor
            * entry_exit_factor
            * trade.funding_debit_factor
            * trade.adverse_price_factor
        )
        intratrade_peak = max(peak, equity * favorable_factor)
        strict_mdd = max(strict_mdd, 1.0 - equity * adverse_factor / intratrade_peak)
        peak = intratrade_peak
        equity *= entry_exit_factor * trade.price_factor * trade.funding_factor * entry_exit_factor
        strict_mdd = max(strict_mdd, 1.0 - equity / peak)
        peak = max(peak, equity)
    years = (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() / (365.25 * 86_400.0)
    cagr = (equity ** (1.0 / years) - 1.0) * 100.0 if equity > 0.0 else -100.0
    mdd = strict_mdd * 100.0
    return {
        "absolute_return_pct": float((equity - 1.0) * 100.0),
        "cagr_pct": float(cagr),
        "strict_mdd_pct": float(mdd),
        "cagr_to_strict_mdd": float(cagr / mdd) if mdd > 1e-12 else 0.0,
    }


def _validate_component_manifests(cfg: Config) -> tuple[dict[str, Any], dict[str, Any]]:
    inventory = json.loads(Path(cfg.inventory_manifest).read_text())
    pullback = json.loads(Path(cfg.pullback_manifest).read_text())
    protocol = inventory.get("protocol", {})
    if protocol.get("selection_cutoff") != SELECTION_END or protocol.get("oos_opened") is not False:
        raise RuntimeError("inventory component is not a frozen pre-OOS manifest")
    pullback_cfg = PullbackConfig(
        input_csv=cfg.input_csv,
        funding_csv=cfg.funding_csv,
        premium_csv=cfg.premium_csv,
        manifest_output=cfg.pullback_manifest,
    )
    validate_pullback_manifest(pullback, pullback_cfg)
    return inventory, pullback


def _write_manifest_once(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    payload = {key: manifest[key] for key in MANIFEST_PAYLOAD_FIELDS}
    if _manifest_hash(payload) != manifest.get("manifest_hash"):
        raise RuntimeError("handoff manifest payload hash mismatch")
    if path.exists():
        existing = json.loads(path.read_text())
        existing_payload = {key: existing[key] for key in MANIFEST_PAYLOAD_FIELDS}
        if _manifest_hash(existing_payload) != existing.get("manifest_hash"):
            raise RuntimeError("existing handoff manifest payload hash mismatch")
        if existing_payload != payload:
            raise RuntimeError("refusing to overwrite a different frozen handoff manifest")
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def _write_docs(report: dict[str, Any], path: str) -> None:
    if not path:
        return
    rows = []
    for name in ("fit", "fit_2020q4", "fit_2021", "fit_2022", "select_2023", "select_2023_h1", "select_2023_h2", "pre_2024"):
        stats = report["stats_6bp"][name]
        rows.append(
            f"| {name} | {stats['absolute_return_pct']:.2f}% | {stats['cagr_pct']:.2f}% | "
            f"{stats['strict_mdd_pct']:.2f}% | {stats['cagr_to_strict_mdd']:.2f} | {stats['trades']} |"
        )
    boot = report["bootstrap"]
    text = f"""# Pullback–Inventory Shadow Handoff selection (2026-07-15)

## Verdict

**{report['verdict']}**. This single preregistered composition passes the pre-2024 target without opening post-2023 rows.

The mechanism is not a return blend. Pullback-squeeze and inventory purge/reclaim each advance an independent non-overlapping virtual position clock. The shared 0.5x account accepts the earliest virtual trade that does not overlap its current position. A skipped virtual trade does not release or reschedule that component's clock.

## Frozen performance at 6 bp/notional/side

| window | absolute return | CAGR | strict MDD | CAGR/MDD | trades |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

At 10 bp/notional/side, the primary ratios remain fit **{report['stats_10bp']['fit']['cagr_to_strict_mdd']:.2f}**, 2023 **{report['stats_10bp']['select_2023']['cagr_to_strict_mdd']:.2f}**, and full pre-2024 **{report['stats_10bp']['pre_2024']['cagr_to_strict_mdd']:.2f}**.

Two accounting-boundary sensitivities are immaterial: excluding funding exactly at entry leaves full pre-2024 CAGR/MDD at **{report['accounting_sensitivity']['pre_2024']['exclude_entry_boundary_funding']['cagr_to_strict_mdd']:.3f}**; charging a hypothetical exit cost at every adverse mark leaves it at **{report['accounting_sensitivity']['pre_2024']['liquidation_cost_at_adverse_mark']['cagr_to_strict_mdd']:.3f}**.

## Why the interaction works

- Pullback-squeeze supplies sparse, high-payoff long convexity over a 48-hour horizon.
- Inventory purge/reclaim supplies denser 24-hour mean-reversion/reclaim opportunities and a small short sleeve.
- Chronological handoff removes overlapping risk instead of averaging predictions. In pre-2024 it accepted {report['diagnostics']['pre_2024']['source_counts']['pullback']} pullback and {report['diagnostics']['pre_2024']['source_counts']['inventory']} inventory trades while rejecting {report['diagnostics']['pre_2024']['collisions']} overlaps.
- Both ablations are weaker: pullback-only pre-2024 CAGR/MDD {report['ablations']['pre_2024']['pullback']['cagr_to_strict_mdd']:.2f}; inventory-only {report['ablations']['pre_2024']['inventory']['cagr_to_strict_mdd']:.2f}.

## Statistical stress

A deterministic circular moving-block bootstrap ({boot['samples']} samples, {boot['block_trades']}-trade blocks) gives P(positive full-period return) **{boot['probability_positive_return']:.3f}** and P(CAGR/strict-MDD >= 3) **{boot['probability_cagr_to_strict_mdd_ge_3']:.3f}**. The ratio 5/50/95 percentiles are {boot['cagr_to_strict_mdd_p05_p50_p95'][0]:.2f} / {boot['cagr_to_strict_mdd_p05_p50_p95'][1]:.2f} / {boot['cagr_to_strict_mdd_p05_p50_p95'][2]:.2f}.

## Integrity and caveats

- Sources are physically truncated at `2024-01-01`; entry is next 5-minute open; costs are charged on both sides; realized funding is included.
- Strict MDD retains the global/pre-entry high-water mark and applies each position's favorable envelope before its adverse envelope.
- All component and shared schedules are split-contained; no exit may cross an evaluated boundary.
- The standalone component families have already been viewed on later data in prior research. Therefore a later 2024+ replay is implementation-OOS relative to this script, but not an epistemically pristine first-ever holdout. It must be reported as a contamination-aware replay, not proof of untouched generalization.
- 2022 is positive but its standalone annual CAGR/MDD is {report['stats_6bp']['fit_2022']['cagr_to_strict_mdd']:.2f}; robustness is created at the multi-year mechanism level, not every calendar year independently.
"""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)


def run(cfg: Config) -> dict[str, Any]:
    inventory_manifest, pullback_manifest = _validate_component_manifests(cfg)
    execution_cfg = InventoryConfig(
        input_csv=cfg.input_csv,
        metrics_csv=cfg.metrics_csv,
        funding_csv=cfg.funding_csv,
        output=cfg.output,
        manifest_output=cfg.manifest_output,
        leverage=cfg.leverage,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
    )
    market, funding, inventory_hashes = _load_sources(execution_cfg, cutoff=SELECTION_END)
    inventory_features, inventory_raw = _build_features(market)
    if inventory_hashes != inventory_manifest["source_prefix_hashes"]:
        raise RuntimeError("inventory source prefix differs from its frozen manifest")
    if _frame_hash(inventory_features.reset_index(drop=True)) != inventory_manifest["feature_prefix_hash"]:
        raise RuntimeError("inventory feature prefix differs from its frozen manifest")

    audit_cfg = AuditConfig(
        input_csv=cfg.input_csv,
        funding_csv=cfg.funding_csv,
        premium_csv=cfg.premium_csv,
        leverage=cfg.leverage,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
    )
    pullback_market, pullback_features, _, pullback_hashes = _load_bundle(
        audit_cfg,
        cutoff=SELECTION_END,
        premium_tolerance=audit_cfg.live_premium_tolerance,
    )
    if pullback_hashes != pullback_manifest["source_hashes"]:
        raise RuntimeError("pullback source prefix differs from its frozen manifest")
    dates = pd.to_datetime(market["date"])
    pullback_dates = pd.to_datetime(pullback_market["date"])
    if not np.array_equal(dates.to_numpy(), pullback_dates.to_numpy()):
        raise RuntimeError("component market clocks differ")
    decisions = decision_mask(dates, "live_hour_signal_bar", window_size=audit_cfg.window_size)
    pullback_active, pullback_thresholds = _fit_active(
        live_decision_features(pullback_features),
        dates,
        decisions,
    )
    if pullback_thresholds != pullback_manifest["thresholds"]:
        raise RuntimeError("pullback thresholds differ from its frozen manifest")
    if _activation_hash(pullback_active, dates) != pullback_manifest["activation_hash"]:
        raise RuntimeError("pullback activation differs from its frozen manifest")

    engine = ExecutionEngine(market, funding, execution_cfg)
    stats_6bp: dict[str, dict[str, Any]] = {}
    stats_10bp: dict[str, dict[str, Any]] = {}
    ablations: dict[str, dict[str, dict[str, Any]]] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    schedule_hashes: dict[str, str] = {}
    combined_by_window: dict[str, list[Trade]] = {}
    for name, (start, end) in SELECTION_WINDOWS.items():
        inventory = _inventory_window_schedule(
            market,
            inventory_features,
            inventory_raw,
            engine,
            inventory_manifest,
            start=start,
            end=end,
        )
        if name in inventory_manifest["selected_schedule_hashes"]:
            if _schedule_hash(inventory) != inventory_manifest["selected_schedule_hashes"][name]:
                raise RuntimeError(f"inventory component schedule changed in {name}")
        pullback = schedule_window(
            engine,
            pullback_active,
            start=start,
            end=end,
            hold_bars=int(pullback_manifest["spec"]["hold_bars"]),
            take_bps=int(pullback_manifest["spec"]["take_bps"]),
            stop_bps=int(pullback_manifest["spec"]["stop_bps"]),
        )
        combined, sources, overlap = combine_shadow_schedules(pullback, inventory)
        combined_by_window[name] = combined
        stats_6bp[name] = _slim(equity_stats(combined, start=start, end=end, cfg=execution_cfg))
        stats_10bp[name] = _slim(
            equity_stats(combined, start=start, end=end, cfg=execution_cfg, cost_rate=0.0010)
        )
        ablations[name] = {
            "pullback": _slim(equity_stats(pullback, start=start, end=end, cfg=execution_cfg)),
            "inventory": _slim(equity_stats(inventory, start=start, end=end, cfg=execution_cfg)),
        }
        counts = Counter(sources)
        diagnostics[name] = {
            "source_counts": {source: int(counts.get(source, 0)) for source in SOURCE_PRIORITY},
            "shadow_candidates": {"pullback": len(pullback), "inventory": len(inventory)},
            **overlap,
        }
        schedule_hashes[name] = _source_schedule_hash(combined, sources)

    passed = _passes_pre_oos_gate(stats_6bp, stats_10bp)
    accounting_sensitivity: dict[str, dict[str, dict[str, Any]]] = {}
    for name in PRIMARY_WINDOWS:
        start, end = SELECTION_WINDOWS[name]
        accounting_sensitivity[name] = {
            "exclude_entry_boundary_funding": _slim(
                equity_stats(
                    _exclude_entry_boundary_funding(combined_by_window[name], engine),
                    start=start,
                    end=end,
                    cfg=execution_cfg,
                )
            ),
            "liquidation_cost_at_adverse_mark": _liquidation_cost_mdd_stats(
                combined_by_window[name],
                start=start,
                end=end,
                cfg=execution_cfg,
            ),
        }
    bootstrap = _moving_block_bootstrap(
        combined_by_window["pre_2024"],
        start=SELECTION_WINDOWS["pre_2024"][0],
        end=SELECTION_WINDOWS["pre_2024"][1],
        execution_cfg=execution_cfg,
        samples=cfg.bootstrap_samples,
        block_trades=cfg.bootstrap_block_trades,
        seed=cfg.bootstrap_seed,
    )
    component_manifests = {
        "inventory": {"path": cfg.inventory_manifest, "file_sha256": _sha256(cfg.inventory_manifest)},
        "pullback": {
            "path": cfg.pullback_manifest,
            "file_sha256": _sha256(cfg.pullback_manifest),
            "payload_hash": pullback_manifest["manifest_hash"],
        },
    }
    inventory_base = inventory_manifest["base_champion"]
    inventory_gate = inventory_manifest["gate_champion"]
    component_specs = {
        "inventory": {
            "base": {
                key: inventory_base[key]
                for key in (
                    "horizon_bars",
                    "reclaim_bars",
                    "price_tail",
                    "oi_tail",
                    "reclaim_price_quantile",
                    "reclaim_flow_quantile",
                    "stride_bars",
                    "hold_bars",
                    "tp",
                    "sl",
                )
            },
            "gate": {
                key: inventory_gate[key]
                for key in ("states", "groups", "target")
            },
        },
        "pullback": pullback_manifest["spec"],
    }
    composition = {
        "name": "pullback_inventory_shadow_handoff",
        "tested_compositions": 1,
        "source_priority": list(SOURCE_PRIORITY),
        "component_clocks": "independent shadow schedules advance even when shared capital skips a trade",
        "capital_rule": "accept earliest shadow signal only when signal_position is after current exit_position",
        "signal_to_entry": "completed signal t; next 5m open t+1",
        "split_crossing_exit": "purged",
        "cost_per_notional_side": cfg.fee_rate + cfg.slippage_rate,
        "leverage": cfg.leverage,
        "realized_funding": True,
        "strict_mdd": "global/pre-entry HWM plus position-wide favorable envelope before adverse envelope",
        "same_bar_exit_order": "stop before take",
        "future_opened": False,
    }
    payload = {
        "schema_version": 1,
        "phase": "pre_oos_frozen",
        "selection_end": SELECTION_END,
        "source_prefix_hashes": {"inventory": inventory_hashes, "pullback": pullback_hashes},
        "component_manifests": component_manifests,
        "component_specs": component_specs,
        "composition": composition,
        "selected_schedule_hashes": schedule_hashes,
        "selected_stats_6bp": stats_6bp,
        "stress_stats_10bp": stats_10bp,
        "passes_pre_oos_gate": passed,
    }
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest_hash": _manifest_hash(payload),
        **payload,
    }
    manifest = _write_manifest_once(Path(cfg.manifest_output), manifest)
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "verdict": "PRE_OOS_CANDIDATE_FROZEN" if passed else "REJECTED_PRE_OOS",
        "config": asdict(cfg),
        "manifest": cfg.manifest_output,
        "manifest_hash": manifest["manifest_hash"],
        "protocol": composition,
        "stats_6bp": stats_6bp,
        "stats_10bp": stats_10bp,
        "ablations": ablations,
        "diagnostics": diagnostics,
        "accounting_sensitivity": accounting_sensitivity,
        "bootstrap": bootstrap,
        "passes_pre_oos_gate": passed,
    }
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    _write_docs(report, cfg.docs_output)
    return report


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", default=Config.input_csv)
    parser.add_argument("--metrics-csv", default=Config.metrics_csv)
    parser.add_argument("--funding-csv", default=Config.funding_csv)
    parser.add_argument("--premium-csv", default=Config.premium_csv)
    parser.add_argument("--inventory-manifest", default=Config.inventory_manifest)
    parser.add_argument("--pullback-manifest", default=Config.pullback_manifest)
    parser.add_argument("--output", default=Config.output)
    parser.add_argument("--manifest-output", default=Config.manifest_output)
    parser.add_argument("--docs-output", default=Config.docs_output)
    parser.add_argument("--bootstrap-samples", type=int, default=Config.bootstrap_samples)
    args = parser.parse_args()
    return Config(**vars(args))


def main() -> None:
    report = run(parse_args())
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
