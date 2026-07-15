"""Replay the frozen pullback/inventory shadow handoff on 2024+ data.

This evaluator first reconstructs the physically truncated selection artifact
and validates its externally pinned hash.  Only then may it load future rows.
The component families were inspected on later data before this composition was
created, so the report deliberately calls the result contamination-aware OOS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
from training.freeze_pullback_inventory_shadow_handoff_alpha import (
    MANIFEST_PAYLOAD_FIELDS,
    Config as SelectionConfig,
    _inventory_window_schedule,
    _manifest_hash,
    _slim,
    _source_schedule_hash,
    _validate_component_manifests,
    combine_shadow_schedules,
    run as replay_selection,
)
from training.search_funding_premium_external_state_gate_alpha import _frame_hash
from training.search_inventory_purge_reclaim_alpha import (
    Config as InventoryConfig,
    ExecutionEngine,
    _build_features,
    _load_sources,
    equity_stats,
)


EXPECTED_FROZEN_MANIFEST_HASH = "3d8e92d1302ba9f79a4e7011d3addb0a2bd2d8a4c7dc5deb38dfd8a9f74f6333"
SELECTION_END = "2024-01-01"
FUTURE_WINDOWS: dict[str, tuple[str, str]] = {
    "test_2024": ("2024-01-01", "2025-01-01"),
    "eval_2025": ("2025-01-01", "2026-01-01"),
    "holdout_2026": ("2026-01-01", "2026-06-02"),
    "oos_2024_2026": ("2024-01-01", "2026-06-02"),
}


@dataclass(frozen=True)
class Config:
    input_csv: str = "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz"
    metrics_csv: str = "data/binance_um_metrics_BTCUSDT_5m_2020-09-01_2026-06-01.csv.gz"
    funding_csv: str = "data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz"
    premium_csv: str = "data/binance_um_aux_btc_2020_2026/BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz"
    inventory_manifest: str = "results/inventory_purge_reclaim_manifest_2026-07-15.json"
    pullback_manifest: str = "results/pullback_squeeze_profit_lock_manifest_2026-07-15.json"
    selection_manifest: str = "results/pullback_inventory_shadow_handoff_manifest_2026-07-15.json"
    output: str = "results/pullback_inventory_shadow_handoff_oos_2026-07-15.json"
    docs_output: str = "docs/pullback-inventory-shadow-handoff-oos-2026-07-15.md"
    exclude_from: str = "2026-06-02"
    leverage: float = 0.5
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0001


def _validate_frozen_manifest(cfg: Config) -> dict[str, Any]:
    manifest = json.loads(Path(cfg.selection_manifest).read_text())
    payload = {key: manifest[key] for key in MANIFEST_PAYLOAD_FIELDS}
    actual_hash = _manifest_hash(payload)
    if actual_hash != manifest.get("manifest_hash"):
        raise RuntimeError("selection manifest payload hash mismatch")
    if actual_hash != EXPECTED_FROZEN_MANIFEST_HASH:
        raise RuntimeError("selection manifest differs from the externally pinned hash")
    if manifest.get("phase") != "pre_oos_frozen" or manifest.get("passes_pre_oos_gate") is not True:
        raise RuntimeError("selection manifest was not admitted before OOS")
    composition = manifest.get("composition", {})
    if composition.get("future_opened") is not False:
        raise RuntimeError("selection manifest already claims future rows were opened")
    if float(composition.get("leverage", -1.0)) != cfg.leverage:
        raise RuntimeError("runtime leverage differs from the frozen composition")
    if float(composition.get("cost_per_notional_side", -1.0)) != cfg.fee_rate + cfg.slippage_rate:
        raise RuntimeError("runtime cost differs from the frozen composition")
    for name, path in (("inventory", cfg.inventory_manifest), ("pullback", cfg.pullback_manifest)):
        expected = manifest["component_manifests"][name]["file_sha256"]
        actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"{name} component manifest file changed after composition freeze")
    return manifest


def _passes_oos_gate(stats: dict[str, dict[str, Any]]) -> bool:
    primary = ("test_2024", "eval_2025")
    return bool(
        all(stats[name]["absolute_return_pct"] > 0.0 for name in primary)
        and all(stats[name]["cagr_to_strict_mdd"] >= 3.0 for name in primary)
        and all(stats[name]["strict_mdd_pct"] <= 15.0 for name in (*primary, "oos_2024_2026"))
        and all(stats[name]["trades"] >= 20 for name in primary)
        and stats["oos_2024_2026"]["cagr_to_strict_mdd"] >= 3.0
        and stats["oos_2024_2026"]["trades"] >= 50
    )


def _write_docs(report: dict[str, Any], path: str) -> None:
    if not path:
        return
    rows = []
    for name in FUTURE_WINDOWS:
        stats = report["stats_6bp"][name]
        rows.append(
            f"| {name} | {stats['absolute_return_pct']:.2f}% | {stats['cagr_pct']:.2f}% | "
            f"{stats['strict_mdd_pct']:.2f}% | {stats['cagr_to_strict_mdd']:.2f} | {stats['trades']} |"
        )
    text = f"""# Pullback–Inventory Shadow Handoff OOS replay (2026-07-15)

## Verdict

**{report['verdict']}**.

| window | absolute return | CAGR | strict MDD | CAGR/MDD | trades |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

At 10 bp/notional/side, full 2024–2026H1 CAGR/MDD is **{report['stats_10bp']['oos_2024_2026']['cagr_to_strict_mdd']:.2f}**.

## Interpretation

- The evaluator validated frozen manifest `{report['selection_manifest_hash']}` and reconstructed every pre-2024 schedule hash before loading future rows.
- The result uses next-bar execution, 6 bp/notional/side, realized funding, split-contained exits, and strict favorable-before-adverse MDD.
- This is an implementation-OOS replay for the newly frozen handoff. It is not epistemically pristine because both component families had already been viewed on later periods before the handoff hypothesis was formed.
- A live-grade promotion requires both 2024 and 2025 independently to have positive return, CAGR/strict-MDD >= 3, strict MDD <= 15%, and at least 20 trades, plus full-period ratio >= 3 with at least 50 trades.
"""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)


def run(cfg: Config) -> dict[str, Any]:
    manifest = _validate_frozen_manifest(cfg)
    # Rebuild the exact truncated artifact first. Its write-once manifest check
    # fails before any future loader is called if a source or schedule changed.
    selection_replay = replay_selection(
        SelectionConfig(
            input_csv=cfg.input_csv,
            metrics_csv=cfg.metrics_csv,
            funding_csv=cfg.funding_csv,
            premium_csv=cfg.premium_csv,
            inventory_manifest=cfg.inventory_manifest,
            pullback_manifest=cfg.pullback_manifest,
            output="/tmp/pullback_inventory_shadow_handoff_selection_replay.json",
            manifest_output=cfg.selection_manifest,
            docs_output="",
            leverage=cfg.leverage,
            fee_rate=cfg.fee_rate,
            slippage_rate=cfg.slippage_rate,
            bootstrap_samples=1,
        )
    )
    if selection_replay["manifest_hash"] != EXPECTED_FROZEN_MANIFEST_HASH:
        raise RuntimeError("selection replay did not reproduce the pinned manifest")

    inventory_manifest, pullback_manifest = _validate_component_manifests(
        SelectionConfig(
            input_csv=cfg.input_csv,
            metrics_csv=cfg.metrics_csv,
            funding_csv=cfg.funding_csv,
            premium_csv=cfg.premium_csv,
            inventory_manifest=cfg.inventory_manifest,
            pullback_manifest=cfg.pullback_manifest,
        )
    )
    execution_cfg = InventoryConfig(
        input_csv=cfg.input_csv,
        metrics_csv=cfg.metrics_csv,
        funding_csv=cfg.funding_csv,
        output=cfg.output,
        manifest_output=cfg.selection_manifest,
        exclude_from=cfg.exclude_from,
        leverage=cfg.leverage,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
    )
    market, funding, _ = _load_sources(execution_cfg, cutoff=cfg.exclude_from)
    inventory_features, inventory_raw = _build_features(market)
    dates = pd.to_datetime(market["date"])
    prefix = dates < pd.Timestamp(SELECTION_END)
    if _frame_hash(inventory_features.loc[prefix].reset_index(drop=True)) != inventory_manifest["feature_prefix_hash"]:
        raise RuntimeError("inventory feature prefix changed before OOS replay")

    audit_cfg = AuditConfig(
        input_csv=cfg.input_csv,
        funding_csv=cfg.funding_csv,
        premium_csv=cfg.premium_csv,
        exclude_from=cfg.exclude_from,
        leverage=cfg.leverage,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
    )
    pullback_market, pullback_features, _, _ = _load_bundle(
        audit_cfg,
        cutoff=cfg.exclude_from,
        premium_tolerance=audit_cfg.live_premium_tolerance,
    )
    pullback_dates = pd.to_datetime(pullback_market["date"])
    if not np.array_equal(dates.to_numpy(), pullback_dates.to_numpy()):
        raise RuntimeError("component market clocks differ in OOS replay")
    decisions = decision_mask(dates, "live_hour_signal_bar", window_size=audit_cfg.window_size)
    pullback_active, pullback_thresholds = _fit_active(
        live_decision_features(pullback_features),
        dates,
        decisions,
    )
    if pullback_thresholds != pullback_manifest["thresholds"]:
        raise RuntimeError("pullback fit thresholds changed in OOS replay")
    if _activation_hash(pullback_active[prefix], dates[prefix]) != pullback_manifest["activation_hash"]:
        raise RuntimeError("pullback activation prefix changed before OOS replay")

    engine = ExecutionEngine(market, funding, execution_cfg)
    # Full-prefix replay is an additional guard against composition drift.
    pre_start = "2020-10-15"
    pre_inventory = _inventory_window_schedule(
        market,
        inventory_features,
        inventory_raw,
        engine,
        inventory_manifest,
        start=pre_start,
        end=SELECTION_END,
    )
    pre_pullback = schedule_window(
        engine,
        pullback_active,
        start=pre_start,
        end=SELECTION_END,
        hold_bars=int(pullback_manifest["spec"]["hold_bars"]),
        take_bps=int(pullback_manifest["spec"]["take_bps"]),
        stop_bps=int(pullback_manifest["spec"]["stop_bps"]),
    )
    pre_combined, pre_sources, _ = combine_shadow_schedules(pre_pullback, pre_inventory)
    if _source_schedule_hash(pre_combined, pre_sources) != manifest["selected_schedule_hashes"]["pre_2024"]:
        raise RuntimeError("pre-2024 combined schedule changed before OOS replay")

    stats_6bp: dict[str, dict[str, Any]] = {}
    stats_10bp: dict[str, dict[str, Any]] = {}
    ablations: dict[str, dict[str, dict[str, Any]]] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    for name, (start, end) in FUTURE_WINDOWS.items():
        inventory = _inventory_window_schedule(
            market,
            inventory_features,
            inventory_raw,
            engine,
            inventory_manifest,
            start=start,
            end=end,
        )
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
            "source_counts": {source: int(counts.get(source, 0)) for source in ("pullback", "inventory")},
            "shadow_candidates": {"pullback": len(pullback), "inventory": len(inventory)},
            **overlap,
        }

    passed = _passes_oos_gate(stats_6bp)
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "verdict": "ALPHA_QUALIFIED_CONTAMINATION_AWARE" if passed else "REJECTED_OOS",
        "config": asdict(cfg),
        "selection_manifest": cfg.selection_manifest,
        "selection_manifest_hash": manifest["manifest_hash"],
        "protocol": manifest["composition"] | {
            "future_opened": True,
            "oos_epistemically_pristine": False,
            "reason": "component families were inspected on later rows before composition",
        },
        "stats_6bp": stats_6bp,
        "stats_10bp": stats_10bp,
        "ablations": ablations,
        "diagnostics": diagnostics,
        "passes_oos_gate": passed,
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
    parser.add_argument("--selection-manifest", default=Config.selection_manifest)
    parser.add_argument("--output", default=Config.output)
    parser.add_argument("--docs-output", default=Config.docs_output)
    parser.add_argument("--exclude-from", default=Config.exclude_from)
    return Config(**vars(parser.parse_args()))


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
