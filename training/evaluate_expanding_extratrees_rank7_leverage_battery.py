#!/usr/bin/env python3
"""Execute the preregistered frozen Rank7 leverage battery exactly once."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import (
    audit_expanding_extratrees_rank7_hardened_strict as hardened,
)
from training import (
    preregister_expanding_extratrees_rank7_leverage_battery as prereg,
)
from training.evaluate_coinm_liquidation_burst_release import (
    stationary_bootstrap_p_value,
)
from training.evaluate_metaorder_fragmentation_impact_curvature import (
    weekly_cluster_sign_flip,
)
from training.evaluate_packet_churn_persistence_pre2024 import strict_equity_stats
from training.search_inventory_purge_reclaim_alpha import (
    ExecutionEngine,
    Trade,
    _schedule_hash,
)
from training.search_stable_ensemble_conditional_pullback_alpha import routed_schedule

OUTPUT = Path(
    "results/expanding_extratrees_rank7_leverage_battery_2026-07-27.json"
)
SUMMARY = Path(
    "docs/expanding-extratrees-rank7-leverage-battery-2026-07-27.md"
)
PREREGISTRATION = prereg.OUTPUT
EXPECTED_PREREGISTRATION_SHA256 = (
    "3fb86f5fc3b22d33451a66a797dec50ca2e0208997a8ee224410df2781fcdc29"
)
EXPECTED_PREREGISTRATION_MANIFEST_HASH = (
    "01fa88ba5e1398c06ea192749c81a15e516982688761c570f160d6e416a16659"
)
SIGNIFICANCE_DRAWS = 50_000
SIGNIFICANCE_SEED = 20_260_727
STRESS_COST_PER_SIDE = 0.0010

WINDOWS = {
    name: (frozen_name, start, end)
    for name, frozen_name, start, end in hardened.AUDIT_WINDOWS
}


def canonical_hash(value: Any) -> str:
    return prereg.canonical_hash(value)


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def validate_execution_state() -> dict[str, str]:
    status = _git_output("status", "--porcelain", "--untracked-files=all")
    if status:
        raise RuntimeError("tracked worktree must be clean before battery execution")
    head = _git_output("rev-parse", "HEAD")
    origin_main = _git_output("rev-parse", "origin/main")
    if head != origin_main:
        raise RuntimeError("battery execution requires HEAD == origin/main")
    return {
        "git_head": head,
        "origin_main": origin_main,
        "runner_sha256": sha256_file(__file__),
    }


def validate_preregistration() -> dict[str, Any]:
    if sha256_file(PREREGISTRATION) != EXPECTED_PREREGISTRATION_SHA256:
        raise RuntimeError("leverage preregistration file drifted")
    payload = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    manifest_hash = payload.pop("manifest_hash")
    if manifest_hash != EXPECTED_PREREGISTRATION_MANIFEST_HASH:
        raise RuntimeError("leverage preregistration manifest drifted")
    if canonical_hash(payload) != manifest_hash:
        raise RuntimeError("leverage preregistration self-hash mismatch")
    if tuple(payload["candidate"]["leverage_grid"]) != prereg.LEVERAGE_GRID:
        raise RuntimeError("leverage grid drifted")
    if tuple(payload["selection"]["windows"]) != prereg.SELECTION_WINDOWS:
        raise RuntimeError("selection boundary drifted")
    if (
        tuple(payload["report_only_evaluation"]["windows"])
        != prereg.REPORT_ONLY_WINDOWS
    ):
        raise RuntimeError("report-only boundary drifted")
    dependencies = payload["integrity"]["frozen_dependencies_sha256"]
    if dependencies != prereg.FROZEN_DEPENDENCIES:
        raise RuntimeError("frozen dependency manifest drifted")
    for path, expected in dependencies.items():
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"frozen dependency drifted: {path}")
    payload["manifest_hash"] = manifest_hash
    return payload


def _require_finite_metrics(row: dict[str, Any], *, label: str) -> None:
    keys = (
        "absolute_return_pct",
        "cagr_pct",
        "strict_mdd_pct",
        "cagr_to_strict_mdd",
        "mean_net_bps",
        "mean_gross_bps",
        "win_rate",
    )
    for key in keys:
        value = float(row[key])
        if not np.isfinite(value):
            raise RuntimeError(f"nonfinite metric: {label}.{key}")
    if int(row["trades"]) < 0:
        raise RuntimeError(f"negative trade count: {label}")


def selection_cell_passes(cell: dict[str, Any]) -> tuple[bool, list[str]]:
    expected = set(prereg.SELECTION_WINDOWS)
    if set(cell["base"]) != expected or set(cell["stress"]) != expected:
        raise ValueError("selection cell contains a non-selection window")
    reasons: list[str] = []
    minimum_trades = {"2023": 12, "2024": 12, "selection": 24}
    for window in prereg.SELECTION_WINDOWS:
        base = cell["base"][window]
        stress = cell["stress"][window]
        _require_finite_metrics(base, label=f"{cell['leverage']}x.base.{window}")
        _require_finite_metrics(stress, label=f"{cell['leverage']}x.stress.{window}")
        if float(base["absolute_return_pct"]) <= 0.0:
            reasons.append(f"{window}:base_nonpositive")
        if float(base["cagr_to_strict_mdd"]) < 3.0:
            reasons.append(f"{window}:base_ratio_lt_3")
        if float(base["strict_mdd_pct"]) > 12.0:
            reasons.append(f"{window}:base_mdd_gt_12")
        if int(base["trades"]) < minimum_trades[window]:
            reasons.append(f"{window}:trades_lt_{minimum_trades[window]}")
        if float(stress["absolute_return_pct"]) <= 0.0:
            reasons.append(f"{window}:stress_nonpositive")
        if float(stress["strict_mdd_pct"]) > 15.0:
            reasons.append(f"{window}:stress_mdd_gt_15")
    return not reasons, reasons


def select_leverage(cells: list[dict[str, Any]]) -> float | None:
    values = tuple(float(cell["leverage"]) for cell in cells)
    if values != prereg.LEVERAGE_GRID:
        raise ValueError("selection cells are not the exact fixed leverage grid")
    passing: list[float] = []
    for cell in cells:
        passed, reasons = selection_cell_passes(cell)
        if bool(cell.get("passes")) != passed:
            raise ValueError("selection pass field does not match fixed gates")
        if list(cell.get("failure_reasons", [])) != reasons:
            raise ValueError("selection failure reasons do not match fixed gates")
        if passed:
            passing.append(float(cell["leverage"]))
    return max(passing) if passing else None


def _trade_returns(
    trades: Iterable[Trade],
    engine: ExecutionEngine,
    *,
    cost_rate: float,
) -> list[float]:
    execution = 1.0 - float(engine.cfg.leverage) * float(cost_rate)
    if not 0.0 < execution <= 1.0:
        raise ValueError("invalid execution factor")
    values = [
        float(execution * trade.price_factor * trade.funding_factor * execution - 1.0)
        for trade in trades
    ]
    if not np.isfinite(values).all():
        raise RuntimeError("nonfinite trade return")
    return values


def _significance(
    trades: list[Trade],
    engine: ExecutionEngine,
) -> dict[str, Any]:
    returns = _trade_returns(
        trades,
        engine,
        cost_rate=float(engine.cfg.fee_rate + engine.cfg.slippage_rate),
    )
    return {
        "weekly_cluster_sign_flip": weekly_cluster_sign_flip(
            returns,
            [trade.entry_date for trade in trades],
            permutations=SIGNIFICANCE_DRAWS,
            seed=SIGNIFICANCE_SEED,
        ),
        "stationary_trade_bootstrap": stationary_bootstrap_p_value(
            np.asarray(returns, dtype=float),
            mean_block_trades=4,
            resamples=SIGNIFICANCE_DRAWS,
            seed=SIGNIFICANCE_SEED,
        ),
    }


def _window_stats(
    base: dict[str, Any],
    active: np.ndarray,
    rank7: dict[str, Any],
    *,
    leverage: float,
    windows: tuple[str, ...],
    include_significance: bool,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, str],
    dict[str, dict[str, Any]],
]:
    cfg = replace(base["engine"].cfg, leverage=float(leverage))
    engine = ExecutionEngine(base["context"]["market"], base["context"]["funding"], cfg)
    base_stats: dict[str, dict[str, Any]] = {}
    stress_stats: dict[str, dict[str, Any]] = {}
    schedule_hashes: dict[str, str] = {}
    significance: dict[str, dict[str, Any]] = {}
    for window in windows:
        frozen_name, start, end = WINDOWS[window]
        trades = routed_schedule(
            base["context"],
            {"engine": engine, "active": active},
            start=start,
            end=end,
        )
        schedule_hash = _schedule_hash(trades)
        expected_hash = rank7["schedule_hashes"][frozen_name]
        if schedule_hash != expected_hash:
            raise RuntimeError(
                f"trade clock drift at {leverage:.2f}x for {window}: "
                f"{schedule_hash} != {expected_hash}"
            )
        hardened_trades = [hardened.harden_trade(trade, engine) for trade in trades]
        base_stats[window] = strict_equity_stats(
            hardened_trades,
            start=start,
            end=end,
            cfg=cfg,
        )
        stress_stats[window] = strict_equity_stats(
            hardened_trades,
            start=start,
            end=end,
            cfg=cfg,
            cost_rate=STRESS_COST_PER_SIDE,
        )
        _require_finite_metrics(
            base_stats[window],
            label=f"{leverage:.2f}x.base.{window}",
        )
        _require_finite_metrics(
            stress_stats[window],
            label=f"{leverage:.2f}x.stress.{window}",
        )
        schedule_hashes[window] = schedule_hash
        if include_significance and window in ("future", "all"):
            significance[window] = _significance(hardened_trades, engine)
    return base_stats, stress_stats, schedule_hashes, significance


def report_only_gates(
    base: dict[str, dict[str, Any]],
    stress: dict[str, dict[str, Any]],
    significance: dict[str, dict[str, Any]],
) -> tuple[bool, list[str]]:
    if set(base) != set(prereg.REPORT_ONLY_WINDOWS):
        raise ValueError("report-only base windows drifted")
    if set(stress) != set(prereg.REPORT_ONLY_WINDOWS):
        raise ValueError("report-only stress windows drifted")
    if set(significance) != {"future", "all"}:
        raise ValueError("report-only significance windows drifted")
    reasons: list[str] = []
    for window, minimum_trades in (("2025", 12), ("2026h1", 6)):
        row = base[window]
        _require_finite_metrics(row, label=f"report.base.{window}")
        if float(row["absolute_return_pct"]) <= 0.0:
            reasons.append(f"{window}:nonpositive")
        if float(row["cagr_to_strict_mdd"]) < 3.0:
            reasons.append(f"{window}:ratio_lt_3")
        if float(row["strict_mdd_pct"]) > 15.0:
            reasons.append(f"{window}:mdd_gt_15")
        if int(row["trades"]) < minimum_trades:
            reasons.append(f"{window}:trades_lt_{minimum_trades}")
    for window, minimum_trades in (("future", 18), ("all", 42)):
        row = base[window]
        _require_finite_metrics(row, label=f"report.base.{window}")
        _require_finite_metrics(stress[window], label=f"report.stress.{window}")
        if float(row["absolute_return_pct"]) <= 0.0:
            reasons.append(f"{window}:nonpositive")
        if float(row["cagr_to_strict_mdd"]) < 3.0:
            reasons.append(f"{window}:ratio_lt_3")
        if float(row["strict_mdd_pct"]) > 15.0:
            reasons.append(f"{window}:mdd_gt_15")
        if int(row["trades"]) < minimum_trades:
            reasons.append(f"{window}:trades_lt_{minimum_trades}")
        if float(stress[window]["absolute_return_pct"]) <= 0.0:
            reasons.append(f"{window}:stress_nonpositive")
        weekly_p = float(
            significance[window]["weekly_cluster_sign_flip"]["p_value_one_sided"]
        )
        bootstrap_p = float(
            significance[window]["stationary_trade_bootstrap"][
                "one_sided_p_value"
            ]
        )
        if not np.isfinite(weekly_p) or weekly_p > 0.10:
            reasons.append(f"{window}:weekly_p_gt_0_10")
        if not np.isfinite(bootstrap_p) or bootstrap_p > 0.10:
            reasons.append(f"{window}:bootstrap_p_gt_0_10")
    return not reasons, reasons


def user_target_hit(base: dict[str, dict[str, Any]]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    for window in ("future", "all"):
        row = base[window]
        if float(row["cagr_pct"]) < 50.0:
            reasons.append(f"{window}:cagr_lt_50")
        if float(row["strict_mdd_pct"]) > 15.0:
            reasons.append(f"{window}:mdd_gt_15")
        if float(row["cagr_to_strict_mdd"]) < 3.0:
            reasons.append(f"{window}:ratio_lt_3")
    return not reasons, reasons


def _selection_grid(
    base: dict[str, Any],
    active: np.ndarray,
    rank7: dict[str, Any],
) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for leverage in prereg.LEVERAGE_GRID:
        stats, stress, schedules, significance = _window_stats(
            base,
            active,
            rank7,
            leverage=float(leverage),
            windows=prereg.SELECTION_WINDOWS,
            include_significance=False,
        )
        if significance:
            raise RuntimeError("future inference leaked into selection")
        cell: dict[str, Any] = {
            "leverage": float(leverage),
            "base": stats,
            "stress": stress,
            "schedule_hashes": schedules,
        }
        passed, reasons = selection_cell_passes(cell)
        cell["passes"] = passed
        cell["failure_reasons"] = reasons
        cells.append(cell)
    return cells


def run() -> dict[str, Any]:
    preregistration = validate_preregistration()
    execution = validate_execution_state()
    frozen, rank7 = hardened.verify_static_dependencies()
    base = hardened.build_base()
    active, fold_meta = hardened._frozen_active(base)
    selected_positions_hash = hardened.sha_obj(
        np.flatnonzero(active).astype(int).tolist()
    )
    if selected_positions_hash != hardened.EXPECTED_SELECTED_POSITIONS_HASH:
        raise RuntimeError("frozen selected-position hash drifted")
    if fold_meta != frozen["folds"]:
        raise RuntimeError("frozen fold metadata drifted")

    selection_grid = _selection_grid(base, active, rank7)
    selected_leverage = select_leverage(selection_grid)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "mode": "frozen_rank7_preregistered_leverage_battery",
        "preregistration": {
            "path": str(PREREGISTRATION),
            "sha256": EXPECTED_PREREGISTRATION_SHA256,
            "manifest_hash": preregistration["manifest_hash"],
        },
        "execution": execution,
        "research_status": preregistration["research_status"],
        "integrity": {
            "selected_positions_hash": selected_positions_hash,
            "selected_positions_match": True,
            "fold_metadata_match": True,
            "selection_uses_only_pre_2025_windows": True,
            "future_repair_or_reselection": False,
        },
        "selection_grid": selection_grid,
        "selected_leverage": selected_leverage,
    }
    if selected_leverage is None:
        payload.update(
            {
                "verdict": "REJECT_SCALING_NO_PRE2025_CELL",
                "report_only": None,
                "robustness_pass": False,
                "robustness_failure_reasons": [
                    "no fixed leverage passed pre-2025 selection"
                ],
                "user_target_hit": False,
                "user_target_failure_reasons": ["no selected leverage"],
            }
        )
    else:
        report_base, report_stress, schedules, significance = _window_stats(
            base,
            active,
            rank7,
            leverage=selected_leverage,
            windows=prereg.REPORT_ONLY_WINDOWS,
            include_significance=True,
        )
        robustness_pass, robustness_reasons = report_only_gates(
            report_base,
            report_stress,
            significance,
        )
        target_hit, target_reasons = user_target_hit(report_base)
        if robustness_pass and target_hit:
            verdict = "TARGET_HIT_PROTOCOL_ISOLATED"
        elif robustness_pass:
            verdict = "ALPHA_SURVIVES_TARGET_NOT_MET"
        else:
            verdict = "REJECT_SCALING_AFTER_REPORT_ONLY"
        payload.update(
            {
                "verdict": verdict,
                "report_only": {
                    "base": report_base,
                    "stress_10bp_per_side": report_stress,
                    "schedule_hashes": schedules,
                    "significance": significance,
                },
                "robustness_pass": robustness_pass,
                "robustness_failure_reasons": robustness_reasons,
                "user_target_hit": target_hit,
                "user_target_failure_reasons": target_reasons,
            }
        )
    payload["result_hash"] = canonical_hash(payload)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    with SUMMARY.open("x", encoding="utf-8") as handle:
        handle.write(render_docs(payload))
    return payload


def _metric(row: dict[str, Any]) -> str:
    return (
        f"abs {row['absolute_return_pct']:.2f}%, CAGR {row['cagr_pct']:.2f}%, "
        f"strict MDD {row['strict_mdd_pct']:.2f}%, "
        f"CAGR/MDD {row['cagr_to_strict_mdd']:.2f}, trades {row['trades']}"
    )


def render_docs(payload: dict[str, Any]) -> str:
    lines = [
        "# ExtraTrees rank-7 preregistered leverage battery — 2026-07-27",
        "",
        f"Verdict: **{payload['verdict']}**",
        "",
        (
            "This is a protocol-isolated sizing audit, not globally pristine "
            "discovery OOS. The leverage was chosen using 2023–2024 only. The "
            "frozen alpha policy, annual refits, features, thresholds, exits, "
            "direction, and trade clocks did not change."
        ),
        "",
        "## Pre-2025 fixed grid",
        "",
        "| Leverage | 2023 | 2024 | Combined | 10bp combined abs | Pass |",
        "| ---: | --- | --- | --- | ---: | :---: |",
    ]
    for cell in payload["selection_grid"]:
        lines.append(
            f"| {cell['leverage']:.2f}x | {_metric(cell['base']['2023'])} | "
            f"{_metric(cell['base']['2024'])} | "
            f"{_metric(cell['base']['selection'])} | "
            f"{cell['stress']['selection']['absolute_return_pct']:.2f}% | "
            f"{'yes' if cell['passes'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            f"Selected leverage: `{payload['selected_leverage']}`",
            "",
        ]
    )
    if payload["report_only"] is not None:
        lines.extend(
            [
                "## Fixed report-only result",
                "",
                "| Period | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | 10bp stress abs |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for window in ("2025", "2026h1", "future", "all"):
            row = payload["report_only"]["base"][window]
            stress = payload["report_only"]["stress_10bp_per_side"][window]
            lines.append(
                f"| {window} | {row['absolute_return_pct']:.2f}% | "
                f"{row['cagr_pct']:.2f}% | {row['strict_mdd_pct']:.2f}% | "
                f"{row['cagr_to_strict_mdd']:.2f} | {row['trades']} | "
                f"{stress['absolute_return_pct']:.2f}% |"
            )
        lines.extend(
            [
                "",
                f"- robustness pass: `{payload['robustness_pass']}`",
                f"- user target hit: `{payload['user_target_hit']}`",
                "- future repair/reselection: `False`",
                "",
            ]
        )
    lines.extend(
        [
            "## Integrity",
            "",
            f"- selected-position hash: `{payload['integrity']['selected_positions_hash']}`",
            "- every leverage cell preserved the exact frozen trade clocks;",
            "- full-calendar CAGR includes idle periods;",
            "- absolute return is reported for every evaluated period;",
            "- hardened strict MDD and conservative funding boundaries are unchanged.",
            "",
            f"Result hash: `{payload['result_hash']}`",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate frozen inputs and clean committed execution state without running models",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validate_only:
        preregistration = validate_preregistration()
        execution = validate_execution_state()
        hardened.verify_static_dependencies()
        print(
            json.dumps(
                {
                    "status": "validated",
                    "preregistration_manifest_hash": preregistration["manifest_hash"],
                    "execution": execution,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    payload = run()
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "summary": str(SUMMARY),
                "verdict": payload["verdict"],
                "selected_leverage": payload["selected_leverage"],
                "robustness_pass": payload["robustness_pass"],
                "user_target_hit": payload["user_target_hit"],
                "report_only": (
                    None
                    if payload["report_only"] is None
                    else payload["report_only"]["base"]
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
