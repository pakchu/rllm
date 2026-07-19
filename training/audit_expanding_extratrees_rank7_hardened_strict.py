#!/usr/bin/env python3
"""Re-audit frozen ExtraTrees rank-7 with the hardened strict metric.

This module is deliberately not a search.  It independently reconstructs the
frozen five-seed, 300-tree annual policy, requires exact activation and trade
clock parity with the historical artifacts, and only then changes two
accounting details:

* an adverse held-price mark includes a virtual liquidation cost; and
* funding credits exactly on entry/exit timestamps are excluded while debits
  are retained.

No feature, label, model, threshold, refit cadence, side, exit, or schedule may
change in this audit.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.audit_expanding_extratrees_rank7_stability import (
    FOLDS,
    SEEDS,
    SPEC,
    build_base,
    fit_seed_predict,
    sha_obj,
    validate_frozen_spec,
)
from training.evaluate_coinm_liquidation_burst_release import (
    stationary_bootstrap_p_value,
)
from training.evaluate_metaorder_fragmentation_impact_curvature import (
    weekly_cluster_sign_flip,
)
from training.evaluate_packet_churn_persistence_pre2024 import (
    _apply_conservative_boundary_funding,
    strict_equity_stats,
)
from training.search_inventory_purge_reclaim_alpha import (
    ExecutionEngine,
    Trade,
    _schedule_hash,
)
from training.search_stable_ensemble_conditional_pullback_alpha import (
    PULLBACK_FEATURE,
    WIDTH_FEATURE,
    FEATURE_COLUMNS,
    routed_schedule,
    source_thresholds,
)


OUTPUT = Path(
    "results/expanding_extratrees_rank7_hardened_strict_audit_2026-07-19.json"
)
SUMMARY = Path("docs/expanding-extratrees-rank7-hardened-strict-audit-2026-07-19.md")
CONTRACT = Path(
    "docs/expanding-extratrees-rank7-hardened-strict-contract-2026-07-19.md"
)
FROZEN_STABILITY = Path("results/expanding_extratrees_rank7_stability_2026-07-15.json")
FROZEN_OOS = Path("results/expanding_extratrees_top10_oos_2026-07-15.json")
FROZEN_MANIFEST = Path(
    "results/expanding_extratrees_top10_pre2025_manifest_2026-07-15.json"
)

EXPECTED_SHA256 = {
    str(FROZEN_STABILITY): (
        "ffdde5450ef73e0f6099a49ff75a6e16a272dc0c55eb68857ed0f6e168a8ccb2"
    ),
    str(FROZEN_OOS): (
        "41a06a4f492a8625fc3023dd8a7e03a0e229182b7325fc9deb13c100cf381c15"
    ),
    str(FROZEN_MANIFEST): (
        "5901c4cdd66f7894e4712c1d3d5f490af39b8557170968442c84981d34b90976"
    ),
    "training/audit_expanding_extratrees_rank7_stability.py": (
        "a5280fda3e9ac6267b82914b3e1a094f1bf5716656554d4e0e5ca32306ae0540"
    ),
    "training/evaluate_packet_churn_persistence_pre2024.py": (
        "a3d8d56ed9cc6a49603206253bef3d19b42854f9bb58b3aca76b803b2f7258e2"
    ),
}

EXPECTED_SELECTED_POSITIONS_HASH = (
    "8ffbd55f07ceda0e82c270fe4b370fffba44bb3fcfc807368c4385d2ba97f531"
)
EXPECTED_MANIFEST_HASH = (
    "c6e7d78a328118456eacf70bc42cb12a48f33e26d13edbe21f2edb3aedea4f8e"
)
TREES = 300
COST_STRESS_PER_SIDE = 0.0010
SIGNIFICANCE_DRAWS = 50_000
SIGNIFICANCE_SEED = 20_260_719

AUDIT_WINDOWS = (
    ("2023", "test_2023", "2023-01-01", "2024-01-01"),
    ("2024", "validation_2024", "2024-01-01", "2025-01-01"),
    ("2025", "eval_2025", "2025-01-01", "2026-01-01"),
    ("2026h1", "holdout_2026h1", "2026-01-01", "2026-06-02"),
    ("selection", "selection_2023_2024", "2023-01-01", "2025-01-01"),
    ("future", "future_2025_2026h1", "2025-01-01", "2026-06-02"),
    ("all", "all_2023_2026h1", "2023-01-01", "2026-06-02"),
)


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_static_dependencies() -> tuple[dict[str, Any], dict[str, Any]]:
    for raw_path, expected in EXPECTED_SHA256.items():
        actual = sha256_file(raw_path)
        if actual != expected:
            raise RuntimeError(f"frozen dependency changed: {raw_path}")
    manifest_hash = validate_frozen_spec()
    if manifest_hash != EXPECTED_MANIFEST_HASH:
        raise RuntimeError("frozen rank-7 manifest hash changed")
    stability = json.loads(FROZEN_STABILITY.read_text(encoding="utf-8"))
    frozen = stability["ensembles"][str(TREES)]
    if frozen["selected_positions_hash"] != EXPECTED_SELECTED_POSITIONS_HASH:
        raise RuntimeError("frozen selected-position hash changed")
    oos = json.loads(FROZEN_OOS.read_text(encoding="utf-8"))
    rank7 = next(row for row in oos["candidates"] if row["rank_position"] == 7)
    return frozen, rank7


def _frozen_active(base: dict[str, Any]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Independently rebuild the frozen rank-7 annual activation."""

    context = base["context"]
    signals = base["signals"]
    funding = base["funding"]
    y = base["y"]
    signal_dates = base["signal_dates"]
    end_dates = base["end_dates"]
    width = np.asarray(
        context["matrix"][:, FEATURE_COLUMNS.index(WIDTH_FEATURE)], dtype=float
    )
    pullback = np.asarray(
        context["matrix"][:, FEATURE_COLUMNS.index(PULLBACK_FEATURE)], dtype=float
    )
    active = np.zeros(len(context["market"]), dtype=bool)
    fold_meta: list[dict[str, Any]] = []

    for name, start, end in FOLDS:
        cutoff = pd.Timestamp(start)
        fit = np.asarray(
            (signal_dates >= pd.Timestamp("2020-07-01"))
            & (signal_dates < cutoff)
            & np.isfinite(y).all(axis=1)
            & (end_dates < cutoff.to_datetime64()),
            dtype=bool,
        )
        predict = np.asarray(
            (signal_dates >= cutoff) & (signal_dates < pd.Timestamp(end)),
            dtype=bool,
        )
        train_predictions: list[np.ndarray] = []
        period_predictions: list[np.ndarray] = []
        for seed in SEEDS:
            train, period = fit_seed_predict(
                base,
                fit,
                predict,
                seed=int(seed),
                trees=TREES,
            )
            train_predictions.append(train)
            period_predictions.append(period)
        train_prediction = np.mean(np.stack(train_predictions), axis=0)
        period_prediction = np.mean(np.stack(period_predictions), axis=0)

        train_score = (
            train_prediction[:, 0] - float(SPEC["lambda"]) * train_prediction[:, 1]
        )
        period_score = (
            period_prediction[:, 0] - float(SPEC["lambda"]) * period_prediction[:, 1]
        )
        fit_source = funding[fit]
        predict_source = funding[predict]
        funding_threshold, premium_threshold = source_thresholds(
            train_score,
            fit_source,
            funding_q=float(SPEC["funding_q"]),
            premium_q=float(SPEC["premium_q"]),
        )
        risk = train_prediction[:, 1]
        funding_risk_cap = float(np.quantile(risk[fit_source], float(SPEC["risk_q"])))
        premium_risk_cap = float(np.quantile(risk[~fit_source], float(SPEC["risk_q"])))
        positions = signals[predict]
        width_q20 = float(np.quantile(width[signals[fit]][fit_source], 0.2))
        pullback_q40 = float(np.quantile(pullback[signals[fit]][fit_source], 0.4))
        weak_interaction_gate = (width[positions] > width_q20) | (
            pullback[positions] <= pullback_q40
        )
        selected = (
            predict_source
            & (period_score >= funding_threshold)
            & (period_prediction[:, 1] <= funding_risk_cap)
            & weak_interaction_gate
        ) | (
            (~predict_source)
            & (period_score >= premium_threshold)
            & (period_prediction[:, 1] <= premium_risk_cap)
        )
        active[positions] = selected
        fold_meta.append(
            {
                "name": name,
                "fit_examples": int(fit.sum()),
                "predict_events": int(predict.sum()),
                "selected_events": int(selected.sum()),
                "funding_score": float(funding_threshold),
                "premium_score": float(premium_threshold),
                "funding_risk_cap": funding_risk_cap,
                "premium_risk_cap": premium_risk_cap,
                "width_q20": width_q20,
                "pullback_q40": pullback_q40,
                "prediction_n_jobs_forced": 1,
            }
        )
    return active, fold_meta


def _net_trade_returns(
    trades: list[Trade], engine: ExecutionEngine, *, cost_rate: float
) -> list[float]:
    execution = 1.0 - float(engine.cfg.leverage) * float(cost_rate)
    return [
        float(execution * trade.price_factor * trade.funding_factor * execution - 1.0)
        for trade in trades
    ]


def harden_trade(trade: Trade, engine: ExecutionEngine) -> Trade:
    return _apply_conservative_boundary_funding(trade, engine)


def _significance(trades: list[Trade], engine: ExecutionEngine) -> dict[str, Any]:
    returns = _net_trade_returns(
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


def hardened_pass(
    stats: dict[str, dict[str, Any]],
    stress: dict[str, dict[str, Any]],
    significance: dict[str, dict[str, Any]],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    for name in ("2023", "2024", "2025", "2026h1"):
        row = stats[name]
        minimum_trades = 6 if name == "2026h1" else 12
        if row["absolute_return_pct"] <= 0.0:
            reasons.append(f"{name}:nonpositive_return")
        if row["cagr_to_strict_mdd"] < 3.0:
            reasons.append(f"{name}:ratio_lt_3")
        if row["strict_mdd_pct"] > 15.0:
            reasons.append(f"{name}:mdd_gt_15")
        if row["trades"] < minimum_trades:
            reasons.append(f"{name}:trades_lt_{minimum_trades}")
    for name, minimum_trades in (("future", 18), ("all", 42)):
        row = stats[name]
        if row["cagr_to_strict_mdd"] < 3.0:
            reasons.append(f"{name}:ratio_lt_3")
        if row["strict_mdd_pct"] > 15.0:
            reasons.append(f"{name}:mdd_gt_15")
        if row["trades"] < minimum_trades:
            reasons.append(f"{name}:trades_lt_{minimum_trades}")
        if stress[name]["absolute_return_pct"] <= 0.0:
            reasons.append(f"{name}:10bp_stress_nonpositive")
        sig = significance[name]
        if sig["weekly_cluster_sign_flip"]["p_value_one_sided"] > 0.10:
            reasons.append(f"{name}:weekly_cluster_p_gt_0_10")
        if sig["stationary_trade_bootstrap"]["one_sided_p_value"] > 0.10:
            reasons.append(f"{name}:bootstrap_p_gt_0_10")
    return not reasons, reasons


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def render_docs(payload: dict[str, Any]) -> str:
    lines = [
        "# ExtraTrees rank-7 hardened strict audit — 2026-07-19",
        "",
        f"Verdict: **{payload['verdict']}**",
        "",
        "This is a retrospective accounting/parity audit, not pristine discovery OOS. "
        "The frozen model, thresholds, annual refits, selected positions, and trade clocks "
        "were required to match before hardened metrics were accepted.",
        "",
        "## Metrics",
        "",
        "| Period | Abs return | CAGR | Hardened strict MDD | CAGR/MDD | Trades | 10bp/side stress abs |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("2023", "2024", "2025", "2026h1", "future", "all"):
        row = payload["hardened_stats"][name]
        stressed = payload["ten_bp_per_side_stress"][name]
        lines.append(
            f"| {name} | {row['absolute_return_pct']:.4f}% | {row['cagr_pct']:.4f}% | "
            f"{row['strict_mdd_pct']:.4f}% | {row['cagr_to_strict_mdd']:.4f} | "
            f"{row['trades']} | {stressed['absolute_return_pct']:.4f}% |"
        )
    lines.extend(["", "## Statistical checks", ""])
    for name in ("future", "all"):
        sig = payload["significance"][name]
        lines.append(
            f"- `{name}`: weekly-cluster sign-flip p "
            f"`{sig['weekly_cluster_sign_flip']['p_value_one_sided']:.6f}`; "
            f"stationary trade-bootstrap p "
            f"`{sig['stationary_trade_bootstrap']['one_sided_p_value']:.6f}`."
        )
    lines.extend(
        [
            "",
            "## Integrity",
            "",
            f"- selected-position hash: `{payload['integrity']['selected_positions_hash']}`",
            "- every frozen per-window trade-clock hash matched: "
            f"`{payload['integrity']['all_schedule_hashes_match']}`",
            "- model/feature/policy changes: none",
            "- exact funding boundary: interior symmetric; entry/exit credits dropped, debits retained",
            "- adverse mark: actual entry cost plus virtual liquidation cost",
            "",
        ]
    )
    if payload["failure_reasons"]:
        lines.append("Failure reasons: " + ", ".join(payload["failure_reasons"]))
        lines.append("")
    return "\n".join(lines)


def run() -> dict[str, Any]:
    frozen, rank7 = verify_static_dependencies()
    base = build_base()
    active, folds = _frozen_active(base)
    selected_positions_hash = sha_obj(np.flatnonzero(active).astype(int).tolist())
    if selected_positions_hash != EXPECTED_SELECTED_POSITIONS_HASH:
        raise RuntimeError(
            "independent rank-7 activation does not match frozen positions"
        )
    if folds != frozen["folds"]:
        raise RuntimeError(
            "independent rank-7 fold replay does not match frozen metadata"
        )

    engine = base["engine"]
    hardened_stats: dict[str, dict[str, Any]] = {}
    stressed_stats: dict[str, dict[str, Any]] = {}
    significance: dict[str, dict[str, Any]] = {}
    schedule_hashes: dict[str, str] = {}
    expected_schedule_hashes: dict[str, str] = {}
    legacy_stats: dict[str, dict[str, Any]] = {}

    for name, frozen_name, start, end in AUDIT_WINDOWS:
        trades = routed_schedule(
            base["context"],
            {"engine": engine, "active": active},
            start=start,
            end=end,
        )
        schedule_hashes[name] = _schedule_hash(trades)
        expected_schedule_hashes[name] = rank7["schedule_hashes"][frozen_name]
        if schedule_hashes[name] != expected_schedule_hashes[name]:
            raise RuntimeError(f"frozen trade clock changed for {name}")
        hardened = [harden_trade(trade, engine) for trade in trades]
        hardened_stats[name] = strict_equity_stats(
            hardened,
            start=start,
            end=end,
            cfg=engine.cfg,
        )
        stressed_stats[name] = strict_equity_stats(
            hardened,
            start=start,
            end=end,
            cfg=engine.cfg,
            cost_rate=COST_STRESS_PER_SIDE,
        )
        legacy_stats[name] = rank7["stats"][frozen_name]
        if name in ("future", "all"):
            significance[name] = _significance(hardened, engine)

    passed, reasons = hardened_pass(hardened_stats, stressed_stats, significance)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "mode": "frozen_rank7_hardened_strict_accounting_replay",
        "verdict": (
            "SURVIVES_HARDENED_STRICT_AUDIT"
            if passed
            else "DOWNGRADE_AFTER_HARDENED_STRICT_AUDIT"
        ),
        "pass": passed,
        "failure_reasons": reasons,
        "research_status": {
            "pristine_discovery_oos": False,
            "reason": "2025+ outcomes had already been opened in the frozen historical study",
            "allowed_change": "accounting strictness only",
            "forbidden_changes": [
                "features",
                "labels",
                "learner",
                "seeds",
                "tree_count",
                "thresholds",
                "refit_cadence",
                "side",
                "exit_rules",
                "trade_clocks",
            ],
        },
        "audit_commit": _git_head(),
        "contract": str(CONTRACT),
        "frozen_manifest_hash": EXPECTED_MANIFEST_HASH,
        "spec": {
            "trees_per_seed": TREES,
            "seeds": list(SEEDS),
            "learner_policy": SPEC,
            "annual_expanding_refits": True,
            "predictor_delay_bars": 12,
            "leverage": float(engine.cfg.leverage),
            "base_cost_per_side": float(engine.cfg.fee_rate + engine.cfg.slippage_rate),
            "stress_cost_per_side": COST_STRESS_PER_SIDE,
        },
        "integrity": {
            "dependency_sha256": EXPECTED_SHA256,
            "selected_positions_hash": selected_positions_hash,
            "selected_positions_match": True,
            "fold_metadata_match": True,
            "schedule_hashes": schedule_hashes,
            "expected_schedule_hashes": expected_schedule_hashes,
            "all_schedule_hashes_match": schedule_hashes == expected_schedule_hashes,
        },
        "metric_contract": {
            "calendar_cagr_counts_idle_time": True,
            "strict_mdd": (
                "global/pre-entry HWM plus entry cost, favorable-before-adverse held "
                "5m OHLC, realized funding debit, virtual adverse-mark liquidation "
                "cost, and actual exit cost"
            ),
            "funding_boundary": (
                "interior settlements symmetric; exact entry/exit debits included "
                "and credits excluded"
            ),
        },
        "legacy_frozen_stats": legacy_stats,
        "hardened_stats": hardened_stats,
        "ten_bp_per_side_stress": stressed_stats,
        "significance": significance,
        "gates": {
            "annual": "abs>0, CAGR/strict-MDD>=3, MDD<=15, >=12 trades (2026H1 >=6)",
            "combined": "future/all CAGR/strict-MDD>=3, MDD<=15, counts >=18/42",
            "stress": "future/all 10bp-per-side absolute return >0",
            "significance": "future/all weekly-cluster and stationary-bootstrap one-sided p<=0.10",
        },
    }
    payload["result_hash"] = sha_obj(payload)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    SUMMARY.write_text(render_docs(payload), encoding="utf-8")
    return payload


def main() -> None:
    payload = run()
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "summary": str(SUMMARY),
                "verdict": payload["verdict"],
                "failure_reasons": payload["failure_reasons"],
                "hardened_stats": payload["hardened_stats"],
                "significance": payload["significance"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
