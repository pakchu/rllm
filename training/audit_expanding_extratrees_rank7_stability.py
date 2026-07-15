#!/usr/bin/env python3
"""Audit frozen ExtraTrees rank 7 across seeds and tree counts."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

# Keep native BLAS/OpenMP deterministic-ish and avoid hidden prediction fanout.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.audit_confirmed_pullback_squeeze_live_parity import _execution_config
from training.audit_stable_ensemble_conditional_pullback_alpha import delayed_feature_context
from training.audit_weak_feature_responsibility_stability import CANDIDATE_SPEC
from training.evaluate_stable_ensemble_conditional_pullback_oos import Config, build_full_design
from training.evaluate_expanding_extratrees_top10_oos import validate_manifest
from training.search_inventory_purge_reclaim_alpha import ExecutionEngine, equity_stats
from training.search_liveparity_state_feature_interactions import immutable_anchors, slim
from training.search_stable_ensemble_conditional_pullback_alpha import (
    FEATURE_COLUMNS,
    PULLBACK_FEATURE,
    WIDTH_FEATURE,
    routed_schedule,
    source_thresholds,
)
from training.select_expanding_extratrees_top10_pre2025 import DEFAULT_MANIFEST

OUT = Path("results/expanding_extratrees_rank7_stability_2026-07-15.json")
SUMMARY = Path("docs/expanding-extratrees-rank7-stability-2026-07-15.md")
FROZEN_RANK = 7
SEEDS = (7, 71, 715, 2026, 71515)
TREE_COUNTS = (300, 1000, 2000)
FOLDS = (
    ("2023", "2023-01-01", "2024-01-01"),
    ("2024", "2024-01-01", "2025-01-01"),
    ("2025", "2025-01-01", "2026-01-01"),
    ("2026h1", "2026-01-01", "2026-06-02"),
)
SPEC = {
    "max_depth": 2,
    "min_samples_leaf": 32,
    "max_features": 0.8,
    "lambda": 0.25,
    "funding_q": 0.40,
    "premium_q": 0.55,
    "risk_q": 0.75,
}


def validate_frozen_spec() -> str:
    manifest, _ = validate_manifest(DEFAULT_MANIFEST)
    row = manifest["top10"][FROZEN_RANK - 1]
    expected = {
        "max_depth": SPEC["max_depth"],
        "min_samples_leaf": SPEC["min_samples_leaf"],
        "max_features": SPEC["max_features"],
    }
    policy = {
        "risk_lambda": SPEC["lambda"],
        "funding_quantile": SPEC["funding_q"],
        "premium_quantile": SPEC["premium_q"],
        "risk_quantile": SPEC["risk_q"],
    }
    if row["rank_position"] != FROZEN_RANK or row["learner"] != expected or row["selection"] != policy:
        raise RuntimeError("frozen rank-7 specification drifted")
    return str(manifest["manifest_hash"])


def action(is_funding: bool) -> tuple[int, int, int]:
    spec = CANDIDATE_SPEC["funding_exit"] if is_funding else CANDIDATE_SPEC["premium_exit"]
    return int(spec["hold_bars"]), int(spec["take_bps"]), int(spec["stop_bps"])


def exact_labels(trade: Any, cfg: Config) -> tuple[float, float]:
    cost = cfg.fee_rate + cfg.slippage_rate
    fee = 1.0 - cfg.leverage * cost
    # Source-owned exact labels: execution-engine price/funding/adverse factors.
    net = fee * trade.price_factor * trade.funding_factor * fee - 1.0
    adverse = max(0.0, 1.0 - fee * trade.funding_debit_factor * trade.adverse_price_factor)
    return float(net), float(adverse)


def deterministic_predict(model: ExtraTreesRegressor, matrix: np.ndarray) -> np.ndarray:
    # Required: prediction must force n_jobs=1.
    model.n_jobs = 1
    return np.asarray(model.predict(matrix), dtype=float)


def sha_obj(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def pass_fail(stats: dict[str, dict[str, float]]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    for name, _, _ in FOLDS:
        s = stats[name]
        min_trades = 6 if name == "2026h1" else 12
        if not (s["absolute_return_pct"] > 0.0):
            reasons.append(f"{name}:nonpositive_return")
        if not (s["cagr_to_strict_mdd"] >= 3.0):
            reasons.append(f"{name}:ratio_lt_3")
        if not (s["strict_mdd_pct"] <= 15.0):
            reasons.append(f"{name}:mdd_gt_15")
        if not (s["trades"] >= min_trades):
            reasons.append(f"{name}:trades_lt_{min_trades}")
    if not (stats["all"]["cagr_to_strict_mdd"] >= 3.0):
        reasons.append("all:ratio_lt_3")
    if not (stats["all"]["trades"] >= 42):
        reasons.append("all:trades_lt_42")
    return not reasons, reasons


def build_base() -> dict[str, Any]:
    cfg = replace(Config(), output="/tmp/no_write_repo.json", docs_output="")
    context = delayed_feature_context(build_full_design(cfg), 12)  # 1h delay at 5m bars.
    dates = context["dates"]
    engine = ExecutionEngine(context["market"], context["funding"], _execution_config(cfg, cfg.leverage))
    signals = np.flatnonzero(immutable_anchors(context["base"], 144))
    funding = np.asarray(context["funding_leg"], dtype=bool)[signals]

    ys: list[tuple[float, float]] = []
    exits: list[int] = []
    for sig, is_funding in zip(signals, funding, strict=True):
        hold, take, stop = action(bool(is_funding))
        trade = engine.trade_at(int(sig), 1, hold, take, stop)  # next-open/source engine execution.
        ys.append((np.nan, np.nan) if trade is None else exact_labels(trade, cfg))
        exits.append(len(dates) if trade is None else int(trade.exit_position))

    return {
        "cfg": cfg,
        "context": context,
        "dates": dates,
        "engine": engine,
        "signals": signals,
        "funding": funding,
        "y": np.asarray(ys, dtype=float),
        "exit_positions": np.asarray(exits, dtype=int),
        "signal_dates": dates.iloc[signals].reset_index(drop=True),
        "end_dates": dates.iloc[np.minimum(exits, len(dates) - 1)].to_numpy(),
        "width": context["matrix"][:, FEATURE_COLUMNS.index(WIDTH_FEATURE)],
        "pullback": context["matrix"][:, FEATURE_COLUMNS.index(PULLBACK_FEATURE)],
    }


def fit_seed_predict(base: dict[str, Any], fit: np.ndarray, pred: np.ndarray, seed: int, trees: int) -> tuple[np.ndarray, np.ndarray]:
    context = base["context"]
    signals = base["signals"]
    funding = base["funding"]
    y = base["y"]
    x = context["matrix"]
    xf = x[signals[fit]]
    yf = y[fit]
    years = pd.to_datetime(context["dates"].iloc[signals[fit]]).dt.year.to_numpy()
    src = funding[fit]
    groups = list(zip(years.tolist(), src.tolist(), strict=True))
    counts = {g: groups.count(g) for g in set(groups)}
    weights = np.array([1.0 / counts[g] for g in groups], dtype=float)
    weights *= len(weights) / weights.sum()
    model = ExtraTreesRegressor(
        n_estimators=int(trees),
        max_depth=int(SPEC["max_depth"]),
        min_samples_leaf=int(SPEC["min_samples_leaf"]),
        max_features=float(SPEC["max_features"]),
        bootstrap=False,
        random_state=int(seed),
        n_jobs=-1,
    ).fit(xf, yf, sample_weight=weights)
    train_pred = deterministic_predict(model, xf)
    year_pred = deterministic_predict(model, x[signals[pred]])
    return train_pred, year_pred


def evaluate(base: dict[str, Any], *, trees: int, seeds: tuple[int, ...], label: str) -> dict[str, Any]:
    context = base["context"]
    signals = base["signals"]
    funding = base["funding"]
    y = base["y"]
    signal_dates = base["signal_dates"]
    end_dates = base["end_dates"]
    width = base["width"]
    pullback = base["pullback"]
    engine = base["engine"]
    cfg = base["cfg"]

    active = np.zeros(len(context["market"]), dtype=bool)
    fold_meta: list[dict[str, Any]] = []
    for name, start, end in FOLDS:
        cutoff = pd.Timestamp(start)
        fit = np.asarray(
            (signal_dates >= pd.Timestamp("2020-07-01"))
            & (signal_dates < cutoff)
            & np.isfinite(y).all(axis=1)
            & (end_dates < cutoff.to_datetime64()),  # purged exits before cutoff.
            dtype=bool,
        )
        pred = np.asarray((signal_dates >= cutoff) & (signal_dates < pd.Timestamp(end)), dtype=bool)

        train_preds: list[np.ndarray] = []
        pred_preds: list[np.ndarray] = []
        for seed in seeds:
            tr, pr = fit_seed_predict(base, fit, pred, int(seed), int(trees))
            train_preds.append(tr)
            pred_preds.append(pr)
        train_pred = np.mean(train_preds, axis=0)
        year_pred = np.mean(pred_preds, axis=0)

        train_score = train_pred[:, 0] - float(SPEC["lambda"]) * train_pred[:, 1]
        pred_score = year_pred[:, 0] - float(SPEC["lambda"]) * year_pred[:, 1]
        fit_src = funding[fit]
        pred_src = funding[pred]
        ft, pt = source_thresholds(
            train_score,
            fit_src,
            funding_q=float(SPEC["funding_q"]),
            premium_q=float(SPEC["premium_q"]),
        )
        risk = train_pred[:, 1]
        f_risk_cap = float(np.quantile(risk[fit_src], float(SPEC["risk_q"])))
        p_risk_cap = float(np.quantile(risk[~fit_src], float(SPEC["risk_q"])))
        pos = signals[pred]
        wt = float(np.quantile(width[signals[fit]][fit_src], 0.2))
        rt = float(np.quantile(pullback[signals[fit]][fit_src], 0.4))
        weak_interaction_gate = (width[pos] > wt) | (pullback[pos] <= rt)
        selected = (
            pred_src & (pred_score >= ft) & (year_pred[:, 1] <= f_risk_cap) & weak_interaction_gate
        ) | ((~pred_src) & (pred_score >= pt) & (year_pred[:, 1] <= p_risk_cap))
        active[pos] = selected
        fold_meta.append(
            {
                "name": name,
                "fit_examples": int(fit.sum()),
                "predict_events": int(pred.sum()),
                "selected_events": int(selected.sum()),
                "funding_score": float(ft),
                "premium_score": float(pt),
                "funding_risk_cap": f_risk_cap,
                "premium_risk_cap": p_risk_cap,
                "width_q20": wt,
                "pullback_q40": rt,
                "prediction_n_jobs_forced": 1,
            }
        )

    stats: dict[str, dict[str, Any]] = {}
    schedules: dict[str, Any] = {}
    for n, s, e in FOLDS + (("all", "2023-01-01", "2026-06-02"),):
        trades = routed_schedule(context, {"engine": engine, "active": active}, start=s, end=e)
        stats[n] = slim(equity_stats(trades, start=s, end=e, cfg=_execution_config(cfg, cfg.leverage)))
        schedules[n] = [
            {
                "entry_position": int(getattr(t, "entry_position", -1)),
                "exit_position": int(getattr(t, "exit_position", -1)),
                "entry_time": str(getattr(t, "entry_time", "")),
                "exit_time": str(getattr(t, "exit_time", "")),
                "price_factor": float(getattr(t, "price_factor", np.nan)),
                "funding_factor": float(getattr(t, "funding_factor", np.nan)),
            }
            for t in trades
        ]
    passed, reasons = pass_fail(stats)
    active_positions = np.flatnonzero(active).astype(int).tolist()
    hash_payload = {"label": label, "trees": trees, "seeds": list(seeds), "stats": stats, "active_positions": active_positions, "schedules": schedules}
    return {
        "label": label,
        "trees": int(trees),
        "seeds": list(map(int, seeds)),
        "pass": bool(passed),
        "fail_reasons": reasons,
        "stats": stats,
        "folds": fold_meta,
        "selected_positions_hash": sha_obj(active_positions),
        "full_result_hash": sha_obj(hash_payload),
    }


def render_md(payload: dict[str, Any]) -> str:
    lines = ["# Frozen ExtraTrees rank-7 stability — 2026-07-15", ""]
    lines.append(f"Artifact JSON: `{OUT}`")
    lines.append("")
    lines.append(f"Manifest: `{payload['manifest_hash']}`, frozen rank: `{FROZEN_RANK}`")
    lines.append("")
    lines.append(
        "Spec: `max_depth=2,min_samples_leaf=32,max_features=.8,lambda=.25,"
        "funding_q=.40,premium_q=.55,risk_q=.75`; 1h delayed features; "
        "source-owned exact labels; purged exits; next-open; exact costs/funding/"
        "strict MDD; prediction `n_jobs=1` forced."
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"Individual seed passes: `{payload['individual_pass_count']}/{len(SEEDS)}`")
    for trees in TREE_COUNTS:
        ens = payload["ensembles"][str(trees)]
        lines.append(f"- {trees}-tree 5-seed mean ensemble: **{'PASS' if ens['pass'] else 'FAIL'}**, hash `{ens['full_result_hash'][:16]}`")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append("| Case | Period | Abs ret | CAGR | MDD | Ratio | Trades | Pass | Hash |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |")
    for r in payload["individuals"] + [payload["ensembles"][str(t)] for t in TREE_COUNTS]:
        for period in ["2023", "2024", "2025", "2026h1", "all"]:
            s = r["stats"][period]
            lines.append(
                f"| {r['label']} | {period} | {s['absolute_return_pct']:.4f}% | {s['cagr_pct']:.4f}% | {s['strict_mdd_pct']:.4f}% | {s['cagr_to_strict_mdd']:.4f} | {int(s['trades'])} | {'PASS' if r['pass'] else 'FAIL'} | `{r['full_result_hash'][:12]}` |"
            )
    lines.append("")
    lines.append("## Determinism")
    lines.append("")
    for d in payload["determinism_checks"]:
        lines.append(f"- {d['label']}: {'MATCH' if d['match'] else 'MISMATCH'} first `{d['first_hash']}`, repeat `{d['repeat_hash']}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    manifest_hash = validate_frozen_spec()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    base = build_base()
    individuals = [evaluate(base, trees=300, seeds=(seed,), label=f"seed{seed}_300") for seed in SEEDS]
    ensembles = {str(trees): evaluate(base, trees=trees, seeds=SEEDS, label=f"ensemble5_{trees}") for trees in TREE_COUNTS}
    # Fresh repeat of each ensemble for determinism evidence. This is intentionally exact rerun, not cached.
    determinism_checks = []
    for trees in TREE_COUNTS:
        repeat = evaluate(base, trees=trees, seeds=SEEDS, label=f"ensemble5_{trees}")
        first = ensembles[str(trees)]
        determinism_checks.append(
            {
                "label": f"ensemble5_{trees}",
                "match": bool(first["full_result_hash"] == repeat["full_result_hash"] and first["selected_positions_hash"] == repeat["selected_positions_hash"]),
                "first_hash": first["full_result_hash"],
                "repeat_hash": repeat["full_result_hash"],
                "first_positions_hash": first["selected_positions_hash"],
                "repeat_positions_hash": repeat["selected_positions_hash"],
            }
        )
    payload = {
        "mode": "frozen_rank7_extratrees_annual_expanding_stability",
        "manifest_hash": manifest_hash,
        "frozen_rank": FROZEN_RANK,
        "outputs": {"json": str(OUT), "summary_md": str(SUMMARY)},
        "spec": SPEC,
        "constraints": {
            "annual_expanding_refits": True,
            "predictor_delay_bars": 12,
            "source_owned_exact_labels": True,
            "label_purging": "trade exit position/date strictly before annual cutoff",
            "execution": "source ExecutionEngine + routed_schedule + equity_stats; next-open; exact costs/funding/strict MDD",
            "prediction_n_jobs_forced": 1,
            "pass_rule": "annual abs>0, ratio>=3, MDD<=15, trades>=12 (>=6 for 2026h1), all ratio>=3 and trades>=42",
        },
        "folds": list(FOLDS),
        "seeds": list(SEEDS),
        "individuals": individuals,
        "individual_pass_count": int(sum(r["pass"] for r in individuals)),
        "ensembles": ensembles,
        "determinism_checks": determinism_checks,
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    SUMMARY.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({
        "json": str(OUT),
        "summary_md": str(SUMMARY),
        "individual_pass_count": payload["individual_pass_count"],
        "ensemble_passes": {k: v["pass"] for k, v in ensembles.items()},
        "determinism": determinism_checks,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
