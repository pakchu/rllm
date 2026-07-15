#!/usr/bin/env python3
"""Freeze a top-10 expanding ExtraTrees family before opening 2025+.

The source graph is physically truncated at 2025-01-01. Models use exact
source-owned return/adverse labels and annual expanding refits. Only 2023 test
and 2024 validation metrics rank the candidate family; 2025 and later cannot
enter this process or its artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
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
from training.search_inventory_purge_reclaim_alpha import (
    ExecutionEngine,
    Trade,
    _schedule_hash,
    equity_stats,
)
from training.search_liveparity_state_feature_interactions import immutable_anchors, slim
from training.search_stable_ensemble_conditional_pullback_alpha import (
    FEATURE_COLUMNS,
    PULLBACK_FEATURE,
    WIDTH_FEATURE,
    deterministic_forest_predict,
    routed_schedule,
    source_thresholds,
)

SELECTION_CUTOFF = "2025-01-01"
FIT_START = pd.Timestamp("2020-07-01")
SEEDS = (7, 71, 715, 2026, 71515)
TREES = 300
SELECTION_WINDOWS = (
    ("test_2023", "2023-01-01", "2024-01-01"),
    ("validation_2024", "2024-01-01", SELECTION_CUTOFF),
)
COMBINED_WINDOW = ("selection_2023_2024", "2023-01-01", SELECTION_CUTOFF)
DEFAULT_OUTPUT = "results/expanding_extratrees_top10_pre2025_2026-07-15.json"
DEFAULT_MANIFEST = "results/expanding_extratrees_top10_pre2025_manifest_2026-07-15.json"
DEFAULT_DOCS = "docs/expanding-extratrees-top10-pre2025-2026-07-15.md"


@dataclass(frozen=True)
class LearnerSpec:
    max_depth: int
    min_samples_leaf: int
    max_features: float


@dataclass(frozen=True)
class SelectionSpec:
    risk_lambda: float
    funding_quantile: float
    premium_quantile: float
    risk_quantile: float


LEARNER_GRID = tuple(
    LearnerSpec(*values)
    for values in itertools.product((2, 3), (16, 32), (0.5, 0.8))
)
SELECTION_GRID = tuple(
    SelectionSpec(*values)
    for values in itertools.product(
        (0.0, 0.25, 0.5),
        (0.30, 0.35, 0.40, 0.45),
        (0.50, 0.55, 0.60),
        (0.70, 0.75, 0.80, 0.85),
    )
)


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _action(is_funding: bool) -> tuple[int, int, int]:
    key = "funding_exit" if is_funding else "premium_exit"
    spec = CANDIDATE_SPEC[key]
    return int(spec["hold_bars"]), int(spec["take_bps"]), int(spec["stop_bps"])


def exact_labels(trade: Trade, cfg: Config) -> tuple[float, float]:
    one_side_cost = float(cfg.fee_rate + cfg.slippage_rate)
    fee_factor = 1.0 - float(cfg.leverage) * one_side_cost
    net = fee_factor * trade.price_factor * trade.funding_factor * fee_factor - 1.0
    adverse = max(
        0.0,
        1.0 - fee_factor * trade.funding_debit_factor * trade.adverse_price_factor,
    )
    return float(net), float(adverse)


def build_selection_base(cfg: Config) -> dict[str, Any]:
    if cfg.exclude_from != SELECTION_CUTOFF:
        raise ValueError(f"selection cutoff must be {SELECTION_CUTOFF}")
    context = delayed_feature_context(build_full_design(cfg), 12)
    dates = pd.to_datetime(context["dates"])
    if len(dates) and dates.max() >= pd.Timestamp(SELECTION_CUTOFF):
        raise RuntimeError("selection graph was not physically truncated before 2025")
    intervals = dates.diff().dropna()
    if len(intervals) and not intervals.eq(pd.Timedelta("5min")).all():
        raise RuntimeError("selection market source is not a complete 5-minute grid")

    execution_cfg = _execution_config(cfg, cfg.leverage)
    engine = ExecutionEngine(context["market"], context["funding"], execution_cfg)
    signals = np.flatnonzero(immutable_anchors(context["base"], 144))
    funding_source = np.asarray(context["funding_leg"], dtype=bool)[signals]
    targets: list[tuple[float, float]] = []
    exits: list[int] = []
    for signal, is_funding in zip(signals, funding_source, strict=True):
        hold, take, stop = _action(bool(is_funding))
        trade = engine.trade_at(int(signal), 1, hold, take, stop)
        targets.append((np.nan, np.nan) if trade is None else exact_labels(trade, cfg))
        exits.append(len(dates) if trade is None else int(trade.exit_position))

    return {
        "cfg": cfg,
        "context": context,
        "dates": dates,
        "execution_cfg": execution_cfg,
        "engine": engine,
        "signals": signals,
        "funding_source": funding_source,
        "targets": np.asarray(targets, dtype=float),
        "exit_positions": np.asarray(exits, dtype=int),
        "signal_dates": dates.iloc[signals].reset_index(drop=True),
        "exit_dates": dates.iloc[np.minimum(exits, len(dates) - 1)].to_numpy(),
        "width": context["matrix"][:, FEATURE_COLUMNS.index(WIDTH_FEATURE)],
        "pullback": context["matrix"][:, FEATURE_COLUMNS.index(PULLBACK_FEATURE)],
    }


def annual_masks(base: dict[str, Any], start: str, end: str) -> tuple[np.ndarray, np.ndarray]:
    cutoff = pd.Timestamp(start)
    fit = np.asarray(
        (base["signal_dates"] >= FIT_START)
        & (base["signal_dates"] < cutoff)
        & np.isfinite(base["targets"]).all(axis=1)
        & (base["exit_dates"] < cutoff.to_datetime64()),
        dtype=bool,
    )
    predict = np.asarray(
        (base["signal_dates"] >= cutoff)
        & (base["signal_dates"] < pd.Timestamp(end)),
        dtype=bool,
    )
    if not fit.any() or not predict.any():
        raise RuntimeError(f"empty annual masks for {start}..{end}")
    return fit, predict


def _balanced_weights(base: dict[str, Any], fit: np.ndarray) -> np.ndarray:
    years = pd.to_datetime(
        base["context"]["dates"].iloc[base["signals"][fit]]
    ).dt.year.to_numpy()
    sources = base["funding_source"][fit]
    groups = list(zip(years.tolist(), sources.tolist(), strict=True))
    counts = {group: groups.count(group) for group in set(groups)}
    weights = np.asarray([1.0 / counts[group] for group in groups], dtype=float)
    return weights * (len(weights) / weights.sum())


def fit_learner_folds(base: dict[str, Any], spec: LearnerSpec) -> list[dict[str, Any]]:
    context = base["context"]
    matrix = np.asarray(context["matrix"], dtype=float)
    folds: list[dict[str, Any]] = []
    for name, start, end in SELECTION_WINDOWS:
        fit, predict = annual_masks(base, start, end)
        x_fit = matrix[base["signals"][fit]]
        y_fit = base["targets"][fit]
        weights = _balanced_weights(base, fit)
        train_predictions: list[np.ndarray] = []
        period_predictions: list[np.ndarray] = []
        for seed in SEEDS:
            model = ExtraTreesRegressor(
                n_estimators=TREES,
                max_depth=spec.max_depth,
                min_samples_leaf=spec.min_samples_leaf,
                max_features=spec.max_features,
                bootstrap=False,
                random_state=seed,
                n_jobs=-1,
            ).fit(x_fit, y_fit, sample_weight=weights)
            train_predictions.append(deterministic_forest_predict(model, x_fit))
            period_predictions.append(
                deterministic_forest_predict(model, matrix[base["signals"][predict]])
            )
        folds.append(
            {
                "name": name,
                "start": start,
                "end": end,
                "fit": fit,
                "predict": predict,
                "train_prediction": np.mean(np.stack(train_predictions), axis=0),
                "period_prediction": np.mean(np.stack(period_predictions), axis=0),
            }
        )
    return folds


def activation_for(
    base: dict[str, Any],
    folds: Iterable[dict[str, Any]],
    spec: SelectionSpec,
) -> np.ndarray:
    active = np.zeros(len(base["context"]["market"]), dtype=bool)
    for fold in folds:
        fit = fold["fit"]
        predict = fold["predict"]
        fit_source = base["funding_source"][fit]
        predict_source = base["funding_source"][predict]
        train_prediction = fold["train_prediction"]
        period_prediction = fold["period_prediction"]
        train_score = train_prediction[:, 0] - spec.risk_lambda * train_prediction[:, 1]
        period_score = period_prediction[:, 0] - spec.risk_lambda * period_prediction[:, 1]
        funding_threshold, premium_threshold = source_thresholds(
            train_score,
            fit_source,
            funding_q=spec.funding_quantile,
            premium_q=spec.premium_quantile,
        )
        funding_risk_cap = float(
            np.quantile(train_prediction[fit_source, 1], spec.risk_quantile)
        )
        premium_risk_cap = float(
            np.quantile(train_prediction[~fit_source, 1], spec.risk_quantile)
        )
        positions = base["signals"][predict]
        funding_interaction = (
            base["width"][positions]
            > float(np.quantile(base["width"][base["signals"][fit]][fit_source], 0.20))
        ) | (
            base["pullback"][positions]
            <= float(np.quantile(base["pullback"][base["signals"][fit]][fit_source], 0.40))
        )
        selected = (
            predict_source
            & (period_score >= funding_threshold)
            & (period_prediction[:, 1] <= funding_risk_cap)
            & funding_interaction
        ) | (
            (~predict_source)
            & (period_score >= premium_threshold)
            & (period_prediction[:, 1] <= premium_risk_cap)
        )
        active[positions] = selected
    return active


def selection_stats(base: dict[str, Any], active: np.ndarray) -> tuple[dict[str, Any], dict[str, str]]:
    stats: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    for name, start, end in SELECTION_WINDOWS + (COMBINED_WINDOW,):
        trades = routed_schedule(
            base["context"], {"engine": base["engine"], "active": active}, start=start, end=end
        )
        stats[name] = slim(
            equity_stats(trades, start=start, end=end, cfg=base["execution_cfg"])
        )
        hashes[name] = _schedule_hash(trades)
    return stats, hashes


def selection_passes(stats: dict[str, Any]) -> bool:
    per_window = all(
        stats[name]["absolute_return_pct"] > 0.0
        and stats[name]["cagr_to_strict_mdd"] >= 3.0
        and stats[name]["strict_mdd_pct"] <= 15.0
        and stats[name]["trades"] >= 12
        for name, _, _ in SELECTION_WINDOWS
    )
    combined = stats[COMBINED_WINDOW[0]]
    return bool(
        per_window
        and combined["cagr_to_strict_mdd"] >= 3.0
        and combined["trades"] >= 24
    )


def selection_rank(stats: dict[str, Any]) -> list[float | int]:
    """Rank exclusively on 2023/2024; callers cannot pass future windows."""
    ratios = [stats[name]["cagr_to_strict_mdd"] for name, _, _ in SELECTION_WINDOWS]
    combined = stats[COMBINED_WINDOW[0]]
    return [
        int(selection_passes(stats)),
        float(min(ratios)),
        float(combined["cagr_to_strict_mdd"]),
        int(combined["trades"]),
        float(combined["absolute_return_pct"]),
    ]


def semantic_tie_break(row: dict[str, Any]) -> tuple[float | int, ...]:
    """Prefer simpler and more conservative cells when performance is identical."""
    learner = row["learner"]
    policy = row["selection"]
    return (
        -int(learner["max_depth"]),
        int(learner["min_samples_leaf"]),
        -float(learner["max_features"]),
        float(policy["risk_lambda"]),
        float(policy["funding_quantile"]),
        float(policy["premium_quantile"]),
        -float(policy["risk_quantile"]),
    )


def unique_schedule_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
    for row in rows:
        identity = (
            str(row["activation_hash"]),
            tuple(sorted((str(key), str(value)) for key, value in row["schedule_hashes"].items())),
        )
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(row)
    return unique


def _render_docs(payload: dict[str, Any], manifest: dict[str, Any]) -> str:
    lines = [
        "# Expanding ExtraTrees top-10 selection before 2025",
        "",
        "2025+ was not available to this selection run. The feature graph was physically "
        "truncated at `2025-01-01`; ranking used only 2023 test and 2024 validation.",
        "",
        f"- Grid: `{payload['grid_cells']}` cells",
        f"- Distinct schedules: `{payload['distinct_schedule_cells']}`",
        f"- Selection-pass cells: `{payload['selection_pass_cells']}`",
        "- Top-10 contains distinct schedules; exact metric ties prefer the simpler/conservative cell.",
        f"- Models: five deterministic `{TREES}`-tree ExtraTrees ensembles",
        f"- Manifest hash: `{manifest['manifest_hash']}`",
        "",
        "| Rank | Learner | Policy | 2023 abs/CAGR/MDD/ratio/trades | 2024 abs/CAGR/MDD/ratio/trades | Combined ratio/trades |",
        "| ---: | --- | --- | --- | --- | --- |",
    ]
    for row in payload["top10"]:
        learner = row["learner"]
        policy = row["selection"]
        values = []
        for name in ("test_2023", "validation_2024"):
            stat = row["stats"][name]
            values.append(
                f"{stat['absolute_return_pct']:.2f}%/{stat['cagr_pct']:.2f}%/"
                f"{stat['strict_mdd_pct']:.2f}%/{stat['cagr_to_strict_mdd']:.2f}/{stat['trades']}"
            )
        combined = row["stats"][COMBINED_WINDOW[0]]
        lines.append(
            f"| {row['rank_position']} | `d{learner['max_depth']}/leaf{learner['min_samples_leaf']}/"
            f"mf{learner['max_features']}` | `λ{policy['risk_lambda']}/fq{policy['funding_quantile']}/"
            f"pq{policy['premium_quantile']}/rq{policy['risk_quantile']}` | {values[0]} | {values[1]} | "
            f"{combined['cagr_to_strict_mdd']:.2f}/{combined['trades']} |"
        )
    lines.extend(
        [
            "",
            "## Frozen execution contract",
            "",
            "- Completed signal at t, next-open entry at t+1.",
            "- All predictive features delayed 12×5m; current source identity only is retained.",
            "- Exact source-owned exits, 6bp/notional/side, realized funding.",
            "- Stop-before-take ambiguity, non-overlap, split-contained exits.",
            "- Wall-clock CAGR and favorable-before-adverse strict global HWM MDD.",
            "- Annual expanding refits purge labels whose exits reach the cutoff.",
            "",
            "## Limitation",
            "",
            "This is a retrospective clean-room reconstruction: the program does not use 2025+, "
            "but earlier human research in this repository had already viewed those periods. The "
            "separate evaluator must therefore report this as algorithmically isolated, not human-pristine.",
            "",
        ]
    )
    return "\n".join(lines)


def run(output: str, manifest_output: str, docs_output: str) -> dict[str, Any]:
    cfg = replace(
        Config(),
        exclude_from=SELECTION_CUTOFF,
        output=output,
        docs_output="",
    )
    base = build_selection_base(cfg)
    rows: list[dict[str, Any]] = []
    for learner in LEARNER_GRID:
        folds = fit_learner_folds(base, learner)
        for selection in SELECTION_GRID:
            active = activation_for(base, folds, selection)
            stats, schedule_hashes = selection_stats(base, active)
            rows.append(
                {
                    "learner": asdict(learner),
                    "selection": asdict(selection),
                    "selection_pass": selection_passes(stats),
                    "rank": selection_rank(stats),
                    "stats": stats,
                    "schedule_hashes": schedule_hashes,
                    "activation_hash": _json_hash(np.flatnonzero(active).tolist()),
                }
            )
    rows.sort(key=lambda row: (row["rank"], semantic_tie_break(row)), reverse=True)
    distinct_rows = unique_schedule_rows(rows)
    top10 = []
    for position, row in enumerate(distinct_rows[:10], start=1):
        top10.append({"rank_position": position, **row})
    payload = {
        "schema_version": 1,
        "mode": "physical_pre2025_expanding_extratrees_top10_selection",
        "selection_cutoff_exclusive": SELECTION_CUTOFF,
        "selection_windows": {name: [start, end] for name, start, end in SELECTION_WINDOWS},
        "combined_window": list(COMBINED_WINDOW[1:]),
        "future_windows_not_opened": {
            "eval_2025": ["2025-01-01", "2026-01-01"],
            "holdout_2026h1": ["2026-01-01", "2026-06-02"],
        },
        "execution": {
            "feature_delay_bars": 12,
            "bar_size": "5min",
            "entry": "next_open",
            "costs": "6bp/notional/side",
            "funding": "realized",
            "strict_mdd": "favorable-before-adverse global/pre-entry HWM",
            "non_overlap": True,
            "split_contained_exits": True,
        },
        "model": {
            "type": "ExtraTreesRegressor multi-output net/adverse",
            "trees_per_seed": TREES,
            "seeds": list(SEEDS),
            "annual_expanding_refit": True,
            "source_year_balanced_weights": True,
            "prediction_n_jobs": 1,
        },
        "grid_cells": len(rows),
        "distinct_schedule_cells": len(distinct_rows),
        "selection_pass_cells": sum(row["selection_pass"] for row in rows),
        "source_hashes": base["context"]["source_hashes"],
        "feature_full_hash": base["context"]["feature_full_hash"],
        "top10": top10,
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "selection_result": str(output_path),
        "selection_result_sha256": _sha256(output_path),
        "selection_cutoff_exclusive": SELECTION_CUTOFF,
        "future_windows": payload["future_windows_not_opened"],
        "top10": [
            {
                "rank_position": row["rank_position"],
                "learner": row["learner"],
                "selection": row["selection"],
                "activation_hash": row["activation_hash"],
                "schedule_hashes": row["schedule_hashes"],
            }
            for row in top10
        ],
    }
    manifest["manifest_hash"] = _json_hash(manifest)
    manifest_path = Path(manifest_output)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if docs_output:
        docs_path = Path(docs_output)
        docs_path.parent.mkdir(parents=True, exist_ok=True)
        docs_path.write_text(_render_docs(payload, manifest), encoding="utf-8")
    return {"result": payload, "manifest": manifest}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest-output", default=DEFAULT_MANIFEST)
    parser.add_argument("--docs-output", default=DEFAULT_DOCS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run(args.output, args.manifest_output, args.docs_output)
    print(
        json.dumps(
            {
                "output": args.output,
                "manifest": args.manifest_output,
                "docs": args.docs_output,
                "grid_cells": report["result"]["grid_cells"],
                "selection_pass_cells": report["result"]["selection_pass_cells"],
                "manifest_hash": report["manifest"]["manifest_hash"],
                "top10": [
                    {
                        "rank_position": row["rank_position"],
                        "learner": row["learner"],
                        "selection": row["selection"],
                        "rank": row["rank"],
                    }
                    for row in report["result"]["top10"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
