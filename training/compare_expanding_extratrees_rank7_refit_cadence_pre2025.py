#!/usr/bin/env python3
"""Freeze annual-vs-monthly rank-7 expanding-refit cadence before opening 2025+.

Both schedules use the already-frozen rank-7 learner and policy. The only changed
dimension is the refit cutoff: January 1 for annual folds versus every calendar
month start for monthly folds. The feature graph is physically truncated at
2025-01-01, labels are purged when their exits reach a fold cutoff, and cadence
selection uses only 2023 test plus 2024 validation.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, replace
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

from training.evaluate_expanding_extratrees_top10_oos import validate_manifest
from training.evaluate_stable_ensemble_conditional_pullback_oos import Config
from training.search_stable_ensemble_conditional_pullback_alpha import (
    deterministic_forest_predict,
)
from training.select_expanding_extratrees_top10_pre2025 import (
    COMBINED_WINDOW,
    DEFAULT_MANIFEST as RANKED_FAMILY_MANIFEST,
    FIT_START,
    SEEDS,
    SELECTION_CUTOFF,
    SELECTION_WINDOWS,
    TREES,
    LearnerSpec,
    SelectionSpec,
    _balanced_weights,
    _json_hash,
    _sha256,
    activation_for,
    build_selection_base,
    selection_passes,
    selection_rank,
    selection_stats,
)

FROZEN_RANK = 7
DEFAULT_OUTPUT = "results/expanding_extratrees_rank7_refit_cadence_pre2025_2026-07-16.json"
DEFAULT_MANIFEST = "results/expanding_extratrees_rank7_refit_cadence_pre2025_manifest_2026-07-16.json"
DEFAULT_DOCS = "docs/expanding-extratrees-rank7-refit-cadence-pre2025-2026-07-16.md"
DIAGNOSTIC_WINDOWS = (
    ("test_2023_h1", "2023-01-01", "2023-07-01"),
    ("test_2023_h2", "2023-07-01", "2024-01-01"),
    ("validation_2024_h1", "2024-01-01", "2024-07-01"),
    ("validation_2024_h2", "2024-07-01", "2025-01-01"),
)


def frozen_rank7_spec(manifest_path: str = RANKED_FAMILY_MANIFEST) -> tuple[str, LearnerSpec, SelectionSpec, dict[str, Any]]:
    manifest, _ = validate_manifest(manifest_path)
    row = manifest["top10"][FROZEN_RANK - 1]
    if int(row["rank_position"]) != FROZEN_RANK:
        raise RuntimeError("frozen rank-7 row is missing")
    learner = LearnerSpec(**row["learner"])
    policy = SelectionSpec(**row["selection"])
    return str(manifest["manifest_hash"]), learner, policy, row


def cadence_windows(cadence: str, start: str, end: str) -> tuple[tuple[str, str, str], ...]:
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    if cadence == "annual":
        rows: list[tuple[str, str, str]] = []
        cursor = start_ts
        while cursor < end_ts:
            next_year = pd.Timestamp(year=cursor.year + 1, month=1, day=1)
            fold_end = min(next_year, end_ts)
            rows.append(
                (
                    f"year_{cursor:%Y}",
                    cursor.strftime("%Y-%m-%d"),
                    fold_end.strftime("%Y-%m-%d"),
                )
            )
            cursor = fold_end
        return tuple(rows)
    if cadence != "monthly":
        raise ValueError(f"unsupported cadence: {cadence}")
    starts = pd.date_range(start_ts, end_ts, freq="MS", inclusive="left")
    rows = []
    for month_start in starts:
        month_end = min(month_start + pd.offsets.MonthBegin(1), end_ts)
        rows.append(
            (
                f"month_{month_start:%Y_%m}",
                month_start.strftime("%Y-%m-%d"),
                pd.Timestamp(month_end).strftime("%Y-%m-%d"),
            )
        )
    return tuple(rows)


def cutoff_masks(base: dict[str, Any], start: str, end: str) -> tuple[np.ndarray, np.ndarray]:
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
    if not fit.any():
        raise RuntimeError(f"empty cadence fit mask for {start}..{end}")
    return fit, predict


def fit_cadence_folds(
    base: dict[str, Any],
    learner: LearnerSpec,
    cadence: str,
    *,
    start: str = "2023-01-01",
    end: str = SELECTION_CUTOFF,
) -> list[dict[str, Any]]:
    matrix = np.asarray(base["context"]["matrix"], dtype=float)
    folds: list[dict[str, Any]] = []
    for name, fold_start, fold_end in cadence_windows(cadence, start, end):
        fit, predict = cutoff_masks(base, fold_start, fold_end)
        x_fit = matrix[base["signals"][fit]]
        y_fit = base["targets"][fit]
        weights = _balanced_weights(base, fit)
        train_predictions: list[np.ndarray] = []
        period_predictions: list[np.ndarray] = []
        for seed in SEEDS:
            model = ExtraTreesRegressor(
                n_estimators=TREES,
                max_depth=learner.max_depth,
                min_samples_leaf=learner.min_samples_leaf,
                max_features=learner.max_features,
                bootstrap=False,
                random_state=seed,
                n_jobs=-1,
            ).fit(x_fit, y_fit, sample_weight=weights)
            train_predictions.append(deterministic_forest_predict(model, x_fit))
            if predict.any():
                period_predictions.append(
                    deterministic_forest_predict(model, matrix[base["signals"][predict]])
                )
            else:
                period_predictions.append(np.empty((0, y_fit.shape[1]), dtype=float))
        folds.append(
            {
                "name": name,
                "start": fold_start,
                "end": fold_end,
                "fit": fit,
                "predict": predict,
                "fit_examples": int(fit.sum()),
                "predict_events": int(predict.sum()),
                "latest_fit_exit": str(pd.Timestamp(base["exit_dates"][fit].max())),
                "train_prediction": np.mean(np.stack(train_predictions), axis=0),
                "period_prediction": np.mean(np.stack(period_predictions), axis=0),
            }
        )
    return folds


def diagnostic_stats(base: dict[str, Any], active: np.ndarray) -> dict[str, Any]:
    from training.search_inventory_purge_reclaim_alpha import equity_stats
    from training.search_liveparity_state_feature_interactions import slim
    from training.search_stable_ensemble_conditional_pullback_alpha import routed_schedule

    out: dict[str, Any] = {}
    for name, start, end in DIAGNOSTIC_WINDOWS:
        trades = routed_schedule(
            base["context"], {"engine": base["engine"], "active": active}, start=start, end=end
        )
        out[name] = slim(equity_stats(trades, start=start, end=end, cfg=base["execution_cfg"]))
    return out


def cadence_result(
    base: dict[str, Any],
    learner: LearnerSpec,
    policy: SelectionSpec,
    cadence: str,
) -> dict[str, Any]:
    folds = fit_cadence_folds(base, learner, cadence)
    active = activation_for(base, folds, policy)
    stats, hashes = selection_stats(base, active)
    return {
        "cadence": cadence,
        "fold_count": len(folds),
        "folds": [
            {
                key: fold[key]
                for key in ("name", "start", "end", "fit_examples", "predict_events", "latest_fit_exit")
            }
            for fold in folds
        ],
        "selection_pass": selection_passes(stats),
        "rank": selection_rank(stats),
        "stats": stats,
        "diagnostics": diagnostic_stats(base, active),
        "schedule_hashes": hashes,
        "activation_hash": _json_hash(np.flatnonzero(active).tolist()),
        "active_events": int(active.sum()),
    }


def select_cadence(results: Iterable[dict[str, Any]]) -> str:
    """Use only the pre-2025 rank tuple; exact ties prefer annual simplicity."""
    rows = list(results)
    if {row["cadence"] for row in rows} != {"annual", "monthly"}:
        raise ValueError("selection requires exactly annual and monthly results")
    return max(
        rows,
        key=lambda row: (tuple(row["rank"]), int(row["cadence"] == "annual")),
    )["cadence"]


def metric_delta(monthly: dict[str, Any], annual: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "absolute_return_pct",
        "cagr_pct",
        "strict_mdd_pct",
        "cagr_to_strict_mdd",
        "trades",
        "mean_net_bps",
        "win_rate",
    )
    return {
        window: {key: float(monthly[window][key]) - float(annual[window][key]) for key in keys}
        for window in annual
    }


def render_docs(payload: dict[str, Any], manifest: dict[str, Any]) -> str:
    annual = payload["cadences"]["annual"]
    monthly = payload["cadences"]["monthly"]
    lines = [
        "# Frozen rank-7 annual vs monthly refit cadence before 2025",
        "",
        "The feature graph was physically truncated at `2025-01-01`. Both cadences use "
        "the same frozen rank-7 learner, policy, five seeds, exact labels, and execution; "
        "only the expanding-refit cutoff changes.",
        "",
        f"- Parent rank-family manifest: `{payload['parent_manifest_hash']}`",
        f"- Cadence manifest: `{manifest['manifest_hash']}`",
        f"- Pre-2025 selected cadence: **{payload['selected_cadence']}**",
        f"- Annual folds: `{annual['fold_count']}`; monthly folds: `{monthly['fold_count']}`",
        "",
        "## Results",
        "",
        "| Cadence | Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Mean net | Win rate | Pass |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for cadence in ("annual", "monthly"):
        row = payload["cadences"][cadence]
        for name, _, _ in SELECTION_WINDOWS + (COMBINED_WINDOW,):
            s = row["stats"][name]
            lines.append(
                f"| {cadence} | {name} | {s['absolute_return_pct']:.4f}% | "
                f"{s['cagr_pct']:.4f}% | {s['strict_mdd_pct']:.4f}% | "
                f"{s['cagr_to_strict_mdd']:.4f} | {s['trades']} | "
                f"{s['mean_net_bps']:.2f} bps | {100*s['win_rate']:.2f}% | "
                f"{'PASS' if row['selection_pass'] else 'FAIL'} |"
            )
    lines.extend(
        [
            "",
            "## Selection rule",
            "",
            "Cadence is selected lexicographically using only 2023/2024: pass flag, "
            "worst yearly CAGR/MDD, combined CAGR/MDD, combined trades, combined absolute "
            "return. Exact ties prefer annual refit as the simpler schedule.",
            "",
            "## Integrity",
            "",
            "- Every monthly fit purges targets whose source-owned exits reach that month start.",
            "- The annual result must exactly reproduce frozen rank-7 2023/2024 schedules.",
            "- No 2025+ source row is opened by this program.",
            "- This artifact freezes cadence choice before the separate future evaluator is run.",
            "",
        ]
    )
    return "\n".join(lines)


def run(output: str, manifest_output: str, docs_output: str) -> dict[str, Any]:
    parent_hash, learner, policy, frozen_row = frozen_rank7_spec()
    cfg = replace(Config(), exclude_from=SELECTION_CUTOFF, output="/tmp/no_write.json", docs_output="")
    base = build_selection_base(cfg)
    annual = cadence_result(base, learner, policy, "annual")
    monthly = cadence_result(base, learner, policy, "monthly")
    if annual["schedule_hashes"] != frozen_row["schedule_hashes"]:
        raise RuntimeError("annual cadence no longer reproduces frozen rank-7 schedules")
    selected = select_cadence((annual, monthly))
    payload = {
        "schema_version": 1,
        "mode": "physical_pre2025_frozen_rank7_refit_cadence_selection",
        "selection_cutoff_exclusive": SELECTION_CUTOFF,
        "future_windows_not_opened": {
            "eval_2025": ["2025-01-01", "2026-01-01"],
            "holdout_2026h1": ["2026-01-01", "2026-06-02"],
        },
        "parent_manifest_hash": parent_hash,
        "frozen_rank": FROZEN_RANK,
        "learner": asdict(learner),
        "selection": asdict(policy),
        "seeds": list(SEEDS),
        "trees_per_seed": TREES,
        "feature_delay_bars": 12,
        "source_year_balanced_weights": True,
        "label_purge": "source-owned exit strictly before each annual/monthly cutoff",
        "selection_rule": [
            "selection_pass",
            "min_yearly_cagr_to_strict_mdd",
            "combined_cagr_to_strict_mdd",
            "combined_trades",
            "combined_absolute_return_pct",
            "annual_on_exact_tie",
        ],
        "selected_cadence": selected,
        "source_hashes": base["context"]["source_hashes"],
        "feature_full_hash": base["context"]["feature_full_hash"],
        "cadences": {"annual": annual, "monthly": monthly},
        "monthly_minus_annual": metric_delta(monthly["stats"], annual["stats"]),
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "selection_result": str(output_path),
        "selection_result_sha256": _sha256(output_path),
        "selection_cutoff_exclusive": SELECTION_CUTOFF,
        "parent_manifest_hash": parent_hash,
        "frozen_rank": FROZEN_RANK,
        "cadences": ["annual", "monthly"],
        "selected_cadence": selected,
        "selection_rule": payload["selection_rule"],
        "future_windows": payload["future_windows_not_opened"],
        "activation_hashes": {
            name: row["activation_hash"] for name, row in payload["cadences"].items()
        },
        "schedule_hashes": {
            name: row["schedule_hashes"] for name, row in payload["cadences"].items()
        },
    }
    manifest["manifest_hash"] = _json_hash(manifest)
    manifest_path = Path(manifest_output)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if docs_output:
        docs_path = Path(docs_output)
        docs_path.parent.mkdir(parents=True, exist_ok=True)
        docs_path.write_text(render_docs(payload, manifest), encoding="utf-8")
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
    result = report["result"]
    print(
        json.dumps(
            {
                "output": args.output,
                "manifest": args.manifest_output,
                "docs": args.docs_output,
                "selected_cadence": result["selected_cadence"],
                "manifest_hash": report["manifest"]["manifest_hash"],
                "annual": result["cadences"]["annual"]["stats"],
                "monthly": result["cadences"]["monthly"]["stats"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
