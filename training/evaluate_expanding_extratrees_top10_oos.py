#!/usr/bin/env python3
"""Evaluate the frozen pre-2025 ExtraTrees top-10 on 2025 and 2026H1."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

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
from training.evaluate_stable_ensemble_conditional_pullback_oos import Config, build_full_design
from training.search_inventory_purge_reclaim_alpha import ExecutionEngine, _schedule_hash, equity_stats
from training.search_liveparity_state_feature_interactions import immutable_anchors, slim
from training.search_stable_ensemble_conditional_pullback_alpha import (
    FEATURE_COLUMNS,
    PULLBACK_FEATURE,
    WIDTH_FEATURE,
    deterministic_forest_predict,
    routed_schedule,
)
from training.select_expanding_extratrees_top10_pre2025 import (
    COMBINED_WINDOW,
    DEFAULT_MANIFEST,
    SEEDS,
    SELECTION_CUTOFF,
    SELECTION_WINDOWS,
    TREES,
    LearnerSpec,
    SelectionSpec,
    _action,
    _balanced_weights,
    _json_hash,
    annual_masks,
    build_selection_base,
    exact_labels,
    activation_for,
)

EXPECTED_MANIFEST_HASH = "c6e7d78a328118456eacf70bc42cb12a48f33e26d13edbe21f2edb3aedea4f8e"
FULL_CUTOFF = "2026-06-02"
REPLAY_WINDOWS = (
    ("test_2023", "2023-01-01", "2024-01-01"),
    ("validation_2024", "2024-01-01", "2025-01-01"),
    ("eval_2025", "2025-01-01", "2026-01-01"),
    ("holdout_2026h1", "2026-01-01", FULL_CUTOFF),
)
FUTURE_WINDOWS = REPLAY_WINDOWS[2:]
FUTURE_COMBINED = ("future_2025_2026h1", "2025-01-01", FULL_CUTOFF)
ALL_COMBINED = ("all_2023_2026h1", "2023-01-01", FULL_CUTOFF)
DEFAULT_OUTPUT = "results/expanding_extratrees_top10_oos_2026-07-15.json"
DEFAULT_DOCS = "docs/expanding-extratrees-top10-oos-2026-07-15.md"


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_manifest(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    embedded_hash = manifest.pop("manifest_hash", None)
    calculated_hash = _json_hash(manifest)
    if embedded_hash != calculated_hash or calculated_hash != EXPECTED_MANIFEST_HASH:
        raise RuntimeError("frozen manifest hash mismatch")
    manifest["manifest_hash"] = embedded_hash
    if manifest["selection_cutoff_exclusive"] != SELECTION_CUTOFF:
        raise RuntimeError("selection cutoff drifted")
    if manifest["future_windows"] != {
        "eval_2025": ["2025-01-01", "2026-01-01"],
        "holdout_2026h1": ["2026-01-01", FULL_CUTOFF],
    }:
        raise RuntimeError("future windows drifted")
    if len(manifest["top10"]) != 10:
        raise RuntimeError("manifest does not contain exactly ten candidates")
    selection_path = Path(manifest["selection_result"])
    if _sha256(selection_path) != manifest["selection_result_sha256"]:
        raise RuntimeError("frozen selection result hash mismatch")
    selection_result = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection_result["selection_cutoff_exclusive"] != SELECTION_CUTOFF:
        raise RuntimeError("selection result cutoff drifted")
    return manifest, selection_result


def build_replay_base(cfg: Config) -> dict[str, Any]:
    if cfg.exclude_from != FULL_CUTOFF:
        raise ValueError(f"replay cutoff must be {FULL_CUTOFF}")
    context = delayed_feature_context(build_full_design(cfg), 12)
    dates = pd.to_datetime(context["dates"])
    if len(dates) and dates.max() >= pd.Timestamp(FULL_CUTOFF):
        raise RuntimeError("replay graph exceeded frozen horizon")
    intervals = dates.diff().dropna()
    if len(intervals) and not intervals.eq(pd.Timedelta("5min")).all():
        raise RuntimeError("replay source is not a complete 5-minute grid")
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


def assert_prefix_parity(selection_base: dict[str, Any], replay_base: dict[str, Any]) -> None:
    selection_context = selection_base["context"]
    replay_context = replay_base["context"]
    prefix_length = len(selection_context["dates"])
    pairs = (
        (selection_context["dates"].to_numpy(), replay_context["dates"].iloc[:prefix_length].to_numpy(), "dates"),
        (selection_context["matrix"], replay_context["matrix"][:prefix_length], "matrix"),
        (selection_context["base"], replay_context["base"][:prefix_length], "base"),
        (selection_context["funding_leg"], replay_context["funding_leg"][:prefix_length], "funding_leg"),
        (selection_context["premium_leg"], replay_context["premium_leg"][:prefix_length], "premium_leg"),
    )
    for expected, actual, name in pairs:
        if not np.array_equal(expected, actual, equal_nan=True):
            raise RuntimeError(f"pre-2025 {name} prefix changed after opening future data")


def fit_replay_folds(base: dict[str, Any], spec: LearnerSpec) -> list[dict[str, Any]]:
    context = base["context"]
    matrix = np.asarray(context["matrix"], dtype=float)
    folds: list[dict[str, Any]] = []
    for name, start, end in REPLAY_WINDOWS:
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


def replay_stats(
    base: dict[str, Any], active: np.ndarray
) -> tuple[dict[str, Any], dict[str, str]]:
    stats: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    windows = REPLAY_WINDOWS + (COMBINED_WINDOW, FUTURE_COMBINED, ALL_COMBINED)
    for name, start, end in windows:
        trades = routed_schedule(
            base["context"], {"engine": base["engine"], "active": active}, start=start, end=end
        )
        stats[name] = slim(equity_stats(trades, start=start, end=end, cfg=base["execution_cfg"]))
        hashes[name] = _schedule_hash(trades)
    return stats, hashes


def future_passes(stats: dict[str, Any]) -> bool:
    per_window = all(
        stats[name]["absolute_return_pct"] > 0.0
        and stats[name]["cagr_to_strict_mdd"] >= 3.0
        and stats[name]["strict_mdd_pct"] <= 15.0
        and stats[name]["trades"] >= (6 if name == "holdout_2026h1" else 12)
        for name, _, _ in FUTURE_WINDOWS
    )
    combined = stats[FUTURE_COMBINED[0]]
    return bool(
        per_window
        and combined["cagr_to_strict_mdd"] >= 3.0
        and combined["trades"] >= 18
    )


def full_passes(stats: dict[str, Any]) -> bool:
    per_window = all(
        stats[name]["absolute_return_pct"] > 0.0
        and stats[name]["cagr_to_strict_mdd"] >= 3.0
        and stats[name]["strict_mdd_pct"] <= 15.0
        and stats[name]["trades"] >= (6 if name == "holdout_2026h1" else 12)
        for name, _, _ in REPLAY_WINDOWS
    )
    combined = stats[ALL_COMBINED[0]]
    return bool(
        per_window
        and combined["cagr_to_strict_mdd"] >= 3.0
        and combined["trades"] >= 42
    )


def assert_selection_replay(
    manifest_row: dict[str, Any],
    selection_row: dict[str, Any],
    selection_prefix_length: int,
    active: np.ndarray,
    stats: dict[str, Any],
    schedule_hashes: dict[str, str],
) -> None:
    activation_hash = _json_hash(np.flatnonzero(active[:selection_prefix_length]).tolist())
    if activation_hash != manifest_row["activation_hash"]:
        raise RuntimeError("frozen activation changed after opening future data")
    for name, _, _ in SELECTION_WINDOWS + (COMBINED_WINDOW,):
        if schedule_hashes[name] != manifest_row["schedule_hashes"][name]:
            raise RuntimeError(f"frozen {name} schedule changed")
        for key, expected in selection_row["stats"][name].items():
            actual = stats[name][key]
            if isinstance(expected, int):
                if int(actual) != expected:
                    raise RuntimeError(f"frozen {name}.{key} changed")
            elif not np.isclose(float(actual), float(expected), rtol=0.0, atol=1e-12):
                raise RuntimeError(f"frozen {name}.{key} changed")


def _render_docs(payload: dict[str, Any]) -> str:
    lines = [
        "# Frozen ExtraTrees top-10 OOS evaluation",
        "",
        f"Manifest: `{payload['manifest_hash']}`",
        "",
        f"- Future-pass candidates: `{payload['future_pass_count']}/10`",
        f"- Full-window-pass candidates: `{payload['full_pass_count']}/10`",
        f"- Frozen rank-1 future pass: `{payload['candidates'][0]['future_pass']}`",
        "",
        "| Frozen rank | 2025 abs/CAGR/MDD/ratio/trades | 2026H1 abs/CAGR/MDD/ratio/trades | Future ratio/trades | All abs/CAGR/MDD/ratio/trades | Future/full pass |",
        "| ---: | --- | --- | --- | --- | --- |",
    ]
    for row in payload["candidates"]:
        values = []
        for name in ("eval_2025", "holdout_2026h1", ALL_COMBINED[0]):
            stat = row["stats"][name]
            values.append(
                f"{stat['absolute_return_pct']:.2f}%/{stat['cagr_pct']:.2f}%/"
                f"{stat['strict_mdd_pct']:.2f}%/{stat['cagr_to_strict_mdd']:.2f}/{stat['trades']}"
            )
        future = row["stats"][FUTURE_COMBINED[0]]
        lines.append(
            f"| {row['rank_position']} | {values[0]} | {values[1]} | "
            f"{future['cagr_to_strict_mdd']:.2f}/{future['trades']} | {values[2]} | "
            f"{row['future_pass']}/{row['full_pass']} |"
        )
    lines.extend(
        [
            "",
            "## Integrity",
            "",
            "- The pre-2025 feature, source-leg, activation, schedule, and metric prefixes reproduced exactly.",
            "- Candidate order is the frozen 2023/2024 order; 2025/2026 did not rerank it.",
            "- Annual expanding refits use only labels whose source-owned exits precede each cutoff.",
            "- Prediction is deterministic (`n_jobs=1`); execution retains next-open, exact costs/funding, non-overlap, split containment, and strict MDD.",
            "",
            "## Limitation",
            "",
            "The evaluator is algorithmically isolated by a committed manifest, but this remains a retrospective reconstruction because earlier human research had viewed the future periods.",
            "",
        ]
    )
    return "\n".join(lines)


def run(manifest_path: str, output: str, docs_output: str) -> dict[str, Any]:
    manifest, selection_result = validate_manifest(manifest_path)
    selection_cfg = replace(
        Config(), exclude_from=SELECTION_CUTOFF, output="/tmp/no.json", docs_output=""
    )
    replay_cfg = replace(Config(), exclude_from=FULL_CUTOFF, output=output, docs_output="")
    selection_base = build_selection_base(selection_cfg)
    replay_base = build_replay_base(replay_cfg)
    assert_prefix_parity(selection_base, replay_base)
    if selection_result["source_hashes"] != selection_base["context"]["source_hashes"]:
        raise RuntimeError("selection source hashes changed")
    if selection_result["feature_full_hash"] != selection_base["context"]["feature_full_hash"]:
        raise RuntimeError("selection feature hash changed")

    manifest_rows = {row["rank_position"]: row for row in manifest["top10"]}
    selection_rows = {row["rank_position"]: row for row in selection_result["top10"]}
    learners = {
        json.dumps(row["learner"], sort_keys=True): LearnerSpec(**row["learner"])
        for row in manifest["top10"]
    }
    fold_cache = {
        key: fit_replay_folds(replay_base, learner) for key, learner in learners.items()
    }
    candidates = []
    prefix_length = len(selection_base["dates"])
    for rank_position in range(1, 11):
        manifest_row = manifest_rows[rank_position]
        selection_row = selection_rows[rank_position]
        if manifest_row["learner"] != selection_row["learner"] or manifest_row["selection"] != selection_row["selection"]:
            raise RuntimeError("manifest and selection result candidate mismatch")
        learner_key = json.dumps(manifest_row["learner"], sort_keys=True)
        policy = SelectionSpec(**manifest_row["selection"])
        active = activation_for(replay_base, fold_cache[learner_key], policy)
        stats, hashes = replay_stats(replay_base, active)
        assert_selection_replay(
            manifest_row,
            selection_row,
            prefix_length,
            active,
            stats,
            hashes,
        )
        candidates.append(
            {
                "rank_position": rank_position,
                "learner": manifest_row["learner"],
                "selection": asdict(policy),
                "future_pass": future_passes(stats),
                "full_pass": full_passes(stats),
                "stats": stats,
                "schedule_hashes": hashes,
            }
        )
    payload = {
        "schema_version": 1,
        "mode": "frozen_pre2025_top10_oos_evaluation",
        "manifest": manifest_path,
        "manifest_hash": manifest["manifest_hash"],
        "selection_commit": "e3b430bca2fb8504ffbc1b3f540e28bbf25c33e0",
        "future_windows": {name: [start, end] for name, start, end in FUTURE_WINDOWS},
        "candidate_order": "frozen_2023_2024_rank_no_future_reranking",
        "future_pass_count": sum(row["future_pass"] for row in candidates),
        "full_pass_count": sum(row["full_pass"] for row in candidates),
        "candidates": candidates,
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if docs_output:
        docs_path = Path(docs_output)
        docs_path.parent.mkdir(parents=True, exist_ok=True)
        docs_path.write_text(_render_docs(payload), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--docs-output", default=DEFAULT_DOCS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run(args.manifest, args.output, args.docs_output)
    print(
        json.dumps(
            {
                "output": args.output,
                "docs": args.docs_output,
                "manifest_hash": payload["manifest_hash"],
                "future_pass_count": payload["future_pass_count"],
                "full_pass_count": payload["full_pass_count"],
                "frozen_rank1": payload["candidates"][0],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
