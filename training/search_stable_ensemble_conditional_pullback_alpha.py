"""Search a stable weak-signal interaction alpha without opening 2024+.

Five large random forests are averaged to remove the seed lottery observed in
the original responsibility model.  Funding and premium events receive
source-specific score thresholds.  Funding events add one price-action
interaction: when the 28-day range is compressed, execution requires a deeper
completed-daily-bar pullback; when the range is already wide, the ensemble
score is sufficient.  The source-owned exits remain unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import itertools
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.audit_confirmed_pullback_squeeze_live_parity import selection_passes
from training.audit_weak_feature_responsibility_stability import (
    CANDIDATE_SPEC,
    FEATURE_COLUMNS,
    STABILITY_SPEC,
    Config as ResponsibilityConfig,
    _activation_hash,
    build_design,
    routed_schedule,
    schedules_and_stats,
)
from training.audit_confirmed_pullback_squeeze_live_parity import _execution_config
from training.search_inventory_purge_reclaim_alpha import ExecutionEngine, _schedule_hash
from training.search_liveparity_state_feature_interactions import immutable_anchors, net_target

SELECTION_END = "2024-01-01"
DEFAULT_OUTPUT = "results/stable_ensemble_conditional_pullback_pre2024_2026-07-15.json"
DEFAULT_DOCS = "docs/stable-ensemble-conditional-pullback-pre2024-2026-07-15.md"
SEEDS = tuple(int(seed) for seed in STABILITY_SPEC["seeds"])
TREES_PER_SEED = 2_000
FUNDING_QUANTILES = (0.3, 0.4, 0.5)
PREMIUM_QUANTILES = (0.3, 0.4, 0.5, 0.6)
LOW_WIDTH_QUANTILES = (0.2, 0.3, 0.4, 0.5, 0.6)
PULLBACK_QUANTILES = (0.3, 0.4, 0.5, 0.6)
WIDTH_FEATURE = "rex_2016_range_width_pct"
PULLBACK_FEATURE = "htf_1d_range_pos"
GRID_KEYS = ("funding_q", "premium_q", "low_width_q", "pullback_q")
GRID_VALUES = {
    "funding_q": FUNDING_QUANTILES,
    "premium_q": PREMIUM_QUANTILES,
    "low_width_q": LOW_WIDTH_QUANTILES,
    "pullback_q": PULLBACK_QUANTILES,
}
EXPECTED_SELECTED_QUANTILES = {
    "funding_q": 0.3,
    "premium_q": 0.5,
    "low_width_q": 0.2,
    "pullback_q": 0.4,
}
EXPECTED_SELECTED_STATS = {
    "train": {
        "absolute_return_pct": 104.15905243960215,
        "cagr_pct": 33.00524257304116,
        "strict_mdd_pct": 8.14837903808553,
        "cagr_to_strict_mdd": 4.05052862891805,
        "trades": 128,
    },
    "select_2023": {
        "absolute_return_pct": 11.092189809025509,
        "cagr_pct": 11.100194077308512,
        "strict_mdd_pct": 3.117323102684766,
        "cagr_to_strict_mdd": 3.560809615066392,
        "trades": 19,
    },
    "select_2023_h2": {
        "absolute_return_pct": 2.010841813806419,
        "cagr_pct": 4.031159221271974,
        "strict_mdd_pct": 2.301955963016322,
        "cagr_to_strict_mdd": 1.7511886786877644,
        "trades": 6,
    },
    "pre_2024": {
        "absolute_return_pct": 126.80476204851078,
        "cagr_pct": 26.347091790517062,
        "strict_mdd_pct": 8.14837903808553,
        "cagr_to_strict_mdd": 3.2334150961032537,
        "trades": 147,
    },
}

SEARCH_SPEC: dict[str, Any] = {
    "name": "stable_ensemble_conditional_pullback",
    "model": "mean prediction of five 2,000-tree RandomForestRegressors",
    "seeds": list(SEEDS),
    "trees_per_seed": TREES_PER_SEED,
    "prediction_reduction": "single-threaded fixed tree order after parallel deterministic fitting",
    "score_calibration": "source-specific train-only quantiles",
    "funding_quantiles": list(FUNDING_QUANTILES),
    "premium_quantiles": list(PREMIUM_QUANTILES),
    "low_width_quantiles": list(LOW_WIDTH_QUANTILES),
    "pullback_quantiles": list(PULLBACK_QUANTILES),
    "interaction": (
        "funding score passes and (28-day range width exceeds its train funding "
        "quantile or completed-daily range position is below its train funding quantile)"
    ),
    "premium_rule": "premium source score passes its train-only source quantile",
    "selection_rule": "absolute formal gate plus at least one Manhattan-adjacent passing cell",
    "grid_cells": (
        len(FUNDING_QUANTILES)
        * len(PREMIUM_QUANTILES)
        * len(LOW_WIDTH_QUANTILES)
        * len(PULLBACK_QUANTILES)
    ),
    "selection_end_exclusive": SELECTION_END,
    "oos_policy": "2024+ remains physically sealed until a separate stability audit and freeze commit",
}


@dataclass(frozen=True)
class Config(ResponsibilityConfig):
    output: str = DEFAULT_OUTPUT
    docs_output: str = DEFAULT_DOCS


def _array_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _spec_hash() -> str:
    encoded = json.dumps(SEARCH_SPEC, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _implementation_hash() -> str:
    functions = (
        ensemble_predictions,
        deterministic_forest_predict,
        fit_deterministic_stability_model,
        source_thresholds,
        conditional_activation,
        quantile_spec,
        adjacent_specs,
        search_payload,
        build_design,
        routed_schedule,
        schedules_and_stats,
    )
    return hashlib.sha256("\n\n".join(inspect.getsource(fn) for fn in functions).encode()).hexdigest()


def ensemble_predictions(fits: list[dict[str, Any]]) -> dict[str, Any]:
    """Average aligned train and anchor predictions from fitted forests."""
    if not fits:
        raise ValueError("at least one fitted model is required")
    first = fits[0]
    for fitted in fits[1:]:
        if not np.array_equal(fitted["train_positions"], first["train_positions"]):
            raise RuntimeError("ensemble train positions differ")
        if not np.array_equal(fitted["anchor_positions"], first["anchor_positions"]):
            raise RuntimeError("ensemble anchor positions differ")
    return {
        "engine": first["engine"],
        "train_positions": np.asarray(first["train_positions"], dtype=np.int64),
        "anchor_positions": np.asarray(first["anchor_positions"], dtype=np.int64),
        "train_predictions": np.mean(
            np.stack([np.asarray(fitted["train_predictions"], dtype=float) for fitted in fits]), axis=0
        ),
        "anchor_predictions": np.mean(
            np.stack([np.asarray(fitted["anchor_predictions"], dtype=float) for fitted in fits]), axis=0
        ),
    }


def deterministic_forest_predict(model: RandomForestRegressor, matrix: np.ndarray) -> np.ndarray:
    """Accumulate tree predictions in fixed order instead of a thread race."""
    model.n_jobs = 1
    return np.asarray(model.predict(matrix), dtype=float)


def fit_deterministic_stability_model(
    context: dict[str, Any],
    cfg: Config,
    *,
    trees: int,
    seed: int,
) -> dict[str, Any]:
    """Fit in parallel, then predict sequentially to avoid ULP-level rank flips."""
    engine = ExecutionEngine(context["market"], context["funding"], _execution_config(cfg, cfg.leverage))
    anchors = immutable_anchors(context["base"], int(CANDIDATE_SPEC["anchor_cooldown_bars"]))
    positions = np.flatnonzero(anchors & context["fit"])
    targets = np.asarray([net_target(engine, int(pos), 576, cfg) for pos in positions], dtype=float)
    finite = np.isfinite(targets)
    positions, targets = positions[finite], targets[finite]
    model = RandomForestRegressor(
        n_estimators=int(trees),
        max_depth=int(CANDIDATE_SPEC["max_depth"]),
        min_samples_leaf=int(CANDIDATE_SPEC["min_samples_leaf"]),
        max_features=float(CANDIDATE_SPEC["max_features"]),
        random_state=int(seed),
        n_jobs=-1,
    ).fit(context["matrix"][positions], targets)
    anchor_positions = np.flatnonzero(anchors)
    return {
        "engine": engine,
        "train_positions": positions,
        "anchor_positions": anchor_positions,
        "train_predictions": deterministic_forest_predict(model, context["matrix"][positions]),
        "anchor_predictions": deterministic_forest_predict(model, context["matrix"][anchor_positions]),
    }


def source_thresholds(
    train_predictions: np.ndarray,
    train_is_funding: np.ndarray,
    *,
    funding_q: float,
    premium_q: float,
) -> tuple[float, float]:
    """Calibrate each event source only from its own fit-window predictions."""
    predictions = np.asarray(train_predictions, dtype=float)
    funding = np.asarray(train_is_funding, dtype=bool)
    if predictions.shape != funding.shape or not funding.any() or funding.all():
        raise ValueError("both aligned funding and premium train examples are required")
    return (
        float(np.quantile(predictions[funding], funding_q)),
        float(np.quantile(predictions[~funding], premium_q)),
    )


def conditional_activation(
    *,
    size: int,
    anchor_positions: np.ndarray,
    anchor_predictions: np.ndarray,
    anchor_is_funding: np.ndarray,
    anchor_width: np.ndarray,
    anchor_pullback: np.ndarray,
    funding_threshold: float,
    premium_threshold: float,
    width_threshold: float,
    pullback_threshold: float,
) -> np.ndarray:
    """Apply the source score gates and compressed-range pullback interaction."""
    positions = np.asarray(anchor_positions, dtype=np.int64)
    scores = np.asarray(anchor_predictions, dtype=float)
    funding = np.asarray(anchor_is_funding, dtype=bool)
    width = np.asarray(anchor_width, dtype=float)
    pullback = np.asarray(anchor_pullback, dtype=float)
    if not (len(positions) == len(scores) == len(funding) == len(width) == len(pullback)):
        raise ValueError("anchor arrays must be aligned")
    funding_active = funding & (scores >= funding_threshold) & (
        (width > width_threshold) | (pullback <= pullback_threshold)
    )
    premium_active = (~funding) & (scores >= premium_threshold)
    active = np.zeros(int(size), dtype=bool)
    active[positions] = funding_active | premium_active
    return active


def quantile_spec(row: dict[str, Any]) -> tuple[float, float, float, float]:
    spec = row["spec"] if "spec" in row else row
    return tuple(float(spec[key]) for key in GRID_KEYS)


def adjacent_specs(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Return true for one-grid-step Manhattan neighbors."""
    left_values = quantile_spec(left)
    right_values = quantile_spec(right)
    distance = 0
    for key, lhs, rhs in zip(GRID_KEYS, left_values, right_values, strict=True):
        grid = GRID_VALUES[key]
        lhs_index = grid.index(lhs)
        rhs_index = grid.index(rhs)
        distance += abs(lhs_index - rhs_index)
    return distance == 1


def _candidate_rank(row: dict[str, Any]) -> tuple[float, float, int]:
    stats = row["stats"]
    return (
        float(stats["pre_2024"]["cagr_to_strict_mdd"]),
        float(stats["select_2023"]["cagr_to_strict_mdd"]),
        int(stats["pre_2024"]["trades"]),
    )


def _search_rows(context: dict[str, Any], ensemble: dict[str, Any], cfg: Config) -> list[dict[str, Any]]:
    train_positions = ensemble["train_positions"]
    anchor_positions = ensemble["anchor_positions"]
    funding_leg = np.asarray(context["funding_leg"], dtype=bool)
    train_funding = funding_leg[train_positions]
    anchor_funding = funding_leg[anchor_positions]
    width_index = FEATURE_COLUMNS.index(WIDTH_FEATURE)
    pullback_index = FEATURE_COLUMNS.index(PULLBACK_FEATURE)
    train_width = np.asarray(context["matrix"][train_positions, width_index], dtype=float)
    train_pullback = np.asarray(context["matrix"][train_positions, pullback_index], dtype=float)
    anchor_width = np.asarray(context["matrix"][anchor_positions, width_index], dtype=float)
    anchor_pullback = np.asarray(context["matrix"][anchor_positions, pullback_index], dtype=float)

    rows: list[dict[str, Any]] = []
    for funding_q, premium_q, low_width_q, pullback_q in itertools.product(
        FUNDING_QUANTILES,
        PREMIUM_QUANTILES,
        LOW_WIDTH_QUANTILES,
        PULLBACK_QUANTILES,
    ):
        funding_threshold, premium_threshold = source_thresholds(
            ensemble["train_predictions"],
            train_funding,
            funding_q=funding_q,
            premium_q=premium_q,
        )
        width_threshold = float(np.quantile(train_width[train_funding], low_width_q))
        pullback_threshold = float(np.quantile(train_pullback[train_funding], pullback_q))
        active = conditional_activation(
            size=len(context["market"]),
            anchor_positions=anchor_positions,
            anchor_predictions=ensemble["anchor_predictions"],
            anchor_is_funding=anchor_funding,
            anchor_width=anchor_width,
            anchor_pullback=anchor_pullback,
            funding_threshold=funding_threshold,
            premium_threshold=premium_threshold,
            width_threshold=width_threshold,
            pullback_threshold=pullback_threshold,
        )
        fitted = {"engine": ensemble["engine"], "active": active}
        schedules, stats = schedules_and_stats(context, fitted, cfg)
        rows.append(
            {
                "spec": {
                    "funding_q": funding_q,
                    "premium_q": premium_q,
                    "low_width_q": low_width_q,
                    "pullback_q": pullback_q,
                    "funding_threshold": funding_threshold,
                    "premium_threshold": premium_threshold,
                    "width_threshold": width_threshold,
                    "pullback_threshold": pullback_threshold,
                },
                "pass": bool(selection_passes(stats)),
                "activation_hash": _activation_hash(active, context["dates"]),
                "schedule_hashes": {name: _schedule_hash(trades) for name, trades in schedules.items()},
                "stats": stats,
            }
        )
    return rows


def _select_cluster_candidate(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    passing = [row for row in rows if row["pass"]]
    for row in passing:
        row["adjacent_passing_specs"] = [
            other["spec"] for other in passing if other is not row and adjacent_specs(row, other)
        ]
    clustered = [row for row in passing if row["adjacent_passing_specs"]]
    if not clustered:
        raise RuntimeError("no adjacent formal-gate passing cluster found")
    selected = max(clustered, key=_candidate_rank)
    return selected, passing


def _result_payload(payload: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "phase",
        "oos_opened",
        "sealed_windows",
        "selection_end_exclusive",
        "search_spec",
        "spec_hash",
        "implementation_hash",
        "source_prefix_hashes",
        "feature_prefix_hash",
        "model",
        "search_summary",
        "selected_candidate",
        "passing_cells",
        "top_rows",
    )
    return {key: payload[key] for key in keys}


def _result_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(_result_payload(payload), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _assert_expected_candidate(candidate: dict[str, Any]) -> None:
    for key, expected in EXPECTED_SELECTED_QUANTILES.items():
        if float(candidate["spec"][key]) != expected:
            raise RuntimeError(f"selected {key} drifted: {candidate['spec'][key]} != {expected}")
    for window, expected_stats in EXPECTED_SELECTED_STATS.items():
        actual = candidate["stats"][window]
        for key, expected in expected_stats.items():
            if isinstance(expected, int):
                if int(actual[key]) != expected:
                    raise RuntimeError(f"{window}.{key} drifted: {actual[key]} != {expected}")
            elif not np.isclose(float(actual[key]), float(expected), rtol=0.0, atol=1e-10):
                raise RuntimeError(f"{window}.{key} drifted: {actual[key]} != {expected}")


def search_payload(cfg: Config) -> dict[str, Any]:
    context = build_design(cfg)
    if len(context["dates"]) and context["dates"].max() >= pd.Timestamp(SELECTION_END):
        raise RuntimeError("search source was not physically truncated before 2024")
    fits = [
        fit_deterministic_stability_model(context, cfg, trees=TREES_PER_SEED, seed=seed)
        for seed in SEEDS
    ]
    ensemble = ensemble_predictions(fits)
    rows = _search_rows(context, ensemble, cfg)
    selected, passing = _select_cluster_candidate(rows)
    _assert_expected_candidate(selected)
    ordered = sorted(rows, key=_candidate_rank, reverse=True)
    payload: dict[str, Any] = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "phase": "pre_2024_stable_ensemble_conditional_pullback_search",
        "oos_opened": False,
        "sealed_windows": ["2024+"],
        "selection_end_exclusive": SELECTION_END,
        "config": asdict(cfg),
        "search_spec": SEARCH_SPEC,
        "spec_hash": _spec_hash(),
        "implementation_hash": _implementation_hash(),
        "source_prefix_hashes": context["source_hashes"],
        "feature_prefix_hash": context["feature_prefix_hash"],
        "model": {
            "seeds": list(SEEDS),
            "trees_per_seed": TREES_PER_SEED,
            "total_trees": len(SEEDS) * TREES_PER_SEED,
            "train_examples": int(len(ensemble["train_positions"])),
            "anchor_examples": int(len(ensemble["anchor_positions"])),
            "train_prediction_hash": _array_hash(ensemble["train_predictions"]),
            "anchor_prediction_hash": _array_hash(ensemble["anchor_predictions"]),
        },
        "search_summary": {
            "cells": len(rows),
            "formal_gate_passes": len(passing),
            "adjacent_cluster_passes": sum(bool(row["adjacent_passing_specs"]) for row in passing),
            "decision": "selection_candidate_found",
            "promotion_status": "requires separate stability audit and freeze before OOS",
        },
        "selected_candidate": selected,
        "passing_cells": sorted(passing, key=_candidate_rank, reverse=True),
        "top_rows": ordered[:120],
    }
    payload["result_hash"] = _result_hash(payload)
    return payload


def validate_result(payload: dict[str, Any]) -> None:
    if payload.get("oos_opened") is not False or payload.get("sealed_windows") != ["2024+"]:
        raise RuntimeError("search must keep 2024+ sealed")
    if payload.get("search_spec") != SEARCH_SPEC:
        raise RuntimeError("search specification changed")
    if payload.get("spec_hash") != _spec_hash() or payload.get("implementation_hash") != _implementation_hash():
        raise RuntimeError("search implementation changed")
    if payload.get("result_hash") != _result_hash(payload):
        raise RuntimeError("result hash mismatch")
    summary = payload["search_summary"]
    if summary["cells"] != SEARCH_SPEC["grid_cells"] or summary["formal_gate_passes"] != 6:
        raise RuntimeError("expected six passing cells in the complete 240-cell search")
    selected = payload["selected_candidate"]
    if not selected["pass"] or not selected["adjacent_passing_specs"]:
        raise RuntimeError("selected candidate lacks an adjacent formal-gate pass")
    if not selection_passes(selected["stats"]):
        raise RuntimeError("selected candidate no longer passes the formal gate")
    _assert_expected_candidate(selected)


def _atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    temporary.replace(target)


def _metric(row: dict[str, Any]) -> str:
    return (
        f"{row['absolute_return_pct']:.2f}% / {row['cagr_pct']:.2f}% / "
        f"{row['strict_mdd_pct']:.2f}% / {row['cagr_to_strict_mdd']:.2f} / {row['trades']}"
    )


def _write_docs(path: str | Path, payload: dict[str, Any]) -> None:
    candidate = payload["selected_candidate"]
    spec = candidate["spec"]
    lines = [
        "# Stable ensemble conditional-pullback selection — 2026-07-15",
        "",
        "**Selection candidate found; 2024+ remains sealed. This is not yet an OOS claim.**",
        "",
        "Metric: absolute return / CAGR / strict MDD / CAGR-to-strict-MDD / trades.",
        "",
        "## Weak-signal interaction",
        "",
        "- Average five independently seeded 2,000-tree shallow forests; this removes the earlier single-seed lottery.",
        "- Calibrate funding and premium score thresholds independently from fit-window examples only.",
        "- Premium events use only their source score threshold.",
        "- Funding events use the score threshold, then require either a non-compressed 28-day range or a deep completed-daily-bar pullback.",
        "- Economic interpretation: avoid ordinary funding entries inside quiet ranges; permit them only at a range extreme, while retaining high-dispersion opportunities.",
        "- Source-owned exits are unchanged: funding 48h/4% take/no stop; premium 12h/no take/3% stop.",
        "- All thresholds are computed on 2020-07-01..2022-12-31; selection is 2023; physical source cutoff is 2024-01-01.",
        "",
        "## Search evidence",
        "",
        f"- Complete cells: **{payload['search_summary']['cells']}**.",
        f"- Formal-gate passes: **{payload['search_summary']['formal_gate_passes']}**.",
        f"- Passing cells with an adjacent pass: **{payload['search_summary']['adjacent_cluster_passes']}**.",
        "- All six passes form one contiguous 3×2 neighborhood; no isolated winner was selected.",
        "",
        "## Selected pre-OOS specification",
        "",
        f"- Funding score quantile: `{spec['funding_q']}`; premium score quantile: `{spec['premium_q']}`.",
        f"- Low-width quantile: `{spec['low_width_q']}` (`{spec['width_threshold']:.8f}`).",
        f"- Pullback quantile: `{spec['pullback_q']}` (`{spec['pullback_threshold']:.8f}`).",
        "",
        "| Window | Result |",
        "|---|---:|",
    ]
    for name in (
        "train",
        "train_2020h2",
        "train_2021",
        "train_2022",
        "select_2023",
        "select_2023_h1",
        "select_2023_h2",
        "pre_2024",
    ):
        lines.append(f"| {name} | {_metric(candidate['stats'][name])} |")
    lines += [
        "",
        "## Stop condition before OOS",
        "",
        "The exact selected rule must survive per-seed, ensemble-size, one-hour-delay, and interaction-ablation audits. Only then may a freeze commit be created and 2024+ opened once.",
        "",
        f"Result hash: `{payload['result_hash']}`",
        "",
    ]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")


def run(cfg: Config) -> dict[str, Any]:
    payload = search_payload(cfg)
    validate_result(payload)
    _atomic_write_json(cfg.output, payload)
    _write_docs(cfg.docs_output, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", default=Config.input_csv)
    parser.add_argument("--funding-csv", default=Config.funding_csv)
    parser.add_argument("--premium-csv", default=Config.premium_csv)
    parser.add_argument("--spot-premium-csv", default=Config.spot_premium_csv)
    parser.add_argument("--output", default=Config.output)
    parser.add_argument("--docs-output", default=Config.docs_output)
    return parser.parse_args()


def main() -> None:
    payload = run(Config(**vars(parse_args())))
    print(
        json.dumps(
            {
                "phase": payload["phase"],
                "result_hash": payload["result_hash"],
                "search_summary": payload["search_summary"],
                "selected_spec": payload["selected_candidate"]["spec"],
                "selected_stats": payload["selected_candidate"]["stats"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
