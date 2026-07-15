"""Audit the pre-2024 conditional-pullback candidate before any OOS access.

The exact quantile rule is tested across model seeds and forest sizes, three-
and five-model ensembles, a one-hour information-delay control, and interaction
ablations.  All source loading remains physically truncated before 2024.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.audit_confirmed_pullback_squeeze_live_parity import _execution_config, selection_passes
from training.search_inventory_purge_reclaim_alpha import ExecutionEngine, _schedule_hash
from training.search_stable_ensemble_conditional_pullback_alpha import (
    EXPECTED_SELECTED_QUANTILES,
    EXPECTED_SELECTED_STATS,
    FEATURE_COLUMNS,
    PULLBACK_FEATURE,
    SEEDS,
    WIDTH_FEATURE,
    Config as SearchConfig,
    _activation_hash,
    _array_hash,
    build_design,
    conditional_activation,
    ensemble_predictions,
    fit_deterministic_stability_model,
    schedules_and_stats,
    source_thresholds,
)

SELECTION_END = "2024-01-01"
DEFAULT_OUTPUT = "results/stable_ensemble_conditional_pullback_audit_pre2024_2026-07-15.json"
DEFAULT_DOCS = "docs/stable-ensemble-conditional-pullback-audit-pre2024-2026-07-15.md"
TREE_COUNTS = (1_000, 2_000)
DELAY_BARS = 12
SOURCE_COLUMNS = ("funding_leg", "premium_leg")
ABLATION_MODES = (
    "conditional",
    "source_only",
    "unconditional_pullback",
    "width_only",
    "reversed_pullback",
)

AUDIT_SPEC: dict[str, Any] = {
    "name": "stable_ensemble_conditional_pullback_audit",
    "candidate_quantiles": EXPECTED_SELECTED_QUANTILES,
    "seeds": list(SEEDS),
    "tree_counts": list(TREE_COUNTS),
    "ensemble_sizes": [3, 5],
    "minimum_individual_passes": {"1000": 3, "2000": 3},
    "delay_control": {
        "bars": DELAY_BARS,
        "duration": "1h",
        "trees_per_seed": 1_000,
        "source_identity_columns_remain_current": list(SOURCE_COLUMNS),
        "minimum_individual_passes": 3,
        "require_five_model_ensemble_pass": True,
    },
    "ablations": list(ABLATION_MODES),
    "promotion_rule": (
        "both forest sizes meet individual pass minimums; 3/5-model ensembles pass; "
        "the delayed five-model ensemble passes; exact interaction passes while every simplification fails"
    ),
    "selection_end_exclusive": SELECTION_END,
    "oos_policy": "2024+ remains sealed; a later freeze commit is mandatory",
}


@dataclass(frozen=True)
class Config(SearchConfig):
    output: str = DEFAULT_OUTPUT
    docs_output: str = DEFAULT_DOCS


def _spec_hash() -> str:
    encoded = json.dumps(AUDIT_SPEC, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _implementation_hash() -> str:
    functions = (
        delayed_feature_context,
        interaction_mask,
        evaluate_model,
        stability_decision,
        audit_payload,
        build_design,
        fit_deterministic_stability_model,
        ensemble_predictions,
        source_thresholds,
        conditional_activation,
        schedules_and_stats,
    )
    return hashlib.sha256("\n\n".join(inspect.getsource(fn) for fn in functions).encode()).hexdigest()


def delayed_feature_context(context: dict[str, Any], bars: int = DELAY_BARS) -> dict[str, Any]:
    """Delay predictive features while keeping current event-source identity."""
    if bars <= 0:
        raise ValueError("delay must be positive")
    matrix = np.asarray(context["matrix"], dtype=float)
    delayed = np.empty_like(matrix)
    delayed[:bars] = matrix[0]
    delayed[bars:] = matrix[:-bars]
    for name in SOURCE_COLUMNS:
        column = FEATURE_COLUMNS.index(name)
        delayed[:, column] = matrix[:, column]
    output = dict(context)
    output["matrix"] = delayed
    return output


def interaction_mask(
    mode: str,
    *,
    width: np.ndarray,
    pullback: np.ndarray,
    width_threshold: float,
    pullback_threshold: float,
) -> np.ndarray:
    """Return the funding-side interaction or one explicit ablation."""
    width = np.asarray(width, dtype=float)
    pullback = np.asarray(pullback, dtype=float)
    if mode == "conditional":
        return (width > width_threshold) | (pullback <= pullback_threshold)
    if mode == "source_only":
        return np.ones(len(width), dtype=bool)
    if mode == "unconditional_pullback":
        return pullback <= pullback_threshold
    if mode == "width_only":
        return width > width_threshold
    if mode == "reversed_pullback":
        return (width > width_threshold) | (pullback >= pullback_threshold)
    raise ValueError(f"unknown interaction mode: {mode}")


def evaluate_model(
    context: dict[str, Any],
    model: dict[str, Any],
    cfg: Config,
    *,
    label: str,
    trees: int,
    seeds: list[int],
    mode: str = "conditional",
) -> dict[str, Any]:
    train_positions = np.asarray(model["train_positions"], dtype=np.int64)
    anchor_positions = np.asarray(model["anchor_positions"], dtype=np.int64)
    funding_leg = np.asarray(context["funding_leg"], dtype=bool)
    train_funding = funding_leg[train_positions]
    anchor_funding = funding_leg[anchor_positions]
    matrix = np.asarray(context["matrix"], dtype=float)
    width = matrix[:, FEATURE_COLUMNS.index(WIDTH_FEATURE)]
    pullback = matrix[:, FEATURE_COLUMNS.index(PULLBACK_FEATURE)]
    quantiles = EXPECTED_SELECTED_QUANTILES
    funding_threshold, premium_threshold = source_thresholds(
        model["train_predictions"],
        train_funding,
        funding_q=quantiles["funding_q"],
        premium_q=quantiles["premium_q"],
    )
    width_threshold = float(
        np.quantile(width[train_positions][train_funding], quantiles["low_width_q"])
    )
    pullback_threshold = float(
        np.quantile(pullback[train_positions][train_funding], quantiles["pullback_q"])
    )
    if mode == "conditional":
        active = conditional_activation(
            size=len(context["market"]),
            anchor_positions=anchor_positions,
            anchor_predictions=model["anchor_predictions"],
            anchor_is_funding=anchor_funding,
            anchor_width=width[anchor_positions],
            anchor_pullback=pullback[anchor_positions],
            funding_threshold=funding_threshold,
            premium_threshold=premium_threshold,
            width_threshold=width_threshold,
            pullback_threshold=pullback_threshold,
        )
    else:
        extra = interaction_mask(
            mode,
            width=width[anchor_positions],
            pullback=pullback[anchor_positions],
            width_threshold=width_threshold,
            pullback_threshold=pullback_threshold,
        )
        predictions = np.asarray(model["anchor_predictions"], dtype=float)
        selected = (
            anchor_funding & (predictions >= funding_threshold) & extra
        ) | ((~anchor_funding) & (predictions >= premium_threshold))
        active = np.zeros(len(context["market"]), dtype=bool)
        active[anchor_positions] = selected
    engine = model.get("engine")
    if engine is None:
        engine = ExecutionEngine(context["market"], context["funding"], _execution_config(cfg, cfg.leverage))
    schedules, stats = schedules_and_stats(context, {"engine": engine, "active": active}, cfg)
    return {
        "label": label,
        "mode": mode,
        "trees": int(trees),
        "seeds": list(seeds),
        "thresholds": {
            "funding": funding_threshold,
            "premium": premium_threshold,
            "width": width_threshold,
            "pullback": pullback_threshold,
        },
        "passes_selection": bool(selection_passes(stats)),
        "activation_hash": _activation_hash(active, context["dates"]),
        "schedule_hashes": {name: _schedule_hash(trades) for name, trades in schedules.items()},
        "stats": stats,
    }


def stability_decision(
    *,
    individual_passes: dict[int, int],
    ensembles_pass: dict[int, bool],
    delayed_individual_passes: int,
    delayed_ensemble_pass: bool,
    ablation_passes: dict[str, bool],
) -> bool:
    minimums = AUDIT_SPEC["minimum_individual_passes"]
    forest_ok = all(individual_passes[count] >= int(minimums[str(count)]) for count in TREE_COUNTS)
    ensembles_ok = all(ensembles_pass.values())
    delay_spec = AUDIT_SPEC["delay_control"]
    delay_ok = (
        delayed_individual_passes >= int(delay_spec["minimum_individual_passes"])
        and delayed_ensemble_pass
    )
    ablation_ok = ablation_passes.get("conditional") is True and not any(
        passed for mode, passed in ablation_passes.items() if mode != "conditional"
    )
    return bool(forest_ok and ensembles_ok and delay_ok and ablation_ok)


def _assert_selected_stats(stats: dict[str, dict[str, Any]]) -> None:
    for window, expected in EXPECTED_SELECTED_STATS.items():
        for key, value in expected.items():
            actual = stats[window][key]
            if isinstance(value, int):
                if int(actual) != value:
                    raise RuntimeError(f"{window}.{key} drifted: {actual} != {value}")
            elif not np.isclose(float(actual), float(value), rtol=0.0, atol=1e-10):
                raise RuntimeError(f"{window}.{key} drifted: {actual} != {value}")


def _audit_payload_hashable(payload: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "phase",
        "oos_opened",
        "sealed_windows",
        "selection_end_exclusive",
        "audit_spec",
        "spec_hash",
        "implementation_hash",
        "source_prefix_hashes",
        "feature_prefix_hash",
        "forest_size_runs",
        "delay_control",
        "ablations",
        "stability_summary",
    )
    return {key: payload[key] for key in keys}


def _audit_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        _audit_payload_hashable(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def audit_payload(cfg: Config) -> dict[str, Any]:
    context = build_design(cfg)
    forest_runs: dict[str, Any] = {}
    fitted_by_size: dict[int, list[dict[str, Any]]] = {}
    individual_passes: dict[int, int] = {}
    ensembles_pass: dict[int, bool] = {}
    for trees in TREE_COUNTS:
        fits = [
            fit_deterministic_stability_model(context, cfg, trees=trees, seed=seed)
            for seed in SEEDS
        ]
        fitted_by_size[trees] = fits
        individual = [
            evaluate_model(
                context,
                ensemble_predictions([fitted]),
                cfg,
                label=f"seed_{seed}_{trees}",
                trees=trees,
                seeds=[seed],
            )
            for seed, fitted in zip(SEEDS, fits, strict=True)
        ]
        mean_three = evaluate_model(
            context,
            ensemble_predictions(fits[:3]),
            cfg,
            label=f"mean_3x{trees}",
            trees=3 * trees,
            seeds=list(SEEDS[:3]),
        )
        mean_five = evaluate_model(
            context,
            ensemble_predictions(fits),
            cfg,
            label=f"mean_5x{trees}",
            trees=5 * trees,
            seeds=list(SEEDS),
        )
        individual_passes[trees] = sum(int(row["passes_selection"]) for row in individual)
        ensembles_pass[trees] = bool(mean_three["passes_selection"] and mean_five["passes_selection"])
        forest_runs[str(trees)] = {
            "individual": individual,
            "mean_three": mean_three,
            "mean_five": mean_five,
        }

    selected = forest_runs["2000"]["mean_five"]
    _assert_selected_stats(selected["stats"])

    delayed_context = delayed_feature_context(context)
    delayed_fits = [
        fit_deterministic_stability_model(delayed_context, cfg, trees=1_000, seed=seed)
        for seed in SEEDS
    ]
    delayed_individual = [
        evaluate_model(
            delayed_context,
            ensemble_predictions([fitted]),
            cfg,
            label=f"delay_1h_seed_{seed}_1000",
            trees=1_000,
            seeds=[seed],
        )
        for seed, fitted in zip(SEEDS, delayed_fits, strict=True)
    ]
    delayed_mean = evaluate_model(
        delayed_context,
        ensemble_predictions(delayed_fits),
        cfg,
        label="delay_1h_mean_5x1000",
        trees=5_000,
        seeds=list(SEEDS),
    )
    delayed_passes = sum(int(row["passes_selection"]) for row in delayed_individual)

    five_model = ensemble_predictions(fitted_by_size[2_000])
    ablations = [
        evaluate_model(
            context,
            five_model,
            cfg,
            label=f"ablation_{mode}",
            trees=10_000,
            seeds=list(SEEDS),
            mode=mode,
        )
        for mode in ABLATION_MODES
    ]
    ablation_passes = {row["mode"]: bool(row["passes_selection"]) for row in ablations}
    promotable = stability_decision(
        individual_passes=individual_passes,
        ensembles_pass=ensembles_pass,
        delayed_individual_passes=delayed_passes,
        delayed_ensemble_pass=bool(delayed_mean["passes_selection"]),
        ablation_passes=ablation_passes,
    )
    payload: dict[str, Any] = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "phase": "pre_2024_stable_ensemble_conditional_pullback_audit",
        "oos_opened": False,
        "sealed_windows": ["2024+"],
        "selection_end_exclusive": SELECTION_END,
        "config": asdict(cfg),
        "audit_spec": AUDIT_SPEC,
        "spec_hash": _spec_hash(),
        "implementation_hash": _implementation_hash(),
        "source_prefix_hashes": context["source_hashes"],
        "feature_prefix_hash": context["feature_prefix_hash"],
        "forest_size_runs": forest_runs,
        "delay_control": {
            "bars": DELAY_BARS,
            "matrix_hash": _array_hash(delayed_context["matrix"]),
            "individual": delayed_individual,
            "mean_five": delayed_mean,
        },
        "ablations": ablations,
        "stability_summary": {
            "individual_passes": {str(key): value for key, value in individual_passes.items()},
            "individual_cells": len(SEEDS),
            "ensemble_3_and_5_pass": {str(key): value for key, value in ensembles_pass.items()},
            "delay_individual_passes": delayed_passes,
            "delay_individual_cells": len(SEEDS),
            "delay_five_model_pass": bool(delayed_mean["passes_selection"]),
            "ablation_passes": ablation_passes,
            "promotable": promotable,
            "decision": "promote_to_freeze" if promotable else "reject",
        },
    }
    if not promotable:
        raise RuntimeError("candidate failed its pre-OOS stability contract")
    payload["audit_hash"] = _audit_hash(payload)
    return payload


def validate_audit(payload: dict[str, Any]) -> None:
    if payload.get("oos_opened") is not False or payload.get("sealed_windows") != ["2024+"]:
        raise RuntimeError("audit must keep 2024+ sealed")
    if payload.get("audit_spec") != AUDIT_SPEC:
        raise RuntimeError("audit specification changed")
    if payload.get("spec_hash") != _spec_hash() or payload.get("implementation_hash") != _implementation_hash():
        raise RuntimeError("audit implementation changed")
    if payload.get("audit_hash") != _audit_hash(payload):
        raise RuntimeError("audit hash mismatch")
    summary = payload["stability_summary"]
    if not summary["promotable"] or summary["decision"] != "promote_to_freeze":
        raise RuntimeError("candidate is not promotable")
    _assert_selected_stats(payload["forest_size_runs"]["2000"]["mean_five"]["stats"])


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
    summary = payload["stability_summary"]
    selected = payload["forest_size_runs"]["2000"]["mean_five"]
    delayed = payload["delay_control"]["mean_five"]
    lines = [
        "# Stable ensemble conditional-pullback audit — 2026-07-15",
        "",
        "**Pre-OOS audit passed; 2024+ is still sealed until a separate freeze commit.**",
        "",
        "Metric: absolute return / CAGR / strict MDD / CAGR-to-strict-MDD / trades.",
        "",
        "## Stability verdict",
        "",
        f"- 1,000-tree individual seeds passing: **{summary['individual_passes']['1000']}/{summary['individual_cells']}**.",
        f"- 2,000-tree individual seeds passing: **{summary['individual_passes']['2000']}/{summary['individual_cells']}**.",
        f"- Three- and five-model ensembles passing at both sizes: **{all(summary['ensemble_3_and_5_pass'].values())}**.",
        f"- One-hour-delay individual seeds passing: **{summary['delay_individual_passes']}/{summary['delay_individual_cells']}**.",
        f"- One-hour-delay five-model ensemble passing: **{summary['delay_five_model_pass']}**.",
        f"- Decision: **{summary['decision']}**.",
        "",
        "## Exact selected model",
        "",
        "| Window | Result |",
        "|---|---:|",
    ]
    for name in ("train", "select_2023", "select_2023_h1", "select_2023_h2", "pre_2024"):
        lines.append(f"| {name} | {_metric(selected['stats'][name])} |")
    lines += [
        "",
        "## One-hour information-delay control",
        "",
        "All predictive inputs except current funding/premium event identity were shifted 12×5m rows before refitting. The delayed model still passed:",
        "",
        "| Window | Result |",
        "|---|---:|",
    ]
    for name in ("train", "select_2023", "select_2023_h2", "pre_2024"):
        lines.append(f"| {name} | {_metric(delayed['stats'][name])} |")
    lines += [
        "",
        "## Interaction ablations",
        "",
        "| Rule | Pass | train | 2023 | pre-2024 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in payload["ablations"]:
        stats = row["stats"]
        lines.append(
            f"| {row['mode']} | {row['passes_selection']} | {_metric(stats['train'])} | "
            f"{_metric(stats['select_2023'])} | {_metric(stats['pre_2024'])} |"
        )
    lines += [
        "",
        "Only the conditional combination passes. Score-only admits weak quiet-range entries; pullback-only destroys train/pre-2024 breadth; width-only loses 2023 H2 count; reversed pullback loses 2023 risk-adjusted return.",
        "",
        f"Audit hash: `{payload['audit_hash']}`",
        "",
    ]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")


def run(cfg: Config) -> dict[str, Any]:
    payload = audit_payload(cfg)
    validate_audit(payload)
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
                "audit_hash": payload["audit_hash"],
                "stability_summary": payload["stability_summary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
