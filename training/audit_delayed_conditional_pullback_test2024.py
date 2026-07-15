"""Audit the one-hour-delayed conditional-pullback alpha through 2024 only.

2020-2022 remain model fit/train, 2023 remains selection, and 2024 is the
single physically opened test year.  This module has no path that loads 2025+.
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
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.audit_confirmed_pullback_squeeze_live_parity import _execution_config, selection_passes
from training.audit_stable_ensemble_conditional_pullback_alpha import (
    DELAY_BARS,
    delayed_feature_context,
    evaluate_model,
)
from training.evaluate_stable_ensemble_conditional_pullback_oos import (
    Config as EvaluatorConfig,
    build_full_design,
)
from training.search_inventory_purge_reclaim_alpha import _schedule_hash, equity_stats
from training.search_liveparity_state_feature_interactions import slim
from training.search_stable_ensemble_conditional_pullback_alpha import (
    EXPECTED_SELECTED_QUANTILES,
    FEATURE_COLUMNS,
    PULLBACK_FEATURE,
    SEEDS,
    WIDTH_FEATURE,
    _activation_hash,
    _array_hash,
    conditional_activation,
    ensemble_predictions,
    fit_deterministic_stability_model,
    routed_schedule,
    source_thresholds,
)

TEST_END = "2025-01-01"
DEFAULT_OUTPUT = "results/delayed_conditional_pullback_test2024_audit_2026-07-15.json"
DEFAULT_DOCS = "docs/delayed-conditional-pullback-test2024-audit-2026-07-15.md"
TREE_COUNTS = (1_000, 2_000)
EXPECTED_SELECTED_STATS = {
    "train": {
        "absolute_return_pct": 105.65210329514696,
        "cagr_pct": 33.393096310062376,
        "strict_mdd_pct": 8.019248776994313,
        "cagr_to_strict_mdd": 4.16411776697348,
        "trades": 126,
    },
    "select_2023": {
        "absolute_return_pct": 11.994063880178496,
        "cagr_pct": 12.002753398129462,
        "strict_mdd_pct": 3.117323102684788,
        "cagr_to_strict_mdd": 3.85033985979577,
        "trades": 26,
    },
    "pre_2024": {
        "absolute_return_pct": 130.31814793529745,
        "cagr_pct": 26.90295756914751,
        "strict_mdd_pct": 8.019248776994313,
        "cagr_to_strict_mdd": 3.3547977269799802,
        "trades": 152,
    },
    "test_2024": {
        "absolute_return_pct": 16.313145371549886,
        "cagr_pct": 16.277132986031017,
        "strict_mdd_pct": 4.619940109081034,
        "cagr_to_strict_mdd": 3.5232346311235516,
        "trades": 27,
    },
}

AUDIT_SPEC: dict[str, Any] = {
    "name": "delayed_conditional_pullback_test2024",
    "information_delay_bars": DELAY_BARS,
    "information_delay": "1h",
    "current_source_identity_columns": ["funding_leg", "premium_leg"],
    "candidate_quantiles": EXPECTED_SELECTED_QUANTILES,
    "seeds": list(SEEDS),
    "tree_counts": list(TREE_COUNTS),
    "ensemble_sizes": [3, 5],
    "minimum_individual_joint_passes": {"1000": 3, "2000": 3},
    "pre_gate": "selection_passes from physically truncated pre-2024 protocol",
    "test_2024_gate": {
        "positive": True,
        "min_ratio": 3.0,
        "max_mdd": 15.0,
        "min_trades": 12,
    },
    "promotion_rule": "minimum individual passes and both 3/5-model means pass at both sizes",
    "opened_window": ["2024-01-01", TEST_END],
    "sealed_windows": ["2025+"],
}


@dataclass(frozen=True)
class Config(EvaluatorConfig):
    output: str = DEFAULT_OUTPUT
    docs_output: str = DEFAULT_DOCS
    exclude_from: str = TEST_END


def _spec_hash() -> str:
    encoded = json.dumps(AUDIT_SPEC, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _implementation_hash() -> str:
    functions = (
        candidate_activation,
        test_2024_passes,
        evaluate_candidate,
        audit_decision,
        audit_payload,
        delayed_feature_context,
        build_full_design,
        fit_deterministic_stability_model,
        ensemble_predictions,
        evaluate_model,
        routed_schedule,
        equity_stats,
    )
    return hashlib.sha256("\n\n".join(inspect.getsource(fn) for fn in functions).encode()).hexdigest()


def candidate_activation(context: dict[str, Any], model: dict[str, Any]) -> tuple[np.ndarray, dict[str, float]]:
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
    return active, {
        "funding": funding_threshold,
        "premium": premium_threshold,
        "width": width_threshold,
        "pullback": pullback_threshold,
    }


def test_2024_passes(stats: dict[str, Any]) -> bool:
    gate = AUDIT_SPEC["test_2024_gate"]
    return bool(
        stats["absolute_return_pct"] > 0.0
        and stats["cagr_to_strict_mdd"] >= gate["min_ratio"]
        and stats["strict_mdd_pct"] <= gate["max_mdd"]
        and stats["trades"] >= gate["min_trades"]
    )


def evaluate_candidate(
    context: dict[str, Any],
    model: dict[str, Any],
    cfg: Config,
    *,
    label: str,
    trees: int,
    seeds: list[int],
) -> dict[str, Any]:
    pre = evaluate_model(
        context,
        model,
        cfg,
        label=label,
        trees=trees,
        seeds=seeds,
    )
    active, thresholds = candidate_activation(context, model)
    trades = routed_schedule(
        context,
        {"engine": model["engine"], "active": active},
        start="2024-01-01",
        end=TEST_END,
    )
    test_stats = slim(
        equity_stats(
            trades,
            start="2024-01-01",
            end=TEST_END,
            cfg=_execution_config(cfg, cfg.leverage),
        )
    )
    pre_pass = bool(selection_passes(pre["stats"]))
    test_pass = test_2024_passes(test_stats)
    return {
        "label": label,
        "trees": trees,
        "seeds": seeds,
        "thresholds": thresholds,
        "pre_pass": pre_pass,
        "test_2024_pass": test_pass,
        "joint_pass": bool(pre_pass and test_pass),
        "activation_hash": _activation_hash(active, context["dates"]),
        "schedule_hashes": {
            **pre["schedule_hashes"],
            "test_2024": _schedule_hash(trades),
        },
        "stats": {**pre["stats"], "test_2024": test_stats},
    }


def audit_decision(
    *,
    individual_passes: dict[int, int],
    mean_three_passes: dict[int, bool],
    mean_five_passes: dict[int, bool],
) -> bool:
    minimums = AUDIT_SPEC["minimum_individual_joint_passes"]
    return bool(
        all(individual_passes[size] >= int(minimums[str(size)]) for size in TREE_COUNTS)
        and all(mean_three_passes.values())
        and all(mean_five_passes.values())
    )


def _assert_expected(candidate: dict[str, Any]) -> None:
    for window, expected in EXPECTED_SELECTED_STATS.items():
        actual = candidate["stats"][window]
        for key, value in expected.items():
            if isinstance(value, int):
                if int(actual[key]) != value:
                    raise RuntimeError(f"{window}.{key} drifted: {actual[key]} != {value}")
            elif not np.isclose(float(actual[key]), float(value), rtol=0.0, atol=1e-10):
                raise RuntimeError(f"{window}.{key} drifted: {actual[key]} != {value}")


def _hashable(payload: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "phase",
        "test_opened",
        "eval_opened",
        "sealed_windows",
        "test_end_exclusive",
        "audit_spec",
        "spec_hash",
        "implementation_hash",
        "source_hashes_through_2024",
        "feature_hash_through_2024",
        "forest_runs",
        "selected_candidate",
        "audit_summary",
    )
    return {key: payload[key] for key in keys}


def _audit_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(_hashable(payload), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def audit_payload(cfg: Config) -> dict[str, Any]:
    if cfg.exclude_from != TEST_END:
        raise RuntimeError("test audit cutoff must remain physically fixed at 2025-01-01")
    context = delayed_feature_context(build_full_design(cfg), DELAY_BARS)
    if len(context["dates"]) and context["dates"].max() >= pd.Timestamp(TEST_END):
        raise RuntimeError("2025+ rows opened during test audit")
    runs: dict[str, Any] = {}
    individual_passes: dict[int, int] = {}
    mean_three_passes: dict[int, bool] = {}
    mean_five_passes: dict[int, bool] = {}
    for trees in TREE_COUNTS:
        fits = [
            fit_deterministic_stability_model(context, cfg, trees=trees, seed=seed)
            for seed in SEEDS
        ]
        individual = [
            evaluate_candidate(
                context,
                ensemble_predictions([fitted]),
                cfg,
                label=f"seed_{seed}_{trees}",
                trees=trees,
                seeds=[seed],
            )
            for seed, fitted in zip(SEEDS, fits, strict=True)
        ]
        mean_three = evaluate_candidate(
            context,
            ensemble_predictions(fits[:3]),
            cfg,
            label=f"mean_3x{trees}",
            trees=3 * trees,
            seeds=list(SEEDS[:3]),
        )
        mean_five = evaluate_candidate(
            context,
            ensemble_predictions(fits),
            cfg,
            label=f"mean_5x{trees}",
            trees=5 * trees,
            seeds=list(SEEDS),
        )
        runs[str(trees)] = {
            "individual": individual,
            "mean_three": mean_three,
            "mean_five": mean_five,
        }
        individual_passes[trees] = sum(int(row["joint_pass"]) for row in individual)
        mean_three_passes[trees] = bool(mean_three["joint_pass"])
        mean_five_passes[trees] = bool(mean_five["joint_pass"])
    selected = runs["2000"]["mean_five"]
    _assert_expected(selected)
    promotable = audit_decision(
        individual_passes=individual_passes,
        mean_three_passes=mean_three_passes,
        mean_five_passes=mean_five_passes,
    )
    payload: dict[str, Any] = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "phase": "test_2024_delayed_conditional_pullback_audit",
        "test_opened": True,
        "eval_opened": False,
        "sealed_windows": ["2025+"],
        "test_end_exclusive": TEST_END,
        "config": asdict(cfg),
        "audit_spec": AUDIT_SPEC,
        "spec_hash": _spec_hash(),
        "implementation_hash": _implementation_hash(),
        "source_hashes_through_2024": context["source_hashes"],
        "feature_hash_through_2024": _array_hash(context["matrix"]),
        "forest_runs": runs,
        "selected_candidate": selected,
        "audit_summary": {
            "individual_joint_passes": {str(key): value for key, value in individual_passes.items()},
            "individual_cells": len(SEEDS),
            "mean_three_passes": {str(key): value for key, value in mean_three_passes.items()},
            "mean_five_passes": {str(key): value for key, value in mean_five_passes.items()},
            "promotable": promotable,
            "decision": "promote_to_eval_freeze" if promotable else "reject",
        },
    }
    if not promotable:
        raise RuntimeError("delayed candidate failed the 2024 test stability contract")
    payload["audit_hash"] = _audit_hash(payload)
    return payload


def validate_audit(payload: dict[str, Any]) -> None:
    if payload.get("test_opened") is not True or payload.get("eval_opened") is not False:
        raise RuntimeError("audit test/eval state changed")
    if payload.get("sealed_windows") != ["2025+"]:
        raise RuntimeError("2025+ must remain sealed")
    if payload.get("audit_spec") != AUDIT_SPEC:
        raise RuntimeError("audit specification changed")
    if payload.get("spec_hash") != _spec_hash() or payload.get("implementation_hash") != _implementation_hash():
        raise RuntimeError("audit implementation changed")
    if payload.get("audit_hash") != _audit_hash(payload):
        raise RuntimeError("audit hash mismatch")
    if not payload["audit_summary"]["promotable"]:
        raise RuntimeError("candidate is not promotable")
    _assert_expected(payload["selected_candidate"])


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
    selected = payload["selected_candidate"]
    summary = payload["audit_summary"]
    lines = [
        "# Delayed conditional-pullback 2024 test audit — 2026-07-15",
        "",
        "**2024 test passed; 2025+ remains sealed.**",
        "",
        "Metric: absolute return / CAGR / strict MDD / CAGR-to-strict-MDD / trades.",
        "",
        "- All predictive features except current funding/premium event identity are delayed by 12×5m (one hour).",
        "- The delay was fixed as a pre-2024 timing control before 2024 was opened; no delay grid was searched on test.",
        f"- 1,000-tree individual joint passes: **{summary['individual_joint_passes']['1000']}/{summary['individual_cells']}**.",
        f"- 2,000-tree individual joint passes: **{summary['individual_joint_passes']['2000']}/{summary['individual_cells']}**.",
        f"- Three/five-model means pass at both sizes: **{all(summary['mean_three_passes'].values()) and all(summary['mean_five_passes'].values())}**.",
        "",
        "## Selected 5×2,000-tree model",
        "",
        "| Window | Result |",
        "|---|---:|",
    ]
    for name in ("train", "select_2023", "select_2023_h1", "select_2023_h2", "pre_2024", "test_2024"):
        lines.append(f"| {name} | {_metric(selected['stats'][name])} |")
    lines += [
        "",
        "The next step is a write-once eval manifest pinning this exact delay, forest ensemble, thresholds, feature graph, source-owned exits, and 2025/2026 windows before either is evaluated.",
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
    print(json.dumps({"phase": payload["phase"], "audit_hash": payload["audit_hash"], "audit_summary": payload["audit_summary"]}, indent=2))


if __name__ == "__main__":
    main()
