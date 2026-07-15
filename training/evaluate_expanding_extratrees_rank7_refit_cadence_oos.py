#!/usr/bin/env python3
"""Evaluate frozen annual-vs-monthly rank-7 refit cadences on 2025/2026H1.

The cadence family and pre-2025 choice are loaded from a hash-pinned manifest.
Both cadences are replayed through the future horizon for comparison, but future
metrics cannot rerank the already-selected annual cadence.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.compare_expanding_extratrees_rank7_refit_cadence_pre2025 import (
    DEFAULT_MANIFEST as CADENCE_MANIFEST,
    LearnerSpec,
    SelectionSpec,
    _json_hash,
    _sha256,
    _balanced_weights,
    activation_for,
    build_selection_base,
    fit_cadence_folds,
    metric_delta,
)
from training.evaluate_expanding_extratrees_top10_oos import (
    ALL_COMBINED,
    COMBINED_WINDOW,
    FULL_CUTOFF,
    FUTURE_COMBINED,
    REPLAY_WINDOWS,
    assert_prefix_parity,
    build_replay_base,
    full_passes,
    future_passes,
)
from training.evaluate_stable_ensemble_conditional_pullback_oos import Config
from training.search_inventory_purge_reclaim_alpha import _schedule_hash, equity_stats
from training.search_liveparity_state_feature_interactions import slim
from training.search_stable_ensemble_conditional_pullback_alpha import routed_schedule

EXPECTED_CADENCE_MANIFEST_HASH = "627441e5a7a3bd070e136e771f7dcc93cea6162565c0dd2226c2140c5c836f21"
REFERENCE_OOS = "results/expanding_extratrees_top10_oos_2026-07-15.json"
DEFAULT_OUTPUT = "results/expanding_extratrees_rank7_refit_cadence_oos_2026-07-16.json"
DEFAULT_DOCS = "docs/expanding-extratrees-rank7-refit-cadence-oos-2026-07-16.md"
ALL_WINDOWS = REPLAY_WINDOWS + (COMBINED_WINDOW, FUTURE_COMBINED, ALL_COMBINED)


def validate_cadence_manifest(path: str = CADENCE_MANIFEST) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    embedded_hash = manifest.pop("manifest_hash", None)
    calculated_hash = _json_hash(manifest)
    if embedded_hash != calculated_hash or calculated_hash != EXPECTED_CADENCE_MANIFEST_HASH:
        raise RuntimeError("frozen cadence manifest hash mismatch")
    manifest["manifest_hash"] = embedded_hash
    if manifest["selection_cutoff_exclusive"] != "2025-01-01":
        raise RuntimeError("cadence selection cutoff drifted")
    if manifest["cadences"] != ["annual", "monthly"]:
        raise RuntimeError("cadence family drifted")
    if manifest["selected_cadence"] != "annual":
        raise RuntimeError("pre-2025 cadence choice drifted")
    result_path = Path(manifest["selection_result"])
    if _sha256(result_path) != manifest["selection_result_sha256"]:
        raise RuntimeError("cadence selection result hash mismatch")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result["selected_cadence"] != manifest["selected_cadence"]:
        raise RuntimeError("cadence result/manifest choice mismatch")
    return manifest, result


def schedule_stats(
    base: dict[str, Any], active: np.ndarray
) -> tuple[dict[str, Any], dict[str, str], dict[str, dict[int, float]]]:
    stats: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    entries: dict[str, dict[int, float]] = {}
    cfg = base["execution_cfg"]
    fee_factor = 1.0 - float(cfg.leverage) * float(cfg.fee_rate + cfg.slippage_rate)
    for name, start, end in ALL_WINDOWS:
        trades = routed_schedule(
            base["context"], {"engine": base["engine"], "active": active}, start=start, end=end
        )
        stats[name] = slim(equity_stats(trades, start=start, end=end, cfg=base["execution_cfg"]))
        hashes[name] = _schedule_hash(trades)
        entries[name] = {
            int(trade.entry_position): float(
                (fee_factor * trade.price_factor * trade.funding_factor * fee_factor - 1.0)
                * 10_000.0
            )
            for trade in trades
        }
    return stats, hashes, entries


def fold_weight_diagnostics(base: dict[str, Any], fit: np.ndarray) -> dict[str, Any]:
    weights = _balanced_weights(base, fit)
    years = base["signal_dates"][fit].dt.year.to_numpy()
    sources = base["funding_source"][fit]
    groups = list(zip(years.tolist(), sources.tolist(), strict=True))
    counts = {group: groups.count(group) for group in set(groups)}
    effective = float(weights.sum() ** 2 / np.square(weights).sum())
    return {
        "source_year_groups": len(counts),
        "min_source_year_group_count": min(counts.values()),
        "max_source_year_group_count": max(counts.values()),
        "max_weight_multiple": float(weights.max() / weights.mean()),
        "effective_fit_examples": effective,
        "effective_fit_fraction": effective / len(weights),
    }


def evaluate_cadence(
    base: dict[str, Any],
    learner: LearnerSpec,
    policy: SelectionSpec,
    cadence: str,
) -> tuple[dict[str, Any], dict[str, dict[int, float]]]:
    folds = fit_cadence_folds(base, learner, cadence, start="2023-01-01", end=FULL_CUTOFF)
    active = activation_for(base, folds, policy)
    stats, hashes, entries = schedule_stats(base, active)
    return (
        {
            "cadence": cadence,
            "fold_count": len(folds),
            "folds": [
                {
                    **{
                        key: fold[key]
                        for key in ("name", "start", "end", "fit_examples", "predict_events", "latest_fit_exit")
                    },
                    **fold_weight_diagnostics(base, fold["fit"]),
                }
                for fold in folds
            ],
            "future_pass": future_passes(stats),
            "full_pass": full_passes(stats),
            "stats": stats,
            "schedule_hashes": hashes,
            "activation_hash": _json_hash(np.flatnonzero(active).tolist()),
            "active_events": int(active.sum()),
        },
        entries,
    )


def schedule_overlap(
    annual: dict[str, dict[int, float]], monthly: dict[str, dict[int, float]]
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in annual:
        left, right = set(annual[name]), set(monthly[name])
        union = left | right
        out[name] = {
            "annual_entries": len(left),
            "monthly_entries": len(right),
            "shared_entries": len(left & right),
            "union_entries": len(union),
            "jaccard": 1.0 if not union else len(left & right) / len(union),
        }
    return out


def _return_summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    return {
        "trades": int(len(array)),
        "mean_net_bps": float(array.mean()) if len(array) else 0.0,
        "win_rate": float((array > 0.0).mean()) if len(array) else 0.0,
    }


def divergent_trade_performance(
    annual: dict[str, dict[int, float]], monthly: dict[str, dict[int, float]]
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in annual:
        left, right = annual[name], monthly[name]
        shared = set(left) & set(right)
        if any(abs(left[key] - right[key]) > 1e-10 for key in shared):
            raise RuntimeError(f"shared trade return drifted for {name}")
        out[name] = {
            "shared": _return_summary([left[key] for key in sorted(shared)]),
            "annual_only": _return_summary([left[key] for key in sorted(set(left) - set(right))]),
            "monthly_only": _return_summary([right[key] for key in sorted(set(right) - set(left))]),
        }
    return out


def weight_summary(folds: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "minimum_effective_fit_fraction": min(fold["effective_fit_fraction"] for fold in folds),
        "maximum_weight_multiple": max(fold["max_weight_multiple"] for fold in folds),
        "minimum_source_year_group_count": min(fold["min_source_year_group_count"] for fold in folds),
    }


def assert_frozen_prefix(
    manifest: dict[str, Any], cadence: str, hashes: dict[str, str]
) -> None:
    expected = manifest["schedule_hashes"][cadence]
    actual = {name: hashes[name] for name in expected}
    if actual != expected:
        raise RuntimeError(f"{cadence} pre-2025 schedule prefix drifted")


def assert_annual_reference(result: dict[str, Any]) -> None:
    reference = json.loads(Path(REFERENCE_OOS).read_text(encoding="utf-8"))["candidates"][6]
    if reference["rank_position"] != 7:
        raise RuntimeError("annual OOS reference rank drifted")
    if result["schedule_hashes"] != reference["schedule_hashes"]:
        raise RuntimeError("annual cadence no longer reproduces frozen rank-7 OOS schedules")
    if result["stats"] != reference["stats"]:
        raise RuntimeError("annual cadence no longer reproduces frozen rank-7 OOS stats")


def render_docs(payload: dict[str, Any]) -> str:
    lines = [
        "# Frozen rank-7 annual vs monthly refit cadence OOS",
        "",
        f"Cadence manifest: `{payload['cadence_manifest_hash']}`",
        "",
        f"Pre-2025 selected cadence: **{payload['selected_cadence']}**. Future metrics below "
        "compare both frozen alternatives but do not rerank that choice.",
        "",
        "## Results",
        "",
        "| Cadence | Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Mean net | Win rate | Future/full pass |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for cadence in ("annual", "monthly"):
        row = payload["cadences"][cadence]
        for name in ("eval_2025", "holdout_2026h1", "future_2025_2026h1", "all_2023_2026h1"):
            s = row["stats"][name]
            lines.append(
                f"| {cadence} | {name} | {s['absolute_return_pct']:.4f}% | "
                f"{s['cagr_pct']:.4f}% | {s['strict_mdd_pct']:.4f}% | "
                f"{s['cagr_to_strict_mdd']:.4f} | {s['trades']} | "
                f"{s['mean_net_bps']:.2f} bps | {100*s['win_rate']:.2f}% | "
                f"{row['future_pass']}/{row['full_pass']} |"
            )
    future_delta = payload["monthly_minus_annual"]["future_2025_2026h1"]
    divergent = payload["divergent_trade_performance"]["future_2025_2026h1"]
    annual_weights = payload["weight_diagnostics"]["annual"]
    monthly_weights = payload["weight_diagnostics"]["monthly"]
    lines.extend(
        [
            "",
            "## Future delta: monthly minus annual",
            "",
            f"- Absolute return: `{future_delta['absolute_return_pct']:+.4f}` percentage points",
            f"- CAGR: `{future_delta['cagr_pct']:+.4f}` percentage points",
            f"- Strict MDD: `{future_delta['strict_mdd_pct']:+.4f}` percentage points",
            f"- CAGR/MDD: `{future_delta['cagr_to_strict_mdd']:+.4f}`",
            f"- Trades: `{future_delta['trades']:+.0f}`",
            f"- Mean net/trade: `{future_delta['mean_net_bps']:+.2f}` bps",
            "",
            "## Mechanism diagnostics",
            "",
            f"- Shared future trades: `{divergent['shared']['trades']}` at "
            f"`{divergent['shared']['mean_net_bps']:.2f}` mean net bps.",
            f"- Annual-only future trades: `{divergent['annual_only']['trades']}` at "
            f"`{divergent['annual_only']['mean_net_bps']:.2f}` mean net bps.",
            f"- Monthly-only future trades: `{divergent['monthly_only']['trades']}` at "
            f"`{divergent['monthly_only']['mean_net_bps']:.2f}` mean net bps.",
            f"- Annual folds: minimum effective-fit fraction "
            f"`{annual_weights['minimum_effective_fit_fraction']:.3f}`, maximum sample-weight "
            f"multiple `{annual_weights['maximum_weight_multiple']:.2f}`.",
            f"- Monthly folds: minimum effective-fit fraction "
            f"`{monthly_weights['minimum_effective_fit_fraction']:.3f}`, maximum sample-weight "
            f"multiple `{monthly_weights['maximum_weight_multiple']:.2f}`.",
            "",
            "## Integrity",
            "",
            "- The cadence manifest and selection-result hashes are pinned in code.",
            "- Annual and monthly pre-2025 schedules reproduce their committed prefixes.",
            "- The annual cadence reproduces the prior frozen rank-7 OOS schedules and metrics exactly.",
            "- Every future monthly/annual fold purges labels whose exits reach its cutoff.",
            "",
        ]
    )
    return "\n".join(lines)


def run(manifest_path: str, output: str, docs_output: str) -> dict[str, Any]:
    manifest, selection_result = validate_cadence_manifest(manifest_path)
    learner = LearnerSpec(**selection_result["learner"])
    policy = SelectionSpec(**selection_result["selection"])

    selection_cfg = Config(exclude_from="2025-01-01", output="/tmp/no_write_selection.json", docs_output="")
    selection_base = build_selection_base(selection_cfg)
    replay_cfg = Config(exclude_from=FULL_CUTOFF, output="/tmp/no_write_replay.json", docs_output="")
    replay_base = build_replay_base(replay_cfg)
    assert_prefix_parity(selection_base, replay_base)

    annual, annual_entries = evaluate_cadence(replay_base, learner, policy, "annual")
    monthly, monthly_entries = evaluate_cadence(replay_base, learner, policy, "monthly")
    assert_frozen_prefix(manifest, "annual", annual["schedule_hashes"])
    assert_frozen_prefix(manifest, "monthly", monthly["schedule_hashes"])
    assert_annual_reference(annual)

    payload = {
        "schema_version": 1,
        "mode": "frozen_rank7_annual_vs_monthly_refit_cadence_oos",
        "cadence_manifest": manifest_path,
        "cadence_manifest_hash": manifest["manifest_hash"],
        "selected_cadence": manifest["selected_cadence"],
        "future_did_not_rerank": True,
        "future_windows": manifest["future_windows"],
        "learner": selection_result["learner"],
        "selection": selection_result["selection"],
        "seeds": selection_result["seeds"],
        "trees_per_seed": selection_result["trees_per_seed"],
        "feature_delay_bars": selection_result["feature_delay_bars"],
        "label_purge": selection_result["label_purge"],
        "cadences": {"annual": annual, "monthly": monthly},
        "monthly_minus_annual": metric_delta(monthly["stats"], annual["stats"]),
        "schedule_overlap": schedule_overlap(annual_entries, monthly_entries),
        "divergent_trade_performance": divergent_trade_performance(annual_entries, monthly_entries),
        "weight_diagnostics": {
            "annual": weight_summary(annual["folds"]),
            "monthly": weight_summary(monthly["folds"]),
        },
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if docs_output:
        docs_path = Path(docs_output)
        docs_path.parent.mkdir(parents=True, exist_ok=True)
        docs_path.write_text(render_docs(payload), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=CADENCE_MANIFEST)
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
                "selected_cadence": payload["selected_cadence"],
                "annual": payload["cadences"]["annual"]["stats"],
                "monthly": payload["cadences"]["monthly"]["stats"],
                "monthly_minus_annual": payload["monthly_minus_annual"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
