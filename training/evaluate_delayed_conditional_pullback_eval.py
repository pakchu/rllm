"""Evaluate the frozen one-hour-delayed alpha on 2025 and 2026H1."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.audit_confirmed_pullback_squeeze_live_parity import _execution_config
from training.audit_delayed_conditional_pullback_test2024 import (
    Config as AuditReplayConfig,
    DEFAULT_OUTPUT as AUDIT_RESULT,
    evaluate_candidate,
)
from training.audit_stable_ensemble_conditional_pullback_alpha import (
    DELAY_BARS,
    delayed_feature_context,
)
from training.evaluate_stable_ensemble_conditional_pullback_oos import (
    Config as BaseConfig,
    build_full_design,
    fit_frozen_ensemble,
    predict_full_ensemble,
)
from training.freeze_delayed_conditional_pullback_eval import (
    AUDIT_FILE_SHA256,
    EVAL_GATE,
    EVAL_WINDOWS,
    MANIFEST_FIELDS,
    manifest_hash,
    validate_manifest,
)
from training.search_inventory_purge_reclaim_alpha import _schedule_hash, equity_stats
from training.search_liveparity_state_feature_interactions import slim
from training.search_stable_ensemble_conditional_pullback_alpha import (
    FEATURE_COLUMNS,
    PULLBACK_FEATURE,
    WIDTH_FEATURE,
    _activation_hash,
    _array_hash,
    conditional_activation,
    immutable_anchors,
    routed_schedule,
)

EXPECTED_MANIFEST_HASH = "2cc5d9f5837188372b3b9239c04f2873e7a43389aa6d0a385f0b100f028839bc"
DEFAULT_MANIFEST = "results/delayed_conditional_pullback_eval_manifest_2026-07-15.json"
DEFAULT_OUTPUT = "results/delayed_conditional_pullback_eval_2026-07-15.json"
DEFAULT_DOCS = "docs/delayed-conditional-pullback-eval-2026-07-15.md"
EVAL_END = "2026-06-02"
PREFIX_END = "2025-01-01"
REPLAY_WINDOWS = {
    "train": ("2020-07-01", "2023-01-01"),
    "train_2020h2": ("2020-07-01", "2021-01-01"),
    "train_2021": ("2021-01-01", "2022-01-01"),
    "train_2022": ("2022-01-01", "2023-01-01"),
    "select_2023": ("2023-01-01", "2024-01-01"),
    "select_2023_h1": ("2023-01-01", "2023-07-01"),
    "select_2023_h2": ("2023-07-01", "2024-01-01"),
    "pre_2024": ("2020-07-01", "2024-01-01"),
    "test_2024": ("2024-01-01", "2025-01-01"),
}


@dataclass(frozen=True)
class Config(BaseConfig):
    manifest: str = DEFAULT_MANIFEST
    output: str = DEFAULT_OUTPUT
    docs_output: str = DEFAULT_DOCS
    exclude_from: str = EVAL_END


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_frozen_manifest(cfg: Config) -> dict[str, Any]:
    manifest = json.loads(Path(cfg.manifest).read_text(encoding="utf-8"))
    validate_manifest(manifest)
    if manifest["manifest_hash"] != EXPECTED_MANIFEST_HASH:
        raise RuntimeError("eval manifest differs from the evaluator pin")
    if manifest_hash(manifest) != EXPECTED_MANIFEST_HASH:
        raise RuntimeError("eval manifest payload differs from the evaluator pin")
    if tuple(key for key in manifest if key in MANIFEST_FIELDS) != MANIFEST_FIELDS:
        raise RuntimeError("eval manifest payload shape changed")
    if _sha256(AUDIT_RESULT) != AUDIT_FILE_SHA256:
        raise RuntimeError("test audit changed after eval freeze")
    if cfg.exclude_from != max(window[1] for window in manifest["eval_windows"].values()):
        raise RuntimeError("exclude_from must equal the frozen eval horizon")
    if manifest["eval_windows"] != {key: list(value) for key, value in EVAL_WINDOWS.items()}:
        raise RuntimeError("eval windows changed")
    if manifest["eval_gate"] != EVAL_GATE:
        raise RuntimeError("eval gate changed")
    return manifest


def _replay_through_2024(
    cfg: Config,
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    prefix_cfg = replace(cfg, exclude_from=PREFIX_END)
    context = delayed_feature_context(build_full_design(prefix_cfg), DELAY_BARS)
    if len(context["dates"]) and context["dates"].max() >= pd.Timestamp(PREFIX_END):
        raise RuntimeError("2025+ opened during frozen-prefix replay")
    if context["source_hashes"] != manifest["source_hashes_through_2024"]:
        raise RuntimeError("source hashes through 2024 changed")
    if _array_hash(context["matrix"]) != manifest["feature_hash_through_2024"]:
        raise RuntimeError("delayed feature graph through 2024 changed")
    fitted = fit_frozen_ensemble(context, cfg)
    replay_cfg = AuditReplayConfig(
        input_csv=cfg.input_csv,
        funding_csv=cfg.funding_csv,
        premium_csv=cfg.premium_csv,
        spot_premium_csv=cfg.spot_premium_csv,
        output="/tmp/delayed_conditional_pullback_prefix_replay.json",
        docs_output="",
    )
    replay = evaluate_candidate(
        context,
        fitted,
        replay_cfg,
        label="frozen_through_2024_replay",
        trees=10_000,
        seeds=list(manifest["model_spec"]["seeds"]),
    )
    if replay["activation_hash"] != manifest["selected_activation_hash"]:
        raise RuntimeError("activation through 2024 did not replay")
    if replay["schedule_hashes"] != manifest["selected_schedule_hashes"]:
        raise RuntimeError("schedules through 2024 did not replay")
    if replay["stats"] != manifest["selected_stats"]:
        raise RuntimeError("statistics through 2024 did not replay")
    for key, expected in manifest["candidate_spec"]["thresholds"].items():
        if not np.isclose(float(replay["thresholds"][key]), float(expected), rtol=0.0, atol=1e-15):
            raise RuntimeError(f"frozen threshold changed: {key}")
    integrity = {
        "feature_hash_through_2024": _array_hash(context["matrix"]),
        "base_hash_through_2024": _activation_hash(context["base"], context["dates"]),
        "anchor_hash_through_2024": _activation_hash(
            immutable_anchors(context["base"], 144), context["dates"]
        ),
    }
    return context, fitted, replay | {"integrity": integrity}


def frozen_activation(
    context: dict[str, Any],
    model: dict[str, Any],
    manifest: dict[str, Any],
) -> np.ndarray:
    positions = np.asarray(model["anchor_positions"], dtype=np.int64)
    funding = np.asarray(context["funding_leg"], dtype=bool)[positions]
    matrix = np.asarray(context["matrix"], dtype=float)
    thresholds = manifest["candidate_spec"]["thresholds"]
    return conditional_activation(
        size=len(context["market"]),
        anchor_positions=positions,
        anchor_predictions=model["anchor_predictions"],
        anchor_is_funding=funding,
        anchor_width=matrix[positions, FEATURE_COLUMNS.index(WIDTH_FEATURE)],
        anchor_pullback=matrix[positions, FEATURE_COLUMNS.index(PULLBACK_FEATURE)],
        funding_threshold=float(thresholds["funding"]),
        premium_threshold=float(thresholds["premium"]),
        width_threshold=float(thresholds["width"]),
        pullback_threshold=float(thresholds["pullback"]),
    )


def passes_eval_gate(stats: dict[str, dict[str, Any]]) -> bool:
    for name, gate in EVAL_GATE.items():
        row = stats[name]
        if gate["positive"] and row["absolute_return_pct"] <= 0.0:
            return False
        if row["cagr_to_strict_mdd"] < gate["min_ratio"]:
            return False
        if row["strict_mdd_pct"] > gate["max_mdd"]:
            return False
        if row["trades"] < gate["min_trades"]:
            return False
    return True


def _metric(row: dict[str, Any]) -> str:
    return (
        f"{row['absolute_return_pct']:.2f}% / {row['cagr_pct']:.2f}% / "
        f"{row['strict_mdd_pct']:.2f}% / {row['cagr_to_strict_mdd']:.2f} / {row['trades']}"
    )


def _write_docs(path: str | Path, report: dict[str, Any]) -> None:
    lines = [
        "# Delayed conditional-pullback eval — 2026-07-15",
        "",
        f"**{report['verdict']}**",
        "",
        "Metric: absolute return / CAGR / strict MDD / CAGR-to-strict-MDD / trades.",
        "",
        "| Window | 6 bp/side result | 10 bp/side stress | Gate pass |",
        "|---|---:|---:|---:|",
    ]
    for name in EVAL_WINDOWS:
        lines.append(
            f"| {name} | {_metric(report['stats_6bp'][name])} | "
            f"{_metric(report['stats_10bp'][name])} | {report['window_gate_passes'][name]} |"
        )
    lines += [
        "",
        "## Integrity",
        "",
        f"- Pinned eval manifest `{report['manifest_hash']}` was validated before 2025+ was opened.",
        "- The delayed feature matrix, activation, thresholds, and every train/selection/test schedule through 2024 replayed exactly.",
        "- No eval quantile, model choice, exit, or threshold is computed from 2025+.",
        "- Execution is next-open, 6 bp/notional/side, realized funding, stop-before-take, split-contained, wall-clock CAGR, and strict path MDD.",
        "- Candidate-level eval is implementation clean; global epistemic purity remains false because related feature families were researched previously.",
        "",
    ]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")


def run(cfg: Config) -> dict[str, Any]:
    manifest = validate_frozen_manifest(cfg)
    prefix_context, fitted, replay = _replay_through_2024(cfg, manifest)
    pre_integrity = replay.pop("integrity")
    fitted.pop("engine")
    del prefix_context
    gc.collect()

    # This is intentionally the first call permitted to open 2025+.
    full_context = delayed_feature_context(build_full_design(cfg), DELAY_BARS)
    prefix = (full_context["dates"] < pd.Timestamp(PREFIX_END)).to_numpy(bool)
    if _array_hash(full_context["matrix"][prefix]) != manifest["feature_hash_through_2024"]:
        raise RuntimeError("full delayed graph changed its frozen 2024 prefix")
    if _activation_hash(full_context["base"][prefix], full_context["dates"][prefix]) != pre_integrity["base_hash_through_2024"]:
        raise RuntimeError("full graph changed base decisions through 2024")
    full_anchors = immutable_anchors(full_context["base"], 144)
    if _activation_hash(full_anchors[prefix], full_context["dates"][prefix]) != pre_integrity["anchor_hash_through_2024"]:
        raise RuntimeError("full graph changed immutable anchors through 2024")
    full_model = predict_full_ensemble(full_context, fitted, cfg)
    active = frozen_activation(full_context, full_model, manifest)
    if _activation_hash(active[prefix], full_context["dates"][prefix]) != manifest["selected_activation_hash"]:
        raise RuntimeError("full graph changed frozen activation through 2024")
    for name, frozen_hash in manifest["selected_schedule_hashes"].items():
        start, end = REPLAY_WINDOWS[name]
        trades = routed_schedule(
            full_context,
            {"engine": full_model["engine"], "active": active},
            start=start,
            end=end,
        )
        if _schedule_hash(trades) != frozen_hash:
            raise RuntimeError(f"full graph changed frozen {name} schedule")

    stats_6bp: dict[str, dict[str, Any]] = {}
    stats_10bp: dict[str, dict[str, Any]] = {}
    schedule_hashes: dict[str, str] = {}
    source_counts: dict[str, dict[str, int]] = {}
    window_gate_passes: dict[str, bool] = {}
    for name, (start, end) in EVAL_WINDOWS.items():
        trades = routed_schedule(
            full_context,
            {"engine": full_model["engine"], "active": active},
            start=start,
            end=end,
        )
        stats = slim(
            equity_stats(
                trades,
                start=start,
                end=end,
                cfg=_execution_config(cfg, cfg.leverage),
            )
        )
        stress = slim(
            equity_stats(
                trades,
                start=start,
                end=end,
                cfg=_execution_config(cfg, cfg.leverage),
                cost_rate=0.0010,
            )
        )
        stats_6bp[name] = stats
        stats_10bp[name] = stress
        schedule_hashes[name] = _schedule_hash(trades)
        funding_count = sum(bool(full_context["funding_leg"][trade.signal_position]) for trade in trades)
        source_counts[name] = {"funding": funding_count, "premium": len(trades) - funding_count}
        gate = EVAL_GATE[name]
        window_gate_passes[name] = bool(
            stats["absolute_return_pct"] > 0.0
            and stats["cagr_to_strict_mdd"] >= gate["min_ratio"]
            and stats["strict_mdd_pct"] <= gate["max_mdd"]
            and stats["trades"] >= gate["min_trades"]
        )
    passed = passes_eval_gate(stats_6bp)
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "verdict": "ALPHA_QUALIFIED_FROZEN_EVAL" if passed else "REJECTED_FROZEN_EVAL",
        "eval_opened": True,
        "candidate_level_eval_clean": True,
        "globally_epistemically_pristine": False,
        "manifest": cfg.manifest,
        "manifest_hash": manifest["manifest_hash"],
        "config": asdict(cfg),
        "integrity": {
            "prefix_replay_activation_hash": replay["activation_hash"],
            "prefix_replay_schedule_hashes": replay["schedule_hashes"],
            "full_feature_hash": _array_hash(full_context["matrix"]),
            "full_feature_prefix_hash": _array_hash(full_context["matrix"][prefix]),
            "full_active_prefix_hash": _activation_hash(active[prefix], full_context["dates"][prefix]),
        },
        "stats_6bp": stats_6bp,
        "stats_10bp": stats_10bp,
        "window_gate_passes": window_gate_passes,
        "schedule_hashes": schedule_hashes,
        "source_counts": source_counts,
        "passes_eval_gate": passed,
    }
    target = Path(cfg.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=False), encoding="utf-8")
    _write_docs(cfg.docs_output, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", default=Config.input_csv)
    parser.add_argument("--funding-csv", default=Config.funding_csv)
    parser.add_argument("--premium-csv", default=Config.premium_csv)
    parser.add_argument("--spot-premium-csv", default=Config.spot_premium_csv)
    parser.add_argument("--manifest", default=Config.manifest)
    parser.add_argument("--output", default=Config.output)
    parser.add_argument("--docs-output", default=Config.docs_output)
    return parser.parse_args()


def main() -> None:
    report = run(Config(**vars(parse_args())))
    print(json.dumps({"verdict": report["verdict"], "manifest_hash": report["manifest_hash"], "passes_eval_gate": report["passes_eval_gate"], "stats_6bp": report["stats_6bp"]}, indent=2))


if __name__ == "__main__":
    main()
