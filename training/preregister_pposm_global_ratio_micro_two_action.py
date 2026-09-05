"""Freeze the micro two-action global-ratio PPOSM source diagnostic.

This preregistration is deliberately label/metric closed: it binds code,
source hashes, lifecycle manifest hashes, feature map, models, folds, and gates
before the evaluator opens lifecycle labels or computes source-support metrics.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Iterable

from training import evaluate_pposm_global_ratio_micro_two_action as diagnostic

PROTOCOL = "pposm_global_count_long_short_ratio_micro_two_action_preregistration_v1"
DEFAULT_OUTPUT = diagnostic.DEFAULT_PREREGISTRATION
EVALUATOR_PATH = Path("training/evaluate_pposm_global_ratio_micro_two_action.py")


def build() -> dict[str, Any]:
    manifest, _ = diagnostic.lifecycle.frozen.load_frozen_manifest(
        diagnostic.Config.manifest
    )
    source_sha = diagnostic.sha256_file(diagnostic.DEFAULT_METRICS)
    if source_sha != diagnostic.EXPECTED_METRICS_SHA256:
        raise RuntimeError("global ratio source hash drift before preregistration")
    core = {
        "protocol": PROTOCOL,
        "as_of": "2026-09-05",
        "objective": (
            "test whether hash-bound Binance global account long-short ratio "
            "supports pre-2024 lifecycle residual KEEP-vs-SWITCH separation "
            "for both SKIP and TP12 before any new SFT/RLVR or OOS"
        ),
        "why_successor": {
            "predecessor": "pposm_global_count_long_short_ratio_skip_source_support_v1",
            "problem": (
                "the previous HGB design could degenerate on the first expanding "
                "fold because only about 14 train rows are available while the "
                "default/min-large leaf regime cannot split usefully"
            ),
            "bounded_fix": (
                "pre-register a micro HGB with max_leaf_nodes=3, max_depth=2, "
                "min_samples_leaf=2, l2=1.0 plus a conservative C=0.25 "
                "balanced polynomial logistic regression"
            ),
        },
        "source": {
            "path": str(diagnostic.DEFAULT_METRICS),
            "sha256": diagnostic.EXPECTED_METRICS_SHA256,
            "pre2024_prefix_sha256": diagnostic.EXPECTED_PRE2024_PREFIX_SHA256,
            "pposm_manifest": str(diagnostic.Config.manifest),
            "pposm_manifest_sha256": diagnostic.sha256_file(
                diagnostic.Config.manifest
            ),
            "pposm_manifest_freeze_hash": manifest["freeze_hash"],
        },
        "design": {
            "actions": list(diagnostic.ACTIONS),
            "default_action": diagnostic.lifecycle.DEFAULT_ACTION,
            "anchors_per_action": diagnostic.EXPECTED_ANCHORS,
            "features": list(diagnostic.FEATURE_COLUMNS),
            "feature_map_fixed_before_outcomes": True,
            "join": (
                "exact source create_time == signal_time - 5 minutes; no ffill, "
                "interpolation, nearest, or stale-source carry"
            ),
            "folds": [
                {
                    "name": name,
                    "train_years": list(train_years),
                    "test_year": test_year,
                }
                for name, train_years, test_year in diagnostic.FOLD_SPECS
            ],
            "models": copy.deepcopy(diagnostic.MODEL_SPECS),
            "gates": dict(diagnostic.GATE_THRESHOLDS),
            "action_gate": "each of SKIP and TP12 must pass exactly three expanding folds independently",
            "overall_gate": "overall pass iff both action gates pass",
            "bootstrap_iterations": 2000,
            "random_seed": 20260905,
            "selection": (
                "passing model first, then pooled AUC, lower95, balanced accuracy"
            ),
        },
        "scope": {
            "train_only": True,
            "binary_source_feasibility_only": True,
            "both_skip_and_tp12_source_support_required": True,
            "full_professor_critic_pass_possible_from_this_artifact_alone": False,
        },
        "implementation": {
            "evaluator": str(EVALUATOR_PATH),
            "evaluator_sha256": diagnostic.sha256_file(EVALUATOR_PATH),
            "runtime_versions": diagnostic.runtime_versions(),
        },
        "evidence_boundary": {
            "lifecycle_labels_opened": 0,
            "oos_labels_opened": 0,
            "source_support_metrics_computed": False,
            "sft_or_rlvr_started": False,
        },
    }
    return {**core, "manifest_hash": diagnostic.sha256_canonical(core)}


def run(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    value = build()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return value


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    value = run(args.output)
    print(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
