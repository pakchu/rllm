"""Freeze the coverage-safe global-ratio SKIP source diagnostic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from training import evaluate_pposm_global_ratio_skip_source as diagnostic

PROTOCOL = "pposm_global_count_long_short_ratio_skip_source_preregistration_v1"
DEFAULT_OUTPUT = diagnostic.DEFAULT_PREREGISTRATION
EVALUATOR_PATH = Path("training/evaluate_pposm_global_ratio_skip_source.py")


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
            "test whether the coverage-safe global account long-short ratio supports "
            "pre-2024 lifecycle SKIP-vs-TP4 separation before any new SFT/RLVR or OOS"
        ),
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
            "action": "SKIP-vs-TP4 only",
            "anchors": diagnostic.EXPECTED_ANCHORS,
            "features": list(diagnostic.FEATURE_COLUMNS),
            "join": (
                "exact source create_time == signal_time - 5 minutes; "
                "no ffill/interpolation"
            ),
            "folds": [
                {
                    "name": name,
                    "train_years": list(train_years),
                    "test_year": test_year,
                }
                for name, train_years, test_year in diagnostic.FOLD_SPECS
            ],
            "models": ["logreg_l2_balanced", "hgb_balanced_weighted"],
            "gates": dict(diagnostic.GATE_THRESHOLDS),
            "bootstrap_iterations": 2000,
            "random_seed": 20260905,
            "selection": (
                "passing model first, then pooled AUC, lower95, balanced accuracy"
            ),
        },
        "scope": {
            "train_only": True,
            "binary_source_feasibility_only": True,
            "tp12_source_support_proven": False,
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
