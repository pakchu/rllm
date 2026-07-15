"""Freeze the audited conditional-pullback alpha before opening 2024+."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.audit_stable_ensemble_conditional_pullback_alpha import (
    AUDIT_SPEC,
    validate_audit,
)
from training.search_stable_ensemble_conditional_pullback_alpha import (
    EXPECTED_SELECTED_QUANTILES,
    FEATURE_COLUMNS,
    SEARCH_SPEC,
    validate_result,
)

SELECTION_RESULT = "results/stable_ensemble_conditional_pullback_pre2024_2026-07-15.json"
AUDIT_RESULT = "results/stable_ensemble_conditional_pullback_audit_pre2024_2026-07-15.json"
DEFAULT_OUTPUT = "results/stable_ensemble_conditional_pullback_manifest_2026-07-15.json"
DEFAULT_DOCS = "docs/stable-ensemble-conditional-pullback-freeze-2026-07-15.md"
SELECTION_COMMIT = "f9f60dbe19587b26658a9147caab34565bf39d0f"
AUDIT_COMMIT = "7d791489e60d5446baeaf5ee9a48b11f55c14047"
SELECTION_FILE_SHA256 = "ef89905d04dbceb1b4a6a8d4058831a704f8375d8f32e1060c821f1c205e7f4a"
AUDIT_FILE_SHA256 = "5d42e8ec48ff4512732f86b334de503869420da939db8bdc88c7b9436343a412"
FUTURE_WINDOWS = {
    "test_2024": ("2024-01-01", "2025-01-01"),
    "eval_2025": ("2025-01-01", "2026-01-01"),
    "holdout_2026h1": ("2026-01-01", "2026-06-02"),
    "oos_2024_2025": ("2024-01-01", "2026-01-01"),
    "oos_all": ("2024-01-01", "2026-06-02"),
}
OOS_GATE = {
    "test_2024": {"positive": True, "min_ratio": 3.0, "max_mdd": 15.0, "min_trades": 12},
    "eval_2025": {"positive": True, "min_ratio": 3.0, "max_mdd": 15.0, "min_trades": 12},
    "holdout_2026h1": {"positive": True, "min_ratio": 3.0, "max_mdd": 15.0, "min_trades": 6},
    "oos_2024_2025": {"positive": True, "min_ratio": 3.0, "max_mdd": 15.0, "min_trades": 30},
    "oos_all": {"positive": True, "min_ratio": 3.0, "max_mdd": 15.0, "min_trades": 36},
}
MANIFEST_FIELDS = (
    "schema_version",
    "phase",
    "oos_opened",
    "selection_end_exclusive",
    "source_commits",
    "source_artifacts",
    "source_prefix_hashes",
    "feature_prefix_hash",
    "model_spec",
    "candidate_spec",
    "execution_spec",
    "selected_activation_hash",
    "selected_schedule_hashes",
    "selected_stats",
    "stability_summary",
    "future_windows",
    "oos_gate",
)


@dataclass(frozen=True)
class Config:
    selection_result: str = SELECTION_RESULT
    audit_result: str = AUDIT_RESULT
    output: str = DEFAULT_OUTPUT
    docs_output: str = DEFAULT_DOCS


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest_hash(payload: dict[str, Any]) -> str:
    core = {key: payload[key] for key in MANIFEST_FIELDS}
    encoded = json.dumps(core, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("phase") != "pre_oos_frozen" or manifest.get("oos_opened") is not False:
        raise RuntimeError("manifest is not a sealed pre-OOS freeze")
    if tuple(key for key in manifest if key in MANIFEST_FIELDS) != MANIFEST_FIELDS:
        raise RuntimeError("manifest field order or membership changed")
    if manifest.get("manifest_hash") != manifest_hash(manifest):
        raise RuntimeError("manifest hash mismatch")
    if manifest.get("candidate_spec", {}).get("quantiles") != EXPECTED_SELECTED_QUANTILES:
        raise RuntimeError("frozen candidate quantiles changed")
    if manifest.get("future_windows") != {key: list(value) for key, value in FUTURE_WINDOWS.items()}:
        raise RuntimeError("future windows changed")
    if manifest.get("oos_gate") != OOS_GATE:
        raise RuntimeError("OOS gate changed")


def _write_once(path: str | Path, manifest: dict[str, Any]) -> dict[str, Any]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        validate_manifest(existing)
        if {key: existing[key] for key in MANIFEST_FIELDS} != {
            key: manifest[key] for key in MANIFEST_FIELDS
        }:
            raise RuntimeError("refusing to replace a different frozen manifest")
        return existing
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=False), encoding="utf-8")
    temporary.replace(target)
    return manifest


def build_manifest(cfg: Config) -> dict[str, Any]:
    if sha256_file(cfg.selection_result) != SELECTION_FILE_SHA256:
        raise RuntimeError("selection artifact differs from its committed SHA-256")
    if sha256_file(cfg.audit_result) != AUDIT_FILE_SHA256:
        raise RuntimeError("audit artifact differs from its committed SHA-256")
    selection = json.loads(Path(cfg.selection_result).read_text(encoding="utf-8"))
    audit = json.loads(Path(cfg.audit_result).read_text(encoding="utf-8"))
    validate_result(selection)
    validate_audit(audit)
    if selection["source_prefix_hashes"] != audit["source_prefix_hashes"]:
        raise RuntimeError("selection and audit source prefixes differ")
    if selection["feature_prefix_hash"] != audit["feature_prefix_hash"]:
        raise RuntimeError("selection and audit feature prefixes differ")
    candidate = selection["selected_candidate"]
    selected_audit = audit["forest_size_runs"]["2000"]["mean_five"]
    if candidate["activation_hash"] != selected_audit["activation_hash"]:
        raise RuntimeError("selection and audit activations differ")
    core: dict[str, Any] = {
        "schema_version": 1,
        "phase": "pre_oos_frozen",
        "oos_opened": False,
        "selection_end_exclusive": "2024-01-01",
        "source_commits": {"selection": SELECTION_COMMIT, "audit": AUDIT_COMMIT},
        "source_artifacts": {
            "selection": {
                "path": cfg.selection_result,
                "file_sha256": SELECTION_FILE_SHA256,
                "result_hash": selection["result_hash"],
                "search_spec_hash": selection["spec_hash"],
            },
            "audit": {
                "path": cfg.audit_result,
                "file_sha256": AUDIT_FILE_SHA256,
                "audit_hash": audit["audit_hash"],
                "audit_spec_hash": audit["spec_hash"],
            },
        },
        "source_prefix_hashes": selection["source_prefix_hashes"],
        "feature_prefix_hash": selection["feature_prefix_hash"],
        "model_spec": {
            "family": "RandomForestRegressor responsibility ensemble",
            "seeds": SEARCH_SPEC["seeds"],
            "trees_per_seed": SEARCH_SPEC["trees_per_seed"],
            "max_depth": 3,
            "min_samples_leaf": 16,
            "max_features": 0.7,
            "target": "48h net trade return",
            "fit_window": ["2020-07-01", "2023-01-01"],
            "anchor_cooldown_bars": 144,
            "prediction_reduction": SEARCH_SPEC["prediction_reduction"],
            "feature_columns": FEATURE_COLUMNS,
        },
        "candidate_spec": {
            "quantiles": {key: candidate["spec"][key] for key in EXPECTED_SELECTED_QUANTILES},
            "thresholds": {
                key: candidate["spec"][key]
                for key in (
                    "funding_threshold",
                    "premium_threshold",
                    "width_threshold",
                    "pullback_threshold",
                )
            },
            "interaction": SEARCH_SPEC["interaction"],
            "width_feature": "rex_2016_range_width_pct",
            "pullback_feature": "htf_1d_range_pos",
            "audit_spec": AUDIT_SPEC,
        },
        "execution_spec": {
            "decision_clock": ":00 after completed hour; current market 5m bar excluded",
            "entry": "next 5m open",
            "side": "long",
            "leverage": 0.5,
            "cost_per_notional_side": 0.0006,
            "funding_leg_exit": {"hold_bars": 576, "take_bps": 400, "stop_bps": 1_000_000},
            "premium_leg_exit": {"hold_bars": 144, "take_bps": 1_000_000, "stop_bps": 300},
            "realized_funding": True,
            "intrabar_priority": "stop before take",
            "non_overlap": True,
            "split_contained_exits": True,
            "mdd": "strict global/pre-entry HWM with favorable-before-adverse path",
            "cagr": "wall-clock including inactive periods",
        },
        "selected_activation_hash": candidate["activation_hash"],
        "selected_schedule_hashes": candidate["schedule_hashes"],
        "selected_stats": candidate["stats"],
        "stability_summary": audit["stability_summary"],
        "future_windows": {key: list(value) for key, value in FUTURE_WINDOWS.items()},
        "oos_gate": OOS_GATE,
    }
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), **core}
    manifest["manifest_hash"] = manifest_hash(manifest)
    validate_manifest(manifest)
    return manifest


def _metric(row: dict[str, Any]) -> str:
    return (
        f"{row['absolute_return_pct']:.2f}% / {row['cagr_pct']:.2f}% / "
        f"{row['strict_mdd_pct']:.2f}% / {row['cagr_to_strict_mdd']:.2f} / {row['trades']}"
    )


def _write_docs(path: str | Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Stable ensemble conditional-pullback freeze — 2026-07-15",
        "",
        "**Frozen before opening any 2024+ rows.**",
        "",
        "Metric: absolute return / CAGR / strict MDD / CAGR-to-strict-MDD / trades.",
        "",
        f"- Selection commit: `{SELECTION_COMMIT}`.",
        f"- Audit commit: `{AUDIT_COMMIT}`.",
        f"- Manifest hash: `{manifest['manifest_hash']}`.",
        "- Five deterministic 2,000-tree forests; source-specific q=.30/.50 score calibration.",
        "- In compressed 28-day ranges, funding events require a completed-daily deep pullback; premium events keep only their source score gate.",
        "",
        "## Frozen selection evidence",
        "",
        "| Window | Result |",
        "|---|---:|",
    ]
    for name in ("train", "select_2023", "select_2023_h1", "select_2023_h2", "pre_2024"):
        lines.append(f"| {name} | {_metric(manifest['selected_stats'][name])} |")
    lines += [
        "",
        "## Frozen OOS gate",
        "",
        "| Window | Minimum ratio | Maximum MDD | Minimum trades |",
        "|---|---:|---:|---:|",
    ]
    for name, gate in OOS_GATE.items():
        lines.append(f"| {name} | {gate['min_ratio']:.1f} | {gate['max_mdd']:.1f}% | {gate['min_trades']} |")
    lines += ["", "All windows must also have positive absolute return. No threshold, feature, exit, or gate may change after this freeze.", ""]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")


def run(cfg: Config) -> dict[str, Any]:
    manifest = _write_once(cfg.output, build_manifest(cfg))
    _write_docs(cfg.docs_output, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-result", default=Config.selection_result)
    parser.add_argument("--audit-result", default=Config.audit_result)
    parser.add_argument("--output", default=Config.output)
    parser.add_argument("--docs-output", default=Config.docs_output)
    return parser.parse_args()


def main() -> None:
    manifest = run(Config(**vars(parse_args())))
    print(json.dumps({"phase": manifest["phase"], "manifest_hash": manifest["manifest_hash"], "oos_opened": manifest["oos_opened"]}, indent=2))


if __name__ == "__main__":
    main()
