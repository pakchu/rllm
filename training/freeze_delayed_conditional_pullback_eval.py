"""Freeze the delayed conditional-pullback candidate before opening 2025+."""

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

from training.audit_delayed_conditional_pullback_test2024 import (
    AUDIT_SPEC,
    DEFAULT_OUTPUT as AUDIT_RESULT,
    validate_audit,
)
from training.search_stable_ensemble_conditional_pullback_alpha import (
    EXPECTED_SELECTED_QUANTILES,
    FEATURE_COLUMNS,
)

DEFAULT_OUTPUT = "results/delayed_conditional_pullback_eval_manifest_2026-07-15.json"
DEFAULT_DOCS = "docs/delayed-conditional-pullback-eval-freeze-2026-07-15.md"
AUDIT_COMMIT = "a6eb32d849fe1e09a2cd86bedb4b0aeb2758b6ac"
AUDIT_FILE_SHA256 = "8652cbf423fb9acc9c9997388d22e74e37b7cb1997a2b7906a6519a4315c6dca"
EVAL_WINDOWS = {
    "eval_2025": ("2025-01-01", "2026-01-01"),
    "holdout_2026h1": ("2026-01-01", "2026-06-02"),
    "eval_all": ("2025-01-01", "2026-06-02"),
}
EVAL_GATE = {
    "eval_2025": {"positive": True, "min_ratio": 3.0, "max_mdd": 15.0, "min_trades": 12},
    "holdout_2026h1": {"positive": True, "min_ratio": 3.0, "max_mdd": 15.0, "min_trades": 6},
    "eval_all": {"positive": True, "min_ratio": 3.0, "max_mdd": 15.0, "min_trades": 18},
}
MANIFEST_FIELDS = (
    "schema_version",
    "phase",
    "test_opened",
    "eval_opened",
    "test_end_exclusive",
    "source_commit",
    "source_artifact",
    "source_hashes_through_2024",
    "feature_hash_through_2024",
    "model_spec",
    "candidate_spec",
    "execution_spec",
    "selected_activation_hash",
    "selected_schedule_hashes",
    "selected_stats",
    "stability_summary",
    "eval_windows",
    "eval_gate",
)


@dataclass(frozen=True)
class Config:
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
    if manifest.get("phase") != "test_2024_frozen_for_eval":
        raise RuntimeError("manifest phase changed")
    if manifest.get("test_opened") is not True or manifest.get("eval_opened") is not False:
        raise RuntimeError("manifest is not sealed before eval")
    if tuple(key for key in manifest if key in MANIFEST_FIELDS) != MANIFEST_FIELDS:
        raise RuntimeError("manifest field order or membership changed")
    if manifest.get("manifest_hash") != manifest_hash(manifest):
        raise RuntimeError("manifest hash mismatch")
    if manifest.get("candidate_spec", {}).get("quantiles") != EXPECTED_SELECTED_QUANTILES:
        raise RuntimeError("candidate quantiles changed")
    if manifest.get("eval_windows") != {key: list(value) for key, value in EVAL_WINDOWS.items()}:
        raise RuntimeError("eval windows changed")
    if manifest.get("eval_gate") != EVAL_GATE:
        raise RuntimeError("eval gate changed")


def _write_once(path: str | Path, manifest: dict[str, Any]) -> dict[str, Any]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        validate_manifest(existing)
        if {key: existing[key] for key in MANIFEST_FIELDS} != {
            key: manifest[key] for key in MANIFEST_FIELDS
        }:
            raise RuntimeError("refusing to replace a different eval manifest")
        return existing
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=False), encoding="utf-8")
    temporary.replace(target)
    return manifest


def build_manifest(cfg: Config) -> dict[str, Any]:
    if sha256_file(cfg.audit_result) != AUDIT_FILE_SHA256:
        raise RuntimeError("test audit differs from its committed SHA-256")
    audit = json.loads(Path(cfg.audit_result).read_text(encoding="utf-8"))
    validate_audit(audit)
    selected = audit["selected_candidate"]
    core: dict[str, Any] = {
        "schema_version": 1,
        "phase": "test_2024_frozen_for_eval",
        "test_opened": True,
        "eval_opened": False,
        "test_end_exclusive": "2025-01-01",
        "source_commit": AUDIT_COMMIT,
        "source_artifact": {
            "path": cfg.audit_result,
            "file_sha256": AUDIT_FILE_SHA256,
            "audit_hash": audit["audit_hash"],
            "spec_hash": audit["spec_hash"],
        },
        "source_hashes_through_2024": audit["source_hashes_through_2024"],
        "feature_hash_through_2024": audit["feature_hash_through_2024"],
        "model_spec": {
            "family": "RandomForestRegressor responsibility ensemble",
            "seeds": AUDIT_SPEC["seeds"],
            "trees_per_seed": 2_000,
            "max_depth": 3,
            "min_samples_leaf": 16,
            "max_features": 0.7,
            "target": "48h net trade return",
            "fit_window": ["2020-07-01", "2023-01-01"],
            "anchor_cooldown_bars": 144,
            "prediction_reduction": "single-threaded fixed tree order",
            "information_delay_bars": 12,
            "current_source_identity_columns": ["funding_leg", "premium_leg"],
            "feature_columns": FEATURE_COLUMNS,
        },
        "candidate_spec": {
            "quantiles": EXPECTED_SELECTED_QUANTILES,
            "thresholds": selected["thresholds"],
            "interaction": (
                "funding score and (delayed 28-day width above threshold or "
                "delayed completed-daily range position below threshold); premium score only"
            ),
            "width_feature": "rex_2016_range_width_pct",
            "pullback_feature": "htf_1d_range_pos",
        },
        "execution_spec": {
            "decision_clock": ":00 after completed hour; market information delayed one additional hour",
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
        "selected_activation_hash": selected["activation_hash"],
        "selected_schedule_hashes": selected["schedule_hashes"],
        "selected_stats": selected["stats"],
        "stability_summary": audit["audit_summary"],
        "eval_windows": {key: list(value) for key, value in EVAL_WINDOWS.items()},
        "eval_gate": EVAL_GATE,
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
        "# Delayed conditional-pullback eval freeze — 2026-07-15",
        "",
        "**Frozen after 2024 test and before opening 2025+.**",
        "",
        f"- Test-audit commit: `{AUDIT_COMMIT}`.",
        f"- Manifest hash: `{manifest['manifest_hash']}`.",
        "- Fixed one-hour information delay; five deterministic 2,000-tree forests.",
        "",
        "Metric: absolute return / CAGR / strict MDD / CAGR-to-strict-MDD / trades.",
        "",
        "| Window | Frozen result |",
        "|---|---:|",
    ]
    for name in ("train", "select_2023", "pre_2024", "test_2024"):
        lines.append(f"| {name} | {_metric(manifest['selected_stats'][name])} |")
    lines += [
        "",
        "## Frozen eval gate",
        "",
        "| Window | Minimum ratio | Maximum MDD | Minimum trades |",
        "|---|---:|---:|---:|",
    ]
    for name, gate in EVAL_GATE.items():
        lines.append(f"| {name} | {gate['min_ratio']:.1f} | {gate['max_mdd']:.1f}% | {gate['min_trades']} |")
    lines += ["", "Every eval window must also have positive absolute return. Parameters may not change after this freeze.", ""]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")


def run(cfg: Config) -> dict[str, Any]:
    manifest = _write_once(cfg.output, build_manifest(cfg))
    _write_docs(cfg.docs_output, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-result", default=Config.audit_result)
    parser.add_argument("--output", default=Config.output)
    parser.add_argument("--docs-output", default=Config.docs_output)
    return parser.parse_args()


def main() -> None:
    manifest = run(Config(**vars(parse_args())))
    print(json.dumps({"phase": manifest["phase"], "manifest_hash": manifest["manifest_hash"], "eval_opened": manifest["eval_opened"]}, indent=2))


if __name__ == "__main__":
    main()
