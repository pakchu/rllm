"""Preregister the interval-corrected fresh-forward PPOSM inventory.

V2 proved every raw source gate but compared its complete warmup-plus-forward
5-minute frame to a forward-only grid.  V3 changes only that check input: the
full frame remains available to feature construction and parity, while the
exact grid gate receives a physical forward-window slice.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training import preregister_pposm_fresh_forward_signal_inventory_v2 as v2

POLICY_ID = "pposm_fresh_forward_signal_inventory_v3"
DEFAULT_OUTPUT = Path(
    "results/pposm_fresh_forward_signal_inventory_v3_preregistration_2026-09-05.json"
)
DEFAULT_MANIFEST = v2.DEFAULT_MANIFEST
DEFAULT_CACHE = v2.DEFAULT_CACHE
SOURCE_ROOT = v2.SOURCE_ROOT
QUERY_START = v2.QUERY_START
FORWARD_START = v2.FORWARD_START
FORWARD_END_EXCLUSIVE = v2.FORWARD_END_EXCLUSIVE
FORWARD_LAST_DECISION = v2.FORWARD_LAST_DECISION
CACHE_PRECEDENCE_BEFORE = v2.CACHE_PRECEDENCE_BEFORE
PARITY_START = v2.PARITY_START
PARITY_END_EXCLUSIVE = v2.PARITY_END_EXCLUSIVE
PARITY_ATOL = v2.PARITY_ATOL
PARITY_RTOL = v2.PARITY_RTOL
STATE_FEATURE_COLUMNS = v2.STATE_FEATURE_COLUMNS
ACTIVE_FEATURE_COLUMNS = v2.ACTIVE_FEATURE_COLUMNS
PARITY_FEATURE_COLUMNS = v2.PARITY_FEATURE_COLUMNS
READ_ONLY_QUERIES = v2.READ_ONLY_QUERIES
SYMBOL = v2.SYMBOL

V2_FAILURE_ARTIFACT = Path(
    "results/pposm_fresh_forward_signal_inventory_v2_2026-09-05.json"
)
V2_FAILURE_ARTIFACT_SHA256 = (
    "eb67715f1aad0cbb3f0a9027f5deb57689131633ff2edb09dc2f1576b475651e"
)
V2_FAILURE_RESULT_HASH = (
    "6bc841b6e44a83ac4c9cabddb1d76333a8281b35bd898757d10376bf6d64f7ab"
)
FORWARD_GRID_CHECK_POLICY = {
    "full_resampled_frame_scope": [QUERY_START, FORWARD_END_EXCLUSIVE],
    "checked_frame_scope": [FORWARD_START, FORWARD_END_EXCLUSIVE],
    "slice_rule": "FORWARD_START <= date < FORWARD_END_EXCLUSIVE",
    "full_frame_retained_for": ["May cache parity", "causal feature warmup", "forward feature construction"],
    "expected_full_rows": 45_337,
    "expected_checked_rows": 27_481,
    "expected_warmup_rows_excluded_from_check": 17_856,
}


@dataclass(frozen=True)
class Config:
    output: Path = DEFAULT_OUTPUT
    manifest: Path = DEFAULT_MANIFEST
    cache: Path = DEFAULT_CACHE


canonical_json = v2.canonical_json
sha256_bytes = v2.sha256_bytes
sha256_file = v2.sha256_file


def _v2_failure_receipt() -> dict[str, Any]:
    if sha256_file(V2_FAILURE_ARTIFACT) != V2_FAILURE_ARTIFACT_SHA256:
        raise RuntimeError("v2 failure artifact bytes changed")
    payload = json.loads(V2_FAILURE_ARTIFACT.read_text(encoding="utf-8"))
    required = {
        "result_hash": V2_FAILURE_RESULT_HASH,
        "terminal": "source_incomplete",
        "forward_counted": False,
        "opened_outcomes": False,
        "trained": False,
    }
    for key, value in required.items():
        if payload.get(key) != value:
            raise RuntimeError(f"v2 failure artifact {key} mismatch")
    checks = payload.get("source_completeness", {}).get("checks", {})
    forward = checks.get("forward_5m", {})
    if forward.get("observed_rows") != 45_337 or forward.get("extra_count") != 17_856:
        raise RuntimeError("v2 failure is not the preregistered warmup-slice defect")
    return required


def code_hashes() -> dict[str, str]:
    project_root = Path(__file__).resolve().parents[1]
    builder_path = project_root / "training/build_pposm_fresh_forward_signal_inventory_v3.py"
    hashes = {f"v2.{key}": value for key, value in v2.code_hashes().items()}
    hashes.update(
        {
            "preregistration_module": sha256_file(__file__),
            "builder_module": sha256_file(builder_path),
            "v2_failure_artifact": sha256_file(V2_FAILURE_ARTIFACT),
            "forward_grid_check_policy": sha256_bytes(
                canonical_json(FORWARD_GRID_CHECK_POLICY).encode()
            ),
        }
    )
    for name in ("forward_grid_frame", "check_source_completeness", "build_from_frames"):
        hashes[f"builder_local.{name}"] = v2.file_function_hash(builder_path, name)
    return hashes


def build_preregistration(cfg: Config = Config()) -> dict[str, Any]:
    receipt = _v2_failure_receipt()
    base = v2.build_preregistration(
        v2.Config(manifest=Path(cfg.manifest), cache=Path(cfg.cache))
    )
    source_contract = dict(base["source_contract"])
    source_contract.update(
        {
            "v2_failure_artifact": str(V2_FAILURE_ARTIFACT),
            "v2_failure_artifact_sha256": V2_FAILURE_ARTIFACT_SHA256,
            "v2_failure_result_hash": V2_FAILURE_RESULT_HASH,
            "v2_failure_receipt": receipt,
            "forward_grid_check_policy": FORWARD_GRID_CHECK_POLICY,
        }
    )
    payload = {
        **base,
        "policy_id": POLICY_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "outcome-blind fresh-forward causal signal inventory with interval-corrected 5m completeness",
        "source_contract": source_contract,
        "code_hashes": code_hashes(),
    }
    payload["preregistration_hash"] = sha256_bytes(
        canonical_json(
            {k: value for k, value in payload.items() if k not in {"created_at", "preregistration_hash"}}
        ).encode()
    )
    return payload


def write_preregistration(cfg: Config = Config()) -> dict[str, Any]:
    output = Path(cfg.output)
    payload = build_preregistration(cfg)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing.get("preregistration_hash") != payload["preregistration_hash"]:
            raise RuntimeError("refusing to overwrite different PPOSM v3 preregistration")
        return existing
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    return parser.parse_args()


def main() -> None:
    payload = write_preregistration(Config(**vars(parse_args())))
    print(
        json.dumps(
            {
                "policy_id": payload["policy_id"],
                "preregistration_hash": payload["preregistration_hash"],
                "db_rows_opened": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
