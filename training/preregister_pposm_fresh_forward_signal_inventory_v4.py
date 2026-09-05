"""Preregister cache-tail repair and full-context PPOSM source parity."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training import preregister_pposm_fresh_forward_signal_inventory_v3 as v3

POLICY_ID = "pposm_fresh_forward_signal_inventory_v4"
DEFAULT_OUTPUT = Path(
    "results/pposm_fresh_forward_signal_inventory_v4_preregistration_2026-09-05.json"
)
DEFAULT_MANIFEST = v3.DEFAULT_MANIFEST
DEFAULT_CACHE = v3.DEFAULT_CACHE
SOURCE_ROOT = v3.SOURCE_ROOT
QUERY_START = v3.QUERY_START
FORWARD_START = v3.FORWARD_START
FORWARD_END_EXCLUSIVE = v3.FORWARD_END_EXCLUSIVE
FORWARD_LAST_DECISION = v3.FORWARD_LAST_DECISION
PARITY_START = v3.PARITY_START
PARITY_END_EXCLUSIVE = "2026-05-31T15:05:00Z"
PARITY_ATOL = v3.PARITY_ATOL
PARITY_RTOL = v3.PARITY_RTOL
CACHE_EXPECTED_LAST = "2026-05-31T15:00:00Z"
DB_PRECEDENCE_FROM = "2026-05-31T15:05:00Z"
CACHE_LOADER_CUTOFF_ARGUMENT = "2026-06-02"

CACHE_DIAGNOSTIC_ARTIFACT = Path(
    "results/pposm_cache_tail_full_context_parity_diagnostic_2026-09-05.json"
)
CACHE_DIAGNOSTIC_ARTIFACT_SHA256 = (
    "c6980918a8a3de2afc4062a772781abb908f9601ab6640ed1cc641d0b607dbd5"
)
CACHE_DIAGNOSTIC_RESULT_HASH = (
    "cd8d4a3d5b86e5a9820f14062b03d66cdadfa1019ed86ba91fba3a53b1efbea7"
)
CONTEXT_AND_SEAM_POLICY = {
    "cache_expected_last": CACHE_EXPECTED_LAST,
    "cache_loader_cutoff_argument": CACHE_LOADER_CUTOFF_ARGUMENT,
    "cache_loader_cutoff_semantics": "same UTC instant as the frozen 2026-06-02 cutoff, represented without timezone only because the immutable CSV date column is naive",
    "parity_window": [PARITY_START, PARITY_END_EXCLUSIVE],
    "parity_expected_hourly_rows": 736,
    "parity_candidate_context": "immutable cache enriched prefix before QUERY_START plus DB enriched rows from QUERY_START; recompute the full feature frame",
    "final_cache_precedence_before": DB_PRECEDENCE_FROM,
    "final_db_precedence_from": DB_PRECEDENCE_FROM,
    "final_grid": "continuous 5min from immutable cache first row through FORWARD_END_EXCLUSIVE",
    "parity_tolerances_unchanged": {"atol": PARITY_ATOL, "rtol": PARITY_RTOL},
}


@dataclass(frozen=True)
class Config:
    output: Path = DEFAULT_OUTPUT
    manifest: Path = DEFAULT_MANIFEST
    cache: Path = DEFAULT_CACHE


canonical_json = v3.canonical_json
sha256_bytes = v3.sha256_bytes
sha256_file = v3.sha256_file


def _cache_diagnostic_receipt() -> dict[str, Any]:
    if sha256_file(CACHE_DIAGNOSTIC_ARTIFACT) != CACHE_DIAGNOSTIC_ARTIFACT_SHA256:
        raise RuntimeError("cache-tail diagnostic bytes changed")
    payload = json.loads(CACHE_DIAGNOSTIC_ARTIFACT.read_text(encoding="utf-8"))
    if payload.get("result_hash") != CACHE_DIAGNOSTIC_RESULT_HASH:
        raise RuntimeError("cache-tail diagnostic result_hash changed")
    cache = payload.get("cache_receipt", {})
    parity = payload.get("common_overlap", {}).get(
        "full_context_hybrid_feature_comparison", {}
    )
    seam = payload.get("seam_reconstruction", {})
    evidence = payload.get("evidence_boundary", {})
    if cache.get("date_max") != CACHE_EXPECTED_LAST:
        raise RuntimeError("diagnostic cache tail differs")
    if not parity.get("passed_at_atol_1e-10_rtol_1e-9"):
        raise RuntimeError("diagnostic full-context parity did not pass")
    if seam.get("missing_5m_rows") != 0 or not seam.get("pre2024_feature_prefix_hash_equal"):
        raise RuntimeError("diagnostic seam reconstruction did not pass")
    if evidence != {
        "pposm_forward_signal_count_computed": False,
        "economic_outcomes_opened": False,
        "trained": False,
    }:
        raise RuntimeError("cache-tail diagnostic crossed the evidence boundary")
    return {
        "result_hash": CACHE_DIAGNOSTIC_RESULT_HASH,
        "cache_tail": CACHE_EXPECTED_LAST,
        "full_context_parity_passed": True,
        "continuous_seam_passed": True,
        "signals_opened": False,
    }


def code_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    builder = root / "training/build_pposm_fresh_forward_signal_inventory_v4.py"
    hashes = {f"v3.{key}": value for key, value in v3.code_hashes().items()}
    hashes.update(
        {
            "preregistration_module": sha256_file(__file__),
            "builder_module": sha256_file(builder),
            "cache_diagnostic_artifact": sha256_file(CACHE_DIAGNOSTIC_ARTIFACT),
            "context_and_seam_policy": sha256_bytes(
                canonical_json(CONTEXT_AND_SEAM_POLICY).encode()
            ),
        }
    )
    for name in (
        "load_frozen_cache_bundle",
        "build_full_context_parity_candidate",
        "check_final_merged_grid",
        "build_from_frames",
    ):
        hashes[f"builder_local.{name}"] = v3.v2.file_function_hash(builder, name)
    return hashes


def build_preregistration(cfg: Config = Config()) -> dict[str, Any]:
    receipt = _cache_diagnostic_receipt()
    base = v3.build_preregistration(
        v3.Config(manifest=Path(cfg.manifest), cache=Path(cfg.cache))
    )
    source_contract = dict(base["source_contract"])
    source_contract.update(
        {
            "cache_precedence_before": DB_PRECEDENCE_FROM,
            "parity_window": [PARITY_START, PARITY_END_EXCLUSIVE],
            "cache_diagnostic_artifact": str(CACHE_DIAGNOSTIC_ARTIFACT),
            "cache_diagnostic_artifact_sha256": CACHE_DIAGNOSTIC_ARTIFACT_SHA256,
            "cache_diagnostic_result_hash": CACHE_DIAGNOSTIC_RESULT_HASH,
            "cache_diagnostic_receipt": receipt,
            "context_and_seam_policy": CONTEXT_AND_SEAM_POLICY,
        }
    )
    terminal_gate = dict(base["terminal_gate"])
    terminal_gate["parity_scope"] = (
        "full-context hybrid causal feature outputs at 736 hourly decisions in the exact common cache/DB overlap"
    )
    payload = {
        **base,
        "policy_id": POLICY_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "outcome-blind fresh-forward inventory with exact cache-tail seam and full-context parity",
        "source_contract": source_contract,
        "terminal_gate": terminal_gate,
        "code_hashes": code_hashes(),
    }
    payload["preregistration_hash"] = sha256_bytes(
        canonical_json(
            {key: value for key, value in payload.items() if key not in {"created_at", "preregistration_hash"}}
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
            raise RuntimeError("refusing to overwrite different PPOSM v4 preregistration")
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
    print(json.dumps({"policy_id": POLICY_ID, "preregistration_hash": payload["preregistration_hash"], "db_rows_opened": 0}, indent=2))


if __name__ == "__main__":
    main()
