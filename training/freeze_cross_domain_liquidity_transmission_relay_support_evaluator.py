"""Hash-freeze the committed CDLTR-72A source-only evaluator before incidence."""

# Pandas Timestamp stubs include NaT even though the imported constants are fixed.
# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Sequence

from training import (
    evaluate_cross_domain_liquidity_transmission_relay_support as evaluate,
)


PROTOCOL_VERSION = "cross_domain_liquidity_transmission_relay_evaluator_freeze_v1"
DEFAULT_OUTPUT = Path(
    "results/cross_domain_liquidity_transmission_relay_evaluator_freeze_2026-07-21.json"
)
FROZEN_EVALUATOR_COMMIT = "6900b42ecc7d64c708218fcf048290e52ceb7a46"
FROZEN_EVALUATOR_SHA256 = (
    "649a4d4da64df32c3acb66ccedc6ad607bc8abef6b247235ff42e837ab3992e1"
)


def _committed_evaluator() -> tuple[str, str]:
    source = str(evaluate.EVALUATOR_SOURCE)
    committed = subprocess.check_output(
        ["git", "show", f"{FROZEN_EVALUATOR_COMMIT}:{source}"],
        cwd=evaluate.REPOSITORY_ROOT,
    )
    committed_sha = hashlib.sha256(committed).hexdigest()
    if committed_sha != FROZEN_EVALUATOR_SHA256:
        raise RuntimeError("CDLTR v1 evaluator commit is not reproducible from Git")
    return FROZEN_EVALUATOR_COMMIT, committed_sha


def build_manifest(commit: str, source_sha256: str) -> dict[str, Any]:
    preregistration = evaluate._load_preregistration()
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "candidate": evaluate.POLICY_ID,
        "evaluator_source": str(evaluate.EVALUATOR_SOURCE),
        "evaluator_source_commit": commit,
        "evaluator_source_sha256": source_sha256,
        "preregistration": {
            "path": str(evaluate.PREREGISTRATION_ARTIFACT),
            "sha256": evaluate.PREREGISTRATION_ARTIFACT_SHA256,
            "manifest_hash": evaluate.PREREGISTRATION_MANIFEST_HASH,
            "policy_hash": preregistration["policy_hash"],
        },
        "frozen_contract": {
            "source_columns": {
                name: list(columns) for name, columns in evaluate.SOURCE_COLUMNS.items()
            },
            "clock_columns": list(evaluate.CLOCK_COLUMNS),
            "primary_clock": evaluate.PRIMARY_CLOCK,
            "controls": list(evaluate.CONTROL_NAMES),
            "support_limits": dict(evaluate.SUPPORT_LIMITS),
            "novelty_limits": dict(evaluate.NOVELTY_LIMITS),
            "evaluation_start_utc": evaluate.EVALUATION_START.isoformat(),
            "train_end_utc": evaluate.TRAIN_END.isoformat(),
            "evaluation_end_utc": evaluate.EVALUATION_END.isoformat(),
            "macro_ttl_hours": int(evaluate.MACRO_TTL.total_seconds() // 3_600),
            "relay_deadline_hours": int(
                evaluate.RELAY_DEADLINE.total_seconds() // 3_600
            ),
            "hold_hours": int(evaluate.HOLD.total_seconds() // 3_600),
            "exposure_grid": "full [2021-01-01, 2024-01-01) UTC at 5m",
        },
        "opened_source_value_rows": 0,
        "derived_real_event_rows": 0,
        "opened_comparator_event_rows": 0,
        "opened_comparator_manifest_values": 0,
        "opened_btc_market_rows": 0,
        "opened_funding_rows": 0,
        "opened_return_rows": 0,
        "opened_pnl_or_equity_fields": 0,
        "opened_post_2023_observation_rows": 0,
        "network_calls": 0,
        "economic_simulation_run": False,
        "mutable_parameters": [],
        "source_support_artifact_exists_before_freeze": False,
        "source_support_clock_exists_before_freeze": False,
        "next_action": "run the exact frozen source-only support and novelty evaluator",
    }
    return {**core, "manifest_hash": evaluate.canonical_hash(core)}


def validate_manifest(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if evaluate.canonical_hash(core) != payload.get("manifest_hash"):
        raise RuntimeError("CDLTR evaluator freeze manifest hash mismatch")
    zero_fields = (
        "opened_source_value_rows",
        "derived_real_event_rows",
        "opened_comparator_event_rows",
        "opened_comparator_manifest_values",
        "opened_btc_market_rows",
        "opened_funding_rows",
        "opened_return_rows",
        "opened_pnl_or_equity_fields",
        "opened_post_2023_observation_rows",
        "network_calls",
    )
    if any(payload.get(field) != 0 for field in zero_fields):
        raise RuntimeError("CDLTR evaluator freeze opened source incidence or outcomes")
    if payload.get("economic_simulation_run") is not False:
        raise RuntimeError("CDLTR evaluator freeze ran an economic simulation")
    if payload.get("mutable_parameters") != []:
        raise RuntimeError("CDLTR evaluator freeze permits parameter repair")
    if payload.get("source_support_artifact_exists_before_freeze") is not False:
        raise RuntimeError("CDLTR source-support artifact predates evaluator freeze")
    if payload.get("source_support_clock_exists_before_freeze") is not False:
        raise RuntimeError("CDLTR source-support clock predates evaluator freeze")


def freeze(output: str | Path = DEFAULT_OUTPUT) -> tuple[str, dict[str, Any]]:
    if evaluate._repository_path(evaluate.DEFAULT_OUTPUT_REPORT).exists():
        raise RuntimeError("CDLTR source-support artifact predates evaluator freeze")
    if evaluate._repository_path(evaluate.DEFAULT_OUTPUT_CLOCK).exists():
        raise RuntimeError("CDLTR source-support clock predates evaluator freeze")
    commit, source_sha = _committed_evaluator()
    payload = build_manifest(commit, source_sha)
    validate_manifest(payload)
    target = evaluate._repository_path(output)
    if target in evaluate._protected_paths():
        raise ValueError("CDLTR evaluator freeze aliases a protected input")
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        validate_manifest(existing)
        if existing != payload:
            raise RuntimeError("refusing to overwrite CDLTR evaluator freeze")
        return "verified_existing", payload
    with target.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
    return "created", payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    status, payload = freeze(args.output)
    print(
        json.dumps(
            {
                "status": status,
                "candidate": payload["candidate"],
                "evaluator_source_commit": payload["evaluator_source_commit"],
                "manifest_hash": payload["manifest_hash"],
                "source_values_opened": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
