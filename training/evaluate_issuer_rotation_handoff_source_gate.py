"""Retire IRH-36 at its earliest outcome-blind source-support failure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


PROTOCOL_VERSION = "issuer_rotation_handoff_source_gate_v1"
POLICY_ID = "IRH-36"
AS_OF_DATE = "2026-07-21"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MECHANISM_FREEZE = Path("docs/issuer-rotation-handoff-mechanism-freeze-2026-07-21.md")
MECHANISM_FREEZE_SHA256 = (
    "24dd0c2621f7f8246bd5c150fe0e6321803f58246e1ef5ec178db36a7f757de9"
)
SOURCE_MANIFEST = Path(
    "results/ethereum_stablecoin_issuance_redemption_source_manifest_2026-07-21.json"
)
SOURCE_MANIFEST_SHA256 = (
    "8ec9ab08c413bf6f5f8170fb800b05105522d4cf1a7932943c214288701e31fe"
)
DEFAULT_OUTPUT = Path("results/issuer_rotation_handoff_source_gate_2026-07-21.json")
EVALUATOR_SOURCE = Path("training/evaluate_issuer_rotation_handoff_source_gate.py")
PRIOR_ROWS_REQUIRED = 32
MINIMUM_ROWS_FOR_ANY_TAIL_EVENT = PRIOR_ROWS_REQUIRED + 1
REQUIRED_REDEEM_KEY = "usdt_eth:redeem"
OBSERVED_REDEEM_ROWS = 3

OUTCOME_BOUNDARY = {
    "mechanism_freeze_bytes_hashed": True,
    "source_manifest_json_parsed": True,
    "source_csv_bytes_hashed": True,
    "source_csv_rows_read": 0,
    "pair_incidence_rows_derived": 0,
    "comparator_clock_rows_read": 0,
    "btc_market_rows_read": 0,
    "funding_rows_read": 0,
    "future_return_rows_read": 0,
    "return_or_pnl_fields_read": 0,
    "post_2023_event_rows_read": 0,
    "network_calls": 0,
    "subprocess_calls": 0,
}

TOP_LEVEL_KEYS = frozenset(
    {
        "protocol_version",
        "policy_id",
        "as_of_date",
        "mechanism_freeze",
        "source",
        "evaluator",
        "gate_contract",
        "earliest_failure",
        "decision",
        "authorization",
        "outcomes_opened",
        "outcome_boundary",
        "manifest_hash",
    }
)


def _path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(_path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("IRH source gate expected a JSON object")
    return payload


def _load_source_manifest() -> dict[str, Any]:
    if sha256_file(SOURCE_MANIFEST) != SOURCE_MANIFEST_SHA256:
        raise RuntimeError("IRH source manifest file hash mismatch")
    payload = _read_json(SOURCE_MANIFEST)
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("IRH source manifest hash mismatch")
    boundary = payload.get("outcome_boundary")
    if not isinstance(boundary, dict) or boundary.get("source_only") is not True:
        raise RuntimeError("IRH source manifest is not source-only")
    forbidden = (
        "btc_market_rows_read",
        "funding_rows_read",
        "future_return_rows_read",
    )
    if any(boundary.get(key) != 0 for key in forbidden):
        raise RuntimeError("IRH source manifest opened a forbidden outcome input")
    if boundary.get("pnl_cagr_mdd_opened") is not False:
        raise RuntimeError("IRH source manifest opened economic outcomes")
    output = payload.get("output")
    if not isinstance(output, dict):
        raise RuntimeError("IRH source manifest lacks output metadata")
    output_path = output.get("path")
    output_sha256 = output.get("sha256")
    if not isinstance(output_path, str) or not isinstance(output_sha256, str):
        raise RuntimeError("IRH source output binding is malformed")
    if sha256_file(output_path) != output_sha256:
        raise RuntimeError("IRH source CSV hash mismatch")
    return payload


def _evaluate_support(event_counts: Mapping[str, Any]) -> dict[str, Any]:
    observed = event_counts.get(REQUIRED_REDEEM_KEY)
    if not isinstance(observed, int) or observed < 0:
        raise RuntimeError("IRH source manifest has invalid USDT redeem count")
    possible_tail_events = max(0, observed - PRIOR_ROWS_REQUIRED)
    return {
        "required_event": REQUIRED_REDEEM_KEY,
        "strictly_prior_rows_required": PRIOR_ROWS_REQUIRED,
        "minimum_total_rows_for_any_tail_event": MINIMUM_ROWS_FOR_ANY_TAIL_EVENT,
        "observed_total_rows": observed,
        "maximum_possible_short_tail_events": possible_tail_events,
        "tail_validity_pass": possible_tail_events > 0,
        "short_template_possible": possible_tail_events > 0,
        "side_balance_possible": possible_tail_events > 0,
    }


def build_report() -> dict[str, Any]:
    if sha256_file(MECHANISM_FREEZE) != MECHANISM_FREEZE_SHA256:
        raise RuntimeError("IRH mechanism freeze hash mismatch")
    source = _load_source_manifest()
    event_counts = source.get("event_counts")
    if not isinstance(event_counts, dict):
        raise RuntimeError("IRH source manifest lacks event counts")
    failure = _evaluate_support(event_counts)
    if failure["observed_total_rows"] != OBSERVED_REDEEM_ROWS:
        raise RuntimeError("IRH frozen USDT redeem count drift")
    if failure["tail_validity_pass"]:
        raise RuntimeError(
            "IRH earliest source gate passed; full pair-incidence evaluator required"
        )
    source_output = source["output"]
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": POLICY_ID,
        "as_of_date": AS_OF_DATE,
        "mechanism_freeze": {
            "path": str(MECHANISM_FREEZE),
            "sha256": MECHANISM_FREEZE_SHA256,
        },
        "source": {
            "manifest_path": str(SOURCE_MANIFEST),
            "manifest_sha256": SOURCE_MANIFEST_SHA256,
            "manifest_hash": source["manifest_hash"],
            "csv_path": source_output["path"],
            "csv_sha256": source_output["sha256"],
            "event_rows": source_output["rows"],
            "event_counts": dict(sorted(event_counts.items())),
            "dual_replay_equal": source["dual_replay"]["canonical_replay_equal"],
            "header_cross_check": source["header_materialization"][
                "event_block_hash_cross_checked"
            ],
        },
        "evaluator": {
            "path": str(EVALUATOR_SOURCE),
            "sha256": sha256_file(EVALUATOR_SOURCE),
        },
        "gate_contract": {
            "tail_quantile": "strictly-prior nearest-rank 90th percentile",
            "minimum_prior_same_type_rows": PRIOR_ROWS_REQUIRED,
            "short_template": "usdt_eth:redeem + usdc_eth:mint",
            "side_balance_minimum_share": 0.30,
            "earliest_failure_short_circuit": True,
            "failure_action": "retire without repair or outcomes",
        },
        "earliest_failure": failure,
        "decision": {
            "status": "retired_before_pair_incidence",
            "pass": False,
            "reason": (
                "the complete source has only 3 USDT redeem rows, below the 33 "
                "rows required for even one strictly-prior SHORT tail event"
            ),
            "failed_gates": ["tail_validity", "side_balance"],
            "pair_incidence_opened": False,
            "comparator_novelty_opened": False,
            "economic_outcomes_opened": False,
            "repair_authorized": False,
        },
        "authorization": {
            "full_pair_support_evaluator": False,
            "comparator_clock_access": False,
            "outcome_evaluator": False,
            "post_2023_event_access": False,
            "next_action": "new independently frozen event-level mechanism only",
        },
        "outcomes_opened": False,
        "outcome_boundary": dict(OUTCOME_BOUNDARY),
    }
    payload["manifest_hash"] = canonical_hash(payload)
    validate_report(payload, verify_files=False)
    return payload


def validate_report(payload: Mapping[str, Any], *, verify_files: bool = True) -> None:
    if frozenset(payload) != TOP_LEVEL_KEYS:
        raise RuntimeError("IRH source gate top-level schema drift")
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("IRH source gate protocol drift")
    if payload.get("policy_id") != POLICY_ID or payload.get("as_of_date") != AS_OF_DATE:
        raise RuntimeError("IRH source gate identity drift")
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("IRH source gate manifest hash mismatch")
    if payload.get("outcomes_opened") is not False:
        raise RuntimeError("IRH source gate opened outcomes")
    if payload.get("outcome_boundary") != OUTCOME_BOUNDARY:
        raise RuntimeError("IRH source gate outcome boundary drift")
    expected_failure = _evaluate_support(
        payload.get("source", {}).get("event_counts", {})
        if isinstance(payload.get("source"), dict)
        else {}
    )
    if payload.get("earliest_failure") != expected_failure:
        raise RuntimeError("IRH source gate earliest failure drift")
    if (
        expected_failure["observed_total_rows"] != OBSERVED_REDEEM_ROWS
        or expected_failure["tail_validity_pass"] is not False
        or expected_failure["side_balance_possible"] is not False
    ):
        raise RuntimeError("IRH source gate decision drift")
    decision = payload.get("decision")
    if not isinstance(decision, dict):
        raise RuntimeError("IRH source gate decision missing")
    if (
        decision.get("pass") is not False
        or decision.get("repair_authorized") is not False
    ):
        raise RuntimeError("IRH source gate decision drift")
    if decision.get("failed_gates") != ["tail_validity", "side_balance"]:
        raise RuntimeError("IRH source gate failed gates drift")
    authorization = payload.get("authorization")
    if not isinstance(authorization, dict) or any(
        authorization.get(key) is not False
        for key in (
            "full_pair_support_evaluator",
            "comparator_clock_access",
            "outcome_evaluator",
            "post_2023_event_access",
        )
    ):
        raise RuntimeError("IRH source gate authorization drift")
    if verify_files:
        if sha256_file(MECHANISM_FREEZE) != MECHANISM_FREEZE_SHA256:
            raise RuntimeError("IRH mechanism freeze changed")
        if sha256_file(SOURCE_MANIFEST) != SOURCE_MANIFEST_SHA256:
            raise RuntimeError("IRH source manifest changed")
        evaluator = payload.get("evaluator")
        if not isinstance(evaluator, dict) or sha256_file(
            evaluator["path"]
        ) != evaluator.get("sha256"):
            raise RuntimeError("IRH source gate evaluator changed")
        source = payload.get("source")
        if (
            not isinstance(source, dict)
            or sha256_file(source["csv_path"]) != source["csv_sha256"]
        ):
            raise RuntimeError("IRH source CSV changed")


def run(output: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    payload = build_report()
    destination = _path(output)
    encoded = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if destination.exists() and destination.read_text(encoding="utf-8") != encoded:
        raise FileExistsError("existing IRH source gate differs from frozen report")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(encoded, encoding="utf-8")
    validate_report(payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args().output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
