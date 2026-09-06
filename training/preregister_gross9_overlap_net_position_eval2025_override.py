"""Freeze a diagnostic eval2025 continuation after the terminal 2024 result."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from training import build_gross9_overlap_net_position_config as net_config
from training import evaluate_gross9_overlap_net_position_portfolio as original_evaluator
from training import preregister_gross9_overlap_net_position_validation as original_validation

POLICY_ID = "G9-OVERLAP-NET-PORT-1-EVAL2025-DIAG-OVERRIDE-V2"
PROTOCOL_VERSION = "gross9_overlap_net_position_eval2025_stop_override_freeze_v2"
AS_OF_DATE = "2026-09-04"
ORIGINAL_FREEZE = original_validation.DEFAULT_OUTPUT
HOLDOUT = original_evaluator.OUTPUTS["holdout_dec2023"]
TEST2024 = original_evaluator.OUTPUTS["test2024"]
EVALUATOR = Path("training/evaluate_gross9_overlap_net_position_eval2025_override.py")
FREEZER = Path("training/preregister_gross9_overlap_net_position_eval2025_override.py")
DEFAULT_OUTPUT = Path(
    "results/gross9_overlap_net_position_eval2025_override_freeze_v2_2026-09-04.json"
)
V1_FREEZE = Path(
    "results/gross9_overlap_net_position_eval2025_override_freeze_2026-09-04.json"
)
ATTEMPT_FAILURE = Path(
    "results/gross9_overlap_net_position_eval2025_attempt_infrastructure_failure_2026-09-04.json"
)


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    return original_validation.sha256_file(path)


def receipt(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    hash_key = "protocol_hash" if "protocol_hash" in value else "manifest_hash"
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        hash_key: value[hash_key],
    }


def _load_original_chain() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    frozen = original_evaluator.load_validation_freeze(ORIGINAL_FREEZE)
    holdout = original_validation.load_hashed_json(HOLDOUT)
    test = original_validation.load_hashed_json(TEST2024)
    freeze_receipt = original_evaluator.validation_freeze_receipt(ORIGINAL_FREEZE, frozen)
    if (
        holdout.get("policy_id") != original_validation.POLICY_ID
        or holdout.get("stage") != "holdout_dec2023"
        or holdout.get("validation_freeze") != freeze_receipt
        or holdout.get("implementation") != frozen["implementation"]
        or holdout.get("predecessor") is not None
        or holdout.get("passed") is not True
        or holdout.get("advance_to_next_stage") is not True
    ):
        raise RuntimeError(f"{POLICY_ID} passing holdout receipt drift")
    expected_holdout = {
        "stage": "holdout_dec2023",
        "path": str(HOLDOUT),
        "sha256": sha256_file(HOLDOUT),
        "manifest_hash": holdout["manifest_hash"],
    }
    if (
        test.get("policy_id") != original_validation.POLICY_ID
        or test.get("stage") != "test2024"
        or test.get("validation_freeze") != freeze_receipt
        or test.get("implementation") != frozen["implementation"]
        or test.get("predecessor") != expected_holdout
        or test.get("passed") is not False
        or test.get("advance_to_next_stage") is not False
        or test.get("status") != "terminal_reject_no_repair"
    ):
        raise RuntimeError(f"{POLICY_ID} terminal test2024 receipt drift")
    return frozen, holdout, test


def build() -> dict[str, Any]:
    frozen, holdout, test = _load_original_chain()
    if not EVALUATOR.is_file():
        raise RuntimeError(f"{POLICY_ID} evaluator missing before freeze")
    config = original_validation.load_hashed_json(net_config.CONFIG_OUTPUT)
    selected_weights = {
        row["sleeve_id"]: float(row["weight"])
        for row in frozen["frozen_inputs"]["selected_clocks"]
    }
    if selected_weights != config.get("sleeve_weights"):
        raise RuntimeError(f"{POLICY_ID} fixed weights drift")
    v1_freeze = original_validation.load_hashed_json(V1_FREEZE)
    attempt_failure = original_validation.load_hashed_json(ATTEMPT_FAILURE)
    if (
        attempt_failure.get("attempt_freeze") != receipt(V1_FREEZE, v1_freeze)
        or attempt_failure.get("economic_metrics_computed") is not False
        or attempt_failure.get("candidate_pass_fail_observed") is not False
        or attempt_failure.get("eval2025_returns_or_pnl_computed") is not False
        or attempt_failure.get("disposition")
        != "infrastructure_failure_reporting_only_successor_allowed"
    ):
        raise RuntimeError(f"{POLICY_ID} V1 infrastructure failure receipt drift")
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": POLICY_ID,
        "as_of_date": AS_OF_DATE,
        "original_policy_id": original_validation.POLICY_ID,
        "objective": "open eval2025 diagnostically after explicit user override of the 2024 stop",
        "original_chain": {
            "validation_freeze": receipt(ORIGINAL_FREEZE, frozen),
            "holdout_dec2023": receipt(HOLDOUT, holdout),
            "test2024_terminal": receipt(TEST2024, test),
            "original_protocol_terminal_reject_preserved": True,
        },
        "v1_infrastructure_failure": {
            "freeze": receipt(V1_FREEZE, v1_freeze),
            "attempt_failure": receipt(ATTEMPT_FAILURE, attempt_failure),
            "economic_metrics_computed": False,
            "candidate_pass_fail_observed": False,
        },
        "fixed_portfolio": {
            "sleeve_weights": selected_weights,
            "selected_clocks": frozen["frozen_inputs"]["selected_clocks"],
            "weights_changed": False,
            "rerank_repair_or_substitution_authorized": False,
        },
        "user_override": {
            "effective_date": AS_OF_DATE,
            "scope": "eval2025 diagnostic only",
            "overrides_original_stop_on_first_failure": True,
            "relabels_test2024_as_pass": False,
            "advance_beyond_eval2025_authorized": False,
            "turnover_frequency_and_cost_caps": "disclosure_only",
        },
        "eval2025": {
            "split": "eval",
            "window": ["2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
            "performance_checks": frozen["gates"]["oos"],
            "checks_are_reported_not_predecessor_authorizing": True,
            "funding_time_normalization": {
                "operation": "floor raw realized funding timestamp to its 5-minute bucket",
                "rationale": "the fixed ledger already owns funding by timestamp.floor('5min')",
                "rate_and_mark_price_changed": False,
                "duplicate_normalized_buckets_allowed": False,
                "normalization_is_reporting_only_infrastructure_repair": True,
            },
        },
        "implementation": {
            "evaluator": {"path": str(EVALUATOR), "sha256": sha256_file(EVALUATOR)},
            "freezer": {"path": str(FREEZER), "sha256": sha256_file(FREEZER)},
            "original_evaluator": frozen["implementation"]["evaluator"],
            "fixed_ledger": frozen["implementation"]["fixed_ledger"],
            "optimizer_utilities": frozen["implementation"]["optimizer_utilities"],
            "net_risk_metrics": frozen["implementation"]["net_risk_metrics"],
            "hash_bound_source_loader": frozen["implementation"][
                "hash_bound_source_loader"
            ],
        },
        "evidence_boundary": {
            "eval2025_market_or_funding_rows_opened": 0,
            "eval2025_outcomes_opened": False,
            "final2026_outcomes_opened": False,
        },
        "live_capital_authorized": False,
        "order_submission_enabled": False,
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: Mapping[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError(f"{POLICY_ID} freeze manifest drift")
    if value.get("protocol_version") != PROTOCOL_VERSION or value.get("policy_id") != POLICY_ID:
        raise RuntimeError(f"{POLICY_ID} freeze identity drift")
    override = value.get("user_override", {})
    if (
        override.get("scope") != "eval2025 diagnostic only"
        or override.get("overrides_original_stop_on_first_failure") is not True
        or override.get("relabels_test2024_as_pass") is not False
        or override.get("advance_beyond_eval2025_authorized") is not False
    ):
        raise RuntimeError(f"{POLICY_ID} user override drift")
    if value.get("original_chain", {}).get("original_protocol_terminal_reject_preserved") is not True:
        raise RuntimeError(f"{POLICY_ID} original terminal decision lost")
    if value.get("fixed_portfolio", {}).get("rerank_repair_or_substitution_authorized") is not False:
        raise RuntimeError(f"{POLICY_ID} no-repair boundary drift")
    if value.get("evidence_boundary", {}).get("eval2025_outcomes_opened") is not False:
        raise RuntimeError(f"{POLICY_ID} freeze opened eval outcomes")
    expected_implementation = {
        "evaluator",
        "freezer",
        "original_evaluator",
        "fixed_ledger",
        "optimizer_utilities",
        "net_risk_metrics",
        "hash_bound_source_loader",
    }
    implementation = value.get("implementation", {})
    if not isinstance(implementation, Mapping) or set(implementation) != expected_implementation:
        raise RuntimeError(f"{POLICY_ID} implementation receipts missing")
    for label, record in implementation.items():
        if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
            raise RuntimeError(f"{POLICY_ID} implementation receipt malformed: {label}")
        path = Path(str(record["path"]))
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"{POLICY_ID} implementation receipt drift: {label}")
    if dict(value) != build():
        raise RuntimeError(f"{POLICY_ID} freeze no longer matches exact bound chain/config")


def run(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    value = build()
    validate(value)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    value = run(args.output)
    print(
        json.dumps(
            {
                "policy_id": POLICY_ID,
                "output": str(args.output),
                "manifest_hash": value["manifest_hash"],
                "eval2025_outcomes_opened": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
