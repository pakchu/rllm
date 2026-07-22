"""Freeze BIRB-120 before exact SEC relay incidence or BTC outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


POLICY_ID = "BIRB-120-SOURCE-FAMILY-SEEN"
PROTOCOL_VERSION = "sec_bitcoin_issuer_reactivation_breadth_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path("training/preregister_sec_bitcoin_issuer_reactivation_breadth.py")
MECHANISM_DECISION = Path(
    "docs/sec-bitcoin-issuer-reactivation-breadth-"
    "mechanism-decision-2026-07-23.md"
)
MECHANISM_DECISION_SHA256 = (
    "11f6e9448bbf55bc9b8e5483bde3b0a204f2408b4c01f485598e2c5592c7e237"
)
DEFAULT_OUTPUT = Path(
    "results/sec_bitcoin_issuer_reactivation_breadth_"
    "preregistration_2026-07-23.json"
)

SOURCE_ARTIFACT = Path("data/sec_edgar_bitcoin_8k_6k_source_2018_2023.jsonl.gz")
SOURCE_ARTIFACT_SHA256 = (
    "c8489dfe9b4ac25da8bea7653115e5b58a44fa897f2815eaf68bad354e10c6ce"
)
SOURCE_CANONICAL_ROWS_SHA256 = (
    "98793185f1e411d8c59736fb54c5ed529d539e81ccddf2c823f24127ecfcef0b"
)
SOURCE_AUDIT = Path("results/sec_edgar_bitcoin_8k_6k_source_audit_2026-07-21.json")
SOURCE_AUDIT_SHA256 = (
    "c1e11d1f5089378ac787fdb2a80474f0feec33d5fb2296fb0c3014d6f1fafec1"
)
SOURCE_MANIFEST_HASH = (
    "b4234f71b559a6b98e4056491f3b726191e9a89c2c0bec1e549249d93840f575"
)
SOURCE_ALLOWED_FIELDS = (
    "acceptance_datetime",
    "accession",
    "amendment",
    "ciks",
    "form",
)

COMPARATOR_SPECS: tuple[Mapping[str, Any], ...] = (
    {
        "name": "prior_microstructure_bundle",
        "path": Path("results/prior_microstructure_comparator_clock_bundle_2026-07-20.json"),
        "sha256": "c5584256140799b380973f9f376e5751ad754a81c9683473467b9d05af0bb9f0",
        "parser": "comparators.*.events[].signal_date; side ignored at source stage",
    },
    {
        "name": "bitmex_trollbox_semantic_clock",
        "path": Path("results/bitmex_trollbox_semantic_clock_2026-07-20.json"),
        "sha256": "af8687564614ec5a1cbd7a1438c908f687af7bd99ceede9539016e5c1b111bd4",
        "parser": "events[].entry_earliest; zero-side rows excluded",
    },
    {
        "name": "live_portfolio_pure_clocks",
        "path": Path("results/cchr_live_portfolio_pure_clocks_2026-07-21.csv.gz"),
        "sha256": "73d6efbd35b3be64b0fa04fa9c8cb2db25866ef884f19b1ae673949e22a42b08",
        "parser": "CSV entry_time grouped by candidate_id",
    },
)

PRIOR_RESEARCH_DISCLOSURE: Mapping[str, Any] = {
    "source_family_hypothesis_number": 3,
    "source_family": "SEC EDGAR Bitcoin-hit non-amendment 8-K/6-K",
    "source_metadata_and_aggregate_counts_opened": True,
    "prior_family_candidates": ["EBCT-72", "BPAX-120"],
    "prior_candidates_stopped_at": "frozen Gemma synthetic interface",
    "prior_sec_filing_bodies_classified": False,
    "prior_sec_candidate_market_outcomes_opened": False,
    "birb_exact_incidence_opened": False,
    "birb_market_outcomes_opened": False,
    "pristine_source_family_claim": False,
}

EXPECTED_OUTCOME_BOUNDARY: Mapping[str, Any] = {
    "source_file_bytes_hashed_during_preregistration": True,
    "source_value_rows_read_during_preregistration": 0,
    "source_audit_metadata_parsed": True,
    "exact_birb_reactivation_rows_derived": 0,
    "exact_birb_breadth_events_derived": 0,
    "comparator_file_bytes_hashed_during_preregistration": True,
    "comparator_value_rows_read_during_preregistration": 0,
    "sec_filing_body_rows_read": 0,
    "btc_market_rows_read": 0,
    "funding_rows_read": 0,
    "future_return_rows_read": 0,
    "pnl_cagr_mdd_opened": False,
    "post_2023_source_value_rows_read": 0,
    "network_calls": 0,
    "subprocess_calls": 0,
}


@dataclass(frozen=True)
class Config:
    output: str = str(DEFAULT_OUTPUT)


def _repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        raise RuntimeError("path must be repository-relative")
    resolved = (REPOSITORY_ROOT / candidate).resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError("path must remain repository-relative") from exc
    return resolved


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _repository_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def policy_payload() -> dict[str, Any]:
    return {
        "candidate": POLICY_ID,
        "singleton": True,
        "research_status": "source-family-seen_candidate-outcome-blind",
        "source_family_hypothesis_number": 3,
        "economic_hypothesis": {
            "mechanism": (
                "three distinct issuers returning to Bitcoin-hit SEC filings after "
                "at least one annual cycle indicate renewed cross-issuer engagement"
            ),
            "side": "LONG",
            "assimilation": "continuation for 120 elapsed hours after breadth first passage",
            "semantic_claim": (
                "none at filing level; breadth is directional object and filing prose is unopened"
            ),
        },
        "source_rows": {
            "forms": ["6-K", "8-K"],
            "amendments_emit": False,
            "accession_deduplication": True,
            "issuer_key": "smallest numeric CIK in frozen accession CIK set",
            "allowed_fields": list(SOURCE_ALLOWED_FIELDS),
            "filing_body_fetch": False,
            "llm_source_labeling": False,
        },
        "atomic_batches": {
            "historical_ready_time": "official acceptanceDateTime UTC + 60 elapsed minutes",
            "processing_sort": ["ready_time", "accession"],
            "grouping_key": "exact ready_time",
            "same_ready_rows_are_simultaneous": True,
            "intra_batch_order_forbidden": True,
            "state_update": "apply whole batch, then evaluate one crossing",
        },
        "primary_clock": {
            "reactivation_gap_elapsed_days_minimum": 365,
            "prior_hit": "immediately previous eligible accession for same issuer",
            "first_ever_issuer_hit": "not a primary reactivation",
            "breadth_window": "(current_ready - 7 elapsed days, current_ready]",
            "breadth_window_elapsed_days": 7,
            "distinct_issuer_threshold": 3,
            "signal": "first below-three to at-least-three transition caused by a new batch",
            "expiry_only_signal": False,
            "issuer_contribution_per_episode": 1,
            "state_reset": "breadth count falls below three",
            "side": "LONG",
        },
        "execution": {
            "entry_time": "ceil_to_5m(signal_ready) + 5 elapsed minutes",
            "exact_grid_signal_still_waits_one_bar": True,
            "hold_elapsed_hours": 120,
            "hold_bars_5m": 1440,
            "notional_exposure": 0.5,
            "global_nonoverlap": True,
            "reservation_interval": "[entry_time, exit_time)",
            "accept_when_entry_at_or_after_prior_exit": True,
            "suppressed_candidate_queueing": False,
            "entry_and_exit_same_split_required": True,
            "split_crossing_action": "skip",
            "stops_take_profit_or_trailing_exit": False,
            "side_override": False,
        },
        "windows": {
            "source_warmup": ["2018-01-01T00:00:00Z", "2020-01-01T00:00:00Z"],
            "train": ["2020-01-01T00:00:00Z", "2023-01-01T00:00:00Z"],
            "selection": ["2023-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "sealed_from": "2024-01-01T00:00:00Z",
        },
        "source_controls": {
            "first_ever_birth_breadth": "three distinct first-ever issuer hits in seven days",
            "any_mention_breadth": "three distinct eligible issuers in seven days",
            "repeat_filer_breadth": "three distinct issuers with prior gap below 90 days",
            "single_reactivation": "each 365-day reactivation emits directly",
            "stale_30d": "shift reactivation ready times +30 elapsed days before breadth",
            "year_cik_permutation": (
                "deterministic SHA-256 permutation of issuer keys within ready-time year"
            ),
            "threshold_two": "report-only breadth specificity control",
            "threshold_four": "report-only breadth specificity control",
        },
        "source_support_gates": {
            "train_total_minimum": 24,
            "selection_total_minimum": 8,
            "each_train_year_minimum": 6,
            "each_selection_half_year_minimum": 3,
            "train_distinct_reactivated_issuers_minimum": 18,
            "selection_distinct_reactivated_issuers_minimum": 6,
            "train_active_quarters_minimum": 8,
            "maximum_month_share": 0.20,
            "maximum_quarter_share": 0.40,
            "maximum_calendar_gap_days": 150,
            "duplicate_accepted_accession_identity_allowed": False,
            "failure_action": "reject candidate without BTC outcomes or repair",
        },
        "mechanism_specificity_gates": {
            "maximum_primary_to_birth_near_containment": 0.50,
            "maximum_primary_to_any_mention_near_containment": 0.60,
            "maximum_primary_to_repeat_near_containment": 0.50,
            "single_reactivation_component_relation": (
                "primary trigger is definitionally a raw reactivation timestamp"
            ),
            "single_reactivation_proximity_is_report_only": True,
            "maximum_primary_to_stale_near_containment": 0.50,
            "maximum_primary_to_permutation_near_containment": 0.50,
            "near_window_elapsed_hours": 12,
            "threshold_controls_are_report_only": True,
        },
        "novelty": {
            "comparators": [spec["name"] for spec in COMPARATOR_SPECS],
            "near_window_elapsed_hours": 12,
            "maximum_exact_entry_jaccard": 0.10,
            "maximum_birb_to_comparator_near_containment": 0.35,
            "minimum_comparator_entries": 10,
            "side_ignored_at_source_stage": True,
        },
        "economic_controls": {
            "direction_flip": "exact primary clock short",
            "deterministic_random_side": "SHA-256 side assignment on exact primary clock",
            "first_ever_birth_breadth": "source control clock long",
            "any_mention_breadth": "source control clock long",
            "repeat_filer_breadth": "source control clock long",
            "single_reactivation": "source control clock long",
            "stale_30d": "source control clock long",
            "time_shift_7d": "exact primary entry shifted +7 elapsed days",
        },
        "strict_economic_gates": {
            "absolute_return_positive": True,
            "cagr_to_strict_mdd_minimum": 3.0,
            "strict_mdd_pct_maximum": 15.0,
            "full_calendar_cagr": True,
            "strict_intratrade_high_water_mdd": True,
            "realized_funding": True,
            "base_cost_notional_per_side": 0.0006,
            "stress_cost_notional_per_side": 0.0010,
            "ten_bp_notional_side_stress_return_positive": True,
            "minimum_trades_train": 20,
            "minimum_trades_selection": 8,
            "selection_h1_and_h2_absolute_return_positive": True,
            "calendar_month_cluster_signflip_p_maximum": 0.10,
            "control_full_qualification_rejects": True,
        },
        "economic_sequence": [
            "source-only support, specificity, and novelty",
            "freeze strict evaluator",
            "train 2020-2022",
            "selection 2023 only after train pass",
            "immutable post-2023 source extension only after pre-2024 pass",
            "test 2024",
            "eval 2025",
            "recent 2026",
        ],
        "rllm_boundary": {
            "authorized_before_deterministic_train_and_selection_pass": False,
            "later_actions": ["TRADE_FIXED_LONG", "ABSTAIN"],
            "may_create_clock_reverse_side_or_change_hold": False,
            "inputs_must_be_causal": True,
            "current_position_state_required": True,
            "reward_penalties": ["strict drawdown", "turnover"],
        },
        "stopping_rule": (
            "any identity, source, causality, support, specificity, novelty, train, "
            "or staged selection failure rejects BIRB-120-SOURCE-FAMILY-SEEN; any "
            "repair requires a new identity frozen before access"
        ),
    }


def _validate_source_binding() -> dict[str, Any]:
    observed_source = sha256_file(SOURCE_ARTIFACT)
    observed_audit = sha256_file(SOURCE_AUDIT)
    observed_decision = sha256_file(MECHANISM_DECISION)
    if observed_source != SOURCE_ARTIFACT_SHA256:
        raise RuntimeError("SEC source artifact hash mismatch")
    if observed_audit != SOURCE_AUDIT_SHA256:
        raise RuntimeError("SEC source audit hash mismatch")
    if observed_decision != MECHANISM_DECISION_SHA256:
        raise RuntimeError("BIRB mechanism decision hash mismatch")
    audit = json.loads(_repository_path(SOURCE_AUDIT).read_text(encoding="utf-8"))
    if audit.get("manifest_hash") != SOURCE_MANIFEST_HASH:
        raise RuntimeError("SEC source audit manifest mismatch")
    if (
        audit.get("source_artifact", {}).get("canonical_rows_sha256")
        != SOURCE_CANONICAL_ROWS_SHA256
    ):
        raise RuntimeError("SEC canonical row hash mismatch")
    decision = audit.get("decision", {})
    if not decision.get("candidate_preregistration_authorized"):
        raise RuntimeError("SEC source audit does not authorize preregistration")
    if decision.get("economic_evaluation_authorized"):
        raise RuntimeError("SEC source audit unexpectedly authorizes economic access")
    return {
        "source": str(SOURCE_ARTIFACT),
        "source_sha256": observed_source,
        "canonical_rows_sha256": SOURCE_CANONICAL_ROWS_SHA256,
        "audit": str(SOURCE_AUDIT),
        "audit_sha256": observed_audit,
        "audit_manifest_hash": SOURCE_MANIFEST_HASH,
        "allowed_fields": list(SOURCE_ALLOWED_FIELDS),
        "value_rows_read_during_preregistration": 0,
    }


def comparator_bindings() -> list[dict[str, Any]]:
    bindings: list[dict[str, Any]] = []
    for spec in COMPARATOR_SPECS:
        observed = sha256_file(spec["path"])
        if observed != spec["sha256"]:
            raise RuntimeError(f"comparator hash mismatch: {spec['name']}")
        bindings.append(
            {
                "name": spec["name"],
                "path": str(spec["path"]),
                "sha256": observed,
                "parser": spec["parser"],
                "comparison": ["2020-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
                "read_mode": "raw bytes for SHA-256 only",
                "value_rows_read_during_preregistration": 0,
            }
        )
    return bindings


def build_preregistration(*, verify_sources: bool = True) -> dict[str, Any]:
    policy = policy_payload()
    source = _validate_source_binding() if verify_sources else {
        "source": str(SOURCE_ARTIFACT),
        "source_sha256": SOURCE_ARTIFACT_SHA256,
        "canonical_rows_sha256": SOURCE_CANONICAL_ROWS_SHA256,
        "audit": str(SOURCE_AUDIT),
        "audit_sha256": SOURCE_AUDIT_SHA256,
        "audit_manifest_hash": SOURCE_MANIFEST_HASH,
        "allowed_fields": list(SOURCE_ALLOWED_FIELDS),
        "value_rows_read_during_preregistration": 0,
    }
    comparators = comparator_bindings() if verify_sources else [
        {
            "name": spec["name"],
            "path": str(spec["path"]),
            "sha256": spec["sha256"],
            "parser": spec["parser"],
            "comparison": ["2020-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "read_mode": "raw bytes for SHA-256 only",
            "value_rows_read_during_preregistration": 0,
        }
        for spec in COMPARATOR_SPECS
    ]
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "candidate": POLICY_ID,
        "config": asdict(Config()),
        "policy": policy,
        "policy_hash": canonical_hash(policy),
        "mechanism_decision": {
            "path": str(MECHANISM_DECISION),
            "sha256": MECHANISM_DECISION_SHA256,
        },
        "source_binding": source,
        "comparator_bindings": comparators,
        "source_family_values_previously_opened": True,
        "exact_source_incidence_opened": False,
        "outcomes_opened": False,
        "performance_values_opened": False,
        "prior_research_disclosure": dict(PRIOR_RESEARCH_DISCLOSURE),
        "outcome_boundary": dict(EXPECTED_OUTCOME_BOUNDARY),
        "preregistration_source": {
            "path": str(SCRIPT_PATH),
            "sha256": sha256_file(SCRIPT_PATH),
        },
        "next_action": "build exact source-only BIRB support, controls, and novelty clocks",
    }
    payload["manifest_hash"] = canonical_hash(payload)
    return payload


def validate_preregistration(
    payload: Mapping[str, Any], *, verify_sources: bool = True
) -> None:
    if payload.get("candidate") != POLICY_ID:
        raise RuntimeError("BIRB candidate identity drift")
    if payload.get("policy") != policy_payload():
        raise RuntimeError("BIRB frozen policy drift")
    if payload.get("policy_hash") != canonical_hash(policy_payload()):
        raise RuntimeError("BIRB policy hash mismatch")
    if payload.get("prior_research_disclosure") != PRIOR_RESEARCH_DISCLOSURE:
        raise RuntimeError("BIRB prior-research disclosure drift")
    if payload.get("outcome_boundary") != EXPECTED_OUTCOME_BOUNDARY:
        raise RuntimeError("BIRB outcome boundary drift")
    if payload.get("exact_source_incidence_opened") is not False:
        raise RuntimeError("BIRB exact incidence must remain unopened")
    if payload.get("outcomes_opened") is not False:
        raise RuntimeError("BIRB outcomes must remain unopened")
    if payload.get("performance_values_opened") is not False:
        raise RuntimeError("BIRB performance values must remain unopened")
    stored_manifest = payload.get("manifest_hash")
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if stored_manifest != canonical_hash(core):
        raise RuntimeError("BIRB canonical hash mismatch")
    if verify_sources:
        expected = build_preregistration(verify_sources=True)
        expected["config"] = dict(payload.get("config", {}))
        expected_core = {
            key: value for key, value in expected.items() if key != "manifest_hash"
        }
        expected["manifest_hash"] = canonical_hash(expected_core)
        if payload != expected:
            raise RuntimeError("BIRB preregistration differs from frozen build")


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_preregistration(cfg: Config = Config()) -> tuple[dict[str, Any], str]:
    output = _repository_path(cfg.output)
    payload = build_preregistration()
    payload["config"] = asdict(cfg)
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    payload["manifest_hash"] = canonical_hash(core)
    validate_preregistration(payload)
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        validate_preregistration(existing)
        if existing != payload:
            raise RuntimeError("existing BIRB preregistration differs; refusing overwrite")
        return payload, "verified_existing"
    _atomic_write(output, payload)
    return payload, "created"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload, status = write_preregistration(Config(output=args.output))
    print(
        json.dumps(
            {
                "status": status,
                "candidate": payload["candidate"],
                "output": args.output,
                "policy_hash": payload["policy_hash"],
                "manifest_hash": payload["manifest_hash"],
                "outcomes_opened": payload["outcomes_opened"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
