"""Freeze SLCS-72 before exact source incidence or BTC outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


POLICY_ID = "SLCS-72-NEW-SOURCE"
PROTOCOL_VERSION = "soma_lending_collateral_scarcity_preregistration_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path("training/preregister_soma_lending_collateral_scarcity.py")
MECHANISM_DECISION = Path(
    "docs/soma-lending-collateral-scarcity-mechanism-decision-2026-07-23.md"
)
MECHANISM_DECISION_SHA256 = (
    "dca45850122efdf90fe266c267bcad8bb33e1526a97bcf17dfd9529b5ba8a325"
)
DEFAULT_OUTPUT = Path(
    "results/soma_lending_collateral_scarcity_preregistration_2026-07-23.json"
)
SOURCE_ROOT = Path("data/new_york_fed_securities_lending_2019_2023")
OPERATIONS = SOURCE_ROOT / "new_york_fed_securities_lending_operations_2019_2023.csv.gz"
OPERATIONS_SHA256 = (
    "99eb8c37c05417789dfad7452c7b2ddc5b6b640078b87451f1c945158af77906"
)
DETAILS = SOURCE_ROOT / "new_york_fed_securities_lending_details_2019_2023.csv.gz"
DETAILS_SHA256 = (
    "27178d8738cb50c4e6c13f1e5940fcfdf4009e6979b006c42fb86fb399d0716d"
)
SOURCE_MANIFEST = SOURCE_ROOT / "build_manifest.json"
SOURCE_MANIFEST_SHA256 = (
    "58b9eb56728065d919978b8969e9bbb4bcb291f723a290d22045fe2ca3da2019"
)

COMPARATOR_SPECS: tuple[Mapping[str, Any], ...] = (
    {
        "name": "federal_liquidity_component_concordance",
        "path": Path("results/federal_liquidity_component_concordance_preregistered_clock_2026-07-17.csv.gz"),
        "sha256": "7ebb0450422d9265e46c596e0b6415b6a8816c66f5e0cbb9ccda14ca6cb4c67c",
        "parser": "entry_time grouped by clock_name",
    },
    {
        "name": "overnight_rrp_flow_release",
        "path": Path("results/overnight_rrp_flow_release_preregistered_clock_2026-07-17.csv.gz"),
        "sha256": "9f09bc88c9661441a33cee724e59524f57c0b021abff0fe81263e1a341b7b7b7",
        "parser": "entry_time where clock_mode == primary",
    },
    {
        "name": "sofr_rate_dislocation",
        "path": Path("results/sofr_rate_dislocation_preregistered_clock_2026-07-17.csv.gz"),
        "sha256": "391c42dd2b0d5b87ffcd73058dd9fa0c4d18fd2f535597effff5a4c8edea2e69",
        "parser": "entry_time where clock_mode == primary",
    },
    {
        "name": "fed_h8_deposit_migration",
        "path": Path("results/fed_h8_deposit_migration_preregistered_clock_2026-07-18.csv.gz"),
        "sha256": "20405f79b86861adcc784c81223baae1c40fdf3c73edda339578471a6a6d1b40",
        "parser": "entry_time where clock_mode == primary",
    },
    {
        "name": "live_portfolio_pure_clocks",
        "path": Path("results/cchr_live_portfolio_pure_clocks_2026-07-21.csv.gz"),
        "sha256": "73d6efbd35b3be64b0fa04fa9c8cb2db25866ef884f19b1ae673949e22a42b08",
        "parser": "entry_time grouped by candidate_id",
    },
)

HISTORY_BINDINGS: tuple[Mapping[str, Any], ...] = (
    {
        "name": "flcc_stage1_outcome_seen",
        "path": Path("results/federal_liquidity_component_concordance_stage1_2020_2022_2026-07-17.json"),
        "sha256": "10dc911ad06c7e523d612ff34675421388fefb94fa93e157bfac7e93bd1d82a6",
    },
    {
        "name": "overnight_rrp_stage1_outcome_seen",
        "path": Path("results/overnight_rrp_flow_release_stage1_2021_2022_2026-07-17.json"),
        "sha256": "57dcfc8d5cf945250f8e1ee18e95dc341d81c5dad372ead166c64ebc38e4d63d",
    },
    {
        "name": "sofr_stage1_outcome_seen",
        "path": Path("results/sofr_rate_dislocation_stage1_2021_2022_2026-07-17.json"),
        "sha256": "b8a3d4dbbf00102bf1c14a156ede95aea344f820fa15a3735e305e532ee7e88e",
    },
    {
        "name": "h8_stage1_outcome_seen",
        "path": Path("results/fed_h8_deposit_migration_stage1_2020_2022_2026-07-18.json"),
        "sha256": "3f5118077cdafb48ffb59fc6cec8e7643613861f921bfd78403097181c287a7f",
    },
)

PRIOR_RESEARCH_DISCLOSURE: Mapping[str, Any] = {
    "soma_securities_lending_candidate_number": 1,
    "exact_source_family_new_to_repository": True,
    "source_row_values_opened_during_source_audit": True,
    "aggregate_source_counts_and_null_patterns_opened": True,
    "candidate_ratios_ranks_states_and_incidence_opened": False,
    "slcs_market_outcomes_opened": False,
    "broader_usd_liquidity_source_families_heavily_seen": True,
    "related_macro_liquidity_btc_outcomes_previously_opened": True,
    "pristine_broad_liquidity_claim": False,
}

EXPECTED_OUTCOME_BOUNDARY: Mapping[str, Any] = {
    "source_file_bytes_hashed_during_preregistration": True,
    "source_manifest_metadata_parsed": True,
    "source_operation_value_rows_read_during_preregistration": 0,
    "source_detail_value_rows_read_during_preregistration": 0,
    "slcs_components_computed": 0,
    "slcs_states_or_events_derived": 0,
    "comparator_file_bytes_hashed_during_preregistration": True,
    "comparator_value_rows_read_during_preregistration": 0,
    "prior_outcome_artifact_bytes_hashed": True,
    "prior_outcome_value_rows_read_during_preregistration": 0,
    "btc_market_rows_read": 0,
    "funding_rows_read": 0,
    "future_return_rows_read": 0,
    "pnl_cagr_mdd_opened": False,
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
    components = {
        "demand_intensity": "sum(par_submitted) / sum(actual_available_to_borrow)",
        "weighted_fee": (
            "sum(par_accepted * weighted_average_rate) / sum(par_accepted); "
            "zero-award N/A rates excluded"
        ),
        "carry_intensity": "sum(outstanding_loans) / sum(actual_available_to_borrow)",
        "demand_breadth": (
            "count(par_submitted > 0) / count(actual_available_to_borrow > 0)"
        ),
    }
    return {
        "candidate": POLICY_ID,
        "singleton": True,
        "research_status": "new-source_candidate-incidence-blind",
        "economic_hypothesis": {
            "mechanism": (
                "agreement among four weak SOMA lending auction observables "
                "identifies Treasury-collateral scarcity or relief"
            ),
            "high_state_side": "SHORT",
            "low_state_side": "LONG",
            "transmission": "temporary dollar-funding pressure over 72 elapsed hours",
            "single_component_claim": False,
        },
        "source": {
            "operation_panel": str(OPERATIONS),
            "detail_panel": str(DETAILS),
            "join_key": ["operation_id", "operation_date", "available_at_utc"],
            "complete_required": True,
            "missing_required_value_action": "operation unavailable and continuity broken",
            "imputation_or_forward_fill": False,
            "post_2023_rows_allowed": False,
        },
        "components": components,
        "normalization": {
            "history_complete_operations": 252,
            "strict_prior_only": True,
            "current_operation_excluded": True,
            "midrank": "(count(prior < current) + 0.5*count(prior == current)) / 252",
            "unit_transform": "u = 2*midrank - 1",
            "tie_rule": "exact decimal equality",
            "expanding_fallback": False,
        },
        "state": {
            "positive_votes_minimum": 3,
            "negative_votes_minimum": 3,
            "positive_score_minimum": 0.50,
            "negative_score_maximum": -0.50,
            "score": "equal mean of four u components",
            "zero_component_votes": False,
            "trigger": "HIGH or LOW differs from immediately prior complete rank-ready state",
            "missing_operation_breaks_continuity": True,
            "first_state_after_break_can_trigger": False,
            "persistence_inside_same_state_trades": False,
        },
        "execution": {
            "signal_time": "source available_at_utc",
            "entry_time": "ceil_to_5m(signal_time) + 5 elapsed minutes",
            "exact_grid_signal_still_waits_one_bar": True,
            "hold_elapsed_hours": 72,
            "hold_bars_5m": 864,
            "notional_exposure": 0.5,
            "global_nonoverlap": True,
            "reservation_interval": "[entry_time, exit_time)",
            "accept_when_entry_at_or_after_prior_exit": True,
            "suppressed_candidate_queueing": False,
            "entry_and_exit_same_split_required": True,
            "split_crossing_action": "skip",
            "stops_take_profit_or_trailing_exit": False,
            "dynamic_size_or_side_override": False,
        },
        "windows": {
            "source_warmup": ["2019-01-01T00:00:00Z", "2020-01-01T00:00:00Z"],
            "train": ["2020-01-01T00:00:00Z", "2023-01-01T00:00:00Z"],
            "selection": ["2023-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "sealed_from": "2024-01-01T00:00:00Z",
        },
        "source_controls": {
            "demand_intensity_only": "single component +/-0.50 transition",
            "weighted_fee_only": "single component +/-0.50 transition",
            "carry_intensity_only": "single component +/-0.50 transition",
            "demand_breadth_only": "single component +/-0.50 transition",
            "mean_without_consensus": "mean score threshold without vote",
            "same_sign_without_magnitude": "three-of-four vote without score threshold",
            "one_operation_stale": "previous complete vector at current availability",
            "five_operation_stale": "five-complete-operation-old vector at current availability",
            "year_component_permutation": (
                "independent deterministic SHA-256 within-year component permutations"
            ),
        },
        "source_support_gates": {
            "train_total_minimum": 60,
            "each_train_year_minimum": 15,
            "train_each_side_minimum": 15,
            "selection_total_minimum": 18,
            "each_selection_half_minimum": 7,
            "selection_each_side_minimum": 4,
            "every_train_and_selection_quarter_active": True,
            "train_maximum_month_share": 0.15,
            "selection_maximum_month_share": 0.20,
            "maximum_accepted_entry_gap_elapsed_days": 45,
            "failure_action": "reject before outcomes without repair",
        },
        "novelty": {
            "comparators": [spec["name"] for spec in COMPARATOR_SPECS],
            "comparison_window": ["2020-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "one_to_one_tolerance_elapsed_hours": 24,
            "maximum_exact_entry_jaccard": 0.10,
            "maximum_slcs_one_day_containment": 0.35,
            "maximum_absolute_signed_exposure_correlation": 0.35,
            "minimum_comparator_entries": 10,
        },
        "economic_controls": [
            "exact_direction_flip",
            "deterministic_random_side",
            "constant_long",
            "constant_short",
            *list(components),
            "mean_without_consensus",
            "same_sign_without_magnitude",
            "one_operation_stale",
            "five_operation_stale",
            "year_component_permutation",
        ],
        "strict_economic_gates": {
            "absolute_return_positive": True,
            "cagr_to_strict_mdd_minimum": 3.0,
            "strict_mdd_pct_maximum": 15.0,
            "full_calendar_cagr": True,
            "strict_intratrade_high_water_mdd": True,
            "realized_funding": True,
            "base_cost_notional_per_side": 0.0006,
            "stress_cost_notional_per_side": 0.0010,
            "stress_return_positive": True,
            "cluster_aware_significance_required": True,
            "required_subperiods_positive": True,
            "component_stale_permutation_and_side_controls_must_fail": True,
        },
        "economic_sequence": [
            "source-only support, specificity, and novelty",
            "freeze strict evaluator",
            "train 2020-2022",
            "selection 2023 only after exact train pass",
            "immutable post-2023 source extension only after pre-2024 pass",
            "test 2024",
            "eval 2025",
            "recent 2026",
        ],
        "rllm_boundary": {
            "authorized_before_deterministic_train_and_selection_pass": False,
            "later_actions": ["TRADE_FIXED_SIDE", "ABSTAIN"],
            "may_create_clock_reverse_side_change_size_or_hold": False,
            "causal_text_inputs": [
                "component midranks",
                "state transition reasons",
                "current position",
                "time in position",
                "risk budget",
            ],
            "reward_penalties": ["strict drawdown", "turnover"],
        },
        "mutable_parameters": [],
        "stopping_rule": (
            "any provenance, causality, source-support, specificity, novelty, "
            "train, or selection failure retires SLCS-72-NEW-SOURCE unchanged"
        ),
    }


def _source_binding() -> dict[str, Any]:
    expected = {
        OPERATIONS: OPERATIONS_SHA256,
        DETAILS: DETAILS_SHA256,
        SOURCE_MANIFEST: SOURCE_MANIFEST_SHA256,
    }
    for path, digest in expected.items():
        if sha256_file(path) != digest:
            raise RuntimeError(f"SLCS source hash mismatch: {path}")
    manifest = json.loads(_repository_path(SOURCE_MANIFEST).read_text(encoding="utf-8"))
    if manifest.get("operations", {}).get("sha256") != OPERATIONS_SHA256:
        raise RuntimeError("SLCS operation manifest binding mismatch")
    if manifest.get("details", {}).get("sha256") != DETAILS_SHA256:
        raise RuntimeError("SLCS detail manifest binding mismatch")
    if manifest.get("research_boundary", {}).get("candidate_incidence_opened") is not False:
        raise RuntimeError("SLCS source manifest opened candidate incidence")
    return {
        "operations": str(OPERATIONS),
        "operations_sha256": OPERATIONS_SHA256,
        "operation_value_rows_read_during_preregistration": 0,
        "details": str(DETAILS),
        "details_sha256": DETAILS_SHA256,
        "detail_value_rows_read_during_preregistration": 0,
        "manifest": str(SOURCE_MANIFEST),
        "manifest_sha256": SOURCE_MANIFEST_SHA256,
        "manifest_metadata_parsed": True,
        "manifest_operation_rows": manifest["operations"]["rows"],
        "manifest_detail_rows": manifest["details"]["rows"],
    }


def _hash_bindings(
    specs: Sequence[Mapping[str, Any]], *, history: bool
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        observed = sha256_file(spec["path"])
        if observed != spec["sha256"]:
            raise RuntimeError(f"SLCS binding hash mismatch: {spec['name']}")
        row = {
            "name": spec["name"],
            "path": str(spec["path"]),
            "sha256": observed,
            "read_mode": "raw bytes for SHA-256 only",
        }
        if history:
            row.update(
                {
                    "historical_values_previously_opened": True,
                    "values_read_during_slcs_preregistration": 0,
                }
            )
        else:
            row.update(
                {
                    "parser": spec["parser"],
                    "comparison": ["2020-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
                    "value_rows_read_during_preregistration": 0,
                }
            )
        rows.append(row)
    return rows


def build_preregistration(*, verify_sources: bool = True) -> dict[str, Any]:
    if verify_sources:
        if sha256_file(MECHANISM_DECISION) != MECHANISM_DECISION_SHA256:
            raise RuntimeError("SLCS mechanism decision hash mismatch")
        source = _source_binding()
        comparators = _hash_bindings(COMPARATOR_SPECS, history=False)
        history = _hash_bindings(HISTORY_BINDINGS, history=True)
    else:
        source = {
            "operations": str(OPERATIONS),
            "operations_sha256": OPERATIONS_SHA256,
            "operation_value_rows_read_during_preregistration": 0,
            "details": str(DETAILS),
            "details_sha256": DETAILS_SHA256,
            "detail_value_rows_read_during_preregistration": 0,
            "manifest": str(SOURCE_MANIFEST),
            "manifest_sha256": SOURCE_MANIFEST_SHA256,
            "manifest_metadata_parsed": True,
            "manifest_operation_rows": 1259,
            "manifest_detail_rows": 182616,
        }
        comparators = [
            {
                "name": spec["name"],
                "path": str(spec["path"]),
                "sha256": spec["sha256"],
                "read_mode": "raw bytes for SHA-256 only",
                "parser": spec["parser"],
                "comparison": ["2020-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
                "value_rows_read_during_preregistration": 0,
            }
            for spec in COMPARATOR_SPECS
        ]
        history = [
            {
                "name": spec["name"],
                "path": str(spec["path"]),
                "sha256": spec["sha256"],
                "read_mode": "raw bytes for SHA-256 only",
                "historical_values_previously_opened": True,
                "values_read_during_slcs_preregistration": 0,
            }
            for spec in HISTORY_BINDINGS
        ]
    policy = policy_payload()
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
        "history_bindings": history,
        "source_family_values_previously_opened": True,
        "source_family_market_outcomes_previously_opened": False,
        "exact_source_incidence_opened": False,
        "outcomes_opened": False,
        "performance_values_opened": False,
        "prior_research_disclosure": dict(PRIOR_RESEARCH_DISCLOSURE),
        "outcome_boundary": dict(EXPECTED_OUTCOME_BOUNDARY),
        "preregistration_source": {
            "path": str(SCRIPT_PATH),
            "sha256": sha256_file(SCRIPT_PATH),
        },
        "next_action": "build exact source-only SLCS and control clocks",
    }
    payload["manifest_hash"] = canonical_hash(payload)
    return payload


def validate_preregistration(
    payload: Mapping[str, Any], *, verify_sources: bool = True
) -> None:
    if payload.get("candidate") != POLICY_ID:
        raise RuntimeError("SLCS candidate identity drift")
    if payload.get("policy") != policy_payload():
        raise RuntimeError("SLCS frozen policy drift")
    if payload.get("policy_hash") != canonical_hash(policy_payload()):
        raise RuntimeError("SLCS policy hash mismatch")
    if payload.get("prior_research_disclosure") != PRIOR_RESEARCH_DISCLOSURE:
        raise RuntimeError("SLCS prior-research disclosure drift")
    if payload.get("outcome_boundary") != EXPECTED_OUTCOME_BOUNDARY:
        raise RuntimeError("SLCS outcome boundary drift")
    if payload.get("exact_source_incidence_opened") is not False:
        raise RuntimeError("SLCS exact incidence must remain unopened")
    if payload.get("outcomes_opened") is not False:
        raise RuntimeError("SLCS outcomes must remain unopened")
    if payload.get("performance_values_opened") is not False:
        raise RuntimeError("SLCS performance values must remain unopened")
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("SLCS canonical hash mismatch")
    if verify_sources:
        expected = build_preregistration(verify_sources=True)
        expected["config"] = dict(payload.get("config", {}))
        expected_core = {
            key: value for key, value in expected.items() if key != "manifest_hash"
        }
        expected["manifest_hash"] = canonical_hash(expected_core)
        if payload != expected:
            raise RuntimeError("SLCS preregistration differs from frozen build")


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
            raise RuntimeError("existing SLCS preregistration differs; refusing overwrite")
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
                "exact_source_incidence_opened": payload[
                    "exact_source_incidence_opened"
                ],
                "outcomes_opened": payload["outcomes_opened"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
