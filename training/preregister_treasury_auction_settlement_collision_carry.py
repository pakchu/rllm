"""Freeze TASCC-72 before exact settlement-basket incidence or BTC outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


POLICY_ID = "TASCC-72-SOURCE-FAMILY-SEEN"
PROTOCOL_VERSION = "treasury_auction_settlement_collision_carry_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path("training/preregister_treasury_auction_settlement_collision_carry.py")
MECHANISM_DECISION = Path(
    "docs/treasury-auction-settlement-collision-carry-"
    "mechanism-decision-2026-07-23.md"
)
MECHANISM_DECISION_SHA256 = (
    "0837517b5891e0dd8cc320e023d37662fe6984af46767aca57edc91e7bf14286"
)
DEFAULT_OUTPUT = Path(
    "results/treasury_auction_settlement_collision_carry_"
    "preregistration_2026-07-23.json"
)

AUCTION_ROOT = Path("data/us_treasury_auction_demand_2016_2023")
AUCTION_PANEL = AUCTION_ROOT / "us_treasury_nominal_original_auctions_2016_2023.csv.gz"
AUCTION_PANEL_SHA256 = (
    "34a19163630c015a4f9d2671c95ca7cf7cc8a8ada024b3ef985405704fe0e4c1"
)
AUCTION_MANIFEST = AUCTION_ROOT / "build_manifest.json"
AUCTION_MANIFEST_SHA256 = (
    "6da6a3848e89c3418efcbf0d836fda34b537a2da87a8777b74670f3912ad94f2"
)
RAW_PAGES: tuple[Mapping[str, Any], ...] = (
    {
        "path": AUCTION_ROOT / "raw/auction_query_page_0.json.gz",
        "sha256": "6e609bdf4e6e859d3d957c638244070e999c343acf7793dc5e9b32988915564b",
    },
    {
        "path": AUCTION_ROOT / "raw/auction_query_page_1.json.gz",
        "sha256": "b20370eacc2c6f030483e49d7e6cf6db6d4dbfa89c8142a3b3e0b5540840d221",
    },
)
PANEL_ALLOWED_COLUMNS = (
    "auction_date",
    "result_available_at_utc",
    "original_security_term",
    "cusip",
    "source_complete",
)
RAW_ALLOWED_FIELDS = (
    "auctionDate",
    "issueDate",
    "cusip",
    "securityType",
    "originalSecurityTerm",
    "reopening",
)

COMPARATOR_SPECS: tuple[Mapping[str, Any], ...] = (
    {
        "name": "tadi_primary",
        "path": Path("results/treasury_auction_demand_impulse_preregistered_clock_2026-07-17.csv.gz"),
        "sha256": "9bb416413a0cfee5a5ebbdb73032e5889735e88098eaa1dc264b6d224fa489f6",
        "parser": "CSV entry_time where clock_mode == primary",
    },
    {
        "name": "dffb_primary",
        "path": Path("results/daily_treasury_fiscal_flow_breadth_primary_clock_2026-07-21.csv.gz"),
        "sha256": "df53e1a27fcbc6ea2c4bc3f462a557a75c76a98db3c362944dad0b4d74382978",
        "parser": "CSV entry_time_utc where clock == primary",
    },
    {
        "name": "flcc_primary",
        "path": Path("results/federal_liquidity_component_concordance_preregistered_clock_2026-07-17.csv.gz"),
        "sha256": "7ebb0450422d9265e46c596e0b6415b6a8816c66f5e0cbb9ccda14ca6cb4c67c",
        "parser": "CSV entry_time where clock_name == component_concordance_only",
    },
    {
        "name": "overnight_rrp_primary",
        "path": Path("results/overnight_rrp_flow_release_preregistered_clock_2026-07-17.csv.gz"),
        "sha256": "9f09bc88c9661441a33cee724e59524f57c0b021abff0fe81263e1a341b7b7b7",
        "parser": "CSV entry_time where clock_mode == primary",
    },
    {
        "name": "live_portfolio_pure_clocks",
        "path": Path("results/cchr_live_portfolio_pure_clocks_2026-07-21.csv.gz"),
        "sha256": "73d6efbd35b3be64b0fa04fa9c8cb2db25866ef884f19b1ae673949e22a42b08",
        "parser": "CSV entry_time grouped by candidate_id",
    },
)

HISTORY_BINDINGS: tuple[Mapping[str, Any], ...] = (
    {
        "name": "tadi_stage1_outcome_seen",
        "path": Path("results/treasury_auction_demand_impulse_stage1_2021_2022_2026-07-17.json"),
        "sha256": "794f954cbae97f10749c5f65a5b6eb51167d7769cb0ad625f4f656756f82a527",
        "historical_values_opened": True,
        "values_read_during_tascc_preregistration": 0,
    },
    {
        "name": "dffb_source_support_seen",
        "path": Path("results/daily_treasury_fiscal_flow_breadth_support_2026-07-21.json"),
        "sha256": "a5bf3b15f40f05d876b7603eaa3104cfa21a867fa3dd1aa4681b6b0875c8f549",
        "historical_values_opened": True,
        "values_read_during_tascc_preregistration": 0,
    },
)

PRIOR_RESEARCH_DISCLOSURE: Mapping[str, Any] = {
    "auction_source_hypothesis_number": 2,
    "broader_treasury_release_family_heavily_seen": True,
    "auction_source_values_and_aggregate_counts_opened": True,
    "tadi_2021_2022_btc_outcomes_opened_and_failed": True,
    "tadi_2023_outcome_opened": False,
    "dffb_auction_and_issue_date_union_opened": True,
    "dffb_btc_outcomes_opened": False,
    "tascc_exact_collision_incidence_opened": False,
    "tascc_market_outcomes_opened": False,
    "pristine_source_family_claim": False,
}

EXPECTED_OUTCOME_BOUNDARY: Mapping[str, Any] = {
    "source_file_bytes_hashed_during_preregistration": True,
    "source_manifest_metadata_parsed": True,
    "source_panel_value_rows_read_during_preregistration": 0,
    "source_raw_value_rows_read_during_preregistration": 0,
    "exact_tascc_baskets_derived": 0,
    "comparator_file_bytes_hashed_during_preregistration": True,
    "comparator_value_rows_read_during_preregistration": 0,
    "prior_outcome_artifact_bytes_hashed": True,
    "prior_outcome_value_rows_read_during_preregistration": 0,
    "btc_market_rows_read": 0,
    "funding_rows_read": 0,
    "future_return_rows_read": 0,
    "pnl_cagr_mdd_opened": False,
    "post_2023_raw_transport_rows_may_be_parsed_for_auction_date_key_filter": True,
    "post_2023_rows_materialized_into_tascc": 0,
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
        "auction_source_hypothesis_number": 2,
        "economic_hypothesis": {
            "mechanism": (
                "same-date settlement of belly and long original Treasury issuance "
                "concentrates duration supply and cash absorption"
            ),
            "side": "SHORT",
            "assimilation": "risk-liquidity drain over 72 elapsed hours",
            "demand_or_price_values_used": False,
        },
        "source_rows": {
            "universe": "original-issue nominal fixed-rate Treasury coupon auctions",
            "terms": ["2-Year", "3-Year", "5-Year", "7-Year", "10-Year", "20-Year", "30-Year"],
            "belly_terms": ["5-Year", "7-Year"],
            "long_terms": ["10-Year", "20-Year", "30-Year"],
            "reopening": "No",
            "source_complete_required": True,
            "panel_allowed_columns": list(PANEL_ALLOWED_COLUMNS),
            "raw_allowed_fields": list(RAW_ALLOWED_FIELDS),
            "join_key": ["auctionDate", "cusip"],
            "raw_transport_boundary": (
                "current raw pages may parse post-2023 objects; inspect auctionDate "
                "only, discard keys outside the frozen pre-2024 panel before "
                "materializing issueDate/term/other TASCC fields, and count all rows"
            ),
            "forbidden_values": [
                "bid_to_cover_ratio",
                "competitive_accepted_usd",
                "bidder_allocations",
                "indirect_competitive_share",
                "Treasury_yield_or_price",
                "crypto_state",
            ],
        },
        "primary_clock": {
            "grouping_key": "exact issueDate",
            "belly_distinct_terms_minimum": 1,
            "long_distinct_terms_minimum": 1,
            "settlement_marker": "issueDate 00:00 UTC",
            "component_result_availability": "each result_available_at_utc <= settlement marker",
            "late_component_action": "skip entire basket",
            "signal_time": "settlement marker",
            "side": "SHORT",
        },
        "execution": {
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
            "side_override": False,
        },
        "windows": {
            "source_warmup": ["2016-01-01T00:00:00Z", "2020-01-01T00:00:00Z"],
            "train": ["2020-01-01T00:00:00Z", "2023-01-01T00:00:00Z"],
            "selection": ["2023-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "sealed_from": "2024-01-01T00:00:00Z",
        },
        "source_controls": {
            "belly_settlement_calendar": "issue dates containing any belly term",
            "long_settlement_calendar": "issue dates containing any long term",
            "any_multitenor_settlement": "issue dates containing at least two distinct terms",
            "single_tenor_settlement": "issue dates containing exactly one distinct term",
            "auction_date_collision": "same belly-plus-long rule grouped by auctionDate",
            "term_year_permutation": (
                "deterministic SHA-256 permutation of term labels within auction year"
            ),
            "result_time_clock": "primary baskets signaled at latest component result time",
            "settlement_plus_7d": "primary settlement marker shifted +7 elapsed days",
        },
        "source_support_gates": {
            "train_total_minimum": 18,
            "selection_total_minimum": 8,
            "each_train_year_minimum": 6,
            "each_selection_half_year_minimum": 3,
            "train_active_quarters_minimum": 8,
            "maximum_month_share": 0.20,
            "maximum_quarter_share": 0.40,
            "maximum_calendar_gap_days": 90,
            "duplicate_accepted_issue_date_allowed": False,
            "duplicate_accepted_cusip_identity_allowed": False,
            "all_component_results_known_by_signal": True,
            "failure_action": "reject candidate without TASCC BTC outcomes or repair",
        },
        "mechanism_specificity_gates": {
            "auction_date_collision_maximum_primary_near_containment": 0.50,
            "term_year_permutation_maximum_primary_near_containment": 0.50,
            "near_window_elapsed_hours": 12,
            "component_and_superset_controls_are_report_only": [
                "belly_settlement_calendar",
                "long_settlement_calendar",
                "any_multitenor_settlement",
                "result_time_clock",
            ],
            "single_tenor_and_delayed_controls_are_report_only": True,
        },
        "novelty": {
            "comparators": [spec["name"] for spec in COMPARATOR_SPECS],
            "near_window_elapsed_hours": 12,
            "maximum_exact_entry_jaccard": 0.10,
            "maximum_tascc_to_comparator_near_containment": 0.35,
            "minimum_comparator_entries": 10,
            "side_ignored_at_source_stage": True,
        },
        "economic_controls": {
            "direction_flip": "exact primary clock long",
            "deterministic_random_side": "SHA-256 side assignment on exact primary clock",
            "belly_settlement_calendar": "component clock short",
            "long_settlement_calendar": "component clock short",
            "any_multitenor_settlement": "superset clock short",
            "single_tenor_settlement": "noncollision clock short",
            "auction_date_collision": "auction-date clock short",
            "term_year_permutation": "permuted-term clock short",
            "result_time_clock": "same baskets at latest result clock short",
            "settlement_plus_7d": "same baskets delayed seven days short",
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
            "minimum_trades_train": 18,
            "minimum_trades_selection": 8,
            "selection_h1_and_h2_absolute_return_positive": True,
            "calendar_month_cluster_signflip_p_maximum": 0.10,
            "primary_cagr_mdd_above_component_controls": True,
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
            "later_actions": ["TRADE_FIXED_SHORT", "ABSTAIN"],
            "may_create_clock_reverse_side_or_change_hold": False,
            "inputs_must_be_causal": True,
            "current_position_state_required": True,
            "reward_penalties": ["strict drawdown", "turnover"],
        },
        "stopping_rule": (
            "any identity, provenance, causality, support, specificity, novelty, "
            "train, or selection failure rejects TASCC-72-SOURCE-FAMILY-SEEN; "
            "repair requires a new identity frozen before access"
        ),
    }


def _source_binding() -> dict[str, Any]:
    if sha256_file(AUCTION_PANEL) != AUCTION_PANEL_SHA256:
        raise RuntimeError("TASCC auction panel hash mismatch")
    if sha256_file(AUCTION_MANIFEST) != AUCTION_MANIFEST_SHA256:
        raise RuntimeError("TASCC auction manifest hash mismatch")
    manifest = json.loads(_repository_path(AUCTION_MANIFEST).read_text(encoding="utf-8"))
    if manifest.get("output_sha256") != AUCTION_PANEL_SHA256:
        raise RuntimeError("TASCC manifest panel binding mismatch")
    expected_raw = {spec["sha256"] for spec in RAW_PAGES}
    observed_raw = {
        row.get("raw_gzip_sha256")
        for row in manifest.get("sources", [])
        if isinstance(row, dict)
    }
    if observed_raw != expected_raw:
        raise RuntimeError("TASCC manifest raw-page binding mismatch")
    raw_pages = []
    for spec in RAW_PAGES:
        observed = sha256_file(spec["path"])
        if observed != spec["sha256"]:
            raise RuntimeError("TASCC raw page hash mismatch")
        raw_pages.append(
            {
                "path": str(spec["path"]),
                "sha256": observed,
                "value_rows_read_during_preregistration": 0,
            }
        )
    return {
        "panel": str(AUCTION_PANEL),
        "panel_sha256": AUCTION_PANEL_SHA256,
        "panel_allowed_columns": list(PANEL_ALLOWED_COLUMNS),
        "panel_value_rows_read_during_preregistration": 0,
        "manifest": str(AUCTION_MANIFEST),
        "manifest_sha256": AUCTION_MANIFEST_SHA256,
        "manifest_metadata_parsed": True,
        "raw_pages": raw_pages,
        "raw_allowed_fields": list(RAW_ALLOWED_FIELDS),
        "raw_value_rows_read_during_preregistration": 0,
    }


def _hash_bindings(specs: Sequence[Mapping[str, Any]], *, history: bool) -> list[dict[str, Any]]:
    bindings: list[dict[str, Any]] = []
    for spec in specs:
        observed = sha256_file(spec["path"])
        if observed != spec["sha256"]:
            raise RuntimeError(f"TASCC binding hash mismatch: {spec['name']}")
        row = {
            "name": spec["name"],
            "path": str(spec["path"]),
            "sha256": observed,
            "read_mode": "raw bytes for SHA-256 only",
        }
        if history:
            row.update(
                {
                    "historical_values_opened": spec["historical_values_opened"],
                    "values_read_during_tascc_preregistration": 0,
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
        bindings.append(row)
    return bindings


def build_preregistration(*, verify_sources: bool = True) -> dict[str, Any]:
    if verify_sources:
        if sha256_file(MECHANISM_DECISION) != MECHANISM_DECISION_SHA256:
            raise RuntimeError("TASCC mechanism decision hash mismatch")
        source = _source_binding()
        comparators = _hash_bindings(COMPARATOR_SPECS, history=False)
        history = _hash_bindings(HISTORY_BINDINGS, history=True)
    else:
        source = {
            "panel": str(AUCTION_PANEL),
            "panel_sha256": AUCTION_PANEL_SHA256,
            "panel_allowed_columns": list(PANEL_ALLOWED_COLUMNS),
            "panel_value_rows_read_during_preregistration": 0,
            "manifest": str(AUCTION_MANIFEST),
            "manifest_sha256": AUCTION_MANIFEST_SHA256,
            "manifest_metadata_parsed": True,
            "raw_pages": [
                {
                    "path": str(spec["path"]),
                    "sha256": spec["sha256"],
                    "value_rows_read_during_preregistration": 0,
                }
                for spec in RAW_PAGES
            ],
            "raw_allowed_fields": list(RAW_ALLOWED_FIELDS),
            "raw_value_rows_read_during_preregistration": 0,
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
                "historical_values_opened": spec["historical_values_opened"],
                "values_read_during_tascc_preregistration": 0,
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
        "source_family_market_outcomes_previously_opened": True,
        "exact_source_incidence_opened": False,
        "outcomes_opened": False,
        "performance_values_opened": False,
        "prior_research_disclosure": dict(PRIOR_RESEARCH_DISCLOSURE),
        "outcome_boundary": dict(EXPECTED_OUTCOME_BOUNDARY),
        "preregistration_source": {
            "path": str(SCRIPT_PATH),
            "sha256": sha256_file(SCRIPT_PATH),
        },
        "next_action": "build exact source-only TASCC baskets, controls, and novelty clocks",
    }
    payload["manifest_hash"] = canonical_hash(payload)
    return payload


def validate_preregistration(
    payload: Mapping[str, Any], *, verify_sources: bool = True
) -> None:
    if payload.get("candidate") != POLICY_ID:
        raise RuntimeError("TASCC candidate identity drift")
    if payload.get("policy") != policy_payload():
        raise RuntimeError("TASCC frozen policy drift")
    if payload.get("policy_hash") != canonical_hash(policy_payload()):
        raise RuntimeError("TASCC policy hash mismatch")
    if payload.get("prior_research_disclosure") != PRIOR_RESEARCH_DISCLOSURE:
        raise RuntimeError("TASCC prior-research disclosure drift")
    if payload.get("outcome_boundary") != EXPECTED_OUTCOME_BOUNDARY:
        raise RuntimeError("TASCC outcome boundary drift")
    if payload.get("exact_source_incidence_opened") is not False:
        raise RuntimeError("TASCC exact incidence must remain unopened")
    if payload.get("outcomes_opened") is not False:
        raise RuntimeError("TASCC outcomes must remain unopened")
    if payload.get("performance_values_opened") is not False:
        raise RuntimeError("TASCC performance values must remain unopened")
    stored_manifest = payload.get("manifest_hash")
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if stored_manifest != canonical_hash(core):
        raise RuntimeError("TASCC canonical hash mismatch")
    if verify_sources:
        expected = build_preregistration(verify_sources=True)
        expected["config"] = dict(payload.get("config", {}))
        expected_core = {
            key: value for key, value in expected.items() if key != "manifest_hash"
        }
        expected["manifest_hash"] = canonical_hash(expected_core)
        if payload != expected:
            raise RuntimeError("TASCC preregistration differs from frozen build")


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
            raise RuntimeError("existing TASCC preregistration differs; refusing overwrite")
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
