"""Freeze ORPB-21 before source incidence, novelty, or outcomes are opened."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


PROTOCOL_VERSION = "overnight_rrp_participant_breadth_preregistration_v1"
POLICY_ID = "ORPB-21"
AS_OF_DATE = "2026-07-21"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DOCUMENT = Path("docs/overnight-rrp-participant-breadth-preregistration-2026-07-21.md")
DOCUMENT_SHA256 = "4e2a7e905e1733e18dbc96b5fb8ef046fba7bbdbf90049f76fda767c6ae5cee5"
GENERATOR = Path("training/preregister_overnight_rrp_participant_breadth.py")
DEFAULT_OUTPUT = Path(
    "results/overnight_rrp_participant_breadth_preregistration_2026-07-21.json"
)

SOURCE_BINDINGS: dict[str, dict[str, str]] = {
    "panel": {
        "path": (
            "data/new_york_fed_overnight_rrp_2018_2023/"
            "new_york_fed_overnight_rrp_2018-01-01_2023-12-31.csv.gz"
        ),
        "sha256": "49f67ed44b7eb81fd35c17a8209cf14d6a8019d7e9f77fce8c343d1a7fb66b27",
    },
    "build_manifest": {
        "path": "data/new_york_fed_overnight_rrp_2018_2023/build_manifest.json",
        "sha256": "4f87e2219da71c94832c8708086ba01387efc145e3488b62cd3b3d07c62d8fee",
    },
    "source_audit": {
        "path": "docs/new-york-fed-overnight-rrp-source-audit-2026-07-17.md",
        "sha256": "329db1cf886bfbceb0a048b1c44c59378af717ddd9731e5e26fd09e14ada8d23",
    },
    "source_builder": {
        "path": "training/build_new_york_fed_overnight_rrp.py",
        "sha256": "0567157dde18b1c6ccfb37b669ceead521360f23dd0b73033fccc08e37c0d42c",
    },
}

COMPARATOR_BINDINGS: dict[str, dict[str, Any]] = {
    "orfr_clocks": {
        "path": "results/overnight_rrp_flow_release_clocks_2026-07-17.csv.gz",
        "sha256": "7242d9870627dfc0cf067ff87d9664a1576dd374cb8985e927b40f15d1e3d480",
        "columns": [
            "control",
            "signal_time",
            "entry_time",
            "exit_time",
            "side",
            "operation_date",
        ],
        "identifier_columns": ["control"],
        "required_groups": {
            "one_day_delta_tail": 346,
            "one_release_delay": 327,
            "primary": 328,
        },
        "expected_rows": 1001,
    },
    "orfr_features": {
        "path": (
            "results/overnight_rrp_flow_release_preregistered_clock_2026-07-17.csv.gz"
        ),
        "sha256": "9f09bc88c9661441a33cee724e59524f57c0b021abff0fe81263e1a341b7b7b7",
        "columns": [
            "operation_date",
            "decision_time",
            "entry_time",
            "scheduled_exit_time",
            "side",
            "clock_mode",
            "log_amount",
            "innovation",
            "innovation_rank",
        ],
        "identifier_columns": ["clock_mode"],
        "required_groups": {"primary": 328},
        "expected_rows": 328,
    },
    "flcc": {
        "path": (
            "results/federal_liquidity_component_concordance_"
            "preregistered_clock_2026-07-17.csv.gz"
        ),
        "sha256": "7ebb0450422d9265e46c596e0b6415b6a8816c66f5e0cbb9ccda14ca6cb4c67c",
        "columns": [
            "candidate_id",
            "clock_name",
            "feature_release_date",
            "signal_release_date",
            "signal_time",
            "entry_time",
            "exit_time",
            "side",
            "horizon_releases",
            "lower_rank_numerator",
            "upper_rank_numerator",
            "prior_lookback",
            "net_rank_numerator",
            "asset_rank_numerator",
            "tga_release_rank_numerator",
            "rrp_release_rank_numerator",
            "component_breadth",
            "component_tail_breadth",
        ],
        "identifier_columns": ["candidate_id", "clock_name"],
        "required_groups": {
            "FLCC-H4-Q60|component_concordance_only": 131,
            "FLCC-H4-Q60|direction_flip": 136,
            "FLCC-H4-Q60|net_only": 173,
            "FLCC-H4-Q60|one_release_delay": 136,
            "FLCC-H4-Q60|primary": 136,
            "FLCC-H4-Q60|random_side": 136,
            "FLCC-H4-Q65|component_concordance_only": 110,
            "FLCC-H4-Q65|direction_flip": 122,
            "FLCC-H4-Q65|net_only": 154,
            "FLCC-H4-Q65|one_release_delay": 122,
            "FLCC-H4-Q65|primary": 122,
            "FLCC-H4-Q65|random_side": 122,
            "FLCC-H8-Q60|component_concordance_only": 117,
            "FLCC-H8-Q60|direction_flip": 125,
            "FLCC-H8-Q60|net_only": 167,
            "FLCC-H8-Q60|one_release_delay": 124,
            "FLCC-H8-Q60|primary": 125,
            "FLCC-H8-Q60|random_side": 125,
            "FLCC-H8-Q65|component_concordance_only": 99,
            "FLCC-H8-Q65|direction_flip": 116,
            "FLCC-H8-Q65|net_only": 153,
            "FLCC-H8-Q65|one_release_delay": 115,
            "FLCC-H8-Q65|primary": 116,
            "FLCC-H8-Q65|random_side": 116,
        },
        "expected_rows": 3098,
    },
    "dffb_primary": {
        "path": (
            "results/daily_treasury_fiscal_flow_breadth_primary_clock_2026-07-21.csv.gz"
        ),
        "sha256": "df53e1a27fcbc6ea2c4bc3f462a557a75c76a98db3c362944dad0b4d74382978",
        "columns": [
            "policy_id",
            "clock",
            "window",
            "signal_record_date",
            "execution_record_date",
            "decision_time_utc",
            "entry_time_utc",
            "exit_time_utc",
            "side",
            "deposit_breadth",
            "withdrawal_breadth",
            "issue_breadth",
            "redemption_breadth",
            "deposit_eligible_categories",
            "withdrawal_eligible_categories",
            "issue_eligible_categories",
            "redemption_eligible_categories",
            "cash_impulse",
            "debt_impulse",
            "cash_rank126",
            "debt_rank126",
            "total_net_cash",
            "total_net_cash_rank126",
        ],
        "identifier_columns": ["policy_id", "clock"],
        "required_groups": {"DFFB-601|primary": 112},
        "expected_rows": 112,
    },
    "dffb_controls": {
        "path": (
            "results/daily_treasury_fiscal_flow_breadth_control_clocks_2026-07-21.csv.gz"
        ),
        "sha256": "416fc8663b292fcee069e4aca53b83e99a05b594a96940ab2c557e6e0d05e312",
        "columns": [
            "policy_id",
            "clock",
            "window",
            "signal_record_date",
            "execution_record_date",
            "decision_time_utc",
            "entry_time_utc",
            "exit_time_utc",
            "side",
            "deposit_breadth",
            "withdrawal_breadth",
            "issue_breadth",
            "redemption_breadth",
            "deposit_eligible_categories",
            "withdrawal_eligible_categories",
            "issue_eligible_categories",
            "redemption_eligible_categories",
            "cash_impulse",
            "debt_impulse",
            "cash_rank126",
            "debt_rank126",
            "total_net_cash",
            "total_net_cash_rank126",
        ],
        "identifier_columns": ["policy_id", "clock"],
        "required_groups": {
            "DFFB-601|cash_only": 384,
            "DFFB-601|debt_only": 394,
            "DFFB-601|deterministic_random_side": 112,
            "DFFB-601|direction_flip": 112,
            "DFFB-601|one_report_delay": 112,
            "DFFB-601|total_net_cash": 388,
        },
        "expected_rows": 1502,
    },
    "sfrd": {
        "path": "results/sofr_rate_dislocation_preregistered_clock_2026-07-17.csv.gz",
        "sha256": "391c42dd2b0d5b87ffcd73058dd9fa0c4d18fd2f535597effff5a4c8edea2e69",
        "columns": [
            "event_index",
            "effective_date",
            "sofr_available_at_utc",
            "delta_bp",
            "rank_twice_numerator",
            "rank_twice_denominator",
            "state",
            "side",
            "entry_time",
            "exit_time",
        ],
        "identifier_columns": [],
        "fixed_candidate_id": "SFRD-1|primary",
        "required_groups": {"SFRD-1|primary": 158},
        "expected_rows": 158,
    },
    "bdrc": {
        "path": "results/bank_deposit_secured_repo_concordance_clocks_2026-07-20.csv.gz",
        "sha256": "1ff3a6075e3ceff928e1dd19d05880dbe9dbab0e07d79b853146d7b4c8f6cabc",
        "columns": [
            "clock_name",
            "release_date",
            "decision_time",
            "entry_time",
            "exit_time",
            "side",
            "h8_sign",
            "repo_sign",
            "repo5_bp",
            "sofr_effective_date",
            "sofr_available_at",
        ],
        "identifier_columns": ["clock_name"],
        "required_groups": {
            "deterministic_random_side": 75,
            "direction_flip": 75,
            "discordant_state": 64,
            "h8_only": 212,
            "nsa_h8": 70,
            "primary": 75,
            "sofr_only_h8_schedule": 197,
            "stale_h8_one_release": 74,
            "stale_sofr_one_observation": 63,
        },
        "expected_rows": 905,
    },
}


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


def verify_binding(binding: Mapping[str, Any]) -> None:
    if sha256_file(str(binding["path"])) != binding["sha256"]:
        raise RuntimeError(f"frozen binding hash drift: {binding['path']}")


def build_registration() -> dict[str, Any]:
    core: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": POLICY_ID,
        "as_of_date": AS_OF_DATE,
        "document": {"path": str(DOCUMENT), "sha256": DOCUMENT_SHA256},
        "generator": {"path": str(GENERATOR), "sha256": sha256_file(GENERATOR)},
        "evidence_boundary": {
            "source_value_rows_read_for_schema": 2,
            "prior_orfr_preregistration_and_support_artifacts_read": 2,
            "prospective_comparator_headers_inspected_before_review": 8,
            "bound_comparator_artifacts": 7,
            "comparator_identifier_rows_projected_for_cohort_freeze": 7104,
            "comparator_entry_exit_or_side_fields_materialized": 0,
            "orpb_residuals_computed": 0,
            "orpb_incidence_or_side_counts_computed": 0,
            "comparator_overlap_metrics_computed": 0,
            "btc_market_rows_read": 0,
            "funding_rows_read": 0,
            "future_return_rows_read": 0,
            "return_or_pnl_fields_read": 0,
            "post_2023_source_rows_read": 0,
            "economic_outcomes_opened": False,
        },
        "source": {
            "bindings": SOURCE_BINDINGS,
            "internal_build_manifest_hash": (
                "de6708a85fd7626e19adb48bf89a27cf2e50cbc09f8caddb9a6f67c03ca7140a"
            ),
            "physical_rows": 1498,
            "complete_rows": 1489,
            "quarantined_rows": 9,
            "start_operation_date": "2018-01-02",
            "end_operation_date": "2023-12-29",
            "allowed_columns": [
                "operation_date",
                "result_available_at_utc",
                "total_amount_accepted_usd",
                "participating_counterparties",
                "accepted_counterparties",
                "source_complete",
            ],
            "quarantine_policy": (
                "clock only; blank values; emit no signal; clear the 21-operation "
                "window; never bridge the row"
            ),
        },
        "feature": {
            "lookback_complete_operations": 21,
            "amount": "A=log1p(total_amount_accepted_usd/1e9)",
            "breadth": "B=log1p(accepted_counterparties)",
            "fit": (
                "OLS with intercept B~A on exactly 21 consecutive complete prior "
                "operations; current row excluded"
            ),
            "current_residual": "B[t]-(alpha[t]+beta[t]*A[t])",
            "rank": (
                "strict-prior midrank against the 21 in-sample residuals from the "
                "same prior-only fit"
            ),
            "lower_tail": 0.10,
            "upper_tail": 0.90,
            "direction": {"lower": "LONG", "upper": "SHORT"},
            "zero_variance_or_nonfinite_fit": "abstain without clearing a complete row",
            "parameter_grid": False,
        },
        "execution": {
            "decision": "result_available_at_utc",
            "entry_delay_minutes": 5,
            "exit": "next normal operation result_available_at_utc plus 5 minutes",
            "last_source_row_omitted": True,
            "leverage": 0.5,
            "base_cost_notional_per_side": 0.0006,
            "stress_cost_notional_per_side": 0.001,
            "funding_interval": "[entry, exit)",
            "nonoverlap": True,
        },
        "controls": [
            "amount_only_tail",
            "raw_accepted_breadth_tail",
            "participating_breadth_residual",
            "direction_flip",
            "one_release_delay",
            "deterministic_random_side",
        ],
        "support": {
            "history": ["2018-01-01", "2021-01-01"],
            "train": ["2021-01-01", "2023-01-01"],
            "selection": ["2023-01-01", "2024-01-01"],
            "train_gates": {
                "events_min": 50,
                "events_max": 130,
                "each_year_events_min": 20,
                "each_side_share_min": 0.25,
                "maximum_month_share": 0.20,
            },
            "selection_gates": {
                "events_min": 25,
                "events_max": 80,
                "each_half_events_min": 8,
                "each_side_share_min": 0.20,
                "maximum_month_share": 0.25,
            },
            "integrity_gates": {
                "source_binding_hashes_exact": True,
                "source_schema_exact": True,
                "source_row_and_quarantine_counts_exact": True,
                "quarantined_values_blank": True,
                "quarantine_clears_feature_window": True,
                "prior_only_ols_replay_exact": True,
                "prior_only_rank_replay_exact": True,
                "decision_clock_exact": True,
                "entry_delay_exact": True,
                "next_operation_exit_exact": True,
                "last_source_row_omitted": True,
                "split_containment_exact": True,
                "nonoverlap_exact": True,
            },
            "short_circuit_before_comparator_access_on_failure": True,
            "repair_authorized": False,
        },
        "novelty": {
            "bindings": COMPARATOR_BINDINGS,
            "comparison_start": "2021-01-01T00:00:00Z",
            "comparison_end_exclusive": "2024-01-01T00:00:00Z",
            "same_source_orfr": {
                "exact_entry_jaccard_max": 0.15,
                "one_rrp_operation_bidirectional_containment_max": 0.35,
                "absolute_signed_exposure_correlation_max": 0.35,
                "absolute_residual_amount_innovation_spearman_max": 0.35,
            },
            "other_macro": {
                "exact_entry_jaccard_max": 0.10,
                "six_hour_bidirectional_containment_max": 0.25,
                "absolute_signed_exposure_correlation_max": 0.35,
            },
            "truncate_to_comparator_observed_prefix": False,
            "fail_closed_on_missing_empty_schema_time_overlap_or_outcome_field": True,
            "repair_authorized": False,
        },
        "later_outcome_contract": {
            "authorized": False,
            "authorization_requires": "unchanged support and novelty pass",
            "stage1": ["2021-01-01", "2023-01-01"],
            "stage2": ["2023-01-01", "2024-01-01"],
            "stage2_requires_exact_stage1_pass": True,
            "gates": {
                "base_and_stress_absolute_return_positive": True,
                "each_contained_subperiod_return_positive": True,
                "cagr_to_strict_mdd_min": 3.0,
                "strict_mdd_pct_max": 15.0,
                "weekly_cluster_signflip_p_max": 0.10,
                "mean_gross_underlying_bp_min": 35.0,
                "mechanism_control_ratio_margin_min": 0.25,
            },
            "absolute_return_reported_with_cagr_mdd": True,
            "post_2023_sealed": True,
            "repair_authorized": False,
        },
        "authorization": {
            "current_action": "source support and novelty only",
            "candidate_count": 1,
            "economic_evaluator_authorized": False,
        },
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate_registration(registration: Mapping[str, Any]) -> None:
    if sha256_file(DOCUMENT) != DOCUMENT_SHA256:
        raise RuntimeError("ORPB preregistration document hash drift")
    for binding in SOURCE_BINDINGS.values():
        verify_binding(binding)
    for binding in COMPARATOR_BINDINGS.values():
        verify_binding(binding)
    if registration["policy_id"] != POLICY_ID:
        raise RuntimeError("ORPB policy identity drift")
    if registration["feature"]["parameter_grid"] is not False:
        raise RuntimeError("ORPB opened a parameter grid")
    if registration["novelty"]["bindings"] != COMPARATOR_BINDINGS:
        raise RuntimeError("ORPB novelty cohort drift")
    if (
        sum(int(binding["expected_rows"]) for binding in COMPARATOR_BINDINGS.values())
        != 7104
    ):
        raise RuntimeError("ORPB novelty cohort row-count drift")
    if (
        sum(len(binding["required_groups"]) for binding in COMPARATOR_BINDINGS.values())
        != 45
    ):
        raise RuntimeError("ORPB novelty candidate-count drift")
    if any(
        sum(int(count) for count in binding["required_groups"].values())
        != binding["expected_rows"]
        for binding in COMPARATOR_BINDINGS.values()
    ):
        raise RuntimeError("ORPB novelty per-candidate count drift")
    if not all(registration["support"]["integrity_gates"].values()):
        raise RuntimeError("ORPB disabled a support integrity gate")
    if registration["later_outcome_contract"]["authorized"] is not False:
        raise RuntimeError("ORPB preregistration opened economic outcomes")
    if registration["evidence_boundary"]["economic_outcomes_opened"] is not False:
        raise RuntimeError("ORPB evidence boundary opened outcomes")
    core = {key: value for key, value in registration.items() if key != "manifest_hash"}
    if canonical_hash(core) != registration["manifest_hash"]:
        raise RuntimeError("ORPB preregistration manifest mismatch")


def write_registration(registration: Mapping[str, Any], output: str | Path) -> None:
    target = _path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(registration, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    registration = build_registration()
    validate_registration(registration)
    write_registration(registration, args.output)
    print(json.dumps(registration, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
