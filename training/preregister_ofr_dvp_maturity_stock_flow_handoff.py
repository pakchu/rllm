"""Freeze DMSH-168 before features, incidence, comparators, or BTC outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


POLICY_ID = "DMSH-168-SOURCE-REUSE"
PROTOCOL_VERSION = "ofr_dvp_maturity_stock_flow_handoff_preregistration_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path("training/preregister_ofr_dvp_maturity_stock_flow_handoff.py")
MECHANISM_DECISION = Path(
    "docs/ofr-dvp-maturity-stock-flow-handoff-mechanism-decision-2026-07-23.md"
)
MECHANISM_DECISION_SHA256 = (
    "82533ea9981015f57fcff5f5cf777668ad60329adec4acbc88441416b1234b92"
)
COMMON_WINDOW_POLICY = Path("docs/novelty-comparator-common-window-policy-2026-07-23.md")
COMMON_WINDOW_POLICY_SHA256 = (
    "928bce6e04fb34001478b4b4ea84156580b661c88a0f0338065a891c009bd580"
)
DEFAULT_OUTPUT = Path(
    "results/ofr_dvp_maturity_stock_flow_handoff_preregistration_2026-07-23.json"
)

SOURCE_ROOT = Path("data/ofr_repo_preliminary_2019_2023")
OBSERVATIONS = SOURCE_ROOT / "ofr_repo_preliminary_observations_2019_2023.csv.gz"
OBSERVATIONS_SHA256 = (
    "6bb24b9a3d6ed6d266b0c885a9d9888bcccb3045df66d37d83031af66c6dc02a"
)
METADATA = SOURCE_ROOT / "ofr_repo_preliminary_metadata_2019_2023.json.gz"
METADATA_SHA256 = (
    "19a04e82eb5d8ddc6c3cb8dc64694438abd6b1987951470bb317659d9c53ef4f"
)
SOURCE_MANIFEST = SOURCE_ROOT / "build_manifest.json"
SOURCE_MANIFEST_SHA256 = (
    "f937f567e1789ecb39a2b84d6288b2cbab931da4e9f1f4e51addea4b3423b705"
)
SOURCE_CANONICAL_MANIFEST_HASH = (
    "802b83a9478711cd29d5b606d9e12eb1e90890e37f5908d4de64d7dd71f6d449"
)
SOURCE_AUDIT = Path("docs/ofr-repo-preliminary-source-audit-2026-07-23.md")
SOURCE_AUDIT_SHA256 = (
    "88e5ee4852acda41759b4c85731e3f6be170869a7485985b9c96507daa387ccb"
)

REQUIRED_SERIES = (
    "REPO-DVP_OV_OO-P",
    "REPO-DVP_OV_LE30-P",
    "REPO-DVP_OV_G30-P",
    "REPO-DVP_TV_OO-P",
    "REPO-DVP_TV_LE30-P",
    "REPO-DVP_TV_G30-P",
    "REPO-DVP_AR_OO-P",
    "REPO-DVP_AR_LE30-P",
    "REPO-DVP_AR_G30-P",
)
COMPONENTS = ("maturity_flow_gap", "curve_gap")


def _comparator(name: str, path: str, digest: str, parser: str) -> Mapping[str, Any]:
    return {"name": name, "path": Path(path), "sha256": digest, "parser": parser}


COMPARATOR_SPECS: tuple[Mapping[str, Any], ...] = (
    _comparator(
        "overnight_rrp_flow_release_all_controls",
        "results/overnight_rrp_flow_release_clocks_2026-07-17.csv.gz",
        "7242d9870627dfc0cf067ff87d9664a1576dd374cb8985e927b40f15d1e3d480",
        "entry_time/exit_time/side grouped by every control",
    ),
    _comparator(
        "overnight_rrp_participant_breadth_all_controls",
        "results/overnight_rrp_participant_breadth_support_clocks_2026-07-21.csv.gz",
        "ef21323229801f11557e0c2d9d4465f7d58b13569552d656d64fdb7d440622ed",
        "entry_time/exit_time/side grouped by every control",
    ),
    _comparator(
        "federal_liquidity_component_concordance_all_groups",
        "results/federal_liquidity_component_concordance_preregistered_clock_2026-07-17.csv.gz",
        "7ebb0450422d9265e46c596e0b6415b6a8816c66f5e0cbb9ccda14ca6cb4c67c",
        "entry_time/exit_time/side grouped by candidate_id and clock_name",
    ),
    _comparator(
        "daily_treasury_fiscal_flow_breadth_primary",
        "results/daily_treasury_fiscal_flow_breadth_primary_clock_2026-07-21.csv.gz",
        "df53e1a27fcbc6ea2c4bc3f462a557a75c76a98db3c362944dad0b4d74382978",
        "entry_time_utc/exit_time_utc/side grouped by policy_id and clock",
    ),
    _comparator(
        "daily_treasury_fiscal_flow_breadth_controls",
        "results/daily_treasury_fiscal_flow_breadth_control_clocks_2026-07-21.csv.gz",
        "416fc8663b292fcee069e4aca53b83e99a05b594a96940ab2c557e6e0d05e312",
        "entry_time_utc/exit_time_utc/side grouped by policy_id and clock",
    ),
    _comparator(
        "sofr_rate_dislocation_primary",
        "results/sofr_rate_dislocation_preregistered_clock_2026-07-17.csv.gz",
        "391c42dd2b0d5b87ffcd73058dd9fa0c4d18fd2f535597effff5a4c8edea2e69",
        "entry_time/exit_time/side fixed group SFRD-1|primary",
    ),
    _comparator(
        "bank_deposit_secured_repo_concordance_all_clocks",
        "results/bank_deposit_secured_repo_concordance_clocks_2026-07-20.csv.gz",
        "1ff3a6075e3ceff928e1dd19d05880dbe9dbab0e07d79b853146d7b4c8f6cabc",
        "entry_time/exit_time/side grouped by every clock_name",
    ),
    _comparator(
        "fed_h8_deposit_migration_primary",
        "results/fed_h8_deposit_migration_preregistered_clock_2026-07-18.csv.gz",
        "20405f79b86861adcc784c81223baae1c40fdf3c73edda339578471a6a6d1b40",
        "entry_time/exit_time/side where clock_mode is primary",
    ),
    _comparator(
        "soma_lending_collateral_scarcity_primary",
        "results/soma_lending_collateral_scarcity_clocks_2026-07-23.csv.gz",
        "b3fe0dc8c895f9a8974cdf08b5bed9d58ff693b8aca7ed59c224627de930a948",
        "entry_time/exit_time/side where control is primary",
    ),
    _comparator(
        "cross_domain_liquidity_transmission_all_clocks",
        "results/cross_domain_liquidity_transmission_relay_support_clock_2026-07-21.csv.gz",
        "aa2bcafd0f62ebe585f93cbd357d29c37ae526a95a90b8a6c0bd7c068cd6e5a1",
        "entry_time_utc/exit_time_utc/side grouped by every clock",
    ),
    _comparator(
        "live_portfolio_pure_clocks",
        "results/cchr_live_portfolio_pure_clocks_2026-07-21.csv.gz",
        "73d6efbd35b3be64b0fa04fa9c8cb2db25866ef884f19b1ae673949e22a42b08",
        "entry_time/exit_time/side grouped by every candidate_id",
    ),
    _comparator(
        "ofr_repo_venue_fragmentation_consensus_primary",
        "results/ofr_repo_venue_fragmentation_consensus_clocks_2026-07-23.csv.gz",
        "b2d251b147feecc98def94610834c7b11f078f35e580f6e7a950fc55bfe0724e",
        "entry_time/exit_time/side where control is primary",
    ),
    _comparator(
        "ofr_repo_mix_shock_resolution_race_primary",
        "results/ofr_repo_mix_shock_resolution_race_clocks_2026-07-23.csv.gz",
        "bed2f522bd68fd389cb0af3d655980999324c0e6adb4f1dc027c6b6ea2cc5ae6",
        "entry_time/exit_time/side where control is primary",
    ),
    _comparator(
        "ofr_repo_collateral_routing_efficiency_primary",
        "results/ofr_repo_collateral_routing_efficiency_clocks_2026-07-23.csv.gz",
        "cbe4e5f6fc52b66062abbf931e46ea4aa0d1f3c0157ffd365d0638aa573c2826",
        "entry_time/exit_time/side where control is primary",
    ),
)

HISTORY_SPECS: tuple[Mapping[str, Any], ...] = (
    _comparator(
        "rvfc_source_support_values_seen",
        "results/ofr_repo_venue_fragmentation_consensus_support_2026-07-23.json",
        "c5918606c958fc8f966e8bd1884e75a91a6cec44074e2edbe86675fa7f978402",
        "history only",
    ),
    _comparator(
        "rvfc_source_support_rejection",
        "docs/ofr-repo-venue-fragmentation-consensus-support-rejection-2026-07-23.md",
        "df97af1f976a08bb7e6870c775ef345e566a505ba0abca3292ae292bf5e32bc8",
        "history only",
    ),
    _comparator(
        "rmsr_source_support_values_seen",
        "results/ofr_repo_mix_shock_resolution_race_support_2026-07-23.json",
        "d42b97bb85f75eba4cb45ea3487af27a44e8bc659a1ee07d73656d3ec5f23cf9",
        "history only",
    ),
    _comparator(
        "rmsr_source_support_rejection",
        "docs/ofr-repo-mix-shock-resolution-race-support-rejection-2026-07-23.md",
        "e9b73818413f14dfff20729a1f9f321c32a6e86d72005df2ec7468f2c5b8038c",
        "history only",
    ),
    _comparator(
        "rcre_source_support_values_seen",
        "results/ofr_repo_collateral_routing_efficiency_support_2026-07-23.json",
        "cd0ce324dfd5661898cee30603500eaf3e76f33604097392c765d7d1386e6451",
        "history only",
    ),
    _comparator(
        "rcre_source_support_rejection",
        "docs/ofr-repo-collateral-routing-efficiency-support-rejection-2026-07-23.md",
        "a79a94e588f65ef8546ef4ad9f378f87e817161c50bdae8623f08681928b1ef1",
        "history only",
    ),
)

PRIOR_RESEARCH_DISCLOSURE: Mapping[str, Any] = {
    "ofr_repo_candidate_number": 4,
    "source_rows_and_metadata_previously_opened": True,
    "twenty_row_head_sample_opened_during_axis_selection": True,
    "dvp_series_valid_row_counts_opened": True,
    "rvfc_rmsr_rcre_incidence_previously_opened": True,
    "maturity_flow_gap_opened": False,
    "curve_gap_opened": False,
    "dmsh_state_or_incidence_opened": False,
    "dmsh_comparator_overlap_opened": False,
    "dmsh_market_outcomes_opened": False,
    "pristine_source_or_discovery_claim": False,
}

EXPECTED_OUTCOME_BOUNDARY: Mapping[str, Any] = {
    "source_file_bytes_hashed": True,
    "source_manifest_metadata_parsed": True,
    "source_observation_value_rows_read": 0,
    "source_metadata_definition_rows_read": 0,
    "history_file_bytes_hashed": True,
    "history_value_rows_read": 0,
    "comparator_file_bytes_hashed": True,
    "comparator_value_rows_read": 0,
    "candidate_features_computed": 0,
    "candidate_incidence_derived": 0,
    "btc_market_rows_read": 0,
    "funding_rows_read": 0,
    "future_return_rows_read": 0,
    "pnl_cagr_mdd_opened": False,
    "network_calls": 0,
    "subprocess_calls": 0,
}
STATIC_TEST_OUTCOME_BOUNDARY: Mapping[str, Any] = {
    **EXPECTED_OUTCOME_BOUNDARY,
    "source_file_bytes_hashed": False,
    "source_manifest_metadata_parsed": False,
    "history_file_bytes_hashed": False,
    "comparator_file_bytes_hashed": False,
}


@dataclass(frozen=True)
class Config:
    output: str = str(DEFAULT_OUTPUT)


def _repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        raise RuntimeError("DMSH path must be repository-relative")
    resolved = (REPOSITORY_ROOT / candidate).resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError("DMSH path must remain repository-relative") from exc
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


def serialized_payload(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def policy_payload() -> dict[str, Any]:
    return {
        "candidate": POLICY_ID,
        "singleton": True,
        "research_status": "source_reuse_feature_and_incidence_blind",
        "mechanism": {
            "object": "DVP maturity stock-flow precursor followed by rate-curve confirmation",
            "compression_confirmation": "p=+1; SHORT; numeric side=-1",
            "extension_confirmation": "p=-1; LONG; numeric side=+1",
            "absorption_or_timeout_trade": False,
            "rvfc_rmsr_rcre_repair": False,
        },
        "contamination": dict(PRIOR_RESEARCH_DISCLOSURE),
        "source": {
            "required_series": list(REQUIRED_SERIES),
            "preliminary_only": True,
            "final_total_b27_b830_gcf_tri_triv1_forbidden": True,
            "availability": "max(observation_date+8d,2020-09-10T00:00:00Z)",
            "equal_availability": "prebatch ranks; greatest complete date decides",
            "invalid_timestamp": "maximum availability among present required rows",
            "zero_of_nine_rows_is_not_a_source_date": True,
            "invalid_vector_action": "cancel pending; no state or event",
            "post_2023_rows_allowed": False,
        },
        "arithmetic": {
            "representation": "exact rational parsed from source decimal text",
            "binary_float_forbidden": True,
            "positive_denominators": [
                "OV_OO+OV_LE30+OV_G30",
                "TV_OO+TV_LE30+TV_G30",
                "TV_LE30+TV_G30",
            ],
        },
        "features": {
            "stock_overnight_share": "OV_OO/(OV_OO+OV_LE30+OV_G30)",
            "flow_overnight_share": "TV_OO/(TV_OO+TV_LE30+TV_G30)",
            "maturity_flow_gap": "flow_overnight_share-stock_overnight_share",
            "term_rate": "(AR_LE30*TV_LE30+AR_G30*TV_G30)/(TV_LE30+TV_G30)",
            "curve_gap": "term_rate-AR_OO",
        },
        "normalization": {
            "components": list(COMPONENTS),
            "history_complete_dates": 252,
            "strict_prior_and_prebatch_only": True,
            "midrank": "(count(prior<current)+0.5*count(prior==current))/252",
            "unit_transform": "u=2*midrank-1",
            "state_thresholds": [-0.5, 0.5],
        },
        "state_machine": {
            "precursor": "flow transition into p while curve state !=p",
            "confirmation_window_complete_rows": 10,
            "same_row_confirmation": False,
            "contradiction_priority": (
                "flow or curve transition into -p cancels before confirmation"
            ),
            "confirmation": "first later curve transition into p after no contradiction",
            "age_ten_confirmation_allowed": True,
            "neutral_cancels": False,
            "same_row_rearm_after_termination": False,
            "invalid_vector_cancels_at_defined_invalidation_time": True,
        },
        "execution": {
            "entry": "ceil_to_5m(confirmation_available_at)+5m",
            "hold_elapsed_hours": 168,
            "hold_bars_5m": 2016,
            "notional_exposure": 0.5,
            "reservation_interval": "[entry,exit)",
            "global_nonoverlap": True,
            "split_containment": True,
            "suppressed_queueing": False,
            "dynamic_exit_size_regime_or_direction_override": False,
        },
        "windows": {
            "warmup": ["2019-01-01T00:00:00Z", "2021-01-01T00:00:00Z"],
            "train": ["2021-01-01T00:00:00Z", "2023-01-01T00:00:00Z"],
            "selection": ["2023-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "sealed_from": "2024-01-01T00:00:00Z",
        },
        "source_support_gates": {
            "train_total_minimum": 40,
            "each_train_year_minimum": 16,
            "each_train_half_minimum": 7,
            "train_each_side_minimum": 10,
            "selection_total_minimum": 18,
            "each_selection_half_minimum": 6,
            "selection_each_side_minimum": 4,
            "every_quarter_active": True,
            "train_maximum_month_share": 0.20,
            "selection_maximum_month_share": 0.25,
            "maximum_entry_gap_elapsed_days": 120,
            "train_each_precursor_polarity_minimum_share": 0.20,
            "selection_each_precursor_polarity_minimum_share": 0.15,
            "train_confirmation_age_bins_required": ["1-3", "4-6", "7-10"],
            "maximum_single_rate_bucket_dominance": 0.85,
            "post_2023_source_rows_read_required": 0,
            "failure_action": "reject before comparator rows and outcomes",
        },
        "controls": {
            "scheduled_causal": [
                "flow_transition_only",
                "curve_transition_only",
                "same_date_conjunction",
                "reverse_order_handoff",
                "five_date_window",
                "twenty_date_window",
                "one_complete_date_stale",
                "five_complete_date_stale",
            ],
            "noncausal_source_placebo_only": [
                "year_curve_permutation_placebo",
                "year_flow_permutation_placebo",
            ],
            "placebo_economic_evaluation_forbidden": True,
            "economic_side": [
                "exact_direction_flip",
                "deterministic_random_side",
                "constant_long",
                "constant_short",
            ],
            "random_side": "SHA256 UTF-8 canonical UTC second; byte<128 LONG else SHORT",
        },
        "common_window_policy": {
            "path": str(COMMON_WINDOW_POLICY),
            "sha256": COMMON_WINDOW_POLICY_SHA256,
            "comparison": ["2021-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "fully_contained_only": True,
        },
        "novelty": {
            "comparators": [spec["name"] for spec in COMPARATOR_SPECS],
            "minimum_comparator_entries": 10,
            "maximum_exact_entry_jaccard": 0.10,
            "one_to_one_tolerance_elapsed_hours": 24,
            "maximum_containment": 0.35,
            "maximum_absolute_signed_exposure_correlation": 0.35,
        },
        "economics": {
            "full_calendar": True,
            "numeric_side": {"LONG": 1, "SHORT": -1},
            "base_cost_notional_per_side": 0.0006,
            "stress_cost_notional_per_side": 0.0010,
            "boundary_funding_debit_kept_credit_discarded": True,
            "strict_global_hwm_favorable_then_adverse": True,
            "base_cagr_to_strict_mdd_minimum": 3.0,
            "stress_cagr_to_strict_mdd_minimum": 2.5,
            "strict_mdd_pct_maximum": 15.0,
            "mean_gross_underlying_bp_minimum": 30.0,
            "subperiod_and_each_side_contribution_positive": True,
            "one_bar_delay_positive": True,
            "weekly_signflip_p_maximum": 0.10,
            "weekly_signflip_draws": 100000,
            "weekly_signflip_seed": 20260723,
            "scheduled_control_cagr_mdd_margin": 0.25,
            "direct_control_mean_gross_margin_bp": 5.0,
        },
        "sequence": [
            "source support",
            "novelty",
            "mechanical evaluator freeze",
            "train 2021-2022",
            "selection 2023",
            "immutable source extension and test 2024",
            "eval 2025",
            "recent 2026",
        ],
        "mutable_parameters": [],
        "stopping_rule": "any failed stage retires DMSH-168-SOURCE-REUSE unchanged",
    }


def _source_binding() -> dict[str, Any]:
    expected = {
        OBSERVATIONS: OBSERVATIONS_SHA256,
        METADATA: METADATA_SHA256,
        SOURCE_MANIFEST: SOURCE_MANIFEST_SHA256,
        SOURCE_AUDIT: SOURCE_AUDIT_SHA256,
    }
    for path, digest in expected.items():
        if sha256_file(path) != digest:
            raise RuntimeError(f"DMSH source hash mismatch: {path}")
    manifest = json.loads(
        _repository_path(SOURCE_MANIFEST).read_text(encoding="utf-8")
    )
    if manifest.get("manifest_hash") != SOURCE_CANONICAL_MANIFEST_HASH:
        raise RuntimeError("DMSH source canonical manifest mismatch")
    if manifest.get("observations", {}).get("sha256") != OBSERVATIONS_SHA256:
        raise RuntimeError("DMSH observation manifest binding mismatch")
    if manifest.get("metadata", {}).get("sha256") != METADATA_SHA256:
        raise RuntimeError("DMSH metadata manifest binding mismatch")
    checks = manifest.get("source_checks")
    if (
        not isinstance(checks, Mapping)
        or not checks
        or not all(v is True for v in checks.values())
    ):
        raise RuntimeError("DMSH source manifest contains a failed check")
    boundary = manifest.get("research_boundary", {})
    expected_boundary = {
        "btc_market_rows_read": 0,
        "funding_rows_read": 0,
        "return_rows_read": 0,
        "pnl_cagr_mdd_opened": False,
        "candidate_incidence_opened": False,
        "candidate_features_computed": [],
        "final_source_rows_read": 0,
    }
    for field, expected_value in expected_boundary.items():
        if boundary.get(field) != expected_value:
            raise RuntimeError(f"DMSH source manifest boundary opened: {field}")
    return {
        "observations": str(OBSERVATIONS),
        "observations_sha256": OBSERVATIONS_SHA256,
        "observation_value_rows_read_during_preregistration": 0,
        "metadata": str(METADATA),
        "metadata_sha256": METADATA_SHA256,
        "metadata_definition_rows_read_during_preregistration": 0,
        "manifest": str(SOURCE_MANIFEST),
        "manifest_sha256": SOURCE_MANIFEST_SHA256,
        "manifest_canonical_hash": SOURCE_CANONICAL_MANIFEST_HASH,
        "manifest_metadata_parsed": True,
        "manifest_observation_rows": manifest["observations"]["rows"],
        "manifest_series": manifest["metadata"]["series"],
        "source_audit": str(SOURCE_AUDIT),
        "source_audit_sha256": SOURCE_AUDIT_SHA256,
    }


def _static_source_binding() -> dict[str, Any]:
    return {
        "observations": str(OBSERVATIONS),
        "observations_sha256": OBSERVATIONS_SHA256,
        "observation_value_rows_read_during_preregistration": 0,
        "metadata": str(METADATA),
        "metadata_sha256": METADATA_SHA256,
        "metadata_definition_rows_read_during_preregistration": 0,
        "manifest": str(SOURCE_MANIFEST),
        "manifest_sha256": SOURCE_MANIFEST_SHA256,
        "manifest_canonical_hash": SOURCE_CANONICAL_MANIFEST_HASH,
        "manifest_metadata_parsed": False,
        "manifest_observation_rows": 77_369,
        "manifest_series": 82,
        "source_audit": str(SOURCE_AUDIT),
        "source_audit_sha256": SOURCE_AUDIT_SHA256,
    }


def _bindings(
    specs: Sequence[Mapping[str, Any]], *, history: bool, verify: bool
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for spec in specs:
        observed = sha256_file(spec["path"]) if verify else spec["sha256"]
        if observed != spec["sha256"]:
            raise RuntimeError(f"DMSH binding hash mismatch: {spec['name']}")
        row: dict[str, Any] = {
            "name": spec["name"],
            "path": str(spec["path"]),
            "sha256": observed,
            "read_mode": (
                "raw bytes for SHA-256 only"
                if verify
                else "static fixture; no file read"
            ),
        }
        if history:
            row.update(
                {
                    "historical_values_previously_opened": True,
                    "value_rows_read_during_preregistration": 0,
                }
            )
        else:
            row.update(
                {
                    "parser": spec["parser"],
                    "comparison": [
                        "2021-01-01T00:00:00Z",
                        "2024-01-01T00:00:00Z",
                    ],
                    "common_window_policy_sha256": COMMON_WINDOW_POLICY_SHA256,
                    "value_rows_read_during_preregistration": 0,
                }
            )
        output.append(row)
    return output


def build_preregistration(*, verify_sources: bool = True) -> dict[str, Any]:
    if verify_sources:
        if sha256_file(MECHANISM_DECISION) != MECHANISM_DECISION_SHA256:
            raise RuntimeError("DMSH mechanism decision hash mismatch")
        if sha256_file(COMMON_WINDOW_POLICY) != COMMON_WINDOW_POLICY_SHA256:
            raise RuntimeError("DMSH common-window policy hash mismatch")
        source = _source_binding()
    else:
        source = _static_source_binding()
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
        "common_window_policy": {
            "path": str(COMMON_WINDOW_POLICY),
            "sha256": COMMON_WINDOW_POLICY_SHA256,
        },
        "source_binding": source,
        "comparator_bindings": _bindings(
            COMPARATOR_SPECS, history=False, verify=verify_sources
        ),
        "history_bindings": _bindings(
            HISTORY_SPECS, history=True, verify=verify_sources
        ),
        "verification_mode": (
            "verified_hashes" if verify_sources else "static_test_fixture"
        ),
        "artifact_eligible": verify_sources,
        "prior_research_disclosure": dict(PRIOR_RESEARCH_DISCLOSURE),
        "candidate_features_or_incidence_opened": False,
        "comparator_rows_opened_during_preregistration": False,
        "outcomes_opened": False,
        "performance_values_opened": False,
        "outcome_boundary": dict(
            EXPECTED_OUTCOME_BOUNDARY if verify_sources else STATIC_TEST_OUTCOME_BOUNDARY
        ),
        "preregistration_source": {"path": str(SCRIPT_PATH), "sha256": sha256_file(SCRIPT_PATH)},
        "next_action": "build exact source-only DMSH and control clocks",
    }
    payload["manifest_hash"] = canonical_hash(payload)
    return payload


def validate_preregistration(
    payload: Mapping[str, Any], *, verify_sources: bool = True
) -> None:
    if payload.get("candidate") != POLICY_ID:
        raise RuntimeError("DMSH candidate identity drift")
    if payload.get("policy") != policy_payload():
        raise RuntimeError("DMSH frozen policy drift")
    if payload.get("policy_hash") != canonical_hash(policy_payload()):
        raise RuntimeError("DMSH policy hash mismatch")
    if payload.get("prior_research_disclosure") != PRIOR_RESEARCH_DISCLOSURE:
        raise RuntimeError("DMSH prior-research disclosure drift")
    boundary = EXPECTED_OUTCOME_BOUNDARY if verify_sources else STATIC_TEST_OUTCOME_BOUNDARY
    if payload.get("outcome_boundary") != boundary:
        raise RuntimeError("DMSH outcome boundary drift")
    if payload.get("artifact_eligible") is not verify_sources:
        raise RuntimeError("DMSH artifact eligibility drift")
    expected_mode = "verified_hashes" if verify_sources else "static_test_fixture"
    if payload.get("verification_mode") != expected_mode:
        raise RuntimeError("DMSH verification mode drift")
    for field in (
        "candidate_features_or_incidence_opened",
        "comparator_rows_opened_during_preregistration",
        "outcomes_opened",
        "performance_values_opened",
    ):
        if payload.get(field) is not False:
            raise RuntimeError(f"DMSH boundary opened: {field}")
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("DMSH canonical hash mismatch")
    if verify_sources:
        expected = build_preregistration(verify_sources=True)
        expected["config"] = dict(payload.get("config", {}))
        expected_core = {key: value for key, value in expected.items() if key != "manifest_hash"}
        expected["manifest_hash"] = canonical_hash(expected_core)
        if payload != expected:
            raise RuntimeError("DMSH preregistration differs from frozen build")


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        path.resolve().relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError("DMSH output must remain inside repository") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(serialized_payload(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_preregistration(cfg: Config = Config()) -> tuple[dict[str, Any], str]:
    output = _repository_path(cfg.output)
    payload = build_preregistration()
    payload["config"] = asdict(cfg)
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    payload["manifest_hash"] = canonical_hash(core)
    validate_preregistration(payload)
    expected_bytes = serialized_payload(payload)
    if output.exists():
        if output.read_bytes() != expected_bytes:
            raise RuntimeError("existing DMSH preregistration differs; refusing overwrite")
        return payload, "verified_existing"
    try:
        _atomic_write(output, payload)
        return payload, "created"
    except FileExistsError:
        if output.read_bytes() != expected_bytes:
            raise RuntimeError("concurrent DMSH preregistration differs")
        return payload, "verified_existing"


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
                "candidate_features_or_incidence_opened": payload[
                    "candidate_features_or_incidence_opened"
                ],
                "comparator_rows_opened": payload[
                    "comparator_rows_opened_during_preregistration"
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
