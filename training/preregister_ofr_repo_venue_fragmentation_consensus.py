"""Freeze RVFC-72 before source incidence, comparator rows, or BTC outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


POLICY_ID = "RVFC-72-NEW-SOURCE"
PROTOCOL_VERSION = "ofr_repo_venue_fragmentation_consensus_preregistration_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path("training/preregister_ofr_repo_venue_fragmentation_consensus.py")
MECHANISM_DECISION = Path(
    "docs/ofr-repo-venue-fragmentation-consensus-mechanism-decision-2026-07-23.md"
)
MECHANISM_DECISION_SHA256 = (
    "e2685823e4258ee8e6aae166e1b703db53b4c0a9cd6678b844c15c7d63353b23"
)
DEFAULT_OUTPUT = Path(
    "results/ofr_repo_venue_fragmentation_consensus_preregistration_2026-07-23.json"
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
SOURCE_AUDIT = Path("docs/ofr-repo-preliminary-source-audit-2026-07-23.md")
SOURCE_AUDIT_SHA256 = (
    "88e5ee4852acda41759b4c85731e3f6be170869a7485985b9c96507daa387ccb"
)
SOURCE_CANONICAL_MANIFEST_HASH = (
    "802b83a9478711cd29d5b606d9e12eb1e90890e37f5908d4de64d7dd71f6d449"
)

REQUIRED_SERIES = (
    "REPO-DVP_AR_TOT-P",
    "REPO-GCF_AR_TOT-P",
    "REPO-TRIV1_AR_TOT-P",
    "REPO-DVP_TV_TOT-P",
    "REPO-GCF_TV_TOT-P",
    "REPO-TRIV1_TV_TOT-P",
    "REPO-GCF_AR_AG-P",
    "REPO-GCF_AR_T-P",
    "REPO-TRIV1_AR_AG-P",
    "REPO-TRIV1_AR_T-P",
    "REPO-GCF_TV_AG-P",
    "REPO-GCF_TV_T-P",
    "REPO-TRIV1_TV_AG-P",
    "REPO-TRIV1_TV_T-P",
)
COMPONENTS = (
    "rate_dispersion",
    "venue_hhi",
    "collateral_rate_disagreement",
    "collateral_mix_disagreement",
)

COMPARATOR_SPECS: tuple[Mapping[str, Any], ...] = (
    {
        "name": "overnight_rrp_flow_release_all_controls",
        "path": Path("results/overnight_rrp_flow_release_clocks_2026-07-17.csv.gz"),
        "sha256": "7242d9870627dfc0cf067ff87d9664a1576dd374cb8985e927b40f15d1e3d480",
        "parser": "entry_time/exit_time/side grouped by every control",
    },
    {
        "name": "overnight_rrp_participant_breadth_all_controls",
        "path": Path(
            "results/overnight_rrp_participant_breadth_support_clocks_2026-07-21.csv.gz"
        ),
        "sha256": "ef21323229801f11557e0c2d9d4465f7d58b13569552d656d64fdb7d440622ed",
        "parser": "entry_time/exit_time/side grouped by every control",
    },
    {
        "name": "federal_liquidity_component_concordance_all_groups",
        "path": Path(
            "results/federal_liquidity_component_concordance_"
            "preregistered_clock_2026-07-17.csv.gz"
        ),
        "sha256": "7ebb0450422d9265e46c596e0b6415b6a8816c66f5e0cbb9ccda14ca6cb4c67c",
        "parser": "entry_time/exit_time/side grouped by candidate_id and clock_name",
    },
    {
        "name": "daily_treasury_fiscal_flow_breadth_primary",
        "path": Path(
            "results/daily_treasury_fiscal_flow_breadth_primary_clock_2026-07-21.csv.gz"
        ),
        "sha256": "df53e1a27fcbc6ea2c4bc3f462a557a75c76a98db3c362944dad0b4d74382978",
        "parser": "entry_time_utc/exit_time_utc/side grouped by policy_id and clock",
    },
    {
        "name": "daily_treasury_fiscal_flow_breadth_controls",
        "path": Path(
            "results/daily_treasury_fiscal_flow_breadth_control_clocks_2026-07-21.csv.gz"
        ),
        "sha256": "416fc8663b292fcee069e4aca53b83e99a05b594a96940ab2c557e6e0d05e312",
        "parser": "entry_time_utc/exit_time_utc/side grouped by policy_id and clock",
    },
    {
        "name": "sofr_rate_dislocation_primary",
        "path": Path("results/sofr_rate_dislocation_preregistered_clock_2026-07-17.csv.gz"),
        "sha256": "391c42dd2b0d5b87ffcd73058dd9fa0c4d18fd2f535597effff5a4c8edea2e69",
        "parser": "entry_time/exit_time/side fixed group SFRD-1|primary",
    },
    {
        "name": "bank_deposit_secured_repo_concordance_all_clocks",
        "path": Path(
            "results/bank_deposit_secured_repo_concordance_clocks_2026-07-20.csv.gz"
        ),
        "sha256": "1ff3a6075e3ceff928e1dd19d05880dbe9dbab0e07d79b853146d7b4c8f6cabc",
        "parser": "entry_time/exit_time/side grouped by every clock_name",
    },
    {
        "name": "fed_h8_deposit_migration_primary",
        "path": Path(
            "results/fed_h8_deposit_migration_preregistered_clock_2026-07-18.csv.gz"
        ),
        "sha256": "20405f79b86861adcc784c81223baae1c40fdf3c73edda339578471a6a6d1b40",
        "parser": "entry_time/exit_time/side where clock_mode is primary",
    },
    {
        "name": "soma_lending_collateral_scarcity_primary",
        "path": Path("results/soma_lending_collateral_scarcity_clocks_2026-07-23.csv.gz"),
        "sha256": "b3fe0dc8c895f9a8974cdf08b5bed9d58ff693b8aca7ed59c224627de930a948",
        "parser": "entry_time/exit_time/side where control is primary",
    },
    {
        "name": "cross_domain_liquidity_transmission_all_clocks",
        "path": Path(
            "results/cross_domain_liquidity_transmission_relay_support_clock_2026-07-21.csv.gz"
        ),
        "sha256": "aa2bcafd0f62ebe585f93cbd357d29c37ae526a95a90b8a6c0bd7c068cd6e5a1",
        "parser": "entry_time_utc/exit_time_utc/side grouped by every clock",
    },
    {
        "name": "live_portfolio_pure_clocks",
        "path": Path("results/cchr_live_portfolio_pure_clocks_2026-07-21.csv.gz"),
        "sha256": "73d6efbd35b3be64b0fa04fa9c8cb2db25866ef884f19b1ae673949e22a42b08",
        "parser": "entry_time/exit_time/side grouped by every candidate_id",
    },
)

HISTORY_BINDINGS: tuple[Mapping[str, Any], ...] = (
    {
        "name": "flcc_stage1_outcome_seen",
        "path": Path(
            "results/federal_liquidity_component_concordance_stage1_2020_2022_2026-07-17.json"
        ),
        "sha256": "10dc911ad06c7e523d612ff34675421388fefb94fa93e157bfac7e93bd1d82a6",
    },
    {
        "name": "overnight_rrp_stage1_outcome_seen",
        "path": Path(
            "results/overnight_rrp_flow_release_stage1_2021_2022_2026-07-17.json"
        ),
        "sha256": "57dcfc8d5cf945250f8e1ee18e95dc341d81c5dad372ead166c64ebc38e4d63d",
    },
    {
        "name": "sofr_stage1_outcome_seen",
        "path": Path(
            "results/sofr_rate_dislocation_stage1_2021_2022_2026-07-17.json"
        ),
        "sha256": "b8a3d4dbbf00102bf1c14a156ede95aea344f820fa15a3735e305e532ee7e88e",
    },
    {
        "name": "h8_stage1_outcome_seen",
        "path": Path(
            "results/fed_h8_deposit_migration_stage1_2020_2022_2026-07-18.json"
        ),
        "sha256": "3f5118077cdafb48ffb59fc6cec8e7643613861f921bfd78403097181c287a7f",
    },
)

PRIOR_RESEARCH_DISCLOSURE: Mapping[str, Any] = {
    "ofr_repo_candidate_number": 1,
    "exact_source_family_new_to_repository": True,
    "source_row_values_opened_during_source_audit": True,
    "source_metadata_coverage_and_missingness_opened": True,
    "cross_series_features_ranks_states_or_incidence_opened": False,
    "rvfc_market_outcomes_opened": False,
    "broader_usd_liquidity_source_families_heavily_seen": True,
    "related_macro_liquidity_btc_outcomes_previously_opened": True,
    "pristine_broad_liquidity_claim": False,
}

EXPECTED_OUTCOME_BOUNDARY: Mapping[str, Any] = {
    "source_file_bytes_hashed_during_preregistration": True,
    "source_manifest_metadata_parsed": True,
    "source_observation_value_rows_read_during_preregistration": 0,
    "source_metadata_definition_rows_read_during_preregistration": 0,
    "rvfc_components_computed": 0,
    "rvfc_states_or_events_derived": 0,
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
        raise RuntimeError("RVFC path must be repository-relative")
    resolved = (REPOSITORY_ROOT / candidate).resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError("RVFC path must remain repository-relative") from exc
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
        "rate_dispersion": (
            "max(DVP_AR_TOT,GCF_AR_TOT,TRIV1_AR_TOT)-"
            "min(DVP_AR_TOT,GCF_AR_TOT,TRIV1_AR_TOT)"
        ),
        "venue_hhi": (
            "sum((venue_TV_TOT / sum(DVP,GCF,TRIV1 TV_TOT))^2 for venue)"
        ),
        "collateral_rate_disagreement": (
            "(abs(GCF_AR_AG-GCF_AR_T)+abs(TRIV1_AR_AG-TRIV1_AR_T))/2"
        ),
        "collateral_mix_disagreement": (
            "abs(GCF_TV_AG/(GCF_TV_AG+GCF_TV_T)-"
            "TRIV1_TV_AG/(TRIV1_TV_AG+TRIV1_TV_T))"
        ),
    }
    return {
        "candidate": POLICY_ID,
        "singleton": True,
        "research_status": "new-source_candidate-incidence-blind",
        "economic_hypothesis": {
            "mechanism": (
                "agreement among four OFR venue/collateral fragmentation observables"
            ),
            "high_state_side": "SHORT",
            "low_state_side": "LONG",
            "transmission": "secured-dollar fragmentation over 72 elapsed hours",
            "single_component_claim": False,
        },
        "source": {
            "required_series": list(REQUIRED_SERIES),
            "preliminary_only": True,
            "TRI_including_fed_forbidden": True,
            "TRIV1_excluding_fed_required": True,
            "exact_observation_date_join": True,
            "vector_availability": "maximum required-row available_at_utc",
            "equal_availability_batch": (
                "only greatest complete observation_date is a decision row"
            ),
            "missing_required_value_action": "date invalid and continuity broken",
            "imputation_or_forward_fill": False,
            "post_2023_rows_allowed": False,
            "sparse_gcf_tenor_buckets_forbidden": ["G30", "LE30", "OO"],
        },
        "arithmetic": {
            "representation": "exact rational converted from source decimal text",
            "binary_float_forbidden": True,
            "tie_rule": "exact rational equality",
        },
        "materiality": {
            "total_venue_volume_sum_strictly_positive": True,
            "gcf_ag_plus_t_strictly_positive": True,
            "triv1_ag_plus_t_strictly_positive": True,
            "each_ag_and_t_share_minimum": "1/20",
            "failure_action": "date invalid and continuity broken",
        },
        "components": components,
        "normalization": {
            "history_complete_dates": 252,
            "strict_prior_only": True,
            "current_date_excluded": True,
            "midrank": "(count(prior<current)+0.5*count(prior==current))/252",
            "unit_transform": "u=2*midrank-1",
            "expanding_fallback": False,
        },
        "state": {
            "positive_votes_minimum": 3,
            "negative_votes_minimum": 3,
            "positive_score_minimum": 0.50,
            "negative_score_maximum": -0.50,
            "score": "equal mean of four u components",
            "zero_component_votes": False,
            "trigger": "current HIGH/LOW differs from immediately prior ready state",
            "missing_date_breaks_continuity": True,
            "first_state_after_break_can_trigger": False,
            "persistence_inside_same_state_trades": False,
        },
        "execution": {
            "signal_time": "vector available_at_utc",
            "entry_time": "ceil_to_5m(signal_time)+5 elapsed minutes",
            "exact_grid_signal_still_waits_one_bar": True,
            "hold_elapsed_hours": 72,
            "hold_bars_5m": 864,
            "notional_exposure": 0.5,
            "global_nonoverlap": True,
            "reservation_interval": "[entry_time,exit_time)",
            "suppressed_candidate_queueing": False,
            "entry_and_exit_same_split_required": True,
            "stops_take_profit_or_trailing_exit": False,
            "dynamic_size_side_price_or_regime_override": False,
        },
        "windows": {
            "source_warmup": ["2019-01-01T00:00:00Z", "2021-01-01T00:00:00Z"],
            "train": ["2021-01-01T00:00:00Z", "2023-01-01T00:00:00Z"],
            "selection": ["2023-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "sealed_from": "2024-01-01T00:00:00Z",
        },
        "source_controls": {
            "component_only": list(COMPONENTS),
            "component_only_state": "u>=+0.50 HIGH; u<=-0.50 LOW",
            "mean_without_consensus": "mean threshold +/-0.50 without votes",
            "same_sign_without_magnitude": "three-of-four strict sign vote",
            "rate_family_only": "components 1 and 3 both beyond same +/-0.50",
            "volume_family_only": "components 2 and 4 both beyond same +/-0.50",
            "leave_one_component": (
                "each remaining three: two same-sign votes and mean beyond +/-0.50"
            ),
            "one_complete_day_stale": "one prior ready vector at current signal time",
            "five_complete_day_stale": "five prior ready vector at current signal time",
            "year_component_permutation": (
                "independent SHA256('RVFC-72|year_component_permutation|year|"
                "component|observation_date') source ordering assigned to chronological"
                " destinations"
            ),
            "all_controls_use_primary_transition_latency_hold_and_nonoverlap": True,
        },
        "diagnostics": {
            "report_only": [
                "dominant_total_rate_venue",
                "dominant_total_volume_venue",
                "dominant_collateral_rate_spread_venue",
                "event_and_year_shares",
            ],
            "may_change_eligibility_or_side": False,
        },
        "economic_controls": {
            "exact_direction_flip": "side=-primary_side",
            "deterministic_random_side": (
                "LONG iff first byte SHA256('RVFC-72|deterministic_random_side|'"
                "+entry_time_utc_iso) < 128"
            ),
            "constant_long": "all primary entries LONG",
            "constant_short": "all primary entries SHORT",
        },
        "source_support_gates": {
            "train_total_minimum": 60,
            "each_train_year_minimum": 20,
            "each_train_half_minimum": 8,
            "train_each_side_minimum": 15,
            "selection_total_minimum": 20,
            "each_selection_half_minimum": 7,
            "selection_each_side_minimum": 5,
            "every_train_and_selection_quarter_active": True,
            "train_maximum_month_share": 0.15,
            "selection_maximum_month_share": 0.20,
            "maximum_accepted_entry_gap_elapsed_days": 45,
            "failure_action": "reject before comparator rows and outcomes without repair",
        },
        "novelty": {
            "comparators": [spec["name"] for spec in COMPARATOR_SPECS],
            "comparison_window": ["2021-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "one_to_one_tolerance_elapsed_hours": 24,
            "maximum_exact_entry_jaccard": 0.10,
            "maximum_rvfc_one_day_containment": 0.35,
            "maximum_absolute_signed_exposure_correlation": 0.35,
            "minimum_comparator_entries": 10,
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
            "stress_return_positive": True,
            "minimum_trades": 20,
            "minimum_trades_each_side": 5,
            "calendar_month_cluster_sign_flip_p_maximum": 0.10,
            "required_subperiods_positive": True,
            "primary_must_beat_family_and_leave_one_out_controls": True,
            "all_source_stale_permutation_and_side_controls_must_fail": True,
        },
        "economic_sequence": [
            "source-only support and controls",
            "frozen comparator novelty",
            "freeze strict evaluator",
            "train 2021-2022",
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
                "bucketed component midranks",
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
            "train, or selection failure retires RVFC-72-NEW-SOURCE unchanged"
        ),
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
            raise RuntimeError(f"RVFC source hash mismatch: {path}")
    manifest = json.loads(_repository_path(SOURCE_MANIFEST).read_text(encoding="utf-8"))
    if manifest.get("manifest_hash") != SOURCE_CANONICAL_MANIFEST_HASH:
        raise RuntimeError("RVFC source canonical manifest mismatch")
    if manifest.get("observations", {}).get("sha256") != OBSERVATIONS_SHA256:
        raise RuntimeError("RVFC observation manifest binding mismatch")
    if manifest.get("metadata", {}).get("sha256") != METADATA_SHA256:
        raise RuntimeError("RVFC metadata manifest binding mismatch")
    if not all(manifest.get("source_checks", {}).values()):
        raise RuntimeError("RVFC source manifest contains a failed check")
    boundary = manifest.get("research_boundary", {})
    if boundary.get("candidate_incidence_opened") is not False:
        raise RuntimeError("RVFC source manifest opened candidate incidence")
    if boundary.get("btc_market_rows_read") != 0:
        raise RuntimeError("RVFC source manifest opened BTC rows")
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


def _hash_bindings(
    specs: Sequence[Mapping[str, Any]], *, history: bool
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        observed = sha256_file(spec["path"])
        if observed != spec["sha256"]:
            raise RuntimeError(f"RVFC binding hash mismatch: {spec['name']}")
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
                    "values_read_during_rvfc_preregistration": 0,
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
                    "value_rows_read_during_preregistration": 0,
                }
            )
        rows.append(row)
    return rows


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
        "manifest_metadata_parsed": True,
        "manifest_observation_rows": 77_369,
        "manifest_series": 82,
        "source_audit": str(SOURCE_AUDIT),
        "source_audit_sha256": SOURCE_AUDIT_SHA256,
    }


def build_preregistration(*, verify_sources: bool = True) -> dict[str, Any]:
    if verify_sources:
        if sha256_file(MECHANISM_DECISION) != MECHANISM_DECISION_SHA256:
            raise RuntimeError("RVFC mechanism decision hash mismatch")
        source = _source_binding()
        comparators = _hash_bindings(COMPARATOR_SPECS, history=False)
        history = _hash_bindings(HISTORY_BINDINGS, history=True)
    else:
        source = _static_source_binding()
        comparators = [
            {
                "name": spec["name"],
                "path": str(spec["path"]),
                "sha256": spec["sha256"],
                "read_mode": "raw bytes for SHA-256 only",
                "parser": spec["parser"],
                "comparison": ["2021-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
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
                "values_read_during_rvfc_preregistration": 0,
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
        "comparator_rows_opened": False,
        "outcomes_opened": False,
        "performance_values_opened": False,
        "prior_research_disclosure": dict(PRIOR_RESEARCH_DISCLOSURE),
        "outcome_boundary": dict(EXPECTED_OUTCOME_BOUNDARY),
        "preregistration_source": {
            "path": str(SCRIPT_PATH),
            "sha256": sha256_file(SCRIPT_PATH),
        },
        "next_action": "build exact source-only RVFC and control clocks",
    }
    payload["manifest_hash"] = canonical_hash(payload)
    return payload


def validate_preregistration(
    payload: Mapping[str, Any], *, verify_sources: bool = True
) -> None:
    if payload.get("candidate") != POLICY_ID:
        raise RuntimeError("RVFC candidate identity drift")
    if payload.get("policy") != policy_payload():
        raise RuntimeError("RVFC frozen policy drift")
    if payload.get("policy_hash") != canonical_hash(policy_payload()):
        raise RuntimeError("RVFC policy hash mismatch")
    if payload.get("prior_research_disclosure") != PRIOR_RESEARCH_DISCLOSURE:
        raise RuntimeError("RVFC prior-research disclosure drift")
    if payload.get("outcome_boundary") != EXPECTED_OUTCOME_BOUNDARY:
        raise RuntimeError("RVFC outcome boundary drift")
    for field in (
        "exact_source_incidence_opened",
        "comparator_rows_opened",
        "outcomes_opened",
        "performance_values_opened",
    ):
        if payload.get(field) is not False:
            raise RuntimeError(f"RVFC boundary opened: {field}")
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("RVFC canonical hash mismatch")
    if verify_sources:
        expected = build_preregistration(verify_sources=True)
        expected["config"] = dict(payload.get("config", {}))
        expected_core = {
            key: value for key, value in expected.items() if key != "manifest_hash"
        }
        expected["manifest_hash"] = canonical_hash(expected_core)
        if payload != expected:
            raise RuntimeError("RVFC preregistration differs from frozen build")


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        path.resolve().relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError("RVFC output must remain inside repository") from exc
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
            raise RuntimeError("existing RVFC preregistration differs; refusing overwrite")
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
                "comparator_rows_opened": payload["comparator_rows_opened"],
                "outcomes_opened": payload["outcomes_opened"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
