"""Freeze RMSR-72 before race incidence, comparator rows, or BTC outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from training import preregister_ofr_repo_venue_fragmentation_consensus as rvfc


POLICY_ID = "RMSR-72-SOURCE-REUSE"
PROTOCOL_VERSION = "ofr_repo_mix_shock_resolution_race_preregistration_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path("training/preregister_ofr_repo_mix_shock_resolution_race.py")
MECHANISM_DECISION = Path(
    "docs/ofr-repo-mix-shock-resolution-race-mechanism-decision-2026-07-23.md"
)
MECHANISM_DECISION_SHA256 = (
    "772151f7f5828b6a27f92b0e1e8eaf22d14f565b6e8e288e6c6903a84f70cad5"
)
DEFAULT_OUTPUT = Path(
    "results/ofr_repo_mix_shock_resolution_race_preregistration_2026-07-23.json"
)

SOURCE_ROOT = Path("data/ofr_repo_preliminary_2019_2023")
OBSERVATIONS = SOURCE_ROOT / "ofr_repo_preliminary_observations_2019_2023.csv.gz"
OBSERVATIONS_SHA256 = rvfc.OBSERVATIONS_SHA256
METADATA = SOURCE_ROOT / "ofr_repo_preliminary_metadata_2019_2023.json.gz"
METADATA_SHA256 = rvfc.METADATA_SHA256
SOURCE_MANIFEST = SOURCE_ROOT / "build_manifest.json"
SOURCE_MANIFEST_SHA256 = rvfc.SOURCE_MANIFEST_SHA256
SOURCE_AUDIT = Path("docs/ofr-repo-preliminary-source-audit-2026-07-23.md")
SOURCE_AUDIT_SHA256 = rvfc.SOURCE_AUDIT_SHA256
SOURCE_CANONICAL_MANIFEST_HASH = rvfc.SOURCE_CANONICAL_MANIFEST_HASH

REQUIRED_SERIES = (
    "REPO-GCF_AR_AG-P",
    "REPO-GCF_AR_T-P",
    "REPO-TRIV1_AR_AG-P",
    "REPO-TRIV1_AR_T-P",
    "REPO-GCF_TV_AG-P",
    "REPO-GCF_TV_T-P",
    "REPO-TRIV1_TV_AG-P",
    "REPO-TRIV1_TV_T-P",
)
COMPONENTS = ("mix_disagreement", "rate_disagreement")

COMPARATOR_SPECS: tuple[Mapping[str, Any], ...] = (
    *rvfc.COMPARATOR_SPECS,
    {
        "name": "ofr_repo_venue_fragmentation_consensus_primary",
        "path": Path(
            "results/ofr_repo_venue_fragmentation_consensus_clocks_2026-07-23.csv.gz"
        ),
        "sha256": (
            "b2d251b147feecc98def94610834c7b11f078f35e580f6e7a950fc55bfe0724e"
        ),
        "parser": "entry_time/exit_time/side where control is primary",
    },
)

HISTORY_BINDINGS: tuple[Mapping[str, Any], ...] = (
    *rvfc.HISTORY_BINDINGS,
    {
        "name": "rvfc_source_support_values_seen",
        "path": Path(
            "results/ofr_repo_venue_fragmentation_consensus_support_2026-07-23.json"
        ),
        "sha256": (
            "c5918606c958fc8f966e8bd1884e75a91a6cec44074e2edbe86675fa7f978402"
        ),
    },
    {
        "name": "rvfc_source_support_rejection_decision",
        "path": Path(
            "docs/ofr-repo-venue-fragmentation-consensus-support-rejection-2026-07-23.md"
        ),
        "sha256": (
            "df97af1f976a08bb7e6870c775ef345e566a505ba0abca3292ae292bf5e32bc8"
        ),
    },
    {
        "name": "rvfc_preregistration_protocol_dependency",
        "path": Path("training/preregister_ofr_repo_venue_fragmentation_consensus.py"),
        "sha256": (
            "e6cddb766e67443f848b6c75ba097ec798d138ced3daa44bb1374a1b7edcd2da"
        ),
    },
)

PRIOR_RESEARCH_DISCLOSURE: Mapping[str, Any] = {
    "ofr_repo_candidate_number": 2,
    "exact_source_family_new_to_repository": False,
    "source_row_values_previously_opened": True,
    "rvfc_component_values_and_incidence_previously_opened": True,
    "exact_rmsr_precursor_terminal_pairing_opened": False,
    "exact_rmsr_race_incidence_or_side_mix_opened": False,
    "rmsr_comparator_overlap_opened": False,
    "rmsr_market_outcomes_opened": False,
    "broader_usd_liquidity_btc_outcomes_previously_opened": True,
    "pristine_source_or_broad_liquidity_claim": False,
}

EXPECTED_OUTCOME_BOUNDARY: Mapping[str, Any] = {
    "source_file_bytes_hashed_during_preregistration": True,
    "source_manifest_metadata_parsed": True,
    "source_observation_value_rows_read_during_preregistration": 0,
    "source_metadata_definition_rows_read_during_preregistration": 0,
    "prior_source_support_artifact_bytes_hashed": True,
    "prior_source_support_value_rows_read_during_preregistration": 0,
    "rmsr_features_computed": 0,
    "rmsr_races_or_events_derived": 0,
    "comparator_file_bytes_hashed_during_preregistration": True,
    "comparator_value_rows_read_during_preregistration": 0,
    "btc_market_rows_read": 0,
    "funding_rows_read": 0,
    "future_return_rows_read": 0,
    "pnl_cagr_mdd_opened": False,
    "network_calls": 0,
    "subprocess_calls": 0,
}

STATIC_TEST_OUTCOME_BOUNDARY: Mapping[str, Any] = {
    **EXPECTED_OUTCOME_BOUNDARY,
    "source_file_bytes_hashed_during_preregistration": False,
    "source_manifest_metadata_parsed": False,
    "prior_source_support_artifact_bytes_hashed": False,
    "comparator_file_bytes_hashed_during_preregistration": False,
}


@dataclass(frozen=True)
class Config:
    output: str = str(DEFAULT_OUTPUT)


def _repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        raise RuntimeError("RMSR path must be repository-relative")
    resolved = (REPOSITORY_ROOT / candidate).resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError("RMSR path must remain repository-relative") from exc
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
        "research_status": "source_reuse_exact_race_incidence_blind",
        "economic_hypothesis": {
            "mechanism": "collateral mix shock first-passage resolution race",
            "price_confirmation": "same-polarity rate extreme arrives first",
            "quantity_absorption": "mix extreme exits before rate confirmation",
            "simultaneous_consensus_claim": False,
            "single_component_claim": False,
        },
        "contamination": dict(PRIOR_RESEARCH_DISCLOSURE),
        "source": {
            "required_series": list(REQUIRED_SERIES),
            "preliminary_only": True,
            "TRI_including_fed_forbidden": True,
            "TRIV1_excluding_fed_required": True,
            "DVP_and_venue_total_rows_forbidden": True,
            "exact_observation_date_join": True,
            "vector_availability": "maximum required-row available_at_utc",
            "required_row_availability": (
                "max(observation_date+8 elapsed calendar days,"
                "2020-09-10T00:00:00Z)"
            ),
            "equal_availability_batch": (
                "only greatest complete observation_date is a decision row"
            ),
            "missing_required_value_action": "date invalid; continuity and race broken",
            "imputation_or_forward_fill": False,
            "post_2023_rows_allowed": False,
        },
        "arithmetic": {
            "representation": "exact rational converted from source decimal text",
            "binary_float_forbidden": True,
            "tie_rule": "exact rational equality",
        },
        "materiality": {
            "gcf_ag_plus_t_strictly_positive": True,
            "triv1_ag_plus_t_strictly_positive": True,
            "each_ag_and_t_share_minimum": "1/20",
            "failure_action": "date invalid; continuity and race broken",
        },
        "components": {
            "mix_disagreement": (
                "abs(GCF_TV_AG/(GCF_TV_AG+GCF_TV_T)-"
                "TRIV1_TV_AG/(TRIV1_TV_AG+TRIV1_TV_T))"
            ),
            "rate_disagreement": (
                "(abs(GCF_AR_AG-GCF_AR_T)+"
                "abs(TRIV1_AR_AG-TRIV1_AR_T))/2"
            ),
        },
        "normalization": {
            "history_complete_dates": 252,
            "strict_prior_only": True,
            "current_date_excluded": True,
            "midrank": "(count(prior<current)+0.5*count(prior==current))/252",
            "unit_transform": "u=2*midrank-1",
            "state": "+1 if u>=+0.50; -1 if u<=-0.50; else 0",
            "expanding_fallback": False,
        },
        "race": {
            "precursor": "mix state transitions into p in {-1,+1}",
            "arm_condition": "rate_state != p on precursor date",
            "already_priced_action": "discard precursor permanently",
            "terminal_window_complete_decision_dates": 20,
            "terminal_must_be_strictly_later": True,
            "price_confirmation": (
                "rate_state transitions into p before mix exits; side=-p"
            ),
            "quantity_absorption": (
                "mix_state exits p before rate confirmation; side=+p"
            ),
            "same_date_confirmation_and_exit": (
                "cancel as AMBIGUOUS_SAME_DATE; emit none; do not rearm"
            ),
            "timeout": "after processing offsets 1 through 20; emit none",
            "terminal_date_cannot_rearm": True,
            "missing_or_invalid_date": "cancel race and break continuity",
            "first_ready_after_break_can_arm_or_terminate": False,
            "precursor_consumed_once": True,
        },
        "execution": {
            "signal_time": "terminal vector available_at_utc",
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
            "mix_transition_only": "eligible mix precursor; side=-p",
            "rate_transition_only": "rate transition into p; side=-p",
            "price_confirmation_only": "primary confirmation terminals only",
            "quantity_absorption_only": "primary absorption terminals only",
            "reverse_race": "rate precursor with mix confirmation versus rate exit",
            "five_date_window": "primary race with timeout 5",
            "forty_date_window": "primary race with timeout 40",
            "one_complete_date_stale": "states stale one complete decision date",
            "five_complete_date_stale": "states stale five complete decision dates",
            "year_rate_permutation": (
                "SHA256('RMSR-72|year_rate_permutation|<year>|"
                "<observation_date>')"
            ),
            "same_date_alignment": "same-date same-polarity transitions; side=-p",
        },
        "economic_controls": {
            "exact_direction_flip": "side=-primary_side",
            "deterministic_random_side": (
                "LONG iff first byte SHA256('RMSR-72|deterministic_random_side|'"
                "+entry_time_utc_iso) < 128"
            ),
            "constant_long": "all primary entries LONG",
            "constant_short": "all primary entries SHORT",
        },
        "source_support_gates": {
            "train_total_minimum": 35,
            "each_train_year_minimum": 12,
            "each_train_half_minimum": 5,
            "train_each_side_minimum": 8,
            "selection_total_minimum": 14,
            "each_selection_half_minimum": 4,
            "selection_each_side_minimum": 3,
            "every_train_and_selection_quarter_active": True,
            "train_maximum_month_share": 0.20,
            "selection_maximum_month_share": 0.25,
            "maximum_accepted_entry_gap_elapsed_days": 90,
            "train_each_terminal_type_minimum_share": 0.20,
            "selection_each_terminal_type_minimum_share": 0.15,
            "maximum_non_tie_dominant_rate_spread_venue_share": 0.85,
            "strict_precursor_terminal_age_range": [1, 20],
            "strict_terminal_after_precursor_required": True,
            "strictly_earlier_eligible_precursor_required": True,
            "accepted_ambiguity_count_required": 0,
            "exact_timing_required": True,
            "unique_entry_time_required": True,
            "entry_and_exit_same_split_required": True,
            "global_nonoverlap_required": True,
            "complete_exact_rational_features_required": True,
            "post_2023_source_rows_read_required": 0,
            "failure_action": "reject before comparator rows and outcomes without repair",
        },
        "novelty": {
            "comparators": [spec["name"] for spec in COMPARATOR_SPECS],
            "comparison_window": ["2021-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "one_to_one_tolerance_elapsed_hours": 24,
            "maximum_exact_entry_jaccard": 0.10,
            "maximum_rmsr_one_day_containment": 0.35,
            "maximum_absolute_signed_exposure_correlation": 0.35,
            "minimum_comparator_entries": 10,
            "constituent_component_clocks_are_specificity_not_novelty_controls": True,
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
            "minimum_trades": {"train": 35, "selection": 14},
            "minimum_trades_each_side": {"train": 8, "selection": 3},
            "calendar_month_cluster_sign_flip_p_maximum": 0.10,
            "required_subperiods_positive": True,
            "primary_must_beat_frozen_mechanism_controls": True,
            "confirmation_and_absorption_subclocks_each_positive": True,
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
            "may_create_event_reverse_side_change_size_or_hold": False,
            "causal_text_inputs": [
                "bucketed precursor polarity and age",
                "bucketed terminal type and component midranks",
                "current position and time in position",
                "risk budget",
            ],
            "reward_penalties": ["strict drawdown", "turnover"],
        },
        "mutable_parameters": [],
        "stopping_rule": (
            "any provenance, causality, source-support, novelty, train, or "
            "selection failure retires RMSR-72-SOURCE-REUSE unchanged"
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
            raise RuntimeError(f"RMSR source hash mismatch: {path}")
    manifest = json.loads(_repository_path(SOURCE_MANIFEST).read_text(encoding="utf-8"))
    if manifest.get("manifest_hash") != SOURCE_CANONICAL_MANIFEST_HASH:
        raise RuntimeError("RMSR source canonical manifest mismatch")
    if manifest.get("observations", {}).get("sha256") != OBSERVATIONS_SHA256:
        raise RuntimeError("RMSR observation manifest binding mismatch")
    if manifest.get("metadata", {}).get("sha256") != METADATA_SHA256:
        raise RuntimeError("RMSR metadata manifest binding mismatch")
    if not all(manifest.get("source_checks", {}).values()):
        raise RuntimeError("RMSR source manifest contains a failed check")
    if manifest.get("research_boundary", {}).get("btc_market_rows_read") != 0:
        raise RuntimeError("RMSR source manifest opened BTC rows")
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


def _hash_bindings(
    specs: Sequence[Mapping[str, Any]], *, history: bool
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        observed = sha256_file(spec["path"])
        if observed != spec["sha256"]:
            raise RuntimeError(f"RMSR binding hash mismatch: {spec['name']}")
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
                    "values_read_during_rmsr_preregistration": 0,
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


def _static_bindings(
    specs: Sequence[Mapping[str, Any]], *, history: bool
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        row = {
            "name": spec["name"],
            "path": str(spec["path"]),
            "sha256": spec["sha256"],
            "read_mode": "declared static fixture binding; no file read or hash",
        }
        if history:
            row.update(
                {
                    "historical_values_previously_opened": True,
                    "values_read_during_rmsr_preregistration": 0,
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


def build_preregistration(*, verify_sources: bool = True) -> dict[str, Any]:
    if verify_sources:
        if sha256_file(MECHANISM_DECISION) != MECHANISM_DECISION_SHA256:
            raise RuntimeError("RMSR mechanism decision hash mismatch")
        source = _source_binding()
        comparators = _hash_bindings(COMPARATOR_SPECS, history=False)
        history = _hash_bindings(HISTORY_BINDINGS, history=True)
    else:
        source = _static_source_binding()
        comparators = _static_bindings(COMPARATOR_SPECS, history=False)
        history = _static_bindings(HISTORY_BINDINGS, history=True)
    policy = policy_payload()
    boundary = (
        EXPECTED_OUTCOME_BOUNDARY
        if verify_sources
        else STATIC_TEST_OUTCOME_BOUNDARY
    )
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
        "source_component_incidence_previously_opened": True,
        "verification_mode": "verified_hashes" if verify_sources else "static_test_fixture",
        "artifact_eligible": verify_sources,
        "exact_race_incidence_opened": False,
        "comparator_rows_opened": False,
        "outcomes_opened": False,
        "performance_values_opened": False,
        "prior_research_disclosure": dict(PRIOR_RESEARCH_DISCLOSURE),
        "outcome_boundary": dict(boundary),
        "preregistration_source": {
            "path": str(SCRIPT_PATH),
            "sha256": sha256_file(SCRIPT_PATH),
        },
        "next_action": "build exact source-only RMSR and control clocks",
    }
    payload["manifest_hash"] = canonical_hash(payload)
    return payload


def validate_preregistration(
    payload: Mapping[str, Any], *, verify_sources: bool = True
) -> None:
    if payload.get("candidate") != POLICY_ID:
        raise RuntimeError("RMSR candidate identity drift")
    if payload.get("policy") != policy_payload():
        raise RuntimeError("RMSR frozen policy drift")
    if payload.get("policy_hash") != canonical_hash(policy_payload()):
        raise RuntimeError("RMSR policy hash mismatch")
    if payload.get("prior_research_disclosure") != PRIOR_RESEARCH_DISCLOSURE:
        raise RuntimeError("RMSR prior-research disclosure drift")
    expected_boundary = (
        EXPECTED_OUTCOME_BOUNDARY
        if verify_sources
        else STATIC_TEST_OUTCOME_BOUNDARY
    )
    if payload.get("outcome_boundary") != expected_boundary:
        raise RuntimeError("RMSR outcome boundary drift")
    expected_mode = "verified_hashes" if verify_sources else "static_test_fixture"
    if payload.get("verification_mode") != expected_mode:
        raise RuntimeError("RMSR verification mode drift")
    if payload.get("artifact_eligible") is not verify_sources:
        raise RuntimeError("RMSR artifact eligibility drift")
    for field in (
        "exact_race_incidence_opened",
        "comparator_rows_opened",
        "outcomes_opened",
        "performance_values_opened",
    ):
        if payload.get(field) is not False:
            raise RuntimeError(f"RMSR boundary opened: {field}")
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("RMSR canonical hash mismatch")
    if verify_sources:
        expected = build_preregistration(verify_sources=True)
        expected["config"] = dict(payload.get("config", {}))
        expected_core = {
            key: value for key, value in expected.items() if key != "manifest_hash"
        }
        expected["manifest_hash"] = canonical_hash(expected_core)
        if payload != expected:
            raise RuntimeError("RMSR preregistration differs from frozen build")


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        path.resolve().relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError("RMSR output must remain inside repository") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
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
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        validate_preregistration(existing)
        if existing != payload:
            raise RuntimeError("existing RMSR preregistration differs; refusing overwrite")
        return payload, "verified_existing"
    try:
        _atomic_write(output, payload)
        return payload, "created"
    except FileExistsError:
        existing = json.loads(output.read_text(encoding="utf-8"))
        validate_preregistration(existing)
        if existing != payload:
            raise RuntimeError(
                "concurrent RMSR preregistration differs; refusing overwrite"
            )
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
                "exact_race_incidence_opened": payload[
                    "exact_race_incidence_opened"
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
