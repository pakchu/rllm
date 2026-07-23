"""Freeze RCRE-72 before signed features, comparator rows, or BTC outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from training import preregister_ofr_repo_mix_shock_resolution_race as rmsr


POLICY_ID = "RCRE-72-SOURCE-REUSE"
PROTOCOL_VERSION = "ofr_repo_collateral_routing_efficiency_preregistration_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path("training/preregister_ofr_repo_collateral_routing_efficiency.py")
MECHANISM_DECISION = Path(
    "docs/ofr-repo-collateral-routing-efficiency-mechanism-decision-2026-07-23.md"
)
MECHANISM_DECISION_SHA256 = (
    "0b772a63093b39407e022cc7687cf8d49b0d476d465c3d0ee8177abf25b90629"
)
COMMON_WINDOW_POLICY = Path(
    "docs/novelty-comparator-common-window-policy-2026-07-23.md"
)
COMMON_WINDOW_POLICY_SHA256 = (
    "928bce6e04fb34001478b4b4ea84156580b661c88a0f0338065a891c009bd580"
)
DEFAULT_OUTPUT = Path(
    "results/ofr_repo_collateral_routing_efficiency_preregistration_2026-07-23.json"
)

OBSERVATIONS = rmsr.OBSERVATIONS
OBSERVATIONS_SHA256 = rmsr.OBSERVATIONS_SHA256
METADATA = rmsr.METADATA
METADATA_SHA256 = rmsr.METADATA_SHA256
SOURCE_MANIFEST = rmsr.SOURCE_MANIFEST
SOURCE_MANIFEST_SHA256 = rmsr.SOURCE_MANIFEST_SHA256
SOURCE_AUDIT = rmsr.SOURCE_AUDIT
SOURCE_AUDIT_SHA256 = rmsr.SOURCE_AUDIT_SHA256
SOURCE_CANONICAL_MANIFEST_HASH = rmsr.SOURCE_CANONICAL_MANIFEST_HASH
REQUIRED_SERIES = rmsr.REQUIRED_SERIES
COMPONENTS = (
    "quantity_gap",
    "rate_gap",
    "routing_pressure",
    "absolute_pressure",
    "absolute_quantity_gap",
    "absolute_rate_gap",
)

COMPARATOR_SPECS: tuple[Mapping[str, Any], ...] = (
    *rmsr.COMPARATOR_SPECS,
    {
        "name": "ofr_repo_mix_shock_resolution_race_primary",
        "path": Path(
            "results/ofr_repo_mix_shock_resolution_race_clocks_2026-07-23.csv.gz"
        ),
        "sha256": (
            "bed2f522bd68fd389cb0af3d655980999324c0e6adb4f1dc027c6b6ea2cc5ae6"
        ),
        "parser": "entry_time/exit_time/side where control is primary",
    },
)

HISTORY_BINDINGS: tuple[Mapping[str, Any], ...] = (
    *rmsr.HISTORY_BINDINGS,
    {
        "name": "rmsr_source_support_values_seen",
        "path": Path(
            "results/ofr_repo_mix_shock_resolution_race_support_2026-07-23.json"
        ),
        "sha256": (
            "d42b97bb85f75eba4cb45ea3487af27a44e8bc659a1ee07d73656d3ec5f23cf9"
        ),
    },
    {
        "name": "rmsr_source_support_rejection_decision",
        "path": Path(
            "docs/ofr-repo-mix-shock-resolution-race-support-rejection-2026-07-23.md"
        ),
        "sha256": (
            "e9b73818413f14dfff20729a1f9f321c32a6e86d72005df2ec7468f2c5b8038c"
        ),
    },
    {
        "name": "rmsr_support_protocol_dependency",
        "path": Path("training/build_ofr_repo_mix_shock_resolution_race_support.py"),
        "sha256": (
            "d00fa29f04c5eb09ffbc7787ccdf959643d579614f3eca8caa52d3ce8c18100d"
        ),
    },
)

PRIOR_RESEARCH_DISCLOSURE: Mapping[str, Any] = {
    "ofr_repo_candidate_number": 3,
    "source_row_values_previously_opened": True,
    "rvfc_and_rmsr_absolute_component_incidence_previously_opened": True,
    "signed_quantity_gap_opened": False,
    "signed_rate_gap_opened": False,
    "routing_pressure_or_rcre_incidence_opened": False,
    "prior_comparator_timing_rows_partially_opened_for_validation": True,
    "rcre_comparator_overlap_opened": False,
    "rcre_market_outcomes_opened": False,
    "pristine_source_or_comparator_claim": False,
}

EXPECTED_OUTCOME_BOUNDARY: Mapping[str, Any] = {
    "source_file_bytes_hashed_during_preregistration": True,
    "source_manifest_metadata_parsed": True,
    "source_observation_value_rows_read_during_preregistration": 0,
    "source_metadata_definition_rows_read_during_preregistration": 0,
    "history_artifact_bytes_hashed": True,
    "history_value_rows_read_during_preregistration": 0,
    "common_window_policy_bytes_hashed": True,
    "signed_features_computed": 0,
    "rcre_states_or_events_derived": 0,
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
    "history_artifact_bytes_hashed": False,
    "common_window_policy_bytes_hashed": False,
    "comparator_file_bytes_hashed_during_preregistration": False,
}


@dataclass(frozen=True)
class Config:
    output: str = str(DEFAULT_OUTPUT)


def _repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        raise RuntimeError("RCRE path must be repository-relative")
    resolved = (REPOSITORY_ROOT / candidate).resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError("RCRE path must remain repository-relative") from exc
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
        "research_status": "source_reuse_signed_feature_incidence_blind",
        "economic_hypothesis": {
            "mechanism": "venue-label-invariant signed price-by-quantity routing",
            "positive_product": "routing friction; SHORT",
            "negative_product": "routing efficiency; LONG",
            "rvfc_threshold_repair": False,
            "rmsr_absorption_only_repair": False,
        },
        "contamination": dict(PRIOR_RESEARCH_DISCLOSURE),
        "source": {
            "required_series": list(REQUIRED_SERIES),
            "preliminary_only": True,
            "disclosure_edit_required": "0",
            "TRI_DVP_venue_totals_and_sparse_tenors_forbidden": True,
            "TRIV1_excluding_fed_required": True,
            "exact_observation_date_join": True,
            "required_row_availability": (
                "max(observation_date+8 elapsed calendar days,"
                "2020-09-10T00:00:00Z)"
            ),
            "equal_availability_batch": (
                "rank every batch row from prebatch history; only greatest date decides"
            ),
            "missing_required_value_action": "date invalid and continuity broken",
            "imputation_or_forward_fill": False,
            "post_2023_rows_allowed": False,
        },
        "materiality": {
            "gcf_ag_plus_t_strictly_positive": True,
            "triv1_ag_plus_t_strictly_positive": True,
            "each_ag_and_t_share_minimum": "1/20",
        },
        "arithmetic": {
            "representation": "exact rational from source decimal text",
            "binary_float_forbidden": True,
            "tie_rule": "exact rational equality",
        },
        "features": {
            "quantity_gap": (
                "GCF_TV_AG/(GCF_TV_AG+GCF_TV_T)-"
                "TRIV1_TV_AG/(TRIV1_TV_AG+TRIV1_TV_T)"
            ),
            "rate_gap": (
                "(GCF_AR_AG-GCF_AR_T)-(TRIV1_AR_AG-TRIV1_AR_T)"
            ),
            "routing_pressure": "quantity_gap*rate_gap",
            "venue_swap_identity": (
                "swap negates both gaps and preserves routing_pressure exactly"
            ),
        },
        "normalization": {
            "history_complete_dates": 252,
            "strict_prior_and_prebatch_only": True,
            "midrank": "(count(prior<current)+0.5*count(prior==current))/252",
            "unit_transform": "u=2*midrank-1",
            "expanding_fallback": False,
        },
        "state": {
            "friction": "product>0 and u_pressure>=+0.50; state=+1; SHORT",
            "efficiency": "product<0 and u_pressure<=-0.50; state=-1; LONG",
            "zero_product_neutral": True,
            "trigger": "current nonzero state differs from prior continuous state",
            "persistence_retrades": False,
            "first_ready_after_break_can_trigger": False,
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
            "suppressed_event_queueing": False,
            "entry_and_exit_same_split_required": True,
            "stops_take_profit_trailing_dynamic_or_regime_override": False,
        },
        "windows": {
            "source_warmup": ["2019-01-01T00:00:00Z", "2021-01-01T00:00:00Z"],
            "train": ["2021-01-01T00:00:00Z", "2023-01-01T00:00:00Z"],
            "selection": ["2023-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "sealed_from": "2024-01-01T00:00:00Z",
        },
        "source_support_gates": {
            "train_total_minimum": 45,
            "each_train_year_minimum": 15,
            "each_train_half_minimum": 6,
            "train_each_side_minimum": 10,
            "selection_total_minimum": 20,
            "each_selection_half_minimum": 7,
            "selection_each_side_minimum": 5,
            "every_train_and_selection_quarter_active": True,
            "train_maximum_month_share": 0.20,
            "selection_maximum_month_share": 0.25,
            "maximum_accepted_entry_gap_elapsed_days": 90,
            "train_each_product_sign_minimum_share": 0.20,
            "selection_each_product_sign_minimum_share": 0.15,
            "train_each_quadrant_minimum_share": 0.10,
            "selection_each_quadrant_minimum_share": 0.05,
            "train_maximum_quadrant_share": 0.50,
            "selection_maximum_quadrant_share": 0.60,
            "venue_swap_identity_required_on_every_complete_date": True,
            "nonzero_exact_rational_gaps_and_product_required": True,
            "exact_timing_uniqueness_split_and_nonoverlap_required": True,
            "post_2023_source_rows_read_required": 0,
            "failure_action": "reject before comparator rows and outcomes",
        },
        "source_controls": {
            "quantity_gap_label_pair": "original and venue-swapped direction flips; diagnostic only",
            "rate_gap_label_pair": "original and venue-swapped direction flips; diagnostic only",
            "absolute_pressure": "rank abs(quantity_gap)*abs(rate_gap)",
            "both_legs_extreme": (
                "both absolute-gap unit ranks>=+0.50; side from product sign"
            ),
            "absolute_rank_additive": (
                "mean absolute-gap unit rank>=+0.50; side from product sign"
            ),
            "sign_without_magnitude": "product sign transition without rank",
            "one_complete_date_stale": "one prior decision state at current time",
            "five_complete_date_stale": "five prior decision states at current time",
            "year_rate_gap_permutation": (
                "SHA256('RCRE-72|year_rate_gap_permutation|<year>|<observation_date>')"
            ),
            "year_product_permutation": (
                "SHA256('RCRE-72|year_product_permutation|<year>|<observation_date>')"
            ),
            "label_pair_controls_can_falsify_economics": False,
        },
        "economic_controls": {
            "exact_direction_flip": "side=-primary_side",
            "deterministic_random_side": (
                "LONG iff first byte SHA256('RCRE-72|deterministic_random_side|'"
                "+entry_time_utc_iso)<128"
            ),
            "constant_long": "all primary entries LONG",
            "constant_short": "all primary entries SHORT",
        },
        "common_window_policy": {
            "path": str(COMMON_WINDOW_POLICY),
            "sha256": COMMON_WINDOW_POLICY_SHA256,
            "comparison_window": ["2021-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "raw_artifact_validation_before_window_filter": True,
            "fully_contained_intervals_only": True,
            "boundary_crossing_intervals_clipped": False,
            "out_of_window_counts_reported": True,
        },
        "novelty": {
            "comparators": [spec["name"] for spec in COMPARATOR_SPECS],
            "one_to_one_tolerance_elapsed_hours": 24,
            "maximum_exact_entry_jaccard": 0.10,
            "maximum_rcre_one_day_containment": 0.35,
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
            "minimum_trades": {"train": 45, "selection": 20},
            "minimum_trades_each_side": {"train": 10, "selection": 5},
            "calendar_month_cluster_sign_flip_p_maximum": 0.10,
            "required_subperiods_positive": True,
            "primary_must_beat_label_invariant_controls": True,
        },
        "economic_sequence": [
            "source-only support and controls",
            "frozen comparator novelty",
            "freeze strict evaluator",
            "train 2021-2022",
            "selection 2023 only after exact train pass",
            "immutable post-2023 source extension",
            "test 2024",
            "eval 2025",
            "recent 2026",
        ],
        "rllm_boundary": {
            "authorized_before_deterministic_train_and_selection_pass": False,
            "later_actions": ["TRADE_FIXED_SIDE", "ABSTAIN"],
            "may_create_event_reverse_side_change_size_or_hold": False,
        },
        "mutable_parameters": [],
        "stopping_rule": (
            "any provenance, causality, source-support, novelty, train, or "
            "selection failure retires RCRE-72-SOURCE-REUSE unchanged"
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
            raise RuntimeError(f"RCRE source hash mismatch: {path}")
    manifest = json.loads(_repository_path(SOURCE_MANIFEST).read_text(encoding="utf-8"))
    if manifest.get("manifest_hash") != SOURCE_CANONICAL_MANIFEST_HASH:
        raise RuntimeError("RCRE source canonical manifest mismatch")
    if manifest.get("observations", {}).get("sha256") != OBSERVATIONS_SHA256:
        raise RuntimeError("RCRE observation manifest binding mismatch")
    if manifest.get("metadata", {}).get("sha256") != METADATA_SHA256:
        raise RuntimeError("RCRE metadata manifest binding mismatch")
    if not all(manifest.get("source_checks", {}).values()):
        raise RuntimeError("RCRE source manifest contains a failed check")
    if manifest.get("research_boundary", {}).get("btc_market_rows_read") != 0:
        raise RuntimeError("RCRE source manifest opened BTC rows")
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
            raise RuntimeError(f"RCRE binding hash mismatch: {spec['name']}")
        row: dict[str, Any] = {
            "name": spec["name"],
            "path": str(spec["path"]),
            "sha256": observed,
            "read_mode": (
                "raw bytes for SHA-256 only"
                if verify
                else "declared static fixture binding; no file read or hash"
            ),
        }
        if history:
            row.update(
                {
                    "historical_values_previously_opened": True,
                    "values_read_during_rcre_preregistration": 0,
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
            raise RuntimeError("RCRE mechanism decision hash mismatch")
        if sha256_file(COMMON_WINDOW_POLICY) != COMMON_WINDOW_POLICY_SHA256:
            raise RuntimeError("RCRE common-window policy hash mismatch")
        source = _source_binding()
    else:
        source = _static_source_binding()
    comparators = _bindings(COMPARATOR_SPECS, history=False, verify=verify_sources)
    history = _bindings(HISTORY_BINDINGS, history=True, verify=verify_sources)
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
        "common_window_policy": {
            "path": str(COMMON_WINDOW_POLICY),
            "sha256": COMMON_WINDOW_POLICY_SHA256,
        },
        "source_binding": source,
        "comparator_bindings": comparators,
        "history_bindings": history,
        "verification_mode": "verified_hashes" if verify_sources else "static_test_fixture",
        "artifact_eligible": verify_sources,
        "source_family_values_previously_opened": True,
        "absolute_component_incidence_previously_opened": True,
        "signed_features_or_rcre_incidence_opened": False,
        "comparator_rows_opened_during_preregistration": False,
        "outcomes_opened": False,
        "performance_values_opened": False,
        "prior_research_disclosure": dict(PRIOR_RESEARCH_DISCLOSURE),
        "outcome_boundary": dict(boundary),
        "preregistration_source": {
            "path": str(SCRIPT_PATH),
            "sha256": sha256_file(SCRIPT_PATH),
        },
        "next_action": "build exact source-only RCRE and control clocks",
    }
    payload["manifest_hash"] = canonical_hash(payload)
    return payload


def validate_preregistration(
    payload: Mapping[str, Any], *, verify_sources: bool = True
) -> None:
    if payload.get("candidate") != POLICY_ID:
        raise RuntimeError("RCRE candidate identity drift")
    if payload.get("policy") != policy_payload():
        raise RuntimeError("RCRE frozen policy drift")
    if payload.get("policy_hash") != canonical_hash(policy_payload()):
        raise RuntimeError("RCRE policy hash mismatch")
    if payload.get("prior_research_disclosure") != PRIOR_RESEARCH_DISCLOSURE:
        raise RuntimeError("RCRE prior-research disclosure drift")
    expected_boundary = (
        EXPECTED_OUTCOME_BOUNDARY
        if verify_sources
        else STATIC_TEST_OUTCOME_BOUNDARY
    )
    if payload.get("outcome_boundary") != expected_boundary:
        raise RuntimeError("RCRE outcome boundary drift")
    if payload.get("artifact_eligible") is not verify_sources:
        raise RuntimeError("RCRE artifact eligibility drift")
    expected_mode = "verified_hashes" if verify_sources else "static_test_fixture"
    if payload.get("verification_mode") != expected_mode:
        raise RuntimeError("RCRE verification mode drift")
    if payload.get("common_window_policy") != {
        "path": str(COMMON_WINDOW_POLICY),
        "sha256": COMMON_WINDOW_POLICY_SHA256,
    }:
        raise RuntimeError("RCRE common-window policy binding drift")
    for field in (
        "signed_features_or_rcre_incidence_opened",
        "comparator_rows_opened_during_preregistration",
        "outcomes_opened",
        "performance_values_opened",
    ):
        if payload.get(field) is not False:
            raise RuntimeError(f"RCRE boundary opened: {field}")
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("RCRE canonical hash mismatch")
    if verify_sources:
        expected = build_preregistration(verify_sources=True)
        expected["config"] = dict(payload.get("config", {}))
        expected_core = {
            key: value for key, value in expected.items() if key != "manifest_hash"
        }
        expected["manifest_hash"] = canonical_hash(expected_core)
        if payload != expected:
            raise RuntimeError("RCRE preregistration differs from frozen build")


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        path.resolve().relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError("RCRE output must remain inside repository") from exc
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
            raise RuntimeError("existing RCRE preregistration differs; refusing overwrite")
        return payload, "verified_existing"
    try:
        _atomic_write(output, payload)
        return payload, "created"
    except FileExistsError:
        existing = json.loads(output.read_text(encoding="utf-8"))
        validate_preregistration(existing)
        if existing != payload:
            raise RuntimeError("concurrent RCRE preregistration differs")
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
                "signed_features_or_rcre_incidence_opened": payload[
                    "signed_features_or_rcre_incidence_opened"
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
