"""Freeze OPRR-288 before decoding candidate incidence or market outcomes."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = (
    "results/cboe_option_pressure_rank_rotation_"
    "preregistration_2026-07-24.json"
)
BOUNDARY_DOCUMENT = (
    "docs/cboe-option-pressure-rank-rotation-boundary-2026-07-24.md"
)
BOUNDARY_DOCUMENT_SHA256 = (
    "48c22ce21d94c8c099df56bf0f057f4892b034346c6301b9e612eff6fc31dfde"
)
MECHANISM_DOCUMENT = (
    "docs/cboe-option-pressure-rank-rotation-"
    "mechanism-decision-2026-07-24.md"
)
MECHANISM_DOCUMENT_SHA256 = (
    "dc60763d147fc52f708bf62f7b5109429dfe1d2d596c87714612395673538203"
)

TERM_SOURCE = (
    "data/cboe_volatility_term_structure_2018_2023/"
    "cboe_vix_term_structure_2018-01-01_2023-12-31.csv.gz"
)
TERM_SOURCE_SHA256 = (
    "6f1b2f7f3a5b1e4d5001d673e6ff54374791879c278248ce27b3d610e4f75dc7"
)
TERM_HEADER_SHA256 = (
    "b2fc60cae8d080d3b47a1a55c48438b63f91530cc345f1b6ef78cee05cc57e20"
)
TERM_MANIFEST = (
    "data/cboe_volatility_term_structure_2018_2023/build_manifest.json"
)
TERM_MANIFEST_SHA256 = (
    "42b2a35ad131bd63574d2adcf684e28766dc3060fa645fc749df10dd3fb27f27"
)
TERM_ALLOWLIST = (
    "observation_date",
    "VIX9D_close",
    "VIX_close",
    "VIX3M_close",
)

TAIL_SOURCE = (
    "data/cboe_tail_risk_2018_2023/"
    "cboe_tail_risk_2018-01-01_2023-12-31.csv.gz"
)
TAIL_SOURCE_SHA256 = (
    "cdde3f8d4bb1e23d00b192f5f9ef759aefba9087be5fd60653e9c02479dfa41a"
)
TAIL_HEADER_SHA256 = (
    "bdc2e42c1d356ebd815c491af9b20211d1bc8f2781c0917d92bbf04f1f0a5dc3"
)
TAIL_MANIFEST = "data/cboe_tail_risk_2018_2023/build_manifest.json"
TAIL_MANIFEST_SHA256 = (
    "9ef80ef3034c93d97c5b2a8160b2502527287d570d15f9d7166d631d9866c7bd"
)
TAIL_ALLOWLIST = (
    "observation_date",
    "SKEW_close",
    "VVIX_close",
    "VIX_close",
)

OPTION_SOURCE = (
    "data/cboe_option_flow_2020_2023/"
    "cboe_option_flow_2020-01-01_2023-12-31.csv.gz"
)
OPTION_SOURCE_SHA256 = (
    "35ef106ef01e3abadbcb4a6227187dd1d7cf2722191bd146bac06d08d1684a78"
)
OPTION_HEADER_SHA256 = (
    "a98314aa376428c5d237837121305c5cc4c4892e25ea3db3127d466b451281d7"
)
OPTION_MANIFEST = "data/cboe_option_flow_2020_2023/build_manifest.json"
OPTION_MANIFEST_SHA256 = (
    "0a513b146ad5857d9ab7311e978152c308de64db8ef29c4d463eb07ea503089e"
)
OPTION_ALLOWLIST = (
    "observation_date",
    "total_volume",
    "index_call_volume",
    "index_put_volume",
    "index_volume",
    "equity_call_volume",
    "equity_put_volume",
    "vix_call_volume",
    "vix_put_volume",
)

SESSION_CLOSURES = (
    "2020-01-01", "2020-01-20", "2020-02-17", "2020-04-10",
    "2020-05-25", "2020-07-03", "2020-09-07", "2020-11-26",
    "2020-12-25",
    "2021-01-01", "2021-01-18", "2021-02-15", "2021-04-02",
    "2021-05-31", "2021-07-05", "2021-09-06", "2021-11-25",
    "2021-12-24",
    "2022-01-17", "2022-02-21", "2022-04-15", "2022-05-30",
    "2022-06-20", "2022-07-04", "2022-09-05", "2022-11-24",
    "2022-12-26",
    "2023-01-02", "2023-01-16", "2023-02-20", "2023-04-07",
    "2023-05-29", "2023-06-19", "2023-07-04", "2023-09-04",
    "2023-11-23", "2023-12-25",
    "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29",
    "2024-05-27", "2024-06-19", "2024-07-04", "2024-09-02",
    "2024-11-28", "2024-12-25",
)

CONTROL_ORDER = (
    "primary",
    "rank_rotation_only",
    "option_own_confirmed",
    "non_option_pair_only",
    "term_sponsor_rotation",
    "tail_sponsor_rotation",
    "one_common_date_stale",
    "exact_direction_flip",
    "deterministic_random_side",
    "one_day_execution_delay",
)


@dataclass(frozen=True)
class Policy:
    policy_id: str = "OPRR-288"
    rank_lookback_observations: int = 252
    rank_minimum_prior_observations: int = 126
    entry_local_hour: int = 9
    entry_local_minute: int = 35
    signal_buffer_minutes: int = 5
    hold_bars: int = 288
    bar_minutes: int = 5
    leverage: float = 0.50


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def csv_header_bytes(path: str | Path) -> bytes:
    source = Path(path)
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rb") as handle:
        header = handle.readline()
    if not header.endswith(b"\n") or b"\n" in header[:-1]:
        raise RuntimeError(f"OPRR-288 CSV header is not one LF line: {path}")
    return header


def csv_header(path: str | Path) -> list[str]:
    header = csv_header_bytes(path).decode("utf-8")
    return next(csv.reader([header.rstrip("\n")]))


def sha256_csv_header(path: str | Path) -> str:
    return hashlib.sha256(csv_header_bytes(path)).hexdigest()


def comparator_contracts() -> list[dict[str, Any]]:
    common = {
        "group_column": "control",
        "entry_column": "entry_time",
        "exit_column": "exit_time",
        "side_column": "side",
        "side_encoding": {"LONG": 1, "SHORT": -1},
        "declared_coverage": [
            "2021-01-01T00:00:00Z",
            "2024-01-01T00:00:00Z",
        ],
    }
    return [
        {
            "id": "CXRT-288",
            "path": (
                "data/cboe_cross_surface_risk_transfer_"
                "clocks_2020_2023.csv.gz"
            ),
            "sha256": (
                "b3cc6f3d6a19cb39ef63ec0ba9908c983ce03c56a0c7dd8786e51c2ef1c0885f"
            ),
            "header": [
                "control", "signal_id", "source_date",
                "signal_available_time", "entry_time", "exit_time", "side",
                "term_vote", "tail_vote", "option_vote", "vote_relation",
                "minority_surface", "term_bucket", "tail_bucket",
                "option_bucket", "term_transition", "tail_transition",
                "option_transition", "prior_majority_transition",
                "calendar_gap_bucket",
            ],
            "header_sha256": (
                "d66a8a9e0593867005d8f47f026bd05556a9ff3c2c3a33e4b4dfc914d99c8591"
            ),
            "selected_groups": [
                "primary", "term_only", "tail_only", "option_only",
                "term_tail_agreement", "one_common_date_stale",
                "exact_direction_flip", "deterministic_random_side",
                "one_day_execution_delay",
            ],
            **common,
        },
        {
            "id": "CVTR-1",
            "path": (
                "results/cboe_volatility_term_rotation_"
                "clocks_2026-07-17.csv.gz"
            ),
            "sha256": (
                "47f4ca447daa2b03a0827ad243ed1107eb34a37e5d7bab18ecd3c4331736959d"
            ),
            "header": [
                "control", "observation_date", "signal_time", "entry_time",
                "exit_time", "side", "front_slope", "broad_slope",
                "front_rank", "broad_rank", "vix_level_rank", "score",
            ],
            "header_sha256": (
                "9628d5d9bb26e18964e87d96e33119d8eba8b11208ed516ce18a336b8e04041c"
            ),
            "selected_groups": [
                "primary", "deterministic_random_side", "constant_long",
            ],
            **common,
        },
        {
            "id": "CTHD-1",
            "path": (
                "results/cboe_tail_hedge_disagreement_"
                "clocks_2026-07-18.csv.gz"
            ),
            "sha256": (
                "0e19455e2fb5ab2d36cc996c9adf514adc85c69dd1a325562344a8015464d546"
            ),
            "header": [
                "control", "observation_date", "signal_time", "entry_time",
                "exit_time", "side", "skew_level", "vvix_relative",
                "vix_level", "skew_rank", "vvix_relative_rank",
                "vix_level_rank", "hidden_pressure", "hidden_pressure_rank",
                "score",
            ],
            "header_sha256": (
                "ed0b1417ea6946fc8427f47f95b5b4dbcd6f377fad8da62484a2c95cbc85da92"
            ),
            "selected_groups": ["primary"],
            **common,
        },
        {
            "id": "CIHM-1",
            "path": (
                "results/cboe_institutional_hedge_migration_"
                "clocks_2026-07-18.csv.gz"
            ),
            "sha256": (
                "5e04cffacb1754c3111fcc32b09d72f06b546a4803b40c77d655a9787b015c0b"
            ),
            "header": [
                "control", "observation_date", "signal_time", "entry_time",
                "exit_time", "side", "clock_mode", "institutional_gap",
                "vix_call_pressure", "index_share",
                "delta_institutional_gap", "delta_vix_call_pressure",
                "delta_index_share", "institutional_gap_rank",
                "vix_call_pressure_rank", "index_share_rank",
                "delta_institutional_gap_rank",
                "delta_vix_call_pressure_rank", "delta_index_share_rank",
                "score",
            ],
            "header_sha256": (
                "6a763bf874f4cd5dc0ea16433d30868c3dee92a70e74f3dbcbfe6329a2d6d2ee"
            ),
            "selected_groups": ["primary"],
            **common,
        },
    ]


def frozen_dependencies() -> dict[str, str]:
    dependencies = {
        BOUNDARY_DOCUMENT: BOUNDARY_DOCUMENT_SHA256,
        MECHANISM_DOCUMENT: MECHANISM_DOCUMENT_SHA256,
        TERM_SOURCE: TERM_SOURCE_SHA256,
        TERM_MANIFEST: TERM_MANIFEST_SHA256,
        TAIL_SOURCE: TAIL_SOURCE_SHA256,
        TAIL_MANIFEST: TAIL_MANIFEST_SHA256,
        OPTION_SOURCE: OPTION_SOURCE_SHA256,
        OPTION_MANIFEST: OPTION_MANIFEST_SHA256,
    }
    dependencies.update(
        {item["path"]: item["sha256"] for item in comparator_contracts()}
    )
    return dependencies


def validate_frozen_dependencies() -> None:
    for path, expected in frozen_dependencies().items():
        if sha256_file(path) != expected:
            raise RuntimeError(f"OPRR-288 frozen dependency changed: {path}")
    source_headers = (
        (TERM_SOURCE, TERM_HEADER_SHA256, TERM_ALLOWLIST),
        (TAIL_SOURCE, TAIL_HEADER_SHA256, TAIL_ALLOWLIST),
        (OPTION_SOURCE, OPTION_HEADER_SHA256, OPTION_ALLOWLIST),
    )
    for path, expected_header_hash, allowlist in source_headers:
        if sha256_csv_header(path) != expected_header_hash:
            raise RuntimeError(f"OPRR-288 source header changed: {path}")
        if not set(allowlist).issubset(csv_header(path)):
            raise RuntimeError(f"OPRR-288 source allowlist missing: {path}")
    for contract in comparator_contracts():
        if sha256_csv_header(contract["path"]) != contract["header_sha256"]:
            raise RuntimeError(
                f"OPRR-288 comparator header hash changed: {contract['id']}"
            )
        if csv_header(contract["path"]) != contract["header"]:
            raise RuntimeError(
                f"OPRR-288 comparator header changed: {contract['id']}"
            )


def _core_manifest() -> dict[str, Any]:
    policy = Policy()
    return {
        "protocol_version": "cboe_option_pressure_rank_rotation_preregistration_v1",
        "policy": asdict(policy),
        "research_history_boundary": {
            "prior_cboe_source_rows_seen": True,
            "prior_cboe_family_outcomes_seen": True,
            "oprr_reserved_before_dclb_and_scaf_incidence": True,
            "dclb_or_scaf_incidence_used_to_define_oprr": False,
            "exact_oprr_ordinal_state_or_candidate_incidence_seen": False,
            "exact_oprr_outcomes_seen": False,
            "global_pristine_holdout_claimed": False,
        },
        "frozen_documents": {
            "boundary": {
                "path": BOUNDARY_DOCUMENT,
                "sha256": BOUNDARY_DOCUMENT_SHA256,
            },
            "mechanism": {
                "path": MECHANISM_DOCUMENT,
                "sha256": MECHANISM_DOCUMENT_SHA256,
            },
        },
        "source_contracts": {
            "exact_date_join": "intersection after independent causal features",
            "missing_policy": "no fill, carry, interpolation, or zero replacement",
            "vix_cross_panel_equality": "term VIX_close equals tail VIX_close exactly",
            "row_validation": {
                "dates": "unique and strictly increasing within each panel",
                "numeric_primitives": (
                    "every retained allowlisted numeric value finite and "
                    "strictly positive"
                ),
                "invalid_primitive_action": (
                    "fail source before pressure, state, or incidence construction"
                ),
                "pre_2024_only": True,
            },
            "term": {
                "path": TERM_SOURCE,
                "sha256": TERM_SOURCE_SHA256,
                "manifest": TERM_MANIFEST,
                "manifest_sha256": TERM_MANIFEST_SHA256,
                "header_sha256": TERM_HEADER_SHA256,
                "allowlist": list(TERM_ALLOWLIST),
                "loader": "pandas.read_csv(usecols=allowlist)",
            },
            "tail": {
                "path": TAIL_SOURCE,
                "sha256": TAIL_SOURCE_SHA256,
                "manifest": TAIL_MANIFEST,
                "manifest_sha256": TAIL_MANIFEST_SHA256,
                "header_sha256": TAIL_HEADER_SHA256,
                "allowlist": list(TAIL_ALLOWLIST),
                "loader": "pandas.read_csv(usecols=allowlist)",
            },
            "option": {
                "path": OPTION_SOURCE,
                "sha256": OPTION_SOURCE_SHA256,
                "manifest": OPTION_MANIFEST,
                "manifest_sha256": OPTION_MANIFEST_SHA256,
                "header_sha256": OPTION_HEADER_SHA256,
                "allowlist": list(OPTION_ALLOWLIST),
                "loader": "pandas.read_csv(usecols=allowlist)",
            },
        },
        "rank_contract": {
            "lookback": policy.rank_lookback_observations,
            "minimum": policy.rank_minimum_prior_observations,
            "formula": "(count(prior<x)+0.5*count(prior==x))/len(prior)",
            "current_appended_after_rank": True,
            "source_histories_independent_before_join": True,
            "future_normalization": False,
        },
        "surface_algebra": {
            "term_pressure": (
                "mean(rank(log(VIX9D/VIX)), rank(log(VIX/VIX3M)))"
            ),
            "tail_pressure": (
                "mean(rank(log(SKEW/100)), rank(log(VVIX/VIX)))"
            ),
            "tail_vix_subtraction": False,
            "tail_second_layer_rank": False,
            "option_levels": {
                "institutional_gap": (
                    "log((index_put+0.5)/(index_call+0.5))"
                    "-log((equity_put+0.5)/(equity_call+0.5))"
                ),
                "vix_call_pressure": "log((vix_call+0.5)/(vix_put+0.5))",
                "index_share": "log((index_volume+1)/(total_volume+1))",
            },
            "option_pressure": (
                "mean(strict-prior ranks of immediately previous option-source "
                "level deltas for all three option levels)"
            ),
        },
        "ordinal_state_contract": {
            "requires_pairwise_distinct_pressures": True,
            "tie_action": "state unavailable; never skip to older valid state",
            "option_position": (
                "1{term_pressure<option_pressure}"
                "+1{tail_pressure<option_pressure}"
            ),
            "positions": {"0": "BELOW", "1": "MIDDLE", "2": "ABOVE"},
            "prior_state": "immediately preceding exact common source date",
        },
        "transition_contract": {
            "rotation": "option_position[t]-option_position[t-1]",
            "eligible": [
                "rotation != 0",
                "sign(delta_option_pressure) == sign(rotation)",
                "sign(delta_term_pressure) == sign(rotation)",
                "sign(delta_tail_pressure) == sign(rotation)",
            ],
            "zero_missing_nonfinite_or_disagreeing_delta": "ineligible",
            "side": {"rotation_positive": "SHORT", "rotation_negative": "LONG"},
            "magnitude_changes_execution": False,
            "numeric_threshold": None,
            "btc_or_calendar_regime_gate": None,
        },
        "session_calendar_contract": {
            "coverage": ["2020-01-01", "2025-01-01"],
            "regular_session": "Monday-Friday excluding frozen full-day closures",
            "full_day_closures": list(SESSION_CLOSURES),
            "early_close_is_session": True,
            "timezone": "America/New_York",
            "next_session": (
                "increment one calendar day from source date until weekday and "
                "not in full_day_closures"
            ),
            "future_source_row_membership_used": False,
            "post_2024_extension": (
                "hash-freeze official future session calendar before opening "
                "that year's source or outcomes"
            ),
        },
        "execution_contract": {
            "source_transition_date": "D[t]",
            "entry_date": "first later prospective regular CBOE session S_next",
            "signal_available": "S_next 09:30 America/New_York",
            "decision_entry": "S_next 09:35 America/New_York",
            "exit": "entry + exactly 288*5m",
            "entry_instrument": "Binance USD-M BTCUSDT",
            "exposure_interval": "[entry,exit)",
            "global_nonoverlap_before_split": True,
            "entry_equal_previous_exit": "accepted",
            "overlapping_candidate": "suppressed and never queued",
            "split_containment": "entry>=start and exit<=end",
            "future_session_source_row": "cannot create suppress or reschedule entry",
        },
        "source_only_controls": {
            "ordered": list(CONTROL_ORDER),
            "definitions": {
                "primary": "exact OPRR transition and three-surface confirmation",
                "rank_rotation_only": (
                    "option ordinal rotation with side from rotation; no deltas"
                ),
                "option_own_confirmed": (
                    "option rotation plus agreeing option delta; no term/tail deltas"
                ),
                "non_option_pair_only": (
                    "agreeing nonzero term/tail deltas; no option rotation or delta"
                ),
                "term_sponsor_rotation": (
                    "exact primary algebra with term as ordinal sponsor"
                ),
                "tail_sponsor_rotation": (
                    "exact primary algebra with tail as ordinal sponsor"
                ),
                "one_common_date_stale": (
                    "at D[t] use primary decision from D[t-2],D[t-1] and schedule "
                    "on prospective S_next after D[t]"
                ),
                "exact_direction_flip": "accepted primary timestamps, opposite side",
                "deterministic_random_side": (
                    "accepted primary timestamps with frozen binary SHA256 side"
                ),
                "one_day_execution_delay": (
                    "primary entry/exit shifted exactly 288 five-minute bars; "
                    "fresh reservation and containment"
                ),
            },
            "independent_reservation": [
                "primary", "rank_rotation_only", "option_own_confirmed",
                "non_option_pair_only", "term_sponsor_rotation",
                "tail_sponsor_rotation", "one_common_date_stale",
                "one_day_execution_delay",
            ],
            "same_clock_side_controls": [
                "exact_direction_flip", "deterministic_random_side",
            ],
            "random_side": {
                "canonical_entry_utc": "YYYY-MM-DDTHH:MM:SSZ",
                "fractional_seconds": False,
                "message": "ASCII bytes b'OPRR-288|'+canonical_entry_utc",
                "digest": "hashlib.sha256(message).digest()",
                "rule": "LONG iff digest[0] < 128 else SHORT",
                "hex_text_used": False,
            },
        },
        "source_support_gate": {
            "warmup": "2020 source only",
            "train": ["2021-01-01T00:00:00Z", "2023-01-01T00:00:00Z"],
            "selection": ["2023-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "train_events_min": 100,
            "each_train_year_events_min": 40,
            "train_active_months_min": 20,
            "train_each_side_share_min": 0.20,
            "train_max_month_share": 0.15,
            "train_max_quarter_share": 0.35,
            "train_max_entry_gap_days": 35.0,
            "train_max_same_side_run": 10,
            "selection_events_min": 45,
            "selection_each_half_events_min": 18,
            "selection_each_quarter_events_min": 6,
            "selection_active_months_min": 10,
            "selection_each_side_share_min": 0.20,
            "selection_max_month_share": 0.22,
            "selection_max_entry_gap_days": 45.0,
            "selection_max_same_side_run": 8,
            "undefined_or_nonfinite": "fail",
            "failure_action": "retire OPRR-288 unchanged before outcomes",
        },
        "rotation_composition_gate": {
            "one_step_share_min": 0.10,
            "two_step_share_min": 0.10,
            "each_undirected_transition_family_share_min": 0.05,
            "transition_families": ["0<->1", "1<->2", "0<->2"],
            "each_prior_position_share_min": 0.08,
            "each_current_position_share_min": 0.08,
            "raw_primary_retention_within_option_own_max": 0.75,
            "raw_primary_retention_within_non_option_pair_max": 0.75,
            "raw_retention_formula": (
                "|primary raw transition dates intersect control raw dates|/"
                "|control raw dates| before reservation; scheduled interval must "
                "be split-contained"
            ),
            "sponsor_exact_entry_jaccard_max": 0.65,
            "stale_same_side_reproduction_max": 0.80,
            "random_same_side_reproduction_max": 0.60,
            "exact_jaccard": "|A intersect B|/|A union B|; nonempty sets required",
            "same_side_reproduction_denominator": (
                "accepted split-contained primary count"
            ),
            "evaluate_train_and_selection_separately": True,
            "failure_action": "retire OPRR-288 unchanged before outcomes",
        },
        "novelty_contract": {
            "comparators": comparator_contracts(),
            "groups_compared_separately": True,
            "window": [
                "2021-01-01T00:00:00Z", "2024-01-01T00:00:00Z",
            ],
            "entry_inclusion": "start<=entry<end after UTC normalization",
            "exact_entry_jaccard": "|A intersect B|/|A union B|",
            "exact_entry_jaccard_max": 0.35,
            "same_entry_same_side_reproduction_max": 0.80,
            "same_side_denominator": "accepted OPRR primary count",
            "tolerant_entry_jaccard": {
                "status": "report_only",
                "local_timezone": "America/New_York",
                "eligible_pair": "absolute local calendar-date difference <=1 day",
                "matching": (
                    "order-preserving DP maximum cardinality with skip-left, "
                    "skip-right, and eligible diagonal match"
                ),
                "formula": "m/(len(A)+len(B)-m)",
                "side_used": False,
            },
            "signed_occupancy": {
                "grid": (
                    "all UTC five-minute left endpoints in [2021-01-01,2024-01-01)"
                ),
                "encoding": "+1 LONG, -1 SHORT, 0 flat",
                "position_interval": "[entry,exit)",
                "same_boundary_order": "previous exit then new entry",
                "intersecting_events_clipped": True,
                "overlap_within_group": "fail",
                "pearson": (
                    "sum((x-mean(x))*(y-mean(y)))/sqrt(sum((x-mean(x))^2)*"
                    "sum((y-mean(y))^2))"
                ),
                "zero_variance": "fail",
            },
            "absolute_signed_occupancy_pearson_max": 0.55,
            "duplicate_entry_or_empty_group": "fail",
            "hash_or_header_drift": "fail",
            "undefined_or_nonfinite": "fail",
            "failure_action": "retire OPRR-288 unchanged before outcomes",
        },
        "economic_rllm_sequence": {
            "source_support_composition_novelty_before_market": True,
            "separate_committed_evaluator_required": True,
            "roles": {
                "warmup": "2020 source only",
                "fit": "2021",
                "inner_validation": "2022",
                "sealed_selection": "2023",
                "post_2023": "separately audited source extension required",
            },
            "cheap_nonleaky_baseline_before_llm_compute": True,
            "qualification": {
                "positive_full_calendar_absolute_return": True,
                "cagr_to_strict_mdd_min": 3.0,
                "strict_mdd_max": 0.15,
                "stress_10bp_per_notional_side_positive": True,
                "one_day_delayed_positive": True,
                "clustered_evidence_required": True,
                "long_and_short_sleeves_positive": True,
            },
        },
        "rllm_boundary": {
            "action_space": ["TRADE_FIXED_SIDE", "ABSTAIN"],
            "allowed_tokens": [
                "fixed_side", "prior_option_position", "current_option_position",
                "rotation_magnitude", "rotation_direction",
                "option_own_change_agreement", "term_confirmation",
                "tail_confirmation", "term_tail_order_relation",
                "term_tail_order_changed", "calendar_gap_bucket",
                "source_validity", "current_position_state",
            ],
            "forbidden": [
                "raw_numeric_values_or_ranks",
                "date_year_month_weekday_timestamp_or_row_identity",
                "source_identifier_or_hash",
                "BTC_price_return_funding_future_path_label_or_reward",
                "PnL_CAGR_MDD_or_split_identity",
                "candidate_creation_side_reversal_hold_leverage_or_time_choice",
            ],
            "prompt_reveals_outcome_summary": False,
        },
        "strict_sequence": {
            "stop_at_first_failure": True,
            "no_parameter_repair": True,
            "stages": [
                "source_support", "rotation_composition", "comparator_novelty",
                "economic_RLLM_evaluator_freeze", "fit_2021",
                "inner_validation_2022", "sealed_selection_2023",
                "post_2023_source_extension", "test_eval_forward",
            ],
        },
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "source_rows_decoded": False,
        "comparator_rows_decoded": False,
    }


def build_manifest() -> dict[str, Any]:
    core = _core_manifest()
    return {**core, "manifest_hash": canonical_hash(core)}


def validate_manifest(payload: dict[str, Any]) -> None:
    expected = build_manifest()
    if payload != expected:
        raise RuntimeError("OPRR-288 manifest core differs from code")
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("OPRR-288 manifest hash mismatch")
    if any(
        payload.get(field) is not False
        for field in (
            "outcomes_opened", "source_incidence_opened",
            "source_rows_decoded", "comparator_rows_decoded",
        )
    ):
        raise RuntimeError("OPRR-288 evidence boundary opened")


def _canonical_manifest_text() -> str:
    return (
        json.dumps(
            build_manifest(), sort_keys=True, indent=2,
            ensure_ascii=True, allow_nan=False,
        )
        + "\n"
    )


def write_once(path: str | Path, payload: dict[str, Any]) -> str:
    validate_frozen_dependencies()
    validate_manifest(payload)
    expected = _canonical_manifest_text().encode("utf-8")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        actual = output.read_bytes()
        if hashlib.sha256(actual).digest() != hashlib.sha256(expected).digest():
            raise RuntimeError("OPRR-288 existing manifest hash mismatch")
        if actual != expected:
            raise RuntimeError("OPRR-288 noncanonical existing manifest")
        return "verified_existing"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", dir=output.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(expected)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError:
            if output.read_bytes() != expected:
                raise RuntimeError("OPRR-288 manifest race drift")
            return "verified_existing"
        return "created"
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build_manifest()
    status = write_once(args.output, payload)
    print(json.dumps({
        "status": status,
        "output": args.output,
        "manifest_hash": payload["manifest_hash"],
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "source_rows_decoded": False,
        "comparator_rows_decoded": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
