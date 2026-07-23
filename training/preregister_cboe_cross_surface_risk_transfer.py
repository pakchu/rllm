"""Freeze CXRT-288 before decoding composite incidence or BTC outcomes."""
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
    "results/cboe_cross_surface_risk_transfer_"
    "preregistration_2026-07-24.json"
)
BOUNDARY_DOCUMENT = (
    "docs/cboe-cross-surface-risk-transfer-boundary-2026-07-24.md"
)
BOUNDARY_DOCUMENT_SHA256 = (
    "1863dc4f6f56592949b2abb8b51cca659dffb0b181e1205a7176e4df79aef12a"
)
MECHANISM_DOCUMENT = (
    "docs/cboe-cross-surface-risk-transfer-"
    "mechanism-decision-2026-07-24.md"
)
MECHANISM_DOCUMENT_SHA256 = (
    "805a94d7c33b5a2e0231e5e848d08f88947cd6f5d3e801e1644541b97671368c"
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

CONTROL_ORDER = (
    "primary",
    "term_only",
    "tail_only",
    "option_only",
    "term_tail_agreement",
    "term_option_agreement",
    "tail_option_agreement",
    "one_common_date_stale",
    "exact_direction_flip",
    "deterministic_random_side",
    "one_day_execution_delay",
)


@dataclass(frozen=True)
class Policy:
    policy_id: str = "CXRT-288"
    rank_lookback_observations: int = 252
    rank_minimum_prior_observations: int = 126
    pressure_center: float = 0.50
    strong_bucket_boundary: float = 0.25
    entry_local_hour: int = 9
    entry_local_minute: int = 35
    signal_buffer_minutes: int = 5
    hold_bars: int = 288
    leverage: float = 0.50
    random_seed: int = 20_260_724


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
        raise RuntimeError(f"CXRT-288 CSV header is not one LF line: {path}")
    return header


def csv_header(path: str | Path) -> list[str]:
    header = csv_header_bytes(path).decode("utf-8")
    return next(csv.reader([header.rstrip("\n")]))


def sha256_csv_header(path: str | Path) -> str:
    return hashlib.sha256(csv_header_bytes(path)).hexdigest()


def comparator_contracts() -> list[dict[str, Any]]:
    return [
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
                "control",
                "observation_date",
                "signal_time",
                "entry_time",
                "exit_time",
                "side",
                "front_slope",
                "broad_slope",
                "front_rank",
                "broad_rank",
                "vix_level_rank",
                "score",
            ],
            "header_sha256": (
                "9628d5d9bb26e18964e87d96e33119d8eba8b11208ed516ce18a336b8e04041c"
            ),
            "group_column": "control",
            "selected_groups": [
                "primary",
                "deterministic_random_side",
                "constant_long",
            ],
            "entry_column": "entry_time",
            "exit_column": "exit_time",
            "side_column": "side",
            "side_encoding": {"LONG": 1, "SHORT": -1},
            "declared_coverage": [
                "2021-01-01T00:00:00Z",
                "2024-01-01T00:00:00Z",
            ],
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
                "control",
                "observation_date",
                "signal_time",
                "entry_time",
                "exit_time",
                "side",
                "skew_level",
                "vvix_relative",
                "vix_level",
                "skew_rank",
                "vvix_relative_rank",
                "vix_level_rank",
                "hidden_pressure",
                "hidden_pressure_rank",
                "score",
            ],
            "header_sha256": (
                "ed0b1417ea6946fc8427f47f95b5b4dbcd6f377fad8da62484a2c95cbc85da92"
            ),
            "group_column": "control",
            "selected_groups": ["primary"],
            "entry_column": "entry_time",
            "exit_column": "exit_time",
            "side_column": "side",
            "side_encoding": {"LONG": 1, "SHORT": -1},
            "declared_coverage": [
                "2021-01-01T00:00:00Z",
                "2024-01-01T00:00:00Z",
            ],
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
                "control",
                "observation_date",
                "signal_time",
                "entry_time",
                "exit_time",
                "side",
                "clock_mode",
                "institutional_gap",
                "vix_call_pressure",
                "index_share",
                "delta_institutional_gap",
                "delta_vix_call_pressure",
                "delta_index_share",
                "institutional_gap_rank",
                "vix_call_pressure_rank",
                "index_share_rank",
                "delta_institutional_gap_rank",
                "delta_vix_call_pressure_rank",
                "delta_index_share_rank",
                "score",
            ],
            "header_sha256": (
                "6a763bf874f4cd5dc0ea16433d30868c3dee92a70e74f3dbcbfe6329a2d6d2ee"
            ),
            "group_column": "control",
            "selected_groups": ["primary"],
            "entry_column": "entry_time",
            "exit_column": "exit_time",
            "side_column": "side",
            "side_encoding": {"LONG": 1, "SHORT": -1},
            "declared_coverage": [
                "2021-01-01T00:00:00Z",
                "2024-01-01T00:00:00Z",
            ],
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
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"CXRT-288 frozen dependency changed: {path}")
    source_headers = (
        (TERM_SOURCE, TERM_HEADER_SHA256, TERM_ALLOWLIST),
        (TAIL_SOURCE, TAIL_HEADER_SHA256, TAIL_ALLOWLIST),
        (OPTION_SOURCE, OPTION_HEADER_SHA256, OPTION_ALLOWLIST),
    )
    for path, expected_header_hash, allowlist in source_headers:
        if sha256_csv_header(path) != expected_header_hash:
            raise RuntimeError(f"CXRT-288 source header changed: {path}")
        if not set(allowlist).issubset(csv_header(path)):
            raise RuntimeError(f"CXRT-288 source allowlist missing: {path}")
    for contract in comparator_contracts():
        if sha256_csv_header(contract["path"]) != contract["header_sha256"]:
            raise RuntimeError(
                f"CXRT-288 comparator header hash changed: {contract['id']}"
            )
        if csv_header(contract["path"]) != contract["header"]:
            raise RuntimeError(
                f"CXRT-288 comparator header changed: {contract['id']}"
            )


def _core_manifest() -> dict[str, Any]:
    policy = Policy()
    return {
        "protocol_version": "cboe_cross_surface_risk_transfer_preregistration_v1",
        "policy": asdict(policy),
        "research_history_boundary": {
            "prior_cboe_source_rows_seen": True,
            "prior_cboe_family_outcomes_seen": True,
            "exact_cxrt_common_state_or_candidate_incidence_seen": False,
            "exact_cxrt_composite_outcomes_seen": False,
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
                    "fail the source before state or incidence construction"
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
                "mean(strict-prior ranks of one-source-observation deltas "
                "for all three option levels)"
            ),
        },
        "vote_contract": {
            "pressure_below_half": {"token": "RELIEF", "vote": 1},
            "pressure_above_half": {"token": "STRESS", "vote": -1},
            "pressure_equal_half": {"token": "NEUTRAL", "vote": 0},
            "eligible": "at least two nonzero votes and vote_sum != 0",
            "side": "LONG iff vote_sum>0; SHORT iff vote_sum<0",
            "fitted_weights": False,
            "tail_threshold": None,
            "btc_gate": None,
            "pressure_buckets": [
                "[0.00,0.25) RELIEF_STRONG",
                "[0.25,0.50) RELIEF_WEAK",
                "0.50 NEUTRAL",
                "(0.50,0.75] STRESS_WEAK",
                "(0.75,1.00] STRESS_STRONG",
            ],
        },
        "execution_contract": {
            "source_date": "D",
            "entry_date": "first later exact common CBOE source date D_next",
            "signal_available": "D_next 09:30 America/New_York",
            "decision_entry": "D_next 09:35 America/New_York",
            "exit": "entry + exactly 288*5m",
            "entry_instrument": "Binance USD-M BTCUSDT",
            "global_nonoverlap_before_split": True,
            "entry_equal_previous_exit": "accepted",
            "missing_common_date": "no synthesized entry or stale carry",
            "live": (
                "predeclared CBOE session calendar plus fail-flat source parity"
            ),
        },
        "source_only_controls": {
            "ordered": list(CONTROL_ORDER),
            "definitions": {
                "primary": "three-surface majority",
                "term_only": "term vote alone when non-neutral",
                "tail_only": "tail vote alone when non-neutral",
                "option_only": "option vote alone when non-neutral",
                "term_tail_agreement": (
                    "emit only when term and tail agree non-neutrally"
                ),
                "term_option_agreement": (
                    "emit only when term and option agree non-neutrally"
                ),
                "tail_option_agreement": (
                    "emit only when tail and option agree non-neutrally"
                ),
                "one_common_date_stale": (
                    "current primary entry clock with all three votes "
                    "replaced by immediately preceding rank-complete common-"
                    "date votes; ineligible when stale majority is undefined"
                ),
                "exact_direction_flip": (
                    "primary timestamps with exact opposite side"
                ),
                "deterministic_random_side": (
                    "primary timestamps with frozen SHA256 side"
                ),
                "one_day_execution_delay": (
                    "primary source state and side; shift entry and exit "
                    "exactly 288 bars, then recompute global overlap and split "
                    "containment"
                ),
            },
            "independent_reservation": [
                "primary",
                "term_only",
                "tail_only",
                "option_only",
                "term_tail_agreement",
                "term_option_agreement",
                "tail_option_agreement",
                "one_day_execution_delay",
            ],
            "same_clock_side_controls": [
                "one_common_date_stale",
                "exact_direction_flip",
                "deterministic_random_side",
            ],
            "random_side": (
                "SHA256('CXRT-288|'+entry_time_utc), "
                "LONG iff first byte<128"
            ),
        },
        "source_support_gate": {
            "train": [
                "2021-01-01T00:00:00Z",
                "2023-01-01T00:00:00Z",
            ],
            "selection": [
                "2023-01-01T00:00:00Z",
                "2024-01-01T00:00:00Z",
            ],
            "train_events_min": 400,
            "each_train_year_events_min": 190,
            "train_active_months_min": 22,
            "train_each_side_share_min": 0.20,
            "train_max_month_share": 0.07,
            "train_max_quarter_share": 0.18,
            "train_max_entry_gap_days": 10.0,
            "train_max_same_side_run": 30,
            "selection_events_min": 190,
            "selection_each_half_events_min": 90,
            "selection_each_quarter_events_min": 40,
            "selection_active_months_min": 11,
            "selection_each_side_share_min": 0.20,
            "selection_max_month_share": 0.11,
            "selection_max_entry_gap_days": 10.0,
            "selection_max_same_side_run": 20,
            "composition": {
                "each_surface_each_vote_share_min": 0.15,
                "each_surface_unique_minority_share_min": 0.08,
                "unanimous_share_range": [0.10, 0.80],
                "single_surface_same_side_reproduction_max": 0.80,
                "stale_same_side_reproduction_max": 0.85,
                "random_same_side_reproduction_max": 0.60,
                "unique_minority": (
                    "one surface vote differs while the other two equal "
                    "the primary nonzero majority; denominator is primary "
                    "non-unanimous dates"
                ),
                "same_side_reproduction": (
                    "same entry and side count divided by primary accepted "
                    "split-contained count; zero denominator fails"
                ),
            },
            "failure_action": "retire CXRT-288 unchanged before outcomes",
        },
        "novelty_contract": {
            "comparators": comparator_contracts(),
            "groups_compared_separately": True,
            "exact_entry_jaccard_max": 0.45,
            "same_entry_same_side_reproduction_max": 0.75,
            "absolute_signed_occupancy_pearson_max": 0.60,
            "one_source_day_tolerant_jaccard": "report_only",
            "failure_conditions": (
                "hash drift, header drift, empty required common-coverage "
                "extraction, or undefined/nonfinite signed-exposure "
                "correlation fails before outcomes"
            ),
            "failure_action": "retire CXRT-288 unchanged before outcomes",
        },
        "economic_rllm_sequence": {
            "source_support_and_novelty_before_market": True,
            "separate_committed_evaluator_required": True,
            "roles": {
                "warmup": "2020 source only",
                "fit": "2021",
                "inner_validation": "2022",
                "sealed_selection": "2023",
                "post_2023": "requires separately audited source extension",
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
                "fixed_side",
                "term_pressure_bucket",
                "tail_pressure_bucket",
                "option_pressure_bucket",
                "surface_votes",
                "vote_relation",
                "minority_surface",
                "surface_vote_transitions",
                "prior_majority_transition",
                "calendar_gap_bucket",
                "source_validity",
                "current_position_state",
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
                "source_support",
                "comparator_novelty",
                "economic_RLLM_evaluator_freeze",
                "fit_2021",
                "inner_validation_2022",
                "sealed_selection_2023",
                "post_2023_source_extension",
                "test_eval_forward",
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
        raise RuntimeError("CXRT-288 manifest core differs from code")
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("CXRT-288 manifest hash mismatch")
    if any(
        payload.get(field) is not False
        for field in (
            "outcomes_opened",
            "source_incidence_opened",
            "source_rows_decoded",
            "comparator_rows_decoded",
        )
    ):
        raise RuntimeError("CXRT-288 evidence boundary opened")


def _canonical_manifest_text() -> str:
    return (
        json.dumps(
            build_manifest(),
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
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
            raise RuntimeError("CXRT-288 existing manifest hash mismatch")
        if actual != expected:
            raise RuntimeError("CXRT-288 noncanonical existing manifest")
        return "verified_existing"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        dir=output.parent,
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
                raise RuntimeError("CXRT-288 manifest race drift")
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
    print(
        json.dumps(
            {
                "status": status,
                "output": args.output,
                "manifest_hash": payload["manifest_hash"],
                "outcomes_opened": False,
                "source_incidence_opened": False,
                "source_rows_decoded": False,
                "comparator_rows_decoded": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
