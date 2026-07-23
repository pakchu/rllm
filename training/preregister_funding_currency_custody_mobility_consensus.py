"""Freeze FCCM-72 before source values, incidence, comparators, or outcomes."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


POLICY_ID = "FCCM-72"
PROTOCOL_VERSION = "funding_currency_custody_mobility_consensus_prereg_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(
    "training/preregister_funding_currency_custody_mobility_consensus.py"
)
TEST_PATH = Path(
    "tests/test_preregister_funding_currency_custody_mobility_consensus.py"
)
MECHANISM_DECISION = Path(
    "docs/funding-currency-custody-mobility-consensus-"
    "mechanism-decision-2026-07-23.md"
)
MECHANISM_DECISION_SHA256 = (
    "444bc1d8b690da5f2e1ce42bb0cb2b3f97eb872f904273bd9a7aab6ec9b58031"
)
BOUNDARY_LEDGER = Path("results/fccm_mechanism_boundary_2026-07-23.json")
BOUNDARY_LEDGER_SHA256 = (
    "08eced75e484d5e0cc18882fef2672d24928f81995cb75ad0191131034c05184"
)
BOUNDARY_MANIFEST_HASH = (
    "571554f181747fb61dd02612d36eacd16af04f81c90643c7266e12b8a0753dec"
)
COMMON_WINDOW_POLICY = Path(
    "docs/novelty-comparator-common-window-policy-2026-07-23.md"
)
COMMON_WINDOW_POLICY_SHA256 = (
    "928bce6e04fb34001478b4b4ea84156580b661c88a0f0338065a891c009bd580"
)
DEFAULT_OUTPUT = Path(
    "results/funding_currency_custody_mobility_consensus_"
    "preregistration_2026-07-23.json"
)

BITFINEX_SOURCE = Path("data/bitfinex_margin_funding_stats_2020_2023.csv.gz")
BITFINEX_SOURCE_SHA256 = (
    "71635b9f3a38efa7422a6fcf616859e6a41636bbb79ff0f85e160ef395b0d53c"
)
BITFINEX_MANIFEST = Path(
    "results/bitfinex_margin_funding_stats_source_manifest_2026-07-20.json"
)
BITFINEX_MANIFEST_SHA256 = (
    "9d7c13d56983d7d33fec1c17e24f1794baca64fcfc666599b798d5d5b49cf9b9"
)
BITFINEX_BUILDER = Path("training/download_bitfinex_margin_funding_stats_v2.py")
BITFINEX_BUILDER_SHA256 = (
    "b3bb9434dec618c8724ad584caa2fb66cd705d210dd66889b32ab80fd8f480ca"
)
BITFINEX_HEADER = (
    "symbol",
    "observation_time",
    "available_at",
    "timestamp_ms",
    "frr",
    "average_period_days",
    "funding_amount",
    "funding_amount_used",
    "funding_below_threshold",
)
BITFINEX_ALLOWED_COLUMNS = (
    "symbol",
    "observation_time",
    "available_at",
    "timestamp_ms",
    "average_period_days",
    "funding_amount",
    "funding_amount_used",
)

WBTC_SOURCE = Path(
    "data/wbtc_custody_bridge_flow_2020_2023/"
    "wbtc_mint_burn_2020_2023.csv.gz"
)
WBTC_SOURCE_SHA256 = (
    "bfcc6ebc2ded0cd8a57e5cda83a77daafe4de325adf606b23ba43ecf486b3b4e"
)
WBTC_MANIFEST = Path(
    "results/wbtc_custody_bridge_flow_source_manifest_2026-07-23.json"
)
WBTC_MANIFEST_SHA256 = (
    "e95267d55f390a35bf609580e014c67d44adabe67526022d5e80d555964274e8"
)
WBTC_MANIFEST_HASH = (
    "4e4344a7f2841803dc8da625ee1320f79e1821d54cb2366a5464728507b4bcab"
)
WBTC_BUILDER = Path("training/build_wbtc_custody_bridge_flow_source.py")
WBTC_BUILDER_SHA256 = (
    "70816fbcc94d5ecd11f99e3b1ebc3087e396c8f6972adcefd7c3a308f7c6fdbf"
)
WBTC_HEADER = (
    "asset",
    "contract_address",
    "event",
    "event_sign",
    "amount_raw",
    "decimals",
    "actor_address",
    "block_number",
    "block_hash",
    "block_timestamp",
    "transaction_hash",
    "transaction_index",
    "semantic_log_index",
    "companion_transfer_log_index",
    "confirmation_block_number",
    "confirmation_block_hash",
    "available_at",
)
WBTC_ALLOWED_COLUMNS = (
    "event",
    "amount_raw",
    "actor_address",
    "block_number",
    "block_hash",
    "transaction_hash",
    "transaction_index",
    "semantic_log_index",
    "available_at",
)

BFMWD_HEADER = (
    "candidate",
    "variant_id",
    "control",
    "split",
    "symbol",
    "side",
    "observation_time",
    "source_available_at",
    "decision_available_at",
    "entry_time",
    "exit_time",
)
WCDR_HEADER = (
    "candidate",
    "control",
    "signal_id",
    "window",
    "decision_time",
    "source_cutoff",
    "entry_time",
    "exit_time",
    "side",
    "wbtc_net_raw",
    "wbtc_gross_raw",
    "wbtc_count_net",
    "wbtc_rows",
    "wbtc_distinct_actors",
    "wbtc_top_actor_share",
    "usdc_net_raw",
    "usdc_gross_raw",
    "usdc_count_net",
    "usdc_rows",
)
WTSL_HEADER = (
    "candidate",
    "control",
    "signal_id",
    "window",
    "decision_time",
    "source_cutoff",
    "entry_time",
    "exit_time",
    "side",
    "wbtc_net_raw",
    "wbtc_gross_raw",
    "wbtc_rows",
    "wbtc_distinct_actors",
    "wbtc_top_actor_share",
    "wbtc_prior_median_twice_raw",
    "stablecoin_scope",
    "stablecoin_net_raw",
    "stablecoin_gross_raw",
    "stablecoin_rows",
    "stablecoin_veto_rows",
    "usdc_net_raw",
    "usdc_gross_raw",
    "usdt_net_raw",
    "usdt_gross_raw",
)
WSCF_HEADER = (
    "candidate",
    "control",
    "signal_id",
    "window",
    "wbtc_available_at",
    "anchor_time",
    "signal_time",
    "entry_time",
    "exit_time",
    "side",
    "wbtc_batch_identity",
    "wbtc_net_raw",
    "wbtc_gross_raw",
    "wbtc_rows",
    "wbtc_distinct_actors",
    "wbtc_top_actor_share",
    "confirmation_batch_identity",
    "stablecoin_scope",
    "cumulative_stablecoin_net_raw",
    "cumulative_stablecoin_gross_raw",
    "stablecoin_batches",
    "confirmation_delay_seconds",
    "usdc_net_raw",
    "usdc_gross_raw",
    "usdt_net_raw",
    "usdt_gross_raw",
)
LIVE_HEADER = (
    "candidate_id",
    "split",
    "decision_time",
    "entry_time",
    "exit_time",
    "side",
)


def _comparator(
    *,
    name: str,
    clock: str,
    clock_sha256: str,
    header: Sequence[str],
    manifest: str,
    manifest_sha256: str,
    manifest_hash: str,
    filters: Mapping[str, str] | None = None,
    group_field: str | None = None,
    groups: Sequence[str] = (),
) -> Mapping[str, Any]:
    return {
        "name": name,
        "clock": Path(clock),
        "clock_sha256": clock_sha256,
        "header": tuple(header),
        "manifest": Path(manifest),
        "manifest_sha256": manifest_sha256,
        "manifest_hash": manifest_hash,
        "filters": dict(filters or {}),
        "group_field": group_field,
        "groups": tuple(groups),
    }


COMPARATOR_SPECS: tuple[Mapping[str, Any], ...] = (
    _comparator(
        name="bfmwd_primary_variants",
        clock="data/bitfinex_margin_warehouse_deployment_clocks_2021_2023.csv.gz",
        clock_sha256=(
            "02b4fcc462a5a48be7673649f4cf4b2f9bb210baca4294eed1696d479820cccc"
        ),
        header=BFMWD_HEADER,
        manifest="results/bitfinex_margin_warehouse_deployment_support_2026-07-20.json",
        manifest_sha256=(
            "c857e070f4cb157a005f4a95bee0bff9c7b30daf97128832a690a68d05bfb79c"
        ),
        manifest_hash=(
            "d1e90ddf786b9e6ad3169260d56475a61395a9ae10da747fb2d7373a740989f2"
        ),
        filters={"control": "primary"},
        group_field="variant_id",
        groups=(
            "bfmwd_w12_d3_z10_h12",
            "bfmwd_w24_d3_z10_h12",
            "bfmwd_w12_d6_z10_h12",
            "bfmwd_w24_d6_z10_h12",
        ),
    ),
    _comparator(
        name="wcdr_primary",
        clock=(
            "data/wrapped_collateral_dollar_liquidity_rotation_2021_2023/"
            "wcdr2016_support_clocks_2021_2023.csv.gz"
        ),
        clock_sha256=(
            "241d96a64a654ba2faeda2d4a8460131269acf21d0bbbf31177d35d1ecd63b3c"
        ),
        header=WCDR_HEADER,
        manifest=(
            "results/wrapped_collateral_dollar_liquidity_rotation_"
            "support_2026-07-23.json"
        ),
        manifest_sha256=(
            "df3101a973c514c7c8297e7132b6c9d95fe7d10adf234dc5b5b29c497e972c35"
        ),
        manifest_hash=(
            "0a28128c820c1f5baf73c7653901056d3803e9bc0cce54b29a03afc7051ef600"
        ),
        filters={"candidate": "WCDR-2016", "control": "primary"},
        groups=("WCDR-2016|primary",),
    ),
    _comparator(
        name="wtsl_primary",
        clock=(
            "data/wbtc_turnover_stablecoin_liquidity_2021_2023/"
            "wtsl168_support_clocks_2021_2023.csv.gz"
        ),
        clock_sha256=(
            "df8cb085d439c9ee9e89334cb891b9e3b04f54c2a8e70bd4f552a90648ea8b6d"
        ),
        header=WTSL_HEADER,
        manifest="results/wbtc_turnover_stablecoin_liquidity_support_2026-07-23.json",
        manifest_sha256=(
            "1415b8e2a40f2aff908bfec1d1faa9621445c3fe87b41c43fd95a991725b23bd"
        ),
        manifest_hash=(
            "b53de47d743f7f61240e59ac3149c0a37467f6bb8ce580c9c3c2bc84341b7e9e"
        ),
        filters={"candidate": "WTSL-168-SOURCE-SEEN", "control": "primary"},
        groups=("WTSL-168-SOURCE-SEEN|primary",),
    ),
    _comparator(
        name="wscf_primary",
        clock=(
            "data/wbtc_stablecoin_finalized_confirmation_relay_2021_2023/"
            "wscf72_support_clocks_2021_2023.csv.gz"
        ),
        clock_sha256=(
            "86565774ae97a1024c5a66b4d59a1f5413bf4608398623359dd3ee24572f0ef3"
        ),
        header=WSCF_HEADER,
        manifest=(
            "results/wbtc_stablecoin_finalized_confirmation_relay_"
            "support_2026-07-23.json"
        ),
        manifest_sha256=(
            "add1f54034953d1040fdf5b34d794865fde84d05675c8b7f7f8e4e8c7918f2bd"
        ),
        manifest_hash=(
            "1a7ec88467779e461217af1430f79f21fdeb127ba7f29abd1a836a36c99b1faf"
        ),
        filters={
            "candidate": "WSCF-72-SOURCE-FAMILY-SEEN",
            "control": "primary",
        },
        groups=("WSCF-72-SOURCE-FAMILY-SEEN|primary",),
    ),
    _comparator(
        name="live_portfolio_pure_clocks",
        clock="results/cchr_live_portfolio_pure_clocks_2026-07-21.csv.gz",
        clock_sha256=(
            "73d6efbd35b3be64b0fa04fa9c8cb2db25866ef884f19b1ae673949e22a42b08"
        ),
        header=LIVE_HEADER,
        manifest="results/cchr_live_portfolio_pure_clock_manifest_2026-07-21.json",
        manifest_sha256=(
            "6c53ae482cf72bba0f286a47626842bf43070276ff5fe359be718e44864af57d"
        ),
        manifest_hash=(
            "2c240d4ffad4b8aa51434acebb74a61e8605fd0fa1a5e8e4554d3fa46fc1baea"
        ),
        group_field="candidate_id",
        groups=(
            "live:cand_rex_veto_7",
            "live:new_long_minimal_funding_premium",
            "live:oi_upbit_ratio288_low",
        ),
    ),
)

PRIOR_RESEARCH_DISCLOSURE: Mapping[str, Any] = {
    "candidate_specific_outcome_blind": True,
    "globally_clean_room": False,
    "bitfinex_source_family_exposed": True,
    "wbtc_source_family_exposed": True,
    "adjacent_source_support_aggregates_exposed": True,
    "wscf_aggregate_comparator_summary_exposed": True,
    "raw_comparator_rows_exposed_during_fccm_selection": False,
    "bfmwd_train_result_opened_during_fccm_selection": False,
    "fccm_source_values_opened": False,
    "fccm_features_or_incidence_opened": False,
    "fccm_market_outcomes_opened": False,
}

EXPECTED_BOUNDARY: Mapping[str, Any] = {
    "mechanism_file_bytes_hashed": True,
    "boundary_ledger_parsed": True,
    "boundary_referenced_file_bytes_hashed": 27,
    "source_file_bytes_hashed": 2,
    "source_manifest_metadata_parsed": 2,
    "source_headers_decoded": 2,
    "source_value_rows_decoded": 0,
    "comparator_file_bytes_hashed": 5,
    "comparator_manifest_file_bytes_hashed": 5,
    "comparator_headers_decoded": 5,
    "comparator_value_rows_decoded": 0,
    "fccm_features_computed": 0,
    "fccm_states_or_incidence_derived": 0,
    "btc_market_rows_decoded": 0,
    "realized_funding_rows_decoded": 0,
    "future_return_rows_decoded": 0,
    "pnl_cagr_mdd_values_decoded": 0,
    "post_2023_source_value_rows_decoded": 0,
    "network_calls": 0,
    "git_protocol_subprocess_calls_during_artifact_write": 2,
}
VERIFIED_UNCOMMITTED_BOUNDARY: Mapping[str, Any] = {
    **EXPECTED_BOUNDARY,
    "git_protocol_subprocess_calls_during_artifact_write": 0,
}
STATIC_BOUNDARY: Mapping[str, Any] = {
    **VERIFIED_UNCOMMITTED_BOUNDARY,
    "mechanism_file_bytes_hashed": False,
    "boundary_ledger_parsed": False,
    "boundary_referenced_file_bytes_hashed": 0,
    "source_file_bytes_hashed": 0,
    "source_manifest_metadata_parsed": 0,
    "source_headers_decoded": 0,
    "comparator_file_bytes_hashed": 0,
    "comparator_manifest_file_bytes_hashed": 0,
    "comparator_headers_decoded": 0,
    "git_protocol_subprocess_calls_during_artifact_write": 0,
}


@dataclass(frozen=True)
class Config:
    output: str = str(DEFAULT_OUTPUT)


def _repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise RuntimeError("FCCM path must be repository-relative")
    resolved = (REPOSITORY_ROOT / candidate).resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError("FCCM path must remain repository-relative") from exc
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


def _read_gzip_header(path: str | Path) -> tuple[str, ...]:
    with gzip.open(_repository_path(path), "rt", encoding="utf-8", newline="") as handle:
        return tuple(next(csv.reader(handle)))


def _validate_hash(path: str | Path, expected: str, label: str) -> None:
    if sha256_file(path) != expected:
        raise RuntimeError(f"FCCM {label} hash mismatch")


def _validate_boundary_ledger() -> Mapping[str, Any]:
    _validate_hash(BOUNDARY_LEDGER, BOUNDARY_LEDGER_SHA256, "boundary ledger")
    payload = json.loads(_repository_path(BOUNDARY_LEDGER).read_text(encoding="utf-8"))
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if canonical_hash(core) != BOUNDARY_MANIFEST_HASH:
        raise RuntimeError("FCCM boundary ledger canonical hash mismatch")
    if payload.get("manifest_hash") != BOUNDARY_MANIFEST_HASH:
        raise RuntimeError("FCCM boundary ledger manifest mismatch")
    if payload.get("candidate") != POLICY_ID:
        raise RuntimeError("FCCM boundary ledger candidate mismatch")
    counters = payload.get("fccm_boundary_counters")
    if not isinstance(counters, Mapping):
        raise RuntimeError("FCCM boundary counters missing")
    for field in (
        "bitfinex_source_value_rows_decoded",
        "wbtc_source_value_rows_decoded",
        "fccm_features_derived",
        "fccm_states_or_transitions_derived",
        "fccm_candidate_incidence_rows_derived",
        "raw_comparator_data_rows_decoded",
        "btc_market_rows_decoded",
        "realized_funding_rows_decoded",
        "future_return_rows_decoded",
        "pnl_cagr_mdd_values_decoded",
        "post_2023_bitfinex_or_wbtc_source_value_rows_decoded",
    ):
        if counters.get(field) != 0:
            raise RuntimeError(f"FCCM boundary opened: {field}")
    artifacts = payload.get("direct_selection_artifacts_seen")
    if not isinstance(artifacts, list) or len(artifacts) != 27:
        raise RuntimeError("FCCM boundary artifact inventory drift")
    for binding in artifacts:
        if sha256_file(binding["path"]) != binding["sha256"]:
            raise RuntimeError("FCCM boundary referenced artifact drift")
    return payload


def _source_bindings(*, verify: bool) -> dict[str, Any]:
    if verify:
        for path, digest, label in (
            (BITFINEX_SOURCE, BITFINEX_SOURCE_SHA256, "Bitfinex source"),
            (BITFINEX_MANIFEST, BITFINEX_MANIFEST_SHA256, "Bitfinex manifest"),
            (BITFINEX_BUILDER, BITFINEX_BUILDER_SHA256, "Bitfinex builder"),
            (WBTC_SOURCE, WBTC_SOURCE_SHA256, "WBTC source"),
            (WBTC_MANIFEST, WBTC_MANIFEST_SHA256, "WBTC manifest"),
            (WBTC_BUILDER, WBTC_BUILDER_SHA256, "WBTC builder"),
        ):
            _validate_hash(path, digest, label)
        if _read_gzip_header(BITFINEX_SOURCE) != BITFINEX_HEADER:
            raise RuntimeError("FCCM Bitfinex source header drift")
        if _read_gzip_header(WBTC_SOURCE) != WBTC_HEADER:
            raise RuntimeError("FCCM WBTC source header drift")
        bitfinex = json.loads(
            _repository_path(BITFINEX_MANIFEST).read_text(encoding="utf-8")
        )
        canonical = bitfinex.get("files", {}).get("canonical", {})
        contract = bitfinex.get("source_contract", {})
        if (
            canonical.get("path") != str(BITFINEX_SOURCE)
            or canonical.get("sha256") != BITFINEX_SOURCE_SHA256
            or canonical.get("rows") != 70_116
            or contract.get("symbols") != ["fUSD", "fBTC"]
            or contract.get("outcomes_opened") is not False
            or contract.get("market_or_pnl_columns_loaded") is not False
            or contract.get("post_2023_rows_requested") is not False
        ):
            raise RuntimeError("FCCM Bitfinex source manifest drift")
        wbtc = json.loads(_repository_path(WBTC_MANIFEST).read_text(encoding="utf-8"))
        outcome = wbtc.get("outcome_boundary", {})
        output = wbtc.get("output", {})
        if (
            wbtc.get("manifest_hash") != WBTC_MANIFEST_HASH
            or output.get("path") != str(WBTC_SOURCE)
            or output.get("sha256") != WBTC_SOURCE_SHA256
            or output.get("rows") != 993
            or outcome.get("btc_market_rows_read") != 0
            or outcome.get("funding_rows_read") != 0
            or outcome.get("future_return_rows_read") != 0
            or outcome.get("post_2023_contract_event_rows_read") != 0
            or outcome.get("pnl_cagr_mdd_opened") is not False
            or outcome.get("mechanism_features_opened") is not False
        ):
            raise RuntimeError("FCCM WBTC source manifest drift")
    return {
        "bitfinex": {
            "source": str(BITFINEX_SOURCE),
            "source_sha256": BITFINEX_SOURCE_SHA256,
            "manifest": str(BITFINEX_MANIFEST),
            "manifest_sha256": BITFINEX_MANIFEST_SHA256,
            "builder": str(BITFINEX_BUILDER),
            "builder_sha256": BITFINEX_BUILDER_SHA256,
            "header": list(BITFINEX_HEADER),
            "allowed_columns": list(BITFINEX_ALLOWED_COLUMNS),
            "manifest_rows": 70_116,
            "value_rows_read_during_preregistration": 0,
        },
        "wbtc": {
            "source": str(WBTC_SOURCE),
            "source_sha256": WBTC_SOURCE_SHA256,
            "manifest": str(WBTC_MANIFEST),
            "manifest_sha256": WBTC_MANIFEST_SHA256,
            "manifest_hash": WBTC_MANIFEST_HASH,
            "builder": str(WBTC_BUILDER),
            "builder_sha256": WBTC_BUILDER_SHA256,
            "header": list(WBTC_HEADER),
            "allowed_columns": list(WBTC_ALLOWED_COLUMNS),
            "manifest_rows": 993,
            "value_rows_read_during_preregistration": 0,
        },
    }


def _comparator_bindings(*, verify: bool) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for spec in COMPARATOR_SPECS:
        if verify:
            _validate_hash(spec["clock"], spec["clock_sha256"], spec["name"])
            _validate_hash(
                spec["manifest"],
                spec["manifest_sha256"],
                f"{spec['name']} manifest",
            )
            if _read_gzip_header(spec["clock"]) != spec["header"]:
                raise RuntimeError(f"FCCM comparator header drift: {spec['name']}")
        output.append(
            {
                "name": spec["name"],
                "clock": str(spec["clock"]),
                "clock_sha256": spec["clock_sha256"],
                "header": list(spec["header"]),
                "manifest": str(spec["manifest"]),
                "manifest_sha256": spec["manifest_sha256"],
                "manifest_hash": spec["manifest_hash"],
                "filters": dict(spec["filters"]),
                "group_field": spec["group_field"],
                "groups": list(spec["groups"]),
                "entry_field": "entry_time",
                "exit_field": "exit_time",
                "side_field": "side",
                "comparison": [
                    "2021-01-01T00:00:00Z",
                    "2024-01-01T00:00:00Z",
                ],
                "value_rows_read_during_preregistration": 0,
            }
        )
    return output


def policy_payload() -> dict[str, Any]:
    return {
        "candidate": POLICY_ID,
        "singleton": True,
        "research_status": "source_family_exposed_candidate_incidence_outcome_blind",
        "contamination": dict(PRIOR_RESEARCH_DISCLOSURE),
        "economic_hypothesis": {
            "direction_source": "three-component Bitfinex fUSD-fBTC consensus",
            "long": "relative fUSD utilization, draw, and tenor deployment",
            "short": "relative fBTC utilization, draw, and tenor deployment",
            "wbtc_role": "signless custody-mobility sponsorship only",
            "mint_or_burn_direction_forbidden": True,
            "stablecoin_confirmation_forbidden": True,
        },
        "source_alignment": {
            "bitfinex_hour": "floor(observation_time,1h)",
            "exact_pair": ["fUSD", "fBTC"],
            "pair_validity": (
                "exactly one row per symbol; 0<=used<=total; positive totals; "
                "all permitted numerics finite"
            ),
            "pair_availability": "max(two source available_at values)",
            "missing_current_known_at": "H+1h+15m",
            "partial_pair_invalid_at": (
                "max(present-row available_at,H+1h+15m)"
            ),
            "duplicate_or_invalid_pair_invalid_at": (
                "max(source available_at among rows)"
            ),
            "exact_lag": "H-24h",
            "current_or_lag_failure": "invalidate H and reset direction to neutral",
            "equal_availability": (
                "one causal batch; prebatch ranks; any invalid resets; greatest H decides"
            ),
            "batch_order": [
                "compute all validity/features/ranks from strictly prebatch history",
                "any invalid anchor resets whole batch and blocks state establishment",
                "append valid feature rows only after every batch rank is fixed",
                "only greatest H can establish, transition, suppress, or emit",
            ],
            "first_valid_after_reset": "baseline only",
            "fill_or_interpolation": False,
        },
        "arithmetic": {
            "representation": "exact rational parsed from source decimal text",
            "binary_float_log_logit_clip_epsilon_forbidden": True,
        },
        "bitfinex_features": {
            "util_s": "used_s[H]/total_s[H]",
            "draw_s": "(unused_s[H-24h]-unused_s[H])/total_s[H-24h]",
            "util_rotation": "util_fUSD-util_fBTC",
            "draw_rotation": "draw_fUSD-draw_fBTC",
            "tenor_rotation": "average_period_days_fUSD-average_period_days_fBTC",
        },
        "bitfinex_normalization": {
            "components": ["util_rotation", "draw_rotation", "tenor_rotation"],
            "strict_prior_valid_pairs": 720,
            "history_availability": "strictly before current causal batch",
            "current_batch_excluded": True,
            "all_prior_values_required": True,
            "midrank_unit": "(2*count(prior<x)+count(prior==x)-720)/720",
            "positive_vote": "u>=1/4",
            "negative_vote": "u<=-1/4",
            "long_state": "at least two positive votes and mean(u)>=1/3",
            "short_state": "at least two negative votes and mean(u)<=-1/3",
            "otherwise": "neutral",
        },
        "wbtc_sponsorship": {
            "daily_anchor": "00:00:00Z",
            "window": "D-14d < available_at <= D",
            "gross": "sum(amount_raw); sign forbidden",
            "actor_count": "distinct nonzero actor_address",
            "top_share": "max(actor gross)/gross",
            "prior_daily_anchors": 180,
            "prior_anchor_range": "D-180d through D-1d inclusive",
            "each_prior_uses_own_14d_available_at_window": True,
            "zero_gross_prior_anchors_included": True,
            "midrank_unit": "(2*count(prior<gross)+count(prior==gross)-180)/180",
            "active": "u_gross>=1/5 and actor_count>=2 and top_share<=4/5",
            "zero_or_lt2_actor_top_share": "null and never compared",
            "coverage_or_integrity_failure": (
                "invalidate current and all dependent anchors; never shorten history"
            ),
            "membership_time": "available_at only; block_timestamp forbidden",
            "directional_fields_forbidden": [
                "event",
                "event_sign",
                "mint_or_burn_direction",
                "net_flow",
            ],
            "latest_state_staleness": "less than 24 elapsed hours",
        },
        "state_machine": {
            "opportunity": "transition from established state !=p into p",
            "wbtc_must_be_active_at_transition": True,
            "state_updates_when_wbtc_inactive": True,
            "inactive_suppression_queued": False,
            "staying_in_p_after_suppression_can_trade": False,
            "invalid_batch_resets": True,
            "same_batch_only_greatest_hour_eligible": True,
            "side": {"1": "LONG", "-1": "SHORT"},
        },
        "execution": {
            "entry": "ceil_to_5m(signal_available_at)+5m",
            "exact_grid_signal_still_adds_5m": True,
            "hold_elapsed_hours": 72,
            "hold_bars_5m": 864,
            "notional_exposure": "1/2",
            "reservation_interval": "[entry,exit)",
            "global_nonoverlap": True,
            "split_containment": True,
            "acceptance": "entry>=prior accepted exit",
            "suppressed_queueing": False,
            "dynamic_exit_size_regime_or_direction_override": False,
        },
        "windows": {
            "warmup": ["2020-01-01T00:00:00Z", "2021-01-01T00:00:00Z"],
            "train": ["2021-01-01T00:00:00Z", "2023-01-01T00:00:00Z"],
            "selection": ["2023-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "sealed_from": "2024-01-01T00:00:00Z",
        },
        "source_support_gates": {
            "train_total_minimum": 60,
            "each_train_year_minimum": 24,
            "each_train_half_minimum": 10,
            "train_each_side_minimum": 15,
            "selection_total_minimum": 24,
            "each_selection_half_minimum": 10,
            "selection_each_side_minimum": 6,
            "every_quarter_active": True,
            "train_maximum_month_share": "3/20",
            "selection_maximum_month_share": "1/5",
            "train_maximum_entry_gap_days": 45,
            "selection_maximum_entry_gap_days": 60,
            "maximum_consecutive_same_side": 8,
            "train_distinct_wbtc_actors_minimum": 10,
            "selection_distinct_wbtc_actors_minimum": 5,
            "wbtc_raw_transition_active_share": {
                "denominator": (
                    "raw_directional_bitfinex_transitions_before_nonoverlap"
                ),
                "train": {"minimum": "1/5", "maximum": "7/10"},
                "selection": {"minimum": "1/5", "maximum": "7/10"},
            },
            "bitfinex_component_vote_with_accepted_side_share": {
                "denominator": "accepted_entries_within_split",
                "components": [
                    "utilization",
                    "draw",
                    "tenor",
                ],
                "train_minimum_each_component": "7/20",
                "selection_minimum_each_component": "7/20",
            },
            "train_distinct_vote_patterns_minimum": 3,
            "selection_distinct_vote_patterns_minimum": 2,
            "uniqueness_required": [
                "bitfinex_anchor",
                "wbtc_anchor",
                "candidate_identity",
                "entry",
                "occupied_interval",
            ],
            "post_2023_source_value_rows": 0,
            "failure_action": "reject before comparator rows and outcomes",
        },
        "controls": {
            "causal": [
                "bitfinex_consensus_only",
                "utilization_only",
                "draw_only",
                "tenor_only",
                "majority_without_score",
                "wbtc_stale_7d",
                "bitfinex_stale_24h",
                "exact_direction_flip",
                "deterministic_random_side",
                "one_bar_delay",
            ],
            "definitions": {
                "bitfinex_consensus_only": (
                    "exact directional transitions without WBTC sponsorship"
                ),
                "utilization_only": "utilization-vote transitions with active WBTC",
                "draw_only": "draw-vote transitions with active WBTC",
                "tenor_only": "tenor-vote transitions with active WBTC",
                "majority_without_score": (
                    "two-of-three vote majority without mean score; active WBTC"
                ),
                "wbtc_stale_7d": (
                    "use already-computed state at D-7d; unchanged Bitfinex clock"
                ),
                "bitfinex_stale_24h": (
                    "shift exact H-24h state/transition to H; current WBTC lookup; "
                    "unchanged H clock"
                ),
                "exact_direction_flip": "primary entries with side reversed",
                "deterministic_random_side": (
                    "SHA256(UTF-8(FCCM-72|random-side|canonical_entry)); "
                    "byte0<128 LONG else SHORT"
                ),
                "one_bar_delay": (
                    "shift primary entry and exit 5m; drop rows leaving original split"
                ),
            },
            "each_causal_control_owns_nonoverlap_scheduler": True,
            "random_side": (
                "SHA256(UTF-8(FCCM-72|random-side|canonical_entry)); "
                "byte0<128 LONG else SHORT"
            ),
            "noncausal_source_placebos": [
                "within_year_wbtc_amount_hash_permutation",
                "within_year_wbtc_actor_hash_permutation",
            ],
            "placebo_assignment": (
                "within available_at year, destination source identities canonical; "
                "values ordered by (SHA256(UTF-8(FCCM-72|placebo|F|year|"
                "source_identity)),source_identity)"
            ),
            "placebo_multiset_preserved": True,
            "placebo_rng_or_tunable_seed": False,
            "placebo_clock_or_economics_forbidden": True,
        },
        "identities": {
            "bitfinex_row": "symbol|timestamp_ms",
            "paired_hour": (
                "SHA256(UTF-8(FCCM-72|bitfinex-pair|H|fUSD|"
                "fUSD_timestamp_ms|fBTC|fBTC_timestamp_ms))"
            ),
            "wbtc_event": "block_hash|transaction_hash|semantic_log_index",
            "wbtc_window": (
                "SHA256(UTF-8(FCCM-72|wbtc-window|D\\n + sorted event "
                "identities joined by newline))"
            ),
            "primary": (
                "SHA256(UTF-8(FCCM-72|candidate|paired_hour_hash|"
                "wbtc_window_hash|p|entry|exit))"
            ),
            "control": (
                "SHA256(UTF-8(FCCM-72|control-row|control|paired_hour_hash|"
                "wbtc_window_hash|side|entry|exit))"
            ),
            "comparator_row": (
                "SHA256(UTF-8(FCCM-72|comparator|artifact_sha256|group_name|"
                "entry_time|exit_time|side))"
            ),
            "primary_uniqueness_scope": "primary rows",
            "control_uniqueness_scope": "within each named control",
            "clock_sort": "(entry,row_identity,control)",
            "serialization": "UTF-8 canonical UTC seconds; sorted JSON; gzip mtime=0",
        },
        "novelty": {
            "comparators": [spec["name"] for spec in COMPARATOR_SPECS],
            "maximum_signless_exact_entry_jaccard": "1/10",
            "one_to_one_tolerance_elapsed_hours": 6,
            "maximum_bidirectional_signless_containment": "7/20",
            "maximum_absolute_signed_exposure_correlation": "2/5",
            "matching": "chronological greedy; nearest then right time then identity",
            "matching_run_independently_both_directions": True,
            "same_side_report_only": True,
            "exposure_grid": "5m [2021-01-01,2024-01-01); side*1/2 else 0",
            "zero_variance_nonempty_group": "fail",
            "empty_overlap_group": "report ineligible-empty; contributes no ratio",
            "duplicates_or_overlapping_group_intervals": "fail closed",
            "required_side_encoding": ["1", "-1"],
        },
        "economics": {
            "instrument_execution": "BTCUSDT USD-M 5m next-open",
            "base_cost_notional_per_side": "0.0006",
            "stress_cost_notional_per_side": "0.0010",
            "realized_funding": True,
            "full_calendar_cagr": True,
            "strict_global_hwm_favorable_then_adverse": True,
            "base_cagr_to_strict_mdd_minimum": "3",
            "stress_cagr_to_strict_mdd_minimum": "5/2",
            "strict_mdd_pct_maximum": "15",
            "mean_gross_underlying_bp_minimum": "30",
            "absolute_return_positive": True,
            "contained_halves_positive": True,
            "each_side_contribution_positive": True,
            "one_bar_delay_positive": True,
            "weekly_sign_draws": 100_000,
            "weekly_sign_draw_indices": "0..99999",
            "weekly_sign_seed_namespace": "FCCM-72|weekly-sign|20260723",
            "weekly_sign_statistic": "arithmetic mean UTC Monday weekly net return",
            "weekly_sign_decimal_precision": 50,
            "weekly_sign_p_formula": "(1+count(T_draw>=T_observed))/100001",
            "weekly_sign_p_maximum": "1/10",
            "minimum_nonzero_weeks": {"train": 20, "selection": 10},
            "control_cagr_mdd_margin": "1/4",
            "control_mean_gross_margin_bp": "5",
            "controls_to_beat": [
                "bitfinex_consensus_only",
                "bitfinex_stale_24h",
                "utilization_only",
                "draw_only",
                "tenor_only",
            ],
            "stale_bitfinex_full_qualification_rejects": True,
            "train_opens_before_selection": True,
            "selection_sealed_on_train_failure": True,
            "post_2023_sealed_until_pre2024_pass": True,
        },
        "sequence": [
            "source support",
            "novelty evaluator freeze and novelty",
            "economic evaluator freeze",
            "train 2021-2022",
            "selection 2023",
            "immutable source extension and test 2024",
            "eval 2025",
            "recent 2026",
        ],
        "rllm_boundary": {
            "authorized_before_deterministic_train_and_selection_pass": False,
            "later_role": "allocation or abstention among independently validated alphas",
            "may_change_features_side_clock_hold_or_economics": False,
        },
        "live_parity": {
            "minimum_shadow_days": 90,
            "finalized_head_coverage_required": True,
            "persist_raw_identity_revision_provider_reorg_state": True,
            "deterministic_replay_required": True,
            "historical_results_authorize_capital": False,
        },
        "mutable_parameters": [],
        "stopping_rule": "any failed stage retires FCCM-72 unchanged",
    }


def _build_preregistration(*, verify_bindings: bool) -> dict[str, Any]:
    if verify_bindings:
        _validate_hash(MECHANISM_DECISION, MECHANISM_DECISION_SHA256, "mechanism")
        _validate_hash(
            COMMON_WINDOW_POLICY,
            COMMON_WINDOW_POLICY_SHA256,
            "common-window policy",
        )
        ledger = _validate_boundary_ledger()
    else:
        ledger = {
            "manifest_hash": BOUNDARY_MANIFEST_HASH,
            "evidence_kind": "hash_bound_self_attested_boundary_ledger",
        }
    policy = policy_payload()
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "candidate": POLICY_ID,
        "config": asdict(Config()),
        "mechanism_decision": {
            "path": str(MECHANISM_DECISION),
            "sha256": MECHANISM_DECISION_SHA256,
        },
        "boundary_ledger": {
            "path": str(BOUNDARY_LEDGER),
            "sha256": BOUNDARY_LEDGER_SHA256,
            "manifest_hash": ledger["manifest_hash"],
            "evidence_kind": ledger["evidence_kind"],
            "independent_os_trace": False,
        },
        "common_window_policy": {
            "path": str(COMMON_WINDOW_POLICY),
            "sha256": COMMON_WINDOW_POLICY_SHA256,
        },
        "source_bindings": _source_bindings(verify=verify_bindings),
        "comparator_bindings": _comparator_bindings(verify=verify_bindings),
        "policy": policy,
        "policy_hash": canonical_hash(policy),
        "prior_research_disclosure": dict(PRIOR_RESEARCH_DISCLOSURE),
        "verification_mode": (
            "verified_hashes_and_headers_uncommitted"
            if verify_bindings
            else "static_test_fixture"
        ),
        "artifact_eligible": False,
        "fccm_source_values_or_incidence_opened": False,
        "comparator_rows_opened_during_preregistration": False,
        "outcomes_opened": False,
        "performance_values_opened": False,
        "outcome_boundary": dict(
            VERIFIED_UNCOMMITTED_BOUNDARY if verify_bindings else STATIC_BOUNDARY
        ),
        "preregistration_source": {
            "path": str(SCRIPT_PATH),
            "sha256": sha256_file(SCRIPT_PATH),
        },
        "next_action": "build committed outcome-blind FCCM source-support clocks",
    }
    payload["manifest_hash"] = canonical_hash(payload)
    return payload


def build_preregistration(*, verify_bindings: bool = True) -> dict[str, Any]:
    """Build an inspectable payload that can never claim artifact eligibility."""

    return _build_preregistration(
        verify_bindings=verify_bindings,
    )


def _validate_preregistration(
    payload: Mapping[str, Any],
    *,
    verify_bindings: bool = True,
    commit_guard_verified: bool = False,
) -> None:
    if commit_guard_verified and not verify_bindings:
        raise RuntimeError("FCCM commit guard cannot qualify an unverified payload")
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("FCCM preregistration canonical hash mismatch")
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("FCCM preregistration protocol drift")
    if payload.get("candidate") != POLICY_ID:
        raise RuntimeError("FCCM preregistration candidate drift")
    if payload.get("policy") != policy_payload():
        raise RuntimeError("FCCM frozen policy drift")
    if payload.get("policy_hash") != canonical_hash(policy_payload()):
        raise RuntimeError("FCCM policy hash drift")
    if payload.get("prior_research_disclosure") != PRIOR_RESEARCH_DISCLOSURE:
        raise RuntimeError("FCCM prior-research disclosure drift")
    expected_boundary = (
        EXPECTED_BOUNDARY
        if commit_guard_verified
        else (
            VERIFIED_UNCOMMITTED_BOUNDARY
            if verify_bindings
            else STATIC_BOUNDARY
        )
    )
    if payload.get("outcome_boundary") != expected_boundary:
        raise RuntimeError("FCCM outcome boundary drift")
    for field in (
        "fccm_source_values_or_incidence_opened",
        "comparator_rows_opened_during_preregistration",
        "outcomes_opened",
        "performance_values_opened",
    ):
        if payload.get(field) is not False:
            raise RuntimeError(f"FCCM boundary opened: {field}")
    if payload.get("artifact_eligible") is not commit_guard_verified:
        raise RuntimeError("FCCM artifact eligibility drift")
    expected_mode = (
        "verified_hashes_headers_and_commit_guard"
        if commit_guard_verified
        else (
            "verified_hashes_and_headers_uncommitted"
            if verify_bindings
            else "static_test_fixture"
        )
    )
    if payload.get("verification_mode") != expected_mode:
        raise RuntimeError("FCCM verification mode drift")
    # Reconstruct the exact expected envelope without reopening source or
    # comparator value containers.  The verified build above already checked
    # every frozen byte hash and decoded headers only; validation must not
    # silently double the declared access counters.
    expected = _build_preregistration(
        verify_bindings=False,
    )
    expected["config"] = dict(payload.get("config", {}))
    if verify_bindings:
        expected["verification_mode"] = expected_mode
        expected["artifact_eligible"] = commit_guard_verified
        expected["outcome_boundary"] = dict(expected_boundary)
    expected_core = {
        key: value for key, value in expected.items() if key != "manifest_hash"
    }
    expected["manifest_hash"] = canonical_hash(expected_core)
    if dict(payload) != expected:
        raise RuntimeError("FCCM preregistration binding drift")


def validate_preregistration(
    payload: Mapping[str, Any], *, verify_bindings: bool = True
) -> None:
    """Validate inspectable direct builds; eligible artifacts are write-path only."""

    if payload.get("artifact_eligible") is True:
        raise RuntimeError(
            "eligible FCCM preregistrations are validated only by the write path"
        )
    _validate_preregistration(
        payload,
        verify_bindings=verify_bindings,
        commit_guard_verified=False,
    )


def _git_check(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git", *args),
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _assert_protocol_committed() -> None:
    paths = (SCRIPT_PATH, TEST_PATH)
    labels = tuple(str(path) for path in paths)
    if not all(_repository_path(path).is_file() for path in paths):
        raise RuntimeError("FCCM preregistration protocol file is missing")
    tracked = _git_check("ls-files", "--error-unmatch", "--", *labels)
    if tracked.returncode != 0:
        raise RuntimeError("FCCM preregistration protocol is not committed")
    clean = _git_check("diff", "--quiet", "HEAD", "--", *labels)
    if clean.returncode != 0:
        raise RuntimeError("FCCM preregistration protocol differs from HEAD")


def _atomic_write(path: Path, payload: bytes) -> None:
    try:
        path.resolve().relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError("FCCM output must remain inside repository") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_preregistration(cfg: Config = Config()) -> tuple[dict[str, Any], str]:
    _assert_protocol_committed()
    output = _repository_path(cfg.output)
    payload = _build_preregistration(verify_bindings=True)
    # Eligibility is only conferred here, after the two commit checks above.
    # Neither public nor private build helpers can manufacture this envelope.
    payload["verification_mode"] = "verified_hashes_headers_and_commit_guard"
    payload["artifact_eligible"] = True
    payload["outcome_boundary"] = dict(EXPECTED_BOUNDARY)
    payload["config"] = asdict(cfg)
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    payload["manifest_hash"] = canonical_hash(core)
    _validate_preregistration(
        payload,
        verify_bindings=True,
        commit_guard_verified=True,
    )
    expected_bytes = serialized_payload(payload)
    if output.exists():
        if output.read_bytes() != expected_bytes:
            raise RuntimeError("existing FCCM preregistration differs")
        return payload, "verified_existing"
    try:
        _atomic_write(output, expected_bytes)
        return payload, "created"
    except FileExistsError:
        if output.read_bytes() != expected_bytes:
            raise RuntimeError("concurrent FCCM preregistration differs")
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
                "candidate": payload["candidate"],
                "status": status,
                "output": args.output,
                "policy_hash": payload["policy_hash"],
                "manifest_hash": payload["manifest_hash"],
                "source_values_or_incidence_opened": payload[
                    "fccm_source_values_or_incidence_opened"
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
