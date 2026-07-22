"""Freeze WSCF-72-SOURCE-FAMILY-SEEN before exact incidence or BTC outcomes."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence


POLICY_ID = "WSCF-72-SOURCE-FAMILY-SEEN"
PROTOCOL_VERSION = "wbtc_stablecoin_finalized_confirmation_relay_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(
    "training/preregister_wbtc_stablecoin_finalized_confirmation_relay.py"
)
MECHANISM_DECISION = Path(
    "docs/wbtc-stablecoin-finalized-confirmation-relay-"
    "mechanism-decision-2026-07-23.md"
)
MECHANISM_DECISION_SHA256 = (
    "5a453f30897a27b9d96fad1af0b8f99d3d5993e439fcfa99927457f11d6ff9ee"
)
DEFAULT_OUTPUT = Path(
    "results/wbtc_stablecoin_finalized_confirmation_relay_"
    "preregistration_2026-07-23.json"
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
WBTC_MANIFEST_FILE_SHA256 = (
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
    "asset",
    "event",
    "event_sign",
    "amount_raw",
    "decimals",
    "actor_address",
    "block_number",
    "transaction_index",
    "semantic_log_index",
    "confirmation_block_number",
    "available_at",
)

STABLECOIN_SOURCE = Path(
    "data/ethereum_stablecoin_issuance_redemption_2020_2023/"
    "ethereum_usdt_usdc_issuance_redemption_2020_2023.csv.gz"
)
STABLECOIN_SOURCE_SHA256 = (
    "70ba3799ba84dc671051623a8d167b1731f043cf84a686b9878a67fcd52e5901"
)
STABLECOIN_MANIFEST = Path(
    "results/ethereum_stablecoin_issuance_redemption_"
    "source_manifest_2026-07-21.json"
)
STABLECOIN_MANIFEST_FILE_SHA256 = (
    "8ec9ab08c413bf6f5f8170fb800b05105522d4cf1a7932943c214288701e31fe"
)
STABLECOIN_MANIFEST_HASH = (
    "a0c7740db64f7779fade68d76985c629cabe81983bf594e8258cef16a5725a1b"
)
STABLECOIN_BUILDER = Path(
    "training/build_ethereum_stablecoin_issuance_redemption.py"
)
STABLECOIN_BUILDER_SHA256 = (
    "af699d91404e44298573ca148c6cf1b90b68d9aac2972da9ad228a26b9e8a9a5"
)
STABLECOIN_HEADER = (
    "asset",
    "contract_address",
    "event",
    "event_sign",
    "amount_raw",
    "decimals",
    "indexed_address_1",
    "indexed_address_2",
    "data_address",
    "block_number",
    "block_hash",
    "block_timestamp",
    "transaction_hash",
    "transaction_index",
    "log_index",
    "confirmation_block_number",
    "confirmation_block_hash",
    "available_at",
)
STABLECOIN_ALLOWED_COLUMNS = (
    "asset",
    "event",
    "event_sign",
    "amount_raw",
    "decimals",
    "block_number",
    "transaction_index",
    "log_index",
    "confirmation_block_number",
    "available_at",
)

WCDR_CLOCK_HEADER = (
    "candidate", "control", "signal_id", "window", "decision_time",
    "source_cutoff", "entry_time", "exit_time", "side", "wbtc_net_raw",
    "wbtc_gross_raw", "wbtc_count_net", "wbtc_rows",
    "wbtc_distinct_actors", "wbtc_top_actor_share", "usdc_net_raw",
    "usdc_gross_raw", "usdc_count_net", "usdc_rows",
)
WTSL_CLOCK_HEADER = (
    "candidate", "control", "signal_id", "window", "decision_time",
    "source_cutoff", "entry_time", "exit_time", "side", "wbtc_net_raw",
    "wbtc_gross_raw", "wbtc_rows", "wbtc_distinct_actors",
    "wbtc_top_actor_share", "wbtc_prior_median_twice_raw",
    "stablecoin_scope", "stablecoin_net_raw", "stablecoin_gross_raw",
    "stablecoin_rows", "stablecoin_veto_rows", "usdc_net_raw",
    "usdc_gross_raw", "usdt_net_raw", "usdt_gross_raw",
)
UGCI_CLOCK_HEADER = (
    "candidate", "control", "signal_id", "source_packet_start",
    "source_packet_end", "feature_available_time", "decision_time",
    "entry_time", "exit_time", "side", "mint_raw", "burn_raw",
    "gross_raw", "net_raw", "imbalance_ratio", "prior_gross_q95",
    "prior_history_packets",
)
SEALED_PRIOR_HEADER = (
    "candidate", "control", "entry_time", "comparison_start",
    "comparison_end_exclusive",
)
LIVE_CLOCK_HEADER = (
    "candidate_id", "split", "decision_time", "entry_time", "exit_time",
    "side",
)

COMPARATOR_SPECS: tuple[dict[str, Any], ...] = (
    {
        "name": "wcdr_primary",
        "clock": Path(
            "data/wrapped_collateral_dollar_liquidity_rotation_2021_2023/"
            "wcdr2016_support_clocks_2021_2023.csv.gz"
        ),
        "clock_sha256": (
            "241d96a64a654ba2faeda2d4a8460131269acf21d0bbbf31177d35d1ecd63b3c"
        ),
        "header": WCDR_CLOCK_HEADER,
        "manifest": Path(
            "results/wrapped_collateral_dollar_liquidity_rotation_"
            "support_2026-07-23.json"
        ),
        "manifest_file_sha256": (
            "df3101a973c514c7c8297e7132b6c9d95fe7d10adf234dc5b5b29c497e972c35"
        ),
        "manifest_hash": (
            "0a28128c820c1f5baf73c7653901056d3803e9bc0cce54b29a03afc7051ef600"
        ),
        "filters": {"candidate": "WCDR-2016", "control": "primary"},
        "entry_field": "entry_time",
        "side_field": "side",
        "comparison": ["2021-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
    },
    {
        "name": "wtsl_primary",
        "clock": Path(
            "data/wbtc_turnover_stablecoin_liquidity_2021_2023/"
            "wtsl168_support_clocks_2021_2023.csv.gz"
        ),
        "clock_sha256": (
            "df8cb085d439c9ee9e89334cb891b9e3b04f54c2a8e70bd4f552a90648ea8b6d"
        ),
        "header": WTSL_CLOCK_HEADER,
        "manifest": Path(
            "results/wbtc_turnover_stablecoin_liquidity_support_2026-07-23.json"
        ),
        "manifest_file_sha256": (
            "1415b8e2a40f2aff908bfec1d1faa9621445c3fe87b41c43fd95a991725b23bd"
        ),
        "manifest_hash": (
            "b53de47d743f7f61240e59ac3149c0a37467f6bb8ce580c9c3c2bc84341b7e9e"
        ),
        "filters": {
            "candidate": "WTSL-168-SOURCE-SEEN",
            "control": "primary",
        },
        "entry_field": "entry_time",
        "side_field": "side",
        "comparison": ["2021-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
    },
    {
        "name": "ugci_primary",
        "clock": Path("data/usdc_gross_clearing_imbalance_clocks_2021_2023.csv.gz"),
        "clock_sha256": (
            "a0f861c69ac171e1efa665dc90a916d0351413ca07e5e46783bb8abd662175fd"
        ),
        "header": UGCI_CLOCK_HEADER,
        "manifest": Path(
            "results/usdc_gross_clearing_imbalance_support_2026-07-22.json"
        ),
        "manifest_file_sha256": (
            "b61fc80bc879f15e9f1d15ac135ecbbda9384301cb8889def5b5a502af6068fa"
        ),
        "manifest_hash": (
            "bc6b79c83e176ebd110a39fdf126b86504c5c9ce9411df65a5fb46670170a8f4"
        ),
        "filters": {"candidate": "UGCI-288", "control": "primary"},
        "entry_field": "entry_time",
        "side_field": "side",
        "comparison": ["2021-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
    },
    {
        "name": "sealed_prior_stablecoin_bundle",
        "clock": Path(
            "results/ugci_prior_comparator_views_pre2024_2026-07-22.csv.gz"
        ),
        "clock_sha256": (
            "dfbf4808813c1b0db4c5a4f05af324473d3a92dfa5cdfc6581e1b07bc17271bd"
        ),
        "header": SEALED_PRIOR_HEADER,
        "manifest": Path(
            "results/ugci_prior_comparator_views_pre2024_manifest_2026-07-22.json"
        ),
        "manifest_file_sha256": (
            "38abf60a8c9aa44c7fb53a5435f22cb650151b58e33a6fca1ffae1aeb36ed5c2"
        ),
        "manifest_hash": (
            "a00301a229bc1c620f355cb42adc05b760d8734d7a903490ab2c1d3a0fd92d33"
        ),
        "filters": {},
        "entry_field": "entry_time",
        "side_field": None,
        "comparison": ["2021-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
    },
    {
        "name": "live_portfolio_pure_clocks",
        "clock": Path("results/cchr_live_portfolio_pure_clocks_2026-07-21.csv.gz"),
        "clock_sha256": (
            "73d6efbd35b3be64b0fa04fa9c8cb2db25866ef884f19b1ae673949e22a42b08"
        ),
        "header": LIVE_CLOCK_HEADER,
        "manifest": Path(
            "results/cchr_live_portfolio_pure_clock_manifest_2026-07-21.json"
        ),
        "manifest_file_sha256": (
            "6c53ae482cf72bba0f286a47626842bf43070276ff5fe359be718e44864af57d"
        ),
        "manifest_hash": (
            "2c240d4ffad4b8aa51434acebb74a61e8605fd0fa1a5e8e4554d3fa46fc1baea"
        ),
        "filters": {},
        "group_field": "candidate_id",
        "entry_field": "entry_time",
        "side_field": "side",
        "comparison": ["2021-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
    },
)

PRIOR_RESEARCH_DISCLOSURE = {
    "wbtc_and_stablecoin_source_values_opened": True,
    "prior_family_source_clocks_opened": True,
    "prior_family_candidates": ["WCDR-2016", "WTSL-168-SOURCE-SEEN"],
    "wscf_exact_incidence_opened": False,
    "wscf_market_outcomes_opened": False,
    "source_family_hypothesis_number": 3,
    "source_support_is_pristine_confirmation": False,
}

EXPECTED_OUTCOME_BOUNDARY = {
    "source_file_bytes_hashed_during_preregistration": True,
    "source_csv_headers_read_during_preregistration": 2,
    "source_value_rows_read_during_preregistration": 0,
    "exact_wscf_feature_rows_derived": 0,
    "exact_wscf_candidate_rows_derived": 0,
    "comparator_file_bytes_hashed_during_preregistration": True,
    "comparator_csv_headers_read_during_preregistration": 5,
    "comparator_value_rows_read_during_preregistration": 0,
    "btc_market_rows_read": 0,
    "funding_rows_read": 0,
    "future_return_rows_read": 0,
    "pnl_cagr_mdd_opened": False,
    "post_2023_contract_event_value_rows_read": 0,
    "network_calls": 0,
    "subprocess_calls": 0,
}


@dataclass(frozen=True)
class Config:
    output: str = str(DEFAULT_OUTPUT)


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    if str(path).startswith("~") or candidate.is_absolute() or ".." in candidate.parts:
        raise RuntimeError("WSCF path must be repository-relative")
    root = REPOSITORY_ROOT.resolve(strict=True)
    current = REPOSITORY_ROOT
    for part in candidate.parts:
        current /= part
        if current.is_symlink():
            raise RuntimeError("WSCF repository path contains a symlink")
        if not current.exists():
            break
    target = REPOSITORY_ROOT / candidate
    try:
        target.resolve(strict=False).relative_to(root)
    except ValueError as error:
        raise RuntimeError("WSCF path escapes the repository") from error
    return target


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: str | Path) -> str:
    return _sha256_path(_repository_path(path))


def _validate_regular_file(
    path: str | Path, expected_sha256: str, label: str
) -> str:
    target = _repository_path(path)
    if target.is_symlink() or not target.is_file():
        raise RuntimeError(f"{label} is missing or not a regular file")
    observed = _sha256_path(target)
    if observed != expected_sha256:
        raise RuntimeError(f"{label} SHA-256 drift")
    return observed


def _read_gzip_header(path: str | Path) -> tuple[str, ...]:
    target = _repository_path(path)
    with gzip.open(target, "rt", encoding="utf-8", newline="") as handle:
        try:
            return tuple(next(csv.reader(handle)))
        except StopIteration as error:
            raise RuntimeError("WSCF bound CSV is empty") from error


def _validate_manifest(
    path: str | Path,
    expected_file_sha256: str,
    expected_manifest_hash: str,
    label: str,
) -> None:
    _validate_regular_file(path, expected_file_sha256, label)
    payload = json.loads(_repository_path(path).read_text(encoding="utf-8"))
    if payload.get("manifest_hash") != expected_manifest_hash:
        raise RuntimeError(f"{label} canonical manifest drift")


def _source_binding(
    *,
    name: str,
    source: Path,
    source_sha256: str,
    manifest: Path,
    manifest_file_sha256: str,
    manifest_hash: str,
    builder: Path,
    builder_sha256: str,
    expected_header: Sequence[str],
    allowed_columns: Sequence[str],
) -> dict[str, Any]:
    _validate_regular_file(source, source_sha256, f"WSCF {name} source")
    header = _read_gzip_header(source)
    if header != tuple(expected_header):
        raise RuntimeError(f"WSCF {name} source header drift")
    if any(column not in header for column in allowed_columns):
        raise RuntimeError(f"WSCF {name} allowed-column drift")
    _validate_manifest(
        manifest,
        manifest_file_sha256,
        manifest_hash,
        f"WSCF {name} source manifest",
    )
    _validate_regular_file(builder, builder_sha256, f"WSCF {name} source builder")
    return {
        "source": str(source),
        "source_sha256": source_sha256,
        "manifest": str(manifest),
        "manifest_file_sha256": manifest_file_sha256,
        "manifest_hash": manifest_hash,
        "builder": str(builder),
        "builder_sha256": builder_sha256,
        "header": list(header),
        "allowed_columns": list(allowed_columns),
        "value_rows_read_during_preregistration": 0,
    }


def source_bindings() -> dict[str, Any]:
    return {
        "wbtc": _source_binding(
            name="WBTC",
            source=WBTC_SOURCE,
            source_sha256=WBTC_SOURCE_SHA256,
            manifest=WBTC_MANIFEST,
            manifest_file_sha256=WBTC_MANIFEST_FILE_SHA256,
            manifest_hash=WBTC_MANIFEST_HASH,
            builder=WBTC_BUILDER,
            builder_sha256=WBTC_BUILDER_SHA256,
            expected_header=WBTC_HEADER,
            allowed_columns=WBTC_ALLOWED_COLUMNS,
        ),
        "stablecoin": _source_binding(
            name="stablecoin",
            source=STABLECOIN_SOURCE,
            source_sha256=STABLECOIN_SOURCE_SHA256,
            manifest=STABLECOIN_MANIFEST,
            manifest_file_sha256=STABLECOIN_MANIFEST_FILE_SHA256,
            manifest_hash=STABLECOIN_MANIFEST_HASH,
            builder=STABLECOIN_BUILDER,
            builder_sha256=STABLECOIN_BUILDER_SHA256,
            expected_header=STABLECOIN_HEADER,
            allowed_columns=STABLECOIN_ALLOWED_COLUMNS,
        ),
    }


def comparator_bindings() -> list[dict[str, Any]]:
    bindings: list[dict[str, Any]] = []
    for spec in COMPARATOR_SPECS:
        clock = spec["clock"]
        _validate_regular_file(
            clock, spec["clock_sha256"], f"WSCF {spec['name']} comparator clock"
        )
        header = _read_gzip_header(clock)
        if header != tuple(spec["header"]):
            raise RuntimeError(f"WSCF {spec['name']} comparator header drift")
        _validate_manifest(
            spec["manifest"],
            spec["manifest_file_sha256"],
            spec["manifest_hash"],
            f"WSCF {spec['name']} comparator manifest",
        )
        bindings.append(
            {
                "name": spec["name"],
                "clock": str(clock),
                "clock_sha256": spec["clock_sha256"],
                "header": list(header),
                "manifest": str(spec["manifest"]),
                "manifest_file_sha256": spec["manifest_file_sha256"],
                "manifest_hash": spec["manifest_hash"],
                "filters": dict(spec["filters"]),
                "group_field": spec.get("group_field"),
                "entry_field": spec["entry_field"],
                "side_field": spec["side_field"],
                "comparison": list(spec["comparison"]),
                "value_rows_read_during_preregistration": 0,
            }
        )
    return bindings


def policy_payload() -> dict[str, Any]:
    return {
        "candidate": POLICY_ID,
        "singleton": True,
        "research_status": "source-family-seen_candidate-outcome-blind",
        "source_family_hypothesis_number": 3,
        "economic_hypothesis": {
            "long": (
                "positive finalized WBTC net batch followed within 12h by "
                "positive finalized stablecoin cumulative net"
            ),
            "short": (
                "negative finalized WBTC net batch followed within 12h by "
                "negative finalized stablecoin cumulative net"
            ),
            "interaction": "two-stage finalized cross-domain first passage",
            "component_controls_must_be_beaten": True,
        },
        "source_rows": {
            "wbtc": {
                "asset": "wbtc_eth",
                "events": ["mint", "burn"],
                "decimals": 8,
                "availability": "confirmation block N+64 available_at",
            },
            "stablecoin": {
                "assets": ["usdc_eth", "usdt_eth"],
                "directional_events": ["mint", "burn", "issue", "redeem"],
                "excluded_events": ["destroyed_black_funds", "deprecate"],
                "decimals": 6,
                "availability": "confirmation block N+64 available_at",
            },
        },
        "atomic_batches": {
            "grouping_key": "exact available_at",
            "same_available_at_rows_are_simultaneous": True,
            "intra_batch_transaction_or_log_order_forbidden": True,
            "wbtc_net": "sum(event_sign*amount_raw)",
            "wbtc_zero_net_action": "no anchor",
            "stablecoin_net": "sum(eligible event_sign*amount_raw)",
            "identity": "SHA-256(sorted canonical row identities)",
        },
        "primary_clock": {
            "anchor": "each nonzero WBTC atomic availability batch",
            "confirmation_interval": (
                "wbtc_available_at < stablecoin_available_at <= "
                "wbtc_available_at + 12 elapsed hours"
            ),
            "cumulative_initial_value": 0,
            "update_frequency": "once per stablecoin atomic availability batch",
            "confirmation": (
                "first stablecoin batch where sign(cumulative_net_raw) "
                "equals sign(wbtc_net_raw)"
            ),
            "signal_time": "confirming stablecoin batch available_at",
            "side": {"positive": "LONG", "negative": "SHORT"},
            "no_confirmation": "no candidate",
            "amount_or_ratio_threshold": None,
            "block_timestamp_forbidden": True,
        },
        "execution": {
            "entry_time": "ceil_to_5m(signal_time) + 5 elapsed minutes",
            "exact_grid_signal_still_waits_one_bar": True,
            "hold_elapsed_hours": 72,
            "hold_bars_5m": 864,
            "notional_exposure": 0.5,
            "reservation_interval": "[entry_time, exit_time)",
            "global_nonoverlap": True,
            "accept_when_entry_at_or_after_prior_exit": True,
            "raw_candidate_sort": [
                "entry_time",
                "signal_time",
                "wbtc_available_at",
                "wbtc_batch_identity",
                "stablecoin_batch_identity",
                "side",
            ],
            "accepted_confirmation_identity_reuse": False,
            "suppressed_candidate_queueing": False,
            "split_crossing_action": "skip",
            "stops_take_profit_or_trailing_exit": False,
        },
        "windows": {
            "warmup": ["2020-01-01T00:00:00Z", "2021-01-01T00:00:00Z"],
            "train": ["2021-01-01T00:00:00Z", "2023-01-01T00:00:00Z"],
            "selection": ["2023-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "sealed_from": "2024-01-01T00:00:00Z",
            "entry_and_exit_same_split_required": True,
        },
        "controls": {
            "direction_flip": "exact primary clock with both sides reversed",
            "deterministic_random_side": (
                "SHA-256 fixed side-count-matched exact-primary permutation"
            ),
            "wbtc_only_direct": "WBTC batch sign at WBTC availability",
            "stablecoin_only_12h_grid": (
                "UTC 00/12 grid; sign of stablecoin net over (D-12h,D]"
            ),
            "anchored_first_nonzero": (
                "WBTC timing anchor; first nonzero stablecoin cumulative sign"
            ),
            "opposite_confirmation": (
                "first stablecoin cumulative sign opposite WBTC within 12h"
            ),
            "lead_lag_reverse": (
                "same-sign aggregate stablecoin net in (WBTC-12h,WBTC]"
            ),
            "stale_wbtc_24h": "shift WBTC availability +24h before confirmation",
            "stale_wbtc_72h": "shift WBTC availability +72h before confirmation",
            "stablecoin_year_amount_permutation": (
                "permute amount within asset/event/year; preserve times/signs"
            ),
            "black_funds_veto": (
                "primary but confiscation availability vetoes anchor"
            ),
            "usdc_only_confirmation": "primary using USDC directional batches only",
            "usdt_only_confirmation": "primary using USDT directional batches only",
        },
        "source_support_gates": {
            "train_total_minimum": 50,
            "selection_total_minimum": 20,
            "each_train_year_minimum": 20,
            "each_train_half_year_minimum": 8,
            "each_selection_half_year_minimum": 6,
            "train_each_side_minimum": 12,
            "selection_each_side_minimum": 4,
            "maximum_month_share": 0.20,
            "maximum_quarter_share": 0.40,
            "maximum_consecutive_same_side": 10,
            "train_distinct_wbtc_actors_minimum": 10,
            "selection_distinct_wbtc_actors_minimum": 5,
            "maximum_calendar_gap_days": 90,
            "duplicate_accepted_wbtc_batch_allowed": False,
            "duplicate_accepted_confirmation_batch_allowed": False,
            "interpretation": "minimum_identifiability_not_pristine_confirmation",
            "failure_action": "reject candidate without outcomes or repair",
        },
        "novelty": {
            "comparators": [spec["name"] for spec in COMPARATOR_SPECS],
            "minimum_comparator_entries": 10,
            "direction_aware_metrics": ["same_side", "signless"],
            "timestamp_only_bundle_metrics": ["signless"],
            "near_window_elapsed_hours": 12,
            "maximum_exact_entry_jaccard": 0.10,
            "maximum_wscf_to_comparator_near_containment": 0.30,
            "reverse_containment": "report_only_due_to_density_asymmetry",
            "live_bundle_group_field": "candidate_id",
        },
        "economic_sequence": [
            "source-only support and novelty",
            "freeze strict evaluator",
            "train 2021-2022",
            "selection 2023 only after train pass",
            "immutable post-2023 source extension only after pre-2024 pass",
            "test 2024",
            "eval 2025",
            "recent 2026",
        ],
        "strict_economic_gates": {
            "absolute_return_positive": True,
            "cagr_to_strict_mdd_minimum": 3.0,
            "strict_mdd_pct_maximum": 15.0,
            "minimum_trades": 20,
            "minimum_trades_each_side": 4,
            "ten_bp_notional_side_stress_return_positive": True,
            "calendar_month_cluster_signflip_p_maximum": 0.10,
            "base_cost_notional_per_side": 0.0006,
            "stress_cost_notional_per_side": 0.0010,
            "realized_funding": True,
            "full_calendar_cagr": True,
            "strict_intratrade_high_water_mdd": True,
            "primary_cagr_mdd_above_component_controls": True,
            "flip_random_opposite_reverse_or_stale_full_qualification_rejects": True,
        },
        "post_2023_extension": {
            "same_contracts_topics_confirmation_and_batch_policy": True,
            "new_bridges_assets_stablecoins_or_address_labels_allowed": False,
            "wbtc_open_api_dates_allowed": False,
        },
        "rllm_boundary": {
            "authorized_before_deterministic_train_and_selection_pass": False,
            "later_actions": ["TRADE_FIXED_SIDE", "ABSTAIN"],
            "may_create_clock_reverse_side_or_change_hold": False,
            "reward_penalties": ["strict drawdown", "turnover"],
        },
        "stopping_rule": (
            "any identity, source, causality, support, novelty, train, or staged "
            "selection failure rejects WSCF-72-SOURCE-FAMILY-SEEN; repair "
            "requires a new identity frozen before access"
        ),
    }


def build_preregistration(cfg: Config | None = None) -> dict[str, Any]:
    frozen_cfg = Config() if cfg is None else cfg
    mechanism_sha = _validate_regular_file(
        MECHANISM_DECISION,
        MECHANISM_DECISION_SHA256,
        "WSCF mechanism decision",
    )
    policy = policy_payload()
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "candidate": POLICY_ID,
        "config": asdict(frozen_cfg),
        "mechanism_decision": {
            "path": str(MECHANISM_DECISION),
            "sha256": mechanism_sha,
        },
        "preregistration_source": {
            "path": str(SCRIPT_PATH),
            "sha256": sha256_file(SCRIPT_PATH),
        },
        "source_bindings": source_bindings(),
        "comparator_bindings": comparator_bindings(),
        "policy": policy,
        "policy_hash": canonical_hash(policy),
        "prior_research_disclosure": dict(PRIOR_RESEARCH_DISCLOSURE),
        "outcomes_opened": False,
        "exact_source_incidence_opened": False,
        "source_family_values_previously_opened": True,
        "performance_values_opened": False,
        "outcome_boundary": dict(EXPECTED_OUTCOME_BOUNDARY),
        "next_action": "build exact source-only WSCF support and novelty clocks",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate_preregistration(
    payload: Mapping[str, Any], *, verify_sources: bool = True
) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("WSCF preregistration canonical hash mismatch")
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("WSCF preregistration protocol drift")
    if payload.get("candidate") != POLICY_ID:
        raise RuntimeError("WSCF preregistration candidate drift")
    if payload.get("policy") != policy_payload():
        raise RuntimeError("WSCF preregistration policy drift")
    if payload.get("policy_hash") != canonical_hash(policy_payload()):
        raise RuntimeError("WSCF policy hash drift")
    if payload.get("outcomes_opened") is not False:
        raise RuntimeError("WSCF preregistration opened outcomes")
    if payload.get("exact_source_incidence_opened") is not False:
        raise RuntimeError("WSCF preregistration opened exact source incidence")
    if payload.get("source_family_values_previously_opened") is not True:
        raise RuntimeError("WSCF source-family disclosure drift")
    if payload.get("prior_research_disclosure") != PRIOR_RESEARCH_DISCLOSURE:
        raise RuntimeError("WSCF prior-research disclosure drift")
    if payload.get("outcome_boundary") != EXPECTED_OUTCOME_BOUNDARY:
        raise RuntimeError("WSCF preregistration outcome boundary drift")
    if verify_sources:
        config = payload.get("config")
        if not isinstance(config, Mapping):
            raise RuntimeError("WSCF preregistration config missing")
        expected = build_preregistration(Config(**dict(config)))
        if dict(payload) != expected:
            raise RuntimeError("WSCF preregistration binding drift")


def _protected_paths() -> set[Path]:
    protected = {
        _repository_path(SCRIPT_PATH),
        _repository_path(MECHANISM_DECISION),
        _repository_path(WBTC_SOURCE),
        _repository_path(WBTC_MANIFEST),
        _repository_path(WBTC_BUILDER),
        _repository_path(STABLECOIN_SOURCE),
        _repository_path(STABLECOIN_MANIFEST),
        _repository_path(STABLECOIN_BUILDER),
    }
    for spec in COMPARATOR_SPECS:
        protected.add(_repository_path(spec["clock"]))
        protected.add(_repository_path(spec["manifest"]))
    return protected


def write_preregistration(cfg: Config | None = None) -> tuple[dict[str, Any], str]:
    frozen_cfg = Config() if cfg is None else cfg
    output = _repository_path(frozen_cfg.output)
    if output.suffix != ".json":
        raise ValueError("WSCF preregistration output must be JSON")
    if output in _protected_paths():
        raise ValueError("WSCF preregistration output aliases a protected input")
    payload = build_preregistration(frozen_cfg)
    validate_preregistration(payload)
    if output.exists() or output.is_symlink():
        if output.is_symlink() or not output.is_file():
            raise FileExistsError("WSCF preregistration output is not a regular file")
        existing = json.loads(output.read_text(encoding="utf-8"))
        validate_preregistration(existing)
        if existing != payload:
            raise RuntimeError("refusing to overwrite WSCF preregistration")
        return payload, "verified_existing"
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=output.parent, prefix=f".{output.name}.", delete=False
    ) as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return payload, "created"


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return Config(**vars(parser.parse_args()))


def main() -> None:
    payload, status = write_preregistration(parse_args())
    print(
        json.dumps(
            {
                "candidate": payload["candidate"],
                "manifest_hash": payload["manifest_hash"],
                "outcomes_opened": payload["outcomes_opened"],
                "exact_source_incidence_opened": payload[
                    "exact_source_incidence_opened"
                ],
                "status": status,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
