"""Produce the frozen, outcome-blind ESDI-288 preregistration artifact."""

from __future__ import annotations

import argparse
import ast
import copy
from fractions import Fraction
import hashlib
import importlib.metadata as importlib_metadata
import json
import os
from pathlib import Path
import platform
import re
import secrets
import stat
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence


POLICY_ID = "ESDI-288"
PROTOCOL_VERSION = "ethereum_settlement_demand_impulse_preregistration_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DOCUMENT_PATH = Path(
    "docs/ethereum-settlement-demand-impulse-preregistration-2026-07-30.md"
)
DOCUMENT_SHA256 = (
    "83fc8b6d83a992e8ecb3077fb55829994073e9c7d3eeb783a8bfeb505e3462a8"
)
DEFAULT_OUTPUT = Path(
    "results/ethereum_settlement_demand_impulse_preregistration_2026-07-30.json"
)
PRODUCER_PATH = Path(
    "training/preregister_ethereum_settlement_demand_impulse.py"
)
TEST_PATH = Path(
    "tests/test_preregister_ethereum_settlement_demand_impulse.py"
)
COMMITTED_PREREGISTRATION_PATHS = (
    DOCUMENT_PATH,
    PRODUCER_PATH,
    TEST_PATH,
)
RUNTIME_CODE_ROOTS = (
    Path("execution/portfolio_live.py"),
    Path("execution/rank7_runtime.py"),
    Path("execution/rex_llm_live.py"),
)
RUNTIME_CODE_CLOSURE_PATHS = (
    Path("execution/__init__.py"),
    Path("execution/binance_aggtrade_stream.py"),
    Path("execution/portfolio_live.py"),
    Path("execution/portfolio_shadow_policies.py"),
    Path("execution/rank7_runtime.py"),
    Path("execution/rex_llm_live.py"),
    Path("execution/wave_execution.py"),
    Path("models/__init__.py"),
    Path("models/option_a.py"),
    Path("models/option_b_vlm.py"),
    Path("preprocessing/__init__.py"),
    Path("preprocessing/binance_aux_features.py"),
    Path("preprocessing/chart_generator.py"),
    Path("preprocessing/external_features.py"),
    Path("preprocessing/indicators.py"),
    Path("preprocessing/live_db_features.py"),
    Path("preprocessing/market_features.py"),
    Path("preprocessing/scalars.py"),
    Path("preprocessing/timeframe.py"),
    Path("training/__init__.py"),
    Path("training/alpha_feature_backtest.py"),
    Path("training/alpha_linear_combo_scan.py"),
    Path("training/audit_confirmed_pullback_squeeze_live_parity.py"),
    Path("training/audit_weak_feature_responsibility_stability.py"),
    Path("training/build_rex_event_reasoning_policy_data.py"),
    Path("training/build_rex_regime_thesis_sft.py"),
    Path("training/eval_pairwise_candidate_backtest.py"),
    Path("training/eval_text_label.py"),
    Path("training/eval_text_trader.py"),
    Path("training/evaluate_oi_llm_selector.py"),
    Path("training/evaluate_portfolio_llm_selector.py"),
    Path("training/event_candidate_pool_probe.py"),
    Path("training/hierarchical_direct_split_search.py"),
    Path("training/hierarchical_regime_filter_search.py"),
    Path("training/long_component_tp_union_scan.py"),
    Path("training/long_regime_alpha_union_validate.py"),
    Path("training/long_regime_combo_scan.py"),
    Path("training/long_regime_gate_scan.py"),
    Path("training/long_regime_interest_gate_validation.py"),
    Path("training/long_regime_score_gate_validation.py"),
    Path("training/online_risk_overlay_backtest.py"),
    Path("training/path_outcome_dataset.py"),
    Path("training/portfolio_opt_new_alpha_pool.py"),
    Path("training/search_bidirectional_state_alpha.py"),
    Path("training/search_bocpd_state_gated_alpha.py"),
    Path("training/search_causal_online_expert_alpha.py"),
    Path("training/search_confirmed_pullback_squeeze_alpha.py"),
    Path("training/search_crossvenue_microstructure_consensus_alpha.py"),
    Path("training/search_deribit_dvol_alpha.py"),
    Path("training/search_funding_premium_external_state_gate_alpha.py"),
    Path("training/search_funding_premium_independent_gate_alpha.py"),
    Path("training/search_gaussian_hmm_regime_alpha.py"),
    Path("training/search_inventory_purge_reclaim_alpha.py"),
    Path("training/search_jump_variation_bidirectional_alpha.py"),
    Path("training/search_kalman_state_gated_alpha.py"),
    Path("training/search_kimchi_leadlag_bidirectional_alpha.py"),
    Path("training/search_liquidity_recovery_bidirectional_alpha.py"),
    Path("training/search_liveparity_state_feature_interactions.py"),
    Path("training/search_market_braid_alpha.py"),
    Path("training/search_nested_barrier_witness_alpha.py"),
    Path("training/search_orderflow_trophic_succession_alpha.py"),
    Path("training/search_positioning_disagreement_alpha.py"),
    Path("training/search_positioning_hgb_path_alpha.py"),
    Path("training/search_semimarkov_duration_alpha.py"),
    Path("training/search_significant_cagr_mdd_pool.py"),
    Path("training/search_specific_pullback_squeeze_alpha.py"),
    Path("training/search_spot_perp_transfer_entropy_alpha.py"),
    Path("training/search_volume_clock_bidirectional_alpha.py"),
    Path("training/strict_bar_backtest.py"),
    Path("training/train_text_sft.py"),
    Path("training/wave_feature_ridge_policy.py"),
    Path("utils.py"),
    Path("pyproject.toml"),
    Path("uv.lock"),
)
FROZEN_RUNTIME_ENVIRONMENT = {
    "python": {
        "implementation": "CPython",
        "version": [3, 10, 10],
    },
    "platform": {
        "system": "Linux",
        "machine": "x86_64",
        "libc": ["glibc", "2.39"],
    },
    "packages": {
        "datasets": "4.6.1",
        "numpy": "2.2.6",
        "pandas": "2.3.3",
        "peft": "0.18.1",
        "scikit-learn": "1.7.2",
        "scipy": "1.15.3",
        "SQLAlchemy": None,
        "torch": "2.9.0",
        "transformers": "5.7.0.dev0",
        "trl": "0.29.0",
        "websockets": "15.0.1",
    },
}
FROZEN_DISTRIBUTION_INVENTORY_COUNT = 108
FROZEN_DISTRIBUTION_INVENTORY_SHA256 = (
    "a5b435e485426d7254ed222692bf3b9c6444ae992e582084398dc57b960549dc"
)

EPOCH_SIZE_BLOCKS = 3_600
FIRST_EPOCH_ID = 4_531
LAST_EPOCH_ID = 7_004
CONFIRMATION_BLOCKS = 64
RANK_LOOKBACK = 180

GROSS9_WEIGHTS = {
    "cand_rex_veto_7": 1.6,
    "fresh_kimchi_fx": 2.0,
    "frozen_annual_rank7": 3.0,
    "markov_transition_long": 2.0,
    "rex_taker_low_range_position": 0.4,
}


def _directional_clock(
    *,
    path: str,
    sha256: str,
    header_line_sha256: str,
    filters: Mapping[str, str],
    entry_column: str,
    exit_column: str,
    side_column: str,
    group_column: str | None = None,
    groups: Sequence[str] = (),
) -> dict[str, Any]:
    required_columns = {
        *filters,
        entry_column,
        exit_column,
        side_column,
    }
    if group_column is not None:
        required_columns.add(group_column)
    return {
        "path": path,
        "sha256": sha256,
        "header_line_sha256": header_line_sha256,
        "filters": dict(filters),
        "group_column": group_column,
        "groups": list(groups),
        "capability": "directional_interval",
        "entry_column": entry_column,
        "exit_column": exit_column,
        "side_column": side_column,
        "required_columns": sorted(required_columns),
        "required_metrics": [
            "exact_entry_jaccard",
            "candidate_24h_containment",
            "absolute_signed_exposure_pearson",
        ],
    }


FROZEN_COMPARATOR_ARTIFACTS = {
    "CAIM": {
        **_directional_clock(
            path=(
                "results/chain_activity_impulse_momentum_"
                "pre2024_comparator_clock_2026-07-21.csv.gz"
            ),
            sha256=(
                "e50cc154e23950a381aa456180970140"
                "882083734128bd7f902257738633f320"
            ),
            header_line_sha256=(
                "af12feb9d921c9f8c9f1b7def3b279a"
                "4e5c1faf9918a8f67e2bddc202a72ee8d"
            ),
            filters={},
            entry_column="entry_time",
            exit_column="exit_time",
            side_column="side",
        ),
    },
    "WCTR-288": {
        **_directional_clock(
            path=(
                "results/witness_composition_transport_"
                "primary_clock_2026-07-20.csv.gz"
            ),
            sha256=(
                "7a6b56a3024d0d087322fad7b3229276"
                "c539b93374691cd2812af0630dc752b1"
            ),
            header_line_sha256=(
                "cd90df97c515162cdd0d2fbca7f341ef"
                "65a61367168ff6bcc9d8d6e93eb3b6cb"
            ),
            filters={"clock": "primary"},
            entry_column="entry_time_utc",
            exit_column="exit_time_utc",
            side_column="side",
        ),
    },
    "BFWC-288": {
        **_directional_clock(
            path=(
                "results/blockspace_fee_witness_concordance_"
                "primary_clock_2026-07-30.csv.gz"
            ),
            sha256=(
                "b125046a1a3defda960e51b42e03ee1c"
                "3bb72a0799c646d3ae16a3e692735ed1"
            ),
            header_line_sha256=(
                "dba03137f60eb5480b5b2f43042e55d1"
                "e59bcc2425de87ef8df980e4b64c77e5"
            ),
            filters={"control": "primary"},
            entry_column="entry_time_utc",
            exit_column="exit_time_utc",
            side_column="side",
        ),
    },
    "BFRT-288": {
        **_directional_clock(
            path="results/block_feerate_breadth_transport_primary_clock_2026-07-20.csv.gz",
            sha256=(
                "33428d29c2ace9b23672b2dc9dc3e9ba"
                "0e3020fa1a6e3845d55fa5d75230d64a"
            ),
            header_line_sha256=(
                "34f459439a8db95467cbbceca6ba1425"
                "4bee97e8eca53a0840da2b68d1167c05"
            ),
            filters={"clock": "primary"},
            entry_column="entry_time_utc",
            exit_column="exit_time_utc",
            side_column="side",
        ),
    },
    "CDLTR-72A": {
        **_directional_clock(
            path=(
                "results/cross_domain_liquidity_transmission_"
                "relay_support_clock_2026-07-21.csv.gz"
            ),
            sha256=(
                "aa2bcafd0f62ebe585f93cbd357d29c3"
                "7ae526a95a90b8a6c0bd7c068cd6e5a1"
            ),
            header_line_sha256=(
                "d4b4ecf5de4c46cc2e75c41e3a35097"
                "d314116b2bf3f0e4565410b509b75db09"
            ),
            filters={"clock": "primary"},
            entry_column="entry_time_utc",
            exit_column="exit_time_utc",
            side_column="side",
        ),
    },
    "CDLTR-prior-chain-network-bundle": {
        "path": "results/cdltr_prior_comparator_views_2026-07-21.csv.gz",
        "sha256": (
            "bffdcf158d7d4e38db5794fb4761de52"
            "8fb73b0b772ae950f3a087a93ab63f1a"
        ),
        "header_line_sha256": (
            "8b05d5e578b01d9040ffecd3ddd55bc9"
            "97fac46d0ea76bbf30b22ffe3a7b654b"
        ),
        "filters": {},
        "group_column": "comparator",
        "capability_column": "capability",
        "entry_column": "entry_time",
        "exit_column": "exit_time",
        "side_column": "side",
        "required_columns": [
            "capability",
            "comparator",
            "entry_time",
            "exit_time",
            "side",
        ],
        "directional_interval_groups": [
            "CVTR-1",
            "DFFB-601",
            "FLCC-1:FLCC-H4-Q60",
            "FLCC-1:FLCC-H4-Q65",
            "FLCC-1:FLCC-H8-Q60",
            "FLCC-1:FLCC-H8-Q65",
            "NTB-7",
            "NWE-8",
            "ORFR-1",
            "chain_activity_impulse_momentum",
        ],
        "timestamp_only_groups": [
            "NWE-7",
            "live_anchor_2023",
            "prior_microstructure:cbfr72",
            "prior_microstructure:mfic_fast",
            "prior_microstructure:mfic_slow",
            "prior_microstructure:mfic_union",
            "prior_microstructure:netf_fast",
            "prior_microstructure:netf_slow",
            "prior_microstructure:netf_union",
            "prior_microstructure:terminal_absorption_wait72_h72",
            "prior_microstructure:wfrs_l288_q90_h144",
        ],
        "required_metrics_by_capability": {
            "directional_interval": [
                "exact_entry_jaccard",
                "candidate_24h_containment",
                "absolute_signed_exposure_pearson",
            ],
            "timestamp_only": [
                "exact_entry_jaccard",
                "candidate_24h_containment",
            ],
        },
        "each_group_is_a_separate_comparator": True,
    },
    "AMTR-48": {
        **_directional_clock(
            path="data/authorized_minter_turnaround_relay_clocks_2020_2023.csv.gz",
            sha256=(
                "30875029daa4d6e2eff9a59f53d45eda5"
                "7dbced05988df089c38a6c81abfa0f6"
            ),
            header_line_sha256=(
                "423287fbc7a50bd00c0ca1de8580c983d"
                "f1a2d128c1cc497d68e1bc74c224ac8"
            ),
            filters={"candidate": "AMTR-48", "control": "primary"},
            entry_column="entry_time",
            exit_column="scheduled_exit",
            side_column="side",
        ),
    },
    "EBLR-60/30": {
        **_directional_clock(
            path="data/eth_btc_liquidation_relay_clocks_2023_2024.csv.gz",
            sha256=(
                "b4b35a0e9ae0cf26bf08df67b5c2fc83"
                "2393c638c97f5b91a86894ee693b430e"
            ),
            header_line_sha256=(
                "17ad2bd574a60355d40fa797e4a4a4b5"
                "7e1f0cfc48e6bf79ca9e8312d1f6da11"
            ),
            filters={"candidate": "EBLR-60/30"},
            entry_column="entry_time",
            exit_column="planned_exit_time",
            side_column="direction",
        ),
    },
    "UGCI-288": {
        **_directional_clock(
            path="data/usdc_gross_clearing_imbalance_clocks_2021_2023.csv.gz",
            sha256=(
                "a0f861c69ac171e1efa665dc90a916d0"
                "351413ca07e5e46783bb8abd662175fd"
            ),
            header_line_sha256=(
                "b79639e44ce1b4488fdf6991e6083122"
                "1cbc9a48565fa42d053faeb71156ad91"
            ),
            filters={"candidate": "UGCI-288", "control": "primary"},
            entry_column="entry_time",
            exit_column="exit_time",
            side_column="side",
        ),
    },
    "WCDR-2016": {
        **_directional_clock(
            path=(
                "data/wrapped_collateral_dollar_liquidity_rotation_2021_2023/"
                "wcdr2016_support_clocks_2021_2023.csv.gz"
            ),
            sha256=(
                "241d96a64a654ba2faeda2d4a8460131"
                "269acf21d0bbbf31177d35d1ecd63b3c"
            ),
            header_line_sha256=(
                "e67cd52d0cadded15fd49f4ed809707e5"
                "d1601260416a93949f452dd7638680e"
            ),
            filters={"candidate": "WCDR-2016", "control": "primary"},
            entry_column="entry_time",
            exit_column="exit_time",
            side_column="side",
        ),
    },
    "WTSL-168-SOURCE-SEEN": {
        **_directional_clock(
            path=(
                "data/wbtc_turnover_stablecoin_liquidity_2021_2023/"
                "wtsl168_support_clocks_2021_2023.csv.gz"
            ),
            sha256=(
                "df8cb085d439c9ee9e89334cb891b9e3"
                "b04f54c2a8e70bd4f552a90648ea8b6d"
            ),
            header_line_sha256=(
                "f206f15f5410c3bb568df4f64c0cffaf"
                "cf077b5ef08dc8c427ac3af33d873937"
            ),
            filters={"candidate": "WTSL-168-SOURCE-SEEN", "control": "primary"},
            entry_column="entry_time",
            exit_column="exit_time",
            side_column="side",
        ),
    },
    "WSCF-72-SOURCE-FAMILY-SEEN": {
        **_directional_clock(
            path=(
                "data/wbtc_stablecoin_finalized_confirmation_relay_2021_2023/"
                "wscf72_support_clocks_2021_2023.csv.gz"
            ),
            sha256=(
                "86565774ae97a1024c5a66b4d59a1f54"
                "13bf4608398623359dd3ee24572f0ef3"
            ),
            header_line_sha256=(
                "adb55cd822efbdcd8469018a51c2b0375"
                "14758633599a403fae1a1868ef2e9f3"
            ),
            filters={
                "candidate": "WSCF-72-SOURCE-FAMILY-SEEN",
                "control": "primary",
            },
            entry_column="entry_time",
            exit_column="exit_time",
            side_column="side",
        ),
    },
    "FCCM-72": {
        **_directional_clock(
            path=(
                "data/funding_currency_custody_mobility_consensus_2021_2023/"
                "fccm72_support_clocks_2021_2023.csv.gz"
            ),
            sha256=(
                "71180862d9dcc4d76e055c52fd72a2424"
                "ee12387a6b8062af8a9382675af3810"
            ),
            header_line_sha256=(
                "ffec7a169e71d896d348e875e4753c880"
                "050c8011b52eb058eee6932a5d4a6d5"
            ),
            filters={"candidate": "FCCM-72", "control": "primary"},
            entry_column="entry_time",
            exit_column="exit_time",
            side_column="side",
        ),
    },
    "URCD-72": {
        **_directional_clock(
            path=(
                "data/usdc_recipient_concentration_dislocation_2021_2023/"
                "urcd72_support_clocks_2021_2023.csv.gz"
            ),
            sha256=(
                "ad9617ec5af0c0189aa384a49ab9244e"
                "957758f7c8abe71b6b61e911b7663ea1"
            ),
            header_line_sha256=(
                "3f0faa3e408dc1343104af9805b0d3561"
                "06b043d7e5eb6c18ee646ac07d2dc65"
            ),
            filters={"candidate": "URCD-72", "control": "primary"},
            entry_column="entry_time",
            exit_column="exit_time",
            side_column="side",
        ),
    },
    "SQFD-6": {
        **_directional_clock(
            path="data/stablecoin_quote_flow_diffusion_clocks_2023_2026.csv.gz",
            sha256=(
                "a81e144eea1e80ae5439fc66db1fad5b"
                "bd00cd9ac177e25142b5cfb5a07bcc5b"
            ),
            header_line_sha256=(
                "2e6d34c734ddc66d15c7718cc0aed3f2"
                "c8903fc02370bd9a2446054ff96a2071"
            ),
            filters={"candidate": "SQFD-6", "control": "primary"},
            entry_column="entry_time",
            exit_column="exit_time",
            side_column="side",
        ),
    },
    "SDDR-12": {
        **_directional_clock(
            path="data/stablecoin_denominator_dislocation_clocks_2023.csv.gz",
            sha256=(
                "eaf2d6c187af9855e76474d2951fcdc1"
                "2267174980a72649b73d068982ca8c69"
            ),
            header_line_sha256=(
                "91e4b4187dccbba5c9a6407316c4205d"
                "17422b1900b319a7ef800a541e1f3550"
            ),
            filters={"candidate": "SDDR-12", "control": "primary"},
            entry_column="entry_time",
            exit_column="exit_time",
            side_column="side",
        ),
    },
    "UCBR-12": {
        **_directional_clock(
            path="data/usdt_collateral_breadth_relay_clocks_2023.csv.gz",
            sha256=(
                "20b3ee9f82696222a3adbde0045dfde53"
                "e0e240e85162e463166aa8fe90b1a8f"
            ),
            header_line_sha256=(
                "a66cd7a33793d7d0b1056171526dd67c"
                "9de5cb95b8847435a8ad1c220757ef10"
            ),
            filters={"candidate": "UCBR-12", "control": "primary"},
            entry_column="entry_time",
            exit_column="exit_time",
            side_column="side",
        ),
    },
    "BFMWD-primary-variants": {
        **_directional_clock(
            path="data/bitfinex_margin_warehouse_deployment_clocks_2021_2023.csv.gz",
            sha256=(
                "02b4fcc462a5a48be7673649f4cf4b2f"
                "9bb210baca4294eed1696d479820cccc"
            ),
            header_line_sha256=(
                "2dbaad2b5a57b577733b5f0a1807514"
                "ef41f1fc9f9a3daf7028152338568c7f8"
            ),
            filters={"control": "primary"},
            entry_column="entry_time",
            exit_column="exit_time",
            side_column="side",
            group_column="variant_id",
            groups=(
                "bfmwd_w12_d3_z10_h12",
                "bfmwd_w24_d3_z10_h12",
                "bfmwd_w12_d6_z10_h12",
                "bfmwd_w24_d6_z10_h12",
            ),
        ),
        "each_group_is_a_separate_comparator": True,
    },
}

FROZEN_COMPARATOR_DOMAINS = {
    "CAIM": ["2023-06-01T00:00:00Z", "2024-01-01T00:00:00Z"],
    "WCTR-288": ["2023-06-01T00:00:00Z", "2026-06-01T00:00:00Z"],
    "BFWC-288": ["2023-06-01T00:00:00Z", "2026-06-01T00:00:00Z"],
    "BFRT-288": ["2023-06-01T00:00:00Z", "2026-06-01T00:00:00Z"],
    "CDLTR-72A": ["2023-06-01T00:00:00Z", "2026-06-01T00:00:00Z"],
    "CDLTR-prior-chain-network-bundle": [
        "2023-06-01T00:00:00Z",
        "2024-01-01T00:00:00Z",
    ],
    "AMTR-48": ["2023-06-01T00:00:00Z", "2024-01-01T00:00:00Z"],
    "EBLR-60/30": ["2023-06-25T00:00:00Z", "2024-10-15T00:00:00Z"],
    "UGCI-288": ["2023-06-01T00:00:00Z", "2024-01-01T00:00:00Z"],
    "WCDR-2016": ["2023-06-01T00:00:00Z", "2024-01-01T00:00:00Z"],
    "WTSL-168-SOURCE-SEEN": [
        "2023-06-01T00:00:00Z",
        "2024-01-01T00:00:00Z",
    ],
    "WSCF-72-SOURCE-FAMILY-SEEN": [
        "2023-06-01T00:00:00Z",
        "2024-01-01T00:00:00Z",
    ],
    "FCCM-72": ["2023-06-01T00:00:00Z", "2024-01-01T00:00:00Z"],
    "URCD-72": ["2023-06-01T00:00:00Z", "2024-01-01T00:00:00Z"],
    "SQFD-6": ["2023-09-01T00:00:00Z", "2024-01-01T00:00:00Z"],
    "SDDR-12": ["2023-09-01T00:00:00Z", "2024-01-01T00:00:00Z"],
    "UCBR-12": ["2023-09-01T00:00:00Z", "2024-01-01T00:00:00Z"],
    "BFMWD-primary-variants": [
        "2023-06-01T00:00:00Z",
        "2024-01-01T00:00:00Z",
    ],
}


def frozen_comparator_registry() -> dict[str, dict[str, Any]]:
    if set(FROZEN_COMPARATOR_ARTIFACTS) != set(FROZEN_COMPARATOR_DOMAINS):
        raise RuntimeError("ESDI-288 comparator registry/domain set differs")
    registry = copy.deepcopy(FROZEN_COMPARATOR_ARTIFACTS)
    for name, domain in FROZEN_COMPARATOR_DOMAINS.items():
        registry[name]["comparison_domain"] = list(domain)
    return registry


GROSS9_AUTHORITY = {
    "portfolio": {
        "path": (
            "configs/shadow/portfolio_rank7_"
            "capacity_candidate_2026-07-28.json"
        ),
        "sha256": (
            "006f82e1f0affad9f96a08a6c600542f"
            "eec4a0e1198ed99b8630627de4913450"
        ),
    },
    "base_portfolio": {
        "path": "configs/live/portfolio_added_alpha_mainnet_live_2026-07-18.json",
        "sha256": (
            "3f6c929f6b03797093b8b81f50ede533"
            "176aa169f5f81a4bb5f616d31afd24ff"
        ),
    },
    "runtime": {
        "portfolio_live.py": {
            "path": "execution/portfolio_live.py",
            "sha256": (
                "5edd4e9aa749e538d7de6a9990e31b94"
                "fbcb444b7e1498714cea82036962863d"
            ),
        },
        "rank7_runtime.py": {
            "path": "execution/rank7_runtime.py",
            "sha256": (
                "1ba1ab8f0af7cee0bac4885836776d50"
                "f2aff9dd30319d47e9a322f82f36c0dc"
            ),
        },
        "rex_llm_live.py": {
            "path": "execution/rex_llm_live.py",
            "sha256": (
                "2e0de376e967b237afb711dd44503ec45"
                "dbb9b6548f575219c1cf93cc2de9c48"
            ),
        },
    },
    "runtime_code_closure": {
        "roots": [str(path) for path in RUNTIME_CODE_ROOTS],
        "paths": [str(path) for path in RUNTIME_CODE_CLOSURE_PATHS],
        "bound_by_git_blob_and_sha256_in_repository_identity": True,
        "ast_import_closure_must_match_before_artifact_creation": True,
        "environment_lock_paths": ["pyproject.toml", "uv.lock"],
        "required_runtime_abi_and_selected_packages": copy.deepcopy(
            FROZEN_RUNTIME_ENVIRONMENT
        ),
        "all_distribution_inventory_count": FROZEN_DISTRIBUTION_INVENTORY_COUNT,
        "all_distribution_inventory_sha256": (
            FROZEN_DISTRIBUTION_INVENTORY_SHA256
        ),
        "runtime_environment_must_match_before_artifact_creation": True,
    },
    "transitive_source_manifest": {
        "path": (
            "configs/shadow/portfolio_added_alpha_"
            "signal_parity_sources_2026-07-16.json"
        ),
        "sha256": (
            "27a5095b18acaf10c9f5aa68c2ddac1"
            "ab1ebe4f506828e1fcfec34c414eb3ba6"
        ),
        "validate_every_declared_source_hash_at_novelty_stage": True,
    },
    "pre2025_anchor": {
        "path": "results/gross9_pre2025_authoritative_anchor_2026-07-28.json",
        "sha256": (
            "329878d90b6cd9c731eb4871ac041256"
            "f95f03c14dd261ada681d3a370709875"
        ),
        "metadata_only_until_economic_stage": True,
    },
    "sleeves": {
        "cand_rex_veto_7": {
            "config": {
                "path": "configs/live/rex_veto_7_candidate.json",
                "sha256": (
                    "36df47c4737eb99f4ca5e2b257d9bd2f"
                    "bf130df9d731b9ac02fcfe5192acd4db"
                ),
            },
            "side": "AUTO from exact REX decision",
            "clock_artifact_preexists": False,
        },
        "fresh_kimchi_fx": {
            "config": {
                "path": "configs/shadow/fresh_kimchi_fx_2026-07-16.json",
                "sha256": (
                    "f3e764d5d065643905105ae1c46668a2"
                    "2684569289c3781b79fc6b2efcc5154f"
                ),
            },
            "side": "AUTO from exact exclusive long/short gates",
            "clock_artifact_preexists": False,
        },
        "frozen_annual_rank7": {
            "config": {
                "path": (
                    "configs/shadow/frozen_annual_rank7_2026-07-16.json"
                ),
                "sha256": (
                    "b75621bb604266d1cd2529a29f8bdb6a"
                    "ec3b1f2c14ff00d88673ef007362526d"
                ),
            },
            "bundle_manifest": {
                "path": "artifacts/rank7/frozen_annual_rank7_2026/manifest.json",
                "sha256": (
                    "2c45484dce48658ef7d342df7a3bb8e8"
                    "3cd0f31d4728bbb72fd38e612ec3b7a9"
                ),
            },
            "side": "LONG",
            "clock_artifact_preexists": False,
            "immutable_model_bundle_preexists": True,
        },
        "markov_transition_long": {
            "config": {
                "path": (
                    "configs/shadow/markov_transition_long_2026-07-16.json"
                ),
                "sha256": (
                    "ebfec66715428b2fffead13e17229fb43"
                    "69816daeeeab2c02cf0115e7110b755"
                ),
            },
            "side": "LONG",
            "clock_artifact_preexists": False,
        },
        "rex_taker_low_range_position": {
            "config": {
                "path": (
                    "configs/shadow/"
                    "rex_taker_low_range_position_2026-07-16.json"
                ),
                "sha256": (
                    "d4c56a6f1659189876c1d3f2e519a3d"
                    "bc2608c754720c5cd1f65a02adb5589e4"
                ),
            },
            "side": "AUTO from exact REX decision",
            "clock_artifact_preexists": False,
        },
    },
    "clock_reconstruction": {
        "stage": "after ESDI source-support pass and before ESDI economics",
        "five_signed_sleeves_required": True,
        "exact_runtime_config_and_transitive_hash_validation_required": True,
        "failure_or_missing_dependency_is_terminal": True,
    },
}


def frozen_gross9_authority() -> dict[str, Any]:
    validate_runtime_environment()
    authority = copy.deepcopy(GROSS9_AUTHORITY)
    authority["runtime_code_closure"][
        "exact_runtime_environment"
    ] = current_runtime_environment()
    return authority


BOUNDARIES = [
    {
        "utc": "2023-01-01T00:00:00Z",
        "first_block_at_or_after": 16_308_190,
        "hash": "0x53dd35d982c984441b3b613919d64dbbf131063d0f85804d77f93f190fa5e106",
    },
    {
        "utc": "2023-06-01T00:00:00Z",
        "first_block_at_or_after": 17_382_266,
        "hash": "0xe0ef11cab4909c80599087b4ffb0bf1e92b1affcc72abc3b802f20a9d5d21096",
    },
    {
        "utc": "2025-01-01T00:00:00Z",
        "first_block_at_or_after": 21_525_891,
        "hash": "0x9512042c5c38145528389a91bd3d63193a1f48fb45d6a3b144ad2d833331fc4c",
    },
    {
        "utc": "2026-01-01T00:00:00Z",
        "first_block_at_or_after": 24_136_053,
        "hash": "0x53e1c0caa885383824d39dc57c0692ea20e971ade409553c4a8031e90f44c516",
    },
    {
        "utc": "2026-06-01T00:00:00Z",
        "first_block_at_or_after": 25_218_798,
        "hash": "0x55f8fdbda40a23cd51a9a2bffba625317ed15d9d1cdc2128c7643bf66e2a906e",
    },
]

EVIDENCE_BOUNDARIES = (
    "full_source_replay_opened",
    "exact_incidence_opened",
    "candidate_overlap_opened",
    "btc_market_rows_opened",
    "funding_rows_opened",
    "gross9_rows_opened",
    "outcomes_opened",
)


def epoch_blocks(epoch_id: int) -> tuple[int, int, int]:
    """Return inclusive start/end and the N+64 confirmation block."""

    if isinstance(epoch_id, bool) or not isinstance(epoch_id, int):
        raise TypeError("ESDI-288 epoch ID must be an integer")
    start = EPOCH_SIZE_BLOCKS * epoch_id
    end = EPOCH_SIZE_BLOCKS * (epoch_id + 1) - 1
    return start, end, end + CONFIRMATION_BLOCKS


def median2(values: Sequence[int]) -> int:
    """Return exactly twice the median of one frozen 3,600-block epoch."""

    if len(values) != EPOCH_SIZE_BLOCKS:
        raise ValueError("ESDI-288 median2 requires exactly 3,600 values")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in values
    ):
        raise ValueError("ESDI-288 base fees must be positive integers")
    ordered = sorted(values)
    return ordered[1_799] + ordered[1_800]


def _rational(value: tuple[int, int]) -> tuple[int, int]:
    numerator, denominator = value
    if (
        isinstance(numerator, bool)
        or isinstance(denominator, bool)
        or not isinstance(numerator, int)
        or not isinstance(denominator, int)
        or numerator < 0
        or denominator <= 0
    ):
        raise ValueError(
            "ESDI-288 ratios require nonnegative integer numerators "
            "and positive denominators"
        )
    return numerator, denominator


def compare_rationals(left: tuple[int, int], right: tuple[int, int]) -> int:
    """Compare nonnegative rationals using integer cross multiplication."""

    left_num, left_den = _rational(left)
    right_num, right_den = _rational(right)
    difference = left_num * right_den - right_num * left_den
    return (difference > 0) - (difference < 0)


def exact_rational_midrank(
    current: tuple[int, int],
    prior: Iterable[tuple[int, int]],
) -> Fraction:
    """Exact midrank of current against exactly 180 strictly-prior ratios."""

    current = _rational(current)
    history = [_rational(value) for value in prior]
    if len(history) != RANK_LOOKBACK:
        raise ValueError("ESDI-288 midrank requires exactly 180 prior ratios")
    lower = sum(compare_rationals(value, current) < 0 for value in history)
    equal = sum(compare_rationals(value, current) == 0 for value in history)
    return Fraction(2 * lower + equal, 2 * RANK_LOOKBACK)


def ceil_5m_plus_one_bar(epoch_seconds: int) -> int:
    """Ceil integer Unix seconds to five minutes, then wait one full bar."""

    if isinstance(epoch_seconds, bool) or not isinstance(epoch_seconds, int):
        raise TypeError("ESDI-288 Unix seconds must be an integer")
    if epoch_seconds < 0:
        raise ValueError("ESDI-288 Unix seconds must be nonnegative")
    return ((epoch_seconds + 299) // 300) * 300 + 300


def canonical_signal_id(epoch_id: int) -> str:
    """Return the only valid same-parent ESDI primary signal identity."""

    if isinstance(epoch_id, bool) or not isinstance(epoch_id, int):
        raise TypeError("ESDI-288 signal epoch ID must be an integer")
    if not FIRST_EPOCH_ID <= epoch_id <= LAST_EPOCH_ID:
        raise ValueError("ESDI-288 signal epoch ID is outside the frozen source")
    return f"{POLICY_ID}|primary|epoch_id={epoch_id}"


def deterministic_random_side(epoch_id: int) -> str:
    """Return the SHA-256-fixed same-parent random-side control."""

    token = f"{canonical_signal_id(epoch_id)}|RANDOM_SIDE".encode("utf-8")
    return "LONG" if hashlib.sha256(token).digest()[0] < 128 else "SHORT"


def base_fee_vector_sha256(values: Sequence[int]) -> str:
    """Hash 3,600 uint256 base fees as concatenated 32-byte big-endian words."""

    if len(values) != EPOCH_SIZE_BLOCKS:
        raise ValueError("ESDI-288 base-fee vector requires exactly 3,600 values")
    digest = hashlib.sha256()
    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 < value < 2**256
        ):
            raise ValueError("ESDI-288 base fees must be positive uint256 integers")
        digest.update(value.to_bytes(32, byteorder="big", signed=False))
    return digest.hexdigest()


def _strict_entry_seconds(values: Iterable[int], label: str) -> tuple[int, ...]:
    entries = tuple(values)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in entries
    ):
        raise ValueError(f"ESDI-288 {label} entries must be nonnegative integers")
    if any(left >= right for left, right in zip(entries, entries[1:])):
        raise ValueError(
            f"ESDI-288 {label} entries must be strictly increasing and unique"
        )
    return entries


def entries_in_domain(
    entries: Iterable[int],
    start_inclusive: int,
    end_exclusive: int,
) -> tuple[int, ...]:
    """Filter canonical entry seconds to one frozen half-open common domain."""

    if (
        isinstance(start_inclusive, bool)
        or isinstance(end_exclusive, bool)
        or not isinstance(start_inclusive, int)
        or not isinstance(end_exclusive, int)
        or start_inclusive < 0
        or end_exclusive <= start_inclusive
    ):
        raise ValueError("ESDI-288 comparison domain is invalid")
    canonical = _strict_entry_seconds(entries, "comparison")
    return tuple(
        value
        for value in canonical
        if start_inclusive <= value < end_exclusive
    )


def exact_entry_jaccard(
    left_entries: Iterable[int],
    right_entries: Iterable[int],
) -> Fraction:
    """Exact set Jaccard after strict duplicate rejection."""

    left = set(_strict_entry_seconds(left_entries, "left"))
    right = set(_strict_entry_seconds(right_entries, "right"))
    union = left | right
    if not union:
        raise ValueError("ESDI-288 exact-entry Jaccard has an empty union")
    return Fraction(len(left & right), len(union))


def fraction_at_most(
    value: Fraction,
    maximum_numerator: int,
    maximum_denominator: int,
) -> bool:
    """Apply every novelty threshold as an exact rational inclusive gate."""

    if not isinstance(value, Fraction) or value < 0:
        raise ValueError("ESDI-288 gate value must be a nonnegative Fraction")
    maximum = _rational((maximum_numerator, maximum_denominator))
    return value.numerator * maximum[1] <= maximum[0] * value.denominator


def fraction_below(
    value: Fraction,
    maximum_numerator: int,
    maximum_denominator: int,
) -> bool:
    """Apply strict source anti-degeneracy thresholds exactly."""

    if not isinstance(value, Fraction) or value < 0:
        raise ValueError("ESDI-288 gate value must be a nonnegative Fraction")
    maximum = _rational((maximum_numerator, maximum_denominator))
    return value.numerator * maximum[1] < maximum[0] * value.denominator


def bidirectional_entry_containment(
    left_entries: Iterable[int],
    right_entries: Iterable[int],
    window_seconds: int,
) -> Fraction:
    """Maximum directional fraction with an opposite entry within +/- window."""

    left = _strict_entry_seconds(left_entries, "left")
    right = _strict_entry_seconds(right_entries, "right")
    if not left or not right:
        raise ValueError("ESDI-288 containment requires two nonempty clocks")
    if (
        isinstance(window_seconds, bool)
        or not isinstance(window_seconds, int)
        or window_seconds < 0
    ):
        raise ValueError("ESDI-288 containment window must be nonnegative")

    def contained(source: tuple[int, ...], target: tuple[int, ...]) -> Fraction:
        count = 0
        cursor = 0
        for value in source:
            while (
                cursor < len(target)
                and target[cursor] < value - window_seconds
            ):
                cursor += 1
            if cursor < len(target) and target[cursor] <= value + window_seconds:
                count += 1
        return Fraction(count, len(source))

    return max(contained(left, right), contained(right, left))


def signed_exposure_5m(
    intervals: Iterable[tuple[int, int, int]],
    start_inclusive: int,
    end_exclusive: int,
) -> tuple[int, ...]:
    """Build a strict nonoverlapping {-1,0,1} exposure vector on 5m bar opens."""

    if (
        isinstance(start_inclusive, bool)
        or isinstance(end_exclusive, bool)
        or not isinstance(start_inclusive, int)
        or not isinstance(end_exclusive, int)
        or start_inclusive < 0
        or end_exclusive <= start_inclusive
        or start_inclusive % 300
        or end_exclusive % 300
    ):
        raise ValueError("ESDI-288 exposure domain must be a positive 5m grid")
    vector = [0] * ((end_exclusive - start_inclusive) // 300)
    previous_exit = start_inclusive
    for entry, exit_, side in intervals:
        if (
            isinstance(entry, bool)
            or isinstance(exit_, bool)
            or isinstance(side, bool)
            or not isinstance(entry, int)
            or not isinstance(exit_, int)
            or not isinstance(side, int)
            or side not in {-1, 1}
            or entry % 300
            or exit_ % 300
            or not start_inclusive <= entry < exit_ <= end_exclusive
            or entry < previous_exit
        ):
            raise ValueError(
                "ESDI-288 intervals must be sorted, nonoverlapping, contained "
                "5m intervals with side +/-1"
            )
        for index in range(
            (entry - start_inclusive) // 300,
            (exit_ - start_inclusive) // 300,
        ):
            vector[index] = side
        previous_exit = exit_
    return tuple(vector)


def _canonical_exposure_vector(
    values: Sequence[int],
    label: str,
) -> tuple[int, ...]:
    vector = tuple(values)
    if any(type(value) is not int or value not in {-1, 0, 1} for value in vector):
        raise ValueError(f"ESDI-288 {label} exposures must be exact integers -1/0/1")
    return vector


def occupied_bar_jaccard(
    left_exposure: Sequence[int],
    right_exposure: Sequence[int],
) -> Fraction:
    left_vector = _canonical_exposure_vector(left_exposure, "left")
    right_vector = _canonical_exposure_vector(right_exposure, "right")
    if len(left_vector) != len(right_vector) or not left_vector:
        raise ValueError("ESDI-288 occupied-bar vectors must have equal length")
    left = {index for index, value in enumerate(left_vector) if value != 0}
    right = {index for index, value in enumerate(right_vector) if value != 0}
    union = left | right
    if not union:
        raise ValueError("ESDI-288 occupied-bar Jaccard has an empty union")
    return Fraction(len(left & right), len(union))


def squared_signed_exposure_pearson(
    left_exposure: Sequence[int],
    right_exposure: Sequence[int],
) -> Fraction:
    """Return exact squared Pearson; zero variance is a terminal metric error."""

    left_vector = _canonical_exposure_vector(left_exposure, "left")
    right_vector = _canonical_exposure_vector(right_exposure, "right")
    if len(left_vector) != len(right_vector) or len(left_vector) < 2:
        raise ValueError("ESDI-288 Pearson vectors must share length >=2")
    count = len(left_vector)
    left_sum = sum(left_vector)
    right_sum = sum(right_vector)
    covariance = (
        count * sum(x * y for x, y in zip(left_vector, right_vector))
        - left_sum * right_sum
    )
    left_variance = count * sum(x * x for x in left_vector) - left_sum**2
    right_variance = count * sum(y * y for y in right_vector) - right_sum**2
    if left_variance == 0 or right_variance == 0:
        raise ValueError("ESDI-288 Pearson exposure has zero variance")
    return Fraction(covariance**2, left_variance * right_variance)


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _open_dependency(path: str | Path) -> int:
    candidate = Path(path)
    if (
        candidate.is_absolute()
        or ".." in candidate.parts
        or candidate.name in {"", ".", ".."}
    ):
        raise RuntimeError("ESDI-288 dependency path must be repository-relative")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(REPOSITORY_ROOT, directory_flags)
    try:
        for part in candidate.parent.parts:
            next_descriptor = os.open(part, directory_flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        file_descriptor = os.open(
            candidate.name,
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=descriptor,
        )
        if not stat.S_ISREG(os.fstat(file_descriptor).st_mode):
            os.close(file_descriptor)
            raise RuntimeError("ESDI-288 dependency must be a regular non-symlink")
        return file_descriptor
    except OSError as error:
        raise RuntimeError(
            "ESDI-288 dependency path is missing or unsafe"
        ) from error
    finally:
        os.close(descriptor)


def sha256_file(path: str | Path) -> str:
    descriptor = _open_dependency(path)
    try:
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _dependency_bytes(path: str | Path) -> bytes:
    descriptor = _open_dependency(path)
    try:
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _local_import_paths(module: str) -> set[Path]:
    parts = module.split(".")
    paths: set[Path] = set()
    module_path = Path(*parts)
    file_path = module_path.with_suffix(".py")
    package_path = module_path / "__init__.py"
    if (REPOSITORY_ROOT / file_path).is_file():
        paths.add(file_path)
    if (REPOSITORY_ROOT / package_path).is_file():
        paths.add(package_path)
    for end in range(1, len(parts)):
        parent_init = Path(*parts[:end]) / "__init__.py"
        if (REPOSITORY_ROOT / parent_init).is_file():
            paths.add(parent_init)
    return paths


def discover_runtime_code_closure() -> tuple[Path, ...]:
    """Discover local absolute-import closure for the frozen Gross9 runtime."""

    discovered: set[Path] = set()
    pending = list(RUNTIME_CODE_ROOTS)
    while pending:
        path = pending.pop()
        if path in discovered:
            continue
        discovered.add(path)
        try:
            tree = ast.parse(_dependency_bytes(path), filename=str(path))
        except (SyntaxError, UnicodeDecodeError) as error:
            raise RuntimeError(
                f"ESDI-288 cannot parse runtime dependency {path}"
            ) from error
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules.append(node.module)
            for module in modules:
                for imported_path in _local_import_paths(module):
                    if imported_path not in discovered:
                        pending.append(imported_path)
    return tuple(sorted(discovered))


def committed_identity_paths() -> tuple[Path, ...]:
    return tuple(
        sorted(
            {
                *COMMITTED_PREREGISTRATION_PATHS,
                *RUNTIME_CODE_CLOSURE_PATHS,
            }
        )
    )


def validate_runtime_code_closure() -> None:
    discovered = discover_runtime_code_closure()
    expected_code = tuple(
        path
        for path in RUNTIME_CODE_CLOSURE_PATHS
        if path.suffix == ".py"
    )
    if discovered != tuple(sorted(expected_code)):
        raise RuntimeError("ESDI-288 Gross9 runtime import closure changed")


def current_distribution_inventory() -> dict[str, str]:
    inventory: dict[str, str] = {}
    for distribution in importlib_metadata.distributions():
        raw_name = distribution.metadata.get("Name")
        if not raw_name:
            continue
        name = re.sub(r"[-_.]+", "-", raw_name).lower()
        version = distribution.version
        if name in inventory and inventory[name] != version:
            raise RuntimeError(
                f"ESDI-288 duplicate distribution versions for {name}"
            )
        inventory[name] = version
    return dict(sorted(inventory.items()))


def _distribution_inventory_hash(inventory: Mapping[str, str]) -> str:
    encoded = json.dumps(
        dict(inventory),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def current_runtime_environment() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for distribution in FROZEN_RUNTIME_ENVIRONMENT["packages"]:
        try:
            packages[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            packages[distribution] = None
    libc_name, libc_version = platform.libc_ver()
    inventory = current_distribution_inventory()
    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": list(sys.version_info[:3]),
        },
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "libc": [libc_name, libc_version],
        },
        "packages": packages,
        "all_distributions": inventory,
        "all_distributions_count": len(inventory),
        "all_distributions_sha256": _distribution_inventory_hash(inventory),
    }


def validate_runtime_environment() -> None:
    current = current_runtime_environment()
    selected = {
        "python": current["python"],
        "platform": current["platform"],
        "packages": current["packages"],
    }
    if selected != FROZEN_RUNTIME_ENVIRONMENT:
        raise RuntimeError("ESDI-288 Gross9 runtime environment changed")
    if (
        current["all_distributions_count"] != FROZEN_DISTRIBUTION_INVENTORY_COUNT
        or current["all_distributions_sha256"]
        != FROZEN_DISTRIBUTION_INVENTORY_SHA256
    ):
        raise RuntimeError("ESDI-288 Gross9 runtime environment changed")


def _committed_file_sha256(path: str | Path, expected_blob: str) -> str:
    """Hash one open file descriptor and prove its Git blob equals HEAD."""

    descriptor = _open_dependency(path)
    try:
        size = os.fstat(descriptor).st_size
        if len(expected_blob) == 40:
            git_digest = hashlib.sha1()
        elif len(expected_blob) == 64:
            git_digest = hashlib.sha256()
        else:
            raise RuntimeError("ESDI-288 Git object format is unsupported")
        git_digest.update(f"blob {size}\0".encode("ascii"))
        file_digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            file_digest.update(chunk)
            git_digest.update(chunk)
        if git_digest.hexdigest() != expected_blob:
            raise RuntimeError(
                "ESDI-288 preregistration file bytes differ from committed blob"
            )
        return file_digest.hexdigest()
    finally:
        os.close(descriptor)


def validate_frozen_document() -> None:
    actual = sha256_file(DOCUMENT_PATH)
    if actual != DOCUMENT_SHA256:
        raise RuntimeError(
            f"ESDI-288 frozen document changed: {actual} != {DOCUMENT_SHA256}"
        )


def _git(*arguments: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("ESDI-288 Git identity validation failed") from error
    return completed.stdout


def frozen_repository_identity() -> dict[str, Any]:
    """Bind the committed mechanism, producer, and test blobs without data access."""

    validate_runtime_code_closure()
    validate_runtime_environment()
    paths = [str(path) for path in committed_identity_paths()]
    status = _git(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        *paths,
    )
    if status:
        raise RuntimeError(
            "ESDI-288 preregistration files must be committed and unchanged"
        )

    records = _git("ls-tree", "-z", "HEAD", "--", *paths).split(b"\0")
    blobs: dict[str, str] = {}
    for record in records:
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode, object_type, object_id = metadata.decode("ascii").split()
        path = raw_path.decode("utf-8")
        if mode != "100644" or object_type != "blob":
            raise RuntimeError("ESDI-288 preregistration path is not a plain blob")
        blobs[path] = object_id
    if set(blobs) != set(paths):
        raise RuntimeError("ESDI-288 committed preregistration blobs are incomplete")

    return {
        "git_blobs": {path: blobs[path] for path in sorted(blobs)},
        "sha256": {
            path: _committed_file_sha256(path, blobs[path])
            for path in sorted(paths)
        },
        "whole_worktree_clean_required": False,
        "bound_paths_clean_against_HEAD_required": True,
    }


def validate_repository_identity(identity: Mapping[str, Any]) -> None:
    expected_paths = sorted(str(path) for path in committed_identity_paths())
    if sorted(identity.get("git_blobs", {})) != expected_paths:
        raise RuntimeError("ESDI-288 repository identity Git paths differ")
    if sorted(identity.get("sha256", {})) != expected_paths:
        raise RuntimeError("ESDI-288 repository identity SHA-256 paths differ")
    for object_id in identity["git_blobs"].values():
        if (
            not isinstance(object_id, str)
            or len(object_id) not in {40, 64}
            or any(character not in "0123456789abcdef" for character in object_id)
        ):
            raise RuntimeError("ESDI-288 repository identity Git blob is invalid")
    for digest in identity["sha256"].values():
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise RuntimeError("ESDI-288 repository identity SHA-256 is invalid")
    if identity.get("whole_worktree_clean_required") is not False:
        raise RuntimeError("ESDI-288 repository identity worktree rule differs")
    if identity.get("bound_paths_clean_against_HEAD_required") is not True:
        raise RuntimeError("ESDI-288 repository identity path rule differs")


def _core_manifest(repository_identity: Mapping[str, Any]) -> dict[str, Any]:
    validate_repository_identity(repository_identity)
    first_start, _, _ = epoch_blocks(FIRST_EPOCH_ID)
    _, last_end, last_confirmation = epoch_blocks(LAST_EPOCH_ID)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": POLICY_ID,
        "status": "outcome_blind_write_once_preincidence",
        "singleton": True,
        "frozen_preregistration": {
            "document": {
                "path": str(DOCUMENT_PATH),
                "sha256": DOCUMENT_SHA256,
            },
            "producer": {
                "path": str(PRODUCER_PATH),
                "test_path": str(TEST_PATH),
                "committed_and_unchanged_at_creation_required": True,
            },
            "repository_identity": copy.deepcopy(dict(repository_identity)),
            "serialization": {
                "encoding": "UTF-8",
                "sort_keys": True,
                "indent": 2,
                "ensure_ascii": True,
                "allow_nan": False,
                "trailing_lf_count": 1,
                "manifest_hash": (
                    "SHA256 compact sorted-key JSON excluding manifest_hash"
                ),
            },
        },
        "source": {
            "authority": "Ethereum mainnet",
            "chain_id": 1,
            "rpc_methods": ["eth_feeHistory", "eth_getBlockByNumber"],
            "https_transports_frozen_before_replay": 2,
            "request_chunk_blocks": 1_024,
            "rpc_attempts_per_request": 1,
            "rpc_retry_backoff_or_resume": False,
            "provider_fallback_after_values_open": False,
            "dual_replay_exact_agreement_required": True,
            "common_finalized_head_min_block": last_confirmation,
            "boundaries": copy.deepcopy(BOUNDARIES),
            "fail_closed_replay_invariants": {
                "shortened_fee_history_subsection_rejected": True,
                "oldest_block_base_fees_and_gas_ratios_equal_between_transports": True,
                "adjacent_chunk_next_base_fee_overlap_equal": True,
                "boundary_previous_timestamp_and_parent_relation_exact": True,
                "epoch_end_and_confirmation_headers_equal_between_transports": True,
                "nonmonotone_missing_duplicate_or_invalid_quantity_rejected": True,
                "any_transport_error_after_run_start_terminal": True,
                "resume_after_terminal_replay_error_allowed": False,
            },
            "epoch_size_blocks": EPOCH_SIZE_BLOCKS,
            "first_epoch_id": FIRST_EPOCH_ID,
            "last_epoch_id": LAST_EPOCH_ID,
            "epoch_count": LAST_EPOCH_ID - FIRST_EPOCH_ID + 1,
            "first_source_block": first_start,
            "last_source_block": last_end,
            "confirmation_blocks_after_end": CONFIRMATION_BLOCKS,
            "availability": "timestamp of end_block+64",
            "partial_epoch_timestamp_resampling_or_imputation": False,
            "median2": (
                "sort 3600 positive integer base fees; "
                "values[1799]+values[1800]"
            ),
            "base_fee_vector_sha256_serialization": (
                "exactly 3600 positive uint256 integers, each as 32-byte "
                "unsigned big-endian, concatenated in ascending block order"
            ),
            "mean_gas_used_ratio_arithmetic": "exact decimal",
            "normalized_row_fields": [
                "epoch_id",
                "start_block",
                "end_block",
                "end_block_hash",
                "end_block_timestamp_utc",
                "confirmation_block",
                "confirmation_block_hash",
                "available_at_utc",
                "median_base_fee_wei_x2",
                "base_fee_vector_sha256",
                "mean_gas_used_ratio_decimal",
            ],
        },
        "feature_and_signal": {
            "lag_epochs": 2,
            "first_feature_epoch_id": 4_533,
            "current": "median2[e]",
            "lagged": "median2[e-2]",
            "sign": "sign(current-lagged)",
            "magnitude_num": "max(current,lagged)",
            "magnitude_den": "min(current,lagged)",
            "ratio_comparison": "integer cross multiplication only",
            "rank_history": (
                "exactly previous 180 finite feature rows; current excluded"
            ),
            "rank_L": "count(prior_ratio < current_ratio)",
            "rank_E": "count(prior_ratio == current_ratio exactly)",
            "rank": "(2*L+E)/360",
            "threshold": {"operator": ">=", "numerator": 3, "denominator": 4},
            "candidate": "rank>=3/4 and sign!=0",
            "side": {"positive": "LONG", "negative": "SHORT", "zero": "ABSTAIN"},
            "signal_id": "ESDI-288|primary|epoch_id=<canonical decimal integer>",
            "event_onset_filter": False,
            "parameter_search_or_alternative_rule": False,
        },
        "execution": {
            "entry": "ceil_to_5m(available_at)+300 elapsed seconds",
            "aligned_availability_still_waits_seconds": 300,
            "hold_bars_5m": 288,
            "hold_seconds": 86_400,
            "leverage": 0.5,
            "base_cost_bp_per_notional_side": 6,
            "stress_cost_bp_per_notional_side": 10,
            "entry_price": "BTCUSDT perpetual 5m open at entry",
            "exit_price": "BTCUSDT perpetual 5m open at exact scheduled exit",
            "funding": {
                "interval": "entry_time <= funding_time < exit_time",
                "cash": "-side_sign*quantity*funding_rate*settlement_mark_price",
                "realized_only": True,
            },
            "candidate_order": ["entry_time", "available_at", "epoch_id", "side"],
            "reservation": {
                "scope": "one global position",
                "interval": "[entry_time,exit_time)",
                "accept": "entry_time >= previous accepted exit_time",
                "suppressed_candidates_queued": False,
            },
            "split_crossing_action": "skip; never truncate",
            "pyramiding_stop_take_profit_trailing_or_early_close": False,
        },
        "calendars": {
            "containment": (
                "half-open; both entry and exit contained; exit may equal end"
            ),
            "full": ["2023-06-01T00:00:00Z", "2026-06-01T00:00:00Z"],
            "selection": ["2023-06-01T00:00:00Z", "2025-01-01T00:00:00Z"],
            "future25": ["2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
            "future26": ["2026-01-01T00:00:00Z", "2026-06-01T00:00:00Z"],
            "selection_reports": {
                "2023H2": ["2023-06-01T00:00:00Z", "2024-01-01T00:00:00Z"],
                "2024H1": ["2024-01-01T00:00:00Z", "2024-07-01T00:00:00Z"],
                "2024H2": ["2024-07-01T00:00:00Z", "2025-01-01T00:00:00Z"],
            },
            "full_cagr_wall_clock_years": 3,
        },
        "controls": {
            "own_nonoverlap_clock": [
                "base_fee_one_epoch_stale",
                "gas_utilization_only",
                "base_fee_no_tail",
            ],
            "same_primary_parent_set": [
                "exact_direction_flip",
                "deterministic_random_side",
                "constant_long",
                "constant_short",
                "one_bar_delayed_entry",
            ],
            "definitions": {
                "base_fee_primary": "exact ESDI primary",
                "base_fee_one_epoch_stale": (
                    "median2[e-1] versus median2[e-3], no earlier than e "
                    "availability"
                ),
                "gas_utilization_only": (
                    "same lag, exact prior-180 midrank, threshold, side, "
                    "scheduler and hold on exact mean gas-used ratio"
                ),
                "base_fee_no_tail": (
                    "all nonzero primary signs, same scheduler and hold"
                ),
                "exact_direction_flip": "accepted primary entries with side flipped",
                "deterministic_random_side": (
                    "SHA256(UTF-8(ESDI-288|primary|epoch_id=<canonical decimal "
                    "integer>|RANDOM_SIDE)); first byte <128 LONG else SHORT"
                ),
                "constant_long": "accepted primary entries fixed LONG",
                "constant_short": "accepted primary entries fixed SHORT",
                "one_bar_delayed_entry": (
                    "accepted primary parent set shifted 300s without rerunning "
                    "non-overlap"
                ),
            },
            "controls_cannot_replace_or_repair_primary": True,
        },
        "support_gates": {
            "source": {
                "exact_epochs": 2_474,
                "missing_epochs": 0,
                "dual_replay_differences": 0,
                "boundary_header_differences": 0,
                "future_append_selection_differences": 0,
            },
            "selection": {
                "total_min": 45,
                "2023H2_min": 12,
                "2024H1_min": 12,
                "2024H2_min": 12,
                "each_side_min": 14,
                "maximum_month_share": 0.20,
            },
            "future25": {
                "total_min": 30,
                "each_side_min": 8,
                "maximum_month_share": 0.25,
            },
            "future26": {
                "total_min": 15,
                "each_side_min": 4,
                "maximum_month_share": 0.30,
            },
            "maximum_accepted_entry_gap_days": 90,
            "maximum_same_side_run": 12,
            "identity_clock_side_rank_tie_source_hash_reproducible": True,
            "independent_control_maxima_strict": {
                "exact_entry_jaccard": {
                    "operator": "<",
                    "numerator": 9,
                    "denominator": 10,
                },
                "candidate_24h_containment": {
                    "operator": "<",
                    "numerator": 19,
                    "denominator": 20,
                },
            },
            "failure_action": "retire ESDI-288 unchanged before outcomes",
        },
        "novelty": {
            "opens_only_after_complete_source_support": True,
            "frozen_comparator_artifacts": frozen_comparator_registry(),
            "comparator_min_entries_to_gate": 10,
            "minimum_count_is_after_common_domain_filter": True,
            "clock_identity": {
                "timestamp": "integer UTC Unix seconds; no rounding",
                "duplicates_or_unsorted_entries": "terminal failure",
                "comparison_domain": (
                    "registry half-open domain; filter both ESDI and comparator "
                    "entries to the identical domain"
                ),
                "intervals": (
                    "5m-aligned, contained, sorted, nonoverlapping [entry,exit)"
                ),
            },
            "executable_metric_functions_bound_by_producer_blob": [
                "entries_in_domain",
                "exact_entry_jaccard",
                "bidirectional_entry_containment",
                "fraction_at_most",
                "fraction_below",
                "signed_exposure_5m",
                "occupied_bar_jaccard",
                "squared_signed_exposure_pearson",
            ],
            "metric_definitions": {
                "exact_entry_jaccard": "|A intersect B| / |A union B|",
                "candidate_containment": (
                    "max(fraction of A within +/-window of any B, fraction of "
                    "B within +/-window of any A)"
                ),
                "signed_exposure": (
                    "{-1,0,1} on every 5m bar open in the common domain"
                ),
                "absolute_signed_exposure_pearson_gate": (
                    "exact squared Pearson; compare prior-source <=4/25 and "
                    "Gross9 <=49/400; report nonnegative square root only"
                ),
                "zero_variance_or_empty_metric_denominator": "terminal failure",
                "occupied_bar_jaccard": (
                    "Jaccard of bar indexes with nonzero signed exposure"
                ),
            },
            "prior_source_family_thresholds_exact_inclusive": {
                "exact_entry_jaccard": {"numerator": 1, "denominator": 5},
                "candidate_24h_containment": {"numerator": 1, "denominator": 2},
                "squared_signed_exposure_pearson": {
                    "numerator": 4,
                    "denominator": 25,
                },
            },
            "metric_applicability": {
                "directional_interval": (
                    "all three prior-source metrics are mandatory"
                ),
                "timestamp_only": (
                    "exact-entry Jaccard and candidate +/-24h containment are "
                    "mandatory; signed-exposure Pearson is inapplicable and "
                    "must be reported as such, never zero-filled"
                ),
                "missing_required_field_or_unknown_capability": (
                    "terminal novelty failure"
                ),
            },
            "gross9_each_positive_weight_sleeve_thresholds_exact_inclusive": {
                "exact_entry_jaccard": {"numerator": 1, "denominator": 10},
                "candidate_6h_containment": {"numerator": 7, "denominator": 20},
                "occupied_bar_jaccard": {"numerator": 1, "denominator": 4},
                "squared_signed_exposure_pearson": {
                    "numerator": 49,
                    "denominator": 400,
                },
            },
            "gross9_common_domain": [
                "2023-06-01T00:00:00Z",
                "2026-06-01T00:00:00Z",
            ],
            "all_frozen_comparator_artifacts_must_be_evaluated": True,
            "comparator_removal_after_overlap_seen": False,
            "failure_is_terminal": True,
        },
        "economic_contract": {
            "evaluator_committed_tested_and_hash_bound_before_rows_open": True,
            "standalone_gate_base_and_stress_each_period": {
                "absolute_return": ">0",
                "full_calendar_cagr_to_strict_mdd": ">=3.0",
                "strict_mdd": "<=0.15",
                "mean_gross_underlying_bp": ">=20",
                "calendar_month_clustered_signflip_p": "<=0.10",
            },
            "strict_mdd": (
                "global/pre-entry HWM; favorable OHLC and funding credits "
                "before adverse OHLC, funding debits, liquidation envelope "
                "and exit cost"
            ),
            "primary_strictly_exceeds": [
                "gas_utilization_only",
                "base_fee_one_epoch_stale",
            ],
            "cannot_completely_qualify": [
                "exact_direction_flip",
                "deterministic_random_side",
                "constant_long",
                "constant_short",
            ],
        },
        "gross9": {
            "authority": frozen_gross9_authority(),
            "weights": copy.deepcopy(GROSS9_WEIGHTS),
            "baseline_gross": 9.0,
            "candidate_weights": [0.25, 0.50, 0.75, 1.00],
            "treatment": "scale every sleeve by (9-w)/9 and add ESDI at w",
            "configured_treatment_gross": 9.0,
            "comparison": "unscaled authoritative Gross9 baseline at gross 9.0",
            "matching_execution_costs_exact_funding_and_strict_mdd": True,
            "selection_periods": ["2023H2", "2024"],
            "requirements": {
                "base_and_stress_cagr_mdd_improvement_min": 0.05,
                "unscaled_absolute_return_retention_min": 0.97,
                "base_and_stress_absolute_return_positive": True,
                "strict_mdd_reduced_in_at_least_one_selection_period": True,
            },
            "ranking": (
                "maximum minimum base/stress 2023H2/2024 improvement; "
                "tie lower weight"
            ),
            "freeze_rank": 1,
            "future_uses_only_frozen_weight": True,
            "future_rerank_or_alternate_weight": False,
        },
        "strict_sequence": [
            "mechanism_decision_commit",
            "write_once_preregistration_producer_and_tests_commit",
            "write_once_preregistration_artifact_commit",
            (
                "source_builder_source_support_incidence_novelty_gross9_and_"
                "strict_economic_evaluators_with_synthetic_tests_commit_bound_"
                "to_preregistration_hash_before_full_source_replay"
            ),
            "one_full_source_replay",
            "commit_write_once_source_artifacts_without_changing_evaluators",
            "source_support_run_stop_on_first_failure",
            "reconstruct_and_run_novelty_stop_on_first_failure",
            (
                "open_2023H2_then_2024_then_selection_then_same_gross_Gross9_"
                "then_future25_then_future26_then_stitched"
            ),
            "clean_checkout_hash_and_test_reproduction_commit_and_push",
        ],
        "sequence_rules": {
            "stop_at_first_failure": True,
            "later_periods_veto_only": True,
            "parameter_repair_polarity_inversion_or_rank2": False,
            "ordinary_failure_repair_under_policy_identity": False,
        },
        "producer_effects": {
            "network_calls": 0,
            "git_metadata_subprocess_calls": 2,
            "data_rows_opened": 0,
            "comparator_or_gross9_artifact_bytes_opened": 0,
            "bound_committed_code_config_and_lock_files_hashed": len(
                committed_identity_paths()
            ),
        },
        **{name: False for name in EVIDENCE_BOUNDARIES},
    }


def build_manifest(repository_identity: Mapping[str, Any]) -> dict[str, Any]:
    core = _core_manifest(repository_identity)
    return {**core, "manifest_hash": canonical_hash(core)}


def validate_manifest(payload: Mapping[str, Any]) -> None:
    identity = payload.get("frozen_preregistration", {}).get(
        "repository_identity"
    )
    if not isinstance(identity, Mapping):
        raise RuntimeError("ESDI-288 repository identity is missing")
    expected = build_manifest(identity)
    if dict(payload) != expected:
        raise RuntimeError("ESDI-288 preregistration differs from frozen code")
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("ESDI-288 internal manifest hash mismatch")
    if any(payload.get(name) is not False for name in EVIDENCE_BOUNDARIES):
        raise RuntimeError("ESDI-288 evidence boundary opened")
    if payload["producer_effects"] != {
        "network_calls": 0,
        "git_metadata_subprocess_calls": 2,
        "data_rows_opened": 0,
        "comparator_or_gross9_artifact_bytes_opened": 0,
        "bound_committed_code_config_and_lock_files_hashed": len(
            committed_identity_paths()
        ),
    }:
        raise RuntimeError("ESDI-288 producer effects changed")
    if sum(GROSS9_WEIGHTS.values()) != 9.0:
        raise RuntimeError("ESDI-288 Gross9 weights do not sum to 9.0")


def canonical_manifest_bytes(payload: Mapping[str, Any]) -> bytes:
    validate_manifest(payload)
    return (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _output_relative(path: str | Path) -> Path:
    candidate = Path(path)
    raw = str(path)
    if (
        candidate.is_absolute()
        or raw.startswith("~")
        or ".." in candidate.parts
        or candidate.name in {"", ".", ".."}
    ):
        raise RuntimeError("ESDI-288 output must be repository-relative")
    if candidate != DEFAULT_OUTPUT:
        raise RuntimeError("ESDI-288 output must equal the frozen singleton path")
    return candidate


def _open_parent(candidate: Path) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(REPOSITORY_ROOT, flags)
    try:
        for part in candidate.parent.parts:
            next_descriptor = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except OSError as error:
        os.close(descriptor)
        raise RuntimeError("ESDI-288 output parent is unsafe") from error


def _read_regular(parent_fd: int, filename: str) -> bytes:
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(filename, flags, dir_fd=parent_fd)
    except OSError as error:
        if isinstance(error, FileNotFoundError):
            raise
        raise RuntimeError("ESDI-288 output path is unsafe") from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise RuntimeError("ESDI-288 output is not a regular file")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def write_once(
    output: str | Path = DEFAULT_OUTPUT,
    payload: Mapping[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Atomically create the artifact or verify an identical regular file."""

    candidate = _output_relative(output)
    validate_frozen_document()
    identity = frozen_repository_identity()
    expected = build_manifest(identity)
    if payload is not None and dict(payload) != expected:
        raise RuntimeError("ESDI-288 supplied preregistration payload drift")
    canonical = canonical_manifest_bytes(expected)
    parent_fd = _open_parent(candidate)
    temporary = f".{candidate.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    temporary_created = False
    try:
        try:
            existing = _read_regular(parent_fd, candidate.name)
        except FileNotFoundError:
            existing = None
        if existing is not None:
            if existing != canonical:
                raise RuntimeError("ESDI-288 existing preregistration drift")
            return "verified_existing", expected

        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_fd,
        )
        temporary_created = True
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical)
            handle.flush()
            os.fchmod(handle.fileno(), 0o444)
            os.fsync(handle.fileno())
        try:
            os.link(
                temporary,
                candidate.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError:
            if _read_regular(parent_fd, candidate.name) != canonical:
                raise RuntimeError("ESDI-288 preregistration race drift")
            return "verified_existing", expected
        os.fsync(parent_fd)
        return "created", expected
    finally:
        if temporary_created:
            try:
                os.unlink(temporary, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
            os.fsync(parent_fd)
        os.close(parent_fd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    status, payload = write_once()
    print(
        json.dumps(
            {
                "status": status,
                "output": str(DEFAULT_OUTPUT),
                "manifest_hash": payload["manifest_hash"],
                **{name: payload[name] for name in EVIDENCE_BOUNDARIES},
            },
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
