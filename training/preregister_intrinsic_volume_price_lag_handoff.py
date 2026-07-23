"""Freeze IVPLH-72 before decoding candidate incidence or market outcomes."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = (
    "results/intrinsic_volume_price_lag_handoff_preregistration_2026-07-23.json"
)
MECHANISM_DOCUMENT = (
    "docs/intrinsic-volume-price-lag-handoff-mechanism-decision-2026-07-23.md"
)
MECHANISM_DOCUMENT_SHA256 = (
    "ce16708f7d6fd3e4f9459ec4f8778f30336c6439f7317f4dbe6d71221cbf86dc"
)
BOUNDARY_DOCUMENT = (
    "docs/intrinsic-volume-price-lag-handoff-boundary-2026-07-23.md"
)
BOUNDARY_DOCUMENT_SHA256 = (
    "e1f12a7ccf693f2aafecd3b14e74090e2c1560a39a89ca59f2da3356c4cf244d"
)
COMMON_WINDOW_POLICY = "docs/novelty-comparator-common-window-policy-2026-07-23.md"
COMMON_WINDOW_POLICY_SHA256 = (
    "928bce6e04fb34001478b4b4ea84156580b661c88a0f0338065a891c009bd580"
)
MARKET_MANIFEST = "data/binance_um_kline_reference_btc_2020_2023/build_manifest.json"
MARKET_MANIFEST_SHA256 = (
    "c04fbbd299cc748a6745c0ef030787da4d560833c744c81c98dd8840efc7913e"
)
MARKET_SOURCE = (
    "data/binance_um_kline_reference_btc_2020_2023/"
    "BTCUSDT_5m_2020-01-01_2023-12-31.csv.gz"
)
MARKET_SOURCE_SHA256 = (
    "e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d"
)


@dataclass(frozen=True)
class Policy:
    policy_id: str = "IVPLH-72"
    utc_day_volume_lookback_days: int = 28
    utc_day_volume_min_days: int = 21
    intrinsic_volume_fraction: float = 0.50
    latest_anchor_minute_utc: int = 17 * 60 + 55
    event_reference_anchors: int = 180
    event_reference_min_anchors: int = 90
    decision_delay_bars: int = 1
    entry_delay_bars: int = 2
    hold_bars: int = 72
    fixed_noon_anchor_minute_utc: int = 11 * 60 + 55
    stale_control_delay_bars: int = 288
    leverage: float = 0.50
    base_cost_notional_per_side: float = 0.0006
    stress_cost_notional_per_side: float = 0.0010
    cluster_signflip_draws: int = 20_000
    mdd_denominator_floor: float = 1e-9


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


def sha256_csv_header(path: str | Path) -> str:
    source = Path(path)
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rb") as handle:
        header = handle.readline()
    if not header.endswith(b"\n") or b"\n" in header[:-1]:
        raise RuntimeError(f"IVPLH-72 comparator header is not one LF line: {path}")
    return hashlib.sha256(header).hexdigest()


def _predecessor_contract() -> dict[str, Any]:
    return {
        "preregistration": {
            "path": "results/intrinsic_volume_flow_handoff_relay_preregistration_2026-07-23.json",
            "sha256": "e01e7f5af034adf98c0eef1e086ed1265c02998641f39d8cddd5137089f4153e",
        },
        "support_report": {
            "path": "results/intrinsic_volume_flow_handoff_relay_support_2026-07-23.json",
            "sha256": "ed2a82e875d650f2e6f3197df1d34e39617d07640b5e13a3cc7ccc4bb09661d4",
        },
        "clock": {
            "path": "data/intrinsic_volume_flow_handoff_relay_clocks_2020_2023.csv.gz",
            "sha256": "ab12762dec9a93d41c293766e46dfc80ade81914fb32753a5923faa6437c338e",
            "header": [
                "clock_name",
                "source_day",
                "decision_time",
                "entry_time",
                "exit_time",
                "side",
            ],
            "header_sha256": "0ad7d7a39f7d772de30d2c47056efd3c9b7740561eea9a1b69007b4870d5d495",
        },
        "known_clock_names": [
            "primary",
            "any_handoff",
            "no_price_lag",
            "no_flow_strength",
            "persistence_level",
            "fixed_noon_handoff",
            "exact_side_flip",
            "deterministic_random_side",
        ],
        "selected_clock_name": "any_handoff",
        "disclosed_global_rows": 66,
        "identity_key": ["source_day", "side", "decision_time"],
        "candidate_decision_equals_predecessor_entry": True,
        "candidate_entry_and_exit_shift_bars": 1,
        "identity_mismatch_action": "reject unchanged before support",
    }


def _comparator_contracts() -> list[dict[str, Any]]:
    return [
        {
            "id": "IVLIR-72",
            "path": "data/intrinsic_volume_latent_impact_relay_clocks_2020_2023.csv.gz",
            "sha256": "523f24a0d955fe99cfb86c62078532c5fc9091234e6669ab9acff2a8f3367788",
            "header": [
                "clock_name",
                "source_day",
                "decision_time",
                "entry_time",
                "exit_time",
                "side",
            ],
            "header_sha256": "0ad7d7a39f7d772de30d2c47056efd3c9b7740561eea9a1b69007b4870d5d495",
            "entry_column": "entry_time",
            "exit_column": "exit_time",
            "side_column": "side",
            "side_encoding": {"LONG": "LONG", "SHORT": "SHORT"},
            "group_column": "clock_name",
            "known_groups": [
                "primary",
                "flow_only",
                "no_under_response",
                "no_headroom",
                "fixed_noon",
                "exact_side_flip",
                "deterministic_random_side",
            ],
            "selected_group": "primary",
            "source_day_column": "source_day",
            "hold_bars": 72,
            "tolerant_containment": True,
        },
        {
            "id": "BAFR-24F",
            "path": "results/binance_aggressor_frustration_clock_2026-07-20.csv",
            "sha256": "f3b816a76decce31136ed23d22f043eb8e80ef1b8697b869241b060062f01747",
            "header": [
                "signal_position",
                "entry_position",
                "exit_position",
                "signal_date",
                "entry_date",
                "exit_date",
                "side",
                "hold_bars",
            ],
            "header_sha256": "437d41a791ba1084c3f38903ed6352f61462c3e3f3c5bb8fa065519a11b13852",
            "entry_column": "entry_date",
            "exit_column": "exit_date",
            "side_column": "side",
            "side_encoding": {"1": "LONG", "-1": "SHORT"},
            "group_column": None,
            "known_groups": ["BAFR-24F"],
            "selected_group": "BAFR-24F",
            "source_day_column": None,
            "hold_bars": 24,
            "tolerant_containment": False,
        },
        {
            "id": "AFCS-144",
            "path": "results/aggregate_fill_compression_sweep_clock_2026-07-17.csv",
            "sha256": "bf1611554604c1930ba2212e674ea434f7c9793377b3f33ef531b3b4e0381688",
            "header": [
                "origin_position",
                "signal_position",
                "entry_position",
                "exit_position",
                "origin_date",
                "signal_date",
                "entry_date",
                "exit_date",
                "side",
                "branch",
                "delay_bars",
                "hold_bars",
            ],
            "header_sha256": "fbe1d4fc7a2981a9fec253c5e6e04874035899626c9a19d96a97774e2b2d1999",
            "entry_column": "entry_date",
            "exit_column": "exit_date",
            "side_column": "side",
            "side_encoding": {"1": "LONG", "-1": "SHORT"},
            "group_column": "branch",
            "known_groups": ["afcs_144"],
            "selected_group": "afcs_144",
            "source_day_column": None,
            "hold_bars": 144,
            "tolerant_containment": True,
        },
        {
            "id": "LVRT-R0",
            "path": "results/liquidity_vacuum_replenishment_clock_2026-07-17.csv",
            "sha256": "ed9dd6391df2118ac09d147a4e57c3cb3f6e105a13f6c0d973ee424cfedd54d2",
            "header": [
                "setup_position",
                "signal_position",
                "entry_position",
                "exit_position",
                "setup_date",
                "signal_date",
                "entry_date",
                "exit_date",
                "side",
                "branch",
                "hold_bars",
            ],
            "header_sha256": "53ede9e934bd3c0612944e9ad678cb81e1400c5e0c3d64a10ed3401157a900e0",
            "entry_column": "entry_date",
            "exit_column": "exit_date",
            "side_column": "side",
            "side_encoding": {"1": "LONG", "-1": "SHORT"},
            "group_column": "branch",
            "known_groups": ["lvrt_r0"],
            "selected_group": "lvrt_r0",
            "source_day_column": None,
            "hold_bars": 12,
            "tolerant_containment": True,
        },
        {
            "id": "SMCC-144",
            "path": "data/same_millisecond_cascade_clock_2020_2023.csv.gz",
            "sha256": "3b255b224ab510afc30edb265d62428db9fdf07d90610499df62efff9ffa410d",
            "header": [
                "decision_time",
                "entry_time",
                "exit_time",
                "side",
                "score",
                "threshold",
            ],
            "header_sha256": "56bc773a89f31d3c29c6ab5177451df6fe40518c0d879ec97248696e1ecb2b9c",
            "entry_column": "entry_time",
            "exit_column": "exit_time",
            "side_column": "side",
            "side_encoding": {"1": "LONG", "-1": "SHORT"},
            "group_column": None,
            "known_groups": ["SMCC-144"],
            "selected_group": "SMCC-144",
            "source_day_column": None,
            "hold_bars": 144,
            "tolerant_containment": True,
        },
        {
            "id": "QLCD-288",
            "path": "data/quantity_lattice_cohort_disagreement_clock_2020_2023.csv.gz",
            "sha256": "ed882ac8a28f1f0b2b7ad7bf3d2de1f37b175cde63b20d4d1c7a290f3eb89bec",
            "header": [
                "decision_time",
                "entry_time",
                "exit_time",
                "side",
                "score",
                "threshold",
            ],
            "header_sha256": "56bc773a89f31d3c29c6ab5177451df6fe40518c0d879ec97248696e1ecb2b9c",
            "entry_column": "entry_time",
            "exit_column": "exit_time",
            "side_column": "side",
            "side_encoding": {"1": "LONG", "-1": "SHORT"},
            "group_column": None,
            "known_groups": ["QLCD-288"],
            "selected_group": "QLCD-288",
            "source_day_column": None,
            "hold_bars": 288,
            "tolerant_containment": True,
        },
    ]


def _core_manifest() -> dict[str, Any]:
    policy = Policy()
    return {
        "protocol_version": "intrinsic_volume_price_lag_handoff_v1",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "predecessor_rows_decoded": False,
        "comparator_rows_decoded": False,
        "policy": asdict(policy),
        "frozen_documents": {
            "boundary": {
                "path": BOUNDARY_DOCUMENT,
                "sha256": BOUNDARY_DOCUMENT_SHA256,
            },
            "mechanism": {
                "path": MECHANISM_DOCUMENT,
                "sha256": MECHANISM_DOCUMENT_SHA256,
            },
            "common_window_policy": {
                "path": COMMON_WINDOW_POLICY,
                "sha256": COMMON_WINDOW_POLICY_SHA256,
            },
        },
        "research_history_boundary": {
            "source_seen_successor": True,
            "predecessor_any_handoff_global_support_seen": True,
            "predecessor_exact_rows_or_split_counts_seen": False,
            "ivplh_post_entry_outcomes_seen": False,
            "post_2023_source_values_seen": False,
            "claim_scope": (
                "source-seen candidate; support is operational adequacy, not discovery evidence"
            ),
            "llm_used_in_this_stage": False,
        },
        "predecessor_lineage": _predecessor_contract(),
        "source_contract": {
            "market_manifest": MARKET_MANIFEST,
            "market_manifest_sha256": MARKET_MANIFEST_SHA256,
            "market": MARKET_SOURCE,
            "market_sha256": MARKET_SOURCE_SHA256,
            "market_rows": 420_768,
            "interval": ["2020-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "bar_interval": "5min",
            "required_columns": [
                "date",
                "open",
                "high",
                "low",
                "close",
                "quote_asset_volume",
                "taker_buy_quote",
            ],
            "state_columns": [
                "date",
                "open",
                "close",
                "quote_asset_volume",
                "taker_buy_quote",
            ],
            "validation_only_columns": ["high", "low"],
            "exact_grid_required": True,
            "complete_rows_per_utc_day": 288,
            "missing_bar_policy": "fail closed; no interpolation or forward fill",
            "taker_quote_tolerance": "max(1e-8, abs(quote_asset_volume)*1e-10)",
        },
        "causal_feature_contract": {
            "daily_expected_volume": (
                "median total quote_asset_volume of previous 28 complete UTC days; "
                "current day excluded; at least 21 days"
            ),
            "intrinsic_target": "0.50 * daily_expected_volume",
            "anchor": (
                "first completed 5m bar reaching target; bar-open <=17:55 UTC"
            ),
            "cumulative_flow": (
                "sum(2*taker_buy_quote-quote_asset_volume) / cumulative quote volume "
                "from UTC-day start through anchor"
            ),
            "flow_side": "LONG iff cumulative_flow>0; SHORT iff cumulative_flow<0",
            "event_reference": (
                "last at most 180 strictly-prior eligible anchors; at least 90 required; "
                "no threshold derived from reference"
            ),
            "calendar_predecessor": (
                "latest prior eligible anchor must have source_day exactly D-1 calendar day"
            ),
            "handoff": "current flow side equals negative previous flow side",
            "price_lag": "flow_side_sign * log(anchor_close/day_open) <= 0",
            "side": "fixed to current/new flow side",
            "future_bar_used_by_signal": False,
            "flow_magnitude_threshold": None,
            "prior_run_minimum": None,
        },
        "execution_contract": {
            "anchor_bar_semantics": "t covers [t,t+5m) and is available at t+5m",
            "decision_time": "anchor bar-open + 5m",
            "entry_time": "anchor bar-open + 10m",
            "exit_time": "entry_time + 72*5m",
            "split_reservations": {
                "lineage_calibration": [
                    "2020-01-01T00:00:00Z",
                    "2023-01-01T00:00:00Z",
                ],
                "selection": [
                    "2023-01-01T00:00:00Z",
                    "2024-01-01T00:00:00Z",
                ],
            },
            "containment": (
                "source day in UTC date range; S<=decision; S<=entry; exit<=E"
            ),
            "reservation": (
                "sort raw candidates by (entry_time,signal_id); accept iff "
                "entry_time>=prior_accepted_exit_time; each control independent"
            ),
            "signal_id": (
                "SHA256 canonical JSON keys control,decision_time,policy_id,side,"
                "source_day,source_panel_sha256"
            ),
            "stop_or_take_profit": None,
        },
        "source_only_controls": {
            "ordered": [
                "primary",
                "handoff_without_price_lag",
                "price_lag_without_handoff",
                "fixed_noon",
                "stale_24h",
                "direction_flip",
                "anchor_side_year_permutation",
                "anchor_return_year_permutation",
                "deterministic_random_side",
            ],
            "permutation_rng": None,
            "permutation_mapping": (
                "within source-day year, SHA256 lexical donor/destination bijection"
            ),
            "random_side": (
                "first SHA256 byte of side-free canonical identity; LONG iff <128"
            ),
            "skipped_trade_releases_reservation": False,
        },
        "source_support_gate": {
            "train_window": ["2021-01-01T00:00:00Z", "2023-01-01T00:00:00Z"],
            "selection_window": [
                "2023-01-01T00:00:00Z",
                "2024-01-01T00:00:00Z",
            ],
            "train_events_min": 24,
            "each_train_year_events_min": 10,
            "each_train_half_events_min": 3,
            "train_each_side_events_min": 6,
            "train_each_side_share_min": 0.20,
            "selection_events_min": 12,
            "each_selection_half_events_min": 4,
            "selection_each_side_events_min": 3,
            "selection_each_side_share_min": 0.20,
            "maximum_split_month_share": 0.20,
            "maximum_split_quarter_share": 0.40,
            "maximum_split_gap_days": 120.0,
            "maximum_split_same_side_run": 10,
            "permutation_exact_entry_jaccard_max": 0.35,
            "permutation_same_side_reproduction_max": 0.60,
            "empty_denominator": "one and fail",
            "statistic_definitions": {
                "subperiod_membership": (
                    "source day in UTC date range; decision and entry >= start; exit <= end"
                ),
                "train_halves": [
                    ["2021_h1", "2021-01-01", "2021-07-01"],
                    ["2021_h2", "2021-07-01", "2022-01-01"],
                    ["2022_h1", "2022-01-01", "2022-07-01"],
                    ["2022_h2", "2022-07-01", "2023-01-01"],
                ],
                "selection_halves": [
                    ["2023_h1", "2023-01-01", "2023-07-01"],
                    ["2023_h2", "2023-07-01", "2024-01-01"],
                ],
                "month_quarter_denominator": "accepted entries in containing split",
                "gap": "maximum elapsed accepted-entry gap within split",
                "same_side_run": "chronological (entry_time,signal_id) within split",
                "permutation_jaccard": "distinct accepted entry-time sets within split",
                "same_side_reproduction": (
                    "exact timestamp and normalized side matches / primary split entries"
                ),
            },
            "failure_action": "retire IVPLH-72 unchanged before comparators/outcomes",
        },
        "novelty_contract": {
            "common_window_policy_path": COMMON_WINDOW_POLICY,
            "common_window_policy_sha256": COMMON_WINDOW_POLICY_SHA256,
            "common_window": [
                "2021-01-01T00:00:00Z",
                "2024-01-01T00:00:00Z",
            ],
            "raw_validation_before_filter": True,
            "full_containment": "entry>=W0 and exit<=W1; never clip",
            "minimum_selected_contained_rows": 10,
            "comparators": _comparator_contracts(),
            "exact_entry_jaccard_max": 0.10,
            "absolute_signed_occupancy_pearson_max": 0.35,
            "zero_or_nonfinite_correlation": "one and fail",
            "tolerant_match_minutes": 60,
            "maximum_bidirectional_containment_max": 0.40,
            "match_pair_order": [
                "absolute_delta",
                "candidate_entry_time",
                "comparator_entry_time",
            ],
            "ivlir_source_day_jaccard_max": 0.25,
            "required_count_report": [
                "raw",
                "contained",
                "before",
                "after",
                "crossing",
            ],
            "failure_action": "retire IVPLH-72 unchanged before outcomes",
        },
        "economic_contract": {
            "instrument": "Binance USD-M BTCUSDT perpetual",
            "leverage": policy.leverage,
            "base_cost_notional_per_side": policy.base_cost_notional_per_side,
            "stress_cost_notional_per_side": policy.stress_cost_notional_per_side,
            "funding_interval": "entry_time <= funding_time < exit_time",
            "funding_cash": "-side_sign*quantity*funding_rate*settlement_mark",
            "quantity": "0.5*pre_entry_equity/entry_open; fixed through exit",
            "mean_gross_underlying_bp": (
                "mean(side_sign*(exit_open/entry_open-1)*10000)"
            ),
            "strict_mdd": (
                "global/pre-entry HWM; entry fee; each held-bar favorable then adverse; "
                "signed same-bar funding; adverse virtual exit fee; scheduled exit fee"
            ),
            "calendar_years": "elapsed_seconds/(365.25*86400)",
            "cluster_signflip": (
                "20000 SHA256 one-sided ISO UTC entry-week sign flips; add-one p-value"
            ),
            "one_extra_bar_delay": (
                "shift entry and exit +5m; same event set/side/72-bar hold; no reschedule"
            ),
        },
        "strict_sequence": {
            "phase_1": "source support and permutation controls",
            "phase_2": "comparator novelty only after phase 1 passes",
            "phase_3": "commit hash-bound strict evaluator before outcomes",
            "stages": [
                ["train", "2021-01-01T00:00:00Z", "2023-01-01T00:00:00Z"],
                ["selection", "2023-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
                ["test_2024", "2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"],
                ["eval_2025", "2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
                ["final_2026_h1", "2026-01-01T00:00:00Z", "2026-07-01T00:00:00Z"],
            ],
            "stop_at_first_failure": True,
            "no_parameter_repair": True,
        },
        "economic_gates": {
            "each_stage_absolute_return_positive": True,
            "each_stage_cagr_to_strict_mdd_min": 3.0,
            "each_stage_strict_mdd_pct_max": 15.0,
            "base_and_stress_cost_positive": True,
            "one_extra_bar_delay_positive": True,
            "mean_gross_underlying_bp_min": 15.0,
            "weekly_cluster_signflip_p_max": 0.10,
            "train_each_year_positive": True,
            "selection_each_half_positive": True,
            "combined_2024_2025_ratio_min": 3.0,
            "combined_2024_2025_absolute_return_positive": True,
            "combined_2024_2025_cluster_p_max": 0.05,
            "component_margin_bp_min": 5.0,
            "component_controls": [
                "handoff_without_price_lag",
                "price_lag_without_handoff",
                "fixed_noon",
                "stale_24h",
            ],
        },
        "llm_boundary": {
            "activation_requires_base_train_and_selection_pass": True,
            "protocol_freeze_after_base_train_before_selection": True,
            "protocol_committed_before_label_dataset": True,
            "train_labels_only": True,
            "selection_feedback_forbidden": True,
            "action_space": ["TRADE_FIXED_SIDE", "ABSTAIN"],
            "may_not_change": [
                "candidate clock",
                "side",
                "entry",
                "hold",
                "leverage",
                "cost model",
            ],
            "raw_timestamp_or_price_forbidden": True,
            "exact_split_identity_forbidden": True,
            "allowed_symbolic_prior_buckets": [
                "new side",
                "previous side",
                "previous run length",
                "anchor time",
                "flow magnitude rank",
                "price lag rank",
                "volume target overshoot rank",
                "time since prior handoff",
                "source freshness",
                "current portfolio position",
            ],
            "skipped_trade_releases_reservation": False,
        },
        "stopping_rule": (
            "first failed identity, integrity, support, permutation, novelty, or "
            "economic gate retires IVPLH-72 unchanged"
        ),
    }


def build_manifest() -> dict[str, Any]:
    core = _core_manifest()
    return {
        **core,
        "manifest_hash": canonical_hash(core),
    }


def validate_manifest(payload: dict[str, Any]) -> None:
    core = {
        key: value
        for key, value in payload.items()
        if key != "manifest_hash"
    }
    if canonical_hash(core) != payload.get("manifest_hash"):
        raise RuntimeError("IVPLH-72 preregistration hash mismatch")
    if core != _core_manifest():
        raise RuntimeError("IVPLH-72 preregistration core differs from code")


def frozen_dependencies() -> dict[str, str]:
    dependencies = {
        BOUNDARY_DOCUMENT: BOUNDARY_DOCUMENT_SHA256,
        MECHANISM_DOCUMENT: MECHANISM_DOCUMENT_SHA256,
        COMMON_WINDOW_POLICY: COMMON_WINDOW_POLICY_SHA256,
        MARKET_MANIFEST: MARKET_MANIFEST_SHA256,
        MARKET_SOURCE: MARKET_SOURCE_SHA256,
    }
    predecessor = _predecessor_contract()
    for key in ("preregistration", "support_report", "clock"):
        item = predecessor[key]
        dependencies[item["path"]] = item["sha256"]
    for item in _comparator_contracts():
        dependencies[item["path"]] = item["sha256"]
    return dependencies


def validate_frozen_dependencies() -> None:
    for path, expected in frozen_dependencies().items():
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"IVPLH-72 frozen dependency changed: {path}")
    predecessor_clock = _predecessor_contract()["clock"]
    if sha256_csv_header(predecessor_clock["path"]) != predecessor_clock["header_sha256"]:
        raise RuntimeError("IVPLH-72 predecessor header changed")
    for item in _comparator_contracts():
        if sha256_csv_header(item["path"]) != item["header_sha256"]:
            raise RuntimeError(f"IVPLH-72 comparator header changed: {item['id']}")


def _canonical_manifest_text() -> str:
    return json.dumps(
        build_manifest(),
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"


def write_once(path: str | Path, payload: dict[str, Any]) -> str:
    output = Path(path)
    validate_frozen_dependencies()
    validate_manifest(payload)
    canonical_text = _canonical_manifest_text()
    output.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(
            file_descriptor,
            "w",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            handle.write(canonical_text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o644)
        try:
            os.link(temporary, output)
        except FileExistsError:
            if (
                output.is_symlink()
                or not output.is_file()
                or output.read_text(encoding="utf-8") != canonical_text
            ):
                raise RuntimeError(
                    "refusing noncanonical existing IVPLH-72 preregistration"
                )
            return "verified_existing"
        return "created"
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build_manifest()
    status = write_once(args.output, payload)
    print(
        json.dumps(
            {
                "status": status,
                "output": str(args.output),
                "manifest_hash": payload["manifest_hash"],
                "outcomes_opened": payload["outcomes_opened"],
                "source_incidence_opened": payload["source_incidence_opened"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
