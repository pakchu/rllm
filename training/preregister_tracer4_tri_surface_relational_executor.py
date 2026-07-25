"""Freeze TRACER-4H source/language support before decoding incidence or outcomes."""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import numpy as np
import pandas as pd


POLICY_ID = "TRACER-4H"
PROTOCOL_VERSION = "tracer4_tri_surface_relational_executor_preregistration_v1"
DEFAULT_OUTPUT = (
    "results/tracer4_tri_surface_relational_executor_"
    "preregistration_2026-07-25.json"
)
BOUNDARY_DOCUMENT = (
    "docs/tracer4-tri-surface-relational-executor-boundary-2026-07-25.md"
)
BOUNDARY_DOCUMENT_SHA256 = (
    "45c15c2ef2664c35857186363894727010538f878cdda1088eb923b5a653f7d5"
)
BOUNDARY_COMMIT = "66218040c8c9aaaaa25fa398a07b823d6b3447ff"

SOURCE_START = "2020-01-01T00:00:00Z"
SOURCE_END_EXCLUSIVE = "2024-01-01T00:00:00Z"

LEADERSHIP_SOURCE = (
    "data/binance_cross_venue_minute_leadership_btc_2020_2023/"
    "BTCUSDT_cross_venue_minute_leadership_5m_2020-01_2023-12.csv.gz"
)
LEADERSHIP_SOURCE_SHA256 = (
    "00ab6a55fc7bfeb3012584db5bc97a7d7b98dd995491acfd3f865c6bd41f92cc"
)
LEADERSHIP_HEADER_SHA256 = (
    "b7c730d6fc2c37d6e94f6a436478fd09ff42d15d7fd81bf521c4ca36465ff49f"
)
LEADERSHIP_MANIFEST = (
    "data/binance_cross_venue_minute_leadership_btc_2020_2023/build_manifest.json"
)
LEADERSHIP_MANIFEST_SHA256 = (
    "544c2945a2b56be478a1edc4abbb93b762bda5afc32cbd0658dd6822ff6b70fa"
)
LEADERSHIP_PHYSICAL_HEADER = tuple(
    "date,feature_available_time_utc,trade_earliest_time_utc,spot_rows,um_rows,"
    "spot_missing_minutes,um_missing_minutes,spot_invalid_source_minutes,"
    "um_invalid_source_minutes,lagged_pair_count,reverse_lagged_pair_count,"
    "simultaneous_flow_pair_count,simultaneous_return_pair_count,"
    "spot_quote_notional,um_quote_notional,spot_trade_count,um_trade_count,"
    "spot_signed_quote_notional,um_signed_quote_notional,spot_flow_fraction,"
    "um_flow_fraction,spot_flow_coherence,um_flow_coherence,spot_log_return_5m,"
    "um_log_return_5m,spot_abs_path_return_bp,um_abs_path_return_bp,"
    "spot_activity_time_centroid,um_activity_time_centroid,"
    "um_minus_spot_activity_time_centroid,spot_flow_time_centroid,"
    "um_flow_time_centroid,um_minus_spot_flow_time_centroid,"
    "spot_return_time_centroid,um_return_time_centroid,"
    "um_minus_spot_return_time_centroid,spot_to_um_lagged_flow_response_bp,"
    "um_to_spot_lagged_flow_response_bp,lagged_flow_response_diff_bp,"
    "spot_to_um_lagged_directional_alignment,"
    "um_to_spot_lagged_directional_alignment,lagged_directional_alignment_diff,"
    "flow_transfer_asymmetry,return_leadership_asymmetry,"
    "reverse_spot_to_um_lagged_flow_response_bp,"
    "reverse_um_to_spot_lagged_flow_response_bp,reverse_lagged_flow_response_diff_bp,"
    "reverse_spot_to_um_lagged_directional_alignment,"
    "reverse_um_to_spot_lagged_directional_alignment,"
    "reverse_lagged_directional_alignment_diff,reverse_flow_transfer_asymmetry,"
    "reverse_return_leadership_asymmetry,simultaneous_flow_sign_agreement,"
    "simultaneous_return_sign_agreement,open_basis_bp,close_basis_bp,basis_change_bp,"
    "log_spot_um_quote_ratio,source_complete,cross_venue_feature_valid,"
    "feature_invalid_reason"
    .split(",")
)
LEADERSHIP_ALLOWLIST = (
    "date",
    "feature_available_time_utc",
    "spot_quote_notional",
    "um_quote_notional",
    "spot_signed_quote_notional",
    "um_signed_quote_notional",
    "spot_to_um_lagged_flow_response_bp",
    "um_to_spot_lagged_flow_response_bp",
    "open_basis_bp",
    "close_basis_bp",
    "source_complete",
    "cross_venue_feature_valid",
)

AGGTRADE_SOURCE = (
    "data/binance_um_aggtrade_microstructure_btc_2020_2023/"
    "BTCUSDT_aggtrade_5m_2020-01-01_2023-12-31.csv.gz"
)
AGGTRADE_SOURCE_SHA256 = (
    "c2bb0e6742f8cdc4e13315e7f0a13d6ab9cd536fb40d9cb4484b7a6ba30131cf"
)
AGGTRADE_HEADER_SHA256 = (
    "fbdbd489b8d0b01262a8f8c73f19ea0ecf4dfb0de86040c1f2933e0374ea2507"
)
AGGTRADE_MANIFEST = (
    "data/binance_um_aggtrade_microstructure_btc_2020_2023/build_manifest.json"
)
AGGTRADE_MANIFEST_SHA256 = (
    "6eec40460a6146c58994e52f1af9ace4eecc0c085887d97af5ef17c30b9f7e73"
)
AGGTRADE_PHYSICAL_HEADER = tuple(
    "date,first_transact_time_ms,last_transact_time_ms,agg_trade_count,"
    "underlying_trade_count,base_volume,quote_notional,buy_quote_notional,"
    "sell_quote_notional,signed_quote_notional,flow_coherence,first_price,"
    "last_price,micro_log_return,signed_price_response,event_notional_mean,"
    "event_notional_std,event_notional_p50,event_notional_p90,event_notional_p99,"
    "event_notional_max,event_notional_hhi,normalized_effective_event_count,"
    "underlying_trades_per_agg_event,signed_event_imbalance,sign_flip_rate,"
    "mean_same_sign_run_length,max_same_sign_run_share,interarrival_mean_ms,"
    "interarrival_std_ms,interarrival_burstiness,buy_sell_event_size_log_ratio"
    .split(",")
)
AGGTRADE_ALLOWLIST = (
    "date",
    "first_transact_time_ms",
    "last_transact_time_ms",
    "agg_trade_count",
    "quote_notional",
    "signed_quote_notional",
    "micro_log_return",
    "event_notional_hhi",
    "normalized_effective_event_count",
    "sign_flip_rate",
    "max_same_sign_run_share",
    "interarrival_mean_ms",
    "interarrival_burstiness",
)

PREMIUM_SOURCE = (
    "data/binance_um_premium_path_btc_2020_2026/"
    "BTCUSDT_premium_path_1m_2020-01-01_2026-06-30.csv.gz"
)
PREMIUM_SOURCE_SHA256 = (
    "7fbaae1f85482b9fc9e148af357c7315e4e7fc4b4e3ae36c31f27545109f8aa9"
)
PREMIUM_HEADER_SHA256 = (
    "8efbf5700dc24aadf216da08b3c74712ceaec0b1a52e21ef01a877b5fbe26274"
)
PREMIUM_PHYSICAL_HEADER = (
    "date",
    "source_close_time",
    "feature_available_time",
    "source_valid",
    "premium_open",
    "premium_high",
    "premium_low",
    "premium_close",
)
PREMIUM_ALLOWLIST = PREMIUM_PHYSICAL_HEADER

PRE2024_CUTS = {
    "leadership": "data/tracer4_source_cuts/pre2024/leadership.csv.gz",
    "aggtrade": "data/tracer4_source_cuts/pre2024/aggtrade.csv.gz",
    "premium": "data/tracer4_source_cuts/pre2024/premium.csv.gz",
}
SOURCE_CUT_MANIFEST = (
    "results/tracer4_source_cut_manifest_pre2024_2026-07-25.json"
)
SUPPORT_OUTPUT = (
    "results/tracer4_tri_surface_relational_executor_support_2026-07-25.json"
)
TOKEN_OUTPUT = "data/tracer4_source_cuts/pre2024/token_support.csv.gz"

TOKEN_SCHEMA: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("sponsor", ("CASH_LEADS", "LEVERAGE_LEADS", "BALANCED")),
    (
        "flow_consensus",
        (
            "CONSENSUS_BUY",
            "CONSENSUS_SELL",
            "CASH_BUY_LEVERAGE_SELL",
            "CASH_SELL_LEVERAGE_BUY",
            "FLOW_NEUTRAL",
        ),
    ),
    (
        "impact_relation",
        (
            "BUY_FOLLOWTHROUGH",
            "BUY_ABSORBED",
            "SELL_FOLLOWTHROUGH",
            "SELL_ABSORBED",
            "RESPONSE_NEUTRAL",
        ),
    ),
    ("participation", ("BROAD", "MIXED", "CONCENTRATED")),
    ("flow_persistence", ("PERSISTENT", "ROTATING", "MIXED")),
    ("auction_tempo", ("BURST", "STEADY", "SLOW")),
    (
        "premium_price_relation",
        (
            "CROWDING_CONFIRMS_UP",
            "CROWDING_CONFIRMS_DOWN",
            "PREMIUM_DIVERGES_FROM_UP",
            "PREMIUM_DIVERGES_FROM_DOWN",
            "PREMIUM_NEUTRAL",
        ),
    ),
    (
        "basis_premium_relation",
        (
            "BOTH_EXPAND",
            "BOTH_COMPRESS",
            "BASIS_ONLY",
            "PREMIUM_ONLY",
            "CROSS_DISAGREE",
            "BOTH_NEUTRAL",
        ),
    ),
    (
        "sponsor_transition",
        (
            "STABLE_CASH",
            "STABLE_LEVERAGE",
            "ROTATED_TO_CASH",
            "ROTATED_TO_LEVERAGE",
            "SPONSOR_MIXED",
        ),
    ),
    (
        "impact_transition",
        (
            "FOLLOWTHROUGH_PERSISTS",
            "ABSORPTION_PERSISTS",
            "FOLLOWTHROUGH_TO_ABSORPTION",
            "ABSORPTION_TO_FOLLOWTHROUGH",
            "IMPACT_MIXED",
        ),
    ),
    (
        "crowding_transition",
        (
            "CROWDING_BUILDS",
            "CROWDING_RELEASES",
            "CROWDING_FLIPS",
            "CROWDING_STABLE",
        ),
    ),
)
TOKEN_COLUMNS = tuple(name for name, _ in TOKEN_SCHEMA)
TOKEN_VOCABULARY = {name: values for name, values in TOKEN_SCHEMA}
SAFETY_TOKENS = (
    "SOURCE_INVALID",
    "FLOW_NEUTRAL",
    "RESPONSE_NEUTRAL",
    "MIXED",
    "MIXED",
    "STEADY",
    "PREMIUM_NEUTRAL",
    "BOTH_NEUTRAL",
    "SPONSOR_MIXED",
    "IMPACT_MIXED",
    "CROWDING_STABLE",
)
POSITION_TOKENS = ("SHORT", "FLAT", "LONG")
BAND_TOKENS = ("LOW", "MID", "HIGH")
CONTROL_IDS = (
    "premium_stale_1440m",
    "cash_perpetual_swap",
    "aggtrade_monthly_rotate_37_rows",
)
FORBIDDEN_SUPPORT_FIELDS = (
    "execution_open",
    "execution_high",
    "execution_low",
    "execution_close",
    "funding_rate",
    "future_return",
    "label",
    "action",
    "target",
    "reward",
    "pnl",
    "cagr",
    "mdd",
    "strict_mdd",
    "portfolio_weight",
    "prior_model_prediction",
)


@dataclass(frozen=True)
class Policy:
    policy_id: str = POLICY_ID
    boundary_hours_utc: tuple[int, ...] = (0, 4, 8, 12, 16, 20)
    source_window_minutes: int = 240
    five_minute_rows: int = 48
    premium_rows: int = 240
    premium_cutoff_seconds: int = 61
    decision_delay_minutes: int = 5
    execution_delay_minutes: int = 10
    sequence_lines: int = 3
    rank_history_max: int = 540
    rank_history_min: int = 360
    rank_quantiles: tuple[float, float] = (0.33, 0.67)
    source_join_min: float = 0.99
    core_valid_min: float = 0.95
    source_invalid_max: float = 0.05
    distinct_signatures_min: int = 300
    signature_share_max: float = 0.05
    category_support_min: float = 0.05
    category_share_max: float = 0.90
    jsd_max: float = 0.25
    control_premium_shift_minutes: int = 1_440
    control_aggtrade_rotate_rows: int = 37
    leverage: float = 0.5
    base_cost_per_changed_notional: float = 0.0006
    stress_cost_per_changed_notional: float = 0.0010
    random_seed: int = 20_260_725


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


def jsonable(payload: Any) -> Any:
    return json.loads(
        json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
    )


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def state_times(boundary: str | pd.Timestamp) -> dict[str, pd.Timestamp]:
    current = _utc(boundary)
    policy = Policy()
    if (
        current.minute != 0
        or current.second != 0
        or current.microsecond != 0
        or current.hour not in policy.boundary_hours_utc
    ):
        raise ValueError("TRACER boundary is not a canonical four-hour UTC boundary")
    return {
        "window_start": current - pd.Timedelta(minutes=policy.source_window_minutes),
        "window_end": current,
        "five_minute_cutoff": current,
        "premium_cutoff": current + pd.Timedelta(seconds=policy.premium_cutoff_seconds),
        "decision_time": current + pd.Timedelta(minutes=policy.decision_delay_minutes),
        "execution_time": current + pd.Timedelta(minutes=policy.execution_delay_minutes),
        "next_execution_time": current
        + pd.Timedelta(hours=4, minutes=policy.execution_delay_minutes),
    }


def strict_prior_band(
    current: float,
    history: Sequence[float],
    *,
    minimum: int = Policy().rank_history_min,
    maximum: int = Policy().rank_history_max,
) -> str:
    value = float(current)
    prior = np.asarray(tuple(float(item) for item in history[-maximum:]), dtype=np.float64)
    if not math.isfinite(value) or not np.isfinite(prior).all():
        raise ValueError("TRACER rank values must be finite")
    if minimum <= 0 or maximum < minimum:
        raise ValueError("TRACER rank history bounds are invalid")
    if len(prior) < minimum:
        raise ValueError("TRACER strict-prior history is not ready")
    low, high = np.quantile(
        prior,
        np.asarray(Policy().rank_quantiles, dtype=np.float64),
        method="linear",
    )
    if value <= low:
        return "LOW"
    if value <= high:
        return "MID"
    return "HIGH"


def validate_tokens(tokens: Mapping[str, str]) -> dict[str, str]:
    if tuple(tokens) != TOKEN_COLUMNS:
        raise ValueError("TRACER token order or schema changed")
    normalized = {name: str(tokens[name]) for name in TOKEN_COLUMNS}
    for name, value in normalized.items():
        if value not in TOKEN_VOCABULARY[name]:
            raise ValueError(f"TRACER token is invalid: {name}={value}")
    return normalized


def canonical_line(tokens: Mapping[str, str]) -> str:
    normalized = validate_tokens(tokens)
    return " | ".join(f"{name}={normalized[name]}" for name in TOKEN_COLUMNS)


def safety_line() -> str:
    return "|".join(SAFETY_TOKENS)


def jensen_shannon_divergence(
    left: Mapping[str, float],
    right: Mapping[str, float],
    vocabulary: Sequence[str],
) -> float:
    values = tuple(vocabulary)
    if not values or len(set(values)) != len(values):
        raise ValueError("TRACER JSD vocabulary must be unique and nonempty")
    p = np.asarray([float(left.get(value, 0.0)) for value in values], dtype=np.float64)
    q = np.asarray([float(right.get(value, 0.0)) for value in values], dtype=np.float64)
    if not np.isfinite(p).all() or not np.isfinite(q).all() or (p < 0).any() or (q < 0).any():
        raise ValueError("TRACER JSD weights must be finite and nonnegative")
    if p.sum() <= 0.0 or q.sum() <= 0.0:
        raise ValueError("TRACER JSD distributions must have positive mass")
    p /= p.sum()
    q /= q.sum()
    midpoint = 0.5 * (p + q)

    def _kl(values_: np.ndarray) -> float:
        active = values_ > 0.0
        return float(np.sum(values_[active] * np.log2(values_[active] / midpoint[active])))

    return 0.5 * (_kl(p) + _kl(q))


def _source_contract(
    *,
    path: str,
    sha256: str,
    header: Sequence[str],
    header_sha256: str,
    allowlist: Sequence[str],
    output: str,
    manifest: str | None = None,
    manifest_sha256: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "path": path,
        "sha256": sha256,
        "physical_header": list(header),
        "physical_header_sha256": header_sha256,
        "cut_allowlist": list(allowlist),
        "cut_output": output,
        "physical_projection_required": True,
        "load_all_then_drop_forbidden": True,
        "source_start": SOURCE_START,
        "source_end_exclusive": SOURCE_END_EXCLUSIVE,
        "post_2023_numeric_conversion_forbidden": True,
    }
    if manifest is not None:
        payload["manifest"] = manifest
        payload["manifest_sha256"] = manifest_sha256
    return payload


def build_manifest() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "boundary": {
            "document": BOUNDARY_DOCUMENT,
            "document_sha256": BOUNDARY_DOCUMENT_SHA256,
            "commit": BOUNDARY_COMMIT,
        },
        "research_history_boundary": {
            "component_family_outcomes_seen": True,
            "tracer_source_values_seen_before_boundary": False,
            "single_clock_timestamp_checked_after_boundary": True,
            "tracer_token_incidence_seen": False,
            "tracer_rewards_seen": False,
            "tracer_model_outcomes_seen": False,
            "global_pristine_holdout_claimed": False,
            "claim_scope": "contaminated-history candidate-specific composition and annual transfer",
        },
        "policy": asdict(Policy()),
        "sources": {
            "leadership": _source_contract(
                path=LEADERSHIP_SOURCE,
                sha256=LEADERSHIP_SOURCE_SHA256,
                header=LEADERSHIP_PHYSICAL_HEADER,
                header_sha256=LEADERSHIP_HEADER_SHA256,
                allowlist=LEADERSHIP_ALLOWLIST,
                output=PRE2024_CUTS["leadership"],
                manifest=LEADERSHIP_MANIFEST,
                manifest_sha256=LEADERSHIP_MANIFEST_SHA256,
            ),
            "aggtrade": _source_contract(
                path=AGGTRADE_SOURCE,
                sha256=AGGTRADE_SOURCE_SHA256,
                header=AGGTRADE_PHYSICAL_HEADER,
                header_sha256=AGGTRADE_HEADER_SHA256,
                allowlist=AGGTRADE_ALLOWLIST,
                output=PRE2024_CUTS["aggtrade"],
                manifest=AGGTRADE_MANIFEST,
                manifest_sha256=AGGTRADE_MANIFEST_SHA256,
            ),
            "premium": _source_contract(
                path=PREMIUM_SOURCE,
                sha256=PREMIUM_SOURCE_SHA256,
                header=PREMIUM_PHYSICAL_HEADER,
                header_sha256=PREMIUM_HEADER_SHA256,
                allowlist=PREMIUM_ALLOWLIST,
                output=PRE2024_CUTS["premium"],
            ),
        },
        "physical_cuts": {
            "paths": PRE2024_CUTS,
            "manifest": SOURCE_CUT_MANIFEST,
            "token_output": TOKEN_OUTPUT,
            "gzip_mtime": 0,
            "utf8": True,
            "line_ending": "LF",
            "write_once": True,
            "all_support_reads_cut_only": True,
        },
        "clock": {
            "source_window": "[B-4h,B)",
            "five_minute_cutoff": "B",
            "premium_cutoff": "B+61s",
            "decision_time": "B+5m",
            "execution_time": "B+10m",
            "premium_availability": "date+61s",
            "premium_date_at_or_after_boundary_forbidden": True,
        },
        "tokens": {
            "schema": [
                {"name": name, "vocabulary": list(values)}
                for name, values in TOKEN_SCHEMA
            ],
            "columns": list(TOKEN_COLUMNS),
            "safety_tokens": list(SAFETY_TOKENS),
            "position_tokens": list(POSITION_TOKENS),
            "band_tokens": list(BAND_TOKENS),
            "sequence_lines": Policy().sequence_lines,
            "raw_numeric_model_input_forbidden": True,
        },
        "support_gates": {
            "source_join_min_each_year": Policy().source_join_min,
            "core_valid_min_each_year": Policy().core_valid_min,
            "source_invalid_max_each_year": Policy().source_invalid_max,
            "sequence_ready_min": {"2020": 1500, "2021": 2000, "2022": 2000, "2023": 2000},
            "quarter_ready_min_after_warmup": 450,
            "category_support_min": Policy().category_support_min,
            "category_share_max": Policy().category_share_max,
            "flow_buy_and_sell_min": 0.15,
            "impact_follow_and_absorb_min": 0.10,
            "sponsor_cash_and_leverage_min": 0.10,
            "distinct_signatures_min": Policy().distinct_signatures_min,
            "signature_share_max": Policy().signature_share_max,
            "adjacent_year_jsd_max": Policy().jsd_max,
            "denominator": "sequence-ready core-valid states in named UTC year",
            "safety_and_position_excluded": True,
            "append_replay_required": True,
            "all_gates_conjunctive": True,
        },
        "controls": {
            "ids": list(CONTROL_IDS),
            "score_bearing_later": True,
            "may_replace_primary": False,
        },
        "stage_authority": {
            "authorized": ["source_cut", "primitive", "rank", "token_support"],
            "forbidden": [
                "execution_market",
                "funding",
                "future_return",
                "reward",
                "model_training",
                "policy_selection",
                "economic_evaluation",
            ],
            "support_pass_required_for_stage_0_5": True,
        },
        "temporal_roles": {
            "fit": ["2020-01-01T00:00:00Z", "2021-01-01T00:00:00Z"],
            "test": ["2021-01-01T00:00:00Z", "2022-01-01T00:00:00Z"],
            "eval": ["2022-01-01T00:00:00Z", "2023-01-01T00:00:00Z"],
            "confirmation": ["2023-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "later_outcomes_conditionally_opened": True,
            "failed_year_may_be_repaired": False,
            "selected_algorithm_changes_after_2021": False,
            "annual_refit": "rolling one-year prior-only",
        },
        "stage_0_5_requirements": {
            "freeze_before_reward": True,
            "algorithm_family_and_grid": True,
            "seeds_and_tie_breaks": True,
            "reward_and_bellman_target": True,
            "cost_funding_mdd_delay": True,
            "familywise_inference": True,
            "killer_baselines": True,
            "prior_clock_novelty": True,
            "clean_committed_runner": True,
            "write_once_schedules": True,
        },
        "forbidden_support_fields": list(FORBIDDEN_SUPPORT_FIELDS),
        "outcome_boundary": {
            "execution_market_rows_opened": 0,
            "funding_rows_opened": 0,
            "future_return_rows_opened": 0,
            "reward_rows_built": 0,
            "model_rows_built": 0,
            "pnl_values_computed": 0,
            "cagr_values_computed": 0,
            "mdd_values_computed": 0,
            "post_2023_numeric_source_rows_opened": 0,
        },
        "support_output": SUPPORT_OUTPUT,
    }
    payload = jsonable(payload)
    payload["manifest_hash"] = canonical_hash(payload)
    return payload


def validate_manifest(payload: Mapping[str, Any]) -> None:
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("TRACER preregistration protocol drift")
    if payload.get("boundary") != {
        "document": BOUNDARY_DOCUMENT,
        "document_sha256": BOUNDARY_DOCUMENT_SHA256,
        "commit": BOUNDARY_COMMIT,
    }:
        raise ValueError("TRACER boundary binding drift")
    if payload.get("policy") != jsonable(asdict(Policy())):
        raise ValueError("TRACER policy drift")
    tokens = payload.get("tokens", {})
    if tokens.get("columns") != list(TOKEN_COLUMNS):
        raise ValueError("TRACER token schema drift")
    authority = payload.get("stage_authority", {})
    if authority.get("authorized") != ["source_cut", "primitive", "rank", "token_support"]:
        raise ValueError("TRACER stage authority drift")
    if any(value != 0 for value in payload.get("outcome_boundary", {}).values()):
        raise ValueError("TRACER preregistration opened an outcome")
    for field in FORBIDDEN_SUPPORT_FIELDS:
        if field in TOKEN_COLUMNS:
            raise ValueError(f"TRACER forbidden field entered token schema: {field}")
    expected_hash = canonical_hash({key: value for key, value in payload.items() if key != "manifest_hash"})
    if payload.get("manifest_hash") != expected_hash:
        raise ValueError("TRACER preregistration manifest hash drift")


def assert_boundary_committed() -> None:
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", BOUNDARY_DOCUMENT],
        text=True,
        capture_output=True,
        check=False,
    )
    if tracked.returncode:
        raise RuntimeError("TRACER boundary is not committed")
    clean = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", BOUNDARY_DOCUMENT],
        check=False,
    )
    if clean.returncode:
        raise RuntimeError("TRACER boundary differs from HEAD")
    if sha256_file(BOUNDARY_DOCUMENT) != BOUNDARY_DOCUMENT_SHA256:
        raise RuntimeError("TRACER boundary hash drift")
    latest = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", BOUNDARY_DOCUMENT],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    if latest != BOUNDARY_COMMIT:
        raise RuntimeError("TRACER boundary commit drift")


def write_once(path: str | Path, payload: Mapping[str, Any]) -> str:
    target = Path(path)
    encoded = (
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    if target.exists():
        if target.read_bytes() != encoded:
            raise RuntimeError(f"TRACER write-once artifact drift: {target}")
        return hashlib.sha256(encoded).hexdigest()
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
    temporary.replace(target)
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    assert_boundary_committed()
    payload = build_manifest()
    validate_manifest(payload)
    artifact_hash = write_once(args.output, payload)
    print(json.dumps({"output": args.output, "sha256": artifact_hash, "manifest_hash": payload["manifest_hash"]}, sort_keys=True))


if __name__ == "__main__":
    main()
