"""Freeze SABLE-8 source/language support before decoding candidate values."""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Any


POLICY_ID = "SABLE-8"
PROTOCOL_VERSION = "sable8_symbolic_alpha_braid_preregistration_v1"
DEFAULT_OUTPUT = (
    "results/sable8_symbolic_alpha_braid_preregistration_2026-07-25.json"
)

BOUNDARY_DOCUMENT = (
    "docs/sable8-symbolic-alpha-braid-boundary-2026-07-25.md"
)
BOUNDARY_DOCUMENT_SHA256 = (
    "ac06223155575405a12b5ed59023aea43dfbfacda680ba36ba449b07a8e36acc"
)
BOUNDARY_COMMIT = "4cbce55"

MARKET_SOURCE = (
    "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_oi.csv.gz"
)
MARKET_SOURCE_SHA256 = (
    "dbc9e53b09551b469168fe19cc750c5c3ea86278db3055d079103f7654050192"
)
MARKET_ALLOWLIST = (
    "date",
    "open",
    "high",
    "low",
    "close",
    "quote_asset_volume",
    "taker_buy_quote",
    "dxy",
    "kimchi_premium",
    "usdkrw",
    "dxy_available",
    "kimchi_available",
    "usdkrw_available",
    "open_interest",
    "open_interest_available",
)

FUNDING_SOURCE = (
    "data/binance_um_aux_btc_2020_2026/"
    "BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz"
)
FUNDING_SOURCE_SHA256 = (
    "4d381be086e275bacaf31df431dc31307a71a26b3947b7082efffc10bb129dd7"
)
FUNDING_ALLOWLIST = ("date", "funding_rate", "funding_time")

PREMIUM_SOURCE = (
    "data/binance_um_aux_btc_2020_2026/"
    "BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz"
)
PREMIUM_SOURCE_SHA256 = (
    "b45fcc5a3cf75c8e594effe61a698c4652f841b1d304107e9669524e0fc9d0d7"
)
PREMIUM_ALLOWLIST = ("date", "close", "close_time")

SOURCE_END_EXCLUSIVE = "2024-01-01T00:00:00Z"
PRE2024_CUTS = {
    "market": "data/sable8_source_cuts/pre2024/market.csv.gz",
    "funding": "data/sable8_source_cuts/pre2024/funding.csv.gz",
    "premium": "data/sable8_source_cuts/pre2024/premium.csv.gz",
}
SOURCE_CUT_MANIFEST = (
    "results/sable8_source_cut_manifest_pre2024_2026-07-25.json"
)
SUPPORT_OUTPUT = (
    "results/sable8_symbolic_alpha_braid_support_2026-07-25.json"
)

PRIMITIVES = (
    "price_return_1d",
    "range_location_7d",
    "volatility_ratio_1d_30d",
    "jump_share_1d",
    "signed_jump_1d",
    "volume_clock_flow_speed_25",
    "liquidity_signed_efficiency_6h",
    "taker_flow_recovery_1h_6h",
    "funding_sum_24h",
    "premium_mean_8h",
    "oi_price_divergence_1d",
    "kimchi_change_12h",
    "usdkrw_change_12h",
    "dxy_change_1d",
)
CORE_PRIMITIVES = PRIMITIVES[:10]
CONTEXT_PRIMITIVES = PRIMITIVES[10:]
BANDS = (
    "EXTREME_LOW",
    "LOW",
    "MIDDLE",
    "HIGH",
    "EXTREME_HIGH",
)
CONTEXT_BANDS = (*BANDS, "STALE")
POSITION_TOKENS = ("SHORT", "FLAT", "LONG")
POSITION_AGE_TOKENS = ("ZERO", "ONE", "TWO", "THREE_PLUS")
DRAWDOWN_TOKENS = ("ZERO", "UNDER_2", "TWO_TO_5", "OVER_5")
FORBIDDEN_INPUT_FIELDS = (
    "timestamp",
    "date",
    "year",
    "month",
    "weekday",
    "hour",
    "row_id",
    "split",
    "future_return",
    "label",
    "reward",
    "pnl",
    "cagr",
    "mdd",
    "oracle_action",
    "prior_alpha",
    "prior_prediction",
    "portfolio_weight",
    "manual_regime",
)


@dataclass(frozen=True)
class Policy:
    policy_id: str = POLICY_ID
    boundary_hours_utc: tuple[int, ...] = (0, 8, 16)
    bar_minutes: int = 5
    state_delay_minutes: int = 5
    decision_delay_minutes: int = 10
    execution_delay_minutes: int = 15
    sequence_lines: int = 6
    rank_history_max: int = 540
    rank_history_min: int = 180
    leverage: float = 0.5
    base_cost_per_changed_notional: float = 0.0006
    stress_cost_per_changed_notional: float = 0.0010


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


def strict_prior_midrank(
    current: float,
    history: Sequence[float],
    *,
    minimum: int = Policy().rank_history_min,
    maximum: int = Policy().rank_history_max,
) -> float:
    """Return the frozen midrank using only the capped strict-prior history."""

    if not math.isfinite(float(current)):
        raise ValueError("SABLE current rank value must be finite")
    if minimum <= 0 or maximum < minimum:
        raise ValueError("SABLE rank history bounds are invalid")
    prior = [float(value) for value in history[-maximum:]]
    if len(prior) < minimum:
        raise ValueError("SABLE strict-prior history is not ready")
    if not all(math.isfinite(value) for value in prior):
        raise ValueError("SABLE prior rank values must be finite")
    lower = sum(value < current for value in prior)
    equal = sum(value == current for value in prior)
    return (lower + 0.5 * equal) / len(prior)


def rank_band(rank: float) -> str:
    """Map one finite [0,1] strict-prior rank to the frozen five-band token."""

    value = float(rank)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("SABLE rank must be finite and inside [0,1]")
    if value < 0.2:
        return "EXTREME_LOW"
    if value < 0.4:
        return "LOW"
    if value <= 0.6:
        return "MIDDLE"
    if value <= 0.8:
        return "HIGH"
    return "EXTREME_HIGH"


def validate_token_line(tokens: Mapping[str, str]) -> dict[str, str]:
    if tuple(tokens) != PRIMITIVES:
        raise ValueError("SABLE token line order or schema differs")
    validated: dict[str, str] = {}
    for primitive in CORE_PRIMITIVES:
        value = str(tokens[primitive])
        if value not in BANDS:
            raise ValueError(f"SABLE core token is invalid: {primitive}")
        validated[primitive] = value
    for primitive in CONTEXT_PRIMITIVES:
        value = str(tokens[primitive])
        if value not in CONTEXT_BANDS:
            raise ValueError(f"SABLE context token is invalid: {primitive}")
        validated[primitive] = value
    return validated


def canonical_line(tokens: Mapping[str, str]) -> str:
    validated = validate_token_line(tokens)
    return " | ".join(
        f"{primitive.upper()}={validated[primitive]}"
        for primitive in PRIMITIVES
    )


def sequence_signature(
    boundaries: Sequence[int],
    lines: Sequence[str],
    *,
    boundary_seconds: int = 8 * 60 * 60,
) -> str:
    """Hash six consecutive canonical lines without exposing their timestamps."""

    if len(boundaries) != Policy().sequence_lines:
        raise ValueError("SABLE sequence must contain exactly six boundaries")
    if len(lines) != len(boundaries):
        raise ValueError("SABLE sequence line count differs")
    times = [int(value) for value in boundaries]
    if any(
        right - left != boundary_seconds
        for left, right in zip(times, times[1:])
    ):
        raise ValueError("SABLE sequence boundaries are not consecutive")
    if any(not value or "\n" in value for value in lines):
        raise ValueError("SABLE sequence lines must be nonempty single lines")
    return canonical_hash({"lines": list(lines)})


def _source_contract(
    *,
    path: str,
    sha256: str,
    allowlist: Sequence[str],
    timestamp_field: str,
) -> dict[str, Any]:
    return {
        "path": path,
        "sha256": sha256,
        "allowlist": list(allowlist),
        "loader": "stream csv.DictReader; reject extra/missing/reordered fields",
        "timestamp_field": timestamp_field,
        "cutoff_exclusive": SOURCE_END_EXCLUSIVE,
        "physical_stop_before_other_field_conversion": True,
    }


def _manifest_core() -> dict[str, Any]:
    policy = asdict(Policy())
    policy["boundary_hours_utc"] = list(policy["boundary_hours_utc"])
    return {
        "protocol_version": PROTOCOL_VERSION,
        "policy": policy,
        "boundary": {
            "path": BOUNDARY_DOCUMENT,
            "sha256": BOUNDARY_DOCUMENT_SHA256,
            "commit": BOUNDARY_COMMIT,
        },
        "research_history_boundary": {
            "component_family_outcomes_seen": True,
            "sable_source_values_seen": False,
            "sable_token_incidence_seen": False,
            "sable_rewards_seen": False,
            "sable_model_outcomes_seen": False,
            "global_pristine_holdout_claimed": False,
            "claim_scope": "contaminated-history candidate research MDP",
        },
        "sources": {
            "market": _source_contract(
                path=MARKET_SOURCE,
                sha256=MARKET_SOURCE_SHA256,
                allowlist=MARKET_ALLOWLIST,
                timestamp_field="date",
            ),
            "funding": _source_contract(
                path=FUNDING_SOURCE,
                sha256=FUNDING_SOURCE_SHA256,
                allowlist=FUNDING_ALLOWLIST,
                timestamp_field="funding_time",
            ),
            "premium": _source_contract(
                path=PREMIUM_SOURCE,
                sha256=PREMIUM_SOURCE_SHA256,
                allowlist=PREMIUM_ALLOWLIST,
                timestamp_field="close_time",
            ),
        },
        "physical_cuts": {
            "paths": PRE2024_CUTS,
            "manifest": SOURCE_CUT_MANIFEST,
            "gzip_mtime": 0,
            "newline": "LF",
            "encoding": "UTF-8",
            "all_support_reads_cut_only": True,
        },
        "primitive_contract": {
            "order": list(PRIMITIVES),
            "core": list(CORE_PRIMITIVES),
            "context": list(CONTEXT_PRIMITIVES),
            "equations": {
                "price_return_1d": "log(c_t/c_t-288)",
                "range_location_7d": (
                    "(c_t-min(c_t-2015..t))/(max(c_t-2015..t)-min(...))"
                ),
                "volatility_ratio_1d_30d": (
                    "log((sum(r^2,t-287..t)+1e-18)/"
                    "(sum(r^2,t-8639..t)/30+1e-18))"
                ),
                "jump_share_1d": "max(RV_1d-BV_1d,0)/RV_1d",
                "signed_jump_1d": "sum(r^3,t-287..t)/RV_1d^(3/2)",
                "volume_clock_flow_speed_25": (
                    "(sum(a,j..t)/sum(q,j..t))/(t-j+1); "
                    "target=.25*sum(q,t-288..t-1)"
                ),
                "liquidity_signed_efficiency_6h": (
                    "log(c_t/c_t-72)/sum(abs(r),t-71..t)"
                ),
                "taker_flow_recovery_1h_6h": (
                    "mean(a/q,t-11..t)-mean(a/q,t-71..t)"
                ),
                "funding_sum_24h": (
                    "sum(funding_rate,cutoff-24h<funding_time<=cutoff)"
                ),
                "premium_mean_8h": (
                    "mean(close,cutoff-8h<close_time<=cutoff)"
                ),
                "oi_price_divergence_1d": (
                    "log(oi_t/oi_t-288)-price_return_1d"
                ),
                "kimchi_change_12h": "kimchi_t-kimchi_t-144",
                "usdkrw_change_12h": "log(usdkrw_t/usdkrw_t-144)",
                "dxy_change_1d": "log(dxy_t/dxy_t-288)",
            },
        },
        "language_contract": {
            "rank_history_max": Policy().rank_history_max,
            "rank_history_min": Policy().rank_history_min,
            "strictly_prior": True,
            "midrank_ties": True,
            "bands": list(BANDS),
            "context_stale_token": "STALE",
            "sequence_lines": Policy().sequence_lines,
            "consecutive_boundary_seconds": 28_800,
            "oldest_first": True,
            "invalid_sequence_action": "FLAT",
            "forbidden_input_fields": list(FORBIDDEN_INPUT_FIELDS),
        },
        "support_gates": {
            "development_2020_2022_token_ready_min": 3_000,
            "each_year_2021_2022_token_ready_min": 900,
            "report_only_2023_token_ready_min": 900,
            "active_months": ["2020-05", "2022-12"],
            "core_missing_share_max": 0.01,
            "occupied_core_bands_min": 4,
            "occupied_oi_bands_min": 4,
            "largest_core_band_share_max": 0.45,
            "oi_fresh_share_min": 0.50,
            "kimchi_fresh_share_min": 0.80,
            "usdkrw_fresh_share_min": 0.55,
            "usdkrw_stale_share_min": 0.05,
            "dxy_fresh_share_min": 0.55,
            "dxy_stale_share_min": 0.05,
            "adjacent_state_change_share_min": 0.95,
            "max_exact_sequence_share": 0.01,
            "prefix_replay_required": True,
            "failure_action": "retire_without_reward_construction",
        },
        "temporal_roles": {
            "fit": ["2020-01-01T00:00:00Z", "2021-01-01T00:00:00Z"],
            "internal_selection": [
                "2021-01-01T00:00:00Z",
                "2022-01-01T00:00:00Z",
            ],
            "confirmation": [
                "2022-01-01T00:00:00Z",
                "2023-01-01T00:00:00Z",
            ],
            "candidate_gate": [
                "2023-01-01T00:00:00Z",
                "2024-01-01T00:00:00Z",
            ],
            "candidate_gate_may_select": False,
        },
        "stage_authority": {
            "authorized": ["source_cut", "primitive", "rank", "token_support"],
            "forbidden": [
                "forward_return",
                "reward",
                "action_label",
                "model_training",
                "economic_metric",
                "post_2023_numeric_source",
            ],
            "next_required_commit": "stage_0_5_evaluator_and_model_freeze",
        },
        "outputs": {
            "preregistration": DEFAULT_OUTPUT,
            "source_cut_manifest": SOURCE_CUT_MANIFEST,
            "support": SUPPORT_OUTPUT,
        },
        "outcome_boundary": {
            "candidate_source_values_decoded": 0,
            "candidate_token_incidence_calculated": 0,
            "future_return_labels_built": 0,
            "rewards_built": 0,
            "market_outcomes_evaluated": 0,
            "funding_cash_flows_evaluated": 0,
            "models_trained": 0,
            "post_2023_numeric_source_rows_parsed": 0,
            "candidate_2023_outcomes_opened": False,
            "candidate_2024_outcomes_opened": False,
            "candidate_2025_outcomes_opened": False,
            "candidate_2026_outcomes_opened": False,
        },
    }


def build_manifest() -> dict[str, Any]:
    payload = _manifest_core()
    return {**payload, "manifest_hash": canonical_hash(payload)}


def validate_manifest(payload: Mapping[str, Any]) -> None:
    expected = build_manifest()
    if dict(payload) != expected:
        raise ValueError("SABLE preregistration differs from frozen contract")
    if payload["manifest_hash"] != canonical_hash(_manifest_core()):
        raise ValueError("SABLE preregistration manifest hash mismatch")


def write_once(path: str | Path, payload: Mapping[str, Any]) -> str:
    target = Path(path)
    encoded = (
        json.dumps(
            dict(payload),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if target.exists():
        if target.read_bytes() != encoded:
            raise RuntimeError(f"SABLE write-once artifact drift: {target}")
        return hashlib.sha256(encoded).hexdigest()
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=target.parent,
        prefix=f".{target.name}.",
        delete=False,
    ) as handle:
        handle.write(encoded)
        temporary = Path(handle.name)
    temporary.replace(target)
    return hashlib.sha256(encoded).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_manifest()
    validate_manifest(payload)
    digest = write_once(args.output, payload)
    print(
        json.dumps(
            {
                "decision": "PREREGISTERED",
                "manifest_hash": payload["manifest_hash"],
                "output": str(args.output),
                "sha256": digest,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
