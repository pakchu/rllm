"""Freeze the outcome-blind BFMWD-144 candidate family.

The preregistration hashes source-contract code and documents only.  It must
not parse Bitfinex numeric rows, comparator clocks, BTC market data, funding,
labels, returns, or PnL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = "bitfinex_margin_warehouse_deployment_preregistration_v1"
CANDIDATE_FAMILY = "BFMWD-144"
AS_OF_DATE = "2026-07-20"
SOURCE_BUILDER = Path("training/download_bitfinex_margin_funding_stats.py")
SOURCE_DECISION = Path(
    "docs/bitfinex-margin-warehouse-deployment-source-decision-2026-07-20.md"
)
PREREGISTRATION_SOURCE = Path(
    "training/preregister_bitfinex_margin_warehouse_deployment.py"
)
PREREGISTRATION_DOCUMENT = Path(
    "docs/bitfinex-margin-warehouse-deployment-preregistration-2026-07-20.md"
)
DEFAULT_OUTPUT = Path(
    "results/bitfinex_margin_warehouse_deployment_preregistration_2026-07-20.json"
)


@dataclass(frozen=True)
class Variant:
    variant_id: str
    warehouse_hours: int
    deployment_hours: int
    robust_z_threshold: float = 1.0
    hold_bars: int = 144


@dataclass(frozen=True)
class Policy:
    symbols: tuple[str, ...] = ("fUSD", "fBTC")
    usd_side: int = 1
    btc_side: int = -1
    history_hours: int = 1_440
    minimum_history_hours: int = 1_080
    robust_mad_scale: float = 1.4826
    utilization_clip: float = 1e-6
    tenor_confirmation: str = "current_average_period_at_or_above_strict_prior_median"
    signal_grid: str = "hourly_official_observation"
    available_at_rule: str = "floor(observation_time, 1h) + 15m"
    entry_delay_minutes: int = 5
    bar_minutes: int = 5
    leverage: float = 0.5
    simultaneous_symbol_conflict: str = "abstain"
    onset_only: bool = True
    global_nonoverlap: bool = True
    warmup_start: str = "2020-01-01T00:00:00Z"
    train_start: str = "2021-01-01T00:00:00Z"
    train_end_exclusive: str = "2023-01-01T00:00:00Z"
    selection_start: str = "2023-01-01T00:00:00Z"
    selection_end_exclusive: str = "2024-01-01T00:00:00Z"


VARIANTS = (
    Variant("bfmwd_w12_d3_z10_h12", warehouse_hours=12, deployment_hours=3),
    Variant("bfmwd_w24_d3_z10_h12", warehouse_hours=24, deployment_hours=3),
    Variant("bfmwd_w12_d6_z10_h12", warehouse_hours=12, deployment_hours=6),
    Variant("bfmwd_w24_d6_z10_h12", warehouse_hours=24, deployment_hours=6),
)
FROZEN_POLICY = Policy()

SUPPORT_GATES = {
    "minimum_train_events": 60,
    "minimum_selection_events": 30,
    "minimum_events_each_train_year": 20,
    "minimum_events_each_selection_half": 12,
    "minimum_side_share": 0.20,
    "maximum_side_share": 0.80,
    "maximum_calendar_month_share": 0.20,
    "maximum_weekday_share": 0.25,
    "maximum_rolling_14day_share": 0.20,
    "maximum_exact_entry_jaccard": 0.10,
    "novelty_containment_hours": 6,
    "maximum_bidirectional_novelty_containment": 0.35,
    "stop_if_no_variant_passes": True,
}

ECONOMIC_GATES = {
    "base_cost_bp_per_side": 6.0,
    "stress_cost_bp_per_side": 10.0,
    "full_calendar_cagr": True,
    "strict_path_mdd": True,
    "minimum_cagr_to_strict_mdd": 3.0,
    "maximum_strict_mdd": 0.15,
    "minimum_stress_cagr_to_strict_mdd": 2.5,
    "minimum_mean_gross_side_adjusted_bp": 30.0,
    "require_positive_absolute_return": True,
    "require_each_calendar_half_positive": True,
    "require_each_side_contribution_positive": True,
    "require_one_bar_delay_positive": True,
    "weekly_cluster_pvalue_maximum": 0.10,
    "multiple_testing": "Romano-Wolf one-sided step-down max-t",
    "block_length_days": 7,
    "draws": 100_000,
    "seed": 20_260_720,
}

SOURCE_ONLY_CONTROLS = (
    "no_warehouse_charge_prerequisite",
    "no_unused_draw_confirmation",
    "no_tenor_confirmation",
    "stale_24h_source",
)

LATER_ECONOMIC_CONTROLS = (
    "direction_flip",
    "fUSD_only",
    "fBTC_only",
    "deterministic_random_side",
    "extra_latency_one_bar",
    "ten_bp_per_side_stress",
)


def _path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_contract() -> None:
    if FROZEN_POLICY.symbols != ("fUSD", "fBTC"):
        raise ValueError("BFMWD source symbols are frozen")
    if FROZEN_POLICY.usd_side != 1 or FROZEN_POLICY.btc_side != -1:
        raise ValueError("BFMWD directional interpretation is frozen")
    if FROZEN_POLICY.entry_delay_minutes != FROZEN_POLICY.bar_minutes:
        raise ValueError("BFMWD must retain one full five-minute latency bar")
    if FROZEN_POLICY.minimum_history_hours > FROZEN_POLICY.history_hours:
        raise ValueError("BFMWD minimum history exceeds the rolling lookback")
    identifiers = [variant.variant_id for variant in VARIANTS]
    if len(identifiers) != len(set(identifiers)) or len(identifiers) != 4:
        raise ValueError("BFMWD must retain exactly four unique variants")
    for variant in VARIANTS:
        if variant.warehouse_hours not in {12, 24}:
            raise ValueError("BFMWD warehouse window escaped the frozen grid")
        if variant.deployment_hours not in {3, 6}:
            raise ValueError("BFMWD deployment window escaped the frozen grid")
        if variant.robust_z_threshold != 1.0 or variant.hold_bars != 144:
            raise ValueError("BFMWD threshold and hold are frozen")


def preregistration_payload() -> dict[str, Any]:
    validate_contract()
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "candidate_family": CANDIDATE_FAMILY,
        "as_of_date": AS_OF_DATE,
        "decision": "freeze_source_support_before_source_values_or_any_outcome",
        "source_contract": {
            "builder": str(SOURCE_BUILDER),
            "builder_sha256": sha256_file(SOURCE_BUILDER),
            "source_decision": str(SOURCE_DECISION),
            "source_decision_sha256": sha256_file(SOURCE_DECISION),
            "physical_start": FROZEN_POLICY.warmup_start,
            "physical_end_exclusive": FROZEN_POLICY.selection_end_exclusive,
            "numeric_source_rows_parsed": 0,
            "network_calls": 0,
        },
        "policy": {
            "config": asdict(FROZEN_POLICY),
            "variants": [asdict(variant) for variant in VARIANTS],
            "official_fields": {
                "total": "funding_amount",
                "used": "funding_amount_used",
                "unused": "funding_amount - funding_amount_used",
                "utilization": "used / total clipped to [1e-6, 1-1e-6]",
                "tenor": "average_period_days",
            },
            "features": {
                "warehouse_charge": (
                    "log1p(unused[t-deployment]) - "
                    "log1p(unused[t-deployment-warehouse])"
                ),
                "used_deployment": ("log1p(used[t]) - log1p(used[t-deployment])"),
                "unused_draw": ("log1p(unused[t-deployment]) - log1p(unused[t])"),
                "utilization_deployment": (
                    "logit(utilization[t]) - logit(utilization[t-deployment])"
                ),
                "standardization": (
                    "median/MAD*1.4826 over the strictly prior 1440 valid hourly "
                    "feature observations, minimum 1080; current anchor excluded"
                ),
            },
            "trigger": (
                "all four feature robust-z values >= 1.0 and current tenor >= "
                "its strictly prior 1440-hour median; exact source coverage; "
                "same-symbol onset only"
            ),
            "direction": "fUSD -> LONG (+1); fBTC -> SHORT (-1)",
            "conflict": "abstain if fUSD and fBTC trigger at the same anchor",
            "entry": "source available_at + one five-minute bar",
            "exit": "scheduled open after exactly 144 five-minute bars",
            "source_only_controls": list(SOURCE_ONLY_CONTROLS),
            "later_economic_controls": list(LATER_ECONOMIC_CONTROLS),
        },
        "support_gates": dict(SUPPORT_GATES),
        "economic_gates": dict(ECONOMIC_GATES),
        "outcome_boundary": {
            "outcomes_opened": False,
            "outcome_sources_opened": False,
            "source_numeric_rows_opened": False,
            "comparator_rows_opened": False,
            "btc_market_rows_read": 0,
            "funding_rows_read": 0,
            "future_return_rows_read": 0,
            "return_or_pnl_fields_read": 0,
            "post_2023_rows_read": 0,
        },
        "one_way_sequence": {
            "source_artifact_after_preregistration": True,
            "source_support_before_market_evaluator": True,
            "evaluator_hash_frozen_before_market_access": True,
            "train_2021_2022_before_selection_2023": True,
            "selection_2023_before_any_2024_plus_access": True,
            "failed_candidate_repair_forbidden": True,
            "llm_rescue_forbidden": True,
        },
        "files": {
            "preregistration_source": str(PREREGISTRATION_SOURCE),
            "preregistration_source_sha256": sha256_file(PREREGISTRATION_SOURCE),
            "preregistration_document": str(PREREGISTRATION_DOCUMENT),
            "preregistration_document_sha256": sha256_file(PREREGISTRATION_DOCUMENT),
        },
    }
    payload["manifest_hash"] = canonical_hash(payload)
    return payload


def write_once_json(payload: dict[str, Any], output: Path) -> None:
    path = _path(output)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != serialized:
            raise FileExistsError(
                f"refusing to overwrite frozen preregistration: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(serialized)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = preregistration_payload()
    write_once_json(payload, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
