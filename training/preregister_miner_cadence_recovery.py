"""Freeze MCR-7 before loading any post-entry BTC market outcome."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = "results/miner_cadence_recovery_preregistration_2026-07-17.json"
DEFAULT_DOCS = "docs/miner-cadence-recovery-mcr7-preregistration-2026-07-17.md"
SOURCE = "data/coinmetrics_btc_miner_security_daily_2019_2023.csv.gz"
SOURCE_SHA256 = "448a101834df33f69abaeafe9aadfccd8ce9c3d6ad7816c1c2448189a12b8379"
SOURCE_MANIFEST = (
    "results/coinmetrics_btc_miner_security_daily_2019_2023_manifest_2026-07-17.json"
)
SOURCE_MANIFEST_SHA256 = (
    "045ccd1e8c842b7d0d56bdb9fb60873ed24e90a4c31b67bbcf47645121622149"
)


@dataclass(frozen=True)
class Policy:
    policy_id: str = "MCR-7"
    hash_change_days: int = 7
    reference_days: int = 180
    reference_min_periods: int = 120
    stress_z_max: float = -1.0
    stress_lookback_days: int = 14
    recovery_z_min: float = 0.0
    cadence_short_days: int = 3
    cadence_reference_days: int = 30
    maximum_source_lag_days: float = 3.0
    entry_delay_bars: int = 1
    hold_bars: int = 2_016
    leverage: float = 0.5
    base_cost_notional_per_side: float = 0.0006
    stress_cost_notional_per_side: float = 0.0010


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def protocol() -> dict[str, Any]:
    policy = Policy()
    return {
        "protocol_version": "miner_cadence_recovery_v1",
        "outcomes_opened": False,
        "policy": asdict(policy),
        "claim": (
            "A causal recovery in seven-day Bitcoin hash-rate growth after a recent "
            "miner-security contraction, confirmed by restored block cadence, marks an "
            "easing of miner operating pressure that supports the following seven-day BTC return."
        ),
        "evidence_boundary": {
            "source_distribution_and_support_shape_seen": True,
            "exact_mcr7_post_entry_returns_opened": False,
            "repository_has_seen_2021_2023_btc_returns_for_other_families": True,
            "2021_2022_label": "development train; not globally pristine",
            "2023_label": "single-policy selection; not globally pristine",
            "2024_label": "first exact-policy code-frozen source-and-outcome-unopened OOS",
            "2024_or_later_miner_source_opened": False,
        },
        "source_contract": {
            "miner_security": SOURCE,
            "miner_security_sha256": SOURCE_SHA256,
            "miner_security_manifest": SOURCE_MANIFEST,
            "miner_security_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "rows": 1_826,
            "interval": ["2019-01-01", "2024-01-01"],
            "columns_allowed": [
                "observation_date",
                "available_at",
                "HashRate",
                "BlkCnt",
            ],
            "columns_forbidden_for_signal": [
                "BTC price",
                "FeeTotNtv",
                "IssTotNtv",
                "TxCnt",
                "address or exchange-tag flows",
                "funding",
                "premium",
                "open interest",
                "Kimchi",
                "FX",
                "DXY",
            ],
            "availability": (
                "the current observation and every reference observation are usable only "
                "according to recorded AssetEODCompletionTime"
            ),
            "stale_backfill": (
                "rows with source lag above three days may seed later strictly causal "
                "references after publication but may never emit a signal"
            ),
            "revision_boundary": (
                "file hashes freeze this downloaded vintage; live promotion additionally "
                "requires forward-vintage parity"
            ),
            "official_docs": [
                "https://gitbook-docs.coinmetrics.io/network-data/network-data-overview/mining/hash-rate",
                "https://gitbook-docs.coinmetrics.io/network-data/network-data-overview/network-usage/blocks",
                "https://gitbook-docs.coinmetrics.io/network-data/network-data-overview/availability/asseteodcompletiontime",
            ],
        },
        "causal_feature_contract": {
            "hash_change": "log(HashRate[t]) - log(HashRate[t-7 observations])",
            "hash_change_z": (
                "current hash_change standardized against at most the last 180 earlier "
                "finite hash_change observations whose available_at is strictly earlier; require 120"
            ),
            "recent_stress": (
                "at least one of the prior 14 observation-date hash_change_z values, "
                "causally available before the current row, is <= -1.0"
            ),
            "recovery_cross": (
                "current hash_change_z >= 0 and the immediately prior observation-date "
                "hash_change_z < 0, with that prior row already available"
            ),
            "cadence_short": "mean(log(BlkCnt)) over t-2 through t",
            "cadence_reference": "mean(log(BlkCnt)) over t-30 through t-1",
            "cadence_recovered": "cadence_short >= cadence_reference",
            "eligible": (
                "finite causal features, source lag <=3 days, recent_stress, "
                "recovery_cross, and cadence_recovered"
            ),
            "event": "every eligible recovery cross; the cross itself prevents state repeats",
            "direction": "long only",
            "price_or_derivative_feature_columns_loaded": [],
        },
        "support_freeze_before_returns": {
            "train_2021_2022_nonoverlap_min": 35,
            "each_train_year_min": 14,
            "selection_2023_min": 24,
            "selection_2023_h1_min": 10,
            "selection_2023_h2_min": 10,
            "maximum_single_month_share": 0.15,
            "market_or_funding_rows_loaded": 0,
            "future_hold_availability_used_to_filter": False,
            "failure_action": "reject without loading any post-entry market outcome",
        },
        "execution_contract": {
            "decision": "only after the source row available_at and causal features are fixed",
            "earliest_tradable_open": "first UTC five-minute open >= available_at",
            "entry": "one complete five-minute latency bar after earliest_tradable_open",
            "exit": "scheduled open 2,016 five-minute bars (seven days) after entry",
            "nonoverlap": "one active trade; ignored overlapping recovery events never delay-enter",
            "side": "long BTCUSDT USD-M perpetual",
            "leverage": policy.leverage,
            "base_cost": "6 bp/notional/side",
            "stress_cost": "10 bp/notional/side",
            "funding_interval": "entry_time <= funding_time < exit_time",
            "stop_or_take_profit": None,
            "cagr": "full wall-clock split including warm-up and idle cash",
            "strict_mdd": (
                "global/pre-entry HWM, favorable-before-adverse held OHLC, funding, "
                "entry/exit/hypothetical-liquidation costs"
            ),
            "absolute_return_always_reported": True,
        },
        "selection_protocol": {
            "candidate_count": 1,
            "train": ["2021-03-01", "2023-01-01"],
            "selection": ["2023-01-01", "2024-01-01"],
            "selection_halves": {
                "h1": ["2023-01-01", "2023-07-01"],
                "h2": ["2023-07-01", "2024-01-01"],
            },
            "sealed": ["2024", "2025", "2026_ytd"],
            "no_parameter_repair": True,
            "gates": {
                "train_and_2023_absolute_return_positive": True,
                "each_train_year_absolute_return_positive": True,
                "2023_h1_and_h2_absolute_return_positive": True,
                "train_and_2023_cagr_to_strict_mdd_min": 3.0,
                "train_and_2023_strict_mdd_pct_max": 15.0,
                "train_and_2023_ten_bp_stress_positive": True,
                "train_and_2023_mean_gross_underlying_bp_min": 40.0,
                "train_and_2023_monthly_cluster_signflip_p_max": 0.10,
                "one_bar_additional_delay_train_and_2023_positive": True,
            },
        },
        "controls": {
            "direction_flip": "same clock, short side; diagnostic only",
            "cadence_confirmation_removed": "hash recovery cross without block-cadence check",
            "stale_hash_state_7d": "all hash recovery inputs delayed seven observations",
            "random_clock": "same yearly trade counts and seven-day hold",
            "constant_long": "nonoverlapping weekly long exposure on the same calendar",
            "mechanism_rule": (
                "reject if cadence-removed or stale-hash control independently passes all "
                "primary gates; controls can never replace the frozen primary"
            ),
        },
        "orthogonality_gate_after_performance": {
            "comparison_set": "gross-3.85 live anchor plus all previously promoted sleeves",
            "anchor_weights": {
                "oi_upbit_ratio288_low": 0.65,
                "new_long_minimal_funding_premium": 1.75,
                "cand_rex_veto_7": 1.45,
            },
            "exact_entry_jaccard_max": 0.05,
            "position_time_jaccard_max": 0.20,
            "absolute_daily_pnl_pearson_max": 0.30,
            "minimum_nonzero_daily_pnl_days": 20,
            "fixed_weight_increment": 0.25,
            "marginal_portfolio_absolute_return_and_ratio_must_improve": True,
            "undefined_metric": "fail_closed",
        },
        "post_selection_sequence": {
            "open_2024_only_after_pre2024_pass": True,
            "open_2025_only_after_2024_pass": True,
            "open_2026_only_after_2025_pass": True,
            "terminal_failure": "write rejection artifact and keep every later period sealed",
            "live_promotion_requires_forward_vintage_parity": True,
            "minimum_forward_shadow_days": 90,
        },
        "rejection_contract": (
            "any support, performance, mechanism, or orthogonality failure rejects MCR-7 "
            "without changing lookbacks, thresholds, side, latency, hold, or costs"
        ),
    }


def manifest() -> dict[str, Any]:
    core = protocol()
    return {
        **core,
        "manifest_hash": canonical_hash(core),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def validate_manifest(payload: dict[str, Any]) -> None:
    body = {k: v for k, v in payload.items() if k not in {"manifest_hash", "created_at"}}
    if payload.get("manifest_hash") != canonical_hash(body):
        raise ValueError("MCR-7 preregistration manifest hash mismatch")
    if payload.get("outcomes_opened") is not False:
        raise ValueError("MCR-7 preregistration cannot open outcomes")
    if payload.get("policy") != asdict(Policy()):
        raise ValueError("MCR-7 policy drifted")


def markdown(payload: dict[str, Any]) -> str:
    policy = payload["policy"]
    return f"""# MCR-7 miner cadence recovery preregistration

Status: **frozen before any exact-policy post-entry BTC return was loaded**.

MCR-7 uses only Coin Metrics `HashRate`, `BlkCnt`, and recorded daily
availability. A seven-day hash-rate change must have been at least one prior-only
standard deviation below normal during the prior 14 observations, then cross
back above its prior-only mean while three-day block cadence is no worse than the
strictly earlier 30-day reference. The side is long only.

- entry: first 5m open after availability, plus one complete 5m latency bar
- hold: {policy['hold_bars']} five-minute bars / seven days
- exposure: {policy['leverage']:.1f}x
- cost: 6 bp/notional/side base; 10 bp/notional/side stress
- 2021-2022: development train
- 2023: one frozen-policy selection year
- 2024+: sealed until every earlier gate passes

Support thresholds were shaped from source timestamps and feature counts only;
no market, funding, post-entry return, CAGR, or drawdown was loaded. Broader repo
research has seen old BTC returns, so 2023 is not described as pristine OOS.

Any failed gate retires this exact policy without threshold, side, hold, or
latency repair. Live promotion additionally requires forward-vintage parity and
90 shadow days.

Protocol hash: `{payload['manifest_hash']}`
"""


def run(output: str = DEFAULT_OUTPUT, docs: str = DEFAULT_DOCS) -> dict[str, Any]:
    payload = manifest()
    validate_manifest(payload)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    docs_path = Path(docs)
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text(markdown(payload))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--docs", default=DEFAULT_DOCS)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.docs), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
