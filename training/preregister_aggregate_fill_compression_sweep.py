"""Freeze AFCS-144 before opening any strategy outcome.

AFCS-144 tests whether a rare, directionally aligned sweep whose public
``aggTrade`` records compress many underlying trade IDs continues over the
next twelve hours.  This module writes only the protocol manifest; it never
loads price outcomes, funding cashflows, CAGR, or drawdown.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = "results/aggregate_fill_compression_sweep_preregistration_2026-07-17.json"
FEATURE_PATH = (
    "data/binance_um_aggtrade_microstructure_btc_2020_2023/"
    "BTCUSDT_aggtrade_5m_2020-01-01_2023-12-31.csv.gz"
)
FEATURE_MANIFEST_PATH = (
    "data/binance_um_aggtrade_microstructure_btc_2020_2023/build_manifest.json"
)
MARKET_PATH = (
    "data/binance_um_kline_reference_btc_2020_2023/"
    "BTCUSDT_5m_2020-01-01_2023-12-31.csv.gz"
)
MARKET_MANIFEST_PATH = (
    "data/binance_um_kline_reference_btc_2020_2023/build_manifest.json"
)
AUDIT_PATH = "results/binance_aggtrade_microstructure_audit_2026-07-14.json"


@dataclass(frozen=True)
class Policy:
    policy_id: str = "AFCS-144"
    baseline_bars: int = 8_640
    baseline_min_periods: int = 2_016
    compression_quantile: float = 0.975
    coherence_quantile: float = 0.90
    response_quantile: float = 0.80
    activity_quantile: float = 0.50
    minimum_agg_trade_count: int = 64
    episode_reset_bars: int = 1
    execution_delay_bars: int = 2
    hold_bars: int = 144
    leverage: float = 0.5
    base_cost_notional_per_side: float = 0.0006
    stress_cost_notional_per_side: float = 0.0010
    post_gap_quarantine_bars: int = 24


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_manifest() -> dict[str, Any]:
    policy = Policy()
    core: dict[str, Any] = {
        "protocol_version": "aggregate_fill_compression_sweep_v1",
        "outcomes_opened": False,
        "policy": asdict(policy),
        "research_history_boundary": {
            "market_history_seen_by_unrelated_repo_research": True,
            "exact_afcs_144_outcomes_opened": False,
            "support_calibration_window": ["2020-01-01", "2023-01-01"],
            "support_only_values_seen": (
                "feature distributions and event counts; no entry-to-exit price, "
                "future OHLC, funding PnL, return, CAGR, or MDD"
            ),
        },
        "novelty_boundary": {
            "excluded_feature_families": [
                "open_interest",
                "funding_or_premium",
                "kimchi_or_fx",
                "REX_or_HTF_price_regime",
                "Markov_state",
                "event_notional_tail",
                "interarrival_burstiness",
            ],
            "nearest_failed_families": {
                "MFIC": "multi-bar fragmentation and impact curvature",
                "NETF": "event-breadth versus capital disagreement and revelation",
                "LVRT": "arrival/HHI shock followed by opposite-flow reversal",
                "TAAR": "event-size tail plus arrival irregularity",
            },
            "distinct_axis": (
                "the inclusive underlying-trade-ID span compressed into each "
                "public aggregate-trade event is the primary state"
            ),
        },
        "source_contract": {
            "features": FEATURE_PATH,
            "feature_manifest": FEATURE_MANIFEST_PATH,
            "feature_sha256": (
                "c2bb0e6742f8cdc4e13315e7f0a13d6ab9cd536fb40d9cb4484b7a6ba30131cf"
            ),
            "feature_manifest_sha256": (
                "6eec40460a6146c58994e52f1af9ace4eecc0c085887d97af5ef17c30b9f7e73"
            ),
            "market": MARKET_PATH,
            "market_manifest": MARKET_MANIFEST_PATH,
            "market_sha256": (
                "e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d"
            ),
            "market_manifest_sha256": (
                "c04fbbd299cc748a6745c0ef030787da4d560833c744c81c98dd8840efc7913e"
            ),
            "source_audit": AUDIT_PATH,
            "source_audit_sha256": (
                "5ac5a342d7f766ea0b6dcf9f97468ab70b9e1194775469ed0245d9208d0dc9c6"
            ),
            "audit_must_pass": True,
            "available_end_exclusive": "2024-01-01",
            "source_gap_policy": (
                "quarantine each verified source-ID-gap UTC day, every missing "
                "feature bucket, and the next 24 five-minute bars; never impute"
            ),
        },
        "claim_boundary": {
            "claim": (
                "a rare coherent sweep with unusually many underlying trade IDs "
                "per aggregate event and aligned same-bucket price response may "
                "represent information diffusion that persists for twelve hours"
            ),
            "not_claimed": [
                "direct observation of one parent order",
                "resting-book depth or queue position",
                "participant identity",
                "every integer in the first/last trade-ID span is one causal fill",
            ],
        },
        "causal_feature_contract": {
            "bucket": "one completed five-minute bucket t",
            "direction": "d = sign(signed_quote_notional[t]); d must be nonzero",
            "compression": (
                "underlying_trades_per_agg_event[t] >= strictly-prior 30-day q97.5"
            ),
            "coherence": "flow_coherence[t] >= strictly-prior 30-day q90",
            "aligned_response": (
                "signed_price_response[t] >= strictly-prior 30-day q80 and > 0"
            ),
            "activity": (
                "quote_notional[t] >= strictly-prior 30-day median and "
                "agg_trade_count[t] >= 64"
            ),
            "baseline": (
                "all thresholds use only clean t-8640..t-1 observations and "
                "require at least 2016 observations"
            ),
            "episode": "eligible at t and ineligible at t-1",
            "side": "continue in direction d",
        },
        "execution_contract": {
            "decision_time": "after bucket t closes",
            "entry": "t+2 open, leaving one full five-minute computation bar",
            "exit": "scheduled open after 144 held five-minute bars",
            "nonoverlap": True,
            "position_size": "0.5x account gross",
            "base_cost": "6 bp/notional/side",
            "stress_cost": "10 bp/notional/side",
            "funding": "exact realized BTCUSDT settlement rates on [entry, exit)",
            "stop_or_take_profit": None,
            "strict_mdd": (
                "global/pre-entry high-water; entry cost; favorable-before-adverse "
                "held OHLC; exact funding; hypothetical liquidation and exit cost"
            ),
            "cagr_clock": "full wall-clock split including warm-up and idle cash",
        },
        "support_freeze_before_returns": {
            "minimum_nonoverlap_total_2020_2022": 300,
            "minimum_nonoverlap_each_year_2020_2022": 40,
            "minimum_nonoverlap_each_half_2020_2022": 20,
            "minimum_each_side_share": 0.25,
            "maximum_each_side_share": 0.75,
            "maximum_single_month_share": 0.10,
            "2023_support_is_diagnostic_only": True,
            "minimum_nonoverlap_2023": 60,
            "minimum_nonoverlap_each_2023_half": 25,
            "required_path": (
                "signal, reserved compute bar, entry, every held bucket, and exit "
                "must stay inside the split and outside quarantine"
            ),
            "failure_action": "reject without opening any strategy outcome",
        },
        "selection_protocol": {
            "train": ["2020-01-01", "2023-01-01"],
            "selection": ["2023-01-01", "2024-01-01"],
            "selection_halves": {
                "h1": ["2023-01-01", "2023-07-01"],
                "h2": ["2023-07-01", "2024-01-01"],
            },
            "sealed": ["2024", "2025", "2026_ytd"],
            "candidate_count": 1,
            "no_parameter_repair": True,
            "gates": {
                "train_and_2023_absolute_return_positive": True,
                "train_and_2023_cagr_to_strict_mdd_min": 3.0,
                "train_and_2023_strict_mdd_pct_max": 15.0,
                "train_and_2023_weekly_cluster_signflip_p_max": 0.10,
                "train_trades_min": 250,
                "2023_trades_min": 60,
                "2023_h1_and_h2_absolute_return_positive": True,
                "2023_h1_and_h2_trades_min": 25,
                "train_and_2023_mean_gross_underlying_bp_min": 20.0,
                "train_and_2023_ten_bp_stress_absolute_return_positive": True,
            },
        },
        "falsification_controls": {
            "direction_flip": "same candidate clock, exact opposite side",
            "no_compression": "own clock with compression threshold removed",
            "no_coherence": "own clock with coherence threshold removed",
            "no_aligned_response": "own clock with response threshold removed",
            "one_hour_signal_delay": "same side and hold, entry delayed 12 bars",
            "one_day_shifted_clock": "same side and hold, signal shifted 288 bars",
            "random_side": "same clock, fixed-seed 20260717 Rademacher sides",
            "mechanism_gate": (
                "primary minimum train/2023 CAGR-MDD must exceed every component "
                "removal; time-shift or random-side full qualification rejects"
            ),
        },
        "orthogonality_after_standalone_pass": {
            "exact_entry_jaccard_max": 0.05,
            "position_time_jaccard_max": 0.15,
            "absolute_daily_pnl_pearson_max": 0.30,
            "marginal_portfolio_improvement_required": True,
        },
        "rejection_contract": (
            "any support or pre-2024 performance failure retires AFCS-144; do not "
            "change its sign, thresholds, delay, hold, costs, or gates"
        ),
    }
    return {
        **core,
        "manifest_hash": canonical_hash(core),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def validate_manifest(payload: dict[str, Any]) -> None:
    core = {
        key: value
        for key, value in payload.items()
        if key not in {"manifest_hash", "created_at"}
    }
    if canonical_hash(core) != payload.get("manifest_hash"):
        raise RuntimeError("AFCS-144 preregistration hash mismatch")
    if payload.get("outcomes_opened") is not False:
        raise RuntimeError("AFCS-144 preregistration cannot open outcomes")
    if payload.get("policy") != asdict(Policy()):
        raise RuntimeError("AFCS-144 frozen policy differs from code")
    if payload.get("selection_protocol", {}).get("candidate_count") != 1:
        raise RuntimeError("AFCS-144 must remain a singleton policy")


def write_manifest_once(path: str | Path, payload: dict[str, Any]) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = json.loads(output.read_text())
        validate_manifest(existing)
        if existing["manifest_hash"] != payload["manifest_hash"]:
            raise RuntimeError("refusing to overwrite frozen AFCS-144 preregistration")
        return "verified_existing"
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return "created"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build_manifest()
    status = write_manifest_once(args.output, payload)
    print(
        json.dumps(
            {
                "status": status,
                "outcomes_opened": False,
                "policy_id": payload["policy"]["policy_id"],
                "manifest_hash": payload["manifest_hash"],
                "output": args.output,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
