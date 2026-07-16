"""Freeze LVRT-R0 before inspecting any policy outcome.

LVRT-R0 infers a liquidity-vacuum/replenishment transition from two completed
Binance USD-M aggTrade buckets.  It does not claim to observe resting-book
liquidity, and this module never loads prices, returns, CAGR, or drawdown.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = (
    "results/liquidity_vacuum_replenishment_preregistration_2026-07-17.json"
)
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
    policy_id: str = "LVRT-R0"
    baseline_bars: int = 8_640
    baseline_min_periods: int = 2_016
    setup_quantile: float = 0.80
    minimum_agg_trade_count: int = 64
    hold_bars: int = 12
    leverage: float = 0.5
    base_cost_notional_per_side: float = 0.0006
    stress_cost_notional_per_side: float = 0.0008
    post_gap_quarantine_bars: int = 24


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_manifest() -> dict[str, Any]:
    policy = Policy()
    core: dict[str, Any] = {
        "protocol_version": "liquidity_vacuum_replenishment_r0_v1",
        "outcomes_opened": False,
        "policy": asdict(policy),
        "research_history_boundary": {
            "2020_2023_market_returns_seen_by_unrelated_repo_research": True,
            "exact_lvrt_r0_outcomes_opened": False,
            "claim": (
                "mechanically frozen exact-policy evaluation, not a globally "
                "pristine historical market holdout; forward shadow/live remains "
                "the final generalization test"
            ),
        },
        "novelty_boundary": {
            "excluded_feature_families": [
                "open_interest",
                "funding_or_premium",
                "kimchi_or_fx",
                "REX_or_HTF_price_regime",
                "Markov_state",
                "spot_perp_basis",
            ],
            "nearest_failed_families": {
                "MFIC": "same-window impact curvature",
                "NETF": "breadth/notional disagreement then capital revelation",
                "RIFT": "persistent spot/perp/path/crowding pressure",
                "TAAR": "same-bar tail-arrival absorption or release",
            },
            "distinct_axis": (
                "a completed bursty/concentrated aggressive-flow vacuum followed "
                "by a separate completed bar with opposite aggressive flow and "
                "price reversal"
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
                "quarantine each source-ID-gap UTC day, every missing feature "
                "bucket, and the following 24 five-minute bars; never impute"
            ),
        },
        "claim_boundary": {
            "claim": (
                "opposite aggressive flow plus price reversal after a concentrated "
                "aggressive-flow vacuum can proxy a short-lived refill transition"
            ),
            "not_claimed": [
                "direct observation of passive depth",
                "queue replenishment",
                "participant identity",
                "hidden orders",
            ],
        },
        "causal_feature_contract": {
            "setup_bucket": "completed five-minute bucket t0",
            "setup_direction": "s0 = sign(signed_quote_notional[t0]); s0 != 0",
            "setup_response": "signed_price_response[t0] > 0",
            "setup_burst": (
                "interarrival_burstiness[t0] >= strictly-prior q80 over the last "
                "8,640 clean observations, requiring 2,016"
            ),
            "setup_concentration": (
                "event_notional_hhi[t0] >= strictly-prior q80 over the last "
                "8,640 clean observations, requiring 2,016"
            ),
            "activity": "agg_trade_count >= 64 on setup and confirmation",
            "confirmation_bucket": "the immediately following completed bucket t1",
            "confirmation_flow": "sign(signed_quote_notional[t1]) == -s0",
            "confirmation_price": "s0 * micro_log_return[t1] < 0",
            "decision_time": "after t1 closes",
            "side": "-s0",
        },
        "execution_contract": {
            "entry": "next five-minute open t2 after t1 closes",
            "exit": "scheduled open after 12 held five-minute bars",
            "nonoverlap": True,
            "position_size": "0.5x account gross",
            "base_cost": "6 bp/notional/side",
            "stress_cost": "8 bp/notional/side",
            "stop_or_take_profit": None,
            "funding": (
                "realized settlement cash is accounting-only if present; funding "
                "is never a signal input"
            ),
            "strict_mdd": (
                "global/pre-entry high-water; held favorable-before-adverse OHLC; "
                "entry, hypothetical liquidation, and exit costs"
            ),
            "cagr_clock": "full wall-clock split including idle cash",
        },
        "support_freeze_before_returns": {
            "minimum_nonoverlap_total": 250,
            "minimum_nonoverlap_each_year_2020_2023": 40,
            "minimum_nonoverlap_each_2023_half": 20,
            "minimum_each_side_share": 0.25,
            "maximum_each_side_share": 0.75,
            "maximum_single_month_share": 0.20,
            "required_path": (
                "setup, confirmation, entry, every held bucket, and exit must be "
                "inside the split and outside quarantine"
            ),
            "failure_action": "reject without opening price outcomes",
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
                "train_trades_min": 120,
                "2023_trades_min": 80,
                "2023_h1_and_h2_absolute_return_positive": True,
                "2023_h1_and_h2_trades_min": 20,
                "train_and_2023_mean_gross_underlying_bp_min": 12.0,
                "train_and_2023_eight_bp_stress_absolute_return_positive": True,
            },
        },
        "falsification_controls": {
            "direction_flip": "same clock, opposite action",
            "no_reversal_confirmation": (
                "setup bucket itself becomes the signal bucket; next-open entry"
            ),
            "one_bar_extra_delay": "same action and hold, entry one bar later",
            "one_day_shifted_setup": (
                "confirmation paired with setup features exactly 288 bars earlier"
            ),
            "sign_permuted_confirmation": (
                "fixed-seed 20260717 permutation of confirmation flow signs; "
                "rejection control"
            ),
            "placebo_rule": (
                "reject if a time-shift or sign-permutation placebo independently "
                "passes all primary train/selection performance gates"
            ),
        },
        "orthogonality_after_outcomes": {
            "compare_against": (
                "all live/shadow portfolio anchors with reproducible entry clocks"
            ),
            "exact_entry_jaccard_max": 0.05,
            "position_time_jaccard_max": 0.15,
            "absolute_daily_pnl_pearson_max": 0.30,
            "minimum_nonzero_daily_pnl_days": 20,
            "undefined_metric": "fail_closed",
            "marginal_portfolio_improvement_required": True,
        },
        "rejection_contract": (
            "any support or pre-2024 performance failure rejects LVRT-R0; do not "
            "change quantile, confirmation, side, hold, costs, or gates"
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
        raise RuntimeError("LVRT-R0 preregistration hash mismatch")
    if payload.get("outcomes_opened") is not False:
        raise RuntimeError("LVRT-R0 preregistration cannot open outcomes")
    if payload.get("policy") != asdict(Policy()):
        raise RuntimeError("LVRT-R0 frozen policy differs from code")
    if payload.get("selection_protocol", {}).get("candidate_count") != 1:
        raise RuntimeError("LVRT-R0 must remain a singleton policy")


def write_manifest_once(path: str | Path, payload: dict[str, Any]) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = json.loads(output.read_text())
        validate_manifest(existing)
        if existing["manifest_hash"] != payload["manifest_hash"]:
            raise RuntimeError("refusing to overwrite frozen LVRT-R0 preregistration")
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
                "outcomes_opened": payload["outcomes_opened"],
                "policy_id": payload["policy"]["policy_id"],
                "manifest_hash": payload["manifest_hash"],
                "output": args.output,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
