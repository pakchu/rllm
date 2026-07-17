"""Freeze ICDR-144 before opening any strategy outcome.

ICDR-144 tests an asymmetric deleveraging mechanism unique to BTC-margined
COIN-M positioning.  This module writes only a deterministic protocol manifest;
it never parses executable prices, future returns, funding cash flow, CAGR, or
drawdown.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = (
    "results/inverse_collateral_deleveraging_reclaim_preregistration_2026-07-17.json"
)
METRICS_PATH = (
    "data/binance_cross_collateral_metrics_btc_2021_2023/"
    "BTC_cross_collateral_metrics_5m_2021-07-08_2023-12-31.csv.gz"
)
METRICS_MANIFEST_PATH = (
    "results/binance_cross_collateral_metrics_btc_2021_2023_manifest.json"
)
MARKET_PATH = (
    "data/binance_um_kline_reference_btc_2020_2023/"
    "BTCUSDT_5m_2020-01-01_2023-12-31.csv.gz"
)
MARKET_MANIFEST_PATH = (
    "data/binance_um_kline_reference_btc_2020_2023/build_manifest.json"
)
FUNDING_PATH = "data/binance_um_btcusdt_funding_marks_2020_2023.csv.gz"
FUNDING_MANIFEST_PATH = (
    "results/binance_um_btcusdt_funding_marks_2020_2023_manifest_2026-07-17.json"
)


@dataclass(frozen=True)
class Policy:
    policy_id: str = "ICDR-144"
    baseline_bars: int = 8_640
    baseline_min_periods: int = 2_016
    oi_change_bars: int = 12
    taker_smoothing_bars: int = 3
    purge_quantiles: tuple[float, ...] = (0.80, 0.85, 0.90, 0.925, 0.95)
    sell_stress_quantile: float = 0.90
    sell_gap_quantile: float = 0.90
    confirmation_window_bars: int = 12
    post_gap_quarantine_bars: int = 24
    execution_delay_bars: int = 2
    hold_bars: int = 144
    leverage: float = 0.5
    base_cost_notional_per_side: float = 0.0006
    stress_cost_notional_per_side: float = 0.0010


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def policy_payload() -> dict[str, Any]:
    """Return the frozen policy in its canonical JSON representation."""
    payload = asdict(Policy())
    payload["purge_quantiles"] = list(payload["purge_quantiles"])
    return payload


def build_manifest() -> dict[str, Any]:
    core: dict[str, Any] = {
        "protocol_version": "inverse_collateral_deleveraging_reclaim_v1",
        "as_of_date": "2026-07-17",
        "outcomes_opened": False,
        "policy": policy_payload(),
        "research_history_boundary": {
            "market_history_seen_by_unrelated_repo_research": True,
            "cross_collateral_metrics_source_values_seen": True,
            "exact_icdr_144_post_entry_outcomes_opened": False,
            "support_only_access_allowed": (
                "current and past metrics, causal feature distributions, event "
                "counts, branch counts, timestamps, and clock overlap only"
            ),
            "forbidden_before_evaluator_freeze": [
                "entry_to_exit_return",
                "post_entry_OHLC",
                "funding_PnL",
                "win_rate",
                "CAGR",
                "drawdown",
            ],
        },
        "novelty_boundary": {
            "distinct_axis": (
                "relative contraction and recovery of BTC-margined COIN-M "
                "contract OI versus USD-M notional OI, confirmed by the reversal "
                "of COIN-M-specific taker selling"
            ),
            "excluded_inputs": [
                "REX_or_rolling_extrema",
                "price_momentum_or_reversion_signal",
                "funding_or_premium_signal",
                "kimchi_FX_or_DXY",
                "Markov_HMM_or_change_point_state",
                "spot_perp_flow_or_centroid",
                "aggregate_trade_tail_arrival_ticket_or_fill_compression",
                "onchain_or_attention",
            ],
            "nearest_existing_families": {
                "inventory_purge_reclaim": (
                    "uses USD-M positioning only; ICDR requires a relative "
                    "BTC-collateral contraction and source-only reclaim sequence"
                ),
                "cross_collateral_book": (
                    "uses resting depth shells; ICDR uses exchange-published "
                    "positioning and taker aggregates"
                ),
                "inferred_liquidation": (
                    "uses price/OI/taker inference on USD-M; ICDR isolates the "
                    "inverse-collateral cohort without price in the signal"
                ),
            },
            "forbidden_repairs_after_outcomes": [
                "reverse direction",
                "add a price gate",
                "change OI units",
                "change quantile grid",
                "change confirmation window",
                "change delay or hold",
                "drop the reclaim confirmation",
            ],
        },
        "source_contract": {
            "source_commit": "8d347432cd36d59458ad9a26c7c8aef1ec94b8ee",
            "metrics": METRICS_PATH,
            "metrics_sha256": (
                "ab9f18ba7745f21b17ac1124c45bb755245d404d66100c595bb77631f4bc1757"
            ),
            "metrics_manifest": METRICS_MANIFEST_PATH,
            "metrics_manifest_sha256": (
                "c0732ca47451209a9bb519545b0e349550994d870d476ee66ecbae81588fb159"
            ),
            "metrics_audit": (
                "docs/binance-cross-collateral-positioning-metrics-source-audit-"
                "2026-07-17.md"
            ),
            "market": MARKET_PATH,
            "market_sha256": (
                "e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d"
            ),
            "market_manifest": MARKET_MANIFEST_PATH,
            "market_manifest_sha256": (
                "c04fbbd299cc748a6745c0ef030787da4d560833c744c81c98dd8840efc7913e"
            ),
            "funding": FUNDING_PATH,
            "funding_sha256": (
                "3284bbb6bb67946acb673c6b67459543e217f752589e1d47b6c7c3b659f733e6"
            ),
            "funding_manifest": FUNDING_MANIFEST_PATH,
            "funding_manifest_sha256": (
                "a0b2d27e1aa8cf2d9ab8cb659b598ee0a6d7bd25401c9e10ae92d1a74415845b"
            ),
            "available_start": "2021-07-08",
            "available_end_exclusive": "2024-01-01",
            "gap_policy": (
                "never impute; current and required lookback must be complete, "
                "then quarantine the next 24 five-minute rows"
            ),
        },
        "causal_feature_contract": {
            "availability": (
                "metrics row t is usable only after t and one complete additional "
                "five-minute availability bucket; all thresholds exclude t"
            ),
            "unit_safe_open_interest": {
                "um": "U[t] = log(um_sum_open_interest_value[t])",
                "cm": "C[t] = log(cm_sum_open_interest[t])",
                "reason": (
                    "USD-M notional value and COIN-M contract count are each used "
                    "only through dimensionless changes; raw levels are never subtracted"
                ),
            },
            "one_hour_changes": {
                "um": "dU[t] = U[t] - U[t-12]",
                "cm": "dC[t] = C[t] - C[t-12]",
                "relative_purge": "P[t] = dU[t] - dC[t]",
            },
            "taker_state": {
                "um_log": (
                    "TU[t] = mean(log(um_taker_ratio[t-2:t])) over three clean bars"
                ),
                "cm_log": (
                    "TC[t] = mean(log(cm_taker_ratio[t-2:t])) over three clean bars"
                ),
                "cm_sell_stress": "S[t] = -TC[t]",
                "cm_specific_sell_gap": "G[t] = TU[t] - TC[t]",
            },
            "lagged_thresholds": (
                "qP(Q), qS(0.90), and qG(0.90) use only clean t-8640..t-1 "
                "values with at least 2016 observations"
            ),
            "setup": (
                "dC[t] < 0, P[t] >= qP(Q), S[t] >= qS(0.90), and "
                "G[t] >= qG(0.90); start only on false-to-true transition"
            ),
            "confirmation": (
                "within bars t+1..t+12 take the first clean row k where "
                "cm_taker_ratio[k] >= 1, cm_taker_ratio[k] >= um_taker_ratio[k], "
                "and log(cm_OI[k]/cm_OI[k-1]) >= 0"
            ),
            "expired_setup": "no trade if no confirmation appears within 12 bars",
            "action": "fixed LONG BTCUSDT USD-M perpetual after confirmation",
            "price_signal_columns": [],
        },
        "support_calibration": {
            "vary_only": "relative purge quantile Q in the frozen policy grid",
            "selection_rule": (
                "select the highest Q passing every 2021-2022 support and novelty "
                "floor; 2023 may only pass or reject that frozen Q and cannot select "
                "a fallback"
            ),
            "train_window": ["2021-07-08", "2023-01-01"],
            "outcome_blind_validation_window": ["2023-01-01", "2024-01-01"],
            "minimum_nonoverlap_train": 100,
            "minimum_2021_partial": 20,
            "minimum_2022": 50,
            "minimum_2023": 75,
            "minimum_each_2023_half": 30,
            "minimum_confirmation_rate": 0.05,
            "maximum_confirmation_rate": 0.80,
            "confirmation_rate_definition": (
                "confirmed setup episodes divided by accepted false-to-true setup "
                "episodes before position non-overlap, calculated separately in "
                "train and 2023"
            ),
            "maximum_single_month_share": 0.15,
            "month_share_definition": (
                "largest UTC entry-month count divided by non-overlapping primary "
                "entries, calculated separately in train and 2023"
            ),
            "jaccard_definition": (
                "Jaccard of primary and control non-overlapping entry timestamps, "
                "calculated separately in train and 2023"
            ),
            "maximum_signal_jaccard": {
                "cm_only_oi": 0.75,
                "no_taker_gap": 0.80,
                "no_reclaim": 0.25,
                "no_oi_stop": 0.80,
                "um_matched": 0.20,
                "one_hour_signal_delay": 0.05,
                "one_day_shifted_clock": 0.02,
            },
            "failure_action": (
                "reject ICDR-144 before opening any outcome; never fall back to a "
                "lower Q after the frozen train-selected Q sees 2023 support"
            ),
        },
        "execution_contract": {
            "decision_time": "after the reclaim-confirmation metrics row is complete",
            "entry": "confirmation row + 2 five-minute opens",
            "exit": "scheduled open after 144 held five-minute bars (12 hours)",
            "nonoverlap": True,
            "clock_reservation": (
                "reserve each policy globally over the complete pre-2024 clock, then "
                "slice only trades whose setup, signal, entry, and exit are contained "
                "inside the requested evaluation window"
            ),
            "position_size": "0.5x fixed account gross",
            "base_cost": "6 bp/notional/side",
            "stress_cost": "10 bp/notional/side",
            "funding": "exact realized BTCUSDT settlements on [entry, exit)",
            "strict_mdd": (
                "global/pre-entry high-water; entry cost; favorable-before-adverse "
                "held OHLC; exact funding; hypothetical liquidation and exit cost"
            ),
            "cagr_clock": "full wall-clock split including warm-up and idle cash",
        },
        "falsification_controls": {
            "direction_flip": "same primary clock, exact SHORT side",
            "cm_only_oi": (
                "own setup clock using dC<0 and prior-Q extreme -dC instead of P, "
                "with the same S, G, and three-part CM reclaim requirements"
            ),
            "no_taker_gap": (
                "own setup clock with only the G>=prior-q90(G) clause removed and "
                "the same three-part CM reclaim"
            ),
            "no_reclaim": (
                "own clock entering two bars after the accepted primary setup onset, "
                "without scanning a confirmation window"
            ),
            "no_oi_stop": (
                "own primary setup clock whose first confirmation requires only "
                "cm_taker_ratio>=1 and cm_taker_ratio>=um_taker_ratio"
            ),
            "um_matched": (
                "own clock with dU<0, prior-Q extreme -dU, prior-q90(-TU), and the "
                "first clean confirmation where um_taker_ratio>=1 and one-bar dU>=0"
            ),
            "one_hour_signal_delay": "same primary confirmation shifted 12 bars",
            "one_day_shifted_clock": "same primary confirmation shifted 288 bars",
            "random_side": "same primary clock with seed-20260717 Rademacher side",
        },
        "selection_protocol": {
            "stage1_train": ["2021-07-08", "2023-01-01"],
            "stage1_subperiods": {
                "2021_partial": ["2021-07-08", "2022-01-01"],
                "2022": ["2022-01-01", "2023-01-01"],
            },
            "stage2_selection": ["2023-01-01", "2024-01-01"],
            "stage2_halves": {
                "h1": ["2023-01-01", "2023-07-01"],
                "h2": ["2023-07-01", "2024-01-01"],
            },
            "sealed": ["2024", "2025", "2026_ytd"],
            "candidate_count": 1,
            "stage2_requires_unchanged_stage1_pass": True,
            "stage2_support_cannot_reselect_q": True,
            "no_parameter_repair": True,
            "gates": {
                "train_and_2023_absolute_return_positive": True,
                "train_and_2023_cagr_to_strict_mdd_min": 3.0,
                "train_and_2023_strict_mdd_pct_max": 15.0,
                "train_and_2023_weekly_cluster_signflip_p_max": 0.10,
                "train_trades_min": 80,
                "2023_trades_min": 60,
                "2021_partial_and_2022_absolute_return_positive": True,
                "2021_partial_trades_min": 20,
                "2022_trades_min": 40,
                "2023_h1_and_h2_absolute_return_positive": True,
                "2023_h1_and_h2_trades_min": 25,
                "train_and_2023_mean_gross_underlying_bp_min": 20.0,
                "train_and_2023_ten_bp_stress_absolute_return_positive": True,
                "mechanism_controls_must_be_beaten": True,
                "stale_or_random_full_qualification_rejects": True,
            },
        },
        "orthogonality_after_standalone_pass": {
            "economic_gate_first": True,
            "exact_entry_jaccard_max": 0.02,
            "near_six_hour_entry_fraction_max": 0.25,
            "strong_target_near_six_hour_fraction_max": 0.10,
            "position_time_jaccard_max": 0.15,
            "absolute_daily_pnl_pearson_max": 0.30,
            "minimum_nonzero_pnl_days": 10,
            "marginal_portfolio_improvement_required": True,
        },
        "rllm_boundary": {
            "enabled_only_after_deterministic_standalone_pass": True,
            "allowed_tokens": [
                "relative_purge_rank",
                "cm_sell_stress_rank",
                "cm_specific_sell_gap_rank",
                "bars_since_setup",
                "reclaim_flags",
                "current_position",
                "time_to_exit",
            ],
            "allowed_actions": ["abstain", "size_fixed_long"],
            "forbidden_actions": ["reverse_short", "change_base_event"],
        },
        "rejection_contract": (
            "any support or staged performance failure retires ICDR-144 without "
            "threshold, sign, unit, feature, confirmation, delay, hold, or gate repair"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_manifest(payload: dict[str, Any], *, verify_sources: bool = True) -> None:
    manifest_hash = payload.get("manifest_hash")
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if manifest_hash != canonical_hash(core):
        raise ValueError("ICDR-144 preregistration manifest hash changed")
    if payload.get("outcomes_opened") is not False:
        raise ValueError("ICDR-144 outcomes opened before preregistration")
    if payload.get("policy") != policy_payload():
        raise ValueError("ICDR-144 policy differs from frozen defaults")
    if payload["selection_protocol"].get("candidate_count") != 1:
        raise ValueError("ICDR-144 must remain a singleton")
    if payload["causal_feature_contract"].get("price_signal_columns") != []:
        raise ValueError("ICDR-144 price entered the source-only signal")
    if verify_sources:
        source = payload["source_contract"]
        for key, hash_key in (
            ("metrics", "metrics_sha256"),
            ("metrics_manifest", "metrics_manifest_sha256"),
            ("market", "market_sha256"),
            ("market_manifest", "market_manifest_sha256"),
            ("funding", "funding_sha256"),
            ("funding_manifest", "funding_manifest_sha256"),
        ):
            if _sha256(source[key]) != source[hash_key]:
                raise ValueError(f"ICDR-144 frozen source changed: {source[key]}")


def write_manifest(output: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    payload = build_manifest()
    validate_manifest(payload)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = write_manifest(args.output)
    print(
        json.dumps(
            {
                "outcomes_opened": payload["outcomes_opened"],
                "policy_id": payload["policy"]["policy_id"],
                "manifest_hash": payload["manifest_hash"],
                "output": args.output,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
