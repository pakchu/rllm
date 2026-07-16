"""Freeze a Coinbase/Binance price-discovery proxy before downloading outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = "results/coinbase_spot_leadership_preregistration_2026-07-16.json"
SELECTION_END = "2023-01-01"
HOLDOUT = ("2023-01-01", "2024-01-01")


@dataclass(frozen=True, order=True)
class Policy:
    policy_id: str
    family: str
    side: int
    hold_bars: int


def policy_grid() -> list[Policy]:
    """Return the immutable, deliberately small directional policy family."""
    specs = (
        ("relative_return_lead", (1, 3)),
        ("premium_shock", (1, 3)),
        ("activity_confirmed_relative", (1, 3)),
        ("activity_premium_confluence", (3,)),
        ("return_premium_confluence", (3,)),
    )
    policies: list[Policy] = []
    for family, holds in specs:
        for hold_bars in holds:
            for side in (-1, 1):
                policies.append(
                    Policy(
                        policy_id=f"P{len(policies) + 1:02d}",
                        family=family,
                        side=side,
                        hold_bars=hold_bars,
                    )
                )
    return sorted(policies)


def canonical_hash(payload: Any) -> str:
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def build_manifest() -> dict[str, Any]:
    core: dict[str, Any] = {
        "protocol_version": "coinbase_spot_leadership_v2",
        "outcomes_opened": False,
        "claim_boundary": {
            "claim": (
                "completed Coinbase BTC-USD candle activity and price changes can "
                "proxy a venue-leadership state before the next Binance-perpetual bar"
            ),
            "not_claimed": [
                "true price discovery",
                "taker-side order-flow imbalance",
                "order-book pressure",
                "a pure Coinbase premium",
            ],
            "reason": (
                "the public historical candle source has no aggressor side or book "
                "state, and the cross-venue comparison mixes USD spot with USDT futures"
            ),
        },
        "novelty_check": {
            "repo_coinbase_family_found": False,
            "nearest_existing_family": "Binance spot-perpetual internal leadership",
            "distinct_axis": "US-regulated USD spot venue versus Binance USDT perpetual",
        },
        "source_contract": {
            "coinbase_endpoint": (
                "https://api.exchange.coinbase.com/products/BTC-USD/candles"
            ),
            "coinbase_official_reference": (
                "https://docs.cdp.coinbase.com/api-reference/exchange-api/rest-api/"
                "products/get-product-candles"
            ),
            "coinbase_rate_limit_reference": (
                "https://docs.cdp.coinbase.com/exchange/rest-api/rate-limits"
            ),
            "coinbase_live_reference": (
                "https://docs.cdp.coinbase.com/exchange/websocket-feed/overview"
            ),
            "coinbase_product": "BTC-USD",
            "coinbase_granularity_seconds": 300,
            "coinbase_max_candles_per_request": 300,
            "coinbase_schema": ["time", "low", "high", "open", "close", "volume"],
            "historical_interval_filter": (
                "sort, deduplicate, and exact-filter returned bucket starts because "
                "the endpoint may return rows outside the requested interval"
            ),
            "coinbase_no_tick_intervals": "missing; never forward-filled",
            "binance_signal_leg": (
                "BTCUSDT perpetual completed 5m OHLC and quote_asset_volume"
            ),
            "execution_leg": "Binance BTCUSDT perpetual",
            "historical_snapshot_is_point_in_time": False,
            "live_parity": (
                "synthesize 5m Coinbase candles from matches, monitor heartbeat and "
                "trade IDs, and recover detected gaps with the public REST trades API"
            ),
            "promotion_requires_forward_websocket_parity": True,
        },
        "currency_contract": {
            "coinbase_quote": "USD",
            "binance_quote": "USDT",
            "raw_cross_venue_ratio_is_not_pure_coinbase_alpha": True,
            "interpretation": (
                "the ratio includes Coinbase pressure, Binance futures basis, and "
                "USD/USDT variation; no claim isolates those components"
            ),
            "no_premium_control": (
                "a return/activity-only control must remain viable without the raw "
                "cross-currency price ratio"
            ),
        },
        "availability_contract": {
            "candle_timestamp": "bucket start",
            "feature_time": "only after both venues complete the 5m bucket",
            "execution": "next 5m Binance-perpetual open",
            "missing_coinbase_bar": (
                "quarantine the missing bucket and the next 12 signal buckets; do not impute"
            ),
            "partial_or_stale": "fail closed",
        },
        "feature_contract": {
            "coinbase_return": "log(Coinbase close_t / Coinbase close_t-1)",
            "binance_return": "log(Binance close_t / Binance close_t-1)",
            "relative_return": "coinbase_return - binance_return",
            "coinbase_quote_notional": "Coinbase BTC volume * Coinbase close",
            "binance_quote_notional": "Binance quote_asset_volume",
            "activity_share": (
                "coinbase_quote_notional / (coinbase_quote_notional + "
                "binance_quote_notional)"
            ),
            "logit_activity_share": (
                "log(clipped activity_share / (1 - clipped activity_share))"
            ),
            "premium": "log(Coinbase close / Binance-perpetual close)",
            "premium_residual": "premium minus strictly-prior 3d median(premium)",
            "premium_residual_change": (
                "premium_residual_t - premium_residual_t-1"
            ),
            "robust_z": (
                "(x - strictly-prior 30d median) / (1.4826 * strictly-prior MAD); "
                "minimum 14d observations, rolling references shifted one bar"
            ),
            "tokens": {
                "ZR": "robust_z(relative_return)",
                "ZP": "robust_z(premium_residual_change)",
                "ZV": "robust_z(logit_activity_share)",
                "ZCB": "robust_z(coinbase_return)",
                "ZBN": "robust_z(binance_return)",
            },
            "causal": "all bars completed and all distribution references strictly prior",
        },
        "policies": [asdict(policy) for policy in policy_grid()],
        "family_rules": {
            "relative_return_lead": "side*ZR >= 2, side*ZCB >= 1, side*ZBN < 1.5",
            "premium_shock": "side*ZP >= 2, side*ZCB >= 0.5",
            "activity_confirmed_relative": "ZV >= 2, side*ZCB >= 1.5, side*ZR >= 1",
            "activity_premium_confluence": "ZV >= 1.5, side*ZP >= 1.5",
            "return_premium_confluence": "side*ZR >= 1.5, side*ZP >= 1.5",
        },
        "support_freeze_before_returns": {
            "paired_family_nonoverlap_events_min_total": 120,
            "paired_family_nonoverlap_events_min_each_year": 25,
            "minimum_each_side_share": 0.20,
            "maximum_single_month_share": 0.20,
            "global_missing_or_quarantined_fraction_max": 0.01,
            "monthly_missing_or_quarantined_fraction_max": 0.03,
            "failure_action": "reject family without computing forward trade returns",
        },
        "selection_protocol": {
            "fit": ["2020-01-01", "2022-01-01"],
            "fit_windows": {
                "fit_2020": ["2020-01-01", "2021-01-01"],
                "fit_2021": ["2021-01-01", "2022-01-01"],
            },
            "selection": ["2022-01-01", SELECTION_END],
            "sealed_holdout": list(HOLDOUT),
            "future_2024_plus_sealed": True,
            "selection_inputs_physically_end_before": SELECTION_END,
            "rank": [
                "descending minimum(fit_2020, fit_2021, selection_2022) CAGR/strict_MDD",
                "descending combined_2020_2022 CAGR/strict_MDD",
                "ascending policy_id",
            ],
            "selection_gates": {
                "every_calendar_year_absolute_return_positive": True,
                "positive_half_years_min_of_six": 5,
                "strict_mdd_pct_max_each_year": 10.0,
                "combined_cagr_to_strict_mdd_min": 2.0,
                "combined_trades_min": 120,
                "each_calendar_year_trades_min": 25,
                "eight_bp_notional_side_cost_stress_absolute_return_positive": True,
                "familywise_weekly_cluster_signflip_p_max": 0.10,
            },
            "holdout_2023_gates": {
                "absolute_return_positive": True,
                "cagr_to_strict_mdd_min": 3.0,
                "strict_mdd_pct_max": 10.0,
                "trades_min": 20,
                "h1_absolute_return_nonnegative": True,
                "h2_absolute_return_nonnegative": True,
                "eight_bp_notional_side_cost_stress_absolute_return_positive": True,
                "one_bar_delay_absolute_return_positive": True,
                "familywise_weekly_cluster_signflip_p_max": 0.10,
            },
            "multiple_testing_hypotheses": len(policy_grid()),
            "familywise_adjustment": "Bonferroni over all 16 preregistered policies",
        },
        "execution_contract": {
            "entry_delay_bars": 1,
            "nonoverlap": True,
            "leverage": 0.5,
            "fee_rate_notional_per_side": 0.0005,
            "slippage_rate_notional_per_side": 0.0001,
            "base_cost_notional_per_side": 0.0006,
            "base_cost_account_per_side_at_half_leverage": 0.0003,
            "stress_cost_notional_per_side": 0.0008,
            "realized_funding": True,
            "take_stop": "none; fixed time exit",
            "strict_mdd": (
                "global/pre-entry high-water plus intratrade favorable-before-adverse "
                "OHLC extremes, entry and hypothetical liquidation cost/slippage, "
                "and realized funding debit"
            ),
            "cagr_clock": "full calendar including idle time",
        },
        "controls": {
            "direction_flip": True,
            "one_bar_delay": True,
            "twelve_bar_delay": True,
            "binance_leader_role_swap": True,
            "relative_return_ablation": True,
            "premium_ablation": True,
            "activity_ablation": True,
            "no_premium_currency_control": True,
            "time_of_week_matched_random_samples": 5000,
            "weekly_cluster_signflip_samples": 5000,
            "random_seed": 20260716,
        },
        "orthogonality_after_holdout": {
            "exact_entry_jaccard_max": 0.02,
            "candidate_entries_near_existing_6h_fraction_target": 0.10,
            "candidate_entries_near_existing_6h_fraction_max": 0.25,
            "position_jaccard_max": 0.15,
            "absolute_daily_pnl_pearson_max": 0.30,
            "minimum_nonzero_daily_pnl_days": 10,
            "undefined_metric": "fail_closed",
            "marginal_portfolio_improvement_required": True,
        },
    }
    return {
        **core,
        "manifest_hash": canonical_hash(core),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def validate_manifest(manifest: dict[str, Any]) -> None:
    core = {
        key: value
        for key, value in manifest.items()
        if key not in {"manifest_hash", "created_at"}
    }
    if canonical_hash(core) != manifest.get("manifest_hash"):
        raise RuntimeError("Coinbase leadership preregistration hash mismatch")
    if manifest.get("outcomes_opened") is not False:
        raise RuntimeError("Coinbase preregistration cannot open outcomes")
    if manifest.get("policies") != [asdict(policy) for policy in policy_grid()]:
        raise RuntimeError("Coinbase policy family differs from preregistration code")
    if manifest["selection_protocol"]["selection"][1] != SELECTION_END:
        raise RuntimeError("Selection boundary changed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def write_manifest_once(path: str | Path, payload: dict[str, Any]) -> str:
    """Create a preregistration once; later runs may only verify it byte-semantically."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = json.loads(output.read_text())
        validate_manifest(existing)
        if existing.get("manifest_hash") != payload.get("manifest_hash"):
            raise RuntimeError("refusing to overwrite an anchored preregistration")
        return "verified_existing"
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return "created"


def main() -> None:
    args = parse_args()
    payload = build_manifest()
    validate_manifest(payload)
    status = write_manifest_once(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "manifest_hash": payload["manifest_hash"],
                "policies": len(payload["policies"]),
                "status": status,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
