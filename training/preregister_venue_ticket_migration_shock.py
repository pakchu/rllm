"""Freeze VTMS-288 before opening any strategy outcome.

VTMS-288 tests whether an abrupt migration in average individual trade ticket
size between Binance Spot and USD-M identifies the venue currently carrying
the informative flow.  This module writes only a protocol manifest; it never
loads future OHLC, funding cashflow, return, CAGR, or drawdown.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = "results/venue_ticket_migration_shock_preregistration_2026-07-17.json"
SPOT_FEATURE_PATH = (
    "data/binance_spot_kline_microstructure_btc_2020_2023/"
    "BTCUSDT_spot_kline_microstructure_5m_2020-01_2023-12.csv.gz"
)
SPOT_MANIFEST_PATH = (
    "data/binance_spot_kline_microstructure_btc_2020_2023/build_manifest.json"
)
PERP_FEATURE_PATH = (
    "data/binance_um_aggtrade_microstructure_btc_2020_2023/"
    "BTCUSDT_aggtrade_5m_2020-01-01_2023-12-31.csv.gz"
)
PERP_MANIFEST_PATH = (
    "data/binance_um_aggtrade_microstructure_btc_2020_2023/build_manifest.json"
)
MARKET_PATH = (
    "data/binance_um_kline_reference_btc_2020_2023/"
    "BTCUSDT_5m_2020-01-01_2023-12-31.csv.gz"
)
MARKET_MANIFEST_PATH = (
    "data/binance_um_kline_reference_btc_2020_2023/build_manifest.json"
)


@dataclass(frozen=True)
class Policy:
    policy_id: str = "VTMS-288"
    baseline_bars: int = 8_640
    baseline_min_periods: int = 2_016
    ticket_level_quantile: float = 0.95
    ticket_change_quantile: float = 0.975
    ticket_change_bars: int = 12
    coherence_quantile: float = 0.75
    response_quantile: float = 0.75
    minimum_perp_agg_trade_count: int = 64
    episode_reset_bars: int = 1
    execution_delay_bars: int = 2
    hold_bars: int = 288
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
        "protocol_version": "venue_ticket_migration_shock_v1",
        "outcomes_opened": False,
        "policy": asdict(policy),
        "research_history_boundary": {
            "market_history_seen_by_unrelated_repo_research": True,
            "exact_vtms_288_outcomes_opened": False,
            "support_only_values_seen": (
                "current-bar ticket, flow, response distributions and event counts; "
                "no entry-to-exit price, future OHLC, funding PnL, return, CAGR, or MDD"
            ),
        },
        "novelty_boundary": {
            "excluded_feature_families": [
                "open_interest",
                "funding_or_premium",
                "kimchi_or_fx",
                "REX_or_HTF_price_regime",
                "Markov_state",
                "event_notional_tail_or_arrival_burstiness",
                "fill_compression_level",
            ],
            "nearest_failed_families": {
                "Coinbase_leadership": "venue return/activity leadership, not ticket migration",
                "CVTT": "within-venue flow/return timing torsion",
                "CSPR_RIFT": "cash/perp flow rejection and path-quality transitions",
                "AFCS": "perp aggregate-event fill compression continuation",
            },
            "distinct_axis": (
                "relative average individual Spot-versus-USD-M trade ticket size and "
                "its abrupt one-hour migration shock"
            ),
        },
        "source_contract": {
            "spot_features": SPOT_FEATURE_PATH,
            "spot_manifest": SPOT_MANIFEST_PATH,
            "spot_feature_sha256": (
                "d558239fa7085083aa002b7898b632df0774425719467709680ecb99718035a9"
            ),
            "spot_manifest_sha256": (
                "69fbce64b4860eecbf1ce414ea719b5c4001852016fe439e61240e050b39b57b"
            ),
            "spot_audit": "results/binance_spot_kline_microstructure_audit_2026-07-14.json",
            "spot_audit_sha256": (
                "2e2faf8d603d84519cd4a335b1c58d7bbe25e2bbeee1de50f725fd8d93288c59"
            ),
            "perp_features": PERP_FEATURE_PATH,
            "perp_manifest": PERP_MANIFEST_PATH,
            "perp_feature_sha256": (
                "c2bb0e6742f8cdc4e13315e7f0a13d6ab9cd536fb40d9cb4484b7a6ba30131cf"
            ),
            "perp_manifest_sha256": (
                "6eec40460a6146c58994e52f1af9ace4eecc0c085887d97af5ef17c30b9f7e73"
            ),
            "perp_audit": "results/binance_aggtrade_microstructure_audit_2026-07-14.json",
            "perp_audit_sha256": (
                "5ac5a342d7f766ea0b6dcf9f97468ab70b9e1194775469ed0245d9208d0dc9c6"
            ),
            "market": MARKET_PATH,
            "market_manifest": MARKET_MANIFEST_PATH,
            "market_sha256": (
                "e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d"
            ),
            "market_manifest_sha256": (
                "c04fbbd299cc748a6745c0ef030787da4d560833c744c81c98dd8840efc7913e"
            ),
            "available_end_exclusive": "2024-01-01",
            "source_gap_policy": (
                "quarantine every incomplete Spot bucket, each verified USD-M "
                "source-ID-gap UTC day, every missing USD-M feature bucket, and "
                "the following 24 five-minute bars; never impute"
            ),
        },
        "claim_boundary": {
            "claim": (
                "an abrupt cross-venue migration in average individual ticket size, "
                "when accepted by coherent same-venue price response, may identify "
                "which venue's flow leads the next one-day repricing"
            ),
            "not_claimed": [
                "participant identity or wealth",
                "one ticket equals one parent order",
                "resting-book depth or queue position",
                "Spot and USD-M trade records have identical matching semantics",
            ],
        },
        "causal_feature_contract": {
            "bucket": "one completed five-minute bucket t",
            "spot_ticket": "spot quote_notional[t] / spot trade_count[t]",
            "perp_ticket": (
                "USD-M quote_notional[t] / inclusive underlying trade-ID count[t]"
            ),
            "ticket_ratio": "r[t] = log(spot_ticket[t] / perp_ticket[t])",
            "ticket_change": "dr[t] = r[t] - r[t-12]",
            "spot_branch": (
                "r[t] >= prior q95 and dr[t] >= prior q97.5; Spot coherence and "
                "signed price response each >= own prior q75; side=sign(Spot flow)"
            ),
            "perp_branch": (
                "r[t] <= prior q5 and dr[t] <= prior q2.5; USD-M coherence and "
                "signed price response each >= own prior q75; side=sign(USD-M flow)"
            ),
            "baseline": (
                "all thresholds use only clean t-8640..t-1 observations and "
                "require at least 2016 observations"
            ),
            "episode": "either branch active at t and neither branch active at t-1",
        },
        "execution_contract": {
            "decision_time": "after bucket t closes",
            "entry": "t+2 open, leaving one full five-minute computation bar",
            "exit": "scheduled open after 288 held five-minute bars",
            "nonoverlap": True,
            "position_size": "0.5x account gross",
            "base_cost": "6 bp/notional/side",
            "stress_cost": "10 bp/notional/side",
            "funding": "exact realized BTCUSDT settlement rates on [entry, exit)",
            "strict_mdd": (
                "global/pre-entry high-water; entry cost; favorable-before-adverse "
                "held OHLC; exact funding; hypothetical liquidation and exit cost"
            ),
            "cagr_clock": "full wall-clock split including warm-up and idle cash",
        },
        "support_freeze_before_returns": {
            "minimum_nonoverlap_train_2020_2022": 300,
            "minimum_nonoverlap_each_train_year": 75,
            "minimum_nonoverlap_2023": 90,
            "minimum_nonoverlap_each_2023_half": 40,
            "minimum_each_side_share": 0.30,
            "minimum_each_branch_share": 0.30,
            "maximum_single_month_share": 0.10,
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
                "2023_trades_min": 75,
                "2023_h1_and_h2_absolute_return_positive": True,
                "2023_h1_and_h2_trades_min": 35,
                "train_and_2023_mean_gross_underlying_bp_min": 20.0,
                "train_and_2023_ten_bp_stress_absolute_return_positive": True,
            },
        },
        "falsification_controls": {
            "direction_flip": "same primary clock, exact opposite side",
            "no_ticket_level": "own clock with q95/q5 ticket level removed",
            "no_ticket_shock": "own clock with q97.5/q2.5 one-hour shock removed",
            "no_coherence": "own clock with dominant-venue coherence removed",
            "no_price_acceptance": "own clock with signed-response threshold removed",
            "other_venue_side": "same clock, side from the non-dominant venue flow",
            "one_hour_signal_delay": "same branch side and hold, signal shifted 12 bars",
            "one_day_shifted_clock": "same branch side and hold, signal shifted 288 bars",
            "random_side": "same clock, fixed-seed 20260717 Rademacher sides",
            "mechanism_gate": (
                "primary train and 2023 CAGR-MDD must beat every component removal; "
                "one-hour, one-day, or random-side full qualification rejects"
            ),
        },
        "orthogonality_after_standalone_pass": {
            "exact_entry_jaccard_max": 0.02,
            "near_six_hour_entry_fraction_max": 0.25,
            "position_time_jaccard_max": 0.15,
            "absolute_daily_pnl_pearson_max": 0.30,
            "marginal_portfolio_improvement_required": True,
        },
        "rejection_contract": (
            "any support or pre-2024 performance failure retires VTMS-288; do not "
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
        raise RuntimeError("VTMS-288 preregistration hash mismatch")
    if payload.get("outcomes_opened") is not False:
        raise RuntimeError("VTMS-288 preregistration cannot open outcomes")
    if payload.get("policy") != asdict(Policy()):
        raise RuntimeError("VTMS-288 frozen policy differs from code")
    if payload.get("selection_protocol", {}).get("candidate_count") != 1:
        raise RuntimeError("VTMS-288 must remain a singleton policy")


def write_manifest_once(path: str | Path, payload: dict[str, Any]) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = json.loads(output.read_text())
        validate_manifest(existing)
        if existing["manifest_hash"] != payload["manifest_hash"]:
            raise RuntimeError("refusing to overwrite frozen VTMS-288 preregistration")
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
