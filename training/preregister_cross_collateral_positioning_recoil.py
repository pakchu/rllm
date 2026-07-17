"""Freeze CCPR-1 before opening any strategy outcome.

CCPR-1 tests whether a source-only crowding burst that is stronger in the
USD-margined BTC perpetual than in the coin-margined BTC perpetual subsequently
recoils.  This module writes a deterministic protocol manifest.  It never
parses executable prices, future returns, funding cash flow, portfolio PnL,
CAGR, or drawdown.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = (
    "results/cross_collateral_positioning_recoil_preregistration_2026-07-17.json"
)
POSITIONING_PATH = (
    "data/binance_cross_collateral_metrics_btc_2021_2023/"
    "BTC_cross_collateral_metrics_5m_2021-07-08_2023-12-31.csv.gz"
)
POSITIONING_MANIFEST_PATH = (
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
    policy_id: str = "CCPR-1"
    usd_m_symbol: str = "BTCUSDT"
    coin_m_symbol: str = "BTCUSD_PERP"
    anchor_minute: int = 55
    oi_change_bars: int = 72
    taker_median_bars: int = 12
    prior_rank_hourly_anchors: int = 168
    rotation_quantiles: tuple[float, ...] = (0.80, 0.85, 0.90)
    taker_rank_floor: float = 0.60
    execution_delay_bars: int = 2
    hold_bars: tuple[int, ...] = (48, 96)
    leverage: float = 0.5
    base_cost_notional_per_side: float = 0.0006
    stress_cost_notional_per_side: float = 0.0010


SOURCE_HASHES = {
    "positioning_sha256": (
        "ab9f18ba7745f21b17ac1124c45bb755245d404d66100c595bb77631f4bc1757"
    ),
    "positioning_manifest_sha256": (
        "c0732ca47451209a9bb519545b0e349550994d870d476ee66ecbae81588fb159"
    ),
    "market_sha256": (
        "e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d"
    ),
    "market_manifest_sha256": (
        "c04fbbd299cc748a6745c0ef030787da4d560833c744c81c98dd8840efc7913e"
    ),
    "funding_sha256": (
        "3284bbb6bb67946acb673c6b67459543e217f752589e1d47b6c7c3b659f733e6"
    ),
    "funding_manifest_sha256": (
        "a0b2d27e1aa8cf2d9ab8cb659b598ee0a6d7bd25401c9e10ae92d1a74415845b"
    ),
}


COMPARATOR_ARTIFACTS = {
    "deduplicated_universe_audit": {
        "path": "results/all_discovered_alpha_universe_audit_2026-07-12.json",
        "sha256": "5bd45f5949cfae7308dfadd3966112e06c7c09dbb92365796237c3b94be85a3e",
    },
    "family_capped_portfolio": {
        "path": (
            "results/portfolio_all_discovered_dedup_familycap2_"
            "trainmdd40_oosmdd20_2026-07-12.json"
        ),
        "sha256": "3becd1ac54e2b12b345036e64652eec14df3e69b216700d9d19e31732396d7a2",
    },
    "added_alpha_shadow": {
        "path": "results/portfolio_added_alpha_update_2026-07-16.json",
        "sha256": "e188917265d986b64b65fc854725f14d5a26372597f0923621d6bd721a468e0c",
    },
    "current_live_config": {
        "path": "configs/live/portfolio_gross385_trainmdd40_2026-07-12.json",
        "sha256": "86f255ca3967245b8b0676b00025b955d7f33668ab1ef9d813623191b4ecd1e7",
    },
}


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def policy_payload() -> dict[str, Any]:
    payload = asdict(Policy())
    payload["rotation_quantiles"] = list(payload["rotation_quantiles"])
    payload["hold_bars"] = list(payload["hold_bars"])
    return payload


def build_manifest() -> dict[str, Any]:
    core: dict[str, Any] = {
        "protocol_version": "cross_collateral_positioning_recoil_v1",
        "as_of_date": "2026-07-17",
        "outcomes_opened": False,
        "policy": policy_payload(),
        "research_history_boundary": {
            "market_history_seen_by_unrelated_repo_research": True,
            "positioning_source_values_seen": True,
            "source_only_density_preflight_seen": True,
            "full_2021_2023_source_support_seen_before_freeze": True,
            "density_preflight_disclosure": {
                "definition": (
                    "the frozen feature equations below were evaluated without "
                    "joining OHLC, funding, returns, labels, or portfolio outcomes"
                ),
                "episode_counts_by_q": {
                    "0.80": {"2021_partial": 47, "2022": 100, "2023": 66},
                    "0.85": {"2021_partial": 35, "2022": 78, "2023": 49},
                    "0.90": {"2021_partial": 24, "2022": 51, "2023": 34},
                },
                "q_0_90_2023_halves": {"H1": 25, "H2": 9},
                "q_0_85_2023_halves": {"H1": 37, "H2": 12},
                "consequence": (
                    "support floors are power and temporal-coverage constraints; "
                    "they cannot be changed after this freeze"
                ),
            },
            "exact_ccpr_1_post_entry_outcomes_opened": False,
            "support_only_access_allowed": (
                "source values, causal source-only features, event counts, "
                "timestamps, side balance, and control-clock overlap"
            ),
            "forbidden_before_evaluator_freeze": [
                "entry_to_exit_return",
                "post_entry_OHLC",
                "funding_PnL",
                "win_rate",
                "absolute_return",
                "CAGR",
                "drawdown",
                "existing_alpha_return_overlap",
            ],
        },
        "novelty_boundary": {
            "distinct_axis": (
                "relative positioning and aggressive-flow crowding between the "
                "same BTC perpetual under USD-margined and coin-margined "
                "collateral constituencies"
            ),
            "excluded_inputs": [
                "OHLC_or_price_return",
                "REX_or_rolling_extrema",
                "funding_premium_basis",
                "kimchi_BTCKRW_USDKRW_or_DXY",
                "Upbit_volume_or_regional_spot_flow",
                "orderbook_or_aggregate_trade_tail_shape",
                "cross_venue_price_lead_lag",
                "volatility_options_CFTC_or_onchain",
                "existing_alpha_state_or_portfolio_PnL",
            ],
            "nearest_existing_families": {
                "oi_upbit": (
                    "uses one futures OI stream plus Korean spot volume; CCPR "
                    "uses the relative change between two collateralized futures "
                    "constituencies and no spot or FX input"
                ),
                "cross_collateral_basis": (
                    "uses price spread or order-book credibility; CCPR excludes "
                    "all price and book fields and uses positioning plus taker ratios"
                ),
                "funding_premium": (
                    "uses carry state; CCPR excludes funding and premium from signals"
                ),
            },
            "forbidden_repairs_after_outcomes": [
                "reverse the primary recoil direction",
                "add a BTC price or regime gate",
                "add funding premium kimchi FX or Upbit inputs",
                "change the anchor minute or source availability delay",
                "change the 6h OI or 1h taker windows",
                "change the prior-only rank length or quantile grid",
                "change the taker rank floor",
                "change the hold family",
                "drop either the OI-rotation or taker-concordance premise",
            ],
        },
        "source_contract": {
            "source_commit": "8d347432cd36d59458ad9a26c7c8aef1ec94b8ee",
            "positioning": POSITIONING_PATH,
            "positioning_sha256": SOURCE_HASHES["positioning_sha256"],
            "positioning_manifest": POSITIONING_MANIFEST_PATH,
            "positioning_manifest_sha256": SOURCE_HASHES["positioning_manifest_sha256"],
            "source_audit": (
                "docs/binance-cross-collateral-positioning-metrics-source-audit-"
                "2026-07-17.md"
            ),
            "market": MARKET_PATH,
            "market_sha256": SOURCE_HASHES["market_sha256"],
            "market_manifest": MARKET_MANIFEST_PATH,
            "market_manifest_sha256": SOURCE_HASHES["market_manifest_sha256"],
            "funding": FUNDING_PATH,
            "funding_sha256": SOURCE_HASHES["funding_sha256"],
            "funding_manifest": FUNDING_MANIFEST_PATH,
            "funding_manifest_sha256": SOURCE_HASHES["funding_manifest_sha256"],
            "available_start": "2021-07-08T00:00:00Z",
            "available_end_exclusive": "2024-01-01T00:00:00Z",
            "gap_policy": (
                "no fill and no stale carry; a current anchor, its complete 73-row "
                "6h endpoint path, and all 168 prior hourly rank anchors must be "
                "source-complete, so a gap quarantines signals until the entire "
                "7d+6h causal history has rebuilt"
            ),
        },
        "causal_feature_contract": {
            "availability": (
                "a metrics row timestamp t is observed as completed source state; "
                "wait one full 5m availability bucket and enter only at the next "
                "open t+10m (execution_delay_bars=2)"
            ),
            "anchor": "UTC hourly source rows whose minute is exactly 55",
            "source_only_columns": [
                "um_sum_open_interest_value",
                "cm_sum_open_interest",
                "um_sum_taker_long_short_vol_ratio",
                "cm_sum_taker_long_short_vol_ratio",
                "source_complete",
            ],
            "oi_rotation": (
                "R[t]=log(UM_OI_value[t]/UM_OI_value[t-72])-"
                "log(CM_OI_contracts[t]/CM_OI_contracts[t-72])"
            ),
            "taker_gap": (
                "T[t]=median over completed rows t-11..t of "
                "log(UM_taker_ratio)-log(CM_taker_ratio)"
            ),
            "strict_prior_ranks": (
                "A[t] and G[t] are empirical mid-ranks of abs(R[t]) and abs(T[t]) "
                "against exactly the 168 hourly anchors immediately before t; "
                "current t is excluded and every supporting source row must be complete"
            ),
            "setup": (
                "A[t]>=Q AND G[t]>=0.60 AND sign(R[t])=sign(T[t])!=0; "
                "a false-to-true transition creates one episode"
            ),
            "action": "side[t]=-sign(T[t]); fade the concordant collateral crowding burst",
            "price_signal_columns": [],
        },
        "support_calibration": {
            "vary_only": "rotation quantile Q in [0.80, 0.85, 0.90]",
            "selection_rule": (
                "select the highest Q passing every source-only density, temporal "
                "coverage, side-balance, concentration, and control-overlap floor; "
                "2023 support can reject but never select an outcome-driven fallback"
            ),
            "train_window": ["2021-07-08", "2023-01-01"],
            "support_seen_outcome_sealed_window": ["2023-01-01", "2024-01-01"],
            "minimum_train_episodes": 100,
            "minimum_2021_partial_episodes": 30,
            "minimum_2022_episodes": 70,
            "minimum_2023_episodes": 40,
            "minimum_each_2023_half": 10,
            "minimum_each_side_share": 0.25,
            "maximum_single_month_share": 0.20,
            "maximum_signal_jaccard": {
                "oi_only": 0.85,
                "taker_only": 0.85,
                "um_only": 0.85,
                "cm_only": 0.85,
            },
        },
        "execution_contract": {
            "instrument": "Binance USD-M BTCUSDT perpetual",
            "fill": "next 5m open at source timestamp t+10m",
            "candidate_holds": {"CCPR-H4": 48, "CCPR-H8": 96},
            "episode_deduplication": (
                "reserve false-to-true clocks chronologically and suppress any clock "
                "whose entry would occur before the previous candidate exit"
            ),
            "sizing": "fixed 0.5x notional, one position at a time",
            "costs": (
                "6bp notional per side primary; 10bp per side mandatory stress; "
                "realized funding marks applied with the frozen strict simulator"
            ),
            "strict_mdd": (
                "same BTC low/high intratrade strict path plus entry-before-position "
                "equity path; adverse bound is applied before favorable bound"
            ),
            "clock_reservation": (
                "feature history may precede a split start, but signal, entry, and "
                "exit must all be physically inside the evaluated split"
            ),
        },
        "falsification_controls": {
            "oi_only": "A[t]>=Q transition; side=-sign(R[t])",
            "taker_only": "G[t]>=0.60 transition; side=-sign(T[t])",
            "um_only": (
                "UM 6h OI-value absolute rank>=Q and UM 1h taker absolute rank>=0.60 "
                "with sign concordance; fade UM taker sign"
            ),
            "cm_only": (
                "CM 6h OI-contract absolute rank>=Q and CM 1h taker absolute "
                "rank>=0.60 with sign concordance; fade CM taker sign"
            ),
            "direction_flip": "same primary entries with side multiplied by -1",
            "entry_shift_plus_1h": "same side and episode, entry and exit shifted +12 bars",
            "deterministic_random_side": (
                "same primary entries, side from SHA256(policy_id|signal timestamp)"
            ),
        },
        "selection_protocol": {
            "candidate_count_after_support": 2,
            "stage1": {
                "window": ["2021-07-08", "2023-01-01"],
                "subperiods": ["2021_partial", "2022_H1", "2022_H2"],
                "rank_rule": (
                    "among candidates passing every gate, maximize Stage1 CAGR/MDD; "
                    "ties prefer lower strict MDD then shorter hold"
                ),
                "gates": {
                    "absolute_return_positive": True,
                    "cagr_mdd_min": 3.0,
                    "strict_mdd_max_pct": 15.0,
                    "trades_min": 80,
                    "weekly_cluster_signflip_p_max": 0.025,
                    "each_subperiod_absolute_return_positive": True,
                    "stress_absolute_return_positive": True,
                    "stress_cagr_mdd_min": 2.5,
                    "mechanism_control_margin_min": 0.25,
                },
            },
            "stage2": {
                "window": ["2023-01-01", "2024-01-01"],
                "opened_only_if_stage1_passes": True,
                "cannot_reselect_or_repair": True,
                "subperiods": ["2023_H1", "2023_H2"],
                "gates": {
                    "absolute_return_positive": True,
                    "cagr_mdd_min": 3.0,
                    "strict_mdd_max_pct": 15.0,
                    "trades_min": 25,
                    "weekly_cluster_signflip_p_max": 0.05,
                    "each_subperiod_absolute_return_positive": True,
                    "stress_absolute_return_positive": True,
                    "stress_cagr_mdd_min": 2.5,
                },
            },
            "statistical_test_contract": {
                "test": "two-sided weekly-cluster sign-flip on net trade PnL",
                "draws": 20_000,
                "seed": 20_260_717,
            },
            "control_comparison_contract": {
                "mechanism_controls": ["oi_only", "taker_only", "um_only", "cm_only"],
                "stage1": (
                    "primary CAGR/MDD must exceed every mechanism control by at "
                    "least 0.25; equality rejects"
                ),
                "all_controls": (
                    "every falsification control is reported under the complete "
                    "profitability, risk, significance, trade-count, stress, and "
                    "subperiod battery rather than headline-only metrics"
                ),
            },
        },
        "orthogonality_after_standalone_pass": {
            "economic_gate_first": True,
            "comparator_universe": COMPARATOR_ARTIFACTS,
            "duplicate_policy": (
                "collapse exact PnL hashes and compare against every canonical "
                "family representative plus synchronized selected portfolios"
            ),
            "exact_entry_jaccard_max": 0.02,
            "near_six_hour_entry_fraction_max": 0.20,
            "position_time_jaccard_max": 0.15,
            "absolute_daily_pnl_pearson_max": 0.30,
            "minimum_nonzero_pnl_days": 20,
            "marginal_portfolio_improvement_required": True,
        },
        "rllm_boundary": {
            "standalone_alpha_is_formulaic": True,
            "llm_not_allowed_to_create_or_repair_signals": True,
            "future_llm_role_after_validation": (
                "explain source state or abstain under a separately frozen policy; "
                "it may not inspect future returns or tune CCPR thresholds"
            ),
        },
        "rejection_contract": {
            "if_support_fails": "reject without opening any execution outcome",
            "if_stage1_fails": "reject and keep 2023 execution outcomes sealed",
            "if_stage2_fails": "reject without threshold hold direction or gate repair",
            "no_salvage": (
                "a failed candidate may remain a source feature for a future clean "
                "family, but CCPR-1 itself cannot be tuned further"
            ),
        },
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate_manifest(manifest: dict[str, Any], *, verify_sources: bool = True) -> None:
    if manifest.get("outcomes_opened") is not False:
        raise ValueError("CCPR-1 preregistration has outcomes opened")
    expected_policy = policy_payload()
    if manifest.get("policy") != expected_policy:
        raise ValueError("CCPR-1 frozen policy differs")
    expected_hash = canonical_hash(
        {key: value for key, value in manifest.items() if key != "manifest_hash"}
    )
    if manifest.get("manifest_hash") != expected_hash:
        raise ValueError("CCPR-1 manifest hash mismatch")
    if manifest["causal_feature_contract"].get("price_signal_columns") != []:
        raise ValueError("CCPR-1 signal must remain price blind")
    if verify_sources:
        source = manifest["source_contract"]
        for key, hash_key in (
            ("positioning", "positioning_sha256"),
            ("positioning_manifest", "positioning_manifest_sha256"),
            ("market", "market_sha256"),
            ("market_manifest", "market_manifest_sha256"),
            ("funding", "funding_sha256"),
            ("funding_manifest", "funding_manifest_sha256"),
        ):
            if _sha256(source[key]) != source[hash_key]:
                raise ValueError(f"CCPR-1 source hash mismatch: {key}")
        for item in manifest["orthogonality_after_standalone_pass"][
            "comparator_universe"
        ].values():
            if _sha256(item["path"]) != item["sha256"]:
                raise ValueError(f"CCPR-1 comparator hash mismatch: {item['path']}")


def write_manifest(path: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    manifest = build_manifest()
    validate_manifest(manifest)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = write_manifest(args.output)
    print(json.dumps(manifest, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
