"""Freeze CMSR-36 before opening any BTCUSDT execution outcome."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = "results/coinm_next_maturity_shock_relay_preregistration_2026-07-19.json"
SOURCE_PATH = (
    "data/binance_coinm_quarterly_strip_pre2024_v2/"
    "BTCUSD_front_next_quarterly_5m_20200701T0000_20231231T2350.csv.gz"
)
SOURCE_MANIFEST_PATH = "data/binance_coinm_quarterly_strip_pre2024_v2/build_manifest.json"
MARKET_PATH = (
    "data/binance_um_kline_reference_btc_2020_2023/"
    "BTCUSDT_5m_2020-01-01_2023-12-31.csv.gz"
)
MARKET_MANIFEST_PATH = "data/binance_um_kline_reference_btc_2020_2023/build_manifest.json"
FUNDING_PATH = "data/binance_um_btcusdt_funding_marks_2020_2023.csv.gz"
FUNDING_MANIFEST_PATH = (
    "results/binance_um_btcusdt_funding_marks_2020_2023_manifest_2026-07-17.json"
)


@dataclass(frozen=True)
class Policy:
    policy_id: str = "CMSR-36"
    path_bars: int = 24
    share_edge_bars: int = 6
    prior_window_bars: int = 8_640
    prior_min_periods: int = 6_000
    prior_nonoverlap_shift_bars: int = 24
    share_slope_quantile: float = 0.90
    next_flow_abs_quantile: float = 0.80
    lead_shock_abs_quantile: float = 0.80
    front_to_next_return_abs_max: float = 0.60
    entry_delay_from_signal_bars: int = 2
    hold_bars: int = 36
    delivery_buffer_hours: float = 12.0
    leverage: float = 0.5
    base_cost_notional_per_side: float = 0.0006
    stress_cost_notional_per_side: float = 0.0010


SOURCE_HASHES = {
    "source_sha256": "d2126e546fa890c3537610a59c0341cb8153c38861d42b59477b340280ced30b",
    "source_manifest_sha256": (
        "29a886f788776dcb3fd8b69b78798bf70ef5e092b54765437a63231c4ffb87af"
    ),
    "market_sha256": "e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d",
    "market_manifest_sha256": (
        "c04fbbd299cc748a6745c0ef030787da4d560833c744c81c98dd8840efc7913e"
    ),
    "funding_sha256": "3284bbb6bb67946acb673c6b67459543e217f752589e1d47b6c7c3b659f733e6",
    "funding_manifest_sha256": (
        "a0b2d27e1aa8cf2d9ab8cb659b598ee0a6d7bd25401c9e10ae92d1a74415845b"
    ),
}


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def policy_payload() -> dict[str, Any]:
    return asdict(Policy())


def build_manifest() -> dict[str, Any]:
    core: dict[str, Any] = {
        "protocol_version": "coinm_next_maturity_shock_relay_v1",
        "as_of_date": "2026-07-19",
        "outcomes_opened": False,
        "policy": policy_payload(),
        "research_history_boundary": {
            "market_history_seen_by_unrelated_repo_research": True,
            "source_only_incidence_grid_seen": True,
            "exact_cmsr_36_post_entry_btcusdt_outcomes_opened": False,
            "incidence_grid": {
                "share_slope_quantiles": [0.80, 0.85, 0.90],
                "next_flow_abs_quantiles": [0.75, 0.80, 0.85],
                "lead_shock_abs_quantiles": [0.70, 0.75, 0.80],
                "cells": 27,
                "selection_rule": (
                    "among cells passing every outcome-free support gate, maximize "
                    "share quantile, then flow quantile, then lead-shock quantile"
                ),
                "selected": [0.90, 0.80, 0.80],
                "selected_support": {
                    "fit": 93,
                    "fit_half_years": [19, 16, 21, 12, 25],
                    "2023": 65,
                    "2023_halves": [35, 30],
                    "fit_minimum_side_share": 0.44,
                    "2023_minimum_side_share": 0.40,
                    "fit_max_month_share": 0.10,
                    "2023_max_month_share": 0.20,
                    "fit_max_pair_share": 0.16,
                    "2023_max_pair_share": 0.32,
                },
                "strictest_flow_0_85_failure": (
                    "every share=0.90/flow=0.85 cell fails the fixed 2022H1 "
                    "minimum-12 support floor; no return selected the fallback"
                ),
            },
            "forbidden_before_evaluator_freeze": [
                "BTCUSDT_entry_or_later_OHLC",
                "BTCUSDT_entry_to_exit_return",
                "funding_cash_flow",
                "strategy_PnL",
                "absolute_return",
                "CAGR",
                "strict_MDD",
                "win_rate",
            ],
        },
        "mechanism": {
            "claim": (
                "a sustained rise in next-quarter contract share accompanied by "
                "accepted next-contract aggressor flow and a next-over-front price "
                "shock can relay into the common BTC perpetual after the completed path"
            ),
            "why_not_prior_roll_repair": (
                "the rejected roll policy used one completed bar, pressure times "
                "sqrt(volume), and a 60m continuation on the next COIN-M contract. "
                "CMSR uses a 24-bar path transition, raw normalized flow, pair-local "
                "nonoverlapping 30d ranks, relative price-shock leadership, delayed "
                "BTCUSDT execution, and a fixed 3h relay horizon"
            ),
            "why_not_ccpr": (
                "CMSR uses dated COIN-M front/next volume, flow, and completed price "
                "paths; it uses no USD-M/COIN-M perpetual OI or taker-ratio gap"
            ),
        },
        "source_contract": {
            "source": SOURCE_PATH,
            "source_sha256": SOURCE_HASHES["source_sha256"],
            "source_manifest": SOURCE_MANIFEST_PATH,
            "source_manifest_sha256": SOURCE_HASHES["source_manifest_sha256"],
            "source_audit": "docs/binance-coinm-quarterly-strip-source-design-2026-07-19.md",
            "market": MARKET_PATH,
            "market_sha256": SOURCE_HASHES["market_sha256"],
            "market_manifest": MARKET_MANIFEST_PATH,
            "market_manifest_sha256": SOURCE_HASHES["market_manifest_sha256"],
            "funding": FUNDING_PATH,
            "funding_sha256": SOURCE_HASHES["funding_sha256"],
            "funding_manifest": FUNDING_MANIFEST_PATH,
            "funding_manifest_sha256": SOURCE_HASHES["funding_manifest_sha256"],
            "available_start": "2020-07-01T00:00:00Z",
            "available_end_exclusive": "2024-01-01T00:00:00Z",
            "gap_policy": (
                "all 24 current path rows must be valid, exact-grid, and retain the "
                "same front/next pair; no fill, promotion, or stale carry"
            ),
        },
        "causal_feature_contract": {
            "current_path": "24 completed five-minute rows ending at signal bar t",
            "next_share": "next_volume/(front_volume+next_volume)",
            "share_slope": (
                "median(next_share[t-5:t])-median(next_share[t-23:t-18])"
            ),
            "next_flow": (
                "sum(t-23:t, 2*next_taker_buy_volume-next_volume)/"
                "sum(t-23:t, next_volume)"
            ),
            "next_return": "log(next_close[t]/next_open[t-23])",
            "front_return": "log(front_close[t]/front_open[t-23])",
            "lead_shock": "next_return-front_return",
            "strict_prior_thresholds": (
                "pair-local rolling quantiles over 8,640 prior anchors with at least "
                "6,000 valid values; the series is shifted 24 bars so no reference "
                "feature path overlaps the current 24-bar path"
            ),
            "setup": (
                "share_slope>=prior_q90; abs(next_flow)>=prior_q80; "
                "abs(lead_shock)>=prior_q80; sign(next_flow)=sign(next_return)="
                "sign(lead_shock)!=0; abs(front_return)<=0.60*abs(next_return); "
                "only a false-to-true transition within the same pair may signal"
            ),
            "action": "side=sign(next_flow) on BTCUSDT USD-M perpetual",
            "price_columns_are_completed_signal_inputs_only": [
                "front_open",
                "front_close",
                "next_open",
                "next_close",
            ],
            "forbidden_feature_columns": [
                "BTCUSDT_execution_OHLC",
                "post_signal_front_or_next_price",
                "funding",
                "future_return_or_PnL",
                "existing_alpha_or_portfolio_state",
            ],
        },
        "support_gate": {
            "fit_window": ["2020-08-01", "2023-01-01"],
            "test_support_window": ["2023-01-01", "2024-01-01"],
            "minimum_fit_events": 90,
            "minimum_each_fit_half_year": 12,
            "minimum_2023_events": 30,
            "minimum_each_2023_half": 10,
            "minimum_each_side_share": 0.25,
            "fit_max_month_share": 0.18,
            "test_max_month_share": 0.25,
            "fit_max_pair_share": 0.35,
            "test_max_pair_share": 0.45,
            "old_roll_exact_signal_jaccard_max": 0.10,
            "old_roll_near_10m_containment_max": 0.25,
            "other_clock_exact_entry_jaccard_max": 0.10,
            "other_clock_near_6h_containment_max": 0.25,
        },
        "execution_contract": {
            "instrument": "Binance USD-M BTCUSDT perpetual",
            "feature_available": "signal bar t closes at t+5m",
            "entry": "leave one complete 5m bucket empty; enter at signal open t+10m",
            "hold": "36 five-minute bars / three hours",
            "delivery_safety": (
                "at signal t both contracts must retain at least the 12h buffer plus "
                "the 3h hold and 10m signal-to-entry latency before delivery"
            ),
            "nonoverlap": "one global BTCUSDT position reserved over [entry, exit)",
            "sizing": "fixed 0.5x notional",
            "costs": "6bp/notional/side base; 10bp/notional/side stress; exact funding",
            "strict_mdd": (
                "global/pre-entry HWM, entry cost, every held favorable-then-adverse "
                "5m OHLC path, conservative funding boundaries, virtual adverse-mark "
                "exit cost, and actual exit cost"
            ),
            "cagr": "full declared calendar including every idle second",
        },
        "falsification_controls": {
            "no_share_transition": "retain accepted flow and lead-shock gates only",
            "no_lead_shock": "retain share transition and accepted-flow gates only",
            "front_led_mirror": (
                "swap front/next roles in share, flow, accepted return, and lead shock"
            ),
            "direction_flip": "same CMSR entries with side multiplied by -1",
            "extra_latency_1h": "same signal and side, entry and exit delayed 12 bars",
            "deterministic_random_side": (
                "same entries with SHA256(policy_id|signal_time) side"
            ),
        },
        "selection_protocol": {
            "candidate_count": 1,
            "train": {
                "window": ["2020-08-01", "2023-01-01"],
                "subperiods": ["2020_H2", "2021_H1", "2021_H2", "2022_H1", "2022_H2"],
                "gates": {
                    "absolute_return_positive": True,
                    "cagr_mdd_min": 3.0,
                    "strict_mdd_max_pct": 15.0,
                    "trades_min": 90,
                    "weekly_cluster_signflip_p_max": 0.10,
                    "each_subperiod_absolute_return_positive": True,
                    "stress_absolute_return_positive": True,
                    "stress_cagr_mdd_min": 2.5,
                    "mechanism_control_margin_min": 0.25,
                },
            },
            "test": {
                "window": ["2023-01-01", "2024-01-01"],
                "opened_only_if_train_passes": True,
                "cannot_reselect_or_repair": True,
                "subperiods": ["2023_H1", "2023_H2"],
                "gates": {
                    "absolute_return_positive": True,
                    "cagr_mdd_min": 3.0,
                    "strict_mdd_max_pct": 15.0,
                    "trades_min": 60,
                    "weekly_cluster_signflip_p_max": 0.10,
                    "each_subperiod_absolute_return_positive": True,
                    "stress_absolute_return_positive": True,
                    "stress_cagr_mdd_min": 2.5,
                },
            },
            "statistical_test": {
                "name": "two-sided weekly-cluster sign flip",
                "draws": 20_000,
                "seed": 20_260_719,
            },
        },
        "rejection_contract": {
            "support_failure": "reject without opening BTCUSDT execution outcomes",
            "train_failure": "reject and keep 2023 BTCUSDT outcomes sealed",
            "test_failure": "reject without direction threshold feature timing or hold repair",
            "2024_plus": (
                "remain sealed until unchanged train and test pass; a later source "
                "extension must be separately checksum-frozen"
            ),
        },
        "rllm_boundary": {
            "standalone_alpha_is_formulaic": True,
            "llm_not_allowed_to_create_or_repair_signals": True,
            "future_role": (
                "after deterministic passage, an LLM may explain the multi-bar "
                "state or abstain under a separately frozen policy"
            ),
        },
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate_manifest(manifest: dict[str, Any], *, verify_sources: bool = True) -> None:
    if manifest.get("outcomes_opened") is not False:
        raise ValueError("CMSR-36 preregistration opened outcomes")
    if manifest.get("policy") != policy_payload():
        raise ValueError("CMSR-36 policy changed")
    core = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    if manifest.get("manifest_hash") != canonical_hash(core):
        raise ValueError("CMSR-36 manifest hash mismatch")
    forbidden = manifest["causal_feature_contract"]["forbidden_feature_columns"]
    if "BTCUSDT_execution_OHLC" not in forbidden:
        raise ValueError("CMSR-36 execution outcome entered the feature contract")
    if verify_sources:
        source = manifest["source_contract"]
        for path_key, hash_key in (
            ("source", "source_sha256"),
            ("source_manifest", "source_manifest_sha256"),
            ("market", "market_sha256"),
            ("market_manifest", "market_manifest_sha256"),
            ("funding", "funding_sha256"),
            ("funding_manifest", "funding_manifest_sha256"),
        ):
            if _sha256(source[path_key]) != source[hash_key]:
                raise ValueError(f"CMSR-36 source hash mismatch: {path_key}")


def write_manifest(path: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    report = build_manifest()
    validate_manifest(report)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = write_manifest(args.output)
    print(json.dumps({"output": args.output, "manifest_hash": report["manifest_hash"]}))


if __name__ == "__main__":
    main()
