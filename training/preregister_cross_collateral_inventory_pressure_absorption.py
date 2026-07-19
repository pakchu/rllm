"""Freeze CIPA-48 before any strategy outcome is opened.

CIPA-48 tests a source-only absorption state in which relative open interest
migrates toward one collateral venue while relative aggressive flow points the
other way.  The policy follows the inventory-rotation side, equivalently fading
the opposing taker-flow gap.  This module only writes a deterministic protocol
manifest; it never parses execution prices, funding, returns, or PnL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = (
    "results/cross_collateral_inventory_pressure_absorption_"
    "preregistration_2026-07-19.json"
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
    policy_id: str = "CIPA-48"
    usd_m_symbol: str = "BTCUSDT"
    coin_m_symbol: str = "BTCUSD_PERP"
    anchor_minute: int = 55
    oi_change_bars: int = 72
    taker_median_bars: int = 12
    prior_rank_hourly_anchors: int = 168
    oi_rotation_rank_min: float = 0.80
    taker_gap_rank_min: float = 0.60
    execution_delay_bars: int = 2
    hold_bars: int = 48
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
        "protocol_version": "cross_collateral_inventory_pressure_absorption_v1",
        "as_of_date": "2026-07-19",
        "outcomes_opened": False,
        "policy": policy_payload(),
        "research_history_boundary": {
            "market_history_seen_by_unrelated_repo_research": True,
            "source_only_density_preflight_seen": True,
            "exact_cipa_48_post_entry_outcomes_opened": False,
            "density_preflight": {
                "description": (
                    "Only positioning-source values, causal ranks, timestamps, "
                    "side counts, and fixed-hold schedule incidence were inspected."
                ),
                "q_0_80_raw_onsets": {
                    "train": 105,
                    "2021_partial": 47,
                    "2022": 58,
                    "2023": 72,
                    "2023_H1": 50,
                    "2023_H2": 22,
                },
                "q_0_80_hold_48_nonoverlap": {
                    "train": 100,
                    "2021_partial": 45,
                    "2022": 55,
                    "2023": 65,
                    "2023_H1": 44,
                    "2023_H2": 21,
                },
                "higher_oi_rank_raw_train_2023": {
                    "0.85": [78, 58],
                    "0.90": [47, 34],
                    "0.925": [37, 28],
                    "0.95": [27, 18],
                },
                "selection_consequence": (
                    "0.80 is frozen as the only policy because stricter ranks "
                    "lack the preregistered power floor; returns did not select it."
                ),
            },
            "forbidden_before_evaluator_freeze": [
                "entry_or_later_OHLC",
                "entry_to_exit_return",
                "funding_cash_flow",
                "strategy_PnL",
                "absolute_return",
                "CAGR",
                "strict_MDD",
                "win_rate",
            ],
        },
        "novelty_boundary": {
            "mechanism": (
                "relative inventory rotates toward one collateral venue while the "
                "relative taker-flow gap points against that rotation; the inventory "
                "side is treated as passive absorption rather than taker continuation"
            ),
            "not_a_ccpr_repair": (
                "CCPR required sign(R)=sign(T) and faded that concordant crowding. "
                "CIPA requires the disjoint sign(R)=-sign(T) quadrant and tests "
                "inventory-versus-aggression disagreement. It does not reverse the "
                "failed CCPR clock, threshold, direction, or hold."
            ),
            "excluded_inputs": [
                "price_or_return",
                "funding_premium_or_basis",
                "REX_or_regime",
                "kimchi_FX_or_DXY",
                "spot_volume_or_order_book",
                "liquidation_or_options",
                "existing_alpha_state_or_PnL",
            ],
            "forbidden_repairs_after_outcomes": [
                "reverse_direction",
                "change_rank_thresholds",
                "change_feature_windows_or_anchor",
                "change_execution_delay_or_hold",
                "add_price_regime_funding_premium_or_portfolio_gate",
                "replace_opposition_with_concordance",
            ],
        },
        "source_contract": {
            "source_commit": "8d347432cd36d59458ad9a26c7c8aef1ec94b8ee",
            "positioning": POSITIONING_PATH,
            "positioning_sha256": SOURCE_HASHES["positioning_sha256"],
            "positioning_manifest": POSITIONING_MANIFEST_PATH,
            "positioning_manifest_sha256": SOURCE_HASHES[
                "positioning_manifest_sha256"
            ],
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
                "no fill or stale carry; current 73-row OI path and all 168 prior "
                "hourly anchors must be source-complete"
            ),
        },
        "causal_feature_contract": {
            "anchor": "UTC hourly source rows with minute=55",
            "availability": (
                "the completed metrics state at t waits one empty five-minute "
                "availability bucket; entry is t+10m"
            ),
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
                "T[t]=median(t-11..t, log(UM_taker_ratio)-log(CM_taker_ratio))"
            ),
            "strict_prior_ranks": (
                "A[t] and G[t] are empirical mid-ranks of abs(R[t]) and abs(T[t]) "
                "against exactly 168 complete hourly anchors immediately before t"
            ),
            "setup": (
                "A[t]>=0.80 AND G[t]>=0.60 AND sign(R[t])=-sign(T[t])!=0; "
                "only a false-to-true transition creates an episode"
            ),
            "action": "side[t]=sign(R[t])=-sign(T[t])",
            "price_signal_columns": [],
        },
        "support_gate": {
            "train_window": ["2021-07-08", "2023-01-01"],
            "test_support_window": ["2023-01-01", "2024-01-01"],
            "minimum_nonoverlap_train": 90,
            "minimum_2021_partial": 40,
            "minimum_2022": 50,
            "minimum_2023": 60,
            "minimum_each_2023_half": 20,
            "minimum_each_side_share": 0.25,
            "maximum_single_month_share": 0.20,
            "ccpr_exact_entry_jaccard_max": 0.02,
            "ccpr_near_one_hour_fraction_max": 0.25,
        },
        "execution_contract": {
            "instrument": "Binance USD-M BTCUSDT perpetual",
            "fill": "next 5m open at source timestamp t+10m",
            "hold": "48 five-minute bars / four hours",
            "nonoverlap": "chronological global reservation by [entry, exit)",
            "sizing": "fixed 0.5x notional",
            "costs": (
                "6bp/notional/side base and 10bp/notional/side stress with exact "
                "funding under the frozen strict simulator"
            ),
            "strict_mdd": (
                "global/pre-entry HWM, entry cost, every held favorable-then-adverse "
                "OHLC path, conservative funding boundaries, virtual adverse-mark "
                "exit cost, and actual exit cost"
            ),
            "cagr": "full declared calendar including idle periods",
        },
        "falsification_controls": {
            "oi_only": "A[t]>=0.80 onset; side=sign(R[t])",
            "taker_only": "G[t]>=0.60 onset; side=-sign(T[t])",
            "direction_flip": "same primary entries with side multiplied by -1",
            "entry_shift_plus_1h": "same primary episode shifted twelve 5m bars",
            "deterministic_random_side": (
                "same primary entries, side from SHA256(policy_id|signal_time)"
            ),
        },
        "selection_protocol": {
            "candidate_count": 1,
            "train": {
                "window": ["2021-07-08", "2023-01-01"],
                "subperiods": ["2021_partial", "2022_H1", "2022_H2"],
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
            "support_failure": "reject without opening execution outcomes",
            "train_failure": "reject and keep 2023 outcomes sealed",
            "test_failure": "reject without threshold direction hold or gate repair",
            "future_extension": (
                "2024+ source and outcomes may be acquired only after unchanged "
                "train and test passage"
            ),
        },
        "rllm_boundary": {
            "standalone_alpha_is_formulaic": True,
            "llm_not_allowed_to_create_or_repair_signals": True,
            "future_role": (
                "after standalone validation, an LLM may explain or abstain from "
                "the frozen state under a separately preregistered policy"
            ),
        },
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate_manifest(manifest: dict[str, Any], *, verify_sources: bool = True) -> None:
    if manifest.get("outcomes_opened") is not False:
        raise ValueError("CIPA-48 preregistration opened outcomes")
    if manifest.get("policy") != policy_payload():
        raise ValueError("CIPA-48 policy changed")
    core = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    if manifest.get("manifest_hash") != canonical_hash(core):
        raise ValueError("CIPA-48 manifest hash mismatch")
    if manifest["causal_feature_contract"].get("price_signal_columns") != []:
        raise ValueError("CIPA-48 signal is not price blind")
    if verify_sources:
        source = manifest["source_contract"]
        for path_key, hash_key in (
            ("positioning", "positioning_sha256"),
            ("positioning_manifest", "positioning_manifest_sha256"),
            ("market", "market_sha256"),
            ("market_manifest", "market_manifest_sha256"),
            ("funding", "funding_sha256"),
            ("funding_manifest", "funding_manifest_sha256"),
        ):
            if _sha256(source[path_key]) != source[hash_key]:
                raise ValueError(f"CIPA-48 source hash mismatch: {path_key}")


def write_manifest(path: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    manifest = build_manifest()
    validate_manifest(manifest)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = write_manifest(args.output)
    print(json.dumps({"output": args.output, "manifest_hash": report["manifest_hash"]}))


if __name__ == "__main__":
    main()
