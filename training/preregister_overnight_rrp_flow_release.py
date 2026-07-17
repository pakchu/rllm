"""Preregister ORFR-1 before opening any BTC outcome."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from training import overnight_rrp_flow_release_clock as clock


DEFAULT_OUTPUT = "results/overnight_rrp_flow_release_preregistration_2026-07-17.json"
SOURCE_PATH = clock.DEFAULT_SOURCE
SOURCE_MANIFEST = "data/new_york_fed_overnight_rrp_2018_2023/build_manifest.json"
MARKET_PATH = (
    "data/binance_um_kline_reference_btc_2020_2023/"
    "BTCUSDT_5m_2020-01-01_2023-12-31.csv.gz"
)
MARKET_MANIFEST = "data/binance_um_kline_reference_btc_2020_2023/build_manifest.json"
FUNDING_PATH = "data/binance_um_btcusdt_funding_marks_2020_2023.csv.gz"
FUNDING_MANIFEST = (
    "results/binance_um_btcusdt_funding_marks_2020_2023_manifest_2026-07-17.json"
)


@dataclass(frozen=True)
class Policy:
    policy_id: str = "ORFR-1"
    baseline_operations: int = 5
    rank_operations: int = 104
    lower_tail_rank: float = 0.125
    source_publication_buffer_minutes: int = 15
    execution_delay_minutes: int = 5
    exit_clock: str = "next normal ON RRP result availability plus 5 minutes"
    leverage: float = 0.5
    base_cost_notional_per_side: float = 0.0006
    stress_cost_notional_per_side: float = 0.0010


SOURCE_HASHES = {
    "rrp_panel_sha256": "49f67ed44b7eb81fd35c17a8209cf14d6a8019d7e9f77fce8c343d1a7fb66b27",
    "rrp_manifest_sha256": "4f87e2219da71c94832c8708086ba01387efc145e3488b62cd3b3d07c62d8fee",
    "market_sha256": "e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d",
    "market_manifest_sha256": "c04fbbd299cc748a6745c0ef030787da4d560833c744c81c98dd8840efc7913e",
    "funding_sha256": "3284bbb6bb67946acb673c6b67459543e217f752589e1d47b6c7c3b659f733e6",
    "funding_manifest_sha256": "a0b2d27e1aa8cf2d9ab8cb659b598ee0a6d7bd25401c9e10ae92d1a74415845b",
}
COMPARATOR_ARTIFACTS = {
    "deduplicated_universe": {
        "path": "results/all_discovered_alpha_universe_audit_2026-07-12.json",
        "sha256": "5bd45f5949cfae7308dfadd3966112e06c7c09dbb92365796237c3b94be85a3e",
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


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def policy_payload() -> dict[str, Any]:
    return asdict(Policy())


def _counts(events: list[clock.Event], start: str, end: str) -> dict[str, Any]:
    selected = [event for event in events if start <= event.entry_time < end]
    month_counts: dict[str, int] = {}
    for event in selected:
        month = event.entry_time[:7]
        month_counts[month] = month_counts.get(month, 0) + 1
    return {
        "events": len(selected),
        "side_counts": {
            side: sum(event.side == side for event in selected)
            for side in ("LONG", "SHORT")
        },
        "months": len(month_counts),
        "max_single_month_count": max(month_counts.values(), default=0),
        "max_single_month_share": (
            max(month_counts.values(), default=0) / len(selected) if selected else 0.0
        ),
    }


def source_only_disclosure() -> dict[str, Any]:
    rows = clock.read_source(SOURCE_PATH)
    primary = clock.build_events(rows)
    delta = clock.build_events(rows, mode="one_day_delta")
    windows = {
        "2021": ("2021-01-01T00:00:00+00:00", "2022-01-01T00:00:00+00:00"),
        "2022": ("2022-01-01T00:00:00+00:00", "2023-01-01T00:00:00+00:00"),
        "train": ("2021-01-01T00:00:00+00:00", "2023-01-01T00:00:00+00:00"),
        "2023_h1": ("2023-01-01T00:00:00+00:00", "2023-07-01T00:00:00+00:00"),
        "2023_h2": ("2023-07-01T00:00:00+00:00", "2024-01-01T00:00:00+00:00"),
        "2023": ("2023-01-01T00:00:00+00:00", "2024-01-01T00:00:00+00:00"),
    }
    return {
        "definition": "source rows, strict-prior transforms, timestamps, sides and counts only",
        "outcomes_joined": False,
        "primary": {name: _counts(primary, *window) for name, window in windows.items()},
        "one_day_delta": {
            name: _counts(delta, *window) for name, window in windows.items()
        },
    }


def build_manifest() -> dict[str, Any]:
    disclosure = source_only_disclosure()
    core: dict[str, Any] = {
        "protocol_version": "overnight_rrp_flow_release_v1",
        "as_of_date": "2026-07-17",
        "outcomes_opened": False,
        "policy": policy_payload(),
        "research_history_boundary": {
            "rrp_source_values_seen": True,
            "source_only_density_preflight_seen": True,
            "exact_post_entry_btc_outcomes_opened": False,
            "density_thresholds_inspected_without_outcomes": [
                0.05,
                0.075,
                0.10,
                0.125,
                0.15,
                0.20,
            ],
            "threshold_choice_rule": (
                "freeze the sparsest inspected symmetric tail with at least 100 "
                "Stage1 events, 45 events in each Stage1 year, 35 events per side, "
                "60 sealed-2023 source events, 15 sealed-2023 events per side, "
                "20 events per half, and <=20% full-window month concentration"
            ),
            "disclosure": disclosure,
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
                "official daily New York Fed ON RRP cash absorption/release; no "
                "crypto-internal or existing-alpha state enters the signal"
            ),
            "excluded_inputs": [
                "BTC_OHLC_or_return",
                "REX_or_rolling_extrema",
                "open_interest_or_taker_flow",
                "funding_premium_basis",
                "kimchi_BTCKRW_USDKRW_DXY_or_EMFX",
                "spot_volume_cross_venue_options_onchain_or_existing_alpha_state",
            ],
            "forbidden_repairs_after_outcomes": [
                "reverse direction",
                "change 5-operation baseline or 104-operation rank window",
                "change 1/8 tails",
                "add calendar price FX crypto or regime gates",
                "change publication buffer execution delay exit clock or size",
                "use quarantined values or bridge a baseline across them",
            ],
        },
        "source_contract": {
            "source_commit": "f90cc3a",
            "rrp_panel": SOURCE_PATH,
            "rrp_panel_sha256": SOURCE_HASHES["rrp_panel_sha256"],
            "rrp_manifest": SOURCE_MANIFEST,
            "rrp_manifest_sha256": SOURCE_HASHES["rrp_manifest_sha256"],
            "source_audit": "docs/new-york-fed-overnight-rrp-source-audit-2026-07-17.md",
            "market": MARKET_PATH,
            "market_sha256": SOURCE_HASHES["market_sha256"],
            "market_manifest": MARKET_MANIFEST,
            "market_manifest_sha256": SOURCE_HASHES["market_manifest_sha256"],
            "funding": FUNDING_PATH,
            "funding_sha256": SOURCE_HASHES["funding_sha256"],
            "funding_manifest": FUNDING_MANIFEST,
            "funding_manifest_sha256": SOURCE_HASHES["funding_manifest_sha256"],
            "incomplete_policy": (
                "blank value, reset local 5-operation baseline and previous-day "
                "state, emit no signal, and never bridge across the row"
            ),
        },
        "causal_feature_contract": {
            "amount": "A[t]=log1p(total accepted USD / 1e9)",
            "local_baseline": "M[t]=median(A[t-5:t]); current t excluded",
            "innovation": "X[t]=A[t]-M[t]",
            "rank": (
                "strict-prior midrank of X[t] against exactly the previous 104 "
                "valid innovations; current t is appended only after ranking"
            ),
            "setup": "LONG iff rank<=0.125; SHORT iff rank>=0.875; otherwise abstain",
            "economic_direction": (
                "an unusually large accepted ON RRP amount shifts more cash from "
                "reserves into the Fed RRP liability and maps to SHORT BTC; an "
                "unusually small amount releases relative liquidity and maps LONG"
            ),
            "price_signal_columns": [],
        },
        "execution_contract": {
            "decision": "frozen source result_available_at_utc (close+15m ET)",
            "entry": "decision + one complete 5m bucket",
            "exit": "next normal ON RRP result availability + one complete 5m bucket",
            "nonoverlap": "event exits no later than the next candidate entry",
            "position_size": "0.5x fixed account gross",
            "base_cost": "6 bp/notional/side",
            "stress_cost": "10 bp/notional/side",
            "funding": "exact realized BTCUSDT funding on [entry, exit)",
            "cagr": "full wall-clock split including warm-up and idle cash",
            "strict_mdd": (
                "global/pre-entry high-water, costs, funding, favorable-before-adverse "
                "held OHLC, hypothetical liquidation and realized exit"
            ),
        },
        "support_gates": {
            "minimum_train_events": 100,
            "minimum_each_stage1_year": 45,
            "minimum_train_each_side": 35,
            "minimum_2023_events": 60,
            "minimum_each_2023_half": 20,
            "minimum_2023_each_side": 15,
            "maximum_single_month_share": 0.20,
            "expected_counts": disclosure["primary"],
            "failure_action": "retire ORFR-1 before loading any BTC outcome",
        },
        "falsification_controls": {
            "one_day_delta_tail": (
                "replace five-operation residual with one-operation log amount "
                "change; retain the same 104-observation rank and 1/8 tails"
            ),
            "direction_flip": "same primary entries with LONG/SHORT swapped",
            "one_release_delay": "same primary side one complete ON RRP operation later",
            "deterministic_random_side": (
                "same primary entries; SHA256('ORFR-1-random-side-20260717|' + "
                "entry_time), first byte <128 => LONG"
            ),
        },
        "selection_protocol": {
            "stage1": ["2021-01-01", "2023-01-01"],
            "stage1_subperiods": {
                "2021": ["2021-01-01", "2022-01-01"],
                "2022": ["2022-01-01", "2023-01-01"],
            },
            "stage2": ["2023-01-01", "2024-01-01"],
            "stage2_subperiods": {
                "2023_h1": ["2023-01-01", "2023-07-01"],
                "2023_h2": ["2023-07-01", "2024-01-01"],
            },
            "candidate_count": 1,
            "stage2_requires_exact_unchanged_stage1_pass": True,
            "gates": {
                "absolute_return_positive": True,
                "cagr_to_strict_mdd_min": 3.0,
                "strict_mdd_pct_max": 15.0,
                "weekly_cluster_signflip_p_max": 0.10,
                "minimum_trades": 100,
                "mean_gross_underlying_bp_min": 35.0,
                "stress_cost_absolute_return_positive": True,
                "each_subperiod_absolute_return_positive": True,
                "each_subperiod_minimum_trades": 45,
                "minimum_each_side_trades": 35,
                "mechanism_margin_ratio_min": 0.25,
            },
            "no_parameter_repair": True,
        },
        "orthogonality_after_standalone_pass": {
            "required": True,
            "not_allowed_before_pass": True,
            "entry_jaccard_max": 0.02,
            "entry_near_6h_fraction_max": 0.25,
            "position_jaccard_max": 0.15,
            "absolute_daily_pnl_pearson_max": 0.30,
            "comparator_universe": COMPARATOR_ARTIFACTS,
        },
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate_manifest(manifest: dict[str, Any], *, verify_sources: bool = True) -> None:
    if manifest.get("outcomes_opened") is not False:
        raise ValueError("ORFR-1 outcomes opened before evaluator freeze")
    if manifest.get("policy") != policy_payload():
        raise ValueError("ORFR-1 policy differs from frozen singleton")
    claimed = manifest.get("manifest_hash")
    actual = canonical_hash({key: value for key, value in manifest.items() if key != "manifest_hash"})
    if claimed != actual:
        raise ValueError("ORFR-1 manifest hash mismatch")
    if not verify_sources:
        return
    source = manifest["source_contract"]
    for path_key, hash_key in (
        ("rrp_panel", "rrp_panel_sha256"),
        ("rrp_manifest", "rrp_manifest_sha256"),
        ("market", "market_sha256"),
        ("market_manifest", "market_manifest_sha256"),
        ("funding", "funding_sha256"),
        ("funding_manifest", "funding_manifest_sha256"),
    ):
        if _sha256(source[path_key]) != source[hash_key]:
            raise ValueError(f"ORFR-1 source hash drift: {path_key}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = build_manifest()
    validate_manifest(manifest)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
