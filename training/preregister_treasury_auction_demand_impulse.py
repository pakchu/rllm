"""Preregister TADI-1 before opening any BTC outcome."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from training import treasury_auction_demand_impulse_clock as clock


DEFAULT_OUTPUT = (
    "results/treasury_auction_demand_impulse_preregistration_2026-07-17.json"
)
SOURCE_PATH = clock.DEFAULT_SOURCE
SOURCE_MANIFEST = "data/us_treasury_auction_demand_2016_2023/build_manifest.json"
MARKET_PATH = (
    "data/binance_um_kline_reference_btc_2020_2023/"
    "BTCUSDT_5m_2020-01-01_2023-12-31.csv.gz"
)
MARKET_MANIFEST = (
    "data/binance_um_kline_reference_btc_2020_2023/build_manifest.json"
)
FUNDING_PATH = "data/binance_um_btcusdt_funding_marks_2020_2023.csv.gz"
FUNDING_MANIFEST = (
    "results/binance_um_btcusdt_funding_marks_2020_2023_manifest_2026-07-17.json"
)


@dataclass(frozen=True)
class Policy:
    policy_id: str = "TADI-1"
    prior_same_tenor_changes: int = 12
    concordant_tail_threshold: float = 0.75
    result_available_hour_utc: int = 22
    execution_delay_minutes: int = 5
    hold_hours: int = 24
    leverage: float = 0.5
    base_cost_notional_per_side: float = 0.0006
    stress_cost_notional_per_side: float = 0.0010


SOURCE_HASHES = {
    "auction_panel_sha256": "34a19163630c015a4f9d2671c95ca7cf7cc8a8ada024b3ef985405704fe0e4c1",
    "auction_manifest_sha256": "6da6a3848e89c3418efcbf0d836fda34b537a2da87a8777b74670f3912ad94f2",
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
    "family_capped_portfolio": {
        "path": "results/portfolio_all_discovered_dedup_familycap2_trainmdd40_oosmdd20_2026-07-12.json",
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
    side_counts = {
        side: sum(event.side == side for event in selected) for side in ("LONG", "SHORT")
    }
    month_counts: dict[str, int] = {}
    term_counts: dict[str, int] = {}
    for event in selected:
        month = event.entry_time[:7]
        month_counts[month] = month_counts.get(month, 0) + 1
        term = event.original_security_term
        term_counts[term] = term_counts.get(term, 0) + 1
    return {
        "events": len(selected),
        "side_counts": side_counts,
        "term_counts": dict(sorted(term_counts.items())),
        "months": len(month_counts),
        "max_single_month_count": max(month_counts.values(), default=0),
        "max_single_month_share": (
            max(month_counts.values(), default=0) / len(selected) if selected else 0.0
        ),
    }


def source_only_disclosure() -> dict[str, Any]:
    rows = clock.read_source(SOURCE_PATH)
    primary = clock.build_events(rows)
    bid_only = clock.build_events(rows, mode="bid_to_cover_only")
    indirect_only = clock.build_events(rows, mode="indirect_only")
    windows = {
        "2021": ("2021-01-01T00:00:00+00:00", "2022-01-01T00:00:00+00:00"),
        "2022": ("2022-01-01T00:00:00+00:00", "2023-01-01T00:00:00+00:00"),
        "train": ("2021-01-01T00:00:00+00:00", "2023-01-01T00:00:00+00:00"),
        "2023_h1": ("2023-01-01T00:00:00+00:00", "2023-07-01T00:00:00+00:00"),
        "2023_h2": ("2023-07-01T00:00:00+00:00", "2024-01-01T00:00:00+00:00"),
        "2023": ("2023-01-01T00:00:00+00:00", "2024-01-01T00:00:00+00:00"),
    }
    return {
        "definition": "source rows, strict-prior changes/ranks, timestamps, sides and counts only",
        "outcomes_joined": False,
        "primary": {name: _counts(primary, *window) for name, window in windows.items()},
        "bid_to_cover_only": {
            name: _counts(bid_only, *window) for name, window in windows.items()
        },
        "indirect_only": {
            name: _counts(indirect_only, *window) for name, window in windows.items()
        },
    }


def build_manifest() -> dict[str, Any]:
    disclosure = source_only_disclosure()
    core: dict[str, Any] = {
        "protocol_version": "treasury_auction_demand_impulse_v1",
        "as_of_date": "2026-07-17",
        "outcomes_opened": False,
        "policy": policy_payload(),
        "research_history_boundary": {
            "auction_source_values_seen": True,
            "source_only_density_preflight_seen": True,
            "exact_post_entry_btc_outcomes_opened": False,
            "support_only_access_allowed": (
                "source values, causal source-only features, event timestamps, "
                "side/term/month counts, and source-clock overlap"
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
            "density_preflight_disclosure": disclosure,
        },
        "novelty_boundary": {
            "distinct_axis": (
                "official U.S. Treasury auction demand change and bidder composition; "
                "no crypto-internal state enters the signal"
            ),
            "excluded_inputs": [
                "BTC_OHLC_or_return",
                "REX_or_rolling_extrema",
                "open_interest_or_taker_flow",
                "funding_premium_basis",
                "kimchi_BTCKRW_USDKRW_DXY_or_EMFX",
                "spot_volume_or_cross_venue_price",
                "options_onchain_attention_or_existing_alpha_state",
            ],
            "forbidden_repairs_after_outcomes": [
                "reverse primary direction",
                "change the 12-change strict-prior window",
                "change the quartile threshold",
                "add yield price FX crypto or regime gates",
                "change the conservative 22:00 UTC clock",
                "change the 24h hold or 0.5x size",
                "include reopenings TIPS FRNs or bills",
                "bridge across source_complete=false rows",
            ],
        },
        "source_contract": {
            "source_commit": "e13ed6e",
            "auction_panel": SOURCE_PATH,
            "auction_panel_sha256": SOURCE_HASHES["auction_panel_sha256"],
            "auction_manifest": SOURCE_MANIFEST,
            "auction_manifest_sha256": SOURCE_HASHES["auction_manifest_sha256"],
            "source_audit": "docs/us-treasury-auction-demand-source-audit-2026-07-17.md",
            "market": MARKET_PATH,
            "market_sha256": SOURCE_HASHES["market_sha256"],
            "market_manifest": MARKET_MANIFEST,
            "market_manifest_sha256": SOURCE_HASHES["market_manifest_sha256"],
            "funding": FUNDING_PATH,
            "funding_sha256": SOURCE_HASHES["funding_sha256"],
            "funding_manifest": FUNDING_MANIFEST,
            "funding_manifest_sha256": SOURCE_HASHES["funding_manifest_sha256"],
            "source_incomplete_policy": (
                "blank demand values, reset previous same-tenor observation, and "
                "never bridge a change across the quarantined auction"
            ),
        },
        "causal_feature_contract": {
            "universe": "original nominal fixed-rate 2y/3y/5y/7y/10y/20y/30y auctions",
            "available_at": "22:00 UTC on auction date",
            "bid_to_cover_change": "B[t]=BTCover[t]-BTCover[previous complete same-tenor auction]",
            "indirect_share": "I[t]=indirectAccepted/(primary+direct+indirect accepted)",
            "indirect_share_change": "J[t]=I[t]-I[previous complete same-tenor auction]",
            "strict_prior_ranks": (
                "mid-rank B[t] and J[t] against exactly 12 prior valid same-tenor "
                "changes; current t is excluded and added only after emission"
            ),
            "setup": (
                "LONG iff both ranks >=0.75; SHORT iff both ranks <=0.25; otherwise none"
            ),
            "economic_direction": (
                "concordant positive auction demand impulse implies easier global "
                "duration absorption and LONG BTC; concordant negative impulse implies SHORT"
            ),
            "price_signal_columns": [],
        },
        "execution_contract": {
            "decision_time": "22:00 UTC auction date",
            "entry": "first 5m open after decision (+5m)",
            "exit": "scheduled 5m open exactly 24h after entry",
            "nonoverlap": (
                "global chronological reservation; same-timestamp conflicts use shortest-tenor-first priority"
            ),
            "position_size": "0.5x fixed account gross",
            "base_cost": "6 bp/notional/side",
            "stress_cost": "10 bp/notional/side",
            "funding": "exact realized BTCUSDT funding settlements on [entry, exit)",
            "strict_mdd": (
                "global/pre-entry high-water; entry cost; favorable-before-adverse "
                "held OHLC; exact funding; hypothetical liquidation and exit cost"
            ),
            "cagr_clock": "full wall-clock split including warm-up and idle cash",
        },
        "support_gates": {
            "minimum_train_events": 25,
            "minimum_2021_events": 12,
            "minimum_2022_events": 12,
            "minimum_2023_events": 20,
            "minimum_each_2023_half": 10,
            "minimum_train_each_side": 8,
            "minimum_2023_each_side": 8,
            "maximum_single_month_share": 0.25,
            "month_share_scope": "train and full 2023 only; halves are density diagnostics",
            "expected_counts": disclosure["primary"],
            "failure_action": "retire TADI-1 before loading any BTC outcome",
        },
        "falsification_controls": {
            "bid_to_cover_only": "same strict-prior clock using only B rank tails",
            "indirect_only": "same strict-prior clock using only J rank tails",
            "direction_flip": "same primary entries with LONG/SHORT swapped",
            "one_auction_delay": "same primary side at next complete same-tenor result clock",
            "deterministic_random_side": (
                "same primary entries; SHA256('TADI-1-random-side-20260717|' + entry_time), first byte <128 => LONG"
            ),
        },
        "selection_protocol": {
            "stage1_train": ["2021-01-01", "2023-01-01"],
            "stage1_subperiods": {
                "2021": ["2021-01-01", "2022-01-01"],
                "2022": ["2022-01-01", "2023-01-01"],
            },
            "stage2_selection": ["2023-01-01", "2024-01-01"],
            "stage2_halves": {
                "h1": ["2023-01-01", "2023-07-01"],
                "h2": ["2023-07-01", "2024-01-01"],
            },
            "candidate_count": 1,
            "stage2_requires_unchanged_stage1_pass": True,
            "gates": {
                "absolute_return_positive": True,
                "cagr_to_strict_mdd_min": 3.0,
                "strict_mdd_pct_max": 15.0,
                "weekly_cluster_signflip_p_max": 0.10,
                "minimum_trades": 25,
                "mean_gross_underlying_bp_min": 35.0,
                "stress_cost_absolute_return_positive": True,
                "each_subperiod_absolute_return_positive": True,
                "each_subperiod_minimum_trades": 12,
                "mechanism_margin_ratio_min": 0.25,
            },
            "statistical_test": (
                "exact two-sided weekly-cluster Rademacher sign-flip; enumerate "
                "when feasible, otherwise deterministic 20000 draws"
            ),
            "no_parameter_repair": True,
        },
        "orthogonality_after_standalone_pass": {
            "required": True,
            "not_allowed_before_pass": True,
            "tests": [
                "entry-clock Jaccard",
                "daily return correlation",
                "trade-source overlap",
                "marginal portfolio CAGR/MDD under frozen allocation rules",
            ],
            "comparator_universe": COMPARATOR_ARTIFACTS,
        },
    }
    core["manifest_hash"] = canonical_hash(core)
    return core


def validate_manifest(manifest: dict[str, Any], *, verify_sources: bool = True) -> None:
    if manifest.get("outcomes_opened") is not False:
        raise ValueError("TADI-1 outcomes opened before freeze")
    if manifest.get("policy") != policy_payload():
        raise ValueError("TADI-1 policy differs from frozen singleton")
    claimed = manifest.get("manifest_hash")
    actual = canonical_hash({k: v for k, v in manifest.items() if k != "manifest_hash"})
    if claimed != actual:
        raise ValueError("TADI-1 manifest hash mismatch")
    if not verify_sources:
        return
    source = manifest["source_contract"]
    for path_key, hash_key in (
        ("auction_panel", "auction_panel_sha256"),
        ("auction_manifest", "auction_manifest_sha256"),
        ("market", "market_sha256"),
        ("market_manifest", "market_manifest_sha256"),
        ("funding", "funding_sha256"),
        ("funding_manifest", "funding_manifest_sha256"),
    ):
        if _sha256(source[path_key]) != source[hash_key]:
            raise ValueError(f"TADI-1 source hash drift: {path_key}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = build_manifest()
    validate_manifest(manifest)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
