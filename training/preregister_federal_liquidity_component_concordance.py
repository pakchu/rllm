"""Preregister the FLCC family without opening any BTC outcome.

Federal Liquidity Component Concordance (FLCC) follows a sufficiently unusual
H.4.1 net-liquidity impulse only when at least two of three weak component
contributions agree with its direction.  This module writes a deterministic
protocol manifest and source-only support summary.  It never parses market
prices, returns, funding cash flow, CAGR, or drawdown.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training import federal_liquidity_component_concordance_clock as exact_clock


DEFAULT_OUTPUT = (
    "results/federal_liquidity_component_concordance_"
    "preregistration_2026-07-17.json"
)
SOURCE_PATH = exact_clock.SOURCE_PATH
SOURCE_BUILD_MANIFEST = (
    "data/federal_reserve_h41_net_liquidity_2018_2023/build_manifest.json"
)
SOURCE_RAW_MANIFEST = (
    "data/federal_reserve_h41_net_liquidity_2018_2023/source_manifest.json"
)
SOURCE_AUDIT = (
    "results/federal_reserve_h41_net_liquidity_source_freeze_2026-07-17.json"
)
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
CLOCK_LEDGER = exact_clock.DEFAULT_OUTPUT

STATIC_SHA256 = {
    SOURCE_PATH: "224883dad01b9d7f17d52eb87f3d7ef9890c8dd055a6c36577a534d2afe69621",
    SOURCE_BUILD_MANIFEST: "1ec212a85de0e49c5a0c2d35b8b22be86eb7d62989f7a0098be1bb1274b2a99b",
    SOURCE_RAW_MANIFEST: "61dca0ae9e29c2c96307a3442037e43aedae15e21d3aedc9ee209c7ebbcac271",
    SOURCE_AUDIT: "765fbc6d37799c9c642fa1b956d717c3a6f917aa384f0516c2f949e3dffdbc8c",
    CLOCK_LEDGER: "03fa41d6bc60bab89af856efe6aaf167f08602b3f688cf3bf1cdb3d84b62eaa3",
    MARKET_PATH: "e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d",
    MARKET_MANIFEST: "c04fbbd299cc748a6745c0ef030787da4d560833c744c81c98dd8840efc7913e",
    FUNDING_PATH: (
        "3284bbb6bb67946acb673c6b67459543e217f752589e1d47b6c7c3b659f733e6"
    ),
    FUNDING_MANIFEST: "a0b2d27e1aa8cf2d9ab8cb659b598ee0a6d7bd25401c9e10ae92d1a74415845b",
}

WINDOWS = {
    "train": ("2020-01-01", "2023-01-01"),
    "2020": ("2020-01-01", "2021-01-01"),
    "2021": ("2021-01-01", "2022-01-01"),
    "2022": ("2022-01-01", "2023-01-01"),
    "2023": ("2023-01-01", "2024-01-01"),
    "2023_h1": ("2023-01-01", "2023-07-01"),
    "2023_h2": ("2023-07-01", "2024-01-01"),
}

EXPECTED_PRIMARY_SUPPORT = {
    "FLCC-H4-Q60": {
        "train": (108, 59, 49),
        "2020": (34, 29, 5),
        "2021": (33, 19, 14),
        "2022": (40, 11, 29),
        "2023": (27, 13, 14),
        "2023_h1": (19, 8, 11),
        "2023_h2": (8, 5, 3),
    },
    "FLCC-H4-Q65": {
        "train": (99, 53, 46),
        "2020": (33, 28, 5),
        "2021": (31, 18, 13),
        "2022": (34, 7, 27),
        "2023": (23, 13, 10),
        "2023_h1": (16, 8, 8),
        "2023_h2": (7, 5, 2),
    },
    "FLCC-H8-Q60": {
        "train": (97, 50, 47),
        "2020": (32, 26, 6),
        "2021": (22, 17, 5),
        "2022": (41, 6, 35),
        "2023": (28, 13, 15),
        "2023_h1": (18, 9, 9),
        "2023_h2": (10, 4, 6),
    },
    "FLCC-H8-Q65": {
        "train": (94, 49, 45),
        "2020": (31, 25, 6),
        "2021": (21, 17, 4),
        "2022": (40, 6, 34),
        "2023": (22, 13, 9),
        "2023_h1": (15, 9, 6),
        "2023_h2": (7, 4, 3),
    },
}


@dataclass(frozen=True)
class ExecutionPolicy:
    execution_delay_bars: int = 1
    hold_bars: int = 1440
    leverage: float = 0.5
    base_cost_notional_per_side: float = 0.0006
    stress_cost_notional_per_side: float = 0.0010
    strict_mdd_denominator_floor: float = 1e-9
    weekly_cluster_draws: int = 100_000
    weekly_cluster_seed: int = 41_170_617


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def _window_summary(
    events: list[exact_clock.Event], start: str, end: str
) -> dict[str, Any]:
    lower = _utc(start)
    upper = _utc(end)
    inside = [
        event
        for event in events
        if exact_clock._parse_utc(event.entry_time) >= lower
        and exact_clock._parse_utc(event.exit_time) < upper
    ]
    month_counts: dict[str, int] = {}
    for event in inside:
        month = event.entry_time[:7]
        month_counts[month] = month_counts.get(month, 0) + 1
    maximum = max(month_counts.values(), default=0)
    return {
        "count": len(inside),
        "long": sum(event.side == 1 for event in inside),
        "short": sum(event.side == -1 for event in inside),
        "max_single_month_count": maximum,
        "max_single_month_share": maximum / len(inside) if inside else 0.0,
    }


def source_only_support() -> dict[str, Any]:
    events = exact_clock.read_event_ledger(CLOCK_LEDGER)
    output: dict[str, Any] = {}
    for spec in exact_clock.CANDIDATE_SPECS:
        primary = [
            event
            for event in events
            if event.candidate_id == spec.candidate_id
            and event.clock_name == "primary"
        ]
        summaries = {
            name: _window_summary(primary, start, end)
            for name, (start, end) in WINDOWS.items()
        }
        expected = EXPECTED_PRIMARY_SUPPORT[spec.candidate_id]
        for name, triplet in expected.items():
            actual = summaries[name]
            if (actual["count"], actual["long"], actual["short"]) != triplet:
                raise ValueError(
                    f"{spec.candidate_id} {name} source support changed: "
                    f"expected={triplet}, actual={actual}"
                )
        output[spec.candidate_id] = {
            "spec": asdict(spec),
            "full_source_primary_events": len(primary),
            "windows": summaries,
        }
    return output


def _support_gates(support: dict[str, Any]) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for candidate_id, payload in support.items():
        windows = payload["windows"]
        checks.update(
            {
                f"{candidate_id}_train_count": windows["train"]["count"] >= 90,
                f"{candidate_id}_train_each_side": min(
                    windows["train"]["long"], windows["train"]["short"]
                )
                >= 40,
                f"{candidate_id}_each_train_year": min(
                    windows[year]["count"] for year in ("2020", "2021", "2022")
                )
                >= 20,
                f"{candidate_id}_2023_count": windows["2023"]["count"] >= 20,
                f"{candidate_id}_2023_each_side": min(
                    windows["2023"]["long"], windows["2023"]["short"]
                )
                >= 7,
                f"{candidate_id}_2023_halves": (
                    windows["2023_h1"]["count"] >= 14
                    and windows["2023_h2"]["count"] >= 6
                ),
                f"{candidate_id}_train_month_concentration": (
                    windows["train"]["max_single_month_share"] <= 0.08
                ),
                f"{candidate_id}_2023_month_concentration": (
                    windows["2023"]["max_single_month_share"] <= 0.20
                ),
            }
        )
    return checks


def build_manifest() -> dict[str, Any]:
    support = source_only_support()
    support_checks = _support_gates(support)
    core: dict[str, Any] = {
        "protocol_version": "federal_liquidity_component_concordance_v1",
        "as_of_date": "2026-07-17",
        "outcomes_opened": False,
        "family_id": "FLCC-1",
        "candidate_class": "source-only-screened four-candidate family",
        "source_commit": "447a9ade6ab5fab0ebc8ae974765b46ae79c28ce",
        "source_contract": {
            "h41_panel": SOURCE_PATH,
            "h41_panel_sha256": STATIC_SHA256[SOURCE_PATH],
            "h41_build_manifest": SOURCE_BUILD_MANIFEST,
            "h41_build_manifest_sha256": STATIC_SHA256[SOURCE_BUILD_MANIFEST],
            "h41_raw_manifest": SOURCE_RAW_MANIFEST,
            "h41_raw_manifest_sha256": STATIC_SHA256[SOURCE_RAW_MANIFEST],
            "h41_audit": SOURCE_AUDIT,
            "h41_audit_sha256": STATIC_SHA256[SOURCE_AUDIT],
            "clock_ledger": CLOCK_LEDGER,
            "clock_ledger_sha256": STATIC_SHA256[CLOCK_LEDGER],
            "market": MARKET_PATH,
            "market_sha256": STATIC_SHA256[MARKET_PATH],
            "market_manifest": MARKET_MANIFEST,
            "market_manifest_sha256": STATIC_SHA256[MARKET_MANIFEST],
            "funding": FUNDING_PATH,
            "funding_sha256": STATIC_SHA256[FUNDING_PATH],
            "funding_manifest": FUNDING_MANIFEST,
            "funding_manifest_sha256": STATIC_SHA256[FUNDING_MANIFEST],
        },
        "research_history_boundary": {
            "unrelated_repo_market_history_seen_through_2023": True,
            "exact_FLCC_post_entry_outcomes_opened": False,
            "source_only_density_seen_through_2023": True,
            "density_screen_disclosure": (
                "Without opening a crypto market field, horizons 4, 8, and 13; "
                "midrank tails 0.55 through 0.85 in 0.05 increments; and component "
                "breadth 2 or 3 were inspected for support. Horizons 4/8, tails "
                "0.60/0.65, and breadth 2 were frozen because every family member "
                "retained at least 90 train entries, 20 2023 entries, both 2023 "
                "directions, and both 2023 halves. No direction PnL, hold return, "
                "CAGR, drawdown, or market value was compared."
            ),
            "interpretation": (
                "2023 event density is in-sample support screening; 2023 BTC outcomes "
                "remain sealed and can only pass or reject the Stage-1-selected member"
            ),
        },
        "novelty_boundary": {
            "distinct_axis": (
                "archived Federal Reserve weekly balance-sheet liquidity, independent "
                "of exchange OI, perpetual carry, crypto price structure, and KRW FX"
            ),
            "allowed_source_columns": [
                "release_date",
                "available_at_utc",
                "total_assets_usd_millions",
                "treasury_general_account_usd_millions",
                "reverse_repurchase_agreements_usd_millions",
                "net_liquidity_usd_millions",
            ],
            "excluded_inputs": [
                "OHLC_or_price_return",
                "crypto_volume_or_taker_flow",
                "open_interest_or_long_short_ratio",
                "perpetual_funding_premium_or_basis",
                "kimchi_BTCKRW_USDKRW_or_DXY",
                "REX_or_rolling_price_extrema",
                "existing_alpha_state_or_portfolio_PnL",
            ],
        },
        "feature_contract": {
            "net_liquidity": "N=A-TGA-RRP, all from the same archived H.4.1 release",
            "component_contributions": {
                "asset": "A[t]-A[t-h]",
                "tga_release": "-(TGA[t]-TGA[t-h])",
                "rrp_release": "-(RRP[t]-RRP[t-h])",
                "net": "N[t]-N[t-h]",
            },
            "strict_prior_midrank": (
                "For each component current impulse x[t], R=2*count(prior<x[t]) + "
                "count(prior==x[t]) over exactly 104 preceding impulse observations; "
                "current and future observations are excluded; denominator is 208"
            ),
            "family": [asdict(spec) for spec in exact_clock.CANDIDATE_SPECS],
            "breadth": (
                "at least two of asset, TGA-release, and RRP-release centered ranks "
                "must have the same sign as the net-liquidity centered rank"
            ),
            "action": "positive net-liquidity tail -> LONG; negative tail -> SHORT",
        },
        "source_only_support": support,
        "support_gates": support_checks,
        "support_passed": all(support_checks.values()),
        "execution_policy": asdict(ExecutionPolicy()),
        "execution_contract": {
            "signal": "H.4.1 available_at_utc, already 5 minutes after official release",
            "entry": "one complete five-minute bar later, at signal+5 minutes",
            "exit": "scheduled open after 1,440 five-minute bars / five calendar days",
            "nonoverlap": (
                "reserve globally per candidate and clock; ignore rather than queue "
                "an event whose entry precedes the active scheduled exit"
            ),
            "funding": "exact BTCUSDT funding settlements on [entry, exit)",
            "cagr_clock": "full wall-clock split including warm-up and idle cash",
            "strict_mdd": (
                "global/pre-entry high-water; entry cost; favorable-before-adverse "
                "held OHLC; exact funding; hypothetical liquidation; exit cost"
            ),
        },
        "falsification_controls": {
            "net_only": "same net tail without the 2-of-3 breadth requirement",
            "component_concordance_only": (
                "2-of-3 component tails in the net direction without requiring net tail"
            ),
            "direction_flip": "same primary entries with every action reversed",
            "one_release_delay": "same feature and side executed at the next H.4.1 release",
            "random_side": "hash-fixed random side on the same primary entries",
        },
        "selection_protocol": {
            "stage1_window": ["2020-01-01", "2023-01-01"],
            "stage1_subperiods": ["2020", "2021", "2022"],
            "family_size": 4,
            "stage1_gates": {
                "absolute_return_positive": True,
                "each_year_absolute_return_positive": True,
                "CAGR_to_strict_MDD_min": 3.0,
                "strict_MDD_pct_max": 15.0,
                "weekly_cluster_signflip_p_max_bonferroni": 0.025,
                "mean_gross_underlying_bp_min": 35.0,
                "ten_bp_stress_absolute_return_positive": True,
                "trades_min": 90,
                "each_side_trades_min": 40,
                "each_year_trades_min": 20,
                "single_entry_month_share_max": 0.08,
                "primary_ratio_strictly_beats_net_and_component_controls": True,
                "direction_flip_random_or_delay_full_qualification_rejects": True,
            },
            "selection_order": [
                "minimum 2020/2021/2022 CAGR-to-strict-MDD",
                "overall Stage1 CAGR-to-strict-MDD",
                "10bp stress absolute return",
                "lexical candidate_id ascending",
            ],
            "failure_action": "reject FLCC-1 without opening 2023 outcomes",
            "stage2_window": ["2023-01-01", "2024-01-01"],
            "stage2_gates": {
                "absolute_return_positive": True,
                "both_halves_absolute_return_positive": True,
                "CAGR_to_strict_MDD_min": 3.0,
                "strict_MDD_pct_max": 15.0,
                "weekly_cluster_signflip_p_max": 0.10,
                "mean_gross_underlying_bp_min": 35.0,
                "ten_bp_stress_absolute_return_positive": True,
                "trades_min": 20,
                "each_side_trades_min": 7,
                "h1_trades_min": 14,
                "h2_trades_min": 6,
                "single_entry_month_share_max": 0.20,
                "minimum_train_2023_primary_ratio_strictly_beats_controls": True,
                "random_or_delay_full_qualification_both_splits_rejects": True,
            },
            "stage2_candidate": "exact Stage1 winner only; no fallback or repair",
        },
        "sealed": ["2023_outcomes_until_stage1_pass", "2024", "2025", "2026_ytd"],
        "forbidden_repairs_after_outcomes": [
            "reverse the action mapping",
            "change horizon rank tails breadth or rank lookback",
            "add BTC trend volatility carry OI kimchi FX or another macro gate",
            "change entry delay hold leverage costs nonoverlap or split boundaries",
            "select a Stage1 runner-up after seeing 2023",
        ],
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate_manifest(payload: dict[str, Any], *, verify_sources: bool = False) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise ValueError("FLCC-1 manifest hash mismatch")
    if payload.get("outcomes_opened") is not False:
        raise ValueError("FLCC-1 preregistration opened outcomes")
    if payload.get("support_passed") is not True:
        raise ValueError("FLCC-1 source-only support did not pass")
    if verify_sources:
        for path in (SOURCE_PATH, SOURCE_BUILD_MANIFEST, SOURCE_RAW_MANIFEST, SOURCE_AUDIT, CLOCK_LEDGER):
            if _sha256(path) != STATIC_SHA256[path]:
                raise ValueError(f"FLCC-1 frozen source changed: {path}")


def build(output_path: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    payload = build_manifest()
    validate_manifest(payload, verify_sources=True)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build(args.output)
    print(
        json.dumps(
            {
                "output": args.output,
                "manifest_hash": payload["manifest_hash"],
                "support_passed": payload["support_passed"],
                "outcomes_opened": payload["outcomes_opened"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
