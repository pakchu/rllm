"""Freeze SFRD-1 before opening any post-entry BTC outcome.

SFRD-1 treats an extreme daily change in the official SOFR median as a sparse
secured-dollar-funding shock.  This module writes only a deterministic protocol
manifest.  It never parses crypto prices, returns, funding cash flow, CAGR, or
drawdown.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from training import sofr_rate_dislocation_clock as exact_clock


DEFAULT_OUTPUT = "results/sofr_rate_dislocation_preregistration_2026-07-17.json"
SOFR_PATH = (
    "data/new_york_fed_sofr_distribution_2018_2023/"
    "new_york_fed_sofr_distribution_2018-04-02_2023-12-28.csv.gz"
)
SOFR_MANIFEST_PATH = (
    "data/new_york_fed_sofr_distribution_2018_2023/build_manifest.json"
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
CLOCK_PATH = exact_clock.DEFAULT_OUTPUT


@dataclass(frozen=True)
class Policy:
    policy_id: str = "SFRD-1"
    delta_rank_lookback_observations: int = 120
    lower_tail_quantile: float = 0.15
    upper_tail_quantile: float = 0.85
    execution_delay_bars: int = 1
    hold_bars: int = 1440
    leverage: float = 0.5
    base_cost_notional_per_side: float = 0.0006
    stress_cost_notional_per_side: float = 0.0010


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def policy_payload() -> dict[str, Any]:
    return asdict(Policy())


def build_manifest() -> dict[str, Any]:
    core: dict[str, Any] = {
        "protocol_version": "sofr_rate_dislocation_v1",
        "as_of_date": "2026-07-17",
        "outcomes_opened": False,
        "policy": policy_payload(),
        "research_history_boundary": {
            "market_history_seen_by_unrelated_repo_research": True,
            "sofr_source_values_seen": True,
            "source_only_density_preflight_seen_through_2023": True,
            "candidate_class": "source-only-screened exploratory singleton",
            "density_preflight_disclosure": (
                "without opening any crypto market field, binary-float arithmetic was "
                "retired as ambiguous, then exact-decimal tail thresholds 0.80, 0.825, "
                "0.85, 0.875, 0.90, 0.925, and 0.95 were inspected for source-only "
                "clock density through 2023; 0.85 was frozen with 48 train and 40 2023 "
                "entries; no crypto outcome, action direction, or hold was compared"
            ),
            "consequence": (
                "event-density evidence through 2023 is in-sample and is not an OOS "
                "support claim; 2023 market outcomes remain sealed and may only pass "
                "or reject the now-frozen singleton"
            ),
            "exact_sfrd_1_post_entry_outcomes_opened": False,
            "support_only_access_allowed": (
                "SOFR rows, strictly causal feature values, event timestamps, side "
                "counts, calendar concentration, and control-clock overlap"
            ),
            "forbidden_before_evaluator_freeze": [
                "entry_to_exit_return",
                "post_entry_OHLC",
                "funding_PnL",
                "win_rate",
                "absolute_return",
                "CAGR",
                "drawdown",
            ],
        },
        "novelty_boundary": {
            "distinct_axis": (
                "official U.S. Treasury-repo secured-funding median-rate shocks, "
                "with no crypto-derived condition in the signal"
            ),
            "excluded_inputs": [
                "OHLC_or_price_return",
                "crypto_taker_flow_or_volume",
                "REX_or_rolling_extrema",
                "perpetual_funding_premium_basis_or_open_interest",
                "kimchi_BTCKRW_USDKRW_or_DXY",
                "orderbook_or_cross_venue_state",
                "SOFR_percentiles_or_volume_summary_statistics",
                "existing_alpha_state_or_portfolio_PnL",
            ],
            "summary_statistics_exclusion_reason": (
                "the historical API can contain quarterly-updated summary values; "
                "SFRD-1 uses only the median rate's separate same-day-final clock"
            ),
            "nearest_existing_families": {
                "crypto_carry": (
                    "uses perpetual funding and basis; SFRD uses neither and observes "
                    "the U.S. Treasury repo cash-funding market"
                ),
                "kimchi_fx": (
                    "uses KRW premium and currency state; SFRD excludes crypto prices "
                    "and FX and responds only to the official SOFR median"
                ),
                "price_regime": (
                    "uses BTC trend, extrema, or volatility; SFRD has no price input"
                ),
            },
            "forbidden_repairs_after_outcomes": [
                "reverse tightening and easing actions",
                "change rank lookback or tail thresholds",
                "add SOFR percentiles volume or another macro gate",
                "add BTC price trend volatility or regime gate",
                "change transition de-duplication",
                "change entry delay hold leverage or costs",
            ],
        },
        "source_contract": {
            "source_commit": "def42123a7830366c2bab85b7c9fccbe8a11de2a",
            "sofr": SOFR_PATH,
            "sofr_sha256": (
                "4993eda2b659e346b4d7b6e3aa0e2ff31cacf868f0e1fe2e1a5a76a03d1b5852"
            ),
            "sofr_manifest": SOFR_MANIFEST_PATH,
            "sofr_manifest_sha256": (
                "873afb5234fd013e3bc454a83713abf34d9f4a4bffc9895683add7891c636598"
            ),
            "sofr_audit": (
                "docs/new-york-fed-sofr-distribution-source-audit-2026-07-17.md"
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
            "available_source_start": "2018-04-02",
            "available_source_end": "2023-12-28",
            "gap_policy": (
                "SOFR effective dates may be one to four calendar days apart; never "
                "create weekend rows, interpolate a rate, or carry an event"
            ),
        },
        "causal_feature_contract": {
            "allowed_source_columns": [
                "effective_date",
                "sofr_available_at_utc",
                "sofr_percent",
            ],
            "forbidden_source_columns": [
                "summary_available_at_utc",
                "percentile_1_percent",
                "percentile_25_percent",
                "percentile_75_percent",
                "percentile_99_percent",
                "volume_usd_billions",
            ],
            "exact_arithmetic": (
                "parse SOFR_percent as base-10 Decimal, multiply by 100, require an "
                "exact integer basis point, then difference integer rates; binary "
                "floating point is forbidden"
            ),
            "delta": "D[t]=SOFR_integer_bp[t]-SOFR_integer_bp[t-1]",
            "strict_prior_rank": (
                "N[t]=2*count(D[t-120:t]<D[t])+count(D[t-120:t]==D[t]); "
                "R[t]=N[t]/240; the slice has exactly 120 prior finite integer "
                "deltas and excludes t"
            ),
            "state": (
                "+1 tightening if N[t]>=204 (R>=0.85); -1 easing if N[t]<=36 "
                "(R<=0.15); 0 otherwise; boundaries are inclusive"
            ),
            "episode": (
                "signal only when nonzero state differs from the immediately prior "
                "SOFR row's state; a direct +1 to -1 or -1 to +1 switch is a new event"
            ),
            "action": (
                "tightening state +1 -> fixed SHORT; easing state -1 -> fixed LONG"
            ),
            "availability": (
                "D[t] and R[t] become usable only at row t sofr_available_at_utc; "
                "wait one complete five-minute bucket before entry"
            ),
            "price_signal_columns": [],
        },
        "support_verification": {
            "candidate_class": "source-only-screened exploratory singleton",
            "pre_freeze_source_only_screen": {
                "tail_quantile_grid": [0.80, 0.825, 0.85, 0.875, 0.90, 0.925, 0.95],
                "selected_lower_upper": [0.15, 0.85],
                "source_support_seen_through_2023": True,
                "crypto_market_or_outcome_fields_seen": False,
                "interpretation": (
                    "density and concentration are in-sample screening evidence, not "
                    "independent generalization evidence"
                ),
            },
            "varying_parameters": [],
            "selection_rule": (
                "no remaining selection after the disclosed source-only screen; exact "
                "replay may reject implementation drift but may not choose a fallback"
            ),
            "train_window": ["2021-01-01", "2023-01-01"],
            "support_seen_outcome_sealed_window": ["2023-01-01", "2024-01-01"],
            "clock_implementation": "training/sofr_rate_dislocation_clock.py",
            "clock_ledger": CLOCK_PATH,
            "clock_ledger_sha256": (
                "391c42dd2b0d5b87ffcd73058dd9fa0c4d18fd2f535597effff5a4c8edea2e69"
            ),
            "clock_ledger_events_full_source": 158,
            "exact_clock_replay_required": True,
            "minimum_nonoverlap_train": 45,
            "minimum_2021": 10,
            "minimum_2022": 35,
            "minimum_2023": 35,
            "minimum_2023_h1": 15,
            "minimum_2023_h2": 18,
            "minimum_train_each_side": 15,
            "minimum_2023_each_side": 18,
            "maximum_single_month_share_train": 0.15,
            "maximum_single_month_share_2023": 0.15,
            "expected_preflight_counts": {
                "train": 48,
                "2021": 12,
                "2022": 36,
                "2023": 40,
                "2023_h1": 18,
                "2023_h2": 21,
                "train_long": 31,
                "train_short": 17,
                "2023_long": 20,
                "2023_short": 20,
                "train_max_single_month_count": 5,
                "train_max_single_month_share": 0.10416666666666667,
                "2023_max_single_month_count": 5,
                "2023_max_single_month_share": 0.125,
            },
            "failure_action": (
                "retire SFRD-1 before loading any market outcome; no fallback"
            ),
        },
        "execution_contract": {
            "decision_time": "row t sofr_available_at_utc (19:00 UTC EDT / 20:00 UTC EST)",
            "entry": "first five-minute open after one complete post-availability bar (+5 minutes)",
            "exit": "scheduled open after 1,440 held five-minute bars (five calendar days)",
            "nonoverlap": True,
            "clock_reservation": (
                "reserve globally over the complete pre-2024 event clock; ignore, do "
                "not queue, any event whose entry precedes the current scheduled exit; "
                "slice only trades whose signal entry and exit are fully inside a split"
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
        "falsification_controls": {
            "direction_flip": "same primary clock with LONG for tightening and SHORT for easing",
            "level_tail": (
                "own 120-prior mid-rank clock on SOFR_percent level with identical "
                "0.15/0.85 state transition; high level -> SHORT, low level -> LONG"
            ),
            "five_observation_change_tail": (
                "own exact-integer clock replacing D[t] with SOFR_bp[t]-SOFR_bp[t-5]; "
                "high change -> SHORT, low change -> LONG, otherwise identical"
            ),
            "month_turn": (
                "own clock on first or last SOFR effective date of each UTC month; "
                "SHORT if D[t]>0, LONG if D[t]<0, no event if D[t]==0"
            ),
            "one_observation_delay": (
                "same primary state and side shifted to the next SOFR availability"
            ),
            "random_side": (
                "same primary clock; for each entry_time ISO string in ascending ledger "
                "order compute SHA256('SFRD-1-random-side-20260717|' + entry_time); "
                "first digest byte <128 -> LONG else SHORT; event-level, no RNG state"
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
            "sealed": ["2024", "2025", "2026_ytd"],
            "candidate_count": 1,
            "stage2_requires_unchanged_stage1_pass": True,
            "no_parameter_repair": True,
            "gates": {
                "train_and_2023_absolute_return_positive": True,
                "train_and_2023_cagr_to_strict_mdd_min": 3.0,
                "train_and_2023_strict_mdd_pct_max": 15.0,
                "train_and_2023_weekly_cluster_signflip_p_max": 0.10,
                "train_trades_min": 45,
                "train_each_side_trades_min": 15,
                "train_single_month_share_max": 0.15,
                "2021_trades_min": 10,
                "2022_trades_min": 35,
                "2023_trades_min": 35,
                "2023_each_side_trades_min": 18,
                "2023_single_month_share_max": 0.15,
                "2023_h1_trades_min": 15,
                "2023_h2_trades_min": 18,
                "2021_2022_and_2023_halves_absolute_return_positive": True,
                "train_and_2023_mean_gross_underlying_bp_min": 35.0,
                "train_and_2023_ten_bp_stress_absolute_return_positive": True,
                "stage1_primary_ratio_strictly_beats_mechanism_controls": True,
                "minimum_train_2023_primary_ratio_strictly_beats_controls": True,
                "one_observation_delay_or_random_full_qualification_rejects": True,
            },
            "statistical_test_contract": {
                "cluster": "UTC ISO week of entry",
                "null": "independent Rademacher sign applied to each weekly cluster",
                "statistic": "mean net account return per trade",
                "alternative": "primary statistic greater than zero",
                "draws": 20_000,
                "seed": 20_260_717,
                "p_value": "(1 + null statistics >= observed)/(20000 + 1)",
            },
            "control_comparison_contract": {
                "mechanism_controls": [
                    "level_tail",
                    "five_observation_change_tail",
                    "month_turn",
                ],
                "ratio": "full-clock CAGR / max(strict MDD, 1e-9)",
                "stage1": (
                    "primary train ratio must be strictly greater than each "
                    "mechanism-control train ratio; equality rejects"
                ),
                "stage2": (
                    "min(primary train ratio, primary 2023 ratio) must be strictly "
                    "greater than the same minimum for each control; equality rejects"
                ),
                "promotion": "no control may replace the singleton primary",
            },
        },
        "orthogonality_after_standalone_pass": {
            "economic_gate_first": True,
            "comparator_universe": {
                "deduplicated_universe_audit": {
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
            },
            "duplicate_policy": (
                "collapse exact PnL hashes and compare against every canonical family "
                "representative plus synchronized selected portfolios"
            ),
            "exact_entry_jaccard_max": 0.02,
            "near_six_hour_entry_fraction_max": 0.25,
            "position_time_jaccard_max": 0.15,
            "absolute_daily_pnl_pearson_max": 0.30,
            "minimum_nonzero_pnl_days": 20,
            "marginal_portfolio_improvement_required": True,
        },
        "rllm_boundary": {
            "enabled_only_after_deterministic_standalone_pass": True,
            "allowed_tokens": [
                "sofr_delta_bp",
                "sofr_delta_prior_rank",
                "tightening_easing_state",
                "effective_date_gap_days",
                "current_position",
                "time_to_exit",
            ],
            "allowed_actions": ["abstain", "size_frozen_side"],
            "forbidden_actions": ["reverse_side", "change_base_event"],
        },
        "rejection_contract": (
            "any support or staged performance failure retires SFRD-1 without "
            "threshold side feature delay hold or gate repair"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_manifest(payload: dict[str, Any], *, verify_sources: bool = True) -> None:
    manifest_hash = payload.get("manifest_hash")
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if manifest_hash != canonical_hash(core):
        raise ValueError("SFRD-1 preregistration manifest hash changed")
    if payload.get("outcomes_opened") is not False:
        raise ValueError("SFRD-1 outcomes opened before preregistration")
    if payload.get("policy") != policy_payload():
        raise ValueError("SFRD-1 policy differs from frozen defaults")
    if payload["selection_protocol"].get("candidate_count") != 1:
        raise ValueError("SFRD-1 must remain a singleton")
    if payload["causal_feature_contract"].get("price_signal_columns") != []:
        raise ValueError("SFRD-1 price entered the source-only signal")
    if payload["support_verification"].get("varying_parameters") != []:
        raise ValueError("SFRD-1 support may not select parameters")
    allowed = set(payload["causal_feature_contract"]["allowed_source_columns"])
    forbidden = set(payload["causal_feature_contract"]["forbidden_source_columns"])
    if allowed & forbidden or "sofr_percent" not in allowed:
        raise ValueError("SFRD-1 source column boundary is inconsistent")
    support = payload["support_verification"]
    gates = payload["selection_protocol"]["gates"]
    count_pairs = (
        ("minimum_nonoverlap_train", "train_trades_min"),
        ("minimum_2021", "2021_trades_min"),
        ("minimum_2022", "2022_trades_min"),
        ("minimum_2023", "2023_trades_min"),
        ("minimum_2023_h1", "2023_h1_trades_min"),
        ("minimum_2023_h2", "2023_h2_trades_min"),
    )
    if any(support[left] != gates[right] for left, right in count_pairs):
        raise ValueError("SFRD-1 support and performance count gates diverged")
    for left, right in (
        ("minimum_train_each_side", "train_each_side_trades_min"),
        ("minimum_2023_each_side", "2023_each_side_trades_min"),
        ("maximum_single_month_share_train", "train_single_month_share_max"),
        ("maximum_single_month_share_2023", "2023_single_month_share_max"),
    ):
        if support[left] != gates[right]:
            raise ValueError("SFRD-1 side or concentration gates diverged")
    controls = set(payload["falsification_controls"])
    comparison = set(
        payload["selection_protocol"]["control_comparison_contract"][
            "mechanism_controls"
        ]
    )
    if not comparison.issubset(controls):
        raise ValueError("SFRD-1 performance control lacks a definition")
    if verify_sources:
        source = payload["source_contract"]
        for key, hash_key in (
            ("sofr", "sofr_sha256"),
            ("sofr_manifest", "sofr_manifest_sha256"),
            ("market", "market_sha256"),
            ("market_manifest", "market_manifest_sha256"),
            ("funding", "funding_sha256"),
            ("funding_manifest", "funding_manifest_sha256"),
        ):
            if _sha256(source[key]) != source[hash_key]:
                raise ValueError(f"SFRD-1 frozen source changed: {source[key]}")
        for item in payload["orthogonality_after_standalone_pass"][
            "comparator_universe"
        ].values():
            if _sha256(item["path"]) != item["sha256"]:
                raise ValueError(f"SFRD-1 comparator changed: {item['path']}")
        if _sha256(support["clock_ledger"]) != support["clock_ledger_sha256"]:
            raise ValueError("SFRD-1 source-only clock ledger changed")
        events = exact_clock.build_events(exact_clock.read_source(source["sofr"]))
        ledger_events = exact_clock.read_event_ledger(support["clock_ledger"])
        if events != ledger_events:
            raise ValueError("SFRD-1 rebuilt clock rows differ from frozen ledger")
        if len(events) != support["clock_ledger_events_full_source"]:
            raise ValueError("SFRD-1 source-only clock event count changed")


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = write_manifest(args.output)
    print(
        json.dumps(
            {
                "output": args.output,
                "policy_id": payload["policy"]["policy_id"],
                "manifest_hash": payload["manifest_hash"],
                "outcomes_opened": payload["outcomes_opened"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
