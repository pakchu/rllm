"""Freeze CCBS-12 before any spread PnL is opened."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = "results/cross_collateral_basis_snapback_preregistration_2026-07-17.json"
DEFAULT_DOCS = "docs/cross-collateral-basis-snapback-preregistration-2026-07-17.md"
SOURCE_MANIFEST = "results/binance_cross_collateral_quarterly_curve_2021_2023_manifest.json"
SOURCE_MANIFEST_CONTENT_HASH = (
    "197755f0ce6823eea7d0fd47e6db5cbec2ddb1a18542fc47b57ab7f02f69321b"
)
SOURCE_MANIFEST_FILE_SHA256 = (
    "cd43af86460780e874f8cf2522820d20dba0fdcc3fdb5f5e07508795065fbd9f"
)
SOURCE_PANEL_FILE_SHA256 = (
    "54addc04b997cfb077197cd845f2aa286a219bdae4a29b49c2086667007046f7"
)


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def protocol() -> dict[str, Any]:
    return {
        "protocol_version": "ccbs12_v1_2026-07-17",
        "name": "CCBS-12 — Cross-Collateral Basis Snapback",
        "claim": (
            "An unusually wide same-maturity price wedge between Binance USD-M and "
            "COIN-M BTC current-quarter futures mean-reverts within twelve hours strongly "
            "enough for an equal-initial-USD-face two-leg spread to survive exact costs."
        ),
        "evidence_boundary": {
            "physical_source_rows_opened": True,
            "feature_support_distribution_opened": True,
            "2023_feature_support_seen": True,
            "spread_entry_to_exit_returns_opened": False,
            "2023_spread_pnl_opened": False,
            "2024_or_later_source_opened_for_this_family": False,
            "adjacent_cross_collateral_directional_families_exist": True,
            "exact_same_maturity_spread_family_algorithmically_unopened": True,
            "repository_wide_human_pristine_claim": False,
            "2023_label": "development; outcome-blind only, not feature-pristine OOS",
            "2024_label": "first code-frozen source-and-outcome-unopened OOS",
        },
        "source_contract": {
            "manifest": SOURCE_MANIFEST,
            "manifest_file_sha256": SOURCE_MANIFEST_FILE_SHA256,
            "manifest_content_hash": SOURCE_MANIFEST_CONTENT_HASH,
            "manifest_content_hash_algorithm": (
                "SHA-256 of canonical JSON body after excluding manifest_hash and created_at"
            ),
            "panel_file_sha256": SOURCE_PANEL_FILE_SHA256,
            "panel": (
                "BTCUSDT and BTCUSD CURRENT_QUARTER continuous-contract 5m OHLC from "
                "official Binance public market data"
            ),
            "official_endpoint": (
                "https://developers.binance.com/en/docs/catalog/"
                "core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data"
                "#continuous-contract-kline-candlestick-data"
            ),
            "declared_range": ["2021-01-01", "2024-01-01"],
            "actual_joint_prefix_starts": "2021-02-03T08:20:00Z",
            "development_outcome_period": ["2023-01-01", "2024-01-01"],
            "first_unopened_oos_period": ["2024-01-01", "2025-01-01"],
            "no_post_2023_source_before_2023_gate": True,
            "clean_row": (
                "source_complete and both OHLC-valid flags true; no fill, nearest join, "
                "or cross-segment carry"
            ),
        },
        "economic_object": {
            "um_leg": "BTCUSDT USD-M linear current-quarter future",
            "cm_leg": "BTCUSD COIN-M inverse current-quarter future",
            "maturity_match": "both legs share the panel delivery_time and contract_segment",
            "initial_leg_gross": {"um": 0.5, "cm": 0.5},
            "gross_exposure": 1.0,
            "funding_cashflow": "none; both instruments are delivery futures",
            "direction_if_um_rich": "short UM and long CM",
            "direction_if_cm_rich": "long UM and short CM",
            "structural_orthogonality_hypothesis": (
                "relative value between collateral conventions with opposite initial "
                "derivative legs; no explicit BTC-direction, perpetual funding/premium, OI, "
                "Kimchi/FX, REX, Markov, tree, or LLM gate"
            ),
        },
        "feature_formula": {
            "wedge": "w_t = log(um_close_t / cm_close_t)",
            "segmentation": "all rolling state resets at each contract_segment",
            "lookback_bars": 4032,
            "lookback_calendar_days": 14,
            "minimum_prior_bars": 3226,
            "minimum_fraction": 0.8,
            "prior": "p_t = w_t shifted by one row within contract_segment",
            "center": "rolling median(p_t, 4032, min_periods=3226)",
            "recursive_mad": (
                "rolling median(abs(p_t - center_t), 4032, min_periods=3226)"
            ),
            "zscore": "z_t = (w_t - center_t) / (1.4826 * recursive_mad_t)",
            "minimum_absolute_dislocation": 0.002,
            "minimum_absolute_dislocation_bp": 20.0,
            "threshold_support_grid": [1.5, 2.0, 2.5, 3.0],
            "outcome_columns_forbidden_during_threshold_selection": True,
        },
        "support_only_selection": {
            "selection_period": ["2021-01-01", "2023-01-01"],
            "2023_support_not_used_for_selection": True,
            "candidate": (
                "clean state onset where abs(z)>=threshold and abs(w-center)>=20bp; a "
                "direct z-sign reversal is a new onset"
            ),
            "reservation": (
                "after a kept candidate, reserve through entry+12h even if the eventual "
                "trade exits early; ignored onsets cannot create a delayed entry"
            ),
            "candidate_must_allow_exit_before_delivery": True,
            "threshold_rule": "select the largest grid value passing every support floor",
            "support_floors": {
                "pre_2023_total_events_at_least": 130,
                "each_calendar_year_events_at_least": 50,
                "each_half_year_events_at_least": 25,
                "each_quarter_events_at_least": 6,
                "each_z_sign_share_at_least": 0.30,
            },
            "reject_if_no_threshold_passes": True,
            "no_return_or_ohlc_path_statistic_may_break_ties": True,
        },
        "signal_and_execution_clock": {
            "signal_bar": "completed 5m bar with open time t",
            "signal_available": "t+5m, after the source close_time",
            "entry": "t+10m open; one full 5m execution-delay bar after availability",
            "maximum_hold_bars": 144,
            "maximum_hold_hours": 12,
            "normalization_exit_trigger": "first completed held bar with abs(z)<=0.5",
            "normalization_exit": (
                "trigger bar open time +10m; one full 5m execution-delay bar after availability"
            ),
            "hard_exit": "entry+12h open",
            "exit_priority": "earlier valid normalization exit, otherwise hard exit",
            "roll_rule": "entry and every possible exit must be strictly before delivery_time",
            "overlap": "one active spread; full reserved clock forbids re-entry",
        },
        "derivative_ledger": {
            "entry_sequence": (
                "read pre-entry equity E, freeze target face F=0.5*E for each leg, freeze "
                "quantities from entry opens, then deduct entry fees; entry fees do not resize"
            ),
            "initial_sizing": "fixed USD face F=0.5*pre-entry equity per leg",
            "um_quantity_btc": "F / um_entry_open",
            "um_quantity_rounding": "none; fractional research quantity",
            "cm_contract_multiplier_usd": 100.0,
            "cm_contract_quantity": "F / 100",
            "cm_contract_rounding": "none; fractional research contract quantity",
            "um_usd_pnl": "side_um * um_quantity_btc * (um_mark - um_entry_open)",
            "cm_coin_pnl": (
                "side_cm * cm_contract_quantity * 100 * "
                "(1/cm_entry_open - 1/cm_mark) in BTC"
            ),
            "cm_usd_derivative_pnl": (
                "cm_coin_pnl * the same contemporaneous cm_mark used inside the inverse "
                "formula; high uses high, low uses low, exit uses exit open"
            ),
            "base_cost_bp_per_notional_side": 6.0,
            "stress_cost_bp_per_notional_side": 10.0,
            "base_gross_one_round_trip_cost_bp": 12.0,
            "um_entry_fee": "base_rate * abs(um_quantity_btc) * um_entry_open",
            "um_exit_fee": "base_rate * abs(um_quantity_btc) * um_exit_or_mark_price",
            "cm_entry_fee": "base_rate * abs(cm_contract_quantity) * 100",
            "cm_exit_fee": "base_rate * abs(cm_contract_quantity) * 100",
            "fees": (
                "charged on each leg at entry and exit from portfolio equity; stress ledger "
                "replaces base_rate with 10bp; no maker rebate"
            ),
            "exit_sequence": (
                "mark both derivative legs at the exit open, realize gross PnL, deduct both "
                "exit fees, then use resulting equity for the next accepted trade"
            ),
            "funding": "zero",
        },
        "strict_risk_contract": {
            "hwm": "global/pre-entry high-water mark across the full declared wall clock",
            "held_bar_order": (
                "mark the favorable two-leg OHLC combination before the adverse combination; "
                "cross-venue extrema are combined adversarially even if not simultaneous"
            ),
            "favorable_prices": {
                "long": "high",
                "short": "low",
            },
            "adverse_prices": {
                "long": "low",
                "short": "high",
            },
            "hypothetical_liquidation_cost": "deduct both-leg exit costs at every adverse mark",
            "hypothetical_um_liquidation_cost": (
                "rate * abs(frozen um quantity) * adverse um mark"
            ),
            "hypothetical_cm_liquidation_cost": (
                "rate * abs(frozen cm contract quantity) * 100 USD"
            ),
            "entry_and_exit_costs": True,
            "missing_held_path": (
                "fail the evaluation as a source-integrity error; never delete or shorten "
                "an accepted trade using future data availability"
            ),
            "cagr": "full wall-clock, including warm-up, idle, and reserved periods",
            "absolute_return_always_reported": True,
        },
        "collateral_accounting_boundary": {
            "research_ledger_includes": "derivative PnL of both legs in USD terms",
            "research_ledger_omits": (
                "mark-to-market of BTC collateral posted for the inverse COIN-M leg, margin "
                "interest, transfers, liquidation-engine interactions, and executable "
                "integer contract/quantity rounding"
            ),
            "consequence": (
                "historical CCBS is not evidence of account-level BTC neutrality and cannot "
                "be promoted directly to live trading"
            ),
            "live_promotion_blocker": (
                "implement and verify an exact BTC-collateral ledger plus either a causal "
                "collateral hedge or documented unified-margin treatment"
            ),
        },
        "frozen_policy": {
            "policy_id": "CCBS12",
            "threshold": "selected once by the support-only largest-passing rule",
            "entry_dislocation_floor_bp": 20.0,
            "normalization_z": 0.5,
            "maximum_hold_hours": 12,
            "gross": 1.0,
            "no_directional_or_regime_gate": True,
        },
        "development_2023_gate": {
            "evidence_label": "outcome-blind but feature-support-seen development gate",
            "single_policy_no_ranking": True,
            "absolute_return_positive": True,
            "cagr_to_strict_mdd_at_least": 3.0,
            "strict_mdd_at_most_pct": 15.0,
            "trades_at_least": 50,
            "h1_and_h2_absolute_return_positive": True,
            "both_z_sign_branches_absolute_return_positive": True,
            "ten_bp_cost_stress_absolute_return_positive": True,
            "gross_profit_exceeds_transaction_cost": True,
            "base_cost_trade_net_return": (
                "realized two-leg derivative PnL minus entry and exit fees, divided by "
                "pre-entry equity"
            ),
            "absolute_return": (
                "compound all base-cost trade net returns on the full declared calendar"
            ),
            "half_and_branch_returns": (
                "compound base-cost net trade returns by entry half or entry z-sign; an empty "
                "bucket fails rather than passing at zero"
            ),
            "gross_profit": "sum realized two-leg derivative PnL before all fees",
            "transaction_cost": "sum all base-cost entry and exit fees on both legs",
            "median_signed_wedge_convergence_positive": True,
            "signed_wedge_convergence": (
                "-sign(z_signal) * (log(um_exit_open/cm_exit_open) - "
                "log(um_entry_open/cm_entry_open))"
            ),
            "signed_wedge_convergence_hit_rate_at_least": 0.55,
            "monthly_cluster_signflip_p_at_most": 0.10,
            "active_entry_months_at_least": 6,
            "monthly_cluster_signflip": (
                "sum base-cost trade net returns by UTC entry month, exclude zero-trade "
                "months, enumerate all 2^m sign assignments to the m active monthly sums, "
                "and set one-sided p=count(permuted total>=observed total)/2^m"
            ),
        },
        "post_2023_sequence": {
            "2023_is_not_called_clean_oos": True,
            "open_2024_only_after_development_2023_pass": True,
            "2024_is_first_code_frozen_unopened_oos": True,
            "open_2025_only_after_2024_pass": True,
            "open_2026_only_after_2025_pass": True,
            "each_new_year_uses_the_identical_policy": True,
            "each_year_absolute_return_positive": True,
            "each_year_cagr_to_strict_mdd_at_least": 3.0,
            "each_year_strict_mdd_at_most_pct": 15.0,
            "each_year_both_halves_positive": True,
            "each_year_both_z_sign_branches_positive": True,
            "each_year_ten_bp_stress_positive": True,
            "each_year_gross_profit_exceeds_cost": True,
            "each_year_convergence_hit_rate_at_least": 0.55,
            "each_year_monthly_signflip_p_at_most": 0.10,
            "terminal_failure": (
                "write a rejection artifact for the failed year, mark CCBS12 rejected, and "
                "do not open or fetch any later historical year for this family"
            ),
            "all_historical_passes_disposition": (
                "research candidate only; do not promote until the collateral ledger and "
                "forward-shadow requirements pass"
            ),
            "minimum_forward_shadow_days_for_live_promotion": 90,
        },
        "orthogonality_gate": {
            "live_anchor": {
                "config": "configs/live/portfolio_gross385_trainmdd40_2026-07-12.json",
                "config_sha256": (
                    "86f255ca3967245b8b0676b00025b955d7f33668ab1ef9d813623191b4ecd1e7"
                ),
                "weights": {
                    "oi_upbit_ratio288_low": 0.65,
                    "new_long_minimal_funding_premium": 1.75,
                    "cand_rex_veto_7": 1.45,
                },
            },
            "frozen_anchor_entry_clock_artifact": (
                "results/cross_collateral_basis_snapback_live_anchor_clock_<year>.json"
            ),
            "anchor_clock_construction": (
                "reconstruct only the three configured sleeve signal positions on the frozen "
                "5m market grid; entry is signal_pos+1; retain only [year-Jan-01, "
                "next-Jan-01) UTC timestamps; record artifact SHA-256 and every input SHA-256"
            ),
            "anchor_clock_sequence": (
                "build the 2023 clock for the development gate; build each 2024+ year clock "
                "only when that year's source is sequentially authorized to open"
            ),
            "anchor_clock_inputs": {
                "builder": {
                    "path": "training/portfolio_opt_added_alpha_update.py",
                    "sha256": (
                        "d98b79db1053190087ed274d0b37f91961f55c2128b64c3066970a957d313db9"
                    ),
                },
                "market": {
                    "path": "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz",
                    "sha256": (
                        "a77cd0ae5b88b3c95e509d8d2610773d34af3afdc9170c63d88564bc3d0b990c"
                    ),
                },
                "open_interest_cache": {
                    "path": "/tmp/btcusdt_open_interest_5m_2020_2026.csv",
                    "sha256": (
                        "e08f93033e56959e8e7a9c1e21f27c5f01efc8d06fa6b4fbbfe7354697122b31"
                    ),
                },
                "upbit_prefix": {
                    "path": (
                        "/home/pakchu/workspace/wave_trading/data/"
                        "2020-01-01_2025-12-15_4bd081fc54811fccdee66850692c435e.csv.gz"
                    ),
                    "sha256": (
                        "7c377c402b4c1c3db3dafb5e15cd06e93f6e9c2c08d154ed88dd47e91f86eb35"
                    ),
                },
                "funding": {
                    "path": "data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz",
                    "sha256": (
                        "4d381be086e275bacaf31df431dc31307a71a26b3947b7082efffc10bb129dd7"
                    ),
                },
                "premium": {
                    "path": "data/binance_um_aux_btc_2020_2026/BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz",
                    "sha256": (
                        "b45fcc5a3cf75c8e594effe61a698c4652f841b1d304107e9669524e0fc9d0d7"
                    ),
                },
                "rex_reasoning_clock": {
                    "path": "data/rex_event_reasoning_policy_sft_20260712.jsonl",
                    "sha256": (
                        "2f5f477ed7ffd6063bd25b1fdbcb6cbaa804685be43b4522b7105dfba1b75d48"
                    ),
                },
            },
            "economic_mechanism_must_remain_relative_value": True,
            "exact_5m_entry_overlap_share_with_live_anchor_at_most": 0.10,
            "exact_5m_entry_overlap_share": (
                "CCBS entries exactly equal to any anchor entry divided by all CCBS entries"
            ),
            "entry_day_jaccard_with_live_anchor_at_most": 0.20,
            "entry_day_jaccard": "intersection over union of UTC entry-date sets",
            "entry_overlap_scope": (
                "apply exact-time overlap and day Jaccard separately inside every evaluated "
                "development or OOS calendar year"
            ),
            "daily_pnl_absolute_correlation_with_live_anchor_at_most": 0.30,
            "daily_pnl_absolute_btc_correlation_at_most": 0.20,
            "daily_pnl_absolute_btc_beta_at_most": 0.15,
            "btc_return_source": {
                "path": "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz",
                "sha256": (
                    "a77cd0ae5b88b3c95e509d8d2610773d34af3afdc9170c63d88564bc3d0b990c"
                ),
                "formula": (
                    "UTC-day final BTCUSDT close divided by prior UTC-day final close minus one"
                ),
            },
            "anchor_daily_pnl_artifact": (
                "results/cross_collateral_basis_snapback_live_anchor_daily_pnl_<year>.json"
            ),
            "anchor_daily_pnl_construction": (
                "base-cost corrected-strict live-anchor replay at frozen weights; compound "
                "5m portfolio returns inside each UTC day and record artifact/input hashes"
            ),
            "daily_alignment": (
                "each evaluated calendar year [Jan-01, next Jan-01), all UTC days included, "
                "CCBS and anchor no-trade days zero-filled; BTC beta is "
                "cov(CCBS daily PnL, BTC daily return)/var(BTC daily return)"
            ),
            "marginal_portfolio_improvement": {
                "ccbs_weight": 0.25,
                "comparison": "frozen live anchor versus live anchor plus 0.25*CCBS",
                "combined_absolute_return_exceeds_anchor": True,
                "combined_cagr_to_strict_mdd_improvement_at_least": 0.25,
                "combined_strict_mdd_increase_at_most_pct_points": 1.0,
                "accounting": (
                    "same full-calendar favorable-before-adverse portfolio ledger and base costs"
                ),
            },
        },
        "diagnostics_not_repair_candidates": [
            "reverse convergence direction",
            "change 14-day lookback or recursive MAD",
            "change 20bp dislocation floor",
            "choose a z threshold using PnL",
            "change normalization or 12-hour exit",
            "add REX, OI, funding, Kimchi/FX, Markov, tree, LLM, or regime gate",
            "drop one losing z-sign branch",
            "cross a delivery roll",
        ],
        "stop_rule": (
            "Reject before opening 2024 if the pre-2023 support-only rule finds no threshold "
            "or CCBS12 fails the frozen 2023 development gate. At any later annual failure, "
            "write the terminal rejection artifact and do not open a subsequent year. Do not "
            "repair sign, lookback, threshold, floor, exit, cost, branch, sizing, or add a "
            "directional gate after outcomes open. A historical pass remains research-only "
            "until the collateral-accounting blocker is closed."
        ),
    }


def markdown(payload: dict[str, Any]) -> str:
    return f"""# CCBS-12 preregistration — 2026-07-17

## Hypothesis and orthogonality

CCBS-12 tests a **hypothesized relative-value mechanism** in the same-maturity
USD-M versus COIN-M BTC current-quarter futures wedge. A rich USD-M leg is sold
against a long COIN-M leg; a rich COIN-M leg is sold against a long USD-M leg.
It has no explicit BTC-direction, perpetual funding/premium, OI, Kimchi/FX,
REX, Markov, tree, LLM, or regime gate. The physical panel comes from Binance's official
[continuous-contract kline endpoint]({payload['protocol']['source_contract']['official_endpoint']}).

No entry-to-exit spread return or 2023 spread PnL was opened before this
protocol. Feature support counts, including 2023 counts, were inspected.
Therefore 2023 is honestly labeled **outcome-blind development**, not pristine
OOS. Threshold selection is restricted to 2021-2022; 2024 is the first code-
frozen source-and-outcome-unopened OOS year.

## Frozen feature and support-only selection

- `w = log(USD-M close / COIN-M close)`;
- all state resets on every delivery-contract segment;
- strictly prior 14-day/4,032-bar rolling median and recursive MAD, with 3,226
  prior observations required;
- state onset requires `abs(z) >= threshold` and at least a 20 bp deviation
  from the rolling center;
- grid: `1.5, 2.0, 2.5, 3.0`; using 2021-2022 only, choose the **largest**
  value with at least 130 events, 50 per year, 25 per half, 6 per quarter, and
  30% per sign;
- PnL, future OHLC paths, and return columns are forbidden in this selection.

The completed signal bar at `t` is available at `t+5m`; entry is delayed to
the `t+10m` open. Each accepted onset reserves a full 12 hours. The spread
exits after a completed `abs(z)<=0.5` trigger with the same one-full-bar delay,
or at 12 hours. No possible path may touch or cross delivery.

## Frozen accounting and strict risk

Before fees, each leg freezes face `F=0.5*pre-entry equity`. The research
ledger uses fractional USD-M BTC quantity `F/entry` and fractional COIN-M
contracts `F/100`; neither is rounded. USD-M uses linear PnL. COIN-M uses the
inverse coin formula and converts PnL with the **same** contemporaneous price
used in that mark. Entry costs do not resize either quantity. Both legs pay 6
bp per transaction side (10 bp stress); delivery futures pay no funding.
Strict MDD uses the global/pre-entry HWM, combines
cross-venue favorable extrema before adverse extrema, and includes hypothetical
two-leg liquidation cost at every adverse mark. CAGR covers the full calendar,
including warm-up and idle time, and absolute return is always reported.

## Hard collateral boundary

The research ledger deliberately measures **derivative spread PnL only**. It
does not model BTC collateral posted for COIN-M, transfers, margin interest, or
the liquidation engine. Therefore a historical pass is not proof of account-
level neutrality and is not live-promotable. It also omits executable integer
contract rounding. An exact BTC-collateral ledger, live contract constraints,
and either a causal collateral hedge or documented unified-margin treatment
are hard prerequisites.

## Sequential gate and no repair

The single pre-2023-support-selected policy first opens 2023 development PnL.
It must produce
positive absolute return, CAGR/strict-MDD >= 3, strict MDD <= 15%, at least 50
trades, positive H1/H2 and both spread branches, positive 10 bp stress, and
direct wedge-convergence attribution. Only then may the first untouched OOS
year, 2024, open, followed by 2025 and 2026 under the identical policy. Every
failure writes a terminal rejection artifact and seals later years. No sign,
lookback, threshold, floor, exit, cost, branch, sizing, or directional-gate
repair is allowed after outcomes open.

Entry-clock overlap is measured against the hash-bound gross-3.85 live anchor.
PnL orthogonality uses full UTC calendars with no-trade days zero-filled, the
hash-bound BTCUSDT daily close return, and a fixed 0.25 CCBS incremental-weight
portfolio comparison. These are gates to test the relative-value hypothesis,
not claims made in advance.

Protocol hash: `{payload['protocol_hash']}`
"""


def run(output: str = DEFAULT_OUTPUT, docs_output: str = DEFAULT_DOCS) -> dict[str, Any]:
    frozen = protocol()
    payload = {
        "protocol": frozen,
        "protocol_hash": canonical_hash(frozen),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    Path(docs_output).parent.mkdir(parents=True, exist_ok=True)
    Path(docs_output).write_text(markdown(payload))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--docs-output", default=DEFAULT_DOCS)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.docs_output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
