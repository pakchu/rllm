"""Outcome-blind preregistration for BSCBR-24."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path(
    "results/bitcoin_stock_correlation_break_relay_preregistration_2026-08-09.json"
)


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "bitcoin_stock_correlation_break_relay_v1",
        "policy_id": "BSCBR-24",
        "as_of_date": "2026-08-09",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": (
                "After a completed US cash session, a sharp one-session fall "
                "(rise) in the causal BTC-SPY return correlation represents a "
                "rise (fall) in Bitcoin diversification demand that should be "
                "worked through the following day. The relay is admitted only "
                "when the completed BTC conditional volatility is already high."
            ),
            "side": "negative sign of the one-session correlation change",
            "external_support": {
                "paper": "Yae and Tian (2024), Volatile safe-haven asset: Evidence from Bitcoin",
                "doi": "10.1016/j.jfs.2024.101285",
                "supported_direction": (
                    "a decrease (increase) in time-varying BTC-stock return "
                    "correlation predicts higher (lower) next-day BTC return"
                ),
                "implementation_is_not_a_replication": True,
                "adaptations": [
                    "SPY is a tradable S&P 500 proxy",
                    "a deterministic Gaussian-QML DCC-GARCH implementation is frozen below",
                    "a causal high-BTC-variation gate targets the requested volatile regime",
                ],
            },
            "why_distinct": (
                "COGR used same-session QQQ/GLD opening gaps and a shallow tree. "
                "Hidden safe-haven work used FX cross-sectional cancellation. "
                "NVXCR used VIX/VXN changes. BSCBR uses the one-session change "
                "of a jointly estimated BTC-SPY close-return correlation and "
                "does not promote any prior control or event set."
            ),
            "why_suited_to_volatile_regimes": (
                "the post-close BTC conditional volatility must rank in the "
                "causal upper 35% of prior US cash sessions"
            ),
            "why_low_gross9_overlap_is_plausible": (
                "one US-close cross-asset state change per cash session is not "
                "an existing Gross9 entry primitive"
            ),
        },
        "clock": {
            "session": "official SPY US cash session in America/New_York",
            "cash_close": (
                "actual official NYSE close: normally 16:00 America/New_York; "
                "13:00 on a frozen enumerated early-close calendar"
            ),
            "feature_available": (
                "16:05 America/New_York after the SPY close and the matched "
                "completed BTC hour are both available"
            ),
            "entry": "exact BTCUSDT 16:10 America/New_York five-minute open",
            "spy_return": (
                "log((split-normalized close_t + effective-date cash "
                "dividend_t) / split-normalized close_previous_SPY_session)"
            ),
            "btc_return": (
                "log(BTC close at current SPY cash close / BTC close at the "
                "previous SPY cash close)"
            ),
            "correlation": (
                "post-close updated bivariate Gaussian GARCH(1,1)-DCC(1,1) "
                "conditional correlation, recursively filtered with frozen "
                "2020-2022 parameters and the just-completed paired return"
            ),
            "correlation_change": (
                "post-close correlation_t minus post-close correlation at the "
                "previous SPY session"
            ),
            "event": (
                "absolute correlation change at least 0.02; the paper reports "
                "about 150bp expected next-day BTC return per 0.10 change, so "
                "0.02 implies about 30bp before the frozen 12bp round trip"
            ),
            "side": "-sign(correlation_change); zero cannot signal",
            "volatility_gate": (
                "current post-close updated BTC conditional sigma strict-prior "
                "midrank against at most 90 prior valid SPY-session sigmas, "
                "minimum 60; current excluded; rank at least 0.65"
            ),
            "hold": "24 elapsed hours",
            "reservation": "global half-open; exit first on equal five-minute open",
            "no_imputation": True,
            "elapsed_gap_policy": (
                "paired returns span consecutive official SPY sessions; BTC "
                "therefore spans the same elapsed weekend/holiday gap, which "
                "is reported and never normalized to 24 hours"
            ),
        },
        "policy": {
            "correlation_model": "bivariate_gaussian_garch11_dcc11",
            "correlation_change_abs_min": 0.02,
            "variation_prior_sessions": 90,
            "variation_prior_min_sessions": 60,
            "variation_midrank_min": 0.65,
            "feature_delay_minutes_after_cash_close": 5,
            "entry_delay_minutes_after_feature": 5,
            "hold_hours": 24,
            "leverage": 0.5,
            "base_cost_per_notional_side": 0.0006,
            "stress_cost_per_notional_side": 0.001,
        },
        "estimator": {
            "fit_window": ["2020-01-01", "2023-01-01"],
            "fit_rows": (
                "all complete consecutive official SPY-session paired returns "
                "whose current session is inside the half-open fit window"
            ),
            "mean_equation": (
                "one constant arithmetic sample mean per asset, estimated once "
                "on the fit rows and frozen"
            ),
            "univariate_variance": (
                "h_t=omega+alpha*epsilon_(t-1)^2+beta*h_(t-1); h_0 is the "
                "population variance of fit residuals"
            ),
            "univariate_qml": (
                "minimize sum(log(h_t)+epsilon_t^2/h_t) with scipy "
                "L-BFGS-B; omega=exp(w), alpha and beta are a 0.999-scaled "
                "three-way softmax ensuring omega>0, alpha>=0, beta>=0, "
                "alpha+beta<0.999"
            ),
            "dcc_recursion": (
                "Q_pre_first=Qbar; R_pre_t=diag(Q_pre_t)^(-1/2)Q_pre_t "
                "diag(Q_pre_t)^(-1/2); after observing z_t at the official "
                "close, Q_post_t=(1-a-b)*Qbar+a*z_t*z_t'+b*Q_pre_t; "
                "rho_post_t is read from normalized Q_post_t and the next "
                "session sets Q_pre_(t+1)=Q_post_t"
            ),
            "dcc_qml": (
                "minimize sum(log(det(R_pre_t))+z_t' inv(R_pre_t) z_t) with "
                "scipy L-BFGS-B; a and b use the same 0.999-scaled softmax"
            ),
            "initial_parameters": {
                "univariate_alpha": 0.05,
                "univariate_beta": 0.90,
                "univariate_omega": "0.05 times fit residual population variance",
                "dcc_a": 0.02,
                "dcc_b": 0.95,
            },
            "optimizer": {
                "method": "L-BFGS-B",
                "parameter_bounds": [-20.0, 20.0],
                "ftol": 1e-12,
                "gtol": 1e-8,
                "maxiter": 2000,
                "randomness": False,
            },
            "initial_correlation": (
                "Qbar is the population covariance matrix of the two frozen "
                "fit standardized-residual series; first Q_pre is Qbar"
            ),
            "sigma_definition": (
                "sqrt(h_post_t), where h_post_t=omega+alpha*epsilon_t^2+beta*h_t; "
                "it is available only after the current close"
            ),
            "failure_policy": (
                "any missing row, nonfinite objective/state, non-positive "
                "variance/determinant, optimizer failure, or boundary drift "
                "terminates BSCBR without fallback or repair"
            ),
        },
        "stages": {
            "train": ["2023-07-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "test": ["2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"],
            "eval": ["2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
            "final": ["2026-01-01T00:00:00Z", "2026-08-01T00:00:00Z"],
        },
        "source_support_gates": {
            "minimum_events": {"train": 8, "test": 12, "eval": 12, "final": 8},
            "minority_side_share_min": 0.20,
            "max_month_share": 0.45,
        },
        "novelty_gates": {
            "exact_entry_jaccard_max": 0.10,
            "candidate_near_6h_share_max": 0.35,
            "occupied_5m_jaccard_max": 0.25,
            "absolute_signed_exposure_pearson_max": 0.35,
            "must_pass_before_economics": True,
        },
        "economic_gates": {
            "absolute_return_positive": True,
            "cagr_to_strict_mdd_min": 3.0,
            "strict_mdd_max_pct": 15.0,
            "mean_gross_underlying_min_bp": 20.0,
            "weekly_signflip_one_sided_p_max": 0.10,
            "stress_absolute_return_positive": True,
            "stress_cagr_to_strict_mdd_min": 2.5,
            "each_calendar_half_positive": True,
            "stop_on_first_failure": True,
            "future_can_rank_repair_or_reselect": False,
            "accounting": (
                "fixed quantity, exact funding marks, 6bp base and 10bp stress "
                "per notional side, every held 5m favorable then adverse, "
                "global HWM, full-calendar CAGR"
            ),
        },
        "source_plan": {
            "spy": (
                "candidate-specific hash-bound Yahoo Finance daily chart cache "
                "with explicit dividends/splits; research only; entitled SPY "
                "cash-close feed and parity audit required before live promotion"
            ),
            "btc_completed_hour": (
                "hash-bound PostgreSQL-derived completed-hour materialization "
                "from before 2020 with coverage through 2026-08-01"
            ),
            "session_calendar": (
                "candidate-owned frozen NYSE session schedule for 2020 through "
                "2026-08-01, including holidays, DST, and enumerated early closes"
            ),
            "execution_price": "sealed until source support and Gross9 novelty pass",
        },
        "diagnostic_controls": {
            "names": [
                "no_volatility_gate",
                "correlation_level_instead_of_change",
                "one_session_stale_change",
                "direction_flip",
                "rolling_60_session_pearson",
                "current_cross_product",
                "weekday_and_elapsed_gap",
            ],
            "diagnostic_controls_cannot_be_promoted": True,
        },
        "research_boundary": {
            "paper_direction_read_before_candidate": True,
            "prior_cross_asset_candidate_incidence_known": True,
            "prior_cross_asset_outcomes_used_to_define_bscbr": False,
            "bscbr_candidate_incidence_opened": False,
            "bscbr_post_entry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": (
                "published BTC-stock diversification-demand direction plus the "
                "user-requested high-volatility regime, fixed before incidence"
            ),
        },
        "stopping_rule": (
            "Terminal first-failure sequence: source support, Gross9 novelty, "
            "strict economics; no threshold, side, hold, clock, source proxy, "
            "or subset repair."
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(build(), indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
