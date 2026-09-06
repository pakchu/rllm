"""Outcome-blind preregistration for HVKAR-12."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


POLICY_ID = "HVKAR-12"
DEFAULT_OUTPUT = Path(
    "results/high_volatility_knn_analog_relay_preregistration_2026-08-09.json"
)
SOURCE_PATH = Path("data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz")
SOURCE_SHA256 = "a77cd0ae5b88b3c95e509d8d2610773d34af3afdc9170c63d88564bc3d0b990c"


def canonical_hash(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_knn_analog_relay_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-09",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": (
                "At a daily boundary in an RV20 top-decile regime, the nearest prior "
                "microstructure states provide a non-parametric estimate of whether urgent "
                "inventory is more likely to persist or mean-revert over the next twelve hours."
            ),
            "side": "sign of the median net twelve-hour return among 21 nearest fit analogs",
            "why_distinct": (
                "HVKAR is a once-daily, fit-period-only analog retrieval policy. It is not a "
                "threshold repair, control promotion, tree/linear classifier, fixed momentum "
                "rule, external release clock, or any Gross9 state-conditioned policy."
            ),
            "why_suited_to_volatile_regimes": (
                "eligibility uses the exact causal RV20 rolling-90th-percentile contract"
            ),
            "why_low_gross9_overlap_is_plausible": (
                "one 00:05 UTC decision per eligible calendar day is structurally sparse"
            ),
        },
        "rv20": {
            "reference": "BTCUSDT perpetual exact UTC calendar-day 00:00 close series",
            "daily_return": "log(C_d/C_{d-1}) assigned to day d",
            "feature": "sqrt(365*mean(r_d^2)) for exact days t-20 through t-1",
            "threshold": (
                "numpy linear 0.90 quantile of the preceding 756 available RV20 observations "
                "with timestamps strictly before t; current excluded"
            ),
            "eligibility": "RV20(t)>=threshold(t)",
            "missing_or_duplicate_day": "ineligible; no imputation",
        },
        "features": {
            "window": "exact 288 completed BTCUSDT 5m bars in [t-24h,t)",
            "vector": [
                "log close/open return",
                "sqrt sum squared close-to-close 5m returns",
                "log maximum high/minimum low",
                "upside minus downside semivariance divided by total semivariance",
                "2*taker_buy_quote/sum_quote_volume-1",
                "log final close/quote-volume-weighted typical price",
                "signed-return entropy over eight equal-width sign-count bins",
                "log total quote volume",
                "log total trade count",
            ],
            "finite_complete_path_required": True,
            "normalization": (
                "componentwise median and max(IQR,1e-12) from fit rows only; frozen after fit"
            ),
        },
        "model": {
            "family": "deterministic Euclidean k-nearest analog regression",
            "fit_decisions": ["2021-01-01T00:00:00Z", "2023-07-01T00:00:00Z"],
            "label": (
                "fixed-quantity 0.5-gross twelve-hour return from 00:05 open to 12:05 open, "
                "including 6bp per-notional-side costs and exact funding; labels whose exit is "
                "not strictly before fit end are excluded"
            ),
            "neighbors": 21,
            "distance": "Euclidean after frozen robust normalization",
            "tie_break": "distance, then decision timestamp ascending",
            "prediction": "median neighbor label; zero abstains",
            "training_labels_cannot_enter_source_support_report": True,
            "refit_after_2023_07_01": False,
        },
        "clock": {
            "decision": "each exact 00:00 UTC after all prior-day bars complete",
            "entry": "exact 00:05 UTC BTCUSDT 5m open",
            "hold": "12 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding": "not a signal input; exact realized funding only",
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
            "occupied_5m_bar_jaccard_max": 0.25,
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
            "accounting": (
                "fixed quantity, exact funding, 6bp base and 10bp stress per notional side, "
                "every held 5m favorable then adverse, global HWM, full-calendar CAGR"
            ),
        },
        "source_plan": {
            "historical": {"path": str(SOURCE_PATH), "sha256": SOURCE_SHA256},
            "live_extension": (
                "read-only Postgres bars_binance BTCUSDT 5m through 2026-08-01; exact overlap "
                "parity required before append"
            ),
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        "research_boundary": {
            "prior_knn_analog_candidate_outcomes_known": False,
            "candidate_incidence_opened": False,
            "postentry_stage_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
        },
        "stopping_rule": (
            "terminal first failure; no feature, k, fit period, RV20, side, timing, hold, "
            "threshold, subset, or gate repair"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("HVKAR preregistration hash mismatch")
    if hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("HVKAR historical source drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(args.output)
