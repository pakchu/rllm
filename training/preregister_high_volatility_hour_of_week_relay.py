"""Outcome-sequenced preregistration for HVHOW-6."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path(
    "results/high_volatility_hour_of_week_relay_preregistration_2026-08-09.json"
)
SOURCE_BINDINGS = {
    "preprocessing/market_features.py": "f9091ecb080656c69a08ac3b4d07f7316cc2ddcc1fe4efacb9e10e8334d5cafa",
    "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz": (
        "a77cd0ae5b88b3c95e509d8d2610773d34af3afdc9170c63d88564bc3d0b990c"
    ),
}


def canonical_hash(payload: Any) -> str:
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_hour_of_week_relay_v1",
        "policy_id": "HVHOW-6",
        "as_of_date": "2026-08-09",
        "oos_outcomes_opened": False,
        "oos_source_incidence_opened": False,
        "pretraining_outcomes_authorized_after_preregistration": True,
        "singleton": True,
        "mechanism": {
            "claim": (
                "Crypto risk transfer has persistent hour-of-week structure. The two strongest "
                "positive and two strongest negative six-hour slots learned only from pre-2023 "
                "high-volatility observations should retain direction when volatility is elevated."
            ),
            "side": "frozen sign of the selected pre-2023 hour-of-week mean return",
            "why_distinct": (
                "HVHOW is a four-slot weekly seasonality model. It uses no event threshold, price-path "
                "direction, cross-asset feed, funding settlement, weekend-only rule, or prior candidate "
                "control. The slot model is frozen before any 2023H2+ incidence is opened."
            ),
            "why_low_gross9_overlap_is_plausible": (
                "only four of 168 weekly slots are eligible and volatility further thins incidence"
            ),
        },
        "training_contract": {
            "fit_window": ["2020-01-01T00:00:00Z", "2023-01-01T00:00:00Z"],
            "calibration_window": ["2023-01-01T00:00:00Z", "2023-07-01T00:00:00Z"],
            "anchors": "each completed xx:55 UTC 5m bar; entry is the next exact hour open",
            "label": "log(open at entry+72 bars/open at entry), six elapsed hours",
            "fit_volatility": "range_vol >= fit-window q60 computed source-only",
            "slot": "entry UTC weekday*24+hour, integer 0..167",
            "slot_floor": "at least 100 complete high-volatility fit labels per slot",
            "slot_score": "arithmetic mean of complete fit log-return labels",
            "selection": (
                "two largest positive means and two smallest negative means; deterministic score then "
                "slot-id tie break; failure if either sign has fewer than two eligible slots"
            ),
            "oos_volatility": "2023H1 source-only range_vol q60",
            "grid": False,
            "refit_after_2022_12_31": False,
            "model_artifact_must_be_frozen_before_oos_incidence": True,
        },
        "oos_clock": {
            "decision": "each completed xx:55 UTC bar from 2023-07-01 onward",
            "eligibility": "entry hour-of-week is one of four frozen slots and range_vol>=frozen q60",
            "entry": "next exact hour 5m open",
            "hold": "6 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
        },
        "stages": {
            "train": ["2023-07-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "test": ["2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"],
            "eval": ["2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
            "final": ["2026-01-01T00:00:00Z", "2026-08-01T00:00:00Z"],
        },
        "source_support_gates": {
            "minimum_events": {"train": 8, "test": 12, "eval": 12, "final": 8},
            "minority_side_share_min": 0.2,
            "max_month_share": 0.45,
        },
        "novelty_gates": {
            "exact_entry_jaccard_max": 0.1,
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
            "weekly_signflip_one_sided_p_max": 0.1,
            "stress_absolute_return_positive": True,
            "stress_cagr_to_strict_mdd_min": 2.5,
            "each_calendar_half_positive": True,
            "stop_on_first_failure": True,
            "accounting": (
                "fixed quantity, exact funding marks, 6bp base and 10bp stress per notional side, "
                "every held 5m favorable then adverse, global HWM, full-calendar CAGR"
            ),
        },
        "diagnostic_controls": {
            "names": [
                "no_volatility_gate",
                "all_selected_slots_constant_long",
                "one_hour_stale_slot",
                "direction_flip",
            ],
            "diagnostic_controls_cannot_be_promoted": True,
        },
        "source_plan": {
            "historical_market": "hash-bound 5m cache through 2026-06-01",
            "live_extension": "read-only Postgres completed bars through 2026-08-01",
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        "source_bindings": SOURCE_BINDINGS,
        "research_boundary": {
            "prior_calendar_candidate_outcomes_known": True,
            "exact_four_slot_hour_of_week_outcomes_known": False,
            "oos_candidate_incidence_opened": False,
            "oos_post_entry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
        },
        "stopping_rule": (
            "freeze preregistration, fit only pre-2023, calibrate source-only 2023H1, freeze model, "
            "then open OOS incidence; terminal first failure with no slot, sign, hold, or gate repair"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("HVHOW preregistration hash mismatch")
    for raw, expected in SOURCE_BINDINGS.items():
        if hashlib.sha256(Path(raw).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"HVHOW source drift: {raw}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build(); validate(result)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
