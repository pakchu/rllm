"""Produce the outcome-blind singleton preregistration for OVEPR-24."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


CANDIDATE = "OVEPR-24"
DEFAULT_OUTPUT = Path(
    "results/options_led_volatility_expansion_premium_relay_"
    "preregistration_2026-08-08.json"
)


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_manifest() -> dict[str, Any]:
    core: dict[str, Any] = {
        "protocol_version": "ovepr_24_preregistration_v1",
        "as_of_date": "2026-08-08",
        "candidate": CANDIDATE,
        "singleton": True,
        "outcomes_opened": False,
        "grid_or_search": False,
        "research_boundary": {
            "disclosure": (
                "Before economic preregistration, outcome-blind source incidence "
                "was inspected in /tmp only; no BTC outcomes were opened."
            ),
            "incidence": {
                "accepted_events": {
                    "train_2023H2": 109,
                    "test_2024": 128,
                    "eval_2025": 66,
                    "final_2026H1": 52,
                },
                "long_short": {
                    "train_2023H2": {"long": 57, "short": 52},
                    "test_2024": {"long": 67, "short": 61},
                    "eval_2025": {"long": 35, "short": 31},
                    "final_2026H1": {"long": 30, "short": 22},
                },
                "maximum_month_share": {
                    "train_2023H2": "201835/1000000",
                    "test_2024": "164063/1000000",
                    "eval_2025": "333333/1000000",
                    "final_2026H1": "384615/1000000",
                },
            },
            "outcome_blind_comparator_probe": {
                "disclosure": (
                    "Before economic preregistration, source-clock overlap was "
                    "inspected without BTC prices, returns, funding, PnL, or Gross9 "
                    "outcomes. Thresholds below bind the observed clock geometry; "
                    "they may not be changed after economic outcomes open."
                ),
                "maximum_observed_prior_family_one_to_one_6h_share": "43/100",
                "maximum_observed_prior_family_occupied_5m_jaccard": "28/100",
            },
            "btc_execution_prices_rows_opened": 0,
            "btc_return_or_pnl_rows_opened": 0,
            "funding_rows_opened": 0,
            "gross9_rows_opened": 0,
            "outcome_files_opened": 0,
            "candidate_count": 1,
            "threshold_direction_latency_hold_search": False,
            "future_cannot_rank_repair_or_substitute": True,
        },
        "source_contract": {
            "join": "exact one-to-one UTC completed-hour inner join",
            "binance_bvol": {
                "path": (
                    "data/binance_btc_bvol_hourly_opdr_2023_2026/"
                    "BTCBVOLUSDT_1h_2023-06-20_2026-06-30.csv.gz"
                ),
                "sha256": (
                    "40c0d1aecb15119e7fab31aae4108c632d25de136401a6896896852c7f4032b1"
                ),
                "completed_clock": "feature_available_time_utc",
                "required_validity": (
                    "source_complete=true AND feature_valid=true AND source_rows=3600"
                ),
            },
            "deribit_dvol": {
                "path": "data/deribit_btc_dvol_1h_2023-06-20_2026-07-01.csv.gz",
                "sha256": (
                    "26b768f81c2fa49fd59d9f1a173a829329a7ed5bb94c2d71af7c33b46f4f02cf"
                ),
                "completed_clock": "close_time",
                "required_end_filter": "close_time < 2026-07-01T00:00:00Z",
            },
            "binance_premium_path": {
                "path": (
                    "data/binance_um_premium_path_btc_2020_2026/"
                    "BTCUSDT_premium_path_1m_2020-01-01_2026-06-30.csv.gz"
                ),
                "sha256": (
                    "7fbaae1f85482b9fc9e148af357c7315e4e7fc4b4e3ae36c31f27545109f8aa9"
                ),
                "hour": "60 exact completed 1m rows in [T-1h,T)",
                "availability": "final source minute's declared availability",
            },
            "causal_availability": (
                "feature_available_time=max(BVOL availability, DVOL availability, "
                "premium aggregate availability); candle-open clocks are forbidden"
            ),
            "gap_duplicate_nearest_fill_or_imputation": "terminal invalid hour",
            "producer_rows_decoded": 0,
            "producer_network_calls": 0,
            "forbidden_clock_fields": [
                "BTC execution price or return",
                "funding",
                "Gross9 state",
                "outcomes or PnL",
            ],
        },
        "mechanism": {
            "name": "Options-led Volatility Expansion Premium Relay",
            "normalized_body": "(close-open)/open using exact decimal arithmetic",
            "primary_setup": [
                "Binance normalized volatility candle body > 0",
                "Deribit normalized volatility candle body > 0",
                "Deribit normalized body > Binance normalized body",
                "current completed-hour premium move != 0",
                (
                    "abs(hour premium move)/sum(60 minute high-low) >= the median "
                    "of strictly prior valid joined hours in [T-720h,T), with at "
                    "least 672 valid hours"
                ),
            ],
            "onset": (
                "emit only a consecutive valid-hour false-to-true transition; "
                "a gap or invalid predecessor cannot emit"
            ),
            "side": "follow premium move: positive=LONG, negative=SHORT",
            "equality_policy": "all strict comparisons fail on equality except efficiency >= median",
            "implementation": (
                "training/options_led_volatility_expansion_premium_relay.py"
            ),
        },
        "execution": {
            "entry": "exactly 5 elapsed minutes after feature_available_time",
            "hold": "24 elapsed hours fixed",
            "reservation": (
                "one global [entry,exit) nonoverlap reservation; suppressed events "
                "are not queued"
            ),
            "leverage": "1/2",
            "base_cost_bp_per_notional_side": 6,
            "stress_cost_bp_per_notional_side": 10,
            "funding": "exact funding cash flows",
            "strict_mdd": (
                "global/pre-entry HWM, entry cost, exact funding boundaries, every "
                "held conservative 5m OHLC path, virtual adverse-mark exit cost, "
                "and actual exit cost"
            ),
            "cagr": "full declared calendar including warm-up and idle time",
        },
        "splits": {
            "train": ["2023-07-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "test": ["2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"],
            "eval": ["2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
            "final": ["2026-01-01T00:00:00Z", "2026-07-01T00:00:00Z"],
        },
        "support_gates": {
            "minimum_events": {"train": 80, "test": 80, "eval": 40, "final": 30},
            "minimum_each_side_share": "3/10",
            "maximum_month_share": {
                "train": "1/4",
                "test": "1/5",
                "eval": "7/20",
                "final": "2/5",
            },
            "operator_policy": {
                "minimums": ">=",
                "maximums": "<=",
            },
            "failure_action": "reject before opening economic outcomes",
        },
        "novelty": {
            "must_complete_before_any_outcome": True,
            "comparators": ["OPDR", "CVVH", "PSR", "PCBR", "CMSR"],
            "gross9": (
                "every sleeve in the then-current sealed Gross9 authority; bind the "
                "authority hash and complete roster without opening outcomes"
            ),
            "common_window": "fully contained rows only; no clip, shift, or split",
            "requirements_each_comparator_and_each_gross9_sleeve": {
                "exact_entry_jaccard_max": "1/10",
                "one_to_one_6h_max_matched_share_max": "9/20",
                "occupied_5m_bar_jaccard_max": "3/10",
                "absolute_signed_exposure_pearson_max": "7/20",
            },
            "all_must_pass": True,
            "failure_action": "reject OVEPR-24 without economic evaluation or repair",
        },
        "controls": {
            "independent_own_clock": {
                "no_deribit_lead": (
                    "both normalized bodies rise; retain nonzero premium move and "
                    "the frozen premium-efficiency gate"
                ),
                "deribit_fall_mirror": (
                    "both normalized bodies fall, abs(DVOL body)>abs(BVOL body), "
                    "retain nonzero premium move and efficiency, side follows premium"
                ),
                "no_premium_efficiency": (
                    "retain primary volatility structure and nonzero premium move"
                ),
            },
            "same_reserved_primary_events": {
                "direction_flip": "multiply primary side by -1",
                "extra_latency_1h": "delay primary entry and exit exactly one hour",
                "deterministic_random_side": (
                    "SHA256('OVEPR-24|decision_time') first-byte parity"
                ),
            },
        },
        "outcome_gate": {
            "sequential_opening": (
                "train_2023H2_then_test_2024_then_eval_2025_then_final_2026H1_"
                "stop_on_first_failure"
            ),
            "requirements_every_opened_split": {
                "absolute_return_positive": True,
                "cagr_to_strict_mdd_min": "3",
                "strict_mdd_max_pct": "15",
                "mean_gross_move_bp_min": "20",
                "clustered_signflip_p_max": "1/10",
                "stress_absolute_return_positive": True,
                "stress_cagr_to_strict_mdd_min": "5/2",
                "each_calendar_half_absolute_return_positive": True,
            },
            "later_failure": (
                "reject; no ranking, threshold, direction, feature, latency, hold, "
                "control, support-gate, or comparator repair"
            ),
        },
        "rllm_boundary": {
            "formulaic_candidate": True,
            "future_llm_may_rank_repair_or_substitute": False,
        },
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate_manifest(manifest: dict[str, Any]) -> None:
    expected = build_manifest()
    if manifest != expected:
        raise ValueError("OVEPR-24 manifest differs from the frozen singleton")
    core = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    if manifest["manifest_hash"] != canonical_hash(core):
        raise ValueError("OVEPR-24 manifest hash mismatch")


def write_manifest(path: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    report = build_manifest()
    validate_manifest(report)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    report = write_manifest(args.output)
    print(json.dumps({"output": args.output, "manifest_hash": report["manifest_hash"]}))


if __name__ == "__main__":
    main()
