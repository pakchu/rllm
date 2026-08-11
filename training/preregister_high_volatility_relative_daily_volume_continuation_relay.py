"""Outcome-blind preregistration for HVRDV-8."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from training import preregister_high_volatility_ticket_elasticity_sponsorship_relay as template

POLICY_ID = "HVRDV-8"
DEFAULT_OUTPUT = Path(
    "results/high_volatility_relative_daily_volume_continuation_relay_"
    "preregistration_2026-08-11.json"
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    contract = copy.deepcopy(template.build())
    contract.pop("manifest_hash")
    contract.update(
        protocol_version="high_volatility_relative_daily_volume_continuation_relay_v1",
        policy_id=POLICY_ID,
        as_of_date="2026-08-11",
        mechanism={
            "claim": "During elevated completed realized variation, a causal Relative Daily Volume move from at-or-below its same-time historical average to above that average identifies newly exceptional participation. Follow the sign of the completed UTC-day displacement for eight hours.",
            "side": "long when the completed UTC-day open-to-current close return is positive; short when it is negative",
            "why_distinct": "Ordinary rolling relative-volume z-scores compare adjacent bars without aligning cumulative volume by UTC time of day. HVRDV compares today's cumulative base volume with the trailing twenty completed days' cumulative base volume at the identical elapsed UTC-day slot, then requires an average-crossing onset. It uses no taker side, quote turnover, volume-price covariance, funding, OI, fitted outcome, reused event set, or promoted control.",
            "why_suited_to_volatile_regimes": "participation onset is admitted only when the completed trailing twenty-four-hour realized variation ranks in its causal upper 35%",
            "why_low_gross9_overlap_is_plausible": "same-time cumulative-volume onsets on a four-hour UTC grid are absent from Gross9 primitives",
        },
        external_basis={
            "origin": "QuantConnect LEAN RelativeDailyVolume canonical implementation",
            "definition_source": "https://github.com/QuantConnect/Lean/blob/master/Indicators/RelativeDailyVolume.cs",
            "definition_source_sha256": "9f030d922396e476bb43fbc1952dadfb93bd55024758d0ec7ec431eb293d4ced",
            "documentation_source": "https://www.quantconnect.com/docs/v2/writing-algorithms/indicators/supported-indicators/relative-daily-volume",
            "fixed_definition": "current cumulative base volume from UTC-day open through the completed slot divided by the simple average cumulative base volume at the identical slot over the previous twenty completed valid UTC days",
            "selection_use": "official cumulative same-time ratio definition; the economically neutral ratio 1.0 denotes the historical average; direction and volatile-regime application are explicitly untested adaptations; no incidence or outcomes",
        },
        features={
            "decision_grid": "every exact 04:00, 08:00, 12:00, 16:00, 20:00, and 00:00 UTC after a completed four-hour slot",
            "source_day": "UTC day containing the completed slot; the 00:00 decision belongs to the just-completed prior UTC day",
            "daily_cumulative_base_volume": "sum of finite nonnegative Binance BTCUSDT base volume from source-day 00:00 through the completed slot",
            "historical_same_slot_average": "simple mean of cumulative base volume through the identical elapsed slot over exactly the latest twenty earlier valid UTC days; current day excluded",
            "relative_daily_volume": "daily_cumulative_base_volume divided by historical_same_slot_average, finite strict positive denominator",
            "onset": "current RDV strictly greater than 1.0 and the immediately previous completed four-hour slot of the same source day has finite RDV at or below 1.0; before 04:00 the frozen prior state is 0.0",
            "direction": "strict sign of log(current completed slot close/source-day first one-minute open)",
            "realized_variation": "sqrt(sum squared one-minute log(close/open) returns over the trailing completed twenty-four hours), finite strict positive",
            "variation_rank": "strict-prior midrank over at most 270 earlier source-valid decision slots, minimum 180, current excluded; rank>=0.65",
            "recursive_reset": "same-slot history and variation-rank history reset after any invalid UTC source day; no value crosses a gap",
            "no_imputation": True,
        },
        clock={
            "feature_available": "exact completed four-hour UTC boundary",
            "entry": "exact BTCUSDT boundary+5m open",
            "hold": "8 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "gross_exposure": 0.5,
            "funding": "not a signal input; exact settlements only after novelty passes",
        },
        policy={
            "relative_daily_volume_days": 20,
            "relative_daily_volume_average_level": 1.0,
            "variation_history_slots": 270,
            "minimum_variation_history_slots": 180,
            "variation_rank_min": 0.65,
            "entry_delay_minutes": 5,
            "hold_hours": 8,
            "leverage": 0.5,
            "base_cost_per_notional_side": 0.0006,
            "stress_cost_per_notional_side": 0.001,
        },
        diagnostic_controls={
            "names": [
                "no_variation_gate",
                "no_rdv_onset_gate",
                "quote_volume_rdv",
                "one_slot_stale_onset",
                "direction_flip",
                "forced_long",
            ],
            "cannot_be_promoted": True,
        },
        source_plan={
            "bars": {
                "table": "bars_binance",
                "symbol": "BTCUSDT",
                "interval": "1m",
                "columns": [
                    "ts",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "quote_asset_volume",
                ],
                "window": [
                    "2023-01-01T00:00:00Z",
                    "2026-08-01T00:00:00Z",
                ],
                "read_after_preregistration": True,
            },
            "execution_price": "sealed until source support and Gross9 novelty pass",
        },
        research_boundary={
            "canonical_relative_daily_volume_definition_read": True,
            "repository_same_time_cumulative_rdv_candidate_found": False,
            "prior_rolling_relative_volume_features_known": True,
            "prior_event_sets_reused": False,
            "prior_candidate_outcomes_used_to_set_formula_side_hold_or_clock": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "canonical same-time cumulative Relative Daily Volume average crossing under the requested high-variation regime",
        },
        stopping_rule="Terminal first failure; no RDV period, average level, onset, variation, direction, hold, clock, subset, threshold, control, or other repair.",
    )
    return {**contract, "manifest_hash": canonical_hash(contract)}


def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core) or value != build():
        raise RuntimeError("HVRDV-8 preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    registration = build()
    validate(registration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(registration, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    print(args.output)
