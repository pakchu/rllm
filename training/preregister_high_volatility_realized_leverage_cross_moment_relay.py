"""Outcome-blind preregistration for HVRLXC-12."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import preregister_high_volatility_energy_technology_spillover_relay as template


DEFAULT_OUTPUT = Path("results/high_volatility_realized_leverage_cross_moment_relay_preregistration_2026-08-12.json")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def build() -> dict[str, Any]:
    core = copy.deepcopy(template.build())
    core.pop("manifest_hash")
    core.update(
        protocol_version="high_volatility_realized_leverage_cross_moment_relay_v1",
        policy_id="HVRLXC-12",
        as_of_date="2026-08-12",
        mechanism={
            "claim": "Cryptocurrency research documents both conventional negative and inverse positive return-volatility dependence, with diffusive and jump components often carrying opposite signs. On completed five-minute BTC returns, an extreme negative return-to-next-variance cross-moment identifies downside leverage and maps short; an extreme positive cross-moment identifies inverse leverage and maps long, only during elevated causal realized variation.",
            "side": "sign of the normalized completed return-to-next-squared-return cross-moment",
            "external_support": {
                "paper": "Huang, Ni and Xu (2022), Leverage effect in cryptocurrency markets, Pacific-Basin Finance Journal 73, 101773",
                "doi": "10.1016/j.pacfin.2022.101773",
                "reported_fact": "The study reports that cryptocurrency diffusive and jump return-volatility relationships often have opposite signs: conventional negative leverage and positive inverse leverage.",
                "inference_disclosure": "The frozen nonparametric 24-hour cross-moment, magnitude rank, four-hour decision grid, side mapping and 12-hour BTC execution are preregistered adaptations, not replication of the paper's stochastic-volatility model.",
            },
            "why_distinct": "Exact repository scans found no realized-leverage, return-to-next-volatility cross-moment, return-volatility covariance, or signed lead variance candidate. It is not bipower jump variation, semivariance, HAR surprise, realized skew, variance signature, candle direction, flow, funding, OI, premium, prior event set, or promoted control.",
            "why_suited_to_volatile_regimes": "Both the absolute leverage cross-moment and completed 24-hour realized variation must be in high causal ranks, targeting July-like volatile states.",
            "why_low_gross9_overlap_is_plausible": "Sparse four-hour decisions selected by a signed lead return-variance dependence statistic are absent from Gross9 primitives.",
        },
        features={
            "decision_grid": "every exact four-hour UTC boundary T",
            "source_bars": "288 exact coherent BTCUSDT five-minute bars reconstructed from 1,440 one-minute rows in [T-24h,T)",
            "returns": "r_i=log(close_i/open_i) for each completed five-minute bar",
            "leverage_cross_moment": "sum over i=1..287 of r_i*r_(i+1)^2 divided by sqrt(sum r_i^2 * sum r_(i+1)^4); finite nonzero denominator required",
            "magnitude_rank": "strict-prior midrank of absolute leverage_cross_moment versus at most 270 previous valid decisions; minimum 180; current excluded; rank>=0.80",
            "variation": "sqrt(sum over all 288 completed r_i^2)",
            "variation_rank": "strict-prior midrank versus at most 270 previous valid decisions; minimum 180; current excluded; rank>=0.65",
            "side": "positive leverage_cross_moment maps long; negative maps short; exact zero is ineligible",
            "missing": "missing, duplicate, incoherent, nonpositive, nonfinite, zero-denominator or rank-warmup decision is ineligible; no imputation",
        },
        clock={
            "decision": "exact four-hour UTC boundary after all source bars complete",
            "entry": "exact BTCUSDT five-minute open 5 minutes after decision",
            "hold": "12 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding_oi_premium": "not signal inputs; exact funding only after novelty passes",
            "no_imputation": True,
        },
        policy={
            "source_window_hours": 24,
            "source_bar_minutes": 5,
            "decision_interval_hours": 4,
            "magnitude_prior_decisions": 270,
            "magnitude_prior_minimum": 180,
            "magnitude_midrank_min": 0.80,
            "variation_prior_decisions": 270,
            "variation_prior_minimum": 180,
            "variation_midrank_min": 0.65,
            "entry_delay_minutes": 5,
            "hold_hours": 12,
            "gross_exposure": 0.5,
            "base_cost_per_notional_side": 0.0006,
            "stress_cost_per_notional_side": 0.001,
        },
        source_plan={
            "btc_1m": {"table": "bars_binance", "symbol": "BTCUSDT", "interval": "1m", "columns": ["ts", "open", "high", "low", "close"], "window": ["2023-01-01T00:00:00Z", "2026-08-01T00:00:00Z"], "read_after_preregistration": True, "read_only": True},
            "execution_price": "sealed until source support and Gross9 novelty pass",
        },
        diagnostic_controls={
            "names": ["no_variation_gate", "leverage_direction_flip", "one_decision_stale_cross_moment", "contemporaneous_return_variance_moment", "same_clock_forced_long"],
            "diagnostic_controls_cannot_be_promoted": True,
        },
        research_boundary={
            "paper_definition_only_opened": True,
            "candidate_source_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "repository_realized_leverage_candidate_found": False,
            "prior_event_sets_reused": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "published sign-varying cryptocurrency leverage effect, source-only nonparametric estimator, high-variation targeting, and exact repository absence",
        },
        stopping_rule="terminal first-failure sequence: source support, Gross9 novelty, train/test/eval/final strict economics, then RV20 q90 audit; no estimator, normalization, window, grid, rank, threshold, side, hold, clock, subset, source, or control repair",
    )
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core) or value != build():
        raise RuntimeError("HVRLXC preregistration drift")
    if value["outcomes_opened"] is not False or value["source_incidence_opened"] is not False:
        raise RuntimeError("HVRLXC boundary drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
