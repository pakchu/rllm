"""Outcome-blind preregistration for HVKSRR-24."""
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


DEFAULT_OUTPUT = Path("results/high_volatility_kalshi_sp500_range_repricing_relay_preregistration_2026-08-12.json")
API_BASE = "https://external-api.kalshi.com/trade-api/v2"


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def build() -> dict[str, Any]:
    core = copy.deepcopy(template.build())
    core.pop("manifest_hash")
    core.update(
        protocol_version="high_volatility_kalshi_sp500_range_repricing_relay_v1",
        policy_id="HVKSRR-24",
        as_of_date="2026-08-12",
        mechanism={
            "claim": "Prices across Kalshi's mutually exclusive daily S&P 500 closing-range contracts aggregate traders' changing beliefs about the same-day U.S. equity close. During elevated causal BTC variation, follow the direction of the normalized probability-weighted ladder-rank change from three hours to two hours before the official strike time for one day.",
            "side": "strictly positive normalized implied-rank change maps long BTC; strictly negative maps short; exact zero is ineligible",
            "external_support": {
                "prediction_market_study": "Prediction Markets, Journal of Economic Perspectives 2004",
                "study_url": "https://www.aeaweb.org/articles?id=10.1257/0895330041371321",
                "reported_fact": "The cited review describes prediction-market prices as aggregating dispersed information and often producing useful forecasts.",
                "inference_disclosure": "Transmitting an S&P closing-distribution repricing direction to next-day BTC direction is a preregistered cross-asset risk-appetite inference, not a direct estimate reported by that study.",
            },
            "why_distinct": "Exact repository scans found no Kalshi, prediction-market, or event-contract price source. The signal uses no Binance direction, flow, funding, OI, premium, conventional S&P cash/futures return, macro release value, lottery sale, DOI count, prior event set, or promoted control.",
            "why_suited_to_volatile_regimes": "Only decisions whose completed prior-24-hour BTC variation ranks in the upper 35% are admitted.",
            "why_low_gross9_overlap_is_plausible": "An external regulated event-contract probability ladder and its official equity-close clock are absent from Gross9 primitives.",
        },
        features={
            "authority": "Kalshi public Trade API events, nested markets, and event candlesticks",
            "series_lineage": "query series_ticker=KXINX; retain only official event rows whose own series_ticker and event-ticker prefix are exactly KXINX or legacy INX",
            "eligible_event": "finite official strike_date in [2022-01-01,2026-08-01), mutually_exclusive true, one less tail, one greater tail, and at least three between markets forming one strictly ordered contiguous non-overlapping S&P range ladder; duplicate event or market ticker rejects",
            "quote_state": "for every ladder market, carry only that market's latest causally completed 60-minute candle ending no later than the anchor; both yes-bid and yes-ask closes must be present, finite, 0<=bid<=ask<=1; no trade-price fallback",
            "implied_rank": "midpoint each yes quote, normalize all positive midpoint mass to one, multiply by fixed ascending ladder rank 0..N-1, sum, and divide by N-1",
            "repricing": "implied_rank at strike_date-2h minus implied_rank at strike_date-3h; both anchors require the identical complete ladder; strict nonzero",
            "side": "strict sign of repricing",
            "btc_variation": "sqrt(sum squared log(close/open)) over 1,440 exact BTCUSDT 1m bars in [decision-24h,decision)",
            "btc_variation_rank": "strict-prior midrank versus at most 270 prior source-valid decisions; minimum 180; current excluded; rank >=0.65",
            "missing": "HTTP/cursor/schema drift, incomplete pagination, malformed ladder, missing quote side, nonpositive quote mass, or missing/duplicate/nonpositive BTC bars rejects as frozen; no imputation",
        },
        clock={
            "decision": "official event strike_date minus exactly two elapsed hours, after the corresponding completed 60-minute candle",
            "entry": "exact BTCUSDT five-minute open 5 minutes after decision",
            "hold": "24 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding_oi_premium": "not signal inputs; exact funding only after novelty passes",
            "no_imputation": True,
        },
        policy={
            "query_series_ticker": "KXINX",
            "accepted_event_series": ["INX", "KXINX"],
            "accepted_event_prefixes": ["INX-", "KXINX-"],
            "anchor_hours_before_strike": [3, 2],
            "candlestick_minutes": 60,
            "minimum_between_markets": 3,
            "variation_prior_days": 270,
            "variation_prior_minimum": 180,
            "variation_midrank_min": 0.65,
            "entry_delay_minutes": 5,
            "hold_hours": 24,
            "gross_exposure": 0.5,
            "base_cost_per_notional_side": 0.0006,
            "stress_cost_per_notional_side": 0.001,
        },
        source_plan={
            "kalshi": {
                "api_base": API_BASE,
                "events_endpoint": "/events?series_ticker=KXINX&limit=200",
                "event_endpoint": "/events/{event_ticker}?with_nested_markets=true",
                "candles_endpoint": "/series/{series_ticker}/events/{event_ticker}/candlesticks",
                "event_start": "2022-01-01T00:00:00Z",
                "event_end_exclusive": "2026-08-01T00:00:00Z",
                "cursor_to_exhaustion": True,
                "download_after_preregistration": True,
                "read_only_snapshot": True,
            },
            "btc_1m": {"table": "bars_binance", "symbol": "BTCUSDT", "interval": "1m", "columns": ["ts", "open", "close"], "read_only": True},
            "execution_price": "sealed until source support and Gross9 novelty pass",
        },
        diagnostic_controls={
            "names": ["no_btc_volatility_gate", "repricing_direction_flip", "one_event_stale_repricing", "one_hour_anchor_shift", "same_clock_forced_long"],
            "diagnostic_controls_cannot_be_promoted": True,
        },
        research_boundary={
            "excluded_post_final_feasibility_probe": "2026-08-11 KXINX event metadata, ladder schema, and candle schema only",
            "event_lineage_metadata_probe": "event tickers and count envelope only; no retained market price, BTC price, return, funding, or Gross9 row",
            "probe_not_used_to_choose_side_hold_rank_or_source_gate": True,
            "full_historical_candidate_incidence_opened": False,
            "source_values_used_to_fit_rule": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "repository_kalshi_candidate_found": False,
            "cross_asset_inference_disclosed": True,
            "prior_event_sets_reused": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "official public event-contract history, immutable event strike clocks, published prediction-market information aggregation, dense daily lineage, and exact repository absence",
        },
        stopping_rule="terminal first-failure sequence: source contract/support, Gross9 novelty, train/test/eval/final strict economics, then RV20 q90 audit; no series, event lineage, ladder, quote rule, anchor, variation threshold, side, hold, clock, subset, source, or control repair",
    )
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core) or value != build():
        raise RuntimeError("HVKSRR preregistration drift")
    if value["outcomes_opened"] is not False or value["source_incidence_opened"] is not False:
        raise RuntimeError("HVKSRR boundary drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
