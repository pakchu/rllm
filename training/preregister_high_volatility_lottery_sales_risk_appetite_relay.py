"""Outcome-blind preregistration for HVLSRA-24."""
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


DEFAULT_OUTPUT = Path("results/high_volatility_lottery_sales_risk_appetite_relay_preregistration_2026-08-12.json")
REPORT_URL = "https://www.texaslottery.com/export/sites/lottery/Documents/Draw_Sales/{date}_pb.txt"


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def build() -> dict[str, Any]:
    core = copy.deepcopy(template.build())
    core.pop("manifest_hash")
    core.update(
        protocol_version="high_volatility_lottery_sales_risk_appetite_relay_v1",
        policy_id="HVLSRA-24",
        as_of_date="2026-08-12",
        mechanism={
            "claim": "Powerball ticket sales are a directly observed, externally timed measure of household demand for positively skewed gambles. Published crypto research finds lottery-like preferences and gambling demand in cryptocurrency markets. During elevated causal BTC variation, follow the sign of the completed Texas Powerball draw-to-draw net-sales change for one day.",
            "side": "strictly higher completed Texas Powerball net sales than the immediately prior scheduled draw maps long; strictly lower maps short",
            "external_support": {
                "bitcoin_gambling_study": "Betting on Bitcoin: Does gambling volume on the blockchain explain Bitcoin price changes?, Economics Letters 191 (2020)",
                "bitcoin_gambling_study_url": "https://doi.org/10.1016/j.econlet.2019.108727",
                "crypto_retail_study": "Gambling on Crypto Tokens?, Journal of Financial and Quantitative Analysis (2025)",
                "crypto_retail_study_url": "https://doi.org/10.1017/S0022109024000681",
                "reported_fact": "The cited literature documents lottery-like preferences in crypto and that gambling preferences strongly predict retail investor interest in crypto.",
                "inference_disclosure": "The sign transmission from an external state-lottery sales change to broad BTC direction is a preregistered behavioral-risk-appetite inference, not a claimed causal estimate from the cited papers.",
            },
            "why_distinct": "Exact repository scans found no Powerball, Mega Millions, lottery-sales, jackpot, or gambling-sentiment trading clock. The signal uses no market direction, flow, funding, OI, premium, macro release, news text, physical hazard, sports result, prior event set, or promoted control.",
            "why_suited_to_volatile_regimes": "Only draw reports whose completed prior-24-hour BTC variation ranks in the upper 35% are admitted.",
            "why_low_gross9_overlap_is_plausible": "Three fixed weekly external lottery-report timestamps and source-derived signs are absent from Gross9 primitives.",
        },
        features={
            "source": "official Texas Lottery Powerball Winner Summary reports",
            "eligible_draw": "every scheduled Monday, Wednesday, or Saturday Powerball draw whose official text report has an exact draw date, report-generation timestamp, and finite positive NET SALES",
            "sales_change": "log(current NET SALES / immediately prior scheduled draw NET SALES), strict nonzero; no weekday adjustment, jackpot input, seasonal adjustment, or imputation",
            "side": "strict sign of sales_change",
            "availability": "12:00 UTC on the calendar day after the official draw, later than the report-generation timestamps observed by the transport-only feasibility probe",
            "btc_variation": "sqrt(sum squared log(close/open)) over 1,440 exact BTCUSDT 1m bars in [decision-24h,decision)",
            "btc_variation_rank": "strict-prior midrank versus at most 180 prior source-valid decisions; minimum 60; current excluded; rank >=0.65",
            "missing": "HTTP/schema drift, wrong draw date, report timestamp after the frozen availability boundary, duplicate draw, missing/nonpositive sales, nonconsecutive scheduled draw, or missing/duplicate/nonpositive BTC bars makes the decision ineligible or rejects as frozen; no imputation",
        },
        clock={
            "decision": "12:00 UTC on the calendar day after each scheduled Powerball draw",
            "entry": "exact BTCUSDT five-minute open 5 minutes after decision",
            "hold": "24 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding_oi_premium": "not signal inputs; exact funding only after novelty passes",
            "no_imputation": True,
        },
        policy={
            "draw_weekdays": ["Monday", "Wednesday", "Saturday"],
            "availability_utc_hour": 12,
            "variation_prior_events": 180,
            "variation_prior_minimum": 60,
            "variation_midrank_min": 0.65,
            "entry_delay_minutes": 5,
            "hold_hours": 24,
            "gross_exposure": 0.5,
            "base_cost_per_notional_side": 0.0006,
            "stress_cost_per_notional_side": 0.001,
        },
        source_plan={
            "lottery_reports": {
                "url_template": REPORT_URL,
                "first_warmup_draw": "2022-01-01",
                "last_draw": "2026-07-29",
                "download_after_preregistration": True,
                "read_only_snapshot": True,
            },
            "btc_1m": {"table": "bars_binance", "symbol": "BTCUSDT", "interval": "1m", "columns": ["ts", "open", "close"], "read_only": True},
            "execution_price": "sealed until source support and Gross9 novelty pass",
        },
        diagnostic_controls={
            "names": ["no_btc_volatility_gate", "sales_direction_flip", "one_draw_stale_sales_change", "weekday_balanced_change", "same_clock_forced_long"],
            "diagnostic_controls_cannot_be_promoted": True,
        },
        research_boundary={
            "transport_only_probe_dates": ["2023-07-03", "2023-07-05", "2023-07-08", "2024-01-01", "2025-01-01", "2026-01-03"],
            "probe_opened_only_http_success_size_and_static_header_prefix": True,
            "numeric_sales_or_candidate_incidence_opened": False,
            "source_values_used_to_select_rule": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "repository_lottery_candidate_found": False,
            "cross_asset_inference_disclosed": True,
            "prior_event_sets_reused": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "official high-frequency draw-sales reports, published crypto gambling-preference evidence, exact causal delay, and repository absence",
        },
        stopping_rule="terminal first-failure sequence: source contract/support, Gross9 novelty, train/test/eval/final strict economics, then RV20 q90 audit; no lottery, report field, delay, variation threshold, side, hold, clock, subset, source, weekday adjustment, jackpot control, or comparator repair",
    )
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core) or value != build():
        raise RuntimeError("HVLSRA preregistration drift")
    if value["outcomes_opened"] is not False or value["source_incidence_opened"] is not False:
        raise RuntimeError("HVLSRA boundary drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
