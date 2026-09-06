"""Outcome-blind preregistration for HVUSDTJ-24."""
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


POLICY_ID = "HVUSDTJ-24"
DEFAULT_OUTPUT = Path(
    "results/high_volatility_tether_jump_spillover_relay_preregistration_2026-08-13.json"
)
ENDPOINT = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
METRICS = ("PriceUSD", "AssetEODCompletionTime")


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
    core = copy.deepcopy(template.build())
    core.pop("manifest_hash")
    core.update(
        protocol_version="high_volatility_tether_jump_spillover_relay_v1",
        policy_id=POLICY_ID,
        as_of_date="2026-08-13",
        singleton=True,
        outcomes_opened=False,
        source_incidence_opened=False,
        gross9_rows_opened=False,
        mechanism={
            "claim": (
                "An extreme completed daily Tether return reveals a discontinuous repricing of crypto's dollar "
                "settlement asset. Peer-reviewed evidence finds positive Tether jumps combined with positive "
                "Tether returns predict subsequent Bitcoin declines. During elevated causal BTC variation, map "
                "the signed stablecoin jump contrarian into Bitcoin for one day."
            ),
            "side": "positive USDT log return maps BTC short; negative USDT log return maps BTC long; exact zero rejects",
            "external_support": {
                "paper": "Grobys and Huynh (2022), When Tether says JUMP! Bitcoin asks How low?, Finance Research Letters 47, 102644",
                "doi": "10.1016/j.frl.2021.102644",
                "reported_fact": (
                    "Using hourly Bitfinex data and bipower-variation jump classification, the paper reports that "
                    "positive Tether jumps interacted with a one-percent positive prior-day Tether return predict "
                    "subsequent daily Bitcoin price changes of approximately -3.65 to -8.49 percent."
                ),
                "primary_open_access_source": "https://osuva.uwasa.fi/handle/10024/14961",
                "inference_disclosure": (
                    "The paper identifies the positive-USDT-return/negative-BTC branch. Extending the mechanism "
                    "symmetrically so negative extreme USDT returns map BTC long, replacing intraday bipower jumps "
                    "with a strict-prior daily absolute-return tail, using Coin Metrics PriceUSD, a conservative "
                    "completion clock, Binance execution, a BTC variation gate, and a 24-hour hold are frozen "
                    "adaptations rather than a replication."
                ),
            },
            "why_distinct": (
                "Repository-wide scans found no USDT PriceUSD, Tether return-jump, stablecoin depeg-return, or "
                "stablecoin-price-to-BTC candidate. Prior stablecoin candidates concern issuance, WBTC liquidity, "
                "or funding/basis objects, not the realized price of the dollar settlement asset."
            ),
            "why_suited_to_volatile_regimes": (
                "Stablecoin price jumps are crypto stress-transfer events, and the frozen rule additionally admits "
                "positions only when completed BTC variation ranks in its causal upper thirty-five percent."
            ),
            "why_low_gross9_overlap_is_plausible": (
                "Asset-specific Coin Metrics completion timestamps and a non-BTC stablecoin jump object are absent "
                "from Gross9 market clocks."
            ),
        },
        features={
            "source_day": "Coin Metrics USDT UTC daily observation D at the frozen current download vintage",
            "usdt_price": "exact finite positive PriceUSD on consecutive source days D and D-1",
            "usdt_log_return": "log(PriceUSD_D / PriceUSD_D-1); exact zero rejects",
            "availability": (
                "D is eligible only if AssetEODCompletionTime is finite, strictly after D+1 00:00 UTC, and no "
                "later than D+2 00:00 UTC; D-1 must be an exact consecutive valid row; late rows reject"
            ),
            "jump_rank": (
                "strict-prior midrank of abs(usdt_log_return) against at most 270 prior source-valid returns, "
                "minimum 180, current excluded; rank>=0.75"
            ),
            "btc_variation": (
                "sqrt(sum squared one-minute log(close/open) returns) over 1,440 exact BTCUSDT rows in "
                "[decision-24h,decision)"
            ),
            "btc_variation_rank": (
                "strict-prior midrank against at most 270 prior source-valid decisions, minimum 180, current "
                "excluded; rank>=0.65"
            ),
            "eligibility": "valid nonzero USDT return, jump rank>=0.75, and BTC variation rank>=0.65",
            "missing": (
                "missing, duplicate, nonpositive, nonconsecutive, completion-before-boundary, completion after "
                "D+2, rank-history, or BTC-grid drift rejects; no imputation"
            ),
        },
        clock={
            "decision": "ceiling to the next exact five-minute boundary at or after AssetEODCompletionTime for D",
            "entry": "exact BTCUSDT five-minute open one five-minute bar after decision",
            "hold": "24 elapsed hours",
            "reservation": "global chronological half-open; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding_oi_premium": "not signal inputs; exact funding only after novelty passes",
        },
        policy={
            "jump_prior_days": 270,
            "jump_prior_minimum": 180,
            "jump_midrank_min": 0.75,
            "variation_prior_days": 270,
            "variation_prior_minimum": 180,
            "variation_midrank_min": 0.65,
            "latest_completion_delay_days": 2,
            "entry_delay_minutes_after_decision": 5,
            "hold_hours": 24,
            "gross_exposure": 0.5,
            "base_cost_per_notional_side": 0.0006,
            "stress_cost_per_notional_side": 0.001,
        },
        source_plan={
            "coin_metrics": {
                "endpoint": ENDPOINT,
                "asset": "usdt",
                "metrics": list(METRICS),
                "frequency": "1d",
                "start_time": "2022-01-01",
                "end_time_inclusive": "2026-07-29",
                "page_size": 10000,
                "current_vintage_not_historical_revision_archive": True,
                "catalog_metadata_opened_before_preregistration": True,
                "historical_values_opened_before_preregistration": False,
                "read_after_preregistration_commit": True,
                "read_only": True,
            },
            "btc_1m": {
                "table": "bars_binance",
                "symbol": "BTCUSDT",
                "interval": "1m",
                "columns": ["ts", "open", "close"],
                "read_only": True,
            },
            "execution_price": "sealed until source support and Gross9 novelty pass",
        },
        diagnostic_controls={
            "names": [
                "no_jump_tail",
                "no_btc_variation_gate",
                "one_day_stale_usdt_return",
                "direction_flip",
                "same_clock_forced_long",
            ],
            "diagnostic_controls_cannot_be_promoted": True,
        },
        research_boundary={
            "paper_and_coin_metrics_catalog_metadata_opened": True,
            "usdt_historical_source_values_opened": False,
            "source_values_used_to_select_rule_or_threshold": False,
            "candidate_source_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "repository_usdt_price_jump_candidate_found": False,
            "prior_stablecoin_issuance_or_flow_event_sets_reused": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": (
                "peer-reviewed signed Tether-jump spillover, unopened free causal source values, explicit "
                "high-variation targeting, and repository formula absence"
            ),
        },
        stopping_rule=(
            "terminal first-failure sequence: source support, Gross9 novelty, train/test/eval/final strict "
            "economics, then RV20 q90 audit; no asset, metric, vintage, return, rank, threshold, variation, side, "
            "hold, clock, subset, source, comparator, or control repair"
        ),
    )
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value != build() or value.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("HVUSDTJ preregistration drift")
    if value["outcomes_opened"] or value["source_incidence_opened"] or value["gross9_rows_opened"]:
        raise RuntimeError("HVUSDTJ research boundary drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
