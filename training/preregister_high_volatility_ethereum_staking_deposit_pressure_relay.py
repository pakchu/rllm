"""Outcome-blind preregistration for HVESDP-24."""
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


DEFAULT_OUTPUT = Path(
    "results/high_volatility_ethereum_staking_deposit_pressure_relay_preregistration_2026-08-12.json"
)
DEPOSIT_CONTRACT = "0x00000000219ab540356cBB839Cbe05303d7705Fa"
DEPOSIT_EVENT_SIGNATURE = "DepositEvent(bytes,bytes,bytes,bytes,bytes)"
DEPOSIT_EVENT_TOPIC = "0x649bbc62d0e31342afea4e5cd82d4049e7e1ee912fc0889aa790803be39038c5"
PRIMARY_RPC = "https://eth.drpc.org"
VERIFY_RPC = "https://ethereum-rpc.publicnode.com"


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def build() -> dict[str, Any]:
    core = copy.deepcopy(template.build())
    core.pop("manifest_hash")
    core.update(
        protocol_version="high_volatility_ethereum_staking_deposit_pressure_relay_v1",
        policy_id="HVESDP-24",
        as_of_date="2026-08-12",
        mechanism={
            "claim": (
                "Immutable DepositEvent logs from Ethereum's official mainnet beacon deposit contract reveal "
                "changes in validator staking commitment. Published staking research reports that staking ratios "
                "positively predict excess returns. During elevated causal BTC variation, go long BTC when the "
                "completed UTC day's deposit-event count exceeds the same weekday one week earlier and short when "
                "it is lower."
            ),
            "side": "daily deposit-event count D greater than D-7 maps long; lower maps short; equality is ineligible",
            "external_support": {
                "paper": "The Tokenomics of Staking (NBER Working Paper 33640, 2025)",
                "url": "https://www.nber.org/papers/w33640",
                "reported_fact": "The paper reports that staking ratios positively predict excess returns.",
                "inference_disclosure": (
                    "Daily Ethereum deposit-event pressure, the same-weekday comparison, and transmission from "
                    "Ethereum staking commitment to BTC are preregistered cross-asset adaptations, not the paper's "
                    "exact staking-ratio estimator."
                ),
            },
            "why_distinct": (
                "Exact repository scans found no beacon-chain validator-deposit, activation, exit, slashing, "
                "consensus-layer, or deposit-contract trading clock. The primitive is irreversible validator staking "
                "commitment, not an existing stablecoin, bridge, governance, implementation-history, market-direction, "
                "flow, funding, OI, premium, prior event set, or promoted control."
            ),
            "why_suited_to_volatile_regimes": (
                "Only decisions whose completed prior-24-hour BTC variation ranks in the upper 35% are admitted."
            ),
            "why_low_gross9_overlap_is_plausible": (
                "Conservative next-day Ethereum validator-deposit clocks and immutable DepositEvent counts are absent "
                "from Gross9."
            ),
        },
        features={
            "authority": (
                "canonical Ethereum mainnet execution logs at the Ethereum Foundation-published beacon deposit "
                f"contract {DEPOSIT_CONTRACT}"
            ),
            "event_contract": DEPOSIT_EVENT_SIGNATURE,
            "event_topic0": DEPOSIT_EVENT_TOPIC,
            "daily_count": (
                "number of canonical logs matching the exact contract and topic0 on each explicit UTC block-timestamp "
                "day, including zero-count days; amount and payload fields are not signal inputs"
            ),
            "pressure_change": "daily_count[D]-daily_count[D-7], strict nonzero",
            "side": "sign of pressure_change",
            "availability": (
                "12:00 UTC on D+1 only after the entire UTC source day is complete, every event block has at least 64 "
                "canonical descendants, and two independent RPC hosts replay the exact same normalized log set; live "
                "use is the later of these conditions"
            ),
            "btc_variation": (
                "sqrt(sum squared log(close/open)) over 1,440 exact BTCUSDT 1m bars in [decision-24h,decision)"
            ),
            "btc_variation_rank": (
                "strict-prior midrank versus at most 270 prior source-valid decisions; minimum 180; current excluded; "
                "rank >=0.65"
            ),
            "missing": (
                "chain-id/address/topic/schema/replay/canonical-block/finality drift or missing, duplicate, nonpositive "
                "BTC bars rejects; no imputation"
            ),
        },
        clock={
            "decision": "12:00 UTC on deposit source day D+1 after the frozen availability conditions",
            "entry": "exact BTCUSDT five-minute open 5 minutes after decision",
            "hold": "24 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding_oi_premium": "not signal inputs; exact funding only after novelty passes",
            "no_imputation": True,
        },
        policy={
            "deposit_contract": DEPOSIT_CONTRACT,
            "deposit_event_signature": DEPOSIT_EVENT_SIGNATURE,
            "deposit_event_topic0": DEPOSIT_EVENT_TOPIC,
            "same_weekday_lag_days": 7,
            "confirmation_blocks": 64,
            "publication_delay_days": 1,
            "decision_utc_hour": 12,
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
            "ethereum_execution_logs": {
                "chain_id": "0x1",
                "primary_rpc": PRIMARY_RPC,
                "verification_rpc": VERIFY_RPC,
                "contract": DEPOSIT_CONTRACT,
                "topic0": DEPOSIT_EVENT_TOPIC,
                "source_day_start": "2022-01-01",
                "source_day_end_exclusive": "2026-07-30",
                "maximum_log_query_block_span": 5000,
                "confirmation_blocks": 64,
                "download_after_preregistration": True,
                "two_host_exact_replay_required": True,
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
                "no_btc_volatility_gate",
                "pressure_direction_flip",
                "one_day_stale_pressure",
                "raw_day_over_day_pressure",
                "same_clock_forced_long",
            ],
            "diagnostic_controls_cannot_be_promoted": True,
        },
        research_boundary={
            "official_contract_address_and_event_schema_opened": True,
            "rpc_chain_id_only_opened": True,
            "historical_execution_logs_opened": False,
            "source_values_used_to_fit_rule": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "repository_staking_deposit_candidate_found": False,
            "cross_asset_inference_disclosed": True,
            "prior_event_sets_reused": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": (
                "official immutable validator-deposit logs, published staking-return evidence, conservative finality "
                "and availability, and exact repository absence"
            ),
        },
        stopping_rule=(
            "terminal first-failure sequence: source contract/support, Gross9 novelty, train/test/eval/final strict "
            "economics, then RV20 q90 audit; no contract, topic, provider, confirmation depth, lag, variation threshold, "
            "side, hold, clock, subset, source, or control repair"
        ),
    )
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core) or value != build():
        raise RuntimeError("HVESDP preregistration drift")
    if value["outcomes_opened"] is not False or value["source_incidence_opened"] is not False:
        raise RuntimeError("HVESDP boundary drift")
    if value["research_boundary"]["historical_execution_logs_opened"] is not False:
        raise RuntimeError("HVESDP source boundary drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    print(args.output)
