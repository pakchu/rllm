"""Outcome-blind preregistration for HVEMSD-24."""
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
    "results/high_volatility_ethereum_missed_slot_delta_relay_preregistration_2026-08-12.json"
)
RPC_HOSTS = (
    "https://eth-mainnet.public.blastapi.io",
    "https://eth.merkle.io",
)


def canonical_hash(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def build() -> dict[str, Any]:
    candidate = copy.deepcopy(template.build())
    candidate.pop("manifest_hash")
    candidate.update(
        protocol_version="high_volatility_ethereum_missed_slot_delta_relay_v1",
        policy_id="HVEMSD-24",
        as_of_date="2026-08-12",
        gross9_rows_opened=False,
        mechanism={
            "claim": (
                "After Ethereum proof of stake, each UTC day has exactly 7,200 twelve-second "
                "consensus slots. The difference between that schedule and canonical execution "
                "blocks measures missed block production without using a mutable explorer chart. "
                "An unusually large day-over-day change in missed slots is a cross-chain validator-"
                "liveness shock; during elevated completed BTC variation, rising missed-slot pressure "
                "maps short BTC and falling pressure maps long BTC for twenty-four hours."
            ),
            "side": "negative strict sign of the completed UTC-day missed-slot-count change",
            "external_support": {
                "ethereum_consensus_spec": "https://github.com/ethereum/consensus-specs/blob/dev/specs/phase0/beacon-chain.md",
                "ethereum_execution_api": "https://ethereum.github.io/execution-apis/api/methods/eth_getBlockByNumber/",
                "reported_facts": (
                    "Ethereum slots are twelve seconds; canonical execution block headers expose "
                    "block number, hash, parent hash and timestamp."
                ),
                "inference_disclosure": (
                    "daily missed-slot change, BTC variation gate, directional map and 24-hour hold "
                    "are preregistered adaptations, not a published trading rule"
                ),
            },
            "why_distinct": (
                "ESDI used Ethereum EIP-1559 base fees as blockspace demand. Ethereum flaw, staking, "
                "bridge and ETH-price candidates use different objects. HVEMSD uses only the immutable "
                "post-Merge execution-block production deficit against the consensus slot schedule; "
                "it uses no fee, gas, transfer, validator identity, ETH return, prior event or control."
            ),
            "why_suited_to_volatile_regimes": (
                "both the absolute validator-liveness change and completed BTC variation must lie in "
                "causal upper tails"
            ),
            "why_low_gross9_overlap_is_plausible": (
                "one delayed cross-chain liveness clock at 00:20 UTC is absent from Gross9"
            ),
        },
        features={
            "source": "dual-replayed canonical Ethereum execution block headers from two fixed RPC hosts",
            "utc_day_boundaries": (
                "for each day D, exact first canonical execution block with timestamp>=D 00:00 UTC "
                "and exact first block with timestamp>=(D+1) 00:00 UTC"
            ),
            "produced_blocks": "next_boundary_block_number-current_boundary_block_number",
            "scheduled_slots": 7200,
            "missed_slots": "7200-produced_blocks; integer in [0,7200]",
            "change": (
                "missed_slots[D]-missed_slots[D-1] for exact consecutive UTC days, strict nonzero"
            ),
            "missed_change_rank": (
                "strict-prior midrank of abs(change) over at most 365 valid changes, minimum 180, "
                "current excluded; rank>=0.70"
            ),
            "availability": (
                "D+1 00:20 UTC, after the day boundary plus at least 64 scheduled slots; both RPC "
                "hosts must agree on boundary block number/hash/timestamp and its 64-slot-later anchor"
            ),
            "btc_variation": (
                "sqrt(sum squared exact BTCUSDT 1m log(close/open)) over 24 elapsed hours ending at availability"
            ),
            "btc_variation_rank": (
                "strict-prior midrank over at most 270 valid decisions, minimum 180, current excluded; rank>=0.65"
            ),
            "missing_reorg_duplicate_or_disagreement": (
                "RPC disagreement, noncanonical parent relation, duplicate boundary, invalid timestamp, "
                "missing BTC minute or nonconsecutive day rejects; no imputation"
            ),
        },
        clock={
            "decision": "00:20 UTC after the completed Ethereum UTC day",
            "entry": "exact BTCUSDT 00:25 UTC five-minute open",
            "hold": "24 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding_oi_premium": "not signal inputs; exact funding only after novelty passes",
            "no_imputation": True,
        },
        policy={
            "missed_change_prior_observations": 365,
            "missed_change_prior_minimum": 180,
            "missed_change_midrank_min": 0.70,
            "variation_prior_observations": 270,
            "variation_prior_minimum": 180,
            "variation_midrank_min": 0.65,
            "confirmation_slots": 64,
            "decision_hour_utc": 0,
            "decision_minute_utc": 20,
            "entry_delay_minutes": 5,
            "hold_hours": 24,
            "gross_exposure": 0.5,
            "base_cost_per_notional_side": 0.0006,
            "stress_cost_per_notional_side": 0.001,
        },
        source_plan={
            "ethereum": {
                "chain_id": 1,
                "rpc_method": "eth_getBlockByNumber",
                "hosts": list(RPC_HOSTS),
                "download_after_preregistration": True,
                "fields": ["number", "hash", "parentHash", "timestamp"],
                "full_transactions": False,
                "post_merge_only": True,
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
                "no_missed_change_tail",
                "one_day_stale_change",
                "missed_level_change",
                "direction_flip",
                "same_clock_forced_long",
            ],
            "diagnostic_controls_cannot_be_promoted": True,
        },
        research_boundary={
            "consensus_schedule_execution_schema_and_transport_hosts_opened": True,
            "ethereum_boundary_blocks_daily_counts_changes_ranks_or_incidence_opened": False,
            "repository_exact_ethereum_missed_slot_delta_candidate_found": False,
            "prior_ethereum_fee_flaw_staking_bridge_and_price_outcomes_known": True,
            "prior_event_sets_or_controls_reused": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "independent immutable cross-chain validator-liveness object",
        },
        stopping_rule=(
            "terminal first-failure sequence: source contract and byte reproduction, source support, "
            "Gross9 novelty, train/test/eval/final strict economics, then RV20 q90; no source, "
            "threshold, side, hold, clock, subset or control repair"
        ),
    )
    return {**candidate, "manifest_hash": canonical_hash(candidate)}


def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value != build() or value.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("HVEMSD preregistration drift")
    if value["outcomes_opened"] is not False or value["source_incidence_opened"] is not False:
        raise RuntimeError("HVEMSD boundary drift")


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
