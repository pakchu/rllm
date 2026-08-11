"""Outcome-blind preregistration for HVUWLS-24."""
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
    "results/high_volatility_uniswap_wbtc_liquidity_shock_relay_preregistration_2026-08-12.json"
)
FACTORY = "0x1F98431c8aD98523631AE4a59f267346ea31F984"
POOL = "0x99ac8cA7087fA4A2A1FB6357269965A2014ABc35"
WBTC = "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599"
USDC = "0xA0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
SWAP_EVENT = "Swap(address,address,int256,int256,uint160,uint128,int24)"
SWAP_TOPIC0 = "0xc42079f94a6350d7e6235f291749249f928cc2ac818eb64d551f58dcb5e7f634"
PRIMARY_RPC = "https://ethereum-rpc.publicnode.com"
VERIFY_RPC = "https://1rpc.io/eth"


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
        protocol_version="high_volatility_uniswap_wbtc_liquidity_shock_relay_v1",
        policy_id="HVUWLS-24",
        as_of_date="2026-08-12",
        mechanism={
            "claim": (
                "Published transaction-level evidence from Uniswap v2/v3 and SushiSwap finds that unusually large "
                "crypto sell orders predict negative future token returns while unusually large buy orders predict "
                "positive future returns. On the canonical Ethereum Uniswap V3 WBTC/USDC 0.3% pool, follow the "
                "completed UTC day's largest absolute WBTC swap during elevated causal BTC variation."
            ),
            "side": (
                "negative pool amount0 means the trader receives WBTC and maps long BTC; positive pool amount0 "
                "means the trader supplies WBTC and maps short BTC"
            ),
            "external_support": {
                "paper": (
                    "Ante (2022), Liquidity Shocks, Token Returns and Market Capitalization in Decentralized "
                    "Finance (DeFi) Markets"
                ),
                "doi": "10.2139/ssrn.4183105",
                "reported_fact": (
                    "Using 2.77 million swaps across Uniswap v2/v3 and SushiSwap, the paper reports that large sell "
                    "size is significantly associated with negative future token returns and large buy size with "
                    "positive future token returns."
                ),
                "inference_disclosure": (
                    "The WBTC/USDC pool, one-largest-swap daily summary, strict-prior magnitude rank, BTC cross-venue "
                    "execution, variation gate, decision delay, and 24-hour hold are preregistered adaptations."
                ),
                "official_pool_schema": "https://github.com/Uniswap/v3-core",
                "official_deployments": "https://developers.uniswap.org/docs/protocols/v3/deployments",
            },
            "why_distinct": (
                "Exact repository scans found no Uniswap V3 WBTC/USDC pool, pool address 0x99ac...bc35, Uniswap "
                "WBTC swap, or immutable DEX WBTC order-flow candidate. It is not CEX taker flow, cross-alt flow, "
                "stablecoin issuance, bridge flow, funding, OI, premium, a prior event set, or a promoted control."
            ),
            "why_suited_to_volatile_regimes": (
                "Both an extreme immutable DEX liquidity shock and upper-35% completed BTC variation are required, "
                "targeting July-like volatile states."
            ),
            "why_low_gross9_overlap_is_plausible": (
                "A conservative next-day clock derived from the largest finalized WBTC/USDC on-chain swap is absent "
                "from Gross9 primitives."
            ),
        },
        features={
            "pool_identity": (
                "Ethereum mainnet Uniswap V3 factory getPool(WBTC,USDC,3000) must equal the frozen pool on both RPC "
                "hosts; token0=WBTC, token1=USDC and fee=3000 must match contract getters"
            ),
            "source_day": (
                "UTC day D bounded by the first canonical block with timestamp >=D 00:00 and the first canonical "
                "block with timestamp >=D+1 00:00"
            ),
            "day_completeness": (
                "the end-exclusive UTC boundary block header, including number/hash/parentHash/timestamp, must replay "
                "identically on both hosts and have at least 64 canonical descendants before D is available; this "
                "also binds valid zero-log days"
            ),
            "swap_normalization": (
                "exact pool address and topic0; three topics; 160-byte ABI data; signed int256 amount0/amount1 must be "
                "nonzero with opposite signs; uint160 sqrtPriceX96 and uint128 liquidity ranges; signed int24 tick "
                "range; canonical blockHash/transactionHash/logIndex; preserve normalized topics and data hashes"
            ),
            "daily_shock": (
                "the unique canonical swap with maximum absolute amount0/1e8 in UTC day D; ties break by earliest "
                "blockNumber, transactionIndex, then logIndex"
            ),
            "shock_magnitude_rank": (
                "strict-prior midrank of absolute amount0/1e8 versus at most 270 previous source-valid daily shocks; "
                "minimum 180; current excluded; rank>=0.80"
            ),
            "side": "amount0<0 maps long BTC; amount0>0 maps short BTC",
            "btc_variation": (
                "sqrt(sum squared log(close/open)) over 1,440 exact BTCUSDT 1m bars in [decision-24h,decision)"
            ),
            "btc_variation_rank": (
                "strict-prior midrank versus at most 270 prior source-valid decisions; minimum 180; current excluded; "
                "rank>=0.65"
            ),
            "missing": (
                "chain, factory, pool, token, fee, ABI, boundary, finality, replay, canonical-block, duplicate, source "
                "or BTC-bar drift rejects; no imputation"
            ),
        },
        clock={
            "decision": "12:00 UTC on D+1 after the frozen full-day replay and finality conditions",
            "entry": "exact BTCUSDT five-minute open 5 minutes after decision",
            "hold": "24 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding_oi_premium": "not signal inputs; exact funding only after novelty passes",
            "no_imputation": True,
        },
        policy={
            "chain_id": "0x1",
            "factory": FACTORY,
            "pool": POOL,
            "wbtc": WBTC,
            "usdc": USDC,
            "fee": 3000,
            "swap_event": SWAP_EVENT,
            "swap_topic0": SWAP_TOPIC0,
            "confirmation_blocks": 64,
            "decision_utc_hour": 12,
            "shock_prior_days": 270,
            "shock_prior_minimum": 180,
            "shock_midrank_min": 0.80,
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
            "ethereum_uniswap_v3_logs": {
                "primary_rpc": PRIMARY_RPC,
                "verification_rpc": VERIFY_RPC,
                "source_day_start": "2022-01-01",
                "source_day_end_exclusive": "2026-07-30",
                "maximum_log_query_block_span": 2000,
                "two_host_exact_replay_required": True,
                "full_day_boundary_header_replay_required": True,
                "read_after_preregistration": True,
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
                "liquidity_shock_direction_flip",
                "one_day_stale_liquidity_shock",
                "daily_net_wbtc_flow",
                "same_clock_forced_long",
            ],
            "diagnostic_controls_cannot_be_promoted": True,
        },
        research_boundary={
            "paper_and_official_abi_only_opened": True,
            "rpc_chain_id_and_factory_get_pool_only_opened": True,
            "historical_uniswap_swap_logs_opened": False,
            "candidate_source_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "repository_uniswap_wbtc_candidate_found": False,
            "prior_event_sets_reused": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": (
                "published DEX liquidity-shock direction, official immutable pool events, high-variation targeting, "
                "two-host source feasibility, and exact repository absence"
            ),
        },
        stopping_rule=(
            "terminal first-failure sequence: source contract/support, Gross9 novelty, train/test/eval/final strict "
            "economics, then RV20 q90 audit; no pool, provider, event, finality, daily statistic, rank, threshold, "
            "side, hold, clock, subset, source, or control repair"
        ),
    )
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core) or value != build():
        raise RuntimeError("HVUWLS preregistration drift")
    if value["outcomes_opened"] is not False or value["source_incidence_opened"] is not False:
        raise RuntimeError("HVUWLS boundary drift")
    if value["research_boundary"]["historical_uniswap_swap_logs_opened"] is not False:
        raise RuntimeError("HVUWLS historical source boundary drift")


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
