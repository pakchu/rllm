"""Freeze BFC-3 before inspecting any post-entry market outcome."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = "results/blockspace_fee_confirmation_preregistration_2026-07-17.json"
BLOCKSPACE_SOURCE = "data/coinmetrics_btc_blockspace_security_daily_2019_2023.csv.gz"
BLOCKSPACE_MANIFEST = (
    "results/coinmetrics_btc_blockspace_security_daily_2019_2023_manifest_2026-07-17.json"
)
MARKET_SOURCE = "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz"
FUNDING_SOURCE = "data/binance_funding_btc_2020_2023.csv.gz"


@dataclass(frozen=True)
class Policy:
    policy_id: str = "BFC-3"
    reference_days: int = 180
    reference_min_periods: int = 120
    fee_share_z_min: float = 1.0
    transaction_density_z_min: float = 0.0
    composite_min: float = 1.5
    transaction_density_weight: float = 0.5
    maximum_source_lag_days: float = 3.0
    entry_delay_bars: int = 1
    hold_bars: int = 864
    leverage: float = 0.5
    base_cost_notional_per_side: float = 0.0006
    stress_cost_notional_per_side: float = 0.0010


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_manifest() -> dict[str, Any]:
    core: dict[str, Any] = {
        "protocol_version": "blockspace_fee_confirmation_v1",
        "outcomes_opened": False,
        "policy": asdict(Policy()),
        "research_history_boundary": {
            "2020_2023_market_returns_seen_by_unrelated_repo_research": True,
            "exact_bfc3_outcomes_opened": False,
            "claim": (
                "exact-policy mechanical freeze, not a globally pristine market "
                "holdout; 2024+ remains sealed until pre-2024 performance and "
                "orthogonality gates pass"
            ),
        },
        "novelty_boundary": {
            "economic_axis": (
                "price-independent Bitcoin blockspace demand: transactor-paid fees "
                "rise relative to protocol issuance while transactions per main-chain "
                "block remain at or above their own baseline"
            ),
            "not": [
                "BTC price momentum, reversal, extrema, wick, or volatility",
                "active-address/transaction/transfer activity shock",
                "exchange-address inflow or outflow tagging",
                "hashrate, difficulty, hashprice, or miner capitulation",
                "funding, premium, basis, OI, Kimchi, DXY, FX, REX, or Markov state",
                "spot/perp/aggTrade microstructure",
            ],
        },
        "metric_semantics": {
            "FeeTotNtv": "sum of transactor-paid fees; excludes newly issued units",
            "IssTotNtv": "sum of newly issued native units",
            "BlkCnt": "main-chain blocks created during the interval",
            "TxCnt": "ledger transactions during the interval",
            "AssetEODCompletionTime": "time at which EOD metrics were fully calculated",
            "official_catalog": (
                "https://community-api.coinmetrics.io/v4/catalog/metrics?"
                "metrics=FeeTotNtv%2CIssTotNtv%2CBlkCnt%2CTxCnt%2CAssetEODCompletionTime"
            ),
            "official_api": "https://docs.coinmetrics.io/api/v4/",
        },
        "source_contract": {
            "blockspace": BLOCKSPACE_SOURCE,
            "blockspace_sha256": (
                "c94fd06ff695d673503a56064284cffbb36e6f1ac847bdc6b38819752a77985b"
            ),
            "blockspace_rows": 1_826,
            "blockspace_manifest": BLOCKSPACE_MANIFEST,
            "blockspace_manifest_sha256": (
                "eb70f5cd38d0895b9c04ca142ce15645696769f10866072b8f8ef64ec7a49cf1"
            ),
            "blockspace_interval": ["2019-01-01", "2024-01-01"],
            "market": MARKET_SOURCE,
            "market_sha256": (
                "a77cd0ae5b88b3c95e509d8d2610773d34af3afdc9170c63d88564bc3d0b990c"
            ),
            "market_columns_allowed": ["date", "open", "high", "low", "close"],
            "market_physical_read_boundary": "date < 2024-01-01",
            "funding": FUNDING_SOURCE,
            "funding_sha256": (
                "f61e6bce5d1b4e3c4b47b1f5819c06cac2c41fb439b5d4d414412afb0b580e04"
            ),
            "funding_physical_read_boundary": "date < 2024-01-01",
            "database_snapshot_is_point_in_time": False,
            "revision_boundary": (
                "AssetEODCompletionTime enforces semantic publication latency and "
                "hashes freeze the downloaded vintage; live promotion requires "
                "forward-vintage parity because historical network metrics can revise"
            ),
            "excluded_for_leakage_risk": (
                "FlowInEx/FlowOutEx/SplyEx exchange-tag metrics are excluded because "
                "a point-in-time address-tag vintage archive is unavailable"
            ),
        },
        "causal_feature_contract": {
            "fee_share": "log(FeeTotNtv / IssTotNtv)",
            "transaction_density": "log(TxCnt / BlkCnt)",
            "reference": (
                "component mean and sample standard deviation from the last 180 "
                "earlier observation dates whose available_at is strictly before "
                "the candidate available_at; require at least 120"
            ),
            "fee_share_z": "(fee_share - strictly-prior mean) / strictly-prior std",
            "transaction_density_z": (
                "(transaction_density - strictly-prior mean) / strictly-prior std"
            ),
            "composite": "fee_share_z + 0.5 * transaction_density_z",
            "eligible": (
                "availability lag <=3 days, fee_share_z>=1.0, "
                "transaction_density_z>=0.0, and composite>=1.5"
            ),
            "event": "first eligible observation after an ineligible observation",
            "direction": "long only",
            "price_or_derivative_feature_columns_loaded": [],
        },
        "execution_contract": {
            "decision": "only after the row's AssetEODCompletionTime available_at",
            "earliest_tradable_open": "first UTC five-minute bar open >= available_at",
            "entry": (
                "one complete five-minute latency bar after earliest_tradable_open"
            ),
            "exit": "scheduled open 864 five-minute bars (three days) after entry",
            "nonoverlap": True,
            "stop_or_take_profit": None,
            "leverage": 0.5,
            "base_cost": "6bp/notional/side",
            "stress_cost": "10bp/notional/side",
            "funding_interval": "entry_time <= funding_time < exit_time",
            "cagr": "full wall-clock split including idle cash",
            "strict_mdd": (
                "global/pre-entry HWM, favorable-before-adverse held OHLC, funding, "
                "entry/exit/hypothetical liquidation costs"
            ),
        },
        "support_freeze_before_returns": {
            "train_2021_2022_nonoverlap_min": 35,
            "each_train_year_min": 14,
            "selection_2023_min": 14,
            "selection_2023_h1_min": 5,
            "selection_2023_h2_min": 5,
            "maximum_single_month_share": 0.20,
            "stale_backfill_rows_may_seed_reference_but_may_not_signal": True,
            "future_hold_availability_used_to_filter": False,
            "failure_action": "reject without opening post-entry outcomes",
        },
        "selection_protocol": {
            "train": ["2021-03-01", "2023-01-01"],
            "selection": ["2023-01-01", "2024-01-01"],
            "selection_halves": {
                "h1": ["2023-01-01", "2023-07-01"],
                "h2": ["2023-07-01", "2024-01-01"],
            },
            "sealed": ["2024", "2025", "2026_ytd"],
            "candidate_count": 1,
            "no_parameter_repair": True,
            "gates": {
                "train_and_2023_absolute_return_positive": True,
                "train_and_2023_cagr_to_strict_mdd_min": 3.0,
                "train_and_2023_strict_mdd_pct_max": 15.0,
                "train_and_2023_weekly_cluster_signflip_p_max": 0.10,
                "train_and_2023_mean_gross_underlying_bp_min": 30.0,
                "train_and_2023_ten_bp_stress_positive": True,
                "2023_h1_and_h2_absolute_return_positive": True,
                "one_bar_delayed_entry_train_and_2023_positive": True,
            },
        },
        "controls": {
            "direction_flip": (
                "primary clock, short side; diagnostic-only and never a repair or replacement"
            ),
            "fee_component_only": (
                "long on onset of fee_share_z>=1.5 without density or composite conditions"
            ),
            "density_component_only": (
                "long on onset of transaction_density_z>=1.5 without fee conditions"
            ),
            "low_fee_mirror": (
                "long on fee_share_z<=-1.0, density_z>=0.0, and mirrored composite<=-1.5"
            ),
            "stale_blockspace_7d": "primary blockspace state delayed by seven observations",
            "one_bar_delayed_entry": (
                "primary side, entry one additional five-minute bar after primary"
            ),
            "year_stratified_random_clock": "same yearly trade counts and three-day hold",
            "mechanism_rejection_rule": (
                "reject BFC-3 if fee-only, density-only, low-fee mirror, or stale "
                "control independently passes every primary gate"
            ),
        },
        "orthogonality_after_performance": {
            "comparison_set": (
                "all promoted/live/shadow sleeves frozen before BFC-3 outcomes open"
            ),
            "exact_entry_jaccard_max": 0.05,
            "position_time_jaccard_max": 0.15,
            "absolute_daily_pnl_pearson_max": 0.30,
            "minimum_nonzero_daily_pnl_days": 20,
            "marginal_portfolio_improvement_required": True,
            "undefined_metric": "fail_closed",
        },
        "rejection_contract": (
            "any support, performance, mechanism, or orthogonality failure rejects "
            "BFC-3 without changing formula, thresholds, side, latency, or hold"
        ),
    }
    return {
        **core,
        "manifest_hash": canonical_hash(core),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def validate_manifest(payload: dict[str, Any]) -> None:
    core = {
        key: value
        for key, value in payload.items()
        if key not in {"manifest_hash", "created_at"}
    }
    if canonical_hash(core) != payload.get("manifest_hash"):
        raise RuntimeError("BFC-3 preregistration hash mismatch")
    if payload.get("outcomes_opened") is not False:
        raise RuntimeError("BFC-3 preregistration cannot open outcomes")
    if payload.get("policy") != asdict(Policy()):
        raise RuntimeError("BFC-3 policy differs from code")
    if payload.get("selection_protocol", {}).get("candidate_count") != 1:
        raise RuntimeError("BFC-3 must remain a singleton")
    if payload.get("causal_feature_contract", {}).get(
        "price_or_derivative_feature_columns_loaded"
    ) != []:
        raise RuntimeError("BFC-3 signal clock must not load market features")


def write_once(path: str | Path, payload: dict[str, Any]) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = json.loads(output.read_text())
        validate_manifest(existing)
        if existing["manifest_hash"] != payload["manifest_hash"]:
            raise RuntimeError("refusing to overwrite frozen BFC-3 preregistration")
        return "verified_existing"
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return "created"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build_manifest()
    status = write_once(args.output, payload)
    print(
        json.dumps(
            {
                "status": status,
                "outcomes_opened": False,
                "policy_id": payload["policy"]["policy_id"],
                "manifest_hash": payload["manifest_hash"],
                "output": args.output,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
