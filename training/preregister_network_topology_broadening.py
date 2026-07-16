"""Freeze NTB-7 before inspecting any post-entry market outcome."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = "results/network_topology_broadening_preregistration_2026-07-17.json"
NETWORK_SOURCE = "data/coinmetrics_btc_network_daily_2020_2023.csv.gz"
NETWORK_MANIFEST = "results/coinmetrics_btc_network_daily_pre2024_manifest_2026-07-16.json"
MARKET_SOURCE = "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz"
FUNDING_SOURCE = "data/binance_funding_btc_2020_2023.csv.gz"


@dataclass(frozen=True)
class Policy:
    policy_id: str = "NTB-7"
    change_days: int = 7
    reference_days: int = 180
    reference_min_periods: int = 120
    breadth_z_min: float = 0.5
    fanout_z_max: float = -0.5
    composite_min: float = 1.5
    maximum_source_lag_days: float = 3.0
    entry_delay_bars: int = 1
    hold_bars: int = 2_016
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
        "protocol_version": "network_topology_broadening_v1",
        "outcomes_opened": False,
        "policy": asdict(Policy()),
        "research_history_boundary": {
            "2020_2023_market_returns_seen_by_unrelated_repo_research": True,
            "exact_ntb7_outcomes_opened": False,
            "claim": (
                "exact-policy mechanical freeze, not a globally pristine market "
                "holdout; 2024+ remains sealed until pre-2024 performance and "
                "orthogonality gates pass"
            ),
        },
        "novelty_boundary": {
            "economic_axis": (
                "daily Bitcoin ledger topology: unique-address participation "
                "broadens while positive-value transfers become less concentrated "
                "inside multi-transfer transactions"
            ),
            "not": [
                "absolute active-address, transaction, or transfer activity shock",
                "price momentum or price reversal clock",
                "funding, premium, basis, OI, Kimchi, DXY, or FX",
                "REX, rolling extrema, Markov state, wick rejection, or aggTrade microstructure",
            ],
        },
        "metric_semantics": {
            "AdrActCnt": (
                "Coin Metrics count of unique addresses active as recipient or "
                "originator during the daily interval"
            ),
            "TxCnt": "Coin Metrics count of ledger transactions during the day",
            "TxTfrCnt": (
                "Coin Metrics count of positive-value transfers; one transaction "
                "may contain multiple transfers such as batched withdrawals or payroll"
            ),
            "official_sources": [
                "https://docs.coinmetrics.io/asset-metrics/addresses/adractcnt",
                "https://docs.coinmetrics.io/asset-metrics/transactions/txcnt",
                "https://docs.coinmetrics.io/asset-metrics/transactions/txtfrcnt",
                "https://docs.coinmetrics.io/api/v4/",
            ],
            "interpretation_limit": (
                "addresses are not users and transfer fan-out is not entity-labelled; "
                "the ratios are topology proxies, not direct exchange-flow labels"
            ),
        },
        "source_contract": {
            "network": NETWORK_SOURCE,
            "network_sha256": (
                "97ab2ca9d0c347d85221b51734f98072763370072ca51f1c40e3214191159b42"
            ),
            "network_rows": 1_461,
            "network_manifest": NETWORK_MANIFEST,
            "network_manifest_sha256": (
                "66b185769800c4732cf748b40ca9cb48c5eee239abf0425ff193c0688111c372"
            ),
            "network_interval": ["2020-01-01", "2024-01-01"],
            "availability_column": "AssetEODCompletionTime stored as available_at",
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
                "file hashes freeze the downloaded vintage; forward promotion still "
                "requires live-vintage parity because historical metrics can be revised"
            ),
        },
        "causal_feature_contract": {
            "fanout": "log(TxTfrCnt / TxCnt)",
            "breadth": "log(AdrActCnt / TxTfrCnt)",
            "fanout_change": "fanout[t] - fanout[t-7 observations]",
            "breadth_change": "breadth[t] - breadth[t-7 observations]",
            "reference": (
                "for each component, mean and sample standard deviation of the last "
                "180 earlier observation dates whose available_at is strictly before "
                "the candidate available_at; require at least 120"
            ),
            "fanout_z": "(fanout_change - strictly-prior reference mean) / reference std",
            "breadth_z": "(breadth_change - strictly-prior reference mean) / reference std",
            "composite": "breadth_z - fanout_z",
            "eligible": (
                "availability lag <=3 days, breadth_z>=0.5, fanout_z<=-0.5, "
                "and composite>=1.5"
            ),
            "event": "first eligible observation after an ineligible observation",
            "direction": "long only",
            "price_or_derivative_feature_columns_loaded": [],
        },
        "execution_contract": {
            "decision": "only after the row's AssetEODCompletionTime available_at",
            "earliest_tradable_open": "first UTC five-minute bar open >= available_at",
            "entry": (
                "one complete five-minute latency bar after earliest_tradable_open "
                "(entry_delay_bars=1)"
            ),
            "exit": "scheduled open 2,016 five-minute bars (seven days) after entry",
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
            "train_2021_2022_nonoverlap_min": 40,
            "each_train_year_min": 15,
            "selection_2023_min": 16,
            "selection_2023_h1_min": 6,
            "selection_2023_h2_min": 6,
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
                "train_and_2023_mean_gross_underlying_bp_min": 40.0,
                "train_and_2023_ten_bp_stress_positive": True,
                "2023_h1_and_h2_absolute_return_positive": True,
                "one_bar_delayed_entry_train_and_2023_positive": True,
            },
        },
        "controls": {
            "direction_flip": (
                "primary clock, short side; diagnostic-only and never a repair or replacement"
            ),
            "breadth_component_only": (
                "long on onset of breadth_z>=1.5 without fanout or composite conditions"
            ),
            "fanout_component_only": (
                "long on onset of fanout_z<=-1.5 without breadth or composite conditions"
            ),
            "absolute_activity_control": (
                "prior chain-activity-shock clock evaluated separately; not eligible "
                "to replace NTB-7"
            ),
            "stale_topology_7d": "primary topology state delayed by seven observations",
            "one_bar_delayed_entry": (
                "primary side, entry one additional five-minute bar after primary"
            ),
            "year_stratified_random_clock": "same yearly trade counts and seven-day hold",
            "mechanism_rejection_rule": (
                "reject NTB-7 if either component-only or stale-topology control "
                "independently passes every primary gate"
            ),
        },
        "orthogonality_after_performance": {
            "comparison_set": (
                "all promoted/live/shadow sleeves frozen before NTB-7 outcomes open"
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
            "NTB-7 without changing ratios, lookback, thresholds, side, latency, or hold"
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
        raise RuntimeError("NTB-7 preregistration hash mismatch")
    if payload.get("outcomes_opened") is not False:
        raise RuntimeError("NTB-7 preregistration cannot open outcomes")
    if payload.get("policy") != asdict(Policy()):
        raise RuntimeError("NTB-7 policy differs from code")
    if payload.get("selection_protocol", {}).get("candidate_count") != 1:
        raise RuntimeError("NTB-7 must remain a singleton")
    loaded = payload.get("causal_feature_contract", {}).get(
        "price_or_derivative_feature_columns_loaded"
    )
    if loaded != []:
        raise RuntimeError("NTB-7 signal clock must not load market features")


def write_once(path: str | Path, payload: dict[str, Any]) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = json.loads(output.read_text())
        validate_manifest(existing)
        if existing["manifest_hash"] != payload["manifest_hash"]:
            raise RuntimeError("refusing to overwrite frozen NTB-7 preregistration")
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
