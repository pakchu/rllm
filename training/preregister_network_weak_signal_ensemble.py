"""Freeze NWE-7 before constructing any return-labelled model sample."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = "results/network_weak_signal_ensemble_preregistration_2026-07-17.json"
BLOCKSPACE_SOURCE = "data/coinmetrics_btc_blockspace_security_daily_2019_2023.csv.gz"
BLOCKSPACE_MANIFEST = (
    "results/coinmetrics_btc_blockspace_security_daily_2019_2023_manifest_2026-07-17.json"
)
NETWORK_SOURCE = "data/coinmetrics_btc_network_daily_2020_2023.csv.gz"
NETWORK_MANIFEST = "results/coinmetrics_btc_network_daily_pre2024_manifest_2026-07-16.json"
MARKET_SOURCE = "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz"
FUNDING_SOURCE = "data/binance_funding_btc_2020_2023.csv.gz"


@dataclass(frozen=True)
class Policy:
    policy_id: str = "NWE-7"
    reference_days: int = 180
    reference_min_periods: int = 120
    feature_change_days: int = 7
    feature_clip: float = 5.0
    decision_weekday: int = 0
    decision_hour_utc: int = 12
    decision_minute_utc: int = 0
    maximum_observation_age_days: float = 3.0
    fit_history_start: str = "2020-01-06"
    prediction_start: str = "2021-03-01"
    minimum_train_samples: int = 52
    maximum_train_samples: int = 104
    ridge_alpha: float = 10.0
    abstain_quantile: float = 0.5
    entry_delay_bars: int = 1
    hold_bars: int = 2_016
    leverage: float = 0.5
    base_cost_notional_per_side: float = 0.0006
    stress_cost_notional_per_side: float = 0.0010


FEATURE_COLUMNS = (
    "fee_share_level_z",
    "fee_share_change_z",
    "transaction_density_level_z",
    "transaction_density_change_z",
    "breadth_level_z",
    "breadth_change_z",
    "fanout_level_z",
    "fanout_change_z",
)


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_manifest() -> dict[str, Any]:
    core: dict[str, Any] = {
        "protocol_version": "network_weak_signal_ensemble_v1",
        "outcomes_opened": False,
        "policy": asdict(Policy()),
        "feature_columns": list(FEATURE_COLUMNS),
        "research_history_boundary": {
            "2020_2023_market_returns_seen_by_unrelated_repo_research": True,
            "exact_nwe7_model_outcomes_opened": False,
            "predecessor_support_only_failures": ["NTB-7", "BFC-3"],
            "predecessor_pnl_opened": False,
            "claim": (
                "NWE-7 is a new continuous weak-signal model, not a threshold repair; "
                "2024+ remains sealed until all pre-2024 gates pass"
            ),
        },
        "novelty_boundary": {
            "economic_axis": (
                "weekly price-independent network economics decoder combining weak "
                "blockspace-demand and ledger-topology states"
            ),
            "not": [
                "tail-event threshold repair of NTB-7 or BFC-3",
                "BTC price, return, volatility, extrema, wick, or volume as model input",
                "exchange-address flow tagging",
                "funding, premium, basis, OI, Kimchi, DXY, FX, REX, or Markov input",
                "spot/perp/aggTrade microstructure",
                "unconditional long drift",
            ],
        },
        "source_contract": {
            "blockspace": BLOCKSPACE_SOURCE,
            "blockspace_sha256": (
                "c94fd06ff695d673503a56064284cffbb36e6f1ac847bdc6b38819752a77985b"
            ),
            "blockspace_manifest": BLOCKSPACE_MANIFEST,
            "blockspace_manifest_sha256": (
                "eb70f5cd38d0895b9c04ca142ce15645696769f10866072b8f8ef64ec7a49cf1"
            ),
            "network": NETWORK_SOURCE,
            "network_sha256": (
                "97ab2ca9d0c347d85221b51734f98072763370072ca51f1c40e3214191159b42"
            ),
            "network_manifest": NETWORK_MANIFEST,
            "network_manifest_sha256": (
                "66b185769800c4732cf748b40ca9cb48c5eee239abf0425ff193c0688111c372"
            ),
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
                "raw on-chain values use AssetEODCompletionTime and frozen hashes, "
                "but are not a full historical revision-vintage archive; live "
                "promotion requires forward-vintage parity"
            ),
            "excluded_for_leakage_risk": (
                "Coin Metrics exchange-tag flows are excluded without point-in-time tags"
            ),
        },
        "causal_feature_contract": {
            "raw_states": {
                "fee_share": "log(FeeTotNtv / IssTotNtv)",
                "transaction_density": "log(TxCnt / BlkCnt)",
                "breadth": "log(AdrActCnt / TxTfrCnt)",
                "fanout": "log(TxTfrCnt / TxCnt)",
            },
            "level_features": (
                "strictly-prior 180-observation z-score of each raw state; require 120"
            ),
            "change_features": (
                "strictly-prior 180-observation z-score of each raw state's seven-day change"
            ),
            "availability": (
                "merge sources by observation date, use max available_at, and include "
                "only reference observations published strictly before the current row"
            ),
            "weekly_snapshot": (
                "each Monday 12:00 UTC uses the latest common observation whose "
                "available_at is not later than the decision and whose age is <=3 days"
            ),
            "clip": "each z feature clipped to [-5, 5] before model fitting",
            "price_or_derivative_feature_columns_loaded": [],
        },
        "online_model_contract": {
            "target": "unlevered log return from scheduled weekly entry open to exit open",
            "label_availability": (
                "a historical sample enters a refit only when both source available_at "
                "and scheduled label exit are <= current weekly decision"
            ),
            "training_window": "most recent 104 fully labelled samples; require 52",
            "feature_standardization": (
                "mean and sample standard deviation fit on training samples only; "
                "zero-variance columns become zero"
            ),
            "target_centering": (
                "subtract training target mean and do not add it back to the forecast; "
                "this removes unconditional BTC drift"
            ),
            "estimator": (
                "closed-form L2 ridge beta=(X'X+10I)^-1 X'(y-mean(y)); no intercept"
            ),
            "abstention": (
                "trade only when abs(out-of-sample forecast) >= median absolute "
                "in-sample fitted centered forecast for that refit"
            ),
            "side": "long for positive forecast, short for negative forecast, else flat",
            "refit": "every Monday; no hyperparameter or feature selection",
        },
        "execution_contract": {
            "decision": "Monday 12:00 UTC after feature snapshot is fixed",
            "entry": "Monday 12:05 UTC open (one complete five-minute latency bar)",
            "exit": "scheduled open 2,016 five-minute bars/seven days after entry",
            "nonoverlap": True,
            "stop_or_take_profit": None,
            "leverage": 0.5,
            "base_cost": "6bp/notional/side",
            "stress_cost": "10bp/notional/side",
            "funding_interval": "entry_time <= funding_time < exit_time",
            "cagr": "full wall-clock split including idle/abstained cash",
            "strict_mdd": (
                "global/pre-entry HWM, favorable-before-adverse held OHLC, funding, "
                "entry/exit/hypothetical liquidation costs"
            ),
        },
        "support_freeze_before_labels": {
            "train_2021_2022_candidate_weeks_min": 90,
            "train_2021_candidate_weeks_min": 42,
            "train_2022_candidate_weeks_min": 50,
            "selection_2023_candidate_weeks_min": 50,
            "selection_2023_h1_candidate_weeks_min": 25,
            "selection_2023_h2_candidate_weeks_min": 25,
            "all_feature_values_finite": True,
            "market_or_return_rows_loaded": 0,
            "failure_action": "reject without building return labels",
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
            "performance_gates": {
                "train_trade_count_min": 35,
                "selection_trade_count_min": 18,
                "each_selection_half_trade_count_min": 8,
                "each_side_share_range": [0.25, 0.75],
                "train_and_2023_absolute_return_positive": True,
                "train_and_2023_cagr_to_strict_mdd_min": 3.0,
                "train_and_2023_strict_mdd_pct_max": 15.0,
                "train_and_2023_weekly_cluster_signflip_p_max": 0.10,
                "train_and_2023_mean_gross_underlying_bp_min": 20.0,
                "train_and_2023_ten_bp_stress_positive": True,
                "2023_h1_and_h2_absolute_return_positive": True,
                "one_bar_delayed_entry_train_and_2023_positive": True,
            },
        },
        "controls": {
            "direction_flip": "same forecasts, opposite side; diagnostic-only",
            "fee_family_only": "same online model using the four fee/density features",
            "topology_family_only": "same online model using the four breadth/fanout features",
            "no_abstention": "same forecasts traded every week",
            "stale_features_7d": "all eight inputs delayed seven observations",
            "year_stratified_feature_permutation": (
                "permute feature rows within year before each deterministic refit"
            ),
            "constant_long": "weekly long with the same execution clock",
            "one_bar_delayed_entry": "same side one extra five-minute bar later",
            "mechanism_rejection_rule": (
                "reject NWE-7 if stale or permuted features independently pass every "
                "primary gate; component-only results are attribution, not replacements"
            ),
        },
        "orthogonality_after_performance": {
            "comparison_set": "all promoted/live/shadow sleeves frozen before NWE-7",
            "exact_entry_jaccard_max": 0.05,
            "position_time_jaccard_max": 0.20,
            "absolute_daily_pnl_pearson_max": 0.30,
            "minimum_nonzero_daily_pnl_days": 20,
            "marginal_portfolio_improvement_required": True,
            "undefined_metric": "fail_closed",
        },
        "rejection_contract": (
            "any support, performance, mechanism, or orthogonality failure rejects "
            "NWE-7 without changing features, ridge, window, abstention, side, or hold"
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
        raise RuntimeError("NWE-7 preregistration hash mismatch")
    if payload.get("outcomes_opened") is not False:
        raise RuntimeError("NWE-7 preregistration cannot open outcomes")
    if payload.get("policy") != asdict(Policy()):
        raise RuntimeError("NWE-7 policy differs from code")
    if payload.get("feature_columns") != list(FEATURE_COLUMNS):
        raise RuntimeError("NWE-7 feature order differs from code")
    if payload.get("selection_protocol", {}).get("candidate_count") != 1:
        raise RuntimeError("NWE-7 must remain a singleton")
    if payload.get("causal_feature_contract", {}).get(
        "price_or_derivative_feature_columns_loaded"
    ) != []:
        raise RuntimeError("NWE-7 support clock must not load market features")


def write_once(path: str | Path, payload: dict[str, Any]) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = json.loads(output.read_text())
        validate_manifest(existing)
        if existing["manifest_hash"] != payload["manifest_hash"]:
            raise RuntimeError("refusing to overwrite frozen NWE-7 preregistration")
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
