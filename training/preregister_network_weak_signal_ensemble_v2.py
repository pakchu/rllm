"""Freeze NWE-8 before constructing any return-labelled model sample."""
from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training import preregister_network_weak_signal_ensemble as nwe7


DEFAULT_OUTPUT = "results/network_weak_signal_ensemble_v2_preregistration_2026-07-17.json"
FEATURE_COLUMNS = nwe7.FEATURE_COLUMNS
canonical_hash = nwe7.canonical_hash


@dataclass(frozen=True)
class Policy:
    policy_id: str = "NWE-8"
    reference_days: int = 180
    reference_min_periods: int = 120
    feature_change_days: int = 7
    feature_clip: float = 5.0
    decision_weekday: int = 0
    decision_hour_utc: int = 12
    decision_minute_utc: int = 0
    maximum_observation_age_days: float = 3.0
    fit_history_start: str = "2020-01-06"
    prediction_start: str = "2021-06-07"
    minimum_train_samples: int = 52
    maximum_train_samples: int = 104
    ridge_alpha: float = 10.0
    abstain_quantile: float = 0.5
    entry_delay_bars: int = 1
    hold_bars: int = 2_016
    leverage: float = 0.5
    base_cost_notional_per_side: float = 0.0006
    stress_cost_notional_per_side: float = 0.0010


def build_manifest() -> dict[str, Any]:
    predecessor = nwe7.build_manifest()
    core = copy.deepcopy(
        {
            key: value
            for key, value in predecessor.items()
            if key not in {"manifest_hash", "created_at"}
        }
    )
    core["protocol_version"] = "network_weak_signal_ensemble_v2"
    core["policy"] = asdict(Policy())
    core["research_history_boundary"] = {
        "2020_2023_market_returns_seen_by_unrelated_repo_research": True,
        "exact_nwe8_model_outcomes_opened": False,
        "nwe7_support_result_opened": True,
        "nwe7_return_labels_or_pnl_opened": False,
        "predecessor_support_only_failures": ["NTB-7", "BFC-3", "NWE-7"],
        "claim": (
            "NWE-8 changes only the first prediction date so the already frozen "
            "52-label causal warm-up can exist; no predecessor return was opened"
        ),
    }
    core["novelty_boundary"] = {
        "economic_axis": (
            "weekly price-independent network economics decoder combining weak "
            "blockspace-demand and ledger-topology states"
        ),
        "governance_change_from_nwe7": (
            "prediction_start moves from 2021-03-01 to 2021-06-07 solely because "
            "NWE-7 disclosed 41 available labels versus the frozen 52 minimum"
        ),
        "not": [
            "a parameter or feature search after seeing NWE-7 returns",
            "tail-event threshold repair of NTB-7 or BFC-3",
            "BTC price, return, volatility, extrema, wick, or volume as model input",
            "exchange-address flow tagging",
            "funding, premium, basis, OI, Kimchi, DXY, FX, REX, or Markov input",
            "spot/perp/aggTrade microstructure",
            "unconditional long drift",
        ],
    }
    core["support_freeze_before_labels"] = {
        "train_2021_2022_candidate_weeks_min": 80,
        "train_2021_candidate_weeks_min": 28,
        "train_2022_candidate_weeks_min": 50,
        "selection_2023_candidate_weeks_min": 50,
        "selection_2023_h1_candidate_weeks_min": 25,
        "selection_2023_h2_candidate_weeks_min": 25,
        "initial_fully_available_training_samples_min": 52,
        "all_feature_values_finite": True,
        "market_or_return_rows_loaded": 0,
        "failure_action": "reject without building return labels",
    }
    core["selection_protocol"] = copy.deepcopy(core["selection_protocol"])
    core["selection_protocol"]["train"] = ["2021-06-07", "2023-01-01"]
    core["selection_protocol"]["performance_gates"]["train_trade_count_min"] = 30
    core["controls"]["mechanism_rejection_rule"] = (
        "reject NWE-8 if stale or permuted features independently pass every "
        "primary gate; component-only results are attribution, not replacements"
    )
    core["orthogonality_after_performance"]["comparison_set"] = (
        "all promoted/live/shadow sleeves frozen before NWE-8"
    )
    core["rejection_contract"] = (
        "any support, performance, mechanism, or orthogonality failure rejects "
        "NWE-8 without changing features, ridge, window, abstention, side, or hold"
    )
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
        raise RuntimeError("NWE-8 preregistration hash mismatch")
    if payload.get("outcomes_opened") is not False:
        raise RuntimeError("NWE-8 preregistration cannot open outcomes")
    if payload.get("policy") != asdict(Policy()):
        raise RuntimeError("NWE-8 policy differs from code")
    if payload.get("feature_columns") != list(FEATURE_COLUMNS):
        raise RuntimeError("NWE-8 feature order differs from code")
    if payload.get("selection_protocol", {}).get("candidate_count") != 1:
        raise RuntimeError("NWE-8 must remain a singleton")
    if payload.get("causal_feature_contract", {}).get(
        "price_or_derivative_feature_columns_loaded"
    ) != []:
        raise RuntimeError("NWE-8 support clock must not load market features")
    history = payload.get("research_history_boundary", {})
    if history.get("nwe7_return_labels_or_pnl_opened") is not False:
        raise RuntimeError("NWE-8 warm-up repair requires sealed NWE-7 outcomes")


def write_once(path: str | Path, payload: dict[str, Any]) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = json.loads(output.read_text())
        validate_manifest(existing)
        if existing["manifest_hash"] != payload["manifest_hash"]:
            raise RuntimeError("refusing to overwrite frozen NWE-8 preregistration")
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
