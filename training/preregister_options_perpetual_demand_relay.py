"""Freeze OPDR-24 before opening any 2024+ candidate outcome."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


CANDIDATE = "OPDR-24"
DEFAULT_OUTPUT = (
    "results/options_perpetual_demand_relay_preregistration_2026-07-19.json"
)
BVOL_PRE2024_PATH = (
    "/home/pakchu/rllm/data/binance_btc_bvol_hourly/"
    "BTCBVOLUSDT_1h_2023-06-20_2023-12-31.csv.gz"
)
BVOL_PRE2024_MANIFEST = (
    "/home/pakchu/rllm/data/binance_btc_bvol_hourly/build_manifest.json"
)
DVOL_PATH = (
    "/home/pakchu/rllm/data/deribit_btc_dvol_1h_2020-09-01_2026-06-02.csv.gz"
)
PREMIUM_PATH = (
    "data/binance_um_premium_path_btc_2020_2026/"
    "BTCUSDT_premium_path_1m_2020-01-01_2026-06-30.csv.gz"
)
PREMIUM_MANIFEST = "results/binance_um_premium_path_btc_2020_2026_manifest.json"
SOURCE_HASHES = {
    "bvol_pre2024": (
        "9581e879e8db4bc82cefc4d8c90558144677432cc1bb6175718e477682357375"
    ),
    "bvol_pre2024_manifest": (
        "852eedef81e566be0c120666d4f2995d2304351eca4b282d0990555a3993e496"
    ),
    "dvol": "b200fc84900152eeb09fdebde73632e65cd37024f293875b3fd3891ee8871aa6",
    "premium": "7fbaae1f85482b9fc9e148af357c7315e4e7fc4b4e3ae36c31f27545109f8aa9",
    "premium_manifest": (
        "821e84f2f03bf893a03d7904bf665b6fd7f6d38edd845d1a9c4eef384d1c1dd8"
    ),
    "bvol_builder": (
        "c686f3a4c4284c9aa61233fa16bfd20c439803a94eb8c9c67e45e6460e6ad0bd"
    ),
    "dvol_builder": (
        "7a9f4f567b85265bb2ed1e70fb18aa86c3b4e96cbed4afee8a95a612bac514c2"
    ),
}


@dataclass(frozen=True)
class Policy:
    policy_id: str = CANDIDATE
    premium_minutes_per_hour: int = 60
    prior_window_hours: int = 720
    prior_min_periods_hours: int = 672
    bvol_dvol_ratio_low_quantile: float = 0.20
    premium_move_abs_quantile: float = 0.80
    premium_efficiency_quantile: float = 0.70
    entry_delay_minutes: int = 5
    hold_hours: int = 24
    leverage: float = 0.5
    base_cost_notional_per_side: float = 0.0006
    stress_cost_notional_per_side: float = 0.0010


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_manifest() -> dict[str, Any]:
    core: dict[str, Any] = {
        "protocol_version": "options_perpetual_demand_relay_v1",
        "as_of_date": "2026-07-19",
        "outcomes_opened": False,
        "policy": asdict(Policy()),
        "research_history_boundary": {
            "train_window": ["2023-07-01", "2024-01-01"],
            "development_outcomes_seen_for_other_families": True,
            "known_development_result": (
                "the separate dvol_rich_move_follow_v80_p80_h48 price-follow "
                "candidate reached 2.47 CAGR/strict-MDD on 29 2023H2 trades "
                "and was rejected"
            ),
            "premium_families_support_seen": ["PSR-30/6", "PCBR-12"],
            "exact_opdr_post_entry_outcomes_opened": False,
            "candidate_count": 1,
            "threshold_grid": False,
            "direction_search": False,
            "hold_search": False,
            "sealed_candidate_windows": [
                "train_2023_h2",
                "test_2024",
                "eval_2025",
                "final_2026_h1",
            ],
        },
        "mechanism": {
            "claim": (
                "when Deribit DVOL is unusually rich to Binance BVOL, an "
                "efficient one-hour BTCUSDT premium-index displacement identifies "
                "the direction in which perpetual demand is transmitting the "
                "options-market uncertainty repricing into BTC"
            ),
            "action": "side equals the sign of the completed one-hour premium move",
            "why_distinct_from_dvol_price_follow": (
                "OPDR excludes BTC price and return from its clock and takes its "
                "direction solely from the premium-index path"
            ),
            "why_distinct_from_psr_pcbr": (
                "OPDR has no recenter, compression cage, outside breakout, or "
                "terminal-pin condition; the options-volatility disagreement is "
                "a required independent state variable"
            ),
        },
        "source_contract": {
            "bvol_pre2024": BVOL_PRE2024_PATH,
            "bvol_pre2024_sha256": SOURCE_HASHES["bvol_pre2024"],
            "bvol_pre2024_manifest": BVOL_PRE2024_MANIFEST,
            "bvol_pre2024_manifest_sha256": SOURCE_HASHES[
                "bvol_pre2024_manifest"
            ],
            "dvol": DVOL_PATH,
            "dvol_sha256": SOURCE_HASHES["dvol"],
            "premium": PREMIUM_PATH,
            "premium_sha256": SOURCE_HASHES["premium"],
            "premium_manifest": PREMIUM_MANIFEST,
            "premium_manifest_sha256": SOURCE_HASHES["premium_manifest"],
            "future_bvol_acquisition": {
                "builder": "training/build_binance_bvol_hourly.py",
                "builder_sha256": SOURCE_HASHES["bvol_builder"],
                "official_archive_root": (
                    "https://data.binance.vision/data/option/daily/BVOLIndex"
                ),
                "symbol": "BTCBVOLUSDT",
                "start": "2023-06-20",
                "end_exclusive": "2026-07-01",
                "may_open_only_after_preregistration_commit": True,
                "missing_or_unverified_archives": "invalid; never imputed",
            },
            "future_dvol_acquisition": {
                "builder": "training/download_deribit_volatility_index.py",
                "builder_sha256": SOURCE_HASHES["dvol_builder"],
                "official_endpoint": (
                    "https://www.deribit.com/api/v2/public/"
                    "get_volatility_index_data"
                ),
                "currency": "BTC",
                "resolution_seconds": 3600,
                "start": "2023-06-20",
                "end": "2026-07-01",
                "may_open_only_after_preregistration_commit": True,
                "availability": "join on close_time, never candle open time",
            },
            "clock_forbidden_fields": [
                "BTCUSDT_price",
                "BTCUSDT_return",
                "volume",
                "funding",
                "open_interest",
                "macro_or_FX",
                "existing_alpha_state",
                "future_premium",
            ],
        },
        "causal_feature_contract": {
            "hour_label": "T is the boundary after a completed UTC hour",
            "bvol": "the BTCBVOLUSDT hourly close available at T",
            "dvol": "the Deribit DVOL hourly close with close_time exactly T",
            "vol_ratio": "log(BVOL_close / DVOL_close)",
            "premium_hour": (
                "60 exact source-valid one-minute premium-index bars in [T-1h,T)"
            ),
            "premium_move": "last premium close minus first premium open, in bp",
            "premium_path_range": "sum of the 60 one-minute high-low ranges, in bp",
            "premium_efficiency": "abs(premium_move) / premium_path_range",
            "strict_prior_thresholds": (
                "each feature threshold uses only the preceding 720 completed "
                "hourly anchors, excludes the current hour, and requires 672 valid "
                "joint observations"
            ),
            "setup": (
                "vol_ratio<=prior q20; abs(premium_move)>=prior q80; "
                "premium_efficiency>=prior q70; false-to-true onset only"
            ),
            "feature_available_time": (
                "max(BVOL T, DVOL T, final premium minute T+1s) = T+1s"
            ),
            "entry": "BTCUSDT open at T+5m",
        },
        "execution_contract": {
            "instrument": "BTCUSDT USD-M perpetual",
            "hold": "24 elapsed hours, fixed; no stop, take-profit, or dynamic exit",
            "nonoverlap": "one global position over [entry,exit)",
            "sizing": "fixed 0.5x notional",
            "costs": "6bp/notional/side base; 10bp/notional/side stress; exact funding",
            "strict_mdd": (
                "global/pre-entry HWM, entry cost, conservative funding boundary "
                "marks, every held favorable-then-adverse 5m OHLC path, virtual "
                "adverse-mark exit cost, and actual exit cost"
            ),
            "cagr": "full declared calendar including warm-up and idle seconds",
        },
        "splits": {
            "train": ["2023-07-01", "2024-01-01"],
            "test": ["2024-01-01", "2025-01-01"],
            "eval": ["2025-01-01", "2026-01-01"],
            "final": ["2026-01-01", "2026-07-01"],
        },
        "support_gate": {
            "minimum_events": {
                "train": 20,
                "2023_q3": 6,
                "2023_q4": 8,
                "test": 40,
                "eval": 40,
                "final": 20,
            },
            "minimum_each_side_share": 0.25,
            "maximum_month_share": {
                "train": 0.35,
                "test": 0.20,
                "eval": 0.20,
                "final": 0.30,
            },
            "old_price_follow_exact_entry_jaccard_max": 0.80,
            "old_price_follow_near_1h_containment_max": 0.80,
            "premium_family_near_6h_containment_max": 0.35,
            "cmsr_near_6h_containment_max": 0.35,
        },
        "mechanism_controls": {
            "no_vol_disagreement": "retain premium displacement and efficiency only",
            "no_premium_efficiency": "retain vol disagreement and premium displacement",
            "dvol_poor_mirror": "replace ratio<=q20 with ratio>=q80",
            "direction_flip": "same primary entries with side multiplied by -1",
            "extra_latency_1h": "same signal and side with entry and exit delayed one hour",
            "deterministic_random_side": (
                "same entries with SHA256(policy_id|decision_time) side"
            ),
        },
        "outcome_gate": {
            "minimum_trades": {
                "train": 20,
                "test": 40,
                "eval": 40,
                "final": 20,
            },
            "absolute_return_positive": True,
            "cagr_to_strict_mdd_min": 3.0,
            "strict_mdd_max_pct": 15.0,
            "weekly_cluster_signflip_p_max": 0.10,
            "mean_gross_underlying_move_bp_min": 20.0,
            "each_half_absolute_return_positive": True,
            "stress_absolute_return_positive": True,
            "stress_cagr_to_strict_mdd_min": 2.5,
            "mechanism_control_margin_min": 0.25,
            "sequential_opening": (
                "train_then_test_then_eval_then_final_stop_on_first_failure"
            ),
        },
        "rejection_contract": {
            "support_failure": "reject without opening candidate outcomes",
            "train_failure": "reject and keep test/eval/final sealed",
            "test_failure": "reject and keep eval/final sealed",
            "later_failure": (
                "reject without threshold, direction, latency, hold, feature, or "
                "support-gate repair"
            ),
        },
        "rllm_boundary": {
            "standalone_alpha_is_formulaic": True,
            "llm_not_allowed_to_create_or_repair_signals": True,
            "future_role": (
                "after deterministic passage, an LLM may reason over the frozen "
                "symbolic vol-disagreement and premium-path state to abstain or size"
            ),
        },
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate_manifest(
    manifest: dict[str, Any], *, verify_sources: bool = True
) -> None:
    if manifest.get("outcomes_opened") is not False:
        raise ValueError("OPDR-24 preregistration opened outcomes")
    if manifest.get("policy") != asdict(Policy()):
        raise ValueError("OPDR-24 policy changed")
    history = manifest.get("research_history_boundary", {})
    if history.get("candidate_count") != 1:
        raise ValueError("OPDR-24 is not a singleton")
    if history.get("threshold_grid") is not False:
        raise ValueError("OPDR-24 contains a threshold grid")
    core = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    if manifest.get("manifest_hash") != canonical_hash(core):
        raise ValueError("OPDR-24 manifest hash mismatch")
    if verify_sources:
        source = manifest["source_contract"]
        checks = (
            (source["bvol_pre2024"], source["bvol_pre2024_sha256"]),
            (
                source["bvol_pre2024_manifest"],
                source["bvol_pre2024_manifest_sha256"],
            ),
            (source["dvol"], source["dvol_sha256"]),
            (source["premium"], source["premium_sha256"]),
            (source["premium_manifest"], source["premium_manifest_sha256"]),
            (
                source["future_bvol_acquisition"]["builder"],
                source["future_bvol_acquisition"]["builder_sha256"],
            ),
            (
                source["future_dvol_acquisition"]["builder"],
                source["future_dvol_acquisition"]["builder_sha256"],
            ),
        )
        for path, expected in checks:
            if _sha256(path) != expected:
                raise ValueError(f"OPDR-24 source changed: {path}")


def write_manifest(path: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    report = build_manifest()
    validate_manifest(report)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = write_manifest(args.output)
    print(json.dumps({"output": args.output, "manifest_hash": report["manifest_hash"]}))


if __name__ == "__main__":
    main()
