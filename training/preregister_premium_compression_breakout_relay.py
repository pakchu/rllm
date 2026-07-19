"""Freeze PCBR-12 before deriving clocks or opening BTC outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


CANDIDATE = "PCBR-12"
DEFAULT_OUTPUT = (
    "results/premium_compression_breakout_relay_preregistration_2026-07-19.json"
)
SOURCE_PATH = (
    "data/binance_um_premium_path_btc_2020_2026/"
    "BTCUSDT_premium_path_1m_2020-01-01_2026-06-30.csv.gz"
)
SOURCE_MANIFEST_PATH = "results/binance_um_premium_path_btc_2020_2026_manifest.json"
MARKET_PATH = (
    "data/binance_um_kline_reference_btc_2020_2023/"
    "BTCUSDT_5m_2020-01-01_2023-12-31.csv.gz"
)
MARKET_MANIFEST_PATH = (
    "data/binance_um_kline_reference_btc_2020_2023/build_manifest.json"
)
FUNDING_PATH = "data/binance_um_btcusdt_funding_marks_2020_2023.csv.gz"
FUNDING_MANIFEST_PATH = (
    "results/binance_um_btcusdt_funding_marks_2020_2023_manifest_2026-07-17.json"
)
SOURCE_HASHES = {
    "premium": "7fbaae1f85482b9fc9e148af357c7315e4e7fc4b4e3ae36c31f27545109f8aa9",
    "premium_manifest": (
        "821e84f2f03bf893a03d7904bf665b6fd7f6d38edd845d1a9c4eef384d1c1dd8"
    ),
    "market": "e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d",
    "market_manifest": (
        "c04fbbd299cc748a6745c0ef030787da4d560833c744c81c98dd8840efc7913e"
    ),
    "funding": "3284bbb6bb67946acb673c6b67459543e217f752589e1d47b6c7c3b659f733e6",
    "funding_manifest": (
        "a0b2d27e1aa8cf2d9ab8cb659b598ee0a6d7bd25401c9e10ae92d1a74415845b"
    ),
}


@dataclass(frozen=True)
class Policy:
    policy_id: str = CANDIDATE
    context_bars_5m: int = 24
    trigger_bars_5m: int = 2
    prior_window_bars_5m: int = 8_640
    prior_min_periods_5m: int = 8_208
    prior_nonoverlap_shift_bars_5m: int = 26
    compression_range_quantile: float = 0.25
    trigger_move_abs_quantile: float = 0.90
    trigger_efficiency_quantile: float = 0.70
    terminal_location_abs_min: float = 0.75
    entry_delay_bars_5m: int = 2
    hold_bars_5m: int = 12
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
        "protocol_version": "premium_compression_breakout_relay_v1",
        "as_of_date": "2026-07-19",
        "outcomes_opened": False,
        "policy": asdict(Policy()),
        "research_history_boundary": {
            "premium_source_used_by_prior_research": True,
            "prior_premium_families_seen": [
                "single_bar_premium_intrabar_shape",
                "PSR-30/6_failed_recenter",
            ],
            "exact_pcbr_post_entry_outcomes_opened": False,
            "candidate_count": 1,
            "threshold_grid": False,
            "direction_search": False,
            "hold_search": False,
            "forbidden_before_evaluator_freeze": [
                "BTCUSDT_execution_OHLC",
                "BTCUSDT_entry_to_exit_return",
                "funding_cash_flow",
                "strategy_PnL",
                "absolute_return",
                "CAGR",
                "strict_MDD",
                "win_rate",
            ],
        },
        "mechanism": {
            "claim": (
                "after two hours of unusually compressed premium-index motion, "
                "an efficient ten-minute break that closes outside the compression "
                "cage and remains pinned near its directional extreme represents "
                "new derivative demand rather than a failed excursion, and may relay "
                "into BTCUSDT after a full empty latency bucket"
            ),
            "why_not_psr": (
                "PSR required high-energy alternating motion that returned near its "
                "center and traded mean reversion. PCBR requires low-range context, "
                "efficient displacement, an outside-cage close, terminal persistence, "
                "and trades continuation"
            ),
            "why_not_price_breakout": (
                "the clock contains premium-index OHLC only; BTC price, volume, "
                "funding, OI and existing alpha state are excluded"
            ),
        },
        "source_contract": {
            "premium": SOURCE_PATH,
            "premium_sha256": SOURCE_HASHES["premium"],
            "premium_manifest": SOURCE_MANIFEST_PATH,
            "premium_manifest_sha256": SOURCE_HASHES["premium_manifest"],
            "execution_market": MARKET_PATH,
            "execution_market_sha256": SOURCE_HASHES["market"],
            "execution_market_manifest": MARKET_MANIFEST_PATH,
            "execution_market_manifest_sha256": SOURCE_HASHES["market_manifest"],
            "funding": FUNDING_PATH,
            "funding_sha256": SOURCE_HASHES["funding"],
            "funding_manifest": FUNDING_MANIFEST_PATH,
            "funding_manifest_sha256": SOURCE_HASHES["funding_manifest"],
            "premium_range": ["2020-01-01", "2026-07-01"],
            "gap_policy": (
                "all one-minute rows forming each five-minute bar, every 24-bar "
                "context, and both trigger bars must be source-valid; no filling"
            ),
            "execution_sources_may_not_be_opened_by_support_builder": True,
        },
        "causal_feature_contract": {
            "aggregation": (
                "five exact completed one-minute premium rows form one completed "
                "five-minute premium OHLC bar"
            ),
            "decision_time": "the close boundary after the second trigger bar",
            "context": "24 completed 5m bars in [T-130m,T-10m)",
            "trigger": "2 completed 5m bars in [T-10m,T)",
            "context_range": "max(context_high)-min(context_low)",
            "trigger_move": "trigger_last_close-trigger_first_open",
            "trigger_path_range": "sum(trigger_high-trigger_low)",
            "trigger_efficiency": "abs(trigger_move)/trigger_path_range",
            "terminal_location": (
                "2*(trigger_last_close-trigger_low)/(trigger_high-trigger_low)-1"
            ),
            "outside_cage": (
                "positive move closes above context_high; negative move closes below "
                "context_low"
            ),
            "strict_prior_thresholds": (
                "rolling 8,640 prior five-minute anchors with 8,208 valid values; "
                "threshold feature samples are shifted 26 bars so no reference path "
                "overlaps the current context or trigger"
            ),
            "setup": (
                "context_range<=prior q25; abs(trigger_move)>=prior q90; "
                "trigger_efficiency>=prior q70; outside_cage; "
                "sign(trigger_move)*terminal_location>=0.75; false-to-true onset"
            ),
            "action": "side=sign(trigger_move) on BTCUSDT USD-M perpetual",
            "forbidden_features": [
                "BTCUSDT_price_or_return",
                "volume",
                "funding",
                "open_interest",
                "macro_or_FX",
                "existing_alpha_or_portfolio_state",
                "future_premium",
            ],
        },
        "execution_contract": {
            "feature_available": "final one-minute source row is available at T+1s",
            "entry": "leave [T,T+5m) empty and enter the BTCUSDT open at T+10m",
            "hold": "12 five-minute bars / one hour",
            "nonoverlap": "one global BTCUSDT position over [entry,exit)",
            "sizing": "fixed 0.5x notional",
            "costs": "6bp/notional/side base; 10bp/notional/side stress; exact funding",
            "strict_mdd": (
                "global/pre-entry HWM, entry cost, conservative funding boundary "
                "marks, every held favorable-then-adverse 5m OHLC path, virtual "
                "adverse-mark exit cost, and actual exit cost"
            ),
            "cagr": "full declared calendar including idle seconds",
        },
        "splits": {
            "train": ["2020-03-01", "2023-01-01"],
            "test": ["2023-01-01", "2024-01-01"],
            "eval": ["2024-01-01", "2026-07-01"],
        },
        "support_gate": {
            "minimum_events": {"train": 180, "test": 60, "eval": 120},
            "minimum_each_side_share": 0.25,
            "maximum_month_share": {"train": 0.12, "test": 0.20, "eval": 0.15},
            "subperiod_minimums": {
                "2020_mar_dec": 30,
                "2021": 50,
                "2022": 50,
                "2023_h1": 20,
                "2023_h2": 20,
                "2024": 40,
                "2025": 40,
                "2026_h1": 20,
            },
            "psr_exact_entry_jaccard_max": 0.10,
            "psr_near_60m_containment_max": 0.25,
            "other_clock_exact_entry_jaccard_max": 0.10,
            "other_clock_near_6h_containment_max": 0.25,
        },
        "mechanism_controls": {
            "no_compression": "retain trigger, outside-cage and terminal gates only",
            "no_terminal_pin": "retain compression, trigger and outside-cage only",
            "no_outside_cage": "retain compression, trigger and terminal gates only",
            "direction_flip": "same primary entries with side multiplied by -1",
            "extra_latency_1h": "same signal and side with entry and exit delayed 12 bars",
            "deterministic_random_side": (
                "same entries with SHA256(policy_id|decision_time) side"
            ),
        },
        "outcome_gate": {
            "train_minimum_trades": 150,
            "test_minimum_trades": 60,
            "eval_minimum_trades": 100,
            "absolute_return_positive": True,
            "cagr_to_strict_mdd_min": 3.0,
            "strict_mdd_max_pct": 15.0,
            "weekly_cluster_signflip_p_max": 0.10,
            "each_subperiod_absolute_return_positive": True,
            "stress_absolute_return_positive": True,
            "stress_cagr_to_strict_mdd_min": 2.5,
            "train_mechanism_control_margin_min": 0.25,
            "sequential_opening": "train_then_test_then_eval_stop_on_first_failure",
        },
        "rejection_contract": {
            "support_failure": "reject without opening BTC execution outcomes",
            "train_failure": "reject and keep test/eval sealed",
            "later_failure": "reject without threshold direction latency or hold repair",
        },
        "rllm_boundary": {
            "standalone_alpha_is_formulaic": True,
            "llm_not_allowed_to_create_or_repair_signals": True,
            "future_role": (
                "after deterministic passage, an LLM may explain or abstain under a "
                "separately frozen policy"
            ),
        },
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate_manifest(
    manifest: dict[str, Any], *, verify_feature_source: bool = True
) -> None:
    if manifest.get("outcomes_opened") is not False:
        raise ValueError("PCBR-12 preregistration opened outcomes")
    if manifest.get("policy") != asdict(Policy()):
        raise ValueError("PCBR-12 policy changed")
    core = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    if manifest.get("manifest_hash") != canonical_hash(core):
        raise ValueError("PCBR-12 manifest hash mismatch")
    if verify_feature_source:
        source = manifest["source_contract"]
        if _sha256(source["premium"]) != source["premium_sha256"]:
            raise ValueError("PCBR-12 premium source hash mismatch")
        if _sha256(source["premium_manifest"]) != source["premium_manifest_sha256"]:
            raise ValueError("PCBR-12 premium source manifest hash mismatch")


def write_manifest(path: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    report = build_manifest()
    validate_manifest(report)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = write_manifest(args.output)
    print(json.dumps({"output": args.output, "manifest_hash": report["manifest_hash"]}))


if __name__ == "__main__":
    main()
