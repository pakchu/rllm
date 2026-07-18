"""Freeze DACC-48 before constructing a signal clock or opening outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = (
    "results/delayed_aftershock_compression_continuation_preregistration_2026-07-18.json"
)
SOURCE_MANIFEST = (
    "data/binance_um_kline_reference_btc_2020_2023/build_manifest.json"
)
MARKET_SOURCE = (
    "data/binance_um_kline_reference_btc_2020_2023/"
    "BTCUSDT_5m_2020-01-01_2023-12-31.csv.gz"
)
FUNDING_SOURCE = "results/binance_um_btcusdt_realized_funding_2020_2023.csv"
FUNDING_MANIFEST = (
    "results/binance_um_btcusdt_realized_funding_2020_2023_manifest.json"
)


@dataclass(frozen=True)
class Policy:
    policy_id: str = "DACC-48"
    reference_bars: int = 8_640
    reference_min_periods: int = 2_016
    shock_abs_quantile: float = 0.995
    minimum_shock_return_bp: float = 40.0
    shock_scale_bars: int = 72
    minimum_shock_scale_multiple: float = 4.0
    minimum_shock_directional_taker_imbalance: float = 0.15
    compression_bars: int = 6
    maximum_compression_to_shock_ratio: float = 0.55
    maximum_compression_net_fraction: float = 0.20
    maximum_retrace_fraction: float = 0.30
    reacceleration_bars: int = 3
    minimum_break_margin_bp: float = 5.0
    minimum_reacceleration_return_bp: float = 15.0
    minimum_reacceleration_box_fraction: float = 0.30
    reacceleration_flow_quantile: float = 0.70
    minimum_directional_flow: float = 0.10
    minimum_flow_acceleration: float = 0.10
    minimum_range_acceleration: float = 1.50
    trigger_offset_bars: int = 9
    entry_offset_bars: int = 10
    delayed_entry_offset_bars: int = 11
    hold_bars: int = 48
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
        "protocol_version": "delayed_aftershock_compression_continuation_v1",
        "outcomes_opened": False,
        "policy": asdict(Policy()),
        "research_history_boundary": {
            "pre2024_market_returns_seen_by_unrelated_repo_research": True,
            "exact_dacc48_outcomes_opened": False,
            "claim": (
                "exact-policy mechanical freeze, not a globally pristine market "
                "holdout; 2024+ and live forward remain final generalization tests"
            ),
        },
        "novelty_boundary": {
            "economic_axis": (
                "an exceptional single-bar impulse is followed by a fixed 30-minute "
                "quiet compression and traded only after a separate 15-minute window "
                "breaks the box with same-direction flow and range reacceleration"
            ),
            "not": [
                "immediate realized-jump continuation",
                "rolling-extrema REX entry",
                "funding/premium carry or gate",
                "OI liquidation/crowding",
                "Kimchi/FX/DXY state",
                "cross-sectional alt rotation",
                "Markov/ExtraTrees/LLM prediction",
                "spot-perp wick or basis rejection",
            ],
        },
        "source_contract": {
            "market_manifest": SOURCE_MANIFEST,
            "market_manifest_sha256": (
                "c04fbbd299cc748a6745c0ef030787da4d560833c744c81c98dd8840efc7913e"
            ),
            "market": MARKET_SOURCE,
            "market_sha256": (
                "e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d"
            ),
            "market_rows": 420_768,
            "funding_manifest": FUNDING_MANIFEST,
            "funding_manifest_sha256": (
                "c70280e46bcbc2410cc59c2bcc93780c40997dbc5d0edb82d82127b59593250c"
            ),
            "funding": FUNDING_SOURCE,
            "funding_sha256": (
                "c19829fa085a50f29c13762373a2b6db1c62025d657be1f5a3fbb9ce254482f7"
            ),
            "funding_rows": 4_383,
            "interval": ["2020-01-01", "2024-01-01"],
            "bar_interval": "5min",
            "source": "official Binance USD-M archives and funding API backfill",
            "database_snapshot_is_point_in_time": False,
            "missing_bar_policy": "fail closed; never fill a missing execution bar",
        },
        "causal_feature_contract": {
            "bar_return": "r[t] = log(close[t] / close[t-1])",
            "shock_direction": "d = sign(r[j]), d != 0",
            "shock_scale": (
                "sigma72_pre[j] = sqrt((pi/2) * mean(abs(r[u])*abs(r[u-1]))) "
                "over u=j-72..j-1"
            ),
            "shock_threshold": (
                "abs(r[j]) >= max(40bp, strictly-prior rolling q99.5 of abs(r) "
                "over 8,640 clean observations with at least 2,016 prior observations)"
            ),
            "shock_confirmation": (
                "abs(r[j])/sigma72_pre[j] >= 4; d*taker_imbalance[j]>=0.15; "
                "quote_volume[j]>=strictly-prior rolling median quote volume"
            ),
            "compression_window": "completed bars j+1 through j+6 inclusive",
            "compression_width": (
                "box_width=log(max(high[j+1:j+6])/min(low[j+1:j+6])) "
                "<=0.55*abs(r[j])"
            ),
            "compression_net": (
                "abs(log(close[j+6]/close[j])) <= 0.20*abs(r[j])"
            ),
            "adverse_box": (
                "maximum adverse box excursion from close[j] <=0.30*abs(r[j])"
            ),
            "compression_flow": (
                "abs(sum(signed_quote[j+1:j+6])/sum(quote_volume[j+1:j+6])) <= "
                "strictly-prior median absolute six-bar flow"
            ),
            "reacceleration_window": "completed bars j+7 through j+9 inclusive",
            "reacceleration": (
                "directional close-to-close return >=max(15bp,0.30*box_width); "
                "close[j+9] clears the directional box edge by >=5bp; directional "
                "three-bar flow >=max(0.10, prior q70 absolute three-bar flow); "
                "directional flow minus compression flow >=0.10; mean true range "
                "over j+7:j+9 >=1.50 times mean true range over j+1:j+6"
            ),
            "taker_imbalance": (
                "(2*taker_buy_quote[t]-quote_asset_volume[t]) / "
                "quote_asset_volume[t]"
            ),
            "availability": (
                "the signal becomes known only after bar j+9 is complete; "
                "all rolling baselines exclude their current observation"
            ),
        },
        "execution_contract": {
            "decision": "after reacceleration bar j+9 closes",
            "entry": "next BTCUSDT perpetual open at j+10",
            "exit": "scheduled open 48 five-minute bars after entry",
            "hold": "240 minutes fixed",
            "nonoverlap": True,
            "stop_or_take_profit": None,
            "leverage": 0.5,
            "base_cost": "6bp/notional/side",
            "stress_cost": "10bp/notional/side",
            "funding_interval": "entry_time <= funding_time < exit_time",
            "cagr": "full wall-clock split including idle cash",
            "strict_mdd": (
                "global/pre-entry HWM, favorable-before-adverse held OHLC, funding, "
                "entry/exit/hypothetical-liquidation costs"
            ),
        },
        "support_freeze_before_returns": {
            "train_2020_2022_nonoverlap_min": 150,
            "each_train_year_min": 30,
            "selection_2023_min": 40,
            "selection_2023_h1_min": 15,
            "selection_2023_h2_min": 15,
            "each_side_share_range": [0.30, 0.70],
            "maximum_single_month_share": 0.15,
            "maximum_single_utc_week_share": 0.08,
            "baseline_exact_entry_jaccard_max": 0.02,
            "baseline_entry_within_six_hours_share_max": 0.25,
            "baseline_position_time_jaccard_max": 0.15,
            "baseline_families": [
                "jump_continuation_72_bidirectional_20260712",
                "jump_continuation_volume_clock_gate_20260712",
                "efficient_recovery_continuation_72_20260712",
            ],
            "failure_action": "reject without opening post-entry outcomes",
        },
        "selection_protocol": {
            "train": ["2020-01-01", "2023-01-01"],
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
                "train_and_2023_mean_gross_underlying_bp_min": 12.0,
                "train_and_2023_ten_bp_stress_positive": True,
                "2023_h1_and_h2_absolute_return_positive": True,
                "one_bar_delayed_entry_train_and_2023_positive": True,
            },
        },
        "controls": {
            "direction_flip": (
                "primary clock with opposite side; diagnostic only and never a repair"
            ),
            "immediate_shock": (
                "same frozen shock events entered at j+1 without compression or trigger"
            ),
            "compression_without_flow": (
                "same shock/compression/breakout but remove all flow requirements"
            ),
            "without_compression": (
                "same shock and reacceleration requirements with compression gates removed"
            ),
            "without_range": (
                "same shock/compression/flow requirements with breakout and range gates removed"
            ),
            "one_bar_delayed_entry": "primary side entered at j+11",
            "shock_time_shift_one_day": (
                "move the complete primary event geometry exactly 288 bars earlier"
            ),
            "mechanism_rejection_rule": (
                "reject DACC-48 if no-compression, no-flow, no-range, or one-day-shift "
                "control independently passes every primary performance gate"
            ),
        },
        "orthogonality_after_performance": {
            "support_clock_gate_already_required": True,
            "absolute_daily_pnl_pearson_max": 0.30,
            "minimum_nonzero_daily_pnl_days": 20,
            "marginal_portfolio_improvement_required": True,
            "undefined_metric": "fail_closed",
        },
        "rejection_contract": (
            "any support, performance, mechanism, or orthogonality failure rejects "
            "DACC-48 without changing its q99.5, windows, ratios, flow floor, direction, "
            "entry delay, hold, cost, or performance threshold"
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
        raise RuntimeError("DACC-48 preregistration hash mismatch")
    if payload.get("outcomes_opened") is not False:
        raise RuntimeError("DACC-48 preregistration cannot open outcomes")
    if payload.get("policy") != asdict(Policy()):
        raise RuntimeError("DACC-48 policy differs from code")
    if payload.get("selection_protocol", {}).get("candidate_count") != 1:
        raise RuntimeError("DACC-48 must remain a singleton")


def write_once(path: str | Path, payload: dict[str, Any]) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = json.loads(output.read_text())
        validate_manifest(existing)
        if existing["manifest_hash"] != payload["manifest_hash"]:
            raise RuntimeError("refusing to overwrite frozen DACC-48 preregistration")
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
