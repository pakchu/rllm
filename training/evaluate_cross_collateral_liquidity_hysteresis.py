"""One-shot calendar-2023 evaluation for frozen CCLH v1.

The preregistration owns the geometry, state machine, and candidate clock. This
module verifies every dependency before opening only the frozen 2023 windows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_cross_collateral_liquidity_vacuum as clv
from training import preregister_cross_collateral_liquidity_void_refill as clvr
from training.evaluate_metaorder_fragmentation_impact_curvature import (
    simulate_schedule,
)
from training.preregister_cross_collateral_liquidity_hysteresis import (
    Config as SignalConfig,
    _log_depth_slope,
    _quarterly_schedule,
    build_signal,
    cross_collateral_geometry,
    hysteresis_state_machine,
    support_summary,
    tolerant_event_jaccard,
)
from training.preregister_metaorder_fragmentation_impact_curvature import (
    nonoverlapping_schedule,
)


PREREGISTRATION_COMMIT = "46825c4"
PREREGISTRATION_SOURCE = Path(
    "training/preregister_cross_collateral_liquidity_hysteresis.py"
)
PREREGISTRATION_SOURCE_SHA256 = (
    "c555d2c567ecb2c5b4a2986a4a1a455521c79bdcf655f7c43a3dfc41187cbdd5"
)
PREREGISTRATION_DOCUMENT = Path(
    "docs/cross-collateral-liquidity-hysteresis-preregistration-2026-07-14.md"
)
PREREGISTRATION_DOCUMENT_SHA256 = (
    "eefc80011a16aaf6b76dc54934a7b9e3bfcff618c278ef3bd009e6d55e5a502d"
)
PREREGISTRATION_RESULT = Path(
    "results/cross_collateral_liquidity_hysteresis_support_2026-07-14.json"
)
PREREGISTRATION_RESULT_SHA256 = (
    "6144cc8ce63635219b35ebba185d3653a5ce466f25fd65c33b2c1916a92bfdcf"
)
FEATURE_SOURCE = Path(
    "training/preregister_cross_collateral_liquidity_void_refill.py"
)
FEATURE_SOURCE_SHA256 = (
    "8465af153f4e5a19299c7ee2b6104e7ea009feb8da80ad10d50cf49bddd7ad51"
)
CLV_SOURCE = Path("training/preregister_cross_collateral_liquidity_vacuum.py")
CLV_SOURCE_SHA256 = (
    "4ea113bb09cb9a00d295c49729c2c876bb6b7f90378fce1b63de6907d4a6b7d7"
)
CLV_SUPPORT = Path(
    "results/cross_collateral_liquidity_vacuum_support_2026-07-14.json"
)
CLV_SUPPORT_SHA256 = (
    "3c661a85e90a6191385b1b69fafac6f8dd146888b6937394493b3aa44096c8d9"
)
SCHEDULER_SOURCE = Path(
    "training/preregister_metaorder_fragmentation_impact_curvature.py"
)
SCHEDULER_SOURCE_SHA256 = (
    "51e99dbdc5ba13e6b4ac15e3915ec5b30e36dff89c1e5b31a5f3f7f272f01a59"
)
EXECUTION_SOURCE = Path(
    "training/evaluate_metaorder_fragmentation_impact_curvature.py"
)
EXECUTION_SOURCE_SHA256 = (
    "1589a52605386570485a7e6be3b8f3aa9439a498abb60eaa42272ac62d4cbed3"
)
TRADE_STATS_SOURCE = Path("training/strict_bar_backtest.py")
TRADE_STATS_SOURCE_SHA256 = (
    "3e95ad320d8869755afa1f4907d2d478200a3ebfc015e4eaeace0be0b15f9682"
)
DEPTH_MANIFEST = Path(
    "results/binance_cross_collateral_book_depth_btc_2023_manifest.json"
)
DEPTH_MANIFEST_SHA256 = (
    "95ec6e133dfcc7ed3c058538f380d24d98552c0a921fc24a679d247159a4f080"
)
MARKET_MANIFEST = Path(
    "data/binance_um_kline_reference_btc_2020_2023/build_manifest.json"
)
MARKET_MANIFEST_SHA256 = (
    "c04fbbd299cc748a6745c0ef030787da4d560833c744c81c98dd8840efc7913e"
)
EVALUATION_SOURCE = Path(
    "training/evaluate_cross_collateral_liquidity_hysteresis.py"
)

WINDOWS: dict[str, tuple[str, str]] = {
    "train2023_h1": ("2023-01-01", "2023-07-01"),
    "select2023_h2": ("2023-07-01", "2024-01-01"),
    "q1": ("2023-01-01", "2023-04-01"),
    "q2": ("2023-04-01", "2023-07-01"),
    "q3": ("2023-07-01", "2023-10-01"),
    "q4": ("2023-10-01", "2024-01-01"),
}
POLICY_NAMES = (
    "cclh",
    "reverse",
    "always_long",
    "always_short",
    "permuted_sign",
    "price_momentum",
    "clv_overlap_removed",
)
QUALIFICATION_CONTROLS = (
    "reverse",
    "always_long",
    "always_short",
    "permuted_sign",
    "price_momentum",
)
STRUCTURAL_DIAGNOSTICS = (
    "um_only",
    "cm_only",
    "pressure_only",
    "elasticity_only",
)


@dataclass(frozen=True)
class EvaluationConfig:
    output: str = (
        "results/cross_collateral_liquidity_hysteresis_selection_2026-07-14.json"
    )
    leverage: float = 0.5
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0001
    cluster_permutations: int = 100_000
    cluster_seed: int = 20_260_714
    sign_permutation_seed: int = 20_260_714
    price_momentum_bars: int = 12


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_preregistration() -> dict[str, Any]:
    for path, expected in (
        (PREREGISTRATION_SOURCE, PREREGISTRATION_SOURCE_SHA256),
        (PREREGISTRATION_DOCUMENT, PREREGISTRATION_DOCUMENT_SHA256),
        (PREREGISTRATION_RESULT, PREREGISTRATION_RESULT_SHA256),
        (FEATURE_SOURCE, FEATURE_SOURCE_SHA256),
        (CLV_SOURCE, CLV_SOURCE_SHA256),
        (CLV_SUPPORT, CLV_SUPPORT_SHA256),
        (SCHEDULER_SOURCE, SCHEDULER_SOURCE_SHA256),
        (EXECUTION_SOURCE, EXECUTION_SOURCE_SHA256),
        (TRADE_STATS_SOURCE, TRADE_STATS_SOURCE_SHA256),
        (DEPTH_MANIFEST, DEPTH_MANIFEST_SHA256),
        (MARKET_MANIFEST, MARKET_MANIFEST_SHA256),
    ):
        if _sha256(path) != expected:
            raise ValueError(f"frozen CCLH dependency changed: {path}")

    result = json.loads(PREREGISTRATION_RESULT.read_text())
    protocol = result.get("protocol", {})
    calibration = result.get("support_calibration", {})
    frozen = result.get("frozen_artifacts", {})
    if protocol.get("outcomes_opened_for_cclh") is not False:
        raise ValueError("CCLH preregistration opened outcomes")
    if protocol.get("support_rejected") is not False:
        raise ValueError("CCLH was support-rejected")
    if protocol.get("support_parameters_searched") is not False:
        raise ValueError("CCLH support unexpectedly searched parameters")
    if result.get("all_support_gates_pass") is not True:
        raise ValueError("CCLH support gates are not frozen as passing")
    if result.get("config") != asdict(SignalConfig()):
        raise ValueError("CCLH signal config differs from frozen support")
    if calibration != {
        "outcomes_opened_for_cclh": False,
        "parameters_searched": False,
        "all_parameters_fixed": True,
        "further_support_repairs_allowed": False,
    }:
        raise ValueError("CCLH support-calibration contract changed")
    if protocol.get("sealed_windows") != ["test2024", "eval2025", "ytd2026"]:
        raise ValueError("CCLH sealed-window contract changed")

    expected_frozen = {
        "preregistration_source_sha256": PREREGISTRATION_SOURCE_SHA256,
        "preregistration_document_sha256": PREREGISTRATION_DOCUMENT_SHA256,
        "feature_source_sha256": FEATURE_SOURCE_SHA256,
        "clv_source_sha256": CLV_SOURCE_SHA256,
        "clv_support_sha256": CLV_SUPPORT_SHA256,
        "scheduler_source_sha256": SCHEDULER_SOURCE_SHA256,
        "depth_manifest_sha256": DEPTH_MANIFEST_SHA256,
        "market_manifest_sha256": MARKET_MANIFEST_SHA256,
    }
    for key, expected in expected_frozen.items():
        if frozen.get(key) != expected:
            raise ValueError(f"CCLH frozen-artifact record changed: {key}")
    return result


def _clv_signal(market: pd.DataFrame, signal_cfg: SignalConfig) -> pd.DataFrame:
    cfg = clv.Config(
        depth_manifest=signal_cfg.depth_manifest,
        market_manifest=signal_cfg.market_manifest,
    )
    features = clvr.build_features(market, cfg)
    return clv.classify_vacuum(features, cfg)


def verify_signal_replay(
    signal: pd.DataFrame,
    clv_signal: pd.DataFrame,
    market: pd.DataFrame,
    cfg: SignalConfig,
    preregistration: dict[str, Any],
) -> None:
    if support_summary(signal, market, cfg) != preregistration.get("support"):
        raise ValueError("CCLH support replay differs from frozen artifact")
    if int(signal["candidate"].sum()) != preregistration.get(
        "raw_candidate_count"
    ):
        raise ValueError("CCLH raw candidate replay differs from artifact")
    schedule = _quarterly_schedule(signal, market)
    side_counts = {
        "long": int(schedule["side"].gt(0).sum()),
        "short": int(schedule["side"].lt(0).sum()),
    }
    if side_counts != preregistration.get("scheduled_side_counts"):
        raise ValueError("CCLH side-count replay differs from artifact")
    branch_counts = {
        name: int(value)
        for name, value in schedule["branch"].value_counts().items()
    }
    if branch_counts != preregistration.get("scheduled_branch_counts"):
        raise ValueError("CCLH branch replay differs from artifact")
    clv_schedule = _quarterly_schedule(clv_signal, market)
    overlap = tolerant_event_jaccard(
        schedule["signal_position"].astype(int).tolist(),
        clv_schedule["signal_position"].astype(int).tolist(),
        tolerance_bars=cfg.clv_overlap_tolerance_bars,
    )
    frozen_overlap = preregistration.get("clv_event_overlap", {})
    for key, value in overlap.items():
        if value != frozen_overlap.get(key):
            raise ValueError(f"CCLH CLV-overlap replay differs: {key}")


def _overlap_mask(
    positions: np.ndarray,
    other_positions: np.ndarray,
    tolerance_bars: int,
) -> np.ndarray:
    if len(other_positions) == 0:
        return np.zeros(len(positions), dtype=bool)
    return np.asarray(
        [
            bool(np.any(np.abs(other_positions - position) <= tolerance_bars))
            for position in positions
        ],
        dtype=bool,
    )


def policy_schedule(
    reserved_schedule: pd.DataFrame,
    market: pd.DataFrame,
    policy: str,
    *,
    permutation_seed: int,
    price_momentum_bars: int,
    clv_reserved_schedule: pd.DataFrame | None = None,
    clv_overlap_tolerance_bars: int = 12,
) -> pd.DataFrame:
    """Change actions only after the non-overlap CCLH clock is reserved."""
    if policy not in POLICY_NAMES:
        raise ValueError(f"unknown CCLH control policy: {policy}")
    output = reserved_schedule.copy()
    if output.empty:
        return output
    if not output["branch"].isin(
        ["bullish_hysteresis", "bearish_hysteresis"]
    ).all():
        raise ValueError("CCLH schedule contains an unknown branch")
    sides = output["side"].to_numpy(np.int8)
    if not np.isin(sides, [-1, 1]).all():
        raise ValueError("CCLH schedule contains an invalid side")

    if policy == "reverse":
        output["side"] = -sides
    elif policy == "always_long":
        output["side"] = np.ones(len(output), dtype=np.int8)
    elif policy == "always_short":
        output["side"] = -np.ones(len(output), dtype=np.int8)
    elif policy == "permuted_sign":
        rng = np.random.default_rng(permutation_seed)
        output["side"] = rng.permutation(sides)
    elif policy == "price_momentum":
        if price_momentum_bars < 1:
            raise ValueError("price momentum bars must be positive")
        positions = output["signal_position"].to_numpy(int)
        if np.any(positions < price_momentum_bars):
            raise ValueError("price momentum lacks causal lookback")
        closes = market["close"].to_numpy(float)
        momentum = np.sign(
            closes[positions] / closes[positions - price_momentum_bars] - 1.0
        ).astype(np.int8)
        output["side"] = momentum
        output = output.loc[output["side"].ne(0)].copy()
    elif policy == "clv_overlap_removed":
        if clv_reserved_schedule is None:
            raise ValueError("CLV schedule is required for overlap removal")
        overlap = _overlap_mask(
            output["signal_position"].to_numpy(int),
            clv_reserved_schedule["signal_position"].to_numpy(int),
            clv_overlap_tolerance_bars,
        )
        output = output.loc[~overlap].copy()
    return output.reset_index(drop=True)


def evaluate_policy(
    market: pd.DataFrame,
    reserved_schedule: pd.DataFrame,
    clv_reserved_schedule: pd.DataFrame,
    *,
    policy: str,
    start: str,
    end: str,
    cfg: EvaluationConfig,
    signal_cfg: SignalConfig,
) -> dict[str, Any]:
    schedule = policy_schedule(
        reserved_schedule,
        market,
        policy,
        permutation_seed=cfg.sign_permutation_seed,
        price_momentum_bars=cfg.price_momentum_bars,
        clv_reserved_schedule=clv_reserved_schedule,
        clv_overlap_tolerance_bars=signal_cfg.clv_overlap_tolerance_bars,
    )
    metrics = simulate_schedule(
        market,
        schedule,
        start=start,
        end=end,
        cfg=cfg,
    )
    metrics.pop("continuation_count", None)
    metrics.pop("fade_count", None)
    metrics["reserved_candidate_count"] = int(len(reserved_schedule))
    metrics["executed_candidate_count"] = int(len(schedule))
    return metrics


def _venue_geometry(frame: pd.DataFrame, venue: str) -> pd.DataFrame:
    if venue not in ("um", "cm"):
        raise ValueError("unknown CCLH structural venue")
    bids = frame[
        [f"{venue}_depth_m{distance}" for distance in range(1, 6)]
    ].to_numpy(float)
    asks = frame[
        [f"{venue}_depth_p{distance}" for distance in range(1, 6)]
    ].to_numpy(float)
    return pd.DataFrame(
        {
            "pressure": np.mean(np.log(bids) - np.log(asks), axis=1),
            "elasticity": _log_depth_slope(asks) - _log_depth_slope(bids),
        }
    )


def _signal_from_geometry(
    market: pd.DataFrame,
    pressure: pd.Series,
    elasticity: pd.Series,
    cfg: SignalConfig,
    *,
    branch: str,
) -> pd.DataFrame:
    clean = market["source_complete"].astype(bool)
    pressure_z = clvr.lagged_robust_zscore(
        pressure.where(clean),
        window=cfg.robust_baseline_bars,
        minimum=cfg.robust_min_periods,
    )
    elasticity_z = clvr.lagged_robust_zscore(
        elasticity.where(clean),
        window=cfg.robust_baseline_bars,
        minimum=cfg.robust_min_periods,
    )
    state = hysteresis_state_machine(pressure_z, elasticity_z, clean, cfg)
    side = state["event_side"].astype(np.int8)
    return pd.DataFrame(
        {
            "date": market["date"],
            "side": side,
            "branch": np.where(side.ne(0), branch, "none"),
            "hold_bars": np.where(side.ne(0), cfg.hold_bars, 0).astype(
                np.int16
            ),
        }
    )


def structural_signals(
    market: pd.DataFrame,
    cfg: SignalConfig,
) -> dict[str, pd.DataFrame]:
    um = _venue_geometry(market, "um")
    cm = _venue_geometry(market, "cm")
    cross = cross_collateral_geometry(market)
    return {
        "um_only": _signal_from_geometry(
            market,
            um["pressure"],
            um["elasticity"],
            cfg,
            branch="um_only",
        ),
        "cm_only": _signal_from_geometry(
            market,
            cm["pressure"],
            cm["elasticity"],
            cfg,
            branch="cm_only",
        ),
        "pressure_only": _signal_from_geometry(
            market,
            cross["cross_pressure"],
            cross["cross_pressure"],
            cfg,
            branch="pressure_only",
        ),
        "elasticity_only": _signal_from_geometry(
            market,
            cross["cross_elasticity"],
            cross["cross_elasticity"],
            cfg,
            branch="elasticity_only",
        ),
    }


def _qualification(windows: dict[str, Any]) -> dict[str, Any]:
    train = windows["train2023_h1"]["cclh"]
    select = windows["select2023_h2"]["cclh"]
    failures: list[str] = []
    for name, metrics in (
        ("train2023_h1", train),
        ("select2023_h2", select),
    ):
        if metrics["absolute_return_pct"] <= 0.0:
            failures.append(f"{name}: non-positive absolute return")
        if metrics["cagr_to_strict_mdd"] < 3.0:
            failures.append(f"{name}: CAGR/strict-MDD below 3")
        if metrics["strict_mdd_pct"] > 15.0:
            failures.append(f"{name}: strict MDD above 15%")
        if metrics["trade_count"] < 45:
            failures.append(f"{name}: fewer than 45 trades")
        if metrics["weekly_cluster_sign_flip"]["p_value_one_sided"] >= 0.10:
            failures.append(f"{name}: weekly-cluster p-value not below 0.10")

    for name in ("q1", "q2", "q3", "q4"):
        metrics = windows[name]["cclh"]
        if metrics["absolute_return_pct"] <= 0.0:
            failures.append(f"{name}: non-positive absolute return")
        if metrics["trade_count"] < 20:
            failures.append(f"{name}: fewer than 20 trades")

    cclh_min_ratio = min(
        train["cagr_to_strict_mdd"],
        select["cagr_to_strict_mdd"],
    )
    for control in QUALIFICATION_CONTROLS:
        control_min_ratio = min(
            windows["train2023_h1"][control]["cagr_to_strict_mdd"],
            windows["select2023_h2"][control]["cagr_to_strict_mdd"],
        )
        if cclh_min_ratio <= control_min_ratio:
            failures.append(
                "cclh: minimum train/select ratio does not beat " + control
            )
    return {
        "qualifies": not failures,
        "failures": failures,
        "cclh_min_train_select_ratio": float(cclh_min_ratio),
    }


def run_evaluation(cfg: EvaluationConfig) -> dict[str, Any]:
    preregistration = verify_preregistration()
    signal_cfg = SignalConfig()
    market, source = clvr.load_sources(signal_cfg)
    signal = build_signal(market, signal_cfg)
    clv_signal = _clv_signal(market, signal_cfg)
    verify_signal_replay(
        signal,
        clv_signal,
        market,
        signal_cfg,
        preregistration,
    )
    diagnostics = structural_signals(market, signal_cfg)

    windows: dict[str, Any] = {}
    structural_windows: dict[str, Any] = {}
    for window_name, (start, end) in WINDOWS.items():
        reserved = nonoverlapping_schedule(
            signal,
            market,
            start=start,
            end=end,
        )
        clv_reserved = nonoverlapping_schedule(
            clv_signal,
            market,
            start=start,
            end=end,
        )
        windows[window_name] = {
            policy: evaluate_policy(
                market,
                reserved,
                clv_reserved,
                policy=policy,
                start=start,
                end=end,
                cfg=cfg,
                signal_cfg=signal_cfg,
            )
            for policy in POLICY_NAMES
        }
        structural_windows[window_name] = {}
        for name, diagnostic_signal in diagnostics.items():
            diagnostic_schedule = nonoverlapping_schedule(
                diagnostic_signal,
                market,
                start=start,
                end=end,
            )
            metrics = simulate_schedule(
                market,
                diagnostic_schedule,
                start=start,
                end=end,
                cfg=cfg,
            )
            metrics.pop("continuation_count", None)
            metrics.pop("fade_count", None)
            structural_windows[window_name][name] = metrics

    qualification = _qualification(windows)
    return {
        "protocol": {
            "name": "CCLH v1 frozen calendar-2023 research evaluation",
            "preregistration_commit": PREREGISTRATION_COMMIT,
            "preregistration_source_sha256": PREREGISTRATION_SOURCE_SHA256,
            "preregistration_document_sha256": PREREGISTRATION_DOCUMENT_SHA256,
            "preregistration_result_sha256": PREREGISTRATION_RESULT_SHA256,
            "evaluation_source_sha256": _sha256(EVALUATION_SOURCE),
            "outcomes_opened_for_cclh": True,
            "opened_windows": list(WINDOWS),
            "sealed_windows": ["test2024", "eval2025", "ytd2026"],
            "signal_parameters_mutable": False,
            "candidate_clock_reserved_before_action_control": True,
            "different_clock_structural_diagnostics_can_win": False,
            "entry": "next 5m open after confirmed state transition",
            "exit": "scheduled open 144 completed 5m bars after entry",
            "strict_mdd": (
                "held path, favorable extreme first then adverse extreme"
            ),
            "cagr": "full wall-clock split including idle cash",
        },
        "evaluation_config": asdict(cfg),
        "signal_config": preregistration["config"],
        "source": source,
        "windows": windows,
        "structural_diagnostics": structural_windows,
        "qualification": qualification,
        "selection": {
            "selected_alpha": "cclh" if qualification["qualifies"] else None,
            "rejected": not qualification["qualifies"],
            "reason": (
                "passed every frozen 2023 development gate"
                if qualification["qualifies"]
                else "CCLH v1 failed at least one frozen 2023 gate"
            ),
        },
    }


def _headline(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: metrics[key]
        for key in (
            "absolute_return_pct",
            "cagr_pct",
            "strict_mdd_pct",
            "cagr_to_strict_mdd",
            "trade_count",
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=EvaluationConfig.output)
    args = parser.parse_args()
    cfg = EvaluationConfig(output=args.output)
    result = run_evaluation(cfg)
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "selection": result["selection"],
                "qualification": result["qualification"],
                "cclh": {
                    name: _headline(policies["cclh"])
                    for name, policies in result["windows"].items()
                },
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
