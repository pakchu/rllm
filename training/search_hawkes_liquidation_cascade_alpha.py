"""Search a causal self-exciting jump/OI-build BTC alpha and freeze it before OOS.

The signal treats standardized price jumps as a two-sided self-exciting point
process.  A directional excitation imbalance is tradable only while delayed
Binance open interest is still building.  Phase one physically truncates both
market and metrics sources before 2024.  The selected policy and prefix hashes
are written before 2024+ rows are loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.search_positioning_disagreement_alpha import (
    _attach_delayed_metrics,
    _future_extreme,
    _simulate_no_stop,
)
from training.search_positioning_hgb_path_alpha import _read_before


SELECTION_END = "2024-01-01"
PHASE1_WINDOWS = {
    "fit_2020q4": ("2020-10-15", "2021-01-01"),
    "fit_2021h1": ("2021-01-01", "2021-07-01"),
    "fit_2021h2": ("2021-07-01", "2022-01-01"),
    "fit_2022h1": ("2022-01-01", "2022-07-01"),
    "fit_2022h2": ("2022-07-01", "2023-01-01"),
    "fit": ("2020-10-15", "2023-01-01"),
    "select_2023h1": ("2023-01-01", "2023-07-01"),
    "select_2023h2": ("2023-07-01", "2024-01-01"),
    "select_2023": ("2023-01-01", "2024-01-01"),
}
ROBUST_SEGMENTS = (
    "fit_2020q4",
    "fit_2021h1",
    "fit_2021h2",
    "fit_2022h1",
    "fit_2022h2",
    "select_2023h1",
    "select_2023h2",
)
OOS_WINDOWS = {
    "test2024": ("2024-01-01", "2025-01-01"),
    "test2024h1": ("2024-01-01", "2024-07-01"),
    "test2024h2": ("2024-07-01", "2025-01-01"),
    "eval2025": ("2025-01-01", "2026-01-01"),
    "eval2025h1": ("2025-01-01", "2025-07-01"),
    "eval2025h2": ("2025-07-01", "2026-01-01"),
    "ytd2026": ("2026-01-01", "2026-06-02"),
    "ytd2026q1": ("2026-01-01", "2026-04-01"),
    "ytd2026q2": ("2026-04-01", "2026-06-02"),
    "combined_oos": ("2024-01-01", "2026-06-02"),
}
HASH_FEATURE_COLUMNS = (
    "hawkes_up_intensity",
    "hawkes_down_intensity",
    "hawkes_imbalance",
    "hawkes_intensity",
    "oi_change_288",
    "price_momentum_12",
)


@dataclass(frozen=True)
class HawkesCascadeConfig:
    input_csv: str = "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_oi.csv.gz"
    metrics_csv: str = "data/binance_um_metrics_BTCUSDT_5m_2020-09-01_2026-06-01.csv.gz"
    manifest_output: str = "results/hawkes_liquidation_cascade_frozen_manifest_2026-07-13.json"
    output: str = "results/hawkes_liquidation_cascade_alpha_scan_2026-07-13.json"
    selection_end: str = SELECTION_END
    top_n: int = 10
    leverage: float = 0.5
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0001
    metrics_tolerance: str = "5min"
    source_delay_bars: int = 1


def load_market(cfg: HawkesCascadeConfig, *, cutoff: str | None) -> pd.DataFrame:
    if cutoff is None:
        market = pd.read_csv(cfg.input_csv, compression="infer")
        metrics = pd.read_csv(cfg.metrics_csv, compression="infer")
    else:
        market = _read_before(cfg.input_csv, "date", cutoff)
        metrics = _read_before(cfg.metrics_csv, "create_time", cutoff)
    attached = _attach_delayed_metrics(
        market,
        metrics,
        tolerance=cfg.metrics_tolerance,
        delay_bars=cfg.source_delay_bars,
    )
    attached["date"] = pd.to_datetime(attached["date"], errors="raise")
    if cutoff is not None:
        boundary = pd.Timestamp(cutoff)
        if attached["date"].max() >= boundary:
            raise RuntimeError("market source was not physically truncated")
        source = pd.to_datetime(attached["positioning_source_time"], errors="coerce")
        if source.notna().any() and source.max() >= boundary:
            raise RuntimeError("positioning source was not physically truncated")
    return attached.reset_index(drop=True)


def build_hawkes_features(
    market: pd.DataFrame,
    *,
    jump_z_threshold: float,
    intensity_half_life_bars: int,
) -> pd.DataFrame:
    close = pd.to_numeric(market["close"], errors="coerce")
    returns = np.log(close / close.shift(1))
    prior_vol = returns.shift(1).rolling(2016, min_periods=576).std(ddof=0).replace(0.0, np.nan)
    standardized = returns / prior_vol
    up_jump = (standardized >= jump_z_threshold).astype(float).to_numpy()
    down_jump = (standardized <= -jump_z_threshold).astype(float).to_numpy()
    decay = float(np.exp(-np.log(2.0) / intensity_half_life_bars))
    up_intensity = np.zeros(len(market), dtype=float)
    down_intensity = np.zeros(len(market), dtype=float)
    for position in range(1, len(market)):
        up_intensity[position] = decay * up_intensity[position - 1] + up_jump[position]
        down_intensity[position] = decay * down_intensity[position - 1] + down_jump[position]
    total = up_intensity + down_intensity
    imbalance = (up_intensity - down_intensity) / (total + 1.0)
    open_interest = np.log(
        pd.to_numeric(market["sum_open_interest"], errors="coerce").where(lambda values: values > 0.0)
    )
    dates = pd.to_datetime(market["date"])
    return pd.DataFrame(
        {
            "hawkes_up_intensity": up_intensity,
            "hawkes_down_intensity": down_intensity,
            "hawkes_imbalance": imbalance,
            "hawkes_intensity": total,
            "oi_change_288": open_interest - open_interest.shift(288),
            "price_momentum_12": np.log(close / close.shift(12)),
            "decision_event": (dates.dt.minute == 0).to_numpy(bool),
        },
        index=market.index,
    ).replace([np.inf, -np.inf], np.nan)


def fit_abs_threshold(values: np.ndarray, fit_mask: np.ndarray, quantile: float) -> float:
    finite = np.isfinite(values) & fit_mask
    if finite.sum() < 10_000:
        raise ValueError("insufficient causal fit observations")
    return float(np.quantile(np.abs(values[finite]), quantile))


def build_signals(
    features: pd.DataFrame,
    *,
    threshold: float,
    mode: str,
    direction_flip: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    imbalance = features["hawkes_imbalance"].to_numpy(float)
    active = (
        features["decision_event"].to_numpy(bool)
        & np.isfinite(imbalance)
        & (np.abs(imbalance) >= threshold)
    )
    side = np.where(imbalance >= 0.0, 1, -1)
    oi_change = features["oi_change_288"].to_numpy(float)
    if mode == "build_follow":
        active &= oi_change > 0.0
    elif mode == "build_fade":
        active &= oi_change > 0.0
        side = -side
    elif mode == "unwind_follow":
        active &= oi_change < 0.0
    elif mode != "all_follow":
        raise KeyError(f"unknown signal mode: {mode}")
    if direction_flip:
        side = -side
    return active & (side > 0), active & (side < 0)


def _frame_hash(frame: pd.DataFrame) -> str:
    values = pd.util.hash_pandas_object(frame, index=False).to_numpy(np.uint64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def _signal_hash(long_signal: np.ndarray, short_signal: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(long_signal, np.uint8).tobytes())
    digest.update(np.asarray(short_signal, np.uint8).tobytes())
    return digest.hexdigest()


def _simulate_bank(
    market: pd.DataFrame,
    dates: pd.Series,
    long_signal: np.ndarray,
    short_signal: np.ndarray,
    *,
    hold: int,
    windows: dict[str, tuple[str, str]],
    leverage: float,
    fee_rate: float,
    slippage_rate: float,
    extremes: tuple[np.ndarray, np.ndarray],
) -> dict[str, dict[str, Any]]:
    return {
        name: _simulate_no_stop(
            market,
            dates,
            long_signal,
            short_signal,
            window=name,
            hold_bars=hold,
            stride_bars=1,
            leverage=leverage,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            extremes=extremes,
            windows=windows,
        )
        for name in windows
    }


def _candidate_identity(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in ("jump_z", "half_life", "tail", "mode", "hold")}


def run(cfg: HawkesCascadeConfig) -> dict[str, Any]:
    phase_market = load_market(cfg, cutoff=cfg.selection_end)
    phase_dates = pd.to_datetime(phase_market["date"])
    fit_mask = (
        (phase_dates >= pd.Timestamp(PHASE1_WINDOWS["fit"][0]))
        & (phase_dates < pd.Timestamp(PHASE1_WINDOWS["fit"][1]))
    ).to_numpy(bool)
    holds = (72, 144, 288)
    extremes = {
        hold: (
            _future_extreme(phase_market["low"].to_numpy(float), hold, "min"),
            _future_extreme(phase_market["high"].to_numpy(float), hold, "max"),
        )
        for hold in holds
    }
    candidates: list[dict[str, Any]] = []
    feature_banks: dict[tuple[float, int], pd.DataFrame] = {}
    for jump_z, half_life in itertools.product((2.0, 2.75), (12, 48, 144)):
        features = build_hawkes_features(
            phase_market,
            jump_z_threshold=jump_z,
            intensity_half_life_bars=half_life,
        )
        feature_banks[(jump_z, half_life)] = features
        values = features["hawkes_imbalance"].to_numpy(float)
        for tail, mode, hold in itertools.product(
            (0.8, 0.9),
            ("unwind_follow", "build_fade", "build_follow", "all_follow"),
            holds,
        ):
            threshold = fit_abs_threshold(values, fit_mask, tail)
            long_signal, short_signal = build_signals(features, threshold=threshold, mode=mode)
            stats = _simulate_bank(
                phase_market,
                phase_dates,
                long_signal,
                short_signal,
                hold=hold,
                windows=PHASE1_WINDOWS,
                leverage=cfg.leverage,
                fee_rate=cfg.fee_rate,
                slippage_rate=cfg.slippage_rate,
                extremes=extremes[hold],
            )
            if stats["select_2023"]["trades"] < 24:
                continue
            candidates.append(
                {
                    "jump_z": jump_z,
                    "half_life": half_life,
                    "tail": tail,
                    "mode": mode,
                    "hold": hold,
                    "threshold": threshold,
                    "positive_segments": sum(stats[name]["return_pct"] > 0.0 for name in ROBUST_SEGMENTS),
                    "min_segment_ratio": min(stats[name]["ratio"] for name in ROBUST_SEGMENTS),
                    "phase1": stats,
                }
            )
    candidates.sort(
        key=lambda row: (
            row["positive_segments"],
            min(row["phase1"]["fit"]["ratio"], row["phase1"]["select_2023"]["ratio"]),
            row["min_segment_ratio"],
            row["phase1"]["fit"]["trades"],
        ),
        reverse=True,
    )
    selected = candidates[: cfg.top_n]
    top = selected[0]
    top_features = feature_banks[(top["jump_z"], top["half_life"])]
    top_long, top_short = build_signals(top_features, threshold=top["threshold"], mode=top["mode"])
    exploratory_admission = bool(
        top["positive_segments"] >= 6
        and top["phase1"]["fit"]["return_pct"] > 0.0
        and top["phase1"]["select_2023"]["ratio"] >= 3.0
        and top["phase1"]["select_2023h1"]["ratio"] >= 3.0
        and top["phase1"]["select_2023h2"]["ratio"] >= 3.0
    )
    prefix_features = pd.concat(
        [phase_dates.rename("date"), top_features[list(HASH_FEATURE_COLUMNS)]],
        axis=1,
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": "frozen_before_future_open",
        "selection_cutoff": cfg.selection_end,
        "max_market_time": str(phase_dates.max()),
        "max_positioning_source_time": str(pd.to_datetime(phase_market["positioning_source_time"]).max()),
        "precut_market_hash": _frame_hash(phase_market[["date", "open", "high", "low", "close"]]),
        "precut_feature_hash": _frame_hash(prefix_features),
        "precut_signal_hash": _signal_hash(top_long, top_short),
        "policy": _candidate_identity(top) | {"threshold": top["threshold"]},
        "exploratory_admission": exploratory_admission,
        "admission_basis": top["phase1"],
        "future_windows_unopened": list(OOS_WINDOWS),
    }
    manifest_path = Path(cfg.manifest_output)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    if not exploratory_admission:
        report = {
            "as_of": datetime.now(timezone.utc).isoformat(),
            "config": asdict(cfg),
            "tested": len(candidates),
            "selected": selected,
            "manifest": manifest,
            "decision": "fail_pre2024_admission",
            "oos": None,
        }
    else:
        full_market = load_market(cfg, cutoff=None)
        full_dates = pd.to_datetime(full_market["date"])
        full_features = build_hawkes_features(
            full_market,
            jump_z_threshold=top["jump_z"],
            intensity_half_life_bars=top["half_life"],
        )
        prefix_n = len(phase_market)
        if _frame_hash(full_market.iloc[:prefix_n][["date", "open", "high", "low", "close"]]) != manifest["precut_market_hash"]:
            raise RuntimeError("market prefix changed after future rows were opened")
        check_features = pd.concat(
            [
                full_dates.iloc[:prefix_n].rename("date").reset_index(drop=True),
                full_features.iloc[:prefix_n][list(HASH_FEATURE_COLUMNS)].reset_index(drop=True),
            ],
            axis=1,
        )
        if _frame_hash(check_features) != manifest["precut_feature_hash"]:
            raise RuntimeError("feature prefix changed after future rows were opened")
        full_long, full_short = build_signals(full_features, threshold=top["threshold"], mode=top["mode"])
        if _signal_hash(full_long[:prefix_n], full_short[:prefix_n]) != manifest["precut_signal_hash"]:
            raise RuntimeError("signal prefix changed after future rows were opened")
        full_extremes = (
            _future_extreme(full_market["low"].to_numpy(float), top["hold"], "min"),
            _future_extreme(full_market["high"].to_numpy(float), top["hold"], "max"),
        )
        primary = _simulate_bank(
            full_market,
            full_dates,
            full_long,
            full_short,
            hold=top["hold"],
            windows=OOS_WINDOWS,
            leverage=cfg.leverage,
            fee_rate=cfg.fee_rate,
            slippage_rate=cfg.slippage_rate,
            extremes=full_extremes,
        )
        passes_live_oos = bool(
            primary["test2024"]["ratio"] >= 3.0
            and primary["eval2025"]["ratio"] >= 3.0
            and primary["ytd2026"]["return_pct"] > 0.0
            and primary["combined_oos"]["ratio"] >= 3.0
        )
        flip_long, flip_short = build_signals(
            full_features,
            threshold=top["threshold"],
            mode=top["mode"],
            direction_flip=True,
        )
        no_gate_long, no_gate_short = build_signals(
            full_features,
            threshold=top["threshold"],
            mode="all_follow",
        )
        oos = {
            "prefix_verified": True,
            "primary_6bp": primary,
            "stress_10bp": _simulate_bank(
                full_market,
                full_dates,
                full_long,
                full_short,
                hold=top["hold"],
                windows=OOS_WINDOWS,
                leverage=cfg.leverage,
                fee_rate=0.0008,
                slippage_rate=0.0002,
                extremes=full_extremes,
            ),
            "direction_flip_6bp": _simulate_bank(
                full_market,
                full_dates,
                flip_long,
                flip_short,
                hold=top["hold"],
                windows=OOS_WINDOWS,
                leverage=cfg.leverage,
                fee_rate=cfg.fee_rate,
                slippage_rate=cfg.slippage_rate,
                extremes=full_extremes,
            ),
            "no_oi_gate_6bp": _simulate_bank(
                full_market,
                full_dates,
                no_gate_long,
                no_gate_short,
                hold=top["hold"],
                windows=OOS_WINDOWS,
                leverage=cfg.leverage,
                fee_rate=cfg.fee_rate,
                slippage_rate=cfg.slippage_rate,
                extremes=full_extremes,
            ),
        }
        report = {
            "as_of": datetime.now(timezone.utc).isoformat(),
            "config": asdict(cfg),
            "tested": len(candidates),
            "selected": selected,
            "manifest": manifest,
            "decision": "candidate_oos" if passes_live_oos else "reject_oos",
            "oos": oos,
        }
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def parse_args() -> HawkesCascadeConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", default=HawkesCascadeConfig.input_csv)
    parser.add_argument("--metrics-csv", default=HawkesCascadeConfig.metrics_csv)
    parser.add_argument("--manifest-output", default=HawkesCascadeConfig.manifest_output)
    parser.add_argument("--output", default=HawkesCascadeConfig.output)
    parser.add_argument("--selection-end", default=HawkesCascadeConfig.selection_end)
    parser.add_argument("--top-n", type=int, default=HawkesCascadeConfig.top_n)
    return HawkesCascadeConfig(**vars(parser.parse_args()))


def main() -> None:
    report = run(parse_args())
    top = report["selected"][0]
    summary = {
        "tested": report["tested"],
        "decision": report["decision"],
        "policy": _candidate_identity(top),
        "fit": top["phase1"]["fit"],
        "select_2023": top["phase1"]["select_2023"],
        "oos": None if report["oos"] is None else report["oos"]["primary_6bp"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
