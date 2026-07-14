"""Support-only preregistration for CLVR.

CLVR compares how BTC liquidity reshapes in Binance's stablecoin-margined and
coin-margined perpetual books after a completed directional shock. This module
contains no future-return, PnL, CAGR, or drawdown calculation.
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

from training.preregister_metaorder_fragmentation_impact_curvature import (
    nonoverlapping_schedule,
)


SELECTION_END = pd.Timestamp("2024-01-01")
SUPPORT_CALIBRATION_GRID = (0.90, 0.925, 0.95, 0.975, 0.99, 0.995)
PREREGISTRATION_SOURCE = Path(
    "training/preregister_cross_collateral_liquidity_void_refill.py"
)
PREREGISTRATION_DOCUMENT = Path(
    "docs/cross-collateral-liquidity-void-refill-preregistration-2026-07-14.md"
)
SCHEDULER_SOURCE = Path(
    "training/preregister_metaorder_fragmentation_impact_curvature.py"
)


@dataclass(frozen=True)
class Config:
    depth_manifest: str = (
        "results/binance_cross_collateral_book_depth_btc_2023_manifest.json"
    )
    market_manifest: str = (
        "data/binance_um_kline_reference_btc_2020_2023/build_manifest.json"
    )
    output: str = (
        "results/cross_collateral_liquidity_void_refill_support_2026-07-14.json"
    )
    response_bars: int = 6
    hold_bars: int = 12
    robust_baseline_bars: int = 8_640
    robust_min_periods: int = 2_016
    score_baseline_bars: int = 17_280
    score_min_periods: int = 4_032
    score_quantile: float = 0.975
    minimum_nonoverlap_total: int = 400
    minimum_nonoverlap_per_half: int = 180
    minimum_nonoverlap_per_quarter: int = 75
    minimum_side_share: float = 0.25
    minimum_branch_share: float = 0.25


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _validate_grid(
    frame: pd.DataFrame,
    *,
    start: str,
    end: str,
    label: str,
) -> None:
    if frame["date"].duplicated().any() or not frame["date"].is_monotonic_increasing:
        raise ValueError(f"{label} timestamps are invalid")
    expected = pd.date_range(start, end, freq="5min", inclusive="left")
    if not pd.DatetimeIndex(frame["date"]).equals(expected):
        raise ValueError(f"{label} is not a complete 5m grid")


def load_sources(cfg: Config) -> tuple[pd.DataFrame, dict[str, Any]]:
    depth_manifest = json.loads(Path(cfg.depth_manifest).read_text())
    if depth_manifest.get("protocol", {}).get("outcomes_opened") is not False:
        raise ValueError("depth manifest opened outcomes")
    if depth_manifest.get("protocol", {}).get("post_2023_rows_requested") is not False:
        raise ValueError("depth manifest requested post-2023 rows")
    depth_item = depth_manifest.get("file", {})
    depth_path = Path(depth_item.get("path", ""))
    if not depth_path.is_file() or _sha256(depth_path) != depth_item.get("sha256"):
        raise ValueError("cross-collateral depth hash mismatch")
    depth = pd.read_csv(depth_path, compression="gzip", parse_dates=["date"])
    _validate_grid(
        depth,
        start="2023-01-01",
        end="2024-01-01",
        label="cross-collateral depth",
    )
    required_depth = [
        f"{venue}_depth_{side}{distance}"
        for venue in ("um", "cm")
        for side in ("m", "p")
        for distance in range(1, 6)
    ]
    if not set(required_depth).issubset(depth.columns):
        raise ValueError("cross-collateral depth columns are incomplete")
    complete = depth["source_complete"]
    if complete.dtype != bool:
        complete = complete.astype("string").str.lower().map(
            {"true": True, "false": False}
        )
    if complete.isna().any():
        raise ValueError("source_complete contains an unknown value")
    depth["source_complete"] = complete.astype(bool)
    if depth.loc[depth["source_complete"], required_depth].isna().any().any():
        raise ValueError("complete depth row contains a missing level")

    market_manifest = json.loads(Path(cfg.market_manifest).read_text())
    if market_manifest.get("protocol", {}).get("outcomes_opened") is not False:
        raise ValueError("market manifest opened outcomes")
    market_path = Path(market_manifest.get("combined_output", ""))
    if not market_path.is_file() or _sha256(market_path) != market_manifest.get(
        "combined_sha256"
    ):
        raise ValueError("execution market hash mismatch")
    market = pd.read_csv(market_path, compression="gzip", parse_dates=["date"])
    market = market.loc[
        market["date"].ge("2023-01-01") & market["date"].lt(SELECTION_END)
    ].reset_index(drop=True)
    _validate_grid(market, start="2023-01-01", end="2024-01-01", label="market")
    frame = market.merge(depth, on="date", validate="one_to_one")
    frame["quarantined"] = False
    metadata = {
        "depth_manifest_sha256": _sha256(cfg.depth_manifest),
        "depth_sha256": _sha256(depth_path),
        "market_manifest_sha256": _sha256(cfg.market_manifest),
        "market_sha256": _sha256(market_path),
        "range_start": "2023-01-01 00:00:00",
        "range_end": "2023-12-31 23:55:00",
        "source_complete_rows": int(frame["source_complete"].sum()),
    }
    return frame, metadata


def lagged_robust_zscore(
    values: pd.Series,
    *,
    window: int,
    minimum: int,
) -> pd.Series:
    if not 1 <= minimum <= window:
        raise ValueError("robust baseline periods are invalid")
    prior = values.astype(float).shift(1)
    center = prior.rolling(window, min_periods=minimum).median()
    mad = (prior - center).abs().rolling(window, min_periods=minimum).median()
    return ((values.astype(float) - center) / (1.4826 * mad.replace(0.0, np.nan))).clip(
        -12.0,
        12.0,
    )


def build_features(frame: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if cfg.response_bars < 1:
        raise ValueError("response bars must be positive")
    output = pd.DataFrame({"date": frame["date"]})
    geometry: dict[str, pd.Series] = {}
    for venue in ("um", "cm"):
        bid_near = frame[f"{venue}_depth_m1"].astype(float)
        bid_far = frame[f"{venue}_depth_m5"].astype(float)
        ask_near = frame[f"{venue}_depth_p1"].astype(float)
        ask_far = frame[f"{venue}_depth_p5"].astype(float)
        geometry[f"{venue}_level"] = np.log(bid_near / ask_near)
        geometry[f"{venue}_shape"] = np.log(
            (bid_near / bid_far) / (ask_near / ask_far)
        )

    cross_level = geometry["cm_level"] - geometry["um_level"]
    cross_shape = geometry["cm_shape"] - geometry["um_shape"]
    close = frame["close"].astype(float)
    response_return = np.log(close / close.shift(cfg.response_bars))
    direction = np.sign(response_return).fillna(0.0).astype(np.int8)
    level_change = cross_level - cross_level.shift(cfg.response_bars)
    shape_change = cross_shape - cross_shape.shift(cfg.response_bars)
    level_response = direction.astype(float) * level_change
    shape_response = direction.astype(float) * shape_change

    quote_volume = frame["quote_asset_volume"].astype(float)
    signed_flow = 2.0 * frame["taker_buy_quote"].astype(float) - quote_volume
    response_flow = signed_flow.rolling(
        cfg.response_bars,
        min_periods=cfg.response_bars,
    ).sum()
    clean = (
        frame["source_complete"]
        .astype(bool)
        .rolling(cfg.response_bars + 1, min_periods=cfg.response_bars + 1)
        .sum()
        .eq(cfg.response_bars + 1)
    )

    output["cross_level"] = cross_level
    output["cross_shape"] = cross_shape
    output["response_return"] = response_return
    output["direction"] = direction
    output["response_flow"] = response_flow
    output["flow_aligned"] = direction.astype(float) * response_flow > 0.0
    output["level_response"] = level_response
    output["shape_response"] = shape_response
    output["clean"] = clean
    for name in ("response_return", "level_response", "shape_response"):
        output[f"{name}_z"] = lagged_robust_zscore(
            output[name].where(clean),
            window=cfg.robust_baseline_bars,
            minimum=cfg.robust_min_periods,
        )
    return output


def classify_features(
    features: pd.DataFrame,
    cfg: Config,
    *,
    score_quantile: float | None = None,
) -> pd.DataFrame:
    quantile = cfg.score_quantile if score_quantile is None else score_quantile
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("score quantile must be in [0, 1]")
    return_z = features["response_return_z"].astype(float)
    level_z = features["level_response_z"].astype(float)
    shape_z = features["shape_response_z"].astype(float)
    available = (
        features["clean"].astype(bool)
        & features["flow_aligned"].astype(bool)
        & features["direction"].ne(0)
        & pd.concat([return_z, level_z, shape_z], axis=1).notna().all(axis=1)
    )
    response_agreement = (
        np.sign(level_z) == np.sign(shape_z)
    ) & level_z.ne(0.0) & shape_z.ne(0.0)
    score = (return_z.abs() * level_z.abs() * shape_z.abs()) ** (1.0 / 3.0)
    eligible = available & response_agreement
    baseline = (
        score.where(eligible)
        .shift(1)
        .rolling(
            cfg.score_baseline_bars,
            min_periods=cfg.score_min_periods,
        )
        .quantile(quantile)
    )
    candidate = eligible & score.ge(baseline)
    void = candidate & level_z.gt(0.0)
    refill = candidate & level_z.lt(0.0)
    side = pd.Series(0, index=features.index, dtype=np.int8)
    side.loc[void] = features.loc[void, "direction"].astype(np.int8)
    side.loc[refill] = -features.loc[refill, "direction"].astype(np.int8)
    branch = pd.Series("none", index=features.index, dtype="string")
    branch.loc[void] = "void"
    branch.loc[refill] = "refill"
    hold = np.where(side.ne(0), cfg.hold_bars, 0).astype(np.int16)
    return pd.DataFrame(
        {
            "date": features["date"],
            "candidate": candidate,
            "direction": features["direction"],
            "response_return_z": return_z,
            "level_response_z": level_z,
            "shape_response_z": shape_z,
            "joint_score": score,
            "score_baseline": baseline,
            "side": side,
            "branch": branch,
            "hold_bars": hold,
        }
    )


def support_summary(
    signal: pd.DataFrame,
    market: pd.DataFrame,
    cfg: Config,
) -> dict[str, Any]:
    periods = {
        "q1": ("2023-01-01", "2023-04-01"),
        "q2": ("2023-04-01", "2023-07-01"),
        "q3": ("2023-07-01", "2023-10-01"),
        "q4": ("2023-10-01", "2024-01-01"),
    }
    quarterly = {
        name: nonoverlapping_schedule(signal, market, start=start, end=end)
        for name, (start, end) in periods.items()
    }
    schedule = pd.concat(quarterly.values(), ignore_index=True)
    h1 = nonoverlapping_schedule(
        signal,
        market,
        start="2023-01-01",
        end="2023-07-01",
    )
    h2 = nonoverlapping_schedule(
        signal,
        market,
        start="2023-07-01",
        end="2024-01-01",
    )
    total = len(schedule)
    by_quarter = {name: len(rows) for name, rows in quarterly.items()}
    long_share = float(schedule["side"].gt(0).mean()) if total else 0.0
    short_share = float(schedule["side"].lt(0).mean()) if total else 0.0
    void_share = float(schedule["branch"].eq("void").mean()) if total else 0.0
    refill_share = float(schedule["branch"].eq("refill").mean()) if total else 0.0
    passes = (
        total >= cfg.minimum_nonoverlap_total
        and len(h1) >= cfg.minimum_nonoverlap_per_half
        and len(h2) >= cfg.minimum_nonoverlap_per_half
        and all(
            value >= cfg.minimum_nonoverlap_per_quarter
            for value in by_quarter.values()
        )
        and min(long_share, short_share) >= cfg.minimum_side_share
        and min(void_share, refill_share) >= cfg.minimum_branch_share
    )
    return {
        "nonoverlap_total": int(total),
        "by_quarter": by_quarter,
        "h1": int(len(h1)),
        "h2": int(len(h2)),
        "long_share": long_share,
        "short_share": short_share,
        "void_share": void_share,
        "refill_share": refill_share,
        "passes_support": bool(passes),
    }


def _selected_support_quantile(trials: list[dict[str, Any]]) -> float | None:
    passing = [
        float(trial["score_quantile"])
        for trial in trials
        if trial["passes_support"]
    ]
    return max(passing) if passing else None


def run_support(cfg: Config) -> dict[str, Any]:
    market, source = load_sources(cfg)
    features = build_features(market, cfg)
    trials: list[dict[str, Any]] = []
    selected_signal: pd.DataFrame | None = None
    selected_support: dict[str, Any] | None = None
    for quantile in SUPPORT_CALIBRATION_GRID:
        signal = classify_features(features, cfg, score_quantile=quantile)
        support = support_summary(signal, market, cfg)
        trials.append(
            {
                "score_quantile": quantile,
                "raw_candidate_count": int(signal["candidate"].sum()),
                **support,
            }
        )
        if quantile == cfg.score_quantile:
            selected_signal = signal
            selected_support = support
    selected = _selected_support_quantile(trials)
    if selected is not None and selected != cfg.score_quantile:
        raise ValueError("configured CLVR quantile violates support stopping rule")
    if selected is not None and (
        selected_signal is None or selected_support is None
    ):
        raise AssertionError("configured CLVR support was not evaluated")

    if selected is None:
        schedule = pd.DataFrame(columns=["branch"])
    else:
        schedule = pd.concat(
            [
                nonoverlapping_schedule(
                    selected_signal,
                    market,
                    start=start,
                    end=end,
                )
                for start, end in (
                    ("2023-01-01", "2023-04-01"),
                    ("2023-04-01", "2023-07-01"),
                    ("2023-07-01", "2023-10-01"),
                    ("2023-10-01", "2024-01-01"),
                )
            ],
            ignore_index=True,
        )
    return {
        "protocol": {
            "name": "CLVR — Cross-Collateral Liquidity Void-Refill",
            "support_only": True,
            "outcomes_opened_for_clvr": False,
            "support_rejected": selected is None,
            "selection_end_exclusive": "2024-01-01 00:00:00",
            "event_clock": "completed 5m USD-M/COIN-M depth bar after a 30m shock",
            "signal_availability": "depth medians and market bar complete; enter next 5m open",
            "branch_rule": {
                "void": "coin-margined stress-side liquidity depletes relative to USD-M; follow shock",
                "refill": "coin-margined stress-side liquidity replenishes relative to USD-M; fade shock",
            },
            "candidate_clock": "fixed before any outcome; both branches share one shock clock",
            "holding_rule": "12 completed 5m bars; scheduled-open exit",
            "source_gap_policy": "current and prior six depth bars must be complete; future depth gaps do not cancel an already entered trade",
            "sealed_windows": ["test2024", "eval2025", "ytd2026"],
        },
        "config": asdict(cfg),
        "frozen_artifacts": {
            "preregistration_source": str(PREREGISTRATION_SOURCE),
            "preregistration_source_sha256": _sha256(PREREGISTRATION_SOURCE),
            "preregistration_document": str(PREREGISTRATION_DOCUMENT),
            "preregistration_document_sha256": _sha256(
                PREREGISTRATION_DOCUMENT
            ),
            "scheduler_source": str(SCHEDULER_SOURCE),
            "scheduler_source_sha256": _sha256(SCHEDULER_SOURCE),
            "depth_manifest_sha256": _sha256(cfg.depth_manifest),
            "market_manifest_sha256": _sha256(cfg.market_manifest),
        },
        "source": source,
        "feature": {
            "response_window_bars": cfg.response_bars,
            "clean_feature_rows": int(features["clean"].sum()),
            "flow_aligned_rows": int(
                (features["clean"] & features["flow_aligned"]).sum()
            ),
            "standardization": (
                "strictly lagged rolling median and recursive MAD; clip [-12, 12]"
            ),
        },
        "support_calibration": {
            "outcomes_opened_for_clvr": False,
            "tested_score_quantiles": list(SUPPORT_CALIBRATION_GRID),
            "all_other_parameters_fixed": True,
            "stopping_rule": (
                "highest tested quantile passing every frozen support floor"
            ),
            "selected_score_quantile": selected,
            "further_support_repairs_allowed": False,
            "trials": trials,
        },
        "raw_candidate_count": (
            int(selected_signal["candidate"].sum())
            if selected_signal is not None and selected is not None
            else None
        ),
        "scheduled_branch_counts": {
            name: int(value)
            for name, value in schedule["branch"].value_counts().items()
        },
        "support": selected_support if selected is not None else None,
        "all_support_gates_pass": bool(
            selected_support is not None
            and selected_support["passes_support"]
            and selected is not None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=Config.output)
    args = parser.parse_args()
    result = run_support(Config(output=args.output))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "outcomes_opened_for_clvr": False,
                "selected_score_quantile": result["support_calibration"][
                    "selected_score_quantile"
                ],
                "support_rejected": result["protocol"]["support_rejected"],
                "support": result["support"],
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
