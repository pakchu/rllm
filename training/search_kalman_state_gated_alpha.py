"""Causal local-linear Kalman-state gate for the fixed funding/premium setup.

The filter is updated only with the completed hourly observation at ``t`` and
the strategy enters on the next 5-minute bar.  Kalman covariance is deliberately
not used as a regime feature: with fixed Q/R it converges deterministically and
does not contain market information.  The state is therefore the 3x3 cross of
filtered slope and standardized innovation.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import training.search_bidirectional_state_alpha as state_sim
from preprocessing.market_features import build_market_feature_frame
from training.long_regime_combo_scan import _load_market, _split_mask
from training.long_regime_interest_gate_validation import build_interest_features
from training.portfolio_opt_new_alpha_pool import _alpha_active, _event_path
from training.search_bidirectional_state_alpha import Config, sim
from training.search_gaussian_hmm_regime_alpha import SPLITS, hourly_features


def kalman_local_linear(
    log_price: np.ndarray,
    q_level: float,
    q_slope: float,
    r_obs: float,
    train_var: float,
) -> np.ndarray:
    """Return causal [level, slope, innovation_z, slope_z] filter outputs."""
    values = np.asarray(log_price, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("log_price must be a non-empty 1-D array")

    transition = np.array([[1.0, 1.0], [0.0, 1.0]])
    observation = np.array([1.0, 0.0])
    process_cov = np.diag([q_level * train_var, q_slope * train_var])
    observation_var = max(r_obs * train_var, 1e-12)
    state = np.array([values[0], 0.0])
    covariance = np.eye(2) * train_var * 100.0
    output = np.empty((len(values), 4), dtype=float)

    for idx, observed in enumerate(values):
        predicted_state = transition @ state
        predicted_cov = transition @ covariance @ transition.T + process_cov
        innovation = float(observed - observation @ predicted_state)
        innovation_var = float(observation @ predicted_cov @ observation + observation_var)
        gain = (predicted_cov @ observation) / innovation_var
        state = predicted_state + gain * innovation
        covariance = (np.eye(2) - np.outer(gain, observation)) @ predicted_cov
        output[idx] = (
            state[0],
            state[1],
            innovation / np.sqrt(innovation_var),
            state[1] / np.sqrt(max(covariance[1, 1], 1e-12)),
        )
    return output


def kalman_hourly_state(
    hourly: pd.DataFrame,
    train_mask: np.ndarray,
    *,
    q_level: float,
    q_slope: float,
    r_obs: float,
    low_quantile: float,
    high_quantile: float,
    train_var: float | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Build a train-frozen 3x3 slope/innovation state on completed hours."""
    log_price = np.log(hourly["close"].to_numpy(float))
    if train_var is None:
        train_returns = np.diff(log_price)[np.asarray(train_mask, dtype=bool)[1:]]
        train_var = float(np.nanvar(train_returns))
    filtered = kalman_local_linear(log_price, q_level, q_slope, r_obs, train_var)
    frame = pd.DataFrame(
        {
            "date": hourly.index.to_numpy(),
            "slope_z": filtered[:, 3],
            "innovation_z": filtered[:, 2],
        }
    )
    fit = frame.loc[np.asarray(train_mask, dtype=bool)]
    thresholds = {
        "slope_low": float(fit["slope_z"].quantile(low_quantile)),
        "slope_high": float(fit["slope_z"].quantile(high_quantile)),
        "innovation_low": float(fit["innovation_z"].quantile(low_quantile)),
        "innovation_high": float(fit["innovation_z"].quantile(high_quantile)),
    }
    slope_bucket = np.where(
        frame["slope_z"] <= thresholds["slope_low"],
        0,
        np.where(frame["slope_z"] >= thresholds["slope_high"], 2, 1),
    )
    innovation_bucket = np.where(
        frame["innovation_z"] <= thresholds["innovation_low"],
        0,
        np.where(frame["innovation_z"] >= thresholds["innovation_high"], 2, 1),
    )
    frame["state"] = slope_bucket * 3 + innovation_bucket
    return frame, thresholds


def map_hourly_state(dates: pd.Series, hourly_state: pd.DataFrame) -> np.ndarray:
    mapped = pd.merge_asof(
        pd.DataFrame({"date": pd.to_datetime(dates), "pos": np.arange(len(dates))}),
        hourly_state[["date", "state"]].sort_values("date"),
        on="date",
        direction="backward",
        tolerance=pd.Timedelta("2h"),
    ).sort_values("pos")
    return mapped["state"].fillna(-1).to_numpy(int)


def _rank_key(row: dict) -> tuple[float, float, float, float]:
    train_ratio = max(0.0, float(row["train"]["ratio"]))
    test_ratio = max(0.0, float(row["test2024"]["ratio"]))
    return (
        min(train_ratio, test_ratio),
        float(np.sqrt(train_ratio * test_ratio)),
        min(float(row["train"]["cagr_pct"]), float(row["test2024"]["cagr_pct"])),
        float(row["test2024"]["return_pct"]),
    )


def _signal_hash(signal: np.ndarray) -> str:
    return hashlib.sha256(np.packbits(signal).tobytes()).hexdigest()[:16]


def _state_descriptions(states: list[int]) -> dict[str, str]:
    names = ("low", "mid", "high")
    return {
        str(state): f"slope={names[state // 3]}, innovation={names[state % 3]}"
        for state in sorted(states)
    }


def frozen_winner_promotions(selected: list[dict]) -> tuple[list[dict], list[dict]]:
    """Only the pre-evaluation rank winner can be promoted."""
    if not selected:
        return [], []
    winner = selected[0]
    alpha_pool = [winner] if winner.get("passes_alpha_pool", False) else []
    live_grade = [winner] if winner.get("passes_live_grade", False) else []
    return alpha_pool, live_grade


def run(cfg: Config) -> dict:
    market = _load_market(cfg)
    dates = pd.to_datetime(market["date"])
    base = build_market_feature_frame(market, window_size=144)
    features = pd.concat([base, build_interest_features(market, base)], axis=1)
    features = features.loc[:, ~features.columns.duplicated(keep="last")]
    setup = _alpha_active(features, "long_minimal_funding_premium")
    hourly, _ = hourly_features(market)
    train_hour = np.asarray(
        (hourly.index >= SPLITS["train"][0]) & (hourly.index < SPLITS["train"][1]),
        dtype=bool,
    )
    train_market = _split_mask(dates, *SPLITS["train"])
    positions = np.arange(143, len(market) - 578, 12)
    log_price = np.log(hourly["close"].to_numpy(float))
    train_var = float(np.nanvar(np.diff(log_price)[train_hour[1:]]))
    raw_rows: list[dict] = []

    for q_level, q_slope, r_obs in itertools.product(
        (0.001, 0.01, 0.1, 1.0),
        (1e-5, 1e-4, 1e-3, 0.01),
        (0.25, 0.5, 1.0, 2.0, 4.0),
    ):
        for low, high in ((0.2, 0.8), (0.25, 0.75), (0.33, 0.67)):
            hourly_state, thresholds = kalman_hourly_state(
                hourly,
                train_hour,
                q_level=q_level,
                q_slope=q_slope,
                r_obs=r_obs,
                low_quantile=low,
                high_quantile=high,
                train_var=train_var,
            )
            state = map_hourly_state(dates, hourly_state)
            quality: dict[int, list[float]] = {}
            next_allowed = 0
            for pos in positions[setup[positions] & train_market[positions]]:
                if pos < next_allowed or state[pos] < 0:
                    continue
                event = _event_path(
                    market,
                    int(pos),
                    side="long",
                    hold=576,
                    cost_rate=0.0006,
                    entry_delay=1,
                    leverage=0.5,
                )
                if event is None:
                    continue
                quality.setdefault(int(state[pos]), []).append(float(event[2]))
                next_allowed = int(pos) + 577

            for min_count, min_edge in itertools.product((5, 8, 12, 16), (0.0, 0.002, 0.005, 0.01)):
                allowed = sorted(
                    state_id
                    for state_id, returns in quality.items()
                    if len(returns) >= min_count and float(np.mean(returns)) >= min_edge
                )
                if not allowed:
                    continue
                long_active = setup & np.isin(state, allowed)
                short_active = np.zeros(len(market), dtype=bool)
                train_stats = sim(market, dates, long_active, short_active, cfg, 576, 12, 10.0, 10.0, "train")
                test_stats = sim(market, dates, long_active, short_active, cfg, 576, 12, 10.0, 10.0, "test2024")
                if train_stats["trades"] < 80 or test_stats["trades"] < 10:
                    continue
                raw_rows.append(
                    {
                        "q_level": q_level,
                        "q_slope": q_slope,
                        "r_obs": r_obs,
                        "state_quantiles": [low, high],
                        "state_thresholds": thresholds,
                        "min_train_state_trades": min_count,
                        "min_train_trade_edge": min_edge,
                        "allowed_states": allowed,
                        "allowed_state_descriptions": _state_descriptions(allowed),
                        "state_quality": {
                            str(state_id): {
                                "n": len(quality[state_id]),
                                "mean_trade_return": float(np.mean(quality[state_id])),
                            }
                            for state_id in allowed
                        },
                        "train": train_stats,
                        "test2024": test_stats,
                        "signal_hash": _signal_hash(long_active),
                        "_packed_signal": np.packbits(long_active),
                    }
                )

    raw_rows.sort(key=_rank_key, reverse=True)
    distinct_rows: list[dict] = []
    seen_signals: set[str] = set()
    for row in raw_rows:
        if row["signal_hash"] in seen_signals:
            continue
        seen_signals.add(row["signal_hash"])
        distinct_rows.append(row)
    selected = distinct_rows[:100]

    for row in selected:
        long_active = np.unpackbits(row.pop("_packed_signal"), count=len(market)).astype(bool)
        short_active = np.zeros(len(market), dtype=bool)
        for split in ("eval2025", "ytd2026"):
            row[split] = sim(market, dates, long_active, short_active, cfg, 576, 12, 10.0, 10.0, split)
        row["robust_train_test_score"] = _rank_key(row)[0]
        row["passes_alpha_pool"] = bool(
            row["train"]["ratio"] >= 1.0
            and row["test2024"]["ratio"] >= 3.0
            and row["eval2025"]["ratio"] >= 3.0
            and row["eval2025"]["trades"] >= 10
        )
        row["passes_live_grade"] = bool(
            row["passes_alpha_pool"]
            and row["ytd2026"]["ratio"] >= 5.0
            and row["ytd2026"]["trades"] >= 6
        )

    baseline = {
        split: sim(market, dates, setup, np.zeros(len(market), dtype=bool), cfg, 576, 12, 10.0, 10.0, split)
        for split in SPLITS
    }
    yearly: dict[str, dict] = {}
    stress: dict[str, dict] = {}
    leave_one: dict[str, dict] = {}
    if selected:
        top = selected[0]
        hourly_state, _ = kalman_hourly_state(
            hourly,
            train_hour,
            q_level=top["q_level"],
            q_slope=top["q_slope"],
            r_obs=top["r_obs"],
            low_quantile=top["state_quantiles"][0],
            high_quantile=top["state_quantiles"][1],
            train_var=train_var,
        )
        state = map_hourly_state(dates, hourly_state)
        top_active = setup & np.isin(state, top["allowed_states"])
        short_active = np.zeros(len(market), dtype=bool)
        for year in range(2020, 2027):
            state_sim.W["yearly"] = (
                f"{year}-01-01",
                f"{year + 1}-01-01" if year < 2026 else "2026-06-02",
            )
            yearly[str(year)] = sim(
                market, dates, top_active, short_active, cfg, 576, 12, 10.0, 10.0, "yearly"
            )
        for bps in (6, 8, 10, 15):
            stressed_cfg = replace(cfg, fee_rate=max(0.0, bps / 10000 - cfg.slippage_rate))
            stress[str(bps)] = {
                split: sim(
                    market, dates, top_active, short_active, stressed_cfg, 576, 12, 10.0, 10.0, split
                )
                for split in SPLITS
            }
        for dropped in top["allowed_states"]:
            active = setup & np.isin(state, [x for x in top["allowed_states"] if x != dropped])
            leave_one[str(dropped)] = {
                split: sim(market, dates, active, short_active, cfg, 576, 12, 10.0, 10.0, split)
                for split in SPLITS
            }

    alpha_pool, live_grade = frozen_winner_promotions(selected)
    output = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "base_setup": "long_minimal_funding_premium",
        "protocol": (
            "causal local-linear Kalman filter; 3x3 slope/innovation states; parameters, bins, "
            "and state quality fit on train; variants ranked by min(train ratio, test2024 ratio); "
            "eval2025/2026 report-only diagnostics; hold576; 6bp/side; strict intrabar MDD"
        ),
        "selection_caveat": (
            "The base setup and later windows were seen in prior research. This is a research-forward "
            "shadow candidate, not pristine untouched OOS evidence."
        ),
        "tested_variants": len(raw_rows),
        "distinct_signal_variants": len(distinct_rows),
        "baseline": baseline,
        "yearly_top_candidate": yearly,
        "cost_stress_bps_per_side": stress,
        "leave_one_state_out": leave_one,
        "selected": selected,
        "diagnostic_later_window_passers": {
            "alpha_pool": sum(bool(row["passes_alpha_pool"]) for row in selected[1:]),
            "live_grade": sum(bool(row["passes_live_grade"]) for row in selected[1:]),
        },
        "alpha_pool_qualifiers": alpha_pool,
        "live_grade": live_grade,
    }
    Path(cfg.output).write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
            default=lambda value: value.item() if isinstance(value, np.generic) else str(value),
        )
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--funding-csv", default="")
    parser.add_argument("--premium-csv", default="")
    parser.add_argument("--exclude-from", default="2026-06-02")
    args = parser.parse_args()
    output = run(Config(**vars(args)))
    print(
        json.dumps(
            {
                "tested_variants": output["tested_variants"],
                "distinct_signal_variants": output["distinct_signal_variants"],
                "qualifiers": len(output["alpha_pool_qualifiers"]),
                "live": len(output["live_grade"]),
                "top": output["selected"][:5],
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
