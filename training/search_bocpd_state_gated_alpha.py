"""Causal Bayesian-online-change-point gate for the funding/premium setup.

The implementation follows the Adams-MacKay run-length recursion with an
independent Normal-Gamma predictive model per input dimension.  Only filtered
posteriors from completed hourly observations are exposed to the strategy.
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
from scipy.special import gammaln

import training.search_bidirectional_state_alpha as state_sim
from preprocessing.market_features import build_market_feature_frame
from training.long_regime_combo_scan import _load_market, _split_mask
from training.long_regime_interest_gate_validation import build_interest_features
from training.portfolio_opt_new_alpha_pool import _alpha_active, _event_path
from training.search_bidirectional_state_alpha import Config, sim
from training.search_gaussian_hmm_regime_alpha import SPLITS, hourly_features


def _student_t_log_predictive(
    observation: np.ndarray,
    mean: np.ndarray,
    kappa: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    degrees = 2.0 * alpha
    scale2 = beta * (kappa[:, None] + 1.0) / (alpha * kappa[:, None])
    centered2 = (observation[None, :] - mean) ** 2
    per_dimension = (
        gammaln((degrees + 1.0) / 2.0)
        - gammaln(degrees / 2.0)
        - 0.5 * (np.log(degrees * np.pi) + np.log(scale2))
        - 0.5 * (degrees + 1.0) * np.log1p(centered2 / (degrees * scale2))
    )
    return per_dimension.sum(axis=1)


def _posterior_update(
    observation: np.ndarray,
    mean: np.ndarray,
    kappa: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    next_kappa = kappa + 1.0
    next_mean = (kappa[:, None] * mean + observation[None, :]) / next_kappa[:, None]
    next_alpha = alpha + 0.5
    next_beta = beta + 0.5 * (
        kappa[:, None] * (observation[None, :] - mean) ** 2 / next_kappa[:, None]
    )
    return next_mean, next_kappa, next_alpha, next_beta


def bocpd_student_t(
    observations: np.ndarray,
    *,
    hazard_lambda: float,
    max_run_length: int = 1000,
    prior_kappa: float = 0.1,
    prior_alpha: float = 2.0,
    prior_beta: float = 1.0,
    short_run_horizon: int = 6,
) -> dict[str, np.ndarray]:
    """Return causal BOCPD diagnostics for standardized observations.

    The run-length tail is truncated at ``max_run_length``.  ``short_mass`` is
    used instead of P(r_t=0), because a constant hazard makes the latter nearly
    constant and therefore uninformative as a market-state feature.
    """
    values = np.asarray(observations, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("observations must be a non-empty 1-D or 2-D array")
    if not np.isfinite(values).all():
        raise ValueError("observations must be finite")
    if hazard_lambda <= 1.0 or max_run_length < 2:
        raise ValueError("invalid BOCPD hazard or run-length cap")

    dimensions = values.shape[1]
    hazard = 1.0 / float(hazard_lambda)
    weights = np.array([1.0])
    mean = np.zeros((1, dimensions), dtype=float)
    kappa = np.array([prior_kappa], dtype=float)
    alpha = np.full((1, dimensions), prior_alpha, dtype=float)
    beta = np.full((1, dimensions), prior_beta, dtype=float)

    expected_run = np.empty(len(values), dtype=float)
    map_run = np.empty(len(values), dtype=float)
    short_mass = np.empty(len(values), dtype=float)
    run_drop = np.empty(len(values), dtype=float)
    surprise = np.empty(len(values), dtype=float)
    posterior_mean = np.empty((len(values), dimensions), dtype=float)
    previous_expected = 0.0

    prior_mean = np.zeros((1, dimensions), dtype=float)
    prior_kappa_array = np.array([prior_kappa], dtype=float)
    prior_alpha_array = np.full((1, dimensions), prior_alpha, dtype=float)
    prior_beta_array = np.full((1, dimensions), prior_beta, dtype=float)

    for idx, observation in enumerate(values):
        log_predictive = _student_t_log_predictive(observation, mean, kappa, alpha, beta)
        log_joint = np.log(np.maximum(weights, 1e-300)) + log_predictive
        offset = float(np.max(log_joint))
        joint = np.exp(log_joint - offset)
        surprise[idx] = -(offset + np.log(np.sum(joint)))

        reset_probability = hazard * float(np.sum(joint))
        growth_probability = (1.0 - hazard) * joint
        next_weights = np.r_[reset_probability, growth_probability]

        reset_params = _posterior_update(
            observation,
            prior_mean,
            prior_kappa_array,
            prior_alpha_array,
            prior_beta_array,
        )
        growth_params = _posterior_update(observation, mean, kappa, alpha, beta)
        next_mean = np.vstack([reset_params[0], growth_params[0]])
        next_kappa = np.r_[reset_params[1], growth_params[1]]
        next_alpha = np.vstack([reset_params[2], growth_params[2]])
        next_beta = np.vstack([reset_params[3], growth_params[3]])

        keep = min(len(next_weights), max_run_length + 1)
        weights = next_weights[:keep]
        weights /= np.sum(weights)
        mean = next_mean[:keep]
        kappa = next_kappa[:keep]
        alpha = next_alpha[:keep]
        beta = next_beta[:keep]

        run_axis = np.arange(keep, dtype=float)
        current_expected = float(weights @ run_axis)
        expected_run[idx] = current_expected
        map_run[idx] = float(np.argmax(weights))
        short_mass[idx] = float(weights[: min(short_run_horizon + 1, keep)].sum())
        expected_without_reset = previous_expected + 1.0
        run_drop[idx] = max(0.0, expected_without_reset - current_expected) / max(
            expected_without_reset, 1.0
        )
        posterior_mean[idx] = weights @ mean
        previous_expected = current_expected

    return {
        "expected_run": expected_run,
        "map_run": map_run,
        "short_mass": short_mass,
        "run_drop": run_drop,
        "surprise": surprise,
        "posterior_mean": posterior_mean,
    }


def _bucket3(values: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.where(values <= low, 0, np.where(values >= high, 2, 1))


def _signal_hash(signal: np.ndarray) -> str:
    return hashlib.sha256(np.packbits(signal).tobytes()).hexdigest()[:16]


def _rank_key(row: dict) -> tuple[float, float, float, float]:
    train_ratio = max(0.0, float(row["train"]["ratio"]))
    test_ratio = max(0.0, float(row["test2024"]["ratio"]))
    return (
        min(train_ratio, test_ratio),
        float(np.sqrt(train_ratio * test_ratio)),
        min(float(row["train"]["cagr_pct"]), float(row["test2024"]["cagr_pct"])),
        float(row["test2024"]["return_pct"]),
    )


def frozen_winner_promotions(selected: list[dict]) -> tuple[list[dict], list[dict]]:
    """Only the pre-evaluation rank winner can be promoted."""
    if not selected:
        return [], []
    winner = selected[0]
    alpha_pool = [winner] if winner.get("passes_alpha_pool", False) else []
    live_grade = [winner] if winner.get("passes_live_grade", False) else []
    return alpha_pool, live_grade


def _describe_state(state: int, primary_name: str, secondary_name: str) -> str:
    tri = ("low", "mid", "high")
    primary = state // 4
    reset = (state % 4) // 2
    secondary = state % 2
    return (
        f"{primary_name}={tri[primary]}, short_run_mass={'high' if reset else 'low'}, "
        f"{secondary_name}={'high' if secondary else 'low'}"
    )


def _model_output(
    feature_frame: pd.DataFrame,
    train_mask: np.ndarray,
    *,
    columns: tuple[str, ...],
    secondary_index: int | None,
    hazard_lambda: int,
) -> tuple[pd.DataFrame, dict]:
    good = feature_frame[list(columns)].notna().all(axis=1).to_numpy()
    fit_mask = good & np.asarray(train_mask, dtype=bool)
    raw_train = feature_frame.loc[fit_mask, list(columns)].to_numpy(float)
    mean = raw_train.mean(axis=0)
    std = raw_train.std(axis=0)
    std[std < 1e-8] = 1.0
    standardized = ((feature_frame.loc[good, list(columns)].to_numpy(float) - mean) / std).clip(-12, 12)
    posterior = bocpd_student_t(
        standardized,
        hazard_lambda=hazard_lambda,
        max_run_length=min(1000, hazard_lambda * 4),
    )
    primary = posterior["posterior_mean"][:, 0]
    if secondary_index is None:
        secondary = posterior["surprise"]
    else:
        secondary = posterior["posterior_mean"][:, secondary_index]
    frame = pd.DataFrame(
        {
            "date": feature_frame.index[good].to_numpy(),
            "primary": primary,
            "short_mass": posterior["short_mass"],
            "run_drop": posterior["run_drop"],
            "secondary": secondary,
            "surprise": posterior["surprise"],
        }
    )
    metadata = {
        "columns": list(columns),
        "train_standardization_mean": mean.tolist(),
        "train_standardization_std": std.tolist(),
        "hazard_lambda_hours": hazard_lambda,
        "max_run_length": min(1000, hazard_lambda * 4),
        "prior_kappa": 0.1,
        "prior_alpha": 2.0,
        "prior_beta": 1.0,
        "short_run_horizon_hours": 6,
    }
    return frame, metadata


def _map_output(dates: pd.Series, output: pd.DataFrame) -> pd.DataFrame:
    return pd.merge_asof(
        pd.DataFrame({"date": pd.to_datetime(dates), "pos": np.arange(len(dates))}),
        output.sort_values("date"),
        on="date",
        direction="backward",
        tolerance=pd.Timedelta("2h"),
    ).sort_values("pos")


def _state_from_mapped(mapped: pd.DataFrame, thresholds: dict[str, float]) -> np.ndarray:
    primary = mapped["primary"].to_numpy(float)
    short_mass = mapped["short_mass"].to_numpy(float)
    secondary = mapped["secondary"].to_numpy(float)
    state = (
        _bucket3(primary, thresholds["primary_low"], thresholds["primary_high"]) * 4
        + (short_mass >= thresholds["short_mass_high"]).astype(int) * 2
        + (secondary >= thresholds["secondary_high"]).astype(int)
    )
    finite = np.isfinite(primary) & np.isfinite(short_mass) & np.isfinite(secondary)
    return np.where(finite, state, -1)


def run(cfg: Config) -> dict:
    market = _load_market(cfg)
    dates = pd.to_datetime(market["date"])
    base = build_market_feature_frame(market, window_size=144)
    features = pd.concat([base, build_interest_features(market, base)], axis=1)
    features = features.loc[:, ~features.columns.duplicated(keep="last")]
    setup = _alpha_active(features, "long_minimal_funding_premium")
    _, hourly_feature = hourly_features(market)
    train_hour = np.asarray(
        (hourly_feature.index >= SPLITS["train"][0])
        & (hourly_feature.index < SPLITS["train"][1]),
        dtype=bool,
    )
    train_market = _split_mask(dates, *SPLITS["train"])
    positions = np.arange(143, len(market) - 578, 12)
    specifications = {
        "return": {
            "columns": ("ret1",),
            "secondary_index": None,
            "primary_name": "segment_return_mean",
            "secondary_name": "predictive_surprise",
        },
        "return_flow": {
            "columns": ("ret1", "flow24"),
            "secondary_index": 1,
            "primary_name": "segment_return_mean",
            "secondary_name": "segment_flow_mean",
        },
        "trend_volterm": {
            "columns": ("trend24", "volterm"),
            "secondary_index": 1,
            "primary_name": "segment_trend_mean",
            "secondary_name": "segment_volterm_mean",
        },
    }
    raw_rows: list[dict] = []
    model_cache: dict[tuple[str, int], tuple[pd.DataFrame, pd.DataFrame, dict]] = {}

    for model_name, spec in specifications.items():
        for hazard_lambda in (72, 168, 336):
            hourly_output, model_metadata = _model_output(
                hourly_feature,
                train_hour,
                columns=spec["columns"],
                secondary_index=spec["secondary_index"],
                hazard_lambda=hazard_lambda,
            )
            mapped = _map_output(dates, hourly_output)
            model_cache[(model_name, hazard_lambda)] = (hourly_output, mapped, model_metadata)
            fit = hourly_output[
                (hourly_output["date"] >= SPLITS["train"][0])
                & (hourly_output["date"] < SPLITS["train"][1])
            ]

            for (low_q, high_q), reset_q, secondary_q in itertools.product(
                ((0.2, 0.8), (0.25, 0.75), (0.33, 0.67)),
                (0.5, 0.75, 0.9),
                (0.5, 0.67),
            ):
                thresholds = {
                    "primary_low": float(fit["primary"].quantile(low_q)),
                    "primary_high": float(fit["primary"].quantile(high_q)),
                    "short_mass_high": float(fit["short_mass"].quantile(reset_q)),
                    "secondary_high": float(fit["secondary"].quantile(secondary_q)),
                }
                state = _state_from_mapped(mapped, thresholds)
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

                for min_count, min_edge in itertools.product((5, 8, 12), (0.0, 0.002, 0.005, 0.01)):
                    allowed = sorted(
                        state_id
                        for state_id, returns in quality.items()
                        if len(returns) >= min_count and float(np.mean(returns)) >= min_edge
                    )
                    if not allowed:
                        continue
                    long_active = setup & np.isin(state, allowed)
                    short_active = np.zeros(len(market), dtype=bool)
                    train_stats = sim(
                        market, dates, long_active, short_active, cfg, 576, 12, 10.0, 10.0, "train"
                    )
                    test_stats = sim(
                        market,
                        dates,
                        long_active,
                        short_active,
                        cfg,
                        576,
                        12,
                        10.0,
                        10.0,
                        "test2024",
                    )
                    if train_stats["trades"] < 80 or test_stats["trades"] < 10:
                        continue
                    raw_rows.append(
                        {
                            "model_name": model_name,
                            "model": model_metadata,
                            "primary_name": spec["primary_name"],
                            "secondary_name": spec["secondary_name"],
                            "state_quantiles": {
                                "primary": [low_q, high_q],
                                "short_mass": reset_q,
                                "secondary": secondary_q,
                            },
                            "state_thresholds": thresholds,
                            "min_train_state_trades": min_count,
                            "min_train_trade_edge": min_edge,
                            "allowed_states": allowed,
                            "allowed_state_descriptions": {
                                str(state_id): _describe_state(
                                    state_id, spec["primary_name"], spec["secondary_name"]
                                )
                                for state_id in allowed
                            },
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
    selected: list[dict] = []
    seen_signals: set[str] = set()
    for row in raw_rows:
        if row["signal_hash"] in seen_signals:
            continue
        seen_signals.add(row["signal_hash"])
        if len(selected) < 100:
            selected.append(row)

    selected_signals: dict[str, np.ndarray] = {}
    for row in selected:
        selected_signals[row["signal_hash"]] = np.unpackbits(
            row.pop("_packed_signal"), count=len(market)
        ).astype(bool)
        short_active = np.zeros(len(market), dtype=bool)
        for split in ("eval2025", "ytd2026"):
            row[split] = sim(
                market,
                dates,
                selected_signals[row["signal_hash"]],
                short_active,
                cfg,
                576,
                12,
                10.0,
                10.0,
                split,
            )
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
        top_active = selected_signals[top["signal_hash"]]
        _, top_mapped, _ = model_cache[(top["model_name"], top["model"]["hazard_lambda_hours"])]
        top_state = _state_from_mapped(top_mapped, top["state_thresholds"])
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
            active = setup & np.isin(
                top_state, [state_id for state_id in top["allowed_states"] if state_id != dropped]
            )
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
            "Adams-MacKay causal BOCPD on completed hourly observations; train-frozen standardization, "
            "state bins and state trade quality; variants ranked by min(train ratio, test2024 ratio); "
            "eval2025/2026 report-only; hold576; 6bp/side; strict intrabar MDD"
        ),
        "source": "https://arxiv.org/abs/0710.3742",
        "selection_caveat": (
            "The BOCPD overlay was frozen before its Eval-2025/2026 diagnostics, but the underlying base "
            "setup has prior research-history exposure. Treat the composite as research-forward shadow."
        ),
        "tested_variants": len(raw_rows),
        "distinct_signal_variants": len(seen_signals),
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
