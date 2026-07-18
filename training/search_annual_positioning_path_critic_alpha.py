"""Search and freeze an annual prequential BTC positioning path critic.

The model is refit only at UTC year boundaries.  Every admitted target has
finished before the refit cutoff, score calibration uses feature scores from
the completed prior calendar year, and fills occur at the next five-minute
open.  Candidate selection is physically truncated before 2024.  ``--open-oos``
replays only the frozen selected policy on 2024 onward.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from training.search_inventory_purge_reclaim_alpha import (
    Config as ExecutionConfig,
    ExecutionEngine,
    Trade,
    _load_sources,
    _schedule_hash,
    equity_stats,
)
from training.search_positioning_hgb_path_alpha import (
    build_model_features,
    executable_path_utilities,
)


SELECTION_END = "2024-01-01"
FULL_END = "2026-05-31 15:05:00"
FIT_START = "2020-10-15"
MODEL_ANCHOR_STRIDE = 12
MODEL_RANDOM_SEED = 42
SIGNFLIP_SEED = 20260719
SIGNFLIP_DRAWS = 20_000

WINDOWS: dict[str, tuple[str, str]] = {
    "development_2022": ("2022-01-01", "2023-01-01"),
    "selection_2023": ("2023-01-01", "2024-01-01"),
    "selection_2023_h1": ("2023-01-01", "2023-07-01"),
    "selection_2023_h2": ("2023-07-01", "2024-01-01"),
    "test_2024": ("2024-01-01", "2025-01-01"),
    "eval_2025": ("2025-01-01", "2026-01-01"),
    "holdout_2026": ("2026-01-01", FULL_END),
    "oos_2024_2026": ("2024-01-01", FULL_END),
}


@dataclass(frozen=True)
class ModelSpec:
    hold_bars: int
    loss: str
    quantile: float | None

    @property
    def name(self) -> str:
        suffix = "mean" if self.loss == "squared_error" else f"q{self.quantile:.2f}"
        return f"annual_h{self.hold_bars}_{suffix}"


MODEL_SPECS = tuple(
    ModelSpec(hold, loss, quantile)
    for hold in (144, 288, 576)
    for loss, quantile in (
        ("squared_error", None),
        ("quantile", 0.20),
        ("quantile", 0.35),
    )
)

EXPECTED_POLICY = {
    "model": "annual_h288_mean",
    "hold_bars": 288,
    "score_quantile": 0.80,
    "side": "both",
    "execution_stride_bars": 12,
}


@dataclass(frozen=True)
class Config:
    input_csv: str
    metrics_csv: str
    funding_csv: str
    output: str
    manifest_output: str
    docs_output: str = ""
    leverage: float = 0.5
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0001
    stress_cost_rate: float = 0.0010
    metrics_tolerance: str = "10min"
    source_delay_bars: int = 1
    open_oos: bool = False


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def build_causal_features(market: pd.DataFrame) -> pd.DataFrame:
    """Build model inputs while failing closed on unavailable external states."""
    features = build_model_features(market)
    availability_contract = {
        "dxy_available": ("dxy_zscore", "dxy_momentum"),
        "kimchi_available": ("kimchi_premium_zscore", "kimchi_premium_change"),
        "usdkrw_available": ("usdkrw_zscore", "usdkrw_momentum"),
    }
    for availability, columns in availability_contract.items():
        if availability not in market.columns:
            raise RuntimeError(f"missing external availability flag: {availability}")
        valid = pd.to_numeric(market[availability], errors="coerce").fillna(0.0).to_numpy(float) > 0.5
        features[availability] = valid.astype(np.float32)
        for column in columns:
            if column not in features.columns:
                raise RuntimeError(f"missing external model feature: {column}")
            features[column] = features[column].where(valid, np.nan)
    return features.astype(np.float32)


def execution_config(cfg: Config) -> ExecutionConfig:
    return ExecutionConfig(
        input_csv=cfg.input_csv,
        metrics_csv=cfg.metrics_csv,
        funding_csv=cfg.funding_csv,
        output=cfg.output,
        manifest_output=cfg.manifest_output,
        metrics_tolerance=cfg.metrics_tolerance,
        source_delay_bars=cfg.source_delay_bars,
        leverage=cfg.leverage,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
        stress_cost_rate=cfg.stress_cost_rate,
    )


def eligible_training_mask(
    signal_dates: np.ndarray,
    exit_positions: np.ndarray,
    *,
    cutoff_date: str | pd.Timestamp,
    cutoff_position: int,
) -> np.ndarray:
    dates = np.asarray(signal_dates, dtype="datetime64[ns]")
    return (
        (dates >= np.datetime64(pd.Timestamp(FIT_START)))
        & (dates < np.datetime64(pd.Timestamp(cutoff_date)))
        & (np.asarray(exit_positions, dtype=np.int64) < int(cutoff_position))
    )


def _fit_one(x: np.ndarray, y: np.ndarray, spec: ModelSpec, seed: int) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        loss=spec.loss,
        quantile=spec.quantile if spec.loss == "quantile" else None,
        learning_rate=0.05,
        max_iter=80,
        max_leaf_nodes=15,
        min_samples_leaf=180,
        l2_regularization=8.0,
        max_bins=127,
        early_stopping=False,
        random_state=seed,
    )
    model.fit(x, y)
    return model


def fit_models(
    market: pd.DataFrame,
    features: pd.DataFrame,
    spec: ModelSpec,
    *,
    cutoff: str,
    cfg: Config,
) -> tuple[tuple[HistGradientBoostingRegressor, HistGradientBoostingRegressor], dict[str, Any]]:
    dates = pd.to_datetime(market["date"])
    positions = np.arange(0, len(market), MODEL_ANCHOR_STRIDE, dtype=np.int64)
    cutoff_position = int(
        np.searchsorted(dates.to_numpy(dtype="datetime64[ns]"), np.datetime64(pd.Timestamp(cutoff)), side="left")
    )
    utilities = executable_path_utilities(
        market,
        positions,
        hold_bars=spec.hold_bars,
        risk_lambda=0.5,
        leverage=cfg.leverage,
        side_cost=(cfg.fee_rate + cfg.slippage_rate) * cfg.leverage,
        available_before_position=cutoff_position,
    )
    signal_dates = dates.iloc[positions].to_numpy(dtype="datetime64[ns]")
    mask = eligible_training_mask(
        signal_dates,
        positions + 1 + spec.hold_bars,
        cutoff_date=cutoff,
        cutoff_position=cutoff_position,
    ) & np.isfinite(utilities).all(axis=1)
    fit_positions = positions[mask]
    if len(fit_positions) < 5_000:
        raise ValueError(f"insufficient annual fit rows at {cutoff}: {len(fit_positions)}")
    x = features.iloc[fit_positions].to_numpy(np.float32)
    y = utilities[mask].copy()
    lower = np.quantile(y, 0.005, axis=0)
    upper = np.quantile(y, 0.995, axis=0)
    y = np.clip(y, lower, upper)
    models = (
        _fit_one(x, y[:, 0], spec, MODEL_RANDOM_SEED),
        _fit_one(x, y[:, 1], spec, MODEL_RANDOM_SEED + 1),
    )
    return models, {
        "cutoff": str(pd.Timestamp(cutoff)),
        "fit_rows": int(len(fit_positions)),
        "last_admitted_signal": str(pd.Timestamp(signal_dates[mask].max())),
        "last_admitted_exit_position": int((positions + 1 + spec.hold_bars)[mask].max()),
        "cutoff_position": cutoff_position,
        "target_clip": {"lower": lower.tolist(), "upper": upper.tolist()},
    }


def predict_pair(
    models: tuple[HistGradientBoostingRegressor, HistGradientBoostingRegressor],
    features: pd.DataFrame,
    indices: np.ndarray,
) -> np.ndarray:
    if len(indices) == 0:
        return np.empty((0, 2), dtype=np.float32)
    x = features.iloc[indices].to_numpy(np.float32)
    return np.column_stack((models[0].predict(x), models[1].predict(x))).astype(np.float32)


def annual_prediction_cache(
    market: pd.DataFrame,
    features: pd.DataFrame,
    spec: ModelSpec,
    *,
    years: Iterable[int],
    cfg: Config,
) -> dict[int, dict[str, Any]]:
    dates = pd.to_datetime(market["date"])
    cache: dict[int, dict[str, Any]] = {}
    for year in years:
        cutoff = f"{int(year)}-01-01"
        models, meta = fit_models(market, features, spec, cutoff=cutoff, cfg=cfg)
        prior = ((dates >= pd.Timestamp(f"{year - 1}-01-01")) & (dates < pd.Timestamp(cutoff))).to_numpy(bool)
        current = ((dates >= pd.Timestamp(cutoff)) & (dates < pd.Timestamp(f"{year + 1}-01-01"))).to_numpy(bool)
        prior_indices = np.flatnonzero(prior)
        current_indices = np.flatnonzero(current)
        prior_predictions = predict_pair(models, features, prior_indices)
        current_predictions = predict_pair(models, features, current_indices)
        prior_confidence = np.max(prior_predictions, axis=1)
        finite_prior = prior_confidence[np.isfinite(prior_confidence)]
        if len(finite_prior) < 50_000:
            raise ValueError(f"insufficient prior-year score calibration for {year}: {len(finite_prior)}")
        cache[int(year)] = {
            "current_indices": current_indices,
            "current_predictions": current_predictions,
            "prior_confidence": finite_prior,
            "meta": {
                **meta,
                "year": int(year),
                "prior_score_rows": int(len(finite_prior)),
                "current_score_rows": int(len(current_indices)),
            },
        }
    return cache


def signal_masks_from_prediction_cache(
    rows: int,
    cache: dict[int, dict[str, Any]],
    *,
    score_quantile: float,
    side: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    if side not in {"both", "long", "short"}:
        raise ValueError(f"invalid side mode: {side}")
    long_active = np.zeros(rows, dtype=bool)
    short_active = np.zeros(rows, dtype=bool)
    fit_log: list[dict[str, Any]] = []
    for year in sorted(cache):
        record = cache[year]
        current_indices = record["current_indices"]
        current_predictions = record["current_predictions"]
        threshold = float(np.quantile(record["prior_confidence"], score_quantile))
        confidence = np.max(current_predictions, axis=1)
        eligible = np.isfinite(confidence) & (confidence >= threshold)
        if side in {"both", "long"}:
            long_active[current_indices] = eligible & (current_predictions[:, 0] > current_predictions[:, 1])
        if side in {"both", "short"}:
            short_active[current_indices] = eligible & (current_predictions[:, 1] > current_predictions[:, 0])
        fit_log.append(
            {
                **record["meta"],
                "score_quantile": float(score_quantile),
                "score_threshold": threshold,
                "raw_long_signals": int(long_active[current_indices].sum()),
                "raw_short_signals": int(short_active[current_indices].sum()),
            }
        )
    return long_active, short_active, fit_log


def annual_signal_masks(
    market: pd.DataFrame,
    features: pd.DataFrame,
    spec: ModelSpec,
    *,
    years: Iterable[int],
    score_quantile: float,
    side: str,
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    cache = annual_prediction_cache(market, features, spec, years=years, cfg=cfg)
    return signal_masks_from_prediction_cache(
        len(market), cache, score_quantile=score_quantile, side=side
    )


def schedule_window(
    engine: ExecutionEngine,
    long_active: np.ndarray,
    short_active: np.ndarray,
    *,
    window: str,
    hold_bars: int,
    stride_bars: int,
    flip: bool = False,
) -> list[Trade]:
    start, end = WINDOWS[window]
    dates = engine.dates
    period = ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy(bool)
    candidates = np.arange(0, len(dates) - hold_bars - 2, stride_bars, dtype=np.int64)
    active = np.logical_xor(long_active[candidates], short_active[candidates])
    candidates = candidates[period[candidates] & active]
    trades: list[Trade] = []
    next_allowed = 0
    for signal in candidates:
        signal = int(signal)
        if signal < next_allowed:
            continue
        side = 1 if bool(long_active[signal]) else -1
        if flip:
            side *= -1
        trade = engine.trade_at(signal, side, hold_bars, 10_000, 10_000)
        if trade is None or not period[trade.exit_position]:
            continue
        trades.append(trade)
        next_allowed = trade.exit_position + 1
    return trades


def stats_for(trades: list[Trade], window: str, cfg: Config, *, cost_rate: float | None = None) -> dict[str, Any]:
    start, end = WINDOWS[window]
    return equity_stats(trades, start=start, end=end, cfg=execution_config(cfg), cost_rate=cost_rate)


def trade_net_returns(trades: list[Trade], cfg: Config, *, cost_rate: float | None = None) -> np.ndarray:
    rate = cfg.fee_rate + cfg.slippage_rate if cost_rate is None else float(cost_rate)
    cost_factor = 1.0 - cfg.leverage * rate
    return np.asarray(
        [cost_factor * trade.price_factor * trade.funding_factor * cost_factor - 1.0 for trade in trades],
        dtype=float,
    )


def weekly_cluster_signflip(trades: list[Trade], cfg: Config) -> dict[str, Any]:
    if not trades:
        return {"p_value_one_sided": 1.0, "clusters": 0, "draws": 0}
    returns = np.log1p(trade_net_returns(trades, cfg))
    dates = pd.to_datetime(pd.Series([trade.entry_date for trade in trades]))
    weeks = (dates - pd.to_timedelta(dates.dt.weekday, unit="D")).dt.floor("D")
    values = pd.DataFrame({"week": weeks, "return": returns}).groupby("week")["return"].sum().to_numpy(float)
    observed = float(values.sum())
    generator = np.random.default_rng(SIGNFLIP_SEED)
    exceed = 0
    remaining = SIGNFLIP_DRAWS
    while remaining:
        batch = min(2_000, remaining)
        signs = generator.integers(0, 2, size=(batch, len(values)), dtype=np.int8) * 2 - 1
        exceed += int(np.count_nonzero(signs @ values >= observed - 1e-15))
        remaining -= batch
    return {
        "p_value_one_sided": float((1 + exceed) / (SIGNFLIP_DRAWS + 1)),
        "clusters": int(len(values)),
        "draws": SIGNFLIP_DRAWS,
        "seed": SIGNFLIP_SEED,
        "observed_cluster_sum": observed,
    }


def _candidate_score(stats: dict[str, dict[str, Any]]) -> float:
    names = ("development_2022", "selection_2023", "selection_2023_h1", "selection_2023_h2")
    rows = [stats[name] for name in names]
    if min(row["trades"] for row in rows[:2]) < 20 or min(row["trades"] for row in rows[2:]) < 8:
        return -1e12
    if min(row["absolute_return_pct"] for row in rows) <= 0.0:
        return -1e12
    ratios = np.asarray([row["cagr_to_strict_mdd"] for row in rows], dtype=float)
    return float(ratios.min() + 0.25 * np.median(ratios))


def _policy(spec: ModelSpec, score_quantile: float, side: str, stride: int) -> dict[str, Any]:
    return {
        "model": spec.name,
        "hold_bars": spec.hold_bars,
        "loss": spec.loss,
        "target_quantile": spec.quantile,
        "score_quantile": float(score_quantile),
        "side": side,
        "execution_stride_bars": int(stride),
        "target": "net fixed-hold return minus 0.5 times same-path MAE at 0.5x",
    }


def _model_from_policy(policy: dict[str, Any]) -> ModelSpec:
    matches = [spec for spec in MODEL_SPECS if spec.name == policy["model"]]
    if len(matches) != 1:
        raise RuntimeError(f"unknown frozen model: {policy['model']}")
    return matches[0]


def _selection(cfg: Config) -> dict[str, Any]:
    market, funding, prefix_hashes = _load_sources(execution_config(cfg), cutoff=SELECTION_END)
    if pd.Timestamp(market["date"].max()) >= pd.Timestamp(SELECTION_END):
        raise RuntimeError("selection market was not physically truncated before 2024")
    features = build_causal_features(market)
    engine = ExecutionEngine(market, funding, execution_config(cfg))
    candidates: list[dict[str, Any]] = []
    fit_logs: dict[str, list[dict[str, Any]]] = {}
    for spec in MODEL_SPECS:
        prediction_cache = annual_prediction_cache(
            market, features, spec, years=(2022, 2023), cfg=cfg
        )
        for score_quantile in (0.80, 0.90, 0.95):
            for side in ("both", "long", "short"):
                signals = signal_masks_from_prediction_cache(
                    len(market),
                    prediction_cache,
                    score_quantile=score_quantile,
                    side=side,
                )
                fit_logs[f"{spec.name}|q{score_quantile:.2f}|{side}"] = signals[2]
                for stride in (12, 24):
                    schedules = {
                        name: schedule_window(
                            engine,
                            signals[0],
                            signals[1],
                            window=name,
                            hold_bars=spec.hold_bars,
                            stride_bars=stride,
                        )
                        for name in (
                            "development_2022",
                            "selection_2023",
                            "selection_2023_h1",
                            "selection_2023_h2",
                        )
                    }
                    stats = {name: stats_for(trades, name, cfg) for name, trades in schedules.items()}
                    candidates.append(
                        {
                            "policy": _policy(spec, score_quantile, side, stride),
                            "score": _candidate_score(stats),
                            "stats": stats,
                            "schedule_hashes": {name: _schedule_hash(trades) for name, trades in schedules.items()},
                        }
                    )
    candidates.sort(
        key=lambda row: (
            row["score"],
            row["stats"]["selection_2023"]["cagr_to_strict_mdd"],
            row["stats"]["selection_2023"]["absolute_return_pct"],
        ),
        reverse=True,
    )
    selected = candidates[0]
    expected = {key: selected["policy"][key] for key in EXPECTED_POLICY}
    if expected != EXPECTED_POLICY:
        raise RuntimeError(f"selection drifted from frozen expected policy: {expected}")
    selected_spec = _model_from_policy(selected["policy"])
    selected_signals = signal_masks_from_prediction_cache(
        len(market),
        annual_prediction_cache(market, features, selected_spec, years=(2022, 2023), cfg=cfg),
        score_quantile=float(selected["policy"]["score_quantile"]),
        side=str(selected["policy"]["side"]),
    )
    selected_schedules = {
        name: schedule_window(
            engine,
            selected_signals[0],
            selected_signals[1],
            window=name,
            hold_bars=selected_spec.hold_bars,
            stride_bars=int(selected["policy"]["execution_stride_bars"]),
        )
        for name in ("development_2022", "selection_2023")
    }
    selected["stress_stats"] = {
        name: stats_for(trades, name, cfg, cost_rate=cfg.stress_cost_rate)
        for name, trades in selected_schedules.items()
    }
    selected["weekly_cluster_signflip"] = {
        name: weekly_cluster_signflip(trades, cfg) for name, trades in selected_schedules.items()
    }
    selected["selection_gates"] = {
        "development_ratio_at_least_3": selected["stats"]["development_2022"]["cagr_to_strict_mdd"] >= 3.0,
        "selection_ratio_at_least_3": selected["stats"]["selection_2023"]["cagr_to_strict_mdd"] >= 3.0,
        "each_half_ratio_at_least_3": min(
            selected["stats"]["selection_2023_h1"]["cagr_to_strict_mdd"],
            selected["stats"]["selection_2023_h2"]["cagr_to_strict_mdd"],
        ) >= 3.0,
        "strict_mdd_at_most_15": max(
            selected["stats"][name]["strict_mdd_pct"]
            for name in ("development_2022", "selection_2023")
        ) <= 15.0,
        "stress_return_positive": min(
            row["absolute_return_pct"] for row in selected["stress_stats"].values()
        ) > 0.0,
        "weekly_signflip_p_at_most_0_10": max(
            row["p_value_one_sided"] for row in selected["weekly_cluster_signflip"].values()
        ) <= 0.10,
    }
    selected["selection_passed"] = all(selected["selection_gates"].values())
    manifest = {
        "protocol_version": "annual_positioning_path_critic_selection_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "search_source": str(Path(__file__).relative_to(Path.cwd())),
        "search_source_sha256": sha256_file(__file__),
        "outcomes_opened": ["development_2022", "selection_2023"],
        "sealed": ["test_2024", "eval_2025", "holdout_2026"],
        "tested_policy_count": len(candidates),
        "multiple_testing_warning": (
            "162 policy cells were searched on 2022/2023; OOS gates are mandatory and "
            "their p-values do not correct this development search."
        ),
        "sources": {
            "paths": {
                "market": cfg.input_csv,
                "metrics": cfg.metrics_csv,
                "funding": cfg.funding_csv,
            },
            "pre2024_frame_hashes": prefix_hashes,
            "feature_columns": list(features.columns),
            "feature_count": int(features.shape[1]),
        },
        "execution": {
            "entry": "next 5m open",
            "exit": "fixed hold at open",
            "leverage": cfg.leverage,
            "base_cost_notional_per_side": cfg.fee_rate + cfg.slippage_rate,
            "stress_cost_notional_per_side": cfg.stress_cost_rate,
            "realized_funding": True,
            "strict_mdd": "global HWM plus favorable-before-adverse held OHLC and funding debit",
            "full_calendar_cagr": True,
            "overlap": "one position; next signal after previous exit",
        },
        "selected": selected,
        "top10": candidates[:10],
        "selected_fit_log": selected_signals[2],
    }
    manifest["manifest_payload_sha256"] = canonical_hash(manifest)
    return manifest


def _verify_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("protocol_version") != "annual_positioning_path_critic_selection_v1":
        raise RuntimeError("unexpected annual path critic manifest version")
    if manifest.get("search_source_sha256") != sha256_file(__file__):
        raise RuntimeError("search/evaluator source changed after selection freeze")
    payload = dict(manifest)
    expected = payload.pop("manifest_payload_sha256")
    if canonical_hash(payload) != expected:
        raise RuntimeError("selection manifest payload hash mismatch")
    selected = manifest["selected"]
    if not selected.get("selection_passed"):
        raise RuntimeError("selection did not pass its frozen gates")
    actual = {key: selected["policy"][key] for key in EXPECTED_POLICY}
    if actual != EXPECTED_POLICY:
        raise RuntimeError(f"manifest policy differs from frozen singleton: {actual}")


def _open_oos(cfg: Config) -> dict[str, Any]:
    manifest = json.loads(Path(cfg.manifest_output).read_text())
    _verify_manifest(manifest)
    market, funding, full_hashes = _load_sources(execution_config(cfg), cutoff=FULL_END)
    features = build_causal_features(market)
    engine = ExecutionEngine(market, funding, execution_config(cfg))
    policy = manifest["selected"]["policy"]
    spec = _model_from_policy(policy)
    signals = annual_signal_masks(
        market,
        features,
        spec,
        years=(2022, 2023, 2024, 2025, 2026),
        score_quantile=float(policy["score_quantile"]),
        side=str(policy["side"]),
        cfg=cfg,
    )
    schedules = {
        name: schedule_window(
            engine,
            signals[0],
            signals[1],
            window=name,
            hold_bars=spec.hold_bars,
            stride_bars=int(policy["execution_stride_bars"]),
        )
        for name in ("test_2024", "eval_2025", "holdout_2026", "oos_2024_2026")
    }
    base = {name: stats_for(trades, name, cfg) for name, trades in schedules.items()}
    stress = {
        name: stats_for(trades, name, cfg, cost_rate=cfg.stress_cost_rate)
        for name, trades in schedules.items()
    }
    flip_schedules = {
        name: schedule_window(
            engine,
            signals[0],
            signals[1],
            window=name,
            hold_bars=spec.hold_bars,
            stride_bars=int(policy["execution_stride_bars"]),
            flip=True,
        )
        for name in ("test_2024", "eval_2025", "holdout_2026", "oos_2024_2026")
    }
    flip = {name: stats_for(trades, name, cfg) for name, trades in flip_schedules.items()}
    signflip = weekly_cluster_signflip(schedules["oos_2024_2026"], cfg)
    gates = {
        "each_oos_window_absolute_return_positive": min(row["absolute_return_pct"] for name, row in base.items() if name != "oos_2024_2026") > 0.0,
        "each_oos_window_ratio_at_least_3": min(row["cagr_to_strict_mdd"] for name, row in base.items() if name != "oos_2024_2026") >= 3.0,
        "each_oos_window_strict_mdd_at_most_15": max(row["strict_mdd_pct"] for name, row in base.items() if name != "oos_2024_2026") <= 15.0,
        "minimum_trade_support": base["test_2024"]["trades"] >= 30 and base["eval_2025"]["trades"] >= 30 and base["holdout_2026"]["trades"] >= 15,
        "stress_return_positive_each_oos_window": min(row["absolute_return_pct"] for name, row in stress.items() if name != "oos_2024_2026") > 0.0,
        "combined_weekly_signflip_p_at_most_0_10": signflip["p_value_one_sided"] <= 0.10,
        "combined_primary_beats_direction_flip": base["oos_2024_2026"]["cagr_to_strict_mdd"] > flip["oos_2024_2026"]["cagr_to_strict_mdd"],
    }
    return {
        "protocol_version": "annual_positioning_path_critic_oos_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_manifest": cfg.manifest_output,
        "selection_manifest_sha256": sha256_file(cfg.manifest_output),
        "selection_manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "policy": policy,
        "full_source_hashes": full_hashes,
        "fit_log": signals[2],
        "base": base,
        "stress_10bp_per_side": stress,
        "direction_flip_control": flip,
        "schedule_hashes": {name: _schedule_hash(trades) for name, trades in schedules.items()},
        "weekly_cluster_signflip": signflip,
        "gates": gates,
        "promote": all(gates.values()),
        "caveat": (
            "2024-2026 BTC history was globally seen by unrelated repository research; this is "
            "an exact-policy mechanically frozen replay, not a pristine market clean room."
        ),
    }


def _write_docs(payload: dict[str, Any], path: str) -> None:
    if not path:
        return
    if payload["protocol_version"].endswith("selection_v1"):
        selected = payload["selected"]
        rows = [
            "# Annual positioning path critic — pre-2024 selection",
            "",
            "## Decision",
            "",
            f"**{'FREEZE FOR OOS' if selected['selection_passed'] else 'REJECT'}**",
            "",
            "The family was searched only on 2022/2023 outcomes. 2024, 2025, and 2026 remained sealed.",
            "",
            "| Window | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades | L/S |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for name in ("development_2022", "selection_2023", "selection_2023_h1", "selection_2023_h2"):
            value = selected["stats"][name]
            rows.append(
                f"| {name} | {value['absolute_return_pct']:.2f}% | {value['cagr_pct']:.2f}% | "
                f"{value['strict_mdd_pct']:.2f}% | {value['cagr_to_strict_mdd']:.2f} | "
                f"{value['trades']} | {value['longs']}/{value['shorts']} |"
            )
        rows += [
            "",
            "## Frozen policy",
            "",
            "```json",
            json.dumps(selected["policy"], indent=2),
            "```",
            "",
            "## Integrity boundary",
            "",
            f"- Tested cells: `{payload['tested_policy_count']}`; this multiple search is explicitly disclosed.",
            "- Model refits annually using only paths that exited before each January 1 cutoff.",
            "- Score threshold uses the completed prior year's feature-score distribution only.",
            "- Exact realized funding, 6 bp/notional-side base cost, 10 bp stress, next-open fill, full-calendar CAGR, and strict held-OHLC MDD are used.",
            "- OOS may be opened only with the byte-identical source and manifest.",
        ]
    else:
        rows = [
            "# Annual positioning path critic — frozen OOS replay",
            "",
            "## Decision",
            "",
            f"**{'PROMOTE' if payload['promote'] else 'REJECT'}**",
            "",
            "| Window | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades | L/S |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for name in ("test_2024", "eval_2025", "holdout_2026", "oos_2024_2026"):
            value = payload["base"][name]
            rows.append(
                f"| {name} | {value['absolute_return_pct']:.2f}% | {value['cagr_pct']:.2f}% | "
                f"{value['strict_mdd_pct']:.2f}% | {value['cagr_to_strict_mdd']:.2f} | "
                f"{value['trades']} | {value['longs']}/{value['shorts']} |"
            )
        rows += ["", "## Gates", ""]
        rows += [f"- `{name}`: {'PASS' if value else 'FAIL'}" for name, value in payload["gates"].items()]
        rows += ["", "## Caveat", "", payload["caveat"]]
    Path(path).write_text("\n".join(rows) + "\n")


def run(cfg: Config) -> dict[str, Any]:
    payload = _open_oos(cfg) if cfg.open_oos else _selection(cfg)
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if not cfg.open_oos:
        Path(cfg.manifest_output).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.manifest_output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _write_docs(payload, cfg.docs_output)
    return payload


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--metrics-csv", required=True)
    parser.add_argument("--funding-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--docs-output", default="")
    parser.add_argument("--leverage", type=float, default=Config.leverage)
    parser.add_argument("--fee-rate", type=float, default=Config.fee_rate)
    parser.add_argument("--slippage-rate", type=float, default=Config.slippage_rate)
    parser.add_argument("--stress-cost-rate", type=float, default=Config.stress_cost_rate)
    parser.add_argument("--metrics-tolerance", default=Config.metrics_tolerance)
    parser.add_argument("--source-delay-bars", type=int, default=Config.source_delay_bars)
    parser.add_argument("--open-oos", action="store_true")
    return Config(**vars(parser.parse_args()))


def main() -> None:
    payload = run(parse_args())
    if payload["protocol_version"].endswith("selection_v1"):
        print(json.dumps(payload["selected"], indent=2))
    else:
        print(json.dumps({"promote": payload["promote"], "base": payload["base"], "gates": payload["gates"]}, indent=2))


if __name__ == "__main__":
    main()
