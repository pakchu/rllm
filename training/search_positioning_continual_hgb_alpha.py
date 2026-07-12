"""Monthly prequential HGB path-utility alpha with delayed positioning data.

At each month boundary the critic is refitted using only trades whose full
execution path has already ended.  Model/update/calibration policies are chosen
on 2023 and physically frozen before 2024+ data is loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from training.search_deribit_dvol_alpha import attach_dvol, build_dvol_features
from training.search_positioning_disagreement_alpha import WINDOWS, _future_extreme, _simulate_no_stop
from training.search_positioning_hgb_path_alpha import (
    FIT_START,
    PositioningHgbConfig,
    _feature_hash,
    _load_attached,
    _read_before,
    build_model_features,
    executable_path_utilities,
)


SELECTION_END = "2024-01-01"
TRAIN_ANCHOR_STRIDE = 12


@dataclass(frozen=True)
class ContinualModelSpec:
    name: str
    hold_bars: int
    risk_lambda: float = 0.5


MODEL_SPECS = (
    ContinualModelSpec("continual_h288_mean_lam05", 288),
    ContinualModelSpec("continual_h576_mean_lam05", 576),
)


@dataclass(frozen=True)
class ContinualConfig:
    input_csv: str
    metrics_csv: str
    output: str
    manifest_output: str
    dvol_csv: str = ""
    top_n: int = 10
    leverage: float = 0.5
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0001
    metrics_tolerance: str = "10min"
    source_delay_bars: int = 1
    min_select_trades: int = 24
    min_half_trades: int = 8
    random_state: int = 42


def _model_by_name(name: str) -> ContinualModelSpec:
    for spec in MODEL_SPECS:
        if spec.name == name:
            return spec
    raise KeyError(name)


def monthly_ranges(start: str, end: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    starts = list(pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="MS"))
    return [(month, min(month + pd.offsets.MonthBegin(1), pd.Timestamp(end))) for month in starts if month < pd.Timestamp(end)]


def eligible_fit_mask(
    signal_dates: np.ndarray,
    exit_positions: np.ndarray,
    *,
    cutoff_position: int,
    cutoff_date: pd.Timestamp,
    train_days: int,
) -> np.ndarray:
    dates = np.asarray(signal_dates, dtype="datetime64[ns]")
    mask = (
        (dates >= np.datetime64(pd.Timestamp(FIT_START)))
        & (dates < np.datetime64(cutoff_date))
        & (np.asarray(exit_positions, dtype=np.int64) < int(cutoff_position))
    )
    if train_days > 0:
        mask &= dates >= np.datetime64(cutoff_date - pd.Timedelta(days=int(train_days)))
    return mask


def _fit_one(x: np.ndarray, y: np.ndarray, *, random_state: int) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=60,
        max_leaf_nodes=15,
        min_samples_leaf=160,
        l2_regularization=5.0,
        max_bins=127,
        early_stopping=False,
        random_state=random_state,
    )
    model.fit(x, y)
    return model


def _fit_month_models(
    market: pd.DataFrame,
    features: pd.DataFrame,
    positions: np.ndarray,
    utilities: np.ndarray,
    spec: ContinualModelSpec,
    *,
    cutoff: pd.Timestamp,
    train_days: int,
    cfg: ContinualConfig,
) -> tuple[HistGradientBoostingRegressor, HistGradientBoostingRegressor, int]:
    dates = pd.to_datetime(market["date"])
    signal_dates = dates.iloc[positions].to_numpy(dtype="datetime64[ns]")
    exit_positions = positions + 1 + spec.hold_bars
    cutoff_position = int(np.searchsorted(dates.to_numpy(dtype="datetime64[ns]"), np.datetime64(cutoff), side="left"))
    mask = eligible_fit_mask(
        signal_dates,
        exit_positions,
        cutoff_position=cutoff_position,
        cutoff_date=cutoff,
        train_days=train_days,
    ) & np.isfinite(utilities).all(axis=1)
    fit_positions = positions[mask]
    if len(fit_positions) < 5_000:
        raise ValueError(f"insufficient continual fit rows at {cutoff}: {len(fit_positions)}")
    x = features.iloc[fit_positions].to_numpy(np.float32)
    y = utilities[mask].copy()
    lower = np.nanquantile(y, 0.005, axis=0)
    upper = np.nanquantile(y, 0.995, axis=0)
    y = np.clip(y, lower, upper)
    return (
        _fit_one(x, y[:, 0], random_state=cfg.random_state),
        _fit_one(x, y[:, 1], random_state=cfg.random_state + 1),
        int(len(x)),
    )


def _path_utilities_for_spec(
    market: pd.DataFrame,
    positions: np.ndarray,
    spec: ContinualModelSpec,
    cfg: ContinualConfig,
) -> np.ndarray:
    return executable_path_utilities(
        market,
        positions,
        hold_bars=spec.hold_bars,
        risk_lambda=spec.risk_lambda,
        leverage=cfg.leverage,
        side_cost=(cfg.fee_rate + cfg.slippage_rate) * cfg.leverage,
        available_before_position=len(market),
    )


def _predict_pair(
    models: tuple[HistGradientBoostingRegressor, HistGradientBoostingRegressor],
    features: pd.DataFrame,
    indices: np.ndarray,
) -> np.ndarray:
    if len(indices) == 0:
        return np.empty((0, 2), dtype=np.float32)
    x = features.iloc[indices].to_numpy(np.float32)
    return np.column_stack((models[0].predict(x), models[1].predict(x))).astype(np.float32)


def _policy_key(model: str, train_days: int, calibration_days: int, quantile: float, side: str) -> str:
    return f"{model}|train{train_days}|cal{calibration_days}|q{quantile:.2f}|{side}"


def _policy_grid() -> list[dict[str, Any]]:
    return [
        {
            "model": model.name,
            "train_days": train_days,
            "calibration_days": calibration_days,
            "score_quantile": quantile,
            "side": side,
        }
        for model in MODEL_SPECS
        for train_days in (0, 730)
        for calibration_days in (90, 180, 365)
        for quantile in (0.80, 0.90, 0.95)
        for side in ("both", "long", "short")
    ]


def generate_prequential_signals(
    market: pd.DataFrame,
    features: pd.DataFrame,
    *,
    start: str,
    end: str,
    policies: Iterable[dict[str, Any]],
    cfg: ContinualConfig,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[dict[str, Any]]]:
    dates = pd.to_datetime(market["date"])
    policies = list(policies)
    signals = {
        _policy_key(row["model"], row["train_days"], row["calibration_days"], row["score_quantile"], row["side"]):
        (np.zeros(len(market), dtype=bool), np.zeros(len(market), dtype=bool))
        for row in policies
    }
    positions = np.arange(0, len(market), TRAIN_ANCHOR_STRIDE, dtype=np.int64)
    specs = {name: _model_by_name(name) for name in {row["model"] for row in policies}}
    utilities = {name: _path_utilities_for_spec(market, positions, spec, cfg) for name, spec in specs.items()}
    fit_log: list[dict[str, Any]] = []
    for model_name, train_days in sorted({(row["model"], row["train_days"]) for row in policies}):
        spec = specs[model_name]
        relevant = [row for row in policies if row["model"] == model_name and row["train_days"] == train_days]
        for month_start, month_end in monthly_ranges(start, end):
            long_model, short_model, fit_rows = _fit_month_models(
                market,
                features,
                positions,
                utilities[model_name],
                spec,
                cutoff=month_start,
                train_days=train_days,
                cfg=cfg,
            )
            fit_log.append({"model": model_name, "train_days": train_days, "cutoff": str(month_start), "fit_rows": fit_rows})
            max_calibration = max(row["calibration_days"] for row in relevant)
            calibration_mask = (dates >= month_start - pd.Timedelta(days=max_calibration)) & (dates < month_start)
            month_mask = (dates >= month_start) & (dates < month_end)
            calibration_indices = np.flatnonzero(calibration_mask.to_numpy(bool))
            month_indices = np.flatnonzero(month_mask.to_numpy(bool))
            calibration_predictions = _predict_pair((long_model, short_model), features, calibration_indices)
            month_predictions = _predict_pair((long_model, short_model), features, month_indices)
            month_long = month_predictions[:, 0]
            month_short = month_predictions[:, 1]
            month_confidence = np.maximum(month_long, month_short)
            for row in relevant:
                keep = dates.iloc[calibration_indices] >= month_start - pd.Timedelta(days=row["calibration_days"])
                history = np.maximum(calibration_predictions[keep.to_numpy(bool), 0], calibration_predictions[keep.to_numpy(bool), 1])
                history = history[np.isfinite(history)]
                if len(history) < 2_000:
                    continue
                threshold = float(np.quantile(history, row["score_quantile"]))
                eligible = np.isfinite(month_confidence) & (month_confidence >= threshold) & (month_confidence > 0.0)
                key = _policy_key(row["model"], row["train_days"], row["calibration_days"], row["score_quantile"], row["side"])
                long_signal, short_signal = signals[key]
                if row["side"] in {"both", "long"}:
                    long_signal[month_indices] = eligible & (month_long > month_short)
                if row["side"] in {"both", "short"}:
                    short_signal[month_indices] = eligible & (month_short > month_long)
    return signals, fit_log


def _signal_hash(long_signal: np.ndarray, short_signal: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(long_signal, dtype=np.uint8).tobytes())
    digest.update(np.asarray(short_signal, dtype=np.uint8).tobytes())
    return digest.hexdigest()


def _load_market(cfg: ContinualConfig, hgb_cfg: PositioningHgbConfig, *, cutoff: str | None = None) -> pd.DataFrame:
    market = _load_attached(hgb_cfg, cutoff=cutoff)
    if not cfg.dvol_csv:
        return market
    dvol = pd.read_csv(cfg.dvol_csv, compression="infer") if cutoff is None else _read_before(cfg.dvol_csv, "close_time", cutoff)
    return attach_dvol(market, dvol, tolerance="65min")


def build_continual_features(market: pd.DataFrame, *, include_dvol: bool) -> pd.DataFrame:
    base = build_model_features(market)
    if not include_dvol:
        return base
    dvol = build_dvol_features(market)
    interactions = pd.DataFrame(index=market.index)
    interactions["dvol_stress_x_trend"] = dvol["dvol_z25920"] * base["price_return_288"]
    interactions["dvol_shock_x_oi_divergence"] = dvol["dvol_logchg2016"] * base["oi_price_divergence_288"]
    interactions["dvol_stress_x_smart_retail"] = dvol["dvol_z25920"] * base["smart_retail_z2016"]
    return pd.concat([base, dvol.add_prefix("option_"), interactions], axis=1).replace([np.inf, -np.inf], np.nan).astype(np.float32)


def run(cfg: ContinualConfig) -> dict[str, Any]:
    hgb_cfg = PositioningHgbConfig(
        input_csv=cfg.input_csv,
        metrics_csv=cfg.metrics_csv,
        output=cfg.output,
        manifest_output=cfg.manifest_output,
        leverage=cfg.leverage,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
        metrics_tolerance=cfg.metrics_tolerance,
        source_delay_bars=cfg.source_delay_bars,
        random_state=cfg.random_state,
    )
    phase1_market = _load_market(cfg, hgb_cfg, cutoff=SELECTION_END)
    phase1_dates = pd.to_datetime(phase1_market["date"])
    phase1_features = build_continual_features(phase1_market, include_dvol=bool(cfg.dvol_csv))
    phase1_hash = _feature_hash(phase1_features, phase1_dates)
    policy_grid = _policy_grid()
    signal_bank, phase1_fit_log = generate_prequential_signals(
        phase1_market,
        phase1_features,
        start="2023-01-01",
        end=SELECTION_END,
        policies=policy_grid,
        cfg=cfg,
    )
    extremes = {
        spec.hold_bars: (
            _future_extreme(phase1_market["low"].to_numpy(float), spec.hold_bars, "min"),
            _future_extreme(phase1_market["high"].to_numpy(float), spec.hold_bars, "max"),
        )
        for spec in MODEL_SPECS
    }
    candidates: list[dict[str, Any]] = []
    for policy in policy_grid:
        key = _policy_key(policy["model"], policy["train_days"], policy["calibration_days"], policy["score_quantile"], policy["side"])
        long_signal, short_signal = signal_bank[key]
        hold = _model_by_name(policy["model"]).hold_bars
        for stride in (6, 12, 24):
            kwargs = {
                "hold_bars": hold,
                "stride_bars": stride,
                "leverage": cfg.leverage,
                "fee_rate": cfg.fee_rate,
                "slippage_rate": cfg.slippage_rate,
                "extremes": extremes[hold],
            }
            selection = _simulate_no_stop(phase1_market, phase1_dates, long_signal, short_signal, window="select2023", **kwargs)
            halves = [
                _simulate_no_stop(phase1_market, phase1_dates, long_signal, short_signal, window=name, **kwargs)
                for name in ("select2023_h1", "select2023_h2")
            ]
            if selection["trades"] < cfg.min_select_trades or min(half["trades"] for half in halves) < cfg.min_half_trades:
                continue
            candidates.append(
                {
                    **policy,
                    "hold_bars": hold,
                    "stride_bars": stride,
                    "select2023": selection,
                    "select2023_halves": halves,
                    "selection_score": {
                        "positive_halves": sum(half["return_pct"] > 0.0 for half in halves),
                        "min_half_ratio": min(half["ratio"] for half in halves),
                        "full_ratio": selection["ratio"],
                    },
                }
            )
    candidates.sort(
        key=lambda row: (
            row["selection_score"]["positive_halves"],
            row["selection_score"]["min_half_ratio"],
            row["selection_score"]["full_ratio"],
            row["select2023"]["return_pct"],
        ),
        reverse=True,
    )
    selected = candidates[: cfg.top_n]
    manifest_policies = [
        {key: row[key] for key in ("model", "train_days", "calibration_days", "score_quantile", "side", "hold_bars", "stride_bars")}
        for row in selected
    ]
    manifest_json = json.dumps(manifest_policies, sort_keys=True, separators=(",", ":"))
    manifest_hash = hashlib.sha256(manifest_json.encode()).hexdigest()
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": "pre_future_manifest",
        "feature_hash": phase1_hash,
        "manifest_hash": manifest_hash,
        "policies": manifest_policies,
        "phase1_signal_hashes": {
            _policy_key(row["model"], row["train_days"], row["calibration_days"], row["score_quantile"], row["side"]):
            _signal_hash(*signal_bank[_policy_key(row["model"], row["train_days"], row["calibration_days"], row["score_quantile"], row["side"])])
            for row in selected
        },
        "phase1_fit_log": phase1_fit_log,
    }
    manifest_path = Path(cfg.manifest_output)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    full_market = _load_market(cfg, hgb_cfg)
    full_dates = pd.to_datetime(full_market["date"])
    full_features = build_continual_features(full_market, include_dvol=bool(cfg.dvol_csv))
    prefix_dates = full_dates.iloc[: len(phase1_dates)].reset_index(drop=True)
    if not prefix_dates.equals(phase1_dates.reset_index(drop=True)):
        raise RuntimeError("continual pre-2024 timestamp prefix changed")
    if _feature_hash(full_features.iloc[: len(phase1_features)], prefix_dates) != phase1_hash:
        raise RuntimeError("continual pre-2024 feature prefix changed")

    unique_policies = []
    seen = set()
    for row in selected:
        compact = {key: row[key] for key in ("model", "train_days", "calibration_days", "score_quantile", "side")}
        key = tuple(compact.values())
        if key not in seen:
            seen.add(key)
            unique_policies.append(compact)
    future_bank, future_fit_log = generate_prequential_signals(
        full_market,
        full_features,
        start="2024-01-01",
        end="2026-06-02",
        policies=unique_policies,
        cfg=cfg,
    )
    extremes_full = {
        spec.hold_bars: (
            _future_extreme(full_market["low"].to_numpy(float), spec.hold_bars, "min"),
            _future_extreme(full_market["high"].to_numpy(float), spec.hold_bars, "max"),
        )
        for spec in MODEL_SPECS
    }
    for row in selected:
        key = _policy_key(row["model"], row["train_days"], row["calibration_days"], row["score_quantile"], row["side"])
        long_signal = np.zeros(len(full_market), dtype=bool)
        short_signal = np.zeros(len(full_market), dtype=bool)
        long_signal[: len(phase1_market)] = signal_bank[key][0]
        short_signal[: len(phase1_market)] = signal_bank[key][1]
        long_signal |= future_bank[key][0]
        short_signal |= future_bank[key][1]
        if _signal_hash(long_signal[: len(phase1_market)], short_signal[: len(phase1_market)]) != manifest["phase1_signal_hashes"][key]:
            raise RuntimeError(f"continual phase1 signal prefix changed for {key}")
        for window in ("test2024", "eval2025", "ytd2026"):
            row[window] = _simulate_no_stop(
                full_market,
                full_dates,
                long_signal,
                short_signal,
                window=window,
                hold_bars=row["hold_bars"],
                stride_bars=row["stride_bars"],
                leverage=cfg.leverage,
                fee_rate=cfg.fee_rate,
                slippage_rate=cfg.slippage_rate,
                extremes=extremes_full[row["hold_bars"]],
            )
        row["passes_alpha_target"] = (
            row["test2024"]["ratio"] >= 3.0
            and row["eval2025"]["ratio"] >= 3.0
            and row["test2024"]["trades"] >= 16
            and row["eval2025"]["trades"] >= 16
        )
        row["passes_live_target"] = (
            row["passes_alpha_target"]
            and row["ytd2026"]["ratio"] >= 5.0
            and row["ytd2026"]["trades"] >= 8
            and row["ytd2026"]["return_pct"] > 0.0
        )
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "protocol": {
            "initial_fit_start": FIT_START,
            "monthly_refit": True,
            "label_admission": "entry+hold exit position strictly before monthly cutoff",
            "selection": WINDOWS["select2023"],
            "manifest_written_before_future": True,
            "future_windows": {name: WINDOWS[name] for name in ("test2024", "eval2025", "ytd2026")},
            "strict_mdd": "worst-order favorable-to-adverse OHLC high-water path drawdown",
            "dvol_input": bool(cfg.dvol_csv),
        },
        "input": {"rows": int(len(full_market)), "features": int(full_features.shape[1])},
        "tested": int(len(candidates)),
        "manifest_hash": manifest_hash,
        "selected": selected,
        "future_fit_log": future_fit_log,
        "alpha_qualifiers": [row for row in selected if row["passes_alpha_target"]],
        "live_qualifiers": [row for row in selected if row["passes_live_target"]],
    }
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> ContinualConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--metrics-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--dvol-csv", default=ContinualConfig.dvol_csv)
    parser.add_argument("--top-n", type=int, default=ContinualConfig.top_n)
    parser.add_argument("--leverage", type=float, default=ContinualConfig.leverage)
    parser.add_argument("--fee-rate", type=float, default=ContinualConfig.fee_rate)
    parser.add_argument("--slippage-rate", type=float, default=ContinualConfig.slippage_rate)
    parser.add_argument("--metrics-tolerance", default=ContinualConfig.metrics_tolerance)
    parser.add_argument("--source-delay-bars", type=int, default=ContinualConfig.source_delay_bars)
    parser.add_argument("--min-select-trades", type=int, default=ContinualConfig.min_select_trades)
    parser.add_argument("--min-half-trades", type=int, default=ContinualConfig.min_half_trades)
    parser.add_argument("--random-state", type=int, default=ContinualConfig.random_state)
    return ContinualConfig(**vars(parser.parse_args()))


def main() -> None:
    report = run(parse_args())
    print(json.dumps({"tested": report["tested"], "manifest_hash": report["manifest_hash"], "alpha": len(report["alpha_qualifiers"]), "live": len(report["live_qualifiers"]), "selected": report["selected"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
