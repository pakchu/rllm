#!/usr/bin/env python3
"""Run the preregistered AFGI-12 six-alt flow-geometry Gross9 battery."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import pickle
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesRegressor

import training.audit_gross9_residual_recovery_extratrees_marginal as runtime
import training.portfolio_opt_added_alpha_update as portfolio
from training.audit_gross9_oi_pullback_marginal import _atomic_json, _sha256
from training.audit_rank7_fresh_kimchi_fixed_portfolio import (
    subaccount_bar_path,
)
from training.build_six_alt_price_free_flow_panel import (
    OUTPUT_COLUMNS as SOURCE_COLUMNS,
)
from training.build_six_alt_price_free_flow_panel import SYMBOLS
from training.search_inventory_purge_reclaim_alpha import (
    Config as ExecutionConfig,
)
from training.search_inventory_purge_reclaim_alpha import ExecutionEngine, Trade


PREREGISTRATION = Path(
    "results/"
    "alt_flow_geometry_inference_marginal_preregistration_2026-07-28.json"
)
OUTPUT = Path(
    "results/"
    "gross9_alt_flow_geometry_inference_marginal_pre2025_2026-07-28.json"
)
DOCS = Path(
    "docs/"
    "gross9-alt-flow-geometry-inference-marginal-pre2025-2026-07-28.md"
)
EXPECTED_PREREGISTRATION_SHA256 = (
    "a27e2ed6afe9dcbf158b9b6fd7091c2d2beb617034e825f7c7444c1fa54c7fa4"
)
EXPECTED_SKLEARN_VERSION = "1.7.2"
SELECTION_CUTOFF = "2025-01-01"
HOLD_BARS = 144
NORMAL_COST_RATE = 0.0006
STRESS_COST_RATE = 0.001
UNIT_LEVERAGE = 0.5
ANCHOR_STRIDE_BARS = 12
WEIGHTS = (0.25, 0.5, 0.75, 1.0)
SEEDS = (7, 71, 715)
COORDINATION_MODES = runtime.COORDINATION_MODES
COMMON_WINDOWS = {
    "selection_2023h2": ("2023-07-01", "2024-01-01"),
    "selection_2024": ("2024-01-01", "2025-01-01"),
}
FOLDS = (
    (
        "calibration_2023q2",
        "2023-04-01",
        "2023-04-01",
        "2023-07-01",
        None,
        False,
    ),
    (
        "selection_2023q3",
        "2023-07-01",
        "2023-07-01",
        "2023-10-01",
        "calibration_2023q2",
        True,
    ),
    (
        "selection_2023q4",
        "2023-10-01",
        "2023-10-01",
        "2024-01-01",
        "selection_2023q3",
        True,
    ),
    (
        "selection_2024",
        "2024-01-01",
        "2024-01-01",
        "2025-01-01",
        "selection_2023q4",
        True,
    ),
)
SYMBOL_FEATURE_SUFFIXES = (
    "flow_now",
    "flow_mean_6h",
    "flow_mean_24h",
    "flow_accel_6h_24h",
    "quote_volume_z720",
    "trade_count_z720",
)
FEATURE_COLUMNS = tuple(
    f"{suffix}_{symbol}"
    for symbol in SYMBOLS
    for suffix in SYMBOL_FEATURE_SUFFIXES
) + (
    "flow_xs_mean_now",
    "flow_xs_std_now",
    "flow_xs_breadth_now",
    "flow_xs_abs_hhi_now",
    "flow_xs_mean_6h",
    "flow_xs_std_6h",
    "flow_xs_breadth_6h",
    "flow_quote_volume_alignment_now",
    "flow_trade_count_alignment_now",
    "pc1_variance_share_168h",
    "flow_effective_rank_168h",
    "pc1_loading_entropy_168h",
    "pc1_projection_now",
    "pc1_projection_6h",
    "pc1_subspace_drift_168h",
    "flow_rotation_cosine_6h_24h",
)
CANDIDATE_NAMES = tuple(
    f"alt_flow_geometry_h144_{mode}" for mode in COORDINATION_MODES
)
SOURCE_READ_COLUMNS = (
    "feature_available_time_utc",
    "symbol",
    "quote_volume_usdt",
    "trade_count",
    "taker_flow_fraction",
    "feature_valid",
)
INPUT_KEYS = (
    "six_alt_flow_panel",
    "six_alt_flow_manifest",
    "market",
    "market_with_oi",
    "funding",
    "premium_1h",
    "gross9_pre2025_anchor",
    "rank7_capacity_evidence",
    "six_alt_source_builder",
    "gross9_context_builder",
    "gross9_runtime",
    "gross9_portfolio_engine",
    "funding_bar_path",
    "execution_engine",
    "fcir_clock",
    "dtac_clock",
    "tgr_clock",
)
PRIOR_CLOCK_KEYS = ("fcir_clock", "dtac_clock", "tgr_clock")


@dataclass(frozen=True)
class Config(runtime.Config):
    preregistration: str = str(PREREGISTRATION)
    output: str = str(OUTPUT)
    docs_output: str = str(DOCS)
    source_panel: str = (
        "data/binance_six_alt_price_free_flow_2023_2026/"
        "six_alt_price_free_flow_1h_2023-01-01_2026-06-01.csv.gz"
    )
    source_manifest: str = (
        "data/binance_six_alt_price_free_flow_2023_2026/build_manifest.json"
    )
    fcir_clock: str = (
        "data/flow_centrality_incubation_relay_clocks_2023_2026.csv.gz"
    )
    dtac_clock: str = (
        "data/discordant_tail_absorption_consensus_clocks_2023_2026.csv.gz"
    )
    tgr_clock: str = "data/ticket_gap_release_clocks_2023_2026.csv.gz"


def validate_frozen_config(cfg: Config) -> None:
    expected = {
        "cutoff": SELECTION_CUTOFF,
        "cost_rate": NORMAL_COST_RATE,
        "stress_cost_rate": STRESS_COST_RATE,
        "leverage": UNIT_LEVERAGE,
        "stride_bars": ANCHOR_STRIDE_BARS,
    }
    for name, value in expected.items():
        if getattr(cfg, name) != value:
            raise RuntimeError(
                f"AFGI frozen config drifted: {name}="
                f"{getattr(cfg, name)!r} != {value!r}"
            )


def load_preregistration(path: str | Path) -> dict[str, Any]:
    observed = _sha256(path)
    if observed != EXPECTED_PREREGISTRATION_SHA256:
        raise RuntimeError(
            f"AFGI preregistration hash drifted: {observed} != "
            f"{EXPECTED_PREREGISTRATION_SHA256}"
        )
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("name") != (
        "alt_flow_geometry_inference_gross9_marginal_battery"
    ):
        raise RuntimeError("unexpected AFGI preregistration")
    if payload.get("physical_selection_cutoff") != SELECTION_CUTOFF:
        raise RuntimeError("AFGI cutoff drifted")
    if tuple(payload["source_contract"]["symbols"]) != SYMBOLS:
        raise RuntimeError("AFGI symbol order drifted")
    contract = payload["feature_contract"]
    if int(contract["count"]) != 52:
        raise RuntimeError("AFGI feature count drifted")
    if tuple(contract["columns"]) != FEATURE_COLUMNS:
        raise RuntimeError("AFGI feature order drifted")
    learner = payload["learner_contract"]
    expected = {
        "estimator": "sklearn.ensemble.ExtraTreesRegressor",
        "sklearn_version": EXPECTED_SKLEARN_VERSION,
        "n_estimators": 256,
        "max_depth": 4,
        "min_samples_leaf": 96,
        "max_features": 0.75,
        "bootstrap": False,
        "score_quantile": 0.95,
        "minimum_score": 0.0,
        "anchor_stride_bars": ANCHOR_STRIDE_BARS,
        "entry_delay_bars": 1,
        "hold_bars": HOLD_BARS,
        "unit_leverage": UNIT_LEVERAGE,
        "normal_cost_per_notional_per_side": NORMAL_COST_RATE,
        "candidate_stress_cost_per_notional_per_side": STRESS_COST_RATE,
    }
    for key, value in expected.items():
        if learner.get(key) != value:
            raise RuntimeError(f"AFGI learner contract drifted: {key}")
    if tuple(learner["seeds"]) != SEEDS:
        raise RuntimeError("AFGI seed contract drifted")
    observed_folds = tuple(
        (
            row["name"],
            row["fit_end_exclusive"],
            row["prediction_start"],
            row["prediction_end_exclusive"],
            row["threshold_source"],
            row["candidate_entries_allowed"],
        )
        for row in payload["expanding_folds"]
    )
    if observed_folds != FOLDS:
        raise RuntimeError("AFGI fold contract drifted")
    universe = payload["candidate_universe"]
    if tuple(row["name"] for row in universe["coordination_modes"]) != (
        COORDINATION_MODES
    ):
        raise RuntimeError("AFGI coordination modes drifted")
    if tuple(float(value) for value in universe["candidate_weight_grid"]) != (
        WEIGHTS
    ):
        raise RuntimeError("AFGI weights drifted")
    if int(universe["portfolio_cells"]) != 12:
        raise RuntimeError("AFGI portfolio-cell count drifted")
    if "close the six-alt hourly" not in payload["prior_family_boundary"][
        "terminal_stop"
    ]:
        raise RuntimeError("AFGI terminal family stop drifted")
    future = payload["future_veto_contract"]
    if future["future_can_rerank"] or future["future_can_repair"]:
        raise RuntimeError("AFGI future repair/rerank enabled")
    return payload


def configured_inputs(cfg: Config) -> dict[str, str]:
    return {
        "six_alt_flow_panel": cfg.source_panel,
        "six_alt_flow_manifest": cfg.source_manifest,
        "market": cfg.market_csv,
        "market_with_oi": cfg.market_with_oi_csv,
        "funding": cfg.funding_csv,
        "premium_1h": cfg.premium_csv,
        "gross9_pre2025_anchor": cfg.gross9_pre2025_anchor,
        "rank7_capacity_evidence": cfg.rank7_capacity_evidence,
        "six_alt_source_builder": (
            "training/build_six_alt_price_free_flow_panel.py"
        ),
        "gross9_context_builder": (
            "training/audit_gross9_fixed_candidate_state_substitution.py"
        ),
        "gross9_runtime": (
            "training/audit_gross9_residual_recovery_extratrees_marginal.py"
        ),
        "gross9_portfolio_engine": (
            "training/portfolio_opt_added_alpha_update.py"
        ),
        "funding_bar_path": (
            "training/audit_rank7_fresh_kimchi_fixed_portfolio.py"
        ),
        "execution_engine": (
            "training/search_inventory_purge_reclaim_alpha.py"
        ),
        "fcir_clock": cfg.fcir_clock,
        "dtac_clock": cfg.dtac_clock,
        "tgr_clock": cfg.tgr_clock,
    }


def validate_inputs(
    cfg: Config,
    preregistration: dict[str, Any],
) -> dict[str, dict[str, str]]:
    configured = configured_inputs(cfg)
    provenance = preregistration["input_provenance"]
    if tuple(configured) != INPUT_KEYS or tuple(provenance) != INPUT_KEYS:
        raise RuntimeError("AFGI input order drifted")
    records: dict[str, dict[str, str]] = {}
    for name in INPUT_KEYS:
        configured_path = Path(configured[name]).resolve(strict=True)
        frozen_path = Path(provenance[name]["path"]).resolve(strict=True)
        if configured_path != frozen_path:
            raise RuntimeError(f"AFGI input path drifted for {name}")
        observed = _sha256(configured_path)
        expected = str(provenance[name]["sha256"])
        if observed != expected:
            raise RuntimeError(
                f"AFGI input hash drifted for {name}: {observed} != {expected}"
            )
        records[name] = {"path": str(configured_path), "sha256": observed}
    return records


def _parse_bool(value: str) -> bool:
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise ValueError(f"invalid boolean text: {value}")


def read_source_prefix(
    path: str | Path,
    *,
    cutoff: str | pd.Timestamp,
) -> pd.DataFrame:
    """Project and parse only the ordered pre-cutoff source prefix."""
    boundary = pd.Timestamp(cutoff)
    rows: list[tuple[Any, ...]] = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = tuple(next(reader))
        if header != SOURCE_COLUMNS:
            raise RuntimeError("AFGI source header drifted")
        positions = {name: header.index(name) for name in SOURCE_READ_COLUMNS}
        previous: pd.Timestamp | None = None
        for raw in reader:
            available = pd.Timestamp(
                raw[positions["feature_available_time_utc"]]
            )
            if previous is not None and available < previous:
                raise RuntimeError("AFGI source is not time ordered")
            previous = available
            if available >= boundary:
                break
            valid = _parse_bool(raw[positions["feature_valid"]])
            rows.append(
                (
                    available,
                    raw[positions["symbol"]],
                    (
                        float(raw[positions["quote_volume_usdt"]])
                        if valid
                        else np.nan
                    ),
                    (
                        float(raw[positions["trade_count"]])
                        if valid
                        else np.nan
                    ),
                    (
                        float(raw[positions["taker_flow_fraction"]])
                        if valid
                        else np.nan
                    ),
                    valid,
                )
            )
    frame = pd.DataFrame(
        rows,
        columns=(
            "date",
            "symbol",
            "quote_volume_usdt",
            "trade_count",
            "taker_flow_fraction",
            "feature_valid",
        ),
    )
    if frame.empty:
        raise RuntimeError("AFGI source prefix is empty")
    if frame.duplicated(["date", "symbol"]).any():
        raise RuntimeError("AFGI source prefix has duplicate hour-symbol rows")
    if set(frame["symbol"]) != set(SYMBOLS):
        raise RuntimeError("AFGI source prefix symbol universe drifted")
    dates = pd.DatetimeIndex(sorted(frame["date"].unique()))
    if len(dates) > 1 and not np.all(np.diff(dates.view("i8")) == 3_600_000_000_000):
        raise RuntimeError("AFGI source prefix is not an exact hourly grid")
    counts = frame.groupby("date", sort=False)["symbol"].nunique()
    if not counts.eq(len(SYMBOLS)).all():
        raise RuntimeError("AFGI source hour does not contain all symbols")
    return frame


def read_funding_prefix(
    path: str | Path,
    *,
    cutoff: str | pd.Timestamp,
) -> pd.DataFrame:
    """Read exact funding timestamps/rates without parsing future numeric rows."""
    boundary = pd.Timestamp(cutoff)
    rows: list[tuple[pd.Timestamp, float]] = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = tuple(next(reader))
        date_position = header.index("date")
        rate_position = header.index("funding_rate")
        previous: pd.Timestamp | None = None
        for raw in reader:
            timestamp = pd.Timestamp(raw[date_position])
            if previous is not None and timestamp < previous:
                raise RuntimeError("AFGI funding source is not time ordered")
            previous = timestamp
            if timestamp >= boundary:
                break
            rows.append((timestamp, float(raw[rate_position])))
    frame = pd.DataFrame(rows, columns=("date", "funding_rate"))
    if frame.empty or frame["date"].duplicated().any():
        raise RuntimeError("AFGI funding prefix is empty or duplicated")
    if not np.isfinite(frame["funding_rate"]).all():
        raise RuntimeError("AFGI funding prefix is non-finite")
    return frame


def robust_z_prior(values: pd.DataFrame) -> pd.DataFrame:
    shifted = values.shift(1)
    rolling = shifted.rolling(window=720, min_periods=672)
    center = rolling.quantile(0.50, interpolation="linear")
    lower = rolling.quantile(0.25, interpolation="linear")
    upper = rolling.quantile(0.75, interpolation="linear")
    scale = (upper - lower) / 1.349
    return (values - center).divide(scale.where(scale > 0.0))


def _population_row_correlation(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    x_centered = x - np.mean(x, axis=1, keepdims=True)
    y_centered = y - np.mean(y, axis=1, keepdims=True)
    covariance = np.mean(x_centered * y_centered, axis=1)
    denominator = np.sqrt(
        np.mean(x_centered**2, axis=1)
        * np.mean(y_centered**2, axis=1)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        output = covariance / denominator
    output[~np.isfinite(denominator) | (denominator <= 0.0)] = np.nan
    return output


def _row_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    denominator = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        output = np.sum(x * y, axis=1) / denominator
    output[~np.isfinite(denominator) | (denominator <= 0.0)] = np.nan
    return output


def _orient_pc1(vector: np.ndarray) -> np.ndarray:
    output = np.asarray(vector, dtype=np.float64).copy()
    total = float(np.sum(output))
    if total < -1e-12:
        output *= -1.0
    elif abs(total) <= 1e-12:
        maximum = float(np.max(np.abs(output)))
        index = int(np.flatnonzero(np.isclose(np.abs(output), maximum))[0])
        if output[index] < 0.0:
            output *= -1.0
    return output


def pca_geometry(
    flow: pd.DataFrame,
    mean6: pd.DataFrame,
) -> pd.DataFrame:
    raw = flow.to_numpy(np.float64)
    mean6_values = mean6.to_numpy(np.float64)
    rows = len(flow)
    output = np.full((rows, 6), np.nan, dtype=np.float64)
    vectors = np.full((rows, len(SYMBOLS)), np.nan, dtype=np.float64)
    for position in range(168, rows):
        window = raw[position - 168 : position]
        if not np.isfinite(window).all():
            continue
        correlation = np.corrcoef(window, rowvar=False)
        if not np.isfinite(correlation).all():
            continue
        eigenvalues, eigenvectors = np.linalg.eigh(correlation)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        trace = float(np.sum(eigenvalues))
        if not np.isfinite(trace) or trace <= 0.0:
            continue
        vector = _orient_pc1(eigenvectors[:, order[0]])
        loading_sum = float(np.sum(np.abs(vector)))
        if loading_sum <= 0.0:
            continue
        probabilities = eigenvalues / trace
        positive = probabilities > 0.0
        effective_rank = float(
            np.exp(-np.sum(probabilities[positive] * np.log(probabilities[positive])))
        )
        loading = np.abs(vector) / loading_sum
        loading_positive = loading > 0.0
        loading_entropy = float(
            -np.sum(loading[loading_positive] * np.log(loading[loading_positive]))
            / np.log(len(SYMBOLS))
        )
        vectors[position] = vector
        output[position, 0] = float(eigenvalues[0] / trace)
        output[position, 1] = effective_rank
        output[position, 2] = loading_entropy
        if np.isfinite(raw[position]).all():
            output[position, 3] = float(np.dot(vector, raw[position]))
        if np.isfinite(mean6_values[position]).all():
            output[position, 4] = float(
                np.dot(vector, mean6_values[position])
            )
        if np.isfinite(vectors[position - 1]).all():
            output[position, 5] = float(
                1.0 - abs(np.dot(vector, vectors[position - 1]))
            )
    return pd.DataFrame(
        output,
        index=flow.index,
        columns=(
            "pc1_variance_share_168h",
            "flow_effective_rank_168h",
            "pc1_loading_entropy_168h",
            "pc1_projection_now",
            "pc1_projection_6h",
            "pc1_subspace_drift_168h",
        ),
    )


def build_feature_frame(source: pd.DataFrame) -> pd.DataFrame:
    index = pd.DatetimeIndex(sorted(source["date"].unique()), name="date")

    def pivot(column: str) -> pd.DataFrame:
        frame = source.pivot(index="date", columns="symbol", values=column)
        return frame.reindex(index=index, columns=SYMBOLS).astype(np.float64)

    valid = source.pivot(
        index="date", columns="symbol", values="feature_valid"
    ).reindex(index=index, columns=SYMBOLS)
    all_valid = valid.fillna(False).astype(bool)
    flow = pivot("taker_flow_fraction").where(all_valid)
    volume = pivot("quote_volume_usdt").where(all_valid)
    count = pivot("trade_count").where(all_valid)
    flow6 = flow.rolling(window=6, min_periods=6).mean()
    flow24 = flow.rolling(window=24, min_periods=24).mean()
    volume_z = robust_z_prior(volume)
    count_z = robust_z_prior(count)

    columns: dict[str, pd.Series] = {}
    for symbol in SYMBOLS:
        columns[f"flow_now_{symbol}"] = flow[symbol]
        columns[f"flow_mean_6h_{symbol}"] = flow6[symbol]
        columns[f"flow_mean_24h_{symbol}"] = flow24[symbol]
        columns[f"flow_accel_6h_24h_{symbol}"] = (
            flow6[symbol] - flow24[symbol]
        )
        columns[f"quote_volume_z720_{symbol}"] = volume_z[symbol]
        columns[f"trade_count_z720_{symbol}"] = count_z[symbol]

    flow_values = flow.to_numpy(np.float64)
    flow6_values = flow6.to_numpy(np.float64)
    flow24_values = flow24.to_numpy(np.float64)
    abs_flow = np.abs(flow_values)
    abs_total = np.sum(abs_flow, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        hhi = np.sum((abs_flow / abs_total[:, None]) ** 2, axis=1)
    hhi[~np.isfinite(abs_total) | (abs_total <= 0.0)] = np.nan

    columns["flow_xs_mean_now"] = pd.Series(
        np.mean(flow_values, axis=1), index=index
    )
    columns["flow_xs_std_now"] = pd.Series(
        np.std(flow_values, axis=1, ddof=0), index=index
    )
    columns["flow_xs_breadth_now"] = pd.Series(
        np.mean(np.sign(flow_values), axis=1), index=index
    )
    columns["flow_xs_abs_hhi_now"] = pd.Series(hhi, index=index)
    columns["flow_xs_mean_6h"] = pd.Series(
        np.mean(flow6_values, axis=1), index=index
    )
    columns["flow_xs_std_6h"] = pd.Series(
        np.std(flow6_values, axis=1, ddof=0), index=index
    )
    columns["flow_xs_breadth_6h"] = pd.Series(
        np.mean(np.sign(flow6_values), axis=1), index=index
    )
    columns["flow_quote_volume_alignment_now"] = pd.Series(
        _population_row_correlation(flow_values, volume_z.to_numpy(float)),
        index=index,
    )
    columns["flow_trade_count_alignment_now"] = pd.Series(
        _population_row_correlation(flow_values, count_z.to_numpy(float)),
        index=index,
    )
    geometry = pca_geometry(flow, flow6)
    for column in geometry:
        columns[column] = geometry[column]
    columns["flow_rotation_cosine_6h_24h"] = pd.Series(
        _row_cosine(flow6_values, flow24_values), index=index
    )
    frame = pd.DataFrame(columns, index=index).loc[:, list(FEATURE_COLUMNS)]
    frame = frame.replace([np.inf, -np.inf], np.nan).astype(np.float64)
    return frame.reset_index()


def _feature_hash(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    dates = pd.to_datetime(frame["date"]).to_numpy(dtype="datetime64[ns]")
    digest.update(np.ascontiguousarray(dates.view("i8")).tobytes())
    values = np.ascontiguousarray(
        frame.loc[:, list(FEATURE_COLUMNS)].to_numpy(np.float64)
    )
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest()


def align_features_to_market(
    market: pd.DataFrame,
    feature_frame: pd.DataFrame,
) -> pd.DataFrame:
    lookup = feature_frame.set_index("date")
    if lookup.index.has_duplicates:
        raise RuntimeError("AFGI feature dates are duplicated")
    aligned = lookup.reindex(
        pd.DatetimeIndex(pd.to_datetime(market["date"]))
    ).reset_index(drop=True)
    if tuple(aligned.columns) != FEATURE_COLUMNS:
        raise RuntimeError("AFGI aligned feature order drifted")
    return aligned.astype(np.float64)


def execution_config(cfg: Config, *, cost_rate: float) -> ExecutionConfig:
    return ExecutionConfig(
        input_csv=cfg.market_csv,
        metrics_csv="",
        funding_csv=cfg.funding_csv,
        output="/tmp/no_write_afgi.json",
        manifest_output="/tmp/no_write_afgi_manifest.json",
        exclude_from=cfg.cutoff,
        leverage=cfg.leverage,
        fee_rate=float(cost_rate),
        slippage_rate=0.0,
    )


def no_stop_trade(
    engine: ExecutionEngine,
    signal: int,
    side: int,
    *,
    hold: int = HOLD_BARS,
) -> Trade | None:
    if side not in (-1, 1):
        raise ValueError("AFGI side must be -1 or 1")
    entry = int(signal) + 1
    exit_position = entry + int(hold)
    if entry >= len(engine.market) or exit_position >= len(engine.market):
        return None
    entry_price = float(engine.open[entry])
    gross_return = side * (
        float(engine.open[exit_position]) / entry_price - 1.0
    )
    held_highs = engine.high[entry:exit_position]
    held_lows = engine.low[entry:exit_position]
    if side > 0:
        favorable_price = float(np.max(held_highs))
        adverse_price = float(np.min(held_lows))
    else:
        favorable_price = float(np.min(held_lows))
        adverse_price = float(np.max(held_highs))
    entry_ns = int(engine.dates.iloc[entry].value)
    exit_ns = int(engine.dates.iloc[exit_position].value)
    left = int(np.searchsorted(engine.funding_times, entry_ns, side="left"))
    right = int(np.searchsorted(engine.funding_times, exit_ns, side="right"))
    factors = (
        1.0
        - float(engine.cfg.leverage)
        * side
        * engine.funding_rates[left:right]
    )
    if not np.isfinite(factors).all() or (factors <= 0.0).any():
        raise RuntimeError("AFGI funding factors are invalid")
    funding_factor = (
        float(np.prod(factors, dtype=float)) if len(factors) else 1.0
    )
    debit_factor = (
        float(np.prod(np.minimum(factors, 1.0), dtype=float))
        if len(factors)
        else 1.0
    )
    leverage = float(engine.cfg.leverage)
    return Trade(
        signal_position=int(signal),
        entry_position=entry,
        exit_position=exit_position,
        side=int(side),
        gross_return=float(gross_return),
        price_factor=max(0.0, 1.0 + leverage * gross_return),
        funding_factor=funding_factor,
        funding_debit_factor=debit_factor,
        favorable_price_factor=max(
            0.0,
            1.0
            + leverage
            * side
            * (favorable_price / entry_price - 1.0),
        ),
        adverse_price_factor=max(
            0.0,
            1.0
            + leverage
            * side
            * (adverse_price / entry_price - 1.0),
        ),
        entry_date=str(engine.dates.iloc[entry]),
    )


def trade_target(trade: Trade, *, leverage: float, cost_rate: float) -> tuple[float, float]:
    cost_factor = 1.0 - float(leverage) * float(cost_rate)
    if cost_factor <= 0.0:
        raise ValueError("AFGI cost factor is non-positive")
    terminal = (
        cost_factor
        * float(trade.price_factor)
        * float(trade.funding_factor)
        * cost_factor
    )
    favorable = cost_factor * float(trade.favorable_price_factor)
    adverse = (
        cost_factor
        * float(trade.funding_debit_factor)
        * float(trade.adverse_price_factor)
    )
    peak = max(1.0, favorable)
    trough = min(adverse, terminal)
    drawdown = max(0.0, 1.0 - trough / max(peak, 1e-15))
    return float(terminal - 1.0), float(drawdown)


def build_targets(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    anchors: np.ndarray,
    cfg: Config,
) -> tuple[np.ndarray, dict[str, Any]]:
    engine = ExecutionEngine(
        market,
        funding,
        execution_config(cfg, cost_rate=cfg.cost_rate),
    )
    targets = np.full((len(anchors), 4), np.nan, dtype=np.float64)
    for index, signal in enumerate(np.asarray(anchors, dtype=np.int64)):
        long_trade = no_stop_trade(engine, int(signal), 1)
        short_trade = no_stop_trade(engine, int(signal), -1)
        if long_trade is None or short_trade is None:
            continue
        long_return, long_adverse = trade_target(
            long_trade,
            leverage=cfg.leverage,
            cost_rate=cfg.cost_rate,
        )
        short_return, short_adverse = trade_target(
            short_trade,
            leverage=cfg.leverage,
            cost_rate=cfg.cost_rate,
        )
        targets[index] = (
            long_return,
            long_adverse,
            short_return,
            short_adverse,
        )
    return targets, {
        "target_hash": runtime._array_hash(targets),
        "finite_targets": int(np.isfinite(targets).all(axis=1).sum()),
        "funding_rows": int(len(funding)),
        "funding_first": str(funding["date"].iloc[0]),
        "funding_last": str(funding["date"].iloc[-1]),
    }


def fold_masks(
    signal_dates: pd.Series,
    exit_dates: pd.Series,
    finite_fit: np.ndarray,
    finite_predict: np.ndarray,
    *,
    fit_end_exclusive: str,
    prediction_start: str,
    prediction_end_exclusive: str,
) -> tuple[np.ndarray, np.ndarray]:
    fit_end = pd.Timestamp(fit_end_exclusive)
    predict_start = pd.Timestamp(prediction_start)
    predict_end = pd.Timestamp(prediction_end_exclusive)
    fit = np.asarray(
        (signal_dates >= pd.Timestamp("2023-01-01"))
        & (exit_dates < fit_end)
        & np.asarray(finite_fit, dtype=bool),
        dtype=bool,
    )
    predict = np.asarray(
        (signal_dates >= predict_start)
        & (signal_dates < predict_end)
        & (exit_dates < predict_end)
        & np.asarray(finite_predict, dtype=bool),
        dtype=bool,
    )
    return fit, predict


def _fit_predict_ensemble(
    matrix: np.ndarray,
    targets: np.ndarray,
    fit: np.ndarray,
    predict: np.ndarray,
    preregistration: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    if int(fit.sum()) < 1_000:
        raise RuntimeError(f"insufficient AFGI fit rows: {int(fit.sum())}")
    if not predict.any():
        raise RuntimeError("empty AFGI prediction fold")
    x_fit = np.asarray(matrix[fit], dtype=np.float64)
    y_fit = np.asarray(targets[fit], dtype=np.float64)
    x_predict = np.asarray(matrix[predict], dtype=np.float64)
    if (
        not np.isfinite(x_fit).all()
        or not np.isfinite(y_fit).all()
        or not np.isfinite(x_predict).all()
    ):
        raise RuntimeError("AFGI model admitted non-finite data")
    learner = preregistration["learner_contract"]
    predictions: list[np.ndarray] = []
    hashes: list[str] = []
    for seed in SEEDS:
        model = ExtraTreesRegressor(
            n_estimators=int(learner["n_estimators"]),
            max_depth=int(learner["max_depth"]),
            min_samples_leaf=int(learner["min_samples_leaf"]),
            max_features=float(learner["max_features"]),
            bootstrap=False,
            random_state=int(seed),
            n_jobs=-1,
        ).fit(x_fit, y_fit)
        predicted = model.predict(x_predict)
        predictions.append(predicted)
        hashes.append(
            hashlib.sha256(
                pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
            ).hexdigest()
        )
    stacked = np.stack(predictions, axis=0)
    return stacked, {
        "fit_rows": int(fit.sum()),
        "predict_rows": int(predict.sum()),
        "model_hashes": hashes,
        "prediction_hash": runtime._array_hash(stacked),
    }


def eligible_anchors(
    matrix: np.ndarray,
    *,
    market_rows: int,
) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or len(values) != int(market_rows):
        raise ValueError("AFGI feature matrix and market rows differ")
    finite_features = np.isfinite(values).all(axis=1)
    target_available = (
        np.arange(int(market_rows), dtype=np.int64) + 1 + HOLD_BARS
        < int(market_rows)
    )
    return np.flatnonzero(
        finite_features & target_available
    ).astype(np.int64)


def expanding_predictions(
    market: pd.DataFrame,
    features: pd.DataFrame,
    funding: pd.DataFrame,
    preregistration: dict[str, Any],
    cfg: Config,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    matrix_all = features.to_numpy(np.float64)
    anchors = eligible_anchors(matrix_all, market_rows=len(market))
    if len(anchors) < 5_000:
        raise RuntimeError("AFGI has insufficient finite hourly anchors")
    targets, target_meta = build_targets(market, funding, anchors, cfg)
    matrix = matrix_all[anchors]
    signal_dates = pd.to_datetime(market["date"]).iloc[anchors].reset_index(
        drop=True
    )
    exit_dates = pd.to_datetime(market["date"]).iloc[
        anchors + 1 + HOLD_BARS
    ].reset_index(drop=True)
    finite_targets = np.isfinite(targets).all(axis=1)
    outputs: dict[str, dict[str, Any]] = {}
    meta: dict[str, Any] = {
        "anchors": int(len(anchors)),
        "anchor_hash": runtime._array_hash(anchors),
        "target": target_meta,
        "folds": {},
    }
    learner = preregistration["learner_contract"]
    for (
        name,
        fit_end,
        prediction_start,
        prediction_end,
        threshold_source,
        candidate_allowed,
    ) in FOLDS:
        fit, predict = fold_masks(
            signal_dates,
            exit_dates,
            finite_targets,
            np.ones(len(anchors), dtype=bool),
            fit_end_exclusive=fit_end,
            prediction_start=prediction_start,
            prediction_end_exclusive=prediction_end,
        )
        predicted, model_meta = _fit_predict_ensemble(
            matrix,
            targets,
            fit,
            predict,
            preregistration,
        )
        score, side = runtime.adjusted_scores(
            predicted,
            risk_lambda=0.5,
            uncertainty_lambda=0.5,
        )
        outputs[name] = {
            "anchors": anchors[predict],
            "score": score,
            "side": side,
            "candidate_allowed": bool(candidate_allowed),
            "threshold_source": threshold_source,
        }
        model_meta.update(
            {
                "fit_end_exclusive": fit_end,
                "prediction_start": prediction_start,
                "prediction_end_exclusive": prediction_end,
                "score_hash": runtime._array_hash(score),
                "side_hash": runtime._array_hash(side),
                "threshold_source": threshold_source,
                "candidate_allowed": bool(candidate_allowed),
            }
        )
        meta["folds"][name] = model_meta
    for name, rows in outputs.items():
        source_name = rows["threshold_source"]
        if source_name is None:
            continue
        calibration = np.asarray(outputs[source_name]["score"], dtype=float)
        finite = calibration[np.isfinite(calibration)]
        if len(finite) < 1_000:
            raise RuntimeError("insufficient AFGI calibration scores")
        raw_quantile = float(
            np.quantile(
                finite,
                float(learner["score_quantile"]),
                method="linear",
            )
        )
        threshold = max(float(learner["minimum_score"]), raw_quantile)
        rows["threshold"] = threshold
        meta["folds"][name]["calibration"] = {
            "source_fold": source_name,
            "scores": int(len(finite)),
            "quantile": float(learner["score_quantile"]),
            "method": "linear",
            "raw_quantile": raw_quantile,
            "threshold": threshold,
            "admission_operator": ">=",
        }
    return outputs, meta


def build_candidate_masks(
    market: pd.DataFrame,
    features: pd.DataFrame,
    funding: pd.DataFrame,
    preregistration: dict[str, Any],
    flat: np.ndarray,
    drawdown: np.ndarray,
    cfg: Config,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    predictions, model_meta = expanding_predictions(
        market, features, funding, preregistration, cfg
    )
    base_long = np.zeros(len(market), dtype=bool)
    base_short = np.zeros(len(market), dtype=bool)
    thresholds: dict[str, float] = {}
    for name, rows in predictions.items():
        if not rows["candidate_allowed"]:
            continue
        anchors = np.asarray(rows["anchors"], dtype=np.int64)
        score = np.asarray(rows["score"], dtype=np.float64)
        side = np.asarray(rows["side"], dtype=np.int8)
        threshold = float(rows["threshold"])
        active = np.isfinite(score) & (score >= threshold)
        base_long[anchors] = active & (side > 0)
        base_short[anchors] = active & (side < 0)
        thresholds[name] = threshold
    candidates: dict[str, dict[str, Any]] = {}
    for mode in COORDINATION_MODES:
        if mode == "unrestricted":
            gate = np.ones(len(market), dtype=bool)
        elif mode == "gross9_flat_at_signal":
            gate = flat
        elif mode == "gross9_drawdown_ge_5pct":
            gate = drawdown >= 0.05
        else:
            raise KeyError(mode)
        long_active = base_long & gate
        short_active = base_short & gate
        name = f"alt_flow_geometry_h144_{mode}"
        candidates[name] = {
            "hold": HOLD_BARS,
            "mode": mode,
            "long_active": long_active,
            "short_active": short_active,
            "raw_active_anchors": int(np.sum(long_active | short_active)),
            "raw_long_anchors": int(long_active.sum()),
            "raw_short_anchors": int(short_active.sum()),
            "thresholds": thresholds,
            "activation_hash": runtime._array_hash(
                np.column_stack([long_active, short_active])
            ),
        }
    if tuple(candidates) != CANDIDATE_NAMES:
        raise RuntimeError("AFGI candidate order drifted")
    return candidates, model_meta


def _install_candidate_universe() -> None:
    names = list(portfolio.SLEEVES)
    for name in CANDIDATE_NAMES:
        if name not in names:
            names.append(name)
    portfolio.SLEEVES = tuple(names)
    portfolio.FAMILIES = {
        **portfolio.FAMILIES,
        **{name: "alt_flow_geometry" for name in CANDIDATE_NAMES},
    }


def _schedule_hash(trades: Iterable[Trade]) -> str:
    rows = [
        [
            int(trade.signal_position),
            int(trade.entry_position),
            int(trade.exit_position),
            int(trade.side),
        ]
        for trade in trades
    ]
    return runtime._json_hash(rows)


def build_schedules(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    masks: dict[str, np.ndarray],
    candidates: dict[str, dict[str, Any]],
    cfg: Config,
) -> tuple[
    dict[str, dict[str, list[Trade]]],
    dict[str, dict[str, dict[str, Any]]],
]:
    engine = ExecutionEngine(
        market,
        funding,
        execution_config(cfg, cost_rate=cfg.cost_rate),
    )
    dates = pd.to_datetime(market["date"])
    schedules: dict[str, dict[str, list[Trade]]] = {}
    meta: dict[str, dict[str, dict[str, Any]]] = {}
    for name, candidate in candidates.items():
        schedules[name] = {}
        meta[name] = {}
        active = candidate["long_active"] | candidate["short_active"]
        positions = np.flatnonzero(active)
        for split, split_mask in masks.items():
            trades: list[Trade] = []
            next_allowed = 0
            for signal in positions[split_mask[positions]]:
                signal = int(signal)
                if signal < next_allowed:
                    continue
                side = 1 if bool(candidate["long_active"][signal]) else -1
                trade = no_stop_trade(engine, signal, side)
                if trade is None:
                    continue
                if (
                    not split_mask[trade.entry_position]
                    or not split_mask[trade.exit_position]
                ):
                    continue
                trades.append(trade)
                next_allowed = int(trade.exit_position) + 1
            schedules[name][split] = trades
            meta[name][split] = {
                "trades": int(len(trades)),
                "longs": int(sum(trade.side > 0 for trade in trades)),
                "shorts": int(sum(trade.side < 0 for trade in trades)),
                "first_entry": (
                    str(dates.iloc[trades[0].entry_position])
                    if trades
                    else None
                ),
                "last_exit": (
                    str(dates.iloc[trades[-1].exit_position])
                    if trades
                    else None
                ),
                "schedule_hash": _schedule_hash(trades),
            }
    return schedules, meta


def append_candidate_paths(
    events: list[dict[str, Any]],
    market: pd.DataFrame,
    funding: pd.DataFrame,
    schedules: dict[str, dict[str, list[Trade]]],
    cfg: Config,
    *,
    cost_rate: float,
) -> dict[str, dict[str, int]]:
    path_cfg = execution_config(cfg, cost_rate=cost_rate)
    counts: dict[str, dict[str, int]] = {}
    cost_factor = 1.0 - cfg.leverage * float(cost_rate)
    for name, split_schedules in schedules.items():
        counts[name] = {}
        for split, trades in split_schedules.items():
            start, end = portfolio.SPLIT_BOUNDS[split]
            path = subaccount_bar_path(
                market,
                funding,
                trades,
                path_cfg,
                start=start,
                end=end,
                hold_bars=lambda _trade: HOLD_BARS,
            )
            event = portfolio.path_event(
                market,
                path,
                split=split,
                sleeve=name,
                trades=trades,
            )
            event["win_count"] = int(
                sum(
                    cost_factor
                    * float(trade.price_factor)
                    * float(trade.funding_factor)
                    * cost_factor
                    > 1.0
                    for trade in trades
                )
            )
            events.append(event)
            counts[name][split] = int(len(trades))
    return counts


def slice_array_data(
    data: dict[str, Any],
    market: pd.DataFrame,
    *,
    start: str,
    end: str,
) -> dict[str, Any]:
    dates = pd.DatetimeIndex(data["dates"])
    start_time = pd.Timestamp(start)
    end_time = pd.Timestamp(end)
    keep = (dates >= start_time) & (dates < end_time)
    if not keep.any():
        raise RuntimeError("AFGI metric slice is empty")
    market_dates = pd.to_datetime(market["date"])
    counts = np.zeros(len(portfolio.SLEEVES), dtype=np.int64)
    for index, name in enumerate(portfolio.SLEEVES):
        entries = np.asarray(data["entry_positions"][name], dtype=np.int64)
        if len(entries):
            entry_dates = market_dates.iloc[entries]
            counts[index] = int(
                ((entry_dates >= start_time) & (entry_dates < end_time)).sum()
            )
    return {
        **data,
        "R": data["R"][:, keep],
        "A": data["A"][:, keep],
        "U": data["U"][:, keep],
        "L": data["L"][:, keep],
        "H": data["H"][:, keep],
        "dates": dates[keep],
        "counts": counts,
        "wins": np.zeros_like(counts),
        "win_rate_available": False,
    }


def common_arrays(
    arrays: dict[str, dict[str, Any]],
    market: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    return {
        "selection_2023h2": slice_array_data(
            arrays["train"],
            market,
            start=COMMON_WINDOWS["selection_2023h2"][0],
            end=COMMON_WINDOWS["selection_2023h2"][1],
        ),
        "selection_2024": arrays["test2024"],
    }


def _years(window: str) -> float:
    start, end = COMMON_WINDOWS[window]
    return (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() / (
        365.25 * 86_400.0
    )


def metric(
    arrays: dict[str, dict[str, Any]],
    window: str,
    weights: dict[str, float],
) -> dict[str, Any]:
    data = arrays[window]
    output = portfolio.strict_metric(data, _years(window), weights)
    if not data.get("win_rate_available", True):
        output["win_rate"] = None
    return output


def standalone_metrics(
    arrays: dict[str, dict[str, Any]],
    candidate: str,
) -> dict[str, dict[str, Any]]:
    return {
        window: metric(arrays, window, {candidate: 1.0})
        for window in COMMON_WINDOWS
    }


def entry_jaccards(
    arrays: dict[str, dict[str, Any]],
    market: pd.DataFrame,
    baseline_weights: dict[str, float],
    candidate: str,
) -> dict[str, float]:
    dates = pd.to_datetime(market["date"])
    start = pd.Timestamp(COMMON_WINDOWS["selection_2023h2"][0])
    end = pd.Timestamp(COMMON_WINDOWS["selection_2024"][1])

    def pooled(name: str) -> set[int]:
        values: set[int] = set()
        for split in ("train", "test2024"):
            for raw in arrays[split]["entry_positions"][name]:
                position = int(raw)
                if start <= dates.iloc[position] < end:
                    values.add(position)
        return values

    candidate_entries = pooled(candidate)
    output: dict[str, float] = {}
    for sleeve, weight in baseline_weights.items():
        if weight <= 0.0:
            continue
        entries = pooled(sleeve)
        union = candidate_entries | entries
        output[sleeve] = (
            float(len(candidate_entries & entries) / len(union))
            if union
            else 0.0
        )
    return output


def _read_clock_entries_prefix(
    path: str | Path,
    *,
    cutoff: str,
) -> pd.DatetimeIndex:
    boundary = pd.Timestamp(cutoff)
    entries: list[pd.Timestamp] = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = tuple(next(reader))
        position = header.index("entry_time")
        previous: pd.Timestamp | None = None
        for raw in reader:
            entry = pd.Timestamp(raw[position])
            if previous is not None and entry < previous:
                raise RuntimeError("AFGI comparator clock is not ordered")
            previous = entry
            if entry >= boundary:
                break
            entries.append(entry)
    return pd.DatetimeIndex(entries)


def _near_share(
    left: pd.DatetimeIndex,
    right: pd.DatetimeIndex,
    *,
    tolerance: pd.Timedelta,
) -> float:
    if not len(left):
        return 0.0
    if not len(right):
        return 0.0
    target = np.sort(right.view("i8"))
    values = left.view("i8")
    tolerance_ns = int(tolerance.value)
    matched = 0
    for value in values:
        index = int(np.searchsorted(target, value))
        distances: list[int] = []
        if index < len(target):
            distances.append(abs(int(target[index]) - int(value)))
        if index > 0:
            distances.append(abs(int(target[index - 1]) - int(value)))
        matched += int(bool(distances) and min(distances) <= tolerance_ns)
    return float(matched / len(values))


def prior_family_overlap(
    schedules: dict[str, dict[str, list[Trade]]],
    market: pd.DataFrame,
    cfg: Config,
) -> dict[str, dict[str, dict[str, float]]]:
    market_dates = pd.to_datetime(market["date"])
    start = pd.Timestamp(COMMON_WINDOWS["selection_2023h2"][0])
    end = pd.Timestamp(COMMON_WINDOWS["selection_2024"][1])
    clocks = {
        "FCIR-12": _read_clock_entries_prefix(cfg.fcir_clock, cutoff=str(end)),
        "DTAC-8": _read_clock_entries_prefix(cfg.dtac_clock, cutoff=str(end)),
        "TGR-12": _read_clock_entries_prefix(cfg.tgr_clock, cutoff=str(end)),
    }
    clocks = {
        name: values[(values >= start) & (values < end)]
        for name, values in clocks.items()
    }
    output: dict[str, dict[str, dict[str, float]]] = {}
    for candidate, splits in schedules.items():
        candidate_entries = pd.DatetimeIndex(
            sorted(
                market_dates.iloc[trade.entry_position]
                for split in ("train", "test2024")
                for trade in splits[split]
                if start
                <= market_dates.iloc[trade.entry_position]
                < end
            )
        )
        candidate_set = set(candidate_entries.view("i8").tolist())
        output[candidate] = {}
        for family, entries in clocks.items():
            family_set = set(entries.view("i8").tolist())
            union = candidate_set | family_set
            output[candidate][family] = {
                "exact_jaccard": (
                    float(len(candidate_set & family_set) / len(union))
                    if union
                    else 0.0
                ),
                "candidate_near_share_12h": _near_share(
                    candidate_entries,
                    entries,
                    tolerance=pd.Timedelta("12h"),
                ),
                "comparator_near_share_12h": _near_share(
                    entries,
                    candidate_entries,
                    tolerance=pd.Timedelta("12h"),
                ),
            }
    return output


def side_counts(
    schedules: dict[str, dict[str, list[Trade]]],
) -> dict[str, dict[str, dict[str, int]]]:
    output: dict[str, dict[str, dict[str, int]]] = {}
    for name, split_rows in schedules.items():
        train = split_rows["train"]
        test = split_rows["test2024"]
        output[name] = {
            "selection_2023h2": {
                "trades": len(train),
                "longs": sum(trade.side > 0 for trade in train),
                "shorts": sum(trade.side < 0 for trade in train),
            },
            "selection_2024": {
                "trades": len(test),
                "longs": sum(trade.side > 0 for trade in test),
                "shorts": sum(trade.side < 0 for trade in test),
            },
        }
    return output


def selection_row(
    preregistration: dict[str, Any],
    *,
    candidate: str,
    candidate_meta: dict[str, Any],
    weight: float,
    baseline: dict[str, dict[str, Any]],
    stats: dict[str, dict[str, Any]],
    comparator: dict[str, dict[str, Any]],
    standalone: dict[str, dict[str, Any]],
    stressed_stats: dict[str, dict[str, Any]],
    stressed_standalone: dict[str, dict[str, Any]],
    sides: dict[str, dict[str, int]],
    max_jaccard: float,
) -> dict[str, Any]:
    standalone_req = preregistration["selection_contract"][
        "standalone_requirements"
    ]
    portfolio_req = preregistration["selection_contract"][
        "portfolio_requirements"
    ]
    improvements = {
        window: float(stats[window]["cagr_to_strict_mdd"])
        - float(comparator[window]["cagr_to_strict_mdd"])
        for window in COMMON_WINDOWS
    }
    retention = {
        window: float(stats[window]["absolute_return_pct"])
        / float(baseline[window]["absolute_return_pct"])
        for window in COMMON_WINDOWS
    }
    mdd_reduction = {
        window: float(baseline[window]["strict_mdd_pct"])
        - float(stats[window]["strict_mdd_pct"])
        for window in COMMON_WINDOWS
    }
    minimum_trades = standalone_req["minimum_trades"]
    side_ok = {}
    for window in COMMON_WINDOWS:
        trades = max(int(sides[window]["trades"]), 1)
        side_ok[window] = (
            sides[window]["longs"] / trades
            >= float(standalone_req["minimum_long_share_each_window"])
            and sides[window]["shorts"] / trades
            >= float(standalone_req["minimum_short_share_each_window"])
        )
    checks = {
        "standalone_positive": all(
            standalone[window]["absolute_return_pct"] > 0.0
            for window in COMMON_WINDOWS
        ),
        "standalone_ratio": all(
            standalone[window]["cagr_to_strict_mdd"]
            >= float(
                standalone_req[
                    "minimum_cagr_to_strict_mdd_each_window"
                ]
            )
            for window in COMMON_WINDOWS
        ),
        "standalone_mdd": all(
            standalone[window]["strict_mdd_pct"]
            <= float(standalone_req["maximum_strict_mdd_pct_each_window"])
            for window in COMMON_WINDOWS
        ),
        "standalone_trades": all(
            standalone[window]["trades"] >= int(minimum_trades[window])
            for window in COMMON_WINDOWS
        ),
        "standalone_side_balance": all(side_ok.values()),
        "stressed_standalone_positive": all(
            stressed_standalone[window]["absolute_return_pct"] > 0.0
            for window in COMMON_WINDOWS
        ),
        "portfolio_mdd": all(
            stats[window]["strict_mdd_pct"]
            <= float(
                portfolio_req["maximum_strict_mdd_pct_each_window"]
            )
            for window in COMMON_WINDOWS
        ),
        "return_retention": all(
            retention[window]
            >= float(
                portfolio_req[
                    "absolute_return_retention_floor_vs_unscaled_gross9"
                ]
            )
            for window in COMMON_WINDOWS
        ),
        "same_gross_improvement": all(
            improvements[window]
            >= float(
                portfolio_req[
                    "minimum_cagr_mdd_improvement_vs_same_gross_prorata_gross9_each_window"
                ]
            )
            for window in COMMON_WINDOWS
        ),
        "mdd_reduction": any(
            mdd_reduction[window] > 0.0 for window in COMMON_WINDOWS
        ),
        "entry_jaccard": max_jaccard
        <= float(
            portfolio_req[
                "maximum_exact_entry_jaccard_vs_any_gross9_sleeve"
            ]
        ),
        "stress_portfolio_positive": all(
            stressed_stats[window]["absolute_return_pct"] > 0.0
            for window in COMMON_WINDOWS
        ),
    }
    mode_ordinal = COORDINATION_MODES.index(str(candidate_meta["mode"]))
    key = (
        int(all(checks.values())),
        min(improvements.values()),
        runtime.signed_geometric_mean(
            improvements["selection_2023h2"],
            improvements["selection_2024"],
        ),
        mdd_reduction["selection_2024"],
        min(
            stressed_standalone[window]["absolute_return_pct"]
            for window in COMMON_WINDOWS
        ),
        -float(weight),
        -int(mode_ordinal),
    )
    return {
        "candidate_name": candidate,
        "hold_bars": HOLD_BARS,
        "coordination_mode": str(candidate_meta["mode"]),
        "coordination_mode_ordinal": int(mode_ordinal),
        "candidate_weight": float(weight),
        "gross": 9.0 + float(weight),
        "passes": bool(all(checks.values())),
        "checks": {name: bool(value) for name, value in checks.items()},
        "stats": stats,
        "same_gross_comparator": comparator,
        "standalone": standalone,
        "stressed_stats": stressed_stats,
        "stressed_standalone": stressed_standalone,
        "standalone_side_counts": sides,
        "same_gross_ratio_improvement": improvements,
        "absolute_return_retention": retention,
        "mdd_reduction_vs_unscaled_gross9": mdd_reduction,
        "max_entry_jaccard": float(max_jaccard),
        "selection_key": list(key),
    }


def rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda row: str(row["candidate_name"]))
    ranked.sort(
        key=lambda row: tuple(float(value) for value in row["selection_key"]),
        reverse=True,
    )
    return ranked


def _metric_cell(metric_row: dict[str, Any]) -> str:
    return (
        f"{metric_row['absolute_return_pct']:.2f}/"
        f"{metric_row['cagr_pct']:.2f}/"
        f"{metric_row['strict_mdd_pct']:.2f}/"
        f"{metric_row['cagr_to_strict_mdd']:.2f}/"
        f"{metric_row['trades']}"
    )


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# Gross9 AFGI-12 pre-2025 marginal battery",
        "",
        "Metric: `absolute return / full-calendar CAGR / strict MDD / CAGR-MDD / trades`.",
        "",
        f"- tested cells: `{payload['tested']}`",
        f"- passing cells: `{payload['passed']}`",
        f"- decision: **{payload['decision']}**",
        f"- future opened: `{payload['future_opened']}`",
        "",
        "## Common-coverage frozen Gross9",
        "",
        f"- 2023 H2: `{_metric_cell(payload['baseline']['selection_2023h2'])}`",
        f"- 2024: `{_metric_cell(payload['baseline']['selection_2024'])}`",
        "",
        "## Ranked cells",
        "",
        "| # | candidate | w | pass | standalone 2023H2 | standalone 2024 | "
        "portfolio 2023H2 | portfolio 2024 | same-gross Δ 2023H2/2024 | "
        "MDD Δ 2023H2/2024 |",
        "|---:|---|---:|:---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(payload["rows"], start=1):
        lines.append(
            "| {rank} | `{name}` | {weight:.2f} | {passed} | {s23} | "
            "{s24} | {p23} | {p24} | {i23:.3f}/{i24:.3f} | "
            "{m23:.2f}/{m24:.2f} |".format(
                rank=rank,
                name=row["candidate_name"],
                weight=row["candidate_weight"],
                passed="Y" if row["passes"] else "N",
                s23=_metric_cell(
                    row["standalone"]["selection_2023h2"]
                ),
                s24=_metric_cell(row["standalone"]["selection_2024"]),
                p23=_metric_cell(row["stats"]["selection_2023h2"]),
                p24=_metric_cell(row["stats"]["selection_2024"]),
                i23=row["same_gross_ratio_improvement"][
                    "selection_2023h2"
                ],
                i24=row["same_gross_ratio_improvement"][
                    "selection_2024"
                ],
                m23=row["mdd_reduction_vs_unscaled_gross9"][
                    "selection_2023h2"
                ],
                m24=row["mdd_reduction_vs_unscaled_gross9"][
                    "selection_2024"
                ],
            )
        )
    lines.extend(
        [
            "",
            "## Fold thresholds",
            "",
            "| fold | fit rows | prediction rows | raw prior q95 | threshold |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, fold in payload["model_meta"]["folds"].items():
        calibration = fold.get("calibration")
        lines.append(
            "| `{name}` | {fit} | {predict} | {raw} | {threshold} |".format(
                name=name,
                fit=fold["fit_rows"],
                predict=fold["predict_rows"],
                raw=(
                    f"{calibration['raw_quantile']:.6f}"
                    if calibration
                    else "calibration only"
                ),
                threshold=(
                    f"{calibration['threshold']:.6f}"
                    if calibration
                    else "—"
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Executable schedules",
            "",
            "| candidate | 2023H2 L/S | 2024 L/S |",
            "|---|---:|---:|",
        ]
    )
    for name in CANDIDATE_NAMES:
        rows = payload["side_counts"][name]
        lines.append(
            f"| `{name}` | {rows['selection_2023h2']['longs']}/"
            f"{rows['selection_2023h2']['shorts']} | "
            f"{rows['selection_2024']['longs']}/"
            f"{rows['selection_2024']['shorts']} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- The source prefix, BTC market, exact funding, and Gross9 context are physically/logically truncated before 2025.",
            "- Q2 2023 predictions calibrate only; every later threshold comes from the immediately prior OOS score fold.",
            "- Exact 10bp stress replays unchanged features, models, scores, sides, gates, and schedules.",
            "- 2025/2026 were not opened and cannot rerank or repair this family.",
            "",
        ]
    )
    return "\n".join(lines)


def run_pre2025(cfg: Config) -> dict[str, Any]:
    validate_frozen_config(cfg)
    preregistration = load_preregistration(cfg.preregistration)
    if sklearn.__version__ != EXPECTED_SKLEARN_VERSION:
        raise RuntimeError(
            f"sklearn version drifted: {sklearn.__version__} != "
            f"{EXPECTED_SKLEARN_VERSION}"
        )
    input_identity = validate_inputs(cfg, preregistration)
    _install_candidate_universe()
    source = read_source_prefix(cfg.source_panel, cutoff=cfg.cutoff)
    feature_frame = build_feature_frame(source)
    if pd.to_datetime(feature_frame["date"]).max() >= pd.Timestamp(cfg.cutoff):
        raise RuntimeError("AFGI feature frame breached cutoff")
    feature_audit = {
        "source_rows": int(len(source)),
        "source_first": str(source["date"].min()),
        "source_last": str(source["date"].max()),
        "feature_rows": int(len(feature_frame)),
        "finite_feature_rows": int(
            np.isfinite(
                feature_frame.loc[
                    :, list(FEATURE_COLUMNS)
                ].to_numpy(float)
            )
            .all(axis=1)
            .sum()
        ),
        "feature_hash": _feature_hash(feature_frame),
        "future_source_rows_parsed": 0,
    }
    funding = read_funding_prefix(cfg.funding_csv, cutoff=cfg.cutoff)
    market, masks, base_events, gross9_meta = runtime.build_gross9_context(cfg)
    features = align_features_to_market(market, feature_frame)
    anchor = json.loads(
        Path(cfg.gross9_pre2025_anchor).read_text(encoding="utf-8")
    )
    baseline_weights = {
        str(name): float(weight) for name, weight in anchor["weights"].items()
    }
    flat, drawdown, state_meta = runtime.gross9_state(
        market, masks, base_events, baseline_weights
    )
    candidates, model_meta = build_candidate_masks(
        market,
        features,
        funding,
        preregistration,
        flat,
        drawdown,
        cfg,
    )
    schedules, schedule_meta = build_schedules(
        market, funding, masks, candidates, cfg
    )
    normal_events = list(base_events)
    normal_counts = append_candidate_paths(
        normal_events,
        market,
        funding,
        schedules,
        cfg,
        cost_rate=cfg.cost_rate,
    )
    normal_arrays = portfolio.split_arrays(normal_events, market, masks)
    stress_events = list(base_events)
    stress_counts = append_candidate_paths(
        stress_events,
        market,
        funding,
        schedules,
        cfg,
        cost_rate=cfg.stress_cost_rate,
    )
    stress_arrays = portfolio.split_arrays(stress_events, market, masks)
    if normal_counts != stress_counts:
        raise RuntimeError("AFGI stress changed schedule counts")
    for name in CANDIDATE_NAMES:
        for split in runtime.SELECTION_SPLITS:
            if not np.array_equal(
                normal_arrays[split]["entry_positions"][name],
                stress_arrays[split]["entry_positions"][name],
            ):
                raise RuntimeError(
                    f"AFGI stress changed entries for {name}/{split}"
                )
    runtime.validate_baseline(normal_arrays, anchor)
    normal_common = common_arrays(normal_arrays, market)
    stress_common = common_arrays(stress_arrays, market)
    baseline = {
        window: metric(normal_common, window, baseline_weights)
        for window in COMMON_WINDOWS
    }
    standalone = {
        name: standalone_metrics(normal_common, name)
        for name in CANDIDATE_NAMES
    }
    stressed_standalone = {
        name: standalone_metrics(stress_common, name)
        for name in CANDIDATE_NAMES
    }
    sides = side_counts(schedules)
    jaccards = {
        name: entry_jaccards(
            normal_arrays, market, baseline_weights, name
        )
        for name in CANDIDATE_NAMES
    }
    family_overlap = prior_family_overlap(schedules, market, cfg)
    rows: list[dict[str, Any]] = []
    for name in CANDIDATE_NAMES:
        for weight in WEIGHTS:
            combined_weights, control_weights = runtime.same_gross_weights(
                baseline_weights, name, float(weight)
            )
            stats = {
                window: metric(normal_common, window, combined_weights)
                for window in COMMON_WINDOWS
            }
            comparator = {
                window: metric(normal_common, window, control_weights)
                for window in COMMON_WINDOWS
            }
            stressed_stats = {
                window: metric(stress_common, window, combined_weights)
                for window in COMMON_WINDOWS
            }
            rows.append(
                selection_row(
                    preregistration,
                    candidate=name,
                    candidate_meta=candidates[name],
                    weight=float(weight),
                    baseline=baseline,
                    stats=stats,
                    comparator=comparator,
                    standalone=standalone[name],
                    stressed_stats=stressed_stats,
                    stressed_standalone=stressed_standalone[name],
                    sides=sides[name],
                    max_jaccard=max(jaccards[name].values(), default=0.0),
                )
            )
    if len(rows) != 12:
        raise RuntimeError(f"AFGI tested-cell drift: {len(rows)}")
    rows = rank_rows(rows)
    passed = [row for row in rows if row["passes"]]
    frozen_top1 = passed[0] if passed else None
    freeze = {
        "preregistration_sha256": _sha256(cfg.preregistration),
        "feature_hash": feature_audit["feature_hash"],
        "baseline_weights": baseline_weights,
        "candidate_activation_hashes": {
            name: candidates[name]["activation_hash"]
            for name in CANDIDATE_NAMES
        },
        "model_meta": model_meta,
        "schedule_meta": schedule_meta,
        "rows": rows,
        "frozen_top1": frozen_top1,
    }
    payload = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "phase": "pre2025_alt_flow_geometry_inference_gross9_marginal",
        "config": asdict(cfg),
        "preregistration": cfg.preregistration,
        "preregistration_sha256": _sha256(cfg.preregistration),
        "input_identity": input_identity,
        "feature_audit": feature_audit,
        "gross9_source_meta": gross9_meta,
        "gross9_observable_state": state_meta,
        "market": {
            "rows": int(len(market)),
            "first": str(pd.to_datetime(market["date"]).iloc[0]),
            "last": str(pd.to_datetime(market["date"]).iloc[-1]),
        },
        "model_meta": model_meta,
        "candidates": {
            name: {
                key: value
                for key, value in candidates[name].items()
                if key not in ("long_active", "short_active")
            }
            for name in CANDIDATE_NAMES
        },
        "schedule_meta": schedule_meta,
        "normal_counts": normal_counts,
        "stress_counts": stress_counts,
        "baseline": baseline,
        "standalone": standalone,
        "stressed_standalone": stressed_standalone,
        "side_counts": sides,
        "entry_jaccards": jaccards,
        "prior_family_overlap": family_overlap,
        "tested": len(rows),
        "passed": len(passed),
        "rows": rows,
        "frozen_top1": frozen_top1,
        "decision": (
            "open_frozen_top1_historical_future_veto"
            if frozen_top1 is not None
            else "reject_and_close_six_alt_flow_model_family"
        ),
        "future_opened": False,
        "future_can_rerank": False,
        "future_can_repair": False,
        "family_closed_if_rejected": True,
        "freeze_hash": runtime._json_hash(freeze),
    }
    _atomic_json(cfg.output, payload)
    docs = Path(cfg.docs_output)
    docs.parent.mkdir(parents=True, exist_ok=True)
    docs.write_text(render(payload), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("pre2025",))
    parser.add_argument("--preregistration", default=Config.preregistration)
    parser.add_argument("--output", default=Config.output)
    parser.add_argument("--docs-output", default=Config.docs_output)
    args = parser.parse_args()
    payload = run_pre2025(
        Config(
            preregistration=args.preregistration,
            output=args.output,
            docs_output=args.docs_output,
        )
    )
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
