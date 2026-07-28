"""Build outcome-blind 2023-2024 support signals for the frozen invariant Top-10.

This builder reproduces the pre-evaluation invariant ensemble policies from the
frozen 2026-07-13 manifests, verifies each 2023 executable signal hash, and
exports only next-open-ready signal coordinates (rank/position/date/side).  It
intentionally does not emit returns, forward returns, prices, or backtest
metrics.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import training.search_bidirectional_state_alpha as state_sim
from preprocessing.market_features import build_market_feature_frame
from training.evaluate_invariant_ensemble_uncertainty import (
    INVARIANT_MODEL_SPECS,
    signed_dynamic_policy_masks,
    uncertainty_score_streams,
)
from training.long_regime_combo_scan import _load_market
from training.long_regime_interest_gate_validation import build_interest_features
from training.search_bidirectional_state_alpha import Config
from training.search_invariant_groupdro_alpha import (
    SEED,
    feature_set_definitions,
    half_year_environments,
    stable_feature_ranking,
    tail_labels,
    train_tail_classifier,
)
from training.search_river_contextual_utility_alpha import effective_selection_signal_hash
from training.search_river_online_alpha import causal_rolling_thresholds, validate_output_paths
from training.search_tabicl_foundation_alpha import (
    ANCHOR_STRIDE,
    HOLD_BARS,
    WINDOWS,
    _file_sha256,
    _git_head,
    _signal_hash,
    anchor_dataset,
    feature_groups,
    split_mask_for_anchors,
)

GROUPDRO_MANIFEST = Path("results/invariant_groupdro_top10_manifest_2026-07-13.json")
ENSEMBLE_MANIFEST = Path(
    "results/invariant_ensemble_uncertainty_top10_manifest_2026-07-13.json"
)
EXPECTED_GROUPDRO_MANIFEST_SHA256 = (
    "bf810164f51351b12fb664ea894762a067c54370a7d574d5f3895bdc650636e4"
)
EXPECTED_ENSEMBLE_MANIFEST_SHA256 = (
    "287ee37fee8abd9aa5560cc1b76bc3060df782ec9d17aaedab038094301fc93e"
)
DEFAULT_INPUT_CSV = "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz"
DEFAULT_FUNDING_CSV = (
    "data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz"
)
DEFAULT_PREMIUM_CSV = (
    "data/binance_um_aux_btc_2020_2026/BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz"
)
DEFAULT_OUTPUT = Path("data/invariant_ensemble_top10_pre2025_support_2026-07-28.csv.gz")
DEFAULT_MANIFEST_OUTPUT = Path(
    "results/invariant_ensemble_top10_pre2025_support_manifest_2026-07-28.json"
)
BUILDER_PATH = Path(__file__)
SUPPORT_WINDOW = ("2023-01-01", "2025-01-01")
SELECTION_WINDOW = tuple(WINDOWS["holdout2023"])
SUPPORT_COLUMNS = ("pre_evaluation_rank", "signal_position", "signal_date", "side")
EXPECTED_TOP10_RANKS = tuple(range(1, 11))
# The frozen 2023 manifest hash was computed over every 6-hour anchor in the
# original market frame through 2026-06-02, including zero decisions outside
# 2023. Preserve that historical hash shape without loading post-2024 values.
FROZEN_MANIFEST_MARKET_SIZE = 674_785
FORBIDDEN_SUPPORT_COLUMN_TOKENS = (
    "return",
    "forward",
    "future",
    "pnl",
    "profit",
    "loss",
    "metric",
    "backtest",
    "ratio",
    "cagr",
    "drawdown",
    "mdd",
    "sharpe",
    "win_rate",
    "price",
    "open",
    "high",
    "low",
    "close",
    "equity",
)
METRIC_KEYS = {
    "holdout2023",
    "test2024",
    "eval2025",
    "ytd2026",
    "passes_alpha_pool",
    "passes_live_grade",
    "return_pct",
    "cagr_pct",
    "strict_mdd_pct",
    "ratio",
    "trades",
    "longs",
    "shorts",
    "win_rate",
    "long_win_rate",
    "short_win_rate",
    "sharpe_like",
}
LATER_OUTCOME_KEYS = {
    "test2024",
    "eval2025",
    "ytd2026",
    "future",
    "future_metrics",
    "later_metrics",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(values: Any) -> str:
    array = np.asarray(values, dtype=np.float64)
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def dataframe_sha256(frame: pd.DataFrame, *, chunk_size: int = 50_000) -> str:
    """Hash the exact ordered, post-filter frame used for feature generation."""
    digest = hashlib.sha256()
    schema = {
        "columns": [str(column) for column in frame.columns],
        "dtypes": [str(dtype) for dtype in frame.dtypes],
        "rows": int(len(frame)),
    }
    digest.update(
        json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    for start in range(0, len(frame), int(chunk_size)):
        row_hashes = pd.util.hash_pandas_object(
            frame.iloc[start : start + int(chunk_size)],
            index=False,
            categorize=False,
        ).to_numpy(dtype=np.uint64, copy=False)
        digest.update(np.ascontiguousarray(row_hashes).tobytes())
    return digest.hexdigest()


def deterministic_csv_gzip(frame: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text:
                frame.to_csv(text, index=False, lineterminator="\n")


def assert_no_forbidden_support_columns(columns: Iterable[str]) -> None:
    bad = [
        column
        for column in columns
        if any(token in column.lower() for token in FORBIDDEN_SUPPORT_COLUMN_TOKENS)
    ]
    if bad:
        raise ValueError(f"support output contains outcome-like columns: {bad}")


def strip_metric_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: strip_metric_fields(item)
            for key, item in value.items()
            if key not in METRIC_KEYS
        }
    if isinstance(value, list):
        return [strip_metric_fields(item) for item in value]
    return value


def nested_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            keys.add(str(key))
            keys.update(nested_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.update(nested_keys(item))
    return keys


def load_frozen_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if payload.get("later_metrics_included") is not False:
        raise ValueError(f"{path} must have later_metrics_included=false")
    if tuple(payload.get("selection_window", ())) != SELECTION_WINDOW:
        raise ValueError(f"{path} selection window is not the frozen 2023 window")
    if len(payload.get("top10", [])) != len(EXPECTED_TOP10_RANKS):
        raise ValueError(f"{path} must contain exactly ten frozen policies")
    forbidden = nested_keys(payload).intersection(LATER_OUTCOME_KEYS)
    if forbidden:
        raise ValueError(f"{path} contains later outcome keys: {sorted(forbidden)}")
    return payload


def validate_frozen_manifest_identities(
    groupdro_path: str | Path,
    ensemble_path: str | Path,
) -> dict[str, str]:
    observed = {
        "groupdro": sha256_file(groupdro_path),
        "ensemble_uncertainty": sha256_file(ensemble_path),
    }
    expected = {
        "groupdro": EXPECTED_GROUPDRO_MANIFEST_SHA256,
        "ensemble_uncertainty": EXPECTED_ENSEMBLE_MANIFEST_SHA256,
    }
    for name, expected_hash in expected.items():
        if observed[name] != expected_hash:
            raise ValueError(
                f"{name} frozen manifest hash drift: "
                f"{observed[name]} != {expected_hash}"
            )
    return observed


def load_and_validate_source_manifests(
    groupdro_path: str | Path,
    ensemble_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    groupdro = load_frozen_manifest(groupdro_path)
    ensemble = load_frozen_manifest(ensemble_path)
    source_ref = Path(str(ensemble.get("source_manifest", ""))).name
    if source_ref != Path(groupdro_path).name:
        raise ValueError("ensemble manifest does not reference the supplied groupdro source")
    if ensemble.get("target") != groupdro.get("target"):
        raise ValueError("source target metadata mismatch")
    if ensemble.get("feature_sets") != groupdro.get("feature_sets"):
        raise ValueError("source feature set metadata mismatch")
    if ensemble.get("data_sha256") != groupdro.get("data_sha256"):
        raise ValueError("source input hash metadata mismatch")
    for rank, row in enumerate(ensemble["top10"], start=1):
        if int(row.get("hold_bars", -1)) != HOLD_BARS:
            raise ValueError(f"rank {rank} hold_bars drifted")
        if int(row.get("anchor_stride_bars", -1)) != ANCHOR_STRIDE:
            raise ValueError(f"rank {rank} anchor_stride_bars drifted")
        if row.get("side_policy") not in {"long", "short", "both"}:
            raise ValueError(f"rank {rank} side_policy is unsupported")
        signal_hash = str(row.get("signal_hash", ""))
        if len(signal_hash) != 16:
            raise ValueError(f"rank {rank} signal_hash is malformed")
    return groupdro, ensemble


def validate_input_hashes(paths: dict[str, str], expected: dict[str, Any]) -> dict[str, str | None]:
    observed: dict[str, str | None] = {}
    for key, path in paths.items():
        if path:
            observed[key] = _file_sha256(path)
        else:
            observed[key] = None
        if key in expected and expected[key] != observed[key]:
            raise ValueError(f"{key} input hash drift: {observed[key]} != {expected[key]}")
    return observed


def validate_processed_cutoff(exclude_from: str) -> None:
    expected = pd.Timestamp(SUPPORT_WINDOW[1])
    observed = pd.Timestamp(exclude_from)
    if observed != expected:
        raise ValueError(
            f"exclude_from must equal frozen pre-2025 cutoff {expected.date()}: "
            f"{observed.date()}"
        )


def validate_prepared_cutoff(prepared: dict[str, Any]) -> None:
    dates = pd.to_datetime(prepared["dates"])
    if dates.empty:
        raise ValueError("processed market is empty")
    cutoff = pd.Timestamp(SUPPORT_WINDOW[1])
    if bool((dates >= cutoff).any()):
        raise ValueError(
            "processed market contains rows at or after the frozen "
            f"pre-2025 cutoff {cutoff.date()}"
        )


def prepare_invariant_inputs(
    cfg: Config,
    source_manifest: dict[str, Any],
) -> dict[str, Any]:
    for name, bounds in WINDOWS.items():
        state_sim.W[name] = bounds
    market = _load_market(cfg)
    dates = pd.to_datetime(market["date"])
    base = build_market_feature_frame(market, window_size=144)
    features = pd.concat([base, build_interest_features(market, base)], axis=1)
    features = features.loc[:, ~features.columns.duplicated(keep="last")]
    positions, targets, _ = anchor_dataset(market, features)
    signal_dates = dates.iloc[positions].reset_index(drop=True)
    masks = {
        name: split_mask_for_anchors(dates, positions, *bounds)
        for name, bounds in WINDOWS.items()
    }
    fit_mask = masks["fit2020_2022"]
    full_columns = feature_groups(features)["full"]
    full_matrix = features.iloc[positions][full_columns].to_numpy(float)
    ranking = stable_feature_ranking(full_matrix, full_columns, targets, signal_dates, fit_mask)
    feature_sets = feature_set_definitions(ranking)
    reproduced_feature_sets = {name: [row["name"] for row in rows] for name, rows in feature_sets.items()}
    if reproduced_feature_sets != source_manifest["feature_sets"]:
        raise ValueError("feature sets do not reproduce frozen source manifest")
    labels, label_thresholds = tail_labels(targets, fit_mask)
    if not np.allclose(label_thresholds, source_manifest["target"]["fit_thresholds"]):
        raise ValueError("tail thresholds do not reproduce frozen source manifest")
    return {
        "market": market,
        "dates": dates,
        "positions": positions,
        "targets": targets,
        "signal_dates": signal_dates,
        "masks": masks,
        "fit_mask": fit_mask,
        "full_matrix": full_matrix,
        "feature_sets": feature_sets,
        "reproduced_feature_sets": reproduced_feature_sets,
        "labels": labels,
        "environments": half_year_environments(signal_dates),
    }


def train_transformed_streams(prepared: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    transformed_streams: dict[str, np.ndarray] = {}
    diagnostics: dict[str, Any] = {}
    for feature_set_name, feature_rows in prepared["feature_sets"].items():
        indices = [int(row["index"]) for row in feature_rows]
        matrix = prepared["full_matrix"][:, indices]
        members: dict[str, np.ndarray] = {}
        member_diagnostics: dict[str, Any] = {}
        for model_id, architecture, objective, vrex_penalty in INVARIANT_MODEL_SPECS:
            full_model_id = f"{model_id}_{feature_set_name}"
            print(f"training {full_model_id}", file=sys.stderr, flush=True)
            scores, _training_diagnostics = train_tail_classifier(
                matrix,
                prepared["labels"],
                prepared["fit_mask"],
                prepared["environments"],
                architecture=architecture,
                objective=objective,
                vrex_penalty=vrex_penalty,
            )
            members[full_model_id] = scores
            member_diagnostics[full_model_id] = {
                "architecture": architecture,
                "objective": objective,
                "vrex_penalty": vrex_penalty,
                "seed": int(SEED),
                "state_dict_sha256": _training_diagnostics[
                    "state_dict_sha256"
                ],
                "scaler_median_sha256": array_sha256(
                    _training_diagnostics["scaler_median"]
                ),
                "scaler_scale_sha256": array_sha256(
                    _training_diagnostics["scaler_scale"]
                ),
            }
        streams, uncertainty_diagnostics = uncertainty_score_streams(members)
        diagnostics[feature_set_name] = {
            "members": member_diagnostics,
            "uncertainty": uncertainty_diagnostics,
        }
        for transform_name, scores in streams.items():
            stream_id = f"invariant_{feature_set_name}_{transform_name}"
            transformed_streams[stream_id] = scores
    return transformed_streams, diagnostics


def policy_masks_for_frozen_row(
    frozen_row: dict[str, Any],
    scores: np.ndarray,
    positions: np.ndarray,
    market_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    low_thresholds, high_thresholds = causal_rolling_thresholds(
        scores,
        window=int(frozen_row["rolling_score_window_anchors"]),
        quantile=float(frozen_row["score_quantile"]),
    )
    return signed_dynamic_policy_masks(
        scores,
        positions,
        market_size,
        side_policy=str(frozen_row["side_policy"]),
        low_thresholds=low_thresholds,
        high_thresholds=high_thresholds,
    )


def effective_signal_rows(
    dates: pd.Series,
    long_active: np.ndarray,
    short_active: np.ndarray,
    *,
    rank: int,
    window: tuple[str, str] = SUPPORT_WINDOW,
    market_size: int | None = None,
    hold_bars: int = HOLD_BARS,
    stride_bars: int = ANCHOR_STRIDE,
    minimum_signal_position: int = 143,
) -> list[dict[str, Any]]:
    if market_size is None:
        market_size = len(dates)
    start, end = map(pd.Timestamp, window)
    date_values = pd.to_datetime(dates).reset_index(drop=True)
    window_mask = np.asarray((date_values >= start) & (date_values < end), dtype=bool)
    positions = np.arange(
        int(minimum_signal_position),
        int(market_size) - int(hold_bars) - 2,
        int(stride_bars),
        dtype=np.int64,
    )
    candidate = positions[
        window_mask[positions]
        & (np.asarray(long_active, dtype=bool)[positions] | np.asarray(short_active, dtype=bool)[positions])
    ]
    rows: list[dict[str, Any]] = []
    next_allowed = 0
    for signal_pos in candidate:
        if int(signal_pos) < next_allowed:
            continue
        is_long = bool(long_active[signal_pos]) and not bool(short_active[signal_pos])
        is_short = bool(short_active[signal_pos]) and not bool(long_active[signal_pos])
        if not is_long and not is_short:
            continue
        exit_pos = int(signal_pos) + 1 + int(hold_bars)
        if exit_pos >= int(market_size) or not bool(window_mask[exit_pos]):
            continue
        rows.append(
            {
                "pre_evaluation_rank": int(rank),
                "signal_position": int(signal_pos),
                "signal_date": date_values.iloc[int(signal_pos)].strftime("%Y-%m-%d %H:%M:%S"),
                "side": "long" if is_long else "short",
            }
        )
        next_allowed = exit_pos + 1
    return rows


def effective_hash_from_rows(
    rows: list[dict[str, Any]],
    *,
    dates: pd.Series,
    window: tuple[str, str] = SELECTION_WINDOW,
    hold_bars: int = HOLD_BARS,
    stride_bars: int = ANCHOR_STRIDE,
    minimum_signal_position: int = 143,
    hash_market_size: int | None = None,
) -> str:
    date_values = pd.to_datetime(dates).reset_index(drop=True)
    start, end = map(pd.Timestamp, window)
    market_size = (
        len(date_values) if hash_market_size is None else int(hash_market_size)
    )
    if market_size < len(date_values):
        raise ValueError("hash_market_size cannot truncate processed dates")
    positions = np.arange(
        int(minimum_signal_position),
        market_size - int(hold_bars) - 2,
        int(stride_bars),
        dtype=np.int64,
    )
    selected_positions: dict[int, str] = {}
    for row in rows:
        signal_position = int(row["signal_position"])
        exit_position = signal_position + 1 + int(hold_bars)
        if exit_position >= len(date_values):
            continue
        signal_date = pd.Timestamp(row["signal_date"])
        exit_date = pd.Timestamp(date_values.iloc[exit_position])
        if (
            start <= signal_date < end
            and start <= exit_date < end
        ):
            selected_positions[signal_position] = str(row["side"])
    long = np.array([selected_positions.get(int(pos)) == "long" for pos in positions], dtype=bool)
    short = np.array([selected_positions.get(int(pos)) == "short" for pos in positions], dtype=bool)
    return _signal_hash(long, short)


def support_rows_sha256(rows: list[dict[str, Any]]) -> str:
    canonical = sorted(
        (
            int(row["pre_evaluation_rank"]),
            int(row["signal_position"]),
            str(row["signal_date"]),
            str(row["side"]),
        )
        for row in rows
    )
    return hashlib.sha256(
        json.dumps(canonical, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def verify_frozen_signal_hash(
    *,
    rank: int,
    expected_hash: str,
    market: pd.DataFrame,
    dates: pd.Series,
    long_active: np.ndarray,
    short_active: np.ndarray,
    window: tuple[str, str] = SELECTION_WINDOW,
    hash_market_size: int | None = None,
) -> str:
    rows = effective_signal_rows(
        dates,
        long_active,
        short_active,
        rank=rank,
        window=window,
        market_size=len(market),
    )
    observed_hash = effective_hash_from_rows(
        rows,
        dates=dates,
        window=window,
        hash_market_size=hash_market_size,
    )
    direct_hash = effective_selection_signal_hash(
        market,
        dates,
        long_active,
        short_active,
        window=window,
    )
    if hash_market_size is None and observed_hash != direct_hash:
        raise ValueError(
            f"rank {rank} row/direct signal hash mismatch: "
            f"{observed_hash} != {direct_hash}"
        )
    if observed_hash != str(expected_hash):
        raise ValueError(
            f"rank {rank} 2023 signal hash drift: {observed_hash} != {expected_hash}"
        )
    return observed_hash


def verify_support_frame(frame: pd.DataFrame) -> None:
    assert_no_forbidden_support_columns(frame.columns)
    if tuple(frame.columns) != SUPPORT_COLUMNS:
        raise ValueError(f"support columns drifted: {list(frame.columns)}")
    dates = pd.to_datetime(frame["signal_date"], errors="raise")
    if not ((dates >= pd.Timestamp(SUPPORT_WINDOW[0])) & (dates < pd.Timestamp(SUPPORT_WINDOW[1]))).all():
        raise ValueError("support contains rows outside 2023-01-01..2025-01-01")
    if not set(frame["side"].unique()).issubset({"long", "short"}):
        raise ValueError("support side column contains unsupported values")
    ranks = pd.to_numeric(frame["pre_evaluation_rank"], errors="raise")
    positions = pd.to_numeric(frame["signal_position"], errors="raise")
    if not np.equal(ranks, np.floor(ranks)).all():
        raise ValueError("support ranks must be integers")
    if not np.equal(positions, np.floor(positions)).all():
        raise ValueError("support positions must be integers")
    if not ranks.between(min(EXPECTED_TOP10_RANKS), max(EXPECTED_TOP10_RANKS)).all():
        raise ValueError("support rank is outside the frozen Top10")
    if (positions < 0).any():
        raise ValueError("support positions must be non-negative")
    if frame.duplicated(["pre_evaluation_rank", "signal_position"]).any():
        raise ValueError("support contains duplicate rank/position rows")


def verify_support_alignment(
    frame: pd.DataFrame,
    *,
    dates: pd.Series,
    required_ranks: tuple[int, ...] = EXPECTED_TOP10_RANKS,
    hold_bars: int = HOLD_BARS,
    stride_bars: int = ANCHOR_STRIDE,
    minimum_signal_position: int = 143,
) -> dict[str, dict[str, int]]:
    if set(frame["pre_evaluation_rank"].astype(int)) != set(required_ranks):
        raise ValueError("support does not contain every frozen Top10 rank")
    date_values = pd.to_datetime(dates).reset_index(drop=True)
    positions = frame["signal_position"].astype(int).to_numpy()
    if (positions >= len(date_values)).any():
        raise ValueError("support position exceeds the processed market")
    if ((positions - int(minimum_signal_position)) % int(stride_bars) != 0).any():
        raise ValueError("support position is off the frozen anchor clock")
    observed_dates = pd.to_datetime(frame["signal_date"]).to_numpy()
    expected_dates = date_values.iloc[positions].to_numpy()
    if not np.array_equal(observed_dates, expected_dates):
        raise ValueError("support position/date alignment drifted")

    counts: dict[str, dict[str, int]] = {}
    for rank in required_ranks:
        rows = frame.loc[
            frame["pre_evaluation_rank"].astype(int) == int(rank)
        ].sort_values("signal_position")
        rank_positions = rows["signal_position"].astype(int).to_numpy()
        if len(rank_positions) > 1 and (
            np.diff(rank_positions) < int(hold_bars) + 2
        ).any():
            raise ValueError(f"rank {rank} support contains overlapping holds")
        year_counts = (
            pd.to_datetime(rows["signal_date"])
            .dt.year.value_counts()
            .sort_index()
            .to_dict()
        )
        counts[str(rank)] = {
            "2023": int(year_counts.get(2023, 0)),
            "2024": int(year_counts.get(2024, 0)),
            "total": int(len(rows)),
        }
        if counts[str(rank)]["2023"] == 0 or counts[str(rank)]["2024"] == 0:
            raise ValueError(f"rank {rank} lacks 2023 or 2024 support")
    return counts


def build_support(args: argparse.Namespace) -> dict[str, Any]:
    validate_output_paths(str(args.output), str(args.manifest_output))
    validate_processed_cutoff(str(args.exclude_from))
    if Path(args.output).exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite support CSV: {args.output}")
    if Path(args.manifest_output).exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite support manifest: {args.manifest_output}")
    manifest_hashes = validate_frozen_manifest_identities(
        args.groupdro_manifest,
        args.ensemble_manifest,
    )
    groupdro_manifest, ensemble_manifest = load_and_validate_source_manifests(
        args.groupdro_manifest,
        args.ensemble_manifest,
    )
    data_hashes = validate_input_hashes(
        {"market": args.input_csv, "funding": args.funding_csv, "premium": args.premium_csv},
        ensemble_manifest["data_sha256"],
    )
    cfg = Config(
        input_csv=args.input_csv,
        output=str(args.output),
        funding_csv=args.funding_csv,
        premium_csv=args.premium_csv,
        exclude_from=args.exclude_from,
    )
    prepared = prepare_invariant_inputs(cfg, ensemble_manifest)
    validate_prepared_cutoff(prepared)
    streams, diagnostics = train_transformed_streams(prepared)

    support_rows: list[dict[str, Any]] = []
    verified_top10: list[dict[str, Any]] = []
    for rank, frozen_row in enumerate(ensemble_manifest["top10"], start=1):
        stream_id = str(frozen_row["stream_id"])
        if stream_id not in streams:
            raise ValueError(f"frozen stream not reproduced: {stream_id}")
        long_active, short_active = policy_masks_for_frozen_row(
            frozen_row,
            streams[stream_id],
            prepared["positions"],
            len(prepared["market"]),
        )
        observed_hash = verify_frozen_signal_hash(
            rank=rank,
            expected_hash=str(frozen_row["signal_hash"]),
            market=prepared["market"],
            dates=prepared["dates"],
            long_active=long_active,
            short_active=short_active,
            hash_market_size=FROZEN_MANIFEST_MARKET_SIZE,
        )
        rows = effective_signal_rows(
            prepared["dates"],
            long_active,
            short_active,
            rank=rank,
            window=SUPPORT_WINDOW,
            market_size=len(prepared["market"]),
        )
        emitted_2023_hash = effective_hash_from_rows(
            rows,
            dates=prepared["dates"],
            window=SELECTION_WINDOW,
            hash_market_size=FROZEN_MANIFEST_MARKET_SIZE,
        )
        if emitted_2023_hash != observed_hash:
            raise ValueError(
                f"rank {rank} emitted 2023 hash drift: "
                f"{emitted_2023_hash} != {observed_hash}"
            )
        support_rows.extend(rows)
        verified_top10.append(
            {
                "pre_evaluation_rank": rank,
                **strip_metric_fields(frozen_row),
                "verified_2023_signal_hash": observed_hash,
                "emitted_2023_signal_hash": emitted_2023_hash,
                "support_rows_sha256": support_rows_sha256(rows),
                "support_signal_count_2023_2024": len(rows),
            }
        )

    support = pd.DataFrame(support_rows, columns=SUPPORT_COLUMNS)
    if not support.empty:
        support = support.sort_values(
            ["signal_date", "pre_evaluation_rank", "signal_position", "side"],
            kind="mergesort",
        ).reset_index(drop=True)
    verify_support_frame(support)
    support_counts = verify_support_alignment(
        support,
        dates=prepared["dates"],
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest_output).parent.mkdir(parents=True, exist_ok=True)
    deterministic_csv_gzip(support, args.output)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_head_before_build": _git_head(),
        "support_window": list(SUPPORT_WINDOW),
        "selection_window": list(SELECTION_WINDOW),
        "later_metrics_included": False,
        "outcome_blind_contract": {
            "model_fit_window": "fit2020_2022 only",
            "selection_source": "frozen 2023 pre-evaluation Top10 manifests",
            "post_2023_outcomes_used": False,
            "emitted_columns": list(SUPPORT_COLUMNS),
            "forbidden_support_column_tokens": list(FORBIDDEN_SUPPORT_COLUMN_TOKENS),
            "raw_source_files_used_for_frozen_identity_only": True,
            "feature_generation_cutoff": SUPPORT_WINDOW[1],
            "frozen_manifest_hash_padding": {
                "market_rows": FROZEN_MANIFEST_MARKET_SIZE,
                "post_2024_values_loaded": False,
                "reason": (
                    "historical 2023 signal hashes include zero-decision "
                    "anchors through the original 2026-06-02 frame"
                ),
            },
        },
        "source_manifests": {
            "groupdro": str(args.groupdro_manifest),
            "groupdro_sha256": manifest_hashes["groupdro"],
            "ensemble_uncertainty": str(args.ensemble_manifest),
            "ensemble_uncertainty_sha256": manifest_hashes[
                "ensemble_uncertainty"
            ],
        },
        "builder": {
            "path": str(BUILDER_PATH),
            "sha256": sha256_file(BUILDER_PATH),
        },
        "inputs_sha256": data_hashes,
        "processed_pre2025_market_frame_sha256": dataframe_sha256(
            prepared["market"]
        ),
        "feature_sets": prepared["reproduced_feature_sets"],
        "feature_set_parity": {
            "matches_groupdro_manifest": prepared["reproduced_feature_sets"] == groupdro_manifest["feature_sets"],
            "matches_ensemble_manifest": prepared["reproduced_feature_sets"] == ensemble_manifest["feature_sets"],
        },
        "top10": verified_top10,
        "support_csv": str(args.output),
        "support_csv_sha256": sha256_file(args.output),
        "support_rows": int(len(support)),
        "support_counts_by_rank_and_year": support_counts,
        "support_columns": list(support.columns),
        "no_future_rows": bool(
            support.empty
            or pd.to_datetime(support["signal_date"]).lt(pd.Timestamp(SUPPORT_WINDOW[1])).all()
        ),
        "no_outcome_like_support_columns": True,
        "deterministic_gzip": {"mtime": 0, "filename": ""},
        "processed_market": {
            "exclude_from": str(args.exclude_from),
            "first_timestamp": str(prepared["dates"].iloc[0]),
            "last_timestamp": str(prepared["dates"].iloc[-1]),
            "rows": int(len(prepared["market"])),
        },
        "diagnostics": {
            "sample_counts": {
                "fit_2020_2022_anchors": int(
                    prepared["masks"]["fit2020_2022"].sum()
                ),
                "selection_2023_anchors": int(
                    prepared["masks"]["holdout2023"].sum()
                ),
                "support_2024_anchors": int(
                    prepared["masks"]["test2024"].sum()
                ),
            },
            "training": diagnostics,
        },
    }
    forbidden_manifest_fields = nested_keys(manifest).intersection(METRIC_KEYS)
    if forbidden_manifest_fields:
        raise ValueError(
            "support manifest contains outcome metrics: "
            f"{sorted(forbidden_manifest_fields)}"
        )
    Path(args.manifest_output).write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groupdro-manifest", default=str(GROUPDRO_MANIFEST))
    parser.add_argument("--ensemble-manifest", default=str(ENSEMBLE_MANIFEST))
    parser.add_argument("--input-csv", default=DEFAULT_INPUT_CSV)
    parser.add_argument("--funding-csv", default=DEFAULT_FUNDING_CSV)
    parser.add_argument("--premium-csv", default=DEFAULT_PREMIUM_CSV)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--manifest-output", default=str(DEFAULT_MANIFEST_OUTPUT))
    parser.add_argument("--exclude-from", default=SUPPORT_WINDOW[1])
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = build_support(args)
    print(
        json.dumps(
            {
                "support_rows": manifest["support_rows"],
                "top10_verified": len(manifest["top10"]),
                "support_csv": manifest["support_csv"],
                "manifest": str(args.manifest_output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
