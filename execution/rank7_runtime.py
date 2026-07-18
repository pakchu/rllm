"""Bounded runtime contract for the frozen annual ExtraTrees Rank7 policy.

The module intentionally separates immutable artifact validation and pure
single-row scoring from live DB/exchange orchestration.  Any contract drift is
an error; callers are expected to catch it and fail the sleeve closed.
"""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

import training.search_market_braid_alpha as market_braid
import training.search_nested_barrier_witness_alpha as nested_barrier
from preprocessing.market_features import build_market_feature_frame
from training.audit_confirmed_pullback_squeeze_live_parity import (
    decision_mask,
    live_decision_features,
)
from training.audit_weak_feature_responsibility_stability import (
    FEATURE_COLUMNS as RESEARCH_FEATURE_COLUMNS,
    OTHER_COLUMNS,
    PA_COLUMNS,
    causal_shift,
    recent_side,
)
from training.long_component_tp_union_scan import _component_mask
from training.long_regime_interest_gate_validation import build_interest_features
from training.search_liveparity_state_feature_interactions import (
    completed_hourly_features,
    feature_matrix as state_feature_matrix,
    hourly_state_features,
    immutable_anchors,
    rank7_state_runtime_cache_ready,
    state_bank,
    state_bank_from_hourly,
)


FEATURE_COLUMNS = tuple(RESEARCH_FEATURE_COLUMNS)
EXPECTED_SEEDS = (7, 71, 715, 2026, 71515)
EXPECTED_MODEL_PARAMS = {
    "max_depth": 2,
    "min_samples_leaf": 32,
    "max_features": 0.8,
    "bootstrap": False,
}
SOURCE_COLUMNS = ("funding_leg", "premium_leg")
SOURCE_PRIORITY = ("funding", "premium")
NO_BARRIER_BPS = 1_000_000.0


class Rank7BundleError(RuntimeError):
    """Raised when a Rank7 artifact is missing, corrupt, or contract-incompatible."""


class Rank7FeatureError(RuntimeError):
    """Raised when live inputs cannot reproduce the frozen Rank7 feature graph."""


@dataclass(frozen=True)
class FrozenExtraTreesModel:
    """Portable, inference-only ExtraTrees snapshot with no pickle dependency."""

    tree_offsets: np.ndarray
    children_left: np.ndarray
    children_right: np.ndarray
    feature: np.ndarray
    threshold: np.ndarray
    value: np.ndarray
    seed: int
    n_jobs: int = 1

    @property
    def n_estimators(self) -> int:
        return int(len(self.tree_offsets) - 1)

    @property
    def n_features_in_(self) -> int:
        used = self.feature[self.feature >= 0]
        return 0 if not len(used) else int(used.max()) + 1

    @property
    def n_outputs_(self) -> int:
        return int(self.value.shape[1])

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        rows = np.asarray(matrix, dtype=float)
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        if rows.ndim != 2 or rows.shape[1] < self.n_features_in_:
            raise Rank7BundleError("portable Rank7 model input shape mismatch")
        output = np.zeros((len(rows), self.n_outputs_), dtype=float)
        for tree_index in range(self.n_estimators):
            start = int(self.tree_offsets[tree_index])
            stop = int(self.tree_offsets[tree_index + 1])
            for row_index, row in enumerate(rows):
                node = 0
                while True:
                    global_node = start + node
                    feature_index = int(self.feature[global_node])
                    if feature_index < 0:
                        output[row_index] += self.value[global_node]
                        break
                    node = int(
                        self.children_left[global_node]
                        if row[feature_index] <= self.threshold[global_node]
                        else self.children_right[global_node]
                    )
                    if node < 0 or start + node >= stop:
                        raise Rank7BundleError("portable Rank7 tree has an invalid child index")
        return output / float(self.n_estimators)


def _write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write compressed NumPy arrays with fixed ZIP metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.save(buffer, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue())


def save_frozen_extra_trees(model: Any, path: str | Path) -> None:
    """Serialize a fitted sklearn ExtraTreesRegressor into portable arrays."""

    estimators = list(getattr(model, "estimators_", ()))
    if not estimators:
        raise ValueError("ExtraTrees model is not fitted")
    offsets = [0]
    left: list[np.ndarray] = []
    right: list[np.ndarray] = []
    feature: list[np.ndarray] = []
    threshold: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for estimator in estimators:
        tree = estimator.tree_
        left.append(np.asarray(tree.children_left, dtype=np.int32))
        right.append(np.asarray(tree.children_right, dtype=np.int32))
        feature.append(np.asarray(tree.feature, dtype=np.int32))
        threshold.append(np.asarray(tree.threshold, dtype=np.float64))
        value = np.asarray(tree.value, dtype=np.float64).reshape(tree.node_count, -1)
        values.append(value)
        offsets.append(offsets[-1] + int(tree.node_count))
    _write_deterministic_npz(
        Path(path),
        {
            "tree_offsets": np.asarray(offsets, dtype=np.int32),
            "children_left": np.concatenate(left),
            "children_right": np.concatenate(right),
            "feature": np.concatenate(feature),
            "threshold": np.concatenate(threshold),
            "value": np.concatenate(values, axis=0),
        },
    )


def load_frozen_extra_trees(path: str | Path, *, seed: int) -> FrozenExtraTreesModel:
    try:
        with np.load(Path(path), allow_pickle=False) as payload:
            names = {
                "tree_offsets",
                "children_left",
                "children_right",
                "feature",
                "threshold",
                "value",
            }
            if set(payload.files) != names:
                raise Rank7BundleError("portable Rank7 model array set mismatch")
            arrays = {name: np.asarray(payload[name]) for name in names}
    except Rank7BundleError:
        raise
    except Exception as exc:
        raise Rank7BundleError(f"portable Rank7 model unreadable: {exc}") from exc
    offsets = arrays["tree_offsets"].astype(np.int64, copy=False)
    node_count = len(arrays["feature"])
    if (
        offsets.ndim != 1
        or len(offsets) != 301
        or offsets[0] != 0
        or offsets[-1] != node_count
        or np.any(np.diff(offsets) <= 0)
    ):
        raise Rank7BundleError("portable Rank7 tree offsets invalid")
    for name in ("children_left", "children_right", "threshold"):
        if arrays[name].shape != (node_count,):
            raise Rank7BundleError(f"portable Rank7 {name} shape invalid")
    if arrays["value"].shape != (node_count, 2):
        raise Rank7BundleError("portable Rank7 value shape invalid")
    if not np.isfinite(arrays["threshold"]).all() or not np.isfinite(arrays["value"]).all():
        raise Rank7BundleError("portable Rank7 model contains non-finite values")
    return FrozenExtraTreesModel(
        tree_offsets=offsets,
        children_left=arrays["children_left"].astype(np.int32, copy=False),
        children_right=arrays["children_right"].astype(np.int32, copy=False),
        feature=arrays["feature"].astype(np.int32, copy=False),
        threshold=arrays["threshold"].astype(np.float64, copy=False),
        value=arrays["value"].astype(np.float64, copy=False),
        seed=int(seed),
    )


def _utc_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank7_manifest_hash(manifest: dict[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("bundle_manifest_hash", None)
    blob = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Rank7BundleError(message)


def _finite_float(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise Rank7BundleError(f"{field} must be numeric") from exc
    if not np.isfinite(number):
        raise Rank7BundleError(f"{field} must be finite")
    return number


def apply_rank7_delay(
    matrix: np.ndarray,
    *,
    bars: int = 12,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
) -> np.ndarray:
    """Reproduce the frozen research delay exactly.

    The first ``bars`` rows repeat ``matrix[0]``; later rows use the value from
    ``bars`` completed 5-minute bars earlier.  Source identity is independently
    timestamped and therefore restored to its current row.
    """

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("matrix must be a non-empty 2-D array")
    columns = tuple(feature_columns)
    if values.shape[1] != len(columns):
        raise ValueError("matrix width differs from feature_columns")
    delay = int(bars)
    if delay < 0:
        raise ValueError("bars must be non-negative")
    out = np.empty_like(values)
    if delay == 0:
        out[:] = values
    else:
        lead = min(delay, len(values))
        out[:lead] = values[0]
        if len(values) > delay:
            out[delay:] = values[:-delay]
    for name in SOURCE_COLUMNS:
        index = columns.index(name)
        out[:, index] = values[:, index]
    return out


@dataclass(frozen=True)
class Rank7Decision:
    active: bool
    source: str | None
    decision_ts: pd.Timestamp
    signal_id: str
    hold_bars: int
    barrier_exit: dict[str, Any] | None
    prediction_net: float | None
    prediction_adverse: float | None
    score: float | None
    reasons: tuple[str, ...]
    model_version: str

    def metadata(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "source": self.source,
            "prediction_net": self.prediction_net,
            "prediction_adverse": self.prediction_adverse,
            "score": self.score,
            "decision_ts": self.decision_ts.isoformat(),
        }


@dataclass(frozen=True)
class Rank7Bundle:
    root: Path
    manifest: dict[str, Any]
    models: tuple[FrozenExtraTreesModel, ...]
    feature_columns: tuple[str, ...]
    medians: np.ndarray
    clip: tuple[float, float]
    delay_bars: int
    valid_from: pd.Timestamp
    valid_until: pd.Timestamp
    hourly_history: pd.DataFrame | None

    @property
    def model_version(self) -> str:
        return str(self.manifest["model_version"])

    @property
    def thresholds(self) -> dict[str, float]:
        return {key: float(value) for key, value in self.manifest["thresholds"].items()}

    @classmethod
    def load(cls, path: str | Path) -> "Rank7Bundle":
        root = Path(path).expanduser().resolve()
        manifest_path = root / "manifest.json"
        _require(manifest_path.is_file(), f"Rank7 manifest missing: {manifest_path}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise Rank7BundleError(f"Rank7 manifest unreadable: {exc}") from exc
        _require(
            str(manifest.get("bundle_manifest_hash", "")) == rank7_manifest_hash(manifest),
            "Rank7 bundle manifest hash mismatch",
        )

        _require(int(manifest.get("schema_version", -1)) == 1, "unsupported Rank7 schema_version")
        _require(manifest.get("policy_type") == "frozen_annual_rank7", "Rank7 policy_type mismatch")
        _require(manifest.get("strategy_id") == "frozen_annual_rank7", "Rank7 strategy_id mismatch")
        _require(manifest.get("selected_cadence") == "annual", "Rank7 cadence must be annual")
        _require(tuple(manifest.get("seeds", ())) == EXPECTED_SEEDS, "Rank7 seed contract mismatch")
        _require(int(manifest.get("trees_per_seed", 0)) == 300, "Rank7 trees_per_seed must be 300")
        _require(int(manifest.get("prediction_n_jobs", 0)) == 1, "Rank7 prediction_n_jobs must be 1")
        _require(tuple(manifest.get("feature_columns", ())) == FEATURE_COLUMNS, "Rank7 feature order mismatch")
        _require(tuple(manifest.get("source_columns", ())) == SOURCE_COLUMNS, "Rank7 source columns mismatch")
        _require(tuple(manifest.get("source_priority", ())) == SOURCE_PRIORITY, "Rank7 source priority mismatch")
        _require(int(manifest.get("delay_bars", -1)) == 12, "Rank7 delay_bars mismatch")
        _require(manifest.get("delay_initial_fill") == "matrix_0", "Rank7 delay fill mismatch")
        _require(int(manifest.get("anchor_cooldown_bars", 0)) == 144, "Rank7 cooldown mismatch")
        _require(manifest.get("no_overlap") is True, "Rank7 no_overlap must be true")

        model_params = manifest.get("extra_trees_params")
        _require(isinstance(model_params, dict), "Rank7 extra_trees_params missing")
        for key, expected in EXPECTED_MODEL_PARAMS.items():
            _require(model_params.get(key) == expected, f"Rank7 model parameter mismatch: {key}")

        parity = manifest.get("parity")
        _require(isinstance(parity, dict) and parity.get("status") == "passed", "Rank7 parity status is not passed")
        for key in ("feature_parity", "prediction_parity", "schedule_parity"):
            _require(parity.get(key) is True, f"Rank7 parity gate failed: {key}")

        medians = np.asarray(manifest.get("nan_fill_medians", []), dtype=float)
        _require(medians.shape == (len(FEATURE_COLUMNS),), "Rank7 median vector shape mismatch")
        _require(np.isfinite(medians).all(), "Rank7 medians must be finite")
        clip = tuple(_finite_float(value, "clip") for value in manifest.get("clip", ()))
        _require(len(clip) == 2 and clip[0] < clip[1], "Rank7 clip contract invalid")
        valid_from = _utc_timestamp(manifest.get("valid_from"))
        valid_until = _utc_timestamp(manifest.get("valid_until"))
        annual_cutoff = _utc_timestamp(manifest.get("annual_cutoff"))
        _require(valid_from == annual_cutoff and valid_until > valid_from, "Rank7 annual validity contract invalid")

        thresholds = manifest.get("thresholds")
        threshold_keys = {
            "funding_score",
            "premium_score",
            "funding_risk_cap",
            "premium_risk_cap",
            "width_q20",
            "pullback_q40",
        }
        _require(isinstance(thresholds, dict) and set(thresholds) == threshold_keys, "Rank7 threshold keys mismatch")
        for key in threshold_keys:
            _finite_float(thresholds[key], f"thresholds.{key}")
        _finite_float(manifest.get("score_lambda"), "score_lambda")

        exits = manifest.get("exits_by_source")
        _require(isinstance(exits, dict) and set(exits) == set(SOURCE_PRIORITY), "Rank7 source exits mismatch")
        for source, expected_hold in (("funding", 576), ("premium", 144)):
            spec = exits[source]
            _require(int(spec.get("hold_bars", 0)) == expected_hold, f"Rank7 {source} hold mismatch")
            _finite_float(spec.get("take_bps"), f"{source}.take_bps")
            _finite_float(spec.get("stop_bps"), f"{source}.stop_bps")

        model_rows = manifest.get("models")
        _require(isinstance(model_rows, list) and len(model_rows) == len(EXPECTED_SEEDS), "Rank7 model list mismatch")
        _require(tuple(int(row.get("seed", -1)) for row in model_rows) == EXPECTED_SEEDS, "Rank7 model seed order mismatch")
        _require(
            manifest.get("model_format") == "extra_trees_npz_v1",
            "Rank7 model_format mismatch",
        )
        loaded: list[FrozenExtraTreesModel] = []
        for row, seed in zip(model_rows, EXPECTED_SEEDS, strict=True):
            relative = Path(str(row.get("path", "")))
            model_path = (root / relative).resolve()
            _require(model_path.is_relative_to(root), "Rank7 model path escapes bundle")
            _require(model_path.is_file(), f"Rank7 model missing: {relative}")
            _require(_sha256(model_path) == str(row.get("sha256", "")), f"Rank7 model checksum mismatch: {relative}")
            _require(row.get("format") == "extra_trees_npz_v1", f"Rank7 model format mismatch: {relative}")
            _require(int(row.get("n_estimators", 0)) == 300, f"Rank7 tree count mismatch: {relative}")
            _require(int(row.get("n_features", 0)) == len(FEATURE_COLUMNS), f"Rank7 feature width mismatch: {relative}")
            _require(int(row.get("n_outputs", 0)) == 2, f"Rank7 output width mismatch: {relative}")
            model = load_frozen_extra_trees(model_path, seed=seed)
            _require(model.n_estimators == 300, f"Rank7 portable tree count mismatch: {relative}")
            _require(model.n_features_in_ <= len(FEATURE_COLUMNS), f"Rank7 portable feature width mismatch: {relative}")
            _require(model.n_outputs_ == 2, f"Rank7 portable output width mismatch: {relative}")
            loaded.append(model)

        fixture = manifest.get("runtime_prediction_fixture")
        _require(isinstance(fixture, dict), "Rank7 runtime prediction fixture missing")
        fixture_rows = np.asarray(fixture.get("rows", []), dtype=float)
        fixture_expected = np.asarray(fixture.get("expected", []), dtype=float)
        _require(
            fixture_rows.ndim == 2 and fixture_rows.shape[1] == len(FEATURE_COLUMNS),
            "Rank7 runtime fixture row shape mismatch",
        )
        _require(
            fixture_expected.shape == (len(fixture_rows), 2),
            "Rank7 runtime fixture prediction shape mismatch",
        )
        fixture_actual = np.mean(
            np.stack([model.predict(fixture_rows) for model in loaded]), axis=0
        )
        _require(
            np.array_equal(fixture_actual, fixture_expected),
            "Rank7 portable prediction fixture mismatch",
        )

        hourly_history: pd.DataFrame | None = None
        history_row = manifest.get("hourly_history")
        if history_row is not None:
            _require(isinstance(history_row, dict), "Rank7 hourly_history contract invalid")
            relative = Path(str(history_row.get("path", "")))
            history_path = (root / relative).resolve()
            _require(history_path.is_relative_to(root), "Rank7 hourly history path escapes bundle")
            _require(history_path.is_file(), f"Rank7 hourly history missing: {relative}")
            _require(
                _sha256(history_path) == str(history_row.get("sha256", "")),
                "Rank7 hourly history checksum mismatch",
            )
            try:
                hourly_history = pd.read_csv(history_path, compression="infer")
            except Exception as exc:
                raise Rank7BundleError(f"Rank7 hourly history unreadable: {exc}") from exc
            required = ("date", "open", "high", "low", "close", "quote", "buy")
            _require(tuple(hourly_history.columns) == required, "Rank7 hourly history columns mismatch")
            hourly_history["date"] = pd.to_datetime(
                hourly_history["date"], utc=True, errors="raise"
            ).dt.tz_convert(None)
            for column in required[1:]:
                hourly_history[column] = pd.to_numeric(hourly_history[column], errors="coerce")
            _require(
                np.isfinite(hourly_history[list(required[1:])].to_numpy(float)).all(),
                "Rank7 hourly history contains non-finite values",
            )
            _require(
                hourly_history["date"].is_monotonic_increasing
                and not hourly_history["date"].duplicated().any(),
                "Rank7 hourly history order/uniqueness invalid",
            )
            intervals = hourly_history["date"].diff().dropna()
            _require(
                intervals.empty or intervals.eq(pd.Timedelta("1h")).all(),
                "Rank7 hourly history grid is incomplete",
            )

        return cls(
            root=root,
            manifest=manifest,
            models=tuple(loaded),
            feature_columns=FEATURE_COLUMNS,
            medians=medians,
            clip=(float(clip[0]), float(clip[1])),
            delay_bars=12,
            valid_from=valid_from,
            valid_until=valid_until,
            hourly_history=hourly_history,
        )

    def predict(self, row: np.ndarray) -> tuple[float, float]:
        values = np.asarray(row, dtype=float)
        _require(values.shape == (len(self.feature_columns),), "Rank7 score row shape mismatch")
        _require(np.isfinite(values).all(), "Rank7 score row contains non-finite values")
        predictions = []
        for model in self.models:
            prediction = np.asarray(model.predict(values.reshape(1, -1)), dtype=float)
            _require(prediction.shape == (1, 2), "Rank7 prediction shape mismatch")
            predictions.append(prediction[0])
        mean = np.mean(np.stack(predictions), axis=0)
        return float(mean[0]), float(mean[1])


def _normalise_rank7_market(market: pd.DataFrame) -> pd.DataFrame:
    required = {
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base",
        "taker_buy_quote",
        "open_interest",
        "open_interest_available",
        "funding_available",
        "premium_available",
        "spot_close",
        "spot_rows",
        "premium_index_1m_close",
        "premium_rows",
    }
    missing = sorted(required - set(market.columns))
    if missing:
        raise Rank7FeatureError(f"Rank7 market frame missing columns: {missing}")
    out = market.copy()
    out["date"] = pd.to_datetime(out["date"], utc=True, errors="raise").dt.tz_convert(None)
    out = out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    if out.empty:
        raise Rank7FeatureError("Rank7 market frame is empty")
    intervals = out["date"].diff().dropna()
    if len(intervals) and not intervals.eq(pd.Timedelta("5min")).all():
        raise Rank7FeatureError("Rank7 market frame is not a complete 5-minute grid")
    for column in required - {"date"}:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    for column in ("spot_rows", "premium_rows"):
        counts = pd.to_numeric(out[column], errors="coerce").tail(3_000).to_numpy(float)
        if not np.isfinite(counts).all() or not np.equal(counts, 5.0).all():
            raise Rank7FeatureError(f"recent Rank7 {column} values must equal 5")
    latest = out.iloc[-1]
    for column in ("open_interest_available", "funding_available", "premium_available"):
        value = float(latest[column])
        if not np.isfinite(value) or value <= 0.5:
            raise Rank7FeatureError(f"latest {column} must be available")
    if not np.isfinite(float(latest["open_interest"])) or float(latest["open_interest"]) <= 0.0:
        raise Rank7FeatureError("latest open_interest must be positive")
    return out


def _hourly_state_inputs(
    market: pd.DataFrame,
    hourly_history: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    current, _ = completed_hourly_features(market)
    if hourly_history is None:
        return current, hourly_state_features(current)
    history = hourly_history.copy()
    history["date"] = pd.to_datetime(history["date"], utc=True, errors="raise").dt.tz_convert(None)
    history = history.set_index("date").sort_index()
    if current.empty:
        combined = history
    else:
        overlap = history.index.intersection(current.index)
        if len(overlap):
            columns = ["open", "high", "low", "close", "quote", "buy"]
            left = history.loc[overlap, columns].to_numpy(float)
            right = current.loc[overlap, columns].to_numpy(float)
            if not np.allclose(left, right, rtol=1e-9, atol=1e-7, equal_nan=False):
                raise Rank7FeatureError("live/hourly warm-start overlap mismatch")
        first = current.index.min()
        if len(history) and first > history.index.max() + pd.Timedelta("1h"):
            raise Rank7FeatureError("live market tail does not overlap Rank7 hourly warm start")
        combined = pd.concat([history.loc[history.index < first], current]).sort_index()
    combined = combined.loc[combined.index <= market["date"].max()].copy()
    intervals = combined.index.to_series().diff().dropna()
    if len(intervals) and not intervals.eq(pd.Timedelta("1h")).all():
        raise Rank7FeatureError("combined Rank7 hourly state grid is incomplete")
    return combined, hourly_state_features(combined)


def rebuild_rank7_feature_context(
    market: pd.DataFrame,
    *,
    medians: np.ndarray,
    clip: tuple[float, float] = (-20.0, 20.0),
    delay_bars: int = 12,
    hourly_history: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Rebuild the exact causal 40-column Rank7 graph on a 5-minute prefix."""

    market = _normalise_rank7_market(market)
    dates = pd.to_datetime(market["date"])
    base_features = build_market_feature_frame(market, window_size=144)
    raw_features = pd.concat(
        [base_features, build_interest_features(market, base_features)], axis=1
    )
    raw_features = raw_features.loc[:, ~raw_features.columns.duplicated(keep="last")]
    features = live_decision_features(raw_features)
    decisions = decision_mask(dates, "live_hour_signal_bar", window_size=144)
    funding_leg = decisions & _component_mask(features, "funding10_trend70")
    premium_leg = decisions & _component_mask(features, "premium20_mom90")
    base = funding_leg | premium_leg

    if hourly_history is None:
        bank = state_bank(market, dates)
    else:
        hourly, hourly_features = _hourly_state_inputs(market, hourly_history)
        bank = state_bank_from_hourly(hourly, hourly_features, dates)
    valid_state = (bank["kalman"] >= 0) & (bank["bocpd"] >= 0) & (bank["semimarkov"] >= 0)
    base &= valid_state

    barrier_bank = nested_barrier.build_barrier_bank(market)
    long_signal, short_signal, info = nested_barrier.coalesced_barrier_signals(
        market,
        barrier_bank,
        min_coalescence=3,
        touch_width=0.001,
        branch="depleted_continuation",
    )
    nested_side = causal_shift(long_signal.astype(np.int8) - short_signal.astype(np.int8))
    nested24, nested_age = recent_side(nested_side, 288)
    nested48, _ = recent_side(nested_side, 576)

    braid_state = market_braid.build_bar_state(market)
    braid_events = market_braid.market_braid_events(
        braid_state,
        shock_z=2.0,
        passage_z=0.5,
        max_age=144,
        topology_mode="relative_order",
    )
    braid_side = causal_shift(braid_events.signal_side.to_numpy(np.int8))
    braid24, braid_age = recent_side(braid_side, 288)
    braid48, _ = recent_side(braid_side, 576)

    state = state_feature_matrix(bank, funding_leg, premium_leg)
    raw = np.column_stack(
        [
            state,
            *[
                pd.to_numeric(features[column], errors="coerce").to_numpy(float)
                for column in PA_COLUMNS + OTHER_COLUMNS
            ],
        ]
    )
    weak = np.column_stack(
        [
            causal_shift(info["high_work_ratio"], np.nan),
            causal_shift(info["low_work_ratio"], np.nan),
            causal_shift(info["high_coalescence"]),
            causal_shift(info["low_coalescence"]),
            nested24,
            nested48,
            np.minimum(nested_age, 576),
            braid24,
            braid48,
            np.minimum(braid_age, 576),
        ]
    )
    unfilled = np.column_stack([raw, weak])
    median_values = np.asarray(medians, dtype=float)
    if median_values.shape != (len(FEATURE_COLUMNS),) or not np.isfinite(median_values).all():
        raise Rank7FeatureError("Rank7 median vector is invalid")
    lower, upper = map(float, clip)
    if not (np.isfinite(lower) and np.isfinite(upper) and lower < upper):
        raise Rank7FeatureError("Rank7 clip contract is invalid")
    filled = np.clip(np.where(np.isfinite(unfilled), unfilled, median_values), lower, upper)
    matrix = apply_rank7_delay(filled, bars=int(delay_bars))
    anchors = immutable_anchors(base, 144)
    return {
        "market": market,
        "dates": dates,
        "features": features,
        "matrix": matrix,
        "unfilled_matrix": unfilled,
        "base": base,
        "anchors": anchors,
        "funding_leg": funding_leg,
        "premium_leg": premium_leg,
        "nested_side": nested_side,
        "braid_side": braid_side,
        "feature_columns": FEATURE_COLUMNS,
    }


def build_rank7_feature_context(market: pd.DataFrame, bundle: Rank7Bundle) -> dict[str, Any]:
    return rebuild_rank7_feature_context(
        market,
        medians=bundle.medians,
        clip=bundle.clip,
        delay_bars=bundle.delay_bars,
        hourly_history=bundle.hourly_history,
    )


def rank7_barrier_contract(spec: dict[str, Any]) -> dict[str, Any]:
    def bps(key: str) -> float | None:
        value = float(spec[key])
        return None if value >= NO_BARRIER_BPS else value

    return {
        "type": "fixed_bps",
        "take_bps": bps("take_bps"),
        "stop_bps": bps("stop_bps"),
        "entry_price_source": "actual_fill_avg",
        "entry_execution": "market",
        "price_source": "last_trade",
        "same_bar_policy": "stop_before_take",
        "live_touch_policy": "first_aggtrade_touch",
        "stream_gap_policy": "market_close_fail_safe",
        "execution": "market",
        "monitor_interval_sec": 1.0,
    }


def score_rank7_row(
    bundle: Rank7Bundle,
    row: np.ndarray,
    *,
    decision_ts: str | pd.Timestamp,
    is_anchor: bool,
) -> Rank7Decision:
    """Score one delayed Rank7 row and attach its source-owned lifecycle."""

    ts = _utc_timestamp(decision_ts)
    values = np.asarray(row, dtype=float)
    reasons: list[str] = []
    validity_ok = bool(bundle.valid_from <= ts < bundle.valid_until)
    clock_ok = bool(ts.minute == 0 and ts.second == 0 and ts.microsecond == 0)
    reasons.append(f"bundle_validity={'pass' if validity_ok else 'fail'}")
    reasons.append(f"decision_clock={'pass' if clock_ok else 'fail'}")
    reasons.append(f"immutable_anchor={'pass' if is_anchor else 'fail'}")

    source: str | None = None
    if values.shape == (len(bundle.feature_columns),) and np.isfinite(values).all():
        funding = values[bundle.feature_columns.index("funding_leg")] > 0.5
        premium = values[bundle.feature_columns.index("premium_leg")] > 0.5
        source = "funding" if funding else "premium" if premium else None
        reasons.append(f"source_identity={source or 'none'}")
    else:
        reasons.append("feature_row=invalid")

    prediction_net: float | None = None
    prediction_adverse: float | None = None
    score: float | None = None
    model_ok = False
    interaction_ok = True
    if source is not None and values.shape == (len(bundle.feature_columns),) and np.isfinite(values).all():
        prediction_net, prediction_adverse = bundle.predict(values)
        score = prediction_net - float(bundle.manifest["score_lambda"]) * prediction_adverse
        thresholds = bundle.thresholds
        model_ok = bool(
            score >= thresholds[f"{source}_score"]
            and prediction_adverse <= thresholds[f"{source}_risk_cap"]
        )
        reasons.append(f"{source}_score_risk={'pass' if model_ok else 'fail'}")
        if source == "funding":
            width = values[bundle.feature_columns.index("rex_2016_range_width_pct")]
            pullback = values[bundle.feature_columns.index("htf_1d_range_pos")]
            interaction_ok = bool(
                width > thresholds["width_q20"] or pullback <= thresholds["pullback_q40"]
            )
            reasons.append(f"funding_interaction={'pass' if interaction_ok else 'fail'}")

    active = bool(validity_ok and clock_ok and is_anchor and source is not None and model_ok and interaction_ok)
    exit_spec = bundle.manifest["exits_by_source"].get(source) if source else None
    hold = int(exit_spec["hold_bars"]) if exit_spec else 0
    barrier = rank7_barrier_contract(exit_spec) if exit_spec else None
    signal_id = f"frozen_annual_rank7:{bundle.model_version}:{source or 'none'}:{ts.isoformat()}"
    return Rank7Decision(
        active=active,
        source=source,
        decision_ts=ts,
        signal_id=signal_id,
        hold_bars=hold,
        barrier_exit=barrier,
        prediction_net=prediction_net,
        prediction_adverse=prediction_adverse,
        score=score,
        reasons=tuple(reasons),
        model_version=bundle.model_version,
    )
