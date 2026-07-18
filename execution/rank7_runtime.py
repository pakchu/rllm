"""Bounded runtime contract for the frozen annual ExtraTrees Rank7 policy.

The module intentionally separates immutable artifact validation and pure
single-row scoring from live DB/exchange orchestration.  Any contract drift is
an error; callers are expected to catch it and fail the sleeve closed.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

from training.audit_weak_feature_responsibility_stability import (
    FEATURE_COLUMNS as RESEARCH_FEATURE_COLUMNS,
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


def _utc_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    models: tuple[ExtraTreesRegressor, ...]
    feature_columns: tuple[str, ...]
    medians: np.ndarray
    clip: tuple[float, float]
    delay_bars: int
    valid_from: pd.Timestamp
    valid_until: pd.Timestamp

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
        loaded: list[ExtraTreesRegressor] = []
        for row, seed in zip(model_rows, EXPECTED_SEEDS, strict=True):
            relative = Path(str(row.get("path", "")))
            model_path = (root / relative).resolve()
            _require(model_path.is_relative_to(root), "Rank7 model path escapes bundle")
            _require(model_path.is_file(), f"Rank7 model missing: {relative}")
            _require(_sha256(model_path) == str(row.get("sha256", "")), f"Rank7 model checksum mismatch: {relative}")
            try:
                model = joblib.load(model_path)
            except Exception as exc:
                raise Rank7BundleError(f"Rank7 model unreadable: {relative}: {exc}") from exc
            _require(isinstance(model, ExtraTreesRegressor), f"Rank7 model type mismatch: {relative}")
            params = model.get_params(deep=False)
            _require(int(params["n_estimators"]) == 300, f"Rank7 tree count mismatch: {relative}")
            _require(int(params["random_state"]) == seed, f"Rank7 random_state mismatch: {relative}")
            for key, expected in EXPECTED_MODEL_PARAMS.items():
                _require(params[key] == expected, f"Rank7 loaded model parameter mismatch: {relative}:{key}")
            _require(int(model.n_features_in_) == len(FEATURE_COLUMNS), f"Rank7 model feature width mismatch: {relative}")
            _require(int(model.n_outputs_) == 2, f"Rank7 model output width mismatch: {relative}")
            model.n_jobs = 1
            loaded.append(model)

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
        )

    def predict(self, row: np.ndarray) -> tuple[float, float]:
        values = np.asarray(row, dtype=float)
        _require(values.shape == (len(self.feature_columns),), "Rank7 score row shape mismatch")
        _require(np.isfinite(values).all(), "Rank7 score row contains non-finite values")
        predictions = []
        for model in self.models:
            model.n_jobs = 1
            prediction = np.asarray(model.predict(values.reshape(1, -1)), dtype=float)
            _require(prediction.shape == (1, 2), "Rank7 prediction shape mismatch")
            predictions.append(prediction[0])
        mean = np.mean(np.stack(predictions), axis=0)
        return float(mean[0]), float(mean[1])


def _barrier_contract(spec: dict[str, Any]) -> dict[str, Any]:
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
    barrier = _barrier_contract(exit_spec) if exit_spec else None
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
