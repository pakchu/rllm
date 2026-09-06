"""Train-only micro source-support diagnostic for PPOSM SKIP and TP12.

This successor to the global-ratio SKIP diagnostic asks only whether the
hash-bound Binance global trader ``count_long_short_ratio`` source contains
pre-2024 causal signal for both lifecycle residual candidate actions.  It is
bounded to strict pre-2024 lifecycle labels and refuses to run unless a matching
zero-label preregistration is present.

Why this micro variant exists: the earlier diagnostic's HistGradientBoosting
configuration inherited the scikit-learn default ``min_samples_leaf=20``-scale
regime through a larger tree, while the first expanding fold can contain only
14 train rows.  This file pre-registers a bounded micro HGB
(``min_samples_leaf=2``) plus a conservative polynomial logistic regression;
2024+ OOS labels remain outside this diagnostic.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import build_pposm_lifecycle_residual_data as lifecycle

DEFAULT_METRICS = Path(
    "/home/pakchu/rllm/data/binance_um_metrics_BTCUSDT_5m_2020-09-01_2026-06-01.csv.gz"
)
DEFAULT_OUTPUT = Path(
    "results/pposm_global_ratio_micro_two_action_source_support_2026-09-05.json"
)
DEFAULT_PREREGISTRATION = Path(
    "results/pposm_global_ratio_micro_two_action_preregistration_2026-09-05.json"
)
EXPECTED_METRICS_SHA256 = "d391022352d5b14dea7ffd207a9d1f84f603d06ddae42da55dd792f722fc0106"
EXPECTED_PRE2024_PREFIX_SHA256 = (
    "1da4e6bb2929050eae69b58c7fa715900258858b76e5442c9897b1601934db99"
)
RATIO_COLUMN = "count_long_short_ratio"
EXPECTED_ANCHORS = 102
ACTIONS = ("SKIP", "TP12")
BASE_FEATURE_COLUMNS = (
    "ratio_log_level",
    "ratio_dlog1",
    "ratio_z288",
)
FEATURE_COLUMNS = (
    "ratio_log_level",
    "ratio_dlog1",
    "ratio_z288",
    "ratio_abs_z288",
    "ratio_z288_sq",
    "ratio_abs_dlog1",
    "ratio_dlog1_sq",
    "ratio_available",
    "ratio_age_bars",
)
FOLD_SPECS: tuple[tuple[str, tuple[int, ...], int], ...] = (
    ("train_2020_test_2021", (2020,), 2021),
    ("train_through_2021_test_2022", (2020, 2021), 2022),
    ("train_through_2022_test_2023", (2020, 2021, 2022), 2023),
)
GATE_THRESHOLDS = {
    "max_feature_null_fraction": 0.05,
    "pooled_auc": 0.60,
    "auc_lower95": 0.50,
    "balanced_accuracy": 0.55,
    "per_year_auc": 0.50,
}
MODEL_SPECS = {
    "logreg_l2_c025_balanced_poly": {
        "family": "LogisticRegression",
        "penalty": "l2",
        "C": 0.25,
        "class_weight": "balanced",
        "features": list(FEATURE_COLUMNS),
    },
    "hgb_micro_leaf2_l2_1": {
        "family": "HistGradientBoostingClassifier",
        "max_leaf_nodes": 3,
        "max_depth": 2,
        "min_samples_leaf": 2,
        "l2_regularization": 1.0,
        "features": list(FEATURE_COLUMNS),
    },
}
PROTOCOL = "pposm_global_count_long_short_ratio_micro_two_action_source_support_v1"


def runtime_versions() -> dict[str, str]:
    return {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }


@dataclass(frozen=True)
class Config:
    manifest: Path = lifecycle.counterfactual.DEFAULT_MANIFEST
    metrics_csv: Path = DEFAULT_METRICS
    output: Path = DEFAULT_OUTPUT
    preregistration: Path = DEFAULT_PREREGISTRATION
    bootstrap_iterations: int = 2000
    random_seed: int = 20260905
    write_output: bool = True
    expected_metrics_sha256: str = EXPECTED_METRICS_SHA256
    expected_pre2024_prefix_sha256: str | None = EXPECTED_PRE2024_PREFIX_SHA256


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_canonical(value: Any) -> str:
    return sha256_text(canonical_json(value))


def preregistration_expected_bindings(cfg: Config) -> dict[str, Any]:
    return {
        "source_sha256": cfg.expected_metrics_sha256,
        "pre2024_prefix_sha256": cfg.expected_pre2024_prefix_sha256,
        "actions": list(ACTIONS),
        "anchors_per_action": EXPECTED_ANCHORS,
        "features": list(FEATURE_COLUMNS),
        "folds": [
            {"name": name, "train_years": list(train_years), "test_year": test_year}
            for name, train_years, test_year in FOLD_SPECS
        ],
        "models": copy.deepcopy(MODEL_SPECS),
        "gates": dict(GATE_THRESHOLDS),
        "bootstrap_iterations": cfg.bootstrap_iterations,
        "random_seed": cfg.random_seed,
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
        "manifest_file_sha256": sha256_file(cfg.manifest),
        "runtime_versions": runtime_versions(),
    }


def validate_preregistration(cfg: Config) -> dict[str, Any]:
    if cfg.preregistration is None:
        raise RuntimeError("matching micro two-action preregistration is required")
    if not cfg.preregistration.is_file():
        raise RuntimeError(f"missing preregistration: {cfg.preregistration}")
    report = json.loads(cfg.preregistration.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise RuntimeError("micro two-action preregistration must be a JSON object")
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    if report.get("manifest_hash") != sha256_canonical(core):
        raise RuntimeError("micro two-action preregistration manifest drift")
    if report.get("protocol") != "pposm_global_count_long_short_ratio_micro_two_action_preregistration_v1":
        raise RuntimeError("micro two-action preregistration protocol mismatch")
    expected = preregistration_expected_bindings(cfg)
    observed = {
        "source_sha256": report.get("source", {}).get("sha256"),
        "pre2024_prefix_sha256": report.get("source", {}).get("pre2024_prefix_sha256"),
        "actions": report.get("design", {}).get("actions"),
        "anchors_per_action": report.get("design", {}).get("anchors_per_action"),
        "features": report.get("design", {}).get("features"),
        "folds": report.get("design", {}).get("folds"),
        "models": report.get("design", {}).get("models"),
        "gates": report.get("design", {}).get("gates"),
        "bootstrap_iterations": report.get("design", {}).get("bootstrap_iterations"),
        "random_seed": report.get("design", {}).get("random_seed"),
        "evaluator_sha256": report.get("implementation", {}).get("evaluator_sha256"),
        "manifest_file_sha256": report.get("source", {}).get("pposm_manifest_sha256"),
        "runtime_versions": report.get("implementation", {}).get("runtime_versions"),
    }
    if observed != expected:
        raise RuntimeError("micro two-action preregistration binding drift")
    boundary = report.get("evidence_boundary", {})
    if (
        boundary.get("lifecycle_labels_opened") != 0
        or boundary.get("oos_labels_opened") != 0
        or boundary.get("source_support_metrics_computed") is not False
        or boundary.get("sft_or_rlvr_started") is not False
    ):
        raise RuntimeError("micro two-action preregistration opened outcomes")
    return report


def load_lifecycle_two_action_labels(manifest_path: Path) -> pd.DataFrame:
    """Return one pre-2024 SWITCH label for each SKIP and TP12 anchor."""

    manifest, strategy_cfg = lifecycle.frozen.load_frozen_manifest(manifest_path)
    market, _, state, active, engine = lifecycle.load_train_context(manifest, strategy_cfg)
    rows = lifecycle.rows_from_train_context(
        market, state, active, manifest, strategy_cfg, engine
    )
    if any(row.get("split") != "train" for row in rows):
        raise RuntimeError("micro diagnostic received non-train lifecycle rows")
    selected = [
        row for row in rows if row.get("metadata", {}).get("candidate_action") in ACTIONS
    ]
    by_candidate = Counter(row["metadata"]["candidate_action"] for row in selected)
    expected_counts = {action: EXPECTED_ANCHORS for action in ACTIONS}
    if dict(sorted(by_candidate.items())) != expected_counts:
        raise RuntimeError(
            f"expected {EXPECTED_ANCHORS} anchors for each action, got {dict(sorted(by_candidate.items()))}"
        )
    records = []
    for row in selected:
        meta = row["metadata"]
        records.append(
            {
                "identity": str(meta["identity"]),
                "base_identity": str(meta["base_identity"]),
                "action": str(meta["candidate_action"]),
                "signal_position": int(meta["signal_position"]),
                "signal_time": pd.Timestamp(meta["signal_time"]).tz_localize(None),
                "target_switch": 1 if row["target"] == "SWITCH" else 0,
            }
        )
    out = pd.DataFrame.from_records(records).sort_values(
        ["signal_time", "action"], kind="mergesort"
    ).reset_index(drop=True)
    if out["signal_time"].max() >= pd.Timestamp("2024-01-01"):
        raise RuntimeError("diagnostic labels must be strictly pre-2024")
    duplicated = out.duplicated(["action", "signal_time"])
    if duplicated.any():
        raise RuntimeError("duplicate lifecycle anchor action/signal_time")
    paired = out.groupby("signal_time")["action"].apply(lambda s: tuple(sorted(s)))
    if not all(value == tuple(ACTIONS) for value in paired):
        raise RuntimeError("each lifecycle anchor must have both SKIP and TP12 rows")
    out.attrs["candidate_counts"] = dict(sorted(by_candidate.items()))
    out.attrs["manifest_freeze_hash"] = manifest.get("freeze_hash")
    out.attrs["manifest_file_sha256"] = sha256_file(manifest_path) if manifest_path.exists() else None
    out.attrs["lifecycle_builder_sha256"] = sha256_file(Path(lifecycle.__file__).resolve())
    out.attrs["anchor_label_identity_sha256"] = sha256_canonical(
        [
            {
                "identity": row.identity,
                "base_identity": row.base_identity,
                "action": row.action,
                "signal_position": int(row.signal_position),
                "signal_time": str(row.signal_time),
                "target_switch": int(row.target_switch),
            }
            for row in out.itertuples(index=False)
        ]
    )
    return out


def load_ratio_source(
    metrics_csv: Path,
    *,
    expected_sha256: str = EXPECTED_METRICS_SHA256,
    expected_pre2024_prefix_sha256: str | None = EXPECTED_PRE2024_PREFIX_SHA256,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source_hash = sha256_file(metrics_csv)
    if source_hash != expected_sha256:
        raise RuntimeError(
            f"metrics source sha256 mismatch: expected {expected_sha256}, got {source_hash}"
        )
    usecols = ["create_time", RATIO_COLUMN]
    metrics = pd.read_csv(metrics_csv, usecols=usecols)
    if RATIO_COLUMN not in metrics:
        raise RuntimeError(f"missing {RATIO_COLUMN} in {metrics_csv}")
    metrics["create_time"] = pd.to_datetime(metrics["create_time"]).dt.tz_localize(None)
    metrics = metrics.sort_values("create_time", kind="mergesort").drop_duplicates(
        "create_time", keep="last"
    )
    metrics = metrics.loc[metrics["create_time"] < pd.Timestamp("2024-01-01")].reset_index(drop=True)
    ratio = pd.to_numeric(metrics[RATIO_COLUMN], errors="coerce")
    if (ratio <= 0).any(skipna=True):
        raise RuntimeError("count_long_short_ratio contains non-positive values")
    metrics["ratio_log_level"] = np.log(ratio)
    metrics["ratio_dlog1"] = metrics["ratio_log_level"].diff()
    rolling = metrics["ratio_log_level"].rolling(288, min_periods=288)
    mean = rolling.mean()
    std = rolling.std(ddof=0)
    metrics["ratio_z288"] = (metrics["ratio_log_level"] - mean) / std.replace(0.0, np.nan)
    prefix_payload = [
        {
            "create_time": str(row.create_time),
            RATIO_COLUMN: None
            if pd.isna(getattr(row, RATIO_COLUMN))
            else float(getattr(row, RATIO_COLUMN)),
        }
        for row in metrics[["create_time", RATIO_COLUMN]].itertuples(index=False)
    ]
    pre2024_prefix_sha256 = sha256_canonical(prefix_payload)
    if (
        expected_pre2024_prefix_sha256 is not None
        and pre2024_prefix_sha256 != expected_pre2024_prefix_sha256
    ):
        raise RuntimeError(
            "metrics pre-2024 prefix sha256 mismatch: "
            f"expected {expected_pre2024_prefix_sha256}, got {pre2024_prefix_sha256}"
        )
    summary = {
        "path": str(metrics_csv),
        "sha256": source_hash,
        "expected_sha256": expected_sha256,
        "sha256_matches_expected": True,
        "pre2024_prefix_sha256": pre2024_prefix_sha256,
        "expected_pre2024_prefix_sha256": expected_pre2024_prefix_sha256,
        "pre2024_prefix_sha256_matches_expected": True,
        "rows": int(len(metrics)),
        "first_create_time": str(metrics["create_time"].min()),
        "last_create_time": str(metrics["create_time"].max()),
        "feature_frame_end_exclusive": "2024-01-01",
        "ratio_column": RATIO_COLUMN,
        "feature_causality": "features are computed at source bar t and joined only when t == signal_time - 5 minutes",
    }
    return metrics[["create_time", *BASE_FEATURE_COLUMNS]], summary


def join_prior_completed_5m(labels: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    """Join only the immediately prior completed 5m source bar; no ffill."""

    out = labels.copy()
    out["source_time_required"] = out["signal_time"] - pd.Timedelta(minutes=5)
    merged = out.merge(
        source,
        left_on="source_time_required",
        right_on="create_time",
        how="left",
        validate="many_to_one",
    )
    merged["ratio_available"] = merged["create_time"].notna().astype(float)
    age_minutes = (merged["signal_time"] - merged["create_time"]).dt.total_seconds() / 60.0
    merged["ratio_age_bars"] = age_minutes / 5.0
    missing = merged["create_time"].isna()
    merged.loc[missing, "ratio_age_bars"] = np.nan
    if (merged.loc[~missing, "ratio_age_bars"] != 1.0).any():
        raise RuntimeError("source join used a bar other than strict signal_time - 5m")
    merged["ratio_abs_z288"] = merged["ratio_z288"].abs()
    merged["ratio_z288_sq"] = merged["ratio_z288"] ** 2
    merged["ratio_abs_dlog1"] = merged["ratio_dlog1"].abs()
    merged["ratio_dlog1_sq"] = merged["ratio_dlog1"] ** 2
    return merged


def feature_null_fractions(frame: pd.DataFrame) -> dict[str, float]:
    return {name: float(frame[name].isna().mean()) for name in FEATURE_COLUMNS}


def valid_model_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.dropna(subset=list(FEATURE_COLUMNS) + ["target_switch"]).copy()


def _classifiers(seed: int) -> dict[str, Any]:
    return {
        "logreg_l2_c025_balanced_poly": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                penalty="l2",
                C=0.25,
                solver="liblinear",
                class_weight="balanced",
                random_state=seed,
                max_iter=1000,
            ),
        ),
        "hgb_micro_leaf2_l2_1": HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.05,
            max_leaf_nodes=3,
            max_depth=2,
            min_samples_leaf=2,
            l2_regularization=1.0,
            random_state=seed,
        ),
    }


def _sample_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y.astype(int), minlength=2).astype(float)
    weights = np.ones_like(y, dtype=float)
    for cls in (0, 1):
        if counts[cls] > 0:
            weights[y == cls] = len(y) / (2.0 * counts[cls])
    return weights


def expanding_oof(frame: pd.DataFrame, *, seed: int) -> dict[str, dict[str, Any]]:
    X_all = frame.loc[:, FEATURE_COLUMNS].to_numpy(float)
    y_all = frame["target_switch"].to_numpy(int)
    years = frame["signal_time"].dt.year.to_numpy(int)
    results: dict[str, dict[str, Any]] = {}
    for model_name, estimator in _classifiers(seed).items():
        oof = np.full(len(frame), np.nan, dtype=float)
        fold_summaries: list[dict[str, Any]] = []
        for fold_name, train_years, test_year in FOLD_SPECS:
            train_mask = np.isin(years, train_years)
            test_mask = years == test_year
            y_train = y_all[train_mask]
            y_test = y_all[test_mask]
            fold = {
                "name": fold_name,
                "train_years": list(train_years),
                "test_year": int(test_year),
                "train_rows": int(train_mask.sum()),
                "test_rows": int(test_mask.sum()),
                "train_class_counts": dict(sorted(Counter(y_train.tolist()).items())),
                "test_class_counts": dict(sorted(Counter(y_test.tolist()).items())),
                "valid": False,
            }
            if train_mask.sum() == 0 or test_mask.sum() == 0 or len(set(y_train)) < 2 or len(set(y_test)) < 2:
                fold_summaries.append(fold)
                continue
            fit_kwargs = {"sample_weight": _sample_weights(y_train)} if model_name.startswith("hgb") else {}
            estimator.fit(X_all[train_mask], y_train, **fit_kwargs)
            proba = estimator.predict_proba(X_all[test_mask])[:, 1]
            oof[test_mask] = proba
            pred = (proba >= 0.5).astype(int)
            fold.update(
                {
                    "valid": True,
                    "auc": float(roc_auc_score(y_test, proba)),
                    "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
                    "accuracy": float(accuracy_score(y_test, pred)),
                }
            )
            fold_summaries.append(fold)
        mask = np.isfinite(oof)
        if mask.sum() and len(set(y_all[mask])) >= 2:
            pred = (oof[mask] >= 0.5).astype(int)
            pooled = {
                "valid_rows": int(mask.sum()),
                "class_counts": dict(sorted(Counter(y_all[mask].tolist()).items())),
                "auc": float(roc_auc_score(y_all[mask], oof[mask])),
                "balanced_accuracy": float(balanced_accuracy_score(y_all[mask], pred)),
                "accuracy": float(accuracy_score(y_all[mask], pred)),
            }
        else:
            pooled = {"valid_rows": int(mask.sum()), "class_counts": {}, "auc": None, "balanced_accuracy": None, "accuracy": None}
        results[model_name] = {"folds": fold_summaries, "pooled": pooled, "oof": oof.tolist()}
    return results


def bootstrap_auc_ci(y: np.ndarray, score: np.ndarray, *, iterations: int, seed: int) -> dict[str, float | int | None]:
    mask = np.isfinite(score)
    y = y[mask].astype(int)
    score = score[mask].astype(float)
    if len(y) == 0 or len(set(y)) < 2:
        return {"iterations": 0, "lower95": None, "upper95": None}
    rng = np.random.default_rng(seed)
    aucs: list[float] = []
    n = len(y)
    for _ in range(int(iterations)):
        idx = rng.integers(0, n, size=n)
        if len(set(y[idx])) < 2:
            continue
        aucs.append(float(roc_auc_score(y[idx], score[idx])))
    if not aucs:
        return {"iterations": 0, "lower95": None, "upper95": None}
    return {
        "iterations": int(len(aucs)),
        "lower95": float(np.quantile(aucs, 0.025)),
        "upper95": float(np.quantile(aucs, 0.975)),
    }


def gate_for_model(model_result: dict[str, Any], ci: dict[str, Any], nulls: dict[str, float]) -> dict[str, Any]:
    pooled = model_result["pooled"]
    valid_folds = [fold for fold in model_result["folds"] if fold.get("valid")]
    checks = {
        "exactly_3_valid_folds": len(valid_folds) == len(FOLD_SPECS),
        "max_feature_null_le_5pct": max(nulls.values()) <= GATE_THRESHOLDS["max_feature_null_fraction"],
        "pooled_auc_ge_0_60": pooled.get("auc") is not None and pooled["auc"] >= GATE_THRESHOLDS["pooled_auc"],
        "lower95_gt_0_50": ci.get("lower95") is not None and ci["lower95"] > GATE_THRESHOLDS["auc_lower95"],
        "balanced_accuracy_ge_0_55": pooled.get("balanced_accuracy") is not None and pooled["balanced_accuracy"] >= GATE_THRESHOLDS["balanced_accuracy"],
        "every_valid_year_auc_gt_0_50": bool(valid_folds) and all(float(fold["auc"]) > GATE_THRESHOLDS["per_year_auc"] for fold in valid_folds),
    }
    return {"pass": bool(all(checks.values())), "checks": checks, "valid_years": [int(f["test_year"]) for f in valid_folds]}


def strip_oof(result: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {name: {key: value for key, value in payload.items() if key != "oof"} for name, payload in result.items()}


def joined_feature_hash(frame: pd.DataFrame) -> str:
    payload = []
    for row in frame.sort_values(["signal_time", "action"], kind="mergesort").itertuples(index=False):
        payload.append(
            {
                "identity": str(row.identity),
                "base_identity": str(row.base_identity),
                "action": str(row.action),
                "signal_time": str(row.signal_time),
                "source_time_required": str(row.source_time_required),
                "source_time": None if pd.isna(row.create_time) else str(row.create_time),
                "target_switch": int(row.target_switch),
                "features": {
                    name: None if pd.isna(getattr(row, name)) else float(getattr(row, name))
                    for name in FEATURE_COLUMNS
                },
            }
        )
    return sha256_canonical(payload)


def oof_score_hash(model_results_by_action: dict[str, dict[str, dict[str, Any]]], frames_by_action: dict[str, pd.DataFrame]) -> str:
    payload = []
    for action in sorted(model_results_by_action):
        frame = frames_by_action[action]
        for model_name in sorted(model_results_by_action[action]):
            scores = np.asarray(model_results_by_action[action][model_name]["oof"], dtype=float)
            for identity, score in zip(frame["identity"].tolist(), scores, strict=True):
                payload.append(
                    {
                        "action": action,
                        "model": model_name,
                        "identity": str(identity),
                        "oof_score": None if not np.isfinite(score) else float(score),
                    }
                )
    return sha256_canonical(payload)


def select_model(model_results: dict[str, dict[str, Any]], ci_by_model: dict[str, Any], gates: dict[str, Any]) -> str:
    def key(name: str) -> tuple[int, float, float, float]:
        pooled = model_results[name]["pooled"]
        auc = -1.0 if pooled.get("auc") is None else float(pooled["auc"])
        lower = -1.0 if ci_by_model[name].get("lower95") is None else float(ci_by_model[name]["lower95"])
        bacc = -1.0 if pooled.get("balanced_accuracy") is None else float(pooled["balanced_accuracy"])
        return (1 if gates[name]["pass"] else 0, auc, lower, bacc)

    return max(model_results, key=key)


def evaluate(cfg: Config) -> dict[str, Any]:
    preregistration = validate_preregistration(cfg)
    labels = load_lifecycle_two_action_labels(cfg.manifest)
    source, source_summary = load_ratio_source(
        cfg.metrics_csv,
        expected_sha256=cfg.expected_metrics_sha256,
        expected_pre2024_prefix_sha256=cfg.expected_pre2024_prefix_sha256,
    )
    joined = join_prior_completed_5m(labels, source)
    if joined["signal_time"].max() >= pd.Timestamp("2024-01-01"):
        raise RuntimeError("joined frame contains OOS labels")

    nulls_by_action: dict[str, dict[str, float]] = {}
    frames_by_action: dict[str, pd.DataFrame] = {}
    model_results_by_action: dict[str, dict[str, dict[str, Any]]] = {}
    ci_by_action: dict[str, dict[str, Any]] = {}
    gates_by_action: dict[str, dict[str, Any]] = {}
    selected_by_action: dict[str, str] = {}
    action_pass: dict[str, bool] = {}

    for action_index, action in enumerate(ACTIONS):
        action_joined = joined.loc[joined["action"] == action].reset_index(drop=True)
        nulls = feature_null_fractions(action_joined)
        frame = valid_model_frame(action_joined)
        if frame["signal_time"].max() >= pd.Timestamp("2024-01-01"):
            raise RuntimeError(f"{action} model frame contains OOS labels")
        results = expanding_oof(frame, seed=cfg.random_seed + action_index * 100)
        y = frame["target_switch"].to_numpy(int)
        ci_for_action: dict[str, Any] = {}
        gates_for_action: dict[str, Any] = {}
        for model_index, (name, payload) in enumerate(results.items()):
            ci = bootstrap_auc_ci(
                y,
                np.asarray(payload["oof"], dtype=float),
                iterations=cfg.bootstrap_iterations,
                seed=cfg.random_seed + action_index * 100 + model_index,
            )
            ci_for_action[name] = ci
            gates_for_action[name] = gate_for_model(payload, ci, nulls)
        selected = select_model(results, ci_for_action, gates_for_action)
        nulls_by_action[action] = nulls
        frames_by_action[action] = frame
        model_results_by_action[action] = results
        ci_by_action[action] = ci_for_action
        gates_by_action[action] = gates_for_action
        selected_by_action[action] = selected
        action_pass[action] = bool(gates_for_action[selected]["pass"])

    joined_sha256 = joined_feature_hash(joined)
    scores_sha256 = oof_score_hash(model_results_by_action, frames_by_action)
    summary = {
        "protocol": PROTOCOL,
        "config": {key: str(value) for key, value in asdict(cfg).items()},
        "preregistration": {
            "path": str(cfg.preregistration),
            "sha256": sha256_file(cfg.preregistration),
            "manifest_hash": preregistration["manifest_hash"],
        },
        "source": source_summary,
        "artifact_bindings": {
            "diagnostic_implementation_sha256": sha256_file(Path(__file__).resolve()),
            "preregistration_sha256": sha256_file(cfg.preregistration),
            "manifest_file_sha256": labels.attrs.get("manifest_file_sha256"),
            "manifest_freeze_hash": labels.attrs.get("manifest_freeze_hash"),
            "lifecycle_builder_sha256": labels.attrs.get("lifecycle_builder_sha256"),
            "anchor_label_identity_sha256": labels.attrs.get("anchor_label_identity_sha256"),
            "joined_feature_sha256": joined_sha256,
            "oof_score_sha256": scores_sha256,
        },
        "runtime_versions": runtime_versions(),
        "scope_limits": {
            "train_only": True,
            "no_oos_labels": True,
            "binary_per_candidate_action_only": True,
            "full_professor_critic_rubric_satisfied": False,
            "reason": "This source-support diagnostic tests only pre-2024 global-ratio separability for SKIP and TP12 lifecycle residual labels before SFT/RLVR or OOS economics.",
        },
        "boundary": {
            "labels": "pre-2024 lifecycle residual KEEP-vs-SWITCH labels for SKIP and TP12 only",
            "oos_labels_built_or_read": False,
            "join": "exact source create_time == signal_time - 5 minutes; no ffill/interpolation",
            "folds": [
                {"name": name, "train_years": list(train_years), "test_year": test_year}
                for name, train_years, test_year in FOLD_SPECS
            ],
        },
        "anchors": {
            "expected_per_action": EXPECTED_ANCHORS,
            "actual_total": int(len(labels)),
            "candidate_counts": labels.attrs.get("candidate_counts", {}),
            "target_counts_by_action": {
                action: dict(sorted(Counter(labels.loc[labels["action"] == action, "target_switch"].tolist()).items()))
                for action in ACTIONS
            },
            "first_signal_time": str(labels["signal_time"].min()),
            "last_signal_time": str(labels["signal_time"].max()),
        },
        "features": {
            "columns": list(FEATURE_COLUMNS),
            "null_fraction_by_action": nulls_by_action,
            "max_null_fraction_by_action": {
                action: float(max(nulls.values())) for action, nulls in nulls_by_action.items()
            },
            "model_rows_after_dropna_by_action": {
                action: int(len(frame)) for action, frame in frames_by_action.items()
            },
        },
        "models": {
            action: strip_oof(results) for action, results in model_results_by_action.items()
        },
        "bootstrap_auc_ci": ci_by_action,
        "gates": gates_by_action,
        "selected_model_train_only_by_action": selected_by_action,
        "selection_rule": "for each action choose a passing model first; within pass/fail bucket break ties by pooled AUC, lower95 CI, then balanced accuracy",
        "selected_gate_by_action": {
            action: gates_by_action[action][selected_by_action[action]] for action in ACTIONS
        },
        "action_pass": action_pass,
        "overall_pass": bool(all(action_pass.values())),
        "overall_pass_rule": "overall pass iff both SKIP and TP12 selected action gates pass",
    }
    payload = canonical_json(summary).encode("utf-8")
    summary["summary_sha256_without_self"] = hashlib.sha256(payload).hexdigest()
    if cfg.write_output:
        cfg.output.parent.mkdir(parents=True, exist_ok=True)
        cfg.output.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return summary


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Config.manifest)
    parser.add_argument("--metrics-csv", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=20260905)
    parser.add_argument("--no-write", action="store_true", help="evaluate without writing a result artifact")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    summary = evaluate(
        Config(
            manifest=args.manifest,
            metrics_csv=args.metrics_csv,
            output=args.output,
            preregistration=args.preregistration,
            bootstrap_iterations=args.bootstrap_iterations,
            random_seed=args.random_seed,
            write_output=not args.no_write,
        )
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
