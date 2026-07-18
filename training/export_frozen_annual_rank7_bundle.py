#!/usr/bin/env python3
"""Export the frozen 2026 annual Rank7 ensemble as a bounded live artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesRegressor

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from execution.rank7_runtime import (
    EXPECTED_MODEL_PARAMS,
    FEATURE_COLUMNS,
    Rank7Bundle,
    build_rank7_feature_context,
    rank7_manifest_hash,
    rebuild_rank7_feature_context,
    save_frozen_extra_trees,
)
from training.audit_expanding_extratrees_rank7_stability import (
    FOLDS,
    SEEDS,
    SPEC,
    action,
    build_base,
    evaluate,
)
from training.evaluate_stable_ensemble_conditional_pullback_oos import (
    Config,
    _load_full_braid,
)
from training.search_liveparity_state_feature_interactions import completed_hourly_features
from training.search_stable_ensemble_conditional_pullback_alpha import source_thresholds


DEFAULT_OUTPUT = Path("artifacts/rank7/frozen_annual_rank7_2026")
RESEARCH_RESULT = Path("results/expanding_extratrees_rank7_stability_2026-07-15.json")
RESEARCH_MANIFEST = Path("results/expanding_extratrees_top10_pre2025_manifest_2026-07-15.json")
ANNUAL_CUTOFF = pd.Timestamp("2026-01-01")
VALID_UNTIL = pd.Timestamp("2027-01-01")
TREES_PER_SEED = 300


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _json_hash(value: Any) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(blob).hexdigest()


def _source_year_weights(years: np.ndarray, source: np.ndarray) -> np.ndarray:
    groups = list(zip(years.tolist(), source.tolist(), strict=True))
    counts = {group: groups.count(group) for group in set(groups)}
    weights = np.asarray([1.0 / counts[group] for group in groups], dtype=float)
    return weights * (len(weights) / weights.sum())


def _fit_export_models(base: dict[str, Any]) -> tuple[list[ExtraTreesRegressor], dict[str, Any]]:
    context = base["context"]
    signals = np.asarray(base["signals"], dtype=int)
    funding = np.asarray(base["funding"], dtype=bool)
    target = np.asarray(base["y"], dtype=float)
    signal_dates = pd.to_datetime(base["signal_dates"])
    end_dates = np.asarray(base["end_dates"])
    fit = np.asarray(
        (signal_dates >= pd.Timestamp("2020-07-01"))
        & (signal_dates < ANNUAL_CUTOFF)
        & np.isfinite(target).all(axis=1)
        & (end_dates < ANNUAL_CUTOFF.to_datetime64()),
        dtype=bool,
    )
    pred = np.asarray(
        (signal_dates >= ANNUAL_CUTOFF) & (signal_dates < pd.Timestamp("2026-06-02")),
        dtype=bool,
    )
    matrix = np.asarray(context["matrix"], dtype=float)
    train_x = matrix[signals[fit]]
    train_y = target[fit]
    years = pd.to_datetime(context["dates"].iloc[signals[fit]]).dt.year.to_numpy()
    weights = _source_year_weights(years, funding[fit])

    models: list[ExtraTreesRegressor] = []
    train_predictions: list[np.ndarray] = []
    live_predictions: list[np.ndarray] = []
    for seed in SEEDS:
        model = ExtraTreesRegressor(
            n_estimators=TREES_PER_SEED,
            max_depth=int(SPEC["max_depth"]),
            min_samples_leaf=int(SPEC["min_samples_leaf"]),
            max_features=float(SPEC["max_features"]),
            bootstrap=False,
            random_state=int(seed),
            n_jobs=-1,
        ).fit(train_x, train_y, sample_weight=weights)
        model.n_jobs = 1
        train_predictions.append(np.asarray(model.predict(train_x), dtype=float))
        live_predictions.append(np.asarray(model.predict(matrix[signals[pred]]), dtype=float))
        models.append(model)

    train_prediction = np.mean(np.stack(train_predictions), axis=0)
    prediction = np.mean(np.stack(live_predictions), axis=0)
    train_score = train_prediction[:, 0] - float(SPEC["lambda"]) * train_prediction[:, 1]
    score = prediction[:, 0] - float(SPEC["lambda"]) * prediction[:, 1]
    fit_source = funding[fit]
    pred_source = funding[pred]
    funding_score, premium_score = source_thresholds(
        train_score,
        fit_source,
        funding_q=float(SPEC["funding_q"]),
        premium_q=float(SPEC["premium_q"]),
    )
    risk = train_prediction[:, 1]
    funding_risk_cap = float(np.quantile(risk[fit_source], float(SPEC["risk_q"])))
    premium_risk_cap = float(np.quantile(risk[~fit_source], float(SPEC["risk_q"])))
    width = np.asarray(base["width"], dtype=float)
    pullback = np.asarray(base["pullback"], dtype=float)
    width_q20 = float(np.quantile(width[signals[fit]][fit_source], 0.2))
    pullback_q40 = float(np.quantile(pullback[signals[fit]][fit_source], 0.4))
    positions = signals[pred]
    interaction = (width[positions] > width_q20) | (pullback[positions] <= pullback_q40)
    selected = (
        pred_source
        & (score >= funding_score)
        & (prediction[:, 1] <= funding_risk_cap)
        & interaction
    ) | (
        (~pred_source)
        & (score >= premium_score)
        & (prediction[:, 1] <= premium_risk_cap)
    )
    thresholds = {
        "funding_score": float(funding_score),
        "premium_score": float(premium_score),
        "funding_risk_cap": funding_risk_cap,
        "premium_risk_cap": premium_risk_cap,
        "width_q20": width_q20,
        "pullback_q40": pullback_q40,
    }
    return models, {
        "fit_examples": int(fit.sum()),
        "predict_events": int(pred.sum()),
        "selected_events": int(selected.sum()),
        "selected_positions": positions[selected].astype(int).tolist(),
        "selected_positions_hash": _json_hash(positions[selected].astype(int).tolist()),
        "thresholds": thresholds,
        "train_prediction_hash": _array_hash(train_prediction),
        "prediction_hash": _array_hash(prediction),
    }


def _market_with_braid_inputs(context: dict[str, Any], cfg: Config) -> pd.DataFrame:
    braid_market, braid_dates = _load_full_braid(cfg, cfg.exclude_from)
    market = context["market"].copy()
    dates = pd.to_datetime(market["date"])
    if not np.array_equal(dates.to_numpy(), braid_dates.to_numpy()):
        raise RuntimeError("Rank7 export braid grid differs from research market")
    columns = [
        "date",
        "open_interest",
        "open_interest_available",
        "spot_close",
        "spot_rows",
        "premium_index_1m_close",
        "premium_rows",
    ]
    return market.merge(
        braid_market[columns],
        on="date",
        how="left",
        validate="one_to_one",
    )


def _assert_feature_parity(
    base: dict[str, Any],
    live_market: pd.DataFrame,
    medians: np.ndarray,
) -> dict[str, Any]:
    expected = base["context"]
    rebuilt = rebuild_rank7_feature_context(
        live_market,
        medians=medians,
        clip=(-20.0, 20.0),
        delay_bars=12,
    )
    matrix_equal = np.array_equal(rebuilt["matrix"], np.asarray(expected["matrix"], dtype=float))
    base_equal = np.array_equal(rebuilt["base"], np.asarray(expected["base"], dtype=bool))
    source_equal = bool(
        np.array_equal(rebuilt["funding_leg"], np.asarray(expected["funding_leg"], dtype=bool))
        and np.array_equal(rebuilt["premium_leg"], np.asarray(expected["premium_leg"], dtype=bool))
    )
    if not (matrix_equal and base_equal and source_equal):
        max_abs = float(np.max(np.abs(rebuilt["matrix"] - np.asarray(expected["matrix"], dtype=float))))
        raise RuntimeError(
            "Rank7 live feature graph differs from research "
            f"(matrix={matrix_equal}, base={base_equal}, source={source_equal}, max_abs={max_abs})"
        )
    return {
        "matrix_equal": True,
        "base_equal": True,
        "source_equal": True,
        "matrix_sha256": _array_hash(rebuilt["matrix"]),
        "base_sha256": _array_hash(rebuilt["base"].astype(np.uint8)),
        "anchor_sha256": _array_hash(rebuilt["anchors"].astype(np.uint8)),
    }


def _write_hourly_history(root: Path, market: pd.DataFrame) -> dict[str, Any]:
    hourly, _ = completed_hourly_features(market)
    path = root / "state" / "completed_hourly_history.csv.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = hourly.reset_index().rename(columns={hourly.index.name or "index": "date"})
    frame = frame[["date", "open", "high", "low", "close", "quote", "buy"]]
    frame.to_csv(
        path,
        index=False,
        float_format="%.17g",
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )
    return {
        "path": str(path.relative_to(root)),
        "sha256": _sha256(path),
        "rows": int(len(frame)),
        "start": pd.Timestamp(frame.iloc[0]["date"]).isoformat(),
        "end": pd.Timestamp(frame.iloc[-1]["date"]).isoformat(),
    }


def export_bundle(output: Path) -> dict[str, Any]:
    if not RESEARCH_RESULT.is_file() or not RESEARCH_MANIFEST.is_file():
        raise FileNotFoundError("frozen Rank7 research artifacts are missing")
    research = json.loads(RESEARCH_RESULT.read_text(encoding="utf-8"))
    expected = research["ensembles"][str(TREES_PER_SEED)]
    cfg = Config(output="/tmp/rank7_bundle_no_write.json", docs_output="")
    base = build_base()
    context = base["context"]
    medians = np.asarray(context["feature_medians"], dtype=float)
    if medians.shape != (len(FEATURE_COLUMNS),) or not np.isfinite(medians).all():
        raise RuntimeError("research context did not expose valid frozen feature medians")
    live_market = _market_with_braid_inputs(context, cfg)
    feature_parity = _assert_feature_parity(base, live_market, medians)

    models, exported = _fit_export_models(base)
    fresh = evaluate(base, trees=TREES_PER_SEED, seeds=SEEDS, label=f"ensemble5_{TREES_PER_SEED}")
    schedule_parity = bool(
        fresh["full_result_hash"] == expected["full_result_hash"]
        and fresh["selected_positions_hash"] == expected["selected_positions_hash"]
        and fresh["stats"] == expected["stats"]
    )
    fresh_2026 = next(row for row in fresh["folds"] if row["name"] == "2026h1")
    threshold_parity = all(
        np.isclose(exported["thresholds"][key], fresh_2026[key], rtol=0.0, atol=1e-15)
        for key in exported["thresholds"]
    )
    prediction_parity = bool(
        threshold_parity
        and exported["fit_examples"] == fresh_2026["fit_examples"]
        and exported["predict_events"] == fresh_2026["predict_events"]
        and exported["selected_events"] == fresh_2026["selected_events"]
    )
    if not (schedule_parity and prediction_parity):
        raise RuntimeError(
            f"Rank7 export parity failed: schedule={schedule_parity}, prediction={prediction_parity}"
        )

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temp:
        root = Path(temp)
        model_dir = root / "models"
        model_dir.mkdir(parents=True)
        model_rows: list[dict[str, Any]] = []
        for seed, model in zip(SEEDS, models, strict=True):
            path = model_dir / f"seed_{seed}.npz"
            save_frozen_extra_trees(model, path)
            model_rows.append(
                {
                    "seed": int(seed),
                    "path": str(path.relative_to(root)),
                    "sha256": _sha256(path),
                    "format": "extra_trees_npz_v1",
                    "n_estimators": TREES_PER_SEED,
                    "n_features": len(FEATURE_COLUMNS),
                    "n_outputs": 2,
                }
            )
        hourly_history = _write_hourly_history(root, context["market"])
        exits = {}
        for source, is_funding in (("funding", True), ("premium", False)):
            hold, take, stop = action(is_funding)
            exits[source] = {"hold_bars": hold, "take_bps": take, "stop_bps": stop}
        manifest = {
            "schema_version": 1,
            "strategy_id": "frozen_annual_rank7",
            "policy_type": "frozen_annual_rank7",
            "model_version": "rank7-annual-2026-v1",
            "selected_cadence": "annual",
            "fit_start": "2020-07-01T00:00:00Z",
            "annual_cutoff": ANNUAL_CUTOFF.tz_localize("UTC").isoformat(),
            "valid_from": ANNUAL_CUTOFF.tz_localize("UTC").isoformat(),
            "valid_until": VALID_UNTIL.tz_localize("UTC").isoformat(),
            "snapshot_end": pd.Timestamp(context["dates"].iloc[-1]).tz_localize("UTC").isoformat(),
            "seeds": list(map(int, SEEDS)),
            "trees_per_seed": TREES_PER_SEED,
            "model_format": "extra_trees_npz_v1",
            "extra_trees_params": EXPECTED_MODEL_PARAMS,
            "prediction_n_jobs": 1,
            "feature_columns": list(FEATURE_COLUMNS),
            "source_columns": ["funding_leg", "premium_leg"],
            "source_priority": ["funding", "premium"],
            "delay_bars": 12,
            "delay_initial_fill": "matrix_0",
            "nan_fill_medians": medians.tolist(),
            "clip": [-20.0, 20.0],
            "score_lambda": float(SPEC["lambda"]),
            "thresholds": exported["thresholds"],
            "exits_by_source": exits,
            "anchor_cooldown_bars": 144,
            "no_overlap": True,
            "models": model_rows,
            "runtime_prediction_fixture": {
                "rows": np.asarray(context["matrix"], dtype=float)[
                    np.asarray(base["signals"], dtype=int)[-3:]
                ].tolist(),
                "expected": np.mean(
                    np.stack(
                        [
                            model.predict(
                                np.asarray(context["matrix"], dtype=float)[
                                    np.asarray(base["signals"], dtype=int)[-3:]
                                ]
                            )
                            for model in models
                        ]
                    ),
                    axis=0,
                ).tolist(),
            },
            "hourly_history": hourly_history,
            "parity": {
                "status": "passed",
                "feature_parity": True,
                "prediction_parity": True,
                "schedule_parity": True,
                "feature": feature_parity,
                "prediction": exported,
                "research_full_result_hash": expected["full_result_hash"],
                "research_selected_positions_hash": expected["selected_positions_hash"],
            },
            "research": {
                "manifest_hash": research["manifest_hash"],
                "research_result_path": str(RESEARCH_RESULT),
                "research_result_sha256": _sha256(RESEARCH_RESULT),
                "selection_manifest_path": str(RESEARCH_MANIFEST),
                "selection_manifest_sha256": _sha256(RESEARCH_MANIFEST),
                "folds": [list(row) for row in FOLDS],
                "stats": expected["stats"],
            },
            "export_software": {
                "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
            },
        }
        manifest["bundle_manifest_hash"] = rank7_manifest_hash(manifest)
        (root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        loaded = Rank7Bundle.load(root)
        warm_context = build_rank7_feature_context(live_market, loaded)
        if not np.array_equal(warm_context["matrix"], np.asarray(context["matrix"], dtype=float)):
            raise RuntimeError("Rank7 serialized hourly warm start changed feature parity")
        if output.exists():
            shutil.rmtree(output)
        shutil.copytree(root, output)

    final = Rank7Bundle.load(output)
    return {
        "output": str(output),
        "model_version": final.model_version,
        "models": len(final.models),
        "hourly_history_rows": 0 if final.hourly_history is None else int(len(final.hourly_history)),
        "feature_matrix_sha256": feature_parity["matrix_sha256"],
        "prediction_selected_events_2026h1": exported["selected_events"],
        "research_full_result_hash": expected["full_result_hash"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(export_bundle(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
