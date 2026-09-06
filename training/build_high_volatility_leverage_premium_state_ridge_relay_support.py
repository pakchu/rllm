"""Open source-only OOS incidence for the frozen HVLPSR-8 model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_high_volatility_leverage_premium_state_ridge_relay as prereg
from training import train_high_volatility_leverage_premium_state_ridge_relay as trained
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


PREREG_SHA = "4c393fa82a306f52c0fe42e7e2dab5cf5af47d95209b4745ddf1e7d8f04069d3"
SOURCE_MANIFEST_SHA = "6503acf269c94537b14ce003054a55fee0c733791a8c6fcf372fb02708f3b457"
PANEL_SHA = "83116f484221baf944fc2f2cdc8220b21e67ae818305979dc6b33bf66b0a4f1a"
MODEL_SHA = "faa07ad4a6de4100d5baead06fe93962d32fba7c4104b7b36553ecaed4755eeb"
CLOCK = Path("data/high_volatility_leverage_premium_state_ridge_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_leverage_premium_state_ridge_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_leverage_premium_state_ridge_relay_support_2026-08-10.json")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_volatility_gate",
    "no_prediction_strength_gate",
    "one_boundary_stale_features",
    "direction_flip",
    "forced_long",
)
COLUMNS = (
    "candidate", "control", "split", "decision_time", "feature_available_time", "entry_time", "exit_time", "side",
    "prediction", "prediction_strength_threshold", *trained.FEATURES,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError(f"manifest drift: {path}")
    return payload


def verify_frozen_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    expected = ((prereg.DEFAULT_OUTPUT, PREREG_SHA), (trained.SOURCE_MANIFEST, SOURCE_MANIFEST_SHA), (trained.PANEL, PANEL_SHA), (trained.MODEL, MODEL_SHA))
    if any(sha(path) != digest for path, digest in expected):
        raise RuntimeError("HVLPSR frozen predecessor hash drift")
    registration = load_json(prereg.DEFAULT_OUTPUT)
    source = load_json(trained.SOURCE_MANIFEST)
    model = load_json(trained.MODEL)
    if source.get("oos_incidence_opened") is not False or source.get("oos_outcomes_opened") is not False:
        raise RuntimeError("HVLPSR source was not OOS sealed")
    if model.get("oos_incidence_opened") is not False or model.get("oos_outcomes_opened") is not False or model.get("refit_authorized") is not False:
        raise RuntimeError("HVLPSR model was not frozen OOS blind")
    if tuple(model.get("feature_order", ())) != trained.FEATURES:
        raise RuntimeError("HVLPSR feature order drift")
    return registration, source, model


def features(model: dict[str, Any]) -> pd.DataFrame:
    frame = pd.read_csv(trained.PANEL, compression="gzip")
    frame["decision_time"] = pd.to_datetime(frame.decision_time, utc=True)
    frame["source_valid"] = frame.source_valid.astype(str).str.lower().eq("true")
    for column in trained.FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    matrix = frame.loc[:, trained.FEATURES].to_numpy(float)
    mean = np.asarray(model["mean"], dtype=float)
    scale = np.asarray(model["scale"], dtype=float)
    coefficient = np.asarray(model["coefficient"], dtype=float)
    prediction = ((matrix - mean) / scale) @ coefficient + float(model["intercept"])
    frame["prediction"] = prediction
    frame["signal_valid"] = frame.source_valid & np.isfinite(frame[list(trained.FEATURES) + ["prediction"]]).all(axis=1)
    return frame


def conditions(frame: pd.DataFrame, model: dict[str, Any], control: str) -> tuple[pd.Series, pd.Series]:
    prediction = frame.prediction
    variation_rank = frame.variation_rank
    if control == "one_boundary_stale_features":
        prediction = prediction.shift(1)
        variation_rank = variation_rank.shift(1)
    volatility_gate = pd.Series(True, index=frame.index) if control == "no_volatility_gate" else variation_rank.ge(0.65)
    threshold = float(model["prediction_strength_threshold"])
    strength_gate = pd.Series(True, index=frame.index) if control == "no_prediction_strength_gate" else prediction.abs().ge(threshold)
    active = frame.signal_valid & np.isfinite(prediction) & np.isfinite(variation_rank) & prediction.ne(0) & volatility_gate & strength_gate
    side = pd.Series(np.where(prediction.gt(0), 1, -1), index=frame.index)
    if control == "direction_flip":
        side = -side
    elif control == "forced_long":
        side = pd.Series(1, index=frame.index)
    return active, side


def clock(frame: pd.DataFrame, model: dict[str, Any], control: str = "primary") -> pd.DataFrame:
    active, side = conditions(frame, model, control)
    rows: list[dict[str, Any]] = []
    for index in frame.index[active & frame.decision_time.ge(SPLITS["train"][0])]:
        decision = pd.Timestamp(frame.at[index, "decision_time"])
        entry = decision + pd.Timedelta(minutes=5)
        exit_ = entry + pd.Timedelta(hours=8)
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_ <= end), None)
        if split is None:
            continue
        row = {
            "candidate": "HVLPSR-8", "control": control, "split": split, "decision_time": decision,
            "feature_available_time": decision, "entry_time": entry, "exit_time": exit_, "side": int(side.at[index]),
            "prediction": float(frame.at[index, "prediction"]),
            "prediction_strength_threshold": float(model["prediction_strength_threshold"]),
        }
        row.update({feature: float(frame.at[index, feature]) for feature in trained.FEATURES})
        rows.append(row)
    return pd.DataFrame(rows, columns=COLUMNS)


def stats(candidate: pd.DataFrame, split: str) -> dict[str, float | int]:
    subset = candidate[candidate.split.eq(split)]
    if subset.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs = int(subset.side.eq(1).sum())
    shorts = int(subset.side.eq(-1).sum())
    months = subset.entry_time.dt.strftime("%Y-%m").value_counts()
    return {"events": len(subset), "longs": longs, "shorts": shorts, "minority_side_share": min(longs, shorts) / len(subset), "max_month_share": int(months.max()) / len(subset)}


def run() -> dict[str, Any]:
    registration, source, model = verify_frozen_inputs()
    frame = features(model)
    primary = clock(frame, model)
    controls = {name: clock(frame, model, name) for name in CONTROLS}
    CLOCK.parent.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(primary, CLOCK)
    for name, candidate in controls.items():
        _write_gzip_csv(candidate, CONTROL_DIR / f"{name}.csv.gz")
    support = {name: stats(primary, name) for name in SPLITS}
    checks: dict[str, bool] = {}
    for name, values in support.items():
        checks[f"{name}_minimum_events"] = values["events"] >= MINIMUM[name]
        checks[f"{name}_side_balance"] = values["minority_side_share"] >= 0.2
        checks[f"{name}_month_concentration"] = values["max_month_share"] <= 0.45
    passed = all(checks.values())
    core = {
        "protocol_version": "hvlpsr_8_oos_source_support_v1", "policy_id": "HVLPSR-8",
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]},
        "source_manifest": {"path": str(trained.SOURCE_MANIFEST), "sha256": SOURCE_MANIFEST_SHA, "manifest_hash": source["manifest_hash"]},
        "model_freeze": {"path": str(trained.MODEL), "sha256": MODEL_SHA, "manifest_hash": model["manifest_hash"], "predecessor_mutated": False},
        "completed_preentry_sources_opened": True, "pretraining_outcomes_opened_as_authorized": True,
        "oos_postentry_return_pnl_execution_price_opened": False, "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(candidate), "promotion_authorized": False} for name, candidate in controls.items()},
        "support": support, "support_checks": checks, "support_passed": passed,
        "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    result = run()
    print(json.dumps({"passed": result["support_passed"], "support": result["support"]}, indent=2))
