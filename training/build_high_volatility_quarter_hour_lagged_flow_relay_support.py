"""Materialize source-only support clocks for frozen HVQHLF-4."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import deque
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from training import preregister_high_volatility_quarter_hour_lagged_flow_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
from training.build_quarter_hour_opening_imbalance_relay_support import strict_prior_midrank


ENV_FILE = "/home/pakchu/rllm/.env"
START = pd.Timestamp("2023-04-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
PREREG_SHA = "130da7294898237b9033ab6fee41bc57f63a9a4507a987f5c24990436be0aa06"
SOURCE_DIR = Path("data/high_volatility_quarter_hour_lagged_flow_relay_sources_2023_2026")
FEATURES = SOURCE_DIR / "quarter_hour_features.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/high_volatility_quarter_hour_lagged_flow_relay_clocks_2023_2026.csv.gz")
SPLIT_DIR = Path("data/high_volatility_quarter_hour_lagged_flow_relay_split_clocks_2023_2026")
CONTROL_DIR = Path("data/high_volatility_quarter_hour_lagged_flow_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_quarter_hour_lagged_flow_relay_support_2026-08-13.json")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), END),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "raw_current_opening_imbalance",
    "include_ols_intercept",
    "no_flow_strength_tail",
    "no_variation_gate",
    "shifted_phase_plus_2m",
    "one_quarter_stale_prediction",
    "direction_flip",
    "same_clock_forced_long",
)
QUERY = """SELECT ts,open,close,volume,taker_buy_base FROM bars_binance
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
ORDER BY ts"""
FEATURE_COLUMNS = (
    "decision_time",
    "source_valid",
    "opening_imbalance",
    "shifted_opening_imbalance",
    "lagged_flow_prediction",
    "lagged_flow_intercept",
    "shifted_lagged_flow_prediction",
    "model_valid",
    "shifted_model_valid",
    "flow_strength_rank",
    "shifted_flow_strength_rank",
    "realized_variation",
    "variation_rank",
)
CLOCK_COLUMNS = (
    "candidate",
    "control",
    "split",
    "decision_time",
    "feature_available_time",
    "entry_time",
    "exit_time",
    "side",
    "lagged_flow_prediction",
    "flow_strength_rank",
    "realized_variation",
    "variation_rank",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def postgres_engine():
    from sqlalchemy import create_engine

    from preprocessing.live_db_features import load_env_file, postgres_url_from_env

    load_env_file(ENV_FILE)
    return create_engine(
        postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10}
    )


def validate_bars(bars: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    required = ["ts", "open", "close", "volume", "taker_buy_base"]
    if list(bars.columns) != required:
        raise RuntimeError(f"HVQHLF source schema must be exactly {required}")
    frame = bars.copy()
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True, errors="raise")
    if frame["ts"].duplicated().any():
        raise RuntimeError("HVQHLF source has duplicate minutes")
    frame = frame.sort_values("ts").reset_index(drop=True)
    expected = pd.date_range(start, end, freq="1min", inclusive="left")
    if len(frame) != len(expected) or not frame["ts"].equals(pd.Series(expected, name="ts")):
        raise RuntimeError("HVQHLF source is not the exact requested one-minute grid")
    for column in required[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    finite = np.isfinite(frame[required[1:]]).all(axis=1)
    valid = (
        finite
        & frame[["open", "close"]].gt(0).all(axis=1)
        & frame["volume"].ge(0)
        & frame["taker_buy_base"].ge(0)
        & frame["taker_buy_base"].le(frame["volume"])
    )
    if not bool(valid.all()):
        raise RuntimeError("HVQHLF source contains invalid price, volume, or taker flow")
    return frame.set_index("ts")


def causal_rolling_prediction(
    imbalance: pd.Series,
    *,
    lag_count: int = 12,
    lookback: int = 8640,
    minimum: int = 5760,
) -> tuple[pd.Series, pd.Series]:
    """Predict each row from lags using only admitted response rows strictly before it."""
    numeric = pd.to_numeric(imbalance, errors="coerce").astype(float)
    lags = pd.concat(
        [numeric.shift(offset).rename(f"lag_{offset}") for offset in range(1, lag_count + 1)],
        axis=1,
    )
    prediction = pd.Series(np.nan, index=numeric.index, dtype=float)
    intercept = pd.Series(np.nan, index=numeric.index, dtype=float)
    history: deque[tuple[np.ndarray, float]] = deque()
    xtx = np.zeros((lag_count + 1, lag_count + 1), dtype=float)
    xty = np.zeros(lag_count + 1, dtype=float)

    for index in numeric.index:
        lag_vector = lags.loc[index].to_numpy(dtype=float)
        lag_valid = np.isfinite(lag_vector).all()
        if lag_valid and len(history) >= minimum:
            if np.linalg.matrix_rank(xtx) == lag_count + 1:
                coefficients = np.linalg.solve(xtx, xty)
                predicted = float(np.dot(coefficients[1:], lag_vector))
                if math.isfinite(predicted) and math.isfinite(float(coefficients[0])):
                    prediction.at[index] = predicted
                    intercept.at[index] = float(coefficients[0])
        response = float(numeric.at[index])
        if lag_valid and math.isfinite(response):
            design = np.concatenate(([1.0], lag_vector))
            if len(history) == lookback:
                old_design, old_response = history.popleft()
                xtx -= np.outer(old_design, old_design)
                xty -= old_design * old_response
            history.append((design, response))
            xtx += np.outer(design, design)
            xty += design * response
    return prediction, intercept


def derive_features(
    bars: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    lag_count: int = 12,
    lookback: int = 8640,
    minimum: int = 5760,
) -> pd.DataFrame:
    source = validate_bars(bars, start, end)
    decisions = pd.date_range(start, end, freq="15min", inclusive="left")
    opening = source.reindex(decisions)
    shifted = source.reindex(decisions + pd.Timedelta(minutes=2))
    opening_volume = opening["volume"].to_numpy(dtype=float)
    shifted_volume = shifted["volume"].to_numpy(dtype=float)
    source_valid = opening_volume > 0
    shifted_valid = shifted_volume > 0
    imbalance_values = np.full(len(decisions), np.nan, dtype=float)
    np.divide(
        2 * opening["taker_buy_base"].to_numpy(dtype=float) - opening_volume,
        opening_volume,
        out=imbalance_values,
        where=source_valid,
    )
    shifted_imbalance_values = np.full(len(decisions), np.nan, dtype=float)
    np.divide(
        2 * shifted["taker_buy_base"].to_numpy(dtype=float) - shifted_volume,
        shifted_volume,
        out=shifted_imbalance_values,
        where=shifted_valid,
    )
    imbalance = pd.Series(imbalance_values, index=decisions)
    shifted_imbalance = pd.Series(shifted_imbalance_values, index=decisions)
    prediction, intercept = causal_rolling_prediction(
        imbalance, lag_count=lag_count, lookback=lookback, minimum=minimum
    )
    shifted_prediction, _ = causal_rolling_prediction(
        shifted_imbalance, lag_count=lag_count, lookback=lookback, minimum=minimum
    )
    squared_returns = np.log(source["close"] / source["open"]).pow(2)
    variation = np.sqrt(
        squared_returns.shift(1).rolling(1440, min_periods=1440).sum()
    ).reindex(decisions)
    flow_rank = strict_prior_midrank(
        prediction.abs(), lookback=lookback, minimum=minimum
    )
    shifted_flow_rank = strict_prior_midrank(
        shifted_prediction.abs(), lookback=lookback, minimum=minimum
    )
    variation_rank = strict_prior_midrank(
        variation,
        lookback=lookback,
        minimum=minimum,
        update_mask=source_valid,
    )
    frame = pd.DataFrame(
        {
            "decision_time": decisions,
            "source_valid": source_valid,
            "opening_imbalance": imbalance.to_numpy(),
            "shifted_opening_imbalance": shifted_imbalance.to_numpy(),
            "lagged_flow_prediction": prediction.to_numpy(),
            "lagged_flow_intercept": intercept.to_numpy(),
            "shifted_lagged_flow_prediction": shifted_prediction.to_numpy(),
            "model_valid": np.isfinite(prediction),
            "shifted_model_valid": np.isfinite(shifted_prediction),
            "flow_strength_rank": flow_rank.to_numpy(),
            "shifted_flow_strength_rank": shifted_flow_rank.to_numpy(),
            "realized_variation": variation.to_numpy(),
            "variation_rank": variation_rank.to_numpy(),
        }
    )
    return frame.loc[:, FEATURE_COLUMNS]


def materialize_features() -> dict[str, Any]:
    from sqlalchemy import text

    engine = postgres_engine()
    try:
        with engine.connect() as connection:
            bars = pd.read_sql_query(
                text(QUERY),
                connection,
                params={"start": START.to_pydatetime(), "end": END.to_pydatetime()},
            )
    finally:
        engine.dispose()
    frame = derive_features(bars, start=START, end=END)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(frame, FEATURES)
    core = {
        "protocol_version": "hvqhlf_4_btc_source_v1",
        "query": QUERY,
        "table": "bars_binance",
        "symbol": "BTCUSDT",
        "interval": "1m",
        "columns": ["ts", "open", "close", "volume", "taker_buy_base"],
        "window": [START.isoformat(), END.isoformat()],
        "exact_minute_grid": True,
        "no_imputation": True,
        "outcomes_opened": False,
        "candidate_incidence_opened": True,
        "output": {
            "path": str(FEATURES),
            "sha256": sha256(FEATURES),
            "rows": len(frame),
            "model_valid_rows": int(frame["model_valid"].sum()),
        },
    }
    payload = {**core, "manifest_hash": canonical_hash(core)}
    SOURCE_MANIFEST.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    return payload


def load_features() -> pd.DataFrame:
    frame = pd.read_csv(FEATURES, compression="gzip")
    frame["decision_time"] = pd.to_datetime(frame["decision_time"], utc=True, errors="raise")
    for column in ("source_valid", "model_valid", "shifted_model_valid"):
        frame[column] = frame[column].astype(str).str.lower().eq("true")
    for column in set(FEATURE_COLUMNS) - {
        "decision_time",
        "source_valid",
        "model_valid",
        "shifted_model_valid",
    }:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def onset_after_previous_valid(valid: pd.Series, eligible: pd.Series) -> pd.Series:
    active = pd.Series(False, index=eligible.index)
    previous_eligible = False
    for index in eligible.index:
        if not bool(valid.at[index]):
            continue
        current = bool(eligible.at[index])
        active.at[index] = current and not previous_eligible
        previous_eligible = current
    return active


def active_and_side(frame: pd.DataFrame, control: str = "primary") -> tuple[pd.Series, pd.Series]:
    if control not in ("primary", *CONTROLS):
        raise ValueError(f"unknown HVQHLF control: {control}")
    prediction = frame["lagged_flow_prediction"].copy()
    strength_rank = frame["flow_strength_rank"].copy()
    valid = frame["model_valid"].copy()
    if control == "raw_current_opening_imbalance":
        prediction = frame["opening_imbalance"].copy()
        strength_rank = strict_prior_midrank(prediction.abs(), lookback=8640, minimum=5760)
        valid = frame["source_valid"].copy()
    elif control == "include_ols_intercept":
        prediction = prediction + frame["lagged_flow_intercept"]
    elif control == "shifted_phase_plus_2m":
        prediction = frame["shifted_lagged_flow_prediction"].copy()
        strength_rank = frame["shifted_flow_strength_rank"].copy()
        valid = frame["shifted_model_valid"].copy()
    elif control == "one_quarter_stale_prediction":
        prediction = prediction.shift(1)
        strength_rank = strength_rank.shift(1)
        valid = valid.shift(1, fill_value=False)
    tail = (
        pd.Series(True, index=frame.index)
        if control == "no_flow_strength_tail"
        else strength_rank.ge(0.95)
    )
    variation = (
        pd.Series(True, index=frame.index)
        if control == "no_variation_gate"
        else frame["variation_rank"].ge(0.65)
    )
    eligible = (
        valid
        & np.isfinite(prediction)
        & prediction.ne(0)
        & np.isfinite(strength_rank)
        & tail
        & np.isfinite(frame["realized_variation"])
        & np.isfinite(frame["variation_rank"])
        & variation
    )
    active = onset_after_previous_valid(valid, eligible)
    side = pd.Series(
        np.where(prediction.gt(0), 1, np.where(prediction.lt(0), -1, 0)),
        index=frame.index,
        dtype=int,
    )
    if control == "direction_flip":
        side = -side
    elif control == "same_clock_forced_long":
        side = pd.Series(1, index=frame.index, dtype=int)
    return active, side


def make_clock(frame: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    active, side = active_and_side(frame, control)
    prediction = (
        frame["opening_imbalance"]
        if control == "raw_current_opening_imbalance"
        else frame["shifted_lagged_flow_prediction"]
        if control == "shifted_phase_plus_2m"
        else frame["lagged_flow_prediction"].shift(1)
        if control == "one_quarter_stale_prediction"
        else frame["lagged_flow_prediction"] + frame["lagged_flow_intercept"]
        if control == "include_ols_intercept"
        else frame["lagged_flow_prediction"]
    )
    strength_rank = (
        frame["shifted_flow_strength_rank"]
        if control == "shifted_phase_plus_2m"
        else frame["flow_strength_rank"].shift(1)
        if control == "one_quarter_stale_prediction"
        else frame["flow_strength_rank"]
    )
    rows: list[dict[str, Any]] = []
    next_allowed: pd.Timestamp | None = None
    for index in frame.index[active]:
        decision = pd.Timestamp(frame.at[index, "decision_time"])
        feature_available = decision + pd.Timedelta(minutes=1) if control == "raw_current_opening_imbalance" else decision
        entry = decision + pd.Timedelta(minutes=5)
        exit_time = entry + pd.Timedelta(hours=4)
        if feature_available > entry:
            raise RuntimeError("HVQHLF feature unavailable at entry")
        if next_allowed is not None and entry < next_allowed:
            continue
        split = next(
            (
                name
                for name, (start, end) in SPLITS.items()
                if entry >= start and exit_time <= end
            ),
            None,
        )
        if split is None:
            continue
        next_allowed = exit_time
        rows.append(
            {
                "candidate": prereg.POLICY_ID,
                "control": control,
                "split": split,
                "decision_time": decision,
                "feature_available_time": feature_available,
                "entry_time": entry,
                "exit_time": exit_time,
                "side": int(side.at[index]),
                "lagged_flow_prediction": float(prediction.at[index]),
                "flow_strength_rank": float(strength_rank.at[index]),
                "realized_variation": float(frame.at[index, "realized_variation"]),
                "variation_rank": float(frame.at[index, "variation_rank"]),
            }
        )
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def support_stats(clock: pd.DataFrame, split: str) -> dict[str, Any]:
    frame = clock[clock["split"].eq(split)]
    if frame.empty:
        return {
            "events": 0,
            "longs": 0,
            "shorts": 0,
            "minority_side_share": 0.0,
            "max_month_share": 0.0,
        }
    longs = int(frame["side"].eq(1).sum())
    shorts = int(frame["side"].eq(-1).sum())
    months = frame["entry_time"].dt.strftime("%Y-%m").value_counts()
    return {
        "events": len(frame),
        "longs": longs,
        "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(frame),
        "max_month_share": int(months.max()) / len(frame),
    }


def support_checks(support: dict[str, dict[str, Any]]) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for split, metrics in support.items():
        checks[f"{split}_minimum_events"] = metrics["events"] >= MINIMUM[split]
        checks[f"{split}_side_balance"] = metrics["minority_side_share"] >= 0.20
        checks[f"{split}_month_concentration"] = metrics["max_month_share"] <= 0.45
    return checks


def run() -> dict[str, Any]:
    if sha256(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVQHLF preregistration hash drift")
    source_manifest = materialize_features()
    frame = load_features()
    primary = make_clock(frame)
    controls = {name: make_clock(frame, name) for name in CONTROLS}
    CLOCK.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(primary, CLOCK)
    for split in SPLITS:
        _write_gzip_csv(
            primary[primary["split"].eq(split)].reset_index(drop=True),
            SPLIT_DIR / f"{split}.csv.gz",
        )
    for name, control_clock in controls.items():
        _write_gzip_csv(control_clock, CONTROL_DIR / f"{name}.csv.gz")
    support = {split: support_stats(primary, split) for split in SPLITS}
    checks = support_checks(support)
    passed = all(checks.values())
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    core = {
        "protocol_version": "hvqhlf_4_source_support_v1",
        "policy_id": prereg.POLICY_ID,
        "preregistration": {
            "path": str(prereg.DEFAULT_OUTPUT),
            "sha256": sha256(prereg.DEFAULT_OUTPUT),
            "manifest_hash": registration["manifest_hash"],
        },
        "source_manifest": {
            "path": str(SOURCE_MANIFEST),
            "sha256": sha256(SOURCE_MANIFEST),
            "manifest_hash": source_manifest["manifest_hash"],
        },
        "completed_preentry_sources_opened": True,
        "postentry_return_pnl_execution_price_opened": False,
        "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha256(CLOCK), "rows": len(primary)},
        "split_clocks": {
            split: {
                "path": str(SPLIT_DIR / f"{split}.csv.gz"),
                "sha256": sha256(SPLIT_DIR / f"{split}.csv.gz"),
                "rows": int(primary["split"].eq(split).sum()),
            }
            for split in SPLITS
        },
        "controls": {
            name: {
                "path": str(CONTROL_DIR / f"{name}.csv.gz"),
                "sha256": sha256(CONTROL_DIR / f"{name}.csv.gz"),
                "rows": len(control_clock),
                "promotion_authorized": False,
            }
            for name, control_clock in controls.items()
        },
        "support": support,
        "support_checks": checks,
        "support_passed": passed,
        "advance_to_gross9_novelty": passed,
        "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    payload = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    return payload


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    result = run()
    print(
        json.dumps(
            {"passed": result["support_passed"], "support": result["support"]},
            indent=2,
            ensure_ascii=False,
        )
    )
