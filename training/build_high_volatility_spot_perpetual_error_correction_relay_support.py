"""Materialize source-only HVSPER-8 support clocks."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_high_volatility_spot_perpetual_error_correction_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


ENV_FILE = "/home/pakchu/rllm/.env"
START = pd.Timestamp("2023-03-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
PREREG_SHA = "bf6d075bb7edca04ab4ba0616ad01f075a1806297973f8e40cd1da2ddcd4b880"
SOURCE_DIR = Path("data/high_volatility_spot_perpetual_error_correction_sources_2023_2026")
STATES = SOURCE_DIR / "states.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/high_volatility_spot_perpetual_error_correction_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_spot_perpetual_error_correction_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_spot_perpetual_error_correction_relay_support_2026-08-10.json")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_variation_gate",
    "no_innovation_tail",
    "no_leadership_gate",
    "one_decision_stale_model",
    "direction_flip",
    "forced_long",
)
QUERY = """SELECT p.ts,
p.open AS perpetual_open,p.high AS perpetual_high,p.low AS perpetual_low,p.close AS perpetual_close,
s.open AS spot_open,s.high AS spot_high,s.low AS spot_low,s.close AS spot_close
FROM bars_binance p
JOIN bars_binance_spot s ON s.symbol=p.symbol AND s.interval=p.interval AND s.ts=p.ts
WHERE p.symbol='BTCUSDT' AND p.interval='1m' AND p.ts>=:start AND p.ts<:end
ORDER BY p.ts"""
CLOCK_COLUMNS = (
    "candidate", "control", "split", "decision_time", "feature_available_time",
    "entry_time", "exit_time", "side", "alpha_spot", "alpha_perpetual",
    "perpetual_leadership_share", "perpetual_hour_return", "spot_hour_return",
    "lead_innovation", "innovation_rank", "realized_variation", "variation_rank",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def strict_prior_midrank(values: pd.Series, lookback: int = 270, minimum: int = 180) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    result = pd.Series(np.nan, index=numeric.index, dtype=float)
    history: list[float] = []
    for index, current in numeric.items():
        prior = history[-lookback:]
        if math.isfinite(current) and len(prior) >= minimum:
            array = np.asarray(prior)
            result.at[index] = (np.sum(array < current) + 0.5 * np.sum(array == current)) / len(array)
        if math.isfinite(current):
            history.append(float(current))
    return result


def fit_error_correction(spot_log: np.ndarray, perpetual_log: np.ndarray) -> tuple[float, float, float]:
    if len(spot_log) != 2016 or len(perpetual_log) != 2016:
        raise ValueError("HVSPER model requires 2016 paired prices")
    delta_spot = np.diff(spot_log)
    delta_perpetual = np.diff(perpetual_log)
    spread = perpetual_log - spot_log
    design = np.column_stack(
        [np.ones(2014), spread[1:-1], delta_spot[:-1], delta_perpetual[:-1]]
    )
    if np.linalg.matrix_rank(design) != 4:
        raise ValueError("HVSPER error-correction design is rank deficient")
    spot_coef = np.linalg.lstsq(design, delta_spot[1:], rcond=None)[0]
    perpetual_coef = np.linalg.lstsq(design, delta_perpetual[1:], rcond=None)[0]
    if not np.isfinite(spot_coef).all() or not np.isfinite(perpetual_coef).all():
        raise ValueError("HVSPER error-correction coefficients are nonfinite")
    alpha_spot = float(spot_coef[1])
    alpha_perpetual = float(perpetual_coef[1])
    share = (
        alpha_spot / (alpha_spot - alpha_perpetual)
        if alpha_spot > 0.0 and alpha_perpetual < 0.0
        else math.nan
    )
    return alpha_spot, alpha_perpetual, float(share)


def postgres_engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env

    load_env_file(ENV_FILE)
    return create_engine(postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10})


def load_pairs() -> pd.DataFrame:
    from sqlalchemy import text

    engine = postgres_engine()
    with engine.connect() as connection:
        raw = pd.read_sql_query(
            text(QUERY),
            connection,
            params={"start": START.to_pydatetime(), "end": END.to_pydatetime()},
        )
    engine.dispose()
    raw["ts"] = pd.to_datetime(raw["ts"], utc=True)
    if raw["ts"].duplicated().any():
        raise RuntimeError("duplicate HVSPER paired source timestamp")
    full_grid = pd.date_range(START, END - pd.Timedelta(minutes=1), freq="1min")
    raw = raw.set_index("ts").reindex(full_grid).rename_axis("ts")
    numeric_columns = [column for column in raw.columns]
    raw[numeric_columns] = raw[numeric_columns].apply(pd.to_numeric, errors="coerce")
    finite_positive = np.isfinite(raw[numeric_columns]).all(axis=1) & raw[numeric_columns].gt(0).all(axis=1)
    coherent = pd.Series(True, index=raw.index)
    for market in ("perpetual", "spot"):
        coherent &= raw[f"{market}_high"].ge(raw[[f"{market}_open", f"{market}_close"]].max(axis=1))
        coherent &= raw[f"{market}_low"].le(raw[[f"{market}_open", f"{market}_close"]].min(axis=1))
        coherent &= raw[f"{market}_high"].ge(raw[f"{market}_low"])
    raw["row_valid"] = finite_positive & coherent
    return raw


def build_states(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    decisions = pd.date_range(START + pd.Timedelta(days=7), END, freq="8h", inclusive="left")
    for decision in decisions:
        window = raw.loc[decision - pd.Timedelta(days=7): decision - pd.Timedelta(minutes=1)]
        source_valid = len(window) == 2016 and bool(window["row_valid"].all())
        row: dict[str, Any] = {"decision_time": decision, "source_valid": source_valid}
        if source_valid:
            spot_log = np.log(window["spot_close"].to_numpy(dtype=float))
            perpetual_log = np.log(window["perpetual_close"].to_numpy(dtype=float))
            try:
                alpha_spot, alpha_perpetual, leadership_share = fit_error_correction(
                    spot_log, perpetual_log
                )
                model_valid = math.isfinite(leadership_share)
            except ValueError:
                alpha_spot = alpha_perpetual = leadership_share = math.nan
                model_valid = False
            hour = window.iloc[-60:]
            perpetual_hour_return = float(
                np.log(hour["perpetual_close"].iloc[-1] / hour["perpetual_open"].iloc[0])
            )
            spot_hour_return = float(np.log(hour["spot_close"].iloc[-1] / hour["spot_open"].iloc[0]))
            lead_innovation = perpetual_hour_return - spot_hour_return
            realized_variation = float(np.sqrt(np.square(np.diff(perpetual_log[-1440:])).sum()))
        else:
            alpha_spot = alpha_perpetual = leadership_share = math.nan
            perpetual_hour_return = spot_hour_return = lead_innovation = realized_variation = math.nan
            model_valid = False
        row.update(
            {
                "model_valid": model_valid,
                "alpha_spot": alpha_spot,
                "alpha_perpetual": alpha_perpetual,
                "perpetual_leadership_share": leadership_share,
                "perpetual_hour_return": perpetual_hour_return,
                "spot_hour_return": spot_hour_return,
                "lead_innovation": lead_innovation,
                "realized_variation": realized_variation,
            }
        )
        rows.append(row)
    states = pd.DataFrame(rows)
    rank_valid = states["source_valid"] & np.isfinite(states["lead_innovation"])
    states["innovation_rank"] = strict_prior_midrank(states["lead_innovation"].abs().where(rank_valid))
    states["variation_rank"] = strict_prior_midrank(states["realized_variation"].where(rank_valid))
    return states


def active_and_side(states: pd.DataFrame, control: str) -> tuple[pd.Series, pd.Series]:
    share = states["perpetual_leadership_share"]
    model_valid = states["model_valid"]
    if control == "one_decision_stale_model":
        share = share.shift(1)
        model_valid = model_valid.shift(1, fill_value=False)
    direction = np.sign(states["perpetual_hour_return"])
    agreement = direction.ne(0) & np.sign(states["lead_innovation"]).eq(direction)
    leadership = pd.Series(True, index=states.index) if control == "no_leadership_gate" else share.ge(0.60)
    innovation = pd.Series(True, index=states.index) if control == "no_innovation_tail" else states["innovation_rank"].ge(0.70)
    variation = pd.Series(True, index=states.index) if control == "no_variation_gate" else states["variation_rank"].ge(0.65)
    active = states["source_valid"] & model_valid.astype(bool) & agreement & leadership & innovation & variation
    side = direction.astype(float)
    if control == "direction_flip":
        side = -side
    elif control == "forced_long":
        side = pd.Series(1.0, index=states.index)
    return active, side


def make_clock(states: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    active, side = active_and_side(states, control)
    rows: list[dict[str, Any]] = []
    next_allowed: pd.Timestamp | None = None
    for index in states.index[active]:
        decision = pd.Timestamp(states.at[index, "decision_time"])
        entry = decision + pd.Timedelta(minutes=5)
        exit_time = entry + pd.Timedelta(hours=8)
        if next_allowed is not None and entry < next_allowed:
            continue
        split = next(
            (name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end),
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
                "feature_available_time": decision,
                "entry_time": entry,
                "exit_time": exit_time,
                "side": int(side.at[index]),
                **{
                    column: float(states.at[index, column])
                    for column in CLOCK_COLUMNS[8:]
                },
            }
        )
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def support_stats(clock: pd.DataFrame, split: str) -> dict[str, Any]:
    frame = clock[clock["split"].eq(split)]
    if frame.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
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


def run() -> dict[str, Any]:
    if sha256(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVSPER preregistration hash drift")
    raw = load_pairs()
    states = build_states(raw)
    del raw
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(states, STATES)
    source_core = {
        "protocol_version": "hvsper_8_synchronized_source_v1",
        "query": QUERY,
        "tables": ["bars_binance", "bars_binance_spot"],
        "symbol": "BTCUSDT",
        "interval": "1m",
        "window": [START.isoformat(), END.isoformat()],
        "outcomes_opened": False,
        "candidate_incidence_opened": True,
        "no_imputation": True,
        "output": {
            "path": str(STATES),
            "sha256": sha256(STATES),
            "rows": len(states),
            "source_valid_rows": int(states["source_valid"].sum()),
            "model_valid_rows": int(states["model_valid"].sum()),
        },
    }
    source_payload = {**source_core, "manifest_hash": canonical_hash(source_core)}
    SOURCE_MANIFEST.write_text(json.dumps(source_payload, indent=2, allow_nan=False) + "\n")
    primary = make_clock(states)
    controls = {name: make_clock(states, name) for name in CONTROLS}
    CLOCK.parent.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(primary, CLOCK)
    for name, frame in controls.items():
        _write_gzip_csv(frame, CONTROL_DIR / f"{name}.csv.gz")
    support = {name: support_stats(primary, name) for name in SPLITS}
    checks: dict[str, bool] = {}
    for name, metrics in support.items():
        checks[f"{name}_minimum_events"] = metrics["events"] >= MINIMUM[name]
        checks[f"{name}_side_balance"] = metrics["minority_side_share"] >= 0.20
        checks[f"{name}_month_concentration"] = metrics["max_month_share"] <= 0.45
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    passed = all(checks.values())
    core = {
        "protocol_version": "hvsper_8_source_support_v1",
        "policy_id": prereg.POLICY_ID,
        "preregistration": {
            "path": str(prereg.DEFAULT_OUTPUT),
            "sha256": sha256(prereg.DEFAULT_OUTPUT),
            "manifest_hash": registration["manifest_hash"],
        },
        "source_manifest": {
            "path": str(SOURCE_MANIFEST),
            "sha256": sha256(SOURCE_MANIFEST),
            "manifest_hash": source_payload["manifest_hash"],
        },
        "completed_preentry_sources_opened": True,
        "postentry_return_pnl_execution_price_opened": False,
        "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha256(CLOCK), "rows": len(primary)},
        "controls": {
            name: {
                "path": str(CONTROL_DIR / f"{name}.csv.gz"),
                "sha256": sha256(CONTROL_DIR / f"{name}.csv.gz"),
                "rows": len(frame),
                "promotion_authorized": False,
            }
            for name, frame in controls.items()
        },
        "support": support,
        "support_checks": checks,
        "support_passed": passed,
        "advance_to_gross9_novelty": passed,
        "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    payload = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    result = run()
    print(json.dumps({"passed": result["support_passed"], "support": result["support"]}, indent=2))
