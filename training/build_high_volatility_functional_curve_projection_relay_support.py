"""Source-only support gate for frozen HVFCPR-24."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_high_volatility_functional_curve_projection_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


ENV_FILE = "/home/pakchu/rllm/.env"
START = pd.Timestamp("2022-01-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
PREREG_SHA = "a3390eb94d7402b43dfdf3a280efb5f952dac3fb2670d206f219539725586cc4"
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM_EVENTS = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_variation_gate",
    "no_forecast_strength_gate",
    "one_day_stale_forecast",
    "rolling_mean_terminal_only",
    "direction_flip",
    "forced_long",
)
ROOT = Path("data/high_volatility_functional_curve_projection_relay_sources_2022_2026")
CURVES = ROOT / "daily_cidr_curves.csv.gz"
STATES = ROOT / "forecast_states.csv.gz"
MANIFEST = ROOT / "manifest.json"
CLOCK = Path("data/high_volatility_functional_curve_projection_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_functional_curve_projection_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_functional_curve_projection_relay_support_2026-08-10.json")
CURVE_COLUMNS = tuple(f"cidr_{index:03d}" for index in range(288))
CLOCK_COLUMNS = (
    "candidate", "control", "split", "source_day", "decision_time",
    "feature_available_time", "entry_time", "exit_time", "side",
    "terminal_forecast", "forecast_strength_rank", "mean_terminal",
    "realized_variation", "variation_rank", "component_count",
    "explained_variance",
)
QUERY = """SELECT ts,open,high,low,close
FROM bars_binance
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
ORDER BY ts"""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def strict_prior_midrank(
    values: pd.Series,
    lookback: int = 252,
    minimum: int = 126,
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
    output = np.full(len(numeric), np.nan)
    history: list[float] = []
    for index, current in enumerate(numeric):
        prior = np.asarray(history[-lookback:], dtype=float)
        if math.isfinite(current) and len(prior) >= minimum:
            output[index] = (
                np.sum(prior < current) + 0.5 * np.sum(prior == current)
            ) / len(prior)
        if math.isfinite(current):
            history.append(float(current))
    return pd.Series(output, index=values.index)


def functional_forecast(curves: np.ndarray) -> tuple[np.ndarray, int, float, float]:
    """Forecast one curve from exactly 180 prior complete CIDR curves."""
    matrix = np.asarray(curves, dtype=float)
    if matrix.shape != (180, 288) or not np.isfinite(matrix).all():
        raise ValueError("HVFCPR forecast requires a finite 180x288 matrix")
    mean_curve = matrix.mean(axis=0)
    centered = matrix - mean_curve
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    energy = np.square(singular_values)
    total = float(energy.sum())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("HVFCPR zero functional variation")
    cumulative = np.cumsum(energy) / total
    components = int(np.searchsorted(cumulative, 0.90, side="left") + 1)
    explained = float(cumulative[components - 1])
    scores = centered @ vt[:components].T
    forecast_scores = np.empty(components, dtype=float)
    design = np.column_stack([np.ones(179), scores[:-1]])
    for component in range(components):
        # Each component uses only its own intercept and lagged score.
        component_design = design[:, [0, component + 1]]
        coefficients = np.linalg.lstsq(
            component_design,
            scores[1:, component],
            rcond=None,
        )[0]
        forecast_scores[component] = (
            coefficients[0] + coefficients[1] * scores[-1, component]
        )
    forecast_curve = mean_curve + forecast_scores @ vt[:components]
    if not np.isfinite(forecast_curve).all():
        raise ValueError("HVFCPR nonfinite curve forecast")
    return forecast_curve, components, explained, float(mean_curve[-1])


def postgres_engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env

    load_env_file(ENV_FILE)
    return create_engine(postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10})


def materialize_curves() -> tuple[pd.DataFrame, dict[str, Any]]:
    from sqlalchemy import text

    database = postgres_engine()
    with database.connect() as connection:
        raw = pd.read_sql_query(
            text(QUERY),
            connection,
            params={"start": START.to_pydatetime(), "end": END.to_pydatetime()},
        )
    database.dispose()
    raw["ts"] = pd.to_datetime(raw["ts"], utc=True)
    for column in ("open", "high", "low", "close"):
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    if raw["ts"].duplicated().any():
        raise RuntimeError("duplicate HVFCPR source timestamps")
    raw["source_day"] = raw["ts"].dt.floor("D")
    rows: list[dict[str, Any]] = []
    for day in pd.date_range(START, END, freq="1D", inclusive="left"):
        frame = raw[raw["source_day"].eq(day)].sort_values("ts")
        expected = pd.date_range(day, day + pd.Timedelta("1D"), freq="1min", inclusive="left")
        valid = (
            len(frame) == 1440
            and np.array_equal(
                frame["ts"].astype("int64").to_numpy(),
                expected.astype("int64").to_numpy(),
            )
            and np.isfinite(frame[["open", "high", "low", "close"]]).all().all()
            and frame[["open", "high", "low", "close"]].gt(0).all().all()
            and frame["high"].ge(frame[["open", "close"]].max(axis=1)).all()
            and frame["low"].le(frame[["open", "close"]].min(axis=1)).all()
            and frame["high"].ge(frame["low"]).all()
        )
        curve = np.full(288, np.nan)
        variation = math.nan
        if valid:
            close = frame["close"].to_numpy(float)
            first_open = float(frame["open"].iloc[0])
            curve = np.log(close[4::5] / first_open)
            variation = float(np.sqrt(np.square(np.diff(np.log(close))).sum()))
            valid = bool(
                curve.shape == (288,)
                and np.isfinite(curve).all()
                and math.isfinite(variation)
                and variation > 0
            )
        row: dict[str, Any] = {
            "source_day": day,
            "source_valid": valid,
            "realized_variation": variation if valid else math.nan,
        }
        row.update(
            {
                column: float(value) if valid else math.nan
                for column, value in zip(CURVE_COLUMNS, curve)
            }
        )
        rows.append(row)
    curves = pd.DataFrame(rows)
    curves["variation_rank"] = strict_prior_midrank(
        curves["realized_variation"].where(curves["source_valid"])
    )
    ROOT.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(curves, CURVES)
    core = {
        "protocol_version": "hvfcpr_daily_cidr_source_v1",
        "query": QUERY,
        "window": [START.isoformat(), END.isoformat()],
        "outcomes_opened": False,
        "candidate_incidence_opened_before_materialization": False,
        "curve_definition": "288 UTC-aligned five-minute closes relative to first minute open",
        "output": {
            "path": str(CURVES),
            "sha256": sha256(CURVES),
            "rows": len(curves),
            "valid_rows": int(curves["source_valid"].sum()),
        },
    }
    manifest = {**core, "manifest_hash": canonical_hash(core)}
    MANIFEST.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return curves, manifest


def build_states(curves: pd.DataFrame) -> pd.DataFrame:
    curve_matrix = curves.loc[:, CURVE_COLUMNS].to_numpy(float)
    source_valid = curves["source_valid"].to_numpy(bool)
    rows: list[dict[str, Any]] = []
    for index in range(len(curves)):
        source_day = pd.Timestamp(curves.at[index, "source_day"])
        decision = source_day + pd.Timedelta("1D")
        valid = index >= 179 and source_valid[index - 179 : index + 1].all()
        terminal = mean_terminal = explained = math.nan
        components = 0
        if valid:
            try:
                forecast, components, explained, mean_terminal = functional_forecast(
                    curve_matrix[index - 179 : index + 1]
                )
                terminal = float(forecast[-1])
                valid = bool(math.isfinite(terminal) and terminal != 0)
            except (ValueError, np.linalg.LinAlgError):
                valid = False
        rows.append(
            {
                "source_day": source_day,
                "decision_time": decision,
                "forecast_valid": valid,
                "terminal_forecast": terminal if valid else math.nan,
                "mean_terminal": mean_terminal if valid else math.nan,
                "component_count": components if valid else 0,
                "explained_variance": explained if valid else math.nan,
                "realized_variation": float(curves.at[index, "realized_variation"]),
                "variation_rank": float(curves.at[index, "variation_rank"]),
            }
        )
    states = pd.DataFrame(rows)
    states["forecast_strength_rank"] = strict_prior_midrank(
        states["terminal_forecast"].abs().where(states["forecast_valid"])
    )
    _write_gzip_csv(states, STATES)
    return states


def conditions(states: pd.DataFrame, control: str) -> tuple[pd.Series, pd.Series]:
    forecast = states["terminal_forecast"]
    strength_rank = states["forecast_strength_rank"]
    if control == "one_day_stale_forecast":
        forecast = forecast.shift(1)
        strength_rank = strength_rank.shift(1)
    if control == "rolling_mean_terminal_only":
        forecast = states["mean_terminal"]
    variation_gate = (
        pd.Series(True, index=states.index)
        if control == "no_variation_gate"
        else states["variation_rank"].ge(0.65)
    )
    strength_gate = (
        pd.Series(True, index=states.index)
        if control == "no_forecast_strength_gate"
        else strength_rank.ge(0.75)
    )
    active = (
        states["forecast_valid"]
        & np.isfinite(forecast)
        & forecast.ne(0)
        & variation_gate
        & strength_gate
    )
    side = np.sign(forecast)
    if control == "direction_flip":
        side = -side
    elif control == "forced_long":
        side = pd.Series(1.0, index=states.index)
    return active, side


def clock(states: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    active, side = conditions(states, control)
    rows: list[dict[str, Any]] = []
    next_available: pd.Timestamp | None = None
    for index in states.index[active]:
        decision = pd.Timestamp(states.at[index, "decision_time"])
        entry = decision + pd.Timedelta("5m")
        exit_time = entry + pd.Timedelta("24h")
        if next_available is not None and entry < next_available:
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
        next_available = exit_time
        rows.append(
            {
                "candidate": POLICY_ID,
                "control": control,
                "split": split,
                "source_day": states.at[index, "source_day"],
                "decision_time": decision,
                "feature_available_time": decision,
                "entry_time": entry,
                "exit_time": exit_time,
                "side": int(side.at[index]),
                "terminal_forecast": float(states.at[index, "terminal_forecast"]),
                "forecast_strength_rank": float(states.at[index, "forecast_strength_rank"]),
                "mean_terminal": float(states.at[index, "mean_terminal"]),
                "realized_variation": float(states.at[index, "realized_variation"]),
                "variation_rank": float(states.at[index, "variation_rank"]),
                "component_count": int(states.at[index, "component_count"]),
                "explained_variance": float(states.at[index, "explained_variance"]),
            }
        )
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def support_stats(candidate_clock: pd.DataFrame, split: str) -> dict[str, float | int]:
    selected = candidate_clock[candidate_clock["split"].eq(split)]
    if selected.empty:
        return {
            "events": 0, "longs": 0, "shorts": 0,
            "minority_side_share": 0.0, "max_month_share": 0.0,
        }
    longs = int(selected["side"].eq(1).sum())
    shorts = int(selected["side"].eq(-1).sum())
    return {
        "events": len(selected),
        "longs": longs,
        "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(selected),
        "max_month_share": (
            int(selected["entry_time"].dt.strftime("%Y-%m").value_counts().max())
            / len(selected)
        ),
    }


def run() -> dict[str, Any]:
    if sha256(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVFCPR preregistration drift")
    curves, source_manifest = materialize_curves()
    states = build_states(curves)
    primary = clock(states)
    controls = {name: clock(states, name) for name in CONTROLS}
    CLOCK.parent.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(primary, CLOCK)
    for name, value in controls.items():
        _write_gzip_csv(value, CONTROL_DIR / f"{name}.csv.gz")
    support = {name: support_stats(primary, name) for name in SPLITS}
    checks: dict[str, bool] = {}
    for name, values in support.items():
        checks[f"{name}_minimum_events"] = values["events"] >= MINIMUM_EVENTS[name]
        checks[f"{name}_side_balance"] = values["minority_side_share"] >= 0.20
        checks[f"{name}_month_concentration"] = values["max_month_share"] <= 0.45
    passed = all(checks.values())
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    core = {
        "protocol_version": "hvfcpr_24_source_support_v1",
        "policy_id": POLICY_ID,
        "preregistration": {
            "path": str(prereg.DEFAULT_OUTPUT),
            "sha256": sha256(prereg.DEFAULT_OUTPUT),
            "manifest_hash": registration["manifest_hash"],
        },
        "source_manifest": {
            "path": str(MANIFEST),
            "sha256": sha256(MANIFEST),
            "manifest_hash": source_manifest["manifest_hash"],
        },
        "completed_preentry_sources_opened": True,
        "postentry_return_pnl_execution_price_opened": False,
        "gross9_rows_opened": False,
        "states": {"path": str(STATES), "sha256": sha256(STATES), "rows": len(states)},
        "clock": {"path": str(CLOCK), "sha256": sha256(CLOCK), "rows": len(primary)},
        "controls": {
            name: {
                "path": str(CONTROL_DIR / f"{name}.csv.gz"),
                "sha256": sha256(CONTROL_DIR / f"{name}.csv.gz"),
                "rows": len(value),
                "promotion_authorized": False,
            }
            for name, value in controls.items()
        },
        "support": support,
        "support_checks": checks,
        "support_passed": passed,
        "advance_to_gross9_novelty": passed,
        "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    report = run()
    print(json.dumps({"passed": report["support_passed"], "support": report["support"]}, indent=2))
