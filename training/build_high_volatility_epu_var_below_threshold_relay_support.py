"""Materialize outcome-blind source support for frozen HVEPUVBT-24."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import preregister_high_volatility_epu_var_below_threshold_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


PREREG_SHA256 = "80abf25850fccab22768cecf1b5ea02bf179311e6ea948d778ffd88be771f56e"
ENV_FILE = "/home/pakchu/rllm/.env"
START = pd.Timestamp("2020-01-03T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
SPLITS = {
    name: tuple(pd.Timestamp(value) for value in bounds)
    for name, bounds in prereg.build()["stages"].items()
}
GATES = prereg.build()["source_support_gates"]
CONTROLS = tuple(prereg.build()["diagnostic_controls"]["names"])
SOURCE_DIR = Path("data/high_volatility_epu_var_below_threshold_relay_sources_2020_2026")
BTC_SOURCE = SOURCE_DIR / "btc_daily_preentry_states.csv.gz"
FEATURE_PANEL = SOURCE_DIR / "daily_feature_panel.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/high_volatility_epu_var_below_threshold_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_epu_var_below_threshold_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_epu_var_below_threshold_relay_support_2026-08-13.json")
BTC_QUERY = """SELECT ts,open,close
FROM bars_binance
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
ORDER BY ts"""
PANEL_COLUMNS = (
    "decision_time", "source_day", "epu", "epu_change", "btc_open", "btc_return",
    "btc_variation", "btc_variation_rank", "btc_forecast", "forecast_dispersion",
    "state_valid",
)
CLOCK_COLUMNS = (
    "candidate", "control", "split", "source_day", "decision_time",
    "feature_available_time", "entry_time", "exit_time", "side", "btc_forecast",
    "forecast_dispersion", "below_threshold", "epu_change", "btc_return",
    "btc_variation", "btc_variation_rank",
)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_prior_midrank(
    values: pd.Series, *, lookback: int = 270, minimum: int = 180
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    output = pd.Series(np.nan, index=numeric.index, dtype=float)
    history: list[float] = []
    for index, current in numeric.items():
        prior = np.asarray(history[-lookback:], dtype=float)
        if np.isfinite(current) and len(prior) >= minimum:
            output.at[index] = (
                np.count_nonzero(prior < current)
                + 0.5 * np.count_nonzero(prior == current)
            ) / len(prior)
        if np.isfinite(current):
            history.append(float(current))
    return output


def expanding_var_forecasts(
    frame: pd.DataFrame, *, minimum: int = 730
) -> pd.Series:
    values = frame[["btc_return", "epu_change"]].to_numpy(dtype=float)
    forecasts = pd.Series(np.nan, index=frame.index, dtype=float)
    for index in range(1, len(frame)):
        dependent = values[1 : index + 1]
        lagged = values[:index]
        valid = np.isfinite(dependent).all(axis=1) & np.isfinite(lagged).all(axis=1)
        if np.count_nonzero(valid) < minimum or not np.isfinite(values[index]).all():
            continue
        design = np.column_stack([np.ones(np.count_nonzero(valid)), lagged[valid]])
        coefficients, _, rank, _ = np.linalg.lstsq(design, dependent[valid], rcond=None)
        if rank == 3:
            current = np.array([1.0, values[index, 0], values[index, 1]])
            forecasts.at[index] = float(current @ coefficients[:, 0])
    return forecasts


def expanding_forecast_dispersion(
    forecasts: pd.Series, *, minimum: int = 365
) -> pd.Series:
    numeric = pd.to_numeric(forecasts, errors="coerce")
    output = pd.Series(np.nan, index=numeric.index, dtype=float)
    history: list[float] = []
    for index, current in numeric.items():
        if np.isfinite(current) and len(history) >= minimum:
            output.at[index] = float(np.std(np.asarray(history), ddof=1))
        if np.isfinite(current):
            history.append(float(current))
    return output


def normalize_btc_minutes(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.columns.tolist() != ["ts", "open", "close"]:
        raise RuntimeError("HVEPUVBT BTC schema drift")
    frame = raw.copy()
    frame.ts = pd.to_datetime(frame.ts, utc=True, errors="raise")
    frame = frame.sort_values("ts").reset_index(drop=True)
    expected = pd.date_range(START, END, freq="1min", inclusive="left")
    if len(frame) != len(expected) or not frame.ts.equals(pd.Series(expected, name="ts")):
        raise RuntimeError("HVEPUVBT BTC source is not the exact requested 1m grid")
    for column in ("open", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    prices = frame[["open", "close"]]
    if not np.isfinite(prices.to_numpy(dtype=float)).all() or not prices.gt(0).all(axis=None):
        raise RuntimeError("HVEPUVBT BTC prices invalid")
    return frame


def load_btc_minutes(env_file: str = ENV_FILE) -> pd.DataFrame:
    from sqlalchemy import create_engine, text

    from preprocessing.live_db_features import load_env_file, postgres_url_from_env

    load_env_file(env_file)
    engine = create_engine(postgres_url_from_env(env_file), connect_args={"connect_timeout": 10})
    try:
        raw = pd.read_sql_query(
            text(BTC_QUERY),
            engine,
            params={"start": START.to_pydatetime(), "end": END.to_pydatetime()},
        )
    finally:
        engine.dispose()
    return normalize_btc_minutes(raw)


def daily_btc_states(minutes: pd.DataFrame) -> pd.DataFrame:
    indexed = minutes.set_index("ts")
    decisions = pd.date_range(START.ceil("1d"), END, freq="1d", inclusive="left")
    rows: list[dict[str, Any]] = []
    prior_open: float | None = None
    for decision in decisions:
        btc_open = float(indexed.at[decision, "open"])
        window = indexed.loc[
            (indexed.index >= decision - pd.Timedelta(days=1)) & (indexed.index < decision)
        ]
        variation = np.nan
        if len(window) == 1440:
            component = np.log(
                window.close.to_numpy(dtype=float) / window.open.to_numpy(dtype=float)
            )
            variation = float(np.sqrt(np.square(component).sum()))
        rows.append(
            {
                "decision_time": decision,
                "btc_open": btc_open,
                "btc_return": (
                    np.log(btc_open / prior_open) if prior_open is not None else np.nan
                ),
                "btc_variation": variation,
                "minute_count": len(window),
            }
        )
        prior_open = btc_open
    return pd.DataFrame(rows)


def normalize_epu(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.columns.tolist() != ["day", "month", "year", "daily_policy_index"]:
        raise RuntimeError("HVEPUVBT EPU schema drift")
    frame = raw.copy()
    frame["source_day"] = pd.to_datetime(
        dict(year=frame.year, month=frame.month, day=frame.day), utc=True, errors="raise"
    )
    frame["epu"] = pd.to_numeric(frame.daily_policy_index, errors="coerce")
    frame = frame.sort_values("source_day").reset_index(drop=True)
    if frame.source_day.duplicated().any():
        raise RuntimeError("HVEPUVBT duplicate EPU source day")
    if not np.isfinite(frame.epu.to_numpy(dtype=float)).all() or not frame.epu.gt(0).all():
        raise RuntimeError("HVEPUVBT EPU values invalid")
    if not frame.source_day.diff().iloc[1:].eq(pd.Timedelta(days=1)).all():
        raise RuntimeError("HVEPUVBT EPU source is not consecutive daily")
    frame["epu_change"] = frame.epu.diff()
    frame["decision_time"] = frame.source_day + pd.Timedelta(days=2)
    return frame[["decision_time", "source_day", "epu", "epu_change"]]


def build_features(epu: pd.DataFrame, btc: pd.DataFrame) -> pd.DataFrame:
    frame = btc.merge(epu, on="decision_time", how="left", validate="one_to_one")
    frame = frame.loc[frame.decision_time.ge(pd.Timestamp("2020-01-04T00:00:00Z"))].copy()
    frame = frame.sort_values("decision_time").reset_index(drop=True)
    base_valid = np.isfinite(
        frame[["epu", "epu_change", "btc_open", "btc_return", "btc_variation"]].to_numpy(
            dtype=float
        )
    ).all(axis=1)
    frame["btc_variation_rank"] = strict_prior_midrank(frame.btc_variation.where(base_valid))
    frame["btc_forecast"] = expanding_var_forecasts(frame)
    frame["forecast_dispersion"] = expanding_forecast_dispersion(frame.btc_forecast)
    frame["state_valid"] = (
        base_valid
        & np.isfinite(frame[["btc_variation_rank", "btc_forecast", "forecast_dispersion"]]).all(
            axis=1
        )
        & frame.btc_forecast.ne(0.0)
        & frame.forecast_dispersion.gt(0.0)
    )
    return frame[list(PANEL_COLUMNS)]


def _ar_forecasts(frame: pd.DataFrame, minimum: int = 730) -> pd.Series:
    values = frame.btc_return.to_numpy(dtype=float)
    output = pd.Series(np.nan, index=frame.index, dtype=float)
    for index in range(1, len(frame)):
        dependent = values[1 : index + 1]
        lagged = values[:index]
        valid = np.isfinite(dependent) & np.isfinite(lagged)
        if np.count_nonzero(valid) < minimum or not np.isfinite(values[index]):
            continue
        design = np.column_stack([np.ones(np.count_nonzero(valid)), lagged[valid]])
        coefficients, _, rank, _ = np.linalg.lstsq(design, dependent[valid], rcond=None)
        if rank == 2:
            output.at[index] = float(np.array([1.0, values[index]]) @ coefficients)
    return output


def signal(frame: pd.DataFrame, control: str = "primary") -> tuple[pd.Series, pd.Series, pd.Series]:
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    forecast = frame.btc_forecast.copy()
    dispersion = frame.forecast_dispersion.copy()
    state_valid = frame.state_valid.copy()
    if control == "one_day_stale_forecast":
        forecast = forecast.shift(1)
        dispersion = dispersion.shift(1)
        state_valid &= forecast.notna() & dispersion.notna()
    elif control == "btc_ar_only":
        forecast = _ar_forecasts(frame)
        dispersion = expanding_forecast_dispersion(forecast)
        state_valid = (
            np.isfinite(frame[["btc_variation_rank", "btc_return"]]).all(axis=1)
            & forecast.notna()
            & forecast.ne(0.0)
            & dispersion.gt(0.0)
        )
    below = forecast.abs().le(dispersion)
    side = pd.Series(np.where(below, np.sign(forecast), 1), index=frame.index, dtype=int)
    if control == "always_forecast_sign":
        side = np.sign(forecast).fillna(0).astype(int)
    active = state_valid & forecast.ne(0.0)
    if control != "no_btc_variation_gate":
        active &= frame.btc_variation_rank.ge(0.65)
    if control == "direction_flip":
        side = -side
    elif control == "same_clock_forced_long":
        side = pd.Series(1, index=frame.index, dtype=int)
    active &= side.ne(0)
    return active.fillna(False), side, below.fillna(False)


def build_clock(frame: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    active, side, below = signal(frame, control)
    rows: list[dict[str, Any]] = []
    for index in frame.index[active]:
        decision = pd.Timestamp(frame.at[index, "decision_time"])
        entry = decision + pd.Timedelta(minutes=5)
        exit_time = entry + pd.Timedelta(hours=24)
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
        rows.append(
            {
                "candidate": prereg.POLICY_ID,
                "control": control,
                "split": split,
                "source_day": frame.at[index, "source_day"],
                "decision_time": decision,
                "feature_available_time": decision,
                "entry_time": entry,
                "exit_time": exit_time,
                "side": int(side.at[index]),
                "btc_forecast": float(frame.at[index, "btc_forecast"]),
                "forecast_dispersion": float(frame.at[index, "forecast_dispersion"]),
                "below_threshold": bool(below.at[index]),
                "epu_change": float(frame.at[index, "epu_change"]),
                "btc_return": float(frame.at[index, "btc_return"]),
                "btc_variation": float(frame.at[index, "btc_variation"]),
                "btc_variation_rank": float(frame.at[index, "btc_variation_rank"]),
            }
        )
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def split_stats(clock: pd.DataFrame, split: str) -> dict[str, Any]:
    subset = clock.loc[clock.split.eq(split)].copy()
    if subset.empty:
        return {
            "events": 0,
            "longs": 0,
            "shorts": 0,
            "minority_side_share": 0.0,
            "max_month_share": 0.0,
        }
    longs = int(subset.side.eq(1).sum())
    shorts = int(subset.side.eq(-1).sum())
    months = pd.to_datetime(subset.entry_time, utc=True).dt.strftime("%Y-%m").value_counts()
    return {
        "events": int(len(subset)),
        "longs": longs,
        "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(subset),
        "max_month_share": int(months.max()) / len(subset),
    }


def run() -> dict[str, Any]:
    bindings = {
        prereg.DEFAULT_OUTPUT: PREREG_SHA256,
        prereg.EPU_SOURCE: prereg.EPU_SOURCE_SHA256,
        prereg.MARKET: prereg.MARKET_SHA256,
    }
    for path, expected in bindings.items():
        if sha256(path) != expected:
            raise RuntimeError(f"HVEPUVBT binding drift: {path}")
    minutes = load_btc_minutes()
    btc = daily_btc_states(minutes)
    epu = normalize_epu(pd.read_csv(prereg.EPU_SOURCE))
    features = build_features(epu, btc)
    primary = build_clock(features)
    controls = {name: build_clock(features, name) for name in CONTROLS}
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(btc, BTC_SOURCE)
    _write_gzip_csv(features, FEATURE_PANEL)
    _write_gzip_csv(primary, CLOCK)
    for name, control in controls.items():
        _write_gzip_csv(control, CONTROL_DIR / f"{name}.csv.gz")
    source_core = {
        "protocol_version": "hvepuvbt_preentry_source_manifest_v1",
        "query": BTC_QUERY,
        "query_start": START.isoformat(),
        "query_end_exclusive": END.isoformat(),
        "btc_daily_states": {"path": str(BTC_SOURCE), "sha256": sha256(BTC_SOURCE), "rows": len(btc)},
        "feature_panel": {"path": str(FEATURE_PANEL), "sha256": sha256(FEATURE_PANEL), "rows": len(features)},
        "epu_snapshot": {"path": str(prereg.EPU_SOURCE), "sha256": prereg.EPU_SOURCE_SHA256},
        "postentry_return_pnl_execution_price_opened": False,
    }
    source_manifest = {**source_core, "manifest_hash": prereg.canonical_hash(source_core)}
    SOURCE_MANIFEST.write_text(
        json.dumps(source_manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    support = {name: split_stats(primary, name) for name in SPLITS}
    checks = {
        key: value
        for name, stats in support.items()
        for key, value in (
            (f"{name}_minimum_events", stats["events"] >= GATES["minimum_events"][name]),
            (f"{name}_side_balance", stats["minority_side_share"] >= GATES["minority_side_share_min"]),
            (f"{name}_month_concentration", stats["max_month_share"] <= GATES["max_month_share"]),
        )
    }
    passed = all(checks.values())
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    core = {
        "protocol_version": "hvepuvbt_24_source_support_v1",
        "policy_id": prereg.POLICY_ID,
        "preregistration": {
            "path": str(prereg.DEFAULT_OUTPUT),
            "sha256": PREREG_SHA256,
            "manifest_hash": registration["manifest_hash"],
        },
        "bindings": {str(path): digest for path, digest in bindings.items()},
        "completed_preentry_sources_opened": True,
        "postentry_return_pnl_execution_price_opened": False,
        "gross9_rows_opened": False,
        "source_manifest": {"path": str(SOURCE_MANIFEST), "sha256": sha256(SOURCE_MANIFEST)},
        "clock": {"path": str(CLOCK), "sha256": sha256(CLOCK), "rows": len(primary)},
        "controls": {
            name: {
                "path": str(CONTROL_DIR / f"{name}.csv.gz"),
                "sha256": sha256(CONTROL_DIR / f"{name}.csv.gz"),
                "rows": len(control),
                "promotion_authorized": False,
            }
            for name, control in controls.items()
        },
        "support": support,
        "support_checks": checks,
        "support_passed": passed,
        "advance_to_gross9_novelty": passed,
        "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": prereg.canonical_hash(core)}
    RESULT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    result = run()
    print(json.dumps({"passed": result["support_passed"], "support": result["support"]}, indent=2))
