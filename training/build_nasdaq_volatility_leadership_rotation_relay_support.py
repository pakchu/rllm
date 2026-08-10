"""Materialize outcome-blind source support for frozen NVLRR-12."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from training import preregister_nasdaq_volatility_leadership_rotation_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


ENV_FILE = "/home/pakchu/rllm/.env"
BUILDER = Path("training/build_nasdaq_volatility_leadership_rotation_relay_support.py")
PREREG_SHA = "c091adf520e5f9b578922ef79f7a4c523da9b0964884bc523afbbe4a90ed4e6d"
SOURCE_DIR = Path("data/nasdaq_volatility_leadership_rotation_relay_sources_2021_2026")
FEATURE_PANEL = SOURCE_DIR / "nasdaq_volatility_leadership_rotation_relay_preentry_features.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/nasdaq_volatility_leadership_rotation_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/nasdaq_volatility_leadership_rotation_relay_controls_2023_2026")
RESULT = Path("results/nasdaq_volatility_leadership_rotation_relay_support_2026-08-10.json")
NY = ZoneInfo("America/New_York")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM_EVENTS = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_btc_variation_gate",
    "no_leadership_tail",
    "vxn_minus_vix_raw",
    "one_session_stale_leadership",
    "direction_flip",
    "forced_long",
)
ECONOMIC_OUTCOMES_AUTHORIZED = False
CHANGE_COLUMNS = ("vix_change", "vxn_change", "gvz_change", "ovx_change")
CLOCK_COLUMNS = (
    "candidate", "control", "split", "residual_source_date", "next_common_source_date",
    "decision_time", "feature_available_time", "entry_time", "exit_time", "side",
    *CHANGE_COLUMNS, "z_vix", "z_vxn", "z_gvz", "z_ovx", "btc_realized_variation",
    "btc_variation_rank", "leadership_residual", "absolute_leadership_rank",
    "vxn_minus_vix_raw", "absolute_vxn_minus_vix_raw_rank",
)
BTC_QUERY = """
SELECT ts,open,close
FROM bars_binance
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
ORDER BY ts
""".strip()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def strict_prior_midrank(
    values: pd.Series, lookback: int = 252, minimum: int = 126
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    output = pd.Series(np.nan, index=numeric.index, dtype=float)
    finite_history: list[float] = []
    for index, current in numeric.items():
        history = finite_history[-lookback:]
        if np.isfinite(current) and len(history) >= minimum:
            array = np.asarray(history, dtype=float)
            output.at[index] = (
                np.count_nonzero(array < current)
                + 0.5 * np.count_nonzero(array == current)
            ) / len(array)
        if np.isfinite(current):
            finite_history.append(float(current))
    return output


def causal_z(
    values: pd.Series, lookback: int = 252, minimum: int = 126
) -> pd.Series:
    """Standardize each finite value against finite strictly prior observations."""
    numeric = pd.to_numeric(values, errors="coerce")
    output = pd.Series(np.nan, index=numeric.index, dtype=float)
    finite_history: list[float] = []
    for index, current in numeric.items():
        history = finite_history[-lookback:]
        if np.isfinite(current) and len(history) >= minimum:
            array = np.asarray(history, dtype=float)
            deviation = float(np.std(array, ddof=1))
            if np.isfinite(deviation) and deviation > 0:
                output.at[index] = (float(current) - float(np.mean(array))) / deviation
        if np.isfinite(current):
            finite_history.append(float(current))
    return output


def _validate_daily(frame: pd.DataFrame, value_column: str, label: str) -> pd.DataFrame:
    result = frame[["observation_date", value_column]].copy()
    result["observation_date"] = pd.to_datetime(result.observation_date, errors="raise")
    result[value_column] = pd.to_numeric(result[value_column], errors="coerce")
    result = result.sort_values("observation_date").reset_index(drop=True)
    if result.empty or result.observation_date.duplicated().any():
        raise RuntimeError(f"NVLRR {label} dates invalid")
    values = result[value_column]
    if not np.isfinite(values).all() or not values.gt(0).all():
        raise RuntimeError(f"NVLRR {label} values invalid")
    return result


def combine_cboe_sources(
    vix: pd.DataFrame, vxn: pd.DataFrame, gvz: pd.DataFrame, ovx: pd.DataFrame
) -> pd.DataFrame:
    """Return the exact positive finite four-index date intersection."""
    inputs = (
        _validate_daily(vix, "VIX_close", "VIX"),
        _validate_daily(vxn, "VXN_close", "VXN"),
        _validate_daily(gvz, "GVZ_close", "GVZ"),
        _validate_daily(ovx, "OVX_close", "OVX"),
    )
    common = inputs[0]
    for other in inputs[1:]:
        common = common.merge(other, on="observation_date", how="inner", validate="one_to_one")
    common = common.sort_values("observation_date").reset_index(drop=True)
    if len(common) < 3:
        raise RuntimeError("NVLRR common Cboe intersection is insufficient")
    for symbol in ("VIX", "VXN", "GVZ", "OVX"):
        common[f"{symbol.lower()}_change"] = np.log(common[f"{symbol}_close"]).diff()
    return common


def load_cboe() -> pd.DataFrame:
    sources = prereg.SOURCES
    for label, specification in sources.items():
        path = Path(specification["path"])
        if sha(path) != specification["sha256"]:
            raise RuntimeError(f"NVLRR {label} source hash drift")
    vix = pd.read_csv(
        Path(sources["vix_panel"]["path"]), compression="gzip",
        usecols=["observation_date", "VIX_close"],
    )
    raw_specs = {
        "vxn": ("CLOSE", "VXN_close"),
        "gvz": ("GVZ", "GVZ_close"),
        "ovx": ("OVX", "OVX_close"),
    }
    parsed: dict[str, pd.DataFrame] = {}
    for label, (raw_value, output_value) in raw_specs.items():
        raw = pd.read_csv(Path(sources[label]["path"]), usecols=["DATE", raw_value])
        raw = raw.rename(columns={"DATE": "observation_date", raw_value: output_value})
        parsed[label] = raw
    return combine_cboe_sources(vix, parsed["vxn"], parsed["gvz"], parsed["ovx"])


def postgres_engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env

    load_env_file(ENV_FILE)
    return create_engine(postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10})


def load_bars(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    from sqlalchemy import text

    engine = postgres_engine()
    try:
        frame = pd.read_sql_query(
            text(BTC_QUERY), engine,
            params={"start": start.to_pydatetime(), "end": end.to_pydatetime()},
        )
    finally:
        engine.dispose()
    if frame.columns.tolist() != ["ts", "open", "close"]:
        raise RuntimeError("NVLRR BTC schema must be exactly ts,open,close")
    frame["ts"] = pd.to_datetime(frame.ts, utc=True, errors="raise")
    frame = frame.sort_values("ts").reset_index(drop=True)
    expected = pd.date_range(start, end, freq="1min", inclusive="left")
    if len(frame) != len(expected) or not frame.ts.equals(pd.Series(expected, name="ts")):
        raise RuntimeError("NVLRR BTC source is not the exact requested 1m grid")
    for column in ("open", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    prices = frame[["open", "close"]]
    if not np.isfinite(prices).all(axis=None) or not prices.gt(0).all(axis=None):
        raise RuntimeError("NVLRR BTC source contains invalid prices")
    return frame.set_index("ts")


def _decision_time(source_date: pd.Timestamp) -> pd.Timestamp:
    local = pd.Timestamp(source_date).normalize().tz_localize(NY)
    return (local + pd.Timedelta(hours=9, minutes=30)).tz_convert("UTC")


def _variation(bars: pd.DataFrame, decision: pd.Timestamp) -> float:
    start = decision - pd.Timedelta(hours=24)
    expected = pd.date_range(start, decision, freq="1min", inclusive="left")
    window = bars.loc[(bars.index >= start) & (bars.index < decision)]
    if len(window) != 1440 or not window.index.equals(expected):
        return np.nan
    returns = np.log(window.close.to_numpy(dtype=float) / window.open.to_numpy(dtype=float))
    return float(np.sqrt(np.square(returns).sum()))


def build_features(
    cboe: pd.DataFrame,
    bars: pd.DataFrame,
    rank_lookback: int = 252,
    rank_minimum: int = 126,
    zscore_lookback: int | None = None,
    zscore_minimum: int | None = None,
) -> pd.DataFrame:
    zscore_lookback = rank_lookback if zscore_lookback is None else zscore_lookback
    zscore_minimum = rank_minimum if zscore_minimum is None else zscore_minimum
    enriched = cboe.copy()
    for column in CHANGE_COLUMNS:
        symbol = column.removesuffix("_change")
        enriched[f"z_{symbol}"] = causal_z(
            enriched[column], lookback=zscore_lookback, minimum=zscore_minimum
        )
    records: list[dict[str, Any]] = []
    for current in range(1, len(enriched) - 1):
        next_date = pd.Timestamp(enriched.at[current + 1, "observation_date"])
        decision = _decision_time(next_date)
        changes = {column: float(enriched.at[current, column]) for column in CHANGE_COLUMNS}
        zscores = {
            f"z_{column.removesuffix('_change')}": float(enriched.at[current, f"z_{column.removesuffix('_change')}"])
            for column in CHANGE_COLUMNS
        }
        records.append(
            {
                "residual_source_date": pd.Timestamp(enriched.at[current, "observation_date"]),
                "next_common_source_date": next_date,
                "decision_time": decision,
                **changes,
                **zscores,
                "btc_realized_variation": _variation(bars, decision),
            }
        )
    frame = pd.DataFrame(records)
    frame["btc_variation_rank"] = strict_prior_midrank(
        frame.btc_realized_variation, lookback=rank_lookback, minimum=rank_minimum
    )
    frame["leadership_residual"] = frame.z_vxn - frame[["z_vix", "z_gvz", "z_ovx"]].median(
        axis=1, skipna=False
    )
    frame["absolute_leadership_rank"] = strict_prior_midrank(
        frame.leadership_residual.abs(), lookback=rank_lookback, minimum=rank_minimum
    )
    frame["vxn_minus_vix_raw"] = frame.vxn_change - frame.vix_change
    frame["absolute_vxn_minus_vix_raw_rank"] = strict_prior_midrank(
        frame.vxn_minus_vix_raw.abs(), lookback=rank_lookback, minimum=rank_minimum
    )
    return frame


def conditions(
    frame: pd.DataFrame, control: str = "primary"
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    used = frame.shift(1) if control == "one_session_stale_leadership" else frame
    if control == "vxn_minus_vix_raw":
        residual = used.vxn_minus_vix_raw
        leadership_rank = used.absolute_vxn_minus_vix_raw_rank
    else:
        residual = used.leadership_residual
        leadership_rank = used.absolute_leadership_rank
    side = -np.sign(residual).fillna(0).astype(int)
    volatility_gate = (
        pd.Series(True, index=frame.index)
        if control == "no_btc_variation_gate"
        else frame.btc_variation_rank.ge(0.65)
    )
    residual_gate = (
        pd.Series(True, index=frame.index)
        if control == "no_leadership_tail"
        else leadership_rank.ge(0.70)
    )
    active = side.ne(0) & volatility_gate & residual_gate
    if control == "direction_flip":
        side = -side
    elif control == "forced_long":
        side = pd.Series(1, index=frame.index, dtype=int)
    return active, side, used


def build_clock(features: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    active, sides, used = conditions(features, control)
    rows: list[dict[str, Any]] = []
    next_allowed: pd.Timestamp | None = None
    for index in features.index[active]:
        decision = pd.Timestamp(features.at[index, "decision_time"])
        entry = decision + pd.Timedelta(minutes=5)
        exit_time = entry + pd.Timedelta(hours=12)
        if next_allowed is not None and entry < next_allowed:
            continue
        split = next(
            (name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end),
            None,
        )
        if split is None:
            continue
        source = used.loc[index]
        next_allowed = exit_time
        rows.append(
            {
                "candidate": "NVLRR-12", "control": control, "split": split,
                "residual_source_date": pd.Timestamp(source.residual_source_date).date().isoformat(),
                "next_common_source_date": pd.Timestamp(features.at[index, "next_common_source_date"]).date().isoformat(),
                "decision_time": decision, "feature_available_time": decision,
                "entry_time": entry, "exit_time": exit_time, "side": int(sides.at[index]),
                **{column: float(source[column]) for column in CHANGE_COLUMNS},
                **{f"z_{column.removesuffix('_change')}": float(source[f"z_{column.removesuffix('_change')}"]) for column in CHANGE_COLUMNS},
                "btc_realized_variation": float(features.at[index, "btc_realized_variation"]),
                "btc_variation_rank": float(features.at[index, "btc_variation_rank"]),
                "leadership_residual": float(source.leadership_residual),
                "absolute_leadership_rank": float(source.absolute_leadership_rank),
                "vxn_minus_vix_raw": float(source.vxn_minus_vix_raw),
                "absolute_vxn_minus_vix_raw_rank": float(source.absolute_vxn_minus_vix_raw_rank),
            }
        )
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def split_stats(clock: pd.DataFrame, split: str) -> dict[str, int | float]:
    selected = clock[clock.split.eq(split)]
    if selected.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs, shorts = int(selected.side.eq(1).sum()), int(selected.side.eq(-1).sum())
    entries = pd.to_datetime(selected.entry_time, utc=True)
    return {
        "events": len(selected), "longs": longs, "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(selected),
        "max_month_share": int(entries.dt.strftime("%Y-%m").value_counts().max()) / len(selected),
    }


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("NVLRR preregistration hash drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    prereg.validate(registration)
    cboe = load_cboe()
    first_decision = _decision_time(pd.Timestamp(cboe.at[2, "observation_date"]))
    last_decision = _decision_time(pd.Timestamp(cboe.at[len(cboe) - 1, "observation_date"]))
    bars = load_bars(first_decision - pd.Timedelta(hours=24), last_decision)
    features = build_features(cboe, bars)
    primary = build_clock(features)
    controls = {name: build_clock(features, name) for name in CONTROLS}

    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    CLOCK.parent.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(features, FEATURE_PANEL)
    _write_gzip_csv(primary, CLOCK)
    for name, frame in controls.items():
        _write_gzip_csv(frame, CONTROL_DIR / f"{name}.csv.gz")

    raw_sources = {
        name: {
            "path": specification["path"],
            "preregistered_sha256": specification["sha256"],
            "observed_sha256": sha(Path(specification["path"])),
        }
        for name, specification in prereg.SOURCES.items()
    }
    source_core = {
        "protocol_version": "nvlrr_12_sources_v1",
        "raw_sources": raw_sources,
        "btc_source": {
            "table": "bars_binance", "symbol": "BTCUSDT", "interval": "1m",
            "query": BTC_QUERY, "start": (first_decision - pd.Timedelta(hours=24)).isoformat(),
            "end": last_decision.isoformat(), "rows": len(bars), "read_only": True,
        },
        "builder": {"path": str(BUILDER), "sha256": sha(BUILDER)},
        "features": {"path": str(FEATURE_PANEL), "sha256": sha(FEATURE_PANEL), "rows": len(features)},
        "candidate_outcomes_opened": False, "gross9_rows_opened": False, "no_imputation": True,
    }
    source_manifest = {**source_core, "manifest_hash": canonical_hash(source_core)}
    SOURCE_MANIFEST.write_text(json.dumps(source_manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")

    support = {name: split_stats(primary, name) for name in SPLITS}
    checks: dict[str, bool] = {}
    for name, values in support.items():
        checks[f"{name}_minimum_events"] = values["events"] >= MINIMUM_EVENTS[name]
        checks[f"{name}_side_balance"] = values["minority_side_share"] >= 0.20
        checks[f"{name}_month_concentration"] = values["max_month_share"] <= 0.45
    passed = all(checks.values())
    core = {
        "protocol_version": "nvlrr_12_source_support_v1", "policy_id": "NVLRR-12",
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]},
        "source_manifest": {"path": str(SOURCE_MANIFEST), "sha256": sha(SOURCE_MANIFEST), "manifest_hash": source_manifest["manifest_hash"]},
        "completed_preentry_sources_opened": True,
        "postentry_return_pnl_execution_price_opened": False, "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "controls": {
            name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(frame), "promotion_authorized": False}
            for name, frame in controls.items()
        },
        "support": support, "support_checks": checks, "support_passed": passed,
        "advance_to_gross9_novelty": passed,
        "advance_to_economic_outcomes": ECONOMIC_OUTCOMES_AUTHORIZED,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    output = run()
    print(json.dumps({"passed": output["support_passed"], "support": output["support"]}, indent=2))
