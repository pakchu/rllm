"""Materialize source-only DFDPR-12 support clocks."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from training import preregister_dollar_factor_dual_phase_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv

ENV_FILE = "/home/pakchu/rllm/.env"
START = pd.Timestamp("2023-01-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
PREREG_SHA = "7471e1404156ed1008c935daed80e808b6a35790de2817222b0e529ced8498f6"
SOURCE_DIR = Path("data/dollar_factor_dual_phase_relay_sources_2023_2026")
SESSION = SOURCE_DIR / "dollar_factor_dual_phase_sessions.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
PRICE = Path("data/options_oi_chase_exhaustion_sources_2023_2026/btc_completed_hour.csv.gz")
PRICE_SHA = "f075a882b80fc1d050aacd9abd417d4be6b6511c4307e39c98ef25f08822c496"
PRICE_MANIFEST = PRICE.parent / "manifest.json"
PRICE_MANIFEST_SHA = "3e350d16da72da7b60d9e91fbfb1ff4c2e13e5cb954b52b19ceaddf8c4f0e66d"
CLOCK = Path("data/dollar_factor_dual_phase_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/dollar_factor_dual_phase_relay_controls_2023_2026")
RESULT = Path("results/dollar_factor_dual_phase_relay_support_2026-08-09.json")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_persistence_rank",
    "no_variation_gate",
    "early_factor_only",
    "late_factor_only",
    "one_session_stale_phases",
    "direction_flip",
    "same_clock_forced_long",
)
SYMBOLS = prereg.SYMBOLS
PHASES = ("early", "late")
DOLLAR_MULTIPLIER = {
    "EURUSD": -1.0, "GBPUSD": -1.0, "USDAUD": 1.0,
    "USDCAD": 1.0, "USDCHF": 1.0, "USDJPY": 1.0,
}
COLUMNS = (
    "candidate", "control", "split", "source_day", "decision_time",
    "feature_available_time", "entry_time", "exit_time", "side",
    "early_factor", "late_factor", "weaker_phase_magnitude", "persistence_rank",
    "btc_realized_variation", "btc_realized_variation_rank",
)
QUERY = """SELECT date_trunc('day',ts) AS source_day,CASE WHEN extract(hour from ts)<17 THEN 'early' ELSE 'late' END AS phase,(array_agg(open ORDER BY ts))[1] AS phase_open,max(high) AS phase_high,min(low) AS phase_low,(array_agg(close ORDER BY ts DESC))[1] AS phase_close,count(DISTINCT ts) AS distinct_timestamps,min(ts) AS first_ts,max(ts) AS last_ts FROM bars_polygon WHERE symbol=:symbol AND interval='1m' AND ts>=:start AND ts<:end AND extract(isodow from ts) BETWEEN 1 AND 5 AND extract(hour from ts)>=13 AND extract(hour from ts)<21 GROUP BY 1,2 ORDER BY 1,2"""


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def chash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def causal_z(values: pd.Series, lookback: int = 90, minimum: int = 60) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
    output = np.full(len(numeric), np.nan)
    history: list[float] = []
    for index, current in enumerate(numeric):
        prior = history[-lookback:]
        if np.isfinite(current) and len(prior) >= minimum:
            deviation = float(np.std(prior, ddof=1))
            if deviation > 0:
                output[index] = (current - float(np.mean(prior))) / deviation
        if np.isfinite(current):
            history.append(float(current))
    return pd.Series(output, index=values.index)


def strict_prior_midrank(values: pd.Series, lookback: int = 90, minimum: int = 60) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    output = pd.Series(np.nan, index=numeric.index, dtype=float)
    history: list[float] = []
    for index, current in numeric.items():
        prior = history[-lookback:]
        if math.isfinite(current) and len(prior) >= minimum:
            array = np.asarray(prior)
            output.at[index] = (
                np.count_nonzero(array < current) + 0.5 * np.count_nonzero(array == current)
            ) / len(array)
        if math.isfinite(current):
            history.append(float(current))
    return output


def postgres_engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env

    load_env_file(ENV_FILE)
    return create_engine(postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10})


def materialize_sessions() -> dict[str, Any]:
    from sqlalchemy import text

    database = postgres_engine()
    pieces = []
    with database.connect() as connection:
        for symbol in SYMBOLS:
            raw = pd.read_sql_query(
                text(QUERY), connection,
                params={"symbol": symbol, "start": START.to_pydatetime(), "end": END.to_pydatetime()},
            )
            raw["source_day"] = pd.to_datetime(raw.source_day, utc=True)
            raw["first_ts"] = pd.to_datetime(raw.first_ts, utc=True)
            raw["last_ts"] = pd.to_datetime(raw.last_ts, utc=True)
            for column in ("phase_open", "phase_high", "phase_low", "phase_close"):
                raw[column] = pd.to_numeric(raw[column], errors="coerce")
            phase_start = raw.source_day + pd.to_timedelta(np.where(raw.phase.eq("early"), 13, 17), unit="h")
            phase_end = raw.source_day + pd.to_timedelta(np.where(raw.phase.eq("early"), 17, 21), unit="h")
            prices = raw[["phase_open", "phase_high", "phase_low", "phase_close"]]
            valid = (
                raw.phase.isin(PHASES) & raw.distinct_timestamps.ge(225)
                & raw.first_ts.le(phase_start + pd.Timedelta(minutes=5))
                & raw.last_ts.ge(phase_end - pd.Timedelta(minutes=5))
                & np.isfinite(prices).all(axis=1) & prices.gt(0).all(axis=1)
                & raw.phase_high.ge(raw[["phase_open", "phase_close"]].max(axis=1))
                & raw.phase_low.le(raw[["phase_open", "phase_close"]].min(axis=1))
            )
            canonical = (DOLLAR_MULTIPLIER[symbol] * np.log(raw.phase_close / raw.phase_open)).where(valid)
            piece = pd.DataFrame({"source_day": raw.source_day, "phase": raw.phase, "valid": valid, "return": canonical})
            for phase in PHASES:
                mask = piece.phase.eq(phase)
                piece.loc[mask, "z"] = causal_z(piece.loc[mask, "return"])
            wide = piece.pivot(index="source_day", columns="phase", values=["valid", "return", "z"])
            wide.columns = [f"{symbol}_{phase}_{metric}" for metric, phase in wide.columns]
            pieces.append(wide.reset_index())
    database.dispose()
    frame = pieces[0]
    for piece in pieces[1:]:
        frame = frame.merge(piece, on="source_day", how="outer", validate="one_to_one")
    frame = frame.sort_values("source_day").reset_index(drop=True)
    valid_columns = [f"{symbol}_{phase}_valid" for symbol in SYMBOLS for phase in PHASES]
    frame["source_valid"] = frame[valid_columns].eq(True).all(axis=1)  # noqa: E712
    for phase in PHASES:
        z_columns = [f"{symbol}_{phase}_z" for symbol in SYMBOLS]
        frame[f"{phase}_factor"] = frame[z_columns].astype(float).median(axis=1).where(frame.source_valid)
        frame[f"{phase}_absolute_rank"] = strict_prior_midrank(frame[f"{phase}_factor"].abs())
    frame["persistent"] = (
        frame.source_valid & frame.early_factor.ne(0) & frame.late_factor.ne(0)
        & np.sign(frame.early_factor).eq(np.sign(frame.late_factor))
    )
    frame["weaker_phase_magnitude"] = frame[["early_factor", "late_factor"]].abs().min(axis=1).where(frame.persistent)
    frame["persistence_rank"] = strict_prior_midrank(frame.weaker_phase_magnitude)
    frame["decision_time"] = frame.source_day + pd.Timedelta(hours=21)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(frame, SESSION)
    core = {
        "protocol_version": "dfdpr_12_dual_phase_fx_source_v1", "query": QUERY,
        "table": "bars_polygon", "symbols": list(SYMBOLS), "phases_utc": {"early": [13, 17], "late": [17, 21]},
        "canonical_dollar_multipliers": DOLLAR_MULTIPLIER, "interval": "1m",
        "window": [START.isoformat(), END.isoformat()], "outcomes_opened": False,
        "candidate_incidence_opened": False, "no_imputation": True,
        "output": {"path": str(SESSION), "sha256": sha(SESSION), "rows": len(frame), "valid_rows": int(frame.source_valid.sum())},
    }
    result = {**core, "manifest_hash": chash(core)}
    SOURCE_MANIFEST.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def features() -> pd.DataFrame:
    if sha(PRICE) != PRICE_SHA or sha(PRICE_MANIFEST) != PRICE_MANIFEST_SHA:
        raise RuntimeError("DFDPR BTC source drift")
    frame = pd.read_csv(SESSION, compression="gzip")
    frame["source_day"] = pd.to_datetime(frame.source_day, utc=True)
    frame["decision_time"] = pd.to_datetime(frame.decision_time, utc=True)
    for column in ("source_valid", "persistent"):
        frame[column] = frame[column].astype(str).str.lower().eq("true")
    factor_columns = (
        "early_factor", "late_factor", "early_absolute_rank", "late_absolute_rank",
        "weaker_phase_magnitude", "persistence_rank",
    )
    for column in factor_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    price = pd.read_csv(PRICE, compression="gzip")
    price["decision_time"] = pd.to_datetime(price.decision_time, utc=True, format="mixed")
    price["open"] = pd.to_numeric(price.open, errors="coerce")
    price["close"] = pd.to_numeric(price.close, errors="coerce")
    price["valid"] = (
        price.source_valid.astype(str).str.lower().eq("true")
        & np.isfinite(price[["open", "close"]]).all(axis=1) & price[["open", "close"]].gt(0).all(axis=1)
    )
    price = price.sort_values("decision_time").reset_index(drop=True)
    price["hour_return"] = np.log(price.close / price.open)
    consecutive = price.decision_time.diff().eq(pd.Timedelta(hours=1))
    price["btc_realized_variation"] = np.sqrt(price.hour_return.pow(2).rolling(24, min_periods=24).sum())
    price["btc_valid"] = (
        price.valid.rolling(24, min_periods=24).sum().eq(24)
        & consecutive.rolling(23, min_periods=23).sum().eq(23)
        & np.isfinite(price.btc_realized_variation)
    )
    frame = frame.merge(
        price[["decision_time", "btc_realized_variation", "btc_valid"]],
        on="decision_time", how="left", validate="one_to_one",
    )
    frame["btc_realized_variation_rank"] = strict_prior_midrank(
        frame.btc_realized_variation.where(frame.btc_valid)
    )
    frame["signal_valid"] = (
        frame.source_valid & frame.btc_valid.fillna(False)
        & np.isfinite(frame[[*factor_columns, "btc_realized_variation", "btc_realized_variation_rank"]]).all(axis=1)
    )
    return frame


def conditions(frame: pd.DataFrame, control: str = "primary") -> tuple[pd.Series, pd.Series]:
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    used = frame.shift(1) if control == "one_session_stale_phases" else frame
    if control == "early_factor_only":
        active = (
            frame.btc_valid.fillna(False) & used.source_valid.fillna(False)
            & np.isfinite(used[["early_factor", "early_absolute_rank"]]).all(axis=1)
            & used.early_factor.ne(0) & used.early_absolute_rank.ge(0.60)
        )
        factor = used.early_factor
    elif control == "late_factor_only":
        active = (
            frame.btc_valid.fillna(False) & used.source_valid.fillna(False)
            & np.isfinite(used[["late_factor", "late_absolute_rank"]]).all(axis=1)
            & used.late_factor.ne(0) & used.late_absolute_rank.ge(0.60)
        )
        factor = used.late_factor
    else:
        rank_gate = pd.Series(True, index=frame.index) if control == "no_persistence_rank" else used.persistence_rank.ge(0.60)
        active = used.signal_valid.fillna(False) & used.persistent.fillna(False) & rank_gate
        factor = used.early_factor
    variation_gate = (
        pd.Series(True, index=frame.index)
        if control == "no_variation_gate" else frame.btc_realized_variation_rank.ge(0.65)
    )
    active &= variation_gate
    side = -np.sign(factor)
    if control == "direction_flip":
        side = -side
    elif control == "same_clock_forced_long":
        side = pd.Series(1.0, index=frame.index)
    return active, side


def clock(frame: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    active, sides = conditions(frame, control)
    rows = []
    next_allowed = None
    for index in frame.index[active]:
        decision = pd.Timestamp(frame.at[index, "decision_time"])
        entry = decision + pd.Timedelta(minutes=5)
        exit_time = entry + pd.Timedelta(hours=12)
        if next_allowed is not None and entry < next_allowed:
            continue
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end), None)
        if split is None:
            continue
        next_allowed = exit_time
        rows.append({
            "candidate": "DFDPR-12", "control": control, "split": split,
            "source_day": frame.at[index, "source_day"], "decision_time": decision,
            "feature_available_time": decision, "entry_time": entry, "exit_time": exit_time,
            "side": int(sides.at[index]), "early_factor": float(frame.at[index, "early_factor"]),
            "late_factor": float(frame.at[index, "late_factor"]),
            "weaker_phase_magnitude": float(frame.at[index, "weaker_phase_magnitude"]),
            "persistence_rank": float(frame.at[index, "persistence_rank"]),
            "btc_realized_variation": float(frame.at[index, "btc_realized_variation"]),
            "btc_realized_variation_rank": float(frame.at[index, "btc_realized_variation_rank"]),
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def stats(candidate: pd.DataFrame, split: str) -> dict[str, int | float]:
    selected = candidate[candidate.split.eq(split)]
    if selected.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs, shorts = int(selected.side.eq(1).sum()), int(selected.side.eq(-1).sum())
    months = pd.to_datetime(selected.entry_time, utc=True).dt.strftime("%Y-%m").value_counts()
    return {"events": len(selected), "longs": longs, "shorts": shorts,
            "minority_side_share": min(longs, shorts) / len(selected),
            "max_month_share": int(months.max()) / len(selected)}


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("DFDPR preregistration hash drift")
    source_manifest = materialize_sessions()
    frame = features()
    primary = clock(frame)
    controls = {name: clock(frame, name) for name in CONTROLS}
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(primary, CLOCK)
    for name, candidate in controls.items():
        _write_gzip_csv(candidate, CONTROL_DIR / f"{name}.csv.gz")
    support = {name: stats(primary, name) for name in SPLITS}
    checks = {}
    for name, values in support.items():
        checks[f"{name}_minimum_events"] = values["events"] >= MINIMUM[name]
        checks[f"{name}_side_balance"] = values["minority_side_share"] >= 0.2
        checks[f"{name}_month_concentration"] = values["max_month_share"] <= 0.45
    passed = all(checks.values())
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    core = {
        "protocol_version": "dfdpr_12_source_support_v1", "policy_id": "DFDPR-12",
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]},
        "source_manifests": {
            "dual_phase_fx": {"path": str(SOURCE_MANIFEST), "sha256": sha(SOURCE_MANIFEST), "manifest_hash": source_manifest["manifest_hash"]},
            "completed_btc": {"path": str(PRICE_MANIFEST), "sha256": PRICE_MANIFEST_SHA},
        },
        "completed_preentry_sources_opened": True,
        "postentry_return_pnl_execution_price_opened": False, "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(candidate), "promotion_authorized": False} for name, candidate in controls.items()},
        "support": support, "support_checks": checks, "support_passed": passed,
        "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": chash(core)}
    RESULT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    result = run()
    print(json.dumps({"passed": result["support_passed"], "support": result["support"]}, indent=2))
