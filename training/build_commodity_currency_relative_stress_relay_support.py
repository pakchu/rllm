"""Materialize source-only CCRSR-12 support clocks."""
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

from training import preregister_commodity_currency_relative_stress_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv

ENV_FILE = "/home/pakchu/rllm/.env"
START = pd.Timestamp("2023-01-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
PREREG_SHA = "744f075a2aa5c2920cd4f0f0e9472e31875bd8a3bb1fc57a9f93ed4b0087612b"
SOURCE_DIR = Path("data/commodity_currency_relative_stress_relay_sources_2023_2026")
SESSION = SOURCE_DIR / "commodity_currency_relative_stress_sessions.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
PRICE = Path("data/options_oi_chase_exhaustion_sources_2023_2026/btc_completed_hour.csv.gz")
PRICE_SHA = "f075a882b80fc1d050aacd9abd417d4be6b6511c4307e39c98ef25f08822c496"
PRICE_MANIFEST = PRICE.parent / "manifest.json"
PRICE_MANIFEST_SHA = "3e350d16da72da7b60d9e91fbfb1ff4c2e13e5cb954b52b19ceaddf8c4f0e66d"
CLOCK = Path("data/commodity_currency_relative_stress_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/commodity_currency_relative_stress_relay_controls_2023_2026")
RESULT = Path("results/commodity_currency_relative_stress_relay_support_2026-08-09.json")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_stress_tail",
    "no_variation_gate",
    "usdcad_only",
    "usdaud_only",
    "one_session_stale_stress",
    "direction_flip",
    "same_clock_forced_long",
)
SYMBOLS = ("USDCAD", "USDAUD")
COLUMNS = (
    "candidate",
    "control",
    "split",
    "source_day",
    "decision_time",
    "feature_available_time",
    "entry_time",
    "exit_time",
    "side",
    "usdcad_return",
    "usdaud_return",
    "relative_stress",
    "absolute_stress_rank",
    "btc_realized_variation",
    "btc_realized_variation_rank",
)
QUERY = """SELECT date_trunc('day',ts) AS source_day,(array_agg(open ORDER BY ts))[1] AS session_open,max(high) AS session_high,min(low) AS session_low,(array_agg(close ORDER BY ts DESC))[1] AS session_close,count(*) AS source_rows,count(DISTINCT ts) AS distinct_timestamps,min(ts) AS first_ts,max(ts) AS last_ts FROM bars_polygon WHERE symbol=:symbol AND interval='1m' AND ts>=:start AND ts<:end AND extract(isodow from ts) BETWEEN 1 AND 5 AND extract(hour from ts)>=13 AND extract(hour from ts)<21 GROUP BY 1 ORDER BY 1"""


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def chash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def strict_prior_midrank(
    values: pd.Series, lookback: int = 252, minimum: int = 126
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    output = pd.Series(np.nan, index=numeric.index, dtype=float)
    history: list[float] = []
    for index, current in numeric.items():
        prior = history[-lookback:]
        if math.isfinite(current) and len(prior) >= minimum:
            array = np.asarray(prior)
            output.at[index] = (
                np.sum(array < current) + 0.5 * np.sum(array == current)
            ) / len(array)
        if math.isfinite(current):
            history.append(current)
    return output


def postgres_engine():
    from sqlalchemy import create_engine

    from preprocessing.live_db_features import load_env_file, postgres_url_from_env

    load_env_file(ENV_FILE)
    return create_engine(
        postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10}
    )


def materialize_sessions() -> dict[str, Any]:
    from sqlalchemy import text

    database = postgres_engine()
    frames = []
    with database.connect() as connection:
        for symbol in SYMBOLS:
            raw = pd.read_sql_query(
                text(QUERY),
                connection,
                params={
                    "symbol": symbol,
                    "start": START.to_pydatetime(),
                    "end": END.to_pydatetime(),
                },
            )
            raw["source_day"] = pd.to_datetime(raw.source_day, utc=True)
            raw["first_ts"] = pd.to_datetime(raw.first_ts, utc=True)
            raw["last_ts"] = pd.to_datetime(raw.last_ts, utc=True)
            price_columns = ("session_open", "session_high", "session_low", "session_close")
            for column in price_columns:
                raw[column] = pd.to_numeric(raw[column], errors="coerce")
            valid = (
                raw.distinct_timestamps.ge(450)
                & raw.first_ts.le(raw.source_day + pd.Timedelta(hours=13, minutes=5))
                & raw.last_ts.ge(raw.source_day + pd.Timedelta(hours=20, minutes=55))
                & np.isfinite(raw[list(price_columns)]).all(axis=1)
                & raw[list(price_columns)].gt(0).all(axis=1)
                & raw.session_high.ge(raw[["session_open", "session_close"]].max(axis=1))
                & raw.session_low.le(raw[["session_open", "session_close"]].min(axis=1))
            )
            session_return = np.log(raw.session_close / raw.session_open).where(valid)
            frames.append(
                pd.DataFrame(
                    {
                        "source_day": raw.source_day,
                        f"{symbol}_valid": valid,
                        f"{symbol}_return": session_return,
                    }
                )
            )
    database.dispose()
    frame = frames[0]
    for piece in frames[1:]:
        frame = frame.merge(piece, on="source_day", how="outer", validate="one_to_one")
    frame = frame.sort_values("source_day").reset_index(drop=True)
    frame["source_valid"] = frame[[f"{symbol}_valid" for symbol in SYMBOLS]].fillna(False).all(axis=1)
    frame["usdcad_return"] = frame.USDCAD_return.where(frame.source_valid)
    frame["usdaud_return"] = frame.USDAUD_return.where(frame.source_valid)
    frame["relative_stress"] = (frame.usdcad_return - frame.usdaud_return).where(frame.source_valid)
    frame["absolute_stress_rank"] = strict_prior_midrank(frame.relative_stress.abs())
    frame["decision_time"] = frame.source_day + pd.Timedelta(hours=21)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(frame, SESSION)
    core = {
        "protocol_version": "ccrsr_12_relative_fx_source_v1",
        "query": QUERY,
        "table": "bars_polygon",
        "symbols": list(SYMBOLS),
        "interval": "1m",
        "session_utc": ["13:00", "21:00"],
        "window": [START.isoformat(), END.isoformat()],
        "outcomes_opened": False,
        "candidate_incidence_opened": False,
        "no_imputation": True,
        "output": {
            "path": str(SESSION),
            "sha256": sha(SESSION),
            "rows": len(frame),
            "valid_rows": int(frame.source_valid.sum()),
        },
    }
    payload = {**core, "manifest_hash": chash(core)}
    SOURCE_MANIFEST.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def features() -> pd.DataFrame:
    if sha(PRICE) != PRICE_SHA or sha(PRICE_MANIFEST) != PRICE_MANIFEST_SHA:
        raise RuntimeError("CCRSR BTC source drift")
    frame = pd.read_csv(SESSION, compression="gzip")
    frame["source_day"] = pd.to_datetime(frame.source_day, utc=True)
    frame["decision_time"] = pd.to_datetime(frame.decision_time, utc=True)
    frame["source_valid"] = frame.source_valid.astype(str).str.lower().eq("true")
    feature_columns = (
        "usdcad_return",
        "usdaud_return",
        "relative_stress",
        "absolute_stress_rank",
    )
    for column in feature_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    price = pd.read_csv(PRICE, compression="gzip")
    price["decision_time"] = pd.to_datetime(price.decision_time, utc=True, format="mixed")
    price["open"] = pd.to_numeric(price.open, errors="coerce")
    price["close"] = pd.to_numeric(price.close, errors="coerce")
    price["valid"] = (
        price.source_valid.astype(str).str.lower().eq("true")
        & np.isfinite(price[["open", "close"]]).all(axis=1)
        & price[["open", "close"]].gt(0).all(axis=1)
    )
    price = price.sort_values("decision_time").reset_index(drop=True)
    price["hour_return"] = np.log(price.close / price.open)
    consecutive = price.decision_time.diff().eq(pd.Timedelta(hours=1))
    price["btc_realized_variation"] = np.sqrt(
        price.hour_return.pow(2).rolling(24, min_periods=24).sum()
    )
    price["btc_valid"] = (
        price.valid.rolling(24, min_periods=24).sum().eq(24)
        & consecutive.rolling(23, min_periods=23).sum().eq(23)
        & np.isfinite(price.btc_realized_variation)
    )
    frame = frame.merge(
        price[["decision_time", "btc_realized_variation", "btc_valid"]],
        on="decision_time",
        how="left",
        validate="one_to_one",
    )
    frame["btc_realized_variation_rank"] = strict_prior_midrank(
        frame.btc_realized_variation.where(frame.btc_valid)
    )
    frame["signal_valid"] = (
        frame.source_valid
        & frame.btc_valid.fillna(False)
        & frame.relative_stress.ne(0)
        & np.isfinite(
            frame[
                [
                    *feature_columns,
                    "btc_realized_variation",
                    "btc_realized_variation_rank",
                ]
            ]
        ).all(axis=1)
    )
    return frame


def conditions(frame: pd.DataFrame, control: str) -> tuple[pd.Series, pd.Series]:
    stress = frame.relative_stress
    stress_rank = frame.absolute_stress_rank
    stress_valid = frame.signal_valid
    if control == "one_session_stale_stress":
        stress = stress.shift(1)
        stress_rank = stress_rank.shift(1)
        stress_valid = (
            frame.btc_valid.fillna(False)
            & np.isfinite(frame[["btc_realized_variation", "btc_realized_variation_rank"]]).all(axis=1)
            & frame.source_valid.shift(1, fill_value=False)
            & np.isfinite(stress)
            & np.isfinite(stress_rank)
            & stress.ne(0)
        )
    stress_tail = pd.Series(True, index=frame.index) if control == "no_stress_tail" else stress_rank.ge(0.70)
    variation_gate = pd.Series(True, index=frame.index) if control == "no_variation_gate" else frame.btc_realized_variation_rank.ge(0.65)
    active = stress_valid & stress_tail & variation_gate
    side = -np.sign(stress)
    if control == "usdcad_only":
        side = -np.sign(frame.usdcad_return)
    elif control == "usdaud_only":
        side = np.sign(frame.usdaud_return)
    elif control == "direction_flip":
        side = -side
    elif control == "same_clock_forced_long":
        side = pd.Series(1.0, index=frame.index)
    return active, side


def clock(frame: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    active, side = conditions(frame, control)
    rows = []
    next_allowed = None
    for index in frame.index[active]:
        decision = pd.Timestamp(frame.at[index, "decision_time"])
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
        next_allowed = exit_time
        rows.append(
            {
                "candidate": "CCRSR-12",
                "control": control,
                "split": split,
                "source_day": frame.at[index, "source_day"],
                "decision_time": decision,
                "feature_available_time": decision,
                "entry_time": entry,
                "exit_time": exit_time,
                "side": int(side.at[index]),
                "usdcad_return": float(frame.at[index, "usdcad_return"]),
                "usdaud_return": float(frame.at[index, "usdaud_return"]),
                "relative_stress": float(frame.at[index, "relative_stress"]),
                "absolute_stress_rank": float(frame.at[index, "absolute_stress_rank"]),
                "btc_realized_variation": float(frame.at[index, "btc_realized_variation"]),
                "btc_realized_variation_rank": float(frame.at[index, "btc_realized_variation_rank"]),
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def stats(candidate: pd.DataFrame, split: str) -> dict[str, Any]:
    subset = candidate[candidate.split.eq(split)]
    if subset.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs = int(subset.side.eq(1).sum())
    shorts = int(subset.side.eq(-1).sum())
    months = subset.entry_time.dt.strftime("%Y-%m").value_counts()
    return {
        "events": len(subset),
        "longs": longs,
        "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(subset),
        "max_month_share": int(months.max()) / len(subset),
    }


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("CCRSR preregistration hash drift")
    source_manifest = materialize_sessions()
    frame = features()
    primary = clock(frame)
    controls = {name: clock(frame, name) for name in CONTROLS}
    CLOCK.parent.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(primary, CLOCK)
    for name, candidate in controls.items():
        _write_gzip_csv(candidate, CONTROL_DIR / f"{name}.csv.gz")
    support = {name: stats(primary, name) for name in SPLITS}
    checks = {}
    for name, values in support.items():
        checks[f"{name}_minimum_events"] = values["events"] >= MINIMUM[name]
        checks[f"{name}_side_balance"] = values["minority_side_share"] >= 0.20
        checks[f"{name}_month_concentration"] = values["max_month_share"] <= 0.45
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    passed = all(checks.values())
    core = {
        "protocol_version": "ccrsr_12_source_support_v1",
        "policy_id": "CCRSR-12",
        "preregistration": {
            "path": str(prereg.DEFAULT_OUTPUT),
            "sha256": sha(prereg.DEFAULT_OUTPUT),
            "manifest_hash": registration["manifest_hash"],
        },
        "source_manifests": {
            "commodity_currency_relative_stress": {
                "path": str(SOURCE_MANIFEST),
                "sha256": sha(SOURCE_MANIFEST),
                "manifest_hash": source_manifest["manifest_hash"],
            },
            "completed_btc": {"path": str(PRICE_MANIFEST), "sha256": sha(PRICE_MANIFEST)},
        },
        "completed_preentry_sources_opened": True,
        "postentry_return_pnl_execution_price_opened": False,
        "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "controls": {
            name: {
                "path": str(CONTROL_DIR / f"{name}.csv.gz"),
                "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"),
                "rows": len(candidate),
                "promotion_authorized": False,
            }
            for name, candidate in controls.items()
        },
        "support": support,
        "support_checks": checks,
        "support_passed": passed,
        "advance_to_gross9_novelty": passed,
        "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    payload = {**core, "manifest_hash": chash(core)}
    RESULT.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    result = run()
    print(json.dumps({"passed": result["support_passed"], "support": result["support"]}, indent=2))
